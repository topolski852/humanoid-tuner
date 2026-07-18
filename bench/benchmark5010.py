#!/usr/bin/env python3
"""Benchmark the sim-tuned ROBUST gains (kp=60, kd=2.03) on the real bare 5010.

Same anomaly monitoring as benchmark_gains.py (firmware fault / hunting / overshoot /
current / settle) but with the bare-200KV safety envelope: connect with the correct
flux offset, hard firmware velocity cap, GENTLE steps only, per-sample over-speed
abort. kp=60 on this light motor over-speeds under big steps — kept small on purpose.
Gains RAM-only, restored on exit; never flashed.

    /usr/bin/python3 bench/benchmark5010.py
"""
from __future__ import annotations
import asyncio, json, os, socket, sys, time, uuid
sys.path.insert(0, "/home/nse/humanoid/humanoid-studio/backend")
from humanoid.daemon_client import DaemonClient
from humanoid.robot_config import RobotConfig

CFG = "/home/nse/humanoid/humanoid-studio/configs/bench_shoulder_roll.json"
JOINT, CHAN, DEV = "right_shoulder_roll_joint", "can_right_arm", 4
KT, GEAR, CURRENT_LIMIT = 0.06588, 15.0, 6.0
POS_LO, POS_HI, MARGIN = 0.0, 1.5708, 0.05
CENTER = 0.785
TEST_GAINS = (60.0, 2.03, 0.0, 2.0)          # kp, kd, ki, torque_limit — ROBUST sim result
P_ERR, P_FLUX, P_VLIM = 0x014, 0x13C, 0x034
OFFSET, VLIM, VEL_ABORT = 74.4469985961914, 3.0, 5.0
OUTDIR = os.path.join(os.path.dirname(__file__), "runs")
ERROR_BITS = {0x0002: "ESTOP", 0x0010: "POWERSTAGE", 0x0040: "WATCHDOG", 0x0080: "OVER_VOLTAGE",
              0x0100: "OVER_CURRENT", 0x0200: "OVER_TEMP", 0x2000: "ENCODER"}


def _daemon_cmd(d):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.settimeout(2.0)
    d = dict(d); d["id"] = str(uuid.uuid4())
    s.sendto(json.dumps(d).encode(), ("127.0.0.1", 9001))
    try: return json.loads(s.recvfrom(4096)[0])
    except Exception: return {}
    finally: s.close()


def rd(c, pid, kind="f32"):
    r = c.generic_sdo_read(CHAN, DEV, pid) or {}
    return r.get(f"value_{kind}")


def err_str(e):
    return "none" if not e else ("|".join(v for k, v in ERROR_BITS.items() if e & k) or f"0x{e:04X}")


def _clamp(x):
    return max(POS_LO + MARGIN, min(POS_HI - MARGIN, x))


class Monitor:
    def __init__(self, start, target):
        self.start, self.target, self.delta = start, target, target - start
        self.samples = []; self.max_err = 0; self.peak_vel = 0.0; self.peak_cur = 0.0
        self.sign_changes = 0; self._last = 0; self._risen = False

    def add(self, t, pos, vel, err, cur):
        self.samples.append((t, pos, vel, err, cur))
        if err: self.max_err |= err
        if vel is not None:
            self.peak_vel = max(self.peak_vel, abs(vel))
            if abs(vel) > 0.05: self._risen = True
            if self._risen:
                s = 1 if vel > 0.02 else (-1 if vel < -0.02 else 0)
                if s and self._last and s != self._last: self.sign_changes += 1
                if s: self._last = s
        if cur is not None: self.peak_cur = max(self.peak_cur, abs(cur))

    def report(self):
        pos = [p for _, p, _, _, _ in self.samples if p is not None]
        if not pos: return {"ok": False, "flags": ["no telemetry"]}
        final, peak = pos[-1], (max(pos) if self.delta > 0 else min(pos))
        overshoot = max(0.0, (peak - self.target) / self.delta) if self.delta else 0.0
        ss_err = self.target - final
        # settle: first time within 2% of delta and stays
        band = 0.02 * abs(self.delta) if self.delta else 0.01
        settle = None
        for t, p, _, _, _ in self.samples:
            if p is not None and abs(self.target - p) <= band:
                settle = t; break
        flags = []
        if self.max_err: flags.append(f"FIRMWARE {err_str(self.max_err)}")
        if overshoot > 0.15: flags.append(f"overshoot {overshoot*100:.0f}%")
        if self.sign_changes > 6: flags.append(f"hunting ({self.sign_changes} reversals)")
        if abs(ss_err) > 0.05: flags.append(f"ss_err {ss_err:+.3f}")
        if self.peak_cur > 0.8 * CURRENT_LIMIT: flags.append(f"high current {self.peak_cur:.1f}A")
        return {"ok": not flags, "flags": flags, "overshoot": overshoot, "ss_err": ss_err,
                "settle": settle, "peak_vel": self.peak_vel, "peak_cur": self.peak_cur,
                "sign_changes": self.sign_changes, "final": final}


class Bench:
    def __init__(self, c):
        self.c = c; self.proxy = c.get_actuator_by_name(JOINT); self.aborted = False

    async def set_gains(self, g):
        await self.proxy.write_gains(g[0], g[2], g[1], g[3])

    async def goto(self, tgt, settle=0.8):
        self.c.set_position(JOINT, _clamp(tgt)); await asyncio.sleep(settle)

    async def step(self, start, target, dur=1.4, hz=200):
        await self.goto(start, 0.8)
        target = _clamp(target); mon = Monitor(start, target)
        t0 = time.perf_counter(); self.c.set_position(JOINT, target)
        while time.perf_counter() - t0 < dur:
            st = self.c.get_cached_joint_state(JOINT) or {}
            v = st.get("velocity"); e = st.get("error", 0) or 0
            mon.add(time.perf_counter() - t0, st.get("position"), v, e, st.get("current"))
            if e:
                return mon, True
            if v is not None and abs(v) > VEL_ABORT:
                await self.proxy.disable(); self.aborted = True
                print(f"    !! over-speed abort {v:.1f} rad/s"); return mon, True
            await asyncio.sleep(1.0 / hz)
        return mon, False


async def main():
    os.makedirs(OUTDIR, exist_ok=True)
    c = DaemonClient(RobotConfig.from_json(CFG)); await c.start(); await asyncio.sleep(0.5)
    loop = asyncio.get_running_loop()

    print("connect (correct flux offset) + firmware velocity cap ...")
    await loop.run_in_executor(None, c.connect_single, JOINT); await asyncio.sleep(0.4)
    off = rd(c, P_FLUX)
    if off is None or abs(off - OFFSET) > 0.5:
        print(f"  bad offset {off} — ABORT"); await c.stop(); return
    await loop.run_in_executor(None, c.generic_sdo_write, CHAN, DEV, P_VLIM, "f32", VLIM, 1.0)
    await asyncio.sleep(0.1)
    _daemon_cmd({"type": "DISABLE_ALL_SLOW_POLL"}); await asyncio.sleep(0.3)
    print(f"  flux_offset={off:.3f}  velocity_limit={rd(c,P_VLIM)} rad/s (cap ~{VLIM*GEAR:.0f} motor)")

    b = Bench(c)
    e0 = rd(c, P_ERR, "u32") or 0
    st = c.get_cached_joint_state(JOINT) or {}
    print(f"pre-flight: pos={st.get('position'):.3f} bus={st.get('bus_voltage'):.1f} err={err_str(e0)}")
    orig = {"kp": 50.0, "ki": 0.0, "kd": 2.0, "tau": 2.0}

    records, aborted = [], False
    try:
        print(f"\nWRITE_GAINS robust: kp={TEST_GAINS[0]} kd={TEST_GAINS[1]} ki={TEST_GAINS[2]} "
              f"tau={TEST_GAINS[3]} (RAM only)")
        await b.set_gains(TEST_GAINS)
        c.set_position(JOINT, CENTER); await b.proxy.enable(); await asyncio.sleep(0.3)

        print("\n[PROBE] gentle 0.06 rad step ...")
        mon, ab = await b.step(CENTER, CENTER + 0.06, dur=1.2)
        rep = mon.report(); records.append(("probe", rep, mon.samples))
        se = f"{rep['settle']:.2f}s" if rep.get("settle") else "—"
        print(f"  overshoot={rep['overshoot']*100:.0f}%  ss_err={rep['ss_err']:+.4f}  settle={se}  "
              f"vpk={rep['peak_vel']:.2f}  ipk={rep['peak_cur']:.2f}A  rev={rep['sign_changes']}  "
              f"-> {'OK' if rep['ok'] else 'FLAGS: '+', '.join(rep['flags'])}")
        if ab or not rep["ok"]:
            aborted = True; print("  probe not clean — stopping.")

        if not aborted:
            print("\n[BATTERY] 0.05 / 0.10 rad steps, both directions ...")
            plan = [("+0.05", CENTER+0.05), ("-0.05", CENTER-0.05),
                    ("+0.10", CENTER+0.10), ("-0.10", CENTER-0.10)]
            for label, tgt in plan:
                mon, ab = await b.step(CENTER, tgt, dur=1.4)
                rp = mon.report(); records.append((label, rp, mon.samples))
                se = f"{rp['settle']:.2f}s" if rp.get("settle") else "—"
                tag = "OK" if rp["ok"] else "⚠ " + ", ".join(rp["flags"])
                print(f"  {label}: over={rp['overshoot']*100:4.0f}%  ss={rp['ss_err']:+.4f}  "
                      f"settle={se}  vpk={rp['peak_vel']:.2f}  ipk={rp['peak_cur']:.2f}A  "
                      f"rev={rp['sign_changes']}  -> {tag}")
                if ab:
                    aborted = True; break
    finally:
        try:
            await b.goto(CENTER, 0.5)
        except Exception:
            pass
        await b.set_gains((orig["kp"], orig["kd"], orig["ki"], orig["tau"]))
        print(f"\nrestored gains kp={orig['kp']} kd={orig['kd']} tau={orig['tau']}")
        await b.proxy.disable()
        print(f"final error: {err_str(rd(c, P_ERR, 'u32') or 0)}")
        _daemon_cmd({"type": "ENABLE_ALL_SLOW_POLL"}); await c.stop()

    out = os.path.join(OUTDIR, f"benchmark5010_{int(time.time())}.json")
    json.dump({"gains": TEST_GAINS, "center": CENTER, "aborted": aborted,
               "records": [{"label": l, "report": {k: v for k, v in r.items()},
                            "samples": s} for l, r, s in records]}, open(out, "w"))
    n_ok = sum(1 for _, r, _ in records if r.get("ok"))
    print(f"\nVERDICT: {n_ok}/{len(records)} steps clean" + (" (ABORTED)" if aborted else ""))
    print(f"wrote {out}")


if __name__ == "__main__":
    asyncio.run(main())

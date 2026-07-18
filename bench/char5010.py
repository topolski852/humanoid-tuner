#!/usr/bin/env python3
"""GENTLE, speed-capped characterization for the bare 5010 200KV (shoulder_roll).

Same proven method as the bare-M6 `characterize.py full` (position steps + friction
ramps, fit with fit_plant.py) but tuned for a light, fast 200KV motor that will
oscillate/over-speed under the M6's aggressive battery:
  * firmware velocity_limit capped LOW (hard speed limit -> no over-speed/aliasing)
  * small steps + low gains only
  * per-sample abort on over-speed or firmware fault
  * incremental save (a stop never loses data)
Non-destructive: WRITE_GAINS + velocity_limit are RAM only; flux offset is applied
correct by the (restarted) daemon config; a power cycle restores flash.

Run (fresh daemon w/ bench_shoulder_roll.json, motor powered):
    /usr/bin/python3 bench/char5010.py
"""
from __future__ import annotations
import asyncio, json, os, sys, time
sys.path.insert(0, "/home/nse/humanoid/humanoid-studio/backend")
from humanoid.daemon_client import DaemonClient
from humanoid.robot_config import RobotConfig

CFG = "/home/nse/humanoid/humanoid-studio/configs/bench_shoulder_roll.json"
JOINT = "right_shoulder_roll_joint"
CHAN, DEV = "can_right_arm", 4
KT, GEAR = 0.06588, 15.0
P_ERR, P_IQ, P_VLIM, P_FLUX = 0x014, 0x0C0, 0x034, 0x13C
EXPECT_OFFSET = 74.4469985961914
VLIM = 3.0                 # rad/s output firmware cap (~45 rad/s motor) — hard safety
VEL_ABORT = 5.0            # rad/s output — software abort (~75 rad/s motor, 4x below vibration)
CENTER = 0.785             # mid of [0,1.57]
OUTDIR = os.path.join(os.path.dirname(__file__), "runs")


def rd(c, pid, kind="f32"):
    r = c.generic_sdo_read(CHAN, DEV, pid) or {}
    return r.get(f"value_{kind}")


class Bench:
    def __init__(self, c):
        self.c = c
        self.aborted = False

    def st(self):
        return self.c.get_cached_joint_state(JOINT) or {}

    async def set_gains(self, kp, kd, ki, tau):
        await self.c.get_actuator_by_name(JOINT).write_gains(kp, ki, kd, tau)

    async def goto(self, target, settle=0.7):
        self.c.set_position(JOINT, target); await asyncio.sleep(settle)

    async def record_step(self, start, target, gains, dur=1.2, hz=200):
        kp, kd, ki, tau = gains
        await self.set_gains(kp, kd, ki, tau)
        await self.goto(start, 0.8)
        t0 = time.perf_counter(); ts, pos, vel = [], [], []
        self.c.set_position(JOINT, target)
        while time.perf_counter() - t0 < dur:
            s = self.st(); v = s.get("velocity")
            ts.append(time.perf_counter() - t0)
            pos.append(s.get("position")); vel.append(v)
            if v is not None and abs(v) > VEL_ABORT:
                await self.c.get_actuator_by_name(JOINT).disable(); self.aborted = True
                print(f"    !! ABORT over-speed {v:.1f} rad/s"); break
            await asyncio.sleep(1.0 / hz)
        return {"kind": "step", "gains": list(gains), "start": start, "target": target,
                "t": ts, "pos": pos, "vel": vel}

    async def record_ramp(self, start, end, speed, gains, hz=150):
        kp, kd, ki, tau = gains
        await self.set_gains(kp, kd, ki, tau)
        await self.goto(start, 0.8)
        direction = 1.0 if end >= start else -1.0
        t0 = time.perf_counter(); ts, pos, vel, cmd, iq = [], [], [], [], []
        cur, last_iq = start, -1.0
        while (direction > 0 and cur < end) or (direction < 0 and cur > end):
            el = time.perf_counter() - t0
            cur = start + direction * speed * el
            self.c.set_position(JOINT, cur)
            s = self.st(); v = s.get("velocity")
            ts.append(el); cmd.append(cur); pos.append(s.get("position")); vel.append(v)
            if el - last_iq > 0.05:
                iq.append((el, rd(self.c, P_IQ))); last_iq = el
            if v is not None and abs(v) > VEL_ABORT:
                await self.c.get_actuator_by_name(JOINT).disable(); self.aborted = True
                print(f"    !! ABORT over-speed {v:.1f} rad/s"); break
            await asyncio.sleep(1.0 / hz)
        return {"kind": "ramp", "gains": list(gains), "start": start, "end": end,
                "speed": speed, "t": ts, "pos": pos, "vel": vel, "cmd": cmd, "iq": iq}


def save(results, orig):
    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, "char5010_latest.json")
    json.dump({"joint": JOINT, "device_id": DEV, "Kt": KT, "gear": GEAR,
               "pos_limits": [0.0, 1.5708], "center": CENTER, "orig_gains": orig,
               "results": results}, open(out, "w"))
    return out


async def main():
    c = DaemonClient(RobotConfig.from_json(CFG)); await c.start(); await asyncio.sleep(0.5)
    loop = asyncio.get_running_loop()
    b = c.get_actuator_by_name(JOINT)

    # wake + connect (daemon config now applies correct flux offset)
    print("connect (IDLE + apply correct config) ...")
    await loop.run_in_executor(None, c.connect_single, JOINT); await asyncio.sleep(0.4)
    off = rd(c, P_FLUX)
    print(f"  flux_offset={off} (want {EXPECT_OFFSET:.3f})")
    if off is None or abs(off - EXPECT_OFFSET) > 0.5:
        print("  !! bad offset — ABORT"); await c.stop(); return
    # HARD speed cap
    await loop.run_in_executor(None, c.generic_sdo_write, CHAN, DEV, P_VLIM, "f32", VLIM, 1.0)
    await asyncio.sleep(0.1)
    print(f"  velocity_limit set = {rd(c, P_VLIM)} rad/s (hard cap ~{VLIM*GEAR:.0f} rad/s motor)")
    await b.clear_error(); await asyncio.sleep(0.2)

    orig = {"kp": 50.0, "ki": 0.0, "kd": 2.0, "tau": 2.0}
    results = []
    bb = Bench(c)
    try:
        c.set_position(JOINT, CENTER); await b.enable(); await asyncio.sleep(0.3)

        # FRICTION RAMPS FIRST — low constant velocity, inherently safe, and the piece
        # we most need (Coulomb). Guaranteed saved before any step risk.
        print("friction ramps (low speed) ...")
        for sp in [0.2, 0.35, 0.5]:
            for sgn in (+1, -1):
                r = await bb.record_ramp(CENTER - sgn*0.25, CENTER + sgn*0.25, sp, (25.0, 1.0, 0.0, 2.0))
                results.append(r); save(results, orig)
                iqs = [v for _, v in r["iq"] if v is not None]
                tau_ss = (sum(sorted([abs(x) for x in iqs])[len(iqs)//3:]) /
                          max(1, len(iqs) - len(iqs)//3)) * KT if iqs else 0
                print(f"  ramp {sgn*sp:+.2f} rad/s: {len(iqs)} iq, friction≈{tau_ss:.4f} N·m motor"
                      + ("  ABORTED" if bb.aborted else ""))
                if bb.aborted:
                    break
            if bb.aborted:
                break

        # GENTLE steps for inertia (small only; abort protects).
        bb.aborted = False
        GAINSETS = [(20.0, 1.0, 0.0, 2.0), (25.0, 1.5, 0.0, 2.0)]
        STEPS = [0.10, -0.10, 0.15, -0.15]
        for g in GAINSETS:
            for s in STEPS:
                r = await bb.record_step(CENTER, CENTER + s, g)
                results.append(r); save(results, orig)
                pk = max((abs(v) for v in r["vel"] if v is not None), default=0)
                fw = rd(c, P_ERR, "u32") or 0
                print(f"  step {s:+.2f} gains{g[:3]}: vpk={pk:.2f} rad/s  fw_err=0x{fw:04X}"
                      + ("  ABORTED" if bb.aborted else ""))
                if bb.aborted or fw:
                    bb.aborted = False   # a step abort shouldn't discard the rest; re-enable next
                    await b.enable(); await asyncio.sleep(0.2)
    finally:
        try:
            c.set_position(JOINT, CENTER); await asyncio.sleep(0.4)
        except Exception:
            pass
        try:
            await b.disable()
        except Exception:
            pass
        await c.stop()
    out = save(results, orig)
    print(f"\nwrote {out}  ({len(results)} records)")


if __name__ == "__main__":
    asyncio.run(main())

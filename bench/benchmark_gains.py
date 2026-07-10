#!/usr/bin/env python3
"""Benchmark sim-tuned gains on the real motor, with full anomaly monitoring.

Applies the sim-optimal gains (WRITE_GAINS, RAM only — never flashed), commands a
gentle probe then a step battery, and monitors EVERY sample for:
  * firmware error bitmask (overcurrent/overtemp/watchdog/...) -> instant abort to IDLE
  * hunting / stick-slip (velocity sign reversals after the rise)
  * overshoot beyond expectation
  * peak current vs the current limit
  * failure to settle / runaway
Original gains are restored on exit. Nothing is written to flash.

Run with the daemon up and the motor powered/free:
    python3 bench/benchmark_gains.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

sys.path.insert(0, "/home/nse/humanoid/humanoid-studio/backend")
from humanoid.daemon_client import DaemonClient   # noqa: E402
from humanoid.robot_config import RobotConfig      # noqa: E402

CFG = "/home/nse/humanoid/humanoid-studio/configs/bench_right_hip_roll.json"
JOINT, CHAN, DEV = "right_hip_roll_joint", "can_right_leg", 4
KT, GEAR, CURRENT_LIMIT = 0.08958, 15.0, 20.0
POS_LO, POS_HI, MARGIN = -0.675, 0.896, 0.05
TEST_GAINS = (60.0, 2.52, 0.0, 2.0)      # kp, kd(velocity_kp), ki, torque_limit — SIM-OPTIMAL
PARAM_ERR, PARAM_MODE = 0x014, 0x010
OUTDIR = os.path.join(os.path.dirname(__file__), "runs")

ERROR_BITS = {0x0002: "ESTOP", 0x0010: "POWERSTAGE", 0x0040: "WATCHDOG", 0x0080: "OVER_VOLTAGE",
              0x0100: "OVER_CURRENT", 0x0200: "OVER_TEMP", 0x2000: "ENCODER"}


def _clamp(x):
    return max(POS_LO + MARGIN, min(POS_HI - MARGIN, x))


def err_str(e):
    if not e:
        return "none"
    return "|".join(v for k, v in ERROR_BITS.items() if e & k) or f"0x{e:04X}"


class Monitor:
    """Live per-step anomaly detector over (pos, vel, error, current) samples."""
    def __init__(self, start, target):
        self.start, self.target = start, target
        self.delta = target - start
        self.samples = []
        self.max_err = 0
        self.peak_vel = 0.0
        self.peak_cur = 0.0
        self.sign_changes = 0
        self._last_vsign = 0
        self._risen = False

    def add(self, t, pos, vel, error, cur):
        self.samples.append((t, pos, vel, error, cur))
        if error:
            self.max_err |= error
        if vel is not None:
            self.peak_vel = max(self.peak_vel, abs(vel))
            # count velocity sign reversals AFTER the initial rise (hunting signal)
            if abs(vel) > 0.05:
                self._risen = True
            if self._risen:
                s = 1 if vel > 0.02 else (-1 if vel < -0.02 else 0)
                if s and self._last_vsign and s != self._last_vsign:
                    self.sign_changes += 1
                if s:
                    self._last_vsign = s
        if cur is not None:
            self.peak_cur = max(self.peak_cur, abs(cur))

    def report(self):
        pos = [p for _, p, _, _, _ in self.samples if p is not None]
        if not pos:
            return {"ok": False, "flags": ["no telemetry"]}
        final = pos[-1]
        peak = max(pos) if self.delta > 0 else min(pos)
        overshoot = max(0.0, (peak - self.target) / self.delta) if self.delta else 0.0
        ss_err = self.target - final
        flags = []
        if self.max_err:
            flags.append(f"FIRMWARE ERROR {err_str(self.max_err)}")
        if overshoot > 0.15:
            flags.append(f"overshoot {overshoot*100:.0f}%")
        if self.sign_changes > 6:
            flags.append(f"hunting ({self.sign_changes} vel reversals)")
        if abs(ss_err) > 0.05:
            flags.append(f"large ss_err {ss_err:+.3f}")
        if self.peak_cur > 0.8 * CURRENT_LIMIT:
            flags.append(f"high current {self.peak_cur:.1f}A")
        return {"ok": not flags, "flags": flags, "overshoot": overshoot,
                "ss_err": ss_err, "peak_vel": self.peak_vel, "peak_cur": self.peak_cur,
                "sign_changes": self.sign_changes, "final": final, "peak": peak}


class Bench:
    def __init__(self, c):
        self.c = c
        self.proxy = c.get_actuator_by_name(JOINT)

    def read_err(self):
        r = self.c.generic_sdo_read(CHAN, DEV, PARAM_ERR)
        return (r or {}).get("value_u32") or 0

    async def set_gains(self, g):
        await self.proxy.write_gains(g[0], g[2], g[1], g[3])   # (kp, ki, kd, tau)

    async def goto(self, target, settle=0.7):
        self.c.set_position(JOINT, _clamp(target)); await asyncio.sleep(settle)

    async def step(self, start, target, dur=1.2, poll_hz=250):
        await self.goto(start, 0.8)
        target = _clamp(target)
        mon = Monitor(start, target)
        t0 = time.perf_counter()
        self.c.set_position(JOINT, target)
        dt = 1.0 / poll_hz
        while time.perf_counter() - t0 < dur:
            st = self.c.get_cached_joint_state(JOINT) or {}
            err = st.get("error", 0) or 0
            mon.add(time.perf_counter() - t0, st.get("position"), st.get("velocity"),
                    err, st.get("current"))
            if err:                                   # instant abort on any firmware error
                return mon, True
            await asyncio.sleep(dt)
        return mon, False


async def main():
    os.makedirs(OUTDIR, exist_ok=True)
    c = DaemonClient(RobotConfig.from_json(CFG)); await c.start(); await asyncio.sleep(0.5)
    b = Bench(c)

    # pre-flight
    e0 = b.read_err()
    st = c.get_cached_joint_state(JOINT) or {}
    base = round(st.get("position", 0.0), 4)
    print(f"pre-flight: pos={base} bus={st.get('bus_voltage')} error={err_str(e0)}")
    if e0:
        print("ABORT: firmware error present before start."); await c.stop(); return
    orig = {n: (c.generic_sdo_read(CHAN, DEV, p) or {}).get("value_f32")
            for n, p in [("kp", 0x020), ("ki", 0x024), ("kd", 0x028), ("tau", 0x030)]}
    print(f"original gains: {orig}")
    center = round((POS_LO + POS_HI) / 2, 3)

    records, aborted = [], False
    try:
        print(f"\nWRITE_GAINS sim-optimal: kp={TEST_GAINS[0]} kd={TEST_GAINS[1]} "
              f"ki={TEST_GAINS[2]} tau_lim={TEST_GAINS[3]} (RAM only)")
        await b.set_gains(TEST_GAINS)
        c.set_position(JOINT, base)
        await b.proxy.enable(); c.set_position(JOINT, base); await asyncio.sleep(0.3)

        # 1) gentle probe
        print("\n[PROBE] gentle 0.12 rad step, full monitoring ...")
        mon, ab = await b.step(center, center + 0.12, dur=1.0)
        rep = mon.report(); records.append(("probe", rep, mon.samples))
        print(f"  overshoot={rep['overshoot']*100:.0f}%  ss_err={rep['ss_err']:+.4f}  "
              f"peak_vel={rep['peak_vel']:.2f} rad/s  peak_cur={rep['peak_cur']:.2f}A  "
              f"vel_reversals={rep['sign_changes']}  -> {'OK' if rep['ok'] else 'FLAGS: '+', '.join(rep['flags'])}")
        if ab or not rep["ok"]:
            print("  probe not clean — stopping before larger steps."); aborted = True

        # 2) step battery (repeatable, both directions)
        if not aborted:
            print("\n[BATTERY] 0.5 & 0.3 rad steps, both directions, 2 reps ...")
            plan = [("+0.5", center + 0.5), ("-0.5", center - 0.5),
                    ("+0.3", center + 0.3), ("-0.3", center - 0.3)]
            for rep_i in range(2):
                for label, tgt in plan:
                    mon, ab = await b.step(center, tgt, dur=1.2)
                    rp = mon.report(); records.append((f"{label}#{rep_i}", rp, mon.samples))
                    tag = "OK" if rp["ok"] else "⚠ " + ", ".join(rp["flags"])
                    print(f"  {label}#{rep_i}: over={rp['overshoot']*100:4.0f}%  "
                          f"ss_err={rp['ss_err']:+.4f}  vpk={rp['peak_vel']:.2f}  "
                          f"ipk={rp['peak_cur']:.2f}A  rev={rp['sign_changes']}  -> {tag}")
                    if ab:
                        print("  ABORT: firmware error mid-step."); aborted = True; break
                if aborted:
                    break
    finally:
        await b.goto(center, 0.5)
        if all(v is not None for v in orig.values()):
            await b.set_gains((orig["kp"], orig["kd"], orig["ki"], orig["tau"]))
            print(f"\nrestored original gains: kp={orig['kp']} kd={orig['kd']} ki={orig['ki']} tau={orig['tau']}")
        await b.proxy.disable()
        print(f"final error state: {err_str(b.read_err())}")
        await c.stop()

    out = os.path.join(OUTDIR, f"benchmark_{int(time.time())}.json")
    json.dump({"gains": TEST_GAINS, "center": center, "aborted": aborted,
               "records": [{"label": l, "report": {k: v for k, v in r.items() if k != "samples"},
                            "samples": s} for l, r, s in records]}, open(out, "w"))
    print(f"\nwrote {out}")
    n_ok = sum(1 for _, r, _ in records if r.get("ok"))
    print(f"VERDICT: {n_ok}/{len(records)} steps clean" + (" (ABORTED)" if aborted else ""))


if __name__ == "__main__":
    asyncio.run(main())

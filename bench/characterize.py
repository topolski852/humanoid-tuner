#!/usr/bin/env python3
"""Position-mode system identification for the bench ESC — real2sim data.

The daemon exposes only POSITION mode, so we identify the plant the way the sim
uses it: drive position steps/ramps with known gains (the firmware control law
sim/control_law.py already replicates), log the 100 Hz response, and later fit
the sim's plant params (inertia, viscous + Coulomb friction) to match.

SAFETY / non-destructive (this ESC carries robot calibration):
  * NEVER calls apply_config or store_to_flash — only WRITE_GAINS (RAM: kp, ki,
    velocity_kp, torque_limit). Original gains are read first and restored at the
    end. A power cycle also restores everything from flash.
  * All motion stays inside the device's real position limits.
  * Leaves the joint IDLE on exit; ESTOP available on 9002.

Usage (daemon running with the bench config, motor free to move):
    python3 bench/characterize.py probe     # one gentle step, verify + log
    python3 bench/characterize.py full       # step battery + friction ramps
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, "/home/nse/humanoid/humanoid-studio/backend")
from humanoid.daemon_client import DaemonClient   # noqa: E402
from humanoid.robot_config import RobotConfig      # noqa: E402

CFG = "/home/nse/humanoid/humanoid-studio/configs/bench_right_hip_roll.json"
JOINT = "right_hip_roll_joint"
CHAN, DEV = "can_right_leg", 4
KT, GEAR = 0.08958, 15.0
PARAM_IQ = 0x0C0
POS_LO, POS_HI = -0.675, 0.896          # device flash limits (read from ESC)
MARGIN = 0.03
OUTDIR = os.path.join(os.path.dirname(__file__), "runs")


def _clamp(x):
    return max(POS_LO + MARGIN, min(POS_HI - MARGIN, x))


class Bench:
    def __init__(self, client: DaemonClient):
        self.c = client

    def state(self):
        return self.c.get_cached_joint_state(JOINT) or {}

    def iq(self):
        r = self.c.generic_sdo_read(CHAN, DEV, PARAM_IQ)
        return r["value_f32"] if r else None

    async def set_gains(self, kp, kd, ki, tau_lim):
        # WRITE_GAINS order: (kp, ki, velocity_kp/Kd, torque_limit) — RAM only.
        await self.c.get_actuator_by_name(JOINT).write_gains(kp, ki, kd, tau_lim)

    async def goto(self, target, settle=0.6):
        target = _clamp(target)
        self.c.set_position(JOINT, target)
        await asyncio.sleep(settle)

    async def record_step(self, start, target, gains, dur=0.9, poll_hz=250):
        """Set gains, park at `start`, command a step to `target`, log the response."""
        kp, kd, ki, tau = gains
        await self.set_gains(kp, kd, ki, tau)
        await self.goto(start, settle=0.7)
        target = _clamp(target)
        t0 = time.perf_counter()
        ts, pos, vel = [], [], []
        self.c.set_position(JOINT, target)
        dt = 1.0 / poll_hz
        while time.perf_counter() - t0 < dur:
            st = self.state()
            ts.append(time.perf_counter() - t0)
            pos.append(st.get("position")); vel.append(st.get("velocity"))
            await asyncio.sleep(dt)
        return {"kind": "step", "gains": gains, "start": start, "target": target,
                "t": ts, "pos": pos, "vel": vel}

    async def record_ramp(self, start, end, speed, gains, poll_hz=200):
        """Constant-velocity ramp; log pos/vel and i_q (steady-state i_q -> friction)."""
        kp, kd, ki, tau = gains
        await self.set_gains(kp, kd, ki, tau)
        await self.goto(start, settle=0.7)
        start, end = _clamp(start), _clamp(end)
        direction = 1.0 if end >= start else -1.0
        t0 = time.perf_counter()
        ts, pos, vel, iq, tcmd = [], [], [], [], []
        dt = 1.0 / poll_hz
        last_iq_t = 0.0
        cur = start
        while (direction > 0 and cur < end) or (direction < 0 and cur > end):
            el = time.perf_counter() - t0
            cur = start + direction * speed * el
            cur = _clamp(cur)
            self.c.set_position(JOINT, cur)
            st = self.state()
            ts.append(el); tcmd.append(cur)
            pos.append(st.get("position")); vel.append(st.get("velocity"))
            # poll i_q ~20 Hz (SDO round-trip is slow)
            if el - last_iq_t > 0.05:
                iq.append((el, self.iq())); last_iq_t = el
            await asyncio.sleep(dt)
        return {"kind": "ramp", "gains": gains, "start": start, "end": end,
                "speed": speed, "t": ts, "pos": pos, "vel": vel, "cmd": tcmd, "iq": iq}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["probe", "full"])
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    c = DaemonClient(RobotConfig.from_json(CFG))
    await c.start()
    await asyncio.sleep(0.5)
    b = Bench(c)

    st = b.state()
    base = round(st.get("position", 0.0), 4)
    print(f"start: joint_state={st.get('joint_state')} pos={base} bus={st.get('bus_voltage')}")
    center = round((POS_LO + POS_HI) / 2, 3)   # ~0.11 rad, centers the excitation

    # snapshot original gains to restore (read from device)
    orig = {p: c.generic_sdo_read(CHAN, DEV, pid) for p, pid in
            [("kp", 0x020), ("ki", 0x024), ("kd", 0x028), ("tau", 0x030)]}
    ok = {k: (v or {}).get("value_f32") for k, v in orig.items()}
    print(f"original gains (to restore): kp={ok['kp']} ki={ok['ki']} kd={ok['kd']} tau_lim={ok['tau']}")

    results = []
    try:
        c.set_position(JOINT, base)                    # lock target to current angle FIRST
        await c.get_actuator_by_name(JOINT).enable()   # SET_MODE POSITION (no jump on enable)
        c.set_position(JOINT, base)
        await asyncio.sleep(0.3)

        if args.mode == "probe":
            print("PROBE: one gentle 0.12 rad step (kp=25, kd=1.5, tau_lim=2.0) ...")
            r = await b.record_step(center, center + 0.12, (25.0, 1.5, 0.0, 2.0))
            results.append(r)
            pk = max([p for p in r["pos"] if p is not None], default=None)
            print(f"  reached max pos {pk:.4f} (target {r['target']:.4f}), "
                  f"{sum(v is not None for v in r['vel'])} samples logged")
        else:
            GAINSETS = [(20.0, 1.0, 0.0, 2.0), (40.0, 2.0, 0.0, 3.0), (25.0, 0.5, 0.0, 2.0)]
            STEPS = [0.15, -0.15, 0.3, -0.3, 0.5, -0.5]
            for g in GAINSETS:
                for s in STEPS:
                    r = await b.record_step(center, center + s, g)
                    results.append(r)
                    print(f"  step {s:+.2f} gains{g[:3]} -> {sum(v is not None for v in r['vel'])} samples")
            print("friction ramps ...")
            for sp in [0.5, 1.0, 2.0]:
                for sgn in (+1, -1):
                    r = await b.record_ramp(center - sgn * 0.4, center + sgn * 0.4, sp, (30.0, 0.3, 0.0, 3.0))
                    iqs = [v for _, v in r["iq"] if v is not None]
                    print(f"  ramp {sgn*sp:+.1f} rad/s: {len(iqs)} i_q samples, "
                          f"mean={sum(iqs)/len(iqs):.4f} A" if iqs else f"  ramp {sgn*sp:+.1f}: no iq")
                    results.append(r)
    finally:
        # park, restore original gains (RAM), IDLE — never flash
        await b.goto(center, settle=0.5)
        if all(ok[k] is not None for k in ok):
            await b.set_gains(ok["kp"], ok["kd"], ok["ki"], ok["tau"])
            print(f"restored original gains kp={ok['kp']} kd={ok['kd']} ki={ok['ki']} tau={ok['tau']}")
        await c.get_actuator_by_name(JOINT).disable()   # IDLE
        await c.stop()

    stamp = int(time.time())
    out = os.path.join(OUTDIR, f"bench_{args.mode}_{stamp}.json")
    json.dump({"joint": JOINT, "device_id": DEV, "Kt": KT, "gear": GEAR,
               "pos_limits": [POS_LO, POS_HI], "center": center,
               "orig_gains": ok, "results": results}, open(out, "w"))
    print(f"wrote {out}  ({len(results)} records)")


if __name__ == "__main__":
    asyncio.run(main())

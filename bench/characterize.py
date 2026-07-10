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
import socket
import sys
import time
import uuid

sys.path.insert(0, "/home/nse/humanoid/humanoid-studio/backend")
from humanoid.daemon_client import DaemonClient   # noqa: E402
from humanoid.robot_config import RobotConfig      # noqa: E402


def _daemon_cmd(d):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.settimeout(2.0)
    d = dict(d); d["id"] = str(uuid.uuid4())
    s.sendto(json.dumps(d).encode(), ("127.0.0.1", 9001))
    try:
        return json.loads(s.recvfrom(4096)[0])
    except Exception:
        return {}
    finally:
        s.close()


def _robust_read(c, pid, expect_max):
    """Median of 5 SDO reads (slow-poll must be OFF to avoid the bus-voltage race)."""
    vals = []
    for _ in range(5):
        v = (c.generic_sdo_read("can_right_leg", 4, pid) or {}).get("value_f32")
        if v is not None and abs(v) <= expect_max:
            vals.append(v)
        time.sleep(0.03)
    vals.sort()
    return vals[len(vals) // 2] if vals else None

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

    async def record_sweep(self, center, amp, speed, gains, cycles=3, poll_hz=200):
        """Slow triangle sweep with multiple reversals — exposes backlash/hysteresis.

        On a free motor-side-encoder shaft backlash is nearly invisible; this pays
        off once a LOAD is on the output. Logs fine pos/vel + i_q through each
        reversal so the dead-band (if any) is captured.
        """
        kp, kd, ki, tau = gains
        await self.set_gains(kp, kd, ki, tau)
        lo, hi = _clamp(center - amp), _clamp(center + amp)
        await self.goto(lo, 0.8)
        t0 = time.perf_counter()
        ts, pos, vel, cmd, iq = [], [], [], [], []
        legs = []
        tgt = lo
        for _ in range(cycles):
            legs += [hi, lo]
        last_iq = -1.0
        for goal in legs:
            direction = 1.0 if goal >= tgt else -1.0
            while (direction > 0 and tgt < goal) or (direction < 0 and tgt > goal):
                el = time.perf_counter() - t0
                tgt = _clamp(tgt + direction * speed * (1.0 / poll_hz))
                self.c.set_position(JOINT, tgt)
                st = self.state()
                ts.append(el); cmd.append(tgt)
                pos.append(st.get("position")); vel.append(st.get("velocity"))
                if el - last_iq > 0.04:
                    iq.append((el, self.iq())); last_iq = el
                await asyncio.sleep(1.0 / poll_hz)
        return {"kind": "sweep", "gains": gains, "center": center, "amp": amp,
                "speed": speed, "cycles": cycles, "t": ts, "pos": pos, "vel": vel,
                "cmd": cmd, "iq": iq}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["probe", "full", "geared"])
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    c = DaemonClient(RobotConfig.from_json(CFG))
    await c.start()
    await asyncio.sleep(0.5)
    _daemon_cmd({"type": "DISABLE_ALL_SLOW_POLL"})   # avoid bus-voltage SDO race on gain reads
    await asyncio.sleep(0.3)
    b = Bench(c)

    st = b.state()
    base = round(st.get("position", 0.0), 4)
    print(f"start: joint_state={st.get('joint_state')} pos={base} bus={st.get('bus_voltage')}")
    center = round((POS_LO + POS_HI) / 2, 3)   # ~0.11 rad, centers the excitation

    # snapshot original gains to restore — robust median reads, sane bounds so a
    # stray bus-voltage read can never poison the restore (the bug that wrote ki=19.7).
    ok = {"kp": _robust_read(c, 0x020, 300), "ki": _robust_read(c, 0x024, 50),
          "kd": _robust_read(c, 0x028, 50), "tau": _robust_read(c, 0x030, 30)}
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
        elif args.mode == "full":
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

        if args.mode == "geared":
            # GEARBOX attached: dynamics are slower + friction is much higher, so use
            # longer step windows, more torque headroom, and LOW-speed ramps over the
            # full range. Backlash needs a load to be observable (motor-side encoder).
            def _err():
                r = c.generic_sdo_read(CHAN, DEV, 0x014)
                return (r or {}).get("value_u32") or 0
            if _err():
                print(f"  ABORT: firmware error 0x{_err():04X} before start")
            else:
                TAU = 4.0                                   # headroom over gearbox friction
                GAINSETS = [(40.0, 2.0, 0.0, TAU), (60.0, 3.0, 0.0, TAU)]
                STEPS = [0.2, -0.2, 0.4, -0.4, 0.6, -0.6]
                print(f"[geared] step battery (2.5s windows, tau_lim={TAU}) ...")
                for g in GAINSETS:
                    for s in STEPS:
                        r = await b.record_step(center, center + s, g, dur=2.5)
                        results.append(r)
                        pk = max((abs(v) for v in r["vel"] if v is not None), default=0)
                        print(f"  step {s:+.2f} gains{g[:3]} -> {sum(v is not None for v in r['vel'])} samples, vpk={pk:.2f}"
                              + ("  ⚠ERROR" if _err() else ""))
                        if _err():
                            print("  ABORT: firmware error mid-battery"); break
                    if _err():
                        break
                if not _err():
                    print("[geared] low-speed friction ramps (full range, long) ...")
                    for sp in [0.1, 0.2, 0.4]:
                        for sgn in (+1, -1):
                            r = await b.record_ramp(center - sgn * 0.5, center + sgn * 0.5, sp, (50.0, 1.0, 0.0, TAU))
                            iqs = [v for _, v in r["iq"] if v is not None]
                            tau_ss = (sum(sorted(iqs)[len(iqs)//3:]) / max(1, len(iqs) - len(iqs)//3)) * KT * GEAR if iqs else 0
                            print(f"  ramp {sgn*sp:+.2f} rad/s: {len(iqs)} iq, friction≈{tau_ss:+.3f} N·m")
                            results.append(r)
                if not _err():
                    print("[geared] slow backlash sweep (0.05 rad/s, needs a load to show play) ...")
                    r = await b.record_sweep(center, amp=0.5, speed=0.05, gains=(50.0, 1.0, 0.0, TAU), cycles=2)
                    results.append(r)
                    print(f"  sweep: {sum(v is not None for v in r['vel'])} samples over {r['t'][-1]:.1f}s")
    finally:
        # park, restore original gains (RAM), IDLE — never flash
        await b.goto(center, settle=0.5)
        if all(ok[k] is not None for k in ok):
            await b.set_gains(ok["kp"], ok["kd"], ok["ki"], ok["tau"])
            print(f"restored original gains kp={ok['kp']} kd={ok['kd']} ki={ok['ki']} tau={ok['tau']}")
        await c.get_actuator_by_name(JOINT).disable()   # IDLE
        _daemon_cmd({"type": "ENABLE_ALL_SLOW_POLL"})
        await c.stop()

    stamp = int(time.time())
    out = os.path.join(OUTDIR, f"bench_{args.mode}_{stamp}.json")
    json.dump({"joint": JOINT, "device_id": DEV, "Kt": KT, "gear": GEAR,
               "pos_limits": [POS_LO, POS_HI], "center": center,
               "orig_gains": ok, "results": results}, open(out, "w"))
    print(f"wrote {out}  ({len(results)} records)")


if __name__ == "__main__":
    asyncio.run(main())

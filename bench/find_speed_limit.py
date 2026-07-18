#!/usr/bin/env python3
"""Find the encoder's tracking ceiling via STEP moves (proven char5010 pattern), then
run randomized-speed excitation under the found limit.

Redesigned after the sustained-ramp version failed: position mode cannot hold a commanded
velocity, and driving a continuously-accumulating target out of range froze the position
count. This version copies the WORKING char5010 command pattern:
  * command a fixed STEP within the real position range; the motor accelerates toward the
    firmware velocity_limit and we capture the PEAK velocity of the transit
  * ALWAYS return to center between steps (never drift out of range)
  * raise velocity_limit stage by stage -> higher peak speed each time
  * per-sample abort on: ENCODER_FAULT bit, a sudden position jump (aliasing), or a hard
    over-speed ceiling. On abort -> IDLE immediately, re-home, report.

The glitch/vibration onset IS the encoder tracking limit; MAX = 0.75 x the peak speed at
which it first breaks. torque_limit capped (--tau) so any glitch is a bounded nudge.

Usage (daemon running with the joint's config, motor IDLE-able, shaft free):
    /usr/bin/python3 bench/find_speed_limit.py search --config <cfg> --joint <name>
    /usr/bin/python3 bench/find_speed_limit.py random --config <cfg> --joint <name> --max-speed 4.0
"""
from __future__ import annotations
import argparse, asyncio, json, os, sys, time
sys.path.insert(0, "/home/nse/humanoid/humanoid-studio/backend")
from humanoid.daemon_client import DaemonClient   # noqa: E402
from humanoid.robot_config import RobotConfig      # noqa: E402

P_ERROR, P_VEL_LIMIT, P_IQ = 0x014, 0x034, 0x0C0
ENCODER_FAULT = 0x2000
OUTDIR = os.path.join(os.path.dirname(__file__), "runs")


class Bench:
    def __init__(self, c, joint, chan, dev, gear):
        self.c, self.joint, self.chan, self.dev, self.gear = c, joint, chan, dev, gear
        self.aborted = False

    def st(self):
        return self.c.get_cached_joint_state(self.joint) or {}

    def fault(self):
        e = (self.c.generic_sdo_read(self.chan, self.dev, P_ERROR) or {}).get("value_u32") or 0
        return e & ENCODER_FAULT, e

    def set_vlim(self, v):
        self.c.generic_sdo_write(self.chan, self.dev, P_VEL_LIMIT, "f32", float(v))

    async def set_gains(self, kp, kd, ki, tau):
        await self.c.get_actuator_by_name(self.joint).write_gains(kp, ki, kd, tau)

    async def goto(self, target, settle=0.7):
        self.c.set_position(self.joint, target); await asyncio.sleep(settle)

    async def step_probe(self, center, target, gains, jump_tol, v_abort, dur=1.0, hz=200):
        """Step center->target, capture peak |vel|, watch for glitch. Always parks at center first."""
        kp, kd, ki, tau = gains
        await self.set_gains(kp, kd, ki, tau)
        await self.goto(center, 0.8)
        prev = self.st().get("position")
        peak, glitch, samples = 0.0, None, []
        t0 = time.perf_counter()
        self.c.set_position(self.joint, target)
        while time.perf_counter() - t0 < dur:
            # per-sample checks use ONLY fast-frame telemetry (no SDO -> no loop stall / race)
            s = self.st(); pos, vel = s.get("position"), s.get("velocity")
            if vel is not None:
                peak = max(peak, abs(vel))
            jump = abs(pos - prev) if (pos is not None and prev is not None) else 0.0
            samples.append({"t": time.perf_counter() - t0, "pos": pos, "vel": vel, "jump": jump})
            if jump > jump_tol:
                glitch = f"position jump {jump:.2f} rad/sample (aliasing)"
            elif vel is not None and abs(vel) > v_abort:
                glitch = f"over-speed {vel:.1f} rad/s (hard ceiling {v_abort})"
            if glitch:
                await self.c.get_actuator_by_name(self.joint).disable()   # KILL now
                self.aborted = True
                break
            prev = pos
            await asyncio.sleep(1.0 / hz)
        # confirm with the firmware fault bit ONCE (SDO), after motion has stopped
        fault, ecode = self.fault()
        if fault and not glitch:
            glitch = f"ENCODER_FAULT 0x{ecode & 0xFFFF:04X}"
        return peak, glitch, samples


async def search(b: Bench, center, args):
    stages, sign = [], 1.0
    vlim = args.v_start
    while vlim <= args.v_ceiling:
        b.set_vlim(vlim); await asyncio.sleep(0.05)
        lo = max(args.pos_lo + 0.05, center - args.step_size)
        hi = min(args.pos_hi - 0.05, center + args.step_size)
        target = hi if sign > 0 else lo
        peak, glitch, samples = await b.step_probe(
            center, target, (args.kp, args.kd, args.ki, args.tau),
            args.jump_tol, args.v_abort)
        stages.append({"vlim": vlim, "target": target, "peak_vel": peak,
                       "glitched": bool(glitch), "reason": glitch, "samples": samples})
        if glitch:
            print(f"  vlim={vlim:.1f}: peak {peak:.2f} rad/s ({peak*b.gear:.0f} motor) — GLITCH: {glitch}")
            # re-enable + re-home for a clean exit
            b.aborted = False
            try:
                self_pos = b.st().get("position", center)
                b.c.set_position(b.joint, self_pos)
                await b.c.get_actuator_by_name(b.joint).enable()
                await b.goto(center, 0.6)
            except Exception:
                pass
            return peak, glitch, stages
        print(f"  vlim={vlim:.1f} rad/s -> peak {peak:.2f} ({peak*b.gear:.0f} motor)  clean")
        await b.goto(center, 0.4)
        sign = -sign
        vlim += args.v_step
    print(f"  reached vlim ceiling {args.v_ceiling} with no glitch")
    return None, "no-glitch", stages


async def random_profile(b: Bench, center, args):
    speeds = [0.6, 3.0, 1.2, 3.8, 0.9, 2.6, 3.5, 1.7, 3.9, 0.7][:]
    speeds = [min(s, args.max_speed) for s in speeds]
    b.set_vlim(min(args.max_speed * 1.5, args.v_ceiling)); await asyncio.sleep(0.05)
    await b.set_gains(args.kp, args.kd, args.ki, args.tau)
    lo = max(args.pos_lo + 0.05, center - args.step_size)
    hi = min(args.pos_hi - 0.05, center + args.step_size)
    dt, log, tgt = 1.0 / 200.0, [], center
    await b.goto(center, 0.6)
    t0 = time.perf_counter()
    for spd in speeds:
        for goal in (hi, lo):
            direction = 1.0 if goal >= tgt else -1.0
            while (direction > 0 and tgt < goal) or (direction < 0 and tgt > goal):
                tgt += direction * spd * dt
                b.c.set_position(b.joint, max(lo, min(hi, tgt)))
                s = b.st(); fault, ecode = b.fault()
                log.append({"t": time.perf_counter() - t0, "cmd": tgt, "spd": spd * direction,
                            "pos": s.get("position"), "vel": s.get("velocity"),
                            "iq": (b.c.generic_sdo_read(b.chan, b.dev, P_IQ) or {}).get("value_f32"),
                            "error": ecode})
                if fault:
                    await b.c.get_actuator_by_name(b.joint).disable()
                    print(f"  ENCODER_FAULT during random run at spd={spd:.1f} — aborted"); return log
                await asyncio.sleep(dt)
    return log


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["search", "random"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--joint", required=True)
    ap.add_argument("--tau", type=float, default=3.0, help="torque_limit safety cap (N·m)")
    ap.add_argument("--kp", type=float, default=60.0, help="position_kp (best-gain default)")
    ap.add_argument("--kd", type=float, default=1.847, help="velocity_kp/Kd (best-gain default)")
    ap.add_argument("--ki", type=float, default=1.355, help="position_ki (best-gain default)")
    ap.add_argument("--step-size", type=float, default=0.6, help="step magnitude from center (rad)")
    ap.add_argument("--center", type=float, default=None, help="excitation center (default = mid-range)")
    # search
    ap.add_argument("--v-start", type=float, default=3.0, help="first velocity_limit (rad/s output)")
    ap.add_argument("--v-step", type=float, default=2.0, help="velocity_limit increment per stage")
    ap.add_argument("--v-ceiling", type=float, default=22.0, help="max velocity_limit to try")
    ap.add_argument("--v-abort", type=float, default=28.0, help="hard per-sample over-speed abort (rad/s)")
    ap.add_argument("--jump-tol", type=float, default=0.30, help="inter-sample jump = aliasing glitch (rad)")
    # random
    ap.add_argument("--max-speed", type=float, default=4.0, help="[random] speed ceiling (rad/s output)")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    cfg = RobotConfig.from_json(args.config)
    jc = cfg.joints[args.joint]
    chan, dev = jc.can_channel, jc.can_id
    gear = abs(jc.gear_ratio)
    args.pos_lo, args.pos_hi = jc.position_limits.lower_bound, jc.position_limits.upper_bound
    center = args.center if args.center is not None else round((args.pos_lo + args.pos_hi) / 2, 3)
    print(f"[cfg] joint={args.joint} dev={dev} gear={gear} limits=[{args.pos_lo:.2f},{args.pos_hi:.2f}] "
          f"center={center} gains=(kp{args.kp},kd{args.kd},ki{args.ki}) tau={args.tau}")

    c = DaemonClient(cfg); await c.start(); await asyncio.sleep(0.5)
    loop = asyncio.get_running_loop()
    c.disable_all_slow_poll()        # STOP slow-poll SDO telemetry — else error/param reads race it
    await asyncio.sleep(0.3)
    await loop.run_in_executor(None, c.connect_single, args.joint)   # DISABLED->IDLE + apply config
    await asyncio.sleep(0.4)
    b = Bench(c, args.joint, chan, dev, gear)
    vlim0 = (c.generic_sdo_read(chan, dev, P_VEL_LIMIT) or {}).get("value_f32")

    result = None
    try:
        await c.get_actuator_by_name(args.joint).clear_error(); await asyncio.sleep(0.2)
        c.set_position(args.joint, center)
        await c.get_actuator_by_name(args.joint).enable()
        c.set_position(args.joint, center); await asyncio.sleep(0.3)

        if args.mode == "search":
            peak, reason, stages = await search(b, center, args)
            rec = (0.75 * peak) if peak else None
            result = {"mode": "search", "peak_at_glitch": peak, "reason": reason,
                      "recommended_max": rec, "stages": stages}
            if peak:
                print(f"\n  RECOMMENDED MAX = {rec:.2f} rad/s output ({rec*gear:.0f} motor)  [0.75 x glitch peak]")
        else:
            log = await random_profile(b, center, args)
            result = {"mode": "random", "max_speed": args.max_speed, "log": log}
            print(f"  random run: {len(log)} samples")
    finally:
        try:
            c.set_position(args.joint, center); await asyncio.sleep(0.4)
        except Exception:
            pass
        if vlim0 is not None:
            b.set_vlim(vlim0)                                  # restore original velocity_limit
        try:
            await c.get_actuator_by_name(args.joint).disable()   # IDLE
        except Exception:
            pass
        c.enable_all_slow_poll()
        await c.stop()

    if result is not None:
        stamp = int(time.time())
        out = os.path.join(OUTDIR, f"speed_{args.mode}_{stamp}.json")
        json.dump({"joint": args.joint, "dev": dev, "gear": gear, **result}, open(out, "w"))
        print(f"wrote {out}")


if __name__ == "__main__":
    asyncio.run(main())

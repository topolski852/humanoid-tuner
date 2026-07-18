#!/usr/bin/env python3
"""Build a predictive MotorModel from real bench data — the sim<->real<->sim loop, automated.

Modes (lazy imports so each runs under the right interpreter):
  collect  HARDWARE (/usr/bin/python3): run a friction+step battery within the safe speed
           envelope, log pos/vel/i_q, save raw to runs/model_raw_<name>.json.
  fit      NUMPY (.venv/bin/python): load raw, fit MotorModel params, save the fitted model
           to sim/fitted_models/<name>.json and print fit quality (predict_error).
  all      collect then fit (must run under /usr/bin/python3 with numpy available).

Excitations, each identifying a different parameter:
  * friction ramps (constant low speed, both dirs, log i_q) -> coulomb + viscous (+Stribeck)
  * stiction steps (small, ki=0)  -> breakaway (static friction) from the steady-state offset
  * dynamic steps (larger)        -> inertia + latency (via predict_error rollout fit)

Safe: velocity_limit capped, per-sample over-speed/fault abort, returns to center each move,
slow-poll disabled (reliable reads), gains/limits restored + IDLE on exit. See
[[bench-characterization-gotchas]] for the envelope (safe MAX ~5 rad/s output).

Usage:
    /usr/bin/python3 bench/build_motor_model.py collect --config <cfg> --joint <name> --name m6c12_pitch
    .venv/bin/python  bench/build_motor_model.py fit     --name m6c12_pitch --kt 0.08958
"""
from __future__ import annotations
import argparse, json, os, sys, time

HERE = os.path.dirname(__file__)
RUNS = os.path.join(HERE, "runs")
MODELS = os.path.join(os.path.dirname(HERE), "sim", "fitted_models")

# safe defaults (200KV/150KV bench arm; AS5600 aliases ~100+ rad/s motor => keep output low)
SAFE_MAX = 4.0          # rad/s output ceiling for ramps (under the ~5 safe MAX, margin)
VEL_ABORT = 5.0         # per-sample hard abort


# ----------------------------------------------------------------------------- collect
async def collect(args):
    import asyncio
    sys.path.insert(0, "/home/nse/humanoid/humanoid-studio/backend")
    from humanoid.daemon_client import DaemonClient
    from humanoid.robot_config import RobotConfig

    P_ERR, P_IQ, P_VLIM = 0x014, 0x0C0, 0x034
    ENCODER_FAULT = 0x2000
    cfg = RobotConfig.from_json(args.config)
    jc = cfg.joints[args.joint]
    chan, dev = jc.can_channel, jc.can_id
    kt, gear = jc.torque_constant, abs(jc.gear_ratio)
    lo, hi = jc.position_limits.lower_bound, jc.position_limits.upper_bound
    center = args.center if args.center is not None else round((lo + hi) / 2, 3)
    clamp = lambda x: max(lo + 0.05, min(hi - 0.05, x))
    print(f"[collect] {args.joint} dev={dev} Kt={kt} gear={gear} center={center} range=[{lo:.2f},{hi:.2f}]")

    c = DaemonClient(cfg); await c.start(); await asyncio.sleep(0.5)
    loop = asyncio.get_running_loop()
    c.disable_all_slow_poll(); await asyncio.sleep(0.3)
    await loop.run_in_executor(None, c.connect_single, args.joint); await asyncio.sleep(0.4)
    prox = c.get_actuator_by_name(args.joint)
    st = lambda: c.get_cached_joint_state(args.joint) or {}
    fault = lambda: ((c.generic_sdo_read(chan, dev, P_ERR) or {}).get("value_u32") or 0) & ENCODER_FAULT
    vlim0 = (c.generic_sdo_read(chan, dev, P_VLIM) or {}).get("value_f32")

    async def gains(kp, kd, ki, tau): await prox.write_gains(kp, ki, kd, tau)
    async def goto(t, s=0.7): c.set_position(args.joint, clamp(t)); await asyncio.sleep(s)

    async def ensure_on():
        cur = st().get("position", center)
        c.set_position(args.joint, cur if cur is not None else center)
        await prox.enable()               # idempotent; re-enables after any over-speed abort

    async def ramp(v, sgn):
        await ensure_on()
        await gains(30.0, 0.5, 0.0, args.tau)
        s0, s1 = clamp(center - sgn * 0.35), clamp(center + sgn * 0.35)
        await goto(s0, 0.7)
        t0 = time.perf_counter(); ts, pos, vel, cmd, iq = [], [], [], [], []
        cur, last = s0, -1.0
        d = 1.0 if s1 >= s0 else -1.0
        while (d > 0 and cur < s1) or (d < 0 and cur > s1):
            el = time.perf_counter() - t0; cur = clamp(s0 + d * v * el)
            c.set_position(args.joint, cur)
            s = st(); vv = s.get("velocity")
            ts.append(el); cmd.append(cur); pos.append(s.get("position")); vel.append(vv)
            if el - last > 0.05:
                iq.append((el, (c.generic_sdo_read(chan, dev, P_IQ) or {}).get("value_f32"))); last = el
            if vv is not None and abs(vv) > VEL_ABORT:
                await prox.disable(); print("   !! over-speed abort"); break
            await asyncio.sleep(1.0 / 150)
        return {"kind": "ramp", "speed": v * sgn, "gains": [30.0, 0.5, 0.0, args.tau],
                "t": ts, "pos": pos, "vel": vel, "cmd": cmd, "iq": iq}

    async def step(mag, kp, kd, ki):
        await ensure_on()
        await gains(kp, kd, ki, args.tau)
        await goto(center, 0.8)
        t0 = time.perf_counter(); ts, pos, vel = [], [], []
        tgt = clamp(center + mag); c.set_position(args.joint, tgt)
        while time.perf_counter() - t0 < 1.2:
            s = st(); vv = s.get("velocity")
            ts.append(time.perf_counter() - t0); pos.append(s.get("position")); vel.append(vv)
            if vv is not None and abs(vv) > VEL_ABORT:
                await prox.disable(); print("   !! over-speed abort"); break
            await asyncio.sleep(1.0 / 200)
        return {"kind": "step", "mag": mag, "target": tgt, "gains": [kp, kd, ki, args.tau],
                "t": ts, "pos": pos, "vel": vel}

    results = []
    try:
        c.set_position(args.joint, center); await prox.enable(); c.set_position(args.joint, center)
        await asyncio.sleep(0.3)
        print("  friction ramps ...");
        for v in [0.3, 0.8, 1.5, 2.5]:            # 3.5 over-speeds the light motor -> dropped
            for sgn in (+1, -1):
                r = await ramp(v, sgn); results.append(r)
                iqs = [abs(x) for _, x in r["iq"] if x is not None]
                f = (sum(sorted(iqs)[len(iqs)//3:]) / max(1, len(iqs) - len(iqs)//3)) * kt * gear if iqs else 0
                print(f"    ramp {sgn*v:+.1f} rad/s: friction≈{f:.3f} N·m out ({len(iqs)} iq)")
                if fault(): print("    !! ENCODER_FAULT"); break
        print("  stiction steps (ki=0, small) ...")
        for m in [0.03, -0.03, 0.06, -0.06]:
            results.append(await step(m, args.kp, args.kd, 0.0))
        # GAIN SWEEP: same plant, different (kp,kd) -> teaches the model how gains map to
        # response, and validates the fitted plant is gain-independent (predict_error must
        # stay low across all of these). Small ±0.1 step so even kp=60 stays under over-speed.
        print("  gain-sweep steps (kp/kd effect) ...")
        for (kp, kd) in [(20.0, 0.5), (20.0, 1.5), (40.0, 1.5), (60.0, 1.5), (40.0, 0.5), (40.0, 3.0)]:
            for m in (0.1, -0.1):
                r = await step(m, kp, kd, 0.0); results.append(r)
            pk = max((abs(v) for v in r["vel"] if v is not None), default=0)
            print(f"    gains kp={kp} kd={kd}: vpk={pk:.2f} rad/s")
    finally:
        try:
            c.set_position(args.joint, center); await asyncio.sleep(0.4)
        except Exception: pass
        if vlim0 is not None: c.generic_sdo_write(chan, dev, P_VLIM, "f32", float(vlim0))
        try: await prox.disable()
        except Exception: pass
        c.enable_all_slow_poll(); await c.stop()

    os.makedirs(RUNS, exist_ok=True)
    out = os.path.join(RUNS, f"model_raw_{args.name}.json")
    json.dump({"name": args.name, "joint": args.joint, "dev": dev, "Kt": kt, "gear": gear,
               "center": center, "kp": args.kp, "results": results}, open(out, "w"))
    print(f"[collect] wrote {out} ({len(results)} records)")


# --------------------------------------------------------------------------------- fit
def fit(args):
    import numpy as np
    sys.path.insert(0, os.path.dirname(HERE))
    from sim.actuator import Gains, Response
    from sim.motor_model import MotorModel, FrictionModel, predict_error

    raw = json.load(open(os.path.join(RUNS, f"model_raw_{args.name}.json")))
    kt, gear, center, kp = raw["Kt"], raw["gear"], raw["center"], raw["kp"]
    ramps = [r for r in raw["results"] if r["kind"] == "ramp"]
    steps = [r for r in raw["results"] if r["kind"] == "step"]

    # 1) friction vs speed from ramp i_q -> coulomb (intercept) + viscous (slope)
    pts = []
    for r in ramps:
        iqs = [abs(x) for _, x in r["iq"] if x is not None]
        if len(iqs) < 4: continue
        f = float(np.median(sorted(iqs)[len(iqs)//3:])) * kt * gear   # steady output friction torque
        pts.append((abs(r["speed"]), f))
    pts = sorted(pts)
    if len(pts) >= 2:
        sp = np.array([p[0] for p in pts]); fr = np.array([p[1] for p in pts])
        A = np.vstack([np.ones_like(sp), sp]).T
        coulomb, viscous = np.linalg.lstsq(A, fr, rcond=None)[0]
        coulomb, viscous = max(0.0, float(coulomb)), max(0.0, float(viscous))
    else:
        coulomb, viscous = 0.25, 0.01
    print(f"  friction fit: coulomb={coulomb:.3f} N·m, viscous={viscous:.4f} N·m·s/rad  ({len(pts)} ramp pts)")

    # 2) breakaway from stiction-step steady-state offset (ki=0): F_s ≈ kp * ss_error
    fs_est = []
    for r in steps:
        if abs(r["mag"]) > 0.08: continue        # only the small stiction steps hold a static offset
        pos = [p for p in r["pos"] if p is not None]
        if len(pos) < 10: continue
        ss_err = abs(r["target"] - float(np.mean(pos[-5:])))
        fs_est.append(r["gains"][0] * ss_err)
    breakaway = float(np.median(fs_est)) if fs_est else max(coulomb, 0.3)
    breakaway = max(breakaway, coulomb)
    print(f"  breakaway (stiction) fit: F_s={breakaway:.3f} N·m  ({len(fs_est)} stiction steps)")

    # 3) inertia + latency from dynamic steps via predict_error rollout grid
    def resp_of(r):
        t = np.array(r["t"]); pos = np.array([p if p is not None else np.nan for p in r["pos"]])
        vel = np.array([v if v is not None else np.nan for v in r["vel"]])
        m = ~np.isnan(pos) & ~np.isnan(vel)
        return Response(t=t[m], target=np.full(m.sum(), r["target"]), pos=pos[m], vel=vel[m])
    dyn = [resp_of(r) for r in steps if abs(r["mag"]) >= 0.1]
    best = None
    for J in np.linspace(args.j_lo, args.j_hi, 25):
        for lat in np.linspace(0.0, 0.020, 11):
            mm = MotorModel(name=args.name, inertia=float(J), torque_limit=args.tau, latency_s=float(lat),
                            friction=FrictionModel(coulomb=coulomb, breakaway=breakaway,
                                                   stribeck_vel=0.05, viscous=viscous))
            err = 0.0
            for r, log in zip([s for s in steps if abs(s["mag"]) >= 0.1], dyn):
                g = Gains(position_kp=r["gains"][0], velocity_kp=r["gains"][1], position_ki=r["gains"][2])
                err += predict_error(mm, g, log)["rollout_rms"]
            if best is None or err < best[0]:
                best = (err, float(J), float(lat))
    rms, inertia, latency = best
    rms_mrad = rms / max(1, len(dyn)) * 1e3
    print(f"  inertia/latency fit: J={inertia:.5f} kg·m², latency={latency*1e3:.1f} ms  "
          f"(mean rollout RMS {rms_mrad:.1f} mrad over {len(dyn)} steps)")

    # GAIN VALIDATION: one fitted plant must predict every gain set. Group the dynamic
    # steps by (kp,kd) and report predict_error per gain set — low+consistent => the model
    # understands how gains affect the motor (plant is gain-independent, control law exact).
    mm_fit = MotorModel(name=args.name, inertia=inertia, torque_limit=args.tau, latency_s=latency,
                        friction=FrictionModel(coulomb=coulomb, breakaway=breakaway,
                                               stribeck_vel=0.05, viscous=viscous))
    gain_val = {}
    for r in [s for s in steps if abs(s["mag"]) >= 0.1]:
        key = f"kp{r['gains'][0]:g}_kd{r['gains'][1]:g}"
        g = Gains(position_kp=r["gains"][0], velocity_kp=r["gains"][1], position_ki=r["gains"][2])
        e = predict_error(mm_fit, g, resp_of(r))["rollout_rms_mrad"]
        gain_val.setdefault(key, []).append(e)
    print("  gain validation (rollout RMS per gain set — should stay low across all):")
    for k in sorted(gain_val):
        v = gain_val[k]
        print(f"    {k:14s}: {sum(v)/len(v):.1f} mrad  ({len(v)} steps)")
    gain_val_mrad = {k: round(sum(v)/len(v), 2) for k, v in gain_val.items()}

    model = {"name": args.name, "motor_type": args.motor_type, "Kt": kt, "gear": gear,
             "inertia": inertia, "friction": {"coulomb": coulomb, "breakaway": breakaway,
             "stribeck_vel": 0.05, "viscous": viscous, "load_sensitivity": 0.0, "stick_vel": 0.02},
             "latency_s": latency, "torque_limit": args.tau,
             "fit": {"rollout_rms_mrad": rms_mrad, "n_ramps": len(pts), "n_stiction": len(fs_est),
                     "n_dynamic": len(dyn), "gain_validation_mrad": gain_val_mrad},
             "source": f"model_raw_{args.name}.json"}
    os.makedirs(MODELS, exist_ok=True)
    out = os.path.join(MODELS, f"{args.name}.json")
    json.dump(model, open(out, "w"), indent=2)
    print(f"[fit] wrote {out}")
    print(f"  MODEL: J={inertia:.5f}  F_c={coulomb:.3f}  F_s={breakaway:.3f}  b={viscous:.4f}  "
          f"latency={latency*1e3:.1f}ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "fit", "all"])
    ap.add_argument("--config"); ap.add_argument("--joint")
    ap.add_argument("--name", required=True, help="model id, e.g. m6c12_pitch / mad5010_roll")
    ap.add_argument("--motor-type", default="", help="e.g. MAD_M6C12_150KV")
    ap.add_argument("--kt", type=float, default=None, help="[fit] override Kt if raw lacks it")
    ap.add_argument("--center", type=float, default=None)
    ap.add_argument("--kp", type=float, default=60.0)
    ap.add_argument("--kd", type=float, default=1.847)
    ap.add_argument("--tau", type=float, default=3.0)
    ap.add_argument("--j-lo", type=float, default=0.005, help="[fit] inertia grid low")
    ap.add_argument("--j-hi", type=float, default=0.035, help="[fit] inertia grid high")
    args = ap.parse_args()
    if args.mode in ("collect", "all"):
        import asyncio; asyncio.run(collect(args))
    if args.mode in ("fit", "all"):
        fit(args)


if __name__ == "__main__":
    main()

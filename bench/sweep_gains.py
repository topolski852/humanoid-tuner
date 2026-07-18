#!/usr/bin/env python3
"""Sweep a gain grid on the REAL gearboxed motor and rank by step-response fitness.

Model-based tuning (bench/tune_gains.py) can't set the real gain ceiling — a smooth model
has no sensor noise / quantization / unmodeled dynamics, so it over-rewards high gains. This
runs the actual grid on hardware and scores the real response with the SAME fitness
(sim/metrics.py) — grounding the winner in the real motor, incl. hunting/chatter the model
can't see.

Split by interpreter (daemon needs /usr/bin/python3; scoring needs the numpy .venv):
  collect  HARDWARE (/usr/bin/python3): run the grid, log raw step responses.
  score    NUMPY (.venv/bin/python): load raw, compute fitness + metrics, rank.

Safe (mirrors char5010): velocity_limit capped, per-sample over-speed + firmware-error abort,
returns to center each step, slow-poll disabled, gains/limits restored + IDLE on exit.

    /usr/bin/python3 bench/sweep_gains.py collect --config <cfg> --joint <name> --name mad5010_roll
    .venv/bin/python  bench/sweep_gains.py score   --name mad5010_roll
"""
from __future__ import annotations
import argparse, json, os, sys, time

HERE = os.path.dirname(__file__)
RUNS = os.path.join(HERE, "runs")
VLIM = 4.0            # rad/s output firmware cap (safe, under the ~5 aliasing-free envelope)
VEL_ABORT = 4.5
P_ERR, P_VLIM = 0x014, 0x034


async def collect(args):
    import asyncio
    sys.path.insert(0, "/home/nse/humanoid/humanoid-studio/backend")
    from humanoid.daemon_client import DaemonClient
    from humanoid.robot_config import RobotConfig

    kps = [float(x) for x in args.kps.split(",")]
    kds = [float(x) for x in args.kds.split(",")]
    kis = [float(x) for x in args.kis.split(",")]
    grid = [(kp, kd, ki) for kp in kps for kd in kds for ki in kis]

    cfg = RobotConfig.from_json(args.config); jc = cfg.joints[args.joint]
    chan, dev = jc.can_channel, jc.can_id
    lo, hi = jc.position_limits.lower_bound, jc.position_limits.upper_bound
    center = round((lo + hi) / 2, 3)
    clamp = lambda x: max(lo + 0.05, min(hi - 0.05, x))
    print(f"[collect] {args.joint} dev={dev} center={center}  {len(grid)} gain sets x 4 steps")

    c = DaemonClient(cfg); await c.start(); await asyncio.sleep(0.5)
    loop = asyncio.get_running_loop()
    c.disable_all_slow_poll(); await asyncio.sleep(0.3)
    await loop.run_in_executor(None, c.connect_single, args.joint); await asyncio.sleep(0.4)
    prox = c.get_actuator_by_name(args.joint)
    st = lambda: c.get_cached_joint_state(args.joint) or {}
    ferr = lambda: (c.generic_sdo_read(chan, dev, P_ERR) or {}).get("value_u32") or 0
    vlim0 = (c.generic_sdo_read(chan, dev, P_VLIM) or {}).get("value_f32")

    async def write_gains_retry(kp, kd, ki, tries=4):
        for i in range(tries):
            try:
                await prox.write_gains(kp, ki, kd, args.tau); return
            except Exception:
                if i == tries - 1:
                    raise
                await asyncio.sleep(0.3)     # transient SDO timeout -> settle and retry

    async def record_step(mag, gains, dur=1.0, hz=200):
        kp, kd, ki = gains
        await write_gains_retry(kp, kd, ki)
        c.set_position(args.joint, center); await asyncio.sleep(0.8)
        start = st().get("position", center)
        tgt = clamp(center + mag); t0 = time.perf_counter()
        ts, pos, vel = [], [], []
        c.set_position(args.joint, tgt)
        while time.perf_counter() - t0 < dur:
            s = st(); v = s.get("velocity")
            ts.append(time.perf_counter() - t0); pos.append(s.get("position")); vel.append(v)
            if (v is not None and abs(v) > VEL_ABORT) or ferr():
                await prox.disable(); await asyncio.sleep(0.05); await prox.enable(); break
            await asyncio.sleep(1.0 / hz)
        return {"mag": mag, "start": start, "t": ts, "pos": pos, "vel": vel}

    grid_results = []
    try:
        c.generic_sdo_write(chan, dev, P_VLIM, "f32", VLIM); await asyncio.sleep(0.1)
        c.set_position(args.joint, center); await prox.enable(); c.set_position(args.joint, center)
        await asyncio.sleep(0.3)
        for gains in grid:
            steps = [await record_step(mag, gains) for mag in (0.15, -0.15, 0.3, -0.3)]
            grid_results.append({"gains": list(gains), "steps": steps})
            ns = sum(sum(v is not None for v in s["vel"]) for s in steps)
            print(f"  kp{gains[0]:g} kd{gains[1]:g} ki{gains[2]:g}: {ns} samples")
    finally:
        try:
            c.set_position(args.joint, center); await asyncio.sleep(0.4)
        except Exception: pass
        if vlim0 is not None: c.generic_sdo_write(chan, dev, P_VLIM, "f32", float(vlim0))
        try: await prox.disable()
        except Exception: pass
        c.enable_all_slow_poll(); await c.stop()

    os.makedirs(RUNS, exist_ok=True)
    out = os.path.join(RUNS, f"gain_sweep_raw_{args.name}.json")
    json.dump({"name": args.name, "joint": args.joint, "center": center, "grid": grid_results}, open(out, "w"))
    print(f"[collect] wrote {out}")


def score(args):
    import numpy as np
    sys.path.insert(0, os.path.dirname(HERE))
    from sim.actuator import Response
    from sim.metrics import step_metrics, fitness

    raw = json.load(open(os.path.join(RUNS, f"gain_sweep_raw_{args.name}.json")))

    def resp_from(rec):
        t = np.array(rec["t"]); pos = np.array([p if p is not None else np.nan for p in rec["pos"]])
        vel = np.array([v if v is not None else np.nan for v in rec["vel"]])
        mask = ~np.isnan(pos) & ~np.isnan(vel); t, pos, vel = t[mask], pos[mask], vel[mask]
        d = 1.0 if rec["mag"] >= 0 else -1.0; mag = abs(rec["mag"])
        return Response(t=t, target=np.full(len(t), mag), pos=d * (pos - rec["start"]), vel=d * vel, step=mag)

    ranked = []
    for g in raw["grid"]:
        fits, mets = [], []
        for rec in g["steps"]:
            r = resp_from(rec)
            if len(r.pos) > 20:
                fits.append(fitness(r)); mets.append(step_metrics(r))
        if not fits:
            continue
        avg = {k: float(np.mean([m[k] for m in mets])) for k in mets[0]}
        ranked.append({"gains": g["gains"], "fitness": float(np.mean(fits)), "metrics": avg})
    ranked.sort(key=lambda r: -r["fitness"])

    print(f"[score] {args.name} — ranked (best first):")
    for r in ranked:
        g, m = r["gains"], r["metrics"]
        print(f"  kp{g[0]:g} kd{g[1]:g} ki{g[2]:g}: fit={r['fitness']:.2f}  "
              f"rise={m['rise_time']*1e3:.0f}ms settle={m['settle_time']*1e3:.0f}ms "
              f"over={m['overshoot']*100:.1f}% osc={m['oscillation']:.1f} ss={m['ss_error']*1e3:.1f}mrad")
    json.dump({"name": args.name, "ranked": ranked},
              open(os.path.join(RUNS, f"gain_sweep_{args.name}.json"), "w"))
    if ranked:
        b = ranked[0]
        print(f"\n  REAL-BEST: kp={b['gains'][0]:g} kd={b['gains'][1]:g} ki={b['gains'][2]:g}  "
              f"(fitness {b['fitness']:.2f}, overshoot {b['metrics']['overshoot']*100:.1f}%, osc {b['metrics']['oscillation']:.1f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "score"])
    ap.add_argument("--config"); ap.add_argument("--joint"); ap.add_argument("--name", required=True)
    ap.add_argument("--kps", default="30,45,60,75"); ap.add_argument("--kds", default="1.5,3.0,4.5")
    ap.add_argument("--kis", default="0.0"); ap.add_argument("--tau", type=float, default=3.0)
    args = ap.parse_args()
    if args.mode == "collect":
        import asyncio; asyncio.run(collect(args))
    else:
        score(args)


if __name__ == "__main__":
    main()

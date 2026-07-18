#!/usr/bin/env python3
"""Find robust base firmware gains for a fitted (gearboxed) motor — per motor.

Optimizes (position_kp, velocity_kp/Kd, position_ki) against the fitted MotorModel
(stick-slip friction + measured inertia), scored by the step-response fitness
(sim/metrics.py) WORST-CASE across step sizes and a domain-randomized inertia+friction
set (no-load -> loaded, ±friction). Worst-case DR => gains that stay stable from the
free bench motor to the loaded robot joint and across per-joint variation — the "base
best gains" to train the policy on. The policy sim (real URDF masses) refines from there.

Numpy only:
    .venv/bin/python bench/tune_gains.py --model mad5010_roll
    .venv/bin/python bench/tune_gains.py --model m6c12_pitch --load-max 3.0
"""
from __future__ import annotations
import argparse, json, os
import numpy as np

from sim.actuator import Gains
from sim.motor_model import MotorModel, FrictionModel
from sim.metrics import fitness, step_metrics, FITNESS_WEIGHTS

MODELS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sim", "fitted_models")


def build_mm(m, inertia_scale=1.0, friction_scale=1.0, latency=None):
    fr = m["friction"]
    return MotorModel(
        name=m["name"], inertia=m["inertia"] * inertia_scale, torque_limit=m["torque_limit"],
        latency_s=(m.get("latency_s", 0.0) if latency is None else latency),
        friction=FrictionModel(coulomb=fr["coulomb"] * friction_scale,
                               breakaway=fr["breakaway"] * friction_scale,
                               stribeck_vel=fr["stribeck_vel"], viscous=fr["viscous"] * friction_scale,
                               stick_vel=fr["stick_vel"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-random", type=int, default=500)
    ap.add_argument("--refine", type=int, default=6)
    ap.add_argument("--load-max", type=float, default=3.0,
                    help="max inertia scale (loaded joint) for the robustness DR set")
    ap.add_argument("--latency", type=float, required=True,
                    help="command->response latency (s) — MEASURED value, NOT the fit's 0 "
                         "(5010/200KV=0.012, M6C12/150KV=0.0072); critical: it's what limits gain aggressiveness")
    ap.add_argument("--kp-max", type=float, default=80.0)
    ap.add_argument("--kd-max", type=float, default=6.0)
    ap.add_argument("--ki-max", type=float, default=3.0)
    args = ap.parse_args()

    m = json.load(open(os.path.join(MODELS, f"{args.model}.json")))
    print(f"[tune] {args.model}: J={m['inertia']:.4f} Fc={m['friction']['coulomb']:.3f} "
          f"Fs={m['friction']['breakaway']:.3f} b={m['friction']['viscous']:.4f}  tau_lim={m['torque_limit']}")
    print(f"  objective weights: {FITNESS_WEIGHTS}")

    # DR set (inertia_scale, friction_scale): nominal, loaded, +friction, -friction, loaded+friction
    dr = [(1.0, 1.0), (args.load_max, 1.0), (1.0, 1.3), (1.0, 0.7), (args.load_max, 1.3)]
    mms = [build_mm(m, i, f, args.latency) for (i, f) in dr]
    steps = [0.1, 0.3, 0.5]
    bounds = np.array([[2.0, args.kp_max], [0.0, args.kd_max], [0.0, args.ki_max]])
    lo, hi = bounds[:, 0], bounds[:, 1]
    rng = np.random.default_rng(0)

    def worst_fitness(x):
        g = Gains(position_kp=float(x[0]), velocity_kp=float(x[1]), position_ki=float(x[2]))
        worst = np.inf
        for mm in mms:
            for s in steps:
                try:
                    f = fitness(mm.rollout(g, float(s), duration=1.0))
                except Exception:
                    f = -1e9
                worst = min(worst, f)
        return worst

    # random search
    best_x = lo + (hi - lo) * rng.random(3)
    best_r = worst_fitness(best_x)
    for _ in range(args.n_random):
        x = lo + (hi - lo) * rng.random(3)
        r = worst_fitness(x)
        if r > best_r:
            best_r, best_x = r, x
    # coordinate refine
    scale = 0.25 * (hi - lo)
    for _ in range(args.refine):
        improved = False
        for d in range(3):
            for sign in (+1, -1):
                x = best_x.copy(); x[d] = np.clip(x[d] + sign * scale[d], lo[d], hi[d])
                r = worst_fitness(x)
                if r > best_r:
                    best_r, best_x, improved = r, x, True
        if not improved:
            scale *= 0.5

    kp, kd, ki = (float(v) for v in best_x)
    print(f"\n  BEST GAINS: kp={kp:.2f}  kd(velocity_kp)={kd:.3f}  ki={ki:.3f}  "
          f"(worst-case fitness {best_r:.3f})")

    # report metrics on the NOMINAL (no-load) and LOADED model, per step
    g = Gains(position_kp=kp, velocity_kp=kd, position_ki=ki)
    print("  step-response on nominal (no-load) model:")
    for s in steps:
        mt = step_metrics(build_mm(m, latency=args.latency).rollout(g, s, duration=1.0))
        print(f"    {s:.1f} rad: rise={mt['rise_time']*1e3:.0f}ms settle={mt['settle_time']*1e3:.0f}ms "
              f"overshoot={mt['overshoot']*100:.1f}% ss_err={mt['ss_error']*1e3:.1f}mrad osc={mt['oscillation']:.0f}")
    print(f"  loaded ({args.load_max}x inertia) 0.3 rad step:")
    mt = step_metrics(build_mm(m, args.load_max, latency=args.latency).rollout(g, 0.3, duration=1.0))
    print(f"    rise={mt['rise_time']*1e3:.0f}ms settle={mt['settle_time']*1e3:.0f}ms "
          f"overshoot={mt['overshoot']*100:.1f}% osc={mt['oscillation']:.0f}")

    out = {"model": args.model, "gains": {"position_kp": kp, "velocity_kp": kd, "position_ki": ki},
           "worst_case_fitness": best_r, "load_max": args.load_max, "weights": FITNESS_WEIGHTS}
    op = os.path.join(MODELS, f"{args.model}_gains.json")
    json.dump(out, open(op, "w"), indent=2)
    print(f"  wrote {op}")


if __name__ == "__main__":
    main()

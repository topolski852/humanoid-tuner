"""Validate a fitted MotorModel in Isaac PhysX: does the stick-slip model integrated by
PhysX agree with the numpy MotorModel — across gains — for the SAME fitted plant?

This is the sim-validate leg of sim->real->sim: the model was fit to real bench data (numpy);
here we confirm the higher-fidelity PhysX substrate reproduces the same response, so the model
we export to the trainer behaves identically in the trainer's engine. Both run the shared
control law + the shared FrictionModel, so any gap is the PhysX solver vs numpy Euler.

Run under the policy venv:
    OMNI_KIT_ACCEPT_EULA=YES /home/nse/humanoid/humanoid-policy/.venv/bin/python \
        -m sim.isaac.validate_models --model mad5010_roll
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from sim.actuator import Gains
from sim.motor_model import MotorModel, FrictionModel

HERE = os.path.dirname(__file__)
MODELS = os.path.join(os.path.dirname(HERE), "fitted_models")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="fitted-model id in sim/fitted_models/")
    ap.add_argument("--step", type=float, default=0.2, help="step magnitude (rad)")
    ap.add_argument("--duration", type=float, default=1.0)
    args = ap.parse_args()

    m = json.load(open(os.path.join(MODELS, f"{args.model}.json")))
    fr = m["friction"]
    print(f"[validate] {args.model}: J={m['inertia']:.5f}  Fc={fr['coulomb']:.3f}  "
          f"Fs={fr['breakaway']:.3f}  b={fr['viscous']:.4f}  tau_lim={m['torque_limit']}")

    # gain sets spanning the validated range — one Isaac rollout scores all of them
    gainsets = [(20.0, 1.5, 0.0), (40.0, 1.5, 0.0), (60.0, 1.5, 0.0)]
    gains = np.array(gainsets, dtype=float)

    # --- numpy MotorModel (the fitted model) ---
    mm = MotorModel(name=args.model, inertia=m["inertia"], torque_limit=m["torque_limit"],
                    latency_s=0.0,  # substrate has no latency; compare like-for-like
                    friction=FrictionModel(coulomb=fr["coulomb"], breakaway=fr["breakaway"],
                                           stribeck_vel=fr["stribeck_vel"], viscous=fr["viscous"],
                                           stick_vel=fr["stick_vel"]))
    numpy_pos = []
    for (kp, kd, ki) in gainsets:
        r = mm.rollout(Gains(position_kp=kp, velocity_kp=kd, position_ki=ki), args.step,
                       duration=args.duration, sample_hz=100.0)
        numpy_pos.append(r.pos)
    T = min(len(p) for p in numpy_pos)
    numpy_pos = np.array([p[:T] for p in numpy_pos])          # [N, T]

    # --- Isaac PhysX substrate (same fitted plant + stick-slip friction) ---
    from .substrate import IsaacSingleJoint
    sim = IsaacSingleJoint(inertia=m["inertia"], num_envs=len(gainsets))
    out = sim.rollout(gains=gains, step_rad=args.step, duration=args.duration,
                      damping=fr["viscous"], coulomb=fr["coulomb"],
                      torque_limit=m["torque_limit"], torque_filter_alpha=0.1454,
                      breakaway=fr["breakaway"], stribeck_vel=fr["stribeck_vel"],
                      stick_vel=fr["stick_vel"], sample_hz=100.0)
    isaac_pos = out["pos"][:, :T]                              # [N, T]

    # --- compare per gain set (write results BEFORE closing the app) ---
    print(f"\n  gain set        RMS Δpos     max Δpos    numpy_final  isaac_final  (target {args.step})")
    lines = []
    for i, (kp, kd, ki) in enumerate(gainsets):
        d = numpy_pos[i] - isaac_pos[i]
        rms, mx = float(np.sqrt(np.mean(d**2))), float(np.max(np.abs(d)))
        lines.append((kp, kd, rms, mx, float(numpy_pos[i, -1]), float(isaac_pos[i, -1])))
        print(f"  kp{kp:g} kd{kd:g}      {rms*1e3:7.2f} mrad  {mx*1e3:7.2f} mrad   "
              f"{numpy_pos[i,-1]:.4f}     {isaac_pos[i,-1]:.4f}")
    overall = float(np.sqrt(np.mean((numpy_pos - isaac_pos) ** 2))) * 1e3
    verdict = "PASS (PhysX matches numpy for the fitted model)" if overall < 5.0 else \
              "CHECK (>5 mrad — investigate solver/friction)"
    print(f"\n  OVERALL numpy-vs-Isaac RMS = {overall:.2f} mrad   -> {verdict}")

    out_path = os.path.join(HERE, f"validate_{args.model}.json")
    json.dump({"model": args.model, "step": args.step, "overall_rms_mrad": overall,
               "per_gain": [{"kp": l[0], "kd": l[1], "rms_mrad": l[2]*1e3, "max_mrad": l[3]*1e3,
                             "numpy_final": l[4], "isaac_final": l[5]} for l in lines]},
              open(out_path, "w"), indent=2)
    print(f"  wrote {out_path}")
    sim.close()


if __name__ == "__main__":
    main()

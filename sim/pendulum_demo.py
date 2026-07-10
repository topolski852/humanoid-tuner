"""Pendulum-load demo — the concrete case for STATE-DEPENDENT gains.

A joint driving a mass under gravity has a pose-dependent load: gravity torque is
zero hanging straight down and peaks with the arm horizontal. A single FIXED gain
set therefore holds different poses with different error (droop) — it is a
compromise no fixed PID escapes. A gain SCHEDULE kp(pose) that stiffens near
horizontal holds every pose uniformly. That schedule is exactly what a learned,
state-dependent tuner produces and hand-tuning cannot — the whole thesis of this
project, now on a plant where it is unavoidable.

Run:
    ./.venv/bin/python -m sim.pendulum_demo
"""

from __future__ import annotations

import numpy as np

from sim.actuator import Gains, Plant, simulate_step
from sim.metrics import step_metrics

# A representative bench pendulum: the geared M6C12 driving a 0.5 kg mass at 0.2 m.
MOTOR_INERTIA = 0.024
MASS, RADIUS = 0.5, 0.2
PLANT_KW = dict(damping=0.05, coulomb=0.46)   # measured geared friction
POSES = [0.2, 0.5, 0.9, 1.2, 1.57]            # rad from hanging (1.57 = horizontal)


def _plant():
    return Plant.pendulum(MOTOR_INERTIA, MASS, RADIUS, **PLANT_KW)


def _droop(kp, kd, ki, theta):
    """Steady-state error holding pose `theta` (rad) with these gains."""
    r = simulate_step(Gains(kp, kd, ki), _plant(), step_rad=theta, duration=3.0)
    return theta - float(r.pos[-1])


def main():
    P = _plant()
    print("=" * 74)
    print(f"PENDULUM: geared M6C12 (J_motor={MOTOR_INERTIA}) + {MASS} kg @ {RADIUS} m")
    print(f"  total inertia {P.inertia:.4f} kg·m²   peak gravity {P.gravity_torque:.3f} N·m")
    print("=" * 74)

    # --- A) fixed gains: droop grows toward horizontal (the problem) -----------
    KP_FIX, KD, KI = 60.0, 3.0, 0.0
    print(f"\nA) FIXED gains kp={KP_FIX} kd={KD} ki={KI}:")
    print(f"   {'pose':>6} {'gravity':>9} {'droop':>9}")
    fixed_droops = []
    for th in POSES:
        d = _droop(KP_FIX, KD, KI, th)
        fixed_droops.append(d)
        print(f"   {np.degrees(th):5.0f}° {P.gravity_torque*np.sin(th):8.3f}N·m {np.degrees(d):7.2f}°")
    print(f"   -> droop spans {np.degrees(min(fixed_droops)):.2f}°..{np.degrees(max(fixed_droops)):.2f}° "
          f"({max(fixed_droops)/min(fixed_droops):.1f}× worse at horizontal). One kp can't hold all poses.")

    # --- B) STATE-DEPENDENT kp: stiffen with gravity so droop is uniform -------
    # A learned tuner outputs gains as a function of pose. Emulate the schedule it
    # would find: scale kp with the local gravity load to hold a constant droop.
    TARGET_DROOP = np.radians(0.15)
    print(f"\nB) STATE-DEPENDENT kp(pose) targeting a uniform {np.degrees(TARGET_DROOP):.2f}° droop:")
    print(f"   {'pose':>6} {'kp(pose)':>9} {'droop':>9}")
    sched_droops = []
    for th in POSES:
        # kp needed ~ gravity_torque·sin(theta)/target_droop; clamp to a sane ceiling
        kp = float(np.clip(P.gravity_torque * np.sin(th) / max(TARGET_DROOP, 1e-3), 40.0, 300.0))
        d = _droop(kp, KD, KI, th)
        sched_droops.append(d)
        print(f"   {np.degrees(th):5.0f}° {kp:8.1f} {np.degrees(d):7.2f}°")
    print(f"   -> droop now {np.degrees(min(sched_droops)):.2f}°..{np.degrees(max(sched_droops)):.2f}° "
          "— roughly uniform across poses. This kp(pose) schedule is what the tuner learns.")

    # --- C) why not just crank fixed kp? saturation + overshoot ----------------
    print("\nC) why not one big fixed kp everywhere?")
    for kp in [60, 150, 300]:
        r = simulate_step(Gains(kp, KD, KI), P, step_rad=1.57, duration=3.0)
        m = step_metrics(r)
        print(f"   kp={kp:3d} @ horizontal: droop={np.degrees(1.57-r.pos[-1]):.2f}° "
              f"overshoot={m['overshoot']*100:.1f}% settle={m['settle_time']:.2f}s")
    print("   -> higher fixed kp cuts droop but risks overshoot/instability near vertical where")
    print("      gravity is weak. Pose-dependent gains get the best of both — the differentiator.")


if __name__ == "__main__":
    main()

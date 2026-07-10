"""Single-actuator sim replicating the firmware POSITION-mode control law.

Faithful to humanoid-studio/HANDOFF.md §2 (verified from firmware): a 2 kHz
position loop feeding an (assumed-ideal) FOC current loop, driving a rotational
plant. Output-shaft units (post-gearbox), matching what the real daemon reports.

Phase 0 keeps the plant a simple second-order system (inertia + viscous damping,
optional Coulomb friction). Phase 1 adds load/inertia/friction/backlash
randomization here — that's the only file that has to change.

Internally integrates at the control rate (2 kHz); exposes samples at 100 Hz —
the SAME rate the real hardware streams position/velocity (DAEMON_SPEC PDO4), so
a reward/feature-extractor sees an identical signal in sim and on the bench.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Gains:
    position_kp: float
    velocity_kp: float          # acts as Kd in POSITION mode (vel target hard-wired 0)
    position_ki: float = 0.0


@dataclass
class Plant:
    inertia: float = 8.0e-4     # kg·m², output-shaft reflected
    damping: float = 0.02       # N·m·s/rad, viscous
    coulomb: float = 0.0        # N·m, dry friction (Phase 1: randomize)
    torque_limit: float = 2.0   # N·m, matches config torque_limit
    torque_filter_alpha: float = 0.1454   # firmware EMA α (~50 Hz at 2 kHz)


@dataclass
class Response:
    """A sampled step response @ sample_hz — the tuner's view of the joint."""
    t: np.ndarray
    target: np.ndarray
    pos: np.ndarray
    vel: np.ndarray
    step: float = field(default=1.0)   # commanded step magnitude (rad)

    @property
    def err(self) -> np.ndarray:
        return self.target - self.pos


def simulate_step(
    gains: Gains,
    plant: Plant | None = None,
    step_rad: float = 0.5,
    duration: float = 1.0,
    ctrl_hz: float = 2000.0,
    sample_hz: float = 100.0,
) -> Response:
    """Drive a position step from 0 → step_rad and return the sampled response."""
    plant = plant or Plant()
    dt = 1.0 / ctrl_hz
    n = int(duration * ctrl_hz)
    stride = max(1, int(round(ctrl_hz / sample_hz)))

    pos = vel = integ = tau_f = 0.0
    tau_lim = plant.torque_limit

    ts, tgt, ps, vs = [], [], [], []
    for i in range(n):
        target = step_rad
        perr = target - pos
        verr = -vel  # firmware: velocity target is hard-wired 0 in POSITION mode

        # firmware accumulates the integrator per-tick (no dt), then clamps to ±τ_lim
        integ = float(np.clip(integ + gains.position_ki * perr, -tau_lim, tau_lim))
        torque_target = (
            gains.position_kp * perr + gains.velocity_kp * verr + integ
        )
        # EMA filter then clamp to the torque limit
        tau_f += plant.torque_filter_alpha * (torque_target - tau_f)
        torque = float(np.clip(tau_f, -tau_lim, tau_lim))

        # plant: J·acc = τ − b·ω − τ_coulomb·sign(ω)   (ideal current loop: i_q→τ)
        friction = plant.coulomb * np.sign(vel)
        acc = (torque - plant.damping * vel - friction) / plant.inertia
        vel += acc * dt
        pos += vel * dt

        if i % stride == 0:
            ts.append(i * dt); tgt.append(target); ps.append(pos); vs.append(vel)

    return Response(
        t=np.asarray(ts), target=np.asarray(tgt),
        pos=np.asarray(ps), vel=np.asarray(vs), step=step_rad,
    )

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

from .control_law import FirmwarePositionController


@dataclass
class Gains:
    position_kp: float
    velocity_kp: float          # acts as Kd in POSITION mode (vel target hard-wired 0)
    position_ki: float = 0.0


@dataclass
class Plant:
    inertia: float = 8.0e-4     # kg·m², motor-side reflected inertia (rotor + gearbox input)
    damping: float = 0.02       # N·m·s/rad, viscous (motor side)
    coulomb: float = 0.0        # N·m, dry friction (gearbox friction lives here once geared)
    torque_limit: float = 2.0   # N·m, matches config torque_limit
    torque_filter_alpha: float = 0.1454   # firmware EMA α (~50 Hz at 2 kHz)

    # --- Phase-1 gearbox terms (backward compatible: all 0 -> single-mass plant) ---
    # A two-mass model: the controlled/encoder-side "motor" (inertia above) drives a
    # LOAD (load_inertia) through a backlash dead-band + mesh spring. On a free light
    # shaft backlash is nearly invisible (encoder is motor-side); it shows up with a
    # LOAD. Calibrate these against real geared+loaded data — first-cut defaults.
    backlash: float = 0.0        # rad, total dead-band width (output frame)
    load_inertia: float = 0.0    # kg·m², inertia behind the backlash (0 -> no load stage)
    load_coulomb: float = 0.0    # N·m, Coulomb on the load side
    mesh_stiffness: float = 5.0e3   # N·m/rad, gearbox torsional stiffness when engaged
    mesh_damping: float = 5.0        # N·m·s/rad, mesh contact damping (numerical stability)

    # --- gravity / pendulum load (Phase-1: pose-dependent torque) ---------------
    # A mass on a horizontal-axis joint: gravity_torque = m·g·r is the PEAK torque
    # (arm horizontal); it vanishes when the load hangs straight down at gravity_zero.
    # The load also adds m·r² of inertia (fold into `inertia`, or use Plant.pendulum).
    gravity_torque: float = 0.0  # N·m, peak gravity load (0 -> no gravity, default)
    gravity_zero: float = 0.0    # rad, joint angle where the load hangs down (τ_grav = 0)

    @classmethod
    def pendulum(cls, motor_inertia: float, mass: float, radius: float,
                 gravity_zero: float = 0.0, g: float = 9.81, **kw) -> "Plant":
        """Build a Plant for a joint driving a point mass `mass` at `radius`.

        Adds the pendulum inertia (m·r²) to the motor's reflected inertia and sets
        gravity_torque = m·g·r. Pass damping/coulomb/etc. via kwargs.
        """
        return cls(inertia=motor_inertia + mass * radius ** 2,
                   gravity_torque=mass * g * radius, gravity_zero=gravity_zero, **kw)


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

    pos = vel = 0.0
    # The controller is the SHARED firmware law (sim/control_law.py) — the exact
    # code the Isaac substrate runs, so an agreement test isolates plant effects.
    ctrl = FirmwarePositionController(
        position_kp=gains.position_kp,
        velocity_kp=gains.velocity_kp,
        position_ki=gains.position_ki,
        torque_limit=plant.torque_limit,
        torque_filter_alpha=plant.torque_filter_alpha,
    )

    geared = plant.backlash > 0.0 or plant.load_inertia > 0.0
    pos_L = vel_L = 0.0           # load state (only used in the two-mass/geared model)
    half_bl = plant.backlash / 2.0

    ts, tgt, ps, vs = [], [], [], []
    for i in range(n):
        target = step_rad
        torque = float(ctrl.step(pos, vel, target))

        if not geared:
            # single-mass plant: J·acc = τ + τ_grav − b·ω − τ_coulomb·sign(ω)
            friction = plant.coulomb * np.sign(vel)
            grav = -plant.gravity_torque * np.sin(pos - plant.gravity_zero)
            acc = (torque + grav - plant.damping * vel - friction) / plant.inertia
            vel += acc * dt
            pos += vel * dt
        else:
            # two-mass: motor (encoder-side, reported) drives a load through a backlash
            # dead-band + mesh spring. Backlash only bites once the load is engaged.
            gap = pos - pos_L
            if gap > half_bl:
                mesh = plant.mesh_stiffness * (gap - half_bl) + plant.mesh_damping * (vel - vel_L)
            elif gap < -half_bl:
                mesh = plant.mesh_stiffness * (gap + half_bl) + plant.mesh_damping * (vel - vel_L)
            else:
                mesh = 0.0        # in the dead-band: motor free, no torque transmitted
            acc = (torque - mesh - plant.damping * vel - plant.coulomb * np.sign(vel)) / plant.inertia
            vel += acc * dt
            pos += vel * dt
            if plant.load_inertia > 0.0:
                grav_L = -plant.gravity_torque * np.sin(pos_L - plant.gravity_zero)
                acc_L = (mesh + grav_L - plant.load_coulomb * np.sign(vel_L)) / plant.load_inertia
                vel_L += acc_L * dt
                pos_L += vel_L * dt

        if i % stride == 0:
            ts.append(i * dt); tgt.append(target); ps.append(pos); vs.append(vel)

    return Response(
        t=np.asarray(ts), target=np.asarray(tgt),
        pos=np.asarray(ps), vel=np.asarray(vs), step=step_rad,
    )

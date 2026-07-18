"""Predictive motor+gearbox model — the forward model this repo exports to the trainer.

This is the deliverable of the tuner: a *proactive, predictive* forward model of a
single joint (motor + gearbox), accurate enough to drop into the robot policy trainer
so the learned policy anticipates the real actuator's behaviour instead of the
firmware PID reactively chasing it.

Design (hybrid — analytic core + learned residual):

    applied joint torque  =  sat(firmware PID)                 # commanded torque
                           -  friction(vel, load)              # sticky gearbox (analytic)
                           +  residual(pos_err, vel, load)     # learned (backlash/wear/per-joint)

The analytic core is fully determined by BENCH-MEASURED params (sim/isaac/motor.py):
effective output inertia, torque limit, latency, and a Stribeck/stiction friction model.
The residual is a plugged-in callable, a no-op until trained on loaded bench / robot data.

Two views, one physics (mirrors sim/actuator.py's control-rate integrate, 100 Hz sample):

  * `applied_torque(...)` — the ACTUATOR function the trainer needs. Given joint state,
    commanded torque and the external load torque (which the multibody sim supplies),
    returns the net torque delivered to the joint. Isaac integrates J·acc itself, so
    LOAD is an *input* here — never a baked-in weight (see [[robot-mass-budget]]).
  * `rollout(...)` — a self-contained integrator (uses the measured output inertia) for
    bench validation and the fast numpy substrate: command trajectory -> predicted motion.

`predict_error(...)` scores the model against a logged (command -> actual) trajectory:
the one-step and free-run rollout RMS that the bench<->robot loop drives toward zero.

Array-native like sim/control_law.py: state/params may be scalars (one joint) or [N]
arrays (N joints/envs), so the same model vectorises across a joint population.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .actuator import Gains, Response
from .control_law import FirmwarePositionController

ArrayLike = float | np.ndarray


@dataclass
class FrictionModel:
    """Stick-slip (Karnopp) dry friction with load-dependent stickiness — the "sticky gearbox".

    Improves on the plant's plain ``coulomb·sign(vel)`` in three ways that matter for
    a real geared joint:

      * **Static hold (stick regime)** — below the stick velocity ``stick_vel`` the joint
        is treated as stopped: friction opposes the *applied* torque exactly, up to the
        breakaway limit ``breakaway``, so the joint stays put and HOLDS a steady-state
        position offset (== applied/kp). A bare ``sign(vel)`` or ``tanh(vel)`` model has
        zero friction at rest and always creeps to target — it cannot reproduce the
        real motor's stiction dead-zone (measured: ~49 mrad on the 200 KV at kp=20, ki=0).
      * **Stribeck sliding** — once moving, kinetic friction starts at ``breakaway`` and
        relaxes to the sliding Coulomb level ``coulomb`` over ``stribeck_vel`` ("breaks
        free, then eases"), plus viscous ``viscous``.
      * **Load-dependent friction** — gearbox tooth-normal force rises with transmitted
        torque, so both levels grow with ``|load|`` at rate ``load_sensitivity``. This is
        the ONLY correct way "load" enters a motor model (as an input, not a weight).

    ``torque(vel, drive, load)`` needs the *drive* torque (net non-friction torque, i.e.
    controller output minus external load) to resolve the static regime. Reduces to plain
    Coulomb+viscous when ``breakaway == coulomb`` and ``load_sensitivity == 0``.
    """

    coulomb: float = 0.0          # F_c, sliding Coulomb level (N·m)
    breakaway: float = 0.0        # F_s, static/stiction peak (N·m); <=coulomb -> no stiction hump
    stribeck_vel: float = 0.05    # v_s, velocity scale over which stiction decays (rad/s)
    viscous: float = 0.0          # b, viscous damping (N·m·s/rad)
    load_sensitivity: float = 0.0  # extra Coulomb per N·m of |load| (dimensionless)
    stick_vel: float = 0.02       # |vel| below this -> static regime (rad/s)

    def torque(self, vel: ArrayLike, drive: ArrayLike, tau_load: ArrayLike = 0.0) -> ArrayLike:
        """Friction torque. Stick regime -> opposes ``drive`` up to F_s (static hold);
        slip regime -> Stribeck kinetic friction opposing ``vel``."""
        vel = np.asarray(vel, dtype=float)
        drive = np.asarray(drive, dtype=float)
        load = np.abs(np.asarray(tau_load, dtype=float))
        fc = self.coulomb + self.load_sensitivity * load
        fs = np.maximum(self.breakaway, self.coulomb) + self.load_sensitivity * load

        stuck = np.abs(vel) < self.stick_vel
        # static: hold against drive, saturating at breakaway (== the stiction dead-zone)
        static = np.clip(drive, -fs, fs)
        # kinetic: Stribeck curve relaxing breakaway -> coulomb, opposing motion
        kinetic = (fc + (fs - fc) * np.exp(-((vel / self.stribeck_vel) ** 2))) * np.sign(vel) \
            + self.viscous * vel
        return np.where(stuck, static, kinetic)


# A learned residual: (pos_err, vel, tau_load) -> torque correction (N·m).
# Scalars or matching [N] arrays. `None` on a MotorModel means "no residual yet".
Residual = Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike]


@dataclass
class MotorModel:
    """Forward model of one joint: PID -> saturation -> sticky friction (+ residual)."""

    name: str
    inertia: float               # kg·m², effective OUTPUT-shaft inertia (== Isaac `armature`)
    friction: FrictionModel = field(default_factory=FrictionModel)
    torque_limit: float = 2.0    # N·m, firmware clamp
    torque_filter_alpha: float = 0.1454
    latency_s: float = 0.0       # command->response transport delay
    residual: Residual | None = None

    @classmethod
    def from_spec(cls, spec, **override) -> "MotorModel":
        """Build from a bench-characterized :class:`sim.isaac.motor.MotorSpec`.

        Uses the MEASURED output inertia/friction/latency. Seeds the friction model
        with the measured Coulomb+viscous and, by default, no stiction hump or load
        term (``breakaway == coulomb``) — those are what the loaded bench + residual
        fill in later. Override any field via kwargs.
        """
        fr = FrictionModel(
            coulomb=spec.coulomb_friction or 0.0,
            breakaway=spec.coulomb_friction or 0.0,
            viscous=spec.viscous_damping or 0.0,
        )
        params = dict(
            name=spec.name,
            inertia=spec.measured_inertia if spec.measured_inertia is not None else spec.reflected_inertia,
            friction=fr,
            torque_limit=spec.torque_limit,
            torque_filter_alpha=spec.torque_filter_alpha,
            latency_s=spec.latency_s or 0.0,
        )
        params.update(override)
        return cls(**params)

    def applied_torque(self, pos_err: ArrayLike, vel: ArrayLike,
                       tau_ctrl: ArrayLike, tau_load: ArrayLike = 0.0) -> ArrayLike:
        """Net joint torque delivered = commanded(saturated) - friction (+ residual).

        This is the actuator map the trainer applies; the multibody sim owns J·acc and
        supplies ``tau_load``. ``tau_ctrl`` is the firmware PID output (already clamped
        and filtered by :class:`FirmwarePositionController`). ``drive = tau_ctrl - tau_load``
        is the net torque friction must resist (needed to resolve the static hold regime).
        """
        drive = tau_ctrl - tau_load
        tau = tau_ctrl - self.friction.torque(vel, drive, tau_load)
        if self.residual is not None:
            tau = tau + self.residual(pos_err, vel, tau_load)
        return tau

    def rollout(self, gains: Gains, targets: ArrayLike, *,
                tau_load: ArrayLike = 0.0, duration: float | None = None,
                ctrl_hz: float = 2000.0, sample_hz: float = 100.0,
                x0: tuple[float, float] = (0.0, 0.0)) -> Response:
        """Integrate the joint under a command trajectory; return the sampled Response.

        ``targets`` is either a scalar (held constant — a step, matching
        ``actuator.simulate_step``) or an array sampled at ``ctrl_hz``. ``tau_load`` is
        the external/gravity torque (scalar or per-tick array). Uses the measured
        output inertia so the free-run prediction is directly comparable to bench data.
        """
        dt = 1.0 / ctrl_hz
        if np.isscalar(targets):
            if duration is None:
                duration = 1.0
            n = int(duration * ctrl_hz)
            tgt_of = lambda i: float(targets)
        else:
            targets = np.asarray(targets, dtype=float)
            n = len(targets)
            tgt_of = lambda i: float(targets[i])

        load_arr = None if np.isscalar(tau_load) else np.asarray(tau_load, dtype=float)
        load_of = (lambda i: float(tau_load)) if load_arr is None else (lambda i: float(load_arr[i]))

        delay = int(round(self.latency_s * ctrl_hz))   # command->response transport lag
        stride = max(1, int(round(ctrl_hz / sample_hz)))

        ctrl = FirmwarePositionController(
            position_kp=gains.position_kp, velocity_kp=gains.velocity_kp,
            position_ki=gains.position_ki, torque_limit=self.torque_limit,
            torque_filter_alpha=self.torque_filter_alpha,
        )

        pos, vel = float(x0[0]), float(x0[1])
        ts, tgt, ps, vs = [], [], [], []
        for i in range(n):
            target = tgt_of(max(0, i - delay))         # controller sees a delayed command
            load = load_of(i)
            tau_ctrl = float(ctrl.step(pos, vel, target))
            tau = self.applied_torque(target - pos, vel, tau_ctrl, load)
            acc = (tau - load) / self.inertia          # J·acc = delivered torque - external load
            vel += acc * dt
            pos += vel * dt
            if i % stride == 0:
                ts.append(i * dt); tgt.append(tgt_of(i)); ps.append(pos); vs.append(vel)

        step_mag = float(targets) if np.isscalar(targets) else float(np.max(np.abs(targets)))
        return Response(
            t=np.asarray(ts), target=np.asarray(tgt),
            pos=np.asarray(ps), vel=np.asarray(vs), step=step_mag,
        )


def predict_error(model: MotorModel, gains: Gains, log: Response, *,
                  tau_load: ArrayLike = 0.0, ctrl_hz: float = 2000.0) -> dict:
    """Score a model's PREDICTION against a logged (command -> actual) trajectory.

    This is the objective the bench<->robot loop minimizes. Two numbers:

      * ``onestep_rms`` — given each logged (pos, vel), predict one control tick ahead
        and compare to the next logged sample. Measures instantaneous dynamics fidelity,
        independent of error accumulation.
      * ``rollout_rms`` — free-run the model from the initial state under the logged
        command and compare the whole predicted trajectory to the actual. Measures how
        the model holds up over a full motion (where residual/friction errors compound).

    ``log`` must carry the actual sampled pos/vel and the command in ``log.target``.
    Returns RMS position errors (rad) plus the predicted rollout for plotting.
    """
    t = np.asarray(log.t, dtype=float)
    m = len(log.pos)
    dt_s = float(np.median(np.diff(t))) if m > 1 else 1.0 / 100.0
    dt = 1.0 / ctrl_hz
    stride = max(1, int(round(dt_s * ctrl_hz)))        # control ticks per logged sample
    delay = int(round(model.latency_s * ctrl_hz))
    load_of = (lambda i: float(tau_load)) if np.isscalar(tau_load) \
        else (lambda i: float(np.asarray(tau_load)[i]))

    def integrate(pos, vel, ctrl, i_start, n_samples, delayed_target):
        """Run n_samples logged intervals at ctrl_hz (ZOH command); return sampled pos."""
        out = []
        for k in range(n_samples):
            target = float(log.target[max(0, i_start + k - delay)]) if delayed_target \
                else float(log.target[i_start + k])
            load = load_of(i_start + k)
            for _ in range(stride):
                tau_ctrl = float(ctrl.step(pos, vel, target))
                tau = model.applied_torque(target - pos, vel, tau_ctrl, load)
                vel += (tau - load) / model.inertia * dt
                pos += vel * dt
            out.append(pos)
        return out, pos, vel

    # --- free-run rollout: one controller, integrate the whole logged command ---
    ctrl = FirmwarePositionController(
        position_kp=gains.position_kp, velocity_kp=gains.velocity_kp,
        position_ki=gains.position_ki, torque_limit=model.torque_limit,
        torque_filter_alpha=model.torque_filter_alpha)
    pred, _, _ = integrate(float(log.pos[0]), float(log.vel[0]), ctrl, 0, m, delayed_target=True)
    pred = np.asarray(pred)
    rollout_rms = float(np.sqrt(np.mean((pred - log.pos) ** 2)))
    roll = Response(t=t, target=np.asarray(log.target), pos=pred,
                    vel=np.asarray(log.vel), step=float(getattr(log, "step", 1.0)))

    # --- one-step: from each real (pos,vel) advance a single logged interval ---
    errs = []
    for i in range(m - 1):
        c = FirmwarePositionController(
            position_kp=gains.position_kp, velocity_kp=gains.velocity_kp,
            position_ki=gains.position_ki, torque_limit=model.torque_limit,
            torque_filter_alpha=model.torque_filter_alpha)
        out, _, _ = integrate(float(log.pos[i]), float(log.vel[i]), c, i, 1, delayed_target=False)
        errs.append(out[0] - float(log.pos[i + 1]))
    onestep_rms = float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")

    return {
        "onestep_rms": onestep_rms,
        "rollout_rms": rollout_rms,
        "onestep_rms_mrad": onestep_rms * 1e3,
        "rollout_rms_mrad": rollout_rms * 1e3,
        "predicted": roll,
    }

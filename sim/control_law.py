"""The firmware POSITION-mode control law — the ONE source of truth.

Both substrates run this exact code so an agreement test measures *plant*
differences, not two drifting re-implementations of the controller:

  - sim/actuator.py       — numpy point-mass plant (fast reward-search loop)
  - sim/isaac/substrate.py — Isaac Lab / PhysX rigid-body plant (validation + RL)

Faithful to humanoid-studio/HANDOFF.md §2 (verified from firmware): a 2 kHz
position PD+I feeding an *ideal* FOC current loop. Ideal current loop is the
right Phase-0 assumption: the firmware computes i_q = tau / Kt / gear and the
loop makes i_q_measured = i_q_target, so tau_produced = i_q * Kt * gear = tau —
Kt and gear cancel. They re-enter only when the current limit binds (20 A ->
~27 Nm at the M6C12 output, far above the 2.0 Nm torque_limit) or when the
electrical dynamics are modelled — neither of which is observable at the 100 Hz
output-shaft sampling this project tunes against (HANDOFF §3). So the current
loop is modelled as tau_setpoint -> tau, and Kt/gear/current-limit live in the
MotorSpec for when Phase 1 load-saturation needs them.

Array-native by design: pos/vel/target and every gain may be a scalar OR a
numpy array of shape [N]. Scalar -> the numpy sim's single joint. Array [N] ->
N Isaac envs each with its own gains, which is exactly the vectorised gain
search the Eureka loop needs (one PhysX rollout scores a whole gain population).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ArrayLike = float | np.ndarray


@dataclass
class FirmwarePositionController:
    """One 2 kHz position-loop tick of the firmware control law.

    Gains and limits may be scalars (one joint) or [N] arrays (N envs, per-env
    gains). State (`integ`, `tau_f`) is lazily shaped to match on first `step`.

    `torque_limit` and `torque_filter_alpha` are firmware/config parameters, not
    plant parameters — they belong to the controller, not the mechanism. The
    returned torque is the clamped, EMA-filtered `torque_setpoint`; the plant is
    responsible for its own damping/friction/inertia.
    """

    position_kp: ArrayLike
    velocity_kp: ArrayLike           # acts as Kd (velocity target hard-wired 0)
    position_ki: ArrayLike = 0.0
    torque_limit: ArrayLike = 2.0    # Nm; firmware clamps integrator AND output to +-this
    torque_filter_alpha: float = 0.1454  # EMA alpha (~50 Hz at 2 kHz)

    integ: ArrayLike = 0.0           # position integrator state (torque units)
    tau_f: ArrayLike = 0.0           # EMA filter state

    def reset(self, shape: tuple[int, ...] | None = None) -> None:
        """Zero the integrator + filter state, optionally shaped to [N]."""
        z = 0.0 if shape is None else np.zeros(shape)
        self.integ = z
        self.tau_f = 0.0 if shape is None else np.zeros(shape)

    def step(self, pos: ArrayLike, vel: ArrayLike, target: ArrayLike) -> ArrayLike:
        """Advance one control tick; return the clamped, filtered output torque.

        Mirrors the firmware exactly (HANDOFF §2): the integrator accumulates
        per-tick with NO dt term and is clamped to +-torque_limit; the summed
        torque_target is EMA-filtered, then clamped again to +-torque_limit.
        """
        tau_lim = self.torque_limit
        perr = target - pos
        verr = -vel                              # velocity target is hard-wired 0
        self.integ = np.clip(self.integ + self.position_ki * perr, -tau_lim, tau_lim)
        torque_target = self.position_kp * perr + self.velocity_kp * verr + self.integ
        self.tau_f = self.tau_f + self.torque_filter_alpha * (torque_target - self.tau_f)
        return np.clip(self.tau_f, -tau_lim, tau_lim)

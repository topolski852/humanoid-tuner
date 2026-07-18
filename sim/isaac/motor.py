"""MotorSpec — the per-motor physical parameters an Isaac actuator model needs.

Two levels of parameter, kept distinct on purpose:

  * CONTROL params (Kt, gear_ratio, torque_limit, current_limit, torque_filter_
    alpha) — what the firmware config holds and the control law uses. Source of
    truth: humanoid-studio/HANDOFF.md §4/§5. `torque_constant` is the MEASURED
    value the firmware actually uses (NOT KV-derived).
  * PLANT params (rotor inertia -> reflected inertia -> Isaac `armature`, phase
    R/L, friction) — what a higher-fidelity plant needs but the config lacks.
    Sourced from datasheets / bench characterization; provenance noted per field.

Phase-0 note: with an ideal current loop (see sim/control_law.py) Kt, gear_ratio,
current_limit, R and L are all INACTIVE for a no-load small step — the response
is governed by {kp,kd,ki, torque_limit, torque_filter_alpha, inertia, friction}.
They are carried here so Phase-1 (load-driven current saturation, latency, an
actuator-net) can switch them on without re-plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MotorSpec:
    name: str

    # --- control-loop params (firmware config; HANDOFF §4/§5) ------------------
    torque_constant: float       # Nm/A, MEASURED (firmware value, not KV-derived)
    gear_ratio: float            # magnitude (sign is a joint-frame convention)
    torque_limit: float = 2.0    # Nm, config default
    current_limit: float = 20.0  # A, config default
    velocity_limit: float = 20.0  # rad/s, config default
    torque_filter_alpha: float = 0.1454

    # --- plant params (datasheet / bench characterization) --------------------
    rotor_inertia: float = 0.0   # kg·m², motor-side rotor moment of inertia
    phase_resistance: float | None = None   # ohm  (Phase-1 electrical model only)
    phase_inductance: float | None = None   # H    (Phase-1 electrical model only)
    mass: float | None = None    # kg, whole-motor mass (informational)
    no_load_current: float | None = None    # A, I0 (rough Coulomb-friction proxy)
    pole_pairs: int | None = None

    # --- bench-MEASURED plant params (real2sim; None until characterized) -----
    # From position-mode system-ID on the bench (bench/fit_plant.py): effective
    # OUTPUT-shaft inertia + friction of the real motor, no load.
    measured_inertia: float | None = None      # kg·m², effective output inertia
    coulomb_friction: float | None = None      # N·m, Coulomb (dominant on this motor)
    viscous_damping: float | None = None        # N·m·s/rad
    latency_s: float | None = None              # command->response latency

    # --- provenance (free-text, so the numbers are never silently trusted) ----
    notes: str = ""

    @property
    def reflected_inertia(self) -> float:
        """Rotor inertia felt at the OUTPUT shaft = J_rotor * gear_ratio².

        This is the number that goes into Isaac's `armature`. For the M6C12 it
        is ~0.0224 kg·m² — about 28× the toy numpy sim's whole-plant 8e-4, i.e.
        the no-load plant is dominated by reflected rotor inertia the point-mass
        sim never modelled. That gap is the concrete reason to bring the real
        motor into a rigid-body sim.
        """
        return self.rotor_inertia * self.gear_ratio ** 2

    def max_output_torque_from_current_limit(self) -> float:
        """Output torque at which the FOC current limit binds = I_lim * Kt * gear.

        For the M6C12: 20 A * 0.08958 * 15 ≈ 26.9 Nm — far above torque_limit
        (2.0 Nm), so the current limit is inactive for Phase-0 no-load steps.
        """
        return self.current_limit * self.torque_constant * self.gear_ratio


# MAD M6C12, 150 KV — the "big leg" motor (hip roll/yaw/pitch + knee).
# Kt, gear, limits, alpha: humanoid-studio/HANDOFF.md §4/§5 (firmware config).
# Rotor inertia, R, L: MEASURED in Berkeley Humanoid Lite motor characterization
#   (same Recoil firmware lineage) — rotor 68 mm dia, 86 g shell, cylindrical-
#   shell approximation -> 9.942e-5 kg·m². Char-test Kt there was 0.0919 Nm/A;
#   we keep the firmware's 0.08958 in the control law and note the delta.
# no_load_current, friction: UNMEASURED (datasheet lists I0 "TBA"). Left absent
#   rather than guessed; Phase-0 no-load uses coulomb=0.
M6C12_150KV = MotorSpec(
    name="MAD_M6C12_150KV",
    torque_constant=0.08958,
    gear_ratio=15.0,
    torque_limit=2.0,
    current_limit=20.0,
    velocity_limit=20.0,
    torque_filter_alpha=0.1454,
    rotor_inertia=9.942e-5,
    phase_resistance=0.1886,
    phase_inductance=0.0325e-3,
    mass=0.260,
    no_load_current=None,          # datasheet: "TBA"; superseded by measured friction below
    pole_pairs=11,                 # 24N22P (datasheet) — NOT 14; firmware config stores 14
    # BENCH-MEASURED (right_hip_yaw ESC, device 4, free shaft) — bench/fitted_plant.json.
    # Effective output inertia 0.0274 (vs 0.0224 datasheet estimate); Coulomb friction
    # 0.30 N·m (step-fit) / 0.33 N·m (ramp i_q) — two independent methods agree.
    measured_inertia=0.02743,
    coulomb_friction=0.302,
    viscous_damping=0.023,
    latency_s=0.0072,
    notes=(
        "Kt firmware-measured 0.08958 (char-test 0.0919). Rotor inertia & R/L from "
        "Berkeley Humanoid Lite characterization. Datasheet pole count is 22 (11 pp). "
        "Bench system-ID (fit RMS 10.8 mrad over 18 steps): effective output inertia "
        "0.0274 kg·m², Coulomb 0.30 N·m, viscous 0.023, latency 7.2 ms — the toy sim "
        "had coulomb=0, so friction is entirely new real physics."
    ),
)

# MAD 5010, 200 KV — the ankle/arm motor. Kept for Phase 2 (multi-motor).
# Kt/gear/limits: HANDOFF §4/§5. Rotor inertia here is scaled from the Berkeley
# 5010 char (110 KV variant: 3.301e-5 kg·m², 53 mm/47 g rotor) — the 200 KV
# rotor geometry is nominally the same stator, so inertia is ~unchanged; flagged
# as ESTIMATED until bench-measured.
MAD5010_200KV = MotorSpec(
    name="MAD_5010_200KV",
    torque_constant=0.06588,
    gear_ratio=15.0,
    torque_limit=2.0,
    current_limit=20.0,
    velocity_limit=20.0,
    torque_filter_alpha=0.1454,
    rotor_inertia=3.301e-5,        # ESTIMATED from Berkeley 5010 (110KV) char; measured below
    phase_resistance=0.176,        # datasheet (madcomponents.co)
    phase_inductance=None,         # unpublished for the 200KV variant
    mass=0.165,
    no_load_current=0.5,           # datasheet: 0.5 A @ 16 V
    pole_pairs=14,                 # 24N28P (datasheet) — matches firmware config
    # BENCH-MEASURED (right_shoulder_roll ESC, device 4, can_right_arm, free shaft) —
    # bench/runs/char5010_latest.json / bench/fitted_plant.json. Free 200KV motor is
    # light + fast, so it was characterized GENTLY (velocity-capped 3 rad/s, small
    # ±0.1–0.15 steps) to avoid the encoder-aliasing over-speed the M6 never hit.
    # Effective output inertia ~0.010–0.014 (rotor ~4.4–6.3e-5, vs 3.3e-5 estimate).
    # Coulomb 0.017 N·m motor-frame (0.25 output) from 6 constant-velocity ramp i_q
    # readings, both directions — the reliable friction; gentle steps don't excite it.
    measured_inertia=0.0120,
    coulomb_friction=0.25,
    viscous_damping=0.005,
    latency_s=0.012,
    notes=(
        "Bench system-ID on the FREE 200KV motor (no gearbox), can_right_arm dev 4, "
        "fit RMS 2.9 mrad over 8 gentle steps: effective output inertia ~0.012 kg·m² "
        "(rotor ~5e-5, ~1.5x the 3.3e-5 estimate), Coulomb 0.017 N·m motor / 0.25 output "
        "(6 ramps agree ±10%), viscous ~0. Inertia has ~40% spread because the light "
        "motor must be driven gently (velocity-capped) — steps are partly rate-limited. "
        "Kt firmware 0.06588. R datasheet; L unpublished. NOTE: fit_plant.py hardcodes "
        "the M6 Kt for its ramp cross-check, so its printed ramp torques run ~1.36x high. "
        "GEARED cross-check (15:1 attached, bench/runs/char5010_geared.json): output "
        "inertia ~0.012 (unchanged), Coulomb 0.019 N·m motor / 0.29 output (~15% higher) "
        "— like the M6, the gearbox only adds friction, so bare-derived gains carry over."
    ),
)

MOTORS = {m.name: m for m in (M6C12_150KV, MAD5010_200KV)}

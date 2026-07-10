"""Bridge from the tuner to real hardware via the Humanoid-Studio daemon.

SKELETON — interface only. See docs/DESIGN.md §3 for the verified daemon protocol.

The tuner never opens a CAN socket. It speaks UDP/JSON to the Studio C++ daemon:
  - 9001  command req/resp   (WRITE_GAINS, APPLY_CONFIG, SET_MODE, SET_POSITION, ...)
  - 9000  telemetry push     (position/velocity @100 Hz; current/torque slow-polled)
  - 9002  ESTOP only

Prefer reusing humanoid-studio/backend/humanoid/daemon_client.py (DaemonClient) over
re-implementing the wire protocol; this module wraps it in tuner-shaped terms.
"""

from __future__ import annotations

from dataclasses import dataclass


# Gains the firmware exposes for the POSITION loop (+ inner current loop). See DESIGN §3.
# NOTE: velocity_kp is the *Kd* term in POSITION mode (velocity target is hard-wired 0).
@dataclass
class Gains:
    position_kp: float
    velocity_kp: float  # == Kd
    position_ki: float = 0.0
    torque_limit: float | None = None
    current_kp: float | None = None
    current_ki: float | None = None


@dataclass
class Sample:
    """One telemetry sample for a single joint (display-frame, output shaft)."""
    t: float
    position: float           # rad, 100 Hz
    velocity: float           # rad/s, 100 Hz
    current: float | None     # A (i_q), ~3.3 Hz slow-poll unless polled harder
    torque: float | None      # Nm, ~3.3 Hz slow-poll


class DaemonBridge:
    """Tuner-facing view of one joint on the real robot, through the Studio daemon."""

    def __init__(self, joint_name: str, host: str = "127.0.0.1",
                 cmd_port: int = 9001, tel_port: int = 9000, estop_port: int = 9002):
        self.joint_name = joint_name
        self.host = host
        self.cmd_port = cmd_port
        self.tel_port = tel_port
        self.estop_port = estop_port

    # --- lifecycle -------------------------------------------------------------
    def connect(self) -> None:
        """Wake+configure the joint (daemon APPLY_ALL_CONFIGS), start telemetry cache."""
        raise NotImplementedError

    def enable(self) -> None:
        """SET_MODE POSITION."""
        raise NotImplementedError

    def idle(self) -> None:
        """SET_MODE IDLE — the clean stop."""
        raise NotImplementedError

    def estop(self) -> None:
        """Priority ESTOP on 9002 — drives the joint IDLE at the daemon's next tick."""
        raise NotImplementedError

    # --- the tuning loop primitives -------------------------------------------
    def write_gains(self, gains: Gains) -> None:
        """WRITE_GAINS / APPLY_CONFIG. Must pass through the safety envelope first."""
        raise NotImplementedError

    def command_position(self, position_rad: float) -> None:
        """SET_POSITION (display-frame rad) — used to drive test excitations."""
        raise NotImplementedError

    def read_sample(self) -> Sample:
        """Latest telemetry sample from the cache (no round-trip)."""
        raise NotImplementedError

    def store_to_flash(self) -> None:
        """Persist accepted gains RAM->Flash. Only after a tune is accepted."""
        raise NotImplementedError

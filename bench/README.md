# bench/

Host-side bridge between the tuner policy and **real hardware**, via the Humanoid-Studio
C++ daemon (UDP/JSON — never CAN, never firmware).

Responsibilities:
- **drive** repeatable test excitations (step / chirp / trajectory replay) on one joint
- **read** telemetry (`position`/`velocity` @ 100 Hz; `current`/`torque` slow-polled)
- **extract** response features per window (see `docs/DESIGN.md §2, §4`)
- **write** gains via the daemon (`WRITE_GAINS` / `APPLY_CONFIG`)
- **enforce** the safety envelope (gain clamps, rate limits, runaway detector)

Same feature-extraction code is shared with `sim/` so the policy sees an identical
observation in sim and on hardware.

`daemon_bridge.py` is a skeleton — the interface is real, the bodies are TODO. It can
reuse `humanoid-studio/backend/humanoid/daemon_client.py` (`DaemonClient`) instead of
re-implementing the UDP protocol.

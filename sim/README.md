# sim/

Single-actuator simulation the tuner policy trains against.

Replicates the **firmware control law** (2 kHz position loop + 10 kHz FOC current loop,
see `docs/DESIGN.md §3`) so a policy trained here transfers to the real ESC. Domain
randomization over load / inertia / friction / backlash / latency / sensor noise is what
turns "a tuner for this motor" into "a policy that understands motors" (Phase 1+).

Substrate decision is open (`docs/DESIGN.md §6`): reuse Isaac Lab + RSL-RL from
`humanoid-policy`, or a lighter standalone ODE model of the single actuator. The control
law is simple enough that a lightweight sim may train faster — revisit once the reward is
proven in Phase 0.

Feature extraction is **shared with `bench/`** so sim and hardware produce an identical
observation vector.

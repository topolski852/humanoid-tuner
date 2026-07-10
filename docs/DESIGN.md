# humanoid-tuner — Design

The reasoning behind the project, the interface it builds on, and the phased plan.
This is the source of truth for *why*; code comments cover *how*.

---

## 1. The core idea

A learned policy that adjusts a motor's control gains as it runs. Two products, one
network:

- **Offline auto-tuner** — robot runs a test move, policy reads the response, outputs
  recommended gains. Replaces the heuristic tuner in Studio. Safe, demoable, shippable.
- **Online adaptive controller** — policy stays in the loop, adjusting gains live as
  load/inertia change. More novel, riskier, needs a safety envelope.

They differ by **one thing**: whether the outer loop is allowed to keep running or is
frozen once the response converges. Same observations, same reward, same net. Build the
offline version first; unfreeze it into the online version later.

### Why learned beats the current heuristic tuner

Classical auto-tuning (Ziegler-Nichols, relay-feedback, gradient methods) assumes a
**linear time-invariant** plant. A real joint is none of that: direction-dependent
friction, backlash dead-zones, pose-dependent inertia, cogging, current limits, all
time-varying as the robot moves. A learned model can represent what a line-fit can't.

### The real unlock: state-dependent gains

Hand-tuning forces **one** fixed gain set — a compromise across loaded stance, free
swing, pre-contact, etc. A policy can output gains as a continuous function of phase and
load. That's a capability no human tuning can reach, and it's the differentiator.

---

## 2. Control architecture — why it's a *tuner*, not a *controller*

The hardware can't run a neural net faster than a PID, so the fast PID **stays on the
MCU** and the policy is a **slow outer loop** that supervises it. This is textbook
cascaded / hierarchical control with outer-loop gain scheduling.

Three principles fall out of it:

1. **Timescale separation is a hard rule.** The outer loop must be meaningfully slower
   than the inner loop's settling time (≈5–10×) or the two loops fight and destabilize.
   Inner PID settles in tens of ms → policy adjusts gains every few hundred ms. This is
   also why the policy is cheap: a small net at 10–50 Hz is trivial on any target.

2. **The policy sees *features*, not raw samples.** Something computes response features
   over each window (tracking-error RMS/peak, overshoot, oscillation amplitude + dominant
   frequency, settling time, steady-state error, effort/current) and the policy consumes
   *those*. Keeps the net tiny, transfers sim→real far better than raw waveforms, and the
   same oscillation detector doubles as the safety monitor.

3. **Reward is evaluated over a window *after* each action.** A gain change plays out over
   the next few hundred ms, so reward measures "did this change improve the features vs.
   the previous window." Getting this window right is most of Phase 0.

### Safety envelope (build in Phase 0, before it protects real hardware)

Hard min/max gain clamps · rate-limit on gain changes · runaway detector (rolling
variance / cheap FFT on the error → if ringing, clamp to a known-safe conservative set).
Real Safe-Torque-Off is a *hardware* feature of a future ESC, not something this repo can
provide; treat the software clamps as risk-reduction, not a safety rating.

### On replacing the PID entirely

Considered and rejected for now. Direct torque-control policies are more expressive but
lose the fast inner loop, sim-to-real robustness, safe fallback, and (decisively) aren't
possible on hardware that only accepts gain/target writes over the bus. A PID whose gains
a net sets every tick is *already* a learned nonlinear controller with good structure —
we get most of the expressiveness with none of the costs. Residual torque-on-top is the
escape hatch if friction/cogging compensation ever demands it.

---

## 3. The interface this builds on (verified from Humanoid-Studio)

The tuner talks **UDP/JSON to the Studio C++ daemon** — never CAN, never firmware.
Sources: `humanoid-studio/DAEMON_SPEC.md`, `humanoid-studio/HANDOFF.md`.

- **Ports (127.0.0.1):** 9001 command req/resp · 9000 telemetry push · 9002 ESTOP.
- **Reuse the client:** `humanoid-studio/backend/humanoid/daemon_client.py`
  (`DaemonClient`) — reconnect, telemetry cache, per-joint proxy. Or speak the JSON.

### Gains we can write (via `WRITE_GAINS` / `APPLY_CONFIG`, SDO under the hood)

| gain | param | role |
|---|---|---|
| `position_kp` | 0x020 | position P |
| `position_ki` | 0x024 | position I |
| `velocity_kp` | 0x028 | **acts as Kd** in POSITION mode (velocity target is hard-wired 0) |
| `current_kp`  | 0x078 | inner FOC current-loop P |
| `current_ki`  | 0x07C | inner FOC current-loop I |
| `torque_limit`| 0x030 | output torque clamp |

`WRITE_GAINS` is a fast ~4-SDO retune; `APPLY_CONFIG` writes the broader set. Persist with
`STORE_TO_FLASH` only when a tune is accepted.

### Telemetry — mind the rates

- **Fast (100 Hz, PDO4):** `position`, `velocity` — output-shaft rad, post-gearbox.
- **Slow (~3.3 Hz, round-robin SDO):** `torque`, `current` (i_q), `bus_voltage`.

**Consequence:** position-loop tuning features (tracking error, overshoot, settling) live
in the 100 Hz position/velocity stream — plenty. Current/"effort" features are coarse
unless the bench issues its own faster `GENERIC_SDO_READ` of i_q (0x0C0). 100 Hz is above
the mechanical bandwidth of the 15:1 output shaft, so it captures overshoot/settling;
electrical-rate ringing is not observable and shouldn't be a tuning target here.

### The control law to replicate in sim (HANDOFF.md §2, verified from firmware)

```
position_error   = clamp(position_target, lo, hi) - position_measured
velocity_error   = 0 - velocity_measured               # target velocity is always 0
position_integ   = clamp(position_integ + position_ki*position_error, -tau_lim, +tau_lim)
torque_target    = position_kp*position_error
                 + velocity_kp*velocity_error           # velocity_kp == Kd
                 + position_integ
                 + torque_ff                            # usually 0
torque_setpoint  = clamp(ema(torque_target, torque_filter_alpha), -tau_lim, +tau_lim)
i_q_target       = torque_setpoint / torque_constant / gear_ratio   # signed gear_ratio
```
Position loop @ **2 kHz**, inner FOC current loop @ **10 kHz**. `position/velocity_measured`
are already gearbox-divided (output shaft). Replicating this exactly is what makes the
sim-trained policy transfer.

### Bench setup

Run the daemon with a **1-joint config** (the `joints` map in a `humanoid_lite`-style
JSON, pointed at the bench motor's bus + device_id). The tuner then drives that one joint
over UDP. Multi-node buses matter later; Phase 0 is one motor.

### First hardware target

STM32G431 (B-G431B-ESC1) running custom "Recoil" firmware, **v3.2.0**
(github.com/topolski852/humanoid-esc-firmware). Motors on the humanoid: 150 KV `MAD_M6C12`
(Kt≈0.0896) on the big leg joints, 200 KV `MAD_5010` (Kt≈0.0659) on ankles/arms — a ready
made "multiple motor types" set for Phase 2.

---

## 4. Observation / action / reward (starting point — Phase 0)

**Action:** gain vector (or bounded delta). Start with `{position_kp, velocity_kp(Kd),
position_ki}`; add `torque_limit` and current-loop gains later.

**Observation:** recent history of response features + current gains + (online) recent
action history. Features per window:
`err_rms, err_peak, overshoot, settle_time, ss_error, osc_amplitude, osc_freq, effort`.

**Reward (over the post-action window):** weighted sum of
`-err_rms, -overshoot, -settle_time, -oscillation, -ss_error, -effort`.
**Expose the weights as a conditioning input** ("responsiveness ↔ compliance ↔ efficiency"
slider) so the tuner tunes *to a spec* instead of one-size-fits-all — near-free
architecturally, killer Studio UX.

**Test excitation:** repeatable step / chirp / trajectory replays on the bench so a gain
change is measured against identical conditions.

---

## 5. Phased plan

| phase | scope | success |
|---|---|---|
| **0** | one motor, no load, sim then bench | policy tunes a single joint to a step-response spec; reward validated |
| **1** | randomize load / inertia / friction / backlash | policy infers load and adapts (RMA-style history encoder → latent → policy) |
| **2** | multiple motor types / weights | generalizes across the 150 KV / 200 KV set and beyond |
| **3** | whole-joint / whole-robot; Studio integration | replaces the heuristic tuner in the app |
| **4** | *(north star)* visual self-observation | **decoupled**; tuner ships on proprioception alone, vision is an added obs source later |

### Sim-to-real bridge

Train in sim with domain randomization over load/inertia/friction/backlash/latency/noise,
using the §3 control law. Deploy with online *adaptation* (RMA latent), not online
learning-from-scratch — the latter is impractical on real hardware; adaptation of a
sim-trained policy is very practical. Optionally fit an actuator-net from bench data to
tighten the sim.

---

## 6. Open questions

- **Bench load rig:** cheapest path to programmable load is a second motor as a
  dynamometer on the same shaft (command arbitrary torque/inertia profiles, sweep the load
  space overnight). Confirm feasibility / parts.
- **i_q polling rate:** is faster `GENERIC_SDO_READ` of 0x0C0 worth it for effort features,
  or is position-only sufficient for Phase 0/1? (Probably position-only to start.)
- **Sim substrate:** ~~reuse Isaac Lab, or a lighter single-actuator ODE model?~~
  **RESOLVED for Phase 0: lightweight numpy** (`sim/actuator.py`, replicates the §3
  control law). Runs in seconds, no Isaac Lab dependency. Revisit for a richer plant
  in Phase 1+ if the numpy model proves too coarse.
- **Studio integration surface:** does the tuner run as a daemon UDP client the app spawns,
  or ship as a library the backend imports? (Mirror how `humanoid-control` plugs in.)
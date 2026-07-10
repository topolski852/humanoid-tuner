# humanoid-tuner

**AI-assisted motor gain tuner.** A learned policy that tunes the control gains of a
real motor joint — replacing hand-tuning (and the math/heuristic tuner in
Humanoid-Studio) with something that *learned* how joints behave.

Trained in sim against the real firmware's control law, deployed against real hardware
through the existing Studio CAN daemon. Designed to be **motor-general** — the Berkeley/
Recoil ESC on the humanoid is its first hardware backend, not its identity.

> Status: **scaffold.** See [docs/DESIGN.md](docs/DESIGN.md) for the full plan. Nothing
> here talks to hardware yet.

## Why this exists

PID/gain tuning is the single hardest, most tedious part of bringing up a joint. A human
tunes **one** fixed gain set that's a compromise across every regime the joint sees. A
learned policy has no such limit — it can produce **state-dependent gains** (stiff in
stance, compliant in swing, damped before contact) that no human tunes by hand. The pitch
isn't "tune faster," it's "produce a controller that's better than any fixed tuning."

## Architecture in one picture

```
  ┌─────────────────────────────────────────────┐
  │  TUNER POLICY  (this repo, host-side)         │   slow outer loop (10–50 Hz)
  │  observes response features → outputs gains    │
  └───────────────┬───────────────────────▲───────┘
     WRITE_GAINS  │                        │  telemetry (pos/vel 100 Hz)
     (SDO)        │  UDP/JSON              │
  ┌───────────────▼────────────────────────┴───────┐
  │  Humanoid-Studio C++ daemon (owns CAN)          │
  └───────────────┬─────────────────────────────────┘
                  │ Recoil CAN protocol
  ┌───────────────▼─────────────────────────────────┐
  │  ESC (STM32G431)  — 2 kHz PID + 10 kHz FOC       │   fast inner loop (on the MCU)
  └──────────────────────────────────────────────────┘
```

The intelligent, hard-to-tune part lives here on the host. The fast, safety-critical
stabilization stays on the MCU. The tuner **never touches firmware or CAN** — it speaks
UDP to the daemon.

## Layout

| dir | what |
|---|---|
| `sim/`    | single-actuator model (replicates the firmware control law) + RL training env |
| `bench/`  | host-side bridge to the Studio daemon: telemetry → response features → gain writes |
| `policy/` | RL training code + trained artifacts |
| `deploy/` | runtime that runs the trained tuner on real hardware / plugs into Studio (later) |
| `docs/`   | [DESIGN.md](docs/DESIGN.md) — architecture, phased plan, interface facts |

## Phases

0. **One motor, no load** — get the reward right. Prove a policy can tune a single joint
   to a step-response spec, in sim then on the bench.
1. **Randomized load/inertia/friction** — the policy *infers* the load and adapts (RMA-style).
2. **Multiple motor types / weights** — generalization across hardware.
3. **Whole-joint / whole-robot**, integrated into Studio.
4. *(north star)* visual self-observation — kept strictly decoupled; the tuner never depends on it.

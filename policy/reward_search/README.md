# reward_search — Eureka-style reward design

Automates the slowest part of the tuner: getting the reward right. Instead of you
hand-writing a reward, training, watching it do something dumb, and guessing again,
**Claude writes candidate reward functions, the loop grades them in sim, and feeds
the results back to Claude to improve.** (Inspired by NVIDIA's *Eureka* / *DrEureka*.)

## The loop

```
        ┌────────────────────────────────────────────────────────┐
        │  propose.py  — Claude writes N reward functions          │
        │               (given prior candidates + their scores)    │
        └───────────────┬────────────────────────────────────────┘
                        │ reward code
        ┌───────────────▼────────────────────────────────────────┐
        │  optimize.py — search GAINS that maximize that reward    │  (inner)
        │                against the single-actuator sim           │
        └───────────────┬────────────────────────────────────────┘
                        │ optimized response
        ┌───────────────▼────────────────────────────────────────┐
        │  sim/metrics.py — GROUND-TRUTH fitness of the response   │  (the judge)
        │   settle time · overshoot · ss-error · rms · oscillation │
        └───────────────┬────────────────────────────────────────┘
                        │ fitness + metrics
                        └──────────────► back to propose.py (reflect) ──┐
                        ▲                                                │
                        └────────────────────────────────────────────── ┘
```

The key idea (from Eureka): the reward Claude writes is **not** the objective. The
objective (`sim/metrics.py`) is fixed and human-specified. A reward is *good* if
optimizing it lands on gains that score well on that objective — so the loop can
rank rewards automatically, no human in the loop.

## Run it

```bash
# no API key — grades the built-in seed rewards so you can see the pipeline work:
python -m policy.reward_search.loop --dry-run

# the real loop — Claude proposes each generation (needs Anthropic API access):
python -m policy.reward_search.loop --iterations 5 --candidates 6
```

Best result → `reward_search_best.json` (reward code + optimized gains + metrics).

Auth uses the standard Anthropic SDK resolution (`ANTHROPIC_API_KEY`, or an
`ant auth login` profile). No key? Use `--dry-run`.

## Two Phase-0 stand-ins (swap these to scale up)

Both are documented at the top of their files; the loop above them doesn't change.

1. **`sim/actuator.py` — lightweight numpy sim.** Replicates the firmware control
   law. Phase 1: add load/inertia/friction/backlash randomization here. Later: swap
   for Isaac Lab if you want a richer plant.
2. **`optimize.py` — gradient-free gains search.** Stands in for the RL
   gain-scheduling policy. It finds one fixed gain set per reward — enough to *judge*
   a reward, but it can't produce the state-dependent gains the RL policy will. Swap
   it for the trainer behind the same `optimize_gains(...)` signature.

## Path to hardware

The same reward that wins in sim is the reward you train the real policy against.
Point the ground-truth metrics at a **bench** response (via `bench/daemon_bridge.py`
→ the Studio daemon) instead of the sim, and the identical loop tunes against real
hardware. Sim gets you the reward cheaply; the bench validates it.

## ⚠️ Safety

`rewards.py:compile_reward()` runs model-generated code via `exec()` in a
restricted namespace (no imports, limited builtins). This is **not** a real
sandbox — run only with rewards from your own Claude calls, on hardware you control.

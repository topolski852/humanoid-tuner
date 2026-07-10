# sim/isaac — Isaac Lab single-joint substrate

A higher-fidelity plant for the tuner, **behind the same `Response` interface the
numpy sim exposes** (`sim/actuator.py`), so `policy/reward_search/` can target
either. The numpy sim stays the fast reward-search substrate (seconds/eval);
Isaac is the validation + eventual RL-training substrate.

## The one design rule: share the controller, swap only the plant

Both substrates run the **exact same** firmware control law —
[`sim/control_law.py`](../control_law.py), `FirmwarePositionController`
(HANDOFF §2, verified from firmware). The numpy sim integrates a point-mass;
Isaac integrates a PhysX rigid body. Because the controller is identical, the
[agreement test](agreement_test.py) measures *only* the plant/integrator
difference — not two drifting re-implementations.

The joint is driven in **pure torque mode** (`ImplicitActuatorCfg` with
`stiffness=damping=0`); each 2 kHz tick we apply the net joint effort
`tau_ctrl − damping·ω − coulomb·sign(ω)` — exactly the torque the numpy plant
integrates. Isaac's `armature` carries the joint-space inertia (this is where
reflected rotor inertia lives).

## Ideal current loop (why Kt/gear/R/L are inactive in Phase 0)

The firmware computes `i_q = tau / Kt / gear`; the FOC loop makes
`i_q_measured = i_q_target`, so `tau_produced = i_q·Kt·gear = tau` — **Kt and
gear cancel.** They (and the current limit, R, L, pole-pairs) re-enter only when
the current limit binds (20 A → ~27 Nm at the M6C12 output, far above the 2.0 Nm
torque_limit) or the electrical dynamics are modelled — neither observable at the
100 Hz output sampling this project tunes against. So Phase-0 no-load response is
governed entirely by `{kp, kd, ki, torque_limit, torque_filter_alpha, inertia,
damping, friction}`, **identical across both motors**. The motor-specific params
live in [`motor.py`](motor.py) `MotorSpec` for when Phase 1 turns them on.

## The reflected-inertia finding

The numpy sim's default `Plant.inertia = 8e-4 kg·m²`. The M6C12's **reflected
rotor inertia alone is 0.0224 kg·m²** (J_rotor 9.942e-5 × 15²) — ~28× larger.
The real no-load plant is dominated by reflected rotor inertia the point-mass sim
never modelled, so reward-search gains tuned on the toy plant are far off for the
real joint. That gap → `armature` in Isaac, and it is the concrete reason to
bring the real motor into a rigid-body sim.

## Files

| file | what |
|---|---|
| `paths.py` | pxr-free constants (asset path, joint name) |
| `build_asset.py` | authors `assets/single_joint.usd` via `pxr` (run once) |
| `motor.py` | `MotorSpec` + measured `M6C12_150KV` / `MAD5010_200KV` params |
| `substrate.py` | `IsaacSingleJoint` (vectorised N-env rollout) + JSON-job CLI |
| `agreement_test.py` | Isaac vs numpy step-response comparison |

## Environment — runs under humanoid-policy's venv, not the tuner's

Isaac Lab 3.0.0b2 needs Python 3.12; the tuner's own venv is 3.11/numpy-only.
Use the policy interpreter (numpy is available there too, so the agreement test
imports both sims in one process):

```bash
PY=/home/nse/humanoid/humanoid-policy/.venv/bin/python
cd /home/nse/humanoid/humanoid-tuner

# 1. author the single-joint USD (once; no Isaac boot needed)
$PY -m sim.isaac.build_asset

# 2. run the agreement test (boots Isaac headless, ~10s + rollout)
OMNI_KIT_ACCEPT_EULA=YES $PY -m sim.isaac.agreement_test

# or run the substrate as a standalone JSON job (one plant, many gains -> N envs)
OMNI_KIT_ACCEPT_EULA=YES $PY -m sim.isaac.substrate --job job.json --out out.json
```

**Gotcha:** never `import pxr` before `AppLauncher` boots — it corrupts the USD
bindings. `substrate.py` imports `paths.py` (pxr-free), never `build_asset.py`.

## Wiring the Eureka loop to Isaac (next)

`substrate.py` runs N candidate gain sets as N parallel envs in one rollout —
so `policy/reward_search/optimize.py`'s gain search can score a whole population
per PhysX rollout. Point `optimize_gains` at an Isaac-backed `simulate_step`
(in-process under this venv, or via the JSON-job subprocess from the numpy venv)
and the existing propose→optimize→grade→reflect loop runs against Isaac unchanged.

# Eureka-Style Reward Design — What It Is and Why It's Worth It

*A brief for the team. Goal: agree on whether to invest in an automated
reward-design loop, and at what scope.*

> **This doc is in two parts:** **Part 1 · Overview** is the read-it-in-the-meeting
> summary. **Part 2 · Technical Details** is the full architecture and integration
> plan for whoever builds or scrutinizes it.

---

# Part 1 · Overview

## TL;DR

Reward design is the slowest, most manual part of every RL project we run — it's
where the human time and the trial-and-error go. **Eureka-style reward design
automates that loop: an LLM (Claude) writes candidate reward functions, we train
against each, an objective score grades the result, and the scores feed back to the
LLM to improve — repeatedly, without a human in the loop.** It was introduced by
NVIDIA (*Eureka*, and its sim-to-real follow-up *DrEureka*) and shown to produce
reward functions that beat human-expert-written ones on many robotics tasks.

**We already have a working reference implementation** in this repo
(`policy/reward_search/`) for the motor-tuning problem — it runs end-to-end today.
This doc explains the idea, what it would take to apply it to our locomotion
training, and what it would cost.

## The problem it solves

In RL, the **reward function** is how you tell the policy what "good" means. Getting
it right is famously hard:

- A reward that looks reasonable often produces a policy that games it in a way you
  didn't anticipate (the classic "reward hacking").
- The loop to fix it is slow and manual: *tweak the reward → launch a training run →
  wait → watch the policy do something dumb → guess at a fix → repeat.*
- Each iteration can take minutes to hours, and it depends entirely on one person's
  intuition about how a dozen weighted terms will interact.

This is exactly where we spend disproportionate time on standup, walking, and now
motor tuning. It's a bottleneck made of human guesswork.

## What Eureka is

Eureka replaces the human in that loop with an LLM that **reads the task, writes
reward functions, learns from how they perform, and rewrites them** — a closed,
automated search over reward designs.

```
        ┌──────────────────────────────────────────────────────────┐
        │  1. PROPOSE   Claude writes N candidate reward functions   │
        │     (given the task + how previous candidates scored)      │
        └───────────────┬──────────────────────────────────────────┘
                        │ reward code
        ┌───────────────▼──────────────────────────────────────────┐
        │  2. TRAIN     train a policy against each candidate reward │
        └───────────────┬──────────────────────────────────────────┘
                        │ trained policy
        ┌───────────────▼──────────────────────────────────────────┐
        │  3. GRADE     score the result on a fixed OBJECTIVE metric │
        │     (did it stand? how fast? how stable? how efficient?)   │
        └───────────────┬──────────────────────────────────────────┘
                        │ score + breakdown
                        └────────► back to step 1 (Claude reflects) ─┐
                        ▲                                            │
                        └──────────────────────────────────────────-┘
```

### The one insight that makes it work

The reward the LLM writes is **not** the thing we're optimizing. There are two
separate levels:

- The **reward** is what the *policy* is trained to maximize (the shaped signal, with
  all its weighted terms).
- The **objective** ("fitness") is what *we* actually care about — a fixed,
  human-defined score like "time to stand up" or "velocity tracking error + fall
  rate." The LLM never gets to touch this.

A reward is judged *good* if training against it produces a policy that scores well
on the objective. That separation is the whole trick: it lets the loop rank reward
functions automatically and honestly, with no human eyeballing training curves. It
also structurally resists reward hacking — a reward the policy games will score badly
on the true objective, so the loop discards it.

## Why it would benefit us

Two concrete applications, one already built, one proposed.

**1. Motor gain tuning — *working reference implementation today.*** The
`humanoid-tuner` project (this repo) uses the Eureka loop to design the reward for
tuning a joint's PID gains. Claude proposes rewards; a fast single-actuator
simulation optimizes gains against each; step-response metrics grade the result and
feed back. It runs end-to-end in seconds per candidate — our proof the machinery
works and that we can operate it.

**2. Locomotion & standup reward design — *the bigger prize.*** Our main training
work is where reward design costs us the most human time. Eureka was originally built
for exactly this kind of Isaac-based locomotion task. The payoff:

- **Compress the reward-tuning cycle from days of human iteration to an overnight
  automated search** — the machine runs "tweak → train → evaluate → tweak" while we
  sleep.
- **Discover reward shapes we wouldn't think to try.** Eureka's headline result was
  finding rewards that outperformed expert hand-tuning, because it explores the space
  systematically instead of relying on one engineer's intuition.
- **Turn tribal knowledge into a repeatable process.** Reward design today lives in
  one person's head; this makes it a documented, rerunnable pipeline.
- **A genuinely differentiated capability** — "we use AI to design our robot's reward
  functions" is a real, defensible research story that few teams our size pursue.

## Cost and scope at a glance

The idea is proven; the honest cost is that **for locomotion, each reward candidate
means a full RL training run** (minutes to an hour on GPU), not a millisecond sim. A
search is dozens of training runs — roughly **~20 GPU-hours per search**. The biggest
mitigation is short "proxy" runs to rank candidates cheaply before committing to full
runs on the winners. (Full mechanics in Part 2.)

Two scope tiers, and we don't have to do both at once:

| Scope | What it does | Effort | Risk |
|---|---|---|---|
| **A — weight auto-tuner** *(recommended start)* | Claude tunes the **weights/scales** of our existing reward terms | ~2–4 focused days | Low — no generated code runs inside training |
| **B — reward synthesis** *(full Eureka)* | Claude writes **new reward-term code** over the env state | ~1–2 weeks on top of A | Higher — generated code, more integration |

## Where we are & the recommended path

- **Today:** a working Eureka loop for motor tuning in this repo
  (`policy/reward_search/`). Its propose → train → grade → reflect structure ports
  directly to the locomotion trainer — the only piece that changes is swapping the
  fast sim inner loop for an Isaac RL training run.
- **Recommended:** start with **scope A (weight-tuning) on short proxy runs** for the
  standup task. A few days, low-risk, and it de-risks the expensive part (RL as the
  inner loop) before we invest further. If proxy-run ranking tracks full-run quality,
  we've essentially got the whole capability; if not, we learned it cheaply.

## What we'd decide as a team

1. **Is it worth the GPU-hours?** — the inner loop is the real cost; are we willing to
   spend ~20 GPU-hours per reward search to save the human days?
2. **What's the objective metric for standup / walking?** — the fixed fitness score.
   This is the most important design decision and the one we'd need to agree on.
3. **Scope A or B first?** — recommendation is A (weight-tuning), then B if it proves
   out.

---

# Part 2 · Technical Details

This part is the full architecture and integration plan, aimed at the locomotion
trainer (`humanoid-policy`). The guiding principle: **the Eureka loop is an *outer
harness that wraps the trainer we already have* — it drives our existing Isaac Lab +
RSL-RL training, it does not rewrite it.** Almost everything is additive.

## Architecture — the components

Five pieces. Four are new code in their own package (e.g. `eureka/`); one is a small,
well-contained hook into the existing reward config.

| Component | What it does | Where it lives |
|---|---|---|
| **Generator** | Calls Claude to propose N candidate rewards, given the task and how prior candidates scored. Same code already working in `humanoid-tuner`'s `reward_search/propose.py`. | new — `eureka/propose.py` |
| **Injector** | Turns a candidate into something the trainer reads — overridden reward **weights** (scope A) or generated reward **term code** (scope B). | new — `eureka/inject.py` + one hook in `RewardsCfg` |
| **Runner** | Launches a real training run for a candidate: shells out to our existing `scripts/rsl_rl/` train entrypoint with the candidate's reward, a bounded iteration count (a short "proxy" run), and a fixed seed. | new — `eureka/run.py` (wraps existing train script) |
| **Evaluator** | After a run, loads the checkpoint from `logs/rsl_rl/<task>/…`, rolls out N episodes, and computes the fixed **objective/fitness** score (stand time, tracking error, fall rate, energy). | new — `eureka/evaluate.py` |
| **Orchestrator** | The outer loop: propose → run → evaluate → collect → reflect → repeat; keeps a ranked archive and writes the winner. Ported from `humanoid-tuner`'s `reward_search/loop.py`. | new — `eureka/search.py` |

Only the **Evaluator** and the **Injector's hook** are genuinely this-repo-specific
work. The Generator and Orchestrator are ports of code that already runs.

## One iteration, step by step

For a single candidate reward, on the standup task:

1. **Propose.** The Generator asks Claude for a candidate. Scope A: a set of
   reward-term weights. Scope B: Python for one or more `RewTerm` functions over the
   env state (`asset.data.*`, sensors, commands).
2. **Inject.** The Injector applies it. Scope A: override the weights in the run's
   `RewardsCfg`. Scope B: write the generated functions into a designated
   `mdp/rewards_generated.py` module that `RewardsCfg` already imports, and register
   the terms.
3. **Train (proxy).** The Runner launches `scripts/rsl_rl/train.py` for the standup
   task with that reward, capped at a small iteration budget (e.g. a few hundred
   iterations rather than a full run) on a fixed seed. This is a normal training run
   — nothing about the trainer changes; it's just parameterized and time-boxed.
4. **Evaluate.** The Evaluator loads the checkpoint the run wrote, rolls out a batch
   of episodes, and computes the **objective** score — the fixed, human-defined
   fitness (e.g. `did_stand·w1 − time_to_stand·w2 − instability·w3 − energy·w4`).
   Critically, this metric is *independent of the reward Claude wrote* — that's what
   makes the ranking honest and hack-resistant.
5. **Record.** Score + metric breakdown go into the archive, keyed to the candidate.

After all candidates in a generation are scored, the Orchestrator feeds the ranked
results (each candidate's reward + its objective breakdown) back to the Generator,
which reflects on *why* the winners won and proposes the next generation.

## What running it means, operationally

It's a single command you kick off and supervise, most naturally overnight:

```
python -m eureka.search --task standup \
    --generations 5 --candidates 6 \
    --proxy-iters 500 --seeds 2
```

Concretely, that:

- Runs **generations × candidates × seeds** training runs (here up to 60), each a
  time-boxed proxy run launched through our normal trainer. On a multi-GPU box these
  can run several at a time.
- Streams progress: each candidate's objective score and metric breakdown as it
  finishes, so you can watch the search improve generation over generation.
- Is **interruptible and resumable** — it's an outer loop over training runs, and the
  archive is checkpointed, so you can stop and pick up.
- **Wall-clock is dominated entirely by the training runs**, not by Claude or the
  harness. That's why proxy (short) runs are the cost knob.
- **Output:** a ranked archive and a single winning **reward config** (weights, or a
  generated rewards module). That winner is an ordinary artifact — you then train it
  to full convergence with the normal `scripts/rsl_rl/train.py`, exactly as you train
  any reward today. The search's job is only to *find the good reward*; producing the
  final policy is your normal pipeline, unchanged.

## How it merges into the code

The guiding principle is **wrap, don't fork.** Concretely:

- **New, self-contained:** the `eureka/` package (propose, inject, run, evaluate,
  search). It imports and drives the existing trainer; it is not entangled with it.
  Delete the folder and the trainer is exactly as it was.
- **The Runner calls the existing entrypoint** — `scripts/rsl_rl/train.py` /
  `variants.py` — as a subprocess with parameters. Our RL algorithm, env, and PPO
  config are untouched; the loop just *invokes* them with different rewards and a
  capped iteration count.
- **The one real touchpoint in existing code** is making `RewardsCfg` accept an
  injected reward:
  - **Scope A (weights):** add a small override path so a run can be given a weight
    dictionary. Our `variants.py` already parameterizes runs, so this is a natural
    extension of an existing mechanism, not a new concept.
  - **Scope B (synthesis):** designate a `mdp/rewards_generated.py` module that
    `RewardsCfg` imports and the Injector rewrites per candidate — an empty,
    clearly-marked file the search owns.
- **The Evaluator is the new substantive piece** — a script that loads a checkpoint,
  rolls out episodes, and returns the scalar objective. Worth building regardless of
  Eureka: a "score a trained standup policy on the metrics we actually care about"
  tool is generally useful.
- **Deliberately not touched:** the PPO/RSL-RL trainer internals, the base
  environment, the observation/action spaces, the sim setup. The loop sits entirely
  *around* them.

So the merge footprint is **one new folder, one new eval script, and one small,
opt-in hook in `RewardsCfg`.** The day after we merge it, everyone's normal `train.py`
workflow is identical, with a new optional tool sitting next to it.

## The cost mechanics in detail

This is where the real team decision lives. The idea is proven; the cost is real.

| Consideration | Reality |
|---|---|
| **The inner loop is a full training run** | Unlike the motor case (milliseconds per candidate), each locomotion reward candidate means an RL training run — minutes to an hour on GPU. This dominates everything. 6 candidates × 5 generations ≈ **30 runs ≈ ~20 GPU-hours** (× seeds). |
| **RL is noisy** | Ranking rewards on a single seed is unreliable; robust ranking may need 2–3 seeds per candidate, multiplying the cost. |
| **We need a good objective metric** | The whole thing hinges on a fixed fitness score that captures what we truly want and can't be gamed. Defining it well for standup/walking is real design work — and worth doing regardless. |
| **Bad rewards crash training** | The harness must treat a diverged/NaN run as a failed candidate, not an error. Standard, but must be built. |

**The single biggest lever** is short **proxy runs**: train only a few hundred
iterations to *rank* candidates, then run the winners to full length. If proxy
rankings correlate with full-run quality, the cost drops dramatically. Validating that
correlation is the first thing scope A should measure.

## References

- Ma et al., **"Eureka: Human-Level Reward Design via Coding Large Language Models"**
  (NVIDIA, 2023) — the original method, tested on Isaac Gym locomotion & manipulation.
- **DrEureka** — the sim-to-real follow-up (auto-tuning rewards *and* domain
  randomization for real-robot transfer).
- Our working reference implementation: [`policy/reward_search/`](../policy/reward_search/)
  in this repo, with its own [README](../policy/reward_search/README.md).

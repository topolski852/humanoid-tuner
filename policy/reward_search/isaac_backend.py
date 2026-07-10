"""Isaac-backed inner optimizer for the Eureka loop — the SAME job optimize.py
does (find gains that maximize a candidate reward), but scored on the Isaac
single-joint plant instead of the numpy sim, and VECTORISED: a whole gain
population is scored in one PhysX rollout.

Drop-in for `optimize.optimize_gains`: `optimize_gains_isaac(reward_fn, scorer)`
returns the same `(best_gains, best_response, best_reward)` tuple, so the outer
propose -> optimize -> grade -> reflect loop (loop.py) is unchanged — point it
here and it "trains itself" against the higher-fidelity motor.

Runs under humanoid-policy's venv (boots Isaac). From the tuner's numpy venv,
drive the substrate's JSON-job CLI across the process boundary instead.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from sim.actuator import Gains, Plant, Response

from .optimize import DEFAULT_BOUNDS, _make_gains


class IsaacBatchScorer:
    """Wraps one Isaac app (fixed population size) as a batch response oracle."""

    def __init__(self, inertia: float, plant: Plant, pop: int, device: str = "cpu",
                 step_rad: float = 0.5, duration: float = 1.0):
        from sim.isaac.substrate import IsaacSingleJoint

        self.pop = int(pop)
        self.plant = plant
        self.step_rad = step_rad
        self.duration = duration
        self.sim = IsaacSingleJoint(inertia=inertia, num_envs=self.pop, device=device)

    def responses(self, gains: np.ndarray) -> list[Response]:
        """Score up to `pop` gain sets in ONE rollout; pad short batches."""
        gains = np.asarray(gains, dtype=float).reshape(-1, 3)
        n = gains.shape[0]
        if n > self.pop:
            raise ValueError(f"batch {n} exceeds population {self.pop}")
        padded = np.zeros((self.pop, 3))
        padded[:n] = gains
        out = self.sim.rollout(
            gains=padded, step_rad=self.step_rad, duration=self.duration,
            damping=self.plant.damping, coulomb=self.plant.coulomb,
            torque_limit=self.plant.torque_limit,
            torque_filter_alpha=self.plant.torque_filter_alpha,
        )
        target = np.full_like(out["t"], self.step_rad)
        return [Response(t=out["t"], target=target, pos=out["pos"][i], vel=out["vel"][i],
                         step=self.step_rad) for i in range(n)]

    def close(self):
        self.sim.close()


def optimize_gains_isaac(
    reward_fn: Callable[[Response], float],
    scorer: IsaacBatchScorer,
    bounds: np.ndarray = DEFAULT_BOUNDS,
    max_rounds: int = 10,
    patience: int = 3,
    seed: int = 0,
) -> tuple[Gains, Response, float]:
    """Vectorised random + local search that LOOPS UNTIL OPTIMISED (patience).

    Round 0 scores a full random population in one rollout. Each later round
    scores `best +- scale` along each gain axis (plus fresh random explorers to
    fill the population) in one rollout, shrinking `scale` when a round fails to
    improve. Stops after `patience` non-improving rounds — the "until the policy
    is optimized" stop condition.
    """
    rng = np.random.default_rng(seed)
    lo, hi = bounds[:, 0], bounds[:, 1]
    pop = scorer.pop

    def score_batch(cands: np.ndarray) -> tuple[np.ndarray, list[Response]]:
        resps = scorer.responses(cands)
        rewards = np.array([_safe_reward(reward_fn, r) for r in resps])
        return rewards, resps

    # round 0: random population
    cands = lo + (hi - lo) * rng.random((pop, 3))
    rewards, resps = score_batch(cands)
    bi = int(np.argmax(rewards))
    best_x, best_r, best_resp = cands[bi].copy(), float(rewards[bi]), resps[bi]

    scale = 0.25 * (hi - lo)
    stale = 0
    for _ in range(max_rounds):
        if stale >= patience:
            break
        # local perturbations of the incumbent along each axis, +-
        perturb = [best_x]
        for d in range(3):
            for s in (+1, -1):
                x = best_x.copy()
                x[d] = np.clip(x[d] + s * scale[d], lo[d], hi[d])
                perturb.append(x)
        perturb = np.array(perturb)
        # fill the rest of the population with random explorers (free — same rollout)
        n_rand = pop - perturb.shape[0]
        if n_rand > 0:
            perturb = np.vstack([perturb, lo + (hi - lo) * rng.random((n_rand, 3))])
        rewards, resps = score_batch(perturb[:pop])
        bi = int(np.argmax(rewards))
        if rewards[bi] > best_r:
            best_x, best_r, best_resp = perturb[bi].copy(), float(rewards[bi]), resps[bi]
            stale = 0
        else:
            scale *= 0.5
            stale += 1

    return _make_gains(best_x), best_resp, best_r


def _safe_reward(reward_fn, resp) -> float:
    try:
        r = reward_fn(resp)
        return r if np.isfinite(r) else -np.inf
    except Exception:
        return -np.inf

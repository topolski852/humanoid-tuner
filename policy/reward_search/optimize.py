"""Inner optimizer: find gains that maximize a candidate reward.

⚠️  PHASE-0 STAND-IN. This is a gradient-free search over a FIXED gain set — the
placeholder for the eventual RL gain-scheduling policy (DESIGN §2/§4). It exists so
the Eureka loop is runnable end-to-end today. When the RL trainer lands, swap this
module out behind the same `optimize_gains(...)` signature: the reward-search loop
above it does not change. A fixed-gain optimizer can't produce state-dependent
gains — that's the capability the RL policy adds later — but it's more than enough
to tell a good reward function from a bad one, which is all the outer loop needs.

Method: random search over the gain box, then a few rounds of local coordinate
refinement around the best point. No scipy/cma dependency.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from sim.actuator import Gains, Plant, Response, simulate_step

# gain search box: (position_kp, velocity_kp/Kd, position_ki)
DEFAULT_BOUNDS = np.array([[0.0, 60.0], [0.0, 5.0], [0.0, 2.0]])


def _make_gains(x: np.ndarray) -> Gains:
    return Gains(position_kp=float(x[0]), velocity_kp=float(x[1]), position_ki=float(x[2]))


def optimize_gains(
    reward_fn: Callable[[Response], float],
    plant: Plant | None = None,
    bounds: np.ndarray = DEFAULT_BOUNDS,
    n_random: int = 200,
    refine_rounds: int = 4,
    step_rad: float = 0.5,
    seed: int = 0,
) -> tuple[Gains, Response, float]:
    """Return (best_gains, best_response, best_reward) for this reward function."""
    plant = plant or Plant()
    rng = np.random.default_rng(seed)
    lo, hi = bounds[:, 0], bounds[:, 1]

    def score(x: np.ndarray) -> tuple[float, Response]:
        resp = simulate_step(_make_gains(x), plant, step_rad=step_rad)
        try:
            return reward_fn(resp), resp
        except Exception:                       # a reward that blows up on some gains
            return -np.inf, resp                # is simply unfit there, not fatal

    # random search
    best_x = lo + (hi - lo) * rng.random(3)
    best_r, best_resp = score(best_x)
    for _ in range(n_random):
        x = lo + (hi - lo) * rng.random(3)
        r, resp = score(x)
        if r > best_r:
            best_r, best_x, best_resp = r, x, resp

    # local coordinate refinement (shrinking steps)
    scale = 0.25 * (hi - lo)
    for _ in range(refine_rounds):
        improved = False
        for d in range(3):
            for sign in (+1, -1):
                x = best_x.copy()
                x[d] = np.clip(x[d] + sign * scale[d], lo[d], hi[d])
                r, resp = score(x)
                if r > best_r:
                    best_r, best_x, best_resp, improved = r, x, resp, True
        if not improved:
            scale *= 0.5

    return _make_gains(best_x), best_resp, float(best_r)

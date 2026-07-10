"""Robustness-aware evaluation — grade gains under plant perturbation.

The nominal fitness (sim/metrics.py) grades gains on ONE plant, so the search
chases the stability cliff: the best nominal gains sit exactly at the edge of
oscillation with zero margin (kd=2.204 on the geared plant — 0.2% less hunts).
That is fragile on real hardware, where inertia/friction/damping drift with load,
temperature and wear.

`robust_fitness` grades the SAME gains across a spread of perturbed plants and
returns the worst case, so cliff-edge gains — which tip into oscillation for some
plant in the spread — are penalized, and the optimum moves to gains WITH margin.
This is the sim-to-real domain randomization the DESIGN doc calls for.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .actuator import Gains, Plant, simulate_step
from .metrics import fitness


def perturbed_plants(plant: Plant, n: int = 12, inertia_spread: float = 0.25,
                     coulomb_spread: float = 0.35, damping_spread: float = 0.6,
                     seed: int = 0) -> list[Plant]:
    """Nominal plant + `n` copies with inertia/friction/damping perturbed ±spread.

    Ranges reflect real uncertainty: load-dependent inertia, temperature/wear/
    direction-dependent friction, poorly-known viscous damping.
    """
    rng = np.random.default_rng(seed)
    out = [plant]
    for _ in range(n):
        out.append(replace(
            plant,
            inertia=plant.inertia * (1 + inertia_spread * rng.uniform(-1, 1)),
            coulomb=max(0.0, plant.coulomb * (1 + coulomb_spread * rng.uniform(-1, 1))),
            damping=max(0.0, plant.damping * (1 + damping_spread * rng.uniform(-1, 1))),
        ))
    return out


def robust_fitness(gains: Gains, plants: list[Plant], step_rad: float = 0.5,
                   agg: str = "worst") -> float:
    """Aggregate fitness of `gains` across a perturbed plant set.

    agg="worst" (default) -> worst-case fitness (min); rewards margin. "mean" ->
    average. "cvar" -> mean of the worst 25% (a softer worst-case).
    """
    fits = np.array([fitness(simulate_step(gains, p, step_rad=step_rad)) for p in plants])
    if agg == "mean":
        return float(fits.mean())
    if agg == "cvar":
        k = max(1, len(fits) // 4)
        return float(np.sort(fits)[:k].mean())
    return float(fits.min())

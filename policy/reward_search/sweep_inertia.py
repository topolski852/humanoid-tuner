"""Chart how the optimal gains track output-shaft inertia — the Phase-1 preview.

For a grid of inertias (bracketing the measured M6C12 reflected 0.0224), find the
gains that maximise the GROUND-TRUTH objective (sim/metrics.fitness) directly —
the achievable frontier per inertia — under two gain boxes: the original
(kp<=60) and a widened one (kp<=150). This shows (a) whether the kp=60 rail was
clipping the optimum and (b) how optimal kd rises with inertia (the "infer the
load, adapt the gains" story the RL policy will automate).

    python -m policy.reward_search.sweep_inertia

Writes runs/inertia_sweep.png and prints a table.
"""

from __future__ import annotations

import numpy as np

from sim.actuator import Plant
from sim.metrics import fitness, step_metrics

from .optimize import optimize_gains

INERTIAS = [0.0008, 0.002, 0.005, 0.0224, 0.05, 0.1]   # kg·m² (0.0008 toy, 0.0224 M6C12)
NARROW = np.array([[0.0, 60.0], [0.0, 5.0], [0.0, 2.0]])
WIDE = np.array([[0.0, 150.0], [0.0, 10.0], [0.0, 2.0]])


def _best(inertia: float, bounds) -> tuple:
    plant = Plant()
    plant.inertia = inertia
    # optimise the ground-truth objective itself -> the achievable frontier
    g, resp, _ = optimize_gains(fitness, plant, bounds=bounds, n_random=400, refine_rounds=6)
    return g, fitness(resp), step_metrics(resp)


def main() -> None:
    rows = []
    print(f"{'inertia':>9s} | {'box':>6s} | {'kp':>6s} {'kd':>6s} {'ki':>5s} | "
          f"{'fitness':>8s} {'settle':>7s} {'over%':>6s}")
    print("-" * 72)
    for J in INERTIAS:
        for tag, b in (("kp<=60", NARROW), ("kp<=150", WIDE)):
            g, fit, m = _best(J, b)
            rows.append((J, tag, g, fit, m))
            print(f"{J:9.4f} | {tag:>6s} | {g.position_kp:6.1f} {g.velocity_kp:6.2f} "
                  f"{g.position_ki:5.2f} | {fit:8.3f} {m['settle_time']:7.3f} "
                  f"{m['overshoot']*100:6.1f}")
        print("-" * 72)

    _plot(rows)


def _plot(rows) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("(matplotlib unavailable — skipping plot)")
        return
    Js = INERTIAS
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for tag, color in (("kp<=60", "tab:blue"), ("kp<=150", "tab:red")):
        sel = [r for r in rows if r[1] == tag]
        kd = [r[2].velocity_kp for r in sel]
        kp = [r[2].position_kp for r in sel]
        fit = [r[3] for r in sel]
        axes[0].plot(Js, kp, "o-", color=color, label=tag)
        axes[1].plot(Js, kd, "o-", color=color, label=tag)
        axes[2].plot(Js, fit, "o-", color=color, label=tag)
    for ax, ttl, yl in zip(axes, ["optimal kp vs inertia", "optimal kd vs inertia",
                                   "achievable fitness vs inertia"],
                           ["position_kp", "velocity_kp (Kd)", "fitness (higher=better)"]):
        ax.set_xscale("log"); ax.set_xlabel("output inertia (kg·m²)"); ax.set_ylabel(yl)
        ax.set_title(ttl); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax.axvline(0.0224, color="0.6", ls="--", lw=0.8)  # M6C12
    fig.tight_layout()
    fig.savefig("runs/inertia_sweep.png", dpi=110)
    print("wrote runs/inertia_sweep.png")


if __name__ == "__main__":
    main()

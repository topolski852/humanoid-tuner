"""The outer Eureka loop: propose → optimize gains → grade → reflect → repeat.

Run it:
    # no API key needed — grades the seed rewards so you can see the pipeline work:
    python -m policy.reward_search.loop --dry-run

    # the real loop — Claude proposes each generation (needs Anthropic API access):
    python -m policy.reward_search.loop --iterations 5 --candidates 6

The best reward found (code + gains + metrics) is written to reward_search_best.json.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from sim.actuator import Plant
from sim.metrics import fitness, step_metrics

from .optimize import optimize_gains
from .propose import Candidate, propose_rewards
from .rewards import SEED_REWARDS, RewardError, compile_reward


def evaluate(name: str, code: str, plant: Plant, seed: int,
             genome: list | None = None, bounds=None) -> Candidate | None:
    """Compile a reward, optimize gains against it, grade with ground-truth fitness."""
    try:
        reward_fn = compile_reward(code)
    except RewardError as e:
        print(f"  [skip] {name}: {e}")
        return None
    kw = {"seed": seed}
    if bounds is not None:
        kw["bounds"] = bounds
    gains, resp, _ = optimize_gains(reward_fn, plant, **kw)
    cand = Candidate(name=name, code=code, fitness=fitness(resp), metrics=step_metrics(resp))
    cand.gains = gains  # type: ignore[attr-defined]  (stashed for reporting)
    cand.genome = genome  # type: ignore[attr-defined]  (local proposer evolves this)
    m = cand.metrics
    print(
        f"  {name:28s} fitness={cand.fitness:8.4f}  "
        f"settle={m['settle_time']:.3f}s overshoot={m['overshoot']*100:4.1f}% "
        f"ss_err={m['ss_error']:.4f}  gains(kp={gains.position_kp:.1f}, "
        f"kd={gains.velocity_kp:.2f}, ki={gains.position_ki:.2f})"
    )
    return cand


def _log_candidate(fp, gen: int, cand: Candidate) -> None:
    """Append one candidate's full record to the JSONL history log."""
    g = cand.gains  # type: ignore[attr-defined]
    fp.write(json.dumps({
        "gen": gen, "name": cand.name, "fitness": cand.fitness,
        "metrics": cand.metrics,
        "gains": {"position_kp": g.position_kp, "velocity_kp": g.velocity_kp,
                  "position_ki": g.position_ki},
        "code": cand.code,
    }) + "\n")
    fp.flush()


def _write_best(path: str, best: Candidate, plant: Plant) -> None:
    g = best.gains  # type: ignore[attr-defined]
    with open(path, "w") as f:
        json.dump({
            "name": best.name, "fitness": best.fitness, "metrics": best.metrics,
            "plant_inertia": plant.inertia,
            "gains": {"position_kp": g.position_kp, "velocity_kp": g.velocity_kp,
                      "position_ki": g.position_ki},
            "code": best.code,
        }, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Eureka-style reward search for gain tuning")
    ap.add_argument("--iterations", type=int, default=3, help="outer generations")
    ap.add_argument("--candidates", type=int, default=6, help="rewards proposed per generation")
    ap.add_argument("--dry-run", action="store_true", help="use seed rewards only; no API calls")
    ap.add_argument("--local", action="store_true",
                    help="evolve rewards with the API-free local proposer (no API key)")
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--seed", type=int, default=0, help="sim/optimizer RNG seed")
    ap.add_argument("--inertia", type=float, default=None,
                    help="plant output-shaft inertia (kg·m²); default = toy 8e-4. "
                         "Use 0.0224 for the M6C12 reflected rotor inertia.")
    ap.add_argument("--damping", type=float, default=None, help="plant viscous damping")
    ap.add_argument("--coulomb", type=float, default=None,
                    help="plant Coulomb friction (Nm); previews a Phase-1 condition")
    ap.add_argument("--kp-max", type=float, default=60.0, help="upper bound of position_kp search")
    ap.add_argument("--kd-max", type=float, default=5.0, help="upper bound of velocity_kp (Kd) search")
    ap.add_argument("--ki-max", type=float, default=2.0, help="upper bound of position_ki search")
    ap.add_argument("--best-out", default="reward_search_best.json")
    ap.add_argument("--log", default=None,
                    help="JSONL path to append every candidate (overnight history)")
    args = ap.parse_args()

    plant = Plant()
    if args.inertia is not None:
        plant.inertia = args.inertia
    if args.damping is not None:
        plant.damping = args.damping
    if args.coulomb is not None:
        plant.coulomb = args.coulomb
    bounds = np.array([[0.0, args.kp_max], [0.0, args.kd_max], [0.0, args.ki_max]])
    rng = np.random.default_rng(args.seed)
    history: list[Candidate] = []
    log_fp = open(args.log, "a") if args.log else None

    def record(gen: int, c: Candidate) -> None:
        history.append(c)
        if log_fp:
            _log_candidate(log_fp, gen, c)

    mode = "local" if args.local else ("dry-run" if args.dry_run else "api")
    print(f"plant inertia={plant.inertia:g} kg·m²  damping={plant.damping:g}  "
          f"coulomb={plant.coulomb:g}   mode={mode}")
    print("Generation 0: seed rewards")
    for name, code in SEED_REWARDS.items():
        c = evaluate(name, code, plant, args.seed, bounds=bounds)
        if c:
            record(0, c)

    best = max(history, key=lambda c: c.fitness)
    _write_best(args.best_out, best, plant)  # checkpoint after gen 0

    if args.dry_run:
        print("\n[dry-run] stopping after seeds (no proposals).")
    else:
        for gen in range(1, args.iterations + 1):
            t0 = time.perf_counter()
            if args.local:
                from .propose_local import propose_rewards_local
                proposed = propose_rewards_local(history, args.candidates, rng)
            else:
                print(f"\nGeneration {gen}: asking {args.model} for {args.candidates} rewards")
                try:
                    proposed = [(n, c, None) for n, c in
                                propose_rewards(history, args.candidates, model=args.model)]
                except Exception as e:                          # noqa: BLE001
                    print(f"  proposal call failed ({e}); stopping. "
                          "Check API access, or run with --local / --dry-run.")
                    break
            gen_best = best.fitness
            for name, code, genome in proposed:
                c = evaluate(f"g{gen}:{name}", code, plant, args.seed, genome=genome, bounds=bounds)
                if c:
                    record(gen, c)
            best = max(history, key=lambda c: c.fitness)
            _write_best(args.best_out, best, plant)            # checkpoint each generation
            improved = "^" if best.fitness > gen_best else " "
            print(f"[gen {gen:3d}] best fitness={best.fitness:8.4f} {improved}  "
                  f"({len(history)} evaluated, {time.perf_counter()-t0:.1f}s)")

    if log_fp:
        log_fp.close()
    best = max(history, key=lambda c: c.fitness)
    _write_best(args.best_out, best, plant)
    g = best.gains  # type: ignore[attr-defined]
    print(f"\nBEST: {best.name}  fitness={best.fitness:.4f}  "
          f"gains(kp={g.position_kp:.1f}, kd={g.velocity_kp:.2f}, ki={g.position_ki:.2f})")
    print(f"wrote {args.best_out}" + (f" and {args.log}" if args.log else ""))


if __name__ == "__main__":
    main()

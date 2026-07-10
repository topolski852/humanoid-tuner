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

from sim.actuator import Plant
from sim.metrics import fitness, step_metrics

from .optimize import optimize_gains
from .propose import Candidate, propose_rewards
from .rewards import SEED_REWARDS, RewardError, compile_reward


def evaluate(name: str, code: str, plant: Plant, seed: int) -> Candidate | None:
    """Compile a reward, optimize gains against it, grade with ground-truth fitness."""
    try:
        reward_fn = compile_reward(code)
    except RewardError as e:
        print(f"  [skip] {name}: {e}")
        return None
    gains, resp, _ = optimize_gains(reward_fn, plant, seed=seed)
    cand = Candidate(name=name, code=code, fitness=fitness(resp), metrics=step_metrics(resp))
    cand.gains = gains  # type: ignore[attr-defined]  (stashed for reporting)
    m = cand.metrics
    print(
        f"  {name:28s} fitness={cand.fitness:8.4f}  "
        f"settle={m['settle_time']:.3f}s overshoot={m['overshoot']*100:4.1f}% "
        f"ss_err={m['ss_error']:.4f}  gains(kp={gains.position_kp:.1f}, "
        f"kd={gains.velocity_kp:.2f}, ki={gains.position_ki:.2f})"
    )
    return cand


def main() -> None:
    ap = argparse.ArgumentParser(description="Eureka-style reward search for gain tuning")
    ap.add_argument("--iterations", type=int, default=3, help="outer generations")
    ap.add_argument("--candidates", type=int, default=6, help="rewards proposed per generation")
    ap.add_argument("--dry-run", action="store_true", help="use seed rewards only; no API calls")
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--seed", type=int, default=0, help="sim/optimizer RNG seed")
    args = ap.parse_args()

    plant = Plant()
    history: list[Candidate] = []

    print("Generation 0: seed rewards")
    for name, code in SEED_REWARDS.items():
        c = evaluate(name, code, plant, args.seed)
        if c:
            history.append(c)

    if args.dry_run:
        print("\n[dry-run] stopping after seeds (no Claude proposals).")
    else:
        for gen in range(1, args.iterations + 1):
            print(f"\nGeneration {gen}: asking {args.model} for {args.candidates} rewards")
            try:
                proposed = propose_rewards(history, args.candidates, model=args.model)
            except Exception as e:                          # noqa: BLE001
                print(f"  proposal call failed ({e}); stopping. "
                      "Check API access, or run with --dry-run.")
                break
            for name, code in proposed:
                c = evaluate(f"g{gen}:{name}", code, plant, args.seed)
                if c:
                    history.append(c)

    best = max(history, key=lambda c: c.fitness)
    print(f"\nBEST: {best.name}  fitness={best.fitness:.4f}")
    g = best.gains  # type: ignore[attr-defined]
    out = {
        "name": best.name,
        "fitness": best.fitness,
        "metrics": best.metrics,
        "gains": {"position_kp": g.position_kp, "velocity_kp": g.velocity_kp,
                  "position_ki": g.position_ki},
        "code": best.code,
    }
    with open("reward_search_best.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote reward_search_best.json")


if __name__ == "__main__":
    main()

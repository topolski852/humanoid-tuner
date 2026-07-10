"""Summarize an overnight reward-search run from its JSONL history log.

    python -m policy.reward_search.analyze runs/real.jsonl [runs/toy.jsonl ...]

Prints, per run: best-so-far fitness at each generation (the convergence curve),
and the winning reward's gains, metrics, and code.
"""

from __future__ import annotations

import json
import sys


def summarize(path: str) -> None:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    if not rows:
        print(f"{path}: empty"); return
    gens = sorted({r["gen"] for r in rows})
    best_so_far = -1e18
    best_row = None
    print(f"\n=== {path}  ({len(rows)} candidates, {len(gens)} generations) ===")
    print("gen  best_so_far   gen_best   n")
    for g in gens:
        gr = [r for r in rows if r["gen"] == g]
        gbest = max(gr, key=lambda r: r["fitness"])
        if gbest["fitness"] > best_so_far:
            best_so_far, best_row = gbest["fitness"], gbest
        print(f"{g:3d}  {best_so_far:10.4f}  {gbest['fitness']:9.4f}  {len(gr)}")

    b = best_row
    m = b["metrics"]
    gn = b["gains"]
    print(f"\nWINNER: {b['name']}  fitness={b['fitness']:.4f}")
    print(f"  gains: kp={gn['position_kp']:.2f} kd={gn['velocity_kp']:.2f} ki={gn['position_ki']:.2f}")
    print(f"  metrics: settle={m['settle_time']:.3f}s overshoot={m['overshoot']*100:.1f}% "
          f"ss_err={m['ss_error']:.4f} rms={m['rms_error']:.4f} osc={m['oscillation']:.0f}")
    print("  reward code:")
    for line in b["code"].strip().splitlines():
        print("    " + line)


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        print("usage: python -m policy.reward_search.analyze <run.jsonl> [...]")
        return
    for p in paths:
        summarize(p)


if __name__ == "__main__":
    main()

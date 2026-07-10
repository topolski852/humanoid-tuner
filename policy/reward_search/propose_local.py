"""API-free reward proposer — a local stand-in for the Claude generator.

Same job and signature as propose.propose_rewards, but needs NO API: it evolves
reward functions with a genetic search over a term-based genome (mutation +
crossover of prior winners, plus random exploration). Lets the Eureka loop run
overnight and show its propose -> optimize -> grade -> reflect dynamics without
credentials. Swap propose.propose_rewards back in when API access exists.

A reward genome is a set of penalty terms; each term is (feature, transform,
weight). The emitted `reward(resp)` returns the NEGATIVE weighted cost (higher =
better), using only `np` + safe builtins so it runs under compile_reward's
restricted exec. Evolution operates on the structured genome, not the text.
"""

from __future__ import annotations

import numpy as np

from .propose import Candidate

# Feature terms (each a cost >= 0, lower is better), as code expressions over the
# locals emitted below. The genome selects a subset and weights/transforms them.
_FEATURES: dict[str, str] = {
    "rms": "float(np.sqrt(np.mean(err ** 2)))",
    "iae": "float(np.mean(np.abs(err)))",
    "peak": "float(np.max(np.abs(err)))",
    "overshoot": "max(0.0, (float(np.max(pos)) - step) / step)",
    "ss_error": "float(np.mean(np.abs(err[-max(1, len(err) // 10):])))",
    "oscillation": "float(np.sum(np.abs(np.diff(np.sign(vel[len(vel)//5:]))) > 0))",
    "effort": "float(np.mean(np.abs(np.diff(vel))))",
    "settle_tw": "float(np.mean(t * np.abs(err)))",   # time-weighted error ~ settling
}

# Transforms applied to a term value x (>= 0). Kept smooth/bounded-ish.
_TRANSFORMS: dict[str, str] = {
    "lin": "{x}",
    "sq": "({x}) ** 2",
    "sqrt": "np.sqrt({x})",
    "log1p": "np.log1p({x})",
}

_FEATURE_KEYS = list(_FEATURES)
_TRANSFORM_KEYS = list(_TRANSFORMS)


def _emit_code(genome: list[tuple[str, str, float]]) -> str:
    """Turn a genome [(feature, transform, weight), ...] into a reward snippet."""
    lines = [
        "def reward(resp):",
        "    err = resp.err; pos = resp.pos; vel = resp.vel; t = resp.t",
        "    step = resp.step if resp.step else 1.0",
        "    cost = 0.0",
    ]
    for feat, trans, w in genome:
        expr = _TRANSFORMS[trans].format(x=_FEATURES[feat])
        lines.append(f"    cost += {w:.4f} * ({expr})")
    lines.append("    return -float(cost)")
    return "\n".join(lines) + "\n"


def _random_genome(rng: np.random.Generator) -> list[tuple[str, str, float]]:
    k = int(rng.integers(2, 6))                       # 2..5 active terms
    feats = list(rng.choice(_FEATURE_KEYS, size=k, replace=False))
    genome = []
    for f in feats:
        trans = _TRANSFORM_KEYS[int(rng.integers(len(_TRANSFORM_KEYS)))]
        w = float(np.round(10 ** rng.uniform(-1.0, 1.0), 3))   # 0.1 .. 10
        genome.append((f, trans, w))
    return genome


def _mutate(genome, rng) -> list[tuple[str, str, float]]:
    g = [list(term) for term in genome]
    # jitter every weight (log-normal), occasionally flip a transform
    for term in g:
        term[2] = float(np.round(max(0.01, term[2] * float(np.exp(rng.normal(0, 0.4)))), 3))
        if rng.random() < 0.25:
            term[1] = _TRANSFORM_KEYS[int(rng.integers(len(_TRANSFORM_KEYS)))]
    # sometimes add a new term
    if rng.random() < 0.4:
        present = {t[0] for t in g}
        avail = [f for f in _FEATURE_KEYS if f not in present]
        if avail:
            f = avail[int(rng.integers(len(avail)))]
            g.append([f, _TRANSFORM_KEYS[int(rng.integers(len(_TRANSFORM_KEYS)))],
                      float(np.round(10 ** rng.uniform(-1, 1), 3))])
    # sometimes drop a term (keep >= 1)
    if rng.random() < 0.3 and len(g) > 1:
        del g[int(rng.integers(len(g)))]
    return [tuple(t) for t in g]


def _crossover(a, b, rng) -> list[tuple[str, str, float]]:
    """Union the parents' features; for shared ones pick a parent's term at random."""
    da = {t[0]: t for t in a}
    db = {t[0]: t for t in b}
    child = []
    for f in set(da) | set(db):
        if f in da and f in db:
            child.append(da[f] if rng.random() < 0.5 else db[f])
        else:
            child.append(da.get(f, db.get(f)))
    # cap size to keep it simple
    rng.shuffle(child)
    return child[:5]


def _genome_from_candidate(c: Candidate) -> list | None:
    """Recover a genome stashed on a Candidate (propose_rewards_local sets it)."""
    return getattr(c, "genome", None)


def propose_rewards_local(
    history: list[Candidate], n: int, rng: np.random.Generator
) -> list[tuple[str, str, list]]:
    """Return [(name, code, genome), ...]. First gen: random. Later: evolve winners."""
    parents = sorted(history, key=lambda c: c.fitness, reverse=True)[:6]
    parent_genomes = [g for g in (_genome_from_candidate(c) for c in parents) if g]

    out = []
    for i in range(n):
        if not parent_genomes:
            genome, how = _random_genome(rng), "rand"
        else:
            roll = rng.random()
            if roll < 0.5 and len(parent_genomes) >= 2:
                pa, pb = (parent_genomes[int(rng.integers(len(parent_genomes)))] for _ in range(2))
                genome, how = _crossover(pa, pb, rng), "xover"
            elif roll < 0.85:
                pa = parent_genomes[int(rng.integers(len(parent_genomes)))]
                genome, how = _mutate(pa, rng), "mut"
            else:
                genome, how = _random_genome(rng), "rand"
        out.append((f"{how}{i}", _emit_code(genome), genome))
    return out

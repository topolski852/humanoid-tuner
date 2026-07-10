"""The Eureka generator: Claude proposes / revises reward functions.

Given the task description and the ranked results of prior candidates (their code,
ground-truth fitness, and raw metrics), ask Claude for the next generation of
reward functions. This is the one piece that needs Claude API access — everything
else (sim, optimizer, metrics) runs locally.

Auth: uses the standard Anthropic SDK credential resolution (ANTHROPIC_API_KEY, or
an `ant auth login` profile). If neither is present, run the loop with --dry-run to
exercise the sim/optimizer/metrics pipeline on the seed rewards alone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

MODEL = "claude-opus-4-8"

SYSTEM = """You are designing REWARD FUNCTIONS for a reinforcement-learning loop \
that tunes the PID gains of a robot joint's position controller.

An inner optimizer searches gains to MAXIMIZE your reward over a simulated step \
response. A separate, fixed ground-truth objective then judges the resulting \
response on: settling time, overshoot, steady-state error, RMS tracking error, \
oscillation, and effort (lower is better on all). Your reward is GOOD if optimizing \
it yields gains that score well on that ground-truth objective. You never see the \
ground-truth formula directly — you infer what works from the feedback.

Each reward is a Python snippet defining exactly:

    def reward(resp):
        # resp.t, resp.target, resp.pos, resp.vel, resp.err : 1-D numpy arrays
        #   (a ~100-sample, 100 Hz step response of one joint, output-shaft rad)
        # resp.step : float, the commanded step magnitude (rad)
        # `np` is in scope. No imports. Return a single float; HIGHER = better.
        ...

Design principles: shape the reward so its optimum coincides with a fast, \
non-overshooting, well-damped, zero-steady-state-error response. Combine terms \
(tracking error, overshoot penalty, settle-time proxy, oscillation/chatter \
penalty, effort). Prefer smooth, bounded terms. Return diverse candidates that \
explore different shaping ideas — not minor variants of one idea."""


@dataclass
class Candidate:
    name: str
    code: str
    fitness: float
    metrics: dict


def _history_block(history: list[Candidate], top_k: int = 6) -> str:
    if not history:
        return "No candidates evaluated yet. Propose a diverse first generation."
    ranked = sorted(history, key=lambda c: c.fitness, reverse=True)[:top_k]
    lines = ["Prior candidates, best first (fitness higher = better):"]
    for c in ranked:
        lines.append(
            f"\n--- {c.name}  fitness={c.fitness:.4f}  metrics={json.dumps({k: round(v, 4) for k, v in c.metrics.items()})}\n"
            f"{c.code.strip()}"
        )
    lines.append(
        "\nAnalyze WHY the better ones win and the worse ones lose, then propose "
        "improved, diverse rewards. Keep what works; fix what doesn't."
    )
    return "\n".join(lines)


_SCHEMA = {
    "type": "object",
    "properties": {
        "rewards": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "rationale": {"type": "string"},
                    "code": {"type": "string"},
                },
                "required": ["name", "rationale", "code"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["rewards"],
    "additionalProperties": False,
}


def propose_rewards(
    history: list[Candidate], n: int, model: str = MODEL
) -> list[tuple[str, str]]:
    """Return [(name, code), ...] proposed by Claude. Requires API access."""
    import anthropic  # imported lazily so --dry-run needs no SDK/key

    client = anthropic.Anthropic()
    user = (
        f"Propose {n} reward functions for the next generation.\n\n"
        f"{_history_block(history)}"
    )
    resp = client.messages.create(
        model=model,
        max_tokens=8000,
        thinking={"type": "adaptive"},
        system=SYSTEM,
        messages=[{"role": "user", "content": user}],
        output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
    )
    text = next(b.text for b in resp.content if b.type == "text")
    data = json.loads(text)
    return [(r["name"], r["code"]) for r in data["rewards"]]

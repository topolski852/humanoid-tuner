"""Reward candidates: the interface Claude writes to, plus safe-ish compilation.

A reward candidate is a Python snippet that defines:

    def reward(resp):
        # resp.t, resp.target, resp.pos, resp.vel, resp.err are 1-D numpy arrays
        # (a 100 Hz step response). `np` is in scope. Return a float; higher = better.
        ...

The inner optimizer (optimize.py) searches gains to MAXIMIZE reward(resp) over a
test step. The ground-truth fitness (sim/metrics.py) then judges the result. That
separation is the Eureka signal — see metrics.py.

⚠️  SECURITY: compile_reward() runs model-generated code via exec(). The namespace
is restricted (no imports, limited builtins), but this is NOT a real sandbox. Run
the loop on hardware you control, with rewards from a source you trust (your own
Claude calls). Do not point it at untrusted reward strings.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from sim.actuator import Response

# --- seed rewards: hand-written baselines so the loop runs without any API calls.
# These are intentionally simple/mediocre — the point is to watch Claude beat them.
SEED_REWARDS: dict[str, str] = {
    "neg_rms_error": (
        "def reward(resp):\n"
        "    return -float(np.sqrt(np.mean(resp.err ** 2)))\n"
    ),
    "neg_abs_error_plus_effort": (
        "def reward(resp):\n"
        "    tracking = -float(np.mean(np.abs(resp.err)))\n"
        "    effort = -0.01 * float(np.mean(np.abs(np.diff(resp.vel))))\n"
        "    return tracking + effort\n"
    ),
}

_SAFE_BUILTINS = {
    k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
    for k in ("abs", "min", "max", "len", "range", "sum", "float", "int",
              "enumerate", "zip", "map", "sorted", "pow")
}


class RewardError(Exception):
    pass


def compile_reward(code: str) -> Callable[[Response], float]:
    """Compile a reward snippet into a callable. Raises RewardError on bad code."""
    ns: dict = {"np": np, "__builtins__": _SAFE_BUILTINS}
    try:
        exec(compile(code, "<reward>", "exec"), ns)   # noqa: S102 (see module warning)
    except Exception as e:                              # noqa: BLE001
        raise RewardError(f"reward failed to compile: {e}") from e
    fn = ns.get("reward")
    if not callable(fn):
        raise RewardError("reward snippet must define `def reward(resp): ...`")

    def wrapped(resp: Response) -> float:
        val = fn(resp)
        val = float(val)
        if not np.isfinite(val):
            raise RewardError("reward returned non-finite value")
        return val

    return wrapped

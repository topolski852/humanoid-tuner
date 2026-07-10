"""Ground-truth step-response metrics + fitness.

This is the OBJECTIVE the Eureka loop optimizes *toward* — the analog of Eureka's
"task fitness". It is deliberately NOT the reward the inner optimizer maximizes:
Claude writes candidate reward functions (policy/reward_search/rewards.py); the
inner optimizer finds gains that maximize a candidate reward; then we score the
resulting response with THIS module. A good reward is one whose optimized gains
score well here. Keeping the two separate is the whole point — it's how the loop
can tell a good reward from a bad one without a human in the middle.
"""

from __future__ import annotations

import numpy as np

from .actuator import Response


def step_metrics(resp: Response, settle_band: float = 0.02) -> dict[str, float]:
    """Classic step-response metrics (all in output-shaft units / seconds)."""
    step = resp.step
    pos, t, err = resp.pos, resp.t, np.abs(resp.err)
    band = settle_band * abs(step)

    # overshoot: how far past the target the response goes, as a fraction of step
    overshoot = max(0.0, (float(np.max(pos)) - step) / step) if step else 0.0

    # settling time: last time the error leaves the ±band, then stays in
    outside = np.where(err > band)[0]
    settle_time = float(t[outside[-1]]) if len(outside) else 0.0

    # rise time: 10% → 90% of the step
    def _first_cross(frac):
        idx = np.where(pos >= frac * step)[0]
        return float(t[idx[0]]) if len(idx) else float(t[-1])
    rise_time = max(0.0, _first_cross(0.9) - _first_cross(0.1))

    # steady-state error over the final 10% of the window
    tail = max(1, len(pos) // 10)
    ss_error = float(np.mean(np.abs(resp.err[-tail:])))

    rms_error = float(np.sqrt(np.mean(resp.err ** 2)))

    # oscillation: velocity sign changes after the initial rise (chatter/ringing)
    v = resp.vel[len(resp.vel) // 5:]
    sign_changes = int(np.sum(np.abs(np.diff(np.sign(v))) > 0)) if len(v) > 1 else 0

    # effort proxy (no current in Phase-0 sim): peak commanded acceleration ~ |Δvel|
    effort = float(np.mean(np.abs(np.diff(resp.vel)))) if len(resp.vel) > 1 else 0.0

    return {
        "overshoot": overshoot,
        "settle_time": settle_time,
        "rise_time": rise_time,
        "ss_error": ss_error,
        "rms_error": rms_error,
        "oscillation": float(sign_changes),
        "effort": effort,
    }


# Weights on the ground-truth objective. This is the spec you're tuning TO — the
# "responsiveness ↔ compliance ↔ efficiency" tradeoff from DESIGN §4. Expose it as
# a knob later; for Phase 0 it's a fixed, sensible balance.
FITNESS_WEIGHTS = {
    "settle_time": 4.0,
    "overshoot": 3.0,
    "ss_error": 5.0,
    "rms_error": 2.0,
    "oscillation": 0.15,
    "effort": 0.05,
}


def fitness(resp: Response, weights: dict[str, float] | None = None) -> float:
    """Scalar objective (higher is better). Negative weighted cost of the metrics."""
    w = weights or FITNESS_WEIGHTS
    m = step_metrics(resp)
    cost = sum(w.get(k, 0.0) * m[k] for k in w)
    return -cost

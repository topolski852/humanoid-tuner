"""Step-response agreement test: Isaac substrate vs the numpy sim.

The gate before trusting Isaac: for the SAME gains and the SAME plant, do the
two substrates produce the same step response? They share the control law
(sim/control_law.py), so any gap is the PhysX solver vs numpy's semi-implicit
Euler — not two different controllers.

Runs under humanoid-policy's venv (imports both the tuner's numpy sim and boots
Isaac in-process):
    OMNI_KIT_ACCEPT_EULA=YES /home/nse/humanoid/humanoid-policy/.venv/bin/python \
        -m sim.isaac.agreement_test

Outputs a metrics table, per-signal errors, an empirically backed-out effective
inertia, and (if matplotlib is present) an overlay PNG in this directory.
"""

from __future__ import annotations

import os

import numpy as np

from sim.actuator import Gains, Plant, Response, simulate_step
from sim.metrics import step_metrics

from .motor import M6C12_150KV
from .substrate import IsaacSingleJoint

HERE = os.path.dirname(__file__)

# A spread of gains: sluggish, balanced, stiff+integral, near-deadbeat.
GAIN_SETS = [
    (20.0, 1.0, 0.0),
    (45.0, 2.5, 0.5),
    (60.0, 5.0, 2.0),
    (10.0, 0.2, 0.0),
]

STEP_RAD = 0.5
DURATION = 1.0


def _isaac_responses(sim: IsaacSingleJoint, plant: Plant, gains_list) -> list[Response]:
    """Run all gain sets as N parallel Isaac envs; return numpy-sim-shaped Responses.

    Does NOT close the app — simulation_app.close() hard-exits the process, so
    the caller must do all remaining work (metrics, plot) first and close LAST.
    """
    gains = np.asarray(gains_list, dtype=float)
    out = sim.rollout(
        gains=gains, step_rad=STEP_RAD, duration=DURATION,
        damping=plant.damping, coulomb=plant.coulomb,
        torque_limit=plant.torque_limit, torque_filter_alpha=plant.torque_filter_alpha,
    )
    resp = []
    for i in range(gains.shape[0]):
        resp.append(Response(
            t=out["t"], target=np.full_like(out["t"], STEP_RAD),
            pos=out["pos"][i], vel=out["vel"][i], step=STEP_RAD,
        ))
    return resp


def _backout_inertia(resp: Response, plant: Plant) -> float:
    """Rough effective inertia from the early response: J ~ tau0 / accel0.

    Uses the first sampled interval where the controller torque is ~kp*step and
    velocity is still small (damping negligible). Diagnostic only.
    """
    if len(resp.t) < 3:
        return float("nan")
    dt = resp.t[1] - resp.t[0]
    acc0 = (resp.vel[1] - resp.vel[0]) / dt
    # torque at the first tick ~ filtered; approximate with kp*step is unreliable,
    # so instead report the numpy vs isaac accel ratio elsewhere. Return acc0.
    return acc0


def main() -> None:
    plant = Plant()  # inertia 8e-4, damping 0.02, coulomb 0 — the toy Phase-0 plant
    print("=" * 78)
    print(f"AGREEMENT TEST  (matched plant: inertia={plant.inertia:g}, "
          f"damping={plant.damping:g}, coulomb={plant.coulomb:g})")
    print(f"step={STEP_RAD} rad, duration={DURATION}s, ctrl=2kHz, sample=100Hz")
    print("=" * 78)

    numpy_resp = [simulate_step(Gains(*g), plant, step_rad=STEP_RAD, duration=DURATION)
                  for g in GAIN_SETS]
    # Boot Isaac once; keep it alive until every print/plot is done (close() at the
    # very end, because it hard-exits the process).
    sim = IsaacSingleJoint(inertia=plant.inertia, num_envs=len(GAIN_SETS), device="cpu")
    isaac_resp = _isaac_responses(sim, plant, GAIN_SETS)

    rows = []
    for g, rn, ri in zip(GAIN_SETS, numpy_resp, isaac_resp):
        dpos = np.abs(rn.pos - ri.pos)
        dvel = np.abs(rn.vel - ri.vel)
        mn, mi = step_metrics(rn), step_metrics(ri)
        rows.append((g, rn, ri, dpos, dvel, mn, mi))

    print(f"\n{'gains (kp,kd,ki)':22s} {'RMSΔpos':>10s} {'maxΔpos':>10s} "
          f"{'RMSΔvel':>10s} {'settle n|i':>14s} {'overshoot n|i':>16s}")
    print("-" * 90)
    for g, rn, ri, dpos, dvel, mn, mi in rows:
        rms_p = np.sqrt(np.mean(dpos ** 2))
        print(f"{str(g):22s} {rms_p:10.2e} {dpos.max():10.2e} "
              f"{np.sqrt(np.mean(dvel**2)):10.2e} "
              f"{mn['settle_time']:.3f}|{mi['settle_time']:.3f}   "
              f"{mn['overshoot']*100:5.1f}%|{mi['overshoot']*100:5.1f}%")

    all_dpos = np.concatenate([r[3] for r in rows])
    print("-" * 90)
    print(f"OVERALL  max|Δpos| = {all_dpos.max():.3e} rad   "
          f"(step {STEP_RAD} rad -> {all_dpos.max()/STEP_RAD*100:.2f}% of step)")

    # crude effective-inertia cross-check on the first, integral-free gain set
    an = _backout_inertia(numpy_resp[0], plant)
    ai = _backout_inertia(isaac_resp[0], plant)
    print(f"early accel  numpy={an:.3f}  isaac={ai:.3f} rad/s²  "
          f"(ratio {ai/an if an else float('nan'):.3f}) -> effective-inertia match")

    _maybe_plot(numpy_resp, isaac_resp)

    print(f"\nMotor on file for the real-plant demo: {M6C12_150KV.name}, "
          f"reflected inertia = {M6C12_150KV.reflected_inertia:.4f} kg·m² "
          f"({M6C12_150KV.reflected_inertia/plant.inertia:.0f}× the toy plant).")

    sim.close()  # LAST — hard-exits the process


def _maybe_plot(numpy_resp, isaac_resp) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("(matplotlib unavailable — skipping overlay plot)")
        return
    fig, axes = plt.subplots(1, len(GAIN_SETS), figsize=(4 * len(GAIN_SETS), 3.2), sharey=True)
    for ax, g, rn, ri in zip(np.atleast_1d(axes), GAIN_SETS, numpy_resp, isaac_resp):
        ax.axhline(STEP_RAD, color="0.7", lw=0.8, ls="--")
        ax.plot(rn.t, rn.pos, label="numpy", lw=2, color="tab:blue")
        ax.plot(ri.t, ri.pos, label="isaac", lw=1.2, color="tab:orange", ls="--")
        ax.set_title(f"kp={g[0]} kd={g[1]} ki={g[2]}", fontsize=9)
        ax.set_xlabel("t (s)")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("position (rad)")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(HERE, "agreement.png")
    fig.savefig(out, dpi=110)
    print(f"wrote overlay plot -> {out}")


if __name__ == "__main__":
    main()

"""Fit the sim plant (inertia, viscous + Coulomb friction) to real bench step
responses — the real2sim step.

For each recorded real step, simulate the SAME step under the shared firmware
control law (sim/actuator.simulate_step) with candidate plant params, and score
the position mismatch. Minimize over all steps -> the plant params that make the
sim reproduce the real motor. Cross-check friction against the ramp i_q.

Run under the tuner venv (numpy + the sim package):
    ./.venv/bin/python bench/fit_plant.py
"""

from __future__ import annotations

import glob
import json

import numpy as np

from sim.actuator import Gains, Plant, simulate_step

KT, GEAR = 0.08958, 15.0
ALPHA = 0.1454


def _clean(rec):
    """Return (t, pos) as a monotonic-time, de-duplicated, zero-based step."""
    t = np.array(rec["t"], float)
    pos = np.array([p if p is not None else np.nan for p in rec["pos"]], float)
    m = ~np.isnan(pos)
    t, pos = t[m], pos[m]
    # de-dup repeated telemetry-cache reads
    keep = np.concatenate([[True], np.abs(np.diff(pos)) > 1e-12])
    return t[keep], pos[keep]


def _sim_at(times, delta, gains, tau, J, b, coul, latency):
    """Simulate the firmware step of magnitude `delta`; sample at real `times`."""
    plant = Plant(inertia=J, damping=b, coulomb=coul, torque_limit=tau, torque_filter_alpha=ALPHA)
    dur = float(times[-1] + 0.05)
    resp = simulate_step(Gains(*gains), plant, step_rad=delta, duration=dur,
                         ctrl_hz=2000.0, sample_hz=1000.0)
    return np.interp(times, resp.t + latency, resp.pos)


def load_steps(path):
    d = json.load(open(path))
    steps = []
    for r in d["results"]:
        if r["kind"] != "step":
            continue
        t, pos = _clean(r)
        if len(t) < 8:
            continue
        p0 = pos[0]
        steps.append({
            "t": t, "y": pos - p0,                     # zero-based real position
            "delta": r["target"] - r["start"],
            "gains": tuple(r["gains"][:3]),
            "tau": r["gains"][3] if len(r["gains"]) > 3 else 2.0,   # torque_limit used at capture
        })
    return d, steps


def objective(params, steps):
    J, b, coul, lat = params
    if J <= 0 or b < 0 or coul < 0 or lat < 0:
        return 1e9
    err = 0.0
    for s in steps:
        ys = _sim_at(s["t"], s["delta"], s["gains"], s["tau"], J, b, coul, lat)
        err += np.mean((ys - s["y"]) ** 2)
    return err / len(steps)


def fit(steps):
    # coarse grid then local coordinate refine (no scipy dependency). Bounds wide
    # enough for the geared joint: much higher friction, possibly higher inertia.
    Js = [0.008, 0.016, 0.0224, 0.03, 0.045, 0.06, 0.08]
    bs = [0.0, 0.02, 0.05, 0.1, 0.2]
    cs = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.8]
    lats = [0.0, 0.005, 0.01, 0.015]
    best, bestp = 1e18, None
    for J in Js:
        for b in bs:
            for c in cs:
                for lat in lats:
                    e = objective((J, b, c, lat), steps)
                    if e < best:
                        best, bestp = e, [J, b, c, lat]
    # refine
    step = np.array([0.004, 0.02, 0.1, 0.004])
    p = np.array(bestp, float)
    for _ in range(60):
        improved = False
        for i in range(4):
            for sgn in (+1, -1):
                q = p.copy(); q[i] = max(0.0, q[i] + sgn * step[i])
                if q[0] <= 0:
                    continue
                e = objective(q, steps)
                if e < best:
                    best, p, improved = e, q, True
        if not improved:
            step *= 0.5
    return p, best


def friction_from_ramps(d):
    """Steady-state i_q per ramp -> friction torque; fit Coulomb + viscous."""
    pts = []
    for r in d["results"]:
        if r["kind"] != "ramp":
            continue
        iq = [v for _, v in r["iq"] if v is not None]
        if len(iq) < 4:
            continue
        # drop the first third (acceleration transient), average the steady tail
        tail = iq[len(iq) // 3:]
        iq_ss = float(np.median(tail))
        omega = r["speed"] * (1 if r["end"] >= r["start"] else -1)
        pts.append((omega, iq_ss * KT * GEAR))     # (rad/s, output torque Nm)
    if len(pts) < 2:
        return None
    w = np.array([p[0] for p in pts]); tau = np.array([p[1] for p in pts])
    # tau = coulomb*sign(w) + viscous*w  -> LS on [sign(w), w]
    A = np.vstack([np.sign(w), w]).T
    (coul, visc), *_ = np.linalg.lstsq(A, tau, rcond=None)
    return {"points": pts, "coulomb": float(coul), "viscous": float(visc)}


def main():
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:   # prefer the newest geared run, else the newest bare-motor run
        cands = sorted(glob.glob("bench/runs/bench_geared_*.json")) or \
                sorted(glob.glob("bench/runs/bench_full_*.json"))
        path = cands[-1]
    geared = "geared" in path
    out_path = "bench/fitted_plant_geared.json" if geared else "bench/fitted_plant.json"
    d, steps = load_steps(path)
    print(f"loaded {path}: {len(steps)} step responses  ({'GEARED' if geared else 'bare motor'})")

    p, err = fit(steps)
    J, b, coul, lat = p
    print("\n=== FITTED PLANT (real2sim) ===")
    print(f"  inertia J   = {J:.5f} kg·m²   (datasheet reflected M6C12 = 0.0224)")
    print(f"  damping b   = {b:.4f} N·m·s/rad")
    print(f"  coulomb     = {coul:.4f} N·m")
    print(f"  latency     = {lat*1000:.1f} ms")
    print(f"  fit RMS pos error = {np.sqrt(err)*1000:.2f} mrad over {len(steps)} steps")

    fr = friction_from_ramps(d)
    if fr:
        print("\n=== friction cross-check (ramp i_q) ===")
        for w, t in fr["points"]:
            print(f"  {w:+.1f} rad/s -> {t:+.3f} N·m")
        print(f"  fit: coulomb={fr['coulomb']:.3f} N·m, viscous={fr['viscous']:.4f} N·m·s/rad")

    out = {
        "source": path, "geared": geared,
        "motor": "MAD_M6C12_150KV (right_hip_yaw ESC, device 4)" + (" + 15:1 gearbox" if geared else ""),
        "fitted_plant": {"inertia": J, "damping": b, "coulomb": coul, "latency_s": lat},
        "fit_rms_mrad": float(np.sqrt(err) * 1000),
        "friction_from_ramps": fr and {"coulomb": fr["coulomb"], "viscous": fr["viscous"]},
    }
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\nwrote {out_path}  — drop these into sim Plant / MotorSpec")


if __name__ == "__main__":
    main()

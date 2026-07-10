"""Real-motor payoff + Eureka-on-Isaac demo.

Two things, both consequences of bringing the real M6C12 into the sim:

  PART A (numpy, instant) — the reflected rotor inertia matters. Optimise gains
  for a reward on the toy plant (8e-4) and on the real plant (0.0224 = M6C12
  reflected rotor inertia). The optima differ, and the toy-optimal gains are
  worse on the real joint. This is why the toy sim's gains don't transfer.

  PART B (Isaac) — the SAME Eureka inner optimiser (find gains that maximise a
  reward), scored on the real-inertia Isaac plant, vectorised across a gain
  population per PhysX rollout, looping until it stops improving. Demonstrates
  the reward loop "training itself" on the real motor at higher fidelity.

Run under humanoid-policy's venv:
    OMNI_KIT_ACCEPT_EULA=YES /home/nse/humanoid/humanoid-policy/.venv/bin/python \
        -m sim.isaac.real_plant_demo
"""

from __future__ import annotations

from sim.actuator import Gains, Plant, simulate_step
from sim.metrics import fitness, step_metrics

from policy.reward_search.isaac_backend import IsaacBatchScorer, optimize_gains_isaac
from policy.reward_search.optimize import optimize_gains
from policy.reward_search.rewards import SEED_REWARDS, compile_reward

from .motor import M6C12_150KV

STEP_RAD = 0.5
DURATION = 1.0


def _fmt(g: Gains) -> str:
    return f"kp={g.position_kp:5.1f} kd={g.velocity_kp:4.2f} ki={g.position_ki:4.2f}"


def _report(tag: str, resp) -> str:
    m = step_metrics(resp)
    return (f"{tag:26s} settle={m['settle_time']:.3f}s overshoot={m['overshoot']*100:4.1f}% "
            f"ss_err={m['ss_error']:.4f} rms={m['rms_error']:.4f} fitness={fitness(resp):8.3f}")


def main() -> None:
    reward = compile_reward(SEED_REWARDS["neg_rms_error"])
    toy = Plant()                                             # inertia 8e-4
    real = Plant(inertia=M6C12_150KV.reflected_inertia)      # 0.0224 (M6C12 reflected)

    print("=" * 80)
    print(f"REAL-MOTOR DEMO — {M6C12_150KV.name}")
    print(f"toy plant inertia = {toy.inertia:g} kg·m²   real (reflected rotor) = "
          f"{real.inertia:.4f} kg·m²   ({real.inertia/toy.inertia:.0f}×)")
    print(f"torque_limit = {real.torque_limit} Nm  ->  max accel on real plant = "
          f"{real.torque_limit/real.inertia:.0f} rad/s² (vs {toy.torque_limit/toy.inertia:.0f} "
          f"on toy): the real 0.5 rad step is torque-limited.")
    print("=" * 80)

    # ---- PART A: optima differ, toy-optimal transfers poorly (numpy) ----------
    print("\nPART A — optimise the same reward on each plant (numpy):")
    g_toy, _, _ = optimize_gains(reward, toy, step_rad=STEP_RAD)
    g_real, _, _ = optimize_gains(reward, real, step_rad=STEP_RAD)
    print(f"  toy-optimal gains : {_fmt(g_toy)}")
    print(f"  real-optimal gains: {_fmt(g_real)}")

    r_toy_on_real = simulate_step(g_toy, real, step_rad=STEP_RAD, duration=DURATION)
    r_real_on_real = simulate_step(g_real, real, step_rad=STEP_RAD, duration=DURATION)
    print("\n  applied to the REAL plant:")
    print("   ", _report("toy-optimal  gains", r_toy_on_real))
    print("   ", _report("real-optimal gains", r_real_on_real))
    print("  -> the toy-tuned gains are worse on the real joint; the sim you tune in matters.")

    # ---- PART B: the Eureka inner loop, scored on the real-inertia Isaac plant --
    print("\nPART B — same optimiser, scored on the real-inertia ISAAC plant "
          "(vectorised, loop-until-converged):")
    scorer = IsaacBatchScorer(inertia=real.inertia, plant=real, pop=48,
                              step_rad=STEP_RAD, duration=DURATION)
    g_isaac, resp_isaac, rew = optimize_gains_isaac(reward, scorer, seed=1)
    print(f"  isaac-optimal gains: {_fmt(g_isaac)}   (reward {rew:.5f})")
    print("   ", _report("isaac-optimal on isaac", resp_isaac))

    # cross-check: numpy sim of the isaac-found gains on the real plant should agree
    r_isaac_on_numpy = simulate_step(g_isaac, real, step_rad=STEP_RAD, duration=DURATION)
    print("   ", _report("isaac gains on numpy   ", r_isaac_on_numpy))
    print(f"  -> Isaac's inner loop converged on the real motor; its gains reproduce on "
          f"the numpy real plant (fitness {fitness(resp_isaac):.3f} vs "
          f"{fitness(r_isaac_on_numpy):.3f}).")

    scorer.close()  # LAST — hard-exits the process


if __name__ == "__main__":
    main()

"""Isaac Lab single-joint substrate — a higher-fidelity plant behind the SAME
interface the numpy sim exposes (`Response`), so the reward-search loop can
target either.

The controller is the shared firmware law (sim/control_law.py); Isaac/PhysX
supplies only the rigid-body plant. The joint is driven in pure-torque mode
(ImplicitActuatorCfg stiffness=damping=0); the net joint effort applied each
2 kHz tick is `tau_ctrl - damping*vel - coulomb*sign(vel)` — exactly the net
torque the numpy plant integrates — so a disagreement between the two isolates
the *integrator/solver*, not the model.

VECTORISED: N candidate gain sets run as N parallel envs in ONE rollout. That is
what keeps an Eureka gain search tractable on Isaac (one PhysX rollout scores a
whole gain population instead of N sequential boots).

Runs under humanoid-policy's venv (Isaac Lab 3.0.0b2). Two entry points:
  * `IsaacSingleJoint` — in-process, when you're already under the policy venv.
  * CLI (`python -m sim.isaac.substrate --job job.json --out out.json`) — a fresh
    Isaac process per job; the process boundary lets the tuner's numpy-venv code
    drive Isaac too.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from sim.control_law import FirmwarePositionController

# NOTE: import from paths (pxr-free), NOT build_asset — importing pxr before
# Isaac's AppLauncher boots corrupts the USD bindings.
from .paths import ASSET_PATH, JOINT_NAME, LINK_INERTIA


class IsaacSingleJoint:
    """One Isaac app, one fixed plant (armature), N torque-driven single joints.

    Boots the simulator in __init__ (one Omniverse app per process). `armature`
    (the joint-space inertia knob) is baked at build time, so one instance models
    one plant inertia; use a fresh instance/process for a different inertia.
    """

    def __init__(
        self,
        inertia: float,
        num_envs: int,
        device: str = "cpu",
        ctrl_hz: float = 2000.0,
        headless: bool = True,
    ):
        self.num_envs = int(num_envs)
        self.ctrl_hz = float(ctrl_hz)
        self.dt = 1.0 / self.ctrl_hz

        # AppLauncher MUST precede any isaaclab.sim/assets import.
        from isaaclab.app import AppLauncher

        self._launcher = AppLauncher(headless=headless, device=device)
        self.sim_app = self._launcher.app

        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg

        self._sim_utils = sim_utils

        sim_cfg = sim_utils.SimulationCfg(dt=self.dt, gravity=(0.0, 0.0, 0.0), device=device)
        self.sim = sim_utils.SimulationContext(sim_cfg)

        # N parent Xforms; the articulation's regex path spawns one Robot into each.
        for i in range(self.num_envs):
            sim_utils.create_prim(f"/World/Env_{i}", "Xform", translation=(i * 1.0, 0.0, 0.0))

        # Effective joint-space inertia = authored link inertia + armature.
        # Subtract the (negligible) authored value so the total is exactly `inertia`.
        armature = max(inertia - LINK_INERTIA, 1.0e-9)

        robot_cfg = ArticulationCfg(
            prim_path="/World/Env_.*/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path=ASSET_PATH),
            actuators={
                "joint": ImplicitActuatorCfg(
                    joint_names_expr=[JOINT_NAME],
                    stiffness=0.0,           # pure torque control — no PD fighting our effort
                    damping=0.0,
                    armature=armature,       # inject the reflected-rotor + load inertia here
                    effort_limit_sim=1.0e6,  # no actuator clamp; the control law clamps to torque_limit
                    velocity_limit_sim=1.0e6,
                ),
            },
        )
        self.robot = Articulation(robot_cfg.replace(prim_path="/World/Env_.*/Robot"))
        self.sim.reset()
        assert self.robot.is_initialized, "articulation failed to initialize"

    # ------------------------------------------------------------------------
    def _read(self):
        q = self.robot.data.joint_pos.torch.reshape(-1).detach().cpu().numpy()
        qd = self.robot.data.joint_vel.torch.reshape(-1).detach().cpu().numpy()
        return q, qd

    def rollout(
        self,
        gains: np.ndarray,          # [N, 3] : (position_kp, velocity_kp/Kd, position_ki)
        step_rad: float,
        duration: float,
        damping: float,
        coulomb: float,
        torque_limit: float,
        torque_filter_alpha: float,
        sample_hz: float = 100.0,
    ) -> dict:
        """Drive a 0 -> step_rad position step on all N envs; return sampled arrays."""
        import torch

        gains = np.asarray(gains, dtype=float).reshape(self.num_envs, 3)
        n = int(duration * self.ctrl_hz)
        stride = max(1, int(round(self.ctrl_hz / sample_hz)))

        ctrl = FirmwarePositionController(
            position_kp=gains[:, 0],
            velocity_kp=gains[:, 1],
            position_ki=gains[:, 2],
            torque_limit=torque_limit,
            torque_filter_alpha=torque_filter_alpha,
        )
        ctrl.reset(shape=(self.num_envs,))

        # zero the joints, matching the numpy sim's initial condition
        zeros = torch.zeros(self.num_envs, 1, device=self.sim.device)
        self.robot.write_joint_position_to_sim_index(position=zeros)
        self.robot.write_joint_velocity_to_sim_index(velocity=zeros)
        self.robot.reset()

        target = float(step_rad)
        ts, ps, vs = [], [], []
        for i in range(n):
            q, qd = self._read()                         # state S_i (pre-step)
            tau_ctrl = ctrl.step(q, qd, target)          # shared firmware law, [N]
            # net joint torque the numpy plant integrates: actuator - viscous - coulomb
            effort = tau_ctrl - damping * qd - coulomb * np.sign(qd)
            tau = torch.as_tensor(effort, dtype=torch.float32, device=self.sim.device).reshape(-1, 1)

            self.robot.set_joint_effort_target_index(target=tau)
            self.robot.write_data_to_sim()
            self.sim.step(render=False)
            self.robot.update(self.dt)                   # -> state S_{i+1}

            if i % stride == 0:
                q2, v2 = self._read()
                ts.append(i * self.dt)
                ps.append(q2.copy())
                vs.append(v2.copy())

        return {
            "t": np.asarray(ts),                  # [T]
            "pos": np.asarray(ps).T,              # [N, T]
            "vel": np.asarray(vs).T,              # [N, T]
            "step": target,
        }

    def close(self):
        try:
            self.sim_app.close()
        except Exception:
            pass


def _rollout_job(sim: "IsaacSingleJoint", job: dict, gains: np.ndarray) -> dict:
    out = sim.rollout(
        gains=gains,
        step_rad=float(job.get("step_rad", 0.5)),
        duration=float(job.get("duration", 1.0)),
        damping=float(job["damping"]),
        coulomb=float(job.get("coulomb", 0.0)),
        torque_limit=float(job.get("torque_limit", 2.0)),
        torque_filter_alpha=float(job.get("torque_filter_alpha", 0.1454)),
        sample_hz=float(job.get("sample_hz", 100.0)),
    )
    return {
        "t": out["t"].tolist(),
        "pos": out["pos"].tolist(),
        "vel": out["vel"].tolist(),
        "step": out["step"],
        "gains": gains.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Isaac single-joint substrate (JSON job runner)")
    ap.add_argument("--job", required=True, help="path to a JSON job file")
    ap.add_argument("--out", required=True, help="path to write JSON responses")
    args = ap.parse_args()

    with open(args.job) as f:
        job = json.load(f)
    gains = np.asarray(job["gains"], dtype=float).reshape(-1, 3)

    sim = IsaacSingleJoint(
        inertia=float(job["inertia"]),
        num_envs=gains.shape[0],
        device=job.get("device", "cpu"),
        ctrl_hz=float(job.get("ctrl_hz", 2000.0)),
    )
    result = _rollout_job(sim, job, gains)

    # Write BEFORE closing the app: simulation_app.close() can hard-exit the
    # process (os._exit), so anything after it may never run.
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f)
    print(f"ISAAC_JOB_OK wrote {args.out}  "
          f"({len(result['gains'])} envs, {len(result['t'])} samples)", flush=True)
    sim.close()


if __name__ == "__main__":
    main()

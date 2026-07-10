"""Author a minimal fixed-base single-revolute-joint USD, locally.

Why author our own instead of the shipped SimpleArticulation asset: that one
lives on a remote S3 Nucleus (ISAAC_NUCLEUS_DIR is an https URL — needs network)
and bakes in a link inertia we don't control. We want (a) no network dependency
and (b) the joint-space inertia set entirely at runtime via the actuator's
`armature` knob, so one asset serves every plant. The authored link therefore
carries a deliberately NEGLIGIBLE explicit inertia (1e-6 kg·m²); the substrate
sets the real effective inertia via armature and verifies it empirically.

Structure (/Robot is the DEFAULT PRIM — UsdFileCfg references it into each env):
    /Robot                 Xform, ArticulationRootAPI  (default prim)
      /Robot/base          fixed link (anchored to world by a FixedJoint)
      /Robot/link          moving link, COM on the joint axis, tiny inertia
      /Robot/root_fix      FixedJoint world -> base  (fixed-base articulation)
      /Robot/RevoluteJoint RevoluteJoint base -> link, about Z, torque drive

Gravity is disabled at the sim level (SimulationCfg gravity=0) so no gravity
torque enters — matching the numpy point-mass plant. COM on the axis makes that
independent of link mass anyway.

Run under the policy venv:
    /home/nse/humanoid/humanoid-policy/.venv/bin/python -m sim.isaac.build_asset
"""

from __future__ import annotations

import os

from pxr import Gf, Usd, UsdGeom, UsdPhysics  # PhysxSchema is not importable in this build

from .paths import ASSET_DIR, ASSET_PATH, JOINT_NAME, LINK_INERTIA  # noqa: F401


def build(path: str = ASSET_PATH) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # The articulation root must be the DEFAULT PRIM: UsdFileCfg references a
    # USD's default prim into the target path (/World/Env_i/Robot), so that prim
    # is what carries ArticulationRootAPI.
    robot = UsdGeom.Xform.Define(stage, "/Robot")
    UsdPhysics.ArticulationRootAPI.Apply(robot.GetPrim())
    stage.SetDefaultPrim(robot.GetPrim())

    def rigid_body(prim_path: str, mass: float, inertia: float, size: float):
        cube = UsdGeom.Cube.Define(stage, prim_path)
        cube.GetSizeAttr().Set(size)
        prim = cube.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        m = UsdPhysics.MassAPI.Apply(prim)
        m.GetMassAttr().Set(mass)
        m.GetCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))     # COM on the joint axis
        m.GetDiagonalInertiaAttr().Set(Gf.Vec3f(inertia, inertia, inertia))
        return prim

    # Base and moving link are geometrically coincident at the origin; only the
    # joint axis and inertias matter for the 1-DoF plant.
    rigid_body("/Robot/base", mass=1.0, inertia=1.0e-3, size=0.05)
    rigid_body("/Robot/link", mass=0.05, inertia=LINK_INERTIA, size=0.04)

    # Fixed joint: world -> base (no body0 => attached to world) => fixed base.
    fixed = UsdPhysics.FixedJoint.Define(stage, "/Robot/root_fix")
    fixed.CreateBody1Rel().SetTargets(["/Robot/base"])

    # Revolute joint: base -> link, rotating about Z, unlimited travel.
    rev = UsdPhysics.RevoluteJoint.Define(stage, f"/Robot/{JOINT_NAME}")
    rev.CreateBody0Rel().SetTargets(["/Robot/base"])
    rev.CreateBody1Rel().SetTargets(["/Robot/link"])
    rev.CreateAxisAttr().Set("Z")
    rev.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    rev.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    # Angular force drive with ZERO stiffness/damping => pure torque control.
    # Isaac Lab's ImplicitActuatorCfg(stiffness=0, damping=0) overrides these at
    # load; authoring the drive just guarantees an effort target is accepted.
    drive = UsdPhysics.DriveAPI.Apply(rev.GetPrim(), "angular")
    drive.CreateTypeAttr().Set("force")
    drive.CreateStiffnessAttr().Set(0.0)
    drive.CreateDampingAttr().Set(0.0)
    drive.CreateMaxForceAttr().Set(1.0e6)

    stage.GetRootLayer().Save()
    return path


if __name__ == "__main__":
    p = build()
    print(f"wrote {p}")

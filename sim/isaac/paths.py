"""Asset paths and constants shared by build_asset.py and substrate.py.

Kept pxr-free ON PURPOSE: importing `pxr` before Isaac's AppLauncher boots
corrupts the USD python bindings, so substrate.py (which boots Isaac) must reach
these constants WITHOUT importing build_asset.py (which imports pxr).
"""

from __future__ import annotations

import os

ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")
ASSET_PATH = os.path.join(ASSET_DIR, "single_joint.usd")

# Negligible explicit inertia authored on the moving link; the real effective
# joint-space inertia is injected at runtime via the actuator's `armature`.
LINK_INERTIA = 1.0e-6   # kg·m²
JOINT_NAME = "RevoluteJoint"

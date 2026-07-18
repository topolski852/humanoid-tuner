#!/usr/bin/env python3
"""READ-ONLY: dump the ESC's actual live config + mode/state. We need the FRESH
calibration values (electrical_offset especially) so that if we connect_single
(apply_config) to enter position mode, we push the SAME values and never clobber
the new encoder calibration."""
from __future__ import annotations
import asyncio, json, sys
sys.path.insert(0, "/home/nse/humanoid/humanoid-studio/backend")
from humanoid.daemon_client import DaemonClient
from humanoid.robot_config import RobotConfig

CFG = "/home/nse/humanoid/humanoid-studio/configs/bench_shoulder_roll.json"
JOINT = "right_shoulder_roll_joint"


async def main():
    cfg = RobotConfig.from_json(CFG)
    c = DaemonClient(cfg)
    await c.start()
    await asyncio.sleep(0.5)

    st = c.get_cached_joint_state(JOINT) or {}
    print(f"state={st.get('state')}  mode={st.get('mode')}  pos={st.get('position')}  "
          f"error=0x{int(st.get('error',0)):04X}  bus={st.get('bus_voltage')}")

    print("\nREAD_CONFIG (ESC live params) ...")
    loop = asyncio.get_running_loop()
    try:
        conf = await loop.run_in_executor(None, c.read_device_config, JOINT)
        for k in sorted(conf):
            print(f"  {k:32s} {conf[k]}")
        # save so we can build a faithful bench config from it
        json.dump(conf, open("/home/nse/humanoid/humanoid-tuner/bench/runs/esc_config_roll.json", "w"), indent=2)
        print("\nsaved -> bench/runs/esc_config_roll.json")
    except Exception as e:
        print(f"  READ_CONFIG failed: {e}")

    await c.stop()


if __name__ == "__main__":
    asyncio.run(main())

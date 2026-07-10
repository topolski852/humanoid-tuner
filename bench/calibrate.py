#!/usr/bin/env python3
"""Flux/encoder calibration for ONE bench motor, via the Humanoid-Studio daemon.

The tuner never opens CAN. This speaks UDP to the Studio C++ daemon, which runs
the firmware's autonomous encoder flux-offset sweep (MODE_CALIBRATION). Two steps,
so we energize in stages:

    check      wake the motor to IDLE (powerstage on, ZERO torque, NO motion) and
               read firmware version / bus voltage / mode / error — a liveness +
               power gate. Safe: the motor does not move.
    calibrate  everything `check` does, then run the flux sweep. The rotor SPINS
               through several electrical revolutions for ~15 s. Requires --yes.
               On success, reads the new offset, stores to flash, saves the config.

Prereqs (see bench/CALIBRATION.md):
    1) can0 up at 1 Mbit:  sudo ip link set can0 type can bitrate 1000000 && \
                           sudo ip link set can0 up
    2) daemon running with the bench config:
       humanoid-studio/daemon/build/humanoid_daemon \
           --config humanoid-studio/configs/bench_right_hip_roll.json --tel-hz 100

Usage:
    python3 bench/calibrate.py check
    python3 bench/calibrate.py calibrate --yes
"""

from __future__ import annotations

import argparse
import asyncio
import sys

# The tuner reuses Studio's DaemonClient; make its backend importable.
STUDIO_BACKEND = "/home/nse/humanoid/humanoid-studio/backend"
sys.path.insert(0, STUDIO_BACKEND)

from humanoid.daemon_client import DaemonClient  # noqa: E402
from humanoid.robot_config import RobotConfig    # noqa: E402

DEFAULT_CONFIG = "/home/nse/humanoid/humanoid-studio/configs/bench_right_hip_roll.json"

# SDO param offsets (HANDOFF §3)
PARAM_FIRMWARE_VERSION = 0x004
PARAM_ERROR = 0x014
PARAM_BUS_VOLTAGE = 0x100
PARAM_FLUX_OFFSET = 0x13C


def _fw(u32: int | None) -> str:
    if not u32:
        return "?"
    return f"v{(u32 >> 24) & 0xFF}.{(u32 >> 16) & 0xFF}.{(u32 >> 8) & 0xFF}"


async def main() -> int:
    ap = argparse.ArgumentParser(description="Single-motor flux calibration via the Studio daemon")
    ap.add_argument("action", choices=["check", "calibrate"])
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--joint", default="right_hip_roll_joint")
    ap.add_argument("--timeout", type=float, default=90.0, help="calibration timeout (s)")
    ap.add_argument("--yes", action="store_true",
                    help="required for 'calibrate' — confirms the motor may SPIN")
    ap.add_argument("--min-vbus", type=float, default=10.0,
                    help="abort if bus voltage is below this (motor supply must be present)")
    args = ap.parse_args()

    cfg = RobotConfig.from_json(args.config)
    if args.joint not in cfg.joints:
        print(f"✗ joint {args.joint!r} not in {args.config}")
        return 2
    jc = cfg.joints[args.joint]
    chan, dev = jc.can_channel, jc.can_id

    client = DaemonClient(cfg)
    await client.start()
    try:
        print(f"[1] Waking {args.joint} (device {dev} on {chan}) to IDLE "
              f"— powerstage on, zero torque, no motion ...")
        client.connect_single(args.joint)     # NMT IDLE + apply config (sets max_calibration_current)
        await asyncio.sleep(0.4)

        fw = client.generic_sdo_read(chan, dev, PARAM_FIRMWARE_VERSION)
        if fw is None:
            print("  ✗ no SDO response — motor is OFFLINE.")
            print("    Check: can0 up at 1 Mbit, device_id, CAN wiring/termination, ESC logic power.")
            return 3
        vb = client.generic_sdo_read(chan, dev, PARAM_BUS_VOLTAGE)
        er = client.generic_sdo_read(chan, dev, PARAM_ERROR)
        fx = client.generic_sdo_read(chan, dev, PARAM_FLUX_OFFSET)
        vbus = vb["value_f32"] if vb else float("nan")
        errcode = er["value_u32"] if er else None
        old_flux = fx["value_f32"] if fx else None

        print(f"  firmware      : {_fw(fw['value_u32'])}  (0x{fw['value_u32']:08X})")
        print(f"  bus voltage   : {vbus:.1f} V")
        print(f"  error bitmask : " + (f"0x{errcode:04X}" if errcode is not None else "?"))
        print(f"  flux offset   : " + (f"{old_flux:.4f} rad (current)" if old_flux is not None else "?"))

        if errcode:
            print(f"  ⚠ firmware error 0x{errcode:04X} set — clearing.")
            client.clear_error(args.joint)

        if not (vbus == vbus) or vbus < args.min_vbus:
            print(f"  ✗ bus voltage {vbus:.1f} V < {args.min_vbus} V — the 12V+ motor supply is not "
                  "present. Flux calibration needs it (it drives current to spin the rotor). Aborting.")
            return 4

        if args.action == "check":
            print("[✓] check OK — motor alive and powered, no motion performed.")
            print("    Next: python3 bench/calibrate.py calibrate --yes")
            return 0

        # ---- calibrate ----
        if not args.yes:
            print("[!] Refusing to run calibration without --yes.")
            print("    The rotor will SPIN through several electrical revolutions for ~15 s.")
            print("    Ensure the output shaft is free/secured, then re-run with --yes.")
            return 5

        print(f"[2] Running flux calibration — MOTOR WILL SPIN (~15 s, {args.timeout:.0f}s timeout) ...")
        proxy = client.get_actuator_by_name(args.joint)
        new_flux = await proxy.calibrate_offset(timeout=args.timeout)
        if old_flux is not None:
            print(f"  ✓ calibration OK. flux offset: {old_flux:.4f} -> {new_flux:.4f} rad")
        else:
            print(f"  ✓ calibration OK. flux offset = {new_flux:.4f} rad")

        print("[3] Storing to flash ...")
        await proxy.store_to_flash()
        await asyncio.sleep(0.3)
        verify = client.generic_sdo_read(chan, dev, PARAM_FLUX_OFFSET)
        if verify:
            print(f"  readback after store: {verify['value_f32']:.4f} rad")

        cfg.joints[args.joint] = proxy.config      # capture new electrical_offset
        cfg.to_json(args.config)
        print(f"[4] Saved new electrical_offset to {args.config}")
        print("[✓] Calibration complete. Motor left in IDLE.")
        return 0
    finally:
        try:
            client.set_mode(args.joint, "IDLE")
        except Exception:
            pass
        await client.stop()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

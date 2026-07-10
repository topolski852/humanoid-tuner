# Bench flux calibration — right_hip_roll (MAD M6C12) on can0

Recalibrate the encoder flux offset after the encoder was moved. The tuner never
opens CAN — everything goes through the Humanoid-Studio daemon over UDP. The
firmware runs an autonomous sweep (`MODE_CALIBRATION`): ramp voltage → sweep
forward → sweep backward → compute offset → store. **The rotor spins for ~15 s.**

## Prereqs (verified present)
- Daemon built: `humanoid-studio/daemon/build/humanoid_daemon`
- Bench config: `humanoid-studio/configs/bench_right_hip_roll.json` (can0, id 2,
  Kt 0.08958, max_calibration_current 5.0 A; old flux offset 72.53 rad = will be
  overwritten)
- Driver: `bench/calibrate.py` (system python3; imports Studio's DaemonClient)

## Step 1 — bring up can0 (needs sudo; run this yourself)
```bash
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 txqueuelen 1000
sudo ip link set can0 up
ip -brief link show can0        # expect: can0  UP
```

## Step 2 — start the daemon (non-root OK; SCHED_OTHER)
```bash
cd /home/nse/humanoid/humanoid-studio/daemon
./build/humanoid_daemon \
  --config /home/nse/humanoid/humanoid-studio/configs/bench_right_hip_roll.json \
  --tel-hz 100 --rt-prio 0 &
```

## Step 3 — liveness check (SAFE: wakes to IDLE, zero torque, NO motion)
```bash
cd /home/nse/humanoid/humanoid-tuner
python3 bench/calibrate.py check
```
Confirms firmware version, bus voltage (must be > 10 V — the 12 V+ motor supply),
error bitmask, and the current flux offset. Aborts if the motor is OFFLINE or the
motor bus is unpowered.

## Step 4 — calibrate (MOTOR SPINS ~15 s). Ensure the shaft is free/secured.
```bash
python3 bench/calibrate.py calibrate --yes
```
On success it prints old→new flux offset, stores to flash, and writes the new
`electrical_offset` back into the bench config.

## Safety
- ESTOP: the daemon listens on UDP 9002; `DaemonClient.estop_all()` drives IDLE.
- The driver always leaves the motor in IDLE on exit.
- To stop everything: `kill %1` (daemon), `sudo ip link set can0 down`.

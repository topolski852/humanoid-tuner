# policy/

RL training code and trained artifacts for the tuner.

- **Action:** gain vector / bounded delta (start with `position_kp, velocity_kp(Kd),
  position_ki`).
- **Observation:** recent response-feature history + current gains + (online) action
  history + the reward-weight conditioning inputs (the "responsiveness ↔ compliance"
  slider).
- **Reward:** post-action-window feature improvement (see `docs/DESIGN.md §4`).
- **Phase 1+ architecture:** RMA-style history-encoder → latent → policy, so the net
  infers hidden load/friction and adapts online.

Artifacts (`*.pt`/`*.onnx`) are gitignored; track them however `humanoid-policy` does.

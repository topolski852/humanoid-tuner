# deploy/

Runtime that runs a trained tuner on real hardware and/or plugs into Humanoid-Studio.
**Later** — nothing here until a policy exists (Phase 3).

Open question (`docs/DESIGN.md §6`): does the tuner run as a daemon UDP client the Studio
app spawns, or ship as a library the backend imports? Mirror however `humanoid-control`
(the learned-policy runner) plugs in at the daemon boundary.

Deployment uses online **adaptation** of a sim-trained policy (RMA latent), not online
learning-from-scratch. Same safety envelope as `bench/` (gain clamps, rate limits,
runaway detector) applies here — with real hardware on the line.

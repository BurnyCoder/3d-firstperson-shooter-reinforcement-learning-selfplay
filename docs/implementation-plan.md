# Implementation and experiment acceptance plan

The original implementation request includes the later additions: watching a trained bot through its eyes and taking over that exact bot without resetting the round. This checklist records delivery status; the [protocol](protocol.md) specifies scientific decisions, and the [architecture](architecture.md) specifies game interfaces.

- [x] One authoritative deterministic simulation for 5v5 bomb rounds, training, evaluation, rendering and replay.
- [x] Human first-person controls, rifle combat, ammunition/reload, two sites, planting/defusing, timers and scoreboard.
- [x] Three launcher modes: watch the arena, watch a bot in first person, and play in first person against bots.
- [x] Spectator Tab selects a living bot; V switches camera; F takes over or hands control back while preserving the live round and player state.
- [x] Trained checkpoint selection applies to both spectator modes and human play; bot takeover uses the same policy when control is returned.
- [x] TorchRL MAPPO, visibility-limited actor observations, centralized critic, historical frozen opponents, fixed-opponent control, death masking and resumable checkpoints.
- [x] Unit/integration tests, native visual checks, exact rendered/headless replay checks and real CPU/CUDA train/save/evaluate/reload workflows.
- [x] Public source repository, merged core feature PR, pre-specified experiment protocol and measured smoke reports/PDF.
- [ ] Complete the six independent training runs and all initial/final tournaments within the total compute allowance.
- [ ] Publish final per-run reports, aggregate uncertainty analysis and sourced PDF; retain failed and negative outcomes.
- [ ] Publish all six model endpoints, initial checkpoints, configurations and metrics in the public Hugging Face collection.

The pilot does not promise statistical significance or strong bots. Its acceptance criterion is an executable, measured, reproducible workflow with complete disclosure of results. Learning strength is an empirical outcome.

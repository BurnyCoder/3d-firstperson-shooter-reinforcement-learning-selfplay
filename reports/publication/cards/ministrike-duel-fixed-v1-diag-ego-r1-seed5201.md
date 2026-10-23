---
license: mit
library_name: pytorch
tags:
- reinforcement-learning
- self-play
- torchrl
- experimental
---
# MiniStrike fixed-spawn duel, seed 5201

Author: Libor Burian. **Experimental research checkpoint** for an original 1v1 elimination game.
**Diagnostic only: never use this model as the deployment checkpoint.**
Source revision: [`6149c2ddef0c987c08005544a855aa00c60f329f`](https://github.com/BurnyCoder/3d-firstperson-shooter-reinforcement-learning-selfplay/tree/6149c2ddef0c987c08005544a855aa00c60f329f). Registration: `A-DIAG-EGO-R1`.

## Method and architecture

Random initialization, historical self-play only: no human demonstrations, imitation learning,
pretrained gameplay policies or scripted training opponents. The actor is a tanh MLP with two
128-unit hidden layers, **192** inputs (four visibility-limited 48-feature frames) and **five categorical**
controls: movement, yaw, pitch, firing and reload. A separate critic has two 256-unit tanh hidden
layers and receives the active learner team's observations; for 1v1 this is one player's input.
TorchRL supplies PPO loss, collection and GAE. Opponents are immutable learned snapshots,
sampled 50% latest / 50% history at round boundaries, with the initial policy retained permanently.
Actor preprocessing is **egocentric-v1**: egocentric-v1 rotates already-visible
relative coordinates into each observation frame's right/up/forward basis; identity retains world axes.
The critic retains original coordinates. No extra hidden information, aiming or navigation is supplied.

Reward `R1` and all coefficients are engineering choices in the
[active protocol](https://github.com/BurnyCoder/3d-firstperson-shooter-reinforcement-learning-selfplay/blob/6149c2ddef0c987c08005544a855aa00c60f329f/docs/duel-protocol.md), not published optima. Exact
optimizer/rollout/hardware settings are in `config.json`. Training recorded **4980736 arena steps**,
**608 updates**; full allocated step budget completed: **True**.

## Measurements and limitations

Final-test status: **pending**. Equal-opponent/side round score:
**pending**. Baseline breakdown: `{}`.
Elimination fraction: **pending**.
Final validation summary (selection evidence, not final testing): `{"baselines": {"initial": {"draws": 0, "losses": 0, "matches": 2, "score": 1.0, "side_scores": [1.0, 1.0], "wins": 2}, "random": {"draws": 0, "losses": 0, "matches": 256, "score": 1.0, "side_scores": [1.0, 1.0], "wins": 256}, "scripted": {"draws": 0, "losses": 0, "matches": 2, "score": 1.0, "side_scores": [1.0, 1.0], "wins": 2}}, "elimination_fraction": 1.0, "score": 1.0}`.
Win/draw/loss score is 1/0.5/0 against random, corresponding initial and moving quantized scripted
evaluation controls. Raw paired worlds and identities are in `initial_test.json` and `final_test.json`
when available; stochastic behavior is separately labeled. Validation files concern model selection,
not held-out final testing. Missing measurements are unavailable, not evidence of success.

Release assessment: `{"final_test_not_accessed_for_this_model": true, "independent_seeds": 1, "later_confirmation_used_fresh_weights": true, "status": "diagnostic research archive only; never deploy"}`. The full confirmation design uses
**three independent** models. Many matches of one model do not establish retraining reliability;
deterministic fixed-layout trajectories are recorded once per spawn end. Neither this task nor
these observations establish random-spawn, tactical-map, bomb-game or human-level competence.
Failures and negative results remain public in the [experiment reports](https://github.com/BurnyCoder/3d-firstperson-shooter-reinforcement-learning-selfplay/tree/main/reports).

## Use and reproducibility

Install the source repository's locked `uv` environment, then select this downloaded `model.pt`:

```powershell
uv run csrl watch --experiment duel-fixed-v1 --first-person --checkpoint PATH/model.pt
uv run csrl play --experiment duel-fixed-v1 --checkpoint PATH/model.pt
```

Tab selects a living bot; V/F1 switches spectator camera; F/F2 takes over or hands control back without
resetting the round. The arena is 14 × 14 metres, with fixed starts six metres apart, 20 Hz simulation,
20-second rounds, no respawns/objectives, and simultaneous lethal shots/timeouts drawing.
`initial.pt` is the untouched model baseline; `model.pt` is the recorded endpoint. Checkpoints
contain optimizer/environment/RNG state and use the project's restricted PyTorch loader.
Checkpoint **SHA256** identities: `{"initial.pt": "173f7fdefce9de1bc4cc4c242f3da4e4e26c8de2b97b4c8ddd825c36394157a1", "model.pt": "79665c155698f57de9cda2ddd38bcaed4fc2e2423abbdf97561384be9ad74002"}`.

Primary method references: [TorchRL PPO/GAE](https://docs.pytorch.org/rl/stable/tutorials/multiagent_ppo.html),
[Sample Factory Doom self-play inspiration](https://proceedings.mlr.press/v119/petrenko20a/petrenko20a.pdf),
[potential shaping](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf),
[independent-seed power](https://arxiv.org/abs/1806.08295), and
[evaluation uncertainty](https://arxiv.org/abs/2108.13264). Library versions and all design limits
are recorded in the source lockfile, protocol and [bibliography](https://github.com/BurnyCoder/3d-firstperson-shooter-reinforcement-learning-selfplay/blob/main/docs/sources.md).

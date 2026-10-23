---
license: mit
library_name: pytorch
tags:
- reinforcement-learning
- self-play
- torchrl
- experimental
---
# MiniStrike fixed-spawn duel, seed 5102

Author: Libor Burian. **Experimental research checkpoint** for an original 1v1 elimination game.
**Engineering acceptance is not implied by completing training.**
Source revision: [`64c366cd43c9739d3596712a4dd277675ecea8aa`](https://github.com/BurnyCoder/3d-firstperson-shooter-reinforcement-learning-selfplay/tree/64c366cd43c9739d3596712a4dd277675ecea8aa). Registration: `A-CONFIRM`.

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
[active protocol](https://github.com/BurnyCoder/3d-firstperson-shooter-reinforcement-learning-selfplay/blob/64c366cd43c9739d3596712a4dd277675ecea8aa/docs/duel-protocol.md), not published optima. Exact
optimizer/rollout/hardware settings are in `config.json`. Training recorded **5464064 arena steps**,
**667 updates**; full allocated step budget completed: **True**.

## Measurements and limitations

Final-test status: **complete**. Equal-opponent/side round score:
**0.7473958333333334**. Baseline breakdown: `{"initial": {"draws": 1, "losses": 0, "matches": 2, "score": 0.75, "side_scores": [1.0, 0.5], "wins": 1}, "random": {"draws": 4, "losses": 0, "matches": 256, "score": 0.9921875, "side_scores": [1.0, 0.984375], "wins": 252}, "scripted": {"draws": 0, "losses": 1, "matches": 2, "score": 0.5, "side_scores": [1.0, 0.0], "wins": 1}}`.
Elimination fraction: **0.9807692307692307**.
Final validation summary (selection evidence, not final testing): `{"baselines": {"initial": {"draws": 1, "losses": 0, "matches": 2, "score": 0.75, "side_scores": [1.0, 0.5], "wins": 1}, "random": {"draws": 3, "losses": 0, "matches": 256, "score": 0.994140625, "side_scores": [1.0, 0.98828125], "wins": 253}, "scripted": {"draws": 0, "losses": 1, "matches": 2, "score": 0.5, "side_scores": [1.0, 0.0], "wins": 1}}, "elimination_fraction": 0.9846153846153847, "score": 0.748046875}`.
Win/draw/loss score is 1/0.5/0 against random, corresponding initial and moving quantized scripted
evaluation controls. Raw paired worlds and identities are in `initial_test.json` and `final_test.json`
when available; stochastic behavior is separately labeled. Validation files concern model selection,
not held-out final testing. Missing measurements are unavailable, not evidence of success.

Release assessment: `{"final_test_access_after_frozen_validation_selection": true, "independent_model_ci95": [0.4121300290211585, 0.41859913764550816], "mean_primary_improvement": 0.4153645833333333, "model_count": 3, "report": "https://github.com/BurnyCoder/3d-firstperson-shooter-reinforcement-learning-selfplay/blob/main/reports/duel-fixed-v1/confirmation/README.md", "selected_for_experimental_viewing": false, "selection_sha256": "ff0773ca006c05bf30282eabba4287aeebbb78d74f0be1c4dc16d8030dc39146", "status": "experimental; engineering acceptance failed", "unmet_criteria": ["median moving-script score 0.25 < 0.55"]}`. The full confirmation design uses
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
Checkpoint **SHA256** identities: `{"initial.pt": "4167b41f716c53afb394e556b68115e6b5a4056c28e71fd10dbac44696f3fcb2", "model.pt": "66a3e6342a79eed81e5d92ff3a5c3c588e1b59fa0e2226c63b7376d1f03ac1a1"}`.

Primary method references: [TorchRL PPO/GAE](https://docs.pytorch.org/rl/stable/tutorials/multiagent_ppo.html),
[Sample Factory Doom self-play inspiration](https://proceedings.mlr.press/v119/petrenko20a/petrenko20a.pdf),
[potential shaping](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf),
[independent-seed power](https://arxiv.org/abs/1806.08295), and
[evaluation uncertainty](https://arxiv.org/abs/2108.13264). Library versions and all design limits
are recorded in the source lockfile, protocol and [bibliography](https://github.com/BurnyCoder/3d-firstperson-shooter-reinforcement-learning-selfplay/blob/main/docs/sources.md).

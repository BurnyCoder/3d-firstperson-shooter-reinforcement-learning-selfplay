---
license: mit
library_name: pytorch
tags:
- reinforcement-learning
- self-play
- torchrl
- experimental
---
# MiniStrike fixed-spawn duel, seed 5103

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
**0.7486979166666666**. Baseline breakdown: `{"initial": {"draws": 0, "losses": 0, "matches": 2, "score": 1.0, "side_scores": [1.0, 1.0], "wins": 2}, "random": {"draws": 2, "losses": 0, "matches": 256, "score": 0.99609375, "side_scores": [0.9921875, 1.0], "wins": 254}, "scripted": {"draws": 1, "losses": 1, "matches": 2, "score": 0.25, "side_scores": [0.0, 0.5], "wins": 0}}`.
Elimination fraction: **0.9923076923076923**.
Final validation summary (selection evidence, not final testing): `{"baselines": {"initial": {"draws": 0, "losses": 0, "matches": 2, "score": 1.0, "side_scores": [1.0, 1.0], "wins": 2}, "random": {"draws": 1, "losses": 0, "matches": 256, "score": 0.998046875, "side_scores": [0.99609375, 1.0], "wins": 255}, "scripted": {"draws": 1, "losses": 1, "matches": 2, "score": 0.25, "side_scores": [0.0, 0.5], "wins": 0}}, "elimination_fraction": 0.9961538461538462, "score": 0.7493489583333334}`.
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
Checkpoint **SHA256** identities: `{"initial.pt": "d1ee259e42a78a1e77af6d468a567be5fbfce86394d9efdff8bdc4f9eae0672d", "model.pt": "a77e9541e91ec40a8e76d746a0d61c2e1b84bea08099dc679417d2c82659abdd"}`.

Primary method references: [TorchRL PPO/GAE](https://docs.pytorch.org/rl/stable/tutorials/multiagent_ppo.html),
[Sample Factory Doom self-play inspiration](https://proceedings.mlr.press/v119/petrenko20a/petrenko20a.pdf),
[potential shaping](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf),
[independent-seed power](https://arxiv.org/abs/1806.08295), and
[evaluation uncertainty](https://arxiv.org/abs/2108.13264). Library versions and all design limits
are recorded in the source lockfile, protocol and [bibliography](https://github.com/BurnyCoder/3d-firstperson-shooter-reinforcement-learning-selfplay/blob/main/docs/sources.md).

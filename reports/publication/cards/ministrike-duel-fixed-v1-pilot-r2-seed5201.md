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
**Engineering acceptance is not implied by completing training.**
Source revision: [`e0a2420e2b57e10b5ccf1448d91d8ecf967a8a1c`](https://github.com/BurnyCoder/3d-firstperson-shooter-reinforcement-learning-selfplay/tree/e0a2420e2b57e10b5ccf1448d91d8ecf967a8a1c). Registration: `A-PILOT-R2`.

## Method and architecture

Random initialization, historical self-play only: no human demonstrations, imitation learning,
pretrained gameplay policies or scripted training opponents. The actor is a tanh MLP with two
128-unit hidden layers, **192** inputs (four visibility-limited 48-feature frames) and **five categorical**
controls: movement, yaw, pitch, firing and reload. A separate critic has two 256-unit tanh hidden
layers and receives the active learner team's observations; for 1v1 this is one player's input.
TorchRL supplies PPO loss, collection and GAE. Opponents are immutable learned snapshots,
sampled 50% latest / 50% history at round boundaries, with the initial policy retained permanently.
Actor preprocessing is **identity**: egocentric-v1 rotates already-visible
relative coordinates into each observation frame's right/up/forward basis; identity retains world axes.
The critic retains original coordinates. No extra hidden information, aiming or navigation is supplied.

Reward `R2` and all coefficients are engineering choices in the
[active protocol](https://github.com/BurnyCoder/3d-firstperson-shooter-reinforcement-learning-selfplay/blob/e0a2420e2b57e10b5ccf1448d91d8ecf967a8a1c/docs/duel-protocol.md), not published optima. Exact
optimizer/rollout/hardware settings are in `config.json`. Training recorded **1245184 arena steps**,
**152 updates**; full allocated step budget completed: **True**.

## Measurements and limitations

Final-test status: **pending**. Equal-opponent/side round score:
**pending**. Baseline breakdown: `{}`.
Elimination fraction: **pending**.
Final validation summary (selection evidence, not final testing): `{"baselines": {"initial": {"draws": 2, "losses": 0, "matches": 2, "score": 0.5, "side_scores": [0.5, 0.5], "wins": 0}, "random": {"draws": 175, "losses": 0, "matches": 256, "score": 0.658203125, "side_scores": [0.73828125, 0.578125], "wins": 81}, "scripted": {"draws": 0, "losses": 2, "matches": 2, "score": 0.0, "side_scores": [0.0, 0.0], "wins": 0}}, "elimination_fraction": 0.3192307692307692, "score": 0.3860677083333333}`.
Win/draw/loss score is 1/0.5/0 against random, corresponding initial and moving quantized scripted
evaluation controls. Raw paired worlds and identities are in `initial_test.json` and `final_test.json`
when available; stochastic behavior is separately labeled. Validation files concern model selection,
not held-out final testing. Missing measurements are unavailable, not evidence of success.

Release assessment: `{"assessment": {"assessed_utc": "20261023T004432.411661Z", "complete_equal_budget": true, "diagnostics": {"final": {"initial": {"aim_saturation": 0.0, "damage_per_round": 0.0, "hits_per_shot": 0.0, "reload_occupancy": 0.0, "seconds_per_round": 20.0, "shots_per_round": 30.0}, "random": {"aim_saturation": 0.0, "damage_per_round": 56.73828125, "hits_per_shot": 0.08114525139664805, "reload_occupancy": 0.07694813656502476, "seconds_per_round": 14.98828125, "shots_per_round": 27.96875}, "scripted": {"aim_saturation": 0.0, "damage_per_round": 0.0, "hits_per_shot": 0.0, "reload_occupancy": 0.0, "seconds_per_round": 0.7250000000000001, "shots_per_round": 5.0}}, "initial": {"initial": {"aim_saturation": 0.9475, "damage_per_round": 0.0, "hits_per_shot": 0.0, "reload_occupancy": 0.44625, "seconds_per_round": 20.0, "shots_per_round": 25.0}, "random": {"aim_saturation": 0.947275390625, "damage_per_round": 9.1796875, "hits_per_shot": 0.014205833459271573, "reload_occupancy": 0.437333984375, "seconds_per_round": 20.0, "shots_per_round": 25.84765625}, "scripted": {"aim_saturation": 0.4, "damage_per_round": 0.0, "hits_per_shot": 0.0, "reload_occupancy": 0.32857142857142857, "seconds_per_round": 1.75, "shots_per_round": 7.5}}}, "final": {"baselines": {"initial": {"draws": 2, "losses": 0, "matches": 2, "score": 0.5, "side_scores": [0.5, 0.5], "wins": 0}, "random": {"draws": 175, "losses": 0, "matches": 256, "score": 0.658203125, "side_scores": [0.73828125, 0.578125], "wins": 81}, "scripted": {"draws": 0, "losses": 2, "matches": 2, "score": 0.0, "side_scores": [0.0, 0.0], "wins": 0}}, "elimination_fraction": 0.3192307692307692, "score": 0.3860677083333333}, "final_checkpoint_sha256": "495446a1142be875a43a6012377af4daa062a7e293b0d141e0c8c8b5ae1d6d79", "improvement": 0.052734375, "initial": {"baselines": {"initial": {"draws": 2, "losses": 0, "matches": 2, "score": 0.5, "side_scores": [0.5, 0.5], "wins": 0}, "random": {"draws": 256, "losses": 0, "matches": 256, "score": 0.5, "side_scores": [0.5, 0.5], "wins": 0}, "scripted": {"draws": 0, "losses": 2, "matches": 2, "score": 0.0, "side_scores": [0.0, 0.0], "wins": 0}}, "elimination_fraction": 0.007692307692307693, "score": 0.3333333333333333}, "initial_actor_sha256": "6fe51f088e16ef414a224ddb01a2495706b314cc3a7eda88477b7b0fb6780935", "initial_checkpoint_sha256": "983f11be7adc776038f0d95c6c14cf11509f14b6883a3672fc56fafb4535a2e8", "initial_critic_sha256": "dd07316b6ca535c9bb37771cb625a0726b081764604091fbb8a37ba815572870", "match_uncertainty": {"ci95": [0.044270833333333336, 0.061197916666666664], "resamples": 5000, "scope": "conditional on one checkpoint pair; fixed deterministic cells add no sampling variance"}, "numerical_gate_passed": false, "reward": "R2", "scope": "Exploratory seed5201 selection; correctness review remains a separate prerequisite", "seed": 5201, "source_revision": "e0a2420e2b57e10b5ccf1448d91d8ecf967a8a1c"}, "final_test_not_accessed_for_this_model": true, "status": "exploratory pilot; primary validation gate failed; not a deployment recommendation"}`. The full confirmation design uses
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
Checkpoint **SHA256** identities: `{"initial.pt": "983f11be7adc776038f0d95c6c14cf11509f14b6883a3672fc56fafb4535a2e8", "model.pt": "495446a1142be875a43a6012377af4daa062a7e293b0d141e0c8c8b5ae1d6d79"}`.

Primary method references: [TorchRL PPO/GAE](https://docs.pytorch.org/rl/stable/tutorials/multiagent_ppo.html),
[Sample Factory Doom self-play inspiration](https://proceedings.mlr.press/v119/petrenko20a/petrenko20a.pdf),
[potential shaping](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf),
[independent-seed power](https://arxiv.org/abs/1806.08295), and
[evaluation uncertainty](https://arxiv.org/abs/2108.13264). Library versions and all design limits
are recorded in the source lockfile, protocol and [bibliography](https://github.com/BurnyCoder/3d-firstperson-shooter-reinforcement-learning-selfplay/blob/main/docs/sources.md).

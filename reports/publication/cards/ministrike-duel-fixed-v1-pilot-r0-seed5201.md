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
Source revision: [`ecd8e3b05a5f42963dc5359e259f4bdf376ff4c8`](https://github.com/BurnyCoder/3d-firstperson-shooter-reinforcement-learning-selfplay/tree/ecd8e3b05a5f42963dc5359e259f4bdf376ff4c8). Registration: `A-PILOT-R0`.

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

Reward `R0` and all coefficients are engineering choices in the
[active protocol](https://github.com/BurnyCoder/3d-firstperson-shooter-reinforcement-learning-selfplay/blob/ecd8e3b05a5f42963dc5359e259f4bdf376ff4c8/docs/duel-protocol.md), not published optima. Exact
optimizer/rollout/hardware settings are in `config.json`. Training recorded **1245184 arena steps**,
**152 updates**; full allocated step budget completed: **True**.

## Measurements and limitations

Final-test status: **pending**. Equal-opponent/side round score:
**pending**. Baseline breakdown: `{}`.
Elimination fraction: **pending**.
Final validation summary (selection evidence, not final testing): `{"baselines": {"initial": {"draws": 2, "losses": 0, "matches": 2, "score": 0.5, "side_scores": [0.5, 0.5], "wins": 0}, "random": {"draws": 237, "losses": 0, "matches": 256, "score": 0.537109375, "side_scores": [0.54296875, 0.53125], "wins": 19}, "scripted": {"draws": 0, "losses": 2, "matches": 2, "score": 0.0, "side_scores": [0.0, 0.0], "wins": 0}}, "elimination_fraction": 0.08076923076923077, "score": 0.345703125}`.
Win/draw/loss score is 1/0.5/0 against random, corresponding initial and moving quantized scripted
evaluation controls. Raw paired worlds and identities are in `initial_test.json` and `final_test.json`
when available; stochastic behavior is separately labeled. Validation files concern model selection,
not held-out final testing. Missing measurements are unavailable, not evidence of success.

Release assessment: `{"assessment": {"assessed_utc": "20261023T000500.321100Z", "complete_equal_budget": true, "diagnostics": {"final": {"initial": {"aim_saturation": 0.0, "damage_per_round": 25.0, "hits_per_shot": 0.03225806451612903, "reload_occupancy": 0.05125, "seconds_per_round": 20.0, "shots_per_round": 31.0}, "random": {"aim_saturation": 0.0, "damage_per_round": 30.6640625, "hits_per_shot": 0.07703631010794897, "reload_occupancy": 0.1560222801369513, "seconds_per_round": 19.1103515625, "shots_per_round": 15.921875}, "scripted": {"aim_saturation": 0.0, "damage_per_round": 25.0, "hits_per_shot": 0.4, "reload_occupancy": 0.45454545454545453, "seconds_per_round": 0.55, "shots_per_round": 2.5}}, "initial": {"initial": {"aim_saturation": 0.9475, "damage_per_round": 0.0, "hits_per_shot": 0.0, "reload_occupancy": 0.44625, "seconds_per_round": 20.0, "shots_per_round": 25.0}, "random": {"aim_saturation": 0.947275390625, "damage_per_round": 9.1796875, "hits_per_shot": 0.014205833459271573, "reload_occupancy": 0.437333984375, "seconds_per_round": 20.0, "shots_per_round": 25.84765625}, "scripted": {"aim_saturation": 0.4, "damage_per_round": 0.0, "hits_per_shot": 0.0, "reload_occupancy": 0.32857142857142857, "seconds_per_round": 1.75, "shots_per_round": 7.5}}}, "final": {"baselines": {"initial": {"draws": 2, "losses": 0, "matches": 2, "score": 0.5, "side_scores": [0.5, 0.5], "wins": 0}, "random": {"draws": 237, "losses": 0, "matches": 256, "score": 0.537109375, "side_scores": [0.54296875, 0.53125], "wins": 19}, "scripted": {"draws": 0, "losses": 2, "matches": 2, "score": 0.0, "side_scores": [0.0, 0.0], "wins": 0}}, "elimination_fraction": 0.08076923076923077, "score": 0.345703125}, "final_checkpoint_sha256": "63a54c05f8d03a712ba3d1fd92cf15644c4221afda6f4facf00c2a2e7b9fe6d9", "improvement": 0.012369791666666685, "initial": {"baselines": {"initial": {"draws": 2, "losses": 0, "matches": 2, "score": 0.5, "side_scores": [0.5, 0.5], "wins": 0}, "random": {"draws": 256, "losses": 0, "matches": 256, "score": 0.5, "side_scores": [0.5, 0.5], "wins": 0}, "scripted": {"draws": 0, "losses": 2, "matches": 2, "score": 0.0, "side_scores": [0.0, 0.0], "wins": 0}}, "elimination_fraction": 0.007692307692307693, "score": 0.3333333333333333}, "initial_actor_sha256": "6fe51f088e16ef414a224ddb01a2495706b314cc3a7eda88477b7b0fb6780935", "initial_checkpoint_sha256": "0a7f75e2291d9c38bd1c5701cfe945d45e98cfc4d9224527392683a354335041", "initial_critic_sha256": "dd07316b6ca535c9bb37771cb625a0726b081764604091fbb8a37ba815572870", "match_uncertainty": {"ci95": [0.007161458333333333, 0.018229166666666668], "resamples": 5000, "scope": "conditional on one checkpoint pair; fixed deterministic cells add no sampling variance"}, "numerical_gate_passed": false, "reward": "R0", "scope": "Exploratory seed5201 selection; correctness review remains a separate prerequisite", "seed": 5201, "source_revision": "ecd8e3b05a5f42963dc5359e259f4bdf376ff4c8"}, "final_test_not_accessed_for_this_model": true, "status": "exploratory pilot; primary validation gate failed; not a deployment recommendation"}`. The full confirmation design uses
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
Checkpoint **SHA256** identities: `{"initial.pt": "0a7f75e2291d9c38bd1c5701cfe945d45e98cfc4d9224527392683a354335041", "model.pt": "63a54c05f8d03a712ba3d1fd92cf15644c4221afda6f4facf00c2a2e7b9fe6d9"}`.

Primary method references: [TorchRL PPO/GAE](https://docs.pytorch.org/rl/stable/tutorials/multiagent_ppo.html),
[Sample Factory Doom self-play inspiration](https://proceedings.mlr.press/v119/petrenko20a/petrenko20a.pdf),
[potential shaping](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf),
[independent-seed power](https://arxiv.org/abs/1806.08295), and
[evaluation uncertainty](https://arxiv.org/abs/2108.13264). Library versions and all design limits
are recorded in the source lockfile, protocol and [bibliography](https://github.com/BurnyCoder/3d-firstperson-shooter-reinforcement-learning-selfplay/blob/main/docs/sources.md).

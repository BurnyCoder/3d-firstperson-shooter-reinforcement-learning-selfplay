---
license: mit
library_name: pytorch
tags:
- reinforcement-learning
- multi-agent
- self-play
- torchrl
---
# MiniStrike selfplay_seed333

Author: Libor Burian. Original 3D 5v5 tactical-game checkpoint from the [source repository](https://github.com/BurnyCoder/3d-firstperson-shooter-reinforcement-learning-selfplay).

Archival status: **DEFERRED, INCOMPLETE: no completed final tournament or tactical-skill claim**.
The deferred suite retains its original deadlines and all interruption amendments in the [learning audit](https://github.com/BurnyCoder/3d-firstperson-shooter-reinforcement-learning-selfplay/blob/main/reports/learning-audit/summary.md). Checkpoint source identity denotes the original learning implementation; later execution/performance revisions are recorded in that audit. This legacy fixed-opponent condition, where present, is outside the later pure-self-play Stage A protocol.

## Method and measured status

Condition: **selfplay**; independent seed **333**; **2211840 arena transitions**; 1080 updates; frame budget completed: **False**. Training source revision: `38697fee4af941b12cfd15489424e2750ef4dd1d`.

The shared actor is a 2x128 tanh MLP with 384 visibility-limited history inputs and six categorical controls. A 2x256 centralized critic is used only for MAPPO training. Gameplay inference uses local observations and raw controls. `initial.pt` is the untouched baseline; `model.pt` is the final saved endpoint, not selected using final-test scores.

Recorded evaluation status: **pending**. Mean round score: **pending**; conditional 95% seed-pair bootstrap interval: **pending**. Score assigns win 1, draw 0.5, loss 0 against frozen random/rush/guard controls with sides swapped. Raw rounds, side/opponent breakdowns, and exact seeds are in `evaluation.json` when available.

This is a small local experiment, not evidence of human-level skill or transfer to commercial Counter-Strike. Smoke runs validate the learning pipeline only. Match confidence intervals do not establish robustness across independently trained models. Failed hypotheses and weak bots remain published.

## Download and play

Install the source repository using its locked uv environment. Download this repository into `artifacts/models/selfplay_seed333`, then run:

```powershell
uv run csrl watch --first-person --checkpoint artifacts/models/selfplay_seed333/model.pt
uv run csrl play --checkpoint artifacts/models/selfplay_seed333/model.pt
```

In spectator mode, Tab selects a living bot, V changes camera, and F takes over or returns control without resetting the round. Checkpoints contain weights, optimizer/environment state and RNGs; the project loader validates the schema and uses PyTorch weights-only loading.

## Reproducibility and limitations

The one original map uses ground-level movement, box geometry, a single rifle and bomb objectives. Inputs are structured observations rather than pixels. Training gamma is 0.999; curriculum fractions are 15/15/70 for combat/objective/standard starts. Configuration, source revision, metrics and initial/final evaluations are provided without omitting negative outcomes. See the source [protocol](https://github.com/BurnyCoder/counter-strike-reinforcement-learning-selfplay/blob/main/docs/protocol.md) and [reports](https://github.com/BurnyCoder/counter-strike-reinforcement-learning-selfplay/tree/main/reports) for the complete comparison and uncertainty assumptions.

Algorithm references: [TorchRL MAPPO tutorial](https://docs.pytorch.org/rl/stable/tutorials/multiagent_ppo.html), [MAPPO paper](https://arxiv.org/abs/2103.01955), [RL evaluation uncertainty](https://arxiv.org/abs/2108.13264).

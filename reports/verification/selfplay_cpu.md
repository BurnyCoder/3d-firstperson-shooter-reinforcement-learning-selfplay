# Experiment: selfplay_cpu

Condition: `selfplay`; seed: `111`; training completed frozen frame budget. Primary-suite eligibility: `False`. Source revision: `a4f8ee7d9451cef61834d71db52c114399f4008b`.

## Method and rationale

MAPPO uses a shared actor with two 128-unit tanh layers and a separate team critic with two 256-unit tanh layers. Each actor observes four local frames; six categorical controls represent movement, yaw, pitch, fire, reload, and interaction. A five-player critic combines that team's local observations. TorchRL supplies clipped PPO and generalized advantage estimation. Historical self-play samples immutable opponents at round boundaries; the control trains against a fixed scripted opponent. See [MAPPO](https://arxiv.org/abs/2103.01955) and [library implementation](https://docs.pytorch.org/rl/stable/tutorials/multiagent_ppo.html).

Actual saved settings: gamma `0.999`, GAE lambda `0.95`, learning rate `0.0003`, PPO epochs `3`, rollout steps `128`, arenas `16`. The default gamma 0.999 retains approximately 0.30 of an outcome sixty seconds later at twenty ticks per second (0.999^1200). These small networks and fixed settings are practical starter choices, not empirically established optima. The curriculum allocates 15/15/70 percent of frame progress to combat/objective/standard reset scenarios; already-running rounds finish their existing scenario.

## Measurements

| Run | Frames | Elapsed seconds | Initial score [95% CI] | Final score [95% CI] |
| --- | --- | --- | --- | --- |
| selfplay_cpu | 32768 | 95.9 | 0.167 [0.167, 0.167] (complete) | 0.167 [0.167, 0.167] (complete) |

Side-specific final scores (0 attackers, 1 defenders): `{"0": 0.0, "1": 0.3333333333333333}`. Baseline-specific final scores: `{"guard": 0.0, "random": 0.5, "rush": 0.0}`. Mean recorded round diagnostics: `{"damage": 1.0416666666666667, "shots": 140.20833333333334, "plants": 0.3333333333333333, "defuses": 0.0, "inactive_fraction": 0.0}`.

Plant and defuse counts cover either team's objective events; damage, shots, and inactivity refer to the evaluated learner team.

## Learnings and evidence limits

The report records what the run measured; it cannot infer undocumented debugging attempts, causes of a score change, or generalization beyond this map and these controls. Initial/final score differences are descriptive within one trained seed. Wins alone do not demonstrate planting or defusing competence; inspect the objective counts and raw rounds. Match confidence intervals condition on this checkpoint and do not quantify retraining uncertainty. No best-epoch selection is performed by this reporting code. Validation performance is not used for checkpoint selection.

Artifact warnings: `[]`. Raw evidence: [run data](data/selfplay_cpu/result.json), [configuration](data/selfplay_cpu/config.json). Additional recorded evaluation and metric files are copied to the same directory when available.

# MiniStrike: a reproducible self-play pilot

Libor Burian

## Abstract

We implement an original 3D five-versus-five tactical shooter and compare historical self-play with fixed-opponent MAPPO training. Recorded training invocations: 2; completed primary-suite frame budgets: 0. The pre-specified comparison is pending: 0/3 matched training-seed pairs have complete, equal-budget results. Available pairs are retained in comparison.json; no subset is presented as the final hypothesis test.

## Introduction and motivation

The practical question is whether historical self-play improves a small local-observation policy's round score against frozen controls relative to equally budgeted fixed-opponent training. The pre-specified practically meaningful difference is 0.10. This experiment is specific to MiniStrike, not commercial Counter-Strike or human-level tactical play.

## Background and related work

MAPPO combines decentralized policy execution with a centralized critic during training [MAPPO](https://arxiv.org/abs/2103.01955). We reuse [TorchRL's multi-agent PPO machinery](https://docs.pytorch.org/rl/stable/tutorials/multiagent_ppo.html). Reporting separates seed variability from match variability, following concerns discussed in [RL evaluation uncertainty](https://arxiv.org/abs/2108.13264) and [RL reproducibility](https://arxiv.org/abs/1709.06560).

## Methods

One NumPy simulation defines movement, visibility, hits, simultaneous damage, finite ammunition, planting, defusing, timers, and terminal outcomes for both rendered play and learning. The actor is a shared 2x128 tanh MLP over four 96-feature observation frames (384 inputs); the central critic is a 2x256 tanh MLP over five players' combined observations. Six independent categorical heads form one joint action per player. Each PPO likelihood ratio exponentiates the difference between summed new and old control log probabilities, not player log probabilities. GAE precedes death masking; actor and entropy losses exclude post-death actions, while critic learning retains terminal team outcomes. Default discount gamma is 0.999 and GAE lambda is 0.95. Exact hyperparameters, frame counts, code revisions, and conditions are stored per run. Terminal team rewards are +1/-1 for win/loss and zero for draws. Zero-sum shaping adds at most 0.1 for effective damage to all opponents and 0.1 for planting; health is not replenished within rounds. Self-play chooses the latest frozen policy with probability 0.5 and historical policies otherwise; snapshots are selected only at round boundaries. The control uses fixed scripted opposition. Both conditions use the same frame-based combat, objective, and standard-round curriculum. The checkpoint is the final saved endpoint, with budget completeness recorded, not the best measured evaluation. Test tournaments pair scenarios with sides swapped against random, rush, and guard baselines. Score is 1 for a win, 0.5 for a draw, and 0 for a loss. The design calls for disjoint training and final-test seed ranges; recorded tournament metadata preserves the actual range. Seed-pair cluster bootstrap intervals condition on each checkpoint. The between-condition comparison uses all three paired training-seed differences and a Student-t interval with two degrees of freedom; it is not an interval over individual matches.

## Results

| Run | Frames | Initial score [95% CI] | Final score [95% CI] |
| --- | --- | --- | --- |
| selfplay_cpu | 32768 | 0.167 [0.167, 0.167] (complete) | 0.167 [0.167, 0.167] (complete) |
| selfplay_cuda | 8192 | 0.167 [0.167, 0.167] (complete) | 0.167 [0.167, 0.167] (complete) |

The pre-specified comparison is pending: 0/3 matched training-seed pairs have complete, equal-budget results. Available pairs are retained in comparison.json; no subset is presented as the final hypothesis test.

## Figures

![Measured training curves](learning_curves.png)

Figure 1. Raw cumulative training score and joint-action entropy against arena transitions. Training opponents and curriculum change over time, so these curves are optimization diagnostics, not held-out evaluation learning curves or causal evidence of superiority. No smoothing is applied.

## Discussion and implications

The fixed-opponent comparison is an intervention on the training opponent distribution while holding the registered architecture, objective, curriculum, and budgets constant. It can support a comparison only for those matched completed runs. More wins without objective completions do not establish tactical competence. This study does not isolate which historical opponent, reward component, observation feature, or curriculum stage causes an effect; those claims require further pre-registered ablations and counterfactual interventions.

## Limitations

Three independent training seeds per condition provide weak information about seed sensitivity. Normality of seed-pair differences is unverified, and a zero empirical variance can yield a zero-width t interval without proving zero population uncertainty. Bootstrap match intervals do not include retraining variability. This release evaluates one map, one weapon, ground-level movement, and limited scripted controls. Policies consume structured observations rather than rendered pixels. Compute limits, incomplete runs, negative outcomes, and missing artifacts are reported explicitly. No significant result, human-level skill, causal mechanism, or transfer to another game is presumed.

## Conclusion

The pre-specified comparison is pending: 0/3 matched training-seed pairs have complete, equal-budget results. Available pairs are retained in comparison.json; no subset is presented as the final hypothesis test. The software artifact establishes an executable game/training/evaluation path. Claims about learned competence must follow the recorded held-out scores and objective diagnostics, with the limitations above.

## Reproducibility appendix

The data directory preserves raw result, configuration, evaluation, and metric artifacts without copying model weights. Individual experiment Markdown files supply side/baseline breakdowns and raw-artifact links. comparison.json records every planned seed-pair difference and exclusion. The source repository and model cards provide installation and checkpoint usage instructions. Accuracy, fill rate, and generic completion rate are N/A: this is round-based reinforcement learning without labeled classification or fill-in targets.

Suite protocol, deadline, selected device, calibrated frame budget, and power sensitivity are pending: no suite.json was present in the requested input directory.

## Sources

- [MAPPO](https://arxiv.org/abs/2103.01955)
- [TorchRL multi-agent PPO](https://docs.pytorch.org/rl/stable/tutorials/multiagent_ppo.html)
- [RL evaluation uncertainty](https://arxiv.org/abs/2108.13264)
- [RL reproducibility](https://arxiv.org/abs/1709.06560)
- [Student-t distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html)
- [Noncentral-t power calculation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nct.html)

# Pre-specified pilot experiment

Registered before the first optimizer training run. Changes after execution must be recorded as amendments with their reason and whether test results were already observed.

## Question and falsifiable hypothesis

Does training against historical self-play opponents improve a shared policy's fixed-benchmark mean round score over training against one fixed, observation-only scripted opponent?

The practical hypothesis is a mean advantage of at least0.10 score units. The endpoint is round score: win1, draw0.5, loss0. Failure to exceed this threshold is not evidence of a practically meaningful benefit; uncertainty spanning it is inconclusive. A negative difference contradicts the directional hypothesis for the measured budget and benchmark. No claim of human-level Counter-Strike skill follows from this experiment.

## Conditions and independent units

- Historical self-play: 50% latest frozen policy,50% older pool candidates. Eight historical candidates plus the untouched initial policy; active opponents remain immutable through their rounds.
- Fixed control: one deterministic balanced scripted opponent using the same legal observations and physical controls. Its pathfinding and aiming are explicit scripted behavior, never assistance inside learned policies.
- Both use identical simulator, rewards, observations, neural architecture, PPO settings, curriculum, and target arena-transition count.
- Independent training seeds111,222,333 per condition. Execution order:selfplay111,fixed111,fixed222,selfplay222,selfplay333,fixed333.
- Calibration seed8001; it is excluded from the primary comparison. Tests/synthetic gradient checks are not training outcomes.

## Budget and calibration

Maximum12hours from the suite start, including calibration, checkpointing, and evaluation. At most10hours are allocated across six independent training runs; the remainder supports calibration/evaluation. A resumed suite retains its original absolute deadline.

Execution accounting clarification, recorded after the engineering smoke runs but before the primary suite: this machine's suite allowance will be shortened by the wall time elapsed since the first optimizer smoke began. This conservatively keeps verification plus the primary experiment within the user's overall 12-hour limit, even counting intervening implementation time. The actual command and absolute deadline are recorded in the launch receipt and `suite.json`. No algorithm setting or hypothesis changes in this clarification.

Calibration measures four actual collection/update batches for each available optimizer device and condition. CPU simulation/inference is shared. Choose the optimizer device by the minimum throughput across conditions, then select the fastest such device. Freeze equal frames per run from the slower-condition rate, the allocated duration, and a20%runtime reserve, rounded down to complete rollout batches. Record exact rates and the frozen target in `suite.json`. Device selection never uses game scores.

Runs stop on their frame or time allowance. A timed-out run is retained and flagged; comparisons requiring equal training frames must not silently include mismatched budgets. Do not extend training to obtain significance. Interruption/resume restores optimizer, simulator, observation history, roles, frozen opponent pool, random generators, counters, and used budget.

### Execution amendment — 22 October 2026

After the primary suite started, hardware-only probes selected six concurrent independent learners, each retaining 16 arenas and 128 ticks per rollout, with one PyTorch CPU thread per process. This replaces the serial execution order above; it does not pool models, trajectories, optimizers, or opponent snapshots. No primary final-test results were observed when selecting this change. All ten probes, including a later one-versus-two-thread comparison and slower configurations, are preserved in the [performance report](../reports/performance/summary.md).

The frozen target remains 4,167,680 arena steps for each of six runs. Runtime caps are reallocated for overlap within the original five-sixths training window, preserving the evaluation reservation and absolute suite deadline of 2026-10-23 07:32:38.503505 UTC. Calibration suspensions and the checkpoint handoff consume that existing wall budget. The handoff archives the original checkpoint/configuration/metrics/manifest and records runtime-field changes; model, optimizer, random-generator, simulator, snapshot, and training-counter states are restored. Reward, curriculum, architecture, PPO, analysis, and evaluation settings remain unchanged. Changed CPU threading/device scheduling may affect numerical execution, so bitwise equivalence across execution modes is not asserted. Timed-out or unequal-frame results still receive the original exclusions and disclosure rules.

## Training settings and rationale

The shared actor has two128-unit hidden layers. The centralized critic has two256-unit hidden layers and receives the team's combined local observations. Categorical heads represent movement9, yaw5, pitch5, and binary fire/reload/interact. Their log probabilities sum within each player, not across players.

Initial defaults: Adam learning rate0.0003, PPO clipping0.2, three epochs per batch,128ticks per rollout,256arena-time rows per minibatch, entropy weight0.01, critic weight0.5, gradient norm cap1.0, GAE lambda0.95. These are engineering starting values adapted from the [TorchRL MAPPO tutorial](https://docs.pytorch.org/rl/stable/tutorials/multiagent_ppo.html), not claimed optimal values.

Discount gamma0.999 at20Hz leaves approximately0.301 weight on a reward60seconds later; gamma0.99 would leave approximately0.0000058. This choice retains meaningful delayed bomb outcomes. The training schedule uses15%close-combat starts,15%objective starts,70%ordinary starts. These are reset curricula; actions remain learned and ordinary evaluation always uses standard5v5 starts.

Dead agents remain slots until the round ends. Returns include delayed team outcomes; the final living action is valid. Actor and entropy losses exclude subsequent dead-player actions, while the centralized critic continues learning through team trajectories. Do not remove dead timesteps before GAE.

## Evaluation and leakage controls

Freeze checkpoint weights during testing. Final-test root seed2000000, reserved validation root1000000, and training seeds are disjoint. No validation-driven hyperparameter search or early promotion is planned. Spawn variation does not imply generalization to unseen maps: this release has one map.

For each initial and final checkpoint, use128seed pairs by default against random,rush,guard controls. Each pair reuses identical initialized arenas with learner side swapped. Batching/arena index is part of seed identity and is recorded. Score all rounds; do not discard draws or weak runs. Models are evaluated deterministically; random opponents have separate reproducible generators.

The deployment checkpoint is **selfplay seed111 final**, chosen before testing. Never select the deployed model by final-test score. Both watch modes and human play load that same checkpoint.

Primary endpoint: equal-weight average round score over the three fixed opponents and both sides. Also report wins/draws/losses, side/opponent-specific scores, round duration, damage, shots, plants, defuses, and zero-action fraction among living decisions. An external evaluation deadline is incomplete evidence, not a draw.

## Statistical analysis

Before training, compute two-sided alpha0.05/power0.80 independent two-sample t-test sensitivity for practical effect0.10 and assumed between-run standard deviations0.05,0.10,0.20 using SciPy's noncentral t distribution. Publish these assumptions and required seed counts. Three seeds per condition are fixed by this pilot's compute budget; do not call the study powered unless justified.

For one frozen checkpoint, percentile bootstrap2000replicates over seed-pair clusters, preserving both sides and all baseline games in a cluster. This measures conditional match uncertainty only. Report all three matched training-seed score differences; when all are available under equal budgets, show their mean and a Student-t95%interval with its small-sample/normality assumptions. Do not confuse thousands of matches with thousands of independent learned models.

The fixed-opponent condition is a causal intervention on opponent generation, with other settings held constant. Side swapping is a paired control for role/map advantages. No inference about representation mechanisms or broader games is supported. Report thermal/device timing, curriculum, shaping, limited seeds, scripted benchmark bias, single-map scope, and potential simulation determinism limitations as confounders/limitations.

## Sources

- [MAPPO: The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955): centralized training and decentralized execution; not a guarantee for this competitive shooter.
- [TorchRL multi-agent PPO](https://docs.pytorch.org/rl/stable/tutorials/multiagent_ppo.html): maintained learning components and data layout.
- [Unity self-play configuration](https://unity-technologies.github.io/ml-agents/Training-Configuration-File/#self-play): historical snapshots and opponent diversity/stability tradeoff.
- [Deep Reinforcement Learning that Matters](https://arxiv.org/abs/1709.06560): independent seeds and reproducible experimental comparisons.
- [How Many Random Seeds? Statistical Power Analysis in Deep Reinforcement Learning Experiments](https://arxiv.org/abs/1806.08295): power sensitivity and run-level variability.
- [Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264): uncertainty and aggregate evaluation.
- [How to avoid machine learning pitfalls](https://arxiv.org/abs/2108.02497): separation of model selection and evaluation.
- [NeurIPS paper checklist](https://neurips.cc/public/guides/PaperChecklist): transparent claims, assumptions, reproducibility, and limitations.

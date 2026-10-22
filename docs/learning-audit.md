# Exploratory learning audit — 22 October 2026

The user requested an investigation after all three fixed-opponent runs logged zero wins. This audit is separate from the frozen primary comparison in [protocol.md](protocol.md). Primary weights, optimizer settings, step budgets, and final-test seeds are unchanged while diagnosing the failure. All diagnostic execution consumes the original wall-clock allowance.

## Questions and tests

1. Do simulator rewards and termination flags reach collected trajectories unchanged? Capture the adapter outputs and compare them with the TorchRL rollout.
2. Does GAE have the correct reward sign, temporal axis, discounting, and round boundaries? Compare production output with exact four-step returns using a zero critic.
3. Does the actual PPO gradient increase the probability of a positively rewarded action? Compare matched fire/no-fire trajectories through the production loss, with a symmetric zero-logit actor and no optimizer step.
4. Do frozen policies improve actual behavior? Compare initial, fixed seed111, self-play seed111, and an identical-scripted-policy scoring control on 16 paired worlds per curriculum, using validation seed1100010. Record every round, damage, shots, duration, held interaction progress, reload occupancy, and steep aim. Stochastic actions match training behavior. This small single-checkpoint diagnostic does not establish independent-model performance.
5. Can the entire production collector, GAE, PPO, optimizer, synchronization, saving, and inference chain learn an easy task? The following positive control was specified before its first optimizer step.

## Fixed positive-control protocol

Two independent seeds, **91001 and 91002**, each receive **65,536 arena steps** (32 production updates), with a 600-second safety cap and the original architecture/PPO settings. CUDA optimizes the model; one CPU thread handles collection. No extension or seed replacement is allowed to obtain a pass.

Only diagnostic starting states and opposition differ: one living attacker starts 2.5 metres from a passive, one-hit defender whose horizontal position varies by ±0.4 metres. A four-tick deadline makes waiting a failure. The actor chooses its own movement, aim, fire, reload, and interaction; it receives no aiming assistance. The actual game collision, visibility, rifle, observations, terminal rewards, and death masks remain in use. A dedicated process replaces the environment constructor through a standard-library test fixture; primary processes are unaffected.

Evaluate each untouched initial and final actor stochastically on the same **512 validation worlds**, seed1190000. Pre-specified engineering success is final score at least0.80 and improvement at least0.20. Retain all binary outcomes and both seeds even if either fails. This is a positive control for learnability of the implementation, not evidence of 5v5 skill or self-play superiority. These diagnostic checkpoints must never be promoted as full-game deployment models.

```powershell
uv run python -m ministrike.experiments.diagnostics --snapshot runs/diagnostics/FROZEN_SNAPSHOT --pairs 16
uv run python -m ministrike.experiments.learning_probe --output runs/diagnostics/firing_seed91001 --seed 91001 --device cuda
uv run python -m ministrike.experiments.learning_probe --output runs/diagnostics/firing_seed91002 --seed 91002 --device cuda
```

## Comparable work and limits

- [Sample Factory, ICML2020](https://proceedings.mlr.press/v119/petrenko20a/petrenko20a.pdf), section4.3 and figures7–8, trained complex Doom battle/duel agents on billions of environment frames; the multiplayer population used 2.5billion frames per agent. Pixel input, frame skipping, reward shaping, hardware, population-based tuning, and task complexity differ from MiniStrike. This is context for possible scale, not a justified frame requirement for this game.
- [The author's PPO+RND Doom repository](https://github.com/callumhay/vizdoom_ppo_rnd) reports basic scenarios below one million global steps and two full levels around six million steps/eight hours on an RTX3080. Those are one author's reports with seed42 and intrinsic rewards, not a controlled comparison or a promised training time here.
- [Stable Baselines3's implementation guidance](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html) recommends checking custom environments and starting with simpler problems and informative rewards. That motivates the positive control; it does not imply any particular cause for our failure.
- [PPO implementation details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) and [TorchRL's clipped PPO reference](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.objectives.ClipPPOLoss.html) ground the probability, value, entropy, and diagnostic checks.

## Evidence handling

Frozen checkpoint hashes and exact frame counts are recorded beside the raw audits. The initial read-only rollout audit's `learner_damage` field used final roles after several resets and is invalid; exclude that field from analysis. The complete-round behavior audit attributes damage from each arena's fixed learner side and is the correct source. Original raw records remain preserved, with this disclosure. No final-test scores have informed these diagnostics.

Results and any justified implementation or curriculum amendment will be recorded separately after the controls finish. Passing numerical checks alone is insufficient to claim that the bots learned useful combat or objectives.

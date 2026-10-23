# Confirmation resolution after the observer-coordinate diagnostic

Resolved **2026-10-23T05:08:00.134460Z**, before any confirmation optimizer update. This amendment implements the conditional confirmation in the [approved Stage A protocol](../duel-protocol.md). It allocates no additional time and does not change any earlier experiment. [Exact runnable configuration](../../configs/duel-confirmation.json), [qualifying evidence](../../reports/duel-fixed-v1/egocentric-v1/README.md), [roadmap](../experiment-roadmap.md).

## Qualification and frozen recipe

`A-DIAG-EGO-R1` completed all4,980,736 arena steps and all six scheduled validation endpoints. Its primary external-control gain is0.619140625, conditional95% paired match interval[0.6103515625,0.6279296875]. Its own-initial gain is2/3; final random/initial/script scores are all1.0, with100% elimination. Thus the pre-specified hypothesis and conservative recipe screen pass. These are one exploratory training seed's outcomes, not independent-model confirmation. The diagnostic checkpoint cannot deploy. All previous pilot/diagnostic failures remain evidence. [Raw analysis](../../reports/duel-fixed-v1/egocentric-v1/data/egocentric/analysis.json), [audit](../../reports/duel-fixed-v1/egocentric-v1/audit.json).

Run `A-CONFIRM` from scratch with **5101,5102,5103**, in that order. No diagnostic tensors, optimizer state, trajectories, human data or scripted data are used for initialization or training. Each seed has independent actor, critic, optimizer, RNGs and historical opponents. All learn the same Stage A fixed-spawn1v1 rules, four48-field observation frames, actor2×128/tanh with `egocentric-v1` preprocessing, raw-input critic2×256/tanh, categorical9/5/5/2/2 controls, R1 outcome/net-effective-damage reward (alpha0.50) and historical self-play. PPO/GAE, Adam3e-4, clip0.2,3epochs,128ticks,256-row minibatches,gamma0.999,lambda0.95,entropy0.01,criticcoefficient0.5,gradientlimit1.0 and snapshot10updates remain unchanged. These are registered engineering choices, not optimality claims; [exact diagnostic specification](2026-10-23-egocentric-diagnostic.md), [library sources](../sources.md).

## Equal exposure and remaining clock

Reuse the [completed hardware calibration](2026-10-23-duel-calibration.md): one serial learner,64arenas, one CPU thread, CUDA learner inference/optimization and CPU frozen opponents. No new throughput probe. At resolution,15,772.493472seconds remained before the09:30:52.627932UTC confirmation cap. The calibrated rate is1,300.936954105218 arena steps/s. Apply the previously registered20% reserve and round down to complete8,192-step rollouts:

`floor(0.8 × 1300.936954105218 × 15772.493472 / 3 / 8192) × 8192 = 5,464,064`.

Each model receives **5,464,064arena steps,667updates,5,464,064learner decisions and10,928,128total player decisions**; per-run elapsed cap5,257seconds. The three targets total16,392,192arena steps. At calibration throughput, training takes12,600.297seconds in total, leaving3,172.196472seconds at resolution for publication, startup, checkpoint overhead, interruptions and throughput variation. Publication consumes this reserve; it does not reset the resolution clock. Launch by05:30UTC, otherwise preserve the registration as unrun and explicitly reassess feasibility without changing completed exposure. Every trainer also enforces the original09:30phase boundary. The overall12-hour deadline remains11:30:52.627932UTC.

Stop on the fixed target, elapsed/absolute cap, nonfinite state or correctness failure. Preserve failed/incomplete seeds; do not replace them, reduce a later seed's target, extend training for significance or resume completed runs. Real interrupted recovery uses the saved optimizer/RNG/environment/pool state, archives uncheckpointed metric suffixes, and charges downtime to the same clock. Keep source/config/HEAD stable through all three workers because subsequent preflights and recovery must retain one implementation identity.

## Frozen evaluation, selection and inference

After all workers stop, evaluate **initial and final** of every seed against the existing random, corresponding initial and moving quantized scripted baselines, using128 paired seeds,32arenas and both spawn ends. Deterministic behavior is primary; stochastic behavior is secondary. Validation root3,000,000; final-test root4,000,000. Deterministic fixed-versus-fixed cells have two unique trajectories, never256 independent replicas. Save all first-seed traces for each baseline/end. No early checkpoint selection or final-test tuning.

Before reading final-test results, freeze the final checkpoint with highest equally weighted primary validation score; exact ties choose lower training seed. Record all candidates' checkpoint hashes and scores. Only completed confirmation models may be selected. Evaluate every initial/final model on final test, regardless of its validation rank, then assess the original mean improvement>=0.20 hypothesis and engineering acceptance: median final scores>=0.70 for random/initial,>=0.55 for scripted, and>=90% elimination. A selected checkpoint remains experimental if these fail. There is no assumption that the exploratory perfect scores replicate.

[Already published power sensitivity](../../reports/duel-fixed-v1/power.json) gives0.908489/0.470749/0.179255 power under between-model difference SD0.05/0.10/0.20 respectively, for paired effect0.20,n3,two-sidedalpha0.05. These assumed normal-difference calculations use [SciPy's noncentral t](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nct.html), motivated by [Colas et al.](https://arxiv.org/abs/1806.08295); they are not measured power. Report every model's delta, the three-model mean and assumption-sensitive t interval with2df, separately from conditional paired match intervals. Degenerate intervals from identical outcomes do not establish generalization. Fixed starts support no claim about random starts, other maps or team objectives. [Evaluation limitations](https://arxiv.org/abs/2108.13264).

## Execution and artifacts

After this configuration, implementation and all prerequisites are committed and pushed:

```powershell
uv run csrl train --experiment duel-fixed-v1 --suite --run-config configs/duel-confirmation.json
```

Use the existing hidden launcher and serial `execute_registered` supervisor, with group receipts `confirmation/group-0.json` through`group-2.json`; preserve stdout/stderr, host telemetry, full per-update metrics and all initial/final checkpoints. No automatic final-test access is part of this training command. Follow-up evaluation must preserve the original deadline and freeze selection first. Publish individual/aggregate reports, loss/reward/combat/activity/optimizer/score plots, raw hashes, model cards and the sourced paper. Native spectator/first-person/human/takeover verification and publication are release gates, not presumed by this registration.

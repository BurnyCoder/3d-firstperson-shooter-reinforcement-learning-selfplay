# Saved experiment graphs

The user requested loss, reward, evaluation and additional informative graphs. Each run with canonical `metrics.jsonl` receives loss PNG/SVG exports. Gameplay runs with the newer behavior logger also receive five four-panel views, individually and in small comparable groups. Raw update averages or explicitly derived interval rates are plotted against arena steps without smoothing or averaging different seeds. Deferred/partial runs remain labeled. Archived recovery copies are preserved elsewhere but are not additional independent experiments. These additional views are descriptive post-pilot analysis, not new registered acceptance criteria.

## Galleries

- [Stage A pilots and all nine calibration cells](../reports/duel-fixed-v1/losses/README.md): 17 calibration workers plus three reward pilots.
- [Reward and gameplay telemetry](../reports/duel-fixed-v1/losses/training/README.md): reward components, combat, behavior, self-play outcomes and optimizer diagnostics for R0/R1/R2. Calibration remains timing evidence, without a duplicate gameplay gallery.
- [Deferred 5v5 runs](../reports/learning-audit/losses/README.md): six canonical interrupted trajectories.
- [Historical shooting diagnostic](../reports/learning-audit/diagnostic-losses/README.md): a correctness fixture, never a deployment model. The failed second diagnostic had no logged optimizer updates, hence no curve.
- [Historical CPU/CUDA verification](../reports/verification/losses/README.md): pipeline smoke checks.
- [Original suite calibration](../reports/performance/initial-calibration-losses/README.md): four short timing runs. Later hardware probes retained throughput evidence, not per-update loss logs; no loss measurements are invented for them.

## Reading the figures

Total optimization loss combines the PPO policy surrogate, weighted value loss and entropy regularization. Policy loss is not a prediction-error percentage or a win rate. Value targets change with policies and opponents. Entropy describes action-distribution spread. Different reward definitions change loss scales, so lower loss across reward variants is not automatically better. These meanings follow the [TorchRL loss interface](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.objectives.ClipPPOLoss.html); the actual coefficients and value-loss type are in each run's configuration and [active protocol](duel-protocol.md).

The legacy logger did not record policy loss separately. Its corresponding panel says **Not recorded**, rather than displaying zeros. Missing/nonfinite values create gaps and counts in `evidence.json`; malformed interior rows or repeated frame counts cause a visible error. An unfinished final append is excluded with a warning, while its original bytes remain saved. The finite canonical logs currently plotted contain no such excluded tail.

[Primary deterministic validation](../reports/duel-fixed-v1/losses/validation-deterministic.png) and [secondary stochastic validation](../reports/duel-fixed-v1/losses/validation-stochastic.png) are separate figures. Bars show round score by frozen opponent; points show final minus initial aggregate score. Error bars preserve initial/final, spawn-end and common-seed pairing using the registered 5,000-resample bootstrap. They condition on one checkpoint pair; they do not measure variation across independent training seeds. Three reward variants of seed 5201 are not three independent confirmations. Only the primary plot carries the registered +0.10 pilot gate. Baseline scores are equally weighted even though deterministic fixed-layout baselines have only two unique trajectories.

The [deterministic breakdown](../reports/duel-fixed-v1/losses/validation-deterministic-breakdown.png) and [stochastic breakdown](../reports/duel-fixed-v1/losses/validation-stochastic-breakdown.png) expose each spawn end in a 0-to-1 score heatmap and show observed wins/draws/losses as stacked bars. Segment labels are counts; `n` is the actual trajectory count. A deterministic two-trajectory control is never presented as 256 independent matches. No new evaluations or final-test access are needed to create these plots; both use their adjacent `validation-<mode>.json` receipt and original endpoint records.

## Action-mode intervention

The [action-mode diagnostic](../reports/duel-fixed-v1/action-modes-v1.md) adds seven PNG/SVG pairs: a score heatmap with learner-only paired contrasts, and all six prospectively selected model/spawn-end path-and-health comparisons. These use frozen validation data only, with DD's two deterministic trajectories shown explicitly. [Raw hashes and derived values](../reports/duel-fixed-v1/action-mode-verification.json). Reproduce from a clone with `uv run python scripts/report_action_modes.py --input reports/duel-fixed-v1/data/action-modes-v1 --output runs/reproduced-action-modes`.

## Reward and activity units

The [logger](../src/ministrike/learning/environment.py) records cumulative learner counters. The [analysis](../src/ministrike/experiments/training_plot_metrics.py) takes adjacent differences before dividing by exposure. With counter `C` and arena steps `S`, the plotted interval rate is `(C[t] - C[t-1]) / (S[t] - S[t-1])`. A zero origin is permitted only when the log starts at update 1. Undefined denominators and unavailable telemetry remain gaps; nonnegative counters that decrease raise an error. An absent sparse event counter means zero only when that row's behavior logger is active. These are accounting rules, not learned estimators; [NumPy's first-difference API](https://numpy.org/doc/stable/reference/generated/numpy.diff.html) supplies the arithmetic.

| View | Plotted quantities and limits |
|---|---|
| Reward | Total and separate outcome, net-damage and aim-potential reward per arena step. Total is the exact sum of the three components. These are interval means over concurrent arenas, **not episode returns**. R0/R1/R2 have different reward definitions, so raw totals are not a common skill scale. |
| Combat | Hits / actual shots, damage health points / 1000 arena steps, shots / 1000 steps, opponent kills / 1000 steps. An interval without shots has undefined accuracy. |
| Behavior | Reload occupancy, absolute pitch at least 1.3 radians, and all decoded controls zero, each per learner decision; timeouts per completed round. Zero controls is a narrow inactivity proxy: it neither measures movement distance nor detects every ineffective action loop. |
| Self-play outcomes | Wins, draws, losses and round score among rounds completed in each interval. Opponents and policies change during training; these are **not fixed-benchmark learning scores**. An interval with no completed rounds remains a gap. |
| Optimization | Logged approximate policy KL, PPO clip fraction, gradient L2 norm returned before clipping, and adjacent arena steps / trainer elapsed seconds. The last quantity includes collection/update/checkpoint overhead between log records; it excludes orchestration outside the trainer and is not the absolute experiment-budget clock. |

Reward-component sums can be negative; signed axes retain that evidence. In particular, R2's undiscounted potential stream is not constrained to have positive mean. The discounted telescoping test required by the [protocol](duel-protocol.md) is a separate mathematical check, not something inferred from this graph. All time-series lines show every logged interval without smoothing or independent-model confidence bands. Fractions use the full 0-to-1 scale. Missing reward telemetry in older runs is not reconstructed from outcome counters.

The architecture visualization remains in the [README data-flow graph](../README.md#features-and-technologies); the selected actor/critic dimensions are specified beside the methodology and in the [protocol](duel-protocol.md). Plot generation reuses one renderer for loss and activity curves, and consumes the same byte snapshots so live appends cannot give the two galleries different inputs.

## Reproduce and audit

```powershell
uv run python scripts/plot_experiment_losses.py --input runs/duel-fixed-v1 --output reports/duel-fixed-v1/losses
uv run python scripts/plot_experiment_losses.py --input runs/suite --output reports/learning-audit/losses --pattern '*_seed*/metrics.jsonl'
uv run python scripts/plot_experiment_losses.py --input runs/diagnostics --output reports/learning-audit/diagnostic-losses
uv run python scripts/plot_experiment_losses.py --input runs/verification --output reports/verification/losses
uv run python scripts/plot_experiment_losses.py --input runs/suite/calibration --output reports/performance/initial-calibration-losses
# Recreate loss curves using only published snapshots after cloning:
uv run python scripts/plot_experiment_losses.py --input reports/duel-fixed-v1/losses/data --output runs/reproduced-losses
```

Each gallery includes exact input bytes under `data/` and an `evidence.json` receipt with seeds, full configuration, source revisions, checkpoint SHA256 identities and missingness counts. `training/<run>.json` also saves every derived interval value, using JSON `null` for unavailable quantities, and the same input identities. Validation PNG/SVG files have adjacent JSON containing endpoint identities, source hashes and all plotted numbers; raw evaluation rows live in the individual pilot report bundles. To reproduce validation comparisons from a clone, arrange those bundles as `runs/reproduced/pilots/R0/seed5201`, `R1/seed5201`, `R2/seed5201` and invoke the same wrapper on `runs/reproduced`. This is analysis only: it neither loads model tensors nor starts an optimizer or final-test evaluation.

`csrl report` invokes the same graph phases automatically. Matplotlib 3.11.2 supplies the standard line, error-bar and PNG/SVG APIs; references are in the [bibliography](sources.md). Figure layout and color choices are engineering decisions. Reports must interpret measured behavior beside these curves rather than using loss alone as evidence of useful play.

## Completed duration/entropy comparison

The [comparison gallery](../reports/duel-fixed-v1/duration-entropy-v1/README.md) adds23PNG/SVG pairs: all loss/reward/combat/behavior/self-play/optimizer curves, primary and secondary initial/early/final scores at their actual step counts, spawn/WDL breakdowns and registered contrasts with conditional match intervals. Reproduce with `uv run python scripts/report_duration_entropy.py`; the reporter verifies all12 endpoints, evaluated checkpoint hashes and the original R1 prefix before copying exact inputs and plotting. It loads checkpoint tensors for provenance checks, but runs no gameplay or optimizer. [Audit and hashes](../reports/duel-fixed-v1/duration-entropy-v1/audit.json). The historical galleries remain unchanged.

## Observer-coordinate comparison

The [completed input intervention](../reports/duel-fixed-v1/egocentric-v1/README.md) adds11PNG/SVG pairs using the same renderers: loss, reward, combat, behavior, optimizer, self-play outcomes, two endpoint comparisons, two spawn/WDL breakdowns and the external-control contrast. The total saved galleries now contain102figure pairs. `uv run python scripts/report_egocentric.py` audits checkpoint identities and raw summaries, retains all43raw inputs and plots without new matches. [Hashes and plotted values](../reports/duel-fixed-v1/egocentric-v1/audit.json). Across representations, corresponding initial opponents differ; only the random/script contrast estimates the registered external effect. Perfect fixed-layout scores and a degenerate conditional interval do not eliminate independent-model uncertainty. All36 prospectively selected traces remain in the six raw endpoint files.

# Saved experiment graphs

The user requested visible loss and evaluation graphs. Each run with canonical `metrics.jsonl` receives a PNG and an SVG. Raw update averages are plotted against arena steps without smoothing or averaging different seeds. Deferred/partial runs remain labeled. Archived recovery copies are preserved elsewhere but are not additional independent experiments.

## Galleries

- [Stage A pilots and all nine calibration cells](../reports/duel-fixed-v1/losses/README.md): 17 calibration workers plus three reward pilots.
- [Deferred 5v5 runs](../reports/learning-audit/losses/README.md): six canonical interrupted trajectories.
- [Historical shooting diagnostic](../reports/learning-audit/diagnostic-losses/README.md): a correctness fixture, never a deployment model. The failed second diagnostic had no logged optimizer updates, hence no curve.
- [Historical CPU/CUDA verification](../reports/verification/losses/README.md): pipeline smoke checks.
- [Original suite calibration](../reports/performance/initial-calibration-losses/README.md): four short timing runs. Later hardware probes retained throughput evidence, not per-update loss logs; no loss measurements are invented for them.

## Reading the figures

Total optimization loss combines the PPO policy surrogate, weighted value loss and entropy regularization. Policy loss is not a prediction-error percentage or a win rate. Value targets change with policies and opponents. Entropy describes action-distribution spread. Different reward definitions change loss scales, so lower loss across reward variants is not automatically better. These meanings follow the [TorchRL loss interface](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.objectives.ClipPPOLoss.html); the actual coefficients and value-loss type are in each run's configuration and [active protocol](duel-protocol.md).

The legacy logger did not record policy loss separately. Its corresponding panel says **Not recorded**, rather than displaying zeros. Missing/nonfinite values create gaps and counts in `evidence.json`; malformed interior rows or repeated frame counts cause a visible error. An unfinished final append is excluded with a warning, while its original bytes remain saved. The finite canonical logs currently plotted contain no such excluded tail.

[Primary deterministic validation](../reports/duel-fixed-v1/losses/validation-deterministic.png) and [secondary stochastic validation](../reports/duel-fixed-v1/losses/validation-stochastic.png) are separate figures. Bars show round score by frozen opponent; points show final minus initial aggregate score. Error bars preserve initial/final, spawn-end and common-seed pairing using the registered 5,000-resample bootstrap. They condition on one checkpoint pair; they do not measure variation across independent training seeds. Three reward variants of seed 5201 are not three independent confirmations. Only the primary plot carries the registered +0.10 pilot gate. Baseline scores are equally weighted even though deterministic fixed-layout baselines have only two unique trajectories.

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

Each gallery includes exact input bytes under `data/` and an `evidence.json` receipt with seeds, full configuration, source revisions, checkpoint SHA256 identities and missingness counts. Validation PNG/SVG files have adjacent JSON containing endpoint identities, source hashes and all plotted numbers; raw evaluation rows live in the individual pilot report bundles. To reproduce validation comparisons from a clone, arrange those bundles as `runs/reproduced/pilots/R0/seed5201`, `R1/seed5201`, `R2/seed5201` and invoke the same wrapper on `runs/reproduced`. This is analysis only: it neither loads model tensors nor starts an optimizer or final-test evaluation.

`csrl report` invokes the same graph phases automatically. Matplotlib 3.11.2 supplies the standard line, error-bar and PNG/SVG APIs; references are in the [bibliography](sources.md). Figure layout and color choices are engineering decisions. Reports must interpret measured behavior beside these curves rather than using loss alone as evidence of useful play.

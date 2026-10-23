# Saved loss graphs

- [legacy comparison](legacy.png) · [SVG](legacy.svg)
- [calibration_cpu_fixed / seed 8001 (complete)](calibration_cpu_fixed.png) · [SVG](calibration_cpu_fixed.svg)
- [calibration_cpu_selfplay / seed 8001 (complete)](calibration_cpu_selfplay.png) · [SVG](calibration_cpu_selfplay.svg)
- [calibration_cuda_fixed / seed 8001 (complete)](calibration_cuda_fixed.png) · [SVG](calibration_cuda_fixed.svg)
- [calibration_cuda_selfplay / seed 8001 (complete)](calibration_cuda_selfplay.png) · [SVG](calibration_cuda_selfplay.svg)

Total loss combines the policy, weighted value and entropy terms. Policy loss is a PPO surrogate, not a win rate. Value targets change as the policy and opponents change; different reward scales also change losses. Entropy measures action-distribution spread, not competence. These are logged update averages, with no extra smoothing or confidence bands. Missing/nonfinite metrics remain gaps or Not recorded panels; evidence.json counts them. No new measurements are inferred.

Exact input snapshots are in data/; evidence.json records source revisions, checkpoint hashes, seeds and configuration. Rerun the plotting wrapper with this data directory as --input to reproduce the curves. [TorchRL PPO loss](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.objectives.ClipPPOLoss.html), [Matplotlib](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html).

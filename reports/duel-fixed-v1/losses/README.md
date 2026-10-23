# Saved loss graphs

- [Reward, combat, behavior, self-play outcomes and optimizer diagnostics](training/README.md)
- [reward-pilots comparison](reward-pilots.png) · [SVG](reward-pilots.svg)
- [calibration__cpu_reference_e16_w1__seed5290 (complete)](calibration__cpu_reference_e16_w1__seed5290.png) · [SVG](calibration__cpu_reference_e16_w1__seed5290.svg)
- [calibration__cuda_infer-cpu_e16_w1__seed5290 (complete)](calibration__cuda_infer-cpu_e16_w1__seed5290.png) · [SVG](calibration__cuda_infer-cpu_e16_w1__seed5290.svg)
- [calibration__cuda_infer-cpu_e16_w3__seed5290 (complete)](calibration__cuda_infer-cpu_e16_w3__seed5290.png) · [SVG](calibration__cuda_infer-cpu_e16_w3__seed5290.svg)
- [calibration__cuda_infer-cpu_e16_w3__seed5291 (complete)](calibration__cuda_infer-cpu_e16_w3__seed5291.png) · [SVG](calibration__cuda_infer-cpu_e16_w3__seed5291.svg)
- [calibration__cuda_infer-cpu_e16_w3__seed5292 (complete)](calibration__cuda_infer-cpu_e16_w3__seed5292.png) · [SVG](calibration__cuda_infer-cpu_e16_w3__seed5292.svg)
- [calibration__cuda_infer-cpu_e64_w1__seed5290 (complete)](calibration__cuda_infer-cpu_e64_w1__seed5290.png) · [SVG](calibration__cuda_infer-cpu_e64_w1__seed5290.svg)
- [calibration__cuda_infer-cpu_e64_w3__seed5290 (complete)](calibration__cuda_infer-cpu_e64_w3__seed5290.png) · [SVG](calibration__cuda_infer-cpu_e64_w3__seed5290.svg)
- [calibration__cuda_infer-cpu_e64_w3__seed5291 (complete)](calibration__cuda_infer-cpu_e64_w3__seed5291.png) · [SVG](calibration__cuda_infer-cpu_e64_w3__seed5291.svg)
- [calibration__cuda_infer-cpu_e64_w3__seed5292 (complete)](calibration__cuda_infer-cpu_e64_w3__seed5292.png) · [SVG](calibration__cuda_infer-cpu_e64_w3__seed5292.svg)
- [calibration__cuda_infer-cuda_e16_w1__seed5290 (complete)](calibration__cuda_infer-cuda_e16_w1__seed5290.png) · [SVG](calibration__cuda_infer-cuda_e16_w1__seed5290.svg)
- [calibration__cuda_infer-cuda_e16_w3__seed5290 (complete)](calibration__cuda_infer-cuda_e16_w3__seed5290.png) · [SVG](calibration__cuda_infer-cuda_e16_w3__seed5290.svg)
- [calibration__cuda_infer-cuda_e16_w3__seed5291 (complete)](calibration__cuda_infer-cuda_e16_w3__seed5291.png) · [SVG](calibration__cuda_infer-cuda_e16_w3__seed5291.svg)
- [calibration__cuda_infer-cuda_e16_w3__seed5292 (complete)](calibration__cuda_infer-cuda_e16_w3__seed5292.png) · [SVG](calibration__cuda_infer-cuda_e16_w3__seed5292.svg)
- [calibration__cuda_infer-cuda_e64_w1__seed5290 (complete)](calibration__cuda_infer-cuda_e64_w1__seed5290.png) · [SVG](calibration__cuda_infer-cuda_e64_w1__seed5290.svg)
- [calibration__cuda_infer-cuda_e64_w3__seed5290 (complete)](calibration__cuda_infer-cuda_e64_w3__seed5290.png) · [SVG](calibration__cuda_infer-cuda_e64_w3__seed5290.svg)
- [calibration__cuda_infer-cuda_e64_w3__seed5291 (complete)](calibration__cuda_infer-cuda_e64_w3__seed5291.png) · [SVG](calibration__cuda_infer-cuda_e64_w3__seed5291.svg)
- [calibration__cuda_infer-cuda_e64_w3__seed5292 (complete)](calibration__cuda_infer-cuda_e64_w3__seed5292.png) · [SVG](calibration__cuda_infer-cuda_e64_w3__seed5292.svg)
- [R0 / seed 5201 (complete)](pilots__R0__seed5201.png) · [SVG](pilots__R0__seed5201.svg)
- [R1 / seed 5201 (complete)](pilots__R1__seed5201.png) · [SVG](pilots__R1__seed5201.svg)
- [R2 / seed 5201 (complete)](pilots__R2__seed5201.png) · [SVG](pilots__R2__seed5201.svg)

Total loss combines the policy, weighted value and entropy terms. Policy loss is a PPO surrogate, not a win rate. Value targets change as the policy and opponents change; different reward scales also change losses. Entropy measures action-distribution spread, not competence. These are logged update averages, with no extra smoothing or confidence bands. Missing/nonfinite metrics remain gaps or Not recorded panels; evidence.json counts them. No new measurements are inferred.

Exact input snapshots are in data/; evidence.json records source revisions, checkpoint hashes, seeds and configuration. Rerun the plotting wrapper with this data directory as --input to reproduce the curves. [TorchRL PPO loss](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.objectives.ClipPPOLoss.html), [Matplotlib](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html).

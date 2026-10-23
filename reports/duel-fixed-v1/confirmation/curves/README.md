# Saved loss graphs

- [Reward, combat, behavior, self-play outcomes and optimizer diagnostics](training/README.md)
- [a-confirm comparison](a-confirm.png) · [SVG](a-confirm.svg)
- [R1 / seed 5101 (complete)](seed5101.png) · [SVG](seed5101.svg)
- [R1 / seed 5102 (complete)](seed5102.png) · [SVG](seed5102.svg)
- [R1 / seed 5103 (complete)](seed5103.png) · [SVG](seed5103.svg)

Total loss combines the policy, weighted value and entropy terms. Policy loss is a PPO surrogate, not a win rate. Value targets change as the policy and opponents change; different reward scales also change losses. Entropy measures action-distribution spread, not competence. These are logged update averages, with no extra smoothing or confidence bands. Missing/nonfinite metrics remain gaps or Not recorded panels; evidence.json counts them. No new measurements are inferred.

Exact input snapshots are in data/; evidence.json records source revisions, checkpoint hashes, seeds and configuration. Rerun the plotting wrapper with this data directory as --input to reproduce the curves. [TorchRL PPO loss](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.objectives.ClipPPOLoss.html), [Matplotlib](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html).

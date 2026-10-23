# Saved loss graphs

- [legacy comparison](legacy.png) · [SVG](legacy.svg)
- [fixed_seed111 / seed 111 (partial or deferred)](fixed_seed111.png) · [SVG](fixed_seed111.svg)
- [fixed_seed222 / seed 222 (partial or deferred)](fixed_seed222.png) · [SVG](fixed_seed222.svg)
- [fixed_seed333 / seed 333 (partial or deferred)](fixed_seed333.png) · [SVG](fixed_seed333.svg)
- [selfplay_seed111 / seed 111 (partial or deferred)](selfplay_seed111.png) · [SVG](selfplay_seed111.svg)
- [selfplay_seed222 / seed 222 (partial or deferred)](selfplay_seed222.png) · [SVG](selfplay_seed222.svg)
- [selfplay_seed333 / seed 333 (partial or deferred)](selfplay_seed333.png) · [SVG](selfplay_seed333.svg)

Total loss combines the policy, weighted value and entropy terms. Policy loss is a PPO surrogate, not a win rate. Value targets change as the policy and opponents change; different reward scales also change losses. Entropy measures action-distribution spread, not competence. These are logged update averages, with no extra smoothing or confidence bands. Missing/nonfinite metrics remain gaps or Not recorded panels; evidence.json counts them. No new measurements are inferred.

Exact input snapshots are in data/; evidence.json records source revisions, checkpoint hashes, seeds and configuration. Rerun the plotting wrapper with this data directory as --input to reproduce the curves. [TorchRL PPO loss](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.objectives.ClipPPOLoss.html), [Matplotlib](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html).

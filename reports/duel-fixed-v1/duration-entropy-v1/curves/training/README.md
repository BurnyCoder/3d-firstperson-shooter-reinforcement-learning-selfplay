# Reward and training behavior

- [Entropy 0.01 / seed 5201: rewards](control__seed5201-rewards.png) · [SVG](control__seed5201-rewards.svg)
- [Entropy 0.01 / seed 5201: combat](control__seed5201-combat.png) · [SVG](control__seed5201-combat.svg)
- [Entropy 0.01 / seed 5201: behavior](control__seed5201-behavior.png) · [SVG](control__seed5201-behavior.svg)
- [Entropy 0.01 / seed 5201: selfplay-outcomes](control__seed5201-selfplay-outcomes.png) · [SVG](control__seed5201-selfplay-outcomes.svg)
- [Entropy 0.01 / seed 5201: optimization](control__seed5201-optimization.png) · [SVG](control__seed5201-optimization.svg)
- [Entropy 0.001 / seed 5201: rewards](low-entropy__seed5201-rewards.png) · [SVG](low-entropy__seed5201-rewards.svg)
- [Entropy 0.001 / seed 5201: combat](low-entropy__seed5201-combat.png) · [SVG](low-entropy__seed5201-combat.svg)
- [Entropy 0.001 / seed 5201: behavior](low-entropy__seed5201-behavior.png) · [SVG](low-entropy__seed5201-behavior.svg)
- [Entropy 0.001 / seed 5201: selfplay-outcomes](low-entropy__seed5201-selfplay-outcomes.png) · [SVG](low-entropy__seed5201-selfplay-outcomes.svg)
- [Entropy 0.001 / seed 5201: optimization](low-entropy__seed5201-optimization.png) · [SVG](low-entropy__seed5201-optimization.svg)
- [duration entropy: rewards](duration-entropy-rewards.png) · [SVG](duration-entropy-rewards.svg)
- [duration entropy: combat](duration-entropy-combat.png) · [SVG](duration-entropy-combat.svg)
- [duration entropy: behavior](duration-entropy-behavior.png) · [SVG](duration-entropy-behavior.svg)
- [duration entropy: selfplay-outcomes](duration-entropy-selfplay-outcomes.png) · [SVG](duration-entropy-selfplay-outcomes.svg)
- [duration entropy: optimization](duration-entropy-optimization.png) · [SVG](duration-entropy-optimization.svg)

[Definitions, source references and reproduction](../../../../docs/experiment-plots.md). Adjacent per-run JSON saves every plotted number (null means undefined), configuration, seed, source and checkpoint hashes. Exact metric bytes are in the parent gallery's data/ directory. Rewards are interval means per arena step, not episode returns. Self-play outcomes use changing opponents; fixed validation belongs in the parent gallery. These are exploratory descriptive plots of all recorded updates, without smoothing or independent-model error bands.

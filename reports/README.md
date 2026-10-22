# Experiment evidence

Training results are not yet available in this initial implementation commit. This file deliberately makes no claim of learned tactical competence.

`uv run csrl report --input runs/suite --output reports` generates individual experiment reports, an aggregate table, measured learning curves, and a PDF paper naming Libor Burian as author. The suite preserves calibration, failed/timed-out runs, initial checkpoints, final checkpoints, and raw evaluation rounds. See the frozen analysis plan in [the protocol](../docs/protocol.md).

Implementation verification before training: simulation, baseline, learning-interface, and renderer tests; native 3D screenshots; exact rendered/headless replay comparison. These validate software behavior, not the learning hypothesis.

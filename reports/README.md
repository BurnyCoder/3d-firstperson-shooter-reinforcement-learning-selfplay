# Experiment evidence

The [verification summary](verification/summary.md) contains CPU and CUDA smoke runs, their raw evidence, individual reports, and a [PDF paper](verification/paper.pdf). Both initial/final smoke evaluations scored 0.167 over 24 rounds; these checks establish a working pipeline, not improved tactical competence. The six-run primary comparison is pending.

`uv run csrl report --input runs/suite --output reports` generates individual experiment reports, an aggregate table, measured learning curves, and a PDF paper naming Libor Burian as author. The suite preserves calibration, failed/timed-out runs, initial checkpoints, final checkpoints, and raw evaluation rounds. See the frozen analysis plan in [the protocol](../docs/protocol.md).

Implementation verification before training: simulation, baseline, learning-interface, and renderer tests; native 3D screenshots; exact rendered/headless replay comparison. These validate software behavior, not the learning hypothesis.

[Engineering verification notes](verification-notes.md) record issues, fixes, throughput measurements, and the limits of the smoke evidence. The final suite will report all six runs, calibration, and failed or incomplete outcomes under the unchanged analysis plan.

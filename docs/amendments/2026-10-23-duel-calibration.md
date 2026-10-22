# Stage A calibration resolution — 2026-10-23

This resolves the engineering selection procedure registered in [the active protocol](../duel-protocol.md). It does not change game, rewards, architecture, learning settings or the original 5v5 deadlines. The new Stage A clock began **2026-10-22 23:30:52.627932 UTC**; its immutable deadline is **2026-10-23 11:30:52.627932 UTC**. Calibration/pilots end by 01:30:52 UTC and confirmation by 09:30:52 UTC. Berlin local dates are one day later than the calibration's UTC start date.

## Measured resolution

All nine cells completed four updates per worker with finite recorded optimizer metrics and no warning/error/traceback in worker logs. Source was merge `6db495937074e63876863b364654aef36a904345`, published by [PR4](https://github.com/BurnyCoder/counter-strike-reinforcement-learning-selfplay/pull/4) before the clock. Full probe sequence, including preflight/startup, took under four minutes. [Raw group receipts, telemetry, per-worker logs/configurations/metrics and hashes](../../reports/duel-fixed-v1/calibration/) are preserved. Diagnostic checkpoints remain local and cannot initialize a pilot or deployment model.

| Optimizer / learner inference | Arenas / workers | Aggregate arena steps/s | Minimum RAM available | Minimum VRAM available | Eligible |
|---|---:|---:|---:|---:|---|
| CPU / CPU | 16 / 1 | 489.07 | 28.37% | 73.62% | yes |
| CUDA / CPU | 16 / 1 | 433.30 | 22.70% | 71.87% | yes |
| CUDA / CPU | 16 / 3 | 1303.08 | 11.71% | 63.80% | no |
| CUDA / CPU | 64 / 1 | 1304.98 | 23.04% | 70.72% | yes |
| CUDA / CPU | 64 / 3 | 4271.12 | 11.29% | 62.51% | no |
| CUDA / CUDA | 16 / 1 | 435.10 | 23.92% | 71.02% | yes |
| CUDA / CUDA | 16 / 3 | 1301.01 | 12.38% | 65.19% | no |
| CUDA / CUDA | 64 / 1 | 1300.94 | 24.42% | 71.23% | **selected** |
| CUDA / CUDA | 64 / 3 | 4265.37 | 12.12% | 62.72% | no |

The four three-worker cells fail the predeclared 20% host-memory headroom rule even though they are faster. Among eligible cells, CUDA/CPU64 and CUDA/CUDA64 differ by only 0.31%, within the registered 5% tie range. CUDA/CUDA64 uses less observed host and GPU memory, so it wins the lower-memory tie rule without needing an arbitrary memory-priority decision. Use **one worker, 64 arenas, one PyTorch CPU thread, CUDA learner inference and optimization**. Frozen opponent inference remains CPU as implemented in `SnapshotPool`. Three confirmation seeds will execute serially with independent state.

These are one short measurement per cell, not precise repeated performance estimates. Startup dominates some small cells; later historical-opponent diversity may reduce sustained throughput. Memory readings include other host applications. No causal claim about GPU utilization is made. Reproduce selection from the retained original run directory with `uv run python scripts/analyze_duel_calibration.py`; do not repeat optimizer probes.

## Frozen dependent settings

Selected full-pipeline rate is 1300.936954105218 arena steps/s. Batch size is 64×128=8192 arena-time rows. The registered formula `floor(0.8 × rate × 1200 / 8192) × 8192` gives **1,245,184 arena steps (152 updates)** per reward pilot, with **1200 seconds maximum training time**. All attempted R0/R1/R2 pilots must use this identical target and seed5201, and verify identical initial actor parameters. Learning and reward definitions remain those in the registration. [R0's resolved config](../../configs/duel-pilot-R0.json) is runnable after this amendment is committed and pushed. R1/R2 remain blocked until their conditional diagnosis gates. No confirmation target is frozen yet.

The preconfirmation [power sensitivity](../../reports/duel-fixed-v1/power.json) yields 90.85%, 47.07% and 17.93% power for assumed paired-model standard deviations 0.05, 0.10 and 0.20, respectively, with effect0.20, n3, two-sided alpha0.05. This follows `paired_power()` and [SciPy's noncentral-t distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nct.html), with the methodological limitations discussed by [Colas et al.](https://arxiv.org/abs/1806.08295). These are assumptions, not measured variances or promised significance.

## Setup and verification accounting

A preceding `uv run` invocation failed while reinstalling editable package metadata, before application entry or optimizer work. The existing project-local `ministrike-0.1.0.dist-info` directory had the Windows ReadOnly attribute. Removing that attribute from the verified exact directory and rerunning locked synchronization succeeded; the cause of that attribute is unknown. No environment was deleted and no antivirus explanation is asserted. This pre-clock installation failure is preserved here; subsequent initialization, collection, optimization, serialization and pauses consume the clock.

Windows push and PR CI passed on implementation head `515bf984707e48eaab1f317a49293a04203d4491`; full local suite75 passed. The [substantive implementation review](../review-stage-a.md), baseline native screenshots and rendered/headless parity precede these experiments. The [historical learning audit](../../reports/learning-audit/summary.md) is resolved with limitations; its old zero wins, positive-but-failed-threshold shooting control and OOM remain distinct evidence.

# Probe parallel_6

Six independent CUDA learners each used 16 arenas and one PyTorch CPU thread. Each completed four measured updates after one warmup. Their combined 49,152 arena steps covered a 15.019915-second measurement envelope: **3,272.46 aggregate steps/s**. Group process wall time including startup and warmup was 30.250 seconds.

| Worker | Condition | Seed | Individual steps/s | Collection share | Optimization share |
|---|---|---:|---:|---:|---:|
| [0](data/wide/parallel_6_0/performance.json) | Self-play | 8002 | 588.74 | 84.55% | 15.02% |
| [1](data/wide/parallel_6_1/performance.json) | Fixed | 8003 | 733.96 | 81.35% | 18.08% |
| [2](data/wide/parallel_6_2/performance.json) | Self-play | 8004 | 594.00 | 83.95% | 15.60% |
| [3](data/wide/parallel_6_3/performance.json) | Fixed | 8005 | 737.24 | 81.06% | 18.38% |
| [4](data/wide/parallel_6_4/performance.json) | Self-play | 8006 | 604.03 | 83.86% | 15.69% |
| [5](data/wide/parallel_6_5/performance.json) | Fixed | 8007 | 738.17 | 80.56% | 18.88% |

The individual rates are not added because their measurement intervals differ. Each worker's peak live PyTorch CUDA allocation was 156,522,496 bytes. All succeeded. Per-worker rates declined versus four workers, but aggregate throughput increased 11.15%. Six was selected to execute the six already-planned primary learners concurrently with 96 total arenas and unchanged per-model batches. No larger worker count was probed because the primary experiment contains six independent models. This is a practical measured choice, not a global hardware optimum.

See the [shared method, execution amendment, and limitations](summary.md) and [sweep receipt](data/wide/sweep.json). The adjacent worker directories preserve full configurations and per-update timings. The throughput ratio does not measure playing skill, nor does it guarantee the same rate during longer rounds or a thermally stable long run.

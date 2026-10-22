# Probe parallel_4

Four independent CUDA learners each used 16 arenas and one PyTorch CPU thread. Each completed four measured updates after one warmup. Their combined 32,768 arena steps covered an 11.129538-second measurement envelope: **2,944.24 aggregate steps/s**. Group process wall time including startup and warmup was 26.140 seconds.

| Worker | Condition | Seed | Individual steps/s | Collection share | Optimization share |
|---|---|---:|---:|---:|---:|
| [0](data/wide/parallel_4_0/performance.json) | Self-play | 8002 | 760.89 | 82.02% | 17.64% |
| [1](data/wide/parallel_4_1/performance.json) | Fixed | 8003 | 837.33 | 77.51% | 22.04% |
| [2](data/wide/parallel_4_2/performance.json) | Self-play | 8004 | 737.01 | 83.31% | 16.33% |
| [3](data/wide/parallel_4_3/performance.json) | Fixed | 8005 | 845.72 | 76.93% | 22.56% |

The individual rates are not added because their measurement intervals differ. Each worker's peak live PyTorch CUDA allocation was 156,522,496 bytes. All succeeded. This balanced condition mix retained good per-worker throughput and raised total throughput, motivating the six-worker probe. This measurement is not a sustained training-rate estimate and supplies no policy-quality evidence.

See the [shared method and limitations](summary.md) and [sweep receipt](data/wide/sweep.json). The adjacent worker directories preserve full configurations and per-update timings.

# Probe parallel_t4

Two independent CUDA learners each used 16 arenas and four PyTorch CPU threads. Each completed four measured updates after one warmup. Their combined 16,384 arena steps covered a 12.612006-second measurement envelope: **1,299.08 aggregate steps/s**. Group process wall time including startup and warmup was 26.687 seconds.

| Worker | Condition | Seed | Individual steps/s | Collection share | Optimization share |
|---|---|---:|---:|---:|---:|
| [0](data/sweep1/parallel_t4_0/performance.json) | Self-play | 8002 | 649.55 | 83.63% | 16.01% |
| [1](data/sweep1/parallel_t4_1/performance.json) | Fixed | 8003 | 754.28 | 79.97% | 19.58% |

The individual rates are not added because their measurement intervals differ. Each worker's peak live PyTorch CUDA allocation was 156,522,496 bytes. Both succeeded. Four threads per process gave lower observed aggregate throughput than the otherwise matched [one-thread pair](parallel_t1.md), supporting one thread for wider probes. A single trial cannot establish a stable threading optimum or attribute the difference uniquely to CPU contention.

See the [shared method and limitations](summary.md) and [sweep receipt](data/sweep1/sweep.json). The adjacent worker directories preserve full configurations and per-update timings.

# Probe parallel_t1

Two independent CUDA learners each used 16 arenas and one PyTorch CPU thread. Each completed four measured updates after one warmup. Their combined 16,384 arena steps covered an 11.044978-second measurement envelope: **1,483.39 aggregate steps/s**. Group process wall time including startup and warmup was 24.078 seconds.

| Worker | Condition | Seed | Individual steps/s | Collection share | Optimization share |
|---|---|---:|---:|---:|---:|
| [0](data/sweep1/parallel_t1_0/performance.json) | Self-play | 8002 | 741.66 | 82.14% | 17.50% |
| [1](data/sweep1/parallel_t1_1/performance.json) | Fixed | 8003 | 846.57 | 77.92% | 21.59% |

The individual rates are not added: measurement intervals were not identical. Each worker's peak live PyTorch CUDA allocation was 156,522,496 bytes. Both succeeded. Their overlap increased measured aggregate throughput while retaining the per-model rollout size. Comparing this mixed-condition group with a self-play-only reference combines concurrency and workload mix; the direct thread comparison is [parallel_t4](parallel_t4.md). No repeated measurements or quality evaluations were performed.

See the [shared method and limitations](summary.md) and [sweep receipt](data/sweep1/sweep.json). The adjacent worker directories preserve full configurations and per-update timings.

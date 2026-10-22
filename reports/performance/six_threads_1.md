# Follow-up: six learners, one CPU thread each

This repeated the six-worker, balanced-condition, 16-arena configuration while the primary suite was stopped at exact periodic checkpoints. Stopping released its RAM and CUDA contexts; paused resident learners would have distorted memory pressure. The same production calibration completed one warmup and four measured updates per worker with seeds 8002 through 8007.

The group processed 49,152 measured arena steps at **3,026.17 steps/s**. Complete process wall time, including startup and warmup, was **34.813 seconds**. Individual worker rates were 531.95, 603.94, 516.03, 603.70, 514.42 and 604.49 steps/s. The aggregate uses the shared measurement envelope, not the sum of these rates.

This was slower than the earlier six-worker probe and faster than the immediately following two-thread probe. Both measurements are retained; one short repetition does not establish a stable performance distribution. The primary suite retained one thread per learner. No final-test scores informed this choice.

See the [raw sweep receipt](data/threads/sweep.json), adjacent worker configurations/timings, [checkpoint pause audit](data/thread-calibration-pause.json), and [shared method and limitations](summary.md).

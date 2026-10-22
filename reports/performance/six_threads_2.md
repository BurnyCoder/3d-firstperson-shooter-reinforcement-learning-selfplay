# Follow-up: six learners, two CPU threads each

This changed PyTorch's intra-operation CPU thread count from one to two in each of six independent workers. Arena count, model architecture, seed identities, rollout length, PPO settings, standard starts and snapshot-grouping setup matched the preceding one-thread probe. Each worker performed one warmup and four measured production updates.

The group processed 49,152 measured arena steps at **2,774.65 steps/s**; complete process wall time was **35.735 seconds**. This aggregate rate was **8.31% lower** than the preceding one-thread probe's 3,026.17 steps/s, so two threads were not adopted. More threads did not improve measured completion speed on this workload.

A separate 25-sample host observation overlapping this comparison measured mean total CPU utilization of 93.33% and minimum available RAM of 2.285 GiB. These samples include startup, transitions and any unrelated host work; they are not per-worker measurements or a controlled causal estimate. The short sequential comparison has no significance claim, confidence interval, or demonstrated thermal steady state.

See the [raw sweep receipt](data/threads/sweep.json), [host samples](data/threads/host-telemetry.json), [one-thread reference](six_threads_1.md), and [shared method and limitations](summary.md). One thread per learner was retained, and the primary suite resumed with its original learning targets and deadline.

# Deferred 5v5: fixed_seed111

**Incomplete, not a deployment recommendation.** Seed 111, condition `fixed`, completed 2,682,880 of 4,167,680 arena steps (1,310 updates) before the user deferred this experiment. No deadline or target was extended and no final tournament was completed.

The last canonical training record reports 12,575 rounds, 0 wins and 0 draws. These are changing-opponent training outcomes, not fixed-benchmark scores or evidence of superiority over the other condition. Unequal completed exposure prevents the registered equal-budget comparison.

[Original config](data/fixed_seed111/config.json), [canonical metrics](data/fixed_seed111/metrics.jsonl), [derived archive status and both checkpoint hashes](data/fixed_seed111/archive-status.json). Initial source identity: `38697fee4af941b12cfd15489424e2750ef4dd1d`. The archive's checkpoint elapsed time is 4168.479187 seconds; [the suite ledger](data/suite.json) additionally preserves interruption/recovery accounting. A final metric timestamp can follow checkpoint serialization.

The [original protocol](../../docs/protocol.md) defines the tactical map, 5v5 rules, two-layer 128-unit shared actor, two-layer 256-unit centralized critic, MAPPO/PPO, curriculum, opponents and shaped reward. Fixed-condition training uses the historical scripted controller; this is outside the later self-play-only Stage A protocol.

[Learning audit](../learning-audit/summary.md), [hardware probes and interruptions](../performance/summary.md), [loss graphs](../learning-audit/losses/README.md), [aggregate limitations](README.md). [Public checkpoint pair](https://huggingface.co/BurnyCoder/ministrike-fixed-seed111).

# Deferred advanced 5v5 experiment

This six-run experiment is preserved as incomplete advanced-stage evidence. It was deferred by the user while investigating zero wins in the fixed-opponent condition. It was not restarted, its deadlines were not extended, and its checkpoints are not treated as trained tactical bots. The [original registration](../../docs/protocol.md), [suite ledger](data/suite.json), [learning audit](../learning-audit/summary.md) and [hardware/interruption report](../performance/summary.md) retain chronology and provenance.

| Condition | Seed | Completed / target arena steps | Updates | Training wins / rounds |
|---|---:|---:|---:|---:|
| [Fixed script](fixed_seed111.md) | 111 | 2,682,880 / 4,167,680 | 1,310 | 0 / 12,575 |
| [Fixed script](fixed_seed222.md) | 222 | 2,682,880 / 4,167,680 | 1,310 | 0 / 13,673 |
| [Fixed script](fixed_seed333.md) | 333 | 2,662,400 / 4,167,680 | 1,300 | 0 / 15,577 |
| [Historical self-play](selfplay_seed111.md) | 111 | 3,604,480 / 4,167,680 | 1,760 | 1,609 / 3,227 |
| [Historical self-play](selfplay_seed222.md) | 222 | 1,904,640 / 4,167,680 | 930 | 885 / 1,790 |
| [Historical self-play](selfplay_seed333.md) | 333 | 2,211,840 / 4,167,680 | 1,080 | 1,027 / 2,049 |

The conditions faced different opponents during training, so their training-win counts cannot establish a self-play advantage. Their unequal completed step budgets and absent complete final tournaments also prevent the registered primary comparison. [Raw records, checkpoint hashes and exact metric boundaries](audit.json). The advanced architecture used the same hidden-layer sizes as the later duel, with ten-player local observations and five-player learner-team critic input; dimensions are derived from match specifications rather than interchangeable across checkpoints.

The audit found excessive reload occupancy and saturated aiming in sampled fixed-condition behavior; it did not establish a PPO/GAE sign error. An old interval damage metric used learner roles after reset and is excluded from causal learning claims; complete-round audit measures are retained. The old scripted controller also had continuous aim, whereas the later duel benchmark uses the same categorical action decoder as learned policies. These are limitations of the legacy comparison, not silently repaired training data. See the [audit's controls, failed firing fixture and sources](../learning-audit/summary.md).

Two earlier interrupted executions and a checkpoint-safe calibration pause are preserved under the suite ledger's execution amendments. Uncheckpointed metric suffixes were archived, resumed identities were retained, and interrupted elapsed compute was charged. A vanished WMI-provider parent is not by itself a worker failure. User-reported Codex restarts are consistent with the interruption times but do not prove the cause. Original training cutoff: `2026-10-23T05:35:52.872233Z`; original overall cutoff: `2026-10-23T07:32:38.503505Z`.

All six initial/latest pairs are public in the [existing collection](https://huggingface.co/collections/BurnyCoder/ministrike-staged-shooter-self-play-6ada65970d163e293d48f3ad), labeled **deferred and incomplete**. The latest checkpoint is an archived endpoint, not a completed final model. Reproduce these reports without training using `uv run python scripts/report_deferred.py` after obtaining the archived artifacts. [Public file verification](../publication/README.md).

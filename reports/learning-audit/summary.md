# Historical learning audit and deferred 5v5 suite

Resolved2026-10-23 by preserving evidence and deferring the advanced task, as requested by the user. No claim of learned5v5 competence or self-play superiority is supported. The original [protocol](../../docs/protocol.md), [audit registration](../../docs/learning-audit.md) and deadlines remain unchanged. This is not a Stage A result.

## What was measured

The frozen behavior audit at source `6acbafcf6afabdaa80f8464aff26850137d11dfb` used16 side-swapped validation worlds (32 rounds per policy/scenario), seed1100010, stochastic learned actions, and the balanced scripted baseline. In standard rounds the sampled fixed checkpoint scored0, dealt0.78125 mean damage, spent0.870541 of live decision time reloading and0.699837 in steep aim. The sampled self-play checkpoint also scored0 against that benchmark. Identical scripted policies scored0.5 with sides swapped, supporting the tested score bookkeeping. Full raw measurements and frozen checkpoint hashes are in [behavior-audit.json](data/behavior-audit.json) and [snapshot.json](data/snapshot.json). These are two sampled checkpoints, not all six final models.

The full-chain passive-target control used the production collector, GAE, PPO, optimizer, serialization and actor inference, but a simpler four-tick one-hit shooting task. At source `33e597b6639614141ba6b15ebbe2102ff30b1de9`, seed91001 completed65,536 arena steps/32updates in145.078recorded seconds. Its512-world stochastic validation score rose from194/512=0.37890625 to381/512=0.744140625, improvement0.365234375. It **failed** the preregistered final-score>=0.80 gate despite exceeding the improvement>=0.20 gate. See [probe-result.json](data/probe-result.json), [configuration](data/probe-config.json) and [protocol](data/probe-protocol.json). This is diagnostic scripted/passive opposition, not pure-self-play evidence and never a deployment candidate. Seed91002 has an initial checkpoint/config but no completed training result after the recorded memory failure; no replacement seed or rerun is counted.

## Original suite at deferral

The canonical metric tails retained after checkpoint-safe recovery were:

| Run | Arena steps | Updates | Training rounds | Training wins |
|---|---:|---:|---:|---:|
|fixed111|2,682,880|1310|12,575|0|
|fixed222|2,682,880|1310|13,673|0|
|fixed333|2,662,400|1300|15,577|0|
|selfplay111|3,604,480|1760|3,227|1,609|
|selfplay222|1,904,640|930|1,790|885|
|selfplay333|2,211,840|1080|2,049|1,027|

Source: [deferral receipt](data/deferral.json), copied canonical last metric records and original manifest identity. These wins are **training diagnostics against different opponents**, not comparable held-out evidence of an advantage. All runs stopped short of the frozen4,167,680-step target. Preserve interrupted suffixes and checkpoint/source identities; completed old work must not be retrained. Ten earlier hardware probes and their raw evidence remain in the [performance report](../performance/summary.md); do not compare their5v5 throughput directly to a duel without measurement.

## Interpretation and correction scope

Exact small GAE and positive-action gradient fixtures test the time axis, sign, PPO ratios and masking. They passed, so those particular bugs were not demonstrated. They do not prove the absence of all implementation errors. Observed reload occupancy and steep aim are behavioral failure modes; their causal contribution is not isolated by this audit. The legacy script uses continuous clipped angular corrections while learned bots use categorical increments, giving the benchmark finer control precision; this is corrected in the new duel benchmark, without rewriting historical results.

The read-only rollout audit's `learner_damage` calculation used roles after resets and is invalid. Exclude that field; the complete-round behavior audit above is the correct source. Raw invalid records remain archived with this disclosure. Hardware concurrency during diagnostics also exhausted host memory; Stage A prohibits overlapping heavy diagnostics with confirmation. The new progression is an engineering response to the combined evidence and user requirements, not a proven causal explanation of the old failure.

## Sources, uncertainty and reproducibility

[TorchRL PPO/GAE](https://docs.pytorch.org/rl/stable/tutorials/multiagent_ppo.html) supplies the learning implementation. [Sample Factory](https://proceedings.mlr.press/v119/petrenko20a/petrenko20a.pdf) is relevant shooter-self-play context but does not determine a MiniStrike learning-time expectation. [Colas et al.](https://arxiv.org/abs/1806.08295) and [Agarwal et al.](https://arxiv.org/abs/2108.13264) motivate explicit independent-run uncertainty. One successful diagnostic seed plus one failure cannot establish reliable learning; the original hypothesis remains unsupported/incomplete. Hashes of copied raw files are in [evidence.json](data/evidence.json). Reproduce numeric summaries directly from those records; no final-test data was used to select the new protocol.

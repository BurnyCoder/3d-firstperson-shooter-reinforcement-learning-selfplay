# Evidence and claim index

This index maps claims to records. A registration is not a result. Stage A has no measured learning result until its manifest and raw records exist. See [sources](sources.md) for methodological citations and [roadmap](experiment-roadmap.md) for planned work.

| Claim class | Canonical evidence / reproducible route | Status |
|---|---|---|
|Approved Stage A design/thresholds |[active registration](duel-protocol.md), `configs/experiments.json`, selected `MatchSpec` |Engineering decisions, not empirical findings |
|Shared simulator correctness |`tests/test_sim.py`, `tests/test_duel.py`; `uv run pytest` plus saved test receipt |Requires passing current revision; do not substitute old receipts |
|Legacy5v5 protocol and accounting |[original registration](protocol.md), `runs/suite/suite.json`, `launch.json`, `execution_amendments/` |Deferred, never reset/retrain to hide failures |
|Ten previous hardware probes |[performance summary](../reports/performance/summary.md), individual reports and their raw data links |Historical workload evidence; not duel throughput |
|Zero-win learning audit |[audit](learning-audit.md), `runs/diagnostics/audit-status.json`, diagnostic raw records |Historical diagnostic results; not deployment quality |
|Stage A elapsed budget/source identity |`runs/duel-fixed-v1/manifest.json`, launch preflight and dated amendments |No clock before published optimizer-bearing experiment |
|Calibration and reward selection |[nine-cell resolution](amendments/2026-10-23-duel-calibration.md), [raw records and hashes](../reports/duel-fixed-v1/calibration/), `runs/duel-fixed-v1/pilots/` |Calibration complete; R0 is the first conditional learning pilot, not confirmation |
|Confirmation and learning quality |Per-run config, initial/model checkpoints, SHA256, metrics and initial/final validation/test JSON |Pending; seed and checkpoint identity required beside any reported number |
|Statistical interpretation |Raw paired rows, power calculation, analysis script and report |Separate match and independent-model uncertainty; no pseudo-replicated deterministic trajectories |
|Deployment behavior |Frozen validation selection, native screenshots/replays of selected checkpoint, replay parity test |Pending trained Stage A model; `docs/images/duel-*-baseline.png` depicts scripted duel baselines, older images depict legacy5v5 |
|Failed Stage A launch |[recovery amendment](amendments/2026-10-23-pilot-launch-recovery.md), [original receipts/errors](../reports/duel-fixed-v1/launch-failure/) |Zero-update status-file failure preserved; clock unchanged |
|Public artifact identity |Git commits/PR/checks, HF commit IDs and collection items |Publish all confirmation outcomes, label failed acceptance experimental |

Every generated table/figure/report must identify raw input paths+hashes, config/seeds, source revision, checkpoint hashes and the command regenerating it. Put lengthy derivations here or in protocol/architecture docs rather than duplicating them across README/model cards. Claims without evidence remain explicitly pending, inferred or removed.

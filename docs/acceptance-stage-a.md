# Stage A acceptance ledger

**Experimental release: the study is complete, but the moving-script capability gate failed.** Implementation or publication completion is not scientific acceptance. Original clocks and failed experiments remain preserved. Later stages have no new budget.

## Registration and implementation

- [x] [Roadmap](experiment-roadmap.md), [protocol](duel-protocol.md), [sources](sources.md), [evidence index](evidence-index.md) and resolved configurations published before their optimizer work; per-worker preflight identities retained.
- [x] Shared simulation, versioned match/observations/actions/rewards, partial resets, visibility isolation, simultaneous outcomes and immutable opponents implemented and tested.
- [x] TorchRL collection, GAE time axes, PPO ratios/gradient direction, death masking, checkpoint compatibility and deadlines checked. The [substantive review](review-stage-a.md) is retained; it was a self-review, not independent external review.
- [x] Final full local suite: 108 tests passed; Ruff passed. After the presentation-only F10 correction, seven view/controller tests and a native exit/replay check passed. [Release verification and logs](../reports/duel-fixed-v1/release-verification.md).
- [x] Fresh project-local Python 3.12 installation from `uv.lock`, CUDA execution, public checkpoint download and restricted loading verified. The original environment's orphaned package metadata warning remains separately documented.
- [x] Real production save/resume matched all 12 comparisons over two updates and 768 newly executed arena steps. [Exact audit and limited scope](../reports/duel-fixed-v1/egocentric-v1/resume.md).

## Registered experiments and inference

- [x] [Deferred 5v5 suite](../reports/deferred-5v5/README.md), interrupted attempts, OOM/control failures and [learning audit](../reports/learning-audit/summary.md) preserved without restarting or extending deadlines.
- [x] All ten earlier hardware probes and nine Stage A calibration cells reported; selection used complete throughput with memory headroom. No later calibration repeated.
- [x] Matched [R0](../reports/duel-fixed-v1/pilot-R0.md), [R1](../reports/duel-fixed-v1/pilot-R1.md) and [R2](../reports/duel-fixed-v1/pilot-R2.md) pilots completed; all failed their registered primary gate.
- [x] [Action-mode](../reports/duel-fixed-v1/action-modes-v1.md) and [duration/entropy](../reports/duel-fixed-v1/duration-entropy-v1/README.md) diagnostics completed with negative hypothesis outcomes. All 774 repeated controls and the 152-update tensor reproduction were exact.
- [x] [Observer-coordinate intervention](../reports/duel-fixed-v1/egocentric-v1/README.md) completed and passed its exploratory screen; only the frozen recipe, not diagnostic weights, entered confirmation.
- [x] [Power sensitivity and equal targets](amendments/2026-10-23-confirmation-resolution.md) published prospectively. Fresh 5101/5102/5103 each completed 5,464,064 steps and 667 updates within the original cutoff.
- [x] All 24 initial/final validation/test endpoints completed; [selection](../reports/duel-fixed-v1/confirmation/data/selection.json) froze seed 5101 before final-test access. No tuning or seed replacement followed.
- [x] All outcomes reported with independent-model t intervals and separate paired conditional match intervals. Deterministic fixed trajectories are not pseudo-replicated. [Confirmation report and exact analysis](../reports/duel-fixed-v1/confirmation/README.md).
- [x] Primary mean improvement 0.415365 exceeds 0.20; random/initial and elimination criteria pass. **Moving-script median 0.25 is below 0.55, so overall capability acceptance fails.** This is a measured negative outcome, not an unfinished experiment.

## Gameplay, reporting and publication

- [x] Selected confirmation checkpoint inspected in arena watch, bot first-person watch, explicit human first-person play, takeover and hand-back. Nine captures replay exactly headlessly. F10 debug-key collision corrected and rechecked. [Images, logs, parity and scope](../reports/duel-fixed-v1/confirmation/native/README.md).
- Native **WASD movement and mouse aiming were not successfully delivered by the remote input tool**. Focused input-mapping tests pass, and human firing/ownership/cameras were observed; exhaustive physical keyboard/mouse playtesting is not claimed.
- [x] Per-experiment Markdown, [all-experiment comparison](../reports/summary.md), 135 PNG/SVG chart pairs with raw inputs/hashes, and the sourced ten-page [paper by Libor Burian](../reports/duel-fixed-v1/paper.pdf) saved and visually inspected.
- [x] Numerical summaries recomputed; source/config/checkpoint/figure identities verified; local documentation links checked; method citations, engineering choices and license boundaries reviewed. Analysis-only reproduction succeeds. [Release audit](../reports/duel-fixed-v1/release-verification.md).
- [x] All 15 appropriate initial/final pairs, configurations, metrics and cards published through HF CLI; all 150 uploaded files match local bytes and public collection membership. Two older smoke entries remain, making 17 collection entries. [Publication audit](../reports/publication/README.md).
- [x] Implementation and evidence submitted to attached [PR5](https://github.com/BurnyCoder/3d-firstperson-shooter-reinforcement-learning-selfplay/pull/5); its live checks and merge record identify the release revision. Prior substantive review is reused; minor report/UI corrections do not change learning methods.

The heartbeat is removed after the final passing-check merge and publication audit. Its deletion receipt belongs to the release chat. No later roadmap entry is launched automatically.

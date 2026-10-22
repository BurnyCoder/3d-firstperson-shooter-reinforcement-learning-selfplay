# Stage A substantive implementation review

Review date2026-10-23; base implementation published through `f90b18320a0a6e57ce3ae27fb81efb9c2a96c222`, before any Stage A optimizer experiment. This is a self-review of correctness, security, reliability, tests and architecture, not independent external review. Prior PR3 performance review is retained; it was not repeated.

## Correctness and design

Reviewed shared movement/LOS/hitscan against versioned map geometry, stable player slots, simultaneous damage before elimination/timeout, effective damage caps, terminal potential zeroing and the R2 discounted identity. The original bomb state machine is not invoked for duels. Dynamic observation/player/action dimensions preserve the original384-input/six-head5v5 interface while duels use192/five heads. Interaction has no categorical entropy term. Actor inputs remain visibility-filtered and the critic never enters inference.

TorchRL owns collection, selective resets, GAE and clipped PPO. Time remains axis1 before flattening arena/time; live-before-action masks affect actor/entropy but not delayed team critic returns. CUDA collection/GAE/loss shape smoke verification performed **without optimizer updates**, using the library's `auto_cast_to_device` instead of a new transport wrapper. Expected loss/clip/KL keys were present. Whole-configuration preflight checks registered runnable status and current remote source identity. Resume rejects incompatible match and changed implementation-tree identity; documentation-only commits do not redefine the learning implementation.

Review corrections: removed an unnecessary custom policy-device wrapper and unreachable legacy CLI branch; made the scripted duel benchmark use the exact action decoder; added explicit match mismatch rejection on resume; documented that R2's individual potential terms need not be zero-sum; kept shared random-world blocks together across baseline/side/checkpoint bootstrap resampling; removed the disabled interaction hint from duel human controls.

## Security and reliability

Checkpoint loading remains `weights_only=True` with explicit NumPy encoding, not unrestricted pickle. Subprocesses use argument lists, no interpolated shell; Windows workers use a hidden console flag. Public upload must use the existing artifact allowlist, never recursive workspace uploads. Credentials/.env/environments/caches remain ignored. Published run configs and their hashes are checked; only the canonical Stage A manifest can start the new clock. Recovery archives uncheckpointed metric suffixes before continuation and preserves original optimizer/RNG/opponent/environment state. Completed results cannot be retrained by suite resume.

The historical suite/audit manifests were copied to `runs/suite/execution_amendments/staged-deferral-20261022T232354Z` before status amendments. Original deadline fields remain unchanged. The untracked user file `scripts/benchmark_throughput (1).py` was left untouched and excluded from commits.

## Verification and remaining release gates

Full local Windows suite:75 passed, before final review corrections; ruff and whitespace checks passed. Final exact-revision CI/check receipt follows in the acceptance ledger. Native flat-arena/free-camera and bot-first-person screenshots were inspected; a40tick recorded sequence reproduced the rendered snapshot exactly in headless simulation. Human mode opened with the correct team1 slot and health/ammunition display. These are scripted-baseline checks, not trained quality evidence. Final trained mode verification, complete optimizer/save/resume/evaluate workflow, budgeted calibration/pilots/confirmation, numerical claim audit and publication remain explicit pending gates. No serious unresolved implementation finding is known; learning success remains an empirical question.

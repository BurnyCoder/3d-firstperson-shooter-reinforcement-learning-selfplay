# Stage A acceptance ledger

This is an implementation/release checklist, not a learning result. Follow [registration](duel-protocol.md), [roadmap](experiment-roadmap.md) and newest run manifest. Check items only with linked evidence.

- [x] Canonical roadmap, active protocol, bibliography/evidence index created before Stage A optimizer work.
- [x] Shared match-specific simulation, duel geometry/rules/rewards and dynamic actor/critic dimensions implemented.
- [x] Focused rule, observation/action, adapter/GAE/PPO and takeover tests pass locally (47 tests before analysis tests were added; final receipt pending).
- [x] Complete implementation/config publication and launch preflight verified against remote revision ([PR4](https://github.com/BurnyCoder/counter-strike-reinforcement-learning-selfplay/pull/4), per-worker preflight receipts).
- [x] Full Windows tests (75 passed) and one [substantive correctness/security/architecture review](review-stage-a.md); both PR4 Windows checks passed.
- [x] Action-mode diagnostic's six reproducibility controls match all 774 raw records; seven figures visually inspected ([report](../reports/duel-fixed-v1/action-modes-v1.md), [audit](../reports/duel-fixed-v1/action-mode-verification.json)). Subsequent local suite: 98 tests passed, including no-optimizer milestone/resume fixtures and diagnostic selection checks; learning acceptance remains unmet.
- [ ] Native arena watch, first-person watch, human play, takeover/hand-back and rendered/headless replay parity.
- [x] [Historical audit](../reports/learning-audit/summary.md) resolved with negative outcomes and uncertainty, original suite deferred without deadline changes.
- [x] [Bounded duel hardware calibration](amendments/2026-10-23-duel-calibration.md); selected execution and equal pilot targets published with this amendment.
- [x] Reward pilots executed in registered order, identical initial actor/critic hashes and equal step targets verified: [R0](../reports/duel-fixed-v1/pilot-R0.md), [R1](../reports/duel-fixed-v1/pilot-R1.md), [R2](../reports/duel-fixed-v1/pilot-R2.md). All failed selection; this is completeness, not acceptance.
- [x] Documented [action-mode diagnostic fallback](../reports/duel-fixed-v1/action-modes-v1.md) and completed [duration/entropy comparison](../reports/duel-fixed-v1/duration-entropy-v1/README.md); both duration/entropy recipes fail their screen. All12 endpoints and exact152-update control reproduction audited. These earlier results did not qualify confirmation; the later representation result is recorded below.
- [x] [Source review and prospective input intervention](shooter-rl-design-review.md), analytic geometry/masking/checkpoint tests; current local suite 104 passed. This is implementation evidence, not learning acceptance.
- [x] Registered real optimizer save/resume equality control: all12 exact comparisons passed over two updates ([report and original audit](../reports/duel-fixed-v1/egocentric-v1/resume.md)); long interruption cases remain outside this fixture's scope.
- [x] [Observer-coordinate evidence](../reports/duel-fixed-v1/egocentric-v1/README.md) passes the single-seed recipe screen;104 local tests and Ruff pass. This qualifies fresh confirmation, not a diagnostic deployment.
- [x] [Power sensitivity and resolved three-seed settings](amendments/2026-10-23-confirmation-resolution.md) registered before confirmation; actual publication identity is recorded in each launch preflight.
- [ ] All three fresh independent seeds completed or explicitly failed/partial within absolute deadline.
- [ ] Deployment frozen by validation before final tests; every initial/final evaluated and reported.
- [ ] Model and match uncertainty separated; deterministic repeated trajectories not pseudo-replicated.
- [ ] Numerical engineering acceptance and >=0.20 hypothesis assessed without significance-driven extensions.
- [ ] Per-run/aggregate reports, plots and sourced PDF by Libor Burian visually verified.
- [ ] Every confirmation checkpoint pair, config, metrics and model card published via HF CLI in existing collection.
- [ ] Final trained checkpoint verified in all requested modes; replay evidence retained.
- [ ] README/docs/paper/cards source and numerical consistency audited; feature PR checked, attached and merged.
- [ ] Heartbeat removed only after all authorized work is complete.

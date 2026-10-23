# Experiment evidence

Stage A fixed-spawn 1v1 training and evaluation are complete. The three fresh models improve primary test score by 0.415365 on average, but fail the moving-script engineering gate. Artifacts remain **experimental**; no later stage starts automatically. [Final confirmation report](duel-fixed-v1/confirmation/README.md), [all-experiment table and terminology](summary.md), [sourced paper](duel-fixed-v1/paper.pdf), [editable paper](duel-fixed-v1/paper.md).

The [confirmation gallery](duel-fixed-v1/confirmation/curves/README.md) and [behavior/reward gallery](duel-fixed-v1/confirmation/curves/training/README.md) retain every model's loss, reward, combat, activity and optimizer curves. [Test scores](duel-fixed-v1/confirmation/test-deterministic.png), [spawn-end outcomes](duel-fixed-v1/confirmation/test-deterministic-breakdown.png) and [uncertainty](duel-fixed-v1/confirmation/uncertainty.png) separate benchmark performance from training loss. [Plot guide](../docs/experiment-plots.md) provides raw-data hashes, units and reproduction commands.

The [publication archive](publication/README.md) verifies every new model repository's public status, collection membership and file bytes. [Native gameplay evidence](duel-fixed-v1/confirmation/native/README.md) covers arena and first-person watch, human ownership, takeover/hand-back and eight exact replay checks, with explicit physical-input test limitations.

Earlier reward pilots, sampling controls, duration/entropy interventions, the successful single-seed representation diagnostic, real resume control, all hardware probes, zero-update failure and deferred 5v5 work remain in the [canonical experiment summary](summary.md). It links each individual report without repeating their full explanations here. The [original 5v5 protocol](../docs/protocol.md) and its deadlines remain frozen; no completed or failed result was replaced.

The [source bibliography](../docs/sources.md), [evidence index](../docs/evidence-index.md), [roadmap](../docs/experiment-roadmap.md) and [acceptance ledger](../docs/acceptance-stage-a.md) distinguish registrations, implementation checks, measured findings and unexecuted future work. Match intervals condition on fixed checkpoints; independent-model intervals describe retraining variability under their stated assumptions. Fixed deterministic trajectory repetitions are not independent evidence.

```powershell
uv run python scripts/report_confirmation.py --input reports/duel-fixed-v1/confirmation/data --output runs/reproduced-confirmation
# On the original experiment workspace, also regenerate the editable paper's PDF:
uv run csrl report --input runs/duel-fixed-v1 --output reports/duel-fixed-v1
```

These reporting commands perform no optimizer updates. The paper and generated figures require visual inspection after editing; a successful renderer exit is not a scientific or visual acceptance check.

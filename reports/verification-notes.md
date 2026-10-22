# Engineering verification before the primary experiment

These checks were run on 22 October 2026 on Windows with an RTX 5070 Laptop GPU, PyTorch 2.14.1+cu130, and the committed uv environment. The [individual experiment reports](verification/summary.md) retain exact configurations, source revisions, full update metrics, and before/after evaluation rounds. The primary self-play hypothesis remains untested by these short runs.

| Check | Measured outcome | Interpretation |
| --- | --- | --- |
| CPU self-play smoke | 32,768 arena transitions, 16 updates, about 95.9 seconds | Finite updates, changed actor weights, optimizer state and checkpoint resume verified |
| CUDA self-play smoke | 8,192 transitions, 4 updates, about 12.4 seconds | Actual GPU optimization works; no rounds ended within this short training horizon |
| Initial/final frozen evaluation | Both smoke runs scored 4 wins, 20 losses, 0 draws at both endpoints over 24 rounds | No measured score improvement; four seed pairs are too few for reliable match uncertainty |
| Observation-cache benchmark | 2,048-frame collection: 4.629 seconds before, 1.568 seconds after | One timing pair, not a robust hardware comparison; action/observation/reward/termination trajectory SHA256 was identical |
| Rendered/headless replay | Exact final-state equality for trained first-person watch, play, and takeover sessions, each 40 ticks | Renderer/control integration preserves the authoritative simulation |

The conditional bootstrap interval was zero width because all four tested world clusters had the same aggregate score. It does not establish certainty, absence of seed sensitivity, or equivalence between training conditions. The short CPU run's first combat rounds persisted past the curriculum's early frame boundaries, so its optimization diagnostics cannot demonstrate exposure to every scenario stage. Longer primary runs use the same fixed schedule with resets at round boundaries.

The initial trained-view launch failed with Windows `WinError 1114` in `c10.dll` when PyTorch loaded after the native graphics engine. Loading the checkpoint before Panda3D initialization fixed the observed failure; the saved model subsequently ran through the actual launcher and both first-person modes. See the related [PyTorch native import-order issue](https://github.com/pytorch/pytorch/issues/166628). Takeover and hand-back each preserved the complete snapshot at the switching tick, including health, ammunition, position and objective state.

Repeated visibility construction made collection unnecessarily expensive. The environment now reuses independent observation copies already returned by reset/step, and reset stacks its existing frame. Public `observe()` remains fresh and pure, including for directly mutated test fixtures. Regression tests cover non-aliasing, partial resets and canonical observation equality. This change affects computation reuse, not rewards, actions, game rules, architecture, or optimization settings.

The initial source/protocol and core feature PR were public before the first optimizer smoke. Source changes for the observation cache were pushed before the CUDA smoke. A correctness/security/architecture review, regression tests, native visual inspection, checkpoint reload, and logged failure diagnosis preceded the primary suite. Smoke scores were not used to select hyperparameters or the deployed seed. All algorithm settings remain those registered in the [protocol](../docs/protocol.md).

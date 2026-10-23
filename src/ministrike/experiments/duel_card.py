"""Local: truthful duel model-card text; global: versioned releases do not inherit legacy claims.

Card format: https://huggingface.co/docs/hub/model-cards . Numerical claims are read from
the selected run's recorded artifacts. Architecture/reward decisions are registered in docs/duel-protocol.md.
"""

import hashlib  # Local: immutable checkpoint identity without loading training tensors.
import json  # Local: retain recorded summaries and provenance rather than inferred results.


def make_duel_card(run, result, config):
    """Local: compose one evidence-grounded card; global: experimental endpoints never imply general skill."""
    path = run / "final_test.json"
    evaluation = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    summary = evaluation.get("summary", {})  # Local: missing evaluation stays pending, never zero.
    hashes = {}  # Local: identify both exact published checkpoints independently of their filenames.
    for name in ("initial.pt", "model.pt"):
        if (run / name).is_file():
            with (run / name).open("rb") as stream:
                hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    revision = result.get("source_revision", config.get("source_revision", "unavailable"))
    source = "https://github.com/BurnyCoder/3d-firstperson-shooter-reinforcement-learning-selfplay"
    context_path = run / "release-context.json"
    context = json.loads(context_path.read_text(encoding="utf-8")) if context_path.exists() else {"status": "pending"}
    validation_path = run / "final_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else {}
    diagnostic = config.get("registration", "").startswith("A-DIAG")
    return f"""---
license: mit
library_name: pytorch
tags:
- reinforcement-learning
- self-play
- torchrl
- experimental
---
# MiniStrike fixed-spawn duel, seed {result['seed']}

Author: Libor Burian. **Experimental research checkpoint** for an original 1v1 elimination game.
{'**Diagnostic only: never use this model as the deployment checkpoint.**' if diagnostic else '**Engineering acceptance is not implied by completing training.**'}
Source revision: [`{revision}`]({source}/tree/{revision}). Registration: `{config.get('registration', 'unavailable')}`.

## Method and architecture

Random initialization, historical self-play only: no human demonstrations, imitation learning,
pretrained gameplay policies or scripted training opponents. The actor is a tanh MLP with two
128-unit hidden layers, **192** inputs (four visibility-limited 48-feature frames) and **five categorical**
controls: movement, yaw, pitch, firing and reload. A separate critic has two 256-unit tanh hidden
layers and receives the active learner team's observations; for 1v1 this is one player's input.
TorchRL supplies PPO loss, collection and GAE. Opponents are immutable learned snapshots,
sampled 50% latest / 50% history at round boundaries, with the initial policy retained permanently.
Actor preprocessing is **{config.get('observation_transform', 'identity')}**: egocentric-v1 rotates already-visible
relative coordinates into each observation frame's right/up/forward basis; identity retains world axes.
The critic retains original coordinates. No extra hidden information, aiming or navigation is supplied.

Reward `{config.get('reward', 'unavailable')}` and all coefficients are engineering choices in the
[active protocol]({source}/blob/{revision}/docs/duel-protocol.md), not published optima. Exact
optimizer/rollout/hardware settings are in `config.json`. Training recorded **{result['frames']} arena steps**,
**{result['updates']} updates**; full allocated step budget completed: **{result['complete']}**.

## Measurements and limitations

Final-test status: **{evaluation.get('status', 'pending')}**. Equal-opponent/side round score:
**{summary.get('score', 'pending')}**. Baseline breakdown: `{json.dumps(summary.get('baselines', {}), sort_keys=True)}`.
Elimination fraction: **{summary.get('elimination_fraction', 'pending')}**.
Final validation summary (selection evidence, not final testing): `{json.dumps(validation.get('summary', {}), sort_keys=True)}`.
Win/draw/loss score is 1/0.5/0 against random, corresponding initial and moving quantized scripted
evaluation controls. Raw paired worlds and identities are in `initial_test.json` and `final_test.json`
when available; stochastic behavior is separately labeled. Validation files concern model selection,
not held-out final testing. Missing measurements are unavailable, not evidence of success.

Release assessment: `{json.dumps(context, sort_keys=True)}`. The full confirmation design uses
**three independent** models. Many matches of one model do not establish retraining reliability;
deterministic fixed-layout trajectories are recorded once per spawn end. Neither this task nor
these observations establish random-spawn, tactical-map, bomb-game or human-level competence.
Failures and negative results remain public in the [experiment reports]({source}/tree/main/reports).

## Use and reproducibility

Install the source repository's locked `uv` environment, then select this downloaded `model.pt`:

```powershell
uv run csrl watch --experiment duel-fixed-v1 --first-person --checkpoint PATH/model.pt
uv run csrl play --experiment duel-fixed-v1 --checkpoint PATH/model.pt
```

Tab selects a living bot; V/F1 switches spectator camera; F/F2 takes over or hands control back without
resetting the round. The arena is 14 × 14 metres, with fixed starts six metres apart, 20 Hz simulation,
20-second rounds, no respawns/objectives, and simultaneous lethal shots/timeouts drawing.
`initial.pt` is the untouched model baseline; `model.pt` is the recorded endpoint. Checkpoints
contain optimizer/environment/RNG state and use the project's restricted PyTorch loader.
Checkpoint **SHA256** identities: `{json.dumps(hashes, sort_keys=True)}`.

Primary method references: [TorchRL PPO/GAE](https://docs.pytorch.org/rl/stable/tutorials/multiagent_ppo.html),
[Sample Factory Doom self-play inspiration](https://proceedings.mlr.press/v119/petrenko20a/petrenko20a.pdf),
[potential shaping](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf),
[independent-seed power](https://arxiv.org/abs/1806.08295), and
[evaluation uncertainty](https://arxiv.org/abs/2108.13264). Library versions and all design limits
are recorded in the source lockfile, protocol and [bibliography]({source}/blob/main/docs/sources.md).
"""  # Local: one template supplies Markdown and Hub text; no second factual narrative is maintained.

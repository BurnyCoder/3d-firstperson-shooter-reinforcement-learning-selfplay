"""Local: publish an explicit allowlist of measured model artifacts. Global: reproducible public checkpoints.

Source: https://huggingface.co/docs/huggingface_hub/en/guides/cli . Authentication stays in hf's existing session.
"""

import argparse  # An explicit module command keeps external publication separate from training.
import json  # Model cards derive numbers from recorded results, never invented defaults.
import logging  # Persist complete CLI responses without exposing authentication tokens.
import shutil  # Stage only allowlisted model/evidence files, excluding arbitrary workspace content.
import subprocess  # Use the authenticated Hugging Face CLI without shell interpolation.
from pathlib import Path  # All staging happens beneath this project's generated run directory.

from ministrike.records import configure_logging, write_json  # Reuse timestamped diagnostics and atomic provenance.

LOGGER = logging.getLogger(__name__)  # Same logging contract as training and evaluation.
FILES = ("model.pt", "initial.pt", "config.json", "result.json", "evaluation.json", "initial_evaluation.json", "metrics.jsonl")  # Credentials and .env are never eligible for upload.
FILES += ("initial_validation.json", "final_validation.json", "initial_test.json", "final_test.json",
          "initial_validation_stochastic.json", "final_validation_stochastic.json", "assessment.json",
          "initial_test_stochastic.json", "final_test_stochastic.json", "release-context.json")  # Local: explicit duel evidence allowlist.
FILES += ("archive-status.json",)  # Local: deferred models get honest derived metadata without rewriting original trainer outputs.


def hf_command(arguments: list[str]) -> dict | str:
    """Local: execute one authorized Hub command. Global: no tokens appear in command text or logs."""
    output = []  # Retain the full service receipt while also streaming progress immediately.
    with subprocess.Popen(["hf", *arguments, "--format", "json"], stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding="utf-8", bufsize=1) as process:  # CLI inherits existing authentication without shell interpolation.
        for line in process.stdout:  # Drain both streams continuously so slow uploads remain observable.
            output.append(line)  # Preserve complete output without truncating service errors or receipts.
            LOGGER.info("hf %s", line.rstrip())  # Each progress line receives the shared UTC timestamp.
        returncode = process.wait()  # Exit status distinguishes a complete commit from transferred but unpublished bytes.
    receipt = "".join(output)  # Combine the exact streamed output for structured parsing or diagnostics.
    if returncode:  # Failed publication must not be reported as complete.
        raise RuntimeError(f"Hugging Face command failed: {receipt}")  # Preserve actionable service diagnostics.
    try:
        return json.loads(receipt)  # JSON mode supports reproducible receipt parsing when progress is absent.
    except json.JSONDecodeError:
        return receipt.strip()  # Mixed progress/JSON output remains a complete, auditable text receipt.


def stage_run(run: Path, destination: Path, result=None) -> dict:
    """Local: prepare a source-grounded model card and strict file allowlist. Global: inspectable public release."""
    result = result or json.loads((run / "result.json").read_text(encoding="utf-8"))  # Local: original trainer output or explicitly identified archival metadata.
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))  # Preserve exact architecture/training controls.
    if not (run / "model.pt").is_file():  # A metrics-only experiment is not a downloadable model.
        raise FileNotFoundError(run / "model.pt")  # Do not create an empty misleading model repository.
    destination.mkdir(parents=True, exist_ok=True)  # Staging stays in generated local outputs.
    if result.get("archived_deferred"):
        write_json(destination / "archive-status.json", result)  # Local: derived archive status is never named result.json.
    for name in FILES:  # Fixed names prevent accidentally uploading secrets or unrelated project content.
        source = run / name  # Model/evaluation artifacts live directly in the selected run directory.
        if source.is_file():  # Missing evaluation remains explicitly pending in the card below.
            shutil.copy2(source, destination / name)  # Preserve checkpoint bytes exactly.
    if config.get("experiment") == "duel-fixed-v1":
        from ministrike.experiments.duel_card import make_duel_card  # Local: keep match-specific prose out of upload orchestration.
        (destination / "README.md").write_text(make_duel_card(run, result, config), encoding="utf-8")
        return result  # Local: duel cards cannot fall through to legacy5v5 claims.
    evaluation_path = run / "evaluation.json"  # Held-out endpoint is separate from changing training rewards.
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8")) if evaluation_path.exists() else {}  # No evaluation is not a zero score.
    summary = evaluation.get("summary", {})  # Missing summaries are labeled pending rather than imputed.
    score = summary.get("mean_score", "pending")  # Use the recorded arithmetic score without a skill label.
    interval = summary.get("score_ci95", "pending")  # This interval conditions on the checkpoint.
    card = f"""---
license: mit
library_name: pytorch
tags:
- reinforcement-learning
- multi-agent
- self-play
- torchrl
---
# MiniStrike {run.name}

Author: Libor Burian. Original 3D 5v5 tactical-game checkpoint from the [source repository](https://github.com/BurnyCoder/3d-firstperson-shooter-reinforcement-learning-selfplay).

Archival status: **{'DEFERRED, INCOMPLETE: no completed final tournament or tactical-skill claim' if result.get('archived_deferred') else 'Verification or measured experiment; inspect the recorded status below'}**.
The deferred suite retains its original deadlines and all interruption amendments in the [learning audit](https://github.com/BurnyCoder/3d-firstperson-shooter-reinforcement-learning-selfplay/blob/main/reports/learning-audit/summary.md). Checkpoint source identity denotes the original learning implementation; later execution/performance revisions are recorded in that audit. This legacy fixed-opponent condition, where present, is outside the later pure-self-play Stage A protocol.

## Method and measured status

Condition: **{result['condition']}**; independent seed **{result['seed']}**; **{result['frames']} arena transitions**; {result['updates']} updates; frame budget completed: **{result['complete']}**. Training source revision: `{result.get('source_revision', 'not recorded')}`.

The shared actor is a 2x128 tanh MLP with 384 visibility-limited history inputs and six categorical controls. A 2x256 centralized critic is used only for MAPPO training. Gameplay inference uses local observations and raw controls. `initial.pt` is the untouched baseline; `model.pt` is the final saved endpoint, not selected using final-test scores.

Recorded evaluation status: **{evaluation.get('status', 'pending')}**. Mean round score: **{score}**; conditional 95% seed-pair bootstrap interval: **{interval}**. Score assigns win 1, draw 0.5, loss 0 against frozen random/rush/guard controls with sides swapped. Raw rounds, side/opponent breakdowns, and exact seeds are in `evaluation.json` when available.

This is a small local experiment, not evidence of human-level skill or transfer to commercial Counter-Strike. Smoke runs validate the learning pipeline only. Match confidence intervals do not establish robustness across independently trained models. Failed hypotheses and weak bots remain published.

## Download and play

Install the source repository using its locked uv environment. Download this repository into `artifacts/models/{run.name}`, then run:

```powershell
uv run csrl watch --first-person --checkpoint artifacts/models/{run.name}/model.pt
uv run csrl play --checkpoint artifacts/models/{run.name}/model.pt
```

In spectator mode, Tab selects a living bot, V changes camera, and F takes over or returns control without resetting the round. Checkpoints contain weights, optimizer/environment state and RNGs; the project loader validates the schema and uses PyTorch weights-only loading.

## Reproducibility and limitations

The one original map uses ground-level movement, box geometry, a single rifle and bomb objectives. Inputs are structured observations rather than pixels. Training gamma is {config['gamma']}; curriculum fractions are 15/15/70 for combat/objective/standard starts. Configuration, source revision, metrics and initial/final evaluations are provided without omitting negative outcomes. See the source [protocol](https://github.com/BurnyCoder/counter-strike-reinforcement-learning-selfplay/blob/main/docs/protocol.md) and [reports](https://github.com/BurnyCoder/counter-strike-reinforcement-learning-selfplay/tree/main/reports) for the complete comparison and uncertainty assumptions.

Algorithm references: [TorchRL MAPPO tutorial](https://docs.pytorch.org/rl/stable/tutorials/multiagent_ppo.html), [MAPPO paper](https://arxiv.org/abs/2103.01955), [RL evaluation uncertainty](https://arxiv.org/abs/2108.13264).
"""  # One template derives every empirical claim from actual artifacts.
    (destination / "README.md").write_text(card, encoding="utf-8")  # HF renders this as the public model card.
    return result  # Publication metadata can record the precise run identity.


def repository_name(namespace, run, config):
    """Local: preserve stable experiment identities; global: repeated seeds cannot overwrite different interventions."""
    prefix = "duel-fixed-v1-" if config.get("experiment") == "duel-fixed-v1" else ""
    registration = config.get("registration", "")
    if registration.startswith("A-PILOT"):
        prefix += f"pilot-{config['reward'].lower()}-"
    elif registration.startswith("A-DIAG"):
        prefix += registration.removeprefix("A-").lower() + "-"
    return f"{namespace}/ministrike-{prefix}{run.name.replace('_', '-')}"


def deferred_result(run):
    """Local: describe the last durable checkpoint; global: a deliberate deferral must never masquerade as a completed trainer."""
    from ministrike.learning.checkpoint import load_checkpoint  # Local: use restricted weights-only deserialization.
    from ministrike.experiments.loss_plots import parse_metrics
    from ministrike.experiments.duel_evaluation import file_hash
    state = load_checkpoint(run / "model.pt")
    rows, warnings = parse_metrics((run / "metrics.jsonl").read_bytes())
    if warnings or rows[-1]["frames"] != state["frames"] or rows[-1]["updates"] != state["updates"]:
        raise ValueError("Archive metric/checkpoint boundary mismatch; preserve and investigate")
    return {"archived_deferred": True, "complete": False, "stop_reason": "deferred_by_user",
            "condition": state["config"]["condition"], "seed": state["config"]["seed"],
            "frames": state["frames"], "updates": state["updates"], "elapsed_seconds": state["elapsed_seconds"],
            "source_revision": state["source_revision"], "saved_at_utc": state["saved_at_utc"],
            "checkpoint_sha256": {name: file_hash(run / name) for name in ("initial.pt", "model.pt")},
            "metrics_sha256": file_hash(run / "metrics.jsonl")}


def publish_models(input_dir: str | Path, collection: str, namespace: str = "BurnyCoder", include_deferred=False) -> dict:
    """Local: upload direct child runs into one collection. Global: preserve all conditions, not only the best score."""
    root = Path(input_dir).resolve()  # The selected input directory bounds discovery.
    if include_deferred and json.loads((root / "suite.json").read_text(encoding="utf-8"))["status"] != "deferred_by_user":
        raise ValueError("Only the explicitly deferred suite permits archival checkpoint publication")
    runs = [path.parent for path in sorted(root.glob("*/config.json"))
            if (path.parent / "result.json").exists() or (include_deferred and (path.parent / "model.pt").exists())]
    if not runs:  # Empty publication must be reported as a problem, not success.
        raise FileNotFoundError(f"No completed run artifacts under {root}")  # User can inspect the explicit input path.
    published = []  # Preserve one verifiable receipt per model repository.
    for run in runs:  # Both conditions and all available seeds are published in stable order.
        staging = root / "publication" / run.name  # Stage inside the same project output tree.
        result = stage_run(run, staging, deferred_result(run) if not (run / "result.json").exists() else None)
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        repository = repository_name(namespace, run, config)  # Local: all diagnostic and pilot identities are collision-free.
        include = [argument for name in (*FILES, "README.md") for argument in ("--include", name)]  # Reapply the allowlist at the actual upload boundary, even if staging already contains unrelated files.
        receipt = hf_command(["upload", repository, str(staging), ".", "--no-private", *include, "--commit-message", f"Publish {result['frames']} transition endpoint and measured evidence"])  # Upload command creates a public repository when needed.
        hf_command(["collections", "add-item", collection, repository, "model", "--exists-ok"])  # Group all released checkpoints explicitly.
        published.append({"repository": repository, "url": f"https://huggingface.co/{repository}", "frames": result["frames"], "receipt": receipt})  # Keep service receipts with provenance.
        write_json(root / "publication.json", {"collection": collection, "models": published})  # Interrupted uploads retain completed receipts.
    return {"collection": collection, "models": published}  # Final reporting can link to actual published artifacts.


def main():
    """Local: explicit publication command. Global: ordinary training never silently uploads user files."""
    parser = argparse.ArgumentParser(description="Publish MiniStrike model artifacts to an existing Hugging Face collection")  # Publication has an explicit boundary.
    parser.add_argument("--input", type=Path, required=True)  # Select the intended experiment directory.
    parser.add_argument("--collection", required=True)  # Require a verified existing collection slug.
    parser.add_argument("--namespace", default="BurnyCoder")  # The authorized account is the default destination.
    parser.add_argument("--include-deferred", action="store_true")  # Local: archival publication requires an explicit invocation.
    arguments = parser.parse_args()  # Reject missing publication inputs before contacting the Hub.
    configure_logging(Path("runs/logs"))  # Persist upload receipts in the same diagnostics tree.
    publish_models(arguments.input, arguments.collection, arguments.namespace, arguments.include_deferred)


if __name__ == "__main__":  # Imports remain offline and side-effect free.
    main()  # Only explicit module invocation performs external writes.

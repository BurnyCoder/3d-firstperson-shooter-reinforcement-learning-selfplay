"""Local: audit completed observer-coordinate evidence; global: publish the selection without new games or optimization.

Analysis is the registered egocentric_diagnostic estimator. Exact-byte copying uses
https://docs.python.org/3.12/library/shutil.html#shutil.copyfile ; existing Matplotlib
reporting phases supply all charts. This file adds no learning method or reward.
"""

import logging  # Local: record reporting phases in the caller's timestamped log.
from pathlib import Path  # Local: explicit project evidence roots only.
import shutil  # Local: preserve raw bytes and hashes, rather than rewriting measurements.

from ministrike.experiments.duel_evaluation import file_hash, summarize
from ministrike.experiments.duration_reporting import plot_duration_evaluations  # Local: reuse endpoint and contrast charts.
from ministrike.experiments.egocentric_diagnostic import compare_egocentric  # Local: one registered estimator, no post-hoc score selection.
from ministrike.experiments.loss_plots import save_loss_gallery  # Local: shared loss/reward/behavior/optimizer accounting.
from ministrike.experiments.registration import read_json
from ministrike.learning.checkpoint import parameter_hash
from ministrike.records import source_revision, timestamp, write_json


def copy_records(root, destination):
    """Local: copy allowlisted raw evidence; global: model binaries and credentials are never included in the Git report bundle."""
    hashes = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() and path.suffix in {".json", ".jsonl", ".log"}:
            target = Path(destination) / path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)  # Local: full logs and traces survive unchanged.
            hashes[target.as_posix()] = file_hash(target)
    return hashes


def audit_egocentric(root, control):
    """Local: reproduce all six endpoints, identities and the comparison; global: success cannot bypass raw-data correctness."""
    config = read_json("configs/duel-egocentric-R1.json")
    run = Path(root) / "seed5201"
    receipt = read_json(Path(root) / "evaluation-receipt.json")
    expected = {f"{endpoint}/{mode}" for endpoint in ("initial", "early", "final") for mode in ("deterministic", "stochastic")}
    result = read_json(run / "result.json")
    if receipt["status"] != "complete" or set(receipt["completed"]) != expected or not result["complete"] or result["frames"] != 4_980_736:
        raise ValueError("Incomplete registered diagnostic")
    if file_hash(Path(control) / "seed5201/final_validation.json") != config["control_validation_sha256"]:
        raise ValueError("Frozen control evidence changed")
    fingerprints = {name: parameter_hash(run / "initial.pt", name) for name in ("actor", "critic")}
    if any(value != config[f"initial_{name}_sha256"] for name, value in fingerprints.items()):
        raise ValueError("Matched initial tensors changed")
    evaluations, hashes = {}, {}
    for label, folder in (("world control", Path(control) / "seed5201"), ("observer coordinates", run)):
        for endpoint, filename in (("initial", "initial.pt"), ("early", "milestones/update-000152.pt"), ("final", "model.pt")):
            digest = file_hash(folder / filename)
            hashes[f"{label}/{endpoint}"] = digest
            for mode, suffix in (("deterministic", ""), ("stochastic", "_stochastic")):
                row = read_json(folder / f"{endpoint}_validation{suffix}.json")
                if row["status"] != "complete" or summarize(row["rounds"]) != row["summary"] or row["identity"]["checkpoint_sha256"] != digest:
                    raise ValueError("Endpoint summary or evaluated checkpoint does not reproduce")
                evaluations.setdefault(mode, {}).setdefault(label, {})[endpoint] = row
    primary = evaluations["deterministic"]
    analysis = compare_egocentric(primary["world control"]["final"], primary["observer coordinates"]["initial"], primary["observer coordinates"]["final"])
    analysis["external_baselines"] = list(analysis["external_baselines"])  # Local: JSON stores the registered tuple as an array.
    analysis["secondary_summaries"] = {name: row["summary"] for name, row in evaluations["stochastic"]["observer coordinates"].items()}
    if analysis != read_json(Path(root) / "analysis.json"):
        raise ValueError("Registered analysis changed")
    return evaluations, analysis, {"initial_fingerprints": fingerprints, "checkpoint_hashes": hashes, "analysis_reproduced": True}


def publish_egocentric(root, control, destination):
    """Local: audit, copy, plot and hash; global: one readable report wrapper retains every endpoint, not only the successful final."""
    root, control, destination = Path(root), Path(control), Path(destination)
    evaluations, analysis, audit = audit_egocentric(root, control)
    destination.mkdir(parents=True, exist_ok=True)
    inputs = copy_records(root, destination / "data/egocentric")
    inputs.update(copy_records(control, destination / "data/control"))
    resume = root.parent / "resume-v1"
    inputs.update(copy_records(resume, destination / "data/resume"))  # Local: publish the completed real resume prerequisite without repeating it.
    for path in (root.parent / "egocentric-R1.json", root.parent.parent / "launch.json",
                 root.parent.parent / "egocentric.stdout.log", root.parent.parent / "egocentric.stderr.log",
                 Path("configs/duel-egocentric-R1.json"), Path("configs/duel-resume-control.json")):
        target = destination / "data/execution" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        inputs[target.as_posix()] = file_hash(target)
    save_loss_gallery(root, destination / "curves")  # Local: all intervals and all existing telemetry categories, no smoothing.
    contrast = {"external": {"delta": analysis["external_delta"], "ci95": analysis["external_match_uncertainty"]["ci95"]}}
    plot_duration_evaluations(evaluations, contrast, destination, title="Observer-coordinate intervention",
                              contrasts={"external": "Observer − world control\nRandom / scripted opponents only"})
    audit.update(created_utc=timestamp(), report_source_revision=source_revision(),
                 raw_inputs={Path(path).relative_to(destination).as_posix(): value for path, value in inputs.items()},
                 figures={path.relative_to(destination).as_posix(): file_hash(path) for path in sorted(destination.rglob("*")) if path.suffix in {".png", ".svg"}},
                 plotted_summaries={mode: {label: {key: row["summary"] for key, row in endpoints.items()} for label, endpoints in runs.items()} for mode, runs in evaluations.items()},
                 independent_training_seeds=1, final_test_accessed=False, deployment_eligible=False)
    write_json(destination / "audit.json", audit)
    logging.info("Observer-coordinate evidence: %s raw records; %s figure pairs", len(inputs), len(audit["figures"]) // 2)


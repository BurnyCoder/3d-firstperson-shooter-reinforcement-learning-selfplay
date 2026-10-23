"""Local: frozen duration/entropy endpoints and paired contrasts; global: preserve the diagnostic/deployment boundary.

Protocol: docs/amendments/2026-10-23-duration-entropy-diagnostic.md.
Library-backed uncertainty is reused from duel_analysis; no optimizer or new reward implementation lives here.
"""

from datetime import datetime, timedelta, timezone  # Local: one durable evaluation cutoff, including recovery downtime.
import logging  # Local: full endpoint identities and errors reach both file and terminal.
from pathlib import Path  # Local: registered paths only.

from ministrike.experiments.duel_analysis import paired_match_interval  # Local: keep checkpoint, baseline, spawn and seed pairing.
from ministrike.experiments.duel_evaluation import evaluate_duel, file_hash  # Local: reuse the authoritative tournament.
from ministrike.experiments.registration import MANIFEST, published_preflight, read_json, remaining_budget
from ministrike.learning.checkpoint import parameter_hash  # Local: safe exact initialization/prefix comparison.
from ministrike.records import write_json  # Local: preserve completed endpoints before moving to the next one.


def compare_duration_entropy(evaluations):
    """Local: evaluate both registered hypotheses and conservative screens; global: no diagnostic checkpoint can deploy."""
    control, lower = evaluations["control"], evaluations["low-entropy"]
    if any(endpoint["status"] != "complete" for run in evaluations.values() for endpoint in run.values()):
        raise ValueError("An incomplete validation cannot qualify a recipe")
    results = {"deployment_eligible": False, "recipes": {}, "eligible_recipe": None,
               "scope": "one matched training seed; paired match uncertainty is conditional on fixed weights"}
    for name, before, after in (("duration", control["early"], control["final"]),
                                ("entropy", control["final"], lower["final"])):
        delta = after["summary"]["score"] - before["summary"]["score"]
        results[name] = {"delta": delta, "threshold_met": delta >= .10, **paired_match_interval(before, after)}
    for name, run in evaluations.items():
        summary = run["final"]["summary"]
        delta = summary["score"] - run["initial"]["summary"]["score"]
        scores = summary["baselines"]
        passed = (delta >= .10 and scores["random"]["score"] >= .70 and scores["initial"]["score"] >= .70
                  and scores["scripted"]["score"] >= .55 and summary["elimination_fraction"] >= .90)
        results["recipes"][name] = {"improvement": delta, "final": summary, "screen_passed": passed,
                                    "match_uncertainty": paired_match_interval(run["initial"], run["final"])}
    eligible = [name for name, row in results["recipes"].items() if row["screen_passed"]]
    if eligible:
        selected = max(eligible, key=lambda name: results["recipes"][name]["final"]["score"])
        if "control" in eligible and results["recipes"][selected]["final"]["score"] - control["final"]["summary"]["score"] <= .02:
            selected = "control"  # Local: preregistered near-tie favors the unchanged entropy coefficient.
        results["eligible_recipe"] = selected  # Local: fresh confirmation still needs a published resolution and budget.
    return results


def evaluate_duration_entropy(config_path):
    """Local: verify checkpoints, evaluate all endpoints, compare; global: no final-test access or favorable endpoint selection."""
    config = read_json(config_path)
    preflight = published_preflight(config_path, "A-DIAG-DURATION-R1")
    root = Path(config["runs"][0]["output_dir"]).parents[1]
    receipt_path = root / "evaluation-receipt.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt["preflight"]["sha256"] != preflight["sha256"] or receipt["preflight"]["source_revision"] != preflight["source_revision"]:
            raise ValueError("Evaluation recovery source/configuration changed")
        if receipt["status"] == "complete":
            return read_json(root / "analysis.json")
    else:
        now = datetime.now(timezone.utc)
        receipt = {"status": "running", "started_utc": now.isoformat(), "preflight": preflight,
                   "deadline_utc": (now + timedelta(seconds=min(config["validation"]["max_seconds"],
                                        remaining_budget(phase="A-DIAG")))).isoformat(), "checkpoints": {}, "completed": []}
        write_json(receipt_path, receipt)  # Local: persist the cutoff before any endpoint inference.
    deadline = datetime.fromisoformat(receipt["deadline_utc"])
    state = read_json(MANIFEST)
    state.update(status="duration_entropy_validation", active_group=str(receipt_path))
    write_json(MANIFEST, state)
    results = {"deterministic": {}, "stochastic": {}}
    try:
        for label, row in zip(("control", "low-entropy"), config["runs"], strict=True):
            run = Path(row["output_dir"])
            training = read_json(run / "result.json")
            if not training["complete"] or training["frames"] != row["total_frames"]:
                raise ValueError("Validation requires the complete equal registered training budgets")
            for component in ("actor", "critic"):
                if parameter_hash(run / "initial.pt", component) != config[f"initial_{component}_sha256"]:
                    raise ValueError(f"{label} initialization differs: {component}")
                if label == "control" and parameter_hash(run / "milestones/update-000152.pt", component) != parameter_hash(
                        "runs/duel-fixed-v1/pilots/R1/seed5201/model.pt", component):
                    raise ValueError(f"Control early endpoint differs from original R1: {component}; investigate before attribution")
            for endpoint, filename in (("initial", "initial.pt"), ("early", "milestones/update-000152.pt"), ("final", "model.pt")):
                model = run / filename
                key = f"{label}/{endpoint}"
                digest = file_hash(model)
                if key in receipt["checkpoints"] and receipt["checkpoints"][key] != digest:
                    raise ValueError("A validation checkpoint changed during recovery")
                receipt["checkpoints"][key] = digest
                write_json(receipt_path, receipt)
                for mode, stochastic in (("deterministic", False), ("stochastic", True)):
                    logging.info("duration_entropy_endpoint %s mode=%s sha256=%s", key, mode, digest)
                    result = evaluate_duel(model, run / "initial.pt", run / f"{endpoint}_validation{('_stochastic' if stochastic else '')}.json",
                                           seed_base=config["validation"]["seed_base"], pairs=config["validation"]["pairs"],
                                           num_envs=config["validation"]["num_envs"], stochastic=stochastic, deadline=deadline)
                    if result["status"] != "complete":
                        receipt["status"] = "budget_exhausted"
                        write_json(receipt_path, receipt)
                        return receipt  # Local: incomplete endpoints remain incomplete and cannot select a recipe.
                    results[mode].setdefault(label, {})[endpoint] = result
                    if f"{key}/{mode}" not in receipt["completed"]:
                        receipt["completed"].append(f"{key}/{mode}")
                    write_json(receipt_path, receipt)
        analysis = compare_duration_entropy(results["deterministic"])
        analysis["secondary_summaries"] = {name: {endpoint: value["summary"] for endpoint, value in run.items()}
                                           for name, run in results["stochastic"].items()}
        write_json(root / "analysis.json", analysis)
        receipt.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(receipt_path, receipt)
        logging.info("duration_entropy_analysis %s", analysis)
        return analysis
    except Exception:
        logging.exception("Duration/entropy validation failed; checkpoints and completed endpoints are preserved")
        receipt["status"] = "failed"
        write_json(receipt_path, receipt)
        raise

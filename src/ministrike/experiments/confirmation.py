"""Local: evaluate every confirmation model; global: validation selection precedes untouched final tests.

The already registered procedure is docs/amendments/2026-10-23-confirmation-resolution.md.
Reuse duel_evaluation and duel_analysis, including their paired bootstrap and SciPy t interval:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html . No optimizer runs here.
"""

from datetime import datetime, timezone  # Local: persisted absolute time survives interruption.
import logging  # Local: full endpoint summaries and failures enter timestamped logs.
from pathlib import Path  # Local: all artifacts remain inside the registered project.

from ministrike.experiments.duel_analysis import model_comparison
from ministrike.experiments.duel_evaluation import evaluate_duel, file_hash
from ministrike.experiments.registration import MANIFEST, published_preflight, read_json, remaining_budget
from ministrike.records import write_json


def freeze_selection(path, candidates, validation):
    """Local: choose a completed final model by primary validation; global: never use final-test scores."""
    if sorted(row["seed"] for row in candidates) != [5101, 5102, 5103]:
        raise ValueError("Incomplete confirmation candidates")  # Local: do not silently drop a weak seed.
    ranked = []
    for candidate in candidates:
        result = validation.get(candidate["seed"], {})
        if result.get("status") != "complete":
            raise ValueError("Incomplete primary validation")
        if result["identity"]["checkpoint_sha256"] != candidate["checkpoint_sha256"]:
            raise ValueError("Validation checkpoint identity changed")
        ranked.append({**candidate, "score": result["summary"]["score"]})
    decision = {"candidates": ranked, "selected": max(ranked, key=lambda row: (row["score"], -row["seed"])),
                "rule": "highest primary validation mean; exact ties choose lower seed", "final_test_accessed": False}
    path = Path(path)
    if path.exists():
        saved = read_json(path)
        if any(saved[key] != value for key, value in decision.items()):
            raise ValueError("Frozen selection cannot change")
        return saved  # Local: recovery preserves both the choice and its original timestamp.
    decision["frozen_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(path, decision)  # Local: publish selection atomically before the first final-test call.
    return decision


def confirmation_candidates(config):
    """Local: bind completed equal-budget runs to immutable artifacts; global: no diagnostic or early model may deploy."""
    candidates = []
    for row in config["runs"]:
        run = Path(row["output_dir"])
        result, preflight = read_json(run / "result.json"), read_json(run / "preflight.json")
        if (not result["complete"] or result["frames"] != row["total_frames"]
                or result["seed"] != row["seed"] or result["execution_revision"] != preflight["source_revision"]):
            raise ValueError("Incomplete or mismatched confirmation training; preserve before recovery")
        candidates.append({"seed": row["seed"], "checkpoint": (run / "model.pt").as_posix(),
                           "checkpoint_sha256": file_hash(run / "model.pt"),
                           "initial_sha256": file_hash(run / "initial.pt"),
                           "training_revision": result["execution_revision"], "arena_steps": result["frames"]})
    return candidates


def evaluate_split(candidates, split, seed_base, receipt, receipt_path):
    """Local: run all initial/final endpoints in both modes; global: no model is excluded by validation rank."""
    endpoints = {"deterministic": {}, "stochastic": {}}
    for candidate in candidates:
        run, seed = Path(candidate["checkpoint"]).parent, candidate["seed"]
        for endpoint, filename in (("initial", "initial.pt"), ("final", "model.pt")):
            for mode, stochastic in (("deterministic", False), ("stochastic", True)):
                suffix = "_stochastic" if stochastic else ""
                result = evaluate_duel(run / filename, run / "initial.pt", run / f"{endpoint}_{split}{suffix}.json",
                                       seed_base=seed_base, pairs=128, num_envs=32, stochastic=stochastic,
                                       trace_first_seed=True, deadline=datetime.fromisoformat(receipt["deadline_utc"]))
                if result["status"] != "complete":
                    raise TimeoutError("Original evaluation deadline exhausted; partial evidence retained")
                endpoints[mode].setdefault(seed, {})[endpoint] = result
                key = f"{split}/{seed}/{endpoint}/{mode}"
                receipt["completed"] = list(dict.fromkeys([*receipt["completed"], key]))
                write_json(receipt_path, receipt)  # Local: completed endpoints will not be rerun on recovery.
                logging.info("confirmation_endpoint %s %s", key, result["summary"])
    comparisons = {mode: model_comparison([{"seed": seed, "complete": True, **pair}
                                          for seed, pair in models.items()]) for mode, models in endpoints.items()}
    return endpoints, comparisons


def evaluate_confirmation(config_path="configs/duel-confirmation.json"):
    """Local: verify, validate, freeze, test and analyze; global: one immutable deadline and no training changes."""
    config = read_json(config_path)
    preflight = published_preflight(config_path, "A-EVAL")
    remaining_budget()  # Local: reject a changed twelve-hour clock using the existing guard.
    candidates = confirmation_candidates(config)
    root = Path(config["runs"][0]["output_dir"]).parent
    receipt_path, selection_path = root / "evaluation-receipt.json", root / "selection.json"
    manifest = read_json(MANIFEST)
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if (receipt["preflight"]["source_revision"] != preflight["source_revision"]
                or receipt["preflight"]["sha256"] != preflight["sha256"] or receipt["checkpoints"] != candidates
                or receipt["deadline_utc"] != manifest["deadline_utc"]):
            raise ValueError("Evaluation recovery identity or deadline changed")
        if receipt["status"] == "complete":
            return read_json(root / "analysis.json")
    else:
        receipt = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                   "deadline_utc": manifest["deadline_utc"], "preflight": preflight,
                   "checkpoints": candidates, "completed": []}
        write_json(receipt_path, receipt)
    if not selection_path.exists() and (manifest.get("final_test_accessed") or list(root.glob("seed*/*_test*.json"))):
        raise ValueError("Final-test access preceded selection; investigate without overwriting evidence")
    try:
        manifest.update(status="confirmation_validation", active_group=str(receipt_path))
        write_json(MANIFEST, manifest)
        validation, comparisons = evaluate_split(candidates, "validation", 3_000_000, receipt, receipt_path)
        selection = freeze_selection(selection_path, candidates, {seed: pair["final"] for seed, pair in validation["deterministic"].items()})
        analysis = {"validation": comparisons, "selection_sha256": file_hash(selection_path)}
        write_json(root / "analysis.json", analysis)
        logging.info("confirmation_selection %s", selection)
        manifest.update(status="confirmation_final_test", final_test_accessed=True,
                        final_test_started_utc=manifest.get("final_test_started_utc", datetime.now(timezone.utc).isoformat()))
        write_json(MANIFEST, manifest)  # Local: record access after the immutable selection, before evaluating test worlds.
        _, analysis["test"] = evaluate_split(candidates, "test", 4_000_000, receipt, receipt_path)
        write_json(root / "analysis.json", analysis)
        receipt.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(receipt_path, receipt)
        manifest.update(status="confirmation_evaluated", last_updated_utc=receipt["finished_utc"])
        write_json(MANIFEST, manifest)
        logging.info("confirmation_analysis %s", analysis)
        return analysis
    except Exception:
        receipt["status"] = "failed"
        write_json(receipt_path, receipt)
        logging.exception("Confirmation evaluation stopped; retain all completed and partial evidence")
        raise

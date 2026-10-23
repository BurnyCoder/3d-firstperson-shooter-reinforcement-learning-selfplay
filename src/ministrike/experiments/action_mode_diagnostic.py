"""Local: orchestrate a registered frozen-policy intervention; global: isolate action selection without another training run.

Protocol and sources: docs/amendments/2026-10-23-action-mode-diagnostic.md.
This module contains no optimizer, scripted training or checkpoint mutation.
"""

from datetime import datetime, timedelta, timezone  # Local: one persisted deadline survives resume and downtime.
import logging  # Local: record cell identities and failures immediately in full.
from pathlib import Path  # Local: all evidence stays in the registered project directory.

from ministrike.experiments.duel_evaluation import evaluate_duel, file_hash  # Local: reuse the authoritative tournament, not a second game loop.
from ministrike.experiments.mode_analysis import compare_modes  # Local: separate execution from statistical interpretation.
from ministrike.experiments.registration import MANIFEST, published_preflight, read_json, remaining_budget  # Local: shared source/deadline gates.
from ministrike.records import write_json  # Local: preserve atomic receipts and every completed cell.


def run_action_modes(config_path):
    """Local: preflight, evaluate all four cells, analyze; global: completed negatives are never replaced by a new identity."""
    config = read_json(config_path)
    preflight = published_preflight(config_path, config["registration"])  # Local: freeze implementation and exact choices before any gameplay diagnostic.
    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / "receipt.json"
    budget = remaining_budget(phase=config["registration"])  # Local: retain the final two-hour reporting/evaluation reservation.
    if budget <= 0:
        raise TimeoutError("The existing Stage A diagnostic window has ended")
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt["preflight"]["sha256"] != preflight["sha256"] or receipt["preflight"]["source_revision"] != preflight["source_revision"]:
            raise ValueError("Diagnostic resume configuration or implementation changed")
    else:
        now = datetime.now(timezone.utc)
        receipt = {"registration": config["registration"], "status": "running", "started_utc": now.isoformat(),
                   "deadline_utc": (now + timedelta(seconds=min(config["max_seconds"], budget))).isoformat(),
                   "preflight": preflight, "completed_cells": []}
        write_json(receipt_path, receipt)  # Local: save the cutoff before the first model inference; resume never refunds time.
        write_json(root / "config.json", config)
    if receipt["status"] == "complete":
        return read_json(root / "analysis.json")  # Local: repeated invocation reuses existing results without new matches.
    deadline = datetime.fromisoformat(receipt["deadline_utc"])
    manifest = read_json(MANIFEST)
    manifest["status"], manifest["active_group"] = "action_mode_diagnostic", str(receipt_path)
    amendment = config["protocol"]
    if amendment not in manifest["amendments"]:
        manifest["amendments"].append(amendment)  # Local: preserve the full history instead of replacing prior registrations.
    write_json(MANIFEST, manifest)
    results = {}
    try:
        for reward, model in config["models"].items():
            for key in ("checkpoint", "initial"):
                if file_hash(model[key]) != model[f"{key}_sha256"]:
                    raise ValueError(f"Frozen {reward} {key} hash changed")
            results[reward] = {}
            for mode in ("DD", "DS", "SD", "SS"):
                cell = f"{reward}-{mode}"
                logging.info("action_mode_cell %s learner=%s opponent=%s", cell, mode[0], mode[1])
                result = evaluate_duel(model["checkpoint"], model["initial"], root / f"{cell}.json",
                                       seed_base=config["seed_base"], pairs=config["pairs"], num_envs=config["num_envs"],
                                       stochastic=mode[0] == "S", initial_stochastic=mode[1] == "S", baselines=("initial",),
                                       trace_first_seed=True, deadline=deadline)
                if result["status"] != "complete":
                    receipt["status"] = "budget_exhausted"
                    write_json(receipt_path, receipt)  # Local: incomplete cells remain failures, not inferred losses or permission to extend.
                    return receipt
                results[reward][mode] = result
                if cell not in receipt["completed_cells"]:
                    receipt["completed_cells"].append(cell)
                write_json(receipt_path, receipt)  # Local: preserve every finished cell before continuing.
        analysis = compare_modes(results)
        write_json(root / "analysis.json", analysis)
        receipt.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(receipt_path, receipt)
        logging.info("action_mode_diagnostic_complete %s", analysis)
        return analysis
    except Exception:
        logging.exception("Action-mode diagnostic failed; completed cells and deadline are preserved")
        receipt["status"] = "failed"
        write_json(receipt_path, receipt)
        raise  # Local: never suppress an invalid inference or identity mismatch.

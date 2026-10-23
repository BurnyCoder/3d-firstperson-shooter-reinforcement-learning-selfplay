"""Local: registered coordinate-representation evaluation; global: one altered mechanism with immutable external controls.

Protocol: docs/amendments/2026-10-23-egocentric-diagnostic.md. All inference,
paired resampling and recipe thresholds reuse existing implementations.
"""

from datetime import datetime, timedelta, timezone  # Local: a persisted evaluation deadline cannot reset on resume.
import logging  # Local: full results and failures remain in timestamped logs.
from pathlib import Path  # Local: registered run paths only.

from ministrike.experiments.duel_analysis import paired_match_interval  # Local: preserve seed and side pairing.
from ministrike.experiments.duel_evaluation import evaluate_duel, file_hash  # Local: unchanged simulator and legal controls.
from ministrike.experiments.registration import MANIFEST, published_preflight, read_json, remaining_budget
from ministrike.learning.checkpoint import parameter_hash  # Local: prove exact scratch parameter initialization.
from ministrike.records import write_json  # Local: atomic durable receipt and analysis.


def compare_egocentric(control, initial, final):
    """Local: compare unchanged random/script controls; global: corresponding initial policies differ under the representation treatment."""
    external = ("random", "scripted")  # Local: frozen before execution; do not confound a changed initial opponent with skill.
    delta = sum(final["summary"]["baselines"][name]["score"] - control["summary"]["baselines"][name]["score"]
                for name in external) / len(external)
    summary = final["summary"]
    improvement = summary["score"] - initial["summary"]["score"]
    scores = summary["baselines"]
    passed = (improvement >= .10 and scores["random"]["score"] >= .70 and scores["initial"]["score"] >= .70
              and scores["scripted"]["score"] >= .55 and summary["elimination_fraction"] >= .90)
    return {"external_baselines": external, "external_delta": delta, "hypothesis_threshold_met": delta >= .10,
            "external_match_uncertainty": paired_match_interval(control, final, baselines=external),
            "own_initial_improvement": improvement, "own_match_uncertainty": paired_match_interval(initial, final),
            "screen_passed": passed, "eligible_recipe": "egocentric-v1" if passed else None, "final": summary,
            "deployment_eligible": False, "scope": "one matched training seed; no independently trained-model interval"}


def evaluate_egocentric(config_path):
    """Local: verify, evaluate all six endpoints, compare; global: no final-test or automatic confirmation."""
    config = read_json(config_path)
    preflight = published_preflight(config_path, "A-DIAG-EGO-R1")
    run = Path(config["runs"][0]["output_dir"])
    receipt_path = run.parent / "evaluation-receipt.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt["preflight"]["source_revision"] != preflight["source_revision"] or receipt["preflight"]["sha256"] != preflight["sha256"]:
            raise ValueError("Evaluation recovery source/configuration changed")
        if receipt["status"] == "complete":
            return read_json(run.parent / "analysis.json")
    else:
        now = datetime.now(timezone.utc)
        receipt = {"status": "running", "started_utc": now.isoformat(), "preflight": preflight, "completed": [],
                   "deadline_utc": (now + timedelta(seconds=min(900, remaining_budget(phase="A-DIAG")))).isoformat()}
        write_json(receipt_path, receipt)
    manifest = read_json(MANIFEST)
    manifest.update(status="egocentric_validation", active_group=str(receipt_path))
    write_json(MANIFEST, manifest)
    try:
        training = read_json(run / "result.json")
        if not training["complete"] or training["frames"] != config["runs"][0]["total_frames"]:
            raise ValueError("Incomplete representation diagnostic cannot qualify")
        if file_hash(config["control_validation"]) != config["control_validation_sha256"]:
            raise ValueError("Frozen external-control evidence changed")
        for component in ("actor", "critic"):
            if parameter_hash(run / "initial.pt", component) != config[f"initial_{component}_sha256"]:
                raise ValueError("Scratch parameter initialization differs from the matched control")
        endpoints = {}
        for endpoint, filename in (("initial", "initial.pt"), ("early", "milestones/update-000152.pt"), ("final", "model.pt")):
            for mode, stochastic in (("deterministic", False), ("stochastic", True)):
                result = evaluate_duel(run / filename, run / "initial.pt", run / f"{endpoint}_validation{('_stochastic' if stochastic else '')}.json",
                                       pairs=128, num_envs=32, stochastic=stochastic, trace_first_seed=True,
                                       deadline=datetime.fromisoformat(receipt["deadline_utc"]))
                if result["status"] != "complete":
                    raise TimeoutError("Registered evaluation allowance exhausted; preserve partial records")
                endpoints.setdefault(mode, {})[endpoint] = result
                receipt["completed"] = list(dict.fromkeys([*receipt["completed"], f"{endpoint}/{mode}"]))
                write_json(receipt_path, receipt)
        analysis = compare_egocentric(read_json(config["control_validation"]), endpoints["deterministic"]["initial"], endpoints["deterministic"]["final"])
        analysis["secondary_summaries"] = {name: row["summary"] for name, row in endpoints["stochastic"].items()}
        write_json(run.parent / "analysis.json", analysis)
        receipt.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(receipt_path, receipt)
        logging.info("egocentric_analysis %s", analysis)
        return analysis
    except Exception:
        receipt["status"] = "failed"
        write_json(receipt_path, receipt)
        logging.exception("Representation diagnostic validation failed; all prior evidence remains preserved")
        raise

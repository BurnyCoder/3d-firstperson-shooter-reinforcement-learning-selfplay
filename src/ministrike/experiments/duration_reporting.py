"""Local: audit and plot completed duration/entropy records; global: no new matches or optimization.

Methods reuse duration_entropy.py and the registered paired analysis. Plotting uses
https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.errorbar.html .
The numerical contrasts are observations, not independent-training-seed confidence intervals.
"""

import logging  # Local: preserve each reporting phase in full timestamped logs.
from pathlib import Path  # Local: explicit roots keep discovery inside this experiment.

import numpy as np  # Local: place endpoints at their actual registered experience counts.

from ministrike.experiments.duration_entropy import compare_duration_entropy  # Local: one registered estimator.
from ministrike.experiments.duel_evaluation import file_hash, summarize  # Local: verify summaries against every raw match.
from ministrike.experiments.evaluation_plots import plot_evaluation_breakdown  # Local: share side/count visualizations.
from ministrike.experiments.loss_plots import parse_metrics, save_loss_gallery  # Local: reuse validated raw curves.
from ministrike.experiments.registration import read_json  # Local: one explicit UTF-8 parser.
from ministrike.learning.checkpoint import parameter_hash  # Local: compare tensors independently of metadata.
from ministrike.records import source_revision, timestamp, write_json  # Local: portable evidence identities.


def audit_duration_evidence(root, destination):
    """Local: verify and copy exact records; global: completed negative endpoints cannot disappear from publication."""
    root, destination = Path(root), Path(destination)
    receipt = read_json(root / "evaluation-receipt.json")
    if receipt["status"] != "complete" or len(receipt["completed"]) != 12:
        raise ValueError("All twelve registered endpoints must finish before comparison publication")
    evaluations, inputs, fingerprints = {}, {}, {}  # Local: separate numeric records from exact-byte provenance.
    for label in ("control", "low-entropy"):
        run = root / label / "seed5201"
        result = read_json(run / "result.json")
        if not result["complete"] or result["frames"] != 4_980_736:
            raise ValueError("Incomplete or unequal diagnostic exposure")
        fingerprints[label] = {component: parameter_hash(run / "initial.pt", component) for component in ("actor", "critic")}
        for endpoint, filename in (("initial", "initial.pt"), ("early", "milestones/update-000152.pt"), ("final", "model.pt")):
            if file_hash(run / filename) != receipt["checkpoints"][f"{label}/{endpoint}"]:
                raise ValueError("An evaluated checkpoint changed after measurement")
            for mode, suffix in (("deterministic", ""), ("stochastic", "_stochastic")):
                record = read_json(run / f"{endpoint}_validation{suffix}.json")
                if record["status"] != "complete" or summarize(record["rounds"]) != record["summary"]:
                    raise ValueError("A saved summary differs from its raw match rows")
                evaluations.setdefault(mode, {}).setdefault(label, {})[endpoint] = record
    if fingerprints["control"] != fingerprints["low-entropy"]:
        raise ValueError("The matched initial tensors differ")
    analysis = compare_duration_entropy(evaluations["deterministic"])
    analysis["secondary_summaries"] = {label: {name: record["summary"] for name, record in endpoints.items()}
                                       for label, endpoints in evaluations["stochastic"].items()}
    if analysis != read_json(root / "analysis.json"):
        raise ValueError("Registered analysis does not reproduce")
    prefix = {"status": "not_available_in_this_copy"}  # Local: a public clone can audit matches without private run paths.
    original = Path("runs/duel-fixed-v1/pilots/R1/seed5201")
    if (original / "metrics.jsonl").exists():
        before, _ = parse_metrics((original / "metrics.jsonl").read_bytes())
        after, _ = parse_metrics((root / "control/seed5201/metrics.jsonl").read_bytes())
        excluded = {"timestamp_utc", "elapsed_seconds", "frames_per_second"}  # Local: hardware timing is not a deterministic learning variable.
        differences = [a["updates"] for a, b in zip(before, after[:len(before)], strict=True)
                       if {k: v for k, v in a.items() if k not in excluded} != {k: v for k, v in b.items() if k not in excluded}]
        tensor_match = {component: parameter_hash(original / "model.pt", component) == parameter_hash(
                       root / "control/seed5201/milestones/update-000152.pt", component) for component in ("actor", "critic")}
        prefix = {"updates": len(before), "differing_updates": differences, "tensor_match": tensor_match}
        if differences or not all(tensor_match.values()):
            raise ValueError("Unchanged control prefix did not reproduce original R1")
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in {".json", ".jsonl", ".log"} or destination in path.parents:
            continue  # Local: allowlisted experiment records only; checkpoints publish separately on the Hub.
        target = destination / "data" / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())  # Local: no JSON reformatting changes the source evidence.
        inputs[target.relative_to(destination).as_posix()] = file_hash(target)
    supplementary = [root.parent / name for name in ("duration-R1.json", "entropy-R1.json")]
    supplementary += [root.parent.parent / f"duration-entropy.{name}.log" for name in ("stdout", "stderr")]
    supplementary.append(Path("configs/duel-duration-entropy-v1.json"))
    for path in supplementary:
        if not path.exists():
            continue  # Local: a copied evidence bundle may omit host receipts; list only inputs actually present.
        target = destination / "data" / "execution" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())  # Local: include raw host telemetry and launch logs without rewriting bytes.
        inputs[target.relative_to(destination).as_posix()] = file_hash(target)
    audit = {"created_utc": timestamp(), "report_source_revision": source_revision(), "raw_inputs": inputs,
             "initial_fingerprints": fingerprints, "control_prefix": prefix, "analysis_reproduced": True,
             "checkpoint_hashes": receipt["checkpoints"], "endpoints": 12, "independent_training_seeds": 1,
             "final_test_accessed": False, "deployment_eligible": False}
    write_json(destination / "audit.json", audit)  # Local: numerical verification is distinct from later visual inspection.
    return evaluations, analysis


def plot_duration_evaluations(evaluations, analysis, destination):
    """Local: show all endpoints and both contrasts; global: distinguish duration from entropy and primary from secondary."""
    import matplotlib  # Local: use the installed standard plotting implementation.
    matplotlib.use("Agg")  # Local: artifact export consumes no game rendering resources.
    from matplotlib import pyplot as plt

    destination = Path(destination)
    steps = np.array([0, 1_245_184, 4_980_736]) / 1e6  # Local: unequal endpoint spacing reflects actual exposure.
    for mode, runs in evaluations.items():
        figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        breakdown = []  # Local: reuse exactly these final summaries for spawn/count breakdowns.
        for label, color in (("control", "#0072B2"), ("low-entropy", "#D55E00")):
            records = [runs[label][endpoint]["summary"] for endpoint in ("initial", "early", "final")]
            for axis, baseline in zip(axes.flat, ("random", "initial", "scripted", None), strict=True):
                scores = [row["score"] if baseline is None else row["baselines"][baseline]["score"] for row in records]
                axis.plot(steps, scores, "o-", color=color, label=label)
                axis.set(title="Equally weighted aggregate" if baseline is None else f"Versus {baseline}",
                         ylim=(0, 1), xlabel="Arena steps (millions)", ylabel="Round score")
                axis.grid(alpha=.2)  # Local: full zero-to-one score scale is shared across all endpoints.
            breakdown.append({"label": label, "final": records[-1]})
        axes[0, 0].legend(frameon=False)
        figure.suptitle(f"Duration / entropy — {mode} validation", fontsize=16)
        figure.text(.08, .025, "One matched training seed. Lines connect three measured endpoints; no intermediate scores are inferred.", fontsize=9)
        figure.subplots_adjust(left=.08, right=.98, bottom=.12, top=.89, hspace=.35, wspace=.27)
        for extension in ("png", "svg"):
            figure.savefig(destination / f"validation-{mode}.{extension}", dpi=160, facecolor="white")
        plt.close(figure)
        plot_evaluation_breakdown(breakdown, destination, mode, title="duration / entropy")
    figure, axis = plt.subplots(figsize=(9, 5))
    for index, name in enumerate(("duration", "entropy")):
        row = analysis[name]
        low, high = row["ci95"]
        axis.errorbar(index, row["delta"], yerr=[[row["delta"] - low], [high - row["delta"]]], fmt="o", capsize=7)
        axis.annotate(f"{row['delta']:+.4f}", (index, high), xytext=(0, 10), textcoords="offset points", ha="center")
    axis.axhline(0, color="#777777", linewidth=1)
    axis.axhline(.10, color="#777777", linestyle="--", label="Registered hypothesis +0.10")
    axis.set(xticks=(0, 1), xticklabels=("Control final − early", "Low entropy − control final"),
             xlim=(-.6, 1.6), ylim=(-.08, .13), ylabel="Change in primary round score", title="Registered diagnostic contrasts")
    axis.legend(frameon=False, loc="upper right")
    figure.text(.12, .025, "95% paired match intervals condition on these weights; one training seed, no retraining uncertainty.", fontsize=9)
    figure.subplots_adjust(bottom=.17, top=.89)
    for extension in ("png", "svg"):
        figure.savefig(destination / f"contrasts.{extension}", dpi=160, facecolor="white")
    plt.close(figure)


def publish_duration_evidence(root, destination):
    """Local: audit, snapshot, render; global: a readable phase wrapper leaves training artifacts untouched."""
    root, destination = Path(root).resolve(), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    evaluations, analysis = audit_duration_evidence(root, destination)
    save_loss_gallery(root, destination / "curves")  # Local: every loss/reward/activity point retains its input hash.
    plot_duration_evaluations(evaluations, analysis, destination)
    audit = read_json(destination / "audit.json")
    audit["figures"] = {path.relative_to(destination).as_posix(): file_hash(path)
                        for path in sorted(destination.rglob("*")) if path.suffix in {".png", ".svg"}}
    write_json(destination / "audit.json", audit)
    logging.info("Duration/entropy evidence saved: %s raw files, %s figure pairs", len(audit["raw_inputs"]), len(audit["figures"]) // 2)

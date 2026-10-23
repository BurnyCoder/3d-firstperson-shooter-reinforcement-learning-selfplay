"""Local: audit and describe all confirmation endpoints; global: preserve failed acceptance and distinct uncertainty levels.

No matches or optimizer steps run here. Statistics reuse duel_analysis.py; original plotting glue
calls https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.errorbar.html .
Protocol thresholds are engineering decisions, not claims that these values are optimal.
"""

from datetime import datetime  # Local: verify the recorded selection preceded final-test access.
import logging  # Local: every publication phase enters the timestamped report log.
from pathlib import Path  # Local: keep raw records and figures in an explicit report root.
import shutil  # Local: exact-byte evidence copies retain their original hashes.

import numpy as np  # Local: plot the recorded scores without smoothing or invented endpoints.

from ministrike.experiments.duel_analysis import model_comparison
from ministrike.experiments.duel_evaluation import file_hash, summarize
from ministrike.experiments.egocentric_reporting import copy_records
from ministrike.experiments.evaluation_plots import plot_evaluation_breakdown
from ministrike.experiments.loss_plots import parse_metrics, save_loss_gallery
from ministrike.experiments.registration import read_json
from ministrike.records import source_revision, timestamp, write_json


def audit_confirmation(root):
    """Local: recalculate all summaries and registered contrasts; global: publication cannot substitute a selected subset."""
    root = Path(root)
    receipt, selection, analysis = [read_json(root / name) for name in ("evaluation-receipt.json", "selection.json", "analysis.json")]
    if receipt["status"] != "complete" or len(set(receipt["completed"])) != 24:
        raise ValueError("All 24 registered endpoints are required")
    if file_hash(root / "selection.json") != analysis["selection_sha256"]:
        raise ValueError("Selection changed after evaluation")
    evaluations, metrics, checked = {}, {}, []
    for candidate in receipt["checkpoints"]:
        seed = candidate["seed"]
        run = root / f"seed{seed}"
        result = read_json(run / "result.json")
        rows, warnings = parse_metrics((run / "metrics.jsonl").read_bytes())
        if not result["complete"] or result["frames"] != 5_464_064 or len(rows) != 667 or warnings:
            raise ValueError("Incomplete or unequal confirmation exposure")
        if any(not np.isfinite(row[key]) for row in rows for key in ("loss", "policy_loss", "critic_loss", "entropy", "gradient_norm", "approx_kl", "clip_fraction")):
            raise ValueError("Nonfinite confirmation optimizer metric")
        metrics[seed] = {"rows": len(rows), "finite_optimizer_metrics": True, "result": result}
        for endpoint, filename, hash_key in (("initial", "initial.pt", "initial_sha256"), ("final", "model.pt", "checkpoint_sha256")):
            path = run / filename
            if path.exists():  # Local: public raw-record reproductions may omit separately hosted checkpoint binaries.
                if file_hash(path) != candidate[hash_key]:
                    raise ValueError("Checkpoint bytes changed")
                checked.append(f"{seed}/{filename}")
            for split in ("validation", "test"):
                for mode, suffix in (("deterministic", ""), ("stochastic", "_stochastic")):
                    row = read_json(run / f"{endpoint}_{split}{suffix}.json")
                    if row["status"] != "complete" or summarize(row["rounds"]) != row["summary"] or row["identity"]["checkpoint_sha256"] != candidate[hash_key]:
                        raise ValueError("Saved endpoint does not reproduce from its raw rounds")
                    evaluations.setdefault(split, {}).setdefault(mode, {}).setdefault(seed, {})[endpoint] = row
    for split, modes in evaluations.items():
        for mode, records in modes.items():
            reproduced = model_comparison([{"seed": seed, "complete": True, **pair} for seed, pair in records.items()])
            if reproduced != analysis[split][mode]:
                raise ValueError("Registered model/match analysis differs")
    ranked = selection["candidates"]
    if selection["selected"] != max(ranked, key=lambda row: (row["score"], -row["seed"])):
        raise ValueError("Selection rule differs")
    return evaluations, analysis, {"checkpoints_verified_from_bytes": checked, "metrics": metrics, "selection": selection,
                                  "all_24_summaries_reproduced": True, "all_four_analyses_reproduced": True}


def plot_confirmation(evaluations, analysis, destination):
    """Local: show each seed, opponent and uncertainty level; global: model replication is never inferred from match count."""
    import matplotlib
    matplotlib.use("Agg")  # Local: generate standalone files without a game window.
    from matplotlib import pyplot as plt

    for split, modes in evaluations.items():
        for mode, records in modes.items():
            figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
            for seed, color in zip(sorted(records), ("#0072B2", "#D55E00", "#009E73"), strict=True):
                summaries = [records[seed][name]["summary"] for name in ("initial", "final")]
                for axis, opponent in zip(axes.flat, ("random", "initial", "scripted", None), strict=True):
                    values = [row["baselines"][opponent]["score"] if opponent else row["score"] for row in summaries]
                    axis.plot([0, 5.464064], values, "o-", label=f"Seed {seed}", color=color)
                    axis.set(title=f"Versus {opponent}" if opponent else "Equal-opponent / spawn-end mean",
                             ylim=(-.03, 1.03), xticks=[0, 5.464064], xticklabels=["Initial", "5.464M steps"], ylabel="Round score")
                    axis.grid(alpha=.2)
            axes[0, 0].legend(frameon=False)
            figure.suptitle(f"Confirmation — {mode} {split}", fontsize=16)
            figure.text(.08, .025, "Three independent models. Lines connect two measured endpoints; intermediate performance was not measured.", fontsize=9)
            figure.subplots_adjust(left=.08, right=.98, bottom=.11, top=.90, hspace=.28)
            _save_figure(figure, destination / f"{split}-{mode}")
            plot_evaluation_breakdown([{"label": f"Seed {seed}", "final": pair["final"]["summary"]} for seed, pair in records.items()],
                destination, mode, title="confirmation", split=split,
                note="Three independent models. n counts recorded trajectories. Fixed deterministic initial/script controls: two trajectories each.")
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    for axis, mode in zip(axes, ("deterministic", "stochastic"), strict=True):
        row = analysis["test"][mode]
        values = [(str(item["seed"]), item["improvement"], item["match_uncertainty"]["ci95"]) for item in row["models"]]
        values.append(("3-model mean", row["mean_improvement"], row["model_ci95"]))
        for index, (_label, delta, bounds) in enumerate(values):
            low, high = bounds
            axis.errorbar(index, delta, yerr=[[max(0, delta - low)], [max(0, high - delta)]], fmt="D" if index == 3 else "o",
                          capsize=6, color="#D55E00" if index == 3 else "#0072B2")
        axis.axhline(.20, linestyle="--", color="#777777", label="Registered improvement 0.20")
        axis.set(xticks=range(4), xticklabels=[value[0] for value in values], xlim=(-.4, 3.5), ylim=(0, .65),
                 title=f"{mode.capitalize()} final test", ylabel="Final minus initial score")
        axis.grid(axis="y", alpha=.2)
        axis.legend(frameon=False)
    figure.text(.075, .025, "Blue: paired conditional-match 95% intervals. Orange: independent-model t interval (n=3, df=2; normality assumed).", fontsize=9)
    figure.subplots_adjust(bottom=.15, top=.88, wspace=.25)
    _save_figure(figure, destination / "uncertainty")


def _save_figure(figure, stem):
    """Local: export the same figure as raster/vector; global: every numerical plot has a shareable original."""
    from matplotlib import pyplot as plt
    for extension in ("png", "svg"):
        figure.savefig(stem.with_suffix(f".{extension}"), dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def write_seed_reports(evaluations, analysis, audit, destination):
    """Local: retain all eight endpoints per model; global: the validation-selected model does not replace the other seeds."""
    for seed, metadata in audit["metrics"].items():
        result = metadata["result"]
        rows = ["| Split / mode / endpoint | Mean score | Random | Initial | Script | Elimination |",
                "|---|---:|---:|---:|---:|---:|"]
        for split, modes in evaluations.items():
            for mode, models in modes.items():
                for endpoint, record in models[seed].items():
                    summary = record["summary"]
                    scores = [summary["baselines"][name]["score"] for name in ("random", "initial", "scripted")]
                    rows.append(f"| {split} / {mode} / {endpoint} | {summary['score']:.9f} | " + " | ".join(f"{score:.9f}" for score in scores) + f" | {summary['elimination_fraction']:.9f} |")
        comparison = next(row for row in analysis["test"]["deterministic"]["models"] if row["seed"] == seed)
        text = (f"# Confirmation seed {seed}\n\n**Experimental: the three-model release fails the moving-script acceptance criterion.** "
                f"This independent scratch run completed {result['frames']:,} arena steps and {result['updates']} updates in "
                f"{result['elapsed_seconds']:.3f} trainer seconds. No replacement seed or extension was used. "
                f"Training revision `{result['source_revision']}`. [Exact config](data/seed{seed}/config.json), "
                f"[result](data/seed{seed}/result.json) and [metrics](data/seed{seed}/metrics.jsonl).\n\n"
                "## Method and rationale\n\nThe [registered protocol](../../../docs/duel-protocol.md) fixes the symmetric14m arena, "
                "six-metre opposing starts,20Hz elimination, PPO/GAE, R1 outcome/net-damage reward and scratch historical self-play. "
                "The actor has192local-history inputs, an observer-coordinate transform and2x128tanh hidden units; the separate "
                "critic has2x256tanh units and original input coordinates. Neither demonstrations nor scripted opponents enter training. "
                "The [confirmation resolution](../../../docs/amendments/2026-10-23-confirmation-resolution.md) explains the equal budget, "
                "20%time reserve, independent seeds and pre-specified thresholds. All hyperparameters are engineering decisions.\n\n"
                "## Every measured endpoint\n\n" + "\n".join(rows) + "\n\n"
                f"Primary paired improvement: **{comparison['improvement']:.9f}**; conditional match95% interval "
                f"`{comparison['match_uncertainty']['ci95']}`. This interval conditions on this one initial/final pair; "
                "the separate three-model analysis is in [the aggregate report](README.md). Fixed deterministic initial/script "
                "controls each supply two trajectories; random and all stochastic endpoints use128paired seeds/256rounds per opponent.\n\n"
                f"[All raw endpoint records](data/seed{seed}/) include wins/losses/draws, sides, damage, accuracy, kills, reload, aim saturation, "
                "inactivity and first-seed traces. [Audit](audit.json) binds every endpoint and checkpoint hash.\n\n"
                f"![Raw loss curves](curves/seed{seed}.png)\n\n[Reward/combat/behavior/optimizer graphs](curves/training/README.md). "
                "Reward curves are interval means per arena step, not episode returns. Self-play opponents change.\n\n"
                "## Interpretation\n\nThe model improves against the registered aggregate, but does not establish robust moving-target combat. "
                "A success on this fixed layout does not establish unseen-spawn or map generalization. No additional tuning followed final tests. "
                "See [sources](../../../docs/sources.md) for algorithm/API references and [the aggregate](README.md) for uncertainty and failed gates.\n")
        (destination / f"seed{seed}.md").write_text(text, encoding="utf-8")


def publish_confirmation_evidence(root, destination):
    """Local: audit, preserve, visualize and explain; global: one wrapper performs reporting without new gameplay."""
    root, destination = Path(root), Path(destination)
    evaluations, analysis, audit = audit_confirmation(root)
    destination.mkdir(parents=True, exist_ok=True)
    inputs = copy_records(root, destination / "data")
    for path in (root.parent / "manifest.json", root.parent / "launch.json", root.parent / "confirmation.stdout.log",
                 root.parent / "confirmation.stderr.log", Path("configs/duel-confirmation.json")):
        if path.exists():
            target = destination / "data/execution" / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            inputs[target.as_posix()] = file_hash(target)
    manifest_path = destination / "data/execution/manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if datetime.fromisoformat(audit["selection"]["frozen_utc"]) >= datetime.fromisoformat(manifest["final_test_started_utc"]):
            raise ValueError("Selection was not frozen before final tests")
        audit["selection_before_final_test"] = True
    save_loss_gallery(root, destination / "curves")
    plot_confirmation(evaluations, analysis, destination)
    write_seed_reports(evaluations, analysis, audit, destination)
    audit.update(created_utc=timestamp(), report_source_revision=source_revision(),
                 raw_inputs={Path(path).relative_to(destination).as_posix(): value for path, value in inputs.items()},
                 figures={path.relative_to(destination).as_posix(): file_hash(path) for path in destination.rglob("*") if path.suffix in {".png", ".svg"}},
                 plotted_summaries={split: {mode: {seed: {endpoint: row["summary"] for endpoint, row in pair.items()} for seed, pair in models.items()} for mode, models in modes.items()} for split, modes in evaluations.items()})
    write_json(destination / "audit.json", audit)
    logging.info("Confirmation report: %s raw files; %s figures; acceptance=%s", len(inputs), len(audit["figures"]), analysis["test"]["deterministic"]["acceptance"])
    return analysis

"""Local: audit and visualize the registered mode intervention; global: retain fixed-policy evidence without new matches.

Original reporting glue, using Matplotlib's documented image/errorbar/plot APIs:
https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.errorbar.html .
Numerical choices and trace selection: docs/amendments/2026-10-23-action-mode-diagnostic.md.
"""

from datetime import datetime, timezone  # Local: timestamp the evidence audit separately from gameplay execution.
from pathlib import Path  # Local: keep raw inputs and figures in this project.
from operator import itemgetter  # Local: sort paired keys with the standard library.
import logging  # Local: retain completed audit counts in the timestamped report log.
import shutil  # Local: copy exact bytes rather than reserialize already published raw records.

import numpy as np  # Local: aggregate explicit counts and preserve paired-analysis values.

from ministrike.experiments.duel_evaluation import file_hash  # Local: reuse SHA256 checkpoint/evidence hashing.
from ministrike.experiments.mode_analysis import compare_modes  # Local: reproduce the registered analysis, without another formula.
from ministrike.experiments.registration import git, read_json  # Local: source and configuration provenance.
from ministrike.records import write_json  # Local: atomic derived-artifact writes.

MODES = ("DD", "DS", "SD", "SS")  # Local: learner mode first, frozen initial-opponent mode second.


def audit_modes(root):
    """Local: reproduce outcomes and statistics; global: tracing must not alter earlier pilot results."""
    cells = {reward: {mode: read_json(root / f"{reward}-{mode}.json") for mode in MODES}
             for reward in ("R0", "R1", "R2")}
    checks, summaries, traces = [], {}, []
    for reward, modes in cells.items():
        for mode, cell in modes.items():
            if cell["status"] != "complete":
                raise ValueError(f"Incomplete diagnostic cell: {reward}-{mode}")
            rows = cell["rounds"]
            if mode in ("DD", "SS"):
                suffix = "_stochastic" if mode == "SS" else ""
                old_path = Path(f"runs/duel-fixed-v1/pilots/{reward}/seed5201/final_validation{suffix}.json")
                previous = [row for row in read_json(old_path)["rounds"] if row["baseline"] == "initial"]
                key = itemgetter("side", "seed")  # Local: ignore serialization order, not any outcome/counter field.
                exact = sorted(rows, key=key) == sorted(previous, key=key)
                checks.append({"cell": f"{reward}-{mode}", "rows": len(rows), "exact": exact,
                               "previous_path": str(old_path), "previous_sha256": file_hash(old_path)})
                if not exact:
                    raise ValueError(f"Reproducibility control differs: {reward}-{mode}")
            decisions, shots = sum(row["decisions"] for row in rows), sum(row["shots"] for row in rows)
            summaries[f"{reward}-{mode}"] = {**cell["summary"],
                "mean_damage": float(np.mean([row["damage"] for row in rows])),
                "mean_seconds": float(np.mean([row["seconds"] for row in rows])),
                "mean_shots": shots / len(rows), "accuracy": sum(row["hits"] for row in rows) / shots if shots else None,
                **{key: sum(row[key] for row in rows) / decisions
                   for key in ("reload_ticks", "aim_saturated_ticks", "inactive_ticks")}}
            for trace in cell["traces"]:
                for player in (0, 1):
                    states = [step["players"][player] for step in trace["steps"]]
                    movement = np.linalg.norm(np.diff([state["position"] for state in states], axis=0), axis=1)
                    traces.append({"cell": f"{reward}-{mode}", "side": trace["side"], "seed": trace["seed"],
                        "controller": "learner" if player == trace["side"] else "opponent", "ticks": len(states),
                        "visible_fraction": float(np.mean([state["visible"] for state in states])),
                        "fire_request_fraction": float(np.mean([state["control"][4] for state in states])),
                        "reload_request_fraction": float(np.mean([state["control"][5] for state in states])),
                        "empty_magazine_fraction": float(np.mean([state["ammo"] == 0 for state in states])),
                        "distance_metres": float(movement.sum()),
                        "stationary_fraction": float(np.mean(movement < 1e-6)) if len(movement) else None})
    analysis = compare_modes(cells)
    if analysis != read_json(root / "analysis.json"):
        raise ValueError("Registered statistical analysis did not reproduce")
    return cells, {"checked_utc": datetime.now(timezone.utc).isoformat(), "source_revision": git("rev-parse", "HEAD"),
                   "reproduction": checks, "analysis_exact": True, "cells": summaries, "trace_summaries": traces}


def save_figure(figure, path):
    """Local: export the identical figure to raster and vector; global: release resources between artifacts."""
    from matplotlib import pyplot as plt
    for suffix in ("png", "svg"):
        figure.savefig(path.with_suffix(f".{suffix}"), dpi=160, facecolor="white")
    plt.close(figure)


def mode_figures(cells, analysis, destination):
    """Local: draw registered comparisons and all fixed trace selections; global: never cherry-pick successful trajectories."""
    import matplotlib
    matplotlib.use("Agg")  # Local: reproducible file rendering without a game window or GPU workload.
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.5, 1]})
    scores = [[analysis["models"][reward]["scores"][mode] for mode in MODES] for reward in cells]
    heat = axes[0].imshow(scores, vmin=0, vmax=1, cmap="cividis", aspect="auto")
    for row, reward in enumerate(cells):
        for column, mode in enumerate(MODES):
            counts = cells[reward][mode]["summary"]["baselines"]["initial"]
            axes[0].text(column, row, f"{scores[row][column]:.3f}\n{counts['wins']}/{counts['draws']}/{counts['losses']}",
                         ha="center", va="center", color="white" if scores[row][column] < .6 else "black")
    axes[0].set(xticks=range(4), xticklabels=MODES, yticks=range(3), yticklabels=list(cells),
                xlabel="Learner / initial-opponent mode", title="Round score; counts below are W / D / L")
    figure.colorbar(heat, ax=axes[0], label="Score (win 1 / draw 0.5 / loss 0)", fraction=.045)
    effects = [analysis["models"][reward]["learner_effect_opponent_D"] for reward in cells] + [analysis["primary"]]
    for index, effect in enumerate(effects):
        low, high = effect["ci95"]
        axes[1].errorbar(effect["delta"], index, xerr=[[effect["delta"] - low], [high - effect["delta"]]],
                         fmt="o", capsize=4, color="#0072B2" if index < 3 else "#D55E00")
    axes[1].axvline(0, color="grey", linewidth=1)
    axes[1].axvline(.10, color="#D55E00", linestyle="--", label="Registered aggregate threshold")
    axes[1].set(yticks=range(4), yticklabels=[*cells, "Mean"], xlabel="Score(SD) minus score(DD)",
                title="Change learner mode, hold opponent D", xlim=(-.12, .14))
    axes[1].invert_yaxis()
    axes[1].legend(fontsize=8, loc="lower left")
    figure.suptitle("Action-mode intervention | frozen reward pilots", fontsize=16)
    figure.text(.06, .025, "D = argmax; S = categorical sampling. DD: 2 fixed trajectories; other cells: 256 matches.\n"
                "95% intervals resample paired match seeds, conditional on these same-seed models; no model-population interval.", fontsize=9)
    figure.subplots_adjust(left=.06, right=.97, bottom=.23, top=.85, wspace=.50)
    save_figure(figure, destination / "mode-comparison")
    for reward, modes in cells.items():
        for side in (0, 1):
            figure, axes = plt.subplots(2, 4, figsize=(14, 6))
            for column, mode in enumerate(MODES):
                trace = next(trace for trace in modes[mode]["traces"] if trace["side"] == side)
                row = next(row for row in modes[mode]["rounds"] if row["side"] == side and row["seed"] == trace["seed"])
                for player, label, color in ((side, "Learner", "#0072B2"), (1 - side, "Opponent", "#D55E00")):
                    positions = np.array([step["players"][player]["position"] for step in trace["steps"]])
                    axes[0, column].plot(positions[:, 0], positions[:, 2], color=color, linewidth=1.2, label=label)
                    axes[0, column].scatter(*positions[0, [0, 2]], color=color, marker="o", s=20)
                    axes[0, column].scatter(*positions[-1, [0, 2]], color=color, marker="x", s=30)
                    axes[1, column].plot([step["tick"] / 20 for step in trace["steps"]],
                                         [step["players"][player]["health"] for step in trace["steps"]], color=color)
                axes[0, column].set(xlim=(-7, 7), ylim=(-7, 7), aspect="equal", xlabel="x (m)", ylabel="z (m)",
                                    title=f"{mode} | score {row['score']:g} | {row['reason']}")
                axes[1, column].set(xlim=(0, 20), ylim=(-5, 105), xlabel="Time (s)", ylabel="Health before action")
                axes[1, column].grid(alpha=.2)
            axes[0, 0].legend(fontsize=8)
            figure.suptitle(f"{reward}, learner spawn end {side}, fixed trace seed 3000000", fontsize=15)
            figure.text(.06, .015, "All prospectively selected traces shown. Circle = first position; cross = last pre-action position. "
                        "Terminal damage occurs after the last health sample.", fontsize=9)
            figure.subplots_adjust(left=.06, right=.98, bottom=.11, top=.86, hspace=.55, wspace=.38)
            save_figure(figure, destination / f"trace-{reward}-end{side}")


def publish_mode_evidence(root, output):
    """Local: verify, copy, plot, hash; global: the publication is reproducible from frozen records alone."""
    root, output = Path(root), Path(output)
    cells, audit = audit_modes(root)  # Local: fail before publication if tracing changed outcomes or paired analysis.
    data = output / "data/action-modes-v1"
    figures = output / "action-modes-v1"
    data.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    inputs = sorted(root.glob("*.json")) + sorted((root / "logs").glob("*.log"))
    for source in inputs:
        target = data / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)  # Local: retain exact raw bytes, including traces and transient retry logs.
    audit["inputs"] = [{"path": str(path), "sha256": file_hash(path)} for path in sorted(data.rglob("*")) if path.is_file()]
    mode_figures(cells, read_json(root / "analysis.json"), figures)
    audit["figures"] = [{"path": str(path), "sha256": file_hash(path)} for path in sorted(figures.glob("*"))]
    write_json(output / "action-mode-verification.json", audit)
    logging.info("mode_evidence_complete cells=%s reproduced_rows=%s figures=%s", len(audit["cells"]),
                 sum(item["rows"] for item in audit["reproduction"]), len(audit["figures"]))
    return audit

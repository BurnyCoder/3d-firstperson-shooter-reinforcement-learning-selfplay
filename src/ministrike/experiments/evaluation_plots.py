"""Local: visualize registered pilot validation; global: keep match uncertainty distinct from retraining uncertainty.

Error-bar API: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.errorbar.html .
The shared paired bootstrap implements the registered analysis in docs/duel-protocol.md.
Additional original plotting glue uses the official annotated-heatmap and stacked-bar APIs:
https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
https://matplotlib.org/stable/gallery/lines_bars_and_markers/horizontal_barchart_distribution.html
"""

import json  # Local: plots consume recorded evaluation endpoints, not training round counts.
import logging  # Local: surface pending endpoints instead of inventing scores.
from pathlib import Path  # Local: separate validation figures from future final-test analysis.

import numpy as np  # Local: place grouped bars and preserve explicit numeric scores.

from ministrike.experiments.duel_analysis import paired_match_interval  # Local: reuse pairing and resampling, not a second estimator.
from ministrike.experiments.curve_plots import COLORS  # Local: shared visual identity across evidence figures.
from ministrike.experiments.duel_evaluation import file_hash  # Local: reuse exact source hashing from the evaluation pipeline.
from ministrike.records import source_revision, timestamp, write_json  # Local: portable plot provenance.


def plot_evaluation_breakdown(records, destination, mode, title="reward pilots"):
    """Local: expose side asymmetry and draws; global: deterministic repetitions never inflate the displayed sample count."""
    from matplotlib import pyplot as plt  # Local: the caller has already selected the noninteractive backend.

    names = ("random", "initial", "scripted")  # Local: preserve the protocol's opponent ordering.
    matrix = np.asarray([[score for name in names for score in record["final"]["baselines"][name]["side_scores"]]
                         for record in records])  # Local: display each spawn end before averaging it away.
    figure, axes = plt.subplots(1, 2, figsize=(14, 6.5), gridspec_kw={"width_ratios": [1.1, 1.25]})
    heat = axes[0].imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")  # Local: same fixed score scale for both action modes.
    axes[0].set(xticks=np.arange(6), xticklabels=[f"{label}\nend {side}" for label in ("Random", "Initial", "Script") for side in (0, 1)],
                yticks=np.arange(len(records)), yticklabels=[record["label"] for record in records], title="Final score by opponent and spawn end")
    for (row, column), value in np.ndenumerate(matrix):
        axes[0].text(column, row, f"{value:.3f}", ha="center", va="center", color="black" if value > .55 else "white")
    figure.colorbar(heat, ax=axes[0], orientation="horizontal", fraction=.05, pad=.17,
                    label="Round score: win 1, draw 0.5, loss 0")  # Local: numerical annotations accompany color encoding.
    entries = [(f"{record['label']} / {name}", record["final"]["baselines"][name]) for record in records for name in names]
    left = np.zeros(len(entries))  # Local: stacked widths partition the actual observed matches.
    for outcome, color in (("wins", "#009E73"), ("draws", "#AEB7C2"), ("losses", "#D55E00")):
        fractions = np.asarray([item[outcome] / item["matches"] for _, item in entries])
        axes[1].barh(np.arange(len(entries)), fractions, left=left, color=color, label=outcome.capitalize())
        for index, ((_, item), fraction) in enumerate(zip(entries, fractions, strict=True)):
            if fraction >= .08:
                axes[1].text(left[index] + fraction / 2, index, str(item[outcome]), ha="center", va="center", fontsize=9)
        left += fractions  # Local: offsets accumulate fractions, while labels retain integer match counts.
    for index, (_, item) in enumerate(entries):
        axes[1].text(1.02, index, f"n={item['matches']}", va="center", fontsize=9, clip_on=False)  # Local: two deterministic matches stay two.
    axes[1].set(yticks=np.arange(len(entries)), yticklabels=[label for label, _ in entries], xlim=(0, 1),
                xlabel="Fraction of observed matches; labels are counts", title="Final wins, draws and losses")
    axes[1].invert_yaxis()  # Local: show R0 first in both panels.
    axes[1].legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(.5, -.16))
    figure.suptitle(f"MiniStrike {title} — {mode} validation breakdown", fontsize=16, fontweight="bold")
    figure.text(.055, .025, "Seed 5201 only. n counts recorded trajectories, not independently trained models. Validation only; no new matches run.", fontsize=9)
    figure.subplots_adjust(left=.055, right=.93, bottom=.22, top=.85, wspace=.40)
    for extension in ("png", "svg"):
        figure.savefig(destination / f"validation-{mode}-breakdown.{extension}", dpi=160, facecolor="white")
    plt.close(figure)  # Local: each action mode has a separate artifact, backed by the existing validation JSON receipt.


def plot_pilot_evaluations(input_root, output_root):
    """Local: export primary and secondary validation figures; global: no final-test access or reward selection occurs here."""
    import matplotlib
    matplotlib.use("Agg")  # Local: standard CPU raster/vector export never launches the game renderer.
    from matplotlib import pyplot as plt

    root, destination = Path(input_root), Path(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    folders = sorted(root.glob("pilots/*/seed*/config.json"))  # Local: calibration and future confirmation are not pilot replicas.
    for stochastic in (False, True):
        suffix = "_stochastic" if stochastic else ""
        records, evidence = [], {}
        for config_path in folders:
            paths = [config_path.parent / f"{endpoint}_validation{suffix}.json" for endpoint in ("initial", "final")]
            if not all(path.exists() for path in paths):
                logging.info("Pilot validation plot pending: %s", paths)
                break  # Local: do not selectively omit an unfinished candidate from a comparison.
            initial, final = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
            if any(item["status"] != "complete" for item in (initial, final)):
                break  # Local: an incomplete endpoint is not an observed zero score.
            config = json.loads(config_path.read_text(encoding="utf-8"))
            records.append({"label": config["reward"], "seed": config["seed"], "initial": initial["summary"],
                            "final": final["summary"], "delta": final["summary"]["score"] - initial["summary"]["score"],
                            "interval": paired_match_interval(initial, final)["ci95"]})
            evidence[config["reward"]] = {"config": config, "inputs": {str(path): file_hash(path) for path in paths},
                                          "endpoint_identities": [initial["identity"], final["identity"]]}
        if not records or len(records) != len(folders):
            continue  # Local: retry after remaining scheduled evaluations finish, without rerunning completed matches.
        mode = "stochastic" if stochastic else "deterministic"
        figure, axes = plt.subplots(1, 2, figsize=(12, 5.4))
        names, positions = ("random", "initial", "scripted"), np.arange(3)
        width = .75 / len(records)
        for index, record in enumerate(records):
            values = [record["final"]["baselines"][name]["score"] for name in names]
            axes[0].bar(positions + (index - (len(records) - 1) / 2) * width, values, width,
                        color=COLORS[index], label=record["label"])  # Local: each bar is one checkpoint's observed score.
            delta, (lower, upper) = record["delta"], record["interval"]
            axes[1].errorbar(index, delta, yerr=[[delta - lower], [upper - delta]], fmt="o", capsize=6,
                             color=COLORS[index], markersize=8)
            axes[1].annotate(f"{delta:+.3f}", (index, upper), xytext=(0, 8), textcoords="offset points", ha="center")
        axes[0].set(xticks=positions, xticklabels=("Random", "Initial policy", "Moving script"),
                    ylim=(0, 1), ylabel="Round score (win 1, draw 0.5, loss 0)", title="Final validation score by opponent")
        axes[0].legend(frameon=False)
        axes[1].set(xticks=np.arange(len(records)), xticklabels=[r["label"] for r in records],
                    xlim=(-.5, len(records) - .5), ylabel="Final minus initial aggregate score",
                    title="Paired improvement with 95% match interval")  # Local: leave room for edge-point labels.
        axes[1].axhline(0, color="#666666", linewidth=1)
        if not stochastic:
            axes[1].axhline(.10, linestyle="--", color="#777777", label="Registered pilot gate +0.10")
            axes[1].legend(frameon=False, loc="upper left", fontsize=9)
        axes[1].set_ylim(-.02, max(.13, max(r["interval"][1] for r in records) + .07))
        for axis in axes:
            axis.grid(axis="y", alpha=.2)
            axis.set_axisbelow(True)
        figure.suptitle(f"MiniStrike reward pilots — {mode} validation", fontsize=16, fontweight="bold")
        figure.text(.075, .035, "Seed 5201 only; validation, not final test. Intervals condition on each model pair, not independent training seeds.", fontsize=9)
        figure.subplots_adjust(left=.075, right=.98, bottom=.16, top=.85, wspace=.32)
        for extension in ("png", "svg"):
            figure.savefig(destination / f"validation-{mode}.{extension}", dpi=160, facecolor="white")
        plt.close(figure)
        write_json(destination / f"validation-{mode}.json", {"created_utc": timestamp(), "plot_source_revision": source_revision(),
                   "records": records, "evidence": evidence, "uncertainty": "paired match bootstrap; one training seed"})
        plot_evaluation_breakdown(records, destination, mode)  # Local: reuse the same endpoint snapshots and identity receipt.
        logging.info("Validation plots saved: %s", destination / f"validation-{mode}.png")

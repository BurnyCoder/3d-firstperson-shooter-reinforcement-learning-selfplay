"""Local: loss plots from recorded updates; global: transparent evidence, never a proxy claim of skill.

Original plotting glue calls Matplotlib's documented APIs:
https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html
https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
All points are raw update averages. Missing/nonfinite values create gaps; there is no smoothing.
"""

import hashlib  # Local: byte identity ties each figure to its exact input snapshot.
import json  # Local: read actual logger records without inferring unavailable metrics.
import logging  # Local: preserve generated-file identities and data warnings in timestamped logs.
from pathlib import Path  # Local: explicit source/output roots bound evidence discovery.

import numpy as np  # Local: NaN values create honest gaps in standard Matplotlib lines.

from ministrike.records import source_revision, timestamp, write_json  # Local: reuse project provenance and persistence.
from ministrike.experiments.duel_evaluation import file_hash  # Local: reuse streamed artifact hashing without loading checkpoint tensors.

METRICS = (("loss", "Total optimization loss"), ("policy_loss", "Policy loss"),
           ("critic_loss", "Value loss"), ("entropy", "Policy entropy (nats)"))  # Local: separate terms have different meanings/scales.
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")  # Local: engineering choice of contrasting series colors.


def parse_metrics(raw):
    """Local: parse one byte snapshot; global: interrupted writes and duplicate updates remain visible."""
    lines, rows, warnings = raw.decode("utf-8").splitlines(), [], []
    for index, line in enumerate(lines):
        if not line.strip():
            continue  # Local: blank separators carry no numeric observation.
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            if index != len(lines) - 1:
                raise ValueError(f"Malformed metrics line {index + 1}") from error
            warnings.append(f"Unfinished final line {index + 1} excluded; original bytes preserved")
            continue  # Local: a live append can end halfway through the last JSON record.
        if rows and row["frames"] <= rows[-1]["frames"]:
            raise ValueError("Canonical metric frames must be strictly increasing; audit recovery suffixes")
        rows.append(row)  # Local: retain original order and every complete record.
    return rows, warnings


def metric_series(rows, key):
    """Local: preserve genuine zero and missingness; global: missing loss is not successful zero error."""
    values = np.asarray([row.get(key, np.nan) for row in rows], dtype=float)
    return np.where(np.isfinite(values), values, np.nan)  # Local: infinities/NaNs are gaps, never clipped into finite values.


def _read_run(path, root, destination):
    """Local: snapshot one canonical run; global: incomplete/deferred experiments remain eligible for plots."""
    name = "__".join(path.parent.relative_to(root).parts) or root.name
    raw = path.read_bytes()  # Local: parse and hash the same read, avoiding races with a growing metrics file.
    rows, warnings = parse_metrics(raw)
    config_path, result_path = path.parent / "config.json", path.parent / "result.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
    registration = config.get("registration", "legacy")
    group = "reward-pilots" if registration.startswith("A-PILOT") else registration.lower()
    status = "complete" if result.get("complete") else "partial or deferred"
    label = f"{config.get('reward', config.get('condition', name))} / seed {config.get('seed', '?')}"
    if registration == "A-CAL":
        label = name  # Local: hardware cells sharing a seed still need distinct identities in individual plots.
    snapshot = destination / "data" / name
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "metrics.jsonl").write_bytes(raw)  # Local: raw evidence makes the figure reproducible after logs advance.
    for original in (config_path, result_path):
        if original.exists():
            (snapshot / original.name).write_bytes(original.read_bytes())  # Local: allowlisted metadata only, never .env.
    missing = {key: int(np.isnan(metric_series(rows, key)).sum()) for key, _ in METRICS}
    receipt = {"label": label, "status": status, "group": group, "source_path": path.as_posix(),
               "source_revision": result.get("source_revision", config.get("source_revision", "unavailable")),
               "metrics_sha256": hashlib.sha256(raw).hexdigest(), "rows": len(rows),
               "last_arena_steps": rows[-1]["frames"] if rows else None, "missing_or_nonfinite": missing,
               "warnings": warnings, "config": config,
               "checkpoint_sha256": {filename: file_hash(path.parent / filename)
                                     for filename in ("initial.pt", "model.pt") if (path.parent / filename).exists()}}
    return {"name": name, "rows": rows, "receipt": receipt}  # Local: numeric data stays separate from display metadata.


def _plot(records, destination, title):
    """Local: export one four-panel figure; global: loss terms and runs are never averaged into a false performance score."""
    import matplotlib  # Local: reporting alone imports the graphics backend.
    matplotlib.use("Agg")  # Local: no window, rendering context or GPU competes with training.
    from matplotlib import pyplot as plt

    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
        figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        for axis, (key, caption) in zip(axes.flat, METRICS, strict=True):
            for index, record in enumerate(records):
                values = metric_series(record["rows"], key)
                if np.isfinite(values).any():
                    x = [row["frames"] / 1e6 for row in record["rows"]]  # Local: common millions-of-arena-steps axis.
                    axis.plot(x, values, color=COLORS[index % len(COLORS)], linewidth=1.15,
                              label=record["receipt"]["label"])  # Local: raw logged update averages, without smoothing.
            axis.set_title(caption, loc="left", fontweight="bold")
            axis.set_xlabel("Arena steps (millions)")
            axis.set_ylabel("Logged value")
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))  # Local: standard scientific notation keeps tiny calibration losses legible.
            axis.grid(alpha=.2)
            axis.set_xlim(left=0)
            if key in {"critic_loss", "entropy"}:
                axis.set_ylim(bottom=0)  # Local: nonnegative diagnostics retain their meaningful origin.
            if axis.lines:
                axis.legend(fontsize=9, frameon=False)
            else:
                axis.text(.5, .5, "Not recorded", transform=axis.transAxes, ha="center", color="#666666")
        figure.suptitle(title, fontsize=17, fontweight="bold", x=.075, ha="left")
        figure.text(.075, .025, "Every logged update; no smoothing. Loss is not a win rate. Reward definitions affect loss scales.", fontsize=10)
        figure.subplots_adjust(left=.10, right=.98, bottom=.12, top=.89, hspace=.40, wspace=.25)
        for suffix in ("png", "svg"):
            figure.savefig(destination.with_suffix(f".{suffix}"), dpi=160, facecolor="white")
        plt.close(figure)  # Local: release plotting memory before the next experiment.


def save_loss_gallery(input_root, output_root, pattern="**/metrics.jsonl"):
    """Local: discover, snapshot, plot and index; global: one reusable reporting phase covers new and deferred runs."""
    root, destination = Path(input_root).resolve(), Path(output_root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(root.glob(pattern)):
        relative = path.relative_to(root)
        if {"publication", "execution_amendments", "recovery"}.intersection(relative.parts) or destination in path.parents:
            continue  # Local: archived copies remain audit evidence, not additional independent models.
        record = _read_run(path, root, destination)
        records.append(record)
        _plot([record], destination / record["name"], f"{record['receipt']['label']} — {record['receipt']['status']}")
        logging.info("Loss figure saved: %s (%s updates)", destination / record["name"], len(record["rows"]))
    groups = sorted({record["receipt"]["group"] for record in records})
    comparisons = []
    for group in groups:
        members = [record for record in records if record["receipt"]["group"] == group]
        if 1 < len(members) <= 6:
            _plot(members, destination / group, f"MiniStrike — {group.replace('-', ' ')}")
            comparisons.append(group)  # Local: large calibration sets keep individual plots instead of an unreadable overlay.
    write_json(destination / "evidence.json", {"created_utc": timestamp(), "plot_source_revision": source_revision(),
               "method": "raw update averages; no smoothing; missing/nonfinite values are gaps", "comparisons": comparisons,
               "runs": {record["name"]: record["receipt"] for record in records}})
    links = [f"- [{name} comparison]({name}.png) · [SVG]({name}.svg)" for name in comparisons]
    links += [f"- [{r['receipt']['label']} ({r['receipt']['status']})]({r['name']}.png) · [SVG]({r['name']}.svg)" for r in records]
    text = ("# Saved loss graphs\n\n" + "\n".join(links) + "\n\n"
            "Total loss combines the policy, weighted value and entropy terms. Policy loss is a PPO surrogate, not a win rate. "
            "Value targets change as the policy and opponents change; different reward scales also change losses. Entropy measures "
            "action-distribution spread, not competence. These are logged update averages, with no extra smoothing or confidence bands. "
            "Missing/nonfinite metrics remain gaps or Not recorded panels; evidence.json counts them. No new measurements are inferred.\n\n"
            "Exact input snapshots are in data/; evidence.json records source revisions, checkpoint hashes, seeds and configuration. "
            "Rerun the plotting wrapper with this data directory as --input to reproduce the curves. "
            "[TorchRL PPO loss](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.objectives.ClipPPOLoss.html), "
            "[Matplotlib](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html).\n")
    (destination / "README.md").write_text(text, encoding="utf-8")
    return {"runs": len(records), "comparisons": comparisons, "directory": str(destination)}

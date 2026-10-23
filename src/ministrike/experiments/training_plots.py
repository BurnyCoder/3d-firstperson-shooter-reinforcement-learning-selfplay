"""Local: save interpretable training telemetry; global: compare measured behavior without confusing self-play with validation.

Counters and their derivation are in training_plot_metrics.py; shared rendering is in curve_plots.py.
The user requested these additional views. Units/limits are engineering display choices, not acceptance thresholds.
"""

import logging  # Local: record every exported group in the timestamped reporting log.

import numpy as np  # Local: convert unavailable values to portable JSON null.

from ministrike.experiments.curve_plots import plot_curves  # Local: reuse the loss gallery's four-panel renderer.
from ministrike.experiments.training_plot_metrics import interval_metrics  # Local: keep arithmetic independently testable.
from ministrike.records import source_revision, timestamp, write_json  # Local: bind outputs to their implementation.

PANELS = {  # Local: key, caption, physical unit and meaningful axis limits for each measured quantity.
    "rewards": (("reward_total", "Total reward", "Reward / arena step", None),
                ("reward_outcome", "Terminal outcome component", "Reward / arena step", None),
                ("reward_damage", "Net damage component", "Reward / arena step", None),
                ("reward_potential", "Aim potential component", "Reward / arena step", None)),
    "combat": (("accuracy", "Hit accuracy", "Hits / shots", (0, 1)),
               ("damage_rate", "Damage dealt", "Health points / 1000 arena steps", (0, None)),
               ("shots_rate", "Shots fired", "Shots / 1000 arena steps", (0, None)),
               ("kills_rate", "Opponent eliminations", "Kills / 1000 arena steps", (0, None))),
    "behavior": (("reload_fraction", "Reload occupancy", "Fraction of learner decisions", (0, 1)),
                 ("aim_saturated_fraction", "Pitch at its limit", "Fraction of learner decisions", (0, 1)),
                 ("inactive_fraction", "All decoded controls zero", "Fraction of learner decisions", (0, 1)),
                 ("timeout_fraction", "Timeouts", "Fraction of completed rounds", (0, 1))),
    "selfplay-outcomes": (("win_fraction", "Wins", "Fraction of completed rounds", (0, 1)),
                          ("draw_fraction", "Draws", "Fraction of completed rounds", (0, 1)),
                          ("loss_fraction", "Losses", "Fraction of completed rounds", (0, 1)),
                          ("round_score", "Training round score", "Win 1 / draw 0.5 / loss 0", (0, 1))),
    "optimization": (("approx_kl", "Approximate policy KL", "Logged estimate", None),
                     ("clip_fraction", "PPO clip fraction", "Fraction", (0, 1)),
                     ("gradient_norm", "Gradient norm before clipping", "L2 norm", (0, None)),
                     ("step_rate", "Collection + update throughput", "Arena steps / trainer second", (0, None))),
}
FOOTERS = {  # Local: semantic limits remain visible in standalone downloaded images.
    "rewards": "Per-update counter differences / arena steps; no smoothing. Reward definitions differ; this is not episode return or skill.",
    "combat": "Per-update learner event rates; no smoothing. Opponents change during self-play; these are not fixed-benchmark scores.",
    "behavior": "Per-update occupancy and round fractions; no smoothing. Zero controls is a narrow inactivity measure, not a movement test.",
    "selfplay-outcomes": "Completed rounds within each update; no smoothing. Opponents change during self-play; validation is reported separately.",
    "optimization": "Logged optimizer averages; throughput uses adjacent trainer elapsed times. No smoothing or independent-model uncertainty.",
}


def save_training_gallery(records, destination):
    """Local: write derived numbers and per-run/comparison figures; global: every curve retains its raw input identity."""
    selected = [record for record in records if record["receipt"]["config"].get("registration") != "A-CAL"
                and any("agent_decisions" in row.get("behavior", {}) for row in record["rows"])]  # Local: timing fixtures/old missing telemetry are not gameplay evidence.
    if not selected:
        return 0  # Local: never invent rewards for historical logs that did not record them.
    destination.mkdir(parents=True, exist_ok=True)
    derived = [dict(record, rows=interval_metrics(record["rows"])) for record in selected]  # Local: derive once from the exact loss-gallery snapshot.
    keys = {key for panels in PANELS.values() for key, *_ in panels} | {"frames"}
    for record in derived:
        rows = [{key: row.get(key) if row.get(key) is not None and np.isfinite(row[key]) else None for key in sorted(keys)}
                for row in record["rows"]]  # Local: JSON null records undefined denominators without the nonstandard NaN literal.
        write_json(destination / f"{record['name']}.json", {"plot_source_revision": source_revision(),
                   "created_utc": timestamp(), "input": record["receipt"], "series": rows})  # Local: include config, checkpoint and raw metric hashes.
    collections = [(record["name"], [record]) for record in derived]  # Local: each actual model has its own complete set.
    for group in sorted({record["receipt"]["group"] for record in derived}):
        members = [record for record in derived if record["receipt"]["group"] == group]
        if 1 < len(members) <= 6:
            collections.append((group, members))  # Local: overlay a small comparable group without averaging seeds.
    links = []
    for name, members in collections:
        title = members[0]["receipt"]["label"] if len(members) == 1 else name.replace("-", " ")
        for category, panels in PANELS.items():
            stem = f"{name}-{category}"
            plot_curves(members, destination / stem, f"MiniStrike — {title} — {category.replace('-', ' ')}", panels, FOOTERS[category])
            links.append(f"- [{title}: {category}]({stem}.png) · [SVG]({stem}.svg)")
        logging.info("Reward/activity figures saved: %s (5 views)", destination / name)
    text = ("# Reward and training behavior\n\n" + "\n".join(links) + "\n\n"
            "[Definitions, source references and reproduction](../../../../docs/experiment-plots.md). "
            "Adjacent per-run JSON saves every plotted number (null means undefined), configuration, seed, source and checkpoint hashes. "
            "Exact metric bytes are in the parent gallery's data/ directory. Rewards are interval means per arena step, not episode returns. "
            "Self-play outcomes use changing opponents; fixed validation belongs in the parent gallery. "
            "These are exploratory descriptive plots of all recorded updates, without smoothing or independent-model error bands.\n")
    (destination / "README.md").write_text(text, encoding="utf-8")
    return len(collections) * len(PANELS)  # Local: counts describe actual exported figure pairs.

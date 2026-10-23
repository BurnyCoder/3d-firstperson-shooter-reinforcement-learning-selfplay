"""Local: reports from immutable raw records; global: claims, hashes and limitations travel together.

Reuse the existing ReportLab Markdown/PDF renderer; no second layout engine. Methods and
attributions are centralized in docs/sources.md; analysis is in duel_analysis.py.
"""

import json  # Local: literal config/results evidence, with no invented missing values.
from pathlib import Path  # Local: explicit project evidence paths.
import shutil  # Local: copy exact raw bytes into the versioned report bundle.

from ministrike.experiments.duel_analysis import model_comparison, paired_power
from ministrike.experiments.duel_evaluation import file_hash
from ministrike.experiments.loss_plots import save_loss_gallery  # Local: graphs use the same reusable evidence workflow as manual plotting.
from ministrike.experiments.evaluation_plots import plot_pilot_evaluations  # Local: pilot plots remain separate from confirmation statistics.
from ministrike.experiments.registration import read_json
from ministrike.experiments.reporting import _render_pdf  # Local: reuse tested source/PDF layout and attribution.
from ministrike.records import write_json


def create_duel_reports(root, destination):
    """Local: discover evidence, analyze and render; global: include incomplete/failed runs and every reward candidate."""
    root, destination = Path(root), Path(destination)
    if (root / "confirmation/analysis.json").exists():
        from ministrike.experiments.confirmation_reporting import publish_confirmation_evidence
        analysis = publish_confirmation_evidence(root / "confirmation", destination / "confirmation")
        if (destination / "paper.md").exists():
            _render_pdf((destination / "paper.md").read_text(encoding="utf-8"), destination / "paper.pdf")
        return analysis  # Local: preserve handwritten historical interpretation when regenerating the completed release.
    destination.mkdir(parents=True, exist_ok=True)
    save_loss_gallery(root, destination / "losses")  # Local: save every run's recorded loss terms without optimizing a policy.
    plot_pilot_evaluations(root, destination / "losses")  # Local: retain primary and secondary validation as different figures.
    records, table, evidence = [], [], {}
    for config_path in sorted(root.rglob("config.json")):
        if {"publication", "recovery", "execution_amendments"}.intersection(config_path.parts):
            continue  # Local: copied/archived files are evidence, not independently trained models.
        folder = config_path.parent
        name = "__".join(folder.relative_to(root).parts)
        config = read_json(config_path)
        result = read_json(folder / "result.json") if (folder / "result.json").exists() else {"complete": False, "stop_reason": "failed or interrupted"}
        data = destination / "data" / name
        data.mkdir(parents=True, exist_ok=True)
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix in {".json", ".jsonl", ".log"}:
                shutil.copyfile(path, data / path.name)  # Local: include metrics/failures, never credentials or model binaries in Git.
                evidence[str((data / path.name).relative_to(destination))] = file_hash(path)
        hashes = {label: file_hash(folder / label) for label in ("initial.pt", "model.pt") if (folder / label).exists()}
        initial = read_json(folder / "initial_test.json") if (folder / "initial_test.json").exists() else None
        final = read_json(folder / "final_test.json") if (folder / "final_test.json").exists() else None
        if config.get("registration") == "A-CONFIRM":
            records.append({"seed": config["seed"], "complete": result.get("complete", False), "initial": initial, "final": final})
        score = final["summary"]["score"] if final else "not evaluated"
        table.append(f"| {name} | {config['seed']} | {config.get('reward', 'R0')} | {result.get('frames', 'missing')} | {score} | {result.get('complete', False)} |")
        report = (f"# Experiment {name}\n\nRegistration `{config.get('registration')}`; seed `{config['seed']}`. "
                  f"Source `{result.get('source_revision', config.get('source_revision', 'unavailable'))}`. "
                  f"Status: `{result.get('stop_reason', 'unknown')}`, completed budget `{result.get('complete', False)}`.\n\n"
                  "## Method and rationale\n\nThe registered fixed-spawn 1v1 protocol specifies scratch self-play, a local four-frame "
                  "2x128 actor, separate2x256 critic, factorized categorical controls and TorchRL PPO/GAE. "
                  "Coefficients are engineering choices; the full rationale and decision rules are in "
                  "[the protocol](../../docs/duel-protocol.md) and [sources](../../docs/sources.md).\n\n"
                  f"Exact saved configuration: `{json.dumps(config, sort_keys=True)}`.\n\n"
                  f"## Measurements\n\nFinal-test score: `{score}`. Checkpoint SHA256: `{json.dumps(hashes)}`. "
                  f"Raw [configuration](data/{name}/config.json), all recorded metrics and evaluations are in the adjacent data folder. "
                  "`evidence.json` supplies raw-file hashes. Missing evaluation means unavailable, never a zero score.\n\n"
                  "## Interpretation and limitations\n\nLoss/entropy changes alone do not establish useful play. Calibration models are "
                  "timing fixtures, reward pilots are exploratory selection, and confirmation models are separate scratch seeds. "
                  "Do not infer a cause for failure without an intervention; fixed-layout behavior does not establish generalization. "
                  "Incomplete/failed runs remain included. No diagnostic checkpoint becomes a deployment model.\n")
        if (destination / "losses" / f"{name}.png").exists():
            report += f"\n## Loss graphs\n\n![Recorded loss diagnostics](losses/{name}.png)\n\n[Vector figure](losses/{name}.svg); [data and interpretation](losses/README.md).\n"
        (destination / f"{name}.md").write_text(report, encoding="utf-8")
    comparison = model_comparison(records)
    write_json(destination / "comparison.json", comparison)
    write_json(destination / "power.json", paired_power())
    write_json(destination / "evidence.json", evidence)
    rows = "\n".join(["| Run | Seed | Reward | Arena steps | Final score | Complete |", "|---|---|---|---|---|---|", *table])
    summary = ("# MiniStrike staged experiment evidence\n\n" + rows + "\n\n"
               "An arena step is one20Hz simultaneous transition; two player decisions occur per duel arena step. "
               "Round score is win1/draw0.5/loss0. An independent model uses its own seed, optimizer and self-play pool; "
               "many matches of one model do not increase the number of independently trained models.\n\n"
               f"Registered comparison: `{json.dumps(comparison)}`. Three-model intervals assume approximately normal independent "
               "model differences; that assumption cannot be validated with three models. Match bootstrap preserves seed/side pairing "
               "and treats repeated deterministic fixed-layout trajectories as one observation per side.\n\n"
               "Historical evidence: [deferred5v5 protocol](../../docs/protocol.md), [learning audit](../../docs/learning-audit.md), "
               "[ten hardware probes](../performance/summary.md). These used different tasks/settings and are not Stage A replications.\n")
    summary += "\n[All loss graphs and raw plotting evidence](losses/README.md).\n"
    (destination / "summary.md").write_text(summary, encoding="utf-8")
    paper = ("# MiniStrike: staged self-play in a fixed-spawn rifle duel\n\nLibor Burian\n\n"
             "## Abstract\n\nThis report records a registered12-hour investigation of scratch self-play in a small original shooter. "
             f"Confirmation analysis status is `{comparison['status']}`. Results are limited to the saved evidence; unfinished analysis "
             "does not support a learning-quality claim.\n\n"
             "## Context, motivation and related work\n\nThe deferred5v5 experiment produced zero-win scripted controls, motivating "
             "a simpler staged task. [Sample Factory](https://proceedings.mlr.press/v119/petrenko20a/petrenko20a.pdf) demonstrates "
             "Doom self-play using a different population/APPO system; its scale does not predict MiniStrike learning time. "
             "[Unity DodgeBall](https://unity.com/blog/engine-platform/ml-agents-plays-dodgeball) informs later team/entity work.\n\n"
             "## Method\n\nA14x14metre flat arena contains two players six metres apart. There are400ticks at20Hz. "
             "Each model starts from scratch and trains only against immutable learned snapshots. No human demonstrations, scripted "
             "training targets or imitation learning are used. Local observations contain four visibility-masked frames. "
             "[TorchRL0.14](https://docs.pytorch.org/rl/stable/tutorials/multiagent_ppo.html) supplies PPO losses, collection and GAE. "
             "R0/R1 use bounded effective-damage shaping; conditional R2 adds the "
             "[potential difference](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf). "
             "All hyperparameters, thresholds and branch-selection rules are registered engineering choices in the protocol.\n\n"
             "## Results and evidence\n\n" + rows + "\n\n" +
             f"Independent-model comparison: `{json.dumps(comparison)}`. Raw data paths and SHA256 identities are supplied in evidence.json. "
             "The complete individual reports contain configurations and checkpoint hashes.\n\n"
             "## Claims, uncertainty and discussion\n\n[Colas et al.](https://arxiv.org/abs/1806.08295) motivates explicit power sensitivity; "
             "power.json uses assumed model-difference SD0.05/0.10/0.20, effect0.20, n3 and a two-sided alpha0.05 noncentral-t model. "
             "[Agarwal et al.](https://arxiv.org/abs/2108.13264) motivates separating finite-run uncertainty from point estimates. "
             "Fixed-layout deterministic repeats do not supply independent trials. There are no significance-driven extensions.\n\n"
             "## Limitations and implications\n\nThree seeds cannot establish broad retraining reliability. A duel result is not evidence "
             "of map navigation, multi-player coordination or bomb objectives. Failed/incomplete runs and reward pilots are retained, "
             "not hidden or treated as confirmation replications. Published models remain experimental unless all engineering gates pass.\n\n"
             "## Conclusion\n\nInterpret only the measured checkpoint/seed-specific outcomes above. The next capability requires a "
             "separate frozen protocol and budget.\n\n"
             "## Appendices and reproducibility\n\nSee docs/experiment-roadmap.md, docs/duel-protocol.md, docs/sources.md, "
             "docs/evidence-index.md, reports/performance/summary.md and the historical learning-audit report for prior failures, "
             "hardware probes and recovery accounting. Regenerate with `uv run csrl report --input runs/duel-fixed-v1 "
             "--output reports/duel-fixed-v1`. No proprietary game assets or third-party source files were copied.\n")
    (destination / "paper.md").write_text(paper, encoding="utf-8")
    _render_pdf(paper, destination / "paper.pdf", None)  # Local: paper currently uses tables; no fake learning plot.
    return comparison

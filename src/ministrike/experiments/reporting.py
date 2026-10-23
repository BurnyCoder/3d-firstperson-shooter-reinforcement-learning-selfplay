"""Local: render measured experiment artifacts; global: reproducible, bounded scientific claims.

Statistics: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
Plots: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
PDFs: https://www.reportlab.com/docs/reportlab-userguide.pdf
"""

import json  # Local: read recorded observations; global: reports never invent measurements.
from pathlib import Path  # Local: isolate report paths; global: reproducible portable artifacts.
import re  # Local: small Markdown presentation adapter; global: one paper source for Markdown/PDF.
import shutil  # Local: copy raw evidence; global: preserve original JSON/JSONL bytes.
from xml.sax.saxutils import escape  # Local: escape PDF paragraph markup; global: safe artifact rendering.

import numpy as np  # Local: arithmetic on independent training-seed differences.
from scipy.stats import t  # Local: maintained Student-t quantiles; global: explicit interval assumptions.

from ministrike.records import timestamp, write_json  # Shared provenance and atomic JSON output.
from ministrike.experiments.loss_plots import parse_metrics, save_loss_gallery  # Local: reuse strict evidence parsing and per-run figures.

SOURCES = {  # Local: primary sources; global: separate external methodology from local evidence.
    "MAPPO": "https://arxiv.org/abs/2103.01955",
    "TorchRL multi-agent PPO": "https://docs.pytorch.org/rl/stable/tutorials/multiagent_ppo.html",
    "RL evaluation uncertainty": "https://arxiv.org/abs/2108.13264",
    "RL reproducibility": "https://arxiv.org/abs/1709.06560",
    "Student-t distribution": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html",
    "Noncentral-t power calculation": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nct.html",
}
EXPECTED_SEEDS = (111, 222, 333)  # Local: the three pre-specified independent initialization seeds.


def _read_json(path):
    """Local: preserve missingness; global: absent evaluations are pending, never zero performance."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def discover_runs(input_dir):
    """Local: read completed/partial run artifacts; global: include unsuccessful and calibration runs."""
    root = Path(input_dir).resolve()  # Resolve only the explicitly requested evidence tree.
    records = []  # One record per discovered training result.
    for path in sorted(root.rglob("result.json")):  # Walk measured result files, never model binaries.
        relative = path.parent.relative_to(root)  # Local: classify paths within only the requested evidence tree.
        if "publication" in relative.parts:  # Local: upload staging copies are not additional experiments.
            continue  # Local: retain original evidence once, excluding duplicated generated publication trees.
        result = _read_json(path)  # Invalid JSON fails visibly instead of inventing fallback values.
        if not isinstance(result, dict) or not {"condition", "seed", "frames"}.issubset(result):
            continue  # Ignore orchestration summaries that do not represent an actual training run.
        name = "__".join(relative.parts) or root.name  # Stable filename without directory traversal.
        canonical = path.parent.name == f"{result['condition']}_seed{result['seed']}"
        primary = canonical and result["seed"] in EXPECTED_SEEDS and "calibration" not in str(relative).lower()
        metrics, warnings = [], []  # Keep complete metric rows and disclose any interrupted last write.
        metrics_path = path.parent / "metrics.jsonl"  # Raw logger output is separate from summarized results.
        if metrics_path.exists():  # Missing metrics remain explicitly unavailable.
            metrics, warnings = parse_metrics(metrics_path.read_bytes())  # Local: share missingness/order checks with loss plots.
        records.append({"name": name, "path": path.parent, "result": result, "primary": primary,
                        "config": _read_json(path.parent / "config.json") or {}, "metrics": metrics,
                        "evaluation": _read_json(path.parent / "evaluation.json"),
                        "initial": _read_json(path.parent / "initial_evaluation.json"), "warnings": warnings})
    return records  # Records contain measured values and provenance, not interpolated missing scores.


def paired_training_comparison(records):
    """Local: pair algorithm outcomes by training seed; global: do not confuse matches with independent models."""
    groups = {}  # Duplicate candidate runs must be diagnosed rather than cherry-picked.
    for record in records:  # Only the frozen suite directories qualify for the primary endpoint.
        if record["primary"]:
            key = (record["result"]["condition"], record["result"]["seed"])
            groups.setdefault(key, []).append(record)  # Keep competing candidates visible.
    differences, exclusions = [], []  # Preserve why any planned comparison remains unavailable.
    for seed in EXPECTED_SEEDS:  # Never select favorable seeds after seeing results.
        candidates = [groups.get((condition, seed), []) for condition in ("selfplay", "fixed")]
        if any(len(items) != 1 for items in candidates):  # Missing or duplicated runs are not resolved by score.
            exclusions.append(f"Seed {seed}: require exactly one run per condition")
            continue
        left, right = candidates[0][0], candidates[1][0]  # Defined contrast is self-play minus fixed-opponent training.
        evaluations = [record["evaluation"] for record in (left, right)]
        if not all(record["result"].get("complete") for record in (left, right)) or not all(
                evaluation and evaluation.get("status") == "complete" for evaluation in evaluations):
            exclusions.append(f"Seed {seed}: training or evaluation is incomplete")
            continue
        if left["result"]["frames"] != right["result"]["frames"]:  # Unequal budgets cannot answer the planned contrast.
            exclusions.append(f"Seed {seed}: unequal environment-step budgets")
            continue
        controlled = ("num_envs", "total_frames", "rollout_steps", "epochs", "minibatch_size", "learning_rate",
                      "gamma", "gae_lambda", "clip_epsilon", "entropy_coefficient", "critic_coefficient")
        if any(left["config"].get(key) != right["config"].get(key) for key in controlled):
            exclusions.append(f"Seed {seed}: controlled training settings differ")  # Do not attribute parameter changes to self-play.
            continue
        if left["result"].get("source_revision") != right["result"].get("source_revision"):
            exclusions.append(f"Seed {seed}: source revisions differ")  # A code change is an uncontrolled intervention.
            continue
        if any(evaluations[0].get(key) != evaluations[1].get(key) for key in ("seed_base", "pairs", "num_envs")):
            exclusions.append(f"Seed {seed}: test tournament configurations differ")  # Preserve the planned common evaluation endpoint.
            continue
        scores = [evaluation.get("summary", {}).get("mean_score") for evaluation in evaluations]
        if any(score is None for score in scores):  # A missing summary is not a zero score.
            exclusions.append(f"Seed {seed}: endpoint score missing")
            continue
        differences.append({"seed": seed, "selfplay": scores[0], "fixed": scores[1],
                            "difference": scores[0] - scores[1], "frames": left["result"]["frames"]})
    if len({pair["frames"] for pair in differences}) > 1:  # The protocol freezes one common budget for all six runs.
        exclusions.append("Frame budgets differ across training-seed pairs")
    result = {"status": "complete" if len(differences) == len(EXPECTED_SEEDS) and not exclusions else "pending",
              "pairs": differences, "excluded": exclusions, "mean_difference": None, "ci95": None,
              "practical_threshold": 0.10, "analysis": "paired training-seed Student-t interval"}
    if result["status"] == "complete":  # Do not present an unfinished subset as the pre-specified endpoint.
        values = np.array([pair["difference"] for pair in differences], dtype=float)
        mean = float(values.mean())  # Each independently trained seed receives equal weight.
        margin = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
        result.update(mean_difference=mean, ci95=[mean - margin, mean + margin])  # Do not truncate wide intervals.
    return result  # Three-seed intervals are exploratory and depend strongly on normal-difference assumptions.


def _number(value, digits=3):
    """Local: format measured scalars; global: preserve missing data without misleading zero fills."""
    return "pending" if value is None else f"{value:.{digits}f}"


def _evaluation_text(evaluation):
    """Local: summarize one frozen checkpoint; global: retain its evaluation completeness."""
    if not evaluation:
        return "pending"  # No tournament artifact means no measured endpoint.
    summary = evaluation.get("summary", {})  # Partial evaluations may contain only completed rounds.
    interval = summary.get("score_ci95")  # Conditional-on-checkpoint interval is distinct from run-level uncertainty.
    suffix = f" [{_number(interval[0])}, {_number(interval[1])}]" if interval else " [CI pending]"
    return f"{_number(summary.get('mean_score'))}{suffix} ({evaluation.get('status', 'unknown')})"


def _results_table(records, compact=False):
    """Local: build auditable result rows; global: show every run rather than the best checkpoint."""
    headers = ["Run", "Frames", "Initial score [95% CI]", "Final score [95% CI]"]
    if not compact:
        headers.insert(2, "Elapsed seconds")  # Timing is measured training execution, not promised throughput.
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for record in records:  # Ordered discovery ensures a stable table across report regeneration.
        result = record["result"]  # This row comes directly from one training result file.
        row = [record["name"], str(result["frames"]), _evaluation_text(record["initial"]),
               _evaluation_text(record["evaluation"])]
        if not compact:
            row.insert(2, _number(result.get("elapsed_seconds"), 1))
        lines.append("| " + " | ".join(row) + " |")  # Markdown and PDF consume the same table data.
    return "\n".join(lines) if records else "No training results are available; all results are pending."


def _comparison_text(comparison):
    """Local: interpret only the registered contrast; global: never promise or manufacture significance."""
    if comparison["status"] != "complete":
        return (f"The pre-specified comparison is pending: {len(comparison['pairs'])}/3 matched training-seed "
                "pairs have complete, equal-budget results. Available pairs are retained in comparison.json; "
                "no subset is presented as the final hypothesis test.")
    interval = comparison["ci95"]  # This interval uses independent model pairs, not individual matches.
    return (f"Across all three planned training seeds, mean final-score difference (self-play minus fixed) is "
            f"{_number(comparison['mean_difference'])}, with Student-t 95% interval "
            f"[{_number(interval[0])}, {_number(interval[1])}]. The practical target was +0.100. "
            "The interval assumes independent seed-pair differences with an approximately normal distribution; "
            "three pairs cannot substantiate that distributional assumption. This is an exploratory pilot result.")


def _run_markdown(record):
    """Local: document one experiment's actual evidence; global: preserve failures and limitations."""
    result, evaluation, config = record["result"], record["evaluation"], record["config"]
    summary = evaluation.get("summary", {}) if evaluation else {}  # Missing evaluations stay pending.
    rows = evaluation.get("rounds", []) if evaluation else []  # Raw endpoints support non-score diagnostics.
    by_side = summary.get("by_side", {})  # Attack/defense asymmetry can conceal weak objective learning.
    by_baseline = summary.get("by_baseline", {})  # Report random/rush/guard separately from their aggregate.
    diagnostics = {key: float(np.mean([row[key] for row in rows if key in row]))
                   for key in ("damage", "shots", "plants", "defuses", "inactive_fraction")
                   if any(key in row for row in rows)}  # Objective event counts cover the whole round, not only learner events.
    status = "completed frozen frame budget" if result.get("complete") else f"stopped: {result.get('stop_reason', 'unknown')}"
    return (f"# Experiment: {record['name']}\n\n"
            f"Condition: `{result['condition']}`; seed: `{result['seed']}`; training {status}. "
            f"Primary-suite eligibility: `{record['primary']}`. Source revision: `{result.get('source_revision', 'unknown')}`.\n\n"
            "## Method and rationale\n\n"
            "MAPPO uses a shared actor with two 128-unit tanh layers and a separate team critic with two "
            "256-unit tanh layers. Each actor observes four local frames; six categorical controls represent "
            "movement, yaw, pitch, fire, reload, and interaction. A five-player critic combines that team's local "
            "observations. TorchRL supplies clipped PPO and generalized advantage estimation. Historical self-play "
            "samples immutable opponents at round boundaries; the control trains against a fixed scripted opponent. "
            f"See [MAPPO]({SOURCES['MAPPO']}) and [library implementation]({SOURCES['TorchRL multi-agent PPO']}).\n\n"
            f"Actual saved settings: gamma `{config.get('gamma', 'pending')}`, GAE lambda `{config.get('gae_lambda', 'pending')}`, "
            f"learning rate `{config.get('learning_rate', 'pending')}`, PPO epochs `{config.get('epochs', 'pending')}`, "
            f"rollout steps `{config.get('rollout_steps', 'pending')}`, arenas `{config.get('num_envs', 'pending')}`. "
            "The default gamma 0.999 retains approximately 0.30 of an outcome sixty seconds later at twenty ticks "
            "per second (0.999^1200). These small networks and fixed settings are practical starter choices, "
            "not empirically established optima. The curriculum allocates 15/15/70 percent of frame progress "
            "to combat/objective/standard reset scenarios; already-running rounds finish their existing scenario.\n\n"
            "## Measurements\n\n" + _results_table([record]) + "\n\n"
            f"Side-specific final scores (0 attackers, 1 defenders): `{json.dumps(by_side)}`. "
            f"Baseline-specific final scores: `{json.dumps(by_baseline)}`. "
            f"Mean recorded round diagnostics: `{json.dumps(diagnostics)}`.\n\n"
            "Plant and defuse counts cover either team's objective events; damage, shots, and inactivity "
            "refer to the evaluated learner team.\n\n"
            "## Learnings and evidence limits\n\n"
            "The report records what the run measured; it cannot infer undocumented debugging attempts, "
            "causes of a score change, or generalization beyond this map and these controls. Initial/final score "
            "differences are descriptive within one trained seed. Wins alone do not demonstrate planting or "
            "defusing competence; inspect the objective counts and raw rounds. Match confidence intervals "
            "condition on this checkpoint and do not quantify retraining uncertainty. No best-epoch selection "
            "is performed by this reporting code. Validation performance is not used for checkpoint selection.\n\n"
            f"Artifact warnings: `{json.dumps(record['warnings'])}`. "
            f"Raw evidence: [run data](data/{record['name']}/result.json), "
            f"[configuration](data/{record['name']}/config.json). Additional recorded evaluation and metric files "
            "are copied to the same directory when available.\n")


def _plot_learning(records, destination):
    """Local: plot raw update measurements; global: distinguish changing training opponents from fixed evaluation."""
    import matplotlib  # Import lazily so reading statistical summaries needs no plotting backend.
    matplotlib.use("Agg")  # Headless generation never opens a GUI window during training/reporting.
    from matplotlib import pyplot as plt  # Standard plotting library produces a standalone reusable figure.
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)  # Equal panels show complementary diagnostics.
    plotted = False  # Missing measurements receive an explicit placeholder instead of a fabricated curve.
    for record in records:  # Plot all discovered runs; no selection by score or visually favorable seed.
        metrics = record["metrics"]  # Original update order and frame counts are preserved.
        scored = [row for row in metrics if row.get("rounds", 0) > 0]  # Rates are undefined before any completed round.
        if scored:
            axes[0].plot([row["frames"] for row in scored],
                         [(row["wins"] + 0.5 * row.get("draws", 0)) / row["rounds"] for row in scored],
                         label=record["name"], linewidth=1)
            plotted = True  # The chart contains measured outcomes.
        entropy = [row for row in metrics if "entropy" in row]  # Do not interpolate missing diagnostic fields.
        if entropy:
            axes[1].plot([row["frames"] for row in entropy], [row["entropy"] for row in entropy],
                         label=record["name"], linewidth=1)
            plotted = True  # Entropy is a measured optimizer diagnostic.
    axes[0].set(title="Cumulative training round score", xlabel="Arena transitions", ylabel="Score")
    axes[0].set_ylim(0, 1)  # A score plot uses its full meaningful range.
    axes[1].set(title="Policy entropy", xlabel="Arena transitions", ylabel="Joint control entropy (nats)")
    for axis in axes:
        axis.grid(alpha=0.2)  # Light reference lines aid reading without suggesting precision.
        if axis.lines:
            axis.legend(fontsize=6, loc="best")  # Every displayed line retains its run identity.
        elif not plotted:
            axis.text(0.5, 0.5, "Measurements pending", ha="center", va="center", transform=axis.transAxes)
    figure.savefig(destination, dpi=180)  # Save a standalone publication asset from real numeric records.
    plt.close(figure)  # Release memory before subsequent report generations.


def _preserve_suite(input_dir, destination):
    """Local: preserve one exact manifest snapshot. Global: appendix claims and copied evidence use identical bytes."""
    source = Path(input_dir) / "suite.json"  # Local: do not guess among unrelated nested suite protocols.
    if not source.is_file():  # Local: individual smoke runs need not have a full experiment-suite manifest.
        return None  # Local: missing protocol evidence must be reported as pending.
    contents = source.read_bytes()  # Local: read once so concurrent manifest replacement cannot mismatch prose and copy.
    suite = json.loads(contents)  # Local: malformed source evidence fails visibly.
    if not isinstance(suite, dict):  # Local: a suite protocol must expose named recorded fields.
        raise ValueError("suite.json must contain a JSON object")  # Local: reject invalid evidence instead of fabricating defaults.
    copied = Path(destination) / "data" / "suite.json"  # Local: stable public link contains only the requested manifest.
    if source.resolve() != copied.resolve():  # Local: generating alongside existing report data must not overwrite itself.
        copied.parent.mkdir(parents=True, exist_ok=True)  # Local: create only the report's existing data tree.
        temporary = copied.with_suffix(".json.tmp")  # Local: publish complete original bytes atomically on the same filesystem.
        temporary.write_bytes(contents)  # Local: preserve whitespace and all recorded failures without reserialization.
        temporary.replace(copied)  # Local: readers see one complete recorded protocol snapshot.
    return suite  # Local: paper content derives from precisely the preserved bytes.


def _suite_appendix(suite):
    """Local: describe recorded budget and power assumptions. Global: missing decisions remain pending, never inferred."""
    if suite is None:  # Local: a smoke run does not establish the full suite's timing or analysis settings.
        return ("Suite protocol, deadline, selected device, calibrated frame budget, and power sensitivity are pending: "
                "no suite.json was present in the requested input directory.")
    paragraphs = [  # Local: compact paragraphs retain exact provenance without repeating every per-run configuration.
        f"Recorded suite protocol: [suite.json](data/suite.json). Status: `{suite.get('status', 'pending')}`; "
        f"started UTC: `{suite.get('started_utc', 'pending')}`; original deadline UTC: `{suite.get('deadline_utc', 'pending')}`. "
        f"Total wall-clock cap: {_number(suite.get('max_hours'), 2)} hours. Selected optimizer device: "
        f"`{suite.get('device', 'pending')}`; parallel arenas: `{suite.get('num_envs', 'pending')}`. "
        f"Calibrated frozen budget per training invocation: `{suite.get('total_frames', 'pending')}` arena transitions "
        f"and {_number(suite.get('per_run_seconds'), 1)} seconds. These are recorded limits, not claims of completed work."
    ]  # Local: actual completion remains reported independently for every training invocation.
    calibration = suite.get("calibration", [])  # Local: include all stored probes rather than choosing favorable throughput examples.
    if calibration:  # Local: throughput exists only when the suite persisted calibration measurements.
        rows = ["| Calibration condition | Device | Measured frames | Arena transitions/s |", "| --- | --- | --- | --- |"]
        for row in calibration:  # Local: each stored calibration invocation remains independently inspectable.
            rows.append(f"| {row.get('condition', 'pending')} | {row.get('device', 'pending')} | "
                        f"{row.get('frames', 'pending')} | {_number(row.get('frames_per_second'), 2)} |")
        paragraphs.append("Recorded calibration throughput:\n\n" + "\n".join(rows))  # Local: label measurements separately from allocated budgets.
    else:
        paragraphs.append("Calibration throughput measurements are pending.")  # Local: an absent probe is not a zero-throughput device.
    power = suite.get("power_sensitivity", [])  # Local: report pre-recorded calculations without rerunning them after seeing outcomes.
    if power:  # Local: show the actual assumptions attached to this suite.
        paragraphs.append("Recorded power sensitivity uses an independent two-group noncentral-t calculation; "
                          "it is an assumption-based planning calculation, not measured power or a power estimate "
                          "for the final three-seed paired analysis. "
                          f"[Distribution implementation]({SOURCES['Noncentral-t power calculation']}).")
        rows = ["| Assumed run SD | Effect | Alpha | Target power | Seeds/condition | Calculated power |",
                "| --- | --- | --- | --- | --- | --- |"]  # Local: expose every assumption behind sample-size requirements.
        for row in power:  # Local: retain all recorded noise assumptions without selecting a convenient row.
            rows.append(f"| {_number(row.get('assumed_run_sd'), 2)} | {_number(row.get('effect'), 2)} | "
                        f"{_number(row.get('alpha'), 2)} | {_number(row.get('target_power'), 2)} | "
                        f"{row.get('seeds_per_condition', 'pending')} | {_number(row.get('achieved_power'), 3)} |")
        paragraphs.append("\n".join(rows))  # Local: Markdown and PDF share identical recorded values.
    else:
        paragraphs.append("Power sensitivity assumptions and calculated seed requirements are pending.")
    if suite.get("error") is not None:  # Local: preserve orchestration failures in addition to per-run outcomes.
        paragraphs.append(f"Recorded suite failure: `{suite['error']}`.")  # Local: do not convert a failed suite into a success narrative.
    return "\n\n".join(paragraphs)  # Local: appendix uses ordinary paper blocks and the existing renderer.


def _paper_sections(records, comparison, suite=None):
    """Local: compose a sourced paper from measured evidence; global: avoid unsupported causal/general claims."""
    primary = [record for record in records if record["primary"]]  # Keep the headline table tied to the registered suite.
    complete = sum(record["result"].get("complete", False) for record in primary)  # Count actual completed budgets.
    return [
        ("Abstract", f"We implement an original 3D five-versus-five tactical shooter and compare historical "
         f"self-play with fixed-opponent MAPPO training. Recorded training invocations: {len(records)}; "
         f"completed primary-suite frame budgets: {complete}. " + _comparison_text(comparison)),
        ("Introduction and motivation", "The practical question is whether historical self-play improves "
         "a small local-observation policy's round score against frozen controls relative to equally budgeted "
         "fixed-opponent training. The pre-specified practically meaningful difference is 0.10. This experiment "
         "is specific to MiniStrike, not commercial Counter-Strike or human-level tactical play."),
        ("Background and related work", f"MAPPO combines decentralized policy execution with a centralized "
         f"critic during training [MAPPO]({SOURCES['MAPPO']}). We reuse "
         f"[TorchRL's multi-agent PPO machinery]({SOURCES['TorchRL multi-agent PPO']}). "
         f"Reporting separates seed variability from match variability, following concerns discussed in "
         f"[RL evaluation uncertainty]({SOURCES['RL evaluation uncertainty']}) and "
         f"[RL reproducibility]({SOURCES['RL reproducibility']})."),
        ("Methods", "One NumPy simulation defines movement, visibility, hits, simultaneous damage, finite "
         "ammunition, planting, defusing, timers, and terminal outcomes for both rendered play and learning. "
         "The actor is a shared 2x128 tanh MLP over four 96-feature observation frames (384 inputs); the "
         "central critic is a 2x256 tanh MLP over five players' combined observations. Six independent "
         "categorical heads form one joint action per player. Each PPO likelihood ratio exponentiates "
         "the difference between summed new and old control log probabilities, "
         "not player log probabilities. GAE precedes death masking; actor and entropy losses exclude "
         "post-death actions, while critic learning retains terminal team outcomes. Default discount "
         "gamma is 0.999 and GAE lambda is 0.95. Exact hyperparameters, frame counts, code revisions, "
         "and conditions are stored per run. Terminal team rewards are +1/-1 for win/loss and zero for draws. "
         "Zero-sum shaping adds at most 0.1 for effective damage to all opponents and 0.1 for planting; "
         "health is not replenished within rounds. Self-play chooses the latest frozen policy with probability "
         "0.5 and historical policies otherwise; snapshots are selected only at round boundaries. "
         "The control uses fixed scripted opposition. Both conditions use the same frame-based combat, "
         "objective, and standard-round curriculum. The checkpoint is the final saved endpoint, with budget completeness recorded, "
         "not the best measured evaluation. Test tournaments pair scenarios with sides swapped against "
         "random, rush, and guard baselines. Score is 1 for a win, 0.5 for a draw, and 0 for a loss. "
         "The design calls for disjoint training and final-test seed ranges; recorded tournament metadata "
         "preserves the actual range. Seed-pair cluster bootstrap intervals condition on each checkpoint. "
         "The between-condition comparison uses all three paired training-seed differences and a "
         "Student-t interval with two degrees of freedom; it is not an interval over individual matches."),
        ("Results", _results_table(primary or records, compact=True) + "\n\n" + _comparison_text(comparison)),
        ("Figures", "![Measured training curves](learning_curves.png)\n\n"
         "Figure 1. Raw cumulative training score and joint-action entropy against arena transitions. "
         "Training opponents and curriculum change over time, so these curves are optimization diagnostics, "
         "not held-out evaluation learning curves or causal evidence of superiority. No smoothing is applied."),
        ("Discussion and implications", "The fixed-opponent comparison is an intervention on the training "
         "opponent distribution while holding the registered architecture, objective, curriculum, and budgets "
         "constant. It can support a comparison only for those matched completed runs. More wins without "
         "objective completions do not establish tactical competence. This study does not isolate which "
         "historical opponent, reward component, observation feature, or curriculum stage causes an effect; "
         "those claims require further pre-registered ablations and counterfactual interventions."),
        ("Limitations", "Three independent training seeds per condition provide weak information about "
         "seed sensitivity. Normality of seed-pair differences is unverified, and a zero empirical variance "
         "can yield a zero-width t interval without proving zero population uncertainty. Bootstrap match "
         "intervals do not include retraining variability. This release evaluates one map, one weapon, "
         "ground-level movement, and limited scripted controls. Policies consume structured observations "
         "rather than rendered pixels. Compute limits, incomplete runs, negative outcomes, and missing "
         "artifacts are reported explicitly. No significant result, human-level skill, causal mechanism, "
         "or transfer to another game is presumed."),
        ("Conclusion", _comparison_text(comparison) + " The software artifact establishes an executable "
         "game/training/evaluation path. Claims about learned competence must follow the recorded held-out "
         "scores and objective diagnostics, with the limitations above."),
        ("Reproducibility appendix", "The data directory preserves raw result, configuration, evaluation, "
         "and metric artifacts without copying model weights. Individual experiment Markdown files supply "
         "side/baseline breakdowns and raw-artifact links. comparison.json records every planned seed-pair "
         "difference and exclusion. The source repository and model cards provide installation and "
         "checkpoint usage instructions. Accuracy, fill rate, and generic completion rate are N/A: "
         "this is round-based reinforcement learning without labeled classification or fill-in targets.\n\n" + _suite_appendix(suite)),
        ("Sources", "\n".join(f"- [{name}]({url})" for name, url in SOURCES.items())),
    ]  # One source of paper content keeps Markdown and PDF claims synchronized.


def _render_pdf(markdown, destination, figure_path):
    """Local: render the same paper text with ReportLab; global: avoid divergent Markdown/PDF claims."""
    from reportlab.lib import colors  # Small publication palette for readable tables.
    from reportlab.lib.pagesizes import A4  # Explicit portable print geometry.
    from reportlab.lib.styles import getSampleStyleSheet  # Reuse tested typography defaults.
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    styles = getSampleStyleSheet()  # Standard styles include title, body, and section headings.
    styles["BodyText"].fontSize, styles["BodyText"].leading = 9, 11.8  # Local: retain readable text with slightly tighter leading.
    styles["BodyText"].spaceAfter = 5  # Local: compact paragraph spacing avoids a nearly empty trailing sources page.
    styles["Heading2"].keepWithNext = True  # Local: section headings remain with their following image, table, or paragraph.
    width = A4[0] - 88  # Match document left/right margins exactly.
    story, table_rows, paragraph = [], [], []  # Maintain simple Markdown block state.

    def rich(text):
        """Local: escape text and retain source links; global: safe readable PDF provenance."""
        escaped = escape(text.replace("`", ""))  # Literal code markers do not belong in rendered prose.
        return re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<link href="\2" color="#235789">\1</link>', escaped)

    def flush():
        """Local: finish pending blocks; global: preserve source order and avoid splitting paragraphs."""
        if paragraph:
            story.append(Paragraph(rich(" ".join(paragraph)), styles["BodyText"]))  # Wrap body text to the content width.
            paragraph.clear()  # Consume exactly this paragraph once.
        if table_rows:
            cells = [[Paragraph(rich(cell), styles["BodyText"]) for cell in row] for row in table_rows]
            table = Table(cells, colWidths=[width / len(cells[0])] * len(cells[0]), repeatRows=1)
            table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF0F5")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"), ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey),
                ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
            story.extend([table, Spacer(1, 10)])  # Tables can split across pages with repeated headers.
            table_rows.clear()  # Keep subsequent table blocks independent.

    for line in markdown.splitlines():  # This parser only handles the limited Markdown emitted above.
        if line.startswith("|"):
            if paragraph:
                flush()  # A preceding prose paragraph must appear before the table.
            if not re.fullmatch(r"[|\- :]+", line):  # Markdown alignment rows are not data.
                table_rows.append([cell.strip() for cell in line.strip("|").split("|")])
        elif line.startswith("#"):
            flush()  # Headings begin a new layout block.
            level = len(line) - len(line.lstrip("#"))  # Generated paper uses title and section headings only.
            story.append(Paragraph(rich(line.lstrip("# ")), styles["Title" if level == 1 else "Heading2"]))
        elif line.startswith("!["):
            flush()  # The figure sits at its explicit source position.
            story.append(Image(str(figure_path), width=width, height=width * 4 / 11))  # Match the saved figure aspect ratio.
        elif not line.strip():
            flush()  # Blank lines delimit generated paragraphs/tables.
        else:
            if table_rows:
                flush()  # Body text following a table must not precede it.
            paragraph.append(line.removeprefix("- ") + ("; " if line.startswith("- ") else ""))  # Consecutive source links form one compact paragraph, avoiding a nearly empty overflow page.
    flush()  # Emit the final body paragraph or table.
    document = SimpleDocTemplate(str(destination), pagesize=A4, rightMargin=44, leftMargin=44,
                                 topMargin=44, bottomMargin=44, title="MiniStrike self-play experiment",
                                 author="Libor Burian")
    document.build(story)  # Rendering is invoked only by the explicit report command.


def create_reports(input_dir, output_dir):
    """Local: discover, summarize, preserve, and render; global: one auditable reporting command."""
    destination = Path(output_dir)  # Caller decides where generated public reports belong.
    destination.mkdir(parents=True, exist_ok=True)  # Create only the requested artifact tree.
    save_loss_gallery(input_dir, destination / "losses")  # Local: include logged losses even for deferred runs without a result.json.
    records = discover_runs(input_dir)  # Ground every empirical statement in recorded input artifacts.
    suite = _preserve_suite(input_dir, destination)  # Local: copied protocol and paper appendix share one exact manifest snapshot.
    comparison = paired_training_comparison(records)  # Freeze inclusion rules before inspecting their scores.
    for record in records:  # Every experiment receives a separate report and raw-evidence directory.
        data_dir = destination / "data" / record["name"]  # Stable relative links survive publication.
        data_dir.mkdir(parents=True, exist_ok=True)  # Keep raw files grouped by source experiment.
        for filename in ("result.json", "config.json", "evaluation.json", "initial_evaluation.json", "metrics.jsonl"):
            source = record["path"] / filename  # Copy only documented measurements/configuration, never credentials.
            if source.exists() and source.resolve() != (data_dir / filename).resolve():
                shutil.copy2(source, data_dir / filename)  # Preserve full original records without truncation.
        (destination / f"{record['name']}.md").write_text(_run_markdown(record), encoding="utf-8")
    write_json(destination / "comparison.json", comparison)  # Preserve precise numbers behind prose rounding.
    figure = destination / "learning_curves.png"  # Standalone plot is usable in reports and presentations.
    _plot_learning(records, figure)  # Produce visual evidence from raw training updates.
    glossary = ("MAPPO is multi-agent proximal policy optimization: local actors learn with a team critic. "
                "Self-play trains against frozen versions of the learner; fixed-opponent training holds the "
                "opposition rule constant. Score assigns win/draw/loss values 1/0.5/0. A 95% confidence interval "
                "describes an estimation procedure under stated assumptions, not a 95% probability that a fixed "
                "unknown value lies inside this particular interval. Match intervals condition on one model; "
                "the paired-seed interval concerns variation across independent training runs. N/A means a "
                "metric is not applicable; pending means evidence has not been recorded.")
    summary = ("# MiniStrike experiment summary\n\n" + _comparison_text(comparison) + "\n\n"
               "## All recorded experiments\n\n" + _results_table(records) + "\n\n"
               "## Networks and metrics\n\n"
               "| Policy | Architecture | Training/objective | Data/curriculum | Domain | Accuracy/fill/completion |\n"
               "| --- | --- | --- | --- | --- | --- |\n"
               "| Learned actor | Shared 2x128 tanh MLP; 384 local-history inputs; 6 categorical heads | MAPPO clipped "
               "policy loss and entropy | Online simulated rounds; combat/objective/standard starts | One MiniStrike map | N/A |\n"
               "| Team critic | 2x256 tanh MLP; 1920 combined inputs; 5 values | Team return regression and GAE | Same online rounds | Training only | N/A |\n"
               "| Random/rush/guard controls | No trained network | Fixed rules; no optimizer | No training data | Same map and observations | N/A |\n\n"
               "Classification accuracy and fill/completion metrics are N/A because this task has no labeled "
               "classification or text-completion targets. Round scores, objective events, and damage are reported instead.\n\n"
               "## Terms and interpretation\n\n" + glossary + "\n\n"
               "![Measured training diagnostics](learning_curves.png)\n\n"
               "Training opponents/curriculum change; cumulative training scores are not held-out learning curves. "
               "Individual reports contain side/baseline breakdowns and raw evidence. "
               "[Paper](paper.md) documents methods and limitations; [comparison data](comparison.json) retains exclusions.\n")
    (destination / "summary.md").write_text(summary, encoding="utf-8")  # Canonical cross-experiment overview.
    sections = _paper_sections(records, comparison, suite)  # Shared scientific content includes recorded suite budget and power assumptions.
    paper = "# MiniStrike: a reproducible self-play pilot\n\nLibor Burian\n\n" + "\n\n".join(
        f"## {heading}\n\n{body}" for heading, body in sections) + "\n"
    (destination / "paper.md").write_text(paper, encoding="utf-8")  # Editable sourced paper text.
    _render_pdf(paper, destination / "paper.pdf", figure)  # Root's report workflow performs visual PDF verification.
    manifest = {"generated_at_utc": timestamp(), "input_dir": str(Path(input_dir).resolve()),
                "experiments": len(records), "comparison_status": comparison["status"],
                "summary": str(destination / "summary.md"), "paper": str(destination / "paper.pdf"),
                "figure": str(figure)}  # Expose actionable output paths to CLI and publishing code.
    write_json(destination / "manifest.json", manifest)  # Record the generation inputs and output inventory.
    return manifest  # Caller can display/open the generated report artifacts.

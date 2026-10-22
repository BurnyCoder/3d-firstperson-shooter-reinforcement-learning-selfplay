"""Local: pre-specified power and score summaries. Global: distinguish match precision from training variance.

Sources: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nct.html and
https://arxiv.org/abs/1806.08295 . Power calculations are conditional sensitivity analyses, not measured guarantees.
"""

from collections import defaultdict  # Preserve each side-swapped seed pair as one resampling unit.

import numpy as np  # Reproducible bootstrap indices and arithmetic summaries.
from scipy.stats import nct, t  # Reuse noncentral t distributions rather than approximate significance claims.


def power_sensitivity(effect: float = 0.10, alpha: float = 0.05, target: float = 0.80) -> list[dict]:
    """Local: solve equal-size independent-run t-test power. Global: expose assumptions before observing outcomes."""
    rows = []  # Retain every planned standard-deviation assumption.
    for deviation in (0.05, 0.10, 0.20):  # These are assumptions, not variance estimates from the game.
        for count in range(2, 1001):  # Search a finite, documented range of seeds per condition.
            degrees = 2 * count - 2  # Two independent groups consume two fitted means.
            threshold = t.ppf(1 - alpha / 2, degrees)  # Two-sided test controls both tails.
            noncentrality = effect / deviation * np.sqrt(count / 2)  # Standardized independent-group mean difference.
            power = nct.sf(threshold, degrees, noncentrality) + nct.cdf(-threshold, degrees, noncentrality)  # Integrate rejection probability.
            if power >= target:  # Choose the first sample count meeting the declared target.
                rows.append({"assumed_run_sd": deviation, "effect": effect, "alpha": alpha, "target_power": target, "seeds_per_condition": count, "achieved_power": float(power)})  # Keep all inferential assumptions with the result.
                break  # Larger sample counts are unnecessary for this sensitivity row.
    return rows  # Three training seeds remain a pilot when the required count exceeds three.


def summarize_rounds(rows: list[dict], seed: int = 918273) -> dict:
    """Local: bootstrap paired seed-cluster scores. Global: this interval conditions on one fixed checkpoint."""
    if not rows:  # Empty or interrupted evaluation must not masquerade as a zero score.
        return {"rounds": 0, "mean_score": None, "score_ci95": None, "wins": 0, "draws": 0, "losses": 0}
    grouped = defaultdict(list)  # Preserve correlated side-swapped games within each pair.
    for row in rows:  # Each record represents one completed round.
        grouped[row["pair"]].append(row["score"])  # Include all baseline/side outcomes for the same seeded world.
    paired_scores = np.array([np.mean(values) for values in grouped.values()])  # Average before resampling, never split paired games.
    rng = np.random.default_rng(seed)  # CI generation cannot consume any training/evaluation RNG state.
    bootstrap = paired_scores[rng.integers(len(paired_scores), size=(2000, len(paired_scores)))].mean(axis=1)  # Cluster bootstrap conditions on the trained model.
    scores = np.array([row["score"] for row in rows])  # Round-level counts remain separately visible.
    return {  # This result deliberately contains no algorithm-level p-value.
        "rounds": len(rows), "seed_pairs": len(paired_scores), "mean_score": float(scores.mean()),
        "score_ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "wins": int(np.sum(scores == 1)), "draws": int(np.sum(scores == 0.5)), "losses": int(np.sum(scores == 0)),
        "interval_unit": "seed pair, conditional on this checkpoint; not independent training runs",
        "by_side": {str(side): float(np.mean([row["score"] for row in rows if row["side"] == side])) for side in (0, 1) if any(row["side"] == side for row in rows)},
        "by_baseline": {name: float(np.mean([row["score"] for row in rows if row["baseline"] == name])) for name in sorted({row["baseline"] for row in rows})},
    }

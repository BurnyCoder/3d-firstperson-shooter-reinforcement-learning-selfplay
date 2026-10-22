"""Local: analysis invariants. Global: prevent misleading uncertainty and budget calculations."""

from ministrike.experiments.analysis import power_sensitivity, summarize_rounds  # Test public reporting functions.


def test_power_requires_more_seeds_for_more_noise():
    """A larger assumed run-level variance cannot require fewer independent runs."""
    rows = power_sensitivity()  # Fix the practical effect, alpha, and desired power in the implementation.
    counts = [row["seeds_per_condition"] for row in rows]  # Read the computed required sample sizes.
    assert counts == sorted(counts) and counts[0] >= 2  # No reversed power curve or impossible single-run t test.


def test_scores_keep_draws_separate_from_wins():
    """A draw contributes half a score, never half a Bernoulli win."""
    rows = [{"pair": str(i), "baseline": "random", "side": 0, "winner": winner, "score": score} for i, (winner, score) in enumerate([(0, 1), (1, 0), (-1, 0.5)])]  # Exact mixed outcomes.
    result = summarize_rounds(rows)  # Summarize measured rounds without changing the primary endpoint.
    assert result["mean_score"] == 0.5  # Arithmetic score is the declared primary metric.
    assert result["wins"] == result["draws"] == result["losses"] == 1  # Preserve all three outcome categories.

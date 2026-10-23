"""Local: convert cumulative telemetry to interval rates; global: preserve units, missingness and signed rewards.

Original analysis glue uses NumPy first differences and guarded division:
https://numpy.org/doc/stable/reference/generated/numpy.diff.html
https://numpy.org/doc/stable/reference/generated/numpy.divide.html
Counter semantics come from learning/environment.py; plotting choices are documented in docs/experiment-plots.md.
"""

import numpy as np  # Local: signed floating differences avoid unsigned counter wraparound.


def _increments(rows, key, behavior=False, sparse=False, signed=False):
    """Local: difference adjacent counters; global: a missing prefix or reset cannot masquerade as progress."""
    values = []  # Local: preserve one position per logged update.
    for row in rows:
        source = row.get("behavior", {}) if behavior else row  # Local: event counters live inside the behavior payload.
        default = 0 if sparse and "agent_decisions" in source else np.nan  # Local: only active event logging permits implicit zeros.
        values.append(source.get(key, default))  # Local: required missing fields remain unknown.
    origin = 0 if rows and rows[0].get("updates") == 1 else np.nan  # Local: no artificial zero before a truncated log prefix.
    deltas = np.diff(np.asarray(values, dtype=float), prepend=origin)  # Local: the interval ends at the row's arena-step count.
    if not signed and np.any(deltas < 0):
        raise ValueError(f"Counter {key} decreased; audit canonical logs before plotting")  # Local: signed reward alone may decrease.
    return np.where(np.isfinite(deltas), deltas, np.nan)  # Local: nonfinite evidence creates gaps.


def _ratio(numerator, denominator):
    """Local: divide only observed positive exposure; global: zero rounds or shots do not imply a zero rate."""
    return np.divide(numerator, denominator, out=np.full_like(denominator, np.nan),
                     where=np.isfinite(denominator) & (denominator > 0))  # Local: undefined fractions remain NaN.


def interval_metrics(rows):
    """Local: derive reward, combat, behavior and outcome series; global: this analysis never changes a reward or policy."""
    steps = _increments(rows, "frames")  # Local: normalize by independent arena transitions, not both players' actions.
    decisions = _increments(rows, "agent_decisions", behavior=True)  # Local: occupancy is measured per learner decision.
    rounds = _increments(rows, "rounds")  # Local: only completed rounds contribute to outcome fractions.
    series = {"step_rate": _ratio(steps, _increments(rows, "elapsed_seconds"))}  # Local: trainer timing includes its own update overhead.
    for key in ("outcome", "damage", "potential"):
        series[f"reward_{key}"] = _ratio(_increments(rows, f"reward_{key}", behavior=True, signed=True), steps)
    series["reward_total"] = sum(series[f"reward_{key}"] for key in ("outcome", "damage", "potential"))  # Local: exact additive reward decomposition.
    for key in ("shots", "damage", "kills"):
        series[f"{key}_rate"] = 1000 * _ratio(_increments(rows, key, behavior=True, sparse=True), steps)  # Local: readable counts per 1000 steps.
    series["accuracy"] = _ratio(_increments(rows, "hits", behavior=True, sparse=True),
                                _increments(rows, "shots", behavior=True, sparse=True))  # Local: hits divided by actual shots, not fire requests.
    for label, counter in (("reload", "reload_ticks"), ("aim_saturated", "aim_saturated_ticks"), ("inactive", "inactive_ticks")):
        series[f"{label}_fraction"] = _ratio(_increments(rows, counter, behavior=True), decisions)  # Local: preserve the logger's occupancy definition.
    series["timeout_fraction"] = _ratio(_increments(rows, "timeout", behavior=True, sparse=True), rounds)
    wins, draws = _increments(rows, "wins"), _increments(rows, "draws")  # Local: all three outcomes use identical completed-round exposure.
    for label, counts in (("win", wins), ("draw", draws), ("loss", rounds - wins - draws)):
        if np.any(counts < 0):
            raise ValueError("Outcome counts exceed completed rounds")  # Local: fail visibly rather than clip an impossible proportion.
        series[f"{label}_fraction"] = _ratio(counts, rounds)
    series["round_score"] = _ratio(wins + .5 * draws, rounds)  # Local: this is changing-opponent self-play score, not benchmark performance.
    return [dict(row, **{key: float(values[index]) for key, values in series.items()}) for index, row in enumerate(rows)]

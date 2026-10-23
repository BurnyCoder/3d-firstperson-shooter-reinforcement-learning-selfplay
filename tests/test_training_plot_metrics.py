"""Local: check graph arithmetic against hand-computed counters; global: visual evidence cannot invent learning."""

import numpy as np  # Local: missing intervals must stay gaps, including undefined ratios.
import pytest  # Local: reject corrupted counters instead of drawing misleading curves.

from ministrike.experiments.training_plot_metrics import interval_metrics  # Local: exercise numeric preparation without graphics.


def test_interval_rewards_and_event_rates_use_counter_differences():
    """Local: two unequal intervals have known rewards; global: cumulative totals are not learning curves."""
    rows = [
        {"frames": 100, "updates": 1, "elapsed_seconds": 2, "rounds": 2, "wins": 1, "draws": 1,
         "behavior": {"agent_decisions": 100, "reward_outcome": 1, "reward_damage": .5,
                      "reward_potential": -.1, "shots": 10, "hits": 2, "damage": 50,
                      "reload_ticks": 30, "aim_saturated_ticks": 0, "inactive_ticks": 2}},
        {"frames": 300, "updates": 2, "elapsed_seconds": 6, "rounds": 6, "wins": 2, "draws": 2,
         "behavior": {"agent_decisions": 300, "reward_outcome": 0, "reward_damage": 1,
                      "reward_potential": -.3, "shots": 30, "hits": 3, "damage": 75,
                      "reload_ticks": 50, "aim_saturated_ticks": 10, "inactive_ticks": 4, "kills": 1, "timeout": 2}},
    ]
    first, second = interval_metrics(rows)  # Local: use a zero origin only for the recorded first update.
    assert first["reward_total"] == pytest.approx(.014)
    assert second["reward_total"] == pytest.approx(-.0035)  # Local: (-1 + .5 - .2) / 200.
    assert second["reward_total"] == pytest.approx(sum(second[f"reward_{key}"] for key in ("outcome", "damage", "potential")))
    assert second["accuracy"] == .05 and second["damage_rate"] == 125  # Local: one hit / 20 shots; 25 damage / 200 steps * 1000.
    assert second["reload_fraction"] == .1 and second["timeout_fraction"] == .5
    assert second["win_fraction"] == .25 and second["draw_fraction"] == .25 and second["loss_fraction"] == .5
    assert second["round_score"] == .375 and second["step_rate"] == 50
    assert first["kills_rate"] == 0  # Local: the active logger omits event counters before their first event.


def test_missing_telemetry_empty_denominators_and_late_log_start_stay_unknown():
    """Local: distinguish unavailable observations from zero events; global: missing evidence must not imply success."""
    no_behavior = interval_metrics([{"frames": 100, "updates": 1, "rounds": 0, "wins": 0, "draws": 0}])[0]
    assert np.isnan(no_behavior["accuracy"]) and np.isnan(no_behavior["reward_total"]) and np.isnan(no_behavior["round_score"])
    empty = interval_metrics([{"frames": 100, "updates": 1, "behavior": {"agent_decisions": 100}}])[0]
    assert np.isnan(empty["accuracy"]) and empty["shots_rate"] == 0 and np.isnan(empty["reward_total"])
    late = interval_metrics([{"frames": 100, "updates": 9, "behavior": {"agent_decisions": 100, "shots": 10}}])[0]
    assert np.isnan(late["shots_rate"])  # Local: the omitted prefix is not an observed first interval.


def test_nonnegative_counter_reset_is_an_error_but_signed_reward_can_fall():
    """Local: recovery must preserve cumulative counts; global: never hide duplicated or reset measurements."""
    rows = [{"frames": 100, "updates": 1, "behavior": {"agent_decisions": 100, "shots": 10}},
            {"frames": 200, "updates": 2, "behavior": {"agent_decisions": 200, "shots": 1}}]
    with pytest.raises(ValueError, match="shots.*decreased"):
        interval_metrics(rows)

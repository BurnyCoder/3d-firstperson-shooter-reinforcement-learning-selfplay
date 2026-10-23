"""Local: paired action-mode contrasts; global: conditional match uncertainty never becomes independent-model evidence.

Use SciPy's percentile bootstrap after forming complete seed blocks:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html .
The prospective analysis is in docs/amendments/2026-10-23-action-mode-diagnostic.md.
"""

import numpy as np  # Local: keep the two spawn ends and common-seed reward variants together.
from scipy.stats import bootstrap  # Local: reuse the maintained resampling implementation.


def _score_blocks(cell, deterministic):
    """Local: form one side-averaged score per stochastic seed; global: deterministic references contribute no sampling noise."""
    identity, rows = cell["identity"], cell["rounds"]
    keyed = {(row["side"], row["seed"]): row["score"] for row in rows}
    if len(keyed) != len(rows):
        raise ValueError("Duplicate action-mode evaluation records")  # Local: duplicated rows are not additional evidence.
    root, pairs = identity["seed_base"], identity["pairs"]
    seeds = [root] if deterministic else list(range(root, root + pairs))
    if set(keyed) != {(side, seed) for side in (0, 1) for seed in seeds}:
        raise ValueError("Action-mode evaluation has missing or unexpected paired worlds")
    scores = np.array([(keyed[0, seed] + keyed[1, seed]) / 2 for seed in seeds])  # Local: preserve side pairing before any resampling.
    return np.repeat(scores, pairs) if deterministic else scores  # Local: broadcast a known constant reference; do not fabricate match rows.


def _effect(differences):
    """Local: summarize paired differences with a library bootstrap; global: intervals condition on these exact frozen policies."""
    result = bootstrap((differences,), np.mean, n_resamples=5000, method="percentile",
                       rng=np.random.default_rng(5701))  # Local: registered analysis RNG is independent of gameplay seeds.
    return {"delta": float(differences.mean()), "ci95": [float(result.confidence_interval.low), float(result.confidence_interval.high)]}


def compare_modes(results):
    """Local: compute all registered contrasts; global: no selection, retraining or gate override happens during analysis."""
    models, primary = {}, []  # Local: reward variants share training seed 5201, so never treat their count as independent runs.
    for reward, cells in sorted(results.items()):
        if len({(cell["identity"]["seed_base"], cell["identity"]["pairs"]) for cell in cells.values()}) != 1:
            raise ValueError("Action-mode cells must share paired seeds")
        scores = {mode: _score_blocks(cells[mode], mode == "DD") for mode in ("DD", "DS", "SD", "SS")}
        delta = scores["SD"] - scores["DD"]  # Local: first letter is learner mode; D means argmax, S categorical sampling.
        primary.append(delta)
        models[reward] = {"scores": {mode: float(values.mean()) for mode, values in scores.items()},
                          "learner_effect_opponent_D": _effect(delta),
                          "learner_effect_opponent_S": _effect(scores["SS"] - scores["DS"]),
                          "opponent_effect_learner_D": _effect(scores["DS"] - scores["DD"]),
                          "interaction": _effect((scores["SS"] - scores["DS"]) - delta)}
    effect = _effect(np.mean(primary, axis=0))  # Local: common seeds remain coupled across all three fixed reward variants.
    return {"models": models, "primary": effect, "hypothesis_point_estimate_met": effect["delta"] >= .10,
            "scope": "paired match bootstrap conditional on fixed seed-5201 models; no independent-model or generalization interval"}

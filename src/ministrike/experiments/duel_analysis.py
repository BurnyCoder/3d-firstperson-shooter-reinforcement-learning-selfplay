"""Local: registered paired statistics; global: never turn match count into independent-model count.

Noncentral t: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nct.html
Small-run limitations: https://arxiv.org/abs/1806.08295 and https://arxiv.org/abs/2108.13264
"""

import numpy as np  # Local: transparent arithmetic and seeded resampling.
from scipy.stats import nct, t  # Local: library distribution tails, no hand-approximated power formula.


def paired_power():
    """Local: n3 paired differences, effect0.20; global: sensitivity assumptions are not measured variance."""
    cutoff = t.ppf(.975, 2)  # Local: two-sided alpha0.05 with three independent model differences.
    return [{"models": 3, "effect": .20, "assumed_sd": sd,
             "power": float(nct.sf(cutoff, 2, .20 * np.sqrt(3) / sd) + nct.cdf(-cutoff, 2, .20 * np.sqrt(3) / sd))}
            for sd in (.05, .10, .20)]  # Local: exactly the preregistered sensitivity cases.


def paired_match_interval(initial, final, resamples=5000, *, baselines=("random", "initial", "scripted")):
    """Local: paired seed resampling within opponent; global: keep sides together and deterministic cells fixed."""
    def keyed(rows):
        """Local: index unique outcomes; global: duplicate rounds are a data error rather than extra evidence."""
        selected = [row for row in rows if row["baseline"] in baselines]  # Local: a prospectively registered subset must use unchanged opponents.
        result = {(row["baseline"], row["side"], row["seed"]): row["score"] for row in selected}
        if len(result) != len(selected):
            raise ValueError("Duplicate paired evaluation records")
        return result

    if not baselines or len(set(baselines)) != len(baselines):
        raise ValueError("Select unique paired baselines")
    before, after = keyed(initial["rounds"]), keyed(final["rounds"])
    if set(name for name, _, _ in before) != set(baselines):
        raise ValueError("Missing paired baseline records")
    if before.keys() != after.keys():
        raise ValueError("Initial/final paired worlds do not match")
    rng = np.random.default_rng(5701)  # Local: frozen analysis RNG, never used for training or evaluation.
    draws = np.zeros(resamples)
    all_seeds = sorted({seed for _, _, seed in before})  # Local: preserve common-random-number coupling across baselines.
    indices = rng.integers(len(all_seeds), size=(resamples, len(all_seeds)))  # Local: resample one complete seed block.
    for baseline in baselines:
        seeds = sorted({seed for name, _, seed in before if name == baseline})
        deltas = np.array([np.mean([after[baseline, side, seed] - before[baseline, side, seed] for side in (0, 1)])
                           for seed in seeds])  # Local: keep the two ends of each world paired.
        if len(seeds) == 1:
            draws += deltas[0] / len(baselines)  # Local: fixed deterministic cell has no fictitious match sampling uncertainty.
        elif seeds == all_seeds:
            draws += deltas[indices].mean(-1) / len(baselines)  # Local: retain baseline/side/checkpoint coupling of each random world.
        else:
            raise ValueError("Unequal stochastic seed blocks cannot use the registered paired bootstrap")
    return {"ci95": np.quantile(draws, [.025, .975]).tolist(), "resamples": resamples,
            "scope": "conditional on one checkpoint pair; fixed deterministic cells add no sampling variance"}


def model_comparison(records):
    """Local: analyze all three independent seeds equally; global: incomplete/weak runs are not replaced."""
    complete = [row for row in records if row.get("complete") and row.get("initial") and row.get("final")]
    result = {"status": "complete" if sorted(row["seed"] for row in complete) == [5101, 5102, 5103] else "incomplete",
              "models": [], "mean_improvement": None, "model_ci95": None, "acceptance": False}
    for row in complete:
        result["models"].append({"seed": row["seed"], "improvement": row["final"]["summary"]["score"] - row["initial"]["summary"]["score"],
                                 "match_uncertainty": paired_match_interval(row["initial"], row["final"])})
    if result["status"] != "complete":
        return result  # Local: no selected subset is presented as the planned confirmation experiment.
    deltas = np.array([row["improvement"] for row in result["models"]])
    mean = float(deltas.mean())
    margin = float(t.ppf(.975, 2) * deltas.std(ddof=1) / np.sqrt(3))  # Local: assumption-sensitive normal-difference CI.
    medians = {name: float(np.median([row["final"]["summary"]["baselines"][name]["score"] for row in complete]))
               for name in ("random", "initial", "scripted")}
    elimination = float(np.mean([row["final"]["summary"]["elimination_fraction"] for row in complete]))
    result.update(mean_improvement=mean, model_ci95=[mean - margin, mean + margin], median_final_scores=medians,
                  elimination_fraction=elimination, hypothesis_point_estimate_met=mean >= .20,
                  acceptance=medians["random"] >= .70 and medians["initial"] >= .70 and medians["scripted"] >= .55 and elimination >= .90)
    return result  # Local: acceptance and hypothesis are separate conclusions, not significance claims.

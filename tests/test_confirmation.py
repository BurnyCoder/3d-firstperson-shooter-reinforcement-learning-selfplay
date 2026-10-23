"""Local: synthetic selection fixtures; global: final-test data cannot choose a deployment checkpoint."""

import pytest  # Local: fail closed on missing or altered validation evidence.

from ministrike.experiments.confirmation import freeze_selection


def candidates():
    """Local: three distinct placeholder artifacts; global: no gameplay or model optimization in this test."""
    return [{"seed": seed, "checkpoint": f"seed{seed}/model.pt", "checkpoint_sha256": str(seed)}
            for seed in (5101, 5102, 5103)]


def validation(scores):
    """Local: complete frozen summaries; global: the wrapper uses primary validation only."""
    return {seed: {"status": "complete", "identity": {"checkpoint_sha256": str(seed)},
                   "summary": {"score": score}} for seed, score in zip((5101, 5102, 5103), scores, strict=True)}


def test_selection_tie_and_resume_cannot_change_the_chosen_model(tmp_path):
    """Local: lower seed wins exact ties; global: later results cannot overwrite the frozen choice."""
    path = tmp_path / "selection.json"
    first = freeze_selection(path, candidates(), validation((.8, 1., 1.)))
    assert first["selected"]["seed"] == 5102
    assert freeze_selection(path, candidates(), validation((.8, 1., 1.))) == first
    with pytest.raises(ValueError, match="Frozen selection"):
        freeze_selection(path, candidates(), validation((1., .8, .9)))


def test_selection_requires_all_final_validation_and_matching_hashes(tmp_path):
    """Local: reject incomplete or wrong-model scores; global: failures cannot disappear from selection."""
    rows = validation((1., 1., 1.))
    rows[5103]["status"] = "budget_exhausted"
    with pytest.raises(ValueError, match="Incomplete"):
        freeze_selection(tmp_path / "selection.json", candidates(), rows)
    rows[5103]["status"] = "complete"
    rows[5103]["identity"]["checkpoint_sha256"] = "different"
    with pytest.raises(ValueError, match="identity"):
        freeze_selection(tmp_path / "selection.json", candidates(), rows)

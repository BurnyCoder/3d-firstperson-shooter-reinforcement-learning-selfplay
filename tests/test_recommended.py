"""Local: select compatible published recommendations; global: never quietly open the wrong game.

The approved project requirements define default access to watch, bot POV and human play.
"""

from ministrike.app import recommended_checkpoint  # Local: exercise the production launcher resolution.
from ministrike.records import write_json  # Local: create the same recommendation records as publication.


def test_recommendation_prefers_duel_but_respects_explicit_legacy(tmp_path):
    """Local: resolve the two compatible match identities; global: a new release keeps legacy playback usable."""
    duel, legacy = tmp_path / "duel.pt", tmp_path / "legacy.pt"
    duel.write_bytes(b"duel fixture")
    legacy.write_bytes(b"legacy fixture")
    write_json(tmp_path / "duel-fixed-v1/recommended.json", {"checkpoint": str(duel)})
    write_json(tmp_path / "suite/recommended.json", {"checkpoint": str(legacy)})
    assert recommended_checkpoint(tmp_path) == duel
    assert recommended_checkpoint(tmp_path, "legacy-5v5-v1") == legacy
    assert recommended_checkpoint(tmp_path, "duel-fixed-v1") == duel

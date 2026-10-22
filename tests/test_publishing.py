"""Local: public-release boundaries. Global: only measured, allowlisted artifacts can leave the workspace.

Source: https://huggingface.co/docs/huggingface_hub/en/guides/cli#upload-a-folder
"""

from ministrike.experiments import publishing  # Replace the external boundary, keeping the real staging workflow.
from ministrike.records import write_json  # Fixtures use the production artifact encoding.


def test_publication_filters_secrets_and_preserves_pending_evidence(tmp_path, monkeypatch):
    """Local: retain weak/incomplete runs and exclude secrets. Global: publication cannot imply unmeasured success."""
    run = tmp_path / "selfplay_smoke"  # A direct child matches production run discovery.
    write_json(run / "result.json", {"condition": "selfplay", "seed": 111, "frames": 128,
                                    "updates": 1, "complete": False})  # An interrupted run remains publishable evidence.
    write_json(run / "config.json", {"gamma": 0.999})  # Only the fields used by the card are needed here.
    (run / "model.pt").write_bytes(b"fixture checkpoint")  # Exercise exact byte preservation without torch loading.
    (run / ".env").write_text("PRIVATE_FIXTURE=must_not_upload", encoding="utf-8")  # A source secret must never be staged.
    staging = tmp_path / "publication" / run.name  # Include a stale staging secret to test the final upload filter too.
    staging.mkdir(parents=True)  # Production will reuse this directory on retries.
    (staging / ".env").write_text("OLD_PRIVATE_FIXTURE=must_not_upload", encoding="utf-8")  # A retry cannot rely on a pristine folder.
    calls = []  # Capture CLI arguments without making external writes.
    monkeypatch.setattr(publishing, "hf_command", lambda arguments: calls.append(arguments) or {"ok": True})  # Keep the upload boundary offline.
    result = publishing.publish_models(tmp_path, "owner/collection", namespace="owner")  # Execute the actual public-release orchestration.
    card = (staging / "README.md").read_text(encoding="utf-8")  # Inspect the exact content that would be published.
    assert "**pending**" in card and "completed: **False**" in card  # Missing evaluation is never imputed as a score or completion.
    assert (staging / "model.pt").read_bytes() == b"fixture checkpoint"  # Checkpoint bytes are unchanged by staging.
    include = [calls[0][index + 1] for index, argument in enumerate(calls[0][:-1]) if argument == "--include"]  # Read the final CLI allowlist.
    assert set(include) == {*publishing.FILES, "README.md"} and ".env" not in include  # Neither source nor stale secrets are eligible.
    assert len(calls) == 2 and result["models"][0]["frames"] == 128  # One upload and one collection insertion retain measured provenance.

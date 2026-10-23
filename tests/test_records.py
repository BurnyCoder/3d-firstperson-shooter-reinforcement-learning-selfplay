"""Local: exercise Windows replacement failures; global: preserve the last durable artifact.

The fault models transient file sharing, documented at https://bugs.python.org/issue46003 .
No gameplay optimization occurs in these filesystem-only tests.
"""

import os  # Local: save the real replacement function before injecting a transient error.

import pytest  # Local: assert permanent failures remain visible to callers.

from ministrike import records  # Local: test the production JSON persistence boundary.


def test_transient_replace_failure_preserves_previous_then_commits(tmp_path, monkeypatch):
    """Local: two denied replacements recover; global: readers never observe half a JSON record."""
    target = tmp_path / "status.json"
    records.write_json(target, {"old": True})  # Local: establish the last durable record.
    replace = os.replace
    attempts = []

    def temporarily_locked(source, destination):
        """Local: emulate sharing denial; global: verify old evidence survives every retry."""
        attempts.append(1)
        assert target.read_text().strip() == '{\n  "old": true\n}'
        if len(attempts) <= 2:
            raise PermissionError("synthetic sharing denial")
        replace(source, destination)  # Local: only the final attempt publishes the whole new record.

    monkeypatch.setattr(records.os, "replace", temporarily_locked)
    monkeypatch.setattr(records.time, "sleep", lambda seconds: None)
    records.write_json(target, {"new": True})
    assert len(attempts) == 3 and '"new": true' in target.read_text()


def test_permanent_replace_failure_is_bounded_and_retains_both_versions(tmp_path, monkeypatch):
    """Local: permanent denial is not hidden; global: recovery can inspect both complete versions."""
    target = tmp_path / "status.json"
    records.write_json(target, {"old": True})
    attempts = []

    def denied(source, destination):
        """Local: every replacement fails; global: no unlink fallback may destroy the old file."""
        attempts.append(1)
        raise PermissionError("synthetic permanent denial")

    monkeypatch.setattr(records.os, "replace", denied)
    monkeypatch.setattr(records.time, "sleep", lambda seconds: None)
    with pytest.raises(PermissionError):
        records.write_json(target, {"new": True})
    assert len(attempts) == 6  # Local: initial attempt plus five bounded retries.
    assert '"old": true' in target.read_text()
    assert '"new": true' in target.with_suffix(".json.tmp").read_text()

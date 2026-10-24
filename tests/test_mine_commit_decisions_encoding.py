"""Non-ASCII git history must not corrupt or kill the commit miner (#294).

`mine()` ran `subprocess.run(..., capture_output=True, text=True)` with no
explicit encoding, so the child's bytes were decoded with the locale codepage.
That has two distinct Windows failure modes, and this file pins both:

* On a codepage that maps every byte (cp1252) the UTF-8 commit subject decodes
  into mojibake and the miner reports a corrupted subject. Silent.
* On a multibyte codepage (cp950, cp932) the decode raises - but it raises on
  subprocess's reader *thread*, so the thread dies, `run()` still returns
  `returncode == 0`, and `stdout` is left `None`. Line 50 then called
  `.splitlines()` on it and the user saw `AttributeError: 'NoneType' object has
  no attribute 'splitlines'` with nothing pointing at an encoding problem.

The round-trip test below fails on either codepage: mojibake breaks the equality
and the None breaks the attribute lookup.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import mine_commit_decisions  # noqa: E402

# A decision-shaped subject (matches SIGNALS) carrying CJK, as in the report.
CJK_SUBJECT = "decided 采用新的缓存策略 for cold reads"
ASCII_SUBJECT = "decided to adopt a new cache policy for cold reads"


def _git_repo(path: Path, *subjects: str) -> Path:
    """A throwaway repo whose log carries exactly the given subjects."""
    path.mkdir(parents=True, exist_ok=True)
    env = {
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t.invalid",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t.invalid",
        "GIT_CONFIG_GLOBAL": str(path / "nonexistent-gitconfig"),
    }

    def git(*args: str) -> None:
        # utf-8 here is about *this* helper reading git, not about the code under
        # test - the miner must not depend on the harness getting it right.
        subprocess.run(["git", "-C", str(path), *args], check=True,
                       capture_output=True, encoding="utf-8", errors="replace",
                       env={**_os_environ(), **env})

    git("init", "-q")
    for i, subject in enumerate(subjects):
        (path / f"f{i}.txt").write_text("x", encoding="utf-8")
        git("add", "-A")
        git("commit", "-q", "-m", subject)
    return path


def _os_environ() -> dict:
    import os
    return dict(os.environ)


def test_cjk_commit_subject_round_trips(tmp_path):
    """The reported bug: a CJK subject must come back byte-for-byte.

    Fails on main two different ways depending on the codepage - corrupted on
    cp1252, AttributeError on cp950 - which is exactly why the report was hard
    to read as an encoding problem.
    """
    repo = _git_repo(tmp_path / "cjk", CJK_SUBJECT)

    candidates = mine_commit_decisions.mine(str(repo), 10)

    assert [c["subject"] for c in candidates] == [CJK_SUBJECT]


def test_ascii_commit_subject_still_mined(tmp_path):
    """Control: the ASCII path, which never depended on the codepage."""
    repo = _git_repo(tmp_path / "ascii", ASCII_SUBJECT)

    candidates = mine_commit_decisions.mine(str(repo), 10)

    assert [c["subject"] for c in candidates] == [ASCII_SUBJECT]
    assert candidates[0]["signal"] == "decided"


def test_undecodable_output_reports_the_cause(monkeypatch, tmp_path):
    """A None stdout must say so, not raise AttributeError three frames later.

    subprocess can still hand back `stdout=None` if a reader thread dies, so the
    guard the owner asked for is pinned independently of the encoding fix.
    """
    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(args=["git"], returncode=0,
                                           stdout=None, stderr="")

    monkeypatch.setattr(mine_commit_decisions.subprocess, "run", fake_run)

    with pytest.raises(SystemExit) as excinfo:
        mine_commit_decisions.mine(str(tmp_path), 10)

    assert "decode" in str(excinfo.value).lower() or "output" in str(excinfo.value).lower()


def test_git_failure_still_reports_stderr(tmp_path):
    """Control: a non-repo path keeps the existing 'git log failed' exit."""
    with pytest.raises(SystemExit) as excinfo:
        mine_commit_decisions.mine(str(tmp_path / "not-a-repo"), 10)

    assert "git log failed" in str(excinfo.value)

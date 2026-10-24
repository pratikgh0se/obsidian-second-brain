"""A failed synthesis must not discard the findings Phase 3 already paid for.

When the Phase 4 synthesis call raised, `run_paid_deep` wrote a banner saying
the raw findings were "below" it, and then wrote nothing below it. Every
gap-fill result was lost, which is the one outcome that branch exists to
prevent. This drives the real function with the provider stubbed: gap
analysis and the web query succeed, synthesis times out, and the saved note
must still carry the query's findings.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_PROBE = """
import sys

from scripts.research import research_deep
from scripts.research.lib import perplexity, web_reader

def fake_call(prompt, **kwargs):
    if prompt.startswith("Research this question:"):
        return {"text": "FINDING-MARKER-7f3a: the widget shipped in 2026-08.", "citations": []}
    if kwargs.get("model") == "sonar-reasoning-pro":
        raise TimeoutError("read timed out")
    return {"text": "- when did the widget ship | web"}

perplexity.call = fake_call
web_reader.available = lambda: False
sys.exit(research_deep.run_paid_deep("widget release"))
"""


def test_synthesis_failure_keeps_phase3_findings(tmp_path):
    fake_home = tmp_path / "home"
    config_dir = fake_home / ".config" / "obsidian-second-brain"
    config_dir.mkdir(parents=True)
    vault = tmp_path / "vault"
    vault.mkdir()
    (config_dir / ".env").write_text(
        f"OBSIDIAN_VAULT_PATH={vault}\nPERPLEXITY_API_KEY=pplx-test-key\n", encoding="utf-8"
    )
    env = os.environ.copy()
    env["HOME"] = str(fake_home)
    env["USERPROFILE"] = str(fake_home)
    env.pop("OBSIDIAN_VAULT_PATH", None)
    env.pop("OBSIDIAN_ENV_FILE", None)

    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=REPO_ROOT, env=env, check=False, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    assert result.returncode == 0, result.stderr

    notes = list((vault / "Research" / "Deep").glob("*.md"))
    assert len(notes) == 1, [str(n) for n in notes]
    text = notes[0].read_text(encoding="utf-8")
    assert "Synthesis unavailable" in text
    assert "FINDING-MARKER-7f3a" in text, "the Phase 3 finding was dropped from the saved note"

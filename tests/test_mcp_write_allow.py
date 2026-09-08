"""OBSIDIAN_MCP_WRITE_ALLOW: a folder allowlist for every MCP write tool.

Motivation (agent-ops decision row 93, 2026-09-08): the Hermes worker profiles
mount this server with the full write surface, and a worker whose prompt has
been injected could rewrite the shared kanban view at Boards/Engineering.md or
a Specs/ note. The client cannot disable single tools, so the fence is here: an
env var per profile naming the folder prefixes that profile may write.

Three states, all meaningful:
  unset  -> unrestricted, exactly today's behaviour for every other client
  set    -> the colon-separated vault-relative prefixes it names
  empty  -> read-only: no write tool can name a path
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "integrations" / "obsidian-mcp-server"))


@pytest.fixture()
def vault(tmp_path, monkeypatch):
    v = tmp_path / "vault"
    (v / "Inbox").mkdir(parents=True)
    (v / "Knowledge").mkdir()
    (v / "Boards").mkdir()
    (v / "Boards" / "Engineering.md").write_text(
        "---\ntype: board\n---\n\n## Backlog\n", encoding="utf-8")
    (v / "Knowledge" / "Finding.md").write_text(
        "---\ntype: note\n---\n\nbody\n", encoding="utf-8")
    (v / "Inbox" / "Idea.md").write_text(
        "---\ntype: idea\n---\n\nbody\n", encoding="utf-8")
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(v))
    import vault_ops
    importlib.reload(vault_ops)
    return v, vault_ops


def test_unset_is_unrestricted(vault, monkeypatch):
    v, ops = vault
    monkeypatch.delenv("OBSIDIAN_MCP_WRITE_ALLOW", raising=False)
    assert ops._write_allow() is None
    assert "updated" in ops.update_note("Boards/Engineering.md", append="STILL ALLOWED")


def test_an_allowed_prefix_still_writes(vault, monkeypatch):
    v, ops = vault
    monkeypatch.setenv("OBSIDIAN_MCP_WRITE_ALLOW", "Knowledge/:Inbox/")
    result = ops.update_note("Knowledge/Finding.md", append="APPENDED")
    assert "updated" in result, result
    assert "APPENDED" in (v / "Knowledge" / "Finding.md").read_text(encoding="utf-8")


def test_a_path_outside_the_list_is_refused_and_writes_nothing(vault, monkeypatch):
    v, ops = vault
    monkeypatch.setenv("OBSIDIAN_MCP_WRITE_ALLOW", "Knowledge/:Inbox/")
    before = (v / "Boards" / "Engineering.md").read_text(encoding="utf-8")
    result = ops.update_note("Boards/Engineering.md", append="FORGED CARD")
    assert "error" in result, result
    assert "Knowledge/" in result["error"] and "Inbox/" in result["error"], \
        "the refusal must name the allowed prefixes"
    assert (v / "Boards" / "Engineering.md").read_text(encoding="utf-8") == before


def test_replace_text_is_fenced_too(vault, monkeypatch):
    v, ops = vault
    monkeypatch.setenv("OBSIDIAN_MCP_WRITE_ALLOW", "Inbox/")
    assert "error" in ops.replace_text("Boards/Engineering.md", "## Backlog", "## Owned")
    assert "## Backlog" in (v / "Boards" / "Engineering.md").read_text(encoding="utf-8")


def test_save_note_with_an_explicit_path_is_fenced(vault, monkeypatch):
    v, ops = vault
    monkeypatch.setenv("OBSIDIAN_MCP_WRITE_ALLOW", "Inbox/")
    assert "error" in ops.save_note("Forged", "body", path="Boards/Forged.md")
    assert not (v / "Boards" / "Forged.md").exists()


def test_a_default_capture_lands_in_inbox_and_is_allowed(vault, monkeypatch):
    v, ops = vault
    monkeypatch.setenv("OBSIDIAN_MCP_WRITE_ALLOW", "Inbox/")
    result = ops.capture_idea("a passing thought worth keeping")
    assert "saved" in result, result
    assert result["saved"].startswith("Inbox/")


def test_a_capture_is_refused_when_inbox_is_not_allowed(vault, monkeypatch):
    v, ops = vault
    monkeypatch.setenv("OBSIDIAN_MCP_WRITE_ALLOW", "Knowledge/")
    result = ops.capture_idea("a passing thought worth keeping")
    assert "error" in result, "capture_idea resolves to Inbox/ and must be fenced there"


def test_an_empty_value_is_read_only(vault, monkeypatch):
    v, ops = vault
    monkeypatch.setenv("OBSIDIAN_MCP_WRITE_ALLOW", "")
    assert ops._write_allow() == []
    assert "error" in ops.update_note("Inbox/Idea.md", append="NOPE")
    assert "error" in ops.save_note("Anything", "body")
    # reads are untouched
    assert "content" in ops.read_note("Inbox/Idea.md") or \
        "text" in ops.read_note("Inbox/Idea.md")
    assert "results" in {"results": ops.search("body")}


def test_traversal_is_refused_whatever_the_list_says(vault, monkeypatch):
    v, ops = vault
    monkeypatch.setenv("OBSIDIAN_MCP_WRITE_ALLOW", "..:/:Inbox/")
    result = ops.update_note("../outside.md", append="ESCAPED")
    assert "error" in result
    assert not (v.parent / "outside.md").exists()


def test_move_checks_both_sides(vault, monkeypatch):
    v, ops = vault
    monkeypatch.setenv("OBSIDIAN_MCP_WRITE_ALLOW", "Inbox/")
    out = ops.move_note("Inbox/Idea.md", "Knowledge/Idea.md")
    assert "error" in out, "destination outside the list must be refused"
    assert (v / "Inbox" / "Idea.md").is_file()
    assert not (v / "Knowledge" / "Idea.md").exists()

    monkeypatch.setenv("OBSIDIAN_MCP_WRITE_ALLOW", "Knowledge/")
    out = ops.move_note("Inbox/Idea.md", "Knowledge/Idea.md")
    assert "error" in out, "source outside the list must be refused"
    assert (v / "Inbox" / "Idea.md").is_file()

    monkeypatch.setenv("OBSIDIAN_MCP_WRITE_ALLOW", "Inbox/:Knowledge/")
    out = ops.move_note("Inbox/Idea.md", "Knowledge/Idea.md")
    assert "error" not in out, out
    assert (v / "Knowledge" / "Idea.md").is_file()


def test_a_prefix_does_not_match_a_sibling_by_name(vault, monkeypatch):
    v, ops = vault
    (v / "Inboxes").mkdir()
    (v / "Inboxes" / "Sneaky.md").write_text("---\ntype: note\n---\n\nbody\n", encoding="utf-8")
    monkeypatch.setenv("OBSIDIAN_MCP_WRITE_ALLOW", "Inbox")
    assert "error" in ops.update_note("Inboxes/Sneaky.md", append="NOPE")
    assert "updated" in ops.update_note("Inbox/Idea.md", append="FINE")

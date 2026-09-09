"""OBSIDIAN_MCP_READ_ALLOW: a folder allowlist for every MCP read path.

Motivation (Capsule Corp decision row 125, 2026-09-09): the write fence
(`OBSIDIAN_MCP_WRITE_ALLOW`, row 93) scopes what a mounted profile may change,
but nothing scoped what it may see. A `reviewer` or `pentest` role that should
read nothing could search and read every note in the vault, and a `dev` role
scoped to `Specs/` could read `Decisions/`. This fence closes that: one env var
per profile naming the vault-relative folder prefixes it may read.

Three states, all meaningful, mirroring the write fence exactly:
  unset  -> allow everything, exactly today's behaviour for every client
  set    -> the colon-separated vault-relative prefixes it names
  empty  -> this connection may read no notes

The per-role folder sets are decided in Capsule's C4 tick table, not here; this
file only pins the mechanism.
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
    """A scratch vault with the same term in two folders, so a fence is visible.

    Every note mentions "widget" so the only thing that can change a result set
    is the fence, never relevance.
    """
    v = tmp_path / "vault"
    (v / "Knowledge").mkdir(parents=True)
    (v / "Decisions").mkdir()
    (v / "Specs").mkdir()
    (v / "Knowledge" / "Public Widget.md").write_text(
        "---\ntype: note\n---\n\n## For future agent\n\nwidget notes here.\n"
        "See [[Secret Widget]].\n", encoding="utf-8")
    (v / "Decisions" / "Secret Widget.md").write_text(
        "---\ntype: decision\n---\n\n## For future agent\n\nwidget decision, "
        "confidential.\n", encoding="utf-8")
    (v / "Decisions" / "Other Widget.md").write_text(
        "---\ntype: decision\n---\n\n## For future agent\n\nwidget, also secret.\n"
        "Links [[Secret Widget]].\n", encoding="utf-8")
    (v / "Specs" / "Widget Spec.md").write_text(
        "---\ntype: spec\n---\n\n## For future agent\n\nwidget spec body.\n",
        encoding="utf-8")
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(v))
    # Keep the semantic arm out of it: these cases are about scope, not ranking.
    monkeypatch.setenv("OBSIDIAN_SEARCH_SEMANTIC", "0")
    monkeypatch.delenv("OBSIDIAN_MCP_READ_ALLOW", raising=False)
    monkeypatch.delenv("OBSIDIAN_MCP_WRITE_ALLOW", raising=False)
    import vault_ops
    importlib.reload(vault_ops)
    return v, vault_ops


def _paths(results):
    return {r["path"].replace("\\", "/") for r in results}


# --------------------------------------------------------------------------
# The default: unset changes nothing
# --------------------------------------------------------------------------

def test_unset_is_allow_all(vault, monkeypatch):
    """The whole point of the default: no env var, no behaviour change."""
    v, ops = vault
    monkeypatch.delenv("OBSIDIAN_MCP_READ_ALLOW", raising=False)
    assert ops._read_allow() is None
    hits = _paths(ops.search("widget", limit=20, semantic=False))
    assert hits == {
        "Knowledge/Public Widget.md",
        "Decisions/Secret Widget.md",
        "Decisions/Other Widget.md",
        "Specs/Widget Spec.md",
    }, hits
    assert "content" in ops.read_note("Decisions/Secret Widget.md")
    assert ops.backlinks("Secret Widget")["count"] == 2


def test_unset_search_results_are_byte_identical_to_fence_absent(vault, monkeypatch):
    """A fence-shaped no-op: setting nothing and unsetting must not differ."""
    v, ops = vault
    monkeypatch.delenv("OBSIDIAN_MCP_READ_ALLOW", raising=False)
    before = ops.search("widget", limit=20, semantic=False)
    monkeypatch.delenv("OBSIDIAN_MCP_READ_ALLOW", raising=False)
    assert ops.search("widget", limit=20, semantic=False) == before


def test_iter_notes_unfenced_yields_everything(vault, monkeypatch):
    v, ops = vault
    monkeypatch.delenv("OBSIDIAN_MCP_READ_ALLOW", raising=False)
    found = {p.relative_to(v).as_posix() for p in ops._iter_notes(v)}
    assert len(found) == 4, found


# --------------------------------------------------------------------------
# Parsing: identical to the write fence
# --------------------------------------------------------------------------

def test_parsing_matches_the_write_fence(vault, monkeypatch):
    v, ops = vault
    monkeypatch.setenv("OBSIDIAN_MCP_READ_ALLOW", " Knowledge/ :/Specs/: ")
    monkeypatch.setenv("OBSIDIAN_MCP_WRITE_ALLOW", " Knowledge/ :/Specs/: ")
    assert ops._read_allow() == ops._write_allow() == ["Knowledge/", "Specs/"]


def test_empty_means_read_nothing_not_read_everything(vault, monkeypatch):
    v, ops = vault
    monkeypatch.setenv("OBSIDIAN_MCP_READ_ALLOW", "")
    assert ops._read_allow() == []
    assert ops.search("widget", limit=20, semantic=False) == []
    result = ops.read_note("Knowledge/Public Widget.md")
    assert "error" in result and "read no notes" in result["error"], result


def test_degenerate_slash_entries_do_not_widen_the_fence(vault, monkeypatch):
    """`/` must never strip to the empty prefix that matches every path."""
    v, ops = vault
    monkeypatch.setenv("OBSIDIAN_MCP_READ_ALLOW", "/://")
    assert ops._read_allow() == []
    assert ops.search("widget", limit=20, semantic=False) == []
    monkeypatch.setenv("OBSIDIAN_MCP_READ_ALLOW", "Knowledge/:/")
    assert _paths(ops.search("widget", limit=20, semantic=False)) == {
        "Knowledge/Public Widget.md"}


def test_prefixes_match_whole_components_case_sensitively(vault, monkeypatch):
    v, ops = vault
    monkeypatch.setenv("OBSIDIAN_MCP_READ_ALLOW", "Know")
    assert ops.search("widget", limit=20, semantic=False) == [], \
        "'Know' must not match 'Knowledge/' as a string prefix"
    monkeypatch.setenv("OBSIDIAN_MCP_READ_ALLOW", "knowledge/")
    assert ops.search("widget", limit=20, semantic=False) == [], \
        "the fence is case-sensitive, like the write fence"


# --------------------------------------------------------------------------
# The fence applied to every read path
# --------------------------------------------------------------------------

def test_read_note_outside_the_fence_is_a_clear_error(vault, monkeypatch):
    """Not an empty string, not a bare not-found: an error naming the scope."""
    v, ops = vault
    monkeypatch.setenv("OBSIDIAN_MCP_READ_ALLOW", "Knowledge/")
    result = ops.read_note("Decisions/Secret Widget.md")
    assert "error" in result, result
    assert "content" not in result, "a fenced read must return no content at all"
    assert "Knowledge/" in result["error"], "the refusal must name the readable scope"
    assert "confidential" not in str(result)


def test_read_note_inside_the_fence_still_works(vault, monkeypatch):
    v, ops = vault
    monkeypatch.setenv("OBSIDIAN_MCP_READ_ALLOW", "Knowledge/")
    result = ops.read_note("Knowledge/Public Widget.md")
    assert "widget notes here" in result["content"], result


def test_the_fence_cannot_be_spelled_around_with_dot_dot(vault, monkeypatch):
    """The check is on the resolved path, so traversal cannot re-enter."""
    v, ops = vault
    monkeypatch.setenv("OBSIDIAN_MCP_READ_ALLOW", "Knowledge/")
    result = ops.read_note("Knowledge/../Decisions/Secret Widget.md")
    assert "content" not in result, result
    assert "error" in result


def test_search_drops_fenced_notes(vault, monkeypatch):
    v, ops = vault
    monkeypatch.setenv("OBSIDIAN_MCP_READ_ALLOW", "Knowledge/")
    hits = _paths(ops.search("widget", limit=20, semantic=False))
    assert hits == {"Knowledge/Public Widget.md"}, hits


def test_search_drops_fenced_notes_before_the_ranking_limit(vault, monkeypatch):
    """A role must not be able to infer content from result counts.

    With limit=1 and an unfenced vault the two `Decisions/` notes compete for
    the slot; fenced, the single readable note must still be returned - if the
    fence were applied after the limit, this would come back empty and the
    absence would itself leak that something better-ranked exists.
    """
    v, ops = vault
    monkeypatch.setenv("OBSIDIAN_MCP_READ_ALLOW", "Knowledge/")
    hits = ops.search("widget", limit=1, semantic=False)
    assert _paths(hits) == {"Knowledge/Public Widget.md"}, hits


def test_backlinks_only_reports_readable_notes(vault, monkeypatch):
    v, ops = vault
    monkeypatch.delenv("OBSIDIAN_MCP_READ_ALLOW", raising=False)
    assert ops.backlinks("Secret Widget")["count"] == 2
    monkeypatch.setenv("OBSIDIAN_MCP_READ_ALLOW", "Knowledge/")
    fenced = ops.backlinks("Secret Widget")
    assert fenced["count"] == 1, fenced
    assert all("Decisions" not in b for b in fenced["backlinks"]), fenced


def test_vault_health_only_counts_readable_notes(vault, monkeypatch):
    v, ops = vault
    monkeypatch.delenv("OBSIDIAN_MCP_READ_ALLOW", raising=False)
    assert ops.vault_health()["notes_scanned"] == 4
    monkeypatch.setenv("OBSIDIAN_MCP_READ_ALLOW", "Knowledge/")
    health = ops.vault_health()
    assert health["notes_scanned"] == 1, health
    assert "Decisions" not in str(health), health


def test_validate_note_is_fenced_too(vault, monkeypatch):
    """Validation reports frontmatter keys and wikilinks: note content by proxy."""
    v, ops = vault
    monkeypatch.setenv("OBSIDIAN_MCP_READ_ALLOW", "Knowledge/")
    result = ops.validate_note("Decisions/Secret Widget.md")
    assert "error" in result and "issues" not in result, result
    assert "ok" in ops.validate_note("Knowledge/Public Widget.md")


def test_the_semantic_arm_is_fenced_as_well(vault, monkeypatch):
    """Semantic hits come from the index, not the walk, so they need their own
    filter or a fenced note returns through the back door."""
    v, ops = vault
    fence = ["Knowledge/"]
    monkeypatch.setattr(ops, "_read_allow", lambda: fence)
    sem_paths = [
        "Knowledge/Public Widget.md",
        "Decisions/Secret Widget.md",
        "Specs/Widget Spec.md",
    ]
    notes = {p: {"title": Path(p).stem, "_unit": [[1.0, 0.0]]} for p in sem_paths}
    monkeypatch.setattr(ops, "_load_index_cached", lambda _p: {"notes": notes})
    monkeypatch.setattr(ops, "_embed_query", lambda *a, **k: [1.0, 0.0])
    monkeypatch.setattr(ops, "_warn_if_index_stale", lambda *a, **k: None)
    (v / ops._SEMANTIC_INDEX_FILE).write_text("{}", encoding="utf-8")
    fused = ops._semantic_fuse("widget notes", [], v, 20, enabled=True, scanned=[])
    assert fused is not None, "fixture should reach the fusion path"
    assert _paths(fused) == {"Knowledge/Public Widget.md"}, fused


# --------------------------------------------------------------------------
# The folders parameter on search
# --------------------------------------------------------------------------

def test_folders_narrows_search(vault, monkeypatch):
    v, ops = vault
    monkeypatch.delenv("OBSIDIAN_MCP_READ_ALLOW", raising=False)
    hits = _paths(ops.search("widget", limit=20, semantic=False, folders=["Specs/"]))
    assert hits == {"Specs/Widget Spec.md"}, hits


def test_folders_accepts_several_prefixes(vault, monkeypatch):
    v, ops = vault
    monkeypatch.delenv("OBSIDIAN_MCP_READ_ALLOW", raising=False)
    hits = _paths(ops.search("widget", limit=20, semantic=False,
                             folders=["Specs/", "Knowledge"]))
    assert hits == {"Specs/Widget Spec.md", "Knowledge/Public Widget.md"}, hits


def test_folders_none_is_unchanged(vault, monkeypatch):
    v, ops = vault
    monkeypatch.delenv("OBSIDIAN_MCP_READ_ALLOW", raising=False)
    assert ops.search("widget", limit=20, semantic=False, folders=None) == \
        ops.search("widget", limit=20, semantic=False)


def test_an_empty_folders_list_returns_nothing(vault, monkeypatch):
    """An explicit narrowing that names no folder must not silently widen."""
    v, ops = vault
    monkeypatch.delenv("OBSIDIAN_MCP_READ_ALLOW", raising=False)
    assert ops.search("widget", limit=20, semantic=False, folders=[]) == []
    assert ops.search("widget", limit=20, semantic=False, folders=["/"]) == []


def test_folders_cannot_widen_past_the_fence(vault, monkeypatch):
    """The two are intersected: naming a fenced folder yields nothing."""
    v, ops = vault
    monkeypatch.setenv("OBSIDIAN_MCP_READ_ALLOW", "Knowledge/")
    assert ops.search("widget", limit=20, semantic=False,
                      folders=["Decisions/"]) == []
    assert _paths(ops.search("widget", limit=20, semantic=False,
                             folders=["Knowledge/", "Decisions/"])) == {
        "Knowledge/Public Widget.md"}


# --------------------------------------------------------------------------
# The two fences are independent and compose
# --------------------------------------------------------------------------

def test_the_write_fence_still_works_on_its_own(vault, monkeypatch):
    """A read fence must not be needed for, or interfere with, the write fence."""
    v, ops = vault
    monkeypatch.delenv("OBSIDIAN_MCP_READ_ALLOW", raising=False)
    monkeypatch.setenv("OBSIDIAN_MCP_WRITE_ALLOW", "Knowledge/")
    assert "updated" in ops.update_note("Knowledge/Public Widget.md", append="OK")
    assert "error" in ops.update_note("Decisions/Secret Widget.md", append="NOPE")
    # ...and reading is still unrestricted, because only the write var is set.
    assert "content" in ops.read_note("Decisions/Secret Widget.md")


def test_a_read_fence_alone_does_not_restrict_writes(vault, monkeypatch):
    v, ops = vault
    monkeypatch.setenv("OBSIDIAN_MCP_READ_ALLOW", "Knowledge/")
    monkeypatch.delenv("OBSIDIAN_MCP_WRITE_ALLOW", raising=False)
    assert "updated" in ops.update_note("Decisions/Secret Widget.md", append="OK"), \
        "the read fence must not silently become a write fence"
    assert "error" in ops.read_note("Decisions/Secret Widget.md")


def test_the_two_fences_compose(vault, monkeypatch):
    """Different scopes per direction: read Decisions/, write only Knowledge/."""
    v, ops = vault
    monkeypatch.setenv("OBSIDIAN_MCP_READ_ALLOW", "Decisions/")
    monkeypatch.setenv("OBSIDIAN_MCP_WRITE_ALLOW", "Knowledge/")
    assert "content" in ops.read_note("Decisions/Secret Widget.md")
    assert "error" in ops.read_note("Knowledge/Public Widget.md")
    assert "error" in ops.update_note("Decisions/Secret Widget.md", append="NOPE")
    assert "updated" in ops.update_note("Knowledge/Public Widget.md", append="OK")


def test_read_only_and_write_only_are_both_expressible(vault, monkeypatch):
    v, ops = vault
    monkeypatch.setenv("OBSIDIAN_MCP_WRITE_ALLOW", "")
    monkeypatch.delenv("OBSIDIAN_MCP_READ_ALLOW", raising=False)
    assert "content" in ops.read_note("Knowledge/Public Widget.md")
    assert "error" in ops.update_note("Knowledge/Public Widget.md", append="NOPE")

    monkeypatch.setenv("OBSIDIAN_MCP_READ_ALLOW", "")
    monkeypatch.delenv("OBSIDIAN_MCP_WRITE_ALLOW", raising=False)
    assert "error" in ops.read_note("Knowledge/Public Widget.md")
    assert "updated" in ops.update_note("Knowledge/Public Widget.md", append="OK")


# --------------------------------------------------------------------------
# The tool surface
# --------------------------------------------------------------------------

def test_search_exposes_folders_over_mcp():
    """The parameter has to reach the tool, or the fence's companion is
    unreachable from a client."""
    src = (REPO_ROOT / "integrations" / "obsidian-mcp-server" / "server.py").read_text(
        encoding="utf-8")
    assert "def obsidian_search(query: str, limit: int = 6, folders: list[str] | None = None)" \
        in src, "obsidian_search must accept folders"
    assert "folders=folders" in src


def test_the_read_fence_is_documented():
    """A fence nobody can find is a fence nobody sets."""
    for doc in (REPO_ROOT / "integrations" / "obsidian-mcp-server" / "README.md",
                REPO_ROOT / "SKILL.md"):
        text = doc.read_text(encoding="utf-8")
        assert "OBSIDIAN_MCP_READ_ALLOW" in text, doc
        assert "OBSIDIAN_MCP_WRITE_ALLOW" in text, doc

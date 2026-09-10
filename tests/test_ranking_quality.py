"""Ranking quality: volume is not relevance (stress-test fix 13/24).

Lexical: term-dense logs took #1 on 7 of 12 audit queries, burying canonical
notes - notes typed log/daily now fade to 0.5 (a moderator, not a mute) and
person/entity dossiers boost 1.5x. Semantic: mean-pooling a long note into one
averaged vector made a person dossier unfindable despite containing the
answer verbatim - the index now stores per-chunk vectors with an identity
header, and a note scores by its best chunk.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "integrations" / "obsidian-mcp-server"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "eval"))

import semantic_search as ss  # noqa: E402
import vault_ops  # noqa: E402


@pytest.fixture()
def vault(tmp_path, monkeypatch):
    v = tmp_path / "vault"
    v.mkdir()
    monkeypatch.setenv(vault_ops._VAULT_ENV, str(v))
    return v


def test_canonical_note_outranks_shouting_log(vault):
    (vault / "wiki" / "concepts").mkdir(parents=True)
    (vault / "wiki" / "logs").mkdir()
    (vault / "wiki" / "concepts" / "gateway-pattern.md").write_text(
        "---\ntype: concept\n---\n\nThe gateway pattern explained: gateway gateway.\n",
        encoding="utf-8",
    )
    # The log mentions the term far more often - volume, not relevance.
    (vault / "wiki" / "logs" / "2026-07-01-worklog.md").write_text(
        "---\ntype: log\n---\n\n" + ("gateway deploy gateway retry gateway " * 40),
        encoding="utf-8",
    )

    hits = vault_ops.search("gateway", limit=2, semantic=False)
    assert hits[0]["path"] == "wiki/concepts/gateway-pattern.md", hits


def test_untyped_note_in_logs_folder_also_fades(vault):
    assert vault_ops._type_weight("wiki/logs/x.md", "no frontmatter") == vault_ops._SEARCH_LOG_WEIGHT
    assert vault_ops._type_weight("Daily/2026-07-11.md", "") == vault_ops._SEARCH_LOG_WEIGHT
    assert vault_ops._type_weight("wiki/concepts/x.md", "") == 1.0


def test_entity_boost_applies(vault):
    w = vault_ops._type_weight("wiki/entities/ken.md", "---\ntype: person\n---")
    assert w == vault_ops._SEARCH_ENTITY_BOOST > 1.0


def test_best_chunk_beats_average(vault, monkeypatch):
    """A dossier with one relevant section must score by that section."""
    index = {
        "model": "fake", "format": 2,
        "notes": {
            # dossier: 9 unrelated chunks + 1 perfect chunk
            "dossier.md": {"title": "dossier",
                           "vecs": [[0.0, 1.0]] * 9 + [[1.0, 0.0]]},
            # short note: one mediocre chunk
            "meh.md": {"title": "meh", "vecs": [[0.7, 0.7]]},
            # legacy single-vec entry must still score (backwards compat)
            "old.md": {"title": "old", "vec": [0.5, 0.85]},
        },
    }
    monkeypatch.setattr(ss, "embed", lambda q, **kw: [1.0, 0.0])
    hits = ss.semantic_search("q", index, limit=3)
    assert hits[0]["path"] == "dossier.md"
    assert {h["path"] for h in hits} == {"dossier.md", "meh.md", "old.md"}


def test_fuse_scores_by_best_chunk_too(vault, monkeypatch):
    index = {
        "model": "fake", "format": 2,
        "notes": {
            "dossier.md": {"title": "dossier", "vecs": [[0.0, 1.0], [1.0, 0.0]]},
            "shallow.md": {"title": "shallow", "vecs": [[0.8, 0.6]]},
        },
    }
    (vault / vault_ops._SEMANTIC_INDEX_FILE).write_text(json.dumps(index), encoding="utf-8")
    monkeypatch.setattr(vault_ops, "_embed_query", lambda q, **kw: [1.0, 0.0])
    fused = vault_ops._semantic_fuse("some multi word query", [], vault, 5, enabled=True)
    assert fused is not None
    assert fused[0]["path"] == "dossier.md"


# --------------------------------------------------------------------------- #
# `chunk`: which section won, on the mode the MCP actually serves
# (Capsule decision row 128 - `retrieval_eval.chunk_hit_position`)
# --------------------------------------------------------------------------- #
def test_fused_results_name_the_winning_chunk_and_lexical_only_hits_do_not(vault, monkeypatch):
    """A semantic hit reports the 1-based index of its best chunk; a hit that
    only the lexical arm found reports None, never a synthesised 1. Without
    this, a chunker that embeds a fraction of every note is invisible to
    recall@k, which is exactly how 45% of a vault went unembedded unnoticed."""
    index = {
        "model": "fake", "format": 2,
        "notes": {
            # The SECOND chunk is the relevant one.
            "dossier.md": {"title": "dossier", "vecs": [[0.0, 1.0], [1.0, 0.0]]},
        },
    }
    (vault / vault_ops._SEMANTIC_INDEX_FILE).write_text(json.dumps(index), encoding="utf-8")
    monkeypatch.setattr(vault_ops, "_embed_query", lambda q, **kw: [1.0, 0.0])
    lexical = [{"path": "wordmatch.md", "title": "wordmatch", "snippet": "", "chunk": None}]
    fused = vault_ops._semantic_fuse("some multi word query", lexical, vault, 5, enabled=True)
    by_path = {r["path"]: r for r in fused}
    assert by_path["dossier.md"]["chunk"] == 2
    assert by_path["wordmatch.md"]["chunk"] is None
    # ...and `score` is still the scalar every consumer sorts on (it is popped
    # from the returned rows, so its absence here is the contract).
    assert "score" not in by_path["dossier.md"]


def _write_note(vault, rel, body):
    p = vault / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"---\ntype: concept\n---\n\n{body}\n", encoding="utf-8")


@pytest.mark.parametrize("semantic", [False, True])
def test_search_results_always_carry_a_chunk_key(vault, monkeypatch, semantic):
    """`chunk` is a uniform part of a result - an int or None - on the fused
    path AND on the pure-lexical fallback a keyless install runs."""
    _write_note(vault, "Knowledge/gateway pattern.md", "The gateway pattern for retry logic.")
    if semantic:
        (vault / vault_ops._SEMANTIC_INDEX_FILE).write_text(json.dumps({
            "model": "fake", "format": 2,
            "notes": {"Knowledge/gateway pattern.md": {
                "title": "gateway pattern", "vecs": [[0.0, 1.0], [1.0, 0.0]]}},
        }), encoding="utf-8")
        monkeypatch.setattr(vault_ops, "_embed_query", lambda q, **kw: [1.0, 0.0])

    hits = vault_ops.search("gateway retry pattern", limit=5, semantic=semantic)
    assert hits, "the lexical arm must still find the note"
    for h in hits:
        assert "chunk" in h, h
        assert h["chunk"] is None or isinstance(h["chunk"], int)
    if semantic:
        assert hits[0]["chunk"] == 2, hits


def test_prepare_note_text_header_and_scaffolding():
    header, body = ss.prepare_note_text(
        "Atlas",
        "---\ntype: project\naliases: [Ada agent]\nrelated-people: [Ada Lovelace]\n---\n\n"
        "## For future agent\n\nPersonal agent gateway for Ada.\n\n"
        "## Empty scaffold\n\n## Also empty\n\n"
        "## Filled\n\ncontent here\n",
    )
    assert header.startswith("Atlas | project | Ada agent | Ada Lovelace")
    assert "Personal agent gateway" in body
    assert "content here" in body
    assert "Empty scaffold" not in body
    assert "Also empty" not in body


def test_all_scaffold_note_still_embeds_identity():
    header, body = ss.prepare_note_text(
        "Ghost Town", "---\ntype: daily\n---\n\n## A\n\n## B\n\n## C\n"
    )
    assert body == ""
    assert "Ghost Town" in header

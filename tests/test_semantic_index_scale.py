"""The semantic index at vault scale: the model's real window, the exclusions
the vault's folder map already demanded, and a reindex that survives being
killed.

Measured on ~/second-brain on 2026-09-09 (Study - Capsule Corp's memory at
scale), the index had three defects that these tests pin shut:

1. **45% of all vault text was never embedded.** The chunker was sized for a
   "typically ~512 token" model - 1,200 chars x 8 chunks - so a note was
   embedded for at most 9,600 characters and everything after it could not be
   retrieved semantically at any rank by any query. 464 of 950 notes exceeded
   that cap. `bge-m3`, the model actually configured, accepts 8,192 tokens.
2. **It indexed what the vault forbids.** 17 of 107 indexed notes came from
   `copilot/`, which `_CLAUDE.md` tells every agent and job to ignore entirely,
   while three quarters of the vault is a skill mirror putting ten identical
   candidates in front of every query.
3. **The nightly build could not finish and kept nothing when killed.** It wrote
   the index once at the end; the 2026-09-08 run embedded "600+ notes and
   climbing", hit the job's wall-clock cap, and wrote zero.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "eval"))

import semantic_search as ss  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _fake_embed(monkeypatch, wall: int | None = None):
    """A deterministic embedder that records what it was asked to embed."""
    seen: list[str] = []

    def fake(text, retries=None, model=None):
        seen.append(text)
        if wall is not None and len(text) > wall:
            raise RuntimeError("HTTP Error 500 (wall)")
        return [float(len(text) % 7), 1.0]

    monkeypatch.setattr(ss, "embed", fake)
    return seen


def _big_note(words: int) -> str:
    """A structured note of roughly `words` words: sections, paragraphs, tables.

    Every word is unique so a dropped span is provable, not merely plausible.
    """
    out = ["---", "type: study", "---", "", "## For future agent", ""]
    n = 0
    section = 0
    while n < words:
        section += 1
        out.append(f"## Section {section}")
        out.append("")
        for _ in range(6):
            para = " ".join(f"w{n + i}" for i in range(40))
            out.append(para)
            out.append("")
            n += 40
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# 1. Chunking: the model's window, not a guess
# --------------------------------------------------------------------------- #
def test_chunk_geometry_matches_the_model_window():
    """The constants must be self-documenting: the ceiling is a safety net, and
    a chunk must be a meaningful fraction of bge-m3's 8,192-token window."""
    assert ss._MODEL_WINDOW_TOKENS == 8192, "bge-m3's real input window"
    assert ss._CHUNK_CHARS >= 4000, "a chunk must hold a whole ## section"
    assert ss._MAX_CHUNKS >= 64, "the count cap must not be the note's size limit"
    # The per-note budget must cover the largest note in the vault (19,649
    # words, ~130k chars) with room to spare.
    assert ss._MAX_CHUNKS * ss._CHUNK_CHARS >= 20_000 * 6
    assert 0 < ss._CHUNK_OVERLAP_CHARS < ss._CHUNK_CHARS


def test_fifteen_thousand_word_note_yields_more_than_eight_chunks():
    """The headline regression: the old geometry gave this note 8 chunks and
    threw away 88% of it."""
    text = _big_note(15_000)
    chunks = ss.chunk_note_text(text, header="Title | study\n")
    assert len(chunks) > 8, f"only {len(chunks)} chunks - the cap is back"
    assert len(chunks) <= ss._MAX_CHUNKS, "must fit under the safety ceiling"


def test_no_text_is_dropped():
    """Every word of a 15k-word note must appear in at least one chunk."""
    text = _big_note(15_000)
    body = text.strip()
    chunks = ss.chunk_note_text(text, header="Title | study\n")
    joined = "\n".join(chunks)
    missing = [w for w in body.split() if w not in joined]
    assert not missing, f"{len(missing)} tokens dropped, e.g. {missing[:5]}"
    # Overlap means the concatenation is longer than the source, never shorter.
    assert len(joined) >= len(body)


def test_chunks_respect_the_header_budget():
    header = "A long identity header | study | alias one | alias two\n"
    for chunk in ss.chunk_note_text(_big_note(4_000), header=header):
        assert len(header) + len(chunk) <= ss._CHUNK_CHARS


def test_chunks_split_on_headings_not_mid_sentence(monkeypatch):
    """Structure is free signal, and `text[i:i+1200]` discarded it: it split
    mid-sentence and mid-table. Chunk boundaries must land on the note's own
    headings whenever its sections are near the chunk size.

    Overlap is disabled here so the boundary itself is what is under test - with
    overlap on, every chunk after the first legitimately opens with the tail of
    its predecessor.
    """
    monkeypatch.setattr(ss, "_CHUNK_OVERLAP_CHARS", 0)
    text = "\n\n".join(f"## Heading {i}\n\n" + ("body sentence. " * 250) for i in range(6))
    chunks = ss.chunk_note_text(text)
    assert len(chunks) > 1
    for c in chunks:
        assert c.lstrip().startswith("## Heading"), f"boundary fell mid-section: {c[:60]!r}"
    # ...and no sentence was cut in half.
    for c in chunks:
        assert c.rstrip().endswith(("sentence.", "sentence")), c[-40:]


def test_unstructured_wall_of_text_is_still_covered():
    """No separator to split on must mean a hard slice, never a truncation."""
    text = "x" * 200_000
    chunks = ss.chunk_note_text(text)
    assert sum(len(c) for c in chunks) >= 200_000


def test_ceiling_warns_by_name_instead_of_truncating_silently(monkeypatch, capsys):
    """A note past the ceiling is a bug alarm. Silence is what we are fixing."""
    _fake_embed(monkeypatch)
    monkeypatch.setattr(ss, "_MAX_CHUNKS", 3)
    vecs = ss.embed_note_chunks(_big_note(20_000), header="T\n")
    assert len(vecs) <= 3
    assert "exceeds the _MAX_CHUNKS" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# 2. Exclusions: one config constant, the vault's folder map
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "rel,reason_fragment",
    [
        ("copilot/skills/x.md", "copilot"),
        ("copilot/conversations/deep/y.md", "copilot"),
        (".obsidian/plugins/z.md", ".obsidian"),
        (".smart-env/index.md", ".smart-env"),
        (".claude-runs/nightly.md", ".claude-runs"),
        ("_trash/deleted.md", "_trash"),
        ("_archive/superseded.md", "_archive"),
        ("Ledger/engineering/CAP-1.md", "ledger/"),
        ("Boards/Engineering.md", "boards/"),
        ("Architecture/skills/claude-code.md", "architecture/skills/"),
        ("Architecture/skills/hermes/dev/apple-reminders.md", "architecture/skills/"),
        ("some/other/skills/hermes/qa/doc.md", "skill mirror"),
        ("drawing.excalidraw.md", "excalidraw"),
        ("Templates/note.md", "templates"),
    ],
)
def test_policy_denies_every_forbidden_shape(rel, reason_fragment):
    verdict = ss.policy_verdict(rel)
    assert verdict is not None, f"{rel} must not be indexed"
    assert reason_fragment.lower() in verdict.lower(), f"{rel}: {verdict}"


@pytest.mark.parametrize(
    "rel",
    [
        "Knowledge/Study - memory at scale.md",
        "Projects/Agent Ops.md",
        "Architecture/agent-ops/design.md",
        "Decisions/2026-09-06.md",
        "Daily/2026-09-09.md",
        "Specs/CAP-12.md",
        "root-note.md",
    ],
)
def test_policy_allows_the_knowledge_vault(rel):
    assert ss.policy_verdict(rel) is None, f"{rel} must be indexed"


def test_allow_prefix_overrides_a_deny():
    """The escape hatch: one canonical copy of a mirrored doc can be re-admitted
    without weakening the pattern that excludes the other 718."""
    policy = dict(ss.INDEX_POLICY)
    policy["allow_path_prefixes"] = ("Architecture/skills/claude-code.md",)
    old = ss.INDEX_POLICY
    try:
        ss.INDEX_POLICY = policy
        assert ss.policy_verdict("Architecture/skills/claude-code.md") is None
        assert ss.policy_verdict("Architecture/skills/hermes/dev/x.md") is not None
    finally:
        ss.INDEX_POLICY = old


def test_the_policy_is_one_constant():
    """Six tools each carrying their own copy is how the drift happened. Every
    list the indexer consults lives in INDEX_POLICY."""
    for key in ("allow_path_prefixes", "deny_dir_names", "deny_path_prefixes",
                "deny_path_patterns", "deny_name_suffixes"):
        assert key in ss.INDEX_POLICY


def test_build_skips_forbidden_folders_in_a_real_vault(tmp_path, monkeypatch, capsys):
    seen = _fake_embed(monkeypatch)
    vault = tmp_path / "vault"
    for rel in (
        "Knowledge/good-one.md",
        "Knowledge/good-two.md",
        "copilot/skills/scaffold.md",
        "Architecture/skills/hermes/dev/mirror.md",
        "Architecture/skills/hermes/lead/mirror.md",
        "Ledger/eng/CAP-1.md",
        "Boards/Engineering.md",
        ".smart-env/idx.md",
        ".claude-runs/run.md",
        "_trash/gone.md",
    ):
        p = vault / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("---\ntype: note\n---\n\nSECRET-BODY prose here.\n", encoding="utf-8")

    index = ss.build_index(vault, verbose=True)

    assert sorted(index["notes"]) == ["Knowledge/good-one.md", "Knowledge/good-two.md"]
    # Stronger than a count: no forbidden note's TEXT was ever sent to the model.
    assert len(seen) == 2, f"embedded {len(seen)} texts, expected 2"
    err = capsys.readouterr().err
    assert "excluded x" in err, "exclusions must be reported, with reasons"


# --------------------------------------------------------------------------- #
# 3. Incremental, atomic, kill-safe reindex
# --------------------------------------------------------------------------- #
def _vault_with(tmp_path, n: int) -> Path:
    vault = tmp_path / "vault"
    (vault / "Knowledge").mkdir(parents=True)
    for i in range(n):
        (vault / "Knowledge" / f"note-{i:02d}.md").write_text(
            f"---\ntype: note\n---\n\n## For future agent\n\nNote {i} prose body.\n",
            encoding="utf-8",
        )
    return vault


def test_second_run_embeds_nothing(tmp_path, monkeypatch, capsys):
    """Incremental by content hash: an unchanged vault costs zero embeddings."""
    seen = _fake_embed(monkeypatch)
    vault = _vault_with(tmp_path, 12)

    ss.build_index(vault, verbose=True)
    first = len(seen)
    assert first == 12
    assert "reindex: 12 notes, 12 embedded" in capsys.readouterr().err

    seen.clear()
    ss.build_index(vault, verbose=True)
    assert len(seen) == 0, "an unchanged vault must re-embed nothing"
    err = capsys.readouterr().err
    assert "reindex: 12 notes, 0 embedded" in err


def test_only_the_changed_note_is_re_embedded(tmp_path, monkeypatch):
    seen = _fake_embed(monkeypatch)
    vault = _vault_with(tmp_path, 8)
    ss.build_index(vault, verbose=False)
    seen.clear()

    (vault / "Knowledge" / "note-03.md").write_text(
        "---\ntype: note\n---\n\nRewritten body, brand new content.\n", encoding="utf-8"
    )
    index = ss.build_index(vault, verbose=False)

    assert len(seen) == 1, "only the edited note may be re-embedded"
    assert "Rewritten" in seen[0]
    assert len(index["notes"]) == 8


def test_deleted_notes_are_removed_from_the_index(tmp_path, monkeypatch, capsys):
    _fake_embed(monkeypatch)
    vault = _vault_with(tmp_path, 5)
    ss.build_index(vault, verbose=False)

    (vault / "Knowledge" / "note-02.md").unlink()
    index = ss.build_index(vault, verbose=True)

    assert "Knowledge/note-02.md" not in index["notes"]
    assert len(index["notes"]) == 4
    assert "removed from index" in capsys.readouterr().err
    on_disk = json.loads((vault / ss.INDEX_FILE).read_text())
    assert "Knowledge/note-02.md" not in on_disk["notes"]


def test_a_killed_run_keeps_its_progress(tmp_path, monkeypatch):
    """The 2026-09-08 failure: the builder wrote once at the end, so a run killed
    by the job's wall-clock cap kept nothing. With batch=1 every embedded note is
    already durable, and the resumed run only pays for what is left."""
    vault = _vault_with(tmp_path, 10)
    calls = {"n": 0}

    def dying(text, retries=None, model=None):
        calls["n"] += 1
        if calls["n"] > 4:
            raise KeyboardInterrupt("wall-clock cap - claude killed")
        return [1.0, 0.0]

    monkeypatch.setattr(ss, "embed", dying)
    with pytest.raises(KeyboardInterrupt):
        ss.build_index(vault, verbose=False, batch=1)

    partial = json.loads((vault / ss.INDEX_FILE).read_text())
    assert len(partial["notes"]) >= 3, "a killed run must leave its work on disk"
    assert partial["format"] == 2 and partial["model"] == ss.EMBED_MODEL

    # Resume: the survivors are reused, only the rest are embedded.
    seen = _fake_embed(monkeypatch)
    index = ss.build_index(vault, verbose=False, batch=1)
    assert len(index["notes"]) == 10
    assert len(seen) == 10 - len(partial["notes"])


def test_interim_flushes_land_at_the_batch_boundary(tmp_path, monkeypatch):
    vault = _vault_with(tmp_path, 9)
    sizes: list[int] = []

    real_write = ss._atomic_write_index

    def spy(path, payload):
        sizes.append(len(payload["notes"]))
        real_write(path, payload)

    _fake_embed(monkeypatch)
    monkeypatch.setattr(ss, "_atomic_write_index", spy)
    ss.build_index(vault, verbose=False, batch=3)

    # 3 interim flushes (after 3, 6, 9) plus the final authoritative one.
    assert sizes == [3, 6, 9, 9], sizes


def test_index_is_written_atomically(tmp_path, monkeypatch):
    """A reader must never see a half-written index, and no .tmp litter is left."""
    _fake_embed(monkeypatch)
    vault = _vault_with(tmp_path, 6)
    ss.build_index(vault, verbose=False, batch=2)

    assert json.loads((vault / ss.INDEX_FILE).read_text())["notes"]
    assert not list(vault.glob(f"{ss.INDEX_FILE}.tmp*")), "temp files left behind"


def test_a_corrupt_index_does_not_stop_a_rebuild(tmp_path, monkeypatch):
    _fake_embed(monkeypatch)
    vault = _vault_with(tmp_path, 3)
    (vault / ss.INDEX_FILE).write_text("{not json", encoding="utf-8")
    assert len(ss.build_index(vault, verbose=False)["notes"]) == 3


def test_summary_line_shape(tmp_path, monkeypatch, capsys):
    """The one line a job log or the status board greps for."""
    _fake_embed(monkeypatch)
    vault = _vault_with(tmp_path, 4)
    (vault / "copilot").mkdir()
    (vault / "copilot" / "x.md").write_text("body", encoding="utf-8")
    ss.build_index(vault, verbose=False)
    ss.build_index(vault, verbose=True)
    line = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("reindex: ")]
    assert len(line) == 1
    assert line[0].startswith("reindex: 4 notes, 0 embedded, 1 skipped, ")
    assert line[0].endswith(" s")


# --------------------------------------------------------------------------- #
# 4. --stats
# --------------------------------------------------------------------------- #
def test_a_chunker_change_invalidates_the_cache(tmp_path, monkeypatch):
    """The subtlest failure in this change: the note's text is unchanged, so the
    hash matches, so the old truncated 8-chunk vectors would be reused forever
    while every counter reported a healthy cached hit. Vectors are comparable
    only if they came from the same slices of text.
    """
    seen = _fake_embed(monkeypatch)
    vault = _vault_with(tmp_path, 5)
    ss.build_index(vault, verbose=False)
    assert json.loads((vault / ss.INDEX_FILE).read_text())["chunker"]
    seen.clear()

    monkeypatch.setattr(ss, "_CHUNK_CHARS", ss._CHUNK_CHARS // 2)
    ss.build_index(vault, verbose=False)
    assert len(seen) == 5, "a chunk-geometry change must re-embed everything"


def test_a_model_or_format_change_still_invalidates_the_cache(tmp_path, monkeypatch):
    seen = _fake_embed(monkeypatch)
    vault = _vault_with(tmp_path, 3)
    ss.build_index(vault, verbose=False)
    payload = json.loads((vault / ss.INDEX_FILE).read_text())
    payload["model"] = "some-other-embedding-model"
    (vault / ss.INDEX_FILE).write_text(json.dumps(payload), encoding="utf-8")
    seen.clear()
    ss.build_index(vault, verbose=False)
    assert len(seen) == 3


def test_stats_reports_coverage_staleness_and_chunk_spread(tmp_path, monkeypatch, capsys):
    _fake_embed(monkeypatch)
    vault = _vault_with(tmp_path, 10)
    (vault / "copilot").mkdir()
    (vault / "copilot" / "scaffold.md").write_text("body", encoding="utf-8")
    ss.build_index(vault, verbose=False)

    # One note edited (stale) and one added (never indexed).
    (vault / "Knowledge" / "note-00.md").write_text("changed body", encoding="utf-8")
    (vault / "Knowledge" / "fresh.md").write_text("brand new", encoding="utf-8")

    s = ss.index_stats(vault)
    assert s["notes_indexed"] == 10
    assert s["notes_eligible"] == 11
    assert s["never_indexed"] == 1
    assert s["stale_hash"] == 1
    assert s["excluded_total"] == 1
    assert s["chunks_p50"] >= 1 and s["chunks_p90"] >= 1
    assert s["oldest_write"] and s["oldest_write"] <= time.time()

    assert ss.print_stats(vault) == 0
    out = capsys.readouterr().out
    for expected in ("coverage", "never indexed", "stale (hash)", "oldest write", "chunks/note"):
        assert expected in out


def test_stats_needs_no_embedding_backend(tmp_path, monkeypatch):
    """--stats is what you run when search is broken, i.e. when the backend is
    down. It must not go anywhere near it."""
    _fake_embed(monkeypatch)
    vault = _vault_with(tmp_path, 3)
    ss.build_index(vault, verbose=False)

    def explode(*a, **k):
        raise AssertionError("--stats must not call the embedding backend")

    monkeypatch.setattr(ss, "embed", explode)
    monkeypatch.setattr(ss, "ollama_available", lambda: False)
    assert ss.main(["prog", "--path", str(vault), "--stats"]) == 0


def test_stats_on_a_vault_with_no_index(tmp_path, capsys):
    vault = _vault_with(tmp_path, 2)
    assert ss.print_stats(vault) == 1
    assert "no index" in capsys.readouterr().err


def test_stats_json_is_machine_readable(tmp_path, monkeypatch, capsys):
    _fake_embed(monkeypatch)
    vault = _vault_with(tmp_path, 3)
    ss.build_index(vault, verbose=False)
    ss.print_stats(vault, as_json=True)
    payload = json.loads(capsys.readouterr().out)
    assert payload["notes_indexed"] == 3
    assert payload["coverage_pct"] == 100.0


def test_stats_flags_orphan_entries(tmp_path, monkeypatch):
    """A note that became ineligible (moved into copilot/, say) leaves an entry
    behind until the next full build. Say so rather than hide it."""
    _fake_embed(monkeypatch)
    vault = _vault_with(tmp_path, 3)
    ss.build_index(vault, verbose=False)
    payload = json.loads((vault / ss.INDEX_FILE).read_text())
    payload["notes"]["copilot/sneaked-in.md"] = {"hash": "x", "title": "s", "vecs": [[1.0, 0.0]]}
    (vault / ss.INDEX_FILE).write_text(json.dumps(payload), encoding="utf-8")
    assert ss.index_stats(vault)["orphan_entries"] == 1


# --------------------------------------------------------------------------- #
# 5. The search API and the eval script keep working unchanged
# --------------------------------------------------------------------------- #
def test_search_api_is_unchanged(tmp_path, monkeypatch):
    _fake_embed(monkeypatch)
    vault = _vault_with(tmp_path, 5)
    ss.build_index(vault, verbose=False)
    index = ss.load_index(vault)

    hits = ss.semantic_search("anything", index, limit=3)
    assert len(hits) == 3
    assert set(hits[0]) == {"path", "title", "score"}

    lexical = [{"path": "Knowledge/note-01.md", "title": "note-01"}]
    fused = ss.hybrid_search("anything", index, lexical, limit=3)
    assert len(fused) == 3
    assert set(fused[0]) == {"path", "title", "score"}


def test_coverage_counts_only_notes_the_index_is_meant_to_hold(tmp_path, monkeypatch):
    """The MCP server ships standalone and cannot import the policy, so the
    builder writes the rules it applied into the index file and vault_ops
    applies them generically. Without this, a vault that is three quarters
    documentation mirror reports a permanent 76% missing and warns forever.
    """
    sys.path.insert(0, str(REPO_ROOT / "integrations" / "obsidian-mcp-server"))
    import vault_ops as vo

    _fake_embed(monkeypatch)
    vault = _vault_with(tmp_path, 6)
    for rel in ("copilot/x.md", "Architecture/skills/hermes/dev/m.md", "Boards/Engineering.md"):
        p = vault / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("body", encoding="utf-8")

    ss.build_index(vault, verbose=False)
    payload = json.loads((vault / ss.INDEX_FILE).read_text())
    assert payload["policy"]["deny_path_prefixes"], "the policy must travel with the index"

    vo._INDEX_CACHE.clear()
    cov = vo.index_coverage(vault)
    assert cov["index"] is True
    assert cov["indexed"] == 6
    assert cov["scanned"] == 6, "excluded notes are not missing coverage"
    assert cov["missing"] == 0
    assert cov["pct_missing"] == 0.0
    assert cov["excluded"] >= 2


def test_coverage_falls_back_when_the_index_has_no_policy(tmp_path, monkeypatch):
    """An index built before this change carries no policy block; behavior there
    must be exactly what it was."""
    sys.path.insert(0, str(REPO_ROOT / "integrations" / "obsidian-mcp-server"))
    import vault_ops as vo

    _fake_embed(monkeypatch)
    vault = _vault_with(tmp_path, 4)
    ss.build_index(vault, verbose=False)
    payload = json.loads((vault / ss.INDEX_FILE).read_text())
    payload.pop("policy")
    (vault / "copilot").mkdir()
    (vault / "copilot" / "x.md").write_text("body", encoding="utf-8")
    (vault / ss.INDEX_FILE).write_text(json.dumps(payload), encoding="utf-8")

    vo._INDEX_CACHE.clear()
    cov = vo.index_coverage(vault)
    assert cov["scanned"] == 5 and cov["missing"] == 1


def test_legacy_single_vector_entries_still_score(tmp_path, monkeypatch):
    """Format-2 per-chunk entries and the old single `vec` shape both rank."""
    _fake_embed(monkeypatch)
    index = {"model": ss.EMBED_MODEL, "format": 2, "notes": {
        "a.md": {"title": "a", "vecs": [[1.0, 0.0]]},
        "b.md": {"title": "b", "vec": [0.0, 1.0]},
    }}
    hits = ss.semantic_search("q", index, limit=2)
    assert {h["path"] for h in hits} == {"a.md", "b.md"}

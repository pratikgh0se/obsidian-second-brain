"""One exclude policy, pinned across every tool that walks the vault.

Six tools each carried a hardcoded set. Only five entries were common to all
six, and the gaps caused real bugs: heal_links rewrote notes inside folders the
health check had excluded, and `Templates` versus `templates` meant the same
folder was skipped or scanned depending on which tool ran.

scripts/vault_scan.py is now the base. Two tools add to it deliberately, and
those additions are asserted here so a future merge cannot quietly erase the
intent behind them.

vault_ops.py keeps its own literal, because the MCP server ships standalone and
must not import from scripts/. That literal is pinned to the base below, which
is the only thing stopping the original drift from happening again.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "eval"))
sys.path.insert(0, str(REPO_ROOT / "integrations" / "obsidian-mcp-server"))


def _lower(s) -> set[str]:
    return {str(x).lower() for x in s}


def test_the_base_covers_every_machine_owned_directory():
    from vault_scan import BASE_EXCLUDE_DIRS

    for d in (".git", ".obsidian", ".trash", "_trash", "_export", "__pycache__",
              "node_modules", ".claude", ".agents", ".codex", ".gemini",
              ".opencode", "templates"):
        assert d in _lower(BASE_EXCLUDE_DIRS), f"{d} dropped out of the base policy"


def test_every_scanning_tool_uses_the_base():
    """The regression this file exists for."""
    import export_okf
    import freshness_lint
    import link_graph
    import vault_health
    import vault_stats
    from vault_scan import BASE_EXCLUDE_DIRS

    base = _lower(BASE_EXCLUDE_DIRS)
    for name, actual in (
        ("vault_health", vault_health.EXCLUDE_DIRS),
        ("link_graph", link_graph.SKIP_DIRS),
        ("export_okf", export_okf.SKIP_DIRS),
        ("vault_stats", vault_stats.EXCLUDED_FOLDERS),
        ("freshness_lint", freshness_lint.SKIP_DIRS),
    ):
        missing = base - _lower(actual)
        assert not missing, f"{name} no longer skips {sorted(missing)}"


def test_the_mcp_server_literal_matches_the_base():
    """vault_ops cannot import from scripts/, so it is pinned instead."""
    import vault_ops
    from vault_scan import BASE_EXCLUDE_DIRS

    missing = _lower(BASE_EXCLUDE_DIRS) - _lower(vault_ops._SKIP_DIRS)
    assert not missing, (
        f"vault_ops._SKIP_DIRS has drifted from scripts/vault_scan.py: missing "
        f"{sorted(missing)}. Update the literal, it ships standalone in the MCP server."
    )


def test_deliberate_per_tool_additions_survive():
    """These encode intent, not oversight, and must not be merged away."""
    import export_okf
    import vault_stats

    assert "excalidraw" in _lower(export_okf.SKIP_DIRS), (
        "export_okf stopped skipping excalidraw; drawings are not exportable prose"
    )
    for d in ("raw", "references"):
        assert d in _lower(vault_stats.EXCLUDED_FOLDERS), (
            f"vault_stats stopped skipping {d}; it is vault content but not a user note"
        )


def test_other_tools_do_not_inherit_those_additions():
    """The flip side: a per-tool skip must stay per-tool."""
    import link_graph
    import vault_health

    for name, actual in (("vault_health", vault_health.EXCLUDE_DIRS),
                         ("link_graph", link_graph.SKIP_DIRS)):
        for d in ("excalidraw", "raw", "references"):
            assert d not in _lower(actual), (
                f"{name} picked up {d!r}, which belongs to a single tool. "
                "Excluding raw/ from the link graph would hide real links."
            )


def test_matching_is_case_insensitive():
    """`Templates` and `templates` must resolve identically.

    The bootstrapper writes capital-T; three tools spelled it lowercase, so the
    same folder was skipped or scanned depending on which tool ran.
    """
    from vault_scan import excluded_dirs, is_excluded

    ex = excluded_dirs()
    assert is_excluded(["Templates", "Daily.md"], ex)
    assert is_excluded(["templates", "Daily.md"], ex)
    assert is_excluded(["TEMPLATES", "Daily.md"], ex)
    assert not is_excluded(["Projects", "Real Note.md"], ex)


def test_lexical_scan_skips_what_the_embed_index_denies():
    """Row 124: the lexical arm must share the embed index's exclusion universe.

    semantic_search.INDEX_POLICY (row 120) denies copilot/, .smart-env/,
    .claude-runs/, _archive/, the Architecture/skills/ mirror prefix and the
    **/skills/<tool>/<role>/<doc>.md pattern - on top of vault_ops._SKIP_DIRS,
    which it already inherits (`from vault_ops import _SKIP_DIRS as SKIP_DIRS`).
    vault_ops cannot import semantic_search (it ships standalone in the MCP
    server), so it carries its own literal copy of the deny list; this test is
    what stops that copy drifting from the source of truth.

    ledger/ and boards/ are deliberately excluded here: INDEX_POLICY denies
    them too (they're machine-generated), but row 124 scopes the lexical fix
    to the skill-mirror/copilot exclusion only - vault_health and the board
    tools read those folders lexically.
    """
    import semantic_search
    import vault_ops

    policy_dir_names = _lower(semantic_search.INDEX_POLICY["deny_dir_names"]) - _lower(
        vault_ops._SKIP_DIRS
    )
    expected_dir_names = {"copilot", ".smart-env", ".claude-runs", "_archive"}
    assert policy_dir_names == expected_dir_names, (
        "semantic_search.INDEX_POLICY['deny_dir_names'] changed shape; update "
        "the lexical-arm-only assumption this test and vault_ops._LEXICAL_DENY_DIR_NAMES "
        "encode, or add the new entry to both"
    )
    assert _lower(vault_ops._LEXICAL_DENY_DIR_NAMES) == expected_dir_names, (
        f"vault_ops._LEXICAL_DENY_DIR_NAMES has drifted from "
        f"semantic_search.INDEX_POLICY['deny_dir_names']: expected {sorted(expected_dir_names)}, "
        f"got {sorted(_lower(vault_ops._LEXICAL_DENY_DIR_NAMES))}"
    )

    policy_path_prefixes = _lower(semantic_search.INDEX_POLICY["deny_path_prefixes"])
    non_generated_prefixes = {p for p in policy_path_prefixes if p not in ("ledger/", "boards/")}
    assert non_generated_prefixes == {"architecture/skills/"}, (
        "semantic_search.INDEX_POLICY['deny_path_prefixes'] changed shape; "
        "vault_ops._LEXICAL_DENY_PATH_PREFIXES assumes only the skill mirror "
        "prefix carries over to the lexical arm"
    )
    assert _lower(vault_ops._LEXICAL_DENY_PATH_PREFIXES) == non_generated_prefixes, (
        "vault_ops._LEXICAL_DENY_PATH_PREFIXES has drifted from "
        "semantic_search.INDEX_POLICY['deny_path_prefixes']"
    )
    # ledger/ and boards/ must stay lexically searchable.
    assert "ledger/" not in _lower(vault_ops._LEXICAL_DENY_PATH_PREFIXES)
    assert "boards/" not in _lower(vault_ops._LEXICAL_DENY_PATH_PREFIXES)

    assert vault_ops._LEXICAL_MIRROR_RE.pattern == semantic_search.INDEX_POLICY[
        "deny_path_patterns"
    ][0], "vault_ops's skill-mirror regex has drifted from semantic_search's"


def test_iter_notes_actually_skips_copilot_and_mirror(tmp_path):
    """End-to-end: _iter_notes must not yield the folders row 124 denies."""
    import vault_ops

    vault = tmp_path
    (vault / "copilot").mkdir()
    (vault / "copilot" / "scaffold.md").write_text("# scaffold\n")
    (vault / ".smart-env").mkdir()
    (vault / ".smart-env" / "index.md").write_text("# index\n")
    (vault / ".claude-runs").mkdir()
    (vault / ".claude-runs" / "run.md").write_text("# run\n")
    (vault / "_archive").mkdir()
    (vault / "_archive" / "old.md").write_text("# old\n")
    (vault / "Architecture" / "skills" / "tool" / "role").mkdir(parents=True)
    (vault / "Architecture" / "skills" / "tool" / "role" / "doc.md").write_text("# doc\n")
    # Ledger/ and Boards/ must survive.
    (vault / "Ledger").mkdir()
    (vault / "Ledger" / "log.md").write_text("# log\n")
    (vault / "Boards").mkdir()
    (vault / "Boards" / "Engineering.md").write_text("# board\n")
    (vault / "Projects").mkdir()
    (vault / "Projects" / "real-note.md").write_text("# real note\n")

    found = {p.relative_to(vault).as_posix() for p in vault_ops._iter_notes(vault)}
    assert found == {"Ledger/log.md", "Boards/Engineering.md", "Projects/real-note.md"}


def test_is_excluded_expects_relative_parts():
    """Absolute parts let an ancestor outside the vault disable the scan.

    That is B29: vault_health's empty-folder check passed absolute parts, so a
    vault living under any directory named Templates reported zero findings.
    """
    from vault_scan import excluded_dirs, is_excluded

    ex = excluded_dirs()
    assert is_excluded(("Users", "someone", "Templates", "myvault"), ex), (
        "sanity: absolute parts DO match, which is exactly why callers must "
        "pass vault-relative parts"
    )
    assert not is_excluded(("Projects", "note.md"), ex)

"""A slash command named in the docs has to exist as a file in `commands/` (#304).

`SKILL.md` told readers to run `/obsidian-setup`, which no adapter builds and
`install.sh` never installs, because `commands/obsidian-setup.md` was never written.
The README's comparison table named seven more by a short form (`/emerge`, `/world`,
`/ingest`) that nothing answers either. A reader who types one of those gets silence
and cannot tell a broken install from a wrong line, and a third-party survey of this
repo repeated `/obsidian-setup` as an install step, so the wrong name travels.

Only backticked names are checked. `/path/to/vault`, `</summary>` and a bare `/dist`
are not commands, and quoting is the one signal in the source that separates a command
from a path or an HTML tag.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = ["SKILL.md", "README.md"]

COMMAND_RE = re.compile(r"`(/[a-z][a-z0-9-]*)`")

# Names in these docs that are real, and are not commands of this repo.
NOT_OURS = {
    # Claude Code's own.
    "/plugin", "/schedule",
    # Other platforms', named while describing their builds.
    "/connect",   # OpenCode's authentication command
    "/skills",    # Codex's skill listing
    "/hooks",     # Codex's native hook trust review
}

# Scheduled agents, documented under "### `obsidian-<name>`" in SKILL.md. They run on
# a timer, not from `commands/`, and the docs name them with a leading slash where a
# user would type one into a scheduler.
SCHEDULED_AGENTS = {
    "/obsidian-morning", "/obsidian-nightly", "/obsidian-weekly", "/obsidian-health-check",
}

# Commands that were consolidated into another one. Every mention is in a sentence
# that says so ("the former `/obsidian-adr` is now `/obsidian-decide --formal`"), which
# is the one case where naming a command that no longer exists is the point.
CONSOLIDATED = {
    "/obsidian-adr", "/obsidian-agenda", "/obsidian-meeting", "/obsidian-schedule",
}

EXEMPT = NOT_OURS | SCHEDULED_AGENTS | CONSOLIDATED


def existing_commands() -> set[str]:
    return {f"/{p.stem}" for p in (REPO_ROOT / "commands").glob("*.md")}


def test_every_command_named_in_the_docs_exists() -> None:
    commands = existing_commands()
    assert commands, "commands/ has no .md files; this test is reading the wrong path"

    missing = []
    for doc in DOCS:
        text = (REPO_ROOT / doc).read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for name in COMMAND_RE.findall(line):
                if name in commands or name in EXEMPT:
                    continue
                missing.append(f"{doc}:{lineno}: `{name}` has no commands/{name[1:]}.md")

    assert missing == [], (
        "the docs name slash commands that do not exist. Fix the name, create the "
        "command, or add it to an exemption set in this file with the reason:\n  "
        + "\n  ".join(missing)
    )


def test_the_exemptions_do_not_hide_a_command_that_exists() -> None:
    """An exemption for a name that later became a real command would stop this fence
    checking it. Whichever list it sits in, the claim is that no file answers it."""
    overlap = sorted(EXEMPT & existing_commands())
    assert overlap == [], (
        "these names are exempted but do exist as commands, so remove them from the "
        f"exemption sets: {', '.join(overlap)}"
    )


def test_the_matcher_ignores_paths_and_html() -> None:
    """The regex is the whole fence. If it starts matching `</summary>` or `/path/to`,
    the failure is a wall of false positives and the next person deletes the test."""
    assert COMMAND_RE.findall("run `/obsidian-save` now") == ["/obsidian-save"]
    assert COMMAND_RE.findall("see `/path/to/vault` and </summary> and /dist") == []
    assert COMMAND_RE.findall("`/Obsidian-Save`") == []

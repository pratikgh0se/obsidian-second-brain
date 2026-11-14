"""Session archive fence: raw chats kept, tool output and secrets not.

All transcripts are synthetic. Fake secrets are built by concatenation so no
secret scanner mistakes this file for a leak.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import session_archive as sa  # noqa: E402

FAKE_ANTHROPIC = "sk-" + "ant-" + "A1b2C3d4" * 4
FAKE_GITHUB = "ghp" + "_" + "Z9y8X7w6" * 5


def entry(kind, content, n, **extra):
    return {"type": kind, "uuid": f"u{n}", "sessionId": "abcd1234-0000-0000-0000-000000000000",
            "cwd": "/work/demo", "timestamp": f"2026-10-28T10:{n:02d}:00Z",
            "message": {"role": kind, "content": content}, **extra}


def write_transcript(path: Path, entries) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")
    return path


def sample(tmp_path: Path) -> Path:
    return write_transcript(tmp_path / "t.jsonl", [
        {"type": "ai-title", "aiTitle": "Demo chat"},
        entry("user", "<system-reminder>injected rules</system-reminder>My commute is 45 minutes", 1),
        entry("assistant", [{"type": "thinking", "thinking": "private reasoning"},
                            {"type": "text", "text": "Noted."},
                            {"type": "tool_use", "name": "Bash", "input": {"command": "env"}}], 2),
        entry("user", [{"type": "tool_result", "content": "OPENAI_API_KEY=leaked"}], 3),
        entry("assistant", [{"type": "text", "text": "Saved it."}], 4),
        entry("user", "subagent chatter", 5, isSidechain=True),
        entry("user", "skill body", 6, isMeta=True),
        entry("user", f"my key is {FAKE_ANTHROPIC} and token {FAKE_GITHUB}", 7),
    ])


def test_answers_to_agent_questions_are_kept(tmp_path):
    path = write_transcript(tmp_path / "q.jsonl", [
        entry("user", "pick a title", 1),
        entry("assistant", [{"type": "tool_use", "name": "AskUserQuestion", "input": {}}], 2),
        entry("user", [{"type": "tool_result", "content": "The user answered: ..."}], 3,
              toolUseResult={"questions": [], "answers": {"Which title?": "neither, too vague"}}),
    ])
    text = "\n".join(t["text"] for t in sa.read_claude_code(path)["turns"])
    assert "Which title?" in text and "neither, too vague" in text
    assert "The user answered" not in text


def test_reader_keeps_only_what_was_said(tmp_path):
    session = sa.read_claude_code(sample(tmp_path))
    text = "\n".join(t["text"] for t in session["turns"])
    assert "My commute is 45 minutes" in text
    assert "Noted." in text and "Saved it." in text
    for dropped in ("injected rules", "private reasoning", "leaked", "subagent chatter",
                    "skill body", '"command"'):
        assert dropped not in text
    # Consecutive agent blocks merge into one turn; the tool result between them is dropped.
    assert [t["role"] for t in session["turns"]] == ["user", "agent", "user"]
    assert session["title"] == "Demo chat"


def test_secrets_are_redacted_before_writing(tmp_path, monkeypatch):
    monkeypatch.setenv(sa.ENABLE_KEY, "1")
    result = sa.archive(sample(tmp_path), tmp_path / "vault")
    note = (tmp_path / "vault" / result["paths"][0]).read_text(encoding="utf-8")
    assert FAKE_ANTHROPIC not in note and FAKE_GITHUB not in note
    assert "[REDACTED:api-key]" in note and "[REDACTED:github-token]" in note
    assert result["redactions"] == 2
    assert "redactions: 2" in note


def test_redact_env_and_bearer_shapes():
    text, n = sa.redact("export MY_SERVICE_TOKEN=abcdef123456 and Authorization: Bearer "
                        + "q" * 24)
    assert "abcdef123456" not in text and "q" * 24 not in text
    assert n == 2
    text, n = sa.redact('{"api_key": "' + "k" * 20 + '"}')
    assert "k" * 20 not in text and n == 1
    # Facts that merely look secret-adjacent must survive.
    for keep in ("the commit is 0e339f7 and max_tokens: 1000",
                 '"Action": "secretsmanager:GetSecretValue"',
                 "set OPENROUTER_API_KEY in your env"):
        clean, n = sa.redact(keep)
        assert n == 0 and clean == keep


def test_note_is_a_raw_source_capture(tmp_path):
    result = sa.archive(sample(tmp_path), tmp_path / "vault")
    assert result["paths"] == ["raw/sessions/2026-10-28 - Demo chat (abcd1234) part 1.md"]
    note = (tmp_path / "vault" / result["paths"][0]).read_text(encoding="utf-8")
    for line in ("type: source", "source_type: session", "capture_scope: bounded-local",
                 "ai-first: true", 'session_id: "abcd1234-0000-0000-0000-000000000000"'):
        assert line in note


def test_second_save_refreshes_the_same_file(tmp_path):
    path = sample(tmp_path)
    vault = tmp_path / "vault"
    first = sa.archive(path, vault)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"type": "ai-title", "aiTitle": "Renamed later"}) + "\n")
        fh.write(json.dumps(entry("user", "one more thing", 8)) + "\n")
    second = sa.archive(path, vault)
    assert second["paths"] == first["paths"]
    files = list((vault / "raw" / "sessions").iterdir())
    assert len(files) == 1
    assert "one more thing" in files[0].read_text(encoding="utf-8")


def test_long_chat_splits_into_linked_parts_that_embed_whole(tmp_path):
    turns = []
    for n in range(1, 41):
        kind = "user" if n % 2 else "assistant"
        content = f"fact number {n} " + "x" * 900
        turns.append(entry(kind, content if kind == "user" else [{"type": "text", "text": content}], n))
    turns.append(entry("user", "y" * 20000, 50))  # one turn longer than a part
    vault = tmp_path / "vault"
    result = sa.archive(write_transcript(tmp_path / "long.jsonl", turns), vault)
    notes = [(vault / p).read_text(encoding="utf-8") for p in result["paths"]]
    assert len(notes) > 3
    joined = "\n".join(notes)
    for n in range(1, 41):
        assert f"fact number {n} " in joined
    assert joined.count("y" * 100) >= 20000 // 100 - len(notes)
    for i, note in enumerate(notes, 1):
        body = note.split("\n---\n", 1)[1]
        assert len(body) < 9000, "part would overflow the 8-chunk embed cap"
        assert f"part: {i}" in note and f"parts: {len(notes)}" in note
    assert "next: [[" in notes[0] and "previous: [[" in notes[-1]


def test_unsplit_file_from_earlier_version_is_replaced(tmp_path):
    vault = tmp_path / "vault"
    sessions = vault / "raw" / "sessions"
    sessions.mkdir(parents=True)
    old = sessions / "2026-10-28 - Old title (abcd1234).md"
    old.write_text('---\nsession_id: "abcd1234-0000-0000-0000-000000000000"\n---\n',
                   encoding="utf-8")
    other = sessions / "2026-10-28 - Someone else (abcd1234).md"
    other.write_text('---\nsession_id: "abcd1234-ffff"\n---\n', encoding="utf-8")
    result = sa.archive(sample(tmp_path), vault)
    assert result["paths"] == ["raw/sessions/2026-10-28 - Old title (abcd1234) part 1.md"]
    assert not old.exists() and other.exists()


def test_latest_picks_the_newest_session_for_the_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "cc"))
    folder = tmp_path / "cc" / "projects" / "-work-demo-app"
    old = write_transcript(folder / "old.jsonl", [entry("user", "old", 1)])
    new = write_transcript(folder / "new.jsonl", [entry("user", "new", 2)])
    os.utime(old, (1_000_000, 1_000_000))
    assert sa.latest_transcript("/work/demo.app") == new
    assert sa.latest_transcript("/nowhere") is None


def test_flag_off_writes_nothing(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    env = {k: v for k, v in os.environ.items() if k != sa.ENABLE_KEY}
    env["OBSIDIAN_ENV_FILE"] = str(tmp_path / "no-config.env")
    out = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "session_archive.py"),
         "--transcript", str(sample(tmp_path)), "--vault", str(vault), "--json"],
        capture_output=True, text=True, env=env, check=True)
    assert json.loads(out.stdout)["status"] == "off"
    assert not (vault / "raw").exists()

# Obsidian Second Brain MCP server

An MCP server that turns an Obsidian vault into a set of tools any MCP client can call - [Hermes Agent](https://github.com/NousResearch/hermes-agent) (via `discover_mcp_tools()`), Claude Desktop, Claude Code, or Cursor.

This is the connector half of [Issue #60](https://github.com/eugeniughelbur/obsidian-second-brain/issues/60): the agent gets a doorway to **use** your vault as a knowledge second brain (search it, read it, add to it) **without** the vault becoming the agent's own behavioral memory. Those stay two distinct things, as requested.

## Status

v0, live-tested at the protocol level. The vault logic (`vault_ops.py`) is pure stdlib and unit-tested, and the full MCP round-trip (a real client connecting over stdio, discovering tools, and calling search / read / save) passes via `live_test.py`. The one thing not yet done is driving it from an actual Hermes instance - see "Testing" below.

## Tools exposed

Data tools (deterministic primitives):

| Tool | What it does |
|---|---|
| `obsidian_search(query, limit=6)` | Ranked keyword search across vault notes; returns snippets + paths |
| `obsidian_read_note(path)` | Read a full note by vault-relative path (path-traversal guarded) |
| `obsidian_save_note(title, content, type, tags)` | Save a new AI-first note to the vault `Inbox/`; the result also reports validation, the index entry, the log line and the post-write command (see Bookkeeping) |
| `obsidian_capture(text, tags)` | Quick-capture an idea as a lightweight `type: idea` note, with the same bookkeeping |

Curator tools (guarded mutation + graph + health, per Issue #79):

| Tool | What it does |
|---|---|
| `obsidian_update_note(path, append, heading, set_fields)` | Guarded edit of an existing note: append a section and/or merge scalar frontmatter; preserves the rest verbatim, never creates, never touches `tags:` blocks, stamps `updated` |
| `obsidian_validate_note(path)` | Check a note for AI-first compliance (frontmatter keys, `## For future agent` preamble) and unresolved `[[wikilinks]]` |
| `obsidian_backlinks(target)` | List every note that links to `target` via `[[wikilink]]` |
| `obsidian_vault_health()` | Bounded structural summary: orphans, wanted notes (linked but unwritten - a wishlist, not errors), notes missing frontmatter (counts + capped samples) |

Skill tools (the higher-level behaviors, per Issue #60 - "use the skills, not just file search"):

| Tool | What it does |
|---|---|
| `obsidian_list_skills()` | List the obsidian-second-brain commands available as skills (name + description) |
| `obsidian_get_skill(name)` | Return a command's playbook (step-by-step instructions) for the agent to execute, using the data tools above for actual vault I/O |

The skill tools expose the command playbooks (e.g. `obsidian-ingest`, `idea-discovery`, `obsidian-find`) so the connecting agent runs the real skill behavior with its own model - ingest, for instance, is multi-step (it rewrites and links existing pages), so it runs as an agent-executed skill rather than a single function. Niche / agent-only / Claude-only commands (challenge, health, the scheduled agents, and the Google Calendar commands) are excluded from the exposed set. Override the commands source with `OBSIDIAN_COMMANDS_DIR` if the server is deployed away from the repo.

Saved notes follow `references/ai-first-rules.md` (frontmatter, `## For future agent` preamble, `source: mcp` marker) so connector-written notes are distinguishable from hand-authored ones.

## Bookkeeping after writes

Every successful write (`save_note`, `capture`, `update_note`, `replace_text`, `move_note`) is followed by the vault's own bookkeeping, done by the server so an agent that may only write to `Inbox/` (any agent working from another project) still leaves the vault consistent:

- **validation** - the note is checked with the same rules as `obsidian_validate_note`; reported as `"validation": {"ok": true, "issues": []}`.
- **index** - a new note gets `- [[note]] - summary` under `index.md`'s section for its folder (`## Inbox/` for captures). Index layouts differ per vault, so a missing section is reported (`"index.md has no '## Inbox/' section; no entry added"`), never guessed.
- **log** - one line in the vault's operation-log convention: `**HH:MM** - capture | Title -> [[Inbox/...]] (tags: ...)` in `Logs/YYYY-MM-DD.md` when that folder exists (as `/obsidian-init` creates it, with the log frontmatter on a new day), otherwise `## [YYYY-MM-DD] capture | ...` appended to `log.md`. Reported as `"log": "Logs/2026-09-05.md"`.
- **post_write** - optional. Set `OBSIDIAN_POST_WRITE_CMD` to a command and the server runs it after the bookkeeping as `<cmd> <vault-path> <note-path> <action>`, from the vault directory, bounded by `OBSIDIAN_POST_WRITE_TIMEOUT` (seconds, default 45). This is where a git commit-and-push belongs: the server never touches git itself. Reported as `"post_write": {"ran": true, "ok": true, "detail": "ok"}`; a failure or timeout is reported, never raised, and the note stays saved. The command string is split shell-style (`shlex`); on Windows prefer a path without spaces or a small wrapper script.

Each part has its own key so `saved` alone never implies the others happened. Writes to `Logs/`, `log.md` and `index.md` themselves are never logged (no loops). `OBSIDIAN_BOOKKEEPING=0` switches validation, index and log off; the post-write command is independent of that switch.

## Scoping a mounted connection: two fences

A client cannot disable individual tools on a server it mounts, so a profile that should only touch part of the vault has to be scoped here. Two independent env vars do that, one per direction. Both take **colon-separated, vault-relative folder prefixes**, both match a **whole path component, case-sensitively** (`Knowledge` covers `Knowledge/x.md` and `Knowledge/sub/x.md`, never `Knowledgebase/x.md` or `knowledge/x.md`), and in both an entry that is only slashes is dropped rather than treated as "everything".

### Write fence

`OBSIDIAN_MCP_WRITE_ALLOW` names the folders this connection may write (agent-ops decision row 93). It gates every write tool: `save_note`, `capture`, `update_note`, `replace_text`, `move_note`.

- **unset** - unrestricted, what Claude Desktop / Claude Code / Cursor see
- **set** - e.g. `OBSIDIAN_MCP_WRITE_ALLOW="Inbox/:Knowledge/"`
- **empty** - read-only: no write tool can name a path

A refused write returns an error naming the allowed prefixes and changes nothing on disk.

### Read fence

`OBSIDIAN_MCP_READ_ALLOW` names the folders this connection may **read** (Capsule Corp decision row 125). Same syntax, same matching rules, applied to every read path the server exposes:

| Tool | Function | How the fence applies |
|---|---|---|
| `obsidian_search` | `search` (+ `_semantic_fuse`) | out-of-scope notes are dropped **in the walk**, before the scan cap and before `limit`, and the semantic arm filters the embedding index before its fusion slice - so a fenced note never occupies a result slot and its existence cannot be inferred from result counts |
| `obsidian_read_note` | `read_note` | a clear error naming the readable scope, never an empty string or a bare not-found; checked on the *resolved* path, so `Knowledge/../Decisions/x.md` is fenced by where it lands |
| `obsidian_backlinks` | `backlinks` | only readable notes are reported as referrers |
| `obsidian_vault_health` | `vault_health` (via `_stem_index`) | counts and samples cover readable notes only |
| `obsidian_validate_note` | `validate_note` | fenced: its issue list reports frontmatter keys and every unresolved wikilink, which is note content by proxy |

The single choke point is `_iter_notes`, which every scanning read path goes through; it takes an `allow=` prefix filter and applies the env fence on top of it.

- **unset or empty-valued-and-absent** - **allow everything, i.e. exactly today's behaviour.** Nothing changes for any deployed client until a profile sets the variable.
- **set** - e.g. `OBSIDIAN_MCP_READ_ALLOW="Specs/:Decisions/"`
- **empty** (`OBSIDIAN_MCP_READ_ALLOW=""`) - this connection may read no notes (the mirror image of an empty write fence, which is a read-only connection)

The two fences are independent and compose: a read fence never restricts writes and a write fence never restricts reads, so `READ_ALLOW="Decisions/"` with `WRITE_ALLOW="Knowledge/"` is a role that reads decisions and files its output elsewhere.

Not covered, deliberately: `obsidian_list_skills` / `obsidian_get_skill` read command playbooks from the repo (`OBSIDIAN_COMMANDS_DIR`), not vault notes. Neither fence is a substitute for filesystem permissions either - a client that also has direct file tools on the vault path can read around both.

**The per-role folder sets are not defined here.** Which folders each Hermes profile (`dev`, `reviewer`, `qa`, `lead`, `researcher`, ...) may read is decided in Capsule Corp's C4 tick table; this repo ships the mechanism and its allow-all default only.

### Companion: the `folders` parameter on search

`obsidian_search(query, limit, folders=["Specs/", "Decisions/"])` restricts one call to those prefixes, using the same prefix syntax. It only ever **narrows**: the effective scope is the intersection of `folders` with the read fence, so naming a fenced folder returns nothing rather than widening anything. `folders=None` (the default) means the fence alone decides; an explicit empty list, or a list of only degenerate entries, names no folder and returns nothing - an explicit narrowing is never silently read as "no filter".

## Run it

Requires the vault path in the environment and the `mcp` package:

```bash
export OBSIDIAN_VAULT_PATH="/path/to/your/vault"
uv run --no-project --with 'mcp<2' python integrations/obsidian-mcp-server/server.py
```

## Wire it into a client

Hermes Agent and most MCP clients take a launch command. Example client config entry:

```json
{
  "mcpServers": {
    "obsidian-second-brain": {
      "command": "uv",
      "args": ["run", "--no-project", "--with", "mcp<2", "python", "/abs/path/integrations/obsidian-mcp-server/server.py"],
      "env": { "OBSIDIAN_VAULT_PATH": "/path/to/your/vault" }
    }
  }
}
```

For Hermes specifically, add the server to its MCP config; Hermes picks the tools up through `discover_mcp_tools()` with zero Hermes-specific code. The same server works unchanged in Claude Desktop / Claude Code / Cursor.

## Testing

`vault_ops.py` is covered by a standalone harness (search / read / save / path-guard) - no `mcp` install needed. `live_test.py` runs the full MCP round-trip with a real client:

```bash
# read-only (safe against a real vault)
OBSIDIAN_VAULT_PATH=/path/to/vault uv run --no-project --with 'mcp<2' python live_test.py "your query"
# also write one test note to Inbox/
OBSIDIAN_VAULT_PATH=/path/to/vault uv run --no-project --with 'mcp<2' python live_test.py --save "your query"
```

Live-test checklist:
  - [x] Server starts and an MCP client completes the handshake.
  - [x] Client lists the three tools.
  - [x] Client calls `obsidian_search` (results), `obsidian_read_note` (content), `obsidian_save_note` (writes a valid AI-first note to `Inbox/`). Verified 2026-06-06 via `live_test.py` against both a throwaway vault and a real vault (read-only).
  - [ ] Connect from a real Hermes instance and confirm the tools appear via `discover_mcp_tools()`.

## Notes

- Search is a bounded linear scan (good for small/medium vaults; large vaults want an index).
- `vault_ops.py` is intentionally dependency-free and overlaps with the memory-provider integration; the two are separate artifacts and can later share a common module if both are kept.

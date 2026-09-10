"""Local semantic search for the vault - meaning-based retrieval, nothing leaves the machine.

This is the "map of meaning" layer: it asks a LOCAL embedding model (via Ollama,
running on your own computer) for each note's coordinates, caches them, and finds
the notes whose meaning is nearest a query - even when they share no words with it.
It is the answer to the ~17% ceiling the lexical eval exposed on paraphrased queries.

Design choices that matter:
- **Local only.** Embeddings come from Ollama on localhost. Note text never leaves
  the machine, so it is safe for private notes (the firewall in _CLAUDE.md).
- **Privacy carve-out.** Folders listed in OBSIDIAN_EMBED_EXCLUDE (comma-separated
  path prefixes) are never embedded at all - belt and braces even though it is local.
- **Pure stdlib.** Cosine similarity is hand-rolled; no numpy/torch dependency in the
  repo. The model lives in Ollama, not in Python.
- **Cached.** The index is a JSON file keyed by note path + content hash, so only
  changed notes are re-embedded on the next run.
- **Default off.** Nothing calls this unless explicitly run; vault_ops.search stays
  the shipped default until the eval proves hybrid beats lexical.

Requires Ollama (https://ollama.com) with an embedding model pulled:
    ollama pull bge-m3
Configure via env: OLLAMA_URL (default http://localhost:11434),
OBSIDIAN_EMBED_MODEL (default bge-m3, multilingual),
OBSIDIAN_EMBED_EXCLUDE (default empty; e.g. "wiki/private/,Journal,Private").
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
EMBED_MODEL = os.environ.get("OBSIDIAN_EMBED_MODEL", "bge-m3")
# Backend selects how embeddings are produced:
#   "ollama" (default) - local Ollama at OLLAMA_URL, fully private/offline.
#   "openai" - ANY OpenAI-compatible /v1/embeddings endpoint, so users without
#              Ollama can point at another local runtime (LM Studio, llama.cpp)
#              or a cloud API (OpenAI, a gateway). Set OBSIDIAN_EMBED_URL (base) and
#              OBSIDIAN_EMBED_KEY (if the endpoint needs auth). Cloud = text leaves
#              the machine, so keep the OBSIDIAN_EMBED_EXCLUDE carve-out in mind.
EMBED_BACKEND = os.environ.get("OBSIDIAN_EMBED_BACKEND", "ollama").lower()
EMBED_URL = os.environ.get("OBSIDIAN_EMBED_URL", OLLAMA_URL).rstrip("/")
EMBED_KEY = os.environ.get("OBSIDIAN_EMBED_KEY", "")
EXCLUDE_PREFIXES = tuple(
    p.strip() for p in os.environ.get("OBSIDIAN_EMBED_EXCLUDE", "").split(",") if p.strip()
)
# Per-vault escape hatch: prefixes here are indexed even if INDEX_POLICY denies
# them (comma-separated, vault-relative), e.g. "Architecture/skills/claude-code".
ALLOW_PREFIXES_ENV = tuple(
    p.strip() for p in os.environ.get("OBSIDIAN_EMBED_ALLOW", "").split(",") if p.strip()
)
# Single source of truth: the MCP server owns the skip set, so the semantic
# index and the lexical scan can never drift into different universes
# (stress-test fix 10/24).
import sys as _sys

_sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "integrations" / "obsidian-mcp-server"))
from vault_ops import _SKIP_DIRS as SKIP_DIRS  # noqa: E402

INDEX_FILE = ".obsidian-semantic-index.json"  # written at vault root

# --------------------------------------------------------------------------- #
# Chunk geometry - sized to the MODEL'S window, not to a guess
# --------------------------------------------------------------------------- #
# `bge-m3` (the default and the model named in every index built so far) accepts
# inputs "from short sentences to long documents of up to 8,192 tokens"
# (M3-Embedding, arXiv:2402.03216, 2024-02-05, rev. 2025-12-12), so a 6,000-char
# chunk FITS the model. On 2026-09-09 that fact alone moved the geometry from
# 1,200 x 8 to 6,000 x 64, to stop the 1,200 x 8 ceiling discarding 45% of vault
# text (464 of 950 notes exceeded 9,600 embedded chars).
#
# Then it was measured. On the same 29-case set on the same vault, 2026-09-10
# (Capsule decision row 128, C4 task 1): the 6,000 x 64 chunker scores default
# recall@5 41.4% (MRR 0.295); the 1,200 x 8 chunker on a scratch rebuild of the
# SAME content scores 51.7% (MRR 0.43). Fitting the window is not the same as
# retrieving well: notes score by their best chunk, and a 6,000-char chunk
# averages a whole page of mixed topics into one vector, so the section that
# actually answers the query is diluted by the five that do not. A smaller chunk
# is a sharper unit of relevance.
#
# So the geometry is back at 1,200 x 8 - ten measured points of recall beat a
# coverage argument. The known cost is real and is the reverse trade: a note past
# 9,600 chars is embedded only up to the ceiling. It is no longer SILENT, which
# was the actual 2026-09-09 defect - `embed_note_chunks` names every note that
# reaches the ceiling on stderr, and `--stats` counts them (`at_ceiling`). The
# principled fix for long notes is a per-note chunk budget rather than a bigger
# chunk; until that is measured, this reverts cleanly by editing two numbers
# (the chunker fingerprint below invalidates the cache either way).
_MODEL_WINDOW_TOKENS = 8192          # bge-m3's real input window - see above
_CHUNK_CHARS = 1200                  # ~300-350 tokens: a sharp unit of relevance
_CHUNK_OVERLAP_CHARS = 500           # a fact on a boundary lands in both chunks
                                     # (self-caps at room // 4, so ~300 here)
# The per-note ceiling: 8 x 1,200 = 9,600 embedded chars. A note that reaches it
# is named on stderr and counted by `--stats`, never truncated in silence.
_MAX_CHUNKS = 8
# Never subdivide below this: past it, chunks carry no context worth embedding.
_MIN_CHUNK_CHARS = 200

# --------------------------------------------------------------------------- #
# What gets indexed - ONE config constant (agent-ops decision row 120)
# --------------------------------------------------------------------------- #
# Excluding is as valuable as including: on 2026-09-09 the index held 107 notes,
# 17 of them from `copilot/` - scaffolding the vault's `_CLAUDE.md` forbids every
# agent and job from even scanning - while three quarters of the vault is a
# duplicated skill mirror that puts ten identical candidates in front of every
# query. The ignore rules existed in `report.sh`'s counts and in the eval's
# exclusions but were never enforced in the index builder. They are here now,
# in one place, so the three lists cannot drift apart again.
#
# Evaluation order per note (vault-relative POSIX path):
#   1. allow_path_prefixes  - an explicit allow wins over every deny below
#   2. deny_dir_names       - directory name at ANY depth (machine-owned dirs)
#   3. deny_path_prefixes   - root-anchored folder (the vault's folder map)
#   4. deny_path_patterns   - regex, for shapes a prefix cannot express
#   5. deny_name_suffixes   - filename suffix
# All string comparisons are case-insensitive. This layers ON TOP of
# `vault_ops._SKIP_DIRS`, the canonical skip set shared with the lexical scan.
INDEX_POLICY: dict[str, tuple | set] = {
    # Escape hatch: index this prefix even if a rule below denies it. Empty by
    # default; `OBSIDIAN_EMBED_ALLOW` extends it.
    "allow_path_prefixes": (),
    # Machine-owned directories, skipped wherever they appear. `.obsidian`,
    # `.git`, `_trash`, `templates`, `node_modules`, `__pycache__` and friends
    # arrive from `_SKIP_DIRS`; these are the ones the vault's folder map names
    # that the shared set does not carry.
    "deny_dir_names": {
        "copilot",       # _CLAUDE.md: "plugin scaffolding... NOT knowledge"
        ".smart-env",    # Smart Connections plugin index
        ".claude-runs",  # scheduled-job scratch
        "_archive",      # superseded notes, kept for history only
    },
    # Root-anchored folders from the vault's folder map. Prefixes, not names, so
    # a user's own `Boards/` elsewhere in a different vault is untouched.
    "deny_path_prefixes": (
        "ledger/",              # machine-written milestone JSONL log
        "boards/",              # GENERATED from the kanban every 10 minutes
        "architecture/skills/",  # the skill mirror - see the pattern below
    ),
    # THE SKILL-MIRROR PATTERN, documented because it is the single biggest win
    # available: the skill mirror reproduces the DEPLOYED LAYOUT rather than the
    # content, so 72 unique upstream docs exist once per role directory -
    # `Architecture/skills/<tool>/<role>/<doc>.md`, ten role dirs, 719 files,
    # 76% of the vault's notes and 64% of its words, every copy identical and
    # every copy a candidate in every search. The pattern is
    # `**/skills/<tool>/<role>/<doc>.md`: a `skills/` directory with a note
    # exactly two levels below it. The `architecture/skills/` prefix above
    # catches the mirror where it lives today; this pattern catches it if it is
    # ever mounted somewhere else.
    "deny_path_patterns": (
        r"(?:^|/)skills/[^/]+/[^/]+/[^/]+\.md$",
    ),
    # Drawings are raw JSON, not prose: they bloat the index and fail embedding.
    "deny_name_suffixes": (".excalidraw.md",),
}


# --------------------------------------------------------------------------- #
# Ollama (local) embedding calls
# --------------------------------------------------------------------------- #
def ollama_available() -> bool:
    """Is the embedding backend reachable? (Name kept for callers.)"""
    if EMBED_BACKEND == "openai":
        return bool(EMBED_URL)  # assume configured endpoint is up; embed() falls back on error
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


_RETRY_WAITS = (1, 3, 8, 15)  # a local model on a laptop can briefly 500 under rapid load


def _embed_request(text: str, model: str | None = None) -> tuple[str, bytes, dict]:
    """Build the (url, body, headers) for the configured backend."""
    if EMBED_BACKEND == "openai":
        headers = {"Content-Type": "application/json"}
        if EMBED_KEY:
            headers["Authorization"] = f"Bearer {EMBED_KEY}"
        body = json.dumps({"model": (model or EMBED_MODEL), "input": text[:_CHUNK_CHARS]}).encode()
        return f"{EMBED_URL}/v1/embeddings", body, headers
    # ollama (default): keep_alive holds the model in memory between calls
    body = json.dumps({"model": (model or EMBED_MODEL), "prompt": text[:_CHUNK_CHARS], "keep_alive": "15m"}).encode()
    return f"{EMBED_URL}/api/embeddings", body, {"Content-Type": "application/json"}


def _parse_embedding(data: dict) -> list[float] | None:
    """Pull the vector out of either response shape."""
    if data.get("embedding"):                       # ollama
        return data["embedding"]
    items = data.get("data")                         # openai-compatible
    if items and isinstance(items, list) and items[0].get("embedding"):
        return items[0]["embedding"]
    return None


def embed(text: str, retries: int | None = None, model: str | None = None) -> list[float]:
    """Return the embedding vector for one text via the configured backend.

    Retries transient errors (HTTP 5xx, connection resets): a local model on a
    laptop can buckle under rapid sequential calls, then recover a second later.
    The last failure is raised so the caller can skip the note. retries caps the
    ladder: the adaptive splitter passes 1, because a deterministic too-many-
    tokens failure repays every retry with the same 500 and the full ladder at
    every split level turned an 11-note repair into a 10-minute stall.
    """
    url, body, headers = _embed_request(text, model)
    last_err: Exception | None = None
    waits = _RETRY_WAITS if retries is None else _RETRY_WAITS[:retries]
    for attempt in range(len(waits) + 1):
        try:
            req = urllib.request.Request(url, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                vec = _parse_embedding(json.loads(r.read()))
            if vec:
                return vec
            last_err = RuntimeError(f"backend returned no embedding (model '{EMBED_MODEL}')")
        except (urllib.error.HTTPError, urllib.error.URLError, ConnectionError, TimeoutError) as e:
            last_err = e
        if attempt < len(waits):
            time.sleep(waits[attempt])
    raise RuntimeError(
        f"Embedding backend '{EMBED_BACKEND}' at {EMBED_URL} failed after retries ({last_err})."
    )


def _mean_pool(vectors: list[list[float]]) -> list[float]:
    """Average several chunk vectors into one note vector (component-wise)."""
    if not vectors:
        return []
    if len(vectors) == 1:
        return vectors[0]
    dim = len(vectors[0])
    n = len(vectors)
    return [sum(v[i] for v in vectors) / n for i in range(dim)]


_FM_RE = re.compile(r"^---\n(.*?)\n---\n?", re.DOTALL)
_FM_LIST_RE = re.compile(r"^(aliases|related-people|related-projects|tags):\s*\[(.+?)\]\s*$", re.MULTILINE)
_FM_TYPE_RE = re.compile(r"^type:\s*[\"\']?([A-Za-z0-9_-]+)", re.MULTILINE)


def prepare_note_text(stem: str, text: str) -> tuple[str, str]:
    """Return (identity_header, cleaned_body) for embedding.

    The header names the note (title, type, aliases, related people/projects) so
    every chunk stays reachable by described role, not just by title words. The
    body drops frontmatter and empty template sections - a daily note that is
    mostly unfilled scaffolding must not have its one real paragraph diluted by
    boilerplate headings (stress-test fix 13/24)."""
    fm = ""
    m = _FM_RE.match(text)
    if m:
        fm = m.group(1)
        text = text[m.end():]
    bits = [stem]
    tm = _FM_TYPE_RE.search(fm)
    if tm:
        bits.append(tm.group(1))
    for _, items in _FM_LIST_RE.findall(fm):
        bits.extend(i.strip().strip("\"\'[]") for i in items.split(",") if i.strip())
    header = " | ".join(dict.fromkeys(b for b in bits if b)) + "\n"
    # Drop sections that contain no prose: a heading directly followed by another
    # heading (or EOF) is template scaffolding, not content.
    lines = text.splitlines()
    kept: list[str] = []
    for i, line in enumerate(lines):
        if line.lstrip().startswith("#"):
            nxt = next((l for l in lines[i + 1:] if l.strip()), "")
            if not nxt or nxt.lstrip().startswith("#"):
                continue
        kept.append(line)
    body = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
    return header, body


def embed_note(text: str) -> list[float]:
    """Embed a whole note into ONE mean-pooled vector (legacy shape; kept for
    compatibility - build_index stores per-chunk vectors via embed_note_chunks)."""
    vecs = embed_note_chunks(text)
    return _mean_pool(vecs) if vecs else []


# --------------------------------------------------------------------------- #
# Structure-aware chunking
# --------------------------------------------------------------------------- #
# Separator ladder, widest structural boundary first. `text[i:i+1200]` used to
# split mid-sentence and mid-table; a note's own headings are free signal, so
# split on them first and only fall back to paragraphs, lines and words.
_SEPARATORS = ("\n# ", "\n## ", "\n### ", "\n#### ", "\n##### ", "\n\n", "\n", " ")


def _split_keeping(text: str, sep: str) -> list[str]:
    """Split on `sep`, re-attaching it to the front of every piece but the first.

    Keeping the separator means the heading stays glued to the section it
    introduces, so a chunk always says which section it came from.
    """
    parts = text.split(sep)
    if len(parts) == 1:
        return [text]
    return [parts[0]] + [sep + p for p in parts[1:]]


def _split_units(text: str, limit: int) -> list[str]:
    """Break `text` into units of at most `limit` chars on structure boundaries.

    Recursive descent down _SEPARATORS: a unit already short enough is returned
    whole; otherwise it is split on the widest separator that actually divides
    it and each piece re-examined. Content with no usable separator (a single
    enormous table row) is hard-sliced as a last resort - never dropped.
    """
    if len(text) <= limit:
        return [text] if text.strip() else []
    for sep in _SEPARATORS:
        # `text[1:]`: a separator at position 0 does not divide anything.
        if sep in text[1:]:
            pieces = _split_keeping(text, sep)
            if len(pieces) > 1:
                out: list[str] = []
                for p in pieces:
                    out.extend(_split_units(p, limit))
                return out
    return [text[i:i + limit] for i in range(0, len(text), limit)]


def chunk_note_text(text: str, header: str = "") -> list[str]:
    """Split a note body into overlapping, structure-aligned chunks.

    Every character of `text` appears in at least one chunk - the whole point of
    the change: nothing is silently truncated any more. Overlap means some
    characters appear in two, which is deliberate (a fact that straddles a
    section boundary must be retrievable from either side).

    Each chunk is at most `_CHUNK_CHARS - len(header)` characters, so
    header + chunk still fits comfortably inside the model's window.
    """
    text = text.strip()
    if not text:
        return []
    room = max(_MIN_CHUNK_CHARS, _CHUNK_CHARS - len(header))
    overlap = min(_CHUNK_OVERLAP_CHARS, room // 4)
    # Pack to (room - overlap) so the prepended tail cannot push a chunk over.
    pack_limit = max(_MIN_CHUNK_CHARS, room - overlap)
    units = _split_units(text, pack_limit)

    chunks: list[str] = []
    cur = ""
    for u in units:
        if cur and len(cur) + len(u) > pack_limit:
            chunks.append(cur)
            cur = (cur[-overlap:] if overlap else "") + u
        else:
            cur += u
    if cur.strip():
        chunks.append(cur)
    return chunks


def embed_note_chunks(text: str, header: str = "") -> list[list[float]]:
    """Embed a note as per-chunk vectors (stress-test fix 13/24).

    Mean-pooling a long note produced one averaged mumble: a 10-section dossier
    answering a query in section 7 has one strong vector and nine unrelated ones,
    and the average drowns the signal 9-to-1. Chunks are scored independently at
    query time (best chunk wins), so each chunk carries the note's identity
    header (title/type/aliases/related) - a mid-dossier section must still know
    who it is about."""
    chunks = chunk_note_text(text, header=header)
    if len(chunks) > _MAX_CHUNKS:
        # The ceiling is a bug alarm, not a size policy. Say so, by name, on
        # stderr: silent truncation is what this whole change exists to kill.
        print(
            f"  [WARNING] {len(chunks)} chunks exceeds the _MAX_CHUNKS={_MAX_CHUNKS} "
            f"safety ceiling ({len(text)} chars); indexing the first {_MAX_CHUNKS} "
            f"only. Split this note.",
            file=sys.stderr,
        )
        chunks = chunks[:_MAX_CHUNKS]
    vecs: list[list[float]] = []
    for c in chunks:
        vecs.extend(_embed_adaptive(c, header))
    return vecs


def _embed_adaptive(text: str, header: str) -> list[list[float]]:
    """Embed one chunk, halving it on failure (stress-test fix 14/24).

    Retry cures transient failures; these were deterministic: token-dense
    content (a 1,066-char table of euro-rows) blows past the model's 512-token
    window at char counts where prose fits fine, and failed on every build.
    Char count is a bad proxy for tokens (even 'x'*1200 fails), so no fixed
    chunk size is safe - halve until it fits, floor at 300 chars. 6-decimal
    rounding is far beyond cosine's needs and halves the on-disk index."""
    try:
        return [[round(x, 6) for x in embed(header + text, retries=1)]]
    except Exception:
        if len(text) <= 300:
            raise
        mid = len(text) // 2
        return _embed_adaptive(text[:mid], header) + _embed_adaptive(text[mid:], header)


# --------------------------------------------------------------------------- #
# Pure-stdlib vector math
# --------------------------------------------------------------------------- #
def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [-1, 1]; how close two meaning-coordinates point."""
    if not a or not b or len(a) != len(b):
        return 0.0
    # strict=True: a and b are already verified equal-length above, so this
    # never raises here - it documents the invariant for future edits (#164).
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()[:16]


def _excluded(rel: str) -> bool:
    return any(rel == p or rel.startswith(p) for p in EXCLUDE_PREFIXES)


_MIRROR_RES = tuple(re.compile(p, re.IGNORECASE) for p in INDEX_POLICY["deny_path_patterns"])


def chunker_fingerprint() -> str:
    """Identify the chunk geometry, so a change to it invalidates the cache.

    Vectors are only comparable to each other if they were produced from the
    same text. Change the chunk size, the overlap or the ceiling and every
    cached vector describes a different slice of its note - which is why this
    joins `format` and `model` in the cache-validity check.
    """
    return f"c{_CHUNK_CHARS}-o{_CHUNK_OVERLAP_CHARS}-m{_MAX_CHUNKS}"


def policy_payload() -> dict:
    """The effective policy as plain JSON, to be stored in the index file.

    Read by `vault_ops.index_coverage` so the coverage report counts only the
    notes the index is actually meant to hold. Sets become sorted lists so the
    payload is stable and diffable.
    """
    return {
        "allow_path_prefixes": sorted(
            tuple(INDEX_POLICY["allow_path_prefixes"]) + ALLOW_PREFIXES_ENV
        ),
        "deny_dir_names": sorted(set(INDEX_POLICY["deny_dir_names"]) | set(SKIP_DIRS)),
        "deny_path_prefixes": sorted(INDEX_POLICY["deny_path_prefixes"]),
        "deny_path_patterns": sorted(INDEX_POLICY["deny_path_patterns"]),
        "deny_name_suffixes": sorted(INDEX_POLICY["deny_name_suffixes"]),
        "deny_env_prefixes": sorted(EXCLUDE_PREFIXES),
    }


def policy_verdict(rel: str) -> str | None:
    """Why `rel` is not indexed, or None if it is. `rel` is vault-relative POSIX.

    Returns a short reason string (used verbatim in the build report and by the
    tests) so an exclusion is always explainable rather than mysterious.
    """
    low = rel.lower()
    for pref in tuple(INDEX_POLICY["allow_path_prefixes"]) + ALLOW_PREFIXES_ENV:
        if low.startswith(pref.lower().rstrip("/") + "/") or low == pref.lower():
            return None
    parts = low.split("/")
    for pt in parts[:-1]:
        if pt in INDEX_POLICY["deny_dir_names"]:
            return f"denied dir: {pt}/"
        if pt in SKIP_DIRS or pt.endswith("templates"):
            return f"skip dir: {pt}/"
    for pref in INDEX_POLICY["deny_path_prefixes"]:
        if low.startswith(pref):
            return f"denied prefix: {pref}"
    for rx in _MIRROR_RES:
        if rx.search(rel):
            return "skill mirror (**/skills/<tool>/<role>/<doc>.md)"
    for suf in INDEX_POLICY["deny_name_suffixes"]:
        if low.endswith(suf):
            return f"denied suffix: {suf}"
    if _excluded(rel):
        return "OBSIDIAN_EMBED_EXCLUDE"
    return None


# --------------------------------------------------------------------------- #
# Index build / load (cached, incremental)
# --------------------------------------------------------------------------- #
def _iter_notes(vault: Path):
    """Yield every note the index policy allows. Excluded notes never even get
    read, so a forbidden folder costs nothing and cannot leak into a vector."""
    for md in sorted(vault.rglob("*.md")):
        if policy_verdict(md.relative_to(vault).as_posix()) is None:
            yield md


def _iter_all_notes(vault: Path):
    """Yield (path, verdict) for every note, indexed or not - for --stats."""
    for md in sorted(vault.rglob("*.md")):
        yield md, policy_verdict(md.relative_to(vault).as_posix())


# How many newly-embedded notes to accumulate before flushing the index to
# disk. The old builder wrote once at the end, so a run killed by the nightly
# job's wall-clock cap wrote NOTHING - 600+ notes embedded on 2026-09-08 and
# zero kept. Flushing every batch turns a killed run into a resumable one:
# whatever was embedded is on disk and the next run reuses it by hash.
_BATCH_WRITE = 25


def _atomic_write_index(index_path: Path, payload: dict) -> None:
    """Write the index via temp file + rename, so it is never half-written.

    A reader (search, vault_health, the MCP server) either sees the previous
    complete index or the new complete one - never a truncated JSON file. The
    temp file is created in the SAME directory so the rename is atomic.
    """
    tmp = index_path.with_name(index_path.name + f".tmp{os.getpid()}")
    try:
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(index_path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def build_index(vault: Path, verbose: bool = True, batch: int = _BATCH_WRITE) -> dict:
    """Embed every eligible note, reusing cached vectors for unchanged notes.

    Incremental by content hash: a note whose sha1 matches the cached entry is
    not re-embedded. Notes deleted from the vault are dropped from the index on
    a completed run. Progress is flushed atomically every `batch` newly-embedded
    notes; those interim flushes carry the union of old and new entries (a
    deletion is only provable once the walk finishes, so a killed run keeps
    stale entries rather than losing good ones - the next full run prunes them).
    """
    started = time.time()
    index_path = vault / INDEX_FILE
    cache: dict = {}
    if index_path.exists():
        try:
            cache = json.loads(index_path.read_text())
        except Exception:
            cache = {}
    # Format 2 = per-chunk vectors with identity headers (fix 13/24). A cache in
    # the old shape must not be reused: the note text is unchanged but what we
    # embed for it is not. Same rule for a MODEL switch (fix 16/24): vectors
    # from different embedding models live in different spaces - mixing them
    # silently would make every similarity meaningless.
    # ...and the same rule for the CHUNKER. A note's text can be unchanged while
    # what we embed for it changes: the pre-2026-09-09 geometry (1,200 chars x 8)
    # embedded at most 9,600 characters of a note, so its cached vectors are
    # truncated. Hash-matching them would keep 45% of the vault unretrievable
    # forever while every counter reported a healthy cached hit. The geometry is
    # fingerprinted into the index and a change invalidates the whole cache.
    cache_ok = (
        cache.get("format") == 2
        and cache.get("model") == EMBED_MODEL
        and cache.get("chunker") == chunker_fingerprint()
    )
    old = cache.get("notes", {}) if cache_ok else {}
    new: dict = {}
    embedded = reused = skipped = failed = degraded = 0
    degraded_paths: list[str] = []
    dropped_paths: list[str] = []
    skipped_reasons: dict[str, int] = {}

    def _flush(final: bool = False) -> dict:
        """Persist progress. Interim flushes keep entries not yet revisited; the
        final flush is authoritative and prunes notes deleted from the vault."""
        notes = new if final else {**old, **new}
        payload = {
            "model": EMBED_MODEL,
            "format": 2,
            "chunker": chunker_fingerprint(),
            "built": int(time.time()) if final else cache.get("built", int(started)),
            # The policy travels WITH the index, as data. The MCP server must
            # ship standalone and cannot import from scripts/, but it has to
            # know which notes the index deliberately omits or it reports every
            # excluded note as missing coverage and warns forever. Carrying the
            # rules here keeps one authoring site (INDEX_POLICY) and no second
            # copy of the list to drift.
            "policy": policy_payload(),
            "notes": notes,
        }
        _atomic_write_index(index_path, payload)
        return payload

    for md, verdict in _iter_all_notes(vault):
        rel = md.relative_to(vault).as_posix()
        if verdict is not None:
            skipped += 1
            skipped_reasons[verdict] = skipped_reasons.get(verdict, 0) + 1
            continue
        try:
            text = md.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        h = _content_hash(text)
        prev = old.get(rel)
        if prev and prev.get("hash") == h:
            new[rel] = prev
            reused += 1
            continue
        # An empty/whitespace-only note has no body to embed; fall back to its title
        # (which still carries meaning) so it stays findable. Skip only if even that is empty.
        header, body = prepare_note_text(md.stem, text)
        # A note that is all scaffolding still embeds its identity header, so it
        # stays findable by name/aliases. Skip only if even that is empty.
        embed_text = body if body else header.strip()
        if not embed_text.strip():
            continue
        degraded_note = False
        try:
            vecs = embed_note_chunks(embed_text, header=header)
        except Exception:
            # The body will not embed even split down - keep the note findable
            # by NAME at least: an identity-only vector beats silent absence.
            try:
                vecs = [[round(x, 6) for x in embed(header.strip() or md.stem)]]
                degraded_note = True
            except Exception as e:  # one bad note must not abort a 1000-note run
                failed += 1
                dropped_paths.append(rel)
                if verbose:
                    print(f"  [skip] {rel}: {e}", file=sys.stderr)
                continue
        tm = _FM_TYPE_RE.search(text[:400])
        # `at` is the embedding timestamp: --stats reports the oldest one, which
        # is how "the index is four days stale" becomes visible without guessing.
        entry = {"hash": h, "title": md.stem, "vecs": vecs, "at": int(time.time())}
        if tm:
            entry["type"] = tm.group(1).lower()
        if degraded_note:
            entry["degraded"] = True
            degraded += 1
            degraded_paths.append(rel)
        new[rel] = entry
        embedded += 1
        if batch > 0 and embedded % batch == 0:
            _flush()
            if verbose:
                print(f"  embedded {embedded} notes (progress saved)...", file=sys.stderr)

    removed = [rel for rel in old if rel not in new]
    out = _flush(final=True)
    elapsed = time.time() - started
    if verbose:
        total_eligible = len(new) + failed
        pct = (100.0 * len(new) / total_eligible) if total_eligible else 100.0
        print(
            f"[semantic] indexed {len(new)} notes ({embedded} new, {reused} cached, "
            f"{skipped} excluded, {degraded} degraded, {failed} dropped, "
            f"{len(removed)} removed) -> {index_path}",
            file=sys.stderr,
        )
        print(f"[semantic] coverage: {len(new)}/{total_eligible} ({pct:.1f}%)", file=sys.stderr)
        for reason, n in sorted(skipped_reasons.items(), key=lambda kv: -kv[1]):
            print(f"  [excluded x{n}] {reason}", file=sys.stderr)
        # Gaps must be a report, not a surprise: name every degraded/dropped note.
        for rel in degraded_paths:
            print(f"  [degraded to identity-only] {rel}", file=sys.stderr)
        for rel in dropped_paths:
            print(f"  [DROPPED - not findable semantically] {rel}", file=sys.stderr)
        for rel in removed:
            print(f"  [removed from index - note deleted] {rel}", file=sys.stderr)
        # The one line a job log or a status board can grep for.
        print(
            f"reindex: {len(new)} notes, {embedded} embedded, {skipped} skipped, "
            f"{elapsed:.1f} s",
            file=sys.stderr,
        )
    return out


def _percentile(values: list[int], p: float) -> int:
    """Nearest-rank percentile on a sorted copy. No numpy in this repo."""
    if not values:
        return 0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
    return s[k]


def index_stats(vault: Path) -> dict:
    """Index coverage, staleness and chunk distribution - no embedding backend
    needed, so it works when Ollama is down and it is cheap enough for a job."""
    index_path = vault / INDEX_FILE
    index: dict = {}
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            index = {}
    notes = index.get("notes", {}) or {}

    eligible: list[str] = []
    excluded: dict[str, int] = {}
    stale = missing = 0
    for md, verdict in _iter_all_notes(vault):
        rel = md.relative_to(vault).as_posix()
        if verdict is not None:
            excluded[verdict] = excluded.get(verdict, 0) + 1
            continue
        eligible.append(rel)
        entry = notes.get(rel)
        if not entry:
            missing += 1
            continue
        try:
            if entry.get("hash") != _content_hash(md.read_text(encoding="utf-8", errors="ignore")):
                stale += 1
        except OSError:
            pass

    chunk_counts = [len(n.get("vecs") or ([n["vec"]] if n.get("vec") else [])) for n in notes.values()]
    ats = [n["at"] for n in notes.values() if isinstance(n.get("at"), (int, float))]
    orphans = [rel for rel in notes if rel not in set(eligible)]
    return {
        "index_path": str(index_path),
        "exists": index_path.exists(),
        "model": index.get("model"),
        "format": index.get("format"),
        "built": index.get("built"),
        "notes_indexed": len(notes),
        "notes_eligible": len(eligible),
        "coverage_pct": round(100.0 * len(notes) / len(eligible), 1) if eligible else 0.0,
        "never_indexed": missing,
        "stale_hash": stale,
        "orphan_entries": len(orphans),
        "excluded": excluded,
        "excluded_total": sum(excluded.values()),
        "oldest_write": min(ats) if ats else None,
        "newest_write": max(ats) if ats else None,
        "undated_entries": len(notes) - len(ats),
        "chunks_p50": _percentile(chunk_counts, 0.50),
        "chunks_p90": _percentile(chunk_counts, 0.90),
        "chunks_max": max(chunk_counts) if chunk_counts else 0,
        "chunks_total": sum(chunk_counts),
        "at_ceiling": sum(1 for c in chunk_counts if c >= _MAX_CHUNKS),
    }


def _fmt_age(ts: float | None) -> str:
    if not ts:
        return "unknown"
    stamp = time.strftime("%Y-%m-%dT%H:%M", time.localtime(ts))
    days = (time.time() - ts) / 86400.0
    return f"{stamp} ({days:.1f} days ago)"


def print_stats(vault: Path, as_json: bool = False) -> int:
    s = index_stats(vault)
    if as_json:
        print(json.dumps(s, indent=2))
        return 0 if s["exists"] else 1
    if not s["exists"]:
        print(f"[semantic] no index at {s['index_path']} - build it: --build", file=sys.stderr)
        return 1
    print(f"index          {s['index_path']}")
    print(f"model/format   {s['model']} / {s['format']}")
    print(f"coverage       {s['notes_indexed']} / {s['notes_eligible']} eligible "
          f"({s['coverage_pct']}%)")
    print(f"never indexed  {s['never_indexed']}")
    print(f"stale (hash)   {s['stale_hash']}")
    print(f"orphan entries {s['orphan_entries']} (indexed but no longer eligible)")
    print(f"excluded       {s['excluded_total']} notes")
    for reason, n in sorted(s["excluded"].items(), key=lambda kv: -kv[1]):
        print(f"                 x{n:<5} {reason}")
    print(f"oldest write   {_fmt_age(s['oldest_write'])}")
    print(f"newest write   {_fmt_age(s['newest_write'])}")
    if s["undated_entries"]:
        print(f"               ({s['undated_entries']} entries predate timestamping)")
    print(f"chunks/note    p50 {s['chunks_p50']}, p90 {s['chunks_p90']}, "
          f"max {s['chunks_max']}, total {s['chunks_total']}")
    if s["at_ceiling"]:
        print(f"AT CEILING     {s['at_ceiling']} notes at _MAX_CHUNKS={_MAX_CHUNKS} "
              f"- text may be truncated; split them")
    return 0


def load_index(vault: Path) -> dict:
    index_path = vault / INDEX_FILE
    if not index_path.exists():
        raise RuntimeError(f"No semantic index at {index_path}. Build it first: --build")
    return json.loads(index_path.read_text())


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #
def semantic_search(query: str, index: dict, limit: int = 10) -> list[dict]:
    """Rank notes by meaning-distance from the query."""
    # The query must live in the same vector space as the index (fix 16/24).
    qvec = embed(query, model=index.get("model"))

    def _score(n: dict) -> tuple[float, int | None]:
        """Best chunk's cosine and its 1-based index.

        The index is reported so the eval can measure WHERE in a note the
        winning evidence sat (`retrieval_eval.chunk_hit_position`): when every
        hit lands in chunk 1, a chunker that drops the rest of the note is
        invisible to recall@k.
        """
        vecs = n.get("vecs") or ([n["vec"]] if n.get("vec") else [])
        best = (0.0, None)
        for i, v in enumerate(vecs, start=1):
            s = cosine(qvec, v)
            if best[1] is None or s > best[0]:
                best = (s, i)
        return best

    scored = []
    for rel, n in index["notes"].items():
        score, chunk = _score(n)
        scored.append({"path": rel, "title": n["title"], "score": score, "chunk": chunk})
    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored[:limit]


def _rank_map(results: list[dict]) -> dict[str, int]:
    return {r["path"]: i for i, r in enumerate(results)}


def hybrid_search(query: str, index: dict, lexical_results: list[dict], limit: int = 10) -> list[dict]:
    """Combine lexical and semantic rankings with Reciprocal Rank Fusion.

    RRF score = sum over each ranking of 1/(k + rank). It needs no score calibration
    between the two systems (lexical counts vs cosine values are not comparable), just
    their rank orders - which is exactly why it is the standard way to fuse them.
    """
    K = 60
    sem = semantic_search(query, index, limit=max(limit, 20))
    sem_rank = _rank_map(sem)
    lex_rank = _rank_map(lexical_results)
    sem_chunk = {r["path"]: r.get("chunk") for r in sem}
    paths = set(sem_rank) | set(lex_rank)
    fused = []
    for p in paths:
        score = 0.0
        if p in lex_rank:
            score += 1.0 / (K + lex_rank[p])
        if p in sem_rank:
            score += 1.0 / (K + sem_rank[p])
        title = next((r["title"] for r in (sem + lexical_results) if r["path"] == p), p)
        # Carry the semantic arm's winning chunk through the fusion; a purely
        # lexical hit has none, and None is the honest answer there.
        fused.append({"path": p, "title": title, "score": score, "chunk": sem_chunk.get(p)})
    fused.sort(key=lambda r: r["score"], reverse=True)
    return fused[:limit]


def main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Local semantic search over the vault (via Ollama)")
    ap.add_argument("--path", required=True, help="Vault root")
    ap.add_argument("--build", action="store_true", help="Build/refresh the embedding index")
    ap.add_argument("--query", help="Run a semantic search and print the top matches")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--stats", action="store_true",
                    help="Print index coverage, staleness and chunk distribution "
                         "(needs no embedding backend)")
    ap.add_argument("--json", action="store_true", help="With --stats: machine-readable output")
    ap.add_argument("--batch", type=int, default=_BATCH_WRITE,
                    help=f"With --build: flush the index every N embedded notes "
                         f"(default {_BATCH_WRITE}; 0 disables interim flushes)")
    args = ap.parse_args(argv[1:])

    vault = Path(args.path).expanduser().resolve()
    if not vault.is_dir():
        print(f"vault path does not exist: {vault}", file=sys.stderr)
        return 2
    # --stats reads the index and the vault only, so it must keep working when
    # the backend is down: that is exactly when someone asks why search is bad.
    if args.stats and not (args.build or args.query):
        return print_stats(vault, as_json=args.json)
    if not ollama_available():
        print(
            f"Local model runtime not found at {OLLAMA_URL}.\n"
            f"Install Ollama (https://ollama.com), open it, then: ollama pull {EMBED_MODEL}",
            file=sys.stderr,
        )
        return 3

    if args.build:
        build_index(vault, batch=args.batch)
    if args.query:
        index = load_index(vault)
        for i, r in enumerate(semantic_search(args.query, index, args.limit), 1):
            print(f"{i:2}. {r['score']:.3f}  {r['path']}")
    if args.stats:
        print_stats(vault, as_json=args.json)
    if not (args.build or args.query or args.stats):
        print("Nothing to do. Pass --build, --query and/or --stats.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

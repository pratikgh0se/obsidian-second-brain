"""Retrieval-quality eval harness for the vault.

Measures how well the vault's search actually finds the right note for a
natural-language question - BEFORE anyone reaches for a vector index. It reuses
the real `search()` from the MCP connector (`integrations/obsidian-mcp-server/
vault_ops.py`), so it scores the exact term-frequency, title-weighted ranking
the skill ships with, not a reimplementation.

Two modes:

  generate  Bootstrap an eval set FROM the vault. Samples notes, and for each
            asks an LLM to write a natural-language question a user would ask
            whose answer lives in that note - deliberately AVOIDING the note's
            title words, so the question tests semantic retrieval, not string
            match. The note's path is the gold answer. Writes cases as JSONL.
            Falls back to a key-free heuristic generator if no XAI_API_KEY.

  eval      (default) Load the cases, run each question through the real
            search, and report recall@1/3/5/10 and MRR, plus the failures -
            including which note DID rank #1 when the gold note lost, so the
            "noisy high-mention note floats above the canonical note" failure
            mode is visible, not just a number.

Usage:
    uv run python scripts/eval/retrieval_eval.py --generate 30
    uv run python scripts/eval/retrieval_eval.py
    uv run python scripts/eval/retrieval_eval.py --cases scripts/eval/retrieval_cases.jsonl --json

Env (from ~/.config/obsidian-second-brain/.env): OBSIDIAN_VAULT_PATH required;
XAI_API_KEY optional (enables the LLM question generator).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
MCP_DIR = REPO_ROOT / "integrations" / "obsidian-mcp-server"
sys.path.insert(0, str(MCP_DIR))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
# This module's own directory, so `import semantic_search` works when
# retrieval_eval is imported (a test) and not only when it is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Load env (OBSIDIAN_VAULT_PATH + optional keys) the same way the research toolkit does.
try:
    from research.lib.config import VAULT_PATH  # noqa: F401  (import triggers dotenv load)
# config.py raises SystemExit (a BaseException) when OBSIDIAN_VAULT_PATH is
# unset, so a bare `except Exception` lets it kill the importing process.
except (Exception, SystemExit):  # pragma: no cover - fall back to a bare dotenv load
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(
            os.environ.get("OBSIDIAN_ENV_FILE")
            or (Path.home() / ".config" / "obsidian-second-brain" / ".env")
        ).expanduser())
    except Exception:
        pass

import vault_ops  # noqa: E402  (depends on sys.path insert above)

DEFAULT_CASES = REPO_ROOT / "scripts" / "eval" / "retrieval_cases.jsonl"
RECALL_KS = (1, 3, 5, 10)
SEARCH_LIMIT = 10

# Folders that hold real, answerable knowledge notes (skip raw sources, exports, config).
_KNOWLEDGE_HINTS = ("wiki/", "Knowledge/", "Ideas/", "Projects/", "Research/", "concepts/")
_SKIP_PREFIXES = ("raw/", "_export/", "templates/", ".")
_MIN_BODY_CHARS = 400


# --------------------------------------------------------------------------- #
# The §4.3 metric set (Capsule decision row 128, M077)
# --------------------------------------------------------------------------- #
# recall@k + MRR measure the search tool. They cannot see the two failures that
# actually happened (a note 88% unembedded still passes if the answer is in the
# first chunk, and coverage was never a metric), so four fields join them. Row
# 128 fixed the floor: recall@5 75.9%, measured 2026-09-02, and a change may
# not regress below it.
FLOOR_RECALL_AT_5 = 0.759
FLOOR_AS_OF = "2026-09-02"
# Comparability comes from KEEPING old sets, not from never making new ones:
# the corpus grew 11x under a frozen set and the job's own 2x guard fired and
# was ignored. So a new set is a NEW FILE, named for the corpus it was cut at.
REGENERATE_AT_MULTIPLE = 2.0


def _top_folder(path: str) -> str:
    """The first path segment, or `(root)` for a note in the vault root. A
    root note is bucketed, never dropped: `index.md` losing is a real miss."""
    head, _, tail = str(path).partition("/")
    return head if tail else "(root)"


def recall_by_folder(per_case: list[dict], k: int = 5) -> dict[str, dict]:
    """recall@k grouped by the gold note's top folder.

    Separates corpus growth from retrieval quality: a vault-wide recall drop
    caused entirely by 400 new `Daily/` notes looks identical to a real
    regression until it is split this way.
    """
    buckets: dict[str, list[dict]] = {}
    for c in per_case:
        gold = (c.get("gold") or [None])[0]
        if not gold:
            continue
        buckets.setdefault(_top_folder(gold), []).append(c)
    out = {}
    for folder, cases in sorted(buckets.items()):
        hits = sum(1 for c in cases if 0 < c.get("rank", 0) <= k)
        out[folder] = {"cases": len(cases), "recall_at_5": round(hits / len(cases), 3)}
    return out


def _pct(values: list[int], q: float) -> int:
    """Nearest-rank percentile on a sorted list - stdlib-shaped, no numpy."""
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def chunk_hit_position(per_case: list[dict]) -> dict | None:
    """p50/p90 of the 1-based chunk index that carried the winning score.

    This is the metric §0.5's failure needed: when the answer is always in
    chunk 1, a broken chunker is invisible. None (absent, not zero) when the
    search engine reports no chunk - `--mode lexical` never will.
    """
    chunks = [int(c["chunk"]) for c in per_case if c.get("chunk")]
    if not chunks:
        return None
    return {"p50": _pct(chunks, 0.5), "p90": _pct(chunks, 0.9),
            "max": max(chunks), "cases": len(chunks)}


def stale_hit_rate(per_case: list[dict], superseded: set[str]) -> float:
    """Share of cases whose top hit is a note carrying `superseded-by:`.

    LongMemEval's knowledge-updates ability, untested here until now: a
    superseded note outranking its successor is worse than a miss, because the
    role gets a confident wrong answer instead of nothing.
    """
    if not per_case:
        return 0.0
    bad = sum(1 for c in per_case if c.get("top_hit") in superseded)
    return round(bad / len(per_case), 3)


def superseded_paths(vault: Path) -> set[str]:
    """Vault-relative paths whose frontmatter carries a `superseded-by:` key.
    Frontmatter only - the string in a note's prose is discussion, not state.

    Walks the whole vault, deliberately NOT `_candidate_notes`: that filter
    answers "is this note worth asking a question about", which is a different
    question. A 100-char stub whose entire content is `superseded-by:` is not a
    candidate for a case, but if it tops a result it is exactly the failure this
    metric exists to catch - and so is a policy-excluded note reached by the arm
    that does not exclude it.
    """
    out = set()
    for md in sorted(vault.rglob("*.md")):
        try:
            head = md.read_text(encoding="utf-8", errors="replace")[:2000]
        except OSError:
            continue
        if not head.startswith("---") or head.count("---") < 2:
            continue
        if "superseded-by:" in head.split("---", 2)[1]:
            out.add(md.relative_to(vault).as_posix())
    return out


def floor_block(recall_at_5: float) -> dict:
    """The floor, as a fact in the summary rather than a number in a note."""
    return {"recall_at_5": FLOOR_RECALL_AT_5, "as_of": FLOOR_AS_OF,
            "measured": round(float(recall_at_5), 3),
            "regressed": round(float(recall_at_5), 3) < FLOOR_RECALL_AT_5}


def needs_regeneration(*, eligible: int, baseline: int) -> bool:
    return baseline > 0 and eligible >= baseline * REGENERATE_AT_MULTIPLE


def cases_filename(eligible: int) -> str:
    return f"retrieval_cases-{int(eligible)}notes.jsonl"


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _vault() -> Path:
    return vault_ops.resolve_vault()


def _lexical_exclusion(rel: str) -> str | None:
    """Why the LEXICAL arm can never return `rel`, or None.

    Reads `vault_ops`' own deny lists (row 124) rather than restating them, so
    this cannot drift from what `search()` actually scans.
    """
    parts = tuple(p.lower() for p in Path(rel).parts)
    for p in parts:
        if p in vault_ops._SKIP_DIRS or p.endswith("templates"):
            return f"skip dir: {p}/"
    for p in parts[:-1]:
        if p in vault_ops._LEXICAL_DENY_DIR_NAMES:
            return f"lexical denied dir: {p}/"
    low = rel.lower()
    for pref in vault_ops._LEXICAL_DENY_PATH_PREFIXES:
        if low.startswith(pref):
            return f"lexical denied prefix: {pref}"
    if vault_ops._LEXICAL_MIRROR_RE.search(rel):
        return "skill mirror (**/skills/<tool>/<role>/<doc>.md)"
    if rel.endswith(".excalidraw.md"):
        return "excalidraw drawing"
    return None


def retrieval_exclusion(rel: str) -> str | None:
    """Why no search arm can ever return `rel`, or None if it is retrievable.

    An eval case whose gold note is unretrievable BY POLICY scores a
    guaranteed miss and measures nothing. On 2026-09-10 a fresh
    `--generate 30 --style semantic` sampled `Architecture/skills/hermes/...`
    mirror notes the index excludes on purpose, and the resulting set scored
    recall@5 10% by construction (C4 task 1, decision row 128).

    Both arms are consulted through their real predicates - the semantic
    index's `policy_verdict` and `vault_ops`' lexical deny lists - so a note
    either arm refuses is out. No third copy of the rules lives here.
    """
    lex = _lexical_exclusion(rel)
    if lex:
        return lex
    try:
        import semantic_search as ss
    except Exception as exc:  # pragma: no cover - the index policy is the point
        print(f"  index-policy check skipped: {exc}", file=sys.stderr)
        return None
    return ss.policy_verdict(rel)


def _candidate_notes(vault: Path) -> list[Path]:
    """Knowledge notes worth asking about - substantial, not raw sources, and
    RETRIEVABLE: a note the search policy excludes can never be found."""
    out: list[Path] = []
    for md in sorted(vault.rglob("*.md")):
        rel = md.relative_to(vault).as_posix()
        if any(rel.startswith(p) for p in _SKIP_PREFIXES):
            continue
        if retrieval_exclusion(rel):
            continue
        try:
            body = md.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if len(body) < _MIN_BODY_CHARS:
            continue
        out.append(md)
    return out


def _title_tokens(stem: str) -> set[str]:
    return {t for t in re.split(r"\W+", stem.lower()) if len(t) > 2}


# --------------------------------------------------------------------------- #
# Generate mode
# --------------------------------------------------------------------------- #
def _llm_question(body: str, title: str, style: str = "semantic") -> str | None:
    """Ask Grok for a natural-language question this note answers.

    style="semantic": forbid the note's title words (tests meaning-based retrieval -
        the hard case lexical search cannot do).
    style="keyword": allow the topic words a real user would recall (tests whether
        the right note ranks above noisy long notes - where re-ranking helps).
    """
    try:
        from research.lib import grok
    except Exception:
        return None
    import os

    if not os.environ.get("XAI_API_KEY", "").strip():
        return None
    excerpt = body[:2500]
    if style == "keyword":
        rule = (
            "Write it the way a person who half-remembers this note would actually "
            "search - you MAY use the note's topic words. "
        )
    else:
        rule = "Hard rule: do NOT reuse the note's title words verbatim. "
    prompt = (
        "Below is one note from a personal knowledge vault. Write ONE natural-language "
        "question a person would realistically ask whose answer is in this note. "
        f"{rule}Do NOT mention that this is a note, keep it under 20 words, "
        "output ONLY the question.\n\n"
        f"Note title: {title}\n\nNote body:\n{excerpt}"
    )
    try:
        res = grok.call(prompt, command="retrieval-eval", max_output_tokens=120)
        q = (res.get("text") or "").strip().splitlines()[0].strip().strip('"')
        return q or None
    except Exception as e:
        print(f"[generate] LLM call failed ({e}); using heuristic for this note", file=sys.stderr)
        return None


def _heuristic_question(body: str, title: str) -> str | None:
    """Key-free fallback: the longest body sentence that avoids the title words."""
    title_toks = _title_tokens(title)
    # strip frontmatter + preamble headers
    text = re.sub(r"^---.*?---", "", body, count=1, flags=re.DOTALL)
    text = re.sub(r"^#.*$", "", text, flags=re.MULTILINE)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    best = ""
    for s in sentences:
        s = s.strip().replace("\n", " ")
        if len(s) < 30 or len(s) > 160:
            continue
        toks = {t for t in re.split(r"\W+", s.lower()) if len(t) > 2}
        if title_toks & toks:  # avoid sentences that echo the title
            continue
        if len(s) > len(best):
            best = s
    return f"What does the vault say about: {best}" if best else None


def generate(n: int, cases_path: Path, style: str = "semantic", force: bool = False) -> int:
    vault = _vault()
    notes = _candidate_notes(vault)
    if not notes:
        print("No candidate knowledge notes found in the vault.", file=sys.stderr)
        return 1
    # The corpus this set is cut at, named in the filename: a set is comparable
    # to its successors only if you can see which corpus each one measured.
    if cases_path == DEFAULT_CASES:
        cases_path = cases_path.with_name(cases_filename(len(notes)))
        print(f"Cutting a new set at {len(notes)} eligible notes -> {cases_path.name}")
        if cases_path.exists() and not force:
            print(f"Refusing to overwrite {cases_path}; pass --force or --cases <new-path>.",
                  file=sys.stderr)
            return 1
    # Deterministic, well-spread sample (no Date/random; stable across runs).
    step = max(1, len(notes) // n)
    sampled = notes[::step][:n]
    cases: list[dict[str, Any]] = []
    for md in sampled:
        rel = md.relative_to(vault).as_posix()
        body = md.read_text(encoding="utf-8", errors="ignore")
        q = _llm_question(body, md.stem, style) or _heuristic_question(body, md.stem)
        if not q:
            continue
        cases.append({"q": q, "gold": [rel], "title": md.stem})
        print(f"  + {md.stem[:50]:52} <- {q[:60]}")
    cases_path.parent.mkdir(parents=True, exist_ok=True)
    with cases_path.open("w", encoding="utf-8") as fh:
        for c in cases:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(cases)} cases to {cases_path}")
    return 0


# --------------------------------------------------------------------------- #
# Eval mode
# --------------------------------------------------------------------------- #
def _rank_of_gold(results: list[dict[str, Any]], gold: list[str]) -> int:
    """1-based rank of the first result whose path matches any gold path; 0 if absent."""
    gold_set = {g.strip() for g in gold}
    for i, r in enumerate(results, start=1):
        if r.get("path") in gold_set:
            return i
    return 0


def _split_external_cmd(cmd: str) -> list[str]:
    r"""Split RETRIEVAL_EVAL_EXTERNAL_CMD into argv.

    Two forms. A JSON array (the value starts with "[") is the exact form: each
    element is one argument and no shell quoting rules apply, only JSON's own
    (a double quote inside a string is \", a backslash is \\, or write Windows
    paths with forward slashes), so spaces, embedded quotes and empty arguments
    pass through exactly:
        ["bash", "C:/Program Files/x/engine.sh", "--out", "say \"hi\""]
    A plain string is split with shell-like rules: POSIX shlex on macOS and
    Linux; on Windows, where POSIX shlex would treat every backslash as an escape
    and eat the separators of a path, non-POSIX splitting that keeps backslashes
    and removes one layer of matching surrounding quotes per token. The plain
    form covers a command plus simple arguments; anything with escaped quotes or
    quotes inside an argument belongs in the JSON form.
    """
    import json
    import shlex

    stripped = cmd.strip()
    if stripped.startswith("["):
        try:
            parts = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"RETRIEVAL_EVAL_EXTERNAL_CMD looks like a JSON array but does not parse: {exc}")
        if not isinstance(parts, list) or not parts or not all(isinstance(p, str) for p in parts):
            raise SystemExit("RETRIEVAL_EVAL_EXTERNAL_CMD as JSON must be a non-empty array of strings")
        return parts
    if os.name != "nt":
        return shlex.split(cmd)
    parts: list[str] = []
    for token in shlex.split(cmd, posix=False):
        if len(token) >= 2 and token[0] == token[-1] and token[0] in ("'", '"'):
            token = token[1:-1]
        parts.append(token)
    return parts


def _searcher(mode: str):
    """Return a (label, fn(query)->results) for the chosen retrieval mode.

    Four modes, four TRUE labels (stress-test fix 10/24 - before this, "lexical"
    silently measured the fused blend and "hybrid" fused an already-fused input,
    double-counting semantic and flipping the semantic-vs-hybrid conclusion):

      lexical   pure word-match, fusion forced off
      default   exactly what the shipped MCP serves (env-driven fusion)
      semantic  local embeddings only
      hybrid    single RRF of pure lexical + semantic
    """
    if mode == "lexical":
        return "pure lexical: term-frequency, title-weighted (fusion off)", \
            lambda q: vault_ops.search(q, limit=SEARCH_LIMIT, semantic=False)
    if mode == "default":
        return "shipped default: vault_ops.search (lexical + semantic RRF when available)", \
            lambda q: vault_ops.search(q, limit=SEARCH_LIMIT)
    if mode == "external":
        # Benchmark ANY external retrieval engine on the same cases: point
        # RETRIEVAL_EVAL_EXTERNAL_CMD at a command that takes the query as its
        # final argument and prints ranked results - a JSON array of paths (or
        # of {"path": ...} objects), or plain newline-separated paths. This is
        # how a TypeAgent / structured-RAG / vector-DB runner competes against
        # the shipped search without being imported or vendored (pattern from
        # the structured-rag eval fork, fork-insights round 2).
        import os
        import subprocess
        cmd = os.environ.get("RETRIEVAL_EVAL_EXTERNAL_CMD", "").strip()
        if not cmd:
            raise SystemExit(
                "External mode needs RETRIEVAL_EVAL_EXTERNAL_CMD - a command that "
                "takes the query as its final argument and prints ranked note "
                "paths (JSON array or one per line). The value is a plain command "
                "line, or a JSON array of arguments for anything shell quoting "
                "cannot express."
            )

        parts = _split_external_cmd(cmd)

        def _external(q: str) -> list[dict[str, Any]]:
            proc = subprocess.run(
                parts + [q],
                capture_output=True, text=True, timeout=120,
            )
            if proc.returncode != 0:
                print(f"[external] engine failed on {q!r}: {proc.stderr.strip()[:200]}", file=sys.stderr)
                return []
            out = proc.stdout.strip()
            if not out:
                return []
            try:
                parsed = json.loads(out)
                items = parsed if isinstance(parsed, list) else []
            except json.JSONDecodeError:
                items = out.splitlines()
            results = []
            for item in items:
                path = item.get("path") if isinstance(item, dict) else item
                if isinstance(path, str) and path.strip():
                    results.append({"path": path.strip()})
            return results[:SEARCH_LIMIT]

        return f"external engine: {cmd}", _external
    import semantic_search as ss  # local module; needs Ollama running
    vault = vault_ops.resolve_vault()
    if not ss.ollama_available():
        raise SystemExit(
            "Semantic/hybrid mode needs the local model runtime. Install Ollama "
            f"(https://ollama.com), then: ollama pull {ss.EMBED_MODEL}, then "
            "build the index: uv run python scripts/eval/semantic_search.py --path <vault> --build"
        )
    index = ss.load_index(vault)
    if mode == "semantic":
        return f"local embeddings: {index.get('model')} (semantic_search)", \
            lambda q: ss.semantic_search(q, index, limit=SEARCH_LIMIT)
    # hybrid: the lexical arm MUST be pure, or semantic gets fused twice
    return f"hybrid: pure lexical + {index.get('model')} (single RRF)", \
        lambda q: ss.hybrid_search(q, index,
                                   vault_ops.search(q, limit=SEARCH_LIMIT, semantic=False),
                                   limit=SEARCH_LIMIT)


def evaluate(cases_path: Path, as_json: bool, mode: str = "lexical") -> int:
    if not cases_path.exists():
        print(
            f"No cases file at {cases_path}.\n"
            f"Bootstrap one first:  uv run python scripts/eval/retrieval_eval.py --generate 30",
            file=sys.stderr,
        )
        return 1
    cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases:
        print("Cases file is empty.", file=sys.stderr)
        return 1

    label, search_fn = _searcher(mode)
    per_case: list[dict[str, Any]] = []
    for c in cases:
        try:
            results = search_fn(c["q"])
        except Exception as e:
            # One bad case must not discard every case already scored. In
            # semantic/hybrid mode embed() uses the full (1,3,8,15) retry ladder
            # at 120s per attempt, so a slow backend can burn ~10 minutes here
            # and then take the entire run down with it. Count it as a miss.
            print(f"  case failed, counted as a miss: {c['q'][:60]!r}: {e}", file=sys.stderr)
            results = []
        rank = _rank_of_gold(results, c.get("gold", []))
        top = results[0]["path"] if results else None
        per_case.append({
            "q": c["q"],
            "gold": c.get("gold", []),
            "rank": rank,
            "top_hit": top,
            "title": c.get("title", ""),
            # Which chunk of the gold note carried the winning score. Unset
            # when the engine does not report one (lexical, external, and the
            # shipped `default` fusion) - never synthesised as 1.
            "chunk": (results[rank - 1].get("chunk") if 0 < rank <= len(results) else None),
        })

    n = len(per_case)
    recall = {k: sum(1 for x in per_case if 0 < x["rank"] <= k) / n for k in RECALL_KS}
    mrr = sum((1.0 / x["rank"]) if x["rank"] else 0.0 for x in per_case) / n
    misses = [x for x in per_case if x["rank"] == 0]
    buried = [x for x in per_case if x["rank"] > 3]

    summary = {
        "cases": n,
        "search": label,
        "recall_at": {str(k): round(recall[k], 3) for k in RECALL_KS},
        "mrr": round(mrr, 3),
        "misses": len(misses),
        "buried_below_3": len(buried),
    }
    # The §4.3 metric set, ADDED to the existing keys so every trend built on
    # them survives (decision row 128).
    summary["recall_at_5_by_folder"] = recall_by_folder(per_case, k=5)
    summary["chunk_hit_position"] = chunk_hit_position(per_case)
    try:
        summary["stale_hit_rate"] = stale_hit_rate(per_case, superseded_paths(_vault()))
    except Exception as exc:  # a vault read must never take the eval down
        print(f"  stale-hit probe skipped: {exc}", file=sys.stderr)
        summary["stale_hit_rate"] = None
    summary["floor"] = floor_block(recall[5])
    summary["cases_file"] = cases_path.name

    if as_json:
        # Force UTF-8 stdout: on Windows a pipe defaults to cp1252, which cannot
        # encode every character a case question or note path may carry.
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
        print(json.dumps({"summary": summary, "cases": per_case}, ensure_ascii=False, indent=2))
        return 0

    print(f"\nRetrieval eval - {n} cases  (engine: {summary['search']})")
    print("-" * 64)
    for k in RECALL_KS:
        bar = "#" * round(recall[k] * 40)
        print(f"  recall@{k:<2} {recall[k]*100:5.1f}%  {bar}")
    print(f"  MRR      {mrr:.3f}")
    if summary["floor"]["regressed"]:
        print(f"  FLOOR      recall@5 {recall[5]*100:.1f}% is BELOW the "
              f"{FLOOR_RECALL_AT_5*100:.1f}% floor of {FLOOR_AS_OF} - a regression")
    for folder, s in summary["recall_at_5_by_folder"].items():
        flag = "  <-- under 50%" if s["cases"] >= 3 and s["recall_at_5"] < 0.5 else ""
        print(f"  {folder:<14} recall@5 {s['recall_at_5']*100:5.1f}%  ({s['cases']} cases){flag}")
    if summary["chunk_hit_position"]:
        print(f"  chunk pos  p50 {summary['chunk_hit_position']['p50']}, "
              f"p90 {summary['chunk_hit_position']['p90']}")
    if summary["stale_hit_rate"]:
        print(f"  stale hits {summary['stale_hit_rate']*100:.1f}% of cases were topped "
              f"by a superseded note")
    print(f"  misses (gold not in top {SEARCH_LIMIT}): {len(misses)}   buried (rank>3): {len(buried)}")

    if misses:
        print("\nMisses - the right note never surfaced:")
        for x in misses[:15]:
            print(f"  Q: {x['q'][:70]}")
            print(f"     want: {x['gold'][0] if x['gold'] else '?'}")
            print(f"     #1 was: {x['top_hit']}")
    if buried:
        print("\nBuried - right note ranked below #3 (often a noisy high-mention note on top):")
        for x in buried[:10]:
            print(f"  rank {x['rank']}: {x['gold'][0] if x['gold'] else '?'}  (Q: {x['q'][:48]})")
            print(f"           #1 was: {x['top_hit']}")
    print()
    return 0


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Vault retrieval-quality eval harness")
    ap.add_argument("--generate", type=int, metavar="N",
                    help="Bootstrap N eval cases from the vault instead of evaluating")
    ap.add_argument("--style", choices=("semantic", "keyword"), default="semantic",
                    help="Question style when generating: semantic (avoid title words, "
                         "the hard case) or keyword (realistic lookup; default semantic)")
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES,
                    help=f"Cases JSONL path (default: {DEFAULT_CASES})")
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of a text report")
    ap.add_argument("--mode", choices=("lexical", "default", "semantic", "hybrid", "external"),
                    default="lexical",
                    help="Retrieval to score: lexical (pure word-match, default), "
                         "default (exactly what the shipped MCP serves), semantic "
                         "(local embeddings), or hybrid (pure lexical + semantic, "
                         "single RRF). semantic/hybrid need Ollama.")
    ap.add_argument("--force", action="store_true",
                    help="Allow --generate to overwrite an existing cases file")
    args = ap.parse_args()

    if args.generate is not None:
        # The default path is not the file that gets written: generate() renames
        # it to the corpus-versioned name and re-checks the guard there.
        if args.cases != DEFAULT_CASES and args.cases.exists() and not args.force:
            print(
                f"Refusing to overwrite existing cases at {args.cases}: regenerating "
                f"mid-experiment breaks the before/after comparison on the SAME cases.\n"
                f"Pass --force to overwrite, or --cases <new-path> for a fresh set.",
                file=sys.stderr,
            )
            return 1
        return generate(args.generate, args.cases, args.style, args.force)
    return evaluate(args.cases, args.json, args.mode)


if __name__ == "__main__":
    raise SystemExit(main())

"""The §4.3 metric set (Capsule decision row 128, M077). Pure functions over
already-scored cases: no vault, no index, no Ollama, no network.

The last two classes do touch a filesystem vault - a `tmp_path` fixture with a
handful of notes in it, never the real one - because the eligibility predicate
and the `superseded-by:` probe are about files by definition.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "eval"))
import retrieval_eval as re_mod


class MetricSet(unittest.TestCase):
    def test_per_folder_recall_groups_by_the_golds_top_folder(self):
        per_case = [
            {"rank": 1, "gold": ["Knowledge/a.md"]},
            {"rank": 9, "gold": ["Knowledge/b.md"]},
            {"rank": 0, "gold": ["Decisions/c.md"]},
        ]
        out = re_mod.recall_by_folder(per_case, k=5)
        self.assertEqual(out["Knowledge"], {"cases": 2, "recall_at_5": 0.5})
        self.assertEqual(out["Decisions"], {"cases": 1, "recall_at_5": 0.0})

    def test_a_gold_with_no_folder_is_bucketed_as_root_not_dropped(self):
        out = re_mod.recall_by_folder([{"rank": 1, "gold": ["index.md"]}], k=5)
        self.assertEqual(out["(root)"]["cases"], 1)

    def test_chunk_hit_position_percentiles(self):
        per_case = [{"chunk": c} for c in (1, 1, 2, 3, 9)]
        out = re_mod.chunk_hit_position(per_case)
        self.assertEqual(out["p50"], 2)
        self.assertEqual(out["p90"], 9)
        self.assertEqual(out["cases"], 5)

    def test_chunk_hit_position_is_absent_not_zero_when_nothing_reports_a_chunk(self):
        self.assertIsNone(re_mod.chunk_hit_position([{"rank": 1}]))

    def test_stale_hit_rate_counts_a_superseded_note_outranking_its_successor(self):
        per_case = [
            {"rank": 2, "top_hit": "Knowledge/old.md", "gold": ["Knowledge/new.md"]},
            {"rank": 1, "top_hit": "Knowledge/new.md", "gold": ["Knowledge/new.md"]},
        ]
        superseded = {"Knowledge/old.md"}
        self.assertEqual(re_mod.stale_hit_rate(per_case, superseded), 0.5)

    def test_the_floor_is_recall_at_5_759_percent_as_of_2026_09_02(self):
        self.assertEqual(re_mod.FLOOR_RECALL_AT_5, 0.759)
        self.assertEqual(re_mod.FLOOR_AS_OF, "2026-09-02")
        self.assertTrue(re_mod.floor_block(0.700)["regressed"])
        self.assertFalse(re_mod.floor_block(0.759)["regressed"])
        self.assertFalse(re_mod.floor_block(0.800)["regressed"])

    def test_the_regeneration_trigger_fires_at_twice_the_recorded_baseline(self):
        self.assertFalse(re_mod.needs_regeneration(eligible=300, baseline=232))
        self.assertTrue(re_mod.needs_regeneration(eligible=464, baseline=232))
        self.assertEqual(re_mod.cases_filename(464), "retrieval_cases-464notes.jsonl")


class GeneratorRespectsTheRetrievalPolicy(unittest.TestCase):
    """A case whose gold note no search arm can return measures nothing.

    On 2026-09-10 `--generate 30 --style semantic` sampled
    `Architecture/skills/hermes/...` mirror notes that the index policy excludes
    by design, and the set scored recall@5 10% by construction (C4 task 1).
    """

    def _vault(self) -> Path:
        import tempfile

        root = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, root, True)
        body = "\n".join(f"Line {i} of substantial answerable content." for i in range(30))
        for rel in (
            "Knowledge/Real note.md",                            # eligible
            "Architecture/skills/hermes/researcher/guide.md",    # skill mirror
            "copilot/scaffold.md",                               # denied dir
            "boards/Kanban.md",                                  # generated
            "Templates/note.md",                                 # skip dir
        ):
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f"---\ntype: note\n---\n\n## For future agent\n\n{body}\n",
                         encoding="utf-8")
        return root

    def test_the_mirror_note_is_excluded_and_the_knowledge_note_is_not(self):
        self.assertIsNone(re_mod.retrieval_exclusion("Knowledge/Real note.md"))
        self.assertTrue(
            re_mod.retrieval_exclusion("Architecture/skills/hermes/researcher/guide.md")
        )

    def test_candidate_notes_offers_only_the_eligible_note(self):
        vault = self._vault()
        rels = sorted(p.relative_to(vault).as_posix() for p in re_mod._candidate_notes(vault))
        self.assertEqual(rels, ["Knowledge/Real note.md"])

    def test_the_exclusion_reason_is_explainable_not_a_bare_bool(self):
        for rel in ("copilot/scaffold.md", "boards/Kanban.md", "Templates/note.md"):
            reason = re_mod.retrieval_exclusion(rel)
            self.assertIsInstance(reason, str, rel)
            self.assertTrue(reason.strip(), rel)


class SupersededProbe(unittest.TestCase):
    def test_only_frontmatter_superseded_by_counts(self):
        import shutil
        import tempfile

        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        filler = "\n".join(f"Line {i} of content that makes the note substantial." for i in range(30))
        (root / "Knowledge").mkdir(parents=True)
        (root / "Knowledge" / "old.md").write_text(
            f"---\ntype: note\nsuperseded-by: \"[[new]]\"\n---\n\n{filler}\n", encoding="utf-8")
        (root / "Knowledge" / "discussion.md").write_text(
            f"---\ntype: note\n---\n\nWe talked about superseded-by: keys today.\n{filler}\n",
            encoding="utf-8")
        self.assertEqual(re_mod.superseded_paths(root), {"Knowledge/old.md"})

    def test_a_short_stub_and_a_policy_excluded_note_still_count_as_superseded(self):
        """The probe walks the whole vault, not `_candidate_notes`.

        "Is this note worth asking a question about" and "would this note give a
        confident wrong answer if it topped a result" are different questions. A
        100-char stub is not a case candidate, and a mirror note is excluded from
        the semantic index - but either one topping a result is precisely the
        failure `stale_hit_rate` exists to count.
        """
        import shutil
        import tempfile

        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        stub = "---\ntype: note\nsuperseded-by: \"[[new]]\"\n---\n\nMoved. See the new note.\n"
        self.assertLess(len(stub), 120, "the point of this test is a note too short to be a case")
        (root / "Knowledge").mkdir(parents=True)
        (root / "Knowledge" / "stub.md").write_text(stub, encoding="utf-8")
        mirror = root / "Architecture" / "skills" / "hermes" / "researcher" / "old.md"
        mirror.parent.mkdir(parents=True)
        mirror.write_text(stub, encoding="utf-8")

        # Neither is a case candidate...
        self.assertEqual(re_mod._candidate_notes(root), [])
        # ...and both are known to be superseded.
        self.assertEqual(
            re_mod.superseded_paths(root),
            {"Knowledge/stub.md", "Architecture/skills/hermes/researcher/old.md"},
        )


if __name__ == "__main__":
    unittest.main()

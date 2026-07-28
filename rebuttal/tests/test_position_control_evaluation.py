import argparse
import importlib.util
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "rebuttal/scripts/6_9_qwen_position_controls.py"
)
SPEC = importlib.util.spec_from_file_location(
    "qwen_position_controls",
    SCRIPT_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PositionControlEvaluationTest(unittest.TestCase):
    def test_vote_uses_drafts_and_ns_minus_one_suffixes(self):
        question = {
            "doc_id": "math-1",
            "gold_answer": "1",
        }
        draft_map = {
            ("math-1", draft_idx): {
                "draft_answer": "1",
                "draft_tokens": 10,
            }
            for draft_idx in range(4)
        }
        suffix_records = []
        assignments = []
        for draft_idx in range(4):
            assignments.append({
                "doc_id": "math-1",
                "draft_idx": draft_idx,
                "strategy": "random_uniform",
                "rollback_step": 0,
            })
            for suffix_idx in range(8):
                suffix_records.append({
                    "doc_id": "math-1",
                    "draft_idx": draft_idx,
                    "strategy": "random_uniform",
                    "suffix_idx": suffix_idx,
                    "suffix_answer": (
                        "1" if suffix_idx < 7 else "999"
                    ),
                    "suffix_tokens": 2,
                    "prefix_tokens": 0,
                })

        results = MODULE.evaluate(
            [question],
            draft_map,
            suffix_records,
            assignments,
            [(4, 8)],
            argparse.Namespace(
                dataset="math500",
                strategies=["random_uniform"],
            ),
        )
        row = results[1]
        self.assertEqual(row["variant"], "draft_plus_suffix")
        self.assertEqual(row["n_answers"], 32)
        self.assertEqual(row["n_draft_answers"], 4)
        self.assertEqual(row["n_suffix_answers"], 28)
        self.assertEqual(row["total_tokens"], 96)
        self.assertEqual(row["acc"], 1.0)


if __name__ == "__main__":
    unittest.main()

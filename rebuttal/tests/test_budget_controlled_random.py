import importlib.util
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "6_8_budget_controlled_random.py"
)
SPEC = importlib.util.spec_from_file_location("budget_random", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RandomRollbackTest(unittest.TestCase):
    def test_degenerate_drafts_roll_back_to_start(self):
        for n_steps in (0, 1):
            self.assertEqual(
                MODULE.compute_rollback_step_random(
                    "doc", 0, n_steps, seed=42
                ),
                0,
            )

    def test_location_is_stable_and_in_range(self):
        first = MODULE.compute_rollback_step_random(
            "doc-17", 3, 11, seed=42
        )
        second = MODULE.compute_rollback_step_random(
            "doc-17", 3, 11, seed=42
        )
        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 11)

    def test_locations_are_approximately_uniform(self):
        n_steps = 5
        counts = [0] * n_steps
        for doc_idx in range(10_000):
            step = MODULE.compute_rollback_step_random(
                f"doc-{doc_idx}", doc_idx % 16, n_steps, seed=42
            )
            counts[step] += 1
        for count in counts:
            self.assertGreater(count, 1_800)
            self.assertLess(count, 2_200)


if __name__ == "__main__":
    unittest.main()

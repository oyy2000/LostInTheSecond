import unittest

from rebuttal.src.position_controls import (
    additional_suffixes_per_draft,
    final_step_prefix,
    final_step_rollback_step,
    fixed_rollback_step,
    random_rollback_step,
    suffixes_per_draft,
)


class PositionControlTest(unittest.TestCase):
    def test_fixed_positions_target_fractional_step(self):
        self.assertEqual(fixed_rollback_step(4, 0.25), 0)
        self.assertEqual(fixed_rollback_step(4, 0.50), 1)
        self.assertEqual(fixed_rollback_step(4, 0.75), 2)
        self.assertEqual(fixed_rollback_step(8, 0.25), 1)
        self.assertEqual(fixed_rollback_step(8, 0.50), 3)
        self.assertEqual(fixed_rollback_step(8, 0.75), 5)

    def test_random_position_is_stable_and_valid(self):
        value = random_rollback_step("doc-1", 3, 9, seed=42)
        self.assertEqual(
            value,
            random_rollback_step("doc-1", 3, 9, seed=42),
        )
        self.assertGreaterEqual(value, 0)
        self.assertLess(value, 9)

    def test_final_step_prefix_retains_all_prior_steps(self):
        steps = [
            "Set up the equation.",
            "Solve the equation.",
            "Final Answer: \\boxed{42}",
        ]
        self.assertEqual(
            final_step_prefix(steps),
            "Set up the equation.\n\nSolve the equation.\n\n",
        )

    def test_single_step_draft_regenerates_the_whole_step(self):
        self.assertEqual(final_step_rollback_step(1), 0)
        self.assertEqual(final_step_prefix(["Only step"]), "")

    def test_final_step_rollback_retains_n_minus_one_steps(self):
        self.assertEqual(final_step_rollback_step(5), 4)

    def test_budget_reuses_suffixes_across_nd_configs(self):
        counts = suffixes_per_draft([(4, 8), (8, 4), (16, 2)])
        self.assertEqual(sum(counts.values()), 64)
        self.assertEqual(counts[0], 8)
        self.assertEqual(counts[4], 4)
        self.assertEqual(counts[8], 2)

    def test_draft_answers_reduce_required_suffixes(self):
        counts = additional_suffixes_per_draft(
            [(4, 8), (8, 4), (16, 2)]
        )
        self.assertEqual(sum(counts.values()), 48)
        self.assertEqual(counts[0], 7)
        self.assertEqual(counts[4], 3)
        self.assertEqual(counts[8], 1)

if __name__ == "__main__":
    unittest.main()

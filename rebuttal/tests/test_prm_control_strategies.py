import unittest

from rebuttal.src.prm_control_strategies import (
    MAX_PRM_DROP_SIGNAL,
    PRM_BELOW_THRESHOLD_SIGNAL,
    max_prm_drop_assignment,
    prm_below_threshold_assignment,
)


class PrmControlStrategiesTest(unittest.TestCase):
    def test_max_drop_selects_first_largest_adjacent_decrease(self):
        assignment = max_prm_drop_assignment(
            [0.95, 0.80, 0.82, 0.67],
            4,
        )
        self.assertEqual(assignment["signal"], MAX_PRM_DROP_SIGNAL)
        self.assertEqual(assignment["rollback_step"], 1)
        self.assertTrue(assignment["triggered"])
        self.assertAlmostEqual(assignment["maximum_drop"], 0.15)

    def test_max_drop_is_defined_for_monotonic_increase(self):
        assignment = max_prm_drop_assignment([0.2, 0.4, 0.7], 3)
        self.assertEqual(assignment["rollback_step"], 1)
        self.assertTrue(assignment["triggered"])
        self.assertAlmostEqual(assignment["maximum_drop"], -0.2)

    def test_one_step_max_drop_uses_step_zero(self):
        assignment = max_prm_drop_assignment([0.9], 1)
        self.assertEqual(assignment["rollback_step"], 0)
        self.assertFalse(assignment["triggered"])

    def test_absolute_threshold_uses_first_score_below_threshold(self):
        assignment = prm_below_threshold_assignment(
            [0.95, 0.81, 0.79, 0.60],
            4,
            0.8,
        )
        self.assertEqual(
            assignment["signal"],
            PRM_BELOW_THRESHOLD_SIGNAL,
        )
        self.assertEqual(assignment["rollback_step"], 2)
        self.assertTrue(assignment["triggered"])

    def test_absolute_threshold_can_trigger_at_first_step(self):
        assignment = prm_below_threshold_assignment(
            [0.7, 0.9],
            2,
            0.8,
        )
        self.assertEqual(assignment["rollback_step"], 0)
        self.assertTrue(assignment["triggered"])

    def test_absolute_threshold_falls_back_before_last(self):
        assignment = prm_below_threshold_assignment(
            [0.95, 0.90, 0.85],
            3,
            0.8,
        )
        self.assertEqual(assignment["rollback_step"], 2)
        self.assertFalse(assignment["triggered"])

    def test_score_count_mismatch_fails(self):
        with self.assertRaises(ValueError):
            max_prm_drop_assignment([0.9], 2)


if __name__ == "__main__":
    unittest.main()

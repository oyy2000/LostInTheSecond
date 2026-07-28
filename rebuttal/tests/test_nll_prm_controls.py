import unittest

from rebuttal.src.nll_prm_controls import (
    NLL_SIGNAL,
    PRM_SIGNAL,
    compute_step_mean_nll,
    nll_rollback_assignment,
    prm_rollback_assignment,
    retained_prefix,
    step_char_bounds,
)


class NllPrmControlsTest(unittest.TestCase):
    def test_step_nll_uses_character_spans(self):
        response = "first step\n\nsecond step"
        steps = ["first step", "second step"]
        bounds = step_char_bounds(response, steps)
        nlls = compute_step_mean_nll(
            [-0.2, -0.4, -0.8, -1.0],
            [0, 6, 12, 19],
            bounds,
        )
        self.assertAlmostEqual(nlls[0], 0.3)
        self.assertAlmostEqual(nlls[1], 0.9)

    def test_nll_first_increase_and_fallback(self):
        triggered = nll_rollback_assignment(
            [0.2, 0.3, 0.7, 0.8], 4, threshold=0.2
        )
        self.assertEqual(triggered["signal"], NLL_SIGNAL)
        self.assertEqual(triggered["rollback_step"], 2)
        self.assertTrue(triggered["triggered"])

        fallback = nll_rollback_assignment(
            [0.2, 0.3, 0.4, 0.5], 4, threshold=0.2
        )
        self.assertEqual(fallback["rollback_step"], 3)
        self.assertFalse(fallback["triggered"])

    def test_prm_first_drop_and_fallback(self):
        triggered = prm_rollback_assignment(
            [0.9, 0.7, 0.4, 0.3], 4, threshold=0.1
        )
        self.assertEqual(triggered["signal"], PRM_SIGNAL)
        self.assertEqual(triggered["rollback_step"], 1)
        self.assertTrue(triggered["triggered"])

        fallback = prm_rollback_assignment(
            [0.9, 0.85, 0.8], 3, threshold=0.1
        )
        self.assertEqual(fallback["rollback_step"], 2)
        self.assertFalse(fallback["triggered"])

    def test_retained_prefix_matches_original_6_8_semantics(self):
        steps = ["one", "two", "three"]
        self.assertEqual(retained_prefix(steps, 0), "")
        self.assertEqual(retained_prefix(steps, 2), "one\n\ntwo\n\n")

    def test_missing_step_tokens_fail_loudly(self):
        with self.assertRaises(ValueError):
            compute_step_mean_nll([-0.5], [0], [(10, 20)])


if __name__ == "__main__":
    unittest.main()

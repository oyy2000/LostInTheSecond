import importlib.util
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "6_15_summarize_qwen_seed_variance.py"
)
SPEC = importlib.util.spec_from_file_location(
    "seed_variance_summary", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SeedVarianceSummaryTest(unittest.TestCase):
    def test_uses_sample_variance_across_all_requested_seeds(self):
        rows = [
            {
                "dataset": "math500",
                "method": "Random_nd4_ns8",
                "seed": seed,
                "acc": value,
                "tokens_per_q": 10.0 + seed,
            }
            for seed, value in zip([1, 2, 3], [0.5, 0.6, 0.7])
        ]
        summary = MODULE._summarize(rows, [1, 2, 3])
        self.assertEqual(len(summary), 1)
        self.assertAlmostEqual(summary[0]["acc_mean"], 0.6)
        self.assertAlmostEqual(summary[0]["acc_variance"], 0.01)
        self.assertAlmostEqual(summary[0]["acc_std"], 0.1)

    def test_missing_seed_fails(self):
        rows = [
            {
                "dataset": "math500",
                "method": "Greedy@1",
                "seed": 1,
                "acc": 0.5,
            }
        ]
        with self.assertRaises(RuntimeError):
            MODULE._summarize(rows, [1, 2])


if __name__ == "__main__":
    unittest.main()

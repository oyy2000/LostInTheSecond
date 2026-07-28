import importlib.util
import json
import unittest
from pathlib import Path

from rebuttal.src.threshold_sweep import (
    active_threshold_columns,
    build_eval_summary_table,
    checkpoint_identity,
    signal_result_dir,
    signal_threshold_name,
    suffixes_per_draft,
    threshold_slug,
    thresholds_for_signal,
    trial_name,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEEDED_ENTRYPOINT = (
    PROJECT_ROOT
    / "rebuttal/scripts/6_8_budget_controlled_multisignal_seeded.py"
)
SPEC = importlib.util.spec_from_file_location(
    "seeded_multisignal_entrypoint", SEEDED_ENTRYPOINT
)
SEEDED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SEEDED)
PIPELINE = SEEDED._load_pipeline()
SINGLE_THRESHOLD_CONFIG = (
    PROJECT_ROOT
    / "rebuttal/configs/"
    "6_19_qwen_nll_0p1_prm_0p05_recompute_sc32.json"
)


class ThresholdSweepTest(unittest.TestCase):
    def test_single_threshold_config_recomputes_sc32_in_isolated_root(self):
        config = json.loads(SINGLE_THRESHOLD_CONFIG.read_text())
        self.assertEqual(config["nll_thresholds"], [0.1])
        self.assertEqual(config["prm_thresholds"], [0.05])
        self.assertEqual(
            config["sc32_checkpoint_policy"],
            "fresh_isolated_shared_checkpoint",
        )
        self.assertEqual(
            config["sc32_supplement_per_question"],
            config["budget"] - max(config["nd"]),
        )
        self.assertEqual(
            set(config["datasets"]),
            {
                "gsm8k",
                "amc2023",
                "aime2024",
                "olympiadbench",
                "2wikimultihopqa_open",
            },
        )
        self.assertIn(
            "sc32_recomputed",
            config["results_root"],
        )

    def test_concurrent_vllm_workers_receive_distinct_ports(self):
        ports = [
            PIPELINE._vllm_port_for_worker(shard_id)
            for shard_id in range(8)
        ]
        self.assertEqual(len(ports), len(set(ports)))
        self.assertTrue(all(20000 <= port < 60000 for port in ports))

    def test_grid_names_are_stable_and_searchable(self):
        self.assertEqual(threshold_slug(0.05), "0p05")
        self.assertEqual(trial_name(0.1, 0.05), "nll_0p1_prm_0p05")

    def test_signal_checkpoint_names_do_not_form_cartesian_grid(self):
        self.assertEqual(
            signal_threshold_name("nll_drop_fb_last", 0.1),
            "nll_0p1",
        )
        self.assertEqual(
            signal_threshold_name("prm_drop_fb_last", 0.05),
            "prm_0p05",
        )
        self.assertEqual(
            signal_result_dir(
                Path("/results"),
                "prm_drop_fb_last",
                0.05,
            ),
            Path("/results/signals/prm_0p05"),
        )

    def test_thresholds_are_selected_per_signal(self):
        config = {
            "nll_thresholds": [0.1, 0.2],
            "prm_thresholds": [0.05, 0.1],
        }
        self.assertEqual(
            thresholds_for_signal(config, "nll_drop_fb_last"),
            [0.1, 0.2],
        )
        self.assertEqual(
            thresholds_for_signal(config, "prm_drop_fb_last"),
            [0.05, 0.1],
        )

    def test_active_threshold_columns_do_not_show_inactive_threshold(self):
        self.assertEqual(
            active_threshold_columns("nll_drop_fb_last", 0.2),
            (0.2, None),
        )
        self.assertEqual(
            active_threshold_columns("prm_drop_fb_last", 0.05),
            (None, 0.05),
        )

    def test_consolidated_table_labels_nll_and_prm_thresholds(self):
        rows = [
            {
                "method": "SC@32",
                "nd": 0,
                "ns": 32,
                "acc": 0.7,
                "tokens_per_q": 100.0,
                "n_answers": 32,
                "nll_threshold": None,
                "prm_threshold": None,
            },
            {
                "method": "rollback_nll",
                "strategy": "nll_drop_fb_last",
                "variant": "fair",
                "nd": 4,
                "ns": 8,
                "acc": 0.72,
                "tokens_per_q": 80.0,
                "n_answers": 32,
                "nll_threshold": 0.2,
                "prm_threshold": None,
            },
            {
                "method": "rollback_prm",
                "strategy": "prm_drop_fb_last",
                "variant": "fair",
                "nd": 4,
                "ns": 8,
                "acc": 0.73,
                "tokens_per_q": 90.0,
                "n_answers": 32,
                "nll_threshold": None,
                "prm_threshold": 0.05,
            },
        ]
        table = build_eval_summary_table("math500", rows)
        self.assertIn("| NLL threshold | PRM threshold |", table)
        self.assertIn(
            "| rollback_nll | nll_drop_fb_last | 0.20 | - |",
            table,
        )
        self.assertIn(
            "| rollback_prm | prm_drop_fb_last | - | 0.05 |",
            table,
        )

    def test_full_suffix_allocation_supports_every_configuration(self):
        self.assertEqual(
            suffixes_per_draft([4, 8, 16], 32),
            {
                **{index: 8 for index in range(4)},
                **{index: 4 for index in range(4, 8)},
                **{index: 2 for index in range(8, 16)},
            },
        )

    def test_fair_answer_allocation_keeps_original_draft_vote(self):
        self.assertEqual(
            PIPELINE.rollback_answer_allocation("fair", 8),
            (1, 7),
        )
        self.assertEqual(
            PIPELINE.rollback_answer_allocation("fair", 2),
            (1, 1),
        )

    def test_full_answer_allocation_keeps_all_suffix_votes(self):
        self.assertEqual(
            PIPELINE.rollback_answer_allocation("full", 8),
            (1, 8),
        )

    def test_fair_new_answer_allocation_uses_only_new_suffixes(self):
        self.assertEqual(
            PIPELINE.rollback_answer_allocation("fair_new", 8),
            (0, 8),
        )
        self.assertEqual(
            PIPELINE.rollback_answer_allocation("fair_new", 2),
            (0, 2),
        )

    def test_suffix_identity_includes_rollback_and_sample(self):
        base = {
            "task_type": "suffix",
            "doc_id": "math500_0",
            "draft_idx": 0,
            "strategy": "nll_drop_fb_last",
            "rollback_step": 2,
            "suffix_idx": 0,
        }
        changed = dict(base, rollback_step=3)
        self.assertNotEqual(
            checkpoint_identity(base), checkpoint_identity(changed)
        )

    def test_pipeline_checkpoint_part_identity_matches_suffix_identity(self):
        record = {
            "task_type": "suffix",
            "doc_id": "math500_0",
            "draft_idx": 0,
            "strategy": "nll_drop_fb_last",
            "rollback_step": 2,
            "suffix_idx": 0,
        }
        self.assertEqual(
            PIPELINE.checkpoint_record_identity(record),
            checkpoint_identity(record),
        )

    def test_seeded_wrapper_allows_only_scoring_workers_without_seed(self):
        self.assertTrue(SEEDED._is_scoring_worker(["--_logprob"]))
        self.assertTrue(SEEDED._is_scoring_worker(["--_prm"]))
        self.assertFalse(
            SEEDED._is_scoring_worker(["--_shard-id", "0"])
        )


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from rebuttal.src.position_control_audit import (
    EXPECTED_CONFIGS,
    audit_position_control_results,
)
from rebuttal.src.position_controls import (
    ALL_STRATEGIES,
    FINAL_STEP_STRATEGY,
    FIXED_FRACTIONS,
    final_step_rollback_step,
    fixed_rollback_step,
    random_rollback_step,
)
from rebuttal.src.stable_sampling import derive_sampling_seed


class PositionControlAuditTest(unittest.TestCase):
    def test_complete_nondefault_seed_fixture_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            config = {
                "model": "Qwen/Qwen2.5-3B-Instruct",
                "dataset": "hotpotqa",
                "n_questions": 1,
                "dataset_seed": 42,
                "random_seed": 123,
                "generation_seed": 123,
                "reuse_checkpoint": "source.jsonl",
                "budget": 32,
                "rollback_configs": EXPECTED_CONFIGS,
                "strategies": ALL_STRATEGIES,
                "fixed_position_semantics": (
                    "target step = ceil(fraction * n_steps); "
                    "retain earlier steps"
                ),
                "final_step_semantics": (
                    "target the final split reasoning step; "
                    "retain earlier steps"
                ),
                "answer_budget_semantics": (
                    "nd original draft answers plus "
                    "nd*(ns-1) suffix answers"
                ),
                "temperature": 0.7,
                "top_p": 0.95,
                "max_tokens": 2048,
            }
            (result_dir / "run_config.json").write_text(
                json.dumps(config), encoding="utf-8"
            )

            checkpoint = result_dir / "checkpoint.jsonl"
            with checkpoint.open("w", encoding="utf-8") as handle:
                for draft_idx in range(16):
                    handle.write(json.dumps({
                        "task_key": f"draft|q0|{draft_idx}",
                        "task_type": "draft",
                        "doc_id": "q0",
                        "draft_idx": draft_idx,
                        "n_steps": 4,
                    }) + "\n")
                for strategy in ALL_STRATEGIES:
                    for draft_idx in range(16):
                        count = (
                            7 if draft_idx < 4
                            else 3 if draft_idx < 8
                            else 1
                        )
                        if strategy == "random_uniform":
                            rollback_step = random_rollback_step(
                                "q0", draft_idx, 4, 123
                            )
                        elif strategy in FIXED_FRACTIONS:
                            rollback_step = fixed_rollback_step(
                                4, FIXED_FRACTIONS[strategy]
                            )
                        elif strategy == FINAL_STEP_STRATEGY:
                            rollback_step = final_step_rollback_step(4)
                        else:
                            self.fail(f"Unexpected strategy: {strategy}")
                        for suffix_idx in range(count):
                            record = {
                                "task_key": (
                                    f"suffix|{strategy}|q0|"
                                    f"{draft_idx}|{suffix_idx}"
                                ),
                                "task_type": "suffix",
                                "doc_id": "q0",
                                "draft_idx": draft_idx,
                                "strategy": strategy,
                                "suffix_idx": suffix_idx,
                                "rollback_step": rollback_step,
                                "generation_kind": "suffix",
                                "suffix_tokens": 1,
                                "random_seed": (
                                    123
                                    if strategy == "random_uniform"
                                    else None
                                ),
                            }
                            if strategy == FINAL_STEP_STRATEGY:
                                record["sampling_seed"] = (
                                    derive_sampling_seed(123, record)
                                )
                            handle.write(json.dumps(record) + "\n")

            assignments = result_dir / "rollback_assignments.jsonl"
            with assignments.open("w", encoding="utf-8") as handle:
                for strategy in ALL_STRATEGIES:
                    for draft_idx in range(16):
                        if strategy == "random_uniform":
                            rollback_step = random_rollback_step(
                                "q0", draft_idx, 4, 123
                            )
                        elif strategy in FIXED_FRACTIONS:
                            rollback_step = fixed_rollback_step(
                                4, FIXED_FRACTIONS[strategy]
                            )
                        elif strategy == FINAL_STEP_STRATEGY:
                            rollback_step = final_step_rollback_step(4)
                        else:
                            self.fail(f"Unexpected strategy: {strategy}")
                        handle.write(json.dumps({
                            "doc_id": "q0",
                            "draft_idx": draft_idx,
                            "strategy": strategy,
                            "n_steps": 4,
                            "rollback_step": rollback_step,
                            "target_step": (
                                4
                                if strategy == FINAL_STEP_STRATEGY
                                else rollback_step + 1
                            ),
                            "target_fraction": (
                                1.0
                                if strategy == FINAL_STEP_STRATEGY
                                else (rollback_step + 1) / 4
                            ),
                        }) + "\n")

            results = [{
                "method": "Greedy@1",
                "acc": 0.0,
                "f1": 0.0,
                "tokens_per_q": 1.0,
            }]
            for strategy in ALL_STRATEGIES:
                for nd, ns in EXPECTED_CONFIGS:
                    results.append({
                        "method": f"{strategy}_nd{nd}_ns{ns}",
                        "strategy": strategy,
                        "nd": nd,
                        "ns": ns,
                        "n_answers": 32,
                        "n_draft_answers": nd,
                        "n_suffix_answers": nd * (ns - 1),
                        "variant": "draft_plus_suffix",
                        "acc": 0.5,
                        "f1": 0.5,
                        "tokens_per_q": 100.0,
                    })
            (result_dir / "eval_summary.json").write_text(
                json.dumps(results), encoding="utf-8"
            )
            (result_dir / "phase_times.json").write_text(
                json.dumps({"draft": 1.0, "suffix": 2.0}),
                encoding="utf-8",
            )
            for name in (
                "eval_summary_table.md",
                "position_controls.png",
                "position_controls.pdf",
            ):
                (result_dir / name).write_bytes(b"fixture")

            report = audit_position_control_results(
                result_dir,
                expected_questions=1,
            )
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["checkpoint_records"], 256)
            self.assertEqual(report["suffix_records"], 240)
            self.assertEqual(report["sampling_seeds_checked"], 48)


if __name__ == "__main__":
    unittest.main()

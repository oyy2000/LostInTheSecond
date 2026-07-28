import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from rebuttal.src.seeded_generation import run_seeded_shard
from rebuttal.src.stable_sampling import (
    derive_sampling_seed,
    task_seed_identity,
)


class StableSamplingTest(unittest.TestCase):
    def _sc_tasks(self, doc_id="math-1"):
        drafts = [
            {
                "task_type": "draft",
                "doc_id": doc_id,
                "draft_idx": index,
            }
            for index in range(16)
        ]
        supplements = [
            {
                "task_type": "fullsc",
                "doc_id": doc_id,
                "sc_idx": index,
            }
            for index in range(16)
        ]
        return drafts + supplements

    def test_sc32_requests_have_distinct_seeds(self):
        seeds = [
            derive_sampling_seed(42, task)
            for task in self._sc_tasks()
        ]
        self.assertEqual(len(seeds), 32)
        self.assertEqual(len(set(seeds)), 32)

    def test_seed_is_stable(self):
        task = {
            "task_type": "suffix",
            "doc_id": "math-1",
            "draft_idx": 3,
            "strategy": "prm_drop_fb_last",
            "rollback_step": 2,
            "suffix_idx": 7,
        }
        self.assertEqual(
            derive_sampling_seed(123, task),
            derive_sampling_seed(123, dict(task)),
        )

    def test_run_seeds_change_request_seeds(self):
        task = self._sc_tasks()[0]
        values = {
            derive_sampling_seed(run_seed, task)
            for run_seed in (42, 123, 456, 789, 1024)
        }
        self.assertEqual(len(values), 5)

    def test_task_identity_distinguishes_generation_phases(self):
        draft, fullsc = self._sc_tasks()[:1] + self._sc_tasks()[16:17]
        self.assertNotEqual(
            task_seed_identity(draft),
            task_seed_identity(fullsc),
        )

    def test_final_step_requests_have_distinct_seeds(self):
        tasks = [
            {
                "task_type": "suffix",
                "doc_id": "hotpot-1",
                "draft_idx": 0,
                "strategy": "final_step_only",
                "rollback_step": 3,
                "suffix_idx": suffix_idx,
            }
            for suffix_idx in range(2)
        ]
        seeds = [derive_sampling_seed(42, task) for task in tasks]
        self.assertEqual(len(set(seeds)), 2)

    def test_worker_passes_one_sampling_params_per_request(self):
        fake_vllm = ModuleType("vllm")

        class FakeSamplingParams:
            def __init__(self, **kwargs):
                self.seed = kwargs["seed"]

        class FakeTokenizer:
            @staticmethod
            def encode(prompt, add_special_tokens=False):
                return [len(prompt)]

        class FakeLLM:
            observed_seeds = []

            def __init__(self, **kwargs):
                pass

            @staticmethod
            def get_tokenizer():
                return FakeTokenizer()

            @classmethod
            def generate(cls, prompts, sampling_params):
                cls.observed_seeds.extend(
                    params.seed for params in sampling_params
                )
                return [
                    SimpleNamespace(
                        outputs=[
                            SimpleNamespace(
                                text=f"sample-{params.seed}",
                                token_ids=[1],
                            )
                        ]
                    )
                    for params in sampling_params
                ]

        fake_vllm.LLM = FakeLLM
        fake_vllm.SamplingParams = FakeSamplingParams
        pipeline = SimpleNamespace(
            MODEL_ID="fake/model",
            DATASET="math500",
            GPU_MEM=0.9,
            MAX_MODEL_LEN=4096,
            MAX_TOKENS=32,
            TEMPERATURE=0.7,
            TOP_P=0.95,
            get_stop_tokens=lambda model: [],
            split_steps=lambda text: [text],
            extract_answer=lambda dataset, text: text,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "shard.jsonl"
            task_path = Path(temp_dir) / "tasks.json"
            timing_path = Path(temp_dir) / "worker_timing.json"
            tasks = [
                {
                    **task,
                    "gold_answer": "1",
                    "prompt": "identical prompt",
                    "_out": str(output_path),
                }
                for task in self._sc_tasks()[:2]
            ]
            task_path.write_text(json.dumps(tasks), encoding="utf-8")
            args = argparse.Namespace(
                _seed=42,
                _gpu="0",
                _task_file=str(task_path),
                _shard_id=0,
                _timing_file=str(timing_path),
                _timing_phase="draft",
                _timing_run_id="stable-sampling-test",
            )
            with patch.dict(sys.modules, {"vllm": fake_vllm}):
                run_seeded_shard(pipeline, args)

            records = [
                json.loads(line)
                for line in output_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            timing = json.loads(
                timing_path.read_text(encoding="utf-8")
            )

        self.assertEqual(len(FakeLLM.observed_seeds), 2)
        self.assertEqual(len(set(FakeLLM.observed_seeds)), 2)
        self.assertEqual(
            FakeLLM.observed_seeds,
            [record["sampling_seed"] for record in records],
        )
        self.assertEqual(timing["phase"], "draft")
        self.assertEqual(timing["tasks"], 2)
        self.assertEqual(timing["batches"][0]["generated_tokens"], 2)


if __name__ == "__main__":
    unittest.main()

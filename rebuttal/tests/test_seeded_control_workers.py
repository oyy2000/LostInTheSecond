import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from subprocess import CompletedProcess
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_script(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, PROJECT_ROOT / "rebuttal/scripts" / filename
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


POSITION = _load_script(
    "qwen_position_seed_worker",
    "6_9_qwen_position_controls.py",
)
SIGNAL = _load_script(
    "qwen_signal_seed_worker",
    "6_11_qwen_nll_prm_controls.py",
)


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.seed = kwargs["seed"]


class FakeTokenizer:
    @staticmethod
    def encode(text, add_special_tokens=False):
        return [len(text)]


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


class SeededControlWorkerTest(unittest.TestCase):
    def setUp(self):
        FakeLLM.observed_seeds = []
        self.fake_vllm = ModuleType("vllm")
        self.fake_vllm.LLM = FakeLLM
        self.fake_vllm.SamplingParams = FakeSamplingParams

    def test_position_worker_uses_distinct_request_seeds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "out.jsonl"
            tasks = [
                {
                    "task_key": f"draft|doc|{draft_idx}",
                    "task_type": "draft",
                    "doc_id": "doc",
                    "draft_idx": draft_idx,
                    "gold_answer": "1",
                    "prompt": "identical prompt",
                    "_out": str(output),
                }
                for draft_idx in range(2)
            ]
            task_file = root / "tasks.json"
            task_file.write_text(json.dumps(tasks), encoding="utf-8")
            args = argparse.Namespace(
                _gpu="0",
                _task_file=str(task_file),
                _shard_id=0,
                _seed=42,
                model="fake/model",
                gpu_memory_utilization=0.1,
                dataset="math500",
            )
            with (
                patch.dict(sys.modules, {"vllm": self.fake_vllm}),
                patch.object(POSITION, "get_stop_tokens", return_value=[]),
                patch.object(POSITION, "split_steps", side_effect=lambda x: [x]),
                patch.object(POSITION, "extract_answer", side_effect=lambda d, x: x),
            ):
                POSITION.run_worker(args)
            records = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(len(set(FakeLLM.observed_seeds)), 2)
        self.assertEqual(
            FakeLLM.observed_seeds,
            [record["sampling_seed"] for record in records],
        )

    def test_signal_worker_uses_distinct_suffix_seeds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "out.jsonl"
            tasks = [
                {
                    "task_key": f"suffix|nll|doc|0|{suffix_idx}",
                    "task_type": "suffix",
                    "doc_id": "doc",
                    "draft_idx": 0,
                    "strategy": "nll_drop_fb_last",
                    "rollback_step": 1,
                    "suffix_idx": suffix_idx,
                    "gold_answer": "1",
                    "retained_text": "step",
                    "prompt": "identical prompt",
                    "_out": str(output),
                }
                for suffix_idx in range(2)
            ]
            task_file = root / "tasks.json"
            task_file.write_text(json.dumps(tasks), encoding="utf-8")
            args = argparse.Namespace(
                _gpu="0",
                _task_file=task_file,
                _shard_id=0,
                _seed=42,
                dataset="math500",
            )
            config = {
                "model": "fake/model",
                "temperature": 0.7,
                "top_p": 0.95,
                "max_tokens": 32,
                "max_model_len": 128,
                "generation_batch_size": 512,
                "datasets": {
                    "math500": {"gpu_memory_utilization": 0.1}
                },
            }
            with (
                patch.dict(sys.modules, {"vllm": self.fake_vllm}),
                patch.object(SIGNAL, "get_stop_tokens", return_value=[]),
                patch.object(SIGNAL, "extract_answer", side_effect=lambda d, x: x),
            ):
                SIGNAL.run_suffix_worker(args, config)
            records = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(len(set(FakeLLM.observed_seeds)), 2)
        self.assertEqual(
            FakeLLM.observed_seeds,
            [record["sampling_seed"] for record in records],
        )

    def test_nll_memory_gate_checks_the_assigned_gpu(self):
        args = argparse.Namespace(
            dataset="math500",
            _gpu="2",
            _shard_id=1,
        )
        config = {
            "datasets": {
                "math500": {"nll_min_free_memory_mib": 20000}
            }
        }
        completed = CompletedProcess(
            args=[],
            returncode=0,
            stdout="24576\n",
            stderr="",
        )
        with patch.object(
            SIGNAL.subprocess,
            "run",
            return_value=completed,
        ) as run:
            SIGNAL._wait_for_nll_gpu_memory(args, config)

        command = run.call_args.args[0]
        self.assertEqual(command[1:3], ["--id", "2"])

    def test_causal_token_logprobs_use_previous_position(self):
        import torch

        logits = torch.tensor([
            [0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ])
        input_ids = torch.tensor([0, 2, 0])
        observed = SIGNAL.causal_token_logprobs(
            logits,
            input_ids,
            token_positions=[1, 2],
            chunk_size=1,
        )
        expected = [
            torch.log_softmax(logits[0], dim=0)[2].item(),
            torch.log_softmax(logits[1], dim=0)[0].item(),
        ]
        for actual, target in zip(observed, expected):
            self.assertAlmostEqual(actual, target, places=6)

    def test_shard_pruning_removes_moved_and_stale_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "shard.jsonl"
            tasks = [
                {
                    "task_key": f"suffix|nll|doc|0|{suffix_idx}",
                    "task_type": "suffix",
                    "doc_id": "doc",
                    "draft_idx": 0,
                    "strategy": "nll_drop_fb_last",
                    "rollback_step": 2,
                    "suffix_idx": suffix_idx,
                }
                for suffix_idx in range(2)
            ]
            valid = dict(tasks[0])
            valid["sampling_seed"] = SIGNAL.derive_sampling_seed(
                42, tasks[0]
            )
            stale_assignment = dict(tasks[1])
            stale_assignment["rollback_step"] = 1
            stale_assignment["sampling_seed"] = (
                SIGNAL.derive_sampling_seed(42, stale_assignment)
            )
            moved = {
                "task_key": "suffix|nll|other|0|0",
                "task_type": "suffix",
            }
            path.write_text(
                "\n".join(
                    json.dumps(record)
                    for record in (valid, stale_assignment, moved)
                )
                + "\n",
                encoding="utf-8",
            )

            removed = SIGNAL._prune_shard_checkpoint(
                path,
                tasks,
                "suffix",
                {"generation_seed": 42},
            )
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(removed, 2)
        self.assertEqual(records, [valid])

    def test_nll_shard_pruning_requires_matching_backend(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "shard.jsonl"
            task = {
                "task_key": "nll|doc|0",
                "task_type": "nll",
                "scoring_backend": "transformers",
            }
            path.write_text(
                json.dumps({
                    "task_key": task["task_key"],
                    "task_type": "nll",
                    "scoring_backend": "vllm",
                })
                + "\n",
                encoding="utf-8",
            )

            removed = SIGNAL._prune_shard_checkpoint(
                path,
                [task],
                "nll",
                {"nll_backend": "transformers"},
            )
            contents = path.read_text(encoding="utf-8")

        self.assertEqual(removed, 1)
        self.assertEqual(contents, "")


if __name__ == "__main__":
    unittest.main()

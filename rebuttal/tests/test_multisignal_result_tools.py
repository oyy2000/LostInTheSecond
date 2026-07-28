import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = (
    PROJECT_ROOT
    / "rebuttal"
    / "scripts"
    / "6_8_4_build_multisignal_comparison.py"
)
VALIDATE_SCRIPT = (
    PROJECT_ROOT
    / "rebuttal"
    / "scripts"
    / "6_8_5_validate_multisignal_results.py"
)


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_and_validate_complete_fixture(tmp_path):
    dataset = "math500"
    signals = ["nll_drop_fb_last", "prm_drop_fb_last"]
    nd_values = [4, 8, 16]
    budget = 32
    n_questions = 2
    results_root = tmp_path / "results"
    out_dir = results_root / dataset
    out_dir.mkdir(parents=True)

    config = {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "prm_model": "Qwen/Qwen2.5-Math-PRM-7B",
        "datasets": {dataset: {"n_sample": n_questions}},
        "signals": signals,
        "budget": budget,
        "nd": nd_values,
        "seed": 42,
        "gpus": "0,1,2,3",
        "results_root": str(results_root),
    }
    config_path = tmp_path / "config.json"
    write_json(config_path, config)

    records = []
    step_tokens = {}
    for question_index in range(n_questions):
        doc_id = f"math500_{question_index}"
        for draft_idx in range(max(nd_values)):
            records.extend(
                [
                    {
                        "task_type": "draft",
                        "doc_id": doc_id,
                        "draft_idx": draft_idx,
                    },
                    {
                        "task_type": "prm",
                        "doc_id": doc_id,
                        "draft_idx": draft_idx,
                    },
                    {
                        "task_type": "logprob",
                        "doc_id": doc_id,
                        "draft_idx": draft_idx,
                    },
                ]
            )
            step_tokens[f"{doc_id}|{draft_idx}"] = {
                "cumulative_tokens": [5]
            }
            suffix_count = max(
                budget // nd for nd in nd_values if draft_idx < nd
            )
            for signal in signals:
                for suffix_idx in range(suffix_count):
                    records.append(
                        {
                            "task_type": "suffix",
                            "doc_id": doc_id,
                            "draft_idx": draft_idx,
                            "strategy": signal,
                            "rollback_step": 0,
                            "suffix_idx": suffix_idx,
                        }
                    )
        for sc_idx in range(budget - max(nd_values)):
            records.append(
                {
                    "task_type": "fullsc",
                    "doc_id": doc_id,
                    "sc_idx": sc_idx,
                }
            )

    checkpoint = "\n".join(json.dumps(record) for record in records) + "\n"
    (out_dir / "checkpoint.jsonl").write_text(
        checkpoint, encoding="utf-8"
    )
    write_json(out_dir / "draft_step_tokens.json", step_tokens)

    eval_rows = [
        {
            "method": "SC@32",
            "nd": 0,
            "ns": 32,
            "acc": 0.5,
            "tokens_per_q": 3200.0,
            "total_tokens": 6400,
        }
    ]
    for signal_index, signal in enumerate(signals):
        for nd in nd_values:
            ns = budget // nd
            eval_rows.append(
                {
                    "method": f"rollback_{signal}_nd{nd}_ns{ns}_fair",
                    "strategy": signal,
                    "variant": "fair",
                    "nd": nd,
                    "ns": ns,
                    "acc": 0.55 + 0.01 * signal_index,
                    "tokens_per_q": 2000.0 + nd,
                    "total_tokens": 4000 + 2 * nd,
                }
            )
    write_json(out_dir / "eval_summary.json", eval_rows)
    for name in (
        "fig_efficiency_frontier.png",
        "fig_efficiency_frontier.pdf",
    ):
        (out_dir / name).write_bytes(b"fixture")

    subprocess.run(
        [
            sys.executable,
            str(BUILD_SCRIPT),
            "--config",
            str(config_path),
            "--dataset",
            dataset,
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(VALIDATE_SCRIPT),
            "--config",
            str(config_path),
            "--dataset",
            dataset,
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )

    report = json.loads(
        (out_dir / "validation_report.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "passed"
    assert report["n_questions"] == n_questions
    assert report["checkpoint_records"] == 384


class MultisignalResultToolsTest(unittest.TestCase):
    def test_build_and_validate_complete_fixture(self):
        with tempfile.TemporaryDirectory(
            prefix="rebuttal_multisignal_test_"
        ) as temp_dir:
            build_and_validate_complete_fixture(Path(temp_dir))


if __name__ == "__main__":
    unittest.main()

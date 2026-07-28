import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from rebuttal.src.wallclock import (
    RunTimingRecorder,
    WorkerTiming,
    build_per_question_timings,
    interval_union_seconds,
    load_manifest,
    publish_per_question_timings,
    summarize_per_question_timings,
    task_key,
    task_set_sha256,
)


def generation_tasks(count=4):
    return [
        {
            "task_type": "suffix",
            "doc_id": f"doc_{index // 2}",
            "draft_idx": index // 2,
            "strategy": "prm_drop_fb_last",
            "rollback_step": 1,
            "suffix_idx": index % 2,
        }
        for index in range(count)
    ]


class WallclockTest(unittest.TestCase):
    def setUp(self):
        self.environment_patch = patch(
            "rebuttal.src.wallclock.environment_metadata",
            return_value={"hostname": "test"},
        )
        self.environment_patch.start()

    def tearDown(self):
        self.environment_patch.stop()

    def test_task_digest_is_order_independent(self):
        tasks = generation_tasks()
        forward = task_set_sha256(task_key(task) for task in tasks)
        reverse = task_set_sha256(
            task_key(task) for task in reversed(tasks)
        )
        self.assertEqual(forward, reverse)

    def test_exact_checkpoint_reuse_inherits_historical_wallclock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks = generation_tasks()
            source = RunTimingRecorder(
                root / "source",
                mode="fresh",
                context={"dataset": "smoke"},
                run_id="source",
            )
            source.record_phase(
                "suffix",
                required=tasks,
                executed=tasks,
                current_wall_sec=12.5,
            )
            source.finalize()
            source_manifest = (
                root / "source/timing/timing_manifest.json"
            )

            resumed = RunTimingRecorder(
                root / "resumed",
                mode="auto",
                context={"dataset": "smoke"},
                source_manifests=[source_manifest],
                run_id="resumed",
            )
            event = resumed.record_phase(
                "suffix",
                required=tasks,
                executed=[],
                current_wall_sec=0.1,
            )
            manifest = resumed.finalize()

            self.assertEqual(event["status"], "fully_reused")
            self.assertEqual(
                event["reconstruction_kind"],
                "exact_historical_full",
            )
            self.assertEqual(
                event["reconstructed_from_scratch_wall_sec"],
                12.5,
            )
            self.assertEqual(
                manifest["reconstructed_from_scratch_wall_sec"],
                12.5,
            )

    def test_partial_source_task_set_is_not_scaled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks = generation_tasks()
            source = RunTimingRecorder(
                root / "source",
                mode="fresh",
                context={},
                run_id="source",
            )
            source.record_phase(
                "suffix",
                required=tasks,
                executed=tasks,
                current_wall_sec=10.0,
            )
            source.finalize()

            resumed = RunTimingRecorder(
                root / "resumed",
                mode="auto",
                context={},
                source_manifests=[
                    root / "source/timing/timing_manifest.json"
                ],
                run_id="resumed",
            )
            event = resumed.record_phase(
                "suffix",
                required=tasks[:2],
                executed=[],
                current_wall_sec=0.01,
            )
            resumed.finalize()

            self.assertEqual(event["status"], "fully_reused")
            self.assertIsNone(
                event["reconstructed_from_scratch_wall_sec"]
            )
            self.assertIsNone(event["reconstruction_kind"])

    def test_interrupted_local_run_reconstructs_completed_phase(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "experiment"
            tasks = generation_tasks()
            interrupted = RunTimingRecorder(
                root,
                mode="auto",
                context={"dataset": "smoke"},
                run_id="interrupted",
            )
            interrupted.record_phase(
                "suffix",
                required=tasks,
                executed=tasks,
                current_wall_sec=7.25,
            )
            # Deliberately do not finalize: a later phase failed.

            resumed = RunTimingRecorder(
                root,
                mode="auto",
                context={"dataset": "smoke"},
                run_id="resumed",
            )
            event = resumed.record_phase(
                "suffix",
                required=tasks,
                executed=[],
                current_wall_sec=0.1,
            )
            manifest = resumed.finalize()

            self.assertEqual(
                event["reconstruction_kind"],
                "exact_local_run_history",
            )
            self.assertEqual(
                event["reconstructed_from_scratch_wall_sec"],
                7.25,
            )
            self.assertEqual(
                manifest["reconstructed_from_scratch_wall_sec"],
                7.25,
            )

    def test_worker_timing_writes_batch_and_model_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "worker.json"
            tasks = generation_tasks(2)
            timing = WorkerTiming(
                path=path,
                phase="suffix",
                run_id="worker-smoke",
                shard_id=0,
                gpu_id="2",
                tasks=tasks,
            )
            started = time.perf_counter_ns()
            timing.record_model_load(started)
            batch_started = time.perf_counter_ns()
            timing.record_batch(
                tasks,
                batch_started,
                generated_tokens=17,
                input_tokens=23,
            )
            timing.record_task(
                tasks[0],
                started_unix_sec=10.0,
                finished_unix_sec=12.0,
                timing_source="test",
            )
            timing.finish()
            value = load_manifest(path)

            self.assertEqual(value["tasks"], 2)
            self.assertEqual(value["batches"][0]["generated_tokens"], 17)
            self.assertEqual(value["batches"][0]["input_tokens"], 23)
            self.assertEqual(value["task_events"][0]["active_wall_sec"], 2.0)
            self.assertGreaterEqual(value["worker_wall_sec"], 0)

    def test_per_question_generation_and_signal_active_wall(self):
        doc_id = "doc_0"
        drafts = [
            {
                "task_type": "draft",
                "doc_id": doc_id,
                "draft_idx": index,
            }
            for index in range(2)
        ]
        nll = [
            {
                "task_type": "logprob",
                "doc_id": doc_id,
                "draft_idx": index,
            }
            for index in range(2)
        ]
        suffixes = [
            {
                "task_type": "suffix",
                "doc_id": doc_id,
                "draft_idx": index,
                "strategy": "nll_drop_fb_last",
                "rollback_step": 1,
                "suffix_idx": 0,
            }
            for index in range(2)
        ]

        def event(task, started, finished):
            return {
                "task_key": task_key(task),
                "started_unix_sec": started,
                "finished_unix_sec": finished,
            }

        events = {
            task_key(drafts[0]): event(drafts[0], 0.0, 2.0),
            task_key(drafts[1]): event(drafts[1], 1.0, 3.0),
            task_key(nll[0]): event(nll[0], 5.0, 6.0),
            task_key(nll[1]): event(nll[1], 5.5, 7.0),
            task_key(suffixes[0]): event(
                suffixes[0], 10.0, 12.0
            ),
            task_key(suffixes[1]): event(
                suffixes[1], 11.0, 13.0
            ),
        }
        rows = build_per_question_timings(
            [*drafts, *nll, *suffixes],
            events,
            ["nll_drop_fb_last"],
        )

        self.assertEqual(interval_union_seconds(list(events.values())), 8.0)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["complete"])
        self.assertEqual(
            rows[0]["draft_generation"]["active_wall_sec"], 3.0
        )
        self.assertEqual(
            rows[0]["suffix_generation"]["active_wall_sec"], 3.0
        )
        self.assertEqual(
            rows[0]["total_generation_active_wall_sec"], 6.0
        )
        self.assertEqual(
            rows[0]["signal_computation"]["active_wall_sec"], 2.0
        )
        summary = summarize_per_question_timings(rows)
        self.assertEqual(
            summary["signals"][0][
                "generation_to_signal_ratio_of_sums"
            ],
            3.0,
        )

    def test_resume_preserves_per_question_task_events(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "result"
            run_dir = output_dir / "timing/runs/fresh"
            worker_path = run_dir / "workers/mixed/shard_0.json"
            records = [
                {
                    "task_type": "draft",
                    "doc_id": "doc_0",
                    "draft_idx": 0,
                },
                {
                    "task_type": "logprob",
                    "doc_id": "doc_0",
                    "draft_idx": 0,
                },
                {
                    "task_type": "suffix",
                    "doc_id": "doc_0",
                    "draft_idx": 0,
                    "strategy": "nll_drop_fb_last",
                    "rollback_step": 1,
                    "suffix_idx": 0,
                },
            ]
            timing = WorkerTiming(
                path=worker_path,
                phase="mixed",
                run_id="fresh",
                shard_id=0,
                gpu_id="0",
                tasks=records,
            )
            for index, record in enumerate(records):
                timing.record_task(
                    record,
                    started_unix_sec=float(index),
                    finished_unix_sec=float(index + 1),
                    timing_source="test",
                )
            timing.finish()
            fresh_summary = publish_per_question_timings(
                output_dir,
                checkpoint_records=records,
                signals=["nll_drop_fb_last"],
                current_run_dir=run_dir,
            )
            per_question_path = (
                output_dir / "timing/per_question.jsonl"
            )
            fresh_content = per_question_path.read_bytes()

            resumed_summary = publish_per_question_timings(
                output_dir,
                checkpoint_records=records,
                signals=["nll_drop_fb_last"],
                current_run_dir=(
                    output_dir / "timing/runs/resumed"
                ),
            )

            self.assertEqual(fresh_summary, resumed_summary)
            self.assertEqual(
                fresh_content, per_question_path.read_bytes()
            )


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Smoke-test exact and partial checkpoint wall-clock provenance."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from rebuttal.src.wallclock import RunTimingRecorder


def tasks(count: int) -> list[dict]:
    return [
        {
            "task_type": "suffix",
            "doc_id": f"smoke_{index // 2}",
            "draft_idx": index // 2,
            "strategy": "prm_drop_fb_last",
            "rollback_step": 1,
            "suffix_idx": index % 2,
        }
        for index in range(count)
    ]


def main() -> None:
    with tempfile.TemporaryDirectory(
        prefix="rebuttal_wallclock_smoke_"
    ) as temporary:
        root = Path(temporary)
        complete_tasks = tasks(4)

        source = RunTimingRecorder(
            root / "source",
            mode="fresh",
            context={"smoke": True},
            run_id="source",
        )
        source.record_phase(
            "suffix",
            required=complete_tasks,
            executed=complete_tasks,
            current_wall_sec=4.0,
        )
        source.finalize()
        source_manifest = root / "source/timing/timing_manifest.json"

        exact = RunTimingRecorder(
            root / "exact",
            mode="auto",
            context={"smoke": True},
            source_manifests=[source_manifest],
            run_id="exact",
        )
        exact_event = exact.record_phase(
            "suffix",
            required=complete_tasks,
            executed=[],
            current_wall_sec=0.01,
        )
        exact_manifest = exact.finalize()

        partial = RunTimingRecorder(
            root / "partial",
            mode="auto",
            context={"smoke": True},
            source_manifests=[source_manifest],
            run_id="partial",
        )
        partial_event = partial.record_phase(
            "suffix",
            required=complete_tasks[:2],
            executed=[],
            current_wall_sec=0.01,
        )
        partial_manifest = partial.finalize()

        if exact_event["reconstructed_from_scratch_wall_sec"] != 4.0:
            raise RuntimeError("Exact checkpoint timing was not inherited")
        if partial_event["reconstructed_from_scratch_wall_sec"] is not None:
            raise RuntimeError("Partial checkpoint timing was scaled")

        print(
            json.dumps(
                {
                    "status": "passed",
                    "exact": {
                        "status": exact_event["status"],
                        "reconstruction_kind": exact_event[
                            "reconstruction_kind"
                        ],
                        "from_scratch_wall_sec": exact_manifest[
                            "reconstructed_from_scratch_wall_sec"
                        ],
                    },
                    "partial": {
                        "status": partial_event["status"],
                        "reconstruction_kind": partial_event[
                            "reconstruction_kind"
                        ],
                        "from_scratch_wall_sec": partial_manifest[
                            "reconstructed_from_scratch_wall_sec"
                        ],
                    },
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()

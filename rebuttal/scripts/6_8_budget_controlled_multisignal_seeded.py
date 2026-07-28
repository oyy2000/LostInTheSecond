#!/usr/bin/env python3
"""Run experiment 6_8 with corrected deterministic per-request seeds.

The original pipeline is loaded as a module so all scoring, rollback, and
evaluation logic stays shared. Only the vLLM generation worker is replaced.
The wrapper also makes child shard processes execute this corrected entrypoint.
"""

import importlib.util
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_SCRIPT = (
    PROJECT_ROOT / "scripts" / "6_8_budget_controlled_multisignal.py"
)
sys.path.insert(0, str(PROJECT_ROOT))

from rebuttal.src.seeded_generation import run_seeded_shard  # noqa: E402


def _load_pipeline():
    spec = importlib.util.spec_from_file_location(
        "budget_controlled_multisignal_base",
        ORIGINAL_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load pipeline: {ORIGINAL_SCRIPT}")
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
    return pipeline


def _is_worker_invocation(argv):
    return "--_shard-id" in argv


def _has_run_seed(argv):
    return "--seed" in argv or "--_seed" in argv


def _is_scoring_worker(argv):
    """Return whether this child does not perform stochastic generation."""
    return any(
        flag in argv
        for flag in ("--_prm", "--_skywork-prm", "--_logprob")
    )


def main():
    argv = sys.argv[1:]
    if not _has_run_seed(argv) and not _is_scoring_worker(argv):
        raise SystemExit(
            "Corrected seeded generation requires an explicit --seed."
        )

    pipeline = _load_pipeline()
    pipeline.run_shard = (
        lambda args: run_seeded_shard(pipeline, args)
    )

    # The base launch_shards function resolves its own __file__ to choose the
    # child entrypoint. Redirect it to this wrapper so workers retain the fix.
    pipeline.__file__ = str(Path(__file__).resolve())
    if not _is_worker_invocation(argv):
        print(
            "Sampling mode: deterministic per-request seeds "
            "(SHA-256 schema per-request-v1)"
        )
    pipeline.main()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

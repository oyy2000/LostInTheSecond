# Checkpoint-aware wall-clock timing

The authoritative timing artifact for experiment 6_8 is:

```text
<out-dir>/timing/timing_manifest.json
```

Every invocation also retains an immutable copy under:

```text
<out-dir>/timing/runs/<run-id>/
```

`phase_times.json` is retained only for compatibility with older result
readers. It is cumulative and must not be used for new wall-clock claims.

## Timing modes

- `--timing-mode auto`: allow checkpoint reuse and record both current work
  and exact historical timing when available.
- `--timing-mode fresh`: require an empty checkpoint and prohibit
  `--draft-checkpoint`; use for a strict from-scratch benchmark.
- `--timing-mode eval-only`: identify an offline evaluation invocation.

Use `--reuse-timing-manifest PATH` more than once when records come from
multiple checkpoints. The runner snapshots every source manifest so later
updates cannot change provenance.

## Interpretation

Each phase reports:

- `observed_current_wall_sec`: time spent by the current invocation;
- `tasks_required`, `tasks_reused`, and `tasks_executed`;
- `reconstructed_from_scratch_wall_sec`;
- `reconstruction_kind`.

Historical wall-clock is inherited only when a source phase has exactly the
same task-set hash and task count. Partial task sets are never scaled
linearly; their reconstructed time remains `null`.

For multi-GPU phases, the launcher records:

- `launcher_wall_sec`: parent-process critical path;
- `gpu_wait_sec`: time waiting for eligible GPUs;
- `worker_wall_sec_max`: slowest worker;
- `gpu_seconds_sum`: sum of worker wall-clock;
- paths to worker timing sidecars.

Seeded vLLM generation workers additionally record model-load time and
per-batch wall-clock, task identities, input tokens, and generated tokens.

## Smoke test

```bash
conda run -n fact python \
  rebuttal/scripts/6_18_smoke_test_wallclock.py
```

The smoke test confirms that an exact reused task set inherits historical
timing and that a partial task set produces `null`.

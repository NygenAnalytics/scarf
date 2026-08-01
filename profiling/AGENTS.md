# Profiling instructions

These instructions apply to local profiling diagnostics and the Modal end-to-end profiler under
`profiling/`.

## Before cloud work

- Read `profiling/BENCHMARKS.md` for the current reference measurements and their interpretation
  limits.
- Run local profiling tests before using cloud resources:

```bash
uv run pytest -n 0 tests/test_profiling_*.py
```

- Copy `profiling/config.example.toml` to the ignored `profiling/config.toml`. Do not commit
  machine-specific configs, endpoints, bucket names, secrets, or result identifiers.
- Set a fresh, non-empty `runTag` for every end-to-end measurement.
- Ask before starting a paid or long-running Modal job.

## Deployment boundary

Never run `modal deploy`. Deployment is a user action. Ask the user to run:

```bash
uv run --group profiling modal deploy --env scarf_profiling \
  -m profiling.modal_app
```

Do not create Modal environments, secrets, or credentials.

## Standard workflow

After the user confirms that the current app is deployed, prepare deterministic CELLxGENE samples
and confirm the requested result before starting the funnel:

```bash
uv run --group profiling modal run --env scarf_profiling \
  -m profiling.modal_app -- prepare \
  --config profiling/config.toml

uv run --group profiling modal run --env scarf_profiling \
  -m profiling.modal_app -- run-e2e \
  --config profiling/config.toml --size 1000000
```

`prepare` spawns work and returns immediately. Confirm that its requested H5AD exists before
`run-e2e`.

Use `run --stage ...` for a targeted stage, including repair of an incomplete `countsT`; use
`run-local` for the Modal ephemeral-disk comparison and `io-baseline` for read-pattern diagnostics.

## Durable execution

- Long work must use `.spawn(...)`, never `.remote()`. A blocking call can be cancelled when the
  local gRPC session disconnects.
- Wait through short `FunctionCall.get(timeout=...)` polls or durable result JSON instead of one
  long blocking request.
- Treat `FAILURE`, `INIT_FAILURE`, `TERMINATED`, and `TIMEOUT` as terminal.
  Modal can raise an empty `TimeoutError` for a failed input.
- Give coordinators about 1 CPU and 2 to 4 GiB with `retries=0`. Do not assign stage-worker
  resources to coordinators.
- Prefer the broad `eu` region and leave the Modal cloud option unset unless the experiment
  explicitly measures another placement.
- Log start, plan, periodic progress, and completion lines.
- Persist each stage result and `funnel.json` before treating a run as complete.

## Measurement discipline

- Never run two jobs with the same `runTag`.
- Change one measured variable at a time and keep workflow seeds fixed.
- Compare runs only when dataset, code revision, settings, storage conditions, and resource
  envelope are stated.
- Do not present measurements from different machine sizes as one scaling curve.
- Do not generalize one run into a hardware guarantee or a biological correctness claim.

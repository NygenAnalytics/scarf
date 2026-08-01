"""Modal entrypoints for one-shot Scarf profiling.

Deploy once (you run this), then trigger jobs that keep running if your laptop
disconnects:

  uv run --group profiling modal deploy --env scarf_profiling -m profiling.modal_app

  uv run --group profiling modal run --env scarf_profiling \\
    -m profiling.modal_app -- prepare --config profiling/config.toml
  uv run --group profiling modal run --env scarf_profiling \\
    -m profiling.modal_app -- run-e2e --config profiling/config.toml --size 1000000

prepare / run / run-all / run-local / run-e2e spawn and return immediately.
run-all fans out one size pipeline per container (stages stay sequential on R2).
run-local runs the full funnel in one container on ephemeral-disk Zarr (fast_local).
run-e2e runs the current core funnel in one container while keeping Zarr on R2.
Watch progress with:
  uv run --group profiling modal app logs scarf-profiling --env scarf_profiling
"""

import argparse
import os
import time
from pathlib import Path
from typing import Any

import modal

from profiling.config import (
    CORE_STAGE_ORDER,
    MAX_TIMEOUT_SECONDS,
    ProfilingConfig,
    StageName,
    StageResources,
    load_profiling_config,
)
from profiling.datasets import (
    SOURCE_SPEC,
    download_source,
    prepare_fixture_datasets,
    prepare_local_datasets,
    sha256_file,
)
from profiling.modal_image import COMMON_FUNCTION_OPTIONS, app
from profiling.modal_resources import (
    BASE_EPHEMERAL_DISK_MB,
    modal_function_options,
    orchestrator_function_options,
    validate_modal_environment,
)
from profiling.io_baseline import run_io_baseline_body
from profiling.r2 import (
    download_file,
    object_exists,
    object_size,
    put_json_if_absent,
    upload_file,
)
from profiling.results import (
    result_exists,
    write_funnel_result,
    write_result,
)
from profiling.spawn_wait import (
    DEFAULT_GRACE_SECONDS,
    DEFAULT_STAGE_SPAWN_ATTEMPTS,
    await_function_call,
    await_many_function_calls,
    await_stage_result,
)
from profiling.metrics import ResourceSampler
from profiling.stages import (
    run_stage,
    summarize_resource_measurement,
)

_WORK = Path("/tmp/scarf-profiling")


def _e2e_conflicting_uris(
    config: ProfilingConfig,
    nRows: int,
) -> list[str]:
    store_uri = config.storeUri(nRows).rstrip("/")
    candidates = [
        f"{store_uri}/zarr.json",
        f"{store_uri}/.zgroup",
        config.e2eClaimUri(),
        config.funnelResultUri(nRows),
        *(config.resultUri(nRows, stage) for stage in CORE_STAGE_ORDER),
    ]
    return [uri for uri in candidates if object_exists(uri)]


def _e2e_function_options(config: ProfilingConfig) -> dict[str, Any]:
    envelope = _e2e_resource_envelope(config)
    resources = _e2e_resources(config)
    peak = max(
        resources,
        key=lambda item: (
            item.modalMemoryLimitMb,
            item.modalCpuLimit,
            item.timeoutSeconds,
        ),
    )
    options = modal_function_options(
        config,
        peak,
        maxContainers=1,
        retries=0,
    )
    options["memory"] = (
        envelope["modalMemoryRequestMb"],
        envelope["modalMemoryLimitMb"],
    )
    options["cpu"] = (
        envelope["modalCpuRequest"],
        envelope["modalCpuLimit"],
    )
    options["timeout"] = MAX_TIMEOUT_SECONDS
    return options


def _e2e_resources(config: ProfilingConfig) -> list[StageResources]:
    missing_resources = [
        stage for stage in CORE_STAGE_ORDER if stage not in config.stageResources
    ]
    if missing_resources:
        raise ValueError(
            "run-e2e is missing stageResources for: " + ", ".join(missing_resources)
        )
    return [config.resourcesFor(stage) for stage in CORE_STAGE_ORDER]


def _e2e_resource_envelope(config: ProfilingConfig) -> dict[str, int | float]:
    resources = _e2e_resources(config)
    requested_ephemeral_disk = max(item.ephemeralDiskMb for item in resources)
    if requested_ephemeral_disk > BASE_EPHEMERAL_DISK_MB:
        raise ValueError(
            "run-e2e cannot apply ephemeralDiskMb above "
            f"{BASE_EPHEMERAL_DISK_MB}; Modal does not allow a dynamic "
            "ephemeral_disk override"
        )
    return {
        "modalMemoryRequestMb": max(item.modalMemoryRequestMb for item in resources),
        "modalMemoryLimitMb": max(item.modalMemoryLimitMb for item in resources),
        "modalCpuRequest": max(item.modalCpuRequest for item in resources),
        "modalCpuLimit": max(item.modalCpuLimit for item in resources),
        "ephemeralDiskMb": BASE_EPHEMERAL_DISK_MB,
        "timeoutSeconds": MAX_TIMEOUT_SECONDS,
    }


@app.function(
    **COMMON_FUNCTION_OPTIONS,
    timeout=86_400,
    memory=196_608,
    cpu=8.0,
    ephemeral_disk=BASE_EPHEMERAL_DISK_MB,
)
def prepare_datasets(configDict: dict[str, Any]) -> dict[str, Any]:
    config = ProfilingConfig.model_validate(configDict)
    os.environ.setdefault("R2_ENDPOINT", config.r2EndpointUrl)
    work = _WORK / "prepare"
    work.mkdir(parents=True, exist_ok=True)
    source_path = work / "source.h5ad"
    source_uri = config.sourceUri()
    source_origin = "local-cache"
    if not source_path.is_file():
        if object_exists(source_uri):
            download_file(source_uri, source_path)
            source_origin = "r2-cache"
        else:
            download_source(
                source_path,
                url=SOURCE_SPEC.url,
                expectedBytes=SOURCE_SPEC.sourceBytes,
            )
            upload_file(source_path, source_uri)
            source_origin = "cellxgene+r2-upload"

    uploaded: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    # Fixture uploads are tiny; real nested samples are much larger.
    minimum_real_bytes = 5_000_000

    pending_sizes: list[int] = []
    for n_rows in config.targetSizes:
        uri = config.datasetUri(n_rows)
        existing = object_size(uri)
        if existing is not None and existing >= minimum_real_bytes:
            skipped.append(
                {
                    "nRows": n_rows,
                    "uri": uri,
                    "fileBytes": existing,
                    "status": "skipped-existing",
                }
            )
            continue
        pending_sizes.append(n_rows)

    def _upload_artifact(artifact: Any) -> None:
        uri = config.datasetUri(artifact.targetRows)
        upload_file(artifact.localPath, uri)
        uploaded.append(
            {
                "nRows": artifact.targetRows,
                "uri": uri,
                "fileBytes": artifact.fileBytes,
                "nnz": artifact.nnz,
                "status": "uploaded",
            }
        )

    source_sha256 = None
    if pending_sizes:
        prepared = prepare_local_datasets(
            source_path,
            work / "subsets",
            targetRows=tuple(pending_sizes),
            seed=config.samplingSeed,
            spec=SOURCE_SPEC,
            onArtifact=_upload_artifact,
        )
        source_sha256 = prepared.sourceSha256
    elif source_path.is_file():
        source_sha256 = sha256_file(source_path)

    return {
        "uploaded": uploaded,
        "skipped": skipped,
        "sourceSha256": source_sha256,
        "sourceOrigin": source_origin,
        "sourceUri": source_uri,
    }


@app.function(
    **COMMON_FUNCTION_OPTIONS,
    timeout=3_600,
    memory=8_192,
    cpu=2.0,
    ephemeral_disk=BASE_EPHEMERAL_DISK_MB,
)
def prepare_fixture_datasets_job(
    configDict: dict[str, Any],
    sizes: list[int] | None = None,
    nColumns: int = 500,
) -> dict[str, Any]:
    """Upload tiny synthetic H5ADs so stage jobs can be tested without Cellxgene."""
    config = ProfilingConfig.model_validate(configDict)
    os.environ.setdefault("R2_ENDPOINT", config.r2EndpointUrl)
    selected = tuple(sizes) if sizes else (10_000,)
    for size in selected:
        if size not in config.targetSizes:
            raise ValueError(
                f"fixture size {size} is not in config.targetSizes; "
                "add it to config or choose an existing size"
            )
    work = _WORK / "fixture"
    if work.exists():
        for path in sorted(work.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    work.mkdir(parents=True, exist_ok=True)
    uploaded: list[dict[str, Any]] = []

    def _upload_artifact(artifact: Any) -> None:
        uri = config.datasetUri(artifact.targetRows)
        upload_file(artifact.localPath, uri)
        uploaded.append(
            {
                "nRows": artifact.targetRows,
                "uri": uri,
                "fileBytes": artifact.fileBytes,
                "nnz": artifact.nnz,
                "nColumns": artifact.nColumns,
            }
        )

    prepare_fixture_datasets(
        work,
        targetRows=selected,
        nColumns=nColumns,
        seed=config.samplingSeed,
        onArtifact=_upload_artifact,
    )
    return {"uploaded": uploaded, "kind": "fixture"}


@app.function(
    **COMMON_FUNCTION_OPTIONS,
    timeout=86_400,
    memory=(65_536, 65_536),
    cpu=(8.0, 8.0),
    ephemeral_disk=BASE_EPHEMERAL_DISK_MB,
)
def io_baseline_job(
    configDict: dict[str, Any],
    nRows: int = 1_000_000,
    resultLabel: str | None = None,
    columnOnly: bool = False,
) -> dict[str, Any]:
    """No-compute R2 stream of HVG, marker, and graph read patterns."""
    config = ProfilingConfig.model_validate(configDict)
    os.environ.setdefault("R2_ENDPOINT", config.r2EndpointUrl)
    return run_io_baseline_body(
        config,
        nRows=nRows,
        resultLabel=resultLabel,
        columnOnly=columnOnly,
    )


@app.function(
    **COMMON_FUNCTION_OPTIONS,
    timeout=86_400,
    memory=65_536,
    cpu=8.0,
    ephemeral_disk=BASE_EPHEMERAL_DISK_MB,
)
def run_stage_job(
    configDict: dict[str, Any],
    nRows: int,
    stage: StageName,
    force: bool = False,
) -> dict[str, Any]:
    config = ProfilingConfig.model_validate(configDict)
    resources = config.resourcesFor(stage)
    os.environ.setdefault("R2_ENDPOINT", config.r2EndpointUrl)
    if result_exists(config, nRows, stage) and not force:
        return {
            "nRows": nRows,
            "stage": stage,
            "status": "skipped",
            "resultUri": config.resultUri(nRows, stage),
        }

    work = _WORK / f"{nRows}-{stage}"
    if work.exists():
        for path in sorted(work.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    work.mkdir(parents=True, exist_ok=True)

    local_h5ad: Path | None = None
    if stage == "createStore":
        local_h5ad = work / f"{nRows}.h5ad"
        download_file(config.datasetUri(nRows), local_h5ad)

    result = run_stage(
        stage,
        nRows=nRows,
        storeUri=config.storeUri(nRows),
        workflow=config.workflow,
        resources=resources,
        localH5adPath=local_h5ad,
        storageLayout=config.storageLayout,
        workDir=work,
        invalidateCache=force,
    )
    write_result(config, result)
    return result.to_json()


@app.function(
    **COMMON_FUNCTION_OPTIONS,
    timeout=86_400,
    memory=32_768,
    cpu=8.0,
    ephemeral_disk=BASE_EPHEMERAL_DISK_MB,
)
def run_local_funnel_job(
    configDict: dict[str, Any],
    nRows: int,
    stages: list[StageName] | None = None,
) -> dict[str, Any]:
    """Full funnel in one container: H5AD + Zarr on ephemeral disk (fast_local).

    Downloads the prepared H5AD from R2 once, writes the store under /tmp, runs
    stages in-process, and still persists each stage result JSON to R2.
    """
    config = ProfilingConfig.model_validate(configDict)
    os.environ.setdefault("R2_ENDPOINT", config.r2EndpointUrl)
    os.environ["SCARF_ZARR_PROFILE"] = "fast_local"
    if nRows not in config.targetSizes:
        raise ValueError(f"size {nRows} is not in config.targetSizes")
    selected_stages = tuple(stages) if stages else config.effectiveStages

    work = _WORK / f"local-{config.runTag or 'untagged'}-{nRows}"
    if work.exists():
        for path in sorted(work.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    work.mkdir(parents=True, exist_ok=True)

    local_h5ad = work / f"{nRows}.h5ad"
    local_store = work / f"{nRows}.zarr"
    store_uri = str(local_store)

    if "createStore" in selected_stages and not result_exists(
        config, nRows, "createStore"
    ):
        print(f"downloading dataset to {local_h5ad}", flush=True)
        download_file(config.datasetUri(nRows), local_h5ad)

    outcomes: list[dict[str, Any]] = []
    for stage in selected_stages:
        if result_exists(config, nRows, stage):
            outcomes.append(
                {
                    "nRows": nRows,
                    "stage": stage,
                    "status": "skipped",
                    "resultUri": config.resultUri(nRows, stage),
                    "storeBackend": "local",
                }
            )
            continue
        if stage == "createStore" and not local_h5ad.is_file():
            raise FileNotFoundError(
                f"createStore needs {local_h5ad}; download failed or was skipped"
            )
        if stage != "createStore" and not local_store.exists():
            raise FileNotFoundError(
                f"stage {stage} needs local store at {local_store}; "
                "include createStore or pre-seed the ephemeral workdir"
            )
        resources = config.resourcesFor(stage)
        print(f"local funnel stage start: {stage}", flush=True)
        result = run_stage(
            stage,
            nRows=nRows,
            storeUri=store_uri,
            workflow=config.workflow,
            resources=resources,
            localH5adPath=local_h5ad if stage == "createStore" else None,
            storageLayout=config.storageLayout,
            workDir=work / stage,
        )
        write_result(config, result)
        payload = result.to_json()
        payload["storeBackend"] = "local"
        outcomes.append(payload)
        print(
            f"local funnel stage done: {stage} status={result.status} "
            f"seconds={result.seconds}",
            flush=True,
        )
        if result.status == "error":
            return {
                "nRows": nRows,
                "stopped": True,
                "storeBackend": "local",
                "storeUri": store_uri,
                "failed": payload,
                "outcomes": outcomes,
            }

    return {
        "nRows": nRows,
        "stopped": False,
        "storeBackend": "local",
        "storeUri": store_uri,
        "outcomes": outcomes,
    }


def run_e2e_funnel_body(
    configDict: dict[str, Any],
    nRows: int,
) -> dict[str, Any]:
    """Run the graph-construction core once in one container against a fresh R2 store."""
    config = ProfilingConfig.model_validate(configDict)
    os.environ.setdefault("R2_ENDPOINT", config.r2EndpointUrl)
    if nRows not in config.targetSizes:
        raise ValueError(f"size {nRows} is not in config.targetSizes")
    if not config.runTag.strip():
        raise ValueError("run-e2e requires a non-empty runTag")
    resource_envelope = _e2e_resource_envelope(config)
    conflicts = _e2e_conflicting_uris(config, nRows)
    if conflicts:
        raise FileExistsError(
            "run-e2e requires a fresh runTag; existing R2 objects: "
            + ", ".join(conflicts)
        )
    claimed = put_json_if_absent(
        config.e2eClaimUri(),
        {
            "runTag": config.runTag,
            "nRows": nRows,
            "status": "claimed",
            "storeUri": config.storeUri(nRows),
        },
    )
    if not claimed:
        raise FileExistsError(
            f"run-e2e runTag was claimed concurrently: {config.runTag}"
        )

    work = _WORK / f"e2e-{config.runTag}-{nRows}"
    work.mkdir(parents=True, exist_ok=False)
    local_h5ad = work / f"{nRows}.h5ad"
    store_uri = config.storeUri(nRows)

    sampler = ResourceSampler()
    sampler.start()
    started = time.perf_counter()
    download_seconds: float | None = None
    outcomes: list[dict[str, Any]] = []
    completed_stages: list[StageName] = []
    status = "ok"
    error: str | None = None
    failed_stage: StageName | None = None
    try:
        download_started = time.perf_counter()
        print(f"e2e dataset download start: {config.datasetUri(nRows)}", flush=True)
        download_file(config.datasetUri(nRows), local_h5ad)
        download_seconds = time.perf_counter() - download_started
        print(
            f"e2e dataset download done: seconds={download_seconds:.1f}",
            flush=True,
        )

        for stage in CORE_STAGE_ORDER:
            failed_stage = stage
            resources = config.resourcesFor(stage)
            stage_work = work / stage
            stage_work.mkdir(parents=True, exist_ok=True)
            print(f"e2e stage start: {stage}", flush=True)
            result = run_stage(
                stage,
                nRows=nRows,
                storeUri=store_uri,
                workflow=config.workflow,
                resources=resources,
                localH5adPath=local_h5ad if stage == "createStore" else None,
                storageLayout=config.storageLayout,
                workDir=stage_work,
                containerMemoryMb=int(resource_envelope["modalMemoryLimitMb"]),
                containerCpuRequest=float(resource_envelope["modalCpuRequest"]),
                containerCpuLimit=float(resource_envelope["modalCpuLimit"]),
                resetCgroupPeak=False,
            )
            result_uri = write_result(config, result)
            payload = result.to_json()
            payload["resultUri"] = result_uri
            payload["storeBackend"] = "r2"
            outcomes.append(payload)
            print(
                f"e2e stage done: {stage} status={result.status} "
                f"seconds={result.seconds}",
                flush=True,
            )
            if result.status != "ok":
                status = "error"
                error = result.error or f"{stage} failed"
                break
            completed_stages.append(stage)
        else:
            failed_stage = None
    except Exception as exc:  # noqa: BLE001 - persist a durable failure summary
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
    finally:
        measurement = sampler.stop()

    summary: dict[str, Any] = {
        "runTag": config.runTag,
        "nRows": nRows,
        "status": status,
        "stopped": status != "ok",
        "error": error,
        "failedStage": failed_stage,
        "storeBackend": "r2",
        "storeUri": store_uri,
        "datasetUri": config.datasetUri(nRows),
        "datasetDownloadSeconds": download_seconds,
        "wholeFunctionSeconds": time.perf_counter() - started,
        "modalResources": resource_envelope,
        "stageOrder": list(CORE_STAGE_ORDER),
        "completedStages": completed_stages,
        "outcomes": outcomes,
        "claimUri": config.e2eClaimUri(),
        "funnelResultUri": config.funnelResultUri(nRows),
        **summarize_resource_measurement(measurement),
    }
    write_funnel_result(config, nRows, summary)
    return summary


@app.function(
    **COMMON_FUNCTION_OPTIONS,
    timeout=MAX_TIMEOUT_SECONDS,
    memory=32_768,
    cpu=8.0,
    ephemeral_disk=BASE_EPHEMERAL_DISK_MB,
)
def run_e2e_funnel_job(
    configDict: dict[str, Any],
    nRows: int,
) -> dict[str, Any]:
    return run_e2e_funnel_body(configDict, nRows)


@app.function(
    **COMMON_FUNCTION_OPTIONS,
    timeout=86_400,
    memory=2048,
    cpu=1.0,
    ephemeral_disk=BASE_EPHEMERAL_DISK_MB,
)
def run_size_jobs(
    configDict: dict[str, Any],
    nRows: int,
    stages: list[StageName] | None = None,
) -> dict[str, Any]:
    """Run stages sequentially for one size (skip if result JSON exists)."""
    config = ProfilingConfig.model_validate(configDict)
    os.environ.setdefault("R2_ENDPOINT", config.r2EndpointUrl)
    if nRows not in config.targetSizes:
        raise ValueError(f"size {nRows} is not in config.targetSizes")
    selected_stages = tuple(stages) if stages else config.effectiveStages
    parallel_sizes = max(1, len(config.targetSizes))
    outcomes: list[dict[str, Any]] = []

    for stage in selected_stages:
        if result_exists(config, nRows, stage):
            outcomes.append(
                {
                    "nRows": nRows,
                    "stage": stage,
                    "status": "skipped",
                    "resultUri": config.resultUri(nRows, stage),
                }
            )
            continue
        resources = config.resourcesFor(stage)
        options = (
            modal_function_options(
                config,
                resources,
                maxContainers=parallel_sizes,
                retries=0,
            )
            if stage == "writeCountsT"
            else modal_function_options(
                config,
                resources,
                maxContainers=parallel_sizes,
            )
        )
        deadline_seconds = float(resources.timeoutSeconds) + DEFAULT_GRACE_SECONDS
        result: dict[str, Any] | None = None
        last_error: BaseException | None = None
        spawn_attempts = 1 if stage == "writeCountsT" else DEFAULT_STAGE_SPAWN_ATTEMPTS
        for attempt in range(1, spawn_attempts + 1):
            if result_exists(config, nRows, stage):
                result = {
                    "nRows": nRows,
                    "stage": stage,
                    "status": "ok",
                    "resultUri": config.resultUri(nRows, stage),
                    "recoveredFromR2": True,
                    "spawnAttempt": attempt,
                }
                break
            call = run_stage_job.with_options(**options).spawn(
                configDict,
                nRows,
                stage,
            )
            try:
                result = await_stage_result(
                    config,
                    nRows,
                    stage,
                    call,
                    deadlineSeconds=deadline_seconds,
                )
                break
            except Exception as exc:  # noqa: BLE001 - Modal surfaces many failure types
                last_error = exc
                if result_exists(config, nRows, stage):
                    result = {
                        "nRows": nRows,
                        "stage": stage,
                        "status": "ok",
                        "resultUri": config.resultUri(nRows, stage),
                        "recoveredFromR2": True,
                        "spawnAttempt": attempt,
                        "callError": str(exc),
                    }
                    break
                if attempt >= spawn_attempts:
                    raise
                print(
                    f"stage {stage} spawn attempt {attempt}/"
                    f"{spawn_attempts} failed ({exc}); retrying",
                    flush=True,
                )
        if result is None:
            raise RuntimeError(
                f"stage {stage} produced no result"
                + (f" after error: {last_error}" if last_error else "")
            )
        outcomes.append(result)
        if result.get("status") == "error":
            return {
                "nRows": nRows,
                "stopped": True,
                "failed": result,
                "outcomes": outcomes,
            }

    return {"nRows": nRows, "stopped": False, "outcomes": outcomes}


@app.function(
    **COMMON_FUNCTION_OPTIONS,
    timeout=86_400,
    memory=2048,
    cpu=1.0,
    ephemeral_disk=BASE_EPHEMERAL_DISK_MB,
)
def run_all_jobs(
    configDict: dict[str, Any],
    sizes: list[int] | None = None,
    stages: list[StageName] | None = None,
) -> dict[str, Any]:
    """Run sizes in parallel; stages within each size stay sequential."""
    config = ProfilingConfig.model_validate(configDict)
    os.environ.setdefault("R2_ENDPOINT", config.r2EndpointUrl)
    selected_sizes = tuple(sizes) if sizes else config.targetSizes
    selected_stages = tuple(stages) if stages else config.effectiveStages
    for n_rows in selected_sizes:
        if n_rows not in config.targetSizes:
            raise ValueError(f"size {n_rows} is not in config.targetSizes")

    parallel_sizes = max(1, len(selected_sizes))
    orchestrator_options = orchestrator_function_options(
        config,
        maxContainers=parallel_sizes,
    )

    stage_list = list(selected_stages)
    handles = [
        run_size_jobs.with_options(**orchestrator_options).spawn(
            configDict,
            n_rows,
            stage_list,
        )
        for n_rows in selected_sizes
    ]
    size_results = await_many_function_calls(
        handles,
        deadlineSeconds=86_400.0,
    )
    failed = [item for item in size_results if item.get("stopped")]
    return {
        "stopped": bool(failed),
        "failed": failed[0] if failed else None,
        "sizes": size_results,
    }


@app.function(
    **COMMON_FUNCTION_OPTIONS,
    timeout=300,
    memory=1024,
    cpu=1.0,
    ephemeral_disk=BASE_EPHEMERAL_DISK_MB,
)
def smoke_check(configDict: dict[str, Any]) -> dict[str, Any]:
    config = ProfilingConfig.model_validate(configDict)
    os.environ.setdefault("R2_ENDPOINT", config.r2EndpointUrl)
    probe = f"{config.resultsUri.rstrip('/')}/smoke/ok.json"
    from profiling.r2 import put_json

    put_json(probe, {"ok": True})
    return {"probeUri": probe, "exists": object_exists(probe)}


def _load_config(path: str) -> ProfilingConfig:
    config = load_profiling_config(path)
    validate_modal_environment(config)
    return config


def _deployed_function(config: ProfilingConfig, name: str) -> modal.Function:
    try:
        return modal.Function.from_name(
            config.modalAppName,
            name,
            environment_name=config.modalEnvironmentName,
        )
    except Exception as exc:
        raise SystemExit(
            f"Could not find deployed function {config.modalAppName}/{name}. "
            "Deploy first with:\n"
            "  uv run --group profiling modal deploy "
            f"--env {config.modalEnvironmentName} -m profiling.modal_app\n"
            f"Original error: {exc}"
        ) from exc


def _print_spawned(label: str, call: Any) -> None:
    call_id = (
        getattr(call, "object_id", None) or getattr(call, "call_id", None) or str(call)
    )
    print(f"spawned {label}: {call_id}")
    print("disconnect is safe; watch with:")
    print(
        "  uv run --group profiling modal app logs "
        "scarf-profiling --env scarf_profiling"
    )


@app.local_entrypoint()
def main(*arg_list: str) -> None:
    parser = argparse.ArgumentParser(prog="profiling.modal_app")
    sub = parser.add_subparsers(dest="command", required=True)

    smoke_parser = sub.add_parser("smoke")
    smoke_parser.add_argument("--config", required=True)

    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--config", required=True)

    fixture_parser = sub.add_parser("prepare-fixture")
    fixture_parser.add_argument("--config", required=True)
    fixture_parser.add_argument("--sizes", nargs="*", type=int, default=[10_000])
    fixture_parser.add_argument("--n-columns", type=int, default=500)

    run_parser = sub.add_parser("run")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--size", type=int, required=True)
    run_parser.add_argument("--stage", choices=CORE_STAGE_ORDER, required=True)
    run_parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Recompute an existing targeted stage, invalidate reusable artifacts, "
            "and overwrite its stage result JSON."
        ),
    )
    run_parser.add_argument(
        "--ephemeral",
        action="store_true",
        help="Spawn from this modal run app (no deploy). Prefer --detach.",
    )

    all_parser = sub.add_parser("run-all")
    all_parser.add_argument("--config", required=True)
    all_parser.add_argument("--sizes", nargs="*", type=int, default=None)
    all_parser.add_argument(
        "--stages", nargs="*", choices=CORE_STAGE_ORDER, default=None
    )
    all_parser.add_argument(
        "--ephemeral",
        action="store_true",
        help="Spawn from this modal run app (no deploy). Prefer --detach.",
    )

    local_parser = sub.add_parser(
        "run-local",
        help=(
            "One-container funnel on ephemeral-disk Zarr (fast_local); "
            "H5AD downloaded once from R2; stage results still written to R2"
        ),
    )
    local_parser.add_argument("--config", required=True)
    local_parser.add_argument("--size", type=int, required=True)
    local_parser.add_argument(
        "--stages", nargs="*", choices=CORE_STAGE_ORDER, default=None
    )

    e2e_parser = sub.add_parser(
        "run-e2e",
        help="One-container graph-construction funnel with a fresh R2 Zarr store",
    )
    e2e_parser.add_argument("--config", required=True)
    e2e_parser.add_argument("--size", type=int, required=True)

    io_parser = sub.add_parser("io-baseline")
    io_parser.add_argument("--config", required=True)
    io_parser.add_argument("--size", type=int, default=1_000_000)
    io_parser.add_argument("--result-label")
    io_parser.add_argument("--column-only", action="store_true")
    io_parser.add_argument("--wait", action="store_true")
    io_parser.add_argument(
        "--ephemeral",
        action="store_true",
        help="Spawn from this modal run app without a deploy.",
    )

    args = parser.parse_args(list(arg_list))
    config = _load_config(args.config)
    payload = config.model_dump(mode="python")

    if args.command == "smoke":
        smoke_options = orchestrator_function_options(config)
        call = (
            _deployed_function(config, "smoke_check")
            .with_options(**smoke_options)
            .spawn(payload)
        )
        print(await_function_call(call, deadlineSeconds=300.0))
        return

    if args.command == "prepare":
        prepare_options = modal_function_options(
            config,
            config.prepareResources,
        )
        call = (
            _deployed_function(config, "prepare_datasets")
            .with_options(**prepare_options)
            .spawn(payload)
        )
        _print_spawned("prepare_datasets", call)
        return

    if args.command == "prepare-fixture":
        sizes = list(args.sizes) if args.sizes else [10_000]
        for size in sizes:
            if size not in config.targetSizes:
                raise SystemExit(f"size {size} is not in config.targetSizes")
        fixture_options = modal_function_options(
            config,
            config.resourcesFor("reopenStore"),
        )
        call = (
            _deployed_function(config, "prepare_fixture_datasets_job")
            .with_options(**fixture_options)
            .spawn(payload, sizes, args.n_columns)
        )
        _print_spawned("prepare_fixture_datasets_job", call)
        return

    if args.command == "run":
        if args.size not in config.targetSizes:
            raise SystemExit(f"size {args.size} is not in config.targetSizes")
        if result_exists(config, args.size, args.stage) and not args.force:
            print(
                {
                    "nRows": args.size,
                    "stage": args.stage,
                    "status": "skipped",
                    "resultUri": config.resultUri(args.size, args.stage),
                }
            )
            return
        resources = config.resourcesFor(args.stage)
        # retries=0 for writeCountsT: a retry that overlaps the original write
        # leaves countsT marked incomplete again.
        options = (
            modal_function_options(config, resources, retries=0)
            if args.stage == "writeCountsT"
            else modal_function_options(config, resources)
        )
        target = (
            run_stage_job
            if args.ephemeral
            else _deployed_function(config, "run_stage_job")
        )
        spawn_args: tuple[Any, ...] = (payload, args.size, args.stage)
        if args.force:
            spawn_args += (True,)
        call = target.with_options(**options).spawn(*spawn_args)
        _print_spawned(f"run_stage_job {args.size}/{args.stage}", call)
        return

    if args.command == "run-all":
        sizes = list(args.sizes) if args.sizes else None
        stages = list(args.stages) if args.stages else None
        if sizes:
            for size in sizes:
                if size not in config.targetSizes:
                    raise SystemExit(f"size {size} is not in config.targetSizes")
        coordinator_options = orchestrator_function_options(config)
        target = (
            run_all_jobs
            if args.ephemeral
            else _deployed_function(config, "run_all_jobs")
        )
        call = target.with_options(**coordinator_options).spawn(payload, sizes, stages)
        _print_spawned("run_all_jobs", call)
        return

    if args.command == "run-e2e":
        if args.size not in config.targetSizes:
            raise SystemExit(f"size {args.size} is not in config.targetSizes")
        if not config.runTag.strip():
            raise SystemExit("run-e2e requires a non-empty runTag")
        options = _e2e_function_options(config)
        call = (
            _deployed_function(config, "run_e2e_funnel_job")
            .with_options(**options)
            .spawn(payload, args.size)
        )
        _print_spawned(f"run_e2e_funnel_job {args.size}", call)
        print(f"result URI (when done): {config.funnelResultUri(args.size)}")
        return

    if args.command == "run-local":
        if args.size not in config.targetSizes:
            raise SystemExit(f"size {args.size} is not in config.targetSizes")
        stages = list(args.stages) if args.stages else None
        selected = tuple(stages) if stages else config.effectiveStages
        # Size the single container to the hungriest stage in the funnel.
        peak = max(
            (config.resourcesFor(stage) for stage in selected),
            key=lambda item: (
                item.modalMemoryLimitMb,
                item.modalCpuLimit,
                item.timeoutSeconds,
            ),
        )
        options = modal_function_options(config, peak, maxContainers=1)
        call = (
            _deployed_function(config, "run_local_funnel_job")
            .with_options(**options)
            .spawn(payload, args.size, stages)
        )
        _print_spawned(f"run_local_funnel_job {args.size}", call)
        return

    if args.command == "io-baseline":
        resources = config.resourcesFor("markHvgs")
        options = modal_function_options(config, resources, maxContainers=1)
        target = (
            io_baseline_job
            if args.ephemeral
            else _deployed_function(config, "io_baseline_job")
        )
        call = target.with_options(**options).spawn(
            payload,
            args.size,
            args.result_label,
            args.column_only,
        )
        _print_spawned(f"io_baseline_job size={args.size}", call)
        if args.wait:
            print(
                await_function_call(
                    call,
                    deadlineSeconds=float(resources.timeoutSeconds),
                )
            )
        print(
            "result URI (when done): "
            f"{config.resultsUri.rstrip('/')}/io-baseline/{config.runTag}"
            f"{'-' + args.result_label if args.result_label else ''}.json"
        )
        return

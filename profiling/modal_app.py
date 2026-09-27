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
run-e2e and run-local run one funnel in one container, with the Zarr store on R2
or on the container's ephemeral disk (fast_local). Like DataStore.pipeline, the
funnel runs UMAP beside Leiden.
Watch progress with:
  uv run --group profiling modal app logs scarf-profiling --env scarf_profiling
"""

import argparse
import dataclasses
import functools
import os
import shutil
import time
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import modal
from scarf.storage import ArtifactRef
from scarf.utils.background import BackgroundTask

from profiling.config import (
    ALL_STAGE_CHOICES,
    CONSUME_STAGES,
    CORE_STAGE_ORDER,
    MAX_TIMEOUT_SECONDS,
    ProfilingConfig,
    StageName,
    StageResources,
    WorkflowParameters,
    bind_cluster_source,
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
from profiling.provenance import attach_client_provenance, provenance_from_config
from profiling.r2 import (
    download_file,
    object_exists,
    object_size,
    put_json_if_absent,
    upload_file,
)
from profiling.results import (
    claim_submission,
    load_result,
    result_exists,
    write_funnel_result,
    write_result,
)
from profiling.spawn_wait import (
    DEFAULT_GRACE_SECONDS,
    await_function_call,
    await_many_function_calls,
    await_stage_result,
)
from profiling.metrics import ResourceSampler
from profiling.stages import (
    StageRunResult,
    discover_consume_inputs,
    profile_stage_inputs,
    run_stage,
    summarize_resource_measurement,
)

_WORK = Path("/tmp/scarf-profiling")


def _fresh_work_dir(path: Path) -> Path:
    """Return an empty local work directory, deleting earlier contents."""
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path


def _load_stage_artifact_ref(
    config: ProfilingConfig,
    nRows: int,
    stage: StageName,
    kind: str,
) -> ArtifactRef:
    payload = load_result(config, nRows, stage)
    if payload is None:
        raise ValueError(f"{stage} stage result is unavailable")
    if payload.get("status") != "ok":
        raise ValueError(f"{stage} stage result is not complete")
    details = payload.get("details")
    if not isinstance(details, dict):
        raise ValueError(f"{stage} stage result has no details")
    artifact = details.get("artifact")
    if not isinstance(artifact, dict):
        raise ValueError(f"{stage} stage result has no artifact reference")
    ref = ArtifactRef.from_dict(artifact)
    expected_scope = "datastore" if kind == "cell_selection" else "assay"
    expected_assay = (
        None if expected_scope == "datastore" else config.workflow.assayName
    )
    if ref.scope != expected_scope or ref.assay != expected_assay or ref.kind != kind:
        raise ValueError(f"{stage} stage result has an invalid {kind} artifact")
    return ref


def _load_stage_input_refs(
    config: ProfilingConfig,
    nRows: int,
    stage: StageName,
    *,
    workflow: WorkflowParameters | None = None,
) -> dict[str, ArtifactRef]:
    resolved_workflow = config.workflow if workflow is None else workflow
    if stage in CONSUME_STAGES:
        return discover_consume_inputs(
            config.storeUri(nRows),
            resolved_workflow,
            stage,
        )
    return {
        name: _load_stage_artifact_ref(
            config,
            nRows,
            source_stage,
            kind,
        )
        for name, (source_stage, kind) in profile_stage_inputs(
            resolved_workflow, stage
        ).items()
    }


def _e2e_conflicting_uris(
    config: ProfilingConfig,
    nRows: int,
    stages: tuple[StageName, ...] = CORE_STAGE_ORDER,
    *,
    storeOnR2: bool = True,
) -> list[str]:
    candidates = [
        config.e2eClaimUri(),
        config.funnelResultUri(nRows),
        *(config.resultUri(nRows, stage) for stage in stages),
    ]
    if storeOnR2:
        candidates.insert(0, f"{config.storeUri(nRows).rstrip('/')}/zarr.json")
    return [uri for uri in candidates if object_exists(uri)]


def _e2e_function_options(
    config: ProfilingConfig,
    stages: tuple[StageName, ...] = CORE_STAGE_ORDER,
) -> dict[str, Any]:
    envelope = _e2e_resource_envelope(config, stages)
    peak = max(
        _e2e_resources(config, stages),
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


def _e2e_resources(
    config: ProfilingConfig,
    stages: tuple[StageName, ...] = CORE_STAGE_ORDER,
) -> list[StageResources]:
    missing_resources = [
        stage for stage in stages if stage not in config.stageResources
    ]
    if missing_resources:
        raise ValueError(
            "The funnel is missing stageResources for: " + ", ".join(missing_resources)
        )
    return [config.resourcesFor(stage) for stage in stages]


def _e2e_resource_envelope(
    config: ProfilingConfig,
    stages: tuple[StageName, ...] = CORE_STAGE_ORDER,
) -> dict[str, int | float]:
    """Size one container to the per-field maximum of the funnel's stages."""
    resources = _e2e_resources(config, stages)
    requested_ephemeral_disk = max(item.ephemeralDiskMb for item in resources)
    if requested_ephemeral_disk > BASE_EPHEMERAL_DISK_MB:
        raise ValueError(
            "The funnel cannot apply ephemeralDiskMb above "
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
    work = _fresh_work_dir(_WORK / "fixture")
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
    memory=65_536,
    cpu=8.0,
    ephemeral_disk=BASE_EPHEMERAL_DISK_MB,
    # Targeted writeCountsT / long stages: avoid worker preemption.
    nonpreemptible=True,
)
def run_stage_job(
    configDict: dict[str, Any],
    nRows: int,
    stage: StageName,
    submissionId: str,
    force: bool = False,
) -> dict[str, Any]:
    config = ProfilingConfig.model_validate(configDict)
    resources = config.resourcesFor(stage)
    os.environ.setdefault("R2_ENDPOINT", config.r2EndpointUrl)
    workflow = bind_cluster_source(config, nRows)
    completed = load_result(config, nRows, stage, submissionId=submissionId)
    if completed is not None:
        return completed
    claim_submission(config, nRows, stage, submissionId)
    if result_exists(config, nRows, stage) and not force:
        raise FileExistsError(
            "This stage has a previous result; use force or a fresh runTag"
        )

    work = _fresh_work_dir(_WORK / f"{submissionId}-{nRows}-{stage}")
    try:
        local_h5ad: Path | None = None
        if stage == "createStore":
            local_h5ad = work / f"{nRows}.h5ad"
            download_file(config.datasetUri(nRows), local_h5ad)

        result = run_stage(
            stage,
            submissionId=submissionId,
            nRows=nRows,
            storeUri=config.storeUri(nRows),
            workflow=workflow,
            resources=resources,
            localH5adPath=local_h5ad,
            countMatrix=config.countMatrix,
            storageIo=config.storageIo,
            workDir=work,
            invalidateCache=force,
            clientProvenance=config.clientProvenance,
            inputRefs=_load_stage_input_refs(
                config,
                nRows,
                stage,
                workflow=workflow,
            ),
        )
    except Exception as exc:
        from profiling.stages import StageRunResult

        result = StageRunResult(
            submissionId=submissionId,
            stage=stage,
            nRows=nRows,
            status="error",
            seconds=None,
            peakRssBytes=None,
            peakCgroupBytes=None,
            modalMemoryMb=resources.modalMemoryLimitMb,
            scarfMemoryBudget=resources.scarfMemoryBudget,
            storeUri=config.storeUri(nRows),
            error=f"{type(exc).__name__}: {exc}",
            workers=resources.workers,
        )
    write_result(config, result, overwrite=force)
    return result.to_json()


# Mirrors DataStore.pipeline: each key runs on a worker thread beside the
# listed later stages, and any other stage waits for it first. A background
# stage must follow the threading rules of scarf's BackgroundTask.
BACKGROUND_OVERLAPS: dict[StageName, frozenset[StageName]] = {
    "runUmap": frozenset({"runLeiden"}),
}


@app.function(
    **COMMON_FUNCTION_OPTIONS,
    timeout=MAX_TIMEOUT_SECONDS,
    memory=32_768,
    cpu=8.0,
    ephemeral_disk=BASE_EPHEMERAL_DISK_MB,
    # One container holds the whole funnel, and run-local keeps the store on
    # its ephemeral disk; avoid preemption (Modal bills about 3x CPU/memory).
    nonpreemptible=True,
)
def run_funnel_job(
    configDict: dict[str, Any],
    nRows: int,
    submissionId: str,
    storeBackend: Literal["r2", "local"],
    stages: list[StageName],
) -> dict[str, Any]:
    """Run one funnel in one container and persist its summary to R2.

    The store lives on R2 (run-e2e) or on the container's ephemeral disk
    (run-local). The create-only runTag claim makes the funnel exclusive, so
    its stages need no claims of their own. Stages in ``BACKGROUND_OVERLAPS``
    overlap later stages; their CPU and memory figures then share a window,
    and each result lists the stages it ran beside in ``concurrentStages``.
    """
    config = ProfilingConfig.model_validate(configDict)
    os.environ.setdefault("R2_ENDPOINT", config.r2EndpointUrl)
    selected = tuple(stages)
    if nRows not in config.targetSizes:
        raise ValueError(f"size {nRows} is not in config.targetSizes")
    if not config.runTag.strip():
        raise ValueError("A funnel requires a non-empty runTag")
    if storeBackend not in ("r2", "local"):
        raise ValueError(f"Unknown store backend: {storeBackend!r}")
    if not selected or selected[0] != "createStore":
        raise ValueError("A funnel must start with createStore")
    local = storeBackend == "local"
    resource_envelope = _e2e_resource_envelope(config, selected)
    conflicts = _e2e_conflicting_uris(config, nRows, selected, storeOnR2=not local)
    if conflicts:
        raise FileExistsError(
            "A funnel requires a fresh runTag; existing R2 objects: "
            + ", ".join(conflicts)
        )
    store_uri = config.storeUri(nRows)
    if not put_json_if_absent(
        config.e2eClaimUri(),
        {
            "runTag": config.runTag,
            "submissionId": submissionId,
            "nRows": nRows,
            "status": "claimed",
            "storeBackend": storeBackend,
        },
    ):
        raise FileExistsError(f"The runTag was claimed concurrently: {config.runTag}")
    if local:
        # Stages wrap the store to count its operations. A wrapped local store
        # is not a LocalStore, so pin the profile the local path resolves to.
        os.environ["SCARF_ZARR_PROFILE"] = "fast_local"

    work = _fresh_work_dir(_WORK / f"{storeBackend}-{config.runTag}-{nRows}")
    local_h5ad = work / f"{nRows}.h5ad"
    if local:
        store_uri = str(work / f"{nRows}.zarr")
    # findMarkers reads imported clusters only when the funnel imports them.
    workflow = (
        bind_cluster_source(config, nRows)
        if "importClusters" in selected
        else config.workflow
    )
    label = "e2e" if not local else "local"

    sampler = ResourceSampler()
    sampler.start()
    started = time.perf_counter()
    download_seconds: float | None = None
    funnel_seconds: float | None = None
    payloads: dict[StageName, dict[str, Any]] = {}
    windows: dict[StageName, tuple[float, float | None]] = {}
    pending: dict[StageName, BackgroundTask[StageRunResult]] = {}
    status = "ok"
    error: str | None = None
    failed_stage: StageName | None = None
    session: dict[str, Any] = {}

    def execute(stage: StageName) -> StageRunResult:
        windows[stage] = (time.perf_counter(), None)
        try:
            stage_work = work / stage
            stage_work.mkdir(parents=True, exist_ok=True)
            return run_stage(
                stage,
                submissionId=submissionId,
                nRows=nRows,
                storeUri=store_uri,
                workflow=workflow,
                resources=config.resourcesFor(stage),
                localH5adPath=local_h5ad if stage == "createStore" else None,
                countMatrix=config.countMatrix,
                storageIo=config.storageIo,
                workDir=stage_work,
                containerMemoryMb=int(resource_envelope["modalMemoryLimitMb"]),
                containerCpuRequest=float(resource_envelope["modalCpuRequest"]),
                containerCpuLimit=float(resource_envelope["modalCpuLimit"]),
                resetCgroupPeak=False,
                # The session probe is reset per stage; a background stage
                # would read the counts of the stages beside it.
                recordStoreOperations=stage not in BACKGROUND_OVERLAPS,
                clientProvenance=config.clientProvenance,
                session=session,
            )
        finally:
            windows[stage] = (windows[stage][0], time.perf_counter())

    def record(stage: StageName, result: StageRunResult) -> bool:
        begin, end = windows[stage]
        concurrent = [
            other
            for other, (other_begin, other_end) in windows.items()
            if other != stage
            and other_begin < (end or time.perf_counter())
            and (other_end is None or other_end > begin)
        ]
        result = dataclasses.replace(result, concurrentStages=concurrent or None)
        payload = result.to_json()
        payload["resultUri"] = write_result(config, result)
        payload["storeBackend"] = storeBackend
        payloads[stage] = payload
        print(
            f"{label} stage done: {stage} status={result.status} "
            f"seconds={result.seconds}",
            flush=True,
        )
        return result.status == "ok"

    try:
        download_started = time.perf_counter()
        print(f"{label} dataset download start: {config.datasetUri(nRows)}", flush=True)
        download_file(config.datasetUri(nRows), local_h5ad)
        download_seconds = time.perf_counter() - download_started
        print(
            f"{label} dataset download done: seconds={download_seconds:.1f}",
            flush=True,
        )
        funnel_started = time.perf_counter()
        for stage in selected:
            for name in [
                name for name in pending if stage not in BACKGROUND_OVERLAPS[name]
            ]:
                if not record(name, pending.pop(name).result()):
                    failed_stage = failed_stage or name
            if failed_stage is not None:
                break
            print(f"{label} stage start: {stage}", flush=True)
            if stage in BACKGROUND_OVERLAPS:
                pending[stage] = BackgroundTask(
                    functools.partial(execute, stage),
                    name=f"profile-{stage}",
                )
            elif not record(stage, execute(stage)):
                failed_stage = stage
                break
        funnel_seconds = time.perf_counter() - funnel_started
    except Exception as exc:  # noqa: BLE001 - persist a durable failure summary
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
    finally:
        # Record background stages even when a later stage failed.
        for name in list(pending):
            try:
                if not record(name, pending.pop(name).result()):
                    failed_stage = failed_stage or name
            except Exception as exc:  # noqa: BLE001 - keep the first failure
                failed_stage = failed_stage or name
                error = error or f"{type(exc).__name__}: {exc}"
        measurement = sampler.stop()
    if failed_stage is not None:
        status = "error"
        error = error or payloads[failed_stage].get("error") or f"{failed_stage} failed"

    outcomes = [payloads[stage] for stage in selected if stage in payloads]
    summary: dict[str, Any] = {
        "runTag": config.runTag,
        "submissionId": submissionId,
        "nRows": nRows,
        "status": status,
        "stopped": status != "ok",
        "error": error,
        "failedStage": failed_stage,
        "storeBackend": storeBackend,
        "storeUri": store_uri,
        "datasetUri": config.datasetUri(nRows),
        "datasetDownloadSeconds": download_seconds,
        "funnelSeconds": funnel_seconds,
        "wholeFunctionSeconds": time.perf_counter() - started,
        "modalResources": resource_envelope,
        "stageOrder": list(selected),
        "completedStages": [
            item["stage"] for item in outcomes if item["status"] == "ok"
        ],
        "outcomes": outcomes,
        "utilization": [
            {"stage": item["stage"], **(item.get("utilization") or {})}
            for item in outcomes
        ],
        "claimUri": config.e2eClaimUri(),
        "funnelResultUri": config.funnelResultUri(nRows),
        **summarize_resource_measurement(measurement),
        "provenance": provenance_from_config(config, nonpreemptible=True),
    }
    write_funnel_result(config, nRows, summary)
    return summary


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
    submissionId: str,
    stages: list[StageName] | None = None,
) -> dict[str, Any]:
    config = ProfilingConfig.model_validate(configDict)
    os.environ.setdefault("R2_ENDPOINT", config.r2EndpointUrl)
    if nRows not in config.targetSizes:
        raise ValueError(f"size {nRows} is not in config.targetSizes")
    # The claim stops a second delivery of this coordinator. Each stage job
    # claims itself and rejects results from earlier submissions.
    claim_submission(config, nRows, "size", submissionId)
    selected_stages = tuple(stages) if stages else config.effectiveStages
    outcomes: list[dict[str, Any]] = []
    for stage in selected_stages:
        resources = config.resourcesFor(stage)
        options = modal_function_options(
            config,
            resources,
            maxContainers=max(1, len(config.targetSizes)),
            retries=0,
        )
        call = run_stage_job.with_options(**options).spawn(
            configDict, nRows, stage, submissionId
        )
        try:
            result = await_stage_result(
                config,
                nRows,
                stage,
                call,
                submissionId=submissionId,
                deadlineSeconds=float(resources.timeoutSeconds) + DEFAULT_GRACE_SECONDS,
            )
        except Exception as exc:
            if stage not in CONSUME_STAGES:
                raise
            result = {
                "submissionId": submissionId,
                "nRows": nRows,
                "stage": stage,
                "status": "error",
                "error": str(exc),
                "resultUri": config.resultUri(nRows, stage),
            }
        outcomes.append(result)
        if result.get("status") == "error" and stage not in CONSUME_STAGES:
            break
    failed = next((item for item in outcomes if item.get("status") == "error"), None)
    return {
        "submissionId": submissionId,
        "nRows": nRows,
        "stopped": failed is not None,
        "failed": failed,
        "outcomes": outcomes,
    }


@app.function(
    **COMMON_FUNCTION_OPTIONS,
    timeout=86_400,
    memory=2048,
    cpu=1.0,
    ephemeral_disk=BASE_EPHEMERAL_DISK_MB,
)
def run_all_jobs(
    configDict: dict[str, Any],
    submissionId: str,
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
            submissionId,
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
        "submissionId": submissionId,
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


def _launch(
    config: ProfilingConfig,
    name: str,
    options: dict[str, Any],
    *args: Any,
    ephemeral: bool = False,
    label: str,
) -> None:
    """Spawn job ``name`` from the deployed app, or from this app when ephemeral.

    An ephemeral app ends with this entrypoint, so wait for the call there.
    """
    function = globals()[name] if ephemeral else _deployed_function(config, name)
    call = function.with_options(**options).spawn(*args)
    _print_spawned(label, call)
    if ephemeral:
        print(await_function_call(call, deadlineSeconds=86_400.0))


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
    run_parser.add_argument("--stage", choices=ALL_STAGE_CHOICES, required=True)
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
        "--stages", nargs="*", choices=ALL_STAGE_CHOICES, default=None
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
        "--stages", nargs="*", choices=ALL_STAGE_CHOICES, default=None
    )
    local_parser.add_argument(
        "--ephemeral",
        action="store_true",
        help="Spawn from this modal run app (no deploy). Prefer --detach.",
    )

    e2e_parser = sub.add_parser(
        "run-e2e",
        help="One-container graph-construction funnel with a fresh R2 Zarr store",
    )
    e2e_parser.add_argument("--config", required=True)
    e2e_parser.add_argument("--size", type=int, required=True)
    e2e_parser.add_argument(
        "--ephemeral",
        action="store_true",
        help=(
            "Spawn from this modal run app (no deploy). Prefer --detach. "
            "Needed to pick up decorator changes such as nonpreemptible "
            "before redeploying."
        ),
    )

    args = parser.parse_args(list(arg_list))
    config = _load_config(args.config)
    payload = attach_client_provenance(
        config.model_dump(mode="python"),
        configPath=args.config,
    )

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
        _launch(
            config,
            "prepare_datasets",
            modal_function_options(config, config.prepareResources),
            payload,
            label="prepare_datasets",
        )
        return

    if args.command == "prepare-fixture":
        sizes = list(args.sizes) if args.sizes else [10_000]
        for size in sizes:
            if size not in config.targetSizes:
                raise SystemExit(f"size {size} is not in config.targetSizes")
        _launch(
            config,
            "prepare_fixture_datasets_job",
            modal_function_options(config, config.resourcesFor("reopenStore")),
            payload,
            sizes,
            args.n_columns,
            label="prepare_fixture_datasets_job",
        )
        return

    submission_id = uuid4().hex

    if args.command == "run":
        if args.size not in config.targetSizes:
            raise SystemExit(f"size {args.size} is not in config.targetSizes")
        # Fail fast here; the stage job itself also rejects an existing result.
        existing = None if args.force else load_result(config, args.size, args.stage)
        if existing is not None:
            failed = existing.get("status") == "error"
            print(
                {
                    "nRows": args.size,
                    "stage": args.stage,
                    "status": "error" if failed else "skipped",
                    "resultUri": config.resultUri(args.size, args.stage),
                }
            )
            if failed:
                raise SystemExit(1)
            return
        spawn_args: tuple[Any, ...] = (payload, args.size, args.stage, submission_id)
        if args.force:
            spawn_args += (True,)
        _launch(
            config,
            "run_stage_job",
            modal_function_options(config, config.resourcesFor(args.stage), retries=0),
            *spawn_args,
            ephemeral=args.ephemeral,
            label=f"run_stage_job {args.size}/{args.stage}",
        )
        return

    if args.command == "run-all":
        sizes = list(args.sizes) if args.sizes else None
        stages = list(args.stages) if args.stages else None
        if sizes:
            for size in sizes:
                if size not in config.targetSizes:
                    raise SystemExit(f"size {size} is not in config.targetSizes")
        _launch(
            config,
            "run_all_jobs",
            orchestrator_function_options(config),
            payload,
            submission_id,
            sizes,
            stages,
            ephemeral=args.ephemeral,
            label="run_all_jobs",
        )
        return

    if args.command in {"run-e2e", "run-local"}:
        if args.size not in config.targetSizes:
            raise SystemExit(f"size {args.size} is not in config.targetSizes")
        if not config.runTag.strip():
            raise SystemExit(f"{args.command} requires a non-empty runTag")
        backend = "r2" if args.command == "run-e2e" else "local"
        stages = (
            list(CORE_STAGE_ORDER)
            if backend == "r2"
            else list(args.stages or config.effectiveStages)
        )
        print(f"result URI (when done): {config.funnelResultUri(args.size)}")
        _launch(
            config,
            "run_funnel_job",
            _e2e_function_options(config, tuple(stages)),
            payload,
            args.size,
            submission_id,
            backend,
            stages,
            ephemeral=args.ephemeral,
            label=f"run_funnel_job {backend} {args.size}",
        )
        return

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
funnel runs its stages one at a time.
Watch progress with:
  uv run --group profiling modal app logs scarf-profiling --env scarf_profiling
"""

import argparse
import dataclasses
import os
import shutil
import time
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import modal
from scarf.storage import ArtifactRef
from scarf.storage.stores import zarr_location_has_content

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
    require_consume_only_override,
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
    require_base_ephemeral_disk,
    validate_modal_environment,
)
from profiling.modal_support import (
    MODAL_APP_NAME,
    MODAL_ENVIRONMENT_NAME,
    await_function_call,
    await_function_calls,
    call_id,
)
from profiling.provenance import attach_client_provenance, provenance_from_config
from profiling.r2 import (
    ObjectDownload,
    delete_prefix,
    download_file,
    object_exists,
    object_size,
    put_json_if_absent,
    storage_options,
    upload_file,
)
from profiling.results import (
    claim_stage,
    claim_submission,
    load_result,
    release_stage_claim,
    result_exists,
    write_funnel_result,
    write_result,
)
from profiling.spawn_wait import DEFAULT_GRACE_SECONDS, await_stage_result
from profiling.metrics import ResourceSampler
from profiling.stages import (
    StageRunResult,
    discover_consume_inputs,
    profile_stage_inputs,
    require_artifact_ref,
    run_stage,
    session_resource_mismatches,
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
    return require_artifact_ref(
        ArtifactRef.from_dict(artifact),
        kind=kind,
        assay=config.workflow.assayName,
        label=f"The {stage} stage result artifact",
    )


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


def _stage_context(
    config: ProfilingConfig,
    stage: StageName,
    stages: list[StageName] | None,
) -> tuple[StageName, ...]:
    """Return the stages of the run that a single stage job belongs to."""
    selected = tuple(stages) if stages else config.effectiveStages
    return selected if stage in selected else (*selected, stage)


def _store_has_content(storeUri: str) -> bool:
    """Return whether the createStore destination holds any key or local path."""
    return zarr_location_has_content(
        storeUri, storage_options=storage_options(storeUri)
    )


def _delete_forced_store(config: ProfilingConfig, nRows: int, storeUri: str) -> None:
    """Delete the store that a forced createStore replaces.

    Writers create a store only at an empty destination and never replace one
    that a DataStore has opened, as initializeStore does, so a forced
    createStore deletes the store itself instead of relying on the import to
    replace it.
    """
    print(
        "[createStore] force: deleting any existing store of "
        f"runTag={config.runTag!r} size={nRows} at {storeUri}",
        flush=True,
    )
    deleted = delete_prefix(storeUri)
    print(
        f"[createStore] force: deleted {deleted.objectCount} objects "
        f"({deleted.totalBytes} bytes) of runTag={config.runTag!r} size={nRows}",
        flush=True,
    )


def _dataset_fields(
    config: ProfilingConfig,
    nRows: int,
    download: ObjectDownload,
) -> dict[str, Any]:
    return {
        "datasetUri": config.datasetUri(nRows),
        "datasetETag": download.eTag,
        "datasetBytes": download.fileBytes,
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
        *(config.stageClaimUri(nRows, stage) for stage in stages),
    ]
    if storeOnR2:
        candidates.insert(0, f"{config.storeUri(nRows).rstrip('/')}/zarr.json")
    return [uri for uri in candidates if object_exists(uri)]


def _require_funnel_settings(
    config: ProfilingConfig,
    nRows: int,
    stages: tuple[StageName, ...],
    *,
    storeOnR2: bool,
) -> WorkflowParameters:
    """Reject a funnel whose settings would not run as recorded; return its workflow."""
    if storeOnR2:
        require_consume_only_override(config, nRows, stages)
    mismatches = session_resource_mismatches(stages, config.resourcesFor)
    if mismatches:
        raise ValueError(
            "A funnel reuses one DataStore across its stages, so they must share "
            "workers and scarfMemoryBudget: " + "; ".join(mismatches)
        )
    return bind_cluster_source(config, nRows, stages)


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
    require_base_ephemeral_disk(max(item.ephemeralDiskMb for item in resources))
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
    """Upload tiny synthetic H5ADs so stage jobs can be tested without Cellxgene.

    Uploads are create-only, so a fixture never replaces a prepared sample. Point
    ``datasetPrefixUri`` at a fixture prefix to test stage jobs with them.
    """
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
        upload_file(artifact.localPath, uri, createOnly=True)
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
    stages: list[StageName] | None = None,
    allowReuse: bool = False,
) -> dict[str, Any]:
    """Run one stage of the run over ``stages`` (default: the configured stages).

    A create-only claim per runTag, size, and stage keeps a second job off the
    same stage, and a runTag held by an e2e funnel is refused. createStore
    refuses a destination that holds anything before it downloads the H5AD;
    a forced createStore deletes the store first instead.
    """
    config = ProfilingConfig.model_validate(configDict)
    resources = config.resourcesFor(stage)
    os.environ.setdefault("R2_ENDPOINT", config.r2EndpointUrl)
    require_consume_only_override(config, nRows, (stage,))
    workflow = bind_cluster_source(config, nRows, _stage_context(config, stage, stages))
    completed = load_result(config, nRows, stage, submissionId=submissionId)
    if completed is not None:
        return completed
    claim_stage(config, nRows, stage, submissionId)
    try:
        if object_exists(config.e2eClaimUri()):
            raise FileExistsError(
                f"runTag {config.runTag!r} is held by an e2e funnel; use a fresh runTag"
            )
        if result_exists(config, nRows, stage) and not force:
            raise FileExistsError(
                "This stage has a previous result; use force or a fresh runTag"
            )
        store_uri = config.storeUri(nRows)
        if stage == "createStore" and not force and _store_has_content(store_uri):
            raise FileExistsError(
                f"createStore would replace the existing store {store_uri}; "
                "use force or a fresh runTag"
            )

        work = _fresh_work_dir(_WORK / f"{submissionId}-{nRows}-{stage}")
        download: ObjectDownload | None = None
        try:
            local_h5ad: Path | None = None
            if stage == "createStore":
                if force:
                    # Deleted before the download, so a failure is recorded
                    # as this stage's result before any expensive work.
                    _delete_forced_store(config, nRows, store_uri)
                local_h5ad = work / f"{nRows}.h5ad"
                download = download_file(config.datasetUri(nRows), local_h5ad)

            result = run_stage(
                stage,
                submissionId=submissionId,
                nRows=nRows,
                storeUri=store_uri,
                workflow=workflow,
                resources=resources,
                localH5adPath=local_h5ad,
                countMatrix=config.countMatrix,
                storageIo=config.storageIo,
                workDir=work,
                invalidateCache=force,
                allowArtifactReuse=allowReuse,
                clientProvenance=config.clientProvenance,
                inputRefs=_load_stage_input_refs(
                    config,
                    nRows,
                    stage,
                    workflow=workflow,
                ),
            )
        except Exception as exc:
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
                storeUri=store_uri,
                error=f"{type(exc).__name__}: {exc}",
                workers=resources.workers,
            )
        if download is not None:
            result = dataclasses.replace(
                result, **_dataset_fields(config, nRows, download)
            )
        write_result(config, result, overwrite=force)
        return result.to_json()
    finally:
        release_stage_claim(config, nRows, stage, submissionId)


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
    (run-local), and the stages run one at a time in order. The create-only
    runTag claim makes the funnel exclusive, and stage jobs refuse a runTag it
    holds.
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
    # findMarkers reads imported clusters only when the funnel imports them.
    workflow = _require_funnel_settings(config, nRows, selected, storeOnR2=not local)
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
    # A stage job may have claimed a stage between the check and the runTag claim.
    late_conflicts = [
        uri
        for uri in _e2e_conflicting_uris(config, nRows, selected, storeOnR2=not local)
        if uri != config.e2eClaimUri()
    ]
    if late_conflicts:
        raise FileExistsError(
            "A stage job started on this runTag: " + ", ".join(late_conflicts)
        )
    if local:
        # Stages wrap the store to count its operations. A wrapped local store
        # is not a LocalStore, so pin the profile the local path resolves to.
        os.environ["SCARF_ZARR_PROFILE"] = "fast_local"

    work = _fresh_work_dir(_WORK / f"{storeBackend}-{config.runTag}-{nRows}")
    local_h5ad = work / f"{nRows}.h5ad"
    if local:
        store_uri = str(work / f"{nRows}.zarr")
    label = "e2e" if not local else "local"

    sampler = ResourceSampler()
    sampler.start()
    started = time.perf_counter()
    download: ObjectDownload | None = None
    download_seconds: float | None = None
    funnel_seconds: float | None = None
    payloads: dict[StageName, dict[str, Any]] = {}
    status = "ok"
    error: str | None = None
    failed_stage: StageName | None = None
    session: dict[str, Any] = {}

    def execute(stage: StageName) -> StageRunResult:
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
            clientProvenance=config.clientProvenance,
            session=session,
        )

    def record(stage: StageName, result: StageRunResult) -> bool:
        if stage == "createStore" and download is not None:
            result = dataclasses.replace(
                result, **_dataset_fields(config, nRows, download)
            )
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
        download = download_file(config.datasetUri(nRows), local_h5ad)
        download_seconds = time.perf_counter() - download_started
        print(
            f"{label} dataset download done: seconds={download_seconds:.1f}",
            flush=True,
        )
        funnel_started = time.perf_counter()
        for stage in selected:
            print(f"{label} stage start: {stage}", flush=True)
            if not record(stage, execute(stage)):
                failed_stage = stage
                break
        funnel_seconds = time.perf_counter() - funnel_started
    except Exception as exc:  # noqa: BLE001 - persist a durable failure summary
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
    finally:
        measurement = sampler.stop()
    if failed_stage is not None:
        status = "error"
        error = payloads[failed_stage].get("error") or f"{failed_stage} failed"

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
        "datasetETag": None if download is None else download.eTag,
        "datasetBytes": None if download is None else download.fileBytes,
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
    allowReuse: bool = False,
) -> dict[str, Any]:
    config = ProfilingConfig.model_validate(configDict)
    os.environ.setdefault("R2_ENDPOINT", config.r2EndpointUrl)
    if nRows not in config.targetSizes:
        raise ValueError(f"size {nRows} is not in config.targetSizes")
    selected_stages = tuple(stages) if stages else config.effectiveStages
    require_consume_only_override(config, nRows, selected_stages)
    bind_cluster_source(config, nRows, selected_stages)
    # The claim stops a second delivery of this coordinator. Each stage job
    # claims its stage and rejects results from earlier submissions.
    claim_submission(config, nRows, "size", submissionId)
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
            configDict,
            nRows,
            stage,
            submissionId,
            False,
            list(selected_stages),
            allowReuse,
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
    allowReuse: bool = False,
) -> dict[str, Any]:
    """Run sizes in parallel; stages within each size stay sequential."""
    config = ProfilingConfig.model_validate(configDict)
    os.environ.setdefault("R2_ENDPOINT", config.r2EndpointUrl)
    selected_sizes = tuple(sizes) if sizes else config.targetSizes
    selected_stages = tuple(stages) if stages else config.effectiveStages
    for n_rows in selected_sizes:
        if n_rows not in config.targetSizes:
            raise ValueError(f"size {n_rows} is not in config.targetSizes")
    if object_exists(config.e2eClaimUri()):
        raise FileExistsError(
            f"runTag {config.runTag!r} is held by an e2e funnel; use a fresh runTag"
        )

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
            allowReuse,
        )
        for n_rows in selected_sizes
    ]
    size_results = await_function_calls(
        handles,
        pollSeconds=20.0,
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
            MODAL_APP_NAME,
            name,
            environment_name=MODAL_ENVIRONMENT_NAME,
        )
    except Exception as exc:
        raise SystemExit(
            f"Could not find deployed function {MODAL_APP_NAME}/{name}. "
            "Deploy first with:\n"
            "  uv run --group profiling modal deploy "
            f"--env {MODAL_ENVIRONMENT_NAME} -m profiling.modal_app\n"
            f"Original error: {exc}"
        ) from exc


def _print_spawned(label: str, call: Any) -> None:
    print(f"spawned {label}: {call_id(call) or call}")
    print("disconnect is safe; watch with:")
    print(
        "  uv run --group profiling modal app logs "
        f"{MODAL_APP_NAME} --env {MODAL_ENVIRONMENT_NAME}"
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
        print(await_function_call(call, pollSeconds=20.0, deadlineSeconds=86_400.0))


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

    allow_reuse_help = (
        "Accept a stage whose artifact already existed. Its seconds then measure a "
        "cache lookup, not the computation."
    )
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--size", type=int, required=True)
    run_parser.add_argument("--stage", choices=ALL_STAGE_CHOICES, required=True)
    run_parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Recompute an existing targeted stage, invalidate reusable artifacts, "
            "and overwrite its stage result JSON. A forced createStore deletes the "
            "existing store first."
        ),
    )
    run_parser.add_argument("--allow-reuse", action="store_true", help=allow_reuse_help)
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
    all_parser.add_argument("--allow-reuse", action="store_true", help=allow_reuse_help)
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
        print(await_function_call(call, pollSeconds=20.0, deadlineSeconds=300.0))
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
        # Fail fast here; the stage job checks the same settings again.
        require_consume_only_override(config, args.size, (args.stage,))
        bind_cluster_source(config, args.size, _stage_context(config, args.stage, None))
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
        _launch(
            config,
            "run_stage_job",
            modal_function_options(config, config.resourcesFor(args.stage), retries=0),
            payload,
            args.size,
            args.stage,
            submission_id,
            args.force,
            None,
            args.allow_reuse,
            ephemeral=args.ephemeral,
            label=f"run_stage_job {args.size}/{args.stage}",
        )
        return

    if args.command == "run-all":
        sizes = list(args.sizes) if args.sizes else None
        stages = list(args.stages) if args.stages else None
        selected_stages = tuple(stages) if stages else config.effectiveStages
        for size in sizes or config.targetSizes:
            if size not in config.targetSizes:
                raise SystemExit(f"size {size} is not in config.targetSizes")
            require_consume_only_override(config, size, selected_stages)
            bind_cluster_source(config, size, selected_stages)
        _launch(
            config,
            "run_all_jobs",
            orchestrator_function_options(config),
            payload,
            submission_id,
            sizes,
            stages,
            args.allow_reuse,
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
        _require_funnel_settings(
            config, args.size, tuple(stages), storeOnR2=backend == "r2"
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

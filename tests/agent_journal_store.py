"""In-memory journal fixture without numerical work or a provider."""

from types import SimpleNamespace

import zarr
from zarr.storage import MemoryStore

from scarf.agent.orchestrator import AutomatedWorkflowConfig, AutomatedWorkflowRequest
from scarf.agent.orchestrator import journal
from scarf.agent.orchestrator.models import OrchestrationRequestRecord


def memory_journal(workspace: str | None = None):
    root = zarr.open_group(store=MemoryStore(), mode="w")
    active = root if workspace is None else root.create_group(workspace)
    store = SimpleNamespace(
        zw=active,
        z=root,
        zarr_loc="analysis.zarr",
        workspace=workspace,
        cells=SimpleNamespace(columns=[]),
        load_artifact=lambda ref: ref,
    )
    prefix = journal._ensure_orchestration_store(store)
    request = AutomatedWorkflowRequest(
        sourcePath="analysis.zarr",
        zarrPath="analysis.zarr",
        workspace=workspace,
        primaryAssay="RNA",
        markerAssay="RNA",
        analysisAssays=["RNA"],
        studyContext="Two replicated conditions",
        studyObjective="Resolve stable populations",
    )
    config = AutomatedWorkflowConfig()
    record = OrchestrationRequestRecord(
        workflowRunId="workflow-1",
        request=request,
        config=config,
        requestSha256=journal._sha256_model(request),
        configSha256=journal._sha256_model(config),
        modelIdentity="test-model",
        inputIdentity={"data": "selected-rna"},
    )
    record = record.model_copy(
        update={"contentSha256": journal._record_checksum(record)}
    )
    journal._write_model_once(
        active, journal._request_key(prefix, record.workflowRunId), record
    )
    return store, prefix, record

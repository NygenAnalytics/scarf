"""Immutable JSON persistence for structured Scarf agent workflows.

Agent records are stored as plain JSON keys beneath an ``agents`` Zarr group in
the active Scarf data group. The namespace group contains metadata only; report
models are not stored as Zarr arrays or encoded chunks. Format version 2 is a
single-writer format with no implicit migration from earlier layouts.
"""

import hashlib
import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Literal, cast

import zarr
from pydantic import Field, field_validator, model_validator
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync

from ..datastore.datastore import DataStore
from ..storage.schema import validate_workspace_name
from ..utils.logging import logger
from . import record_io
from .biological_interpretation import BiologicalInterpretationReport
from .config import AgentRunConfig
from .data_enrichment import DataEnrichmentReport
from .experimental_context import ExperimentalContextResult
from .parameter_tuning import ParameterTuningReport
from .types import (
    AgentDataModel,
    ArtifactReferenceModel,
    ExperimentalBiologyHandoff,
    ExperimentalTuningHandoff,
    TuningBiologyHandoff,
)

type AgentName = Literal[
    "data_enrichment",
    "experimental_context",
    "parameter_tuning",
    "biological_interpretation",
]
type AgentReportType = Literal[
    "",
    "DataEnrichmentReport",
    "ExperimentalContextResult",
    "ParameterTuningReport",
    "BiologicalInterpretationReport",
]
type AgentReport = (
    DataEnrichmentReport
    | ExperimentalContextResult
    | ParameterTuningReport
    | BiologicalInterpretationReport
)
type AgentPersistenceTarget = str | Path | zarr.Group | DataStore
type AgentWorkflowStatus = Literal["running", "completed", "failed", "abandoned"]
type AgentTerminalStatus = Literal["completed", "failed", "abandoned"]

_FORMAT = "scarf_agent_reports"
_RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,127}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REPORT_TYPES: dict[AgentName, type[AgentDataModel]] = {
    "data_enrichment": DataEnrichmentReport,
    "experimental_context": ExperimentalContextResult,
    "parameter_tuning": ParameterTuningReport,
    "biological_interpretation": BiologicalInterpretationReport,
}
_AGENT_NAMES: dict[type[AgentDataModel], AgentName] = {
    report_type: agent_name for agent_name, report_type in _REPORT_TYPES.items()
}


class AgentReportLink(AgentDataModel):
    """Immutable identity of one report used as an invocation parent."""

    type: Literal["agentReportLink"] = "agentReportLink"
    workflowRunId: str = ""
    workspace: str | None = None
    agentName: AgentName = "data_enrichment"
    agentRunId: str = ""
    contentSha256: str = ""

    @field_validator("workflowRunId", "agentRunId")
    @classmethod
    def validate_run_ids(cls, value: str) -> str:
        if value:
            _validate_run_id(value, "run ID")
        return value

    @field_validator("workspace")
    @classmethod
    def validate_workspace(cls, value: str | None) -> str | None:
        validate_workspace_name(value)
        return value

    @field_validator("contentSha256")
    @classmethod
    def validate_content_sha256(cls, value: str) -> str:
        if value and _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("contentSha256 must be a lowercase SHA-256 digest")
        return value

    @model_validator(mode="after")
    def validate_complete_identity(self) -> "AgentReportLink":
        if self.workflowRunId or self.agentRunId or self.contentSha256:
            if not self.workflowRunId or not self.agentRunId or not self.contentSha256:
                raise ValueError("A parent report link requires a complete identity")
        return self

    @classmethod
    def from_reference(cls, reference: "AgentReportReference") -> "AgentReportLink":
        return cls(
            workflowRunId=reference.workflowRunId,
            workspace=reference.workspace,
            agentName=reference.agentName,
            agentRunId=reference.agentRunId,
            contentSha256=reference.contentSha256,
        )

    @classmethod
    def get_blank(cls) -> "AgentReportLink":
        return cls()

    @classmethod
    def get_example(cls) -> "AgentReportLink":
        return cls(
            workflowRunId="workflow-1",
            agentName="experimental_context",
            agentRunId="experimental-run-1",
            contentSha256="0" * 64,
        )


class AgentInvocation(AgentDataModel):
    """Replay-relevant inputs and typed handoffs for one agent invocation."""

    agentName: AgentName = "data_enrichment"
    parentReports: list[AgentReportLink] = Field(default_factory=list)
    inputs: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, ArtifactReferenceModel] = Field(default_factory=dict)
    runConfig: AgentRunConfig = Field(default_factory=AgentRunConfig)
    experimentalTuningHandoff: ExperimentalTuningHandoff | None = None
    experimentalBiologyHandoff: ExperimentalBiologyHandoff | None = None
    tuningBiologyHandoff: TuningBiologyHandoff | None = None

    @model_validator(mode="after")
    def validate_parent_reports(self) -> "AgentInvocation":
        identities = [
            (parent.workflowRunId, parent.agentName, parent.agentRunId)
            for parent in self.parentReports
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("parentReports must not contain duplicate reports")
        return self

    @classmethod
    def get_blank(cls) -> "AgentInvocation":
        return cls()

    @classmethod
    def get_example(cls) -> "AgentInvocation":
        return cls(
            agentName="parameter_tuning",
            parentReports=[AgentReportLink.get_example()],
            inputs={"fromAssay": "RNA", "cellKey": "I"},
            runConfig=AgentRunConfig.get_example(),
            experimentalTuningHandoff=ExperimentalTuningHandoff(batchAction="skip"),
        )


class AgentReportReference(AgentDataModel):
    """Stable identity for one immutable agent report."""

    type: Literal["agentReport"] = "agentReport"
    workflowRunId: str = ""
    workspace: str | None = None
    agentName: AgentName = "data_enrichment"
    agentRunId: str = ""
    reportType: AgentReportType = ""
    executionRunId: str = ""
    createdAtNs: int = Field(default=0, ge=0, strict=True)
    complete: bool = Field(default=False, strict=True)
    parentReports: list[AgentReportLink] = Field(default_factory=list)
    contentSha256: str = ""

    @field_validator("workflowRunId", "agentRunId")
    @classmethod
    def validate_run_ids(cls, value: str) -> str:
        if value:
            _validate_run_id(value, "run ID")
        return value

    @field_validator("workspace")
    @classmethod
    def validate_workspace(cls, value: str | None) -> str | None:
        validate_workspace_name(value)
        return value

    @field_validator("contentSha256")
    @classmethod
    def validate_content_sha256(cls, value: str) -> str:
        if value and _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("contentSha256 must be a lowercase SHA-256 digest")
        return value

    @model_validator(mode="after")
    def validate_complete_identity(self) -> "AgentReportReference":
        has_identity = bool(
            self.workflowRunId
            or self.agentRunId
            or self.reportType
            or self.createdAtNs
            or self.complete
            or self.contentSha256
        )
        if has_identity and (
            not self.workflowRunId
            or not self.agentRunId
            or not self.reportType
            or self.createdAtNs < 1
            or not self.complete
            or not self.contentSha256
        ):
            raise ValueError("An agent report reference requires a complete identity")
        return self

    @classmethod
    def get_blank(cls) -> "AgentReportReference":
        return cls()

    @classmethod
    def get_example(cls) -> "AgentReportReference":
        return cls(
            workflowRunId="workflow-1",
            agentName="data_enrichment",
            agentRunId="agent-run-1",
            reportType="DataEnrichmentReport",
            executionRunId="provider-run-1",
            createdAtNs=1,
            complete=True,
            contentSha256="0" * 64,
        )


class AgentReportRecord(AgentDataModel):
    """Complete JSON envelope for one immutable report and its invocation."""

    recordType: Literal["agentReport"] = "agentReport"
    formatVersion: Literal[2] = 2
    reference: AgentReportReference = Field(default_factory=AgentReportReference)
    invocation: AgentInvocation = Field(default_factory=AgentInvocation)
    report: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_identity(self) -> "AgentReportRecord":
        if self.reference.agentName != self.invocation.agentName:
            raise ValueError("Report reference and invocation agent names differ")
        if self.reference.parentReports != self.invocation.parentReports:
            raise ValueError("Report reference and invocation parents differ")
        if any(
            parent.workflowRunId == self.reference.workflowRunId
            and parent.agentName == self.reference.agentName
            and parent.agentRunId == self.reference.agentRunId
            for parent in self.invocation.parentReports
        ):
            raise ValueError("An agent report cannot cite itself as a parent")
        return self

    @classmethod
    def get_blank(cls) -> "AgentReportRecord":
        return cls()

    @classmethod
    def get_example(cls) -> "AgentReportRecord":
        report = DataEnrichmentReport.get_example()
        return cls(
            reference=AgentReportReference.get_example(),
            invocation=AgentInvocation(agentName="data_enrichment"),
            report=report.model_dump(mode="json"),
        )


class AgentWorkflowRun(AgentDataModel):
    """One dataset-bound workflow and its immutable report records."""

    type: Literal["agentWorkflowRun"] = "agentWorkflowRun"
    formatVersion: Literal[2] = 2
    workflowRunId: str = ""
    workspace: str | None = None
    createdAtNs: int = Field(default=0, ge=0, strict=True)
    finalizedAtNs: int = Field(default=0, ge=0, strict=True)
    status: AgentWorkflowStatus = "running"
    finalizationMessage: str = ""
    analysisStore: str = ""
    datasetFingerprints: dict[str, str] = Field(default_factory=dict)
    reports: list[AgentReportReference] = Field(default_factory=list)

    @field_validator("workflowRunId")
    @classmethod
    def validate_workflow_run_id(cls, value: str) -> str:
        if value:
            _validate_run_id(value, "workflowRunId")
        return value

    @field_validator("workspace")
    @classmethod
    def validate_workspace(cls, value: str | None) -> str | None:
        validate_workspace_name(value)
        return value

    @field_validator("datasetFingerprints")
    @classmethod
    def validate_dataset_fingerprints(cls, value: dict[str, str]) -> dict[str, str]:
        if any(not assay or not fingerprint for assay, fingerprint in value.items()):
            raise ValueError("Dataset fingerprint names and values must be non-empty")
        return dict(sorted(value.items()))

    @model_validator(mode="after")
    def validate_lifecycle(self) -> "AgentWorkflowRun":
        if self.workflowRunId and self.createdAtNs < 1:
            raise ValueError("A workflow requires a positive createdAtNs")
        if self.workflowRunId and not self.datasetFingerprints:
            raise ValueError("A workflow requires exact dataset fingerprints")
        if self.status == "running" and self.finalizedAtNs != 0:
            raise ValueError("A running workflow cannot have finalizedAtNs")
        if self.status == "running" and self.finalizationMessage:
            raise ValueError("A running workflow cannot have a finalizationMessage")
        if self.status != "running" and self.finalizedAtNs < 1:
            raise ValueError("A terminal workflow requires finalizedAtNs")
        if (
            self.status != "running"
            and self.createdAtNs
            and self.finalizedAtNs < self.createdAtNs
        ):
            raise ValueError("finalizedAtNs cannot precede createdAtNs")
        return self

    @classmethod
    def get_blank(cls) -> "AgentWorkflowRun":
        return cls()

    @classmethod
    def get_example(cls) -> "AgentWorkflowRun":
        return cls(
            workflowRunId="workflow-1",
            createdAtNs=1,
            analysisStore="analysis.zarr",
            datasetFingerprints={"RNA": "dataset-1"},
            reports=[AgentReportReference.get_example()],
        )


class AgentStoreManifest(AgentDataModel):
    """Identity document for one workspace-local agent JSON store."""

    type: Literal["agentReportStore"] = "agentReportStore"
    format: Literal["scarf_agent_reports"] = "scarf_agent_reports"
    formatVersion: Literal[2] = 2
    workspace: str | None = None

    @field_validator("workspace")
    @classmethod
    def validate_workspace(cls, value: str | None) -> str | None:
        validate_workspace_name(value)
        return value

    @classmethod
    def get_blank(cls) -> "AgentStoreManifest":
        return cls()

    @classmethod
    def get_example(cls) -> "AgentStoreManifest":
        return cls(workspace="analysis")


class AgentWorkflowFinalization(AgentDataModel):
    """Immutable terminal event for a workflow."""

    recordType: Literal["agentWorkflowFinalization"] = "agentWorkflowFinalization"
    formatVersion: Literal[2] = 2
    workflowRunId: str = ""
    workspace: str | None = None
    status: AgentTerminalStatus = "completed"
    finalizedAtNs: int = Field(default=0, ge=0, strict=True)
    message: str = ""

    @field_validator("workflowRunId")
    @classmethod
    def validate_workflow_run_id(cls, value: str) -> str:
        if value:
            _validate_run_id(value, "workflowRunId")
        return value

    @field_validator("workspace")
    @classmethod
    def validate_workspace(cls, value: str | None) -> str | None:
        validate_workspace_name(value)
        return value

    @model_validator(mode="after")
    def validate_finalization(self) -> "AgentWorkflowFinalization":
        if self.workflowRunId and self.finalizedAtNs < 1:
            raise ValueError("A finalization requires a positive finalizedAtNs")
        return self

    @classmethod
    def get_blank(cls) -> "AgentWorkflowFinalization":
        return cls()

    @classmethod
    def get_example(cls) -> "AgentWorkflowFinalization":
        return cls(
            workflowRunId="workflow-1",
            status="completed",
            finalizedAtNs=2,
        )


def _validate_run_id(value: str, label: str) -> str:
    if _RUN_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{label} must be one safe path component containing 1-128 ASCII "
            "lowercase letters, numbers, underscores, or hyphens"
        )
    return value


def _key_exists(group: zarr.Group, key: str) -> bool:
    return record_io.read_key(group, key) is not None


def _list_keys(group: zarr.Group, prefix: str) -> list[str]:
    if not group.store.supports_listing:
        raise NotImplementedError("Agent persistence requires a listable Zarr store")
    return record_io.list_keys(group, prefix)


def _write_key_once(group: zarr.Group, key: str, payload: bytes) -> None:
    store = group.store
    if bool(getattr(store, "read_only", False)) or not bool(
        getattr(store, "supports_writes", True)
    ):
        raise PermissionError("Agent persistence target is read-only")
    if _key_exists(group, key):
        raise FileExistsError(f"Immutable agent record {key!r} already exists")
    buffer = default_buffer_prototype().buffer.from_bytes(payload)
    sync(store.set_if_not_exists(key, buffer))
    stored = record_io.read_key(group, key)
    if stored is None:
        raise RuntimeError(f"Agent record {key!r} was not stored")
    if stored != payload:
        raise FileExistsError(
            f"Immutable agent record {key!r} was written by another writer"
        )


def _read_json_model(
    group: zarr.Group,
    key: str,
    model_type: type[AgentDataModel],
) -> AgentDataModel:
    payload = record_io.read_key(group, key)
    if payload is None:
        raise FileNotFoundError(key)
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Agent JSON record {key!r} is malformed") from exc
    try:
        return model_type.model_validate(decoded)
    except ValueError as exc:
        raise ValueError(
            f"Agent JSON record {key!r} does not match its Pydantic schema"
        ) from exc


def _validate_scarf_data_group(group: zarr.Group) -> None:
    import_source = group.attrs.get("scarf:import_source")
    if import_source is not None and not bool(
        group.attrs.get("scarf:import_complete", False)
    ):
        raise RuntimeError(f"{import_source} import is incomplete")
    if group.attrs.get("format") == _FORMAT:
        raise ValueError(
            "Standalone agent-report sidecars require an explicit migration"
        )
    if "cellData" not in group or not isinstance(group["cellData"], zarr.Group):
        raise ValueError("Agent persistence requires an existing Scarf data group")
    assay_names = []
    for name in sorted(dict.fromkeys(group.group_keys())):
        child = group[name]
        if isinstance(child, zarr.Group) and "is_assay" in child.attrs:
            assay_names.append(name)
    if not assay_names:
        raise ValueError("Agent persistence requires at least one Scarf assay")


def _resolve_target(
    target: AgentPersistenceTarget,
    *,
    write: bool,
    workspace: str | None,
) -> tuple[zarr.Group, DataStore | None, str | None, str]:
    validate_workspace_name(workspace)
    datastore: DataStore | None = None
    analysis_store = ""
    if isinstance(target, DataStore):
        datastore = target
        if workspace is not None and workspace != target.workspace:
            raise ValueError("workspace does not match the DataStore workspace")
        if write and target.zarr_mode != "r+":
            raise PermissionError("Agent persistence requires a writable DataStore")
        group = target.zw
        resolved_workspace = target.workspace
        analysis_store = str(target.zarr_loc)
    elif isinstance(target, zarr.Group):
        target_path = str(getattr(target, "path", "")).strip("/")
        if workspace is None:
            group = target
            resolved_workspace = target_path or None
            if resolved_workspace is not None:
                validate_workspace_name(resolved_workspace)
        elif target_path == workspace:
            group = target
            resolved_workspace = workspace
        elif not target_path and workspace in target:
            child = target[workspace]
            if not isinstance(child, zarr.Group):
                raise TypeError(f"Workspace {workspace!r} is not a Zarr group")
            group = child
            resolved_workspace = workspace
        else:
            raise ValueError("workspace does not match the supplied Zarr group")
    else:
        location = str(target)
        if isinstance(target, Path) and not target.exists():
            raise FileNotFoundError(target)
        root = zarr.open_group(location, mode="r+" if write else "r")
        analysis_store = location
        if workspace is None:
            group = root
            resolved_workspace = None
        else:
            if workspace not in root:
                raise KeyError(f"Unknown Scarf workspace {workspace!r}")
            child = root[workspace]
            if not isinstance(child, zarr.Group):
                raise TypeError(f"Workspace {workspace!r} is not a Zarr group")
            group = child
            resolved_workspace = workspace
    _validate_scarf_data_group(group)
    return group, datastore, resolved_workspace, analysis_store


def _live_dataset_fingerprints(
    group: zarr.Group,
    datastore: DataStore | None,
    *,
    ensure: bool,
) -> dict[str, str]:
    if datastore is not None:
        assay_names = list(datastore.assay_names)
        fingerprints = {
            assay_name: (
                datastore._ensure_dataset_fingerprint(assay_name)
                if ensure
                else str(
                    datastore._get_assay(assay_name).attrs.get("dataset_fingerprint")
                    or ""
                )
            )
            for assay_name in assay_names
        }
    else:
        assay_names = []
        fingerprints = {}
        for name in sorted(dict.fromkeys(group.group_keys())):
            child = group[name]
            if isinstance(child, zarr.Group) and "is_assay" in child.attrs:
                assay_names.append(name)
                fingerprints[name] = str(child.attrs.get("dataset_fingerprint") or "")
    if not assay_names:
        raise ValueError("Dataset binding requires at least one assay")
    missing = [name for name in assay_names if not fingerprints[name]]
    if missing:
        raise ValueError(
            "Dataset fingerprints are missing for assays " + repr(sorted(missing))
        )
    return dict(sorted(fingerprints.items()))


def _validate_dataset_binding(
    stored: dict[str, str],
    observed: dict[str, str],
) -> None:
    if stored != observed:
        raise ValueError(
            "Agent workflow dataset fingerprints do not match the current store: "
            f"stored={stored!r}, observed={observed!r}"
        )


def _agents_prefix(group: zarr.Group) -> str:
    return record_io.join_key(str(getattr(group, "path", "")), "agents")


def _manifest_key(prefix: str) -> str:
    return record_io.join_key(prefix, "store.json")


def _workflow_prefix(prefix: str, workflow_run_id: str) -> str:
    return record_io.join_key(prefix, "runs", workflow_run_id)


def _workflow_key(prefix: str, workflow_run_id: str) -> str:
    return record_io.join_key(
        _workflow_prefix(prefix, workflow_run_id),
        "workflow.json",
    )


def _finalization_key(prefix: str, workflow_run_id: str) -> str:
    return record_io.join_key(
        _workflow_prefix(prefix, workflow_run_id),
        "finalization.json",
    )


def _report_key(
    prefix: str,
    workflow_run_id: str,
    agent_name: AgentName,
    agent_run_id: str,
) -> str:
    return record_io.join_key(
        _workflow_prefix(prefix, workflow_run_id),
        agent_name,
        agent_run_id,
        "report.json",
    )


def _open_agent_store(
    group: zarr.Group,
    *,
    workspace: str | None,
    initialize: bool,
) -> str:
    prefix = _agents_prefix(group)
    if "agents" in group:
        node = group["agents"]
        if (
            isinstance(node, zarr.Group)
            and node.attrs.get("format") == _FORMAT
            and node.attrs.get("format_version") == 1
        ):
            raise ValueError(
                "Zarr-backed agent report format version 1 requires an explicit "
                "migration"
            )
        if not isinstance(node, zarr.Group):
            raise ValueError("The agents namespace must be a Zarr group")
        if node.attrs.get("format") != _FORMAT or node.attrs.get("format_version") != 2:
            raise ValueError(
                "The agents namespace collides with an unrecognized Zarr group"
            )
    else:
        existing_keys = _list_keys(group, prefix)
        if existing_keys:
            raise ValueError(
                "A plain-JSON agents hierarchy without Zarr group metadata "
                "requires an explicit migration"
            )
        if not initialize:
            raise FileNotFoundError(_manifest_key(prefix))
        if bool(getattr(group.store, "read_only", False)) or not bool(
            getattr(group.store, "supports_writes", True)
        ):
            raise PermissionError("Agent persistence target is read-only")
        group.create_group(
            "agents",
            attributes={"format": _FORMAT, "format_version": 2},
        )

    manifest_key = _manifest_key(prefix)
    payload = record_io.read_key(group, manifest_key)
    if payload is None:
        if not initialize:
            raise FileNotFoundError(manifest_key)
        metadata_keys = {
            record_io.join_key(prefix, "zarr.json"),
            record_io.join_key(prefix, ".zgroup"),
            record_io.join_key(prefix, ".zattrs"),
        }
        if any(key not in metadata_keys for key in _list_keys(group, prefix)):
            raise ValueError("Refusing to initialize a non-empty agents hierarchy")
        manifest = AgentStoreManifest(workspace=workspace)
        _write_key_once(
            group,
            manifest_key,
            record_io.display_json_bytes(manifest.model_dump(mode="json")),
        )
    manifest = cast(
        AgentStoreManifest,
        _read_json_model(group, manifest_key, AgentStoreManifest),
    )
    if manifest.workspace != workspace:
        raise ValueError("Agent store workspace does not match the active data group")
    return prefix


def _load_workflow_record(
    group: zarr.Group,
    prefix: str,
    workflow_run_id: str,
    workspace: str | None,
) -> AgentWorkflowRun:
    workflow_run_id = _validate_run_id(workflow_run_id, "workflow_run_id")
    key = _workflow_key(prefix, workflow_run_id)
    try:
        workflow = cast(
            AgentWorkflowRun,
            _read_json_model(group, key, AgentWorkflowRun),
        )
    except FileNotFoundError as exc:
        raise KeyError(f"Unknown agent workflow {workflow_run_id!r}") from exc
    if workflow.workflowRunId != workflow_run_id:
        raise ValueError("Agent workflow identity does not match its path")
    if workflow.workspace != workspace:
        raise ValueError("Agent workflow workspace does not match its path")
    if (
        workflow.status != "running"
        or workflow.finalizedAtNs != 0
        or workflow.finalizationMessage
        or workflow.reports
    ):
        raise ValueError(
            "Immutable workflow.json must contain the running identity only"
        )
    finalization_payload = record_io.read_key(
        group,
        _finalization_key(prefix, workflow_run_id),
    )
    if finalization_payload is None:
        return workflow
    finalization = cast(
        AgentWorkflowFinalization,
        _read_json_model(
            group,
            _finalization_key(prefix, workflow_run_id),
            AgentWorkflowFinalization,
        ),
    )
    if finalization.workflowRunId != workflow_run_id:
        raise ValueError("Agent workflow finalization identity does not match its path")
    if finalization.workspace != workspace:
        raise ValueError(
            "Agent workflow finalization workspace does not match its path"
        )
    return AgentWorkflowRun.model_validate(
        {
            **workflow.model_dump(mode="json"),
            "status": finalization.status,
            "finalizedAtNs": finalization.finalizedAtNs,
            "finalizationMessage": finalization.message,
        }
    )


def _report_checksum(record: AgentReportRecord) -> str:
    reference = record.reference.model_dump(
        mode="json",
        exclude={"contentSha256"},
    )
    content = {
        "recordType": record.recordType,
        "formatVersion": record.formatVersion,
        "reference": reference,
        "invocation": record.invocation.model_dump(mode="json"),
        "report": record.report,
    }
    return hashlib.sha256(record_io.canonical_json_bytes(content)).hexdigest()


def _load_report_record_at(
    group: zarr.Group,
    key: str,
    *,
    workflow_run_id: str,
    agent_name: AgentName,
    agent_run_id: str,
    workspace: str | None,
) -> AgentReportRecord:
    record = cast(
        AgentReportRecord,
        _read_json_model(group, key, AgentReportRecord),
    )
    reference = record.reference
    expected_type = cast(AgentReportType, _REPORT_TYPES[agent_name].__name__)
    if (
        reference.workflowRunId != workflow_run_id
        or reference.agentName != agent_name
        or reference.agentRunId != agent_run_id
    ):
        raise ValueError("Agent report identity does not match its path")
    if reference.workspace != workspace:
        raise ValueError("Agent report workspace does not match its path")
    if reference.reportType != expected_type:
        raise ValueError(
            f"Agent report type must be {expected_type!r} for {agent_name!r}"
        )
    if not reference.complete:
        raise ValueError("Atomic agent report records must be complete")
    if _report_checksum(record) != reference.contentSha256:
        raise ValueError(
            "Agent report content checksum does not match its JSON payload"
        )
    report_type = _REPORT_TYPES[agent_name]
    try:
        report = cast(AgentReport, report_type.model_validate(record.report))
    except ValueError as exc:
        raise ValueError(
            "Agent report payload does not match its Pydantic schema"
        ) from exc
    if report.runInfo.runId != reference.executionRunId:
        raise ValueError("Agent report execution ID does not match its JSON payload")
    return record


def _report_references(
    group: zarr.Group,
    prefix: str,
    workflow_run_id: str,
    workspace: str | None,
) -> list[AgentReportReference]:
    run_prefix = _workflow_prefix(prefix, workflow_run_id)
    references: list[AgentReportReference] = []
    for key in _list_keys(group, run_prefix):
        relative = key.removeprefix(f"{run_prefix}/")
        if relative in {"workflow.json", "finalization.json"}:
            continue
        parts = relative.split("/")
        if len(parts) != 3 or parts[2] != "report.json":
            raise ValueError(f"Unexpected agent workflow record {key!r}")
        raw_agent_name, agent_run_id, _filename = parts
        if raw_agent_name not in _REPORT_TYPES:
            raise ValueError(f"Unknown agent name {raw_agent_name!r}")
        agent_name = raw_agent_name
        _validate_run_id(agent_run_id, "agent_run_id")
        record = _load_report_record_at(
            group,
            key,
            workflow_run_id=workflow_run_id,
            agent_name=agent_name,
            agent_run_id=agent_run_id,
            workspace=workspace,
        )
        _validate_invocation(
            group,
            prefix,
            workflow_run_id,
            workspace,
            agent_name,
            record.invocation,
        )
        references.append(record.reference)
    return sorted(
        references,
        key=lambda item: (item.createdAtNs, item.agentName, item.agentRunId),
    )


def _load_parent_records(
    group: zarr.Group,
    prefix: str,
    workflow_run_id: str,
    workspace: str | None,
    invocation: AgentInvocation,
) -> dict[AgentName, list[AgentReportRecord]]:
    parents: dict[AgentName, list[AgentReportRecord]] = {
        name: [] for name in _REPORT_TYPES
    }
    for link in invocation.parentReports:
        if link.workflowRunId != workflow_run_id:
            raise ValueError("Parent reports must belong to the same workflow")
        if link.workspace != workspace:
            raise ValueError("Parent reports must belong to the same workspace")
        key = _report_key(
            prefix,
            workflow_run_id,
            link.agentName,
            link.agentRunId,
        )
        try:
            record = _load_report_record_at(
                group,
                key,
                workflow_run_id=workflow_run_id,
                agent_name=link.agentName,
                agent_run_id=link.agentRunId,
                workspace=workspace,
            )
        except FileNotFoundError as exc:
            raise ValueError(
                f"Unknown parent agent report {link.agentRunId!r}"
            ) from exc
        if AgentReportLink.from_reference(record.reference) != link:
            raise ValueError("Parent report link does not match the stored report")
        parents[link.agentName].append(record)
    return parents


def _has_handoff_parent(
    *,
    label: str,
    supplied: AgentDataModel | None,
    parent_records: list[AgentReportRecord],
) -> bool:
    if supplied is None and parent_records:
        raise ValueError(f"{label} is required when its parent report is cited")
    if supplied is not None and len(parent_records) != 1:
        raise ValueError(f"{label} requires exactly one matching parent report")
    return supplied is not None


def _validate_invocation(
    group: zarr.Group,
    prefix: str,
    workflow_run_id: str,
    workspace: str | None,
    agent_name: AgentName,
    invocation: AgentInvocation,
) -> None:
    if invocation.agentName != agent_name:
        raise ValueError("Invocation agentName does not match the report type")
    if not invocation.inputs:
        raise ValueError("Invocation inputs must record the agent call arguments")
    parents = _load_parent_records(
        group,
        prefix,
        workflow_run_id,
        workspace,
        invocation,
    )
    if agent_name == "parameter_tuning":
        experimental_parents = parents["experimental_context"]
        if _has_handoff_parent(
            label="experimentalTuningHandoff",
            supplied=invocation.experimentalTuningHandoff,
            parent_records=experimental_parents,
        ):
            expected = ExperimentalContextResult.model_validate(
                experimental_parents[0].report
            ).to_parameter_tuning_handoff()
            if invocation.experimentalTuningHandoff != expected:
                raise ValueError(
                    "experimentalTuningHandoff does not match the cited parent report"
                )
    elif invocation.experimentalTuningHandoff is not None:
        raise ValueError("experimentalTuningHandoff is only valid for parameter_tuning")

    if agent_name == "biological_interpretation":
        experimental_parents = parents["experimental_context"]
        if len(experimental_parents) > 1:
            raise ValueError(
                "Biological Interpretation accepts at most one Experimental "
                "Context parent report"
            )
        if invocation.experimentalBiologyHandoff is not None:
            if len(experimental_parents) != 1:
                raise ValueError(
                    "experimentalBiologyHandoff requires exactly one matching "
                    "parent report"
                )
            assert invocation.experimentalBiologyHandoff is not None
            expected_experimental = ExperimentalContextResult.model_validate(
                experimental_parents[0].report
            ).to_biological_handoff(
                invocation.experimentalBiologyHandoff.conditionColumn
            )
            if invocation.experimentalBiologyHandoff != expected_experimental:
                raise ValueError(
                    "experimentalBiologyHandoff does not match the cited parent report"
                )
        tuning_parents = parents["parameter_tuning"]
        if _has_handoff_parent(
            label="tuningBiologyHandoff",
            supplied=invocation.tuningBiologyHandoff,
            parent_records=tuning_parents,
        ):
            expected_tuning = ParameterTuningReport.model_validate(
                tuning_parents[0].report
            ).to_biological_handoff()
            if invocation.tuningBiologyHandoff != expected_tuning:
                raise ValueError(
                    "tuningBiologyHandoff does not match the cited parent report"
                )
    elif (
        invocation.experimentalBiologyHandoff is not None
        or invocation.tuningBiologyHandoff is not None
    ):
        raise ValueError(
            "Biology handoffs are only valid for biological_interpretation"
        )


def _resolved_workflow(
    group: zarr.Group,
    datastore: DataStore | None,
    prefix: str,
    workflow_run_id: str,
    workspace: str | None,
    *,
    ensure_fingerprints: bool,
) -> AgentWorkflowRun:
    workflow = _load_workflow_record(
        group,
        prefix,
        workflow_run_id,
        workspace,
    )
    observed = _live_dataset_fingerprints(
        group,
        datastore,
        ensure=ensure_fingerprints,
    )
    _validate_dataset_binding(workflow.datasetFingerprints, observed)
    return workflow


def create_agent_workflow(
    target: AgentPersistenceTarget,
    *,
    workflow_run_id: str | None = None,
    analysis_store: str = "",
    dataset_fingerprints: dict[str, str] | None = None,
    workspace: str | None = None,
) -> AgentWorkflowRun:
    """Create an immutable, dataset-bound workflow in the active data group."""
    if not isinstance(analysis_store, str):
        raise TypeError("analysis_store must be a string")
    group, datastore, resolved_workspace, inferred_store = _resolve_target(
        target,
        write=True,
        workspace=workspace,
    )
    observed = _live_dataset_fingerprints(group, datastore, ensure=True)
    if dataset_fingerprints is not None:
        if not isinstance(dataset_fingerprints, dict) or any(
            not isinstance(assay, str) or not isinstance(fingerprint, str)
            for assay, fingerprint in dataset_fingerprints.items()
        ):
            raise TypeError("dataset_fingerprints must map assay names to strings")
        supplied = dict(sorted(dataset_fingerprints.items()))
        _validate_dataset_binding(supplied, observed)
    resolved_run_id = _validate_run_id(
        uuid.uuid4().hex if workflow_run_id is None else workflow_run_id,
        "workflow_run_id",
    )
    prefix = _open_agent_store(
        group,
        workspace=resolved_workspace,
        initialize=True,
    )
    key = _workflow_key(prefix, resolved_run_id)
    if _list_keys(group, _workflow_prefix(prefix, resolved_run_id)):
        raise FileExistsError(f"Agent workflow {resolved_run_id!r} already exists")
    workflow = AgentWorkflowRun(
        workflowRunId=resolved_run_id,
        workspace=resolved_workspace,
        createdAtNs=time.time_ns(),
        analysisStore=analysis_store or inferred_store,
        datasetFingerprints=observed,
    )
    _write_key_once(
        group,
        key,
        record_io.display_json_bytes(workflow.model_dump(mode="json")),
    )
    logger.info(
        f"Created agent workflow {resolved_run_id}: workspace="
        f"{resolved_workspace or 'root'}, assays={len(observed)}"
    )
    return workflow


def save_agent_report(
    target: AgentPersistenceTarget,
    workflow_run_id: str,
    report: AgentReport,
    *,
    invocation: AgentInvocation,
    agent_run_id: str | None = None,
    workspace: str | None = None,
) -> AgentReportReference:
    """Persist one immutable report together with its replay-relevant inputs."""
    agent_name = _AGENT_NAMES.get(type(report))
    if agent_name is None:
        raise TypeError("report must be one of the four Scarf agent report models")
    if not isinstance(invocation, AgentInvocation):
        raise TypeError("invocation must be an AgentInvocation")
    resolved_workflow_id = _validate_run_id(workflow_run_id, "workflow_run_id")
    resolved_agent_run_id = _validate_run_id(
        uuid.uuid4().hex if agent_run_id is None else agent_run_id,
        "agent_run_id",
    )
    group, datastore, resolved_workspace, _analysis_store = _resolve_target(
        target,
        write=True,
        workspace=workspace,
    )
    prefix = _open_agent_store(
        group,
        workspace=resolved_workspace,
        initialize=False,
    )
    workflow = _resolved_workflow(
        group,
        datastore,
        prefix,
        resolved_workflow_id,
        resolved_workspace,
        ensure_fingerprints=True,
    )
    if workflow.status != "running":
        raise RuntimeError(
            f"Cannot save a report to a {workflow.status!r} agent workflow"
        )
    _validate_invocation(
        group,
        prefix,
        resolved_workflow_id,
        resolved_workspace,
        agent_name,
        invocation,
    )
    key = _report_key(
        prefix,
        resolved_workflow_id,
        agent_name,
        resolved_agent_run_id,
    )
    if _key_exists(group, key):
        raise FileExistsError(
            f"Agent report {resolved_agent_run_id!r} already exists for {agent_name!r}"
        )
    reference = AgentReportReference(
        workflowRunId=resolved_workflow_id,
        workspace=resolved_workspace,
        agentName=agent_name,
        agentRunId=resolved_agent_run_id,
        reportType=cast(AgentReportType, type(report).__name__),
        executionRunId=report.runInfo.runId,
        createdAtNs=time.time_ns(),
        complete=True,
        parentReports=list(invocation.parentReports),
        contentSha256="0" * 64,
    )
    record = AgentReportRecord(
        reference=reference,
        invocation=invocation,
        report=report.model_dump(mode="json"),
    )
    checksum = _report_checksum(record)
    reference = reference.model_copy(update={"contentSha256": checksum})
    record = record.model_copy(update={"reference": reference})
    _write_key_once(
        group,
        key,
        record_io.display_json_bytes(record.model_dump(mode="json")),
    )
    stored = _load_report_record_at(
        group,
        key,
        workflow_run_id=resolved_workflow_id,
        agent_name=agent_name,
        agent_run_id=resolved_agent_run_id,
        workspace=resolved_workspace,
    )
    logger.info(
        f"Saved {agent_name} report {resolved_agent_run_id} for workflow "
        f"{resolved_workflow_id}: status={getattr(report, 'status', 'done')}, "
        f"parents={len(invocation.parentReports)}"
    )
    return stored.reference


def _validate_supplied_reference(
    supplied: AgentReportReference,
    stored: AgentReportReference,
) -> None:
    if supplied.workflowRunId != stored.workflowRunId:
        raise ValueError(
            "Agent report reference workflow does not match stored metadata"
        )
    if supplied.workspace != stored.workspace:
        raise ValueError(
            "Agent report reference workspace does not match stored metadata"
        )
    if (
        supplied.agentName != stored.agentName
        or supplied.agentRunId != stored.agentRunId
    ):
        raise ValueError(
            "Agent report reference identity does not match stored metadata"
        )
    if supplied.reportType and supplied.reportType != stored.reportType:
        raise ValueError("Agent report reference type does not match stored metadata")
    if supplied.executionRunId and supplied.executionRunId != stored.executionRunId:
        raise ValueError("Agent report reference execution ID does not match metadata")
    if supplied.createdAtNs and supplied.createdAtNs != stored.createdAtNs:
        raise ValueError(
            "Agent report reference timestamp does not match stored metadata"
        )
    if supplied.parentReports and supplied.parentReports != stored.parentReports:
        raise ValueError("Agent report reference parents do not match stored metadata")
    if supplied.contentSha256 and supplied.contentSha256 != stored.contentSha256:
        raise ValueError(
            "Agent report reference checksum does not match stored metadata"
        )


def load_agent_record(
    target: AgentPersistenceTarget,
    reference: AgentReportReference,
    *,
    workspace: str | None = None,
) -> AgentReportRecord:
    """Load and validate a report envelope, including lineage and inputs."""
    group, datastore, resolved_workspace, _analysis_store = _resolve_target(
        target,
        write=False,
        workspace=workspace,
    )
    prefix = _open_agent_store(
        group,
        workspace=resolved_workspace,
        initialize=False,
    )
    _resolved_workflow(
        group,
        datastore,
        prefix,
        reference.workflowRunId,
        resolved_workspace,
        ensure_fingerprints=False,
    )
    key = _report_key(
        prefix,
        reference.workflowRunId,
        reference.agentName,
        reference.agentRunId,
    )
    try:
        record = _load_report_record_at(
            group,
            key,
            workflow_run_id=reference.workflowRunId,
            agent_name=reference.agentName,
            agent_run_id=reference.agentRunId,
            workspace=resolved_workspace,
        )
    except FileNotFoundError as exc:
        raise KeyError(f"Unknown agent report at {key!r}") from exc
    _validate_invocation(
        group,
        prefix,
        reference.workflowRunId,
        resolved_workspace,
        reference.agentName,
        record.invocation,
    )
    _validate_supplied_reference(reference, record.reference)
    return record


def load_agent_report(
    target: AgentPersistenceTarget,
    reference: AgentReportReference,
    *,
    workspace: str | None = None,
) -> AgentReport:
    """Load and strictly revalidate one persisted agent report."""
    record = load_agent_record(target, reference, workspace=workspace)
    report_type = _REPORT_TYPES[record.reference.agentName]
    return cast(AgentReport, report_type.model_validate(record.report))


def list_agent_reports(
    target: AgentPersistenceTarget,
    workflow_run_id: str,
    *,
    agent_name: AgentName | None = None,
    include_incomplete: bool = False,
    workspace: str | None = None,
) -> list[AgentReportReference]:
    """List atomic report references for one dataset-bound workflow."""
    del include_incomplete
    if agent_name is not None and agent_name not in _REPORT_TYPES:
        raise ValueError(f"Unknown agent name {agent_name!r}")
    group, datastore, resolved_workspace, _analysis_store = _resolve_target(
        target,
        write=False,
        workspace=workspace,
    )
    prefix = _open_agent_store(
        group,
        workspace=resolved_workspace,
        initialize=False,
    )
    _resolved_workflow(
        group,
        datastore,
        prefix,
        workflow_run_id,
        resolved_workspace,
        ensure_fingerprints=False,
    )
    references = _report_references(
        group,
        prefix,
        workflow_run_id,
        resolved_workspace,
    )
    if agent_name is not None:
        references = [item for item in references if item.agentName == agent_name]
    return references


def load_agent_workflow(
    target: AgentPersistenceTarget,
    workflow_run_id: str,
    *,
    include_incomplete: bool = False,
    workspace: str | None = None,
) -> AgentWorkflowRun:
    """Load one workflow, its lifecycle state, and all report references."""
    del include_incomplete
    group, datastore, resolved_workspace, _analysis_store = _resolve_target(
        target,
        write=False,
        workspace=workspace,
    )
    prefix = _open_agent_store(
        group,
        workspace=resolved_workspace,
        initialize=False,
    )
    workflow = _resolved_workflow(
        group,
        datastore,
        prefix,
        workflow_run_id,
        resolved_workspace,
        ensure_fingerprints=False,
    )
    reports = _report_references(
        group,
        prefix,
        workflow_run_id,
        resolved_workspace,
    )
    if workflow.status == "completed" and not reports:
        raise ValueError("A completed workflow must contain at least one report")
    return AgentWorkflowRun.model_validate(
        {
            **workflow.model_dump(mode="json"),
            "reports": reports,
        }
    )


def list_agent_workflows(
    target: AgentPersistenceTarget,
    *,
    include_incomplete: bool = False,
    workspace: str | None = None,
) -> list[AgentWorkflowRun]:
    """List terminal workflows, optionally including workflows still running."""
    group, datastore, resolved_workspace, _analysis_store = _resolve_target(
        target,
        write=False,
        workspace=workspace,
    )
    prefix = _open_agent_store(
        group,
        workspace=resolved_workspace,
        initialize=False,
    )
    runs_prefix = record_io.join_key(prefix, "runs")
    workflow_ids: set[str] = set()
    for key in _list_keys(group, runs_prefix):
        relative = key.removeprefix(f"{runs_prefix}/")
        parts = relative.split("/")
        if len(parts) < 2:
            raise ValueError(f"Unexpected agent workflow key {key!r}")
        workflow_ids.add(_validate_run_id(parts[0], "workflow_run_id"))
    workflows = [
        load_agent_workflow(
            target,
            workflow_run_id,
            workspace=workspace,
        )
        for workflow_run_id in sorted(workflow_ids)
    ]
    if not include_incomplete:
        workflows = [item for item in workflows if item.status != "running"]
    return sorted(
        workflows,
        key=lambda item: (item.createdAtNs, item.workflowRunId),
    )


def finalize_agent_workflow(
    target: AgentPersistenceTarget,
    workflow_run_id: str,
    *,
    status: AgentTerminalStatus,
    message: str = "",
    workspace: str | None = None,
) -> AgentWorkflowRun:
    """Write the one terminal event allowed for a running workflow."""
    if status not in {"completed", "failed", "abandoned"}:
        raise ValueError("status must be completed, failed, or abandoned")
    if not isinstance(message, str):
        raise TypeError("message must be a string")
    group, datastore, resolved_workspace, _analysis_store = _resolve_target(
        target,
        write=True,
        workspace=workspace,
    )
    prefix = _open_agent_store(
        group,
        workspace=resolved_workspace,
        initialize=False,
    )
    workflow = _resolved_workflow(
        group,
        datastore,
        prefix,
        workflow_run_id,
        resolved_workspace,
        ensure_fingerprints=True,
    )
    if workflow.status != "running":
        raise FileExistsError(
            f"Agent workflow {workflow_run_id!r} is already {workflow.status!r}"
        )
    reports = _report_references(
        group,
        prefix,
        workflow_run_id,
        resolved_workspace,
    )
    if status == "completed" and not reports:
        raise ValueError("A completed workflow must contain at least one report")
    finalization = AgentWorkflowFinalization(
        workflowRunId=workflow_run_id,
        workspace=resolved_workspace,
        status=status,
        finalizedAtNs=time.time_ns(),
        message=message,
    )
    _write_key_once(
        group,
        _finalization_key(prefix, workflow_run_id),
        record_io.display_json_bytes(finalization.model_dump(mode="json")),
    )
    finalized = load_agent_workflow(
        target,
        workflow_run_id,
        workspace=workspace,
    )
    logger.info(
        f"Finalized agent workflow {workflow_run_id}: status={status}, "
        f"reports={len(finalized.reports)}"
    )
    return finalized


__all__ = [
    "AgentInvocation",
    "AgentName",
    "AgentPersistenceTarget",
    "AgentReport",
    "AgentReportLink",
    "AgentReportRecord",
    "AgentReportReference",
    "AgentReportType",
    "AgentTerminalStatus",
    "AgentWorkflowRun",
    "AgentWorkflowStatus",
    "create_agent_workflow",
    "finalize_agent_workflow",
    "list_agent_reports",
    "list_agent_workflows",
    "load_agent_record",
    "load_agent_report",
    "load_agent_workflow",
    "save_agent_report",
]

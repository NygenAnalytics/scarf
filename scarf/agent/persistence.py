"""Immutable sidecar persistence for structured Scarf agent reports.

The sidecar format is deliberately separate from Scarf's analysis artifact
contract. Readers support exactly format version 1 and reject other versions;
there are no implicit migrations. The first implementation is for local,
single-writer use. A completed persistence record may contain an agent report
whose own status is ``done``, ``needsInput``, or ``failed``.
"""

import hashlib
import json
import re
import time
import uuid
from pathlib import Path
from typing import Literal, cast

import numpy as np
import zarr
from pydantic import Field, field_validator

from .biological_interpretation import BiologicalInterpretationReport
from .data_enrichment import DataEnrichmentReport
from .experimental_context import ExperimentalContextResult
from .parameter_tuning import ParameterTuningReport
from .types import AgentDataModel

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

_FORMAT = "scarf_agent_reports"
_FORMAT_VERSION = 1
_REPORT_ARRAY = "report_json"
_RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,127}$")
_REPORT_TYPES: dict[AgentName, type[AgentDataModel]] = {
    "data_enrichment": DataEnrichmentReport,
    "experimental_context": ExperimentalContextResult,
    "parameter_tuning": ParameterTuningReport,
    "biological_interpretation": BiologicalInterpretationReport,
}
_AGENT_NAMES: dict[type[AgentDataModel], AgentName] = {
    report_type: agent_name for agent_name, report_type in _REPORT_TYPES.items()
}


class AgentReportReference(AgentDataModel):
    """Stable identity for one immutable report in an agent sidecar."""

    type: Literal["agentReport"] = "agentReport"
    workflowRunId: str = ""
    agentName: AgentName = "data_enrichment"
    agentRunId: str = ""
    reportType: AgentReportType = ""
    executionRunId: str = ""
    createdAtNs: int = Field(default=0, ge=0, strict=True)
    complete: bool = Field(default=False, strict=True)

    @field_validator("workflowRunId", "agentRunId")
    @classmethod
    def validate_run_ids(cls, value: str) -> str:
        if value:
            _validate_run_id(value, "run ID")
        return value

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
        )


class AgentWorkflowRun(AgentDataModel):
    """One workflow and the report records currently stored beneath it."""

    type: Literal["agentWorkflowRun"] = "agentWorkflowRun"
    workflowRunId: str = ""
    createdAtNs: int = Field(default=0, ge=0, strict=True)
    analysisStore: str = ""
    datasetFingerprints: dict[str, str] = Field(default_factory=dict)
    reports: list[AgentReportReference] = Field(default_factory=list)

    @field_validator("workflowRunId")
    @classmethod
    def validate_workflow_run_id(cls, value: str) -> str:
        if value:
            _validate_run_id(value, "workflowRunId")
        return value

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


def _sidecar_path(path: str | Path) -> Path:
    if "://" in str(path):
        raise ValueError("Agent report persistence currently supports local paths only")
    sidecar_path = Path(path)
    if not sidecar_path.name.endswith(".agents.zarr"):
        raise ValueError("Agent report sidecars must end with '.agents.zarr'")
    if any(parent.name.endswith(".zarr") for parent in sidecar_path.parents):
        raise ValueError(
            "Agent report sidecars must not be placed inside another Zarr store"
        )
    return sidecar_path


def _validate_run_id(value: str, label: str) -> str:
    if _RUN_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{label} must be one safe path component containing 1-128 ASCII "
            "lowercase letters, numbers, underscores, or hyphens"
        )
    return value


def _validate_root(root: zarr.Group) -> None:
    if root.attrs.get("format") != _FORMAT:
        raise ValueError("Zarr store is not a Scarf agent-report sidecar")
    format_version = root.attrs.get("format_version")
    if type(format_version) is not int or format_version != _FORMAT_VERSION:
        raise ValueError(
            "Unsupported Scarf agent-report format version; only version 1 is "
            "supported and no automatic migration is performed"
        )


def _open_sidecar(
    path: str | Path,
    *,
    write: bool,
    initialize: bool = False,
) -> zarr.Group:
    sidecar_path = _sidecar_path(path)
    if not initialize and not sidecar_path.exists():
        raise FileNotFoundError(sidecar_path)
    if sidecar_path.exists():
        if not sidecar_path.is_dir():
            raise ValueError("Agent report sidecar path must be a directory")
        entries = list(sidecar_path.iterdir())
        if entries and not (sidecar_path / "zarr.json").is_file():
            raise ValueError(
                "Refusing to initialize or modify a non-Zarr directory as an agent "
                "sidecar"
            )
        if not initialize and not entries:
            raise ValueError("Zarr store is not a Scarf agent-report sidecar")
    if write:
        root = zarr.open_group(str(sidecar_path), mode="a", zarr_format=3)
    else:
        root = zarr.open_group(str(sidecar_path), mode="r")
    if "format" not in root.attrs:
        if not initialize:
            raise ValueError("Zarr store is not a Scarf agent-report sidecar")
        if list(root.group_keys()) or list(root.array_keys()) or dict(root.attrs):
            raise ValueError(
                "Refusing to initialize a non-empty Zarr store as an agent sidecar"
            )
        root.attrs.update(
            {
                "format": _FORMAT,
                "format_version": _FORMAT_VERSION,
            }
        )
    _validate_root(root)
    return root


def _workflow_group(root: zarr.Group, workflow_run_id: str) -> zarr.Group:
    workflow_run_id = _validate_run_id(workflow_run_id, "workflow_run_id")
    path = f"runs/{workflow_run_id}"
    if path not in root:
        raise KeyError(f"Unknown agent workflow {workflow_run_id!r}")
    node = root[path]
    if not isinstance(node, zarr.Group):
        raise TypeError(f"Agent workflow path {path!r} is not a Zarr group")
    if node.attrs.get("record_type") != "agent_workflow_run":
        raise ValueError(f"Malformed agent workflow record at {path!r}")
    if node.attrs.get("workflow_run_id") != workflow_run_id:
        raise ValueError(f"Agent workflow identity mismatch at {path!r}")
    return node


def _report_reference(
    group: zarr.Group,
    *,
    workflow_run_id: str,
    agent_name: AgentName,
    agent_run_id: str,
) -> AgentReportReference:
    attrs = group.attrs
    expected_type = cast(AgentReportType, _REPORT_TYPES[agent_name].__name__)
    if attrs.get("record_type") != "agent_report":
        raise ValueError("Malformed agent report record type")
    record_version = attrs.get("format_version")
    if type(record_version) is not int or record_version != _FORMAT_VERSION:
        raise ValueError("Unsupported agent report record version")
    if attrs.get("workflow_run_id") != workflow_run_id:
        raise ValueError("Agent report workflow identity mismatch")
    if attrs.get("agent_name") != agent_name:
        raise ValueError("Agent report name identity mismatch")
    if attrs.get("agent_run_id") != agent_run_id:
        raise ValueError("Agent report run identity mismatch")
    if attrs.get("report_type") != expected_type:
        raise ValueError(
            f"Agent report type must be {expected_type!r} for {agent_name!r}"
        )
    created_at_ns = attrs.get("created_at_ns")
    if type(created_at_ns) is not int or created_at_ns < 1:
        raise ValueError("Agent report created_at_ns is malformed")
    execution_run_id = attrs.get("execution_run_id", "")
    if not isinstance(execution_run_id, str):
        raise ValueError("Agent report execution_run_id is malformed")
    complete = attrs.get("complete")
    if not isinstance(complete, bool):
        raise ValueError("Agent report complete flag is malformed")
    payload_bytes = attrs.get("payload_bytes")
    payload_sha256 = attrs.get("payload_sha256")
    if type(payload_bytes) is not int or payload_bytes < 1:
        raise ValueError("Agent report payload_bytes is malformed")
    if (
        not isinstance(payload_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", payload_sha256) is None
    ):
        raise ValueError("Agent report payload_sha256 is malformed")
    return AgentReportReference(
        workflowRunId=workflow_run_id,
        agentName=agent_name,
        agentRunId=agent_run_id,
        reportType=expected_type,
        executionRunId=execution_run_id,
        createdAtNs=created_at_ns,
        complete=complete,
    )


def create_agent_workflow(
    path: str | Path,
    *,
    workflow_run_id: str | None = None,
    analysis_store: str = "",
    dataset_fingerprints: dict[str, str] | None = None,
) -> AgentWorkflowRun:
    """Create one immutable-identity workflow container in a sidecar store."""
    if not isinstance(analysis_store, str):
        raise TypeError("analysis_store must be a string")
    if dataset_fingerprints is None:
        resolved_fingerprints: dict[str, str] = {}
    elif not isinstance(dataset_fingerprints, dict) or any(
        not isinstance(assay, str) or not isinstance(fingerprint, str)
        for assay, fingerprint in dataset_fingerprints.items()
    ):
        raise TypeError("dataset_fingerprints must map assay names to strings")
    else:
        resolved_fingerprints = dict(dataset_fingerprints)
    resolved_run_id = _validate_run_id(
        uuid.uuid4().hex if workflow_run_id is None else workflow_run_id,
        "workflow_run_id",
    )
    root = _open_sidecar(path, write=True, initialize=True)
    runs = root["runs"] if "runs" in root else root.create_group("runs")
    if not isinstance(runs, zarr.Group):
        raise TypeError("Agent sidecar runs path is not a Zarr group")
    if resolved_run_id in runs:
        raise FileExistsError(f"Agent workflow {resolved_run_id!r} already exists")
    created_at_ns = time.time_ns()
    runs.create_group(
        resolved_run_id,
        attributes={
            "record_type": "agent_workflow_run",
            "workflow_run_id": resolved_run_id,
            "created_at_ns": created_at_ns,
            "analysis_store": analysis_store,
            "dataset_fingerprints": resolved_fingerprints,
        },
    )
    return AgentWorkflowRun(
        workflowRunId=resolved_run_id,
        createdAtNs=created_at_ns,
        analysisStore=analysis_store,
        datasetFingerprints=resolved_fingerprints,
    )


def save_agent_report(
    path: str | Path,
    workflow_run_id: str,
    report: AgentReport,
    *,
    agent_run_id: str | None = None,
) -> AgentReportReference:
    """Persist one report verbatim and return its sidecar reference.

    Existing report paths, including incomplete ones, are immutable and are never
    overwritten. The concrete report class determines the agent directory; model
    output fields are not trusted for routing.
    """
    agent_name = _AGENT_NAMES.get(type(report))
    if agent_name is None:
        raise TypeError("report must be one of the four Scarf agent report models")
    resolved_workflow_id = _validate_run_id(workflow_run_id, "workflow_run_id")
    resolved_agent_run_id = _validate_run_id(
        uuid.uuid4().hex if agent_run_id is None else agent_run_id,
        "agent_run_id",
    )
    root = _open_sidecar(path, write=True)
    workflow = _workflow_group(root, resolved_workflow_id)
    agent_group = (
        workflow[agent_name]
        if agent_name in workflow
        else workflow.create_group(agent_name)
    )
    if not isinstance(agent_group, zarr.Group):
        raise TypeError(f"Agent path {agent_name!r} is not a Zarr group")
    if resolved_agent_run_id in agent_group:
        raise FileExistsError(
            f"Agent report {resolved_agent_run_id!r} already exists for {agent_name!r}"
        )

    payload = json.dumps(
        report.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    payload_hash = hashlib.sha256(payload).hexdigest()
    created_at_ns = time.time_ns()
    group = agent_group.create_group(
        resolved_agent_run_id,
        attributes={
            "record_type": "agent_report",
            "format_version": _FORMAT_VERSION,
            "workflow_run_id": resolved_workflow_id,
            "agent_name": agent_name,
            "agent_run_id": resolved_agent_run_id,
            "report_type": type(report).__name__,
            "execution_run_id": report.runInfo.runId,
            "created_at_ns": created_at_ns,
            "payload_bytes": len(payload),
            "payload_sha256": payload_hash,
            "complete": False,
        },
    )
    values = np.frombuffer(payload, dtype=np.uint8).copy()
    payload_array = group.create_array(
        _REPORT_ARRAY,
        data=values,
        chunks=(min(len(values), 64 * 1024),),
    )
    stored_payload = np.asarray(payload_array[:])
    if (
        stored_payload.ndim != 1
        or stored_payload.dtype != np.dtype(np.uint8)
        or stored_payload.tobytes() != payload
    ):
        raise RuntimeError("Agent report payload verification failed")
    group.attrs["complete"] = True
    return _report_reference(
        group,
        workflow_run_id=resolved_workflow_id,
        agent_name=agent_name,
        agent_run_id=resolved_agent_run_id,
    )


def load_agent_report(
    path: str | Path,
    reference: AgentReportReference,
) -> AgentReport:
    """Load and strictly revalidate one complete persisted report."""
    root = _open_sidecar(path, write=False)
    workflow_run_id = _validate_run_id(reference.workflowRunId, "workflow_run_id")
    agent_run_id = _validate_run_id(reference.agentRunId, "agent_run_id")
    _workflow_group(root, workflow_run_id)
    report_path = f"runs/{workflow_run_id}/{reference.agentName}/{agent_run_id}"
    if report_path not in root:
        raise KeyError(f"Unknown agent report at {report_path!r}")
    node = root[report_path]
    if not isinstance(node, zarr.Group):
        raise TypeError(f"Agent report path {report_path!r} is not a Zarr group")
    stored_reference = _report_reference(
        node,
        workflow_run_id=workflow_run_id,
        agent_name=reference.agentName,
        agent_run_id=agent_run_id,
    )
    if reference.reportType and reference.reportType != stored_reference.reportType:
        raise ValueError("Agent report reference type does not match stored metadata")
    if (
        reference.executionRunId
        and reference.executionRunId != stored_reference.executionRunId
    ):
        raise ValueError("Agent report reference execution ID does not match metadata")
    if reference.createdAtNs and reference.createdAtNs != stored_reference.createdAtNs:
        raise ValueError(
            "Agent report reference timestamp does not match stored metadata"
        )
    if not stored_reference.complete:
        raise RuntimeError(f"Agent report at {report_path!r} is incomplete")
    if _REPORT_ARRAY not in node:
        raise ValueError("Complete agent report is missing its JSON payload")
    payload_node = node[_REPORT_ARRAY]
    if not isinstance(payload_node, zarr.Array):
        raise ValueError("Agent report JSON payload is not a Zarr array")
    raw_payload = np.asarray(payload_node[:])
    if raw_payload.ndim != 1 or raw_payload.dtype != np.dtype(np.uint8):
        raise ValueError(
            "Agent report JSON payload must be a one-dimensional uint8 array"
        )
    payload = raw_payload.tobytes()
    if node.attrs.get("payload_bytes") != len(payload):
        raise ValueError("Agent report payload length does not match stored metadata")
    if node.attrs.get("payload_sha256") != hashlib.sha256(payload).hexdigest():
        raise ValueError("Agent report payload checksum does not match stored metadata")
    try:
        decoded = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Agent report payload is not valid UTF-8") from exc
    report_type = _REPORT_TYPES[stored_reference.agentName]
    try:
        loaded = report_type.model_validate_json(decoded)
    except ValueError as exc:
        raise ValueError(
            "Agent report payload does not match its Pydantic schema"
        ) from exc
    loaded_report = cast(AgentReport, loaded)
    if loaded_report.runInfo.runId != stored_reference.executionRunId:
        raise ValueError("Agent report execution ID does not match its JSON payload")
    return loaded_report


def list_agent_reports(
    path: str | Path,
    workflow_run_id: str,
    *,
    agent_name: AgentName | None = None,
    include_incomplete: bool = False,
) -> list[AgentReportReference]:
    """List report references for one workflow in deterministic order."""
    root = _open_sidecar(path, write=False)
    workflow_run_id = _validate_run_id(workflow_run_id, "workflow_run_id")
    workflow = _workflow_group(root, workflow_run_id)
    if agent_name is not None and agent_name not in _REPORT_TYPES:
        raise ValueError(f"Unknown agent name {agent_name!r}")
    if list(workflow.array_keys()):
        raise ValueError("Agent workflow contains unexpected arrays")
    unknown_agent_groups = sorted(set(workflow.group_keys()) - set(_REPORT_TYPES))
    if unknown_agent_groups:
        raise ValueError(
            f"Agent workflow contains unknown agent groups {unknown_agent_groups}"
        )
    selected_names = [agent_name] if agent_name is not None else sorted(_REPORT_TYPES)
    references: list[AgentReportReference] = []
    for selected_name in selected_names:
        if selected_name not in workflow:
            continue
        agent_group = workflow[selected_name]
        if not isinstance(agent_group, zarr.Group):
            raise TypeError(f"Agent path {selected_name!r} is not a Zarr group")
        if list(agent_group.array_keys()):
            raise ValueError(f"Agent path {selected_name!r} contains unexpected arrays")
        for agent_run_id in sorted(agent_group.group_keys()):
            _validate_run_id(agent_run_id, "agent_run_id")
            report_group = agent_group[agent_run_id]
            if not isinstance(report_group, zarr.Group):
                raise TypeError("Agent report entry is not a Zarr group")
            complete = report_group.attrs.get("complete")
            if not include_incomplete and (complete is False or complete is None):
                continue
            reference = _report_reference(
                report_group,
                workflow_run_id=workflow_run_id,
                agent_name=selected_name,
                agent_run_id=agent_run_id,
            )
            references.append(reference)
    return sorted(
        references,
        key=lambda item: (item.createdAtNs, item.agentName, item.agentRunId),
    )


def load_agent_workflow(
    path: str | Path,
    workflow_run_id: str,
    *,
    include_incomplete: bool = False,
) -> AgentWorkflowRun:
    """Load one workflow manifest reconstructed from immutable report records."""
    root = _open_sidecar(path, write=False)
    workflow_run_id = _validate_run_id(workflow_run_id, "workflow_run_id")
    workflow = _workflow_group(root, workflow_run_id)
    created_at_ns = workflow.attrs.get("created_at_ns")
    if type(created_at_ns) is not int or created_at_ns < 1:
        raise ValueError("Agent workflow created_at_ns is malformed")
    analysis_store = workflow.attrs.get("analysis_store", "")
    dataset_fingerprints = workflow.attrs.get("dataset_fingerprints", {})
    if not isinstance(analysis_store, str) or not isinstance(
        dataset_fingerprints, dict
    ):
        raise ValueError("Agent workflow metadata is malformed")
    if any(
        not isinstance(assay, str) or not isinstance(fingerprint, str)
        for assay, fingerprint in dataset_fingerprints.items()
    ):
        raise ValueError("Agent workflow dataset fingerprints are malformed")
    return AgentWorkflowRun(
        workflowRunId=workflow_run_id,
        createdAtNs=created_at_ns,
        analysisStore=analysis_store,
        datasetFingerprints=dataset_fingerprints,
        reports=list_agent_reports(
            path,
            workflow_run_id,
            include_incomplete=include_incomplete,
        ),
    )


def list_agent_workflows(
    path: str | Path,
    *,
    include_incomplete: bool = False,
) -> list[AgentWorkflowRun]:
    """List all workflow manifests in deterministic order."""
    root = _open_sidecar(path, write=False)
    if list(root.array_keys()):
        raise ValueError("Agent sidecar root contains unexpected arrays")
    unknown_root_groups = sorted(set(root.group_keys()) - {"runs"})
    if unknown_root_groups:
        raise ValueError(
            f"Agent sidecar root contains unknown groups {unknown_root_groups}"
        )
    if "runs" not in root:
        return []
    runs = root["runs"]
    if not isinstance(runs, zarr.Group):
        raise TypeError("Agent sidecar runs path is not a Zarr group")
    if list(runs.array_keys()):
        raise ValueError("Agent sidecar runs path contains unexpected arrays")
    workflows = [
        load_agent_workflow(
            path,
            workflow_run_id,
            include_incomplete=include_incomplete,
        )
        for workflow_run_id in sorted(runs.group_keys())
    ]
    return sorted(
        workflows,
        key=lambda item: (item.createdAtNs, item.workflowRunId),
    )


__all__ = [
    "AgentName",
    "AgentReport",
    "AgentReportReference",
    "AgentReportType",
    "AgentWorkflowRun",
    "create_agent_workflow",
    "list_agent_reports",
    "list_agent_workflows",
    "load_agent_report",
    "load_agent_workflow",
    "save_agent_report",
]

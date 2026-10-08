"""One immutable result summary in the local store, separate from core artifacts."""

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import zarr

from .records import RecordError, digest
from .rendering import stage_data

if TYPE_CHECKING:
    from .result import AnalysisRun

_NAMESPACE = "agent_results"
_OWNER = "scarf.agent"


def _local_root(result: "AnalysisRun", *, writable: bool) -> zarr.Group:
    namespace = result.source / _NAMESPACE
    child = namespace / _run_id(result)
    if namespace.is_symlink() or child.is_symlink():
        raise RecordError("Agent result paths cannot be filesystem symlinks")
    # A Path, because Zarr reads a string as a URL and would end a local path
    # at '#', '?', or ';'.
    root = zarr.open_group(
        Path(result.source), mode="r+" if writable else "r", use_consolidated=False
    )
    if _NAMESPACE not in root:
        if namespace.exists():
            raise RecordError("agent_results collides with an unowned filesystem path")
    else:
        group = root[_NAMESPACE]
        if isinstance(group, zarr.Group) and child.name not in group and child.exists():
            raise RecordError(
                "The agent result collides with an unowned filesystem path"
            )
    return root


def _namespace(root: zarr.Group, *, create: bool) -> zarr.Group | None:
    if _NAMESPACE not in root:
        return (
            root.create_group(_NAMESPACE, attributes={"owner": _OWNER})
            if create
            else None
        )
    group = root[_NAMESPACE]
    if (
        not isinstance(group, zarr.Group)
        or dict(group.attrs) != {"owner": _OWNER}
        or "cellData" in group
        or "matrices" in group
    ):
        raise RecordError("agent_results collides with an unowned store namespace")
    return group


def _run_id(result: "AnalysisRun") -> str:
    identifier = result._records.manifest["runId"]
    if not isinstance(identifier, str) or not re.fullmatch(
        r"[A-Za-z0-9_-]+", identifier
    ):
        raise RecordError("The agent runId is not a valid local result group name")
    return identifier


def _payload(result: "AnalysisRun") -> dict[str, Any]:
    """Read only verified core lineage and the final, accepted agent selection."""
    if result.status != "completed":
        raise RecordError("Only completed analyses can publish compact results")
    records = result._records
    manifest = records.manifest
    store, run = result._bound_pipeline()
    final = stage_data(records, "finalize")
    prepared = stage_data(records, "preprocess")
    exploration = stage_data(records, "explore")
    if not final or not prepared or not exploration:
        raise RecordError("Completed analysis is missing frozen selection evidence")
    if prepared["assay"] != run.assay or final["runId"] != run.run_id:
        raise RecordError("Compact result does not match the verified final pipeline")
    config = run.report()["run"]["config"]
    partition = exploration["partitions"][final["selected"]]
    candidate_id = partition["candidateId"]
    if config["leiden"]["selected"] != partition["resolution"]:
        raise RecordError(
            "Final pipeline resolution differs from the accepted selection"
        )
    selected: dict[str, Any] = {
        "pipelineConfig": config,
        "candidateId": candidate_id,
        "resolution": config["leiden"]["selected"],
        "requestedHvgCount": config["hvgCount"],
        "pcaDims": config["pcaDims"],
        "neighborsK": config["neighborsK"],
        "useHarmony": bool(config["harmonyBatchColumns"]),
    }
    for measured in exploration["summaries"]:
        if measured["candidateId"] == candidate_id and "actualHvgCount" in measured:
            selected["actualHvgCount"] = measured["actualHvgCount"]
            break
    accepted = [
        event
        for event in records.events()
        if event["kind"] == "decisionAccepted" and event.get("stage") == "finalists"
    ]
    if not accepted:
        raise RecordError("Completed analysis is missing its final selection decision")
    rationale = accepted[-1]["output"]["rationale"]
    for event in records.events():
        if (
            event["kind"] == "decisionResolved"
            and event["acceptedSequence"] == accepted[-1]["sequence"]
        ):
            rationale += f" Automatic resolution: {event['rule']}."
    return {
        "agentRunId": _run_id(result),
        "finalPipelineRunId": run.run_id,
        "assay": run.assay,
        "workspace": store.workspace,
        "sourceFingerprint": prepared["fingerprint"],
        "selectedParameters": selected,
        "selectionRationale": rationale,
        "externalRunLocator": os.path.relpath(result.run_dir, result.source),
        "procedureIdentity": manifest["procedureIdentity"],
    }


def _stored(group: zarr.Group, identifier: str) -> dict[str, Any] | None:
    if identifier not in group:
        return None
    child = group[identifier]
    if not isinstance(child, zarr.Group) or set(child.attrs) != {"result"}:
        raise RecordError("The agent result group collides with existing store data")
    payload = child.attrs["result"]
    if not isinstance(payload, dict):
        raise RecordError("The stored compact result is not a JSON object")
    return payload


def _verify(stored: dict[str, Any], expected: dict[str, Any]) -> None:
    # The original locator is advisory. Moving the store or history never
    # authorizes replacing an immutable scientific result.
    locator = stored.get("externalRunLocator")
    if not isinstance(locator, str) or not locator or os.path.isabs(locator):
        raise RecordError("Stored compact result has no valid relative history locator")
    if {key: value for key, value in stored.items() if key != "externalRunLocator"} != {
        key: value for key, value in expected.items() if key != "externalRunLocator"
    }:
        raise RecordError(
            "Stored compact result differs from verified analysis identity"
        )


def publish_result(result: "AnalysisRun") -> None:
    """Publish under existing agent run/source locks; recover identical writes."""
    payload = _payload(result)
    records = result._records
    supplied = records.latest("inputsResolved") or records.manifest
    if (
        payload["assay"] == _NAMESPACE
        or supplied["config"].get("workspace") == _NAMESPACE
    ):
        raise RecordError("agent_results is already the selected assay or workspace")
    # Open the actual local root, never a mounted source's overlay or count store.
    root = _local_root(result, writable=True)
    previous = records.latest("resultPublished")
    group = _namespace(root, create=False)
    if group is None:
        if previous is not None:
            raise RecordError("Previously published compact result is missing")
        group = _namespace(root, create=True)
    assert group is not None
    identifier = payload["agentRunId"]
    stored = _stored(group, identifier)
    recovered = stored is not None
    if stored is None:
        if previous is not None:
            raise RecordError("Previously published compact result is missing")
        group.create_group(identifier, attributes={"result": payload})
        stored = payload
    else:
        _verify(stored, payload)
    result_path = f"{_NAMESPACE}/{identifier}"
    publication = {"resultPath": result_path, "resultDigest": digest(stored)}
    if previous is None:
        records.append("resultPublished", **publication, recovered=recovered)
    elif any(previous.get(key) != value for key, value in publication.items()):
        raise RecordError("Recorded compact publication differs from the stored result")
    if stored["externalRunLocator"] != payload["externalRunLocator"]:
        locator = {
            "resultPath": result_path,
            "storedLocator": stored["externalRunLocator"],
            "currentLocator": payload["externalRunLocator"],
        }
        previous = records.latest("resultLocatorStale")
        if previous is None or any(
            previous.get(key) != value for key, value in locator.items()
        ):
            records.append("resultLocatorStale", **locator)


def read_result(result: "AnalysisRun") -> dict[str, Any] | None:
    """Read and verify an existing compact result without creating or updating it."""
    root = _local_root(result, writable=False)
    group = _namespace(root, create=False)
    if group is None:
        return None
    stored = _stored(group, _run_id(result))
    if stored is not None:
        _verify(stored, _payload(result))
        previous = result._records.latest("resultPublished")
        if previous is not None and previous["resultDigest"] != digest(stored):
            raise RecordError(
                "Stored compact result differs from its publication record"
            )
    return stored

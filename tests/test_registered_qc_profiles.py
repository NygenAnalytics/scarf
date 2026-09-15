"""Tests for registered one-sided cell-quality profiles."""

import asyncio
from copy import deepcopy
from typing import Any

import numpy as np
import pytest
import zarr
from pydantic import ValidationError
from zarr.storage import MemoryStore

import scarf.agent.orchestrator.preprocessing as preprocessing_module
from scarf.agent.cell_quality.execution import (
    execute_auto_cell_qc,
    execute_registered_cell_qc,
)
from scarf.agent.cell_quality.profiles import (
    RegisteredQcProjection,
    offered_registered_qc_profiles,
    project_auto_filter_profile,
    project_registered_qc_profile,
    qc_metric_execution_name,
)
from scarf.agent.experimental_context import (
    CellQcPlan,
    CellQcProfileEvidence,
    ExperimentalContextResult,
    inspect_cell_covariates,
)
from scarf.agent.orchestrator.main import AgentOrchestrator
from scarf.agent.types import ArtifactReferenceModel
from scarf.datastore._operations.quality_control import _QualityControlOperationsMixin
from scarf.metadata.artifacts import (
    plan_cell_data_artifact,
    write_cell_data_artifact,
)
from scarf.metadata.selection import NamedCellArtifact
from scarf.storage.artifacts import (
    ArtifactRef,
    artifact_group,
    inspect_artifact,
)
from scarf.storage.selections import (
    read_stored_selection_mask,
    resolve_selection_artifact,
)
from tests.test_agent_experimental_context import _Cells, _Store, _context


def _quality_values() -> dict[str, np.ndarray]:
    return {
        "RNA_nCounts": np.concatenate(
            [np.linspace(80, 120, 39), np.asarray([1.0, 1_000.0, 100.0])]
        ),
        "RNA_nFeatures": np.concatenate(
            [np.linspace(40, 60, 39), np.asarray([1.0, 500.0, 50.0])]
        ),
        "RNA_percentMito": np.concatenate([np.linspace(1, 5, 41), np.asarray([40.0])]),
        "RNA_percentRibo": np.linspace(5, 25, 42),
    }


def _profile_parameters(
    projection: RegisteredQcProjection,
) -> dict[str, Any]:
    return {
        "policyVersion": 1,
        "profile": projection.profile,
        "nMads": 3.0 if projection.profile == "captureMad3Sensitivity" else 5.0,
        "boundPolicy": {
            "count": {"remove": "lower", "flag": "upper"},
            "feature": {"remove": "lower", "flag": "upper"},
            "mitochondrial": {"remove": "upper", "fixedCutoff": None},
            "diagnostic": {"remove": "none"},
        },
        "resolvedBounds": [threshold.to_dict() for threshold in projection.thresholds],
        "captureSizes": projection.captureSizes,
        "captureComparisons": [
            comparison.to_dict() for comparison in projection.captureComparisons
        ],
        "captureComparisonSource": None,
        "pooledReferenceCaptures": [],
    }


class _MemoryQcCells:
    def __init__(self, group: zarr.Group) -> None:
        self._group = group
        self.N = int(group["ids"].shape[0])

    @property
    def columns(self) -> list[str]:
        return list(self._group.array_keys())

    def _get_array(self, column: str):
        return self._group[column]

    def fetch_all(self, column: str) -> np.ndarray:
        return np.asarray(self._group[column][:])


class _MemoryQcStore(_QualityControlOperationsMixin):
    def __init__(self, root: zarr.Group) -> None:
        self.zw = root
        self.cells = _MemoryQcCells(root["cellData"])

    def snapshot_cell_selection(self, column: str = "I") -> ArtifactRef:
        values = np.asarray(self.cells.fetch_all(column), dtype=bool)
        return resolve_selection_artifact(
            self.zw,
            scope="datastore",
            kind="cell_selection",
            values=values,
            row_ids=self.cells.fetch_all("ids"),
            operation="snapshot_cell_selection",
            parameters={"column": column},
            inputs={},
            source_column=column,
        )

    def inspect_artifact(self, ref: ArtifactRef):
        return inspect_artifact(self.zw, ref)


def _memory_qc_store(
    values_by_metric: dict[str, np.ndarray],
) -> tuple[_MemoryQcStore, ArtifactRef]:
    first = next(iter(values_by_metric.values()))
    n_cells = len(first)
    if any(len(values) != n_cells for values in values_by_metric.values()):
        raise ValueError("Memory QC metrics must have equal lengths")
    root = zarr.open_group(store=MemoryStore(), mode="w")
    cell_data = root.create_group("cellData")
    cell_data.create_array(
        "ids",
        data=np.asarray([f"cell-{index}" for index in range(n_cells)]),
    )
    cell_data.create_array("I", data=np.ones(n_cells, dtype=bool))
    for name, values in values_by_metric.items():
        cell_data.create_array(name, data=np.asarray(values))
    store = _MemoryQcStore(root)
    return store, store.snapshot_cell_selection("I")


def _write_memory_cell_artifact(
    store: _MemoryQcStore,
    *,
    selection: ArtifactRef,
    name: str,
    kind: str,
    values: np.ndarray,
    assay: str,
) -> NamedCellArtifact:
    resolved = np.asarray(values)
    planned = plan_cell_data_artifact(
        store.zw,
        scope="assay",
        assay=assay,
        kind=kind,
        operation=f"test_{kind}",
        parameters={"name": name},
        inputs={},
        execution_options={},
        cell_selection=selection,
        arrays={"values": (resolved.shape, None)},
    )
    write_cell_data_artifact(store.zw, planned, {"values": resolved})
    return NamedCellArtifact(name=name, artifact=planned.ref)


def _registered_execution_case() -> tuple[
    _MemoryQcStore,
    ArtifactRef,
    RegisteredQcProjection,
    dict[str, Any],
]:
    values = np.linspace(80.0, 120.0, 42)
    store, selection = _memory_qc_store(
        {
            "RNA_nCounts": values,
            "capture": np.asarray(["a"] * 21 + ["b"] * 21),
        }
    )
    projection = project_registered_qc_profile(
        "globalMad5",
        values_by_metric={"RNA_nCounts": values},
        active=np.ones(42, dtype=bool),
    )
    return (
        store,
        selection,
        projection,
        {
            "profile_parameters": _profile_parameters(projection),
            "expected_active_cells": 42,
            "expected_retained_cells": projection.retainedCells,
            "expected_flag_counts": projection.flagCounts,
            "attrs": ["RNA_nCounts"],
            "cell_selection": selection,
        },
    )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("unknownProfile", "Unknown registered"),
        ("invalidActive", "positive integer"),
        ("invalidRetained", "non-negative integer"),
        ("invalidFlagCount", "non-empty names"),
        ("parameterInventory", "do not match policy version"),
        ("policyVersion", "policyVersion must be 1"),
        ("profileMismatch", "profile and parameters disagree"),
        ("nMads", "requires nMads"),
        ("boundPolicy", "boundPolicy is not supported"),
        ("resolvedBoundsType", "resolvedBounds must be a list"),
        ("pooledReferencesType", "must be a list of strings"),
        ("attributeType", "attrs must contain only column names"),
        ("artifactType", "NamedCellArtifact"),
        ("artifactKind", "quality_metric"),
        ("duplicateArtifactName", "unique semantic names"),
        ("unexpectedCapture", "cannot use a capture source"),
        ("missingAttribute", "not found"),
        ("activeMismatch", "active-cell count differs"),
        ("nonfiniteMetadata", "non-finite entries"),
        ("retainedMismatch", "retained-cell count differs"),
        ("flagMismatch", "diagnostic-flag counts differ"),
    ],
)
def test_registered_qc_execution_rejects_inconsistent_evidence(
    case: str,
    message: str,
) -> None:
    store, selection, projection, arguments = _registered_execution_case()
    profile = "globalMad5"
    parameters = deepcopy(arguments["profile_parameters"])
    arguments["profile_parameters"] = parameters

    if case == "unknownProfile":
        profile = "invented"
    elif case == "invalidActive":
        arguments["expected_active_cells"] = False
    elif case == "invalidRetained":
        arguments["expected_retained_cells"] = -1
    elif case == "invalidFlagCount":
        arguments["expected_flag_counts"] = {"": 0}
    elif case == "parameterInventory":
        parameters.pop("captureSizes")
    elif case == "policyVersion":
        parameters["policyVersion"] = 2
    elif case == "profileMismatch":
        parameters["profile"] = "retainWithFlags"
    elif case == "nMads":
        parameters["nMads"] = 4.0
    elif case == "boundPolicy":
        parameters["boundPolicy"] = {}
    elif case == "resolvedBoundsType":
        parameters["resolvedBounds"] = {}
    elif case == "pooledReferencesType":
        parameters["pooledReferenceCaptures"] = [1]
    elif case == "attributeType":
        arguments["attrs"] = [1]
    elif case == "artifactType":
        arguments["artifact_metrics"] = [object()]
    elif case == "artifactKind":
        arguments["artifact_metrics"] = [
            NamedCellArtifact(name="metric", artifact=selection)
        ]
    elif case == "duplicateArtifactName":
        metric = _write_memory_cell_artifact(
            store,
            selection=selection,
            name="metric",
            kind="quality_metric",
            values=np.linspace(1.0, 2.0, 42),
            assay="RNA",
        )
        arguments["artifact_metrics"] = [metric, metric]
    elif case == "unexpectedCapture":
        arguments["sample_column"] = "capture"
    elif case == "missingAttribute":
        arguments["attrs"] = ["missing"]
    elif case == "activeMismatch":
        arguments["expected_active_cells"] = 41
    elif case == "nonfiniteMetadata":
        store.cells._get_array("RNA_nCounts")[0] = np.nan
    elif case == "retainedMismatch":
        arguments["expected_retained_cells"] = projection.retainedCells - 1
    elif case == "flagMismatch":
        arguments["expected_flag_counts"] = {"invented": 1}

    with pytest.raises((KeyError, TypeError, ValueError), match=message):
        execute_registered_cell_qc(store, profile, **arguments)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("unknownAction", "Unknown automatic"),
        ("attributeType", "attrs must contain only column names"),
        ("sampleSources", "sample_column and sample_artifact"),
        ("captureSources", "capture_column and capture_artifact"),
        ("missingSampleSource", "requires exactly one sample source"),
        ("unexpectedSample", "cannot use a core sample source"),
        ("missingGrouping", "grouping column"),
        ("missingAttribute", "not found"),
        ("invalidActive", "positive integer"),
        ("invalidRetained", "non-negative integer"),
        ("invalidFlagCount", "non-empty names"),
        ("activeMismatch", "active-cell count differs"),
        ("parameterMismatch", "parameters do not match"),
        ("boundsMismatch", "resolved bounds differ"),
        ("retainedMismatch", "retained-cell count differs"),
        ("flagMismatch", "flag counts differ"),
    ],
)
def test_auto_qc_execution_rejects_inconsistent_evidence(
    case: str,
    message: str,
) -> None:
    values = np.linspace(80.0, 120.0, 42)
    labels = np.asarray(["a"] * 21 + ["b"] * 21)
    store, selection = _memory_qc_store({"RNA_nCounts": values, "capture": labels})
    projection = project_auto_filter_profile(
        "globalGaussian",
        values_by_metric={"RNA_nCounts": values},
        active=np.ones(42, dtype=bool),
    )
    action = "globalGaussian"
    arguments: dict[str, Any] = {
        "profile_parameters": deepcopy(projection.parameters),
        "expected_active_cells": 42,
        "expected_retained_cells": projection.retainedCells,
        "expected_flag_counts": projection.flagCounts,
        "expected_resolved_bounds": deepcopy(projection.parameters["resolvedBounds"]),
        "attrs": ["RNA_nCounts"],
        "cell_selection": selection,
    }
    identity = _write_memory_cell_artifact(
        store,
        selection=selection,
        name="capture",
        kind="hto_identity",
        values=labels,
        assay="HTO",
    )

    if case == "unknownAction":
        action = "invented"
    elif case == "attributeType":
        arguments["attrs"] = [1]
    elif case == "sampleSources":
        arguments["sample_column"] = "capture"
        arguments["sample_artifact"] = identity
    elif case == "captureSources":
        arguments["capture_column"] = "capture"
        arguments["capture_artifact"] = identity
    elif case == "missingSampleSource":
        action = "sampleMad"
    elif case == "unexpectedSample":
        arguments["sample_column"] = "capture"
    elif case == "missingGrouping":
        arguments["capture_column"] = "missing"
    elif case == "missingAttribute":
        arguments["attrs"] = ["missing"]
    elif case == "invalidActive":
        arguments["expected_active_cells"] = False
    elif case == "invalidRetained":
        arguments["expected_retained_cells"] = -1
    elif case == "invalidFlagCount":
        arguments["expected_flag_counts"] = {"": 0}
    elif case == "activeMismatch":
        arguments["expected_active_cells"] = 41
    elif case == "parameterMismatch":
        arguments["profile_parameters"]["minP"] = 0.02
    elif case == "boundsMismatch":
        arguments["expected_resolved_bounds"] = {}
    elif case == "retainedMismatch":
        arguments["expected_retained_cells"] = projection.retainedCells - 1
    elif case == "flagMismatch":
        arguments["expected_flag_counts"] = {"invented": 1}

    with pytest.raises((KeyError, TypeError, ValueError), match=message):
        execute_auto_cell_qc(store, action, **arguments)


def test_global_registered_profile_uses_one_sided_data_derived_bounds() -> None:
    values = _quality_values()
    projection = project_registered_qc_profile(
        "globalMad5",
        values_by_metric=values,
        active=np.ones(42, dtype=bool),
    )

    assert projection.keep[39] == np.False_
    assert projection.keep[40] == np.True_
    assert projection.keep[41] == np.False_
    assert projection.flags["RNA_nCounts:high"][40] == np.True_
    assert projection.flags["RNA_nFeatures:high"][40] == np.True_
    assert "RNA_percentRibo:highMito" not in projection.flags
    assert not any(
        threshold.metric == "RNA_percentRibo" for threshold in projection.thresholds
    )

    count_threshold = next(
        threshold
        for threshold in projection.thresholds
        if threshold.metric == "RNA_nCounts"
    )
    mito_threshold = next(
        threshold
        for threshold in projection.thresholds
        if threshold.metric == "RNA_percentMito"
    )
    assert count_threshold.lowerRemoval is not None
    assert count_threshold.upperRemoval is None
    assert count_threshold.upperFlag is not None
    assert mito_threshold.lowerRemoval is None
    assert mito_threshold.upperRemoval is not None
    assert mito_threshold.upperRemoval != pytest.approx(8.0)


def test_retain_with_flags_never_removes_flagged_cells() -> None:
    projection = project_registered_qc_profile(
        "retainWithFlags",
        values_by_metric=_quality_values(),
        active=np.ones(42, dtype=bool),
    )

    assert projection.retainedCells == 42
    assert projection.keep.all()
    assert projection.flags["RNA_nCounts:lowQuality"][39] == np.True_
    assert projection.flags["RNA_nCounts:high"][40] == np.True_
    assert projection.flags["RNA_percentMito:highMito"][41] == np.True_


def test_capture_profiles_require_proof_and_minimum_capture_size() -> None:
    values = _quality_values()
    active = np.ones(42, dtype=bool)
    labels = np.asarray(["a"] * 21 + ["b"] * 21)

    unproven = offered_registered_qc_profiles(
        values_by_metric=values,
        active=active,
        capture_labels=labels,
        grouping_proven=False,
    )
    proven = offered_registered_qc_profiles(
        values_by_metric=values,
        active=active,
        capture_labels=labels,
        grouping_proven=True,
        min_cells_per_capture=20,
    )
    undersized = offered_registered_qc_profiles(
        values_by_metric=values,
        active=active,
        capture_labels=np.asarray(["a"] * 19 + ["b"] * 23),
        grouping_proven=True,
        min_cells_per_capture=20,
    )

    assert [projection.profile for projection in unproven] == [
        "retainWithFlags",
        "globalMad5",
    ]
    assert [projection.profile for projection in proven] == [
        "retainWithFlags",
        "globalMad5",
        "captureMad5",
        "captureMad3Sensitivity",
    ]
    global_projection = next(
        projection for projection in proven if projection.profile == "globalMad5"
    )
    assert {item.capture for item in global_projection.captureComparisons} == {
        "a",
        "b",
    }
    assert [projection.profile for projection in undersized] == [
        "retainWithFlags",
        "globalMad5",
    ]


def test_capture_profiles_surface_adverse_global_capture_comparison() -> None:
    labels = np.asarray(["good-a"] * 20 + ["good-b"] * 20 + ["failed"] * 20)
    values = {
        "RNA_nCounts": np.concatenate(
            [
                np.linspace(90, 110, 20),
                np.linspace(95, 115, 20),
                np.linspace(3, 7, 20),
            ]
        ),
        "RNA_nFeatures": np.concatenate(
            [
                np.linspace(45, 55, 20),
                np.linspace(48, 58, 20),
                np.linspace(2, 6, 20),
            ]
        ),
        "RNA_percentMito": np.concatenate(
            [np.linspace(1, 3, 20), np.linspace(1, 4, 20), np.linspace(25, 35, 20)]
        ),
    }

    projection = project_registered_qc_profile(
        "captureMad5",
        values_by_metric=values,
        active=np.ones(60, dtype=bool),
        capture_labels=labels,
        grouping_proven=True,
        min_cells_per_capture=20,
    )

    assert projection.captureSizes == {"good-a": 20, "good-b": 20, "failed": 20}
    assert "failed" in projection.failedCaptureCandidates
    failed = next(
        comparison
        for comparison in projection.captureComparisons
        if comparison.capture == "failed"
    )
    assert failed.adverseGlobalOutlier is True
    assert len(failed.reasons) >= 2
    assert projection.retainedByCapture["failed"] > 0


def test_pooled_reference_profile_requires_explicit_eligible_captures() -> None:
    values = _quality_values()
    active = np.ones(42, dtype=bool)
    labels = np.asarray(["small-a"] * 10 + ["small-b"] * 10 + ["large"] * 22)

    without_reference = offered_registered_qc_profiles(
        values_by_metric=values,
        active=active,
        capture_labels=labels,
        grouping_proven=True,
        min_cells_per_capture=20,
    )
    with_reference = offered_registered_qc_profiles(
        values_by_metric=values,
        active=active,
        capture_labels=labels,
        grouping_proven=True,
        min_cells_per_capture=20,
        pooled_reference_captures=("small-a", "small-b"),
    )

    assert "pooledReferenceMad5" not in {
        projection.profile for projection in without_reference
    }
    pooled = next(
        projection
        for projection in with_reference
        if projection.profile == "pooledReferenceMad5"
    )
    assert {threshold.group for threshold in pooled.thresholds} == {"pooledReference"}


def test_experimental_context_offers_registered_global_profile() -> None:
    store = _Store()
    store.cells._values["RNA_nCounts"] = np.linspace(10, 100, 12)
    store.cells._values["RNA_nFeatures"] = np.linspace(5, 50, 12)
    context = _context(
        store,
        directions={"cellQc": {"registeredProfile": "globalMad5"}},
    )

    inspected = asyncio.run(inspect_cell_covariates(context))
    offered = {
        profile.registeredProfile: profile
        for profile in inspected.qcProfiles
        if profile.registeredProfile is not None
    }

    assert {"retainWithFlags", "globalMad5"}.issubset(offered)
    assert offered["retainWithFlags"].action == "skip"
    assert offered["globalMad5"].action == "registeredMad"
    assert offered["globalMad5"].parameters["boundPolicy"]["count"] == {
        "remove": "lower",
        "flag": "upper",
    }
    assert offered["globalMad5"].parameters["boundPolicy"]["mitochondrial"] == {
        "remove": "upper",
        "fixedCutoff": None,
    }

    with pytest.raises(ValidationError, match="must use the registeredMad action"):
        CellQcPlan(
            action="globalGaussian",
            registeredProfile="globalMad5",
            profileId=offered["globalMad5"].profileId,
            attributes=offered["globalMad5"].attributes,
        )


def test_experimental_context_does_not_infer_capture_from_observation_unit() -> None:
    store = _Store()
    store.cells._values["RNA_nCounts"] = np.linspace(10, 100, 12)
    store.cells._values["RNA_nFeatures"] = np.linspace(5, 50, 12)
    context = _context(store)

    inspected = asyncio.run(inspect_cell_covariates(context))

    registered = {
        profile.registeredProfile
        for profile in inspected.qcProfiles
        if profile.registeredProfile is not None
    }
    assert registered == {"retainWithFlags", "globalMad5"}


def test_experimental_context_offers_capture_profiles_only_for_explicit_source() -> (
    None
):
    store = _Store()
    n_cells = 60
    store.cells = _Cells(
        {
            "I": np.ones(n_cells, dtype=bool),
            "ids": np.asarray([f"cell-{index}" for index in range(n_cells)]),
            "names": np.asarray([f"cell-{index}" for index in range(n_cells)]),
            "capture": np.asarray(["a"] * 20 + ["b"] * 20 + ["c"] * 20),
            "RNA_nCounts": np.linspace(10, 100, n_cells),
            "RNA_nFeatures": np.linspace(5, 50, n_cells),
        }
    )
    store.zw = zarr.open_group(store=MemoryStore(), mode="w")
    cell_data = store.zw.create_group("cellData")
    cell_data.create_array("ids", data=store.cells._values["ids"].astype("U16"))
    cell_data.create_array("I", data=store.cells._values["I"])
    store.refresh_cell_selection()
    context = _context(
        store,
        directions={
            "physicalCaptureColumn": "capture",
            "cellQc": {"pooledReferenceCaptures": ["a", "b"]},
        },
    )

    inspected = asyncio.run(inspect_cell_covariates(context))

    registered = {
        profile.registeredProfile: profile
        for profile in inspected.qcProfiles
        if profile.registeredProfile is not None
    }
    assert set(registered) == {
        "retainWithFlags",
        "globalMad5",
        "captureMad5",
        "captureMad3Sensitivity",
        "pooledReferenceMad5",
    }
    assert registered["captureMad5"].sampleColumn == "capture"
    assert registered["captureMad5"].parameters["captureSizes"] == {
        "a": 20,
        "b": 20,
        "c": 20,
    }
    assert registered["globalMad5"].parameters["captureComparisonSource"] == (
        "metadata:capture"
    )
    assert registered["pooledReferenceMad5"].parameters["pooledReferenceCaptures"] == [
        "a",
        "b",
    ]


def test_datastore_executes_exact_registered_bounds_and_persists_flags() -> None:
    n_cells = 64
    counts = np.linspace(80.0, 120.0, n_cells)
    counts[0] = 1.0
    counts[-1] = 10_000.0
    mito = np.linspace(1.0, 5.0, n_cells)
    mito[-2] = 80.0
    store, source = _memory_qc_store({"RNA_nCounts": counts, "RNA_percentMito": mito})
    active = np.ones(n_cells, dtype=bool)
    projection = project_registered_qc_profile(
        "globalMad5",
        values_by_metric={
            "RNA_nCounts": counts,
            "RNA_percentMito": mito,
        },
        active=active,
    )
    parameters = _profile_parameters(projection)
    live_before = np.asarray(store.cells.fetch_all("I"), dtype=bool).copy()

    selected, flags = execute_registered_cell_qc(
        store,
        "globalMad5",
        profile_parameters=parameters,
        expected_active_cells=n_cells,
        expected_retained_cells=projection.retainedCells,
        expected_flag_counts=projection.flagCounts,
        attrs=["RNA_nCounts", "RNA_percentMito"],
        cell_selection=source,
    )

    stored_selection = read_stored_selection_mask(
        store.zw,
        selected,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    np.testing.assert_array_equal(stored_selection, projection.keep)
    np.testing.assert_array_equal(store.cells.fetch_all("I"), live_before)
    assert flags is not None
    selection_status = store.inspect_artifact(selected)
    assert selection_status.operation == "run_registered_cell_qc"
    assert selection_status.parameters["profileParameters"] == parameters
    assert selection_status.inputs["diagnostic_flags"] == flags.to_dict()
    flag_status = store.inspect_artifact(flags)
    flag_names = flag_status.parameters["flagNames"]
    flag_values = np.asarray(artifact_group(store.zw, flags)["values"][:], dtype=bool)
    assert flag_values.shape == (n_cells, len(flag_names))
    assert {
        name: int(flag_values[:, index].sum()) for index, name in enumerate(flag_names)
    } == projection.flagCounts
    assert projection.keep[-1]
    assert projection.flags["RNA_nCounts:high"][-1]
    assert not projection.keep[-2]

    repeated, repeated_flags = execute_registered_cell_qc(
        store,
        "globalMad5",
        profile_parameters=parameters,
        expected_active_cells=n_cells,
        expected_retained_cells=projection.retainedCells,
        expected_flag_counts=projection.flagCounts,
        attrs=["RNA_nCounts", "RNA_percentMito"],
        cell_selection=source,
    )
    assert repeated == selected
    assert repeated_flags == flags


def test_datastore_rejects_modified_registered_bounds() -> None:
    counts = np.concatenate([np.linspace(80.0, 120.0, 39), np.asarray([1.0])])
    store, source = _memory_qc_store({"RNA_nCounts": counts})
    active = read_stored_selection_mask(
        store.zw,
        source,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    active_counts = np.asarray(store.cells.fetch_all("RNA_nCounts"), dtype=float)[
        active
    ]
    projection = project_registered_qc_profile(
        "globalMad5",
        values_by_metric={"RNA_nCounts": active_counts},
        active=np.ones(len(active_counts), dtype=bool),
    )
    parameters = deepcopy(_profile_parameters(projection))
    parameters["resolvedBounds"][0]["lowerRemoval"] += 1.0

    with pytest.raises(ValueError, match="resolved bounds do not match"):
        execute_registered_cell_qc(
            store,
            "globalMad5",
            profile_parameters=parameters,
            expected_active_cells=int(active.sum()),
            expected_retained_cells=projection.retainedCells,
            expected_flag_counts=projection.flagCounts,
            attrs=["RNA_nCounts"],
            cell_selection=source,
        )


def test_execute_auto_cell_qc_global_gaussian_persists_exact_outputs() -> None:
    values = _quality_values()
    store, source = _memory_qc_store(values)
    active = np.ones(store.cells.N, dtype=bool)
    projection = project_auto_filter_profile(
        "globalGaussian",
        values_by_metric=values,
        active=active,
    )
    live_before = store.cells.fetch_all("I").copy()

    selected, flags = execute_auto_cell_qc(
        store,
        "globalGaussian",
        profile_parameters=projection.parameters,
        expected_active_cells=store.cells.N,
        expected_retained_cells=projection.retainedCells,
        expected_flag_counts=projection.flagCounts,
        expected_resolved_bounds=projection.parameters["resolvedBounds"],
        attrs=list(values),
        cell_selection=source,
    )

    stored_selection = read_stored_selection_mask(
        store.zw,
        selected,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    np.testing.assert_array_equal(stored_selection, projection.keep)
    np.testing.assert_array_equal(store.cells.fetch_all("I"), live_before)
    assert flags is not None
    assert store.inspect_artifact(selected).operation == "auto_filter_cells"
    flag_status = store.inspect_artifact(flags)
    assert flag_status.operation == "run_auto_cell_qc_flags"
    flag_names = flag_status.parameters["flagNames"]
    flag_values = np.asarray(artifact_group(store.zw, flags)["values"][:], dtype=bool)
    assert {
        name: int(flag_values[:, index].sum()) for index, name in enumerate(flag_names)
    } == projection.flagCounts

    repeated = execute_auto_cell_qc(
        store,
        "globalGaussian",
        profile_parameters=projection.parameters,
        expected_active_cells=store.cells.N,
        expected_retained_cells=projection.retainedCells,
        expected_flag_counts=projection.flagCounts,
        expected_resolved_bounds=projection.parameters["resolvedBounds"],
        attrs=list(values),
        cell_selection=source,
    )
    assert repeated == (selected, flags)


@pytest.mark.parametrize("capture_kind", ["metadata", "artifact"])
def test_execute_auto_cell_qc_global_gaussian_tracks_artifact_and_capture_sources(
    capture_kind: str,
) -> None:
    labels = np.asarray(["capture-a"] * 30 + ["capture-b"] * 30)
    metadata_counts = np.concatenate(
        [np.linspace(90.0, 110.0, 30), np.linspace(95.0, 115.0, 30)]
    )
    artifact_mito = np.concatenate(
        [np.linspace(1.0, 3.0, 30), np.linspace(2.0, 4.0, 30)]
    )
    artifact_mito[-1] = 50.0
    initial = {"RNA_nCounts": metadata_counts}
    if capture_kind == "metadata":
        initial["capture"] = labels
    store, source = _memory_qc_store(initial)
    metric = _write_memory_cell_artifact(
        store,
        selection=source,
        name="RNA_percentMito",
        kind="quality_metric",
        values=artifact_mito,
        assay="RNA",
    )
    capture = (
        None
        if capture_kind == "metadata"
        else _write_memory_cell_artifact(
            store,
            selection=source,
            name="capture",
            kind="hto_identity",
            values=labels,
            assay="HTO",
        )
    )
    projection = project_auto_filter_profile(
        "globalGaussian",
        values_by_metric={
            "RNA_nCounts": metadata_counts,
            "RNA_percentMito": artifact_mito,
        },
        active=np.ones(store.cells.N, dtype=bool),
        sample_labels=labels,
        grouping_proven=True,
    )

    selected, flags = execute_auto_cell_qc(
        store,
        "globalGaussian",
        profile_parameters=projection.parameters,
        expected_active_cells=store.cells.N,
        expected_retained_cells=projection.retainedCells,
        expected_flag_counts=projection.flagCounts,
        expected_resolved_bounds=projection.parameters["resolvedBounds"],
        attrs=["RNA_nCounts"],
        artifact_metrics=[metric],
        cell_selection=source,
        capture_column="capture" if capture_kind == "metadata" else None,
        capture_artifact=capture,
    )

    assert flags is not None
    np.testing.assert_array_equal(
        read_stored_selection_mask(
            store.zw,
            selected,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ),
        projection.keep,
    )
    inputs = store.inspect_artifact(flags).inputs
    assert inputs["artifact_metrics"] == {"RNA_percentMito": metric.artifact.to_dict()}
    if capture_kind == "metadata":
        assert inputs["grouping_source"]["column"] == "capture"
    else:
        assert capture is not None
        assert inputs["grouping_source"]["artifact"] == capture.artifact.to_dict()


@pytest.mark.parametrize("source_kind", ["metadata", "artifact"])
def test_execute_auto_cell_qc_sample_mad_uses_exact_grouping_source(
    source_kind: str,
) -> None:
    labels = np.asarray(["sample-a"] * 22 + ["sample-b"] * 22)
    counts = np.concatenate(
        [np.linspace(90.0, 110.0, 22), np.append(np.linspace(95.0, 115.0, 21), 1000.0)]
    )
    initial = {"RNA_nCounts": counts}
    if source_kind == "metadata":
        initial["sample"] = labels
    store, source = _memory_qc_store(initial)
    sample_artifact = (
        None
        if source_kind == "metadata"
        else _write_memory_cell_artifact(
            store,
            selection=source,
            name="sample",
            kind="hto_identity",
            values=labels,
            assay="HTO",
        )
    )
    projection = project_auto_filter_profile(
        "sampleMad",
        values_by_metric={"RNA_nCounts": counts},
        active=np.ones(store.cells.N, dtype=bool),
        sample_labels=labels,
        grouping_proven=True,
        n_mads=3.0,
        min_cells_per_sample=20,
    )
    skip_reasons = projection.parameters["skipReasons"]
    assert isinstance(skip_reasons, dict)
    parameters = {
        "nMads": 3.0,
        "minCellsPerSample": 20,
        "nSamples": len(projection.captureSizes),
        "nSkippedSamples": len(skip_reasons),
    }

    selected, flags = execute_auto_cell_qc(
        store,
        "sampleMad",
        profile_parameters=parameters,
        expected_active_cells=store.cells.N,
        expected_retained_cells=projection.retainedCells,
        expected_flag_counts=projection.flagCounts,
        expected_resolved_bounds=projection.parameters["resolvedBounds"],
        attrs=["RNA_nCounts"],
        cell_selection=source,
        sample_column="sample" if source_kind == "metadata" else None,
        sample_artifact=sample_artifact,
    )

    stored_selection = read_stored_selection_mask(
        store.zw,
        selected,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    np.testing.assert_array_equal(stored_selection, projection.keep)
    assert not stored_selection[-1]
    assert flags is not None
    grouping_source = store.inspect_artifact(flags).inputs["grouping_source"]
    if source_kind == "metadata":
        assert grouping_source["source"] == "metadataColumn"
        assert grouping_source["column"] == "sample"
        assert grouping_source["fingerprint"]
    else:
        assert sample_artifact is not None
        assert grouping_source == {
            "source": "artifact",
            "artifact": sample_artifact.artifact.to_dict(),
        }


def test_execute_registered_capture_qc_resolves_artifact_metric_collision() -> None:
    labels = np.asarray(["capture-a"] * 30 + ["capture-b"] * 30)
    metadata_counts = np.concatenate(
        [np.linspace(90.0, 110.0, 30), np.linspace(95.0, 115.0, 30)]
    )
    metadata_counts[0] = 1.0
    artifact_counts = np.concatenate(
        [np.linspace(45.0, 55.0, 30), np.linspace(48.0, 58.0, 30)]
    )
    artifact_counts[-1] = 500.0
    store, source = _memory_qc_store({"RNA_nCounts": metadata_counts})
    metric = _write_memory_cell_artifact(
        store,
        selection=source,
        name="RNA_nCounts",
        kind="quality_metric",
        values=artifact_counts,
        assay="RNA",
    )
    capture = _write_memory_cell_artifact(
        store,
        selection=source,
        name="capture",
        kind="hto_identity",
        values=labels,
        assay="HTO",
    )
    execution_name = qc_metric_execution_name(
        metric.name,
        artifact_id=metric.artifact.artifact_id,
        collides_with_metadata=True,
    )
    projection = project_registered_qc_profile(
        "captureMad5",
        values_by_metric={
            "RNA_nCounts": metadata_counts,
            execution_name: artifact_counts,
        },
        active=np.ones(store.cells.N, dtype=bool),
        capture_labels=labels,
        grouping_proven=True,
        min_cells_per_capture=20,
    )
    parameters = _profile_parameters(projection)

    selected, flags = execute_registered_cell_qc(
        store,
        "captureMad5",
        profile_parameters=parameters,
        expected_active_cells=store.cells.N,
        expected_retained_cells=projection.retainedCells,
        expected_flag_counts=projection.flagCounts,
        attrs=["RNA_nCounts"],
        artifact_metrics=[metric],
        cell_selection=source,
        sample_artifact=capture,
    )

    stored_selection = read_stored_selection_mask(
        store.zw,
        selected,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    np.testing.assert_array_equal(stored_selection, projection.keep)
    assert flags is not None
    selection_status = store.inspect_artifact(selected)
    assert selection_status.parameters["profileParameters"]["captureSizes"] == {
        "capture-a": 30,
        "capture-b": 30,
    }
    assert selection_status.inputs["capture_artifact"] == capture.artifact.to_dict()
    flag_status = store.inspect_artifact(flags)
    assert flag_status.inputs["artifact_metrics"] == {
        execution_name: metric.artifact.to_dict()
    }


def test_orchestrator_executes_retain_with_flags_instead_of_plain_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {"RNA_nCounts": np.asarray([1.0, 10.0, 11.0, 100.0])}
    projection = project_registered_qc_profile(
        "retainWithFlags",
        values_by_metric=values,
        active=np.ones(4, dtype=bool),
    )
    parameters = _profile_parameters(projection)
    profile = CellQcProfileEvidence(
        profileId="registered-retain",
        action="skip",
        registeredProfile="retainWithFlags",
        driverAssay="RNA",
        driverAssayType="RNA",
        attributes=["RNA_nCounts"],
        parameters=parameters,
        activeCells=4,
        retainedCells=projection.retainedCells,
        retainedFraction=1.0,
        flaggedCells=projection.flagCounts,
        evidenceId="qcProfile:registered-retain",
    )
    plan = CellQcPlan(
        action=profile.action,
        registeredProfile=profile.registeredProfile,
        profileId=profile.profileId,
        driverAssay=profile.driverAssay,
        driverAssayType=profile.driverAssayType,
        attributes=profile.attributes,
        evidenceIds=[profile.evidenceId],
    )
    experimental = ExperimentalContextResult.get_blank().model_copy(
        update={"cellQc": plan, "qcProfiles": [profile]}
    )
    source = ArtifactRef(
        scope="datastore",
        kind="cell_selection",
        artifact_id="a" * 64,
    )
    selected = ArtifactRef(
        scope="datastore",
        kind="cell_selection",
        artifact_id="b" * 64,
    )
    flags = ArtifactRef(
        scope="datastore",
        kind="metadata_snapshot",
        artifact_id="c" * 64,
    )
    captured: dict[str, Any] = {}

    class Store:
        pass

    def fake_execute(
        store: Any,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[ArtifactRef, ArtifactRef]:
        captured["store"] = store
        captured["args"] = args
        captured.update(kwargs)
        return selected, flags

    monkeypatch.setattr(
        preprocessing_module,
        "execute_registered_cell_qc",
        fake_execute,
    )

    actions: list[str] = []
    operations: list[dict[str, Any]] = []
    store = Store()
    result = AgentOrchestrator(object()).apply_cell_qc(
        store, experimental, source, actions, operations
    )

    assert result == selected
    assert captured["store"] is store
    assert captured["args"] == ("retainWithFlags",)
    assert captured["profile_parameters"] == parameters
    assert actions == ["cell_qc_registered:retainWithFlags"]
    assert operations[0]["diagnosticFlags"] == (
        ArtifactReferenceModel.from_artifact_ref(flags).model_dump(mode="json")
    )


@pytest.mark.parametrize("sample_aware", [False, True])
def test_audited_core_qc_policy_executes_exact_projected_selection(
    monkeypatch: pytest.MonkeyPatch,
    sample_aware: bool,
) -> None:
    from types import SimpleNamespace

    from scarf.agent.decisions.rna import QcGroupingExecutorPayload

    values = _quality_values()
    labels = np.asarray(["a"] * 21 + ["b"] * 21)
    store, source = _memory_qc_store({**values, "capture": labels})
    action = "sampleMad" if sample_aware else "globalGaussian"
    policy = "coreSampleMad3" if sample_aware else "coreGlobalGaussian"
    projection = project_auto_filter_profile(
        action,
        values_by_metric=values,
        active=np.ones(42, dtype=bool),
        sample_labels=labels if sample_aware else None,
        grouping_proven=sample_aware,
    )
    parameters = (
        projection.parameters
        if not sample_aware
        else {
            "nMads": 3.0,
            "minCellsPerSample": 20,
            "nSamples": 2,
            "nSkippedSamples": 0,
        }
    )
    profile = CellQcProfileEvidence(
        profileId=f"cellQc:RNA:{policy}",
        action=action,
        driverAssay="RNA",
        driverAssayType="RNA",
        sampleColumn="capture" if sample_aware else None,
        captureColumn="capture" if sample_aware else None,
        attributes=list(values),
        parameters=parameters,
        resolvedBounds=projection.parameters["resolvedBounds"],
        activeCells=42,
        retainedCells=projection.retainedCells,
        retainedFraction=projection.retainedCells / 42,
        activeCellsByCapture=projection.captureSizes,
        sampleRetainedCells=projection.retainedByCapture,
        flaggedCells=projection.flagCounts,
        evidenceId=f"qcProfile:{policy}",
    )
    from scarf.agent.experimental_context.contracts import (
        CovariateCharacterization,
        ExperimentalContextDecision,
    )

    experimental = ExperimentalContextResult(
        status="done",
        decision=ExperimentalContextDecision(),
        characterization=CovariateCharacterization(status="done"),
        qcProfiles=[profile],
    )
    orchestrator = AgentOrchestrator(object())

    def resolve(_store, _request, definition, bundle, _answers, *, qc_evidence):
        assert (
            qc_evidence["policies"][0]["resolvedBounds"]
            == profile.model_dump(mode="json")["resolvedBounds"]
        )
        assert qc_evidence["policies"][0]["retainedCells"] == projection.retainedCells
        option = definition.executor_option(f"cellQuality:{policy}")
        assert option.payload.lowerCountMad is None
        assert option.payload.upperMitoMad is None
        if not sample_aware:
            assert definition.spec.baselineOptionId == "cellQuality:coreGlobalGaussian"
            assert (
                definition.spec.options[0].optionId == "cellQuality:coreGlobalGaussian"
            )
        assert bundle.evidence[0].evidenceId == profile.evidenceId
        return SimpleNamespace(
            compiled=SimpleNamespace(executorPayload=option.payload),
            record=SimpleNamespace(
                rationale="Exact core QC preserves valid cells.",
                evidenceIds=[profile.evidenceId],
            ),
            checkpointSha256="a" * 64,
        )

    monkeypatch.setattr(orchestrator, "_resolve_rna_decision", resolve)
    payload, plan, _ = orchestrator._resolve_cell_quality_decision(
        store,
        None,
        experimental,
        QcGroupingExecutorPayload(
            groupingMode="physicalCapture" if sample_aware else "global"
        ),
        {},
    )
    from scarf.agent.orchestrator.models import AutomatedPreprocessingPlan

    saved = AutomatedPreprocessingPlan(cellQc=plan, cellQualityPayload=payload)
    restored = AutomatedPreprocessingPlan.model_validate_json(saved.model_dump_json())
    assert restored.cellQualityPayload == payload
    assert restored.cellQc == plan
    with pytest.raises(ValidationError, match="exact selected QC policy"):
        AutomatedPreprocessingPlan(
            cellQc=plan,
            cellQualityPayload=payload.model_copy(
                update={
                    "profile": "coreGlobalGaussian"
                    if sample_aware
                    else "coreSampleMad3",
                    "groupByCapture": not sample_aware,
                }
            ),
        )
    actions: list[str] = []
    operations: list[dict[str, Any]] = []
    selected = orchestrator.apply_cell_qc(
        store,
        experimental,
        source,
        actions,
        operations,
        selected_plan=plan,
        decision_payload=payload,
    )
    np.testing.assert_array_equal(
        read_stored_selection_mask(
            store.zw,
            selected,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ),
        projection.keep,
    )
    assert operations[0]["profileParameters"] == parameters
    assert operations[0]["resolvedBounds"] == projection.parameters["resolvedBounds"]
    assert operations[0]["diagnosticFlags"] is not None
    np.testing.assert_array_equal(store.cells.fetch_all("I"), True)


def test_core_sample_qc_requires_exact_physical_capture_proof() -> None:
    profile = CellQcProfileEvidence(
        profileId="cellQc:sample",
        action="sampleMad",
        sampleColumn="sample",
        attributes=["RNA_nCounts"],
        activeCells=40,
        retainedCells=39,
        retainedFraction=39 / 40,
    )
    assert not AgentOrchestrator._profile_is_safe(profile)
    profile.captureColumn = "different"
    assert not AgentOrchestrator._profile_is_safe(profile)
    profile.captureColumn = "sample"
    assert AgentOrchestrator._profile_is_safe(profile)
    profile.unsafeRetentionGroups = ["combination:treatment,time=case,late"]
    assert not AgentOrchestrator._profile_is_safe(profile)

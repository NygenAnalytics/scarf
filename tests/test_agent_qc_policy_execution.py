"""Offered QC alternatives execute their exact frozen cohort and diagnostic flags."""

from copy import deepcopy
from typing import Any

import numpy as np
import pytest

from scarf.agent.cell_quality.execution import (
    execute_auto_cell_qc,
    execute_registered_cell_qc,
)
from scarf.agent.cell_quality.profiles import (
    project_auto_filter_profile,
    project_registered_qc_profile,
)
from scarf.agent.experimental_context.characterization import _SelectionBoundCells
from scarf.storage.artifacts import artifact_group
from scarf.storage.refs import ArtifactRef
from scarf.storage.selections import read_stored_selection_mask
from tests.test_pipeline import _insert_nullable_cell_column
from tests.test_registered_qc_profiles import (
    _memory_qc_store,
    _profile_parameters,
    _quality_values,
    _write_memory_cell_artifact,
)


@pytest.mark.parametrize(
    "policy",
    [
        "retainWithFlags",
        "globalMad5",
        "captureMad5",
        "captureMad3Sensitivity",
        "pooledReferenceMad5",
    ],
)
def test_qc_policy_projection_matches_execution_on_a_preselected_cohort(
    policy: str,
) -> None:
    values = _quality_values()
    captures = np.asarray(["a"] * 21 + ["b"] * 21)
    store, _ = _memory_qc_store({**values, "capture": captures})
    store.cells._get_array("I")[0] = False
    source = store.snapshot_cell_selection("I")
    active = store.cells.fetch_all("I")
    grouped = policy in {"captureMad5", "captureMad3Sensitivity", "pooledReferenceMad5"}
    references = ("a", "b") if policy == "pooledReferenceMad5" else ()
    projection = project_registered_qc_profile(
        policy,
        values_by_metric={name: value[active] for name, value in values.items()},
        active=np.ones(int(active.sum()), dtype=bool),
        capture_labels=captures[active] if grouped else None,
        grouping_proven=grouped,
        pooled_reference_captures=references,
    )
    parameters = _profile_parameters(projection)
    parameters["pooledReferenceCaptures"] = list(references)
    selected, flags = execute_registered_cell_qc(
        store,
        policy,
        profile_parameters=parameters,
        expected_active_cells=int(active.sum()),
        expected_retained_cells=projection.retainedCells,
        expected_flag_counts=projection.flagCounts,
        attrs=list(values),
        cell_selection=source,
        sample_column="capture" if grouped else None,
    )
    observed = read_stored_selection_mask(
        store.zw,
        selected,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    expected = np.zeros(len(active), dtype=bool)
    expected[active] = projection.keep
    np.testing.assert_array_equal(observed, expected)
    np.testing.assert_array_equal(store.cells.fetch_all("I"), active)
    assert flags is not None
    np.testing.assert_array_equal(
        artifact_group(store.zw, flags)["values"][:],
        np.column_stack([projection.flags[name] for name in sorted(projection.flags)]),
    )
    if policy == "retainWithFlags":
        np.testing.assert_array_equal(observed, active)
        assert any(count > 0 for count in projection.flagCounts.values())
    for name, original in values.items():
        np.testing.assert_array_equal(store.cells.fetch_all(name), original)


@pytest.mark.parametrize("changed", ["captureSizes", "captureComparisons", "metric"])
def test_capture_execution_rejects_stale_evidence_before_saving_a_selection(
    changed: str,
) -> None:
    values = _quality_values()
    captures = np.asarray(["a"] * 21 + ["b"] * 21)
    store, source = _memory_qc_store({**values, "capture": captures})
    projection = project_registered_qc_profile(
        "captureMad5",
        values_by_metric=values,
        active=np.ones(42, dtype=bool),
        capture_labels=captures,
        grouping_proven=True,
    )
    parameters = deepcopy(_profile_parameters(projection))
    if changed == "captureSizes":
        parameters["captureSizes"]["a"] -= 1
    elif changed == "captureComparisons":
        parameters["captureComparisons"][0]["retainedCells"] = -1
    else:
        store.cells._get_array("RNA_nCounts")[0] = 0
    with pytest.raises(ValueError, match="do not match"):
        execute_registered_cell_qc(
            store,
            "captureMad5",
            profile_parameters=parameters,
            expected_active_cells=42,
            expected_retained_cells=projection.retainedCells,
            expected_flag_counts=projection.flagCounts,
            attrs=list(values),
            cell_selection=source,
            sample_column="capture",
        )
    np.testing.assert_array_equal(store.cells.fetch_all("I"), np.ones(42, dtype=bool))


@pytest.mark.parametrize(
    "invalid", ["empty", "matrixSelection", "matrixMetric", "unaligned", "nonfinite"]
)
def test_qc_projection_rejects_unmeasurable_or_misaligned_selected_cells(
    invalid: str,
) -> None:
    active = np.ones(4, dtype=bool)
    counts = np.asarray([10.0, 11, 12, 13])
    if invalid == "empty":
        active[:] = False
    elif invalid == "matrixSelection":
        active = active.reshape(2, 2)
    elif invalid == "matrixMetric":
        counts = counts.reshape(2, 2)
    elif invalid == "unaligned":
        counts = counts[:-1]
    else:
        counts[0] = np.nan
    with pytest.raises(ValueError):
        project_registered_qc_profile(
            "globalMad5", values_by_metric={"RNA_nCounts": counts}, active=active
        )


def test_qc_ignores_unselected_missing_values_and_diagnostic_only_covariates() -> None:
    active = np.ones(40, dtype=bool)
    active[0] = False
    counts = np.arange(40, dtype=float) + 100
    counts[0] = np.nan
    reference = project_registered_qc_profile(
        "globalMad5", values_by_metric={"RNA_nCounts": counts}, active=active
    )
    measured = project_registered_qc_profile(
        "globalMad5",
        values_by_metric={"RNA_nCounts": counts, "age": np.arange(40) ** 4},
        active=active,
    )
    np.testing.assert_array_equal(measured.keep, reference.keep)
    assert measured.thresholds == reference.thresholds
    assert not measured.keep[0]


@pytest.mark.parametrize("references", [("a", "a"), ("a", "missing"), ("a", "b")])
def test_invalid_reference_pool_is_unavailable_without_changing_global_evidence(
    references: tuple[str, ...],
) -> None:
    from scarf.agent.cell_quality.profiles import offered_registered_qc_profiles

    active = np.ones(30, dtype=bool)
    captures = np.asarray(["a"] * 5 + ["b"] * 5 + ["c"] * 20)
    values = {"RNA_nCounts": np.arange(30, dtype=float) + 100}
    with pytest.raises(
        ValueError, match="reference captures|reference captures do not"
    ):
        project_registered_qc_profile(
            "pooledReferenceMad5",
            values_by_metric=values,
            active=active,
            capture_labels=captures,
            grouping_proven=True,
            pooled_reference_captures=references,
        )
    offered = offered_registered_qc_profiles(
        values_by_metric=values,
        active=active,
        capture_labels=captures,
        grouping_proven=True,
        pooled_reference_captures=references,
    )
    assert {profile.profile for profile in offered} == {"retainWithFlags", "globalMad5"}
    global_profile = next(
        profile for profile in offered if profile.profile == "globalMad5"
    )
    reference = project_registered_qc_profile(
        "globalMad5", values_by_metric=values, active=active
    )
    np.testing.assert_array_equal(global_profile.keep, reference.keep)


def test_capture_provenance_cannot_merge_distinct_typed_labels() -> None:
    captures = np.asarray([1] * 20 + ["1"] * 20, dtype=object)
    with pytest.raises(ValueError, match="consistent label type"):
        project_registered_qc_profile(
            "captureMad5",
            values_by_metric={"RNA_nCounts": np.arange(40, dtype=float) + 100},
            active=np.ones(40, dtype=bool),
            capture_labels=captures,
            grouping_proven=True,
        )


_MASKED_SOURCES = [
    "metadataMetric",
    "artifactMetric",
    "metadataCapture",
    "captureArtifact",
]


def _stored_paths(store: Any) -> list[str]:
    return sorted(name for name, _ in store.zw.members(max_depth=None))


def _mask_artifact(store: Any, ref: ArtifactRef, missing: np.ndarray) -> None:
    group = artifact_group(store.zw, ref)
    group.create_array("__scarf_missing__values", data=missing)
    group["values"].attrs["missing_mask"] = "__scarf_missing__values"


def _placeholder_case(
    masked: str,
) -> tuple[Any, ArtifactRef, dict[str, Any], np.ndarray, np.ndarray]:
    """Store placeholders that only a linked missing-value mask marks as missing.

    Evidence reads the masked rows as missing, but the stored placeholders (0)
    look like measured values to a reader that ignores the mask.
    """
    counts = np.linspace(80.0, 120.0, 60)
    captures = np.repeat([1, 2, 3], 20)
    missing = np.zeros(60, dtype=bool)
    if masked.endswith("Metric"):
        missing[[4, 33]] = True
        counts[missing] = 0.0
    else:
        missing[40:] = True
        captures[missing] = 0
    columns: dict[str, np.ndarray] = {}
    if masked != "metadataMetric":
        columns["RNA_nCounts"] = counts
    if masked != "metadataCapture":
        columns["capture"] = captures
    store, source = _memory_qc_store(columns)
    sources: dict[str, Any] = {"attrs": ["RNA_nCounts"], "sample_column": "capture"}
    if masked == "metadataMetric":
        _insert_nullable_cell_column(store, "RNA_nCounts", counts, missing)
    elif masked == "metadataCapture":
        _insert_nullable_cell_column(store, "capture", captures, missing)
    elif masked == "artifactMetric":
        metric = _write_memory_cell_artifact(
            store,
            selection=source,
            name="RNA_nCounts",
            kind="quality_metric",
            values=counts,
            assay="RNA",
        )
        _mask_artifact(store, metric.artifact, missing)
        sources.update(attrs=[], artifact_metrics=[metric])
    else:
        identity = _write_memory_cell_artifact(
            store,
            selection=source,
            name="capture",
            kind="hto_identity",
            values=captures,
            assay="HTO",
        )
        _mask_artifact(store, identity.artifact, missing)
        del sources["sample_column"]
        sources["sample_artifact"] = identity
    return store, source, sources, counts, captures


@pytest.mark.parametrize("masked", _MASKED_SOURCES)
def test_registered_execution_refuses_masked_inputs_before_saving(masked: str) -> None:
    store, source, sources, counts, captures = _placeholder_case(masked)
    projection = project_registered_qc_profile(
        "captureMad5",
        values_by_metric={"RNA_nCounts": counts},
        active=np.ones(60, dtype=bool),
        capture_labels=captures,
        grouping_proven=True,
    )
    before = _stored_paths(store)
    with pytest.raises(ValueError, match="non-finite|missing labels"):
        execute_registered_cell_qc(
            store,
            "captureMad5",
            profile_parameters=_profile_parameters(projection),
            expected_active_cells=60,
            expected_retained_cells=projection.retainedCells,
            expected_flag_counts=projection.flagCounts,
            cell_selection=source,
            **sources,
        )
    assert _stored_paths(store) == before


@pytest.mark.parametrize("masked", _MASKED_SOURCES)
def test_auto_execution_refuses_masked_inputs_before_core_filtering(
    masked: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, source, sources, counts, captures = _placeholder_case(masked)
    # globalGaussian takes its physical capture labels as a capture source.
    if "sample_column" in sources:
        sources["capture_column"] = sources.pop("sample_column")
    else:
        sources["capture_artifact"] = sources.pop("sample_artifact")
    projection = project_auto_filter_profile(
        "globalGaussian",
        values_by_metric={"RNA_nCounts": counts},
        active=np.ones(60, dtype=bool),
        sample_labels=captures,
        grouping_proven=True,
    )

    def core_filtering(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("Core filtering ran on inputs with masked rows")

    monkeypatch.setattr(store, "auto_filter_cells", core_filtering)
    before = _stored_paths(store)
    with pytest.raises(ValueError, match="finite|missing labels"):
        execute_auto_cell_qc(
            store,
            "globalGaussian",
            profile_parameters=projection.parameters,
            expected_active_cells=60,
            expected_retained_cells=projection.retainedCells,
            expected_flag_counts=projection.flagCounts,
            expected_resolved_bounds=projection.parameters["resolvedBounds"],
            cell_selection=source,
            **sources,
        )
    assert _stored_paths(store) == before


def test_bound_evidence_reads_masked_artifact_labels_as_missing() -> None:
    store, source, sources, _, captures = _placeholder_case("captureArtifact")
    cells = _SelectionBoundCells(
        store.zw,
        store.cells,
        source,
        artifacts={"hto": sources["sample_artifact"].artifact},
    )
    expected = [None if value == 0 else value for value in captures.tolist()]
    assert cells.fetch("hto").tolist() == expected

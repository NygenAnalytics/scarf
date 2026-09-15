"""Offered QC alternatives execute their exact frozen cohort and diagnostic flags."""

from copy import deepcopy

import numpy as np
import pytest

from scarf.agent.cell_quality.execution import execute_registered_cell_qc
from scarf.agent.cell_quality.profiles import project_registered_qc_profile
from scarf.storage.artifacts import artifact_group
from scarf.storage.selections import read_stored_selection_mask
from tests.test_registered_qc_profiles import (
    _memory_qc_store,
    _profile_parameters,
    _quality_values,
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

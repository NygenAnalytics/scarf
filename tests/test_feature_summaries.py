import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.assay import ATACassay, RNAassay
from scarf.assay.feature_summary import feature_summary_values
from scarf.datastore.datastore import DataStore
from scarf.metadata import MetaData
from scarf.metadata.arguments import CellCycleArguments, PrevalentPeakArguments
from scarf.storage.artifact_writer import finish_artifact, plan_artifact, start_artifact
from scarf.storage.artifacts import (
    ArtifactRef,
    artifact_path,
    callable_identity,
    fingerprint_stored_arrays,
    inspect_artifact,
)


def _ref(kind: str, token: str, *, scope: str = "assay") -> ArtifactRef:
    return ArtifactRef(
        scope=scope,  # type: ignore[arg-type]
        assay="RNA" if scope == "assay" else None,
        kind=kind,
        artifact_id=token * 64,
    )


def test_rna_summary_derives_zero_safe_values_without_persisting_them() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    planned = plan_artifact(
        root,
        scope="assay",
        assay="RNA",
        kind="feature_summary",
        operation="summarize_rna_features",
        parameters={},
        inputs={},
        execution_options={},
    )
    group = start_artifact(root, planned)
    group.create_array("normed_tot", data=np.array([6.0, 0.0, 3.0]))
    group.create_array("normed_n", data=np.array([3.0, 0.0, 1.0]))
    group.create_array("sigmas", data=np.array([1.0, 0.0, 2.0]))
    group.attrs["payload_fingerprint"] = fingerprint_stored_arrays(
        group,
        ("normed_tot", "normed_n", "sigmas"),
    )
    finish_artifact(group, planned)

    values = feature_summary_values(root, planned.ref, n_selected=3)

    np.testing.assert_array_equal(values["avg"], np.array([2.0, 0.0, 1.0]))
    np.testing.assert_array_equal(values["nz_mean"], np.array([2.0, 0.0, 3.0]))
    empty_values = feature_summary_values(root, planned.ref, n_selected=0)
    np.testing.assert_array_equal(empty_values["avg"], np.zeros(3))
    np.testing.assert_array_equal(
        empty_values["nz_mean"],
        np.array([2.0, 0.0, 3.0]),
    )
    assert "avg" not in group
    assert "nz_mean" not in group


def test_cell_cycle_and_prevalence_records_have_exact_direct_inputs() -> None:
    summary = _ref("feature_summary", "1")
    cells = _ref("cell_selection", "2", scope="datastore")
    cell_cycle = CellCycleArguments(
        feature_summary=summary,
        cell_selection=cells,
        s_gene_indices=(1, 3),
        g2m_gene_indices=(2,),
        control_size=1,
        n_bins=10,
        rand_seed=7,
        invalidate_cache=False,
    ).to_record()
    assert set(cell_cycle.inputs) == {"feature_summary", "cell_selection"}
    assert set(cell_cycle.parameters) == {
        "s_gene_indices",
        "g2m_gene_indices",
        "control_size",
        "n_bins",
        "rand_seed",
    }

    prevalence = PrevalentPeakArguments(
        feature_summary=summary,
        top_n=100,
        invalidate_cache=False,
    ).to_record()
    assert set(prevalence.inputs) == {"feature_summary"}
    assert prevalence.parameters == {"top_n": 100}
    assert "algorithm_version" not in prevalence.parameters


def test_prevalent_peak_kernel_is_compute_only() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    features = root.create_group("featureData")
    features.create_array("I", data=np.ones(4, dtype=bool))
    features.create_array("ids", data=np.asarray(["a", "b", "c", "d"]))
    features.create_array("names", data=np.asarray(["a", "b", "c", "d"]))
    assay = ATACassay.__new__(ATACassay)
    assay.feats = MetaData(features)

    values = assay._prevalent_peak_mask(np.array([0.1, 2.0, 1.0, 0.2]), 2)

    np.testing.assert_array_equal(values, np.array([False, True, True, False]))
    assert set(assay.feats.columns) == {"I", "ids", "names"}
    assert not hasattr(RNAassay, "set_hvgs")
    assert not hasattr(RNAassay, "mark_hvgs")
    assert not hasattr(ATACassay, "mark_prevalent_peaks")


def test_prevalent_peak_kernel_pins_descending_and_tie_order() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    features = root.create_group("featureData")
    names = np.asarray(["a", "b", "c", "d", "e"])
    features.create_array("I", data=np.ones(len(names), dtype=bool))
    features.create_array("ids", data=names)
    features.create_array("names", data=names)
    assay = ATACassay.__new__(ATACassay)
    assay.feats = MetaData(features)

    values = assay._prevalent_peak_mask(
        np.array([1.0, 2.0, 2.0, 2.0, 0.0]),
        2,
    )

    # Preserve the historical pandas descending-sort tie order: indices 1 and 3
    # are selected before the third equal-valued peak at index 2.
    np.testing.assert_array_equal(
        values,
        np.array([False, True, False, True, False]),
    )


def test_corrupt_feature_summary_is_never_reused(datastore_ephemeral) -> None:
    store = datastore_ephemeral
    cells = store.snapshot_cell_selection()
    first = store.select_detected_features(
        cells,
        from_assay="RNA",
        min_cells=1,
    )
    first_status = inspect_artifact(store.zw, first)
    first_summary = ArtifactRef.from_dict(first_status.inputs["feature_summary"])
    first_group = store.zw[artifact_path(first_summary)]
    first_group["normed_tot"][0] = float(first_group["normed_tot"][0]) + 1.0

    second = store.select_detected_features(
        cells,
        from_assay="RNA",
        min_cells=1,
    )
    second_status = inspect_artifact(store.zw, second)
    second_summary = ArtifactRef.from_dict(second_status.inputs["feature_summary"])

    assert second_summary != first_summary
    assert second != first


def test_feature_summary_with_unexpected_payload_is_never_reused(
    datastore_ephemeral,
) -> None:
    store = datastore_ephemeral
    cells = store.snapshot_cell_selection()
    first = store.select_detected_features(
        cells,
        from_assay="RNA",
        min_cells=1,
    )
    first_status = inspect_artifact(store.zw, first)
    first_summary = ArtifactRef.from_dict(first_status.inputs["feature_summary"])
    first_group = store.zw[artifact_path(first_summary)]
    first_group.create_array(
        "unexpected",
        data=np.zeros(store.RNA.feats.N, dtype=np.float64),
    )

    second = store.select_detected_features(
        cells,
        from_assay="RNA",
        min_cells=1,
    )
    second_status = inspect_artifact(store.zw, second)
    second_summary = ArtifactRef.from_dict(second_status.inputs["feature_summary"])

    assert second_summary != first_summary
    assert second != first


def test_rna_summary_and_detected_selection_have_exact_ledger_identity(
    datastore_ephemeral,
) -> None:
    store = datastore_ephemeral
    cells = store.snapshot_cell_selection()
    detected = store.select_detected_features(
        cells,
        from_assay="RNA",
        min_cells=0,
    )
    detected_status = inspect_artifact(store.zw, detected)
    assert detected_status.parameters == {"min_cells": 0}
    assert set(detected_status.inputs or {}) == {"feature_summary"}

    summary_ref = ArtifactRef.from_dict(detected_status.inputs["feature_summary"])
    summary_status = inspect_artifact(store.zw, summary_ref)
    assert summary_status.operation == "summarize_rna_features"
    assert summary_status.parameters == {
        "normalization_method": callable_identity(store.RNA.normMethod),
        "size_factor": store.RNA.sf,
    }
    assert set(summary_status.inputs or {}) == {"cell_selection"}
    summary = store.zw[artifact_path(summary_ref)]
    assert set(summary.array_keys()) == {"normed_tot", "normed_n", "sigmas"}
    assert isinstance(summary.attrs["ordered_feature_ids_fingerprint"], str)
    assert summary.attrs["payload_fingerprint"] == fingerprint_stored_arrays(
        summary,
        ("normed_tot", "normed_n", "sigmas"),
    )
    np.testing.assert_array_equal(
        store.zw[artifact_path(detected)]["values"][:],
        np.asarray(summary["normed_n"][:]) >= 0,
    )
    assert not any(name.startswith("summary_stats_") for name in store.RNA.z)


def test_assay_score_features_is_compute_only_on_read_only_store(
    datastore_ephemeral,
) -> None:
    writable = datastore_ephemeral
    assay_groups = set(writable.RNA.z.group_keys())
    feature_columns = set(writable.RNA.feats.columns)
    feature_attrs = dict(writable.zw["RNA/featureData"].attrs)
    read_only = DataStore(
        writable.zarr_loc,
        default_assay="RNA",
        zarr_mode="r",
    )
    feature_name = str(read_only.RNA.feats.fetch_all("names")[0])

    scores = read_only.RNA.score_features(
        [feature_name],
        "I",
        1,
        10,
        7,
    )

    assert scores.shape == (int(read_only.cells.fetch_all("I").sum()),)
    assert set(writable.RNA.z.group_keys()) == assay_groups
    assert set(writable.RNA.feats.columns) == feature_columns
    assert dict(writable.zw["RNA/featureData"].attrs) == feature_attrs


def test_datastore_cell_cycle_read_only_guard_precedes_planning(
    datastore_ephemeral,
) -> None:
    writable = datastore_ephemeral
    assay_groups = set(writable.RNA.z.group_keys())
    cell_columns = set(writable.cells.columns)
    read_only = DataStore(
        writable.zarr_loc,
        default_assay="RNA",
        zarr_mode="r",
    )
    selection = writable.snapshot_cell_selection()

    with pytest.raises(
        PermissionError,
        match="Cell-cycle scoring requires a DataStore opened with zarr_mode='r\\+'",
    ):
        read_only.run_cell_cycle_scoring(selection)

    assert set(writable.RNA.z.group_keys()) == assay_groups
    assert set(writable.cells.columns) == cell_columns

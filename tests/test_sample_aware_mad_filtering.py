import numpy as np
import pytest
from scarf.metadata.artifacts import (
    plan_cell_data_artifact,
    write_cell_data_artifact,
)
from scarf.metadata.selection import NamedCellArtifact
from scarf.storage.artifacts import ArtifactRef
from scarf.storage.artifacts import fingerprint_array, fingerprint_strings
from scarf.storage.selections import read_stored_selection_mask
from scarf.utils import logger

from scarf.quality_control.filtering import (
    _apply_bounds,
    _mad_bounds,
    _metric_policy,
    _sample_aware_mad_mask,
    gaussian_quantile_bounds,
)


def _selection_mask(datastore, ref: ArtifactRef) -> np.ndarray:
    return read_stored_selection_mask(
        datastore.zw,
        ref,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )


def _write_cell_vector(
    datastore,
    *,
    selection: ArtifactRef,
    name: str,
    kind: str,
    values: np.ndarray,
    assay: str,
) -> NamedCellArtifact:
    resolved = np.asarray(values)
    fingerprint = (
        fingerprint_strings(resolved)
        if resolved.dtype.kind in {"O", "S", "U"}
        else fingerprint_array(resolved)
    )
    planned = plan_cell_data_artifact(
        datastore.zw,
        scope="assay",
        assay=assay,
        kind=kind,
        operation=f"test_cell_vector_{kind}",
        parameters={"name": name, "values_fingerprint": fingerprint},
        inputs={},
        execution_options={},
        cell_selection=selection,
        arrays={"values": ((len(resolved),), None)},
    )
    write_cell_data_artifact(
        datastore.zw,
        planned,
        {"values": resolved},
    )
    return NamedCellArtifact(name=name, artifact=planned.ref)


def test_mad_bounds_matches_hand_computed_example():
    values = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    median = 3.0
    scaled_mad = 1.4826 * np.median(np.abs(values - median))
    low, high, reported_mad = _mad_bounds(values, n_mads=3.0)

    assert reported_mad == pytest.approx(scaled_mad)
    assert low == pytest.approx(median - 3.0 * scaled_mad)
    assert high == pytest.approx(median + 3.0 * scaled_mad)


def test_apply_bounds_supports_open_and_closed_intervals():
    values = np.array([-np.inf, 0.0, 1.0, 2.0, np.inf, np.nan])

    np.testing.assert_array_equal(
        _apply_bounds(values, 0.0, 2.0),
        [False, False, True, False, False, False],
    )
    np.testing.assert_array_equal(
        _apply_bounds(values, 0.0, 2.0, keep_bounds=True),
        [False, True, True, True, False, False],
    )
    np.testing.assert_array_equal(
        _apply_bounds(values, None, None, keep_bounds=True),
        [True, True, True, True, True, False],
    )


def test_apply_bounds_rejects_non_vector_values():
    with pytest.raises(ValueError, match="one-dimensional"):
        _apply_bounds(np.ones((2, 2)), 0.0, 2.0)


def test_metric_policy_defaults_and_custom_attrs():
    assert _metric_policy("RNA_nCounts") == {
        "transform": "log1p",
        "bound_direction": "two_sided",
    }
    assert _metric_policy("RNA_percentMito") == {
        "transform": "identity",
        "bound_direction": "upper",
    }
    assert _metric_policy("custom_score") == {
        "transform": "identity",
        "bound_direction": "two_sided",
    }


@pytest.mark.parametrize("attr", ["RNA_nCounts", "RNA_nFeatures"])
def test_sample_aware_mask_rejects_negative_count_values(attr):
    with pytest.raises(ValueError, match="non-negative before log1p"):
        _sample_aware_mad_mask(
            values_by_attr={attr: np.array([1.0, 2.0, -2.0])},
            sample_labels=np.array(["A", "A", "A"]),
            active=np.ones(3, dtype=bool),
            n_mads=3.0,
            min_cells_per_sample=2,
            attrs=[attr],
        )


def test_sample_aware_mask_rejects_mixed_label_types_without_collision():
    with pytest.raises(ValueError, match="one consistent label type"):
        _sample_aware_mad_mask(
            values_by_attr={"score": np.arange(8, dtype=float)},
            sample_labels=np.array([1] * 4 + ["1"] * 4, dtype=object),
            active=np.ones(8, dtype=bool),
            n_mads=3.0,
            min_cells_per_sample=2,
            attrs=["score"],
        )


def test_sample_aware_mask_rejects_blank_bytes_labels():
    with pytest.raises(ValueError, match="missing labels"):
        _sample_aware_mad_mask(
            values_by_attr={"score": np.arange(4, dtype=float)},
            sample_labels=np.array([b"A", b"A", b"  ", b"  "]),
            active=np.ones(4, dtype=bool),
            n_mads=3.0,
            min_cells_per_sample=2,
            attrs=["score"],
        )


def test_sample_aware_mask_isolates_outliers_per_sample():
    # Sample A and B have different depths. A severe outlier in B must not
    # change A's bounds, and must be removed only from B.
    sample_a = np.linspace(9.0, 11.0, 20)
    sample_b = np.concatenate([np.linspace(99.0, 101.0, 19), [1000.0]])
    n_counts = np.concatenate([sample_a, sample_b])
    labels = np.array(["A"] * 20 + ["B"] * 20)
    active = np.ones(40, dtype=bool)

    keep, provenance = _sample_aware_mad_mask(
        values_by_attr={"RNA_nCounts": n_counts},
        sample_labels=labels,
        active=active,
        n_mads=3.0,
        min_cells_per_sample=5,
        attrs=["RNA_nCounts"],
    )

    assert bool(keep[:20].all())
    assert bool(keep[20:39].all())
    assert not bool(keep[-1])
    a_high = provenance["resolved_bounds"]["A"]["RNA_nCounts"]["high"]
    b_high = provenance["resolved_bounds"]["B"]["RNA_nCounts"]["high"]
    assert a_high < 50
    assert b_high > 50
    assert a_high < b_high


def test_sample_aware_mask_uses_log_counts_and_upper_only_percent():
    n_counts = np.expm1(
        np.array([2.0, 2.1, 1.9, 2.05, 2.02, 2.01, 1.98, 2.03, 1.99, 2.04, 8.0])
    )
    percent_mito = np.array(
        [1.0, 1.1, 0.9, 1.05, 1.02, 1.01, 0.98, 1.03, 0.99, 1.04, 40.0]
    )
    labels = np.array(["S"] * 11)
    active = np.ones(11, dtype=bool)

    keep, provenance = _sample_aware_mad_mask(
        values_by_attr={
            "RNA_nCounts": n_counts,
            "RNA_percentMito": percent_mito,
        },
        sample_labels=labels,
        active=active,
        n_mads=3.0,
        min_cells_per_sample=5,
        attrs=["RNA_nCounts", "RNA_percentMito"],
    )

    count_bounds = provenance["resolved_bounds"]["S"]["RNA_nCounts"]
    mito_bounds = provenance["resolved_bounds"]["S"]["RNA_percentMito"]
    assert count_bounds["transform"] == "log1p"
    assert count_bounds["bound_direction"] == "two_sided"
    assert count_bounds["low"] is not None
    assert mito_bounds["transform"] == "identity"
    assert mito_bounds["bound_direction"] == "upper"
    assert mito_bounds["low"] is None
    assert not bool(keep[-1])
    # An unusually low mito percentage must not be filtered by upper-only bounds.
    low_mito = percent_mito.copy()
    low_mito[0] = 0.0
    keep_low, _ = _sample_aware_mad_mask(
        values_by_attr={
            "RNA_nCounts": n_counts,
            "RNA_percentMito": low_mito,
        },
        sample_labels=labels,
        active=active,
        n_mads=3.0,
        min_cells_per_sample=5,
        attrs=["RNA_percentMito"],
    )
    assert bool(keep_low[0])


def test_sample_aware_mask_skips_small_and_zero_mad_groups():
    values = np.array([1.0, 2.0, 3.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    labels = np.array(["tiny"] * 3 + ["flat"] * 5)
    active = np.ones(8, dtype=bool)

    keep, provenance = _sample_aware_mad_mask(
        values_by_attr={"custom_score": values},
        sample_labels=labels,
        active=active,
        n_mads=3.0,
        min_cells_per_sample=5,
        attrs=["custom_score"],
    )

    assert bool(keep.all())
    assert provenance["skip_reasons"]["tiny"] == "insufficient_cells"
    assert (
        provenance["resolved_bounds"]["flat"]["custom_score"]["skip_reason"]
        == "zero_mad"
    )
    assert any("tiny" in message for message in provenance["warnings"])
    assert any("zero MAD" in message for message in provenance["warnings"])


def test_auto_filter_cells_without_sample_column_matches_gaussian(
    datastore_ephemeral,
):
    attrs = ["RNA_nCounts", "RNA_nFeatures"]
    expected_bounds = {
        attr: dict(
            zip(
                ("low", "high"),
                gaussian_quantile_bounds(
                    np.asarray(datastore_ephemeral.cells.fetch_all(attr)),
                    0.01,
                    0.99,
                ),
                strict=True,
            )
        )
        for attr in attrs
    }

    before = np.asarray(datastore_ephemeral.cells.fetch_all("I"), dtype=bool).copy()
    cell_ref = datastore_ephemeral.auto_filter_cells(attrs=attrs)
    after = np.asarray(datastore_ephemeral.cells.fetch_all("I"), dtype=bool)

    status = datastore_ephemeral.inspect_artifact(cell_ref)
    assert status.operation == "auto_filter_cells"
    assert "sample_column" not in status.parameters
    assert status.parameters["resolved_bounds"] == {
        attr: {
            "low": float(expected_bounds[attr]["low"]),
            "high": float(expected_bounds[attr]["high"]),
        }
        for attr in attrs
    }
    filtered = _selection_mask(datastore_ephemeral, cell_ref)
    assert filtered.sum() < before.sum()
    np.testing.assert_array_equal(after, before)


def test_auto_filter_cells_global_combines_metadata_and_exact_artifact_metrics(
    datastore_ephemeral,
):
    metadata_values = np.asarray(
        datastore_ephemeral.cells.fetch_all("RNA_nCounts"),
        dtype=float,
    )
    base_active = np.asarray(
        datastore_ephemeral.cells.fetch_all("I"),
        dtype=bool,
    )
    base_indices = np.flatnonzero(base_active)
    assert len(base_indices) > 4
    subset = base_active.copy()
    excluded_index = int(base_indices[0])
    subset[excluded_index] = False
    datastore_ephemeral.cells.insert("artifact_qc_subset", subset, overwrite=True)
    prior = datastore_ephemeral.snapshot_cell_selection("artifact_qc_subset")

    metadata_values[excluded_index] = 1e12
    datastore_ephemeral.cells.insert(
        "RNA_nCounts",
        metadata_values,
        overwrite=True,
    )
    selected_indices = np.flatnonzero(subset)
    selected_counts = metadata_values[selected_indices]
    artifact_values = np.linspace(1.0, 2.0, len(selected_indices))
    artifact_values[-1] = 50.0
    metric = _write_cell_vector(
        datastore_ephemeral,
        selection=prior,
        name="percentMito",
        kind="quality_metric",
        values=artifact_values,
        assay="RNA",
    )
    count_low, count_high = gaussian_quantile_bounds(selected_counts, 0.01, 0.99)
    metric_low, metric_high = gaussian_quantile_bounds(
        artifact_values,
        0.01,
        0.99,
    )
    expected_compact = (
        (selected_counts > count_low)
        & (selected_counts < count_high)
        & (artifact_values > metric_low)
        & (artifact_values < metric_high)
    )
    expected = np.zeros(datastore_ephemeral.cells.N, dtype=bool)
    expected[selected_indices] = expected_compact

    result = datastore_ephemeral.auto_filter_cells(
        attrs=["RNA_nCounts"],
        artifact_metrics=[metric],
        cell_selection=prior,
    )

    np.testing.assert_array_equal(
        _selection_mask(datastore_ephemeral, result), expected
    )
    status = datastore_ephemeral.inspect_artifact(result)
    assert status.parameters["resolved_bounds"]["RNA_nCounts"] == {
        "low": float(count_low),
        "high": float(count_high),
    }
    assert status.parameters["resolved_bounds"]["percentMito"] == {
        "low": float(metric_low),
        "high": float(metric_high),
    }
    assert status.inputs["artifact_metrics"] == {
        "percentMito": metric.artifact.to_dict()
    }
    assert status.parameters["metric_sources"] == [
        {
            "name": "RNA_nCounts",
            "source": "metadataColumn",
            "column": "RNA_nCounts",
        },
        {"name": "percentMito", "source": "artifact"},
    ]
    full_low, full_high = gaussian_quantile_bounds(metadata_values, 0.01, 0.99)
    assert (count_low, count_high) != pytest.approx((full_low, full_high))
    assert "percentMito" not in datastore_ephemeral.cells.columns


def test_auto_filter_cells_sample_column_raises_on_conflicts(
    datastore_ephemeral,
):
    n = datastore_ephemeral.cells.N
    datastore_ephemeral.cells.insert(
        "sample_id",
        np.array(["A"] * (n // 2) + ["B"] * (n - n // 2)),
        overwrite=True,
    )

    with pytest.raises(ValueError, match="min_p and max_p"):
        datastore_ephemeral.auto_filter_cells(
            attrs=["RNA_nCounts"],
            sample_column="sample_id",
            min_p=0.05,
        )

    with pytest.raises(ValueError, match="not found"):
        datastore_ephemeral.auto_filter_cells(
            attrs=["RNA_nCounts"],
            sample_column="missing_sample",
        )


@pytest.mark.parametrize("n_mads", [np.nan, np.inf, -np.inf])
def test_auto_filter_cells_rejects_nonfinite_n_mads_without_mutating_selection(
    datastore_ephemeral,
    n_mads,
):
    n = datastore_ephemeral.cells.N
    datastore_ephemeral.cells.insert(
        "sample_id",
        np.array(["A"] * (n // 2) + ["B"] * (n - n // 2)),
        overwrite=True,
    )
    before = np.asarray(
        datastore_ephemeral.cells.fetch_all("I"),
        dtype=bool,
    ).copy()

    with pytest.raises(ValueError, match="finite"):
        datastore_ephemeral.auto_filter_cells(
            attrs=["RNA_nCounts"],
            sample_column="sample_id",
            n_mads=n_mads,
            min_cells_per_sample=2,
        )

    np.testing.assert_array_equal(
        datastore_ephemeral.cells.fetch_all("I"),
        before,
    )


def test_auto_filter_cells_rejects_overflowing_finite_n_mads_without_mutation(
    datastore_ephemeral,
):
    n = datastore_ephemeral.cells.N
    datastore_ephemeral.cells.insert(
        "sample_id",
        np.array(["A"] * n),
        overwrite=True,
    )
    selection = datastore_ephemeral.zw["cellData"]["I"]
    selection_before = np.asarray(selection[:], dtype=bool).copy()
    provenance_before = dict(selection.attrs)

    with pytest.raises(ValueError, match="non-finite"):
        datastore_ephemeral.auto_filter_cells(
            attrs=["RNA_nCounts"],
            sample_column="sample_id",
            n_mads=np.finfo(float).max,
            min_cells_per_sample=2,
        )

    np.testing.assert_array_equal(selection[:], selection_before)
    assert dict(selection.attrs) == provenance_before


def test_auto_filter_cells_validates_provenance_before_selection_mutation(
    datastore_ephemeral,
    monkeypatch,
):
    import scarf.datastore._operations.quality_control as qc_operations

    n = datastore_ephemeral.cells.N
    datastore_ephemeral.cells.insert(
        "sample_id",
        np.array(["A"] * n),
        overwrite=True,
    )
    selection = datastore_ephemeral.zw["cellData"]["I"]
    selection_before = np.asarray(selection[:], dtype=bool).copy()
    provenance_before = dict(selection.attrs)

    def malformed_provenance(**kwargs):
        del kwargs
        return np.ones(n, dtype=bool), {
            "mad_scale": 1.4826,
            "metric_policies": {"RNA_nCounts": {"transform": object()}},
            "sample_sizes": {"A": n},
            "skip_reasons": {},
            "resolved_bounds": {},
            "warnings": [],
        }

    monkeypatch.setattr(
        qc_operations,
        "_sample_aware_mad_mask",
        malformed_provenance,
    )
    with pytest.raises(TypeError, match="Unsupported provenance value"):
        datastore_ephemeral.auto_filter_cells(
            attrs=["RNA_nCounts"],
            sample_column="sample_id",
            min_cells_per_sample=2,
        )

    np.testing.assert_array_equal(selection[:], selection_before)
    assert dict(selection.attrs) == provenance_before


@pytest.mark.parametrize("attr", ["RNA_nCounts", "RNA_nFeatures"])
def test_auto_filter_cells_rejects_negative_counts_without_selection_mutation(
    datastore_ephemeral,
    attr,
):
    n = datastore_ephemeral.cells.N
    datastore_ephemeral.cells.insert(
        "sample_id",
        np.array(["A"] * n),
        overwrite=True,
    )
    bad = np.asarray(datastore_ephemeral.cells.fetch_all(attr), dtype=float)
    active = np.asarray(datastore_ephemeral.cells.fetch_all("I"), dtype=bool)
    bad[int(np.flatnonzero(active)[0])] = -2.0
    datastore_ephemeral.cells.insert(attr, bad, overwrite=True)
    selection = datastore_ephemeral.zw["cellData"]["I"]
    selection_before = active.copy()
    provenance_before = dict(selection.attrs)

    with pytest.raises(ValueError, match="non-negative before log1p"):
        datastore_ephemeral.auto_filter_cells(
            attrs=[attr],
            sample_column="sample_id",
            min_cells_per_sample=2,
        )

    np.testing.assert_array_equal(
        datastore_ephemeral.cells.fetch_all("I"),
        selection_before,
    )
    assert dict(selection.attrs) == provenance_before


def test_auto_filter_cells_sample_column_raises_on_missing_and_nonfinite(
    datastore_ephemeral,
):
    n = datastore_ephemeral.cells.N
    labels = np.array(["A"] * n, dtype=object)
    labels[0] = ""
    datastore_ephemeral.cells.insert("sample_id", labels, overwrite=True)
    with pytest.raises(ValueError, match="missing labels"):
        datastore_ephemeral.auto_filter_cells(
            attrs=["RNA_nCounts"],
            sample_column="sample_id",
            min_cells_per_sample=2,
        )

    datastore_ephemeral.cells.insert(
        "sample_id",
        np.array(["A"] * n),
        overwrite=True,
    )
    bad = np.asarray(datastore_ephemeral.cells.fetch_all("RNA_nCounts"), dtype=float)
    bad[0] = np.nan
    datastore_ephemeral.cells.insert("RNA_nCounts", bad, overwrite=True)
    with pytest.raises(ValueError, match="non-finite"):
        datastore_ephemeral.auto_filter_cells(
            attrs=["RNA_nCounts"],
            sample_column="sample_id",
            min_cells_per_sample=2,
        )


def test_auto_filter_cells_sample_column_records_provenance(
    datastore_ephemeral,
):
    from scarf.storage.artifacts import fingerprint_array, fingerprint_strings

    n = datastore_ephemeral.cells.N
    labels = np.array(["A"] * (n - 5) + ["tiny"] * 5)
    datastore_ephemeral.cells.insert("sample_id", labels, overwrite=True)
    active = np.asarray(datastore_ephemeral.cells.fetch_all("I"), dtype=bool)
    prior_selection = datastore_ephemeral.snapshot_cell_selection("I")
    expected_sample_fingerprint = fingerprint_strings(labels[active])
    expected_metric_fingerprints = {
        attr: fingerprint_array(
            np.asarray(datastore_ephemeral.cells.fetch_all(attr), dtype=float)[active]
        )
        for attr in ("RNA_nCounts", "RNA_percentMito")
    }

    captured: list[str] = []
    sink = logger.add(
        lambda message: captured.append(message.record["message"]),
        level="WARNING",
    )
    try:
        cell_ref = datastore_ephemeral.auto_filter_cells(
            attrs=["RNA_nCounts", "RNA_percentMito"],
            cell_selection=prior_selection,
            sample_column="sample_id",
            n_mads=3.0,
            min_cells_per_sample=20,
        )
    finally:
        logger.remove(sink)

    status = datastore_ephemeral.inspect_artifact(cell_ref)
    assert status.operation == "auto_filter_cells"
    assert status.parameters["sample_column"] == "sample_id"
    assert status.parameters["n_mads"] == 3.0
    assert status.parameters["mad_scale"] == 1.4826
    assert status.parameters["skip_reasons"]["tiny"] == "insufficient_cells"
    assert status.parameters["metric_policies"]["RNA_nCounts"]["transform"] == "log1p"
    assert (
        status.parameters["metric_policies"]["RNA_percentMito"]["bound_direction"]
        == "upper"
    )
    assert status.inputs["prior_cell_selection"] == prior_selection.to_dict()
    assert (
        status.inputs["sample_assignments_fingerprint"] == expected_sample_fingerprint
    )
    assert status.inputs["qc_metric_fingerprints"] == expected_metric_fingerprints
    assert any("tiny" in message for message in captured)


def test_auto_filter_cells_sample_mad_combines_exact_metric_and_hto_artifacts(
    datastore_ephemeral,
):
    prior = datastore_ephemeral.snapshot_cell_selection("I")
    active = _selection_mask(datastore_ephemeral, prior)
    active_indices = np.flatnonzero(active)
    n_active = len(active_indices)
    assert n_active >= 10
    split = n_active // 2
    labels = np.asarray(["sample-a"] * split + ["sample-b"] * (n_active - split))
    percent_mito = np.concatenate(
        [
            np.linspace(1.0, 2.0, split),
            np.linspace(2.0, 3.0, n_active - split),
        ]
    )
    percent_mito[-1] = 80.0
    sample_source = _write_cell_vector(
        datastore_ephemeral,
        selection=prior,
        name="HTO_htoIdentity",
        kind="hto_identity",
        values=labels,
        assay="assay2",
    )
    metric_source = _write_cell_vector(
        datastore_ephemeral,
        selection=prior,
        name="percentMito",
        kind="quality_metric",
        values=percent_mito,
        assay="RNA",
    )
    counts = np.asarray(
        datastore_ephemeral.cells.fetch_all("RNA_nCounts"),
        dtype=float,
    )[active_indices]
    expected_compact, expected_provenance = _sample_aware_mad_mask(
        values_by_attr={
            "RNA_nCounts": counts,
            "percentMito": percent_mito,
        },
        sample_labels=labels,
        active=np.ones(n_active, dtype=bool),
        n_mads=3.0,
        min_cells_per_sample=5,
        attrs=["RNA_nCounts", "percentMito"],
    )
    expected = np.zeros(datastore_ephemeral.cells.N, dtype=bool)
    expected[active_indices] = expected_compact

    result = datastore_ephemeral.auto_filter_cells(
        attrs=["RNA_nCounts"],
        artifact_metrics=[metric_source],
        cell_selection=prior,
        sample_artifact=sample_source,
        n_mads=3.0,
        min_cells_per_sample=5,
    )

    np.testing.assert_array_equal(
        _selection_mask(datastore_ephemeral, result), expected
    )
    status = datastore_ephemeral.inspect_artifact(result)
    assert status.parameters["sample_source"] == {
        "name": "HTO_htoIdentity",
        "source": "artifact",
    }
    assert (
        status.parameters["metric_policies"] == expected_provenance["metric_policies"]
    )
    assert status.inputs["sample_artifact"] == sample_source.artifact.to_dict()
    assert status.inputs["artifact_metrics"] == {
        "percentMito": metric_source.artifact.to_dict()
    }
    assert "percentMito" not in datastore_ephemeral.cells.columns
    assert "HTO_htoIdentity" not in datastore_ephemeral.cells.columns


def test_pipeline_passes_sample_column_through(datastore_ephemeral):
    n = datastore_ephemeral.cells.N
    datastore_ephemeral.cells.insert(
        "sample_id",
        np.array(["A"] * (n // 2) + ["B"] * (n - n // 2)),
        overwrite=True,
    )
    run = datastore_ephemeral.pipeline.run(
        filtering={
            "sample_column": "sample_id",
            "n_mads": 3.0,
            "min_cells_per_sample": 20,
            "attrs": ["RNA_nCounts"],
        },
        cell_cycle=False,
        hvg_count=50,
        pca_dims=5,
        neighbors_k=3,
        umap=False,
        leiden=False,
        paris=False,
        doublets=False,
        markers=False,
    )
    status = datastore_ephemeral.inspect_artifact(run["analysis_cell_selection"])
    assert status.operation == "filter_pipeline_cells"
    assert status.parameters["sampleColumn"] == "sample_id"
    assert status.parameters["nMads"] == 3.0
    assert status.parameters["minCellsPerSample"] == 20
    assert status.inputs is not None
    assert set(status.inputs) == {
        "cell_snapshot",
        "input_cell_selection",
        "ordered_row_ids_fingerprint",
        "values_fingerprint",
    }

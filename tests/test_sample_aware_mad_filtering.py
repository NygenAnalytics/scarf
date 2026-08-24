import numpy as np
import pytest
from scarf.storage.artifacts import ArtifactRef
from scarf.utils import logger

from scarf.quality_control.filtering import (
    _mad_bounds,
    _metric_policy,
    _sample_aware_mad_mask,
    gaussian_quantile_bounds,
)


def test_mad_bounds_matches_hand_computed_example():
    values = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    median = 3.0
    scaled_mad = 1.4826 * np.median(np.abs(values - median))
    low, high, reported_mad = _mad_bounds(values, n_mads=3.0)

    assert reported_mad == pytest.approx(scaled_mad)
    assert low == pytest.approx(median - 3.0 * scaled_mad)
    assert high == pytest.approx(median + 3.0 * scaled_mad)


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
    datastore_ephemeral.auto_filter_cells(attrs=attrs, show_qc_plots=False)
    after = np.asarray(datastore_ephemeral.cells.fetch_all("I"), dtype=bool)

    cell_ref = ArtifactRef.from_dict(
        datastore_ephemeral.zw["cellData"]["I"].attrs["source_artifact"]
    )
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
    assert after.sum() < before.sum()


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
            show_qc_plots=False,
        )

    with pytest.raises(ValueError, match="not found"):
        datastore_ephemeral.auto_filter_cells(
            attrs=["RNA_nCounts"],
            sample_column="missing_sample",
            show_qc_plots=False,
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
            show_qc_plots=False,
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
            show_qc_plots=False,
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
            show_qc_plots=False,
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
            show_qc_plots=False,
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
            show_qc_plots=False,
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
            show_qc_plots=False,
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
    prior_selection = datastore_ephemeral._ensure_cell_selection("I")
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
        datastore_ephemeral.auto_filter_cells(
            attrs=["RNA_nCounts", "RNA_percentMito"],
            sample_column="sample_id",
            n_mads=3.0,
            min_cells_per_sample=20,
            show_qc_plots=False,
        )
    finally:
        logger.remove(sink)

    cell_ref = ArtifactRef.from_dict(
        datastore_ephemeral.zw["cellData"]["I"].attrs["source_artifact"]
    )
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


def test_pipeline_passes_sample_column_through(datastore_ephemeral):
    n = datastore_ephemeral.cells.N
    datastore_ephemeral.cells.insert(
        "sample_id",
        np.array(["A"] * (n // 2) + ["B"] * (n - n // 2)),
        overwrite=True,
    )
    artifacts = datastore_ephemeral.pipeline.run(
        pipeline_id="basic_rna_analysis",
        filtering={
            "sample_column": "sample_id",
            "n_mads": 3.0,
            "min_cells_per_sample": 20,
            "attrs": ["RNA_nCounts"],
        },
        cell_cycle_scoring=False,
        highly_variable_features={
            "top_n": 50,
            "hvg_key_name": "mad_pipeline_hvgs",
        },
        pca={"dims": 5, "n_centroids": 10},
        ann_index={"ann_m": 16},
        neighbors={"k": 3},
        connectivity={},
        umap=False,
        leiden={},
        paris=False,
        doublet_scoring=False,
        markers=False,
    )
    status = datastore_ephemeral.inspect_artifact(artifacts["cell_selection"])
    assert status.operation == "auto_filter_cells"
    assert status.parameters["sample_column"] == "sample_id"
    assert status.parameters["n_mads"] == 3.0

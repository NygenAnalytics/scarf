"""Pooled and per-sample MAD and Gaussian cell filtering against NumPy references."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from scarf import DataStore
from scarf.metadata.artifacts import (
    plan_cell_data_artifact,
    write_cell_data_artifact,
)
from scarf.metadata.selection import NamedCellArtifact
from scarf.quality_control.filtering import (
    _apply_bounds,
    _metric_policy,
    clamp_metric_bound,
    filter_cell_metrics,
    gaussian_quantile_bounds,
    mad_bounds,
    sample_aware_mad_mask,
    unique_label_keys,
    validate_cell_filter_sources,
)
from scarf.storage.artifacts import ArtifactRef, fingerprint_array, fingerprint_strings
from scarf.storage.selections import read_stored_selection_mask
from scarf.utils import logger
from tests.qc_helpers import (
    N_CELLS,
    open_small_store,
    reference_gaussian_bounds,
    reference_mad_bounds,
    reference_mad_keep,
    small_rna_counts,
    write_small_store,
)


@pytest.fixture(scope="module")
def mad_template(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("mad_template") / "store.zarr"
    write_small_store(path)
    return path


@pytest.fixture
def store(mad_template, tmp_path) -> DataStore:
    target = tmp_path / "store.zarr"
    shutil.copytree(mad_template, target)
    return open_small_store(target)


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
    assay: str = "RNA",
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


def _capture_warnings(action):
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]), level="WARNING"
    )
    try:
        result = action()
    finally:
        logger.remove(sink)
    return result, messages


def test_mad_bounds_matches_hand_computed_example():
    values = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    # Median 3; absolute deviations 2, 1, 0, 1, 97 have median 1.
    low, high, reported_mad = mad_bounds(values, n_mads=3.0)

    assert reported_mad == pytest.approx(1.4826)
    assert low == pytest.approx(3.0 - 3 * 1.4826)
    assert high == pytest.approx(3.0 + 3 * 1.4826)


@pytest.mark.parametrize("n_mads", [0.0, -1.0, np.nan, np.inf])
def test_mad_bounds_rejects_nonpositive_and_nonfinite_scales(n_mads):
    with pytest.raises(ValueError, match="n_mads must be finite and greater than 0"):
        mad_bounds(np.array([1.0, 2.0, 3.0]), n_mads)


def test_mad_bounds_rejects_nonfinite_values_and_overflowing_bounds():
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="MAD input values must all be finite"):
            mad_bounds(np.array([1.0, bad, 3.0]), 3.0)
    # A scaled MAD of 14.826 times the largest float overflows the distance.
    with pytest.raises(ValueError, match="n_mads produces non-finite MAD bounds"):
        mad_bounds(np.array([0.0, 10.0, 20.0]), np.finfo(float).max)


def test_metric_bounds_clamp_to_each_metric_range():
    assert clamp_metric_bound(-0.5, transform="log1p", is_percent=False) == 0.0
    assert clamp_metric_bound(7.5, transform="log1p", is_percent=False) == 7.5
    assert clamp_metric_bound(103.0, transform="identity", is_percent=True) == 100.0
    assert clamp_metric_bound(-2.0, transform="identity", is_percent=True) == 0.0
    assert clamp_metric_bound(-2.0, transform="identity", is_percent=False) == -2.0
    for bound in (np.inf, -np.inf, np.nan):
        with pytest.raises(ValueError, match="Resolved MAD bounds must be finite"):
            clamp_metric_bound(bound, transform="identity", is_percent=False)

    # Through the mask: a negative log-scale count bound and a percentage
    # bound above 100 are recorded at the edge of their range.
    counts = np.array([0.0, 0.0, 1.0, 1.0, 3.0, 3.0, 3.0])
    percent = np.array([90.0, 95.0, 99.0, 99.0, 100.0, 100.0, 100.0])
    _keep, provenance = sample_aware_mad_mask(
        values_by_attr={"RNA_nCounts": counts, "RNA_percentMito": percent},
        sample_labels=None,
        active=np.ones(7, dtype=bool),
        n_mads=3.0,
        min_cells_per_sample=2,
        attrs=["RNA_nCounts", "RNA_percentMito"],
    )
    bounds = provenance["resolved_bounds"]["all"]
    log_counts = np.log1p(counts)
    center = np.median(log_counts)
    spread = 1.4826 * np.median(np.abs(log_counts - center))
    assert np.expm1(center - 3 * spread) < 0
    assert bounds["RNA_nCounts"]["low"] == 0.0
    assert bounds["RNA_nCounts"]["high"] == pytest.approx(
        np.expm1(center + 3 * spread), rel=1e-12
    )
    assert 99.0 + 3 * 1.4826 > 100
    assert bounds["RNA_percentMito"]["low"] is None
    assert bounds["RNA_percentMito"]["high"] == 100.0


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


@pytest.mark.parametrize("dtype", [str, object])
def test_apply_bounds_compares_text_with_an_open_side(dtype):
    # An open side was compared as infinity, which text cannot be ordered against.
    values = np.array(["a", "b", "c"], dtype=dtype)

    np.testing.assert_array_equal(
        _apply_bounds(values, "a", None),
        [False, True, True],
    )
    np.testing.assert_array_equal(
        _apply_bounds(values, None, "b", keep_bounds=True),
        [True, True, False],
    )


def test_metric_policy_defaults_and_custom_attrs():
    for attr in ("RNA_nCounts", "nFeatures", "ADT_nCounts"):
        assert _metric_policy(attr) == {
            "transform": "log1p",
            "bound_direction": "two_sided",
        }
    for attr in ("RNA_percentMito", "percentRibo"):
        assert _metric_policy(attr) == {
            "transform": "identity",
            "bound_direction": "upper",
        }
    # Only a whole suffix after "_" names a QC metric.
    for attr in ("custom_score", "RNAnCounts", "percentMito_ratio"):
        assert _metric_policy(attr) == {
            "transform": "identity",
            "bound_direction": "two_sided",
        }


@pytest.mark.parametrize("attr", ["RNA_nCounts", "RNA_nFeatures"])
def test_sample_aware_mask_rejects_negative_count_values(attr):
    with pytest.raises(ValueError, match="non-negative before log1p"):
        sample_aware_mad_mask(
            values_by_attr={attr: np.array([1.0, 2.0, -2.0])},
            sample_labels=np.array(["A", "A", "A"]),
            active=np.ones(3, dtype=bool),
            n_mads=3.0,
            min_cells_per_sample=2,
            attrs=[attr],
        )


def test_sample_aware_mask_rejects_mixed_label_types_without_collision():
    with pytest.raises(ValueError, match="one consistent label type"):
        sample_aware_mad_mask(
            values_by_attr={"score": np.arange(8, dtype=float)},
            sample_labels=np.array([1] * 4 + ["1"] * 4, dtype=object),
            active=np.ones(8, dtype=bool),
            n_mads=3.0,
            min_cells_per_sample=2,
            attrs=["score"],
        )


def test_label_keys_decode_bytes_and_reject_collisions():
    assert unique_label_keys([b"A", np.int64(2), 1.5], label_name="Sample labels") == [
        "A",
        "2",
        "1.5",
    ]
    with pytest.raises(ValueError, match="Sample labels collide"):
        unique_label_keys([b"A", "A"], label_name="Sample labels")


def test_cell_filter_sources_reject_repeated_colliding_and_double_sources():
    metric = NamedCellArtifact(
        "RNA_percentMito", ArtifactRef("assay", "quality_metric", "a" * 64, "RNA")
    )
    sample = NamedCellArtifact(
        "sample", ArtifactRef("datastore", "hto_identity", "b" * 64)
    )
    assert validate_cell_filter_sources(
        ["RNA_nCounts"], [metric], sample_artifact=sample
    ) == (["RNA_nCounts"], [metric], sample)
    for arguments, error, message in (
        ({"attrs": [1]}, TypeError, "only column names"),
        ({"attrs": ["RNA_nCounts"] * 2}, ValueError, "duplicate columns"),
        (
            {"attrs": ["RNA_percentMito"], "artifact_metrics": [metric]},
            ValueError,
            "distinct names",
        ),
        (
            {
                "attrs": [],
                "artifact_metrics": [metric],
                "sample_artifact": NamedCellArtifact(
                    "RNA_percentMito", sample.artifact
                ),
            },
            ValueError,
            "Sample and metric artifact names",
        ),
        (
            {"attrs": [], "sample_column": "sample", "sample_artifact": sample},
            ValueError,
            "mutually exclusive",
        ),
        ({"attrs": [], "sample_artifact": metric}, ValueError, "hto_identity"),
        (
            {"attrs": [], "artifact_metrics": [metric.artifact]},
            TypeError,
            "^artifact_metrics must contain NamedCellArtifact values$",
        ),
        (
            {"attrs": [], "artifact_metrics": [metric, metric]},
            ValueError,
            "^artifact_metrics must use unique semantic names$",
        ),
    ):
        with pytest.raises(error, match=message):
            validate_cell_filter_sources(**arguments)


def test_sample_aware_mask_rejects_blank_bytes_labels():
    with pytest.raises(ValueError, match="missing labels"):
        sample_aware_mad_mask(
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

    keep, provenance = sample_aware_mad_mask(
        values_by_attr={"RNA_nCounts": n_counts},
        sample_labels=labels,
        active=active,
        n_mads=3.0,
        min_cells_per_sample=5,
        attrs=["RNA_nCounts"],
    )

    expected = reference_mad_keep({"RNA_nCounts": n_counts}, labels, min_cells=5)
    np.testing.assert_array_equal(keep, expected)
    np.testing.assert_array_equal(np.flatnonzero(~keep), [39])
    for label, values in (("A", sample_a), ("B", sample_b)):
        low, high = reference_mad_bounds(values, "RNA_nCounts")
        recorded = provenance["resolved_bounds"][label]["RNA_nCounts"]
        assert recorded["low"] == pytest.approx(low, rel=1e-12)
        assert recorded["high"] == pytest.approx(high, rel=1e-12)
        log_values = np.log1p(values)
        assert recorded["scaled_mad"] == pytest.approx(
            1.4826 * np.median(np.abs(log_values - np.median(log_values))),
            rel=1e-12,
        )
    assert provenance["sample_sizes"] == {"A": 20, "B": 20}
    assert provenance["skip_reasons"] == {}
    assert provenance["warnings"] == []


def test_sample_aware_mask_uses_log_counts_and_upper_only_percent():
    n_counts = np.expm1(
        np.array([2.0, 2.1, 1.9, 2.05, 2.02, 2.01, 1.98, 2.03, 1.99, 2.04, 8.0])
    )
    percent_mito = np.array(
        [1.0, 1.1, 0.9, 1.05, 1.02, 1.01, 0.98, 1.03, 0.99, 1.04, 40.0]
    )
    labels = np.array(["S"] * 11)
    active = np.ones(11, dtype=bool)
    values = {"RNA_nCounts": n_counts, "RNA_percentMito": percent_mito}

    keep, provenance = sample_aware_mad_mask(
        values_by_attr=values,
        sample_labels=labels,
        active=active,
        n_mads=3.0,
        min_cells_per_sample=5,
        attrs=["RNA_nCounts", "RNA_percentMito"],
    )

    np.testing.assert_array_equal(keep, reference_mad_keep(values, min_cells=5))
    assert not bool(keep[-1])
    count_bounds = provenance["resolved_bounds"]["S"]["RNA_nCounts"]
    mito_bounds = provenance["resolved_bounds"]["S"]["RNA_percentMito"]
    assert count_bounds["transform"] == "log1p"
    assert count_bounds["bound_direction"] == "two_sided"
    low, high = reference_mad_bounds(n_counts, "RNA_nCounts")
    assert count_bounds["low"] == pytest.approx(low, rel=1e-12)
    assert count_bounds["high"] == pytest.approx(high, rel=1e-12)
    assert mito_bounds["transform"] == "identity"
    assert mito_bounds["bound_direction"] == "upper"
    assert mito_bounds["low"] is None
    assert mito_bounds["high"] == pytest.approx(
        reference_mad_bounds(percent_mito, "RNA_percentMito")[1], rel=1e-12
    )
    # An unusually low mito percentage must not be filtered by upper-only bounds.
    low_mito = percent_mito.copy()
    low_mito[0] = 0.0
    keep_low, _ = sample_aware_mad_mask(
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
    np.testing.assert_array_equal(np.flatnonzero(~keep_low), [10])


def test_sample_aware_mask_skips_small_and_zero_mad_groups():
    values = np.array([1.0, 2.0, 3.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    labels = np.array(["tiny"] * 3 + ["flat"] * 5)
    active = np.ones(8, dtype=bool)

    keep, provenance = sample_aware_mad_mask(
        values_by_attr={"custom_score": values},
        sample_labels=labels,
        active=active,
        n_mads=3.0,
        min_cells_per_sample=5,
        attrs=["custom_score"],
    )

    assert bool(keep.all())
    assert provenance["sample_sizes"] == {"tiny": 3, "flat": 5}
    assert provenance["skip_reasons"] == {"tiny": "insufficient_cells"}
    assert provenance["resolved_bounds"] == {
        "tiny": {},
        "flat": {
            "custom_score": {
                "low": None,
                "high": None,
                "skip_reason": "zero_mad",
                "transform": "identity",
                "bound_direction": "two_sided",
                "scaled_mad": 0.0,
            }
        },
    }
    assert provenance["warnings"] == [
        "Sample 'tiny': fewer than 5 active cells; retaining them without MAD "
        "filtering",
        "Sample 'flat': zero MAD for 'custom_score'; retaining cells for this metric",
    ]


def _labelled_filter(labels, active=None, values=None):
    labels = np.asarray(labels)
    n_cells = len(labels)
    return filter_cell_metrics(
        {"score": np.arange(n_cells, dtype=float) if values is None else values},
        {},
        np.ones(n_cells, dtype=bool) if active is None else active,
        method="mad",
        sample_labels=labels,
        min_cells_per_sample=2,
    )


@pytest.mark.parametrize(
    ("labels", "keys", "kind"),
    [
        (np.array([True] * 4 + [False] * 4), ["True", "False"], bool),
        (np.array([1.5] * 4 + [-2.0] * 4), ["1.5", "-2.0"], float),
        (
            np.array([np.int64(7)] * 4 + [np.int64(3)] * 4, dtype=object),
            ["7", "3"],
            int,
        ),
        (np.array([b"a"] * 4 + [b"b"] * 4), ["a", "b"], bytes),
    ],
    ids=["bool", "float", "object_numpy_ints", "bytes"],
)
def test_mad_sample_labels_of_each_kind_group_cells(labels, keys, kind):
    values = np.array([1.0, 2.0, 3.0, 40.0, 10.0, 11.0, 12.0, -50.0])

    result = _labelled_filter(labels, values=values)

    assert result.mad_provenance["sample_sizes"] == dict.fromkeys(keys, 4)
    np.testing.assert_array_equal(
        result.retained,
        reference_mad_keep({"score": values}, labels, min_cells=2),
    )
    np.testing.assert_array_equal(np.flatnonzero(~result.retained), [3, 7])
    # Validated labels are Python scalars, never NumPy scalars.
    assert {type(value) for value in result.sample_labels} == {kind}


@pytest.mark.parametrize(
    ("labels", "error", "message"),
    [
        (
            np.array([1.0, 1.0, np.nan, 2.0, 2.0, 2.0]),
            ValueError,
            "^sample labels contains missing labels among active cells$",
        ),
        (
            np.array([None, "a", "a", "b", "b", "b"], dtype=object),
            ValueError,
            "^sample labels contains missing labels among active cells$",
        ),
        (
            np.array([1.0, 1.0, np.inf, 2.0, 2.0, 2.0]),
            ValueError,
            "^sample labels must contain finite labels$",
        ),
        (
            np.array([b"\xff"] * 3 + [b"a"] * 3),
            ValueError,
            "^sample labels contains a non-UTF-8 bytes label$",
        ),
        (
            np.array([1j] * 3 + [2j] * 3),
            TypeError,
            "^sample labels contains unsupported label type 'complex'$",
        ),
        (
            np.array(["a", "b", "a", "b", "a"]),
            ValueError,
            "^Sample labels and active selection must be aligned vectors$",
        ),
    ],
    ids=["nan", "none", "infinite", "non_utf8", "complex", "misaligned"],
)
def test_mad_sample_labels_reject_missing_and_unsupported_values(
    labels, error, message
):
    with pytest.raises(error, match=message):
        filter_cell_metrics(
            {"score": np.arange(6, dtype=float)},
            {},
            np.ones(6, dtype=bool),
            method="mad",
            sample_labels=labels,
            min_cells_per_sample=2,
        )


def test_mad_ignores_labels_and_metrics_of_inactive_cells():
    labels = np.array([np.nan, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    values = np.array([np.inf, 1.0, 2.0, 30.0, 5.0, 6.0, 7.0])
    active = np.array([False, True, True, True, True, True, True])

    result = filter_cell_metrics(
        {"score": values},
        {},
        active,
        method="mad",
        sample_labels=labels,
        min_cells_per_sample=3,
    )

    assert result.mad_provenance["sample_sizes"] == {"1.0": 3, "2.0": 3}
    np.testing.assert_array_equal(
        result.retained,
        [False, *reference_mad_keep({"score": values[1:]}, labels[1:], min_cells=3)],
    )


def test_mad_rejects_nonfinite_metrics_that_skip_the_float_check():
    # Object-typed metrics skip the floating-point screen of selected cells,
    # so the MAD work-scale check reports them instead.
    values = np.array([1.0, np.nan, 2.0, 3.0], dtype=object)

    with pytest.raises(
        ValueError, match="^QC values in 'score' contain non-finite entries$"
    ):
        filter_cell_metrics({"score": values}, {}, np.ones(4, dtype=bool), method="mad")


def test_sample_mad_excludes_masked_metric_rows_from_bounds_and_selection():
    labels = np.array(["A"] * 6 + ["B"] * 6)
    values = np.array(
        [10.0, 11.0, 12.0, 13.0, 1000.0, 50.0, 1.0, 1.1, 1.2, 1.3, 9.0, 0.0]
    )
    missing = np.zeros(12, dtype=bool)
    # A masked placeholder would widen A's bounds and pass B's if counted.
    missing[[4, 11]] = True

    result = filter_cell_metrics(
        {"score": values},
        {"score": missing},
        np.ones(12, dtype=bool),
        method="mad",
        sample_labels=labels,
        min_cells_per_sample=3,
    )

    complete = ~missing
    expected = np.zeros(12, dtype=bool)
    expected[complete] = reference_mad_keep(
        {"score": values[complete]}, labels[complete], min_cells=3
    )
    np.testing.assert_array_equal(result.retained, expected)
    assert not result.retained[missing].any()
    assert result.mad_provenance["sample_sizes"] == {"A": 5, "B": 5}
    np.testing.assert_array_equal(np.flatnonzero(~result.retained), [4, 5, 10, 11])


@pytest.mark.parametrize("n_cells", [15, 19, 20])
def test_default_pooled_mad_retains_small_selections_with_a_warning(store, n_cells):
    before = store.cells.fetch_all("I").copy()
    active = np.arange(store.cells.N) < n_cells
    values = np.zeros(store.cells.N)
    values[:n_cells] = np.r_[np.linspace(1, 2, n_cells - 1), 99.0]
    store.cells.insert("boundary_score", values)
    store.cells.insert("small_selection", active)
    selection = store.snapshot_cell_selection("small_selection")

    result, messages = _capture_warnings(
        lambda: store.auto_filter_cells(
            attrs=["boundary_score"], cell_selection=selection
        )
    )

    expected = active.copy()
    if n_cells >= 20:
        expected[n_cells - 1] = False
    expected_compact = reference_mad_keep({"boundary_score": values[:n_cells]})
    np.testing.assert_array_equal(expected[:n_cells], expected_compact)
    np.testing.assert_array_equal(_selection_mask(store, result), expected)
    np.testing.assert_array_equal(store.cells.fetch_all("I"), before)
    parameters = store.inspect_artifact(result).parameters
    assert parameters["method"] == "mad"
    assert parameters["sample_source"] == {"source": "pooled"}
    assert parameters["sample_sizes"] == {"all": n_cells}
    if n_cells < 20:
        assert parameters["skip_reasons"] == {"all": "insufficient_cells"}
        assert messages == [
            "Selected cells: fewer than 20 active cells; retaining them without "
            "MAD filtering"
        ]
    else:
        assert parameters["skip_reasons"] == {}
        assert messages == []


def test_auto_filter_cells_explicit_gaussian_matches_quantile_bounds(store):
    attrs = ["RNA_nCounts", "RNA_nFeatures"]
    counts = small_rna_counts()
    metrics = {
        "RNA_nCounts": counts.sum(axis=1).astype(float),
        "RNA_nFeatures": (counts > 0).sum(axis=1).astype(float),
    }
    for attr in attrs:
        np.testing.assert_array_equal(store.cells.fetch_all(attr), metrics[attr])
    expected_bounds = {attr: reference_gaussian_bounds(metrics[attr]) for attr in attrs}
    expected = np.ones(N_CELLS, dtype=bool)
    for attr, (low, high) in expected_bounds.items():
        expected &= (metrics[attr] > low) & (metrics[attr] < high)

    before = np.asarray(store.cells.fetch_all("I"), dtype=bool).copy()
    cell_ref = store.auto_filter_cells(attrs=attrs, method="gaussian")

    status = store.inspect_artifact(cell_ref)
    assert status.operation == "auto_filter_cells"
    assert "sample_column" not in status.parameters
    for attr, (low, high) in expected_bounds.items():
        recorded = status.parameters["resolved_bounds"][attr]
        assert recorded == {
            "low": pytest.approx(low, rel=1e-12),
            "high": pytest.approx(high, rel=1e-12),
        }
    filtered = _selection_mask(store, cell_ref)
    np.testing.assert_array_equal(filtered, expected)
    # The nearly empty and the very deep cell both fall outside a bound.
    assert not filtered[0] and not filtered[1]
    np.testing.assert_array_equal(store.cells.fetch_all("I"), before)


def test_auto_filter_cells_global_combines_metadata_and_exact_artifact_metrics(
    store,
):
    metadata_values = np.asarray(store.cells.fetch_all("RNA_nCounts"), dtype=float)
    base_active = np.asarray(store.cells.fetch_all("I"), dtype=bool)
    base_indices = np.flatnonzero(base_active)
    subset = base_active.copy()
    excluded_index = int(base_indices[0])
    subset[excluded_index] = False
    store.cells.insert("artifact_qc_subset", subset, overwrite=True)
    prior = store.snapshot_cell_selection("artifact_qc_subset")

    metadata_values[excluded_index] = 1e12
    store.zw["cellData"].create_array(
        "RNA_nCounts", data=metadata_values, overwrite=True
    )
    selected_indices = np.flatnonzero(subset)
    selected_counts = metadata_values[selected_indices]
    artifact_values = np.linspace(1.0, 2.0, len(selected_indices))
    artifact_values[-1] = 50.0
    metric = _write_cell_vector(
        store,
        selection=prior,
        name="percentMito",
        kind="quality_metric",
        values=artifact_values,
    )
    count_low, count_high = reference_gaussian_bounds(selected_counts)
    metric_low, metric_high = reference_gaussian_bounds(artifact_values)
    expected_compact = (
        (selected_counts > count_low)
        & (selected_counts < count_high)
        & (artifact_values > metric_low)
        & (artifact_values < metric_high)
    )
    assert not expected_compact[-1]
    expected = np.zeros(store.cells.N, dtype=bool)
    expected[selected_indices] = expected_compact

    result = store.auto_filter_cells(
        attrs=["RNA_nCounts"],
        artifact_metrics=[metric],
        cell_selection=prior,
        method="gaussian",
    )

    np.testing.assert_array_equal(_selection_mask(store, result), expected)
    status = store.inspect_artifact(result)
    assert status.parameters["resolved_bounds"]["RNA_nCounts"] == {
        "low": pytest.approx(count_low, rel=1e-12),
        "high": pytest.approx(count_high, rel=1e-12),
    }
    assert status.parameters["resolved_bounds"]["percentMito"] == {
        "low": pytest.approx(metric_low, rel=1e-12),
        "high": pytest.approx(metric_high, rel=1e-12),
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
    # The excluded cell's huge count would have moved the bounds.
    full_low, full_high = reference_gaussian_bounds(metadata_values[base_active])
    assert (count_low, count_high) != pytest.approx((full_low, full_high))
    assert "percentMito" not in store.cells.columns


def test_auto_filter_cells_sample_column_raises_on_conflicts(store):
    n = store.cells.N
    store.cells.insert(
        "sample_id",
        np.array(["A"] * (n // 2) + ["B"] * (n - n // 2)),
        overwrite=True,
    )

    with pytest.raises(ValueError, match="min_p and max_p"):
        store.auto_filter_cells(
            attrs=["RNA_nCounts"],
            sample_column="sample_id",
            min_p=0.05,
        )

    with pytest.raises(ValueError, match="not found"):
        store.auto_filter_cells(
            attrs=["RNA_nCounts"],
            sample_column="missing_sample",
        )

    for options, message in (
        ({"method": "unknown"}, "method must be"),
        ({"min_p": 0.05}, "min_p and max_p"),
        ({"method": "gaussian", "sample_column": "sample_id"}, "sample source"),
        ({"method": "gaussian", "n_mads": 4.0}, "apply only"),
        ({"method": "gaussian", "min_cells_per_sample": 2}, "apply only"),
    ):
        with pytest.raises(ValueError, match=message):
            store.auto_filter_cells(attrs=["RNA_nCounts"], **options)


def test_auto_filter_cells_rejects_an_empty_selection_without_mutating_live_cells(
    store,
):
    before = store.cells.fetch_all("I").copy()
    store.cells.insert("empty_selection", np.zeros(store.cells.N, dtype=bool))
    selection = store.snapshot_cell_selection("empty_selection")
    with pytest.raises(ValueError, match="Cell selection contains no active cells"):
        store.auto_filter_cells(attrs=["RNA_nCounts"], cell_selection=selection)
    np.testing.assert_array_equal(store.cells.fetch_all("I"), before)
    assert not _selection_mask(store, selection).any()


def _no_filter_written(store: DataStore) -> bool:
    return (
        store.list_artifacts(
            scope="datastore", kind="cell_selection", operation="auto_filter_cells"
        )
        == []
    )


@pytest.mark.parametrize("n_mads", [np.nan, np.inf, -np.inf])
def test_auto_filter_cells_rejects_nonfinite_n_mads_without_mutating_selection(
    store,
    n_mads,
):
    n = store.cells.N
    store.cells.insert(
        "sample_id",
        np.array(["A"] * (n // 2) + ["B"] * (n - n // 2)),
        overwrite=True,
    )
    before = np.asarray(store.cells.fetch_all("I"), dtype=bool).copy()

    with pytest.raises(ValueError, match="n_mads must be finite and greater than 0"):
        store.auto_filter_cells(
            attrs=["RNA_nCounts"],
            sample_column="sample_id",
            n_mads=n_mads,
            min_cells_per_sample=2,
        )

    np.testing.assert_array_equal(store.cells.fetch_all("I"), before)
    assert _no_filter_written(store)


def test_auto_filter_cells_rejects_overflowing_finite_n_mads_without_mutation(store):
    n = store.cells.N
    store.cells.insert("sample_id", np.array(["A"] * n), overwrite=True)
    selection = store.zw["cellData"]["I"]
    selection_before = np.asarray(selection[:], dtype=bool).copy()
    provenance_before = dict(selection.attrs)

    with pytest.raises(
        ValueError,
        match="MAD bound is non-finite after converting from the log1p scale",
    ):
        store.auto_filter_cells(
            attrs=["RNA_nCounts"],
            sample_column="sample_id",
            n_mads=np.finfo(float).max,
            min_cells_per_sample=2,
        )

    np.testing.assert_array_equal(selection[:], selection_before)
    assert dict(selection.attrs) == provenance_before
    assert _no_filter_written(store)


def test_auto_filter_cells_validates_provenance_before_selection_mutation(
    store,
    monkeypatch,
):
    import scarf.quality_control.filtering as qc_filtering

    n = store.cells.N
    store.cells.insert("sample_id", np.array(["A"] * n), overwrite=True)
    selection = store.zw["cellData"]["I"]
    selection_before = np.asarray(selection[:], dtype=bool).copy()
    provenance_before = dict(selection.attrs)

    def malformed_provenance(**kwargs):
        return np.ones_like(kwargs["active"], dtype=bool), {
            "mad_scale": 1.4826,
            "metric_policies": {"RNA_nCounts": {"transform": object()}},
            "sample_sizes": {"A": n},
            "skip_reasons": {},
            "resolved_bounds": {},
            "warnings": [],
        }

    monkeypatch.setattr(
        qc_filtering,
        "sample_aware_mad_mask",
        malformed_provenance,
    )
    with pytest.raises(TypeError, match="Unsupported provenance value"):
        store.auto_filter_cells(
            attrs=["RNA_nCounts"],
            sample_column="sample_id",
            min_cells_per_sample=2,
        )

    np.testing.assert_array_equal(selection[:], selection_before)
    assert dict(selection.attrs) == provenance_before
    assert _no_filter_written(store)


@pytest.mark.parametrize("attr", ["RNA_nCounts", "RNA_nFeatures"])
def test_auto_filter_cells_rejects_negative_counts_without_selection_mutation(
    store,
    attr,
):
    n = store.cells.N
    store.cells.insert("sample_id", np.array(["A"] * n), overwrite=True)
    bad = np.asarray(store.cells.fetch_all(attr), dtype=float)
    active = np.asarray(store.cells.fetch_all("I"), dtype=bool)
    bad[int(np.flatnonzero(active)[0])] = -2.0
    store.zw["cellData"].create_array(attr, data=bad, overwrite=True)
    selection = store.zw["cellData"]["I"]
    provenance_before = dict(selection.attrs)

    with pytest.raises(
        ValueError, match=f"^QC values in '{attr}' must be non-negative before log1p$"
    ):
        store.auto_filter_cells(
            attrs=[attr],
            sample_column="sample_id",
            min_cells_per_sample=2,
        )

    np.testing.assert_array_equal(store.cells.fetch_all("I"), active)
    assert dict(selection.attrs) == provenance_before
    assert _no_filter_written(store)


def test_auto_filter_cells_sample_column_raises_on_missing_and_nonfinite(store):
    n = store.cells.N
    labels = np.array(["A"] * n, dtype=object)
    labels[0] = ""
    store.cells.insert("sample_id", labels, overwrite=True)
    with pytest.raises(
        ValueError,
        match="^sample_column 'sample_id' contains missing labels among active cells$",
    ):
        store.auto_filter_cells(
            attrs=["RNA_nCounts"],
            sample_column="sample_id",
            min_cells_per_sample=2,
        )

    store.cells.insert("sample_id", np.array(["A"] * n), overwrite=True)
    bad = np.asarray(store.cells.fetch_all("RNA_nCounts"), dtype=float)
    bad[0] = np.nan
    store.zw["cellData"].create_array("RNA_nCounts", data=bad, overwrite=True)
    with pytest.raises(
        ValueError,
        match=rf"QC metric 'RNA_nCounts' has 1 non-finite value\(s\) among {n} "
        "selected cells",
    ):
        store.auto_filter_cells(
            attrs=["RNA_nCounts"],
            sample_column="sample_id",
            min_cells_per_sample=2,
        )
    assert _no_filter_written(store)


def test_auto_filter_cells_sample_column_records_provenance(store):
    n = store.cells.N
    labels = np.array(["A"] * (n - 5) + ["tiny"] * 5)
    store.cells.insert("sample_id", labels, overwrite=True)
    percent_mito = np.random.default_rng(3).uniform(1.0, 5.0, n)
    percent_mito[[7, n - 1]] = 60.0
    store.cells.insert("RNA_percentMito", percent_mito)
    active = np.asarray(store.cells.fetch_all("I"), dtype=bool)
    prior_selection = store.snapshot_cell_selection("I")
    metrics = {
        "RNA_nCounts": np.asarray(store.cells.fetch_all("RNA_nCounts"), dtype=float),
        "RNA_percentMito": percent_mito,
    }

    cell_ref, captured = _capture_warnings(
        lambda: store.auto_filter_cells(
            attrs=["RNA_nCounts", "RNA_percentMito"],
            cell_selection=prior_selection,
            sample_column="sample_id",
            n_mads=3.0,
            min_cells_per_sample=20,
        )
    )

    expected = reference_mad_keep(metrics, labels, min_cells=20)
    # Cell 7 has an outlying percentage; the tiny sample keeps its outlier.
    assert not expected[7] and expected[n - 1]
    np.testing.assert_array_equal(_selection_mask(store, cell_ref), expected & active)
    status = store.inspect_artifact(cell_ref)
    assert status.operation == "auto_filter_cells"
    assert status.parameters["sample_column"] == "sample_id"
    assert status.parameters["n_mads"] == 3.0
    assert status.parameters["mad_scale"] == 1.4826
    assert status.parameters["sample_sizes"] == {"A": n - 5, "tiny": 5}
    assert status.parameters["skip_reasons"] == {"tiny": "insufficient_cells"}
    assert status.parameters["metric_policies"]["RNA_nCounts"]["transform"] == "log1p"
    assert (
        status.parameters["metric_policies"]["RNA_percentMito"]["bound_direction"]
        == "upper"
    )
    a_rows = labels == "A"
    for attr, values in metrics.items():
        low, high = reference_mad_bounds(values[a_rows], attr)
        recorded = status.parameters["resolved_bounds"]["A"][attr]
        assert recorded["low"] == (None if low is None else pytest.approx(low))
        assert recorded["high"] == pytest.approx(high)
    assert status.inputs["prior_cell_selection"] == prior_selection.to_dict()
    assert status.inputs["sample_assignments_fingerprint"] == fingerprint_strings(
        labels[active]
    )
    assert status.inputs["qc_metric_fingerprints"] == {
        attr: fingerprint_array(values[active]) for attr, values in metrics.items()
    }
    assert captured == [
        "Sample 'tiny': fewer than 20 active cells; retaining them without MAD "
        "filtering"
    ]


def test_auto_filter_cells_sample_mad_combines_exact_metric_and_hto_artifacts(store):
    prior = store.snapshot_cell_selection("I")
    active = _selection_mask(store, prior)
    active_indices = np.flatnonzero(active)
    n_active = len(active_indices)
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
        store,
        selection=prior,
        name="HTO_htoIdentity",
        kind="hto_identity",
        values=labels,
    )
    metric_source = _write_cell_vector(
        store,
        selection=prior,
        name="percentMito",
        kind="quality_metric",
        values=percent_mito,
    )
    counts = np.asarray(store.cells.fetch_all("RNA_nCounts"), dtype=float)[
        active_indices
    ]
    expected_compact = reference_mad_keep(
        {"RNA_nCounts": counts, "percentMito": percent_mito},
        labels,
        min_cells=5,
    )
    assert not expected_compact[-1]
    expected = np.zeros(store.cells.N, dtype=bool)
    expected[active_indices] = expected_compact

    result = store.auto_filter_cells(
        attrs=["RNA_nCounts"],
        artifact_metrics=[metric_source],
        cell_selection=prior,
        sample_artifact=sample_source,
        n_mads=3.0,
        min_cells_per_sample=5,
    )

    np.testing.assert_array_equal(_selection_mask(store, result), expected)
    status = store.inspect_artifact(result)
    assert status.parameters["sample_source"] == {
        "name": "HTO_htoIdentity",
        "source": "artifact",
    }
    assert status.parameters["metric_policies"] == {
        "RNA_nCounts": {"transform": "log1p", "bound_direction": "two_sided"},
        "percentMito": {"transform": "identity", "bound_direction": "upper"},
    }
    assert status.parameters["sample_sizes"] == {
        "sample-a": split,
        "sample-b": n_active - split,
    }
    assert status.inputs["sample_artifact"] == sample_source.artifact.to_dict()
    assert status.inputs["artifact_metrics"] == {
        "percentMito": metric_source.artifact.to_dict()
    }
    assert "percentMito" not in store.cells.columns
    assert "HTO_htoIdentity" not in store.cells.columns


def test_pipeline_passes_sample_column_through(store):
    n = store.cells.N
    labels = np.array(["A"] * (n // 2) + ["B"] * (n - n // 2))
    store.cells.insert("sample_id", labels, overwrite=True)
    counts = np.asarray(store.cells.fetch_all("RNA_nCounts"), dtype=float)

    run = store.pipeline.run(
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

    selection = run["analysis_cell_selection"]
    expected = reference_mad_keep({"RNA_nCounts": counts}, labels, min_cells=20)
    # The nearly empty and the very deep cell both leave sample A.
    assert not expected[0] and not expected[1]
    np.testing.assert_array_equal(_selection_mask(store, selection), expected)
    status = store.inspect_artifact(selection)
    assert status.operation == "filter_pipeline_cells"
    assert status.parameters["sampleColumn"] == "sample_id"
    assert status.parameters["nMads"] == 3.0
    assert status.parameters["minCellsPerSample"] == 20
    assert status.parameters["mad"]["sample_sizes"] == {
        "A": n // 2,
        "B": n - n // 2,
    }
    assert status.inputs is not None
    assert set(status.inputs) == {
        "cell_snapshot",
        "input_cell_selection",
        "ordered_row_ids_fingerprint",
        "values_fingerprint",
    }


def test_gaussian_quantile_bounds_match_a_normal_reference():
    values = np.array([3.0, 7.0, 1.0, 9.0, 4.0, 4.0])

    for min_p, max_p in ((0.01, 0.99), (0.2, 0.7)):
        assert gaussian_quantile_bounds(values, min_p, max_p) == pytest.approx(
            reference_gaussian_bounds(values, min_p, max_p), rel=1e-12
        )

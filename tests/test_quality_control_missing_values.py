"""Missing-value masks, bound validation, and guards in quality control."""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scarf import DataStore
from scarf.quality_control.filtering import filter_cell_metrics
from scarf.datastore.pipeline_accessor import PipelineExecutionError
from scarf.storage.artifacts import ArtifactRef, fingerprint_array, fingerprint_strings
from scarf.storage.selections import read_stored_selection_mask
from tests.fixtures_datastore import build_neighbourhood_graph
from tests.storage_helpers import insert_nullable_cell_column
from tests.qc_helpers import (
    create_labelled_qc_store,
    open_qc_store,
    open_small_store,
    fresh_qc_store,
    reference_gaussian_bounds,
    reference_mad_keep,
    write_small_store,
)


def _selection_mask(store: Any, ref: ArtifactRef) -> np.ndarray:
    return read_stored_selection_mask(
        store.zw,
        ref,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )


def _run_options(filtering: dict[str, Any]) -> dict[str, Any]:
    """Pipeline options that run the always-on stages at their smallest size."""
    return {
        "filtering": filtering,
        "cell_cycle": False,
        "hvg_count": 50,
        "pca_dims": 3,
        "neighbors_k": 3,
        "umap": False,
        "leiden": False,
        "paris": False,
        "doublets": False,
        "markers": False,
    }


@pytest.fixture(scope="module")
def small_template(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("qc_missing_template") / "store.zarr"
    write_small_store(path)
    return path


@pytest.fixture
def small_store(small_template, tmp_path) -> DataStore:
    target = tmp_path / "store.zarr"
    shutil.copytree(small_template, target)
    return open_small_store(target)


def import_nullable_cluster_h5ad(
    directory: Path,
    *,
    n_cells: int = 200,
    n_genes: int = 60,
) -> tuple[DataStore, Any, np.ndarray, np.ndarray]:
    """Import an AnnData file whose nullable Int64 clusters contain NA.

    Returns the opened store, the import result, the cluster codes, and the
    rows whose cluster is NA.
    """
    anndata = pytest.importorskip("anndata")
    from scipy.sparse import csr_matrix

    from scarf.readers import H5adReader
    from scarf.writers import H5adToZarr

    rng = np.random.default_rng(7)
    counts = rng.poisson(3.0, size=(n_cells, n_genes)).astype(np.float32)
    # Keep counts wider than uint8, whose library-size scaling overflows.
    counts[0, 0] = 300
    codes = np.arange(n_cells) % 3
    missing = np.zeros(n_cells, dtype=bool)
    missing[::5] = True
    clusters = pd.array(codes, dtype="Int64")
    clusters[missing] = pd.NA
    obs = pd.DataFrame(
        {"clusters": clusters},
        index=[f"c{index}" for index in range(n_cells)],
    )
    var = pd.DataFrame(
        {"gene_short_name": [f"G{index}" for index in range(n_genes)]},
        index=[f"f{index}" for index in range(n_genes)],
    )
    source = directory / "nullable_clusters.h5ad"
    anndata.AnnData(X=csr_matrix(counts), obs=obs, var=var).write_h5ad(source)
    reader = H5adReader(str(source), cluster_keys=("clusters",))
    try:
        result = H5adToZarr(
            reader,
            zarr_loc=str(directory / "nullable_clusters.zarr"),
            nthreads=1,
        ).dump()
    finally:
        reader.h5.close()
    store = DataStore(
        str(directory / "nullable_clusters.zarr"),
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=1,
    )
    return store, result, codes, missing


def _imported_graph(store: DataStore, cells: ArtifactRef) -> ArtifactRef:
    features = store.select_hvgs(
        cells,
        top_n=40,
        show_plot=False,
        min_cells=5,
        max_cells=np.inf,
        blacklist="",
    )
    return build_neighbourhood_graph(
        store,
        cell_selection=cells,
        features=features,
        dims=5,
        k=5,
        local_cache=False,
    )


@dataclass(frozen=True)
class _NullableStore:
    store: DataStore
    cells: ArtifactRef
    clusters: ArtifactRef
    codes: np.ndarray
    missing: np.ndarray
    graph: ArtifactRef
    complete: ArtifactRef


@pytest.fixture(scope="module")
def nullable_template(tmp_path_factory) -> tuple[Path, _NullableStore]:
    """Import the nullable clusters once, with a graph and complete clusters."""
    directory = tmp_path_factory.mktemp("nullable_clusters")
    store, result, codes, missing = import_nullable_cluster_h5ad(directory)
    graph = _imported_graph(store, result.cellSelection)
    return directory / "nullable_clusters.zarr", _NullableStore(
        store=store,
        cells=result.cellSelection,
        clusters=result.clusterArtifacts["clusters"],
        codes=codes,
        missing=missing,
        graph=graph,
        complete=store.run_leiden_clustering(graph),
    )


@pytest.fixture
def nullable(nullable_template, tmp_path) -> _NullableStore:
    template, refs = nullable_template
    target = tmp_path / "nullable_clusters.zarr"
    shutil.copytree(template, target)
    store = DataStore(
        str(target), default_assay="RNA", min_features_per_cell=0, nthreads=1
    )
    return _NullableStore(
        store=store,
        cells=refs.cells,
        clusters=refs.clusters,
        codes=refs.codes,
        missing=refs.missing,
        graph=refs.graph,
        complete=refs.complete,
    )


def test_integer_qc_metrics_exclude_missing_values_from_gaussian_bounds():
    result = filter_cell_metrics(
        {"counts": np.array([1, 2, 3, 1000])},
        {"counts": np.array([False, False, False, True])},
        np.ones(4, dtype=bool),
        method="gaussian",
    )

    np.testing.assert_array_equal(result.retained, [True, True, True, False])
    bounds = result.gaussian_bounds["counts"]
    # Median 2 and population deviation sqrt(2/3) of the three recorded values.
    assert bounds["low"] == pytest.approx(0.10054491479716932)
    assert bounds["high"] == pytest.approx(3.8994550852028307)
    assert (bounds["low"], bounds["high"]) == pytest.approx(
        reference_gaussian_bounds(np.array([1, 2, 3]))
    )


@pytest.mark.parametrize("method", ["gaussian", "mad"])
def test_automatic_qc_rejects_selections_with_no_complete_metrics(method):
    with pytest.raises(ValueError, match="no selected cells with complete metrics"):
        filter_cell_metrics(
            {"counts": np.array([1.0, 2.0])},
            {"counts": np.array([True, True])},
            np.ones(2, dtype=bool),
            method=method,
        )


def test_gaussian_qc_rejects_bounds_that_overflow_on_finite_metrics():
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(
            ValueError, match="QC metric 'counts' produced non-finite Gaussian bounds"
        ):
            filter_cell_metrics(
                {"counts": np.array([1e308, 1e308, -1e308, -1e308])},
                {},
                np.ones(4, dtype=bool),
                method="gaussian",
            )


def test_qc_rejects_unknown_filter_method():
    with pytest.raises(ValueError, match="method must be"):
        filter_cell_metrics(
            {"counts": np.array([1, 2])}, {}, np.ones(2, dtype=bool), method="unknown"
        )


def test_manual_qc_never_passes_a_masked_row_inside_its_bounds():
    values = np.array([5.0, 6.0, 7.0, 6.0, 100.0])
    missing = np.array([False, True, False, False, False])
    active = np.array([True, True, True, False, True])

    result = filter_cell_metrics(
        {"score": values},
        {"score": missing},
        active,
        method="manual",
        lows=[5.0],
        highs=[7.0],
        keep_bounds=True,
    )

    # Row 1 is masked and row 3 inactive although both lie inside [5, 7].
    np.testing.assert_array_equal(result.retained, [True, False, True, False, False])
    assert result.gaussian_bounds is None
    assert result.mad_provenance is None
    with pytest.raises(ValueError, match="removed every selected cell"):
        filter_cell_metrics(
            {"score": values},
            {"score": np.ones(5, dtype=bool)},
            active,
            method="manual",
            lows=[None],
            highs=[None],
        )


@pytest.mark.parametrize("method", ["manual", "mad", "gaussian"])
def test_public_and_pipeline_filters_agree_on_nullable_columns(
    small_store,
    method,
) -> None:
    store = small_store
    active = np.asarray(store.cells.fetch_all("I"), dtype=bool)
    counts = np.asarray(store.cells.fetch_all("RNA_nCounts"), dtype=float)
    missing = np.zeros(store.cells.N, dtype=bool)
    missing[np.flatnonzero(active)[2::9]] = True
    values = counts.copy()
    # A placeholder inside the manual bounds would pass if the mask were
    # ignored; a NaN placeholder would break automatic bounds.
    values[missing] = np.median(counts[active]) if method == "manual" else np.nan
    insert_nullable_cell_column(store, "nullable_qc", values, missing)
    low, high = (float(bound) for bound in np.quantile(counts[active], [0.1, 0.9]))
    complete = active & ~missing
    expected = np.zeros(store.cells.N, dtype=bool)

    if method == "manual":
        public = store.filter_cells(
            ["nullable_qc"],
            [low],
            [high],
            keep_bounds=True,
        )
        filtering: dict[str, Any] = {
            "method": "manual",
            "attrs": ["nullable_qc"],
            "lows": [low],
            "highs": [high],
            "keep_bounds": True,
        }
        expected_fingerprint = fingerprint_array(missing)
        expected[complete] = (values[complete] >= low) & (values[complete] <= high)
    else:
        options = {} if method == "mad" else {"method": "gaussian"}
        public = store.auto_filter_cells(attrs=["nullable_qc"], **options)
        filtering = {"attrs": ["nullable_qc"], **options}
        expected_fingerprint = fingerprint_array(missing[active])
        if method == "mad":
            expected[complete] = reference_mad_keep({"nullable_qc": values[complete]})
        else:
            bound_low, bound_high = reference_gaussian_bounds(values[complete])
            expected[complete] = (values[complete] > bound_low) & (
                values[complete] < bound_high
            )
    run = store.pipeline.run(**_run_options(filtering))

    public_mask = _selection_mask(store, public)
    np.testing.assert_array_equal(public_mask, expected)
    np.testing.assert_array_equal(
        public_mask,
        _selection_mask(store, run["analysis_cell_selection"]),
    )
    # Bounds removed measured cells too, and the very deep cell is one of them.
    assert 0 < public_mask.sum() < complete.sum()
    assert not public_mask[1]
    status = store.inspect_artifact(public)
    assert status.inputs["missing_mask_fingerprints"] == {
        "nullable_qc": expected_fingerprint
    }
    complete_ref = store.auto_filter_cells(attrs=["RNA_nCounts"], method="gaussian")
    assert (
        "missing_mask_fingerprints" not in store.inspect_artifact(complete_ref).inputs
    )
    np.testing.assert_array_equal(store.cells.fetch_all("I"), active)


def test_auto_filter_cells_rejects_masked_integer_sample_labels(small_store) -> None:
    store = small_store
    active = np.asarray(store.cells.fetch_all("I"), dtype=bool)
    labels = (np.arange(store.cells.N) % 2 + 1).astype(np.int64)
    missing = np.zeros(store.cells.N, dtype=bool)
    missing[np.flatnonzero(active)[:3]] = True
    labels[missing] = 0
    insert_nullable_cell_column(store, "nullable_donor", labels, missing)

    with pytest.raises(
        ValueError,
        match="^sample_column 'nullable_donor' contains missing labels among active "
        "cells$",
    ):
        store.auto_filter_cells(
            attrs=["RNA_nCounts"],
            sample_column="nullable_donor",
            min_cells_per_sample=2,
        )
    with pytest.raises(PipelineExecutionError) as caught:
        store.pipeline.run(
            **_run_options(
                {
                    "attrs": ["RNA_nCounts"],
                    "sample_column": "nullable_donor",
                    "min_cells_per_sample": 2,
                }
            )
        )
    assert caught.value.stage == "filtering"
    assert str(caught.value.__cause__) == (
        "sample column 'nullable_donor' contains missing labels among active cells"
    )

    labelled = active & ~missing
    store.cells.insert("labelled_cells", labelled)
    ref = store.auto_filter_cells(
        attrs=["RNA_nCounts"],
        cell_selection=store.snapshot_cell_selection("labelled_cells"),
        sample_column="nullable_donor",
        min_cells_per_sample=2,
    )
    counts = np.asarray(store.cells.fetch_all("RNA_nCounts"), dtype=float)
    expected = np.zeros(store.cells.N, dtype=bool)
    expected[labelled] = reference_mad_keep(
        {"RNA_nCounts": counts[labelled]}, labels[labelled], min_cells=2
    )
    np.testing.assert_array_equal(_selection_mask(store, ref), expected)
    assert store.inspect_artifact(ref).parameters["sample_sizes"] == {
        "1": int(np.sum(labelled & (labels == 1))),
        "2": int(np.sum(labelled & (labels == 2))),
    }


def test_merged_partial_integer_column_cannot_seed_a_fake_qc_sample(
    tmp_path,
) -> None:
    from scarf.merge import DataStoreMerge

    left = create_labelled_qc_store(tmp_path / "left")
    right = create_labelled_qc_store(tmp_path / "right")
    left.cells.insert("donor", np.array([1, 1, 2, 2, 1, 2], dtype=np.int64))
    destination = str(tmp_path / "merged")
    DataStoreMerge([left, right], destination, ["left", "right"], nthreads=1).dump()
    merged = DataStore(destination, nthreads=1, min_features_per_cell=0)
    active = np.asarray(merged.cells.fetch_all("I"), dtype=bool)
    missing = merged.cells._get_missing_mask_array("orig_donor")
    assert missing is not None
    unlabelled = np.asarray(missing[:], dtype=bool)
    ids = np.asarray(merged.cells.fetch_all("ids")).astype(str)
    # The right store has no donor column, so exactly its cells are unlabelled.
    np.testing.assert_array_equal(unlabelled, np.char.startswith(ids, "right__"))
    donors = dict(zip([f"left__c{i}" for i in range(6)], [1, 1, 2, 2, 1, 2]))
    np.testing.assert_array_equal(
        merged.cells.fetch_all("orig_donor")[~unlabelled],
        [donors[cell] for cell in ids[~unlabelled]],
    )
    assert np.any(active & unlabelled)
    np.testing.assert_array_equal(merged.cells.fetch_all("orig_donor")[unlabelled], 0)

    with pytest.raises(ValueError, match="contains missing labels among active"):
        merged.auto_filter_cells(
            attrs=["RNA_nCounts"],
            sample_column="orig_donor",
            min_cells_per_sample=2,
        )


def test_filter_cells_rejects_invalid_bounds_and_empty_results(small_store) -> None:
    store = small_store
    before = np.asarray(store.cells.fetch_all("I"), dtype=bool).copy()
    counts = np.asarray(store.cells.fetch_all("RNA_nCounts"), dtype=float)
    cases: list[tuple[dict[str, Any], type[Exception], str]] = [
        ({"lows": [True], "highs": [None]}, TypeError, "finite numbers"),
        ({"lows": [np.bool_(False)], "highs": [None]}, TypeError, "finite numbers"),
        ({"lows": [np.nan], "highs": [None]}, ValueError, "must be finite"),
        ({"lows": [None], "highs": [np.inf]}, ValueError, "must be finite"),
        ({"lows": [object()], "highs": [None]}, TypeError, "finite numbers"),
        ({"lows": [10], "highs": [5]}, ValueError, "cannot exceed"),
        ({"lows": [0], "highs": ["z"]}, TypeError, "numbers or both text"),
        ({"lows": [None], "highs": [None], "keep_bounds": 1}, TypeError, "boolean"),
    ]
    for options, error, message in cases:
        with pytest.raises(error, match=message):
            store.filter_cells(attrs=["RNA_nCounts"], **options)
    with pytest.raises(ValueError, match="duplicate"):
        store.filter_cells(
            ["RNA_nCounts", "RNA_nCounts"],
            [None, None],
            [None, None],
        )
    with pytest.raises(KeyError, match="Cell metadata columns not found: 'absent'"):
        store.filter_cells(["absent"], [None], [None])
    with pytest.raises(ValueError, match="removed every selected cell"):
        store.filter_cells(["RNA_nCounts"], [float(counts.max())], [None])
    store.cells.insert("no_cells", np.zeros(store.cells.N, dtype=bool))
    with pytest.raises(ValueError, match="contains no active cells"):
        store.filter_cells(
            ["RNA_nCounts"],
            [None],
            [None],
            cell_selection=store.snapshot_cell_selection("no_cells"),
        )

    assert (
        store.list_artifacts(
            scope="datastore",
            kind="cell_selection",
            operation="filter_cells",
        )
        == []
    )
    np.testing.assert_array_equal(store.cells.fetch_all("I"), before)
    threshold = int(np.median(counts))
    ref = store.filter_cells(["RNA_nCounts"], [np.int64(threshold)], [None])
    assert store.inspect_artifact(ref).parameters["lows"] == [threshold]
    # The lower bound is exclusive unless keep_bounds is set.
    np.testing.assert_array_equal(
        _selection_mask(store, ref), before & (counts > threshold)
    )
    kept_on_bound = store.filter_cells(
        ["RNA_nCounts"], [threshold], [None], keep_bounds=True
    )
    np.testing.assert_array_equal(
        _selection_mask(store, kept_on_bound), before & (counts >= threshold)
    )


def test_zero_count_percentages_fail_with_actionable_error_in_both_paths() -> None:
    storage, _ = fresh_qc_store()
    dataset = open_qc_store(storage, min_features_per_cell=-1)
    active = np.asarray(dataset.cells.fetch_all("I"), dtype=bool)
    counts = np.asarray(dataset.cells.fetch_all("RNA_nCounts"))
    assert active.all() and np.any(counts == 0)
    message = r"1 non-finite value\(s\).*exclude zero-count cells"

    for options in ({"min_cells_per_sample": 2}, {"method": "gaussian"}):
        with pytest.raises(ValueError, match=message):
            dataset.auto_filter_cells(attrs=["RNA_percentMito"], **options)
    for filtering in (
        {"method": "gaussian", "attrs": ["RNA_percentMito"]},
        {"attrs": ["RNA_percentMito"], "min_cells_per_sample": 2},
    ):
        with pytest.raises(PipelineExecutionError) as caught:
            dataset.pipeline.run(**_run_options(filtering))
        assert caught.value.stage == "filtering"
        assert isinstance(caught.value.__cause__, ValueError)
        assert "exclude zero-count cells" in str(caught.value.__cause__)

    dataset.cells.insert("has_counts", counts > 0)
    ref = dataset.auto_filter_cells(
        attrs=["RNA_percentMito"],
        cell_selection=dataset.snapshot_cell_selection("has_counts"),
        method="gaussian",
    )
    percent = np.asarray(dataset.cells.fetch_all("RNA_percentMito"), dtype=float)
    low, high = reference_gaussian_bounds(percent[counts > 0])
    np.testing.assert_array_equal(
        _selection_mask(dataset, ref), (counts > 0) & (percent > low) & (percent < high)
    )


def test_typed_sample_labels_validate_without_per_cell_scan(
    small_store,
    monkeypatch,
) -> None:
    import scarf.quality_control.filtering as filtering

    store = small_store
    n = store.cells.N
    labels = np.array(["A"] * (n // 2) + ["B"] * (n - n // 2))
    store.cells.insert("typed_sample", labels, overwrite=True)
    active = np.asarray(store.cells.fetch_all("I"), dtype=bool)
    checked: list[object] = []
    label_kind = filtering._sample_label_kind

    def counted_label_kind(value: object, label_name: str) -> str:
        checked.append(value)
        return label_kind(value, label_name)

    monkeypatch.setattr(filtering, "_sample_label_kind", counted_label_kind)
    ref = store.auto_filter_cells(
        attrs=["RNA_nCounts"],
        sample_column="typed_sample",
        min_cells_per_sample=2,
    )
    # Two distinct labels, validated once for errors and once for bounds.
    assert sorted(checked) == ["A", "A", "B", "B"]
    assert len(checked) < int(active.sum())
    counts = np.asarray(store.cells.fetch_all("RNA_nCounts"), dtype=float)
    np.testing.assert_array_equal(
        _selection_mask(store, ref),
        reference_mad_keep({"RNA_nCounts": counts}, labels, min_cells=2) & active,
    )
    status = store.inspect_artifact(ref)
    assert status.inputs["sample_assignments_fingerprint"] == fingerprint_strings(
        labels[active]
    )
    assert status.parameters["sample_sizes"] == {"A": n // 2, "B": n - n // 2}

    checked.clear()
    run = store.pipeline.run(
        **_run_options(
            {
                "attrs": ["RNA_nCounts"],
                "sample_column": "typed_sample",
                "min_cells_per_sample": 2,
            }
        )
    )
    assert sorted(checked) == ["A", "A", "B", "B"]
    np.testing.assert_array_equal(
        _selection_mask(store, run["analysis_cell_selection"]),
        _selection_mask(store, ref),
    )


def test_select_cells_excludes_missing_artifact_labels(nullable) -> None:
    store, missing, codes = nullable.store, nullable.missing, nullable.codes
    stored = store.load_artifact(nullable.clusters)
    np.testing.assert_array_equal(stored["__scarf_missing__values"][:], missing)
    np.testing.assert_array_equal(stored["values"][:][missing], 0)

    zero = store.select_cells(nullable.clusters, include=[0])
    np.testing.assert_array_equal(
        _selection_mask(store, zero),
        (codes == 0) & ~missing,
    )
    labelled = store.select_cells(nullable.clusters, low=-1.0)
    np.testing.assert_array_equal(_selection_mask(store, labelled), ~missing)


def test_doublet_detection_rejects_missing_cluster_labels(nullable) -> None:
    store = nullable.store

    with pytest.raises(ValueError, match="clusters contains missing cluster labels"):
        store.run_doublet_detection(nullable.clusters, nullable.graph)
    assert store.list_artifacts(kind="mapping_reference") == []
    assert store.list_artifacts(kind="doublet_score") == []


def test_label_producers_reject_missing_cluster_labels(nullable) -> None:
    store, clusters, graph = nullable.store, nullable.clusters, nullable.graph
    (features,) = store.list_artifacts(kind="feature_selection")
    complete = nullable.complete

    with pytest.raises(
        ValueError, match="clusters contains missing labels.*select_cells"
    ):
        store.run_marker_search(clusters, features=features)
    with pytest.raises(
        ValueError,
        match=r"clusters contains missing labels\. Freeze the labels over the "
        r"graph's cell selection",
    ):
        store.calc_membership_strength(clusters, graph)
    both = (
        r"contains missing labels\. Select the cells labelled in both artifacts "
        r"with select_cells\(\.\.\., include=\[\.\.\.\]\) and freeze both over "
        r"that selection with snapshot_cluster_labels\(\.\.\., cell_selection=\.\.\.\)$"
    )
    with pytest.raises(ValueError, match=f"^to_relabel {both}"):
        store.smart_label(clusters, complete)
    with pytest.raises(ValueError, match=f"^base_label {both}"):
        store.smart_label(complete, clusters)
    for kind in ("marker_table", "membership_strength", "smart_label"):
        assert store.list_artifacts(kind=kind) == []


def test_make_bulk_excludes_missing_artifact_labels(nullable) -> None:
    store, clusters = nullable.store, nullable.clusters
    labelled = store.select_cells(clusters, low=-1.0)
    options: dict[str, Any] = {
        "aggr_type": "sum",
        "feature_label": "id",
        "remove_empty_features": False,
    }

    masked = store.make_bulk(clusters, **options)
    restricted = store.make_bulk(clusters, cell_selection=labelled, **options)

    assert list(masked.columns) == ["0", "1", "2"]
    pd.testing.assert_frame_equal(masked, restricted)
    # Each bulk column sums the raw counts of its labelled cells only.
    raw = np.asarray(store.RNA.rawData[np.arange(store.cells.N)].compute())
    for code in range(3):
        rows = (nullable.codes == code) & ~nullable.missing
        np.testing.assert_array_equal(
            masked[str(code)].to_numpy(), raw[rows].sum(axis=0)
        )


def test_live_masked_columns_are_missing_values_or_rejected(nullable) -> None:
    from scarf.metrics import silhouette_scoring

    store = nullable.store
    n_cells = store.cells.N
    missing = np.zeros(n_cells, dtype=bool)
    missing[::7] = True
    donor = np.where(missing, 0, np.arange(n_cells) % 2 + 1).astype(np.int64)
    insert_nullable_cell_column(store, "donor", donor, missing)
    active = store.cells.active_index("I")

    values = store.get_cell_vals("RNA", "I", "donor")
    assert values.dtype == np.float64
    np.testing.assert_array_equal(np.isnan(values), missing[active])
    np.testing.assert_array_equal(
        values[~missing[active]], donor[active][~missing[active]]
    )
    counts = store.get_cell_vals("RNA", "I", "RNA_nCounts")
    assert counts.dtype == store.cells.get_dtype("RNA_nCounts")

    diffusion = store.run_diffusion_operator(nullable.graph)
    with pytest.raises(ValueError, match="'donor' contains missing values"):
        store.get_imputed("donor", diffusion)
    assert np.isfinite(store.get_imputed("RNA_nCounts", diffusion)).all()
    with pytest.raises(ValueError, match="'donor' contains missing values"):
        silhouette_scoring(
            store,
            None,
            np.zeros((len(active), 2)),
            "RNA",
            "donor",
            distance_metric="l2",
        )


def test_doublet_detection_requires_a_connectivity_graph_and_writable_store(
    nullable,
) -> None:
    store, graph, clusters = nullable.store, nullable.graph, nullable.complete
    neighbors = ArtifactRef.from_dict(store.inspect_artifact(graph).inputs["neighbors"])

    with pytest.raises(ValueError, match="connectivity_map or integrated_graph"):
        store.run_doublet_detection(clusters, neighbors)
    assert store.list_artifacts(kind="mapping_reference") == []

    read_only = DataStore(store.zarr_loc, zarr_mode="r", nthreads=1)
    with pytest.raises(PermissionError, match="run_doublet_detection requires"):
        read_only.run_doublet_detection(clusters, graph)


def test_doublet_detection_validates_arguments_before_any_work(nullable) -> None:
    store, graph, clusters = nullable.store, nullable.graph, nullable.complete
    invalid = (
        ({"smoothing_t": 0}, ValueError, "smoothing_t"),
        ({"save_k": True}, TypeError, "save_k"),
        ({"max_cells_per_cluster": 0}, ValueError, "max_cells_per_cluster"),
        ({"cluster_sample_fraction": 0.0}, ValueError, "cluster_sample_fraction"),
        ({"cluster_sample_fraction": 1.5}, ValueError, "cluster_sample_fraction"),
        ({"simulation_ratio": -1.0}, ValueError, "simulation_ratio"),
        ({"heterotypic_fraction": 1.5}, ValueError, "heterotypic_fraction"),
        ({"random_seed": -1}, ValueError, "random_seed"),
        ({"normalize_scores": 1}, TypeError, "normalize_scores"),
    )
    for options, error, name in invalid:
        with pytest.raises(error, match=name):
            store.run_doublet_detection(clusters, graph, **options)

    assert store.list_artifacts(kind="mapping_reference") == []
    assert store.list_artifacts(kind="doublet_score") == []


def test_read_only_selection_and_derived_assay_producers_raise_permission_errors(
    tmp_path,
) -> None:
    storage, _ = fresh_qc_store()
    writable = open_qc_store(storage)
    cells = writable.snapshot_cell_selection("I")
    kept = writable.filter_cells(["RNA_nCounts"], [0], [None], cell_selection=cells)
    read_only = open_qc_store(storage, zarr_mode="r")

    assert (
        read_only.filter_cells(["RNA_nCounts"], [0], [None], cell_selection=cells)
        == kept
    )
    with pytest.raises(PermissionError, match="filter_cells requires"):
        read_only.filter_cells(["RNA_nCounts"], [1], [None], cell_selection=cells)
    with pytest.raises(PermissionError, match="auto_filter_cells requires"):
        read_only.auto_filter_cells(
            ["RNA_nCounts"], method="gaussian", cell_selection=cells
        )
    with pytest.raises(PermissionError, match="add_grouped_assay requires"):
        read_only.add_grouped_assay("names", assay_label="GROUPED")
    with pytest.raises(PermissionError, match="add_melded_assay requires"):
        read_only.add_melded_assay(
            external_bed_fn=str(tmp_path / "missing.bed"), assay_label="MELDED"
        )


def test_read_only_qc_producers_reuse_results_and_refuse_new_work(
    datastore_ephemeral,
) -> None:
    storage, _ = fresh_qc_store()
    writable = open_qc_store(storage)
    cells = writable.snapshot_cell_selection("I")
    names = writable.RNA.feats.fetch_all("names").astype(str)
    mito = writable.set_feature_selection(
        from_assay="RNA",
        mask=np.char.startswith(names, "MT-"),
    )
    ribo = writable.set_feature_selection(
        from_assay="RNA",
        mask=np.char.startswith(names, "RP"),
    )
    stored = writable.run_feature_percentage(cells, mito)
    read_only = open_qc_store(storage, zarr_mode="r")
    assert read_only.run_feature_percentage(cells, mito) == stored
    with pytest.raises(PermissionError, match="run_feature_percentage requires"):
        read_only.run_feature_percentage(cells, ribo)

    hto_store = datastore_ephemeral
    assay_types = dict(hto_store.zw.attrs["assayTypes"])
    assay_types["assay2"] = "HTO"
    hto_store.zw.attrs["assayTypes"] = assay_types
    selection = hto_store.snapshot_cell_selection("I")
    hto_read_only = DataStore(hto_store.zarr_loc, zarr_mode="r", nthreads=1)
    with pytest.raises(PermissionError, match="run_hto_demultiplexing requires"):
        hto_read_only.run_hto_demultiplexing(selection, from_assay="assay2")

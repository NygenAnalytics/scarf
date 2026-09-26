"""Missing-value masks, bound validation, and guards in quality control."""

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
from tests.test_datastore import _open_qc_store, _qc_store
from tests.test_pipeline import _insert_nullable_cell_column, _minimal_run_options


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
    return {**_minimal_run_options(), "filtering": filtering}


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


def test_integer_qc_metrics_exclude_missing_values_from_gaussian_bounds():
    result = filter_cell_metrics(
        {"counts": np.array([1, 2, 3, 1000])},
        {"counts": np.array([False, False, False, True])},
        np.ones(4, dtype=bool),
        method="gaussian",
    )

    np.testing.assert_array_equal(result.retained, [True, True, True, False])
    bounds = result.gaussian_bounds["counts"]
    assert bounds["low"] == pytest.approx(0.10054491479716932)
    assert bounds["high"] == pytest.approx(3.8994550852028307)


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
        with pytest.raises(ValueError, match="non-finite Gaussian bounds"):
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


@pytest.mark.parametrize("method", ["manual", "mad", "gaussian"])
def test_public_and_pipeline_filters_agree_on_nullable_columns(
    datastore_ephemeral,
    method,
) -> None:
    store = datastore_ephemeral
    active = np.asarray(store.cells.fetch_all("I"), dtype=bool)
    counts = np.asarray(store.cells.fetch_all("RNA_nCounts"), dtype=float)
    missing = np.zeros(store.cells.N, dtype=bool)
    missing[np.flatnonzero(active)[::9]] = True
    values = counts.copy()
    # A placeholder inside the manual bounds would pass if the mask were
    # ignored; a NaN placeholder would break automatic bounds.
    values[missing] = np.median(counts[active]) if method == "manual" else np.nan
    _insert_nullable_cell_column(store, "nullable_qc", values, missing)
    low, high = (float(bound) for bound in np.quantile(counts[active], [0.1, 0.9]))

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
    else:
        options = {} if method == "mad" else {"method": "gaussian"}
        public = store.auto_filter_cells(attrs=["nullable_qc"], **options)
        filtering = {"attrs": ["nullable_qc"], **options}
        expected_fingerprint = fingerprint_array(missing[active])
    run = store.pipeline.run(**_run_options(filtering))

    public_mask = _selection_mask(store, public)
    np.testing.assert_array_equal(
        public_mask,
        _selection_mask(store, run["analysis_cell_selection"]),
    )
    assert public_mask.any()
    assert not public_mask[missing].any()
    assert not public_mask[~active].any()
    status = store.inspect_artifact(public)
    assert status.inputs["missing_mask_fingerprints"] == {
        "nullable_qc": expected_fingerprint
    }
    complete = store.auto_filter_cells(attrs=["RNA_nCounts"], method="gaussian")
    assert "missing_mask_fingerprints" not in store.inspect_artifact(complete).inputs
    np.testing.assert_array_equal(store.cells.fetch_all("I"), active)


def test_auto_filter_cells_rejects_masked_integer_sample_labels(
    datastore_ephemeral,
) -> None:
    store = datastore_ephemeral
    active = np.asarray(store.cells.fetch_all("I"), dtype=bool)
    labels = (np.arange(store.cells.N) % 2 + 1).astype(np.int64)
    missing = np.zeros(store.cells.N, dtype=bool)
    missing[np.flatnonzero(active)[:3]] = True
    labels[missing] = 0
    _insert_nullable_cell_column(store, "nullable_donor", labels, missing)

    with pytest.raises(ValueError, match="contains missing labels among active"):
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
    assert "contains missing labels among active" in str(caught.value.__cause__)

    store.cells.insert("labelled_cells", active & ~missing)
    ref = store.auto_filter_cells(
        attrs=["RNA_nCounts"],
        cell_selection=store.snapshot_cell_selection("labelled_cells"),
        sample_column="nullable_donor",
        min_cells_per_sample=2,
    )
    assert set(store.inspect_artifact(ref).parameters["sample_sizes"]) == {"1", "2"}


def test_merged_partial_integer_column_cannot_seed_a_fake_qc_sample(
    tmp_path,
) -> None:
    from scarf.merge import DataStoreMerge
    from tests.test_preparation import _create_store

    left = _create_store(tmp_path / "left")
    right = _create_store(tmp_path / "right")
    left.cells.insert("donor", np.array([1, 1, 2, 2, 1, 2], dtype=np.int64))
    destination = str(tmp_path / "merged")
    DataStoreMerge([left, right], destination, ["left", "right"], nthreads=1).dump()
    merged = DataStore(destination, nthreads=1, min_features_per_cell=0)
    active = np.asarray(merged.cells.fetch_all("I"), dtype=bool)
    missing = merged.cells._get_missing_mask_array("orig_donor")
    assert missing is not None
    unlabelled = np.asarray(missing[:], dtype=bool)
    assert np.any(active & unlabelled)
    np.testing.assert_array_equal(merged.cells.fetch_all("orig_donor")[unlabelled], 0)

    with pytest.raises(ValueError, match="contains missing labels among active"):
        merged.auto_filter_cells(
            attrs=["RNA_nCounts"],
            sample_column="orig_donor",
            min_cells_per_sample=2,
        )


def test_filter_cells_rejects_invalid_bounds_and_empty_results(
    datastore_ephemeral,
) -> None:
    store = datastore_ephemeral
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
    ref = store.filter_cells(["RNA_nCounts"], [np.int64(500)], [None])
    assert store.inspect_artifact(ref).parameters["lows"] == [500]


def test_zero_count_percentages_fail_with_actionable_error_in_both_paths() -> None:
    storage, _ = _qc_store()
    dataset = _open_qc_store(storage, min_features_per_cell=-1)
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
    assert not _selection_mask(dataset, ref)[counts == 0].any()


def test_typed_sample_labels_validate_without_per_cell_scan(
    datastore_ephemeral,
    monkeypatch,
) -> None:
    import scarf.quality_control.filtering as filtering

    store = datastore_ephemeral
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
    assert 0 < len(checked) <= 4 < int(active.sum())
    status = store.inspect_artifact(ref)
    assert status.inputs["sample_assignments_fingerprint"] == fingerprint_strings(
        labels[active]
    )
    assert set(status.parameters["sample_sizes"]) == {"A", "B"}

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
    assert 0 < len(checked) <= 4
    np.testing.assert_array_equal(
        _selection_mask(store, run["analysis_cell_selection"]),
        _selection_mask(store, ref),
    )


def test_select_cells_excludes_missing_artifact_labels(tmp_path) -> None:
    store, result, codes, missing = import_nullable_cluster_h5ad(tmp_path)
    clusters = result.clusterArtifacts["clusters"]
    stored = store.load_artifact(clusters)
    np.testing.assert_array_equal(stored["__scarf_missing__values"][:], missing)
    np.testing.assert_array_equal(stored["values"][:][missing], 0)

    zero = store.select_cells(clusters, include=[0])
    np.testing.assert_array_equal(
        _selection_mask(store, zero),
        (codes == 0) & ~missing,
    )
    labelled = store.select_cells(clusters, low=-1.0)
    np.testing.assert_array_equal(_selection_mask(store, labelled), ~missing)


def test_doublet_detection_rejects_missing_cluster_labels(tmp_path) -> None:
    store, result, _, _ = import_nullable_cluster_h5ad(tmp_path)
    graph = _imported_graph(store, result.cellSelection)

    with pytest.raises(ValueError, match="clusters contains missing cluster labels"):
        store.run_doublet_detection(result.clusterArtifacts["clusters"], graph)
    assert store.list_artifacts(kind="mapping_reference") == []
    assert store.list_artifacts(kind="doublet_score") == []


def test_doublet_detection_requires_a_connectivity_graph_and_writable_store(
    analyzed_datastore_ephemeral,
) -> None:
    store = analyzed_datastore_ephemeral
    (graph,) = store.list_artifacts(kind="connectivity_map")
    clusters = store.run_leiden_clustering(graph)
    neighbors = ArtifactRef.from_dict(store.inspect_artifact(graph).inputs["neighbors"])

    with pytest.raises(ValueError, match="connectivity_map or integrated_graph"):
        store.run_doublet_detection(clusters, neighbors)
    assert store.list_artifacts(kind="mapping_reference") == []

    read_only = DataStore(store.zarr_loc, zarr_mode="r", nthreads=1)
    with pytest.raises(PermissionError, match="run_doublet_detection requires"):
        read_only.run_doublet_detection(clusters, graph)


def test_read_only_qc_producers_reuse_results_and_refuse_new_work(
    datastore_ephemeral,
) -> None:
    storage, _ = _qc_store()
    writable = _open_qc_store(storage)
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
    read_only = _open_qc_store(storage, zarr_mode="r")
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

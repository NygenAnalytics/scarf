"""Argument, dtype, and record contracts of the quality-control operations."""

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scarf import DataStore
from scarf.metadata.artifacts import plan_cell_data_artifact, write_cell_data_artifact
from scarf.metadata.selection import NamedCellArtifact
from scarf.storage.artifact_writer import artifact_transaction, plan_artifact
from scarf.storage.artifacts import (
    ArtifactRef,
    artifact_group,
    fingerprint_array,
    fingerprint_strings,
    new_artifact_id,
)
from scarf.storage.errors import ArtifactResolutionError
from scarf.storage.feature_selection import (
    _feature_selection_plan,
    _ordered_feature_ids_fingerprint,
    _write_feature_selection,
)
from scarf.storage.selections import read_stored_selection_mask
from scarf.utils.logging import logger
from tests.test_graph_operation_paths import N_CELLS, open_store, write_graph_template
from tests.qc_helpers import reference_mad_keep

# A cell-selection input that names no artifact.
MALFORMED_SELECTION = {
    "type": "artifact",
    "scope": "datastore",
    "kind": "cell_selection",
}


def cell_values(
    store: DataStore,
    cells: ArtifactRef,
    kind: str,
    values: np.ndarray,
    *,
    missing: np.ndarray | None = None,
) -> ArtifactRef:
    """Store one value per selected cell as a custom per-cell artifact."""
    values = np.asarray(values)
    fingerprint = (
        fingerprint_strings(values)
        if values.dtype.kind in {"O", "S", "U"}
        else fingerprint_array(values)
    )
    planned = plan_cell_data_artifact(
        store.zw,
        scope="assay",
        assay="RNA",
        kind=kind,
        operation="custom_cell_values",
        parameters={"masked": missing is not None},
        inputs={"values_fingerprint": fingerprint},
        execution_options={},
        cell_selection=cells,
        arrays={"values": (values.shape, None)},
    )
    group = write_cell_data_artifact(store.zw, planned, {"values": values})
    if missing is not None:
        group.create_array("__scarf_missing__values", data=np.asarray(missing))
        group["values"].attrs["missing_mask"] = "__scarf_missing__values"
    return planned.ref


def foreign_record(
    store: DataStore,
    kind: str,
    name: str,
    *,
    inputs: dict[str, Any],
    arrays: dict[str, np.ndarray],
    attrs: dict[str, Any] | None = None,
) -> ArtifactRef:
    """Store a complete record that no scarf producer writes, as an import might."""
    planned = plan_artifact(
        store.zw,
        scope="assay",
        assay="RNA",
        kind=kind,
        operation=f"import_{kind}",
        parameters={"name": name},
        inputs=inputs,
        execution_options={},
    )
    with artifact_transaction(store.zw, planned) as group:
        for array_name, values in arrays.items():
            group.create_array(array_name, data=values)
        for key, value in (attrs or {}).items():
            group.attrs[key] = value
    return planned.ref


def write_qc_template(zarr_loc: Path) -> dict[str, ArtifactRef]:
    """Add selections, cluster labels, per-cell values, and foreign records.

    The per-cell values hold floating, integer, text, complex, and matrix
    values, and each foreign record breaks one cell-selection or graph
    contract.
    """
    refs = write_graph_template(zarr_loc)
    store = open_store(zarr_loc)
    cells = refs["cells"]
    order = np.arange(N_CELLS)
    store.cells.insert("cluster", order % 3)
    for column, values in (
        ("first_cells", order < 30),
        ("three_cells", order < 3),
        ("no_cells", np.zeros(N_CELLS, dtype=bool)),
    ):
        store.cells.insert(column, values)
        refs[column] = store.snapshot_cell_selection(column)
    refs["clusters"] = store.snapshot_cluster_labels("cluster", cell_selection=cells)
    refs["first_clusters"] = store.snapshot_cluster_labels(
        "cluster",
        cell_selection=refs["first_cells"],
    )
    refs["metric"] = cell_values(
        store, cells, "quality_metric", np.linspace(10.0, 50.0, N_CELLS)
    )
    refs["matrix"] = cell_values(store, cells, "quality_metric", np.ones((N_CELLS, 2)))
    refs["complex_values"] = cell_values(
        store, cells, "quality_metric", (order + 1j).astype(np.complex64)
    )
    refs["codes"] = cell_values(
        store, cells, "hto_identity", (order % 2).astype(np.int16)
    )
    refs["tags"] = cell_values(
        store, cells, "hto_identity", np.where(order % 2, "tag-a", "tag-b")
    )
    labels = (order % 3).astype(np.int64)
    refs["metric_with_malformed_selection"] = foreign_record(
        store,
        "quality_metric",
        "malformed",
        inputs={"cell_selection": MALFORMED_SELECTION},
        arrays={"values": np.linspace(10.0, 50.0, N_CELLS)},
    )
    refs["clusters_without_selection"] = foreign_record(
        store, "cluster_labels", "unselected", inputs={}, arrays={"values": labels}
    )
    refs["clusters_with_malformed_selection"] = foreign_record(
        store,
        "cluster_labels",
        "malformed",
        inputs={"cell_selection": MALFORMED_SELECTION},
        arrays={"values": labels},
    )
    refs["clusters_missing_a_cell"] = foreign_record(
        store,
        "cluster_labels",
        "short",
        inputs={"cell_selection": cells},
        arrays={"values": labels[:-1]},
    )
    refs["clusters_in_rows"] = foreign_record(
        store,
        "cluster_labels",
        "rows",
        inputs={"cell_selection": cells},
        arrays={"values": labels[:, None]},
    )
    neighbors = np.asarray(
        artifact_group(store.zw, refs["rna_neighbors"])["indices"][:]
    )
    n_neighbors = neighbors.shape[1]
    edges = np.column_stack(
        [np.repeat(order, n_neighbors), neighbors.reshape(-1)]
    ).astype(np.uint32)
    refs["imported_connectivity"] = foreign_record(
        store,
        "connectivity_map",
        "imported",
        inputs={"neighbors": refs["rna_neighbors"]},
        arrays={"edges": edges, "weights": np.ones(len(edges), dtype=np.float32)},
        attrs={"n_cells": N_CELLS, "n_neighbors": n_neighbors},
    )
    return refs


@pytest.fixture(scope="module")
def qc_template(tmp_path_factory) -> tuple[Path, dict[str, ArtifactRef]]:
    zarr_loc = tmp_path_factory.mktemp("qc_template") / "store.zarr"
    return zarr_loc, write_qc_template(zarr_loc)


@pytest.fixture(scope="module")
def read_only_store(qc_template) -> tuple[DataStore, dict[str, ArtifactRef]]:
    """The template opened read only, for calls that must fail before writing."""
    zarr_loc, refs = qc_template
    return open_store(zarr_loc, zarr_mode="r"), refs


@pytest.fixture
def qc_store(qc_template, tmp_path) -> tuple[DataStore, dict[str, ArtifactRef]]:
    zarr_loc, refs = qc_template
    target = tmp_path / "store.zarr"
    shutil.copytree(zarr_loc, target)
    return open_store(target), refs


def _selected(store: DataStore, ref: ArtifactRef) -> np.ndarray:
    return read_stored_selection_mask(
        store.zw,
        ref,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )


def test_filter_cells_rejects_unpaired_bounds_and_named_selections(
    read_only_store,
) -> None:
    store, _refs = read_only_store

    with pytest.raises(ValueError, match="must have the same length"):
        store.filter_cells(["RNA_nCounts", "RNA_nFeatures"], [None, None], [None])
    with pytest.raises(TypeError, match="cell_selection must be an ArtifactRef"):
        store.filter_cells(["RNA_nCounts"], [None], [None], cell_selection="I")


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        (
            {"low": 0.0, "keep_bounds": "yes"},
            TypeError,
            "keep_bounds must be a boolean",
        ),
        ({"include": "tag-a"}, TypeError, "must be a sequence of scalar values"),
        ({"include": {"tag-a"}}, TypeError, "must be a sequence of scalar values"),
        ({"include": [("tag-a",)]}, TypeError, "must contain only scalar values"),
        ({"include": [np.float32(np.inf)]}, ValueError, "only finite values"),
        ({"include": []}, ValueError, "must contain at least one value"),
        ({"low": np.nan}, ValueError, "low must be finite or None"),
    ],
    ids=[
        "keep_bounds",
        "text",
        "unordered",
        "nested",
        "infinite",
        "empty",
        "nan_bound",
    ],
)
def test_select_cells_rejects_invalid_bounds_and_include_values(
    read_only_store,
    options,
    error,
    message,
) -> None:
    store, refs = read_only_store

    with pytest.raises(error, match=message):
        store.select_cells(refs["metric"], **options)


@pytest.mark.parametrize(
    ("values", "error", "message"),
    [
        (lambda refs: "RNA_nCounts", TypeError, "values must be an ArtifactRef"),
        (
            lambda refs: ArtifactRef(
                scope="assay",
                assay="RNA",
                kind="quality_metric",
                artifact_id=new_artifact_id(),
            ),
            ValueError,
            "Cell-aligned artifact is unavailable or incomplete",
        ),
        # select_cells reads its input with the shared cell-aligned reader.
        (
            lambda refs: refs["cells"],
            ValueError,
            "'cell_selection' is not a cell-aligned artifact kind",
        ),
        (
            lambda refs: refs["metric_with_malformed_selection"],
            ArtifactResolutionError,
            "quality_metric artifact has a malformed 'cell_selection' input",
        ),
        (
            lambda refs: refs["rna_normalized"],
            ValueError,
            "'normalized' is not a cell-aligned artifact kind",
        ),
        (lambda refs: refs["matrix"], ValueError, "one value per source-selected cell"),
    ],
    ids=[
        "column_name",
        "absent",
        "selection",
        "malformed_record",
        "normalized",
        "matrix",
    ],
)
def test_select_cells_needs_one_stored_value_per_cell(
    read_only_store,
    values,
    error,
    message,
) -> None:
    store, refs = read_only_store

    with pytest.raises(error, match=message):
        store.select_cells(values(refs), low=0.0)


@pytest.mark.parametrize(
    ("artifact", "include", "error", "message"),
    [
        ("tags", [1], TypeError, "strings for a string artifact"),
        ("codes", [40_000], ValueError, "out-of-range integer"),
        # An integer beyond the float64 range has no floating value.
        ("metric", [10**400], ValueError, "out-of-range integer"),
        ("metric", ["30"], TypeError, "numeric for a floating artifact"),
        ("metric", [True], TypeError, "numeric for a floating artifact"),
        ("complex_values", [1], TypeError, "must contain scalar values"),
    ],
    ids=["text", "int16", "float_overflow", "float_text", "float_bool", "complex"],
)
def test_select_cells_include_values_must_match_the_artifact_dtype(
    read_only_store,
    artifact,
    include,
    error,
    message,
) -> None:
    store, refs = read_only_store

    with pytest.raises(error, match=message):
        store.select_cells(refs[artifact], include=include)


def test_select_cells_includes_boolean_artifact_values(qc_store) -> None:
    store, refs = qc_store
    flags = np.arange(N_CELLS) % 4 == 0
    flagged = cell_values(store, refs["cells"], "quality_metric", flags)

    selected = store.select_cells(flagged, include=[np.True_])

    np.testing.assert_array_equal(_selected(store, selected), flags)
    assert store.inspect_artifact(selected).parameters["include"] == [True]
    with pytest.raises(TypeError, match="booleans for a boolean artifact"):
        store.select_cells(flagged, include=[1])


def test_select_cells_composes_only_with_an_artifact_ref(read_only_store) -> None:
    store, refs = read_only_store

    with pytest.raises(TypeError, match="cell_selection must be an ArtifactRef"):
        store.select_cells(refs["metric"], low=0.0, cell_selection="I")


def test_auto_filter_cells_defaults_to_the_qc_metrics_the_store_has(qc_store) -> None:
    store, _refs = qc_store
    # No synthetic gene matches the mitochondrial or ribosomal pattern.
    assert "RNA_percentMito" not in store.cells.columns
    assert "RNA_percentRibo" not in store.cells.columns

    retained = store.auto_filter_cells()

    parameters = store.inspect_artifact(retained).parameters
    assert parameters["attrs"] == ["RNA_nCounts", "RNA_nFeatures"]
    metrics = {
        attr: np.asarray(store.cells.fetch_all(attr), dtype=float)
        for attr in parameters["attrs"]
    }
    np.testing.assert_array_equal(
        _selected(store, retained), reference_mad_keep(metrics, min_cells=20)
    )


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"n_mads": "3"}, TypeError, "n_mads must be a positive number"),
        ({"n_mads": True}, TypeError, "n_mads must be a positive number"),
        ({"min_cells_per_sample": 1}, ValueError, "integer >= 2"),
        ({"min_cells_per_sample": 2.0}, ValueError, "integer >= 2"),
        ({"min_cells_per_sample": True}, ValueError, "integer >= 2"),
    ],
)
def test_mad_filtering_validates_its_settings(
    read_only_store,
    options,
    error,
    message,
) -> None:
    store, _refs = read_only_store

    with pytest.raises(error, match=message):
        store.auto_filter_cells(**options)


def test_mad_filtering_never_retains_cells_whose_artifact_metric_is_masked(
    qc_store,
) -> None:
    store, refs = qc_store
    missing = np.zeros(N_CELLS, dtype=bool)
    missing[[3, 17, 31]] = True
    values = np.linspace(10.0, 50.0, N_CELLS)
    # The placeholder lies inside the MAD bounds of the measured values.
    values[missing] = 0.0
    unmasked = cell_values(store, refs["cells"], "quality_metric", values)
    masked = cell_values(
        store, refs["cells"], "quality_metric", values, missing=missing
    )

    def retained(metric: ArtifactRef) -> np.ndarray:
        selection = store.auto_filter_cells(
            attrs=[],
            artifact_metrics=[NamedCellArtifact("score", metric)],
        )
        return _selected(store, selection)

    assert retained(unmasked).all()
    np.testing.assert_array_equal(retained(masked), ~missing)


@pytest.mark.parametrize(
    ("call", "error", "message"),
    [
        (
            lambda store, refs: store.run_feature_percentage("I", refs["rna_features"]),
            TypeError,
            "cell_selection must be an ArtifactRef",
        ),
        (
            lambda store, refs: store.run_feature_percentage(refs["cells"], "^MT-"),
            TypeError,
            "features must be an ArtifactRef",
        ),
        (
            lambda store, refs: store.run_feature_percentage(
                refs["rna_features"], refs["cells"]
            ),
            ValueError,
            "features must be an assay-scoped ArtifactRef",
        ),
        (
            lambda store, refs: store.run_feature_percentage(
                refs["no_cells"], refs["rna_features"]
            ),
            ValueError,
            "must select at least one cell",
        ),
    ],
    ids=["named_cells", "pattern", "swapped", "no_cells"],
)
def test_feature_percentage_validates_its_selections(
    read_only_store,
    call,
    error,
    message,
) -> None:
    store, refs = read_only_store

    with pytest.raises(error, match=message):
        call(store, refs)


def test_feature_percentage_refuses_a_stored_selection_without_features(
    qc_store,
) -> None:
    store, refs = qc_store
    assay = store.get_assay("RNA")
    # No scarf producer writes an empty selection, but a foreign writer can
    # store one; the feature selection reader refuses it.
    values = np.zeros(assay.feats.N, dtype=bool)
    feature_ids_fingerprint = _ordered_feature_ids_fingerprint(assay.z)
    planned = _feature_selection_plan(
        store.zw,
        assay="RNA",
        n_features=assay.feats.N,
        ordered_feature_ids_fingerprint=feature_ids_fingerprint,
        operation="set_feature_selection",
        parameters={"values_fingerprint": fingerprint_array(values)},
        inputs={"all_features": refs["rna_features"]},
        execution_options={},
    )
    _write_feature_selection(
        store.zw,
        planned,
        ordered_feature_ids_fingerprint=feature_ids_fingerprint,
        payload={"values": values},
    )
    metrics = store.list_artifacts(kind="quality_metric", from_assay="RNA")

    # Otherwise every cell would get a percentage of zero.
    with pytest.raises(ValueError, match="must select at least one feature"):
        store.run_feature_percentage(refs["cells"], planned.ref)
    assert store.list_artifacts(kind="quality_metric", from_assay="RNA") == metrics


def test_hto_demultiplexing_reads_the_hto_assay_by_default(qc_store) -> None:
    store, refs = qc_store
    # Without a recorded type, the assay named HTO declares that preset, which
    # a read-only open resolves but cannot record.
    del store.zw.attrs["assayTypes"]
    read_only = open_store(Path(store.zarr_loc), zarr_mode="r")

    # The three hashtags of the HTO assay need at least four selected cells.
    with pytest.raises(ValueError, match="at least 4 selected cells"):
        read_only.run_hto_demultiplexing(refs["three_cells"])


def test_hto_demultiplexing_validates_its_arguments(read_only_store) -> None:
    store, refs = read_only_store

    with pytest.raises(TypeError, match="cell_selection must be an ArtifactRef"):
        store.run_hto_demultiplexing("I")
    with pytest.raises(TypeError, match="random_seed must be an integer"):
        store.run_hto_demultiplexing(refs["cells"], random_seed=True)


def test_cell_cycle_scoring_requires_an_rna_assay_and_an_artifact_ref(
    qc_store,
) -> None:
    store, refs = qc_store

    with pytest.raises(TypeError, match="RNAassay; received ADTassay"):
        store.run_cell_cycle_scoring(refs["cells"], from_assay="ADT")
    with pytest.raises(TypeError, match="cell_selection must be an ArtifactRef"):
        store.run_cell_cycle_scoring("I")


def test_cell_cycle_scoring_matches_gene_names_and_validates_gene_lists(
    qc_store,
) -> None:
    store, refs = qc_store
    cells = refs["cells"]
    for options, error, message in (
        ({"s_genes": "RNA0"}, TypeError, "^s_genes must be a sequence of gene names$"),
        (
            {"s_genes": ["RNA0"], "g2m_genes": ["RNA1", 2]},
            TypeError,
            "^g2m_genes must be a sequence of gene names$",
        ),
        (
            {"s_genes": ["RNA0"], "g2m_genes": ["RNA1"], "log_transform": 1},
            TypeError,
            "^log_transform must be a bool$",
        ),
        (
            {"s_genes": ["absent"], "g2m_genes": ["RNA1"]},
            ValueError,
            "^None of the s_genes match the assay feature names$",
        ),
    ):
        with pytest.raises(error, match=message):
            store.run_cell_cycle_scoring(cells, **options)
    assert store.list_artifacts(kind="cell_cycle", from_assay="RNA") == []

    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]), level="WARNING"
    )
    try:
        ref = store.run_cell_cycle_scoring(
            cells,
            s_genes=["rna0", "RNA2", "absent"],
            g2m_genes=["Rna1", "RNA3"],
            n_bins=3,
        )
    finally:
        logger.remove(sink)

    # Names match without case; the control size defaults to the shorter list.
    parameters = store.inspect_artifact(ref).parameters
    assert parameters["s_gene_indices"] == [0, 2]
    assert parameters["g2m_gene_indices"] == [1, 3]
    assert parameters["control_size"] == 2
    assert messages == ["1 of 3 s_genes were not found in the assay feature names"]
    phase = store.load_artifact(ref)["phase"][:]
    assert phase.shape == (N_CELLS,)
    assert set(phase) <= {"G1", "S", "G2M"}


def test_prevalent_peak_selection_requires_an_artifact_ref(qc_store) -> None:
    store, _refs = qc_store

    with pytest.raises(TypeError, match="cell_selection must be an ArtifactRef"):
        store.select_prevalent_peaks("I", from_assay="ATAC")


def test_doublet_detection_takes_artifact_refs(read_only_store) -> None:
    store, refs = read_only_store

    with pytest.raises(TypeError, match="graph must be an ArtifactRef"):
        store.run_doublet_detection(refs["clusters"], "graph")
    with pytest.raises(TypeError, match="clusters must be an ArtifactRef"):
        store.run_doublet_detection("clusters", refs["wnn"], from_assay="RNA")


def test_doublet_detection_on_an_integrated_graph_scores_its_rna_source(
    read_only_store,
) -> None:
    store, refs = read_only_store

    with pytest.raises(ValueError, match="from_assay is required"):
        store.run_doublet_detection(refs["clusters"], refs["wnn"])
    with pytest.raises(TypeError, match="RNA assays; received ADTassay"):
        store.run_doublet_detection(refs["clusters"], refs["wnn"], from_assay="ADT")


def test_doublet_detection_needs_clusters_over_the_graph_cells(
    read_only_store,
) -> None:
    store, refs = read_only_store

    with pytest.raises(ValueError, match="Cluster and graph cell selections"):
        store.run_doublet_detection(
            refs["first_clusters"],
            refs["wnn"],
            from_assay="RNA",
        )


@pytest.mark.parametrize(
    ("clusters", "message"),
    [
        ("clusters_without_selection", "has no cell-selection input"),
        ("clusters_with_malformed_selection", "cell selection is malformed"),
        ("clusters_missing_a_cell", "one label per selected cell"),
        ("clusters_in_rows", "one label per selected cell"),
    ],
)
def test_doublet_detection_rejects_cluster_records_that_do_not_label_the_cells(
    read_only_store,
    clusters,
    message,
) -> None:
    store, refs = read_only_store

    with pytest.raises(ValueError, match=message):
        store.run_doublet_detection(refs[clusters], refs["wnn"], from_assay="RNA")


def test_doublet_detection_requires_a_connectivity_map_built_by_scarf(
    read_only_store,
) -> None:
    store, refs = read_only_store

    with pytest.raises(ValueError, match="requires a build_connectivity_map artifact"):
        store.run_doublet_detection(refs["clusters"], refs["imported_connectivity"])

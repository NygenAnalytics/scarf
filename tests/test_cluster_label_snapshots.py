"""Cluster labels frozen from cell metadata or narrowed from label artifacts."""

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import zarr

from scarf import DataStore, mount_datastore
from scarf.storage.arrays import MISSING_MASK_PREFIX
from scarf.storage.artifacts import ArtifactRef, artifact_group, fingerprint_array
from scarf.storage.errors import ArtifactResolutionError
from scarf.storage.types import as_zarr_group
from tests.fixtures_datastore import build_neighbourhood_graph

N_CELLS = 120
N_GENES = 30
UNLABELLED = 6
CELL_TYPES = np.array(["T", "B", "NK"])[np.arange(N_CELLS) % 3]
DISEASE = np.array(["normal", "covid"])[(np.arange(N_CELLS) // 3) % 2]
LEIDEN = np.arange(N_CELLS) % 3
DONOR = np.arange(N_CELLS) // 10 + 1
NO_DONOR = np.array([1, 4])
COVID_GENES = {f"G{index}" for index in range(20, 25)}


def _write_h5ad(path: Path) -> None:
    """Write cell types with planted markers and covid genes within each type.

    The first ``UNLABELLED`` cells have no ``leiden`` label and the
    ``NO_DONOR`` cells have no ``donor``. Both are nullable integers, which
    the import stores as 0 under a linked missing mask. ``leiden_float``
    holds the ``leiden`` ids as float64 with NaN, as pandas stores integer
    ids with missing values, and the import masks its NaN rows the same way.
    """
    anndata = pytest.importorskip("anndata")
    from scipy.sparse import csr_matrix

    rng = np.random.default_rng(0)
    counts = rng.poisson(1.0, size=(N_CELLS, N_GENES)).astype(np.float32)
    for offset, cell_type in enumerate(["T", "B", "NK"]):
        counts[CELL_TYPES == cell_type, offset * 5 : offset * 5 + 5] += 8
    counts[DISEASE == "covid", 20:25] += 6
    leiden = pd.array(LEIDEN, dtype="Int64")
    leiden[:UNLABELLED] = pd.NA
    leiden_float = LEIDEN.astype(np.float64)
    leiden_float[:UNLABELLED] = np.nan
    donor = pd.array(DONOR, dtype="Int64")
    donor[NO_DONOR] = pd.NA
    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(CELL_TYPES),
            "disease": pd.Categorical(DISEASE),
            "leiden": leiden,
            "leiden_float": leiden_float,
            "donor": donor,
        },
        index=[f"cell{index}" for index in range(N_CELLS)],
    )
    var = pd.DataFrame(
        {"gene_short_name": [f"G{index}" for index in range(N_GENES)]},
        index=[f"ENSG{index:05d}" for index in range(N_GENES)],
    )
    anndata.AnnData(X=csr_matrix(counts), obs=obs, var=var).write_h5ad(path)


@pytest.fixture(scope="module")
def imported_store(
    tmp_path_factory,
) -> tuple[Path, dict[str, ArtifactRef], ArtifactRef]:
    """Import the H5AD once, with both ``leiden`` columns as cluster labels."""
    from scarf.readers import H5adReader
    from scarf.writers import H5adToZarr

    directory = tmp_path_factory.mktemp("cluster_label_snapshots")
    h5ad_path = directory / "cells.h5ad"
    _write_h5ad(h5ad_path)
    store = directory / "cells.zarr"
    reader = H5adReader(str(h5ad_path), cluster_keys=("leiden", "leiden_float"))
    try:
        result = H5adToZarr(reader, zarr_loc=str(store), nthreads=1).dump()
    finally:
        reader.h5.close()
    # One writable open prepares the store for read-only and mounted opens.
    _open(store)
    return store, result.clusterArtifacts, result.cellSelection


@pytest.fixture
def store_path(imported_store, tmp_path) -> Path:
    path = tmp_path / "store.zarr"
    shutil.copytree(imported_store[0], path)
    return path


@pytest.fixture
def leiden(imported_store) -> ArtifactRef:
    return imported_store[1]["leiden"]


@pytest.fixture
def import_selection(imported_store) -> ArtifactRef:
    return imported_store[2]


def _open(path: Path, **options: Any) -> DataStore:
    return DataStore(
        str(path),
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=1,
        **options,
    )


def _selection(ds: DataStore, name: str, mask: np.ndarray) -> ArtifactRef:
    ds.cells.insert(name, mask, overwrite=True)
    return ds.snapshot_cell_selection(name)


def _labels(ds: DataStore, ref: ArtifactRef) -> np.ndarray:
    return np.asarray(ds.load_artifact(ref)["values"][:])


def _snapshots(ds: DataStore) -> list[ArtifactRef]:
    return ds.list_artifacts(kind="cluster_labels", scope="datastore")


def _marker_groups(ds: DataStore, markers: ArtifactRef) -> set[str]:
    table = ds.get_markers(markers, min_score=0, min_frac_exp=0)
    return set(table["group_id"])


def _store_files(path: Path) -> dict[str, bytes]:
    return {
        str(file.relative_to(path)): file.read_bytes()
        for file in path.rglob("*")
        if file.is_file()
    }


def test_snapshot_on_a_mount_finds_disease_markers_within_one_cell_type(
    store_path, tmp_path
):
    source_before = _store_files(store_path)
    target = tmp_path / "target.zarr"
    ds = mount_datastore(
        str(store_path),
        at=str(target),
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=1,
    )
    t_cells = _selection(ds, "is_t_cell", CELL_TYPES == "T")

    labels = ds.snapshot_cluster_labels("disease", cell_selection=t_cells)
    markers = ds.run_marker_search(
        labels, features=ds.select_all_features(from_assay="RNA")
    )

    assert (labels.scope, labels.assay, labels.kind) == (
        "datastore",
        None,
        "cluster_labels",
    )
    t_disease = DISEASE[CELL_TYPES == "T"]
    status = ds.inspect_artifact(labels)
    assert status.operation == "snapshot_cluster_labels"
    assert status.parameters == {"source_column": "disease"}
    assert status.inputs == {
        "cell_selection": t_cells.to_dict(),
        "values_fingerprint": fingerprint_array(t_disease),
    }
    np.testing.assert_array_equal(_labels(ds, labels), t_disease)
    assert _marker_groups(ds, markers) == {"covid", "normal"}
    covid = ds.get_markers(markers, group_id="covid", min_score=0, min_frac_exp=0)
    top = covid.sort_values("score", ascending=False)["feature_name"]
    assert set(top[: len(COVID_GENES)]) == COVID_GENES
    assert _store_files(store_path) == source_before
    path = f"artifacts/cluster_labels/{labels.artifact_id}"
    assert path in zarr.open_group(str(target), mode="r")
    assert path not in zarr.open_group(str(store_path), mode="r")


def test_snapshot_reuses_identical_labels_and_never_reuses_changed_labels(
    store_path,
):
    ds = _open(store_path)
    t_cells = _selection(ds, "is_t_cell", CELL_TYPES == "T")
    first = ds.snapshot_cluster_labels("disease", cell_selection=t_cells)

    assert ds.snapshot_cluster_labels("disease", cell_selection=t_cells) == first
    assert _snapshots(ds) == [first]

    flipped = np.where(DISEASE == "covid", "normal", "covid")
    ds.cells.insert("disease", flipped, overwrite=True)
    changed = ds.snapshot_cluster_labels("disease", cell_selection=t_cells)
    assert changed != first
    np.testing.assert_array_equal(_labels(ds, first), DISEASE[CELL_TYPES == "T"])
    np.testing.assert_array_equal(_labels(ds, changed), flipped[CELL_TYPES == "T"])

    # The same text under another column is a different label source.
    ds.cells.insert("condition", flipped, overwrite=True)
    copied = ds.snapshot_cluster_labels("condition", cell_selection=t_cells)
    assert copied not in {first, changed}
    assert ds.list_artifacts(
        kind="cluster_labels",
        scope="datastore",
        operation="snapshot_cluster_labels",
        parameters={"source_column": "condition"},
    ) == [copied]

    # Integer labels never reuse text labels of equal text.
    codes = np.arange(N_CELLS) % 2
    ds.cells.insert("groups", codes.astype(str), overwrite=True)
    text = ds.snapshot_cluster_labels("groups", cell_selection=t_cells)
    ds.cells.insert("groups", codes, overwrite=True)
    integers = ds.snapshot_cluster_labels("groups", cell_selection=t_cells)
    assert integers != text
    np.testing.assert_array_equal(
        _labels(ds, text), codes[CELL_TYPES == "T"].astype(str)
    )
    np.testing.assert_array_equal(_labels(ds, integers), codes[CELL_TYPES == "T"])


@pytest.mark.parametrize(
    "values",
    [
        np.arange(N_CELLS) % 4,
        (np.arange(N_CELLS) % 4).astype(np.uint8),
        np.arange(N_CELLS) % 2 == 0,
    ],
    ids=["int64", "uint8", "bool"],
)
def test_snapshot_keeps_integer_and_boolean_labels(store_path, values):
    ds = _open(store_path)
    ds.cells.insert("groups", values, overwrite=True)

    labels = ds.snapshot_cluster_labels(
        "groups", cell_selection=ds.snapshot_cell_selection()
    )

    stored = _labels(ds, labels)
    assert stored.dtype == values.dtype
    np.testing.assert_array_equal(stored, values)


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.float16])
def test_snapshot_stores_whole_number_float_labels_as_int64(store_path, dtype):
    ds = _open(store_path)
    all_cells = ds.snapshot_cell_selection()
    codes = np.arange(N_CELLS) % 4
    ds.cells.insert("groups", codes, overwrite=True)
    integers = ds.snapshot_cluster_labels("groups", cell_selection=all_cells)
    ds.cells.insert("groups", codes.astype(dtype), overwrite=True)

    floats = ds.snapshot_cluster_labels("groups", cell_selection=all_cells)

    stored = _labels(ds, floats)
    assert stored.dtype == np.int64
    np.testing.assert_array_equal(stored, codes)
    # The stored labels are equal, so the float column reuses the int64 one.
    assert floats == integers


@pytest.mark.parametrize(
    ("label", "shown"),
    [(0.5, r"0\.5"), (np.inf, "inf"), (2.0**63, r"9\.223372036854776e\+18")],
    ids=["fraction", "infinity", "beyond_int64"],
)
def test_snapshot_rejects_float_labels_that_are_not_whole_numbers(
    store_path, label, shown
):
    ds = _open(store_path)
    all_cells = ds.snapshot_cell_selection()
    values = (np.arange(N_CELLS) % 3).astype(np.float64)
    values[7] = label
    ds.cells.insert("level", values, overwrite=True)

    with pytest.raises(
        TypeError,
        match=r"^Floating-point cluster labels must be whole numbers within the "
        rf"int64 range, but column 'level' holds {shown}$",
    ):
        ds.snapshot_cluster_labels("level", cell_selection=all_cells)
    assert _snapshots(ds) == []


def test_snapshot_identity_ignores_the_declared_text_width(store_path):
    ds = _open(store_path)
    t_cells = _selection(ds, "is_t_cell", CELL_TYPES == "T")
    before = ds.snapshot_cluster_labels("disease", cell_selection=t_cells)

    widened = DISEASE.astype(object)
    widened[np.flatnonzero(CELL_TYPES == "B")[0]] = "covid, second wave"
    ds.cells.insert("disease", widened.astype(str), overwrite=True)

    assert ds.cells.get_dtype("disease") == np.dtype("<U18")
    assert ds.snapshot_cluster_labels("disease", cell_selection=t_cells) == before
    assert _labels(ds, before).dtype == np.dtype("<U6")


@pytest.mark.filterwarnings("ignore::zarr.errors.UnstableSpecificationWarning")
@pytest.mark.parametrize(
    "encode",
    [
        lambda values: np.char.encode(values, "utf-8"),
        lambda values: values.astype(np.dtypes.StringDType()),
    ],
    ids=["bytes", "variable_width"],
)
def test_snapshot_identity_ignores_the_text_encoding(store_path, encode):
    ds = _open(store_path)
    t_cells = _selection(ds, "is_t_cell", CELL_TYPES == "T")
    cell_data = as_zarr_group(ds.zw["cellData"], name="cellData")
    cell_data.create_array("condition", data=encode(DISEASE), overwrite=True)
    encoded = ds.snapshot_cluster_labels("condition", cell_selection=t_cells)

    ds.cells.insert("condition", DISEASE, overwrite=True)

    assert ds.snapshot_cluster_labels("condition", cell_selection=t_cells) == encoded
    assert _labels(ds, encoded).dtype == np.dtype("<U6")


@pytest.mark.parametrize("key", ["leiden", "leiden_float"])
def test_snapshot_narrows_imported_labels_to_the_labelled_cells(
    store_path, imported_store, import_selection, key
):
    imported = imported_store[1][key]
    ds = _open(store_path)
    features = ds.select_all_features(from_assay="RNA")
    with pytest.raises(
        ValueError,
        match=r"^clusters contains missing labels\. Select the labelled cells with "
        r"select_cells\(clusters, include=\[\.\.\.\]\) and freeze their labels "
        r"with snapshot_cluster_labels\(clusters, cell_selection=\.\.\.\)$",
    ):
        ds.run_marker_search(imported, features=features)

    labelled = ds.select_cells(imported, include=[0, 1, 2])
    narrowed = ds.snapshot_cluster_labels(imported, cell_selection=labelled)
    markers = ds.run_marker_search(narrowed, features=features)

    assert ds.snapshot_cluster_labels(imported, cell_selection=labelled) == narrowed
    status = ds.inspect_artifact(narrowed)
    assert status.parameters == {}
    assert status.input_ref("source_labels") == imported
    assert status.input_ref("cell_selection") == labelled
    # Whole float64 ids are stored as the int64 labels of the integer import.
    assert status.inputs["values_fingerprint"] == fingerprint_array(LEIDEN[UNLABELLED:])
    stored = _labels(ds, narrowed)
    assert stored.dtype == np.int64
    np.testing.assert_array_equal(stored, LEIDEN[UNLABELLED:])
    assert _marker_groups(ds, markers) == {"0", "1", "2"}
    lineage = ds.lineage(markers).graph
    assert lineage.edges[imported, narrowed]["inputs"] == ("source_labels",)
    assert lineage.edges[labelled, narrowed]["inputs"] == ("cell_selection",)
    assert import_selection in lineage


def test_snapshot_restricts_labels_to_a_subset_of_clusters(store_path, leiden):
    ds = _open(store_path)
    subset = ds.select_cells(leiden, include=[1, 2])

    labels = ds.snapshot_cluster_labels(leiden, cell_selection=subset)
    markers = ds.run_marker_search(
        labels, features=ds.select_all_features(from_assay="RNA")
    )

    in_subset = (np.arange(N_CELLS) >= UNLABELLED) & (LEIDEN != 0)
    np.testing.assert_array_equal(_labels(ds, labels), LEIDEN[in_subset])
    assert _marker_groups(ds, markers) == {"1", "2"}


@pytest.mark.parametrize(
    ("values", "unlabelled"),
    [
        (np.where(np.arange(N_CELLS) < 2, "", DISEASE), 2),
        (np.where(np.arange(N_CELLS) == 0, "  ", DISEASE), 1),
        (np.where(np.arange(N_CELLS) < 3, None, DISEASE.astype(object)), 3),
        (np.where(np.arange(N_CELLS) < 4, np.nan, LEIDEN), 4),
    ],
    ids=["blank", "whitespace", "none", "nan"],
)
def test_snapshot_rejects_unlabelled_column_values(store_path, values, unlabelled):
    ds = _open(store_path)
    all_cells = ds.snapshot_cell_selection()
    ds.cells.insert("condition", values, overwrite=True)

    with pytest.raises(
        ValueError,
        match=rf"^{unlabelled} of {N_CELLS} selected cells have no label in "
        r"column 'condition'\. Pass a cell_selection of labelled cells only, "
        r"such as snapshot_cell_selection of a boolean column that marks them$",
    ):
        ds.snapshot_cluster_labels("condition", cell_selection=all_cells)
    assert _snapshots(ds) == []


def test_snapshot_rejects_rows_that_a_linked_missing_mask_flags(
    store_path, leiden, import_selection
):
    ds = _open(store_path)
    all_cells = ds.snapshot_cell_selection()
    # The masked rows store 0, which would otherwise pass as a label.
    assert not ds.cells.fetch_all("donor")[NO_DONOR].any()
    assert not _labels(ds, leiden)[:UNLABELLED].any()
    # A masked placeholder is not a label, so its fraction is no type error.
    levels = DONOR.astype(np.float64)
    levels[NO_DONOR] = 0.5
    no_level = np.isin(np.arange(N_CELLS), NO_DONOR)
    cell_data = as_zarr_group(ds.zw["cellData"], name="cellData")
    cell_data.create_array("level", data=levels)
    cell_data.create_array(f"{MISSING_MASK_PREFIX}level", data=no_level)
    cell_data["level"].attrs["missing_mask"] = f"{MISSING_MASK_PREFIX}level"

    for column in ("donor", "level"):
        with pytest.raises(
            ValueError,
            match=rf"^{len(NO_DONOR)} of {N_CELLS} selected cells have no label in "
            rf"column '{column}'",
        ):
            ds.snapshot_cluster_labels(column, cell_selection=all_cells)
    with pytest.raises(
        ValueError,
        match=rf"^{UNLABELLED} of {N_CELLS} selected cells have no label in the "
        r"cluster_labels artifact\. Pass a cell_selection of labelled cells "
        r"only, such as select_cells\(labels, include=\[\.\.\.\]\)$",
    ):
        ds.snapshot_cluster_labels(leiden, cell_selection=import_selection)
    assert _snapshots(ds) == []

    has_donor = _selection(ds, "has_donor", ~no_level)
    for column in ("donor", "level"):
        donors = ds.snapshot_cluster_labels(column, cell_selection=has_donor)
        np.testing.assert_array_equal(_labels(ds, donors), DONOR[~no_level])


def test_snapshot_rejects_invalid_sources_and_selections(store_path):
    ds = _open(store_path)
    t_cells = _selection(ds, "is_t_cell", CELL_TYPES == "T")
    b_cells = _selection(ds, "is_b_cell", CELL_TYPES == "B")
    t_labels = ds.snapshot_cluster_labels("disease", cell_selection=t_cells)
    collected = np.datetime64("2020-03-01") + np.arange(N_CELLS)
    ds.cells.insert("collected", collected, overwrite=True)
    # The type is checked first, so a missing float label is not reported as
    # an unlabelled cell.
    scores = np.linspace(0.0, 1.0, N_CELLS)
    scores[0] = np.nan
    ds.cells.insert("score", scores, overwrite=True)

    with pytest.raises(
        TypeError,
        match=r"^Cluster labels must be text, integer, boolean, or whole-number "
        r"values, but column 'collected' holds datetime64\[D\] values$",
    ):
        ds.snapshot_cluster_labels("collected", cell_selection=t_cells)
    with pytest.raises(TypeError, match=r"column 'score' holds 0\.0252"):
        ds.snapshot_cluster_labels("score", cell_selection=t_cells)
    with pytest.raises(
        ValueError, match="Grouping artifacts must contain categorical cell labels"
    ):
        ds.snapshot_cluster_labels(t_cells, cell_selection=t_cells)
    with pytest.raises(
        ValueError,
        match="cell_selection must be a subset of the artifact cell selection",
    ):
        ds.snapshot_cluster_labels(t_labels, cell_selection=b_cells)
    with pytest.raises(
        ArtifactResolutionError,
        match="Expected datastore-scoped cell_selection artifact",
    ):
        ds.snapshot_cluster_labels("disease", cell_selection=t_labels)
    with pytest.raises(KeyError, match="unknown does not exist"):
        ds.snapshot_cluster_labels("unknown", cell_selection=t_cells)
    for labels in (3, None, CELL_TYPES):
        with pytest.raises(
            TypeError,
            match="labels must be a cell metadata column name or a cell-label "
            "ArtifactRef",
        ):
            ds.snapshot_cluster_labels(labels, cell_selection=t_cells)
    for selection in ("I", None, t_cells.to_dict()):
        with pytest.raises(TypeError, match="cell_selection must be an ArtifactRef"):
            ds.snapshot_cluster_labels("disease", cell_selection=selection)
    assert _snapshots(ds) == [t_labels]


def _edit_a_label(group: zarr.Group) -> None:
    values = group["values"]
    values[0] = "covid" if values[0] == "normal" else "normal"


def _link_a_missing_mask(group: zarr.Group) -> None:
    mask_name = f"{MISSING_MASK_PREFIX}values"
    group.create_array(mask_name, data=np.zeros(group["values"].shape, dtype=bool))
    group["values"].attrs["missing_mask"] = mask_name


@pytest.mark.parametrize(
    "tamper", [_edit_a_label, _link_a_missing_mask], ids=["edited", "masked"]
)
def test_snapshot_never_reuses_a_tampered_payload(store_path, tamper):
    ds = _open(store_path)
    t_cells = _selection(ds, "is_t_cell", CELL_TYPES == "T")
    labels = ds.snapshot_cluster_labels("disease", cell_selection=t_cells)
    tamper(artifact_group(ds.zw, labels))

    fresh = ds.snapshot_cluster_labels("disease", cell_selection=t_cells)

    assert fresh != labels
    np.testing.assert_array_equal(_labels(ds, fresh), DISEASE[CELL_TYPES == "T"])
    assert set(_snapshots(ds)) == {labels, fresh}


def test_read_only_store_reuses_a_snapshot_and_refuses_a_new_one(store_path):
    ds = _open(store_path)
    t_cells = _selection(ds, "is_t_cell", CELL_TYPES == "T")
    labels = ds.snapshot_cluster_labels("disease", cell_selection=t_cells)
    read_only = _open(store_path, zarr_mode="r")

    assert read_only.snapshot_cluster_labels("disease", cell_selection=t_cells) == (
        labels
    )
    with pytest.raises(
        PermissionError,
        match=r"snapshot_cluster_labels requires a DataStore opened with "
        r"zarr_mode='r\+'",
    ):
        read_only.snapshot_cluster_labels("cell_type", cell_selection=t_cells)
    assert _snapshots(read_only) == [labels]


@pytest.mark.filterwarnings("ignore:Cell-level statistical testing:UserWarning")
def test_label_consumers_accept_a_snapshot(store_path, leiden):
    ds = _open(store_path)
    t_cells = _selection(ds, "is_t_cell", CELL_TYPES == "T")
    disease = ds.snapshot_cluster_labels("disease", cell_selection=t_cells)

    bulk = ds.make_bulk(
        disease, aggr_type="sum", feature_label="id", remove_empty_features=False
    )
    counts = np.asarray(ds.RNA.rawData.compute())
    for group in ("covid", "normal"):
        cells = (CELL_TYPES == "T") & (DISEASE == group)
        np.testing.assert_allclose(bulk[group].to_numpy(), counts[cells].sum(axis=0))

    tested = ds.run_statistical_testing(["G20"], disease, skip_save=True)
    assert tested.grouping == disease
    assert tested.n_cells == int((CELL_TYPES == "T").sum())
    assert set(tested.group_order) == {"covid", "normal"}

    covid = ds.select_cells(disease, include=["covid"])
    np.testing.assert_array_equal(
        np.asarray(ds.load_artifact(covid)["values"][:]),
        (CELL_TYPES == "T") & (DISEASE == "covid"),
    )

    labelled = ds.select_cells(leiden, include=[0, 1, 2])
    clusters = ds.snapshot_cluster_labels(leiden, cell_selection=labelled)
    cell_types = ds.snapshot_cluster_labels("cell_type", cell_selection=labelled)
    assert ds.metric_label_concordance(clusters, cell_types) == pytest.approx(1.0)
    relabelled = _labels(ds, ds.smart_label(clusters, cell_types))
    assert all(
        label.startswith(cell_type)
        for label, cell_type in zip(relabelled, CELL_TYPES[UNLABELLED:], strict=True)
    )


def test_membership_strength_takes_labels_frozen_over_the_graph_cell_selection(
    store_path, leiden
):
    ds = _open(store_path)
    labelled = ds.select_cells(leiden, include=[0, 1, 2])
    graph = build_neighbourhood_graph(
        ds,
        cell_selection=labelled,
        features=ds.select_all_features(from_assay="RNA"),
        dims=5,
        k=5,
        local_cache=False,
    )
    # Every graph cell has a label, but the clustering also covers others.
    with pytest.raises(
        ValueError,
        match=r"^clusters contains missing labels\. Freeze the labels over the "
        r"graph's cell selection with snapshot_cluster_labels\(clusters, "
        r"cell_selection=\.\.\.\)\. If some graph cells have no label, first "
        r"build the graph over select_cells\(clusters, include=\[\.\.\.\]\)$",
    ):
        ds.calc_membership_strength(leiden, graph)

    clusters = ds.snapshot_cluster_labels(leiden, cell_selection=labelled)
    strength = _labels(ds, ds.calc_membership_strength(clusters, graph))

    # The clusters are the planted cell types, so neighbours share them.
    assert strength.shape == (N_CELLS - UNLABELLED,)
    assert strength.mean() > 0.9


def test_snapshot_keeps_labels_that_cannot_name_a_marker_group(store_path):
    ds = _open(store_path)
    all_cells = ds.snapshot_cell_selection()
    subtype = np.where(CELL_TYPES == "B", "B", "NK/T")
    ds.cells.insert("subtype", subtype, overwrite=True)

    labels = ds.snapshot_cluster_labels("subtype", cell_selection=all_cells)

    np.testing.assert_array_equal(_labels(ds, labels), subtype)
    assert set(ds.make_bulk(labels).columns) == {"B", "NK/T"}
    with pytest.raises(
        ValueError,
        match=r"'NK/T' cannot name a stored marker group.*snapshot_cluster_labels",
    ):
        ds.run_marker_search(labels, features=ds.select_all_features(from_assay="RNA"))
    assert ds.list_artifacts(kind="marker_table", from_assay="RNA") == []

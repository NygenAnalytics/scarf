"""``DataStore.load_cell_values`` over the one table of cell-aligned values."""

import hashlib
import re
import shutil
import tracemalloc
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import scarf.datastore.base_datastore as base_datastore_module
import scarf.metadata.selection as selection_module
from scarf import DataStore
from scarf.embeddings.imported import write_imported_coordinates
from scarf.metadata import CellValues
from scarf.metadata.artifacts import plan_cell_data_artifact, write_cell_data_artifact
from scarf.metadata.selection import (
    CELL_VALUE_NAMES,
    CellValueSpec,
    cell_value_spec,
    resolve_cell_aligned_artifact,
)
from scarf.storage.arrays import MISSING_MASK_PREFIX
from scarf.storage.artifact_writer import artifact_transaction
from scarf.storage.artifacts import ArtifactRef, fingerprint_array, new_artifact_id
from scarf.storage.errors import ArtifactResolutionError
from scarf.storage.selections import resolve_metadata_snapshot, snapshot_run_metadata
from scarf.trajectory.artifacts import load_cell_artifact_values
from tests.storage_helpers import write_count_store

N_CELLS = 8
# The cells of the artifacts' own selection, and a subset of them.
SELECTED = np.arange(6)
SUBSET = np.array([0, 2, 4])

# Arrays that hold a row of values per cell.
_TWO_DIMENSIONAL = frozenset(
    {
        ("embedding", "values"),
        ("enrichment_scores", "scores"),
        ("fate_map", "probabilities"),
        ("imported_coordinates", "data"),
        ("label_transfer", "vote_class_codes"),
        ("label_transfer", "vote_class_fractions"),
    }
)


def _open(path: Path) -> DataStore:
    return DataStore(
        str(path), default_assay="RNA", min_features_per_cell=0, nthreads=1
    )


@pytest.fixture(scope="module")
def store_template(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("cell_values") / "store.zarr"
    counts = np.random.default_rng(3).poisson(4.0, size=(N_CELLS, 5)) + 1
    write_count_store(str(path), {"RNA": counts}, "uint16")
    store = _open(path)
    rows = np.arange(N_CELLS)
    store.cells.insert("selected", np.isin(rows, SELECTED))
    store.cells.insert("subset", np.isin(rows, SUBSET))
    store.cells.insert("outside", np.isin(rows, [1, 7]))
    store.cells.insert("cluster", np.array(list("aabbccdd")))
    return path


@pytest.fixture
def store(store_template, tmp_path) -> DataStore:
    target = tmp_path / "store.zarr"
    shutil.copytree(store_template, target)
    return _open(target)


def _declared(kind: str) -> set[str]:
    spec = CELL_VALUE_NAMES[kind]
    return {spec.name, *spec.alternatives}


def _example(kind: str, name: str, categorical: bool, offset: int) -> np.ndarray:
    rows = np.arange(len(SELECTED))
    if (kind, name) in _TWO_DIMENSIONAL:
        matrix = np.stack([rows, rows + 10], axis=1)
        return matrix.astype(np.int64) if categorical else matrix + 0.25 + offset
    if categorical:
        return np.asarray([f"{name}-{row % 3}" for row in rows])
    return rows + 0.5 + offset


def _write(
    store: DataStore,
    kind: str,
    arrays: dict[str, np.ndarray],
    *,
    case: str = "",
) -> ArtifactRef:
    planned = plan_cell_data_artifact(
        store.zw,
        scope="assay",
        assay="RNA",
        kind=kind,
        operation="test_cell_values",
        parameters={"case": case},
        inputs={},
        execution_options={},
        cell_selection=store.snapshot_cell_selection("selected"),
        arrays={name: (values.shape, None) for name, values in arrays.items()},
    )
    write_cell_data_artifact(store.zw, planned, arrays)
    return planned.ref


def _write_declared(
    store: DataStore, kind: str
) -> tuple[ArtifactRef, dict[str, np.ndarray]]:
    """Store every per-cell array that the table declares for ``kind``."""
    spec = CELL_VALUE_NAMES[kind]
    arrays = {
        name: _example(kind, name, categorical, offset)
        for offset, (name, categorical) in enumerate(
            [(spec.name, spec.categorical), *spec.alternatives.items()]
        )
    }
    return _write(store, kind, arrays), arrays


def _write_masked(
    store: DataStore,
    kind: str,
    name: str,
    values: np.ndarray,
    missing: np.ndarray,
    *,
    case: str,
    cells: str = "selected",
) -> ArtifactRef:
    """Store one array and the linked mask of its missing values."""
    planned = plan_cell_data_artifact(
        store.zw,
        scope="assay",
        assay="RNA",
        kind=kind,
        operation="test_cell_values",
        parameters={"case": case},
        inputs={},
        execution_options={},
        cell_selection=store.snapshot_cell_selection(cells),
        arrays={name: (values.shape, None)},
    )
    mask_name = f"{MISSING_MASK_PREFIX}{name}"
    with artifact_transaction(store.zw, planned) as group:
        group.create_array(name, data=values)
        group.create_array(mask_name, data=missing)
        group[name].attrs["missing_mask"] = mask_name
    return planned.ref


def _cell_ids(rows: np.ndarray) -> list[str]:
    return [f"cell{row}" for row in rows]


# Labels in a named canonical array, one numeric value, and a row per cell.
@pytest.mark.parametrize("kind", ["cell_cycle", "doublet_score", "embedding"])
def test_every_listed_kind_reads_its_canonical_values(store, kind) -> None:
    ref, arrays = _write_declared(store, kind)
    spec = CELL_VALUE_NAMES[kind]

    loaded = store.load_cell_values(ref)

    assert isinstance(loaded, CellValues)
    assert loaded.source == ref
    assert (loaded.value, loaded.categorical) == (spec.name, spec.categorical)
    np.testing.assert_array_equal(loaded.values, arrays[spec.name])
    np.testing.assert_array_equal(loaded.cell_idx, SELECTED)
    assert loaded.cell_ids.tolist() == _cell_ids(SELECTED)
    assert loaded.cell_selection == store.snapshot_cell_selection("selected")
    assert loaded.missing is None
    table = loaded.to_pandas()
    assert table.index.tolist() == _cell_ids(SELECTED)
    np.testing.assert_array_equal(table.to_numpy(), arrays[spec.name])


def test_value_reads_another_per_cell_array_and_refuses_the_rest(store) -> None:
    # Sampling has labels and measurements among its other per-cell arrays.
    ref, arrays = _write_declared(store, "sampling")
    spec = CELL_VALUE_NAMES["sampling"]
    canonical = store.load_cell_values(ref, value=spec.name)
    assert (canonical.value, canonical.categorical) == (spec.name, spec.categorical)
    for name, categorical in spec.alternatives.items():
        loaded = store.load_cell_values(ref, value=name)
        assert (loaded.value, loaded.categorical) == (name, categorical)
        np.testing.assert_array_equal(loaded.values, arrays[name])

    with pytest.raises(
        ValueError,
        match="'sampling' artifacts have no per-cell array 'sample'; their "
        "per-cell arrays are sampled, seeds, density, mean_snn",
    ):
        store.load_cell_values(ref, value="sample")
    with pytest.raises(TypeError, match="value must be a string or None"):
        store.load_cell_values(ref, value=1)

    # A stored array that is not a per-cell array of the kind is refused even
    # when it has one row per cell, as a label transfer's classes can.
    transfer = _write(
        store,
        "label_transfer",
        {
            "labels": np.array(list("abcabc")),
            "categories": np.array(list("abcdef")),
        },
        case="classes",
    )
    with pytest.raises(ValueError, match="no per-cell array 'categories'"):
        store.load_cell_values(transfer, value="categories")


def test_cell_selection_reads_a_contained_subset_in_cell_order(store) -> None:
    scores, score_arrays = _write_declared(store, "doublet_score")
    layout, layout_arrays = _write_declared(store, "embedding")
    subset = store.snapshot_cell_selection("subset")

    loaded = store.load_cell_values(scores, cell_selection=subset)
    coordinates = store.load_cell_values(layout, cell_selection=subset)

    np.testing.assert_array_equal(loaded.values, score_arrays["values"][SUBSET])
    np.testing.assert_array_equal(loaded.cell_idx, SUBSET)
    assert loaded.cell_ids.tolist() == _cell_ids(SUBSET)
    assert loaded.cell_selection == subset
    np.testing.assert_array_equal(coordinates.values, layout_arrays["values"][SUBSET])
    with pytest.raises(ValueError, match="must be a subset"):
        store.load_cell_values(
            scores, cell_selection=store.snapshot_cell_selection("outside")
        )
    with pytest.raises(TypeError, match="cell_selection must be an ArtifactRef"):
        store.load_cell_values(scores, cell_selection="subset")


def test_rows_recorded_missing_are_flagged_and_shown_missing(store) -> None:
    missing = np.array([False, True, False, True, False, False])
    labels = _write_masked(
        store,
        "cluster_labels",
        "values",
        np.array([3, 0, 1, 0, 2, 1]),
        missing,
        case="labels",
    )
    scores = _write_masked(
        store,
        "doublet_score",
        "values",
        np.arange(6.0),
        missing,
        case="scores",
    )

    loaded_labels = store.load_cell_values(labels)
    loaded_scores = store.load_cell_values(scores)
    subset = store.load_cell_values(
        labels, cell_selection=store.snapshot_cell_selection("subset")
    )

    np.testing.assert_array_equal(loaded_labels.missing, missing)
    # The stored placeholders stay in the values.
    np.testing.assert_array_equal(loaded_labels.values, [3, 0, 1, 0, 2, 1])
    series = loaded_labels.to_pandas()
    assert series.name == "values"
    assert series.index.name == "ids"
    assert series.index.tolist() == _cell_ids(SELECTED)
    assert series.tolist() == [3, None, 1, None, 2, 1]
    numeric = loaded_scores.to_pandas()
    assert numeric.dtype == np.float64
    np.testing.assert_array_equal(numeric.isna(), missing)
    np.testing.assert_array_equal(subset.missing, [False, False, False])
    assert subset.to_pandas().tolist() == [3, 1, 2]


def test_two_dimensional_values_keep_one_row_and_one_mask_entry_per_cell(
    store,
) -> None:
    layout, arrays = _write_declared(store, "embedding")
    probabilities = np.arange(12.0).reshape(6, 2)
    row_missing = np.zeros((6, 2), dtype=bool)
    row_missing[2] = True
    masked = _write_masked(
        store,
        "fate_map",
        "probabilities",
        probabilities,
        row_missing,
        case="rows",
    )
    part_missing = np.zeros((6, 2), dtype=bool)
    part_missing[3, 0] = True
    partly_masked = _write_masked(
        store,
        "fate_map",
        "probabilities",
        probabilities,
        part_missing,
        case="part",
    )

    loaded = store.load_cell_values(layout)
    loaded_masked = store.load_cell_values(masked)

    assert loaded.values.shape == (6, 2)
    frame = loaded.to_pandas()
    assert isinstance(frame, pd.DataFrame)
    assert frame.index.tolist() == _cell_ids(SELECTED)
    assert frame.columns.tolist() == [0, 1]
    assert frame.columns.name == "values"
    np.testing.assert_array_equal(frame.to_numpy(), arrays["values"])
    np.testing.assert_array_equal(
        loaded_masked.missing, [False, False, True, False, False, False]
    )
    masked_frame = loaded_masked.to_pandas()
    assert masked_frame.loc["cell2"].isna().all()
    assert not masked_frame.drop(index="cell2").isna().to_numpy().any()
    with pytest.raises(ValueError, match="only some values of a cell as missing"):
        store.load_cell_values(partly_masked)
    with pytest.raises(ValueError, match="one or two dimensions"):
        replace(loaded, values=loaded.values[:, :, None]).to_pandas()


def test_a_complete_cell_selection_and_a_reduction_are_refused(store) -> None:
    with pytest.raises(ValueError, match="'cell_selection' is not a cell-aligned"):
        store.load_cell_values(store.snapshot_cell_selection("selected"))
    reduction = ArtifactRef(
        scope="assay", assay="RNA", kind="reduction", artifact_id=new_artifact_id()
    )
    with pytest.raises(ValueError, match="take their cells from their lineage"):
        store.load_cell_values(reduction)
    with pytest.raises(TypeError, match="ref must be an ArtifactRef"):
        store.load_cell_values("values")


def test_values_beyond_the_memory_budget_are_refused_before_anything_is_read(
    store, monkeypatch
) -> None:
    scores, _ = _write_declared(store, "doublet_score")
    reads: list[str] = []

    def recorded(name, function):
        def record(*args, **kwargs):
            reads.append(name)
            return function(*args, **kwargs)

        return record

    for module, name in (
        (selection_module, "_selection_indices"),
        (selection_module, "read_array_rows_chunkwise"),
        (base_datastore_module, "read_metadata_rows_chunkwise"),
    ):
        monkeypatch.setattr(module, name, recorded(name, getattr(module, name)))
    # 48 bytes hold the six float64 values but not what reading them is
    # charged, so no cell row, value or cell id is read.
    store.memoryBytes = 6 * 8

    with pytest.raises(MemoryError, match="but the memory budget is 48 bytes"):
        store.load_cell_values(scores)
    assert reads == []


def _required_bytes(store: DataStore, ref: ArtifactRef, **options) -> int:
    """Return the bytes that ``load_cell_values`` charges for a read."""
    budget = store.memoryBytes
    store.memoryBytes = 1
    try:
        with pytest.raises(MemoryError) as raised:
            store.load_cell_values(ref, **options)
    finally:
        store.memoryBytes = budget
    found = re.search(r"needs about (\d+) bytes", str(raised.value))
    assert found is not None, raised.value
    return int(found.group(1))


@pytest.fixture
def large_store(tmp_path) -> DataStore:
    """A store of 8,000 cells whose column ``half`` selects every second one."""
    path = tmp_path / "large.zarr"
    n_cells = 8_000
    counts = np.ones((n_cells, 2), dtype=np.uint16)
    write_count_store(str(path), {"RNA": counts}, "uint16")
    store = DataStore(
        str(path), default_assay="RNA", min_features_per_cell=-1, nthreads=1
    )
    store.cells.insert("half", np.arange(n_cells) % 2 == 0)
    return store


def test_the_charged_bytes_bound_what_load_cell_values_allocates(
    large_store,
) -> None:
    store = large_store
    n_cells = store.cells.N
    missing = np.arange(n_cells) % 5 == 0
    scores = _write_masked(
        store,
        "doublet_score",
        "values",
        np.linspace(0.0, 1.0, n_cells),
        missing,
        case="budget",
        cells="I",
    )
    half = store.snapshot_cell_selection("half")

    for options in ({}, {"cell_selection": half}):
        required = _required_bytes(store, scores, **options)
        store.memoryBytes = required - 1
        with pytest.raises(MemoryError, match=f"needs about {required} bytes"):
            store.load_cell_values(scores, **options)
        store.memoryBytes = required
        tracemalloc.start()
        try:
            loaded = store.load_cell_values(scores, **options)
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        expected = n_cells if not options else n_cells // 2
        assert loaded.values.shape == (expected,)
        assert loaded.cell_ids.shape == (expected,)
        # Everything the read allocated, the cell ids and the int64 cell
        # rows and positions among them, fits the bytes it was charged.
        assert peak <= required, (peak, required)


def test_a_snapshot_without_a_cell_selection_is_refused_as_not_cell_aligned(
    store,
) -> None:
    cell_snapshot = snapshot_run_metadata(
        store.zw,
        table_path="cellData",
        id_column="ids",
        columns=("cluster",),
        axis="cell",
    )
    feature_snapshot = snapshot_run_metadata(
        store.zw,
        table_path="RNA/featureData",
        id_column="ids",
        columns=("names",),
        axis="feature",
        assay="RNA",
    )
    other_snapshot = resolve_metadata_snapshot(
        store.zw,
        values=np.arange(N_CELLS, dtype=np.float64),
        row_ids=np.asarray(store.cells.fetch_all("ids")),
        operation="test_cell_values",
        parameters={},
        inputs={},
        source_columns=["values"],
    )

    # select_cells reads its values with the same reader.
    for read, snapshot, remedy in (
        (store.load_cell_values, cell_snapshot, "run.cells.fetch(column)"),
        (store.load_cell_values, feature_snapshot, "run.features.fetch(column)"),
        (store.load_cell_values, other_snapshot, "Open it with load_artifact"),
        (
            lambda ref: store.select_cells(ref, include=["a"]),
            cell_snapshot,
            "run.cells.fetch(column)",
        ),
    ):
        with pytest.raises(
            ValueError, match="rows are not aligned to a recorded cell selection"
        ) as raised:
            read(snapshot)
        # A run's snapshot is valid; it is not cell-aligned, not corrupt.
        assert not isinstance(raised.value, ArtifactResolutionError)
        assert remedy in str(raised.value)


def test_select_cells_admits_its_read_against_the_memory_budget(
    store, monkeypatch
) -> None:
    """select_cells charges its read as load_cell_values does, before reading."""
    scores, arrays = _write_declared(store, "doublet_score")
    reads: list[object] = []
    original = selection_module.read_array_rows_chunkwise

    def recording(array, rows):
        reads.append(array)
        return original(array, rows)

    monkeypatch.setattr(selection_module, "read_array_rows_chunkwise", recording)
    budget = store.memoryBytes
    store.memoryBytes = 64

    with pytest.raises(MemoryError, match="but the memory budget is 64 bytes"):
        store.select_cells(scores, low=0.0)
    assert reads == []

    store.memoryBytes = budget
    selected = store.select_cells(scores, low=float(np.median(arrays["values"])))
    np.testing.assert_array_equal(
        store.load_artifact(selected)["values"][:],
        np.isin(
            np.arange(N_CELLS),
            SELECTED[arrays["values"] >= float(np.median(arrays["values"]))],
        ),
    )


def test_a_snapshot_whose_cell_selection_input_is_malformed_stays_corrupt(
    monkeypatch,
) -> None:
    snapshot = ArtifactRef(
        scope="datastore",
        assay=None,
        kind="metadata_snapshot",
        artifact_id=new_artifact_id(),
    )
    status = SimpleNamespace(
        exists=True,
        complete=True,
        operation="snapshot_pseudotime_source_sink",
        parameters={"shape": [6]},
        inputs={"cell_selection": {"type": "wrong"}},
    )
    monkeypatch.setattr(selection_module, "inspect_artifact", lambda *_: status)

    with pytest.raises(
        ArtifactResolutionError, match="malformed 'cell_selection'"
    ) as raised:
        resolve_cell_aligned_artifact(None, snapshot)
    assert raised.value.code == "corrupt_payload"


def test_real_producers_store_the_declared_arrays(store) -> None:
    cells = store.snapshot_cell_selection("selected")
    ids = np.asarray(store.cells.fetch_all("ids"))[SELECTED]
    clusters = store.snapshot_cluster_labels("cluster", cell_selection=cells)
    vector = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    snapshot = resolve_metadata_snapshot(
        store.zw,
        values=vector,
        row_ids=ids,
        operation="snapshot_pseudotime_source_sink",
        parameters={},
        inputs={"cell_selection": cells},
        source_columns=["ss_vec"],
    )
    coordinates = np.arange(12, dtype=np.float32).reshape(6, 2)
    imported = write_imported_coordinates(
        store.zw,
        assay="RNA",
        dimreduc_key="pca",
        role="pca",
        coordinates=coordinates,
        source_digest=hashlib.sha256(b"cell-values").digest(),
        payload_fingerprints={"data": fingerprint_array(coordinates)},
        source_cell_ids=ids,
        cell_selection=cells,
        block_rows=2,
    )
    enrichment = store.run_aucell(
        pd.DataFrame({"source": ["set", "set"], "target": ["RNA0", "RNA1"]}),
        cells,
        features=store.select_all_features(from_assay="RNA"),
        tmin=1,
        n_up=2,
    )
    scores = np.asarray(store.load_artifact(enrichment)["scores"][:])

    for ref, expected in (
        (clusters, np.array(list("aabbcc"))),
        (snapshot, vector),
        (imported, coordinates),
        (enrichment, scores),
    ):
        loaded = store.load_cell_values(ref)
        np.testing.assert_array_equal(loaded.values, expected)
        assert loaded.cell_ids.tolist() == _cell_ids(SELECTED)
        group = store.load_artifact(ref)
        per_cell = {
            name
            for name in group.array_keys()
            if group[name].shape[0] == len(SELECTED)
            and not name.startswith(MISSING_MASK_PREFIX)
        }
        # An enrichment also stores the cell row of each score row, which
        # CellValues.cell_idx gives.
        assert per_cell - {"cell_index"} == _declared(ref.kind)
    assert scores.shape == (len(SELECTED), 1)


def test_label_transfers_declare_every_per_cell_array() -> None:
    from scarf.mapping.label_transfer import _transfer_array_requirements

    n_cells = 7
    requirements = _transfer_array_requirements(
        n_cells, 3, n_classes=2, class_dtype=np.dtype("<U4")
    )
    per_cell = {
        requirement.name
        for requirement in requirements
        if requirement.shape is not None
        and requirement.shape[0] == n_cells
        and not requirement.name.startswith(MISSING_MASK_PREFIX)
    }
    assert per_cell == _declared("label_transfer")


def test_select_cells_plots_and_trajectories_read_the_canonical_array(
    store,
) -> None:
    """select_cells, plot groupings and trajectories read a kind's canonical array."""
    from scarf.plotting._data import _resolve_grouping

    pseudotime, arrays = _write_declared(store, "pseudotime")
    layout, _ = _write_declared(store, "embedding")
    transfer, transfer_arrays = _write_declared(store, "label_transfer")
    sampling = _write(
        store,
        "sampling",
        {
            "sampled": np.array([True, False, True, False, False, True]),
            "seeds": np.zeros(6, dtype=bool),
            "density": np.ones(6),
            "mean_snn": np.ones(6),
        },
        case="sampled",
    )

    late = store.select_cells(pseudotime, low=3.0)
    sampled = store.select_cells(sampling, include=[True])

    np.testing.assert_array_equal(
        store.load_artifact(late)["values"][:], np.isin(np.arange(N_CELLS), [3, 4, 5])
    )
    np.testing.assert_array_equal(
        store.load_artifact(sampled)["values"][:],
        np.isin(np.arange(N_CELLS), [0, 2, 5]),
    )

    _, cell_idx, columns, missing = _resolve_grouping(
        store, group_by=None, groups=pseudotime, cell_key="I"
    )
    np.testing.assert_array_equal(cell_idx, SELECTED)
    np.testing.assert_array_equal(columns[0], arrays["pseudotime"])
    assert missing is None
    with pytest.raises(ValueError, match="one value per source-selected cell"):
        _resolve_grouping(store, group_by=None, groups=layout, cell_key="I")

    labels, selection, missing = load_cell_artifact_values(store.zw, transfer)
    np.testing.assert_array_equal(labels, transfer_arrays["labels"])
    assert selection == store.snapshot_cell_selection("selected")
    assert missing is None


def test_cell_value_specs_are_immutable_and_exact() -> None:
    spec = CELL_VALUE_NAMES["cell_cycle"]
    assert spec == CellValueSpec("phase", True, {"s_score": False, "g2m_score": False})
    with pytest.raises(TypeError):
        spec.alternatives["phase"] = True  # type: ignore[index]
    with pytest.raises(TypeError):
        CELL_VALUE_NAMES["normalized"] = spec  # type: ignore[index]
    with pytest.raises(ValueError, match="cannot also be an alternative"):
        CellValueSpec("values", False, {"values": False})
    with pytest.raises(TypeError, match="must be booleans"):
        CellValueSpec("values", 1)  # type: ignore[arg-type]
    assert cell_value_spec("pseudotime").name == "pseudotime"
    with pytest.raises(ValueError, match="'normalized' is not a cell-aligned"):
        cell_value_spec("normalized")

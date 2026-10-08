"""Operations refuse to read an assay over cells that it did not measure.

The fixture merges a source with RNA, ATAC, HTO, and ADT assays and a source
with ADT only. The merged RNA, ATAC, and HTO assays measured the cells of the
first source only: their membership columns ``<assay>_I`` are False for the
cells of the second source, which hold zero counts of those assays. Every
operation that reads an assay's values over cells refuses unmeasured cells
with ``UnmeasuredCellsError`` before it writes or reuses anything, on a
writable and on a read-only store, and runs over the cells that
``select_measured_cells`` keeps. Display reads show unmeasured cells as
missing values instead of zero expression.
"""

import pickle
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import zarr

from scarf import DataStore
from scarf.datastore._pipeline_recipe import resolve_pipeline_recipe
from scarf.datastore.base_datastore import BaseDataStore
from scarf.features.values import (
    fetch_normalized_feature_matrix,
    iter_normalized_feature_blocks,
    resolve_feature,
)
from scarf.mapping.reference import MappingReference
from scarf.merge import DataStoreMerge
from scarf.metadata import MetaData
from scarf.metadata.membership import (
    UnmeasuredCellsError,
    measured_rows,
    membership_attributes,
    require_measured_cells,
)
from scarf.metadata.selection import CellField, FeatureRef, NormalizationSpec
from scarf.storage.artifacts import ArtifactRef, fingerprint_array, list_artifacts
from scarf.writers._store import write_membership_column
from tests.fixtures_datastore import build_neighbourhood_graph
from tests.storage_helpers import write_count_store
from tests.test_hto import planted_hto_counts
from tests.test_mapping_reference import mapping_counts

_N_FULL = 64
_N_ADT = 16
_N_CELLS = _N_FULL + _N_ADT
# The merged assays that measured only the cells of the first source.
_PARTIAL = ("RNA", "ATAC", "HTO")


@dataclass(frozen=True)
class _MeasuredStore:
    """A merged store whose RNA, ATAC, and HTO assays measured some cells.

    ``refs`` holds the artifacts that the entry points read: ``all``, every
    cell; ``measured_<assay>``, the cells that an assay measured; label,
    graph, pseudotime, and diffusion artifacts over either; and ``legacy_*``
    artifacts that an earlier release computed over every cell, RNA's
    unmeasured cells included, which this release refuses to compute.
    """

    path: Path
    reference: MappingReference
    refs: dict[str, ArtifactRef]

    def open(self, zarr_mode: str = "r+") -> DataStore:
        return DataStore(
            str(self.path),
            default_assay="RNA",
            min_features_per_cell=-1,
            nthreads=1,
            zarr_mode=zarr_mode,
        )


def _input(store: DataStore, ref: ArtifactRef, name: str) -> ArtifactRef:
    return store.inspect_artifact(ref).input_ref(name)


def _write_sources(base: Path) -> tuple[DataStore, DataStore]:
    rng = np.random.default_rng(5)
    programs = np.arange(_N_FULL) % 4
    peaks = rng.binomial(1, 0.15, size=(_N_FULL, 40))
    for program in range(4):
        peaks[programs == program, 5 * program : 5 * program + 5] = 1
    hashtags, _truth = planted_hto_counts(n_singlets=18, n_doublets=5, n_negatives=5)
    write_count_store(
        str(base / "full.zarr"),
        {
            "RNA": mapping_counts(),
            "ATAC": peaks,
            "HTO": hashtags.to_numpy(),
            "ADT": rng.integers(1, 20, size=(_N_FULL, 3)),
        },
        "uint32",
    )
    write_count_store(
        str(base / "adt.zarr"),
        {"ADT": rng.integers(1, 20, size=(_N_ADT, 3))},
        "uint32",
    )
    full = DataStore(
        str(base / "full.zarr"),
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=1,
    )
    adt = DataStore(
        str(base / "adt.zarr"),
        default_assay="ADT",
        min_features_per_cell=0,
        nthreads=1,
    )
    return full, adt


def _mapping_reference(full: DataStore) -> MappingReference:
    """Build a reference from the RNA source, a store apart from the query."""
    cells = full.snapshot_cell_selection("I")
    features = full.set_feature_selection(
        from_assay="RNA", feature_indexes=list(range(26))
    )
    reduction = full.run_pca(
        full.run_normalization(cells, features), dims=4, local_cache=False
    )
    neighbors = full.query_neighbors(
        full.build_ann_index(reduction), coordinates=reduction, k=5
    )
    return full.get_mapping_reference(full.build_mapping_reference(neighbors))


def _graph(
    store: DataStore, assay: str, cells: ArtifactRef, features: ArtifactRef
) -> ArtifactRef:
    return build_neighbourhood_graph(
        store,
        from_assay=assay,
        cell_selection=cells,
        features=features,
        dims=2 if assay == "ADT" else 3,
        k=5,
        local_cache=False,
    )


@pytest.fixture(scope="module")
def measured(tmp_path_factory) -> _MeasuredStore:
    base = tmp_path_factory.mktemp("measured_cells")
    full, adt = _write_sources(base)
    path = base / "merged.zarr"
    DataStoreMerge([full, adt], str(path), ["full", "adt"], nthreads=1).dump()
    reference = _mapping_reference(full)
    store = DataStore(
        str(path), default_assay="RNA", min_features_per_cell=-1, nthreads=1
    )
    membership = np.asarray(store.cells.fetch_all("RNA_I"), dtype=bool)
    assert int(membership.sum()) == _N_FULL
    store.cells.insert(
        "program", np.array([f"p{index % 4}" for index in range(_N_CELLS)])
    )
    store.cells.insert("adt_cells", ~membership)
    # A live layout: x is the cell's row.
    store.cells.insert("xy1", np.arange(_N_CELLS, dtype=np.float64))
    store.cells.insert("xy2", (np.arange(_N_CELLS) % 7).astype(np.float64))

    refs: dict[str, ArtifactRef] = {"all": store.snapshot_cell_selection("I")}
    for assay in _PARTIAL:
        refs[f"measured_{assay}"] = store.select_measured_cells(
            assay, cell_selection=refs["all"]
        )
    refs["adt_cells"] = store.snapshot_cell_selection("adt_cells")
    refs["rna_features"] = store.select_all_features(from_assay="RNA")
    refs["rna_some"] = store.set_feature_selection(
        from_assay="RNA", feature_indexes=[0, 1, 2]
    )
    refs["labels_all"] = store.snapshot_cluster_labels(
        "program", cell_selection=refs["all"]
    )
    refs["labels_measured"] = store.snapshot_cluster_labels(
        "program", cell_selection=refs["measured_RNA"]
    )
    adt_features = store.select_all_features(from_assay="ADT")
    # ADT measured every cell, so its graph over all of them is valid.
    adt_all = _graph(store, "ADT", refs["all"], adt_features)
    refs["adt_neighbors_all"] = _input(store, adt_all, "neighbors")
    refs["pseudotime_all"] = store.run_pseudotime_scoring(
        adt_all, source_sink=refs["labels_all"], sources=["p0"], sinks=["p3"]
    )
    adt_measured = _graph(store, "ADT", refs["measured_RNA"], adt_features)
    refs["adt_neighbors_measured"] = _input(store, adt_measured, "neighbors")
    rna_measured = _graph(store, "RNA", refs["measured_RNA"], refs["rna_features"])
    refs["graph_measured"] = rna_measured
    refs["rna_neighbors_measured"] = _input(store, rna_measured, "neighbors")
    refs["clusters_measured"] = store.run_leiden_clustering(rna_measured)
    refs["pseudotime_measured"] = store.run_pseudotime_scoring(
        rna_measured,
        source_sink=refs["labels_measured"],
        sources=["p0"],
        sinks=["p3"],
    )
    refs["diffusion_measured"] = store.run_diffusion_operator(rna_measured)
    with pytest.MonkeyPatch.context() as patch:
        # An earlier release computed these over the cells that RNA did not
        # measure; this release refuses to.
        patch.setattr(
            BaseDataStore, "_require_measured_cells", lambda *_args, **_kwargs: None
        )
        legacy = _graph(store, "RNA", refs["all"], refs["rna_features"])
    refs["legacy_graph"] = legacy
    refs["legacy_neighbors"] = _input(store, legacy, "neighbors")
    refs["legacy_normalized"] = _input(
        store, _input(store, refs["legacy_neighbors"], "coordinates"), "normalized"
    )
    refs["legacy_clusters"] = store.run_leiden_clustering(legacy)
    refs["legacy_diffusion"] = store.run_diffusion_operator(legacy)
    return _MeasuredStore(path=path, reference=reference, refs=refs)


def _tree(path: Path) -> dict[str, tuple[int, int]]:
    """Return the size and modification time of every file of a local store."""
    tree = {}
    for file in path.rglob("*"):
        info = file.stat()
        if stat.S_ISREG(info.st_mode):
            tree[str(file.relative_to(path))] = (info.st_size, info.st_mtime_ns)
    return tree


_NETWORK = pd.DataFrame(
    {"source": ["s"] * 8, "target": [f"RNA{index}" for index in range(8)]}
)
_CELL_CYCLE: dict[str, Any] = {
    "s_genes": ["RNA0", "RNA1", "RNA2"],
    "g2m_genes": ["RNA5", "RNA6", "RNA7"],
    "n_bins": 5,
}
_HVG: dict[str, Any] = {"top_n": 5, "min_cells": 1, "max_cells": np.inf}
_AGGREGATION: dict[str, Any] = {"n_clusters": 2, "window_size": 10, "chunk_size": 5}
_PIPELINE: dict[str, Any] = {
    "filtering": False,
    "cell_cycle": False,
    "umap": False,
    "paris": False,
    "doublets": False,
    "markers": False,
    "hvg_count": 10,
    "pca_dims": 3,
    "neighbors_k": 5,
    "leiden": {"partitions": [0.5, 1.0]},
    # Infinity is no setting, so a cutoff above every cell count keeps the
    # genes that these small stores detect in nearly every cell.
    "params": {"hvg": {"min_cells": 1, "max_cells": _N_CELLS + 1}},
}


def _recipe(store: DataStore, cell_key: str, **settings: Any) -> Any:
    return resolve_pipeline_recipe(
        store,
        assay=None,
        label=None,
        cell_key=cell_key,
        harmony_batch_columns=None,
        snapshot_columns=(),
        **{**_PIPELINE, **settings},
    )


def _narrowed_labels(store: DataStore, labels: ArtifactRef) -> ArtifactRef:
    """Narrow labels to the cells that RNA measured, as the refusal says."""
    cells = _input(store, labels, "cell_selection")
    return store.snapshot_cluster_labels(
        labels,
        cell_selection=store.select_measured_cells("RNA", cell_selection=cells),
    )


type _Call = Callable[[DataStore, _MeasuredStore], Any]


@dataclass(frozen=True)
class _Entry:
    """An entry point, a request over unmeasured cells, and a measured one."""

    operation: str
    assay: str
    refuse: _Call
    accept: _Call
    remedy: str
    # Whether the request reads every cell of the store.
    every_cell: bool = True


_SELECTION = "select_measured_cells("
_LABELS = "snapshot_cluster_labels(labels, cell_selection=select_measured_cells("
_SELECTION_KEYWORD = "Pass `cell_selection=select_measured_cells('RNA', "
_GRAPH = "Build the graph over a selection of measured cells"
_CELL_KEY = (
    "Pass as cell_key a boolean cell column that is True only for the cells of "
    "'I' that it measured, such as 'RNA_measured' after "
    "`ds.cells.insert('RNA_measured', ds.cells.fetch_all('I') & "
    "ds.cells.fetch_all('RNA_I'))`."
)

_ENTRIES = (
    _Entry(
        "run_normalization",
        "RNA",
        lambda s, m: s.run_normalization(m.refs["all"], m.refs["rna_features"]),
        lambda s, m: s.run_normalization(
            m.refs["measured_RNA"], m.refs["rna_features"]
        ),
        _SELECTION,
    ),
    _Entry(
        "integrate_assays",
        "RNA",
        lambda s, m: s.integrate_assays(
            [m.refs["legacy_neighbors"], m.refs["adt_neighbors_all"]], method="wnn"
        ),
        lambda s, m: s.integrate_assays(
            [m.refs["rna_neighbors_measured"], m.refs["adt_neighbors_measured"]],
            method="wnn",
        ),
        _GRAPH,
    ),
    _Entry(
        "select_hvgs",
        "RNA",
        lambda s, m: s.select_hvgs(m.refs["all"], show_plot=False, **_HVG),
        lambda s, m: s.select_hvgs(m.refs["measured_RNA"], show_plot=False, **_HVG),
        _SELECTION,
    ),
    _Entry(
        "select_detected_features",
        "RNA",
        lambda s, m: s.select_detected_features(m.refs["all"], min_cells=1),
        lambda s, m: s.select_detected_features(m.refs["measured_RNA"], min_cells=1),
        _SELECTION,
    ),
    _Entry(
        "run_waggr",
        "RNA",
        lambda s, m: s.run_waggr(
            _NETWORK, m.refs["all"], features=m.refs["rna_features"], tmin=3
        ),
        lambda s, m: s.run_waggr(
            _NETWORK, m.refs["measured_RNA"], features=m.refs["rna_features"], tmin=3
        ),
        _SELECTION,
    ),
    _Entry(
        "run_aucell",
        "RNA",
        lambda s, m: s.run_aucell(
            _NETWORK, m.refs["all"], features=m.refs["rna_features"], tmin=3
        ),
        lambda s, m: s.run_aucell(
            _NETWORK, m.refs["measured_RNA"], features=m.refs["rna_features"], tmin=3
        ),
        _SELECTION,
    ),
    _Entry(
        "run_marker_search",
        "RNA",
        lambda s, m: s.run_marker_search(
            m.refs["labels_all"], features=m.refs["rna_features"]
        ),
        lambda s, m: s.run_marker_search(
            _narrowed_labels(s, m.refs["labels_all"]), features=m.refs["rna_features"]
        ),
        _LABELS,
    ),
    # Label consumers accept labels of measured cells, and also every label
    # with the remedy's measured cell_selection.
    _Entry(
        "make_bulk",
        "RNA",
        lambda s, m: s.make_bulk(m.refs["labels_all"], from_assay="RNA"),
        lambda s, m: (
            s.make_bulk(m.refs["labels_measured"], from_assay="RNA"),
            s.make_bulk(
                m.refs["labels_all"],
                from_assay="RNA",
                cell_selection=m.refs["measured_RNA"],
            ),
        ),
        _SELECTION_KEYWORD,
    ),
    _Entry(
        "run_statistical_testing",
        "RNA",
        lambda s, m: s.run_statistical_testing(["RNA0"], m.refs["labels_all"]),
        lambda s, m: (
            s.run_statistical_testing(["RNA0"], m.refs["labels_measured"]),
            s.run_statistical_testing(
                ["RNA0"], m.refs["labels_all"], cell_selection=m.refs["measured_RNA"]
            ),
            # Only the cells that the design keeps are tested and checked.
            s.run_statistical_testing(
                ["RNA0"], m.refs["labels_all"], subset_by="RNA_I"
            ),
        ),
        _SELECTION_KEYWORD,
    ),
    _Entry(
        "select_prevalent_peaks",
        "ATAC",
        lambda s, m: s.select_prevalent_peaks(
            m.refs["all"], from_assay="ATAC", top_n=5
        ),
        lambda s, m: s.select_prevalent_peaks(
            m.refs["measured_ATAC"], from_assay="ATAC", top_n=5
        ),
        _SELECTION,
    ),
    _Entry(
        "run_cell_cycle_scoring",
        "RNA",
        lambda s, m: s.run_cell_cycle_scoring(m.refs["all"], **_CELL_CYCLE),
        lambda s, m: s.run_cell_cycle_scoring(m.refs["measured_RNA"], **_CELL_CYCLE),
        _SELECTION,
    ),
    _Entry(
        "run_feature_percentage",
        "RNA",
        lambda s, m: s.run_feature_percentage(m.refs["all"], m.refs["rna_some"]),
        lambda s, m: s.run_feature_percentage(
            m.refs["measured_RNA"], m.refs["rna_some"]
        ),
        _SELECTION,
    ),
    _Entry(
        "run_hto_demultiplexing",
        "HTO",
        lambda s, m: s.run_hto_demultiplexing(m.refs["all"]),
        lambda s, m: s.run_hto_demultiplexing(m.refs["measured_HTO"]),
        _SELECTION,
    ),
    _Entry(
        "run_doublet_detection",
        "RNA",
        lambda s, m: s.run_doublet_detection(
            m.refs["legacy_clusters"], m.refs["legacy_graph"]
        ),
        lambda s, m: s.run_doublet_detection(
            m.refs["clusters_measured"], m.refs["graph_measured"]
        ),
        _GRAPH,
    ),
    _Entry(
        "run_pseudotime_marker_search",
        "RNA",
        lambda s, m: s.run_pseudotime_marker_search(
            m.refs["pseudotime_all"], features=m.refs["rna_features"]
        ),
        lambda s, m: s.run_pseudotime_marker_search(
            m.refs["pseudotime_measured"], features=m.refs["rna_features"]
        ),
        _GRAPH,
        # A pseudotime reads its valid cells, which can be fewer.
        every_cell=False,
    ),
    _Entry(
        "run_pseudotime_aggregation",
        "RNA",
        lambda s, m: s.run_pseudotime_aggregation(
            m.refs["pseudotime_all"], features=m.refs["rna_features"], **_AGGREGATION
        ),
        lambda s, m: s.run_pseudotime_aggregation(
            m.refs["pseudotime_measured"],
            features=m.refs["rna_features"],
            **_AGGREGATION,
        ),
        _GRAPH,
        every_cell=False,
    ),
    _Entry(
        "get_imputed",
        "RNA",
        lambda s, m: s.get_imputed("RNA0", m.refs["legacy_diffusion"]),
        lambda s, m: s.get_imputed("RNA0", m.refs["diffusion_measured"]),
        _GRAPH,
    ),
    _Entry(
        "run_mapping",
        "RNA",
        lambda s, m: s.run_mapping(m.reference, m.refs["all"]),
        lambda s, m: s.run_mapping(m.reference, m.refs["measured_RNA"]),
        _SELECTION,
    ),
    _Entry(
        "to_anndata",
        "RNA",
        lambda s, m: s.to_anndata(from_assay="RNA", matrix="normed"),
        lambda s, m: s.to_anndata(from_assay="RNA", cell_key="RNA_I", matrix="normed"),
        _CELL_KEY,
    ),
    _Entry(
        "pipeline.run",
        "RNA",
        lambda s, m: _recipe(s, "I"),
        lambda s, m: _recipe(s, "RNA_I"),
        _CELL_KEY,
    ),
)
_IDS = [entry.operation for entry in _ENTRIES]


def _assert_refusal(error: UnmeasuredCellsError, entry: _Entry) -> None:
    assert isinstance(error, ValueError)
    assert error.operation == entry.operation
    assert error.assay == entry.assay
    assert error.column == f"{entry.assay}_I"
    if entry.every_cell:
        assert (error.unmeasured, error.selected) == (_N_ADT, _N_CELLS)
    else:
        assert 0 < error.unmeasured <= _N_ADT < error.selected <= _N_CELLS
    message = str(error)
    assert message.startswith(
        f"`{entry.operation}` reads assay {entry.assay!r} over {error.unmeasured} "
        f"of {error.selected} selected cells that it did not measure: cell column "
        f"'{entry.assay}_I' is False for them. "
    )
    assert entry.remedy in message


@pytest.mark.parametrize("entry", _ENTRIES, ids=_IDS)
def test_an_unmeasured_cell_is_refused_before_anything_is_written(
    measured, entry
) -> None:
    # A read-only store refuses as a writable one does, not for being read-only.
    stores = [measured.open(), measured.open(zarr_mode="r")]
    before = _tree(measured.path)

    for store in stores:
        with pytest.raises(UnmeasuredCellsError) as raised:
            entry.refuse(store, measured)
        _assert_refusal(raised.value, entry)

    assert _tree(measured.path) == before


# Cell-level tests warn that they are descriptive.
@pytest.mark.filterwarnings("ignore:Cell-level statistical testing:UserWarning")
@pytest.mark.parametrize("entry", _ENTRIES, ids=_IDS)
def test_measured_cells_are_accepted(measured, entry, monkeypatch) -> None:
    # test_hto covers the demultiplexing; here only the check matters.
    monkeypatch.setattr(
        "scarf.datastore._operations.quality_control.hto_demux",
        lambda frame, **_kwargs: frame.idxmax(axis=1),
    )
    store = measured.open()

    entry.accept(store, measured)


def test_select_measured_cells_snapshots_the_measured_cells(measured) -> None:
    store = measured.open()
    every = measured.refs["all"]
    selection = measured.refs["measured_RNA"]
    membership = np.asarray(store.cells.fetch_all("RNA_I"), dtype=bool)

    np.testing.assert_array_equal(
        np.asarray(store.load_artifact(selection)["values"][:]), membership
    )
    assert (selection.scope, selection.assay, selection.kind) == (
        "datastore",
        None,
        "cell_selection",
    )
    status = store.inspect_artifact(selection)
    assert status.operation == "select_measured_cells"
    assert status.parameters == {"assay": "RNA"}
    assert status.input_ref("prior_cell_selection") == every
    assert status.inputs["membership_fingerprint"] == fingerprint_array(membership)
    assert set(status.inputs) == {
        "prior_cell_selection",
        "membership_fingerprint",
        "ordered_row_ids_fingerprint",
        "values_fingerprint",
    }
    # The same request reuses the artifact, also from a read-only store, and
    # an omitted selection snapshots I, which here holds every cell.
    assert store.select_measured_cells("RNA", cell_selection=every) == selection
    assert store.select_measured_cells("RNA") == selection
    read_only = measured.open(zarr_mode="r")
    assert read_only.select_measured_cells("RNA", cell_selection=every) == selection
    # ATAC and HTO measured the same cells; each records its own membership.
    for assay in ("ATAC", "HTO"):
        other = measured.refs[f"measured_{assay}"]
        assert other != selection
        assert store.inspect_artifact(other).parameters == {"assay": assay}
        np.testing.assert_array_equal(
            np.asarray(store.load_artifact(other)["values"][:]), membership
        )
    fresh = store.select_measured_cells(
        "RNA", cell_selection=every, invalidate_cache=True
    )
    assert fresh != selection
    np.testing.assert_array_equal(
        np.asarray(store.load_artifact(fresh)["values"][:]), membership
    )


def test_select_measured_cells_returns_a_measured_selection_unchanged(
    measured,
) -> None:
    store = measured.open()
    every = measured.refs["all"]
    selection = measured.refs["measured_RNA"]
    before = _tree(measured.path)

    # ADT measured every cell, and RNA every cell of its own selection.
    assert store.select_measured_cells("ADT", cell_selection=every) == every
    assert store.select_measured_cells("RNA", cell_selection=selection) == selection
    assert _tree(measured.path) == before


def test_select_measured_cells_raises_when_no_cell_remains(measured) -> None:
    store = measured.open()

    with pytest.raises(
        ValueError,
        match=(
            r"^Assay 'RNA' measured none of the 16 selected cells: cell column "
            r"'RNA_I' is False for all of them\.$"
        ),
    ):
        store.select_measured_cells("RNA", cell_selection=measured.refs["adt_cells"])
    with pytest.raises(ValueError, match="Assay 'GENES' not found"):
        store.select_measured_cells("GENES")
    with pytest.raises(TypeError, match="assay must be the name of an assay"):
        store.select_measured_cells("")
    with pytest.raises(TypeError, match="cell_selection must be an ArtifactRef"):
        store.select_measured_cells("RNA", cell_selection="RNA_I")  # type: ignore[arg-type]


def _rna_source(path: Path, seed: int, *, membership: bool = False) -> DataStore:
    counts = np.random.default_rng(seed).poisson(4.0, size=(24, 12)) + 1
    write_count_store(str(path), {"RNA": counts}, "uint32")
    if membership:
        # Every cell is a member, as a merge of RNA sources records it.
        cells = zarr.open_group(str(path), mode="r+")["cellData"]
        write_membership_column(cells, "RNA", np.ones(len(counts), dtype=bool))
    return DataStore(str(path), default_assay="RNA", min_features_per_cell=-1)


def test_fully_measured_stores_keep_their_identities(tmp_path) -> None:
    """Membership that marks every cell measured changes no result.

    A store whose ``RNA_I`` is True for every cell, and a store without
    membership columns, pass the check, and ``select_measured_cells`` returns
    the selection that it narrows, so the results of a workflow that narrows
    its cells first are the results of one that does not, with the same
    identities. Results record no membership input, as before.
    """
    member = _rna_source(tmp_path / "member.zarr", 1, membership=True)
    plain = _rna_source(tmp_path / "plain.zarr", 2)
    assert np.asarray(member.cells.fetch_all("RNA_I"), dtype=bool).all()
    assert "RNA_I" not in plain.cells.columns

    for store in (member, plain):
        cells = store.snapshot_cell_selection("I")
        features = store.select_all_features(from_assay="RNA")
        normalized = store.run_normalization(cells, features)
        narrowed = store.select_measured_cells("RNA", cell_selection=cells)
        assert narrowed == cells
        assert store.select_measured_cells("RNA") == cells
        assert store.run_normalization(narrowed, features) == normalized
        status = store.inspect_artifact(normalized)
        assert set(status.inputs) == {
            "cell_selection",
            "dataset_fingerprint",
            "feature_selection",
        }
        values = fetch_normalized_feature_matrix(
            store,
            [resolve_feature(store, "RNA0")],
            np.arange(store.cells.N),
        )
        assert np.isfinite(values).all()
        assert not list_artifacts(
            store.zw, scope="datastore", operation="select_measured_cells"
        )


@pytest.mark.slow
def test_the_pipeline_runs_over_the_cells_that_its_remedy_names(measured) -> None:
    store = measured.open()
    # The column that the remedy writes holds the cells of I that RNA measured.
    store.cells.insert(
        "RNA_measured", store.cells.fetch_all("I") & store.cells.fetch_all("RNA_I")
    )
    run = store.pipeline.run(label="measured", cell_key="RNA_measured", **_PIPELINE)

    assert run.status == "completed"
    np.testing.assert_array_equal(
        np.asarray(store.load_artifact(run["input_cell_selection"])["values"][:]),
        np.asarray(store.cells.fetch_all("RNA_I"), dtype=bool)
        & np.asarray(store.cells.fetch_all("I"), dtype=bool),
    )


@pytest.mark.slow
def test_results_of_earlier_releases_stay_readable_and_their_runs_reopen(
    measured,
) -> None:
    """Results computed over unmeasured cells are neither detected nor repaired.

    They stay listable, loadable, and traceable, a completed run of such a
    configuration reopens, and running its configuration again raises before
    any stage.
    """
    store = measured.open()
    with pytest.MonkeyPatch.context() as patch:
        # An earlier release ran the pipeline over every cell.
        patch.setattr(
            BaseDataStore, "_require_measured_cells", lambda *_args, **_kwargs: None
        )
        patch.setattr(
            "scarf.datastore._pipeline_recipe.require_measured_cells",
            lambda *_args, **_kwargs: None,
        )
        store.pipeline.run(label="earlier", **_PIPELINE)
    legacy = measured.refs["legacy_normalized"]

    assert legacy in store.list_artifacts(from_assay="RNA", kind="normalized")
    assert store.inspect_artifact(legacy).complete
    assert store.load_artifact(legacy)["data"].shape[0] == _N_CELLS
    assert measured.refs["all"] in set(store.lineage(legacy).graph)
    reopened = store.pipeline.open(label="earlier")
    assert reopened.status == "completed"
    assert reopened.report()["run"]["config"]["cellKey"] == "I"

    runs = [run.run_id for run in store.pipeline.list_runs()]
    before = _tree(measured.path)
    with pytest.raises(UnmeasuredCellsError, match="pipeline.run"):
        store.pipeline.run(label="again", **_PIPELINE)
    assert [run.run_id for run in store.pipeline.list_runs()] == runs
    assert _tree(measured.path) == before


def test_display_reads_show_unmeasured_cells_as_missing(measured) -> None:
    store = measured.open()
    membership = np.asarray(store.cells.fetch_all("RNA_I"), dtype=bool)
    every = np.arange(_N_CELLS)
    rna = resolve_feature(store, "RNA0")
    adt = resolve_feature(store, FeatureRef("ADT0", assay="ADT"))

    matrix = fetch_normalized_feature_matrix(store, [rna, adt], every)

    assert np.isnan(matrix[~membership, 0]).all()
    assert np.isfinite(matrix[membership, 0]).all()
    # ADT measured every cell, so each of its values is read.
    assert np.isfinite(matrix[:, 1]).all()
    # Measured cells keep the values that a read of them alone gives.
    np.testing.assert_array_equal(
        matrix[membership, 0],
        fetch_normalized_feature_matrix(store, [rna], np.flatnonzero(membership))[:, 0],
    )
    raw = fetch_normalized_feature_matrix(
        store, [rna], every, NormalizationSpec(source="raw", transform="log1p")
    )
    assert np.isnan(raw[~membership, 0]).all()
    assert (raw[membership, 0] > 0).all()
    blocks = [
        (start, values)
        for _slots, start, values in iter_normalized_feature_blocks(store, [rna], every)
    ]
    streamed = np.concatenate([values for _start, values in blocks])
    np.testing.assert_array_equal(streamed, matrix[:, :1])

    values = store.get_cell_vals("RNA", "I", "RNA0")
    np.testing.assert_array_equal(values, matrix[:, 0])
    assert np.isnan(store.get_cell_vals("RNA", "adt_cells", "RNA0")).all()
    clipped = store.get_cell_vals("RNA", "I", "RNA0", clip_fraction=0.1)
    assert np.isnan(clipped[~membership]).all()
    # A cell metadata column is read as it is stored.
    np.testing.assert_array_equal(
        store.get_cell_vals("RNA", "I", "RNA_nCounts")[~membership], 0
    )


def test_plots_draw_unmeasured_cells_as_missing(measured) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from scarf import plotting

    store = measured.open()
    membership = np.asarray(store.cells.fetch_all("RNA_I"), dtype=bool)
    expression = fetch_normalized_feature_matrix(
        store, [resolve_feature(store, "RNA0")], np.flatnonzero(membership)
    )[:, 0]

    layout = plotting.embedding(store, layout_key="xy", color_by="RNA0", show=False)
    try:
        (axis,) = layout.axes.values()
        points = axis.collections[0]
        rows = np.asarray(points.get_offsets())[:, 0].astype(int)
        grey = np.all(
            np.isclose(
                points.get_facecolors(),
                matplotlib.colors.to_rgba(plotting.ColorScale().missing_color),
            ),
            axis=1,
        )
        # Unmeasured cells take the color of missing values, and the color
        # limits come from measured values only.
        np.testing.assert_array_equal(grey, ~membership[rows])
        assert layout.provenance.extras["color_limits"] == {
            "RNA0": (float(expression.min()), float(expression.max()))
        }
        assert layout.provenance.extras["unmeasured_cells"] == {"RNA": _N_ADT}
    finally:
        layout.close()

    violins = plotting.distribution(
        store, "RNA0", grouping=CellField("program"), show=False
    )
    try:
        table = violins.tables["RNA0"]
        # The unmeasured cells have no value to draw or tabulate.
        assert violins.provenance.extras["unmeasured_cells"] == {"RNA": _N_ADT}
        assert len(table) == _N_FULL
        np.testing.assert_allclose(
            np.sort(table["value"].to_numpy()), np.sort(expression)
        )
    finally:
        violins.close()


def test_require_measured_cells_reads_masks_rows_and_columns(measured) -> None:
    store = measured.open()
    membership = np.asarray(store.cells.fetch_all("RNA_I"), dtype=bool)
    stored_mask = zarr.open_group(str(measured.path), mode="r")["cellData/I"]
    cases: list[Any] = [
        "I",
        np.ones(_N_CELLS, dtype=bool),
        stored_mask,
        np.arange(_N_CELLS)[::-1],
        np.concatenate([np.arange(_N_CELLS), [70, 71]]),
        list(range(_N_CELLS)),
    ]
    for cells in cases:
        with pytest.raises(UnmeasuredCellsError) as raised:
            require_measured_cells(store.cells, "RNA", cells, operation="probe")
        assert (raised.value.unmeasured, raised.value.selected) == (
            _N_ADT,
            _N_CELLS,
        )
        assert "Pass a selection of measured cells from " in str(raised.value)
    # Measured cells, an assay that measured every cell, and an empty read pass.
    for cells in ("RNA_I", membership, np.flatnonzero(membership)):
        require_measured_cells(store.cells, "RNA", cells, operation="probe")
    require_measured_cells(store.cells, "ADT", "I", operation="probe")
    require_measured_cells(
        store.cells, "RNA", np.array([], dtype=np.int64), operation="probe"
    )
    with pytest.raises(ValueError, match="one entry per cell"):
        require_measured_cells(
            store.cells, "RNA", np.ones(3, dtype=bool), operation="probe"
        )
    with pytest.raises(ValueError, match="one-dimensional"):
        require_measured_cells(
            store.cells, "RNA", np.zeros((2, 2), dtype=np.int64), operation="probe"
        )
    # A membership column of two-row chunks is read one block at a time.
    group = zarr.open_group(store=zarr.storage.MemoryStore(), mode="w")
    for name, values in (
        ("ids", np.array(["c0", "c1", "c2", "c3"])),
        ("names", np.array(["c0", "c1", "c2", "c3"])),
        ("I", np.ones(4, dtype=bool)),
        ("RNA_I", np.array([True, True, False, True])),
    ):
        group.create_array(name, data=values, chunks=(2,))
    group["RNA_I"].attrs.update(membership_attributes("RNA"))
    with pytest.raises(UnmeasuredCellsError) as raised:
        require_measured_cells(
            MetaData(group),
            "RNA",
            np.array([True, True, True, False]),
            operation="probe",
        )
    assert (raised.value.unmeasured, raised.value.selected) == (1, 3)


def test_unmeasured_cells_error_keeps_its_fields_through_pickling() -> None:
    error = UnmeasuredCellsError("run_pca", "ADT", "ADT_I", 2, 9, "graph")

    copy = pickle.loads(pickle.dumps(error))

    assert isinstance(copy, UnmeasuredCellsError)
    assert isinstance(copy, ValueError)
    assert (copy.operation, copy.assay, copy.column) == ("run_pca", "ADT", "ADT_I")
    assert (copy.unmeasured, copy.selected, copy.remedy) == (2, 9, "graph")
    assert copy.cell_key is None
    assert str(copy) == str(error)
    keyed = UnmeasuredCellsError(
        "pipeline.run", "RNA", "RNA_I", 1, 4, "cell_key", "kept"
    )
    assert "cells of 'kept' that it measured" in str(keyed)
    assert pickle.loads(pickle.dumps(keyed)).cell_key == "kept"
    assert str(pickle.loads(pickle.dumps(keyed))) == str(keyed)


# ---------------------------------------------------------------------------
# Display values, summary plots, QC, bulk profiles, and exports over a merge
# whose RNA assay measured the cells of one source.

_N_RNA = 40
_N_OTHER = 20


def _merge_partial_rna(base: Path) -> Path:
    """Merge 40 cells with RNA and ADT and 20 cells with ADT only.

    RNA0 is detected in about half of the measured cells. The merge keeps
    each source's cells together, so the first 40 cells are the measured
    ones. ``grp`` alternates two groups over every cell, ``source`` names
    each cell's source, ``adt_only`` marks the unmeasured cells, and ``xy``
    is a live layout.
    """
    rng = np.random.default_rng(3)
    rna = rng.poisson(3.0, size=(_N_RNA, 12)) + 1
    rna[:, 0] = rng.binomial(1, 0.5, size=_N_RNA) * 5
    write_count_store(
        str(base / "full.zarr"),
        {"RNA": rna, "ADT": rng.integers(1, 20, size=(_N_RNA, 3))},
        "uint32",
    )
    write_count_store(
        str(base / "adt.zarr"),
        {"ADT": rng.integers(1, 20, size=(_N_OTHER, 3))},
        "uint32",
    )
    sources = [
        DataStore(
            str(base / f"{name}.zarr"),
            default_assay=assay,
            min_features_per_cell=0,
            nthreads=1,
        )
        for name, assay in (("full", "RNA"), ("adt", "ADT"))
    ]
    path = base / "merged.zarr"
    DataStoreMerge(sources, str(path), ["full", "adt"], nthreads=1).dump()
    store = _open(path)
    n_cells = store.cells.N
    store.cells.insert("grp", np.where(np.arange(n_cells) % 2 == 0, "g0", "g1"))
    ids = np.asarray(store.cells.fetch_all("ids")).astype(str)
    store.cells.insert("source", np.array([value.split("__")[0] for value in ids]))
    store.cells.insert(
        "adt_only", ~np.asarray(store.cells.fetch_all("RNA_I"), dtype=bool)
    )
    store.cells.insert("xy1", np.arange(n_cells, dtype=np.float64))
    store.cells.insert("xy2", (np.arange(n_cells) % 7).astype(np.float64))
    return path


def _open(path: Path, zarr_mode: str = "r+") -> DataStore:
    return DataStore(
        str(path),
        default_assay="RNA",
        min_features_per_cell=-1,
        nthreads=1,
        zarr_mode=zarr_mode,
    )


@pytest.fixture(scope="module")
def partial_rna(tmp_path_factory) -> Path:
    return _merge_partial_rna(tmp_path_factory.mktemp("partial_rna"))


def _plotting() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    from scarf import plotting

    return plotting


def test_a_grouped_assay_fits_its_source_normalization_on_measured_cells(
    measured, tmp_path
) -> None:
    """Group means of a partly measured assay ignore the cells it did not measure.

    Before, ATAC document frequencies counted the zero rows of the unmeasured
    cells, and the wrong means sat on rows that the membership marks measured.
    The merged store's measured cells now hold the means that the source store,
    which holds only those cells, gives; the others hold zeros.
    """
    import shutil

    stores = {}
    for name in ("merged.zarr", "full.zarr"):
        path = tmp_path / name
        shutil.copytree(measured.path.parent / name, path)
        store = DataStore(
            str(path), default_assay="RNA", min_features_per_cell=-1, nthreads=1
        )
        n_peaks = store.ATAC.feats.N
        store.ATAC.feats.insert("module", np.arange(n_peaks) % 2, overwrite=True)
        store.add_grouped_assay("module", assay_label="modules", from_assay="ATAC")
        stores[name] = store
    merged, alone = stores["merged.zarr"], stores["full.zarr"]
    means = np.asarray(merged.modules.rawData, dtype=np.float64)
    expected = np.asarray(alone.modules.rawData, dtype=np.float64)
    member = np.asarray(merged.cells.fetch_all("ATAC_I"), dtype=bool)
    # The merge names each cell <source>__<id> and may reorder the cells.
    row_of = {str(cell): row for row, cell in enumerate(alone.cells.fetch_all("ids"))}
    rows = [
        row_of[str(cell).split("__", 1)[1]]
        for cell in np.asarray(merged.cells.fetch_all("ids"))[member]
    ]

    assert member.sum() == _N_FULL and means.shape == (_N_CELLS, 2)
    np.testing.assert_allclose(means[member], expected[rows], rtol=1e-12)
    np.testing.assert_array_equal(means[~member], 0.0)


def test_display_values_of_measured_cells_ignore_unmeasured_cells(measured) -> None:
    """A display read normalizes only the cells that the assay measured.

    TF-IDF fits document frequency over the cells that it normalizes, so
    unmeasured cells in the read raised the values of measured cells 1.4
    times. A measured cell now reads the value that a read of the measured
    cells alone gives, for ATAC as for RNA, in every block order.
    """
    store = measured.open()
    membership = np.asarray(store.cells.fetch_all("ATAC_I"), dtype=bool)
    measured_rows_ = np.flatnonzero(membership)
    every = np.arange(_N_CELLS)
    peak = resolve_feature(store, FeatureRef("ATAC30", assay="ATAC"))
    alone = fetch_normalized_feature_matrix(store, [peak], measured_rows_)[:, 0]
    assert np.isfinite(alone).all() and (alone > 0).any()

    whole = fetch_normalized_feature_matrix(store, [peak], every)[:, 0]

    assert np.isnan(whole[~membership]).all()
    np.testing.assert_array_equal(whole[membership], alone)
    # Unmeasured cells before the measured ones, and between them.
    reverse = fetch_normalized_feature_matrix(store, [peak], every[::-1])[:, 0]
    np.testing.assert_array_equal(reverse, whole[::-1])
    mixed = np.array([70, 0, 71, 5, 64, 63, 79])
    read = membership[mixed]
    expected = np.full(len(mixed), np.nan)
    expected[read] = fetch_normalized_feature_matrix(store, [peak], mixed[read])[:, 0]
    np.testing.assert_array_equal(
        fetch_normalized_feature_matrix(store, [peak], mixed)[:, 0], expected
    )
    # A read of unmeasured cells only has no value to show.
    assert np.isnan(
        fetch_normalized_feature_matrix(store, [peak], np.flatnonzero(~membership))
    ).all()
    values = store.get_cell_vals("ATAC", "I", "ATAC30")
    assert np.isnan(values[~membership]).all()
    np.testing.assert_allclose(values[membership], alone, rtol=1e-12)


@pytest.mark.slow
def test_summary_statistics_cover_the_cells_with_a_value(partial_rna) -> None:
    """Every summary statistic has the measured cells of its group as cells.

    The review's merge: each group holds 30 cells, RNA measured 20 of them,
    and RNA0 is detected in 10 and 11 of those. The fraction was the
    detected cells over all 30 cells, 0.333 and 0.367, and ``n_cells``
    counted 30. A group without a measured cell has no fraction, and an
    aggregate over samples skips such a sample as it skips its mean.
    """
    plotting = _plotting()
    store = _open(partial_rna)
    membership = np.asarray(store.cells.fetch_all("RNA_I"), dtype=bool)
    groups = np.asarray(store.cells.fetch_all("grp")).astype(str)
    detected = np.asarray(store.RNA.rawData[:, [0]].compute()).ravel() > 0
    expected = {
        group: (
            int((membership & (groups == group)).sum()),
            float(detected[membership & (groups == group)].mean()),
        )
        for group in ("g0", "g1")
    }
    assert expected == {"g0": (20, 0.5), "g1": (20, 0.55)}

    result = plotting.dotplot(store, features=["RNA0"], group_by="grp", show=False)
    try:
        table = result.tables["aggregate"].set_index("group")
        for group, (n_cells, fraction) in expected.items():
            assert table.loc[group, "n_cells"] == n_cells
            assert table.loc[group, "fraction"] == pytest.approx(fraction)
        assert result.provenance.extras["unmeasured_cells"] == {"RNA": _N_OTHER}
    finally:
        result.close()

    by_source = plotting.matrixplot(
        store, features=["RNA0"], group_by="source", value="fraction", show=False
    )
    try:
        table = by_source.tables["aggregate"].set_index("group")
        assert table.loc["adt", "n_cells"] == 0
        assert np.isnan(
            table.loc["adt", ["mean", "fraction", "variance"]].to_numpy(dtype=float)
        ).all()
        assert table.loc["full", "n_cells"] == _N_RNA
        assert table.loc["full", "fraction"] == pytest.approx(
            detected[membership].mean()
        )
        assert np.isnan(by_source.tables["matrix"].loc["RNA0", "adt"])
        assert by_source.provenance.extras["unmeasured_cells"] == {"RNA": _N_OTHER}
    finally:
        by_source.close()

    sampled = plotting.dotplot(
        store, features=["RNA0"], group_by="grp", sample_by="source", show=False
    )
    try:
        aggregate = sampled.tables["aggregate"].set_index("group")
        per_sample = sampled.tables["per_sample"].set_index(["sample", "group"])
        for group, (n_cells, fraction) in expected.items():
            assert aggregate.loc[group, "fraction"] == pytest.approx(fraction)
            assert aggregate.loc[group, "n_cells"] == n_cells
            assert aggregate.loc[group, "n_samples"] == 1
            assert per_sample.loc[("adt", group), "n_cells"] == 0
            assert np.isnan(per_sample.loc[("adt", group), "fraction"])
    finally:
        sampled.close()


def test_plots_record_the_cells_that_an_assay_did_not_measure(partial_rna) -> None:
    plotting = _plotting()
    store = _open(partial_rna)

    # Over unmeasured cells only, no cell has a value to set color limits.
    unmeasured = plotting.embedding(
        store, layout_key="xy", color_by="RNA0", cell_key="adt_only", show=False
    )
    try:
        assert unmeasured.provenance.extras["color_limits"] == {"RNA0": None}
        assert unmeasured.provenance.extras["unmeasured_cells"] == {"RNA": _N_OTHER}
    finally:
        unmeasured.close()
    # ADT measured every cell, so nothing is recorded for it.
    adt = plotting.embedding(
        store, layout_key="xy", color_by=FeatureRef("ADT0", assay="ADT"), show=False
    )
    try:
        assert "unmeasured_cells" not in adt.provenance.extras
    finally:
        adt.close()
    with pytest.raises(ValueError, match="No finite values remain") as raised:
        plotting.distribution(store, "RNA0", subset_by="adt_only", show=False)
    message = str(raised.value)
    assert "assay 'RNA' did not measure" in message
    assert "subset_by='RNA_I'" in message
    # The remedy's subset leaves every unmeasured cell out, so none is recorded.
    for result in (
        plotting.distribution(store, "RNA0", subset_by="RNA_I", show=False),
        plotting.embedding(
            store, layout_key="xy", color_by="RNA0", subset_by="RNA_I", show=False
        ),
    ):
        try:
            assert "unmeasured_cells" not in result.provenance.extras
        finally:
            result.close()
    # Unmeasured cells without a group join no summary, so none is recorded.
    store.cells.insert(
        "rna_grp",
        np.where(np.arange(_N_RNA) % 2 == 0, "g0", "g1"),
        key="RNA_I",
        overwrite=True,
    )
    grouped = plotting.dotplot(store, features=["RNA0"], group_by="rna_grp", show=False)
    try:
        assert grouped.provenance.extras["dropped_group_cells"] == _N_OTHER
        assert "unmeasured_cells" not in grouped.provenance.extras
    finally:
        grouped.close()


def _merge_low_quality(base: Path) -> Path:
    """Merge 40 RNA cells, five of them low quality, with 50 ADT-only cells."""
    rng = np.random.default_rng(1)
    rna = rng.poisson(3.0, size=(40, 30)) + 1
    rna[:5, 3:] = 0
    write_count_store(
        str(base / "full.zarr"),
        {"RNA": rna, "ADT": rng.integers(1, 9, size=(40, 3))},
        "uint32",
    )
    write_count_store(
        str(base / "adt.zarr"),
        {"ADT": rng.integers(1, 9, size=(50, 3))},
        "uint32",
    )
    sources = [
        DataStore(
            str(base / f"{name}.zarr"),
            default_assay=assay,
            min_features_per_cell=0,
            nthreads=1,
        )
        for name, assay in (("full", "RNA"), ("adt", "ADT"))
    ]
    path = base / "merged.zarr"
    DataStoreMerge(sources, str(path), ["full", "adt"], nthreads=1).dump()
    _open(path)
    return path


@pytest.mark.slow
def test_quality_control_refuses_metrics_of_unmeasured_cells(tmp_path) -> None:
    """QC thresholds over an assay's metrics read that assay's cells.

    With 50 of 90 cells unmeasured, the zeros of their RNA metrics set a
    zero MAD, so ``auto_filter_cells`` kept every cell, the five low-quality
    measured cells included.
    """
    from scarf.metadata.selection import NamedCellArtifact

    path = _merge_low_quality(tmp_path)
    store = _open(path)
    membership = np.asarray(store.cells.fetch_all("RNA_I"), dtype=bool)
    with pytest.MonkeyPatch.context() as patch:
        # An earlier release computed this metric over every cell.
        patch.setattr(
            BaseDataStore, "_require_measured_cells", lambda *_args, **_kwargs: None
        )
        legacy_metric = store.run_feature_percentage(
            store.snapshot_cell_selection("I"),
            store.set_feature_selection(from_assay="RNA", feature_indexes=[0, 1]),
        )
    before = _tree(path)

    for call in (
        lambda s: s.auto_filter_cells(attrs=["RNA_nCounts"]),
        lambda s: s.auto_filter_cells(attrs=["RNA_nCounts"], method="gaussian"),
        lambda s: s.auto_filter_cells(
            attrs=[], artifact_metrics=[NamedCellArtifact("pct", legacy_metric)]
        ),
        lambda s: s.filter_cells(["RNA_nFeatures"], [5], [None]),
    ):
        for opened in (store, _open(path, zarr_mode="r")):
            with pytest.raises(UnmeasuredCellsError) as raised:
                call(opened)
            assert (raised.value.assay, raised.value.column) == ("RNA", "RNA_I")
            assert (raised.value.unmeasured, raised.value.selected) == (50, 90)
            assert "Pass `cell_selection=select_measured_cells('RNA'" in str(
                raised.value
            )
    assert _tree(path) == before

    # A metric of an assay that measured every cell reads every cell, and a
    # user column named like a metric is no metric.
    assert store.load_artifact(store.filter_cells(["ADT_nCounts"], [0], [None]))[
        "values"
    ][:].all()
    store.cells.insert("RNA_score", np.ones(store.cells.N))
    assert store.load_artifact(store.filter_cells(["RNA_score"], [0], [None]))[
        "values"
    ][:].all()
    # Over the measured cells, the MAD bounds remove the low-quality cells.
    kept = store.auto_filter_cells(
        attrs=["RNA_nCounts"], cell_selection=store.select_measured_cells("RNA")
    )
    values = np.asarray(store.load_artifact(kept)["values"][:], dtype=bool)
    low_quality = membership & (np.asarray(store.cells.fetch_all("RNA_nFeatures")) < 5)
    assert int(low_quality.sum()) == 5
    assert not (values & low_quality).any()
    assert not (values & ~membership).any()
    assert int(values.sum()) >= 30


def test_membership_reads_return_rows_in_order_and_skip_full_columns(
    measured,
) -> None:
    store = measured.open()
    membership = np.asarray(store.cells.fetch_all("RNA_I"), dtype=bool)
    every = np.arange(_N_CELLS)

    for rows in (
        every,
        every[::-1],
        np.array([0, 0, 3, 3, 70, 70, 79]),
        np.array([79, 3, 70, 3, 0]),
        np.array([63, 64]),
    ):
        np.testing.assert_array_equal(
            measured_rows(store.cells, "RNA", rows), membership[rows]
        )
    # Rows that the assay measured, any rows of an assay that measured every
    # cell, and an empty read have nothing to show as missing.
    assert measured_rows(store.cells, "RNA", np.flatnonzero(membership)) is None
    assert measured_rows(store.cells, "ADT", every[::-1]) is None
    assert measured_rows(store.cells, "RNA", np.array([], dtype=np.int64)) is None
    with pytest.raises(TypeError, match="integers or a boolean mask"):
        measured_rows(store.cells, "RNA", np.array([0.5]))
    for assay in ("RNA", "ADT"):
        for rows in (np.array([_N_CELLS]), np.array([-1]), np.array([3, _N_CELLS])):
            with pytest.raises(IndexError, match="out of bounds"):
                measured_rows(store.cells, assay, rows)
            with pytest.raises(IndexError, match="out of bounds"):
                require_measured_cells(store.cells, assay, rows, operation="probe")


def test_a_few_membership_rows_read_only_their_chunks(measured, monkeypatch) -> None:
    """A lookup of a few cells never streams the whole column.

    Before, finding the membership of one cell near the end of the column
    read every block from the first.
    """
    import scarf.metadata.membership as membership_module

    store = measured.open()
    membership = np.asarray(store.cells.fetch_all("RNA_I"), dtype=bool)

    def no_scan(*args, **kwargs):
        raise AssertionError("a lookup of a few rows streamed the column")

    monkeypatch.setattr(membership_module, "iter_metadata_column_blocks", no_scan)
    unmeasured = int(np.flatnonzero(~membership)[-1])
    for rows in (
        np.array([unmeasured]),
        np.array([_N_CELLS - 1, 3, _N_CELLS - 1]),
        np.array([unmeasured, 0]),
    ):
        expected = membership[rows]
        observed = measured_rows(store.cells, "RNA", rows)
        if expected.all():
            assert observed is None
        else:
            np.testing.assert_array_equal(observed, expected)
    measured_row = int(np.flatnonzero(membership)[0])
    with pytest.raises(UnmeasuredCellsError) as raised:
        require_measured_cells(
            store.cells,
            "RNA",
            np.array([unmeasured, measured_row, unmeasured]),
            operation="probe",
        )
    # A repeated row is one cell.
    assert (raised.value.unmeasured, raised.value.selected) == (1, 2)


def test_make_bulk_refuses_unmeasured_cells_before_it_writes(partial_rna) -> None:
    """A metadata grouping without a selection is checked over live cells.

    ``make_bulk`` snapshotted the live ``I`` column for such a grouping. It
    wrote that snapshot before it refused, and a read-only store raised
    ``PermissionError`` for the snapshot instead of the refusal. It now reads
    the live cells without a snapshot.
    """
    before = _tree(partial_rna)
    for zarr_mode in ("r+", "r"):
        store = _open(partial_rna, zarr_mode=zarr_mode)
        with pytest.raises(UnmeasuredCellsError) as raised:
            store.make_bulk("grp", from_assay="RNA")
        assert (raised.value.unmeasured, raised.value.selected) == (
            _N_OTHER,
            _N_RNA + _N_OTHER,
        )
        assert "Pass `cell_selection=select_measured_cells('RNA'" in str(raised.value)
    assert _tree(partial_rna) == before


def test_make_bulk_checks_only_the_cells_that_it_reads(partial_rna, measured) -> None:
    """Cells that join no bulk column are not read, unless a fit reads them.

    RNA sums and RNA library-size means read the cells of the bulk columns
    only, so unmeasured cells that ``null_vals`` excludes pass. A mean off
    the RNA stream fits the assay's normalization, such as ATAC TF-IDF, over
    every selected cell, so it still reads them.
    """
    store = _open(partial_rna)
    for aggr_type in ("sum", "mean"):
        bulk = store.make_bulk(
            "source", from_assay="RNA", null_vals=["adt"], aggr_type=aggr_type
        )
        assert list(bulk.columns) == ["full"]

    multiome = measured.open()
    sums = multiome.make_bulk(
        "adt_cells", from_assay="ATAC", null_vals=[True], aggr_type="sum"
    )
    assert list(sums.columns) == ["False"]
    with pytest.raises(UnmeasuredCellsError) as raised:
        multiome.make_bulk("adt_cells", from_assay="ATAC", null_vals=[True])
    assert (raised.value.assay, raised.value.unmeasured) == ("ATAC", _N_ADT)


@pytest.mark.parametrize("scope", [None, "panel"])
def test_a_colorbar_keeps_its_own_scale_beside_a_color_without_values(
    partial_rna, scope
) -> None:
    """Each colorbar keeps its color's scale; a color without values keeps none.

    Composed figures pair colorbars with scales in order. Before, the scale of
    a color without values stayed first, so the next colorbar took it.
    """
    plotting = _plotting()
    store = _open(partial_rna)
    result = plotting.embedding(
        store,
        layout_key="xy",
        color_by=["RNA0", FeatureRef("ADT0", assay="ADT")],
        cell_key="adt_only",
        color_scale=None if scope is None else plotting.ColorScale(scope=scope),
        show=False,
    )
    try:
        colorbars = [legend for legend in result.legends if legend.kind == "colorbar"]
        scales = [
            scale for scale in result.scales if isinstance(scale, plotting.ColorScale)
        ]
        assert len(colorbars) == len(scales) == 1
        assert "RNA0" not in colorbars[0].label
    finally:
        result.close()


def test_a_color_without_values_draws_no_colorbar(partial_rna) -> None:
    """Cells that the assay did not measure get no invented color scale.

    Before, the colorbar showed limits of 0 to 1, and a log scale raised
    "Log color scale requires positive values" although no value existed.
    """
    plotting = _plotting()
    store = _open(partial_rna)
    for scale in (
        plotting.ColorScale(),
        plotting.ColorScale(scale="log"),
        plotting.ColorScale(scope="panel"),
        # A diverging center outside the placeholder limits raised.
        plotting.ColorScale(vcenter=0.0),
        plotting.ColorScale(scope="panel", vcenter=0.0),
        plotting.ColorScale(scope="shared", vcenter=0.0),
        plotting.ColorScale(vcenter=1e20),
    ):
        result = plotting.embedding(
            store,
            layout_key="xy",
            color_by="RNA0",
            cell_key="adt_only",
            color_scale=scale,
            show=False,
        )
        try:
            assert [legend.kind for legend in result.legends] == []
            assert len(result.figure.axes) == 1
        finally:
            result.close()
    # Explicit limits still draw their colorbar.
    explicit = plotting.embedding(
        store,
        layout_key="xy",
        color_by="RNA0",
        cell_key="adt_only",
        color_scale=plotting.ColorScale(vmin=0, vmax=2),
        show=False,
    )
    try:
        assert [legend.kind for legend in explicit.legends] == ["colorbar"]
    finally:
        explicit.close()


def test_raw_exports_refuse_cells_whose_membership_they_cannot_declare(
    partial_rna, tmp_path
) -> None:
    """An MTX directory cannot declare which cells an assay measured.

    Its zero rows for unmeasured cells read back as measured cells through
    ``MtxToZarr``, so the export refuses them before it writes anything.
    """
    from scarf.writers import to_mtx

    store = _open(partial_rna)
    target = tmp_path / "rna_mtx"

    with pytest.raises(UnmeasuredCellsError) as raised:
        to_mtx(store.RNA, str(target))

    assert (raised.value.operation, raised.value.assay) == ("to_mtx", "RNA")
    assert (raised.value.unmeasured, raised.value.selected) == (
        _N_OTHER,
        _N_RNA + _N_OTHER,
    )
    assert "SubsetZarr(..., cell_key='RNA_I')" in str(raised.value)
    assert "to_h5ad" in str(raised.value)
    assert not target.exists()
    # ADT measured every cell.
    to_mtx(store.ADT, str(tmp_path / "adt_mtx"))
    assert (tmp_path / "adt_mtx" / "matrix.mtx").exists()


def _write_shared_features(
    path: Path, counts: dict[str, np.ndarray], feature_ids: list[str]
) -> None:
    """Write a store whose assays share their feature IDs, as layers need."""
    from scarf.storage.schema import create_cell_data, create_zarr_count_assay
    from scarf.storage.stores import load_zarr
    from scarf.writers.counts_t import finalize_writer_counts_t
    from tests.storage_helpers import finalize_test_counts

    root = load_zarr(zarr_loc=str(path), mode="w")
    (n_cells,) = {len(values) for values in counts.values()}
    cell_ids = np.array([f"cell{index}" for index in range(n_cells)])
    create_cell_data(root, None, ids=cell_ids, names=cell_ids)
    for assay, values in counts.items():
        array = create_zarr_count_assay(
            root,
            assay,
            None,
            n_cells,
            feat_ids=np.array(feature_ids),
            feat_names=np.array(feature_ids),
            dtype="uint32",
        )
        array[:] = values.astype(np.uint32)
        finalize_test_counts(array)
        finalize_writer_counts_t(root, assay, None)


def _merge_spliced(base: Path) -> Path:
    """Merge 48 cells of spliced and unspliced counts with 24 of spliced only.

    RNA holds the spliced counts of every cell. URNA holds unspliced counts
    over the same feature IDs and measured the first 48 cells only.
    """
    rng = np.random.default_rng(11)
    genes = [f"g{index}" for index in range(40)]
    programs = np.arange(48) % 4
    spliced = rng.poisson(1.0, (48, 40))
    for program in range(4):
        spliced[programs == program, 10 * program : 10 * program + 10] += 6
    _write_shared_features(
        base / "spliced.zarr",
        {"RNA": spliced, "URNA": rng.integers(1, 9, (48, 40))},
        genes,
    )
    _write_shared_features(
        base / "plain.zarr", {"RNA": rng.poisson(3.0, (24, 40)) + 1}, genes
    )
    sources = [
        DataStore(
            str(base / f"{name}.zarr"),
            default_assay="RNA",
            min_features_per_cell=0,
            nthreads=1,
        )
        for name in ("spliced", "plain")
    ]
    path = base / "merged.zarr"
    DataStoreMerge(sources, str(path), ["spliced", "plain"], nthreads=1).dump()
    return path


@pytest.fixture(scope="module")
def spliced(tmp_path_factory) -> Path:
    return _merge_spliced(tmp_path_factory.mktemp("spliced"))


@pytest.mark.slow
def test_layers_refuse_cells_that_their_assay_did_not_measure(spliced) -> None:
    """A layer's assay has no membership declaration in the exported object."""
    store = _open(spliced)

    with pytest.raises(UnmeasuredCellsError) as raised:
        store.to_anndata(layers={"unspliced": "URNA"})

    assert (raised.value.operation, raised.value.assay) == ("to_anndata", "URNA")
    assert (raised.value.unmeasured, raised.value.selected) == (24, 72)
    assert raised.value.cell_key == "I"
    exported = store.to_anndata(cell_key="URNA_I", layers={"unspliced": "URNA"})
    assert exported.layers["unspliced"].shape == (48, 40)
    # The exported assay declares its own membership, so its layer is kept.
    assert store.to_anndata(layers={"raw": "RNA"}).layers["raw"].shape == (72, 40)

    # A run export declares no membership at all.
    run = store.pipeline.run(label="layers", **_PIPELINE)
    with pytest.raises(UnmeasuredCellsError, match="SubsetZarr") as raised:
        store.to_anndata(run=run, layers={"unspliced": "URNA"})
    assert (raised.value.unmeasured, raised.value.selected) == (24, 72)
    assert store.to_anndata(run=run, layers={"raw": "RNA"}).layers["raw"].shape == (
        72,
        40,
    )


@pytest.mark.slow
def test_pipeline_filtering_on_another_assays_metric_reads_that_assay(
    spliced,
) -> None:
    """The pipeline's QC filter reads the assay of each metric, as QC does."""
    store = _open(spliced)
    before = _tree(spliced)
    filtering = {"method": "mad", "attrs": ["RNA_nCounts", "URNA_nCounts"]}

    with pytest.raises(UnmeasuredCellsError) as raised:
        store.pipeline.run(label="urna", **{**_PIPELINE, "filtering": filtering})

    assert (raised.value.operation, raised.value.assay) == ("pipeline.run", "URNA")
    assert (raised.value.unmeasured, raised.value.selected) == (24, 72)
    assert "ds.cells.fetch_all('URNA_I')" in str(raised.value)
    assert _tree(spliced) == before
    # The pipeline's own assay measured every cell, so its metrics pass.
    _recipe(store, "I", filtering={"attrs": ["RNA_nCounts"]})


def test_cluster_tree_fill_values_show_unmeasured_cells_as_missing(measured) -> None:
    store = measured.open()
    membership = np.asarray(store.cells.fetch_all("RNA_I"), dtype=bool)
    graph = measured.refs["legacy_graph"]
    clusters = store.run_paris_clustering(graph)

    prepared = store._prepare_cluster_tree(
        graph=graph, clusters=clusters, fill_by_value="RNA0"
    )

    values = np.asarray(prepared["color_values"])
    assert values.shape == (_N_CELLS,)
    assert np.isnan(values[~membership]).all()
    np.testing.assert_allclose(
        values[membership],
        fetch_normalized_feature_matrix(
            store, [resolve_feature(store, "RNA0")], np.flatnonzero(membership)
        )[:, 0],
        rtol=1e-12,
    )


def test_marker_heatmaps_leave_unmeasured_cells_out_of_group_means(measured) -> None:
    """A marker table that an earlier release found over unmeasured cells."""
    plotting = _plotting()
    store = measured.open()
    membership = np.asarray(store.cells.fetch_all("RNA_I"), dtype=bool)
    labels = measured.refs["labels_all"]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            BaseDataStore, "_require_measured_cells", lambda *_args, **_kwargs: None
        )
        legacy = store.run_marker_search(labels, features=measured.refs["rna_features"])

    result = plotting.marker_heatmap(
        store,
        marker=legacy,
        topn=2,
        cluster_rows=False,
        cluster_columns=False,
        show=False,
    )
    try:
        matrix = result.tables["matrix"]
        assert result.provenance.extras["unmeasured_cells"] == {"RNA": _N_ADT}
        # Group means average measured cells only; z-scores cannot show it,
        # as the unmeasured cells spread evenly over the programs.
        assert result.provenance.n_cells == _N_FULL
    finally:
        result.close()
    rows = np.flatnonzero(membership)
    groups = np.asarray(store.load_cell_values(labels).values).astype(str)[rows]
    for name in matrix.index:
        values = fetch_normalized_feature_matrix(
            store,
            [resolve_feature(store, name)],
            rows,
            NormalizationSpec(transform="log1p"),
        )[:, 0]
        means = pd.Series(values).groupby(groups).mean()
        expected = (means - means.mean()) / means.std()
        np.testing.assert_allclose(
            matrix.loc[name, expected.index.tolist()].to_numpy(dtype=float),
            expected.to_numpy(),
            rtol=1e-6,
        )


@pytest.mark.parametrize(
    ("operation", "call", "error", "match"),
    [
        (
            "select_prevalent_peaks",
            lambda s, m: s.select_prevalent_peaks(m.refs["all"], top_n=5),
            TypeError,
            "ATACassay",
        ),
        (
            "run_cell_cycle_scoring",
            lambda s, m: s.run_cell_cycle_scoring(
                m.refs["all"], from_assay="ATAC", **_CELL_CYCLE
            ),
            TypeError,
            "RNAassay",
        ),
        (
            "run_waggr",
            lambda s, m: s.run_waggr(
                _NETWORK,
                m.refs["all"],
                from_assay="ATAC",
                features=m.refs["rna_features"],
            ),
            TypeError,
            "RNAassay",
        ),
        (
            "run_aucell",
            lambda s, m: s.run_aucell(_NETWORK, m.refs["all"], features="RNA0"),
            TypeError,
            "features must be an ArtifactRef",
        ),
        (
            "select_detected_features",
            lambda s, m: s.select_detected_features(m.refs["all"], min_cells=-1),
            ValueError,
            "min_cells must be non-negative",
        ),
        (
            "run_normalization",
            lambda s, m: s.run_normalization(
                m.refs["all"], m.refs["rna_features"], log_transform="yes"
            ),
            TypeError,
            "log_transform",
        ),
    ],
)
def test_argument_and_assay_type_errors_come_before_the_membership_check(
    measured, operation, call, error, match
) -> None:
    """The membership check runs after an operation's own argument checks.

    ``select_prevalent_peaks`` with the default RNA assay named RNA's
    unmeasured cells instead of saying that it needs an ATAC assay.
    """
    with pytest.raises(error, match=match) as raised:
        call(measured.open(), measured)
    assert not isinstance(raised.value, UnmeasuredCellsError), operation

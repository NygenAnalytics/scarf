import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.datastore.datastore import DataStore
from scarf.datastore.graph_datastore import GraphDataStore
from scarf.graph.encoded_paths import parse_assay_graph_paths
from scarf.graph.paths import StoredAssayGraph
from scarf.graph.state import validate_legacy_graph_selection


_DEFAULT_CORPUS = Path("/home/parashar/data/scarf_master_compat/1k_citeseq")
_MASTER_COMMIT = "1ce016ed17710b7daebcf187c34c6f9b23aae0b4"
_FINAL_TREE_DIGEST = "b27aee365d89d6653d0ac5912f0044d6fe78e5f3a5e20bd164824085c441fedd"

_RNA_GRAPH_SMALL = (
    "RNA/normed__I__compat_hvgs_100/reduction__pca__8__I/"
    "ann__l2__50__50__48__4466/knn__7/graph__1.0__1.5"
)
_RNA_GRAPH_ALTERNATE = (
    "RNA/normed__I__compat_hvgs_100/reduction__pca__12__I/"
    "ann__l2__50__50__48__4466/knn__11/graph__1.2__1.3"
)
_RNA_GRAPH_HARMONY = (
    "RNA/normed__I__compat_hvgs_200/reduction__pca__10__I/"
    "ann__l2__50__50__48__4466/knn__9/graph__1.0__1.5"
)
_ADT_GRAPH = (
    "assay2/normed__I__I/reduction__pca__8__I/"
    "ann__l2__50__50__48__4466/knn__9/graph__1.0__1.5"
)


class _ObservedMasterGraphStore(GraphDataStore):
    @property
    def assay_names(self) -> list[str]:
        return ["RNA"]


def _observed_multibranch_store() -> _ObservedMasterGraphStore:
    datastore = _ObservedMasterGraphStore.__new__(_ObservedMasterGraphStore)
    datastore.z = zarr.open_group(store=MemoryStore(), mode="w")
    datastore.workspace = None
    datastore.zarr_mode = "r+"
    datastore._defaultAssay = "RNA"
    datastore._integratedGraphsLoc = "integratedGraphs"
    datastore._cachedMagicOperator = None
    datastore._cachedMagicOperatorLoc = None
    datastore.nthreads = 1

    normed_path = "RNA/normed__I__compat_hvgs_100"
    normed = datastore.z.create_group(normed_path)
    for graph_path in (_RNA_GRAPH_SMALL, _RNA_GRAPH_ALTERNATE):
        knn_path = graph_path.rsplit("/", 1)[0]
        ann_path = knn_path.rsplit("/", 1)[0]
        reduction_path = ann_path.rsplit("/", 1)[0]
        reduction = datastore.z.create_group(reduction_path)
        ann = datastore.z.create_group(ann_path)
        knn = datastore.z.create_group(knn_path)
        graph = datastore.z.create_group(graph_path)
        k = int(knn_path.rsplit("/", 1)[-1].removeprefix("knn__"))
        reduction.attrs["latest_ann"] = ann_path
        ann.attrs["latest_knn"] = knn_path
        knn.attrs["latest_graph"] = graph_path
        knn.create_array("indices", data=np.zeros((3, k), dtype=np.uint64))
        graph.attrs["n_cells"] = 3
        graph.attrs["n_neighbors"] = k
        graph.create_array(
            "edges",
            data=np.array(
                [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]],
                dtype=np.uint64,
            ),
        )
        graph.create_array("weights", data=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    alternate_knn = _RNA_GRAPH_ALTERNATE.rsplit("/", 1)[0]
    alternate_ann = alternate_knn.rsplit("/", 1)[0]
    normed.attrs["latest_reduction"] = alternate_ann.rsplit("/", 1)[0]
    return datastore


def _corpus_root() -> Path:
    return Path(os.environ.get("SCARF_MASTER_CITESEQ_CORPUS", _DEFAULT_CORPUS))


def _tree_digest(root: Path) -> str:
    digest = hashlib.blake2b(digest_size=32)
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        relative_path = str(path.relative_to(root))
        payload = path.read_bytes()
        encoded_path = relative_path.encode()
        digest.update(len(encoded_path).to_bytes(8, "little"))
        digest.update(encoded_path)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def master_citeseq_corpus() -> tuple[Path, dict]:
    root = _corpus_root()
    store = root / "enriched"
    manifest_path = root / "master_manifest.json"
    if not store.is_dir() or not manifest_path.is_file():
        pytest.skip(
            "Master CITE-seq corpus is unavailable. Set "
            "SCARF_MASTER_CITESEQ_CORPUS to its external directory."
        )
    return store, json.loads(manifest_path.read_text())


@pytest.fixture(scope="module")
def master_citeseq_datastore(
    master_citeseq_corpus: tuple[Path, dict],
) -> DataStore:
    store, _manifest = master_citeseq_corpus
    return DataStore(
        str(store),
        default_assay="RNA",
        assay_types={"assay2": "ADT"},
        min_cells_per_feature=0,
        min_features_per_cell=0,
        zarr_mode="r",
    )


def test_observed_master_graph_branches_parse_without_fixture() -> None:
    small = parse_assay_graph_paths(_RNA_GRAPH_SMALL)
    alternate = parse_assay_graph_paths(_RNA_GRAPH_ALTERNATE)
    harmony = parse_assay_graph_paths(_RNA_GRAPH_HARMONY)
    adt = parse_assay_graph_paths(_ADT_GRAPH)

    assert (small.dims, small.k, small.local_connectivity, small.bandwidth) == (
        8,
        7,
        1.0,
        1.5,
    )
    assert (
        alternate.dims,
        alternate.k,
        alternate.local_connectivity,
        alternate.bandwidth,
    ) == (12, 11, 1.2, 1.3)
    assert harmony.harmony_contract_hash is None
    assert harmony.k == 9
    assert adt.from_assay == "assay2"
    assert adt.feat_key == "I"


def test_observed_master_siblings_keep_latest_and_explicit_reads_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datastore = _observed_multibranch_store()
    monkeypatch.setattr(
        "scarf.datastore._operations.graph.validate_legacy_graph_selection",
        lambda *_args, **_kwargs: None,
    )

    assert (
        datastore.get_latest_graph_loc("RNA", "I", "compat_hvgs_100")
        == _RNA_GRAPH_ALTERNATE
    )
    latest = datastore.load_graph(
        from_assay="RNA", cell_key="I", feat_key="compat_hvgs_100"
    )
    explicit = datastore.load_graph(
        from_assay="RNA",
        cell_key="I",
        feat_key="compat_hvgs_100",
        graph_loc=_RNA_GRAPH_SMALL,
    )
    assert latest.shape == (3, 3)
    assert explicit.shape == (3, 3)
    assert (latest != explicit).nnz == 0


@pytest.mark.integration
def test_master_citeseq_manifest_covers_requested_artifacts(
    master_citeseq_corpus: tuple[Path, dict],
) -> None:
    store, manifest = master_citeseq_corpus
    successful = {
        step["name"] for step in manifest["steps"] if step["status"] == "success"
    }
    steps = {step["name"]: step for step in manifest["steps"]}
    required = {
        "mark_hvgs_100",
        "mark_hvgs_200",
        "rna_graph_small",
        "rna_graph_small_repeat",
        "rna_graph_alternate",
        "rna_graph_unscaled",
        "rna_graph_harmony",
        "adt_graph",
        "integrate_assays_snn",
        "integrate_assays_wnn",
        "run_umap_a",
        "run_umap_b",
        "run_leiden_0_5",
        "run_leiden_1_0",
        "run_leiden_1_5",
        "run_paris_fixed",
        "run_paris_balanced",
        "run_marker_search",
        "get_imputed_t2",
        "get_imputed_t3",
        "run_pseudotime_scoring",
        "run_pseudotime_marker_search",
        "run_pseudotime_aggregation",
        "run_mapping",
        "run_mapping_coral",
        "run_unified_umap",
        "run_topacedo_sampler",
        "metric_lisi",
    }

    assert manifest["master_commit"] == _MASTER_COMMIT
    assert required <= successful
    assert manifest["summary"] == {"failed": 2, "skipped": 1, "success": 39}
    assert len(manifest["layout"]) == 308
    assert manifest["final_tree_digest"] == _FINAL_TREE_DIGEST
    assert _tree_digest(store) == _FINAL_TREE_DIGEST
    assert (
        steps["rna_graph_small"]["result"] == steps["rna_graph_small_repeat"]["result"]
    )
    assert steps["rna_graph_small_repeat"]["changed_file_count"] > 0
    assert steps["rna_graph_unscaled"]["result"] == steps["rna_graph_harmony"]["result"]
    assert any(
        path.endswith("/harmonizedData")
        for path in steps["rna_graph_harmony"]["changed_nodes"]
    )
    assert any(
        path.startswith("RNA/normed__")
        for path in steps["integrate_assays_wnn"]["changed_nodes"]
    )


@pytest.mark.integration
def test_current_reader_loads_master_graph_branches(
    master_citeseq_datastore: DataStore,
) -> None:
    datastore = master_citeseq_datastore
    n_cells = int(datastore.cells.fetch_all("I").sum())

    assert (
        datastore.get_normalized_group_path("RNA", "I", "compat_hvgs_100")
        == "RNA/normed__I__compat_hvgs_100"
    )
    assert (
        datastore.get_latest_graph_loc("RNA", "I", "compat_hvgs_100")
        == _RNA_GRAPH_ALTERNATE
    )
    assert (
        datastore.get_latest_graph_loc("RNA", "I", "compat_hvgs_200")
        == _RNA_GRAPH_HARMONY
    )
    assert datastore.get_latest_graph_loc(
        "RNA", "I", "compat_hvgs_100"
    ) == datastore.get_latest_graph_loc("RNA", "I", "compat_hvgs_100")

    small = datastore.load_graph(graph_loc=_RNA_GRAPH_SMALL)
    alternate = datastore.load_graph(graph_loc=_RNA_GRAPH_ALTERNATE)
    harmony = datastore.load_graph(graph_loc=_RNA_GRAPH_HARMONY)
    adt = datastore.load_graph(graph_loc=_ADT_GRAPH)
    snn = datastore.load_graph(graph_loc="integratedGraphs/compat_snn")
    wnn = datastore.load_graph(graph_loc="integratedGraphs/compat_wnn")

    for graph in (small, alternate, harmony, adt, snn, wnn):
        assert graph.shape == (n_cells, n_cells)
        assert graph.nnz > 0

    stored = datastore._lookup_stored_graph("RNA", "I", "compat_hvgs_100")
    assert isinstance(stored, StoredAssayGraph)
    assert stored.paths.cell_graph_group_path == _RNA_GRAPH_ALTERNATE
    assert stored.dims == 12
    assert stored.k == 11

    harmony_knn = _RNA_GRAPH_HARMONY.rsplit("/", 1)[0]
    harmony_ann = harmony_knn.rsplit("/", 1)[0]
    harmony_reduction = harmony_ann.rsplit("/", 1)[0]
    assert datastore.zw[harmony_ann].attrs["isHarmonized"] is False
    assert "harmonizedData" in datastore.zw[harmony_reduction]

    ann_stream = datastore._load_ann_stream(
        "RNA",
        "I",
        "compat_hvgs_100",
        knn_loc=_RNA_GRAPH_ALTERNATE.rsplit("/", 1)[0],
    )
    assert ann_stream.annIdx is not None
    assert ann_stream.k == 11

    initial_embedding = datastore._get_ini_embed("RNA", "I", "compat_hvgs_100", 2)
    assert initial_embedding.shape == (n_cells, 2)
    assert np.isfinite(initial_embedding).all()


@pytest.mark.integration
def test_current_reader_validates_released_selection_hash(
    master_citeseq_datastore: DataStore,
) -> None:
    datastore = master_citeseq_datastore
    knn_path = _RNA_GRAPH_ALTERNATE.rsplit("/", 1)[0]
    stored_hash = datastore.zw["RNA/normed__I__compat_hvgs_100"].attrs["subset_hash"]

    assert isinstance(stored_hash, int) and not isinstance(stored_hash, bool)
    validate_legacy_graph_selection(
        datastore,
        knn_path,
        "RNA",
        "I",
        "compat_hvgs_100",
    )
    assert datastore._keys_from_knn_path("RNA", knn_path) == (
        "I",
        "compat_hvgs_100",
    )


@pytest.mark.integration
def test_current_reader_loads_master_derived_results(
    master_citeseq_datastore: DataStore,
) -> None:
    datastore = master_citeseq_datastore
    expected_columns = {
        "RNA_compat_umap_a1",
        "RNA_compat_umap_a2",
        "RNA_compat_umap_b1",
        "RNA_compat_umap_b2",
        "RNA_compat_leiden_05",
        "RNA_compat_leiden_1",
        "RNA_compat_leiden_15",
        "RNA_compat_paris_8",
        "RNA_compat_paris_balanced",
        "RNA_compat_pseudotime",
        "RNA_compat_cell_cycle",
        "RNA_I_cluster_membership_strength",
    }
    assert expected_columns <= set(datastore.cells.columns)

    umap = np.column_stack(
        (
            datastore.cells.fetch_all("RNA_compat_umap_a1"),
            datastore.cells.fetch_all("RNA_compat_umap_a2"),
        )
    )
    assert np.isfinite(umap[datastore.cells.fetch_all("I")]).all()
    assert len(set(datastore.cells.fetch("RNA_compat_leiden_1").tolist())) > 1

    markers = datastore.get_markers(
        from_assay="RNA",
        cell_key="I",
        group_key="RNA_compat_leiden_1",
        group_id=1,
        min_score=0.2,
        min_frac_exp=0,
    )
    assert not markers.empty
    assert {"feature_name", "score", "p_value"} <= set(markers.columns)

    feature_name = str(datastore.get_assay("RNA").feats.fetch_all("names")[0])
    imputed = datastore.get_imputed(
        from_assay="RNA",
        cell_key="I",
        feat_key="compat_hvgs_100",
        feature_name=feature_name,
        t=2,
    )
    assert imputed.shape == datastore.cells.fetch_all("I").shape

    with pytest.raises(
        ValueError,
        match=r"build_mapping_reference\(neighbors\)",
    ):
        datastore.get_mapping_reference(from_assay="RNA")

    assert (
        datastore.list_artifacts(
            kind="projection",
            from_assay="RNA",
        )
        == []
    )
    for legacy_name in ("compat_self", "compat_unified_umap"):
        with pytest.raises(ValueError, match="MappingReference is required"):
            datastore.get_mapping_result(legacy_name)

    with pytest.raises(ValueError, match="MappingReference is required"):
        datastore.get_target_classes(
            "compat_self",
            reference_class_group="RNA_compat_leiden_1",
        )
    with pytest.raises(ValueError, match="MappingReference is required"):
        list(datastore.get_mapping_score("compat_self"))

    with pytest.raises(AttributeError, match="_load_unified_layout_data"):
        datastore._load_unified_layout_data("compat_unified_umap", "RNA")
    with pytest.raises(AttributeError, match="unified_embedding"):
        datastore.plots.unified_embedding()


@pytest.mark.integration
def test_current_reader_does_not_mutate_enriched_master_store(
    master_citeseq_corpus: tuple[Path, dict],
) -> None:
    store, manifest = master_citeseq_corpus
    before = _tree_digest(store)
    datastore = DataStore(
        str(store),
        default_assay="RNA",
        assay_types={"assay2": "ADT"},
        min_cells_per_feature=0,
        min_features_per_cell=0,
        zarr_mode="r",
    )

    datastore.get_normalized_group_path("RNA", "I", "compat_hvgs_100")
    datastore.get_latest_graph_loc("RNA", "I", "compat_hvgs_100")
    datastore.load_graph(graph_loc=_RNA_GRAPH_SMALL)
    datastore.load_graph(graph_loc="integratedGraphs/compat_wnn")
    datastore._lookup_stored_graph("RNA", "I", "compat_hvgs_100")
    datastore._get_ini_embed("RNA", "I", "compat_hvgs_100", 2)

    after = _tree_digest(store)
    assert before == manifest["final_tree_digest"]
    assert after == before

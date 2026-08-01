from dataclasses import fields

import numpy as np
import pytest

from scarf import ArtifactRef
from scarf.storage.artifacts import parse_artifact_path
from scarf.graph.state import AssayState
from tests.fixtures_datastore import build_neighbourhood_graph

pytestmark = pytest.mark.slow


def _graph_kwargs() -> dict:
    return {
        "from_assay": "RNA",
        "cell_key": "I",
        "feat_key": "artifact_hvgs",
        "dims": 5,
        "k": 3,
        "n_centroids": 10,
        "batch_size": 200,
        "local_cache": False,
    }


def _prepare_features(datastore) -> None:
    if "I__artifact_hvgs" not in datastore.get_assay("RNA").feats.columns:
        datastore.mark_hvgs(
            from_assay="RNA",
            cell_key="I",
            top_n=100,
            hvg_key_name="artifact_hvgs",
            show_plot=False,
        )


def _state_refs(state: AssayState) -> dict[str, str]:
    return {
        field.name: ref.artifact_id
        for field in fields(state)
        if (ref := getattr(state, field.name, None)) is not None
        and hasattr(ref, "artifact_id")
    }


def test_graph_construction_chain_writes_only_artifacts_and_state(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    _prepare_features(datastore)
    assay_attrs_before = dict(datastore.get_assay("RNA").attrs)
    build_neighbourhood_graph(datastore, **_graph_kwargs())
    state = datastore.get_assay_state("RNA")

    assert state is not None
    assert state.cell_key == "I"
    assert state.feat_key == "artifact_hvgs"
    stored = datastore._lookup_stored_graph("RNA", "I", "artifact_hvgs")
    assert stored.reduction_method == "pca"
    assert stored.dims == 5
    assert stored.pca_cell_key == "I"
    assert stored.feat_scaling is True
    assert stored.ann_metric is not None
    assert stored.ann_efc is not None
    assert stored.ann_ef is not None
    assert stored.ann_m is not None
    assert stored.rand_state is not None
    assert stored.k == 3
    assert stored.local_connectivity is not None
    assert stored.bandwidth is not None
    assert stored.n_centroids == 10
    for field_name in (
        "normalized",
        "feature_scaling",
        "reduction",
        "ann_index",
        "embedding_initialization",
        "neighbors",
        "connectivity_map",
    ):
        ref = getattr(state, field_name)
        assert ref is not None
        status = datastore.inspect_artifact(ref)
        assert status.complete
        assert status.provenance is not None
        assert status.path.startswith(f"RNA/artifacts/{ref.kind}/")
        attrs = datastore.zw[status.path].attrs
        assert set(attrs) >= {
            "artifact_id",
            "kind",
            "provenance",
            "execution_options",
            "complete",
        }
    assert state.reduction is not None
    reduction_execution = datastore.inspect_artifact(state.reduction).execution_options
    assert reduction_execution is not None
    assert reduction_execution["local_cache"] is False

    graph = datastore.load_graph()
    assert graph.shape[0] == int(datastore.cells.fetch_all("I").sum())
    assert graph.nnz > 0
    assert datastore.get_assay("RNA").attrs.get(
        "latest_cell_key"
    ) == assay_attrs_before.get("latest_cell_key")
    assert datastore.get_assay("RNA").attrs.get(
        "latest_feat_key"
    ) == assay_attrs_before.get("latest_feat_key")
    assert "dataset_fingerprint" in datastore.get_assay("RNA").attrs
    assert "RNA/normed__I__artifact_hvgs" not in datastore.zw
    assert datastore.list_artifacts(
        kind="cell_selection",
        scope="datastore",
    )
    assert datastore.list_artifacts(
        kind="feature_selection",
        from_assay="RNA",
    )
    datastore.run_umap(n_epochs=10, label="artifact_umap")
    datastore.run_leiden_clustering(label="artifact_leiden")
    lisi = datastore.metric_lisi(
        ["RNA_artifact_leiden"],
        perplexity=1,
    )
    assert "RNA_artifact_umap1" in datastore.cells.columns
    assert "RNA_artifact_leiden" in datastore.cells.columns
    assert set(lisi) == {"RNA_artifact_leiden"}
    scores = lisi["RNA_artifact_leiden"]
    assert len(scores) == len(datastore.cells.fetch("RNA_artifact_leiden", key="I"))
    assert np.isfinite(scores).all()
    assert (scores >= 1).all()


def test_graph_construction_chain_reuses_provenance(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    _prepare_features(datastore)
    kwargs = _graph_kwargs()
    build_neighbourhood_graph(datastore, **kwargs)
    first = datastore.get_assay_state("RNA")
    assert first is not None
    first_refs = _state_refs(first)

    build_neighbourhood_graph(datastore, **kwargs)
    reused = datastore.get_assay_state("RNA")
    assert reused is not None
    assert _state_refs(reused) == first_refs


def test_graph_construction_chain_records_pca_selection(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    _prepare_features(datastore)
    pca_cells = np.asarray(datastore.cells.fetch_all("I"), dtype=bool)
    pca_cells[np.flatnonzero(pca_cells)[::3]] = False
    datastore.cells.insert("pca_cells", pca_cells, overwrite=True)
    kwargs = _graph_kwargs() | {"pca_cell_key": "pca_cells"}

    build_neighbourhood_graph(datastore, **kwargs)
    first = datastore.get_assay_state("RNA")
    assert first is not None

    assert datastore.get_assay_state("RNA") == first
    assert datastore._lookup_stored_graph().pca_cell_key == "pca_cells"


def test_lsi_artifact_replay_keeps_requested_dimensions(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    _prepare_features(datastore)
    kwargs = _graph_kwargs() | {
        "reduction_method": "lsi",
        "dims": 3,
        "feat_scaling": False,
        "lsi_skip_first": True,
    }
    build_neighbourhood_graph(datastore, **kwargs)

    state = datastore.get_assay_state("RNA")
    assert state is not None and state.reduction is not None
    reduction = datastore.load_artifact(state.reduction)
    assert reduction["loadings"].shape[1] == 3
    stored = datastore._lookup_stored_graph()
    assert stored.reduction_method == "lsi"
    assert stored.dims == 3
    assert stored.pca_cell_key is None
    assert stored.feat_scaling is False


def test_graph_construction_chain_treats_structurally_invalid_cache_as_missing(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    _prepare_features(datastore)
    kwargs = _graph_kwargs()
    build_neighbourhood_graph(datastore, **kwargs)
    first = datastore.get_assay_state("RNA")
    assert first is not None and first.normalized is not None
    normalized_path = datastore.inspect_artifact(first.normalized).path
    del datastore.zw[normalized_path]["data"]

    build_neighbourhood_graph(datastore, **kwargs)
    repaired = datastore.get_assay_state("RNA")

    assert repaired is not None and repaired.normalized is not None
    assert repaired.normalized != first.normalized
    assert "data" in datastore.zw[datastore.inspect_artifact(repaired.normalized).path]
    assert datastore.load_graph().nnz > 0


def test_paris_writes_hierarchy_cut_and_dendrogram_artifacts(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    _prepare_features(datastore)
    build_neighbourhood_graph(datastore, **_graph_kwargs())

    result = datastore.run_paris_clustering(
        from_assay="RNA",
        cell_key="I",
        feat_key="artifact_hvgs",
        n_clusters=5,
        label="artifact_paris",
    )
    column = datastore.zw["cellData"]["RNA_artifact_paris"]
    cut_ref = ArtifactRef.from_dict(column.attrs["source_artifact"])

    assert result.n_clusters == 5
    assert cut_ref.kind == "cluster_cut"
    assert datastore.inspect_artifact(cut_ref).complete
    assert datastore.list_artifacts(
        kind="cluster_hierarchy",
        from_assay="RNA",
    )
    assert datastore.list_artifacts(kind="dendrogram", from_assay="RNA")

    adaptive = datastore.run_paris_clustering(
        from_assay="RNA",
        cell_key="I",
        feat_key="artifact_hvgs",
        n_clusters="auto",
        min_cluster_size=10,
        label="artifact_paris_auto",
    )
    assert adaptive.mode == "auto"
    assert adaptive.n_clusters >= 1

    tree = datastore._prepare_cluster_tree(
        from_assay="RNA",
        cell_key="I",
        feat_key="artifact_hvgs",
        cluster_key="RNA_artifact_paris",
    )
    assert tree["graph"].number_of_nodes() > 0
    assert datastore.list_artifacts(kind="coalesced_tree", from_assay="RNA")


def test_integrated_graphs_use_random_artifacts_and_label_index(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    _prepare_features(datastore)
    build_neighbourhood_graph(datastore, **_graph_kwargs())
    build_neighbourhood_graph(
        datastore,
        from_assay="assay2",
        cell_key="I",
        feat_key="I",
        dims=5,
        k=3,
        n_centroids=10,
        batch_size=200,
        local_cache=False,
    )

    integrated = datastore.integrate_assays(
        ["RNA", "assay2"],
        label="artifact_snn",
        method="snn",
    )
    path = datastore._resolve_integrated_graph_path("artifact_snn")
    ref = parse_artifact_path(path)
    assert integrated == ref
    assert ref.kind == "integrated_graph"
    assert datastore.inspect_artifact(ref).complete
    assert "integratedGraphs/artifact_snn" not in datastore.zw

    umap_ref = datastore.run_umap(
        integrated_graph="artifact_snn",
        n_epochs=10,
        label="UMAP",
    )
    assert "artifact_snn_UMAP1" in datastore.cells.columns
    assert datastore.run_umap(integrated, n_epochs=10, label="UMAP") == umap_ref
    with pytest.raises(ValueError, match="not both"):
        datastore.run_umap(
            integrated,
            integrated_graph="artifact_snn",
            n_epochs=10,
            label="UMAP",
        )
    datastore.run_paris_clustering(
        integrated_graph="artifact_snn",
        n_clusters=3,
        label="paris",
    )
    tree = datastore._prepare_cluster_tree(
        integrated_graph="artifact_snn",
        cluster_key="artifact_snn_paris",
    )
    assert tree["coalesced_location"].startswith(
        "artifacts/coalesced_tree/",
    )

    alias = datastore.integrate_assays(
        ["RNA", "assay2"],
        label="artifact_snn_alias",
        method="snn",
    )
    assert alias == integrated
    with pytest.raises(ValueError, match="shared by labels"):
        datastore.run_umap(integrated, n_epochs=10, label="UMAP")


def test_wnn_uses_exact_nondefault_cell_selection(
    datastore_ephemeral,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datastore = datastore_ephemeral
    _prepare_features(datastore)
    cells = np.asarray(datastore.cells.fetch_all("I"), dtype=bool)
    datastore.cells.insert("wnn_cells", cells, overwrite=True)
    datastore.cells.insert(
        "wnn_batch",
        np.where(np.arange(datastore.cells.N) % 2, "a", "b"),
        overwrite=True,
    )
    datastore.RNA.feats.insert(
        "wnn_cells__artifact_hvgs",
        datastore.RNA.feats.fetch_all("I__artifact_hvgs"),
        overwrite=True,
    )
    build_neighbourhood_graph(
        datastore,
        from_assay="RNA",
        cell_key="wnn_cells",
        feat_key="artifact_hvgs",
        dims=3,
        k=3,
        n_centroids=10,
        harmonize=True,
        batch_columns=["wnn_batch"],
        harmony_params={"nclust": 5},
        local_cache=False,
    )
    build_neighbourhood_graph(
        datastore,
        from_assay="assay2",
        cell_key="wnn_cells",
        feat_key="I",
        dims=3,
        k=3,
        n_centroids=10,
        local_cache=False,
    )

    load_graph = datastore.load_graph

    def reject_load_graph(*_args, **_kwargs):
        raise AssertionError("WNN integration must read neighbors indices directly")

    monkeypatch.setattr(datastore, "load_graph", reject_load_graph)
    datastore.integrate_assays(
        ["RNA", "assay2"],
        label="artifact_wnn",
        method="wnn",
    )

    path = datastore._resolve_integrated_graph_path("artifact_wnn")
    assert path.startswith("artifacts/integrated_graph/")
    ref = parse_artifact_path(path)
    group = datastore.zw[path]
    graph = load_graph(graph_loc=path)
    assert graph.shape[0] == int(cells.sum())
    assert group["edges"].dtype == np.dtype(np.uint32)
    assert group["weights"].dtype == np.dtype(np.float32)
    assert group["modality_weights"].dtype == np.dtype(np.float32)
    assert group["modality_weights"].shape == (int(cells.sum()), 2)
    assert group.attrs["assays"] == ["RNA", "assay2"]
    np.testing.assert_allclose(
        np.asarray(group["modality_weights"][:]).sum(axis=1),
        1,
        rtol=1e-6,
    )

    status = datastore.inspect_artifact(ref)
    assert status.parameters == {
        "method": "wnn",
        "assays": ["RNA", "assay2"],
        "l2_normalize": True,
    }
    assert set(status.inputs["RNA"]) == {"neighbors", "coordinates"}
    assert set(status.inputs["assay2"]) == {"neighbors", "coordinates"}

    weight_columns = [
        "artifact_wnn_RNA_weight",
        "artifact_wnn_assay2_weight",
    ]
    for index, column in enumerate(weight_columns):
        assert column in datastore.cells.columns
        np.testing.assert_allclose(
            datastore.cells.fetch(column, "wnn_cells"),
            np.asarray(group["modality_weights"][:, index]),
        )
        attrs = datastore.zw["cellData"][column].attrs
        assert attrs["source_artifact"] == ref.to_dict()
        assert attrs["source_value"] == "modality_weights"
        assert attrs["value_index"] == index

    for column in weight_columns:
        datastore.cells.drop(column)
    datastore.integrate_assays(
        ["RNA", "assay2"],
        label="artifact_wnn",
        method="wnn",
    )
    assert datastore._resolve_integrated_graph_path("artifact_wnn") == path
    assert all(column in datastore.cells.columns for column in weight_columns)

    datastore.integrate_assays(
        ["RNA", "assay2"],
        label="artifact_wnn_raw",
        method="wnn",
        l2_normalize=False,
    )
    raw_path = datastore._resolve_integrated_graph_path("artifact_wnn_raw")
    assert raw_path != path
    raw_ref = parse_artifact_path(raw_path)
    assert datastore.inspect_artifact(raw_ref).parameters["l2_normalize"] is False
    datastore.integrate_assays(
        ["RNA", "assay2"],
        label="artifact_wnn_raw",
        method="wnn",
        l2_normalize=False,
    )
    assert datastore._resolve_integrated_graph_path("artifact_wnn_raw") == raw_path

    expected_edges = np.asarray(group["edges"][:])
    expected_weights = np.asarray(group["weights"][:])
    expected_modality_weights = np.asarray(group["modality_weights"][:])
    datastore.integrate_assays(
        ["RNA", "assay2"],
        label="artifact_wnn_rechunked",
        method="wnn",
        chunk_size=17,
        invalidate_cache=True,
    )
    rechunked_path = datastore._resolve_integrated_graph_path("artifact_wnn_rechunked")
    assert rechunked_path != path
    rechunked = datastore.zw[rechunked_path]
    np.testing.assert_array_equal(rechunked["edges"][:], expected_edges)
    np.testing.assert_array_equal(rechunked["weights"][:], expected_weights)
    np.testing.assert_array_equal(
        rechunked["modality_weights"][:],
        expected_modality_weights,
    )

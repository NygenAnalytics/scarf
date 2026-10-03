import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import zarr
from zarr.storage import MemoryStore

import scarf.metadata.artifacts as metadata_artifacts_module
import scarf.metadata.selection as metadata_selection_module
from scarf.datastore.datastore import DataStore
from scarf.graph.feature_projection import graph_cell_selection
from scarf.metadata.artifacts import (
    artifact_values,
    categorical_display,
    continuous_display,
    plan_cell_data_artifact,
    validate_display_metadata,
    write_cell_data_artifact,
)
from scarf.metadata.selection import resolve_cell_aligned_artifact
from scarf.quality_control.cell_cycle_genes import g2m_phase_genes, s_phase_genes
from scarf.storage.artifacts import (
    ArtifactRef,
    artifact_group,
    artifact_path,
    inspect_artifact,
)
from scarf.storage.errors import ArtifactResolutionError
from scarf.storage.selections import (
    read_stored_selection_indices,
    resolve_generated_selection_artifact,
)
from tests.fixtures_datastore import build_neighbourhood_graph


def _ensure_graph(datastore) -> ArtifactRef:
    cell_selection = datastore.auto_filter_cells()
    feature_selection = datastore.select_hvgs(
        cell_selection,
        from_assay="RNA",
        top_n=100,
        show_plot=False,
        min_cells=int(0.01 * datastore.cells.N),
        max_cells=np.inf,
        blacklist="^MT-|^RPS|^RPL|^MRPS|^MRPL|^CCN|^HLA-|^H2-|^HIST",
    )
    return build_neighbourhood_graph(
        datastore,
        from_assay="RNA",
        cell_selection=cell_selection,
        features=feature_selection,
        dims=5,
        k=3,
        n_centroids=10,
        local_cache=False,
    )


@pytest.fixture(scope="module")
def graph_template(datastore_zarr_root, tmp_path_factory) -> tuple[Path, ArtifactRef]:
    """Build the small-k graph of the 1K PBMC store once for this module."""
    path = tmp_path_factory.mktemp("metadata_graph") / "store.zarr"
    shutil.copytree(datastore_zarr_root, path)
    return path, _ensure_graph(DataStore(str(path), default_assay="RNA"))


@pytest.fixture
def graph_store(graph_template, tmp_path) -> tuple[DataStore, ArtifactRef]:
    """A private copy of the graph store, which a test may change freely."""
    template, graph = graph_template
    target = tmp_path / "store.zarr"
    shutil.copytree(template, target)
    return DataStore(str(target), default_assay="RNA"), graph


def _selected_cells(datastore, selection: ArtifactRef) -> np.ndarray:
    return read_stored_selection_indices(
        datastore.zw,
        selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )


def _input(datastore, ref: ArtifactRef, name: str) -> ArtifactRef:
    return ArtifactRef.from_dict(datastore.inspect_artifact(ref).inputs[name])


def _graph_neighbors(datastore, graph: ArtifactRef) -> ArtifactRef:
    return ArtifactRef.from_dict(datastore.inspect_artifact(graph).inputs["neighbors"])


def _graph_coordinates(datastore, graph: ArtifactRef) -> ArtifactRef:
    neighbors = _graph_neighbors(datastore, graph)
    return ArtifactRef.from_dict(
        datastore.inspect_artifact(neighbors).inputs["coordinates"]
    )


def _graph_initialization(datastore, graph: ArtifactRef) -> ArtifactRef:
    coordinates = _graph_coordinates(datastore, graph)
    matches = [
        ref
        for ref in datastore.list_artifacts(
            kind="embedding_initialization",
            from_assay=coordinates.assay,
            scope="assay",
            complete_only=True,
        )
        if ArtifactRef.from_dict(datastore.inspect_artifact(ref).inputs["coordinates"])
        == coordinates
    ]
    assert len(matches) == 1
    return matches[0]


def _memory_metadata_root() -> tuple[zarr.Group, ArtifactRef]:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    root.create_group("RNA").create_group("featureData")
    cell_data = root.create_group("cellData")
    cell_ids = np.asarray(["cell-0", "cell-1", "cell-2"])
    selection = np.asarray([True, False, True])
    cell_data.create_array("ids", data=cell_ids)
    cell_data.create_array("I", data=selection)
    selection_ref = resolve_generated_selection_artifact(
        root,
        scope="datastore",
        kind="cell_selection",
        values=selection,
        row_ids=cell_ids,
        operation="manual_selection",
        parameters={},
        inputs={},
        source_column="I",
    )[0]
    return root, selection_ref


def _metadata_snapshot(datastore) -> dict[str, tuple[np.ndarray, dict]]:
    return {
        column: (
            np.asarray(datastore.cells.fetch_all(column)).copy(),
            dict(datastore.zw["cellData"][column].attrs),
        )
        for column in datastore.cells.columns
    }


def _assert_metadata_unchanged(datastore, before) -> None:
    assert set(datastore.cells.columns) == set(before)
    for column, (values, attrs) in before.items():
        np.testing.assert_array_equal(datastore.cells.fetch_all(column), values)
        assert dict(datastore.zw["cellData"][column].attrs) == attrs


def test_cell_data_artifact_cache_hit_miss_and_payload_validation(
    monkeypatch,
) -> None:
    root, selection = _memory_metadata_root()
    common = {
        "scope": "datastore",
        "kind": "metadata_snapshot",
        "operation": "cache_metadata",
        "parameters": {"label": "batch"},
        "inputs": {},
        "execution_options": {"source_column": "batch"},
        "cell_selection": selection,
        "arrays": {"values": ((2,), "f")},
    }
    values = np.asarray([0.25, 0.75], dtype=np.float64)
    first = plan_cell_data_artifact(root, **common)
    first_group = write_cell_data_artifact(root, first, {"values": values})

    assert first.reused is False
    assert inspect_artifact(root, first.ref).complete
    np.testing.assert_array_equal(first_group["values"][:], values)

    cached = plan_cell_data_artifact(root, **common)
    assert cached.reused is True
    assert cached.ref == first.ref

    def fail_if_written(*_args, **_kwargs):
        raise AssertionError("a cached metadata artifact was rewritten")

    with monkeypatch.context() as cache_hit:
        cache_hit.setattr(
            metadata_artifacts_module,
            "create_zarr_dataset",
            fail_if_written,
        )
        reused_group = write_cell_data_artifact(root, cached, {"values": values})
    assert reused_group.path == first_group.path

    changed = plan_cell_data_artifact(
        root,
        **{**common, "parameters": {"label": "condition"}},
    )
    invalidated = plan_cell_data_artifact(
        root,
        **{**common, "invalidate_cache": True},
    )
    assert changed.reused is False
    assert changed.ref != first.ref
    assert invalidated.reused is False
    assert invalidated.ref != first.ref

    del first_group["values"]
    corrupt_miss = plan_cell_data_artifact(root, **common)
    assert inspect_artifact(root, first.ref).complete
    assert corrupt_miss.reused is False
    assert corrupt_miss.ref != first.ref


def test_cell_aligned_artifact_resolver_validates_lineage_and_reads_subset(
    monkeypatch,
) -> None:
    root, source_selection = _memory_metadata_root()
    planned = plan_cell_data_artifact(
        root,
        scope="assay",
        assay="RNA",
        kind="quality_metric",
        operation="test_resolve_cell_aligned_artifact",
        parameters={},
        inputs={},
        execution_options={},
        cell_selection=source_selection,
        arrays={"values": ((2,), "f")},
    )
    write_cell_data_artifact(
        root,
        planned,
        {"values": np.asarray([10.0, 30.0])},
    )
    cell_ids = np.asarray(root["cellData"]["ids"][:])
    target_selection = resolve_generated_selection_artifact(
        root,
        scope="datastore",
        kind="cell_selection",
        values=np.asarray([False, False, True]),
        row_ids=cell_ids,
        operation="target_subset",
        parameters={},
        inputs={},
        source_column="artifact",
    )[0]
    read_positions: list[np.ndarray] = []
    read_rows = metadata_selection_module.read_array_rows_chunkwise

    def capture_rows(array, rows):
        read_positions.append(np.asarray(rows).copy())
        return read_rows(array, rows)

    monkeypatch.setattr(
        metadata_selection_module,
        "read_array_rows_chunkwise",
        capture_rows,
    )

    resolved = resolve_cell_aligned_artifact(
        root,
        planned.ref,
        cell_selection=target_selection,
        expected_kind="quality_metric",
    )

    np.testing.assert_array_equal(resolved.values, [30.0])
    np.testing.assert_array_equal(resolved.cell_idx, [2])
    assert resolved.source_cell_selection == source_selection
    assert resolved.cell_selection == target_selection
    assert len(read_positions) == 1
    np.testing.assert_array_equal(read_positions[0], [1])

    outside_selection = resolve_generated_selection_artifact(
        root,
        scope="datastore",
        kind="cell_selection",
        values=np.asarray([False, True, False]),
        row_ids=cell_ids,
        operation="outside_source_selection",
        parameters={},
        inputs={},
        source_column="artifact",
    )[0]
    with pytest.raises(ValueError, match="subset"):
        resolve_cell_aligned_artifact(
            root,
            planned.ref,
            cell_selection=outside_selection,
        )

    artifact_group(root, planned.ref).attrs["complete"] = False
    with pytest.raises(ValueError, match="unavailable or incomplete"):
        resolve_cell_aligned_artifact(root, planned.ref)


def test_cell_data_artifact_validation_and_failed_write_removes_slot() -> None:
    root, selection = _memory_metadata_root()
    wrong_selection = ArtifactRef(
        scope="datastore",
        kind="metadata_snapshot",
        artifact_id="f" * 64,
    )
    common = {
        "scope": "datastore",
        "kind": "metadata_snapshot",
        "operation": "validate_metadata",
        "parameters": {},
        "inputs": {},
        "execution_options": {},
    }

    with pytest.raises(ValueError, match="cell-selection"):
        plan_cell_data_artifact(
            root,
            **common,
            cell_selection=wrong_selection,
            arrays={"values": ((2,), None)},
        )
    with pytest.raises(ValueError, match="selected cell count"):
        plan_cell_data_artifact(
            root,
            **common,
            cell_selection=selection,
            arrays={"values": ((1,), None)},
        )

    planned = plan_cell_data_artifact(
        root,
        **common,
        cell_selection=selection,
        arrays={"values": ((2, 1), None)},
    )
    with pytest.raises(ValueError, match="one-dimensional"):
        write_cell_data_artifact(
            root,
            planned,
            {"values": np.asarray([["a"], ["b"]])},
        )

    assert not inspect_artifact(root, planned.ref).exists
    retry = plan_cell_data_artifact(
        root,
        **common,
        cell_selection=selection,
        arrays={"values": ((2, 1), None)},
    )
    assert retry.reused is False
    assert retry.ref != planned.ref


def test_datastore_rejects_incomplete_import_status(datastore_ephemeral) -> None:
    datastore = datastore_ephemeral
    datastore.zw.attrs["scarf:import_source"] = "synthetic"
    datastore.zw.attrs["scarf:import_complete"] = False

    with pytest.raises(RuntimeError, match="synthetic import is incomplete"):
        DataStore(datastore.zarr_loc, default_assay="RNA")


def test_datastore_rejects_corrupt_imported_metadata(datastore_ephemeral) -> None:
    datastore = datastore_ephemeral
    datastore.zw.attrs["scarf:import_source"] = "synthetic"
    datastore.zw.attrs["scarf:import_complete"] = True
    datastore.zw["cellData"].create_array(
        "truncated_imported_metadata",
        data=np.asarray(["only-one-row"]),
    )

    with pytest.raises(ValueError, match="Metadata table is corrupted"):
        DataStore(datastore.zarr_loc, default_assay="RNA")


def test_embedding_and_clustering_are_artifact_only(graph_store) -> None:
    datastore, graph = graph_store
    before = _metadata_snapshot(datastore)

    embedding = datastore.run_umap(
        graph,
        _graph_initialization(datastore, graph),
        n_epochs=10,
    )
    leiden = datastore.run_leiden_clustering(graph)
    # This k=3 graph has several components, and a fixed cut cannot request
    # fewer clusters than components, so the fixed cut asks for one cluster.
    paris = datastore.run_paris_clustering(graph, n_clusters=1)

    assert embedding.kind == "embedding"
    assert leiden.kind == "cluster_labels"
    assert paris.kind == "cluster_cut"
    n_cells = datastore.load_graph(graph).shape[0]
    coordinates = datastore.load_artifact(embedding)["values"][:]
    assert coordinates.shape == (n_cells, 2)
    assert np.isfinite(coordinates).all()
    assert datastore.inspect_artifact(leiden).parameters["backend"] == "igraph"
    assert len(artifact_values(artifact_group(datastore.zw, leiden), "values")) == (
        n_cells
    )
    # One requested cluster puts every graph cell in the same cluster.
    cut = artifact_values(artifact_group(datastore.zw, paris), "labels")
    assert len(cut) == n_cells
    assert len(np.unique(cut)) == 1
    _assert_metadata_unchanged(datastore, before)


def test_leiden_backend_is_part_of_artifact_identity(graph_store) -> None:
    datastore, graph = graph_store

    native = datastore.run_leiden_clustering(graph)
    legacy = datastore.run_leiden_clustering(graph, backend="leidenalg")

    assert native != legacy
    assert datastore.inspect_artifact(native).parameters["backend"] == "igraph"
    assert datastore.inspect_artifact(legacy).parameters["backend"] == "leidenalg"
    with pytest.raises(ValueError, match="backend"):
        datastore.run_leiden_clustering(
            graph,
            backend="unknown",  # type: ignore[arg-type]
        )


def test_leiden_graph_flags_reach_the_graph_loader_as_booleans(
    graph_store,
    monkeypatch,
):
    datastore, graph = graph_store
    loads: list[tuple[object, object]] = []
    original = datastore._load_graph_artifact

    def recording(graph_ref, *, symmetric, upper_only, use_k):
        loads.append((symmetric, upper_only))
        return original(
            graph_ref, symmetric=symmetric, upper_only=upper_only, use_k=use_k
        )

    monkeypatch.setattr(datastore, "_load_graph_artifact", recording)
    ref = datastore.run_leiden_clustering(
        graph,
        resolution=0.9,
        symmetric_graph=np.True_,
        graph_upper_only=np.False_,
        invalidate_cache=True,
    )

    # A numpy True is recorded as True, so it must also symmetrize the graph.
    assert loads == [(True, False)]
    assert all(type(flag) is bool for flag in loads[0])
    assert datastore.inspect_artifact(ref).parameters["symmetric_graph"] is True
    for flag in ("symmetric_graph", "graph_upper_only"):
        with pytest.raises(TypeError, match=f"{flag} must be a boolean"):
            datastore.run_leiden_clustering(graph, **{flag: 1})


def test_leiden_does_not_reuse_artifacts_without_edge_weighting(graph_store):
    datastore, graph = graph_store
    prepared = datastore._prepare_leiden_clustering(graph)
    provenance = prepared.planned.provenance
    parameters = dict(provenance["parameters"])
    assert parameters.pop("edge_weighting") == "graph"
    selection = ArtifactRef.from_dict(provenance["inputs"]["cell_selection"])
    previous = plan_cell_data_artifact(
        datastore.zw,
        scope="assay",
        assay="RNA",
        kind="cluster_labels",
        operation="run_leiden_clustering",
        parameters=parameters,
        inputs=provenance["inputs"],
        execution_options={},
        cell_selection=selection,
        arrays={"values": ((prepared.n_cells,), "i")},
    )
    write_cell_data_artifact(
        datastore.zw,
        previous,
        {"values": np.full(prepared.n_cells, -1, dtype=np.int64)},
    )

    actual = datastore.run_leiden_clustering(graph)

    assert actual != previous.ref
    assert np.all(artifact_group(datastore.zw, actual)["values"][:] > 0)
    assert datastore.run_leiden_clustering(graph) == actual


def test_membership_and_smart_labels_are_artifact_only(graph_store) -> None:
    datastore, graph = graph_store
    clusters = datastore.run_leiden_clustering(graph)
    columns_before = set(datastore.cells.columns)

    membership = datastore.calc_membership_strength(clusters, graph)
    smart = datastore.smart_label(clusters, clusters)

    assert membership.kind == "membership_strength"
    assert smart.kind == "smart_label"
    assert datastore.calc_membership_strength(clusters, graph) == membership
    assert datastore.smart_label(clusters, clusters) == smart
    assert set(datastore.cells.columns) == columns_before


def test_hto_identity_is_artifact_backed(
    datastore_ephemeral,
    monkeypatch,
) -> None:
    datastore = datastore_ephemeral
    keep = np.asarray(datastore.cells.fetch_all("I"), dtype=bool).copy()
    keep[np.flatnonzero(keep)[::4]] = False
    datastore.cells.insert("hto_cells", keep)
    selection = datastore.snapshot_cell_selection("hto_cells")
    columns_before = set(datastore.cells.columns)
    calls: list[tuple[pd.DataFrame, dict]] = []

    def demultiplex(counts: pd.DataFrame, **kwargs) -> pd.Series:
        # The stand-in names each cell's largest HTO, so the stored values
        # reveal which rows it received and in which order.
        calls.append((counts.copy(), kwargs))
        return counts.idxmax(axis=1)

    monkeypatch.setattr(
        "scarf.datastore._operations.quality_control.hto_demux", demultiplex
    )
    assay_types = dict(datastore.zw.attrs["assayTypes"])
    assay_types["assay2"] = "HTO"
    datastore.zw.attrs["assayTypes"] = assay_types

    ref = datastore.run_hto_demultiplexing(
        selection, from_assay="assay2", random_seed=5
    )

    assert ref.kind == "hto_identity"
    status = datastore.inspect_artifact(ref)
    assert status.operation == "run_hto_demultiplexing"
    parameters = status.parameters
    assert parameters is not None
    assert parameters["method"]["normalization"] == "clr_per_hto"
    assert parameters["random_seed"] == 5
    assert "algorithm_version" not in parameters
    cells = np.flatnonzero(keep)
    raw = np.asarray(datastore.assay2.rawData[cells].compute())
    feature_ids = np.asarray(datastore.assay2.feats.fetch_all("ids"))
    ((counts, kwargs),) = calls
    assert kwargs == {"random_seed": 5}
    assert counts.columns.tolist() == feature_ids.tolist()
    np.testing.assert_array_equal(counts.to_numpy(), raw)
    np.testing.assert_array_equal(
        artifact_values(artifact_group(datastore.zw, ref), "values"),
        feature_ids[raw.argmax(axis=1)],
    )
    datastore.memoryBytes = 1
    assert (
        datastore.run_hto_demultiplexing(
            selection,
            from_assay="assay2",
            random_seed=5,
        )
        == ref
    )
    assert len(calls) == 1
    assert set(datastore.cells.columns) == columns_before


def test_hto_demultiplexing_rejects_non_hto_assay(datastore_ephemeral) -> None:
    datastore = datastore_ephemeral
    selection = datastore.snapshot_cell_selection()

    with pytest.raises(
        TypeError,
        match="^HTO demultiplexing requires an assay declared with type 'HTO'; "
        "'assay2' is declared as 'Assay'$",
    ):
        datastore.run_hto_demultiplexing(selection, from_assay="assay2")
    with pytest.raises(TypeError, match="'HTO' is declared as None"):
        datastore.run_hto_demultiplexing(selection)
    assert datastore.list_artifacts(kind="hto_identity") == []


def test_hto_demultiplexing_respects_datastore_memory_budget(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    selection = datastore.snapshot_cell_selection()
    assay_types = dict(datastore.zw.attrs["assayTypes"])
    assay_types["assay2"] = "HTO"
    datastore.zw.attrs["assayTypes"] = assay_types
    datastore.memoryBytes = 1

    with pytest.raises(MemoryError, match="exceeds the datastore memory budget"):
        datastore.run_hto_demultiplexing(selection, from_assay="assay2")


def test_cell_cycle_scoring_returns_one_artifact_without_writing_columns(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    selection = datastore.auto_filter_cells()
    columns_before = set(datastore.cells.columns)

    ref = datastore.run_cell_cycle_scoring(selection)

    assert ref.kind == "cell_cycle"
    group = artifact_group(datastore.zw, ref)
    assert set(group.array_keys()) == {"s_score", "g2m_score", "phase"}
    n_cells = len(_selected_cells(datastore, selection))
    s_score = artifact_values(group, "s_score")
    g2m_score = artifact_values(group, "g2m_score")
    phase = artifact_values(group, "phase")
    assert s_score.shape == g2m_score.shape == phase.shape == (n_cells,)
    assert np.isfinite(s_score).all() and np.isfinite(g2m_score).all()
    # G1 when both scores are negative, else G2M when it outscores S.
    expected_phase = np.where(
        (s_score < 0) & (g2m_score < 0),
        "G1",
        np.where(g2m_score > s_score, "G2M", "S"),
    )
    np.testing.assert_array_equal(phase, expected_phase)
    assert set(phase) == {"G1", "G2M", "S"}
    # Genes match the default lists by case-insensitive feature name.
    names = np.char.upper(datastore.RNA.feats.fetch_all("names").astype(str))
    status = datastore.inspect_artifact(ref)
    for key, genes in (
        ("s_gene_indices", s_phase_genes),
        ("g2m_gene_indices", g2m_phase_genes),
    ):
        expected = [int(i) for gene in genes for i in np.flatnonzero(names == gene)]
        assert status.parameters[key] == expected
        assert expected
    assert status.parameters["control_size"] == 43
    assert _input(datastore, ref, "cell_selection") == selection
    assert set(datastore.cells.columns) == columns_before


def test_imputation_batches_preserve_requested_columns_and_stream_rows(
    graph_store,
    monkeypatch,
) -> None:
    store, graph = graph_store
    diffusion = store.run_diffusion_operator(graph)
    operator, _graph, selection = store._load_diffusion_operator_with_lineage(diffusion)
    rows = read_stored_selection_indices(
        store.zw,
        selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    genes = np.flatnonzero(store.RNA.feats.fetch_all("nCells") > 10)[:3]
    names = store.RNA.feats.fetch_all("names").copy()
    names[genes[:2]] = "duplicate"
    store.RNA.feats.insert("names", names, overwrite=True)
    numeric = np.arange(store.cells.N, dtype=np.float64)
    metadata_name = str(names[genes[2]])
    store.cells.insert(metadata_name, numeric)
    expression = store.RNA.normed(rows, genes[:2]).compute().mean(axis=1)
    expected = np.column_stack(
        (
            operator.dot(numeric[rows]),
            operator.dot(expression),
            operator.dot(expression),
        )
    )
    original_load = store._load_diffusion_operator_with_lineage
    calls = []
    original_columns = store.cells._column_names
    column_scans = []

    def columns():
        column_scans.append(True)
        return original_columns()

    def load(ref, **kwargs):
        calls.append(ref)
        return original_load(ref, **kwargs)

    original_normed = store.RNA.normed

    def normed(*args, **kwargs):
        return original_normed(*args, **kwargs)._with_block_size(29)

    monkeypatch.setattr(store, "_load_diffusion_operator_with_lineage", load)
    monkeypatch.setattr(store.cells, "_column_names", columns)
    monkeypatch.setattr(store.RNA, "normed", normed)
    actual = store.get_imputed([metadata_name, "DUPLICATE", "duplicate"], diffusion)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert calls == [diffusion]
    assert len(column_scans) <= 2
    assert store.get_imputed([metadata_name], diffusion).shape == (len(rows), 1)
    assert store.get_imputed(metadata_name, diffusion).shape == (len(rows),)
    for invalid in ([], [""], [None]):
        with pytest.raises(ValueError, match="non-empty strings"):
            store.get_imputed(invalid, diffusion)
    with pytest.raises(ValueError, match="not found"):
        store.get_imputed(["not_a_gene"], diffusion)
    monkeypatch.setattr(store, "memoryBytes", 1)
    with pytest.raises(MemoryError, match="fewer features"):
        store.get_imputed([metadata_name, "duplicate"], diffusion)


def test_get_imputed_accepts_array_like_feature_names(
    datastore,
    connectivity_graph,
) -> None:
    diffusion = datastore.run_diffusion_operator(connectivity_graph, t=2)
    names = [str(name) for name in datastore.RNA.feats.fetch_all("names")[:3]]
    expected = datastore.get_imputed(names, diffusion)

    for container in (
        np.asarray(names),
        np.asarray(names, dtype=object),
        pd.Series(names),
        pd.Series(names, index=[7, 3, 5]),
    ):
        actual = datastore.get_imputed(container, diffusion)
        assert actual.shape == expected.shape
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        datastore.get_imputed(np.asarray(names[:1]), diffusion),
        expected[:, :1],
    )

    with pytest.raises(ValueError, match="one-dimensional"):
        datastore.get_imputed(np.asarray([names]), diffusion)
    with pytest.raises(ValueError, match="non-empty strings"):
        datastore.get_imputed(np.asarray([], dtype=str), diffusion)
    with pytest.raises(ValueError, match="non-empty strings"):
        datastore.get_imputed(pd.Series([names[0], 3]), diffusion)


def test_explicit_graph_consumers_ignore_later_live_selection_changes(
    graph_store,
) -> None:
    datastore, graph = graph_store
    graph_n = datastore.load_graph(graph).shape[0]
    initialization = _graph_initialization(datastore, graph)
    mask = np.asarray(datastore.cells.fetch_all("I"), dtype=bool)
    selected = np.flatnonzero(mask)
    assert len(selected) > 1
    mask[selected[0]] = False
    datastore.cells.insert("I", mask, overwrite=True, force=True)

    clusters = datastore.run_leiden_clustering(graph)
    embedding = datastore.run_umap(graph, initialization, n_epochs=10)
    diffusion = datastore.run_diffusion_operator(graph, invalidate_cache=True)
    operator = datastore.load_diffusion_operator(diffusion)
    feature_name = str(datastore.RNA.feats.fetch_all("names")[0])
    imputed = datastore.get_imputed(feature_name, diffusion)

    assert clusters.kind == "cluster_labels"
    assert embedding.kind == "embedding"
    assert diffusion.kind == "diffusion_operator"
    # Every consumer keeps the graph's frozen cells, including the one that
    # the live selection dropped.
    graph_selection = graph_cell_selection(datastore.zw, graph)
    graph_cells = _selected_cells(datastore, graph_selection)
    assert len(graph_cells) == graph_n
    assert selected[0] in graph_cells
    assert _input(datastore, clusters, "cell_selection") == graph_selection
    assert len(artifact_values(artifact_group(datastore.zw, clusters), "values")) == (
        graph_n
    )
    assert datastore.load_artifact(embedding)["values"].shape == (graph_n, 2)
    assert operator.shape == (graph_n, graph_n)
    assert imputed.shape == (graph_n,)


def test_graph_consumers_require_explicit_artifact_refs(graph_store) -> None:
    datastore, graph = graph_store
    coordinates = _graph_coordinates(datastore, graph)
    initialization = _graph_initialization(datastore, graph)

    first = datastore.run_leiden_clustering(graph)
    side_neighbors = datastore.query_neighbors(
        ArtifactRef.from_dict(
            datastore.inspect_artifact(_graph_neighbors(datastore, graph)).inputs[
                "ann_index"
            ]
        ),
        k=5,
    )
    side_graph = datastore.build_connectivity_map(side_neighbors)
    assert datastore.run_leiden_clustering(side_graph) != first

    with pytest.raises(TypeError, match="ArtifactRef"):
        datastore.run_umap(
            "RNA/graph",  # type: ignore[arg-type]
            initialization,
            n_epochs=10,
        )
    with pytest.raises(ValueError, match="connectivity_map"):
        datastore.run_umap(coordinates, initialization, n_epochs=10)


def test_neighbor_metrics_reject_incomplete_ann_dependency(graph_store) -> None:
    datastore, graph = graph_store
    neighbors = _graph_neighbors(datastore, graph)
    ann = ArtifactRef.from_dict(
        datastore.inspect_artifact(neighbors).inputs["ann_index"]
    )
    ann_group = datastore.zw[artifact_path(ann)]
    ann_group.attrs["complete"] = False

    try:
        with pytest.raises(
            ArtifactResolutionError,
            match=r"(?i)artifact is incomplete",
        ) as error:
            datastore.metric_ilisi("names", neighbors=neighbors, perplexity=1)
        assert error.value.code == "incomplete_artifact"
    finally:
        ann_group.attrs["complete"] = True


@pytest.mark.parametrize(
    ("display", "error_type", "message"),
    [
        ({"kind": "continuous"}, ValueError, "incomplete"),
        (
            {
                "kind": "continuous",
                "colormap": 1,
                "minimum": 0.0,
                "maximum": 1.0,
                "scale": "linear",
            },
            TypeError,
            "colormap",
        ),
        (
            {
                "kind": "continuous",
                "colormap": "viridis",
                "minimum": 2.0,
                "maximum": 1.0,
                "scale": "linear",
            },
            ValueError,
            "exceeds",
        ),
        (
            {"kind": "categorical", "categories": "not-a-list"},
            TypeError,
            "must be a list",
        ),
        (
            {"kind": "categorical", "categories": ["not-a-mapping"]},
            TypeError,
            "must be a mapping",
        ),
        (
            {
                "kind": "categorical",
                "categories": [{"value": 1, "label": "one", "color": "red"}],
            },
            ValueError,
            "hex color",
        ),
        ({"kind": "unknown"}, ValueError, "continuous or categorical"),
    ],
)
def test_display_validation_rejects_malformed_contracts(
    display: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        validate_display_metadata(display)


def test_display_validation_rejects_nonfinite_duplicates_and_collisions() -> None:
    with pytest.raises(TypeError, match="minimum"):
        validate_display_metadata(
            {
                "kind": "continuous",
                "colormap": "viridis",
                "minimum": np.nan,
                "maximum": 1.0,
                "scale": "linear",
            }
        )
    with pytest.raises(ValueError, match="unique"):
        validate_display_metadata(
            {
                "kind": "categorical",
                "categories": [
                    {"value": 1, "label": "A", "color": "#123456"},
                    {"value": 1, "label": "B", "color": "#654321"},
                ],
            }
        )
    with pytest.raises(ValueError, match="collide"):
        validate_display_metadata(
            {
                "kind": "categorical",
                "categories": [
                    {"value": True, "label": "Yes", "color": "#123456"},
                    {"value": 1, "label": "One", "color": "#654321"},
                ],
            }
        )


def test_display_metadata_builders_are_deterministic() -> None:
    assert continuous_display(np.asarray([0.25, np.nan, 0.75])) == {
        "kind": "continuous",
        "colormap": "viridis",
        "minimum": 0.25,
        "maximum": 0.75,
        "scale": "linear",
    }
    categorical = categorical_display(np.asarray([2, 1, 2]))
    assert [item["value"] for item in categorical["categories"]] == [1, 2]

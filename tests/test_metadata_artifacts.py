import numpy as np
import pandas as pd
import pytest
import zarr
from zarr.storage import MemoryStore

import scarf.metadata.artifacts as metadata_artifacts_module
import scarf.metadata.selection as metadata_selection_module
from scarf.datastore.datastore import DataStore
from scarf.metadata.artifacts import (
    artifact_values,
    categorical_display,
    continuous_display,
    plan_cell_data_artifact,
    validate_display_metadata,
    write_cell_data_artifact,
)
from scarf.metadata.selection import resolve_cell_aligned_artifact
from scarf.storage.artifacts import (
    ArtifactRef,
    artifact_group,
    artifact_path,
    inspect_artifact,
)
from scarf.storage.errors import ArtifactResolutionError
from scarf.storage.selections import resolve_selection_artifact
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
    selection_ref = resolve_selection_artifact(
        root,
        scope="datastore",
        kind="cell_selection",
        values=selection,
        row_ids=cell_ids,
        operation="manual_selection",
        parameters={},
        inputs={},
        source_column="I",
    )
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
    target_selection = resolve_selection_artifact(
        root,
        scope="datastore",
        kind="cell_selection",
        values=np.asarray([False, False, True]),
        row_ids=cell_ids,
        operation="target_subset",
        parameters={},
        inputs={},
        source_column="artifact",
    )
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

    outside_selection = resolve_selection_artifact(
        root,
        scope="datastore",
        kind="cell_selection",
        values=np.asarray([False, True, False]),
        row_ids=cell_ids,
        operation="outside_source_selection",
        parameters={},
        inputs={},
        source_column="artifact",
    )
    with pytest.raises(ValueError, match="subset"):
        resolve_cell_aligned_artifact(
            root,
            planned.ref,
            cell_selection=outside_selection,
        )

    artifact_group(root, planned.ref).attrs["complete"] = False
    with pytest.raises(ValueError, match="unavailable or incomplete"):
        resolve_cell_aligned_artifact(root, planned.ref)


def test_cell_data_artifact_validation_and_failed_write_status() -> None:
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

    failed_status = inspect_artifact(root, planned.ref)
    assert failed_status.exists
    assert failed_status.complete is False
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


def test_embedding_and_clustering_are_artifact_only(datastore_ephemeral) -> None:
    datastore = datastore_ephemeral
    graph = _ensure_graph(datastore)
    before = _metadata_snapshot(datastore)

    embedding = datastore.run_umap(
        graph,
        _graph_initialization(datastore, graph),
        n_epochs=10,
    )
    leiden = datastore.run_leiden_clustering(graph)
    paris = datastore.run_paris_clustering(graph, n_clusters=3)

    assert embedding.kind == "embedding"
    assert leiden.kind == "cluster_labels"
    assert paris.kind == "cluster_cut"
    assert datastore.load_artifact(embedding)["values"].shape[1] == 2
    assert datastore.inspect_artifact(leiden).parameters["backend"] == "igraph"
    _assert_metadata_unchanged(datastore, before)


def test_leiden_backend_is_part_of_artifact_identity(datastore_ephemeral) -> None:
    datastore = datastore_ephemeral
    graph = _ensure_graph(datastore)

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


def test_membership_smart_labels_and_lisi_are_artifact_only(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    graph = _ensure_graph(datastore)
    neighbors = _graph_neighbors(datastore, graph)
    clusters = datastore.run_leiden_clustering(graph)
    columns_before = set(datastore.cells.columns)

    membership = datastore.calc_membership_strength(clusters, graph)
    smart = datastore.smart_label(clusters, clusters)
    lisi = datastore.metric_lisi(["names"], neighbors, perplexity=1)

    assert membership.kind == "membership_strength"
    assert smart.kind == "smart_label"
    assert lisi.kind == "quality_metric"
    assert datastore.calc_membership_strength(clusters, graph) == membership
    assert datastore.smart_label(clusters, clusters) == smart
    loaded = datastore.load_metric_lisi(lisi)
    assert loaded["names"].shape == (datastore.load_graph(graph).shape[0],)
    assert set(datastore.cells.columns) == columns_before


def test_hto_identity_is_artifact_backed(
    datastore_ephemeral,
    monkeypatch,
) -> None:
    datastore = datastore_ephemeral
    selection = datastore.snapshot_cell_selection()
    n_active = len(datastore.cells.active_index("I"))
    columns_before = set(datastore.cells.columns)
    expected = np.asarray(
        ["negative" if index % 2 else "tag" for index in range(n_active)]
    )
    monkeypatch.setattr(
        "scarf.datastore._operations.quality_control.hto_demux",
        lambda counts, **kwargs: pd.Series(expected[: len(counts)]),
    )
    assay_types = dict(datastore.zw.attrs["assayTypes"])
    assay_types["assay2"] = "HTO"
    datastore.zw.attrs["assayTypes"] = assay_types

    ref = datastore.run_hto_demultiplexing(selection, from_assay="assay2")

    assert ref.kind == "hto_identity"
    status = datastore.inspect_artifact(ref)
    assert status.operation == "run_hto_demultiplexing"
    parameters = status.parameters
    assert parameters is not None
    assert parameters["method"]["normalization"] == "clr_per_hto"
    assert "algorithm_version" not in parameters
    np.testing.assert_array_equal(
        artifact_values(artifact_group(datastore.zw, ref), "values"),
        expected,
    )
    datastore.memoryBytes = 1
    assert (
        datastore.run_hto_demultiplexing(
            selection,
            from_assay="assay2",
        )
        == ref
    )
    assert set(datastore.cells.columns) == columns_before


def test_hto_demultiplexing_rejects_non_hto_assay(datastore_ephemeral) -> None:
    datastore = datastore_ephemeral
    selection = datastore.snapshot_cell_selection()

    with pytest.raises(TypeError, match="declared with type 'HTO'"):
        datastore.run_hto_demultiplexing(selection, from_assay="assay2")


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
    assert len(artifact_values(group, "phase")) == int(
        artifact_values(artifact_group(datastore.zw, selection), "values").sum()
    )
    assert set(datastore.cells.columns) == columns_before


def test_explicit_graph_consumers_ignore_later_live_selection_changes(
    datastore_ephemeral,
) -> None:
    datastore = datastore_ephemeral
    graph = _ensure_graph(datastore)
    graph_n = datastore.load_graph(graph).shape[0]
    neighbors = _graph_neighbors(datastore, graph)
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
    lisi = datastore.metric_lisi(["names"], neighbors, perplexity=1)

    assert clusters.kind == "cluster_labels"
    assert embedding.kind == "embedding"
    assert diffusion.kind == "diffusion_operator"
    assert operator.shape == (graph_n, graph_n)
    assert imputed.shape == (graph_n,)
    assert datastore.load_metric_lisi(lisi)["names"].shape == (graph_n,)


def test_graph_consumers_require_explicit_artifact_refs(datastore_ephemeral) -> None:
    datastore = datastore_ephemeral
    graph = _ensure_graph(datastore)
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


def test_lisi_rejects_incomplete_ann_dependency(datastore_ephemeral) -> None:
    datastore = datastore_ephemeral
    graph = _ensure_graph(datastore)
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
            datastore.metric_lisi(["names"], neighbors=neighbors, perplexity=1)
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

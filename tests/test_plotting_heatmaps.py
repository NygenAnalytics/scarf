import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import scarf.plotting as splt
from scarf.plotting.cluster_tree import _hierarchy_positions
from scarf.plotting._heatmap_utils import (
    annotation_colors,
    normalize_annotations,
    order_heatmap,
)


def test_hierarchy_positions_are_pure_and_complete():
    graph = nx.DiGraph([(4, 2), (4, 3), (2, 0), (2, 1)])
    before = graph.copy()

    positions = _hierarchy_positions(graph, width=2.0)

    assert set(positions) == set(graph)
    assert nx.utils.graphs_equal(graph, before)
    assert positions[4][1] == 0.0
    assert positions[0][1] < positions[2][1]


def test_hierarchy_positions_rejects_non_trees():
    cyclic = nx.DiGraph([(0, 1), (1, 2), (2, 0)])
    with pytest.raises(TypeError, match="not a tree"):
        _hierarchy_positions(cyclic)


def test_tree_palette_requires_complete_color_key():
    from scarf.plotting.cluster_tree import _tree_palette

    with pytest.raises(KeyError, match="missing in `color_key`"):
        _tree_palette(
            object(),
            ["A", "B"],
            cmap="tab20",
            color_key={"A": "#ff0000"},
        )
    color_key = {"A": "#ff0000", "B": "#00ff00"}
    palette = _tree_palette(
        object(),
        ["A", "B"],
        cmap="tab20",
        color_key=color_key,
    )
    assert palette == color_key
    assert palette is not color_key


def test_cluster_tree_rejects_misaligned_color_values():
    from types import SimpleNamespace

    from scarf.plotting.cluster_tree import cluster_tree

    store = SimpleNamespace(
        _prepare_cluster_tree=lambda **_kwargs: {
            "graph": nx.DiGraph([(2, 0), (2, 1)]),
            "clusters": np.array([1, 1]),
            "color_values": np.array([0.1, 0.2, 0.3]),
        }
    )
    with pytest.raises(ValueError, match="misaligned"):
        cluster_tree(store, show=False)


def test_writable_float64_accumulator_accepts_readonly_blocks() -> None:
    from scarf.plotting.heatmaps import _writable_float64

    first = np.array([1.0, 2.0], dtype=np.float64)
    first.flags.writeable = False
    second = np.array([3.0, 4.0], dtype=np.float64)
    second.flags.writeable = False

    total = _writable_float64(first)
    total += _writable_float64(second)

    np.testing.assert_allclose(total, [4.0, 6.0])
    assert first.flags.writeable is False
    np.testing.assert_array_equal(first, [1.0, 2.0])


def test_clip_marker_means_does_not_write_through_readonly_values() -> None:
    from scarf.plotting.heatmaps import _clip_marker_means

    values = np.array([[-2.0, 0.5], [3.0, 1.0]], dtype=np.float64)
    values.flags.writeable = False
    group_means = pd.DataFrame(values, index=["a", "b"], columns=["g1", "g2"])

    matrix = _clip_marker_means(group_means, vmin=-1.0, vmax=2.0)

    assert list(matrix.index) == ["g1", "g2"]
    assert list(matrix.columns) == ["a", "b"]
    np.testing.assert_allclose(matrix.to_numpy(), [[-1.0, 2.0], [0.5, 1.0]])
    assert values[0, 0] == -2.0


def test_marker_heatmap_returns_owned_result(
    marker_search,
    datastore,
):
    import matplotlib.pyplot as plt

    result = splt.marker_heatmap(
        datastore,
        group_key="RNA_cluster",
        topn=3,
        figsize=(4, 6),
        show=False,
    )

    assert isinstance(result, splt.PlotResult)
    assert result.owns_figure is True
    assert tuple(result.figure.get_size_inches()) == (4, 6)
    assert {"heatmap", "row_dendrogram", "column_dendrogram", "colorbar"} <= set(
        result.axes
    )
    assert not result.tables["matrix"].empty
    assert not result.tables["markers"].empty
    assert result.legends
    assert result.scales
    assert result.provenance.notes[0] == "marker_heatmap"
    figure_number = result.figure.number
    assert plt.fignum_exists(figure_number)
    result.close()
    assert not plt.fignum_exists(figure_number)


def test_marker_heatmap_selects_features_by_named_score(
    marker_search,
    datastore,
):
    from scarf.features.markers.table import load_marker_table

    assay = datastore.RNA
    marker_slot = datastore._resolve_marker_group("RNA", "I", "RNA_cluster")
    feature_names = np.asarray(assay.feats.fetch_all("names"))
    feature_ids = np.asarray(assay.feats.fetch_all("ids"))
    expected_by_group: dict[str, list[str]] = {}
    for group_name in marker_slot.group_keys():
        markers = load_marker_table(
            marker_slot,
            marker_slot[group_name],
            feature_names,
            group_id=group_name,
            feature_ids=feature_ids,
        )
        ranked = markers.sort_values(
            ["score", "feature_name"],
            ascending=[False, True],
            kind="mergesort",
        ).head(2)
        expected_by_group[group_name] = ranked["feature_name"].astype(str).tolist()

    result = splt.marker_heatmap(
        datastore,
        group_key="RNA_cluster",
        topn=2,
        cluster_rows=False,
        cluster_columns=False,
        show=False,
    )
    selected = result.tables["markers"]
    for group_name, expected_names in expected_by_group.items():
        got = (
            selected.loc[selected["group"].astype(str) == str(group_name)]
            .sort_values("rank")["feature"]
            .astype(str)
            .tolist()
        )
        assert got == expected_names
    result.close()


def test_marker_heatmap_propagates_marker_metadata_errors(
    marker_search,
    datastore,
):
    marker_slot = datastore._resolve_marker_group("RNA", "I", "RNA_cluster")
    original_method = marker_slot.attrs["method"]
    marker_slot.attrs["method"] = "ttest"
    try:
        with pytest.raises(ValueError, match="Canonical marker metadata 'method'"):
            splt.marker_heatmap(
                datastore,
                group_key="RNA_cluster",
                topn=2,
                show=False,
            )
    finally:
        marker_slot.attrs["method"] = original_method


def test_marker_heatmap_skips_unresolved_legacy_names_with_warning(
    datastore_ephemeral,
):
    from scarf.utils import logger

    assay = datastore_ephemeral.RNA
    group_key = "legacy_heatmap_groups"
    groups = np.arange(datastore_ephemeral.cells.N) % 2
    datastore_ephemeral.cells.insert(group_key, groups, overwrite=True)
    markers_group = (
        assay.z["markers"] if "markers" in assay.z else assay.z.create_group("markers")
    )
    slot = markers_group.create_group(f"I__{group_key}")
    feature_ids = np.asarray(assay.feats.fetch_all("ids")).astype(str)
    for group_id, feature_index in enumerate((0, 1)):
        cluster = slot.create_group(str(group_id))
        cluster.create_array(
            "names",
            data=np.array(
                [
                    f"removed_feature_{group_id}",
                    feature_ids[feature_index],
                ]
            ),
        )
        cluster.create_array("scores", data=np.array([1.0, 0.9]))

    warnings: list[str] = []
    sink = logger.add(
        lambda message: warnings.append(message.record["message"]),
        level="WARNING",
    )
    try:
        result = splt.marker_heatmap(
            datastore_ephemeral,
            from_assay="RNA",
            group_key=group_key,
            cell_key="I",
            topn=1,
            cluster_rows=False,
            cluster_columns=False,
            show=False,
        )
    finally:
        logger.remove(sink)

    assert set(result.tables["markers"]["feature_index"]) == {0, 1}
    assert all(
        "removed_feature" not in value for value in result.tables["matrix"].index
    )
    assert any("unresolved legacy marker" in message for message in warnings)
    result.close()


def test_marker_heatmap_accepts_explicit_order_annotations_and_target(
    marker_search,
    datastore,
):
    import matplotlib.pyplot as plt

    baseline = splt.marker_heatmap(
        datastore,
        group_key="RNA_cluster",
        topn=2,
        cluster_rows=False,
        cluster_columns=False,
        show=False,
    )
    row_order = list(reversed(baseline.tables["matrix"].index.tolist()))
    column_order = list(reversed(baseline.tables["matrix"].columns.tolist()))
    baseline.close()
    row_annotation = {
        feature: "first" if index < len(row_order) / 2 else "second"
        for index, feature in enumerate(row_order)
    }
    figure, ax = plt.subplots(figsize=(4, 4))

    result = splt.marker_heatmap(
        datastore,
        group_key="RNA_cluster",
        topn=2,
        row_order=row_order,
        column_order=column_order,
        cluster_rows=False,
        cluster_columns=False,
        row_annotations={"marker set": row_annotation},
        target=ax,
        show_legend=False,
        show=False,
    )

    assert result.owns_figure is False
    assert result.tables["matrix"].index.tolist() == row_order
    assert result.tables["matrix"].columns.tolist() == column_order
    assert "row_annotations" in result.tables
    assert result.provenance.extras["cluster_rows"] is False
    assert ax.patches
    result.close()
    assert plt.fignum_exists(figure.number)
    plt.close(figure)


def test_matrixplot_orders_clusters_and_annotates_axes(
    umap,
    leiden_clustering,
    datastore,
):
    genes = [str(value) for value in datastore.RNA.feats.fetch_all("names")[:4]]
    baseline = splt.matrixplot(
        datastore,
        features=genes,
        group_by="RNA_leiden_cluster",
        show=False,
    )
    groups = baseline.tables["matrix"].columns[1:].tolist()
    baseline.close()
    feature_order = list(reversed(genes))
    group_order = list(reversed(groups))

    ordered = splt.matrixplot(
        datastore,
        features=genes,
        group_by="RNA_leiden_cluster",
        feature_order=feature_order,
        group_order=group_order,
        row_annotations={
            "panel": {
                gene: "A" if index < 2 else "B"
                for index, gene in enumerate(feature_order)
            }
        },
        column_annotations={
            "parity": {
                group: "even" if int(group) % 2 == 0 else "odd" for group in group_order
            }
        },
        show=False,
    )

    matrix = ordered.tables["matrix"]
    assert matrix["feature"].tolist() == feature_order
    assert matrix.columns[1:].tolist() == group_order
    assert {"row_annotations", "column_annotations"} <= set(ordered.tables)
    assert len(ordered.scales) == 3
    assert ordered.axes["matrixplot"].patches
    annotation_labels = {text.get_text() for text in ordered.axes["matrixplot"].texts}
    assert {"panel", "parity"} <= annotation_labels
    ordered.close()

    requested = list(reversed(genes))
    input_ordered = splt.matrixplot(
        datastore,
        features=requested,
        group_by="RNA_leiden_cluster",
        show=False,
    )
    assert input_ordered.tables["matrix"]["feature"].tolist() == requested
    input_ordered.close()

    clustered = splt.matrixplot(
        datastore,
        features=genes,
        group_by="RNA_leiden_cluster",
        cluster_features=True,
        cluster_groups=True,
        show=False,
    )
    assert clustered.provenance.extras["cluster_features"] is True
    assert clustered.provenance.extras["cluster_groups"] is True
    clustered.close()


def test_annotation_strips_work_before_tick_labels_are_created():
    import matplotlib.pyplot as plt

    from scarf.plotting._heatmap_utils import draw_annotation_strips

    figure, ax = plt.subplots()
    row_colors = pd.DataFrame({"row group": ["#111111", "#222222"]})
    column_colors = pd.DataFrame({"column group": ["#333333", "#444444"]})

    xlim, ylim = draw_annotation_strips(
        ax,
        row_colors=row_colors,
        column_colors=column_colors,
        n_rows=2,
        n_columns=2,
    )

    assert xlim[0] < -0.5
    assert ylim[1] < -0.5
    assert {text.get_text() for text in ax.texts} == {
        "row group",
        "column group",
    }
    plt.close(figure)


def test_clustermap_annotation_legend_reserves_space_with_column_tree(
    marker_search,
    datastore,
):
    baseline = splt.marker_heatmap(
        datastore,
        group_key="RNA_cluster",
        topn=2,
        cluster_columns=False,
        show=False,
    )
    groups = baseline.tables["matrix"].columns.tolist()
    baseline.close()
    levels = (
        "relative cycling share: low",
        "relative cycling share: medium",
        "relative cycling share: high",
    )
    annotation = {
        group: levels[index % len(levels)] for index, group in enumerate(groups)
    }

    result = splt.marker_heatmap(
        datastore,
        group_key="RNA_cluster",
        topn=2,
        cluster_columns=True,
        column_annotations={"parity": annotation},
        figsize=(6, 6),
        show=False,
    )

    result.figure.canvas.draw()
    renderer = result.figure.canvas.get_renderer()
    legend_box = result.figure.legends[0].get_window_extent(renderer)
    assert {text.get_text() for text in result.figure.legends[0].get_texts()} == {
        f"parity: {level}" for level in levels
    }
    figure_box = result.figure.bbox
    assert legend_box.x0 >= figure_box.x0
    assert legend_box.x1 <= figure_box.x1
    assert legend_box.y0 >= figure_box.y0
    assert legend_box.y1 <= figure_box.y1
    overlaps = [
        name
        for name, axis in result.axes.items()
        if axis.get_visible() and legend_box.overlaps(axis.get_window_extent(renderer))
    ]
    assert overlaps == []
    result.close()


def test_clustermap_annotation_legend_without_dendrogram_is_owned_and_closed(
    marker_search,
    datastore,
):
    import matplotlib.pyplot as plt

    baseline = splt.marker_heatmap(
        datastore,
        group_key="RNA_cluster",
        topn=2,
        cluster_rows=False,
        cluster_columns=False,
        show_legend=False,
        show=False,
    )
    features = baseline.tables["matrix"].index.tolist()
    baseline.close()
    annotations = {
        feature: "first" if index % 2 == 0 else "second"
        for index, feature in enumerate(features)
    }

    result = splt.marker_heatmap(
        datastore,
        group_key="RNA_cluster",
        topn=2,
        cluster_rows=False,
        cluster_columns=False,
        row_annotations={"set": annotations},
        show=False,
    )

    assert result.owns_figure is True
    assert result.figure.legends
    assert result.provenance.extras["cluster_columns"] is False
    figure_number = result.figure.number
    result.close()
    assert not plt.fignum_exists(figure_number)


def test_cluster_tree_prepares_cache_and_returns_tables(
    paris_clustering,
    datastore,
):
    prepared = datastore._prepare_cluster_tree(cluster_key="RNA_cluster")
    assert prepared["coalesced_location"] in datastore.zw
    assert nx.is_tree(prepared["graph"])

    cached = datastore._prepare_cluster_tree(cluster_key="RNA_cluster")
    cached_labels = {
        data["partition_id"]
        for _, data in cached["graph"].nodes(data=True)
        if "partition_id" in data
    }
    assert cached_labels == set(paris_clustering)

    result = splt.cluster_tree(
        datastore,
        cluster_key="RNA_cluster",
        figsize=(4, 4),
        show=False,
    )

    assert isinstance(result, splt.PlotResult)
    assert result.owns_figure is True
    assert tuple(result.figure.get_size_inches()) == (4, 4)
    assert {"nodes", "edges", "positions", "cluster_summary"} <= set(result.tables)
    assert len(result.tables["cluster_summary"]) == len(set(paris_clustering))
    assert result.provenance.extras["coalesced_location"] in datastore.zw
    result.close()


def test_pseudotime_heatmap_returns_aligned_tables(
    pseudotime_aggregation,
    datastore,
):
    result = splt.pseudotime_heatmap(
        datastore,
        cell_key="I",
        feat_key="I",
        feature_cluster_key="pseudotime_clusters",
        pseudotime_key="RNA_pseudotime",
        show_features=["Wsb1", "Rest"],
        figsize=(4, 6),
        show=False,
    )

    assert isinstance(result, splt.PlotResult)
    assert result.owns_figure is True
    assert tuple(result.figure.get_size_inches()) == (4, 6)
    assert set(result.axes) == {
        "heatmap",
        "feature_clusters",
        "colorbar",
        "pseudotime",
    }
    assert result.tables["matrix"].shape[0] == len(result.tables["features"])
    assert result.tables["matrix"].shape[1] == len(result.tables["pseudotime_bins"])
    assert len(result.tables["pseudotime"]) == result.provenance.n_cells
    assert result.legends
    assert result.scales
    result.close()


def test_pseudotime_heatmap_accepts_composable_target(
    pseudotime_aggregation,
    datastore,
):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplot_mosaic(
        [
            ["heatmap", "feature_clusters", "colorbar"],
            ["pseudotime", "pseudotime", "colorbar"],
        ],
        figsize=(6, 4),
    )
    result = splt.pseudotime_heatmap(
        datastore,
        cell_key="I",
        feat_key="I",
        feature_cluster_key="pseudotime_clusters",
        pseudotime_key="RNA_pseudotime",
        target=axes,
        show=False,
    )

    assert result.owns_figure is False
    assert result.figure is figure
    assert np.isfinite(result.tables["matrix"].to_numpy()).all()
    assert (
        result.provenance.extras["feature_order"]
        == result.tables["features"]["feature"].tolist()
    )
    result.close()
    assert plt.fignum_exists(figure.number)
    plt.close(figure)


def test_pseudotime_heatmap_rejects_malformed_artifact_link(
    pseudotime_aggregation,
    datastore,
):
    column = datastore.RNA.z["featureData/pseudotime_clusters"]
    original_ref = dict(column.attrs["source_artifact"])
    column.attrs["source_artifact"] = "broken"
    try:
        with pytest.raises(ValueError, match="malformed source artifact"):
            splt.pseudotime_heatmap(
                datastore,
                cell_key="I",
                feat_key="I",
                feature_cluster_key="pseudotime_clusters",
                pseudotime_key="RNA_pseudotime",
                show=False,
            )
    finally:
        column.attrs["source_artifact"] = original_ref


def test_pseudotime_heatmap_validates_artifact_link_identity(
    pseudotime_aggregation,
    datastore,
):
    column = datastore.RNA.z["featureData/pseudotime_clusters"]
    original_ref = dict(column.attrs["source_artifact"])
    original_source_value = column.attrs["source_value"]
    try:
        column.attrs["source_artifact"] = {"scope": "assay"}
        with pytest.raises(ValueError, match="invalid source artifact"):
            splt.pseudotime_heatmap(
                datastore,
                cell_key="I",
                feat_key="I",
                feature_cluster_key="pseudotime_clusters",
                pseudotime_key="RNA_pseudotime",
                show=False,
            )

        column.attrs["source_artifact"] = original_ref
        column.attrs["source_value"] = "not_cluster_values"
        with pytest.raises(ValueError, match="not linked to a complete"):
            splt.pseudotime_heatmap(
                datastore,
                cell_key="I",
                feat_key="I",
                feature_cluster_key="pseudotime_clusters",
                pseudotime_key="RNA_pseudotime",
                show=False,
            )
    finally:
        column.attrs["source_artifact"] = original_ref
        column.attrs["source_value"] = original_source_value


def test_pseudotime_heatmap_validates_artifact_payload(
    pseudotime_aggregation,
    datastore,
):
    from scarf.plotting.heatmaps import _prepare_pseudotime_heatmap
    from scarf.storage.artifacts import ArtifactRef, inspect_artifact

    column = datastore.RNA.z["featureData/pseudotime_clusters"]
    ref = ArtifactRef.from_dict(dict(column.attrs["source_artifact"]))
    status = inspect_artifact(datastore.zw, ref)
    group = datastore.zw[status.path]
    valid_features = np.asarray(group["valid_features"][:], dtype=bool)

    def prepare():
        return _prepare_pseudotime_heatmap(
            datastore,
            from_assay="RNA",
            cell_key="I",
            feat_key="I",
            feature_cluster_key="pseudotime_clusters",
            pseudotime_key="RNA_pseudotime",
        )

    del group["valid_features"]
    try:
        with pytest.raises(ValueError, match="is incomplete"):
            prepare()
    finally:
        group.create_array("valid_features", data=valid_features)

    del group["valid_features"]
    group.create_array("valid_features", data=valid_features[:-1])
    try:
        with pytest.raises(ValueError, match="validity mask are misaligned"):
            prepare()
    finally:
        del group["valid_features"]
        group.create_array("valid_features", data=valid_features)

    data = group["data"]
    original_value = data[0, 0]
    data[0, 0] = np.nan
    try:
        with pytest.raises(ValueError, match="contains non-finite values"):
            prepare()
    finally:
        data[0, 0] = original_value


def test_heatmap_ordering_is_stable_for_empty_explicit_and_clustered_inputs():
    empty = pd.DataFrame(dtype=np.float64)
    ordered_empty, row_linkage, column_linkage = order_heatmap(
        empty,
        row_order=None,
        column_order=None,
        cluster_rows=True,
        cluster_columns=True,
        method="average",
        metric="euclidean",
    )
    assert ordered_empty.empty
    assert row_linkage is None
    assert column_linkage is None

    matrix = pd.DataFrame(
        [
            [0.0, 0.1, 2.0],
            [0.2, np.nan, 2.2],
            [3.0, 2.9, 0.0],
        ],
        index=["r1", "r2", "r3"],
        columns=["c1", "c2", "c3"],
    )
    explicit, row_linkage, column_linkage = order_heatmap(
        matrix,
        row_order=["r3", "r1", "r2"],
        column_order=["c2", "c3", "c1"],
        cluster_rows=True,
        cluster_columns=True,
        method="average",
        metric="euclidean",
    )
    assert explicit.index.tolist() == ["r3", "r1", "r2"]
    assert explicit.columns.tolist() == ["c2", "c3", "c1"]
    assert row_linkage is None
    assert column_linkage is None

    first, first_rows, first_columns = order_heatmap(
        matrix,
        row_order=None,
        column_order=None,
        cluster_rows=True,
        cluster_columns=True,
        method="average",
        metric="euclidean",
    )
    second, second_rows, second_columns = order_heatmap(
        matrix,
        row_order=None,
        column_order=None,
        cluster_rows=True,
        cluster_columns=True,
        method="average",
        metric="euclidean",
    )
    assert first.index.tolist() == second.index.tolist()
    assert first.columns.tolist() == second.columns.tolist()
    assert first_rows is not None
    assert first_columns is not None
    np.testing.assert_allclose(first_rows, second_rows)
    np.testing.assert_allclose(first_columns, second_columns)
    pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize(
    ("row_order", "column_order", "message"),
    [
        (["r1", "r1"], None, "row_order cannot contain duplicates"),
        (None, ["c1"], "column_order must contain every observed label"),
        (
            None,
            ["c1", "c2", "unexpected"],
            "column_order must contain every observed label",
        ),
    ],
)
def test_heatmap_ordering_rejects_malformed_orders(
    row_order,
    column_order,
    message,
):
    matrix = pd.DataFrame(
        [[0.0, 1.0], [2.0, 3.0]],
        index=["r1", "r2"],
        columns=["c1", "c2"],
    )

    with pytest.raises(ValueError, match=message):
        order_heatmap(
            matrix,
            row_order=row_order,
            column_order=column_order,
            cluster_rows=False,
            cluster_columns=False,
            method="average",
            metric="euclidean",
        )


def test_heatmap_clustering_rejects_infinite_values():
    matrix = pd.DataFrame(
        [[0.0, np.inf], [1.0, 2.0], [3.0, 4.0]],
        index=["r1", "r2", "r3"],
        columns=["c1", "c2"],
    )

    with pytest.raises(ValueError, match="finite values"):
        order_heatmap(
            matrix,
            row_order=None,
            column_order=None,
            cluster_rows=True,
            cluster_columns=False,
            method="average",
            metric="euclidean",
        )


def test_heatmap_annotations_validate_empty_alignment_and_scales():
    empty = normalize_annotations(
        [],
        {"program": []},
        axis_name="row",
    )
    assert empty.shape == (0, 1)

    with pytest.raises(ValueError, match="missing labels: r2"):
        normalize_annotations(
            ["r1", "r2"],
            {"program": {"r1": "A"}},
            axis_name="row",
        )
    with pytest.raises(ValueError, match="must have 2 values"):
        normalize_annotations(
            ["r1", "r2"],
            {"program": ["A"]},
            axis_name="row",
        )

    annotations = normalize_annotations(
        ["r1", "r2"],
        {"program": ["A", "B"]},
        axis_name="row",
    )
    with pytest.raises(ValueError, match="scale 'program' is missing values: B"):
        annotation_colors(
            annotations,
            {"program": splt.CategoricalScale(order=("A",))},
        )
    with pytest.raises(KeyError, match="Category 'B' missing from palette"):
        annotation_colors(
            annotations,
            {
                "program": splt.CategoricalScale(
                    order=("A", "B"),
                    palette={"A": "#111111"},
                )
            },
        )


def test_marker_heatmap_rejects_empty_marker_groups(datastore_ephemeral):
    assay = datastore_ephemeral.RNA
    markers = (
        assay.z["markers"] if "markers" in assay.z else assay.z.create_group("markers")
    )
    slot = markers.create_group("I__empty_heatmap_groups")
    slot.create_group("0")

    with pytest.raises(ValueError, match="Marker list is empty"):
        splt.marker_heatmap(
            datastore_ephemeral,
            group_key="empty_heatmap_groups",
            show=False,
        )


def test_marker_heatmap_categorical_legend_serializes_and_preserves_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    import json

    import matplotlib.pyplot as plt

    import scarf.plotting.heatmaps as heatmap_plotting

    matrix = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]],
        index=["gene1", "gene2"],
        columns=["group1", "group2"],
    )
    monkeypatch.setattr(
        heatmap_plotting,
        "_prepare_marker_heatmap",
        lambda *_args, **_kwargs: {
            "matrix": matrix,
            "markers": pd.DataFrame(
                {
                    "group": ["group1", "group2"],
                    "rank": [1, 1],
                    "feature_index": [0, 1],
                    "score": [1.0, 0.9],
                    "feature": ["gene1", "gene2"],
                }
            ),
            "assay": "RNA",
            "cell_key": "I",
            "group_key": "cluster",
            "n_cells": 4,
        },
    )
    annotation_scale = splt.CategoricalScale(
        order=("late", "early"),
        palette={"late": "#222222", "early": "#dddddd"},
        labels={"late": "Late", "early": "Early"},
    )
    figure, ax = plt.subplots()

    result = heatmap_plotting.marker_heatmap(
        object(),
        group_key="cluster",
        cluster_rows=False,
        cluster_columns=False,
        row_annotations={
            "program": {
                "gene1": "early",
                "gene2": "late",
            }
        },
        annotation_scales={"program": annotation_scale},
        target=ax,
        show=False,
    )

    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == [
        "program: Late",
        "program: Early",
    ]
    payload = json.loads(
        result.save_provenance(tmp_path / "marker_heatmap.json").read_text()
    )
    categorical_orders = [
        scale["values"]["order"]
        for scale in payload["scales"]
        if scale["type"] == "CategoricalScale"
    ]
    assert ["late", "early"] in categorical_orders
    assert payload["tables"]["matrix"] == {
        "columns": ["group1", "group2"],
        "rows": 2,
    }

    result.close()
    assert plt.fignum_exists(figure.number)
    plt.close(figure)


def test_marker_heatmap_validates_cluster_kwargs_and_target_layout(
    monkeypatch: pytest.MonkeyPatch,
):
    import matplotlib.pyplot as plt

    import scarf.plotting.heatmaps as heatmap_plotting

    with pytest.raises(ValueError, match="clustering controls"):
        heatmap_plotting.marker_heatmap(
            object(),
            group_key="cluster",
            row_linkage=np.eye(2),
            show=False,
        )
    with pytest.raises(ValueError, match="already standardizes"):
        heatmap_plotting.marker_heatmap(
            object(),
            group_key="cluster",
            z_score=0,
            show=False,
        )

    matrix = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]],
        index=["gene1", "gene2"],
        columns=["group1", "group2"],
    )
    monkeypatch.setattr(
        heatmap_plotting,
        "_prepare_marker_heatmap",
        lambda *_args, **_kwargs: {
            "matrix": matrix,
            "markers": pd.DataFrame(),
            "assay": "RNA",
            "cell_key": "I",
            "group_key": "cluster",
            "n_cells": 4,
        },
    )
    figure, ax = plt.subplots()
    with pytest.raises(TypeError, match="Unsupported heatmap keyword"):
        heatmap_plotting.marker_heatmap(
            object(),
            group_key="cluster",
            cluster_rows=False,
            cluster_columns=False,
            target=ax,
            unsupported_option=True,
            show=False,
        )
    with pytest.raises(ValueError, match="figsize is invalid"):
        heatmap_plotting.marker_heatmap(
            object(),
            group_key="cluster",
            cluster_rows=False,
            cluster_columns=False,
            target=ax,
            figsize=(3, 3),
            show=False,
        )
    plt.close(figure)


def test_pseudotime_heatmap_validates_orders_and_target_layout(
    monkeypatch: pytest.MonkeyPatch,
):
    import matplotlib.pyplot as plt

    import scarf.plotting.heatmaps as heatmap_plotting

    prepared = {
        "matrix": np.arange(12, dtype=np.float64).reshape(3, 4),
        "feature_indices": np.array([0, 1, 2]),
        "feature_clusters": np.array(["B", "A", "B"]),
        "feature_labels": np.array(["gene1", "gene2", "gene3"]),
        "pseudotime": np.array([0.0, 0.25, 0.5, 0.75]),
        "assay": "RNA",
        "cell_key": "I",
        "feat_key": "I",
        "feature_cluster_key": "clusters",
        "pseudotime_key": "pseudotime",
        "aggregation_location": "artifact",
    }
    monkeypatch.setattr(
        heatmap_plotting,
        "_prepare_pseudotime_heatmap",
        lambda *_args, **_kwargs: prepared,
    )

    with pytest.raises(ValueError, match="feature_order cannot contain duplicates"):
        heatmap_plotting.pseudotime_heatmap(
            object(),
            cell_key="I",
            feat_key="I",
            feature_cluster_key="clusters",
            pseudotime_key="pseudotime",
            feature_order=["gene1", "gene1", "gene3"],
            show=False,
        )
    with pytest.raises(ValueError, match="contain every observed feature cluster"):
        heatmap_plotting.pseudotime_heatmap(
            object(),
            cell_key="I",
            feat_key="I",
            feature_cluster_key="clusters",
            pseudotime_key="pseudotime",
            feature_cluster_order=["A"],
            show=False,
        )

    first_figure, first_axes = plt.subplots(1, 2)
    second_figure, second_axis = plt.subplots()
    with pytest.raises(ValueError, match="target is missing axes: pseudotime"):
        heatmap_plotting.pseudotime_heatmap(
            object(),
            cell_key="I",
            feat_key="I",
            feature_cluster_key="clusters",
            pseudotime_key="pseudotime",
            target={
                "heatmap": first_axes[0],
                "feature_clusters": first_axes[1],
            },
            show=False,
        )
    with pytest.raises(ValueError, match="target axes must share a figure"):
        heatmap_plotting.pseudotime_heatmap(
            object(),
            cell_key="I",
            feat_key="I",
            feature_cluster_key="clusters",
            pseudotime_key="pseudotime",
            target={
                "heatmap": first_axes[0],
                "feature_clusters": first_axes[1],
                "pseudotime": second_axis,
            },
            show=False,
        )
    plt.close(first_figure)
    plt.close(second_figure)


def test_pseudotime_heatmap_applies_explicit_order_scales_and_target_ownership(
    monkeypatch: pytest.MonkeyPatch,
):
    import matplotlib.pyplot as plt

    import scarf.plotting.heatmaps as heatmap_plotting

    prepared = {
        "matrix": np.arange(12, dtype=np.float64).reshape(3, 4),
        "feature_indices": np.array([10, 11, 12]),
        "feature_clusters": np.array(["B", "A", "B"]),
        "feature_labels": np.array(["gene1", "gene2", "gene3"]),
        "pseudotime": np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        "assay": "RNA",
        "cell_key": "I",
        "feat_key": "I",
        "feature_cluster_key": "clusters",
        "pseudotime_key": "pseudotime",
        "aggregation_location": "artifact",
    }
    monkeypatch.setattr(
        heatmap_plotting,
        "_prepare_pseudotime_heatmap",
        lambda *_args, **_kwargs: prepared,
    )
    figure, target_axes = plt.subplots(1, 4, figsize=(8, 3))
    target = {
        "heatmap": target_axes[0],
        "feature_clusters": target_axes[1],
        "pseudotime": target_axes[2],
        "colorbar": target_axes[3],
    }

    result = heatmap_plotting.pseudotime_heatmap(
        object(),
        cell_key="I",
        feat_key="I",
        feature_cluster_key="clusters",
        pseudotime_key="pseudotime",
        feature_order=("gene3", "gene1", "gene2"),
        feature_cluster_order=("B", "A"),
        feature_cluster_scale=splt.CategoricalScale(
            order=("B", "A"),
            palette={"B": "#222222", "A": "#dddddd"},
            labels={"B": "Beta", "A": "Alpha"},
        ),
        color_scale=splt.ColorScale(
            cmap="magma",
            vmin=-1,
            vmax=12,
            vcenter=5,
        ),
        pseudotime_scale=splt.ColorScale(
            cmap="plasma",
            vmin=0,
            vmax=1,
        ),
        show_features=["GENE2", "absent"],
        target=target,
        show_legend=False,
        show=False,
    )

    assert result.owns_figure is False
    assert result.tables["matrix"].index.tolist() == ["gene3", "gene1", "gene2"]
    assert result.tables["features"]["feature_index"].tolist() == [12, 10, 11]
    assert result.tables["features"]["cluster"].tolist() == ["B", "B", "A"]
    assert target_axes[3].axison is False
    assert result.scales[1].order == ("B", "A")
    assert [tick.get_text() for tick in target_axes[0].get_yticklabels()] == ["GENE2"]
    result.close()
    assert plt.fignum_exists(figure.number)
    plt.close(figure)


def test_marker_heatmap_surfaces_missing_seaborn_without_opening_a_figure(
    monkeypatch: pytest.MonkeyPatch,
):
    import matplotlib.pyplot as plt

    import scarf.plotting.heatmaps as heatmap_plotting

    matrix = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]],
        index=["gene1", "gene2"],
        columns=["group1", "group2"],
    )
    monkeypatch.setattr(
        heatmap_plotting,
        "_prepare_marker_heatmap",
        lambda *_args, **_kwargs: {
            "matrix": matrix,
            "markers": pd.DataFrame(),
            "assay": "RNA",
            "cell_key": "I",
            "group_key": "cluster",
            "n_cells": 4,
        },
    )

    def missing_seaborn():
        raise ImportError("Scarf plotting requires seaborn")

    monkeypatch.setattr(heatmap_plotting, "require_seaborn", missing_seaborn)
    open_figures = plt.get_fignums()
    with pytest.raises(ImportError, match="requires seaborn"):
        heatmap_plotting.marker_heatmap(
            object(),
            group_key="cluster",
            cluster_rows=False,
            cluster_columns=False,
            show=False,
        )
    assert plt.get_fignums() == open_figures

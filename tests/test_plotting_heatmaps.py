import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import scarf.plotting as splt
from scarf.plotting.cluster_tree import _hierarchy_positions


def test_hierarchy_positions_are_pure_and_complete():
    graph = nx.DiGraph([(4, 2), (4, 3), (2, 0), (2, 1)])
    before = graph.copy()

    positions = _hierarchy_positions(graph, width=2.0)

    assert set(positions) == set(graph)
    assert nx.utils.graphs_equal(graph, before)
    assert positions[4][1] == 0.0
    assert positions[0][1] < positions[2][1]


def test_marker_heatmap_returns_owned_result(
    marker_search,
    datastore,
):
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
    result.close()


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

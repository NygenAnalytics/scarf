from inspect import Parameter, signature
from types import SimpleNamespace

import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import scarf.plotting as splt
from scarf.storage.artifacts import ArtifactRef
from scarf.plotting.cluster_tree import _hierarchy_positions
from scarf.plotting._heatmap_utils import (
    annotation_colors,
    normalize_annotations,
    order_heatmap,
)


def _plot_ref(kind: str, digit: str, *, assay: str | None = "RNA") -> ArtifactRef:
    return ArtifactRef(
        scope="assay" if assay is not None else "datastore",
        assay=assay,
        kind=kind,
        artifact_id=digit * 64,
    )


_MARKER_CELLS = 30
# Stored counts are normalized in float32.
_FLOAT32_RTOL = 1e-5
_MARKER_GENES = ("CD3E", "MS4A1", "LYZ", "NKG7", "GNLY", "FCGR3A")


@pytest.fixture(scope="module")
def marker_store(tmp_path_factory):
    """A small imported store with three clusters and their saved markers."""
    from tests.test_plotting_foundation import _imported_plot_store

    labels = np.arange(_MARKER_CELLS) % 3
    store, imported, counts = _imported_plot_store(
        tmp_path_factory.mktemp("marker_heatmap"),
        coordinates=np.random.default_rng(1).normal(size=(_MARKER_CELLS, 2)),
        clusters=labels,
        genes=_MARKER_GENES,
    )
    clusters = imported.clusterArtifacts["clusters"]
    features = store.select_detected_features(imported.cellSelection, min_cells=1)
    return SimpleNamespace(
        store=store,
        clusters=clusters,
        markers=store.run_marker_search(clusters, features=features),
        labels=labels,
        counts=counts,
    )


def _library_normalized(counts: np.ndarray) -> np.ndarray:
    """RNA assay normalization: counts per 1000 per cell."""
    return counts / counts.sum(axis=1, keepdims=True) * 1000.0


def _top_marker_features(data, topn: int) -> list[str]:
    """Each group's ``topn`` markers by score, ties broken by name."""
    from scarf.features.markers.table import load_marker_table

    _, slot = data.store._resolve_marker_group(data.markers)
    names = np.asarray(slot["feature_names"][:]).astype(str)
    chosen: set[int] = set()
    for group in slot.group_keys():
        table = load_marker_table(slot, slot[group], names, group_id=group)
        ranked = table.sort_values(
            ["score", "feature_name"],
            ascending=[False, True],
            kind="mergesort",
        ).head(topn)
        chosen.update(ranked["feature_index"].astype(int))
    return [str(names[index]) for index in sorted(chosen)]


def _marker_oracle(data, topn: int, *, log_transform: bool = True) -> pd.DataFrame:
    """Standardized group means of the top markers, rows in feature order."""
    features = _top_marker_features(data, topn)
    columns = [_MARKER_GENES.index(feature) for feature in features]
    values = _library_normalized(data.counts)[:, columns]
    if log_transform:
        values = np.log1p(values)
    means = pd.DataFrame(values, columns=features).groupby(data.labels).mean()
    return ((means - means.mean()) / means.std()).T


def _optimal_leaves(values: np.ndarray, method: str) -> list[int]:
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import pdist

    return leaves_list(linkage(pdist(values), method=method, optimal_ordering=True))


def test_hierarchy_positions_are_pure_and_complete():
    graph = nx.DiGraph([(4, 2), (4, 3), (2, 0), (2, 1)])
    before = graph.copy()

    positions = _hierarchy_positions(graph, width=2.0)

    assert set(positions) == set(graph)
    assert nx.utils.graphs_equal(graph, before)
    assert positions[4][1] == 0.0
    assert positions[0][1] < positions[2][1]


def test_artifact_plots_require_their_graph_and_aggregation_inputs() -> None:
    from scarf.plotting.cluster_connectivity import cluster_connectivity
    from scarf.plotting.cluster_tree import cluster_tree
    from scarf.plotting.heatmaps import pseudotime_heatmap

    # Accessor parity is pinned in test_datastore_plot_accessor.py.
    assert signature(pseudotime_heatmap).parameters["aggregation"].default is (
        Parameter.empty
    )
    for function in (cluster_connectivity, cluster_tree):
        assert signature(function).parameters["graph"].default is Parameter.empty


def test_hierarchy_positions_rejects_non_trees():
    cyclic = nx.DiGraph([(0, 1), (1, 2), (2, 0)])
    with pytest.raises(TypeError, match="not a tree"):
        _hierarchy_positions(cyclic)


def test_tree_palette_requires_complete_color_key():
    from scarf.plotting.cluster_tree import _tree_palette

    with pytest.raises(KeyError, match="missing in `color_key`"):
        _tree_palette(
            ["A", "B"],
            cmap="tab20",
            color_key={"A": "#ff0000"},
        )
    color_key = {"A": "#ff0000", "B": "#00ff00"}
    palette = _tree_palette(
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
    graph = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="connectivity_map",
        artifact_id="1" * 64,
    )
    with pytest.raises(ValueError, match="misaligned"):
        cluster_tree(
            store,
            graph=graph,
            clusters=_plot_ref("cluster_cut", "2"),
            show=False,
        )


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


def test_marker_heatmap_display_limits_do_not_change_clustering(marker_store):
    data = marker_store
    result = splt.marker_heatmap(
        data.store,
        marker=data.markers,
        topn=2,
        vmin=-0.1,
        vmax=0.1,
        show=False,
    )

    oracle = _marker_oracle(data, topn=2)
    matrix = result.tables["matrix"]
    # The standardized values are returned unclipped.
    np.testing.assert_allclose(
        matrix, oracle.loc[matrix.index, matrix.columns], rtol=_FLOAT32_RTOL
    )
    assert np.abs(matrix.to_numpy()).max() > 0.1
    mesh = result.axes["heatmap"].collections[0]
    assert (mesh.norm.vmin, mesh.norm.vmax) == (-0.1, 0.1)
    # Ward clustering of the unclipped values decides both dendrogram orders.
    rows = _optimal_leaves(oracle.to_numpy(), "ward")
    columns = _optimal_leaves(oracle.to_numpy().T, "ward")
    assert result.provenance.extras["row_order"] == [oracle.index[i] for i in rows]
    assert result.provenance.extras["column_order"] == [
        oracle.columns[i] for i in columns
    ]
    result.close()


def test_marker_heatmap_returns_owned_result(marker_store):
    import matplotlib.pyplot as plt

    data = marker_store
    result = splt.marker_heatmap(
        data.store,
        marker=data.markers,
        topn=2,
        figsize=(4, 6),
        show=False,
    )

    assert isinstance(result, splt.PlotResult)
    assert result.owns_figure is True
    assert tuple(result.figure.get_size_inches()) == (4, 6)
    assert {"heatmap", "row_dendrogram", "column_dendrogram", "colorbar"} <= set(
        result.axes
    )
    matrix = result.tables["matrix"]
    oracle = _marker_oracle(data, topn=2)
    np.testing.assert_allclose(
        matrix, oracle.loc[matrix.index, matrix.columns], rtol=_FLOAT32_RTOL
    )
    heatmap = result.axes["heatmap"]
    mesh = heatmap.collections[0]
    np.testing.assert_allclose(np.asarray(mesh.get_array()), matrix.to_numpy())
    assert (mesh.norm.vmin, mesh.norm.vmax) == (-1.0, 2.0)
    assert [text.get_text() for text in heatmap.get_yticklabels()] == list(matrix.index)
    assert [text.get_text() for text in heatmap.get_xticklabels()] == [
        str(group) for group in matrix.columns
    ]
    assert result.axes["colorbar"].get_ylabel() == "standardized expression"
    markers = result.tables["markers"]
    assert sorted(markers["feature"].unique()) == sorted(matrix.index)
    assert markers.groupby("group")["rank"].agg(list).tolist() == [[1, 2]] * 3
    assert result.legends[0].extras == {"vmin": -1.0, "vmax": 2.0}
    assert result.provenance.notes == ("marker_heatmap", "clustered")
    assert result.provenance.n_cells == _MARKER_CELLS
    figure_number = result.figure.number
    assert plt.fignum_exists(figure_number)
    result.close()
    assert not plt.fignum_exists(figure_number)


def test_marker_heatmap_selects_features_by_named_score(marker_store):
    import matplotlib.pyplot as plt

    from scarf.features.markers.table import load_marker_table

    data = marker_store
    _, marker_slot = data.store._resolve_marker_group(data.markers)
    feature_names = np.asarray(marker_slot["feature_names"][:])
    expected_by_group: dict[str, list[str]] = {}
    for group_name in marker_slot.group_keys():
        markers = load_marker_table(
            marker_slot,
            marker_slot[group_name],
            feature_names,
            group_id=group_name,
        )
        ranked = markers.sort_values(
            ["score", "feature_name"],
            ascending=[False, True],
            kind="mergesort",
        ).head(2)
        expected_by_group[group_name] = ranked["feature_name"].astype(str).tolist()

    figure, ax = plt.subplots(figsize=(2, 2))
    result = splt.marker_heatmap(
        data.store,
        marker=data.markers,
        topn=2,
        cluster_rows=False,
        cluster_columns=False,
        target=ax,
        show=False,
    )
    selected = result.tables["markers"]
    assert set(selected["group"].astype(str)) == set(expected_by_group)
    for group_name, expected_names in expected_by_group.items():
        got = (
            selected.loc[selected["group"].astype(str) == str(group_name)]
            .sort_values("rank")["feature"]
            .astype(str)
            .tolist()
        )
        assert got == expected_names
    result.close()
    plt.close(figure)


def test_marker_heatmap_propagates_marker_metadata_errors(marker_store):
    data = marker_store
    _, marker_slot = data.store._resolve_marker_group(data.markers)
    original_method = marker_slot.attrs["method"]
    marker_slot.attrs["method"] = "ttest"
    try:
        with pytest.raises(ValueError, match="Canonical marker metadata 'method'"):
            splt.marker_heatmap(
                data.store,
                marker=data.markers,
                topn=2,
                show=False,
            )
    finally:
        marker_slot.attrs["method"] = original_method


def test_marker_heatmap_requires_an_explicit_marker_artifact():
    class UntouchedStore:
        def __getattr__(self, name):
            raise AssertionError(f"marker validation read store.{name}")

    with pytest.raises(TypeError, match="^marker must be an ArtifactRef$"):
        splt.marker_heatmap(
            UntouchedStore(),
            marker="legacy_marker",
            topn=1,
            cluster_rows=False,
            cluster_columns=False,
            show=False,
        )


def test_marker_heatmap_accepts_explicit_order_annotations_and_target(marker_store):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    data = marker_store
    oracle = _marker_oracle(data, topn=2)
    row_order = list(reversed(oracle.index))
    column_order = list(reversed(oracle.columns))
    row_annotation = {
        feature: "first" if index < len(row_order) / 2 else "second"
        for index, feature in enumerate(row_order)
    }
    palette = {"first": "#111111", "second": "#eeeeee"}
    figure, ax = plt.subplots(figsize=(4, 4))

    result = splt.marker_heatmap(
        data.store,
        marker=data.markers,
        topn=2,
        row_order=row_order,
        column_order=column_order,
        cluster_rows=False,
        cluster_columns=False,
        row_annotations={"marker set": row_annotation},
        column_annotations={"kind": {group: "cluster" for group in column_order}},
        annotation_scales={
            "marker set": splt.CategoricalScale(
                order=("first", "second"), palette=palette
            )
        },
        target=ax,
        show_legend=False,
        show=False,
    )

    assert result.owns_figure is False
    matrix = result.tables["matrix"]
    assert matrix.index.tolist() == row_order
    assert matrix.columns.tolist() == column_order
    # Tables name their axes like matrixplot's; the drawn axes stay unlabelled.
    assert (matrix.index.name, matrix.columns.name) == ("feature", "group")
    assert result.tables["row_annotations"].index.name == "feature"
    assert result.tables["column_annotations"].index.name == "group"
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("", "")
    np.testing.assert_allclose(
        ax.images[0].get_array(),
        oracle.loc[row_order, column_order].to_numpy(),
        rtol=_FLOAT32_RTOL,
    )
    assert [text.get_text() for text in ax.get_yticklabels()] == row_order
    assert [text.get_text() for text in ax.get_xticklabels()] == [
        str(group) for group in column_order
    ]
    # One unit-tall strip patch per row carries its annotation color.
    strips = [patch for patch in ax.patches if patch.get_height() == 1]
    assert [to_hex(patch.get_facecolor()) for patch in strips] == [
        palette[row_annotation[feature]] for feature in row_order
    ]
    assert result.tables["row_annotations"]["marker set"].tolist() == [
        row_annotation[feature] for feature in row_order
    ]
    assert result.provenance.extras["cluster_rows"] is False
    result.close()
    assert plt.fignum_exists(figure.number)
    plt.close(figure)


def test_matrixplot_orders_clusters_and_annotates_axes(marker_store):
    from matplotlib.colors import to_hex

    data = marker_store
    genes = ["CD3E", "MS4A1", "LYZ", "NKG7"]
    columns = [_MARKER_GENES.index(gene) for gene in genes]
    means = (
        pd.DataFrame(_library_normalized(data.counts)[:, columns], columns=genes)
        .groupby(data.labels)
        .mean()
        .T
    )
    feature_order = list(reversed(genes))
    # Summary plots label groups by their text.
    group_order = ["2", "1", "0"]
    panel = {
        gene: "A" if index < 2 else "B" for index, gene in enumerate(feature_order)
    }
    parity = {group: "even" if int(group) % 2 == 0 else "odd" for group in group_order}
    panel_palette = {"A": "#111111", "B": "#222222"}
    parity_palette = {"even": "#333333", "odd": "#444444"}

    ordered = splt.matrixplot(
        data.store,
        features=genes,
        groups=data.clusters,
        feature_order=feature_order,
        group_order=group_order,
        row_annotations={"panel": panel},
        column_annotations={"parity": parity},
        annotation_scales={
            "panel": splt.CategoricalScale(order=("A", "B"), palette=panel_palette),
            "parity": splt.CategoricalScale(
                order=("even", "odd"), palette=parity_palette
            ),
        },
        show=False,
    )

    matrix = ordered.tables["matrix"]
    assert matrix.index.tolist() == feature_order
    assert matrix.columns.tolist() == group_order
    expected = means.loc[feature_order, [2, 1, 0]].to_numpy()
    np.testing.assert_allclose(
        matrix.to_numpy(dtype=float), expected, rtol=_FLOAT32_RTOL
    )
    axis = ordered.axes["matrixplot"]
    np.testing.assert_allclose(axis.images[0].get_array(), expected, rtol=_FLOAT32_RTOL)
    assert [text.get_text() for text in axis.get_yticklabels()] == feature_order
    assert [text.get_text() for text in axis.get_xticklabels()] == ["2", "1", "0"]
    row_strips = [patch for patch in axis.patches if patch.get_height() == 1]
    column_strips = [patch for patch in axis.patches if patch.get_width() == 1]
    assert [to_hex(patch.get_facecolor()) for patch in row_strips] == [
        panel_palette[panel[gene]] for gene in feature_order
    ]
    assert [to_hex(patch.get_facecolor()) for patch in column_strips] == [
        parity_palette[parity[group]] for group in group_order
    ]
    assert len(ordered.scales) == 3
    annotation_labels = {text.get_text() for text in axis.texts}
    assert {"panel", "parity"} <= annotation_labels
    ordered.close()

    requested = list(reversed(genes))
    input_ordered = splt.matrixplot(
        data.store,
        features=requested,
        groups=data.clusters,
        show=False,
    )
    assert input_ordered.tables["matrix"].index.tolist() == requested
    assert input_ordered.tables["matrix"].columns.tolist() == ["0", "1", "2"]
    input_ordered.close()

    clustered = splt.matrixplot(
        data.store,
        features=genes,
        groups=data.clusters,
        cluster_features=True,
        cluster_groups=True,
        show=False,
    )
    rows = _optimal_leaves(means.to_numpy(), "average")
    clustered_columns = _optimal_leaves(means.to_numpy().T, "average")
    assert clustered.tables["matrix"].index.tolist() == [genes[i] for i in rows]
    assert clustered.tables["matrix"].columns.tolist() == [
        str(means.columns[i]) for i in clustered_columns
    ]
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


def test_clustermap_annotation_legend_reserves_space_with_column_tree(marker_store):
    data = marker_store
    levels = (
        "relative cycling share: low",
        "relative cycling share: medium",
        "relative cycling share: high",
    )
    annotation = {group: levels[group] for group in range(3)}

    result = splt.marker_heatmap(
        data.store,
        marker=data.markers,
        topn=2,
        cluster_columns=True,
        column_annotations={"parity": annotation},
        figsize=(6, 6),
        show=False,
    )

    result.figure.canvas.draw()
    renderer = result.figure.canvas.get_renderer()
    legend_box = result.figure.legends[0].get_window_extent(renderer)
    # Annotation values without an explicit order sort naturally.
    assert [text.get_text() for text in result.figure.legends[0].get_texts()] == [
        f"parity: {level}" for level in sorted(levels)
    ]
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
    marker_store,
):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    data = marker_store
    features = _top_marker_features(data, 2)
    annotations = {
        feature: "first" if index % 2 == 0 else "second"
        for index, feature in enumerate(features)
    }
    palette = {"first": "#123456", "second": "#abcdef"}

    result = splt.marker_heatmap(
        data.store,
        marker=data.markers,
        topn=2,
        cluster_rows=False,
        cluster_columns=False,
        row_annotations={"set": annotations},
        annotation_scales={
            "set": splt.CategoricalScale(order=("first", "second"), palette=palette)
        },
        show=False,
    )

    assert result.owns_figure is True
    (legend,) = result.figure.legends
    assert [text.get_text() for text in legend.get_texts()] == [
        "set: first",
        "set: second",
    ]
    assert [
        to_hex(handle.get_markerfacecolor()) for handle in legend.legend_handles
    ] == ["#123456", "#abcdef"]
    assert result.tables["matrix"].index.tolist() == features
    assert result.provenance.extras["cluster_columns"] is False
    figure_number = result.figure.number
    result.close()
    assert not plt.fignum_exists(figure_number)


def test_cluster_tree_prepares_cache_and_returns_tables(
    paris_clustering,
    connectivity_graph,
    datastore,
):
    prepared = datastore._prepare_cluster_tree(
        graph=connectivity_graph,
        clusters=paris_clustering,
    )
    assert prepared["coalesced_location"] in datastore.zw
    assert nx.is_tree(prepared["graph"])

    cached = datastore._prepare_cluster_tree(
        graph=connectivity_graph,
        clusters=paris_clustering,
    )
    cached_labels = {
        data["partition_id"]
        for _, data in cached["graph"].nodes(data=True)
        if "partition_id" in data
    }
    assert cached_labels == set(
        datastore.load_paris_clustering(paris_clustering).labels
    )

    result = splt.cluster_tree(
        datastore,
        graph=connectivity_graph,
        clusters=paris_clustering,
        figsize=(4, 4),
        show=False,
    )

    assert isinstance(result, splt.PlotResult)
    assert result.owns_figure is True
    assert tuple(result.figure.get_size_inches()) == (4, 4)
    assert {"nodes", "edges", "positions", "cluster_summary"} <= set(result.tables)
    assert len(result.tables["cluster_summary"]) == len(
        set(datastore.load_paris_clustering(paris_clustering).labels)
    )
    assert result.provenance.extras["coalesced_location"] in datastore.zw
    result.close()


def _saved_pseudotime(datastore, aggregation):
    """The saved aggregation rows by feature index and the valid scores.

    Frozen feature names can repeat, so rows are matched by feature index.
    """
    loaded = datastore.load_pseudotime_aggregation(aggregation)
    scoring = datastore.load_pseudotime_scoring(loaded.pseudotime)
    rows = pd.DataFrame(
        np.asarray(loaded.data[:]),
        index=np.asarray(loaded.feature_indices, dtype=np.int64),
    )
    return rows, np.asarray(scoring.values[scoring.valid], dtype=np.float64)


def _plotted_rows(saved_rows, result):
    return saved_rows.loc[result.tables["features"]["feature_index"]].to_numpy()


def test_pseudotime_heatmap_returns_aligned_tables(
    pseudotime_aggregation,
    datastore,
):
    result = splt.pseudotime_heatmap(
        datastore,
        aggregation=pseudotime_aggregation,
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
    saved_rows, scores = _saved_pseudotime(datastore, pseudotime_aggregation)
    matrix = result.tables["matrix"]
    np.testing.assert_array_equal(matrix.to_numpy(), _plotted_rows(saved_rows, result))
    np.testing.assert_array_equal(
        result.axes["heatmap"].images[0].get_array(), matrix.to_numpy()
    )
    clusters = result.tables["features"]["cluster"].to_numpy()
    # Feature clusters form contiguous, ascending blocks.
    assert (np.diff(clusters) >= 0).all()
    np.testing.assert_array_equal(result.tables["pseudotime"]["pseudotime"], scores)
    expected_bins = [
        values.mean() for values in np.array_split(np.sort(scores), matrix.shape[1])
    ]
    np.testing.assert_allclose(
        result.tables["pseudotime_bins"]["pseudotime"], expected_bins
    )
    labels = matrix.index.str.lower().tolist()
    shown = [name for name in ("Wsb1", "Rest") if name.lower() in labels]
    heatmap = result.axes["heatmap"]
    assert sorted(text.get_text() for text in heatmap.get_yticklabels()) == sorted(
        shown
    )
    assert sorted(heatmap.get_yticks()) == sorted(
        labels.index(name.lower()) for name in shown
    )
    assert result.provenance.n_cells == len(scores)
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
        aggregation=pseudotime_aggregation,
        target=axes,
        show=False,
    )

    assert result.owns_figure is False
    assert result.figure is figure
    assert all(result.axes[name] is axes[name] for name in axes)
    saved_rows, _ = _saved_pseudotime(datastore, pseudotime_aggregation)
    matrix = result.tables["matrix"]
    np.testing.assert_array_equal(matrix.to_numpy(), _plotted_rows(saved_rows, result))
    np.testing.assert_array_equal(
        axes["heatmap"].images[0].get_array(), matrix.to_numpy()
    )
    np.testing.assert_allclose(
        axes["pseudotime"].images[0].get_array(),
        [result.tables["pseudotime_bins"]["pseudotime"].to_numpy()],
    )
    result.close()
    assert plt.fignum_exists(figure.number)
    plt.close(figure)


def test_pseudotime_heatmap_requires_an_explicit_aggregation(marker_store):
    with pytest.raises(TypeError, match="aggregation must be an ArtifactRef"):
        splt.pseudotime_heatmap(marker_store.store, aggregation="latest", show=False)
    missing = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="pseudotime_aggregation",
        artifact_id="f" * 64,
    )
    with pytest.raises(ValueError, match="unavailable or invalid"):
        splt.pseudotime_heatmap(marker_store.store, aggregation=missing, show=False)


def test_pseudotime_heatmap_is_independent_of_the_live_cell_selection(
    pseudotime_aggregation,
    datastore,
):
    baseline = splt.pseudotime_heatmap(
        datastore,
        aggregation=pseudotime_aggregation,
        show=False,
    )
    baseline.close()
    live = datastore.zw["cellData/I"]
    original = np.asarray(live[:], dtype=bool)
    try:
        live[:] = np.zeros(live.shape, dtype=bool)
        result = splt.pseudotime_heatmap(
            datastore,
            aggregation=pseudotime_aggregation,
            show=False,
        )
        result.close()
    finally:
        live[:] = original

    for name in ("matrix", "features", "pseudotime", "pseudotime_bins"):
        pd.testing.assert_frame_equal(result.tables[name], baseline.tables[name])
    assert result.provenance.n_cells == baseline.provenance.n_cells > 0


def test_pseudotime_heatmap_uses_frozen_feature_names(
    pseudotime_aggregation,
    datastore,
):
    from scarf.plotting.heatmaps import _prepare_pseudotime_heatmap
    from scarf.storage.artifacts import artifact_path, fingerprint_stored_strings

    loaded = datastore.load_pseudotime_aggregation(pseudotime_aggregation)
    group = datastore.zw[artifact_path(pseudotime_aggregation)]
    inputs = datastore.inspect_artifact(pseudotime_aggregation).inputs or {}
    assert inputs["ordered_feature_ids_fingerprint"] == fingerprint_stored_strings(
        group["feature_ids"]
    )
    assert inputs["ordered_feature_names_fingerprint"] == (
        fingerprint_stored_strings(group["feature_names"])
    )
    expected = np.asarray(loaded.feature_names).copy()
    live_names = datastore.RNA.feats._get_array("names")
    original = np.asarray(live_names[:]).copy()
    try:
        renamed = original.astype(str)
        renamed[:] = [f"renamed_{index}" for index in range(len(renamed))]
        live_names[:] = renamed
        prepared = _prepare_pseudotime_heatmap(
            datastore,
            aggregation=pseudotime_aggregation,
        )
        order = np.argsort(np.asarray(loaded.feature_clusters))
        np.testing.assert_array_equal(prepared["feature_labels"], expected[order])
    finally:
        live_names[:] = original


def test_pseudotime_heatmap_validates_artifact_payload(
    pseudotime_aggregation,
    datastore,
):
    from scarf.plotting.heatmaps import _prepare_pseudotime_heatmap
    from scarf.storage.artifacts import artifact_path

    group = datastore.zw[artifact_path(pseudotime_aggregation)]

    def prepare():
        return _prepare_pseudotime_heatmap(
            datastore,
            aggregation=pseudotime_aggregation,
        )

    data = group["data"]
    valid_row = int(np.flatnonzero(np.asarray(group["valid_features"][:]))[0])
    original_value = data[valid_row, 0]
    data[valid_row, 0] = np.nan
    try:
        with pytest.raises(ValueError, match="payload is invalid"):
            prepare()
    finally:
        data[valid_row, 0] = original_value


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


def test_heatmap_clustering_orders_constant_rows_under_correlation():
    matrix = pd.DataFrame(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [3.0, 1.0, 2.0]],
        index=["unexpressed", "g1", "g2"],
        columns=["a", "b", "c"],
    )
    for metric in ("correlation", "cosine"):
        ordered, row_linkage, _ = order_heatmap(
            matrix,
            row_order=None,
            column_order=None,
            cluster_rows=True,
            cluster_columns=False,
            method="average",
            metric=metric,
        )
        assert row_linkage is not None
        assert set(ordered.index) == set(matrix.index)


class _UnreadableStore:
    """A store whose every read fails, so a check must run before any read."""

    def __getattr__(self, name):
        raise AssertionError(f"the store was read through {name!r}")


# The message names the remedy for other metrics.
_WARD_CORRELATION = (
    "ward",
    "correlation",
    "cluster_method='ward' requires cluster_metric='euclidean'.*"
    "Use cluster_method='average' or 'complete'",
)


@pytest.mark.parametrize(
    ("method", "metric", "message"),
    [
        _WARD_CORRELATION,
        ("centroid", "cosine", "cluster_method='centroid' requires"),
        ("median", "cityblock", "cluster_method='median' requires"),
        ("wards", "euclidean", "cluster_method must be one of"),
    ],
)
def test_heatmap_ordering_rejects_invalid_linkage(method, metric, message):
    matrix = pd.DataFrame(
        [[0.0, 1.0, 2.0], [2.0, 3.0, 1.0], [1.0, 0.5, 4.0]],
        index=["r1", "r2", "r3"],
        columns=["c1", "c2", "c3"],
    )

    with pytest.raises(ValueError, match=message):
        order_heatmap(
            matrix,
            row_order=None,
            column_order=None,
            cluster_rows=True,
            cluster_columns=False,
            method=method,
            metric=metric,
        )


@pytest.mark.parametrize(
    ("method", "metric"),
    [("ward", "euclidean"), ("complete", "correlation"), ("average", "cosine")],
)
def test_heatmap_ordering_accepts_valid_linkage(method, metric):
    matrix = pd.DataFrame(
        [[0.0, 1.0, 2.0], [2.0, 3.0, 1.0], [1.0, 0.5, 4.0]],
        index=["r1", "r2", "r3"],
        columns=["c1", "c2", "c3"],
    )

    ordered, row_linkage, _ = order_heatmap(
        matrix,
        row_order=None,
        column_order=None,
        cluster_rows=True,
        cluster_columns=False,
        method=method,
        metric=metric,
    )

    assert row_linkage is not None
    assert sorted(ordered.index) == ["r1", "r2", "r3"]


@pytest.mark.parametrize("clustered", [True, False])
def test_heatmaps_reject_invalid_linkage_before_reading_the_store(clustered):
    method, metric, message = _WARD_CORRELATION
    # An axis that is not clustered still records the pair, so it is checked.
    with pytest.raises(ValueError, match=message):
        splt.matrixplot(
            _UnreadableStore(),
            features=["CD3E", "LYZ"],
            group_by="clusters",
            cluster_features=clustered,
            cluster_groups=clustered,
            cluster_method=method,
            cluster_metric=metric,
            show=False,
        )
    with pytest.raises(ValueError, match=message):
        splt.marker_heatmap(
            _UnreadableStore(),
            marker=_plot_ref("marker_table", "4"),
            cluster_rows=clustered,
            cluster_columns=clustered,
            cluster_method=method,
            cluster_metric=metric,
            show=False,
        )
    # The legacy seaborn keywords name the same controls.
    with pytest.raises(ValueError, match=message):
        splt.marker_heatmap(
            _UnreadableStore(),
            marker=_plot_ref("marker_table", "4"),
            row_cluster=clustered,
            col_cluster=clustered,
            method=method,
            metric=metric,
            show=False,
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
    with pytest.raises(
        ValueError,
        match=r"annotation_scales\['program'\]\.order is missing observed values: B",
    ):
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


def test_marker_heatmap_rejects_missing_and_empty_marker_tables(marker_store):
    data = marker_store
    with pytest.raises(ValueError, match="does not exist"):
        splt.marker_heatmap(
            data.store,
            marker=_plot_ref("marker_table", "e"),
            show=False,
        )
    with pytest.raises(ValueError) as raised:
        splt.marker_heatmap(data.store, marker=data.markers, topn=0, show=False)
    assert raised.value.args == ("ERROR: Marker list is empty for all the groups",)


def test_marker_heatmap_validates_log_transform_and_color_limits(marker_store):
    data = marker_store
    with pytest.raises(TypeError) as raised:
        splt.marker_heatmap(
            data.store, marker=data.markers, log_transform="yes", show=False
        )
    assert raised.value.args == ("log_transform must be a boolean or None",)

    linear = splt.marker_heatmap(
        data.store,
        marker=data.markers,
        topn=2,
        log_transform=np.bool_(False),
        cluster_rows=False,
        cluster_columns=False,
        show=False,
    )
    oracle = _marker_oracle(data, topn=2, log_transform=False)
    np.testing.assert_allclose(linear.tables["matrix"], oracle, rtol=_FLOAT32_RTOL)
    linear.close()

    marker = _plot_ref("marker_table", "3")
    with pytest.raises(NotImplementedError, match="only linear color scales"):
        splt.marker_heatmap(
            object(),
            marker=marker,
            color_scale=splt.ColorScale(scale="log"),
            show=False,
        )
    # An explicit scale's lower limit can still exceed the default upper one.
    with pytest.raises(ValueError, match="^vmax must be greater than vmin$"):
        splt.marker_heatmap(
            object(),
            marker=marker,
            color_scale=splt.ColorScale(vmin=3.0),
            show=False,
        )


def test_marker_heatmap_rejects_clusters_misaligned_with_their_selection(
    marker_store,
    monkeypatch: pytest.MonkeyPatch,
):
    import scarf.plotting.heatmaps as heatmap_plotting

    data = marker_store
    monkeypatch.setattr(
        heatmap_plotting,
        "read_stored_selection_indices",
        lambda *args, **kwargs: np.arange(_MARKER_CELLS + 1),
    )
    with pytest.raises(ValueError) as raised:
        splt.marker_heatmap(data.store, marker=data.markers, show=False)
    assert raised.value.args == (
        "Marker clusters do not align with their cell selection",
    )


def test_marker_group_means_do_not_depend_on_streamed_block_boundaries(
    marker_store,
    monkeypatch: pytest.MonkeyPatch,
):
    data = marker_store
    assay_type = type(data.store.RNA)
    original_normed = assay_type.normed

    class SingleRowBlocks:
        def __init__(self, values):
            self._values = values

        def stream_blocks(self, *args, **kwargs):
            for block in self._values.stream_blocks(*args, **kwargs):
                yield from (row[np.newaxis, :] for row in block)

    def normed_in_single_rows(self, *args, **kwargs):
        return SingleRowBlocks(original_normed(self, *args, **kwargs))

    monkeypatch.setattr(assay_type, "normed", normed_in_single_rows)
    result = splt.marker_heatmap(
        data.store,
        marker=data.markers,
        topn=2,
        cluster_rows=False,
        cluster_columns=False,
        show=False,
    )

    np.testing.assert_allclose(
        result.tables["matrix"], _marker_oracle(data, topn=2), rtol=_FLOAT32_RTOL
    )
    result.close()


def test_marker_heatmap_shows_owned_results_by_default(
    marker_store,
    monkeypatch: pytest.MonkeyPatch,
):
    shown = []
    monkeypatch.setattr(splt.PlotResult, "show", lambda result: shown.append(result))

    result = splt.marker_heatmap(
        marker_store.store,
        marker=marker_store.markers,
        topn=1,
        cluster_rows=False,
        cluster_columns=False,
    )

    assert shown == [result]
    result.close()


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
            "marker_ref": _plot_ref("marker_table", "3"),
            "clusters_ref": _plot_ref("cluster_labels", "4"),
            "cell_selection": _plot_ref("cell_selection", "5", assay=None),
            "n_cells": 4,
            "unmeasured_cells": {},
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
        marker=_plot_ref("marker_table", "3"),
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
            marker=_plot_ref("marker_table", "3"),
            row_linkage=np.eye(2),
            show=False,
        )
    with pytest.raises(ValueError, match="already standardizes"):
        heatmap_plotting.marker_heatmap(
            object(),
            marker=_plot_ref("marker_table", "3"),
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
            "marker_ref": _plot_ref("marker_table", "3"),
            "clusters_ref": _plot_ref("cluster_labels", "4"),
            "cell_selection": _plot_ref("cell_selection", "5", assay=None),
            "n_cells": 4,
        },
    )
    figure, ax = plt.subplots()
    with pytest.raises(TypeError, match="Unsupported heatmap keyword"):
        heatmap_plotting.marker_heatmap(
            object(),
            marker=_plot_ref("marker_table", "3"),
            cluster_rows=False,
            cluster_columns=False,
            target=ax,
            unsupported_option=True,
            show=False,
        )
    with pytest.raises(ValueError, match="figsize is invalid"):
        heatmap_plotting.marker_heatmap(
            object(),
            marker=_plot_ref("marker_table", "3"),
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
        "cell_selection": _plot_ref("cell_selection", "3", assay=None),
        "feature_selection": ArtifactRef(
            scope="assay",
            assay="RNA",
            kind="feature_selection",
            artifact_id="1" * 64,
        ),
        "pseudotime_ref": _plot_ref("pseudotime", "4"),
        "aggregation_ref": _plot_ref("pseudotime_aggregation", "5"),
    }
    monkeypatch.setattr(
        heatmap_plotting,
        "_prepare_pseudotime_heatmap",
        lambda *_args, **_kwargs: prepared,
    )

    with pytest.raises(ValueError, match="feature_order cannot contain duplicates"):
        heatmap_plotting.pseudotime_heatmap(
            object(),
            aggregation=_plot_ref("pseudotime_aggregation", "5"),
            feature_order=["gene1", "gene1", "gene3"],
            show=False,
        )
    with pytest.raises(ValueError, match="contain every observed feature cluster"):
        heatmap_plotting.pseudotime_heatmap(
            object(),
            aggregation=_plot_ref("pseudotime_aggregation", "5"),
            feature_cluster_order=["A"],
            show=False,
        )

    first_figure, first_axes = plt.subplots(1, 2)
    second_figure, second_axis = plt.subplots()
    with pytest.raises(ValueError, match="target is missing axes: pseudotime"):
        heatmap_plotting.pseudotime_heatmap(
            object(),
            aggregation=_plot_ref("pseudotime_aggregation", "5"),
            target={
                "heatmap": first_axes[0],
                "feature_clusters": first_axes[1],
            },
            show=False,
        )
    with pytest.raises(ValueError, match="target axes must share a figure"):
        heatmap_plotting.pseudotime_heatmap(
            object(),
            aggregation=_plot_ref("pseudotime_aggregation", "5"),
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
        "cell_selection": _plot_ref("cell_selection", "3", assay=None),
        "feature_selection": ArtifactRef(
            scope="assay",
            assay="RNA",
            kind="feature_selection",
            artifact_id="2" * 64,
        ),
        "pseudotime_ref": _plot_ref("pseudotime", "4"),
        "aggregation_ref": _plot_ref("pseudotime_aggregation", "5"),
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
        aggregation=_plot_ref("pseudotime_aggregation", "5"),
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
            "marker_ref": _plot_ref("marker_table", "3"),
            "clusters_ref": _plot_ref("cluster_labels", "4"),
            "cell_selection": _plot_ref("cell_selection", "5", assay=None),
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
            marker=_plot_ref("marker_table", "3"),
            cluster_rows=False,
            cluster_columns=False,
            show=False,
        )
    assert plt.get_fignums() == open_figures


_AGGREGATION = _plot_ref("pseudotime_aggregation", "5")
_PSEUDOTIME_SELECTION = _plot_ref("cell_selection", "3", assay=None)


def _pseudotime_store(*, scoring=None, **aggregation):
    """A store whose saved pseudotime outputs are small, explicit arrays."""
    loaded = {
        "ref": _AGGREGATION,
        "pseudotime": _plot_ref("pseudotime", "4"),
        "cell_selection": _PSEUDOTIME_SELECTION,
        "data": np.arange(6, dtype=np.float64).reshape(3, 2),
        "feature_indices": np.array([4, 5, 6]),
        "feature_clusters": np.array([1, 0, 1]),
        "feature_names": np.array(["g4", "g5", "g6"]),
        "assay": "RNA",
        "feature_selection": _plot_ref("feature_selection", "2"),
        **aggregation,
    }
    scored = {
        "cell_selection": _PSEUDOTIME_SELECTION,
        "values": np.array([0.2, np.nan, 0.6, 0.4, 0.8]),
        "valid": np.array([True, False, True, True, True]),
        **(scoring or {}),
    }
    return SimpleNamespace(
        load_pseudotime_aggregation=lambda ref: SimpleNamespace(**loaded),
        load_pseudotime_scoring=lambda ref: SimpleNamespace(**scored),
    )


def test_pseudotime_heatmap_draws_saved_rows_clusters_and_bins():
    from matplotlib.colors import to_hex

    result = splt.pseudotime_heatmap(
        _pseudotime_store(),
        aggregation=_AGGREGATION,
        feature_cluster_scale=splt.CategoricalScale(
            order=(0, 1),
            palette={0: "#111111", 1: "#eeeeee"},
        ),
        show_features=["G6"],
        show=False,
    )

    # Rows sort by feature cluster: g5 (cluster 0), then g4 and g6 (cluster 1).
    matrix = result.tables["matrix"]
    assert matrix.index.tolist() == ["g5", "g4", "g6"]
    np.testing.assert_array_equal(matrix.to_numpy(), [[2, 3], [0, 1], [4, 5]])
    np.testing.assert_array_equal(
        result.axes["heatmap"].images[0].get_array(), matrix.to_numpy()
    )
    features = result.tables["features"]
    assert features["feature_index"].tolist() == [5, 4, 6]
    assert features["cluster"].tolist() == [0, 1, 1]
    # The four valid scores split into two bins: [0.2, 0.4] and [0.6, 0.8].
    np.testing.assert_allclose(
        result.tables["pseudotime"]["pseudotime"], [0.2, 0.6, 0.4, 0.8]
    )
    np.testing.assert_allclose(
        result.tables["pseudotime_bins"]["pseudotime"], [0.3, 0.7]
    )
    np.testing.assert_allclose(
        result.axes["pseudotime"].images[0].get_array(), [[0.3, 0.7]]
    )
    cluster_image = result.axes["feature_clusters"].images[0]
    np.testing.assert_array_equal(cluster_image.get_array(), [[0], [1], [1]])
    assert [to_hex(color) for color in cluster_image.cmap.colors] == [
        "#111111",
        "#eeeeee",
    ]
    # Requested feature labels match case-insensitively at their rows.
    heatmap = result.axes["heatmap"]
    assert [text.get_text() for text in heatmap.get_yticklabels()] == ["G6"]
    np.testing.assert_array_equal(heatmap.get_yticks(), [2])
    assert result.provenance.n_cells == 4
    result.close()


@pytest.mark.parametrize(
    ("store_changes", "message"),
    [
        (
            {"ref": _plot_ref("pseudotime_aggregation", "6")},
            "Loaded pseudotime aggregation does not match the request",
        ),
        (
            {
                "scoring": {
                    "cell_selection": _plot_ref("cell_selection", "9", assay=None)
                }
            },
            "Pseudotime aggregation and scoring use different cell selections",
        ),
        (
            {"data": np.arange(4, dtype=np.float64).reshape(2, 2)},
            "Aggregated feature matrix and indices are misaligned",
        ),
        (
            {"data": np.arange(3, dtype=np.float64)},
            "Aggregated feature matrix and indices are misaligned",
        ),
        (
            {"feature_clusters": np.array([1, 0])},
            "Aggregated feature clusters and indices are misaligned",
        ),
        (
            {"data": np.array([[0.0, 1.0], [np.inf, 2.0], [3.0, 4.0]])},
            "Aggregated feature matrix contains non-finite values",
        ),
        (
            {"feature_names": np.array(["g4", "g5"])},
            "Frozen feature labels do not align with aggregation rows",
        ),
        (
            {"scoring": {"valid": np.zeros(5, dtype=bool)}},
            "Pseudotime artifact has no finite scored cells",
        ),
        (
            {"scoring": {"valid": np.ones(5, dtype=bool)}},
            "Pseudotime artifact has no finite scored cells",
        ),
    ],
)
def test_pseudotime_heatmap_rejects_inconsistent_saved_outputs(store_changes, message):
    with pytest.raises(ValueError) as raised:
        splt.pseudotime_heatmap(
            _pseudotime_store(**store_changes), aggregation=_AGGREGATION, show=False
        )

    assert raised.value.args == (message,)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        (
            {"color_scale": splt.ColorScale(scale="log")},
            NotImplementedError,
            "pseudotime_heatmap supports linear scales",
        ),
        (
            {"color_scale": splt.ColorScale(vmin=3.0)},
            ValueError,
            "vmax must be greater than vmin",
        ),
        (
            {"pseudotime_scale": splt.ColorScale(scale="symlog")},
            NotImplementedError,
            "pseudotime annotations support linear scales",
        ),
        (
            {"feature_cluster_order": [0, 0, 1]},
            ValueError,
            "feature_cluster_order cannot contain duplicates",
        ),
        (
            {"feature_order": ["g4", "g5", "other"]},
            ValueError,
            "feature_order must contain every plotted feature",
        ),
    ],
)
def test_pseudotime_heatmap_rejects_invalid_display_options(kwargs, error, message):
    with pytest.raises(error) as raised:
        splt.pseudotime_heatmap(
            _pseudotime_store(), aggregation=_AGGREGATION, show=False, **kwargs
        )

    assert raised.value.args == (message,)


def test_pseudotime_heatmap_shows_owned_results_by_default(monkeypatch):
    shown = []
    monkeypatch.setattr(splt.PlotResult, "show", lambda result: shown.append(result))

    result = splt.pseudotime_heatmap(_pseudotime_store(), aggregation=_AGGREGATION)

    assert shown == [result]
    result.close()


def test_annotation_legend_handles_name_values_and_skip_unresolved_scales():
    import matplotlib as mpl

    from scarf.plotting._heatmap_utils import annotation_legend_handles

    handles = annotation_legend_handles(
        mpl,
        ["unresolved", "program"],
        [
            splt.CategoricalScale(),
            splt.CategoricalScale(
                order=("late", "early"),
                palette={"late": "#222222", "early": "#dddddd"},
                labels={"late": "Late"},
            ),
        ],
    )

    assert [handle.get_label() for handle in handles] == [
        "program: Late",
        "program: early",
    ]
    assert [handle.get_markerfacecolor() for handle in handles] == [
        "#222222",
        "#dddddd",
    ]


def test_pseudotime_feature_order_refuses_to_drop_rows_with_repeated_names():
    store = _pseudotime_store(feature_names=np.array(["g4", "g4", "g6"]))

    # Without an explicit order every row is plotted, repeated names included.
    result = splt.pseudotime_heatmap(store, aggregation=_AGGREGATION, show=False)
    assert len(result.tables["matrix"]) == 3
    result.close()
    # An order by name cannot place the two g4 rows, so it is rejected rather
    # than dropping one of them.
    with pytest.raises(
        ValueError,
        match=r"^feature_order cannot order features whose names repeat: 'g4'$",
    ):
        splt.pseudotime_heatmap(
            store,
            aggregation=_AGGREGATION,
            feature_order=["g6", "g4"],
            show=False,
        )


def test_marker_heatmap_clusters_markers_that_share_a_feature_name(tmp_path):
    from tests.test_plotting_foundation import _imported_plot_store

    labels = np.arange(30) % 3
    store, imported, counts = _imported_plot_store(
        tmp_path,
        coordinates=np.random.default_rng(2).normal(size=(30, 2)),
        clusters=labels,
        genes=("CD3E", "LYZ", "CD3E", "NKG7"),
    )
    clusters = imported.clusterArtifacts["clusters"]
    features = store.select_detected_features(imported.cellSelection, min_cells=1)
    markers = store.run_marker_search(clusters, features=features)

    result = splt.marker_heatmap(store, marker=markers, topn=4, show=False)

    # A repeated name also shows its feature ID; other names are unchanged.
    rows = ["CD3E (f0)", "LYZ", "CD3E (f2)", "NKG7"]
    values = np.log1p(_library_normalized(counts.astype(np.float64)))
    means = pd.DataFrame(values, columns=rows).groupby(labels).mean()
    oracle = ((means - means.mean()) / means.std()).T
    matrix = result.tables["matrix"]
    assert sorted(matrix.index) == sorted(rows)
    np.testing.assert_allclose(
        matrix, oracle.loc[matrix.index, matrix.columns], rtol=_FLOAT32_RTOL
    )
    assert set(result.tables["markers"]["feature"]) == set(rows)
    shown = [label.get_text() for label in result.axes["heatmap"].get_yticklabels()]
    assert sorted(shown) == sorted(rows)
    result.close()


def test_marker_heatmap_rejects_a_negative_topn(marker_store):
    with pytest.raises(ValueError, match="^topn must be at least 0$"):
        splt.marker_heatmap(
            marker_store.store,
            marker=marker_store.markers,
            topn=-1,
            cluster_rows=False,
            cluster_columns=False,
            show=False,
        )

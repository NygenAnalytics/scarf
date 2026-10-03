"""Focused behavior tests for publication plotting features."""

import warnings
from importlib import import_module
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

import scarf.plotting as splt

from scarf.storage import ArtifactRef
from scarf.storage.selections import read_stored_selection_indices


def _artifact_cell_indices(store, ref: ArtifactRef) -> np.ndarray:
    status = store.inspect_artifact(ref)
    selection = ArtifactRef.from_dict(status.inputs["cell_selection"])
    return read_stored_selection_indices(
        store.zw,
        selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )


class _SyntheticCells:
    def __init__(self, **columns):
        self._columns = {name: np.asarray(values) for name, values in columns.items()}
        self.columns = tuple(self._columns)
        self.N = len(next(iter(self._columns.values())))

    def _get_array(self, column):
        return self._columns[column]

    def fetch(self, column, key="I"):
        assert key == "I"
        return self._columns[column]

    def fetch_all(self, column):
        return self._columns[column]

    def active_index(self, key="I"):
        assert key == "I"
        return np.arange(len(next(iter(self._columns.values()))))


def _synthetic_plot_store(**columns):
    return SimpleNamespace(
        cells=_SyntheticCells(**columns),
        _defaultAssay="RNA",
        zw=None,
        _stored_display_metadata=lambda _column: None,
    )


def _synthetic_stats_result(
    store,
    table,
    *,
    method="welch",
    posthoc_table=None,
    sample_by=None,
    pair_by=None,
    sample_stat="mean",
    expression_cutoff=0.0,
):
    from scarf.features.statistical import value_fingerprint
    from scarf.storage.artifacts import provenance_hash

    values = store.cells.fetch("metric")
    groups = store.cells.fetch("group")
    unique_groups = list(pd.unique(groups))
    cell_selection = store.cells.active_index("I")
    samples = store.cells.fetch(sample_by) if sample_by is not None else None
    pairs = store.cells.fetch(pair_by) if pair_by is not None else None
    return SimpleNamespace(
        method=method,
        posthoc="dunn" if posthoc_table is not None else None,
        adjustment_method="fdr_bh",
        grouping=None,
        group_field=splt.CellField("group"),
        sample_by=sample_by,
        pair_by=pair_by,
        sample_stat=sample_stat,
        expression_cutoff=expression_cutoff,
        n_groups=len(unique_groups),
        n_cells=len(values),
        tested_features=(
            provenance_hash(
                {
                    "source": "cell_metadata",
                    "column": "metric",
                    "values_fingerprint": value_fingerprint(values),
                    "missing_fingerprint": None,
                }
            ),
        ),
        value_fingerprints=(value_fingerprint(np.asarray(values, dtype=np.float64)),),
        summary_scope="sample" if sample_by is not None else "cell",
        tables={"metric": table},
        posthoc_tables=({"metric": posthoc_table} if posthoc_table is not None else {}),
        cell_selection=None,
        cell_selection_fingerprint=value_fingerprint(cell_selection),
        group_fingerprint=value_fingerprint(groups),
        group_order=tuple(sorted(unique_groups, key=str)),
        normalization={},
        normalization_method=None,
        size_factor=None,
        source_assays=(None,),
        sample_fingerprint=(
            value_fingerprint(samples) if samples is not None else None
        ),
        pair_fingerprint=value_fingerprint(pairs) if pairs is not None else None,
        artifact=None,
    )


def test_stored_display_metadata_does_not_hide_malformed_stores():
    from scarf.plotting._display import stored_display_metadata

    class MissingPlotStore:
        pass

    class MalformedPlotStore:
        zw = {}

    assert stored_display_metadata(MissingPlotStore(), "group") is None
    with pytest.raises(KeyError, match="cellData"):
        stored_display_metadata(MalformedPlotStore(), "group")


def _scatter_store(n=2000, seed=0, **extra):
    rng = np.random.default_rng(seed)
    return _synthetic_plot_store(
        I=np.ones(n, dtype=bool),
        umap1=rng.normal(size=n),
        umap2=rng.normal(size=n),
        score=rng.normal(size=n),
        **extra,
    )


def test_embedding_density_highlight_and_labeled_colorbar():
    from matplotlib.contour import ContourSet

    flag = np.arange(2000) < 200
    store = _scatter_store(flag=flag)
    x = store.cells.fetch("umap1")
    y = store.cells.fetch("umap2")
    score = store.cells.fetch("score")

    result = splt.embedding(
        store,
        layout_key="umap",
        color_by="score",
        density_overlay=splt.DensityOverlay(pixels=32, levels=3, sigma=1),
        highlight=splt.Highlight(by="flag"),
        point_size_range=(2, 20),
        show_titles=False,
        show=False,
    )

    ax = result.axes["score"]
    base, *overlays, highlight = ax.collections
    assert any(isinstance(overlay, ContourSet) for overlay in overlays)
    # The highlight redraws the flagged cells at 1.5x size over dimmed cells.
    np.testing.assert_allclose(highlight.get_offsets(), np.column_stack((x, y))[flag])
    assert highlight.get_sizes()[0] == pytest.approx(1.5 * base.get_sizes()[0])
    assert base.get_alpha() == pytest.approx(0.12)
    assert result.provenance.extras["highlight"]["n_highlighted"] == 200
    assert all(
        2 <= value <= 20
        for value in result.provenance.extras["point_size_by_panel"].values()
    )
    (colorbar,) = [axis for axis in result.figure.axes if axis is not ax]
    assert colorbar.get_xlabel() == "score"
    assert colorbar.get_xlim() == pytest.approx((score.min(), score.max()))
    assert ax.get_title() == ""
    result.close()


def test_embedding_mean_contours_require_and_use_continuous_values():
    from matplotlib.contour import ContourSet

    rng = np.random.default_rng(0)
    # Two clusters: values 3 around x = -3 and values 5 around x = +3.
    x = np.concatenate((rng.normal(-3, 0.4, 100), rng.normal(3, 0.4, 100)))
    y = rng.normal(0, 0.4, 200)
    values = np.repeat([3.0, 5.0], 100)
    store = _synthetic_plot_store(
        I=np.ones(200, dtype=bool), umap1=x, umap2=y, value=values
    )

    def lowest_level_vertices(max_hotspots):
        result = splt.embedding(
            store,
            layout_key="umap",
            color_by="value",
            density_overlay=splt.DensityOverlay(
                statistic="mean",
                pixels=32,
                sigma=1.5,
                min_support=0.05,
                levels=(2.0, 4.5),
                max_hotspots=max_hotspots,
            ),
            show=False,
        )
        overlay = next(
            collection
            for collection in result.axes["value"].collections
            if isinstance(collection, ContourSet)
        )
        paths = [path.vertices for path in overlay.get_paths()]
        extras = result.provenance.extras["density_overlay"]
        result.close()
        return paths, extras

    both, extras = lowest_level_vertices(None)
    assert extras["statistic"] == "mean"
    # Level 2 encloses both clusters; level 4.5 only the high-valued one.
    assert both[0][:, 0].min() < 0 < both[0][:, 0].max()
    assert both[1][:, 0].min() > 0
    strongest, extras = lowest_level_vertices(1)
    assert extras["max_hotspots"] == 1
    assert strongest[0][:, 0].min() > 0

    with pytest.raises(ValueError, match="continuous color_by"):
        splt.embedding(
            store,
            layout_key="umap",
            color_by=splt.CellField("I", kind="categorical"),
            density_overlay=splt.DensityOverlay(statistic="mean"),
            show=False,
        )
    with pytest.raises(ValueError, match="positive integer"):
        splt.DensityOverlay(max_hotspots=0)


def test_contour_hotspot_limit_keeps_the_strongest_region():
    from scipy.ndimage import label

    from scarf.plotting.embedding import _retain_strongest_hotspots

    surface = np.zeros((12, 12), dtype=np.float64)
    support = np.ones_like(surface)
    surface[1:4, 1:4] = 2.0
    surface[7:11, 7:11] = 3.0
    surface[8:10, 8:10] = 0.0
    support[7:11, 7:11] = 2.0

    filtered = _retain_strongest_hotspots(
        surface,
        support,
        level=1.0,
        max_hotspots=1,
    )
    _, n_hotspots = label(filtered >= 1.0)

    assert n_hotspots == 1
    assert np.all(filtered[1:4, 1:4] < 1.0)
    assert np.all(filtered[7:11, 7:11] >= 1.0)


def test_embedding_point_size_uses_final_panel_area():
    result = splt.embedding(
        _scatter_store(),
        layout_key="umap",
        color_by="score",
        point_size_range=(2, 20),
        show=False,
    )

    result.figure.canvas.draw()
    ax = result.axes["score"]
    bbox = ax.get_position()
    width, height = result.figure.get_size_inches()
    panel_area = float(bbox.width * width * bbox.height * height)
    # Size 16 at 1000 cells on a 3.2 inch square, scaled by panel area to the
    # 0.72 power and by sqrt(1000 / cells), within the requested range.
    expected = min(20.0, max(2.0, 16 * (panel_area / 3.2**2) ** 0.72 * 0.5**0.5))
    observed = next(iter(result.provenance.extras["point_size_by_panel"].values()))
    assert observed == pytest.approx(expected)
    np.testing.assert_allclose(ax.collections[0].get_sizes(), expected)
    result.close()


def test_embedding_validates_layout_coordinate_and_facet_inputs():
    store = _synthetic_plot_store(
        layout1=[0.0, 1.0, 2.0],
        layout2=[0.0, 1.0, 0.0],
        other1=[2.0, 1.0, 0.0],
        other2=[0.0, -1.0, 0.0],
        score=[1.0, 2.0, 3.0],
    )

    with pytest.raises(ValueError, match="at least one layout"):
        splt.embedding(store, layout_key=[], show=False)
    with pytest.raises(TypeError, match="Every layout_key entry"):
        splt.embedding(store, layout_key=["layout", 3], show=False)
    with pytest.raises(ValueError, match="must be unique"):
        splt.embedding(store, layout_key=["layout", "layout"], show=False)
    with pytest.raises(ValueError, match="color_by must contain at least one"):
        splt.embedding(
            store,
            layout_key=["layout", "other"],
            color_by=[],
            show=False,
        )
    with pytest.raises(ValueError, match="panel_keys must be non-empty"):
        splt.embedding(store, layout_key="layout", color_by=[], show=False)

    invalid = _synthetic_plot_store(
        layout1=[np.nan, np.inf],
        layout2=[np.nan, -np.inf],
    )
    with pytest.raises(ValueError, match="has no finite coordinates"):
        splt.embedding(invalid, layout_key="layout", show=False)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"rasterize_threshold": -1}, "rasterize_threshold"),
        ({"point_size_range": (0.0, 2.0)}, "point_size_range"),
        ({"point_size_range": (3.0, 2.0)}, "point_size_range"),
        ({"point_edgewidth": -0.1}, "point_edgewidth"),
        ({"point_alpha": 1.1}, "point_alpha"),
        ({"max_on_data_labels": 0}, "max_on_data_labels"),
        ({"point_size": np.nan}, "point_size must be finite"),
        ({"point_sizes": [1.0, 2.0]}, "point_sizes length"),
        ({"point_sizes": [1.0, 2.0, np.inf]}, "finite positive"),
        ({"clip_fraction": 0.5}, "clip_fraction"),
    ],
)
def test_embedding_validates_limits_and_point_sizes(options, message):
    store = _synthetic_plot_store(
        layout1=[0.0, 1.0, 2.0],
        layout2=[0.0, 1.0, 0.0],
        score=[1.0, 2.0, 3.0],
    )

    with pytest.raises(ValueError, match=message):
        splt.embedding(
            store,
            layout_key="layout",
            color_by=splt.CellField("score", kind="continuous"),
            show=False,
            **options,
        )


def test_embedding_categorical_colors_and_scatter_sizes_are_preserved():
    categories = np.asarray(["b", "a", None, "b"], dtype=object)
    sizes = np.asarray([4.0, 9.0, 16.0, 25.0])
    store = _synthetic_plot_store(
        layout1=[0.0, 1.0, 2.0, 3.0],
        layout2=[0.0, 1.0, 0.0, 1.0],
        category=categories,
    )
    scale = splt.CategoricalScale(
        order=("b", "a"),
        palette={"b": "#ff0000", "a": "#0000ff"},
        labels={"b": "Beta", "a": "Alpha"},
        missing_color="#00ff00",
        missing_label="Missing",
    )

    result = splt.embedding(
        store,
        layout_key="layout",
        color_by=splt.CellField("category", kind="categorical"),
        categorical_scale=scale,
        point_sizes=sizes,
        point_edgecolor="#111111",
        point_edgewidth=0.6,
        point_alpha=0.4,
        rasterize_threshold=4,
        show_legend=False,
        show=False,
    )

    collection = result.axes["category"].collections[0]
    np.testing.assert_array_equal(collection.get_sizes(), sizes)
    assert collection.get_alpha() == pytest.approx(0.4)
    assert collection.get_linewidths() == pytest.approx([0.6])
    assert collection.get_rasterized() is True
    expected_rgb = np.asarray(
        [
            matplotlib.colors.to_rgba("#ff0000")[:3],
            matplotlib.colors.to_rgba("#0000ff")[:3],
            matplotlib.colors.to_rgba("#00ff00")[:3],
            matplotlib.colors.to_rgba("#ff0000")[:3],
        ]
    )
    np.testing.assert_allclose(collection.get_facecolors()[:, :3], expected_rgb)
    resolved = next(
        value for value in result.scales if isinstance(value, splt.CategoricalScale)
    )
    assert resolved.order == ("b", "a")
    assert resolved.labels == {"b": "Beta", "a": "Alpha"}
    assert result.provenance.extras["point_size_by_panel"]["category"] == pytest.approx(
        12.5
    )
    result.close()


def test_embedding_continuous_sorting_keeps_color_size_and_scatter_options_aligned():
    scores = np.asarray([2.0, np.nan, 1.0, 3.0])
    sizes = np.asarray([10.0, 20.0, 30.0, 40.0])
    store = _synthetic_plot_store(
        layout1=[0.0, 1.0, 2.0, 3.0],
        layout2=[0.0, 1.0, 0.0, 1.0],
        score=scores,
    )

    result = splt.embedding(
        store,
        layout_key="layout",
        color_by=splt.CellField("score", kind="continuous"),
        color_scale=splt.ColorScale(
            cmap="viridis",
            vmin=1.0,
            vmax=3.0,
            missing_color="#ff00ff",
        ),
        point_sizes=sizes,
        point_edgecolor="#222222",
        point_edgewidth=0.4,
        point_alpha=0.7,
        sort_values=True,
        rasterize_threshold=4,
        show_legend=False,
        show=False,
    )

    collection = result.axes["score"].collections[0]
    np.testing.assert_allclose(
        np.asarray(collection.get_offsets()),
        np.asarray([[1.0, 1.0], [2.0, 0.0], [0.0, 0.0], [3.0, 1.0]]),
    )
    np.testing.assert_array_equal(
        collection.get_sizes(),
        np.asarray([20.0, 30.0, 10.0, 40.0]),
    )
    np.testing.assert_allclose(
        collection.get_facecolors()[0, :3],
        matplotlib.colors.to_rgba("#ff00ff")[:3],
    )
    # Scores 1, 2 and 3 span the explicit [1, 3] limits.
    np.testing.assert_allclose(
        collection.get_facecolors()[1:, :3],
        plt.get_cmap("viridis")([0.0, 0.5, 1.0])[:, :3],
    )
    assert collection.get_alpha() == pytest.approx(0.7)
    assert collection.get_linewidths() == pytest.approx([0.4])
    assert collection.get_rasterized() is True
    assert result.provenance.extras["color_limits"]["score"] == pytest.approx(
        (1.0, 3.0)
    )
    assert result.provenance.extras["point_size_by_panel"]["score"] == pytest.approx(
        25.0
    )
    result.close()


def test_embedding_feature_matrix_prefetch_batches_feature_slots(monkeypatch):
    embedding_module = import_module("scarf.plotting.embedding")
    store = _synthetic_plot_store(category=["a", "b", "a", "b"])
    matrix = np.asarray(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ]
    )
    resolved_items = []
    fetches = []

    def resolve_feature(_store, item, *, from_assay):
        resolved_items.append((item, from_assay))
        label = item.label if isinstance(item, splt.FeatureRef) else str(item)
        return SimpleNamespace(label=label)

    def fetch_matrix(_store, resolved, cell_idx, *, normalization):
        fetches.append((resolved, cell_idx.copy(), normalization))
        return matrix

    monkeypatch.setattr(embedding_module, "resolve_feature", resolve_feature)
    monkeypatch.setattr(
        embedding_module,
        "fetch_normalized_feature_matrix",
        fetch_matrix,
    )
    normalization = splt.NormalizationSpec(transform="log1p")

    prefetched = embedding_module._prefetch_colors(
        store,
        [
            splt.FeatureRef("gene_a", label="Gene A"),
            splt.CellField("category", kind="categorical"),
            "gene_b",
        ],
        metadata_columns=set(store.cells.columns),
        from_assay="RNA",
        cell_key="I",
        n_cells=4,
        normalization=normalization,
    )

    assert [item for item, _ in resolved_items] == [
        splt.FeatureRef("gene_a", label="Gene A"),
        "gene_b",
    ]
    assert len(fetches) == 1
    np.testing.assert_array_equal(fetches[0][1], np.arange(4))
    assert fetches[0][2] is normalization
    np.testing.assert_array_equal(prefetched[0][0], matrix[:, 0])
    assert prefetched[0][1:] == ("Gene A", False, False)
    np.testing.assert_array_equal(prefetched[1][0], ["a", "b", "a", "b"])
    assert prefetched[1][1:] == ("category", True, False)
    np.testing.assert_array_equal(prefetched[2][0], matrix[:, 1])
    assert prefetched[2][1:] == ("gene_b", False, False)


def test_embedding_multi_layout_facets_include_requested_empty_panels():
    store = _synthetic_plot_store(
        first1=[0.0, 1.0, 2.0, 3.0],
        first2=[0.0, 1.0, 0.0, 1.0],
        second1=[3.0, 2.0, 1.0, 0.0],
        second2=[1.0, 0.0, 1.0, 0.0],
        facet=["a", "b", "a", "b"],
        score=[0.0, 10.0, 1.0, 11.0],
    )
    facets = ("b", "a", "missing")
    layouts = ("first", "second")

    result = splt.embedding(
        store,
        layout_key=layouts,
        color_by=splt.CellField("score", kind="continuous"),
        facet_by="facet",
        facet_order=facets,
        color_scale=splt.ColorScale(scope="panel"),
        point_size=5.0,
        show_legend=False,
        show=False,
    )

    expected_keys = [(layout, "score", facet) for layout in layouts for facet in facets]
    assert list(result.axes) == expected_keys
    assert result.provenance.extras["n_layouts"] == 2
    # Facet b holds cells 1 and 3, facet a cells 0 and 2.
    expected_offsets = {
        ("first", "score", "b"): [[1.0, 1.0], [3.0, 1.0]],
        ("first", "score", "a"): [[0.0, 0.0], [2.0, 0.0]],
        ("second", "score", "b"): [[2.0, 0.0], [0.0, 0.0]],
        ("second", "score", "a"): [[3.0, 1.0], [1.0, 1.0]],
    }
    for key, offsets in expected_offsets.items():
        np.testing.assert_allclose(
            result.axes[key].collections[0].get_offsets(), offsets
        )
    for layout in layouts:
        child = result.provenance.extras["layout_provenance"][layout]
        assert child.extras["n_facets"] == 3
        assert child.extras["color_scale_scope"] == "panel"
        # Panel scope gives each facet the range of its own scores.
        assert child.extras["color_limits"] == {
            "('score', 'b')": pytest.approx((10.0, 11.0)),
            "('score', 'a')": pytest.approx((0.0, 1.0)),
        }
        assert result.axes[(layout, "score", "missing")].axison is False
    assert result.axes[("first", "score", "missing")].get_title() == (
        "first | score | facet=missing (empty)"
    )
    result.close()


def test_embedding_places_one_side_legend_per_categorical_panel():
    rng = np.random.default_rng(0)
    n_cells = 120
    store = _synthetic_plot_store(
        layout1=rng.normal(size=n_cells),
        layout2=rng.normal(size=n_cells),
        cluster=np.repeat(["a", "b", "c", "d"], n_cells // 4),
        sample=np.tile(["s1", "s2", "s3"], n_cells // 3),
    )

    single = splt.embedding(
        store,
        layout_key="layout",
        color_by="cluster",
        show=False,
    )
    try:
        assert [legend.get_title().get_text() for legend in single.figure.legends] == [
            "cluster"
        ]
    finally:
        single.close()

    result = splt.embedding(
        store,
        layout_key="layout",
        color_by=["cluster", "sample"],
        show=False,
    )
    try:
        figure = result.figure
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        boxes = [legend.get_window_extent(renderer) for legend in figure.legends]
        assert [legend.get_title().get_text() for legend in figure.legends] == [
            "cluster",
            "sample",
        ]
        assert not boxes[0].overlaps(boxes[1])
    finally:
        result.close()


def test_embedding_multi_layout_derives_facets_from_selected_cells():
    store = _synthetic_plot_store(
        first1=[0.0, 1.0, 2.0, 3.0, 4.0],
        first2=[0.0, 1.0, 0.0, 1.0, 0.0],
        second1=[4.0, 3.0, 2.0, 1.0, 0.0],
        second2=[1.0, 0.0, 1.0, 0.0, 1.0],
        facet=np.asarray(["b", np.nan, "a", np.nan, "ignored"], dtype=object),
        selected=[True, True, True, True, False],
    )

    result = splt.embedding(
        store,
        layout_key=("first", "second"),
        color_by=None,
        facet_by="facet",
        subset_by="selected",
        point_size=6.0,
        frame="axes",
        show_legend=False,
        show=False,
    )

    assert result.owns_figure is True
    assert len(result.axes) == 6
    # Selected cells carry facets a, b and missing; the unselected cell's facet
    # never becomes a panel.
    offsets = [
        np.asarray(axis.collections[0].get_offsets()).tolist()
        for axis in result.axes.values()
    ]
    assert offsets == [
        [[2.0, 0.0]],
        [[0.0, 0.0]],
        [[1.0, 1.0], [3.0, 1.0]],
        [[2.0, 1.0]],
        [[4.0, 1.0]],
        [[3.0, 0.0], [1.0, 0.0]],
    ]
    first_row = list(result.axes.values())[:3]
    assert [axis.get_xlabel() for axis in first_row] == ["first1"] * 3
    assert [axis.get_ylabel() for axis in first_row] == ["first2", "", ""]
    assert result.provenance.n_cells == 4
    assert result.provenance.extras["n_layouts"] == 2
    assert all(
        child.extras["n_facets"] == 3
        for child in result.provenance.extras["layout_provenance"].values()
    )
    figure_number = result.figure.number
    result.close()
    assert not plt.fignum_exists(figure_number)


def test_embedding_facets_preserve_sizes_filters_and_target_legend():
    store = _synthetic_plot_store(
        layout1=np.arange(8, dtype=np.float64),
        layout2=[0.0, 1.0, 0.2, 1.2, 0.4, 1.4, 0.6, 1.6],
        facet=["left"] * 4 + ["right"] * 4,
        category=["a", "b", "a", "b", "a", "b", "a", "b"],
        highlight_group=["hot", "cold", "hot", "cold"] * 2,
        density_group=["keep", "keep", "keep", "drop"] * 2,
    )
    sizes = np.arange(2, 10, dtype=np.float64)
    figure, target_axes = plt.subplots(1, 2, figsize=(6, 3))
    targets = {
        ("category", "left"): target_axes[0],
        ("category", "right"): target_axes[1],
    }

    result = splt.embedding(
        store,
        layout_key="layout",
        color_by=splt.CellField("category", kind="categorical"),
        facet_by="facet",
        groups=("left", "right"),
        point_sizes=sizes,
        legend_loc="right",
        highlight=splt.Highlight(by="highlight_group", groups=("hot",)),
        density_overlay=splt.DensityOverlay(
            group_by="density_group",
            groups=("keep",),
            pixels=16,
            sigma=1,
            levels=2,
        ),
        target=targets,
        show=False,
    )

    assert result.owns_figure is False
    np.testing.assert_array_equal(
        target_axes[0].collections[0].get_sizes(),
        sizes[:4],
    )
    np.testing.assert_array_equal(
        target_axes[1].collections[0].get_sizes(),
        sizes[4:],
    )
    assert target_axes[0].get_legend() is None
    assert target_axes[1].get_legend() is not None
    assert result.provenance.extras["highlight"]["n_highlighted"] == 4
    result.close()
    assert plt.fignum_exists(figure.number)
    plt.close(figure)


def test_embedding_on_data_legend_reports_omitted_labels():
    store = _synthetic_plot_store(
        layout1=[0.0, 0.2, 1.0, 1.2, 2.0, 2.2],
        layout2=[0.0, 0.1, 1.0, 1.1, 0.0, 0.1],
        category=["a", "a", "b", "b", "b", "b"],
    )
    scale = splt.CategoricalScale(
        order=("a", "b", "not observed"),
        palette={
            "a": "#111111",
            "b": "#777777",
            "not observed": "#dddddd",
        },
    )

    result = splt.embedding(
        store,
        layout_key="layout",
        color_by=splt.CellField("category", kind="categorical"),
        categorical_scale=scale,
        legend_loc="on_data",
        max_on_data_labels=1,
        point_size=5,
        show=False,
    )

    assert [text.get_text() for text in result.axes["category"].texts] == ["b"]
    assert result.provenance.extras["omitted_labels"]["category"] == ["a"]
    result.close()


def test_embedding_renders_non_linear_continuous_scales():
    from matplotlib.colors import SymLogNorm

    store = _synthetic_plot_store(
        I=np.ones(4, dtype=bool),
        umap1=np.arange(4.0),
        umap2=np.zeros(4),
        positive=np.array([0.25, 0.5, 1.0, 2.0]),
        signed=np.array([-2.0, -0.5, 0.5, 3.0]),
    )
    viridis = plt.get_cmap("viridis")

    log = splt.embedding(
        store,
        layout_key="umap",
        color_by="positive",
        color_scale=splt.ColorScale(scale="log"),
        show=False,
    )
    points = log.axes["positive"].collections[0]
    # Doubling values sit evenly on a log scale.
    np.testing.assert_allclose(
        points.get_facecolors(), viridis([0.0, 1 / 3, 2 / 3, 1.0]), atol=1e-6
    )
    assert log.provenance.extras["color_limits"]["positive"] == (0.25, 2.0)
    log.close()

    symlog = splt.embedding(
        store,
        layout_key="umap",
        color_by="signed",
        color_scale=splt.ColorScale(scale="symlog"),
        show=False,
    )
    points = symlog.axes["signed"].collections[0]
    # The linear region spans 0.1% of the value range.
    norm = SymLogNorm(linthresh=0.005, vmin=-2.0, vmax=3.0)
    expected = viridis(norm(np.array([-2.0, -0.5, 0.5, 3.0])))
    assert symlog.provenance.extras["color_limits"]["signed"] == (-2.0, 3.0)
    np.testing.assert_allclose(points.get_facecolors(), expected, atol=1e-6)
    linear = viridis(np.array([0.0, 0.3, 0.5, 1.0]))
    assert not np.allclose(points.get_facecolors(), linear, atol=1e-3)
    symlog.close()


def test_imported_embedding_reuse_guard_and_validator_reject_damage():
    from scarf.embeddings.imported import (
        _payloads_match,
        validate_imported_embedding_artifact,
    )
    from scarf.embeddings.imported_storage import ImportedArtifactStorage
    from scarf.storage.artifacts import fingerprint_array
    from tests.test_imported_coordinates import (
        _root_with_selection,
        _tamper_artifact_attribute,
        _write_embedding_fixture,
    )

    root, selection, cell_ids, mask = _root_with_selection()
    ref, coordinates = _write_embedding_fixture(root, selection, cell_ids, mask)
    storage = ImportedArtifactStorage(root)
    group = storage.artifact_group(ref)
    fingerprint = fingerprint_array(coordinates)

    assert not _payloads_match(
        storage,
        group,
        shapes={"missing": coordinates.shape},
        fingerprints={"missing": fingerprint},
    )
    assert not _payloads_match(
        storage,
        group,
        shapes={"values": (len(coordinates) + 1, coordinates.shape[1])},
        fingerprints={"values": fingerprint},
    )
    group.create_group("not_an_array")
    assert not _payloads_match(
        storage,
        group,
        shapes={"not_an_array": coordinates.shape},
        fingerprints={"not_an_array": fingerprint},
    )

    del group["values"]
    with pytest.raises(ValueError, match="has no values array"):
        validate_imported_embedding_artifact(root, ref)

    other_root, other_selection, other_ids, other_mask = _root_with_selection()
    other_ref, _ = _write_embedding_fixture(
        other_root,
        other_selection,
        other_ids,
        other_mask,
    )
    _tamper_artifact_attribute(
        other_root,
        other_ref,
        "provenance",
        ("inputs", "source_digest"),
        {"bytes_hex": "a" * 63},
    )
    with pytest.raises(ValueError, match="source digest is missing"):
        validate_imported_embedding_artifact(other_root, other_ref)


def _dotplot_store():
    """Twelve cells in groups 0, 1 and 2 with four features."""
    from tests.test_plotting_foundation import _ArrayStore

    values = np.column_stack(
        (
            np.arange(12.0),
            np.repeat([0.0, 1.0, 2.0], 4),
            np.tile([0.0, 0.0, 3.0, 3.0], 3),
            np.ones(12),
        )
    )
    return _ArrayStore(
        {"I": np.ones(12, dtype=bool), "cluster": np.repeat(["0", "1", "2"], 4)},
        values,
        names=("GeneA", "GeneB", "GeneC", "GeneD"),
    )


def test_dotplot_feature_brackets_and_axis_swap():
    genes = ["GeneA", "GeneB", "GeneC", "GeneD"]
    result = splt.dotplot(
        _dotplot_store(),
        features={"Lineage": genes[:2], "State": genes[2:]},
        group_by="cluster",
        swap_axes=True,
        marker_linewidth=0.6,
        show=False,
    )

    ax = result.axes["dotplot"]
    # Swapped axes put features on x and groups on y.
    assert [text.get_text() for text in ax.get_xticklabels()] == genes
    assert [text.get_text() for text in ax.get_yticklabels()] == ["0", "1", "2"]
    assert ax.get_ylabel() == "cluster"
    assert ax.get_xlabel() == ""
    dots = ax.collections[0]
    offsets = np.asarray(dots.get_offsets())
    assert sorted(map(tuple, offsets)) == [
        (float(feature), float(group)) for feature in range(4) for group in range(3)
    ]
    assert dots.get_linewidths().tolist() == [0.6]
    brackets = [line for line in ax.lines if line.get_gid() == "feature-group-bracket"]
    # Brackets above the plot span each feature group's columns.
    assert [
        (line.get_xdata().tolist(), line.get_ydata().tolist()) for line in brackets
    ] == [
        (pytest.approx([-0.35, 1.35]), pytest.approx([1.03, 1.03])),
        (pytest.approx([1.65, 3.35]), pytest.approx([1.03, 1.03])),
    ]
    assert result.provenance.extras["feature_group_brackets"] == 2
    assert [(text.get_text(), text.get_position()) for text in ax.texts] == [
        ("Lineage", pytest.approx((0.5, 1.07))),
        ("State", pytest.approx((2.5, 1.07))),
    ]
    result.close()


def test_dotplot_marker_sizes_follow_physical_grid_cells():
    genes = ["GeneA", "GeneB", "GeneC", "GeneD"]
    features = {"Lineage": genes[:2], "State": genes[2:]}
    sizes = {}
    for name, figsize in (("compact", (3, 2)), ("large", (8, 6))):
        figure, ax = plt.subplots(figsize=figsize)
        result = splt.dotplot(
            _dotplot_store(),
            features=features,
            group_by="cluster",
            target=ax,
            show_legend=False,
            show=False,
        )
        figure.canvas.draw()
        # Marker diameters follow the physical size of one grid cell.
        box = ax.get_position()
        slot = min(box.width * figsize[0] * 72 / 3, box.height * figsize[1] * 72 / 4)
        largest = np.clip(0.72 * slot, 2.5, 26.0)
        smallest = np.clip(0.18 * largest, 1.0, 3.0)
        assert result.provenance.extras["size_range"] == pytest.approx(
            [smallest**2, largest**2]
        )
        fractions = result.tables["aggregate"]["fraction"].to_numpy()
        np.testing.assert_allclose(
            np.sort(ax.collections[0].get_sizes()),
            np.sort(smallest**2 + fractions * (largest**2 - smallest**2)),
        )
        assert result.provenance.extras["size_scale_source"] == "panel"
        sizes[name] = largest
        if name == "compact":
            renderer = figure.canvas.get_renderer()
            label_left = min(
                label.get_window_extent(renderer).x0
                for label in ax.get_yticklabels()
                if label.get_text()
            )
            bracket_right = max(
                line.get_transform().transform(line.get_xydata())[:, 0].max()
                for line in ax.lines
                if line.get_gid() == "feature-group-bracket"
            )
            assert bracket_right < label_left
        result.close()
        plt.close(figure)
    assert sizes["large"] > sizes["compact"]


def test_dotplot_left_group_labels_clear_feature_tick_labels():
    genes = ["GeneA", "GeneB", "GeneC", "GeneD"]
    result = splt.dotplot(
        _dotplot_store(),
        features={"Myeloid lineage": genes[:2], "Cell state": genes[2:]},
        group_by="cluster",
        show_legend=False,
        show=False,
    )

    figure = result.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    feature_label_left = min(
        label.get_window_extent(renderer).x0
        for label in result.axes["dotplot"].get_yticklabels()
        if label.get_text()
    )
    group_label_right = max(
        label.get_window_extent(renderer).x1
        for label in result.axes["dotplot"].texts
        if label.get_gid() == "feature-group-label"
    )

    assert feature_label_left - group_label_right >= figure.dpi * 6.0 / 72.0
    result.close()


def _violin_store():
    """Groups a, b and c of four cells with two metadata keys and two genes.

    ``metric`` has group means 1.5, 5.5 and 9.5; ``metric2`` has 5.15, 1.15
    and 3.15. The genes carry the same values as features.
    """
    from tests.test_plotting_foundation import _ArrayStore

    metric = np.arange(12, dtype=float)
    metric2 = np.repeat([5.0, 1.0, 3.0], 4) + np.tile([0.0, 0.1, 0.2, 0.3], 3)
    return _ArrayStore(
        {
            "I": np.ones(12, dtype=bool),
            "group": np.repeat(["a", "b", "c"], 4).astype(object),
            "metric": metric,
            "metric2": metric2,
            "constant": np.full(12, 5.0),
        },
        np.column_stack((metric, metric2)),
        names=("GeneA", "GeneB"),
    )


def _violin_colors(ax):
    """Fill colors of the violin bodies, in group order."""
    return np.asarray([body.get_facecolor()[0][:3] for body in ax.collections])


def _expected_colors(fractions, cmap="viridis"):
    from matplotlib import colormaps
    from matplotlib.colors import to_rgb
    from seaborn.utils import desaturate

    return np.asarray(
        [to_rgb(desaturate(colormaps[cmap](float(t)), 0.9)) for t in fractions]
    )


def _stacked(store, keys, **kwargs):
    return splt.distribution(
        store,
        keys,
        grouping=splt.CellField("group"),
        kind="stacked_violin",
        max_points=0,
        show=False,
        **kwargs,
    )


def test_stacked_violin_standardizes_rows():
    result = _stacked(_violin_store(), ["metric", "metric2"], row_standardize=True)

    assert list(result.axes) == ["metric", "metric2"]
    store = _violin_store()
    for key, table in result.tables.items():
        values = store.cells.fetch(key)
        # Rows standardize with the population standard deviation.
        np.testing.assert_allclose(
            table["display_value"], (values - values.mean()) / values.std()
        )
    assert result.axes["metric2"].get_ylabel() == "standardized value"
    assert result.provenance.extras["row_standardize"] is True
    result.close()


def test_distribution_reads_cell_cycle_scores_from_exact_artifact(
    cell_cycle_scoring,
    leiden_clustering,
    datastore,
):
    from scarf.storage.artifacts import artifact_group

    result = splt.distribution(
        datastore,
        keys=cell_cycle_scoring,
        grouping=leiden_clustering,
        kind="box",
        max_points=0,
        show=False,
    )

    stored = artifact_group(datastore.zw, cell_cycle_scoring)
    cells = _artifact_cell_indices(datastore, cell_cycle_scoring)
    np.testing.assert_array_equal(
        cells, _artifact_cell_indices(datastore, leiden_clustering)
    )
    labels = np.asarray(artifact_group(datastore.zw, leiden_clustering)["values"][:])
    assert set(result.tables) == {"s_score", "g2m_score"}
    for name, table in result.tables.items():
        np.testing.assert_allclose(table["value"], stored[name][:])
        np.testing.assert_array_equal(table["group"], labels)
    assert result.provenance.cell_key is None
    assert result.provenance.assay is None
    assert result.provenance.n_cells == len(cells)
    assert result.provenance.extras["values"] == cell_cycle_scoring.to_dict()
    assert result.provenance.extras["grouping"] == leiden_clustering.to_dict()
    assert (
        result.provenance.extras["cell_selection"]
        == datastore.inspect_artifact(leiden_clustering).inputs["cell_selection"]
    )
    assert "group_by" not in result.provenance.extras
    assert "cell_key" not in result.provenance.extras
    assert result.provenance.extras["normalization"] is None
    result.close()


def test_distribution_plots_cell_cycle_scores_without_grouping(
    cell_cycle_scoring,
    datastore,
):
    from scarf.storage.artifacts import artifact_group

    result = splt.distribution(
        datastore,
        keys=cell_cycle_scoring,
        kind="hist",
        show=False,
    )

    stored = artifact_group(datastore.zw, cell_cycle_scoring)
    cells = _artifact_cell_indices(datastore, cell_cycle_scoring)
    # Without a grouping the artifact's own selection decides the cells.
    for name in ("s_score", "g2m_score"):
        np.testing.assert_allclose(result.tables[name]["value"], stored[name][:])
    assert result.provenance.n_cells == len(cells)
    assert (
        result.provenance.extras["cell_selection"]
        == datastore.inspect_artifact(cell_cycle_scoring).inputs["cell_selection"]
    )
    result.close()


def test_distribution_rejects_cell_cycle_scores_for_other_cells(
    cell_cycle_scoring,
    datastore,
):
    cells = _artifact_cell_indices(datastore, cell_cycle_scoring)
    assert len(cells) < datastore.cells.N
    datastore.cells.insert(
        "plot_every_cell", np.ones(datastore.cells.N, dtype=bool), overwrite=True
    )
    datastore.cells.insert(
        "plot_cell_parity",
        np.asarray(["even", "odd"] * (datastore.cells.N // 2 + 1))[: datastore.cells.N],
        overwrite=True,
    )

    with pytest.raises(ValueError) as raised:
        splt.distribution(
            datastore,
            keys=cell_cycle_scoring,
            grouping=splt.CellField("plot_cell_parity"),
            cell_selection=datastore.snapshot_cell_selection("plot_every_cell"),
            show=False,
        )

    assert raised.value.args == (
        "keys and grouping artifacts must use the same ordered cells",
    )


@pytest.mark.parametrize(
    ("arrays", "error", "message"),
    [
        (
            {"g2m_score": "aligned"},
            ValueError,
            "Cell-cycle artifact has no canonical 's_score' array",
        ),
        (
            {"s_score": "short", "g2m_score": "aligned"},
            ValueError,
            "Cell-cycle 's_score' values do not align with their cell selection",
        ),
        (
            {"s_score": "text", "g2m_score": "aligned"},
            TypeError,
            "Cell-cycle 's_score' values must be numeric",
        ),
    ],
)
def test_distribution_rejects_malformed_cell_cycle_payloads(
    cell_cycle_scoring,
    datastore,
    monkeypatch,
    arrays,
    error,
    message,
):
    import zarr
    from zarr.storage import MemoryStore

    distribution_module = import_module("scarf.plotting.distribution")
    n_cells = len(_artifact_cell_indices(datastore, cell_cycle_scoring))
    payloads = {
        "aligned": np.zeros(n_cells),
        "short": np.zeros(n_cells - 1),
        "text": np.asarray(["S"] * n_cells),
    }
    group = zarr.open_group(store=MemoryStore(), mode="w")
    for name, payload in arrays.items():
        group.create_array(name, data=payloads[payload])
    original = distribution_module.artifact_group
    monkeypatch.setattr(
        distribution_module,
        "artifact_group",
        lambda root, ref: group if ref == cell_cycle_scoring else original(root, ref),
    )

    with pytest.raises(error) as raised:
        splt.distribution(datastore, keys=cell_cycle_scoring, show=False)

    assert raised.value.args == (message,)


def test_distribution_rejects_non_cell_cycle_or_mixed_value_artifacts(
    cell_cycle_scoring,
    leiden_clustering,
    datastore,
):
    with pytest.raises(ValueError, match="assay-scoped cell_cycle"):
        splt.distribution(datastore, keys=leiden_clustering, show=False)
    with pytest.raises(TypeError, match="complete keys argument"):
        splt.distribution(
            datastore,
            keys=["RNA_nCounts", cell_cycle_scoring],  # type: ignore[list-item]
            show=False,
        )


def test_stacked_violin_can_share_value_scale():
    store = _synthetic_plot_store(
        I=np.ones(12, dtype=bool),
        group=np.repeat(["a", "b", "c"], 4),
        metric=np.arange(12, dtype=float),
        metric2=10 * np.arange(12, dtype=float),
    )

    shared = _stacked(store, ["metric", "metric2"], share_y=True)
    # Every row spans the pooled values 0 to 110 with 5% padding.
    assert [axis.get_ylim() for axis in shared.axes.values()] == [
        pytest.approx((-5.5, 115.5))
    ] * 2
    assert shared.provenance.extras["share_y"] is True
    shared.close()

    independent = _stacked(store, ["metric", "metric2"])
    limits = [axis.get_ylim() for axis in independent.axes.values()]
    assert limits[0] != pytest.approx(limits[1])
    independent.close()


def test_distribution_aggregates_biological_samples():
    store = _synthetic_plot_store(
        I=np.ones(12, dtype=bool),
        group=np.repeat(["a", "b"], 6),
        sample=np.tile(["s1", "s1", "s2"], 4),
        metric=np.array([1, 5, 2, 9, 4, 7, 3, 3, 8, 6, 0, 1], dtype=float),
    )

    result = splt.distribution(
        store,
        "metric",
        grouping=splt.CellField("group"),
        sample_by="sample",
        sample_stat="median",
        kind="box",
        max_points=0,
        show=False,
    )

    table = result.tables["metric"].sort_values(["sample", "group"])
    # Medians per sample and group: s1/a of [1, 5, 9, 4], s1/b of [3, 3, 6, 0],
    # s2/a of [2, 7] and s2/b of [8, 1].
    assert list(zip(table["sample"], table["group"], strict=True)) == [
        ("s1", "a"),
        ("s1", "b"),
        ("s2", "a"),
        ("s2", "b"),
    ]
    np.testing.assert_allclose(table["value"], [4.5, 3.0, 4.5, 4.5])
    assert table["nCells"].tolist() == [4, 4, 2, 2]
    assert result.provenance.n_samples == 2
    assert result.provenance.extras["sample_stat"] == "median"
    assert result.axes["metric"].get_ylabel() == "Sample median metric"
    result.close()


def test_distribution_draws_sample_aware_split_violins():
    from matplotlib.colors import to_rgb
    from seaborn.utils import desaturate

    cells = np.arange(48)
    sample_index = cells % 8
    metric = cells.astype(float)
    store = _synthetic_plot_store(
        I=np.ones(48, dtype=bool),
        group=np.where(cells < 24, "a", "b"),
        sample=np.asarray([f"s{index}" for index in sample_index]),
        cond=np.where(sample_index % 2 == 0, "control", "stimulated"),
        metric=metric,
    )

    result = splt.distribution(
        store,
        "metric",
        grouping=splt.CellField("group"),
        split_by="cond",
        study_design=splt.StudyDesign(sample_by="sample"),
        kind="violin",
        max_points=0,
        show=False,
    )

    table = result.tables["metric"].sort_values(["group", "sample"])
    # Sample s holds cells s, s + 8 and s + 16 in group a and s + 24, s + 32 and
    # s + 40 in group b, so its means are s + 8 and s + 32.
    assert len(table) == 16
    expected = [index + 8.0 for index in range(8)] + [
        index + 32.0 for index in range(8)
    ]
    np.testing.assert_allclose(table["value"], expected)
    assert table["split"].tolist() == ["control", "stimulated"] * 8
    scale = result.scales[0]
    assert scale.order == ("control", "stimulated")
    assert result.legends[0].label == "cond"
    ax = result.axes["metric"]
    legend = ax.get_legend()
    assert legend.get_title().get_text() == "cond"
    assert [text.get_text() for text in legend.get_texts()] == ["control", "stimulated"]
    bodies = [body.get_facecolor()[0][:3] for body in ax.collections]
    expected_colors = [
        to_rgb(desaturate(scale.palette[split], 0.9))
        for split in ("control", "stimulated")
    ] * 2
    np.testing.assert_allclose(bodies, expected_colors, atol=0.005)
    result.close()


def test_distribution_seed_repeats_point_jitter():
    rng = np.random.default_rng(3)
    store = _synthetic_plot_store(
        I=np.ones(120, dtype=bool),
        group=np.repeat(["a", "b", "c"], 40),
        metric=rng.normal(size=120),
    )
    kwargs = {
        "grouping": splt.CellField("group"),
        "kind": "box",
        "max_points": 80,
        "show": False,
    }
    global_state = np.random.get_state()[1].copy()

    def strip_offsets(seed):
        result = splt.distribution(store, "metric", seed=seed, **kwargs)
        offsets = np.concatenate(
            [
                np.asarray(collection.get_offsets())
                for collection in result.axes["metric"].collections
            ]
        )
        result.close()
        return offsets

    first = strip_offsets(17)
    repeated = strip_offsets(17)
    other = strip_offsets(18)

    assert len(first) == 80
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, other)
    # Jitter draws from a private generator and leaves NumPy's global state.
    np.testing.assert_array_equal(np.random.get_state()[1], global_state)


@pytest.fixture(scope="module")
def imported_plot_store(tmp_path_factory):
    from tests.test_plotting_foundation import _imported_plot_store

    store, imported, counts = _imported_plot_store(
        tmp_path_factory.mktemp("modern_plots"),
        coordinates=np.column_stack((np.arange(16.0), np.arange(16.0) % 4)),
        clusters=np.arange(16) % 2,
    )
    return SimpleNamespace(
        store=store,
        imported=imported,
        layout=imported.embeddingArtifacts["X_umap"],
        normalized=counts / counts.sum(axis=1, keepdims=True) * 1000.0,
    )


def test_sample_aggregated_feature_axis_retains_requested_italics(
    imported_plot_store,
):
    data = imported_plot_store
    samples = np.asarray([f"sample_{index % 4}" for index in range(16)])
    data.store.cells.insert("plot_italic_sample", samples, overwrite=True)

    result = splt.distribution(
        data.store,
        keys="CD3E",
        cell_selection=data.imported.cellSelection,
        sample_by="plot_italic_sample",
        kind="box",
        italicize_features=True,
        show=False,
    )

    axis = result.axes["CD3E"]
    assert axis.get_ylabel() == "Sample mean CD3E"
    assert axis.yaxis.label.get_fontstyle() == "italic"
    table = result.tables["CD3E"].sort_values("sample")
    expected = [data.normalized[samples == f"sample_{i}", 0].mean() for i in range(4)]
    np.testing.assert_allclose(table["value"], expected, rtol=1e-5)
    assert (
        result.provenance.extras["cell_selection"]
        == data.imported.cellSelection.to_dict()
    )
    result.close()


def test_per_sample_composition_reports_uncertainty():
    from scipy.stats import t as student_t

    # Ten cells per sample; the count of category "a" per sample.
    a_counts = {"c1": 2, "c2": 3, "c3": 3, "c4": 4, "t1": 5, "t2": 6, "t3": 6, "t4": 7}
    samples = np.repeat(list(a_counts), 10)
    categories = np.concatenate(
        [["a"] * count + ["b"] * (10 - count) for count in a_counts.values()]
    )
    store = _synthetic_plot_store(
        I=np.ones(80, dtype=bool),
        cat=categories,
        sample=samples,
        condition=np.where(np.char.startswith(samples, "c"), "control", "stimulated"),
        subject=np.char.add(
            "s", np.char.replace(np.char.replace(samples, "c", ""), "t", "")
        ),
    )

    result = splt.composition(
        store,
        category_by="cat",
        study_design=splt.StudyDesign(
            sample_by="sample",
            condition_by="condition",
            subject_by="subject",
        ),
        kind="per_sample",
        uncertainty="ci95",
        show=False,
    )

    summary = result.tables["summary"].set_index(["category", "condition"])
    critical = student_t.ppf(0.975, 3)
    for category, condition, proportions in (
        ("a", "control", [0.2, 0.3, 0.3, 0.4]),
        ("a", "stimulated", [0.5, 0.6, 0.6, 0.7]),
        ("b", "control", [0.8, 0.7, 0.7, 0.6]),
        ("b", "stimulated", [0.5, 0.4, 0.4, 0.3]),
    ):
        mean = np.mean(proportions)
        margin = critical * np.std(proportions, ddof=1) / 2
        row = summary.loc[(category, condition)]
        assert row["mean_proportion"] == pytest.approx(mean)
        assert (row["lower"], row["upper"]) == pytest.approx(
            (mean - margin, mean + margin)
        )
        assert row["n_samples"] == 4
    assert result.provenance.extras["uncertainty"] == "ci95"
    assert result.provenance.extras["n_pair_lines"] == 8
    axis = result.axes["composition"]
    assert axis.get_ylim()[1] >= float(result.tables["summary"]["upper"].max())
    assert [
        (
            legend.get_title().get_text(),
            [text.get_text() for text in legend.get_texts()],
        )
        for legend in result.figure.legends
    ] == [
        ("cat", ["a", "b"]),
        ("Condition", ["control", "stimulated"]),
        ("Summary", ["mean with 95% CI"]),
    ]
    result.close()


def test_embedding_side_legend_stays_page_sized_for_many_categories():
    labels = np.full(300, "type_149", dtype=object)
    labels[:149] = [f"type_{index:03d}" for index in range(149)]
    store = _scatter_store(n=300, plot_many_types=labels)

    result = splt.embedding(
        store,
        layout_key="umap",
        color_by="plot_many_types",
        legend_loc="right",
        show=False,
    )

    width, _ = result.figure.get_size_inches()
    assert width <= 12
    legend = result.figure.legends[0]
    shown = [text.get_text() for text in legend.get_texts()]
    assert len(shown) == 80
    assert legend.get_title().get_text() == "plot_many_types (80 of 150)"
    assert "type_149" in shown
    omitted = result.provenance.extras["omitted_legend_entries"]
    panel_key = str(next(iter(result.axes)))
    # The largest category stays; the tail of the singletons is omitted.
    assert omitted[panel_key] == [f"type_{index:03d}" for index in range(79, 149)]
    result.close()


def test_cluster_connectivity_runs_on_real_datastore_graph(
    umap,
    leiden_clustering,
    connectivity_graph,
    datastore,
):
    from scipy import sparse

    from scarf.storage.artifacts import artifact_group

    result = datastore.plots.cluster_connectivity(
        groups=leiden_clustering,
        layout=umap,
        graph=connectivity_graph,
        minimum_edge_weight=0,
        max_edges_per_node=3,
        show_cells=True,
        show=False,
    )

    labels = np.asarray(artifact_group(datastore.zw, leiden_clustering)["values"][:])
    coordinates = np.asarray(artifact_group(datastore.zw, umap)["values"][:])
    graph = sparse.csr_matrix(
        datastore.load_graph(connectivity_graph, symmetric=True), dtype=np.float64
    )
    graph.setdiag(0.0)
    graph.eliminate_zeros()

    nodes = result.tables["nodes"].set_index("category")
    order = list(nodes.index)
    assert sorted(order) == sorted(np.unique(labels).tolist())
    for category in order:
        cells = labels == category
        assert nodes.loc[category, "nCells"] == cells.sum()
        assert nodes.loc[category, "x"] == pytest.approx(
            np.median(coordinates[cells, 0])
        )
        assert nodes.loc[category, "y"] == pytest.approx(
            np.median(coordinates[cells, 1])
        )

    # Undirected intercluster weights, each cell edge counted once.
    upper = sparse.triu(graph, k=1).tocoo()
    pairs = pd.DataFrame(
        {
            "a": labels[upper.row],
            "b": labels[upper.col],
            "weight": upper.data,
        }
    )
    pairs = pairs[pairs["a"] != pairs["b"]]
    pairs[["a", "b"]] = np.sort(pairs[["a", "b"]].to_numpy(), axis=1)
    raw = pairs.groupby(["a", "b"])["weight"].sum()
    incident = pd.Series(np.asarray(graph.sum(axis=1)).ravel()).groupby(labels).sum()
    edges = result.tables["edges"]
    assert result.provenance.extras["n_aggregated_edges"] == len(raw)
    for row in edges.itertuples():
        key = tuple(sorted((row.source, row.target)))
        assert row.rawWeight == pytest.approx(raw.loc[key])
        assert row.normalizedWeight == pytest.approx(
            raw.loc[key] / np.sqrt(incident[row.source] * incident[row.target])
        )
    degree = pd.concat([edges["source"], edges["target"]]).value_counts()
    assert degree.max() <= 3

    background = result.axes["cluster_connectivity"].collections[0]
    assert background.get_alpha() == pytest.approx(0.3)
    assert background.get_sizes()[0] >= 4
    np.testing.assert_allclose(background.get_offsets(), coordinates)
    palette = result.scales[0].palette
    np.testing.assert_allclose(
        background.get_facecolors()[:, :3],
        [mcolors.to_rgb(palette[label]) for label in labels],
    )
    assert result.provenance.extras["cell_size_source"] == "panel"
    result.close()


def test_composition_borders_labels_and_stored_palette():
    display = {
        "kind": "categorical",
        "categories": [
            {"value": "a", "label": "Alpha", "color": "#123456"},
            {"value": "b", "label": "Beta", "color": "#abcdef"},
        ],
    }
    store = _synthetic_plot_store(
        I=np.ones(8, dtype=bool),
        umap1=np.arange(8.0),
        umap2=np.zeros(8),
        category=np.array(list("aaababbb"), dtype=object),
        sample=np.repeat(["s0", "s1"], 4),
    )
    store._stored_display_metadata = lambda column: (
        display if column == "category" else None
    )

    embedding = splt.embedding(
        store, layout_key="umap", color_by="category", show=False
    )
    composition = splt.composition(
        store,
        category_by="category",
        sample_by="sample",
        segment_linewidth=0.8,
        show_percent_labels=True,
        label_min_fraction=0.3,
        show=False,
    )

    points = embedding.axes["category"].collections[0]
    assert [mcolors.to_hex(color) for color in points.get_facecolors()] == [
        "#123456" if value == "a" else "#abcdef" for value in "aaababbb"
    ]
    assert [text.get_text() for text in embedding.figure.legends[0].get_texts()] == [
        "Alpha",
        "Beta",
    ]
    # Sample s0 is 3/4 a and s1 is 1/4 a; bars stack a then b.
    bars = composition.axes["composition"].patches
    assert [mcolors.to_hex(bar.get_facecolor()) for bar in bars] == [
        "#123456",
        "#123456",
        "#abcdef",
        "#abcdef",
    ]
    assert [bar.get_height() for bar in bars] == pytest.approx([0.75, 0.25, 0.25, 0.75])
    assert all(bar.get_linewidth() == pytest.approx(0.8) for bar in bars)
    assert [
        (text.get_text(), text.get_position())
        for text in composition.axes["composition"].texts
    ] == [("75%", pytest.approx((0.0, 0.375))), ("75%", pytest.approx((0.94, 0.625)))]
    assert [text.get_text() for text in composition.figure.legends[0].get_texts()] == [
        "Alpha",
        "Beta",
    ]
    embedding.close()
    composition.close()


def test_composition_uses_two_exact_artifact_axes(
    cell_cycle_scoring,
    leiden_clustering,
    datastore,
):
    result = splt.composition(
        datastore,
        categories=cell_cycle_scoring,
        grouping=leiden_clustering,
        kind="stacked",
        show=False,
    )

    assert "per_group" in result.tables
    assert "per_sample" not in result.tables
    assert result.provenance.cell_key is None
    assert result.provenance.n_samples is None
    assert result.provenance.extras["categories"] == cell_cycle_scoring.to_dict()
    assert result.provenance.extras["grouping"] == leiden_clustering.to_dict()
    assert result.axes["composition"].get_xlabel() == "grouping"
    result.close()

    with pytest.raises(ValueError, match="mutually exclusive"):
        splt.composition(
            datastore,
            categories=cell_cycle_scoring,
            grouping=leiden_clustering,
            sample_by="RNA_nCounts",
            show=False,
        )
    with pytest.raises(ValueError, match="only for kind='stacked'"):
        splt.composition(
            datastore,
            categories=cell_cycle_scoring,
            grouping=leiden_clustering,
            kind="per_sample",
            show=False,
        )


def test_compose_results_namespaces_tables_and_renders_shared_legend():
    store = _scatter_store(
        n=60,
        cluster=np.repeat(["c1", "c2", "c3"], 20),
        plot_composite_phase=np.tile(["G1", "S", "G2M"], 20),
    )
    figure, axes = plt.subplot_mosaic(
        [["embedding", "composition"]],
        figsize=(7, 3),
        layout="constrained",
    )
    first = splt.embedding(
        store,
        layout_key="umap",
        color_by="cluster",
        target=axes["embedding"],
        show_legend=True,
        theme="paper",
        show=False,
    )
    second = splt.composition(
        store,
        category_by="plot_composite_phase",
        target=axes["composition"],
        show_legend=True,
        theme="paper",
        show=False,
    )

    result = splt.compose_results(
        figure,
        {"embedding": first, "composition": second},
        theme="paper",
    )

    assert result.owns_figure is False
    assert result.provenance.notes == ("composite",)
    assert "composition:aggregate" in result.tables
    assert [
        (
            legend.get_title().get_text(),
            [text.get_text() for text in legend.get_texts()],
        )
        for legend in figure.legends
    ] == [
        ("cluster", ["c1", "c2", "c3"]),
        ("plot_composite_phase", ["G1", "G2M", "S"]),
    ]
    assert all(axis.get_legend() is None for axis in axes.values())
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    legend_boxes = [legend.get_window_extent(renderer) for legend in figure.legends]
    assert not legend_boxes[0].overlaps(legend_boxes[1])
    panel_labels = [
        text
        for axis in axes.values()
        for text in axis.texts
        if text.get_text() in {"A", "B"}
    ]
    assert {text.get_fontsize() for text in panel_labels} == {
        splt.THEMES["paper"]["axes.titlesize"]
    }
    assert {
        text.get_fontsize() for legend in figure.legends for text in legend.get_texts()
    } == {splt.THEMES["paper"]["legend.fontsize"]}
    result.close()
    assert plt.fignum_exists(figure.number)
    plt.close(figure)


def test_recipe_execution_is_headless_and_read_only(imported_plot_store):
    data = imported_plot_store
    store = data.store
    columns_before = frozenset(store.cells.columns)
    artifacts_before = frozenset(store.list_artifacts())
    recipe = splt.PlotRecipe(
        (
            splt.PlotStep(
                name="overview",
                plot="embedding",
                kwargs={
                    "color_by": "RNA_nCounts",
                },
                artifact_kwargs={"layout": "umap"},
            ),
        )
    )

    execution = store.plots.run_recipe(recipe, artifacts={"umap": data.layout})

    assert not execution.written_paths
    assert not execution.failures
    assert len(execution.outputs) == 1
    assert frozenset(store.cells.columns) == columns_before
    assert frozenset(store.list_artifacts()) == artifacts_before
    result = execution.outputs[0].result
    np.testing.assert_allclose(
        result.axes["RNA_nCounts"].collections[0].get_offsets(),
        np.column_stack((np.arange(16.0), np.arange(16.0) % 4)),
    )
    assert plt.fignum_exists(result.figure.number)
    result.close()


def test_recipe_batch_output_closes_owned_figure(imported_plot_store, tmp_path):
    from PIL import Image

    recipe = splt.PlotRecipe(
        (
            splt.PlotStep(
                name="overview",
                plot="embedding",
                kwargs={
                    "color_by": "RNA_nCounts",
                    "figsize": (3.0, 2.0),
                },
                artifact_kwargs={"layout": "umap"},
                output=splt.PlotOutputSettings(
                    filename="overview.png",
                    dpi=90,
                ),
            ),
        )
    )

    execution = splt.run_recipe(
        imported_plot_store.store,
        recipe,
        artifacts={"umap": imported_plot_store.layout},
        output_dir=tmp_path,
    )
    plot_result = execution.outputs[0].result

    assert execution.written_paths == (tmp_path / "overview.png",)
    with Image.open(execution.written_paths[0]) as image:
        assert image.size == (270, 180)
    assert not plt.fignum_exists(plot_result.figure.number)


def test_stacked_violin_mean_color_expression():
    result = _stacked(
        _violin_store(),
        ["GeneA", "GeneB"],
        color_by="mean",
        color_scale=splt.ColorScale(scope="shared"),
    )
    try:
        assert len(result.axes) == 2
        assert result.legends[0].kind == "colorbar"
        assert result.legends[0].label == "mean expression"
        assert result.provenance.extras["color_by"] == "mean"
        # One scale spans every group mean: 1.15 to 9.5.
        assert (result.provenance.extras["vmin"], result.provenance.extras["vmax"]) == (
            pytest.approx(1.15),
            pytest.approx(9.5),
        )
        (colorbar,) = [
            ax for ax in result.figure.axes if ax.get_label().startswith("<colorbar")
        ]
        assert colorbar.get_ylim() == pytest.approx((1.15, 9.5))
        assert colorbar.get_ylabel() == "mean expression"
        for axis, means in zip(
            result.axes.values(), ([1.5, 5.5, 9.5], [5.15, 1.15, 3.15]), strict=True
        ):
            fractions = (np.asarray(means) - 1.15) / 8.35
            np.testing.assert_allclose(
                _violin_colors(axis), _expected_colors(fractions), atol=0.005
            )
    finally:
        result.close()


def test_stacked_violin_mean_color_explicit_bounds():
    result = _stacked(
        _violin_store(),
        "metric",
        color_by="mean",
        color_scale=splt.ColorScale(cmap="magma", vmin=0.0, vmax=10.0, scope="shared"),
    )
    try:
        color_scale = next(
            scale for scale in result.scales if isinstance(scale, splt.ColorScale)
        )
        assert (color_scale.cmap, color_scale.vmin, color_scale.vmax) == (
            "magma",
            0.0,
            10.0,
        )
        assert result.legends[0].extras == {"vmin": 0.0, "vmax": 10.0}
        assert result.provenance.extras["color_scale_scope"] == "shared"
        (colorbar,) = [
            ax for ax in result.figure.axes if ax.get_label().startswith("<colorbar")
        ]
        assert colorbar.get_ylim() == pytest.approx((0.0, 10.0))
        # Group means 1.5, 5.5 and 9.5 sit at those fractions of [0, 10].
        np.testing.assert_allclose(
            _violin_colors(result.axes["metric"]),
            _expected_colors([0.15, 0.55, 0.95], cmap="magma"),
            atol=0.005,
        )
    finally:
        result.close()


def test_stacked_violin_mean_color_constant_row():
    result = _stacked(
        _violin_store(),
        "constant",
        row_standardize=True,
        color_by="mean",
        color_scale=splt.ColorScale(scope="shared"),
    )
    try:
        table = result.tables["constant"]
        assert (table["display_value"] == 0).all()
        # Tied means follow the shared limit policy: the tied value takes the
        # low end and the colourbar still renders a unit range.
        assert result.provenance.extras["vmin"] == pytest.approx(0.0)
        assert result.provenance.extras["vmax"] == pytest.approx(1.0)
        colorbar = next(
            ax for ax in result.figure.axes if ax.get_label().startswith("<colorbar")
        )
        assert colorbar.get_ylabel() == "mean standardized value"
    finally:
        result.close()


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        (
            {"color_scale": splt.ColorScale(cmap="magma")},
            ValueError,
            "color_scale applies only when color_by='mean'",
        ),
        (
            {"color_by": "mean", "split_by": "group"},
            ValueError,
            "color_by='mean' cannot be combined with split_by",
        ),
        (
            {"color_by": "mean", "color_scale": splt.ColorScale(scale="log")},
            NotImplementedError,
            "distribution mean coloring currently supports only linear color scales",
        ),
    ],
)
def test_stacked_violin_mean_color_rejects_unsupported_options(kwargs, error, message):
    store = _synthetic_plot_store(
        I=np.ones(4, dtype=bool), group=np.array(["a", "b"] * 2), metric=np.ones(4)
    )
    with pytest.raises(error) as raised:
        splt.distribution(
            store,
            "metric",
            grouping=splt.CellField("group"),
            kind="stacked_violin",
            show=False,
            **kwargs,
        )

    assert raised.value.args == (message,)


def test_stacked_violin_mean_color_quantiles():
    result = _stacked(
        _violin_store(),
        ["metric", "metric2"],
        color_by="mean",
        color_scale=splt.ColorScale(quantiles=(0.25, 0.75), scope="shared"),
    )
    try:
        color_scale = next(
            scale for scale in result.scales if isinstance(scale, splt.ColorScale)
        )
        pooled = np.array([1.5, 5.5, 9.5, 5.15, 1.15, 3.15])
        low, high = np.quantile(pooled, [0.25, 0.75])
        assert (color_scale.vmin, color_scale.vmax) == pytest.approx((1.9125, 5.4125))
        assert (low, high) == pytest.approx((1.9125, 5.4125))
        # Means outside the quantile window clip to the colormap ends.
        for axis, means in zip(
            result.axes.values(), ([1.5, 5.5, 9.5], [5.15, 1.15, 3.15]), strict=True
        ):
            fractions = np.clip((np.asarray(means) - low) / (high - low), 0, 1)
            np.testing.assert_allclose(
                _violin_colors(axis), _expected_colors(fractions), atol=0.005
            )
    finally:
        result.close()


def test_stacked_violin_mean_color_panel_scope():
    result = _stacked(
        _violin_store(),
        ["metric", "metric2"],
        color_by="mean",
        color_scale=splt.ColorScale(scope="panel"),
    )
    try:
        color_scale = next(
            scale for scale in result.scales if isinstance(scale, splt.ColorScale)
        )
        assert color_scale.scope == "panel"
        assert result.provenance.extras["color_scale_scope"] == "panel"
        # Panel scope draws one reference colorbar on the unit relative scale.
        colorbars = [
            ax for ax in result.figure.axes if ax.get_label().startswith("<colorbar")
        ]
        assert len(colorbars) == 1
        assert colorbars[0].get_ylabel() == "Relative Value Per Key"
        assert (color_scale.vmin, color_scale.vmax) == (0.0, 1.0)
        assert result.legends[0].extras == {"vmin": 0.0, "vmax": 1.0}
        # Each row rescales its own group means to [0, 1].
        np.testing.assert_allclose(
            _violin_colors(result.axes["metric"]),
            _expected_colors([0.0, 0.5, 1.0]),
            atol=0.005,
        )
        np.testing.assert_allclose(
            _violin_colors(result.axes["metric2"]),
            _expected_colors([1.0, 0.0, 0.5]),
            atol=0.005,
        )
    finally:
        result.close()


def test_stacked_violin_scope_follows_share_y():
    independent = _stacked(_violin_store(), "GeneA", color_by="mean")
    try:
        assert independent.provenance.extras["color_scale_scope"] == "panel"
        assert independent.legends[0].label == "Relative Expression Per Gene"
    finally:
        independent.close()
    shared = _stacked(_violin_store(), "GeneA", color_by="mean", share_y=True)
    try:
        assert shared.provenance.extras["color_scale_scope"] == "shared"
        assert shared.legends[0].label == "mean expression"
        assert (shared.provenance.extras["vmin"], shared.provenance.extras["vmax"]) == (
            pytest.approx(1.5),
            pytest.approx(9.5),
        )
    finally:
        shared.close()


def test_stacked_violin_mean_color_default_scale_scope_is_ergonomic():
    result = _stacked(
        _violin_store(),
        "metric2",
        color_by="mean",
        color_scale=splt.ColorScale(cmap="magma"),
    )
    try:
        scale = next(s for s in result.scales if isinstance(s, splt.ColorScale))
        # The general "feature" default resolves to panel scope here.
        assert (scale.scope, scale.cmap) == ("panel", "magma")
        np.testing.assert_allclose(
            _violin_colors(result.axes["metric2"]),
            _expected_colors([1.0, 0.0, 0.5], cmap="magma"),
            atol=0.005,
        )
    finally:
        result.close()


def test_stacked_violin_mean_color_no_colorbar_on_target():
    fig, axes = plt.subplots(1, 2)
    result = _stacked(
        _violin_store(),
        ["metric", "metric2"],
        color_by="mean",
        color_scale=splt.ColorScale(scope="shared"),
        target=[axes[0], axes[1]],
    )
    try:
        assert result.owns_figure is False
        assert list(result.axes.values()) == [axes[0], axes[1]]
        assert fig.axes == [axes[0], axes[1]]
        assert result.legends[0].kind == "colorbar"
        assert result.legends[0].extras == {
            "vmin": pytest.approx(1.15),
            "vmax": pytest.approx(9.5),
        }
    finally:
        result.close()
        plt.close(fig)


def test_stacked_violin_mean_color_missing_group_missing_color():
    from scarf.plotting.distribution import _mean_group_palette

    means = pd.Series({"a": 1.0, "b": np.nan})
    color_scale = splt.ColorScale(scope="shared")
    palette = _mean_group_palette(
        means,
        ["a", "b", "c"],
        color_scale=color_scale,
        lo=0.0,
        hi=2.0,
    )
    assert palette["b"] == color_scale.missing_color
    assert palette["c"] == color_scale.missing_color
    # 1.0 sits halfway between the limits.
    assert mcolors.to_hex(palette["a"]) == mcolors.to_hex(plt.get_cmap("viridis")(0.5))


def test_stacked_violin_mean_color_vcenter_extends_bounds():
    result = _stacked(
        _violin_store(),
        "metric",
        color_by="mean",
        color_scale=splt.ColorScale(vcenter=0.0, scope="shared"),
    )
    try:
        scale = next(s for s in result.scales if isinstance(s, splt.ColorScale))
        # All means are positive, so the low limit moves just below the pivot.
        assert scale.vmin < 0.0
        assert scale.vmin == pytest.approx(0.0, abs=1e-4)
        assert scale.vmax == pytest.approx(9.5)
        # Above the pivot, values fill the upper half of the colormap.
        fractions = 0.5 + 0.5 * np.array([1.5, 5.5, 9.5]) / 9.5
        np.testing.assert_allclose(
            _violin_colors(result.axes["metric"]),
            _expected_colors(fractions),
            atol=0.005,
        )
    finally:
        result.close()


def test_stacked_violin_explicit_none_uses_no_overlay():
    result = splt.distribution(
        _violin_store(),
        "metric",
        grouping=splt.CellField("group"),
        kind="stacked_violin",
        max_points=None,
        show=False,
    )
    try:
        assert result.provenance.extras["max_points"] == 0
        # Only the three violin bodies are drawn; no point overlay.
        collections = result.axes["metric"].collections
        assert len(collections) == 3
        assert all(len(body.get_offsets()) <= 1 for body in collections)
    finally:
        result.close()


@pytest.mark.parametrize(
    "scale",
    [
        splt.ColorScale(scope="panel", vmin=0.0),
        splt.ColorScale(scope="panel", quantiles=(0.1, 0.9)),
        splt.ColorScale(scope="panel", vcenter=0.0),
    ],
)
def test_stacked_violin_panel_scope_rejects_bounds(scale):
    store = _synthetic_plot_store(
        I=np.ones(4, dtype=bool), group=np.array(["a", "b"] * 2), metric=np.ones(4)
    )
    with pytest.raises(ValueError, match="apply only to scope='shared'"):
        splt.distribution(
            store,
            "metric",
            grouping=splt.CellField("group"),
            kind="stacked_violin",
            color_by="mean",
            color_scale=scale,
            max_points=0,
            show=False,
        )


def test_stacked_violin_sparse_quantile_limits_preserve_outlier_color():
    from scarf.plotting.distribution import _mean_color_limits, _mean_group_palette

    means = pd.Series({"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0, "e": 10.0})
    scale = splt.ColorScale(scope="shared", quantiles=(0.25, 0.75))

    limits, reference = _mean_color_limits([means], scale)
    palette = _mean_group_palette(
        means,
        list(means.index),
        color_scale=scale,
        lo=limits[0][0],
        hi=limits[0][1],
    )

    # Collapsed quantiles keep the tied mean at the low end while the outlying
    # mean clips to the high end.
    assert limits == [(0.0, 1.0)]
    assert reference == (0.0, 1.0)
    assert mcolors.to_hex(palette["a"]) == "#440154"
    assert mcolors.to_hex(palette["e"]) == "#fde725"


def test_stacked_violin_mean_color_honors_hidden_legend_and_generic_label():
    store = _synthetic_plot_store(
        I=np.ones(12, dtype=bool),
        group=np.repeat(["a", "b", "c"], 4),
        metric=np.arange(12, dtype=float),
    )
    result = splt.distribution(
        store,
        "metric",
        grouping=splt.CellField("group"),
        kind="stacked_violin",
        color_by="mean",
        color_scale=splt.ColorScale(scope="shared"),
        max_points=0,
        show_legend=False,
        show=False,
    )
    try:
        assert len(result.figure.axes) == 1
        assert result.legends[0].label == "mean value"
    finally:
        result.close()


def test_stacked_violin_public_default_keeps_point_overlay():
    store = _synthetic_plot_store(
        I=np.ones(12, dtype=bool),
        group=np.repeat(["a", "b", "c"], 4),
        metric=np.arange(12, dtype=float),
    )
    result = splt.distribution(
        store,
        "metric",
        grouping=splt.CellField("group"),
        kind="stacked_violin",
        show=False,
    )
    try:
        assert result.provenance.extras["max_points"] == 10000
        assert result.provenance.extras["grouping"] == {
            "type": "cell_field",
            "key": "group",
            "kind": "auto",
            "label": None,
        }
        assert result.provenance.extras["cell_selection"] is None
        assert any(
            len(collection.get_offsets()) > 1
            for collection in result.axes["metric"].collections
            if hasattr(collection, "get_offsets")
        )
    finally:
        result.close()


def test_distribution_public_signature_preserves_compatibility_defaults():
    from inspect import signature

    from scarf.datastore._plot_accessor import DataStorePlotAccessor

    function_parameters = signature(splt.distribution).parameters
    accessor_parameters = signature(DataStorePlotAccessor.distribution).parameters
    assert function_parameters["max_points"].default == 10000
    assert accessor_parameters["max_points"].default == 10000
    assert "grouping" in function_parameters
    assert "grouping" in accessor_parameters
    assert "cell_selection" in function_parameters
    assert "cell_selection" in accessor_parameters
    assert "group_by" not in function_parameters
    assert "group_by" not in accessor_parameters
    assert "cell_key" not in function_parameters
    assert "cell_key" not in accessor_parameters
    assert "stats_method" not in function_parameters
    assert "stats_method" not in accessor_parameters


@pytest.mark.parametrize(
    ("orientation", "posthoc_table", "expected_text"),
    [
        ("vertical", None, "p=0.02"),
        (
            "horizontal",
            pd.DataFrame({"group_1": ["a"], "group_2": ["c"], "p_value": [0.01]}),
            "p=0.01",
        ),
    ],
)
def test_distribution_annotates_kruskal_omnibus_and_dunn_posthoc(
    orientation,
    posthoc_table,
    expected_text,
):
    store = _synthetic_plot_store(
        I=np.ones(12, dtype=bool),
        group=np.repeat(["a", "b", "c"], 4),
        metric=np.arange(12, dtype=float),
    )
    result_table = pd.DataFrame(
        {"kruskal_statistic": [7.0], "df": [2.0], "p_value": [0.02]}
    )
    stats = _synthetic_stats_result(
        store,
        result_table,
        method="kruskal_wallis",
        posthoc_table=posthoc_table,
    )
    result = splt.distribution(
        store,
        "metric",
        grouping=splt.CellField("group"),
        orientation=orientation,
        max_points=0,
        stats_results=stats,
        show=False,
    )
    try:
        assert expected_text in [
            text.get_text() for text in result.axes["metric"].texts
        ]
        assert result.provenance.extras["stats_annotated"] is True
    finally:
        result.close()


def test_distribution_stats_rejects_same_size_different_identity():
    store = _synthetic_plot_store(
        I=np.ones(12, dtype=bool),
        group=np.repeat(["a", "b", "c"], 4),
        metric=np.arange(12, dtype=float),
    )
    table = pd.DataFrame({"group_1": ["a"], "group_2": ["b"], "p_value": [0.01]})
    stats = _synthetic_stats_result(store, table)
    stats.cell_selection_fingerprint = "different-selection"

    with pytest.warns(UserWarning, match="cell selection does not match"):
        result = splt.distribution(
            store,
            "metric",
            grouping=splt.CellField("group"),
            max_points=0,
            stats_results=stats,
            show=False,
        )
    try:
        assert result.provenance.extras["stats_annotated"] is False
        assert not result.axes["metric"].texts
    finally:
        result.close()

    incomplete_stats = _synthetic_stats_result(store, table)
    incomplete_stats.cell_selection_fingerprint = None
    with pytest.warns(UserWarning, match="does not include cell-selection identity"):
        result = splt.distribution(
            store,
            "metric",
            grouping=splt.CellField("group"),
            max_points=0,
            stats_results=incomplete_stats,
            show=False,
        )
    result.close()


def _three_group_metric_store():
    return _synthetic_plot_store(
        I=np.ones(12, dtype=bool),
        group=np.repeat(["a", "b", "c"], 4),
        metric=np.arange(12, dtype=float),
    )


def _plot_metric_with_stats(store, stats):
    return splt.distribution(
        store,
        "metric",
        grouping=splt.CellField("group"),
        max_points=0,
        stats_results=stats,
        show=False,
    )


def test_distribution_stats_skip_panels_without_a_result_or_table():
    store = _three_group_metric_store()
    table = pd.DataFrame({"group_1": ["a"], "group_2": ["b"], "p_value": [0.01]})
    stats = _synthetic_stats_result(store, table)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        unmatched = _plot_metric_with_stats(store, {"other_panel": stats})
        # The panel keeps its tested identity but carries no table to draw.
        stats.tables = {"metric": None}
        untabled = _plot_metric_with_stats(store, stats)
    try:
        for result in (unmatched, untabled):
            assert result.provenance.extras["stats_annotated"] is False
            assert len(result.axes["metric"].texts) == 0
    finally:
        unmatched.close()
        untabled.close()

    stats.tables = {"metric": table}
    annotated = _plot_metric_with_stats(store, {"metric": stats})
    try:
        assert annotated.provenance.extras["stats_annotated"] is True
        assert [text.get_text() for text in annotated.axes["metric"].texts] == [
            "p=0.01"
        ]
    finally:
        annotated.close()


@pytest.mark.parametrize(
    "table",
    [
        pd.DataFrame({"group_1": ["a"], "group_2": ["b"], "statistic": [3.0]}),
        pd.DataFrame({"statistic": [3.0], "p_value": [0.01]}),
    ],
    ids=["no-p-value", "no-group-columns"],
)
def test_distribution_stats_warn_when_no_table_can_be_annotated(table):
    store = _three_group_metric_store()
    stats = _synthetic_stats_result(store, table, method="welch")

    with pytest.warns(UserWarning) as warned:
        result = _plot_metric_with_stats(store, stats)
    try:
        assert [str(warning.message) for warning in warned] == [
            "stats_results contains no supported pairwise or omnibus annotation "
            "table for method 'welch'; skipping statistical annotations"
        ]
        assert result.provenance.extras["stats_annotated"] is False
        assert len(result.axes["metric"].texts) == 0
    finally:
        result.close()


def test_distribution_excludes_cells_whose_subset_flag_is_masked():
    class MaskedSubsetCells(_SyntheticCells):
        def _get_missing_mask_array(self, column):
            if column == "subset":
                return np.array([False, True, False, False, False, True])
            return None

    store = SimpleNamespace(
        cells=MaskedSubsetCells(
            I=np.ones(6, dtype=bool),
            group=np.repeat(["a", "b"], 3),
            subset=np.array([True, True, False, True, True, True]),
            metric=np.arange(6, dtype=float),
        ),
        _defaultAssay="RNA",
        zw=None,
        _stored_display_metadata=lambda _column: None,
    )

    result = splt.distribution(
        store,
        "metric",
        grouping=splt.CellField("group"),
        subset_by="subset",
        kind="box",
        max_points=0,
        show=False,
    )
    try:
        # Cell 2 is unflagged and cells 1 and 5 hold masked placeholder flags.
        table = result.tables["metric"]
        assert table["value"].tolist() == [0.0, 3.0, 4.0]
        assert table["group"].tolist() == ["a", "b", "b"]
        assert result.provenance.n_cells == 3
    finally:
        result.close()


def test_distribution_stats_rejects_changed_assay_normalization_state():
    from scarf.features.statistical import value_fingerprint
    from scarf.plotting.distribution import _stat_result_compatibility_issue

    store = _synthetic_plot_store(
        I=np.ones(12, dtype=bool),
        group=np.repeat(["a", "b", "c"], 4),
        metric=np.arange(12, dtype=float),
    )
    table = pd.DataFrame({"group_1": ["a"], "group_2": ["b"], "p_value": [0.01]})
    stats = _synthetic_stats_result(store, table)
    stats.source_assays = ("RNA",)
    stats.normalization = {"source": "assay", "transform": "none"}
    stats.normalization_method = {"module": "old", "qualname": "normalize"}
    stats.size_factor = 1_000.0
    values = store.cells.fetch("metric")
    groups = store.cells.fetch("group")
    cells = store.cells.active_index("I")

    def compatibility_issue(*, method, size_factor):
        return _stat_result_compatibility_issue(
            stats,
            label="metric",
            expected_identity=stats.tested_features[0],
            expected_value_fingerprint=stats.value_fingerprints[0],
            expected_source_assay="RNA",
            grouping=splt.CellField("group"),
            cell_selection=None,
            n_cells=len(values),
            n_groups=3,
            group_order=("a", "b", "c"),
            sample_by=None,
            pair_by=None,
            sample_fingerprint=None,
            pair_fingerprint=None,
            sample_stat="mean",
            expression_cutoff=0.0,
            normalization=splt.NormalizationSpec(),
            normalization_method=method,
            size_factor=size_factor,
            cell_selection_fingerprint=value_fingerprint(cells),
            group_fingerprint=value_fingerprint(groups),
        )

    assert "normalization method" in compatibility_issue(
        method={"module": "new", "qualname": "normalize"},
        size_factor=1_000.0,
    )
    assert "size factor" in compatibility_issue(
        method=stats.normalization_method,
        size_factor=2_000.0,
    )


def test_distribution_stats_and_plot_drop_the_same_invalid_group_labels():
    from scarf.features.statistical import value_fingerprint

    store = _synthetic_plot_store(
        I=np.ones(8, dtype=bool),
        group=np.array(["a", "a", "b", "b", None, "", "   ", np.nan], dtype=object),
        metric=np.arange(8, dtype=float),
    )
    table = pd.DataFrame({"group_1": ["a"], "group_2": ["b"], "p_value": [0.01]})
    stats = _synthetic_stats_result(store, table)
    retained = np.arange(4, dtype=np.int64)
    stats.n_cells = 4
    stats.n_groups = 2
    stats.cell_selection_fingerprint = value_fingerprint(retained)
    stats.group_fingerprint = value_fingerprint(store.cells.fetch("group")[:4])
    stats.group_order = ("a", "b")
    stats.value_fingerprints = (value_fingerprint(store.cells.fetch("metric")[:4]),)

    result = splt.distribution(
        store,
        "metric",
        grouping=splt.CellField("group"),
        max_points=0,
        stats_results=stats,
        show=False,
    )
    try:
        assert result.provenance.n_cells == 4
        assert result.provenance.extras["dropped_group_cells"] == 4
        assert result.provenance.extras["stats_annotated"] is True
    finally:
        result.close()


def test_distribution_stats_rejects_sample_and_split_mismatches():
    store = _synthetic_plot_store(
        I=np.ones(12, dtype=bool),
        group=np.repeat(["a", "b", "c"], 4),
        split=np.tile(["x", "y"], 6),
        sample=np.repeat(["s1", "s2", "s3", "s4"], 3),
        pair=np.repeat(["p1", "p2", "p3", "p4"], 3),
        metric=np.arange(12, dtype=float),
    )
    table = pd.DataFrame({"group_1": ["a"], "group_2": ["b"], "p_value": [0.01]})
    sample_stats = _synthetic_stats_result(store, table, sample_by="sample")

    with pytest.warns(UserWarning, match="sample_by does not match"):
        result = splt.distribution(
            store,
            "metric",
            grouping=splt.CellField("group"),
            max_points=0,
            stats_results=sample_stats,
            show=False,
        )
    result.close()

    sample_identity_stats = _synthetic_stats_result(
        store,
        table,
        sample_by="sample",
    )
    sample_identity_stats.sample_fingerprint = "different-samples"
    with pytest.warns(UserWarning, match="sample values do not match"):
        result = splt.distribution(
            store,
            "metric",
            grouping=splt.CellField("group"),
            sample_by="sample",
            max_points=0,
            stats_results=sample_identity_stats,
            show=False,
        )
    result.close()

    paired_stats = _synthetic_stats_result(
        store,
        table,
        sample_by="sample",
        pair_by="pair",
    )
    with pytest.warns(UserWarning, match="pair_by does not match"):
        result = splt.distribution(
            store,
            "metric",
            grouping=splt.CellField("group"),
            sample_by="sample",
            max_points=0,
            stats_results=paired_stats,
            show=False,
        )
    result.close()

    # A study-design pairing column applies only to paired Wilcoxon results.
    matching_paired_stats = _synthetic_stats_result(
        store,
        table,
        method="wilcoxon",
        sample_by="sample",
        pair_by="pair",
    )
    result = splt.distribution(
        store,
        "metric",
        grouping=splt.CellField("group"),
        study_design=splt.StudyDesign(sample_by="sample", subject_by="pair"),
        max_points=0,
        stats_results=matching_paired_stats,
        show=False,
    )
    try:
        assert result.provenance.extras["stats_annotated"] is True
        assert result.provenance.extras["pair_by"] == "pair"
    finally:
        result.close()

    independent_sample_stats = _synthetic_stats_result(
        store,
        table,
        method="mann_whitney",
        sample_by="sample",
    )
    result = splt.distribution(
        store,
        "metric",
        grouping=splt.CellField("group"),
        study_design=splt.StudyDesign(sample_by="sample", subject_by="pair"),
        max_points=0,
        stats_results=independent_sample_stats,
        show=False,
    )
    try:
        assert result.provenance.extras["stats_annotated"] is True
        assert result.provenance.extras["pair_by"] is None
    finally:
        result.close()

    with pytest.raises(ValueError, match="cannot be combined with split_by"):
        splt.distribution(
            store,
            "metric",
            grouping=splt.CellField("group"),
            split_by="split",
            stats_results=sample_stats,
            show=False,
        )


@pytest.mark.parametrize("orientation", ["vertical", "horizontal"])
def test_distribution_stats_preserve_shared_value_axis(orientation):
    store = _synthetic_plot_store(
        I=np.ones(12, dtype=bool),
        group=np.repeat(["a", "b", "c"], 4),
        metric=np.arange(12, dtype=float),
        metric2=np.arange(12, dtype=float) * 10,
    )
    table = pd.DataFrame({"group_1": ["a"], "group_2": ["b"], "p_value": [0.01]})
    stats = _synthetic_stats_result(store, table)

    result = splt.distribution(
        store,
        ["metric", "metric2"],
        grouping=splt.CellField("group"),
        orientation=orientation,
        share_y=True,
        max_points=0,
        stats_results=stats,
        stats_keys=["metric"],
        show=False,
    )
    try:
        limits = [
            axis.get_ylim() if orientation == "vertical" else axis.get_xlim()
            for axis in result.axes.values()
        ]
        assert limits[0] == pytest.approx(limits[1])
    finally:
        result.close()


def test_distribution_study_design_pair_is_not_resolved_without_stats():
    store = _synthetic_plot_store(
        I=np.ones(8, dtype=bool),
        group=np.repeat(["a", "b"], 4),
        sample=np.repeat(["s1", "s2", "s3", "s4"], 2),
        pair=np.array(["p1", "p1", "p2", "p2", "p3", "p3", None, None]),
        metric=np.arange(8, dtype=float),
    )

    result = splt.distribution(
        store,
        "metric",
        grouping=splt.CellField("group"),
        study_design=splt.StudyDesign(sample_by="sample", subject_by="pair"),
        max_points=0,
        show=False,
    )
    try:
        assert result.provenance.n_cells == 8
        assert set(result.tables["metric"]["sample"]) == {"s1", "s2", "s3", "s4"}
        assert result.provenance.extras["pair_by"] is None
        assert result.provenance.extras["dropped_pair_cells"] == 0
    finally:
        result.close()


def test_distribution_paired_stats_reject_missing_pair_values():
    store = _synthetic_plot_store(
        I=np.ones(8, dtype=bool),
        group=np.repeat(["a", "b"], 4),
        sample=np.repeat(["s1", "s2", "s3", "s4"], 2),
        pair=np.array(["p1", "p1", "p2", "p2", "p3", "p3", None, None]),
        metric=np.arange(8, dtype=float),
    )
    table = pd.DataFrame({"group_1": ["a"], "group_2": ["b"], "p_value": [0.01]})
    stats = _synthetic_stats_result(
        store,
        table,
        method="wilcoxon",
        sample_by="sample",
        pair_by="pair",
    )

    with pytest.raises(ValueError, match="valid pair value for every cell"):
        splt.distribution(
            store,
            "metric",
            grouping=splt.CellField("group"),
            study_design=splt.StudyDesign(sample_by="sample", subject_by="pair"),
            max_points=0,
            stats_results=stats,
            show=False,
        )


def test_distribution_stats_annotations_use_custom_dark_theme_foreground(monkeypatch):
    theme_name = "test-distribution-custom-dark"
    monkeypatch.setitem(
        splt.THEMES,
        theme_name,
        {**splt.THEMES["dark"], "font.size": 9},
    )
    store = _synthetic_plot_store(
        I=np.ones(12, dtype=bool),
        group=np.repeat(["a", "b", "c"], 4),
        metric=np.arange(12, dtype=float),
    )
    table = pd.DataFrame({"group_1": ["a"], "group_2": ["b"], "p_value": [0.01]})
    stats = _synthetic_stats_result(store, table)

    result = splt.distribution(
        store,
        "metric",
        grouping=splt.CellField("group"),
        theme=theme_name,
        max_points=0,
        stats_results=stats,
        show=False,
    )
    try:
        bracket = next(
            line for line in result.axes["metric"].lines if len(line.get_xdata()) == 4
        )
        assert bracket.get_color() == "#e8e8e8"
        assert result.axes["metric"].texts[0].get_color() == "#e8e8e8"
    finally:
        result.close()


def test_distribution_stats_rejects_changed_realized_values():
    store = _synthetic_plot_store(
        I=np.ones(12, dtype=bool),
        group=np.repeat(["a", "b", "c"], 4),
        metric=np.arange(12, dtype=float),
    )
    table = pd.DataFrame({"group_1": ["a"], "group_2": ["b"], "p_value": [0.01]})
    stats = _synthetic_stats_result(store, table)
    stats.value_fingerprints = ("different-values",)

    with pytest.warns(UserWarning, match="realized values do not match"):
        result = splt.distribution(
            store,
            "metric",
            grouping=splt.CellField("group"),
            max_points=0,
            stats_results=stats,
            show=False,
        )
    try:
        assert result.provenance.extras["stats_annotated"] is False
        assert not result.axes["metric"].texts
    finally:
        result.close()


@pytest.mark.parametrize("height", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_distribution_stats_bracket_height_requires_finite_positive_value(height):
    store = _synthetic_plot_store(
        I=np.ones(6, dtype=bool),
        group=np.repeat(["a", "b"], 3),
        metric=np.arange(6, dtype=float),
    )

    with pytest.raises(ValueError, match="finite and positive"):
        splt.distribution(
            store,
            "metric",
            grouping=splt.CellField("group"),
            stats_results=object(),
            stats_bracket_height=height,
            show=False,
        )


def test_distribution_masks_metadata_placeholders_per_panel():
    from scarf.features.statistical import value_fingerprint
    from scarf.plotting.distribution import _fetch_series
    from scarf.storage.artifacts import provenance_hash

    class MaskedCells(_SyntheticCells):
        def __init__(self, missing_masks, **columns):
            super().__init__(**columns)
            self.missing_masks = {
                key: np.asarray(value, dtype=bool)
                for key, value in missing_masks.items()
            }

        def _get_missing_mask_array(self, column):
            return self.missing_masks.get(column)

    columns = {
        "I": np.ones(6, dtype=bool),
        "group": np.repeat(["a", "b"], 3),
        "sample": np.repeat(["s1", "s2"], 3),
        "metric": np.array([1.0, 999.0, 0.0, 0.0, 2.0, 2.0]),
    }
    masked_store = SimpleNamespace(
        cells=MaskedCells(
            {"metric": np.array([False, True, False, False, False, False])},
            **columns,
        ),
        _defaultAssay="RNA",
        zw=None,
        _stored_display_metadata=lambda _column: None,
    )
    plain_store = _synthetic_plot_store(**columns)
    masked_values, _label, _is_feature, masked_identity, _assay = _fetch_series(
        masked_store,
        "metric",
        metadata_columns=set(masked_store.cells.columns),
        cell_indices=np.arange(6, dtype=np.int64),
        from_assay=None,
        normalization=splt.NormalizationSpec(),
    )
    _values, _label, _is_feature, plain_identity, _assay = _fetch_series(
        plain_store,
        "metric",
        metadata_columns=set(plain_store.cells.columns),
        cell_indices=np.arange(6, dtype=np.int64),
        from_assay=None,
        normalization=splt.NormalizationSpec(),
    )
    assert np.isnan(masked_values[1])
    assert masked_identity == provenance_hash(
        {
            "source": "cell_metadata",
            "column": "metric",
            "values_fingerprint": value_fingerprint(columns["metric"]),
            "missing_fingerprint": value_fingerprint(
                np.array([False, True, False, False, False, False])
            ),
        }
    )
    assert masked_identity != plain_identity

    result = splt.distribution(
        masked_store,
        "metric",
        grouping=splt.CellField("group"),
        sample_by="sample",
        sample_stat="fraction",
        expression_cutoff=0.0,
        max_points=0,
        show=False,
    )
    try:
        sample_a = result.tables["metric"].set_index("sample").loc["s1"]
        assert sample_a["value"] == pytest.approx(0.5)
        assert sample_a["nCells"] == 2
    finally:
        result.close()


def test_distribution_masked_subset_still_requires_boolean_dtype():
    class MaskedSubsetCells(_SyntheticCells):
        def _get_missing_mask_array(self, column):
            if column == "subset":
                return np.zeros(self.N, dtype=bool)
            return None

    store = SimpleNamespace(
        cells=MaskedSubsetCells(
            I=np.ones(6, dtype=bool),
            group=np.repeat(["a", "b"], 3),
            subset=np.array([0, 1, 1, 0, 1, 1], dtype=np.int64),
            metric=np.arange(6, dtype=float),
        ),
        _defaultAssay="RNA",
        zw=None,
        _stored_display_metadata=lambda _column: None,
    )

    with pytest.raises(TypeError, match="must be boolean"):
        splt.distribution(
            store,
            "metric",
            grouping=splt.CellField("group"),
            subset_by="subset",
            max_points=0,
            show=False,
        )


@pytest.mark.parametrize("infinite", [np.inf, -np.inf])
def test_distribution_panel_rejects_infinite_values(infinite):
    from scarf.plotting.distribution import _panel_display_frame

    with pytest.raises(ValueError, match="infinite entries"):
        _panel_display_frame(
            np.array([0.0, infinite]),
            np.array(["a", "a"], dtype=object),
            split_arr=None,
            sample_arr=np.array(["s1", "s1"], dtype=object),
            sample_stat="fraction",
            expression_cutoff=0.0,
            row_standardize=False,
        )


class _MaskedSyntheticCells(_SyntheticCells):
    def __init__(self, missing_masks, **columns):
        super().__init__(**columns)
        self._missing = {
            key: np.asarray(value, dtype=bool) for key, value in missing_masks.items()
        }

    def _get_missing_mask_array(self, column):
        return self._missing.get(column)


def _masked_plot_store(missing_masks, **columns):
    store = _synthetic_plot_store(**columns)
    store.cells = _MaskedSyntheticCells(missing_masks, **columns)
    return store


def test_composition_shows_masked_categories_and_samples_as_missing():
    store = _masked_plot_store(
        {
            "cluster": [False, True, False, False, False, True],
            "sample": [False, False, False, True, False, False],
        },
        I=np.ones(6, dtype=bool),
        cluster=np.array([0, 0, 1, 0, 1, 0]),
        sample=np.array(["s1", "s1", "s1", "", "s2", "s2"]),
    )

    result = splt.composition(store, category_by="cluster", show=False)
    try:
        aggregate = result.tables["aggregate"]
        assert aggregate["category"].tolist() == [0, 1, None]
        np.testing.assert_allclose(aggregate["proportion"], [1 / 3] * 3)
    finally:
        result.close()

    result = splt.composition(
        store, category_by="cluster", sample_by="sample", show=False
    )
    try:
        assert set(result.tables["per_sample"]["sample"]) == {"s1", "s2"}
        assert result.provenance.extras["dropped_sample_cells"] == 1
    finally:
        result.close()


def test_masked_metadata_is_missing_in_embedding_and_connectivity_inputs():
    from scarf.plotting.cluster_connectivity import _fetch_inputs
    from scarf.plotting.embedding import (
        _multi_layout_facets,
        _prefetch_colors,
        _selected_metadata_column,
    )

    missing = [False, True, False, False]
    store = _masked_plot_store(
        {"donor": missing, "depth": missing, "flag": missing, "umap1": missing},
        I=np.ones(4, dtype=bool),
        donor=np.array([1, 0, 2, 1]),
        depth=np.array([5, 0, 7, 9]),
        flag=np.array([True, False, True, True]),
        umap1=np.array([0.0, 0.0, 1.0, 2.0]),
        umap2=np.array([0.0, 1.0, 1.0, 2.0]),
    )

    (donor, _label, donor_is_categorical, _uniform), (depth, *_rest) = _prefetch_colors(
        store,
        ["donor", splt.CellField("depth", kind="continuous")],
        metadata_columns=store.cells.columns,
        from_assay=None,
        cell_key="I",
        n_cells=4,
        normalization=splt.NormalizationSpec(),
    )
    assert donor_is_categorical is True
    assert donor.tolist() == [1, None, 2, 1]
    np.testing.assert_array_equal(depth, [5.0, np.nan, 7.0, 9.0])
    flag = _selected_metadata_column(store, "flag", cell_key="I", cell_indices=None)
    assert flag.tolist() == [True, False, True, True]
    assert _multi_layout_facets(
        store,
        facet_by="donor",
        facet_order=None,
        groups=None,
        subset_by=None,
        cell_key="I",
    ) == [1, 2, None]

    with pytest.raises(ValueError, match="non-finite coordinates"):
        _fetch_inputs(store, group_by="donor", layout_key="umap", cell_key="I")
    store.cells._missing.pop("umap1")
    with pytest.raises(ValueError, match="'donor' contains missing values"):
        _fetch_inputs(store, group_by="donor", layout_key="umap", cell_key="I")


def test_grouping_plots_exclude_masked_labels_and_samples(tmp_path):
    from tests.storage_helpers import insert_nullable_cell_column
    from tests.test_quality_control_missing_values import (
        import_nullable_cluster_h5ad,
    )

    store, result, codes, missing = import_nullable_cluster_h5ad(tmp_path)
    clusters = result.clusterArtifacts["clusters"]
    donor_missing = np.zeros(store.cells.N, dtype=bool)
    donor_missing[1::7] = True
    donor = np.where(donor_missing, 0, np.arange(store.cells.N) % 2 + 1)
    insert_nullable_cell_column(store, "donor", donor.astype(np.int64), donor_missing)
    gene = str(store.RNA.feats.fetch_all("names")[0])
    members = {group: int(((codes == group) & ~missing).sum()) for group in range(3)}

    for plot in (splt.dotplot, splt.matrixplot):
        plotted = plot(store, features=[gene], groups=clusters, show=False)
        try:
            aggregate = plotted.tables["aggregate"]
            assert dict(zip(aggregate["groups"], aggregate["n_cells"])) == members
            assert plotted.provenance.extras["dropped_group_cells"] == missing.sum()
        finally:
            plotted.close()

    plotted = splt.dotplot(store, features=[gene], group_by="donor", show=False)
    try:
        aggregate = plotted.tables["aggregate"]
        assert dict(zip(aggregate["donor"], aggregate["n_cells"])) == {
            label: int(((donor == label) & ~donor_missing).sum()) for label in (1, 2)
        }
    finally:
        plotted.close()

    plotted = splt.dotplot(
        store, features=[gene], groups=clusters, sample_by="donor", show=False
    )
    try:
        assert plotted.provenance.n_samples == 2
        assert plotted.provenance.extras["dropped_sample_cells"] == donor_missing.sum()
        assert set(plotted.tables["per_sample"]["sample"]) == {1, 2}
    finally:
        plotted.close()

    plotted = splt.composition(store, categories=clusters, show=False)
    try:
        aggregate = plotted.tables["aggregate"]
        assert aggregate["category"].tolist() == [0, 1, 2, None]
        np.testing.assert_allclose(
            aggregate["proportion"],
            [*(np.array(list(members.values())) / len(codes)), missing.mean()],
        )
    finally:
        plotted.close()

    plotted = splt.composition(
        store, categories=clusters, sample_by="donor", show=False
    )
    try:
        per_sample = plotted.tables["per_sample"]
        assert set(per_sample["sample"]) == {1, 2}
        assert per_sample["category"].tolist()[-2:] == [None, None]
        assert plotted.provenance.extras["dropped_sample_cells"] == donor_missing.sum()
    finally:
        plotted.close()

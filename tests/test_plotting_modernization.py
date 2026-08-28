"""Focused behavior tests for publication plotting features."""

from importlib import import_module
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import scarf.plotting as splt
from scarf.plotting._style import default_point_size, resolve_legend_loc


class _SyntheticCells:
    def __init__(self, **columns):
        self._columns = {name: np.asarray(values) for name, values in columns.items()}
        self.columns = tuple(self._columns)
        self.N = len(next(iter(self._columns.values())))

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
    from scarf.plotting.distribution import _value_fingerprint
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
        group_key="group",
        cell_key="I",
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
                    "values_fingerprint": _value_fingerprint(values),
                    "missing_fingerprint": None,
                }
            ),
        ),
        value_fingerprints=(_value_fingerprint(np.asarray(values, dtype=np.float64)),),
        summary_scope="sample" if sample_by is not None else "cell",
        tables={"metric": table},
        posthoc_tables=({"metric": posthoc_table} if posthoc_table is not None else {}),
        cell_selection=None,
        cell_selection_fingerprint=_value_fingerprint(cell_selection),
        group_fingerprint=_value_fingerprint(groups),
        group_order=tuple(sorted(unique_groups, key=str)),
        normalization={},
        normalization_method=None,
        size_factor=None,
        source_assays=(None,),
        sample_fingerprint=(
            _value_fingerprint(samples) if samples is not None else None
        ),
        pair_fingerprint=_value_fingerprint(pairs) if pairs is not None else None,
        artifact=None,
    )


def test_point_size_uses_population_and_panel_area():
    assert default_point_size(500, panel_area=16) > default_point_size(
        5_000,
        panel_area=16,
    )
    assert default_point_size(5_000, panel_area=16) > default_point_size(
        5_000,
        panel_area=4,
    )
    assert resolve_legend_loc(80) == "right"


def test_register_theme_extends_rcparams():
    name = "test-publication-theme"
    splt.register_theme(name, {"font.size": 7.5}, base="paper")
    with pytest.raises(ValueError, match="already exists"):
        splt.register_theme(name, {"font.size": 8})
    with splt.theme_context(name):
        assert matplotlib.rcParams["font.size"] == pytest.approx(7.5)


def test_stored_display_metadata_does_not_hide_malformed_stores():
    from scarf.plotting._display import stored_display_metadata

    class MissingPlotStore:
        pass

    class MalformedPlotStore:
        zw = {}

    assert stored_display_metadata(MissingPlotStore(), "group") is None
    with pytest.raises(KeyError, match="cellData"):
        stored_display_metadata(MalformedPlotStore(), "group")


def test_embedding_density_highlight_and_labeled_colorbar(umap, datastore):
    n = len(datastore.cells.active_index("I"))
    highlighted = np.zeros(n, dtype=bool)
    highlighted[: max(3, n // 10)] = True
    datastore.cells.insert("plot_highlight", highlighted, overwrite=True)

    result = splt.embedding(
        datastore,
        layout_key="RNA_UMAP",
        color_by="RNA_nCounts",
        density_overlay=splt.DensityOverlay(pixels=32, levels=3, sigma=1),
        highlight=splt.Highlight(by="plot_highlight"),
        point_size_range=(2, 20),
        show_titles=False,
        show=False,
    )

    ax = next(iter(result.axes.values()))
    assert len(ax.collections) >= 3
    assert result.provenance.extras["highlight"]["n_highlighted"] == highlighted.sum()
    assert all(
        2 <= value <= 20
        for value in result.provenance.extras["point_size_by_panel"].values()
    )
    auxiliary_labels = [
        axis.get_xlabel() for axis in result.figure.axes if axis is not ax
    ]
    assert "RNA_nCounts" in auxiliary_labels
    assert ax.get_title() == ""
    result.close()


def test_embedding_mean_contours_require_and_use_continuous_values(
    umap,
    datastore,
):
    result = splt.embedding(
        datastore,
        layout_key="RNA_UMAP",
        color_by="RNA_nCounts",
        density_overlay=splt.DensityOverlay(
            statistic="mean",
            pixels=32,
            sigma=1.5,
            min_support=0.05,
            levels=(0.6, 0.85),
            max_hotspots=1,
        ),
        show=False,
    )

    ax = next(iter(result.axes.values()))
    assert len(ax.collections) > 1
    assert result.provenance.extras["density_overlay"]["statistic"] == "mean"
    assert result.provenance.extras["density_overlay"]["max_hotspots"] == 1
    result.close()

    with pytest.raises(ValueError, match="continuous color_by"):
        splt.embedding(
            datastore,
            layout_key="RNA_UMAP",
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


def test_embedding_point_size_uses_final_panel_area(umap, datastore):
    result = splt.embedding(
        datastore,
        layout_key="RNA_UMAP",
        color_by="RNA_nCounts",
        point_size_range=(2, 20),
        show=False,
    )

    result.figure.canvas.draw()
    ax = next(iter(result.axes.values()))
    bbox = ax.get_position()
    width, height = result.figure.get_size_inches()
    panel_area = float(bbox.width * width * bbox.height * height)
    expected = default_point_size(
        len(datastore.cells.active_index("I")),
        panel_area=panel_area,
        size_min=2,
        size_max=20,
    )
    observed = next(iter(result.provenance.extras["point_size_by_panel"].values()))
    assert observed == pytest.approx(expected)
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

    mismatched = _synthetic_plot_store(
        layout1=[0.0, 1.0, 2.0],
        layout2=[0.0, 1.0],
    )
    with pytest.raises(ValueError):
        splt.embedding(mismatched, layout_key="layout", show=False)

    invalid = _synthetic_plot_store(
        layout1=[np.nan, np.inf],
        layout2=[np.nan, -np.inf],
    )
    with pytest.raises(ValueError, match="has no finite coordinates"):
        splt.embedding(invalid, layout_key="layout", show=False)

    bad_facet = _synthetic_plot_store(
        layout1=[0.0, 1.0, 2.0],
        layout2=[0.0, 1.0, 0.0],
        facet=["a", "b"],
    )
    with pytest.raises(ValueError):
        splt.embedding(
            bad_facet,
            layout_key="layout",
            facet_by="facet",
            facet_order=["a", "b"],
            show=False,
        )


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
    for layout in layouts:
        child = result.provenance.extras["layout_provenance"][layout]
        assert child.extras["n_facets"] == 3
        assert child.extras["color_scale_scope"] == "panel"
        assert result.axes[(layout, "score", "missing")].axison is False
    result.close()


def test_embedding_figure_legend_uses_backend_compatible_fallback(monkeypatch):
    from matplotlib.figure import Figure

    original_legend = Figure.legend
    attempted_locations = []

    def reject_outside_location(self, *args, **kwargs):
        attempted_locations.append(kwargs.get("loc"))
        if kwargs.get("loc") == "outside right center":
            raise ValueError("outside legends unsupported")
        return original_legend(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "legend", reject_outside_location)
    store = _synthetic_plot_store(
        layout1=[0.0, 1.0, 2.0, 3.0],
        layout2=[0.0, 1.0, 0.0, 1.0],
        category=["a", "b", "a", "b"],
    )

    result = splt.embedding(
        store,
        layout_key="layout",
        color_by=splt.CellField("category", kind="categorical"),
        legend_loc="right",
        show=False,
    )

    assert attempted_locations == ["outside right center", "center left"]
    assert len(result.figure.legends) == 1
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


@pytest.mark.parametrize(
    ("scale", "values"),
    [
        ("log", [0.25, 0.5, 1.0, 2.0]),
        ("symlog", [-2.0, -0.5, 0.5, 3.0]),
    ],
)
def test_embedding_renders_non_linear_continuous_scales(scale, values):
    store = _synthetic_plot_store(
        layout1=[0.0, 1.0, 0.0, 1.0],
        layout2=[0.0, 0.0, 1.0, 1.0],
        score=values,
    )

    result = splt.embedding(
        store,
        layout_key="layout",
        color_by=splt.CellField("score", kind="continuous"),
        color_scale=splt.ColorScale(scale=scale),
        point_size=5,
        show_legend=False,
        show=False,
    )

    collection = result.axes["score"].collections[0]
    assert np.isfinite(collection.get_facecolors()).all()
    assert result.scales[0].scale == scale
    result.close()


def test_embedding_continuous_limits_handle_degenerate_ranges():
    from scarf.plotting.embedding import _continuous_limits

    assert _continuous_limits(
        np.asarray([np.nan, np.inf]),
        splt.ColorScale(),
    ) == (0.0, 1.0)

    values = np.asarray([0.0, 1.0, 2.0, 100.0])
    quantile_scale = splt.ColorScale(quantiles=(0.25, 0.75))
    assert _continuous_limits(values, quantile_scale) == pytest.approx(
        tuple(np.quantile(values, (0.25, 0.75)))
    )
    assert _continuous_limits(
        values,
        splt.ColorScale(vmin=4.0, vmax=4.0),
    ) == pytest.approx((3.5, 4.5))
    assert _continuous_limits(
        values,
        splt.ColorScale(vmin=2.0, vmax=2.0, scale="log"),
    ) == pytest.approx((1.98, 2.02))
    with pytest.raises(ValueError, match="positive values"):
        _continuous_limits(
            values,
            splt.ColorScale(vmin=0.0, vmax=0.0, scale="log"),
        )


def test_imported_embedding_reuse_guard_and_validator_reject_damage():
    from scarf.embeddings.imported import (
        _payloads_match,
        validate_imported_embedding_artifact,
    )
    from scarf.graph.state import ImportedArtifactStorage
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


def test_dotplot_feature_brackets_and_axis_swap(umap, leiden_clustering, datastore):
    genes = [str(value) for value in datastore.RNA.feats.fetch_all("names")[:4]]
    features = {"Lineage": genes[:2], "State": genes[2:]}
    result = splt.dotplot(
        datastore,
        features=features,
        group_by="RNA_leiden_cluster",
        swap_axes=True,
        marker_linewidth=0.6,
        show=False,
    )

    ax = result.axes["dotplot"]
    assert result.provenance.extras["feature_group_brackets"] == 2
    assert sum(line.get_gid() == "feature-group-bracket" for line in ax.lines) == 2
    assert ax.get_xlabel() == ""
    result.close()


def test_dotplot_marker_sizes_follow_physical_grid_cells(
    umap,
    leiden_clustering,
    datastore,
):
    genes = [str(value) for value in datastore.RNA.feats.fetch_all("names")[:4]]
    features = {"Lineage": genes[:2], "State": genes[2:]}
    compact_figure, compact_ax = plt.subplots(figsize=(3, 2))
    large_figure, large_ax = plt.subplots(figsize=(8, 6))

    compact = splt.dotplot(
        datastore,
        features=features,
        group_by="RNA_leiden_cluster",
        target=compact_ax,
        show_legend=False,
        show=False,
    )
    large = splt.dotplot(
        datastore,
        features=features,
        group_by="RNA_leiden_cluster",
        target=large_ax,
        show_legend=False,
        show=False,
    )

    compact_sizes = compact_ax.collections[0].get_sizes()
    large_sizes = large_ax.collections[0].get_sizes()
    assert large_sizes.max() > compact_sizes.max()
    assert compact.provenance.extras["size_scale_source"] == "panel"

    compact_figure.canvas.draw()
    renderer = compact_figure.canvas.get_renderer()
    label_left = min(
        label.get_window_extent(renderer).x0
        for label in compact_ax.get_yticklabels()
        if label.get_text()
    )
    bracket_right = max(
        line.get_transform().transform(line.get_xydata())[:, 0].max()
        for line in compact_ax.lines
        if line.get_gid() == "feature-group-bracket"
    )
    assert bracket_right < label_left

    compact.close()
    large.close()
    plt.close(compact_figure)
    plt.close(large_figure)


def test_dotplot_left_group_labels_clear_feature_tick_labels(
    umap,
    leiden_clustering,
    datastore,
):
    genes = [str(value) for value in datastore.RNA.feats.fetch_all("names")[:4]]
    result = splt.dotplot(
        datastore,
        features={"Myeloid lineage": genes[:2], "Cell state": genes[2:]},
        group_by="RNA_leiden_cluster",
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


def test_stacked_violin_standardizes_rows(umap, leiden_clustering, datastore):
    genes = [str(value) for value in datastore.RNA.feats.fetch_all("names")[:2]]
    result = splt.distribution(
        datastore,
        keys=genes,
        group_by="RNA_leiden_cluster",
        kind="stacked_violin",
        row_standardize=True,
        max_points=0,
        show=False,
    )

    assert len(result.axes) == 2
    for table in result.tables.values():
        assert np.nanmean(table["display_value"]) == pytest.approx(0, abs=1e-7)
    assert result.provenance.extras["row_standardize"] is True
    result.close()


def test_stacked_violin_can_share_value_scale(umap, leiden_clustering, datastore):
    genes = [str(value) for value in datastore.RNA.feats.fetch_all("names")[:2]]
    result = splt.distribution(
        datastore,
        keys=genes,
        group_by="RNA_leiden_cluster",
        kind="stacked_violin",
        share_y=True,
        max_points=0,
        show=False,
    )

    limits = [axis.get_ylim() for axis in result.axes.values()]
    assert all(limit == pytest.approx(limits[0]) for limit in limits[1:])
    assert result.provenance.extras["share_y"] is True
    result.close()


def test_distribution_aggregates_biological_samples(
    umap,
    leiden_clustering,
    datastore,
):
    n = len(datastore.cells.active_index("I"))
    samples = np.asarray([f"sample_{index % 8}" for index in range(n)])
    datastore.cells.insert("plot_distribution_sample", samples, overwrite=True)

    result = splt.distribution(
        datastore,
        keys="RNA_nCounts",
        group_by="RNA_leiden_cluster",
        sample_by="plot_distribution_sample",
        sample_stat="median",
        kind="box",
        max_points=0,
        show=False,
    )

    table = result.tables["RNA_nCounts"]
    assert {"sample", "group", "value", "display_value", "nCells"} <= set(table)
    assert not table.duplicated(["sample", "group"]).any()
    assert result.provenance.n_samples == 8
    assert result.provenance.extras["sample_stat"] == "median"
    result.close()


def test_distribution_draws_sample_aware_split_violins(
    umap,
    leiden_clustering,
    datastore,
):
    n = len(datastore.cells.active_index("I"))
    sample_index = np.arange(n) % 8
    samples = np.asarray([f"sample_{index}" for index in sample_index])
    conditions = np.where(sample_index % 2 == 0, "control", "stimulated")
    datastore.cells.insert("plot_split_sample", samples, overwrite=True)
    datastore.cells.insert("plot_split_condition", conditions, overwrite=True)

    result = splt.distribution(
        datastore,
        keys="RNA_nCounts",
        group_by="RNA_leiden_cluster",
        split_by="plot_split_condition",
        study_design=splt.StudyDesign(sample_by="plot_split_sample"),
        kind="violin",
        max_points=30,
        show=False,
    )

    table = result.tables["RNA_nCounts"]
    assert set(table["split"]) == {"control", "stimulated"}
    scale = result.scales[0]
    assert isinstance(scale, splt.CategoricalScale)
    assert scale.order == ("control", "stimulated")
    assert result.legends[0].label == "plot_split_condition"
    result.close()


def test_distribution_seed_repeats_point_jitter(
    umap,
    leiden_clustering,
    datastore,
):
    kwargs = {
        "keys": "RNA_nCounts",
        "group_by": "RNA_leiden_cluster",
        "kind": "box",
        "max_points": 80,
        "seed": 17,
        "show": False,
    }
    first = splt.distribution(datastore, **kwargs)
    second = splt.distribution(datastore, **kwargs)

    first_offsets = [
        np.asarray(collection.get_offsets())
        for collection in next(iter(first.axes.values())).collections
    ]
    second_offsets = [
        np.asarray(collection.get_offsets())
        for collection in next(iter(second.axes.values())).collections
    ]
    assert len(first_offsets) == len(second_offsets)
    for observed, repeated in zip(first_offsets, second_offsets, strict=True):
        np.testing.assert_array_equal(observed, repeated)
    first.close()
    second.close()


def test_sample_aggregated_feature_axis_retains_requested_italics(
    umap,
    datastore,
):
    n = len(datastore.cells.active_index("I"))
    samples = np.asarray([f"sample_{index % 8}" for index in range(n)])
    datastore.cells.insert("plot_italic_sample", samples, overwrite=True)
    gene = str(datastore.RNA.feats.fetch_all("names")[0])

    result = splt.distribution(
        datastore,
        keys=gene,
        sample_by="plot_italic_sample",
        kind="box",
        italicize_features=True,
        show=False,
    )

    axis = next(iter(result.axes.values()))
    assert axis.get_ylabel() == f"Sample mean {gene}"
    assert axis.yaxis.label.get_fontstyle() == "italic"
    result.close()


def test_per_sample_composition_reports_uncertainty(
    umap,
    leiden_clustering,
    datastore,
):
    n = len(datastore.cells.active_index("I"))
    sample_index = np.arange(n) % 8
    samples = np.asarray([f"sample_{index}" for index in sample_index])
    conditions = np.where(sample_index < 4, "control", "stimulated")
    subjects = np.asarray([f"subject_{index % 4}" for index in sample_index])
    datastore.cells.insert("plot_ci_sample", samples, overwrite=True)
    datastore.cells.insert("plot_ci_condition", conditions, overwrite=True)
    datastore.cells.insert("plot_ci_subject", subjects, overwrite=True)

    result = splt.composition(
        datastore,
        category_by="RNA_leiden_cluster",
        study_design=splt.StudyDesign(
            sample_by="plot_ci_sample",
            condition_by="plot_ci_condition",
            subject_by="plot_ci_subject",
        ),
        kind="per_sample",
        uncertainty="ci95",
        show=False,
    )

    summary = result.tables["summary"]
    assert {"mean_proportion", "lower", "upper", "n_samples"} <= set(summary)
    assert np.all(summary["lower"] <= summary["mean_proportion"])
    assert np.all(summary["mean_proportion"] <= summary["upper"])
    assert result.provenance.extras["uncertainty"] == "ci95"
    assert result.provenance.extras["n_pair_lines"] > 0
    assert any(legend.kind == "marker" for legend in result.legends)
    axis = next(iter(result.axes.values()))
    assert axis.get_ylim()[1] >= float(summary["upper"].max())
    for legend in result.figure.legends:
        if legend.get_title().get_text() == "Condition":
            assert all(
                not text.get_text().startswith("mean") for text in legend.get_texts()
            )
    rendered_labels = [
        text.get_text()
        for legend in result.figure.legends
        for text in legend.get_texts()
    ]
    assert any(
        label.startswith("mean") or label.startswith("Summary: mean")
        for label in rendered_labels
    )
    result.close()


def test_embedding_side_legend_stays_page_sized_for_many_categories(
    umap,
    datastore,
):
    n = len(datastore.cells.active_index("I"))
    labels = np.full(n, "type_149", dtype=object)
    labels[:149] = [f"type_{index:03d}" for index in range(149)]
    datastore.cells.insert("plot_many_types", labels, overwrite=True)

    result = splt.embedding(
        datastore,
        layout_key="RNA_UMAP",
        color_by="plot_many_types",
        legend_loc="right",
        show=False,
    )

    width, _ = result.figure.get_size_inches()
    assert width <= 12
    legend = result.figure.legends[0]
    assert len(legend.get_texts()) == 80
    assert legend.get_title().get_text() == "plot_many_types (80 of 150)"
    assert "type_149" in [text.get_text() for text in legend.get_texts()]
    omitted = result.provenance.extras["omitted_legend_entries"]
    panel_key = str(next(iter(result.axes)))
    assert len(omitted[panel_key]) == 70
    result.close()


def test_cluster_connectivity_runs_on_real_datastore_graph(
    umap,
    leiden_clustering,
    datastore,
):
    result = datastore.plots.cluster_connectivity(
        group_by="RNA_leiden_cluster",
        layout_key="RNA_UMAP",
        minimum_edge_weight=0,
        max_edges_per_node=3,
        show_cells=True,
        show=False,
    )

    observed = len(np.unique(datastore.cells.fetch("RNA_leiden_cluster", key="I")))
    assert len(result.tables["nodes"]) == observed
    assert set(result.tables["edges"]) == {
        "source",
        "target",
        "rawWeight",
        "normalizedWeight",
    }
    assert result.provenance.extras["n_edges"] <= 3 * observed // 2
    background = result.axes["cluster_connectivity"].collections[0]
    assert background.get_alpha() == pytest.approx(0.3)
    assert background.get_sizes()[0] >= 4
    assert not hasattr(background, "_scarf_layout_point_size")
    assert len(np.unique(background.get_facecolors(), axis=0)) > 1
    assert result.provenance.extras["cell_size_source"] == "panel"
    result.close()


def test_composition_borders_labels_and_stored_palette(
    umap,
    leiden_clustering,
    datastore,
):
    n = len(datastore.cells.active_index("I"))
    samples = np.asarray([f"s{index % 3}" for index in range(n)], dtype=object)
    datastore.cells.insert("plot_modern_samples", samples, overwrite=True)

    embedding = splt.embedding(
        datastore,
        layout_key="RNA_UMAP",
        color_by="RNA_leiden_cluster",
        show_legend=False,
        show=False,
    )
    composition = splt.composition(
        datastore,
        category_by="RNA_leiden_cluster",
        sample_by="plot_modern_samples",
        segment_linewidth=0.8,
        show_percent_labels=True,
        label_min_fraction=0.01,
        show=False,
    )

    bars = composition.axes["composition"].patches
    assert bars
    assert all(bar.get_linewidth() == pytest.approx(0.8) for bar in bars)
    assert composition.axes["composition"].texts
    embedding_scale = next(
        scale for scale in embedding.scales if isinstance(scale, splt.CategoricalScale)
    )
    composition_scale = next(
        scale
        for scale in composition.scales
        if isinstance(scale, splt.CategoricalScale)
    )
    assert embedding_scale.palette == composition_scale.palette
    embedding.close()
    composition.close()


def test_compose_results_namespaces_tables_and_renders_shared_legend(
    umap,
    leiden_clustering,
    datastore,
):
    n = len(datastore.cells.active_index("I"))
    phases = np.asarray(["G1", "S", "G2M"] * (n // 3 + 1), dtype=object)[:n]
    datastore.cells.insert("plot_composite_phase", phases, overwrite=True)
    figure, axes = plt.subplot_mosaic(
        [["embedding", "composition"]],
        figsize=(7, 3),
        layout="constrained",
    )
    first = splt.embedding(
        datastore,
        layout_key="RNA_UMAP",
        color_by="RNA_leiden_cluster",
        target=axes["embedding"],
        show_legend=True,
        theme="paper",
        show=False,
    )
    second = splt.composition(
        datastore,
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
    assert len(figure.legends) == 2
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


def test_recipe_execution_is_headless_and_preserves_analysis_state(umap, datastore):
    before = datastore.get_assay_state("RNA")
    recipe = splt.PlotRecipe(
        (
            splt.PlotStep(
                name="overview",
                plot="embedding",
                kwargs={
                    "layout_key": "RNA_UMAP",
                    "color_by": "RNA_nCounts",
                },
            ),
        )
    )

    execution = datastore.plots.run_recipe(recipe)

    assert not execution.written_paths
    assert not execution.failures
    assert len(execution.results) == 1
    assert datastore.get_assay_state("RNA") == before
    result = execution.results[0]
    assert plt.fignum_exists(result.figure.number)
    result.close()


def test_recipe_batch_output_closes_owned_figure(umap, datastore, tmp_path):
    recipe = splt.PlotRecipe(
        (
            splt.PlotStep(
                name="overview",
                plot="embedding",
                kwargs={
                    "layout_key": "RNA_UMAP",
                    "color_by": "RNA_nCounts",
                },
                output=splt.PlotOutputSettings(
                    filename="overview.png",
                    dpi=90,
                ),
            ),
        )
    )

    execution = splt.run_recipe(datastore, recipe, output_dir=tmp_path)
    plot_result = execution.results[0]

    assert execution.written_paths == (tmp_path / "overview.png",)
    assert execution.written_paths[0].stat().st_size > 0
    assert not plt.fignum_exists(plot_result.figure.number)


def _expressed_gene_names(datastore, n=3):
    names = datastore.RNA.feats.fetch_all("names")
    counts = np.asarray(
        datastore.RNA.rawData[:, : len(names)].sum(axis=0).compute(),
        dtype=np.float64,
    ).ravel()
    order = np.argsort(-counts)
    chosen = []
    for index in order:
        if counts[index] > 0:
            chosen.append(str(names[index]))
        if len(chosen) == n:
            break
    return chosen


def test_stacked_violin_mean_color_expression(umap, leiden_clustering, datastore):
    genes = _expressed_gene_names(datastore)
    result = splt.distribution(
        datastore,
        keys=genes,
        group_by="RNA_leiden_cluster",
        kind="stacked_violin",
        color_by="mean",
        color_scale=splt.ColorScale(scope="shared"),
        max_points=0,
        show=False,
    )
    try:
        assert len(result.axes) == len(genes)
        assert result.legends[0].kind == "colorbar"
        assert result.legends[0].label == "mean expression"
        assert any(isinstance(scale, splt.ColorScale) for scale in result.scales)
        assert any(ax.get_label().startswith("<colorbar") for ax in result.figure.axes)
        assert result.provenance.extras["color_by"] == "mean"
        # Shared scale spans the min/max of every group mean.
        all_means = [
            mean
            for table in result.tables.values()
            for mean in table.groupby("group")["display_value"].mean()
        ]
        assert result.provenance.extras["vmin"] == pytest.approx(min(all_means))
        assert result.provenance.extras["vmax"] == pytest.approx(max(all_means))
        # Mean coloring gives each group a distinct colour within a row.
        ax = list(result.axes.values())[0]
        face_colors = {
            tuple(np.round(color.get_facecolor()[0][:3], 3))
            for color in ax.collections
            if hasattr(color, "get_facecolor") and len(color.get_facecolor())
        }
        assert len(face_colors) >= 2
    finally:
        result.close()


def test_stacked_violin_mean_color_explicit_bounds(umap, leiden_clustering, datastore):
    gene = _expressed_gene_names(datastore, n=1)[0]
    result = splt.distribution(
        datastore,
        keys=gene,
        group_by="RNA_leiden_cluster",
        kind="stacked_violin",
        color_by="mean",
        color_scale=splt.ColorScale(cmap="magma", vmin=0.0, vmax=5.0, scope="shared"),
        max_points=0,
        show=False,
    )
    try:
        color_scale = next(
            scale for scale in result.scales if isinstance(scale, splt.ColorScale)
        )
        assert color_scale.cmap == "magma"
        assert color_scale.vmin == 0.0
        assert color_scale.vmax == 5.0
        assert color_scale.scope == "shared"
        assert result.legends[0].extras["vmin"] == 0.0
        assert result.legends[0].extras["vmax"] == 5.0
        assert result.provenance.extras["vmin"] == 0.0
        assert result.provenance.extras["vmax"] == 5.0
        assert result.provenance.extras["color_scale_scope"] == "shared"
    finally:
        result.close()


def test_stacked_violin_mean_color_constant_row(umap, leiden_clustering, datastore):
    n = len(datastore.cells.active_index("I"))
    datastore.cells.insert("constant_metric", np.full(n, 5.0), overwrite=True)
    result = splt.distribution(
        datastore,
        keys="constant_metric",
        group_by="RNA_leiden_cluster",
        kind="stacked_violin",
        row_standardize=True,
        color_by="mean",
        color_scale=splt.ColorScale(scope="shared"),
        max_points=0,
        show=False,
    )
    try:
        table = list(result.tables.values())[0]
        assert np.nanmean(table["display_value"]) == pytest.approx(0, abs=1e-9)
        # Degenerate scale is padded symmetrically around zero so the colourbar
        # still renders, labelled for the standardized values.
        vmin = result.provenance.extras["vmin"]
        vmax = result.provenance.extras["vmax"]
        assert vmin == pytest.approx(-vmax)
        assert any(ax.get_label().startswith("<colorbar") for ax in result.figure.axes)
        colorbar = next(
            ax for ax in result.figure.axes if ax.get_label().startswith("<colorbar")
        )
        assert colorbar.get_ylabel() == "mean standardized value"
    finally:
        result.close()


def test_stacked_violin_color_scale_requires_mean(umap, leiden_clustering, datastore):
    gene = _expressed_gene_names(datastore, n=1)[0]
    with pytest.raises(
        ValueError, match="color_scale applies only when color_by='mean'"
    ):
        splt.distribution(
            datastore,
            keys=gene,
            group_by="RNA_leiden_cluster",
            kind="stacked_violin",
            color_scale=splt.ColorScale(cmap="magma"),
            show=False,
        )


def test_stacked_violin_mean_color_rejects_split(umap, leiden_clustering, datastore):
    gene = _expressed_gene_names(datastore, n=1)[0]
    with pytest.raises(ValueError, match="cannot be combined with split_by"):
        splt.distribution(
            datastore,
            keys=gene,
            group_by="RNA_leiden_cluster",
            split_by="RNA_leiden_cluster",
            kind="stacked_violin",
            color_by="mean",
            show=False,
        )


def test_stacked_violin_mean_color_rejects_log_scale(
    umap, leiden_clustering, datastore
):
    gene = _expressed_gene_names(datastore, n=1)[0]
    with pytest.raises(NotImplementedError, match="linear"):
        splt.distribution(
            datastore,
            keys=gene,
            group_by="RNA_leiden_cluster",
            kind="stacked_violin",
            color_by="mean",
            color_scale=splt.ColorScale(scale="log"),
            show=False,
        )


def test_stacked_violin_mean_color_quantiles(umap, leiden_clustering, datastore):
    genes = _expressed_gene_names(datastore, n=2)
    result = splt.distribution(
        datastore,
        keys=genes,
        group_by="RNA_leiden_cluster",
        kind="stacked_violin",
        color_by="mean",
        color_scale=splt.ColorScale(quantiles=(0.25, 0.75), scope="shared"),
        max_points=0,
        show=False,
    )
    try:
        color_scale = next(
            scale for scale in result.scales if isinstance(scale, splt.ColorScale)
        )
        all_means = [
            mean
            for table in result.tables.values()
            for mean in table.groupby("group")["display_value"].mean()
        ]
        finite = np.asarray([m for m in all_means if np.isfinite(m)])
        assert color_scale.vmin == pytest.approx(np.quantile(finite, 0.25))
        assert color_scale.vmax == pytest.approx(np.quantile(finite, 0.75))
        # Rendered face colours honour the quantile clip: the lowest mean maps
        # to the bottom of the colormap and the highest to the top. Seaborn
        # desaturates the fills by ``saturation=0.9``.
        from matplotlib import colormaps
        from matplotlib.colors import to_rgb
        from seaborn.utils import desaturate

        face_colors = {
            tuple(np.round(color.get_facecolor()[0][:3], 3))
            for ax in result.axes.values()
            for color in ax.collections
            if hasattr(color, "get_facecolor") and len(color.get_facecolor())
        }

        def desat(t: float) -> tuple[float, float, float]:
            return tuple(
                np.round(to_rgb(desaturate(to_rgb(colormaps["viridis"](t)), 0.9)), 3)
            )

        def close_to(fc: tuple[float, ...], expected: tuple[float, ...]) -> bool:
            return all(abs(a - b) <= 0.01 for a, b in zip(fc, expected))

        assert any(close_to(fc, desat(0.0)) for fc in face_colors)
        assert any(close_to(fc, desat(1.0)) for fc in face_colors)
        assert len(face_colors) >= 3
    finally:
        result.close()


def test_stacked_violin_mean_color_panel_scope(umap, leiden_clustering, datastore):
    genes = _expressed_gene_names(datastore, n=2)
    result = splt.distribution(
        datastore,
        keys=genes,
        group_by="RNA_leiden_cluster",
        kind="stacked_violin",
        color_by="mean",
        color_scale=splt.ColorScale(scope="panel"),
        max_points=0,
        show=False,
    )
    try:
        color_scale = next(
            scale for scale in result.scales if isinstance(scale, splt.ColorScale)
        )
        assert color_scale.scope == "panel"
        assert result.provenance.extras["color_scale_scope"] == "panel"
        # Panel scope still draws a single reference colourbar from the pooled
        # group means.
        colorbars = [
            ax for ax in result.figure.axes if ax.get_label().startswith("<colorbar")
        ]
        assert len(colorbars) == 1
        assert colorbars[0].get_ylabel() == "Relative Expression Per Gene"
        # Panel scope draws a single colorbar on the unit 0-to-1 relative scale.
        assert color_scale.vmin == pytest.approx(0.0)
        assert color_scale.vmax == pytest.approx(1.0)
        assert result.legends[0].extras["vmin"] == pytest.approx(0.0)
        assert result.legends[0].extras["vmax"] == pytest.approx(1.0)
        assert result.provenance.extras["vmin"] == pytest.approx(0.0)
        assert result.provenance.extras["vmax"] == pytest.approx(1.0)
    finally:
        result.close()


def test_stacked_violin_scope_follows_share_y(umap, leiden_clustering, datastore):
    gene = _expressed_gene_names(datastore, n=1)[0]
    independent = splt.distribution(
        datastore,
        keys=gene,
        group_by="RNA_leiden_cluster",
        kind="stacked_violin",
        color_by="mean",
        max_points=0,
        show=False,
    )
    try:
        assert independent.provenance.extras["color_scale_scope"] == "panel"
        assert independent.legends[0].label == "Relative Expression Per Gene"
    finally:
        independent.close()
    shared = splt.distribution(
        datastore,
        keys=gene,
        group_by="RNA_leiden_cluster",
        kind="stacked_violin",
        color_by="mean",
        share_y=True,
        max_points=0,
        show=False,
    )
    try:
        assert shared.provenance.extras["color_scale_scope"] == "shared"
        assert shared.legends[0].label == "mean expression"
    finally:
        shared.close()


def test_stacked_violin_mean_color_default_scale_scope_is_ergonomic(
    umap, leiden_clustering, datastore
):
    gene = _expressed_gene_names(datastore, n=1)[0]
    result = splt.distribution(
        datastore,
        keys=gene,
        group_by="RNA_leiden_cluster",
        kind="stacked_violin",
        color_by="mean",
        color_scale=splt.ColorScale(cmap="magma"),
        max_points=0,
        show=False,
    )
    try:
        scale = next(s for s in result.scales if isinstance(s, splt.ColorScale))
        assert scale.scope == "panel"
        assert scale.cmap == "magma"
    finally:
        result.close()


def test_stacked_violin_mean_color_no_colorbar_on_target(
    umap, leiden_clustering, datastore
):
    genes = _expressed_gene_names(datastore, n=2)
    fig, axes = plt.subplots(1, 2)
    result = splt.distribution(
        datastore,
        keys=genes,
        group_by="RNA_leiden_cluster",
        kind="stacked_violin",
        color_by="mean",
        color_scale=splt.ColorScale(scope="shared"),
        max_points=0,
        target=[axes[0], axes[1]],
        show=False,
    )
    try:
        assert result.owns_figure is False
        assert not any(
            ax.get_label().startswith("<colorbar") for ax in result.figure.axes
        )
        assert any(legend.kind == "colorbar" for legend in result.legends)
    finally:
        result.close()
        plt.close(fig)


def test_stacked_violin_mean_color_missing_group_missing_color():
    from scarf.plotting.distribution import _mean_group_palette

    means = pd.Series({"a": 1.0, "b": np.nan})
    color_scale = splt.ColorScale(scope="shared")
    palette = _mean_group_palette(
        means,
        ["a", "b"],
        color_scale=color_scale,
        lo=0.0,
        hi=2.0,
    )
    assert palette["b"] == color_scale.missing_color
    assert palette["a"] != color_scale.missing_color


def test_stacked_violin_mean_color_vcenter_extends_bounds(
    umap, leiden_clustering, datastore
):
    gene = _expressed_gene_names(datastore, n=1)[0]
    result = splt.distribution(
        datastore,
        keys=gene,
        group_by="RNA_leiden_cluster",
        kind="stacked_violin",
        color_by="mean",
        color_scale=splt.ColorScale(vcenter=0.0, scope="shared"),
        max_points=0,
        show=False,
    )
    try:
        scale = next(s for s in result.scales if isinstance(s, splt.ColorScale))
        assert scale.vmin <= 0.0
        assert scale.vmax > 0.0
        assert result.legends[0].kind == "colorbar"
    finally:
        result.close()


def test_stacked_violin_explicit_none_uses_no_overlay(
    umap, leiden_clustering, datastore
):
    gene = _expressed_gene_names(datastore, n=1)[0]
    result = splt.distribution(
        datastore,
        keys=gene,
        group_by="RNA_leiden_cluster",
        kind="stacked_violin",
        max_points=None,
        show=False,
    )
    try:
        assert result.provenance.extras["max_points"] == 0
        for ax in result.axes.values():
            point_collections = [
                collection
                for collection in ax.collections
                if hasattr(collection, "get_offsets")
            ]
            # Violin bodies carry a single default offset; a jitter overlay
            # would add a collection with one offset per drawn cell.
            assert all(
                len(collection.get_offsets()) <= 1 for collection in point_collections
            )
    finally:
        result.close()


def test_stacked_violin_panel_scope_strict_minmax(umap, leiden_clustering, datastore):
    genes = _expressed_gene_names(datastore, n=2)
    result = splt.distribution(
        datastore,
        keys=genes,
        group_by="RNA_leiden_cluster",
        kind="stacked_violin",
        color_by="mean",
        color_scale=splt.ColorScale(scope="panel"),
        max_points=0,
        show=False,
    )
    try:
        from matplotlib import colormaps
        from matplotlib.colors import to_rgb
        from seaborn.utils import desaturate

        def desat(t: float) -> tuple[float, float, float]:
            return tuple(
                np.round(to_rgb(desaturate(to_rgb(colormaps["viridis"](t)), 0.9)), 3)
            )

        def close_to(fc: tuple[float, ...], expected: tuple[float, ...]) -> bool:
            return all(abs(a - b) <= 0.01 for a, b in zip(fc, expected))

        lo_color, hi_color = desat(0.0), desat(1.0)
        # Every panel rescales to its own 0-to-1 range, so the lowest and
        # highest-mean clusters in EVERY row land on the colormap endpoints.
        for ax in result.axes.values():
            face_colors = {
                tuple(np.round(c.get_facecolor()[0][:3], 3))
                for c in ax.collections
                if hasattr(c, "get_facecolor") and len(c.get_facecolor())
            }
            assert any(close_to(fc, lo_color) for fc in face_colors)
            assert any(close_to(fc, hi_color) for fc in face_colors)
    finally:
        result.close()


def test_stacked_violin_panel_scope_rejects_bounds(umap, leiden_clustering, datastore):
    gene = _expressed_gene_names(datastore, n=1)[0]
    with pytest.raises(ValueError, match="apply only to scope='shared'"):
        splt.distribution(
            datastore,
            keys=gene,
            group_by="RNA_leiden_cluster",
            kind="stacked_violin",
            color_by="mean",
            color_scale=splt.ColorScale(scope="panel", vmin=0.0),
            max_points=0,
            show=False,
        )
    with pytest.raises(ValueError, match="apply only to scope='shared'"):
        splt.distribution(
            datastore,
            keys=gene,
            group_by="RNA_leiden_cluster",
            kind="stacked_violin",
            color_by="mean",
            color_scale=splt.ColorScale(scope="panel", quantiles=(0.1, 0.9)),
            max_points=0,
            show=False,
        )
    with pytest.raises(ValueError, match="apply only to scope='shared'"):
        splt.distribution(
            datastore,
            keys=gene,
            group_by="RNA_leiden_cluster",
            kind="stacked_violin",
            color_by="mean",
            color_scale=splt.ColorScale(scope="panel", vcenter=0.0),
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

    assert reference[0] < 0 < reference[1]
    assert palette["a"] != palette["e"]


def test_stacked_violin_mean_color_honors_hidden_legend_and_generic_label():
    store = _synthetic_plot_store(
        I=np.ones(12, dtype=bool),
        group=np.repeat(["a", "b", "c"], 4),
        metric=np.arange(12, dtype=float),
    )
    result = splt.distribution(
        store,
        "metric",
        group_by="group",
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
        group_by="group",
        kind="stacked_violin",
        show=False,
    )
    try:
        assert result.provenance.extras["max_points"] == 10000
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
        group_by="group",
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
            group_by="group",
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
            group_by="group",
            max_points=0,
            stats_results=incomplete_stats,
            show=False,
        )
    result.close()


def test_distribution_stats_rejects_changed_assay_normalization_state():
    from scarf.plotting.distribution import (
        _stat_result_compatibility_issue,
        _value_fingerprint,
    )

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
            group_by="group",
            cell_key="I",
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
            cell_selection_fingerprint=_value_fingerprint(cells),
            group_fingerprint=_value_fingerprint(groups),
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
    from scarf.plotting.distribution import _value_fingerprint

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
    stats.cell_selection_fingerprint = _value_fingerprint(retained)
    stats.group_fingerprint = _value_fingerprint(store.cells.fetch("group")[:4])
    stats.group_order = ("a", "b")
    stats.value_fingerprints = (_value_fingerprint(store.cells.fetch("metric")[:4]),)

    result = splt.distribution(
        store,
        "metric",
        group_by="group",
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
            group_by="group",
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
            group_by="group",
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
            group_by="group",
            sample_by="sample",
            max_points=0,
            stats_results=paired_stats,
            show=False,
        )
    result.close()

    matching_paired_stats = _synthetic_stats_result(
        store,
        table,
        sample_by="sample",
        pair_by="pair",
    )
    result = splt.distribution(
        store,
        "metric",
        group_by="group",
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

    with pytest.raises(ValueError, match="cannot be combined with split_by"):
        splt.distribution(
            store,
            "metric",
            group_by="group",
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
        group_by="group",
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
        group_by="group",
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

    with pytest.raises(ValueError, match="pair values must be present"):
        splt.distribution(
            store,
            "metric",
            group_by="group",
            study_design=splt.StudyDesign(sample_by="sample", subject_by="pair"),
            max_points=0,
            stats_results=stats,
            show=False,
        )


def test_distribution_stats_annotations_use_theme_foreground():
    from scarf.plotting._style import foreground_color

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
        group_by="group",
        theme="dark",
        max_points=0,
        stats_results=stats,
        show=False,
    )
    try:
        bracket = next(
            line for line in result.axes["metric"].lines if len(line.get_xdata()) == 4
        )
        assert bracket.get_color() == foreground_color("dark")
        assert result.axes["metric"].texts[0].get_color() == foreground_color("dark")
    finally:
        result.close()


def test_distribution_stats_annotations_use_custom_dark_theme_foreground():
    theme_name = "test-distribution-custom-dark"
    splt.register_theme(theme_name, {"font.size": 9}, base="dark", overwrite=True)
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
        group_by="group",
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
            group_by="group",
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
            group_by="group",
            stats_results=object(),
            stats_bracket_height=height,
            show=False,
        )


def test_distribution_masks_metadata_placeholders_per_panel():
    from scarf.plotting.distribution import _fetch_series, _value_fingerprint
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
    )
    plain_store = _synthetic_plot_store(**columns)
    masked_values, _label, _is_feature, masked_identity, _assay = _fetch_series(
        masked_store,
        "metric",
        cell_key="I",
        from_assay=None,
        normalization=splt.NormalizationSpec(),
    )
    _values, _label, _is_feature, plain_identity, _assay = _fetch_series(
        plain_store,
        "metric",
        cell_key="I",
        from_assay=None,
        normalization=splt.NormalizationSpec(),
    )
    assert np.isnan(masked_values[1])
    assert masked_identity == provenance_hash(
        {
            "source": "cell_metadata",
            "column": "metric",
            "values_fingerprint": _value_fingerprint(columns["metric"]),
            "missing_fingerprint": _value_fingerprint(
                np.array([False, True, False, False, False, False])
            ),
        }
    )
    assert masked_identity != plain_identity

    result = splt.distribution(
        masked_store,
        "metric",
        group_by="group",
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
    )

    with pytest.raises(TypeError, match="must be boolean"):
        splt.distribution(
            store,
            "metric",
            group_by="group",
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

"""Plot scale and theme behavior tests."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import scarf.plotting as splt
from scarf.plotting._deps import require_matplotlib
from scarf.plotting._style import (
    categorical_color_map,
    continuous_norm,
    palette_for_n,
    theme_context,
)


@pytest.mark.parametrize(
    ("size", "index", "color"),
    [
        # Up to 10 categories use tab10, then tab20, then 28 and 102 color
        # tables, and husl beyond those.
        (10, 1, "#ff7f0e"),
        (11, 1, "#aec7e8"),
        (20, 19, "#9edae5"),
        (21, 0, "#023fa5"),
        (28, 27, "#336600"),
        (29, 0, "#FFFF00"),
        (102, 101, "#324E72"),
    ],
)
def test_palette_for_n_switches_tables_at_their_sizes(size, index, color):
    colors = palette_for_n(size)

    assert len(colors) == size
    assert len(set(colors)) == size
    assert colors[index] == color


def test_palette_for_n_falls_back_to_evenly_spaced_hues():
    import seaborn as sns

    assert palette_for_n(103) == sns.color_palette("husl", n_colors=103).as_hex()
    assert palette_for_n(12, palette_name="colorblind")[:2] == ["#0072B2", "#E69F00"]
    assert (
        palette_for_n(13, palette_name="colorblind")
        == sns.color_palette("husl", n_colors=13).as_hex()
    )


@pytest.mark.parametrize("palette_name", ["default", "colorblind"])
@pytest.mark.parametrize("size", [8, 30, 110])
def test_palette_for_n_never_recycles_colors(size, palette_name):
    colors = palette_for_n(size, palette_name=palette_name)
    assert len(set(colors)) == size


def test_side_legend_columns_stay_page_sized():
    from scarf.plotting._style import LEGEND_SIDE_MAX_COLUMNS, legend_side_columns

    # One column holds up to 20 entries.
    assert [legend_side_columns(n) for n in (1, 20, 21, 40, 41)] == [1, 1, 2, 2, 3]
    assert legend_side_columns(5_000) == LEGEND_SIDE_MAX_COLUMNS


def test_categorical_color_map_validates_custom_palette():
    with pytest.raises(KeyError, match="missing from palette"):
        categorical_color_map(["a", "b"], palette={"a": "red"})
    assert categorical_color_map(["a"], palette={"a": "red"}) == {"a": "red"}


def test_continuous_norm_supports_center_and_validates_bounds():
    _, mpl = require_matplotlib()
    norm = continuous_norm(mpl, vmin=-2, vmax=3, vcenter=0)
    assert isinstance(norm, mpl.colors.TwoSlopeNorm)
    # Each side of the centre maps linearly onto its half of the colormap.
    np.testing.assert_allclose(
        norm(np.array([-2.0, -1.0, 0.0, 1.5, 3.0])), [0.0, 0.25, 0.5, 0.75, 1.0]
    )
    with pytest.raises(ValueError, match="vcenter"):
        continuous_norm(mpl, vmin=0, vmax=3, vcenter=4)


def test_generated_palette_and_flat_norm_edge_cases():
    generated = categorical_color_map(["b", "a"])

    assert list(generated) == ["b", "a"]
    assert list(generated.values()) == palette_for_n(2)
    assert palette_for_n(0) == []
    with pytest.raises(ValueError, match="palette_name"):
        palette_for_n(3, palette_name="unknown")

    _, mpl = require_matplotlib()
    for vmax in (2.0, 1.0):
        norm = continuous_norm(mpl, vmin=2.0, vmax=vmax, vcenter=None)
        assert type(norm) is mpl.colors.Normalize
        assert norm.vmin == pytest.approx(2.0)
        assert norm.vmax == pytest.approx(3.0)


def test_square_axis_limits_and_dark_theme():
    from matplotlib.colors import to_rgba

    from scarf.plotting._style import (
        apply_figure_chrome,
        square_axis_limits,
        scatter_edgecolor,
    )

    # The shorter axis widens around its centre to the longer span.
    assert square_axis_limits((0.0, 2.0), (-1.0, 0.0)) == (
        pytest.approx((0.0, 2.0)),
        pytest.approx((-1.5, 0.5)),
    )
    assert scatter_edgecolor("dark") == "#8f8f8f"
    assert scatter_edgecolor("notebook") == "#333333"
    with theme_context("dark"):
        _, mpl = require_matplotlib()
        assert mpl.rcParams["axes.edgecolor"] == "#e8e8e8"
    plt, _ = require_matplotlib()
    fig, ax = plt.subplots(1, 1)
    apply_figure_chrome(fig, "notebook")
    assert fig.patch.get_facecolor() == to_rgba("white")
    assert ax.patch.get_alpha() == 1.0
    assert not ax.spines["top"].get_visible()
    assert not ax.spines["right"].get_visible()
    apply_figure_chrome(fig, "dark")
    assert fig.patch.get_alpha() == 0
    plt.close(fig)


def test_point_size_helpers_validate_bounds_and_cover_density_bands():
    from scarf.plotting._style import default_point_edgewidth, default_point_size

    with pytest.raises(ValueError, match="panel_area must be positive"):
        default_point_size(100, panel_area=0)
    with pytest.raises(ValueError, match="point-size bounds"):
        default_point_size(100, size_min=0)
    with pytest.raises(ValueError, match="point-size bounds"):
        default_point_size(100, size_min=5, size_max=4)

    assert default_point_size(1, size_min=1, size_max=5) == pytest.approx(5)
    assert default_point_size(10**12, size_min=2, size_max=5) == pytest.approx(2)
    assert default_point_edgewidth(500, point_size=10) == pytest.approx(0.15)
    assert default_point_edgewidth(500, point_size=5) == pytest.approx(0.05)
    assert default_point_edgewidth(500, point_size=2) == 0.0
    assert default_point_edgewidth(10_000, point_size=10) == pytest.approx(0.05)


def test_registered_layout_point_sizes_follow_resolved_axis_area():
    from scarf.plotting._style import (
        default_point_size,
        refresh_layout_point_sizes,
        register_layout_point_size,
    )

    plt, _ = require_matplotlib()
    figure, axis = plt.subplots(figsize=(4.0, 3.0), layout="constrained")
    collection = axis.scatter([0.0, 1.0, 2.0], [0.0, 1.0, 0.0], s=99)
    register_layout_point_size(
        collection,
        n_points=5_000,
        size_min=2.0,
        size_max=20.0,
        multiplier=1.25,
    )

    refresh_layout_point_sizes(figure)

    bbox = axis.get_position()
    width, height = figure.get_size_inches()
    panel_area = float(bbox.width * width * bbox.height * height)
    expected = (
        default_point_size(
            5_000,
            panel_area=panel_area,
            size_min=2.0,
            size_max=20.0,
        )
        * 1.25
    )
    assert collection.get_sizes() == pytest.approx(np.full(3, expected))
    plt.close(figure)


def test_registered_point_sizes_can_refresh_marker_edges():
    from scarf.plotting._style import (
        default_point_edgewidth,
        refresh_layout_point_sizes,
        register_layout_point_size,
    )

    plt, mpl = require_matplotlib()
    figure, axis = plt.subplots(figsize=(4.0, 3.0), layout="constrained")
    derived = axis.scatter([0.0, 1.0], [0.0, 1.0], s=99, linewidths=3.0)
    explicit = axis.scatter([0.0, 1.0], [1.0, 0.0], s=99, linewidths=3.0)
    for collection, edgewidth in ((derived, None), (explicit, 0.0)):
        register_layout_point_size(
            collection,
            n_points=500,
            size_min=1.0,
            size_max=28.0,
            edgecolor="#333333",
            edgewidth=edgewidth,
        )

    refresh_layout_point_sizes(figure)

    size = float(derived.get_sizes()[0])
    assert derived.get_linewidths()[0] == pytest.approx(
        default_point_edgewidth(500, point_size=size)
    )
    assert tuple(derived.get_edgecolors()[0]) == pytest.approx(
        mpl.colors.to_rgba("#333333")
    )
    assert explicit.get_linewidths()[0] == 0.0
    plt.close(figure)


def test_color_limits_share_one_policy():
    from scarf.plotting._style import resolve_color_limits

    values = np.array([0.0, 1.0, 2.0, 100.0, np.nan, np.inf])
    assert resolve_color_limits(values, splt.ColorScale()) == (0.0, 100.0)
    assert resolve_color_limits(
        values,
        splt.ColorScale(quantiles=(0.25, 0.75)),
    ) == pytest.approx(tuple(np.quantile([0.0, 1.0, 2.0, 100.0], (0.25, 0.75))))
    assert resolve_color_limits(values, splt.ColorScale(vmin=1.0)) == (1.0, 100.0)
    assert resolve_color_limits([np.nan], splt.ColorScale()) == (0.0, 1.0)
    # Constant values take the low end of the map.
    assert resolve_color_limits([3.0, 3.0], splt.ColorScale()) == (3.0, 4.0)
    # A one-sided explicit bound keeps its side when the data lie beyond it.
    assert resolve_color_limits([5.0, 6.0], splt.ColorScale(vmax=2.0)) == (1.0, 2.0)
    assert resolve_color_limits([0.0, 1.0], splt.ColorScale(vmin=5.0)) == (5.0, 6.0)
    # A pivot widens derived limits so one-sided values still diverge.
    low, high = resolve_color_limits([1.0, 3.0], splt.ColorScale(vcenter=0.0))
    assert low < 0.0 < high == 3.0
    low, high = resolve_color_limits([0.0, 0.0], splt.ColorScale(vcenter=0.0))
    assert low < 0.0 < high
    with pytest.raises(ValueError, match="vcenter"):
        resolve_color_limits([1.0, 3.0], splt.ColorScale(vmin=0.5, vcenter=0.0))
    with pytest.raises(ValueError, match="positive values"):
        resolve_color_limits([0.0, 3.0], splt.ColorScale(scale="log"))
    with pytest.raises(ValueError, match="greater than vmin"):
        splt.ColorScale(vmin=4.0, vmax=4.0)


def test_continuous_norm_follows_color_scale_scale():
    _, mpl = require_matplotlib()
    log = continuous_norm(mpl, vmin=1.0, vmax=10.0, vcenter=None, scale="log")
    assert isinstance(log, mpl.colors.LogNorm)
    # The geometric midpoint of [1, 10] maps to the middle of the colormap.
    assert log(10**0.5) == pytest.approx(0.5)
    symlog = continuous_norm(mpl, vmin=-1.0, vmax=10.0, vcenter=None, scale="symlog")
    assert isinstance(symlog, mpl.colors.SymLogNorm)
    # The linear region spans 0.1% of the larger absolute limit.
    assert symlog.linthresh == pytest.approx(0.011)
    assert (symlog.vmin, symlog.vmax) == (-1.0, 10.0)
    with pytest.raises(ValueError, match="positive values"):
        continuous_norm(mpl, vmin=0.0, vmax=10.0, vcenter=None, scale="log")


def test_category_scale_shows_observed_categories_with_stable_colors():
    from scarf.plotting._style import resolve_category_scale

    natural = resolve_category_scale(
        np.array(["c10", None, "c2", "c1", np.nan], dtype=object),
        None,
    )
    assert natural.order == ("c1", "c2", "c10")
    assert natural.labels is None

    ordered = splt.CategoricalScale(order=("a", "b", "c"), labels={"a": "Alpha"})
    subset = resolve_category_scale(np.array(["c", "a"], dtype=object), ordered)
    full = resolve_category_scale(np.array(["a", "b", "c"], dtype=object), ordered)
    assert subset.order == ("a", "c")
    # Generated colors belong to the full explicit order, not the subset.
    assert subset.palette == {"a": full.palette["a"], "c": full.palette["c"]}
    assert subset.labels == {"a": "Alpha", "c": "c"}
    with pytest.raises(ValueError, match=r"categorical_scale\.order is missing"):
        resolve_category_scale(np.array(["d"], dtype=object), ordered)
    with pytest.raises(ValueError, match="duplicates"):
        resolve_category_scale(
            np.array(["a"], dtype=object),
            splt.CategoricalScale(order=("a", "a")),
        )


def test_missing_categories_include_nulls_but_not_containers():
    import pandas as pd

    from scarf.plotting._style import _is_missing_category, resolve_category_scale

    assert _is_missing_category(None)
    assert _is_missing_category(float("nan"))
    assert _is_missing_category(pd.NA)
    assert not _is_missing_category("a")
    # A sequence label has no single missingness answer.
    assert not _is_missing_category((1, None))
    assert not _is_missing_category([1, None])
    assert not _is_missing_category(np.array([np.nan, np.nan]))

    labels = np.empty(3, dtype=object)
    labels[:] = [("a", 1), None, ("b", 2)]
    assert resolve_category_scale(labels, None).order == (("a", 1), ("b", 2))


def test_padded_square_limits_and_colormap_palette():
    from scarf.plotting._style import colormap_palette, padded_square_limits

    xlim, ylim = padded_square_limits(
        np.array([0.0, 10.0, np.nan]),
        np.array([0.0, 2.0, 5.0]),
    )
    # Only rows finite on both axes count: x spans [0, 10], y spans [0, 2].
    assert xlim == pytest.approx((-0.5, 10.5))
    assert ylim == pytest.approx((-4.5, 6.5))
    with pytest.raises(ValueError, match="No finite coordinates"):
        padded_square_limits(np.array([np.nan]), np.array([1.0]))
    with pytest.raises(ValueError, match="matching shapes"):
        padded_square_limits(np.zeros(2), np.zeros(3))

    _, mpl = require_matplotlib()
    palette = colormap_palette(["a", "b", "c"], "viridis")
    colormap = mpl.colormaps["viridis"]
    assert palette == {
        "a": mpl.colors.to_hex(colormap(0.0)),
        "b": mpl.colors.to_hex(colormap(0.5)),
        "c": mpl.colors.to_hex(colormap(1.0)),
    }


@pytest.mark.parametrize(
    ("frame", "expected_xlabel", "expected_ylabel"),
    [
        ("axes", "UMAP 1", "UMAP 2"),
        ("minimal", "", ""),
        ("none", "", ""),
    ],
)
def test_finish_embedding_axes_applies_frame_contract(
    frame,
    expected_xlabel,
    expected_ylabel,
):
    from scarf.plotting._style import finish_embedding_axes

    plt, _ = require_matplotlib()
    figure, axis = plt.subplots()

    finish_embedding_axes(
        axis,
        xlim=(-2.0, 3.0),
        ylim=(-1.0, 4.0),
        xlabel="UMAP 1",
        ylabel="UMAP 2",
        title="Embedding",
        frame=frame,
    )

    assert axis.get_xlim() == pytest.approx((-2.0, 3.0))
    assert axis.get_ylim() == pytest.approx((-1.0, 4.0))
    assert axis.get_aspect() == pytest.approx(1.0)
    assert axis.get_box_aspect() == pytest.approx(1.0)
    assert axis.get_xticks().size == 0
    assert axis.get_yticks().size == 0
    assert axis.get_xlabel() == expected_xlabel
    assert axis.get_ylabel() == expected_ylabel
    assert axis.get_title() == "Embedding"
    if frame == "none":
        assert not any(spine.get_visible() for spine in axis.spines.values())
    plt.close(figure)


def test_axis_and_layout_helpers_reject_invalid_options():
    from scarf.plotting._style import (
        capped_figsize,
        finish_embedding_axes,
        foreground_color,
        resolve_legend_loc,
        square_axis_limits,
    )

    assert capped_figsize(9.0, 4.0, max_width=None) == (9.0, 4.0)
    with pytest.raises(ValueError, match="max_width must be positive"):
        capped_figsize(4.0, 3.0, max_width=0)
    with pytest.raises(ValueError, match="legend_loc"):
        resolve_legend_loc(3, "outside")
    assert foreground_color("notebook") == "#333333"
    assert foreground_color("dark") == "#e8e8e8"

    xlim, ylim = square_axis_limits((2.0, 2.0), (3.0, 3.0))
    assert sum(xlim) / 2 == pytest.approx(2.0)
    assert sum(ylim) / 2 == pytest.approx(3.0)
    assert xlim[1] - xlim[0] == pytest.approx(ylim[1] - ylim[0])
    assert xlim[1] > xlim[0]

    plt, _ = require_matplotlib()
    figure, axis = plt.subplots()
    with pytest.raises(ValueError, match="frame must be one of"):
        finish_embedding_axes(
            axis,
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.0),
            frame="invalid",
        )
    plt.close(figure)


def test_density_and_legend_helpers():
    from scarf.plotting._style import (
        capped_figsize,
        default_point_edgewidth,
        default_point_size,
        resolve_legend_loc,
    )
    from scarf.utils.arrays import sort_categories

    # Size 16 at 1000 cells on a 3.2 inch panel shrinks with sqrt(cells) and
    # grows with panel area to the 0.72 power, within [2, 28].
    assert default_point_size(1000) == pytest.approx(16.0)
    assert default_point_size(4000) == pytest.approx(8.0)
    assert default_point_size(20_000) == pytest.approx(16 * 0.05**0.5)
    assert default_point_size(1000, panel_area=3.2**2 / 4) == pytest.approx(
        16 * 0.25**0.72
    )
    assert default_point_size(100) == pytest.approx(28.0)
    assert default_point_edgewidth(500) == pytest.approx(0.15)
    assert default_point_edgewidth(20_000) == 0.0
    # On-data labels serve 13 to 40 categories.
    assert [resolve_legend_loc(n) for n in (12, 13, 40, 41)] == [
        "right",
        "on_data",
        "on_data",
        "right",
    ]
    assert resolve_legend_loc(20, "right") == "right"
    assert capped_figsize(20.0, 4.0)[0] == pytest.approx(7.5)
    assert sort_categories([1, 10, 2, "B", "A10", "A2"]) == [
        1,
        2,
        10,
        "A2",
        "A10",
        "B",
    ]
    assert sort_categories(["10", "2", "1"]) == ["1", "2", "10"]


def test_sort_categories_handles_numpy_booleans_and_missing_values():
    from scarf.utils.arrays import sort_categories

    ordered = sort_categories(
        [
            None,
            np.nan,
            np.bool_(True),
            "item10",
            np.float64(2.5),
            False,
            np.int64(2),
            "item2",
        ]
    )

    assert ordered[:6] == [2, 2.5, False, "item2", "item10", np.bool_(True)]
    assert ordered[-2] is None
    assert np.isnan(ordered[-1])


def test_embedding_on_data_legend():
    from types import SimpleNamespace

    class Cells:
        def __init__(self, **columns):
            self._columns = {name: np.asarray(value) for name, value in columns.items()}
            self.columns = tuple(self._columns)
            self.N = 15

        def fetch(self, column, key="I"):
            return self._columns[column]

        def fetch_all(self, column):
            return self._columns[column]

        def active_index(self, key="I"):
            return np.arange(self.N)

    offset = np.tile([0.0, 0.1, 0.2, 0.3, 0.4], 3)
    store = SimpleNamespace(
        cells=Cells(
            I=np.ones(15, dtype=bool),
            UMAP1=np.repeat([0.0, 10.0, 20.0], 5) + offset,
            UMAP2=np.repeat([0.0, 10.0, 5.0], 5) + offset,
            cluster=np.repeat(["c1", "c2", "c10"], 5).astype(object),
        ),
        _defaultAssay="RNA",
        zw=None,
        _stored_display_metadata=lambda _column: None,
    )

    result = splt.embedding(
        store,
        layout_key="UMAP",
        color_by="cluster",
        legend_loc="on_data",
        frame="none",
        show=False,
    )

    ax = result.axes["cluster"]
    # One label per category at its cells' median, placed from the lowest up.
    assert [(text.get_text(), text.get_position()) for text in ax.texts] == [
        ("c1", pytest.approx((0.2, 0.2))),
        ("c10", pytest.approx((20.2, 5.2))),
        ("c2", pytest.approx((10.2, 10.2))),
    ]
    assert ax.get_legend() is None
    assert not result.figure.legends
    assert ax.get_xlabel() == ""
    assert ax.get_ylabel() == ""
    assert not any(spine.get_visible() for spine in ax.spines.values())
    result.close()


def test_theme_context_restores_matplotlib_state():
    _, mpl = require_matplotlib()
    original = mpl.rcParams["font.size"]
    with theme_context("paper"):
        assert mpl.rcParams["font.size"] == 8
    assert mpl.rcParams["font.size"] == original


def test_theme_context_restores_state_after_error_and_rejects_unknown_theme():
    _, mpl = require_matplotlib()
    original = mpl.rcParams["font.size"]

    with pytest.raises(RuntimeError, match="plot failed"):
        with theme_context("paper"):
            assert mpl.rcParams["font.size"] == 8
            raise RuntimeError("plot failed")

    assert mpl.rcParams["font.size"] == original
    with pytest.raises(KeyError, match="Unknown theme"):
        with theme_context("missing-theme"):
            pass

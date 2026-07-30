"""Plot scale and theme behavior tests."""

import matplotlib
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


@pytest.mark.parametrize("size", [5, 15, 25, 50, 110])
def test_palette_for_n_returns_requested_colors(size):
    assert len(palette_for_n(size)) == size


@pytest.mark.parametrize("palette_name", ["default", "colorblind"])
@pytest.mark.parametrize("size", [8, 30, 110])
def test_palette_for_n_never_recycles_colors(size, palette_name):
    colors = palette_for_n(size, palette_name=palette_name)
    assert len(set(colors)) == size


def test_side_legend_columns_stay_page_sized():
    from scarf.plotting._style import LEGEND_SIDE_MAX_COLUMNS, legend_side_columns

    assert legend_side_columns(1) == 1
    assert legend_side_columns(40) == 2
    assert legend_side_columns(5_000) == LEGEND_SIDE_MAX_COLUMNS


def test_categorical_color_map_validates_custom_palette():
    with pytest.raises(KeyError, match="missing from palette"):
        categorical_color_map(["a", "b"], palette={"a": "red"})
    colors = categorical_color_map(
        ["a"],
        palette={"a": "red"},
        missing_label="NA",
        missing_color="gray",
    )
    assert colors == {"a": "red", "NA": "gray"}


def test_continuous_norm_supports_center_and_validates_bounds():
    _, mpl = require_matplotlib()
    norm = continuous_norm(mpl, vmin=-2, vmax=3, vcenter=0)
    assert norm.__class__.__name__ == "TwoSlopeNorm"
    with pytest.raises(ValueError, match="vcenter"):
        continuous_norm(mpl, vmin=0, vmax=3, vcenter=4)


def test_square_axis_limits_and_dark_theme():
    from scarf.plotting._style import (
        apply_figure_chrome,
        square_axis_limits,
        scatter_edgecolor,
        THEMES,
    )

    xlim, ylim = square_axis_limits((0.0, 2.0), (-1.0, 0.0))
    assert xlim[1] - xlim[0] == pytest.approx(ylim[1] - ylim[0])
    assert scatter_edgecolor("dark") == "#8f8f8f"
    assert THEMES["notebook"]["figure.facecolor"] == "white"
    assert THEMES["paper"]["savefig.transparent"] is False
    assert THEMES["dark"]["text.color"] == "#e8e8e8"
    with theme_context("dark"):
        _, mpl = require_matplotlib()
        assert mpl.rcParams["axes.edgecolor"] == "#e8e8e8"
    plt, _ = require_matplotlib()
    fig, ax = plt.subplots(1, 1)
    apply_figure_chrome(fig, "notebook")
    assert fig.patch.get_alpha() == 1.0
    assert ax.patch.get_alpha() == 1.0
    apply_figure_chrome(fig, "dark")
    assert fig.patch.get_alpha() == 0
    plt.close(fig)


def test_density_and_legend_helpers():
    from scarf.plotting._style import (
        capped_figsize,
        default_point_edgewidth,
        default_point_size,
        resolve_legend_loc,
        sort_categories,
    )

    assert default_point_size(100) > default_point_size(20_000)
    assert default_point_edgewidth(500) > 0
    assert default_point_edgewidth(20_000) == 0.0
    assert resolve_legend_loc(8) == "right"
    assert resolve_legend_loc(20) == "on_data"
    assert resolve_legend_loc(80) == "right"
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


def test_embedding_on_data_legend(umap, leiden_clustering, datastore):
    result = splt.embedding(
        datastore,
        layout_key="RNA_UMAP",
        color_by="RNA_leiden_cluster",
        legend_loc="on_data",
        frame="none",
        show=False,
    )
    ax = next(iter(result.axes.values()))
    assert ax.get_xlabel() == ""
    assert ax.get_ylabel() == ""
    assert len(ax.texts) >= 1
    assert not result.figure.legends
    result.close()


def test_embedding_panel_is_square(umap, leiden_clustering, datastore):
    result = splt.embedding(
        datastore,
        layout_key="RNA_UMAP",
        color_by="RNA_leiden_cluster",
        show=False,
    )
    ax = next(iter(result.axes.values()))
    result.figure.canvas.draw()
    bbox = ax.get_window_extent()
    assert ax.get_box_aspect() == pytest.approx(1.0)
    assert bbox.width == pytest.approx(bbox.height, rel=1e-3)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    assert (xlim[1] - xlim[0]) == pytest.approx(ylim[1] - ylim[0])
    result.close()


def test_theme_context_restores_matplotlib_state():
    _, mpl = require_matplotlib()
    original = mpl.rcParams["font.size"]
    with theme_context("paper"):
        assert mpl.rcParams["font.size"] == 8
    assert mpl.rcParams["font.size"] == original

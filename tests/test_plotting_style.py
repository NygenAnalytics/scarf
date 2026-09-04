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


def test_generated_palette_and_flat_norm_edge_cases():
    generated = categorical_color_map(
        ["b", "a"],
        missing_label="NA",
        missing_color="#cccccc",
    )

    assert list(generated) == ["b", "a", "NA"]
    assert list(generated.values())[:2] == palette_for_n(2)
    assert generated["NA"] == "#cccccc"
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


def test_sort_categories_handles_numpy_booleans_and_missing_values():
    from scarf.plotting._style import sort_categories

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


def test_embedding_on_data_legend(umap, leiden_clustering, datastore):
    result = splt.embedding(
        datastore,
        layout=umap,
        color_by=leiden_clustering,
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
        layout=umap,
        color_by=leiden_clustering,
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


def test_register_theme_validates_inputs_and_cleans_up_temporary_theme(monkeypatch):
    from scarf.plotting._style import THEMES, register_theme

    with pytest.raises(ValueError, match="non-empty"):
        register_theme("", {})
    with pytest.raises(ValueError, match="already exists"):
        register_theme("notebook", {})
    with pytest.raises(KeyError, match="Unknown base theme"):
        register_theme("wave4-invalid-base", {}, base="missing")
    with pytest.raises(KeyError, match="Unknown Matplotlib rcParams"):
        register_theme("wave4-invalid-key", {"not.a.real.rcparam": 1})

    name = "wave4-temporary-theme"
    monkeypatch.setitem(THEMES, name, {"font.size": 1})
    register_theme(
        name,
        {"font.size": 7.25},
        base="paper",
        overwrite=True,
    )
    assert THEMES[name]["font.size"] == pytest.approx(7.25)
    assert THEMES[name]["axes.labelsize"] == THEMES["paper"]["axes.labelsize"]

    register_theme(
        name,
        {"font.size": 11},
        base=None,
        overwrite=True,
    )
    assert THEMES[name] == {"font.size": 11}

"""Figure ownership and composition helper tests."""

import warnings

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from matplotlib.lines import Line2D

import scarf.plotting as splt
from scarf.plotting._figure import (
    PlotResult,
    _remove_child_legend_artists,
    compose_results,
    normalize_axes_target,
)


def _categorical_child(figure, axis, name: str, values: tuple[str, ...]) -> PlotResult:
    palette = {
        value: plt.get_cmap("tab20")(index % 20) for index, value in enumerate(values)
    }
    return PlotResult(
        figure=figure,
        axes={name: axis},
        tables={},
        legends=(splt.LegendSpec(kind="categorical", label=name),),
        scales=(splt.CategoricalScale(order=values, palette=palette),),
        provenance=splt.PlotProvenance(scarf_version="test"),
        owns_figure=False,
    )


def _plot_result(*, owns_figure: bool = True) -> PlotResult:
    fig, ax = plt.subplots()
    ax.plot([0, 1])
    return PlotResult(
        figure=fig,
        axes={"main": ax},
        tables={},
        legends=(),
        scales=(),
        provenance=splt.PlotProvenance(scarf_version="test"),
        owns_figure=owns_figure,
    )


def test_plot_result_show_uses_ipython_display_under_agg(monkeypatch):
    result = _plot_result()
    displayed = []
    show_calls = []
    assert result.figure.canvas.required_interactive_framework is None
    monkeypatch.setattr("IPython.get_ipython", lambda: object())
    monkeypatch.setattr("IPython.display.display", displayed.append)
    monkeypatch.setattr(plt, "show", lambda: show_calls.append(True))

    result.show()

    assert displayed == [result.figure]
    assert show_calls == []
    assert not plt.fignum_exists(result.figure.number)
    assert "rendered=True" in repr(result)


def test_plot_result_show_delegates_for_interactive_gui_canvas(monkeypatch):
    result = _plot_result()
    displayed = []
    show_calls = []
    monkeypatch.setattr("IPython.get_ipython", lambda: None)
    monkeypatch.setattr("IPython.display.display", displayed.append)
    monkeypatch.setattr(plt, "show", lambda: show_calls.append(True))
    monkeypatch.setattr(
        result.figure.canvas,
        "required_interactive_framework",
        "qt",
        raising=False,
    )

    result.show()

    assert displayed == []
    assert show_calls == [True]
    assert not plt.fignum_exists(result.figure.number)
    assert "rendered=True" in repr(result)


def test_plot_result_show_is_warning_free_noop_for_headless_agg(monkeypatch):
    result = _plot_result(owns_figure=False)
    monkeypatch.setattr("IPython.get_ipython", lambda: None)
    monkeypatch.setattr(
        plt,
        "show",
        lambda: pytest.fail("headless Agg must not call plt.show()"),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result.show()

    assert plt.fignum_exists(result.figure.number)
    assert "rendered=False" in repr(result)
    plt.close(result.figure)


def test_plot_result_show_closes_owned_figure_and_retains_metadata(monkeypatch):
    result = _plot_result()
    figure = result.figure
    axes = result.axes
    tables = result.tables
    legends = result.legends
    scales = result.scales
    provenance = result.provenance
    monkeypatch.setattr("IPython.get_ipython", lambda: object())
    monkeypatch.setattr("IPython.display.display", lambda _: None)

    result.show()

    assert not plt.fignum_exists(figure.number)
    assert result.figure is figure
    assert result.axes is axes
    assert result.tables is tables
    assert result.legends is legends
    assert result.scales is scales
    assert result.provenance is provenance

    caller_owned = _plot_result(owns_figure=False)
    caller_owned.close()
    assert plt.fignum_exists(caller_owned.figure.number)
    plt.close(caller_owned.figure)


def test_plot_result_can_save_after_show(monkeypatch, tmp_path):
    result = _plot_result()
    monkeypatch.setattr("IPython.get_ipython", lambda: object())
    monkeypatch.setattr("IPython.display.display", lambda _: None)

    result.show()
    output = result.save(tmp_path / "after-show.png", dpi=80)

    assert output.exists()
    assert output.stat().st_size > 0


def test_dark_plot_exports_with_readable_opaque_background(tmp_path):
    result = _plot_result()
    result.theme = "dark"

    output = result.save(tmp_path / "dark.png", dpi=60)
    pixels = plt.imread(output)

    assert float(pixels[0, 0, :3].mean()) < 0.2
    assert result.figure.patch.get_facecolor()[:3] == pytest.approx((1, 1, 1))
    result.close()


def test_plot_result_repr_is_compact_and_never_renders(monkeypatch):
    result = _plot_result()
    displayed = []
    show_calls = []
    monkeypatch.setattr("IPython.get_ipython", lambda: object())
    monkeypatch.setattr("IPython.display.display", displayed.append)
    monkeypatch.setattr(plt, "show", lambda: show_calls.append(True))

    unrendered = repr(result)

    assert unrendered == (
        "PlotResult(axes=1, tables=0, legends=0, scales=0, "
        "owns_figure=True, rendered=False)"
    )
    assert displayed == []
    assert show_calls == []
    assert plt.fignum_exists(result.figure.number)

    result.show()
    rendered = repr(result)

    assert rendered == (
        "PlotResult(axes=1, tables=0, legends=0, scales=0, "
        "owns_figure=True, rendered=True)"
    )
    assert "figure=<" not in rendered
    assert displayed == [result.figure]
    assert show_calls == []


def test_rendered_plot_result_publishes_no_extra_notebook_output(monkeypatch):
    from IPython.core.formatters import DisplayFormatter

    result = _plot_result()
    monkeypatch.setattr("IPython.get_ipython", lambda: object())
    monkeypatch.setattr("IPython.display.display", lambda _: None)
    result.show()

    data, metadata = DisplayFormatter().format(result)

    # An empty format bundle makes the IPython display hook publish nothing, so
    # the already-displayed figure stays the only output of the cell.
    assert data == {}
    assert metadata == {}


def test_unrendered_plot_result_still_displays_its_summary(monkeypatch):
    published = []
    monkeypatch.setattr("IPython.display.display", lambda *a, **k: published.append(a))
    result = _plot_result()

    result._ipython_display_()

    assert published == [({"text/plain": repr(result)},)]
    assert "rendered=False" in published[0][0]["text/plain"]
    plt.close(result.figure)


def test_normalize_axes_target_accepts_mappings_and_sequences():
    fig, axes = plt.subplots(1, 2)
    _, mapped, owns = normalize_axes_target(
        {"left": axes[0], "right": axes[1]},
        panel_keys=["left", "right"],
        figsize=None,
    )
    assert owns is False
    assert mapped == {"left": axes[0], "right": axes[1]}

    _, sequenced, owns = normalize_axes_target(
        axes,
        panel_keys=["left", "right"],
        figsize=None,
    )
    assert owns is False
    assert list(sequenced.values()) == list(axes)
    plt.close(fig)


def test_normalize_axes_target_rejects_invalid_ownership():
    fig_a, ax_a = plt.subplots()
    fig_b, ax_b = plt.subplots()
    with pytest.raises(KeyError, match="missing panel keys"):
        normalize_axes_target(
            {"left": ax_a},
            panel_keys=["left", "right"],
            figsize=None,
        )
    with pytest.raises(ValueError, match="same figure"):
        normalize_axes_target(
            [ax_a, ax_b],
            panel_keys=["left", "right"],
            figsize=None,
        )
    with pytest.raises(TypeError, match="single Axes"):
        normalize_axes_target(
            ax_a,
            panel_keys=["left", "right"],
            figsize=None,
        )
    plt.close(fig_a)
    plt.close(fig_b)


def test_shared_legends_stay_separated_beyond_three_scales():
    figure, axes = plt.subplots(2, 2, figsize=(6, 5), layout="constrained")
    flat = axes.ravel()
    children = [
        _categorical_child(figure, flat[index], f"scale{index}", ("a", "b", "c"))
        for index in range(4)
    ]

    result = compose_results(figure, children, panel_labels=False)

    assert len(figure.legends) == 3
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    boxes = [legend.get_window_extent(renderer) for legend in figure.legends]
    assert not any(
        boxes[first].overlaps(boxes[second])
        for first in range(len(boxes))
        for second in range(first + 1, len(boxes))
    )
    merged = [text.get_text() for text in figure.legends[-1].get_texts()]
    assert merged[:2] == ["scale2: a", "scale2: b"]
    assert "scale3: c" in merged
    result.close()
    plt.close(figure)


def test_tall_shared_legends_merge_instead_of_overlapping():
    figure, axes = plt.subplots(2, 2, figsize=(6, 5), layout="constrained")
    values = tuple(f"value{index}" for index in range(18))
    children = [
        _categorical_child(figure, axis, f"scale{index}", values)
        for index, axis in enumerate(axes.ravel()[:3])
    ]

    result = compose_results(figure, children, panel_labels=False)

    assert len(figure.legends) == 1
    assert not any(isinstance(artist, Legend) for artist in figure.artists)
    labels = [text.get_text() for text in figure.legends[0].get_texts()]
    assert labels[0] == "scale0: value0"
    assert labels[-1] == "scale2: value17"
    result.close()
    plt.close(figure)


def test_child_legend_removal_clears_all_axis_and_figure_legends():
    figure, ax = plt.subplots()
    first = ax.legend(
        handles=[Line2D([], [], label="first")],
        loc="upper left",
    )
    ax.add_artist(first)
    ax.legend(
        handles=[Line2D([], [], label="second")],
        loc="lower left",
    )
    figure.legend(handles=[Line2D([], [], label="figure")])

    _remove_child_legend_artists(figure, [ax])

    assert not any(isinstance(artist, Legend) for artist in ax.get_children())
    assert figure.legends == []
    plt.close(figure)


def test_panel_labels_and_legend_collection(umap, leiden_clustering, datastore):
    first = splt.embedding(
        datastore,
        layout_key="RNA_UMAP",
        color_by="RNA_leiden_cluster",
        show=False,
    )
    second = splt.embedding(
        datastore,
        layout_key="RNA_UMAP",
        color_by="RNA_nCounts",
        show=False,
    )
    axes = list(first.axes.values())
    splt.label_panels(first.axes)
    assert axes[0].texts[-1].get_text() == "A"
    with pytest.raises(ValueError, match="labels length"):
        splt.label_panels(first.axes, labels=["A", "B"])

    legends = splt.collect_legends(first.figure, [first, second])
    assert legends == first.legends + second.legends
    first.close()
    second.close()

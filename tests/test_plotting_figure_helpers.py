"""Figure ownership and composition helper tests."""

import json
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
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
    label_panels,
    normalize_axes_target,
)


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    yield
    plt.close("all")


def _child_result(
    figure,
    axis,
    *,
    legends=(),
    scales=(),
    tables=None,
    n_cells: int = 0,
    theme: str = "notebook",
) -> PlotResult:
    return PlotResult(
        figure=figure,
        axes={"main": axis},
        tables={} if tables is None else tables,
        legends=legends,
        scales=scales,
        provenance=splt.PlotProvenance(scarf_version="test", n_cells=n_cells),
        owns_figure=False,
        theme=theme,
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


def test_plot_result_save_restores_chrome_when_backend_raises(monkeypatch, tmp_path):
    result = _plot_result()
    result.theme = "dark"
    axis = result.axes["main"]
    result.figure.patch.set_facecolor("#aa0000")
    result.figure.patch.set_alpha(0.25)
    axis.patch.set_facecolor("#00aa00")
    axis.patch.set_alpha(0.4)
    figure_face = result.figure.patch.get_facecolor()
    figure_alpha = result.figure.patch.get_alpha()
    axis_face = axis.patch.get_facecolor()
    axis_alpha = axis.patch.get_alpha()
    observed = {}

    def fail_savefig(path, **kwargs):
        observed["path"] = Path(path)
        observed["figure_face"] = result.figure.patch.get_facecolor()
        observed["figure_alpha"] = result.figure.patch.get_alpha()
        observed["axis_face"] = axis.patch.get_facecolor()
        observed["axis_alpha"] = axis.patch.get_alpha()
        raise RuntimeError("backend failed")

    monkeypatch.setattr(result.figure, "savefig", fail_savefig)

    with pytest.raises(RuntimeError, match="backend failed"):
        result.save(tmp_path / "plot.png")

    dark_rgb = matplotlib.colors.to_rgb("#111111")
    assert observed["path"] == tmp_path / "plot.png"
    assert observed["figure_face"][:3] == pytest.approx(dark_rgb)
    assert observed["figure_alpha"] == 1.0
    assert observed["axis_face"][:3] == pytest.approx(dark_rgb)
    assert observed["axis_alpha"] == 1.0
    assert result.figure.patch.get_facecolor() == pytest.approx(figure_face)
    assert result.figure.patch.get_alpha() == figure_alpha
    assert axis.patch.get_facecolor() == pytest.approx(axis_face)
    assert axis.patch.get_alpha() == axis_alpha


def test_save_provenance_serializes_supported_metadata(tmp_path):
    result = _plot_result()
    result.figure.set_size_inches(2.5, 1.5)
    result.figure.set_dpi(80)
    result.tables["summary"] = pd.DataFrame({"group": ["a", "b"], 1: [2, 3]})
    result.legends = (
        splt.LegendSpec(
            kind="categorical",
            label="Group",
            extras={"source": tmp_path / "input.zarr", "flags": {"only"}},
        ),
    )
    result.scales = (
        splt.CategoricalScale(
            order=(1, 2),
            palette={1: "#111111", 2: "#222222"},
        ),
    )
    result.provenance = splt.PlotProvenance(
        scarf_version="test",
        n_cells=2,
        notes=("deterministic",),
        extras={
            "array": np.array([1, 2], dtype=np.int64),
            "scalar": np.float32(1.5),
            "path": tmp_path / "store.zarr",
            "nan": np.float32(np.nan),
            "positiveInfinity": float("inf"),
            "negativeInfinity": float("-inf"),
            7: "numeric key",
        },
    )

    output = result.save_provenance(
        tmp_path / "metadata.json",
        figure_path=tmp_path / "plot.SVG",
        dpi=144,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert output.read_text(encoding="utf-8").endswith("\n")
    assert payload["figure"] == {
        "dpi": 80.0,
        "height_inches": 1.5,
        "width_inches": 2.5,
    }
    assert payload["export"] == {
        "dpi": 144.0,
        "filename": "plot.SVG",
        "format": "svg",
    }
    assert payload["tables"]["summary"] == {
        "columns": ["group", "1"],
        "rows": 2,
    }
    assert payload["scales"][0]["type"] == "CategoricalScale"
    assert payload["scales"][0]["values"]["palette"] == {
        "1": "#111111",
        "2": "#222222",
    }
    assert payload["legends"][0]["extras"] == {
        "flags": ["only"],
        "source": str(tmp_path / "input.zarr"),
    }
    extras = payload["provenance"]["extras"]
    assert extras["array"] == [1, 2]
    assert extras["scalar"] == pytest.approx(1.5)
    assert extras["path"] == str(tmp_path / "store.zarr")
    assert extras["nan"] is None
    assert extras["positiveInfinity"] is None
    assert extras["negativeInfinity"] is None
    assert extras["7"] == "numeric key"


def test_save_provenance_rejects_unsupported_values_without_partial_file(tmp_path):
    result = _plot_result()
    result.provenance.extras["unsupported"] = object()
    output = tmp_path / "metadata.json"

    with pytest.raises(TypeError, match="object is not JSON serializable"):
        result.save_provenance(output)

    assert not output.exists()


def test_save_writes_default_sidecar_and_guards_against_path_collision(
    monkeypatch,
    tmp_path,
):
    result = _plot_result()

    def fake_savefig(path, **kwargs):
        Path(path).write_bytes(b"figure")

    monkeypatch.setattr(result.figure, "savefig", fake_savefig)
    output = result.save(
        tmp_path / "plot.png",
        dpi=72,
        provenance_sidecar=True,
    )
    sidecar = tmp_path / "plot.png.json"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))

    assert output.read_bytes() == b"figure"
    assert payload["export"] == {
        "dpi": 72.0,
        "filename": "plot.png",
        "format": "png",
    }

    collision = tmp_path / "collision.png"
    with pytest.raises(ValueError, match="must differ from figure path"):
        result.save(collision, provenance_sidecar=collision)


def test_save_validates_options_and_configures_transparent_tiff(monkeypatch, tmp_path):
    result = _plot_result()
    calls = []

    def fake_savefig(path, **kwargs):
        Path(path).write_bytes(b"figure")
        calls.append((Path(path), kwargs))

    monkeypatch.setattr(result.figure, "savefig", fake_savefig)

    with pytest.raises(ValueError, match="file extension"):
        result.save(tmp_path / "no-extension")
    with pytest.raises(ValueError, match="dpi must be positive"):
        result.save(tmp_path / "plot.png", dpi=0)
    with pytest.raises(ValueError, match="Unsupported export format"):
        result.save(tmp_path / "plot.unsupported")

    output = result.save(
        tmp_path / "plot.tiff",
        transparent=True,
        exact_size=False,
        tiff_compression="raw",
    )

    assert output.read_bytes() == b"figure"
    assert calls == [
        (
            output,
            {
                "transparent": True,
                "dpi": 300,
                "pil_kwargs": {"compression": "raw"},
                "bbox_inches": "tight",
            },
        )
    ]


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

    returned_figure, single, owns = normalize_axes_target(
        axes[0],
        panel_keys=["only"],
        figsize=None,
    )
    assert returned_figure is fig
    assert single == {"only": axes[0]}
    assert owns is False
    plt.close(fig)


def test_normalize_axes_target_creates_owned_single_and_sparse_grid():
    from scarf.plotting._style import DEFAULT_PANEL_INCHES

    single_figure, single_axes, owns = normalize_axes_target(
        None,
        panel_keys=["only"],
        figsize=None,
    )

    assert owns is True
    assert list(single_axes) == ["only"]
    assert single_axes["only"].figure is single_figure
    assert single_figure.get_size_inches() == pytest.approx(
        (DEFAULT_PANEL_INCHES + 0.4, DEFAULT_PANEL_INCHES)
    )

    keys = ["a", "b", "c", "d", "e"]
    grid_figure, grid_axes, owns = normalize_axes_target(
        None,
        panel_keys=keys,
        figsize=(5.0, 4.0),
        n_columns=2,
    )

    assert owns is True
    assert list(grid_axes) == keys
    assert len(grid_figure.axes) == len(keys)
    assert all(axis.figure is grid_figure for axis in grid_axes.values())
    assert grid_figure.get_size_inches() == pytest.approx((5.0, 4.0))
    plt.close(single_figure)
    plt.close(grid_figure)


def test_normalize_axes_target_rejects_invalid_ownership():
    fig_a, ax_a = plt.subplots()
    fig_b, ax_b = plt.subplots()
    with pytest.raises(ValueError, match="panel_keys must be non-empty"):
        normalize_axes_target(None, panel_keys=[], figsize=None)
    with pytest.raises(ValueError, match="figsize is invalid"):
        normalize_axes_target(
            ax_a,
            panel_keys=["only"],
            figsize=(2.0, 2.0),
        )
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
    with pytest.raises(ValueError, match="same figure"):
        normalize_axes_target(
            {"left": ax_a, "right": ax_b},
            panel_keys=["left", "right"],
            figsize=None,
        )
    with pytest.raises(ValueError, match="Expected 2 axes"):
        normalize_axes_target(
            [ax_a],
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


def test_label_panels_accepts_sequences_and_custom_typography():
    figure, axes = plt.subplots(1, 2)

    label_panels(
        axes,
        labels=["I", "II"],
        fontsize=13,
        fontweight="normal",
        x=0.1,
        y=0.9,
    )

    for axis, expected in zip(axes, ("I", "II")):
        label = axis.texts[-1]
        assert label.get_text() == expected
        assert label.get_position() == pytest.approx((0.1, 0.9))
        assert label.get_fontsize() == pytest.approx(13)
        assert label.get_fontweight() == "normal"
        assert label.get_transform() is axis.transAxes
    plt.close(figure)


def test_composition_renders_linear_and_centered_colorbars(monkeypatch):
    figure, axes = plt.subplots(1, 3, figsize=(8, 3), layout="constrained")
    linear_legend = splt.LegendSpec(
        kind="colorbar",
        label="Linear",
        extras={"vmin": 1.0, "vmax": 5.0},
    )
    linear_scale = splt.ColorScale(cmap="viridis", vmin=0.0, vmax=10.0)
    centered_legend = splt.LegendSpec(kind="colorbar", label="Centered")
    centered_scale = splt.ColorScale(
        cmap="coolwarm",
        vmin=-2.0,
        vmax=2.0,
        vcenter=0.0,
    )
    missing_legend = splt.LegendSpec(kind="colorbar", label="Missing limits")
    children = [
        _child_result(
            figure,
            axes[0],
            legends=(linear_legend,),
            scales=(linear_scale,),
        ),
        _child_result(
            figure,
            axes[1],
            legends=(centered_legend,),
            scales=(centered_scale,),
        ),
        _child_result(
            figure,
            axes[2],
            legends=(linear_legend,),
            scales=(linear_scale,),
        ),
        _child_result(
            figure,
            axes[2],
            legends=(missing_legend,),
            scales=(splt.ColorScale(cmap="magma"),),
        ),
    ]
    calls = []
    original_colorbar = figure.colorbar

    def capture_colorbar(mappable, *args, **kwargs):
        calls.append((mappable, kwargs.copy()))
        return original_colorbar(mappable, *args, **kwargs)

    monkeypatch.setattr(figure, "colorbar", capture_colorbar)

    result = compose_results(figure, children, panel_labels=False)

    assert len(calls) == 2
    linear_mappable, linear_kwargs = calls[0]
    centered_mappable, centered_kwargs = calls[1]
    assert type(linear_mappable.norm) is matplotlib.colors.Normalize
    assert linear_mappable.norm.vmin == pytest.approx(1.0)
    assert linear_mappable.norm.vmax == pytest.approx(5.0)
    assert isinstance(centered_mappable.norm, matplotlib.colors.TwoSlopeNorm)
    assert centered_mappable.norm.vmin == pytest.approx(-2.0)
    assert centered_mappable.norm.vcenter == pytest.approx(0.0)
    assert centered_mappable.norm.vmax == pytest.approx(2.0)
    for kwargs in (linear_kwargs, centered_kwargs):
        assert kwargs["location"] == "bottom"
        assert kwargs["orientation"] == "horizontal"
        assert kwargs["shrink"] == pytest.approx(0.45)
    main_axes = set(axes)
    colorbar_labels = [
        axis.get_xlabel() for axis in figure.axes if axis not in main_axes
    ]
    assert colorbar_labels == ["Linear", "Centered"]
    assert result.legends == (linear_legend, centered_legend, missing_legend)
    plt.close(figure)


def test_composition_renders_deduplicated_size_legend():
    figure, axes = plt.subplots(1, 2, figsize=(6, 4), layout="constrained")
    legend_spec = splt.LegendSpec(kind="size", label="Detection")
    size_scale = splt.SizeScale(
        vmin=0.0,
        vmax=1.0,
        size_min=16.0,
        size_max=400.0,
    )
    children = [
        _child_result(
            figure,
            axis,
            legends=(legend_spec,),
            scales=(size_scale,),
        )
        for axis in axes
    ]

    result = compose_results(figure, children, panel_labels=False)

    assert len(figure.legends) == 1
    legend = figure.legends[0]
    assert legend.get_title().get_text() == "Detection"
    assert [text.get_text() for text in legend.get_texts()] == [
        "0%",
        "33%",
        "67%",
        "100%",
    ]
    values = np.linspace(0.0, 1.0, 4)
    areas = size_scale.areas(values)
    area_factor = min(1.0, 180.0 / float(areas.max()))
    expected_sizes = np.sqrt(areas * area_factor)
    observed_sizes = [handle.get_markersize() for handle in legend.legend_handles]
    assert observed_sizes == pytest.approx(expected_sizes)
    assert result.legends == (legend_spec,)
    plt.close(figure)


def test_composition_renders_marker_legend_and_skips_malformed_specs():
    figure, axes = plt.subplots(1, 3, figsize=(7, 4), layout="constrained")
    marker_spec = splt.LegendSpec(
        kind="marker",
        label="Condition",
        extras={
            "values": ("control", "treated"),
            "markers": ("s", "^"),
        },
    )
    malformed = splt.LegendSpec(
        kind="marker",
        label="Malformed",
        extras={"values": ("one", "two"), "markers": ("o",)},
    )
    children = [
        _child_result(figure, axes[0], legends=(marker_spec,)),
        _child_result(figure, axes[1], legends=(marker_spec,)),
        _child_result(figure, axes[2], legends=(malformed,)),
    ]

    result = compose_results(figure, children, panel_labels=False)

    assert len(figure.legends) == 1
    legend = figure.legends[0]
    assert legend.get_title().get_text() == "Condition"
    assert [text.get_text() for text in legend.get_texts()] == [
        "control",
        "treated",
    ]
    assert [handle.get_marker() for handle in legend.legend_handles] == ["s", "^"]
    assert result.legends == (marker_spec, malformed)
    plt.close(figure)


def test_mixed_shared_legends_use_distinct_nonoverlapping_slots():
    figure, axis = plt.subplots(figsize=(8, 8), layout="constrained")
    categorical = _categorical_child(figure, axis, "Group", ("a", "b"))
    sized = _child_result(
        figure,
        axis,
        legends=(
            splt.LegendSpec(
                kind="size",
                label="Magnitude",
                extras={"domain": (2.0, 8.0)},
            ),
        ),
        scales=(
            splt.SizeScale(
                vmin=2.0,
                vmax=8.0,
                size_min=10.0,
                size_max=100.0,
            ),
        ),
    )
    marked = _child_result(
        figure,
        axis,
        legends=(
            splt.LegendSpec(
                kind="marker",
                label="Condition",
                extras={
                    "values": ("control", "treated"),
                    "markers": ("o", "D"),
                },
            ),
        ),
    )
    colored = _child_result(
        figure,
        axis,
        legends=(splt.LegendSpec(kind="colorbar", label="Score"),),
        scales=(splt.ColorScale(cmap="viridis", vmin=0.0, vmax=1.0),),
    )

    compose_results(
        figure,
        [categorical, categorical, sized, marked, colored],
        panel_labels=False,
    )

    assert [legend.get_title().get_text() for legend in figure.legends] == [
        "Group",
        "Magnitude",
        "Condition",
    ]
    assert [text.get_text() for text in figure.legends[1].get_texts()] == [
        "2",
        "4",
        "6",
        "8",
    ]
    colorbar_axes = [candidate for candidate in figure.axes if candidate is not axis]
    assert [candidate.get_xlabel() for candidate in colorbar_axes] == ["Score"]
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    boxes = [legend.get_window_extent(renderer) for legend in figure.legends]
    centers = [(box.y0 + box.y1) / 2 for box in boxes]
    assert centers == sorted(centers, reverse=True)
    assert not any(
        boxes[first].overlaps(boxes[second])
        for first in range(len(boxes))
        for second in range(first + 1, len(boxes))
    )
    plt.close(figure)


def test_composition_namespaces_metadata_and_respects_caller_options():
    figure, axes = plt.subplots(1, 2)
    shared_legend = splt.LegendSpec(kind="categorical", label="Group")
    left_table = pd.DataFrame({"value": [1]})
    right_table = pd.DataFrame({"value": [2]})
    left = _child_result(
        figure,
        axes[0],
        legends=(shared_legend,),
        tables={"summary": left_table},
        n_cells=4,
    )
    right = _child_result(
        figure,
        axes[1],
        legends=(shared_legend,),
        tables={"summary": right_table},
        n_cells=7,
    )

    result = compose_results(
        figure,
        {"left": left, "right": right},
        panel_labels=["L", "R"],
        shared_legends=False,
        owns_figure=True,
        theme="paper",
    )

    assert result.axes == {
        ("left", "main"): axes[0],
        ("right", "main"): axes[1],
    }
    assert result.tables == {
        "left:summary": left_table,
        "right:summary": right_table,
    }
    assert result.legends == (shared_legend,)
    assert result.provenance.n_cells == 7
    assert result.provenance.notes == ("composite",)
    assert result.provenance.extras["shared_legends"] is False
    assert set(result.provenance.extras["children"]) == {"left", "right"}
    assert result.theme == "paper"
    assert [axes[0].texts[-1].get_text(), axes[1].texts[-1].get_text()] == [
        "L",
        "R",
    ]
    assert figure.legends == []
    figure_number = figure.number
    result.close()
    assert not plt.fignum_exists(figure_number)


def test_composition_rejects_empty_and_foreign_results():
    figure, axis = plt.subplots()
    other_figure, other_axis = plt.subplots()
    foreign = _child_result(other_figure, other_axis)

    with pytest.raises(ValueError, match="results must be non-empty"):
        compose_results(figure, [])
    with pytest.raises(ValueError, match="supplied figure"):
        compose_results(figure, [foreign])

    plt.close(figure)
    plt.close(other_figure)


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
    colorbar = figure.colorbar(ax.imshow([[0.0, 1.0], [1.0, 0.0]]), ax=ax)

    _remove_child_legend_artists(figure, [ax])

    assert not any(isinstance(artist, Legend) for artist in ax.get_children())
    assert figure.legends == []
    assert colorbar.ax not in figure.axes
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

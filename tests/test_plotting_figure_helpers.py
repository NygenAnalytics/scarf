"""Figure ownership and composition helper tests."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import scarf.plotting as splt
from scarf.plotting._figure import PlotResult, as_2d_axes_array, normalize_axes_target


def test_plot_result_show_displays_once_and_closes_owned_inline_figure(monkeypatch):
    fig, ax = plt.subplots()
    ax.plot([0, 1])
    displayed = []
    monkeypatch.setattr(
        matplotlib,
        "get_backend",
        lambda: "module://matplotlib_inline.backend_inline",
    )
    monkeypatch.setattr("IPython.display.display", displayed.append)
    result = PlotResult(
        figure=fig,
        axes={"main": ax},
        tables={},
        legends=(),
        scales=(),
        provenance=splt.PlotProvenance(scarf_version="test"),
        owns_figure=True,
    )

    result.show()

    assert displayed == [fig]
    assert not plt.fignum_exists(fig.number)


def test_plot_result_show_displays_under_agg_when_ipython_active(monkeypatch):
    fig, ax = plt.subplots()
    ax.plot([0, 1])
    displayed = []
    monkeypatch.setattr(matplotlib, "get_backend", lambda: "Agg")
    monkeypatch.setattr("IPython.get_ipython", lambda: object())
    monkeypatch.setattr("IPython.display.display", displayed.append)
    result = PlotResult(
        figure=fig,
        axes={"main": ax},
        tables={},
        legends=(),
        scales=(),
        provenance=splt.PlotProvenance(scarf_version="test"),
        owns_figure=True,
    )

    result.show()

    assert displayed == [fig]
    assert not plt.fignum_exists(fig.number)


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


def test_panel_labels_legend_collection_and_axes_normalization(
    umap, leiden_clustering, datastore
):
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
    assert as_2d_axes_array(axes[0]).shape == (1, 1)
    assert as_2d_axes_array(np.asarray(axes)).shape == (1, 1)
    with pytest.raises(ValueError, match="in_ax is None"):
        as_2d_axes_array(None)
    first.close()
    second.close()

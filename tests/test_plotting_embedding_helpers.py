from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest

from scarf.plotting import DensityOverlay, Highlight
from scarf.plotting.embedding import (
    _color_labels,
    _density_selection_mask,
    _draw_density_overlay,
    _draw_highlight,
    _embedding_panel_keys,
    _multi_layout_facets,
    _resolve_highlight_mask,
    _retain_strongest_hotspots,
    _scatter_edges,
    _smoothed_local_mean,
    _soft_clip,
    _weighted_quantiles,
)

embedding_module = import_module("scarf.plotting.embedding")


def test_embedding_helper_labels_duplicate_panels_and_explicit_facets(monkeypatch):
    store = SimpleNamespace(cells=SimpleNamespace(columns=("group",)))
    monkeypatch.setattr(
        embedding_module,
        "resolve_feature",
        lambda *_args, **_kwargs: SimpleNamespace(label="resolved feature"),
    )

    assert _color_labels(store, [None, "group", "gene"], from_assay=None) == [
        "cells",
        "group",
        "resolved feature",
    ]
    assert _embedding_panel_keys(["same", "same"], [None]) == [
        (0, "same"),
        (1, "same"),
    ]
    assert _multi_layout_facets(
        store,
        facet_by="group",
        facet_order=("b", "a"),
        groups=None,
        subset_by=None,
        cell_key="I",
    ) == ["b", "a"]


def test_highlight_and_density_masks_validate_selected_metadata(monkeypatch):
    store = object()
    with pytest.raises(IndexError, match="outside the selected cell range"):
        _resolve_highlight_mask(
            store,
            Highlight(indices=(3,)),
            cell_key="I",
            n_cells=3,
        )

    monkeypatch.setattr(
        embedding_module,
        "_selected_metadata_column",
        lambda *_args, **_kwargs: np.array([True]),
    )
    with pytest.raises(ValueError, match="highlight metadata length"):
        _resolve_highlight_mask(
            store,
            Highlight(by="selected"),
            cell_key="I",
            n_cells=2,
        )
    with pytest.raises(ValueError, match="density metadata length"):
        _density_selection_mask(
            store,
            DensityOverlay(group_by="group"),
            cell_key="I",
            n_cells=2,
        )

    monkeypatch.setattr(
        embedding_module,
        "_selected_metadata_column",
        lambda *_args, **_kwargs: np.array([1, 0]),
    )
    with pytest.raises(TypeError, match="requires a boolean metadata column"):
        _resolve_highlight_mask(
            store,
            Highlight(by="selected"),
            cell_key="I",
            n_cells=2,
        )
    np.testing.assert_array_equal(
        _resolve_highlight_mask(
            store,
            Highlight(by="group", groups=(1,)),
            cell_key="I",
            n_cells=2,
        ),
        [True, False],
    )
    np.testing.assert_array_equal(
        _density_selection_mask(
            store,
            DensityOverlay(group_by="group"),
            cell_key="I",
            n_cells=2,
        ),
        [True, True],
    )


def test_embedding_numeric_helpers_cover_degenerate_inputs():
    assert _scatter_edges("black", 0) == ("none", 0.0)
    np.testing.assert_allclose(
        _weighted_quantiles(
            np.array([3.0, 1.0, 2.0]),
            np.zeros(3),
            np.array([0.5]),
        ),
        [2.0],
    )
    with pytest.raises(ValueError, match="contour values must match"):
        _smoothed_local_mean(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([1.0]),
            grid_pixels=16,
            x_range=(0.0, 1.0),
            y_range=(0.0, 1.0),
            sigma=1.0,
            min_support=0.25,
        )

    values = np.array([1.0, np.nan, 3.0])
    assert _soft_clip(values, 0) is values
    np.testing.assert_array_equal(
        np.isnan(_soft_clip(np.array([np.nan, np.inf]), 0.1)),
        [True, False],
    )


def test_hotspot_filter_retains_only_the_strongest_component():
    surface = np.array(
        [
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0],
        ]
    )
    support = np.ones_like(surface)

    unchanged = _retain_strongest_hotspots(
        surface,
        support,
        level=1.0,
        max_hotspots=2,
    )
    np.testing.assert_array_equal(unchanged, surface)

    filtered = _retain_strongest_hotspots(
        surface,
        support,
        level=1.0,
        max_hotspots=1,
    )
    assert filtered[2, 2] == 3.0
    assert filtered[0, 0] < 1.0


def test_density_and_highlight_draw_helpers_return_early_without_points():
    ax = SimpleNamespace()

    assert (
        _draw_density_overlay(
            ax,
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            overlay=DensityOverlay(),
            values=None,
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.0),
            theme="light",
        )
        is None
    )
    assert (
        _draw_highlight(
            ax,
            np.array([]),
            np.array([]),
            np.array([]),
            highlight=Highlight(indices=()),
            edgecolor="black",
            rasterized=False,
        )
        is None
    )

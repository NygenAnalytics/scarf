"""Final maps fail visibly on invalid artifacts and retain dense population legends."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest

from scarf.agent import _plots
from scarf.storage.refs import ArtifactRef
from tests.test_agent_analysis_plots import display_store


@pytest.mark.parametrize("limit", [False, 0, 50_001, 1.5])
def test_invalid_display_limit_fails_before_artifact_reads(
    monkeypatch: pytest.MonkeyPatch, limit: Any
) -> None:
    store, refs, arrays = display_store(monkeypatch, n=12)
    with pytest.raises(ValueError, match="max_points"):
        _plots.plot_final_umap(store, **refs, max_points=limit, show=False)
    assert not any(array.read_sizes for array in arrays.values())


@pytest.mark.parametrize(
    "damage",
    [
        "incomplete",
        "missingSelection",
        "wrongAssay",
        "wrongKind",
        "nonNumeric",
        "empty",
        "graphSelection",
    ],
)
def test_final_map_validates_all_artifact_boundaries(
    monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    store, refs, arrays = display_store(monkeypatch, n=0 if damage == "empty" else 12)
    inspect = store.inspect_artifact

    def changed(ref: ArtifactRef) -> Any:
        status = inspect(ref)
        if damage == "incomplete":
            status.complete = False
        elif damage == "missingSelection":
            status.inputs.pop("cell_selection")
        return status

    store.inspect_artifact = changed
    if damage == "wrongAssay":
        refs["umap"] = ArtifactRef("assay", "embedding", "4" * 64, "RNA")
    elif damage == "wrongKind":
        refs["umap"] = refs["graph"]
    elif damage == "nonNumeric":
        arrays["umap"].dtype = np.dtype("U8")
    elif damage == "graphSelection":
        monkeypatch.setattr(
            _plots,
            "graph_cell_selection",
            lambda *_: ArtifactRef("datastore", "cell_selection", "f" * 64),
        )
    with pytest.raises((ValueError, TypeError)):
        _plots.plot_final_umap(store, **refs, show=False)
    assert not any(array.read_sizes for array in arrays.values())


def test_many_clusters_are_annotated_and_plot_failure_closes_figure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, refs, arrays = display_store(monkeypatch, n=82)
    arrays["clusters"] = np.repeat(np.arange(41), 2)
    plot = _plots.plot_final_umap(store, **refs, show=False)
    try:
        assert len(plot.axes["clusters"].texts) == 41
        assert len(plot.scales[0].palette) == 41
        assert plot.tables["cluster_counts"]["cells"].sum() == 82
    finally:
        plot.close()
    before = set(plt.get_fignums())
    from matplotlib.axes import Axes

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("rendering interrupted")

    monkeypatch.setattr(Axes, "scatter", fail)
    with pytest.raises(RuntimeError, match="rendering interrupted"):
        _plots.plot_final_umap(store, **refs, show=False)
    assert set(plt.get_fignums()) == before


def test_display_cannot_hide_clusters_to_fit_an_impossible_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, refs, arrays = display_store(monkeypatch, n=12)
    arrays["clusters"] = np.arange(12)
    with pytest.raises(RuntimeError, match="more clusters"):
        _plots.plot_final_umap(store, **refs, max_points=10, show=False)
    assert not arrays["umap"].read_sizes


def test_nonfinite_umap_fails_without_leaking_a_figure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, refs, arrays = display_store(monkeypatch, n=12)
    coordinates = np.zeros((12, 2))
    coordinates[3, 1] = np.nan
    arrays["umap"] = coordinates
    before = set(plt.get_fignums())
    with pytest.raises(ValueError, match="finite"):
        _plots.plot_final_umap(store, **refs, show=False)
    assert set(plt.get_fignums()) == before


def test_mid_sized_population_palette_and_default_display_use_existing_plot_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scarf.plotting import PlotResult

    store, refs, arrays = display_store(monkeypatch, n=30)
    arrays["clusters"] = np.repeat(np.arange(15), 2)
    shown = []
    monkeypatch.setattr(PlotResult, "show", lambda self: shown.append(self))
    plot = _plots.plot_final_umap(store, **refs)
    try:
        assert shown == [plot]
        assert len(plot.scales[0].palette) == 15
        assert len(plot.axes["clusters"].get_legend().get_texts()) == 15
    finally:
        plot.close()


def test_cluster_count_summary_rejects_matrix_valued_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, refs, arrays = display_store(monkeypatch, n=12)
    arrays["clusters"] = np.ones((12, 2), dtype=int)
    with pytest.raises(ValueError, match="one-dimensional"):
        _plots.cluster_counts(store, refs["clusters"])

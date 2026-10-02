"""Marker displays use saved statistics aligned by immutable feature identity."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from scarf.agent.plots import marker_dotplot
from scarf.plotting import PlotResult
from scarf.storage.artifacts import ArtifactRef


class _Run(dict[str, ArtifactRef]):
    run_id = "final-run"
    assay = "RNA"

    def __init__(self) -> None:
        super().__init__(
            markers=ArtifactRef("assay", "marker_table", "a" * 64, "RNA"),
            clusters=ArtifactRef("assay", "cluster_labels", "b" * 64, "RNA"),
        )
        self.cells = SimpleNamespace(fetch=self.fetch)

    @staticmethod
    def fetch(key: str) -> np.ndarray:
        assert key == "clusters"
        return np.array(["10", "2", "1", "10", "1", "2"])


def _table(
    group: str, rows: list[tuple[int, str, float, float, float]]
) -> pd.DataFrame:
    frame = pd.DataFrame(
        rows, columns=["feature_index", "feature_name", "score", "mean", "frac_exp"]
    )
    frame.insert(0, "group_id", group)
    return frame


class _Store:
    """Only the public marker reader is exposed, so raw count reads fail."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.tables = {
            "1": _table(
                "1",
                [
                    (2, "Shared", 0.9, 4.0, 0.8),
                    (3, "Shared", 0.8, 2.0, 0.5),
                    (4, "Later", 0.2, 0.0, 0.0),
                ],
            ),
            "2": _table(
                "2", [(4, "Later", 0.8, 5.0, 0.9), (2, "Shared", 0.1, 1.0, 0.1)]
            ),
            "10": _table(
                "10",
                [
                    (5, "Last", 0.7, 3.0, 0.7),
                    (3, "Shared", 0.6, 2.0, 0.5),
                    (2, "Shared", 0.1, 0.0, 0.0),
                    (4, "Later", 0.1, 1.0, 0.1),
                ],
            ),
        }

    def get_markers(self, **kwargs: Any) -> pd.DataFrame:
        self.calls.append(kwargs)
        return self.tables[kwargs["group_id"]].copy()


def test_marker_panel_uses_frozen_groups_and_rank_round_robin() -> None:
    store, run = _Store(), _Run()
    result = marker_dotplot(store, run, top_n=2, max_genes=3)
    try:
        assert isinstance(result, PlotResult)
        assert result.provenance.extras["featureIndices"] == [2, 4, 5]
        assert result.provenance.extras["groupOrder"] == ["1", "2", "10"]
        assert result.provenance.extras["markers"] == run["markers"].to_dict()
        assert result.provenance.extras["clusters"] == run["clusters"].to_dict()
        assert result.provenance.extras["runId"] == run.run_id
        assert result.provenance.n_cells == 6
        assert result.figure.dpi == 200
        assert [call["group_id"] for call in store.calls] == ["1", "2", "10"] * 2
        assert all(
            call
            == {
                "marker": run["markers"],
                "group_id": call["group_id"],
                "min_score": -1,
                "min_frac_exp": -1,
            }
            for call in store.calls
        )
        table = result.tables["markers"].set_index(["group_id", "feature_index"])
        assert table.loc[("1", 4), "frac_exp"] == 0
        assert np.isnan(table.loc[("1", 5), "frac_exp"])
        assert table.loc[("2", 2), "mean"] == 1
        assert len(table) == 9
        np.testing.assert_allclose(table.log1p_mean, np.log1p(table["mean"]))
        measured = table.loc[table["mean"].notna()]
        dots = result.axes["markers"].collections[0]
        np.testing.assert_allclose(dots.get_sizes(), measured.frac_exp * 140)
        np.testing.assert_allclose(dots.get_array(), measured.log1p_mean)
        assert dots.norm.vmin == 0
        assert "Not measured" in [
            text.get_text() for text in result.axes["markers"].legend_.texts
        ]
        assert (
            result.figure.axes[1].get_xlabel() == "log(1 + mean normalized expression)"
        )
    finally:
        result.close()


def test_duplicate_feature_names_have_distinct_labels_without_aggregation() -> None:
    result = marker_dotplot(_Store(), _Run())
    try:
        table = result.tables["markers"]
        assert table.loc[
            table.feature_index == 2, "feature_label"
        ].unique().tolist() == ["Shared [2]"]
        assert table.loc[
            table.feature_index == 3, "feature_label"
        ].unique().tolist() == ["Shared [3]"]
        assert result.provenance.extras["featureIndices"] == [2, 4, 5, 3]
    finally:
        result.close()


@pytest.mark.parametrize(
    "options",
    [
        {"top_n": 0},
        {"top_n": 11},
        {"top_n": True},
        {"top_n": 1.5},
        {"max_genes": 0},
        {"max_genes": 61},
        {"max_genes": False},
        {"max_genes": "2"},
    ],
)
def test_panel_limits_fail_before_any_store_read(options: dict[str, Any]) -> None:
    store = _Store()
    with pytest.raises(ValueError, match="must be an integer"):
        marker_dotplot(store, _Run(), **options)
    assert store.calls == []


def test_no_qualifying_markers_is_explicit_and_does_not_create_figure() -> None:
    store = _Store()
    for table in store.tables.values():
        table["score"] = 0.1
    before = plt.get_fignums()
    with pytest.raises(ValueError, match="No markers meet"):
        marker_dotplot(store, _Run())
    assert len(store.calls) == 3
    assert plt.get_fignums() == before


@pytest.mark.parametrize(
    "column,value",
    [
        ("mean", np.inf),
        ("mean", -1),
        ("frac_exp", np.nan),
        ("frac_exp", 1.5),
        ("frac_exp", -0.2),
    ],
)
def test_invalid_saved_statistics_are_not_plotted(column: str, value: float) -> None:
    store = _Store()
    store.tables["10"].loc[2, column] = value
    with pytest.raises(ValueError, match="must be finite and valid"):
        marker_dotplot(store, _Run())


def test_show_is_explicit_and_plot_failure_closes_only_its_figure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shown = []
    monkeypatch.setattr(PlotResult, "show", lambda self: shown.append(self))
    result = marker_dotplot(_Store(), _Run(), show=True)
    try:
        assert shown == [result]
    finally:
        result.close()
    existing = plt.figure()
    figures_before_failure = set(plt.get_fignums())

    def fail(self: PlotResult) -> None:
        raise RuntimeError("display unavailable")

    monkeypatch.setattr(PlotResult, "show", fail)
    try:
        with pytest.raises(RuntimeError, match="display unavailable"):
            marker_dotplot(_Store(), _Run(), show=True)
        assert set(plt.get_fignums()) == figures_before_failure
    finally:
        plt.close(existing)


def test_complete_marker_panel_has_no_missing_measurements_legend() -> None:
    store = _Store()
    for group in store.tables:
        store.tables[group] = _table(group, [(2, "Shared", 0.9, 1.0, 0.2)])
    result = marker_dotplot(store, _Run(), max_genes=1)
    try:
        assert not result.tables["markers"].isna().any().any()
        assert "Not measured" not in [
            text.get_text() for text in result.axes["markers"].legend_.texts
        ]
        result.figure.canvas.draw()
    finally:
        result.close()

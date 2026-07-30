import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import scarf.plotting as splt
import scarf.plotting.diagnostics as diagnostics
from scarf.plotting._figure import PlotResult


class _KneeLocator:
    def __init__(self, x, y, **kwargs):
        self.elbow = 1


@pytest.fixture
def deterministic_kneed(monkeypatch):
    monkeypatch.setattr(diagnostics, "require_kneed", lambda: _KneeLocator)


def _qc_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "groups": ["a"] * 4 + ["b"] * 4,
            "nCounts": [10.0, 12.0, 11.0, 13.0, 20.0, 22.0, 21.0, 23.0],
            "nFeatures": [4.0, 5.0, 5.0, 6.0, 8.0, 9.0, 9.0, 10.0],
        }
    )


def _graph():
    return sparse.csr_matrix(
        np.array(
            [
                [0.0, 0.5, 0.0],
                [0.5, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
    )


def _hvg_arguments() -> tuple[np.ndarray, ...]:
    return (
        np.array([1.0, 2.0, 4.0, 8.0]),
        np.array([2.0, 3.0, 8.0, 12.0]),
        np.array([10, 20, 30, 40]),
        np.array([False, True, False, True]),
    )


def test_diagnostics_show_default_and_suppression(
    monkeypatch,
    deterministic_kneed,
):
    shown = []

    def track_show(result):
        shown.append(result)

    monkeypatch.setattr(PlotResult, "show", track_show)
    defaults = [
        splt.qc(_qc_data(), max_points=0),
        splt.elbow([0.5, 0.3, 0.2]),
        splt.graph_qc(_graph()),
        splt.highly_variable_features(*_hvg_arguments()),
    ]
    suppressed = [
        splt.qc(_qc_data(), max_points=0, show=False),
        splt.elbow([0.5, 0.3, 0.2], show=False),
        splt.graph_qc(_graph(), show=False),
        splt.highly_variable_features(*_hvg_arguments(), show=False),
    ]

    assert shown == defaults
    for result in defaults + suppressed:
        result.close()


def test_qc_result_contains_tables_metadata_and_artists():
    data = _qc_data()
    result = splt.qc(data, max_points=5, seed=13, show=False)

    assert isinstance(result, PlotResult)
    assert result.owns_figure is True
    assert list(result.axes) == ["nCounts", "nFeatures"]
    pd.testing.assert_frame_equal(result.tables["data"], data)
    assert set(result.tables["summary"]["metric"]) == {"nCounts", "nFeatures"}
    assert result.provenance.n_cells == len(data)
    assert result.provenance.notes == ("qc",)
    assert result.provenance.extras["displayed_points"] == {
        "nCounts": 5,
        "nFeatures": 5,
    }
    assert result.provenance.extras["seed"] == 13
    assert result.legends[0].kind == "categorical"
    assert all(ax.collections for ax in result.axes.values())
    result.close()


def test_elbow_result_records_detection_and_line_artists(deterministic_kneed):
    values = np.array([0.6, 0.25, 0.1])
    result = splt.elbow(values, show=False)

    table = result.tables["variance_explained"]
    np.testing.assert_array_equal(table["variance_explained"], values)
    assert table.loc[table["is_elbow"], "component"].tolist() == [1]
    assert result.provenance.extras["elbow"] == 1
    assert result.legends[0].extras["component"] == 1
    assert len(result.axes["elbow"].lines) == 2
    result.close()


def test_graph_qc_result_contains_graph_tables_and_histograms():
    result = splt.graph_qc(_graph(), show=False)

    assert result.provenance.n_cells == 3
    assert result.provenance.extras["n_edges"] == 4
    assert result.tables["node_degrees"]["degree"].tolist() == [1, 2, 1]
    assert result.tables["degree_frequencies"].to_dict("list") == {
        "degree": [1, 2],
        "frequency": [2, 1],
    }
    assert len(result.tables["edge_weights"]) == 4
    assert len(result.axes["node_degree"].patches) == 2
    assert len(result.axes["edge_weight"].patches) == 30
    result.close()


def test_highly_variable_features_result_contains_feature_table_and_scatters():
    args = _hvg_arguments()
    result = splt.highly_variable_features(*args, show=False)

    table = result.tables["features"]
    assert table["selected"].tolist() == [False, True, False, True]
    np.testing.assert_allclose(table["log2_mean_nonzero"], [0.0, 1.0, 2.0, 3.0])
    assert result.provenance.extras["n_features"] == 4
    assert result.provenance.extras["n_selected"] == 2
    assert result.legends[0].kind == "categorical"
    assert len(result.axes["highly_variable_features"].collections) == 2
    result.close()

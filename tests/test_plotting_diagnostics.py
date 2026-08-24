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


@pytest.mark.parametrize(
    ("data", "kwargs", "error", "message"),
    [
        pytest.param(
            pd.DataFrame({"value": [1.0]}),
            {},
            KeyError,
            "groups",
            id="missing-groups",
        ),
        pytest.param(
            pd.DataFrame(columns=["groups", "value"]),
            {},
            ValueError,
            "at least one row",
            id="empty",
        ),
        pytest.param(
            _qc_data(),
            {"max_points": -1},
            ValueError,
            "max_points",
            id="negative-max-points",
        ),
        pytest.param(
            pd.DataFrame({"groups": ["a"]}),
            {},
            ValueError,
            "metric column",
            id="missing-metric",
        ),
        pytest.param(
            pd.DataFrame(
                [["a", 1.0, 2.0]],
                columns=["groups", "metric", "metric"],
            ),
            {},
            ValueError,
            "unique",
            id="duplicate-metric",
        ),
        pytest.param(
            pd.DataFrame({"groups": ["a"], "metric": ["not-numeric"]}),
            {},
            TypeError,
            "must be numeric",
            id="non-numeric",
        ),
    ],
)
def test_qc_rejects_malformed_dataframes(data, kwargs, error, message):
    with pytest.raises(error, match=message):
        splt.qc(data, show=False, **kwargs)


def test_qc_vertical_layout_single_group_titles_and_figure_size():
    import matplotlib.pyplot as plt

    data = pd.DataFrame(
        {
            "groups": ["only"] * 4,
            "metricA": [1.0, 2.0, 3.0, 4.0],
            "metricB": [10.0, 12.0, 14.0, 16.0],
        }
    )
    result = splt.qc(
        data,
        color="#123456",
        max_points=0,
        show_on_single_row=False,
        sup_title="QC overview",
        show=False,
    )
    result.figure.canvas.draw()

    assert result.figure.get_size_inches() == pytest.approx((3.0, 6.0))
    assert result.figure._suptitle.get_text() == "QC overview"
    assert result.scales[0].order == ("only",)
    assert result.scales[0].palette == {"only": "#123456"}
    assert result.axes["metricA"].get_position().y0 > (
        result.axes["metricB"].get_position().y0
    )
    assert result.axes["metricA"].get_title() == "Median: 2.5"
    assert result.axes["metricB"].get_title() == "Median: 13.0"
    assert all(len(axis.get_xticks()) == 0 for axis in result.axes.values())
    figure_number = result.figure.number
    result.close()
    assert not plt.fignum_exists(figure_number)


def test_qc_orders_natural_and_missing_groups_without_subsampling():
    groups = ["group10", "group2", None] * 3
    data = pd.DataFrame(
        {
            "groups": groups,
            "metric": np.arange(len(groups), dtype=float),
        }
    )
    result = splt.qc(data, max_points=100, show=False)

    assert result.scales[0].order == ("group2", "group10", "NA")
    assert result.provenance.extras["displayed_points"] == {"metric": len(data)}
    assert result.tables["summary"]["group"].tolist() == [
        "group10",
        "group2",
        "NA",
    ]
    result.close()


def test_elbow_without_detection_uses_data_driven_width(monkeypatch):
    class NoKneeLocator:
        def __init__(self, x, y, **kwargs):
            self.elbow = None

    monkeypatch.setattr(diagnostics, "require_kneed", lambda: NoKneeLocator)
    values = np.linspace(1.0, 0.1, 8)
    result = splt.elbow(values, show=False)

    assert result.figure.get_size_inches() == pytest.approx((2.0, 2.0))
    assert result.provenance.extras["elbow"] is None
    assert result.legends == ()
    assert len(result.axes["elbow"].lines) == 1
    assert not result.tables["variance_explained"]["is_elbow"].any()
    result.close()


@pytest.mark.parametrize(
    "values",
    [
        pytest.param([], id="empty"),
        pytest.param([[1.0, 0.5]], id="two-dimensional"),
        pytest.param([1.0, np.inf], id="non-finite"),
    ],
)
def test_elbow_rejects_malformed_variance_arrays(values):
    with pytest.raises(ValueError, match="variance_explained"):
        splt.elbow(values, show=False)


def test_graph_qc_clips_single_degree_outlier_and_records_limit():
    node_count = 1001
    leaves = np.arange(1, node_count, dtype=np.int64)
    rows = np.concatenate((np.zeros(node_count - 1, dtype=np.int64), leaves))
    columns = np.concatenate((leaves, np.zeros(node_count - 1, dtype=np.int64)))
    graph = sparse.csr_matrix(
        (np.ones(len(rows), dtype=float), (rows, columns)),
        shape=(node_count, node_count),
    )
    result = splt.graph_qc(graph, show=False)

    assert result.provenance.extras["degree_clip_limit"] == pytest.approx(6.0)
    assert result.axes["node_degree"].get_xlim() == pytest.approx((0.0, 6.0))
    assert [text.get_text() for text in result.axes["node_degree"].texts] == [
        "plot is clipped (max degree: 1000)"
    ]
    assert result.tables["node_degrees"]["degree"].max() == 1000
    result.close()


def test_graph_qc_rejects_malformed_sparse_adapters():
    class MissingData:
        shape = (2, 2)

    class BrokenDegreeCalculation:
        shape = (2, 2)
        data = np.array([1.0])

        def __ne__(self, other):
            return object()

    with pytest.raises(TypeError, match="two-dimensional sparse matrix"):
        splt.graph_qc(object(), show=False)
    with pytest.raises(TypeError, match="expose sparse edge weights"):
        splt.graph_qc(MissingData(), show=False)
    with pytest.raises(TypeError, match="non-zero degree calculation"):
        splt.graph_qc(BrokenDegreeCalculation(), show=False)


@pytest.mark.parametrize(
    ("arguments", "kwargs", "message"),
    [
        pytest.param(
            (
                np.ones((2, 2)),
                np.ones(4),
                np.ones(4),
                np.zeros(4, dtype=bool),
            ),
            {},
            "one-dimensional",
            id="dimensions",
        ),
        pytest.param(
            (
                np.ones(3),
                np.ones(4),
                np.ones(4),
                np.zeros(4, dtype=bool),
            ),
            {},
            "matching lengths",
            id="lengths",
        ),
        pytest.param(
            _hvg_arguments(),
            {"point_sizes": (3,)},
            "point_sizes",
            id="point-sizes",
        ),
        pytest.param(
            _hvg_arguments(),
            {"colormaps": ("viridis",)},
            "colormaps",
            id="colormaps",
        ),
    ],
)
def test_highly_variable_features_rejects_malformed_adapters(
    arguments,
    kwargs,
    message,
):
    with pytest.raises(ValueError, match=message):
        splt.highly_variable_features(*arguments, show=False, **kwargs)


def test_highly_variable_features_accepts_empty_feature_arrays():
    empty_float = np.array([], dtype=float)
    empty_bool = np.array([], dtype=bool)
    result = splt.highly_variable_features(
        empty_float,
        empty_float,
        empty_float,
        empty_bool,
        show=False,
    )

    assert result.tables["features"].empty
    assert result.provenance.extras["n_features"] == 0
    assert result.provenance.extras["max_expressing_cells"] == 0
    result.close()


def test_diagnostic_default_show_closes_all_owned_results(
    deterministic_kneed,
):
    import matplotlib.pyplot as plt

    results = [
        splt.qc(_qc_data(), max_points=0),
        splt.elbow([1.0, 0.5, 0.25]),
        splt.graph_qc(_graph()),
        splt.highly_variable_features(*_hvg_arguments()),
    ]

    assert all(result.owns_figure for result in results)
    assert all(not plt.fignum_exists(result.figure.number) for result in results)

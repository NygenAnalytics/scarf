import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import scarf.plotting as splt
from scarf.plotting._figure import PlotResult


# A steep drop followed by a slow tail has its only corner at component 2.
_CORNERED_VARIANCE = np.array([9.0, 5.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5])


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


def test_diagnostics_show_default_and_suppression(monkeypatch):
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
    from matplotlib.colors import to_rgba

    data = _qc_data()
    result = splt.qc(data, max_points=5, seed=13, show=False)

    assert isinstance(result, PlotResult)
    assert result.owns_figure is True
    assert list(result.axes) == ["nCounts", "nFeatures"]
    pd.testing.assert_frame_equal(result.tables["data"], data)
    summary = result.tables["summary"]
    assert list(zip(summary["group"], summary["metric"], strict=True)) == [
        ("a", "nCounts"),
        ("b", "nCounts"),
        ("a", "nFeatures"),
        ("b", "nFeatures"),
    ]
    assert summary["count"].tolist() == [4, 4, 4, 4]
    # Medians of [10, 12, 11, 13], [20, 22, 21, 23], [4, 5, 5, 6], [8, 9, 9, 10].
    np.testing.assert_allclose(summary["median"], [11.5, 21.5, 5.0, 9.0])
    assert result.provenance.n_cells == len(data)
    assert result.provenance.notes == ("qc",)
    assert result.provenance.extras["displayed_points"] == {
        "nCounts": 5,
        "nFeatures": 5,
    }
    assert result.provenance.extras["seed"] == 13
    assert result.legends[0].kind == "categorical"
    assert result.legends[0].extras == {"categories": ["a", "b"]}
    # Two groups take the first two tab20 colors.
    assert result.scales[0].palette == {"a": "#1f77b4", "b": "#aec7e8"}

    sampled_rows: dict[str, list[int]] = {}
    for metric, ax in result.axes.items():
        assert ax.get_ylabel() == metric
        assert [label.get_text() for label in ax.get_xticklabels()] == ["a", "b"]
        violins = ax.collections[:2]
        for violin, color in zip(violins, ("#1f77b4", "#aec7e8"), strict=True):
            np.testing.assert_allclose(
                violin.get_facecolor()[0],
                to_rgba(color, alpha=0.6),
            )
        strips = ax.collections[2:]
        assert len(strips) == 2
        rows: list[int] = []
        for position, (strip, group) in enumerate(zip(strips, "ab", strict=True)):
            offsets = np.asarray(strip.get_offsets())
            np.testing.assert_allclose(strip.get_facecolor()[0], (0, 0, 0, 0.4))
            assert np.all(np.abs(offsets[:, 0] - position) <= 0.4)
            group_values = data.loc[data["groups"] == group, metric]
            for value in offsets[:, 1]:
                matches = group_values.index[group_values == value]
                assert len(matches) >= 1
                rows.append(int(matches[0]))
        assert len(rows) == 5
        sampled_rows[metric] = rows
    # One seed samples the same cells for every metric.
    np.testing.assert_array_equal(
        data.loc[sorted(sampled_rows["nCounts"]), "nFeatures"].sort_values(),
        np.sort(
            np.concatenate(
                [
                    np.asarray(strip.get_offsets())[:, 1]
                    for strip in result.axes["nFeatures"].collections[2:]
                ]
            )
        ),
    )
    result.close()


def test_elbow_result_records_detection_and_line_artists():
    from matplotlib.colors import to_rgba

    values = _CORNERED_VARIANCE
    result = splt.elbow(values, show=False)

    table = result.tables["variance_explained"]
    np.testing.assert_array_equal(table["component"], np.arange(8))
    np.testing.assert_array_equal(table["variance_explained"], values)
    assert table.loc[table["is_elbow"], "component"].tolist() == [2]
    assert result.provenance.extras == {
        "elbow": 2,
        "n_components": 8,
        "sensitivity": 1.0,
    }
    assert result.legends[0].label == "Elbow"
    assert result.legends[0].extras == {"component": 2}
    # The default width spends a quarter inch on each component.
    assert result.figure.get_size_inches() == pytest.approx((2.0, 2.0))

    ax = result.axes["elbow"]
    curve, marker = ax.lines
    np.testing.assert_array_equal(curve.get_xdata(), np.arange(8))
    np.testing.assert_array_equal(curve.get_ydata(), values)
    np.testing.assert_array_equal(marker.get_xdata(), [2, 2])
    assert to_rgba(marker.get_color()) == to_rgba("red")
    np.testing.assert_array_equal(ax.get_xticks(), np.arange(8))
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["Elbow"]
    assert ax.get_xlabel() == "Principal components"
    assert ax.get_ylabel() == "% Variance explained"
    result.close()


def test_graph_qc_result_contains_graph_tables_and_histograms():
    result = splt.graph_qc(_graph(), show=False)

    assert result.provenance.n_cells == 3
    # The 99.5th percentile of degrees [1, 2, 1] is 1.99; the limit adds five.
    assert result.provenance.extras == {
        "graph_shape": [3, 3],
        "n_edges": 4,
        "degree_clip_limit": pytest.approx(6.99),
    }
    assert result.tables["node_degrees"]["node"].tolist() == [0, 1, 2]
    assert result.tables["node_degrees"]["degree"].tolist() == [1, 2, 1]
    assert result.tables["degree_frequencies"].to_dict("list") == {
        "degree": [1, 2],
        "frequency": [2, 1],
    }
    assert sorted(result.tables["edge_weights"]["edge_weight"]) == [
        0.5,
        0.5,
        1.0,
        1.0,
    ]

    degree_ax = result.axes["node_degree"]
    assert [
        (patch.get_x() + patch.get_width() / 2, patch.get_height())
        for patch in degree_ax.patches
    ] == [(1.0, 2), (2.0, 1)]
    assert degree_ax.get_xlim() == pytest.approx((0.0, 6.99))
    assert len(degree_ax.texts) == 0
    weight_patches = result.axes["edge_weight"].patches
    heights = [patch.get_height() for patch in weight_patches]
    assert len(heights) == 30
    # Thirty bins span the observed weights [0.5, 1.0]: both ends hold two edges.
    assert heights[0] == heights[-1] == 2
    assert sum(heights) == 4
    assert weight_patches[0].get_x() == pytest.approx(0.5)
    assert weight_patches[-1].get_x() + weight_patches[-1].get_width() == (
        pytest.approx(1.0)
    )
    result.close()


def test_highly_variable_features_result_contains_feature_table_and_scatters():
    args = _hvg_arguments()
    result = splt.highly_variable_features(*args, show=False)

    table = result.tables["features"]
    assert table["selected"].tolist() == [False, True, False, True]
    np.testing.assert_allclose(table["log2_mean_nonzero"], [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        table["log2_corrected_variance"],
        [1.0, np.log2(3.0), 3.0, np.log2(12.0)],
    )
    assert result.provenance.extras["n_features"] == 4
    assert result.provenance.extras["n_selected"] == 2
    assert result.provenance.extras["max_expressing_cells"] == 40.0
    assert result.legends[0].kind == "categorical"

    ax = result.axes["highly_variable_features"]
    unselected, selected = ax.collections
    # Unselected features 0 and 2 then selected features 1 and 3, colored by
    # their expressing-cell counts.
    np.testing.assert_allclose(unselected.get_offsets(), [[0.0, 1.0], [2.0, 3.0]])
    np.testing.assert_allclose(
        selected.get_offsets(),
        [[1.0, np.log2(3.0)], [3.0, np.log2(12.0)]],
    )
    np.testing.assert_array_equal(unselected.get_array(), [10.0, 30.0])
    np.testing.assert_array_equal(selected.get_array(), [20.0, 40.0])
    assert unselected.get_sizes().tolist() == [3]
    assert selected.get_sizes().tolist() == [30]
    assert unselected.get_cmap().name == "winter"
    assert selected.get_cmap().name == "magma_r"
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        "Not selected",
        "Selected",
    ]
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


def test_elbow_without_detection_uses_data_driven_width():
    # A straight line has no point of maximum curvature, so no elbow.
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


def test_diagnostic_default_show_closes_all_owned_results():
    import matplotlib.pyplot as plt

    results = [
        splt.qc(_qc_data(), max_points=0),
        splt.elbow([1.0, 0.5, 0.25]),
        splt.graph_qc(_graph()),
        splt.highly_variable_features(*_hvg_arguments()),
    ]

    assert all(result.owns_figure for result in results)
    assert all(not plt.fignum_exists(result.figure.number) for result in results)

from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import scarf.mapping.confidence as mapping_confidence
import scarf.plotting as splt
import scarf.plotting.mapping as plotting_mapping
from tests.test_mapping_label_transfer import (
    _copied_query,
    _plain_reference,
    _write_reference_column,
    _write_projection,
    _write_reference_layout,
)


@pytest.fixture
def plotting_mapping_context(analyzed_datastore_ephemeral, tmp_path: Path):
    reference_store = analyzed_datastore_ephemeral
    reference = _plain_reference(reference_store)
    query = _copied_query(reference_store, tmp_path / "plotting_query.zarr")
    reference_layout, layout_ref = _write_reference_layout(
        reference,
        name="mapping_layout",
    )
    reference_labels = np.full(
        reference.selected_cell_count,
        "other",
        dtype=object,
    )
    reference_labels[:4] = ["A", "A", "B", "B"]
    _write_reference_column(reference, "mapping_label", reference_labels)

    query.cells.insert(
        "mapping_layout1",
        np.full(query.cells.N, -1000.0),
        overwrite=True,
    )
    query.cells.insert(
        "mapping_layout2",
        np.full(query.cells.N, 1000.0),
        overwrite=True,
    )
    query.cells.insert(
        "mapping_label",
        np.full(query.cells.N, "query_only", dtype=object),
        overwrite=True,
    )
    indices = np.asarray(
        [
            [0, 1],
            [0, 1],
            [2, 3],
            [2, 3],
            [0, 2],
            [2, 3],
        ],
        dtype=np.uint64,
    )
    distances = np.asarray(
        [
            [1.0, 9.0],
            [2.0, 8.0],
            [1.0, 9.0],
            [2.0, 8.0],
            [1.0, 1.0],
            [1.0, 9.0],
        ],
        dtype=np.float64,
    )
    uninformative = np.asarray(
        [False, False, False, False, True, False],
        dtype=bool,
    )
    result = _write_projection(
        query,
        reference,
        indices=indices,
        distances=distances,
        uninformative=uninformative,
    )
    return {
        "reference": reference,
        "reference_layout": reference_layout,
        "layout_ref": layout_ref,
        "reference_labels": reference_labels,
        "query": query,
        "result": result,
        "query_groups": np.asarray(["q1", "q1", "q2", "q2", "q1", "q2"]),
        "known_labels": np.asarray(["A", "A", "B", "B", "A", "B"]),
    }


def _sorted_rows(values: np.ndarray) -> np.ndarray:
    return values[np.lexsort((values[:, 1], values[:, 0]))]


def _controlled_mapping_store(
    monkeypatch: pytest.MonkeyPatch,
    *,
    evidence: pd.DataFrame | None = None,
    score_rows: list[tuple[object, np.ndarray]] | None = None,
    n_reference: int | None = None,
):
    if n_reference is None:
        if score_rows:
            n_reference = len(score_rows[0][1])
        elif evidence is not None:
            n_reference = len(evidence)
        else:
            n_reference = 0
    reference = SimpleNamespace(selected_cell_count=n_reference)
    mapping = SimpleNamespace(
        reference=reference,
        ref=SimpleNamespace(assay="RNA"),
    )
    monkeypatch.setattr(
        plotting_mapping,
        "_mapping_result",
        lambda *_args, **_kwargs: mapping,
    )
    methods = {}
    if evidence is not None:
        methods["get_target_label_evidence"] = lambda *_args, **_kwargs: evidence.copy()
    if score_rows is not None:
        methods["get_mapping_score"] = lambda *_args, **_kwargs: list(score_rows)
    return SimpleNamespace(**methods)


def test_mapping_plot_families_use_query_result_and_reference_semantics(
    plotting_mapping_context,
):
    context = plotting_mapping_context
    query = context["query"]
    result = context["result"]
    reference = context["reference"]
    query_groups = context["query_groups"]
    known_labels = context["known_labels"]

    score = splt.mapping_score(
        query,
        result,
        reference=reference,
        target_groups=query_groups,
        layout=context["layout_ref"],
        show=False,
    )
    evidence = splt.mapping_evidence(
        query,
        result,
        reference=reference,
        reference_class_group="mapping_label",
        target_groups=query_groups,
        metrics=("voteFraction", "topTwoMargin"),
        show=False,
    )
    score_boxes = splt.mapping_score(
        query,
        result,
        reference=reference,
        target_groups=query_groups,
        kind="box",
        reference_class_group="mapping_label",
        show=False,
    )
    sized = splt.mapping_score(
        query,
        result,
        reference=reference,
        target_groups=query_groups,
        layout=context["layout_ref"],
        size_by_score=True,
        show=False,
    )
    confusion = splt.mapping_confusion(
        query,
        result,
        reference=reference,
        reference_class_group="mapping_label",
        known_labels=known_labels,
        show=False,
    )
    calibration = splt.mapping_calibration(
        query,
        result,
        reference=reference,
        reference_class_group="mapping_label",
        known_labels=known_labels,
        n_thresholds=3,
        chosen_threshold=0.75,
        show=False,
    )

    assert set(score.axes) == {"q1", "q2"}
    n_reference = reference.selected_cell_count
    assert len(score.tables["scores"]) == 2 * n_reference
    score_coordinates = np.asarray(
        next(iter(score.axes.values())).collections[0].get_offsets(),
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        _sorted_rows(score_coordinates),
        _sorted_rows(context["reference_layout"]),
    )
    assert "mapping_name" not in score.provenance.extras

    assert set(evidence.axes) == {"voteFraction", "topTwoMargin"}
    assert len(evidence.tables["evidence"]) == len(query_groups)
    assert set(score_boxes.axes) == {"q1", "q2"}
    assert "referenceClass" in score_boxes.tables["scores"]
    assert set(sized.axes) == {"q1", "q2"}
    assert sized.provenance.extras["size_by_score"] is True
    sized_collections = sized.axes["q1"].collections
    background_sizes = sized_collections[0].get_sizes()
    mapped_sizes = sized_collections[1].get_sizes()
    assert mapped_sizes.min() > background_sizes.max()
    assert np.ptp(mapped_sizes) > 0
    assert confusion.tables["counts"].set_index("known").loc["A", "A"] >= 2
    assert {"precision", "recall", "support"} <= set(confusion.tables["perClass"])
    assert {"coverage", "accuracy", "accuracyLower", "accuracyUpper"} <= set(
        calibration.tables["calibration"]
    )
    assert calibration.tables["calibration"]["coverage"].between(0, 1).all()

    for plot in (
        score,
        evidence,
        score_boxes,
        sized,
        confusion,
        calibration,
    ):
        plot.close()


def test_mapping_plots_resolve_explicit_artifact_result(
    plotting_mapping_context,
):
    context = plotting_mapping_context
    plot = splt.mapping_score(
        context["query"],
        context["result"],
        reference=context["reference"],
        kind="histogram",
        show=False,
    )

    assert "mapping_name" not in plot.provenance.extras
    plot.close()


def test_mapping_evidence_box_kind_draws_one_box_per_query_group(
    plotting_mapping_context,
):
    context = plotting_mapping_context
    evidence = splt.mapping_evidence(
        context["query"],
        context["result"],
        reference=context["reference"],
        reference_class_group="mapping_label",
        target_groups=context["query_groups"],
        metrics=("voteFraction",),
        kind="box",
        show=False,
    )

    axis = evidence.axes["voteFraction"]
    assert [text.get_text() for text in axis.get_xticklabels()] == ["q1", "q2"]
    assert all(text.get_ha() == "right" for text in axis.get_xticklabels())
    evidence.close()


def test_mapping_score_box_kind_groups_by_reference_class(
    plotting_mapping_context,
):
    context = plotting_mapping_context
    plot = splt.mapping_score(
        context["query"],
        context["result"],
        reference=context["reference"],
        target_groups=context["query_groups"],
        kind="box",
        reference_class_group="mapping_label",
        show=False,
    )

    axis = plot.axes["q1"]
    assert [text.get_text() for text in axis.get_xticklabels()] == ["A", "B", "other"]
    assert all(text.get_ha() == "right" for text in axis.get_xticklabels())
    plot.close()


def test_mapping_score_uses_generic_default_title(
    plotting_mapping_context,
):
    context = plotting_mapping_context
    plot = splt.mapping_score(
        context["query"],
        context["result"],
        reference=context["reference"],
        layout=context["layout_ref"],
        show=False,
    )

    assert plot.tables["scores"]["group"].unique().tolist() == [0]
    assert next(iter(plot.axes.values())).get_title() == "all query cells"
    plot.close()


def test_mapping_calibration_allows_genuine_zero_accuracy(
    plotting_mapping_context,
):
    context = plotting_mapping_context
    plot = splt.mapping_calibration(
        context["query"],
        context["result"],
        reference=context["reference"],
        reference_class_group="mapping_label",
        known_labels=np.asarray(["B", "B", "A", "A", "B", "A"]),
        n_thresholds=3,
        show=False,
    )

    assert (plot.tables["calibration"]["accuracy"] == 0).all()
    plot.close()


def test_mapping_calibration_rejects_pairwise_label_type_mismatch(
    plotting_mapping_context,
):
    context = plotting_mapping_context
    reference = context["reference"]
    numeric_labels = np.zeros(reference.selected_cell_count, dtype=np.int64)
    numeric_labels[:4] = [1, 1, 2, 2]
    _write_reference_column(reference, "mapping_numeric_label", numeric_labels)

    with pytest.raises(ValueError, match="after text conversion"):
        splt.mapping_calibration(
            context["query"],
            context["result"],
            reference=reference,
            reference_class_group="mapping_numeric_label",
            known_labels=np.asarray(["1", "1", "2", "2", "1", "2"]),
            n_thresholds=3,
            show=False,
        )


def test_mapping_calibration_warns_when_threshold_retains_nothing(
    plotting_mapping_context,
):
    context = plotting_mapping_context
    with pytest.warns(RuntimeWarning, match="retained no mapped cells"):
        plot = splt.mapping_calibration(
            context["query"],
            context["result"],
            reference=context["reference"],
            reference_class_group="mapping_label",
            known_labels=context["known_labels"],
            chosen_threshold=2.0,
            n_thresholds=3,
            show=False,
        )

    assert not any(
        "voteFraction =" in text.get_text()
        for text in plot.axes["mapping_calibration"].texts
    )
    plot.close()


def test_mapping_diagnostic_plots_accept_caller_owned_targets(
    plotting_mapping_context,
):
    import matplotlib.pyplot as pyplot

    context = plotting_mapping_context
    figure, axes = pyplot.subplots(1, 2)
    plot = splt.mapping_evidence(
        context["query"],
        context["result"],
        reference=context["reference"],
        reference_class_group="mapping_label",
        metrics=("voteFraction", "topTwoMargin"),
        target={
            "voteFraction": axes[0],
            "topTwoMargin": axes[1],
        },
        show=False,
    )

    assert plot.figure is figure
    assert not plot.owns_figure
    plot.close()
    assert pyplot.fignum_exists(figure.number)
    pyplot.close(figure)


def test_mapping_plots_do_not_project_or_weight_coordinates(
    plotting_mapping_context,
    monkeypatch: pytest.MonkeyPatch,
):
    context = plotting_mapping_context
    query = context["query"]
    result = context["result"]
    reference = context["reference"]
    score_rows = list(
        query.get_mapping_score(
            result,
            target_groups=context["query_groups"],
            reference=reference,
        )
    )
    evidence = query.get_target_label_evidence(
        result,
        reference_class_group="mapping_label",
        reference=reference,
    )

    def cached_scores(*args, **kwargs):
        yield from score_rows

    def cached_evidence(*args, **kwargs):
        return evidence.copy()

    def unexpected(*args, **kwargs):
        raise AssertionError("plotting attempted mapping computation")

    monkeypatch.setattr(type(query), "get_mapping_score", cached_scores)
    monkeypatch.setattr(type(query), "get_target_label_evidence", cached_evidence)
    monkeypatch.setattr(mapping_confidence, "distance_weights", unexpected)

    plots = (
        splt.mapping_score(
            query,
            result,
            reference=reference,
            target_groups=context["query_groups"],
            layout=context["layout_ref"],
            show=False,
        ),
        splt.mapping_score(
            query,
            result,
            reference=reference,
            target_groups=context["query_groups"],
            kind="box",
            reference_class_group="mapping_label",
            show=False,
        ),
        splt.mapping_evidence(
            query,
            result,
            reference=reference,
            reference_class_group="mapping_label",
            target_groups=context["query_groups"],
            metrics=("voteFraction",),
            kind="box",
            show=False,
        ),
        splt.mapping_confusion(
            query,
            result,
            reference=reference,
            reference_class_group="mapping_label",
            known_labels=context["known_labels"],
            show=False,
        ),
        splt.mapping_calibration(
            query,
            result,
            reference=reference,
            reference_class_group="mapping_label",
            known_labels=context["known_labels"],
            n_thresholds=3,
            show=False,
        ),
    )

    for plot in plots:
        plot.close()


def test_mapping_calibration_respects_direction_and_draws_uncertainty(
    monkeypatch: pytest.MonkeyPatch,
):
    evidence = pd.DataFrame(
        {
            "label": ["A", "B", "B", "A"],
            "isUnknown": [False, False, False, False],
            "voteFraction": [0.9, 0.7, 0.4, 0.1],
            "meanNeighborDistance": [0.1, 0.3, 0.6, 0.9],
            "customConfidence": [0.9, 0.7, 0.4, 0.1],
        }
    )
    store = _controlled_mapping_store(monkeypatch, evidence=evidence)
    known = np.asarray(["A", "B", "A", "B"])

    higher = plotting_mapping.mapping_calibration(
        store,
        object(),
        reference=object(),
        reference_class_group="label",
        known_labels=known,
        metric="voteFraction",
        direction="auto",
        thresholds=[0.0, 0.5, 0.8],
        chosen_threshold=0.5,
        show=False,
    )
    lower = plotting_mapping.mapping_calibration(
        store,
        object(),
        reference=object(),
        reference_class_group="label",
        known_labels=known,
        metric="meanNeighborDistance",
        direction="auto",
        thresholds=[0.2, 0.5, 1.0],
        show=False,
    )
    explicit = plotting_mapping.mapping_calibration(
        store,
        object(),
        reference=object(),
        reference_class_group="label",
        known_labels=known,
        metric="customConfidence",
        direction="higher",
        thresholds=[0.0, 0.5],
        show=False,
    )

    higher_rows = higher.tables["calibration"].set_index("threshold")
    lower_rows = lower.tables["calibration"].set_index("threshold")
    np.testing.assert_allclose(
        higher_rows.loc[[0.0, 0.5, 0.8], "coverage"],
        [1.0, 0.5, 0.25],
    )
    np.testing.assert_array_equal(
        higher_rows.loc[[0.0, 0.5, 0.8], "nAccepted"],
        [4, 2, 1],
    )
    np.testing.assert_allclose(
        lower_rows.loc[[0.2, 0.5, 1.0], "coverage"],
        [0.25, 0.5, 1.0],
    )
    np.testing.assert_array_equal(
        lower_rows.loc[[0.2, 0.5, 1.0], "nAccepted"],
        [1, 2, 4],
    )
    assert higher.provenance.extras["direction"] == "higher"
    assert lower.provenance.extras["direction"] == "lower"
    assert explicit.provenance.extras["direction"] == "higher"
    assert any(
        text.get_text() == "voteFraction = 0.5"
        for text in higher.axes["mapping_calibration"].texts
    )

    for plot in (higher, lower, explicit):
        calibration = plot.tables["calibration"]
        assert (calibration["accuracyLower"] <= calibration["accuracy"]).all()
        assert (calibration["accuracy"] <= calibration["accuracyUpper"]).all()
        assert calibration["accuracyLower"].between(0, 1).all()
        assert calibration["accuracyUpper"].between(0, 1).all()
        axis = plot.axes["mapping_calibration"]
        assert axis.collections[0].get_paths()
        band_vertices = np.concatenate(
            [path.vertices for path in axis.collections[0].get_paths()]
        )
        assert np.isfinite(band_vertices).all()
        np.testing.assert_allclose(
            axis.lines[0].get_xdata(),
            calibration["coverage"],
        )
        np.testing.assert_allclose(
            axis.lines[0].get_ydata(),
            calibration["accuracy"],
        )
        plot.close()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_thresholds": 1}, "at least 2"),
        ({"direction": "sideways"}, "direction must be"),
        ({"metric": "customConfidence"}, "Cannot infer threshold direction"),
        ({"thresholds": []}, "finite numeric values"),
        ({"thresholds": [0.1, np.nan]}, "finite numeric values"),
        (
            {"thresholds": [0.1], "chosen_threshold": np.inf},
            "chosen_threshold must be finite",
        ),
    ],
)
def test_mapping_calibration_rejects_malformed_threshold_controls(
    monkeypatch: pytest.MonkeyPatch,
    kwargs,
    message,
):
    evidence = pd.DataFrame(
        {
            "label": ["A", "B"],
            "isUnknown": [False, False],
            "voteFraction": [0.8, 0.2],
            "customConfidence": [0.8, 0.2],
        }
    )
    store = _controlled_mapping_store(monkeypatch, evidence=evidence)

    with pytest.raises(ValueError, match=message):
        plotting_mapping.mapping_calibration(
            store,
            object(),
            reference=object(),
            reference_class_group="label",
            known_labels=np.asarray(["A", "B"]),
            show=False,
            **kwargs,
        )


def test_mapping_calibration_rejects_nonfinite_or_unretained_evidence(
    monkeypatch: pytest.MonkeyPatch,
):
    evidence = pd.DataFrame(
        {
            "label": ["A", "B"],
            "isUnknown": [False, False],
            "voteFraction": [np.nan, np.nan],
        }
    )
    store = _controlled_mapping_store(monkeypatch, evidence=evidence)
    with pytest.raises(ValueError, match="No finite metric values"):
        plotting_mapping.mapping_calibration(
            store,
            object(),
            reference=object(),
            reference_class_group="label",
            known_labels=np.asarray(["A", "B"]),
            show=False,
        )

    finite_evidence = evidence.assign(voteFraction=[0.8, 0.2])
    store = _controlled_mapping_store(monkeypatch, evidence=finite_evidence)
    with pytest.raises(ValueError, match="No threshold retained"):
        plotting_mapping.mapping_calibration(
            store,
            object(),
            reference=object(),
            reference_class_group="label",
            known_labels=np.asarray(["A", "B"]),
            thresholds=[2.0],
            show=False,
        )


def test_mapping_plots_reject_empty_and_misaligned_data(
    monkeypatch: pytest.MonkeyPatch,
):
    empty_store = _controlled_mapping_store(
        monkeypatch,
        score_rows=[],
        n_reference=2,
    )
    with pytest.raises(ValueError, match="produced no score groups"):
        plotting_mapping.mapping_score(
            empty_store,
            object(),
            reference=object(),
            kind="histogram",
            show=False,
        )

    mismatched_store = _controlled_mapping_store(
        monkeypatch,
        score_rows=[
            ("first", np.asarray([0.1, 0.2])),
            ("second", np.asarray([0.3])),
        ],
        n_reference=2,
    )
    with pytest.raises(ValueError, match="incompatible lengths"):
        plotting_mapping.mapping_score(
            mismatched_store,
            object(),
            reference=object(),
            kind="histogram",
            show=False,
        )

    evidence = pd.DataFrame(
        {
            "label": ["A", "B"],
            "isUnknown": [False, False],
            "voteFraction": [0.8, 0.2],
        }
    )
    evidence_store = _controlled_mapping_store(monkeypatch, evidence=evidence)
    with pytest.raises(ValueError, match="one value per mapped cell"):
        plotting_mapping.mapping_evidence(
            evidence_store,
            object(),
            reference=object(),
            reference_class_group="label",
            target_groups=["only-one"],
            metrics=("voteFraction",),
            show=False,
        )
    with pytest.raises(ValueError, match="cannot contain missing values"):
        plotting_mapping.mapping_evidence(
            evidence_store,
            object(),
            reference=object(),
            reference_class_group="label",
            target_groups=["first", None],
            metrics=("voteFraction",),
            show=False,
        )
    with pytest.raises(ValueError, match="metrics must be non-empty"):
        plotting_mapping.mapping_evidence(
            evidence_store,
            object(),
            reference=object(),
            reference_class_group="label",
            metrics=(),
            show=False,
        )
    with pytest.raises(KeyError, match="Unknown evidence metrics"):
        plotting_mapping.mapping_evidence(
            evidence_store,
            object(),
            reference=object(),
            reference_class_group="label",
            metrics=("missingMetric",),
            show=False,
        )


def test_mapping_layout_rejects_wrong_shape_and_infinite_coordinates():
    wrong_shape = SimpleNamespace(
        selected_cell_count=2,
        fetch_layout=lambda _key: np.zeros((2, 3)),
    )
    with pytest.raises(ValueError, match="two columns"):
        plotting_mapping._reference_layout(wrong_shape, "layout")

    infinite = SimpleNamespace(
        selected_cell_count=2,
        fetch_layout=lambda _key: np.asarray([[0.0, 1.0], [np.inf, 2.0]]),
    )
    with pytest.raises(ValueError, match="infinite coordinates"):
        plotting_mapping._reference_layout(infinite, "layout")


def test_mapping_categorical_legends_serialize_and_owned_figures_close(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    import json

    import matplotlib.pyplot as plt

    evidence = pd.DataFrame(
        {
            "label": ["A", "B", "A"],
            "isUnknown": [False, False, False],
            "voteFraction": [0.9, 0.4, 0.7],
        }
    )
    score_rows = [
        ("beta", np.asarray([0.1, 0.2, 0.3])),
        ("alpha", np.asarray([0.4, 0.5, 0.6])),
    ]
    store = _controlled_mapping_store(
        monkeypatch,
        evidence=evidence,
        score_rows=score_rows,
    )
    scale = splt.CategoricalScale(
        order=("alpha", "beta"),
        palette={"alpha": "#336699", "beta": "#cc5500"},
    )
    scores = plotting_mapping.mapping_score(
        store,
        object(),
        reference=object(),
        kind="histogram",
        categorical_scale=scale,
        bins=3,
        show=False,
    )
    evidence_plot = plotting_mapping.mapping_evidence(
        store,
        object(),
        reference=object(),
        reference_class_group="label",
        target_groups=["beta", "alpha", "beta"],
        metrics=("voteFraction",),
        categorical_scale=scale,
        bins=3,
        show=False,
    )

    score_legend = scores.axes["mapping_score"].get_legend()
    evidence_legend = evidence_plot.axes["voteFraction"].get_legend()
    assert score_legend is not None
    assert evidence_legend is not None
    assert [text.get_text() for text in score_legend.get_texts()] == [
        "alpha",
        "beta",
    ]
    assert [text.get_text() for text in evidence_legend.get_texts()] == [
        "alpha",
        "beta",
    ]
    assert scores.scales[0].order == ("alpha", "beta")
    assert evidence_plot.scales[0].palette == scale.palette

    payload = json.loads(
        scores.save_provenance(tmp_path / "mapping_scores.json").read_text()
    )
    assert payload["scales"][0]["type"] == "CategoricalScale"
    assert payload["scales"][0]["values"]["order"] == ["alpha", "beta"]
    assert payload["tables"]["scores"] == {
        "columns": ["group", "referenceIndex", "score"],
        "rows": 6,
    }

    figure_numbers = [scores.figure.number, evidence_plot.figure.number]
    scores.close()
    evidence_plot.close()
    assert all(not plt.fignum_exists(number) for number in figure_numbers)


def test_mapping_score_surfaces_missing_matplotlib_without_opening_a_figure(
    monkeypatch: pytest.MonkeyPatch,
):
    import matplotlib.pyplot as plt

    store = _controlled_mapping_store(
        monkeypatch,
        score_rows=[("all", np.asarray([0.1, 0.2]))],
    )

    def missing_matplotlib():
        raise ImportError("Scarf plotting requires matplotlib")

    monkeypatch.setattr(
        plotting_mapping,
        "require_matplotlib",
        missing_matplotlib,
    )
    open_figures = plt.get_fignums()
    with pytest.raises(ImportError, match="requires matplotlib"):
        plotting_mapping.mapping_score(
            store,
            object(),
            reference=object(),
            kind="histogram",
            show=False,
        )
    assert plt.get_fignums() == open_figures

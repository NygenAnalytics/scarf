from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import scarf.mapping.confidence as mapping_confidence
import scarf.plotting as splt
from tests.test_mapping_label_transfer import (
    _copied_query,
    _plain_reference,
    _write_projection,
    _write_reference_layout,
)


@pytest.fixture
def plotting_mapping_context(analyzed_datastore_ephemeral, tmp_path: Path):
    reference_store = analyzed_datastore_ephemeral
    reference = _plain_reference(reference_store)
    query = _copied_query(reference_store, tmp_path / "plotting_query.zarr")
    reference_layout, _ = _write_reference_layout(
        reference,
        layout_key="mapping_layout",
        linked=True,
    )
    reference_labels = np.full(
        reference.selected_cell_count,
        "other",
        dtype=object,
    )
    reference_labels[:4] = ["A", "A", "B", "B"]
    reference.datastore.cells.insert(
        "mapping_label",
        reference_labels,
        key=reference.cell_key,
        overwrite=True,
    )

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
        mapping_name="plot_diagnostics",
        indices=indices,
        distances=distances,
        uninformative=uninformative,
    )
    return {
        "reference": reference,
        "reference_layout": reference_layout,
        "reference_labels": reference_labels,
        "query": query,
        "result": result,
        "query_groups": np.asarray(["q1", "q1", "q2", "q2", "q1", "q2"]),
        "known_labels": np.asarray(["A", "A", "B", "B", "A", "B"]),
    }


def _sorted_rows(values: np.ndarray) -> np.ndarray:
    return values[np.lexsort((values[:, 1], values[:, 0]))]


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
        layout_key="mapping_layout",
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
        layout_key="mapping_layout",
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
    assert score.provenance.extras["mapping_name"] == result.mapping_name

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


def test_mapping_plots_resolve_string_result_with_query_assay(
    plotting_mapping_context,
):
    context = plotting_mapping_context
    plot = splt.mapping_score(
        context["query"],
        "plot_diagnostics",
        reference=context["reference"],
        query_assay="RNA",
        kind="histogram",
        show=False,
    )

    assert plot.provenance.extras["mapping_name"] == "plot_diagnostics"
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


def test_mapping_score_uses_mapping_name_as_default_title(
    plotting_mapping_context,
):
    context = plotting_mapping_context
    plot = splt.mapping_score(
        context["query"],
        context["result"],
        reference=context["reference"],
        layout_key="mapping_layout",
        show=False,
    )

    assert plot.tables["scores"]["group"].unique().tolist() == [0]
    assert next(iter(plot.axes.values())).get_title() == "plot_diagnostics"
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
    reference.datastore.cells.insert(
        "mapping_numeric_label",
        numeric_labels,
        key=reference.cell_key,
        overwrite=True,
    )

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
        )
    )
    evidence = query.get_target_label_evidence(
        result,
        reference_class_group="mapping_label",
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
            layout_key="mapping_layout",
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

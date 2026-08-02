from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import scarf.mapping.confidence as mapping_confidence
import scarf.plotting as splt
from scarf.storage.artifacts import artifact_group
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
    embedding = query.project_reference_embedding(
        result,
        reference_layout_key="mapping_layout",
    )
    query_coordinates = np.asarray(
        artifact_group(query.zw, embedding)["values"][:],
        dtype=np.float64,
    )
    return {
        "reference": reference,
        "reference_layout": reference_layout,
        "reference_labels": reference_labels,
        "query": query,
        "result": result,
        "embedding": embedding,
        "query_coordinates": query_coordinates,
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
    evidence_embedding = splt.mapping_evidence(
        query,
        result,
        reference=reference,
        reference_class_group="mapping_label",
        metrics=("voteFraction", "referenceDistancePercentile"),
        kind="embedding",
        reference_layout_key="mapping_layout",
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
    projection = splt.mapping_projection(
        query,
        result,
        reference=reference,
        reference_layout_key="mapping_layout",
        reference_class_group="mapping_label",
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
    assert set(evidence_embedding.axes) == {
        "voteFraction",
        "referenceDistancePercentile",
    }
    assert confusion.tables["counts"].set_index("known").loc["A", "A"] >= 2
    assert {"precision", "recall", "support"} <= set(confusion.tables["perClass"])
    assert {"coverage", "accuracy", "accuracyLower", "accuracyUpper"} <= set(
        calibration.tables["calibration"]
    )
    assert calibration.tables["calibration"]["coverage"].between(0, 1).all()

    projected_cells = projection.tables["cells"]
    reference_rows = projected_cells.loc[projected_cells["source"] == "reference"]
    query_rows = projected_cells.loc[projected_cells["source"] == "query"]
    np.testing.assert_allclose(
        reference_rows[["x", "y"]].to_numpy(dtype=np.float64),
        context["reference_layout"],
    )
    np.testing.assert_allclose(
        query_rows[["x", "y"]].to_numpy(dtype=np.float64),
        context["query_coordinates"],
        equal_nan=True,
    )
    assert "query_only" not in set(reference_rows["group"])
    assert set(context["reference_labels"]) <= set(reference_rows["group"])
    assert projection.provenance.extras["n_reference"] == n_reference
    assert projection.provenance.extras["mapping_name"] == result.mapping_name

    for plot in (
        score,
        evidence,
        evidence_embedding,
        confusion,
        calibration,
        projection,
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
    evidence.close()


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


@pytest.mark.parametrize("plot_name", ("mapping_evidence", "mapping_projection"))
def test_mapping_embedding_plots_require_persisted_reference_embedding(
    analyzed_datastore_ephemeral,
    tmp_path: Path,
    plot_name: str,
):
    reference = _plain_reference(analyzed_datastore_ephemeral)
    query = _copied_query(
        analyzed_datastore_ephemeral,
        tmp_path / f"missing_{plot_name}.zarr",
    )
    _write_reference_layout(
        reference,
        layout_key="missing_layout",
        linked=False,
    )
    reference.datastore.cells.insert(
        "missing_labels",
        np.full(reference.selected_cell_count, "A", dtype=object),
        key=reference.cell_key,
        overwrite=True,
    )
    result = _write_projection(
        query,
        reference,
        mapping_name="missing_embedding",
        indices=np.asarray([[0, 1], [1, 0]], dtype=np.uint64),
        distances=np.ones((2, 2), dtype=np.float64),
        uninformative=np.zeros(2, dtype=bool),
    )
    before = set(query.list_artifacts(kind="embedding", from_assay="RNA"))

    kwargs = {
        "reference": reference,
        "reference_layout_key": "missing_layout",
        "show": False,
    }
    if plot_name == "mapping_evidence":
        kwargs.update(
            {
                "reference_class_group": "missing_labels",
                "kind": "embedding",
                "metrics": ("voteFraction",),
            }
        )
    with pytest.raises(ValueError, match="project_reference_embedding"):
        getattr(splt, plot_name)(query, result, **kwargs)

    assert set(query.list_artifacts(kind="embedding", from_assay="RNA")) == before


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
    monkeypatch.setattr(type(query), "project_reference_embedding", unexpected)
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
        splt.mapping_evidence(
            query,
            result,
            reference=reference,
            reference_class_group="mapping_label",
            kind="embedding",
            metrics=("voteFraction",),
            reference_layout_key="mapping_layout",
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
        splt.mapping_projection(
            query,
            result,
            reference=reference,
            reference_layout_key="mapping_layout",
            target_groups=context["query_groups"],
            show=False,
        ),
    )

    for plot in plots:
        plot.close()


def test_mapping_projection_background_mode_keeps_query_on_top(
    plotting_mapping_context,
):
    context = plotting_mapping_context
    plot = splt.mapping_projection(
        context["query"],
        context["result"],
        reference=context["reference"],
        reference_layout_key="mapping_layout",
        target_groups=context["query_groups"],
        reference_mode="background",
        show=False,
    )
    ax = plot.axes["mapping_projection"]
    assert len(ax.collections) == 2
    query_offsets = np.asarray(ax.collections[1].get_offsets(), dtype=np.float64)
    expected_visible = int(np.isfinite(context["query_coordinates"]).all(axis=1).sum())
    assert len(query_offsets) == expected_visible
    with pytest.raises(ValueError, match="reference_mode"):
        splt.mapping_projection(
            context["query"],
            context["result"],
            reference=context["reference"],
            reference_layout_key="mapping_layout",
            reference_mode="overlay",
            show=False,
        )
    plot.close()


def test_uninformative_query_rows_are_not_drawn_as_mapping_points(
    plotting_mapping_context,
):
    context = plotting_mapping_context
    expected_visible = int(np.isfinite(context["query_coordinates"]).all(axis=1).sum())
    projection = splt.mapping_projection(
        context["query"],
        context["result"],
        reference=context["reference"],
        reference_layout_key="mapping_layout",
        target_groups=context["query_groups"],
        show=False,
    )
    evidence = splt.mapping_evidence(
        context["query"],
        context["result"],
        reference=context["reference"],
        reference_class_group="mapping_label",
        metrics=("voteFraction",),
        kind="embedding",
        reference_layout_key="mapping_layout",
        show_unknown=False,
        show=False,
    )

    projection_offsets = (
        projection.axes["mapping_projection"].collections[1].get_offsets()
    )
    evidence_offsets = evidence.axes["voteFraction"].collections[1].get_offsets()
    assert len(projection_offsets) == expected_visible
    assert len(evidence_offsets) == expected_visible
    assert expected_visible == context["result"].n_cells - 1
    query_rows = projection.tables["cells"].loc[
        projection.tables["cells"]["source"] == "query"
    ]
    assert query_rows[["x", "y"]].isna().all(axis=1).sum() == 1
    assert evidence.tables["evidence"].loc[4, "isUnknown"]
    assert np.isnan(evidence.tables["evidence"].loc[4, "voteFraction"])
    projection.close()
    evidence.close()


def test_reference_embedding_adapter_chooses_newest_exact_artifact(
    plotting_mapping_context,
):
    context = plotting_mapping_context
    query = context["query"]
    first = context["embedding"]
    second = query.project_reference_embedding(
        context["result"],
        reference=context["reference"],
        reference_layout_key="mapping_layout",
        invalidate_cache=True,
    )
    first_group = artifact_group(query.zw, first)
    second_group = artifact_group(query.zw, second)
    first_group.attrs["created_at_ns"] = 10
    second_group.attrs["created_at_ns"] = 20
    expected = np.asarray(second_group["values"][:], dtype=np.float64)
    expected[np.isfinite(expected)] += 100.0
    second_group["values"][:] = expected

    loaded = query._load_mapping_reference_embedding(
        context["result"],
        "mapping_layout",
        reference=context["reference"],
    )

    np.testing.assert_allclose(loaded, expected, equal_nan=True)


def test_reference_embedding_adapter_matches_current_layout_fingerprint(
    analyzed_datastore_ephemeral,
    tmp_path: Path,
):
    reference = _plain_reference(analyzed_datastore_ephemeral)
    query = _copied_query(
        analyzed_datastore_ephemeral,
        tmp_path / "layout_fingerprint_query.zarr",
    )
    layout, _ = _write_reference_layout(
        reference,
        layout_key="mutable_layout",
        linked=False,
    )
    result = _write_projection(
        query,
        reference,
        mapping_name="layout_fingerprint",
        indices=np.asarray([[0, 1], [1, 0]], dtype=np.uint64),
        distances=np.ones((2, 2), dtype=np.float64),
        uninformative=np.zeros(2, dtype=bool),
    )
    query.project_reference_embedding(
        result,
        reference_layout_key="mutable_layout",
    )
    changed = layout[:, 0] + 1.0
    reference.datastore.cells.insert(
        "mutable_layout1",
        changed,
        key=reference.cell_key,
        overwrite=True,
    )

    with pytest.raises(ValueError, match="project_reference_embedding"):
        query._load_mapping_reference_embedding(
            result,
            "mutable_layout",
            reference=reference,
        )


def test_reference_embedding_adapter_rejects_infinite_coordinates(
    plotting_mapping_context,
):
    context = plotting_mapping_context
    group = artifact_group(context["query"].zw, context["embedding"])
    values = np.asarray(group["values"][:], dtype=np.float64)
    values[0, 0] = np.inf
    group["values"][:] = values

    with pytest.raises(ValueError, match="infinite coordinates"):
        context["query"]._load_mapping_reference_embedding(
            context["result"],
            "mapping_layout",
            reference=context["reference"],
        )

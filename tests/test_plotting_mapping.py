import shutil
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
from scarf.datastore.datastore import DataStore
from scarf.mapping.projection import (
    NO_QUERY_BATCH_FINGERPRINT,
    ProjectionWriter,
    plan_projection,
)
from scarf.metadata.artifacts import (
    plan_cell_data_artifact,
    write_cell_data_artifact,
)
from scarf.storage import ArtifactRef
from scarf.storage.artifacts import fingerprint_array
from scarf.storage.feature_selection import (
    _feature_selection_plan,
    _ordered_feature_ids_fingerprint,
    _write_feature_selection,
)
from scarf.storage.selections import (
    read_stored_selection_indices,
    resolve_generated_selection_artifact,
)

# Mapping store builders, kept local so this module does not depend on another
# test module's private helpers.


def _plain_reference(datastore):
    graphs = datastore.list_artifacts(
        kind="connectivity_map",
        from_assay="RNA",
        scope="assay",
        complete_only=True,
    )
    assert len(graphs) == 1
    neighbors = ArtifactRef.from_dict(
        datastore.inspect_artifact(graphs[0]).inputs["neighbors"]
    )
    reference_ref = datastore.build_mapping_reference(neighbors)
    return datastore.get_mapping_reference(reference_ref)


def _copied_query(datastore, path: Path, *, zarr_mode: str = "r+") -> DataStore:
    shutil.copytree(datastore.zarr_loc, path)
    return DataStore(
        str(path),
        default_assay="RNA",
        zarr_mode=zarr_mode,
    )


def _write_projection(
    query,
    reference,
    *,
    indices: np.ndarray,
    distances: np.ndarray,
    uninformative: np.ndarray,
    cell_key: str = "mapping_cells",
    feature_coverage: float = 1.0,
) -> ArtifactRef:
    index_values = np.asarray(indices, dtype=np.uint64)
    distance_values = np.asarray(distances, dtype=np.float64)
    uninformative_values = np.asarray(uninformative, dtype=bool)
    n_cells = len(index_values)
    if cell_key == "I":
        cell_mask = np.asarray(query.cells.fetch_all("I"), dtype=bool)
        assert int(cell_mask.sum()) == n_cells
    else:
        cell_mask = np.zeros(query.cells.N, dtype=bool)
        cell_mask[:n_cells] = True
        query.cells.insert(cell_key, cell_mask, overwrite=True)
    cell_selection = resolve_generated_selection_artifact(
        query.zw,
        scope="datastore",
        kind="cell_selection",
        values=cell_mask,
        row_ids=np.asarray(query.cells.fetch_all("ids")),
        operation="manual_selection",
        parameters={},
        inputs={},
        source_column=cell_key,
    )[0]
    all_features = query.select_all_features(from_assay="RNA")
    query_feature_ids = np.asarray(query.RNA.feats.fetch_all("ids")).astype(str)
    reference_feature_ids = np.asarray(reference.feature_ids).astype(str)
    feature_mask = np.isin(query_feature_ids, reference_feature_ids)
    feature_ids_fingerprint = _ordered_feature_ids_fingerprint(query.RNA.z)
    feature_plan = _feature_selection_plan(
        query.zw,
        assay="RNA",
        n_features=query.RNA.feats.N,
        ordered_feature_ids_fingerprint=feature_ids_fingerprint,
        operation="select_mapping_overlap",
        parameters={},
        inputs={
            "mapping_reference": reference.external_ref,
            "all_features": all_features,
        },
        execution_options={},
        expected_payload_fingerprint=fingerprint_array(feature_mask),
    )
    _write_feature_selection(
        query.zw,
        feature_plan,
        ordered_feature_ids_fingerprint=feature_ids_fingerprint,
        payload={"values": feature_mask},
    )
    feature_selection = feature_plan.ref
    planned = plan_projection(
        query.zw,
        query_assay="RNA",
        n_cells=n_cells,
        save_k=index_values.shape[1],
        missing_feature_policy="reference_mean",
        correction_method="none",
        cell_selection=cell_selection,
        feature_selection=feature_selection,
        query_dataset_fingerprint=query._ensure_dataset_fingerprint("RNA"),
        query_batch_fingerprint=NO_QUERY_BATCH_FINGERPRINT,
        query_batch_count=1,
        mapping_reference=reference.external_ref,
        reference=reference,
        reference_cell_count=reference.selected_cell_count,
    )
    writer = ProjectionWriter(
        query.zw,
        planned,
        chunk_rows=max(1, min(n_cells, 2)),
    )
    writer.write_block(
        0,
        index_values,
        distance_values,
        uninformative_values,
    )
    ref = writer.finish(
        {
            "featureCoverage": float(feature_coverage),
            "queryBatchCount": 1,
            "algorithmVariant": "scaled_pca",
            "uninformativeCellCount": int(np.count_nonzero(uninformative_values)),
            "queryScaledDispersion": 1.0,
        }
    )
    return ref


def _reference_cell_indices(reference) -> np.ndarray:
    return read_stored_selection_indices(
        reference.datastore.zw,
        reference.cell_selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )


def _write_reference_column(reference, name: str, values: np.ndarray) -> None:
    compact = np.asarray(values)
    indices = _reference_cell_indices(reference)
    if compact.ndim != 1 or len(compact) != len(indices):
        raise ValueError("Reference metadata must have one value per selected cell")
    full = np.empty(reference.datastore.cells.N, dtype=compact.dtype)
    if compact.dtype.kind in {"O", "S", "U"}:
        full[:] = ""
    else:
        full[:] = 0
    full[indices] = compact
    reference.datastore.cells.insert(name, full, overwrite=True)


def _write_reference_layout(
    reference,
    *,
    name: str,
) -> tuple[np.ndarray, ArtifactRef]:
    first = np.arange(reference.selected_cell_count, dtype=np.float64) * 10
    layout = np.column_stack((first, first + 10))
    planned = plan_cell_data_artifact(
        reference.datastore.zw,
        scope="assay",
        assay=reference.assay_name,
        kind="embedding",
        operation="manual_reference_embedding",
        parameters={"name": name},
        inputs={},
        execution_options={},
        cell_selection=reference.cell_selection,
        arrays={"values": (layout.shape, "f")},
    )
    write_cell_data_artifact(
        reference.datastore.zw,
        planned,
        {"values": layout},
    )
    return layout, planned.ref


_RESULT_REF = ArtifactRef(
    scope="assay", assay="RNA", kind="projection", artifact_id="a" * 64
)
_TRANSFER_REF = ArtifactRef(
    scope="assay", assay="RNA", kind="label_transfer", artifact_id="b" * 64
)


# Plots only read saved mapping results, so one context serves the module. The
# one test that adds a reference column and a second transfer leaves every
# saved artifact the other tests read untouched.
@pytest.fixture(scope="module")
def plotting_mapping_context(analyzed_datastore_zarr_root, tmp_path_factory):
    root = tmp_path_factory.mktemp("plotting_mapping")
    shutil.copytree(analyzed_datastore_zarr_root, root / "reference.zarr")
    reference_store = DataStore(str(root / "reference.zarr"), default_assay="RNA")
    reference = _plain_reference(reference_store)
    query = _copied_query(reference_store, root / "plotting_query.zarr")
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
    transfer = query.run_label_transfer(
        result,
        reference=reference,
        reference_labels="mapping_label",
    )
    return {
        "reference": reference,
        "reference_layout": reference_layout,
        "layout_ref": layout_ref,
        "reference_labels": reference_labels,
        "query": query,
        "result": result,
        "transfer": transfer,
        "query_groups": np.asarray(["q1", "q1", "q2", "q2", "q1", "q2"]),
        "known_labels": np.asarray(["A", "A", "B", "B", "A", "B"]),
    }


def _sorted_rows(values: np.ndarray) -> np.ndarray:
    return values[np.lexsort((values[:, 1], values[:, 0]))]


def _neighbor_weight(distance: float) -> float:
    """The documented reference-side weight of one neighbor at ``distance``."""
    return 1.0 / (np.log1p(distance) + 1.0)


def _expected_scores(n_reference: int) -> dict[str, np.ndarray]:
    """Scores of the context's projection, derived from its neighbor table.

    Each group sums its informative rows' neighbor weights, scales them by
    1000 / (rows * k) with k = 2 neighbors, and takes log1p. Query row 4 is
    uninformative and adds nothing.
    """
    w = _neighbor_weight
    q1 = np.zeros(n_reference)
    q1[[0, 1]] = np.log1p(250.0 * np.array([w(1) + w(2), w(9) + w(8)]))
    q2 = np.zeros(n_reference)
    q2[[2, 3]] = np.log1p(1000.0 / 6.0 * np.array([2 * w(1) + w(2), 2 * w(9) + w(8)]))
    pooled = np.zeros(n_reference)
    pooled[:4] = np.log1p(
        100.0 * np.array([w(1) + w(2), w(9) + w(8), 2 * w(1) + w(2), 2 * w(9) + w(8)])
    )
    return {"q1": q1, "q2": q2, "pooled": pooled}


def _score_matrix(table: pd.DataFrame) -> pd.DataFrame:
    return table.pivot(index="referenceIndex", columns="group", values="score")


def _controlled_mapping_store(
    *,
    evidence: pd.DataFrame | None = None,
    score_rows: list[tuple[object, np.ndarray]] | None = None,
    n_reference: int | None = None,
    threshold_fraction: float = 0.5,
    max_distance: float | None = None,
    reference_classes: np.ndarray | None = None,
    layout: np.ndarray | None = None,
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
    methods = {}
    if evidence is not None:
        methods["get_label_transfer"] = lambda *_args, **_kwargs: SimpleNamespace(
            evidence=evidence.copy(),
            threshold_fraction=threshold_fraction,
            max_distance=max_distance,
        )
    if score_rows is not None:
        methods["_mapping_score_data"] = lambda *_args, **_kwargs: (
            mapping,
            list(score_rows),
            reference_classes,
            layout,
        )
    return SimpleNamespace(**methods)


def test_mapping_plot_families_use_query_result_and_reference_semantics(
    plotting_mapping_context,
    monkeypatch,
):
    import scarf.datastore._operations.mapping as mapping_operations
    import scarf.mapping.artifact as mapping_artifact

    binding_checks = 0
    original_binding = mapping_artifact.validate_mapping_reference_binding

    def observe_binding(reference):
        nonlocal binding_checks
        binding_checks += 1
        return original_binding(reference)

    monkeypatch.setattr(
        mapping_artifact, "validate_mapping_reference_binding", observe_binding
    )
    monkeypatch.setattr(
        mapping_operations, "validate_mapping_reference_binding", observe_binding
    )
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
    transfer = context["transfer"]
    evidence = splt.mapping_evidence(
        query,
        transfer,
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
        reference_labels="mapping_label",
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
        transfer,
        known_labels=known_labels,
        show=False,
    )
    calibration = splt.mapping_calibration(
        query,
        transfer,
        known_labels=known_labels,
        n_thresholds=3,
        chosen_threshold=0.75,
        show=False,
    )

    assert set(score.axes) == {"q1", "q2"}
    n_reference = reference.selected_cell_count
    assert len(score.tables["scores"]) == 2 * n_reference
    expected = _expected_scores(n_reference)
    scores = _score_matrix(score.tables["scores"])
    np.testing.assert_allclose(scores["q1"], expected["q1"])
    np.testing.assert_allclose(scores["q2"], expected["q2"])
    score_coordinates = np.asarray(
        next(iter(score.axes.values())).collections[0].get_offsets(),
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        _sorted_rows(score_coordinates),
        _sorted_rows(context["reference_layout"]),
    )
    # Only reference cells that received weight are overlaid, lowest first.
    mapped = score.axes["q1"].collections[1]
    np.testing.assert_allclose(
        mapped.get_offsets(),
        context["reference_layout"][[1, 0]],
    )
    np.testing.assert_allclose(mapped.get_array(), expected["q1"][[1, 0]])
    assert "mapping_name" not in score.provenance.extras

    assert set(evidence.axes) == {"voteFraction", "topTwoMargin"}
    assert len(evidence.tables["evidence"]) == len(query_groups)
    assert set(score_boxes.axes) == {"q1", "q2"}
    assert [text.get_text() for text in score_boxes.axes["q1"].get_xticklabels()] == [
        "A",
        "B",
        "other",
    ]
    assert (
        score_boxes.tables["scores"]["referenceClass"].tolist()
        == list(context["reference_labels"]) * 2
    )
    assert set(sized.axes) == {"q1", "q2"}
    assert sized.provenance.extras["size_by_score"] is True
    sized_collections = sized.axes["q1"].collections
    background_sizes = sized_collections[0].get_sizes()
    mapped_sizes = sized_collections[1].get_sizes()
    assert mapped_sizes.min() > background_sizes.max()
    # Sizes grow with the score, and cells are drawn in ascending score order.
    assert np.all(np.diff(mapped_sizes) > 0)
    counts = confusion.tables["counts"].set_index("known")
    assert counts.loc["A", "A"] == 2
    assert counts.loc["A", "Abstained"] == 1
    assert counts.loc["B", "B"] == 3
    per_class = confusion.tables["perClass"].set_index("label")
    np.testing.assert_allclose(
        per_class.loc[["A", "B"], ["precision", "recall", "support"]],
        [[1.0, 2 / 3, 3], [1.0, 1.0, 3]],
    )
    table = calibration.tables["calibration"]
    assert {"coverage", "accuracy", "accuracyLower", "accuracyUpper"} <= set(table)
    assert calibration.tables["calibration"]["coverage"].between(0, 1).all()
    # Every cell the transfer labels is labelled correctly at any threshold.
    assert (table["accuracy"] == 1.0).all()
    assert table["nAccepted"].max() == 5
    for plot in (evidence, confusion, calibration):
        assert plot.provenance.extras["label_transfer"] == transfer.to_dict()
    assert evidence.provenance.extras["threshold_fraction"] == 0.5
    # Only the score plots read the reference; transfer plots read saved results.
    assert binding_checks == 3

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

    n_reference = context["reference"].selected_cell_count
    scores = _score_matrix(plot.tables["scores"])
    assert list(scores.columns) == [0]
    expected = _expected_scores(n_reference)["pooled"]
    np.testing.assert_allclose(scores[0], expected)
    # A step histogram traces each bin height on its horizontal segments.
    step = plot.axes["mapping_score"].patches[0].get_xy()
    heights, _ = np.histogram(expected, np.histogram_bin_edges(expected, bins=40))
    np.testing.assert_array_equal(step[1:-1:2, 1], heights)
    assert plot.provenance.extras["groups"] == [0]
    assert "mapping_name" not in plot.provenance.extras
    plot.close()


def _box_statistics(axis) -> list[tuple[float, float, float]]:
    """Return (first quartile, median, third quartile) of each drawn box.

    A median is the flat line that spans its box's full width.
    """
    statistics = []
    for patch in axis.patches:
        vertices = patch.get_path().vertices
        left, right = vertices[:, 0].min(), vertices[:, 0].max()
        (median,) = [
            line
            for line in axis.lines
            if np.allclose(line.get_xdata(), [left, right])
            and np.ptp(line.get_ydata()) == 0
        ]
        statistics.append(
            (
                float(vertices[:, 1].min()),
                float(median.get_ydata()[0]),
                float(vertices[:, 1].max()),
            )
        )
    return statistics


def test_mapping_evidence_box_kind_draws_one_box_per_query_group():
    from matplotlib.colors import to_rgba

    evidence = pd.DataFrame(
        {
            "label": ["A", "A", "B", "B", None, "B"],
            "voteFraction": [0.2, 0.4, 0.6, 0.8, np.nan, 1.0],
        }
    )
    scale = splt.CategoricalScale(
        order=("q1", "q2"),
        palette={"q1": "#336699", "q2": "#cc5500"},
    )
    plot = plotting_mapping.mapping_evidence(
        _controlled_mapping_store(evidence=evidence),
        _TRANSFER_REF,
        target_groups=["q1", "q1", "q1", "q2", "q2", "q2"],
        metrics=("voteFraction",),
        kind="box",
        categorical_scale=scale,
        show=False,
    )

    axis = plot.axes["voteFraction"]
    assert [text.get_text() for text in axis.get_xticklabels()] == ["q1", "q2"]
    assert all(text.get_ha() == "right" for text in axis.get_xticklabels())
    # q1 holds 0.2, 0.4 and 0.6; q2 holds 0.8 and 1.0 once its NaN is dropped.
    np.testing.assert_allclose(
        _box_statistics(axis),
        [(0.3, 0.4, 0.5), (0.85, 0.9, 0.95)],
    )
    assert [patch.get_facecolor() for patch in axis.patches] == [
        to_rgba("#336699", 0.8),
        to_rgba("#cc5500", 0.8),
    ]
    assert axis.get_xlabel() == "voteFraction"
    plot.close()


def test_mapping_score_box_kind_groups_by_reference_class():
    store = _controlled_mapping_store(
        score_rows=[("all", np.asarray([1.0, 2.0, 3.0, 10.0, 0.0, 5.0]))],
        reference_classes=np.asarray(["A", "A", "A", "B", None, "B"], dtype=object),
    )
    plot = plotting_mapping.mapping_score(
        store,
        _RESULT_REF,
        reference=object(),
        kind="box",
        reference_labels="mapping_label",
        show=False,
    )

    axis = plot.axes["all"]
    # The unlabelled reference cell joins no box.
    assert [text.get_text() for text in axis.get_xticklabels()] == ["A", "B"]
    assert all(text.get_ha() == "right" for text in axis.get_xticklabels())
    np.testing.assert_allclose(
        _box_statistics(axis),
        [(1.5, 2.0, 2.5), (6.25, 7.5, 8.75)],
    )
    assert axis.get_title() == "all query cells"
    classes = plot.tables["scores"]["referenceClass"]
    assert classes.isna().tolist() == [False] * 4 + [True, False]
    assert classes.dropna().tolist() == ["A", "A", "A", "B", "B"]
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
        context["transfer"],
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
    transfer = context["query"].run_label_transfer(
        context["result"],
        reference=reference,
        reference_labels="mapping_numeric_label",
    )

    with pytest.raises(ValueError, match="after text conversion"):
        splt.mapping_calibration(
            context["query"],
            transfer,
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
            context["transfer"],
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
        context["transfer"],
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
    import scarf.datastore._operations.mapping as mapping_operations
    import scarf.mapping.label_transfer as label_transfer

    context = plotting_mapping_context
    query = context["query"]
    result = context["result"]
    transfer = context["transfer"]
    reference = context["reference"]

    def unexpected(*args, **kwargs):
        raise AssertionError("plotting attempted mapping computation")

    # Score plots read the saved projection and never project query cells.
    monkeypatch.setattr(mapping_operations, "AlignedFeatureStream", unexpected)
    monkeypatch.setattr(mapping_operations, "plan_projection", unexpected)
    monkeypatch.setattr(mapping_confidence, "distance_weights", unexpected)
    # Transfer plots read the saved votes and never vote again.
    monkeypatch.setattr(label_transfer, "distance_weights", unexpected)
    monkeypatch.setattr(label_transfer, "_label_vote_block", unexpected)

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
            reference_labels="mapping_label",
            show=False,
        ),
        splt.mapping_evidence(
            query,
            transfer,
            target_groups=context["query_groups"],
            metrics=("voteFraction",),
            kind="box",
            show=False,
        ),
        splt.mapping_confusion(
            query,
            transfer,
            known_labels=context["known_labels"],
            show=False,
        ),
        splt.mapping_calibration(
            query,
            transfer,
            known_labels=context["known_labels"],
            n_thresholds=3,
            show=False,
        ),
    )

    expected = _expected_scores(reference.selected_cell_count)
    for score_plot in plots[:2]:
        scores = _score_matrix(score_plot.tables["scores"])
        np.testing.assert_allclose(scores["q1"], expected["q1"])
        np.testing.assert_allclose(scores["q2"], expected["q2"])
    for plot in plots:
        plot.close()


def test_mapping_calibration_respects_direction_and_draws_uncertainty():
    evidence = pd.DataFrame(
        {
            "label": ["A", "B", "B", "A"],
            "candidateLabel": ["A", "B", "B", "A"],
            "voteFraction": [0.9, 0.7, 0.4, 0.1],
            "meanNeighborDistance": [0.1, 0.3, 0.6, 0.9],
            "customConfidence": [0.9, 0.7, 0.4, 0.1],
        }
    )
    # A zero vote cutoff keeps every candidate eligible for the other metrics.
    store = _controlled_mapping_store(evidence=evidence, threshold_fraction=0.0)
    known = np.asarray(["A", "B", "A", "B"])

    higher = plotting_mapping.mapping_calibration(
        store,
        _TRANSFER_REF,
        known_labels=known,
        metric="voteFraction",
        direction="auto",
        thresholds=[0.0, 0.5, 0.8],
        chosen_threshold=0.5,
        show=False,
    )
    lower = plotting_mapping.mapping_calibration(
        store,
        _TRANSFER_REF,
        known_labels=known,
        metric="meanNeighborDistance",
        direction="auto",
        thresholds=[0.2, 0.5, 1.0],
        show=False,
    )
    explicit = plotting_mapping.mapping_calibration(
        store,
        _TRANSFER_REF,
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

    from scipy.stats import binomtest

    # Thresholds 0.8, 0.5 and 0.0 keep 1, 2 and 4 cells, of which 1, 2 and 2
    # carry their known label.
    calibration = higher.tables["calibration"]
    np.testing.assert_allclose(calibration["coverage"], [0.25, 0.5, 1.0])
    np.testing.assert_allclose(calibration["accuracy"], [1.0, 1.0, 0.5])
    for row, (correct, accepted) in zip(
        calibration.itertuples(), [(1, 1), (2, 2), (2, 4)], strict=True
    ):
        interval = binomtest(correct, accepted).proportion_ci(method="wilson")
        assert (row.accuracyLower, row.accuracyUpper) == pytest.approx(
            (interval.low, interval.high)
        )
        assert row.nAccepted == accepted

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
    kwargs,
    message,
):
    evidence = pd.DataFrame(
        {
            "label": ["A", "B"],
            "candidateLabel": ["A", "B"],
            "voteFraction": [0.8, 0.2],
            "customConfidence": [0.8, 0.2],
        }
    )
    store = _controlled_mapping_store(evidence=evidence)

    with pytest.raises(ValueError, match=message):
        plotting_mapping.mapping_calibration(
            store,
            _TRANSFER_REF,
            known_labels=np.asarray(["A", "B"]),
            show=False,
            **kwargs,
        )


def test_mapping_calibration_rejects_nonfinite_or_unretained_evidence():
    evidence = pd.DataFrame(
        {
            "label": ["A", "B"],
            "candidateLabel": ["A", "B"],
            "voteFraction": [np.nan, np.nan],
        }
    )
    store = _controlled_mapping_store(evidence=evidence)
    with pytest.raises(ValueError, match="No finite metric values"):
        plotting_mapping.mapping_calibration(
            store,
            _TRANSFER_REF,
            known_labels=np.asarray(["A", "B"]),
            show=False,
        )

    finite_evidence = evidence.assign(voteFraction=[0.8, 0.2])
    store = _controlled_mapping_store(evidence=finite_evidence)
    with pytest.raises(ValueError, match="No threshold retained"):
        plotting_mapping.mapping_calibration(
            store,
            _TRANSFER_REF,
            known_labels=np.asarray(["A", "B"]),
            thresholds=[2.0],
            show=False,
        )


def test_mapping_calibration_marks_the_transfer_threshold_without_extra_rows():
    import warnings

    evidence = pd.DataFrame(
        {
            "label": ["A", "B", "B", "A"],
            "candidateLabel": ["A", "B", "B", "A"],
            "voteFraction": [0.9, 0.7, 0.4, 0.1],
        }
    )
    known = np.asarray(["A", "B", "A", "B"])
    store = _controlled_mapping_store(evidence=evidence, threshold_fraction=0.5)

    default = plotting_mapping.mapping_calibration(
        store, _TRANSFER_REF, known_labels=known, show=False
    )
    explicit = plotting_mapping.mapping_calibration(
        store,
        _TRANSFER_REF,
        known_labels=known,
        thresholds=[0.6, 0.8],
        show=False,
    )
    unreachable = _controlled_mapping_store(evidence=evidence, threshold_fraction=0.95)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        silent = plotting_mapping.mapping_calibration(
            unreachable, _TRANSFER_REF, known_labels=known, show=False
        )

    assert 0.5 in default.tables["calibration"]["threshold"].tolist()
    assert default.provenance.extras["marked_threshold"] == 0.5
    assert any(
        text.get_text() == "voteFraction = 0.5"
        for text in default.axes["mapping_calibration"].texts
    )
    # Explicit thresholds are evaluated exactly, without the transfer's own.
    assert sorted(explicit.tables["calibration"]["threshold"]) == [0.6, 0.8]
    assert explicit.provenance.extras["marked_threshold"] is None
    assert explicit.provenance.extras["chosen_threshold"] is None
    # An implicit marker that retains no cells is omitted without a warning.
    assert silent.provenance.extras["marked_threshold"] is None
    for plot in (default, explicit, silent):
        plot.close()


def test_mapping_calibration_keeps_the_transfers_other_rules():
    evidence = pd.DataFrame(
        {
            "label": ["A", None, None],
            "candidateLabel": ["A", "B", "A"],
            "voteFraction": [0.9, 0.6, 0.95],
            "topTwoMargin": [0.8, 0.2, 0.9],
            "nearestDistance": [1.0, 1.0, 5.0],
        }
    )
    known = np.asarray(["A", "A", "B"])
    # The second cell falls below the vote cutoff and the third lies beyond
    # the distance limit, so the transfer labels only the first.
    store = _controlled_mapping_store(
        evidence=evidence,
        threshold_fraction=0.8,
        max_distance=2.0,
    )

    def first_row(metric: str) -> dict[str, float]:
        plot = plotting_mapping.mapping_calibration(
            store,
            _TRANSFER_REF,
            known_labels=known,
            metric=metric,
            direction="higher" if metric != "nearestDistance" else "lower",
            thresholds=[0.0] if metric != "nearestDistance" else [10.0],
            show=False,
        )
        row = plot.tables["calibration"].iloc[0]
        plot.close()
        return {"coverage": row["coverage"], "accuracy": row["accuracy"]}

    # Another metric is calibrated among the cells the transfer labelled.
    assert first_row("topTwoMargin") == {"coverage": 1 / 3, "accuracy": 1.0}
    # Sweeping a rule's own metric replaces that rule and keeps the other one.
    assert first_row("voteFraction") == {"coverage": 2 / 3, "accuracy": 0.5}
    assert first_row("nearestDistance") == {"coverage": 2 / 3, "accuracy": 0.5}

    marked = plotting_mapping.mapping_calibration(
        store,
        _TRANSFER_REF,
        known_labels=known,
        metric="nearestDistance",
        show=False,
    )
    assert marked.provenance.extras["marked_threshold"] == 2.0
    marked.close()


def test_label_transfer_plots_require_a_transfer_loader_and_label():
    with pytest.raises(TypeError, match="does not provide label transfers"):
        plotting_mapping.mapping_evidence(object(), _TRANSFER_REF, show=False)
    evidence = pd.DataFrame({"label": ["A"], "voteFraction": [0.9]})
    store = _controlled_mapping_store(evidence=evidence)
    for label in ("", None):
        with pytest.raises(TypeError, match="abstention_label must be"):
            plotting_mapping.mapping_confusion(
                store,
                _TRANSFER_REF,
                known_labels=np.asarray(["A"]),
                abstention_label=label,
                show=False,
            )


def test_mapping_score_rejects_a_malformed_reference_label_source():
    with pytest.raises(ValueError, match="reference_labels is required"):
        plotting_mapping.mapping_score(
            object(), _RESULT_REF, reference=object(), kind="box", show=False
        )
    for source in ("", 3):
        with pytest.raises(TypeError, match="reference_labels must be"):
            plotting_mapping.mapping_score(
                object(),
                _RESULT_REF,
                reference=object(),
                kind="box",
                reference_labels=source,
                show=False,
            )


def test_mapping_score_and_evidence_reject_nonpositive_bins():
    with pytest.raises(ValueError, match="^bins must be positive$"):
        plotting_mapping.mapping_score(
            object(),
            _RESULT_REF,
            reference=object(),
            kind="histogram",
            bins=0,
            show=False,
        )
    with pytest.raises(ValueError, match="^bins must be positive$"):
        plotting_mapping.mapping_evidence(object(), _TRANSFER_REF, bins=0, show=False)
    with pytest.raises(ValueError, match="^kind must be 'histogram' or 'box'$"):
        plotting_mapping.mapping_evidence(
            object(), _TRANSFER_REF, kind="violin", show=False
        )


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        (
            {"kind": "violin"},
            ValueError,
            "kind must be 'embedding', 'histogram', or 'box'",
        ),
        ({}, ValueError, "layout is required for an embedding mapping score"),
        (
            {"kind": "histogram", "size_by_score": True},
            ValueError,
            "size_by_score is only supported for kind='embedding'",
        ),
        (
            {"kind": "histogram"},
            TypeError,
            "store does not provide mapping score data",
        ),
    ],
)
def test_mapping_score_rejects_inconsistent_requests(kwargs, error, message):
    with pytest.raises(error) as raised:
        plotting_mapping.mapping_score(
            object(), _RESULT_REF, reference=object(), show=False, **kwargs
        )

    assert raised.value.args == (message,)


def test_mapping_score_requires_scores_aligned_with_the_reference():
    misaligned = _controlled_mapping_store(
        score_rows=[("all", np.asarray([0.1, 0.2]))],
        n_reference=3,
    )
    with pytest.raises(ValueError) as raised:
        plotting_mapping.mapping_score(
            misaligned, _RESULT_REF, reference=object(), kind="histogram", show=False
        )
    assert raised.value.args == (
        "Mapping scores do not match the selected reference cells",
    )

    short_classes = _controlled_mapping_store(
        score_rows=[("all", np.asarray([0.1, 0.2]))],
        reference_classes=np.asarray(["A"], dtype=object),
    )
    with pytest.raises(ValueError) as raised:
        plotting_mapping.mapping_score(
            short_classes,
            _RESULT_REF,
            reference=object(),
            kind="box",
            reference_labels="mapping_label",
            show=False,
        )
    assert raised.value.args == (
        "Reference class labels must contain one value per reference cell",
    )


def test_mapping_score_embedding_without_weighted_cells_colors_nothing():
    layout = np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    store = _controlled_mapping_store(
        score_rows=[("q1", np.zeros(3)), ("q2", np.zeros(3))],
        layout=layout,
    )
    plot = plotting_mapping.mapping_score(
        store,
        _RESULT_REF,
        reference=object(),
        target_groups=["q1", "q2"],
        layout=ArtifactRef(
            scope="assay", assay="RNA", kind="embedding", artifact_id="c" * 64
        ),
        color_scale=splt.ColorScale(cmap="magma", scope="shared"),
        show=False,
    )

    for group in ("q1", "q2"):
        background, mapped = plot.axes[group].collections
        np.testing.assert_allclose(background.get_offsets(), layout)
        assert len(mapped.get_offsets()) == 0
    # With no positive score the shared limits come from every score; a flat
    # range keeps its low end and widens by one.
    assert [spec.extras for spec in plot.legends] == [
        {"vmin": 0.0, "vmax": 1.0},
        {"vmin": 0.0, "vmax": 1.0},
    ]
    plot.close()


def test_mapping_plots_reject_empty_and_misaligned_data():
    empty_store = _controlled_mapping_store(
        score_rows=[],
        n_reference=2,
    )
    with pytest.raises(ValueError, match="produced no score groups"):
        plotting_mapping.mapping_score(
            empty_store,
            _RESULT_REF,
            reference=object(),
            kind="histogram",
            show=False,
        )

    mismatched_store = _controlled_mapping_store(
        score_rows=[
            ("first", np.asarray([0.1, 0.2])),
            ("second", np.asarray([0.3])),
        ],
        n_reference=2,
    )
    with pytest.raises(ValueError, match="incompatible lengths"):
        plotting_mapping.mapping_score(
            mismatched_store,
            _RESULT_REF,
            reference=object(),
            kind="histogram",
            show=False,
        )

    evidence = pd.DataFrame(
        {
            "label": ["A", "B"],
            "candidateLabel": ["A", "B"],
            "voteFraction": [0.8, 0.2],
        }
    )
    evidence_store = _controlled_mapping_store(evidence=evidence)
    with pytest.raises(ValueError, match="one value per mapped cell"):
        plotting_mapping.mapping_evidence(
            evidence_store,
            _TRANSFER_REF,
            target_groups=["only-one"],
            metrics=("voteFraction",),
            show=False,
        )
    with pytest.raises(ValueError, match="cannot contain missing values"):
        plotting_mapping.mapping_evidence(
            evidence_store,
            _TRANSFER_REF,
            target_groups=["first", None],
            metrics=("voteFraction",),
            show=False,
        )
    with pytest.raises(ValueError, match="metrics must be non-empty"):
        plotting_mapping.mapping_evidence(
            evidence_store,
            _TRANSFER_REF,
            metrics=(),
            show=False,
        )
    with pytest.raises(KeyError, match="Unknown evidence metrics"):
        plotting_mapping.mapping_evidence(
            evidence_store,
            _TRANSFER_REF,
            metrics=("missingMetric",),
            show=False,
        )


def test_mapping_layout_rejects_wrong_shape_and_infinite_coordinates():
    with pytest.raises(ValueError, match="two columns"):
        plotting_mapping._reference_layout(np.zeros((2, 3)), 2)

    with pytest.raises(ValueError, match="infinite coordinates"):
        plotting_mapping._reference_layout(np.asarray([[0.0, 1.0], [np.inf, 2.0]]), 2)


def test_mapping_categorical_legends_serialize_and_owned_figures_close(
    tmp_path: Path,
):
    import json

    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    evidence = pd.DataFrame(
        {
            "label": ["A", "B", "A"],
            "candidateLabel": ["A", "B", "A"],
            "voteFraction": [0.9, 0.4, 0.7],
        }
    )
    score_rows = [
        ("beta", np.asarray([0.1, 0.2, 0.3])),
        ("alpha", np.asarray([0.4, 0.5, 0.6])),
    ]
    store = _controlled_mapping_store(
        evidence=evidence,
        score_rows=score_rows,
    )
    scale = splt.CategoricalScale(
        order=("alpha", "beta"),
        palette={"alpha": "#336699", "beta": "#cc5500"},
    )
    scores = plotting_mapping.mapping_score(
        store,
        _RESULT_REF,
        reference=object(),
        kind="histogram",
        categorical_scale=scale,
        bins=3,
        show=False,
    )
    evidence_plot = plotting_mapping.mapping_evidence(
        store,
        _TRANSFER_REF,
        target_groups=["beta", "alpha", "beta"],
        metrics=("voteFraction",),
        categorical_scale=scale,
        bins=3,
        show=False,
    )

    # Shared edges over the six scores, [0.1, 0.2667, 0.4333, 0.6], put beta
    # in the lower bins and alpha in the upper ones.
    alpha_step, beta_step = (
        patch.get_xy()[1:-1:2, 1] for patch in scores.axes["mapping_score"].patches
    )
    np.testing.assert_array_equal(alpha_step, [0, 1, 2])
    np.testing.assert_array_equal(beta_step, [2, 1, 0])
    assert [
        to_hex(patch.get_edgecolor()) for patch in scores.axes["mapping_score"].patches
    ] == ["#336699", "#cc5500"]
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
            _RESULT_REF,
            reference=object(),
            kind="histogram",
            show=False,
        )
    assert plt.get_fignums() == open_figures


def _confusion_store():
    evidence = pd.DataFrame(
        {
            "label": ["A", "A", "B", None, "B", "B"],
            "candidateLabel": ["A", "A", "B", "A", "B", "B"],
            "voteFraction": [0.9, 0.8, 0.7, 0.3, 0.9, 0.6],
        }
    )
    return _controlled_mapping_store(evidence=evidence)


@pytest.mark.parametrize(
    ("normalize", "expected"),
    [
        # Known A cells were labelled A, A, B and abstained once; both known B
        # cells were labelled B.
        ("none", [[2, 1, 1], [0, 0, 2]]),
        ("true", [[0.5, 0.25, 0.25], [0, 0, 1]]),
        ("predicted", [[1, 1, 1 / 3], [0, 0, 2 / 3]]),
        ("all", [[2 / 6, 1 / 6, 1 / 6], [0, 0, 2 / 6]]),
    ],
)
def test_mapping_confusion_normalizes_counts(normalize, expected):
    plot = plotting_mapping.mapping_confusion(
        _confusion_store(),
        _TRANSFER_REF,
        known_labels=np.asarray(["A", "A", "A", "A", "B", "B"]),
        normalize=normalize,
        show=False,
    )

    matrix = plot.tables["matrix"].set_index("known")
    assert list(matrix.columns) == ["A", "Abstained", "B"]
    np.testing.assert_allclose(matrix.to_numpy(), expected)
    np.testing.assert_allclose(
        plot.axes["mapping_confusion"].images[0].get_array(), expected
    )
    assert plot.legends[0].label == (
        "Cells" if normalize == "none" else "Fraction of cells"
    )
    plot.close()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"normalize": "rows"},
            "normalize must be 'none', 'true', 'predicted', or 'all'",
        ),
        (
            {"known_labels": np.asarray(["A", "B"])},
            "known_labels must have one value per mapped cell",
        ),
        ({"known_order": ["A"]}, "known_order is missing observed labels"),
        (
            {"predicted_order": ["A", "B"]},
            "predicted_order is missing observed labels",
        ),
    ],
)
def test_mapping_confusion_rejects_inconsistent_labels_and_orders(kwargs, message):
    options = {"known_labels": np.asarray(["A", "A", "B", "A", "B", "B"]), **kwargs}

    with pytest.raises(ValueError) as raised:
        plotting_mapping.mapping_confusion(
            _confusion_store(), _TRANSFER_REF, show=False, **options
        )

    assert raised.value.args == (message,)


def test_mapping_calibration_rejects_unknown_metrics_and_misaligned_labels():
    store = _confusion_store()

    with pytest.raises(KeyError, match="Evidence has no threshold metric 'margin'"):
        plotting_mapping.mapping_calibration(
            store,
            _TRANSFER_REF,
            known_labels=np.asarray(["A"] * 6),
            metric="margin",
            direction="higher",
            show=False,
        )
    with pytest.raises(ValueError) as raised:
        plotting_mapping.mapping_calibration(
            store, _TRANSFER_REF, known_labels=np.asarray(["A"]), show=False
        )
    assert raised.value.args == ("known_labels must have one value per mapped cell",)


def test_mapping_plots_show_owned_results_by_default(monkeypatch):
    from scarf.plotting._figure import PlotResult

    shown = []
    monkeypatch.setattr(PlotResult, "show", lambda result: shown.append(result))
    known = np.asarray(["A", "A", "B", "A", "B", "B"])
    store = _confusion_store()
    score_store = _controlled_mapping_store(
        score_rows=[("all", np.asarray([0.1, 0.4]))],
    )

    plots = [
        plotting_mapping.mapping_score(
            score_store, _RESULT_REF, reference=object(), kind="histogram"
        ),
        plotting_mapping.mapping_evidence(
            store, _TRANSFER_REF, metrics=("voteFraction",)
        ),
        plotting_mapping.mapping_confusion(store, _TRANSFER_REF, known_labels=known),
        plotting_mapping.mapping_calibration(
            store, _TRANSFER_REF, known_labels=known, n_thresholds=3
        ),
    ]

    assert shown == plots
    for plot in plots:
        plot.close()

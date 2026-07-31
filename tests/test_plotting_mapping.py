import warnings

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import scarf.plotting as splt
from scarf.writers import create_zarr_dataset
from tests.fixtures_datastore import build_neighbourhood_graph


def _ensure_graph(datastore) -> None:
    try:
        datastore.get_latest_graph_loc("RNA", "I", "hvgs")
    except KeyError:
        datastore.auto_filter_cells(show_qc_plots=False)
        datastore.mark_hvgs(top_n=100, show_plot=False, bin_strategy="fixed")
        build_neighbourhood_graph(datastore, feat_key="hvgs")


def _manual_mapping(datastore) -> tuple[np.ndarray, np.ndarray]:
    _ensure_graph(datastore)
    labels = np.full(datastore.cells.N, "other", dtype=object)
    labels[:4] = ["A", "A", "B", "B"]
    datastore.cells.insert("mapping_label", labels, overwrite=True)
    layout_x = np.arange(datastore.cells.N, dtype=np.float64)
    layout_y = np.sin(layout_x / 3)
    datastore.cells.insert("mapping_layout1", layout_x, overwrite=True)
    datastore.cells.insert("mapping_layout2", layout_y, overwrite=True)

    indices = np.asarray(
        [
            [0, 1],
            [0, 1],
            [2, 3],
            [2, 3],
            [0, 2],
            [2, 0],
        ],
        dtype=np.uint64,
    )
    distances = np.asarray(
        [
            [0.0, 1.0],
            [0.2, 0.8],
            [0.0, 1.0],
            [0.2, 0.8],
            [0.1, 0.9],
            [0.1, 0.9],
        ],
        dtype=np.float64,
    )
    assay = datastore.RNA
    if "projections" not in assay.z:
        assay.z.create_group("projections")
    projection = assay.z["projections"].create_group(
        "plot_diagnostics",
        overwrite=True,
    )
    stored_indices = create_zarr_dataset(
        projection,
        "indices",
        indices.shape,
        "u8",
        indices.shape,
    )
    stored_distances = create_zarr_dataset(
        projection,
        "distances",
        distances.shape,
        "f8",
        distances.shape,
    )
    stored_indices[:] = indices
    stored_distances[:] = distances

    before = np.column_stack(
        (
            np.linspace(-2, 2, len(indices)),
            np.asarray([-1.0, -0.8, 0.9, 1.1, -0.2, 0.2]),
        )
    )
    after = before * 0.65
    for name, values in (
        ("uncorrected_latent", before),
        ("corrected_latent", after),
    ):
        array = create_zarr_dataset(
            projection,
            name,
            values.shape,
            "f8",
            values.shape,
        )
        array[:] = values
    projection.attrs.update(
        {
            "assay": "RNA",
            "complete": True,
            "correction_method": "symphony",
        }
    )
    return indices, distances


def test_mapping_diagnostic_plots_use_persisted_mapping_arrays(
    datastore_ephemeral,
):
    datastore = datastore_ephemeral
    _manual_mapping(datastore)
    query_groups = np.asarray(["q1", "q1", "q2", "q2", "q1", "q2"])
    known_labels = np.asarray(["A", "A", "B", "B", "A", "B"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        score = splt.mapping_score(
            datastore,
            target_name="plot_diagnostics",
            target_groups=query_groups,
            layout_key="mapping_layout",
            show=False,
        )
        evidence = splt.mapping_evidence(
            datastore,
            target_name="plot_diagnostics",
            reference_class_group="mapping_label",
            target_groups=query_groups,
            metrics=("voteFraction", "topTwoMargin"),
            show=False,
        )
        evidence_embedding = splt.mapping_evidence(
            datastore,
            target_name="plot_diagnostics",
            reference_class_group="mapping_label",
            metrics=("voteFraction", "referenceDistancePercentile"),
            kind="embedding",
            reference_layout_key="mapping_layout",
            show=False,
        )
        confusion = splt.mapping_confusion(
            datastore,
            target_name="plot_diagnostics",
            reference_class_group="mapping_label",
            known_labels=known_labels,
            show=False,
        )
        calibration = splt.mapping_calibration(
            datastore,
            target_name="plot_diagnostics",
            reference_class_group="mapping_label",
            known_labels=known_labels,
            n_thresholds=3,
            chosen_threshold=0.75,
            show=False,
        )
        correction = splt.mapping_correction(
            datastore,
            target_name="plot_diagnostics",
            batch_labels=query_groups,
            show=False,
        )
        projection = splt.mapping_projection(
            datastore,
            target_name="plot_diagnostics",
            reference_layout_key="mapping_layout",
            reference_class_group="mapping_label",
            show=False,
        )

    assert set(score.axes) == {"q1", "q2"}
    n_reference = len(datastore.cells.fetch("mapping_layout1", key="I"))
    assert len(score.tables["scores"]) == 2 * n_reference
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
    assert set(correction.axes) == {"before", "after", "displacement"}
    assert correction.tables["cells"]["displacement"].gt(0).all()
    projected_cells = projection.tables["cells"]
    assert (projected_cells["source"] == "query").sum() == len(query_groups)
    assert projection.provenance.extras["n_reference"] == n_reference


def test_mapping_evidence_box_kind_draws_one_box_per_query_group(
    datastore_ephemeral,
):
    datastore = datastore_ephemeral
    _manual_mapping(datastore)
    query_groups = np.asarray(["q1", "q1", "q2", "q2", "q1", "q2"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        evidence = splt.mapping_evidence(
            datastore,
            target_name="plot_diagnostics",
            reference_class_group="mapping_label",
            target_groups=query_groups,
            metrics=("voteFraction",),
            kind="box",
            show=False,
        )

    axis = evidence.axes["voteFraction"]
    assert [text.get_text() for text in axis.get_xticklabels()] == ["q1", "q2"]


def test_mapping_score_keeps_store_group_value_and_uses_target_as_title(
    datastore_ephemeral,
):
    datastore = datastore_ephemeral
    _manual_mapping(datastore)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = splt.mapping_score(
            datastore,
            target_name="plot_diagnostics",
            layout_key="mapping_layout",
            show=False,
        )

    assert result.tables["scores"]["group"].unique().tolist() == [0.0]
    assert next(iter(result.axes.values())).get_title() == "plot_diagnostics"
    result.close()


def test_mapping_calibration_allows_genuine_zero_accuracy(
    datastore_ephemeral,
):
    datastore = datastore_ephemeral
    _manual_mapping(datastore)
    known_labels = np.asarray(["B", "B", "A", "A", "B", "A"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = splt.mapping_calibration(
            datastore,
            target_name="plot_diagnostics",
            reference_class_group="mapping_label",
            known_labels=known_labels,
            n_thresholds=3,
            show=False,
        )

    assert (result.tables["calibration"]["accuracy"] == 0).all()
    result.close()


def test_mapping_calibration_rejects_pairwise_label_type_mismatch(
    datastore_ephemeral,
):
    datastore = datastore_ephemeral
    _manual_mapping(datastore)
    numeric_labels = np.zeros(datastore.cells.N, dtype=np.int64)
    numeric_labels[:4] = [1, 1, 2, 2]
    datastore.cells.insert("mapping_numeric_label", numeric_labels, overwrite=True)
    known_labels = np.asarray(["1", "1", "2", "2", "1", "2"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(ValueError, match="after text conversion"):
            splt.mapping_calibration(
                datastore,
                target_name="plot_diagnostics",
                reference_class_group="mapping_numeric_label",
                known_labels=known_labels,
                n_thresholds=3,
                show=False,
            )


def test_mapping_calibration_warns_when_chosen_threshold_retains_nothing(
    datastore_ephemeral,
):
    datastore = datastore_ephemeral
    _manual_mapping(datastore)
    known_labels = np.asarray(["A", "A", "B", "B", "A", "B"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.warns(RuntimeWarning, match="retained no mapped cells"):
            result = splt.mapping_calibration(
                datastore,
                target_name="plot_diagnostics",
                reference_class_group="mapping_label",
                known_labels=known_labels,
                chosen_threshold=2.0,
                n_thresholds=3,
                show=False,
            )

    assert not any(
        "voteFraction =" in text.get_text()
        for text in result.axes["mapping_calibration"].texts
    )
    result.close()


def test_mapping_diagnostic_plots_accept_caller_owned_targets(
    datastore_ephemeral,
):
    datastore = datastore_ephemeral
    _manual_mapping(datastore)
    import matplotlib.pyplot as pyplot

    figure, axes = pyplot.subplots(1, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = splt.mapping_evidence(
            datastore,
            target_name="plot_diagnostics",
            reference_class_group="mapping_label",
            metrics=("voteFraction", "topTwoMargin"),
            target={
                "voteFraction": axes[0],
                "topTwoMargin": axes[1],
            },
            show=False,
        )

    assert result.figure is figure
    assert not result.owns_figure

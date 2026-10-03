"""Feature-operation paths of the DataStore on small stores.

Each test reaches one path of ``scarf.datastore._operations.features``
through the public API: argument validation, reuse of saved results, empty
selections, and saved records that a tool outside Scarf edited.
"""

import dataclasses
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
import zarr

from scarf import ArtifactRef
from scarf.datastore._operations import features as features_module
from scarf.datastore.datastore import DataStore
from scarf.metadata.selection import CellField, StudyDesign
from scarf.storage.artifacts import (
    artifact_path,
    fingerprint_array,
    fingerprint_stored_arrays,
)
from scarf.storage.errors import ArtifactResolutionError
from scarf.storage.feature_selection import (
    _feature_selection_plan,
    _ordered_feature_ids_fingerprint,
    _write_feature_selection,
)
from tests.storage_helpers import write_count_store

pytestmark = pytest.mark.filterwarnings("ignore:Cell-level statistical testing")

N_CELLS = 24
N_GENES = 12
EMPTY_SELECTION = "Feature selection must select at least one feature"
VALUES = np.random.default_rng(17).normal(size=N_CELLS)


def _counts() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(11)
    return {
        "RNA": rng.poisson(3.0, size=(N_CELLS, N_GENES)),
        "ADT": rng.poisson(5.0, size=(N_CELLS, 4)) + 1,
    }


def _open(zarr_loc: str) -> DataStore:
    return DataStore(
        zarr_loc,
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=1,
    )


def _new_store(
    directory: Any,
    counts: dict[str, np.ndarray] | None = None,
    dtype: str = "uint16",
) -> DataStore:
    zarr_loc = str(directory / "store.zarr")
    write_count_store(zarr_loc, _counts() if counts is None else counts, dtype)
    store = _open(zarr_loc)
    store.cells.insert("everyone", np.ones(store.cells.N, dtype=bool), overwrite=True)
    return store


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> DataStore:
    store = _new_store(tmp_path_factory.mktemp("feature_paths"))
    cells = np.arange(N_CELLS)
    samples = cells % 6
    columns = {
        "nobody": np.zeros(N_CELLS, dtype=bool),
        "grp": np.where(cells % 2 == 0, "a", "b").astype(object),
        "grp3": np.array(["a", "b", "c"], dtype=object)[cells % 3],
        "dose": np.where(cells % 2 == 0, 0.5, 1.5),
        "clusters": cells % 2 + 1,
        "sample": np.array([f"s{sample}" for sample in samples], dtype=object),
        "cond": np.where(samples % 2 == 0, "a", "b").astype(object),
        "subject": np.array([f"d{sample // 2}" for sample in samples], dtype=object),
        "val": VALUES,
    }
    for name, values in columns.items():
        store.cells.insert(name, values, overwrite=True)
    return store


def _cells(store: DataStore) -> ArtifactRef:
    return store.snapshot_cell_selection("everyone")


def _artifact_group(store: DataStore, ref: ArtifactRef) -> zarr.Group:
    """Open a saved artifact for editing, as a tool outside Scarf would."""
    return zarr.open_group(store.zw.store, mode="r+", path=artifact_path(ref))


def _empty_feature_selection(store: DataStore) -> ArtifactRef:
    """Save an empty selection the way a foreign writer could.

    Scarf's own selection producers refuse empty masks, so the record is
    written with the storage layer directly.
    """
    values = np.zeros(N_GENES, dtype=bool)
    ids_fingerprint = _ordered_feature_ids_fingerprint(store.RNA.z)
    values_fingerprint = fingerprint_array(values)
    planned = _feature_selection_plan(
        store.zw,
        assay="RNA",
        n_features=N_GENES,
        ordered_feature_ids_fingerprint=ids_fingerprint,
        operation="set_feature_selection",
        parameters={"values_fingerprint": values_fingerprint},
        inputs={"all_features": store.select_all_features()},
        execution_options={"invalidate_cache": False},
        expected_payload_fingerprint=values_fingerprint,
    )
    _write_feature_selection(
        store.zw,
        planned,
        ordered_feature_ids_fingerprint=ids_fingerprint,
        payload={"values": values},
    )
    return planned.ref


def _network(targets: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": ["set"] * targets,
            "target": [f"RNA{index}" for index in range(targets)],
            "weight": np.ones(targets),
        }
    )


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({}, ValueError, "exactly one of mask or feature_indexes"),
        (
            {"mask": np.ones(N_GENES, dtype=bool), "feature_indexes": [0]},
            ValueError,
            "exactly one of mask or feature_indexes",
        ),
        ({"mask": [True] * N_GENES}, TypeError, "mask must be a NumPy array"),
        ({"mask": np.ones(N_GENES - 1, dtype=bool)}, ValueError, "mask must have"),
        ({"mask": np.ones(N_GENES, dtype=np.int8)}, TypeError, "boolean dtype"),
        ({"feature_indexes": [[0, 1]]}, ValueError, "one-dimensional"),
        ({"feature_indexes": [0.0, 1.0]}, TypeError, "only integers"),
        ({"feature_indexes": [N_GENES]}, IndexError, "out-of-range"),
        ({"feature_indexes": [-1]}, IndexError, "out-of-range"),
        ({"feature_indexes": [1, 1]}, ValueError, "duplicate indexes"),
        ({"feature_indexes": []}, ValueError, "at least one feature"),
        ({"mask": np.zeros(N_GENES, dtype=bool)}, ValueError, "at least one feature"),
    ],
)
def test_set_feature_selection_rejects_invalid_requests(
    store: DataStore,
    arguments: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        store.set_feature_selection(**arguments)


def test_feature_mask_and_feature_indexes_give_one_selection(store: DataStore) -> None:
    mask = np.zeros(N_GENES, dtype=bool)
    mask[[0, 5]] = True

    by_indexes = store.set_feature_selection(feature_indexes=np.array([5, 0]))

    assert store.set_feature_selection(mask=mask) == by_indexes
    np.testing.assert_array_equal(store.load_artifact(by_indexes)["values"][:], mask)


@pytest.mark.parametrize(
    ("min_cells", "selection", "error", "message"),
    [
        (True, None, TypeError, "min_cells must be an integer"),
        (2.0, None, TypeError, "min_cells must be an integer"),
        (-1, None, ValueError, "min_cells must be non-negative"),
        (1, "everyone", TypeError, "cell_selection must be an ArtifactRef"),
    ],
)
def test_select_detected_features_rejects_invalid_arguments(
    store: DataStore,
    min_cells: Any,
    selection: str | None,
    error: type[Exception],
    message: str,
) -> None:
    cells = _cells(store) if selection is None else selection
    with pytest.raises(error, match=message):
        store.select_detected_features(cells, min_cells=min_cells)  # type: ignore[arg-type]


def test_repeated_detected_feature_selection_reuses_the_saved_artifact(
    store: DataStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    cells = _cells(store)
    first = store.select_detected_features(cells, min_cells=10)

    def no_summary(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("A reused selection must not read the feature summary")

    monkeypatch.setattr(features_module, "feature_summary_values", no_summary)
    assert store.select_detected_features(cells, min_cells=10) == first


def test_hvgs_over_few_cells_skip_the_ubiquitous_gene_filter(store: DataStore) -> None:
    cells = _cells(store)
    # 24 cells leave no room for max_cells = n_selected - 20 above min_cells.
    default = store.select_hvgs(cells, min_cells=5, top_n=4, n_bins=4, show_plot=False)

    assert (
        store.select_hvgs(
            cells, min_cells=5, top_n=4, n_bins=4, max_cells=np.inf, show_plot=False
        )
        == default
    )


def test_hvg_plot_shows_the_saved_selection_when_computed_and_reused(
    store: DataStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scarf.plotting as plotting

    plotted: list[dict[str, Any]] = []

    def record(**kwargs: Any) -> None:
        plotted.append(kwargs)

    monkeypatch.setattr(plotting, "highly_variable_features", record)
    cells = _cells(store)
    options = {"min_cells": 2, "max_cells": np.inf, "top_n": 5, "n_bins": 6}
    first = store.select_hvgs(cells, show_plot=True, **options)
    again = store.select_hvgs(cells, show_plot=True, **options)

    assert again == first
    saved = store.load_artifact(first)
    assert len(plotted) == 2
    for call in plotted:
        assert call["show"] is True
        np.testing.assert_array_equal(call["selected"], saved["values"][:])
        np.testing.assert_array_equal(
            call["corrected_variance"], saved["corrected_variance"][:]
        )
    np.testing.assert_array_equal(
        plotted[0]["mean_nonzero"], plotted[1]["mean_nonzero"]
    )
    np.testing.assert_array_equal(plotted[0]["n_cells"], plotted[1]["n_cells"])


def test_select_hvgs_requires_an_artifact_cell_selection(store: DataStore) -> None:
    with pytest.raises(TypeError, match="cell_selection must be an ArtifactRef"):
        store.select_hvgs("everyone", show_plot=False)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("producer", "options", "payload_names"),
    [
        (
            "select_hvgs",
            {
                "min_cells": 1,
                "max_cells": np.inf,
                "top_n": 4,
                "n_bins": 4,
                "show_plot": False,
            },
            ("values", "corrected_variance"),
        ),
        ("select_detected_features", {"min_cells": 1}, ("values",)),
    ],
)
def test_reused_selection_emptied_outside_scarf_is_refused_by_its_consumers(
    tmp_path,
    producer: str,
    options: dict[str, Any],
    payload_names: tuple[str, ...],
) -> None:
    store = _new_store(tmp_path)
    cells = _cells(store)
    select = getattr(store, producer)
    ref = select(cells, **options)
    # Values and fingerprint edited together, as a foreign writer would.
    group = _artifact_group(store, ref)
    group["values"][:] = np.zeros(N_GENES, dtype=bool)
    group.attrs["payload_fingerprint"] = fingerprint_stored_arrays(group, payload_names)

    # The producer reuses the record, and the feature-selection check refuses it.
    assert select(cells, **options) == ref
    with pytest.raises(ArtifactResolutionError, match=EMPTY_SELECTION) as caught:
        store.run_normalization(cells, ref)
    assert caught.value.code == "corrupt_payload"


@pytest.mark.parametrize("method", ["run_waggr", "run_aucell"])
def test_enrichment_requires_an_artifact_cell_selection(
    store: DataStore, method: str
) -> None:
    with pytest.raises(TypeError, match="cell_selection must be an ArtifactRef"):
        getattr(store, method)(
            _network(), "everyone", features=store.select_all_features(), tmin=2
        )


@pytest.mark.parametrize("method", ["run_waggr", "run_aucell"])
def test_enrichment_rejects_empty_selections(store: DataStore, method: str) -> None:
    score = getattr(store, method)
    nobody = store.snapshot_cell_selection("nobody")
    with pytest.raises(ValueError, match="Cell selection contains no active cells"):
        score(_network(), nobody, features=store.select_all_features(), tmin=2)
    with pytest.raises(ArtifactResolutionError, match=EMPTY_SELECTION):
        score(_network(), _cells(store), features=_empty_feature_selection(store))


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"mode": "median"}, ValueError, "mode must be 'wmean' or 'wsum'"),
        ({"log_transform": 1}, TypeError, "log_transform must be a boolean"),
    ],
)
def test_waggr_rejects_invalid_options(
    store: DataStore,
    options: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        store.run_waggr(
            _network(),
            _cells(store),
            features=store.select_all_features(),
            tmin=2,
            **options,
        )


@pytest.mark.parametrize("size_factor", [None, "many", 0, -1.0, np.inf, np.nan])
def test_waggr_requires_a_finite_positive_size_factor(
    store: DataStore, monkeypatch: pytest.MonkeyPatch, size_factor: Any
) -> None:
    monkeypatch.setattr(store.RNA, "sf", size_factor)
    with pytest.raises(ValueError, match="finite positive size factor"):
        store.run_waggr(
            _network(), _cells(store), features=store.select_all_features(), tmin=2
        )


def test_waggr_rejects_negative_cell_totals(tmp_path) -> None:
    counts = _counts()["RNA"].astype(np.int16)
    counts[0] = 0
    counts[0, 0] = -5
    store = _new_store(tmp_path, {"RNA": counts}, "int16")

    with pytest.raises(ValueError, match="RNA_nCounts holds negative or non-finite"):
        store.run_waggr(
            _network(), _cells(store), features=store.select_all_features(), tmin=2
        )
    assert store.list_artifacts(kind="enrichment_scores") == []


def test_get_enrichment_rejects_other_artifacts(store: DataStore) -> None:
    with pytest.raises(ValueError, match="assay enrichment_scores artifact"):
        store.get_enrichment(store.select_all_features())
    adt_scores = ArtifactRef(
        scope="assay", assay="ADT", kind="enrichment_scores", artifact_id="0" * 64
    )
    with pytest.raises(TypeError, match="only available for an RNAassay"):
        store.get_enrichment(adt_scores)


def test_marker_search_rejects_an_empty_feature_selection(store: DataStore) -> None:
    clusters = store.snapshot_cluster_labels("clusters", cell_selection=_cells(store))

    with pytest.raises(ArtifactResolutionError, match=EMPTY_SELECTION):
        store.run_marker_search(clusters, features=_empty_feature_selection(store))


@pytest.mark.parametrize(
    ("groups", "options", "error", "message"),
    [
        ("artifact", {"from_assay": "ADT"}, ValueError, "conflicts with the groups"),
        (7, {}, TypeError, "ArtifactRef or metadata column name"),
        ("", {}, ValueError, "column must be non-empty"),
        ("excluded_module", {}, ValueError, "No feature groups remain"),
    ],
)
def test_add_grouped_assay_rejects_invalid_groups(
    store: DataStore,
    groups: Any,
    options: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    store.RNA.feats.insert(
        "excluded_module", np.full(N_GENES, -1), fill_value=-1, overwrite=True
    )
    if groups == "artifact":
        groups = store.select_all_features()
    with pytest.raises(error, match=message):
        store.add_grouped_assay(groups, assay_label="grouped", **options)
    assert "grouped" not in store.assay_names


def test_add_melded_assay_rejects_invalid_inputs(store: DataStore, tmp_path) -> None:
    bed = tmp_path / "genes.bed"
    bed.write_text("chr1\t0\t100\tg1\tG1\n")

    with pytest.raises(ValueError, match="value for `assay_label`"):
        store.add_melded_assay(external_bed_fn=str(bed))
    with pytest.raises(ValueError, match="value for `external_bed_fn`"):
        store.add_melded_assay(assay_label="melded")
    with pytest.raises(ValueError, match="at least one selected cell"):
        store.add_melded_assay(
            external_bed_fn=str(bed), assay_label="melded", cell_key="nobody"
        )
    with pytest.raises(ValueError, match="element: RNA0 \\(position 0\\)"):
        store.add_melded_assay(external_bed_fn=str(bed), assay_label="melded")
    assert "melded" not in store.assay_names


def test_make_bulk_rejects_other_group_sources(store: DataStore) -> None:
    with pytest.raises(TypeError, match="ArtifactRef or column name"):
        store.make_bulk(3.5)  # type: ignore[arg-type]


def test_make_bulk_treats_fewer_than_one_pseudo_replicate_as_one(
    store: DataStore,
) -> None:
    cells = _cells(store)
    pd.testing.assert_frame_equal(
        store.make_bulk("grp3", cell_selection=cells, pseudo_reps=0),
        store.make_bulk("grp3", cell_selection=cells, pseudo_reps=1),
    )


def test_make_bulk_gives_zero_profiles_to_empty_pseudo_replicates(
    store: DataStore,
) -> None:
    labels = np.array(["solo"] + ["rest"] * (N_CELLS - 1), dtype=object)
    store.cells.insert("solo_group", labels, overwrite=True)

    bulk = store.make_bulk(
        "solo_group",
        from_assay="ADT",
        cell_selection=_cells(store),
        pseudo_reps=2,
        remove_empty_features=False,
    )

    assert bulk.columns.tolist() == ["rest_Rep1", "rest_Rep2", "solo_Rep1", "solo_Rep2"]
    assert bulk["solo_Rep2"].eq(0).all()
    assert not bulk["solo_Rep1"].eq(0).all()


def _statistical_tests(
    store: DataStore,
    keys: Any = "val",
    grouping: str = "grp",
    **options: Any,
) -> Any:
    return store.run_statistical_testing(
        keys,
        CellField(grouping),
        cell_selection=_cells(store),
        **options,
    )


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"alternative": "both"}, ValueError, "alternative must be"),
        ({"posthoc": "tukey"}, ValueError, "posthoc must be"),
        ({"sample_stat": "sum"}, ValueError, "sample_stat must be"),
        ({"test": "chi_square"}, ValueError, "test must be 'auto'"),
        (
            {"study_design": StudyDesign(sample_by="sample"), "sample_by": "subject"},
            ValueError,
            "conflicts with study_design.sample_by",
        ),
        ({"groups": []}, ValueError, "groups must be non-empty"),
        ({"comparisons": []}, ValueError, "comparisons must be non-empty"),
        ({"expression_cutoff": 0.5}, ValueError, "expression_cutoff requires"),
        ({"subset_by": "RNA_nCounts"}, TypeError, "must be boolean"),
        ({"subset_by": "nobody"}, ValueError, "No cells remain"),
    ],
)
def test_statistical_testing_rejects_invalid_requests(
    store: DataStore,
    options: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        _statistical_tests(store, skip_save=True, **options)


def test_statistical_testing_requires_keys(store: DataStore) -> None:
    with pytest.raises(ValueError, match="keys must be non-empty"):
        _statistical_tests(store, keys=[], skip_save=True)


@pytest.mark.parametrize("index", [N_GENES, -1])
def test_statistical_feature_keys_by_index_must_index_the_assay(
    store: DataStore, index: int
) -> None:
    from scarf.metadata.selection import FeatureRef

    with pytest.raises(
        KeyError, match=rf"Feature index {index} out of range for assay 'RNA' \(N=12\)"
    ):
        _statistical_tests(
            store, keys=FeatureRef(value=index, by="index"), skip_save=True
        )


def test_feature_values_accept_a_normalizer_that_returns_one_feature_as_a_vector(
    store: DataStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scarf.features.values import fetch_normalized_feature_matrix, resolve_feature

    def squeezed(_assay: Any, counts: Any) -> np.ndarray:
        # A user normalizer that drops the feature axis of one feature.
        return 2.0 * np.asarray(counts.compute())[:, 0]

    monkeypatch.setattr(store.ADT, "normMethod", squeezed)

    values = fetch_normalized_feature_matrix(
        store,
        [resolve_feature(store, "ADT1", from_assay="ADT")],
        np.arange(N_CELLS),
    )

    np.testing.assert_array_equal(values, 2.0 * _counts()["ADT"][:, [1]])


def test_automatic_test_for_three_groups_is_kruskal_wallis(store: DataStore) -> None:
    result = _statistical_tests(store, grouping="grp3", skip_save=True)

    assert result.method == "kruskal_wallis"
    assert result.group_order == ("a", "b", "c")


def test_statistical_feature_keys_come_from_the_requested_assay(
    store: DataStore,
) -> None:
    with pytest.raises(KeyError, match="not found in assay 'RNA'"):
        _statistical_tests(store, keys="ADT0", skip_save=True)

    result = _statistical_tests(store, keys="ADT0", from_assay="ADT", skip_save=True)

    assert result.source_assays == ("ADT",)


def test_explicit_pair_by_is_used_with_a_study_design(store: DataStore) -> None:
    paired = _statistical_tests(
        store,
        grouping="cond",
        study_design=StudyDesign(sample_by="sample", subject_by="subject"),
        pair_by="subject",
        test="wilcoxon",
        skip_save=True,
    )
    direct = _statistical_tests(
        store,
        grouping="cond",
        sample_by="sample",
        pair_by="subject",
        test="wilcoxon",
        skip_save=True,
    )

    assert (paired.method, paired.sample_by, paired.pair_by) == (
        "wilcoxon",
        "sample",
        "subject",
    )
    pd.testing.assert_frame_equal(paired.tables["val"], direct.tables["val"])


def test_study_design_without_subjects_tests_independent_samples(
    store: DataStore,
) -> None:
    designed = _statistical_tests(
        store,
        grouping="cond",
        study_design=StudyDesign(sample_by="sample"),
        test="mann_whitney",
        skip_save=True,
    )
    direct = _statistical_tests(
        store, grouping="cond", sample_by="sample", test="mann_whitney", skip_save=True
    )

    assert (designed.sample_by, designed.pair_by) == ("sample", None)
    pd.testing.assert_frame_equal(designed.tables["val"], direct.tables["val"])


def test_subset_by_leaves_out_cells_its_missing_mask_flags(store: DataStore) -> None:
    store.cells.insert("keep", np.ones(N_CELLS, dtype=bool), overwrite=True)
    missing = np.zeros(N_CELLS, dtype=bool)
    missing[:4] = True
    cell_data = store.cells.locations["primary"]
    cell_data.create_array(
        "__scarf_missing__keep", data=missing, chunks=(N_CELLS,), overwrite=True
    )
    cell_data["keep"].attrs["missing_mask"] = "__scarf_missing__keep"

    result = _statistical_tests(store, subset_by="keep", skip_save=True)

    assert result.n_cells == N_CELLS - 4


def test_float_group_labels_round_trip_through_the_saved_result(
    store: DataStore,
) -> None:
    result = _statistical_tests(store, grouping="dose")
    loaded = store.get_statistical_tests(result.artifact).tables["val"]

    assert loaded["group_1"].dtype == np.float64
    assert loaded.loc[0, ["group_1", "group_2"]].tolist() == [0.5, 1.5]
    pd.testing.assert_frame_equal(loaded, result.tables["val"])


def _replace_array(group: zarr.Group, name: str, width: int) -> None:
    del group[name]
    group.create_array(name, data=np.zeros((1, width)))


_POSTHOC = {"grouping": "grp3", "test": "kruskal_wallis", "posthoc": "dunn"}

_REUSE_EDITS: dict[str, tuple[dict[str, Any], Callable[[zarr.Group], None]]] = {
    "identity attribute": ({}, lambda group: group.attrs.update({"n_cells": 999})),
    "p-value method": (
        {},
        lambda group: group.attrs.update({"p_value_method": "permutation"}),
    ),
    "value fingerprints": (
        {},
        lambda group: group.attrs.update({"value_fingerprints": [""]}),
    ),
    "statistics width": ({}, lambda group: _replace_array(group["0"], "stats", 1)),
    "statistics dtypes": (
        {},
        lambda group: group["0"].attrs.update({"stats_dtypes": {"p_value": "x"}}),
    ),
    "missing key group": ({}, lambda group: group.__delitem__("0")),
    "post-hoc width": (
        _POSTHOC,
        lambda group: _replace_array(group["0"], "posthoc_stats", 1),
    ),
    "post-hoc dtypes": (
        _POSTHOC,
        lambda group: group["0"].attrs.update({"posthoc_stats_dtypes": {}}),
    ),
}


@pytest.mark.parametrize("edit", sorted(_REUSE_EDITS))
def test_statistical_result_edited_outside_scarf_is_recomputed(
    store: DataStore, edit: str
) -> None:
    options, apply_edit = _REUSE_EDITS[edit]
    key = f"reuse_edit_{sorted(_REUSE_EDITS).index(edit)}"
    store.cells.insert(key, VALUES, overwrite=True)
    first = _statistical_tests(store, keys=key, **options)
    apply_edit(_artifact_group(store, first.artifact))

    second = _statistical_tests(store, keys=key, **options)

    assert second.artifact != first.artifact
    pd.testing.assert_frame_equal(second.tables[key], first.tables[key])


def test_statistical_values_that_change_during_a_call_abort_it(
    store: DataStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.cells.insert("drifting", VALUES, overwrite=True)
    _statistical_tests(store, keys="drifting")
    saved = set(store.list_artifacts(kind="statistical_tests", scope="datastore"))
    original = store._iter_statistical_values
    passes: list[int] = []

    def drifting(keys: Any, **options: Any) -> Any:
        # Each pass reads values that another writer changed in between.
        passes.append(len(passes) + 1)
        for values in original(keys, **options):
            yield values + passes[-1]

    monkeypatch.setattr(store, "_iter_statistical_values", drifting)
    with pytest.raises(RuntimeError, match="changed while the result was being"):
        _statistical_tests(store, keys="drifting")

    assert passes == [1, 2]
    assert set(store.list_artifacts(kind="statistical_tests", scope="datastore")) == (
        saved
    )


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("p_value", np.nan, "must not contain NaN"),
        ("mean_1", np.inf, "Only t_statistic and f_statistic may be infinite"),
    ],
)
def test_statistical_results_with_undefined_statistics_are_not_saved(
    store: DataStore,
    monkeypatch: pytest.MonkeyPatch,
    column: str,
    value: float,
    message: str,
) -> None:
    original = features_module.compare_group_distributions

    def undefined(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        table = result.table.copy()
        table[column] = value
        return dataclasses.replace(result, table=table)

    monkeypatch.setattr(features_module, "compare_group_distributions", undefined)
    key = f"undefined_{column}"
    store.cells.insert(key, VALUES, overwrite=True)
    saved = set(store.list_artifacts(kind="statistical_tests", scope="datastore"))

    with pytest.raises(ValueError, match=message):
        _statistical_tests(store, keys=key)
    assert set(store.list_artifacts(kind="statistical_tests", scope="datastore")) == (
        saved
    )


def _rename_operation(group: zarr.Group) -> None:
    provenance = dict(group.attrs["provenance"])
    provenance["operation"] = "import_statistical_tests"
    group.attrs["provenance"] = provenance


_SLOT_EDITS: dict[
    str,
    tuple[dict[str, Any], Callable[[zarr.Group], None], type[Exception], str],
] = {
    "p-value method": (
        {"test": "welch"},
        lambda group: group.attrs.update({"p_value_method": "exact"}),
        ValueError,
        "p-value method metadata is invalid",
    ),
    "key labels": (
        {},
        lambda group: group.attrs.update({"key_labels": "values"}),
        ValueError,
        "key-label metadata is invalid",
    ),
    "value fingerprints": (
        {},
        lambda group: group.attrs.update({"value_fingerprints": []}),
        ValueError,
        "value-fingerprint metadata is invalid",
    ),
    "statistics dtypes": (
        {},
        lambda group: group["0"].attrs.update({"stats_dtypes": "float64"}),
        ValueError,
        "test dtype metadata is invalid",
    ),
    "post-hoc dtypes": (
        _POSTHOC,
        lambda group: group["0"].attrs.update({"posthoc_stats_dtypes": {}}),
        ValueError,
        "post-hoc dtype metadata is invalid",
    ),
    "no grouping": (
        {},
        lambda group: group.attrs.update({"group_field": None}),
        ValueError,
        "lacks one explicit grouping source",
    ),
    "grouping record": (
        {},
        lambda group: group.attrs.update({"grouping": "cells", "group_field": None}),
        ValueError,
        "grouping metadata is invalid",
    ),
    "group field": (
        {},
        lambda group: group.attrs.update({"group_field": ""}),
        ValueError,
        "group field is invalid",
    ),
    "incomplete": (
        {},
        lambda group: group.attrs.update({"complete": False}),
        RuntimeError,
        "artifact is incomplete",
    ),
    "foreign operation": (
        {},
        _rename_operation,
        ValueError,
        "not produced by run_statistical_testing",
    ),
}


@pytest.mark.parametrize("edit", sorted(_SLOT_EDITS))
def test_get_statistical_tests_rejects_records_edited_outside_scarf(
    store: DataStore, edit: str
) -> None:
    options, apply_edit, error, message = _SLOT_EDITS[edit]
    key = f"slot_edit_{sorted(_SLOT_EDITS).index(edit)}"
    store.cells.insert(key, VALUES, overwrite=True)
    saved = _statistical_tests(store, keys=key, **options).artifact
    apply_edit(_artifact_group(store, saved))

    with pytest.raises(error, match=message):
        store.get_statistical_tests(saved)

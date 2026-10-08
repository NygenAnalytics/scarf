"""The configured normalizer, ``log_transform``, and ``renormalize_subset``.

Saved normalization applies the assay's ``normMethod``: the optimized subset
writer runs only for library-size normalization. ``log_transform`` takes
``log1p`` of the configured normalizer's own output on every path for the
normalizers whose values are not logarithms, ``norm_lib_size``,
``norm_dummy``, and custom normalizers of RNA, ADT, and generic assays, and
``renormalize_subset`` hands an RNA normalizer the totals over the selected
features through ``assay.scalar``. A flag that a normalizer cannot apply
defaults to False and rejects True. Defaults turn flags on only for the
normalizers that compute from the counts, so ``norm_dummy`` and ADT values
are logged only when asked for.
"""

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scarf.assay.normalization import (
    norm_clr,
    norm_dummy,
    norm_lib_size,
    norm_lib_size_log,
    norm_tf_idf,
)
from scarf.datastore.datastore import DataStore
from scarf.features.markers import find_markers_by_rank, find_markers_by_regression
from scarf.features.markers.rank import _batch_stats
from scarf.features.scoring import binned_sampling
from scarf.plotting.heatmaps import _marker_log_transform, _prepare_marker_heatmap
from scarf.storage.artifacts import artifact_group, callable_identity
from scarf.storage.operation_revisions import effective_revision
from tests.storage_helpers import write_count_store

N_CELLS = 30
SIZE_FACTOR = 1000.0
SUBSET = np.array([0, 2, 3, 5, 6])
GROUPS = np.array(["a", "b", "c"])[np.arange(N_CELLS) % 3]


def _counts() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(41)
    counts = {
        "RNA": rng.poisson(2.0, size=(N_CELLS, 9)),
        "ADT": rng.poisson(6.0, size=(N_CELLS, 4)) + 1,
        "ATAC": rng.poisson(0.7, size=(N_CELLS, 8)),
    }
    # A cell without counts, and one without counts in the subset.
    counts["RNA"][4] = 0
    counts["RNA"][7, SUBSET] = 0
    return counts


COUNTS = _counts()


def scaled_by_totals(assay: Any, counts: Any) -> Any:
    """A linear custom RNA normalizer that reads the totals ``normed`` sets."""
    return 50.0 * counts / assay.scalar.reshape(-1, 1)


def doubled(_assay: Any, counts: Any) -> Any:
    """A custom normalizer that reads no totals."""
    return 2.0 * counts


def logged_scaled_by_totals(assay: Any, counts: Any) -> Any:
    """``log1p`` of ``scaled_by_totals``, the reference for ``log_transform``."""
    return np.log1p(scaled_by_totals(assay, counts))


def _scaled_reference(
    raw: np.ndarray, *, subset: np.ndarray | None = None, log: bool = False
) -> np.ndarray:
    """Return ``scaled_by_totals`` of every cell in float64, as ``normed`` does."""
    widened = raw.astype(np.float64)
    if subset is not None:
        widened = widened[:, subset]
    totals = widened.sum(axis=1)
    values = 50.0 * widened / np.where(totals == 0, 1.0, totals)[:, None]
    return np.log1p(values) if log else values


def _library_size_reference(
    raw: np.ndarray, *, subset: np.ndarray | None = None, log: bool = False
) -> np.ndarray:
    widened = raw.astype(np.float64)
    if subset is not None:
        widened = widened[:, subset]
    totals = widened.sum(axis=1)
    values = SIZE_FACTOR * widened / np.where(totals == 0, 1.0, totals)[:, None]
    return np.log1p(values) if log else values


def _open(path: Path) -> DataStore:
    # Every cell stays active, including those without counts.
    return DataStore(
        str(path), default_assay="RNA", min_features_per_cell=-1, nthreads=1
    )


@pytest.fixture(scope="module")
def store_template(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("normalization_flags") / "store.zarr"
    write_count_store(str(path), COUNTS, "uint16")
    _open(path)
    return path


@pytest.fixture
def store(store_template, tmp_path) -> DataStore:
    target = tmp_path / "store.zarr"
    shutil.copytree(store_template, target)
    return _open(target)


def _selections(store: DataStore, assay: str, features: np.ndarray):
    return (
        store.snapshot_cell_selection(),
        store.set_feature_selection(from_assay=assay, feature_indexes=features),
    )


def _stored(store: DataStore, ref) -> np.ndarray:
    return np.asarray(artifact_group(store.zw, ref)["data"][:])


def _normalized_refs(store: DataStore, assay: str) -> list:
    return list(store.list_artifacts(kind="normalized", from_assay=assay))


def _marker_tables(store: DataStore, ref) -> dict[str, dict[str, np.ndarray]]:
    group = artifact_group(store.zw, ref)
    return {
        name: {
            column: np.asarray(group[name][column][:])
            for column in group[name].array_keys()
        }
        for name in sorted(group.group_keys())
    }


def _every_path(
    store: DataStore, normalizer: Callable[..., Any], *, log_transform: bool
) -> dict[str, Any]:
    """Return the values that each normalized path computes for one setup."""
    rna = store.RNA
    rna.normMethod = normalizer
    cells = np.arange(N_CELLS)
    flags = {"log_transform": log_transform}
    regressor = np.linspace(0.0, 1.0, N_CELLS) + np.sin(np.arange(N_CELLS))
    cell_selection, features = _selections(store, "RNA", SUBSET)
    store.cells.insert("flag_groups", GROUPS, overwrite=True)
    clusters = store.snapshot_cluster_labels(
        "flag_groups", cell_selection=cell_selection
    )
    markers = store.run_marker_search(clusters, features=features, **flags)
    return {
        "normed": rna.normed(cells, SUBSET, **flags).compute(),
        "feature_wise": np.concatenate(
            [
                values
                for values, _labels in rna.iter_normed_feature_wise(
                    cells, SUBSET, 2, None, as_dataframe=False, **flags
                )
            ]
        ),
        "rank": find_markers_by_rank(rna, GROUPS, cells, SUBSET, **flags).statistics,
        "regression": find_markers_by_regression(
            rna, cells, SUBSET, regressor, 2, **flags
        ).to_numpy(),
        "summary": rna._compute_feature_summary(cells, np.arange(rna.feats.N), **flags),
        "marker_tables": _marker_tables(store, markers),
        "heatmap": _prepare_marker_heatmap(
            store, marker=markers, topn=2, log_transform=log_transform
        )["matrix"].to_numpy(),
    }


def _assert_same(expected: Any, actual: Any, path: str = "") -> None:
    if isinstance(expected, dict):
        assert expected.keys() == actual.keys(), path
        for key in expected:
            _assert_same(expected[key], actual[key], f"{path}/{key}")
        return
    np.testing.assert_array_equal(
        np.asarray(actual), np.asarray(expected), err_msg=path
    )


@pytest.mark.slow
def test_custom_normalizer_log_transform_logs_its_own_output_on_every_path(
    store,
) -> None:
    original = store.RNA.normMethod
    try:
        logged = _every_path(store, scaled_by_totals, log_transform=True)
        reference = _every_path(store, logged_scaled_by_totals, log_transform=False)
    finally:
        store.RNA.normMethod = original

    _assert_same(reference, logged)
    expected = _scaled_reference(COUNTS["RNA"], log=True)[:, SUBSET]
    np.testing.assert_array_equal(logged["normed"], expected)
    np.testing.assert_array_equal(logged["feature_wise"], expected.T)
    codes = np.unique(GROUPS, return_inverse=True)[1].astype(np.int64)
    np.testing.assert_array_equal(
        logged["rank"], _batch_stats(expected, codes, np.bincount(codes), N_CELLS)
    )
    # Library-size values would differ, so the custom normalizer was applied.
    assert not np.allclose(
        logged["normed"],
        _library_size_reference(COUNTS["RNA"], log=True)[:, SUBSET],
    )


def test_custom_normalizer_flags_apply_only_when_asked(store, monkeypatch) -> None:
    """Scarf cannot know a custom normalizer's scale, so defaults apply no flag."""
    monkeypatch.setattr(store.RNA, "normMethod", scaled_by_totals)
    cells, features = _selections(store, "RNA", SUBSET)

    default = store.run_normalization(cells, features)
    np.testing.assert_array_equal(
        _stored(store, default),
        _scaled_reference(COUNTS["RNA"])[:, SUBSET].astype(np.float32),
    )
    parameters = store.inspect_artifact(default).parameters
    assert parameters["log_transform"] is False
    assert parameters["renormalize_subset"] is False

    stored = store.run_normalization(
        cells, features, log_transform=True, renormalize_subset=True
    )
    np.testing.assert_array_equal(
        _stored(store, stored),
        _scaled_reference(COUNTS["RNA"], subset=SUBSET, log=True).astype(np.float32),
    )
    parameters = store.inspect_artifact(stored).parameters
    assert parameters["log_transform"] is True
    assert parameters["renormalize_subset"] is True


_LOG_ERROR = (
    r"does not support log_transform=True: log_transform takes log1p of values "
    r"that are not logarithms.*\. Pass log_transform=False\.$"
)
_SUBSET_ERROR = (
    r"does not support renormalize_subset=True: renormalize_subset hands totals "
    r"over the selected features.*\. Pass renormalize_subset=False\.$"
)


@pytest.mark.parametrize(
    ("assay_name", "normalizer", "subset_allowed"),
    [
        ("RNA", norm_clr, False),
        ("RNA", norm_lib_size_log, True),
        ("ATAC", norm_dummy, False),
    ],
    ids=["rna-clr", "rna-lib-size-log", "atac-dummy"],
)
def test_normalizers_reject_flags_they_cannot_apply(
    store, monkeypatch, assay_name, normalizer, subset_allowed
) -> None:
    assay = store.get_assay(assay_name)
    monkeypatch.setattr(assay, "normMethod", normalizer)
    cells = np.arange(N_CELLS)
    features = np.arange(assay.feats.N)
    cell_selection, feature_ref = _selections(store, assay_name, features[:3])
    store.cells.insert("flag_groups", GROUPS, overwrite=True)
    clusters = store.snapshot_cluster_labels(
        "flag_groups", cell_selection=cell_selection
    )
    log_calls: list[Callable[[], Any]] = [
        lambda: assay.normed(cells, features, log_transform=True),
        lambda: next(
            assay.iter_normed_feature_wise(
                cells, features, None, None, log_transform=True
            )
        ),
        lambda: store.run_normalization(
            cell_selection, feature_ref, log_transform=True
        ),
        lambda: find_markers_by_rank(
            assay, GROUPS, cells, features, log_transform=True
        ),
        lambda: find_markers_by_regression(
            assay,
            cells,
            features,
            np.arange(N_CELLS, dtype=float),
            2,
            log_transform=True,
        ),
        lambda: store.run_marker_search(
            clusters, from_assay=assay_name, features=feature_ref, log_transform=True
        ),
        lambda: assay.score_features(
            [f"{assay_name}0"], "I", 2, 2, 7, log_transform=True
        ),
    ]
    for call in log_calls:
        with pytest.raises(ValueError, match=_LOG_ERROR):
            call()
    if not subset_allowed:
        for call in (
            lambda: assay.normed(cells, features[:2], renormalize_subset=True),
            lambda: store.run_normalization(
                cell_selection, feature_ref, renormalize_subset=True
            ),
            lambda: find_markers_by_rank(
                assay, GROUPS, cells, features, renormalize_subset=True
            ),
            lambda: store.run_marker_search(
                clusters,
                from_assay=assay_name,
                features=feature_ref,
                renormalize_subset=True,
            ),
        ):
            with pytest.raises(ValueError, match=_SUBSET_ERROR):
                call()
    assert _normalized_refs(store, assay_name) == []

    # Defaults apply only what the normalizer can apply.
    stored = store.run_normalization(cell_selection, feature_ref)
    parameters = store.inspect_artifact(stored).parameters
    assert parameters["log_transform"] is False
    assert parameters["renormalize_subset"] is (subset_allowed and assay_name == "RNA")
    expected = assay.normed(
        cells,
        features[:3],
        renormalize_subset=parameters["renormalize_subset"],
    ).compute()
    np.testing.assert_allclose(
        _stored(store, stored), np.asarray(expected, dtype=np.float32), rtol=1e-6
    )


@pytest.mark.parametrize(
    ("assay_name", "normalizer"),
    [("RNA", norm_dummy), ("ADT", doubled)],
    ids=["rna-dummy", "adt-custom"],
)
def test_values_that_are_not_logarithms_are_logged_when_asked(
    store, monkeypatch, assay_name, normalizer
) -> None:
    """``norm_dummy`` and custom ADT values take ``log_transform`` but no defaults."""
    assay = store.get_assay(assay_name)
    monkeypatch.setattr(assay, "normMethod", normalizer)
    cells = np.arange(N_CELLS)
    features = np.arange(assay.feats.N)
    unlogged = np.asarray(assay.normed(cells, features).compute(), dtype=np.float64)
    expected = np.log1p(unlogged)

    logged = assay.normed(cells, features, log_transform=True).compute()
    assert logged.dtype == np.float64
    np.testing.assert_array_equal(logged, expected)
    streamed = np.hstack(
        [
            np.asarray(frame, dtype=np.float64)
            for frame in assay.iter_normed_feature_wise(
                cells, features, None, None, log_transform=True
            )
        ]
    )
    np.testing.assert_allclose(streamed, expected, rtol=1e-12)

    cell_selection, feature_ref = _selections(store, assay_name, features[:3])
    default = store.run_normalization(cell_selection, feature_ref)
    assert store.inspect_artifact(default).parameters["log_transform"] is False
    np.testing.assert_array_equal(
        _stored(store, default), unlogged[:, :3].astype(np.float32)
    )
    stored = store.run_normalization(cell_selection, feature_ref, log_transform=True)
    assert store.inspect_artifact(stored).parameters["log_transform"] is True
    np.testing.assert_array_equal(
        _stored(store, stored), expected[:, :3].astype(np.float32)
    )

    store.cells.insert("flag_groups", GROUPS, overwrite=True)
    clusters = store.snapshot_cluster_labels(
        "flag_groups", cell_selection=cell_selection
    )
    markers = store.run_marker_search(
        clusters, from_assay=assay_name, features=feature_ref, log_transform=True
    )
    status = store.inspect_artifact(markers)
    assert status.parameters["normalization"]["log_transform"] is True
    assert status.revision == 3
    scores = assay.score_features([f"{assay_name}0"], "I", 2, 2, 7, log_transform=True)
    controls = binned_sampling(pd.Series(expected.mean(axis=0)), [0], 2, 2, 7)
    np.testing.assert_allclose(
        scores, expected[:, 0] - expected[:, controls].mean(axis=1), rtol=1e-12
    )
    with pytest.raises(ValueError, match=_SUBSET_ERROR):
        assay.normed(cells, features[:2], renormalize_subset=True)


@pytest.mark.parametrize("assay_name", ["RNA", "ADT", "ATAC"])
def test_normalized_reads_reject_unknown_keywords_and_non_boolean_flags(
    store, assay_name
) -> None:
    assay = store.get_assay(assay_name)
    cells = np.arange(N_CELLS)
    features = np.arange(assay.feats.N)
    for options, message in (
        ({"log1p": True}, "unexpected keyword argument 'log1p'"),
        ({"log_transform": 1}, "log_transform must be a boolean"),
    ):
        with pytest.raises(TypeError, match=message):
            assay.normed(cells, features, **options)
        with pytest.raises(TypeError, match=message):
            next(assay.iter_normed_feature_wise(cells, features, None, None, **options))
        with pytest.raises(TypeError, match=message):
            find_markers_by_rank(assay, GROUPS, cells, features, **options)


def test_marker_heatmap_defaults_follow_the_normalizer(store, monkeypatch) -> None:
    assert _marker_log_transform(store.RNA, None) is True
    monkeypatch.setattr(store.RNA, "normMethod", norm_dummy)
    assert _marker_log_transform(store.RNA, None) is False
    assert _marker_log_transform(store.RNA, True) is True
    monkeypatch.setattr(store.RNA, "normMethod", scaled_by_totals)
    assert _marker_log_transform(store.RNA, None) is False
    assert _marker_log_transform(store.RNA, True) is True
    assert _marker_log_transform(store.ADT, None) is False
    assert _marker_log_transform(store.ATAC, None) is False
    with pytest.raises(ValueError, match=_LOG_ERROR):
        _marker_log_transform(store.ADT, True)


_LIBRARY_SIZE = callable_identity(norm_lib_size)
_CUSTOM = callable_identity(scaled_by_totals)


def _normalization_record(method, size_factor, log_transform, renormalize_subset):
    return {
        "normalization_method": {"external_hook": True, **method},
        "size_factor": size_factor,
        "log_transform": log_transform,
        "renormalize_subset": renormalize_subset,
    }


@pytest.mark.parametrize(
    ("parameters", "revision"),
    [
        (_normalization_record(_LIBRARY_SIZE, 1000.0, True, True), 1),
        (_normalization_record(_CUSTOM, 1000.0, True, False), 2),
        # A custom normalizer named norm_lib_size is another normalizer.
        (
            _normalization_record(
                {"module": "lab.normalizers", "qualname": "norm_lib_size"},
                1000.0,
                True,
                False,
            ),
            2,
        ),
        (_normalization_record(_CUSTOM, 1000.0, False, False), 1),
        (
            _normalization_record(
                callable_identity(norm_lib_size_log), 1000.0, False, True
            ),
            2,
        ),
        # ADT, ATAC, and generic assays record no size factor, so only a
        # recorded log_transform supersedes their records.
        (_normalization_record(callable_identity(norm_clr), None, True, True), 2),
        (_normalization_record(callable_identity(norm_tf_idf), None, False, True), 1),
        ({}, 1),
    ],
)
def test_normalization_revision_applies_to_flags_of_other_normalizers(
    parameters, revision
) -> None:
    assert (
        effective_revision("run_normalization", "normalized", parameters, {})
        == revision
    )


def _feature_record(method, log_transform, renormalize_subset=False):
    return {
        "normalization": {
            "log_transform": log_transform,
            "renormalize_subset": renormalize_subset,
        },
        "normalization_method": method,
        "size_factor": 1000,
    }


# Marker search revision 2 (fold change) applies to every marker table, so its
# normalizer revision is 3; the pseudotime analyses have only the scoped one.
@pytest.mark.parametrize(
    ("operation", "kind", "base", "scoped"),
    [
        ("run_marker_search", "marker_table", 2, 3),
        ("run_pseudotime_marker_search", "pseudotime_markers", 1, 2),
        ("run_pseudotime_aggregation", "pseudotime_aggregation", 1, 2),
    ],
)
def test_feature_revisions_apply_to_logged_records_of_other_normalizers(
    operation, kind, base, scoped
) -> None:
    def revision(parameters: dict[str, Any]) -> int:
        return effective_revision(operation, kind, parameters, {})

    assert revision(_feature_record(_LIBRARY_SIZE, True)) == base
    assert revision(_feature_record(_LIBRARY_SIZE, False, True)) == base
    assert revision(_feature_record(_CUSTOM, False, True)) == base
    assert revision(_feature_record(callable_identity(norm_clr), False)) == base
    assert revision(_feature_record(_CUSTOM, True)) == scoped
    assert revision(_feature_record(callable_identity(norm_dummy), True)) == scoped
    assert revision(_feature_record({"identity": "custom:v1"}, True, True)) == scoped
    assert revision({}) == base
    assert revision({"normalization": None, "normalization_method": None}) == base


@pytest.mark.slow
def test_saved_results_record_the_scoped_revisions(store, monkeypatch) -> None:
    cell_selection, features = _selections(store, "RNA", SUBSET)
    store.cells.insert("flag_groups", GROUPS, overwrite=True)
    clusters = store.snapshot_cluster_labels(
        "flag_groups", cell_selection=cell_selection
    )
    library_size = store.run_normalization(cell_selection, features)
    library_markers = store.run_marker_search(
        clusters, features=features, log_transform=True
    )
    monkeypatch.setattr(store.RNA, "normMethod", scaled_by_totals)
    custom = store.run_normalization(
        cell_selection, features, log_transform=True, renormalize_subset=True
    )
    unlogged = store.run_normalization(cell_selection, features)
    custom_markers = store.run_marker_search(
        clusters, features=features, log_transform=True
    )
    unlogged_markers = store.run_marker_search(clusters, features=features)

    revisions = {
        name: store.inspect_artifact(ref).revision
        for name, ref in {
            "library_size": library_size,
            "library_markers": library_markers,
            "custom": custom,
            "unlogged": unlogged,
            "custom_markers": custom_markers,
            "unlogged_markers": unlogged_markers,
        }.items()
    }
    assert revisions == {
        "library_size": 1,
        "library_markers": 2,
        "custom": 2,
        "unlogged": 1,
        "custom_markers": 3,
        "unlogged_markers": 2,
    }

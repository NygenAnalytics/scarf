"""Assay normalization, feature-summary, and scoring paths on small stores.

The stores hold RNA counts with ``countsT`` beside ADT and ATAC counts
without it, so each test reaches the generic ``Assay`` paths, the RNA
library-size paths, or the ATAC TF-IDF paths through the operation a
caller uses.
"""

import shutil
import warnings

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.assay.normalization import _normalize_rows, library_size_values
from scarf.datastore.datastore import DataStore
from scarf.storage.artifacts import ArtifactRef, artifact_path
from scarf.storage.budget import ResourceBudget
from scarf.storage.io_policy import StorageIoPolicy
from scarf.storage.layout import normed_array_spec
from scarf.storage.profiles import resolve_storage_profile
from scarf.storage.sharding import plan_dense_write
from scarf.utils import configure_output, logger
from scarf.writers import write_renorm_subset_to_zarr
from tests.storage_helpers import write_count_store

SIZE_FACTOR = 1000.0


def _counts() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(29)
    return {
        "RNA": rng.poisson(2.0, size=(16, 8)),
        "ADT": rng.poisson(6.0, size=(16, 4)) + 1,
        "ATAC": rng.poisson(0.6, size=(16, 10)),
    }


COUNTS = _counts()


def _open(zarr_loc: str) -> DataStore:
    return DataStore(
        zarr_loc,
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=1,
    )


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> DataStore:
    zarr_loc = str(tmp_path_factory.mktemp("assay_paths") / "store.zarr")
    write_count_store(zarr_loc, COUNTS, "uint16")
    opened = _open(zarr_loc)
    opened.cells.insert("nobody", np.zeros(opened.cells.N, dtype=bool), overwrite=True)
    return opened


@pytest.mark.parametrize("dtype", ["bool", "uint8", "int16", "float32", "float64"])
@pytest.mark.parametrize("log_transform", [False, True])
def test_row_normalization_kernel_rounds_float64_library_sizes_once(
    dtype: str,
    log_transform: bool,
) -> None:
    counts = np.random.default_rng(5).integers(0, 4, size=(6, 5))
    counts[2] = 0
    block = counts.astype(dtype)
    totals = block.sum(axis=1, dtype=np.float64)
    totals[totals == 0] = 1
    expected = library_size_values(
        block,
        totals,
        SIZE_FACTOR,
        dtype=np.float64,
        log_transform=log_transform,
    )
    # Logarithms are compared after the float32 rounding the payloads store;
    # a scaled count is one product and one quotient, so float64 is exact.
    out_dtype = np.float32 if log_transform else np.float64
    out = np.full(block.shape, np.nan, dtype=out_dtype)

    _normalize_rows.py_func(block, totals, SIZE_FACTOR, log_transform, out)

    np.testing.assert_array_equal(out, expected.astype(out_dtype))
    np.testing.assert_array_equal(out[block == 0], 0.0)
    np.testing.assert_array_equal(out[2], 0.0)


def test_renormalized_subset_progress_bar_shows_the_given_message(
    store: DataStore,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rna = store.RNA
    features = np.array([0, 2, 5])
    destination = zarr.open_group(store=MemoryStore(), mode="w")

    configure_output(progress=True)
    try:
        write_renorm_subset_to_zarr(
            rna,
            np.arange(rna.cells.N),
            features,
            destination,
            "normalized",
            1,
            msg="Renormalizing marker genes",
        )
    finally:
        configure_output(progress=False)

    assert "Renormalizing marker genes" in capsys.readouterr().err
    raw = COUNTS["RNA"][:, features].astype(np.float64)
    totals = raw.sum(axis=1)
    totals[totals == 0] = 1
    np.testing.assert_array_equal(
        destination["normalized"][:],
        (rna.sf * raw / totals[:, None]).astype(np.float32),
    )


def test_renormalized_subset_rejects_a_budget_that_only_fits_its_writer(
    store: DataStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rna = store.RNA
    cells = np.arange(rna.cells.N)
    features = np.array([0, 2, 5])
    destination = zarr.open_group(store=MemoryStore(), mode="w")
    # A budget equal to what one serial writer of the output reserves leaves
    # the feature-major producer no memory at all.
    writer = plan_dense_write(
        normed_array_spec(
            len(cells),
            len(features),
            profile=resolve_storage_profile(destination.store),
        ),
        ResourceBudget(1024**3, 1),
        1,
        io=StorageIoPolicy(readWorkers=1, computeWorkers=1, writeWorkers=1),
    )
    monkeypatch.setattr(rna, "resources", ResourceBudget(writer.reservedBytes, 1))

    with pytest.raises(MemoryError, match="both its producer and writer"):
        write_renorm_subset_to_zarr(rna, cells, features, destination, "normalized", 1)


def test_reduction_rejects_normalized_data_copied_from_another_dataset(
    tmp_path,
) -> None:
    source_counts = COUNTS["RNA"]
    target_counts = source_counts.copy()
    target_counts[0, 0] += 3
    write_count_store(str(tmp_path / "source.zarr"), {"RNA": source_counts}, "uint16")
    write_count_store(str(tmp_path / "target.zarr"), {"RNA": target_counts}, "uint16")
    source = _open(str(tmp_path / "source.zarr"))
    target = _open(str(tmp_path / "target.zarr"))

    def normalize(dataset: DataStore):
        return dataset.run_normalization(
            dataset.snapshot_cell_selection(),
            dataset.select_all_features(from_assay="RNA"),
        )

    copied = normalize(source)
    normalize(target)
    path = artifact_path(copied)
    shutil.copytree(tmp_path / "source.zarr" / path, tmp_path / "target.zarr" / path)

    assert target.inspect_artifact(copied).complete
    with pytest.raises(ValueError, match="does not match the current prepared dataset"):
        target.run_pca(copied, dims=2)


def test_feature_summaries_reject_assays_without_summary_statistics(
    store: DataStore,
) -> None:
    with pytest.raises(TypeError, match="received ADTassay"):
        store.select_detected_features(
            store.snapshot_cell_selection(),
            from_assay="ADT",
            min_cells=1,
        )


def _first_two_features(_assay, counts):
    """A user normalizer that wrongly drops features."""
    return counts[:, :2]


def test_feature_summary_rejects_a_normalizer_that_drops_features(
    store: DataStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(store.ATAC, "normMethod", _first_two_features)
    summaries = set(store.list_artifacts(kind="feature_summary", from_assay="ATAC"))

    with pytest.raises(ValueError, match=r"has shape \(2,\); expected \(10,\)"):
        store.select_detected_features(
            store.snapshot_cell_selection(),
            from_assay="ATAC",
            min_cells=1,
        )
    assert (
        set(store.list_artifacts(kind="feature_summary", from_assay="ATAC"))
        == summaries
    )


def test_feature_summary_with_an_unreadable_payload_is_recomputed(tmp_path) -> None:
    zarr_loc = tmp_path / "store.zarr"
    write_count_store(str(zarr_loc), {"RNA": COUNTS["RNA"]}, "uint16")
    dataset = _open(str(zarr_loc))
    cells = dataset.snapshot_cell_selection()

    def detected_summary() -> ArtifactRef:
        detected = dataset.select_detected_features(cells, min_cells=1)
        return dataset.inspect_artifact(detected).input_ref("feature_summary")

    first = detected_summary()
    totals = dataset.zw[artifact_path(first)]["normed_tot"]
    # A chunk copied from a shorter array with the same codecs decodes but
    # cannot fill the chunk it replaces, so its payload cannot be read. Such a
    # summary must be recomputed, neither reused nor failing the operation.
    short = zarr.create_array(
        str(tmp_path / "short.zarr"),
        shape=(3,),
        chunks=(3,),
        dtype=totals.dtype,
        compressors=totals.compressors,
        filters=totals.filters,
        serializer=totals.serializer,
    )
    short[:] = 1.0
    shutil.copyfile(
        tmp_path / "short.zarr" / "c" / "0",
        zarr_loc / artifact_path(first) / "normed_tot" / "c" / "0",
    )

    second = detected_summary()

    assert second != first
    np.testing.assert_array_equal(
        dataset.zw[artifact_path(second)]["normed_n"][:],
        np.count_nonzero(COUNTS["RNA"][dataset.cells.fetch_all("I")], axis=0),
    )


def test_feature_scores_reject_a_normalizer_that_drops_features(
    store: DataStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(store.ADT, "normMethod", _first_two_features)

    with pytest.raises(ValueError, match=r"feature_avg must have shape \(4,\)"):
        store.ADT.score_features(["ADT0"], "I", 1, 2, 0)


def test_peak_summary_of_an_empty_cell_selection_detects_no_peak(
    store: DataStore,
) -> None:
    empty = store.snapshot_cell_selection("nobody")

    with pytest.raises(ValueError, match="contains no features"):
        store.select_detected_features(empty, from_assay="ATAC", min_cells=1)
    every_peak = store.select_detected_features(empty, from_assay="ATAC", min_cells=0)

    np.testing.assert_array_equal(
        store.load_artifact(every_peak)["values"][:],
        np.ones(store.ATAC.feats.N, dtype=bool),
    )


@pytest.mark.parametrize(
    ("cell_idx", "feat_idx"),
    [
        (np.zeros((2, 2), dtype=np.int64), np.arange(3)),
        (np.arange(2), np.zeros((1, 3), dtype=np.int64)),
    ],
)
def test_feature_wise_normalized_batches_require_one_dimensional_indices(
    store: DataStore,
    cell_idx: np.ndarray,
    feat_idx: np.ndarray,
) -> None:
    batches = store.ADT.iter_normed_feature_wise(cell_idx, feat_idx, None, None)

    with pytest.raises(ValueError, match="must be one-dimensional"):
        next(batches)


@pytest.mark.parametrize(
    ("assay_name", "features"),
    [("ADT", ["ADT0"]), ("ATAC", ["ATAC0", "ATAC3"])],
)
def test_scores_of_an_empty_cell_selection_are_empty(
    store: DataStore,
    assay_name: str,
    features: list[str],
) -> None:
    assay = store.get_assay(assay_name)
    # CLR and TF-IDF are fitted on the selected cells. CLR would divide by the
    # cell count, so an empty selection must not be normalized at all.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        scores = assay.score_features(features, "nobody", 2, 2, 0)

    assert scores.shape == (0,)
    assert scores.dtype == np.float64
    assert assay.score_features(features, "I", 2, 2, 0).shape == (
        int(store.cells.fetch_all("I").sum()),
    )


def test_aggregated_ordering_caps_window_and_bins_at_the_cell_count(
    store: DataStore,
) -> None:
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]),
        level="WARNING",
    )
    try:
        prepared = store.ATAC._prepare_aggregated_ordering(
            np.arange(5),
            np.arange(3),
            np.linspace(0.0, 1.0, 5),
            window_size=50,
            chunk_size=20,
        )
        unchanged = store.ATAC._prepare_aggregated_ordering(
            np.arange(5),
            np.arange(3),
            np.linspace(0.0, 1.0, 5),
            window_size=3,
            chunk_size=4,
        )
    finally:
        logger.remove(sink)

    assert prepared[3:5] == (5, 5)
    assert unchanged[3:5] == (3, 4)
    assert messages == [
        "Reducing window_size from 50 to 5 for the selected cell count",
        "Reducing chunk_size from 20 to 5 for the selected cell count",
    ]

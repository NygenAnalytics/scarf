import inspect
from types import SimpleNamespace
from typing import get_type_hints

import numpy as np
import pytest
import zarr
from scipy.sparse import csr_matrix
from zarr.storage import MemoryStore

import scarf.assay as assay_module
from scarf.assay import (
    ADTassay,
    ATACassay,
    Assay,
    RNAassay,
    lib_size_feature_stream_eligible,
    norm_clr,
    norm_dummy,
    norm_lib_size,
    norm_tf_idf,
)
from scarf.assay.base import raw_csr
from scarf.storage.artifacts import ArtifactRef, artifact_group
from scarf.matrix import ChunkedArray
from scarf.datastore.datastore import DataStore
from scarf.writers import SparseToZarr
from tests.signature_contracts import signature_digest


_PUBLIC_CLASS_METHODS = {
    Assay: (
        "__init__",
        "normed",
        "iter_normed_feature_wise",
        "score_features",
        "__repr__",
    ),
    RNAassay: (
        "__init__",
        "iter_normed_feature_wise",
        "normed",
    ),
    ATACassay: (
        "__init__",
        "normed",
    ),
    ADTassay: ("__init__",),
}
_PUBLIC_CLASS_SIGNATURE_DIGESTS = {
    Assay: "045ac1edc448b0ef88037663d347adbfe34d70d32d281ea974179ce23f5928f7",
    RNAassay: "74fc5e54bc871c516fa8adca9cd8bcdec92bbcba94c118965b933159b2ef19ac",
    ATACassay: "1732f9ac8b4f368185e0965becb94b9472186db4ee53d372a8a2852d30dd42a4",
    ADTassay: "393df3ce24ede0e7affeba88c1970e1a77f0fac31adc07276be28dd40f1c1c4a",
}
_MODULE_FUNCTIONS = (
    "lib_size_feature_stream_eligible",
    "norm_clr",
    "norm_dummy",
    "norm_lib_size",
    "norm_lib_size_log",
    "norm_tf_idf",
)
_MODULE_SIGNATURE_DIGEST = (
    "3e073d20c4cb115fc4e8aed99fa62687509321692419598072535a3ec67f9223"
)


def test_assay_facade_surface_is_stable():
    assert not hasattr(Assay, "add_percent_feature")
    assert not hasattr(Assay, "save_aggregated_ordering")
    assert not hasattr(Assay, "save_normalized_data")
    assert not hasattr(RNAassay, "save_normalized_data")
    assert assay_module.__all__ == [
        "Assay",
        "RNAassay",
        "ATACassay",
        "ADTassay",
        "is_rna_assay_type",
        "lookup_persisted_assay_type",
        "preset_assay_types",
        "resolve_persisted_assay_type",
    ]
    expected = {
        "ADTassay",
        "ATACassay",
        "Assay",
        "NormMethod",
        "PercentFeatures",
        "RNAassay",
        "_read_block",
        "is_rna_assay_type",
        "lib_size_feature_stream_eligible",
        "lookup_persisted_assay_type",
        "norm_clr",
        "norm_dummy",
        "norm_lib_size",
        "norm_lib_size_log",
        "norm_tf_idf",
        "preset_assay_types",
        "resolve_persisted_assay_type",
    }
    assert expected.issubset(vars(assay_module))


def test_assay_class_and_method_signatures_are_stable():
    for cls, names in _PUBLIC_CLASS_METHODS.items():
        methods = {name: getattr(cls, name) for name in names}
        assert signature_digest(methods) == _PUBLIC_CLASS_SIGNATURE_DIGESTS[cls]


def test_assay_module_function_signatures_are_stable():
    methods = {name: getattr(assay_module, name) for name in _MODULE_FUNCTIONS}
    assert signature_digest(methods) == _MODULE_SIGNATURE_DIGEST


def test_assay_normalization_type_hints_resolve_from_facade():
    for name in (
        "lib_size_feature_stream_eligible",
        "norm_clr",
        "norm_dummy",
        "norm_lib_size",
        "norm_lib_size_log",
        "norm_tf_idf",
    ):
        assert get_type_hints(getattr(assay_module, name))


def test_assay_public_metadata_remains_on_facade():
    for cls, names in _PUBLIC_CLASS_METHODS.items():
        assert cls.__module__ == "scarf.assay"
        for name in names:
            descriptor = inspect.getattr_static(cls, name)
            if isinstance(descriptor, staticmethod):
                method = descriptor.__func__
            else:
                method = descriptor
            assert method.__module__ == "scarf.assay"
            assert method.__qualname__.startswith(f"{cls.__name__}.")

    for name in _MODULE_FUNCTIONS:
        assert getattr(assay_module, name).__module__ == "scarf.assay"


def test_assay_subclass_and_static_method_contracts_are_stable():
    assert issubclass(RNAassay, Assay)
    assert issubclass(ATACassay, Assay)
    assert issubclass(ADTassay, Assay)
    assert "normed" not in vars(ADTassay)


def test_default_normalizer_identity_is_stable(monkeypatch):
    def initialize_base(self, *args, **kwargs):
        self.z = zarr.open_group(store=MemoryStore(), mode="w")
        self.attrs = self.z.attrs

    monkeypatch.setattr(Assay, "__init__", initialize_base)
    rna = RNAassay(None, "RNA", None)
    atac = ATACassay(None, "ATAC", None)
    adt = ADTassay(None, "ADT", None)

    assert rna.normMethod is norm_lib_size
    assert lib_size_feature_stream_eligible(rna)
    # The RNA feature streams rely on eligibility to guarantee a size factor.
    rna.sf = None
    assert not lib_size_feature_stream_eligible(rna)
    rna.sf = 1000
    rna.normMethod = norm_dummy
    assert not lib_size_feature_stream_eligible(rna)
    assert atac.normMethod is norm_tf_idf
    assert adt.normMethod is norm_clr


def test_normalization_numerical_contracts_are_stable():
    counts = np.array([[1.0, 3.0], [2.0, 4.0]])

    clr_expected_scale = np.exp(np.log1p(counts).sum(axis=0) / len(counts))
    np.testing.assert_allclose(
        norm_clr(None, counts),
        np.log1p(counts / clr_expected_scale),
    )

    atac = SimpleNamespace(
        n_term_per_doc=np.array([4.0, 6.0]),
        n_docs=2,
        n_docs_per_term=np.array([2.0, 1.0]),
    )
    tf = counts / atac.n_term_per_doc.reshape(-1, 1)
    idf = np.log2(1 + atac.n_docs / (atac.n_docs_per_term + 1))
    np.testing.assert_allclose(norm_tf_idf(atac, counts), tf * idf)

    rna = SimpleNamespace(
        sf=1000.0,
        scalar=np.array([4.0, 6.0]),
    )
    np.testing.assert_allclose(
        norm_lib_size(rna, counts),
        1000.0 * counts / rna.scalar.reshape(-1, 1),
    )


def test_clr_normalizes_features_when_chunked_selection_is_square():
    values = np.array([[1.0, 4.0, 9.0], [2.0, 20.0, 3.0], [12.0, 2.0, 2.0]])
    counts = ChunkedArray.from_numpy(values, block_size=2)
    expected_scale = np.exp(np.log1p(values).sum(axis=0) / len(values))

    actual = norm_clr(None, counts).compute()

    np.testing.assert_allclose(actual, np.log1p(values / expected_scale[None, :]))


def test_normalization_kernels_share_arrays_and_chunked_arrays():
    from scarf.assay.normalization import (
        clr_values,
        inverse_document_frequency,
        library_size_values,
        stream_document_frequency,
        term_frequencies,
        tfidf_values,
    )

    counts = np.array(
        [[1, 0, 9], [2, 20, 0], [12, 2, 2], [0, 0, 5]],
        dtype=np.uint16,
    )
    chunked = ChunkedArray.from_numpy(counts, block_size=3)
    totals = np.array([10.0, 22.0, 16.0, 5.0])
    idf = inverse_document_frequency(4, np.count_nonzero(counts, axis=0))

    np.testing.assert_array_equal(idf, np.log2(1 + 4 / np.array([4.0, 3.0, 4.0])))
    np.testing.assert_allclose(
        clr_values(chunked).compute(), clr_values(counts), rtol=1e-12
    )
    np.testing.assert_array_equal(
        tfidf_values(chunked, totals, idf).compute(),
        tfidf_values(counts, totals, idf),
    )
    np.testing.assert_array_equal(
        tfidf_values(counts, totals, idf),
        term_frequencies(counts, totals) * idf,
    )
    np.testing.assert_allclose(
        library_size_values(counts, totals, 1000, dtype=np.float64, log_transform=True),
        np.log1p(1000 * counts / totals[:, None]),
    )

    selected = np.array([True, False, True, True])
    frequency, term_sums = stream_document_frequency(
        ChunkedArray.from_numpy(counts, block_size=1),
        memory_bytes=1024**2,
        nthreads=1,
        msg="",
        operation="Test frequency",
        row_mask=selected,
        term_totals=totals[selected],
    )
    np.testing.assert_array_equal(frequency, np.count_nonzero(counts[selected], axis=0))
    np.testing.assert_allclose(
        term_sums, (counts[selected] / totals[selected, None]).sum(axis=0)
    )
    with pytest.raises(MemoryError, match="Test frequency needs about"):
        stream_document_frequency(
            chunked,
            memory_bytes=8,
            nthreads=1,
            msg="",
            operation="Test frequency",
        )


def test_clr_does_not_reuse_artifacts_with_ambiguous_axis(monkeypatch):
    values = np.array([[1.0, 4.0, 9.0], [2.0, 20.0, 3.0], [12.0, 2.0, 2.0]])
    store = MemoryStore()
    SparseToZarr(
        csr_matrix(values),
        store,
        ["a", "b", "c"],
        ["x", "y", "z"],
        assay_name="ADT",
        nthreads=1,
    ).dump()
    datastore = DataStore(store, default_assay="ADT", nthreads=1)
    cells = datastore.snapshot_cell_selection()
    features = datastore.select_all_features(from_assay="ADT")
    scale = np.exp(np.log1p(values).sum(axis=0) / len(values))
    with monkeypatch.context() as previous_identity:
        previous_identity.delattr(norm_clr, "artifact_identity")
        previous = datastore.run_normalization(cells, features)
    artifact_group(datastore.zw, previous)["data"][:] = np.log1p(
        values / scale[:, None]
    )

    actual = datastore.run_normalization(cells, features)

    assert actual != previous
    np.testing.assert_allclose(
        artifact_group(datastore.zw, actual)["data"][:],
        np.log1p(values / scale[None, :]),
        rtol=1e-6,
    )
    assert datastore.run_normalization(cells, features) == actual


def test_assay_read_block_facade_remains_patchable(monkeypatch):
    from scarf.storage.budget import ResourceBudget

    root = zarr.open_group(store=MemoryStore(), mode="w")
    values = np.arange(1, 13, dtype=np.uint32).reshape(3, 4)
    counts = root.create_array("counts", data=values)
    totals = values.sum(axis=1).astype(np.float64)
    rna = RNAassay.__new__(RNAassay)
    rna.name = "RNA"
    rna.normMethod = norm_lib_size
    rna.sf = 10
    rna.cells = SimpleNamespace(fetch_all=lambda _column: totals)
    rna.rawData = SimpleNamespace(_backing=counts)
    rna.resources = ResourceBudget(memoryBytes=1024**2, workers=1)

    original = assay_module._read_block
    calls = []

    def counted_read(array, rows, columns):
        calls.append((rows.copy(), columns.copy()))
        return original(array, rows, columns)

    monkeypatch.setattr(assay_module, "_read_block", counted_read)
    means = rna._mean_normed_feature_groups(
        np.array([0, 2]),
        {"pair": np.array([1, 3])},
    )

    assert len(calls) == 1
    expected = (10 * values[[0, 2]][:, [1, 3]] / totals[[0, 2], None]).mean(axis=1)
    np.testing.assert_allclose(means["pair"], expected)


def test_base_assay_defaults_validation_and_representation():
    cells = SimpleNamespace(
        active_index=lambda _key: np.array([0, 1]),
        columns=["I"],
        get_dtype=lambda _key: bool,
    )
    feats = SimpleNamespace(
        N=3,
        active_index=lambda _key: np.array([0, 2]),
        columns=["I"],
        get_dtype=lambda _key: bool,
        fetch_all=lambda _key: np.array([True, False, True]),
    )
    root = zarr.open_group(store=MemoryStore(), mode="w")
    root.attrs.update({"prepared": True, "dataset_fingerprint": "test-dataset"})
    assay = SimpleNamespace(
        z=root,
        attrs={"percentFeatures": "invalid"},
        cells=cells,
        feats=feats,
        rawData=np.arange(6).reshape(2, 3),
        normMethod=lambda _assay, counts: counts,
        name="toy",
    )

    assert Assay._percent_features(assay) == {}
    np.testing.assert_array_equal(
        Assay.normed(assay),
        np.array([[0, 1, 2], [3, 4, 5]]),
    )
    assert "toy with 3 features" in Assay.__repr__(assay)

    invalid_cells = SimpleNamespace(columns=[], get_dtype=lambda _key: bool)
    invalid = SimpleNamespace(cells=invalid_cells, feats=feats)
    with pytest.raises(ValueError, match="missing_cell"):
        Assay._get_cell_idx(invalid, "missing_cell")


def test_base_assay_sparse_export_combines_streamed_blocks():
    class StreamedRaw:
        def __getitem__(self, _selection):
            return self

        @staticmethod
        def stream_blocks(**_kwargs):
            yield np.array([[1, 0], [0, 2]])
            yield np.array([[3, 4]])

    assay = SimpleNamespace(rawData=StreamedRaw(), nthreads=1, name="toy")

    observed = raw_csr(assay, np.arange(3))
    np.testing.assert_array_equal(
        observed.toarray(),
        np.array([[1, 0], [0, 2], [3, 4]]),
    )


def test_feature_percentage_is_datastore_owned_and_artifact_only(
    datastore_ephemeral,
) -> None:
    store = datastore_ephemeral
    cells = store.snapshot_cell_selection()
    features = store.set_feature_selection(
        from_assay="RNA",
        feature_indexes=[0],
    )
    cell_columns = set(store.cells.columns)
    assay_attrs = dict(store.RNA.attrs)

    ref = store.run_feature_percentage(cells, features)

    assert isinstance(ref, ArtifactRef)
    assert ref.kind == "quality_metric"
    assert ref.scope == "assay"
    assert ref.assay == "RNA"
    status = store.inspect_artifact(ref)
    assert status.operation == "run_feature_percentage"
    assert status.parameters == {"scale": 100.0}
    assert ArtifactRef.from_dict(status.inputs["cell_selection"]) == cells
    assert ArtifactRef.from_dict(status.inputs["feature_selection"]) == features
    cell_index = store.cells.active_index("I")
    counts = store.RNA.rawData[cell_index, :].compute(nthreads=1)
    expected = np.divide(
        100.0 * counts[:, 0],
        counts.sum(axis=1),
        out=np.full(len(cell_index), np.nan),
        where=counts.sum(axis=1) != 0,
    )
    np.testing.assert_allclose(
        artifact_group(store.zw, ref)["values"][:],
        expected,
    )
    assert store.run_feature_percentage(cells, features) == ref
    assert not hasattr(store.RNA, "add_percent_feature")
    assert set(store.cells.columns) == cell_columns
    assert dict(store.RNA.attrs) == assay_attrs


def test_base_assay_score_features_covers_generic_normalization(
    monkeypatch,
):
    class DeferredMatrix:
        def __init__(self, feature_index):
            self.values = np.tile(np.asarray(feature_index), (2, 1))

        def mean(self, axis):
            return SimpleNamespace(compute=lambda: self.values.mean(axis=axis))

    feats = SimpleNamespace(
        N=3,
        get_index_by=lambda _values, _column, _key: np.array([], dtype=int),
        fetch_all=lambda _key: np.array([0.1, 0.2, 0.3]),
    )
    assay = SimpleNamespace(
        feats=feats,
        _get_cell_idx=lambda _cell_key: np.array([0, 1]),
        normed=lambda *, cell_idx, feat_idx: DeferredMatrix(feat_idx),
    )
    assay._score_feature_indices = lambda *args, **kwargs: Assay._score_feature_indices(
        assay, *args, **kwargs
    )

    with pytest.raises(ValueError, match="No feature ids found"):
        Assay.score_features(assay, ["Missing"], "I", 1, 2, 0)

    feats.get_index_by = lambda _values, _column, _key: np.array([1], dtype=int)
    monkeypatch.setattr(
        "scarf.features.scoring.binned_sampling",
        lambda *_args, **_kwargs: [2],
    )
    np.testing.assert_array_equal(
        Assay.score_features(assay, ["GeneB"], "I", 1, 2, 0),
        np.array([-1.0, -1.0]),
    )


def test_rna_gene_major_kernel_accumulates_selected_cells():
    from scarf.assay.rna import _hvg_stats_gene_major_kernel

    values = np.array(
        [
            [1, 0, 2],
            [0, 5, 0],
            [3, 4, 0],
        ],
        dtype=np.uint32,
    )
    destinations = np.array([0, -1, 1], dtype=np.int64)
    selected = np.array([0, 2], dtype=np.int64)
    inverse_scalars = np.array([0.5, 0.25])
    nonzero = np.zeros(2)
    totals = np.zeros(2)
    squares = np.zeros(2)

    _hvg_stats_gene_major_kernel.py_func(
        values,
        inverse_scalars,
        2.0,
        destinations,
        selected,
        nonzero,
        totals,
        squares,
    )

    np.testing.assert_array_equal(nonzero, np.array([2.0, 1.0]))
    np.testing.assert_allclose(totals, np.array([2.0, 3.0]))
    np.testing.assert_allclose(squares, np.array([2.0, 9.0]))


def test_rna_gene_major_kernel_log_transform_matches_log1p():
    from scarf.assay.rna import _hvg_stats_gene_major_kernel

    values = np.array(
        [
            [1, 0, 2],
            [0, 5, 0],
            [3, 4, 0],
        ],
        dtype=np.uint32,
    )
    destinations = np.array([0, -1, 1], dtype=np.int64)
    selected = np.array([0, 2], dtype=np.int64)
    inverse_scalars = np.array([0.5, 0.25])
    logged = np.log1p(2.0 * values[[0, 2]][:, selected] * inverse_scalars)

    for kernel in (
        _hvg_stats_gene_major_kernel.py_func,
        _hvg_stats_gene_major_kernel,
    ):
        nonzero = np.zeros(2)
        totals = np.zeros(2)
        squares = np.zeros(2)
        kernel(
            values,
            inverse_scalars,
            2.0,
            destinations,
            selected,
            nonzero,
            totals,
            squares,
            True,
        )

        np.testing.assert_array_equal(nonzero, np.array([2.0, 1.0]))
        np.testing.assert_allclose(totals, logged.sum(axis=1))
        np.testing.assert_allclose(squares, np.square(logged).sum(axis=1))


def test_rna_requires_zarr_v3_counts_t():
    from scarf.storage.counts_t_contract import validate_count_matrix

    root = zarr.open_group(store=MemoryStore(), mode="w", zarr_format=2)
    root.create_array("counts", data=np.ones((2, 3), dtype=np.uint32))
    root.create_array("countsT", data=np.ones((3, 2), dtype=np.uint32))
    with pytest.raises(ValueError, match="Zarr v3|not finalized"):
        validate_count_matrix(root, require_transpose=True)


def test_rna_normed_zero_total_cells_are_zero(tmp_path):
    raw = np.array([[3, 1, 0], [0, 0, 0], [2, 0, 2]], dtype=np.uint32)
    path = tmp_path / "rna.zarr"
    SparseToZarr(
        csr_matrix(raw),
        zarr_loc=str(path),
        cell_ids=["c0", "c1", "c2"],
        feature_ids=["g0", "g1", "g2"],
        assay_name="RNA",
        nthreads=1,
    ).dump(batch_size=3)
    store = DataStore(str(path), default_assay="RNA", min_features_per_cell=0)
    cells = np.arange(3)
    expressed = raw[[0, 2]]
    lib_size = store.RNA.sf * expressed / expressed.sum(axis=1, keepdims=True)

    for log_transform in (False, True):
        values = store.RNA.normed(
            cell_idx=cells,
            feat_idx=np.arange(3),
            log_transform=log_transform,
        ).compute()
        assert np.isfinite(values).all()
        np.testing.assert_array_equal(values[1], 0.0)
        np.testing.assert_allclose(
            values[[0, 2]],
            np.log1p(lib_size) if log_transform else lib_size,
        )

    store.cells.insert("everyone", np.ones(3, dtype=bool), overwrite=True)
    np.testing.assert_allclose(
        store.get_cell_vals(from_assay="RNA", cell_key="everyone", k="g0"),
        [store.RNA.sf * 3 / 4, 0.0, store.RNA.sf * 2 / 4],
    )


def _zero_count_store(tmp_path) -> DataStore:
    raw = np.array([[3, 1, 0, 2], [0, 0, 0, 0], [1, 1, 4, 0]], dtype=np.uint32)
    path = tmp_path / "zero.zarr"
    SparseToZarr(
        csr_matrix(raw),
        zarr_loc=str(path),
        cell_ids=["c0", "c1", "c2"],
        feature_ids=["g0", "g1", "g2", "g3"],
        feature_names=["MT-A", "MT-B", "G1", "G2"],
        assay_name="RNA",
        nthreads=1,
    ).dump(batch_size=3)
    return DataStore(str(path), default_assay="RNA", min_features_per_cell=0)


def test_feature_percentages_share_one_definition_for_zero_count_cells(tmp_path):
    store = _zero_count_store(tmp_path)
    expected = [100 * 4 / 6, np.nan, 100 * 2 / 6]

    np.testing.assert_allclose(store.cells.fetch_all("RNA_percentMito"), expected)
    np.testing.assert_allclose(
        store.RNA._compute_feature_percentage(np.arange(3), np.array([0, 1])),
        expected,
    )
    store.cells.insert("everyone", np.ones(3, dtype=bool), overwrite=True)
    ref = store.run_feature_percentage(
        store.snapshot_cell_selection("everyone"),
        store.set_feature_selection(from_assay="RNA", feature_indexes=[0, 1]),
    )
    np.testing.assert_allclose(store.load_artifact(ref)["values"][:], expected)


def test_concurrent_rna_normed_calls_keep_their_own_normalization(
    tmp_path, monkeypatch
):
    import threading
    import time

    import scarf.assay.normalization as normalization

    raw = np.random.default_rng(3).integers(0, 9, (6, 4)).astype(np.uint32)
    raw[:, 0] += 1
    SparseToZarr(
        csr_matrix(raw),
        zarr_loc=str(tmp_path / "rna.zarr"),
        cell_ids=[f"c{i}" for i in range(6)],
        feature_ids=[f"g{i}" for i in range(4)],
        assay_name="RNA",
        nthreads=1,
    ).dump(batch_size=6)
    rna = DataStore(
        str(tmp_path / "rna.zarr"), default_assay="RNA", min_features_per_cell=0
    ).RNA
    feats = np.arange(4)
    requests = {"first": (np.arange(6), True), "second": (np.array([1, 3, 5]), False)}
    expected = {
        name: rna.normed(cells, feats, log_transform=log).compute()
        for name, (cells, log) in requests.items()
    }

    # The first call reads its totals slowly, and every call builds its result
    # slowly, so an unguarded call would read the other call's method or totals.
    def delayed(function, seconds, thread=None):
        def call(*args, **kwargs):
            if thread in (None, threading.current_thread().name):
                time.sleep(seconds)
            return function(*args, **kwargs)

        return call

    monkeypatch.setattr(
        RNAassay,
        "_cell_count_totals",
        delayed(RNAassay._cell_count_totals, 0.2, "first"),
    )
    monkeypatch.setattr(
        normalization,
        "_library_size_scaled",
        delayed(normalization._library_size_scaled, 0.3),
    )
    results = {}

    def normalize(name):
        cells, log = requests[name]
        results[name] = rna.normed(cells, feats, log_transform=log).compute()

    threads = [
        threading.Thread(target=normalize, args=(name,), name=name) for name in requests
    ]
    for thread in threads:
        thread.start()
        time.sleep(0.05)
    for thread in threads:
        thread.join()

    for name in requests:
        np.testing.assert_array_equal(results[name], expected[name])
    assert rna.normMethod is norm_lib_size
    assert rna.scalar is None


def test_rna_streaming_stats_and_group_means_handle_missing_inputs():
    from scarf.metadata import MetaData

    root = zarr.open_group(store=MemoryStore(), mode="w")
    counts = root.create_array(
        "counts",
        data=np.array([[1, 0], [0, 1]], dtype=np.uint32),
    )
    rna = RNAassay.__new__(RNAassay)
    rna.name = "RNA"
    rna.normMethod = norm_lib_size
    rna.sf = None
    cell_data = root.create_group("cellData")
    cell_data.create_array("RNA_nCounts", data=np.array([2.0, 3.0]))
    rna.cells = MetaData(cell_data)
    rna.rawData = SimpleNamespace(_backing=counts)

    with pytest.raises(ValueError, match="size factor"):
        rna._mean_normed_feature_groups(
            np.array([0, 1]),
            {"target": np.array([0])},
        )

    rna.sf = 1000
    empty_means = rna._mean_normed_feature_groups(
        np.array([], dtype=int),
        {"target": np.array([0])},
    )
    assert empty_means["target"].shape == (0,)

    empty_stats = rna._streaming_feature_stats(
        np.array([], dtype=int),
        np.array([0], dtype=int),
    )
    np.testing.assert_array_equal(empty_stats["normed_n"], np.zeros(1))

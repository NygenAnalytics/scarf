import inspect
from types import SimpleNamespace
from typing import get_type_hints

import numpy as np
import pytest
import zarr
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
from tests.signature_contracts import signature_digest


_PUBLIC_CLASS_METHODS = {
    Assay: (
        "__init__",
        "normed",
        "to_raw_sparse",
        "add_percent_feature",
        "save_normalized_data",
        "iter_normed_feature_wise",
        "save_aggregated_ordering",
        "mean_features",
        "score_features",
        "__repr__",
    ),
    RNAassay: (
        "__init__",
        "iter_normed_feature_wise",
        "save_normalized_data",
        "normed",
        "iter_raw_column_blocks",
        "iter_raw_feature_columns",
        "set_feature_stats",
        "get_feature_stats",
        "set_summary_stats",
        "set_hvgs",
        "mark_hvgs",
    ),
    ATACassay: (
        "__init__",
        "normed",
        "set_feature_stats",
        "mark_prevalent_peaks",
    ),
    ADTassay: (
        "__init__",
        "normed",
    ),
}
_PUBLIC_CLASS_SIGNATURE_DIGESTS = {
    Assay: "fb2d878f78b4b561da78a1fad09efaf34f2e3c6cc47cf359021816a1adbf5ddf",
    RNAassay: "26951c69d48c7cb169d8d59014850ff1cf47cdb7dc4cff63e3daf8851f74c604",
    ATACassay: "491bb1c63ad83fa5d9634200c5b3778a3018e39abdd3ae87208eb3e85659633c",
    ADTassay: "a1ff1bebdd8fcd3f30a1b64a42dbf2931f6b8031ddce75abcf362750eb4e9c34",
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
    assert assay_module.__all__ == [
        "Assay",
        "RNAassay",
        "ATACassay",
        "ADTassay",
        "is_rna_assay_type",
        "lookup_persisted_assay_type",
        "preset_assay_types",
        "resolve_persisted_assay_type",
        "rna_assay_type_names",
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
        "rna_assay_type_names",
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
    for name in (
        "_create_subset_hash",
        "_finalize_staged_mirror",
        "_get_summary_stats_loc",
    ):
        assert isinstance(inspect.getattr_static(Assay, name), staticmethod)


def test_default_normalizer_identity_is_stable(monkeypatch):
    def initialize_base(self, *args, **kwargs):
        self.attrs = {}

    monkeypatch.setattr(Assay, "__init__", initialize_base)
    rna = RNAassay(None, "RNA", None)
    atac = ATACassay(None, "ATAC", None)
    adt = ADTassay(None, "ADT", None)

    assert rna.normMethod is norm_lib_size
    assert lib_size_feature_stream_eligible(rna)
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


def test_assay_read_block_facade_remains_patchable(monkeypatch):
    from scarf.storage.budget import ResourceBudget

    root = zarr.open_group(store=MemoryStore(), mode="w")
    counts_t = root.create_array(
        "countsT",
        data=np.arange(12, dtype=np.uint32).reshape(4, 3),
    )
    rna = RNAassay.__new__(RNAassay)
    rna.name = "RNA"
    rna.z = root
    rna.rawDataT = counts_t
    rna.resources = ResourceBudget(memoryBytes=1024**2, workers=1)

    original = assay_module._read_block
    calls = []

    def counted_read(array, rows, columns):
        calls.append((rows.copy(), columns.copy()))
        return original(array, rows, columns)

    monkeypatch.setattr(assay_module, "_read_block", counted_read)
    blocks = list(
        rna.iter_raw_column_blocks(
            cell_idx=np.array([0, 2]),
            feat_idx=np.array([1, 3]),
            batch_size=2,
        )
    )

    assert len(calls) == 1
    expected = np.asarray(counts_t[:])[[1, 3], :][:, [0, 2]].T
    np.testing.assert_array_equal(blocks[0][1], expected)


@pytest.mark.parametrize(
    ("include_adaptive", "expected_column", "expected_variance"),
    [
        (False, "c_var__200__0.1", np.array([40.0, 20.0])),
        (True, "c_var__adaptive__200__0.1", np.array([400.0, 200.0])),
    ],
)
def test_get_feature_stats_reads_cached_columns_in_feature_key_order(
    include_adaptive,
    expected_column,
    expected_variance,
):
    root = zarr.open_group(store=MemoryStore(), mode="w")
    stats = root.create_group("summary_stats_I")
    stats.create_array("nz_mean", data=np.array([1.0, 2.0, 3.0, 4.0]))
    stats.create_array(
        "c_var__200__0.1",
        data=np.array([10.0, 20.0, 30.0, 40.0]),
    )
    if include_adaptive:
        stats.create_array(
            "c_var__adaptive__200__0.1",
            data=np.array([100.0, 200.0, 300.0, 400.0]),
        )
    stats.create_array("normed_n", data=np.array([5.0, 6.0, 7.0, 8.0]))

    rna = RNAassay.__new__(RNAassay)
    rna.z = root
    rna.feats = SimpleNamespace(
        active_index=lambda key: np.array([3, 1]) if key == "selected" else None
    )
    rna._get_cell_feat_idx = lambda cell_key, feat_key: (
        np.array([0, 2]),
        np.arange(4),
    )
    validation_calls = []

    def validate(stats_loc, cell_idx, feat_idx, delete_on_fail=True):
        validation_calls.append((stats_loc, cell_idx, feat_idx, delete_on_fail))
        return True

    rna._validate_stats_loc = validate
    values = rna.get_feature_stats("I", feat_key="selected")

    assert list(values) == ["nz_mean", expected_column, "normed_n"]
    np.testing.assert_array_equal(values["nz_mean"], np.array([4.0, 2.0]))
    np.testing.assert_array_equal(values[expected_column], expected_variance)
    np.testing.assert_array_equal(values["normed_n"], np.array([8.0, 6.0]))
    assert validation_calls[0][3] is False


def test_get_feature_stats_does_not_recompute_or_delete_invalid_cache():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    root.create_group("summary_stats_subset")

    rna = RNAassay.__new__(RNAassay)
    rna.z = root
    rna.feats = SimpleNamespace(active_index=lambda key: np.arange(2))
    rna._get_cell_feat_idx = lambda cell_key, feat_key: (
        np.array([0]),
        np.arange(2),
    )
    rna._validate_stats_loc = (
        lambda stats_loc, cell_idx, feat_idx, delete_on_fail=True: False
    )
    rna.set_feature_stats = lambda cell_key: pytest.fail(
        "get_feature_stats must not compute statistics"
    )

    with pytest.raises(KeyError, match="have not been calculated"):
        rna.get_feature_stats("subset", columns=["nz_mean"])

    assert "summary_stats_subset" in root


def test_base_assay_defaults_validation_and_representation():
    cells = SimpleNamespace(
        active_index=lambda _key: np.array([0, 1]),
        columns=["I"],
        get_dtype=lambda _key: bool,
    )
    feats = SimpleNamespace(
        active_index=lambda _key: np.array([0, 2]),
        columns=["I"],
        get_dtype=lambda _key: bool,
        fetch_all=lambda _key: np.array([True, False, True]),
    )
    assay = SimpleNamespace(
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
        np.array([[0, 2], [3, 5]]),
    )
    assert "toy with 2(3) features" in Assay.__repr__(assay)

    invalid_cells = SimpleNamespace(columns=[], get_dtype=lambda _key: bool)
    invalid = SimpleNamespace(cells=invalid_cells, feats=feats)
    with pytest.raises(ValueError, match="missing_cell"):
        Assay._verify_keys(invalid, "missing_cell", "I")

    invalid.cells = cells
    invalid.feats = SimpleNamespace(columns=[], get_dtype=lambda _key: bool)
    with pytest.raises(ValueError, match="missing_feature"):
        Assay._verify_keys(invalid, "I", "missing_feature")


def test_base_assay_sparse_export_combines_streamed_blocks():
    class StreamedRaw:
        def __getitem__(self, _selection):
            return self

        @staticmethod
        def stream_blocks(**_kwargs):
            yield np.array([[1, 0], [0, 2]])
            yield np.array([[3, 4]])

    assay = SimpleNamespace(
        rawData=StreamedRaw(),
        cells=SimpleNamespace(active_index=lambda _key: np.arange(3)),
        nthreads=1,
        name="toy",
    )

    observed = Assay.to_raw_sparse(assay, "I")
    np.testing.assert_array_equal(
        observed.toarray(),
        np.array([[1, 0], [0, 2], [3, 4]]),
    )


def test_base_assay_percent_feature_helpers_cover_noop_and_write_paths():
    zero_assay = SimpleNamespace(cells=None)
    Assay._write_percent_feature(
        zero_assay,
        "percent_zero",
        np.zeros(2),
    )

    writes = []
    assay = SimpleNamespace(
        _plan_percent_feature=lambda _pattern, _name: None,
        rawData=np.array([[2, 1], [0, 3]]),
        nthreads=1,
        name="toy",
        _write_percent_feature=lambda name, values: writes.append((name, values)),
    )
    assert Assay.add_percent_feature(assay, "^missing$", "percent_missing") is None

    assay._plan_percent_feature = lambda _pattern, _name: np.array([0])
    Assay.add_percent_feature(assay, "^gene", "percent_gene")
    assert writes[0][0] == "percent_gene"
    np.testing.assert_array_equal(writes[0][1], np.array([2, 0]))


def test_base_assay_cache_and_staged_mirror_helpers():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    stale = root.create_group("summary_stats")
    stale.attrs["subset_hash"] = "stale"
    assay = SimpleNamespace(
        z=root,
        _create_subset_hash=lambda _cells, _features: "current",
    )

    assert not Assay._validate_stats_loc(
        assay,
        "summary_stats",
        np.array([0]),
        np.array([0]),
    )
    assert "summary_stats" not in root

    mirror = root.create_array("mirror", data=np.zeros((2, 2)))
    params = {"log_transform": True}
    Assay._finalize_staged_mirror(mirror, "digest", params)
    assert mirror.attrs["staged_subset_hash"] == "digest"
    assert mirror.attrs["staged_subset_params"] == params
    assert mirror.attrs["staged_complete"] is True


def test_base_assay_mean_features_validates_requests_and_generic_path():
    class DeferredMean:
        @staticmethod
        def mean(axis):
            assert axis == 1
            return DeferredMean()

        @staticmethod
        def compute():
            return np.array([1.5, 2.5])

    feature_names = np.array(["GeneA", "GeneB", "GeneC"])
    feats = SimpleNamespace(fetch_all=lambda _key: feature_names)
    assay = SimpleNamespace(
        feats=feats,
        _get_cell_feat_idx=lambda _cell_key, _feat_key: (
            np.array([0, 1]),
            np.arange(3),
        ),
        normed=lambda **_kwargs: DeferredMean(),
    )

    with pytest.raises(ValueError, match="missing must be"):
        Assay.mean_features(assay, ["GeneA"], missing="ignore")
    with pytest.raises(ValueError, match="must be non-empty"):
        Assay.mean_features(assay, [])
    with pytest.raises(ValueError, match="duplicate names"):
        Assay.mean_features(assay, ["GeneA", "genea"])
    with pytest.raises(ValueError, match="No requested features"):
        Assay.mean_features(assay, ["Missing"], missing="skip")

    feats.fetch_all = lambda _key: np.array(["GeneA", "genea", "GeneC"])
    with pytest.raises(ValueError, match="matches multiple"):
        Assay.mean_features(assay, ["GeneA"])

    feats.fetch_all = lambda _key: feature_names
    np.testing.assert_array_equal(
        Assay.mean_features(assay, ["GeneA"]),
        np.array([1.5, 2.5]),
    )


def test_base_assay_score_features_covers_generic_normalization(
    monkeypatch,
):
    class DeferredMean:
        def __init__(self, value):
            self.value = value

        def mean(self, axis):
            assert axis == 1
            return self

        def compute(self):
            return np.full(2, self.value)

    feats = SimpleNamespace(
        get_index_by=lambda _values, _column, _key: np.array([], dtype=int),
        fetch_all=lambda _key: np.array([0.1, 0.2, 0.3]),
    )
    assay = SimpleNamespace(
        feats=feats,
        _load_stats_loc=lambda _cell_key: "stats",
        _get_cell_feat_idx=lambda _cell_key, _feat_key: (
            np.array([0, 1]),
            np.arange(3),
        ),
        normed=lambda *, cell_idx, feat_idx: DeferredMean(float(feat_idx.mean())),
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


@pytest.mark.parametrize(
    ("values", "error_type", "match"),
    [
        ("0,1", TypeError, "sequence of integer"),
        ([[0, 1]], ValueError, "one-dimensional"),
        ([0.5, 1.5], TypeError, "only integer"),
    ],
)
def test_rna_feature_index_validation_rejects_malformed_values(
    values,
    error_type,
    match,
):
    from scarf.assay.rna import _as_feature_indexes

    with pytest.raises(error_type, match=match):
        _as_feature_indexes(
            values,
            n_features=3,
            name="features",
            require_unique=True,
        )


@pytest.mark.parametrize(
    ("n_bins", "lowess_frac", "strategy", "error_type", "match"),
    [
        (0, 0.1, "adaptive", ValueError, "greater than 0"),
        (10, "wide", "adaptive", TypeError, "must be numeric"),
        (10, 1.5, "fixed", ValueError, "between 0 and 1"),
    ],
)
def test_corrected_variance_column_rejects_invalid_parameters(
    n_bins,
    lowess_frac,
    strategy,
    error_type,
    match,
):
    from scarf.assay.rna import _corrected_variance_column

    with pytest.raises(error_type, match=match):
        _corrected_variance_column(n_bins, lowess_frac, strategy)


def test_rna_requires_zarr_v3_counts_t():
    rna = RNAassay.__new__(RNAassay)
    rna.name = "RNA"
    rna.rawDataT = SimpleNamespace(
        metadata=SimpleNamespace(zarr_format=2),
    )

    with pytest.raises(ValueError, match="requires Zarr v3"):
        rna._require_counts_t()


def test_rna_feature_major_reads_support_both_raw_orientations():
    from scarf.storage.partition import IndexBlock

    root = zarr.open_group(store=MemoryStore(), mode="w")
    feature_major = root.create_array(
        "countsT",
        data=np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ],
            dtype=np.uint32,
        ),
    )
    cell_major = root.create_array(
        "counts",
        data=np.asarray(feature_major[:]).T,
    )
    block = IndexBlock(
        indices=np.array([0, 2], dtype=np.int64),
        destinations=np.array([0, 1], dtype=np.int64),
        bins=(0,),
    )
    plan_feature_major = SimpleNamespace(
        featureAxis=0,
        blocks=(block,),
        readWorkers=1,
        ioConcurrency=1,
    )
    plan_cell_major = SimpleNamespace(
        featureAxis=1,
        blocks=(block,),
        readWorkers=1,
        ioConcurrency=1,
    )
    rna = RNAassay.__new__(RNAassay)
    rna.name = "RNA"
    rna.rawDataT = feature_major
    rna.rawData = SimpleNamespace(_backing=cell_major)
    cells = np.array([0, 2], dtype=np.int64)

    feature_blocks = list(
        rna.iter_raw_feature_major_blocks(cells, plan_feature_major, "Reading")
    )
    np.testing.assert_array_equal(
        feature_blocks[0][1],
        np.array([[1, 3], [7, 9]], dtype=np.uint32),
    )
    with pytest.raises(ValueError, match="does not match"):
        list(rna.iter_raw_feature_major_blocks(cells, plan_cell_major))

    rna.rawDataT = None
    cell_blocks = list(
        rna.iter_raw_feature_major_blocks(cells, plan_cell_major, "Reading")
    )
    np.testing.assert_array_equal(cell_blocks[0][1], feature_blocks[0][1])

    column_blocks = list(
        rna._iter_raw_column_blocks(
            cells,
            np.array([0, 2]),
            batch_size=None,
            plan=plan_cell_major,
        )
    )
    np.testing.assert_array_equal(
        column_blocks[0][1],
        np.array([[1, 7], [3, 9]], dtype=np.uint32),
    )


def test_rna_raw_feature_columns_log_and_normalize_batches():
    rna = RNAassay.__new__(RNAassay)
    rna.name = "RNA"
    rna._iter_raw_column_blocks = lambda **_kwargs: iter(
        [
            (
                0,
                np.array([[2, 4], [6, 8]], dtype=np.uint32),
                np.array([1, 3]),
                0.1,
                "memory",
            )
        ]
    )
    plan = SimpleNamespace(blocks=(object(),))

    batches = list(
        rna._iter_raw_feature_columns(
            np.array([0, 1]),
            np.array([1, 3]),
            batch_size=None,
            scalar=np.array([2.0, 4.0]),
            sf=2.0,
            log_transform=True,
            msg="Normalizing",
            plan=plan,
        )
    )

    np.testing.assert_allclose(
        batches[0][0],
        np.log1p(np.array([[2.0, 4.0], [3.0, 4.0]])),
    )
    np.testing.assert_array_equal(batches[0][1], np.array([1, 3]))


def test_rna_streaming_stats_and_group_means_handle_missing_inputs():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    counts = root.create_array(
        "counts",
        data=np.array([[1, 0], [0, 1]], dtype=np.uint32),
    )
    rna = RNAassay.__new__(RNAassay)
    rna.name = "RNA"
    rna.normMethod = norm_lib_size
    rna.sf = None
    rna.cells = SimpleNamespace(fetch_all=lambda _key: np.array([2.0, 3.0]))
    rna.rawData = SimpleNamespace(_backing=counts)
    rna.rawDataT = None

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

    with pytest.raises(ValueError, match="requires sharded countsT"):
        rna._streaming_feature_stats(
            np.array([0], dtype=int),
            np.array([0], dtype=int),
        )


def test_rna_feature_stat_arguments_and_missing_columns():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    stats = root.create_group("summary_stats_I")
    stats.attrs["subset_hash"] = "valid"
    rna = RNAassay.__new__(RNAassay)
    rna.z = root
    rna.feats = SimpleNamespace(active_index=lambda _key: np.array([0]))
    rna._get_cell_feat_idx = lambda _cell_key, _feat_key: (
        np.array([0]),
        np.array([0]),
    )
    rna._validate_stats_loc = lambda *_args, **_kwargs: True

    with pytest.raises(TypeError, match="not a string"):
        rna.get_feature_stats("I", columns="nz_mean")
    with pytest.raises(TypeError, match="only strings"):
        rna.get_feature_stats("I", columns=["nz_mean", 1])
    with pytest.raises(KeyError, match="not available"):
        rna.get_feature_stats("I", columns=["missing"])


def test_rna_summary_and_hvg_validation_paths():
    inserted = {}
    feats = SimpleNamespace(
        N=3,
        columns=[],
        grep=lambda _pattern: [],
        get_index_by=lambda _values, _column: np.array([], dtype=np.int64),
        insert=lambda name, values, **_kwargs: inserted.update({name: values}),
    )
    rna = RNAassay.__new__(RNAassay)
    rna.feats = feats
    rna.set_feature_stats = lambda _cell_key: None
    rna._load_stats_loc = lambda _cell_key: "stats"

    with pytest.raises(KeyError, match="stats_normed_tot"):
        rna.set_summary_stats(cell_key=None)
    with pytest.raises(TypeError, match="mask must be a NumPy array"):
        rna.set_hvgs("I", mask=[True, False, False])

    rna.set_summary_stats = lambda *_args, **_kwargs: ("stats", "c_var")
    column = rna.set_hvgs(
        "I",
        mask=np.array([True, False, False]),
        blacklist="",
    )
    assert column == "I__hvgs"
    np.testing.assert_array_equal(inserted[column], np.array([True, False, False]))


def test_rna_feature_stream_defaults_require_a_size_factor(monkeypatch):
    monkeypatch.setattr(
        "scarf.assay.rna.lib_size_feature_stream_eligible",
        lambda *_args, **_kwargs: True,
    )
    rna = RNAassay.__new__(RNAassay)
    rna.normMethod = norm_lib_size
    rna.cells = SimpleNamespace(
        N=2,
        active_index=lambda _key: np.array([0, 1]),
    )
    rna.feats = SimpleNamespace(
        N=2,
        active_index=lambda _key: np.array([0, 1]),
    )
    rna.sf = None

    with pytest.raises(ValueError, match="requires a size factor"):
        list(
            rna.iter_normed_feature_wise(
                cell_key=None,
                feat_key=None,
                batch_size=None,
                msg=None,
            )
        )


def test_rna_generic_feature_statistics_compute_all_reductions():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    inserted = {}
    locations = {"stats_I": object()}
    feats = SimpleNamespace(
        locations=locations,
        mount_location=lambda _group, _identifier: None,
        insert=lambda name, values, **_kwargs: inserted.update({name: values}),
        active_index=lambda _key: np.array([0, 1]),
        unmount_location=lambda _identifier: None,
    )
    rna = RNAassay.__new__(RNAassay)
    rna.name = "RNA"
    rna.z = root
    rna.feats = feats
    rna.nthreads = 1
    rna.normMethod = lambda _assay, values: values
    rna._get_cell_feat_idx = lambda _cell_key, _feat_key: (
        np.array([0, 1]),
        np.array([0, 1]),
    )
    rna._get_summary_stats_loc = lambda _cell_key: ("stats_I", "summary_stats_I")
    rna._validate_stats_loc = lambda *_args, **_kwargs: False
    rna._create_subset_hash = lambda _cells, _features: "digest"
    rna.normed = lambda *_args, **_kwargs: np.array([[1.0, 0.0], [3.0, 4.0]])

    rna.set_feature_stats("I")

    assert "stats_I" not in locations
    np.testing.assert_array_equal(inserted["normed_n"], np.array([2.0, 1.0]))
    np.testing.assert_array_equal(inserted["normed_tot"], np.array([4.0, 4.0]))
    np.testing.assert_allclose(inserted["sigmas"], np.array([1.0, 4.0]))

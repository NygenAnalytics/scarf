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

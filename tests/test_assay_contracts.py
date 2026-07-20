import inspect
from types import SimpleNamespace
from typing import get_type_hints

import numpy as np
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
        "set_summary_stats",
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
    Assay: "ec3721b2cc5d31858beb1652cc3ea20050dc675c70328b444bfb2cbb535038bc",
    RNAassay: "d45713ea3a32764b0be1f4b96221cc9f455c71af7c7af4be3859eb73df2a585b",
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
    assert assay_module.__all__ == ["Assay", "RNAassay", "ATACassay", "ADTassay"]
    expected = {
        "ADTassay",
        "ATACassay",
        "Assay",
        "NormMethod",
        "PSEUDOTIME_AGGREGATION_SCHEMA_VERSION",
        "PercentFeatures",
        "RNAassay",
        "_feature_stats_tile_shape",
        "_read_block",
        "lib_size_feature_stream_eligible",
        "norm_clr",
        "norm_dummy",
        "norm_lib_size",
        "norm_lib_size_log",
        "norm_tf_idf",
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
    root = zarr.open_group(store=MemoryStore(), mode="w")
    counts_t = root.create_array(
        "countsT",
        data=np.arange(12, dtype=np.uint32).reshape(4, 3),
    )
    rna = RNAassay.__new__(RNAassay)
    rna.name = "RNA"
    rna.z = root
    rna.rawDataT = counts_t

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

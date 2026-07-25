import pickle

import numpy as np
import zarr
from zarr.storage import MemoryStore

import scarf.metadata as metadata
from scarf.metadata.rows import MetaDataRowBlock as implementation_row_block
from scarf.metadata.table import MetaData as implementation_metadata
from tests.signature_contracts import signature_digest


_METHODS = {
    "__init__",
    "__repr__",
    "_col_renamer",
    "_column_map",
    "_fill_to_index",
    "_get_array",
    "_get_loc",
    "_get_size",
    "_save",
    "_verify_bool",
    "active_index",
    "default_block_rows",
    "drop",
    "fetch",
    "fetch_all",
    "get_dtype",
    "get_index_by",
    "grep",
    "head",
    "index_to_bool",
    "insert",
    "iter_row_blocks",
    "mount_location",
    "multi_sift",
    "remove_trend",
    "reset_key",
    "sift",
    "to_pandas_dataframe",
    "unmount_location",
    "update_key",
}


def _metadata_fixture() -> metadata.MetaData:
    group = zarr.open_group(store=MemoryStore(), mode="w")
    group.create_array(
        "I",
        data=np.array([True, False, True, True]),
        chunks=(2,),
    )
    group.create_array(
        "ids",
        data=np.array(["a", "b", "c", "d"]),
        chunks=(2,),
    )
    group.create_array(
        "names",
        data=np.array(["Alpha", "Beta", "Alpine", "Delta"]),
        chunks=(2,),
    )
    group.create_array(
        "score",
        data=np.array([0.5, 2.0, 3.5, 5.0]),
        chunks=(2,),
    )
    return metadata.MetaData(group)


def test_metadata_facade_exports_canonical_objects():
    assert metadata.__all__ == ["MetaData", "MetaDataRowBlock"]
    assert metadata.MetaData is implementation_metadata
    assert metadata.MetaDataRowBlock is implementation_row_block
    assert metadata.MetaData.__module__ == "scarf.metadata"
    assert metadata.MetaDataRowBlock.__module__ == "scarf.metadata"


def test_metadata_method_ownership_and_signatures_remain_stable():
    assert _METHODS <= set(metadata.MetaData.__dict__)
    methods = {name: getattr(metadata.MetaData, name) for name in _METHODS}

    assert signature_digest(methods) == (
        "f882e8fb00453f282ab117edfa27a4433ee2e46446b71631f8c4c450f0327d5d"
    )


def test_metadata_rows_and_queries_match_table_contract():
    table = _metadata_fixture()

    assert table.default_block_rows() == 2
    blocks = list(table.iter_row_blocks(columns=["score"], block_rows=2))
    np.testing.assert_array_equal(
        np.concatenate([block.active_global_indices for block in blocks]),
        [0, 2, 3],
    )
    np.testing.assert_allclose(
        np.concatenate([block.values["score"] for block in blocks]),
        [0.5, 3.5, 5.0],
    )
    np.testing.assert_array_equal(
        table.sift("score", 1.0, 4.0), [False, True, True, False]
    )
    assert table.grep("^AL") == ["ALPHA", "ALPINE"]
    assert table.head(2)["score"].tolist() == [0.5, 2.0]


def test_metadata_remove_trend_preserves_fixed_strategy():
    from scarf.features.variability import fit_lowess

    rng = np.random.default_rng(5)
    means = rng.uniform(0.5, 20.0, 80)
    variances = np.clip(means**1.5 + rng.normal(0, 0.05, len(means)), 0.1, None)
    group = zarr.open_group(store=MemoryStore(), mode="w")
    group.create_array("I", data=np.ones(len(means), dtype=bool))
    group.create_array("means", data=means)
    group.create_array("variances", data=variances)
    table = metadata.MetaData(group)

    observed = table.remove_trend("means", "variances", n_bins=8, lowess_frac=0.6)
    expected = fit_lowess(
        means,
        variances,
        n_bins=8,
        lowess_frac=0.6,
        bin_strategy="fixed",
    )

    np.testing.assert_array_equal(observed, expected)


def test_metadata_row_blocks_remain_pickle_resolvable():
    block = metadata.MetaDataRowBlock(
        start=0,
        stop=2,
        active_global_indices=np.array([0]),
        values={"score": np.array([0.5])},
    )

    restored = pickle.loads(pickle.dumps(block))

    assert type(restored) is metadata.MetaDataRowBlock
    np.testing.assert_array_equal(restored.active_global_indices, [0])

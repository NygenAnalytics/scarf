from types import MethodType

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.assay import ADTassay, RNAassay
from scarf.datastore.datastore import DataStore
from scarf.metadata import MetaData


def _rna_with_feature_names(
    names: list[str],
) -> tuple[RNAassay, list[tuple[str, int, float]]]:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    feature_data = root.create_group("featureData")
    feature_data.create_array("I", data=np.ones(len(names), dtype=bool))
    feature_data.create_array(
        "ids",
        data=np.asarray([f"feature_{index}" for index in range(len(names))]),
    )
    feature_data.create_array("names", data=np.asarray(names))

    assay = RNAassay.__new__(RNAassay)
    assay.z = root
    assay.feats = MetaData(feature_data)
    calls: list[tuple[str, int, float]] = []

    def set_summary_stats(
        self: RNAassay,
        cell_key: str,
        n_bins: int,
        lowess_frac: float,
    ) -> tuple[str, str]:
        calls.append((cell_key, n_bins, lowess_frac))
        return f"stats_{cell_key}", f"c_var__{n_bins}__{lowess_frac}"

    assay.set_summary_stats = MethodType(set_summary_stats, assay)
    return assay, calls


def test_set_hvgs_installs_mask_and_applies_regex_exclusions():
    assay, calls = _rna_with_feature_names(["MT-A", "GENE1", "RPL3", "KEEP"])

    column = assay.set_hvgs(
        "selected_cells",
        mask=np.ones(4, dtype=bool),
        blacklist="^MT-|^RPL",
        blacklist_exclusions="^RPL",
    )

    assert column == "selected_cells__hvgs"
    np.testing.assert_array_equal(
        assay.feats.fetch_all(column),
        np.array([False, True, True, True]),
    )
    assert calls == [("selected_cells", 200, 0.1)]


def test_set_hvgs_accepts_global_indexes_and_prefers_blacklist_indexes():
    assay, calls = _rna_with_feature_names(["MT-A", "GENE1", "GENE2", "KEEP"])

    column = assay.set_hvgs(
        "I",
        feature_indexes=[0, 1, 3],
        hvg_key_name="custom",
        n_bins=50,
        lowess_frac=0.2,
        blacklist="^KEEP",
        blacklist_indexes=[1],
    )

    assert column == "I__custom"
    np.testing.assert_array_equal(
        assay.feats.fetch_all(column),
        np.array([True, False, False, True]),
    )
    assert calls == [("I", 50, 0.2)]


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({}, ValueError),
        (
            {
                "mask": np.array([True, False]),
                "feature_indexes": [0],
            },
            ValueError,
        ),
        ({"mask": np.array([1, 0, 0, 0])}, TypeError),
        ({"mask": np.array([True, False])}, ValueError),
        ({"feature_indexes": []}, ValueError),
        ({"feature_indexes": [0, 0]}, ValueError),
        ({"feature_indexes": [4]}, IndexError),
        (
            {
                "feature_indexes": [0],
                "blacklist_exclusions": "^MT-",
            },
            ValueError,
        ),
    ],
)
def test_set_hvgs_validates_selection_before_computing_stats(kwargs, error):
    assay, calls = _rna_with_feature_names(["A", "B", "C", "D"])

    with pytest.raises(error):
        assay.set_hvgs("I", **kwargs)

    assert calls == []


def test_datastore_set_hvgs_delegates_and_rejects_non_rna_assays():
    assay = RNAassay.__new__(RNAassay)
    delegated = []

    def set_hvgs(self: RNAassay, cell_key: str, **kwargs) -> str:
        delegated.append((cell_key, kwargs))
        return f"{cell_key}__hvgs"

    assay.set_hvgs = MethodType(set_hvgs, assay)
    store = DataStore.__new__(DataStore)
    store._get_assay = lambda from_assay: assay

    result = store.set_hvgs(
        from_assay="RNA",
        cell_key="selected",
        feature_indexes=[2],
        blacklist_indexes=[1],
    )

    assert result == "selected__hvgs"
    assert delegated[0][0] == "selected"
    assert delegated[0][1]["feature_indexes"] == [2]
    assert delegated[0][1]["blacklist_indexes"] == [1]

    adt = ADTassay.__new__(ADTassay)
    store._get_assay = lambda from_assay: adt
    with pytest.raises(TypeError, match="RNAassay"):
        store.set_hvgs(cell_key="I", feature_indexes=[0])

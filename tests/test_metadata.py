import warnings

import numpy as np
import pytest


@pytest.fixture
def dummy_metadata(tmp_path):
    import zarr

    from scarf.metadata import MetaData

    fn = str(tmp_path / "dummy_metadata.zarr")
    g = zarr.open_group(fn, mode="w")
    data = np.array([1, 1, 1, 1, 0, 0, 1, 1, 1]).astype(bool)
    g.create_array(
        "I",
        data=data,
        chunks=(100000,),
    )
    yield MetaData(g)


def test_metadata_attrs(dummy_metadata):
    assert dummy_metadata.N == 9
    assert np.all(dummy_metadata.index == np.array(range(9)))


def test_metadata_fetch(dummy_metadata):
    assert len(dummy_metadata.fetch("I")) == 7
    assert len(dummy_metadata.fetch_all("I")) == 9


def test_metadata_grep_preserves_regex_character_classes(dummy_metadata):
    dummy_metadata.insert(
        "names",
        np.array(["RPS3", "RPSX", "mt-Co1", "MT1A", "g_a", "g-", "x y", "xy", "RPS4"]),
        overwrite=True,
    )

    assert dummy_metadata.grep(r"^RPS\d+$") == ["RPS3", "RPS4"]
    assert dummy_metadata.grep(r"^g_\w+$") == ["G_A"]
    assert dummy_metadata.grep(r"^x\sy$") == ["X Y"]
    assert dummy_metadata.grep("^MT-") == ["MT-CO1"]
    assert dummy_metadata.grep(r"^RPS\d+$", only_valid=True) == ["RPS3", "RPS4"]


def test_metadata_insert_encodes_none_as_missing_text(dummy_metadata):
    values = np.array(["a", None, "b", "", "a", "b", "a", "b", "a"], dtype=object)

    dummy_metadata.insert("group", values)

    stored = dummy_metadata.fetch_all("group")
    assert stored.tolist() == ["a", "", "b", "", "a", "b", "a", "b", "a"]


def test_metadata_verify_bool(dummy_metadata):
    assert dummy_metadata._verify_bool("I") is True


def test_metadata_active_index(dummy_metadata):
    a = np.array([0, 1, 2, 3, 6, 7, 8])
    assert np.all(dummy_metadata.active_index(key="I") == a)


def test_metadata_partial_float_fill_does_not_cast_uninitialized_values(dummy_metadata):
    values = np.arange(7, dtype=np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        filled = dummy_metadata._fill_to_index(values, np.nan, "I")

    selected = dummy_metadata.fetch_all("I")
    assert filled.dtype == values.dtype
    np.testing.assert_array_equal(filled[selected], values)
    assert np.isnan(filled[~selected]).all()

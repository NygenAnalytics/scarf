import pytest

from scarf.storage.metadata_keys import (
    is_metadata_column_key,
    is_reserved_metadata_name,
    metadata_column_key,
    metadata_column_keys,
    validate_metadata_column_name,
)


def test_metadata_column_key_replaces_both_path_separators():
    assert metadata_column_key("Baseline eGFR (ml/min/1.73m2) (Binned)") == (
        "Baseline eGFR (ml_min_1.73m2) (Binned)"
    )
    assert metadata_column_key("a\\b/c") == "a_b_c"
    assert metadata_column_key("/") == "_"
    assert metadata_column_key("plain") == "plain"


def test_metadata_column_keys_keeps_valid_names_and_renames_the_rest():
    keys = metadata_column_keys(["cell_type", "a/b", "c\\d"])

    assert keys == {"cell_type": "cell_type", "a/b": "a_b", "c\\d": "c_d"}


def test_metadata_column_keys_lets_exact_names_win_clashes():
    # The exact name keeps its key even when a renamed name comes first.
    assert metadata_column_keys(["a/b", "a_b"]) == {"a/b": "a_b_2", "a_b": "a_b"}
    assert metadata_column_keys(["a/b", "a\\b"]) == {"a/b": "a_b", "a\\b": "a_b_2"}
    assert metadata_column_keys(["a/b", "a_b", "a_b_2", "a\\b"]) == {
        "a/b": "a_b_3",
        "a_b": "a_b",
        "a_b_2": "a_b_2",
        "a\\b": "a_b_4",
    }


def test_metadata_column_keys_respects_names_already_taken():
    keys = metadata_column_keys(["a_b", "a/b", "x"], taken=["I", "ids", "names", "x"])

    assert keys == {"a_b": "a_b", "a/b": "a_b_2"}


def test_metadata_column_keys_leaves_out_reserved_and_unstorable_names():
    keys = metadata_column_keys(
        ["I", "ids", "names", "__scarf_missing__x", "__scarf/missing__x", "", ".", ".."]
        + ["kept", "ids/extra"]
    )

    assert keys == {"kept": "kept", "ids/extra": "ids_extra"}


def test_metadata_column_keys_keeps_source_order_and_plans_repeats_once():
    keys = metadata_column_keys(["z/1", "b", "a", "b", "z/1"])

    assert list(keys) == ["z/1", "b", "a"]
    assert keys["z/1"] == "z_1"


def test_metadata_column_keys_rejects_names_that_are_not_text():
    with pytest.raises(TypeError, match="must be strings"):
        metadata_column_keys(["a", 1])  # type: ignore[list-item]


@pytest.mark.parametrize("name", ["cell_type", "Baseline eGFR (ml_min)", "a.b", "_"])
def test_validate_metadata_column_name_accepts_storable_names(name):
    validate_metadata_column_name(name)


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("a/b", "use 'a_b' instead"),
        ("a\\b", "use 'a_b' instead"),
        ("", "cannot name a Zarr array"),
        (".", "cannot name a Zarr array"),
        ("..", "cannot name a Zarr array"),
    ],
)
def test_validate_metadata_column_name_rejects_unstorable_names(name, message):
    with pytest.raises(ValueError, match=message):
        validate_metadata_column_name(name)


def test_validate_metadata_column_name_rejects_names_that_are_not_text():
    with pytest.raises(TypeError, match="must be strings"):
        validate_metadata_column_name(3)  # type: ignore[arg-type]


def test_is_reserved_metadata_name_covers_columns_and_mask_prefix():
    assert all(is_reserved_metadata_name(name) for name in ("I", "ids", "names"))
    assert is_reserved_metadata_name("__scarf_missing__cell_type")
    assert not is_reserved_metadata_name("ids_2")


@pytest.mark.parametrize(
    "name", ["__scarf_missing__score", "__scarf/missing__score", "__scarf_missing__"]
)
def test_validate_metadata_column_name_rejects_the_mask_prefix(name):
    with pytest.raises(ValueError, match="reserves for missing-value masks"):
        validate_metadata_column_name(name)


def test_is_metadata_column_key_accepts_only_names_stored_as_given():
    assert is_metadata_column_key("cell_type")
    assert not any(
        is_metadata_column_key(name) for name in ("a/b", "a\\b", "", ".", "..", 3)
    )

import h5py
import numpy as np
import pytest

from scarf.readers._h5ad_columns import (
    column_order,
    is_column,
    read_column,
    table_column_names,
    table_members,
)

# Old AnnData versions wrote this CELLxGENE column as nested HDF5 groups.
NESTED_COLUMN = "Baseline eGFR (ml/min/1.73m2) (Binned)"


def _write_categorical(parent: h5py.Group, name: str, labels: list[str]) -> None:
    categories = sorted(set(labels))
    group = parent.create_group(name)
    group.attrs["encoding-type"] = "categorical"
    group.create_dataset("categories", data=np.asarray(categories, dtype="S"))
    group.create_dataset(
        "codes", data=np.asarray([categories.index(label) for label in labels])
    )


@pytest.fixture
def obs(tmp_path):
    with h5py.File(tmp_path / "table.h5ad", "w") as h5:
        group = h5.create_group("obs")
        group.attrs["_index"] = "_index"
        group.create_dataset("_index", data=np.asarray(["c1", "c2"], dtype="S"))
        group.create_dataset("b", data=np.asarray([1, 2]))
        _write_categorical(group, NESTED_COLUMN, ["30-60", ">60"])
        group.create_dataset("c\\d", data=np.asarray([3.0, 4.0]))
        group.create_dataset("extra", data=np.asarray([5, 6]))
        group.create_group("unsupported").create_dataset("x", data=np.arange(2))
        h5.create_dataset("X", data=np.asarray([7, 8]))
        yield group


def test_table_members_resolves_names_that_hdf5_nests(obs):
    obs.attrs["column-order"] = ["b", NESTED_COLUMN, "c\\d", "gone", "/X", ".", "a//b"]

    members = table_members(obs)

    assert [name for name, _node in members.members] == [
        "b",
        NESTED_COLUMN,
        "c\\d",
        "_index",
        "extra",
        "unsupported",
    ]
    # Absolute and dot paths would resolve outside the table or to itself.
    assert members.unresolved == ("gone", "/X", ".", "a//b")
    nested = dict(members.members)[NESTED_COLUMN]
    values, missing = read_column(nested)
    assert values.tolist() == [b"30-60", b">60"]
    assert not missing.any()
    assert table_column_names(obs) == ["b", NESTED_COLUMN, "c\\d", "_index", "extra"]


def test_table_members_without_column_order_lists_direct_children(obs):
    members = table_members(obs)

    names = [name for name, _node in members.members]
    assert names == list(obs.keys())
    assert "Baseline eGFR (ml" in names
    assert not is_column(obs["Baseline eGFR (ml"])
    assert NESTED_COLUMN not in table_column_names(obs)
    assert members.unresolved == ()


def test_column_order_decodes_byte_names_and_drops_repeats(obs):
    obs.attrs["column-order"] = np.asarray([b"b", NESTED_COLUMN.encode(), b"b"])

    assert column_order(obs) == ("b", NESTED_COLUMN)
    assert table_column_names(obs)[:2] == ["b", NESTED_COLUMN]


def test_column_order_reads_an_empty_table(obs):
    # AnnData stores an empty column list as an empty float array.
    obs.attrs["column-order"] = np.asarray([], dtype=np.float64)

    assert column_order(obs) == ()
    assert table_column_names(obs) == ["_index", "b", "c\\d", "extra"]


def test_column_order_rejects_values_that_are_not_names(obs):
    obs.attrs["column-order"] = np.asarray([1.0, 2.0])

    with pytest.raises(ValueError, match="does not list column names"):
        column_order(obs)


def test_table_column_names_lists_compound_dataset_fields(tmp_path):
    with h5py.File(tmp_path / "legacy.h5ad", "w") as h5:
        fields = np.dtype([("index", "S2"), ("a/b", "i8")])
        table = h5.create_dataset("obs", data=np.zeros(2, dtype=fields))

        assert table_column_names(table) == ["index", "a/b"]


def test_table_members_reads_no_column_values(obs, monkeypatch):
    obs.attrs["column-order"] = ["b", NESTED_COLUMN, "c\\d"]

    def refuse(*_args, **_kwargs):
        raise AssertionError("column values were read")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", refuse)

    assert [name for name, _node in table_members(obs).members][:3] == [
        "b",
        NESTED_COLUMN,
        "c\\d",
    ]
    assert table_column_names(obs)[:3] == ["b", NESTED_COLUMN, "c\\d"]


def test_table_members_resolves_an_index_that_hdf5_nests(tmp_path):
    with h5py.File(tmp_path / "index.h5ad", "w") as h5:
        obs = h5.create_group("obs")
        obs.attrs["_index"] = "cell/id"
        obs.create_dataset("cell/id", data=np.asarray(["c1", "c2"], dtype="S"))
        obs.create_dataset("b", data=np.asarray([1, 2]))
        obs.attrs["column-order"] = ["b"]

        members = table_members(obs)

        assert [name for name, _node in members.members] == ["b", "cell/id"]
        assert members.unresolved == ()
        assert table_column_names(obs) == ["b", "cell/id"]


def test_table_members_resolves_a_nested_index_without_column_order(tmp_path):
    with h5py.File(tmp_path / "index.h5ad", "w") as h5:
        obs = h5.create_group("obs")
        obs.attrs["_index"] = "cell/id"
        obs.create_dataset("cell/id", data=np.asarray(["c1", "c2"], dtype="S"))
        obs.create_dataset("b", data=np.asarray([1, 2]))

        members = table_members(obs)

        assert [name for name, _node in members.members] == ["cell/id", "b"]
        assert table_column_names(obs) == ["cell/id", "b"]


def test_table_members_keeps_a_column_that_a_listed_path_runs_through(obs):
    # A malformed column-order that lists a path inside a real categorical
    # must not hide that categorical.
    _write_categorical(obs, "q", ["x", "y"])
    obs.attrs["column-order"] = ["b", "q/codes"]

    names = [name for name, _node in table_members(obs).members]

    assert names[:2] == ["b", "q/codes"]
    assert "q" in names
    assert "q" in table_column_names(obs)


def test_table_members_keeps_nested_groups_that_hold_unresolved_columns(tmp_path):
    with h5py.File(tmp_path / "siblings.h5ad", "w") as h5:
        obs = h5.create_group("obs")
        obs.attrs["_index"] = "cell/id"
        obs.create_dataset("cell/id", data=np.asarray(["c1", "c2"], dtype="S"))
        obs.create_dataset("cell/type", data=np.asarray(["T", "B"], dtype="S"))
        obs.create_dataset("a/b", data=np.asarray([1, 2]))
        obs.create_dataset("a/c", data=np.asarray([3, 4]))
        obs.create_dataset("x/y/z", data=np.asarray([5, 6]))
        obs.attrs["column-order"] = ["a/b", "x/y/z"]

        names = [name for name, _node in table_members(obs).members]

        # 'cell/type' and 'a/c' are not listed, so their groups stay visible
        # and the reader reports them instead of dropping them silently.
        assert names == ["a/b", "x/y/z", "cell/id", "a", "cell"]
        assert table_column_names(obs) == ["a/b", "x/y/z", "cell/id"]


def test_table_column_names_leaves_out_multidimensional_datasets(obs):
    obs.create_dataset("two_d", data=np.zeros((2, 3)))

    assert "two_d" in [name for name, _node in table_members(obs).members]
    assert "two_d" not in table_column_names(obs)

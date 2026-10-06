"""Metadata tables validate a change in full before they change the store."""

import threading

import numpy as np
import pytest
import zarr
from zarr.abc.store import Store
from zarr.core.buffer import Buffer
from zarr.storage import MemoryStore, WrapperStore

from scarf.metadata import MetaData
from scarf.storage.identity import replace_metadata_column
from tests.store_probes import RecordingStore

_MASK = "__scarf_missing__label"


def _table() -> MetaData:
    group = zarr.open_group(store=MemoryStore(), mode="w")
    group.create_array("I", data=np.array([True, False, True, True]))
    group.create_array("ids", data=np.array(["a", "b", "c", "d"]))
    group.create_array("names", data=np.array(["a", "b", "c", "d"]))
    score = group.create_array("score", data=np.array([0.5, 2.0, 3.5, 5.0]))
    score.attrs["unit"] = "pg"
    label = group.create_array("label", data=np.array(["x", "", "y", "x"]))
    group.create_array(_MASK, data=np.array([False, True, False, False]))
    label.attrs["missing_mask"] = _MASK
    return MetaData(group)


def _state(table: MetaData) -> dict[str, tuple[list, dict]]:
    group = table.locations["primary"]
    return {
        name: (np.asarray(array[:]).tolist(), dict(array.attrs))
        for name, array in group.arrays()
    }


@pytest.mark.parametrize(
    ("column", "values"),
    [
        ("score", np.array([b"\xff", b"a", b"b", b"c"])),
        ("label", np.array([b"\xff", "a", None, "c"], dtype=object)),
    ],
    ids=["score-bytes", "label-object"],
)
def test_a_rejected_replacement_keeps_the_column_and_its_mask(column, values) -> None:
    table = _table()
    before = _state(table)
    with pytest.raises(
        ValueError, match=f"Column '{column}' holds text that is not valid UTF-8"
    ) as raised:
        table.insert(column, values, overwrite=True)
    assert isinstance(raised.value.__cause__, UnicodeDecodeError)
    assert _state(table) == before
    with pytest.raises(ValueError, match="holds text that is not valid UTF-8"):
        replace_metadata_column(table.locations["primary"], column, values)
    with pytest.raises(ValueError, match="misaligned missing mask"):
        replace_metadata_column(
            table.locations["primary"], column, np.zeros(4), np.zeros(3, dtype=bool)
        )
    assert _state(table) == before


def test_a_valid_replacement_drops_the_previous_mask() -> None:
    table = _table()
    table.insert("label", np.array([b"caf\xc3\xa9", b"b", b"c", b"d"]), overwrite=True)
    group = table.locations["primary"]
    assert group["label"][:].tolist() == ["café", "b", "c", "d"]
    assert "missing_mask" not in group["label"].attrs
    assert _MASK not in group


def test_multi_sift_applies_every_filter() -> None:
    table = _table()
    expected = [False, True, True, False]
    np.testing.assert_array_equal(
        table.multi_sift(["score", "score"], [1.0, 0.0], [10.0, 4.0]), expected
    )
    # Any iterable of bounds is read once, in column order.
    np.testing.assert_array_equal(
        table.multi_sift(
            ("score", "score"),
            (low for low in [1.0, 0.0]),
            np.array([10.0, 4.0]),
        ),
        expected,
    )


@pytest.mark.parametrize(
    ("columns", "lows", "highs", "error", "message"),
    [
        ("score", [0.0], [1.0], TypeError, "columns must be a sequence of"),
        (["score"], [0.0], 1.0, TypeError, "highs must be a sequence of"),
        ([], [], [], ValueError, "multi_sift requires at least one column"),
        (
            ["score", "score"],
            [0.0],
            [1.0, 2.0],
            ValueError,
            "one lower and one upper bound for each column: 2 columns, 1 lows, "
            "and 2 highs",
        ),
    ],
)
def test_multi_sift_rejects_filters_it_cannot_pair(
    columns, lows, highs, error, message
) -> None:
    with pytest.raises(error, match=message):
        _table().multi_sift(columns, lows, highs)


class _FailingWrites(WrapperStore[Store]):
    """A backend whose chunk writes under the listed key prefixes fail.

    Writes of an array's ``zarr.json`` succeed, so a failure lands after the
    metadata of a new column and before its chunks, as an interrupted upload
    does. A prefix in ``failing`` fails every write; a prefix in ``once``
    fails one write, so the restore that follows succeeds.
    """

    def __init__(
        self,
        store: Store,
        failing: set[str] | None = None,
        once: set[str] | None = None,
    ) -> None:
        super().__init__(store)
        self.failing = set() if failing is None else failing
        self.once = set() if once is None else once
        self._lock = threading.Lock()

    def _with_store(self, store: Store) -> "_FailingWrites":
        return type(self)(store, self.failing, self.once)

    def _check(self, key: str) -> None:
        if key.endswith("zarr.json"):
            return
        with self._lock:
            once = {prefix for prefix in self.once if key.startswith(prefix)}
            self.once -= once
        if once or any(key.startswith(prefix) for prefix in self.failing):
            raise OSError(f"write rejected: {key}")

    async def set(self, key: str, value: Buffer) -> None:
        self._check(key)
        await self._store.set(key, value)

    def set_sync(self, key: str, value: Buffer) -> None:
        self._check(key)
        self._store.set_sync(key, value)  # type: ignore[attr-defined]


def _failing_table() -> tuple[MetaData, _FailingWrites]:
    store = _FailingWrites(MemoryStore())
    group = zarr.open_group(store=store, mode="w")
    group.create_array("I", data=np.array([True, False, True, True]))
    group.create_array("ids", data=np.array(["a", "b", "c", "d"]))
    group.create_array("names", data=np.array(["a", "b", "c", "d"]))
    # Columns of several chunks are restored whole.
    group.create_array(
        "score",
        data=np.array([0.5, 2.0, 3.5, 5.0]),
        chunks=(2,),
        attributes={"unit": "pg"},
    )
    group.create_array(_MASK, data=np.array([False, True, False, False]), chunks=(2,))
    group.create_array(
        "label",
        data=np.array(["x", "", "y", "x"]),
        chunks=(1,),
        attributes={"missing_mask": _MASK, "levels": ["x", "y"], "ordered": True},
    )
    return MetaData(group), store


@pytest.mark.parametrize(
    ("column", "failing", "values"),
    [
        ("score", "score/", np.array([1.0, 2.0, 3.0, 4.0])),
        ("label", "label/", np.array(["p", None, "r", "s"], dtype=object)),
        ("label", _MASK + "/", np.array(["p", None, "r", "s"], dtype=object)),
    ],
    ids=["numeric_values", "values_of_a_nullable_column", "missing_mask"],
)
def test_a_failed_write_restores_the_previous_column(column, failing, values) -> None:
    table, store = _failing_table()
    before = _state(table)
    store.once.add(failing)

    with pytest.raises(OSError, match="write rejected"):
        table.insert(column, values, overwrite=True)

    # The previous values, attributes, and mask are back; nothing of the new
    # column, which would read as fill values, remains.
    assert not store.once
    assert _state(table) == before
    np.testing.assert_array_equal(
        table.to_pandas_dataframe(["label"])["label"].isna(),
        [False, True, False, False],
    )


def test_a_failed_write_of_a_new_column_leaves_no_partial_column() -> None:
    table, store = _failing_table()
    before = _state(table)
    store.once.add("depth/")

    with pytest.raises(OSError, match="write rejected"):
        table.insert("depth", np.arange(4.0))

    assert _state(table) == before
    assert "depth" not in table.columns


def test_a_failed_key_reset_restores_the_previous_key() -> None:
    table, store = _failing_table()
    before = _state(table)

    store.once.add("I/")
    with pytest.raises(OSError, match="write rejected"):
        table.reset_key("I")
    store.once.add("I/")
    with pytest.raises(OSError, match="write rejected"):
        table.update_key(np.array([True, False, False, True]), "I")

    assert _state(table) == before


def test_a_failed_restore_adds_a_note_to_the_original_error() -> None:
    table, store = _failing_table()
    # Every chunk write of the column fails, including the restore.
    store.failing.add("label/")

    with pytest.raises(OSError, match="write rejected") as raised:
        table.insert("label", np.array(["p", "q", "r", "s"]), overwrite=True)

    notes = "\n".join(raised.value.__notes__)
    assert "Metadata column 'label' could not be restored (OSError" in notes


def test_an_interrupted_write_restores_the_previous_column(monkeypatch) -> None:
    table = _table()
    before = _state(table)
    create_array = zarr.Group.create_array
    interrupted: list[str] = []

    def interrupt_after_metadata(group, name, **kwargs):
        if name == "label" and not interrupted:
            interrupted.append(name)
            # The column's metadata is written; its chunks never are.
            create_array(group, name, **{**kwargs, "write_data": False})
            raise KeyboardInterrupt
        return create_array(group, name, **kwargs)

    monkeypatch.setattr(zarr.Group, "create_array", interrupt_after_metadata)
    with pytest.raises(KeyboardInterrupt):
        table.insert("label", np.array(["p", "q", "r", "s"]), overwrite=True)

    assert interrupted == ["label"]
    assert _state(table) == before


def test_a_replacement_links_its_mask_in_the_first_metadata_write() -> None:
    store = RecordingStore()
    group = zarr.open_group(store=store, mode="w")
    group.create_array("I", data=np.array([True, False, True, True]))
    # A stale mask that no column links is removed by the next replacement.
    group.create_array(_MASK, data=np.array([True, True, True, True]))
    group.create_array("label", data=np.array(["x", "y", "z", "w"]))
    table = MetaData(group)
    store.reset()

    table.insert("label", np.array(["p", "r", "s"]), overwrite=True)

    writes = [key for kind, key in store.ops if kind == "set"]
    metadata_writes = [key for key in writes if key.endswith("zarr.json")]
    # The mask is written before the values, whose one metadata write links it.
    assert metadata_writes == [f"{_MASK}/zarr.json", "label/zarr.json"]
    assert writes.index(f"{_MASK}/zarr.json") < writes.index("label/zarr.json")
    assert dict(group["label"].attrs) == {"missing_mask": _MASK}
    assert group[_MASK][:].tolist() == [False, True, False, False]


def test_a_read_only_table_refuses_a_write_before_changing_anything(tmp_path) -> None:
    path = str(tmp_path / "table.zarr")
    group = zarr.open_group(path, mode="w")
    group.create_array("I", data=np.array([True, False, True]))
    group.create_array("score", data=np.array([1.0, 2.0, 3.0]))
    table = MetaData(zarr.open_group(path, mode="r"))

    for write in (
        lambda: table.insert("score", np.zeros(3), overwrite=True),
        lambda: table.insert("new", np.zeros(3)),
        lambda: table.reset_key("I"),
    ):
        with pytest.raises(ValueError, match="the store is open read-only"):
            write()
    reopened = zarr.open_group(path, mode="r")
    assert sorted(reopened.array_keys()) == ["I", "score"]
    assert reopened["score"][:].tolist() == [1.0, 2.0, 3.0]

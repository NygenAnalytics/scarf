"""Partial and nullable inserts flag missing rows under the linked mask."""

import datetime

import numpy as np
import pandas as pd
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.metadata import MetaData
from scarf.metadata.encoding import missing_rows, nullable_values
from scarf.metadata.rows import metadata_column_fingerprint, metadata_missing_mask
from scarf.storage.selections import _stored_selection_summary
from scarf.writers._store import write_metadata_column

_SELECTED = np.array([True, False, True])
_OUTSIDE = (~_SELECTED).tolist()
_DATES = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
_DAYS = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")
_SECONDS = np.array([1, 2], dtype="timedelta64[s]")
_NANOSECONDS = np.array([1, 2], dtype="timedelta64[ns]")
_NAT = np.datetime64("NaT", "ns")


def _table() -> MetaData:
    group = zarr.open_group(store=MemoryStore(), mode="w")
    group.create_array("I", data=_SELECTED)
    group.create_array("ids", data=np.array(["a", "b", "c"]))
    group.create_array("names", data=np.array(["A", "B", "C"]))
    group.create_array("every", data=np.ones(3, dtype=bool))
    return MetaData(group)


def _mask(table: MetaData, column: str) -> list[bool] | None:
    mask = metadata_missing_mask(table, column)
    return None if mask is None else np.asarray(mask[:]).tolist()


def _same(left: list, right: list) -> bool:
    return len(left) == len(right) and all(
        a == b or (pd.isna(a) and pd.isna(b)) for a, b in zip(left, right)
    )


@pytest.mark.parametrize(
    ("values", "dtype", "stored"),
    [
        (np.array(["A", "B"]), "<U1", ["A", "", "B"]),
        (np.array([b"A", b"Bc"]), "<U2", ["A", "", "Bc"]),
        (np.array(["A", "B"], dtype=object), "<U1", ["A", "", "B"]),
        (pd.Categorical(["A", "B"]), "<U1", ["A", "", "B"]),
        (np.array([3, 4]), "int64", [3, 0, 4]),
        (np.array([True, True]), "bool", [True, False, True]),
        (np.array([1.5, 2.5]), "float64", [1.5, np.nan, 2.5]),
        (np.array([1 + 2j, 3j]), "complex128", [1 + 2j, np.nan, 3j]),
        (_DATES, "<M8[ns]", [_DATES[0], _NAT, _DATES[1]]),
        (
            np.array([1, 2], dtype="timedelta64[s]"),
            "<m8[s]",
            [
                np.timedelta64(1, "s"),
                np.timedelta64("NaT", "s"),
                np.timedelta64(2, "s"),
            ],
        ),
        (pd.array([1, 2], dtype="Int64"), "int64", [1, 0, 2]),
        (pd.array([True, False], dtype="boolean"), "bool", [True, False, False]),
    ],
    ids=[
        "U1",
        "bytes",
        "object",
        "categorical",
        "int64",
        "bool",
        "float64",
        "complex",
        "datetime",
        "timedelta",
        "Int64",
        "boolean",
    ],
)
def test_a_partial_insert_flags_the_other_rows_as_missing(values, dtype, stored):
    table = _table()

    table.insert("x", values)

    assert table.get_dtype("x") == np.dtype(dtype)
    raw = table.fetch_all("x").tolist()
    assert _same(raw, list(np.asarray(stored, dtype=dtype).tolist())), raw
    assert _mask(table, "x") == _OUTSIDE
    # fetch stays raw; the rows that I selects hold the supplied values.
    assert len(table.fetch("x")) == 2
    frame = table.to_pandas_dataframe(["x"])["x"]
    assert frame.isna().tolist() == _OUTSIDE
    if np.dtype(dtype).kind in "Mm":
        # Missing datetimes and timedeltas are NaT, so the column keeps its dtype.
        assert frame.dtype == np.dtype(dtype)
    assert table.head(3)["x"].isna().tolist() == _OUTSIDE


def test_missing_supplied_values_are_flagged_in_full_and_partial_inserts():
    table = _table()

    table.insert("full", np.array(["a", None, "b"], dtype=object))
    table.insert("partial", np.array(["a", pd.NA], dtype=object))
    table.insert("nullable", pd.array([1, None], dtype="Int64"))
    table.insert("categories", pd.Categorical(["a", None, "b"]))
    table.insert("nan_text", pd.Series(["a", np.nan, "b"], dtype=object))
    table.insert(
        "strings",
        np.array(["a", None, "b"], dtype=np.dtypes.StringDType(na_object=None)),
    )

    assert table.fetch_all("full").tolist() == ["a", "", "b"]
    assert _mask(table, "full") == [False, True, False]
    assert table.fetch_all("partial").tolist() == ["a", "", ""]
    assert _mask(table, "partial") == [False, True, True]
    assert table.get_dtype("nullable") == np.int64
    assert table.fetch_all("nullable").tolist() == [1, 0, 0]
    assert _mask(table, "nullable") == [False, True, True]
    assert _mask(table, "categories") == [False, True, False]
    assert _mask(table, "nan_text") == [False, True, False]
    assert table.fetch_all("strings").tolist() == ["a", "", "b"]
    assert _mask(table, "strings") == [False, True, False]
    # A float NaN that NumPy already holds stays a value, as imports keep it.
    table.insert("real_nan", np.array([1.0, np.nan, 2.0]))
    assert _mask(table, "real_nan") is None


@pytest.mark.parametrize(
    ("values", "fill_value", "dtype", "stored"),
    [
        (np.array(["A", "B"]), "unknown", "<U7", ["A", "unknown", "B"]),
        (np.array([3, 4]), 7, "int64", [3, 7, 4]),
        (np.array([3, 4], dtype=np.uint8), np.uint8(255), "uint8", [3, 255, 4]),
        (np.array([True, True]), False, "bool", [True, False, True]),
        (np.array([1.5, 2.5]), np.nan, "float64", [1.5, np.nan, 2.5]),
        (np.array([1.5, 2.5]), 10**20, "float64", [1.5, 1e20, 2.5]),
        (np.array([1j, 2j]), 1.5, "complex128", [1j, 1.5, 2j]),
        (
            _DATES,
            np.datetime64("2021-01-01"),
            "<M8[ns]",
            [_DATES[0], np.datetime64("2021-01-01", "ns"), _DATES[1]],
        ),
        (
            _DATES,
            pd.Timestamp("2021-01-01 00:00:00.000000001"),
            "<M8[ns]",
            [_DATES[0], np.datetime64("2021-01-01T00:00:00.000000001"), _DATES[1]],
        ),
        (
            _DAYS,
            datetime.datetime(2021, 1, 1),
            "<M8[D]",
            [_DAYS[0], np.datetime64("2021-01-01", "D"), _DAYS[1]],
        ),
        (_DATES, _NAT, "<M8[ns]", [_DATES[0], _NAT, _DATES[1]]),
        (
            _SECONDS,
            np.timedelta64(2, "m"),
            "<m8[s]",
            [_SECONDS[0], np.timedelta64(120, "s"), _SECONDS[1]],
        ),
        (
            _NANOSECONDS,
            pd.Timedelta(1, "ns"),
            "<m8[ns]",
            [_NANOSECONDS[0], np.timedelta64(1, "ns"), _NANOSECONDS[1]],
        ),
        (
            _SECONDS,
            pd.NaT,
            "<m8[s]",
            [_SECONDS[0], np.timedelta64("NaT", "s"), _SECONDS[1]],
        ),
        (np.array([b"A", b"B"]), "unknown", "<U7", ["A", "unknown", "B"]),
    ],
)
def test_an_explicit_fill_is_stored_as_a_real_value(values, fill_value, dtype, stored):
    table = _table()

    table.insert("x", values, fill_value=fill_value)

    assert table.get_dtype("x") == np.dtype(dtype)
    raw = table.fetch_all("x").tolist()
    assert _same(raw, list(np.asarray(stored, dtype=dtype).tolist())), raw
    assert _mask(table, "x") is None


@pytest.mark.parametrize(
    ("values", "fill_value"),
    [
        (np.array(["A", "B"]), 0),
        (np.array([3, 4]), 2.7),
        (np.array([3, 4]), True),
        (np.array([3, 4], dtype=np.int8), 300),
        (np.array([True, False]), "x"),
        (np.array([1.5, 2.5], dtype=np.float32), 1e300),
        (_DATES, "2021-01-01"),
        # A fill that the unit of the values cannot hold exactly: truncated,
        # out of range, or tied to a time zone.
        (_DAYS, datetime.datetime(2020, 1, 1, 12, 30)),
        (_DATES, datetime.datetime(3000, 1, 1)),
        (_DATES, datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc)),
        (_SECONDS, np.timedelta64(1500, "ms")),
        (_NANOSECONDS, np.timedelta64(10**6, "D")),
        (_SECONDS, _DATES[0]),
        (np.array([3, 4]), np.timedelta64(5, "s")),
        (np.array([1.5, 2.5]), 10**400),
    ],
)
def test_an_explicit_fill_that_does_not_fit_is_rejected_before_writing(
    values, fill_value
):
    table = _table()
    table.insert("x", np.array([7, 8, 9]))

    with pytest.raises(ValueError, match="fill_value"):
        table.insert("x", values, fill_value=fill_value, overwrite=True)

    assert table.fetch_all("x").tolist() == [7, 8, 9]
    assert _mask(table, "x") is None


def test_an_explicit_fill_is_not_needed_when_values_cover_every_row():
    table = _table()

    table.insert("x", np.array([True, False, True]), fill_value=np.nan)

    assert table.fetch_all("x").tolist() == [True, False, True]
    assert _mask(table, "x") is None


def test_reserved_columns_hold_no_missing_values():
    table = _table()
    values = np.array([True, True])

    for supplied in (values, np.array([True, None, True], dtype=object)):
        with pytest.raises(ValueError, match="cannot hold missing values"):
            table.insert("I", supplied, overwrite=True, force=True)
    assert table.fetch_all("I").tolist() == [True, False, True]

    table.insert("I", values, fill_value=False, overwrite=True, force=True)
    assert _mask(table, "I") is None


def test_update_key_keeps_an_explicit_false_without_a_mask():
    table = _table()

    table.update_key(np.array([True, False]), "I")
    assert table.fetch_all("I").tolist() == [True, False, False]
    assert _mask(table, "I") is None

    table.update_key(np.array([True, False, True]), "every")
    assert table.fetch_all("every").tolist() == [True, False, True]
    # A key is only restricted: True never selects a row again.
    table.update_key(np.array([True, True, True]), "every")
    assert table.fetch_all("every").tolist() == [True, False, True]
    with pytest.raises(TypeError, match="booleans"):
        table.update_key(np.array([1, 0, 1]), "every")


def test_a_masked_boolean_key_selects_what_an_explicit_false_selects():
    table = _table()

    table.insert("masked", np.array([True, True]))
    table.insert("filled", np.array([True, True]), fill_value=False)

    assert _mask(table, "masked") == _OUTSIDE
    np.testing.assert_array_equal(table.active_index("masked"), [0, 2])
    np.testing.assert_array_equal(
        table.fetch("ids", key="masked"), table.fetch("ids", key="filled")
    )
    # Stored selections read raw values, so their identity does not change.
    assert _stored_selection_summary(
        table._get_array("masked")
    ) == _stored_selection_summary(table._get_array("filled"))
    # Column fingerprints cover the mask, so they differ.
    assert metadata_column_fingerprint(table, "masked") != metadata_column_fingerprint(
        table, "filled"
    )


def test_nullable_values_follow_the_import_contract():
    values, missing = nullable_values(
        np.array([1, None, 2.5, pd.NA], dtype=object), name="x"
    )
    assert values.dtype == np.float64
    assert _same(values.tolist(), [1.0, np.nan, 2.5, np.nan])
    assert missing.tolist() == [False, True, False, True]
    for shaped in (np.zeros((2, 2)), np.float64(1.0)):
        with pytest.raises(ValueError, match="one value per row"):
            nullable_values(shaped, name="x")
    # A bool is no integer and a timedelta no number, so these share no
    # number dtype and are stored as text, as values that are all missing are.
    for supplied, text in (
        ([True, 2, None], ["True", "2", ""]),
        ([np.timedelta64(1, "s"), None], ["1 seconds", ""]),
        ([None, None], ["", ""]),
    ):
        values, _ = nullable_values(np.array(supplied, dtype=object), name="x")
        assert (values.dtype.kind, values.tolist()) == ("U", text)
    values, missing = nullable_values(np.array([2j, None], dtype=object), name="x")
    assert values.dtype == np.complex128
    assert missing.tolist() == [False, True]
    with pytest.raises(ValueError, match="Column 'x' holds text that is not valid"):
        nullable_values(np.array([b"\xff", None], dtype=object), name="x")
    assert missing_rows(
        np.array([None, np.nan, pd.NA, pd.NaT, "", [1]], dtype=object)
    ).tolist() == [True, True, True, True, False, False]
    assert not missing_rows(np.array([np.nan, 1.0])).any()


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([2**63, 1, None], "holds integers outside the int64 range"),
        ([10**400, 1.5, None], "holds numbers too large for float64"),
        pytest.param(
            np.array([np.longdouble("1e400"), None], dtype=object),
            "holds numbers too large for float64",
            marks=pytest.mark.skipif(
                np.finfo(np.longdouble).max <= np.finfo(np.float64).max,
                reason="this platform's long double is a float64",
            ),
        ),
    ],
    ids=["above_int64", "beyond_float64", "long_double_beyond_float64"],
)
def test_numbers_that_their_shared_dtype_cannot_hold_are_rejected(values, message):
    with pytest.raises(ValueError, match=f"Column 'x' {message}"):
        nullable_values(values, name="x")
    table = _table()
    with pytest.raises(ValueError, match=f"Column 'x' {message}"):
        table.insert("x", values)
    assert "x" not in table.columns


def test_lookups_never_match_a_missing_row():
    table = _table()
    table.insert("x", np.array([0, None, 5], dtype=object))
    table.insert("last_two", np.array([False, True, True]))

    # The missing row stores the placeholder 0.
    assert table.fetch_all("x").tolist() == [0, 0, 5]
    assert table.get_index_by([0], "x").tolist() == [0]
    assert table.get_index_by([0, 5], "x", key="last_two").tolist() == [1]


def test_integers_at_the_int64_limits_keep_int64():
    values, missing = nullable_values([2**63 - 1, -(2**63), None], name="x")

    assert values.dtype == np.int64
    assert values.tolist() == [2**63 - 1, -(2**63), 0]
    assert missing.tolist() == [False, False, True]


def test_imports_flag_every_kind_of_missing_object_value():
    group = zarr.open_group(store=MemoryStore(), mode="w")

    write_metadata_column(
        group, "label", np.array(["a", pd.NA, None, pd.NaT], dtype=object)
    )
    write_metadata_column(group, "when", _DATES, missing=np.array([False, True]))

    assert group["label"][:].tolist() == ["a", "", "", ""]
    assert group["__scarf_missing__label"][:].tolist() == [False, True, True, True]
    assert np.isnat(group["when"][:]).tolist() == [False, True]
    assert group["when"][0] == _DATES[0]

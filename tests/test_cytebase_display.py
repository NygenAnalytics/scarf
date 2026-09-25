"""Tests for the Markdown tables returned by Cytebase catalog queries."""

import pytest

from scarf.cytebase.display import CatalogResults, _cell, _validate_max_cell_chars

ROWS = [
    {
        "cytebase_id": "lung_a",
        "cell_count": 1234,
        "year": 2024,
        "first_author": "Smith",
    },
    {"cytebase_id": "lung_b", "cell_count": None, "year": 2023, "first_author": "Lee"},
    {"cytebase_id": "lung_c", "cell_count": 7, "year": 2022, "first_author": "Kim"},
]


class _Printer:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def text(self, value: str) -> None:
        self.texts.append(value)


@pytest.mark.parametrize(
    ("value", "limit", "expected"),
    [
        (None, None, ("", False)),
        (["lung", None, "blood"], None, ("lung, blood", False)),
        (("a", "b"), None, ("a, b", False)),
        (1234567, None, ("1,234,567", False)),
        (True, None, ("True", False)),
        (1.5, None, ("1.5", False)),
        ("two \n  lines", None, ("two lines", False)),
        ("abc defgh", 5, ("abc…", True)),
        ("exact", 5, ("exact", False)),
        ("a|b_c*d", None, ("a&#124;b&#95;c&#42;d", False)),
    ],
)
def test_cell_formats_values(value, limit, expected):
    assert _cell(value, limit) == expected


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "3"])
def test_max_cell_chars_must_be_positive_integers(value):
    with pytest.raises(ValueError, match="max_cell_chars"):
        _validate_max_cell_chars(value)
    with pytest.raises(ValueError, match="max_cell_chars"):
        CatalogResults(ROWS, max_cell_chars=value)


@pytest.mark.parametrize("value", [None, 1, 80])
def test_max_cell_chars_accepts_none_and_positive_integers(value):
    _validate_max_cell_chars(value)


def test_results_behave_like_lists_of_complete_rows():
    results = CatalogResults(ROWS, columns=("cytebase_id",))
    assert results[0] is ROWS[0]
    assert results[:2] == ROWS[:2]
    assert list(results) == ROWS


def test_markdown_uses_headers_alignment_and_display_columns():
    results = CatalogResults(ROWS, columns=("cytebase_id", "cell_count"))
    assert results.to_markdown() == (
        "| Cytebase ID | Cells |\n"
        "| --- | ---: |\n"
        "| lung&#95;a | 1,234 |\n"
        "| lung&#95;b |  |\n"
        "| lung&#95;c | 7 |\n"
        "\n"
        "3 rows."
    )


def test_markdown_defaults_to_every_column_and_titles_unknown_headers():
    text = CatalogResults(ROWS[:1]).to_markdown()
    assert text.splitlines()[:2] == [
        "| Cytebase ID | Cells | Year | First Author |",
        "| --- | ---: | ---: | --- |",
    ]
    assert text.endswith("\n\n1 row.")


def test_markdown_limits_rows_and_reports_shortened_cells():
    results = CatalogResults(ROWS, columns=("first_author",))
    text = results.to_markdown(max_rows=2, max_cell_chars=2)
    assert "| S… |" in text
    assert text.endswith(
        "Showing 2 of 3 rows. Use `to_markdown(max_rows=None)` to display all rows. "
        "Long cells are shortened; use `max_cell_chars=None` to display complete "
        "values. Row dictionaries retain the full values."
    )
    assert results.to_markdown(max_rows=None).endswith("3 rows.")
    assert results.to_markdown(columns=["cytebase_id"], max_rows=0).startswith(
        "| Cytebase ID |"
    )


@pytest.mark.parametrize("max_rows", [True, -1, 1.5])
def test_markdown_rejects_invalid_row_limits(max_rows):
    with pytest.raises(ValueError, match="max_rows"):
        CatalogResults(ROWS).to_markdown(max_rows=max_rows)


def test_markdown_rejects_invalid_columns():
    results = CatalogResults(ROWS)
    with pytest.raises(ValueError, match="max_cell_chars"):
        results.to_markdown(max_cell_chars=0)
    with pytest.raises(TypeError, match="sequence of column names"):
        results.to_markdown(columns="cytebase_id")
    with pytest.raises(ValueError, match="Choose at least one column"):
        results.to_markdown(columns=[])
    with pytest.raises(ValueError, match="Unknown display columns: missing"):
        results.to_markdown(columns=["missing"])


def test_markdown_for_empty_results():
    assert CatalogResults([], columns=("cytebase_id",)).to_markdown() == (
        "No matching rows."
    )


def test_representations_use_the_configured_cell_limit():
    long_title = "A" * 150
    results = CatalogResults([{"title": long_title}], max_cell_chars=None)
    assert long_title in results._repr_markdown_()
    assert repr(results) == str(results) == results._repr_markdown_()
    assert long_title not in repr(CatalogResults([{"title": long_title}]))
    printer = _Printer()
    results._repr_pretty_(printer, cycle=False)
    results._repr_pretty_(printer, cycle=True)
    assert printer.texts == [results._repr_markdown_(), "CatalogResults([...])"]

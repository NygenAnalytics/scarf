"""Compact notebook tables that retain the complete catalog row dictionaries."""

from collections.abc import Iterable, Sequence
from numbers import Integral
from typing import Any

_HEADERS = {
    "cytebase_id": "Cytebase ID",
    "status": "Status",
    "title": "Title",
    "cell_count": "Cells",
    "primary_cell_count": "Primary cells",
    "n_genes": "Genes",
    "tissue_labels": "Tissues",
    "disease_labels": "Diseases",
    "zarr_uri": "Zarr URI",
    "label": "Label",
    "term_id": "Ontology ID",
    "n_datasets": "Datasets",
}
_NUMERIC_COLUMNS = {"cell_count", "primary_cell_count", "n_genes", "n_datasets", "year"}
_MARKDOWN_ESCAPES = str.maketrans(
    {char: f"&#{ord(char)};" for char in "\\`*_{}[]()#|~&<>\"'"}
)


def _cell(value: Any, max_characters: int | None) -> tuple[str, bool]:
    if value is None:
        return "", False
    if isinstance(value, (list, tuple)):
        text = ", ".join(str(item) for item in value if item is not None)
    elif isinstance(value, Integral) and not isinstance(value, bool):
        text = f"{value:,}"
    else:
        text = str(value)
    text = " ".join(text.split())
    shortened = max_characters is not None and len(text) > max_characters
    if shortened and max_characters is not None:
        text = text[: max_characters - 1].rstrip() + "…"
    return text.translate(_MARKDOWN_ESCAPES), shortened


def _validate_max_cell_chars(value: int | None) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or value < 1
    ):
        raise ValueError("max_cell_chars must be a positive integer or None")


class CatalogResults(list[dict[str, Any]]):
    """List-compatible rows with a compact Markdown representation.

    Indexing and iteration expose the original complete row dictionaries. As
    with a normal list, slicing returns a list. ``list(results)`` gives a plain
    list for callers that require the exact built-in type. Display truncation
    never modifies records or changes the query's ordering. ``max_cell_chars``
    sets the notebook and text representation limit; ``None`` shows full values.
    """

    def __init__(
        self,
        rows: Iterable[dict[str, Any]] = (),
        *,
        columns: Sequence[str] = (),
        max_cell_chars: int | None = 100,
    ) -> None:
        _validate_max_cell_chars(max_cell_chars)
        super().__init__(rows)
        self._display_columns = tuple(columns)
        self._max_cell_chars = max_cell_chars

    def to_markdown(
        self,
        columns: Sequence[str] | None = None,
        max_rows: int | None = 20,
        max_cell_chars: int | None = 100,
    ) -> str:
        """Render selected columns, using ``max_rows=None`` to display every row.

        Long cell values are shortened to ``max_cell_chars`` characters. Set
        ``max_cell_chars=None`` to display full label lists, identifiers, and URIs.
        """
        if max_rows is not None and (
            isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 0
        ):
            raise ValueError("max_rows must be a nonnegative integer or None")
        _validate_max_cell_chars(max_cell_chars)
        if isinstance(columns, str):
            raise TypeError("columns must be a sequence of column names")
        available = dict.fromkeys(self._display_columns)
        for row in self:
            available.update(dict.fromkeys(row))
        selected = (
            tuple(columns)
            if columns is not None
            else self._display_columns or tuple(available)
        )
        if columns is not None and not selected:
            raise ValueError("Choose at least one column")
        if not self:
            return "No matching rows."
        unknown = [column for column in selected if column not in available]
        if unknown:
            raise ValueError(f"Unknown display columns: {', '.join(unknown)}")
        headers = [
            _HEADERS.get(column, column.replace("_", " ").title())
            for column in selected
        ]
        lines = [
            "| " + " | ".join(_cell(header, None)[0] for header in headers) + " |",
            "| "
            + " | ".join(
                "---:" if column in _NUMERIC_COLUMNS else "---" for column in selected
            )
            + " |",
        ]
        shown = self if max_rows is None else self[:max_rows]
        shortened = False
        for row in shown:
            values = [_cell(row.get(column), max_cell_chars) for column in selected]
            shortened = shortened or any(truncated for _, truncated in values)
            lines.append("| " + " | ".join(value for value, _ in values) + " |")
        if len(shown) < len(self):
            summary = (
                f"Showing {len(shown):,} of {len(self):,} rows. "
                "Use `to_markdown(max_rows=None)` to display all rows."
            )
        else:
            summary = f"{len(self):,} {'row' if len(self) == 1 else 'rows'}."
        if shortened:
            summary += (
                " Long cells are shortened; use `max_cell_chars=None` to display "
                "complete values. Row dictionaries retain the full values."
            )
        return "\n".join(lines) + "\n\n" + summary

    def _repr_markdown_(self) -> str:
        return self.to_markdown(max_cell_chars=self._max_cell_chars)

    def _repr_pretty_(self, printer: Any, cycle: bool) -> None:
        printer.text("CatalogResults([...])" if cycle else self._repr_markdown_())

    def __repr__(self) -> str:
        return self._repr_markdown_()

    def __str__(self) -> str:
        return self._repr_markdown_()

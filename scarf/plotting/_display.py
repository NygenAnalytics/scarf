"""Resolve stored display metadata into plotting scale contracts."""

from typing import Any

from ..metadata.artifacts import column_display, validate_display_metadata
from ._contracts import CategoricalScale


def stored_display_metadata(store: Any, column: str) -> dict[str, Any] | None:
    """Return validated display metadata for a cell column, when available."""
    frozen_display = getattr(store, "_stored_display_metadata", None)
    if callable(frozen_display):
        display = frozen_display(column)
        return None if display is None else validate_display_metadata(display)
    try:
        root = store.zw
    except AttributeError:
        return None
    return column_display(root, column)


def stored_categorical_scale(store: Any, column: str) -> CategoricalScale | None:
    """Resolve a stored categorical display contract for one cell column."""
    display = stored_display_metadata(store, column)
    if display is None or display["kind"] != "categorical":
        return None
    categories = display["categories"]
    return CategoricalScale(
        order=tuple(category["value"] for category in categories),
        palette={category["value"]: str(category["color"]) for category in categories},
        labels={category["value"]: str(category["label"]) for category in categories},
        missing_color=str(display.get("missing_color", "#bdbdbd")),
        missing_label=str(display.get("missing_label", "NA")),
    )


def resolve_categorical_scale(
    store: Any,
    column: str,
    explicit: CategoricalScale | None,
) -> CategoricalScale | None:
    """Prefer an explicit scale, then stored cell display metadata."""
    return explicit if explicit is not None else stored_categorical_scale(store, column)

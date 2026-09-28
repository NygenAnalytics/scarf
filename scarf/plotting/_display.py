"""Resolve stored display metadata into plotting scale contracts."""

from typing import Any

from ..metadata.artifacts import column_display, validate_display_metadata
from ._contracts import CategoricalScale, ColorScale


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


def categorical_display_scale(display: dict[str, Any]) -> CategoricalScale:
    """Build the categorical scale a stored display contract describes."""
    categories = display["categories"]
    return CategoricalScale(
        order=tuple(category["value"] for category in categories),
        palette={category["value"]: str(category["color"]) for category in categories},
        labels={category["value"]: str(category["label"]) for category in categories},
        missing_color=str(display.get("missing_color", "#bdbdbd")),
        missing_label=str(display.get("missing_label", "NA")),
    )


def continuous_display_scale(display: dict[str, Any]) -> ColorScale:
    """Build the color scale a stored continuous display contract describes.

    The stored minimum and maximum become fixed limits only when both exist
    and span a range; otherwise limits come from the plotted values.
    """
    minimum = display["minimum"]
    maximum = display["maximum"]
    fixed = minimum is not None and maximum is not None and maximum > minimum
    return ColorScale(
        cmap=str(display["colormap"]),
        vmin=float(minimum) if fixed else None,
        vmax=float(maximum) if fixed else None,
        scale=display["scale"],
    )


def stored_categorical_scale(store: Any, column: str) -> CategoricalScale | None:
    """Resolve a stored categorical display contract for one cell column."""
    display = stored_display_metadata(store, column)
    if display is None or display["kind"] != "categorical":
        return None
    return categorical_display_scale(display)


def resolve_categorical_scale(
    store: Any,
    column: str,
    explicit: CategoricalScale | None,
) -> CategoricalScale | None:
    """Prefer an explicit scale, then stored cell display metadata."""
    return explicit if explicit is not None else stored_categorical_scale(store, column)

"""Names of Zarr-backed metadata columns.

Zarr reads ``/`` and ``\\`` in an array name as path separators, so a column
named with either one would nest into groups instead of forming one array.
Imports store such source columns under a key with ``_`` in place of each
separator. Scarf's own ``I``, ``ids``, and ``names`` columns and the
missing-value masks linked through ``__scarf_missing__<name>`` are reserved.
"""

from collections.abc import Iterable

from .arrays import MISSING_MASK_PREFIX

RESERVED_METADATA_COLUMNS = frozenset({"I", "ids", "names"})
"""Columns that Scarf writes into every cell and feature table."""

_UNSTORABLE_NAMES = frozenset({"", ".", ".."})


def is_reserved_metadata_name(name: str) -> bool:
    """Return whether a column name collides with Scarf's own columns."""
    return name in RESERVED_METADATA_COLUMNS or name.startswith(MISSING_MASK_PREFIX)


def metadata_column_key(name: str) -> str:
    """Return ``name`` with each ``/`` and ``\\`` replaced by ``_``."""
    return name.replace("/", "_").replace("\\", "_")


def is_metadata_column_key(name: object) -> bool:
    """Return whether ``name`` names one metadata array as it stands."""
    return (
        isinstance(name, str)
        and name not in _UNSTORABLE_NAMES
        and metadata_column_key(name) == name
    )


def nested_group_error(column: str) -> TypeError:
    """Return the error for a table member that is a group, not a column.

    Stores written before Scarf renamed imported columns nested a source
    column whose name contains ``/`` or ``\\`` into groups.
    """
    return TypeError(
        f"The metadata table holds a nested group named {column!r} instead of "
        "a column. A source column name containing '/' or '\\' created it; "
        "import the source again to store that column under a name with '_' "
        "in their place."
    )


def validate_metadata_column_name(name: str) -> None:
    """Reject a metadata column name that Zarr cannot store as one array.

    Names that start with the missing-value mask prefix are rejected too, so
    a written column can never replace or orphan a linked mask.

    Args:
        name: Column name to check.

    Raises:
        TypeError: If ``name`` is not a string.
        ValueError: If ``name`` is empty, ``.``, or ``..``, contains ``/`` or
            ``\\``, or starts with ``__scarf_missing__``.
    """
    if not isinstance(name, str):
        raise TypeError(
            f"Metadata column names must be strings, not {type(name).__name__}"
        )
    if name in _UNSTORABLE_NAMES:
        raise ValueError(f"Metadata column name {name!r} cannot name a Zarr array")
    key = metadata_column_key(name)
    if key.startswith(MISSING_MASK_PREFIX):
        raise ValueError(
            f"Metadata column name {name!r} uses the prefix "
            f"{MISSING_MASK_PREFIX!r}, which Scarf reserves for missing-value masks"
        )
    if key != name:
        raise ValueError(
            f"Metadata column name {name!r} must not contain '/' or '\\' because "
            f"Zarr reads them as path separators; use {key!r} instead"
        )


def metadata_column_keys(
    names: Iterable[str],
    *,
    taken: Iterable[str] = (),
) -> dict[str, str]:
    """Plan the key that stores each source metadata column.

    Source names that are already valid keys keep their name. Each name that
    contains ``/`` or ``\\`` takes :func:`metadata_column_key` of itself, or,
    when that key is in use, the first free key among ``<key>_2``,
    ``<key>_3``, and so on, in source order. A name is left out when it or its
    key is reserved, when it cannot name a Zarr array, or when it is valid but
    already in ``taken``.

    Args:
        names: Source column names in source order. Repeated names are planned
            once.
        taken: Names already present in the destination table.

    Returns:
        The key of each planned source name, in source order.

    Raises:
        TypeError: If a name is not a string.
    """
    sources = list(dict.fromkeys(names))
    for name in sources:
        if not isinstance(name, str):
            raise TypeError(
                f"Metadata column names must be strings, not {type(name).__name__}"
            )
    used = set(taken)
    keys: dict[str, str] = {}
    for name in sources:
        if metadata_column_key(name) != name:
            continue
        if name in _UNSTORABLE_NAMES or is_reserved_metadata_name(name):
            continue
        if name in used:
            continue
        keys[name] = name
        used.add(name)
    for name in sources:
        base = metadata_column_key(name)
        if base == name or is_reserved_metadata_name(base):
            continue
        key, suffix = base, 2
        while key in used or is_reserved_metadata_name(key):
            key, suffix = f"{base}_{suffix}", suffix + 1
        keys[name] = key
        used.add(key)
    return {name: keys[name] for name in sources if name in keys}

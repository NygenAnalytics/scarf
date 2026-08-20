"""Shared assay-type classification used by writers, merge, repack, and load."""

from collections.abc import Mapping
from typing import Any


def preset_assay_types() -> dict[str, type]:
    """Return the DataStore assay-type preset map (single source of truth).

    Returns:
        Mapping from preset type strings to assay classes.
    """
    from .adt import ADTassay
    from .atac import ATACassay
    from .base import Assay
    from .rna import RNAassay

    return {
        "RNA": RNAassay,
        "ATAC": ATACassay,
        "ADT": ADTassay,
        "HTO": ADTassay,
        "CRISPR": Assay,
        "ANTIGEN": Assay,
        "CUSTOM": Assay,
        "GeneActivity": RNAassay,
        "GeneScores": RNAassay,
        "URNA": RNAassay,
        "Assay": Assay,
    }


def rna_assay_type_names() -> frozenset[str]:
    """Return preset type strings that map to ``RNAassay``.

    Returns:
        Names such as ``RNA``, ``GeneActivity``, ``GeneScores``, and ``URNA``.
    """
    from .rna import RNAassay

    return frozenset(
        name
        for name, assay_cls in preset_assay_types().items()
        if issubclass(assay_cls, RNAassay)
    )


def resolve_persisted_assay_type(
    assay_name: str,
    assay_type: str | None = None,
) -> str:
    """Return a preset key safe to store in ``assayTypes``.

    Unknown assay names become ``Assay`` unless ``assay_type`` is an explicit
    recognized preset (for example declaring a custom group as ``RNA``).
    Unrecognized ``assay_type`` values also become ``Assay``.

    Args:
        assay_name: Assay group name in the store.
        assay_type: Optional explicit preset to persist. Unrecognized values
                    become ``Assay``.

    Returns:
        A key present in :func:`preset_assay_types`.
    """
    presets = preset_assay_types()
    if assay_type is not None:
        return assay_type if assay_type in presets else "Assay"
    if assay_name in presets:
        return assay_name
    return "Assay"


def lookup_persisted_assay_type(
    assay_name: str,
    assay_types: Mapping[str, Any] | None = None,
    *,
    assay_type: str | None = None,
) -> str:
    """Resolve a persisted type from an explicit value or an ``assayTypes`` map.

    Preference order: ``assay_type``, then ``assay_types[assay_name]``, then
    ``assay_name`` when it is a recognized preset.

    Args:
        assay_name: Assay group name in the store.
        assay_types: Optional persisted ``assayTypes`` mapping.
        assay_type: Optional explicit preset that wins over the mapping.

    Returns:
        A key present in :func:`preset_assay_types`.
    """
    if assay_type is not None:
        return resolve_persisted_assay_type(assay_name, assay_type)
    if assay_types is not None and assay_name in assay_types:
        return resolve_persisted_assay_type(assay_name, str(assay_types[assay_name]))
    return resolve_persisted_assay_type(assay_name)


def is_rna_assay_type(name_or_type: str | type | Any) -> bool:
    """Return True when the value names or is an RNA-class assay.

    Accepts:
    - preset type strings (``"RNA"``, ``"GeneActivity"``, …)
    - assay class objects (``RNAassay`` and subclasses)
    - assay instances (``isinstance(..., RNAassay)``)

    Args:
        name_or_type: Preset string, assay class, or assay instance.

    Returns:
        True when the value is an RNA-class assay.
    """
    from .rna import RNAassay

    if isinstance(name_or_type, str):
        assay_cls = preset_assay_types().get(name_or_type)
        return assay_cls is not None and issubclass(assay_cls, RNAassay)
    if isinstance(name_or_type, type):
        return issubclass(name_or_type, RNAassay)
    return isinstance(name_or_type, RNAassay)

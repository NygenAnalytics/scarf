"""Shared assay-type classification used by writers, merge, repack, and load."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import zarr

from ..storage.types import as_zarr_array


def default_feature_sets(assay: zarr.Group) -> list[np.ndarray]:
    """Return the features of each default RNA percentage gene family.

    Writers total these sets while transposing counts, so preparation with the
    default patterns needs no second read of the matrix.
    """
    from ..features.gene_families import PERCENT_FAMILIES, gene_family_mask

    names = as_zarr_array(assay["featureData/names"], name="featureData/names")
    values = np.asarray(names[:]).astype(str)
    return [
        np.flatnonzero(gene_family_mask(values, family))
        for family in PERCENT_FAMILIES.values()
    ]


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


def _preset_guidance() -> str:
    """Name every preset and the explicit choice for a generic assay."""
    names = ", ".join(repr(name) for name in preset_assay_types() if name != "Assay")
    return (
        f"Use one of {names}, or 'Assay' for a generic assay without "
        "modality-specific normalization."
    )


def validate_assay_type(assay_type: Any, *, assay: str | None = None) -> None:
    """Raise ``ValueError`` unless ``assay_type`` is None or a preset."""
    if assay_type is None:
        return
    if isinstance(assay_type, str) and assay_type in preset_assay_types():
        return
    subject = f"assay_type {assay_type!r}"
    if assay is not None:
        subject += f" of assay {assay!r}"
    raise ValueError(
        f"{subject} is not a preset; names are case-sensitive. {_preset_guidance()}"
    )


def validate_assay_types(
    assay_types: Mapping[str, Any] | None,
    assay_names: Sequence[str],
) -> dict[str, str]:
    """Return ``assay_types`` as a new dict after checking its assays and presets."""
    if assay_types is None:
        return {}
    if not isinstance(assay_types, Mapping):
        raise TypeError(
            "assay_types must be a mapping from assay name to preset type, not "
            f"{type(assay_types).__name__}"
        )
    explicit = dict(assay_types)
    unknown = [name for name in explicit if name not in assay_names]
    if unknown:
        raise ValueError(
            "assay_types names assays that are not in the store: "
            + ", ".join(repr(name) for name in unknown)
            + ". Assays in the store: "
            + ", ".join(repr(name) for name in assay_names)
        )
    for name, assay_type in explicit.items():
        validate_assay_type(assay_type, assay=name)
    return explicit


def recorded_assay_types(
    record: Any,
    assay_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return the ``assayTypes`` record of a root or workspace as a mapping.

    The values are not checked against the presets.
    """
    if record is None:
        return {}
    if isinstance(record, Mapping):
        return {str(name): value for name, value in record.items()}
    if assay_names:
        example = ", ".join(f"{name!r}: <preset>" for name in assay_names)
        remedy = f"for every assay, assay_types={{{example}}}"
    else:
        remedy = "for every assay of the store"
    raise ValueError(
        f"The assayTypes attribute of the store is {record!r}, which is not a "
        "mapping from assay name to preset type, so it declares no assay type. "
        "Open the store with zarr_mode='r+' and an assay_types that names a "
        f"preset {remedy}, to replace it. {_preset_guidance()}"
    )


def resolve_persisted_assay_type(
    assay_name: str,
    assay_type: str | None = None,
) -> str:
    """Return a preset key safe to store in ``assayTypes``.

    Unknown assay names become ``Assay`` unless ``assay_type`` is an explicit
    recognized preset (for example declaring a custom group as ``RNA``).
    Unrecognized ``assay_type`` values raise ``ValueError``.

    Args:
        assay_name: Assay group name in the store.
        assay_type: Optional explicit preset to persist.

    Returns:
        A key present in :func:`preset_assay_types`.
    """
    if assay_type is not None:
        validate_assay_type(assay_type, assay=assay_name)
        return assay_type
    if assay_name in preset_assay_types():
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

    Raises:
        ValueError: If the explicit or recorded type is not a preset.
    """
    if assay_type is not None:
        return resolve_persisted_assay_type(assay_name, assay_type)
    if assay_types is not None and assay_name in assay_types:
        recorded = assay_types[assay_name]
        if isinstance(recorded, str) and recorded in preset_assay_types():
            return recorded
        raise ValueError(
            f"Assay {assay_name!r} is recorded in assayTypes as {recorded!r}, "
            "which is not a preset. Open the store with zarr_mode='r+' and "
            f"assay_types={{{assay_name!r}: <preset>}} to record its type. "
            f"{_preset_guidance()}"
        )
    return resolve_persisted_assay_type(assay_name)


def declared_assay_type(assay: Any) -> str:
    """Return the preset type that an open assay's store declares for it."""
    declared = getattr(assay, "assayType", None)
    if not isinstance(declared, str):
        raise ValueError(
            f"Assay {assay.name!r} carries no declared type. Use an assay of an "
            "open DataStore, or construct it with an explicit assay_type."
        )
    return declared


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

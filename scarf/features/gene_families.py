"""Gene families that Scarf recognizes by feature name.

Each family is a regular expression whose alternatives are anchored at the
start of a feature name. Families match case-insensitively, as the default HVG
blacklist and the default RNA percentage columns apply them, so mouse ``mt-``
genes are mitochondrial.

``ribosomal`` covers cytosolic and mitochondrial ribosomal protein genes (RPS,
RPL, MRPS, and MRPL), as the default ``percentRibo`` column does.
``mitochondrial`` covers ``MT-`` genes, as the default ``percentMito`` column
does. ``sexLinked`` lists ten sex-linked genes by exact name. The default HVG
blacklist excludes every registered family, in registry order.
"""

from collections.abc import Sequence
from types import MappingProxyType
from typing import Any

import numpy as np

from ..utils.arrays import regex_match_mask

__all__ = [
    "DEFAULT_PERCENT_PATTERNS",
    "GENE_FAMILY_PATTERNS",
    "PERCENT_FAMILIES",
    "gene_family_mask",
]

# Case-insensitive name pattern of each registered gene family.
GENE_FAMILY_PATTERNS = MappingProxyType(
    {
        "mitochondrial": "^MT-",
        "ribosomal": "^RPS|^RPL|^MRPS|^MRPL",
        "cellCycleCcn": "^CCN",
        "hla": "^HLA-",
        "h2": "^H2-",
        "histone": "^HIST",
        "sexLinked": (
            "^XIST$|^DDX3Y$|^USP9Y$|^EIF1AY$|^KDM5D$|^SRY$|^ZFY$|^UTY$|"
            "^TMSB4Y$|^NLGN4Y$"
        ),
    }
)
# The gene family that each default RNA percentage column measures.
PERCENT_FAMILIES = MappingProxyType(
    {"percentMito": "mitochondrial", "percentRibo": "ribosomal"}
)
# The feature-name pattern of each default RNA percentage column.
DEFAULT_PERCENT_PATTERNS = MappingProxyType(
    {
        suffix: GENE_FAMILY_PATTERNS[family]
        for suffix, family in PERCENT_FAMILIES.items()
    }
)


def gene_family_mask(names: Sequence[Any] | np.ndarray, family: str) -> np.ndarray:
    """Return where feature names belong to one registered gene family.

    Args:
        names: Feature names. Each value is matched as text.
        family: A key of ``GENE_FAMILY_PATTERNS``.

    Returns:
        A boolean array aligned with ``names``.

    Raises:
        ValueError: If ``family`` is not registered.
    """
    pattern = GENE_FAMILY_PATTERNS.get(family)
    if pattern is None:
        raise ValueError(f"Unknown gene family {family!r}")
    return regex_match_mask(names, pattern)

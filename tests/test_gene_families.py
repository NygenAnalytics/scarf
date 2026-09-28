"""One registry defines the gene families that Scarf matches by feature name."""

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.assay.classification import default_feature_sets
from scarf.features.gene_families import (
    DEFAULT_PERCENT_PATTERNS,
    GENE_FAMILY_PATTERNS,
    PERCENT_FAMILIES,
    gene_family_mask,
)
from scarf.features.variability import DEFAULT_HVG_BLACKLIST
from scarf.utils.arrays import regex_match_mask

_NAMES = np.asarray(
    [
        "MT-CO1",
        "mt-Nd1",
        "MTOR",
        "RPS3",
        "Rpl13",
        "MRPS5",
        "Mrpl3",
        "CCNB1",
        "HLA-A",
        "H2-K1",
        "HIST1H1C",
        "XIST",
        "Uty",
        "XISTR",
        "ACTB",
    ]
)


def test_default_blacklist_is_the_union_of_every_registered_family() -> None:
    # HVG selections record the blacklist text, so it must not drift.
    assert DEFAULT_HVG_BLACKLIST == (
        "^MT-|^RPS|^RPL|^MRPS|^MRPL|^CCN|^HLA-|^H2-|^HIST|"
        "^XIST$|^DDX3Y$|^USP9Y$|^EIF1AY$|^KDM5D$|^SRY$|^ZFY$|^UTY$|^TMSB4Y$|^NLGN4Y$"
    )
    union = np.logical_or.reduce(
        [gene_family_mask(_NAMES, family) for family in GENE_FAMILY_PATTERNS]
    )
    np.testing.assert_array_equal(
        union, regex_match_mask(_NAMES, DEFAULT_HVG_BLACKLIST)
    )


def test_default_percentages_measure_the_registered_families() -> None:
    assert dict(PERCENT_FAMILIES) == {
        "percentMito": "mitochondrial",
        "percentRibo": "ribosomal",
    }
    assert dict(DEFAULT_PERCENT_PATTERNS) == {
        "percentMito": "^MT-",
        "percentRibo": "^RPS|^RPL|^MRPS|^MRPL",
    }


def test_families_match_names_case_insensitively_from_the_start() -> None:
    expected = {
        "mitochondrial": {"MT-CO1", "mt-Nd1"},
        "ribosomal": {"RPS3", "Rpl13", "MRPS5", "Mrpl3"},
        "cellCycleCcn": {"CCNB1"},
        "hla": {"HLA-A"},
        "h2": {"H2-K1"},
        "histone": {"HIST1H1C"},
        "sexLinked": {"XIST", "Uty"},
    }
    assert set(expected) == set(GENE_FAMILY_PATTERNS)
    for family, members in expected.items():
        assert set(_NAMES[gene_family_mask(_NAMES, family)]) == members
    with pytest.raises(ValueError, match="Unknown gene family"):
        gene_family_mask(_NAMES, "mitoribosomal")


def test_default_feature_sets_follow_the_percentage_families() -> None:
    group = zarr.open_group(store=MemoryStore(), mode="w")
    group.create_array("featureData/names", data=_NAMES.astype("U"))
    mito, ribo = default_feature_sets(group)
    np.testing.assert_array_equal(mito, [0, 1])
    np.testing.assert_array_equal(ribo, [3, 4, 5, 6])

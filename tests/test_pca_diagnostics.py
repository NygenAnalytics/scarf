"""Tests for pure PCA diagnostic helpers."""

import numpy as np

from scarf.metrics.pca_diagnostics import (
    branch_nuisance_summary,
    family_loading_concentration,
    per_pc_covariate_associations,
)


def test_per_pc_covariate_associations_flags_floor() -> None:
    coordinates = np.column_stack(
        [
            np.linspace(0.0, 1.0, 20),
            np.random.default_rng(0).normal(size=20),
        ]
    )
    batch = np.array(["A"] * 10 + ["B"] * 10)
    reports = per_pc_covariate_associations(
        coordinates,
        {"batch": batch},
        columnKinds={"batch": "categorical"},
        associationFloor=0.1,
    )
    assert len(reports) == 2
    assert reports[0]["pc"] == 1
    assert reports[0]["flagged"] is True
    assert reports[0]["association"]["status"] == "ok"


def test_signed_spearman_uses_absolute_strength() -> None:
    coordinates = np.column_stack(
        [
            np.linspace(0.0, 1.0, 30),
            np.random.default_rng(1).normal(size=30),
        ]
    )
    anti = -coordinates[:, 0]
    reports = per_pc_covariate_associations(
        coordinates,
        {"score": anti},
        columnKinds={"score": "continuous"},
        associationFloor=0.1,
    )
    assert reports[0]["association"]["measure"] == "spearmanRho"
    assert reports[0]["association"]["value"] < 0
    assert reports[0]["strength"] == abs(reports[0]["association"]["value"])
    assert reports[0]["flagged"] is True
    summary = branch_nuisance_summary(
        reports,
        technicalCovariates=["score"],
        associationFloor=0.1,
    )
    assert summary["technical"]["meanAssociation"] > 0.5
    assert summary["technical"]["nFlaggedPcs"] >= 1


def test_per_pc_associations_read_coordinate_columns_incrementally() -> None:
    values = np.column_stack(
        [
            np.linspace(0.0, 1.0, 20),
            np.linspace(1.0, 0.0, 20),
        ]
    )

    class ColumnSource:
        shape = values.shape

        def __array__(self):
            raise AssertionError("coordinate matrix must not be materialized")

        def __getitem__(self, key):
            return values[key]

    reports = per_pc_covariate_associations(
        ColumnSource(),
        {"score": values[:, 0]},
        columnKinds={"score": "continuous"},
    )

    assert len(reports) == values.shape[1]


def test_family_loading_concentration_and_branch_summary() -> None:
    loadings = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    reports = family_loading_concentration(
        loadings,
        featureIndexes=[10, 11, 12, 13],
        familyIndexes={"mitochondrial": [10, 11], "sex": [12]},
        featureNames=["MT-A", "MT-B", "XIST", "ACTB"],
        topN=2,
    )
    assert reports[0]["familyShares"]["mitochondrial"] == 1.0
    assert reports[0]["topLoadings"][0]["featureName"] in {"MT-A", "MT-B"}
    assert reports[1]["familyShares"]["sex"] == 0.5

    associations = [
        {
            "pc": 1,
            "covariate": "batch",
            "association": {"status": "ok", "value": 0.4},
        },
        {
            "pc": 2,
            "covariate": "batch",
            "association": {"status": "ok", "value": 0.05},
        },
        {
            "pc": 1,
            "covariate": "condition",
            "association": {"status": "ok", "value": 0.3},
        },
    ]
    summary = branch_nuisance_summary(
        associations,
        technicalCovariates=["batch"],
        protectedCovariates=["condition"],
        associationFloor=0.1,
    )
    assert summary["technical"]["nFlaggedPcs"] == 1
    assert summary["technical"]["flaggedPcs"] == [1]
    assert summary["protected"]["meanAssociation"] == 0.3

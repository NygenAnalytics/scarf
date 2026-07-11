from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from scarf.mapping_reference import (
    LATEST_MAPPING_REFERENCE_ATTRIBUTE,
    MAPPING_REFERENCE_GROUP,
    MAPPING_REFERENCES_GROUP,
    load_mapping_reference,
    persist_mapping_reference,
)
from scarf.symphony import SymphonyReferenceModel


def _model() -> SymphonyReferenceModel:
    return SymphonyReferenceModel(
        feature_means=np.array([1.0, 2.0]),
        feature_scales=np.array([0.5, 2.0]),
        loadings=np.eye(2),
        centroids=np.array([[1.0, 0.0], [0.0, 1.0]]),
        raw_centroids=np.array([[1.0, 1.0], [2.0, 2.0]]),
        corrected_centroids=np.array([[0.5, 1.0], [1.5, 2.0]]),
        cluster_mass=np.array([5.0, 4.0]),
        sigma=np.array([0.1, 0.2]),
        correction_ridge=1.0,
    )


def test_mapping_reference_roundtrip(tmp_path):
    root = zarr.open_group(str(tmp_path / "reference.zarr"), mode="w")
    reduction = root.create_group("RNA/reduction")
    metadata = {
        "assay": "RNA",
        "cellKey": "I",
        "featureKey": "hvgs",
        "reductionPath": "RNA/reduction",
        "annPath": "RNA/reduction/ann",
        "featureHash": "features",
        "cellHash": "cells",
        "batchValueHash": "batches",
        "batchColumns": ["batch"],
        "subsetHash": 1,
        "subsetParams": {"log_transform": True},
        "loadingsHash": "loadings",
        "reductionMethod": "pca",
    }
    artifact_path = persist_mapping_reference(
        reduction,
        _model(),
        np.array(["gene_a", "gene_b"]),
        metadata,
        reference_distance_quantiles=np.array([0.0, 1.0]),
        reference_distance_values=np.array([0.1, 2.0]),
    )

    reference = load_mapping_reference(
        SimpleNamespace(zw=root),
        "RNA",
        "I",
        "hvgs",
        "RNA/reduction",
        "RNA/reduction/ann",
    )

    assert artifact_path.startswith(f"{MAPPING_REFERENCES_GROUP}/")
    assert f"RNA/reduction/{artifact_path}" in root
    assert (
        reduction.attrs[LATEST_MAPPING_REFERENCE_ATTRIBUTE]
        == artifact_path.rsplit("/", 1)[-1]
    )
    np.testing.assert_array_equal(reference.feature_ids, ["gene_a", "gene_b"])
    np.testing.assert_allclose(
        reference.model.corrected_centroids, _model().corrected_centroids
    )
    np.testing.assert_allclose(reference.reference_distance_values, [0.1, 2.0])
    assert reference.metadata["algorithmVariant"] == "symphonyStyleV1"

    repeated_path = persist_mapping_reference(
        reduction,
        _model(),
        np.array(["gene_a", "gene_b"]),
        metadata,
        reference_distance_quantiles=np.array([0.0, 1.0]),
        reference_distance_values=np.array([0.1, 2.0]),
    )
    assert repeated_path == artifact_path
    assert len(reduction[MAPPING_REFERENCES_GROUP]) == 1

    reduction[artifact_path]["loadings"][0, 0] = 2.0
    with pytest.raises(ValueError, match="artifact hash"):
        load_mapping_reference(
            SimpleNamespace(zw=root),
            "RNA",
            "I",
            "hvgs",
            "RNA/reduction",
            "RNA/reduction/ann",
        )


def test_legacy_mapping_reference_remains_readable(tmp_path):
    root = zarr.open_group(str(tmp_path / "legacy_reference.zarr"), mode="w")
    reduction = root.create_group("RNA/reduction")
    legacy = reduction.create_group(MAPPING_REFERENCE_GROUP)
    legacy.attrs["schemaVersion"] = 1
    legacy.attrs["complete"] = True
    legacy.attrs["correctionRidge"] = 1.0
    legacy.attrs["batchColumns"] = ["batch"]
    model = _model()
    for name, values in {
        "featureMeans": model.feature_means,
        "featureScales": model.feature_scales,
        "loadings": model.loadings,
        "centroids": model.centroids,
        "rawCentroids": model.raw_centroids,
        "correctedCentroids": model.corrected_centroids,
        "clusterMass": model.cluster_mass,
        "sigma": model.sigma,
        "featureIds": np.array(["gene_a", "gene_b"]),
    }.items():
        legacy.create_array(name, data=values)

    with pytest.warns(DeprecationWarning, match="legacy"):
        reference = load_mapping_reference(
            SimpleNamespace(zw=root),
            "RNA",
            "I",
            "hvgs",
            "RNA/reduction",
            "RNA/reduction/ann",
        )

    assert reference.artifact_path.endswith(MAPPING_REFERENCE_GROUP)


def test_incomplete_content_addressed_reference_is_rejected(tmp_path):
    root = zarr.open_group(str(tmp_path / "incomplete_reference.zarr"), mode="w")
    reduction = root.create_group("RNA/reduction")
    artifact_path = persist_mapping_reference(
        reduction,
        _model(),
        np.array(["gene_a", "gene_b"]),
        {
            "assay": "RNA",
            "cellKey": "I",
            "featureKey": "hvgs",
            "reductionPath": "RNA/reduction",
            "annPath": "RNA/reduction/ann",
            "batchColumns": ["batch"],
        },
    )
    reduction[artifact_path].attrs["complete"] = False

    with pytest.raises(ValueError, match="incomplete"):
        load_mapping_reference(
            SimpleNamespace(zw=root),
            "RNA",
            "I",
            "hvgs",
            "RNA/reduction",
            "RNA/reduction/ann",
        )

from dataclasses import fields

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.graph.arguments import (
    AnnIndexArguments,
    ConnectivityMapArguments,
    CustomReductionArguments,
    EmbeddingInitializationArguments,
    FeatureScalingArguments,
    HarmonyArguments,
    LsiArguments,
    NeighborQueryArguments,
    NormalizationArguments,
    PcaArguments,
    artifact_input,
    execution,
    parameter,
)
from scarf.storage.artifact_writer import finish_artifact, start_artifact
from scarf.storage.artifacts import ArtifactRef


def _ref(
    kind: str,
    token: str,
    *,
    scope: str = "assay",
) -> ArtifactRef:
    return ArtifactRef(
        scope=scope,  # type: ignore[arg-type]
        assay="RNA" if scope == "assay" else None,
        kind=kind,
        artifact_id=token * 64,
    )


def test_normalization_arguments_partition_every_value() -> None:
    arguments = NormalizationArguments(
        from_assay="RNA",
        cell_key="I",
        cell_selection=_ref("cell_selection", "2", scope="datastore"),
        feature_selection=_ref("feature_selection", "3"),
        normalization_method="norm_lib_size",
        size_factor=1000.0,
        log_transform=True,
        renormalize_subset=False,
        update_state=True,
        invalidate_cache=False,
    )
    record = arguments.to_record()

    assert set(record.inputs) == {"cell_selection", "feature_selection"}
    assert record.parameters == {
        "normalization_method": "norm_lib_size",
        "size_factor": 1000.0,
        "log_transform": True,
        "renormalize_subset": False,
    }
    assert record.execution_options == {
        "from_assay": "RNA",
        "cell_key": "I",
        "update_state": True,
        "invalidate_cache": False,
    }


def test_execution_options_do_not_change_provenance_hash() -> None:
    common = {
        "from_assay": "RNA",
        "cell_key": "I",
        "cell_selection": _ref("cell_selection", "2", scope="datastore"),
        "feature_selection": _ref("feature_selection", "3"),
        "normalization_method": "norm_lib_size",
        "size_factor": 1000.0,
        "log_transform": True,
        "renormalize_subset": False,
        "update_state": True,
    }
    small_batch = NormalizationArguments(
        invalidate_cache=False,
        **common,
    )
    invalidated = NormalizationArguments(
        invalidate_cache=True,
        **common,
    )
    changed_parameter = NormalizationArguments(
        invalidate_cache=False,
        **(common | {"log_transform": False}),
    )
    changed_size_factor = NormalizationArguments(
        invalidate_cache=False,
        **(common | {"size_factor": 2000.0}),
    )

    assert small_batch.provenance_hash() == invalidated.provenance_hash()
    assert small_batch.provenance_hash() != changed_parameter.provenance_hash()
    assert small_batch.provenance_hash() != changed_size_factor.provenance_hash()


def test_reduction_fingerprints_custom_loadings_as_input() -> None:
    common = {
        "normalized": _ref("normalized", "a"),
        "feature_scaling": _ref("feature_scaling", "c"),
        "dims": 2,
        "feat_scaling": False,
        "update_state": False,
        "invalidate_cache": False,
    }
    first = CustomReductionArguments(
        loadings=np.arange(6, dtype=np.float64).reshape(3, 2),
        **common,
    )
    copied = CustomReductionArguments(
        loadings=np.arange(6, dtype=np.float64).reshape(3, 2),
        **common,
    )
    changed = CustomReductionArguments(
        loadings=np.ones((3, 2), dtype=np.float64),
        **common,
    )

    assert first.provenance_hash() == copied.provenance_hash()
    assert first.provenance_hash() != changed.provenance_hash()
    assert "value_fingerprint" in first.to_record().inputs["loadings"]


def test_stage_models_chain_logical_artifact_refs() -> None:
    scaling = FeatureScalingArguments(
        normalized=_ref("normalized", "0"),
        enabled=True,
        batch_size=100,
    )
    harmony = HarmonyArguments(
        reduction=_ref("reduction", "1"),
        batch_values=_ref("metadata_snapshot", "2", scope="datastore"),
        batch_columns=("donor", "sample"),
        harmony_parameters={"theta": 2.0},
        algorithm_version="centroid_snapshot_v2",
        batch_size=100,
    )
    neighbors = NeighborQueryArguments(
        ann_index=_ref("ann_index", "3"),
        coordinates=_ref("batch_correction", "4"),
        k=15,
        distance_metric="l2",
        batch_size=100,
    )
    connectivity = ConnectivityMapArguments(
        neighbors=_ref("neighbors", "5"),
        local_connectivity=1.0,
        bandwidth=1.5,
    )

    assert harmony.to_record().parameters["batch_columns"] == ["donor", "sample"]
    assert scaling.to_record().parameters == {"enabled": True}
    assert neighbors.to_record().inputs["ann_index"]["kind"] == "ann_index"
    assert connectivity.to_record().parameters == {
        "local_connectivity": 1.0,
        "bandwidth": 1.5,
    }


def test_reduction_batch_size_does_not_change_reuse() -> None:
    normalized = _ref("normalized", "0")
    scaling_small = FeatureScalingArguments(
        normalized=normalized,
        enabled=False,
        batch_size=100,
    )
    scaling_large = FeatureScalingArguments(
        normalized=normalized,
        enabled=False,
        batch_size=500,
    )
    lsi_common = {
        "normalized": normalized,
        "feature_scaling": _ref("feature_scaling", "1"),
        "dims": 5,
        "skip_first": True,
        "rand_state": 4466,
        "solver": "streaming",
        "n_iter": 5,
        "n_oversamples": 10,
        "update_state": False,
        "invalidate_cache": False,
    }
    lsi_small = LsiArguments(batch_size=100, **lsi_common)
    lsi_large = LsiArguments(batch_size=500, **lsi_common)
    pca_common = {
        "normalized": normalized,
        "feature_scaling": _ref("feature_scaling", "1"),
        "pca_cell_selection": _ref("cell_selection", "2", scope="datastore"),
        "pca_cell_key": "I",
        "dims": 5,
        "feat_scaling": True,
        "show_elbow_plot": False,
        "update_state": False,
        "invalidate_cache": False,
    }
    pca_small = PcaArguments(batch_size=100, **pca_common)
    pca_large = PcaArguments(batch_size=500, **pca_common)

    assert scaling_small.provenance_hash() == scaling_large.provenance_hash()
    assert lsi_small.provenance_hash() == lsi_large.provenance_hash()
    assert pca_small.provenance_hash() == pca_large.provenance_hash()


def test_embedding_initialization_parameters_change_provenance() -> None:
    initialization_common = {
        "reduction": _ref("reduction", "3"),
        "n_centroids": 20,
        "rand_state": 4466,
        "invalidate_cache": False,
    }
    initialization_small = EmbeddingInitializationArguments(
        batch_size=100,
        **initialization_common,
    )
    initialization_large = EmbeddingInitializationArguments(
        batch_size=500,
        **initialization_common,
    )
    initialization_sampled = EmbeddingInitializationArguments(
        batch_size=100,
        kmeans_sampling=0.2,
        **initialization_common,
    )
    initialization_larger_minibatch = EmbeddingInitializationArguments(
        batch_size=100,
        kmeans_batch_size=20_000,
        **initialization_common,
    )
    initialization_new_algorithm = EmbeddingInitializationArguments(
        batch_size=100,
        algorithm_version="minibatch_kmeans_v3",
        **initialization_common,
    )

    assert (
        initialization_small.provenance_hash() != initialization_large.provenance_hash()
    )
    assert (
        initialization_small.provenance_hash()
        != initialization_sampled.provenance_hash()
    )
    assert (
        initialization_small.provenance_hash()
        != initialization_larger_minibatch.provenance_hash()
    )
    assert (
        initialization_small.provenance_hash()
        != initialization_new_algorithm.provenance_hash()
    )


def test_ann_parallel_is_normal_provenance_not_cache_policy() -> None:
    common = {
        "coordinates": _ref("reduction", "1"),
        "ann_metric": "l2",
        "ann_efc": 50,
        "ann_ef": 50,
        "ann_m": 16,
        "rand_state": 4466,
        "batch_size": 100,
        "invalidate_cache": False,
    }
    serial = AnnIndexArguments(
        ann_parallel=False,
        parallel_threads=None,
        **common,
    )
    parallel = AnnIndexArguments(
        ann_parallel=True,
        parallel_threads=4,
        **common,
    )

    assert serial.provenance_hash() != parallel.provenance_hash()
    assert parallel.to_record().parameters["ann_parallel"] is True


def test_dynamic_callable_requires_explicit_identity() -> None:
    with pytest.raises(ValueError, match="artifact_identity"):
        NormalizationArguments(
            from_assay="RNA",
            cell_key="I",
            cell_selection=_ref(
                "cell_selection",
                "2",
                scope="datastore",
            ),
            feature_selection=_ref("feature_selection", "3"),
            normalization_method=lambda values: values,
            size_factor=1000.0,
            log_transform=True,
            renormalize_subset=False,
            update_state=True,
        ).to_record()


def test_field_factories_set_argument_roles() -> None:
    assert parameter().metadata["argument_role"] == "parameter"
    assert execution(False).metadata["argument_role"] == "execution"
    assert artifact_input().metadata["argument_role"] == "input"
    roles = {
        model_field.name: model_field.metadata["argument_role"]
        for model_field in fields(NormalizationArguments)
    }
    assert roles["normalization_method"] == "parameter"
    assert roles["cell_selection"] == "input"
    assert roles["invalidate_cache"] == "execution"


def test_operation_arguments_plan_reuses_complete_artifact() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    arguments = FeatureScalingArguments(
        normalized=_ref("normalized", "1"),
        enabled=True,
        batch_size=100,
        invalidate_cache=False,
    )
    first = arguments.plan(root, scope="assay", assay="RNA")
    group = start_artifact(root, first)
    group.create_array("mean", data=np.array([1.0, 2.0]))
    finish_artifact(group, first)

    reused = arguments.plan(root, scope="assay", assay="RNA")
    assert reused.reused
    assert reused.ref == first.ref

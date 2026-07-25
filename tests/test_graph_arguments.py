import inspect

import numpy as np
import pytest

from scarf.datastore._operations.graph import _GraphOperationsMixin
from scarf.graph.arguments import (
    MAKE_GRAPH_ARGUMENT_OWNERS,
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
)
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
        feat_key="hvgs",
        cell_selection=_ref("cell_selection", "2", scope="datastore"),
        feature_selection=_ref("feature_selection", "3"),
        normalization_method="norm_lib_size",
        size_factor=1000.0,
        log_transform=True,
        renormalize_subset=False,
        batch_size=100,
        update_state=True,
        local_cache="auto",
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
        "feat_key": "hvgs",
        "batch_size": 100,
        "update_state": True,
        "local_cache": "auto",
        "invalidate_cache": False,
    }


def test_execution_options_do_not_change_provenance_hash() -> None:
    common = {
        "from_assay": "RNA",
        "cell_key": "I",
        "feat_key": "hvgs",
        "cell_selection": _ref("cell_selection", "2", scope="datastore"),
        "feature_selection": _ref("feature_selection", "3"),
        "normalization_method": "norm_lib_size",
        "size_factor": 1000.0,
        "log_transform": True,
        "renormalize_subset": False,
        "update_state": True,
        "local_cache": "auto",
    }
    small_batch = NormalizationArguments(
        batch_size=100,
        invalidate_cache=False,
        **common,
    )
    invalidated = NormalizationArguments(
        batch_size=1000,
        invalidate_cache=True,
        **common,
    )
    changed_parameter = NormalizationArguments(
        batch_size=100,
        invalidate_cache=False,
        **(common | {"log_transform": False}),
    )
    changed_size_factor = NormalizationArguments(
        batch_size=100,
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
        calculation_batch_size=100,
        batch_size=100,
    )
    harmony = HarmonyArguments(
        reduction=_ref("reduction", "1"),
        batch_values=_ref("metadata_snapshot", "2", scope="datastore"),
        batch_columns=("donor", "sample"),
        harmony_parameters={"theta": 2.0},
        batch_size=100,
        force_refit=False,
    )
    neighbors = NeighborQueryArguments(
        ann_index=_ref("ann_index", "3"),
        coordinates=_ref("batch_correction", "4"),
        k=15,
        batch_size=100,
    )
    connectivity = ConnectivityMapArguments(
        neighbors=_ref("neighbors", "5"),
        local_connectivity=1.0,
        bandwidth=1.5,
        batch_size=100,
    )

    assert harmony.to_record().parameters["batch_columns"] == ["donor", "sample"]
    assert scaling.to_record().parameters == {
        "enabled": True,
        "calculation_batch_size": 100,
    }
    assert neighbors.to_record().inputs["ann_index"]["kind"] == "ann_index"
    assert connectivity.to_record().parameters == {
        "local_connectivity": 1.0,
        "bandwidth": 1.5,
    }


def test_disabled_scaling_and_lsi_ignore_batch_size_for_reuse() -> None:
    normalized = _ref("normalized", "0")
    scaling_small = FeatureScalingArguments(
        normalized=normalized,
        enabled=False,
        calculation_batch_size=None,
        batch_size=100,
    )
    scaling_large = FeatureScalingArguments(
        normalized=normalized,
        enabled=False,
        calculation_batch_size=None,
        batch_size=500,
    )
    lsi_common = {
        "normalized": normalized,
        "feature_scaling": _ref("feature_scaling", "1"),
        "dims": 5,
        "skip_first": True,
        "rand_state": 4466,
        "update_state": False,
        "invalidate_cache": False,
    }
    lsi_small = LsiArguments(batch_size=100, **lsi_common)
    lsi_large = LsiArguments(batch_size=500, **lsi_common)

    assert scaling_small.provenance_hash() == scaling_large.provenance_hash()
    assert lsi_small.provenance_hash() == lsi_large.provenance_hash()


def test_pca_and_embedding_initialization_include_batch_size_in_provenance() -> None:
    normalized = _ref("normalized", "0")
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

    assert pca_small.provenance_hash() != pca_large.provenance_hash()
    assert (
        initialization_small.provenance_hash() != initialization_large.provenance_hash()
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
        "ann_index_fetcher": None,
        "ann_index_saver": None,
        "local_cache": False,
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


def test_ann_fetcher_identity_is_an_input() -> None:
    def first_fetcher(_path: str) -> None:
        return None

    def second_fetcher(_path: str) -> None:
        return None

    first_fetcher.artifact_identity = "first"  # type: ignore[attr-defined]
    second_fetcher.artifact_identity = "second"  # type: ignore[attr-defined]
    common = {
        "coordinates": _ref("reduction", "1"),
        "ann_metric": "l2",
        "ann_efc": 50,
        "ann_ef": 50,
        "ann_m": 16,
        "rand_state": 4466,
        "ann_parallel": False,
        "parallel_threads": None,
        "batch_size": 100,
        "ann_index_saver": None,
        "local_cache": False,
        "invalidate_cache": False,
    }
    first = AnnIndexArguments(ann_index_fetcher=first_fetcher, **common)
    second = AnnIndexArguments(ann_index_fetcher=second_fetcher, **common)

    assert first.provenance_hash() != second.provenance_hash()
    assert "ann_index_fetcher" in first.to_record().inputs
    second_fetcher.artifact_identity = "first"  # type: ignore[attr-defined]
    equivalent = AnnIndexArguments(ann_index_fetcher=second_fetcher, **common)
    assert first.provenance_hash() == equivalent.provenance_hash()
    assert first.to_record().inputs["ann_index_fetcher"] == {
        "external_hook": True,
        "identity": "first",
    }


def test_dynamic_callable_requires_explicit_identity() -> None:
    with pytest.raises(ValueError, match="artifact_identity"):
        NormalizationArguments(
            from_assay="RNA",
            cell_key="I",
            feat_key="hvgs",
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
            batch_size=100,
            update_state=True,
            local_cache=False,
        ).to_record()


def test_make_graph_signature_has_an_argument_owner() -> None:
    signature = inspect.signature(_GraphOperationsMixin.make_graph)
    public_arguments = set(signature.parameters) - {"self"}
    valid_owners = {
        "ann_index",
        "connectivity_map",
        "embedding_initialization",
        "facade",
        "harmony",
        "neighbors",
        "normalization",
        "reduction",
        "reduction_ann_index_and_embedding_initialization",
        "stage_execution",
        "stage_specific_parameter_or_execution_option",
    }

    assert public_arguments == set(MAKE_GRAPH_ARGUMENT_OWNERS)
    assert set(MAKE_GRAPH_ARGUMENT_OWNERS.values()) <= valid_owners

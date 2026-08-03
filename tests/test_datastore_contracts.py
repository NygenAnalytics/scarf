import inspect

from scarf.datastore.base_datastore import BaseDataStore
from scarf.datastore.datastore import DataStore
from scarf.datastore.graph_datastore import GraphDataStore
from scarf.datastore.mapping_datastore import MappingDatastore
from tests.signature_contracts import signature_digest


_METHODS = {
    BaseDataStore: (
        "__init__",
        "get_cell_vals",
        "get_assay_state",
        "inspect_artifact",
        "lineage",
        "list_artifacts",
        "load_artifact",
        "set_default_assay",
        "summary",
    ),
    GraphDataStore: (
        "__init__",
        "build_ann_index",
        "build_connectivity_map",
        "build_embedding_initialization",
        "build_mapping_reference",
        "get_diffusion_operator",
        "get_latest_graph_loc",
        "get_normalized_group_path",
        "get_imputed",
        "get_mapping_reference",
        "integrate_assays",
        "load_graph",
        "query_neighbors",
        "run_fate_mapping",
        "run_leiden_clustering",
        "run_lsi",
        "run_custom_reduction",
        "run_harmony",
        "run_normalization",
        "run_paris_clustering",
        "run_pca",
        "run_pseudotime_scoring",
        "run_topacedo_sampler",
        "run_tsne",
        "run_umap",
    ),
    MappingDatastore: (
        "calibrate_label_transfer_threshold",
        "get_mapping_result",
        "get_mapping_score",
        "get_target_classes",
        "get_target_label_evidence",
        "run_mapping",
    ),
    DataStore: (
        "__init__",
        "add_grouped_assay",
        "add_melded_assay",
        "auto_filter_cells",
        "calc_membership_strength",
        "export_markers_to_csv",
        "filter_cells",
        "get_assay",
        "get_enrichment",
        "get_markers",
        "make_bulk",
        "mark_hto_identities",
        "mark_hvgs",
        "mark_prevalent_peaks",
        "metric_clisi",
        "metric_cluster_separability",
        "metric_graph_connectivity",
        "metric_graph_silhouette",
        "metric_ilisi",
        "metric_label_concordance",
        "metric_lisi",
        "metric_proportional_batch_mixing",
        "run_aucell",
        "run_cell_cycle_scoring",
        "run_doublet_detection",
        "run_marker_search",
        "run_pseudotime_aggregation",
        "run_pseudotime_marker_search",
        "run_waggr",
        "set_hvgs",
        "show_zarr_tree",
        "smart_label",
        "to_anndata",
    ),
}

_SIGNATURE_DIGESTS = {
    BaseDataStore: "618821f381727cab2b5b19ec983871b486c5bdcab992435f3a0cb34033dfbcf4",
    GraphDataStore: "eecfb41646f3b515e2979882a1afe56b0d1497775562110af406c8f178f7005a",
    MappingDatastore: "1be2723f0c659336ff3dba16d58f4c9a7aa2f73a9a4a33bf3af232a280f1d7ec",
    DataStore: "1f1cfcc32db7daa61687f1620d80fa7cf74495c89747fe9ae99bc7864062d98c",
}


def test_datastore_public_method_signatures_are_stable():
    for cls, names in _METHODS.items():
        methods = {name: getattr(cls, name) for name in names}
        assert signature_digest(methods) == _SIGNATURE_DIGESTS[cls]


def test_datastore_public_class_chain_is_stable():
    public_classes = {BaseDataStore, GraphDataStore, MappingDatastore, DataStore}
    assert [cls for cls in DataStore.mro() if cls in public_classes] == [
        DataStore,
        MappingDatastore,
        GraphDataStore,
        BaseDataStore,
    ]
    assert DataStore.__module__ == "scarf.datastore.datastore"
    assert MappingDatastore.__module__ == "scarf.datastore.mapping_datastore"
    assert GraphDataStore.__module__ == "scarf.datastore.graph_datastore"
    assert BaseDataStore.__module__ == "scarf.datastore.base_datastore"


def test_stored_graph_lookup_remains_internal():
    assert not hasattr(GraphDataStore, "lookup_stored_graph")
    assert hasattr(GraphDataStore, "_lookup_stored_graph")


def test_datastore_property_contracts_are_stable():
    for name in ("assay_names", "zw"):
        descriptor = inspect.getattr_static(BaseDataStore, name)
        assert isinstance(descriptor, property)
        assert descriptor.fget is not None
        assert list(inspect.signature(descriptor.fget).parameters) == ["self"]
        assert inspect.getattr_static(DataStore, name) is descriptor


def test_datastore_plot_namespace_contract_is_stable():
    descriptor = inspect.getattr_static(DataStore, "plots")

    assert isinstance(descriptor, property)
    assert descriptor.fget is not None
    assert list(inspect.signature(descriptor.fget).parameters) == ["self"]
    assert "plots" in DataStore.__dict__
    for cls in (BaseDataStore, GraphDataStore, MappingDatastore):
        assert not hasattr(cls, "plots")


def test_datastore_static_method_contracts_are_stable():
    static_methods = {
        GraphDataStore: ("_resolve_local_cache_plan",),
        MappingDatastore: (
            "_label_vote_decision",
            "_projection_block_size",
            "_query_batch_codes",
            "calibrate_label_transfer_threshold",
        ),
        DataStore: ("_write_marker_slot",),
    }
    for cls, names in static_methods.items():
        assert all(
            isinstance(inspect.getattr_static(cls, name), staticmethod)
            for name in names
        )

    for name in (
        "_same_assay_store",
        "_validate_projection_arrays",
        "_projection_has_provenance",
        "_PROJECTION_SCHEMA_VERSION",
        "_LEGACY_PROJECTION_SCHEMA_VERSIONS",
        "_LEGACY_PROJECTION_ATTRS",
        "_LEGACY_PROJECTION_ARRAYS",
        "_projection_attr",
        "_projection_array_name",
    ):
        assert not hasattr(MappingDatastore, name)


def test_graph_datastore_private_mixin_order_is_stable():
    from scarf.datastore._operations.clustering import _ClusteringOperationsMixin
    from scarf.datastore._operations.embeddings import _EmbeddingOperationsMixin
    from scarf.datastore._operations.graph import _GraphOperationsMixin
    from scarf.datastore._operations.mapping_reference import (
        _MappingReferenceOperationsMixin,
    )
    from scarf.datastore._operations.trajectory import _TrajectoryOperationsMixin

    assert GraphDataStore.__bases__ == (
        _EmbeddingOperationsMixin,
        _ClusteringOperationsMixin,
        _TrajectoryOperationsMixin,
        _MappingReferenceOperationsMixin,
        _GraphOperationsMixin,
        BaseDataStore,
    )
    assert not hasattr(GraphDataStore, "make_graph")


def test_mapping_datastore_private_mixin_order_is_stable():
    from scarf.datastore._operations.mapping import _MappingOperationsMixin

    assert MappingDatastore.__bases__ == (
        _MappingOperationsMixin,
        GraphDataStore,
    )
    assert MappingDatastore.mro()[:3] == [
        MappingDatastore,
        _MappingOperationsMixin,
        GraphDataStore,
    ]


def test_retired_unified_mapping_apis_are_absent():
    for name in (
        "_load_unified_layout_data",
        "load_unified_graph",
        "run_unified_tsne",
        "run_unified_umap",
    ):
        assert not hasattr(MappingDatastore, name)


def test_datastore_private_mixin_order_is_stable():
    from scarf.datastore._operations.features import _FeatureOperationsMixin
    from scarf.datastore._operations.integration_metrics import (
        _IntegrationMetricsOperationsMixin,
    )
    from scarf.datastore._operations.presentation import _PresentationOperationsMixin
    from scarf.datastore._operations.quality_control import (
        _QualityControlOperationsMixin,
    )
    from scarf.datastore._operations.trajectory import (
        _TrajectoryFeatureOperationsMixin,
    )

    assert DataStore.__bases__ == (
        _QualityControlOperationsMixin,
        _FeatureOperationsMixin,
        _TrajectoryFeatureOperationsMixin,
        _IntegrationMetricsOperationsMixin,
        _PresentationOperationsMixin,
        MappingDatastore,
    )
    assert DataStore.mro()[:7] == [
        DataStore,
        _QualityControlOperationsMixin,
        _FeatureOperationsMixin,
        _TrajectoryFeatureOperationsMixin,
        _IntegrationMetricsOperationsMixin,
        _PresentationOperationsMixin,
        MappingDatastore,
    ]


def test_datastore_temporary_factory_uses_parent_budget_and_local_profile(tmp_path):
    import numpy as np
    from scipy.sparse import csr_matrix

    from scarf.datastore._operations.quality_control import (
        _QualityControlOperationsMixin,
    )
    from scarf.quality_control.doublets import write_doublet_target_zarr

    descriptor = inspect.getattr_static(DataStore, "_create_temporary_datastore")
    assert not isinstance(descriptor, staticmethod)
    assert "_create_temporary_datastore" in DataStore.__dict__
    assert "_create_temporary_datastore" not in _QualityControlOperationsMixin.__dict__

    class DataStoreSubclass(DataStore):
        pass

    path = tmp_path / "temporary.zarr"
    write_doublet_target_zarr(
        zarr_loc=str(path),
        assay_name="RNA",
        sim_counts=csr_matrix(np.array([[1, 0], [0, 2]], dtype=np.uint16)),
        feat_ids=np.array(["f1", "f2"]),
        feat_names=np.array(["g1", "g2"]),
        dtype="uint16",
        mem_budget=64 * 1024 * 1024,
        nthreads=1,
        profile="fast_local",
    )

    source = object.__new__(DataStoreSubclass)
    source.memoryBytes = 64 * 1024 * 1024
    source.storageProfile = "cloud"
    result = source._create_temporary_datastore(
        str(path),
        default_assay="RNA",
        assay_types={"RNA": "RNA"},
        nthreads=3,
    )
    assert result.memoryBytes == 64 * 1024 * 1024
    assert result.nthreads == 3
    assert result.storageProfile == "fast_local"


def test_datastore_facades_only_own_composition_methods():
    def defined_methods(cls: type) -> set[str]:
        return {
            name
            for name, value in cls.__dict__.items()
            if inspect.isfunction(value)
            or isinstance(value, (classmethod, staticmethod))
        }

    assert defined_methods(GraphDataStore) == {"__init__"}
    assert defined_methods(MappingDatastore) == set()
    assert defined_methods(DataStore) == {
        "__init__",
        "_create_temporary_datastore",
        "get_assay",
    }


def test_datastore_operation_mixins_have_unique_method_owners():
    from scarf.datastore._operations.clustering import _ClusteringOperationsMixin
    from scarf.datastore._operations.embeddings import _EmbeddingOperationsMixin
    from scarf.datastore._operations.features import _FeatureOperationsMixin
    from scarf.datastore._operations.graph import _GraphOperationsMixin
    from scarf.datastore._operations.integration_metrics import (
        _IntegrationMetricsOperationsMixin,
    )
    from scarf.datastore._operations.mapping import _MappingOperationsMixin
    from scarf.datastore._operations.mapping_reference import (
        _MappingReferenceOperationsMixin,
    )
    from scarf.datastore._operations.presentation import _PresentationOperationsMixin
    from scarf.datastore._operations.quality_control import (
        _QualityControlOperationsMixin,
    )
    from scarf.datastore._operations.trajectory import (
        _TrajectoryFeatureOperationsMixin,
        _TrajectoryOperationsMixin,
    )

    mixins = (
        _EmbeddingOperationsMixin,
        _ClusteringOperationsMixin,
        _TrajectoryOperationsMixin,
        _MappingReferenceOperationsMixin,
        _GraphOperationsMixin,
        _MappingOperationsMixin,
        _QualityControlOperationsMixin,
        _FeatureOperationsMixin,
        _TrajectoryFeatureOperationsMixin,
        _IntegrationMetricsOperationsMixin,
        _PresentationOperationsMixin,
    )
    owners: dict[str, list[str]] = {}
    for mixin in mixins:
        for name, value in mixin.__dict__.items():
            if inspect.isfunction(value) or isinstance(
                value, (classmethod, staticmethod)
            ):
                owners.setdefault(name, []).append(mixin.__name__)
    assert {name: owner for name, owner in owners.items() if len(owner) > 1} == {}


def test_feature_selection_and_pseudotime_methods_have_domain_owners():
    from scarf.datastore._operations.features import _FeatureOperationsMixin
    from scarf.datastore._operations.quality_control import (
        _QualityControlOperationsMixin,
    )
    from scarf.datastore._operations.trajectory import (
        _TrajectoryFeatureOperationsMixin,
        _TrajectoryOperationsMixin,
    )

    assert "mark_hvgs" in _FeatureOperationsMixin.__dict__
    assert "get_enrichment" in _FeatureOperationsMixin.__dict__
    assert "run_aucell" in _FeatureOperationsMixin.__dict__
    assert "run_waggr" in _FeatureOperationsMixin.__dict__
    assert "mark_hvgs" not in _QualityControlOperationsMixin.__dict__
    assert "run_fate_mapping" in _TrajectoryOperationsMixin.__dict__
    assert "run_fate_mapping" not in _TrajectoryFeatureOperationsMixin.__dict__
    assert "run_pseudotime_marker_search" in (
        _TrajectoryFeatureOperationsMixin.__dict__
    )
    assert "run_pseudotime_aggregation" in _TrajectoryFeatureOperationsMixin.__dict__
    assert "run_pseudotime_marker_search" not in _FeatureOperationsMixin.__dict__
    assert "run_pseudotime_aggregation" not in _FeatureOperationsMixin.__dict__
    assert not hasattr(GraphDataStore, "run_pseudotime_marker_search")
    assert not hasattr(GraphDataStore, "run_pseudotime_aggregation")
    assert not hasattr(MappingDatastore, "run_pseudotime_marker_search")
    assert not hasattr(MappingDatastore, "run_pseudotime_aggregation")

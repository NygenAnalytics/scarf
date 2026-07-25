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
        "list_artifacts",
        "load_artifact",
        "set_default_assay",
    ),
    GraphDataStore: (
        "__init__",
        "_get_latest_graph_loc",
        "build_ann_index",
        "build_connectivity_map",
        "build_mapping_reference",
        "get_diffusion_operator",
        "get_latest_graph_loc",
        "get_normalized_group_path",
        "get_imputed",
        "get_mapping_reference",
        "integrate_assays",
        "load_graph",
        "make_graph",
        "query_neighbors",
        "run_clustering",
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
        "load_unified_graph",
        "project_mapping_layout",
        "run_mapping",
        "run_unified_tsne",
        "run_unified_umap",
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
        "metric_batch_mixing",
        "metric_clisi",
        "metric_graph_connectivity",
        "metric_graph_silhouette",
        "metric_ilisi",
        "metric_integration",
        "metric_label_concordance",
        "metric_lisi",
        "metric_proportional_batch_mixing",
        "metric_silhouette",
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
    BaseDataStore: "1057b1cbeb909e7f7f599f88d91fae2aacb024a3da626b8a028bdd600644e248",
    GraphDataStore: "9a7880a2d14851d71a5ab89bcf1db2f8e54b5a3b7a26d994c04e54f8c9760099",
    MappingDatastore: "eaa8df1bda8fb83066fbac9e30a112fb8a0a46b1e278cce304b300ac5603e16f",
    DataStore: "f496fc7cc0237ee43465167546999f4953e65816c8f95d956b318b16009b36b8",
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
        GraphDataStore: (
            "_choose_reduction_method",
            "_resolve_local_cache_plan",
        ),
        MappingDatastore: (
            "_label_vote_decision",
            "_projection_block_size",
            "_query_batch_codes",
            "_same_assay_store",
            "_validate_projection_arrays",
            "calibrate_label_transfer_threshold",
        ),
        DataStore: ("_write_marker_slot",),
    }
    for cls, names in static_methods.items():
        assert all(
            isinstance(inspect.getattr_static(cls, name), staticmethod)
            for name in names
        )

    assert hasattr(MappingDatastore, "_projection_has_provenance")
    assert not hasattr(MappingDatastore, "_PROJECTION_SCHEMA_VERSION")
    assert not hasattr(MappingDatastore, "_LEGACY_PROJECTION_SCHEMA_VERSIONS")


def test_graph_datastore_private_mixin_order_is_stable():
    from scarf.datastore._operations.clustering import _ClusteringOperationsMixin
    from scarf.datastore._operations.embeddings import _EmbeddingOperationsMixin
    from scarf.datastore._operations.graph import _GraphOperationsMixin
    from scarf.datastore._operations.trajectory import _TrajectoryOperationsMixin

    assert GraphDataStore.__bases__ == (
        _EmbeddingOperationsMixin,
        _ClusteringOperationsMixin,
        _TrajectoryOperationsMixin,
        _GraphOperationsMixin,
        BaseDataStore,
    )


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


def test_unified_layout_adapter_signature_is_stable():
    assert str(inspect.signature(MappingDatastore._load_unified_layout_data)) == (
        "(self, layout_key: str, from_assay: str | None = None) -> "
        "tuple[numpy.ndarray, numpy.ndarray, int, list[int], list[str]]"
    )


def test_datastore_private_mixin_order_is_stable():
    from scarf.datastore._operations.features import _FeatureOperationsMixin
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
        _PresentationOperationsMixin,
        MappingDatastore,
    )
    assert DataStore.mro()[:6] == [
        DataStore,
        _QualityControlOperationsMixin,
        _FeatureOperationsMixin,
        _TrajectoryFeatureOperationsMixin,
        _PresentationOperationsMixin,
        MappingDatastore,
    ]


def test_datastore_temporary_factory_is_static_and_facade_owned(monkeypatch):
    from importlib import import_module

    from scarf.datastore._operations.quality_control import (
        _QualityControlOperationsMixin,
    )

    descriptor = inspect.getattr_static(DataStore, "_create_temporary_datastore")
    assert isinstance(descriptor, staticmethod)
    assert "_create_temporary_datastore" in DataStore.__dict__
    assert "_create_temporary_datastore" not in _QualityControlOperationsMixin.__dict__

    class DataStoreSubclass(DataStore):
        pass

    calls = []
    sentinel = object()

    def construct_concrete(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    module = import_module("scarf.datastore.datastore")
    monkeypatch.setattr(module, "DataStore", construct_concrete)
    result = DataStoreSubclass._create_temporary_datastore(
        "temporary.zarr",
        default_assay="RNA",
        assay_types={"RNA": "RNA"},
        nthreads=3,
    )
    assert result is sentinel
    assert calls == [
        (
            ("temporary.zarr",),
            {
                "default_assay": "RNA",
                "assay_types": {"RNA": "RNA"},
                "nthreads": 3,
            },
        )
    ]


def test_datastore_temporary_factory_restores_process_resources(monkeypatch):
    from importlib import import_module

    import zarr

    from scarf.storage.budget import (
        ResourceBudget,
        _get_resource_budget_override,
        set_resource_budget,
    )
    from scarf.storage.profiles import (
        _get_storage_profile_override,
        set_storage_profile,
    )

    module = import_module("scarf.datastore.datastore")
    previous_profile = _get_storage_profile_override()
    previous_budget = _get_resource_budget_override()
    previous_concurrency = zarr.config.get("async.concurrency")
    expected_budget = ResourceBudget(
        memoryBytes=64 * 1024 * 1024,
        workers=3,
        workingCopies=2,
    )
    sentinel = object()

    def construct_temporary(*args, **kwargs):
        set_storage_profile(None)
        set_resource_budget(ResourceBudget(memoryBytes=1, workers=1, workingCopies=1))
        zarr.config.set({"async.concurrency": 1})
        return sentinel

    try:
        set_storage_profile("cloud")
        set_resource_budget(expected_budget)
        zarr.config.set({"async.concurrency": expected_budget.workers})
        monkeypatch.setattr(module, "DataStore", construct_temporary)

        result = DataStore._create_temporary_datastore(
            "temporary.zarr",
            default_assay="RNA",
            assay_types={"RNA": "RNA"},
            nthreads=3,
        )

        assert result is sentinel
        assert _get_storage_profile_override() == "cloud"
        assert _get_resource_budget_override() is expected_budget
        assert zarr.config.get("async.concurrency") == expected_budget.workers
    finally:
        set_storage_profile(previous_profile)
        set_resource_budget(previous_budget)
        zarr.config.set({"async.concurrency": previous_concurrency})


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
    from scarf.datastore._operations.mapping import _MappingOperationsMixin
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
        _GraphOperationsMixin,
        _MappingOperationsMixin,
        _QualityControlOperationsMixin,
        _FeatureOperationsMixin,
        _TrajectoryFeatureOperationsMixin,
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

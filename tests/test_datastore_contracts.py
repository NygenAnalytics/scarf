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
        "set_default_assay",
    ),
    GraphDataStore: (
        "__init__",
        "build_mapping_reference",
        "get_imputed",
        "get_mapping_reference",
        "integrate_assays",
        "load_graph",
        "make_graph",
        "run_clustering",
        "run_fate_mapping",
        "run_leiden_clustering",
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
        "run_cell_cycle_scoring",
        "run_doublet_detection",
        "run_marker_search",
        "run_pseudotime_aggregation",
        "run_pseudotime_marker_search",
        "show_zarr_tree",
        "smart_label",
        "to_anndata",
    ),
}

_SIGNATURE_DIGESTS = {
    BaseDataStore: "ed2e08a687942d2339c1ab09bf1eac3af20c290319cf527b8662fb937a401cd2",
    GraphDataStore: "0f6098aa5ab1e926da212aa882dc90ab07fc632a302a45eb79ce353b5e228d59",
    MappingDatastore: "a1e2fe91d8430b54a2a67f42c537a8a5e189691f81e2fddc8a959e1fd87fb3a9",
    DataStore: "a0f2ab91f02cf4268673541c70f00d7141cbc4920bb78b97bf87ccfb331374cc",
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


def test_datastore_property_contracts_are_stable():
    for name in ("assay_names", "zw"):
        descriptor = inspect.getattr_static(BaseDataStore, name)
        assert isinstance(descriptor, property)
        assert descriptor.fget is not None
        assert list(inspect.signature(descriptor.fget).parameters) == ["self"]
        assert inspect.getattr_static(DataStore, name) is descriptor


def test_datastore_static_method_contracts_are_stable():
    static_methods = {
        GraphDataStore: (
            "_choose_reduction_method",
            "_normed_data_cached",
            "_resolve_local_cache_plan",
            "_should_cache_ann_embeddings",
            "_staged_normed_cached",
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

    assert MappingDatastore._PROJECTION_SCHEMA_VERSION == 2
    assert MappingDatastore._LEGACY_PROJECTION_SCHEMA_VERSIONS == {1}


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

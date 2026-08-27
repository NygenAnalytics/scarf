import inspect
from dataclasses import fields

from scarf.datastore.datastore import DataStore
from scarf.metadata.arguments import (
    DoubletScoreArguments,
    FateMappingArguments,
    LeidenArguments,
    MembershipStrengthArguments,
    PseudotimeScoringArguments,
    TopacedoArguments,
    TsneArguments,
    UmapArguments,
)


def _parameter_names(method: object) -> list[str]:
    return list(inspect.signature(method).parameters)


def test_frozen_imputation_and_membership_signatures() -> None:
    assert not hasattr(DataStore, "get_diffusion_operator")
    assert _parameter_names(DataStore.get_imputed) == [
        "self",
        "feature_name",
        "diffusion",
        "from_assay",
    ]
    diffusion_runner = inspect.signature(DataStore.run_diffusion_operator).parameters
    assert list(diffusion_runner) == ["self", "graph", "t", "invalidate_cache"]
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in list(diffusion_runner.values())[2:]
    )
    assert _parameter_names(DataStore.load_diffusion_operator) == [
        "self",
        "diffusion",
    ]
    membership = inspect.signature(DataStore.calc_membership_strength).parameters
    assert list(membership) == [
        "self",
        "clusters",
        "graph",
        "invalidate_cache",
    ]
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in list(membership.values())[3:]
    )


def test_doublet_graph_is_explicit_without_feat_key() -> None:
    parameters = inspect.signature(DataStore.run_doublet_detection).parameters
    assert list(parameters)[:5] == [
        "self",
        "clusters",
        "graph",
        "from_assay",
        "cluster_sample_fraction",
    ]
    assert "feat_key" not in parameters
    assert parameters["graph"].default is inspect.Parameter.empty


def test_graph_and_neighbor_consumers_have_no_path_selectors() -> None:
    graph_methods = (
        DataStore.load_graph,
        DataStore.run_umap,
        DataStore.run_tsne,
        DataStore.run_leiden_clustering,
        DataStore.run_paris_clustering,
        DataStore.run_topacedo_sampler,
        DataStore.run_diffusion_operator,
        DataStore.run_pseudotime_scoring,
        DataStore.metric_graph_connectivity,
    )
    for method in graph_methods:
        names = _parameter_names(method)
        assert "graph" in names
        assert "feat_key" not in names
        assert "integrated_graph" not in names
        assert "graph_loc" not in names

    fate_names = _parameter_names(DataStore.run_fate_mapping)
    assert "pseudotime" in fate_names
    assert "sink_labels" in fate_names
    assert "graph" not in fate_names

    neighbor_methods = (
        DataStore.metric_lisi,
        DataStore.metric_ilisi,
        DataStore.metric_clisi,
        DataStore.metric_graph_silhouette,
        DataStore.metric_proportional_batch_mixing,
    )
    for method in neighbor_methods:
        names = _parameter_names(method)
        assert "neighbors" in names
        assert "use_latest_knn" not in names
        assert "knn_loc" not in names


def test_graph_consumer_argument_records_have_no_feature_or_path_routes() -> None:
    models = (
        UmapArguments,
        TsneArguments,
        LeidenArguments,
        TopacedoArguments,
        DoubletScoreArguments,
        PseudotimeScoringArguments,
        FateMappingArguments,
        MembershipStrengthArguments,
    )
    for model in models:
        names = {field.name for field in fields(model)}
        assert "feat_key" not in names
        assert "integrated_graph" not in names
        assert "graph_loc" not in names


def test_integrated_snn_loads_captured_graph_refs() -> None:
    source = inspect.getsource(DataStore.integrate_assays)
    assert "self._load_graph_artifact(" in source
    assert "self.load_graph(" not in source
    assert "self._store_to_sparse(" not in source


def test_removed_public_path_locators_are_absent() -> None:
    assert not hasattr(DataStore, "get_latest_graph_loc")
    assert not hasattr(DataStore, "get_normalized_group_path")

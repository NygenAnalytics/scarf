import ast
import importlib
import inspect
import pkgutil
import textwrap
from collections.abc import Callable
from dataclasses import dataclass, field, fields

import pytest

from scarf.datastore import _operations as operations_package
from scarf.datastore._operations.clustering import _ClusteringOperationsMixin
from scarf.datastore._operations.graph import _GraphOperationsMixin
from scarf.datastore.datastore import DataStore
from scarf.graph import arguments as graph_arguments
from scarf.graph.arguments import OperationArguments
from scarf.metadata import arguments as metadata_arguments


_SIGNATURE_REASONS = {
    "execution",
    "parent_stage",
    "publication",
    "routing",
    "transformed",
}
_MODEL_REASONS = {
    "algorithm_version",
    "derived",
    "execution",
    "output",
    "resolved_input",
}
_OPERATION_MODULES = tuple(
    importlib.import_module(f"{operations_package.__name__}.{module.name}")
    for module in pkgutil.iter_modules(operations_package.__path__)
)


def _classified(reason: str, *names: str) -> dict[str, str]:
    return dict.fromkeys(names, reason)


@dataclass(frozen=True, slots=True)
class OperationContract:
    producer: Callable[..., object]
    arguments: type[OperationArguments]
    constructor: Callable[..., object] | None = None
    aliases: dict[str, str] = field(default_factory=dict)
    signature_only: dict[str, str] = field(default_factory=dict)
    model_only: dict[str, str] = field(default_factory=dict)


_CONTRACTS = (
    OperationContract(
        DataStore.build_ann_index,
        graph_arguments.AnnIndexArguments,
        signature_only={
            **_classified("routing", "from_assay"),
            **_classified("publication", "update_state"),
        },
        model_only={
            **_classified("derived", "parallel_threads"),
        },
    ),
    OperationContract(
        DataStore.build_connectivity_map,
        graph_arguments.ConnectivityMapArguments,
        signature_only={
            **_classified("routing", "from_assay"),
            **_classified("publication", "update_state"),
        },
    ),
    OperationContract(
        DataStore.run_custom_reduction,
        graph_arguments.CustomReductionArguments,
        constructor=_GraphOperationsMixin._run_reduction_artifact_impl,
        signature_only={
            **_classified("execution", "batch_size"),
            **_classified("execution", "local_cache"),
            **_classified("routing", "from_assay"),
        },
        model_only=_classified("resolved_input", "feature_scaling"),
    ),
    OperationContract(
        _GraphOperationsMixin._build_embedding_initialization,
        graph_arguments.EmbeddingInitializationArguments,
    ),
    OperationContract(
        _GraphOperationsMixin._run_reduction_artifact_impl,
        graph_arguments.FeatureScalingArguments,
        aliases={"feat_scaling": "enabled"},
        signature_only=_classified(
            "parent_stage",
            "custom_loadings",
            "dims",
            "from_assay",
            "lsi_n_iter",
            "lsi_n_oversamples",
            "lsi_solver",
            "lsi_skip_first",
            "method",
            "pca_cell_key",
            "rand_state",
            "show_elbow_plot",
            "update_state",
        ),
    ),
    OperationContract(
        DataStore.run_harmony,
        graph_arguments.HarmonyArguments,
        aliases={"harmony_params": "harmony_parameters"},
        signature_only={
            **_classified("routing", "from_assay"),
            **_classified("publication", "update_state"),
        },
        model_only={
            **_classified("resolved_input", "batch_values"),
            **_classified("algorithm_version", "algorithm_version"),
        },
    ),
    OperationContract(
        DataStore.run_lsi,
        graph_arguments.LsiArguments,
        constructor=_GraphOperationsMixin._run_reduction_artifact_impl,
        signature_only={
            **_classified("execution", "local_cache"),
            **_classified("routing", "from_assay"),
        },
        model_only=_classified("resolved_input", "feature_scaling"),
    ),
    OperationContract(
        DataStore.query_neighbors,
        graph_arguments.NeighborQueryArguments,
        signature_only={
            **_classified("routing", "from_assay"),
            **_classified("publication", "update_state"),
        },
        model_only=_classified("derived", "distance_metric"),
    ),
    OperationContract(
        DataStore.run_normalization,
        graph_arguments.NormalizationArguments,
        model_only={
            **_classified(
                "resolved_input",
                "cell_selection",
                "feature_selection",
            ),
            **_classified(
                "derived",
                "normalization_method",
                "size_factor",
            ),
        },
    ),
    OperationContract(
        DataStore.run_pca,
        graph_arguments.PcaArguments,
        constructor=_GraphOperationsMixin._run_reduction_artifact_impl,
        signature_only={
            **_classified("execution", "local_cache"),
            **_classified("routing", "from_assay"),
        },
        model_only=_classified(
            "resolved_input",
            "feature_scaling",
            "pca_cell_selection",
        ),
    ),
    OperationContract(
        DataStore.run_aucell,
        metadata_arguments.AucellArguments,
        aliases={"net": "network_digest"},
        model_only={
            **_classified("algorithm_version", "algorithm_version"),
            **_classified(
                "resolved_input",
                "cell_selection",
                "feature_selection",
            ),
        },
    ),
    OperationContract(
        DataStore.run_cell_cycle_scoring,
        metadata_arguments.CellCycleArguments,
        aliases={
            "g2m_genes": "g2m_gene_indices",
            "s_genes": "s_gene_indices",
        },
        model_only=_classified(
            "derived",
            "control_size",
            "normalization_method",
            "size_factor",
        ),
    ),
    OperationContract(
        DataStore.run_doublet_detection,
        metadata_arguments.DoubletScoreArguments,
        aliases={"cluster_key": "clusters"},
        model_only=_classified("resolved_input", "connectivity_map"),
    ),
    OperationContract(
        DataStore.run_fate_mapping,
        metadata_arguments.FateMappingArguments,
        model_only=_classified(
            "resolved_input",
            "cell_selection",
            "connectivity_map",
            "pseudotime",
            "sink_labels",
        ),
    ),
    OperationContract(
        DataStore.mark_hto_identities,
        metadata_arguments.HtoIdentityArguments,
        model_only={
            **_classified("derived", "method"),
            **_classified(
                "resolved_input",
                "cell_selection",
                "feature_ids_fingerprint",
            ),
        },
    ),
    OperationContract(
        DataStore.run_leiden_clustering,
        metadata_arguments.LeidenArguments,
        constructor=_ClusteringOperationsMixin._prepare_leiden_clustering,
    ),
    OperationContract(
        DataStore.run_marker_search,
        metadata_arguments.MarkerTableArguments,
        aliases={"norm_params": "normalization"},
        signature_only={
            **_classified("routing", "from_assay"),
            **_classified("publication", "skip_save"),
        },
        model_only=_classified(
            "resolved_input",
            "cell_selection",
            "clusters",
            "feature_selection",
            "normalization_method",
            "size_factor",
        )
        | _classified(
            "derived",
            "method",
            "alternative",
            "tie_correction",
            "continuity_correction",
            "adjustment_method",
            "adjustment_scope",
        ),
    ),
    OperationContract(
        DataStore.calc_membership_strength,
        metadata_arguments.MembershipStrengthArguments,
        model_only={
            **_classified("algorithm_version", "algorithm_version"),
            **_classified(
                "resolved_input",
                "cell_selection",
                "clusters",
                "connectivity_map",
            ),
            **_classified("derived", "decimals"),
            **_classified("output", "output_key"),
        },
    ),
    OperationContract(
        DataStore.mark_prevalent_peaks,
        metadata_arguments.PrevalentPeakArguments,
        model_only={
            **_classified("algorithm_version", "algorithm_version"),
            **_classified(
                "resolved_input",
                "cell_selection",
                "feature_selection",
            ),
            **_classified("derived", "normalization_method"),
        },
    ),
    OperationContract(
        DataStore.run_pseudotime_aggregation,
        metadata_arguments.PseudotimeAggregationArguments,
        aliases={"norm_params": "normalization"},
        model_only=_classified(
            "resolved_input",
            "cell_selection",
            "feature_selection",
            "normalization_method",
            "pseudotime",
            "size_factor",
        )
        | _classified("execution", "nthreads"),
    ),
    OperationContract(
        DataStore.run_pseudotime_marker_search,
        metadata_arguments.PseudotimeMarkerArguments,
        aliases={"norm_params": "normalization"},
        model_only=_classified(
            "resolved_input",
            "cell_selection",
            "feature_selection",
            "normalization_method",
            "pseudotime",
            "size_factor",
        )
        | _classified(
            "derived",
            "association_method",
            "p_value_method",
            "adjustment_method",
            "adjustment_scope",
        )
        | _classified("execution", "nthreads"),
    ),
    OperationContract(
        DataStore.run_pseudotime_scoring,
        metadata_arguments.PseudotimeScoringArguments,
        signature_only=_classified(
            "transformed",
            "source_sink_key",
            "ss_vec",
        ),
        model_only=_classified(
            "resolved_input",
            "cell_selection",
            "connectivity_map",
            "source_sink",
        ),
    ),
    OperationContract(
        DataStore.smart_label,
        metadata_arguments.SmartLabelArguments,
        model_only={
            **_classified("algorithm_version", "algorithm_version"),
            **_classified(
                "resolved_input",
                "base_labels",
                "cell_selection",
                "values",
            ),
            **_classified("derived", "suffix_style"),
        },
    ),
    OperationContract(
        DataStore.run_topacedo_sampler,
        metadata_arguments.TopacedoArguments,
        model_only=_classified(
            "resolved_input",
            "cell_selection",
            "clusters",
            "dendrogram",
        ),
    ),
    OperationContract(
        DataStore.run_tsne,
        metadata_arguments.TsneArguments,
        aliases={
            "ini_embed": "initialization",
            "nthreads": "parallel_threads",
        },
    ),
    OperationContract(
        DataStore.run_umap,
        metadata_arguments.UmapArguments,
        aliases={
            "ini_embed": "initialization",
            "nthreads": "parallel_threads",
        },
    ),
    OperationContract(
        DataStore.run_waggr,
        metadata_arguments.WaggrArguments,
        aliases={"net": "network_digest"},
        model_only={
            **_classified("algorithm_version", "algorithm_version"),
            **_classified(
                "resolved_input",
                "cell_selection",
                "feature_selection",
            ),
            **_classified(
                "derived",
                "normalization_method",
                "size_factor",
            ),
        },
    ),
)


def _concrete_argument_classes() -> set[type[OperationArguments]]:
    classes: set[type[OperationArguments]] = set()
    for module in (graph_arguments, metadata_arguments):
        for _name, candidate in inspect.getmembers(module, inspect.isclass):
            if (
                candidate is not OperationArguments
                and issubclass(candidate, OperationArguments)
                and candidate.__module__ == module.__name__
            ):
                classes.add(candidate)
    return classes


def _parameter_names(producer: Callable[..., object]) -> set[str]:
    return set(inspect.signature(producer).parameters) - {"self"}


def _model_field_names(arguments: type[OperationArguments]) -> set[str]:
    return {model_field.name for model_field in fields(arguments)}


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def test_every_operation_arguments_class_has_a_contract() -> None:
    registered = [contract.arguments for contract in _CONTRACTS]

    assert len(registered) == len(set(registered))
    assert set(registered) == _concrete_argument_classes()


@pytest.mark.parametrize(
    "contract",
    _CONTRACTS,
    ids=lambda contract: contract.arguments.operation,
)
def test_operation_signature_and_model_fields_are_classified(
    contract: OperationContract,
) -> None:
    signature_names = _parameter_names(contract.producer)
    model_names = _model_field_names(contract.arguments)
    alias_sources = set(contract.aliases)
    alias_targets = set(contract.aliases.values())

    assert alias_sources <= signature_names - model_names
    assert alias_targets <= model_names - signature_names
    assert len(alias_targets) == len(contract.aliases)
    assert set(contract.signature_only.values()) <= _SIGNATURE_REASONS
    assert set(contract.model_only.values()) <= _MODEL_REASONS
    assert signature_names - model_names == (
        alias_sources | set(contract.signature_only)
    )
    assert model_names - signature_names == (alias_targets | set(contract.model_only))


@pytest.mark.parametrize(
    "contract",
    _CONTRACTS,
    ids=lambda contract: contract.arguments.operation,
)
def test_operation_constructor_wires_every_model_field(
    contract: OperationContract,
) -> None:
    constructor = contract.constructor or contract.producer
    source = textwrap.dedent(inspect.getsource(constructor))
    tree = ast.parse(source)
    calls = [
        call
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and _call_name(call) == contract.arguments.__name__
    ]

    assert calls, f"{contract.arguments.__name__} is not built by its producer"
    expected_fields = _model_field_names(contract.arguments)
    for call in calls:
        keyword_names = {
            keyword.arg for keyword in call.keywords if keyword.arg is not None
        }
        assert keyword_names == expected_fields


@pytest.mark.parametrize(
    "contract",
    _CONTRACTS,
    ids=lambda contract: contract.arguments.operation,
)
def test_operation_model_has_no_unregistered_producer(
    contract: OperationContract,
) -> None:
    constructor = contract.constructor or contract.producer
    expected_module = inspect.getmodule(constructor)
    assert expected_module is not None
    calls_by_module: dict[str, int] = {}
    for module in _OPERATION_MODULES:
        tree = ast.parse(textwrap.dedent(inspect.getsource(module)))
        count = sum(
            1
            for call in ast.walk(tree)
            if isinstance(call, ast.Call)
            and _call_name(call) == contract.arguments.__name__
        )
        if count:
            calls_by_module[module.__name__] = count

    assert calls_by_module == {expected_module.__name__: 1}

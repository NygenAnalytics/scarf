import ast
import inspect
import textwrap
from collections.abc import Callable
from copy import copy
from typing import Any, get_type_hints

import pytest
import zarr
from zarr.storage import MemoryStore

import scarf.plotting as splt
from scarf.datastore.datastore import DataStore
from scarf.datastore.plot_accessor import DataStorePlotAccessor


_STORE_PLOT_METHODS = (
    "cluster_connectivity",
    "cluster_tree",
    "composition",
    "distribution",
    "dotplot",
    "embedding",
    "embedding_raster",
    "marker_heatmap",
    "mapping_calibration",
    "mapping_confusion",
    "mapping_evidence",
    "mapping_score",
    "matrixplot",
    "pseudotime_heatmap",
    "run_recipe",
)


def _annotation_text(annotation: ast.expr | None) -> str | None:
    if annotation is None:
        return None
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        annotation = ast.parse(annotation.value, mode="eval").body
    return ast.unparse(annotation)


def _source_annotations(function: Callable[..., Any]) -> dict[str, str | None]:
    source = textwrap.dedent(inspect.getsource(function))
    node = next(
        entry
        for entry in ast.parse(source).body
        if isinstance(entry, ast.FunctionDef | ast.AsyncFunctionDef)
    )
    arguments = node.args
    annotations = {
        argument.arg: _annotation_text(argument.annotation)
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        )
    }
    if arguments.vararg is not None:
        annotations[arguments.vararg.arg] = _annotation_text(
            arguments.vararg.annotation
        )
    if arguments.kwarg is not None:
        annotations[arguments.kwarg.arg] = _annotation_text(arguments.kwarg.annotation)
    annotations["return"] = _annotation_text(node.returns)
    return annotations


def _parameter_contract(
    function: Callable[..., Any],
) -> list[tuple[str, Any, Any]]:
    return [
        (parameter.name, parameter.kind, parameter.default)
        for parameter in list(inspect.signature(function).parameters.values())[1:]
    ]


def test_plot_accessor_surface_matches_store_first_plotting_exports():
    accessor_methods = {
        name
        for name, value in vars(DataStorePlotAccessor).items()
        if not name.startswith("_") and inspect.isfunction(value)
    }
    store_first_exports = {
        name
        for name in splt.__all__
        if inspect.isfunction(value := getattr(splt, name))
        and next(iter(inspect.signature(value).parameters), None) == "store"
    }

    assert accessor_methods == set(_STORE_PLOT_METHODS)
    assert store_first_exports == set(_STORE_PLOT_METHODS)


def test_unified_plot_accessor_is_absent():
    assert "unified_embedding" not in splt.__all__
    assert not hasattr(DataStorePlotAccessor, "unified_embedding")


@pytest.mark.parametrize("name", _STORE_PLOT_METHODS)
def test_plot_accessor_signatures_match_standalone_functions(name: str):
    standalone = getattr(splt, name)
    accessor_method = getattr(DataStorePlotAccessor, name)

    assert _parameter_contract(accessor_method) == _parameter_contract(standalone)

    standalone_annotations = _source_annotations(standalone)
    accessor_annotations = _source_annotations(accessor_method)
    standalone_annotations.pop("store")
    accessor_annotations.pop("self")
    assert accessor_annotations == standalone_annotations


@pytest.mark.parametrize("name", _STORE_PLOT_METHODS)
def test_plot_accessor_type_hints_match_standalone_functions(name: str):
    standalone_hints = get_type_hints(getattr(splt, name))
    accessor_hints = get_type_hints(getattr(DataStorePlotAccessor, name))

    standalone_hints.pop("store")
    assert accessor_hints == standalone_hints


@pytest.mark.parametrize(
    ("name", "args", "kwargs"),
    [
        (
            "embedding",
            (),
            {"layout_key": "RNA_UMAP", "rasterize_threshold": 17},
        ),
        ("embedding_raster", (), {"layout_key": "RNA_UMAP", "pixels": 32}),
        (
            "dotplot",
            (),
            {
                "features": ["GeneA"],
                "group_by": "cluster",
                "expression_cutoff": 1.5,
            },
        ),
        (
            "matrixplot",
            (),
            {"features": ["GeneA"], "group_by": "cluster", "value": "fraction"},
        ),
        ("composition", (), {"category_by": "cluster", "kind": "per_sample"}),
        ("distribution", ("RNA_nCounts",), {"bins": 17}),
        ("marker_heatmap", (), {"topn": 7, "linewidths": 0.25}),
        (
            "mapping_calibration",
            ("atlas",),
            {
                "reference_class_group": "label",
                "known_labels": ["a"],
            },
        ),
        (
            "mapping_confusion",
            ("atlas",),
            {
                "reference_class_group": "label",
                "known_labels": ["a"],
            },
        ),
        (
            "mapping_evidence",
            ("atlas",),
            {"reference_class_group": "label"},
        ),
        (
            "mapping_score",
            ("atlas",),
            {
                "kind": "histogram",
            },
        ),
        (
            "cluster_connectivity",
            (),
            {
                "group_by": "cluster",
                "layout_key": "RNA_UMAP",
                "minimum_edge_weight": 0.1,
            },
        ),
        ("run_recipe", ("recipe.toml",), {"show": False}),
        ("cluster_tree", (), {"width": 2.5}),
        (
            "pseudotime_heatmap",
            (),
            {"features": "all_features", "vmax": 3.0},
        ),
    ],
)
def test_plot_accessor_forwards_to_canonical_function(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
):
    store = object.__new__(DataStore)
    accessor = DataStorePlotAccessor(store)
    sentinel = object()
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def canonical(*call_args: Any, **call_kwargs: Any) -> object:
        calls.append((call_args, call_kwargs))
        return sentinel

    monkeypatch.setattr(splt, name, canonical)
    method = getattr(accessor, name)
    bound = inspect.signature(method).bind(*args, **kwargs)
    bound.apply_defaults()
    expected_kwargs = dict(bound.arguments)
    expected_kwargs.update(expected_kwargs.pop("heatmap_kwargs", {}))
    expected_args = [store]
    if name == "distribution":
        expected_args.append(expected_kwargs.pop("keys"))
    elif name.startswith("mapping_"):
        expected_args.append(expected_kwargs.pop("result"))
    elif name == "run_recipe":
        expected_args.append(expected_kwargs.pop("recipe"))

    assert method(*args, **kwargs) is sentinel
    assert calls == [(tuple(expected_args), expected_kwargs)]


def test_datastore_plots_returns_a_fresh_store_bound_namespace():
    store = object.__new__(DataStore)

    first = store.plots
    second = store.plots

    assert first is not second
    assert first._store is store
    assert second._store is store
    assert not hasattr(first, "__dict__")
    assert store.__dict__ == {}


def test_shallow_copy_gets_an_accessor_bound_to_the_copy():
    store = object.__new__(DataStore)
    original_accessor = store.plots

    clone = copy(store)
    clone_accessor = clone.plots

    assert clone_accessor is not original_accessor
    assert clone_accessor._store is clone


@pytest.mark.parametrize("workspace", [None, "analysis"])
@pytest.mark.parametrize("assay_name", ["plots", "summary"])
def test_datastore_rejects_reserved_assay_before_store_mutation(
    workspace: str | None,
    assay_name: str,
):
    memory_store = MemoryStore()
    root = zarr.open_group(store=memory_store, mode="w")
    active = root if workspace is None else root.create_group(workspace)
    assay = active.create_group(assay_name)
    assay.attrs["is_assay"] = True

    with pytest.raises(
        ValueError,
        match=rf"reserved for DataStore\.{assay_name}",
    ):
        DataStore(
            memory_store,
            default_assay=assay_name,
            workspace=workspace,
            min_features_per_cell=0,
        )

    assert "defaultAssay" not in active.attrs
    assert "assayTypes" not in active.attrs

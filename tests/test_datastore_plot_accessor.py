import ast
import inspect
import textwrap
from collections.abc import Callable
from copy import copy
from typing import Any, get_type_hints

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

import scarf.plotting as splt
from scarf.datastore.datastore import DataStore
from scarf.datastore.plot_accessor import DataStorePlotAccessor
from scarf.storage import ArtifactRef


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
    "modality_weights",
    "pseudotime_heatmap",
    "run_recipe",
)

_GRAPH_REF = ArtifactRef(
    scope="assay",
    assay="RNA",
    kind="connectivity_map",
    artifact_id="a" * 64,
)
_CLUSTER_REF = ArtifactRef(
    scope="assay",
    assay="RNA",
    kind="cluster_cut",
    artifact_id="b" * 64,
)
_MARKER_REF = ArtifactRef(
    scope="assay",
    assay="RNA",
    kind="marker_table",
    artifact_id="c" * 64,
)
_AGGREGATION_REF = ArtifactRef(
    scope="assay",
    assay="RNA",
    kind="pseudotime_aggregation",
    artifact_id="d" * 64,
)
_PROJECTION_REF = ArtifactRef(
    scope="assay",
    assay="RNA",
    kind="projection",
    artifact_id="e" * 64,
)
_EMBEDDING_REF = ArtifactRef(
    scope="assay",
    assay="RNA",
    kind="embedding",
    artifact_id="f" * 64,
)
_INTEGRATED_GRAPH_REF = ArtifactRef(
    scope="datastore",
    kind="integrated_graph",
    artifact_id="2" * 64,
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

    accessor_contract = _parameter_contract(accessor_method)
    if name in {"embedding", "embedding_raster"}:
        accessor_contract = [
            parameter for parameter in accessor_contract if parameter[0] != "run"
        ]
        layout_index = next(
            index
            for index, parameter in enumerate(accessor_contract)
            if parameter[0] == "layout"
        )
        accessor_contract[layout_index] = _parameter_contract(standalone)[layout_index]
    assert accessor_contract == _parameter_contract(standalone)

    standalone_annotations = _source_annotations(standalone)
    accessor_annotations = _source_annotations(accessor_method)
    standalone_annotations.pop("store")
    accessor_annotations.pop("self")
    if name in {"embedding", "embedding_raster"}:
        accessor_annotations.pop("run")
        accessor_annotations["layout"] = standalone_annotations["layout"]
    assert accessor_annotations == standalone_annotations


@pytest.mark.parametrize("name", _STORE_PLOT_METHODS)
def test_plot_accessor_type_hints_match_standalone_functions(name: str):
    standalone_hints = get_type_hints(getattr(splt, name))
    accessor_hints = get_type_hints(getattr(DataStorePlotAccessor, name))

    standalone_hints.pop("store")
    if name in {"embedding", "embedding_raster"}:
        accessor_hints.pop("run")
        accessor_hints["layout"] = standalone_hints["layout"]
    assert accessor_hints == standalone_hints


@pytest.mark.parametrize(
    ("name", "args", "kwargs"),
    [
        (
            "embedding",
            (),
            {"layout_key": "RNA_UMAP", "rasterize_threshold": 17},
        ),
        ("embedding_raster", (), {"layout": _EMBEDDING_REF, "pixels": 32}),
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
        (
            "marker_heatmap",
            (),
            {"marker": _MARKER_REF, "topn": 7, "linewidths": 0.25},
        ),
        (
            "mapping_calibration",
            (_PROJECTION_REF,),
            {
                "reference": object(),
                "reference_class_group": "label",
                "known_labels": ["a"],
            },
        ),
        (
            "mapping_confusion",
            (_PROJECTION_REF,),
            {
                "reference": object(),
                "reference_class_group": "label",
                "known_labels": ["a"],
            },
        ),
        (
            "mapping_evidence",
            (_PROJECTION_REF,),
            {"reference": object(), "reference_class_group": "label"},
        ),
        (
            "mapping_score",
            (_PROJECTION_REF,),
            {
                "reference": object(),
                "kind": "histogram",
            },
        ),
        (
            "cluster_connectivity",
            (),
            {
                "group_by": "cluster",
                "layout_key": "RNA_UMAP",
                "graph": _GRAPH_REF,
                "cell_key": "I",
                "minimum_edge_weight": 0.1,
            },
        ),
        (
            "modality_weights",
            (),
            {
                "graph": _INTEGRATED_GRAPH_REF,
                "layout": _EMBEDDING_REF,
                "point_alpha": 0.8,
            },
        ),
        ("run_recipe", ("recipe.toml",), {"show": False}),
        (
            "cluster_tree",
            (),
            {
                "graph": _GRAPH_REF,
                "clusters": _CLUSTER_REF,
                "width": 2.5,
            },
        ),
        (
            "pseudotime_heatmap",
            (),
            {"aggregation": _AGGREGATION_REF, "vmax": 3.0},
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
    expected_kwargs.pop("run", None)
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


def test_embedding_run_adapter_uses_exact_outputs_and_frozen_fields(
    monkeypatch: pytest.MonkeyPatch,
):
    import scarf.datastore._plot_accessor as plot_accessor_module

    layout = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="embedding",
        artifact_id="f" * 64,
    )
    clusters = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="cluster_cut",
        artifact_id="1" * 64,
    )
    owner = type("Owner", (), {"zw": object()})()
    frozen_cells = type("FrozenCells", (), {"columns": ("sample_id", "clusters")})()

    class FakeRun:
        assay = "RNA"
        cells = frozen_cells

        def __init__(self) -> None:
            self._owner = owner
            self._outputs = {"umap": layout, "clusters": clusters}

        def __contains__(self, key: object) -> bool:
            return key in self._outputs

        def __getitem__(self, key: str) -> ArtifactRef:
            return self._outputs[key]

    calls: list[tuple[object, dict[str, Any]]] = []

    def canonical(store: object, **kwargs: Any) -> object:
        calls.append((store, kwargs))
        return object()

    monkeypatch.setattr(plot_accessor_module, "PipelineRun", FakeRun)
    monkeypatch.setattr(splt, "embedding", canonical)
    accessor = DataStorePlotAccessor(owner)  # type: ignore[arg-type]
    run = FakeRun()

    accessor.embedding(
        run=run,  # type: ignore[arg-type]
        layout="umap",
        color_by="sample_id",
        show=False,
    )
    proxy, kwargs = calls.pop()
    assert proxy is not owner
    assert proxy.zw is owner.zw
    assert proxy.cells._cells is frozen_cells
    assert kwargs["layout"] == layout
    assert kwargs["color_by"] == "sample_id"

    accessor.embedding(
        run=run,  # type: ignore[arg-type]
        layout="umap",
        color_by=None,
        show=False,
    )
    _, kwargs = calls.pop()
    assert kwargs["color_by"] is None

    accessor.embedding(
        run=run,  # type: ignore[arg-type]
        layout="umap",
        color_by="clusters",
        show=False,
    )
    _, kwargs = calls.pop()
    assert kwargs["layout"] == layout
    assert kwargs["color_by"] == "clusters"

    with pytest.raises(ValueError, match="mutually exclusive"):
        accessor.embedding(
            run=run,  # type: ignore[arg-type]
            layout=layout,
            show=False,
        )
    with pytest.raises(KeyError, match="no frozen cell field"):
        accessor.embedding(
            run=run,  # type: ignore[arg-type]
            layout="umap",
            color_by="live_only",
            show=False,
        )
    with pytest.raises(KeyError, match="no frozen cell field"):
        accessor.embedding(
            run=run,  # type: ignore[arg-type]
            layout="umap",
            color_by="umap",
            show=False,
        )

    monkeypatch.setattr(splt, "embedding_raster", canonical)
    accessor.embedding_raster(
        run=run,  # type: ignore[arg-type]
        layout="umap",
        color_by="sample_id",
        show=False,
    )
    proxy, kwargs = calls.pop()
    assert proxy is not owner
    assert proxy.zw is owner.zw
    assert proxy.cells._cells is frozen_cells
    assert kwargs["layout"] == layout
    assert kwargs["color_by"] == "sample_id"

    with pytest.raises(ValueError, match="mutually exclusive"):
        accessor.embedding_raster(
            run=run,  # type: ignore[arg-type]
            layout=layout,
            show=False,
        )
    with pytest.raises(KeyError, match="no frozen cell field"):
        accessor.embedding_raster(
            run=run,  # type: ignore[arg-type]
            layout="umap",
            color_by="live_only",
            show=False,
        )
    with pytest.raises(TypeError, match="color_by must name a frozen cell field"):
        accessor.embedding(
            run=run,  # type: ignore[arg-type]
            layout="umap",
            color_by=layout,
            show=False,
        )
    with pytest.raises(TypeError, match="color_by must name a frozen cell field"):
        accessor.embedding_raster(
            run=run,  # type: ignore[arg-type]
            layout="umap",
            color_by=layout,
            show=False,
        )
    with pytest.raises(TypeError, match="color_by must name a frozen cell field"):
        accessor.embedding_raster(
            run=run,  # type: ignore[arg-type]
            layout="umap",
            color_by=splt.CellField("sample_id"),
            show=False,
        )


def test_frozen_run_plot_cells_falls_back_without_selected_fetch() -> None:
    from scarf.datastore._plot_accessor import _FrozenRunPlotCells

    class CompactCells:
        columns = ("clusters",)

        def _plot_fetch_all(self, column: str) -> np.ndarray:
            return np.asarray([0, -1, 1, -1])

        def fetch_all(self, column: str) -> np.ndarray:
            assert column == "I"
            return np.asarray([True, False, True, False])

    cells = _FrozenRunPlotCells(CompactCells())
    np.testing.assert_array_equal(cells.fetch("clusters"), [0, 1])
    with pytest.raises(ValueError, match="frozen pipeline cell selection"):
        cells.fetch("clusters", key="filtered")


def test_selected_metadata_column_uses_frozen_fetch_or_full_axis_fallback() -> None:
    from types import SimpleNamespace

    from scarf.plotting.embedding import _selected_metadata_column

    class FrozenCells:
        _selection_ref = object()

        def fetch(self, column: str, key: str = "I") -> np.ndarray:
            assert key == "I"
            return np.asarray([1, 3])

        def fetch_all(self, column: str) -> np.ndarray:
            raise AssertionError("matching frozen fetch must not expand")

    values = _selected_metadata_column(
        SimpleNamespace(cells=FrozenCells()),
        "clusters",
        cell_key="I",
        cell_indices=np.asarray([0, 2]),
    )
    np.testing.assert_array_equal(values, [1, 3])

    class MismatchedCells:
        _selection_ref = object()

        def fetch(self, column: str, key: str = "I") -> np.ndarray:
            return np.asarray([1])

        def fetch_all(self, column: str) -> np.ndarray:
            return np.asarray([10, 20, 30])

    fallback = _selected_metadata_column(
        SimpleNamespace(cells=MismatchedCells()),
        "clusters",
        cell_key="I",
        cell_indices=np.asarray([0, 2]),
    )
    np.testing.assert_array_equal(fallback, [10, 30])

    class KeylessCells:
        def fetch(self, column: str) -> np.ndarray:
            return np.asarray([7, 8])

        def fetch_all(self, column: str) -> np.ndarray:
            raise AssertionError("live fetch without indices must not expand")

    live = _selected_metadata_column(
        SimpleNamespace(cells=KeylessCells()),
        "clusters",
        cell_key="I",
        cell_indices=None,
    )
    np.testing.assert_array_equal(live, [7, 8])

    class KeylessFrozenCells:
        _selection_ref = object()

        def fetch(self, column: str) -> np.ndarray:
            return np.asarray([4, 5])

        def fetch_all(self, column: str) -> np.ndarray:
            raise AssertionError("keyless frozen fetch must not expand")

    keyless_frozen = _selected_metadata_column(
        SimpleNamespace(cells=KeylessFrozenCells()),
        "clusters",
        cell_key="I",
        cell_indices=np.asarray([1, 2]),
    )
    np.testing.assert_array_equal(keyless_frozen, [4, 5])


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

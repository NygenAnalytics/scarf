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
_TRANSFER_REF = ArtifactRef(
    scope="assay",
    assay="RNA",
    kind="label_transfer",
    artifact_id="f" * 64,
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
            (_TRANSFER_REF,),
            {"known_labels": ["a"]},
        ),
        (
            "mapping_confusion",
            (_TRANSFER_REF,),
            {"known_labels": ["a"], "abstention_label": "none"},
        ),
        (
            "mapping_evidence",
            (_TRANSFER_REF,),
            {"metrics": ("voteFraction",)},
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
    elif name == "mapping_score":
        expected_args.append(expected_kwargs.pop("result"))
    elif name.startswith("mapping_"):
        expected_args.append(expected_kwargs.pop("transfer"))
    elif name == "run_recipe":
        expected_args.append(expected_kwargs.pop("recipe"))

    assert method(*args, **kwargs) is sentinel
    assert calls == [(tuple(expected_args), expected_kwargs)]


class _RunHarness:
    """A run-backed accessor whose canonical plots record their calls."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import scarf.datastore._plot_accessor as plot_accessor_module

        self.layout = _EMBEDDING_REF
        clusters = ArtifactRef(
            scope="assay",
            assay="RNA",
            kind="cluster_cut",
            artifact_id="1" * 64,
        )
        self.owner = type("Owner", (), {"zw": object()})()
        self.frozen_cells = type(
            "FrozenCells", (), {"columns": ("sample_id", "clusters")}
        )()
        harness = self

        class FakeRun:
            assay = "RNA"
            cells = harness.frozen_cells

            def __init__(self) -> None:
                self._owner = harness.owner
                self._outputs = {"umap": harness.layout, "clusters": clusters}

            def __contains__(self, key: object) -> bool:
                return key in self._outputs

            def __getitem__(self, key: str) -> ArtifactRef:
                return self._outputs[key]

        self.run_type = FakeRun
        self.calls: list[tuple[str, object, dict[str, Any]]] = []
        self.defaults: dict[str, dict[str, Any]] = {}
        monkeypatch.setattr(plot_accessor_module, "PipelineRun", FakeRun)
        for name in ("embedding", "embedding_raster"):
            parameters = inspect.signature(getattr(splt, name)).parameters
            self.defaults[name] = {
                parameter.name: parameter.default
                for parameter in list(parameters.values())[1:]
            }
            monkeypatch.setattr(splt, name, self._recorder(name))
        self.accessor = DataStorePlotAccessor(self.owner)  # type: ignore[arg-type]
        self.run = FakeRun()

    def _recorder(self, name: str) -> Callable[..., object]:
        def canonical(store: object, **kwargs: Any) -> object:
            self.calls.append((name, store, kwargs))
            return object()

        return canonical


@pytest.mark.parametrize("name", ["embedding", "embedding_raster"])
def test_run_plots_forward_frozen_cells_and_the_named_output(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
):
    harness = _RunHarness(monkeypatch)
    method = getattr(harness.accessor, name)

    for color_by in ("sample_id", "clusters", None):
        method(run=harness.run, layout="umap", color_by=color_by, show=False)
        called, proxy, kwargs = harness.calls.pop()
        expected = dict(harness.defaults[name])
        expected.update(layout=harness.layout, color_by=color_by, show=False)
        assert called == name
        assert kwargs == expected
        assert proxy is not harness.owner
        assert proxy._defaultAssay == "RNA"
        assert proxy.zw is harness.owner.zw
        assert proxy.cells._cells is harness.frozen_cells

    method(run=harness.run, color_by="sample_id", show=False)
    _, _, kwargs = harness.calls.pop()
    assert kwargs["layout"] == harness.layout
    assert harness.calls == []


@pytest.mark.parametrize("name", ["embedding", "embedding_raster"])
@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        (
            {"layout": _EMBEDDING_REF},
            ValueError,
            "run is mutually exclusive with layout_key or an ArtifactRef layout",
        ),
        (
            {"layout": "umap", "layout_key": "RNA_UMAP"},
            ValueError,
            "run is mutually exclusive with layout_key or an ArtifactRef layout",
        ),
        (
            {"layout": "umap", "color_by": "live_only"},
            KeyError,
            "Pipeline run has no frozen cell field 'live_only'",
        ),
        (
            {"layout": "umap", "color_by": "umap"},
            KeyError,
            "Pipeline run has no frozen cell field 'umap'",
        ),
        (
            {"layout": "umap", "color_by": _EMBEDDING_REF},
            TypeError,
            "color_by must name a frozen cell field or be None",
        ),
        (
            {"layout": "umap", "color_by": splt.CellField("sample_id")},
            TypeError,
            "color_by must name a frozen cell field or be None",
        ),
        ({"layout": 3}, TypeError, "layout must name a pipeline output"),
    ],
)
def test_run_plots_reject_ambiguous_layouts_and_unfrozen_fields(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    kwargs: dict[str, Any],
    error: type[Exception],
    message: str,
):
    harness = _RunHarness(monkeypatch)

    with pytest.raises(error) as raised:
        getattr(harness.accessor, name)(run=harness.run, show=False, **kwargs)

    assert raised.value.args == (message,)
    assert harness.calls == []


@pytest.mark.parametrize("name", ["embedding", "embedding_raster"])
def test_run_plots_require_a_run_opened_from_this_datastore(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
):
    harness = _RunHarness(monkeypatch)
    method = getattr(harness.accessor, name)
    foreign = harness.run_type()
    foreign._owner = object()

    with pytest.raises(TypeError, match="^run must be a PipelineRun$"):
        method(run=object(), layout="umap", show=False)
    with pytest.raises(ValueError, match="^run must be opened from this datastore$"):
        method(run=foreign, layout="umap", show=False)
    with pytest.raises(TypeError, match="^String layout names require a pipeline run$"):
        method(layout="umap", show=False)
    assert harness.calls == []


_LIVE_ONLY_MESSAGE = (
    "Run embedding uses frozen layout and color outputs; live selection, "
    "feature, facet, and subset inputs are unavailable"
)


@pytest.mark.parametrize(
    ("name", "kwargs", "message"),
    [
        ("embedding", {"cell_key": "filtered"}, _LIVE_ONLY_MESSAGE),
        ("embedding", {"from_assay": "ADT"}, _LIVE_ONLY_MESSAGE),
        (
            "embedding",
            {"normalization": splt.NormalizationSpec()},
            _LIVE_ONLY_MESSAGE,
        ),
        ("embedding", {"point_sizes": [1.0, 2.0]}, _LIVE_ONLY_MESSAGE),
        ("embedding", {"facet_by": "sample_id"}, _LIVE_ONLY_MESSAGE),
        ("embedding", {"facet_order": ["s1"]}, _LIVE_ONLY_MESSAGE),
        ("embedding", {"subset_by": "sample_id"}, _LIVE_ONLY_MESSAGE),
        (
            "embedding",
            {"density_overlay": splt.DensityOverlay(group_by="sample_id")},
            "Run embedding density filters cannot use live metadata",
        ),
        (
            "embedding",
            {"highlight": splt.Highlight(by="sample_id", groups=("s1",))},
            "Run embedding highlights cannot use live metadata",
        ),
        (
            "embedding_raster",
            {"cell_key": "filtered"},
            "Run raster uses the frozen pipeline cell selection",
        ),
    ],
)
def test_run_plots_reject_live_only_inputs(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    kwargs: dict[str, Any],
    message: str,
):
    harness = _RunHarness(monkeypatch)

    with pytest.raises(ValueError) as raised:
        getattr(harness.accessor, name)(
            run=harness.run,
            layout="umap",
            show=False,
            **kwargs,
        )

    assert raised.value.args == (message,)
    assert harness.calls == []


def test_run_embedding_forwards_overlays_that_need_no_live_metadata(
    monkeypatch: pytest.MonkeyPatch,
):
    harness = _RunHarness(monkeypatch)
    overlay = splt.DensityOverlay()
    highlight = splt.Highlight(indices=(0, 2))

    harness.accessor.embedding(
        run=harness.run,
        layout="umap",
        density_overlay=overlay,
        highlight=highlight,
        show=False,
    )

    (_, _, kwargs) = harness.calls.pop()
    assert kwargs["density_overlay"] is overlay
    assert kwargs["highlight"] is highlight


def _variadic_positional(self, *values):
    return values


def _positional_only(self, value, /):
    return value


@pytest.mark.parametrize(
    ("method", "description"),
    [
        (_variadic_positional, "variadic positional"),
        (_positional_only, "positional-only"),
    ],
)
def test_forwarding_layout_rejects_parameters_it_cannot_forward(
    monkeypatch: pytest.MonkeyPatch,
    method: Callable[..., Any],
    description: str,
):
    from scarf.datastore._plot_accessor import _forwarding_layout

    monkeypatch.setattr(
        DataStorePlotAccessor,
        "_unforwardable",
        method,
        raising=False,
    )

    with pytest.raises(TypeError) as raised:
        _forwarding_layout("_unforwardable")

    assert raised.value.args == (
        f"_unforwardable() has an unsupported {description} parameter",
    )


def test_frozen_run_plot_cells_fetch_selected_rows_of_the_run() -> None:
    from scarf.datastore._plot_accessor import _FrozenRunPlotCells, _FrozenRunPlotStore

    selection = ArtifactRef(
        scope="datastore",
        kind="cell_selection",
        artifact_id="3" * 64,
    )
    display = {"kind": "categorical", "categories": []}
    blocks = object()

    class CompactCells:
        columns = ["clusters", "score"]
        _selection_ref = selection

        def _plot_fetch_selected(self, column: str) -> np.ndarray:
            assert column == "clusters"
            return np.asarray([0, 1])

        def _plot_fetch_all(self, column: str) -> list[int]:
            assert column == "clusters"
            return [5, 0, 1]

        def _field_dtype(self, column: str) -> np.dtype:
            return {"clusters": np.dtype("int32"), "score": np.dtype("float32")}[column]

        def _field_display(self, column: str) -> dict[str, Any] | None:
            return display if column == "clusters" else None

        def _iter_selected_blocks(self, columns, block_rows):
            assert (tuple(columns), block_rows) == (("score",), 2)
            return blocks

    compact = CompactCells()
    cells = _FrozenRunPlotCells(compact)

    assert cells.columns == ("clusters", "score")
    assert cells._selection_ref is selection
    np.testing.assert_array_equal(cells.fetch("clusters"), [0, 1])
    fetched_all = cells.fetch_all("clusters")
    assert isinstance(fetched_all, np.ndarray)
    np.testing.assert_array_equal(fetched_all, [5, 0, 1])
    assert cells.get_dtype("clusters") == np.dtype("int32")
    assert cells.get_dtype("score") == np.dtype("float32")
    assert cells._field_display("clusters") is display
    assert cells._field_display("score") is None
    assert cells._iter_selected_blocks(["score"], 2) is blocks
    with pytest.raises(ValueError, match="frozen pipeline cell selection"):
        cells.fetch("clusters", key="filtered")

    owner = type("Owner", (), {"zw": object()})()
    store = _FrozenRunPlotStore(owner, assay="ADT", cells=compact)  # type: ignore[arg-type]
    assert store._defaultAssay == "ADT"
    assert store.zw is owner.zw
    assert store.cells._cells is compact
    assert store._stored_display_metadata("clusters") is display
    assert store._stored_display_metadata("score") is None


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

    class LiveCells:
        def fetch(self, column: str, key: str = "I") -> np.ndarray:
            assert key == "filtered"
            return np.asarray([7, 8])

        def fetch_all(self, column: str) -> np.ndarray:
            raise AssertionError("live fetch without indices must not expand")

    live = _selected_metadata_column(
        SimpleNamespace(cells=LiveCells()),
        "clusters",
        cell_key="filtered",
        cell_indices=None,
    )
    np.testing.assert_array_equal(live, [7, 8])


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

import copy
from collections.abc import Callable
from importlib import import_module
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import zarr
from matplotlib.colors import to_rgba
from zarr.storage import MemoryStore

from scarf.plotting._figure import PlotResult
from scarf.plotting.modality_weights import modality_weights
from scarf.storage.artifacts import artifact_path, make_provenance
from scarf.storage.refs import ArtifactRef

modality_weights_module = import_module("scarf.plotting.modality_weights")


def _ref(
    kind: str,
    token: str,
    *,
    assay: str | None = None,
) -> ArtifactRef:
    return ArtifactRef(
        scope="assay" if assay is not None else "datastore",
        assay=assay,
        kind=kind,
        artifact_id=token * 64,
    )


def _wnn_store(
    weights: np.ndarray,
) -> tuple[SimpleNamespace, ArtifactRef, ArtifactRef, ArtifactRef]:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    selection = _ref("cell_selection", "a")
    layout = _ref("embedding", "b")
    graph = _ref("integrated_graph", "c")
    sources = {
        "source_0": {
            "neighbors": _ref("neighbors", "d", assay="RNA"),
            "coordinates": _ref("reduction", "e", assay="RNA"),
        },
        "source_1": {
            "neighbors": _ref("neighbors", "f", assay="ADT"),
            "coordinates": _ref("reduction", "1", assay="ADT"),
        },
        "cell_selection": selection,
    }
    group = root.create_group(artifact_path(graph))
    group.attrs.update(
        {
            "artifact_id": graph.artifact_id,
            "kind": graph.kind,
            "provenance": make_provenance(
                operation="integrate_assays",
                parameters={"method": "wnn", "assays": ["RNA", "ADT"]},
                inputs=sources,
            ),
            "execution_options": {},
            "complete": True,
            "assays": ["RNA", "ADT"],
        }
    )
    group.create_array("modality_weights", data=np.asarray(weights))
    return SimpleNamespace(zw=root), graph, layout, selection


def _resolved_layout(
    selection: ArtifactRef,
) -> tuple[np.ndarray, np.ndarray, ArtifactRef]:
    return (
        np.asarray([[0.0, 0.5], [1.0, 1.5], [2.0, 0.0]]),
        np.asarray([2, 4, 7]),
        selection,
    )


def test_modality_weights_plots_ordered_validated_wnn_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weights = np.asarray(
        [[0.8, 0.2], [0.25, 0.75], [0.5, 0.5]],
        dtype=np.float32,
    )
    store, graph, layout, selection = _wnn_store(weights)
    monkeypatch.setattr(
        modality_weights_module,
        "_resolve_layout",
        lambda *_args: _resolved_layout(selection),
    )

    result = modality_weights(
        store,
        graph=graph,
        layout=layout,
        point_size=7.0,
        show=False,
    )

    assert isinstance(result, PlotResult)
    assert list(result.axes) == ["RNA", "ADT"]
    assert list(result.tables["weights"].columns) == ["RNA", "ADT"]
    assert list(result.tables["weights"].index) == [2, 4, 7]
    assert result.tables["weights"].index.name == "cell_index"
    np.testing.assert_allclose(result.tables["weights"].to_numpy(), weights)
    assert result.provenance.n_cells == 3
    assert result.provenance.notes == ("modality_weights", "artifact")
    assert result.provenance.extras == {
        "graph": graph.to_dict(),
        "layout": layout.to_dict(),
        "cell_selection": selection.to_dict(),
        "assays": ["RNA", "ADT"],
        "rasterize_threshold": 50_000,
    }
    assert result.legends[0].extras == {"assays": ["RNA", "ADT"]}
    assert (result.scales[0].vmin, result.scales[0].vmax) == (0.0, 1.0)
    coordinates = _resolved_layout(selection)[0]
    for column, (assay, axis) in enumerate(result.axes.items()):
        points = axis.collections[0]
        # Each assay panel colors the same cells by that assay's weight column.
        np.testing.assert_allclose(points.get_offsets(), coordinates)
        np.testing.assert_allclose(points.get_array(), weights[:, column])
        assert points.get_clim() == (0.0, 1.0)
        assert points.get_cmap().name == "viridis"
        assert points.get_sizes().tolist() == [7.0]
        assert points.get_rasterized() is False
        assert axis.get_title() == assay
        assert points.colorbar.ax.get_xlabel() == f"{assay} weight"
    result.close()


def test_modality_weights_respects_columns_colormap_and_rasterization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weights = np.asarray([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]], dtype=np.float64)
    store, graph, layout, selection = _wnn_store(weights)
    monkeypatch.setattr(
        modality_weights_module,
        "_resolve_layout",
        lambda *_args: _resolved_layout(selection),
    )

    result = modality_weights(
        store,
        graph=graph,
        layout=layout,
        cmap="magma",
        n_columns=1,
        point_alpha=0.5,
        rasterize_threshold=3,
        show=False,
    )

    rna_axis, adt_axis = result.axes.values()
    # One column stacks the assay panels and sizes the figure per panel.
    assert rna_axis.get_position().y0 > adt_axis.get_position().y0
    assert rna_axis.get_position().x0 == pytest.approx(adt_axis.get_position().x0)
    width, height = result.figure.get_size_inches()
    assert height == pytest.approx(2 * width)
    for axis, column in ((rna_axis, 0), (adt_axis, 1)):
        points = axis.collections[0]
        assert points.get_cmap().name == "magma"
        assert points.get_alpha() == 0.5
        assert points.get_rasterized() is True
        points.update_scalarmappable()
        expected = [
            to_rgba(points.get_cmap()(value), 0.5) for value in weights[:, column]
        ]
        np.testing.assert_allclose(points.get_facecolors(), expected)
    assert result.scales[0].cmap == "magma"
    assert result.provenance.extras["rasterize_threshold"] == 3
    result.close()


def test_modality_weights_shows_owned_results_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, graph, layout, selection = _wnn_store(np.full((3, 2), 0.5, dtype=np.float32))
    monkeypatch.setattr(
        modality_weights_module,
        "_resolve_layout",
        lambda *_args: _resolved_layout(selection),
    )
    shown: list[PlotResult] = []
    monkeypatch.setattr(PlotResult, "show", lambda result: shown.append(result))

    result = modality_weights(store, graph=graph, layout=layout)
    suppressed = modality_weights(store, graph=graph, layout=layout, show=False)

    assert shown == [result]
    result.close()
    suppressed.close()


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"point_size": 0.0}, ValueError, "point_size must be positive and finite"),
        (
            {"point_size": float("nan")},
            ValueError,
            "point_size must be positive and finite",
        ),
        (
            {"point_alpha": 1.5},
            ValueError,
            "point_alpha must be between zero and one",
        ),
        ({"cmap": ""}, TypeError, "cmap must be a non-empty string"),
        (
            {"rasterize_threshold": True},
            ValueError,
            "rasterize_threshold must be a positive integer",
        ),
        (
            {"rasterize_threshold": 0},
            ValueError,
            "rasterize_threshold must be a positive integer",
        ),
        ({"n_columns": 0}, ValueError, "n_columns must be a positive integer"),
        ({"n_columns": 1.5}, ValueError, "n_columns must be a positive integer"),
    ],
)
def test_modality_weights_rejects_invalid_plot_options(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    store, graph, layout, selection = _wnn_store(np.full((3, 2), 0.5, dtype=np.float32))
    monkeypatch.setattr(
        modality_weights_module,
        "_resolve_layout",
        lambda *_args: _resolved_layout(selection),
    )

    with pytest.raises(error) as raised:
        modality_weights(store, graph=graph, layout=layout, show=False, **kwargs)

    assert raised.value.args == (message,)


def test_modality_weights_requires_exact_layout_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, graph, layout, _selection = _wnn_store(
        np.full((3, 2), 0.5, dtype=np.float32)
    )
    other_selection = _ref("cell_selection", "9")
    monkeypatch.setattr(
        modality_weights_module,
        "_resolve_layout",
        lambda *_args: _resolved_layout(other_selection),
    )

    with pytest.raises(ValueError, match="exact cell-selection artifact"):
        modality_weights(store, graph=graph, layout=layout, show=False)


@pytest.mark.parametrize(
    ("weights", "error", "message"),
    [
        (np.full((2, 2), 0.5, dtype=np.float32), ValueError, "one row"),
        (
            np.asarray([[np.nan, np.nan]] * 3, dtype=np.float32),
            ValueError,
            "finite",
        ),
        (
            np.asarray([[-0.1, 1.1]] * 3, dtype=np.float32),
            ValueError,
            "non-negative",
        ),
        (
            np.asarray([[0.2, 0.2]] * 3, dtype=np.float32),
            ValueError,
            "sum to one",
        ),
        (np.ones((3, 2), dtype=np.int32), TypeError, "floating-point"),
    ],
)
def test_modality_weights_rejects_invalid_weight_payloads(
    monkeypatch: pytest.MonkeyPatch,
    weights: np.ndarray,
    error: type[Exception],
    message: str,
) -> None:
    store, graph, layout, selection = _wnn_store(weights)
    monkeypatch.setattr(
        modality_weights_module,
        "_resolve_layout",
        lambda *_args: _resolved_layout(selection),
    )

    with pytest.raises(error, match=message):
        modality_weights(store, graph=graph, layout=layout, show=False)


def test_modality_weights_rejects_non_wnn_and_assay_order_mismatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, graph, layout, selection = _wnn_store(np.full((3, 2), 0.5, dtype=np.float32))
    monkeypatch.setattr(
        modality_weights_module,
        "_resolve_layout",
        lambda *_args: _resolved_layout(selection),
    )
    group = store.zw[artifact_path(graph)]
    provenance = dict(group.attrs["provenance"])
    provenance["parameters"] = {"method": "snn", "assays": ["RNA", "ADT"]}
    group.attrs["provenance"] = provenance
    with pytest.raises(ValueError, match="WNN integrated graph"):
        modality_weights(store, graph=graph, layout=layout, show=False)

    provenance["parameters"] = {"method": "wnn", "assays": ["ADT", "RNA"]}
    group.attrs["provenance"] = provenance
    with pytest.raises(ValueError, match="source order"):
        modality_weights(store, graph=graph, layout=layout, show=False)


def _set_parameters(**parameters: Any) -> Callable[[dict[str, Any]], None]:
    def mutate(attrs: dict[str, Any]) -> None:
        attrs["provenance"]["parameters"].update(parameters)

    return mutate


def _set_source(index: int, field: str, value: Any) -> Callable[[dict[str, Any]], None]:
    def mutate(attrs: dict[str, Any]) -> None:
        attrs["provenance"]["inputs"][f"source_{index}"][field] = value

    return mutate


def _drop_source(index: int) -> Callable[[dict[str, Any]], None]:
    def mutate(attrs: dict[str, Any]) -> None:
        del attrs["provenance"]["inputs"][f"source_{index}"]

    return mutate


def _replace_source(index: int, value: Any) -> Callable[[dict[str, Any]], None]:
    def mutate(attrs: dict[str, Any]) -> None:
        attrs["provenance"]["inputs"][f"source_{index}"] = value

    return mutate


def _set_attr(name: str, value: Any) -> Callable[[dict[str, Any]], None]:
    def mutate(attrs: dict[str, Any]) -> None:
        if value is None:
            del attrs[name]
        else:
            attrs[name] = value

    return mutate


def _set_operation(attrs: dict[str, Any]) -> None:
    attrs["provenance"]["operation"] = "integrate_graphs"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            _set_attr("complete", False),
            "Integrated graph artifact is unavailable or incomplete",
        ),
        (_set_operation, "graph must be produced by integrate_assays"),
        (
            _set_parameters(assays="RNA"),
            "WNN graph provenance has no ordered assay list",
        ),
        (
            _set_parameters(assays=["RNA"]),
            "WNN graph provenance has an invalid ordered assay list",
        ),
        (
            _set_parameters(assays=["RNA", "RNA"]),
            "WNN graph provenance has an invalid ordered assay list",
        ),
        (
            _set_parameters(assays=["RNA", ""]),
            "WNN graph provenance has an invalid ordered assay list",
        ),
        (_drop_source(1), "WNN graph sources do not match its ordered assay list"),
        (_replace_source(1, "ADT"), "WNN graph source provenance is malformed"),
        (
            _set_source(
                1, "coordinates", _ref("neighbors", "1", assay="ADT").to_dict()
            ),
            "WNN coordinate source has an invalid artifact kind",
        ),
        (
            _set_source(
                1, "coordinates", _ref("reduction", "1", assay="RNA").to_dict()
            ),
            "WNN coordinate source order does not match its assays",
        ),
        (_set_attr("assays", None), "WNN graph payload has no ordered assay list"),
        (_set_attr("assays", "RNA,ADT"), "WNN graph payload has no ordered assay list"),
        (
            _set_attr("assays", ["ADT", "RNA"]),
            "WNN graph payload assay order disagrees with provenance",
        ),
    ],
)
def test_modality_weights_rejects_inconsistent_wnn_records(
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    store, graph, layout, selection = _wnn_store(np.full((3, 2), 0.5, dtype=np.float32))
    monkeypatch.setattr(
        modality_weights_module,
        "_resolve_layout",
        lambda *_args: _resolved_layout(selection),
    )
    group = store.zw[artifact_path(graph)]
    attrs = copy.deepcopy(dict(group.attrs))
    mutate(attrs)
    group.attrs.clear()
    group.attrs.update(attrs)

    with pytest.raises(ValueError) as raised:
        modality_weights(store, graph=graph, layout=layout, show=False)

    assert raised.value.args == (message,)


def test_modality_weights_requires_an_integrated_graph_with_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, graph, layout, selection = _wnn_store(np.full((3, 2), 0.5, dtype=np.float32))
    monkeypatch.setattr(
        modality_weights_module,
        "_resolve_layout",
        lambda *_args: _resolved_layout(selection),
    )

    with pytest.raises(TypeError, match="^graph must be an ArtifactRef$"):
        modality_weights(store, graph=graph.to_dict(), layout=layout, show=False)
    for wrong in (
        _ref("connectivity_map", "c", assay="RNA"),
        _ref("cell_selection", "c"),
    ):
        with pytest.raises(
            ValueError,
            match="^graph must identify a datastore integrated_graph artifact$",
        ):
            modality_weights(store, graph=wrong, layout=layout, show=False)

    del store.zw[artifact_path(graph)]["modality_weights"]
    with pytest.raises(ValueError, match="^WNN graph has no modality_weights array$"):
        modality_weights(store, graph=graph, layout=layout, show=False)

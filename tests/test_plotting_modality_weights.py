from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
import zarr
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
    np.testing.assert_allclose(result.tables["weights"].to_numpy(), weights)
    assert result.provenance.n_cells == 3
    assert result.provenance.extras["assays"] == ["RNA", "ADT"]
    assert result.provenance.extras["cell_selection"] == selection.to_dict()
    for axis in result.axes.values():
        assert axis.collections[0].get_clim() == (0.0, 1.0)
    result.close()


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

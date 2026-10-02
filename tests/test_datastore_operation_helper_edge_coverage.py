from types import SimpleNamespace

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.datastore._operations import trajectory as trajectory_operations
from scarf.datastore._operations.features import (
    _statistical_normalization,
    _statistical_storage_columns,
)
from scarf.features.statistical import distinct_label_keys
from scarf.graph import distances as graph_distances
from scarf.metadata.selection import NamedCellArtifact
from scarf.quality_control.filtering import validate_named_cell_artifacts
from scarf.storage.artifacts import ArtifactRef, artifact_path
from scarf.storage.errors import ArtifactResolutionError


def _ref(kind: str, token: str = "a") -> ArtifactRef:
    return ArtifactRef(
        scope="assay",
        kind=kind,
        artifact_id=token * 64,
        assay="RNA",
    )


def _connectivity_payload(
    edges: np.ndarray,
    weights: np.ndarray,
    *,
    n_cells: int = 3,
    n_neighbors: int = 2,
) -> tuple[zarr.Group, ArtifactRef]:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    ref = _ref("connectivity_map")
    group = root.create_group(artifact_path(ref))
    group.attrs.update({"n_cells": n_cells, "n_neighbors": n_neighbors})
    group.create_array("edges", data=edges, chunks=(2, 2))
    group.create_array("weights", data=weights, chunks=(2,))
    return root, ref


def test_statistical_helpers_reject_unknown_contract_values() -> None:
    with pytest.raises(ValueError, match="Unknown statistical test method"):
        _statistical_storage_columns("permutation")
    with pytest.raises(ValueError, match="normalization.source"):
        _statistical_normalization(SimpleNamespace(source="scaled", transform="none"))
    with pytest.raises(ValueError, match="normalization.transform"):
        _statistical_normalization(SimpleNamespace(source="raw", transform="sqrt"))
    assert distinct_label_keys(["same", "same"]) == [0, 1]
    assert distinct_label_keys(["a", "b"]) == ["a", "b"]


def test_named_cell_artifact_collection_validation() -> None:
    labels = NamedCellArtifact("labels", _ref("cluster_labels", "b"))
    assert validate_named_cell_artifacts(
        [labels], expected_kind="cluster_labels", label="sources"
    ) == [labels]
    with pytest.raises(TypeError, match="NamedCellArtifact"):
        validate_named_cell_artifacts(
            [object()], expected_kind="cluster_labels", label="sources"
        )
    with pytest.raises(ValueError, match="cluster_labels"):
        validate_named_cell_artifacts(
            [NamedCellArtifact("cells", _ref("cell_selection", "c"))],
            expected_kind="cluster_labels",
            label="sources",
        )
    with pytest.raises(ValueError, match="unique semantic names"):
        validate_named_cell_artifacts(
            [labels, NamedCellArtifact("labels", _ref("cluster_labels", "d"))],
            expected_kind="cluster_labels",
            label="sources",
        )


def test_graph_payload_error_and_dimension_contracts() -> None:
    ref = _ref("connectivity_map")
    error = graph_distances.payload_error(ref, "broken payload")
    assert error.code == "corrupt_payload"
    assert error.context["artifact_id"] == ref.artifact_id

    for attrs in (
        {"n_cells": True, "n_neighbors": 2},
        {"n_cells": 3, "n_neighbors": 0},
        {"n_cells": 2**32, "n_neighbors": 2},
    ):
        with pytest.raises(ArtifactResolutionError, match="invalid n_cells"):
            graph_distances._payload_dimensions(SimpleNamespace(attrs=attrs), ref)
    with pytest.raises(ArtifactResolutionError, match="invalid neighbor count"):
        graph_distances._payload_dimensions(
            SimpleNamespace(attrs={"n_cells": 3, "n_neighbors": 3}), ref
        )
    assert graph_distances._payload_dimensions(
        SimpleNamespace(attrs={"n_cells": np.int64(3), "n_neighbors": np.int64(2)}),
        ref,
    ) == (3, 2)


def test_graph_payload_block_rows_use_chunks_and_fallback() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    array = root.create_array("values", shape=(7, 2), chunks=(3, 2), dtype=np.float32)
    narrow = root.create_array("narrow", shape=(7,), chunks=(2,), dtype=np.float32)
    assert graph_distances._payload_block_rows(array) == 3
    assert graph_distances._payload_block_rows(array, narrow) == 2
    without_chunks = SimpleNamespace(chunks=None, shape=(0, 2))
    assert graph_distances._payload_block_rows(without_chunks) == 1


def test_connectivity_payload_rejects_missing_and_misshapen_arrays() -> None:
    ref = _ref("connectivity_map")
    empty_root = zarr.open_group(store=MemoryStore(), mode="w")
    with pytest.raises(ArtifactResolutionError, match="payload is unreadable"):
        graph_distances.validate_connectivity_payload(empty_root, ref)
    with pytest.raises(ValueError, match="requires a connectivity_map"):
        graph_distances.validate_connectivity_payload(empty_root, _ref("neighbors"))

    root, ref = _connectivity_payload(
        np.zeros((5, 2), dtype=np.uint32),
        np.ones(5, dtype=np.float32),
    )
    with pytest.raises(ArtifactResolutionError, match="stored dimensions"):
        graph_distances.validate_connectivity_payload(root, ref)


def test_connectivity_payload_rejects_unreadable_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenArray:
        metadata = SimpleNamespace(shards=None)

        def __init__(self, shape: tuple[int, ...], dtype: np.dtype) -> None:
            self.shape = shape
            self.ndim = len(shape)
            self.dtype = dtype
            self.chunks = (2,) + shape[1:]

        def __getitem__(self, key: object) -> np.ndarray:
            raise OSError("storage read failed")

    class Payload(dict[str, BrokenArray]):
        attrs = {"n_cells": 3, "n_neighbors": 2}

    payload = Payload(
        edges=BrokenArray((6, 2), np.dtype(np.uint32)),
        weights=BrokenArray((6,), np.dtype(np.float32)),
    )
    monkeypatch.setattr(graph_distances, "artifact_group", lambda root, ref: payload)
    monkeypatch.setattr(graph_distances, "as_zarr_array", lambda node, name: node)
    with pytest.raises(ArtifactResolutionError, match="arrays are unreadable"):
        graph_distances.validate_connectivity_payload(
            object(), _ref("connectivity_map")
        )


def test_connectivity_payload_rejects_values_and_row_geometry() -> None:
    edges = np.asarray(
        [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]], dtype=np.uint32
    )
    weights = np.ones(6, dtype=np.float32)

    invalid_edges = edges.copy()
    invalid_edges[0, 0] = 3
    root, ref = _connectivity_payload(invalid_edges, weights)
    with pytest.raises(ArtifactResolutionError, match="invalid edge"):
        graph_distances.validate_connectivity_payload(root, ref)

    unbalanced = edges.copy()
    unbalanced[2, 0] = 0
    root, ref = _connectivity_payload(unbalanced, weights)
    with pytest.raises(ArtifactResolutionError, match="rows do not match"):
        graph_distances.validate_connectivity_payload(root, ref)

    root, ref = _connectivity_payload(edges, weights)
    assert graph_distances.validate_connectivity_payload(root, ref) == 3
    assert graph_distances.validate_integration_source_payload(root, ref) == 3
    with pytest.raises(ArtifactResolutionError, match="unsupported artifact kind"):
        graph_distances.validate_integration_source_payload(root, _ref("reduction"))


def test_trajectory_identity_helpers_cover_drift_and_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = SimpleNamespace(name="RNA", normMethod=object(), sf=None)

    monkeypatch.setattr(
        trajectory_operations,
        "callable_identity",
        lambda method: {"callable": "stable"},
    )
    trajectory_operations._validate_normalization_identity(
        stored,
        normalization_method={"callable": "stable"},
        size_factor=None,
        context="diffusion",
    )
    stored.sf = True
    with pytest.raises(ValueError, match="normalization settings changed"):
        trajectory_operations._validate_normalization_identity(
            stored,
            normalization_method={"callable": "stable"},
            size_factor=None,
            context="diffusion",
        )
    stored.sf = 2.0
    with pytest.raises(ValueError, match="normalization settings changed"):
        trajectory_operations._validate_normalization_identity(
            stored,
            normalization_method={"callable": "stable"},
            size_factor=None,
            context="diffusion",
        )
    stored.sf = None
    with pytest.raises(ValueError, match="normalization settings changed"):
        trajectory_operations._validate_normalization_identity(
            stored,
            normalization_method={"callable": "changed"},
            size_factor=None,
            context="diffusion",
        )

    monkeypatch.setattr(
        trajectory_operations,
        "callable_identity",
        lambda method: (_ for _ in ()).throw(ValueError("unstable")),
    )
    with pytest.raises(ValueError, match="normalization settings changed"):
        trajectory_operations._validate_normalization_identity(
            stored,
            normalization_method={"callable": "stable"},
            size_factor=None,
            context="diffusion",
        )


def test_trajectory_feature_and_parameter_guards() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    feature_data = root.create_group("featureData")
    feature_data.create_array("names", data=np.asarray(["a", "b"]))
    feature_data.create_array("ids", data=np.asarray(["a"]))
    assay = SimpleNamespace(z=root, name="RNA")
    with pytest.raises(ValueError, match="do not align"):
        trajectory_operations._feature_identity_arrays(assay)

    with pytest.raises(TypeError, match="ArtifactRef"):
        trajectory_operations._resolve_feature_indices(object(), assay, object())

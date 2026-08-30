from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

import scarf.assay
from scarf.features.enrichment.results import EnrichmentResult
from scarf.features.markers import batching
from scarf.matrix._reductions import _Reduction
from scarf.matrix.blocks import Block
from scarf.matrix.chunked import ChunkedArray
from scarf.storage.budget import ResourceBudget
from scarf.storage.refs import ArtifactRef
from scarf.trajectory.results import (
    FateMappingResult,
    PseudotimeAggregationResult,
    PseudotimeMarkerResult,
    PseudotimeScoreResult,
)


class _ReductionParent:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def _reduce(
        self,
        op: str,
        axis: int | None,
        nthreads: int | None,
        msg: str | None,
    ) -> np.ndarray:
        self.calls.append((op, axis, nthreads, msg))
        return np.array([1.0, 2.0, 4.0])


class _BlockParent:
    out_cols = 3
    dtype = np.dtype(np.float64)

    def _materialize_range(self, start: int, end: int) -> np.ndarray:
        return np.arange(start * 3, end * 3, dtype=np.float64).reshape(-1, 3)


def _ref(
    kind: str,
    token: str,
    *,
    scope: str = "assay",
    assay: str | None = "RNA",
) -> ArtifactRef:
    return ArtifactRef(
        scope=cast(Any, scope),
        kind=kind,
        artifact_id=token * 64,
        assay=assay,
    )


def _trajectory_refs() -> dict[str, ArtifactRef]:
    return {
        "result": _ref("fate_map", "a"),
        "graph": _ref("connectivity_map", "b"),
        "pseudotime": _ref("pseudotime", "c"),
        "sink_labels": _ref("cluster_labels", "d"),
        "cells": _ref("cell_selection", "e", scope="datastore", assay=None),
        "features": _ref("feature_selection", "f"),
    }


def test_reduction_array_protocol_and_cached_collection_behavior() -> None:
    parent = _ReductionParent()
    reduction = _Reduction(cast(Any, parent), "sum", 0)

    np.testing.assert_array_equal(reduction.compute(3, "sum"), [1.0, 2.0, 4.0])
    np.testing.assert_array_equal(reduction.compute(9, "ignored"), [1.0, 2.0, 4.0])
    assert parent.calls == [("sum", 0, 3, "sum")]
    assert np.asarray(reduction, dtype=np.float32).dtype == np.float32
    assert reduction.shape == (3,)
    assert len(reduction) == 3
    assert list(reduction) == [1.0, 2.0, 4.0]
    assert reduction[1] == 2.0
    assert repr(reduction) == "<deferred sum(axis=0)>"


def test_reduction_ufunc_and_binary_operator_protocols() -> None:
    left = _Reduction(cast(Any, _ReductionParent()), "sum", 0)
    right = _Reduction(cast(Any, _ReductionParent()), "mean", 0)

    assert left.__array_ufunc__(np.add, "reduce", left) is NotImplemented
    np.testing.assert_array_equal(np.add(left, right), [2.0, 4.0, 8.0])
    np.testing.assert_array_equal(left * 2, [2.0, 4.0, 8.0])
    np.testing.assert_array_equal(2 * left, [2.0, 4.0, 8.0])
    np.testing.assert_array_equal(left / 2, [0.5, 1.0, 2.0])
    np.testing.assert_array_equal(8 / left, [8.0, 4.0, 2.0])
    np.testing.assert_array_equal(left + 2, [3.0, 4.0, 6.0])
    np.testing.assert_array_equal(2 + left, [3.0, 4.0, 6.0])
    np.testing.assert_array_equal(left - 2, [-1.0, 0.0, 2.0])
    np.testing.assert_array_equal(8 - left, [7.0, 6.0, 4.0])
    np.testing.assert_array_equal(left > 2, [False, False, True])
    np.testing.assert_array_equal(left < 2, [True, False, False])
    np.testing.assert_array_equal(left >= 2, [False, True, True])
    np.testing.assert_array_equal(left <= 2, [True, True, False])


def test_block_shape_dtype_materialization_and_row_selection() -> None:
    parent = cast(Any, _BlockParent())
    block = Block(parent, 1, 4)

    assert block.shape == (3, 3)
    assert block.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(
        block.compute(), np.arange(3, 12, dtype=np.float64).reshape(3, 3)
    )
    assert np.asarray(block, dtype=np.float32).dtype == np.float32

    selected = block[[2, 0]]
    tuple_selected = block[([1, 0], slice(None))]
    assert selected.shape == (2, 3)
    np.testing.assert_array_equal(
        selected.compute(), np.array([[9.0, 10.0, 11.0], [3.0, 4.0, 5.0]])
    )
    np.testing.assert_array_equal(
        tuple_selected.compute(), np.array([[6.0, 7.0, 8.0], [3.0, 4.0, 5.0]])
    )


def test_marker_feature_column_chunk_uses_each_backing_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, int]] = []

    def fake_chunk(backing: object, *, featureAxis: int) -> int:
        calls.append((backing, featureAxis))
        return 7 + featureAxis

    class FakeRNA:
        def __init__(self, raw_data_t: object | None) -> None:
            self.rawDataT = raw_data_t
            self.rawData = SimpleNamespace(_backing="rna-cells")

    monkeypatch.setattr(batching, "_feature_column_chunk", fake_chunk)
    monkeypatch.setattr(scarf.assay, "RNAassay", FakeRNA)

    assert batching.feature_column_chunk(FakeRNA("rna-features"), 99) == 7
    assert (
        batching.feature_column_chunk(
            SimpleNamespace(rawData=SimpleNamespace(_backing="other")), 99
        )
        == 8
    )
    assert batching.feature_column_chunk(SimpleNamespace(), 4) == 4
    assert calls == [("rna-features", 0), ("other", 1)]


def test_marker_batch_size_respects_chunk_feature_and_memory_caps() -> None:
    resources = ResourceBudget(memoryBytes=6_400, workers=8)
    with pytest.warns(DeprecationWarning):
        assert (
            batching.resolve_marker_gene_batch_size(
                n_features=5,
                n_cells=10,
                column_chunk=8,
                resources=resources,
            )
            == 5
        )

    with (
        pytest.warns(DeprecationWarning),
        pytest.raises(MemoryError, match="does not fit"),
    ):
        batching.resolve_marker_gene_batch_size(
            n_features=0,
            n_cells=0,
            column_chunk=0,
            resources=ResourceBudget(memoryBytes=1, workers=1),
        )


def test_fate_and_pseudotime_results_reject_each_invalid_dimension() -> None:
    refs = _trajectory_refs()
    fate_args = (
        refs["result"],
        refs["graph"],
        refs["pseudotime"],
        refs["sink_labels"],
        refs["cells"],
    )
    with pytest.raises(ValueError, match="two-dimensional"):
        FateMappingResult(
            *fate_args,
            ("A",),
            np.array([1.0]),
            np.array([True]),
        )
    with pytest.raises(ValueError, match="one-dimensional"):
        FateMappingResult(
            *fate_args,
            ("A",),
            np.array([[1.0]]),
            np.array([[True]]),
        )
    with pytest.raises(ValueError, match="sink labels"):
        FateMappingResult(
            *fate_args,
            ("A", "B"),
            np.array([[1.0]]),
            np.array([True]),
        )

    with pytest.raises(ValueError, match="one-dimensional"):
        PseudotimeScoreResult(
            refs["pseudotime"],
            refs["graph"],
            refs["cells"],
            np.array([[0.0, 1.0]]),
            np.array([True, True]),
        )


def test_pseudotime_marker_rejects_a_feature_selection_from_another_assay() -> None:
    refs = _trajectory_refs()
    table = pd.DataFrame(
        {
            "feature_index": [0],
            "feature_name": ["g0"],
            "r_value": [1.0],
            "p_value": [0.0],
        }
    )
    with pytest.raises(ValueError, match="belong to its assay"):
        PseudotimeMarkerResult(
            _ref("pseudotime_markers", "1"),
            table,
            "RNA",
            refs["cells"],
            _ref("feature_selection", "2", assay="ATAC"),
            refs["pseudotime"],
        )


def _aggregation(
    *,
    data_shape: tuple[int, ...] = (2, 3),
    feature_indices: np.ndarray | None = None,
    feature_clusters: np.ndarray | None = None,
    feature_selection: ArtifactRef | None = None,
) -> PseudotimeAggregationResult:
    refs = _trajectory_refs()
    return PseudotimeAggregationResult(
        _ref("pseudotime_aggregation", "3"),
        cast(ChunkedArray, SimpleNamespace(shape=data_shape)),
        np.array([0, 1]) if feature_indices is None else feature_indices,
        np.array([1, 2]) if feature_clusters is None else feature_clusters,
        "RNA",
        refs["cells"],
        refs["features"] if feature_selection is None else feature_selection,
        refs["pseudotime"],
    )


def test_pseudotime_aggregation_rejects_invalid_shapes_and_selection() -> None:
    with pytest.raises(ValueError, match="belong to its assay"):
        _aggregation(feature_selection=_ref("feature_selection", "4", assay="ATAC"))
    with pytest.raises(ValueError, match="two-dimensional"):
        _aggregation(data_shape=(2, 3, 4))
    with pytest.raises(ValueError, match="one-dimensional"):
        _aggregation(feature_indices=np.array([[0, 1]]))
    with pytest.raises(ValueError, match="Feature indices"):
        _aggregation(feature_indices=np.array([0]))


def test_pseudotime_aggregation_feature_identity_is_immutable_and_aligned() -> None:
    aggregation = _aggregation()
    with pytest.raises(RuntimeError, match="feature names"):
        _ = aggregation.feature_names
    with pytest.raises(RuntimeError, match="feature IDs"):
        _ = aggregation.feature_ids
    with pytest.raises(ValueError, match="names and IDs must align"):
        aggregation._attach_feature_identity(np.array(["g0", "g1"]), np.array(["id0"]))
    with pytest.raises(ValueError, match="aggregation rows"):
        aggregation._attach_feature_identity(np.array(["g0"]), np.array(["id0"]))

    aggregation._attach_feature_identity(
        np.array(["g0", "g1"]), np.array(["id0", "id1"])
    )
    assert aggregation.feature_names.tolist() == ["g0", "g1"]
    assert aggregation.feature_ids.tolist() == ["id0", "id1"]
    assert not aggregation.feature_names.flags.writeable
    assert not aggregation.feature_ids.flags.writeable


def _enrichment(
    *,
    artifact: ArtifactRef | None = None,
    feature_selection: ArtifactRef | None = None,
    cell_selection: ArtifactRef | None = None,
    data_shape: tuple[int, ...] = (2, 2),
    source_names: np.ndarray | None = None,
    source_sizes: np.ndarray | None = None,
    cell_index: np.ndarray | None = None,
    method: str = "waggr",
) -> EnrichmentResult:
    refs = _trajectory_refs()
    return EnrichmentResult(
        data=cast(ChunkedArray, SimpleNamespace(shape=data_shape)),
        source_names=(
            np.array(["set_a", "set_b"]) if source_names is None else source_names
        ),
        source_sizes=(np.array([2, 3]) if source_sizes is None else source_sizes),
        cell_index=(np.array([0, 1]) if cell_index is None else cell_index),
        artifact=(_ref("enrichment_scores", "5") if artifact is None else artifact),
        storage_path="artifacts/enrichment",
        assay="RNA",
        cell_selection=refs["cells"] if cell_selection is None else cell_selection,
        feature_selection=(
            refs["features"] if feature_selection is None else feature_selection
        ),
        method=method,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"artifact": _ref("enrichment_scores", "6", assay="ATAC")}, "artifact"),
        (
            {"feature_selection": _ref("feature_selection", "7", assay="ATAC")},
            "feature_selection",
        ),
        ({"cell_selection": _ref("cell_selection", "8")}, "cell_selection"),
        ({"data_shape": (2, 2, 1)}, "two-dimensional"),
        ({"source_names": np.array([["a", "b"]])}, "one-dimensional"),
        ({"cell_index": np.array([0])}, "cell indices"),
        ({"source_names": np.array(["a"])}, "source names"),
        ({"source_sizes": np.array([1])}, "names and sizes"),
        ({"source_names": np.array(["a", "a"])}, "names must be unique"),
        ({"cell_index": np.array([0, 0])}, "indices must be unique"),
        ({"source_sizes": np.array([1, 0])}, "sizes must be positive"),
        ({"method": "unknown"}, "Unknown enrichment method"),
    ],
)
def test_enrichment_result_rejects_invalid_contracts(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _enrichment(**cast(Any, kwargs))

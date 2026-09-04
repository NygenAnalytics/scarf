"""Fast unit tests for immutable enrichment payload helpers."""

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.datastore._operations.enrichment_store import (
    _enrichment_artifact_matches,
    _write_enrichment_slot,
)


def _payload() -> dict[str, object]:
    return {
        "attrs": {"method": "waggr", "layout": "cells_by_sources"},
        "n_cells": 3,
        "source_names": np.array(["Alpha", "Beta"]),
        "source_sizes": np.array([2, 2], dtype=np.int64),
        "cell_index": np.array([0, 2, 5], dtype=np.int64),
        "matched_feature_index": np.array([1, 3], dtype=np.int64),
        "rank_feature_index": np.array([3, 1], dtype=np.int64),
    }


def test_write_enrichment_slot_persists_and_matches_exact_payload() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    slot = root.create_group("slot")
    payload = _payload()
    scores = np.array(
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        dtype=np.float64,
    )

    _write_enrichment_slot(
        slot,
        score_batches=iter([scores[:2], scores[2:]]),
        **payload,
    )

    assert slot.attrs["complete"] is True
    np.testing.assert_allclose(slot["scores"][:], scores.astype(np.float32))
    np.testing.assert_array_equal(slot["cell_index"][:], [0, 2, 5])
    np.testing.assert_array_equal(slot["rank_feature_index"][:], [3, 1])
    match_payload = {key: value for key, value in payload.items() if key != "n_cells"}
    assert _enrichment_artifact_matches(slot, **match_payload)
    assert not _enrichment_artifact_matches(
        slot,
        **{**match_payload, "attrs": {"method": "aucell"}},
    )
    assert not _enrichment_artifact_matches(
        slot,
        **{**match_payload, "rank_feature_index": None},
    )


@pytest.mark.parametrize(
    ("updates", "score_batches", "message"),
    [
        (
            {"n_cells": 0, "cell_index": np.array([], dtype=np.int64)},
            [],
            "empty or misaligned",
        ),
        (
            {"matched_feature_index": np.array([], dtype=np.int64)},
            [np.ones((3, 2))],
            "no matched features",
        ),
        ({}, [np.ones((3, 1))], "invalid shape"),
        (
            {},
            [np.array([[np.nan, 0.0], [0.0, 1.0], [0.0, 1.0]])],
            "non-finite",
        ),
    ],
)
def test_write_enrichment_slot_rejects_invalid_payloads(
    updates: dict[str, object],
    score_batches: list[np.ndarray],
    message: str,
) -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    slot = root.create_group("slot")
    payload = {**_payload(), **updates}

    with pytest.raises(ValueError, match=message):
        _write_enrichment_slot(
            slot,
            score_batches=iter(score_batches),
            **payload,
        )

    assert slot.attrs.get("complete") is not True

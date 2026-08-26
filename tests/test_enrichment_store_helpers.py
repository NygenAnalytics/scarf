"""Fast unit tests for enrichment store helpers (not marked slow)."""

from types import SimpleNamespace

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.datastore._operations.enrichment_store import (
    _ENRICHMENT_ACTIVE_SLOT,
    _ENRICHMENT_ARTIFACT_RESULTS,
    _ENRICHMENT_LEGACY_ARTIFACTS,
    _ENRICHMENT_RUN_PREFIX,
    _enrichment_artifact_entry,
    _enrichment_artifact_matches,
    _enrichment_artifact_ref,
    _execution_digest,
    _legacy_enrichment_slot,
    _load_enrichment_result,
    _publish_enrichment_artifact,
    _resolve_enrichment_slot,
    _validate_enrichment_label,
    _write_enrichment_slot,
)
from scarf.storage.artifacts import ArtifactRef


def _assay(root: zarr.Group, name: str = "RNA") -> SimpleNamespace:
    return SimpleNamespace(name=name, z=root.create_group(name))


def test_validate_enrichment_label_accepts_and_rejects():
    assert _validate_enrichment_label("waggr_1") == "waggr_1"
    with pytest.raises(TypeError, match="must be a string"):
        _validate_enrichment_label(1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty path component"):
        _validate_enrichment_label("")
    with pytest.raises(ValueError, match="non-empty path component"):
        _validate_enrichment_label(".")
    with pytest.raises(ValueError, match="path separators or control"):
        _validate_enrichment_label("a/b")
    with pytest.raises(ValueError, match="path separators or control"):
        _validate_enrichment_label("a\\b")
    with pytest.raises(ValueError, match="path separators or control"):
        _validate_enrichment_label("bad\nlabel")


def test_execution_digest_is_stable_and_rejects_unsafe_payloads():
    first = _execution_digest({"b": 2, "a": 1})
    second = _execution_digest({"a": 1, "b": 2})
    assert first == second
    assert len(first) == 32
    with pytest.raises(ValueError, match="JSON-safe"):
        _execution_digest({"bad": {1, 2}})
    with pytest.raises(ValueError, match="JSON-safe"):
        _execution_digest({"bad": float("nan")})


def test_resolve_enrichment_slot_uses_active_run_or_rejects_poison():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    label_group = root.create_group("label")
    assert _resolve_enrichment_slot(label_group, label="label") is label_group

    run = label_group.create_group(f"{_ENRICHMENT_RUN_PREFIX}1")
    run.attrs["marker"] = "active-run"
    label_group.attrs[_ENRICHMENT_ACTIVE_SLOT] = f"{_ENRICHMENT_RUN_PREFIX}1"
    resolved = _resolve_enrichment_slot(label_group, label="label")
    assert resolved.path == run.path
    assert resolved.attrs["marker"] == "active-run"

    label_group.attrs[_ENRICHMENT_ACTIVE_SLOT] = "../escape"
    with pytest.raises(ValueError, match="invalid active result"):
        _resolve_enrichment_slot(label_group, label="label")

    label_group.attrs[_ENRICHMENT_ACTIVE_SLOT] = f"{_ENRICHMENT_RUN_PREFIX}missing"
    with pytest.raises(ValueError, match="invalid active result"):
        _resolve_enrichment_slot(label_group, label="label")

    label_group.attrs[_ENRICHMENT_ACTIVE_SLOT] = 12
    with pytest.raises(ValueError, match="invalid active result"):
        _resolve_enrichment_slot(label_group, label="label")


def test_enrichment_artifact_index_round_trip_and_validation():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    assay = _assay(root)
    assert _enrichment_artifact_entry(assay, "missing") is None
    assert _enrichment_artifact_ref(assay, "missing") is None
    assert _legacy_enrichment_slot(assay, "missing") is None

    ref = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="enrichment_scores",
        artifact_id="a" * 64,
    )
    _publish_enrichment_artifact(
        assay,
        "scores",
        ref,
        cell_key="I",
    )
    entry = _enrichment_artifact_entry(assay, "scores")
    assert entry is not None
    assert entry[0] == ref
    assert entry[1] == "I"
    assert _enrichment_artifact_ref(assay, "scores") == ref

    enrichment = assay.z["enrichment"]
    enrichment.attrs[_ENRICHMENT_ARTIFACT_RESULTS] = ["not-a-dict"]
    with pytest.raises(ValueError, match="artifact index is invalid"):
        _enrichment_artifact_entry(assay, "scores")

    enrichment.attrs[_ENRICHMENT_ARTIFACT_RESULTS] = {"scores": "bad"}
    with pytest.raises(ValueError, match="invalid artifact entry"):
        _enrichment_artifact_entry(assay, "scores")

    enrichment.attrs[_ENRICHMENT_ARTIFACT_RESULTS] = {
        "scores": {"artifact": ref.to_dict(), "cell_key": ""}
    }
    with pytest.raises(ValueError, match="invalid execution metadata"):
        _enrichment_artifact_entry(assay, "scores")

    del enrichment.attrs[_ENRICHMENT_ARTIFACT_RESULTS]
    enrichment.attrs[_ENRICHMENT_LEGACY_ARTIFACTS] = {"scores": ref.to_dict()}
    assert _enrichment_artifact_entry(assay, "scores") is None
    enrichment.attrs[_ENRICHMENT_LEGACY_ARTIFACTS] = {"scores": "bad"}
    assert _enrichment_artifact_entry(assay, "scores") is None


def test_write_enrichment_slot_persists_scores_and_rejects_bad_batches():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    slot = root.create_group("slot")
    attrs = {"method": "waggr", "layout": "cells_by_sources"}
    cells = np.array([0, 2, 5], dtype=np.int64)
    matched = np.array([1, 3], dtype=np.int64)
    names = np.array(["Alpha", "Beta"])
    sizes = np.array([2, 2], dtype=np.int64)
    scores = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float64)

    _write_enrichment_slot(
        slot,
        attrs=attrs,
        score_batches=iter([scores[:2], scores[2:]]),
        n_cells=3,
        source_names=names,
        source_sizes=sizes,
        cell_index=cells,
        matched_feature_index=matched,
        rank_feature_index=np.array([3, 1], dtype=np.int64),
    )
    assert slot.attrs["complete"] is True
    np.testing.assert_allclose(slot["scores"][:], scores.astype(np.float32))
    np.testing.assert_array_equal(slot["cell_index"][:], cells)
    np.testing.assert_array_equal(slot["rank_feature_index"][:], [3, 1])
    assert _enrichment_artifact_matches(
        slot,
        attrs=attrs,
        cell_index=cells,
        matched_feature_index=matched,
        source_names=names,
        source_sizes=sizes,
        rank_feature_index=np.array([3, 1], dtype=np.int64),
    )
    assert not _enrichment_artifact_matches(
        slot,
        attrs={**attrs, "method": "aucell"},
        cell_index=cells,
        matched_feature_index=matched,
        source_names=names,
        source_sizes=sizes,
        rank_feature_index=np.array([3, 1], dtype=np.int64),
    )
    assert not _enrichment_artifact_matches(
        slot,
        attrs=attrs,
        cell_index=cells,
        matched_feature_index=matched,
        source_names=names,
        source_sizes=sizes,
        rank_feature_index=None,
    )

    bad = root.create_group("bad")
    with pytest.raises(ValueError, match="empty or misaligned"):
        _write_enrichment_slot(
            bad,
            attrs=attrs,
            score_batches=iter([]),
            n_cells=0,
            source_names=names,
            source_sizes=sizes,
            cell_index=np.array([], dtype=np.int64),
            matched_feature_index=matched,
            rank_feature_index=None,
        )
    with pytest.raises(ValueError, match="no matched features"):
        _write_enrichment_slot(
            bad,
            attrs=attrs,
            score_batches=iter([scores]),
            n_cells=3,
            source_names=names,
            source_sizes=sizes,
            cell_index=cells,
            matched_feature_index=np.array([], dtype=np.int64),
            rank_feature_index=None,
        )
    with pytest.raises(ValueError, match="invalid shape"):
        _write_enrichment_slot(
            bad,
            attrs=attrs,
            score_batches=iter([np.ones((3, 1))]),
            n_cells=3,
            source_names=names,
            source_sizes=sizes,
            cell_index=cells,
            matched_feature_index=matched,
            rank_feature_index=None,
        )
    with pytest.raises(ValueError, match="non-finite"):
        _write_enrichment_slot(
            bad,
            attrs=attrs,
            score_batches=iter([np.array([[np.nan, 0.0], [0.0, 1.0], [0.0, 1.0]])]),
            n_cells=3,
            source_names=names,
            source_sizes=sizes,
            cell_index=cells,
            matched_feature_index=matched,
            rank_feature_index=None,
        )
    assert bad.attrs.get("complete") is False


def test_legacy_enrichment_slot_resolves_active_child():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    assay = _assay(root)
    enrichment = assay.z.create_group("enrichment")
    label = enrichment.create_group("legacy")
    label.create_group(f"{_ENRICHMENT_RUN_PREFIX}a")
    label.attrs[_ENRICHMENT_ACTIVE_SLOT] = f"{_ENRICHMENT_RUN_PREFIX}a"
    resolved = _legacy_enrichment_slot(assay, "legacy")
    assert resolved is not None
    assert resolved.path.endswith(f"{_ENRICHMENT_RUN_PREFIX}a")
    assert _legacy_enrichment_slot(assay, "absent") is None


def test_load_enrichment_result_rejects_unindexed_legacy_slot():
    root = zarr.open_group(store=MemoryStore(), mode="w")
    assay = _assay(root)
    assay.nthreads = 1
    enrichment = assay.z.create_group("enrichment")
    enrichment.create_group("legacy")

    with pytest.raises(KeyError, match="was not found"):
        _load_enrichment_result(
            assay,
            label="legacy",
            sources=None,
            artifact_root=root,
        )

"""Unit tests for writing, matching, and loading enrichment payloads.

The payload tests write slots to memory. The loader tests edit copies of one
small store that holds a WAGGR and an AUCell result.
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.datastore._operations.enrichment_store import (
    _enrichment_artifact_matches,
    _write_enrichment_slot,
)
from scarf.datastore.datastore import DataStore
from scarf.storage.artifacts import artifact_path
from scarf.storage.budget import resolve_budget
from scarf.storage.io_policy import StorageIoPolicy
from tests.storage_helpers import write_count_store


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
        resources=resolve_budget(workers=1),
        io=None,
        **payload,
    )

    # finish_artifact, not the payload writer, marks the artifact complete.
    assert slot.attrs["complete"] is False
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
    assert not _enrichment_artifact_matches(
        slot,
        **{**match_payload, "cell_index": np.array([0, 2, 6], dtype=np.int64)},
    )
    del slot["matched_feature_index"]
    assert not _enrichment_artifact_matches(slot, **match_payload)
    del slot["scores"]
    assert not _enrichment_artifact_matches(slot, **match_payload)


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
        (
            {"source_sizes": np.array([2], dtype=np.int64)},
            [np.ones((3, 2))],
            "source metadata is empty or misaligned",
        ),
        # The writer refuses a short stream before the slot could count it.
        ({}, [np.ones((2, 2))], "Dense stream contains 2 rows, expected 3"),
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
            resources=resolve_budget(workers=1),
            io=None,
            **payload,
        )

    assert slot.attrs.get("complete") is not True


def test_write_enrichment_slot_streams_within_the_datastore_budget(monkeypatch) -> None:
    import scarf.storage.sharding as sharding

    root = zarr.open_group(store=MemoryStore(), mode="w")
    slot = root.create_group("slot")
    resources = resolve_budget("64M", workers=1)
    io = StorageIoPolicy(writeWorkers=1)
    received: dict[str, object] = {}
    writer = sharding.write_dense_from_row_batches

    def recording_writer(*args, **kwargs):
        received.update(resources=kwargs["resources"], io=kwargs["io"])
        return writer(*args, **kwargs)

    monkeypatch.setattr(sharding, "write_dense_from_row_batches", recording_writer)
    _write_enrichment_slot(
        slot,
        score_batches=iter([np.ones((3, 2))]),
        resources=resources,
        io=io,
        **_payload(),
    )

    assert received == {"resources": resources, "io": io}


_COUNTS = np.array(
    [
        [6, 5, 4, 3, 2, 1],
        [1, 2, 3, 4, 5, 6],
        [6, 1, 5, 2, 4, 3],
        [3, 6, 1, 5, 2, 4],
        [2, 5, 6, 1, 4, 3],
    ]
)
_NETWORK = pd.DataFrame(
    {
        "source": ["A"] * 3 + ["B"] * 3,
        "target": [f"RNA{index}" for index in range(6)],
        "weight": [1.0, -1.0, 2.0, 0.5, 1.5, -0.5],
    }
)


def _open(path) -> DataStore:
    return DataStore(
        str(path), default_assay="RNA", min_features_per_cell=0, nthreads=1
    )


@pytest.fixture(scope="module")
def enrichment_store(tmp_path_factory):
    """A store with WAGGR and AUCell results over every cell and gene."""
    path = tmp_path_factory.mktemp("enrichment") / "store.zarr"
    write_count_store(str(path), {"RNA": _COUNTS}, "uint32")
    store = _open(path)
    cells = store.snapshot_cell_selection()
    features = store.select_all_features(from_assay="RNA")
    refs = {
        "waggr": store.run_waggr(_NETWORK, cells, features=features, tmin=3),
        "aucell": store.run_aucell(
            _NETWORK, cells, features=features, tmin=3, n_up=4, tie_seed=7
        ),
    }
    saved = tmp_path_factory.mktemp("saved_results")
    for method, ref in refs.items():
        # Each result loads before a test edits it.
        assert store.get_enrichment(ref).method == method
        shutil.copytree(path / artifact_path(ref), saved / method)
    return store, refs, saved


@pytest.fixture
def editable_results(enrichment_store):
    """Yield the result store, and restore each result after a test edits it."""
    store, refs, saved = enrichment_store
    yield store, refs
    root = Path(store.zarr_loc)
    for method, ref in refs.items():
        shutil.rmtree(root / artifact_path(ref))
        shutil.copytree(saved / method, root / artifact_path(ref))


def _set(name, value):
    return lambda slot: slot.attrs.update({name: value})


def _delete(name):
    return lambda slot: slot.__delitem__(name)


def _delete_attribute(name):
    return lambda slot: slot.attrs.__delitem__(name)


def _replace(name, data):
    def edit(slot):
        del slot[name]
        slot.create_array(name, data=np.asarray(data))

    return edit


def _record(section, name, value):
    def edit(slot):
        provenance = dict(slot.attrs["provenance"])
        provenance[section] = {**provenance[section], name: value}
        slot.attrs["provenance"] = provenance

    return edit


_LOADER_EDITS = {
    "operation": ("waggr", _set("method", "aucell"), "mismatched artifact operation"),
    "method-metadata": (
        "waggr",
        _delete_attribute("waggr_mode"),
        "missing method metadata",
    ),
    "algorithm": ("waggr", _set("algorithm_version", True), "unsupported algorithm"),
    "tmin": ("waggr", _set("tmin", 0), "invalid tmin metadata"),
    "digest": ("waggr", _set("cell_digest", ""), "invalid cell_digest metadata"),
    "size-factor": ("waggr", _set("size_factor", -1.0), "WAGGR slot .* method meta"),
    "mode": ("waggr", _set("waggr_mode", "median"), "WAGGR slot .* method meta"),
    "normalization-record": (
        "waggr",
        _record("parameters", "normalization_method", "custom"),
        "normalization provenance is invalid",
    ),
    "parameters": ("waggr", _set("tmin", 4), "parameters do not match its metadata"),
    "selection-record": (
        "waggr",
        _record("inputs", "cell_selection", "everyone"),
        "missing selection provenance",
    ),
    "selection-digest": (
        "waggr",
        _set("cell_digest", "0" * 64),
        "'cell_selection' input does not match its metadata",
    ),
    "arrays": ("waggr", _delete("source_sizes"), "missing required arrays"),
    "unexpected-rank": (
        "waggr",
        lambda slot: slot.create_array("rank_feature_index", data=np.arange(6)),
        "unexpected rank metadata",
    ),
    "dimensions": (
        "waggr",
        _replace("cell_index", [[0, 1, 2, 3, 4]]),
        "invalid array dimensions",
    ),
    "index-dtype": ("waggr", _replace("cell_index", np.arange(5.0)), "index dtypes"),
    "size-dtype": ("waggr", _replace("source_sizes", [3.0, 3.0]), "invalid source"),
    "score-shape": ("waggr", _replace("cell_index", np.arange(6)), "misaligned"),
    "source-alignment": (
        "waggr",
        _replace("source_sizes", [3, 3, 3]),
        "source metadata is misaligned",
    ),
    "duplicate-source": ("waggr", _replace("source_names", ["A", "A"]), "duplicate"),
    "empty-source": ("waggr", _replace("source_names", ["A", ""]), "empty source"),
    "empty-set": ("waggr", _replace("source_sizes", [3, 0]), "invalid source sizes"),
    "duplicate-cell": (
        "waggr",
        _replace("cell_index", [0, 0, 2, 3, 4]),
        "duplicate cell indices",
    ),
    "cell-order": (
        "waggr",
        _replace("cell_index", [4, 3, 2, 1, 0]),
        "mismatched cell digest",
    ),
    "matched-order": (
        "waggr",
        _replace("matched_feature_index", [5, 4, 3, 2, 1, 0]),
        "invalid matched features",
    ),
    "n-up": ("aucell", _set("n_up", 1), "AUCell slot .* invalid method metadata"),
    "tie-seed": ("aucell", _set("tie_seed", -1), "AUCell slot .* method metadata"),
    "missing-rank": (
        "aucell",
        _delete("rank_feature_index"),
        "missing its ranking universe",
    ),
    "rank-dimensions": (
        "aucell",
        _replace("rank_feature_index", [[0, 1, 2, 3, 4, 5]]),
        "invalid rank features",
    ),
    "rank-duplicates": (
        "aucell",
        _replace("rank_feature_index", [0, 0, 1, 2, 3, 4]),
        "invalid rank features",
    ),
    "rank-digest": (
        "aucell",
        _replace("rank_feature_index", [0, 1, 2, 3, 4]),
        "mismatched feature digest",
    ),
    "unmatched": (
        "aucell",
        _replace("matched_feature_index", [0, 1, 2, 3, 4, 7]),
        "unmatched network features",
    ),
}


@pytest.mark.parametrize("edit", sorted(_LOADER_EDITS))
def test_enrichment_loader_rejects_payloads_edited_outside_scarf(
    editable_results, edit
) -> None:
    store, refs = editable_results
    method, apply_edit, message = _LOADER_EDITS[edit]
    apply_edit(store.zw[artifact_path(refs[method])])

    with pytest.raises(ValueError, match=message):
        store.get_enrichment(refs[method])


def test_enrichment_loader_selects_sources_by_name_only(enrichment_store) -> None:
    writable, refs, _ = enrichment_store
    store = DataStore(writable.zarr_loc, default_assay="RNA", zarr_mode="r")
    loaded = store.get_enrichment(refs["aucell"])

    reordered = store.get_enrichment(refs["aucell"], sources=["B", "A"])

    np.testing.assert_array_equal(loaded.source_names, ["A", "B"])
    np.testing.assert_array_equal(reordered.source_names, ["B", "A"])
    np.testing.assert_array_equal(
        reordered.data.compute(), loaded.data.compute()[:, [1, 0]]
    )
    with pytest.raises(TypeError, match="not a string"):
        store.get_enrichment(refs["aucell"], sources="A")
    with pytest.raises(TypeError, match="only strings"):
        store.get_enrichment(refs["aucell"], sources=["A", 1])

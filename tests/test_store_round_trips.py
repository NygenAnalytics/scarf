"""Metadata round trips that object stores pay once per request."""

import threading
from collections import Counter

import numpy as np
import pytest
import zarr
from zarr.storage import LocalStore, MemoryStore

import scarf.storage.stores as stores
from scarf.datastore.datastore import DataStore
from scarf.metadata import MetaData
from scarf.storage.artifacts import (
    ArtifactRef,
    artifact_path,
    find_reusable_artifacts,
    make_provenance,
    new_artifact_id,
)
from scarf.storage.stores import (
    REMOTE_METADATA_WORKERS,
    metadata_workers,
    run_concurrently,
)
from tests.store_probes import StoreProbe, wrap_recording_store


def _gets(probe: StoreProbe) -> Counter[str]:
    return Counter(key for kind, key in probe.ops if kind == "get")


def _recorded_root() -> tuple[zarr.Group, StoreProbe]:
    probe = StoreProbe()
    store = wrap_recording_store(MemoryStore(), probe=probe)
    return zarr.open_group(store=store, mode="w"), probe


def test_run_concurrently_keeps_task_order_and_overlaps_tasks() -> None:
    assert run_concurrently([lambda: 1, lambda: 2], workers=1) == [1, 2]

    # Each task waits for the others, so the call finishes only if all overlap.
    barrier = threading.Barrier(3, timeout=10)

    def waiting(value: int):
        def run() -> int:
            barrier.wait()
            return value

        return run

    assert run_concurrently([waiting(index) for index in range(3)], workers=3) == [
        0,
        1,
        2,
    ]

    def failing() -> int:
        raise ValueError("unreadable node")

    with pytest.raises(ValueError, match="unreadable node"):
        run_concurrently([lambda: 1, failing], workers=2)


def test_metadata_workers_overlap_only_remote_stores(monkeypatch) -> None:
    group = zarr.open_group(store=MemoryStore(), mode="w")
    assert metadata_workers(group) == 1
    monkeypatch.setattr(stores, "is_remote_datastore", lambda *args, **kwargs: True)
    assert metadata_workers(group) == REMOTE_METADATA_WORKERS


def test_metadata_reads_each_column_description_once() -> None:
    root, probe = _recorded_root()
    cells = root.create_group("cellData")
    cells.create_array("I", data=np.array([True, False, True]), chunks=(3,))
    cells.create_array("ids", data=np.array(["a", "b", "c"]), chunks=(3,))
    cells.create_array("score", data=np.arange(3.0), chunks=(3,))

    probe.reset()
    table = MetaData(cells)
    column_reads = {
        key: count
        for key, count in _gets(probe).items()
        if key.endswith("/zarr.json") and key != "cellData/zarr.json"
    }
    assert set(column_reads) == {
        "cellData/I/zarr.json",
        "cellData/ids/zarr.json",
        "cellData/score/zarr.json",
    }
    assert set(column_reads.values()) == {1}

    probe.reset()
    np.testing.assert_array_equal(table.fetch("score", key="I"), [0.0, 2.0])
    gets = _gets(probe)
    assert gets["cellData/score/zarr.json"] == 1
    assert gets["cellData/I/zarr.json"] == 1

    with pytest.raises(KeyError, match="does not exist"):
        table.fetch_all("missing")


def test_open_scans_assays_once(datastore_zarr_root) -> None:
    probe = StoreProbe()
    store = wrap_recording_store(LocalStore(datastore_zarr_root), probe=probe)
    datastore = DataStore(store, default_assay="RNA", zarr_mode="r")

    gets = _gets(probe)
    # Rescanning the hierarchy used to read each assay group dozens of times.
    assert 0 < gets["RNA/zarr.json"] <= 8
    assert 0 < gets["assay2/zarr.json"] <= 8

    probe.reset()
    assert datastore.assay_names == ["RNA", "assay2"]
    assert datastore.get_assay("assay2") is datastore.assay2
    assert probe.ops == []


def test_reuse_lookup_reads_each_candidate_once() -> None:
    root, probe = _recorded_root()
    provenance = make_provenance(
        operation="run_normalization",
        parameters={"log_transform": True},
        inputs={},
    )
    refs = []
    for _ in range(3):
        ref = ArtifactRef(
            scope="assay",
            assay="RNA",
            kind="normalized",
            artifact_id=new_artifact_id(),
        )
        root.create_group(artifact_path(ref)).attrs.update(
            {
                "artifact_id": ref.artifact_id,
                "kind": ref.kind,
                "provenance": provenance,
                "execution_options": {},
                "complete": True,
            }
        )
        refs.append(ref)

    probe.reset()
    found = find_reusable_artifacts(
        root,
        scope="assay",
        assay="RNA",
        kind="normalized",
        provenance=provenance,
    )

    assert sorted(found, key=lambda ref: ref.artifact_id) == sorted(
        refs, key=lambda ref: ref.artifact_id
    )
    gets = _gets(probe)
    for ref in refs:
        assert gets[f"{artifact_path(ref)}/zarr.json"] == 1


def test_validation_scope_reuses_results_only_inside_one_call(monkeypatch) -> None:
    from scarf.storage import selections
    from scarf.storage.artifacts import artifact_group
    from scarf.storage.errors import ArtifactResolutionError
    from scarf.storage.validation_scope import validation_scope

    root = zarr.open_group(store=MemoryStore(), mode="w")
    cells = root.create_group("cellData")
    cells.create_array("ids", data=np.array(["a", "b", "c"]))
    ref, _ = selections.resolve_generated_selection_artifact(
        root,
        scope="datastore",
        kind="cell_selection",
        values=np.array([True, False, True]),
        row_ids=cells["ids"],
        operation="manual_selection",
        parameters={},
        inputs={},
        source_column="I",
    )
    calls: list[ArtifactRef] = []
    original = selections._validate_stored_selection_integrity

    def counted(root_group, selection, **kwargs):
        calls.append(selection)
        return original(root_group, selection, **kwargs)

    monkeypatch.setattr(selections, "_validate_stored_selection_integrity", counted)
    arguments = dict(
        kind="cell_selection", scope="datastore", assay=None, table_path="cellData"
    )

    with validation_scope():
        first = selections.validate_stored_selection_integrity(root, ref, **arguments)
        again = selections.validate_stored_selection_integrity(root, ref, **arguments)
    assert again is first
    assert len(calls) == 1

    # Outside a scope every call validates, so changes between calls are seen.
    selections.validate_stored_selection_integrity(root, ref, **arguments)
    selections.validate_stored_selection_integrity(root, ref, **arguments)
    assert len(calls) == 3

    # Failures are never cached.
    artifact_group(root, ref)["values"][0] = False
    with validation_scope():
        for _ in range(2):
            with pytest.raises(ArtifactResolutionError):
                selections.validate_stored_selection_integrity(root, ref, **arguments)
    assert len(calls) == 5


def test_public_datastore_methods_open_one_scope_and_keep_signatures() -> None:
    import inspect

    for name in ("run_umap", "auto_filter_cells", "run_marker_search", "run_pca"):
        method = getattr(DataStore, name)
        assert getattr(method, "__validation_scoped__", False), name
        assert inspect.signature(method) == inspect.signature(method.__wrapped__)

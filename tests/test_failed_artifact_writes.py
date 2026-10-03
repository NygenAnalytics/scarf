"""A failed or interrupted artifact write leaves no incomplete slot behind."""

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.storage.artifact_writer import discard_artifact, plan_artifact
from scarf.storage.artifacts import inspect_artifact, list_artifacts
from scarf.storage.selections import resolve_generated_selection_artifact


def _all_refs(store, kind: str) -> set:
    return set(store.list_artifacts(kind=kind, complete_only=False))


def _normalized(store):
    refs = store.list_artifacts(kind="normalized", complete_only=True)
    assert refs
    return refs[0]


def test_failed_pca_write_deletes_its_slot(analyzed_datastore_ephemeral, monkeypatch):
    store = analyzed_datastore_ephemeral
    normalized = _normalized(store)
    before = _all_refs(store, "reduction")

    destinations: list[str] = []

    def fail(destination, *args, **kwargs):
        # The started slot exists while its payload is written.
        destinations.append(destination.path)
        assert destination.path in store.zw
        raise RuntimeError("injected score write failure")

    monkeypatch.setattr(
        "scarf.datastore._operations.graph.write_dense_from_row_batches", fail
    )
    with pytest.raises(RuntimeError, match="injected score write failure"):
        store.run_pca(normalized, dims=3, invalidate_cache=True)

    assert len(destinations) == 1
    slot = destinations[0].rsplit("/", 1)[0]
    assert "/artifacts/reduction/" in f"/{slot}"
    assert slot not in store.zw
    assert _all_refs(store, "reduction") == before


def test_interrupted_ann_index_write_deletes_its_slot(
    analyzed_datastore_ephemeral, monkeypatch
):
    store = analyzed_datastore_ephemeral
    reductions = store.list_artifacts(kind="reduction", complete_only=True)
    assert reductions
    before = _all_refs(store, "ann_index")

    slots: list[str] = []

    def interrupt(group, *args, **kwargs):
        slots.append(group.path)
        raise KeyboardInterrupt

    monkeypatch.setattr("scarf.datastore._operations.graph.save_ann_index", interrupt)
    with pytest.raises(KeyboardInterrupt):
        store.build_ann_index(reductions[0], invalidate_cache=True)

    # The slot was started before the interruption and is gone after it.
    assert len(slots) == 1
    assert "/artifacts/ann_index/" in f"/{slots[0]}"
    assert slots[0] not in store.zw
    assert _all_refs(store, "ann_index") == before


def test_failed_selection_write_deletes_its_slot(monkeypatch):
    import scarf.storage.selections as selections

    root = zarr.open_group(store=MemoryStore(), mode="w")
    cell_data = root.create_group("cellData")
    row_ids = np.asarray([f"cell-{index}" for index in range(4)])
    cell_data.create_array("ids", data=row_ids)
    monkeypatch.setattr(selections, "_stored_selection_fingerprint", lambda _: "x")

    with pytest.raises(RuntimeError, match="payload changed while it was stored"):
        resolve_generated_selection_artifact(
            root,
            scope="datastore",
            kind="cell_selection",
            values=np.asarray([True, False, True, True]),
            row_ids=row_ids,
            operation="manual_selection",
            parameters={},
            inputs={},
            source_column="manual",
        )

    assert list_artifacts(root, scope="datastore", kind="cell_selection") == []


def test_discard_artifact_logs_instead_of_masking_the_write_error() -> None:
    from scarf.storage.artifacts import artifact_path
    from scarf.utils import logger

    root = zarr.open_group(store=MemoryStore(), mode="w")
    planned = plan_artifact(
        root,
        scope="assay",
        assay="RNA",
        kind="reduction",
        operation="manual_reduction",
        parameters={},
        inputs={},
        execution_options={},
    )

    class Undeletable:
        def __delitem__(self, path: str) -> None:
            raise OSError("store is unavailable")

    warnings: list[str] = []
    sink = logger.add(
        lambda message: warnings.append(message.record["message"]), level="WARNING"
    )
    try:
        discard_artifact(Undeletable(), planned)  # type: ignore[arg-type]
        # A slot that was never started is already gone, so nothing is logged.
        discard_artifact(root, planned)
    finally:
        logger.remove(sink)
    assert warnings == [
        "Could not remove the incomplete artifact at "
        f"{artifact_path(planned.ref)}: store is unavailable"
    ]
    assert not inspect_artifact(root, planned.ref).exists

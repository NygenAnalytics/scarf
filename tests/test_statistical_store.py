"""The writer and the reuse check of stored statistical-test results agree.

``statistical_store`` derives a slot's attributes in one function: the writer
records them and the reuse check compares a stored slot with them. Editing any
attribute that the planned request determines therefore stops reuse, and the
reader requires exactly the attributes that the writer records.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest
import zarr

from scarf.datastore._operations.statistical_store import _STATISTICAL_SLOT_ATTRIBUTES
from scarf.datastore.datastore import DataStore
from scarf.metadata.selection import CellField
from scarf.storage.artifacts import artifact_path
from tests.storage_helpers import write_count_store

_N_CELLS = 24


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> DataStore:
    zarr_loc = str(tmp_path_factory.mktemp("statistical_store") / "store.zarr")
    counts = np.random.default_rng(5).poisson(3.0, size=(_N_CELLS, 6))
    write_count_store(zarr_loc, {"RNA": counts}, "uint16")
    store = DataStore(
        zarr_loc, default_assay="RNA", min_features_per_cell=0, nthreads=1
    )
    cells = np.arange(_N_CELLS)
    store.cells.insert("everyone", np.ones(_N_CELLS, dtype=bool), overwrite=True)
    store.cells.insert(
        "grp", np.where(cells % 2 == 0, "a", "b").astype(object), overwrite=True
    )
    store.cells.insert(
        "sample",
        np.array([f"s{cell % 6}" for cell in cells], dtype=object),
        overwrite=True,
    )
    return store


def _run(store: DataStore, key: str) -> Any:
    return store.run_statistical_testing(
        key,
        CellField("grp"),
        cell_selection=store.snapshot_cell_selection("everyone"),
        sample_by="sample",
    )


def _slot(store: DataStore, result: Any) -> zarr.Group:
    """Open a saved result for editing, as a tool outside Scarf would."""
    return zarr.open_group(
        store.zw.store, mode="r+", path=artifact_path(result.artifact)
    )


# Every attribute goes through one comparison; earlier releases left these out.
@pytest.mark.parametrize("attribute", ["pair_by", "sample_by", "summary_scope"])
def test_editing_an_attribute_the_request_determines_stops_reuse(
    store: DataStore, attribute: str
) -> None:
    key = f"value_{attribute}"
    store.cells.insert(
        key, np.random.default_rng(len(key)).normal(size=_N_CELLS), overwrite=True
    )
    first = _run(store, key)
    assert _run(store, key).artifact == first.artifact
    slot = _slot(store, first)
    slot.attrs[attribute] = f"{slot.attrs[attribute]}-edited"

    second = _run(store, key)

    assert second.artifact != first.artifact
    pd.testing.assert_frame_equal(second.tables[key], first.tables[key])


def test_the_reader_requires_exactly_the_attributes_the_writer_records(
    store: DataStore,
) -> None:
    store.cells.insert(
        "value_recorded",
        np.random.default_rng(3).normal(size=_N_CELLS),
        overwrite=True,
    )
    result = _run(store, "value_recorded")
    recorded = set(dict(_slot(store, result).attrs)) - {
        "artifact_id",
        "complete",
        "created_at_ns",
        "execution_options",
        "kind",
        "provenance",
        "scarf_version",
    }

    assert recorded == _STATISTICAL_SLOT_ATTRIBUTES

"""The destination rule that writers apply before they create a store."""

import os
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
import zarr
from scipy.sparse import csr_matrix
from zarr.storage import LocalStore, LoggingStore, MemoryStore

from scarf import DataStore
from scarf.storage import destinations
from scarf.storage.destinations import check_destination, create_destination
from scarf.storage.stores import MATRIX_SOURCE_ATTR
from scarf.writers import SparseToZarr

_COUNTS = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=np.uint8)
_ONLY_UNPREPARED = (
    "overwrite=True replaces only a Scarf store that no DataStore has opened"
)


def _import(location, *, workspace: str | None = None, overwrite: bool = False):
    SparseToZarr(
        csr_matrix(_COUNTS),
        location,
        ["c1", "c2", "c3"],
        ["f1", "f2", "f3"],
        workspace=workspace,
        nthreads=1,
        overwrite=overwrite,
    ).dump()


def _keys(store: MemoryStore) -> dict[str, bytes]:
    return {key: bytes(value.to_bytes()) for key, value in store._store_dict.items()}


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_a_writer_creates_its_store_at_an_empty_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    for location in (str(tmp_path / "absent.zarr"), str(empty), MemoryStore()):
        check_destination(location)
        root = create_destination(location)
        assert list(root.keys()) == [] and dict(root.attrs) == {}
    assert (empty / "zarr.json").is_file()

    # A key that another writer adds after the check is never deleted.
    raced = MemoryStore()
    check = destinations._check

    def racing_check(location, **options) -> bool:
        replace = check(location, **options)
        zarr.open_group(store=raced, mode="w").create_group("other")
        return replace

    monkeypatch.setattr(destinations, "_check", racing_check)
    with pytest.raises(FileExistsError, match="already contains data"):
        create_destination(raced)
    assert "other" in zarr.open_group(store=raced, mode="r")


def test_a_destination_that_holds_any_key_needs_overwrite(tmp_path: Path) -> None:
    bare = MemoryStore()
    zarr.open_group(store=bare, mode="w")
    other = tmp_path / "other"
    other.mkdir()
    (other / ".DS_Store").write_bytes(b"\0")
    for location in (bare, str(other)):
        with pytest.raises(FileExistsError, match="is not empty. Pass overwrite=True"):
            create_destination(location)
    assert set(_keys(bare)) == {"zarr.json"}
    assert _files(other) == {".DS_Store": b"\0"}
    file = tmp_path / "file.zarr"
    file.write_bytes(b"\0")
    for overwrite in (False, True):
        with pytest.raises(FileExistsError, match="is a file, not a store"):
            create_destination(str(file), overwrite=overwrite)
    assert file.read_bytes() == b"\0"


def test_overwrite_replaces_an_unprepared_store(tmp_path: Path) -> None:
    # The importer tests replace stores without a workspace.
    location = tmp_path / "import.zarr"
    _import(str(location), workspace="ws")
    root = create_destination(str(location), overwrite=True)
    assert list(root.keys()) == []
    assert _files(location).keys() == {"zarr.json"}


@pytest.mark.parametrize("case", ["root", "workspace", "mounted"])
def test_overwrite_never_replaces_a_prepared_store(case) -> None:
    store = MemoryStore()
    if case == "mounted":
        zarr.open_group(store=store, mode="w", attributes={MATRIX_SOURCE_ATTR: {}})
    else:
        workspace = "ws" if case == "workspace" else None
        _import(store, workspace=workspace)
        # A writable open prepares every assay of the store or workspace.
        DataStore(store, workspace=workspace, min_features_per_cell=0, nthreads=1)
    before = _keys(store)
    reason = {
        "root": "holds the prepared assays ['RNA']",
        "workspace": "holds the prepared assays ['ws/RNA']",
        "mounted": f"records a {MATRIX_SOURCE_ATTR}",
    }[case]
    with pytest.raises(FileExistsError, match=re.escape(reason)) as raised:
        create_destination(store, overwrite=True)
    assert _ONLY_UNPREPARED in str(raised.value)
    assert _keys(store) == before


def test_overwrite_never_replaces_content_that_scarf_did_not_write(
    tmp_path: Path,
) -> None:
    template = tmp_path / "template.zarr"
    _import(str(template))
    # A file beside the groups that Scarf writes, inside one of them, or inside
    # one of their arrays is not Scarf's.
    stores = {}
    for index, member in enumerate(
        (
            "notes.txt",
            "RNA/notes.txt",
            "RNA/featureData/notes.txt",
            "cellData/ids/notes.txt",
        )
    ):
        stores[member] = tmp_path / f"store{index}.zarr"
        shutil.copytree(template, stores[member])
        (stores[member] / member).write_text("keep")
    other = tmp_path / "other"
    other.mkdir()
    (other / "notes.txt").write_text("keep")
    workspace = MemoryStore()
    _import(workspace, workspace="ws")
    zarr.open_group(store=workspace, mode="r+")["ws"].create_array(
        "extra", data=np.arange(2)
    )
    before = _keys(workspace)
    in_workspace = tmp_path / "workspace.zarr"
    _import(str(in_workspace), workspace="ws")
    (in_workspace / "ws/RNA/notes.txt").write_text("keep")
    array = MemoryStore()
    zarr.create_array(store=array, data=np.arange(2))
    cases = [
        (str(location), f"holds {member!r}, which is not part of a Scarf store")
        for member, location in stores.items()
    ]
    cases += [
        (str(other), "holds content without a Zarr root group"),
        (array, "holds content without a Zarr root group"),
        (workspace, "holds 'ws/extra', which is not part of a Scarf store"),
        (
            str(in_workspace),
            "holds 'ws/RNA/notes.txt', which is not part of a Scarf store",
        ),
    ]
    for location, reason in cases:
        with pytest.raises(FileExistsError, match=re.escape(reason)):
            create_destination(location, overwrite=True)
    for member, location in stores.items():
        assert (location / member).read_text() == "keep"
    assert (in_workspace / "ws/RNA/notes.txt").read_text() == "keep"
    assert (other / "notes.txt").read_text() == "keep"
    assert _keys(workspace) == before
    np.testing.assert_array_equal(zarr.open_array(store=array, mode="r")[:], [0, 1])


def test_a_local_destination_inside_a_store_is_refused(tmp_path: Path) -> None:
    store = tmp_path / "s.zarr"
    _import(str(store))
    before = _files(store)
    inside = f"lies inside the Zarr store at {re.escape(str(store.resolve()))}"
    for member in ("RNA", "RNA/featureData/new.zarr", "new.zarr"):
        nested = store / member
        for location in (
            str(nested),
            f"file://{nested}",
            LocalStore(str(nested)),
            LoggingStore(LocalStore(str(nested))),
        ):
            for overwrite in (False, True):
                with pytest.raises(ValueError, match=inside):
                    create_destination(location, overwrite=overwrite)
    assert _files(store) == before
    # A Zarr v2 store is marked by its .zgroup document.
    legacy = tmp_path / "legacy.zarr"
    zarr.open_group(str(legacy), mode="w", zarr_format=2)
    with pytest.raises(ValueError, match="lies inside the Zarr store at"):
        create_destination(str(legacy / "new.zarr"))
    assert not (legacy / "new.zarr").exists()


def test_a_destination_whose_root_is_an_assay_group_is_refused() -> None:
    # A store without a path, such as an object-store prefix, can still be
    # the assay group of another store.
    store = MemoryStore()
    zarr.open_group(store=store, mode="w", attributes={"is_assay": True})
    before = _keys(store)
    for overwrite in (False, True):
        with pytest.raises(ValueError, match="is an assay group of a Zarr store"):
            create_destination(store, overwrite=overwrite)
    assert _keys(store) == before


@pytest.mark.skipif(os.name == "nt", reason="Windows file names cannot hold '?'")
def test_a_local_path_is_opened_as_it_is_written(tmp_path: Path) -> None:
    name = "run#1;v?.zarr"
    location = str(tmp_path / name)
    # Zarr reads a string as a URL; before, it opened <tmp>/run for
    # <tmp>/run#1.zarr while the checks read the path as written.
    _import(location)
    assert sorted(path.name for path in tmp_path.iterdir()) == [name]
    with pytest.raises(FileExistsError, match="is not empty"):
        _import(location)
    _import(location, overwrite=True)
    assert sorted(path.name for path in tmp_path.iterdir()) == [name]
    assert DataStore(location, min_features_per_cell=0, nthreads=1).cells.N == 3


@pytest.mark.skipif(os.name == "nt", reason="Windows file names cannot hold '#'")
def test_a_local_cache_path_is_opened_as_it_is_written(tmp_path: Path) -> None:
    from scarf.storage.copy import create_or_open_staged_normed_array

    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "keep.txt").write_bytes(b"\0")
    cache = tmp_path / "run#1" / "normed.zarr"
    cache.parent.mkdir()
    # Before, Zarr read the string as a URL and opened, with mode "w", and so
    # emptied, <tmp>/run.
    staged = create_or_open_staged_normed_array(str(cache), (2, 3))
    staged.attrs["complete"] = True
    assert (tmp_path / "run" / "keep.txt").is_file()
    reopened = create_or_open_staged_normed_array(str(cache), (2, 3))
    assert reopened.attrs["complete"] is True

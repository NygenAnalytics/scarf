"""The type that a DataStore resolves for an assay is the type its store declares.

Merge, subset, and HTO demultiplexing read the declared type of an open assay.
It is the type that the open resolved, which a writable open records and a
read-only open must match, so no reader sees a type that differs from the
assay's class or from the store's record.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import zarr

from scarf import DataStore
from scarf.assay import ADTassay, Assay
from scarf.assay.classification import declared_assay_type
from scarf.merge import DataStoreMerge
from scarf.storage.schema import create_cell_data, create_zarr_count_assay
from scarf.storage.stores import load_zarr
from scarf.writers import SubsetZarr
from scarf.writers.counts_t import finalize_writer_counts_t
from tests.storage_helpers import finalize_test_counts

_RNA = np.random.default_rng(3).poisson(4.0, size=(12, 5)).astype(np.uint32) + 1
_TAGS = np.random.default_rng(4).poisson(6.0, size=(12, 3)).astype(np.uint32) + 1


def _write_store(
    path: Path,
    counts: Mapping[str, np.ndarray],
    *,
    workspace: str | None = None,
) -> str:
    """Write a fresh import whose assays share one set of cells."""
    root = load_zarr(zarr_loc=str(path), mode="w")
    (n_cells,) = {len(values) for values in counts.values()}
    cell_ids = np.asarray([f"cell{index}" for index in range(n_cells)])
    create_cell_data(root, workspace, ids=cell_ids, names=cell_ids)
    for assay, values in counts.items():
        feature_ids = np.asarray(
            [f"{assay}{index}" for index in range(values.shape[1])]
        )
        array = create_zarr_count_assay(
            root,
            assay,
            workspace,
            n_cells,
            feat_ids=feature_ids,
            feat_names=feature_ids,
            dtype="uint32",
        )
        array[:] = values
        finalize_test_counts(array)
        finalize_writer_counts_t(root, assay, workspace)
    return str(path)


def _open(location: str, **options: Any) -> DataStore:
    return DataStore(
        location,
        default_assay="RNA",
        min_features_per_cell=-1,
        nthreads=1,
        **options,
    )


def _attribute_root(location: str, workspace: str | None) -> zarr.Group:
    root = zarr.open_group(location, mode="r")
    return root if workspace is None else root[workspace]


def _snapshot(location: str) -> dict[str, bytes]:
    root = Path(location)
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.mark.slow
@pytest.mark.parametrize("workspace", [None, "ws"], ids=["root", "workspace"])
def test_an_explicit_type_is_the_declared_type_of_the_whole_session(
    tmp_path, workspace, monkeypatch
) -> None:
    # Before, a workspace store opened with assay_types={"hashtags": "HTO"}
    # recorded HTO and opened the assay as ADTassay, while merge and subset
    # read the workspace attributes that the assay captured before the open
    # recorded the type, and declared it as Assay.
    counts = {"RNA": _RNA, "hashtags": _TAGS}
    first = _open(
        _write_store(tmp_path / "first.zarr", counts, workspace=workspace),
        workspace=workspace,
        assay_types={"hashtags": "HTO"},
    )
    second = _open(
        _write_store(tmp_path / "second.zarr", counts, workspace=workspace),
        workspace=workspace,
        assay_types={"hashtags": "HTO"},
    )
    hashtags = first.get_assay("hashtags")
    assert type(hashtags) is ADTassay
    assert hashtags.assayType == "HTO"
    assert declared_assay_type(hashtags) == "HTO"
    assert _attribute_root(str(first.zarr_loc), workspace).attrs["assayTypes"] == {
        "RNA": "RNA",
        "hashtags": "HTO",
    }

    merger = DataStoreMerge(
        [first, second], str(tmp_path / "merged.zarr"), ["a", "b"], nthreads=1
    )
    plan = merger.plan()
    assert {assay.assayName: assay.assayType for assay in plan.assays} == {
        "RNA": "RNA",
        "hashtags": "HTO",
    }
    assert plan.manifest["assayTypes"] == {"RNA": "RNA", "hashtags": "HTO"}
    merger.dump()
    merged = zarr.open_group(str(tmp_path / "merged.zarr"), mode="r")
    assert merged.attrs["assayTypes"] == {"RNA": "RNA", "hashtags": "HTO"}

    subset_path = str(tmp_path / "subset.zarr")
    SubsetZarr(
        subset_path,
        assays=[first.RNA, hashtags],
        out_workspace=workspace,
        cell_idx=np.arange(8),
        nthreads=1,
    ).dump()
    assert _attribute_root(subset_path, workspace).attrs["assayTypes"] == {
        "RNA": "RNA",
        "hashtags": "HTO",
    }

    # Demultiplexing accepts the assay in the session that declared it.
    monkeypatch.setattr(
        "scarf.datastore._operations.quality_control.hto_demux",
        lambda frame, **_kwargs: frame.idxmax(axis=1),
    )
    cells = first.snapshot_cell_selection("I")
    identity = first.run_hto_demultiplexing(cells, from_assay="hashtags")
    assert identity.kind == "hto_identity"


def test_the_declared_type_is_the_one_the_open_resolved(tmp_path) -> None:
    location = _write_store(tmp_path / "data.zarr", {"RNA": _RNA, "hashtags": _TAGS})
    datastore = _open(location)
    assert type(datastore.get_assay("hashtags")) is Assay

    # A record changed behind the open does not change the open assays, whose
    # class and normalization follow the type the open resolved.
    zarr.open_group(location, mode="r+").attrs["assayTypes"] = {
        "RNA": "RNA",
        "hashtags": "HTO",
    }
    assert declared_assay_type(datastore.get_assay("hashtags")) == "Assay"
    cells = datastore.snapshot_cell_selection("I")
    with pytest.raises(TypeError, match="'hashtags' is declared as 'Assay'"):
        datastore.run_hto_demultiplexing(cells, from_assay="hashtags")
    reopened = _open(location)
    assert declared_assay_type(reopened.get_assay("hashtags")) == "HTO"


def test_a_read_only_open_rejects_a_type_that_differs_from_the_store(
    tmp_path,
) -> None:
    location = _write_store(tmp_path / "data.zarr", {"RNA": _RNA, "hashtags": _TAGS})
    _open(location)
    assert zarr.open_group(location, mode="r").attrs["assayTypes"] == {
        "RNA": "RNA",
        "hashtags": "Assay",
    }
    before = _snapshot(location)

    # Before, the assay opened as ADTassay while merge and subset read the
    # recorded Assay.
    message = (
        r"assay_types declares assay 'hashtags' as 'HTO', but the store declares "
        r"it as 'Assay'\. A read-only open cannot record a type.*"
        r"zarr_mode='r\+' and assay_types=\{'hashtags': 'HTO'\}"
    )
    with pytest.raises(ValueError, match=message):
        _open(location, zarr_mode="r", assay_types={"hashtags": "HTO"})
    # An assay without a record entry declares the preset of its name.
    with pytest.raises(ValueError, match="declares assay 'RNA' as 'URNA'"):
        _open(location, zarr_mode="r", assay_types={"RNA": "URNA"})
    assert _snapshot(location) == before

    matching = _open(
        location, zarr_mode="r", assay_types={"RNA": "RNA", "hashtags": "Assay"}
    )
    assert declared_assay_type(matching.get_assay("hashtags")) == "Assay"
    assert declared_assay_type(matching.RNA) == "RNA"
    assert _snapshot(location) == before

    # A writable open records the new type, and read-only opens then match it.
    _open(location, assay_types={"hashtags": "HTO"})
    reopened = _open(location, zarr_mode="r", assay_types={"hashtags": "HTO"})
    assert type(reopened.get_assay("hashtags")) is ADTassay
    assert declared_assay_type(reopened.get_assay("hashtags")) == "HTO"

    # Without a record, an assay declares the preset of its name or Assay.
    del zarr.open_group(location, mode="r+").attrs["assayTypes"]
    matching = _open(
        location, zarr_mode="r", assay_types={"RNA": "RNA", "hashtags": "Assay"}
    )
    assert declared_assay_type(matching.get_assay("hashtags")) == "Assay"
    with pytest.raises(ValueError, match="declares it as 'Assay'"):
        _open(location, zarr_mode="r", assay_types={"hashtags": "ADT"})


def test_a_record_that_is_not_a_mapping_raises_its_remedy(tmp_path) -> None:
    location = _write_store(tmp_path / "data.zarr", {"RNA": _RNA, "hashtags": _TAGS})
    _open(location)
    zarr.open_group(location, mode="r+").attrs["assayTypes"] = ["RNA", "HTO"]
    before = _snapshot(location)

    message = (
        r"The assayTypes attribute of the store is .*, which is not a mapping.*"
        r"zarr_mode='r\+'.*assay_types=\{'RNA': <preset>, 'hashtags': <preset>\}"
    )
    # Before, the record was treated as empty and a writable open replaced it.
    with pytest.raises(ValueError, match=message):
        _open(location)
    with pytest.raises(ValueError, match=message):
        _open(location, zarr_mode="r")
    with pytest.raises(ValueError, match=message):
        _open(location, zarr_mode="r", assay_types={"RNA": "RNA", "hashtags": "HTO"})
    # Declaring only some assays leaves the others without a type.
    with pytest.raises(ValueError, match=message):
        _open(location, assay_types={"hashtags": "HTO"})
    assert _snapshot(location) == before

    # A writable open that declares every assay replaces the record.
    datastore = _open(location, assay_types={"RNA": "RNA", "hashtags": "HTO"})
    assert declared_assay_type(datastore.get_assay("hashtags")) == "HTO"
    assert zarr.open_group(location, mode="r").attrs["assayTypes"] == {
        "RNA": "RNA",
        "hashtags": "HTO",
    }


def test_repack_rejects_a_record_that_is_not_a_mapping(tmp_path) -> None:
    from scarf.tools.repack_zarr import repack_store

    location = _write_store(tmp_path / "data.zarr", {"RNA": _RNA})
    _open(location)
    zarr.open_group(location, mode="r+").attrs["assayTypes"] = ["RNA"]
    output = tmp_path / "repacked.zarr"
    with pytest.raises(ValueError, match="which is not a mapping"):
        repack_store(location, str(output))
    assert not output.exists()


def test_a_derived_assay_declares_the_type_it_was_registered_with(tmp_path) -> None:
    peaks = ["chr1:100-200", "chr1:250-350", "chr2:100-200"]
    counts = np.array([[2, 0, 1], [0, 3, 0], [1, 0, 4], [3, 1, 1]], dtype=np.uint32)
    root = load_zarr(zarr_loc=str(tmp_path / "atac.zarr"), mode="w")
    cell_ids = np.asarray([f"cell{index}" for index in range(len(counts))])
    create_cell_data(root, None, ids=cell_ids, names=cell_ids)
    array = create_zarr_count_assay(
        root,
        "ATAC",
        None,
        len(counts),
        feat_ids=np.asarray(peaks),
        feat_names=np.asarray(peaks),
        dtype="uint32",
    )
    array[:] = counts
    finalize_test_counts(array)
    finalize_writer_counts_t(root, "ATAC", None)
    datastore = DataStore(
        str(tmp_path / "atac.zarr"),
        default_assay="ATAC",
        min_features_per_cell=-1,
        nthreads=1,
    )
    bed = tmp_path / "genes.bed"
    pd.DataFrame(
        [
            ("chr1", 120, 300, "gene_a", "GENE_A"),
            ("chr2", 120, 180, "gene_b", "GENE_B"),
        ]
    ).to_csv(bed, sep="\t", header=False, index=False)

    datastore.add_melded_assay(
        from_assay="ATAC",
        external_bed_fn=str(bed),
        assay_label="GENES",
        assay_type="GeneActivity",
        renormalization=False,
    )
    datastore.ATAC.feats.insert("module", np.array([1, 1, 2]), overwrite=True)
    datastore.add_grouped_assay("module", from_assay="ATAC", assay_label="MODULES")

    # Every assay of the session carries its type, including the ones that
    # were rebuilt when the derived assays were registered.
    assert {
        name: declared_assay_type(datastore.get_assay(name))
        for name in datastore.assay_names
    } == {"ATAC": "ATAC", "GENES": "GeneActivity", "MODULES": "Assay"}
    assert zarr.open_group(str(tmp_path / "atac.zarr"), mode="r").attrs[
        "assayTypes"
    ] == {"ATAC": "ATAC", "GENES": "GeneActivity", "MODULES": "Assay"}

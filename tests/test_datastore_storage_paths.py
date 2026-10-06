"""DataStore opening, storage locations, references, mounts, and presentation."""

import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr
from obstore.store import MemoryStore as ObjectMemoryStore
from zarr.abc.store import Store
from zarr.core.buffer import Buffer
from zarr.storage import MemoryStore, ObjectStore, WrapperStore

from scarf import DataStore, mount_datastore
from scarf.assay import Assay, RNAassay
from scarf.metadata.artifacts import (
    categorical_display,
    column_display,
    plan_cell_data_artifact,
    validate_display_metadata,
    write_cell_data_artifact,
)
from scarf.storage.artifacts import (
    ArtifactRef,
    ExternalArtifactRef,
    artifact_path,
    inspect_artifact,
    make_provenance,
    new_artifact_id,
)
from scarf.storage.pipeline_runs import create_pipeline_run_record
from scarf.storage.profiles import resolve_storage_profile
from scarf.storage.schema import create_cell_data, create_zarr_count_assay
from scarf.storage.selections import resolve_generated_selection_artifact
from scarf.storage.stores import (
    MATRIX_SOURCE_ATTR,
    create_matrix_source,
    load_zarr,
    locations_overlap,
    make_store,
    open_store,
)
from scarf.utils.logging import logger
from scarf.writers.counts_t import finalize_writer_counts_t
from tests.storage_helpers import finalize_test_counts

_COUNTS = np.random.default_rng(5).poisson(3.0, size=(16, 6)).astype(np.uint32)


def _write_store(
    path: Path,
    counts: Mapping[str, np.ndarray],
    *,
    names: Mapping[str, Sequence[str]] | None = None,
) -> str:
    """Write a fresh import whose assays share one set of cells."""
    root = load_zarr(zarr_loc=str(path), mode="w")
    (n_cells,) = {len(values) for values in counts.values()}
    cell_ids = np.asarray([f"cell{index}" for index in range(n_cells)])
    create_cell_data(root, None, ids=cell_ids, names=cell_ids)
    for assay, values in counts.items():
        feature_ids = np.asarray(
            [f"{assay}{index}" for index in range(values.shape[1])]
        )
        array = create_zarr_count_assay(
            root,
            assay,
            None,
            n_cells,
            feat_ids=feature_ids,
            feat_names=np.asarray((names or {}).get(assay, feature_ids)),
            dtype="uint32",
        )
        array[:] = values
        finalize_test_counts(array)
        finalize_writer_counts_t(root, assay, None)
    return str(path)


def _open(location: str, **options: Any) -> DataStore:
    return DataStore(
        location,
        default_assay=options.pop("default_assay", "RNA"),
        min_features_per_cell=1,
        **options,
    )


@contextmanager
def _captured_warnings() -> Iterator[list[str]]:
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]),
        level="WARNING",
    )
    try:
        yield messages
    finally:
        logger.remove(sink)


def _assay_ref() -> ArtifactRef:
    return ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="normalized",
        artifact_id="a" * 64,
    )


def _write_record(
    datastore: DataStore,
    kind: str,
    *,
    operation: str,
    inputs: Mapping[str, ArtifactRef] | None = None,
    attrs: Mapping[str, int] | None = None,
    arrays: Mapping[str, np.ndarray] | None = None,
) -> ArtifactRef:
    """Write a complete RNA artifact record, as a hand-edited store holds it."""
    ref = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind=kind,
        artifact_id=new_artifact_id(),
    )
    group = datastore.zw.create_group(artifact_path(ref))
    group.attrs.update(
        {
            "artifact_id": ref.artifact_id,
            "kind": kind,
            "provenance": make_provenance(
                operation=operation,
                parameters={},
                inputs=dict(inputs or {}),
            ),
            "execution_options": {},
            "complete": True,
            **(attrs or {}),
        }
    )
    for name, values in (arrays or {}).items():
        group.create_array(name, data=values)
    return ref


def test_zarr_profile_environment_selects_the_datastore_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    location = _write_store(tmp_path / "data.zarr", {"RNA": _COUNTS})

    monkeypatch.setenv("SCARF_ZARR_PROFILE", "cloud")
    assert _open(location).storageProfile == "cloud"
    assert _open(location, zarrProfile="fast_local").storageProfile == "fast_local"
    monkeypatch.setenv("SCARF_ZARR_PROFILE", "fast_local")
    assert resolve_storage_profile("s3://bucket/data.zarr") == "fast_local"
    assert resolve_storage_profile("s3://bucket/data.zarr", "cloud") == "cloud"
    # An empty value is unset, as for SCARF_WORKERS and SCARF_MEM_BUDGET.
    monkeypatch.setenv("SCARF_ZARR_PROFILE", "")
    assert resolve_storage_profile("s3://bucket/data.zarr") == "cloud"


@pytest.mark.parametrize("value", ["Cloud", "fast-local", "cloud "])
def test_zarr_profile_environment_rejects_unknown_profiles(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    value: str,
) -> None:
    monkeypatch.setenv("SCARF_ZARR_PROFILE", value)
    message = re.escape(f"Invalid SCARF_ZARR_PROFILE={value!r}")

    with pytest.raises(ValueError, match=message):
        resolve_storage_profile("s3://bucket/data.zarr")
    # A datastore refuses the value before it opens the store.
    with pytest.raises(ValueError, match=message):
        DataStore(str(tmp_path / "missing.zarr"))
    assert resolve_storage_profile("s3://bucket/data.zarr", "cloud") == "cloud"


def test_artifact_ref_rejects_an_unknown_scope() -> None:
    with pytest.raises(ValueError, match="Invalid artifact scope: 'workspace'"):
        ArtifactRef(
            scope="workspace",  # type: ignore[arg-type]
            assay="RNA",
            kind="normalized",
            artifact_id="a" * 64,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("kind", 3, "kind and artifact_id must be strings"),
        ("artifact_id", None, "kind and artifact_id must be strings"),
        ("assay", 7, "assay must be a string or null"),
    ],
)
def test_serialized_artifact_ref_rejects_non_string_fields(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        ArtifactRef.from_dict({**_assay_ref().to_dict(), field: value})


def test_external_artifact_ref_rejects_mistyped_values() -> None:
    ref = _assay_ref()
    with pytest.raises(TypeError, match="dataset_fingerprint must be a string"):
        ExternalArtifactRef(dataset_fingerprint=b"reference", ref=ref)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ref must be an ArtifactRef"):
        ExternalArtifactRef(dataset_fingerprint="reference", ref=ref.to_dict())  # type: ignore[arg-type]

    serialized = ExternalArtifactRef("reference", ref).to_dict()
    with pytest.raises(
        TypeError, match="External artifact reference must be a mapping"
    ):
        ExternalArtifactRef.from_dict([serialized])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="dataset_fingerprint must be a string"):
        ExternalArtifactRef.from_dict({**serialized, "dataset_fingerprint": 5})
    with pytest.raises(TypeError, match="External artifact ref must be a mapping"):
        ExternalArtifactRef.from_dict({**serialized, "ref": artifact_path(ref)})


def test_uri_locations_overlap_only_within_one_bucket_path() -> None:
    assert not locations_overlap("s3://bucket/data.zarr", "gs://bucket/data.zarr")
    assert not locations_overlap("s3://bucket/data.zarr", "s3://archive/data.zarr")
    # A version query names the same object path.
    assert locations_overlap(
        "s3://bucket/data.zarr?versionId=2", "s3://bucket/data.zarr"
    )


def test_native_obstore_stores_open_through_zarr_object_stores() -> None:
    native = ObjectMemoryStore()
    read_only = make_store(native, read_only=True)
    assert isinstance(read_only, ObjectStore)
    assert read_only.read_only

    open_store(native, mode="w").create_group("cellData")
    assert "cellData" in open_store(native, mode="r")
    with pytest.raises(TypeError, match="path string or zarr Store"):
        make_store(42)  # type: ignore[arg-type]


def test_matrix_source_must_be_a_store_location_that_holds_assays(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.zarr"
    for source in ("", MemoryStore()):
        with pytest.raises(TypeError, match="non-empty string"):
            create_matrix_source(source, str(target), required_transposes=frozenset())  # type: ignore[arg-type]
    empty = tmp_path / "empty.zarr"
    zarr.open_group(str(empty), mode="w")
    with pytest.raises(ValueError, match="No assays found in the matrix source"):
        create_matrix_source(str(empty), str(target), required_transposes=frozenset())
    assert not target.exists()


class _FailingTarget(WrapperStore[Store]):
    """A backend that rejects table writes and cannot delete anything."""

    async def set(self, key: str, value: Buffer) -> None:
        if key.startswith("cellData/"):
            raise OSError(f"write rejected: {key}")
        await self._store.set(key, value)

    async def delete_dir(self, prefix: str) -> None:
        raise OSError("delete rejected")


def test_failed_mount_warns_when_its_target_cannot_be_removed(tmp_path: Path) -> None:
    source = _write_store(tmp_path / "source.zarr", {"RNA": _COUNTS})
    _open(source)

    with _captured_warnings() as messages, pytest.raises(OSError, match="cellData"):
        mount_datastore(source, at=_FailingTarget(MemoryStore()), default_assay="RNA")

    (warning,) = [text for text in messages if "incomplete mount target" in text]
    assert warning.startswith("Could not remove the incomplete mount target")
    assert "delete rejected" in warning
    assert warning.endswith("Delete it before mounting again.")


def test_mount_manifest_must_follow_the_matrix_source_contract(tmp_path: Path) -> None:
    source = _write_store(tmp_path / "source.zarr", {"RNA": _COUNTS})
    _open(source)
    target = str(tmp_path / "target.zarr")
    other_mount = str(tmp_path / "other_mount.zarr")
    mounted = mount_datastore(source, at=target, default_assay="RNA")
    mount_datastore(source, at=other_mount, default_assay="RNA")
    described = repr(mounted.z.store)
    assert described.startswith("MountedArtifactStore(")
    assert "target.zarr" in described
    assert "source=" in described and "source.zarr" in described

    root = zarr.open_group(target, mode="r+")
    manifest = dict(root.attrs[MATRIX_SOURCE_ATTR])
    for changed, message in (
        (source, "matrixSource attribute must be a mapping"),
        ({**manifest, "location": ""}, "matrixSource.location must be a non-empty"),
        ({**manifest, "workspace": 3}, "matrixSource.workspace must be a string"),
        ({**manifest, "assays": {}}, "matrixSource.assays must be a non-empty mapping"),
        (
            {**manifest, "assays": {"RNA": "fingerprint"}},
            "matrixSource assay entry for 'RNA' must be a mapping",
        ),
        # The source location now holds a mount instead of its counts.
        ({**manifest, "location": other_mount}, "source must own its count matrices"),
    ):
        root.attrs[MATRIX_SOURCE_ATTR] = changed
        with pytest.raises(ValueError, match=message):
            _open(target)
    root.attrs[MATRIX_SOURCE_ATTR] = manifest
    assert _open(target).workspace is None


def test_open_rejects_legacy_assay_state(tmp_path: Path) -> None:
    location = _write_store(tmp_path / "data.zarr", {"RNA": _COUNTS})
    zarr.open_group(location, mode="r+")["RNA"].create_group("state")

    with pytest.raises(
        ValueError, match="Legacy assay state is unsupported: RNA/state"
    ):
        _open(location)


def test_open_requires_cell_metadata(tmp_path: Path) -> None:
    location = _write_store(tmp_path / "data.zarr", {"RNA": _COUNTS})
    del zarr.open_group(location, mode="r+")["cellData"]

    # The error names the store that lacks the table.
    message = f"cellData not found in zarr file at file://{location}"
    with pytest.raises(KeyError, match=re.escape(message)):
        _open(location)


def test_default_assay_is_required_when_ambiguous_and_must_exist(
    tmp_path: Path,
) -> None:
    location = _write_store(
        tmp_path / "data.zarr",
        {"RNA": _COUNTS, "ADT": _COUNTS[:, :3]},
    )

    with pytest.raises(ValueError, match="more than one assay"):
        DataStore(location)
    with pytest.raises(ValueError, match="default assay name: ATAC was not found"):
        _open(location, default_assay="ATAC")
    assert "defaultAssay" not in zarr.open_group(location, mode="r").attrs


def test_unrecognized_assay_types_are_rejected_before_any_write(
    tmp_path: Path,
) -> None:
    location = _write_store(
        tmp_path / "data.zarr",
        {"RNA": _COUNTS, "Spatial": _COUNTS[:, :4], "Protein": _COUNTS[:, :2]},
    )
    # A store written without recorded assay types, as other tools write them.
    del zarr.open_group(location, mode="r+").attrs["assayTypes"]

    def recorded_attributes() -> dict[str, Any]:
        return dict(zarr.open_group(location, mode="r").attrs)

    before = recorded_attributes()
    with pytest.raises(
        ValueError,
        match=r"assay_type 'Imaging' of assay 'Spatial' is not a preset"
        r".*'Assay' for a generic assay",
    ):
        _open(location, assay_types={"Spatial": "Imaging"})
    with pytest.raises(
        ValueError, match=r"assay_types names assays that are not in the store: 'Image'"
    ):
        _open(location, assay_types={"Image": "RNA"})
    assert recorded_attributes() == before

    # Only an assay without any declaration falls back to the generic class.
    with _captured_warnings() as messages:
        datastore = _open(location, assay_types={"Spatial": "Assay"})
    assert isinstance(datastore.RNA, RNAassay)
    assert type(datastore.Spatial) is Assay
    assert type(datastore.Protein) is Assay
    assert datastore.zw.attrs["assayTypes"] == {
        "RNA": "RNA",
        "Spatial": "Assay",
        "Protein": "Assay",
    }
    assert any("Protein was set as a generic Assay" in text for text in messages)
    assert not any("Spatial was set as a generic Assay" in text for text in messages)
    reopened = _open(location)
    assert type(reopened.Spatial) is Assay
    assert type(reopened.Protein) is Assay

    # An unrecognized recorded type names its remedy, and an explicit preset
    # replaces it.
    recorded = {"RNA": "RNA", "Spatial": "Imaging", "Protein": "Assay"}
    zarr.open_group(location, mode="r+").attrs["assayTypes"] = recorded
    with pytest.raises(
        ValueError,
        match=r"Assay 'Spatial' is recorded in assayTypes as 'Imaging'.*"
        r"zarr_mode='r\+' and assay_types=\{'Spatial': ",
    ):
        _open(location)
    assert zarr.open_group(location, mode="r").attrs["assayTypes"] == recorded
    datastore = _open(location, assay_types={"Spatial": "ADT"})
    assert datastore.zw.attrs["assayTypes"] == {**recorded, "Spatial": "ADT"}


def test_load_artifact_distinguishes_missing_and_incomplete_artifacts(
    tmp_path: Path,
) -> None:
    datastore = _open(_write_store(tmp_path / "data.zarr", {"RNA": _COUNTS}))
    missing = ArtifactRef(
        scope="datastore",
        kind="cell_selection",
        artifact_id=new_artifact_id(),
    )
    with pytest.raises(KeyError, match="Artifact does not exist"):
        datastore.load_artifact(missing)

    selection = datastore.snapshot_cell_selection()
    # An interrupted write leaves its artifact marked incomplete.
    datastore.zw[artifact_path(selection)].attrs["complete"] = False
    with pytest.raises(RuntimeError, match="Artifact is incomplete"):
        datastore.load_artifact(selection)


def test_lineage_accepts_only_mapping_references(tmp_path: Path) -> None:
    datastore = _open(_write_store(tmp_path / "data.zarr", {"RNA": _COUNTS}))
    selection = datastore.snapshot_cell_selection()

    with pytest.raises(TypeError, match="references must be a MappingReference"):
        datastore.lineage(selection, references="reference.zarr")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"references\[0\] must be a MappingReference"):
        datastore.lineage(selection, references=[datastore])  # type: ignore[list-item]


def test_feature_values_by_name_require_a_match_and_average_duplicates(
    tmp_path: Path,
) -> None:
    counts = _COUNTS.copy()
    counts[:, 2] = counts[:, 0]
    counts[:, 3] = counts[:, 1]
    location = _write_store(
        tmp_path / "data.zarr",
        {"RNA": counts},
        names={"RNA": ["A", "B", "DUP", "DUP", "C", "D"]},
    )
    datastore = _open(location)

    with pytest.raises(ValueError, match="MISSING not found in RNA assay"):
        datastore.get_cell_vals("RNA", "I", "MISSING")
    with _captured_warnings() as messages:
        duplicated = datastore.get_cell_vals("RNA", "I", "DUP")

    assert "Plotting mean of 2 features because DUP is not unique." in messages
    first = datastore.get_cell_vals("RNA", "I", "A")
    second = datastore.get_cell_vals("RNA", "I", "B")
    np.testing.assert_allclose(duplicated, (first + second) / 2, rtol=1e-6)


def test_assay_and_feature_lookups_reject_missing_inputs(tmp_path: Path) -> None:
    datastore = _open(_write_store(tmp_path / "data.zarr", {"RNA": _COUNTS}))

    with pytest.raises(ValueError, match="Provide the name of an assay"):
        datastore.get_assay("")
    with pytest.raises(TypeError, match="features must be an ArtifactRef"):
        datastore.resolve_features("RNA", "highly_variable")  # type: ignore[arg-type]


def test_run_aware_export_requires_a_run_opened_from_this_datastore(
    tmp_path: Path,
) -> None:
    location = _write_store(tmp_path / "data.zarr", {"RNA": _COUNTS})
    datastore = _open(location)
    record = create_pipeline_run_record(
        datastore.zw,
        recipe="basic_rna_analysis",
        requested_label=None,
        assay="RNA",
        config={},
        stage_order=("input_snapshot",),
        scarf_version="1.0.0",
    )
    foreign = _open(location, zarr_mode="r").pipeline.open(run_id=record.run_id)

    with pytest.raises(TypeError, match="run must be a PipelineRun"):
        datastore.to_anndata(run=record)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="run must be opened from this datastore"):
        datastore.to_anndata(run=foreign)


def test_graph_consumers_reject_unusable_graph_references(tmp_path: Path) -> None:
    datastore = _open(_write_store(tmp_path / "data.zarr", {"RNA": _COUNTS}))
    datastore.cells.insert("group", np.asarray(["a", "b"] * 8), overwrite=True)
    labels = datastore.snapshot_cluster_labels(
        "group",
        cell_selection=datastore.snapshot_cell_selection(),
    )
    missing_graph = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="connectivity_map",
        artifact_id=new_artifact_id(),
    )

    with pytest.raises(TypeError, match="graph must be an ArtifactRef"):
        datastore.calc_membership_strength(labels, "connectivity_map")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Graph artifact is unavailable or incomplete"):
        datastore.calc_membership_strength(labels, missing_graph)
    with pytest.raises(TypeError, match="graph must be an ArtifactRef"):
        datastore.plots.cluster_tree(
            graph="connectivity_map",  # type: ignore[arg-type]
            clusters=labels,
            show=False,
        )


@pytest.mark.parametrize(
    ("n_cells", "n_neighbors", "edges", "message"),
    [
        # One row more than the graph's cell selection holds.
        (17, 1, np.zeros((17, 2), np.uint64), "align with the selected cell count"),
        # No neighbours, with the empty edge table that agrees with that.
        (16, 0, np.zeros((0, 2), np.uint64), "at least one neighbour per cell"),
    ],
    ids=["extra_row", "no_neighbours"],
)
def test_membership_strength_rejects_hand_edited_graph_dimensions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    n_cells: int,
    n_neighbors: int,
    edges: np.ndarray,
    message: str,
) -> None:
    datastore = _open(_write_store(tmp_path / "data.zarr", {"RNA": _COUNTS}))
    datastore.cells.insert("group", np.asarray(["a", "b"] * 8), overwrite=True)
    selection = datastore.snapshot_cell_selection()
    labels = datastore.snapshot_cluster_labels("group", cell_selection=selection)
    graph = _write_record(
        datastore,
        "connectivity_map",
        operation="build_connectivity_map",
        attrs={"n_cells": n_cells, "n_neighbors": n_neighbors},
        arrays={"edges": edges},
    )
    # The record names no upstream stages, so its cells are resolved directly.
    monkeypatch.setattr(
        "scarf.datastore._operations.presentation.graph_cell_selection",
        lambda _root, _graph: selection,
    )

    with pytest.raises(ValueError, match=message):
        datastore.calc_membership_strength(labels, graph)


def test_cluster_tree_rejects_paris_cuts_with_inconsistent_records(
    tmp_path: Path,
) -> None:
    datastore = _open(_write_store(tmp_path / "data.zarr", {"RNA": _COUNTS}))
    graph, other_graph = (
        _write_record(datastore, "connectivity_map", operation="build_connectivity_map")
        for _ in range(2)
    )
    hierarchy, foreign_hierarchy = (
        _write_record(
            datastore,
            "cluster_hierarchy",
            operation="fit_paris_hierarchy",
            inputs={"connectivity_map": fitted_graph},
        )
        for fitted_graph in (graph, other_graph)
    )
    missing_hierarchy = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="cluster_hierarchy",
        artifact_id=new_artifact_id(),
    )
    inputs = {
        "cluster_hierarchy": hierarchy,
        "connectivity_map": graph,
        "cell_selection": datastore.snapshot_cell_selection(),
    }
    labels = np.zeros(16, dtype=np.int64)

    for cut_inputs, cut_labels, message in (
        (
            {**inputs, "cluster_hierarchy": missing_hierarchy},
            labels,
            "does not have a complete hierarchy for the requested graph",
        ),
        (
            {**inputs, "cluster_hierarchy": foreign_hierarchy},
            labels,
            "does not have a complete hierarchy for the requested graph",
        ),
        (
            {name: ref for name, ref in inputs.items() if name != "cell_selection"},
            labels,
            "Cluster cut has no cell-selection input",
        ),
        (inputs, labels[:-1], "do not align with the stored cell selection"),
    ):
        cut = _write_record(
            datastore,
            "cluster_cut",
            operation="cut_paris_hierarchy",
            inputs=cut_inputs,
            arrays={"labels": cut_labels},
        )
        with pytest.raises(ValueError, match=message):
            datastore._prepare_artifact_cluster_tree(
                graph_ref=graph,
                clusters_ref=cut,
                from_assay="RNA",
                fill_by_value=None,
                invalidate_cache=False,
            )


def test_smart_label_needs_one_cell_selection_and_handles_no_cells(
    tmp_path: Path,
) -> None:
    datastore = _open(_write_store(tmp_path / "data.zarr", {"RNA": _COUNTS}))
    datastore.cells.insert("group", np.asarray(["a", "b"] * 8), overwrite=True)
    datastore.cells.insert("base", np.asarray(["x"] * 8 + ["y"] * 8), overwrite=True)
    datastore.cells.insert(
        "first_half", np.asarray([True] * 8 + [False] * 8), overwrite=True
    )
    datastore.cells.insert("nobody", np.zeros(16, dtype=bool), overwrite=True)

    everyone = datastore.snapshot_cluster_labels(
        "group", cell_selection=datastore.snapshot_cell_selection()
    )
    half = datastore.snapshot_cluster_labels(
        "base", cell_selection=datastore.snapshot_cell_selection("first_half")
    )
    with pytest.raises(ValueError, match="must share one cell selection"):
        datastore.smart_label(everyone, half)

    empty = datastore.snapshot_cell_selection("nobody")
    relabelled = datastore.smart_label(
        datastore.snapshot_cluster_labels("group", cell_selection=empty),
        datastore.snapshot_cluster_labels("base", cell_selection=empty),
    )
    assert relabelled.kind == "smart_label"
    assert datastore.load_artifact(relabelled)["values"].shape == (0,)
    assert inspect_artifact(datastore.zw, relabelled).inputs["cell_selection"] == (
        empty.to_dict()
    )


@pytest.mark.parametrize(
    ("display", "error_type", "message"),
    [
        (
            {
                "kind": "continuous",
                "colormap": "viridis",
                "minimum": 0.0,
                "maximum": 1.0,
                "scale": "linear",
                "unit": "counts",
            },
            ValueError,
            "Continuous display metadata has unknown fields",
        ),
        (
            {
                "kind": "continuous",
                "colormap": "viridis",
                "minimum": 0.0,
                "maximum": 1.0,
                "scale": "sqrt",
            },
            ValueError,
            "Continuous display scale is invalid",
        ),
        *(
            (
                {
                    "kind": "continuous",
                    "colormap": "viridis",
                    "minimum": 0.0,
                    "maximum": maximum,
                    "scale": "log",
                },
                TypeError,
                "maximum must be numeric or null",
            )
            for maximum in (True, "1", float("inf"))
        ),
        (
            {"kind": "categorical", "missing_label": "NA"},
            ValueError,
            "Categorical display metadata is incomplete",
        ),
        (
            {"kind": "categorical", "categories": [], "order": []},
            ValueError,
            "Categorical display metadata has unknown fields",
        ),
        (
            {"kind": "categorical", "categories": [{"value": 1, "label": "1"}]},
            ValueError,
            "requires value, label, and color",
        ),
        *(
            (
                {
                    "kind": "categorical",
                    "categories": [{"value": value, "label": "v", "color": "#123456"}],
                },
                TypeError,
                "Display category value must be a JSON scalar",
            )
            for value in (None, [1], float("nan"))
        ),
        (
            {
                "kind": "categorical",
                "categories": [{"value": 1, "label": 1, "color": "#123456"}],
            },
            TypeError,
            "Display category label must be a string",
        ),
        (
            {"kind": "categorical", "categories": [], "missing_label": 0},
            TypeError,
            "missing_label must be a string",
        ),
        (
            {"kind": "categorical", "categories": [], "missing_color": "grey"},
            ValueError,
            "missing_color must be a hex color",
        ),
    ],
)
def test_display_metadata_rejects_malformed_fields(
    display: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        validate_display_metadata(display)


def test_column_display_returns_the_validated_stored_contract() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    cell_data = root.create_group("cellData")
    display = {
        "kind": "categorical",
        "categories": [{"value": "a", "label": "A", "color": "#112233"}],
        "missing_label": "NA",
        "missing_color": "#bdbdbd",
    }
    cell_data.create_array("group", data=np.asarray(["a", "a"])).attrs["display"] = (
        display
    )
    cell_data.create_array("score", data=np.asarray([1.0, 2.0])).attrs["display"] = (
        "viridis"
    )

    assert column_display(root, "group") == display
    with pytest.raises(TypeError, match="Existing display metadata must be a mapping"):
        column_display(root, "score")


def test_categorical_display_keeps_first_seen_order_for_unorderable_values() -> None:
    display = categorical_display(np.asarray(["b", 2, "a", 2, None], dtype=object))

    assert [category["value"] for category in display["categories"]] == ["b", 2, "a"]
    assert [category["label"] for category in display["categories"]] == ["b", "2", "a"]
    assert display["missing_label"] == "NA"
    assert validate_display_metadata(display) == display


def test_cell_data_artifact_rejects_scalar_payloads_and_leaves_no_artifact() -> None:
    root = zarr.open_group(store=MemoryStore(), mode="w")
    cell_ids = np.asarray(["c0", "c1"])
    root.create_group("cellData").create_array("ids", data=cell_ids)
    selection = resolve_generated_selection_artifact(
        root,
        scope="datastore",
        kind="cell_selection",
        values=np.ones(2, dtype=bool),
        row_ids=cell_ids,
        operation="test_selection",
        parameters={},
        inputs={},
        source_column="I",
    )[0]
    planned = plan_cell_data_artifact(
        root,
        scope="datastore",
        kind="metadata_snapshot",
        operation="test_scalar_payload",
        parameters={},
        inputs={},
        execution_options={},
        cell_selection=selection,
        arrays={"values": ((2,), "f")},
    )

    with pytest.raises(ValueError, match="at least one dimension"):
        write_cell_data_artifact(root, planned, {"values": np.float64(1.0)})
    assert not inspect_artifact(root, planned.ref).exists


@pytest.mark.parametrize("n_extra", [0, 2])
def test_repr_lists_metadata_columns_in_rows_of_five(tmp_path: Path, n_extra) -> None:
    datastore = _open(_write_store(tmp_path / "s.zarr", {"RNA": np.eye(4) + 1}))
    columns = list(datastore.cells.columns)
    for index in range(10 - len(columns) % 5 + n_extra):
        datastore.cells.insert(f"extra{index}", np.zeros(4), overwrite=True)
    columns = list(datastore.cells.columns)
    text = repr(datastore)
    listing = text.split("Cell metadata:")[1].split("\n   RNA assay")[0]
    rows = listing.lstrip("\n").split("\n")
    assert [row.strip() for row in rows] == [
        ", ".join(f"'{name}'" for name in columns[start : start + 5])
        + ("," if start + 5 < len(columns) else "")
        for start in range(0, len(columns), 5)
    ]

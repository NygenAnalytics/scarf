"""A mounted target resolves its matrix source's artifacts read only."""

import shutil
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import zarr
from obstore.store import MemoryStore as ObjectMemoryStore
from zarr.abc.store import RangeByteRequest, Store
from zarr.core import codec_pipeline as zarr_codec_pipeline
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import collect_aiterator, sync
from zarr.storage import LocalStore, ObjectStore

from scarf import cytebase
from scarf.datastore.datastore import DataStore, mount_datastore
from scarf.readers import H5adReader
from scarf.storage.artifact_writer import artifact_transaction, plan_artifact
from scarf.storage.artifacts import artifact_group, artifact_path
from scarf.storage.profiles import resolve_storage_profile
from scarf.storage.selections import read_stored_selection_indices
from scarf.storage.stores import (
    MATRIX_SOURCE_ATTR,
    REMOTE_METADATA_WORKERS,
    MountedArtifactStore,
    is_remote_datastore,
    metadata_workers,
    zarr_root_path,
)
from scarf.tools.repack_zarr import repack_store
from scarf.writers import H5adToZarr

N_CELLS = 240
RECIPE = {
    "filtering": False,
    "hvg_count": 40,
    "pca_dims": 5,
    "neighbors_k": 8,
    "umap": False,
    "cell_cycle": False,
    "doublets": False,
    "paris": False,
    "markers": True,
}


def _write_h5ad(path: Path, n_cells: int = N_CELLS, n_genes: int = 90) -> None:
    rng = np.random.default_rng(11)
    types = np.arange(n_cells) % 3
    programs = np.ones((3, n_genes))
    for kind in range(3):
        programs[kind, kind * 10 : kind * 10 + 10] = 8.0
    counts = rng.poisson(0.6 * programs[types]).astype(np.float32)
    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical([f"type_{kind}" for kind in types]),
            "condition": pd.Categorical(np.where(np.arange(n_cells) % 2, "a", "b")),
        },
        index=[f"cell{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(
        {"gene_short_name": [f"G{i}" for i in range(n_genes)]},
        index=[f"gene{i}" for i in range(n_genes)],
    )
    umap = np.column_stack([types, np.arange(n_cells)]).astype(np.float32)
    adata = ad.AnnData(X=sp.csr_matrix(counts), obs=obs, var=var, obsm={"X_umap": umap})
    adata.write_h5ad(path)


def _import(h5ad: Path, location: str | Store, *, workspace: str | None = None):
    reader = H5adReader(
        str(h5ad), cluster_keys=["cell_type"], embedding_roles={"X_umap": "umap"}
    )
    try:
        return H5adToZarr(
            reader, zarr_loc=location, assay_name="RNA", workspace=workspace
        ).dump()
    finally:
        reader.h5.close()


def _prepared_import(tmp_path: Path, location: str | Store, **options):
    """Import a small H5AD and prepare it with one writable open."""
    h5ad = tmp_path / "small.h5ad"
    _write_h5ad(h5ad, n_cells=30, n_genes=12)
    imported = _import(h5ad, location, **options)
    DataStore(location, default_assay="RNA", min_features_per_cell=-1, **options)
    return imported


def _mount(source: str, target: Path, **options) -> DataStore:
    return mount_datastore(
        source,
        at=str(target),
        default_assay="RNA",
        min_features_per_cell=-1,
        **options,
    )


def _serve_remote(monkeypatch, location: str, store: Store) -> None:
    """Open ``store`` for ``location``, as an object-store URI would be."""
    from scarf.storage import stores as stores_module

    real_make_store = stores_module.make_store

    def fake_make_store(loc, *, storage_options=None, read_only=False):
        if loc == location:
            return store.with_read_only(read_only)
        return real_make_store(
            loc, storage_options=storage_options, read_only=read_only
        )

    monkeypatch.setattr(stores_module, "make_store", fake_make_store)


def _files(path: str | Path) -> dict[str, bytes]:
    root = Path(path)
    return {
        str(file.relative_to(root)): file.read_bytes()
        for file in sorted(root.rglob("*"))
        if file.is_file()
    }


def _artifact_kinds(root: Path) -> list[str]:
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def _complete_lineage(datastore: DataStore, ref) -> list:
    statuses = [
        data["status"] for _, data in datastore.lineage(ref).graph.nodes(data=True)
    ]
    assert statuses
    assert all(status.complete for status in statuses)
    return sorted(status.ref.artifact_id for status in statuses)


@pytest.fixture(scope="module")
def published(tmp_path_factory):
    """A source imported with labels and an embedding, plus a labeled run."""
    root = tmp_path_factory.mktemp("published")
    h5ad = root / "published.h5ad"
    _write_h5ad(h5ad)
    location = str(root / "source.zarr")
    imported = _import(h5ad, location)
    source = DataStore(location, default_assay="RNA", min_features_per_cell=-1)
    run = source.pipeline.run(label="published", leiden={"partitions": [1.0]}, **RECIPE)
    summary = run.report()["summary"]
    return SimpleNamespace(
        location=location,
        imported=imported,
        outputs={key: run[key] for key in run},
        planned=len(summary["createdArtifacts"]) + len(summary["reusedArtifacts"]),
        assay_artifacts=source.list_artifacts(from_assay="RNA"),
        datastore_artifacts=source.list_artifacts(scope="datastore"),
        markers=source.get_markers(run["markers"]),
        lineage=_complete_lineage(source, run["markers"]),
    )


@pytest.fixture(scope="module")
def branched(published, tmp_path_factory):
    """A mount of a copy of the published source that ran a branch recipe."""
    root = tmp_path_factory.mktemp("branched")
    source = root / "source.zarr"
    shutil.copytree(published.location, source)
    before = _files(source)
    target = root / "target.zarr"
    mounted = _mount(str(source), target)
    run = mounted.pipeline.run(label="branch", leiden={"partitions": [0.5]}, **RECIPE)
    return SimpleNamespace(
        source=source, before=before, target=target, mounted=mounted, run=run
    )


def test_mount_lists_inspects_loads_and_traces_source_artifacts(published, tmp_path):
    before = _files(published.location)
    mounted = _mount(published.location, tmp_path / "target.zarr")

    assert isinstance(mounted.z.store, MountedArtifactStore)
    assert len(published.assay_artifacts) >= 12
    assert len(published.datastore_artifacts) >= 3
    assert mounted.list_artifacts(from_assay="RNA") == published.assay_artifacts
    assert mounted.list_artifacts(scope="datastore") == published.datastore_artifacts
    for ref in published.assay_artifacts + published.datastore_artifacts:
        assert mounted.inspect_artifact(ref).complete
    markers = published.outputs["markers"]
    assert set(mounted.load_artifact(markers).group_keys())
    pd.testing.assert_frame_equal(mounted.get_markers(markers), published.markers)
    assert _complete_lineage(mounted, markers) == published.lineage
    summary = mounted.summary()
    assert len(summary.assays[0].artifacts) == len(published.assay_artifacts)
    assert len(summary.artifacts) == len(published.datastore_artifacts)
    # Runs and their labels stay with the store that holds them.
    assert summary.pipeline_run_counts["total"] == 0
    assert mounted.pipeline.list_runs() == ()
    with pytest.raises(KeyError):
        mounted.pipeline.open(label="published")
    assert not is_remote_datastore(None, mounted.zw)
    assert metadata_workers(mounted.zw) == 1
    assert zarr_root_path(mounted.z) == str(tmp_path / "target.zarr")
    assert _files(published.location) == before


def test_mount_uses_artifacts_imported_into_its_source(published, tmp_path):
    before = _files(published.location)
    target = tmp_path / "target.zarr"
    mounted = _mount(published.location, target)
    labels = published.imported.clusterArtifacts["cell_type"]
    embedding = published.imported.embeddingArtifacts["X_umap"]

    selection = mounted.select_cells(labels, include=["type_1"])
    selected = read_stored_selection_indices(
        mounted.zw,
        selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    np.testing.assert_array_equal(selected, np.arange(1, N_CELLS, 3))
    markers = mounted.run_marker_search(
        labels, features=published.outputs["highly_variable_features"]
    )
    assert set(mounted.get_markers(markers)["group_id"]) == {
        "type_0",
        "type_1",
        "type_2",
    }
    assert cytebase.embeddings(mounted) == {"X_umap": embedding}
    assert cytebase.embedding(mounted) == embedding
    source = DataStore(
        published.location, zarr_mode="r", default_assay="RNA", min_features_per_cell=-1
    )
    pd.testing.assert_frame_equal(
        cytebase.embedding_coordinates(mounted, embedding),
        cytebase.embedding_coordinates(source, embedding),
    )
    assert (target / artifact_path(selection)).is_dir()
    assert (target / artifact_path(markers)).is_dir()
    assert _files(published.location) == before


def test_identical_recipe_on_a_mount_reuses_every_source_artifact(published, tmp_path):
    before = _files(published.location)
    target = tmp_path / "target.zarr"
    mounted = _mount(published.location, target)

    run = mounted.pipeline.run(
        label="published", leiden={"partitions": [1.0]}, **RECIPE
    )

    summary = run.report()["summary"]
    assert summary["createdArtifacts"] == []
    assert len(summary["reusedArtifacts"]) == published.planned
    assert {key: run[key] for key in run} == published.outputs
    assert not (target / "artifacts").exists()
    assert not (target / "RNA" / "artifacts").exists()
    assert _files(published.location) == before


def test_branch_recipe_writes_only_its_new_artifacts_to_the_target(published, branched):
    summary = branched.run.report()["summary"]
    created = sorted(item["kind"] for item in summary["createdArtifacts"])
    assert created == ["cluster_labels", "cluster_selection", "marker_table"]
    assert len(summary["reusedArtifacts"]) == published.planned - len(created)
    assert _artifact_kinds(branched.target / "RNA" / "artifacts") == created
    assert not (branched.target / "artifacts").exists()
    _complete_lineage(branched.mounted, branched.run["markers"])
    assert _files(branched.source) == branched.before


def test_store_writes_inside_source_groups_are_refused(published, tmp_path):
    source_before = _files(published.location)
    target = tmp_path / "target.zarr"
    mounted = _mount(published.location, target)
    target_before = _files(target)
    store = mounted.z.store
    group = artifact_path(published.outputs["markers"])
    document = f"{group}/zarr.json"
    payload = default_buffer_prototype().buffer.from_bytes(b"{}")

    for write in (
        lambda: store.set(document, payload),
        lambda: store.set_if_not_exists(f"{group}/extra/zarr.json", payload),
        lambda: store._set_many([(f"{group}/extra/zarr.json", payload)]),
        lambda: store.delete(document),
        lambda: store.delete_dir(group),
        lambda: store.delete_dir(f"{group}/feature_index"),
    ):
        with pytest.raises(PermissionError, match="read-only matrix source"):
            sync(write())
    with pytest.raises(PermissionError, match="read-only matrix source"):
        mounted.zw[group].attrs["note"] = "changed"
    with pytest.raises(PermissionError, match="read-only matrix source"):
        del mounted.zw[group]

    assert mounted.inspect_artifact(published.outputs["markers"]).complete
    assert _files(published.location) == source_before
    assert _files(target) == target_before


def test_failed_artifact_write_on_a_mount_discards_only_its_own_group(
    published, tmp_path
):
    source_before = _files(published.location)
    target = tmp_path / "target.zarr"
    mounted = _mount(published.location, target)
    kept = mounted.select_cells(
        published.imported.clusterArtifacts["cell_type"], include=["type_0"]
    )
    planned = plan_artifact(
        mounted.zw,
        scope="assay",
        assay="RNA",
        kind="feature_summary",
        operation="interrupted_write",
        parameters={},
        inputs={"cell_selection": kept},
        execution_options={},
    )

    with pytest.raises(RuntimeError, match="injected"):
        with artifact_transaction(mounted.zw, planned) as group:
            group.create_array("values", data=np.arange(4))
            assert (target / artifact_path(planned.ref)).is_dir()
            raise RuntimeError("injected failure")

    assert not (target / artifact_path(planned.ref)).exists()
    assert not mounted.inspect_artifact(planned.ref).exists
    assert mounted.inspect_artifact(kept).complete
    assert mounted.list_artifacts(from_assay="RNA") == published.assay_artifacts
    assert _files(published.location) == source_before


def test_namespace_never_reads_source_tables_counts_or_runs(published, tmp_path):
    target = tmp_path / "target.zarr"
    mounted = _mount(published.location, target)
    mounted.cells.drop("condition")

    reopened = DataStore(str(target), default_assay="RNA", min_features_per_cell=-1)
    assert "condition" not in reopened.cells.columns
    source = DataStore(
        published.location, zarr_mode="r", default_assay="RNA", min_features_per_cell=-1
    )
    assert "condition" in source.cells.columns
    prototype = default_buffer_prototype()
    for key in (
        "cellData/condition/zarr.json",
        "RNA/counts/zarr.json",
        "pipeline/zarr.json",
    ):
        assert sync(source.z.store.exists(key))
        assert not sync(reopened.z.store.exists(key))
        assert sync(reopened.z.store.get(key, prototype)) is None
    # The target holds no artifacts yet; the listings above the roots name them,
    # and the source holds their directory documents.
    assert not (target / "artifacts").exists()
    assert not (target / "RNA" / "artifacts").exists()
    names = set(collect_aiterator(reopened.z.store.list_dir("")))
    assert "artifacts" in names
    assert "pipeline" not in names
    assert "artifacts" in set(collect_aiterator(reopened.z.store.list_dir("RNA")))
    for document in (
        "artifacts/zarr.json",
        "artifacts/cell_selection/zarr.json",
        "RNA/artifacts/zarr.json",
    ):
        assert sync(reopened.z.store.exists(document))
    assert not sync(reopened.z.store.exists("RNA/artifacts/missing/zarr.json"))


def test_read_only_reopen_resolves_source_artifacts(published, branched):
    reopened = DataStore(
        str(branched.target),
        zarr_mode="r",
        default_assay="RNA",
        min_features_per_cell=-1,
    )

    assert reopened.z.store.read_only
    assert reopened.list_artifacts(from_assay="RNA") == branched.mounted.list_artifacts(
        from_assay="RNA"
    )
    pd.testing.assert_frame_equal(
        reopened.get_markers(published.outputs["markers"]), published.markers
    )
    pd.testing.assert_frame_equal(
        reopened.get_markers(reopened.pipeline.open(label="branch")["markers"]),
        branched.mounted.get_markers(branched.run["markers"]),
    )
    with pytest.raises(PermissionError):
        reopened.run_marker_search(
            published.imported.clusterArtifacts["cell_type"],
            features=published.outputs["highly_variable_features"],
        )


def test_repacked_mount_is_self_contained_after_its_source_moves(
    published, branched, tmp_path
):
    mounted = branched.mounted
    assay_artifacts = mounted.list_artifacts(from_assay="RNA")
    datastore_artifacts = mounted.list_artifacts(scope="datastore")
    markers = mounted.get_markers(branched.run["markers"])
    lineage = _complete_lineage(mounted, branched.run["markers"])
    assert len(assay_artifacts) == len(published.assay_artifacts) + 3

    output = tmp_path / "repacked.zarr"
    repack_store(str(branched.target), str(output), nthreads=1)
    moved = branched.source.with_name("moved.zarr")
    branched.source.rename(moved)
    try:
        assert MATRIX_SOURCE_ATTR not in zarr.open_group(str(output), mode="r").attrs
        repacked = DataStore(str(output), default_assay="RNA", min_features_per_cell=-1)
        assert not isinstance(repacked.z.store, MountedArtifactStore)
        assert repacked.list_artifacts(from_assay="RNA") == assay_artifacts
        assert repacked.list_artifacts(scope="datastore") == datastore_artifacts
        run = repacked.pipeline.open(label="branch")
        assert _complete_lineage(repacked, run["markers"]) == lineage
        pd.testing.assert_frame_equal(repacked.get_markers(run["markers"]), markers)
    finally:
        # The other tests of this module share the mounted source.
        moved.rename(branched.source)


def test_workspace_mount_resolves_both_artifact_roots(tmp_path):
    source = str(tmp_path / "source.zarr")
    imported = _prepared_import(tmp_path, source, workspace="analysis")
    labels = imported.clusterArtifacts["cell_type"]
    target = tmp_path / "target.zarr"

    mounted = _mount(source, target, workspace="analysis")

    assert mounted.workspace == "analysis"
    assert mounted.list_artifacts(scope="datastore") == [imported.cellSelection]
    assert set(mounted.list_artifacts(from_assay="RNA")) == {
        labels,
        imported.embeddingArtifacts["X_umap"],
    }
    assert mounted.inspect_artifact(imported.cellSelection).complete
    selection = mounted.select_cells(labels, include=["type_2"])
    assert (target / "analysis" / artifact_path(selection)).is_dir()
    assert not (target / "analysis" / artifact_path(labels)).exists()


def test_object_store_source_resolves_artifacts(tmp_path, monkeypatch):
    remote = ObjectStore(store=ObjectMemoryStore())
    imported = _prepared_import(tmp_path, remote)
    labels = imported.clusterArtifacts["cell_type"]
    location = "s3://atlas/published.zarr"
    _serve_remote(monkeypatch, location, remote)
    target = tmp_path / "target.zarr"
    mounted = _mount(location, target)

    assert mounted.inspect_artifact(labels).complete
    np.testing.assert_array_equal(
        mounted.load_artifact(labels)["values"][:],
        zarr.open_group(store=remote, mode="r")[artifact_path(labels)]["values"][:],
    )
    assert is_remote_datastore(None, mounted.zw)
    assert metadata_workers(mounted.zw) == REMOTE_METADATA_WORKERS
    assert zarr_root_path(mounted.z) == str(target)
    assert mounted.select_cells(labels, include=["type_0"]).kind == "cell_selection"


def test_local_cache_follows_the_store_that_holds_the_normalized_artifact(
    tmp_path, monkeypatch
):
    remote = ObjectStore(store=ObjectMemoryStore())
    _prepared_import(tmp_path, remote)
    source = DataStore(remote, default_assay="RNA", min_features_per_cell=-1)
    cells = source.snapshot_cell_selection()
    features = source.select_all_features(from_assay="RNA")
    published = source.run_normalization(cells, features)
    location = "s3://atlas/published.zarr"
    _serve_remote(monkeypatch, location, remote)
    target = tmp_path / "target.zarr"
    mounted = _mount(location, target)

    assert mounted.run_normalization(cells, features) == published
    written = mounted.run_normalization(cells, features, log_transform=False)
    assert (target / artifact_path(written)).is_dir()
    # A node is judged by the store that serves it, whatever the location.
    reused_data = artifact_group(mounted.zw, published)["data"]
    written_data = artifact_group(mounted.zw, written)["data"]
    for zarr_loc in (None, mounted.zarr_loc, LocalStore(str(target))):
        assert is_remote_datastore(zarr_loc, mounted.zw)
        assert is_remote_datastore(zarr_loc, reused_data)
        assert not is_remote_datastore(zarr_loc, written_data)

    # Only the normalized artifact read from the remote source is staged.
    scratch = tmp_path / "scratch"
    for normalized in (published, written):
        mounted.run_pca(normalized, dims=3, local_cache=str(scratch))
    assert [path.name for path in scratch.iterdir()] == [published.artifact_id]
    assert (scratch / published.artifact_id / "normed.zarr").is_dir()


@pytest.mark.parametrize(
    "codec_pipeline",
    [
        "zarr.core.codec_pipeline.BatchedCodecPipeline",
        pytest.param(
            "zarr.core.codec_pipeline.FusedCodecPipeline",
            marks=pytest.mark.skipif(
                not hasattr(zarr_codec_pipeline, "FusedCodecPipeline"),
                reason="This Zarr has no synchronous codec pipeline",
            ),
        ),
    ],
)
def test_partial_shard_reads_of_source_artifacts_go_to_the_source(
    tmp_path, codec_pipeline
):
    source = str(tmp_path / "source.zarr")
    imported = _prepared_import(tmp_path, source)
    path = f"{artifact_path(imported.embeddingArtifacts['X_umap'])}/sharded"
    values = np.arange(1, 65, dtype=np.float32).reshape(16, 4)
    zarr.open_group(source, mode="r+").create_array(
        path, data=values, chunks=(2, 4), shards=(8, 4)
    )
    mounted = _mount(source, tmp_path / "target.zarr")

    with zarr.config.set({"codec_pipeline.path": codec_pipeline}):
        array = zarr.open_array(store=mounted.z.store, path=path, mode="r")
        # Rows 2 to 5 are two inner chunks of the first shard, a ranged read.
        np.testing.assert_array_equal(array[2:6], values[2:6])
        np.testing.assert_array_equal(array[:], values)


def test_store_protocol_routes_each_key_to_the_store_that_holds_it(tmp_path):
    source = str(tmp_path / "source.zarr")
    imported = _prepared_import(tmp_path, source)
    target = tmp_path / "target.zarr"
    mounted = _mount(source, target)
    selection = mounted.select_cells(
        imported.clusterArtifacts["cell_type"], include=["type_1"]
    )
    store = mounted.z.store
    prototype = default_buffer_prototype()
    source_key = f"{artifact_path(imported.cellSelection)}/zarr.json"
    target_key = f"{artifact_path(selection)}/zarr.json"
    direct = {
        source_key: zarr.open_group(source, mode="r").store,
        target_key: zarr.open_group(str(target), mode="r").store,
    }
    expected = {
        key: sync(owner.get(key, prototype)).to_bytes() for key, owner in direct.items()
    }
    keys = [source_key, target_key, source_key]
    # Counts live only in the source, outside the artifact roots.
    counts_key = "RNA/counts/zarr.json"

    key_ranges = [(key, None) for key in keys]
    key_ranges += [(source_key, RangeByteRequest(1, 5)), (counts_key, None)]
    *values, ranged, counts = sync(store.get_partial_values(prototype, key_ranges))
    assert [value.to_bytes() for value in values] == [expected[key] for key in keys]
    assert ranged.to_bytes() == expected[source_key][1:5]
    assert counts is None
    requests = [(key, prototype, None) for key in [*keys, counts_key]]
    many = dict(collect_aiterator(store._get_many(requests)))
    assert many.pop(counts_key) is None
    assert {key: value.to_bytes() for key, value in many.items()} == expected
    assert sync(store.getsize(source_key)) == len(expected[source_key])
    listed = set(collect_aiterator(store.list_prefix("artifacts/")))
    assert {source_key, target_key, "artifacts/cell_selection/zarr.json"} <= listed
    every_key = set(collect_aiterator(store.list()))
    assert {source_key, target_key, "cellData/ids/zarr.json"} <= every_key
    assert counts_key not in every_key
    assert set(collect_aiterator(store.list_dir("artifacts/cell_selection"))) == {
        "zarr.json",
        imported.cellSelection.artifact_id,
        selection.artifact_id,
    }
    source_group = artifact_path(imported.cellSelection)
    # A stray target file inside a source group is not part of the namespace.
    stray = target / source_group / "stray"
    stray.parent.mkdir(parents=True)
    stray.write_bytes(b"x")
    assert set(collect_aiterator(store.list_prefix(f"{source_group}/"))) == set(
        collect_aiterator(direct[source_key].list_prefix(f"{source_group}/"))
    )
    assert set(collect_aiterator(store.list_dir(source_group))) == set(
        collect_aiterator(direct[source_key].list_dir(source_group))
    )
    assert not sync(store.exists(f"{source_group}/stray"))
    assert not sync(store.is_empty(source_group))
    read_only = store.with_read_only(True)
    assert read_only == MountedArtifactStore(
        store._store.with_read_only(True), store._source, store._roots
    )
    assert read_only.read_only
    assert read_only._in_source is store._in_source
    assert str(store) == str(store._store)
    for call in (
        lambda: store.get_sync(source_key),
        lambda: store.set_sync(target_key, prototype.buffer.from_bytes(b"{}")),
        lambda: store.delete_sync(target_key),
    ):
        with pytest.raises(TypeError, match="asynchronous"):
            call()
    # Clearing the namespace clears the target alone.
    sync(store.clear())
    assert not target.exists() or not any(target.iterdir())
    assert not sync(store.exists(target_key))
    assert sync(store.exists(source_key))


def test_group_origins_follow_the_stores_after_misses_and_deletes(tmp_path):
    source = str(tmp_path / "source.zarr")
    imported = _prepared_import(tmp_path, source)
    target = tmp_path / "target.zarr"
    mounted = _mount(source, target)
    store = mounted.z.store
    prototype = default_buffer_prototype()
    payload = prototype.buffer.from_bytes(b"{}")
    published_later = f"RNA/artifacts/feature_summary/{'a' * 64}"
    created_later = f"RNA/artifacts/feature_summary/{'b' * 64}"
    for group in (published_later, created_later):
        assert not sync(store.exists(f"{group}/zarr.json"))
        assert sync(store.get(f"{group}/zarr.json", prototype)) is None

    # A miss is not remembered: each group resolves to the store that gains it.
    zarr.open_group(source, mode="r+").create_group(
        published_later, attributes={"origin": "source"}
    )
    mounted.zw.create_group(created_later, attributes={"origin": "target"})
    assert mounted.zw[published_later].attrs["origin"] == "source"
    assert mounted.zw[created_later].attrs["origin"] == "target"
    with pytest.raises(PermissionError, match="read-only matrix source"):
        sync(store.set(f"{published_later}/extra", payload))
    sync(store.set(f"{created_later}/extra", payload))
    sync(store._set_many([(f"{created_later}/many", payload)]))
    assert (target / created_later / "extra").is_file()
    assert (target / created_later / "many").is_file()
    assert not (target / published_later).exists()

    # A target copy of a source group wins until it is deleted.
    shared = artifact_path(imported.cellSelection)
    shutil.copytree(Path(source) / shared, target / shared)
    mounted.zw[shared].attrs["copy"] = "target"
    del mounted.zw[shared]
    assert not (target / shared).exists()
    assert "copy" not in mounted.zw[shared].attrs
    assert mounted.inspect_artifact(imported.cellSelection).complete


def test_mounting_a_fresh_import_asks_for_one_writable_open(tmp_path):
    h5ad = tmp_path / "fresh.h5ad"
    _write_h5ad(h5ad, n_cells=30, n_genes=12)
    source = str(tmp_path / "source.zarr")
    imported = _import(h5ad, source)
    target = tmp_path / "target.zarr"

    with pytest.raises(
        ValueError, match=r"not prepared yet\. Open the store once with zarr_mode='r\+'"
    ):
        _mount(source, target)
    assert not target.exists()

    DataStore(source, default_assay="RNA", min_features_per_cell=-1)
    mounted = _mount(source, target)
    assert mounted.inspect_artifact(imported.cellSelection).complete


def test_cytebase_close_closes_the_target_and_the_source_once(
    published, tmp_path, monkeypatch
):
    from scarf.cytebase.connector import _close

    mounted = _mount(published.location, tmp_path / "target.zarr")
    closed: list[str] = []
    for name, store in (
        ("target", mounted.z.store._store),
        ("source", mounted._matrix_z.store),
    ):
        monkeypatch.setattr(store, "close", lambda name=name: closed.append(name))

    _close(mounted)

    assert sorted(closed) == ["source", "target"]


def test_arrays_written_on_a_local_mount_use_the_local_profile(published, tmp_path):
    mounted = _mount(published.location, tmp_path / "target.zarr")
    plain = DataStore(published.location, default_assay="RNA", min_features_per_cell=-1)

    assert resolve_storage_profile(mounted.z.store) == "fast_local"
    mounted.cells.insert("flag", np.ones(N_CELLS, dtype=bool), overwrite=True)
    written = mounted.z["cellData"]["flag"]
    reference = plain.z["cellData"]["I"]
    assert [type(codec) for codec in written.compressors] == [
        type(codec) for codec in reference.compressors
    ]

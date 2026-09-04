import sys

import numpy as np
import pytest
import zarr
from zarr.codecs import ZstdCodec

import scarf.storage.pipeline_runs as pipeline_run_storage
from scarf.storage.ann_index import (
    ANN_INDEX_CHUNK_BYTES,
    ANN_INDEX_FORMAT_VERSION,
    _ANN_INDEX_METADATA,
)
from scarf.storage.layout import normalize_chunks
from scarf.storage.pipeline_runs import (
    PipelineOutputRecord,
    PipelineStageMetrics,
    PipelineStageOutputRecord,
    complete_pipeline_run_record,
    create_pipeline_run_record,
    finish_pipeline_stage_record,
    load_pipeline_run_record,
    start_pipeline_stage_record,
)
from scarf.storage.refs import ArtifactRef, artifact_path
from scarf.storage.types import array_metadata_shards
from scarf.tools import repack_zarr as repack_module
from scarf.tools.repack_zarr import repack_store


def test_repack_store_round_trip(toy_crdir_writer, tmp_path):
    output = tmp_path / "repacked.zarr"
    repack_store(toy_crdir_writer, str(output), profile="fast_local")

    src = zarr.open_group(toy_crdir_writer, mode="r")
    dst = zarr.open_group(str(output), mode="r")

    assert set(src.keys()) == set(dst.keys())
    assay_names = [name for name in src.keys() if src[name].attrs.get("is_assay")]
    from scarf.assay.classification import is_rna_assay_type

    for assay_name in assay_names:
        src_assay = src[assay_name]
        dst_assay = dst[assay_name]
        assert src_assay.attrs.get("is_assay") is True
        assert "counts" in dst_assay
        assert src_assay["counts"].shape == dst_assay["counts"].shape
        assert (src_assay["counts"][...] == dst_assay["counts"][...]).all()
        if is_rna_assay_type(assay_name):
            assert dst_assay["countsT"].attrs["complete"] is True
            np.testing.assert_array_equal(
                dst_assay["countsT"][:],
                np.asarray(dst_assay["counts"][:]).T,
            )
        else:
            assert "countsT" not in dst_assay


def test_repack_v2_without_counts_t_builds_complete_transpose(tmp_path):
    source = tmp_path / "source_v2.zarr"
    output = tmp_path / "output_v3.zarr"
    root = zarr.open_group(str(source), mode="w", zarr_format=2)
    assay = root.create_group("RNA")
    assay.attrs["is_assay"] = True
    values = np.arange(12, dtype=np.uint32).reshape(3, 4)
    assay.create_array("counts", data=values, chunks=(2, 2))

    repack_store(str(source), str(output))

    result = zarr.open_group(str(output), mode="r")
    assert result.metadata.zarr_format == 3
    assert result["RNA/countsT"].attrs["complete"] is True
    np.testing.assert_array_equal(result["RNA/counts"][:], values)
    np.testing.assert_array_equal(result["RNA/countsT"][:], values.T)


def test_repack_rebuilds_incorrect_source_counts_t(tmp_path):
    source = tmp_path / "source_wrong.zarr"
    output = tmp_path / "output_fixed.zarr"
    root = zarr.open_group(str(source), mode="w")
    assay = root.create_group("RNA")
    assay.attrs["is_assay"] = True
    values = np.arange(15, dtype=np.uint32).reshape(5, 3)
    assay.create_array("counts", data=values, chunks=(2, 3))
    stale = assay.create_array(
        "countsT",
        data=np.zeros((3, 5), dtype=np.uint32),
    )
    stale.attrs["complete"] = True

    repack_store(str(source), str(output))

    result = zarr.open_group(str(output), mode="r")
    np.testing.assert_array_equal(result["RNA/countsT"][:], values.T)
    assert result["RNA/countsT"].attrs["complete"] is True


def test_repack_discards_retired_assay_state(tmp_path):
    source = tmp_path / "source_with_state.zarr"
    output = tmp_path / "output_without_state.zarr"
    root = zarr.open_group(str(source), mode="w")
    assay = root.create_group("RNA")
    assay.attrs["is_assay"] = True
    assay.create_array(
        "counts",
        data=np.arange(6, dtype=np.uint32).reshape(2, 3),
    )
    state = assay.create_group("state")
    state.create_array("legacy", data=np.array([1], dtype=np.uint8))

    repack_store(str(source), str(output))

    result = zarr.open_group(str(output), mode="r")
    assert "state" not in result["RNA"]
    np.testing.assert_array_equal(result["RNA/counts"][:], assay["counts"][:])


def test_repack_discards_state_from_every_workspace_sharing_an_assay(tmp_path):
    source = tmp_path / "source_with_workspace_state.zarr"
    output = tmp_path / "output_without_workspace_state.zarr"
    root = zarr.open_group(str(source), mode="w")
    for workspace_name in ("first", "second"):
        assay = root.create_group(f"{workspace_name}/RNA")
        assay.attrs["is_assay"] = True
        assay.create_array("metadata", data=np.array([1], dtype=np.uint8))
        state = assay.create_group("state")
        state.create_array("legacy", data=np.array([1], dtype=np.uint8))
    counts = root.create_group("matrices/RNA")
    values = np.arange(6, dtype=np.uint32).reshape(2, 3)
    counts.create_array("counts", data=values)

    repack_store(str(source), str(output))

    result = zarr.open_group(str(output), mode="r")
    assert "state" not in result["first/RNA"]
    assert "state" not in result["second/RNA"]
    np.testing.assert_array_equal(result["first/RNA/metadata"][:], [1])
    np.testing.assert_array_equal(result["second/RNA/metadata"][:], [1])
    np.testing.assert_array_equal(result["matrices/RNA/counts"][:], values)


def test_repack_workspace_counts_uses_requested_profile(tmp_path):
    source = tmp_path / "source_workspace.zarr"
    output = tmp_path / "output_workspace.zarr"
    root = zarr.open_group(str(source), mode="w")
    metadata = root.create_group("workspace/RNA")
    metadata.attrs["is_assay"] = True
    counts_group = root.create_group("matrices/RNA")
    values = np.arange(8, dtype=np.uint32).reshape(4, 2)
    counts_group.create_array("counts", data=values, chunks=(2, 2))

    repack_store(str(source), str(output), profile="cloud")

    result = zarr.open_group(str(output), mode="r")
    assay = result["matrices/RNA"]
    counts = assay["counts"]
    counts_t = assay["countsT"]
    np.testing.assert_array_equal(counts_t[:], values.T)
    assert counts_t.attrs["complete"] is True
    spec = assay.attrs["scarf:zarr_spec"]
    assert spec["profile"] == "cloud"
    assert list(spec["chunks"]) == list(counts.chunks)
    stored_shards = array_metadata_shards(counts)
    assert spec["shards"] == (None if stored_shards is None else list(stored_shards))
    assert isinstance(counts.compressors[0], ZstdCodec)
    assert isinstance(counts_t.compressors[0], ZstdCodec)


def test_repack_rejects_source_destination_alias_before_overwrite(tmp_path):
    source = tmp_path / "source.zarr"
    root = zarr.open_group(str(source), mode="w")
    root.create_array("sentinel", data=np.array([1, 2, 3]))
    equivalent_path = source / ".." / source.name

    with pytest.raises(ValueError, match="different stores"):
        repack_store(str(source), str(equivalent_path))

    reopened = zarr.open_group(str(source), mode="r")
    np.testing.assert_array_equal(reopened["sentinel"][:], [1, 2, 3])


@pytest.mark.parametrize("destination_relation", ["child", "parent"])
def test_repack_rejects_overlapping_local_paths_before_overwrite(
    tmp_path,
    destination_relation,
):
    if destination_relation == "child":
        source = tmp_path / "source.zarr"
        output = source / "nested.zarr"
    else:
        output = tmp_path / "container.zarr"
        source = output / "source.zarr"

    root = zarr.open_group(str(source), mode="w")
    root.create_array("sentinel", data=np.array([1, 2, 3]))

    with pytest.raises(ValueError, match="must not overlap"):
        repack_store(str(source), str(output))

    reopened = zarr.open_group(str(source), mode="r")
    np.testing.assert_array_equal(reopened["sentinel"][:], [1, 2, 3])


@pytest.mark.parametrize(
    ("source", "output"),
    [
        ("s3://bucket/source.zarr", "s3://bucket/source.zarr/nested.zarr"),
        ("s3://bucket/container.zarr/source.zarr", "s3://bucket/container.zarr"),
    ],
)
def test_repack_rejects_overlapping_uri_paths_before_open(
    source,
    output,
    monkeypatch,
):
    monkeypatch.setattr(
        repack_module,
        "open_store",
        lambda *_args, **_kwargs: pytest.fail(
            "overlap must be rejected before opening either store"
        ),
    )

    with pytest.raises(ValueError, match="must not overlap"):
        repack_store(source, output)


def test_repack_preserves_root_attrs(tmp_path):
    source = tmp_path / "source.zarr"
    output = tmp_path / "output.zarr"
    root = zarr.open_group(str(source), mode="w", zarr_format=2)
    root.attrs["defaultAssay"] = "RNA"
    root.attrs["assayTypes"] = {"RNA": "RNA"}
    root.attrs["complete"] = True
    assay = root.create_group("RNA")
    assay.attrs["is_assay"] = True
    values = np.arange(6, dtype=np.uint32).reshape(2, 3)
    assay.create_array("counts", data=values, chunks=(2, 3))

    repack_store(str(source), str(output))

    result = zarr.open_group(str(output), mode="r")
    assert result.attrs["defaultAssay"] == "RNA"
    assert result.attrs["assayTypes"] == {"RNA": "RNA"}
    assert result.attrs["complete"] is True


def test_repack_preserves_non_count_completion_attrs(tmp_path):
    source = tmp_path / "source.zarr"
    output = tmp_path / "output.zarr"
    root = zarr.open_group(str(source), mode="w")
    assay = root.create_group("RNA")
    assay.attrs["is_assay"] = True
    assay.create_array(
        "counts",
        data=np.arange(6, dtype=np.uint32).reshape(2, 3),
        chunks=(2, 3),
    )
    cell = root.create_group("cellData")
    cell.attrs["complete"] = True
    cell.create_array("ids", data=np.array(["c1", "c2"]))
    artifacts = root.create_group("artifacts")
    table = artifacts.create_group("marker_table")
    slot = table.create_group("slot")
    slot.attrs["complete"] = True
    slot.create_array("values", data=np.array([1.0, 2.0]))

    repack_store(str(source), str(output))

    result = zarr.open_group(str(output), mode="r")
    assert result["cellData"].attrs["complete"] is True
    assert result["artifacts/marker_table/slot"].attrs["complete"] is True


def _seed_labeled_pipeline_records(
    root,
    *,
    artifact_id,
    completed_run_id,
    running_run_id,
    completed_label,
    running_label,
):
    artifact = ArtifactRef(
        scope="datastore",
        kind="cell_selection",
        artifact_id=artifact_id,
    )
    artifact_node = root.create_group(artifact_path(artifact))
    artifact_node.attrs.update(
        {
            "artifact_id": artifact.artifact_id,
            "kind": artifact.kind,
            "provenance": {
                "operation": "test_selection",
                "parameters": {},
                "inputs": {},
            },
            "execution_options": {},
            "complete": True,
        }
    )
    artifact_node.create_array("values", data=np.asarray([True, False]))

    completed = create_pipeline_run_record(
        root,
        recipe="basic_rna_analysis",
        requested_label=completed_label,
        assay="RNA",
        config={"filtering": False},
        stage_order=("snapshot",),
        scarf_version="1.0.0",
        run_id=completed_run_id,
        started_at_ns=100,
    )
    start_pipeline_stage_record(
        root,
        run_id=completed.run_id,
        ordinal=0,
        stage="snapshot",
        started_at_ns=110,
    )
    finish_pipeline_stage_record(
        root,
        run_id=completed.run_id,
        ordinal=0,
        status="completed",
        outputs=(PipelineStageOutputRecord("selection", artifact, False),),
        metrics=PipelineStageMetrics(
            wall_seconds=0.01,
            rss_baseline_bytes=None,
            rss_peak_bytes=None,
            rss_incremental_peak_bytes=None,
            sample_interval_seconds=0.1,
            sample_count=0,
            sampling_error_count=0,
            rss_unavailable_reason="test",
        ),
        finished_at_ns=120,
    )
    complete_pipeline_run_record(
        root,
        run_id=completed.run_id,
        outputs=(PipelineOutputRecord("selection", artifact),),
        fields=(),
        finished_at_ns=130,
    )
    running = create_pipeline_run_record(
        root,
        recipe="basic_rna_analysis",
        requested_label=running_label,
        assay="RNA",
        config={"filtering": True},
        stage_order=("snapshot",),
        scarf_version="1.0.0",
        run_id=running_run_id,
        started_at_ns=200,
    )
    start_pipeline_stage_record(
        root,
        run_id=running.run_id,
        ordinal=0,
        stage="snapshot",
        started_at_ns=210,
    )
    pipeline_run_storage._claim_pipeline_label(
        root,
        running_label,
        running.run_id,
    )
    return completed.run_id, running.run_id


def test_repack_copies_pipeline_records_and_label_claims_in_all_workspaces(tmp_path):
    source = tmp_path / "source_runs.zarr"
    output = tmp_path / "output_runs.zarr"
    root = zarr.open_group(str(source), mode="w", zarr_format=2)
    assay = root.create_group("RNA")
    assay.attrs["is_assay"] = True
    assay.create_array(
        "counts",
        data=np.arange(6, dtype=np.uint32).reshape(2, 3),
        chunks=(2, 3),
    )

    root_run_ids = _seed_labeled_pipeline_records(
        root,
        artifact_id="c" * 64,
        completed_run_id="a" * 64,
        running_run_id="b" * 64,
        completed_label="baseline",
        running_label="interrupted",
    )
    workspace = root.create_group("analysis_workspace")
    workspace_run_ids = _seed_labeled_pipeline_records(
        workspace,
        artifact_id="f" * 64,
        completed_run_id="d" * 64,
        running_run_id="e" * 64,
        completed_label="workspace-baseline",
        running_label="workspace-interrupted",
    )
    namespaces = (
        ("", root, root_run_ids, "baseline", "interrupted"),
        (
            "analysis_workspace",
            workspace,
            workspace_run_ids,
            "workspace-baseline",
            "workspace-interrupted",
        ),
    )
    record_paths = tuple(
        f"{prefix + '/' if prefix else ''}pipeline/runs/{run_id}{suffix}"
        for prefix, _namespace, run_ids, _completed_label, _running_label in namespaces
        for run_id in run_ids
        for suffix in ("", "/stages/0")
    )
    source_attrs = {path: dict(root[path].attrs) for path in record_paths}

    repack_store(str(source), str(output))

    result = zarr.open_group(str(output), mode="r+")
    assert {path: dict(result[path].attrs) for path in record_paths} == source_attrs
    for prefix, source_root, run_ids, completed_label, running_label in namespaces:
        result_root = result if prefix == "" else result[prefix]
        completed_run_id, running_run_id = run_ids
        assert load_pipeline_run_record(
            result_root,
            completed_run_id,
        ) == load_pipeline_run_record(source_root, completed_run_id)
        assert load_pipeline_run_record(
            result_root,
            running_run_id,
        ) == load_pipeline_run_record(source_root, running_run_id)

        with pytest.raises(ValueError, match="already committed"):
            pipeline_run_storage._claim_pipeline_label(
                result_root,
                completed_label,
                "1" * 64,
            )
        with pytest.raises(ValueError, match="currently being finalized"):
            pipeline_run_storage._claim_pipeline_label(
                result_root,
                running_label,
                "2" * 64,
            )


def test_repack_skips_copying_counts_t_when_sharding(tmp_path, monkeypatch):
    source = tmp_path / "source.zarr"
    output = tmp_path / "output.zarr"
    root = zarr.open_group(str(source), mode="w")
    assay = root.create_group("RNA")
    assay.attrs["is_assay"] = True
    values = np.arange(12, dtype=np.uint32).reshape(3, 4)
    assay.create_array("counts", data=values, chunks=(2, 2))
    stale = assay.create_array("countsT", data=np.zeros((4, 3), dtype=np.uint32))
    stale.attrs["complete"] = True

    created: list[str] = []
    real_create = zarr.Group.create_array

    def tracking_create(self, name, *args, **kwargs):
        created.append(name)
        return real_create(self, name, *args, **kwargs)

    monkeypatch.setattr(zarr.Group, "create_array", tracking_create)
    repack_store(str(source), str(output))

    assert created.count("countsT") == 1
    result = zarr.open_group(str(output), mode="r")
    np.testing.assert_array_equal(result["RNA/countsT"][:], values.T)


def test_repack_streams_non_count_2d_arrays(tmp_path):
    source = tmp_path / "source.zarr"
    output = tmp_path / "output.zarr"
    root = zarr.open_group(str(source), mode="w")
    assay = root.create_group("RNA")
    assay.attrs["is_assay"] = True
    values = np.arange(12, dtype=np.uint32).reshape(3, 4)
    assay.create_array("counts", data=values, chunks=(2, 2))
    embedding = np.arange(15, dtype=np.float32).reshape(3, 5)
    assay.create_array("embedding", data=embedding, chunks=(2, 5))
    cell = root.create_group("cellData")
    cell.create_array("ids", data=np.array(["c1", "c2", "c3"]))

    repack_store(str(source), str(output))

    result = zarr.open_group(str(output), mode="r")
    np.testing.assert_array_equal(result["RNA/embedding"][:], embedding)
    np.testing.assert_array_equal(result["cellData/ids"][:], ["c1", "c2", "c3"])


def test_repack_preserves_ann_like_1d_chunks_and_attrs(tmp_path):
    source = tmp_path / "source.zarr"
    output = tmp_path / "output.zarr"
    root = zarr.open_group(str(source), mode="w")
    assay = root.create_group("RNA")
    assay.attrs["is_assay"] = True
    assay.create_array("counts", data=np.ones((2, 1), dtype=np.uint32))
    n_bytes = ANN_INDEX_CHUNK_BYTES + 1000
    payload = np.arange(n_bytes, dtype=np.uint8)
    ann = root.create_group("ann")
    arr = ann.create_array(
        "ann_idx_bytes",
        data=payload,
        chunks=(ANN_INDEX_CHUNK_BYTES,),
    )
    arr.attrs["ann_index_format_version"] = ANN_INDEX_FORMAT_VERSION
    arr.attrs["metric"] = "l2"
    arr.attrs["dimensions"] = 15
    arr.attrs["element_count"] = 100
    arr.attrs["payload_sha256"] = "abc"
    arr.attrs["byte_length"] = n_bytes

    repack_store(str(source), str(output))

    result = zarr.open_group(str(output), mode="r")["ann/ann_idx_bytes"]
    expected_chunks = normalize_chunks((ANN_INDEX_CHUNK_BYTES,), (n_bytes,))
    assert result.chunks == expected_chunks
    np.testing.assert_array_equal(result[:], payload)
    for key in _ANN_INDEX_METADATA:
        assert result.attrs[key] == arr.attrs[key]
    assert result.attrs["byte_length"] == n_bytes


def test_repack_preserves_non_count_2d_shards(tmp_path):
    source = tmp_path / "source.zarr"
    output = tmp_path / "output.zarr"
    root = zarr.open_group(str(source), mode="w")
    assay = root.create_group("RNA")
    assay.attrs["is_assay"] = True
    assay.create_array("counts", data=np.ones((8, 2), dtype=np.uint32))
    embedding = np.arange(40, dtype=np.float32).reshape(8, 5)
    assay.create_array(
        "embedding",
        data=embedding,
        chunks=(2, 5),
        shards=(4, 5),
        fill_value=np.nan,
    )

    repack_store(str(source), str(output), profile="cloud")

    result = zarr.open_group(str(output), mode="r")["RNA/embedding"]
    np.testing.assert_array_equal(result[:], embedding)
    assert result.chunks == (2, 5)
    assert array_metadata_shards(result) == (4, 5)
    assert isinstance(result.compressors[0], ZstdCodec)


def test_repack_realigns_shards_when_chunks_clamp(tmp_path):
    source = tmp_path / "source.zarr"
    output = tmp_path / "output.zarr"
    root = zarr.open_group(str(source), mode="w")
    assay = root.create_group("RNA")
    assay.attrs["is_assay"] = True
    assay.create_array("counts", data=np.ones((4, 2), dtype=np.uint32))
    values = np.arange(150 * 10, dtype=np.float32).reshape(150, 10)
    # Source claims tall chunks/shards that clamp against shape on repack.
    assay.create_array(
        "embedding",
        data=values,
        chunks=(200, 10),
        shards=(200, 10),
    )

    repack_store(str(source), str(output))

    result = zarr.open_group(str(output), mode="r")["RNA/embedding"]
    np.testing.assert_array_equal(result[:], values)
    assert result.chunks == normalize_chunks((200, 10), values.shape)
    assert array_metadata_shards(result) == result.chunks


def test_main_keeps_remote_uris_and_storage_options(monkeypatch):
    captured: dict = {}

    def fake_repack(input_path, output_path, **kwargs):
        captured["input"] = input_path
        captured["output"] = output_path
        captured["storage_options"] = kwargs.get("storage_options")
        captured["profile"] = kwargs.get("profile")
        captured["mem_budget"] = kwargs.get("mem_budget")
        captured["nthreads"] = kwargs.get("nthreads")

    monkeypatch.setattr(repack_module, "repack_store", fake_repack)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "repack_zarr",
            "s3://bucket/in.zarr",
            "s3://bucket/out.zarr",
            "--profile",
            "cloud",
            "--mem-budget",
            "8G",
            "--nthreads",
            "4",
            "--storage-options",
            '{"skip_signature": true}',
        ],
    )
    repack_module.main()
    assert captured["input"] == "s3://bucket/in.zarr"
    assert captured["output"] == "s3://bucket/out.zarr"
    assert captured["profile"] == "cloud"
    assert captured["mem_budget"] == "8G"
    assert captured["nthreads"] == 4
    assert captured["storage_options"] == {"skip_signature": True}

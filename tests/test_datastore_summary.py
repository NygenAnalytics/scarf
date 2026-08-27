import json
import shutil
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
import zarr
from scipy.sparse import csr_matrix

import scarf
from scarf.datastore.datastore import DataStore
from scarf.metadata import MetaData
from scarf.storage.artifact_writer import (
    finish_artifact,
    plan_artifact,
    start_artifact,
)
from scarf.storage.pipeline_runs import (
    PipelineOutputRecord,
    PipelineStageMetrics,
    PipelineStageOutputRecord,
    complete_pipeline_run_record,
    create_pipeline_run_record,
    fail_pipeline_run_record,
    finish_pipeline_stage_record,
    start_pipeline_stage_record,
)
from scarf.storage.refs import ArtifactRef, ArtifactScope
from scarf.writers import SparseToZarr


@pytest.fixture
def summary_datastore(toy_crdir_writer: str, tmp_path: Path) -> DataStore:
    location = tmp_path / "summary.zarr"
    shutil.copytree(toy_crdir_writer, location)
    return DataStore(
        str(location),
        default_assay="RNA",
        nthreads=3,
        mem_budget=64 * 1024 * 1024,
    )


def _create_minimal_datastore(location: Path) -> DataStore:
    writer = SparseToZarr(
        csr_matrix(np.array([[1, 0], [0, 2]], dtype=np.uint16)),
        str(location),
        cell_ids=["cell-1", "cell-2"],
        feature_ids=["gene-1", "gene-2"],
        mem_budget=64 * 1024 * 1024,
        nthreads=1,
    )
    writer.dump()
    return DataStore(
        str(location),
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=1,
        mem_budget=64 * 1024 * 1024,
    )


def _add_artifact(
    datastore: DataStore,
    *,
    scope: ArtifactScope,
    kind: str,
    operation: str,
    assay: str | None = None,
    parameter: int,
    complete: bool,
) -> ArtifactRef:
    planned = plan_artifact(
        datastore.zw,
        scope=scope,
        assay=assay,
        kind=kind,
        operation=operation,
        parameters={"branch": parameter},
        inputs={},
        execution_options={},
    )
    group = start_artifact(datastore.zw, planned)
    if complete:
        finish_artifact(group, planned)
    return planned.ref


def _file_snapshot(location: Path) -> dict[str, tuple[int, int]]:
    return {
        str(path.relative_to(location)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in location.rglob("*")
        if path.is_file()
    }


def _metrics() -> PipelineStageMetrics:
    return PipelineStageMetrics(
        wall_seconds=0.01,
        rss_baseline_bytes=None,
        rss_peak_bytes=None,
        rss_incremental_peak_bytes=None,
        sample_interval_seconds=0.1,
        sample_count=0,
        sampling_error_count=0,
        rss_unavailable_reason="test",
    )


def test_summary_of_minimal_store_is_frozen_and_metadata_only(tmp_path: Path) -> None:
    datastore = _create_minimal_datastore(tmp_path / "minimal.zarr")

    summary = datastore.summary()

    assert isinstance(summary, scarf.DataStoreSummary)
    assert summary.zarr_mode == "r+"
    assert summary.workspace is None
    assert summary.default_assay == "RNA"
    assert summary.scarf_version == scarf.__version__
    assert summary.resources.memory_bytes == 64 * 1024 * 1024
    assert summary.resources.workers == 1
    assert summary.resources.storage_profile == "fast_local"
    assert summary.total_cells == 2
    assert summary.active_cells == 2
    assert summary.cell_columns == tuple(sorted(summary.cell_columns))
    assert len(summary.assays) == 1
    assert summary.assays[0].name == "RNA"
    assert summary.assays[0].assay_type == "RNA"
    assert summary.assays[0].total_features == 2
    assert summary.assays[0].active_features == 2
    assert not hasattr(summary.assays[0], "state")
    assert summary.artifacts == ()
    assert summary.pipeline_run_counts == {
        "total": 0,
        "running": 0,
        "completed": 0,
        "failed": 0,
        "interrupted": 0,
        "incomplete": 0,
    }
    assert summary.labeled_pipeline_runs == ()
    assert not hasattr(summary, "__dict__")
    with pytest.raises(FrozenInstanceError):
        summary.total_cells = 3  # type: ignore[misc]


def test_summary_discovers_runs_and_labels(
    summary_datastore: DataStore,
) -> None:
    artifact = _add_artifact(
        summary_datastore,
        scope="datastore",
        kind="quality_metric",
        operation="summary_root",
        parameter=1,
        complete=True,
    )
    completed = create_pipeline_run_record(
        summary_datastore.zw,
        recipe="basic_rna_analysis",
        requested_label="baseline",
        assay="RNA",
        config={},
        stage_order=("snapshot",),
        scarf_version=scarf.__version__,
        run_id="a" * 64,
        started_at_ns=100,
    )
    start_pipeline_stage_record(
        summary_datastore.zw,
        run_id=completed.run_id,
        ordinal=0,
        stage="snapshot",
        started_at_ns=110,
    )
    finish_pipeline_stage_record(
        summary_datastore.zw,
        run_id=completed.run_id,
        ordinal=0,
        status="completed",
        outputs=(PipelineStageOutputRecord("root", artifact, False),),
        metrics=_metrics(),
        finished_at_ns=120,
    )
    complete_pipeline_run_record(
        summary_datastore.zw,
        run_id=completed.run_id,
        outputs=(PipelineOutputRecord("root", artifact),),
        fields=(),
        finished_at_ns=130,
    )
    failed = create_pipeline_run_record(
        summary_datastore.zw,
        recipe="basic_rna_analysis",
        requested_label="failed-attempt",
        assay="RNA",
        config={},
        stage_order=("snapshot",),
        scarf_version=scarf.__version__,
        run_id="b" * 64,
        started_at_ns=200,
    )
    fail_pipeline_run_record(
        summary_datastore.zw,
        run_id=failed.run_id,
        error=ValueError("deliberate failure"),
        finished_at_ns=210,
    )
    create_pipeline_run_record(
        summary_datastore.zw,
        recipe="basic_rna_analysis",
        requested_label="interrupted",
        assay="RNA",
        config={},
        stage_order=("snapshot",),
        scarf_version=scarf.__version__,
        run_id="c" * 64,
        started_at_ns=300,
    )
    summary = summary_datastore.summary()

    assert summary.pipeline_run_counts == {
        "total": 3,
        "running": 1,
        "completed": 1,
        "failed": 1,
        "interrupted": 0,
        "incomplete": 1,
    }
    assert summary.labeled_pipeline_runs == (("baseline", completed.run_id),)
    assert summary.to_dict()["labeled_pipeline_runs"] == [
        {"label": "baseline", "run_id": completed.run_id}
    ]


def test_summary_counts_selections_blockwise(
    summary_datastore: DataStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_cells = sum(
        block.active_global_indices.size
        for block in summary_datastore.cells.iter_row_blocks(cell_key="I")
    )
    expected_features = {
        assay_name: sum(
            block.active_global_indices.size
            for block in summary_datastore._get_assay(assay_name).feats.iter_row_blocks(
                cell_key="I"
            )
        )
        for assay_name in summary_datastore.assay_names
    }

    def fail_materialization(*args: object, **kwargs: object) -> None:
        raise AssertionError("summary materialized a complete metadata column")

    monkeypatch.setattr(MetaData, "active_index", fail_materialization)
    monkeypatch.setattr(MetaData, "fetch_all", fail_materialization)

    summary = summary_datastore.summary()

    assert summary.active_cells == expected_cells
    assert {
        assay.name: assay.active_features for assay in summary.assays
    } == expected_features


def test_summary_groups_multiple_assays_and_artifact_branches(
    summary_datastore: DataStore,
) -> None:
    complete_normalized = _add_artifact(
        summary_datastore,
        scope="assay",
        assay="RNA",
        kind="normalized",
        operation="run_normalization",
        parameter=2,
        complete=True,
    )
    alternate_normalized = _add_artifact(
        summary_datastore,
        scope="assay",
        assay="RNA",
        kind="normalized",
        operation="run_normalization",
        parameter=1,
        complete=True,
    )
    incomplete_reduction = _add_artifact(
        summary_datastore,
        scope="assay",
        assay="RNA",
        kind="reduction",
        operation="run_pca",
        parameter=1,
        complete=False,
    )
    integrated = _add_artifact(
        summary_datastore,
        scope="datastore",
        kind="integrated_graph",
        operation="integrate_assays",
        parameter=1,
        complete=True,
    )

    summary = summary_datastore.summary()
    rna = next(assay for assay in summary.assays if assay.name == "RNA")

    assert tuple(assay.name for assay in summary.assays) == tuple(
        sorted(summary_datastore.assay_names)
    )
    assert {assay.name for assay in summary.assays} == {"ADT", "HTO", "RNA"}
    assert {artifact.ref for artifact in rna.artifacts} == {
        complete_normalized,
        alternate_normalized,
        incomplete_reduction,
    }
    assert tuple(
        (artifact.ref.kind, artifact.ref.artifact_id) for artifact in rna.artifacts
    ) == tuple(
        sorted(
            (artifact.ref.kind, artifact.ref.artifact_id) for artifact in rna.artifacts
        )
    )
    assert (
        next(
            artifact
            for artifact in rna.artifacts
            if artifact.ref == incomplete_reduction
        ).complete
        is False
    )
    assert all(
        artifact.operation == "run_normalization"
        for artifact in rna.artifacts
        if artifact.ref.kind == "normalized"
    )
    assert tuple(artifact.ref for artifact in summary.artifacts) == (integrated,)
    assert summary.artifacts[0].operation == "integrate_assays"
    assert all(
        artifact.ref.scope == "assay"
        for assay in summary.assays
        for artifact in assay.artifacts
    )
    assert all(artifact.ref.scope == "datastore" for artifact in summary.artifacts)


def test_summary_to_dict_is_deterministic_json_and_omits_locations(
    summary_datastore: DataStore,
) -> None:
    first = summary_datastore.summary().to_dict()
    second = summary_datastore.summary().to_dict()

    assert first == second
    serialized = json.dumps(first, allow_nan=False, sort_keys=True)
    assert str(summary_datastore.zarr_loc) not in serialized
    assert "zarr_loc" not in serialized
    assert "storage_options" not in serialized
    assert "credentials" not in serialized
    assert "private_bucket" not in serialized


def test_summary_does_not_mutate_read_only_store(
    tmp_path: Path,
) -> None:
    location = tmp_path / "read-only.zarr"
    _create_minimal_datastore(location)
    root = zarr.open_group(str(location), mode="r+")
    root["RNA"].attrs.pop("dataset_fingerprint", None)
    del root

    datastore = DataStore(
        str(location),
        default_assay="RNA",
        zarr_mode="r",
        nthreads=1,
        mem_budget=64 * 1024 * 1024,
    )
    before = _file_snapshot(location)

    summary = datastore.summary()

    assert _file_snapshot(location) == before
    assert all(assay.dataset_fingerprint is None for assay in summary.assays)
    reopened = zarr.open_group(str(location), mode="r")
    assert "dataset_fingerprint" not in reopened["RNA"].attrs


def test_summarize_zarr_readonly_does_not_write_default_assay(tmp_path: Path) -> None:
    from scarf.datastore.summary import summarize_zarr_readonly

    location = tmp_path / "readonly-default.zarr"
    datastore = _create_minimal_datastore(location)
    root = zarr.open_group(str(location), mode="r+")
    root.attrs.pop("defaultAssay", None)
    del root
    before = _file_snapshot(location)

    summary = summarize_zarr_readonly(str(location), default_assay=None)
    assert summary.default_assay == "RNA"
    assert _file_snapshot(location) == before
    reopened = zarr.open_group(str(location), mode="r")
    assert "defaultAssay" not in reopened.attrs

    # Explicit direction also must not persist.
    summary2 = summarize_zarr_readonly(str(location), default_assay="RNA")
    assert summary2.default_assay == "RNA"
    assert _file_snapshot(location) == before
    del datastore


def test_summarize_zarr_readonly_supports_workspace(tmp_path: Path) -> None:
    from scarf.datastore.summary import summarize_zarr_readonly

    location = tmp_path / "readonly-workspace.zarr"
    writer = SparseToZarr(
        csr_matrix(np.array([[1, 0], [0, 2]], dtype=np.uint16)),
        str(location),
        cell_ids=["cell-1", "cell-2"],
        feature_ids=["gene-1", "gene-2"],
        workspace="analysis",
        mem_budget=64 * 1024 * 1024,
        nthreads=1,
    )
    writer.dump()
    before = _file_snapshot(location)

    summary = summarize_zarr_readonly(str(location), workspace="analysis")

    assert summary.workspace == "analysis"
    assert summary.default_assay == "RNA"
    assert summary.total_cells == 2
    assert [assay.name for assay in summary.assays] == ["RNA"]
    assert summary.assays[0].total_features == 2
    assert _file_snapshot(location) == before

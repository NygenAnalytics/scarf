import os
import select
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from scarf import DataStore
from scarf.storage.pipeline_runs import (
    list_pipeline_run_records,
    load_pipeline_stage_records,
)


_CHILD = textwrap.dedent(
    """
    import sys
    import time

    from scarf import DataStore
    from scarf.utils.shutdown import shutdown_checkpoint

    store = DataStore(sys.argv[1], default_assay="RNA")
    cooperative = sys.argv[2] == "cooperative"

    def paused_pca(self, *_args, **_kwargs):
        print("READY", flush=True)
        while True:
            if cooperative:
                shutdown_checkpoint()
            time.sleep(0.01)

    type(store).run_pca = paused_pca
    store.pipeline.run(
        filtering=False,
        cell_cycle=False,
        hvg_count=50,
        pca_dims=3,
        neighbors_k=3,
        umap=False,
        leiden=False,
        paris=False,
        doublets=False,
        markers=False,
    )
    """
)


_SIGNAL_RACE_CHILD = textwrap.dedent(
    """
    import os
    import signal
    import sys
    import time

    from scarf import DataStore
    import scarf.datastore.pipeline_accessor as pipeline_accessor
    from scarf.storage.pipeline_runs import (
        PipelineStageMetrics,
        complete_pipeline_run_record,
        create_pipeline_run_record,
        fail_pipeline_run_record,
        finish_pipeline_stage_record,
        start_pipeline_stage_record,
    )
    from scarf.utils.shutdown import current_shutdown_token

    store = DataStore(sys.argv[1], default_assay="RNA")
    case = sys.argv[2]

    def metrics():
        return PipelineStageMetrics(
            wall_seconds=0.0,
            rss_baseline_bytes=None,
            rss_peak_bytes=None,
            rss_incremental_peak_bytes=None,
            sample_interval_seconds=0.1,
            sample_count=0,
            sampling_error_count=0,
            rss_unavailable_reason="not sampled",
        )

    def execute_with_signal_race(
        self,
        recipe,
        _callback,
        *,
        active_run_id,
        **_kwargs,
    ):
        stage = "pca" if case == "operation_failure" else "handoff"
        record = create_pipeline_run_record(
            self._store.zw,
            recipe="signal_regression",
            requested_label=None,
            assay=recipe.assay,
            config={"case": case},
            stage_order=(stage,),
            scarf_version="test",
        )
        active_run_id.append(record.run_id)
        start_pipeline_stage_record(
            self._store.zw,
            run_id=record.run_id,
            ordinal=0,
            stage=stage,
        )
        if case == "operation_failure":
            print("READY", flush=True)
            while True:
                token = current_shutdown_token()
                if token is not None and token.requested:
                    error = RuntimeError("PCA failed after SIGTERM")
                    finish_pipeline_stage_record(
                        self._store.zw,
                        run_id=record.run_id,
                        ordinal=0,
                        status="failed",
                        metrics=metrics(),
                        error=error,
                    )
                    fail_pipeline_run_record(
                        self._store.zw,
                        run_id=record.run_id,
                        error=error,
                    )
                    raise error
                time.sleep(0.01)
        finish_pipeline_stage_record(
            self._store.zw,
            run_id=record.run_id,
            ordinal=0,
            status="skipped",
            metrics=metrics(),
        )
        complete_pipeline_run_record(
            self._store.zw,
            run_id=record.run_id,
            outputs=(),
            fields=(),
        )
        result = pipeline_accessor.open_pipeline_run(
            self._store,
            run_id=record.run_id,
        )
        os.kill(os.getpid(), signal.SIGTERM)
        return result

    pipeline_accessor.PipelineAccessor._execute_recipe = execute_with_signal_race
    store.pipeline.run(
        filtering=False,
        cell_cycle=False,
        hvg_count=50,
        pca_dims=3,
        neighbors_k=3,
        umap=False,
        leiden=False,
        paris=False,
        doublets=False,
        markers=False,
    )
    """
)


def _child_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "2",
            "NUMEXPR_MAX_THREADS": "1",
        }
    )
    return environment


def _start_child(path: Path, child: str, *arguments: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-c", child, str(path), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_child_environment(),
    )


def _wait_for_stdout(process: subprocess.Popen[str], marker: str) -> None:
    assert process.stdout is not None
    observed: list[str] = []
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        ready, _, _ = select.select(
            [process.stdout],
            [],
            [],
            max(0.0, deadline - time.monotonic()),
        )
        if not ready:
            break
        line = process.stdout.readline().strip()
        if line == marker:
            return
        if line:
            observed.append(line)
        if process.poll() is not None:
            break
    process.kill()
    _stdout, stderr = process.communicate(timeout=10)
    pytest.fail(
        f"Pipeline child did not emit {marker!r}: "
        + " | ".join(observed[-3:])
        + f"\n{stderr}"
    )


def _communicate(
    process: subprocess.Popen[str],
    *,
    timeout: float,
) -> tuple[str, str]:
    try:
        return process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate(timeout=10)
        pytest.fail(f"Pipeline child timed out:\n{stdout}\n{stderr}")


def _paused_pipeline(path: Path, *, cooperative: bool) -> subprocess.Popen[str]:
    process = _start_child(
        path,
        _CHILD,
        "cooperative" if cooperative else "blocked",
    )
    _wait_for_stdout(process, "READY")
    return process


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal contract")
def test_sigterm_commits_interrupted_pipeline_before_propagation(
    datastore_zarr_root: str,
    tmp_path: Path,
) -> None:
    location = tmp_path / "sigterm.zarr"
    shutil.copytree(datastore_zarr_root, location)
    process = _paused_pipeline(location, cooperative=True)

    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=30) == -signal.SIGTERM
    root = DataStore(str(location), default_assay="RNA").zw
    runs = list_pipeline_run_records(root, limit=100)
    assert len(runs) == 1
    assert runs[0].status == "interrupted"
    assert runs[0].complete
    assert runs[0].interruption is not None
    stages = load_pipeline_stage_records(root, runs[0].run_id)
    assert stages[-1].stage == "pca"
    assert stages[-1].status == "interrupted"
    assert stages[-1].complete


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal contract")
def test_pending_sigterm_propagates_after_operation_failure(
    datastore_zarr_root: str,
    tmp_path: Path,
) -> None:
    location = tmp_path / "sigterm-operation-failure.zarr"
    shutil.copytree(datastore_zarr_root, location)
    process = _start_child(location, _SIGNAL_RACE_CHILD, "operation_failure")
    _wait_for_stdout(process, "READY")

    process.send_signal(signal.SIGTERM)
    _stdout, stderr = _communicate(process, timeout=30)
    returncode = process.returncode

    root = DataStore(str(location), default_assay="RNA").zw
    runs = list_pipeline_run_records(root, limit=100)
    assert len(runs) == 1
    assert runs[0].status == "failed"
    assert runs[0].complete
    assert runs[0].error is not None
    assert runs[0].error.type == "RuntimeError"
    stages = load_pipeline_stage_records(root, runs[0].run_id)
    assert stages[-1].stage == "pca"
    assert stages[-1].status == "failed"
    assert stages[-1].complete
    assert stages[-1].error is not None
    assert stages[-1].error.type == "RuntimeError"
    assert returncode == -signal.SIGTERM, stderr


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal contract")
def test_pending_sigterm_propagates_after_completed_run_handoff(
    datastore_zarr_root: str,
    tmp_path: Path,
) -> None:
    location = tmp_path / "sigterm-completed-handoff.zarr"
    shutil.copytree(datastore_zarr_root, location)
    process = _start_child(location, _SIGNAL_RACE_CHILD, "completed_handoff")

    _stdout, stderr = _communicate(process, timeout=60)
    returncode = process.returncode

    root = DataStore(str(location), default_assay="RNA").zw
    runs = list_pipeline_run_records(root, limit=100)
    assert len(runs) == 1
    assert runs[0].status == "completed"
    assert runs[0].complete
    stages = load_pipeline_stage_records(root, runs[0].run_id)
    assert stages
    assert all(stage.complete for stage in stages)
    assert returncode == -signal.SIGTERM, stderr


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal contract")
def test_sigkill_leaves_incomplete_run_and_a_new_run_reuses_only_complete_outputs(
    datastore_zarr_root: str,
    tmp_path: Path,
) -> None:
    location = tmp_path / "sigkill.zarr"
    shutil.copytree(datastore_zarr_root, location)
    process = _paused_pipeline(location, cooperative=False)

    process.send_signal(signal.SIGKILL)
    assert process.wait(timeout=30) == -signal.SIGKILL
    store = DataStore(str(location), default_assay="RNA")
    incomplete = list_pipeline_run_records(store.zw, limit=100)
    assert len(incomplete) == 1
    assert incomplete[0].status == "running"
    assert not incomplete[0].complete
    assert load_pipeline_stage_records(store.zw, incomplete[0].run_id)[-1].status == (
        "running"
    )

    recovered = store.pipeline.run(
        filtering=False,
        cell_cycle=False,
        hvg_count=50,
        pca_dims=3,
        neighbors_k=3,
        umap=False,
        leiden=False,
        paris=False,
        doublets=False,
        markers=False,
    )
    assert recovered.status == "completed"
    report = recovered.report()
    receipts = [plan for stage in report["stages"] for plan in stage["plans"]]
    assert any(plan["disposition"] == "reused" for plan in receipts)
    assert all(
        store.inspect_artifact(plan_ref).complete for plan_ref in recovered.values()
    )

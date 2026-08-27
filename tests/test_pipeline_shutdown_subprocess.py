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


def _paused_pipeline(path: Path, *, cooperative: bool) -> subprocess.Popen[str]:
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
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _CHILD,
            str(path),
            "cooperative" if cooperative else "blocked",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
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
        if line == "READY":
            return process
        if line:
            observed.append(line)
        if process.poll() is not None:
            break
    process.kill()
    _stdout, stderr = process.communicate(timeout=10)
    pytest.fail(
        "Pipeline child did not reach PCA: " + " | ".join(observed[-3:]) + f"\n{stderr}"
    )


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

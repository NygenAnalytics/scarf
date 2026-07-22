import numpy as np
import zarr

from profiling.config import StageResources
from profiling.stages import repair_counts_t
from scarf.storage.sharding import write_counts_t


def test_repair_counts_t_rewrites_incomplete_array(tmp_path):
    root_path = tmp_path / "store.zarr"
    root = zarr.open_group(str(root_path), mode="w")
    group = root.create_group("RNA")
    values = np.arange(24, dtype=np.uint32).reshape(6, 4)
    counts = group.create_array("counts", shape=values.shape, dtype=values.dtype)
    counts[:] = values
    counts_t = write_counts_t(counts, group)
    assert counts_t is not None
    counts_t.attrs["complete"] = False
    counts_t[0, 0] = 999

    resources = StageResources(
        modalMemoryRequestMb=4096,
        modalMemoryLimitMb=4096,
        modalCpuRequest=1.0,
        modalCpuLimit=1.0,
        scarfMemoryBudget=2 * 1024**3,
        workers=1,
        workingCopies=1,
        timeoutSeconds=600,
        ephemeralDiskMb=524288,
    )
    result = repair_counts_t(
        storeUri=str(root_path),
        assayName="RNA",
        resources=resources,
        nCheckTiles=2,
        seed=1,
    )
    assert result["status"] == "ok"
    assert result["complete"] is True
    assert result["beforeComplete"] is False
    reopened = zarr.open_group(str(root_path), mode="r")
    fixed = reopened["RNA/countsT"]
    assert fixed.attrs["complete"] is True
    np.testing.assert_array_equal(fixed[:], values.T)

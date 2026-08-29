"""Reusable Zarr store builder for automated-orchestrator tests."""

from pathlib import Path

import numpy as np
import zarr

from scarf.storage.budget import ResourceBudget
from scarf.storage.schema import create_cell_data, create_zarr_count_assay
from scarf.storage.sharding import write_counts_t


def create_store(path: Path, *, workspace: str | None = None) -> Path:
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    values = np.asarray(
        [
            [4, 0, 1, 0],
            [0, 3, 0, 2],
            [2, 1, 0, 0],
            [0, 0, 5, 1],
        ],
        dtype=np.uint32,
    )
    cell_ids = np.asarray([f"cell-{index}" for index in range(values.shape[0])])
    feature_ids = np.asarray([f"feature-{index}" for index in range(values.shape[1])])
    feature_names = np.asarray(["MT-CO1", "RPS3", "GENE1", "GENE2"])
    create_cell_data(
        root,
        workspace,
        ids=cell_ids,
        names=cell_ids,
        profile="fast_local",
    )
    counts = create_zarr_count_assay(
        root,
        "RNA",
        workspace,
        values.shape[0],
        feat_ids=feature_ids,
        feat_names=feature_names,
        dtype="uint32",
        profile="fast_local",
    )
    counts[:] = values
    count_group = root["RNA"] if workspace is None else root["matrices/RNA"]
    write_counts_t(
        counts,
        count_group,
        resources=ResourceBudget(1024**3, 2),
    )
    active = root if workspace is None else root[workspace]
    active.attrs["assayTypes"] = {"RNA": "RNA"}
    active["RNA"].attrs["dataset_fingerprint"] = "dataset-rna"
    return path

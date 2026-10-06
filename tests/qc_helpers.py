"""Quality-control stores and NumPy reference rules shared by QC tests.

Tests import these from here rather than from each other, so a change
to one test module cannot break another.
"""

from functools import cache
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import zarr

from scarf import DataStore
from scarf.storage.count_matrix import CountMatrixPolicy
from scarf.storage.schema import create_cell_data, create_zarr_count_assay
from scarf.writers.counts_t import finalize_writer_counts_t
from tests.storage_helpers import (
    finalize_test_counts,
    write_count_store,
)
from tests.store_probes import RecordingStore


QC_VALUES = np.array(
    [
        [5, 0, 1, 0, 0, 2],
        [0, 3, 0, 4, 0, 0],
        [1, 2, 0, 0, 0, 0],
        [0, 0, 0, 5, 0, 1],
        [0, 0, 0, 0, 0, 0],
        [2, 1, 3, 1, 0, 0],
    ],
    dtype=np.uint32,
)

QC_FEATURE_NAMES = np.array(["MT-CO1", "RPS3", "GENE_A", "RPL5", "ZERO", "GENE_B"])


@cache
def _qc_store_template() -> tuple[dict[str, Any], int]:
    """Write the fresh, unprepared QC import once per process."""
    store = RecordingStore()
    root = zarr.open_group(store=store, mode="w")
    n_cells, n_features = QC_VALUES.shape
    create_cell_data(
        root,
        None,
        ids=np.array([f"c{i}" for i in range(n_cells)]),
        names=np.array([f"c{i}" for i in range(n_cells)]),
        profile="fast_local",
    )
    counts = create_zarr_count_assay(
        root,
        "RNA",
        None,
        n_cells,
        feat_ids=np.array([f"f{i}" for i in range(n_features)]),
        feat_names=QC_FEATURE_NAMES,
        dtype="uint32",
        profile="fast_local",
        policy=CountMatrixPolicy(unitBytes=48, chunkBytes=16),
    )
    counts[:] = QC_VALUES
    assert counts.shards is not None
    from tests.storage_helpers import finalize_test_counts
    from scarf.writers.counts_t import finalize_writer_counts_t

    finalize_test_counts(counts)

    finalize_writer_counts_t(root, "RNA", None, profile="fast_local")
    expected_reads = int(np.ceil(n_cells / counts.shards[0]))
    return dict(store._store_dict), expected_reads


def fresh_qc_store() -> tuple[RecordingStore, int]:
    """Return an independent copy of the fresh QC import with empty records.

    Memory stores replace whole values on write, so copies share no state.
    """
    contents, expected_reads = _qc_store_template()
    return RecordingStore(dict(contents)), expected_reads


def open_qc_store(store: RecordingStore, **overrides) -> DataStore:
    options = {
        "default_assay": "RNA",
        "min_features_per_cell": 0,
        "mito_pattern": "^MT-",
        "ribo_pattern": "^(RPS|RPL)",
        "nthreads": 1,
        "zarrProfile": "fast_local",
    }
    options.update(overrides)
    return DataStore(store, **options)


def create_labelled_qc_store(path):
    root = zarr.open_group(str(path), mode="w")
    create_cell_data(
        root,
        None,
        np.array([f"c{i}" for i in range(6)]),
        np.array([f"c{i}" for i in range(6)]),
    )
    counts = create_zarr_count_assay(
        root,
        "RNA",
        None,
        6,
        feat_ids=np.array([f"f{i}" for i in range(6)]),
        feat_names=QC_FEATURE_NAMES,
        dtype="uint32",
        policy=CountMatrixPolicy(unitBytes=48, chunkBytes=16),
    )
    counts[:] = QC_VALUES
    finalize_test_counts(counts)
    finalize_writer_counts_t(root, "RNA", None)
    store = DataStore(
        str(path), default_assay="RNA", min_features_per_cell=0, nthreads=1
    )
    # insert flags the None labels in the linked mask __scarf_missing__label.
    store.cells.insert(
        "label", np.array(["a", None, "b", "a", None, "b"], dtype=object)
    )
    return store


_COUNT_METRICS = ("nCounts", "nFeatures")

_PERCENT_METRICS = ("percentMito", "percentRibo")


def _is_metric(attr: str, names: tuple[str, ...]) -> bool:
    return any(attr == name or attr.endswith(f"_{name}") for name in names)


def reference_mad_bounds(
    values: np.ndarray,
    attr: str,
    n_mads: float = 3.0,
) -> tuple[float | None, float] | None:
    """Return the raw-scale MAD bounds of one metric, or None for a zero MAD.

    Counts are bounded on the log1p scale and percentages from above only, as
    the filtering documentation describes.
    """
    raw = np.asarray(values, dtype=float)
    counts = _is_metric(attr, _COUNT_METRICS)
    work = np.log1p(raw) if counts else raw
    center = np.median(work)
    spread = 1.4826 * np.median(np.abs(work - center))
    if spread == 0:
        return None
    low, high = center - n_mads * spread, center + n_mads * spread
    if counts:
        return max(0.0, float(np.expm1(low))), max(0.0, float(np.expm1(high)))
    if _is_metric(attr, _PERCENT_METRICS):
        return None, min(100.0, max(0.0, float(high)))
    return float(low), float(high)


def reference_mad_keep(
    values_by_attr: dict[str, np.ndarray],
    labels: np.ndarray | None = None,
    *,
    n_mads: float = 3.0,
    min_cells: int = 20,
) -> np.ndarray:
    """Keep cells strictly inside every MAD bound of their sample."""
    n_cells = len(next(iter(values_by_attr.values())))
    labels = np.zeros(n_cells, dtype=int) if labels is None else np.asarray(labels)
    keep = np.ones(n_cells, dtype=bool)
    for label in dict.fromkeys(labels.tolist()):
        rows = np.flatnonzero(labels == label)
        if len(rows) < min_cells:
            continue
        for attr, values in values_by_attr.items():
            sample = np.asarray(values, dtype=float)[rows]
            bounds = reference_mad_bounds(sample, attr, n_mads)
            if bounds is None:
                continue
            low, high = bounds
            inside = sample < high
            if low is not None:
                inside &= sample > low
            keep[rows] &= inside
    return keep


def reference_gaussian_bounds(
    values: np.ndarray,
    min_p: float = 0.01,
    max_p: float = 0.99,
) -> tuple[float, float]:
    """Return median-centered normal quantiles with the population deviation."""
    values = np.asarray(values, dtype=float)
    center, spread = float(np.median(values)), float(np.std(values))
    if spread == 0:
        return center, center
    normal = NormalDist(center, spread)
    return normal.inv_cdf(min_p), normal.inv_cdf(max_p)


N_CELLS = 96
N_FEATURES = 80


def small_rna_counts() -> np.ndarray:
    """Counts whose first cell is nearly empty and whose second is very deep."""
    rng = np.random.default_rng(11)
    # Most genes are detected in a quarter to three quarters of the cells, so
    # the pipeline finds variable genes that are neither rare nor ubiquitous.
    means = rng.uniform(0.2, 2.0, size=N_FEATURES)
    counts = rng.poisson(means, size=(N_CELLS, N_FEATURES))
    counts[0] = 0
    counts[0, :2] = 1
    counts[1] *= 6
    return counts.astype(np.uint16)


def open_small_store(path: Path) -> DataStore:
    return DataStore(
        str(path), default_assay="RNA", min_features_per_cell=0, nthreads=1
    )


def write_small_store(path: Path) -> None:
    """Write the small RNA store and its cell QC metrics."""
    write_count_store(str(path), {"RNA": small_rna_counts()}, "uint16")
    # Opening the store once writes its cell QC metrics.
    open_small_store(path)

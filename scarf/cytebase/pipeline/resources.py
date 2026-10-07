"""Resource tiers for dataset workers, the tier each dataset starts in, and refusals."""

import math
from dataclasses import dataclass

from .models import DatasetRecord


@dataclass(frozen=True)
class ProcessResources:
    cpu: int
    memoryMiB: int
    memBudget: str


PROCESS_RESOURCES = (
    ProcessResources(cpu=4, memoryMiB=16_384, memBudget="12G"),
    ProcessResources(cpu=8, memoryMiB=32_768, memBudget="24G"),
    ProcessResources(cpu=16, memoryMiB=65_536, memBudget="48G"),
)

# CELLxGENE stores counts as float32, and most count matrices hold values above
# 255 and below 65,536, so they import as uint16. A uint8 import packs twice the
# rows into each count band and needs more memory; its refusal moves the
# dataset up a tier.
_SOURCE_DTYPE = "float32"
_STORAGE_DTYPE = "uint16"
# A refused attempt repeats the download and the full value scan. Above this
# source size that costs more than a larger container, so the estimate assumes
# the densest count band holds LARGE_SOURCE_MARGIN times the mean genes per cell.
LARGE_SOURCE_BYTES = 2_000_000_000
LARGE_SOURCE_MARGIN = 1.5


class ImportMemoryRefusal(MemoryError):
    """Default-layout admission refused before creating the local store."""


def default_layout_fits(
    memBudget: int | str, *, nCells: int, nFeatures: int, windowGenesPerCell: float
) -> bool:
    """Return whether a CSR H5AD import admits the default count layout.

    This runs the admission of ``scarf.H5adToZarr`` for one RNA assay without
    reading a file: every count band holds ``windowGenesPerCell`` values per
    row, counts import as uint16 from float32, and row pointers are int64.

    Args:
        memBudget: The import's ``mem_budget``, in bytes or as a size such as
            ``"12G"``.
        nCells: Number of cells.
        nFeatures: Number of features.
        windowGenesPerCell: Mean stored values per cell in the densest band.
    """
    import numpy as np

    from scarf.storage.budget import resolve_budget
    from scarf.storage.count_matrix import DEFAULT_COUNT_MATRIX_POLICY
    from scarf.storage.identity import CountSummary
    from scarf.storage.sharding import fit_count_layout, sparse_counts_admission
    from scarf.writers.counts_t import counts_t_assays

    pointer = np.dtype(np.int64).itemsize
    staging = 2 * pointer + np.dtype(np.int32).itemsize

    def window_values(rows: int) -> int:
        return math.ceil(min(rows, nCells) * windowGenesPerCell)

    def staging_bytes(rows: int) -> int:
        return (min(max(1, rows), nCells) + 1) * staging

    try:
        fit_count_layout(
            {"RNA": (nFeatures, np.dtype(_STORAGE_DTYPE))},
            nCells=nCells,
            profile="cloud",
            memoryBytes=resolve_budget(memBudget, 1).memoryBytes,
            transposed=counts_t_assays(("RNA",)),
            admitCounts=sparse_counts_admission(
                nRows=nCells,
                maxWindowNnz=window_values,
                sourceDtype=np.dtype(_SOURCE_DTYPE),
                residentBytes=(nCells + 1) * pointer
                + CountSummary.nbytes_for(nCells, nFeatures),
                producerStagingBytes=staging_bytes,
            ),
            requested=DEFAULT_COUNT_MATRIX_POLICY,
        )
    except MemoryError:
        return False
    return True


def tier_with_memory(memoryMiB: int) -> int:
    """Return the smallest tier with at least ``memoryMiB``, or the largest tier."""
    return next(
        (
            tier
            for tier, resources in enumerate(PROCESS_RESOURCES)
            if resources.memoryMiB >= memoryMiB
        ),
        len(PROCESS_RESOURCES) - 1,
    )


def initial_tier(record: DatasetRecord) -> int:
    """Return the tier of a dataset's first worker attempt.

    The smallest tier whose budget admits the estimated import is chosen; a
    record without its cell count, gene count, or mean genes per cell starts
    at tier 0, and an estimate above every tier starts at the largest. A store
    built for the latest version raises the start to the tier it was built in.
    """
    tier = 0
    n_cells, n_features, genes = (
        record.cellCount,
        record.nGenes,
        record.meanGenesPerCell,
    )
    if n_cells and n_features and genes:
        margin = (
            LARGE_SOURCE_MARGIN
            if (record.sourceBytes or 0) > LARGE_SOURCE_BYTES
            else 1.0
        )
        tier = next(
            (
                index
                for index, resources in enumerate(PROCESS_RESOURCES)
                if default_layout_fits(
                    resources.memBudget,
                    nCells=n_cells,
                    nFeatures=n_features,
                    windowGenesPerCell=genes * margin,
                )
            ),
            len(PROCESS_RESOURCES) - 1,
        )
    receipt = record.buildReceipt or {}
    built = receipt.get("resources")
    if (
        receipt.get("datasetVersionId") == str(record.latestVersionId)
        and isinstance(built, dict)
        and isinstance(built.get("memoryMiB"), int)
    ):
        tier = max(tier, tier_with_memory(built["memoryMiB"]))
    return tier

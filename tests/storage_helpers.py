"""Storage helpers for tests that build Scarf stores directly."""

import zarr

from scarf.storage import async_execution
from scarf.storage.identity import CountSummary, finalize_counts


def finalize_test_counts(counts: zarr.Array) -> str:
    """Summarize values already written to ``counts`` and finalize them.

    Writers fill the summary while they write. Tests that assign counts
    directly summarize the stored values here instead.
    """
    summary = CountSummary(counts)
    rows = max(1, int(counts.chunks[0]))
    for start in range(0, int(counts.shape[0]), rows):
        summary.update(start, counts[start : start + rows])
    return finalize_counts(counts, summary=summary)


def reset_zarr_runtime() -> None:
    """Restore Zarr's process defaults so the next storage call plans afresh."""
    zarr.config.set({"threading.max_workers": None, "async.concurrency": 10})
    async_execution._HOST_THREAD_CEILING = None

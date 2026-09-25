import zarr

from .count_matrix import require_count_matrix_layout
from .types import as_zarr_array
from .validation_scope import store_key, validated_once


def validate_count_matrix(
    matrix: zarr.Group, *, require_transpose: bool
) -> tuple[zarr.Array, zarr.Array | None]:
    # Finalized counts are immutable, so one operation validates them once.
    return validated_once(
        ("count_matrix", *store_key(matrix), require_transpose),
        lambda: _validate_count_matrix(matrix, require_transpose=require_transpose),
    )


def _validate_count_matrix(
    matrix: zarr.Group, *, require_transpose: bool
) -> tuple[zarr.Array, zarr.Array | None]:
    from .identity import REBUILD_REQUIRED, fresh_group, opened_count_fingerprint

    matrix = fresh_group(matrix)
    # Each lookup reads the child's metadata, so the arrays carry fresh attributes.
    try:
        counts = as_zarr_array(matrix["counts"], name="counts")
    except KeyError:
        raise ValueError(f"Raw counts are missing. {REBUILD_REQUIRED}") from None
    if counts.ndim != 2:
        raise ValueError(f"Raw counts must have two dimensions. {REBUILD_REQUIRED}")
    fingerprint = opened_count_fingerprint(counts)
    counts_t = None
    if require_transpose:
        try:
            counts_t = as_zarr_array(matrix["countsT"], name="countsT")
        except KeyError:
            raise ValueError(
                f"Required countsT matrix is missing. {REBUILD_REQUIRED}"
            ) from None
        if (
            counts_t.attrs.get("complete") is not True
            or counts_t.attrs.get("source_fingerprint") != fingerprint
            or counts_t.shape != counts.shape[::-1]
            or counts_t.dtype != counts.dtype
        ):
            raise ValueError(
                f"countsT is incomplete or does not match finalized counts. {REBUILD_REQUIRED}"
            )
    require_count_matrix_layout(matrix, counts, counts_t)
    return counts, counts_t

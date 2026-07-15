import hashlib
import os
import uuid
from collections.abc import Callable, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import h5py
import numpy as np

DEFAULT_SAMPLING_SEED = 0
DEFAULT_TARGET_SIZES = (
    10_000,
    25_000,
    50_000,
    100_000,
    250_000,
    500_000,
    1_000_000,
    2_500_000,
    5_000_000,
    10_000_000,
)

_MASK_64 = (1 << 64) - 1
_SPLITMIX_INCREMENT = 0x9E3779B97F4A7C15
_SPLITMIX_MULTIPLIER_1 = 0xBF58476D1CE4E5B9
_SPLITMIX_MULTIPLIER_2 = 0x94D049BB133111EB
_SAMPLING_DOMAIN = b"scarf-cellxgene-row-sampling-v1"
_SOURCE_ROWS_DIGEST_DOMAIN = b"scarf-ordered-source-rows-v1\0"
_DEFAULT_IO_CHUNK_BYTES = 16 * 1024 * 1024
_DEFAULT_ROW_BATCH_SIZE = 1024
_DEFAULT_COPY_BUFFER_BYTES = 64 * 1024 * 1024
_DEFAULT_INDPTR_CHUNK_ROWS = 1_000_000
_DEFAULT_LOAD_CHUNK_ELEMENTS = 8_388_608
_H5_DATA_CHUNK_BYTES = 4 * 1024 * 1024


def sha256_file(path: str | Path, *, chunkBytes: int = 16 * 1024 * 1024) -> str:
    if chunkBytes <= 0:
        raise ValueError("chunkBytes must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunkBytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SourceSpec:
    datasetId: str
    versionId: str
    url: str
    nRows: int
    nColumns: int
    nnz: int
    sourceBytes: int
    matrixKey: str = "X"
    matrixEncoding: str = "csr_matrix"
    rawCounts: bool = True
    cellIdsKey: str = "obs/_index"
    featureIdsKey: str = "var/_index"
    featureNameKey: str = "var/feature_name"
    dataDtype: str = "float32"
    indicesDtype: str = "int64"
    indptrDtype: str = "int64"


@dataclass(frozen=True, slots=True)
class DownloadResult:
    filePath: Path
    fileBytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class SourceValidation:
    nRows: int
    nColumns: int
    nnz: int
    finalRowNnz: int
    dataDtype: str
    indicesDtype: str
    indptrDtype: str


@dataclass(frozen=True, slots=True)
class H5adWriteResult:
    filePath: Path
    nRows: int
    nColumns: int
    nnz: int
    sourceRowsSha256: str
    finalSourceRow: int
    dataDtype: str
    indicesDtype: str
    indptrDtype: str


@dataclass(frozen=True, slots=True)
class InMemoryCsrSource:
    """CSR matrix plus labels held in process memory for fast row gathers."""

    data: np.ndarray
    indices: np.ndarray
    indptr: np.ndarray
    nRows: int
    nColumns: int
    dataDtype: np.dtype
    indicesDtype: np.dtype
    indptrDtype: np.dtype
    cellIds: np.ndarray
    featureIds: np.ndarray
    featureNames: np.ndarray

    @property
    def residentBytes(self) -> int:
        return int(
            self.data.nbytes
            + self.indices.nbytes
            + self.indptr.nbytes
            + self.cellIds.nbytes
            + self.featureIds.nbytes
            + self.featureNames.nbytes
        )


@dataclass(frozen=True, slots=True)
class PreparedValidation:
    fileBytes: int
    sha256: str
    sourceRowsSha256: str
    nRows: int
    nColumns: int
    nnz: int
    finalSourceRow: int


@dataclass(frozen=True, slots=True)
class PreparedArtifact:
    localPath: Path
    targetRows: int
    nColumns: int
    nnz: int
    fileBytes: int
    sha256: str
    sourceRowsSha256: str
    finalSourceRow: int
    dataDtype: str
    indicesDtype: str
    indptrDtype: str


@dataclass(frozen=True, slots=True)
class PreparationResult:
    sourcePath: Path
    sourceSha256: str
    sourceSpec: SourceSpec
    seed: int
    artifacts: tuple[PreparedArtifact, ...]


SOURCE_SPEC = SourceSpec(
    datasetId="dcfd4feb-18a3-4b30-81d7-1b0c544a8ab3",
    versionId="1bc30289-9565-4099-abf9-3326328c11ac",
    url=(
        "https://datasets.cellxgene.cziscience.com/"
        "1bc30289-9565-4099-abf9-3326328c11ac.h5ad"
    ),
    nRows=11_441_407,
    nColumns=45_525,
    nnz=19_516_755_155,
    sourceBytes=46_292_192_475,
)


def splitmix64(value: int) -> int:
    """Return SplitMix64 for one unsigned 64-bit input."""
    mixed = (value + _SPLITMIX_INCREMENT) & _MASK_64
    mixed = ((mixed ^ (mixed >> 30)) * _SPLITMIX_MULTIPLIER_1) & _MASK_64
    mixed = ((mixed ^ (mixed >> 27)) * _SPLITMIX_MULTIPLIER_2) & _MASK_64
    return mixed ^ (mixed >> 31)


def sampling_salt(
    seed: int,
    sourceVersion: str = SOURCE_SPEC.versionId,
) -> int:
    material = (
        _SAMPLING_DOMAIN + b"\0" + sourceVersion.encode() + b"\0" + str(seed).encode()
    )
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little")


def row_priorities(
    nRows: int,
    *,
    seed: int = DEFAULT_SAMPLING_SEED,
    sourceVersion: str = SOURCE_SPEC.versionId,
) -> np.ndarray:
    if nRows < 0:
        raise ValueError("nRows must be nonnegative")

    # For source row i:
    # priority(i) = SplitMix64((i + salt(sourceVersion, seed)) mod 2**64).
    # SplitMix64 is a fixed integer permutation, so this is reproducible across
    # Python and NumPy versions. Selection ranks (priority, source row index),
    # which makes the row index the explicit collision tie-break.
    values = np.arange(nRows, dtype=np.uint64)
    values += np.uint64(sampling_salt(seed, sourceVersion))
    values += np.uint64(_SPLITMIX_INCREMENT)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(_SPLITMIX_MULTIPLIER_1)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(_SPLITMIX_MULTIPLIER_2)
    return values ^ (values >> np.uint64(31))


def _normalize_targets(targetRows: Sequence[int], nRows: int) -> tuple[int, ...]:
    targets = tuple(int(value) for value in targetRows)
    if not targets:
        raise ValueError("At least one target row count is required")
    if any(value <= 0 for value in targets):
        raise ValueError("Target row counts must be positive")
    if tuple(sorted(set(targets))) != targets:
        raise ValueError("Target row counts must be unique and increasing")
    if targets[-1] > nRows:
        raise ValueError(f"Largest target has {targets[-1]} rows, source has {nRows}")
    return targets


def select_rows_from_priorities(
    priorities: np.ndarray,
    targetRows: Sequence[int],
) -> dict[int, np.ndarray]:
    priorities = np.asarray(priorities)
    if priorities.ndim != 1:
        raise ValueError("priorities must be one-dimensional")
    if not np.issubdtype(priorities.dtype, np.integer):
        raise TypeError("priorities must use an integer dtype")
    targets = _normalize_targets(targetRows, len(priorities))
    source_rows = np.arange(len(priorities), dtype=np.int64)
    ranked = np.lexsort((source_rows, priorities.astype(np.uint64, copy=False)))
    return {
        target: np.sort(ranked[:target].astype(np.int64, copy=False), kind="stable")
        for target in targets
    }


def select_nested_rows(
    nRows: int,
    targetRows: Sequence[int] = DEFAULT_TARGET_SIZES,
    *,
    seed: int = DEFAULT_SAMPLING_SEED,
    sourceVersion: str = SOURCE_SPEC.versionId,
) -> dict[int, np.ndarray]:
    priorities = row_priorities(
        nRows,
        seed=seed,
        sourceVersion=sourceVersion,
    )
    return select_rows_from_priorities(priorities, targetRows)


def ordered_source_row_digest(
    sourceRows: np.ndarray | Sequence[int],
    *,
    chunkRows: int = 1_000_000,
) -> str:
    rows = np.asarray(sourceRows)
    if rows.ndim != 1:
        raise ValueError("sourceRows must be one-dimensional")
    if not np.issubdtype(rows.dtype, np.integer):
        raise TypeError("sourceRows must use an integer dtype")
    if chunkRows <= 0:
        raise ValueError("chunkRows must be positive")
    if len(rows) and int(rows.min()) < 0:
        raise ValueError("sourceRows cannot contain negative values")
    if len(rows) and int(rows.max()) > np.iinfo(np.uint64).max:
        raise ValueError("sourceRows exceed uint64")

    digest = hashlib.sha256()
    digest.update(_SOURCE_ROWS_DIGEST_DOMAIN)
    digest.update(len(rows).to_bytes(8, "little"))
    for start in range(0, len(rows), chunkRows):
        normalized = np.asarray(rows[start : start + chunkRows], dtype="<u8")
        digest.update(normalized.tobytes(order="C"))
    return digest.hexdigest()


def download_source(
    destination: str | Path,
    *,
    url: str = SOURCE_SPEC.url,
    expectedBytes: int = SOURCE_SPEC.sourceBytes,
    chunkBytes: int = _DEFAULT_IO_CHUNK_BYTES,
    opener: Callable[[str], Any] = urlopen,
) -> DownloadResult:
    if expectedBytes < 0:
        raise ValueError("expectedBytes must be nonnegative")
    if chunkBytes <= 0:
        raise ValueError("chunkBytes must be positive")

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
    digest = hashlib.sha256()
    file_bytes = 0
    try:
        with closing(opener(url)) as response:
            headers = getattr(response, "headers", {})
            declared = headers.get("Content-Length") or headers.get("content-length")
            if declared is not None and int(declared) != expectedBytes:
                raise ValueError(
                    f"Source declares {declared} bytes, expected {expectedBytes}"
                )
            with temporary.open("xb") as handle:
                while chunk := response.read(chunkBytes):
                    handle.write(chunk)
                    digest.update(chunk)
                    file_bytes += len(chunk)
        if file_bytes != expectedBytes:
            raise ValueError(f"Downloaded {file_bytes} bytes, expected {expectedBytes}")
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    return DownloadResult(
        filePath=destination,
        fileBytes=file_bytes,
        sha256=digest.hexdigest(),
    )


def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _require_encoding(
    value: h5py.File | h5py.Group | h5py.Dataset,
    path: str,
    encodingType: str,
    encodingVersion: str,
) -> None:
    actual_type = value.attrs.get("encoding-type")
    actual_version = value.attrs.get("encoding-version")
    if actual_type is None or _decode_text(actual_type) != encodingType:
        raise ValueError(
            f"{path} encoding-type is {actual_type!r}, expected {encodingType!r}"
        )
    if actual_version is None or _decode_text(actual_version) != encodingVersion:
        raise ValueError(
            f"{path} encoding-version is {actual_version!r}, "
            f"expected {encodingVersion!r}"
        )


def _require_group(h5: h5py.File, path: str) -> h5py.Group:
    if path not in h5 or not isinstance(h5[path], h5py.Group):
        raise ValueError(f"Required HDF5 group is missing: {path}")
    return h5[path]


def _require_dataset(h5: h5py.File, path: str) -> h5py.Dataset:
    if path not in h5 or not isinstance(h5[path], h5py.Dataset):
        raise ValueError(f"Required HDF5 dataset is missing: {path}")
    return h5[path]


def _validate_string_dataset(
    dataset: h5py.Dataset,
    path: str,
    expectedLength: int,
    *,
    requireEncoding: bool,
) -> None:
    if dataset.shape != (expectedLength,):
        raise ValueError(
            f"{path} shape is {dataset.shape}, expected {(expectedLength,)}"
        )
    if h5py.check_string_dtype(dataset.dtype) is None:
        raise ValueError(f"{path} must contain strings, got {dataset.dtype}")
    if requireEncoding:
        _require_encoding(dataset, path, "string-array", "0.2.0")


def _decode_string_array(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values).reshape(-1)
    if flat.dtype.kind in {"U", "S", "O"}:
        decoded = [
            item.decode() if isinstance(item, bytes | np.bytes_) else str(item)
            for item in flat.tolist()
        ]
        return np.asarray(decoded, dtype=object)
    raise ValueError(f"Unsupported string array dtype: {flat.dtype}")


def _load_string_column(
    h5: h5py.File,
    path: str,
    *,
    expectedLength: int,
) -> np.ndarray:
    """Load a string AnnData column from a dataset or categorical group."""
    if path not in h5:
        raise ValueError(f"Required HDF5 path is missing: {path}")
    value = h5[path]
    if isinstance(value, h5py.Dataset):
        if value.shape != (expectedLength,):
            raise ValueError(
                f"{path} shape is {value.shape}, expected {(expectedLength,)}"
            )
        return _decode_string_array(np.asarray(value[:]))
    if isinstance(value, h5py.Group):
        encoding = value.attrs.get("encoding-type")
        if encoding is None or _decode_text(encoding) != "categorical":
            raise ValueError(f"{path} is a group but not a categorical string column")
        categories = _decode_string_array(
            np.asarray(_require_dataset(h5, f"{path}/categories")[:])
        )
        codes = np.asarray(_require_dataset(h5, f"{path}/codes")[:])
        if codes.shape != (expectedLength,):
            raise ValueError(
                f"{path}/codes shape is {codes.shape}, expected {(expectedLength,)}"
            )
        if codes.min(initial=0) < -1 or codes.max(initial=-1) >= len(categories):
            raise ValueError(f"{path}/codes contains out-of-range category ids")
        # AnnData uses -1 for missing; map those to empty strings.
        output = np.empty(expectedLength, dtype=object)
        missing = codes < 0
        output[missing] = ""
        output[~missing] = categories[codes[~missing]]
        return output
    raise ValueError(f"{path} must be a string dataset or categorical group")


def _validate_indptr(
    indptr: h5py.Dataset,
    *,
    nRows: int,
    nnz: int,
    expectedDtype: str,
    chunkRows: int,
) -> int:
    if chunkRows <= 0:
        raise ValueError("chunkRows must be positive")
    if indptr.shape != (nRows + 1,):
        raise ValueError(f"CSR indptr shape is {indptr.shape}, expected {(nRows + 1,)}")
    if indptr.dtype != np.dtype(expectedDtype):
        raise ValueError(
            f"CSR indptr dtype is {indptr.dtype}, expected {expectedDtype}"
        )
    if int(indptr[0]) != 0:
        raise ValueError("CSR indptr must start at zero")
    if int(indptr[-1]) != nnz:
        raise ValueError(f"CSR indptr terminates at {int(indptr[-1])}, expected {nnz}")

    previous: int | None = None
    for start in range(0, nRows + 1, chunkRows):
        values = np.asarray(indptr[start : min(start + chunkRows, nRows + 1)])
        if previous is not None and len(values) and int(values[0]) < previous:
            raise ValueError(f"CSR indptr decreases at position {start}")
        if len(values) > 1 and np.any(values[1:] < values[:-1]):
            offset = int(np.flatnonzero(values[1:] < values[:-1])[0]) + start + 1
            raise ValueError(f"CSR indptr decreases at position {offset}")
        if len(values):
            previous = int(values[-1])

    if nRows == 0:
        return 0
    return int(indptr[-1]) - int(indptr[-2])


def validate_source_h5ad(
    path: str | Path,
    *,
    spec: SourceSpec = SOURCE_SPEC,
    indptrChunkRows: int = _DEFAULT_INDPTR_CHUNK_ROWS,
) -> SourceValidation:
    path = Path(path)
    file_bytes = path.stat().st_size
    if file_bytes != spec.sourceBytes:
        raise ValueError(f"Source has {file_bytes} bytes, expected {spec.sourceBytes}")

    with h5py.File(path, mode="r") as h5:
        _require_encoding(h5, "/", "anndata", "0.1.0")
        matrix = _require_group(h5, spec.matrixKey)
        _require_encoding(
            matrix,
            spec.matrixKey,
            spec.matrixEncoding,
            "0.1.0",
        )
        shape = tuple(int(value) for value in matrix.attrs.get("shape", ()))
        expected_shape = (spec.nRows, spec.nColumns)
        if shape != expected_shape:
            raise ValueError(
                f"{spec.matrixKey} shape is {shape}, expected {expected_shape}"
            )

        data = _require_dataset(h5, f"{spec.matrixKey}/data")
        indices = _require_dataset(h5, f"{spec.matrixKey}/indices")
        indptr = _require_dataset(h5, f"{spec.matrixKey}/indptr")
        if data.shape != (spec.nnz,):
            raise ValueError(f"CSR data shape is {data.shape}, expected {(spec.nnz,)}")
        if indices.shape != (spec.nnz,):
            raise ValueError(
                f"CSR indices shape is {indices.shape}, expected {(spec.nnz,)}"
            )
        if data.dtype != np.dtype(spec.dataDtype):
            raise ValueError(
                f"CSR data dtype is {data.dtype}, expected {spec.dataDtype}"
            )
        if indices.dtype != np.dtype(spec.indicesDtype):
            raise ValueError(
                f"CSR indices dtype is {indices.dtype}, expected {spec.indicesDtype}"
            )
        final_row_nnz = _validate_indptr(
            indptr,
            nRows=spec.nRows,
            nnz=spec.nnz,
            expectedDtype=spec.indptrDtype,
            chunkRows=indptrChunkRows,
        )

        cell_ids = _require_dataset(h5, spec.cellIdsKey)
        feature_ids = _require_dataset(h5, spec.featureIdsKey)
        _validate_string_dataset(
            cell_ids,
            spec.cellIdsKey,
            spec.nRows,
            requireEncoding=True,
        )
        _validate_string_dataset(
            feature_ids,
            spec.featureIdsKey,
            spec.nColumns,
            requireEncoding=True,
        )
        feature_names = _load_string_column(
            h5,
            spec.featureNameKey,
            expectedLength=spec.nColumns,
        )
        if feature_names.shape != (spec.nColumns,):
            raise ValueError(
                f"{spec.featureNameKey} resolved length is {feature_names.shape}, "
                f"expected {(spec.nColumns,)}"
            )

    return SourceValidation(
        nRows=spec.nRows,
        nColumns=spec.nColumns,
        nnz=spec.nnz,
        finalRowNnz=final_row_nnz,
        dataDtype=spec.dataDtype,
        indicesDtype=spec.indicesDtype,
        indptrDtype=spec.indptrDtype,
    )


def _validate_selected_rows(
    sourceRows: np.ndarray | Sequence[int],
    nRows: int,
) -> np.ndarray:
    rows = np.asarray(sourceRows)
    if rows.ndim != 1:
        raise ValueError("sourceRows must be one-dimensional")
    if not np.issubdtype(rows.dtype, np.integer):
        raise TypeError("sourceRows must use an integer dtype")
    if len(rows) == 0:
        raise ValueError("sourceRows cannot be empty")
    if int(rows.min()) < 0 or int(rows.max()) >= nRows:
        raise ValueError(f"sourceRows must be between 0 and {nRows - 1}")
    rows = rows.astype(np.int64, copy=False)
    if np.any(rows[1:] <= rows[:-1]):
        raise ValueError("sourceRows must be unique and in source order")
    return rows


def _row_boundaries(
    indptr: h5py.Dataset,
    rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    span_rows = int(rows[-1] - rows[0] + 1)
    if span_rows <= len(rows) * 4:
        pointers = np.asarray(indptr[int(rows[0]) : int(rows[-1]) + 2])
        local_rows = rows - rows[0]
        return pointers[local_rows], pointers[local_rows + 1]
    return np.asarray(indptr[rows]), np.asarray(indptr[rows + 1])


def _selected_nnz(
    indptr: h5py.Dataset,
    rows: np.ndarray,
    rowBatchSize: int,
) -> int:
    total = 0
    for start in range(0, len(rows), rowBatchSize):
        batch = rows[start : start + rowBatchSize]
        row_starts, row_ends = _row_boundaries(indptr, batch)
        counts = row_ends - row_starts
        if np.any(counts < 0):
            raise ValueError("Source CSR indptr is not monotonic")
        total += int(counts.sum(dtype=np.int64))
    return total


def _chunk_elements(length: int, dtype: np.dtype[Any]) -> int:
    desired = max(1, _H5_DATA_CHUNK_BYTES // max(1, dtype.itemsize))
    return max(1, min(max(1, length), desired))


def _create_numeric_dataset(
    group: h5py.Group,
    name: str,
    length: int,
    dtype: np.dtype[Any],
) -> h5py.Dataset:
    return group.create_dataset(
        name,
        shape=(length,),
        maxshape=(None,),
        dtype=dtype,
        chunks=(_chunk_elements(length, dtype),),
        compression="gzip",
        compression_opts=4,
    )


def _create_string_dataset(
    group: h5py.Group,
    name: str,
    length: int,
) -> h5py.Dataset:
    dataset = group.create_dataset(
        name,
        shape=(length,),
        maxshape=(None,),
        dtype=h5py.string_dtype(encoding="utf-8"),
        chunks=(max(1, min(max(1, length), 65_536)),),
        compression="gzip",
        compression_opts=4,
    )
    dataset.attrs["encoding-type"] = "string-array"
    dataset.attrs["encoding-version"] = "0.2.0"
    return dataset


def _create_dataframe_group(
    h5: h5py.File,
    name: str,
    columns: Sequence[str],
) -> h5py.Group:
    group = h5.create_group(name)
    group.attrs["encoding-type"] = "dataframe"
    group.attrs["encoding-version"] = "0.2.0"
    group.attrs["_index"] = "_index"
    group.attrs["column-order"] = np.asarray(
        columns,
        dtype=h5py.string_dtype(encoding="utf-8"),
    )
    return group


def _copy_pair_slice(
    sourceData: h5py.Dataset,
    sourceIndices: h5py.Dataset,
    destinationData: h5py.Dataset,
    destinationIndices: h5py.Dataset,
    sourceStart: int,
    sourceEnd: int,
    destinationStart: int,
    copyBufferBytes: int,
) -> int:
    item_bytes = sourceData.dtype.itemsize + sourceIndices.dtype.itemsize
    chunk_elements = max(1, copyBufferBytes // max(1, item_bytes))
    output_position = destinationStart
    for start in range(sourceStart, sourceEnd, chunk_elements):
        end = min(start + chunk_elements, sourceEnd)
        data_values = np.asarray(sourceData[start:end])
        index_values = np.asarray(sourceIndices[start:end])
        output_end = output_position + len(data_values)
        destinationData[output_position:output_end] = data_values
        destinationIndices[output_position:output_end] = index_values
        output_position = output_end
    return output_position


def _copy_masked_csr_window(
    sourceData: h5py.Dataset,
    sourceIndices: h5py.Dataset,
    destinationData: h5py.Dataset,
    destinationIndices: h5py.Dataset,
    rowStarts: np.ndarray,
    rowEnds: np.ndarray,
    destinationStart: int,
) -> int:
    span_start = int(rowStarts[0])
    span_end = int(rowEnds[-1])
    span_nnz = span_end - span_start
    selected_nnz = int((rowEnds - rowStarts).sum(dtype=np.int64))
    if selected_nnz == 0:
        return destinationStart

    data_values = np.asarray(sourceData[span_start:span_end])
    index_values = np.asarray(sourceIndices[span_start:span_end])
    local_starts = rowStarts.astype(np.int64, copy=False) - span_start
    local_ends = rowEnds.astype(np.int64, copy=False) - span_start
    deltas = np.zeros(span_nnz + 1, dtype=np.int16)
    np.add.at(deltas, local_starts, 1)
    np.add.at(deltas, local_ends, -1)
    mask = np.cumsum(deltas[:-1], dtype=np.int32) > 0
    if int(mask.sum()) != selected_nnz:
        raise ValueError("Selected CSR row mask has an unexpected size")

    destination_end = destinationStart + selected_nnz
    destinationData[destinationStart:destination_end] = data_values[mask]
    destinationIndices[destinationStart:destination_end] = index_values[mask]
    return destination_end


def _copy_csr_batch(
    sourceData: h5py.Dataset,
    sourceIndices: h5py.Dataset,
    destinationData: h5py.Dataset,
    destinationIndices: h5py.Dataset,
    rows: np.ndarray,
    rowStarts: np.ndarray,
    rowEnds: np.ndarray,
    destinationStart: int,
    copyBufferBytes: int,
) -> int:
    counts = rowEnds - rowStarts
    selected_nnz = int(counts.sum(dtype=np.int64))
    if selected_nnz == 0:
        return destinationStart

    span_start = int(rowStarts[0])
    span_end = int(rowEnds[-1])
    span_nnz = span_end - span_start
    item_bytes = sourceData.dtype.itemsize + sourceIndices.dtype.itemsize
    estimated_bytes = span_nnz * (2 * item_bytes + 7)
    if span_nnz <= selected_nnz * 2 and estimated_bytes <= copyBufferBytes:
        return _copy_masked_csr_window(
            sourceData,
            sourceIndices,
            destinationData,
            destinationIndices,
            rowStarts,
            rowEnds,
            destinationStart,
        )

    output_position = destinationStart
    max_span_nnz = max(1, copyBufferBytes // max(1, 2 * item_bytes + 7))
    window_start = 0
    while window_start < len(rows):
        window_end = window_start + 1
        while (
            window_end < len(rows)
            and int(rowEnds[window_end] - rowStarts[window_start]) <= max_span_nnz
        ):
            window_end += 1

        window_starts = rowStarts[window_start:window_end]
        window_ends = rowEnds[window_start:window_end]
        if int(window_ends[-1] - window_starts[0]) > max_span_nnz:
            output_position = _copy_pair_slice(
                sourceData,
                sourceIndices,
                destinationData,
                destinationIndices,
                int(window_starts[0]),
                int(window_ends[0]),
                output_position,
                copyBufferBytes,
            )
        else:
            output_position = _copy_masked_csr_window(
                sourceData,
                sourceIndices,
                destinationData,
                destinationIndices,
                window_starts,
                window_ends,
                output_position,
            )
        window_start = window_end
    if output_position != destinationStart + selected_nnz:
        raise ValueError("Copied CSR data length does not match selected rows")
    return output_position


def _copy_selected_strings(
    source: h5py.Dataset,
    destination: h5py.Dataset,
    rows: np.ndarray,
    rowBatchSize: int,
) -> None:
    output_start = 0
    for start in range(0, len(rows), rowBatchSize):
        batch = rows[start : start + rowBatchSize]
        span_rows = int(batch[-1] - batch[0] + 1)
        if span_rows <= len(batch) * 2:
            values = np.asarray(source[int(batch[0]) : int(batch[-1]) + 1])
            values = values[batch - batch[0]]
        else:
            values = np.asarray(source[batch])
        output_end = output_start + len(batch)
        destination[output_start:output_end] = values
        output_start = output_end


def _check_integer_capacity(dtype: np.dtype[Any], maximum: int) -> None:
    if not np.issubdtype(dtype, np.integer):
        raise ValueError(f"CSR index dtype must be integer, got {dtype}")
    if maximum > int(np.iinfo(dtype).max):
        raise OverflowError(f"{maximum} does not fit in {dtype}")


def _load_numeric_dataset(
    dataset: h5py.Dataset,
    *,
    dtype: np.dtype[Any] | None = None,
    chunkElements: int = _DEFAULT_LOAD_CHUNK_ELEMENTS,
) -> np.ndarray:
    if dataset.ndim != 1:
        raise ValueError("Expected a 1-D HDF5 dataset")
    if chunkElements <= 0:
        raise ValueError("chunkElements must be positive")
    length = int(dataset.shape[0])
    out_dtype = np.dtype(dataset.dtype if dtype is None else dtype)
    output = np.empty(length, dtype=out_dtype)
    for start in range(0, length, chunkElements):
        end = min(start + chunkElements, length)
        output[start:end] = np.asarray(dataset[start:end], dtype=out_dtype)
    return output


def load_csr_source_into_memory(
    path: str | Path,
    *,
    spec: SourceSpec = SOURCE_SPEC,
    chunkElements: int = _DEFAULT_LOAD_CHUNK_ELEMENTS,
) -> InMemoryCsrSource:
    """Load the source CSR into RAM for repeated nested subset writes.

    Column indices are stored as int32 when they fit, which roughly halves
    resident memory versus the on-disk int64 index array.
    """
    path = Path(path)
    with h5py.File(path, mode="r") as h5:
        data_dataset = _require_dataset(h5, f"{spec.matrixKey}/data")
        indices_dataset = _require_dataset(h5, f"{spec.matrixKey}/indices")
        indptr_dataset = _require_dataset(h5, f"{spec.matrixKey}/indptr")
        if data_dataset.shape != (spec.nnz,):
            raise ValueError(
                f"CSR data shape is {data_dataset.shape}, expected {(spec.nnz,)}"
            )
        if indices_dataset.shape != (spec.nnz,):
            raise ValueError(
                f"CSR indices shape is {indices_dataset.shape}, expected {(spec.nnz,)}"
            )
        if indptr_dataset.shape != (spec.nRows + 1,):
            raise ValueError(
                f"CSR indptr shape is {indptr_dataset.shape}, "
                f"expected {(spec.nRows + 1,)}"
            )

        data = _load_numeric_dataset(data_dataset, chunkElements=chunkElements)
        max_column = spec.nColumns - 1
        if max_column <= int(np.iinfo(np.int32).max):
            indices = _load_numeric_dataset(
                indices_dataset,
                dtype=np.dtype(np.int32),
                chunkElements=chunkElements,
            )
        else:
            indices = _load_numeric_dataset(
                indices_dataset,
                chunkElements=chunkElements,
            )
        indptr = _load_numeric_dataset(indptr_dataset, chunkElements=chunkElements)
        if int(indptr[0]) != 0 or int(indptr[-1]) != spec.nnz:
            raise ValueError("Loaded CSR indptr does not match source nnz")

        cell_ids = _decode_string_array(_require_dataset(h5, spec.cellIdsKey)[:])
        feature_ids = _decode_string_array(_require_dataset(h5, spec.featureIdsKey)[:])
        feature_names = _load_string_column(
            h5,
            spec.featureNameKey,
            expectedLength=spec.nColumns,
        )

    if cell_ids.shape != (spec.nRows,):
        raise ValueError(
            f"Loaded cell ids shape is {cell_ids.shape}, expected {(spec.nRows,)}"
        )
    if feature_ids.shape != (spec.nColumns,):
        raise ValueError(
            f"Loaded feature ids shape is {feature_ids.shape}, "
            f"expected {(spec.nColumns,)}"
        )
    if feature_names.shape != (spec.nColumns,):
        raise ValueError(
            f"Loaded feature names shape is {feature_names.shape}, "
            f"expected {(spec.nColumns,)}"
        )

    return InMemoryCsrSource(
        data=data,
        indices=indices,
        indptr=indptr,
        nRows=spec.nRows,
        nColumns=spec.nColumns,
        dataDtype=np.dtype(spec.dataDtype),
        indicesDtype=np.dtype(spec.indicesDtype),
        indptrDtype=np.dtype(spec.indptrDtype),
        cellIds=cell_ids,
        featureIds=feature_ids,
        featureNames=feature_names,
    )


def _copy_memory_csr_batch(
    source: InMemoryCsrSource,
    destinationData: h5py.Dataset,
    destinationIndices: h5py.Dataset,
    rows: np.ndarray,
    destinationStart: int,
) -> int:
    starts = source.indptr[rows]
    ends = source.indptr[rows + 1]
    counts = ends - starts
    if np.any(counts < 0):
        raise ValueError("Source CSR indptr is not monotonic")
    selected_nnz = int(counts.sum(dtype=np.int64))
    if selected_nnz == 0:
        return destinationStart

    buffer_data = np.empty(selected_nnz, dtype=source.data.dtype)
    buffer_indices = np.empty(selected_nnz, dtype=source.indices.dtype)
    position = 0
    for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
        width = end - start
        buffer_data[position : position + width] = source.data[start:end]
        buffer_indices[position : position + width] = source.indices[start:end]
        position += width
    if position != selected_nnz:
        raise ValueError("Copied CSR batch length does not match selected nnz")

    destination_end = destinationStart + selected_nnz
    destinationData[destinationStart:destination_end] = buffer_data
    if buffer_indices.dtype == source.indicesDtype:
        destinationIndices[destinationStart:destination_end] = buffer_indices
    else:
        destinationIndices[destinationStart:destination_end] = buffer_indices.astype(
            source.indicesDtype,
            copy=False,
        )
    return destination_end


def _copy_selected_strings_array(
    source: np.ndarray,
    destination: h5py.Dataset,
    rows: np.ndarray,
    rowBatchSize: int,
) -> None:
    output_start = 0
    for start in range(0, len(rows), rowBatchSize):
        batch = rows[start : start + rowBatchSize]
        output_end = output_start + len(batch)
        destination[output_start:output_end] = source[batch]
        output_start = output_end


def write_h5ad_sample_from_memory(
    source: InMemoryCsrSource,
    destinationPath: str | Path,
    sourceRows: np.ndarray | Sequence[int],
    *,
    rowBatchSize: int = _DEFAULT_ROW_BATCH_SIZE,
) -> H5adWriteResult:
    if rowBatchSize <= 0:
        raise ValueError("rowBatchSize must be positive")

    destination_path = Path(destinationPath)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists():
        raise FileExistsError(f"Destination already exists: {destination_path}")
    temporary_path = destination_path.with_name(
        f".{destination_path.name}.{uuid.uuid4().hex}.part"
    )
    rows = _validate_selected_rows(sourceRows, source.nRows)
    nnz = int((source.indptr[rows + 1] - source.indptr[rows]).sum(dtype=np.int64))
    _check_integer_capacity(source.indptrDtype, nnz)
    _check_integer_capacity(source.indicesDtype, source.nColumns - 1)

    try:
        with h5py.File(temporary_path, mode="w") as destination:
            destination.attrs["encoding-type"] = "anndata"
            destination.attrs["encoding-version"] = "0.1.0"
            matrix = destination.create_group("X")
            matrix.attrs["encoding-type"] = "csr_matrix"
            matrix.attrs["encoding-version"] = "0.1.0"
            matrix.attrs["shape"] = np.asarray(
                [len(rows), source.nColumns],
                dtype=np.int64,
            )
            destination_data = _create_numeric_dataset(
                matrix,
                "data",
                nnz,
                source.dataDtype,
            )
            destination_indices = _create_numeric_dataset(
                matrix,
                "indices",
                nnz,
                source.indicesDtype,
            )
            destination_indptr = _create_numeric_dataset(
                matrix,
                "indptr",
                len(rows) + 1,
                source.indptrDtype,
            )
            destination_indptr[0] = 0

            output_data_position = 0
            output_row_position = 0
            for start in range(0, len(rows), rowBatchSize):
                batch = rows[start : start + rowBatchSize]
                batch_starts = source.indptr[batch]
                batch_ends = source.indptr[batch + 1]
                counts = batch_ends - batch_starts
                cumulative = np.cumsum(counts, dtype=np.int64) + output_data_position
                output_row_end = output_row_position + len(batch)
                destination_indptr[output_row_position + 1 : output_row_end + 1] = (
                    cumulative.astype(source.indptrDtype, copy=False)
                )
                output_data_position = _copy_memory_csr_batch(
                    source,
                    destination_data,
                    destination_indices,
                    batch,
                    output_data_position,
                )
                output_row_position = output_row_end
            if output_data_position != nnz:
                raise ValueError(
                    f"Copied {output_data_position} values, expected {nnz}"
                )

            obs = _create_dataframe_group(destination, "obs", ())
            output_cell_ids = _create_string_dataset(obs, "_index", len(rows))
            _copy_selected_strings_array(
                source.cellIds,
                output_cell_ids,
                rows,
                rowBatchSize,
            )

            var = _create_dataframe_group(
                destination,
                "var",
                ("feature_name",),
            )
            output_feature_ids = _create_string_dataset(
                var,
                "_index",
                source.nColumns,
            )
            output_feature_names = _create_string_dataset(
                var,
                "feature_name",
                source.nColumns,
            )
            output_feature_ids[:] = source.featureIds
            output_feature_names[:] = source.featureNames

        os.replace(temporary_path, destination_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    return H5adWriteResult(
        filePath=destination_path,
        nRows=len(rows),
        nColumns=source.nColumns,
        nnz=nnz,
        sourceRowsSha256=ordered_source_row_digest(rows),
        finalSourceRow=int(rows[-1]),
        dataDtype=str(source.dataDtype),
        indicesDtype=str(source.indicesDtype),
        indptrDtype=str(source.indptrDtype),
    )


def write_h5ad_sample(
    sourcePath: str | Path,
    destinationPath: str | Path,
    sourceRows: np.ndarray | Sequence[int],
    *,
    spec: SourceSpec = SOURCE_SPEC,
    rowBatchSize: int = _DEFAULT_ROW_BATCH_SIZE,
    copyBufferBytes: int = _DEFAULT_COPY_BUFFER_BYTES,
) -> H5adWriteResult:
    if rowBatchSize <= 0:
        raise ValueError("rowBatchSize must be positive")
    if copyBufferBytes <= 0:
        raise ValueError("copyBufferBytes must be positive")

    source_path = Path(sourcePath)
    destination_path = Path(destinationPath)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists():
        raise FileExistsError(f"Destination already exists: {destination_path}")
    temporary_path = destination_path.with_name(
        f".{destination_path.name}.{uuid.uuid4().hex}.part"
    )

    try:
        with (
            h5py.File(source_path, mode="r") as source,
            h5py.File(temporary_path, mode="w") as destination,
        ):
            source_matrix = _require_group(source, spec.matrixKey)
            shape = tuple(int(value) for value in source_matrix.attrs.get("shape", ()))
            if len(shape) != 2:
                raise ValueError(f"{spec.matrixKey} has invalid shape metadata")
            n_source_rows, n_columns = shape
            rows = _validate_selected_rows(sourceRows, n_source_rows)

            source_data = _require_dataset(source, f"{spec.matrixKey}/data")
            source_indices = _require_dataset(
                source,
                f"{spec.matrixKey}/indices",
            )
            source_indptr = _require_dataset(
                source,
                f"{spec.matrixKey}/indptr",
            )
            if source_indptr.shape != (n_source_rows + 1,):
                raise ValueError("Source CSR indptr length does not match shape")
            if not np.issubdtype(source_indices.dtype, np.integer):
                raise ValueError("Source CSR indices must use an integer dtype")
            if not np.issubdtype(source_indptr.dtype, np.integer):
                raise ValueError("Source CSR indptr must use an integer dtype")
            data_dtype = str(source_data.dtype)
            indices_dtype = str(source_indices.dtype)
            indptr_dtype = str(source_indptr.dtype)

            nnz = _selected_nnz(source_indptr, rows, rowBatchSize)
            _check_integer_capacity(source_indptr.dtype, nnz)
            _check_integer_capacity(source_indices.dtype, n_columns - 1)

            destination.attrs["encoding-type"] = "anndata"
            destination.attrs["encoding-version"] = "0.1.0"
            matrix = destination.create_group("X")
            matrix.attrs["encoding-type"] = "csr_matrix"
            matrix.attrs["encoding-version"] = "0.1.0"
            matrix.attrs["shape"] = np.asarray(
                [len(rows), n_columns],
                dtype=np.int64,
            )
            destination_data = _create_numeric_dataset(
                matrix,
                "data",
                nnz,
                source_data.dtype,
            )
            destination_indices = _create_numeric_dataset(
                matrix,
                "indices",
                nnz,
                source_indices.dtype,
            )
            destination_indptr = _create_numeric_dataset(
                matrix,
                "indptr",
                len(rows) + 1,
                source_indptr.dtype,
            )
            destination_indptr[0] = 0

            output_data_position = 0
            output_row_position = 0
            for start in range(0, len(rows), rowBatchSize):
                batch = rows[start : start + rowBatchSize]
                row_starts, row_ends = _row_boundaries(source_indptr, batch)
                counts = row_ends - row_starts
                cumulative = np.cumsum(counts, dtype=np.int64) + output_data_position
                output_row_end = output_row_position + len(batch)
                destination_indptr[output_row_position + 1 : output_row_end + 1] = (
                    cumulative.astype(source_indptr.dtype, copy=False)
                )
                output_data_position = _copy_csr_batch(
                    source_data,
                    source_indices,
                    destination_data,
                    destination_indices,
                    batch,
                    row_starts,
                    row_ends,
                    output_data_position,
                    copyBufferBytes,
                )
                output_row_position = output_row_end
            if output_data_position != nnz:
                raise ValueError(
                    f"Copied {output_data_position} values, expected {nnz}"
                )

            obs = _create_dataframe_group(destination, "obs", ())
            output_cell_ids = _create_string_dataset(obs, "_index", len(rows))
            source_cell_ids = _require_dataset(source, spec.cellIdsKey)
            _copy_selected_strings(
                source_cell_ids,
                output_cell_ids,
                rows,
                rowBatchSize,
            )

            var = _create_dataframe_group(
                destination,
                "var",
                ("feature_name",),
            )
            output_feature_ids = _create_string_dataset(
                var,
                "_index",
                n_columns,
            )
            output_feature_names = _create_string_dataset(
                var,
                "feature_name",
                n_columns,
            )
            source_feature_ids = _require_dataset(source, spec.featureIdsKey)
            source_feature_names = _load_string_column(
                source,
                spec.featureNameKey,
                expectedLength=n_columns,
            )
            output_feature_ids[:] = source_feature_ids[:]
            output_feature_names[:] = source_feature_names[:]

        os.replace(temporary_path, destination_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    return H5adWriteResult(
        filePath=destination_path,
        nRows=len(rows),
        nColumns=n_columns,
        nnz=nnz,
        sourceRowsSha256=ordered_source_row_digest(rows),
        finalSourceRow=int(rows[-1]),
        dataDtype=data_dtype,
        indicesDtype=indices_dtype,
        indptrDtype=indptr_dtype,
    )


def read_csr_row(
    path: str | Path,
    row: int,
    *,
    matrixKey: str = SOURCE_SPEC.matrixKey,
) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, mode="r") as h5:
        matrix = _require_group(h5, matrixKey)
        shape = tuple(int(value) for value in matrix.attrs.get("shape", ()))
        if len(shape) != 2 or row < 0 or row >= shape[0]:
            raise IndexError(f"CSR row {row} is outside matrix shape {shape}")
        indptr = _require_dataset(h5, f"{matrixKey}/indptr")
        start, end = (int(value) for value in indptr[row : row + 2])
        indices = np.asarray(_require_dataset(h5, f"{matrixKey}/indices")[start:end])
        data = np.asarray(_require_dataset(h5, f"{matrixKey}/data")[start:end])
    return indices, data


def _attribute_strings(value: Any) -> list[str]:
    return [_decode_text(item) for item in np.asarray(value).reshape(-1)]


def _validate_output_dataframe(
    h5: h5py.File,
    name: str,
    expectedLength: int,
    columns: Sequence[str],
) -> None:
    group = _require_group(h5, name)
    _require_encoding(group, name, "dataframe", "0.2.0")
    if _decode_text(group.attrs.get("_index")) != "_index":
        raise ValueError(f"{name} dataframe index must be _index")
    if _attribute_strings(group.attrs.get("column-order", ())) != list(columns):
        raise ValueError(f"{name} dataframe columns do not match {list(columns)}")
    expected_keys = {"_index", *columns}
    if set(group.keys()) != expected_keys:
        raise ValueError(
            f"{name} contains {sorted(group.keys())}, expected {sorted(expected_keys)}"
        )
    _validate_string_dataset(
        _require_dataset(h5, f"{name}/_index"),
        f"{name}/_index",
        expectedLength,
        requireEncoding=True,
    )
    for column in columns:
        _validate_string_dataset(
            _require_dataset(h5, f"{name}/{column}"),
            f"{name}/{column}",
            expectedLength,
            requireEncoding=True,
        )


def validate_prepared_h5ad(
    path: str | Path,
    *,
    expectedRows: int,
    expectedColumns: int,
    expectedNnz: int,
    expectedSourceRows: np.ndarray | Sequence[int],
    expectedFinalIndices: np.ndarray,
    expectedFinalData: np.ndarray,
    expectedDataDtype: str,
    expectedIndicesDtype: str,
    expectedIndptrDtype: str,
    expectedSha256: str | None = None,
    indptrChunkRows: int = _DEFAULT_INDPTR_CHUNK_ROWS,
) -> PreparedValidation:
    path = Path(path)
    source_rows = _validate_selected_rows(expectedSourceRows, max(1, 1 << 63))
    if len(source_rows) != expectedRows:
        raise ValueError(
            f"Expected source row list has {len(source_rows)} rows, "
            f"expected {expectedRows}"
        )

    with h5py.File(path, mode="r") as h5:
        if set(h5.keys()) != {"X", "obs", "var"}:
            raise ValueError(
                f"Prepared H5AD root contains unexpected keys: {sorted(h5.keys())}"
            )
        _require_encoding(h5, "/", "anndata", "0.1.0")
        matrix = _require_group(h5, "X")
        _require_encoding(matrix, "X", "csr_matrix", "0.1.0")
        shape = tuple(int(value) for value in matrix.attrs.get("shape", ()))
        if shape != (expectedRows, expectedColumns):
            raise ValueError(
                f"Prepared matrix shape is {shape}, expected "
                f"{(expectedRows, expectedColumns)}"
            )

        data = _require_dataset(h5, "X/data")
        indices = _require_dataset(h5, "X/indices")
        indptr = _require_dataset(h5, "X/indptr")
        if data.shape != (expectedNnz,) or indices.shape != (expectedNnz,):
            raise ValueError("Prepared CSR data lengths do not match expected nnz")
        if data.dtype != np.dtype(expectedDataDtype):
            raise ValueError(
                f"Prepared data dtype is {data.dtype}, expected {expectedDataDtype}"
            )
        if indices.dtype != np.dtype(expectedIndicesDtype):
            raise ValueError(
                f"Prepared indices dtype is {indices.dtype}, "
                f"expected {expectedIndicesDtype}"
            )
        _validate_indptr(
            indptr,
            nRows=expectedRows,
            nnz=expectedNnz,
            expectedDtype=expectedIndptrDtype,
            chunkRows=indptrChunkRows,
        )

        final_start, final_end = (
            int(value) for value in indptr[expectedRows - 1 : expectedRows + 1]
        )
        final_indices = np.asarray(indices[final_start:final_end])
        final_data = np.asarray(data[final_start:final_end])
        if not np.array_equal(final_indices, expectedFinalIndices):
            raise ValueError("Prepared final-row indices do not match source")
        if not np.array_equal(final_data, expectedFinalData):
            raise ValueError("Prepared final-row data do not match source")

        _validate_output_dataframe(h5, "obs", expectedRows, ())
        _validate_output_dataframe(
            h5,
            "var",
            expectedColumns,
            ("feature_name",),
        )

    import anndata

    backed = anndata.read_h5ad(path, backed="r")
    try:
        if backed.shape != (expectedRows, expectedColumns):
            raise ValueError(
                f"AnnData sees shape {backed.shape}, expected "
                f"{(expectedRows, expectedColumns)}"
            )
    finally:
        backed.file.close()

    from scarf.readers import H5adReader

    reader = H5adReader(
        str(path),
        matrix_key="X",
        cell_attrs_key="obs",
        cell_ids_key="_index",
        feature_attrs_key="var",
        feature_ids_key="_index",
        feature_name_key="feature_name",
    )
    try:
        if (reader.nCells, reader.nFeatures) != (
            expectedRows,
            expectedColumns,
        ):
            raise ValueError(
                "Scarf H5adReader dimensions do not match prepared artifact"
            )
        if np.dtype(reader.matrixDtype) != np.dtype(expectedDataDtype):
            raise ValueError("Scarf H5adReader sees an unexpected data dtype")
    finally:
        reader.h5.close()

    actual_sha256 = sha256_file(path)
    if expectedSha256 is not None and actual_sha256 != expectedSha256.lower():
        raise ValueError(
            f"Prepared SHA256 is {actual_sha256}, expected {expectedSha256}"
        )
    return PreparedValidation(
        fileBytes=path.stat().st_size,
        sha256=actual_sha256,
        sourceRowsSha256=ordered_source_row_digest(source_rows),
        nRows=expectedRows,
        nColumns=expectedColumns,
        nnz=expectedNnz,
        finalSourceRow=int(source_rows[-1]),
    )


def prepare_local_datasets(
    sourcePath: str | Path,
    outputDirectory: str | Path,
    *,
    targetRows: Sequence[int] = DEFAULT_TARGET_SIZES,
    seed: int = DEFAULT_SAMPLING_SEED,
    spec: SourceSpec = SOURCE_SPEC,
    rowBatchSize: int = _DEFAULT_ROW_BATCH_SIZE,
    copyBufferBytes: int = _DEFAULT_COPY_BUFFER_BYTES,
    onArtifact: Callable[[PreparedArtifact], None] | None = None,
) -> PreparationResult:
    del copyBufferBytes  # retained for call-site compatibility; unused in-memory path
    source_path = Path(sourcePath)
    output_directory = Path(outputDirectory)
    output_directory.mkdir(parents=True, exist_ok=True)
    source_validation = validate_source_h5ad(source_path, spec=spec)
    source_sha256 = sha256_file(source_path)
    selections = select_nested_rows(
        spec.nRows,
        targetRows,
        seed=seed,
        sourceVersion=spec.versionId,
    )
    memory_source = load_csr_source_into_memory(source_path, spec=spec)

    artifacts: list[PreparedArtifact] = []
    for target_rows, source_rows in selections.items():
        output_path = output_directory / f"{target_rows}.h5ad"
        if output_path.exists():
            output_path.unlink()
        written = write_h5ad_sample_from_memory(
            memory_source,
            output_path,
            source_rows,
            rowBatchSize=rowBatchSize,
        )
        final_start = int(memory_source.indptr[written.finalSourceRow])
        final_end = int(memory_source.indptr[written.finalSourceRow + 1])
        final_indices = np.asarray(
            memory_source.indices[final_start:final_end],
            dtype=memory_source.indicesDtype,
        )
        final_data = np.asarray(memory_source.data[final_start:final_end])
        validated = validate_prepared_h5ad(
            output_path,
            expectedRows=target_rows,
            expectedColumns=spec.nColumns,
            expectedNnz=written.nnz,
            expectedSourceRows=source_rows,
            expectedFinalIndices=final_indices,
            expectedFinalData=final_data,
            expectedDataDtype=source_validation.dataDtype,
            expectedIndicesDtype=source_validation.indicesDtype,
            expectedIndptrDtype=source_validation.indptrDtype,
        )
        if validated.sourceRowsSha256 != written.sourceRowsSha256:
            raise ValueError("Source-row digest changed during validation")
        artifact = PreparedArtifact(
            localPath=output_path,
            targetRows=target_rows,
            nColumns=spec.nColumns,
            nnz=written.nnz,
            fileBytes=validated.fileBytes,
            sha256=validated.sha256,
            sourceRowsSha256=validated.sourceRowsSha256,
            finalSourceRow=written.finalSourceRow,
            dataDtype=written.dataDtype,
            indicesDtype=written.indicesDtype,
            indptrDtype=written.indptrDtype,
        )
        artifacts.append(artifact)
        if onArtifact is not None:
            onArtifact(artifact)

    return PreparationResult(
        sourcePath=source_path,
        sourceSha256=source_sha256,
        sourceSpec=spec,
        seed=seed,
        artifacts=tuple(artifacts),
    )


def write_fixture_h5ad(
    destinationPath: str | Path,
    *,
    nRows: int,
    nColumns: int = 500,
    seed: int = 0,
    avgNnzPerRow: int = 40,
) -> PreparedArtifact:
    """Write a small synthetic CSR H5AD for downstream stage smoke tests."""
    if nRows <= 0 or nColumns <= 0:
        raise ValueError("nRows and nColumns must be positive")
    if avgNnzPerRow <= 0:
        raise ValueError("avgNnzPerRow must be positive")

    destination_path = Path(destinationPath)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists():
        raise FileExistsError(destination_path)

    rng = np.random.default_rng(seed)
    nnz_per_row = np.clip(
        rng.poisson(avgNnzPerRow, size=nRows),
        1,
        nColumns,
    ).astype(np.int64)
    indptr = np.empty(nRows + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(nnz_per_row, out=indptr[1:])
    nnz = int(indptr[-1])
    indices = np.empty(nnz, dtype=np.int64)
    data = rng.integers(1, 20, size=nnz, dtype=np.int32).astype(np.float32)
    for row, start, end in zip(
        range(nRows),
        indptr[:-1],
        indptr[1:],
        strict=True,
    ):
        chosen = rng.choice(nColumns, size=int(end - start), replace=False)
        chosen.sort()
        indices[start:end] = chosen

    feature_ids = np.asarray(
        [f"ENSG{i:011d}" for i in range(nColumns)],
        dtype=object,
    )
    feature_names = np.asarray(
        ["MT-ND1" if i % 50 == 0 else f"GENE{i}" for i in range(nColumns)],
        dtype=object,
    )
    cell_ids = np.asarray([f"cell-{i}" for i in range(nRows)], dtype=object)

    temporary_path = destination_path.with_name(
        f".{destination_path.name}.{uuid.uuid4().hex}.part"
    )
    try:
        with h5py.File(temporary_path, mode="w") as h5:
            h5.attrs["encoding-type"] = "anndata"
            h5.attrs["encoding-version"] = "0.1.0"
            matrix = h5.create_group("X")
            matrix.attrs["encoding-type"] = "csr_matrix"
            matrix.attrs["encoding-version"] = "0.1.0"
            matrix.attrs["shape"] = np.asarray([nRows, nColumns], dtype=np.int64)
            _create_numeric_dataset(matrix, "data", nnz, data.dtype)[:] = data
            _create_numeric_dataset(matrix, "indices", nnz, indices.dtype)[:] = indices
            _create_numeric_dataset(
                matrix,
                "indptr",
                nRows + 1,
                indptr.dtype,
            )[:] = indptr

            obs = _create_dataframe_group(h5, "obs", ())
            _create_string_dataset(obs, "_index", nRows)[:] = cell_ids
            var = _create_dataframe_group(h5, "var", ("feature_name",))
            _create_string_dataset(var, "_index", nColumns)[:] = feature_ids
            _create_string_dataset(var, "feature_name", nColumns)[:] = feature_names
        os.replace(temporary_path, destination_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    return PreparedArtifact(
        localPath=destination_path,
        targetRows=nRows,
        nColumns=nColumns,
        nnz=nnz,
        fileBytes=destination_path.stat().st_size,
        sha256=sha256_file(destination_path),
        sourceRowsSha256=ordered_source_row_digest(np.arange(nRows, dtype=np.int64)),
        finalSourceRow=nRows - 1,
        dataDtype=str(data.dtype),
        indicesDtype=str(indices.dtype),
        indptrDtype=str(indptr.dtype),
    )


def prepare_fixture_datasets(
    outputDirectory: str | Path,
    *,
    targetRows: Sequence[int],
    nColumns: int = 500,
    seed: int = 0,
    onArtifact: Callable[[PreparedArtifact], None] | None = None,
) -> tuple[PreparedArtifact, ...]:
    output_directory = Path(outputDirectory)
    output_directory.mkdir(parents=True, exist_ok=True)
    artifacts: list[PreparedArtifact] = []
    for n_rows in targetRows:
        artifact = write_fixture_h5ad(
            output_directory / f"{n_rows}.h5ad",
            nRows=int(n_rows),
            nColumns=nColumns,
            seed=seed + int(n_rows),
        )
        artifacts.append(artifact)
        if onArtifact is not None:
            onArtifact(artifact)
    return tuple(artifacts)

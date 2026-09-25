"""Inspect local H5ADs, build Scarf stores, and verify exact published arrays."""

import hashlib
import math
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import h5py
import numpy as np
import zarr

import scarf
from scarf import inspect_h5ad
from scarf.storage.count_matrix import require_count_matrix_layout
from scarf.storage.profiles import is_remote_zarr_location
from scarf.storage.stores import make_store

from .._storage import Bucket, dataset_prefix, retry
from .models import DatasetRecord, Manifest

_BLOCK_VALUES = 1_000_000
_VALIDATION_ROWS = 100
_UNS_MAX_VALUES = 100_000
_UNS_MAX_BYTES = 1_048_576


def _json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(_json_safe(key)): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def _uns_value(node: h5py.Group | h5py.Dataset) -> Any:
    if isinstance(node, h5py.Group):
        return {key: _uns_value(child) for key, child in node.items()}
    if node.size > _UNS_MAX_VALUES or node.size * node.dtype.itemsize > _UNS_MAX_BYTES:
        return {
            "omitted": True,
            "reason": "Array exceeds the metadata export size limit",
            "shape": list(node.shape),
            "dtype": str(node.dtype),
        }
    return _json_safe(node[()])


def _sample_row_blocks(
    values: h5py.Dataset,
    row: int,
    columns: int,
    indices: h5py.Dataset | None,
    indptr: h5py.Dataset | None,
) -> Iterator[np.ndarray]:
    """Read all columns in one sampled row without materializing sparse zeros."""
    if indices is None or indptr is None:
        for start in range(0, columns, _BLOCK_VALUES):
            yield values[row, start : start + _BLOCK_VALUES]
        return
    start, stop = (int(value) for value in indptr[row : row + 2])
    if not 0 <= start <= stop <= values.size:
        raise ValueError(f"Sparse row {row} has invalid indptr boundaries")
    for offset in range(start, stop, _BLOCK_VALUES):
        end = min(offset + _BLOCK_VALUES, stop)
        gene_indices = indices[offset:end]
        if np.any(gene_indices < 0) or np.any(gene_indices >= columns):
            raise ValueError(
                f"Sparse indices in sampled row {row} exceed gene dimensions"
            )
        yield values[offset:end]


def _matrix_info(node: h5py.Group | h5py.Dataset) -> dict[str, Any]:
    """Read structure only; candidate validation samples rows separately."""
    sparse = isinstance(node, h5py.Group)
    values = node.get("data") if sparse else node
    shape = (
        node.attrs.get("shape", node.attrs.get("h5sparse_shape"))
        if sparse
        else node.shape
    )
    if sparse and shape is None and isinstance(node.get("shape"), h5py.Dataset):
        shape = node["shape"][:]
    encoding = _json_safe(
        node.attrs.get(
            "encoding-type",
            node.attrs.get("h5sparse_format", "csr" if sparse else "array"),
        )
    )
    return {
        "encoding": encoding,
        "shape": _json_safe(shape),
        "dtype": str(values.dtype) if isinstance(values, h5py.Dataset) else None,
        "nnz": int(values.size)
        if sparse and isinstance(values, h5py.Dataset)
        else None,
    }


def _column_length(node: h5py.Group | h5py.Dataset) -> int | None:
    if isinstance(node, h5py.Dataset):
        return int(node.shape[0]) if node.ndim == 1 else None
    for key in ("codes", "values"):
        if isinstance(node.get(key), h5py.Dataset):
            return _column_length(node[key])
    return None


def _table_length(node: h5py.Group | None) -> int | None:
    if not isinstance(node, h5py.Group):
        return None
    index = _json_safe(node.attrs.get("_index", "_index"))
    if index in node:
        return _column_length(node[index])
    for column in node.values():
        if (length := _column_length(column)) is not None:
            return length
    return None


def _validate_candidate(
    h5: h5py.File,
    key: str,
    info: dict[str, Any],
    progress: Callable | None,
) -> dict[str, Any]:
    """Check up to 100 evenly spaced cells across every feature, using bounded reads."""
    result = info | {
        "present": True,
        "validCounts": False,
        "validationComplete": False,
        "validationMode": "sampled_rows",
        "fullMatrixValidated": False,
        "reasons": [],
    }
    reasons = result["reasons"]
    node = h5[key]
    sparse = isinstance(node, h5py.Group)
    values = node.get("data") if sparse else node
    shape = info["shape"]
    feature_key = "raw/var" if key == "raw/X" else "var"
    expected = [_table_length(h5.get("obs")), _table_length(h5.get(feature_key))]
    result["featureAttrsKey"] = feature_key
    result["expectedShape"] = expected
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or shape != expected
        or any(size is None or size < 1 for size in expected)
    ):
        reasons.append(
            f"Matrix shape {shape} does not match obs and {feature_key} dimensions {expected}"
        )
        return result
    if not isinstance(values, h5py.Dataset) or values.ndim != (1 if sparse else 2):
        reasons.append(
            "Stored matrix values are missing or have unsupported dimensions"
        )
        return result
    if values.dtype.kind not in "iuf":
        reasons.append(f"Expected real numeric counts; got {values.dtype}")
        return result

    encoding = str(info["encoding"]).lower()
    result["formatSupported"] = not sparse or encoding in {"csr", "csr_matrix"}
    if not result["formatSupported"]:
        reasons.append(
            f"Sparse encoding {encoding!r} cannot be sampled by cell for bounded conversion. "
            "Scarf materializes CSC inputs. Provide CSR or dense counts."
        )
        return result
    indices = indptr = None
    if sparse:
        indices, indptr = node.get("indices"), node.get("indptr")
        if not all(
            isinstance(part, h5py.Dataset)
            and part.ndim == 1
            and part.dtype.kind in "iu"
            for part in (indices, indptr)
        ):
            reasons.append(
                "Sparse indices and indptr must be one-dimensional integer arrays"
            )
            return result
        if indices.size != values.size:
            reasons.append("Sparse indices and values have different lengths")
            return result
        if indptr.size != shape[0] + 1 or indptr[0] != 0 or indptr[-1] != values.size:
            reasons.append(
                "Sparse indptr endpoints or length do not match the matrix dimensions"
            )
            return result

    rows = np.linspace(0, shape[0] - 1, min(_VALIDATION_ROWS, shape[0]), dtype=np.int64)
    result.update(sampledRows=rows.tolist(), columnsChecked=shape[1])
    checked = 0
    finite, nonnegative, integral = True, True, True
    maximum = 0
    if progress is not None:
        progress(
            "validating_counts",
            completed=0,
            total=len(rows),
            unit="rows",
            message=f"{key}: sampling {len(rows)} cells across all {shape[1]} genes",
        )
    for completed, row in enumerate(rows, start=1):
        try:
            for block in _sample_row_blocks(
                values, int(row), shape[1], indices, indptr
            ):
                is_finite = np.isfinite(block)
                finite = finite and bool(np.all(is_finite))
                nonnegative = nonnegative and bool(np.all(block >= 0))
                if values.dtype.kind == "f":
                    integral = integral and bool(np.all(block == np.floor(block)))
                finite_values = block[is_finite]
                if finite_values.size:
                    maximum = max(maximum, finite_values.max().item())
                checked += int(block.size)
        except ValueError as error:
            reasons.append(str(error))
            return result
        if progress is not None:
            progress(
                "validating_counts",
                completed=completed,
                total=len(rows),
                unit="rows",
                storedValuesChecked=checked,
                message=f"{key}: sampling cells across all {shape[1]} genes",
            )
    if not finite:
        reasons.append("Sampled rows include NaN or infinity")
    if not nonnegative:
        reasons.append("Sampled rows include negative counts")
    if not integral:
        reasons.append("Sampled rows include non-integer counts")
    result.update(
        validationComplete=True,
        validCounts=not reasons,
        rowsChecked=len(rows),
        valuesChecked=checked,
        finite=finite,
        nonnegative=nonnegative,
        integerLike=finite and integral,
        sampleMax=maximum,
    )
    return result


def _select_counts(
    h5: h5py.File,
    matrices: dict,
    progress: Callable | None,
    raw_data_location: str | None = None,
) -> tuple[str, dict, dict | None]:
    candidates = ("raw/X", "X", "layers/counts", "layers/raw_counts")
    diagnostics: dict[str, dict[str, Any]] = {
        key: {
            "present": key in h5,
            "validationComplete": False,
            "validCounts": False,
            "reasons": ["Not evaluated"] if key in h5 else ["Matrix is absent"],
        }
        for key in candidates
    }

    def validate(key: str) -> bool:
        if key not in h5:
            return False
        diagnostics[key] = _validate_candidate(h5, key, matrices[key], progress)
        matrices[key] = diagnostics[key]
        return bool(diagnostics[key]["validCounts"])

    selected = "none"
    if raw_data_location is not None:
        selected = {"raw.X": "raw/X", "X": "X"}.get(raw_data_location, "none")
        if selected == "none":
            return (
                selected,
                diagnostics,
                {
                    "question": f"CELLxGENE raw_data_location is {raw_data_location!r}; expected 'X' or 'raw.X'. Correct the metadata before conversion.",
                    "options": ["X", "raw.X"],
                    "candidates": diagnostics,
                },
            )
        if not validate(selected):
            reasons = "; ".join(diagnostics[selected]["reasons"])
            return (
                "none",
                diagnostics,
                {
                    "question": f"CELLxGENE specifies {raw_data_location} for raw counts, but {selected} cannot be used: {reasons}. Correct the source or metadata before conversion.",
                    "options": [selected],
                    "candidates": diagnostics,
                },
            )
    else:
        for key in candidates[:2]:
            if validate(key):
                selected = key
                break
            if diagnostics[key].get("formatSupported") is False:
                return (
                    "none",
                    diagnostics,
                    {
                        "question": f"Preferred matrix {key} is unsupported: "
                        + "; ".join(diagnostics[key]["reasons"]),
                        "options": ["CSR H5AD", "Dense H5AD"],
                        "candidates": diagnostics,
                    },
                )
    if selected == "none":
        valid_layers = [key for key in candidates[2:] if validate(key)]
        if len(valid_layers) == 1:
            selected = valid_layers[0]
        elif len(valid_layers) == 2:
            return (
                selected,
                diagnostics,
                {
                    "question": "Both counts layers pass the sampled counts check. Select the intended layer explicitly before conversion.",
                    "options": valid_layers,
                    "candidates": diagnostics,
                },
            )
    if selected == "none":
        return (
            selected,
            diagnostics,
            {
                "question": "No supported candidate passes the sampled counts and dimension checks. Review the candidate reasons and provide a corrected H5AD.",
                "options": [key for key in candidates if key in h5],
                "candidates": diagnostics,
            },
        )
    return selected, diagnostics, None


def _column_info(node: h5py.Group | h5py.Dataset) -> dict[str, Any]:
    categorical = isinstance(node, h5py.Group) and "categories" in node
    values = node["categories"] if categorical else node
    if isinstance(values, h5py.Group):
        values = values.get("values")
    return {
        "dtype": str(values.dtype)
        if isinstance(values, h5py.Dataset)
        else "unsupported",
        "encoding": _json_safe(node.attrs.get("encoding-type")),
        "categoryCount": int(node["categories"].size) if categorical else None,
    }


def _column_values(node: h5py.Group | h5py.Dataset) -> np.ndarray:
    if isinstance(node, h5py.Dataset):
        return np.asarray(node[:])
    if "categories" in node and "codes" in node:
        codes = node["codes"][:]
        categories = node["categories"][:]
        if np.any(codes < -1) or np.any(codes >= len(categories)):
            raise ValueError(f"Invalid categorical codes in {node.name}")
        values = np.empty(codes.shape, dtype=object)
        values[:] = None
        present = codes >= 0
        values[present] = categories[codes[present]]
        return values
    if "values" in node and "mask" in node:
        values = np.asarray(node["values"][:]).astype(object)
        values[node["mask"][:]] = None
        return values
    raise ValueError(f"Unsupported metadata column encoding: {node.name}")


def _column_summary(node: h5py.Group | h5py.Dataset) -> dict[str, Any]:
    values = _column_values(node)
    info = _column_info(node)
    numeric = info["categoryCount"] is None and (
        np.issubdtype(np.dtype(info["dtype"]), np.number)
    )
    if numeric:
        numeric_values = np.asarray(values, dtype=np.float64)
        finite = numeric_values[np.isfinite(numeric_values)]
        return {
            "min": float(finite.min()) if finite.size else None,
            "max": float(finite.max()) if finite.size else None,
            "mean": float(finite.mean()) if finite.size else None,
            "missing": int(numeric_values.size - finite.size),
        }
    decoded = [_json_safe(value) for value in values]
    counts = Counter(value for value in decoded if value is not None)
    return {
        "uniqueCount": len(counts),
        "topValues": [
            {"value": value, "count": count} for value, count in counts.most_common(50)
        ],
        "missing": sum(value is None for value in decoded),
    }


def inspect_file(
    path: str | Path,
    raw_data_location: str | None = None,
    *,
    progress: Callable | None = None,
) -> dict[str, Any]:
    """Use the declared raw matrix, discovering candidates only when unspecified.

    Matrix validation and metadata export still run for a declared location;
    an invalid declaration produces needsInput instead of selecting another matrix.
    """
    with h5py.File(path, "r") as h5:
        matrix_paths = [key for key in ("X", "raw/X") if key in h5]
        matrix_paths.extend(f"layers/{key}" for key in h5.get("layers", {}))
        matrices = {key: _matrix_info(h5[key]) for key in matrix_paths}
        counts_location, diagnostics, needs_input = _select_counts(
            h5, matrices, progress, raw_data_location
        )
        if progress is not None:
            progress(
                "inspecting_metadata",
                message="Reading feature metadata, cell annotations and source embeddings",
            )
        selected = matrices.get(counts_location)
        feature_attrs = "raw/var" if counts_location == "raw/X" else "var"
        feature_table = h5.get(feature_attrs)
        feature_id = (
            str(_json_safe(feature_table.attrs.get("_index", "_index")))
            if feature_table is not None
            else "_index"
        )
        feature_name = feature_id
        scarf_inspection = None
        if selected is not None and needs_input is None:
            try:
                inspection = inspect_h5ad(str(path), matrix_key=counts_location)
                if inspection.featureAttrsKey != feature_attrs:
                    raise ValueError(
                        f"Scarf selected {inspection.featureAttrsKey} instead of {feature_attrs}"
                    )
                feature_id = inspection.featureIdsKey
                feature_name = (
                    "feature_name"
                    if "feature_name" in feature_table
                    else inspection.featureNameKey
                )
                if _column_length(feature_table[feature_name]) != selected["shape"][1]:
                    raise ValueError(
                        f"Feature-name column {feature_name!r} has unsupported dimensions"
                    )
                scarf_inspection = asdict(inspection)
                scarf_inspection.pop("h5adFn")
            except (ValueError, KeyError, TypeError) as error:
                needs_input = {
                    "question": f"Scarf cannot read selected counts {counts_location}: {error}. Confirm the matrix and feature metadata before conversion.",
                    "options": [counts_location],
                    "candidates": diagnostics,
                }
        if progress is not None:
            progress("inspecting_metadata")
        uns = _uns_value(h5["uns"]) if "uns" in h5 else {}
        embeddings = list(h5.get("obsm", {}))
        primary = (
            _column_values(h5["obs/is_primary_data"])
            if "obs/is_primary_data" in h5
            else np.array([], dtype=bool)
        )
        manifest = {
            "title": uns.get("title"),
            "citation": uns.get("citation"),
            "schemaVersion": uns.get("schema_version"),
            "organism": uns.get("organism"),
            "nObs": selected["shape"][0]
            if selected
            else (_table_length(h5.get("obs")) or 0),
            "nVars": selected["shape"][1]
            if selected
            else (_table_length(feature_table) or 0),
            "countsLocation": counts_location,
            "apiRawDataLocation": raw_data_location,
            "countsSelectionSource": "curation_api"
            if raw_data_location is not None
            else "inspection",
            "countsDtype": selected["dtype"] if selected else None,
            "countsIntegerLike": selected["integerLike"] if selected else None,
            "countsMax": None,
            "countsSampleMax": selected["sampleMax"] if selected else None,
            "countsValidationMode": "sampled_rows" if selected else None,
            "isPrimaryDataCounts": {
                "true": int(np.count_nonzero(primary == True)),  # noqa: E712
                "false": int(np.count_nonzero(primary == False)),  # noqa: E712
            },
            "featureIdKey": feature_id,
            "featureNameKey": feature_name,
            "featureAttrsKey": feature_attrs if selected else None,
            "selectionDiagnostics": diagnostics,
            "selectionNeedsInput": needs_input,
            "embeddings": embeddings,
        }
        h5ad_keys = {
            "scarfInspection": _json_safe(scarf_inspection),
            "apiRawDataLocation": raw_data_location,
            "matrices": matrices,
            **{
                key: {name: _column_info(node) for name, node in h5[key].items()}
                for key in ("obs", "var", "raw/var")
                if key in h5
            },
            **{key: list(h5.get(key, {})) for key in ("obsm", "varm", "obsp", "uns")},
        }
        obs_summary = {
            name: _column_summary(node)
            for name, node in h5["obs"].items()
            if name != "observation_joinid"
        }
    return {
        "manifest": manifest,
        "h5ad_keys": h5ad_keys,
        "obs_summary": obs_summary,
        "uns": uns,
    }


def _needs_input(record: dict, question: str, options: list[str]) -> dict:
    return record | {
        "status": "needsInput",
        "needsInput": {"question": question, "options": options},
    }


def _validated_manifest(value: dict) -> dict:
    """Require a current inspection result before consuming its count selection."""
    required = {
        "apiRawDataLocation",
        "countsSelectionSource",
        "featureAttrsKey",
        "selectionDiagnostics",
        "selectionNeedsInput",
        "countsValidationMode",
    }
    if missing := required - value.keys():
        raise ValueError(
            "Conversion requires a current inspection manifest; missing "
            + ", ".join(sorted(missing))
            + ". Inspect the source before conversion."
        )
    manifest = Manifest.model_validate(value).model_dump(mode="json")
    api_location = manifest["apiRawDataLocation"]
    selection_source = "inspection" if api_location is None else "curation_api"
    if manifest["countsSelectionSource"] != selection_source:
        raise ValueError(
            "Counts selection provenance disagrees with apiRawDataLocation"
        )
    key = manifest["countsLocation"]
    needs_input = manifest["selectionNeedsInput"]
    if api_location is not None and key != "none":
        if key != {"X": "X", "raw.X": "raw/X"}.get(api_location):
            raise ValueError("Selected counts disagree with apiRawDataLocation")
    if needs_input is not None:
        if not needs_input.get("question") or not isinstance(
            needs_input.get("options"), list
        ):
            raise ValueError("selectionNeedsInput must include a question and options")
        return manifest
    selected = manifest["selectionDiagnostics"].get(key, {})
    if not (
        selected.get("validationComplete") is True
        and selected.get("validCounts") is True
        and selected.get("formatSupported") is True
        and selected.get("validationMode") == "sampled_rows"
        and manifest["countsValidationMode"] == "sampled_rows"
    ):
        raise ValueError(
            "Conversion requires the current sampled-row counts check. Inspect the source before conversion."
        )
    feature_table = "raw/var" if key == "raw/X" else "var"
    if (
        key == "none"
        or selected.get("shape") != [manifest["nObs"], manifest["nVars"]]
        or manifest["featureAttrsKey"] != feature_table
        or selected.get("featureAttrsKey") != feature_table
    ):
        raise ValueError(
            "Selected matrix dimensions or feature table disagree with the manifest"
        )
    return manifest


def convert_local(
    source: Path,
    destination: Path,
    manifest: dict,
    *,
    progress: Callable | None = None,
) -> dict:
    """Convert the explicit count selection from a current, validated manifest."""
    manifest = _validated_manifest(manifest)
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Scarf destination already exists: {destination}")
    size = source.stat().st_size
    if size != manifest["sourceBytes"]:
        raise ValueError("Source byte size does not match the inspection manifest")
    if progress is not None:
        progress("verifying_source")
    with source.open("rb") as handle:
        checksum = hashlib.file_digest(handle, "sha256").hexdigest()
    if checksum != manifest["sourceSha256"]:
        raise ValueError("Source SHA-256 does not match the inspection manifest")

    record: dict[str, Any] = {
        "sourceFormat": "h5ad",
        "zarrPath": str(destination),
        "sourceBytes": size,
        "sourceSha256": checksum,
        "scarfVersion": scarf.__version__,
        "nObs": manifest["nObs"],
        "nVars": manifest["nVars"],
        "conversion": {
            "matrixKey": manifest["countsLocation"],
            "apiRawDataLocation": manifest.get("apiRawDataLocation"),
            "countsSelectionSource": manifest.get("countsSelectionSource"),
            "featureAttrsKey": manifest.get("featureAttrsKey"),
            "featureIdKey": manifest["featureIdKey"],
            "featureNameKey": manifest["featureNameKey"],
            "selectionDiagnostics": manifest["selectionDiagnostics"],
            "storageDtypePolicy": "preserve_source",
            "assayName": "RNA",
            "storageProfile": "cloud",
        },
    }
    if manifest.get("selectionNeedsInput") is not None:
        return record | {
            "status": "needsInput",
            "needsInput": manifest["selectionNeedsInput"],
        }

    try:
        inspection = scarf.inspect_h5ad(
            str(source), matrix_key=manifest["countsLocation"]
        )
    except (ValueError, KeyError, TypeError) as error:
        return _needs_input(
            record,
            f"Scarf cannot inspect selected counts: {error}. Confirm the matrix and feature metadata.",
            [manifest["countsLocation"]],
        )
    if inspection.matrixEncoding not in {"csr", "dense"}:
        return _needs_input(
            record,
            "Scarf materializes CSC inputs during conversion. Provide a CSR or dense H5AD for bounded conversion.",
            ["CSR H5AD", "Dense H5AD"],
        )
    if (inspection.nCells, inspection.nFeatures) != (
        manifest["nObs"],
        manifest["nVars"],
    ):
        raise ValueError("Source dimensions do not match the inspection manifest")
    if inspection.featureIdsKey != manifest["featureIdKey"]:
        return _needs_input(
            record,
            "Scarf and the source manifest disagree on feature IDs. Confirm the feature-ID column.",
            [inspection.featureIdsKey, manifest["featureIdKey"]],
        )
    if inspection.featureAttrsKey != manifest["featureAttrsKey"]:
        return _needs_input(
            record,
            "Scarf and the source manifest disagree on the selected feature table. Confirm the feature metadata.",
            [inspection.featureAttrsKey, manifest["featureAttrsKey"]],
        )
    with h5py.File(source, "r") as h5:
        source_matrix = h5[manifest["countsLocation"]]
        source_values = (
            source_matrix["data"]
            if isinstance(source_matrix, h5py.Group)
            else source_matrix
        )
        source_dtype = str(source_values.dtype)
        if source_dtype != manifest["countsDtype"]:
            raise ValueError("Source dtype does not match the inspection manifest")
        if manifest["featureNameKey"] not in h5[inspection.featureAttrsKey]:
            return _needs_input(
                record,
                "The selected feature-name column is missing. Confirm the feature-name column.",
                list(h5[inspection.featureAttrsKey]),
            )
        embedding_roles = {"X_umap": "umap"} if "obsm/X_umap" in h5 else {}
    record["conversion"]["embeddingRoles"] = embedding_roles
    record["conversion"]["scarfSuggestedFeatureNameKey"] = inspection.featureNameKey

    reader = scarf.H5adReader.from_inspect(
        inspection,
        feature_ids_key=manifest["featureIdKey"],
        feature_name_key=manifest["featureNameKey"],
        embedding_roles=embedding_roles,
        # Preserve losslessly without Scarf's separate full-matrix dtype scan.
        dtype=source_dtype,
    )
    writer = None
    try:
        if progress is not None:
            progress(
                "converting",
                message=f"Building RNA counts and countsT; preserving source dtype {source_dtype}",
            )
        writer = scarf.H5adToZarr(
            reader,
            zarr_loc=str(destination),
            assay_name="RNA",
            profile="cloud",
            nthreads=8,
            mem_budget="12G",
        )
        imported = writer.dump()
    finally:
        reader.h5.close()
        if writer is not None:
            writer.z.store.close()

    if progress is not None:
        progress("initializing_qc")
    datastore = scarf.DataStore(
        str(destination),
        default_assay="RNA",
        min_features_per_cell=-1,
        nthreads=8,
        mem_budget="12G",
    )
    try:
        record["qcSummary"] = datastore.summary().to_dict()
    finally:
        datastore.z.store.close()

    record.update(
        status="done",
        importedArtifacts={
            "cellSelection": imported.cellSelection.to_dict(),
            "embeddings": {
                key: ref.to_dict() for key, ref in imported.embeddingArtifacts.items()
            },
            "clusters": {
                key: ref.to_dict() for key, ref in imported.clusterArtifacts.items()
            },
        },
    )
    return record


def _open_verified_arrays(
    location: str,
    manifest: dict,
    storage_options: dict | None,
) -> tuple[zarr.Group, zarr.Array, zarr.Array, dict]:
    """Open only known paths and validate metadata without reading array values."""
    store = make_store(location, storage_options=storage_options, read_only=True)
    root = zarr.open_group(store=store, mode="r", use_consolidated=False)
    try:
        group = root["RNA"]
        counts, counts_t = root["RNA/counts"], root["RNA/countsT"]
        if (
            not isinstance(group, zarr.Group)
            or not isinstance(counts, zarr.Array)
            or not isinstance(counts_t, zarr.Array)
        ):
            raise ValueError("Scarf requires RNA/counts and RNA/countsT arrays")
        expected = (manifest["nObs"], manifest["nVars"])
        if tuple(counts.shape) != expected or tuple(counts_t.shape) != expected[::-1]:
            raise ValueError("Scarf count dimensions do not match the source manifest")
        if any(int(array.metadata.zarr_format) != 3 for array in (counts, counts_t)):
            raise ValueError("Scarf count arrays must use the paired Zarr v3 layout")
        if np.dtype(counts.dtype).kind not in "iuf" or np.dtype(
            counts_t.dtype
        ) != np.dtype(counts.dtype):
            raise ValueError(
                "Scarf counts and countsT must have matching numeric dtypes"
            )
        if counts_t.attrs.get("complete") is not True:
            raise ValueError("Scarf countsT is incomplete")
        for path, size in (
            ("cellData/ids", expected[0]),
            ("RNA/featureData/ids", expected[1]),
        ):
            ids = root[path]
            if not isinstance(ids, zarr.Array) or tuple(ids.shape) != (size,):
                raise ValueError(f"Scarf {path} dimensions do not match the source")
        plan = require_count_matrix_layout(group, counts, counts_t)
        metadata = {
            "nObs": expected[0],
            "nVars": expected[1],
            "countsDtype": str(np.dtype(counts.dtype)),
            "countsTShape": list(counts_t.shape),
            "countsTComplete": True,
            "layoutFingerprint": plan.fingerprint,
        }
        return root, counts, counts_t, metadata
    except BaseException:
        root.store.close()
        raise


def verify_store(
    location: str,
    manifest: dict,
    storage_options: dict | None = None,
    *,
    local_location: str | None = None,
) -> dict:
    """Compare fixed remote metadata and a 3 by 5 sample with the local store."""
    if local_location is None and is_remote_zarr_location(location):
        raise ValueError("Remote verification requires the local converted store")
    root, counts, counts_t, metadata = _open_verified_arrays(
        location, manifest, storage_options
    )
    local_root = None
    try:
        rows, columns = min(3, manifest["nObs"]), min(5, manifest["nVars"])
        block = np.asarray(counts[:rows, :columns])
        transposed = np.asarray(counts_t[:columns, :rows]).T
        if not np.array_equal(block, transposed):
            raise ValueError("Scarf counts and countsT samples do not agree")
        if local_location is not None:
            local_root, local_counts, local_counts_t, local_metadata = (
                _open_verified_arrays(local_location, manifest, None)
            )
            if metadata != local_metadata:
                raise ValueError(
                    "Published Scarf metadata differs from the local store"
                )
            local_block = np.asarray(local_counts[:rows, :columns])
            local_transposed = np.asarray(local_counts_t[:columns, :rows]).T
            if not np.array_equal(block, local_block) or not np.array_equal(
                block, local_transposed
            ):
                raise ValueError(
                    "Published Scarf counts differ from the local store sample"
                )
        return metadata | {
            "countsTMatches": True,
            "sourceSampleMatches": True,
            "countsBlock": block.tolist(),
        }
    finally:
        if local_root is not None:
            local_root.store.close()
        root.store.close()


def replacement_paths(record: DatasetRecord, storage: Bucket) -> list[str]:
    """Inventory generated output eligible for explicitly approved replacement."""
    prefix = f"{dataset_prefix(record.cytebaseId)}/"
    paths = []
    for path in storage.list_files(prefix):
        if not path.startswith(prefix):
            continue
        relative = path.removeprefix(prefix)
        if relative == "scarf_ingest.json" or relative.startswith(
            ("data.zarr/", "metadata/")
        ):
            paths.append(path)
    return sorted(set(paths))


def _source_details(record: DatasetRecord, size: int, checksum: str) -> None:
    record.sourceBytes = size
    for version in record.versions:
        if version.datasetVersionId == record.latestVersionId:
            version.sourceSha256 = checksum


def build_local(
    record: DatasetRecord,
    source: Path,
    store: Path,
    raw: dict,
    size: int,
    checksum: str,
    progress: Callable,
) -> tuple[Manifest, dict]:
    """Inspect and convert one verified source in a caller-owned workspace."""
    step = perf_counter()
    progress("inspecting", message="Inspecting H5AD structure and metadata")
    inspected = inspect_file(source, raw.get("raw_data_location"), progress=progress)
    fields = inspected["manifest"] | {
        "title": record.title,
        "citation": record.citation,
        "doi": record.doi,
        "schemaVersion": record.schemaVersion,
        "metadataSource": "curation_api",
        "organism": ", ".join(term.label for term in record.facets.get("organism", []))
        or None,
    }
    manifest = Manifest(
        **fields,
        collectionId=record.collectionId,
        datasetId=record.datasetId,
        datasetVersionId=record.latestVersionId,
        sourceUrl=record.sourceUrl,
        sourceBytes=size,
        sourceSha256=checksum,
        ingestedAt=datetime.now(UTC),
        pipelineVersion=record.pipelineVersion,
    )
    primary_count = manifest.isPrimaryDataCounts["true"]
    if primary_count == 0 or (
        record.primaryCellCount is not None and record.primaryCellCount != primary_count
    ):
        raise ValueError("H5AD primary-cell count disagrees with registration")
    record.timings["inspectSeconds"] = perf_counter() - step
    step = perf_counter()
    converted = convert_local(
        source, store, manifest.model_dump(mode="json"), progress=progress
    )
    record.timings["convertSeconds"] = perf_counter() - step
    converted.update(
        cytebaseId=record.cytebaseId,
        collectionId=str(record.collectionId),
        datasetId=str(record.datasetId),
        datasetVersionId=str(record.latestVersionId),
        pipelineVersion=record.pipelineVersion,
        sourcePath=record.sourceUrl,
        inspection={
            name: inspected[name] for name in ("h5ad_keys", "obs_summary", "uns")
        },
    )
    if converted["status"] not in {"done", "needsInput"}:
        raise RuntimeError("Scarf conversion did not complete")
    return manifest, converted


def publish_store(
    record: DatasetRecord,
    request: dict,
    storage: Bucket,
    store: Path,
    manifest: Manifest,
    converted: dict,
    progress: Callable,
    assert_owner: Callable[[], None],
) -> dict:
    """Publish verified local output and commit readiness before local cleanup."""
    prefix = dataset_prefix(record.cytebaseId)
    version_id = str(record.latestVersionId)
    zarr_uri = f"{storage.root}/{prefix}/data.zarr"
    converted["zarrPath"] = zarr_uri
    assert_owner()
    replacements = replacement_paths(record, storage)
    if converted["status"] == "needsInput":
        if not replacements:
            progress(
                "uploading_metadata", message="Preserving inspection and provenance"
            )
            assert_owner()
            storage.write_json(f"{prefix}/scarf_ingest.json", converted)
        record.inspection = manifest
        _source_details(record, manifest.sourceBytes, manifest.sourceSha256)
        record.status = "needsInput"
        record.needsInput = converted.get("needsInput")
        return {
            "outcome": "needsInput",
            "message": "Counts conversion needs a user decision",
        }
    approved = set(request.get("approvedDeletionPaths", []))
    missing = sorted(set(replacements) - approved)
    if missing:
        return {
            "outcome": "needsApproval",
            "message": "Review these exact generated paths and resubmit with approvedDeletionPaths",
            "deletionPaths": missing,
        }
    # Commit the unavailable state before replacing a previously ready store.
    record.status = "processing"
    record.processedVersionId = None
    record.processedAt = None
    record.zarrUri = None
    record.buildReceipt = None
    record.updatedAt = datetime.now(UTC)
    assert_owner()
    storage.write_json(f"{prefix}/dataset.json", record.model_dump(mode="json"))
    assert_owner()
    replacements = replacement_paths(record, storage)
    missing = sorted(set(replacements) - approved)
    if missing:
        return {
            "outcome": "needsApproval",
            "message": "Generated output changed during publication; review these exact paths and resubmit with approvedDeletionPaths",
            "deletionPaths": missing,
        }
    if replacements:
        progress("replacing", message="Removing explicitly approved generated files")
        assert_owner()
        storage.delete_exact(replacements)
    step = perf_counter()
    progress("uploading_store", message="Uploading Scarf Zarr hierarchy")
    assert_owner()
    storage.sync_store(store, record.cytebaseId)
    record.timings["uploadSeconds"] = perf_counter() - step
    step = perf_counter()
    progress(
        "verifying_store",
        message="Comparing published counts with the local store sample",
    )
    converted["verification"] = retry(
        lambda: verify_store(
            zarr_uri,
            manifest.model_dump(mode="json"),
            storage_options={"token": storage.token, "skip_instance_cache": True},
            local_location=str(store),
        ),
        progress=progress,
    )
    record.timings["verifySeconds"] = perf_counter() - step
    completed = datetime.now(UTC)
    converted["completedAt"] = completed.isoformat()
    assert_owner()
    storage.write_json(f"{prefix}/scarf_ingest.json", converted)
    record.processedVersionId = record.latestVersionId
    record.processedAt = completed
    record.zarrUri = zarr_uri
    record.h5adUri = None
    record.inspection = manifest
    record.cellCount, record.nGenes = manifest.nObs, manifest.nVars
    record.primaryCellCount = manifest.isPrimaryDataCounts["true"]
    record.needsInput = None
    _source_details(record, manifest.sourceBytes, manifest.sourceSha256)
    for version in record.versions:
        if version.datasetVersionId == record.latestVersionId:
            version.processedAt = completed
    record.buildReceipt = {
        "datasetVersionId": version_id,
        "sourceSha256": manifest.sourceSha256,
        "zarrUri": zarr_uri,
        "verifiedAt": completed.isoformat(),
        "verification": converted["verification"],
    }
    ready = record.model_copy(
        update={"status": "ready", "stageOutcome": "succeeded", "updatedAt": completed}
    )
    assert_owner()
    storage.write_json(f"{prefix}/dataset.json", ready.model_dump(mode="json"))
    record.status, record.stageOutcome, record.updatedAt = (
        ready.status,
        ready.stageOutcome,
        ready.updatedAt,
    )
    return {"outcome": "succeeded"}

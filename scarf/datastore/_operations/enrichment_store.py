import hashlib
import json
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import zarr

from ...assay import Assay
from ...features.enrichment.results import EnrichmentResult
from ...graph.arguments import OperationArguments
from ...storage.artifact_writer import ArrayRequirement
from ...storage.artifacts import (
    ArtifactRef,
    ArtifactStatus,
    callable_identity,
    inspect_artifact,
)
from ...storage.feature_selection import resolve_feature_selection
from ...storage.types import as_zarr_array, as_zarr_group
from ...utils.arrays import array_digest


_ENRICHMENT_LAYOUT = "cells_by_sources"
_ENRICHMENT_ACTIVE_SLOT = "_active_slot"
_ENRICHMENT_RUN_PREFIX = "_run_"
_ENRICHMENT_ARTIFACT_RESULTS = "artifact_results"
_ENRICHMENT_LEGACY_ARTIFACTS = "artifacts"


@dataclass(frozen=True, slots=True)
class _EnrichmentScorer:
    """Method-specific enrichment scoring inputs for the shared driver."""

    method: str
    algorithm_version: int
    method_payload: dict[str, Any]
    arguments: OperationArguments
    cell_index: np.ndarray
    feature_index: np.ndarray
    matched_feature_index: np.ndarray
    source_names: np.ndarray
    source_sizes: np.ndarray
    rank_feature_index: np.ndarray | None
    extra_required_arrays: tuple[ArrayRequirement, ...]
    score_batches: Callable[[], Iterator[np.ndarray]]
    write_context: Callable[[], AbstractContextManager[None]] = nullcontext


def _validate_enrichment_label(label: str) -> str:
    if not isinstance(label, str):
        raise TypeError("Enrichment label must be a string")
    if not label or label in {".", ".."}:
        raise ValueError("Enrichment label must be a non-empty path component")
    if "/" in label or "\\" in label or any(ord(char) < 32 for char in label):
        raise ValueError(
            "Enrichment label must not contain path separators or control characters"
        )
    return label


def _execution_digest(payload: dict[str, Any]) -> str:
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Enrichment execution metadata must be JSON-safe") from exc
    return hashlib.blake2b(encoded, digest_size=16).hexdigest()


def _resolve_enrichment_slot(
    label_group: zarr.Group,
    *,
    label: str,
) -> zarr.Group:
    active_slot = label_group.attrs.get(_ENRICHMENT_ACTIVE_SLOT)
    if active_slot is None:
        return label_group
    if (
        not isinstance(active_slot, str)
        or not active_slot.startswith(_ENRICHMENT_RUN_PREFIX)
        or "/" in active_slot
        or active_slot not in label_group
    ):
        raise ValueError(f"Enrichment slot {label!r} has an invalid active result")
    return as_zarr_group(
        label_group[active_slot],
        name=f"{label}/{active_slot}",
    )


def _enrichment_artifact_entry(
    assay: Assay,
    label: str,
) -> tuple[ArtifactRef, str] | None:
    if "enrichment" not in assay.z:
        return None
    enrichment_group = as_zarr_group(
        assay.z["enrichment"],
        name=f"{assay.name}/enrichment",
    )
    raw_results = enrichment_group.attrs.get(_ENRICHMENT_ARTIFACT_RESULTS)
    if raw_results is not None:
        if not isinstance(raw_results, dict):
            raise ValueError("Enrichment artifact index is invalid")
        raw_entry = raw_results.get(label)
        if raw_entry is not None:
            if not isinstance(raw_entry, dict):
                raise ValueError(
                    f"Enrichment label {label!r} has an invalid artifact entry"
                )
            raw_ref = raw_entry.get("artifact")
            cell_key = raw_entry.get("cell_key")
            if (
                set(raw_entry) != {"artifact", "cell_key"}
                or not isinstance(raw_ref, dict)
                or not isinstance(cell_key, str)
                or not cell_key
            ):
                raise ValueError(
                    f"Enrichment label {label!r} has invalid execution metadata"
                )
            return ArtifactRef.from_dict(raw_ref), cell_key
    return None


def _enrichment_artifact_ref(
    assay: Assay,
    label: str,
) -> ArtifactRef | None:
    entry = _enrichment_artifact_entry(assay, label)
    return None if entry is None else entry[0]


def _legacy_enrichment_slot(
    assay: Assay,
    label: str,
) -> zarr.Group | None:
    if "enrichment" not in assay.z:
        return None
    enrichment_group = as_zarr_group(
        assay.z["enrichment"],
        name=f"{assay.name}/enrichment",
    )
    if label not in enrichment_group:
        return None
    label_group = as_zarr_group(
        enrichment_group[label],
        name=f"{assay.name}/enrichment/{label}",
    )
    return _resolve_enrichment_slot(label_group, label=label)


def _publish_enrichment_artifact(
    assay: Assay,
    label: str,
    ref: ArtifactRef,
    *,
    cell_key: str,
) -> None:
    if "enrichment" not in assay.z:
        assay.z.create_group("enrichment")
    enrichment_group = as_zarr_group(
        assay.z["enrichment"],
        name=f"{assay.name}/enrichment",
    )
    raw_results = enrichment_group.attrs.get(_ENRICHMENT_ARTIFACT_RESULTS, {})
    if not isinstance(raw_results, dict):
        raise ValueError("Enrichment artifact index is invalid")
    results = dict(raw_results)
    results[label] = {
        "artifact": ref.to_dict(),
        "cell_key": cell_key,
    }
    enrichment_group.attrs[_ENRICHMENT_ARTIFACT_RESULTS] = results


def _write_enrichment_slot(
    slot: zarr.Group,
    *,
    attrs: dict[str, Any],
    score_batches: Iterator[np.ndarray],
    n_cells: int,
    source_names: np.ndarray,
    source_sizes: np.ndarray,
    cell_index: np.ndarray,
    matched_feature_index: np.ndarray,
    rank_feature_index: np.ndarray | None,
) -> None:
    from ...storage.arrays import create_metadata_column, create_numeric_array
    from ...storage.layout import normed_array_spec
    from ...storage.profiles import resolve_storage_profile
    from ...storage.sharding import write_dense_from_row_batches

    names = np.asarray(source_names)
    sizes = np.asarray(source_sizes, dtype=np.int64)
    cells = np.asarray(cell_index, dtype=np.int64)
    matched = np.asarray(matched_feature_index, dtype=np.int64)
    rank = (
        None
        if rank_feature_index is None
        else np.asarray(rank_feature_index, dtype=np.int64)
    )
    n_sources = len(names)
    if n_cells < 1 or len(cells) != n_cells:
        raise ValueError("Enrichment cell index is empty or misaligned")
    if n_sources < 1 or len(sizes) != n_sources:
        raise ValueError("Enrichment source metadata is empty or misaligned")
    if len(matched) < 1:
        raise ValueError("Enrichment has no matched features")
    json.dumps(attrs, sort_keys=True, allow_nan=False)

    slot.attrs["complete"] = False
    for key, value in attrs.items():
        slot.attrs[key] = value
    try:
        create_metadata_column(
            slot,
            "cell_index",
            data=cells,
            dtype=np.int64,
            chunkSize=100_000,
        )
        create_metadata_column(
            slot,
            "matched_feature_index",
            data=matched,
            dtype=np.int64,
            chunkSize=100_000,
        )
        if rank is not None:
            create_metadata_column(
                slot,
                "rank_feature_index",
                data=rank,
                dtype=np.int64,
                chunkSize=100_000,
            )
        create_metadata_column(
            slot,
            "source_names",
            data=names,
            chunkSize=100_000,
        )
        create_metadata_column(
            slot,
            "source_sizes",
            data=sizes,
            dtype=np.int64,
            chunkSize=100_000,
        )
        scores = create_numeric_array(
            slot,
            "scores",
            normed_array_spec(
                n_cells,
                n_sources,
                profile=resolve_storage_profile(slot.store),
            ),
        )

        def checked_batches() -> Iterator[np.ndarray]:
            for batch in score_batches:
                values = np.asarray(batch, dtype=np.float64)
                if values.ndim != 2 or values.shape[1] != n_sources:
                    raise ValueError("Enrichment score batch has an invalid shape")
                if not np.isfinite(values).all():
                    raise ValueError(
                        "Enrichment score batch contains non-finite values"
                    )
                yield values

        written = write_dense_from_row_batches(
            scores,
            checked_batches(),
            dtype=np.float32,
            msg=f"Writing {attrs['method']} enrichment",
        )
        if written != n_cells:
            raise ValueError(
                f"Enrichment writer produced {written} rows, expected {n_cells}"
            )
        slot.attrs["complete"] = True
    except Exception:
        slot.attrs["complete"] = False
        raise


def _enrichment_artifact_matches(
    group: zarr.Group,
    *,
    attrs: dict[str, Any],
    cell_index: np.ndarray,
    matched_feature_index: np.ndarray,
    source_names: np.ndarray,
    source_sizes: np.ndarray,
    rank_feature_index: np.ndarray | None,
) -> bool:
    for key, expected in attrs.items():
        if key in {"cell_key", "complete"}:
            continue
        if group.attrs.get(key) != expected:
            return False
    try:
        scores = as_zarr_array(group["scores"], name="scores")
    except (KeyError, TypeError):
        return False
    if (
        scores.ndim != 2
        or scores.shape != (len(cell_index), len(source_names))
        or np.dtype(scores.dtype) != np.dtype(np.float32)
    ):
        return False
    expected_arrays = {
        "cell_index": np.asarray(cell_index, dtype=np.int64),
        "matched_feature_index": np.asarray(
            matched_feature_index,
            dtype=np.int64,
        ),
        "source_names": np.asarray(source_names).astype(str),
        "source_sizes": np.asarray(source_sizes, dtype=np.int64),
    }
    if rank_feature_index is not None:
        expected_arrays["rank_feature_index"] = np.asarray(
            rank_feature_index,
            dtype=np.int64,
        )
    try:
        for name, expected in expected_arrays.items():
            stored = np.asarray(as_zarr_array(group[name], name=name)[:])
            if name == "source_names":
                stored = stored.astype(str)
            if not np.array_equal(stored, expected):
                return False
    except (KeyError, TypeError, ValueError):
        return False
    if rank_feature_index is None and "rank_feature_index" in group:
        return False
    return True


def _validate_enrichment_artifact_provenance(
    root: zarr.Group,
    assay: Assay,
    status: ArtifactStatus,
    group: zarr.Group,
    method: str,
    cell_key: str,
) -> tuple[str, ArtifactRef]:
    parameters = status.parameters or {}
    inputs = status.inputs or {}
    execution = status.execution_options or {}
    expected_parameters: dict[str, Any] = {
        "algorithm_version": group.attrs["algorithm_version"],
        "tmin": group.attrs["tmin"],
    }
    if method == "waggr":
        normalization_method = parameters.get("normalization_method")
        if normalization_method != callable_identity(assay.normMethod):
            raise ValueError("Enrichment artifact normalization provenance is invalid")
        expected_parameters.update(
            {
                "log_transform": group.attrs["log_transform"],
                "mode": group.attrs["waggr_mode"],
                "size_factor": group.attrs["size_factor"],
            }
        )
    else:
        expected_parameters.update(
            {
                "n_up": group.attrs["n_up"],
                "tie_seed": group.attrs["tie_seed"],
            }
        )
    if any(parameters.get(key) != value for key, value in expected_parameters.items()):
        raise ValueError("Enrichment artifact parameters do not match its metadata")
    if inputs.get("network_digest") != group.attrs["network_digest"]:
        raise ValueError(
            "Enrichment artifact network input does not match its metadata"
        )
    if execution.get("cell_key") != group.attrs["cell_key"]:
        raise ValueError("Enrichment artifact cell selection does not match metadata")
    if "feat_key" in execution or "feat_key" in group.attrs:
        raise ValueError("Enrichment artifact uses the legacy feature contract")

    raw_cell_selection = inputs.get("cell_selection")
    raw_feature_selection = inputs.get("feature_selection")
    if not isinstance(raw_cell_selection, dict) or not isinstance(
        raw_feature_selection, dict
    ):
        raise ValueError("Enrichment artifact is missing selection provenance")
    cell_selection = ArtifactRef.from_dict(raw_cell_selection)
    feature_selection = ArtifactRef.from_dict(raw_feature_selection)

    from ...graph.state import validate_cell_selection_artifact

    validate_cell_selection_artifact(root, cell_selection, cell_key)
    feature_selection = resolve_feature_selection(
        root,
        assay.name,
        feature_selection,
    )
    for selection_ref, digest_name in (
        (cell_selection, "cell_digest"),
        (feature_selection, "feature_digest"),
    ):
        selection_status = inspect_artifact(root, selection_ref)
        selection_group = as_zarr_group(
            root[selection_status.path],
            name=selection_status.path,
        )
        selection_values = np.asarray(
            as_zarr_array(selection_group["values"], name="values")[:]
        )
        selected_index = np.flatnonzero(selection_values).astype(np.int64)
        if array_digest(selected_index) != group.attrs[digest_name]:
            raise ValueError(
                f"Enrichment artifact {selection_ref.kind!r} input "
                "does not match its metadata"
            )
    return cell_key, feature_selection


def _load_enrichment_result(
    assay: Assay,
    *,
    label: str,
    sources: Sequence[str] | None,
    artifact_root: zarr.Group | None = None,
) -> EnrichmentResult:
    from ...matrix import ChunkedArray

    artifact_entry = _enrichment_artifact_entry(assay, label)
    if artifact_entry is None:
        raise KeyError(f"Enrichment label {label!r} was not found for {assay.name}")
    if artifact_root is None:
        raise ValueError("Artifact root is required for indexed enrichment")
    ref, indexed_cell_key = artifact_entry
    status = inspect_artifact(artifact_root, ref)
    if (
        ref.kind != "enrichment_scores"
        or ref.scope != "assay"
        or ref.assay != assay.name
        or not status.complete
    ):
        raise ValueError(f"Enrichment label {label!r} has an invalid artifact")
    slot = as_zarr_group(
        artifact_root[status.path],
        name=status.path,
    )
    root_path = str(getattr(artifact_root, "path", "")).strip("/")
    storage_path = f"{root_path}/{status.path}" if root_path else status.path
    if slot.attrs.get("complete") is not True:
        raise ValueError(f"Enrichment slot {label!r} is incomplete")
    method = str(slot.attrs.get("method", ""))
    if method not in {"waggr", "aucell"}:
        raise ValueError(f"Enrichment slot {label!r} has an unknown method")
    if status.operation != f"run_{method}":
        raise ValueError(
            f"Enrichment label {label!r} has a mismatched artifact operation"
        )
    required_attrs = {
        "algorithm_version",
        "cell_digest",
        "cell_key",
        "feature_digest",
        "network_digest",
        "tmin",
    }
    if not required_attrs.issubset(slot.attrs):
        raise ValueError(f"Enrichment slot {label!r} is missing required metadata")
    method_attrs = (
        {"log_transform", "normalization", "size_factor", "waggr_mode"}
        if method == "waggr"
        else {"n_up", "tie_seed"}
    )
    if not method_attrs.issubset(slot.attrs):
        raise ValueError(f"Enrichment slot {label!r} is missing method metadata")
    algorithm_version = slot.attrs["algorithm_version"]
    if (
        isinstance(algorithm_version, bool)
        or not isinstance(algorithm_version, (int, np.integer))
        or int(algorithm_version) != 1
    ):
        raise ValueError(f"Enrichment slot {label!r} has an unsupported algorithm")
    tmin = slot.attrs["tmin"]
    if (
        isinstance(tmin, bool)
        or not isinstance(tmin, (int, np.integer))
        or int(tmin) < 1
    ):
        raise ValueError(f"Enrichment slot {label!r} has invalid tmin metadata")
    for digest_name in (
        "cell_digest",
        "feature_digest",
        "network_digest",
    ):
        digest_value = slot.attrs[digest_name]
        if not isinstance(digest_value, str) or not digest_value:
            raise ValueError(
                f"Enrichment slot {label!r} has invalid {digest_name} metadata"
            )
    key_value = slot.attrs["cell_key"]
    if not isinstance(key_value, str) or not key_value:
        raise ValueError(f"Enrichment slot {label!r} has invalid cell_key metadata")
    stored_n_up: int | None = None
    if method == "waggr":
        size_factor = slot.attrs["size_factor"]
        if (
            isinstance(size_factor, bool)
            or not isinstance(size_factor, (int, float, np.integer, np.floating))
            or not np.isfinite(float(size_factor))
            or float(size_factor) <= 0
            or slot.attrs["normalization"] != "norm_lib_size"
            or slot.attrs["waggr_mode"] not in {"wmean", "wsum"}
            or not isinstance(slot.attrs["log_transform"], bool)
        ):
            raise ValueError(f"WAGGR slot {label!r} has invalid method metadata")
    else:
        n_up = slot.attrs["n_up"]
        tie_seed = slot.attrs["tie_seed"]
        if (
            isinstance(n_up, bool)
            or not isinstance(n_up, (int, np.integer))
            or int(n_up) < 2
            or isinstance(tie_seed, bool)
            or not isinstance(tie_seed, (int, np.integer))
            or int(tie_seed) < 0
        ):
            raise ValueError(f"AUCell slot {label!r} has invalid method metadata")
        stored_n_up = int(n_up)
    resolved_cell_key, feature_selection = _validate_enrichment_artifact_provenance(
        artifact_root,
        assay,
        status,
        slot,
        method,
        indexed_cell_key,
    )

    required_arrays = {
        "cell_index",
        "matched_feature_index",
        "scores",
        "source_names",
        "source_sizes",
    }
    if not required_arrays.issubset(slot):
        raise ValueError(f"Enrichment slot {label!r} is missing required arrays")
    if method == "aucell" and "rank_feature_index" not in slot:
        raise ValueError(f"AUCell slot {label!r} is missing its ranking universe")
    if method == "waggr" and "rank_feature_index" in slot:
        raise ValueError(f"WAGGR slot {label!r} contains unexpected rank metadata")

    scores = as_zarr_array(slot["scores"], name=f"{storage_path}/scores")
    cell_node = as_zarr_array(slot["cell_index"], name="cell_index")
    matched_node = as_zarr_array(
        slot["matched_feature_index"],
        name="matched_feature_index",
    )
    names_node = as_zarr_array(slot["source_names"], name="source_names")
    sizes_node = as_zarr_array(slot["source_sizes"], name="source_sizes")
    sidecars = (cell_node, matched_node, names_node, sizes_node)
    if any(node.ndim != 1 for node in sidecars) or scores.ndim != 2:
        raise ValueError(f"Enrichment slot {label!r} contains invalid array dimensions")
    if np.dtype(scores.dtype) != np.dtype(np.float32):
        raise ValueError(f"Enrichment slot {label!r} has an invalid score dtype")
    if not np.issubdtype(cell_node.dtype, np.integer) or not np.issubdtype(
        matched_node.dtype, np.integer
    ):
        raise ValueError(f"Enrichment slot {label!r} has invalid index dtypes")
    if not np.issubdtype(sizes_node.dtype, np.integer):
        raise ValueError(f"Enrichment slot {label!r} has invalid source sizes")

    cell_index = np.asarray(cell_node[:], dtype=np.int64)
    matched_feature_index = np.asarray(matched_node[:], dtype=np.int64)
    source_names = np.asarray(names_node[:]).astype(str)
    source_sizes = np.asarray(sizes_node[:], dtype=np.int64)
    if scores.shape != (len(cell_index), len(source_names)):
        raise ValueError(f"Enrichment slot {label!r} score shape is misaligned")
    if len(source_names) == 0 or len(source_names) != len(source_sizes):
        raise ValueError(f"Enrichment slot {label!r} source metadata is misaligned")
    if np.unique(source_names).size != len(source_names):
        raise ValueError(f"Enrichment slot {label!r} contains duplicate sources")
    if np.any(source_names == ""):
        raise ValueError(f"Enrichment slot {label!r} contains empty source names")
    if np.any(source_sizes <= 0):
        raise ValueError(f"Enrichment slot {label!r} contains invalid source sizes")
    if np.any(cell_index < 0) or np.unique(cell_index).size != len(cell_index):
        raise ValueError(f"Enrichment slot {label!r} contains duplicate cell indices")
    if array_digest(cell_index) != slot.attrs["cell_digest"]:
        raise ValueError(f"Enrichment slot {label!r} has a mismatched cell digest")
    if (
        len(matched_feature_index) == 0
        or np.any(matched_feature_index < 0)
        or not np.array_equal(matched_feature_index, np.unique(matched_feature_index))
    ):
        raise ValueError(f"Enrichment slot {label!r} has invalid matched features")
    if method == "aucell":
        rank_node = as_zarr_array(
            slot["rank_feature_index"],
            name="rank_feature_index",
        )
        if rank_node.ndim != 1 or not np.issubdtype(rank_node.dtype, np.integer):
            raise ValueError(f"AUCell slot {label!r} has invalid rank features")
        rank_feature_index = np.asarray(rank_node[:], dtype=np.int64)
        if (
            len(rank_feature_index) < 2
            or np.any(rank_feature_index < 0)
            or np.unique(rank_feature_index).size != len(rank_feature_index)
            or stored_n_up is None
            or stored_n_up > len(rank_feature_index)
        ):
            raise ValueError(f"AUCell slot {label!r} has invalid rank features")
        if array_digest(np.sort(rank_feature_index)) != slot.attrs["feature_digest"]:
            raise ValueError(f"AUCell slot {label!r} has a mismatched feature digest")
        if not np.isin(matched_feature_index, rank_feature_index).all():
            raise ValueError(f"AUCell slot {label!r} has unmatched network features")

    data = ChunkedArray(scores, nthreads=assay.nthreads)
    if sources is not None:
        if isinstance(sources, str):
            raise TypeError("sources must be a sequence of source names, not a string")
        requested = list(sources)
        if not requested:
            raise ValueError("sources must be non-empty when provided")
        if not all(isinstance(source, str) for source in requested):
            raise TypeError("sources must contain only strings")
        if len(set(requested)) != len(requested):
            raise ValueError("sources contains duplicate names")
        source_positions = {
            source: index for index, source in enumerate(source_names.tolist())
        }
        missing = [source for source in requested if source not in source_positions]
        if missing:
            raise KeyError("Enrichment sources not found: " + ", ".join(missing))
        positions = np.asarray(
            [source_positions[source] for source in requested],
            dtype=np.int64,
        )
        data = data[:, positions]
        source_names = source_names[positions]
        source_sizes = source_sizes[positions]

    return EnrichmentResult(
        data=data,
        source_names=source_names,
        source_sizes=source_sizes,
        cell_index=cell_index,
        label=label,
        storage_path=storage_path,
        assay=assay.name,
        cell_key=resolved_cell_key,
        feature_selection=feature_selection,
        method=method,
    )

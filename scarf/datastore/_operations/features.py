import hashlib
import json
import time
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import zarr
from numpy.typing import NDArray

from ...storage.types import as_zarr_array, as_zarr_group
from ...storage.artifact_writer import (
    ArrayRequirement,
    AttributeRequirement,
    finish_artifact,
    plan_artifact,
    start_artifact,
)
from ...storage.artifacts import (
    ArtifactRef,
    ArtifactStatus,
    artifact_path,
    callable_identity,
    inspect_artifact,
)
from ...storage.selections import resolve_selection_artifact
from ...assay import Assay, RNAassay, lib_size_feature_stream_eligible
from ...features.enrichment.results import EnrichmentResult
from ...features.markers import resolve_marker_gene_batch_size, sort_marker_results
from ...metadata.arguments import AucellArguments, MarkerTableArguments, WaggrArguments
from ...metadata.artifacts import (
    categorical_display,
    feature_column_display,
    link_feature_data_column,
)
from ...utils.arrays import array_digest
from ...utils.compute import controlled_compute
from ...utils.logging import logger
from ...utils.progress import tqdmbar

if TYPE_CHECKING:
    from ..mapping_datastore import MappingDatastore as _FeatureOperationsBase
else:
    _FeatureOperationsBase = object


_MARKER_STAT_COLUMNS = (
    "score",
    "mean",
    "mean_rest",
    "frac_exp",
    "frac_exp_rest",
    "fold_change",
    "p_value",
)
_MARKER_OUT_COLUMNS = ("feature_index", *_MARKER_STAT_COLUMNS)
_ENRICHMENT_LAYOUT = "cells_by_sources"
_ENRICHMENT_ACTIVE_SLOT = "_active_slot"
_ENRICHMENT_RUN_PREFIX = "_run_"
_ENRICHMENT_ARTIFACT_RESULTS = "artifact_results"
_ENRICHMENT_LEGACY_ARTIFACTS = "artifacts"


def _feature_column_chunk(assay: Assay, n_features: int) -> int:
    # RNA feature-column streams (markers, HVG, pseudotime) prefer countsT
    # when present; other assays keep cell-major batch sizing.
    if isinstance(assay, RNAassay):
        counts_t = getattr(assay, "rawDataT", None)
        if counts_t is not None:
            chunks = getattr(counts_t, "chunks", None)
            if chunks and len(chunks) > 0:
                return max(1, int(chunks[0]))
    backing = getattr(assay.rawData, "_backing", None)
    chunks = getattr(backing, "chunks", None)
    if chunks and len(chunks) > 1:
        return max(1, int(chunks[1]))
    return max(1, int(n_features))


def _shared_marker_feature_index(markers: dict[Any, pd.DataFrame]) -> np.ndarray:
    for vals in markers.values():
        if len(vals) != 0:
            return np.sort(np.asarray(vals.index.values, dtype=np.int32))
    raise ValueError("Cannot save empty marker results")


def _marker_stats_matrix(vals: pd.DataFrame, feature_index: np.ndarray) -> np.ndarray:
    aligned = vals.reindex(feature_index)
    return np.asarray(
        aligned.loc[:, list(_MARKER_STAT_COLUMNS)].to_numpy(dtype=np.float64)
    )


def _write_compact_marker_stats(
    cluster_group: zarr.Group,
    stats: np.ndarray,
) -> None:
    from ...storage.arrays import create_numeric_array
    from ...storage.layout import ZarrArraySpec

    n_features = int(stats.shape[0])
    spec = ZarrArraySpec(
        shape=(n_features, len(_MARKER_STAT_COLUMNS)),
        chunks=(n_features, len(_MARKER_STAT_COLUMNS)),
        dtype="float64",
        overwrite=True,
    )
    arr = create_numeric_array(cluster_group, "stats", spec)
    arr[:] = stats


def _load_marker_cluster_frame(
    slot_group: zarr.Group,
    cluster_group: zarr.Group,
    feature_names: np.ndarray,
    *,
    group_id: Any,
) -> pd.DataFrame:
    out_cols = list(_MARKER_OUT_COLUMNS)
    if "feature_index" in slot_group and "stats" in cluster_group:
        feature_index = np.asarray(
            as_zarr_array(slot_group["feature_index"], name="feature_index")[:]
        )
        stats = np.asarray(as_zarr_array(cluster_group["stats"], name="stats")[:])
        df = pd.DataFrame(stats, columns=list(_MARKER_STAT_COLUMNS))
        df["feature_index"] = feature_index
        df["feature_name"] = feature_names[feature_index.astype(int)]
        df["group_id"] = group_id
        return sort_marker_results(df[["group_id", "feature_name", *out_cols[1:]]])

    available_cols = [col for col in out_cols if col in cluster_group]
    if not available_cols:
        return pd.DataFrame([[] for _ in out_cols], index=out_cols).T
    cols = [
        np.asarray(as_zarr_array(cluster_group[x], name=x)[:]) for x in available_cols
    ]
    df = pd.DataFrame(cols, index=available_cols).T
    df["group_id"] = group_id
    df["feature_name"] = feature_names[df.feature_index.astype("int")]
    return df[["group_id", "feature_name", *available_cols[1:]]]


def _group_assignment_digest(values: np.ndarray) -> str:
    return array_digest(np.asarray(values).astype(str))


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
) -> tuple[ArtifactRef, str | None, str | None] | None:
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
            feat_key = raw_entry.get("feat_key")
            if (
                not isinstance(raw_ref, dict)
                or not isinstance(cell_key, str)
                or not cell_key
                or not isinstance(feat_key, str)
                or not feat_key
            ):
                raise ValueError(
                    f"Enrichment label {label!r} has invalid execution metadata"
                )
            return ArtifactRef.from_dict(raw_ref), cell_key, feat_key
    raw_artifacts = enrichment_group.attrs.get(_ENRICHMENT_LEGACY_ARTIFACTS)
    if raw_artifacts is None:
        return None
    if not isinstance(raw_artifacts, dict):
        raise ValueError("Enrichment artifact index is invalid")
    raw_ref = raw_artifacts.get(label)
    if raw_ref is None:
        return None
    if not isinstance(raw_ref, dict):
        raise ValueError(f"Enrichment label {label!r} has an invalid artifact ref")
    return ArtifactRef.from_dict(raw_ref), None, None


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
    feat_key: str,
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
        "feat_key": feat_key,
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
            normed_array_spec(n_cells, n_sources),
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
        if key in {"cell_key", "complete", "feat_key"}:
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
    feat_key: str,
) -> tuple[str, str]:
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
    if (
        execution.get("cell_key") != group.attrs["cell_key"]
        or execution.get("feat_key") != group.attrs["feat_key"]
    ):
        raise ValueError("Enrichment artifact selection keys do not match its metadata")

    resolved_columns: dict[str, str] = {}
    for (
        input_name,
        expected_kind,
        expected_scope,
        digest_name,
        source_column,
        metadata,
    ) in (
        (
            "cell_selection",
            "cell_selection",
            "datastore",
            "cell_digest",
            cell_key,
            assay.cells,
        ),
        (
            "feature_selection",
            "feature_selection",
            "assay",
            "feature_digest",
            feat_key,
            assay.feats,
        ),
    ):
        raw_ref = inputs.get(input_name)
        if not isinstance(raw_ref, dict):
            raise ValueError(
                f"Enrichment artifact is missing {input_name!r} provenance"
            )
        selection_ref = ArtifactRef.from_dict(raw_ref)
        selection_status = inspect_artifact(root, selection_ref)
        if (
            selection_ref.kind != expected_kind
            or selection_ref.scope != expected_scope
            or (expected_scope == "assay" and selection_ref.assay != status.ref.assay)
            or not selection_status.exists
            or not selection_status.complete
        ):
            raise ValueError(f"Enrichment artifact has an invalid {input_name!r} input")
        selection_group = as_zarr_group(
            root[selection_status.path],
            name=selection_status.path,
        )
        selection_values = np.asarray(
            as_zarr_array(selection_group["values"], name="values")[:]
        )
        if selection_values.ndim != 1 or selection_values.dtype != np.dtype(bool):
            raise ValueError(f"Enrichment artifact has malformed {input_name!r} values")
        candidates = [source_column]
        candidates.extend(
            sorted(column for column in metadata.columns if column != source_column)
        )
        resolved_column = None
        for candidate in candidates:
            try:
                current_values = np.asarray(metadata.fetch_all(candidate))
            except (KeyError, TypeError, ValueError):
                continue
            if (
                current_values.ndim == 1
                and current_values.dtype == np.dtype(bool)
                and current_values.shape == selection_values.shape
                and np.array_equal(current_values, selection_values)
            ):
                resolved_column = candidate
                break
        if resolved_column is None:
            raise ValueError(
                f"Enrichment selection column {source_column!r} "
                "is unavailable and no equivalent column exists"
            )
        resolved_columns[input_name] = resolved_column
        selected_index = np.flatnonzero(selection_values).astype(np.int64)
        if array_digest(selected_index) != group.attrs[digest_name]:
            raise ValueError(
                f"Enrichment artifact {input_name!r} does not match its metadata"
            )
    return (
        resolved_columns["cell_selection"],
        resolved_columns["feature_selection"],
    )


def _load_enrichment_result(
    assay: Assay,
    *,
    label: str,
    sources: Sequence[str] | None,
    artifact_root: zarr.Group | None = None,
) -> EnrichmentResult:
    from ...matrix import ChunkedArray

    artifact_entry = _enrichment_artifact_entry(assay, label)
    ref = None if artifact_entry is None else artifact_entry[0]
    indexed_cell_key = None if artifact_entry is None else artifact_entry[1]
    indexed_feat_key = None if artifact_entry is None else artifact_entry[2]
    artifact_backed = ref is not None
    artifact_operation: str | None = None
    artifact_status: ArtifactStatus | None = None
    resolved_cell_key: str | None = indexed_cell_key
    resolved_feat_key: str | None = indexed_feat_key
    if ref is not None:
        if artifact_root is None:
            raise ValueError("Artifact root is required for indexed enrichment")
        status = inspect_artifact(artifact_root, ref)
        artifact_status = status
        if (
            ref.kind != "enrichment_scores"
            or ref.scope != "assay"
            or ref.assay != assay.name
            or not status.complete
        ):
            raise ValueError(f"Enrichment label {label!r} has an invalid artifact")
        artifact_operation = status.operation
        slot = as_zarr_group(
            artifact_root[status.path],
            name=status.path,
        )
        root_path = str(getattr(artifact_root, "path", "")).strip("/")
        storage_path = f"{root_path}/{status.path}" if root_path else status.path
    else:
        storage_path = f"{getattr(assay.z, 'path', assay.name)}/enrichment/{label}"
        if "enrichment" not in assay.z:
            raise KeyError(f"Enrichment label {label!r} was not found for {assay.name}")
        enrichment_group = as_zarr_group(
            assay.z["enrichment"],
            name=f"{assay.name}/enrichment",
        )
        if label not in enrichment_group:
            raise KeyError(f"Enrichment label {label!r} was not found for {assay.name}")
        label_group = as_zarr_group(enrichment_group[label], name=storage_path)
        slot = _resolve_enrichment_slot(label_group, label=label)
    if slot.attrs.get("complete") is not True:
        raise ValueError(f"Enrichment slot {label!r} is incomplete")
    method = str(slot.attrs.get("method", ""))
    if method not in {"waggr", "aucell"}:
        raise ValueError(f"Enrichment slot {label!r} has an unknown method")
    if artifact_backed and artifact_operation != f"run_{method}":
        raise ValueError(
            f"Enrichment label {label!r} has a mismatched artifact operation"
        )
    required_attrs = {
        "algorithm_version",
        "cell_digest",
        "cell_key",
        "feature_digest",
        "feat_key",
        "network_digest",
        "tmin",
    }
    if not artifact_backed:
        required_attrs.add("execution_digest")
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
    if not artifact_backed:
        execution_digest = slot.attrs["execution_digest"]
        if not isinstance(execution_digest, str) or not execution_digest:
            raise ValueError(
                f"Enrichment slot {label!r} has invalid execution_digest metadata"
            )
    for key_name in ("cell_key", "feat_key"):
        key_value = slot.attrs[key_name]
        if not isinstance(key_value, str) or not key_value:
            raise ValueError(
                f"Enrichment slot {label!r} has invalid {key_name} metadata"
            )
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
    if artifact_backed:
        assert artifact_root is not None and artifact_status is not None
        resolved_cell_key, resolved_feat_key = _validate_enrichment_artifact_provenance(
            artifact_root,
            assay,
            artifact_status,
            slot,
            method,
            (
                indexed_cell_key
                if indexed_cell_key is not None
                else str(slot.attrs["cell_key"])
            ),
            (
                indexed_feat_key
                if indexed_feat_key is not None
                else str(slot.attrs["feat_key"])
            ),
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
        cell_key=(
            resolved_cell_key
            if resolved_cell_key is not None
            else str(slot.attrs["cell_key"])
        ),
        feature_key=(
            resolved_feat_key
            if resolved_feat_key is not None
            else str(slot.attrs["feat_key"])
        ),
        method=method,
    )


class _FeatureOperationsMixin(_FeatureOperationsBase):
    def set_hvgs(
        self,
        *,
        from_assay: str | None = None,
        cell_key: str,
        mask: np.ndarray | None = None,
        feature_indexes: Sequence[int] | None = None,
        hvg_key_name: str = "hvgs",
        n_bins: int = 200,
        lowess_frac: float = 0.1,
        bin_strategy: Literal["fixed", "adaptive"] = "adaptive",
        blacklist: str | None = None,
        blacklist_exclusions: str | None = None,
        blacklist_indexes: Sequence[int] | None = None,
        invalidate_cache: bool = False,
    ) -> str:
        """Install a supplied HVG selection on an RNA assay."""
        assay = self._get_assay(from_assay)
        if not isinstance(assay, RNAassay):
            raise TypeError(
                "set_hvgs can only be applied to an RNAassay; "
                f"received {type(assay).__name__}"
            )
        expected_key = f"{cell_key}__{hvg_key_name}"
        storage_backed = (
            hasattr(self, "z") and hasattr(self, "cells") and hasattr(assay, "z")
        )
        preserved_display = (
            feature_column_display(assay.z, expected_key) if storage_backed else None
        )
        stored_key = assay.set_hvgs(
            cell_key,
            mask=mask,
            feature_indexes=feature_indexes,
            hvg_key_name=hvg_key_name,
            n_bins=n_bins,
            lowess_frac=lowess_frac,
            bin_strategy=bin_strategy,
            blacklist=blacklist,
            blacklist_exclusions=blacklist_exclusions,
            blacklist_indexes=blacklist_indexes,
        )
        if not storage_backed:
            return stored_key
        cell_selection = self._linked_cell_selection(cell_key)
        if cell_selection is None:
            cell_selection = self._record_cell_selection(
                column=cell_key,
                operation="manual_selection",
                parameters={},
                inputs={},
            )
        feature_selection = resolve_selection_artifact(
            self.zw,
            scope="assay",
            assay=assay.name,
            kind="feature_selection",
            values=np.asarray(assay.feats.fetch_all(stored_key)),
            row_ids=np.asarray(assay.feats.fetch_all("ids")),
            operation="set_hvgs",
            parameters={
                "n_bins": n_bins,
                "lowess_frac": lowess_frac,
                "bin_strategy": bin_strategy,
                "blacklist": blacklist,
                "blacklist_exclusions": blacklist_exclusions,
                "blacklist_indexes": list(blacklist_indexes)
                if blacklist_indexes is not None
                else None,
            },
            inputs={"cell_selection": cell_selection},
            source_column=stored_key,
            invalidate_cache=invalidate_cache,
        )
        link_feature_data_column(
            assay.z,
            stored_key,
            feature_selection,
            value_name="values",
            default_display=categorical_display(
                np.asarray(assay.feats.fetch_all(stored_key))
            ),
            preserved_display=preserved_display,
        )
        return stored_key

    def mark_hvgs(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        min_cells: int | None = None,
        top_n: int = 500,
        min_var: float = -np.inf,
        max_var: float = np.inf,
        min_mean: float = -np.inf,
        max_mean: float = np.inf,
        n_bins: int = 200,
        lowess_frac: float = 0.1,
        blacklist: str = "^MT-|^RPS|^RPL|^MRPS|^MRPL|^CCN|^HLA-|^H2-|^HIST",
        keep_bounds: bool = False,
        show_plot: bool = True,
        hvg_key_name: str = "hvgs",
        max_cells: float | None = None,
        bin_strategy: Literal["fixed", "adaptive"] = "adaptive",
        invalidate_cache: bool = False,
        **plot_kwargs: Any,
    ) -> None:
        """Identify and mark genes as highly variable genes (HVGs). This is a
        critical and required feature selection step and is only applicable to
        RNAassay type of assays.

        Args:
            from_assay: Assay to use for graph creation. If no value is provided then `defaultAssay` will be used
            cell_key: Cells to use for HVG selection. By default, all cells with True value in 'I' will be used.
                      The provided value for `cell_key` should be a column in cell metadata table with boolean values.
            min_cells: Minimum number of cells where a gene should have non-zero expression values for it to be
                       considered a candidate for HVG selection. Large values for this parameter might make it difficult
                       to identify rare populations of cells. Very small values might lead to a higher signal-to-noise
                       ratio in the selected features. By default, a value is set assuming smallest population has no
                       less than 1% of all cells. So for example, if you have 1000 cells (as per cell_key parameter)
                       then `min_cells` will be set to 10.
            max_cells: Maximum number of cells where a gene should have non-zero expression values for it to be
                       considered a candidate for HVG selection. This can be useful to filter out genes that are
                       expressed in too many cells. Default value is infinity, meaning no upper limit.
            top_n: Number of top most variable genes to be set as HVGs. This value is ignored if a value is provided
                   for `min_var` parameter. (Default: 500)
            min_var: Minimum variance threshold for HVG selection. (Default: -Infinity)
            max_var: Maximum variance threshold for HVG selection. (Default: Infinity)
            min_mean: Minimum mean value of expression threshold for HVG selection. (Default: -Infinity)
            max_mean: Maximum mean value of expression threshold for HVG selection. (Default: Infinity)
            n_bins: Number of bins into which the mean expression is binned. (Default: 200)
            lowess_frac: Between 0 and 1. The fraction of the data used when estimating the fit between mean and
                         variance. This is same as `frac` in statsmodels.nonparametric.smoothers_lowess.lowess
                         (Default: 0.1)
            bin_strategy: Strategy used to construct bins and variance anchors. (Default: 'adaptive')
            blacklist: This is a regular expression (regex) string that can be used to exclude genes from being marked
                       as HVGs. By default, we exclude mitochondrial, ribosomal, some cell-cycle related, histone and
                       HLA genes. (Default: '^MT- | ^RPS | ^RPL | ^MRPS | ^MRPL | ^CCN | ^HLA- | ^H2- | ^HIST' )
            keep_bounds: If True, then the boundary values are retained and not filtered out (Default value: False)
            show_plot: If True then a diagnostic scatter plot is shown with HVGs highlighted. (Default: True)
            hvg_key_name: Base label for HVGs in the features metadata column. The value for
                          'cell_key' parameter is prepended to this value. (Default value: 'hvgs')
            plot_kwargs: Named parameters forwarded to ``plotting.highly_variable_features``
                         (``figsize``, ``label_size``, ``point_sizes``, ``colormaps``).

        Returns:
            None
        """

        if cell_key is None:
            cell_key = "I"
        assay = self._get_assay(from_assay)
        if type(assay) != RNAassay:  # noqa: E721
            raise TypeError(
                f"ERROR: This method of feature selection can only be applied to RNAassay type of assay. "
                f"The provided assay is {type(assay)} type"
            )
        if min_cells is None:
            min_cells = int(0.01 * self.cells.N)
            logger.info(
                f"Setting `min_cells` to {min_cells}. Only those genes that are present in atleast this number "
                f"of cells will be considered HVGs."
            )
        if max_cells is None or max_cells == np.inf:
            max_cells_int: int | float = np.inf
        else:
            max_cells_int = int(max_cells)
        stored_key = f"{cell_key}__{hvg_key_name}"
        preserved_display = feature_column_display(
            assay.z,
            stored_key,
        )
        assay.mark_hvgs(
            cell_key=cell_key,
            min_cells=min_cells,
            max_cells=max_cells_int,
            top_n=top_n,
            min_var=min_var,
            max_var=max_var,
            min_mean=min_mean,
            max_mean=max_mean,
            n_bins=n_bins,
            lowess_frac=lowess_frac,
            bin_strategy=bin_strategy,
            blacklist=blacklist,
            hvg_key_name=hvg_key_name,
            keep_bounds=keep_bounds,
            show_plot=show_plot,
            **plot_kwargs,
        )
        cell_selection = self._linked_cell_selection(cell_key)
        if cell_selection is None:
            cell_selection = self._record_cell_selection(
                column=cell_key,
                operation="manual_selection",
                parameters={},
                inputs={},
            )
        feature_selection = resolve_selection_artifact(
            self.zw,
            scope="assay",
            assay=assay.name,
            kind="feature_selection",
            values=np.asarray(assay.feats.fetch_all(stored_key)),
            row_ids=np.asarray(assay.feats.fetch_all("ids")),
            operation="mark_hvgs",
            parameters={
                "min_cells": min_cells,
                "max_cells": max_cells_int,
                "top_n": top_n,
                "min_var": min_var,
                "max_var": max_var,
                "min_mean": min_mean,
                "max_mean": max_mean,
                "n_bins": n_bins,
                "lowess_frac": lowess_frac,
                "blacklist": blacklist,
                "keep_bounds": keep_bounds,
                "bin_strategy": bin_strategy,
            },
            inputs={"cell_selection": cell_selection},
            source_column=stored_key,
            invalidate_cache=invalidate_cache,
        )
        link_feature_data_column(
            assay.z,
            stored_key,
            feature_selection,
            value_name="values",
            default_display=categorical_display(
                np.asarray(assay.feats.fetch_all(stored_key))
            ),
            preserved_display=preserved_display,
        )

    def run_waggr(
        self,
        net: pd.DataFrame,
        label: str,
        *,
        from_assay: str | None = None,
        cell_key: str = "I",
        feat_key: str = "I",
        mode: Literal["wmean", "wsum"] = "wmean",
        tmin: int = 5,
        log_transform: bool = False,
        overwrite: bool = False,
        invalidate_cache: bool = False,
    ) -> EnrichmentResult:
        """Score weighted gene sets from streamed normalized RNA counts.

        Targets are matched to active feature names without case sensitivity. Sources
        with fewer than ``tmin`` matched non-zero edges are removed. Results are
        written to the assay's enrichment group and returned lazily.

        Args:
            net: Network with ``source`` and ``target`` columns. An optional
                ``weight`` column supplies signed numeric edge weights. Missing
                weights default to one.
            label: Name used to persist and retrieve the result.
            from_assay: RNA assay to score. The default assay is used when omitted.
            cell_key: Cell metadata key that selects score rows.
            feat_key: Feature metadata key that defines the matching universe.
            mode: ``"wmean"`` divides each weighted sum by the sum of absolute
                source weights. ``"wsum"`` returns the weighted sum.
            tmin: Minimum number of matched targets required per source.
            log_transform: Apply ``log1p`` after library-size normalization.
            overwrite: Replace a complete result with different execution metadata.
                The previous result remains active until the replacement is complete.

        Returns:
            A persisted result with a lazy cells-by-sources score matrix.

        Note:
            Cache identity covers selections, method parameters, normalization, and
            the prepared network. It assumes the stored count matrix is immutable.
        """
        from ...features.enrichment.net import prepare_network
        from ...features.enrichment.waggr import (
            WAGGR_ALGORITHM_VERSION,
            build_waggr_model,
            score_waggr_block,
        )

        if self.zarr_mode != "r+":
            raise ValueError("WAGGR requires a DataStore opened with zarr_mode='r+'")
        _validate_enrichment_label(label)
        if mode not in {"wmean", "wsum"}:
            raise ValueError("mode must be 'wmean' or 'wsum'")
        if not isinstance(log_transform, bool):
            raise TypeError("log_transform must be a boolean")
        if not isinstance(overwrite, bool):
            raise TypeError("overwrite must be a boolean")

        assay = self._get_assay(from_assay)
        if not isinstance(assay, RNAassay):
            raise TypeError("WAGGR can only be run on an RNAassay")
        cell_index, feature_index = assay._get_cell_feat_idx(cell_key, feat_key)
        cell_index = np.asarray(cell_index, dtype=np.int64)
        feature_index = np.asarray(feature_index, dtype=np.int64)
        if len(cell_index) == 0:
            raise ValueError(f"Cell key {cell_key!r} selects no active cells")
        feature_names = np.asarray(assay.feats.fetch_all("names"))[feature_index]
        network = prepare_network(
            net,
            active_feature_names=feature_names,
            active_feature_index=feature_index,
            tmin=tmin,
            weighted=True,
        )
        if not lib_size_feature_stream_eligible(assay):
            raise ValueError(
                "WAGGR requires the default norm_lib_size RNA normalization"
            )
        if assay.sf is None:
            raise ValueError("WAGGR requires a finite positive size factor")
        try:
            size_factor = float(assay.sf)
        except (TypeError, ValueError) as exc:
            raise ValueError("WAGGR requires a finite positive size factor") from exc
        if not np.isfinite(size_factor) or size_factor <= 0:
            raise ValueError("WAGGR requires a finite positive size factor")

        cell_digest = array_digest(cell_index)
        feature_digest = array_digest(feature_index)
        execution = _execution_digest(
            {
                "algorithm_version": WAGGR_ALGORITHM_VERSION,
                "cell_digest": cell_digest,
                "cell_key": cell_key,
                "feature_digest": feature_digest,
                "feat_key": feat_key,
                "log_transform": log_transform,
                "method": "waggr",
                "network_digest": network.network_digest,
                "normalization": "norm_lib_size",
                "size_factor": size_factor,
                "tmin": tmin,
                "waggr_mode": mode,
            }
        )
        attrs: dict[str, Any] = {
            "algorithm_version": WAGGR_ALGORITHM_VERSION,
            "cell_digest": cell_digest,
            "cell_key": cell_key,
            "complete": False,
            "feature_digest": feature_digest,
            "feat_key": feat_key,
            "layout": _ENRICHMENT_LAYOUT,
            "log_transform": log_transform,
            "method": "waggr",
            "network_digest": network.network_digest,
            "normalization": "norm_lib_size",
            "size_factor": size_factor,
            "tmin": tmin,
            "waggr_mode": mode,
        }
        cell_selection = self._ensure_cell_selection(cell_key)
        feature_values = np.asarray(assay.feats.fetch_all(feat_key), dtype=bool)
        feature_selection = self._resolve_selection_input(
            metadata_group=as_zarr_group(
                assay.z["featureData"],
                name="featureData",
            ),
            column=feat_key,
            values=feature_values,
            row_ids=np.asarray(assay.feats.fetch_all("ids")),
            scope="assay",
            kind="feature_selection",
            assay=assay.name,
            invalidate_cache=False,
        )
        arguments = WaggrArguments(
            cell_selection=cell_selection,
            feature_selection=feature_selection,
            network_digest=network.network_digest,
            algorithm_version=WAGGR_ALGORITHM_VERSION,
            mode=mode,
            tmin=tmin,
            log_transform=log_transform,
            normalization_method=callable_identity(assay.normMethod),
            size_factor=size_factor,
            from_assay=assay.name,
            cell_key=cell_key,
            feat_key=feat_key,
            label=label,
            overwrite=overwrite,
            invalidate_cache=invalidate_cache,
        )
        record = arguments.to_record()
        planned = plan_artifact(
            self.zw,
            scope="assay",
            assay=assay.name,
            kind=arguments.artifact_kind,
            operation=arguments.operation,
            parameters=record.parameters,
            inputs=record.inputs,
            execution_options=record.execution_options,
            invalidate_cache=invalidate_cache,
            required_arrays=(
                ArrayRequirement(
                    "scores",
                    shape=(len(cell_index), len(network.source_names)),
                    dtype_kind="f",
                ),
                ArrayRequirement(
                    "cell_index",
                    shape=(len(cell_index),),
                    dtype_kind="i",
                ),
                ArrayRequirement("matched_feature_index", dtype_kind="i"),
                ArrayRequirement(
                    "source_names",
                    shape=(len(network.source_names),),
                ),
                ArrayRequirement(
                    "source_sizes",
                    shape=(len(network.source_names),),
                    dtype_kind="i",
                ),
            ),
            required_attributes=(
                "algorithm_version",
                "method",
                "network_digest",
            ),
            reuse_validator=lambda _ref, group: _enrichment_artifact_matches(
                group,
                attrs=attrs,
                cell_index=cell_index,
                matched_feature_index=network.matched_feature_index,
                source_names=network.source_names,
                source_sizes=network.source_sizes,
                rank_feature_index=None,
            ),
        )
        existing_ref = _enrichment_artifact_ref(assay, label)
        legacy_slot = (
            _legacy_enrichment_slot(assay, label) if existing_ref is None else None
        )
        if existing_ref is not None:
            existing_status = inspect_artifact(self.zw, existing_ref)
            if (
                existing_ref.kind != "enrichment_scores"
                or existing_ref.scope != "assay"
                or existing_ref.assay != assay.name
                or not existing_status.complete
            ):
                raise ValueError(f"Enrichment label {label!r} has an invalid artifact")
            if existing_status.provenance != planned.provenance and not overwrite:
                raise ValueError(
                    f"Enrichment label {label!r} already contains a different "
                    f"{existing_status.operation!r} execution; pass overwrite=True "
                    "to replace it"
                )
        elif legacy_slot is not None:
            same_legacy_execution = (
                str(legacy_slot.attrs.get("execution_digest", "")) == execution
            )
            if (
                legacy_slot.attrs.get("complete") is True
                and same_legacy_execution
                and not invalidate_cache
            ):
                return _load_enrichment_result(
                    assay,
                    label=label,
                    sources=None,
                    artifact_root=self.zw,
                )
            if (
                legacy_slot.attrs.get("complete") is True
                and not same_legacy_execution
                and not overwrite
            ):
                method = legacy_slot.attrs.get("method", "unknown")
                raise ValueError(
                    f"Enrichment label {label!r} already contains a different "
                    f"{method!r} execution; pass overwrite=True to replace it"
                )
        if planned.reused:
            _publish_enrichment_artifact(
                assay,
                label,
                planned.ref,
                cell_key=cell_key,
                feat_key=feat_key,
            )
            return _load_enrichment_result(
                assay,
                label=label,
                sources=None,
                artifact_root=self.zw,
            )
        slot = start_artifact(self.zw, planned)

        cell_scalars = np.asarray(
            assay.cells.fetch_all(f"{assay.name}_nCounts")[cell_index],
            dtype=np.float64,
        )
        if not np.isfinite(cell_scalars).all() or np.any(cell_scalars < 0):
            raise ValueError("WAGGR cell normalization scalars must be finite")
        cell_scalars[cell_scalars == 0] = 1.0
        model = build_waggr_model(network)
        raw = assay.rawData[:, network.matched_feature_index][cell_index, :]

        def score_batches() -> Iterator[np.ndarray]:
            offset = 0
            for raw_block in raw.stream_blocks(
                nthreads=self.nthreads,
                msg="Scoring WAGGR",
                prefetch=1,
            ):
                block = np.asarray(raw_block, dtype=np.float64)
                end = offset + block.shape[0]
                if end > len(cell_scalars):
                    raise ValueError(
                        "WAGGR raw blocks exceed the active cell selection"
                    )
                values = size_factor * block / cell_scalars[offset:end].reshape(-1, 1)
                if log_transform:
                    values = np.log1p(values)
                yield score_waggr_block(values, model, mode=mode)
                offset = end
            if offset != len(cell_scalars):
                raise ValueError(
                    f"WAGGR streamed {offset} cells, expected {len(cell_scalars)}"
                )

        _write_enrichment_slot(
            slot,
            attrs=attrs,
            score_batches=score_batches(),
            n_cells=len(cell_index),
            source_names=network.source_names,
            source_sizes=network.source_sizes,
            cell_index=cell_index,
            matched_feature_index=network.matched_feature_index,
            rank_feature_index=None,
        )
        finish_artifact(slot, planned)
        _publish_enrichment_artifact(
            assay,
            label,
            planned.ref,
            cell_key=cell_key,
            feat_key=feat_key,
        )
        return _load_enrichment_result(
            assay,
            label=label,
            sources=None,
            artifact_root=self.zw,
        )

    def run_aucell(
        self,
        net: pd.DataFrame,
        label: str,
        *,
        from_assay: str | None = None,
        cell_key: str = "I",
        feat_key: str = "I",
        tmin: int = 5,
        n_up: int | None = None,
        tie_seed: int = 0,
        overwrite: bool = False,
        invalidate_cache: bool = False,
    ) -> EnrichmentResult:
        """Score gene sets by recovery among each cell's top-ranked RNA features.

        AUCell ranks every feature selected by ``feat_key`` from raw counts. Network
        weights are ignored. Targets are matched without case sensitivity, then
        sources with fewer than ``tmin`` matched targets are removed.

        Args:
            net: Network with ``source`` and ``target`` columns.
            label: Name used to persist and retrieve the result.
            from_assay: RNA assay to score. The default assay is used when omitted.
            cell_key: Cell metadata key that selects score rows.
            feat_key: Feature metadata key that defines the ranking universe.
            tmin: Minimum number of matched targets required per source.
            n_up: Number of top-ranked features used for recovery. When omitted,
                five percent of the ranking universe is used, clipped to its valid
                range.
            tie_seed: Seed for the global feature permutation used to resolve ties.
            overwrite: Replace a complete result with different execution metadata.
                The previous result remains active until the replacement is complete.

        Returns:
            A persisted result with lazy scores in the interval from zero to one.

        Note:
            Cache identity covers selections, method parameters, and the prepared
            network. It assumes the stored count matrix is immutable.
        """
        from ...features.enrichment.aucell import (
            AUCELL_ALGORITHM_VERSION,
            build_gene_set_index,
            make_rank_permutation,
            resolve_n_up,
            score_aucell_block,
        )
        from ...features.enrichment.net import prepare_network

        if self.zarr_mode != "r+":
            raise ValueError("AUCell requires a DataStore opened with zarr_mode='r+'")
        _validate_enrichment_label(label)
        if not isinstance(overwrite, bool):
            raise TypeError("overwrite must be a boolean")

        assay = self._get_assay(from_assay)
        if not isinstance(assay, RNAassay):
            raise TypeError("AUCell can only be run on an RNAassay")
        cell_index, feature_index = assay._get_cell_feat_idx(cell_key, feat_key)
        cell_index = np.asarray(cell_index, dtype=np.int64)
        feature_index = np.asarray(feature_index, dtype=np.int64)
        if len(cell_index) == 0:
            raise ValueError(f"Cell key {cell_key!r} selects no active cells")
        resolved_n_up = resolve_n_up(len(feature_index), n_up)
        feature_names = np.asarray(assay.feats.fetch_all("names"))[feature_index]
        network = prepare_network(
            net,
            active_feature_names=feature_names,
            active_feature_index=feature_index,
            tmin=tmin,
            weighted=False,
        )
        permutation = make_rank_permutation(len(feature_index), tie_seed)
        rank_feature_index = feature_index[permutation]
        sets = build_gene_set_index(network, rank_feature_index)

        cell_digest = array_digest(cell_index)
        feature_digest = array_digest(feature_index)
        execution = _execution_digest(
            {
                "algorithm_version": AUCELL_ALGORITHM_VERSION,
                "cell_digest": cell_digest,
                "cell_key": cell_key,
                "feature_digest": feature_digest,
                "feat_key": feat_key,
                "method": "aucell",
                "n_up": resolved_n_up,
                "network_digest": network.network_digest,
                "tie_seed": tie_seed,
                "tmin": tmin,
            }
        )
        attrs = {
            "algorithm_version": AUCELL_ALGORITHM_VERSION,
            "cell_digest": cell_digest,
            "cell_key": cell_key,
            "complete": False,
            "feature_digest": feature_digest,
            "feat_key": feat_key,
            "layout": _ENRICHMENT_LAYOUT,
            "method": "aucell",
            "n_up": resolved_n_up,
            "network_digest": network.network_digest,
            "tie_seed": tie_seed,
            "tmin": tmin,
        }
        cell_selection = self._ensure_cell_selection(cell_key)
        feature_values = np.asarray(assay.feats.fetch_all(feat_key), dtype=bool)
        feature_selection = self._resolve_selection_input(
            metadata_group=as_zarr_group(
                assay.z["featureData"],
                name="featureData",
            ),
            column=feat_key,
            values=feature_values,
            row_ids=np.asarray(assay.feats.fetch_all("ids")),
            scope="assay",
            kind="feature_selection",
            assay=assay.name,
            invalidate_cache=False,
        )
        arguments = AucellArguments(
            cell_selection=cell_selection,
            feature_selection=feature_selection,
            network_digest=network.network_digest,
            algorithm_version=AUCELL_ALGORITHM_VERSION,
            tmin=tmin,
            n_up=resolved_n_up,
            tie_seed=tie_seed,
            from_assay=assay.name,
            cell_key=cell_key,
            feat_key=feat_key,
            label=label,
            overwrite=overwrite,
            invalidate_cache=invalidate_cache,
        )
        record = arguments.to_record()
        planned = plan_artifact(
            self.zw,
            scope="assay",
            assay=assay.name,
            kind=arguments.artifact_kind,
            operation=arguments.operation,
            parameters=record.parameters,
            inputs=record.inputs,
            execution_options=record.execution_options,
            invalidate_cache=invalidate_cache,
            required_arrays=(
                ArrayRequirement(
                    "scores",
                    shape=(len(cell_index), len(network.source_names)),
                    dtype_kind="f",
                ),
                ArrayRequirement(
                    "cell_index",
                    shape=(len(cell_index),),
                    dtype_kind="i",
                ),
                ArrayRequirement("matched_feature_index", dtype_kind="i"),
                ArrayRequirement(
                    "rank_feature_index",
                    shape=(len(rank_feature_index),),
                    dtype_kind="i",
                ),
                ArrayRequirement(
                    "source_names",
                    shape=(len(network.source_names),),
                ),
                ArrayRequirement(
                    "source_sizes",
                    shape=(len(network.source_names),),
                    dtype_kind="i",
                ),
            ),
            required_attributes=(
                "algorithm_version",
                "method",
                "network_digest",
            ),
            reuse_validator=lambda _ref, group: _enrichment_artifact_matches(
                group,
                attrs=attrs,
                cell_index=cell_index,
                matched_feature_index=network.matched_feature_index,
                source_names=network.source_names,
                source_sizes=network.source_sizes,
                rank_feature_index=rank_feature_index,
            ),
        )
        existing_ref = _enrichment_artifact_ref(assay, label)
        legacy_slot = (
            _legacy_enrichment_slot(assay, label) if existing_ref is None else None
        )
        if existing_ref is not None:
            existing_status = inspect_artifact(self.zw, existing_ref)
            if (
                existing_ref.kind != "enrichment_scores"
                or existing_ref.scope != "assay"
                or existing_ref.assay != assay.name
                or not existing_status.complete
            ):
                raise ValueError(f"Enrichment label {label!r} has an invalid artifact")
            if existing_status.provenance != planned.provenance and not overwrite:
                raise ValueError(
                    f"Enrichment label {label!r} already contains a different "
                    f"{existing_status.operation!r} execution; pass overwrite=True "
                    "to replace it"
                )
        elif legacy_slot is not None:
            same_legacy_execution = (
                str(legacy_slot.attrs.get("execution_digest", "")) == execution
            )
            if (
                legacy_slot.attrs.get("complete") is True
                and same_legacy_execution
                and not invalidate_cache
            ):
                return _load_enrichment_result(
                    assay,
                    label=label,
                    sources=None,
                    artifact_root=self.zw,
                )
            if (
                legacy_slot.attrs.get("complete") is True
                and not same_legacy_execution
                and not overwrite
            ):
                method = legacy_slot.attrs.get("method", "unknown")
                raise ValueError(
                    f"Enrichment label {label!r} already contains a different "
                    f"{method!r} execution; pass overwrite=True to replace it"
                )
        if planned.reused:
            _publish_enrichment_artifact(
                assay,
                label,
                planned.ref,
                cell_key=cell_key,
                feat_key=feat_key,
            )
            return _load_enrichment_result(
                assay,
                label=label,
                sources=None,
                artifact_root=self.zw,
            )
        slot = start_artifact(self.zw, planned)

        raw = assay.rawData[:, feature_index][cell_index, :]

        def score_batches() -> Iterator[np.ndarray]:
            offset = 0
            for raw_block in raw.stream_blocks(
                nthreads=self.nthreads,
                msg="Scoring AUCell",
                prefetch=1,
            ):
                scores = score_aucell_block(
                    np.asarray(raw_block),
                    permutation,
                    sets,
                    n_up=resolved_n_up,
                )
                offset += scores.shape[0]
                if offset > len(cell_index):
                    raise ValueError(
                        "AUCell raw blocks exceed the active cell selection"
                    )
                yield scores
            if offset != len(cell_index):
                raise ValueError(
                    f"AUCell streamed {offset} cells, expected {len(cell_index)}"
                )

        import numba

        previous_threads = numba.get_num_threads()
        numba.set_num_threads(
            min(max(1, int(self.nthreads)), numba.config.NUMBA_NUM_THREADS)
        )
        try:
            _write_enrichment_slot(
                slot,
                attrs=attrs,
                score_batches=score_batches(),
                n_cells=len(cell_index),
                source_names=network.source_names,
                source_sizes=network.source_sizes,
                cell_index=cell_index,
                matched_feature_index=network.matched_feature_index,
                rank_feature_index=rank_feature_index,
            )
        finally:
            numba.set_num_threads(previous_threads)
        finish_artifact(slot, planned)
        _publish_enrichment_artifact(
            assay,
            label,
            planned.ref,
            cell_key=cell_key,
            feat_key=feat_key,
        )
        return _load_enrichment_result(
            assay,
            label=label,
            sources=None,
            artifact_root=self.zw,
        )

    def get_enrichment(
        self,
        label: str,
        *,
        from_assay: str | None = None,
        sources: Sequence[str] | None = None,
    ) -> EnrichmentResult:
        """Load a persisted enrichment result without materializing its scores.

        Args:
            label: Label passed to ``run_waggr`` or ``run_aucell``.
            from_assay: RNA assay that owns the result. The default assay is used
                when omitted.
            sources: Optional source names to select and order.

        Returns:
            The stored metadata and a lazy cells-by-sources score matrix.
        """
        _validate_enrichment_label(label)
        assay = self._get_assay(from_assay)
        if not isinstance(assay, RNAassay):
            raise TypeError("Enrichment results are only available for an RNAassay")
        return _load_enrichment_result(
            assay,
            label=label,
            sources=sources,
            artifact_root=self.zw,
        )

    def run_marker_search(
        self,
        from_assay: str | None = None,
        group_key: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        gene_batch_size: int | None = None,
        n_threads: int | None = None,
        skip_save: bool = False,
        invalidate_cache: bool = False,
        **norm_params: Any,
    ) -> dict[str, Any] | None:
        """Identifies group specific features for a given assay.

        Please check out the ``find_markers_by_rank`` function for further details of how marker features for groups
        are identified. The results are saved into the Zarr hierarchy under `markers` group.

        Args:
            from_assay: Name of the assay to be used. If no value is provided then the default assay will be used.
            group_key: Required parameter. This has to be a column name from cell metadata table. This column dictates
                       how the cells will be grouped. Usually this would be a column denoting cell clusters.
            cell_key: To run the test on specific subset of cells, provide the name of a boolean column in
                        the cell metadata table. (Default value: 'I')
            feat_key: Boolean feature metadata column selecting features (default: ``'I'``).
            gene_batch_size: Number of genes loaded per batch; all selected cells are loaded for each batch.
                             When None (default), the batch size is the minimum of the on-disk feature chunk
                             width and a budget-safe cap derived from the active memory budget.
            n_threads: Threads for marker search.
            skip_save: If True, return results without writing to Zarr.
            **norm_params: Extra keyword arguments forwarded to ``normed``.

        Returns:
            Marker dict if ``skip_save`` is True, else None.
        """
        from ...features.markers import find_markers_by_rank

        if group_key is None:
            raise ValueError(
                "ERROR: Please provide a value for `group_key`. This should be the name of a column from "
                "cell metadata object that has information on how cells should be grouped."
            )
        from_assay, cell_key, _ = self._get_latest_keys(from_assay, cell_key, None)
        if feat_key is None:
            feat_key = "I"
        if n_threads is None:
            n_threads = self.nthreads
        assay = self._get_assay(from_assay)
        resolved_norm_params = {
            **norm_params,
            "log_transform": norm_params.get("log_transform", False),
            "renormalize_subset": norm_params.get(
                "renormalize_subset",
                False,
            ),
        }

        n_features = len(assay.feats.active_index(feat_key))
        if gene_batch_size is None:
            gene_batch_size = resolve_marker_gene_batch_size(
                n_features=n_features,
                n_cells=len(assay.cells.active_index(cell_key)),
                column_chunk=_feature_column_chunk(assay, n_features),
            )

        slot_name = f"{cell_key}__{group_key}"
        logger.debug(
            f"Running marker search for {from_assay}/{slot_name} "
            f"(feat_key={feat_key}, batch_size={gene_batch_size})"
        )
        assay_grp = as_zarr_group(self.zw[assay.name], name=assay.name)
        if "markers" not in assay_grp:
            assay_grp.create_group("markers")
        markers_grp = as_zarr_group(assay_grp["markers"], name="markers")
        planned = None
        if not skip_save:
            cell_selection = self._ensure_cell_selection(cell_key)
            cluster_input = self._resolve_cell_data_provenance_input(
                group_key,
                cell_key=cell_key,
            )
            feature_values = np.asarray(
                assay.feats.fetch_all(feat_key),
                dtype=bool,
            )
            preserved_feature_display = feature_column_display(
                assay.z,
                feat_key,
            )
            feature_selection = resolve_selection_artifact(
                self.zw,
                scope="assay",
                assay=from_assay,
                kind="feature_selection",
                values=feature_values,
                row_ids=np.asarray(assay.feats.fetch_all("ids")),
                operation="manual_selection",
                parameters={},
                inputs={},
                source_column=feat_key,
                invalidate_cache=False,
            )
            link_feature_data_column(
                assay.z,
                feat_key,
                feature_selection,
                value_name="values",
                default_display=categorical_display(feature_values),
                preserved_display=preserved_feature_display,
            )
            arguments = MarkerTableArguments(
                cell_selection=cell_selection,
                feature_selection=feature_selection,
                clusters=cluster_input,
                normalization=resolved_norm_params,
                normalization_method=callable_identity(assay.normMethod),
                size_factor=getattr(assay, "sf", None),
                group_key=group_key,
                cell_key=cell_key,
                feat_key=feat_key,
                gene_batch_size=gene_batch_size,
                n_threads=n_threads,
                invalidate_cache=invalidate_cache,
            )
            record = arguments.to_record()
            planned = plan_artifact(
                self.zw,
                scope="assay",
                assay=from_assay,
                kind=arguments.artifact_kind,
                operation=arguments.operation,
                parameters=record.parameters,
                inputs=record.inputs,
                execution_options=record.execution_options,
                invalidate_cache=invalidate_cache,
                required_arrays=(
                    ArrayRequirement(
                        "feature_index",
                        dtype_kind="i",
                    ),
                ),
                required_attributes=(
                    AttributeRequirement(
                        "stat_columns",
                        expected_types=(list, tuple),
                    ),
                ),
            )

            def select_marker_artifact(ref: ArtifactRef) -> None:
                raw_artifacts = markers_grp.attrs.get(
                    "artifacts",
                    {},
                )
                if "artifacts" in markers_grp.attrs and not isinstance(
                    raw_artifacts,
                    dict,
                ):
                    raise RuntimeError("Marker artifact index is invalid")
                artifacts = (
                    dict(raw_artifacts) if isinstance(raw_artifacts, dict) else {}
                )
                artifacts[slot_name] = ref.to_dict()
                markers_grp.attrs["artifacts"] = artifacts

            if planned.reused:
                select_marker_artifact(planned.ref)
                return None

        markers = find_markers_by_rank(
            assay=assay,
            group_key=group_key,
            cell_key=cell_key,
            feat_key=feat_key,
            batch_size=gene_batch_size,
            n_threads=n_threads,
            **resolved_norm_params,
        )

        if skip_save:
            return markers

        from ...storage.stores import is_remote_datastore

        remote = is_remote_datastore(self.zarr_loc, self.z)
        t_save = time.perf_counter()
        assert planned is not None
        remote_slot = start_artifact(self.zw, planned)
        workers = max(1, int(n_threads or self.nthreads))
        self._write_marker_slot(
            remote_slot,
            markers,
            workers=workers if remote else 1,
        )
        finish_artifact(remote_slot, planned)
        select_marker_artifact(planned.ref)
        logger.info(
            f"Saved marker results to {artifact_path(planned.ref)} "
            f"in {time.perf_counter() - t_save:.1f}s "
            f"({len(markers)} clusters)"
        )
        return None

    @staticmethod
    def _write_marker_slot(
        group: zarr.Group,
        markers: dict[Any, pd.DataFrame],
        *,
        workers: int = 1,
    ) -> None:
        from ...storage.arrays import create_metadata_column

        feature_index = _shared_marker_feature_index(markers)
        group.attrs["stat_columns"] = list(_MARKER_STAT_COLUMNS)
        create_metadata_column(
            group,
            "feature_index",
            data=feature_index,
            dtype=np.int32,
            overwrite=True,
        )

        def write_cluster(item: tuple[Any, pd.DataFrame]) -> None:
            cluster_id, vals = item
            if len(vals) == 0:
                return
            cluster_group = group.create_group(str(cluster_id))
            stats = _marker_stats_matrix(vals, feature_index)
            _write_compact_marker_stats(
                cluster_group,
                stats,
            )

        items = list(markers.items())
        if workers <= 1:
            for item in items:
                write_cluster(item)
        else:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(write_cluster, items))

    def _resolve_marker_group(
        self,
        from_assay: str | None,
        cell_key: str,
        group_key: str,
    ) -> zarr.Group:
        assay = self._get_assay(from_assay)
        markers_group = as_zarr_group(
            assay.z["markers"],
            name="markers",
        )
        slot_name = f"{cell_key}__{group_key}"
        raw_artifacts = markers_group.attrs.get("artifacts", {})
        if "artifacts" in markers_group.attrs and not isinstance(
            raw_artifacts,
            dict,
        ):
            raise ValueError("Marker artifact index is invalid")
        artifacts = dict(raw_artifacts) if isinstance(raw_artifacts, dict) else {}
        raw_ref = artifacts.get(slot_name)
        if slot_name in artifacts and not isinstance(raw_ref, dict):
            raise ValueError("Marker artifact index is invalid")
        if isinstance(raw_ref, dict):
            ref = ArtifactRef.from_dict(raw_ref)
            if (
                ref.kind != "marker_table"
                or ref.scope != "assay"
                or ref.assay != assay.name
            ):
                raise ValueError("Marker artifact ref is invalid")
            status = self.inspect_artifact(ref)
            if not status.complete:
                raise ValueError("Marker artifact is incomplete")
            inputs = status.inputs or {}
            current_selection = self._ensure_cell_selection(cell_key)
            stored_selection = inputs.get("cell_selection")
            if not isinstance(stored_selection, dict) or not (
                self._selection_artifacts_match(
                    ArtifactRef.from_dict(stored_selection),
                    current_selection,
                )
            ):
                raise ValueError("Marker artifact cell selection is stale")
            current_clusters = self._resolve_cell_data_provenance_input(
                group_key,
                cell_key=cell_key,
            )
            stored_clusters = inputs.get("clusters")
            if (
                stored_clusters != current_clusters
                and stored_clusters != current_clusters["artifact"]
            ):
                raise ValueError("Marker artifact cluster labels are stale")
            return as_zarr_group(
                self.zw[artifact_path(ref)],
                name=artifact_path(ref),
            )
        return as_zarr_group(
            markers_group[slot_name],
            name=slot_name,
        )

    def get_markers(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        group_key: str | None = None,
        group_id: str | int | None = None,
        min_score: float = 0.25,
        min_frac_exp: float = 0.2,
    ) -> pd.DataFrame:
        """Return marker features from `run_marker_search`.

        When ``group_id`` is ``None`` (default), markers for every group under
        ``group_key`` are returned in one long table with a ``group_id`` column.
        Pass a specific ``group_id`` to return markers for that group only.
        For a wide export of marker names only, use ``export_markers_to_csv``.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: To run the test on specific subset of cells, provide the name of a boolean column in
                        the cell metadata table.
            group_key: Required parameter. This has to be a column name from cell metadata table.
                       Usually this would be a column denoting cell clusters. Please use the same value as used
                       when ran `run_marker_search`
            group_id: One value from the ``group_key`` column, or ``None`` for all groups.
            min_score: This value dictates how specific the feature value has to be in a group before it is
                       considered a marker for that group. The value has to be greater than 0 but less than or equal to
                       1 (Default value: 0.25)
            min_frac_exp: Minimum fraction of cells in a group that must have a non-zero value for a gene to be
                          considered a marker for that group.

        Returns:
            Pandas dataframe with marker statistics. All-group results include a ``group_id`` column.
        """

        if cell_key is None:
            from_assay, cell_key, _ = self._get_latest_keys(from_assay, cell_key, None)
        if group_key is None:
            raise ValueError(
                "ERROR: Please provide a value for group_key. "
                "This should be same as used for `run_marker_search`"
            )
        assay = self._get_assay(from_assay)
        try:
            g = self._resolve_marker_group(
                from_assay,
                cell_key,
                group_key,
            )
        except KeyError:
            raise KeyError(
                "ERROR: Couldn't find the location of markers. Please make sure that you have already called "
                "`run_marker_search` method with same value of `cell_key` and `group_key`"
            )
        out_cols = list(_MARKER_OUT_COLUMNS)
        gids = sorted(set(assay.cells.fetch(group_key, key=cell_key)))
        if group_id is not None:
            gids = [group_id]

        feature_names = assay.feats.fetch_all("names")
        dfs = []
        for gid in gids:
            group_name = str(gid)
            if group_name in g:
                marker_grp = as_zarr_group(g[group_name], name=group_name)
                df = _load_marker_cluster_frame(
                    g,
                    marker_grp,
                    feature_names,
                    group_id=gid,
                )
            else:
                logger.debug(f"No markers found for {gid} returning empty dataframe")
                df = pd.DataFrame([[] for _ in out_cols], index=out_cols).T
                df["group_id"] = []
                df["feature_name"] = []
                df = df[["group_id", "feature_name"] + list(out_cols[1:])]
            dfs.append(df)
        dfs = pd.concat(dfs)
        return dfs[
            (dfs.score >= min_score) & (dfs.frac_exp >= min_frac_exp)
        ].reset_index(drop=True)

    def export_markers_to_csv(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        group_key: str | None = None,
        csv_filename: str | None = None,
        min_score: float = 0.25,
        min_frac_exp: float = 0.2,
    ) -> None:
        """Export markers of each cluster/group to a CSV file where each column
        contains the marker names sorted by score (descending order, highest
        first). This function does not export the scores of markers as they can
        be obtained using `get_markers` function.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: To run the test on specific subset of cells, provide the name of a boolean column in
                        the cell metadata table.
            group_key: Required parameter. This has to be a column name from cell metadata table.
                       Usually this would be a column denoting cell clusters. Please use the same value as used
                       when ran `run_marker_search`
            csv_filename: Required parameter. Name, with path, of CSV file where the marker table is to be saved.
            min_score: This value dictates how specific the feature value has to be in a group before it is
                       considered a marker for that group. The value has to be greater than 0 but less than or equal to
                       1 (Default value: 0.25)
            min_frac_exp: Minimum fraction of cells in a group that must have a non-zero value for a gene to be
                          considered a marker for that group.

        Returns:
        """
        # Not testing the values of from_assay and cell_key because they will be tested in `get_markers`
        if group_key is None:
            raise ValueError(
                "ERROR: Please provide a value for group_key. "
                "This should be same as used for `run_marker_search`"
            )
        if csv_filename is None:
            raise ValueError(
                "ERROR: Please provide a value for parameter `csv_filename`"
            )
        from_assay, cell_key, _ = self._get_latest_keys(from_assay, cell_key, None)
        clusters = self.cells.fetch(group_key, key=cell_key)
        markers_table = {}
        for group_id in sorted(set(clusters)):
            m = self.get_markers(
                from_assay=from_assay,
                cell_key=cell_key,
                group_key=group_key,
                group_id=group_id,
                min_score=min_score,
                min_frac_exp=min_frac_exp,
            )
            if len(m) > 0:
                markers_table[group_id] = m["feature_name"].reset_index(drop=True)
            else:
                markers_table[group_id] = pd.Series([])
        pd.DataFrame(markers_table).fillna("").to_csv(csv_filename, index=False)
        return None

    def add_grouped_assay(
        self,
        from_assay: str | None = None,
        group_key: str | None = None,
        assay_label: str | None = None,
        exclude_values: list | None = None,
    ) -> None:
        """Add a new assay to the DataStore by grouping together multiple
        features and taking their means. This method requires that the features
        are already assigned a group/cluster identity. The new assay will have
        all the cells but only features that marked by 'feat_key' and contain a
        group identity not present in `exclude_values`.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            group_key: This is mandatory parameter. Name of the column in feature metadata table to be used for
                       grouping features.
            assay_label: This is mandatory parameter. A name for the new assay.
            exclude_values: These groups/clusters will be ignored and not added to new assay. By default, it is set to
                            [-1], this means that all the features that have the group identity of -1 are not used.

        Returns: None
        """

        from ...storage.sharding import write_dense_in_shard_rows

        from ...storage.schema import create_zarr_count_assay, finalize_counts

        if assay_label is None:
            raise ValueError(
                "ERROR: Please provide a value for `assay_label`. "
                "It will be used to create a new assay"
            )
        if group_key is None:
            raise ValueError(
                "ERROR: Please provide a value for `group_key`. "
                "This should be name of the column in the feature attribute table that contains the group/cluster "
                "identity of each feature."
            )

        assay = self._get_assay(from_assay)
        groups = assay.feats.fetch_all(group_key)
        if exclude_values is None:
            exclude_values = [-1]
        group_set = sorted(set(groups).difference(exclude_values))

        module_ids = [f"group_{x}" for x in group_set]
        g = create_zarr_count_assay(
            z=self.zw,
            assay_name=assay_label,
            workspace=self.workspace,
            chunk_size=assay.rawData.chunksize,  # type: ignore
            n_cells=assay.cells.N,
            feat_ids=module_ids,
            feat_names=module_ids,
            dtype="float",
        )

        cell_idx = np.array(list(range(assay.cells.N)))
        n_groups = len(group_set)
        matrix = np.zeros((assay.cells.N, n_groups), dtype=np.float64)
        for n, i in tqdmbar(
            enumerate(group_set), desc="Computing grouped means", total=len(group_set)
        ):
            feat_idx = np.where(groups == i)[0]
            matrix[:, n] = (
                assay.normed(cell_idx=cell_idx, feat_idx=feat_idx)
                .mean(axis=1)
                .compute()
            )
        write_dense_in_shard_rows(
            g,
            lambda start, end: matrix[start:end, :],
            msg="Writing grouped assay",
        )
        finalize_counts(self.zw, assay_label, self.workspace)

        self._load_assays(min_cells=0, custom_assay_types={assay_label: "Assay"})
        self._ini_cell_props(min_features=0, mito_pattern="", ribo_pattern="")
        grouped_assay = self._get_assay(assay_label)
        grouped_assay.attrs["grouped_from_assay"] = assay.name
        grouped_assay.attrs["grouped_group_key"] = group_key
        group_column = as_zarr_array(
            as_zarr_group(assay.z["featureData"], name="featureData")[group_key],
            name=group_key,
        )
        raw_source = group_column.attrs.get("source_artifact")
        if isinstance(raw_source, dict):
            try:
                source_ref = ArtifactRef.from_dict(raw_source)
                source_status = inspect_artifact(self.zw, source_ref)
            except (KeyError, TypeError, ValueError):
                source_ref = None
            if source_ref is not None and source_status.complete:
                grouped_assay.attrs["grouped_group_artifact"] = source_ref.to_dict()
                if "grouped_group_digest" in grouped_assay.attrs:
                    del grouped_assay.attrs["grouped_group_digest"]
            else:
                grouped_assay.attrs["grouped_group_digest"] = _group_assignment_digest(
                    groups
                )
        else:
            grouped_assay.attrs["grouped_group_digest"] = _group_assignment_digest(
                groups
            )

    def add_melded_assay(
        self,
        from_assay: str | None = None,
        external_bed_fn: str | None = None,
        assay_label: str | None = None,
        peaks_col: str = "ids",
        scalar_coeff: float = 1e5,
        renormalization: bool = True,
        assay_type: str = "Assay",
    ) -> None:
        """This method performs "assay melding" and can be only be used for
        assay's wherein features have genomic coordinates. In the process of
        melding the input genomic coordinates from `external_bed_fn` are
        intersected with the assay's features. Based on this intersection a
        mapping is created wherein each coordinate interval maps to one or more
        feature coordinates from the assay.

        This method has been designed for snATAC-Seq data and can be used to quantify accessibility of specific
        genomic loci such as gene bodies, promoters, enhancers, motifs, etc.
        Features from the BED file are retained even when they do not overlap any peak; those zero-count features
        are marked invalid during assay initialization.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            external_bed_fn: This is mandatory parameter. This file should be a BED format file with at least five
                             columns containing: chromosome, start position, end position, feature id and feature name.
                             Coordinates should be in half open format. That means that actual end position is -1
            assay_label: This is mandatory parameter. A name for the new assay.
            peaks_col: The column in feature metadata table that contains the genomic coordinate information of each
                       feature. The genomic coordinates are represented as strings in this format: chr:start-end
                       (Default value: 'ids')
            scalar_coeff: An arbitrary scalar multiplier. Only used when renormalization is True (Default value: 1e5)
            renormalization: Whether to rescale the sum of feature values for each cell to `scalar_coeff`
                         (Default value: True)
            assay_type: The new assay (melded assay) is saved as this type. This can be any type of Assay class from
                        `assay` module. Please provide string representation of class. By default, the assay is assigned
                        a generic class and has a dummy normalization function (Default value: 'Assay')

        Returns:
            None
        """

        from ...features.genomic.melding import coordinate_melding

        if assay_label is None:
            raise ValueError(
                "ERROR: Please provide a value for `assay_label`. "
                "It will be used to create a new assay"
            )
        if external_bed_fn is None:
            raise ValueError(
                "ERROR: Please provide a value for `feature_bed_fn`. "
                "This should be a BED format file with atleast 5 columns."
            )

        assay = self._get_assay(from_assay)
        feature_bed = pd.read_csv(external_bed_fn, header=None, sep="\t").sort_values(
            by=[0, 1]  # type: ignore
        )

        peaks_coords = assay.feats.fetch_all(peaks_col)
        coords_ser = pd.Series(peaks_coords, dtype="object")
        string_mask = coords_ser.map(lambda x: isinstance(x, str))
        colon_counts = coords_ser.str.count(":")
        hyphen_counts = coords_ser.str.split(":").str[-1].str.count("-")
        invalid_mask = (
            ~string_mask
            | colon_counts.ne(1).fillna(True)
            | hyphen_counts.ne(1).fillna(True)
        )
        invalid_coords = invalid_mask.to_numpy(dtype=bool)
        if invalid_coords.any():
            n = int(np.flatnonzero(invalid_coords)[0])
            raise ValueError(
                f"ERROR: Coordinate format check failed for element: {peaks_coords[n]} (position {n}). "
                f"The format should be chr:start-end. Please note the colon and hyphen position"
            )

        coordinate_melding(
            assay,
            workspace=self.workspace,
            feature_bed=feature_bed,
            new_assay_name=assay_label,
            peaks_col=peaks_col,
            scalar_coeff=scalar_coeff,
            renormalization=renormalization,
            peaks_coords=peaks_coords,
        )

        self._load_assays(min_cells=10, custom_assay_types={assay_label: assay_type})
        self._ini_cell_props(min_features=0, mito_pattern=None, ribo_pattern=None)

    def make_bulk(
        self,
        from_assay: str | None = None,
        cell_key: str = "I",
        group_key: str | None = None,
        secondary_group_key: str | None = None,
        aggr_type: Literal["mean", "sum"] = "mean",
        return_fraction: bool = False,
        feature_label: Literal["index", "id", "name"] = "index",
        remove_empty_features: bool = True,
        pseudo_reps: int = 1,
        null_vals: list[Any] | None = None,
        secondary_null_vals: list[Any] | None = None,
        random_seed: int = 4466,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """Merge data from cells to create a bulk profile.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Name of the column in cell metadata table to be used for selecting cells.
            group_key: Required cell metadata column used to group cells.
                Passing ``None`` raises ``ValueError``.
            secondary_group_key: Name of the column in cell metadata table to be used for sub-grouping cells.
            aggr_type: Type of aggregation to be used. Can be either 'mean' or 'sum'. (Default value: 'mean')
            return_fraction: Return the fraction of cells expressing a gene in each group. (Default value: False)
            feature_label: The column in feature metadata table to use as row labels. (Default value: 'index')
            pseudo_reps: Within each group, randomly split cells into this many
                pseudo-replicates. (Default value: 1)
            remove_empty_features: Remove features that are not expressed in any cell. (Default value: True)
            null_vals: Values to be considered as missing values in the `group_key` column. These values will be skipped.
            secondary_null_vals: Values to be considered as missing values in the `secondary_group_key` column.
                                 These values will be skipped.
            random_seed: Seed used when assigning cells to pseudo-replicates.

        Returns:
            A pandas dataframe containing the bulk profile. If `return_fraction` is True, then a tuple of two dataframes
            is returned. The second dataframe contains the fraction of cells expressing each feature in each group.
        """

        def make_reps(v: NDArray[Any], n_reps: int, seed: int) -> list[NDArray[Any]]:
            v_list = list(v)
            random_state = np.random.RandomState(seed)
            shuffled_idx = random_state.choice(v_list, len(v_list), replace=False)
            rep_idx = np.array_split(shuffled_idx, n_reps)
            return [np.array(sorted(x)) for x in rep_idx]

        if pseudo_reps < 1:
            pseudo_reps = 1
        if null_vals is None:
            null_vals = []
        if secondary_null_vals is None:
            secondary_null_vals = []
        if group_key is None:
            raise ValueError("ERROR: Please provide a value for `group_key` parameter")
        else:
            groups = self.cells.fetch_all(group_key)
            active_idx = self.cells.active_index(cell_key)
            groups_set = sorted(set(groups[active_idx]))
        if secondary_group_key is None:
            sec_groups: NDArray[Any] = np.array([None], dtype=object)
            sec_groups_set: list[Any] = [None]
        else:
            sec_groups = self.cells.fetch_all(secondary_group_key)
            sec_groups_set = sorted(set(sec_groups[active_idx]))

        assay = self._get_assay(from_assay)

        vals: dict[str, NDArray[Any]] = {}
        fracs: dict[str, NDArray[Any]] = {}
        all_feat_idx = np.arange(assay.feats.N)
        active_mask = np.zeros(self.cells.N, dtype=bool)
        active_mask[active_idx] = True
        for g in tqdmbar(groups_set):
            if g in null_vals:
                continue
            for sg in sec_groups_set:  # type: ignore
                if sg in secondary_null_vals:
                    continue
                if sg is None and len(sec_groups) == 1:
                    g_idx = np.where((groups == g) & active_mask)[0]
                else:
                    g_idx = np.where((groups == g) & (sec_groups == sg) & active_mask)[
                        0
                    ]
                rep_indices = make_reps(g_idx, pseudo_reps, random_seed)
                for n, idx in enumerate(rep_indices):
                    if sg is None and len(sec_groups) == 1:
                        col_name = f"{g}"
                    else:
                        col_name = f"{g}_{sg}"
                    if pseudo_reps > 1:
                        col_name += f"_Rep{n + 1}"
                    if len(idx) == 0:
                        vals[col_name] = np.zeros(assay.feats.N)
                        continue
                    if aggr_type == "sum":
                        vals[col_name] = controlled_compute(
                            assay.rawData[idx].sum(axis=0), self.nthreads
                        )
                    elif aggr_type == "mean":
                        vals[col_name] = controlled_compute(
                            assay.normed(cell_idx=idx, feat_idx=all_feat_idx).mean(
                                axis=0
                            ),
                            self.nthreads,
                        )
                    else:
                        raise ValueError(
                            "ERROR: `aggr_type` can only be either 'sum' or 'mean'"
                        )
                    if return_fraction:
                        fracs[col_name] = (
                            (assay.rawData[idx] > 0).mean(axis=0).compute()
                        )

        vals_df = pd.DataFrame(vals).fillna(0)

        empty_idx = None
        if remove_empty_features:
            empty_idx = vals_df.sum(axis=1) != 0
            vals_df = vals_df.loc[empty_idx]

        if feature_label == "id":
            vals_df.set_index(
                pd.Series(assay.feats.fetch_all("ids")).reindex(vals_df.index).values,
                inplace=True,
                drop=True,
            )
        elif feature_label == "name":
            vals_df.set_index(
                pd.Series(assay.feats.fetch_all("names")).reindex(vals_df.index).values,
                inplace=True,
                drop=True,
            )

        if return_fraction:
            fracs_df = pd.DataFrame(fracs).fillna(0)
            if empty_idx is not None:
                fracs_df = fracs_df[empty_idx]
            fracs_df.set_index(vals_df.index, inplace=True, drop=True)
            return vals_df, fracs_df
        return vals_df

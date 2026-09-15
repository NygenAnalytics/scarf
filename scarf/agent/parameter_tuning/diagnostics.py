"""Deterministic representation and partition evidence for RNA decisions."""

import hashlib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from ...clustering.leiden import leiden_membership
from ...metadata.rows import (
    read_metadata_missing_rows_chunkwise,
    read_metadata_rows_chunkwise,
)
from ...metadata.selection import resolve_cell_aligned_artifact
from ...quality_control.cell_cycle_genes import (
    g2m_phase_genes,
    g2m_phase_genes_mouse,
    s_phase_genes,
    s_phase_genes_mouse,
)
from ...storage.arrays import create_zarr_dataset
from ...storage.artifact_writer import (
    ArrayRequirement,
    AttributeRequirement,
    finish_artifact,
    plan_artifact,
    start_artifact,
)
from ...storage.artifacts import fingerprint_stored_arrays
from ...storage.feature_selection import read_feature_selection_indices
from ...storage.refs import ArtifactRef
from ...storage.selections import read_stored_selection_indices
from ...storage.types import as_zarr_array
from ...utils.logging import logger
from .contracts import ArtifactRecord, ParameterCandidateEvaluation
from .execution import (
    _cached_candidate_metric,
    _metadata_column_fingerprint,
    diagnostic_call,
    diagnostic_reuse,
)
from .selection import annotate_candidate_dominance

_PCA_DIAGNOSTIC_ARRAYS = (
    "component_variance",
    "explained_variance_ratio",
    "top_loading_feature_indices",
    "top_loading_values",
    "family_enrichment",
    "covariate_association",
    "adjacent_neighbor_overlap",
)
_MAX_DOUBLET_CAPTURES = 512
SCARF_DEFAULT_DIAGNOSTIC_FAMILIES = (
    "mitochondrial",
    "ribosomal",
    "mitoribosomal",
    "cellCycleCcn",
    "hla",
    "h2",
    "histone",
    "sexLinked",
)


@dataclass(frozen=True, slots=True)
class AdvisoryDoubletScores:
    """Advisory score artifacts and their exact cell-axis selections."""

    scores: tuple[ArtifactRef, ...]
    cell_selections: tuple[ArtifactRef, ...]
    native_graph: ArtifactRef
    native_clusters: ArtifactRef
    capture_values: tuple[str, ...] = ()
    score_summaries: tuple[dict[str, float], ...] = ()
    score_quantiles: Mapping[str, float] = field(default_factory=dict)
    capture_coverage: float | None = None
    capture_column: str | None = None
    limitations: tuple[str, ...] = ()


def restore_advisory_doublets(
    evaluation: ParameterCandidateEvaluation,
    *,
    capture_column: str | None,
) -> AdvisoryDoubletScores:
    """Restore the exact doublet inputs from an augmented phase evaluation."""
    score_keys = sorted(
        (key for key in evaluation.artifacts if key.startswith("doubletScore:")),
        key=lambda key: int(key.split(":", 1)[1]),
    )
    if score_keys != [f"doubletScore:{index}" for index in range(len(score_keys))]:
        raise ValueError("Persisted doublet score inventory is incomplete")
    captures = tuple(evaluation.metrics.doubletScoreByCapture)
    if len(captures) != len(score_keys):
        raise ValueError("Persisted doublet capture summaries do not align")
    restored = AdvisoryDoubletScores(
        scores=tuple(_artifact_ref(evaluation, key) for key in score_keys),
        cell_selections=tuple(
            _artifact_ref(evaluation, f"doubletCellSelection:{index}")
            for index in range(len(score_keys))
        ),
        native_graph=_artifact_ref(evaluation, "doubletNativeGraph"),
        native_clusters=_artifact_ref(evaluation, "doubletNativeClusters"),
        capture_values=captures,
        score_summaries=tuple(
            dict(evaluation.metrics.doubletScoreByCapture[capture])
            for capture in captures
        ),
        score_quantiles=dict(evaluation.metrics.doubletScoreQuantiles),
        capture_coverage=evaluation.metrics.doubletCaptureCoverage,
        capture_column=capture_column,
        limitations=tuple(
            warning for warning in evaluation.warnings if "doublet" in warning.lower()
        ),
    )
    diagnostic_reuse("diagnostic.advisoryDoublets", "restored")
    return restored


def _artifact_ref(
    evaluation: ParameterCandidateEvaluation,
    name: str,
) -> ArtifactRef:
    artifact = evaluation.artifacts.get(name)
    if artifact is None:
        raise ValueError(
            f"Candidate {evaluation.candidateId!r} lacks {name!r} evidence"
        )
    return ArtifactRef(
        scope=artifact.scope,
        assay=artifact.assay,
        kind=artifact.kind,
        artifact_id=artifact.artifactId,
    )


def _cluster_labels(store: Any, cluster_ref: ArtifactRef) -> np.ndarray:
    group = store.load_artifact(cluster_ref)
    values = as_zarr_array(group["values"], name="values")
    labels = np.asarray(values[:])
    if labels.ndim != 1:
        raise ValueError("Cluster evidence must be a one-dimensional label vector")
    return labels


def _selected_feature_names(
    store: Any,
    feature_selection: ArtifactRef,
) -> tuple[np.ndarray, np.ndarray]:
    if feature_selection.assay is None:
        raise ValueError("PCA feature selection must belong to one assay")
    indices = read_feature_selection_indices(
        store.zw,
        feature_selection.assay,
        feature_selection,
    ).astype(np.int64, copy=False)
    names = np.asarray(
        store.get_assay(feature_selection.assay).feats.fetch_all("names")
    ).astype(str)
    return indices, names[indices]


def _family_mask(names: np.ndarray, family: str) -> np.ndarray | None:
    upper = np.char.upper(names.astype(str))
    if family == "mitochondrial":
        return np.char.startswith(upper, "MT-")
    if family in {"ribosomal", "ribosomalProtein"}:
        return np.asarray(
            np.logical_or.reduce(
                [np.char.startswith(upper, prefix) for prefix in ("RPS", "RPL")]
            ),
            dtype=bool,
        )
    if family == "mitoribosomal":
        return np.asarray(
            np.logical_or.reduce(
                [np.char.startswith(upper, prefix) for prefix in ("MRPS", "MRPL")]
            ),
            dtype=bool,
        )
    if family == "cellCycleCcn":
        return np.char.startswith(upper, "CCN")
    if family == "cellCycle":
        cycle_genes = {
            *s_phase_genes,
            *g2m_phase_genes,
            *s_phase_genes_mouse,
            *g2m_phase_genes_mouse,
        }
        return np.asarray(
            np.char.startswith(upper, "CCN")
            | np.isin(upper, [value.upper() for value in cycle_genes]),
            dtype=bool,
        )
    if family in {"hla", "HLA"}:
        return np.char.startswith(upper, "HLA-")
    if family in {"h2", "H2"}:
        return np.char.startswith(upper, "H2-")
    if family == "histone":
        return np.char.startswith(upper, "HIST")
    if family in {"sex", "sexLinked"}:
        return np.isin(
            upper,
            [
                "XIST",
                "DDX3Y",
                "USP9Y",
                "EIF1AY",
                "KDM5D",
                "SRY",
                "ZFY",
                "UTY",
                "TMSB4Y",
                "NLGN4Y",
            ],
        )
    if family == "hemoglobin":
        return np.char.startswith(upper, "HB")
    if family == "immuneReceptor":
        return np.asarray(
            np.logical_or.reduce(
                [
                    np.char.startswith(upper, prefix)
                    for prefix in ("IGH", "IGK", "IGL", "TRA", "TRB", "TRD", "TRG")
                ]
            ),
            dtype=bool,
        )
    if family == "stress":
        return np.asarray(
            np.logical_or.reduce(
                [
                    np.char.startswith(upper, prefix)
                    for prefix in ("FOS", "JUN", "HSP", "DUSP", "EGR")
                ]
            ),
            dtype=bool,
        )
    if family == "dissociation":
        return np.isin(
            upper,
            [
                "ATF3",
                "BTG1",
                "BTG2",
                "DUSP1",
                "EGR1",
                "FOS",
                "FOSB",
                "IER2",
                "JUN",
                "JUNB",
                "JUND",
                "ZFP36",
            ],
        )
    return None


def _component_variance(values: Any) -> np.ndarray:
    if len(values.shape) != 2:
        raise ValueError("PCA coordinates must be a two-dimensional array")
    n_rows, n_components = values.shape
    if n_rows < 1 or n_components < 1:
        raise ValueError("PCA coordinates cannot be empty")
    totals = np.zeros(n_components, dtype=np.float64)
    totals_squared = np.zeros(n_components, dtype=np.float64)
    for start in range(0, n_rows, 65_536):
        block = np.asarray(values[start : start + 65_536], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError("PCA coordinates must be finite")
        totals += block.sum(axis=0)
        totals_squared += np.square(block).sum(axis=0)
    variance = totals_squared / n_rows - np.square(totals / n_rows)
    return np.asarray(np.maximum(variance, 0.0), dtype=np.float64)


def _scaled_total_variance(
    store: Any,
    reduction_status: Any,
    *,
    n_rows: int,
    n_features: int,
    feature_selection: ArtifactRef,
) -> float:
    def input_ref(value: Any, label: str) -> ArtifactRef:
        if isinstance(value, ArtifactRef):
            return value
        if isinstance(value, Mapping):
            return ArtifactRef.from_dict(dict(value))
        raise ValueError(f"Candidate PCA lacks its {label} input")

    inputs = getattr(reduction_status, "inputs", None) or {}
    normalized_ref = input_ref(inputs.get("normalized"), "normalized matrix")
    pca_cell_selection = input_ref(
        inputs.get("pca_cell_selection"),
        "PCA cell-selection",
    )
    normalized_status = store.inspect_artifact(normalized_ref)
    normalized_inputs = getattr(normalized_status, "inputs", None) or {}
    normalized_cell_selection = input_ref(
        normalized_inputs.get("cell_selection"),
        "normalized cell-selection",
    )
    normalized_feature_selection = input_ref(
        normalized_inputs.get("feature_selection"),
        "normalized feature-selection",
    )
    if pca_cell_selection != normalized_cell_selection:
        raise ValueError(
            "Explained-variance ratios require PCA fitted on all normalized cells"
        )
    if normalized_feature_selection != feature_selection:
        raise ValueError("PCA loading genes do not match the normalized features")
    normalized_group = store.load_artifact(normalized_ref)
    normalized = as_zarr_array(normalized_group["data"], name="data")
    if normalized.shape != (n_rows, n_features):
        raise ValueError("Candidate PCA and normalized matrix shapes do not align")
    if "feature_sum" in normalized_group and "feature_squared_sum" in normalized_group:
        totals = np.asarray(
            as_zarr_array(normalized_group["feature_sum"], name="feature_sum")[:],
            dtype=np.float64,
        )
        totals_squared = np.asarray(
            as_zarr_array(
                normalized_group["feature_squared_sum"],
                name="feature_squared_sum",
            )[:],
            dtype=np.float64,
        )
    else:
        totals = np.zeros(n_features, dtype=np.float64)
        totals_squared = np.zeros(n_features, dtype=np.float64)
        for start in range(0, n_rows, 8192):
            block = np.asarray(
                normalized[start : start + 8192],
                dtype=np.float64,
            )
            if not np.isfinite(block).all():
                raise ValueError("Normalized PCA input must be finite")
            totals += block.sum(axis=0)
            totals_squared += np.square(block).sum(axis=0)
    if totals.shape != (n_features,) or totals_squared.shape != (n_features,):
        raise ValueError("Normalized PCA feature summaries do not align")
    variance = np.maximum(
        totals_squared / n_rows - np.square(totals / n_rows),
        0.0,
    )
    total_scaled_variance = float(np.count_nonzero(variance))
    if total_scaled_variance <= 0:
        raise ValueError("Feature-scaled PCA input has no non-constant features")
    return total_scaled_variance


def _top_loadings(
    loadings: Any,
    selected_indices: np.ndarray,
    family_masks: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(loadings.shape) != 2 or loadings.shape[0] != len(selected_indices):
        raise ValueError("PCA loadings do not align with selected features")
    if loadings.shape[0] < 1 or loadings.shape[1] < 1:
        raise ValueError("PCA loadings cannot be empty")
    if any(mask.shape != selected_indices.shape for mask in family_masks.values()):
        raise ValueError("PCA family masks must align with selected features")
    top_n = min(20, loadings.shape[0])
    top_rows = np.empty((loadings.shape[1], 0), dtype=np.int64)
    top_values = np.empty((loadings.shape[1], 0), dtype=np.float64)
    for start in range(0, loadings.shape[0], 8192):
        block = np.abs(np.asarray(loadings[start : start + 8192], dtype=np.float64))
        if not np.isfinite(block).all():
            raise ValueError("PCA loadings must be finite")
        block_rows = np.arange(start, start + len(block), dtype=np.int64)
        retained_rows = np.empty(
            (loadings.shape[1], min(top_n, top_rows.shape[1] + len(block))),
            dtype=np.int64,
        )
        retained_values = np.empty(retained_rows.shape, dtype=np.float64)
        for component in range(loadings.shape[1]):
            candidate_rows = np.concatenate((top_rows[component], block_rows))
            candidate_values = np.concatenate(
                (top_values[component], block[:, component])
            )
            order = np.lexsort((candidate_rows, -candidate_values))[:top_n]
            retained_rows[component] = candidate_rows[order]
            retained_values[component] = candidate_values[order]
        top_rows = retained_rows
        top_values = retained_values
    top_indices = selected_indices[top_rows]
    enrichment = np.zeros(
        (len(family_masks), loadings.shape[1]),
        dtype=np.float64,
    )
    for family_index, mask in enumerate(family_masks.values()):
        for component in range(loadings.shape[1]):
            background = float(mask.mean())
            enrichment[family_index, component] = (
                float(mask[top_rows[component]].mean()) / background
                if background > 0
                else 0.0
            )
    return top_indices, top_values, enrichment


def _aligned_metadata_values(
    store: Any,
    cell_selection: ArtifactRef,
    column: str,
) -> np.ndarray:
    indices = read_stored_selection_indices(
        store.zw,
        cell_selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    ).astype(np.int64, copy=False)
    values = np.asarray(read_metadata_rows_chunkwise(store.cells, column, indices))
    if values.shape != (len(indices),):
        raise ValueError(f"Metadata column {column!r} does not align with PCA")
    missing = read_metadata_missing_rows_chunkwise(store.cells, column, indices)
    if missing is not None and np.any(missing):
        values = values.astype(object)
        values[missing] = None
    return values


def _numeric_association(coordinates: Any, values: np.ndarray) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(
        dtype=np.float64
    )
    valid = np.isfinite(numeric)
    n_rows, n_components = coordinates.shape
    total_x = np.zeros(n_components, dtype=np.float64)
    total_x2 = np.zeros(n_components, dtype=np.float64)
    total_xy = np.zeros(n_components, dtype=np.float64)
    total_y = float(numeric[valid].sum())
    total_y2 = float(np.square(numeric[valid]).sum())
    for start in range(0, n_rows, 65_536):
        block = np.asarray(coordinates[start : start + 65_536], dtype=np.float64)
        selected = valid[start : start + len(block)]
        y = numeric[start : start + len(block)][selected]
        block = block[selected]
        total_x += block.sum(axis=0)
        total_x2 += np.square(block).sum(axis=0)
        total_xy += (block * y[:, None]).sum(axis=0)
    n_rows = int(valid.sum())
    numerator = n_rows * total_xy - total_x * total_y
    denominator = np.sqrt(
        np.maximum(n_rows * total_x2 - np.square(total_x), 0.0)
        * max(n_rows * total_y2 - total_y * total_y, 0.0)
    )
    return np.asarray(
        np.divide(
            np.abs(numerator),
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0,
        ),
        dtype=np.float64,
    )


def _categorical_association(coordinates: Any, values: np.ndarray) -> np.ndarray:
    valid = np.asarray(~pd.isna(values), dtype=bool)
    labels = values[valid].astype(str)
    _levels, codes = np.unique(labels, return_inverse=True)
    n_rows, n_components = coordinates.shape
    totals = np.zeros(n_components, dtype=np.float64)
    totals_squared = np.zeros(n_components, dtype=np.float64)
    if not len(codes):
        return np.zeros(n_components, dtype=np.float64)
    all_codes = np.full(len(values), -1, dtype=np.int64)
    all_codes[valid] = codes
    group_sums = np.zeros((int(codes.max()) + 1, n_components), dtype=np.float64)
    group_counts = np.bincount(codes, minlength=group_sums.shape[0]).astype(np.float64)
    for start in range(0, n_rows, 65_536):
        block = np.asarray(coordinates[start : start + 65_536], dtype=np.float64)
        selected = valid[start : start + len(block)]
        block_codes = all_codes[start : start + len(block)][selected]
        block = block[selected]
        totals += block.sum(axis=0)
        totals_squared += np.square(block).sum(axis=0)
        np.add.at(group_sums, block_codes, block)
    n_rows = int(valid.sum())
    grand_mean = totals / n_rows
    group_means = np.divide(
        group_sums,
        group_counts[:, None],
        out=np.zeros_like(group_sums),
        where=group_counts[:, None] > 0,
    )
    between = (
        group_counts[:, None] * np.square(group_means - grand_mean[None, :])
    ).sum(axis=0)
    total = totals_squared - n_rows * np.square(grand_mean)
    return np.asarray(
        np.sqrt(
            np.divide(
                between,
                total,
                out=np.zeros_like(between),
                where=total > 0,
            )
        ),
        dtype=np.float64,
    )


def _covariate_associations(
    store: Any,
    cell_selection: ArtifactRef,
    coordinates: Any,
    columns: Sequence[str],
    roles: Sequence[str],
    column_kinds: Mapping[str, str] | None = None,
    support: dict[str, Any] | None = None,
    column_artifacts: Mapping[str, ArtifactRef] | None = None,
) -> np.ndarray:
    if len(columns) != len(roles):
        raise ValueError("PCA covariate columns and roles must align")
    associations = np.zeros((len(columns), coordinates.shape[1]), dtype=np.float64)
    from ..experimental_context.characterization import _infer_kind

    for index, column in enumerate(columns):
        artifact = (column_artifacts or {}).get(column)
        values = (
            _aligned_metadata_values(store, cell_selection, column)
            if artifact is None
            else np.asarray(
                resolve_cell_aligned_artifact(
                    store.zw,
                    artifact,
                    cell_selection=cell_selection,
                    expected_kind="quality_metric",
                ).values
            )
        )
        if values.shape != (coordinates.shape[0],):
            raise ValueError(f"Covariate {column!r} does not align with PCA rows")
        kind = (column_kinds or {}).get(column) or (
            "continuous" if artifact is not None else _infer_kind(values)
        )
        if artifact is not None and kind != "continuous":
            raise ValueError(
                "QC percentage artifacts require continuous covariate evidence"
            )
        if kind not in {"continuous", "categorical"}:
            raise ValueError(f"Unknown covariate kind for {column!r}: {kind!r}")
        valid = np.asarray(~pd.isna(values), dtype=bool)
        if kind == "continuous":
            numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(
                dtype=float
            )
            valid &= np.isfinite(numeric)
            associations[index] = _numeric_association(coordinates, values)
        else:
            if len(pd.unique(values[valid])) < int(valid.sum()):
                associations[index] = _categorical_association(coordinates, values)
        levels = len(pd.unique(values[valid]))
        if support is not None:
            support[column] = {
                "kind": kind,
                "method": "absolutePearson"
                if kind == "continuous"
                else "correlationRatio",
                "completeRows": int(valid.sum()),
                "missingRows": int((~valid).sum()),
                "levels": levels,
                "status": "computed"
                if int(valid.sum()) >= 2
                and levels >= 2
                and (kind == "continuous" or levels < int(valid.sum()))
                else "notComputed",
            }
    return associations


def _neighbor_overlap(store: Any, left: ArtifactRef, right: ArtifactRef) -> float:
    left_values = as_zarr_array(store.load_artifact(left)["indices"], name="indices")
    right_values = as_zarr_array(store.load_artifact(right)["indices"], name="indices")
    if left_values.shape != right_values.shape or len(left_values.shape) != 2:
        raise ValueError("Adjacent PCA neighbor artifacts must align exactly")
    total = 0.0
    rows = left_values.shape[0]
    for start in range(0, rows, 4096):
        left_block = np.asarray(left_values[start : start + 4096])
        right_block = np.asarray(right_values[start : start + 4096])
        intersection = (
            (left_block[:, :, None] == right_block[:, None, :]).any(axis=2).sum(axis=1)
        )
        union = left_block.shape[1] + right_block.shape[1] - intersection
        total += float(np.divide(intersection, union).sum())
    return total / rows


def _write_pca_diagnostic(
    store: Any,
    evaluation: ParameterCandidateEvaluation,
    *,
    feature_selection: ArtifactRef,
    selected_indices: np.ndarray,
    family_masks: Mapping[str, np.ndarray],
    covariate_columns: Sequence[str],
    covariate_roles: Sequence[str],
    adjacent_overlap: float | None,
    column_kinds: Mapping[str, str] | None = None,
    column_artifacts: Mapping[str, ArtifactRef] | None = None,
) -> tuple[
    ArtifactRef,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    if column_artifacts:
        column_kinds = dict(column_kinds or {})
        for column, artifact in column_artifacts.items():
            if (
                artifact.kind != "quality_metric"
                or column_kinds.get(column, "continuous") != "continuous"
            ):
                raise ValueError(
                    "QC percentage artifacts require continuous covariate evidence"
                )
            column_kinds[column] = "continuous"
    reduction = _artifact_ref(evaluation, "pca")
    neighbors = _artifact_ref(evaluation, "neighbors")
    reduction_status = store.inspect_artifact(reduction)
    reduction_parameters = getattr(reduction_status, "parameters", None) or {}
    if reduction_parameters.get("feat_scaling") is not True:
        raise ValueError(
            "Explained-variance ratios require the feature-scaled candidate PCA"
        )
    reduction_group = store.load_artifact(reduction)
    coordinates = as_zarr_array(reduction_group["data"], name="data")
    loadings = as_zarr_array(reduction_group["loadings"], name="loadings")
    dimensions = int(coordinates.shape[1])
    top_n = min(20, len(selected_indices))
    if loadings.shape != (len(selected_indices), dimensions):
        raise ValueError("PCA loadings do not align with selected features")
    planned = plan_artifact(
        store.zw,
        scope="assay",
        assay=reduction.assay,
        kind="feature_summary",
        operation="diagnose_pca_representation",
        parameters={
            "family_names": list(family_masks),
            "covariate_columns": list(covariate_columns),
            "covariate_roles": list(covariate_roles),
            "covariate_kinds": {
                column: (column_kinds or {}).get(column, "inferred")
                for column in covariate_columns
            },
            "covariate_method": "typedCompleteCaseAssociation",
            "top_loading_count": top_n,
            "family_mask_fingerprints": {
                family: hashlib.sha256(
                    np.asarray(mask, dtype=bool).tobytes()
                ).hexdigest()
                for family, mask in family_masks.items()
            },
            "covariate_fingerprints": {
                column: (
                    column_artifacts[column].to_dict()
                    if column_artifacts and column in column_artifacts
                    else _metadata_column_fingerprint(store.cells, column)
                )
                for column in covariate_columns
            },
            "adjacent_neighbor_overlap": adjacent_overlap,
            "explained_variance_basis": "scaled_nonconstant_features",
        },
        inputs={
            "reduction": reduction,
            "neighbors": neighbors,
            "feature_selection": feature_selection,
            "covariate_artifacts": dict(column_artifacts or {}),
        },
        execution_options={},
        invalidate_cache=False,
        required_arrays=(
            ArrayRequirement(
                "component_variance", shape=(dimensions,), dtype=np.float64
            ),
            ArrayRequirement(
                "explained_variance_ratio", shape=(dimensions,), dtype=np.float64
            ),
            ArrayRequirement(
                "top_loading_feature_indices", shape=(dimensions, top_n), dtype=np.int64
            ),
            ArrayRequirement(
                "top_loading_values", shape=(dimensions, top_n), dtype=np.float64
            ),
            ArrayRequirement(
                "family_enrichment",
                shape=(len(family_masks), dimensions),
                dtype=np.float64,
            ),
            ArrayRequirement(
                "covariate_association",
                shape=(len(covariate_columns), dimensions),
                dtype=np.float64,
            ),
            ArrayRequirement("adjacent_neighbor_overlap", shape=(1,), dtype=np.float64),
        ),
        required_attributes=(
            AttributeRequirement("family_names", expected_types=(list,)),
            AttributeRequirement("covariate_columns", expected_types=(list,)),
            AttributeRequirement("covariate_roles", expected_types=(list,)),
            AttributeRequirement("covariate_support", expected_types=(dict,)),
            AttributeRequirement("payload_fingerprint", expected_types=(str,)),
        ),
    )
    if planned.reused:
        group = store.load_artifact(planned.ref)
        if (
            fingerprint_stored_arrays(group, _PCA_DIAGNOSTIC_ARRAYS)
            != group.attrs["payload_fingerprint"]
        ):
            raise ValueError("Stored PCA diagnostic payload fingerprint does not match")
        restored = (
            planned.ref,
            np.asarray(
                as_zarr_array(group["component_variance"], name="component_variance")[:]
            ),
            np.asarray(
                as_zarr_array(
                    group["explained_variance_ratio"], name="explained_variance_ratio"
                )[:]
            ),
            np.asarray(
                as_zarr_array(
                    group["top_loading_feature_indices"],
                    name="top_loading_feature_indices",
                )[:]
            ),
            np.asarray(
                as_zarr_array(group["top_loading_values"], name="top_loading_values")[:]
            ),
            np.asarray(
                as_zarr_array(group["family_enrichment"], name="family_enrichment")[:]
            ),
            np.asarray(
                as_zarr_array(
                    group["covariate_association"], name="covariate_association"
                )[:]
            ),
        )
        diagnostic_reuse("diagnostic.pca", "artifactReuses")
        return restored
    component_variance = diagnostic_call(
        "metric.pcaVariance", _component_variance, coordinates
    )
    total_scaled_variance = diagnostic_call(
        "metric.pcaScaledVariance",
        _scaled_total_variance,
        store,
        reduction_status,
        n_rows=int(coordinates.shape[0]),
        n_features=len(selected_indices),
        feature_selection=feature_selection,
    )
    explained_variance_ratio = np.clip(
        component_variance / total_scaled_variance,
        0.0,
        1.0,
    )
    top_indices, top_values, family_enrichment = diagnostic_call(
        "metric.pcaLoadings",
        _top_loadings,
        loadings,
        selected_indices,
        family_masks,
    )
    covariate_support: dict[str, Any] = {}
    associations = (
        diagnostic_call(
            "metric.pcaCovariates",
            _covariate_associations,
            store,
            ArtifactRef(
                scope=evaluation.cellSelection.scope,
                assay=evaluation.cellSelection.assay,
                kind=evaluation.cellSelection.kind,
                artifact_id=evaluation.cellSelection.artifactId,
            ),
            coordinates,
            covariate_columns,
            covariate_roles,
            column_kinds,
            covariate_support,
            column_artifacts,
        )
        if evaluation.cellSelection is not None
        else np.zeros((len(covariate_columns), coordinates.shape[1]), dtype=np.float64)
    )
    overlap_array = np.asarray(
        [np.nan if adjacent_overlap is None else adjacent_overlap],
        dtype=np.float64,
    )
    payload = {
        "component_variance": component_variance,
        "explained_variance_ratio": explained_variance_ratio,
        "top_loading_feature_indices": top_indices,
        "top_loading_values": top_values,
        "family_enrichment": family_enrichment,
        "covariate_association": associations,
        "adjacent_neighbor_overlap": overlap_array,
    }
    if not planned.reused:
        group = start_artifact(store.zw, planned)
        for name, values in payload.items():
            chunks = tuple(max(1, min(size, 4096)) for size in values.shape)
            array = create_zarr_dataset(
                group,
                name,
                chunks,
                values.dtype,
                values.shape,
            )
            array[:] = values
        group.attrs["family_names"] = list(family_masks)
        group.attrs["covariate_columns"] = list(covariate_columns)
        group.attrs["covariate_roles"] = list(covariate_roles)
        group.attrs["covariate_support"] = covariate_support
        group.attrs["payload_fingerprint"] = fingerprint_stored_arrays(
            group,
            _PCA_DIAGNOSTIC_ARRAYS,
        )
        finish_artifact(group, planned)
    return (
        planned.ref,
        component_variance,
        explained_variance_ratio,
        top_indices,
        top_values,
        family_enrichment,
        associations,
    )


def augment_pca_evaluations(
    store: Any,
    evaluations: Sequence[ParameterCandidateEvaluation],
    *,
    feature_selection: ArtifactRef,
    nominated_families: Sequence[str],
    protected_families: Sequence[str],
    technical_columns: Sequence[str],
    protected_columns: Sequence[str],
    qc_columns: Sequence[str],
    batch_columns: Sequence[str] = (),
    column_kinds: Mapping[str, str] | None = None,
    qc_artifacts: Mapping[str, ArtifactRef] | None = None,
) -> tuple[ParameterCandidateEvaluation, ...]:
    """Attach persisted PCA loading, variance, topology, and covariate evidence."""
    selected_indices, selected_names = _selected_feature_names(
        store,
        feature_selection,
    )
    family_masks = {
        family: mask
        for family in dict.fromkeys([*nominated_families, *protected_families])
        for mask in [_family_mask(selected_names, family)]
        if mask is not None and bool(mask.any())
    }
    requested_role_columns = {
        "technical": tuple(technical_columns),
        "batch": tuple(batch_columns or technical_columns),
        "protected": tuple(protected_columns),
        "qc": tuple(qc_columns),
    }
    columns: list[str] = []
    roles: list[str] = []
    for role, values in (
        ("technical", technical_columns),
        ("batch", batch_columns),
        ("protected", protected_columns),
        ("qc", qc_columns),
    ):
        for column in values:
            if (
                column in store.cells.columns or column in (qc_artifacts or {})
            ) and column not in columns:
                columns.append(column)
                roles.append(role)
    completed = [
        evaluation
        for evaluation in evaluations
        if evaluation.status == "done"
        and evaluation.eligible
        and evaluation.cellSelection is not None
        and "pca" in evaluation.artifacts
        and "neighbors" in evaluation.artifacts
    ]
    previous_by_id: dict[str, float | None] = {}
    previous: ParameterCandidateEvaluation | None = None
    for evaluation in sorted(
        completed,
        key=lambda value: value.parameters.dimensions,
    ):
        overlap = (
            diagnostic_call(
                "metric.neighborOverlap",
                _neighbor_overlap,
                store,
                _artifact_ref(previous, "neighbors"),
                _artifact_ref(evaluation, "neighbors"),
            )
            if previous is not None
            else None
        )
        previous_by_id[evaluation.candidateId] = overlap
        previous = evaluation

    augmented: list[ParameterCandidateEvaluation] = []
    for evaluation in evaluations:
        if evaluation.candidateId not in previous_by_id:
            augmented.append(evaluation)
            continue
        (
            diagnostic,
            variance,
            explained_variance_ratio,
            top_indices,
            _top_values,
            family_enrichment,
            associations,
        ) = diagnostic_call(
            "diagnostic.pca",
            _write_pca_diagnostic,
            store,
            evaluation,
            feature_selection=feature_selection,
            selected_indices=selected_indices,
            family_masks=family_masks,
            covariate_columns=columns,
            covariate_roles=roles,
            adjacent_overlap=previous_by_id[evaluation.candidateId],
            column_kinds=column_kinds,
            column_artifacts=qc_artifacts,
        )
        family_maxima = {
            family: float(family_enrichment[index].max(initial=0.0))
            for index, family in enumerate(family_masks)
        }
        family_by_component = {
            f"PC{component + 1}": {
                family: float(family_enrichment[family_index, component])
                for family_index, family in enumerate(family_masks)
            }
            for component in range(family_enrichment.shape[1])
        }
        feature_name_by_index = dict(
            zip(selected_indices.tolist(), selected_names.tolist(), strict=True)
        )
        top_loading_genes = {
            f"PC{component + 1}": [
                str(feature_name_by_index[int(index)])
                for index in top_indices[component]
            ]
            for component in range(top_indices.shape[0])
        }
        support = dict(store.load_artifact(diagnostic).attrs["covariate_support"])
        column_index = {
            column: index
            for index, column in enumerate(columns)
            if support.get(column, {}).get("status") == "computed"
        }
        unsupported_covariates = [
            column for column in columns if column not in column_index
        ]
        component_associations = {
            role: {
                column: associations[column_index[column]].tolist()
                for column in role_columns
                if column in column_index
            }
            for role, role_columns in requested_role_columns.items()
        }
        role_associations = {
            role: {
                column: float(associations[column_index[column]].max(initial=0.0))
                for column in role_columns
                if column in column_index
            }
            for role, role_columns in requested_role_columns.items()
        }
        metrics = evaluation.metrics.model_copy(
            update={
                "componentVariance": variance.tolist(),
                "pcaExplainedVarianceRatio": explained_variance_ratio.tolist(),
                "pcaCumulativeExplainedVarianceRatio": np.cumsum(
                    explained_variance_ratio
                )
                .clip(max=1.0)
                .tolist(),
                "topLoadingGenes": top_loading_genes,
                "loadingFamilyEnrichment": family_maxima,
                "loadingFamilyEnrichmentByComponent": family_by_component,
                "pcaComponentAssociations": component_associations,
                "batchPcaAssociation": role_associations["batch"],
                "technicalPcaAssociation": role_associations["technical"],
                "protectedPcaAssociation": role_associations["protected"],
                "qcPcaAssociation": role_associations["qc"],
                "neighborPrefixOverlap": previous_by_id[evaluation.candidateId],
            }
        )
        artifact = ArtifactRecord.from_ref(diagnostic)
        augmented.append(
            evaluation.model_copy(
                update={
                    "metrics": metrics,
                    "warnings": [
                        *evaluation.warnings,
                        *(
                            f"PCA association for {column!r} is unavailable with {support.get(column, {}).get('completeRows', 0)} complete rows and {support.get(column, {}).get('levels', 0)} distinct values."
                            for column in unsupported_covariates
                        ),
                    ],
                    "artifacts": {
                        **evaluation.artifacts,
                        "representationDiagnostic": artifact,
                    },
                    "evidenceIds": list(
                        dict.fromkeys(
                            [
                                *evaluation.evidenceIds,
                                f"candidate:{evaluation.candidateId}:pcaVariance",
                                (
                                    f"candidate:{evaluation.candidateId}:"
                                    "pcaExplainedVariance"
                                ),
                                f"candidate:{evaluation.candidateId}:pcaLoadings",
                                f"candidate:{evaluation.candidateId}:pcaCovariates",
                                *(
                                    [
                                        f"candidate:{evaluation.candidateId}:neighborPrefixOverlap"
                                    ]
                                    if previous_by_id[evaluation.candidateId]
                                    is not None
                                    else []
                                ),
                            ]
                        )
                    ),
                }
            )
        )
    return annotate_candidate_dominance(augmented)


def _bounded_score_summary(
    values: Any,
    *,
    maximum_sample_size: int,
) -> tuple[dict[str, float], np.ndarray]:
    if maximum_sample_size < 1:
        raise ValueError("maximum_sample_size must be positive")
    if len(values.shape) != 1 or values.shape[0] < 1:
        raise ValueError("Doublet scores must be one non-empty vector")
    n_values = int(values.shape[0])
    stride = max(1, (n_values + maximum_sample_size - 1) // maximum_sample_size)
    sampled: list[np.ndarray] = []
    minimum = float("inf")
    maximum = float("-inf")
    for start in range(0, n_values, 65_536):
        block = np.asarray(values[start : start + 65_536], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError("Doublet scores must be finite")
        minimum = min(minimum, float(block.min()))
        maximum = max(maximum, float(block.max()))
        offset = (-start) % stride
        sampled.append(block[offset::stride])
    sample = np.concatenate(sampled)
    return (
        {
            "nCells": float(n_values),
            "sampleSize": float(len(sample)),
            "minimum": minimum,
            "p50": float(np.quantile(sample, 0.5)),
            "p90": float(np.quantile(sample, 0.9)),
            "p95": float(np.quantile(sample, 0.95)),
            "p99": float(np.quantile(sample, 0.99)),
            "maximum": maximum,
        },
        sample,
    )


def _build_advisory_doublet_scores(
    store: Any,
    *,
    scores: Sequence[ArtifactRef],
    cell_selections: Sequence[ArtifactRef],
    native_graph: ArtifactRef,
    native_clusters: ArtifactRef,
    parent_selection: ArtifactRef,
    capture_values: Sequence[str],
    capture_column: str | None,
    limitations: Sequence[str],
) -> AdvisoryDoubletScores:
    score_refs = tuple(scores)
    selections = tuple(cell_selections)
    captures = tuple(capture_values)
    if not score_refs or not (len(score_refs) == len(selections) == len(captures)):
        raise ValueError("Doublet score artifacts require aligned capture summaries")
    parent_indices = read_stored_selection_indices(
        store.zw,
        parent_selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    if len(parent_indices) < 1:
        raise ValueError("Doublet evidence parent selection cannot be empty")
    score_arrays = tuple(
        as_zarr_array(store.load_artifact(score_ref)["values"], name="values")
        for score_ref in score_refs
    )
    total_score_cells = sum(int(values.shape[0]) for values in score_arrays)
    if total_score_cells < 1:
        raise ValueError("Doublet score artifacts cannot all be empty")
    summaries: list[dict[str, float]] = []
    samples: list[np.ndarray] = []
    covered_cells = 0
    for values, selection_ref in zip(score_arrays, selections, strict=True):
        selection_indices = read_stored_selection_indices(
            store.zw,
            selection_ref,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
        if values.shape != selection_indices.shape:
            raise ValueError("Doublet scores do not align with their cell selection")
        summary, sample = diagnostic_call(
            "metric.doubletScoreSummary",
            _bounded_score_summary,
            values,
            maximum_sample_size=max(
                1,
                int(65_536 * int(values.shape[0]) / total_score_cells),
            ),
        )
        summaries.append(summary)
        samples.append(sample)
        covered_cells += len(selection_indices)
    combined = np.concatenate(samples)
    aggregate = {
        "p50": float(np.quantile(combined, 0.5)),
        "p90": float(np.quantile(combined, 0.9)),
        "p95": float(np.quantile(combined, 0.95)),
        "p99": float(np.quantile(combined, 0.99)),
        "maximum": max(summary["maximum"] for summary in summaries),
    }
    reported_limitations = list(limitations)
    if total_score_cells > 65_536:
        reported_limitations.append(
            "Doublet score quantiles use deterministic bounded samples above "
            "65,536 scored cells."
        )
    return AdvisoryDoubletScores(
        scores=score_refs,
        cell_selections=selections,
        native_graph=native_graph,
        native_clusters=native_clusters,
        capture_values=captures,
        score_summaries=tuple(summaries),
        score_quantiles=aggregate,
        capture_coverage=min(1.0, covered_cells / len(parent_indices)),
        capture_column=capture_column,
        limitations=tuple(reported_limitations),
    )


def _select_capture_cells(
    store: Any,
    parent: ArtifactRef,
    *,
    column: str,
    value: Any,
    active_indices: np.ndarray,
    active_values: np.ndarray,
) -> tuple[ArtifactRef, int]:
    if active_indices.shape != active_values.shape:
        raise ValueError("Capture values must align with the selected cells")
    labels = active_values.astype(str)
    selected = labels == str(value)
    expected = np.zeros(store.cells.N, dtype=bool)
    expected[active_indices] = selected
    reference = diagnostic_call(
        "core.captureSelection",
        store.filter_cells,
        [column],
        [value],
        [value],
        cell_selection=parent,
        keep_bounds=True,
        invalidate_cache=False,
    )
    stored_array = cast(Any, store.load_artifact(reference)["values"])
    stored = np.asarray(stored_array[:], dtype=bool)
    if not np.array_equal(stored, expected):
        raise RuntimeError("Capture selection does not match its validated evidence")
    return reference, int(stored.sum())


def resolve_native_doublet_inputs(
    store: Any,
    selected: ParameterCandidateEvaluation,
    evaluations: Sequence[ParameterCandidateEvaluation],
) -> tuple[ArtifactRef, ArtifactRef]:
    """Return or reconstruct the parameter-matched uncorrected graph and clusters."""
    parameters = selected.parameters
    exact_native = next(
        (
            evaluation
            for evaluation in evaluations
            if evaluation.status == "done"
            and evaluation.eligible
            and not evaluation.parameters.useHarmony
            and evaluation.parameters.dimensions == parameters.dimensions
            and evaluation.parameters.neighborsK == parameters.neighborsK
            and evaluation.parameters.leidenResolution == parameters.leidenResolution
            and "clusters" in evaluation.artifacts
            and "connectivityMap" in evaluation.artifacts
        ),
        None,
    )
    if exact_native is not None:
        inputs = (
            _artifact_ref(exact_native, "clusters"),
            _artifact_ref(exact_native, "connectivityMap"),
        )
        diagnostic_reuse("diagnostic.nativeDoubletInputs", "artifactReuses")
        return inputs
    if not selected.parameters.useHarmony:
        inputs = (
            _artifact_ref(selected, "clusters"),
            _artifact_ref(selected, "connectivityMap"),
        )
        diagnostic_reuse("diagnostic.nativeDoubletInputs", "artifactReuses")
        return inputs
    reduction = _artifact_ref(selected, "pca")
    ann = diagnostic_call(
        "core.nativeDoubletAnn",
        store.build_ann_index,
        reduction,
        ann_metric="l2",
        ann_parallel=False,
        rand_state=4444,
        invalidate_cache=False,
    )
    neighbors = diagnostic_call(
        "core.nativeDoubletNeighbors",
        store.query_neighbors,
        ann,
        coordinates=reduction,
        k=parameters.neighborsK,
        invalidate_cache=False,
    )
    graph = diagnostic_call(
        "core.nativeDoubletGraph",
        store.build_connectivity_map,
        neighbors,
        local_connectivity=1.0,
        bandwidth=1.5,
        invalidate_cache=False,
    )
    clusters = diagnostic_call(
        "core.nativeDoubletPartition",
        store.run_leiden_clustering,
        graph,
        resolution=parameters.leidenResolution,
        backend="igraph",
        symmetric_graph=False,
        graph_upper_only=False,
        random_seed=4444,
        invalidate_cache=False,
    )
    return clusters, graph


def score_advisory_doublets(
    store: Any,
    selected: ParameterCandidateEvaluation,
    evaluations: Sequence[ParameterCandidateEvaluation],
    *,
    assay: str,
    feature_selection: ArtifactRef,
    capture_column: str | None,
) -> AdvisoryDoubletScores:
    """Score doublet evidence without making singlet or removal decisions."""
    if selected.cellSelection is None:
        raise ValueError("Doublet scoring requires an exact selected cell axis")
    parent_selection = ArtifactRef(
        scope=selected.cellSelection.scope,
        assay=selected.cellSelection.assay,
        kind=selected.cellSelection.kind,
        artifact_id=selected.cellSelection.artifactId,
    )
    native_clusters, native_graph = resolve_native_doublet_inputs(
        store,
        selected,
        evaluations,
    )
    feature_ids = np.asarray(store.get_assay(assay).feats.fetch_all("ids")).astype(str)
    unique_ids, id_counts = np.unique(feature_ids, return_counts=True)
    duplicate_ids = unique_ids[id_counts > 1]
    if duplicate_ids.size:
        examples = duplicate_ids[:5].tolist()
        limitation = (
            f"Advisory doublet scoring was not run for assay {assay!r} because "
            f"{duplicate_ids.size} feature identifiers are duplicated. "
            "Simulated-to-observed mapping requires unique identifiers. "
            f"Examples: {examples}."
        )
        logger.warning(limitation)
        return AdvisoryDoubletScores(
            scores=(),
            cell_selections=(),
            native_graph=native_graph,
            native_clusters=native_clusters,
            capture_column=capture_column,
            limitations=(limitation,),
        )
    limitations: list[str] = []
    if capture_column is None or capture_column not in store.cells.columns:
        score = diagnostic_call(
            "core.doubletDetection",
            store.run_doublet_detection,
            native_clusters,
            native_graph,
            from_assay=assay,
            invalidate_cache=False,
        )
        limitations.append(
            "Physical capture identity was unavailable, so advisory doublet "
            "scores were computed across the selected dataset."
        )
        return _build_advisory_doublet_scores(
            store,
            scores=(score,),
            cell_selections=(parent_selection,),
            native_graph=native_graph,
            native_clusters=native_clusters,
            parent_selection=parent_selection,
            capture_values=("allSelectedCells",),
            capture_column=None,
            limitations=limitations,
        )

    active_indices = read_stored_selection_indices(
        store.zw,
        parent_selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    ).astype(np.int64, copy=False)
    capture_values = read_metadata_rows_chunkwise(
        store.cells,
        capture_column,
        active_indices,
    )
    capture_labels = capture_values.astype(str)
    unique_capture_labels, first_capture_indices = np.unique(
        capture_labels,
        return_index=True,
    )
    capture_groups = unique_capture_labels.tolist()
    raw_capture_values = {
        str(label): capture_values[int(index)]
        for label, index in zip(
            unique_capture_labels,
            first_capture_indices,
            strict=True,
        )
    }
    if len(capture_groups) > _MAX_DOUBLET_CAPTURES:
        raise ValueError(
            "Physical capture column exceeds the advisory doublet limit of "
            f"{_MAX_DOUBLET_CAPTURES} values"
        )
    if len(capture_groups) == 1:
        score = diagnostic_call(
            "core.doubletDetection",
            store.run_doublet_detection,
            native_clusters,
            native_graph,
            from_assay=assay,
            invalidate_cache=False,
        )
        return _build_advisory_doublet_scores(
            store,
            scores=(score,),
            cell_selections=(parent_selection,),
            native_graph=native_graph,
            native_clusters=native_clusters,
            parent_selection=parent_selection,
            capture_values=(capture_groups[0],),
            capture_column=capture_column,
            limitations=limitations,
        )

    n_features = len(read_feature_selection_indices(store.zw, assay, feature_selection))
    scores: list[ArtifactRef] = []
    selections: list[ArtifactRef] = []
    scored_captures: list[str] = []
    for capture_value in capture_groups:
        capture_selection, capture_cells = _select_capture_cells(
            store,
            parent_selection,
            column=capture_column,
            value=raw_capture_values[capture_value],
            active_indices=active_indices,
            active_values=capture_values,
        )
        dimensions = min(
            selected.parameters.dimensions,
            capture_cells - 1,
            n_features - 1,
        )
        neighbors_k = min(selected.parameters.neighborsK, capture_cells - 1)
        if dimensions < 2 or neighbors_k < 2:
            limitations.append(
                "Advisory doublet scores were not computed for capture "
                f"{capture_value!r} because it contains only {capture_cells} "
                "selected cells."
            )
            continue
        normalized = diagnostic_call(
            "core.captureNormalization",
            store.run_normalization,
            capture_selection,
            features=feature_selection,
            log_transform=True,
            renormalize_subset=True,
            invalidate_cache=False,
        )
        reduction = diagnostic_call(
            "core.capturePca",
            store.run_pca,
            normalized,
            dims=dimensions,
            feat_scaling=True,
            invalidate_cache=False,
        )
        ann = diagnostic_call(
            "core.captureAnn",
            store.build_ann_index,
            reduction,
            ann_metric="l2",
            ann_parallel=False,
            rand_state=4444,
            invalidate_cache=False,
        )
        neighbors = diagnostic_call(
            "core.captureNeighbors",
            store.query_neighbors,
            ann,
            coordinates=reduction,
            k=neighbors_k,
            invalidate_cache=False,
        )
        graph = diagnostic_call(
            "core.captureGraph",
            store.build_connectivity_map,
            neighbors,
            local_connectivity=1.0,
            bandwidth=1.5,
            invalidate_cache=False,
        )
        clusters = diagnostic_call(
            "core.capturePartition",
            store.run_leiden_clustering,
            graph,
            resolution=selected.parameters.leidenResolution,
            backend="igraph",
            symmetric_graph=False,
            graph_upper_only=False,
            random_seed=4444,
            invalidate_cache=False,
        )
        scores.append(
            diagnostic_call(
                "core.doubletDetection",
                store.run_doublet_detection,
                clusters,
                graph,
                from_assay=assay,
                invalidate_cache=False,
            )
        )
        selections.append(capture_selection)
        scored_captures.append(capture_value)
    if not scores:
        raise ValueError("No physical capture had enough cells for doublet scoring")
    return _build_advisory_doublet_scores(
        store,
        scores=scores,
        cell_selections=selections,
        native_graph=native_graph,
        native_clusters=native_clusters,
        parent_selection=parent_selection,
        capture_values=scored_captures,
        capture_column=capture_column,
        limitations=limitations,
    )


def _doublet_concentration(
    store: Any,
    labels: np.ndarray,
    cell_selection: ArtifactRef,
    evidence: AdvisoryDoubletScores,
) -> float | None:
    parent_indices = read_stored_selection_indices(
        store.zw,
        cell_selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    ).astype(np.int64, copy=False)
    if labels.shape != parent_indices.shape:
        raise ValueError("Cluster labels do not align with advisory doublet evidence")
    if len(parent_indices) > 1 and np.any(parent_indices[1:] <= parent_indices[:-1]):
        raise ValueError("Doublet parent selection indices must be strictly increasing")
    high_score = np.zeros(len(parent_indices), dtype=bool)
    covered = np.zeros(len(parent_indices), dtype=bool)
    if evidence.score_summaries and len(evidence.score_summaries) != len(
        evidence.scores
    ):
        raise ValueError("Doublet score summaries do not align with score artifacts")
    for score_index, (score_ref, selection_ref) in enumerate(
        zip(
            evidence.scores,
            evidence.cell_selections,
            strict=True,
        )
    ):
        score_values = as_zarr_array(
            store.load_artifact(score_ref)["values"],
            name="values",
        )
        selection_indices = read_stored_selection_indices(
            store.zw,
            selection_ref,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ).astype(np.int64, copy=False)
        if score_values.shape != selection_indices.shape:
            raise ValueError("Doublet scores do not align with their cell selection")
        summary = (
            evidence.score_summaries[score_index]
            if evidence.score_summaries
            else diagnostic_call(
                "metric.doubletScoreSummary",
                _bounded_score_summary,
                score_values,
                maximum_sample_size=65_536,
            )[0]
        )
        threshold = float(summary["p90"])
        for start in range(0, len(selection_indices), 65_536):
            local_indices = selection_indices[start : start + 65_536]
            local_positions = np.searchsorted(parent_indices, local_indices)
            if np.any(local_positions >= len(parent_indices)) or not np.array_equal(
                parent_indices[local_positions],
                local_indices,
            ):
                raise ValueError(
                    "Doublet score selection is outside its parent selection"
                )
            if covered[local_positions].any():
                raise ValueError("Doublet score selections must not overlap")
            local_scores = np.asarray(
                score_values[start : start + len(local_indices)],
                dtype=np.float64,
            )
            if not np.isfinite(local_scores).all():
                raise ValueError("Doublet scores must be finite")
            covered[local_positions] = True
            high_score[local_positions] = local_scores >= threshold
    if not covered.any() or not high_score[covered].any():
        return None
    baseline = float(high_score[covered].mean())
    enrichments = [
        float(high_score[covered & (labels == cluster)].mean()) / baseline
        for cluster in np.unique(labels[covered])
        if bool((covered & (labels == cluster)).any())
    ]
    return max(enrichments, default=0.0)


def _aligned_metadata(
    store: Any,
    cell_selection: ArtifactRef,
    column: str,
) -> np.ndarray:
    indices = read_stored_selection_indices(
        store.zw,
        cell_selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    ).astype(np.int64, copy=False)
    values = np.asarray(read_metadata_rows_chunkwise(store.cells, column, indices))
    if values.shape != (len(indices),):
        raise ValueError(f"Metadata column {column!r} does not align with clusters")
    return values.astype(str)


def _cross_unit_support(labels: np.ndarray, units: np.ndarray) -> float | None:
    available_units = np.unique(units)
    if len(available_units) < 2:
        return None
    supported = [
        len(np.unique(units[labels == cluster])) >= 2 for cluster in np.unique(labels)
    ]
    return float(np.mean(supported)) if supported else None


def population_support_evidence(
    store: Any,
    evaluation: ParameterCandidateEvaluation,
    columns: Sequence[str],
) -> dict[str, Any]:
    """Describe observed population support using exact cells and requested units.

    This is descriptive support, not a validation of population identity or
    independent replication. Display limits do not change any count or fraction.
    """
    if evaluation.status != "done" or evaluation.cellSelection is None:
        raise ValueError("Population support requires completed, cell-bound evidence")
    selection = ArtifactRef(
        scope=evaluation.cellSelection.scope,
        assay=evaluation.cellSelection.assay,
        kind=evaluation.cellSelection.kind,
        artifact_id=evaluation.cellSelection.artifactId,
    )
    clusters = _artifact_ref(evaluation, "clusters")
    status = store.inspect_artifact(clusters)
    if not status.exists or not status.complete:
        raise ValueError("Population support requires complete cluster evidence")
    raw_selection = (status.inputs or {}).get("cell_selection")
    if (
        not isinstance(raw_selection, Mapping)
        or ArtifactRef.from_dict(dict(raw_selection)) != selection
    ):
        raise ValueError("Population support cluster and candidate cells differ")
    if clusters.kind not in {"cluster_labels", "cluster_cut"}:
        raise ValueError("Population support requires a clustering artifact")
    indices = read_stored_selection_indices(
        store.zw,
        selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    if not len(indices) or np.any(indices[1:] <= indices[:-1]):
        raise ValueError("Population support requires distinct ordered selected cells")
    labels = as_zarr_array(
        store.load_artifact(clusters)[
            "values" if clusters.kind == "cluster_labels" else "labels"
        ],
        name="cluster labels",
    )
    if labels.shape != indices.shape or labels.dtype.kind not in "iuf":
        raise ValueError("Population labels do not align with the selected cells")
    requested = list(dict.fromkeys(columns))
    selected_columns = requested[:2]
    available = [column for column in selected_columns if column in store.cells.columns]
    totals: Counter[int] = Counter()
    group_totals: dict[str, Counter[tuple[str, Any]]] = {
        column: Counter() for column in available
    }
    counts: dict[str, dict[int, Counter[tuple[str, Any]]]] = {
        column: defaultdict(Counter) for column in available
    }
    for start in range(0, len(indices), 65_536):
        stop = min(start + 65_536, len(indices))
        block = np.asarray(labels[start:stop])
        if not np.isfinite(block).all() or not np.equal(block, np.floor(block)).all():
            raise ValueError("Population labels must be finite integers")
        population_ids = block.astype(np.int64)
        unique, sizes = np.unique(population_ids, return_counts=True)
        totals.update(
            {
                int(population): int(size)
                for population, size in zip(unique, sizes, strict=True)
            }
        )
        rows = indices[start:stop]
        for column in available:
            values = np.asarray(read_metadata_rows_chunkwise(store.cells, column, rows))
            missing = read_metadata_missing_rows_chunkwise(store.cells, column, rows)
            if (
                values.shape != rows.shape
                or missing is not None
                and missing.shape != rows.shape
            ):
                raise ValueError(
                    f"Population metadata {column!r} does not align with cells"
                )
            for offset, (population, value) in enumerate(
                zip(population_ids, values, strict=True)
            ):
                value = value.item() if isinstance(value, np.generic) else value
                if (
                    missing is not None
                    and missing[offset]
                    or pd.isna(value)
                    or isinstance(value, float)
                    and not np.isfinite(value)
                ):
                    continue
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
                if not isinstance(value, (str, int, float, bool)):
                    raise ValueError("Population unit labels must be scalar values")
                if isinstance(value, str) and not value.strip():
                    continue
                key = (type(value).__name__, value)
                group_totals[column][key] += 1
                counts[column][int(population)][key] += 1
    evidence_columns: dict[str, Any] = {}
    for column in selected_columns:
        if column not in available:
            evidence_columns[column] = {
                "status": "unavailable",
                "reason": "Column is absent.",
            }
            continue
        population_rows: list[dict[str, Any]] = []
        for population, size in totals.items():
            group_counts = counts[column][population]
            ordered = sorted(
                group_counts.items(),
                key=lambda item: (-item[1], item[0][0], str(item[0][1])),
            )
            displayed = ordered[:5]
            covered = sum(group_counts.values())
            population_rows.append(
                {
                    "cluster": str(population),
                    "cells": size,
                    "coveredCells": covered,
                    "missingCells": size - covered,
                    "coverageFraction": covered / size,
                    "supportingGroups": len(group_counts),
                    "groupsWithAtLeast5Cells": sum(
                        count >= 5 for count in group_counts.values()
                    ),
                    "largestGroupFraction": ordered[0][1] / size if ordered else None,
                    "topGroups": [
                        {
                            "value": key[1],
                            "valueType": key[0],
                            "cells": count,
                            "fractionOfPopulation": count / size,
                            "fractionOfGroup": count / group_totals[column][key],
                        }
                        for key, count in displayed
                    ],
                    "omittedGroups": len(ordered) - len(displayed),
                    "omittedCells": sum(count for _, count in ordered[5:]),
                }
            )
        population_rows.sort(
            key=lambda row: (
                -(row["largestGroupFraction"] or 0),
                row["cells"],
                row["cluster"],
            )
        )
        covered = sum(group_totals[column].values())
        evidence_columns[column] = {
            "status": "computed",
            "observedGroups": len(group_totals[column]),
            "coveredCells": covered,
            "missingCells": len(indices) - covered,
            "coverageFraction": covered / len(indices),
            "populations": population_rows[:64],
            "omittedPopulations": max(0, len(population_rows) - 64),
            "omittedPopulationCells": sum(row["cells"] for row in population_rows[64:]),
        }
    return {
        "candidateId": evaluation.candidateId,
        "cellSelection": selection.to_dict(),
        "clusters": clusters.to_dict(),
        "selectedCells": len(indices),
        "observedPopulations": len(totals),
        "columns": evidence_columns,
        "omittedColumns": requested[2:],
        "displayLimits": {
            "columns": 2,
            "populationsPerColumn": 64,
            "groupsPerPopulation": 5,
        },
        "interpretation": (
            "Counts use all exact selected cells. Population fractions include cells with missing unit metadata; "
            "group fractions use all selected cells with that unit value. Missing values are unassigned. "
            "Displayed populations prioritize concentration in one group, then smaller size. "
            "A group with at least five cells is a descriptive count, not a replication threshold. "
            "Shared donor or capture support does not establish biological identity or rule out artifacts."
        ),
    }


def _subsample_partition_stability(
    graph: Any,
    labels: np.ndarray,
    resolution: float,
) -> float:
    if graph.shape != (len(labels), len(labels)):
        raise ValueError("Candidate graph does not align with cluster labels")
    selected = np.arange(len(labels)) % 5 != 0
    if int(selected.sum()) < 3:
        selected = np.ones(len(labels), dtype=bool)
    subsample_labels = diagnostic_call(
        "core.subsamplePartition",
        leiden_membership,
        graph[selected][:, selected],
        resolution,
        4444,
        backend="igraph",
    )
    return float(
        diagnostic_call(
            "metric.partitionAgreement",
            adjusted_rand_score,
            labels[selected],
            subsample_labels,
        )
    )


def augment_cluster_evaluations(
    store: Any,
    evaluations: Sequence[ParameterCandidateEvaluation],
    *,
    marker_assay: str,
    marker_features: ArtifactRef,
    independent_unit_columns: Sequence[str],
    technical_columns: Sequence[str],
    nominated_families: Sequence[str] = (),
    protected_families: Sequence[str] = (),
    doublet_evidence: AdvisoryDoubletScores | None = None,
) -> tuple[ParameterCandidateEvaluation, ...]:
    """Add seed, marker, unit-support, and technical-association evidence."""
    _marker_indices, marker_feature_names = _selected_feature_names(
        store,
        marker_features,
    )
    family_masks = {
        family: mask
        for family in dict.fromkeys([*nominated_families, *protected_families])
        for mask in [_family_mask(marker_feature_names, family)]
        if mask is not None and bool(mask.any())
    }
    augmented: list[ParameterCandidateEvaluation] = []
    for evaluation in evaluations:
        if evaluation.status != "done" or evaluation.cellSelection is None:
            augmented.append(evaluation)
            continue
        graph_ref = _artifact_ref(evaluation, "connectivityMap")
        clusters_ref = _artifact_ref(evaluation, "clusters")
        labels = _cluster_labels(store, clusters_ref)
        alternative_ref = diagnostic_call(
            "core.alternateSeedPartition",
            store.run_leiden_clustering,
            graph_ref,
            resolution=evaluation.parameters.leidenResolution,
            backend="igraph",
            symmetric_graph=False,
            graph_upper_only=False,
            random_seed=9173,
            invalidate_cache=False,
        )
        alternative = _cluster_labels(store, alternative_ref)
        if alternative.shape != labels.shape:
            raise ValueError("Alternate-seed clusters do not align with the candidate")
        seed_stability = float(
            diagnostic_call(
                "metric.partitionAgreement", adjusted_rand_score, labels, alternative
            )
        )
        subsample_stability = _cached_candidate_metric(
            (
                id(store),
                "subsample_stability",
                graph_ref,
                clusters_ref,
                evaluation.parameters.leidenResolution,
            ),
            lambda: _subsample_partition_stability(
                store.load_graph(graph_ref),
                labels,
                evaluation.parameters.leidenResolution,
            ),
        )

        cluster_count = len(np.unique(labels))
        marker_ref = None
        markers = pd.DataFrame()
        if cluster_count >= 2:
            marker_ref = diagnostic_call(
                "core.markers",
                store.run_marker_search,
                clusters_ref,
                from_assay=marker_assay,
                features=marker_features,
                invalidate_cache=False,
            )
            markers = store.get_markers(
                marker_ref,
                min_score=0.25,
                min_frac_exp=0.2,
            )
        marker_groups = (
            set(markers["group_id"].astype(str))
            if "group_id" in markers.columns
            else set()
        )
        marker_coherence = (
            float(len(marker_groups) / cluster_count)
            if marker_ref is not None
            else None
        )
        marker_names = (
            markers["feature_name"].astype(str).to_numpy()
            if "feature_name" in markers.columns
            else np.asarray([], dtype=str)
        )
        marker_specificity: dict[str, float] = {}
        marker_auc: dict[str, float] = {}
        top_marker_genes: dict[str, list[str]] = {}
        marker_group_values = (
            markers["group_id"].astype(str) if "group_id" in markers.columns else None
        )
        for cluster in np.unique(labels):
            cluster_id = str(cluster)
            cluster_markers = (
                markers.loc[marker_group_values == cluster_id]
                if marker_group_values is not None
                else markers.iloc[0:0]
            )
            if "score" in cluster_markers:
                cluster_markers = cluster_markers.sort_values(
                    "score",
                    ascending=False,
                    kind="stable",
                )
                top_scores = cluster_markers["score"].to_numpy(
                    dtype=np.float64,
                )[:10]
                top_scores = top_scores[np.isfinite(top_scores)]
                if len(top_scores):
                    marker_specificity[cluster_id] = float(np.median(top_scores))
            if "auc" in cluster_markers:
                top_auc = cluster_markers["auc"].to_numpy(dtype=np.float64)[:10]
                top_auc = top_auc[np.isfinite(top_auc)]
                if len(top_auc):
                    marker_auc[cluster_id] = float(np.median(top_auc))
            top_marker_genes[cluster_id] = (
                cluster_markers["feature_name"].astype(str).head(10).tolist()
                if "feature_name" in cluster_markers
                else []
            )
        marker_specificity_median = (
            float(np.median(list(marker_specificity.values())))
            if marker_specificity
            else None
        )
        marker_family_enrichment: dict[str, float] = {}
        protected_marker_families: list[str] = []
        for family, mask in family_masks.items():
            marker_mask = _family_mask(marker_names, family)
            marker_fraction = (
                float(marker_mask.mean())
                if marker_mask is not None and marker_mask.size
                else 0.0
            )
            background = float(mask.mean())
            enrichment = marker_fraction / background if background > 0 else 0.0
            if marker_ref is not None and family in nominated_families:
                marker_family_enrichment[family] = enrichment
            if family in protected_families and marker_fraction > 0:
                protected_marker_families.append(family)

        selection_ref = ArtifactRef(
            scope=evaluation.cellSelection.scope,
            assay=evaluation.cellSelection.assay,
            kind=evaluation.cellSelection.kind,
            artifact_id=evaluation.cellSelection.artifactId,
        )
        doublet_concentration = (
            diagnostic_call(
                "metric.doubletConcentration",
                _doublet_concentration,
                store,
                labels,
                selection_ref,
                doublet_evidence,
            )
            if doublet_evidence is not None and doublet_evidence.scores
            else None
        )
        unit_scores = [
            score
            for column in independent_unit_columns
            if column in store.cells.columns
            for score in [
                diagnostic_call(
                    "metric.crossUnitSupport",
                    _cross_unit_support,
                    labels,
                    _aligned_metadata(store, selection_ref, column),
                )
            ]
            if score is not None
        ]
        cross_unit_support = min(unit_scores) if unit_scores else None
        technical_association = {
            column: float(
                diagnostic_call(
                    "metric.technicalAssociation",
                    normalized_mutual_info_score,
                    labels,
                    _aligned_metadata(store, selection_ref, column),
                )
            )
            for column in technical_columns
            if column in store.cells.columns
        }

        metrics = evaluation.metrics.model_copy(
            update={
                "seedStability": seed_stability,
                "subsampleStability": subsample_stability,
                "markerCoherence": marker_coherence,
                "markerSpecificityMedian": marker_specificity_median,
                "markerSpecificityByCluster": marker_specificity,
                "markerAucByCluster": marker_auc,
                "topMarkerGenes": top_marker_genes,
                "crossUnitSupport": cross_unit_support,
                "technicalAssociation": technical_association,
                "markerFamilyEnrichment": marker_family_enrichment,
                "protectedMarkerFamilies": protected_marker_families,
                "doubletHighScoreConcentration": doublet_concentration,
                "doubletScoreQuantiles": (
                    dict(doublet_evidence.score_quantiles)
                    if doublet_evidence is not None
                    else {}
                ),
                "doubletScoreByCapture": (
                    {
                        capture: dict(summary)
                        for capture, summary in zip(
                            doublet_evidence.capture_values,
                            doublet_evidence.score_summaries,
                            strict=True,
                        )
                    }
                    if doublet_evidence is not None
                    else {}
                ),
                "doubletCaptureCoverage": (
                    doublet_evidence.capture_coverage
                    if doublet_evidence is not None
                    else None
                ),
            }
        )
        evidence_ids = [
            *evaluation.evidenceIds,
            f"candidate:{evaluation.candidateId}:seedStability",
            f"candidate:{evaluation.candidateId}:subsampleStability",
            *(
                [
                    f"candidate:{evaluation.candidateId}:markerCoherence",
                    f"candidate:{evaluation.candidateId}:markerSpecificity",
                    f"candidate:{evaluation.candidateId}:markerFamilies",
                ]
                if marker_ref is not None
                else []
            ),
            *(
                [f"candidate:{evaluation.candidateId}:crossUnitSupport"]
                if cross_unit_support is not None
                else []
            ),
            *[
                f"candidate:{evaluation.candidateId}:technicalAssociation:{column}"
                for column in technical_association
            ],
            *(
                [f"candidate:{evaluation.candidateId}:doubletConcentration"]
                if doublet_concentration is not None
                else []
            ),
            *(
                [
                    f"candidate:{evaluation.candidateId}:doubletScoreTails",
                    f"candidate:{evaluation.candidateId}:doubletCaptureCoverage",
                ]
                if doublet_evidence is not None and doublet_evidence.scores
                else []
            ),
        ]
        artifacts = {
            **evaluation.artifacts,
            "stabilityClusters": ArtifactRecord.from_ref(alternative_ref),
            **(
                {"markerTable": ArtifactRecord.from_ref(marker_ref)}
                if marker_ref is not None
                else {}
            ),
            **(
                {
                    f"doubletScore:{index}": ArtifactRecord.from_ref(score)
                    for index, score in enumerate(doublet_evidence.scores)
                }
                if doublet_evidence is not None
                else {}
            ),
            **(
                {
                    f"doubletCellSelection:{index}": ArtifactRecord.from_ref(selection)
                    for index, selection in enumerate(doublet_evidence.cell_selections)
                }
                if doublet_evidence is not None
                else {}
            ),
            **(
                {
                    "doubletNativeGraph": ArtifactRecord.from_ref(
                        doublet_evidence.native_graph
                    ),
                    "doubletNativeClusters": ArtifactRecord.from_ref(
                        doublet_evidence.native_clusters
                    ),
                }
                if doublet_evidence is not None
                else {}
            ),
        }
        augmented.append(
            evaluation.model_copy(
                update={
                    "metrics": metrics,
                    "eligible": evaluation.eligible and cluster_count >= 2,
                    "eligibilityReasons": list(
                        dict.fromkeys(
                            [
                                *evaluation.eligibilityReasons,
                                *(
                                    [
                                        "Marker contrasts require at least two populated clusters"
                                    ]
                                    if cluster_count < 2
                                    else []
                                ),
                            ]
                        )
                    ),
                    "evidenceIds": list(dict.fromkeys(evidence_ids)),
                    "artifacts": artifacts,
                    "warnings": list(
                        dict.fromkeys(
                            [
                                *evaluation.warnings,
                                *(
                                    doublet_evidence.limitations
                                    if doublet_evidence is not None
                                    else ()
                                ),
                            ]
                        )
                    ),
                }
            )
        )
    return annotate_candidate_dominance(augmented)


__all__ = [
    "AdvisoryDoubletScores",
    "augment_cluster_evaluations",
    "augment_pca_evaluations",
    "population_support_evidence",
    "resolve_native_doublet_inputs",
    "score_advisory_doublets",
]

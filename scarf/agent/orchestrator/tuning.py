"""Sequential RNA parameter tuning and review stages."""

import io
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np

from ...datastore.datastore import DataStore
from ...metadata.rows import read_metadata_rows_chunkwise
from ...storage.refs import ArtifactRef
from ...storage.selections import read_stored_selection_indices
from ...storage.types import as_zarr_array
from ..config.agent_exec import (
    ImageEvidence,
)
from ..experimental_context.contracts import ExperimentalContextResult
from ..experimental_context.study import StudyContract
from ..parameter_tuning.contracts import (
    ParameterCandidateEvaluation,
    ParameterTuningReport,
)
from ..parameter_tuning.execution import (
    _metadata_column_fingerprint,
    candidate_metric_cache,
)
from ..types import ArtifactReferenceModel
from . import journal
from .decisions import DecisionStagesMixin
from .models import (
    AutomatedPreprocessingPlan,
    OrchestrationRequestRecord,
    OrchestrationResumeRecord,
    PreprocessedAssayHandoff,
    StageEvidenceReference,
    WorkflowIdentity,
    WorkflowNeedsInput,
    WorkflowQuestion,
    WorkflowStageAttempt,
    WorkflowStageLink,
    WorkflowStageName,
    artifact_model_to_ref,
)


def _analysis_visual_content(
    store: DataStore,
    selected: ParameterCandidateEvaluation,
    candidates: Sequence[ParameterCandidateEvaluation],
    *,
    qc_columns: Sequence[str] = (),
    qc_artifact_metrics: Sequence[tuple[str, ArtifactReferenceModel]] = (),
) -> list[ImageEvidence]:
    """Render one bounded diagnostic board for multimodal review."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Visual adjudication requires the installed plotting dependencies"
        ) from exc

    def image_content(figure: Any, identifier: str) -> ImageEvidence:
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=120)
        plt.close(figure)
        return ImageEvidence(
            identifier=identifier,
            data=buffer.getvalue(),
            media_type="image/png",
        )

    def sampled_values(array: Any, selection: tuple[Any, ...]) -> np.ndarray:
        return np.asarray(
            cast(Any, as_zarr_array(array, name="diagnostic")).get_orthogonal_selection(
                selection
            )
        )

    coordinate_name = (
        "harmony"
        if selected.parameters.useHarmony
        else selected.parameters.reductionMethod
    )
    coordinate_record = selected.artifacts.get(coordinate_name)
    cluster_record = selected.artifacts.get("clusters")
    if coordinate_record is None or cluster_record is None:
        raise ValueError(
            "Selected candidate lacks visualizable coordinates or clusters"
        )
    coordinate_group = store.load_artifact(
        artifact_model_to_ref(
            ArtifactReferenceModel.model_validate(coordinate_record.model_dump())
        )
    )
    cluster_group = store.load_artifact(
        artifact_model_to_ref(
            ArtifactReferenceModel.model_validate(cluster_record.model_dump())
        )
    )
    coordinate_values = as_zarr_array(coordinate_group["data"], name="data")
    cluster_values = as_zarr_array(cluster_group["values"], name="values")
    if len(coordinate_values.shape) != 2 or coordinate_values.shape[1] < 2:
        raise ValueError("Selected coordinates need at least two dimensions")
    if coordinate_values.shape[0] != cluster_values.shape[0]:
        raise ValueError("Selected coordinates and clusters do not align")
    sample_count = min(5_000, coordinate_values.shape[0])
    sample_indices = np.linspace(
        0,
        coordinate_values.shape[0] - 1,
        sample_count,
        dtype=np.int64,
    )
    coordinates = sampled_values(
        coordinate_values,
        (sample_indices, slice(0, 2)),
    ).astype(np.float32, copy=False)
    labels = sampled_values(cluster_values, (sample_indices,))
    _, label_codes = np.unique(labels, return_inverse=True)

    comparison = max(
        (
            candidate
            for candidate in candidates
            if candidate.candidateId != selected.candidateId
            and candidate.status == "done"
            and candidate.eligible
            and candidate.parameters.dimensions == selected.parameters.dimensions
            and candidate.parameters.neighborsK == selected.parameters.neighborsK
            and candidate.parameters.useHarmony == selected.parameters.useHarmony
            and candidate.artifacts.get("graphFeatures")
            == selected.artifacts.get("graphFeatures")
            and candidate.cellSelection == selected.cellSelection
        ),
        key=lambda value: (
            value.metrics.markerCoherence or 0.0,
            value.metrics.seedStability or 0.0,
        ),
        default=None,
    )

    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    variance = np.asarray(selected.metrics.componentVariance, dtype=float)
    if variance.size:
        axes[0, 0].plot(np.arange(1, variance.size + 1), variance, marker=".")
        axes[0, 0].set_title("PCA explained variance")
        axes[0, 0].set_xlabel("Component")
    else:
        axes[0, 0].text(0.5, 0.5, "Variance unavailable", ha="center")
        axes[0, 0].set_axis_off()

    axes[0, 1].scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=label_codes,
        cmap="tab20",
        s=2,
        alpha=0.65,
        rasterized=True,
    )
    axes[0, 1].set_title(f"Selected partition: {selected.candidateId}")

    if comparison is not None and "clusters" in comparison.artifacts:
        comparison_group = store.load_artifact(
            artifact_model_to_ref(
                ArtifactReferenceModel.model_validate(
                    comparison.artifacts["clusters"].model_dump()
                )
            )
        )
        comparison_labels = sampled_values(
            comparison_group["values"],
            (sample_indices,),
        )
        _, comparison_codes = np.unique(
            comparison_labels,
            return_inverse=True,
        )
        axes[0, 2].scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=comparison_codes,
            cmap="tab20",
            s=2,
            alpha=0.65,
            rasterized=True,
        )
        axes[0, 2].set_title(f"Matched alternative: {comparison.candidateId}")
    else:
        axes[0, 2].text(0.5, 0.5, "No matched alternative", ha="center")
        axes[0, 2].set_axis_off()

    completed = [
        candidate
        for candidate in candidates
        if candidate.status == "done"
        and candidate.eligible
        and candidate.parameters.dimensions == selected.parameters.dimensions
        and candidate.parameters.neighborsK == selected.parameters.neighborsK
        and candidate.parameters.useHarmony == selected.parameters.useHarmony
    ]
    resolutions = [value.parameters.leidenResolution for value in completed]
    axes[1, 0].plot(
        resolutions,
        [value.metrics.markerCoherence or 0.0 for value in completed],
        marker="o",
        label="marker coherence",
    )
    axes[1, 0].plot(
        resolutions,
        [value.metrics.seedStability or 0.0 for value in completed],
        marker="o",
        label="seed stability",
    )
    axes[1, 0].plot(
        resolutions,
        [value.metrics.crossUnitSupport or 0.0 for value in completed],
        marker="o",
        label="cross-unit support",
    )
    axes[1, 0].set_title("Partition evidence")
    axes[1, 0].set_xlabel("Leiden resolution")
    axes[1, 0].legend(fontsize=8)

    doublet_records = {
        name.removeprefix("doubletScore:"): value
        for name, value in selected.artifacts.items()
        if name.startswith("doubletScore:")
    }
    doublet_selections = {
        name.removeprefix("doubletCellSelection:"): value
        for name, value in selected.artifacts.items()
        if name.startswith("doubletCellSelection:")
    }
    doublet_sample = np.full(sample_count, np.nan, dtype=np.float64)
    doublet_hist_values: list[np.ndarray] = []
    if doublet_records:
        if selected.cellSelection is None:
            raise ValueError("Doublet visuals require an exact cell selection")
        parent_selection = artifact_model_to_ref(selected.cellSelection)
        parent_indices = read_stored_selection_indices(
            store.zw,
            parent_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ).astype(np.int64, copy=False)
        sampled_global_indices = parent_indices[sample_indices]
        for score_index, record in sorted(doublet_records.items()):
            score_group = store.load_artifact(
                artifact_model_to_ref(
                    ArtifactReferenceModel.model_validate(record.model_dump())
                )
            )
            score_values = as_zarr_array(score_group["values"], name="values")
            selection_record = doublet_selections.get(score_index)
            if selection_record is None:
                if score_values.shape != parent_indices.shape:
                    raise ValueError(
                        "Doublet score artifact lacks its exact cell selection"
                    )
                local_positions = sample_indices
                matched_positions = np.arange(sample_count, dtype=np.int64)
            else:
                score_selection = artifact_model_to_ref(
                    ArtifactReferenceModel.model_validate(selection_record.model_dump())
                )
                score_indices = read_stored_selection_indices(
                    store.zw,
                    score_selection,
                    kind="cell_selection",
                    scope="datastore",
                    assay=None,
                    table_path="cellData",
                ).astype(np.int64, copy=False)
                if score_values.shape != score_indices.shape:
                    raise ValueError(
                        "Doublet scores do not align with their cell selection"
                    )
                candidate_positions = np.searchsorted(
                    score_indices,
                    sampled_global_indices,
                )
                within = candidate_positions < len(score_indices)
                matched = np.zeros(sample_count, dtype=bool)
                matched[within] = (
                    score_indices[candidate_positions[within]]
                    == sampled_global_indices[within]
                )
                matched_positions = np.flatnonzero(matched)
                local_positions = candidate_positions[matched]
            sampled_scores = sampled_values(score_values, (local_positions,)).astype(
                np.float64,
                copy=False,
            )
            doublet_sample[matched_positions] = sampled_scores
            doublet_hist_values.append(sampled_scores)
        finite_doublets = np.concatenate(doublet_hist_values)
        finite_doublets = finite_doublets[np.isfinite(finite_doublets)]
        axes[1, 1].hist(finite_doublets, bins=40)
        axes[1, 1].set_title("Advisory doublet scores")
    else:
        axes[1, 1].text(0.5, 0.5, "Doublet scores unavailable", ha="center")
        axes[1, 1].set_axis_off()

    family_values = {
        **selected.metrics.loadingFamilyEnrichment,
        **selected.metrics.markerFamilyEnrichment,
    }
    if family_values:
        ordered = sorted(
            family_values.items(),
            key=lambda item: (-item[1], item[0]),
        )[:12]
        axes[1, 2].barh(
            [name for name, _value in reversed(ordered)],
            [value for _name, value in reversed(ordered)],
        )
        axes[1, 2].set_title("Feature-family enrichment")
    else:
        axes[1, 2].text(0.5, 0.5, "Family evidence unavailable", ha="center")
        axes[1, 2].set_axis_off()

    content = [image_content(figure, "analysis-overview")]

    paired_corrections: list[
        tuple[ParameterCandidateEvaluation, ParameterCandidateEvaluation]
    ] = []
    candidates_by_parameters: dict[
        tuple[str, int, int, float, str, str],
        dict[bool, ParameterCandidateEvaluation],
    ] = {}
    for candidate in candidates:
        if candidate.status != "done" or not candidate.eligible:
            continue
        key = (
            candidate.parameters.reductionMethod,
            candidate.parameters.dimensions,
            candidate.parameters.neighborsK,
            candidate.parameters.leidenResolution,
            candidate.artifacts["graphFeatures"].model_dump_json()
            if "graphFeatures" in candidate.artifacts
            else "",
            candidate.cellSelection.model_dump_json()
            if candidate.cellSelection is not None
            else "",
        )
        candidates_by_parameters.setdefault(key, {})[
            candidate.parameters.useHarmony
        ] = candidate
    for correction_pair in candidates_by_parameters.values():
        if False in correction_pair and True in correction_pair:
            paired_corrections.append((correction_pair[False], correction_pair[True]))
    if paired_corrections:
        native, harmony = min(
            paired_corrections,
            key=lambda pair: (
                abs(pair[0].parameters.dimensions - selected.parameters.dimensions),
                abs(pair[0].parameters.neighborsK - selected.parameters.neighborsK),
                abs(
                    pair[0].parameters.leidenResolution
                    - selected.parameters.leidenResolution
                ),
            ),
        )
        correction_figure, correction_axes = plt.subplots(
            2,
            2,
            figsize=(11, 10),
            constrained_layout=True,
        )
        batch_columns = [
            column
            for column in (
                *native.metrics.batchMixing,
                *harmony.metrics.batchMixing,
            )
            if column in store.cells.columns
        ]
        batch_codes: np.ndarray | None = None
        batch_label = "Batch unavailable"
        if batch_columns and selected.cellSelection is not None:
            parent_selection = artifact_model_to_ref(selected.cellSelection)
            parent_indices = read_stored_selection_indices(
                store.zw,
                parent_selection,
                kind="cell_selection",
                scope="datastore",
                assay=None,
                table_path="cellData",
            ).astype(np.int64, copy=False)
            batch_label = batch_columns[0]
            batch_values = read_metadata_rows_chunkwise(
                store.cells,
                batch_label,
                parent_indices[sample_indices],
            ).astype(str)
            _, batch_codes = np.unique(batch_values, return_inverse=True)
        for column_index, candidate in enumerate((native, harmony)):
            name = (
                "harmony"
                if candidate.parameters.useHarmony
                else candidate.parameters.reductionMethod
            )
            candidate_coordinate = candidate.artifacts.get(name)
            candidate_cluster = candidate.artifacts.get("clusters")
            if candidate_coordinate is None or candidate_cluster is None:
                raise ValueError("Matched correction candidate lacks visual artifacts")
            candidate_coordinate_group = store.load_artifact(
                artifact_model_to_ref(
                    ArtifactReferenceModel.model_validate(
                        candidate_coordinate.model_dump()
                    )
                )
            )
            candidate_cluster_group = store.load_artifact(
                artifact_model_to_ref(
                    ArtifactReferenceModel.model_validate(
                        candidate_cluster.model_dump()
                    )
                )
            )
            candidate_coordinates = sampled_values(
                candidate_coordinate_group["data"],
                (sample_indices, slice(0, 2)),
            )
            candidate_labels = sampled_values(
                candidate_cluster_group["values"],
                (sample_indices,),
            )
            _, candidate_codes = np.unique(
                candidate_labels,
                return_inverse=True,
            )
            title = "Harmony" if candidate.parameters.useHarmony else "Native PCA"
            correction_axes[0, column_index].scatter(
                candidate_coordinates[:, 0],
                candidate_coordinates[:, 1],
                c=candidate_codes,
                cmap="tab20",
                s=2,
                alpha=0.65,
                rasterized=True,
            )
            correction_axes[0, column_index].set_title(f"{title}, partition colors")
            if batch_codes is not None:
                correction_axes[1, column_index].scatter(
                    candidate_coordinates[:, 0],
                    candidate_coordinates[:, 1],
                    c=batch_codes,
                    cmap="tab20",
                    s=2,
                    alpha=0.65,
                    rasterized=True,
                )
                correction_axes[1, column_index].set_title(
                    f"{title}, {batch_label} colors"
                )
            else:
                correction_axes[1, column_index].text(
                    0.5,
                    0.5,
                    batch_label,
                    ha="center",
                )
                correction_axes[1, column_index].set_axis_off()
        correction_figure.suptitle(
            "Parameter-matched native and Harmony representations"
        )
        content.append(image_content(correction_figure, "native-harmony-comparison"))

    marker_record = selected.artifacts.get("markerTable")
    marker_genes = list(
        dict.fromkeys(
            gene for genes in selected.metrics.topMarkerGenes.values() for gene in genes
        )
    )[:24]
    if marker_record is not None and marker_genes:
        marker_ref = artifact_model_to_ref(
            ArtifactReferenceModel.model_validate(marker_record.model_dump())
        )
        marker_table = store.get_markers(
            marker_ref,
            min_score=0.25,
            min_frac_exp=0.2,
        )
        required_marker_columns = {"group_id", "feature_name", "score"}
        if required_marker_columns.issubset(marker_table.columns):
            marker_groups = sorted(
                selected.metrics.topMarkerGenes,
                key=lambda value: (
                    (0, int(value)) if value.lstrip("-").isdigit() else (1, value)
                ),
            )
            marker_scores = np.zeros(
                (len(marker_groups), len(marker_genes)),
                dtype=np.float64,
            )
            group_positions = {
                value: index for index, value in enumerate(marker_groups)
            }
            gene_positions = {value: index for index, value in enumerate(marker_genes)}
            for row in marker_table.itertuples(index=False):
                group = str(getattr(row, "group_id"))
                gene = str(getattr(row, "feature_name"))
                if group not in group_positions or gene not in gene_positions:
                    continue
                score = float(getattr(row, "score"))
                if np.isfinite(score):
                    marker_scores[group_positions[group], gene_positions[gene]] = max(
                        marker_scores[
                            group_positions[group],
                            gene_positions[gene],
                        ],
                        score,
                    )

            def tagged_gene(gene: str) -> str:
                upper = gene.upper()
                if upper.startswith("MT-"):
                    return f"{gene} [mitochondrial]"
                if upper.startswith(("MRPS", "MRPL")):
                    return f"{gene} [mitoribosomal]"
                if upper.startswith(("RPS", "RPL")):
                    return f"{gene} [ribosomal]"
                if upper.startswith("CCN"):
                    return f"{gene} [CCN]"
                if upper.startswith("HLA-"):
                    return f"{gene} [HLA]"
                if upper.startswith("H2-"):
                    return f"{gene} [H2]"
                if upper.startswith("HIST"):
                    return f"{gene} [histone]"
                if upper in {
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
                }:
                    return f"{gene} [sex-linked]"
                return gene

            marker_figure, marker_axis = plt.subplots(
                figsize=(max(9, len(marker_genes) * 0.45), 6),
                constrained_layout=True,
            )
            marker_image = marker_axis.imshow(
                marker_scores,
                aspect="auto",
                interpolation="nearest",
                cmap="viridis",
            )
            marker_axis.set_xticks(
                np.arange(len(marker_genes)),
                [tagged_gene(gene) for gene in marker_genes],
                rotation=75,
                ha="right",
                fontsize=8,
            )
            marker_axis.set_yticks(
                np.arange(len(marker_groups)),
                marker_groups,
            )
            marker_axis.set_xlabel("Top marker feature")
            marker_axis.set_ylabel("Cluster")
            marker_axis.set_title("Feature-level marker score heatmap")
            marker_figure.colorbar(marker_image, ax=marker_axis, label="marker score")
            content.append(image_content(marker_figure, "marker-score-heatmap"))

    available_qc_columns = [
        column
        for column in dict.fromkeys([*qc_columns, *selected.metrics.qcPcaAssociation])
        if column in store.cells.columns
    ]
    qc_sources: list[tuple[str, ArtifactReferenceModel | None]] = [
        (column, None) for column in available_qc_columns
    ]
    known_qc_names = set(available_qc_columns)
    for qc_name, reference in qc_artifact_metrics:
        if qc_name not in known_qc_names:
            qc_sources.append((qc_name, reference))
            known_qc_names.add(qc_name)
    qc_sources = qc_sources[:4]
    if qc_sources or np.isfinite(doublet_sample).any():
        diagnostic_count = len(qc_sources) + 1
        qc_figure, qc_axes = plt.subplots(
            1,
            diagnostic_count,
            figsize=(max(5, diagnostic_count * 3.2), 4),
            constrained_layout=True,
        )
        qc_axis_list = np.atleast_1d(qc_axes).tolist()
        if selected.cellSelection is None:
            raise ValueError("QC visuals require an exact cell selection")
        parent_selection = artifact_model_to_ref(selected.cellSelection)
        parent_indices = read_stored_selection_indices(
            store.zw,
            parent_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ).astype(np.int64, copy=False)
        sampled_parent_indices = parent_indices[sample_indices]
        for axis, (qc_name, artifact) in zip(
            qc_axis_list,
            qc_sources,
            strict=False,
        ):
            if artifact is None:
                values = np.asarray(
                    read_metadata_rows_chunkwise(
                        store.cells,
                        qc_name,
                        sampled_parent_indices,
                    ),
                    dtype=np.float64,
                )
            else:
                artifact_ref = artifact_model_to_ref(artifact)
                artifact_group = store.load_artifact(artifact_ref)
                artifact_values = as_zarr_array(
                    artifact_group["values"],
                    name="values",
                )
                if artifact_values.shape == parent_indices.shape:
                    artifact_positions = sample_indices
                else:
                    artifact_status = store.inspect_artifact(artifact_ref)
                    raw_selection = (
                        getattr(artifact_status, "inputs", None) or {}
                    ).get("cell_selection")
                    if not isinstance(raw_selection, Mapping):
                        raise ValueError(
                            f"QC artifact {qc_name!r} lacks its cell selection"
                        )
                    artifact_selection = ArtifactRef.from_dict(dict(raw_selection))
                    artifact_indices = read_stored_selection_indices(
                        store.zw,
                        artifact_selection,
                        kind="cell_selection",
                        scope="datastore",
                        assay=None,
                        table_path="cellData",
                    ).astype(np.int64, copy=False)
                    candidate_positions = np.searchsorted(
                        artifact_indices,
                        sampled_parent_indices,
                    )
                    if np.any(
                        candidate_positions >= len(artifact_indices)
                    ) or not np.array_equal(
                        artifact_indices[candidate_positions],
                        sampled_parent_indices,
                    ):
                        raise ValueError(
                            f"QC artifact {qc_name!r} does not cover selected cells"
                        )
                    artifact_positions = candidate_positions
                values = sampled_values(
                    artifact_values,
                    (artifact_positions,),
                ).astype(np.float64, copy=False)
            finite_values = values[np.isfinite(values)]
            axis.violinplot(finite_values, showmedians=True)
            jitter = (
                (np.arange(len(finite_values), dtype=np.float64) % 23.0) - 11.0
            ) / 115.0
            axis.scatter(
                np.ones(len(finite_values)) + jitter,
                finite_values,
                s=1,
                alpha=0.15,
                rasterized=True,
            )
            axis.set_xticks([])
            axis.set_title(qc_name)
        doublet_axis = qc_axis_list[-1]
        finite_doublet_mask = np.isfinite(doublet_sample)
        if finite_doublet_mask.any():
            doublet_plot = doublet_axis.scatter(
                coordinates[finite_doublet_mask, 0],
                coordinates[finite_doublet_mask, 1],
                c=doublet_sample[finite_doublet_mask],
                cmap="magma",
                s=2,
                alpha=0.7,
                rasterized=True,
            )
            doublet_axis.set_title("Advisory doublet score")
            qc_figure.colorbar(
                doublet_plot,
                ax=doublet_axis,
                label="score",
            )
        else:
            doublet_axis.text(
                0.5,
                0.5,
                "Doublet embedding unavailable",
                ha="center",
            )
            doublet_axis.set_axis_off()
        qc_figure.suptitle("Selected-cell QC and non-removing doublet evidence")
        content.append(image_content(qc_figure, "qc-doublet-diagnostics"))
    return content


def _tuning_revision_provenances(
    store: DataStore,
    prefix: str,
    workflow_run_id: str,
    request: OrchestrationRequestRecord,
    experimental_reference: StageEvidenceReference,
    inputs: dict[str, Any],
    previous_starts: Sequence[WorkflowStageAttempt],
) -> list[dict[str, Any]]:
    """Authorize changed interpretation only along committed context revisions."""
    previous_inputs = [
        {
            key: value
            for key, value in previous.inputs.items()
            if key not in {"resumeAnswers", "answeredAttempt"}
        }
        for previous in previous_starts
    ]
    changed = [value for value in previous_inputs if value != inputs]
    if not changed:
        return []
    contexts = journal._stage_outcomes(
        store.zw, prefix, workflow_run_id, "experimental_context"
    )
    current = next(
        (row for row in contexts if experimental_reference in row.reportReferences),
        None,
    )
    if (
        current is None
        or current.status != "done"
        or current.outputs.get("studyContract") != inputs["studyContract"]
        or current.requestSha256 != request.requestSha256
        or current.configSha256 != request.configSha256
    ):
        raise ValueError(
            "Tuning inputs changed; changed evidence requires its exact completed context revision"
        )
    ancestors: list[dict[str, Any]] = []
    visited: set[str] = set()
    while current is not None and "reassessContextReport" in current.inputs:
        if current.attemptId in visited:
            raise ValueError("Context revision ancestry contains a cycle")
        visited.add(current.attemptId)
        previous_ref = current.inputs["reassessContextReport"]
        current = next(
            (
                row
                for row in contexts
                if any(
                    ref.model_dump(mode="json") == previous_ref
                    for ref in row.reportReferences
                )
            ),
            None,
        )
        if current is None or current.status != "done":
            raise ValueError(
                "Context revision is missing its committed previous evidence"
            )
        if (
            current.requestSha256 != request.requestSha256
            or current.configSha256 != request.configSha256
        ):
            raise ValueError(
                "Context revision belongs to different request or configuration inputs"
            )
        ancestors.append(current.outputs["studyContract"])
    authorized: list[dict[str, Any]] = []
    for previous in changed:
        if (
            previous.get("studyContract") not in ancestors
            or previous.get("preprocessedAssays") != inputs["preprocessedAssays"]
            or previous.get("featureMetadataFingerprints")
            != inputs["featureMetadataFingerprints"]
            or any(
                inputs["metadataFingerprints"].get(column) != fingerprint
                for column, fingerprint in previous.get(
                    "metadataFingerprints", {}
                ).items()
            )
            or set(previous) != set(inputs)
        ):
            raise ValueError(
                "Tuning inputs changed beyond an explicit context-evidence revision; "
                "restore the original cohort, feature inputs and metadata or start a new workflow"
            )
        provenance = {
            **previous,
            "requestSha256": request.requestSha256,
            "configSha256": request.configSha256,
        }
        if provenance not in authorized:
            authorized.append(provenance)
    return authorized


class TuningStagesMixin(DecisionStagesMixin):
    """Run bounded experiments and return validated full-cohort artifacts."""

    def parameter_tuning_stage(
        self,
        store: DataStore,
        workflow: WorkflowIdentity,
        request_record: OrchestrationRequestRecord,
        parents: Sequence[WorkflowStageLink],
        plan: AutomatedPreprocessingPlan,
        preprocessed: Sequence[PreprocessedAssayHandoff],
        experimental: ExperimentalContextResult,
        enrichment_reference: StageEvidenceReference,
        experimental_reference: StageEvidenceReference,
        answers: Mapping[str, Any],
        *,
        study_contract: StudyContract | None = None,
        resume_record: OrchestrationResumeRecord | None = None,
        stage_name: WorkflowStageName = "parameter_tuning",
    ) -> tuple[WorkflowStageAttempt, ParameterTuningReport]:
        from .rna_tuning import RnaTuningRun

        del enrichment_reference
        if len(preprocessed) != 1 or study_contract is None:
            raise ValueError("RNA tuning requires one assay and a study contract")
        handoff = preprocessed[0]
        if handoff.cellSelection is None:
            raise ValueError("RNA tuning requires a frozen full-cohort selection")
        columns = {
            *study_contract.technicalBatchColumns,
            *study_contract.protectedColumns,
            *study_contract.independentUnitColumns,
            *study_contract.conditionColumns,
            *plan.cellQc.attributes,
        }
        if study_contract.physicalCaptureColumn is not None:
            columns.add(study_contract.physicalCaptureColumn)
        metadata_fingerprints = {
            column: _metadata_column_fingerprint(store.cells, column)
            for column in sorted(columns)
            if column in store.cells.columns
        }
        feature_metadata = store.get_assay(plan.primaryAssay).feats
        feature_fingerprints = {
            column: _metadata_column_fingerprint(feature_metadata, column)
            for column in ("ids", "names")
        }
        inputs = {
            "preprocessedAssays": [handoff.model_dump(mode="json")],
            "studyContract": study_contract.model_dump(mode="json"),
            "metadataFingerprints": metadata_fingerprints,
            "featureMetadataFingerprints": feature_fingerprints,
        }
        prefix = journal._ensure_orchestration_store(store)
        previous_provenances = _tuning_revision_provenances(
            store,
            prefix,
            workflow.workflowRunId,
            request_record,
            experimental_reference,
            inputs,
            journal._stage_starts(store.zw, prefix, workflow.workflowRunId, stage_name),
        )
        existing = journal._validated_done_outcome(
            store,
            prefix,
            workflow.workflowRunId,
            stage_name,
            request_record,
            parents,
        )
        if existing is not None:
            return existing, cast(
                ParameterTuningReport,
                journal.load_stage_report(store, existing, ParameterTuningReport),
            )
        started = journal._start_attempt(
            store.zw,
            prefix,
            workflow.workflowRunId,
            stage_name,
            request_record,
            parents,
            inputs=inputs,
            resume_record=resume_record,
        )
        runner = RnaTuningRun(
            self,
            store,
            workflow,
            request_record,
            plan,
            handoff,
            study_contract,
            answers,
            {
                **inputs,
                "requestSha256": request_record.requestSha256,
                "configSha256": request_record.configSha256,
            },
            design_comparisons=experimental.characterization.comparisons,
            previous_provenances=previous_provenances,
        )
        try:
            with candidate_metric_cache():
                report, evidence = runner.run()
            saved_report, reference = journal._save_stage_report(
                store,
                started,
                report,
                expected_type=ParameterTuningReport,
                attempt_owned=True,
            )
            report = cast(ParameterTuningReport, saved_report)
            artifacts = {
                f"{candidate.candidateId}:{name}": ArtifactReferenceModel.model_validate(
                    value.model_dump()
                )
                for candidate in report.evaluations
                for name, value in candidate.artifacts.items()
            }
            pending = None
            if report.status == "needsInput":
                assert report.needsInput is not None
                pending = WorkflowNeedsInput(
                    questions=[
                        WorkflowQuestion(
                            questionId="parameter_tuning",
                            question=report.needsInput.question,
                            options=report.needsInput.options,
                            evidenceIds=report.needsInput.evidenceIds,
                        )
                    ]
                )
            outcome = journal._complete_attempt(
                started,
                status=report.status,
                report_references=[reference],
                artifacts=artifacts,
                outputs={
                    "tuningEvidence": evidence,
                    "candidateCount": report.totalCandidates,
                },
                needs_input=pending,
                actions=[
                    "assess_rna_defaults",
                    "execute_evidence_requested_experiments",
                    "validate_full_cohort",
                ],
            )
            journal._save_outcome(store.zw, prefix, outcome)
            return outcome, report
        except Exception as exc:
            outcome = journal.finish_exception(
                store,
                prefix,
                workflow,
                started,
                exc,
                outputs={"tuningEvidence": runner.summary()},
            )
            return outcome, ParameterTuningReport.get_blank()

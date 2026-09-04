"""Tests for the bounded Biological Interpretation Agent."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import zarr
from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from zarr.storage import MemoryStore

import scarf.agent.biological_interpretation as biological_module
from scarf.agent.biological_interpretation import (
    _SYSTEM_PROMPT,
    BiologicalContext,
    BiologicalInterpretationAgent,
    BiologicalInterpretationDependencies,
    BiologicalInterpretationNeedsInput,
    BiologicalInterpretationReport,
    ClusterCompositionEvidence,
    ClusterInterpretation,
    ClusterMarkerBatchEvidence,
    ClusterMarkerEvidence,
    ConditionClusterSummary,
    FollowUpRecommendation,
    MarkerFeature,
    TreatmentObservation,
    inspect_cluster_composition,
    inspect_cluster_markers_batch,
    inspect_cluster_markers,
    validate_biological_interpretation_report,
)
from scarf.agent.types import (
    AgentRunInfo,
    ArtifactReferenceModel,
    ExperimentalBiologyHandoff,
    TuningBiologyHandoff,
)
from scarf.storage.refs import ArtifactRef
from scarf.storage.selections import resolve_selection_artifact


class FakeCells:
    def __init__(self, *, replicated: bool = False) -> None:
        if replicated:
            self.cluster_values = np.array(
                [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]
            )
            selected_samples = np.repeat(["s1", "s2", "s3", "s4"], 4)
            selected_conditions = np.repeat(
                ["control", "control", "treated", "treated"],
                4,
            )
        else:
            self.cluster_values = np.array([0, 0, 1, 1, 0, 1, 0, 1])
            selected_samples = np.array(
                ["s1", "s1", "s1", "s1", "s2", "s2", "s2", "s2"]
            )
            selected_conditions = np.array(
                [
                    "control",
                    "control",
                    "control",
                    "control",
                    "treated",
                    "treated",
                    "treated",
                    "treated",
                ]
            )
        self.selected_indices = np.arange(
            1,
            2 * len(self.cluster_values),
            2,
            dtype=np.int64,
        )
        self.N = 2 * len(self.cluster_values)
        selection = np.zeros(self.N, dtype=bool)
        selection[self.selected_indices] = True
        samples = np.full(self.N, "", dtype=selected_samples.dtype)
        samples[self.selected_indices] = selected_samples
        conditions = np.full(self.N, "", dtype=selected_conditions.dtype)
        conditions[self.selected_indices] = selected_conditions
        self.values = {
            "I": selection,
            "ids": np.asarray([f"cell_{index}" for index in range(self.N)]),
            "sample": samples,
            "condition": conditions,
        }
        self.columns = list(self.values)

    def _get_array(self, name: str) -> np.ndarray:
        return self.values[name]

    def fetch(self, name: str, *, key: str) -> np.ndarray:
        mask = np.asarray(self.values[key], dtype=bool)
        return np.asarray(self.values[name])[mask]


class FakeStore:
    def __init__(self, *, replicated: bool = False) -> None:
        self.cells = FakeCells(replicated=replicated)
        self.zw = zarr.open_group(store=MemoryStore(), mode="w")
        cell_data = self.zw.create_group("cellData")
        cell_data.create_array("ids", data=self.cells.values["ids"])
        cell_data.create_array("I", data=self.cells.values["I"])
        self.cell_selection = resolve_selection_artifact(
            self.zw,
            scope="datastore",
            kind="cell_selection",
            values=self.cells.values["I"],
            row_ids=self.cells.values["ids"],
            operation="test_biological_interpretation_selection",
            parameters={},
            inputs={},
            source_column="I",
        )
        self.marker_calls = 0
        self.marker = ArtifactRef(
            scope="assay",
            assay="RNA",
            kind="marker_table",
            artifact_id="a" * 64,
        )
        self.cluster = ArtifactRef(
            scope="assay",
            assay="RNA",
            kind="cluster_labels",
            artifact_id="b" * 64,
        )
        self.marker_features = ArtifactRef(
            scope="assay",
            assay="RNA",
            kind="feature_selection",
            artifact_id="d" * 64,
        )
        self.cluster_selection = self.cell_selection
        self.marker_cluster = self.cluster
        self.marker_arguments: list[dict[str, object]] = []
        self.marker_search_arguments: list[tuple[ArtifactRef, ArtifactRef]] = []

    def run_marker_search(
        self,
        cluster: ArtifactRef,
        *,
        features: ArtifactRef,
    ) -> ArtifactRef:
        self.marker_calls += 1
        self.marker_search_arguments.append((cluster, features))
        return self.marker

    def get_markers(
        self,
        marker: ArtifactRef,
        *,
        group_id: object,
        min_score: float,
        min_frac_exp: float,
    ) -> pd.DataFrame:
        self.marker_arguments.append(
            {
                "marker": marker,
                "group_id": group_id,
                "min_score": min_score,
                "min_frac_exp": min_frac_exp,
            }
        )
        cluster = int(group_id)
        return pd.DataFrame(
            {
                "feature_name": ["C1QA" if cluster == 0 else "CD3D", "ACTB"],
                "feature_index": [10 + cluster, 20],
                "score": [0.9, 0.3],
                "fold_change": [3.0, 1.1],
                "frac_exp": [0.8, 0.9],
                "frac_exp_rest": [0.2, 0.8],
                "mean": [2.0, 3.0],
                "mean_rest": [0.5, 2.8],
                "auc": [0.92, 0.55],
                "p_value_adjusted": [0.001, 0.5],
            }
        )

    def inspect_artifact(self, artifact: object) -> SimpleNamespace:
        inputs: dict[str, object]
        if artifact == self.marker:
            inputs = {"clusters": self.marker_cluster.to_dict()}
        elif artifact == self.cluster:
            inputs = {"cell_selection": self.cluster_selection.to_dict()}
        else:
            inputs = {}
        return SimpleNamespace(exists=True, complete=True, inputs=inputs)

    def load_artifact(self, artifact: object) -> dict[str, np.ndarray]:
        assert artifact == self.cluster
        return {"values": self.cells.cluster_values}


def artifact_model(ref: ArtifactRef) -> ArtifactReferenceModel:
    return ArtifactReferenceModel.from_artifact_ref(ref)


def context(
    store: FakeStore,
    *,
    marker: ArtifactRef | None = None,
    allow_marker_search: bool = False,
) -> RunContext[BiologicalInterpretationDependencies]:
    return RunContext(
        deps=BiologicalInterpretationDependencies(
            store=store,
            cluster=store.cluster,
            fromAssay="RNA",
            graphAssay="RNA",
            markerAssay="RNA",
            markerAssayType="RNA",
            sampleColumn="sample",
            conditionColumn="condition",
            marker=marker,
            markerFeatures=store.marker_features,
            allowMarkerSearch=allow_marker_search,
            designHandoff=ExperimentalBiologyHandoff(
                cellSelection=artifact_model(store.cell_selection),
                conditionColumn="condition",
                observationUnit="sample",
                independentUnit="sample",
                coefficientScope="betweenUnit",
                estimability={"status": "ok", "coefficientEstimable": True},
            ),
        ),
        model=TestModel(),
        usage=RunUsage(),
    )


def test_models_have_blank_and_example_constructors() -> None:
    models = (
        BiologicalContext,
        ConditionClusterSummary,
        ClusterCompositionEvidence,
        MarkerFeature,
        ClusterMarkerEvidence,
        ClusterMarkerBatchEvidence,
        ClusterInterpretation,
        TreatmentObservation,
        FollowUpRecommendation,
        BiologicalInterpretationNeedsInput,
        BiologicalInterpretationReport,
        BiologicalInterpretationDependencies,
        ExperimentalBiologyHandoff,
        TuningBiologyHandoff,
    )
    for model in models:
        assert isinstance(model.get_blank(), model)
        assert isinstance(model.get_example(), model)
        assert all("_" not in field_name for field_name in model.model_fields)


def test_system_prompt_does_not_embed_fictional_output_values() -> None:
    assert "Return this report shape" not in _SYSTEM_PROMPT
    assert "C1QA" not in _SYSTEM_PROMPT
    assert "alveolar macrophage" not in _SYSTEM_PROMPT
    assert "cluster-versus-rest marker specificity" in _SYSTEM_PROMPT
    assert "inspect_cluster_markers_batch exactly once" in _SYSTEM_PROMPT
    assert "Independent units may occur in more than one condition" in _SYSTEM_PROMPT


def test_composition_is_sample_level_and_hides_sample_identifiers() -> None:
    store = FakeStore()
    run_context = context(store)

    result = asyncio.run(inspect_cluster_composition(run_context))

    assert result.totalCells == 8
    assert result.clusterCounts == {"0": 4, "1": 4}
    assert result.clusterArtifact is not None
    assert result.clusterArtifact.artifactId == "b" * 64
    assert result.cellSelection == artifact_model(store.cell_selection)
    assert run_context.deps.cellSelection == store.cell_selection
    np.testing.assert_array_equal(
        run_context.deps.cellIndices,
        store.cells.selected_indices,
    )
    assert {item.nSamples for item in result.conditionSummaries} == {1}
    assert {item.meanFraction for item in result.conditionSummaries} == {0.5}
    serialized = result.model_dump_json()
    assert "s1" not in serialized
    assert "s2" not in serialized


def test_composition_supports_paired_independent_units() -> None:
    store = FakeStore(replicated=True)
    donor_values = np.full(store.cells.N, "", dtype="<U2")
    donor_values[store.cells.selected_indices] = np.repeat(
        ["d1", "d2", "d1", "d2"],
        4,
    )
    store.cells.values["donor"] = donor_values
    store.cells.columns.append("donor")
    run_context = context(store, marker=store.marker)
    run_context.deps.sampleColumn = "donor"

    result = asyncio.run(inspect_cluster_composition(run_context))

    summaries = {
        (summary.condition, summary.clusterId): summary
        for summary in result.conditionSummaries
    }
    assert summaries[("control", "0")].nSamples == 2
    assert summaries[("control", "0")].meanFraction == pytest.approx(0.625)
    assert summaries[("control", "0")].minFraction == pytest.approx(0.5)
    assert summaries[("control", "0")].maxFraction == pytest.approx(0.75)
    assert summaries[("control", "0")].cellCount == 5
    assert summaries[("control", "1")].meanFraction == pytest.approx(0.375)
    assert summaries[("control", "1")].cellCount == 3
    assert summaries[("treated", "0")].meanFraction == pytest.approx(0.25)
    assert summaries[("treated", "0")].cellCount == 2
    assert summaries[("treated", "1")].meanFraction == pytest.approx(0.75)
    assert summaries[("treated", "1")].cellCount == 6
    serialized = result.model_dump_json()
    assert '"d1"' not in serialized
    assert '"d2"' not in serialized


def test_marker_tool_uses_exact_artifact_and_bounded_rows() -> None:
    store = FakeStore()
    run_context = context(store, marker=store.marker)
    asyncio.run(inspect_cluster_composition(run_context))

    result = asyncio.run(inspect_cluster_markers(run_context, cluster_id="0"))

    assert result.markerArtifact is not None
    assert result.markerArtifact.artifactId == "a" * 64
    assert [marker.featureName for marker in result.markers] == ["C1QA", "ACTB"]
    assert result.markers[0].featureIndex == 10
    assert store.marker_calls == 0
    assert store.marker_arguments[0] == {
        "marker": store.marker,
        "group_id": 0,
        "min_score": 0.25,
        "min_frac_exp": 0.2,
    }


def test_marker_search_runs_once_only_when_authorized() -> None:
    store = FakeStore()
    run_context = context(store, allow_marker_search=True)
    asyncio.run(inspect_cluster_composition(run_context))

    asyncio.run(inspect_cluster_markers(run_context, cluster_id="0"))
    asyncio.run(inspect_cluster_markers(run_context, cluster_id="1"))

    assert store.marker_calls == 1
    assert store.marker_search_arguments == [
        (store.cluster, store.marker_features),
    ]


def test_marker_batch_returns_all_selected_clusters_in_one_result() -> None:
    store = FakeStore()
    run_context = context(store, marker=store.marker)
    asyncio.run(inspect_cluster_composition(run_context))

    result = asyncio.run(
        inspect_cluster_markers_batch(run_context, cluster_ids=["0", "1"])
    )

    assert [cluster.clusterId for cluster in result.clusters] == ["0", "1"]
    assert result.evidenceIds == [
        f"markers:{'a' * 64}:clusters:{'b' * 64}:cluster:0",
        f"markers:{'a' * 64}:clusters:{'b' * 64}:cluster:1",
    ]


def test_marker_artifact_must_use_the_exact_cluster_artifact() -> None:
    store = FakeStore()
    store.marker_cluster = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="cluster_labels",
        artifact_id="c" * 64,
    )
    run_context = context(store, marker=store.marker)
    asyncio.run(inspect_cluster_composition(run_context))

    with pytest.raises(ModelRetry, match="exact cluster artifact"):
        asyncio.run(inspect_cluster_markers(run_context, cluster_id="0"))


def test_empty_marker_table_does_not_create_marker_evidence() -> None:
    store = FakeStore()
    store.get_markers = lambda _marker, **_kwargs: pd.DataFrame(
        columns=["feature_name", "score"]
    )
    run_context = context(store, marker=store.marker)
    asyncio.run(inspect_cluster_composition(run_context))

    result = asyncio.run(inspect_cluster_markers(run_context, cluster_id="0"))

    assert result.markers == []
    assert result.evidenceId == ""
    assert run_context.deps.markerEvidenceIds == {}


def test_validator_attaches_exact_marker_evidence_when_omitted() -> None:
    store = FakeStore()
    run_context = context(store, marker=store.marker)
    asyncio.run(inspect_cluster_composition(run_context))
    marker = asyncio.run(inspect_cluster_markers(run_context, cluster_id="0"))
    report = BiologicalInterpretationReport(
        status="done",
        clusterInterpretations=[
            ClusterInterpretation(
                clusterId="0",
                proposedIdentity="macrophage-like",
                evidenceIds=[],
            )
        ],
    )

    validated = validate_biological_interpretation_report(
        report,
        run_context.deps,
    )

    assert validated.clusterInterpretations[0].evidenceIds == [marker.evidenceId]
    assert marker.evidenceId in validated.evidenceIds


def test_atac_markers_force_low_confidence_identity_hypotheses() -> None:
    store = FakeStore()
    run_context = context(store, marker=store.marker)
    run_context.deps.markerAssayType = "ATAC"
    asyncio.run(inspect_cluster_composition(run_context))
    marker = asyncio.run(inspect_cluster_markers(run_context, cluster_id="0"))
    report = BiologicalInterpretationReport(
        status="done",
        clusterInterpretations=[
            ClusterInterpretation(
                clusterId="0",
                proposedIdentity="confident peak-only identity",
                identityIsHypothesis=False,
                confidence="high",
                evidenceIds=[marker.evidenceId],
            )
        ],
    )

    validated = validate_biological_interpretation_report(report, run_context.deps)

    interpretation = validated.clusterInterpretations[0]
    assert interpretation.identityIsHypothesis is True
    assert interpretation.confidence == "low"
    assert any(
        "ATAC peak markers are descriptive" in value for value in validated.limitations
    )


def test_validator_omits_interpretations_without_marker_evidence() -> None:
    store = FakeStore()
    run_context = context(store, marker=store.marker)
    asyncio.run(inspect_cluster_composition(run_context))
    marker = asyncio.run(inspect_cluster_markers(run_context, cluster_id="0"))
    report = BiologicalInterpretationReport(
        status="done",
        clusterInterpretations=[
            ClusterInterpretation(
                clusterId="0",
                proposedIdentity="macrophage-like",
            ),
            ClusterInterpretation(
                clusterId="1",
                proposedIdentity="T cell-like",
            ),
        ],
    )

    validated = validate_biological_interpretation_report(
        report,
        run_context.deps,
    )

    assert [item.clusterId for item in validated.clusterInterpretations] == ["0"]
    assert validated.clusterInterpretations[0].evidenceIds == [marker.evidenceId]
    assert any(
        "were omitted for clusters: 1" in limitation
        for limitation in validated.limitations
    )


def test_done_report_requires_at_least_one_supported_interpretation() -> None:
    store = FakeStore()
    run_context = context(store, marker=store.marker)
    asyncio.run(inspect_cluster_composition(run_context))
    report = BiologicalInterpretationReport(
        status="done",
        clusterInterpretations=[
            ClusterInterpretation(
                clusterId="1",
                proposedIdentity="T cell-like",
            )
        ],
    )

    with pytest.raises(ModelRetry, match="at least one cluster interpretation"):
        validate_biological_interpretation_report(report, run_context.deps)


def test_validator_rejects_unobserved_evidence() -> None:
    store = FakeStore()
    run_context = context(store, marker=store.marker)
    asyncio.run(inspect_cluster_composition(run_context))
    report = BiologicalInterpretationReport(
        status="done",
        clusterInterpretations=[
            ClusterInterpretation(
                clusterId="0",
                proposedIdentity="invented",
                evidenceIds=[f"markers:{'a' * 64}:clusters:{'b' * 64}:cluster:0"],
            )
        ],
        evidenceIds=["evidence:invented"],
    )

    with pytest.raises(Exception, match="Unknown evidenceIds"):
        validate_biological_interpretation_report(report, run_context.deps)


def test_treatment_observation_requires_evidence_for_its_exact_cluster() -> None:
    store = FakeStore()
    run_context = context(store, marker=store.marker)
    composition = asyncio.run(inspect_cluster_composition(run_context))
    cluster_zero_evidence = [
        summary.evidenceId
        for summary in composition.conditionSummaries
        if summary.clusterId == "0"
    ]
    report = BiologicalInterpretationReport(
        status="needsInput",
        treatmentObservations=[
            TreatmentObservation(
                clusterId="1",
                referenceCondition="control",
                comparisonCondition="treated",
                direction="equal",
                observation="Cluster 1 differs between conditions.",
                evidenceIds=cluster_zero_evidence,
            )
        ],
        needsInput=BiologicalInterpretationNeedsInput(
            question="More biological evidence is required."
        ),
    )

    with pytest.raises(ModelRetry, match="exact cluster"):
        validate_biological_interpretation_report(report, run_context.deps)


def test_treatment_observation_requires_sample_level_replication() -> None:
    store = FakeStore()
    run_context = context(store, marker=store.marker)
    composition = asyncio.run(inspect_cluster_composition(run_context))
    evidence_ids = [
        summary.evidenceId
        for summary in composition.conditionSummaries
        if summary.clusterId == "0"
    ]
    report = BiologicalInterpretationReport(
        status="needsInput",
        treatmentObservations=[
            TreatmentObservation(
                clusterId="0",
                referenceCondition="control",
                comparisonCondition="treated",
                direction="equal",
                evidenceIds=evidence_ids,
            )
        ],
        needsInput=BiologicalInterpretationNeedsInput(
            question="More samples required."
        ),
    )

    with pytest.raises(ModelRetry, match="at least two samples"):
        validate_biological_interpretation_report(report, run_context.deps)

    run_context = context(store, marker=store.marker)
    run_context.deps.sampleColumn = None
    composition = asyncio.run(inspect_cluster_composition(run_context))
    report.treatmentObservations[0].evidenceIds = [
        summary.evidenceId
        for summary in composition.conditionSummaries
        if summary.clusterId == "0"
    ]
    with pytest.raises(ModelRetry, match="independent-unit composition"):
        validate_biological_interpretation_report(report, run_context.deps)


def test_valid_treatment_observation_is_canonical_and_descriptive() -> None:
    store = FakeStore(replicated=True)
    run_context = context(store, marker=store.marker)
    run_context.deps.designHandoff = ExperimentalBiologyHandoff(
        cellSelection=artifact_model(store.cell_selection),
        conditionColumn="condition",
        observationUnit="sample",
        coefficientScope="unresolvedUnit",
        estimability={"status": "notComputed"},
    )
    composition = asyncio.run(inspect_cluster_composition(run_context))
    evidence_ids = [
        summary.evidenceId
        for summary in composition.conditionSummaries
        if summary.clusterId == "0"
    ]
    report = BiologicalInterpretationReport(
        status="needsInput",
        treatmentObservations=[
            TreatmentObservation(
                clusterId="0",
                referenceCondition="control",
                comparisonCondition="treated",
                direction="lower",
                observation="Treatment caused cluster 0 depletion.",
                evidenceIds=evidence_ids,
            )
        ],
        needsInput=BiologicalInterpretationNeedsInput(
            question="Run a replicated differential-abundance analysis."
        ),
    )

    with pytest.raises(ModelRetry, match="estimable experimental contrast"):
        validate_biological_interpretation_report(report, run_context.deps)

    run_context.deps.designHandoff = ExperimentalBiologyHandoff(
        cellSelection=artifact_model(store.cell_selection),
        conditionColumn="condition",
        observationUnit="sample",
        independentUnit="sample",
        coefficientScope="betweenUnit",
        estimability={"status": "ok", "coefficientEstimable": True},
    )
    validated = validate_biological_interpretation_report(report, run_context.deps)

    observation = validated.treatmentObservations[0].observation
    assert observation.startswith(
        "Cluster 0 has a lower mean independent-unit fraction in treated"
    )
    assert "caused" not in observation
    assert any("descriptive summaries" in item for item in validated.limitations)


def test_treatment_observation_requires_the_aggregated_unit_to_be_independent() -> None:
    store = FakeStore(replicated=True)
    run_context = context(store, marker=store.marker)
    run_context.deps.designHandoff = ExperimentalBiologyHandoff(
        cellSelection=artifact_model(store.cell_selection),
        conditionColumn="condition",
        observationUnit="sample",
        independentUnit="donor",
        coefficientScope="betweenUnit",
        estimability={"status": "ok", "coefficientEstimable": True},
    )
    composition = asyncio.run(inspect_cluster_composition(run_context))
    evidence_ids = [
        summary.evidenceId
        for summary in composition.conditionSummaries
        if summary.clusterId == "0"
    ]
    report = BiologicalInterpretationReport(
        status="needsInput",
        treatmentObservations=[
            TreatmentObservation(
                clusterId="0",
                referenceCondition="control",
                comparisonCondition="treated",
                direction="lower",
                evidenceIds=evidence_ids,
            )
        ],
        needsInput=BiologicalInterpretationNeedsInput(
            question="Aggregate by the independent donor unit."
        ),
    )

    with pytest.raises(ModelRetry, match="aggregation at the independent unit"):
        validate_biological_interpretation_report(report, run_context.deps)

    run_context = context(store, marker=store.marker)
    run_context.deps.sampleColumn = "donor"
    run_context.deps.designHandoff = ExperimentalBiologyHandoff(
        cellSelection=artifact_model(store.cell_selection),
        conditionColumn="condition",
        observationUnit="sample",
        independentUnit="donor",
        coefficientScope="betweenUnit",
        estimability={"status": "ok", "coefficientEstimable": True},
    )
    donor_values = np.full(store.cells.N, "", dtype="<U2")
    donor_values[store.cells.selected_indices] = np.repeat(
        ["d1", "d2", "d3", "d4"],
        4,
    )
    store.cells.values["donor"] = donor_values
    store.cells.columns.append("donor")
    composition = asyncio.run(inspect_cluster_composition(run_context))
    evidence_ids = [
        summary.evidenceId
        for summary in composition.conditionSummaries
        if summary.clusterId == "0"
    ]
    report.treatmentObservations[0].evidenceIds = evidence_ids

    validated = validate_biological_interpretation_report(report, run_context.deps)

    assert validated.treatmentObservations
    assert "independent-unit fraction" in validated.treatmentObservations[0].observation


def test_treatment_direction_and_condition_names_must_match_evidence() -> None:
    store = FakeStore(replicated=True)
    run_context = context(store, marker=store.marker)
    composition = asyncio.run(inspect_cluster_composition(run_context))
    evidence_ids = [
        summary.evidenceId
        for summary in composition.conditionSummaries
        if summary.clusterId == "0"
    ]
    observation = TreatmentObservation(
        clusterId="0",
        referenceCondition="control",
        comparisonCondition="treated",
        direction="higher",
        evidenceIds=evidence_ids,
    )
    report = BiologicalInterpretationReport(
        status="needsInput",
        treatmentObservations=[observation],
        needsInput=BiologicalInterpretationNeedsInput(question="Check the contrast."),
    )

    with pytest.raises(ModelRetry, match="direction does not match"):
        validate_biological_interpretation_report(report, run_context.deps)

    observation.direction = "lower"
    observation.comparisonCondition = "other"
    with pytest.raises(ModelRetry, match="conditions must match"):
        validate_biological_interpretation_report(report, run_context.deps)


def test_cluster_identity_rejects_condition_evidence() -> None:
    store = FakeStore(replicated=True)
    run_context = context(store, marker=store.marker)
    composition = asyncio.run(inspect_cluster_composition(run_context))
    marker = asyncio.run(inspect_cluster_markers(run_context, cluster_id="0"))
    condition_id = next(
        summary.evidenceId
        for summary in composition.conditionSummaries
        if summary.clusterId == "0"
    )
    report = BiologicalInterpretationReport(
        status="done",
        clusterInterpretations=[
            ClusterInterpretation(
                clusterId="0",
                proposedIdentity="macrophage-like",
                evidenceIds=[marker.evidenceId, condition_id],
            )
        ],
    )

    with pytest.raises(ModelRetry, match="only their exact marker evidence"):
        validate_biological_interpretation_report(report, run_context.deps)


def test_agent_waits_for_tools_and_returns_audited_report() -> None:
    store = FakeStore()
    seen_tools: set[str] = set()

    async def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen_tools.update(tool.name for tool in info.function_tools)
        returns = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if not returns:
            return ModelResponse(
                parts=[ToolCallPart(tool_name="inspect_cluster_composition", args={})]
            )
        if len(returns) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="inspect_cluster_markers_batch",
                        args={"cluster_ids": ["0"]},
                    )
                ]
            )
        report = BiologicalInterpretationReport(
            status="done",
            clusterInterpretations=[
                ClusterInterpretation(
                    clusterId="0",
                    proposedIdentity="macrophage-like",
                    confidence="medium",
                    rationale="C1QA is the strongest returned feature.",
                    evidenceIds=[f"markers:{'a' * 64}:clusters:{'b' * 64}:cluster:0"],
                )
            ],
            evidenceIds=[
                f"composition:{'b' * 64}:counts",
                f"markers:{'a' * 64}:clusters:{'b' * 64}:cluster:0",
            ],
            limitations=["Identity is a hypothesis."],
            stopReason="One requested cluster was reviewed.",
        )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args=report.model_dump(),
                )
            ]
        )

    tuning_handoff = TuningBiologyHandoff(
        cellSelection=artifact_model(store.cell_selection),
        recommendedCandidateId="baseline",
        clusterArtifact=artifact_model(store.cluster),
        evidenceIds=["candidate:baseline:clusters"],
    )
    experimental_handoff = ExperimentalBiologyHandoff(
        cellSelection=artifact_model(store.cell_selection),
        conditionColumn="condition",
        observationUnit="sample",
        independentUnit="sample",
        coefficientScope="betweenUnit",
        estimability={"status": "ok", "coefficientEstimable": True},
        evidenceIds=["estimability:condition"],
    )
    result = BiologicalInterpretationAgent(FunctionModel(reply)).run(
        store,
        cluster=store.cluster,
        biological_context=BiologicalContext(
            organism="Homo sapiens",
            tissue="lung",
        ),
        tuning_handoff=tuning_handoff,
        experimental_handoff=experimental_handoff,
        marker=store.marker,
    )

    assert result.status == "done"
    assert result.clusterInterpretations[0].proposedIdentity == "macrophage-like"
    assert result.markerArtifact is not None
    assert result.runInfo.agentName == "biological_interpretation"
    assert result.runInfo.usage.toolCalls == 2
    assert seen_tools == {
        "inspect_cluster_composition",
        "inspect_cluster_markers_batch",
    }


def test_exact_cluster_artifact_conflict_is_rejected_before_model_execution() -> None:
    store = FakeStore()
    tuning_handoff = TuningBiologyHandoff(
        cellSelection=artifact_model(store.cell_selection),
        clusterArtifact=artifact_model(store.cluster),
    )
    other_cluster = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="cluster_labels",
        artifact_id="c" * 64,
    )

    with pytest.raises(ValueError, match="cluster conflicts with tuning_handoff"):
        BiologicalInterpretationAgent(object()).run(
            store,
            cluster=other_cluster,
            tuning_handoff=tuning_handoff,
            marker=store.marker,
        )


def test_handoff_selection_must_match_exact_cluster_selection() -> None:
    store = FakeStore()
    other_values = np.asarray(store.cells.values["I"], dtype=bool).copy()
    other_values[0] = True
    other_values[1] = False
    other_selection = resolve_selection_artifact(
        store.zw,
        scope="datastore",
        kind="cell_selection",
        values=other_values,
        row_ids=store.cells.values["ids"],
        operation="test_other_biological_interpretation_selection",
        parameters={},
        inputs={},
        source_column="other",
    )
    tuning_handoff = TuningBiologyHandoff(
        cellSelection=artifact_model(other_selection),
        clusterArtifact=artifact_model(store.cluster),
    )

    with pytest.raises(
        ValueError, match="handoff cell selection conflicts with cluster"
    ):
        BiologicalInterpretationAgent(object()).run(
            store,
            tuning_handoff=tuning_handoff,
            marker=store.marker,
        )


def test_integrated_handoff_uses_marker_assay_without_claiming_graph_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scarf.agent import biological_interpretation as module

    store = FakeStore()
    store.cluster = ArtifactRef(
        scope="datastore",
        assay=None,
        kind="cluster_labels",
        artifact_id="b" * 64,
    )
    store.marker_cluster = store.cluster
    handoff = TuningBiologyHandoff(
        fromAssay="RNA",
        graphAssay=None,
        markerAssay="RNA",
        cellSelection=artifact_model(store.cell_selection),
        recommendedCandidateId="wnn_1",
        clusterArtifact=ArtifactReferenceModel(
            scope="datastore",
            assay=None,
            kind="cluster_labels",
            artifactId="b" * 64,
        ),
        evidenceIds=["integration:wnn_1:clusters"],
    )

    def fake_run_agent_sync(**kwargs: object) -> SimpleNamespace:
        deps = kwargs["deps"]
        assert isinstance(deps, BiologicalInterpretationDependencies)
        assert deps.graphAssay is None
        assert deps.fromAssay == "RNA"
        assert deps.markerAssay == "RNA"
        deps.clusterValues = {"0": 0}
        return SimpleNamespace(
            output=BiologicalInterpretationReport(
                status="needsInput",
                needsInput=BiologicalInterpretationNeedsInput(
                    question="Inspect markers before interpretation.",
                    requiredInputs=["markerEvidence"],
                ),
            ),
            runInfo=AgentRunInfo(agentName="biological_interpretation"),
        )

    monkeypatch.setattr(module, "run_agent_sync", fake_run_agent_sync)

    result = BiologicalInterpretationAgent(object()).run(
        store,
        tuning_handoff=handoff,
        marker=store.marker,
    )

    assert result.status == "needsInput"
    assert result.graphAssay is None
    assert result.markerAssay == "RNA"
    assert result.clusterArtifact is not None
    assert result.clusterArtifact.scope == "datastore"


def test_integrated_handoff_requires_explicit_marker_assay() -> None:
    store = FakeStore()
    handoff = TuningBiologyHandoff(
        fromAssay="RNA",
        cellSelection=artifact_model(store.cell_selection),
        clusterArtifact=ArtifactReferenceModel(
            scope="datastore",
            assay=None,
            kind="cluster_labels",
            artifactId="b" * 64,
        ),
    )

    with pytest.raises(ValueError, match="markerAssay"):
        BiologicalInterpretationAgent(object()).run(
            store,
            tuning_handoff=handoff,
            marker=store.marker,
        )


def test_biological_scalar_and_column_validation_edges() -> None:
    assert biological_module._finite_float(None) is None
    assert biological_module._finite_float("not-a-number") is None
    assert biological_module._finite_float(float("nan")) is None
    assert biological_module._string_value(np.int64(3)) == "3"
    with pytest.raises(ValueError, match="not present in cell metadata"):
        biological_module._check_column(FakeStore(), "missing", "sample column")

    tool = SimpleNamespace(name="future_tool")
    deps = context(FakeStore()).deps
    assert (
        biological_module._prepare_biological_interpretation_tool(
            SimpleNamespace(deps=deps), tool
        )
        is tool
    )


def test_cluster_composition_artifact_validation_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeStore()

    def run(deps: BiologicalInterpretationDependencies) -> None:
        asyncio.run(inspect_cluster_composition(SimpleNamespace(deps=deps)))

    deps = context(store).deps
    deps.cluster = None
    with pytest.raises(ValueError, match="exact cluster artifact"):
        run(deps)

    deps = context(store).deps
    deps.cluster = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="neighbors",
        artifact_id="1" * 64,
    )
    with pytest.raises(ValueError, match="cluster_labels or cluster_cut"):
        run(deps)

    deps = context(store).deps
    deps.cluster = ArtifactRef(
        scope="assay",
        assay="ADT",
        kind="cluster_labels",
        artifact_id="2" * 64,
    )
    with pytest.raises(ValueError, match="different assay"):
        run(deps)

    original_inspect = store.inspect_artifact
    for status, message in (
        (SimpleNamespace(exists=False, complete=True, inputs={}), "does not exist"),
        (SimpleNamespace(exists=True, complete=False, inputs={}), "is incomplete"),
        (SimpleNamespace(exists=True, complete=True, inputs={}), "no cell-selection"),
        (
            SimpleNamespace(
                exists=True,
                complete=True,
                inputs={
                    "cell_selection": ArtifactRef(
                        scope="assay",
                        assay="RNA",
                        kind="cell_selection",
                        artifact_id="3" * 64,
                    ).to_dict()
                },
            ),
            "invalid cell-selection",
        ),
    ):
        monkeypatch.setattr(store, "inspect_artifact", lambda _ref, value=status: value)
        with pytest.raises(ValueError, match=message):
            run(context(store).deps)
    monkeypatch.setattr(store, "inspect_artifact", original_inspect)

    deps = context(store).deps
    deps.cellSelection = ArtifactRef(
        scope="datastore",
        kind="cell_selection",
        artifact_id="4" * 64,
    )
    with pytest.raises(ValueError, match="conflicts with the prepared"):
        run(deps)

    monkeypatch.setattr(store, "load_artifact", lambda _ref: {})
    with pytest.raises(ValueError, match="label vector"):
        run(context(store).deps)
    monkeypatch.setattr(
        store,
        "load_artifact",
        lambda _ref: {"values": np.asarray([0])},
    )
    with pytest.raises(ValueError, match="do not align"):
        run(context(store).deps)
    monkeypatch.setattr(
        store,
        "load_artifact",
        lambda _ref: {"values": np.asarray([], dtype=int)},
    )
    monkeypatch.setattr(
        biological_module,
        "read_stored_selection_indices",
        lambda *_args, **_kwargs: np.asarray([], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="selects no cells"):
        run(context(store).deps)


def test_cluster_composition_metadata_alignment_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeStore()
    original_rows = biological_module.read_metadata_rows
    original_missing = biological_module.read_metadata_missing_rows

    monkeypatch.setattr(
        biological_module,
        "read_metadata_rows",
        lambda *_args, **_kwargs: np.asarray(["control"]),
    )
    with pytest.raises(ValueError, match="condition and cluster"):
        asyncio.run(inspect_cluster_composition(context(store)))

    monkeypatch.setattr(biological_module, "read_metadata_rows", original_rows)
    monkeypatch.setattr(
        biological_module,
        "read_metadata_missing_rows",
        lambda *_args, **_kwargs: np.asarray(
            [True] + [False] * (len(store.cells.cluster_values) - 1)
        ),
    )
    with pytest.raises(ValueError, match="condition column contains missing"):
        asyncio.run(inspect_cluster_composition(context(store)))

    monkeypatch.setattr(
        biological_module, "read_metadata_missing_rows", original_missing
    )

    def sample_misaligned(
        table: object, column: str, indices: np.ndarray
    ) -> np.ndarray:
        values = original_rows(table, column, indices)
        return values[:-1] if column == "sample" else values

    monkeypatch.setattr(biological_module, "read_metadata_rows", sample_misaligned)
    with pytest.raises(ValueError, match="sample and cluster"):
        asyncio.run(inspect_cluster_composition(context(store)))

    monkeypatch.setattr(biological_module, "read_metadata_rows", original_rows)

    def sample_missing(_table: object, column: str, indices: np.ndarray) -> np.ndarray:
        if column == "sample":
            return np.asarray([True] + [False] * (len(indices) - 1))
        return np.zeros(len(indices), dtype=bool)

    monkeypatch.setattr(biological_module, "read_metadata_missing_rows", sample_missing)
    with pytest.raises(ValueError, match="sample column contains missing"):
        asyncio.run(inspect_cluster_composition(context(store)))


def test_marker_tool_validation_and_batch_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeStore()
    unprepared = context(store)
    with pytest.raises(ModelRetry, match="composition"):
        asyncio.run(inspect_cluster_markers(unprepared, "0"))
    with pytest.raises(ModelRetry, match="composition"):
        asyncio.run(inspect_cluster_markers_batch(unprepared, ["0"]))

    prepared = context(store)
    asyncio.run(inspect_cluster_composition(prepared))
    with pytest.raises(ModelRetry, match="must be one of"):
        asyncio.run(inspect_cluster_markers(prepared, "missing"))

    prepared.deps.marker = None
    prepared.deps.allowMarkerSearch = False
    unavailable = asyncio.run(inspect_cluster_markers(prepared, "0"))
    assert not unavailable.evidenceId

    prepared.deps.allowMarkerSearch = True
    prepared.deps.markerFeatures = None
    missing_features = asyncio.run(inspect_cluster_markers(prepared, "0"))
    assert "exact feature selection" in missing_features.warnings[0]

    prepared.deps.markerFeatures = store.marker_features
    monkeypatch.setattr(store, "run_marker_search", lambda *_args, **_kwargs: object())
    with pytest.raises(RuntimeError, match="artifact reference"):
        asyncio.run(inspect_cluster_markers(prepared, "0"))

    for marker, message in (
        (
            ArtifactRef(
                scope="assay",
                assay="RNA",
                kind="neighbors",
                artifact_id="5" * 64,
            ),
            "marker_table",
        ),
        (
            ArtifactRef(
                scope="assay",
                assay="ADT",
                kind="marker_table",
                artifact_id="6" * 64,
            ),
            "different assay",
        ),
    ):
        prepared.deps.marker = marker
        with pytest.raises(ModelRetry, match=message):
            asyncio.run(inspect_cluster_markers(prepared, "0"))

    prepared.deps.marker = store.marker
    original_inspect = store.inspect_artifact
    for status, message in (
        (SimpleNamespace(exists=False, complete=True, inputs={}), "does not exist"),
        (SimpleNamespace(exists=True, complete=False, inputs={}), "is incomplete"),
    ):
        monkeypatch.setattr(store, "inspect_artifact", lambda _ref, value=status: value)
        with pytest.raises(ModelRetry, match=message):
            asyncio.run(inspect_cluster_markers(prepared, "0"))
    monkeypatch.setattr(store, "inspect_artifact", original_inspect)

    with pytest.raises(ModelRetry, match="at least one"):
        asyncio.run(inspect_cluster_markers_batch(prepared, []))
    prepared.deps.maxClusters = 1
    with pytest.raises(ModelRetry, match="at most"):
        asyncio.run(inspect_cluster_markers_batch(prepared, ["0", "1"]))
    prepared.deps.maxClusters = 4
    with pytest.raises(ModelRetry, match="duplicates"):
        asyncio.run(inspect_cluster_markers_batch(prepared, ["0", "0"]))


def test_treatment_observation_validation_edges() -> None:
    store = FakeStore(replicated=True)
    deps = context(store).deps
    deps.clusterValues = {"0": 0}
    control = ConditionClusterSummary(
        condition="control",
        clusterId="0",
        nSamples=2,
        meanFraction=0.5,
        minFraction=0.5,
        maxFraction=0.5,
        cellCount=4,
        evidenceId="control:0",
    )
    treated = control.model_copy(
        update={"condition": "treated", "evidenceId": "treated:0"}
    )
    deps.conditionEvidence = {
        control.evidenceId: control,
        treated.evidenceId: treated,
    }
    deps.evidenceIds = set(deps.conditionEvidence)
    observation = TreatmentObservation(
        clusterId="0",
        referenceCondition="control",
        comparisonCondition="treated",
        direction="equal",
        evidenceIds=[control.evidenceId, treated.evidenceId],
    )

    def report(value: TreatmentObservation) -> BiologicalInterpretationReport:
        return BiologicalInterpretationReport(
            status="needsInput",
            treatmentObservations=[value],
        )

    without_condition = deps.model_copy(update={"conditionColumn": None})
    with pytest.raises(ModelRetry, match="condition column"):
        biological_module._canonicalize_treatment_observations(
            report(observation), without_condition
        )
    with pytest.raises(ModelRetry, match="remain descriptive"):
        biological_module._canonicalize_treatment_observations(
            report(observation.model_copy(update={"isDescriptiveOnly": False})),
            deps,
        )
    with pytest.raises(ModelRetry, match="exactly two distinct"):
        biological_module._canonicalize_treatment_observations(
            report(
                observation.model_copy(
                    update={"evidenceIds": [control.evidenceId, control.evidenceId]}
                )
            ),
            deps,
        )
    with pytest.raises(ModelRetry, match="condition composition evidence"):
        biological_module._canonicalize_treatment_observations(
            report(
                observation.model_copy(
                    update={"evidenceIds": [control.evidenceId, "unknown"]}
                )
            ),
            deps,
        )
    with pytest.raises(ModelRetry, match="distinct named conditions"):
        biological_module._canonicalize_treatment_observations(
            report(observation.model_copy(update={"comparisonCondition": "control"})),
            deps,
        )
    canonical = biological_module._canonicalize_treatment_observations(
        report(observation), deps
    )
    assert "equal mean independent-unit fractions" in canonical[0].observation

    higher = treated.model_copy(update={"meanFraction": 0.75})
    deps.conditionEvidence[treated.evidenceId] = higher
    canonical = biological_module._canonicalize_treatment_observations(
        report(observation.model_copy(update={"direction": "higher"})), deps
    )
    assert "higher mean independent-unit fraction" in canonical[0].observation

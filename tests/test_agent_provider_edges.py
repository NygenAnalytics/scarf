"""Standalone scientific agent and provider failure edge cases."""

from tests.agent_examples import example

import asyncio
from types import SimpleNamespace

from scarf.storage.refs import ArtifactRef

import pytest
from pydantic import ValidationError
from pydantic_ai import ModelRetry, UnexpectedModelBehavior

import scarf.agent.biological_interpretation.tools as biological_tools
import scarf.agent.biological_interpretation.validation as biological_validation
import scarf.agent.config.agent_exec as agent_exec_module
import scarf.agent.data_enrichment.agent as enrichment_agent
import scarf.agent.data_enrichment.tools as enrichment_tools
import scarf.agent.data_enrichment.validation as enrichment_validation
import scarf.agent.experimental_context.tools as experimental_tools
import scarf.agent.experimental_context.validation as experimental_validation
from scarf.agent.biological_interpretation import (
    BiologicalInterpretationNeedsInput,
    BiologicalInterpretationReport,
    ClusterCompositionEvidence,
    ClusterMarkerEvidence,
)
from scarf.agent.biological_interpretation.contracts import (
    BiologicalInterpretationDependencies,
)
from scarf.agent.data_enrichment import (
    AssayFeatureInspection,
    DataEnrichmentAgent,
    DataEnrichmentDependencies,
    DataEnrichmentToolCall,
)
from scarf.agent.experimental_context import (
    CellQcProfileEvidence,
    ExperimentalContextDependencies,
)
from scarf.agent.experimental_context.contracts import (
    CovariateCharacterization,
    CovariateProposal,
)


def test_data_enrichment_cache_rollback_and_pending_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspection = example(AssayFeatureInspection)
    completed = DataEnrichmentDependencies(
        store=object(),
        assays=["RNA"],
        inspections={"RNA": inspection},
        toolCalls=[
            DataEnrichmentToolCall(
                name="inspect_assay_features_batch",
                assay="all",
            )
        ],
    )
    completed_context = SimpleNamespace(deps=completed)

    assert (
        asyncio.run(
            enrichment_tools.inspect_assay_features(
                completed_context,
                assay_name="RNA",
            )
        )
        == inspection
    )
    cached_batch = asyncio.run(
        enrichment_tools.inspect_assay_features_batch(completed_context)
    )
    assert cached_batch.inspections == [inspection]
    assert cached_batch.evidenceIds == inspection.evidenceIds

    incomplete = DataEnrichmentDependencies(
        assays=["RNA"],
        toolCalls=[DataEnrichmentToolCall(name="sentinel", assay="RNA")],
    )
    with pytest.raises(ModelRetry, match="datastore"):
        asyncio.run(
            enrichment_tools.inspect_assay_features_batch(
                SimpleNamespace(deps=incomplete)
            )
        )
    assert [call.name for call in incomplete.toolCalls] == ["sentinel"]

    provider_error = UnexpectedModelBehavior("provider output failed")
    failed = enrichment_validation.failed_data_enrichment_report(
        DataEnrichmentDependencies(
            assays=["RNA"],
            inspections={"RNA": inspection},
            evidenceIds=set(inspection.evidenceIds),
        ),
        error=provider_error,
        model_name="test-model",
    )
    assert failed.status == "failed"
    assert failed.policies == []
    assert failed.inspections == [inspection]

    def fail_before_inspection(**_kwargs: object) -> object:
        raise UnexpectedModelBehavior("no inspection completed")

    monkeypatch.setattr(enrichment_agent, "run_agent_sync", fail_before_inspection)
    store = SimpleNamespace(assay_names=["RNA"])
    failed = DataEnrichmentAgent(object()).run(store)
    assert failed.status == "failed"
    assert failed.inspections == []
    assert failed.policies == []


def test_biological_interpretation_cache_and_fallback_branches() -> None:
    composition = example(ClusterCompositionEvidence)
    composition_deps = BiologicalInterpretationDependencies(
        compositionEvidence=composition
    )
    assert (
        asyncio.run(
            biological_tools.inspect_cluster_composition(
                SimpleNamespace(deps=composition_deps)
            )
        )
        == composition
    )

    marker = example(ClusterMarkerEvidence)
    marker_deps = BiologicalInterpretationDependencies(
        clusterValues={marker.clusterId: 0},
        markerEvidence={marker.clusterId: marker},
    )
    assert (
        asyncio.run(
            biological_tools.inspect_cluster_markers(
                SimpleNamespace(deps=marker_deps),
                cluster_id=marker.clusterId,
            )
        )
        == marker
    )

    invalid_report = BiologicalInterpretationReport(
        status="done",
        needsInput=BiologicalInterpretationNeedsInput(question="More context?"),
    )
    with pytest.raises(ModelRetry, match="Only a needsInput"):
        biological_validation.validate_biological_interpretation_report(
            invalid_report,
            BiologicalInterpretationDependencies(clusterValues={"0": 0}),
        )

    provider_error = UnexpectedModelBehavior("structured output failed")
    with pytest.raises(UnexpectedModelBehavior, match="structured output failed"):
        biological_validation.fallback_biological_interpretation_report(
            BiologicalInterpretationDependencies(),
            error=provider_error,
            model_name="test-model",
        )
    needs_markers = biological_validation.fallback_biological_interpretation_report(
        BiologicalInterpretationDependencies(
            clusterValues={"0": 0},
            evidenceIds={"composition:clusters"},
        ),
        error=provider_error,
        model_name="test-model",
    )
    assert needs_markers.status == "failed"
    assert needs_markers.needsInput is None
    assert needs_markers.evidenceIds == ["composition:clusters"]


def test_experimental_context_rejects_invalid_batches_and_preserves_failed_evidence() -> (
    None
):
    invalid_batches = (
        (
            [{"name": "batch", "domain": "technical", "kind": "categorical"}],
            "missing",
            "Unknown batch column",
        ),
        (
            [{"name": "condition", "domain": "biological", "kind": "categorical"}],
            "condition",
            "must be classified as technical",
        ),
        (
            [{"name": "depth", "domain": "technical", "kind": "continuous"}],
            "depth",
            "must be categorical",
        ),
    )
    for columns, batch_column, message in invalid_batches:
        deps = ExperimentalContextDependencies(
            characterization=CovariateCharacterization(
                status="done",
                columns=columns,
            )
        )
        with pytest.raises(ModelRetry, match=message):
            asyncio.run(
                experimental_tools.analyze_experimental_design(
                    SimpleNamespace(deps=deps),
                    column_domains={},
                    coefficients_of_interest=[],
                    units_of_inference={},
                    batch_columns=[batch_column],
                )
            )

    characterization = CovariateCharacterization(status="done")
    observed = example(CellQcProfileEvidence)
    failed_deps = ExperimentalContextDependencies(
        cellSelection=ArtifactRef(
            scope="datastore", kind="cell_selection", artifact_id="c" * 64
        ),
        characterization=characterization,
        qcProfiles={observed.profileId: observed},
        htoIdentityColumns=["hto_identity"],
    )
    failed = experimental_validation.failed_experimental_context_result(
        failed_deps,
        error=UnexpectedModelBehavior("design output failed"),
        model_name="test-model",
    )
    assert failed.status == "failed"
    assert failed.characterization is characterization
    assert failed.cellQc.profileId == ""
    assert failed.qcProfiles == [observed]
    assert failed.decision.batchCorrection.action == "needsInput"
    assert failed.runInfo.agentName == "experimental_context_failed"


@pytest.mark.parametrize("validation_failure", [False, True])
def test_context_failure_preserves_actionable_cause_without_argument_payload(
    validation_failure: bool,
) -> None:
    deps = ExperimentalContextDependencies(
        cellSelection=ArtifactRef(
            scope="datastore", kind="cell_selection", artifact_id="c" * 64
        )
    )
    try:
        if validation_failure:
            CovariateProposal(
                response="tissue",
                explanatoryColumns=["tissue", "condition"],
                observationUnit="sample",
                independentUnit="donor",
                rationale="PRIVATE STUDY TEXT MUST NOT APPEAR IN ERROR NOTES",
            )
        else:
            raise ModelRetry("Unknown batch column 'missing_batch'")
    except (ValidationError, ModelRetry) as cause:
        error = UnexpectedModelBehavior("Design tool retry limit reached")
        error.__cause__ = cause
    result = experimental_validation.failed_experimental_context_result(
        deps, error=error, model_name="test-model"
    )
    notes = " ".join(result.notes)
    assert result.status == "failed"
    assert result.decision.batchCorrection.action == "needsInput"
    assert "Design tool retry limit reached" in notes
    if validation_failure:
        assert "Invalid tool arguments" in notes
        assert "tissue" in notes
    else:
        assert "Unknown batch column 'missing_batch'" in notes
    assert "PRIVATE STUDY TEXT" not in notes


def test_agent_execution_logs_nested_failures_for_sync_and_async_runners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingAgent:
        async def __aenter__(self) -> "FailingAgent":
            return self

        async def __aexit__(self, *_args: object) -> bool:
            return False

        async def run(self, *_args: object, **_kwargs: object) -> object:
            try:
                raise ValueError("inner failure")
            except ValueError as cause:
                raise RuntimeError("outer failure") from cause

    monkeypatch.setattr(
        agent_exec_module,
        "_build_agent",
        lambda **_kwargs: FailingAgent(),
    )
    messages: list[str] = []
    monkeypatch.setattr(agent_exec_module.logger, "error", messages.append)
    with pytest.raises(RuntimeError, match="outer failure"):
        agent_exec_module.run_agent_sync(
            model=object(),
            output_type=dict,
            system_prompt="system",
            user_prompt="user",
            name="sync-failure",
        )
    with pytest.raises(RuntimeError, match="outer failure"):
        asyncio.run(
            agent_exec_module.run_agent(
                model=object(),
                output_type=dict,
                system_prompt="system",
                user_prompt="user",
                name="async-failure",
            )
        )
    assert all("caused by ValueError: inner failure" in message for message in messages)
    assert "sync-failure" in messages[0]
    assert "async-failure" in messages[1]

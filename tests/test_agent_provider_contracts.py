"""Provider schemas expose choices while exact evidence remains programmatic."""

from typing import Any

import pytest

from scarf.agent.biological_interpretation.contracts import (
    BiologicalInterpretationReport,
)
from scarf.agent.data_enrichment.contracts import (
    DataEnrichmentContext,
    DataEnrichmentReport,
    FeatureSelectionPolicy,
    StudyContextSummary,
)
from scarf.agent.data_enrichment.validation import _ground_study_context_summary
from scarf.agent.parameter_tuning.contracts import (
    FinalGraphSelection,
    ParameterTuningReport,
)
from tests.agent_examples import example


@pytest.mark.parametrize(
    ("model", "derived", "authored"),
    [
        (
            DataEnrichmentReport,
            {"inspections", "evidenceIds", "runInfo", "toolCalls"},
            {"policies", "status"},
        ),
        (
            FeatureSelectionPolicy,
            {"assayType", "graphEligible", "organismName", "exactTagFeatures"},
            {"assay", "species", "excludeFamilies", "rationale"},
        ),
        (
            ParameterTuningReport,
            {"evaluations", "selectedArtifacts", "runInfo", "finalSelection"},
            {"recommendedCandidateId", "rationale", "comparisons", "assayReports"},
        ),
        (
            FinalGraphSelection,
            {"graphMethod", "nativeAssay", "integrationId", "runInfo"},
            {"selectedOptionId", "rationale", "comparisons"},
        ),
        (
            BiologicalInterpretationReport,
            {"clusterArtifact", "markerArtifact", "runInfo"},
            {"clusterInterpretations", "treatmentObservations", "status"},
        ),
    ],
)
def test_derived_fields_are_not_requested_from_models(
    model: Any, derived: set[str], authored: set[str]
) -> None:
    schema = model.model_json_schema()
    if "$ref" in schema:
        schema = schema["$defs"][schema["$ref"].rsplit("/", 1)[-1]]
    properties = set(schema["properties"])
    assert not properties.intersection(derived)
    assert authored <= properties
    payload = example(model).model_dump(mode="json")
    assert derived <= payload.keys()
    assert model.model_validate(payload).model_dump(mode="json") == payload


def test_bounded_model_excerpts_preserve_long_caller_context_and_references() -> None:
    references = [
        (f"Experiment {index}: " + "method " * 40).strip() for index in range(14)
    ]
    context = DataEnrichmentContext(
        studyContext="Human lung study with repeated donors and objective-specific observations.",
        studyObjective="Compare the complete design, including all supplied experiments.",
        experimentalDetails=references,
    )
    summary = _ground_study_context_summary(
        context,
        StudyContextSummary(analysisIntentReferences=["Compare the complete design"]),
    )
    assert summary.studyContext == context.studyContext
    assert summary.studyObjective == context.studyObjective
    assert summary.experimentalReferences == references
    assert _ground_study_context_summary(context, summary) == summary

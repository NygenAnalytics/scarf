"""RNA assessments preserve scientific checks for visual and text-only models."""

import json
import hashlib
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from scarf.agent.config.agent_exec import ImageEvidence, ImageInputUnsupportedError
from scarf.agent.experimental_context.study import StudyContract
from scarf.agent.orchestrator import rna_tuning, tuning
from scarf.agent.orchestrator.models import (
    AutomatedPreprocessingPlan,
    AutomatedWorkflowConfig,
    PreprocessedAssayHandoff,
)
from scarf.agent.parameter_tuning.contracts import ParameterCandidateEvaluation
from tests.agent_examples import example
from tests.agent_comparison_examples import comparison_review
from tests.test_agent_rna_adaptive import checkpoints as memory_checkpoints  # noqa: F401


pytestmark = pytest.mark.usefixtures("memory_checkpoints")


def comparison_coverage(run: rna_tuning.RnaTuningRun, scope: str) -> dict[str, Any]:
    """A completed sensitivity fixture with the exact candidate under assessment."""
    panel = comparison_review(scope)
    coverage = panel["comparisonCoverage"]
    selected = run.evaluations[scope][0]
    selected_id = selected.candidateId
    setting = run.settings[selected_id].model_dump(mode="json")
    baseline = {
        **setting,
        "scope": scope,
        "status": selected.status,
        "cellSelection": selected.cellSelection.model_dump(mode="json"),
        "metrics": selected.metrics.model_dump(mode="json"),
    }
    template_baseline = coverage["candidateSettings"]["baseline"]
    settings = {}
    for identifier, template in coverage["candidateSettings"].items():
        row = deepcopy(baseline)
        for field in ("hvgCount", "ranking", "rankingColumn", "eligibleFeatures"):
            if template[field] != template_baseline[field]:
                row[field] = deepcopy(template[field])
        if row["hvgCount"] != baseline["hvgCount"]:
            row["features"]["artifactId"] = hashlib.sha256(
                identifier.encode()
            ).hexdigest()
        for field, value in template["parameters"].items():
            if value != template_baseline["parameters"][field]:
                row["parameters"][field] = value
        settings[selected_id if identifier == "baseline" else identifier] = row
    coverage["candidateSettings"] = settings
    coverage["combinedCandidateId"] = selected_id
    coverage["resolutionCandidateIds"] = [
        selected_id if identifier == "baseline" else identifier
        for identifier in coverage["resolutionCandidateIds"]
    ]
    for row in coverage["comparisons"]:
        row["baselineCandidateId"] = selected_id
        if "observedProof" in row:
            row["observedProof"]["baselineFeatures"] = deepcopy(baseline["features"])
    for candidate in run.evaluations[scope][1:]:
        settings[candidate.candidateId] = {
            **run.settings[candidate.candidateId].model_dump(mode="json"),
            "scope": scope,
            "status": candidate.status,
            "cellSelection": candidate.cellSelection.model_dump(mode="json"),
            "metrics": candidate.metrics.model_dump(mode="json"),
        }
        coverage["fullRepair"] = {
            "baselineCandidateId": selected_id,
            "selectedCandidateId": candidate.candidateId,
        }
    return coverage


def comparison_conclusions(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    coverage = evidence["comparisonCoverage"]
    by_axis: dict[str, set[str]] = {}
    for row in coverage["comparisons"]:
        ids = by_axis.setdefault(row["axis"], set())
        ids.add(row["baselineCandidateId"])
        if row["alternativeCandidateId"] is not None:
            ids.add(row["alternativeCandidateId"])
    return [
        {
            "axis": axis,
            "candidateIds": sorted(ids),
            "preferredCandidateId": evidence["currentCandidateId"],
            "quantitativeReason": "The supplied fixture metrics support retaining the observed baseline.",
            "biologicalReason": "The supplied marker program remains represented.",
            "plainLanguageSummary": "The observed alternatives do not justify changing this setting.",
        }
        for axis, ids in by_axis.items()
    ]


def make_run(
    monkeypatch: pytest.MonkeyPatch, model: Any
) -> tuple[rna_tuning.RnaTuningRun, ParameterCandidateEvaluation]:
    handoff = example(PreprocessedAssayHandoff)
    handoff.graphFeatureCandidates = {"eligibleDefault": handoff.graphFeatures}
    run = rna_tuning.RnaTuningRun(
        SimpleNamespace(model=model),
        SimpleNamespace(),
        SimpleNamespace(workflowRunId="workflow"),
        SimpleNamespace(config=AutomatedWorkflowConfig()),
        example(AutomatedPreprocessingPlan),
        handoff,
        StudyContract.get_blank(),
        {},
        {"scientificInputs": "frozen"},
    )
    selected = example(ParameterCandidateEvaluation)
    selected.parameters.useHarmony = False
    selected.parameters.dimensions = 20
    selected.parameters.neighborsK = 11
    selected.parameters.leidenResolution = 1.0
    selected.metrics.nClusters = 2
    selected.metrics.topMarkerGenes = {
        "0": ["NKG7", "GNLY"],
        "1": ["MS4A1", "CD79A"],
    }
    for field in (
        "seedStability",
        "subsampleStability",
        "markerCoherence",
        "membershipStrengthMean",
        "clusterConnectivity",
    ):
        setattr(selected.metrics, field, 0.9)
    run.evaluations["full"] = [selected]
    run.settings[selected.candidateId] = run.baseline().model_copy(
        update={"parameters": selected.parameters}
    )
    run.store = SimpleNamespace(
        inspect_artifact=lambda _ref: SimpleNamespace(
            exists=True,
            complete=True,
            inputs={"cell_selection": run.cells.to_dict()},
        ),
        load_artifact=lambda _ref: {"values": np.repeat([0, 1], 50)},
    )
    monkeypatch.setattr(
        rna_tuning,
        "uniform_screening_selection",
        lambda _store, cells, **_kwargs: cells,
    )
    monkeypatch.setattr(
        run,
        "comparison_coverage",
        lambda scope, _cells: comparison_coverage(run, scope),
    )
    monkeypatch.setattr(run, "_feature_experiments", lambda _setting: {})
    monkeypatch.setattr(
        run,
        "feature_evidence",
        lambda _: {"topSelectedGenes": ["NKG7", "GNLY"]},
    )
    monkeypatch.setattr(
        rna_tuning,
        "population_support_evidence",
        lambda _store, item, columns: {
            "candidateId": item.candidateId,
            "columns": list(columns),
        },
    )
    monkeypatch.setattr(
        tuning,
        "_analysis_visual_content",
        lambda *args, **kwargs: [
            ImageEvidence(identifier="observed-plot", data=b"image")
        ],
    )
    return run, selected


def assess(**kwargs: Any) -> Any:
    prompt = kwargs["user_prompt"]
    evidence = json.loads(prompt if isinstance(prompt, str) else prompt[0])
    selected = evidence["currentCandidateId"]
    action = rna_tuning.TuningAction(
        action="accept",
        selectedCandidateId=selected,
        correctionNeed="notApplicable",
        comparisonConclusions=comparison_conclusions(evidence),
        plainLanguageSummary="The observed settings preserve the reported cytotoxic program.",
        evidenceIds=[
            f"candidate:{selected}",
            "featureEvidence",
            *evidence["imageHashes"],
        ],
        quantitativeFindings=["The supplied stability and marker metrics are 0.9."],
        qualitativeFindings=["NKG7 and GNLY support a coherent cytotoxic program."],
        objectivePreservation="Preserve the observed cytotoxic population.",
        rationale="The supplied diagnostics support the selected partition.",
    )
    return SimpleNamespace(output=kwargs["output_validator"](action))


@pytest.mark.parametrize(
    "model",
    [
        SimpleNamespace(supports_image_input=False),
        SimpleNamespace(profile={"supports_image_input": False}),
    ],
)
def test_declared_text_only_model_gets_structured_biological_evidence(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest, model: Any
) -> None:
    saved = request.getfixturevalue("memory_checkpoints")
    run, selected = make_run(monkeypatch, model)
    monkeypatch.setattr(
        tuning,
        "_analysis_visual_content",
        lambda *a, **kw: pytest.fail("A text-only assessment must not render images"),
    )

    def text_assessment(**kwargs: Any) -> Any:
        assert isinstance(kwargs["user_prompt"], str)
        evidence = json.loads(kwargs["user_prompt"])
        assert evidence["evidenceMode"] == "structured"
        assert evidence["visualInspection"] == "unavailable"
        assert evidence["imageHashes"] == {}
        assert evidence["featureEvidence"][selected.candidateId][
            "topSelectedGenes"
        ] == ["NKG7", "GNLY"]
        assert "no images were supplied" in kwargs["system_prompt"]
        assert "Do not claim to have seen" in kwargs["system_prompt"]
        assert "structured_evidence" in next(iter(saved))
        return assess(**kwargs)

    monkeypatch.setattr(rna_tuning, "run_agent_sync", text_assessment)
    assert run.review("full", 0, selected, {}).action == "accept"
    history = run.history[-1]
    assert history["evidenceMode"] == "structured"
    assert history["visualInspection"] == "unavailable"
    assert history["imageHashes"] == {}
    assert "observed-plot" not in history["review"]["evidenceIds"]
    assert (
        rna_tuning._STRUCTURED_VISUAL_LIMITATION
        in run.report(None, "Inspect another setting").limitations
    )


@pytest.mark.parametrize(
    "model",
    [
        object(),
        SimpleNamespace(profile={"supports_image_output": False}),
        SimpleNamespace(supports_image_input=True),
    ],
)
def test_unknown_or_positive_input_capability_retains_visual_assessment(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest, model: Any
) -> None:
    saved = request.getfixturevalue("memory_checkpoints")
    run, selected = make_run(monkeypatch, model)

    def visual_assessment(**kwargs: Any) -> Any:
        assert not isinstance(kwargs["user_prompt"], str)
        evidence = json.loads(kwargs["user_prompt"][0])
        assert evidence["evidenceMode"] == "visual"
        assert evidence["visualInspection"] == "available"
        assert evidence["imageHashes"]
        return assess(**kwargs)

    monkeypatch.setattr(rna_tuning, "run_agent_sync", visual_assessment)
    assert run.review("full", 0, selected, {}).action == "accept"
    assert "parameter_tuning/structured_evidence" not in saved


def test_explicit_image_rejection_is_saved_before_retry_and_reused(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    saved = request.getfixturevalue("memory_checkpoints")
    run, selected = make_run(monkeypatch, object())
    modes = []
    renders = []

    def render(*args: Any, **kwargs: Any) -> list[ImageEvidence]:
        renders.append(True)
        return [ImageEvidence(identifier="observed-plot", data=b"image")]

    def rejecting_assessment(**kwargs: Any) -> Any:
        prompt = kwargs["user_prompt"]
        modes.append("structured" if isinstance(prompt, str) else "visual")
        if modes[-1] == "visual":
            raise ImageInputUnsupportedError("Image input is not supported")
        assert saved["parameter_tuning/structured_evidence"]["outputs"] == {
            "evidenceMode": "structured",
            "reason": "providerRejectedImageInput",
        }
        return assess(**kwargs)

    monkeypatch.setattr(tuning, "_analysis_visual_content", render)
    monkeypatch.setattr(rna_tuning, "run_agent_sync", rejecting_assessment)
    assert run.review("full", 0, selected, {}).action == "accept"
    assert run.review("full", 1, selected, {}).action == "accept"
    assert modes == ["visual", "structured", "structured"]
    assert len(renders) == 1
    assert len(run.history) == 2
    assert all(row["evidenceMode"] == "structured" for row in run.history)


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("Interrupted during structured assessment"),
        ModelHTTPError(429, "test-model", "Too Many Requests"),
    ],
    ids=["interrupted", "rate-limited"],
)
def test_interrupted_structured_retry_does_not_probe_images_on_resume(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    failure: Exception,
) -> None:
    saved = request.getfixturevalue("memory_checkpoints")
    run, selected = make_run(monkeypatch, object())

    def interrupt(**kwargs: Any) -> Any:
        if not isinstance(kwargs["user_prompt"], str):
            raise ImageInputUnsupportedError("Image input is not supported")
        raise failure

    monkeypatch.setattr(rna_tuning, "run_agent_sync", interrupt)
    with pytest.raises(type(failure)) as caught:
        run.review("full", 0, selected, {})
    assert caught.value is failure
    assert set(saved) == {
        "parameter_tuning/structured_evidence",
        "parameter_tuning/full/review0/evidence/visual",
        "parameter_tuning/full/review0/evidence/structured",
    }
    original = deepcopy(saved)
    resumed, selected = make_run(monkeypatch, object())
    monkeypatch.setattr(
        tuning,
        "_analysis_visual_content",
        lambda *a, **kw: pytest.fail("Resume must preserve the rejected capability"),
    )
    for name in (
        "partition_comparison_evidence",
        "population_support_evidence",
        "_neighbor_overlap",
    ):
        monkeypatch.setattr(
            rna_tuning,
            name,
            lambda *a, **kw: pytest.fail(
                "Output recovery must reuse saved diagnostics"
            ),
        )
    monkeypatch.setattr(
        resumed,
        "feature_evidence",
        lambda *a, **kw: pytest.fail("Output recovery must reuse saved gene evidence"),
    )

    def recover(**kwargs):
        evidence = original["parameter_tuning/full/review0/evidence/structured"][
            "outputs"
        ]["evidence"]
        assert json.loads(kwargs["user_prompt"]) == evidence
        assert kwargs["user_prompt"] == json.dumps(
            evidence, sort_keys=True, separators=(",", ":")
        )
        return assess(**kwargs)

    monkeypatch.setattr(rna_tuning, "run_agent_sync", recover)
    assert resumed.review("full", 0, selected, {}).action == "accept"
    assert resumed.history[-1]["evidenceMode"] == "structured"
    assert {key: saved[key] for key in original} == original


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("Provider timed out"),
        ModelHTTPError(401, "test-model", {"message": "Unauthorized"}),
    ],
)
def test_unrelated_model_failure_never_changes_evidence_mode(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest, error: Exception
) -> None:
    saved = request.getfixturevalue("memory_checkpoints")
    run, selected = make_run(monkeypatch, object())

    def fail(**kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(rna_tuning, "run_agent_sync", fail)
    with pytest.raises(type(error)) as caught:
        run.review("full", 0, selected, {})
    assert caught.value is error
    assert set(saved) == {"parameter_tuning/full/review0/evidence/visual"}


def test_committed_visual_review_replays_without_images_or_neighbor_diagnostics(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    saved = request.getfixturevalue("memory_checkpoints")
    run, selected = make_run(monkeypatch, object())
    prototype = next(iter(selected.artifacts.values()))
    selected.artifacts["neighbors"] = prototype.model_copy(
        update={"kind": "neighbors", "artifactId": "e" * 64}
    )
    alternate = selected.model_copy(deep=True)
    alternate.candidateId = "alternative"
    alternate.parameters.candidateId = alternate.candidateId
    alternate.artifacts["neighbors"] = selected.artifacts["neighbors"].model_copy(
        update={"artifactId": "f" * 64}
    )
    run.evaluations["full"].append(alternate)
    run.settings[alternate.candidateId] = run.settings[selected.candidateId].model_copy(
        update={"parameters": alternate.parameters}
    )
    monkeypatch.setattr(rna_tuning, "_neighbor_overlap", lambda *a: 0.8)
    monkeypatch.setattr(rna_tuning, "run_agent_sync", assess)
    run.review("full", 0, selected, {})
    original = run.history[-1]
    assert (
        saved["parameter_tuning/full/review0"]["inputs"]["neighborComparisons"][0][
            "meanNeighborJaccard"
        ]
        == 0.8
    )
    saved["parameter_tuning/structured_evidence"] = {
        "inputs": {**run.provenance, "configuredImageInput": None},
        "outputs": {
            "evidenceMode": "structured",
            "reason": "providerRejectedImageInput",
        },
    }
    resumed, resumed_selected = make_run(monkeypatch, object())
    resumed.evaluations = run.evaluations
    resumed.settings = run.settings

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Committed review must precede image, neighbor and model work")

    monkeypatch.setattr(tuning, "_analysis_visual_content", unexpected)
    monkeypatch.setattr(rna_tuning, "_neighbor_overlap", unexpected)
    monkeypatch.setattr(rna_tuning, "run_agent_sync", unexpected)
    resumed.review("full", 0, resumed_selected, {})
    assert resumed.history[-1] == original
    resumed_selected.metrics.seedStability = 0.5
    resumed.evaluations["full"][0] = resumed_selected
    with pytest.raises(ValueError, match="different candidate evidence"):
        resumed.review("full", 0, resumed_selected, {})


@pytest.mark.parametrize("invalid", ["inventedImage", "missingStability"])
def test_structured_assessment_keeps_evidence_and_scientific_checks(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    invalid: str,
) -> None:
    saved = request.getfixturevalue("memory_checkpoints")
    run, selected = make_run(monkeypatch, SimpleNamespace(supports_image_input=False))
    if invalid == "missingStability":
        selected.metrics.subsampleStability = None

    def invalid_assessment(**kwargs: Any) -> Any:
        action = assess(**{**kwargs, "output_validator": lambda action: action}).output
        if invalid == "inventedImage":
            action.evidenceIds.append("observed-plot")
        return SimpleNamespace(output=kwargs["output_validator"](action))

    monkeypatch.setattr(rna_tuning, "run_agent_sync", invalid_assessment)
    with pytest.raises(
        ValueError,
        match="unknown evidence"
        if invalid == "inventedImage"
        else "evidence is missing",
    ):
        run.review("full", 0, selected, {})
    assert "parameter_tuning/full/review0" not in saved


@pytest.mark.parametrize("text_only", [False, True])
def test_assessment_accepts_the_detailed_evidence_it_supplies(
    monkeypatch: pytest.MonkeyPatch, text_only: bool
) -> None:
    run, selected = make_run(
        monkeypatch, SimpleNamespace(supports_image_input=not text_only)
    )
    metric = f"candidate:{selected.candidateId}:seedStability"
    selected.evidenceIds = [metric, metric]
    run.study.evidenceIds = ["study:observed", "shared:observed"]
    run.plan.cellQc.evidenceIds = ["qc:observed", "shared:observed"]
    citations = [metric, "study:observed", "qc:observed", "shared:observed"]

    def cite_supplied_evidence(**kwargs: Any) -> Any:
        prompt = kwargs["user_prompt"]
        evidence = json.loads(prompt if isinstance(prompt, str) else prompt[0])
        catalogue = evidence["availableEvidenceIds"]
        assert len(catalogue) == len(set(catalogue))
        assert set(citations).issubset(catalogue)
        assert set(evidence["candidates"][0]["evidenceIds"]).issubset(catalogue)
        action = assess(**{**kwargs, "output_validator": lambda action: action}).output
        action.evidenceIds = [*citations, *evidence["imageHashes"]]
        return SimpleNamespace(output=kwargs["output_validator"](action))

    monkeypatch.setattr(rna_tuning, "run_agent_sync", cite_supplied_evidence)
    action = run.review("full", 0, selected, {})
    assert action.action == "accept"
    assert metric in action.evidenceIds
    assert f"candidate:{selected.candidateId}" not in action.evidenceIds


@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        ("inventedMetric", "unknown evidence"),
        ("otherCandidateOnly", "selected numerical evidence"),
        ("missingImage", "visual evidence"),
    ],
)
def test_detailed_citations_keep_grounding_and_visual_requirements(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    invalid: str,
    message: str,
) -> None:
    saved = request.getfixturevalue("memory_checkpoints")
    run, selected = make_run(monkeypatch, object())
    metric = f"candidate:{selected.candidateId}:seedStability"
    selected.evidenceIds = [metric]
    alternative = selected.model_copy(deep=True)
    alternative.candidateId = "alternative"
    alternative.parameters.candidateId = alternative.candidateId
    alternative.evidenceIds = ["candidate:alternative:seedStability"]
    run.evaluations["full"].append(alternative)
    run.settings[alternative.candidateId] = run.settings[
        selected.candidateId
    ].model_copy(update={"parameters": alternative.parameters})

    def invalid_assessment(**kwargs: Any) -> Any:
        action = assess(**{**kwargs, "output_validator": lambda action: action}).output
        action.evidenceIds = [metric, "observed-plot"]
        if invalid == "inventedMetric":
            action.evidenceIds.append(
                f"candidate:{selected.candidateId}:inventedMetric"
            )
        elif invalid == "otherCandidateOnly":
            action.evidenceIds = [*alternative.evidenceIds, "observed-plot"]
        else:
            action.evidenceIds = [metric]
        return SimpleNamespace(output=kwargs["output_validator"](action))

    monkeypatch.setattr(rna_tuning, "run_agent_sync", invalid_assessment)
    with pytest.raises(ValueError, match=message):
        run.review("full", 0, selected, {})
    assert "parameter_tuning/full/review0" not in saved


@pytest.mark.parametrize("decision", ["accept", "defer"])
@pytest.mark.parametrize("legacy_catalogue", [False, True])
def test_committed_review_preserves_its_exact_evidence_catalogue(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    legacy_catalogue: bool,
    decision: str,
) -> None:
    saved = request.getfixturevalue("memory_checkpoints")
    run, selected = make_run(monkeypatch, object())
    metric = f"candidate:{selected.candidateId}:seedStability"
    selected.evidenceIds = [metric]
    run.study.evidenceIds = ["study:observed"]
    run.plan.cellQc.evidenceIds = ["qc:observed"]

    def decide(**kwargs: Any) -> Any:
        action = assess(**{**kwargs, "output_validator": lambda action: action}).output
        action.action = decision
        return SimpleNamespace(output=kwargs["output_validator"](action))

    monkeypatch.setattr(rna_tuning, "run_agent_sync", decide)
    expected = run.review("full", 0, selected, {})
    key = "parameter_tuning/full/review0"
    if legacy_catalogue:
        # Model an existing committed review from before detailed IDs were listed.
        saved[key]["inputs"]["availableEvidenceIds"] = [
            f"candidate:{selected.candidateId}",
            "observed-plot",
            "studyContract",
            "qcPolicy",
            "samplingCoverage",
            "featureEvidence",
            "neighborComparisons",
        ]
    before = json.dumps(saved[key], sort_keys=True)
    resumed, _ = make_run(monkeypatch, object())
    resumed.evaluations = run.evaluations
    resumed.settings = run.settings
    resumed.study = run.study
    resumed.plan = run.plan

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Committed evidence must replay without model or plot work")

    monkeypatch.setattr(rna_tuning, "run_agent_sync", unexpected)
    monkeypatch.setattr(tuning, "_analysis_visual_content", unexpected)
    assert resumed.review("full", 0, selected, {}) == expected
    assert json.dumps(saved[key], sort_keys=True) == before


def test_real_agent_retry_repairs_a_citation_from_actionable_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = 0
    feedback: list[str] = []

    async def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal requests
        requests += 1
        user_part = next(
            part
            for message in messages
            for part in message.parts
            if isinstance(part, UserPromptPart)
        )
        assert isinstance(user_part.content, str)
        evidence = json.loads(user_part.content)
        selected_id = evidence["currentCandidateId"]
        metric = evidence["candidates"][0]["evidenceIds"][0]
        invalid_id = f"candidate:{selected_id}:inventedMetric"
        if requests == 2:
            feedback.extend(
                str(part.content)
                for message in messages
                for part in message.parts
                if isinstance(part, RetryPromptPart)
            )
            assert any(invalid_id in text for text in feedback)
            assert any("availableEvidenceIds" in text for text in feedback)
            assert any(f"candidate:{selected_id}" in text for text in feedback)
        action = rna_tuning.TuningAction(
            action="accept",
            selectedCandidateId=selected_id,
            correctionNeed="notApplicable",
            comparisonConclusions=comparison_conclusions(evidence),
            plainLanguageSummary="The observed settings preserve the reported cytotoxic program.",
            evidenceIds=[invalid_id if requests == 1 else metric],
            quantitativeFindings=["The supplied seed stability is 0.9."],
            qualitativeFindings=["NKG7 and GNLY support a cytotoxic program."],
            objectivePreservation="Preserve the observed cytotoxic population.",
            rationale="The observed diagnostics support this partition.",
        )
        assert info.output_tools
        return ModelResponse(
            parts=[
                ToolCallPart(info.output_tools[0].name, action.model_dump(mode="json"))
            ]
        )

    model = FunctionModel(reply, profile={"supports_image_input": False})
    run, selected = make_run(monkeypatch, model)
    selected.evidenceIds = [f"candidate:{selected.candidateId}:seedStability"]
    action = run.review("full", 0, selected, {})
    assert action.action == "accept"
    assert action.evidenceIds == selected.evidenceIds
    assert requests == 2


@pytest.mark.parametrize(
    "provider_failure", [False, True], ids=["deferred", "rate-limited"]
)
def test_saved_scientific_defer_replays_completed_candidates_without_new_work(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    provider_failure: bool,
) -> None:
    saved = request.getfixturevalue("memory_checkpoints")
    run, prototype = make_run(monkeypatch, object())
    run.handoff.nCells = 100
    run.evaluations["full"] = []
    run.settings = {}
    monkeypatch.setattr(
        run,
        "comparison_coverage",
        rna_tuning.RnaTuningRun.comparison_coverage.__get__(run),
    )
    run.handoff.graphFeatureCandidates["eligibleAll"] = (
        run.handoff.graphFeatureCandidates["eligibleDefault"]
    )

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("A committed defer cannot trigger new model or scientific work")

    run.store = SimpleNamespace(
        inspect_artifact=lambda _: SimpleNamespace(
            exists=True, complete=True, inputs={"cell_selection": run.cells.to_dict()}
        ),
        load_artifact=lambda ref: {
            "values": np.ones(1000, dtype=bool)
            if ref.kind == "feature_selection"
            else np.repeat([0, 1], 50)
        },
        get_assay=lambda _: SimpleNamespace(
            feats=SimpleNamespace(
                fetch_all=lambda _: np.asarray([f"G{i}" for i in range(1000)])
            )
        ),
        run_normalization=unexpected,
    )
    monkeypatch.setattr(
        rna_tuning,
        "screening_coverage",
        lambda *args, **kwargs: ({"screeningCells": 100}, []),
    )
    baseline = run.baseline()
    baseline_identity = rna_tuning.candidate_identity(
        run.execution_inputs(run.cells, baseline)
    )
    prepared_baseline = baseline.model_copy(
        update={
            "parameters": baseline.parameters.model_copy(
                update={"candidateId": f"rna_{baseline_identity[:24]}"}
            )
        }
    )
    for count in (2000, 4000):
        saved[f"parameter_tuning/sample0/sensitivity/hvgCount:{count}/setting"] = {
            "inputs": {
                "baseline": prepared_baseline.model_dump(mode="json"),
                "experiment": {"parameter": "hvgCount", "value": count},
                "cells": run.cells.to_dict(),
            },
            "outputs": {"setting": prepared_baseline.model_dump(mode="json")},
        }
    settings = [run.baseline(resolution) for resolution in (0.5, 0.75, 1.0, 1.25)]
    settings += [
        baseline.model_copy(
            update={"parameters": baseline.parameters.model_copy(update={field: value})}
        )
        for field, values in (("dimensions", (10, 30)), ("neighborsK", (21, 41)))
        for value in values
    ]
    for setting in settings:
        inputs = run.execution_inputs(run.cells, setting)
        candidate_id = f"rna_{rna_tuning.candidate_identity(inputs)[:24]}"
        candidate = prototype.model_copy(deep=True)
        candidate.candidateId = candidate_id
        candidate.parameters = setting.parameters.model_copy(
            update={"candidateId": candidate_id}
        )
        candidate.evidenceIds = [f"candidate:{candidate_id}:seedStability"]
        admission = run.budget.admit("sample0", inputs)
        run.budget.complete(
            admission, {"evaluation": candidate.model_dump(mode="json")}
        )

    def defer(**kwargs: Any) -> Any:
        action = assess(**{**kwargs, "output_validator": lambda action: action}).output
        action.action = "defer"
        action.rationale = (
            "Independent evidence is needed before accepting this partition."
        )
        return SimpleNamespace(output=kwargs["output_validator"](action))

    def unavailable(**kwargs):
        raise ModelHTTPError(429, "test-model", "Too Many Requests")

    monkeypatch.setattr(
        rna_tuning, "run_agent_sync", unavailable if provider_failure else defer
    )
    if provider_failure:
        with pytest.raises(ModelHTTPError, match="429"):
            run.run()
    else:
        first_report, _ = run.run()
        assert first_report.status == "needsInput"
    first_budget = run.budget.summary()
    before = json.dumps(saved, sort_keys=True)

    resumed, _ = make_run(monkeypatch, object())
    resumed.handoff = run.handoff
    resumed.store = run.store
    resumed.evaluations["full"] = []
    resumed.settings = {}
    monkeypatch.setattr(
        resumed,
        "comparison_coverage",
        rna_tuning.RnaTuningRun.comparison_coverage.__get__(resumed),
    )
    monkeypatch.setattr(
        rna_tuning, "run_agent_sync", defer if provider_failure else unexpected
    )
    monkeypatch.setattr(tuning, "_analysis_visual_content", unexpected)
    resumed_report, resumed_summary = resumed.run()
    if not provider_failure:
        assert resumed_report == first_report
    assert resumed_report.status == "needsInput"
    assert resumed_summary["budget"] == first_budget
    original = json.loads(before)
    assert {key: saved[key] for key in original} == original
    appended = {key: value for key, value in saved.items() if key not in original}
    assert len(appended) == (3 if provider_failure else 2)
    assert sum("/diagnostic_attempts/" in key for key in appended) == 2
    if provider_failure:
        assert (
            saved["parameter_tuning/sample0/review0"]["outputs"]["action"]["action"]
            == "defer"
        )
    counts = resumed_summary["diagnosticOperations"]["operations"]
    assert all(row["attempted"] == 0 for row in counts.values())
    assert counts["diagnostic.primaryCandidateEvidence"]["restored"] == len(settings)
    assert counts["diagnostic.reviewEvidence"]["restored"] == 1


def test_model_failure_is_not_reported_as_a_scientific_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, _ = make_run(monkeypatch, SimpleNamespace(supports_image_input=False))
    failure = UnexpectedModelBehavior("Exceeded maximum output retries")

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise failure

    monkeypatch.setattr(run, "assess_scope", fail)
    with pytest.raises(UnexpectedModelBehavior) as caught:
        run.run()
    assert caught.value is failure


def test_exhausted_scientific_work_stays_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, _ = make_run(monkeypatch, SimpleNamespace(supports_image_input=False))

    def exhaust(*args: Any, **kwargs: Any) -> Any:
        raise rna_tuning.CandidateBudgetExceeded("Screening candidate limit reached")

    monkeypatch.setattr(run, "assess_scope", exhaust)
    report, _ = run.run()
    assert report.status == "needsInput"
    assert report.needsInput is not None
    assert "Screening candidate limit reached" in report.needsInput.question


@pytest.mark.parametrize("scope", ["sample0", "full"])
@pytest.mark.parametrize("experiments", [[], ["neighborsK:21", "dimensions:30"]])
def test_assessment_schema_limits_choices_without_changing_saved_fields(
    scope: str, experiments: list[str]
) -> None:
    output_type = rna_tuning._assessment_output_type(
        ["observed_a", "observed_b", "observed_a"], experiments, scope=scope
    )
    properties = output_type.model_json_schema()["properties"]
    assert properties["selectedCandidateId"]["enum"] == ["observed_a", "observed_b"]
    assert ("enlarge" in properties["action"]["enum"]) is (scope != "full")
    assert ("experiment" in properties["action"]["enum"]) is bool(experiments)
    if experiments:
        assert properties["experimentId"]["anyOf"][0]["enum"] == experiments
    else:
        assert properties["experimentId"]["type"] == "null"
    assert (
        output_type.model_fields.keys() == rna_tuning.TuningAction.model_fields.keys()
    )
    saved_action = rna_tuning.TuningAction(
        action="defer",
        selectedCandidateId="observed_a",
        correctionNeed="notApplicable",
        comparisonConclusions=[],
        plainLanguageSummary="Independent evidence is still needed.",
        evidenceIds=["candidate:observed_a"],
        quantitativeFindings=["Observed stability needs further assessment."],
        qualitativeFindings=["Marker support remains unresolved."],
        objectivePreservation="Preserve marker-supported populations.",
        rationale="Defer pending essential evidence.",
    )
    assert (
        output_type.model_validate(saved_action.model_dump()).model_dump()
        == saved_action.model_dump()
    )
    if scope == "full":
        with pytest.raises(ValidationError, match="accept.*defer"):
            output_type.model_validate(
                {**saved_action.model_dump(), "action": "enlarge"}
            )


@pytest.mark.parametrize("invalid_field", ["selectedCandidateId", "experimentId"])
def test_real_agent_retries_choices_against_the_current_output_schema(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    invalid_field: str,
) -> None:
    requests = 0
    feedback: list[str] = []
    chosen_experiment = ""

    async def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal requests, chosen_experiment
        requests += 1
        user_part = next(
            part
            for message in messages
            for part in message.parts
            if isinstance(part, UserPromptPart)
        )
        assert isinstance(user_part.content, str)
        evidence = json.loads(user_part.content)
        selected_id = evidence["currentCandidateId"]
        chosen_experiment = next(iter(evidence["experiments"]))
        assert info.output_tools
        properties = info.output_tools[0].parameters_json_schema["properties"]
        assert properties["selectedCandidateId"]["const"] == selected_id
        assert set(properties["experimentId"]["anyOf"][0]["enum"]) == set(
            evidence["experiments"]
        )
        action = rna_tuning.TuningAction(
            action="experiment",
            selectedCandidateId=selected_id,
            experimentId=chosen_experiment,
            correctionNeed="notApplicable",
            comparisonConclusions=[],
            plainLanguageSummary="Request one observed-evidence-driven comparison.",
            evidenceIds=[f"candidate:{selected_id}"],
            quantitativeFindings=["The supplied stability metric is 0.9."],
            qualitativeFindings=["Reported markers support a cytotoxic program."],
            concern="Assess sensitivity to the offered parameter change.",
            expectedImprovement="Test whether the comparison improves population support.",
            objectivePreservation="Preserve the observed cytotoxic population.",
            rationale=f"Request the offered next comparison {chosen_experiment}.",
        ).model_dump(mode="json")
        if requests == 1:
            action[invalid_field] = "invented_choice"
        else:
            feedback.extend(
                str(part.content)
                for message in messages
                for part in message.parts
                if isinstance(part, RetryPromptPart)
            )
            assert any(invalid_field in text for text in feedback)
            assert any("invented_choice" in text for text in feedback)
            assert any(
                (
                    selected_id
                    if invalid_field == "selectedCandidateId"
                    else chosen_experiment
                )
                in text
                for text in feedback
            )
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, action)])

    model = FunctionModel(reply, profile={"supports_image_input": False})
    run, selected = make_run(monkeypatch, model)
    action = run.review("full", 0, selected, {})
    assert requests == 2
    assert action.action == "experiment"
    assert action.experimentId == chosen_experiment
    saved = request.getfixturevalue("memory_checkpoints")
    assert saved["parameter_tuning/full/review0"]["outputs"]["action"] == (
        rna_tuning.TuningAction.model_validate(action.model_dump()).model_dump(
            mode="json"
        )
    )


def test_invalid_experiment_feedback_names_the_available_choices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, selected = make_run(monkeypatch, object())

    def invalid_assessment(**kwargs: Any) -> Any:
        action = assess(**{**kwargs, "output_validator": lambda action: action}).output
        action.action = "experiment"
        action.experimentId = "invented_choice"
        action.concern = "Assess an observed concern."
        action.expectedImprovement = "Improve the supported representation."
        return SimpleNamespace(output=kwargs["output_validator"](action))

    monkeypatch.setattr(rna_tuning, "run_agent_sync", invalid_assessment)
    with pytest.raises(ValueError, match="Unknown experiment ID") as caught:
        run.review("full", 0, selected, {})
    assert "invented_choice" in str(caught.value)
    assert selected.candidateId in str(caught.value)
    assert next(iter(run.experiments(selected))) in str(caught.value)

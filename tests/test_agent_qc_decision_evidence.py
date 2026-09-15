"""QC choices distinguish reference grouping, measured retention and biology."""

from types import SimpleNamespace
from copy import deepcopy

import pytest

from scarf.agent.experimental_context.contracts import (
    CaptureFailureEvidence,
    CellQcProfileEvidence,
)
from scarf.agent.experimental_context.study import StudyContract
from scarf.agent.orchestrator import AgentOrchestrator
from scarf.agent.orchestrator.preprocessing import PreprocessingStagesMixin
from scarf.agent.decisions.rna import QcGroupingExecutorPayload


def profile(policy: str, retained: int) -> CellQcProfileEvidence:
    return CellQcProfileEvidence.model_validate(
        {
            "action": "skip" if policy == "retainWithFlags" else "registeredMad",
            "registeredProfile": policy,
            "attributes": [] if policy == "retainWithFlags" else ["RNA_percentMito"],
            "captureColumn": "library",
            "sampleColumn": "library" if policy == "captureMad5" else None,
            "activeCells": 100,
            "retainedCells": retained,
            "evidenceId": f"qc:{policy}",
        }
    )


def test_qc_evidence_reports_true_median_fractions_and_metric_flags() -> None:
    selected = profile("globalMad5", 40)
    selected.sampleRetainedCells = {"a": 10, "b": 30}
    selected.retainedCellsByColumn = {"condition": {"a": 10, "b": 30}}
    selected.metricFlaggedCells = {"RNA_percentMito": {"highMito": 60}}
    selected.captureFailureEvidence = [
        CaptureFailureEvidence(
            capture="a", activeCells=50, retainedCells=10, retainedFraction=0.2
        ),
        CaptureFailureEvidence(
            capture="b", activeCells=50, retainedCells=30, retainedFraction=0.6
        ),
    ]
    reference = profile("retainWithFlags", 100)
    reference.retainedCellsByColumn = {"condition": {"a": 50, "b": 50}}
    summary = PreprocessingStagesMixin._profile_evidence(selected, reference).summary
    assert "min/median/max=10/20/30" in summary
    assert "min/median/max=20.0%/40.0%/60.0%" in summary
    assert "a=10/50 (20.0%)" in summary
    assert "b=30/50 (60.0%)" in summary
    assert "'highMito': 60" in summary
    assert "flags may overlap" in summary
    assert (
        "fractions unavailable"
        in PreprocessingStagesMixin._profile_evidence(selected).summary
    )


def test_qc_grouping_compares_the_same_cutoff_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = AgentOrchestrator("test-model")
    profiles = [
        CellQcProfileEvidence(
            action="globalGaussian",
            attributes=["RNA_percentMito"],
            activeCells=100,
            retainedCells=60,
            evidenceId="qc:globalGaussian",
        ),
        CellQcProfileEvidence(
            action="sampleMad",
            sampleColumn="library",
            attributes=["RNA_percentMito"],
            activeCells=100,
            retainedCells=55,
            evidenceId="qc:sampleMad3",
        ),
        profile("globalMad5", 90),
        profile("captureMad5", 95),
        profile("retainWithFlags", 100),
    ]
    seen = {}

    def resolve(_store, _request, definition, bundle, _answers, **kwargs):
        seen["definition"] = definition
        seen["bundle"] = bundle
        seen["qcEvidence"] = kwargs["qc_evidence"]
        return SimpleNamespace(
            compiled=SimpleNamespace(
                executorPayload=QcGroupingExecutorPayload(groupingMode="global")
            ),
            checkpointSha256="a" * 64,
        )

    monkeypatch.setattr(orchestrator, "_resolve_rna_decision", resolve)
    orchestrator._resolve_qc_grouping_decision(
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(qcProfiles=profiles),
        StudyContract.get_blank().model_copy(
            update={"physicalCaptureColumn": "library"}
        ),
        {},
    )
    options = seen["definition"].spec.option_by_id()
    assert "qc:globalMad5" in options["qcGrouping:global"].requiredEvidenceIds
    assert "qc:captureMad5" in options["qcGrouping:physicalCapture"].requiredEvidenceIds
    assert (
        "qc:sampleMad3" not in options["qcGrouping:physicalCapture"].requiredEvidenceIds
    )
    design = next(
        row.summary for row in seen["bundle"].evidence if row.evidenceClass == "design"
    )
    assert "need not be independent biological units or healthy references" in design
    assert "different cutoff methods cannot isolate" in design
    assert len(seen["qcEvidence"]["policies"]) == len(profiles)


def test_qc_handoff_deduplicates_without_losing_any_policy_measurements() -> None:
    policies = [profile("globalMad5", 90), profile("captureMad5", 95)]
    for item in policies:
        item.resolvedBounds = [
            {"group": "a", "metric": "RNA_percentMito", "upper": 7.125}
        ]
        item.activeCellsByCapture = {"a": 100}
        item.parameters = {
            "resolvedBounds": deepcopy(item.resolvedBounds),
            "captureSizes": {"a": 100},
            "captureComparisons": [
                {
                    "capture": "a",
                    "mitoQuantiles": [1.0, 3.1, 7.125],
                    "missingFraction": 0.05,
                }
            ],
        }
        item.retainedCellsByCombination = {
            'joint:["sex","condition"]': {
                "F/treated": 0,
                "M/control": item.retainedCells,
            }
        }
        item.unsafeRetentionGroups = ["The protected joint group F/treated is absent"]
        item.notes = ["A supported reference pool is unavailable"]
        item.captureFailureEvidence = [
            CaptureFailureEvidence(
                capture="a",
                activeCells=100,
                retainedCells=item.retainedCells,
                retainedFraction=item.retainedCells / 100,
                conditionAndUnitSafety=[
                    {
                        "column": "age",
                        "kind": "continuous",
                        "missingFraction": 0.2,
                        "status": "unsupported",
                        "quantilesBeforeExclusion": [20, 50, 80],
                    }
                ],
            )
        ]
    originals = [item.model_dump(mode="json") for item in policies]
    payload = PreprocessingStagesMixin._qc_decision_evidence(policies)
    shared = payload["sharedMeasurements"]
    assert (
        payload["policies"][0]["captureFailureEvidence"][0]["conditionAndUnitSafetyRef"]
        == payload["policies"][1]["captureFailureEvidence"][0][
            "conditionAndUnitSafetyRef"
        ]
    )
    assert (
        payload["policies"][0]["parameters"]["captureComparisonsRef"]
        == payload["policies"][1]["parameters"]["captureComparisonsRef"]
    )
    for saved, compact in zip(originals, payload["policies"], strict=True):
        restored = deepcopy(compact)
        for name in ("metricSources", "sourceConcordance"):
            restored[name] = shared[restored.pop(name + "Ref")]
        parameters = restored["parameters"]
        parameters["captureComparisons"] = shared[
            parameters.pop("captureComparisonsRef")
        ]
        parameters["resolvedBounds"] = deepcopy(restored["resolvedBounds"])
        parameters["captureSizes"] = deepcopy(restored["activeCellsByCapture"])
        for capture in restored["captureFailureEvidence"]:
            capture["conditionAndUnitSafety"] = shared[
                capture.pop("conditionAndUnitSafetyRef")
            ]
        assert restored == saved
    assert [item.model_dump(mode="json") for item in policies] == originals


def test_qc_handoff_preserves_disagreeing_duplicate_thresholds_for_investigation() -> (
    None
):
    selected = profile("globalMad5", 80)
    selected.resolvedBounds = {"RNA_percentMito": [0.0, 7.0]}
    selected.parameters["resolvedBounds"] = {"RNA_percentMito": [0.0, 10.0]}
    payload = PreprocessingStagesMixin._qc_decision_evidence([selected])["policies"][0]
    assert payload["resolvedBounds"] != payload["parameters"]["resolvedBounds"]

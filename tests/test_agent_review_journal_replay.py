"""Interrupted RNA reviews replay against the real Zarr journal."""

from types import SimpleNamespace

import numpy as np
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.agent.config.agent_exec import ImageEvidence
from scarf.agent.experimental_context.study import StudyContract
from scarf.agent.orchestrator import journal, rna_tuning, tuning
from scarf.agent.orchestrator.models import (
    AutomatedPreprocessingPlan,
    AutomatedWorkflowConfig,
    PreprocessedAssayHandoff,
)
from scarf.agent.parameter_tuning.contracts import ParameterCandidateEvaluation
from tests.agent_examples import example
from tests.test_agent_rna_evidence_mode import comparison_coverage


def _review_run(
    monkeypatch: pytest.MonkeyPatch, zw: zarr.Group, *, harmony: bool
) -> tuple[rna_tuning.RnaTuningRun, ParameterCandidateEvaluation]:
    handoff = example(PreprocessedAssayHandoff)
    handoff.graphFeatureCandidates = {"eligibleDefault": handoff.graphFeatures}
    run = rna_tuning.RnaTuningRun(
        SimpleNamespace(model=object()),
        SimpleNamespace(zw=zw),
        SimpleNamespace(workflowRunId="workflow"),
        SimpleNamespace(config=AutomatedWorkflowConfig()),
        example(AutomatedPreprocessingPlan),
        handoff,
        StudyContract.get_blank(),
        {},
        {"scientificInputs": "frozen"},
    )
    selected = example(ParameterCandidateEvaluation)
    selected.parameters.useHarmony = harmony
    selected.parameters.dimensions = 20
    selected.parameters.neighborsK = 11
    selected.metrics.nClusters = 2
    selected.metrics.topMarkerGenes = {"0": ["NKG7"], "1": ["MS4A1"]}
    run.evaluations["full"] = [selected]
    run.settings[selected.candidateId] = run.baseline().model_copy(
        update={"parameters": selected.parameters}
    )
    run.store = SimpleNamespace(
        zw=zw,
        inspect_artifact=lambda _ref: SimpleNamespace(
            exists=True, complete=True, inputs={"cell_selection": run.cells.to_dict()}
        ),
        load_artifact=lambda _ref: {"values": np.repeat([0, 1], 50)},
    )
    monkeypatch.setattr(
        run,
        "comparison_coverage",
        lambda scope, _cells: comparison_coverage(run, scope),
    )
    monkeypatch.setattr(run, "_feature_experiments", lambda _setting: {})
    monkeypatch.setattr(run, "feature_evidence", lambda _setting: {"genes": ["NKG7"]})
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
        lambda *args, **kwargs: [ImageEvidence(identifier="plot", data=b"image")],
    )
    return run, selected


@pytest.mark.parametrize("harmony", [False, True])
def test_interrupted_review_with_harmony_evidence_resumes_its_model_call(
    monkeypatch: pytest.MonkeyPatch, harmony: bool
) -> None:
    zw = zarr.open_group(store=MemoryStore(), mode="w")
    journal._ensure_orchestration_store(SimpleNamespace(zw=zw))
    calls: list[int] = []

    def unavailable(**kwargs):
        calls.append(1)
        raise RuntimeError("Provider unavailable")

    monkeypatch.setattr(rna_tuning, "run_agent_sync", unavailable)
    run, selected = _review_run(monkeypatch, zw, harmony=harmony)
    with pytest.raises(RuntimeError, match="Provider unavailable"):
        run.review("full", 0, selected, {})
    resumed, selected = _review_run(monkeypatch, zw, harmony=harmony)
    with pytest.raises(RuntimeError, match="Provider unavailable"):
        resumed.review("full", 0, selected, {})
    assert len(calls) == 2

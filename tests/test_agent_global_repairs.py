"""Global full-cohort repair admissions survive retries and context revisions."""

import hashlib
from copy import deepcopy
from typing import Any

import pytest

from scarf.agent import record_io
from scarf.agent.orchestrator import journal, rna_tuning
from scarf.agent.orchestrator.budget import CandidateBudget, CandidateBudgetExceeded
from scarf.agent.orchestrator.models import AutomatedWorkflowConfig
from scarf.agent.types import ArtifactReferenceModel
from tests.agent_journal_store import memory_journal
from tests.test_agent_required_comparisons import panel_run  # noqa: F401


def repair_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        ArtifactReferenceModel(
            scope="datastore", kind="cell_selection", artifactId="a" * 64
        ).model_dump(mode="json"),
        {
            "features": "fixed",
            "parameters": {
                "candidateId": "baseline",
                "dimensions": 21,
                "neighborsK": 11,
                "leidenResolution": 1.0,
            },
        },
        {"parameter": "leidenResolution", "value": 1.5},
    )


def save_review(
    store: Any, prefix: str, run_id: str, key: str, *, harmony: bool = False
) -> None:
    cells, setting, experiment = repair_inputs()
    if harmony:
        experiment = {"parameter": "useHarmony", "value": True}
    journal.save_checkpoint(
        store,
        prefix,
        run_id,
        key,
        {
            "comparisonCoverage": {"phase": "validation"},
            "candidates": [{"candidateId": "baseline", "cellSelection": cells}],
            "settings": {"baseline": setting},
            "experiments": {"target": experiment},
        },
        {
            "action": {
                "action": "experiment",
                "selectedCandidateId": "baseline",
                "experimentId": "target",
            }
        },
    )


def test_repair_reservation_is_global_idempotent_and_does_not_charge_primary_work() -> (
    None
):
    store, prefix, record = memory_journal()
    original = {"study": "original"}
    budget = CandidateBudget(
        store, prefix, record.workflowRunId, record.config, original
    )
    cells, setting, experiment = repair_inputs()
    budget.admit_repair(cells, setting, experiment)
    before = journal._list_keys(store.zw, prefix)
    assert len(budget.repairs) == 1
    assert budget.summary()["scopes"]["full"]["reserved"] == {
        "graphs": 0,
        "partitions": 0,
    }
    resumed = CandidateBudget(
        store,
        prefix,
        record.workflowRunId,
        record.config,
        {"study": "revised"},
        previous_provenances=(original,),
    )
    renamed = deepcopy(setting)
    renamed["parameters"]["candidateId"] = "replayed-label"
    resumed.admit_repair(cells, renamed, experiment)
    assert journal._list_keys(store.zw, prefix) == before
    with pytest.raises(CandidateBudgetExceeded, match="repair has been used"):
        resumed.admit_repair(cells, setting, {"parameter": "dimensions", "value": 30})
    assert len(resumed.repairs) == 1


@pytest.mark.parametrize("revised", [False, True])
def test_old_committed_repair_is_counted_without_rewriting_it(revised: bool) -> None:
    store, prefix, record = memory_journal()
    original = {"study": "original"}
    digest = hashlib.sha256(record_io.canonical_json_bytes(original)).hexdigest()
    key = (
        f"parameter_tuning/evidence_revisions/{digest}/full/review0"
        if revised
        else "parameter_tuning/full/review0"
    )
    save_review(store, prefix, record.workflowRunId, key)
    before = journal._list_keys(store.zw, prefix)
    saved = journal.read_checkpoint(store, prefix, record.workflowRunId, key)
    budget = CandidateBudget(
        store,
        prefix,
        record.workflowRunId,
        record.config,
        {"study": "new"},
        previous_provenances=(original,),
    )
    assert len(budget.repairs) == 1
    budget.admit_repair(*repair_inputs())
    assert journal._list_keys(store.zw, prefix) == before
    assert journal.read_checkpoint(store, prefix, record.workflowRunId, key) == saved
    with pytest.raises(CandidateBudgetExceeded):
        cells, setting, _ = repair_inputs()
        budget.admit_repair(cells, setting, {"parameter": "dimensions", "value": 30})


def test_harmony_comparison_does_not_consume_a_targeted_repair() -> None:
    store, prefix, record = memory_journal()
    save_review(
        store,
        prefix,
        record.workflowRunId,
        "parameter_tuning/full/review0",
        harmony=True,
    )
    budget = CandidateBudget(store, prefix, record.workflowRunId, record.config, {})
    assert not budget.repairs
    budget.admit_repair(*repair_inputs())
    assert len(budget.repairs) == 1


def test_unadmitted_proposal_does_not_inflate_explicit_repair_accounting() -> None:
    store, prefix, record = memory_journal()
    budget = CandidateBudget(store, prefix, record.workflowRunId, record.config, {})
    cells, setting, _ = repair_inputs()
    budget.admit_repair(cells, setting, {"parameter": "dimensions", "value": 30})
    save_review(store, prefix, record.workflowRunId, "parameter_tuning/full/review0")
    restored = CandidateBudget(store, prefix, record.workflowRunId, record.config, {})
    assert restored.repairs == budget.repairs


def test_disabled_repair_limit_rejects_before_any_primary_admission() -> None:
    store, prefix, record = memory_journal()
    budget = CandidateBudget(
        store,
        prefix,
        record.workflowRunId,
        AutomatedWorkflowConfig(maxFullRepairs=0),
        {},
    )
    with pytest.raises(CandidateBudgetExceeded):
        budget.admit_repair(*repair_inputs())
    assert not budget.repairs
    assert not budget.admissions["full"]


def test_full_recovery_executes_requested_resolution_and_preserves_baseline_panel(
    request: pytest.FixtureRequest,
    monkeypatch: Any,
) -> None:
    run = request.getfixturevalue("panel_run")
    run.recovery_scope = "sample1"
    monkeypatch.setattr(
        rna_tuning,
        "screening_coverage",
        lambda *args: ({"screeningCells": 100, "populationCells": 100}, []),
    )
    monkeypatch.setattr(
        run,
        "experiments",
        lambda selected: {"target": {"parameter": "leidenResolution", "value": 1.5}},
    )
    calls = []

    def review(scope: str, index: int, selected: Any, coverage: Any) -> Any:
        calls.append(selected.parameters.leidenResolution)
        return rna_tuning.TuningAction(
            action="experiment" if index == 0 else "accept",
            selectedCandidateId=selected.candidateId,
            experimentId="target" if index == 0 else None,
            rationale="The observed population split needs a finer partition.",
            plainLanguageSummary="Check the supported split.",
            correctionNeed="notApplicable",
            evidenceIds=[f"candidate:{selected.candidateId}"],
            quantitativeFindings=["The supplied fixture has completed graph evidence."],
            qualitativeFindings=["A supported population may contain a finer split."],
            comparisonConclusions=[],
            objectivePreservation="Retain the frozen cohort and gene selection.",
            concern="The supported split requires resolution sensitivity evidence.",
            expectedImprovement="Resolve the split without changing the graph.",
        )

    monkeypatch.setattr(run, "review", review)
    status, selected = run.assess_scope("full", run.cells, run.baseline())
    assert status == "accept"
    assert selected.parameters.leidenResolution == 1.5
    assert calls == [1.0, 1.5]
    assert len(run.resolution_candidates["full"]) == 4
    assert run.budget.summary()["scopes"]["full"]["completed"] == {
        "graphs": 1,
        "partitions": 5,
    }
    assert len(run.budget.repairs) == run.full_repairs == 1
    assert run.full_repair["selectedCandidateId"] == selected.candidateId


def test_targeted_recovery_plans_do_not_collide_for_same_graph_different_question(
    request: pytest.FixtureRequest,
) -> None:
    run = request.getfixturevalue("panel_run")
    run.recovery_scope = "sample1"
    setting = run.baseline()
    run._resolution_panel("full", run.cells, setting)
    counts = run.budget.summary()
    run.last_action = rna_tuning.TuningAction(
        action="enlarge",
        selectedCandidateId="observed",
        rationale="The larger sample left donor support uncertain.",
        plainLanguageSummary="Validate donor support with all cells.",
        correctionNeed="notApplicable",
        evidenceIds=["candidate:observed"],
        quantitativeFindings=["Coverage was insufficient in the observed sample."],
        qualitativeFindings=["Small populations require more donor support evidence."],
        comparisonConclusions=[],
        objectivePreservation="Keep rare populations eligible for assessment.",
    )
    run._resolution_panel("full", run.cells, setting)
    assert run.budget.summary() == counts

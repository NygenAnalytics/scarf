"""Bounded RNA admission, uniform sampling and evidence-driven acceptance."""

from tests.agent_examples import example
from tests.agent_comparison_examples import observed_action

import copy
import hashlib
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scarf.agent.orchestrator import journal, rna_tuning
from scarf.agent import record_io
from scarf.agent.orchestrator.budget import CandidateBudget, CandidateBudgetExceeded
from scarf.agent.orchestrator.models import (
    AutomatedPreprocessingPlan,
    AutomatedWorkflowConfig,
    PreprocessedAssayHandoff,
)
from scarf.agent.experimental_context.study import StudyContract
from scarf.agent.parameter_tuning.contracts import ParameterCandidateEvaluation
from scarf.agent.parameter_tuning.hvg import core_hvg_evidence, rank_core_hvgs
from scarf.agent.types import ArtifactReferenceModel
from scarf.storage.selections import read_stored_selection_indices


@pytest.fixture
def checkpoints(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    saved: dict[str, Any] = {}

    def load(
        _store: Any, _prefix: str, _workflow: str, key: str, inputs: Any = None
    ) -> Any:
        if key not in saved:
            return None
        if inputs is not None and inputs != saved[key]["inputs"]:
            raise ValueError("changed checkpoint inputs")
        return copy.deepcopy(saved[key]["outputs"])

    def save(
        _store: Any, _prefix: str, _workflow: str, key: str, inputs: Any, outputs: Any
    ) -> Any:
        record = {"inputs": copy.deepcopy(inputs), "outputs": copy.deepcopy(outputs)}
        if key in saved and saved[key] != record:
            raise ValueError("checkpoint conflict")
        saved[key] = record
        return outputs

    def read(_store: Any, _prefix: str, _workflow: str, key: str) -> Any:
        if key not in saved:
            return None
        record = copy.deepcopy(saved[key])
        return {
            **record,
            "contentSha256": hashlib.sha256(
                record_io.canonical_json_bytes(record)
            ).hexdigest(),
        }

    monkeypatch.setattr(journal, "load_checkpoint", load)
    monkeypatch.setattr(journal, "save_checkpoint", save)
    monkeypatch.setattr(journal, "read_checkpoint", read)
    monkeypatch.setattr(journal, "_ensure_orchestration_store", lambda store: "test")
    return saved


def numerical_inputs(
    resolution: float, *, harmony: bool = False, dimensions: int = 21
) -> dict[str, Any]:
    return {
        "cells": "full",
        "features": "default",
        "parameters": {
            "candidateId": "example",
            "dimensions": dimensions,
            "neighborsK": 11,
            "leidenResolution": resolution,
            "useHarmony": harmony,
        },
    }


def test_full_defaults_share_one_graph_and_resume_admissions(
    checkpoints: dict[str, Any],
) -> None:
    config = AutomatedWorkflowConfig()
    budget = CandidateBudget(None, "test", "workflow", config, {"input": "frozen"})
    defaults = [numerical_inputs(value) for value in (0.5, 0.75, 1.0, 1.25)]
    budget.admit_many("full", defaults)
    assert budget.summary()["scopes"]["full"]["reserved"]["graphs"] == 1
    assert budget.summary()["scopes"]["full"]["reserved"]["partitions"] == 4
    assert budget.summary()["scopes"]["full"]["completed"] == {
        "graphs": 0,
        "partitions": 0,
    }
    first = budget.admit("full", defaults[0])
    budget.complete(first, {"evaluation": "complete"})
    budget.complete(first, {"evaluation": "complete"})
    assert budget.summary()["scopes"]["full"]["completed"] == {
        "graphs": 1,
        "partitions": 1,
    }
    resumed = CandidateBudget(None, "test", "workflow", config, {"input": "frozen"})
    resumed.admit_many("full", defaults)
    assert resumed.completed(resumed.admit("full", defaults[0])) == {
        "evaluation": "complete"
    }
    assert resumed.summary() == budget.summary()
    resumed.complete(
        resumed.admit("full", defaults[1]), {"evaluation": "another partition"}
    )
    assert resumed.summary()["scopes"]["full"] == {
        "reserved": {"graphs": 1, "partitions": 4},
        "completed": {"graphs": 1, "partitions": 2},
    }
    with pytest.raises(ValueError, match="changed checkpoint inputs"):
        CandidateBudget(None, "test", "workflow", config, {"input": "changed"})


def test_matched_pair_admission_fails_before_either_branch(
    checkpoints: dict[str, Any],
) -> None:
    config = AutomatedWorkflowConfig(maxFullGraphs=1)
    budget = CandidateBudget(None, "test", "workflow", config, {})
    with pytest.raises(CandidateBudgetExceeded, match="full-cohort comparison"):
        budget.admit_many(
            "full", [numerical_inputs(1.0), numerical_inputs(1.0, harmony=True)]
        )
    assert checkpoints == {}
    assert budget.summary()["scopes"]["full"]["reserved"]["partitions"] == 0


def test_summary_counts_unique_diagnostic_evidence_and_labels_reuse(
    checkpoints: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    run = object.__new__(rna_tuning.RnaTuningRun)
    run.budget = CandidateBudget(
        None, "test", "workflow", AutomatedWorkflowConfig(), {}
    )
    run.budget.admit_many("full", [numerical_inputs(0.5), numerical_inputs(0.75)])
    evaluation = example(ParameterCandidateEvaluation)
    evaluation.metrics.subsampleStability = 0.9
    prototype = next(iter(evaluation.artifacts.values()))
    evaluation.artifacts = {
        name: prototype.model_copy(update={"artifactId": f"{index:064x}"})
        for index, name in enumerate(
            (
                "representationDiagnostic",
                "stabilityClusters",
                "markerTable",
                "doubletScore:0",
            )
        )
    }
    evaluation.artifacts["doubletScore:1"] = evaluation.artifacts["doubletScore:0"]
    evaluation.artifacts["clusters"] = prototype.model_copy(
        update={"artifactId": "f" * 64}
    )
    run.budget.complete(
        run.budget.admit("full", numerical_inputs(0.5)),
        {"evaluation": evaluation.model_dump(mode="json")},
    )
    failed = evaluation.model_copy(
        update={
            "candidateId": "failed",
            "status": "failed",
            "artifacts": {"markerTable": evaluation.artifacts["clusters"]},
        }
    )
    run.evaluations = {
        "sample0": [],
        "sample1": [],
        "full": [evaluation, evaluation.model_copy(), failed],
    }
    run.history = []
    run.full_repairs = 0
    run.diagnostic_counts = {}
    messages = []
    monkeypatch.setattr(rna_tuning.logger, "info", messages.append)
    summary = run.summary()
    assert summary["diagnosticEvidence"]["uniqueArtifacts"] == {
        "pcaDiagnostics": 1,
        "stabilityClusters": 1,
        "markerTable": 1,
        "doubletScore": 1,
    }
    assert summary["diagnosticEvidence"]["subsampleStabilityEvaluations"] == {
        "sample0": 0,
        "sample1": 0,
        "full": 1,
    }
    assert "1/1 graphs and 1/2 partitions completed/reserved" in messages[0]
    assert messages[1].startswith("Tuning limits:")
    assert "Saved evidence may be reused" in messages[2]
    assert "not computation counts" in messages[2]
    assert run.summary() == summary


def test_full_fallback_correction_and_one_repair_fit_declared_limits(
    checkpoints: dict[str, Any],
) -> None:
    budget = CandidateBudget(None, "test", "workflow", AutomatedWorkflowConfig(), {})
    budget.admit_many(
        "full", [numerical_inputs(value) for value in (0.5, 0.75, 1.0, 1.25)]
    )
    budget.admit("full", numerical_inputs(1.0, harmony=True))
    budget.admit_many(
        "full",
        [
            numerical_inputs(1.0, dimensions=30),
            numerical_inputs(1.0, dimensions=30, harmony=True),
        ],
    )
    assert budget.summary()["scopes"]["full"]["reserved"]["graphs"] == 4
    assert budget.summary()["scopes"]["full"]["reserved"]["partitions"] == 7
    with pytest.raises(CandidateBudgetExceeded, match="graph limit"):
        budget.admit("full", numerical_inputs(1.0, dimensions=50))


def test_screen_admissions_include_enlargement_and_no_double_charge(
    checkpoints: dict[str, Any],
) -> None:
    config = AutomatedWorkflowConfig(
        maxScreeningEvaluations=4, maxTotalScreeningEvaluations=6
    )
    budget = CandidateBudget(None, "test", "workflow", config, {})
    rows = [numerical_inputs(value) for value in (0.5, 0.75, 1.0, 1.25)]
    budget.admit_many("sample0", rows)
    budget.admit("sample0", rows[0])
    with pytest.raises(CandidateBudgetExceeded, match="screening comparison"):
        budget.admit_many("sample1", rows)
    assert budget.summary()["scopes"]["sample0"]["reserved"]["partitions"] == 4
    assert budget.summary()["scopes"]["sample1"]["reserved"]["partitions"] == 0


def test_uniform_screen_is_nested_exact_and_keeps_live_selection(
    datastore_ephemeral: Any,
) -> None:
    store = datastore_ephemeral
    parent = store.snapshot_cell_selection("I")
    before = store.cells.fetch_all("I").copy()
    small = rna_tuning.uniform_screening_selection(store, parent, size=8, seed=4444)
    large = rna_tuning.uniform_screening_selection(store, parent, size=16, seed=4444)

    def rows(selection: Any) -> np.ndarray:
        return read_stored_selection_indices(
            store.zw,
            selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )

    assert len(rows(small)) == 8
    assert len(rows(large)) == 16
    assert set(rows(small)) < set(rows(large)) <= set(rows(parent))
    assert (
        rna_tuning.uniform_screening_selection(store, parent, size=8, seed=4444)
        == small
    )
    np.testing.assert_array_equal(before, store.cells.fetch_all("I"))


def test_canonical_hvg_global_ranking_matches_exact_core_default(
    datastore_ephemeral: Any,
) -> None:
    store = datastore_ephemeral
    cells = store.snapshot_cell_selection("I")
    refs = core_hvg_evidence(store, assay="RNA", cells=cells)
    exact = store.select_hvgs(cells, from_assay="RNA", show_plot=False)
    assert refs["scarfDefault"] == exact
    ranked = rank_core_hvgs(
        store,
        eligible=refs["eligibleDefault"],
        statistics=refs["eligibleAll"],
        top_n=1000,
    )
    np.testing.assert_array_equal(
        store.load_artifact(ranked)["values"][:],
        store.load_artifact(exact)["values"][:],
    )


@pytest.mark.parametrize("unsupported", [False, True])
def test_native_acceptance_requires_resolved_supported_correction_need(
    checkpoints: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    unsupported: bool,
) -> None:
    from tests.test_agent_rna_evidence_mode import assess, make_run

    run, evaluation = make_run(monkeypatch, object())
    run.study = run.study.model_copy(
        update={
            "correctionLicense": "safe",
            "technicalBatchColumns": ["batch"],
            "unsupportedProtection": ["age"] if unsupported else [],
        }
    )
    run.batch_columns = ["batch"]
    run.evaluations["sample0"] = [evaluation]

    def propose_native(**kwargs: Any) -> Any:
        action = assess(**{**kwargs, "output_validator": lambda value: value}).output
        return SimpleNamespace(
            output=action.model_copy(update={"correctionNeed": "needed"})
        )

    monkeypatch.setattr(rna_tuning, "run_agent_sync", propose_native)
    with pytest.raises(
        ValueError,
        match="biological protection is unsupported"
        if unsupported
        else "required correction unresolved",
    ):
        run.review("sample0", 0, evaluation, {})
    assert "parameter_tuning/sample0/review0" not in checkpoints
    assert "parameter_tuning/sample0/review0/evidence/visual" in checkpoints


@pytest.mark.slow
def test_required_comparisons_and_resume_reuse_augmented_evidence(
    datastore_ephemeral: Any,
    checkpoints: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    store = datastore_ephemeral
    cells = store.snapshot_cell_selection("I")
    features = core_hvg_evidence(store, assay="RNA", cells=cells)
    feature_models = {
        key: ArtifactReferenceModel.from_artifact_ref(value)
        for key, value in features.items()
    }
    mask = np.asarray(
        store.load_artifact(features["scarfDefault"])["values"][:], dtype=bool
    )
    n_cells = len(
        read_stored_selection_indices(
            store.zw,
            cells,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
    )
    handoff = PreprocessedAssayHandoff(
        assay="RNA",
        assayType="RNA",
        cellSelection=ArtifactReferenceModel.from_artifact_ref(cells),
        graphFeatures=feature_models["scarfDefault"],
        graphFeatureCandidates=feature_models,
        markerFeatures=ArtifactReferenceModel.from_artifact_ref(
            store.select_all_features(from_assay="RNA")
        ),
        nCells=n_cells,
        nFeatures=int(mask.sum()),
        reductionMethod="pca",
    )
    request = SimpleNamespace(config=AutomatedWorkflowConfig())
    plan = example(AutomatedPreprocessingPlan)
    plan.cellQc.attributes = []
    model_calls: list[str] = []

    def assess(**kwargs: Any) -> Any:
        evidence = json.loads(kwargs["user_prompt"][0])
        selected = evidence["currentCandidateId"]
        model_calls.append(selected)
        action = rna_tuning.TuningAction.model_validate(observed_action(evidence))
        return SimpleNamespace(output=kwargs["output_validator"](action))

    monkeypatch.setattr(rna_tuning, "run_agent_sync", assess)

    def runner() -> rna_tuning.RnaTuningRun:
        return rna_tuning.RnaTuningRun(
            SimpleNamespace(model=object()),
            store,
            SimpleNamespace(workflowRunId="full-test"),
            request,
            plan,
            handoff,
            StudyContract.get_blank(),
            {},
            {"dataset": "frozen"},
        )

    first, history = runner().run()
    assert first.status == "done"
    assert first.cellSelection == handoff.cellSelection
    assert first.selectedArtifacts["normalized"]
    assert len(model_calls) >= 3
    expected_calls = len(model_calls)
    assert history["budget"]["scopes"]["sample0"]["reserved"]["partitions"] >= 8
    assert history["budget"]["scopes"]["full"]["reserved"] == {
        "graphs": 0,
        "partitions": 0,
    }
    assert history["fullRepairs"] == 0

    def no_recomputation(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("A fully augmented candidate must not be recomputed on resume")

    monkeypatch.setattr(rna_tuning, "augment_pca_evaluations", no_recomputation)
    monkeypatch.setattr(rna_tuning, "augment_cluster_evaluations", no_recomputation)
    monkeypatch.setattr(rna_tuning, "execute_parameter_candidate", no_recomputation)
    monkeypatch.setattr(rna_tuning, "partition_comparison_evidence", no_recomputation)
    resumed, resumed_history = runner().run()
    assert resumed == first
    assert {
        key: value
        for key, value in resumed_history.items()
        if key != "diagnosticOperations"
    } == {key: value for key, value in history.items() if key != "diagnosticOperations"}
    measured = history["diagnosticOperations"]["operations"]
    assert measured["core.primaryNormalization"]["completed"] > 0
    restored = resumed_history["diagnosticOperations"]["operations"]
    assert restored["diagnostic.primaryCandidateEvidence"]["restored"] > 0
    assert all(row["attempted"] == 0 for row in restored.values())
    assert len(model_calls) == expected_calls


@pytest.mark.parametrize(
    "enabled,license,batch_columns,expected_scores",
    [
        (True, "notApplicable", [], 1),
        (False, "notApplicable", [], 0),
        (False, "unsafeConfounded", ["batch"], 0),
        (False, "safe", ["batch"], 1),
    ],
)
def test_failed_execution_retries_and_doublets_bind_exact_feature_mask(
    checkpoints: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    license: str,
    batch_columns: list[str],
    expected_scores: int,
) -> None:
    handoff = example(PreprocessedAssayHandoff)
    handoff.graphFeatureCandidates = {"eligibleDefault": handoff.graphFeatures}
    normalized = rna_tuning.artifact_model_to_ref(handoff.normalized)
    store = SimpleNamespace(run_normalization=lambda *args, **kwargs: normalized)
    run = rna_tuning.RnaTuningRun(
        SimpleNamespace(model=object()),
        store,
        SimpleNamespace(workflowRunId="retry"),
        SimpleNamespace(config=AutomatedWorkflowConfig(scoreDoublets=enabled)),
        example(AutomatedPreprocessingPlan),
        handoff,
        StudyContract.get_blank().model_copy(
            update={
                "correctionLicense": license,
                "technicalBatchColumns": batch_columns,
            }
        ),
        {},
        {},
    )
    baseline = run.baseline()
    old = example(ParameterCandidateEvaluation)
    old.parameters = baseline.parameters.model_copy(update={"candidateId": "old"})
    old.candidateId = "old"
    old.cellSelection = handoff.cellSelection
    old.artifacts["graphFeatures"] = handoff.graphFeatures.model_copy(
        update={"artifactId": "f" * 64}
    )
    run.evaluations["full"] = [old]
    run.settings["old"] = baseline.model_copy(
        update={"features": old.artifacts["graphFeatures"]}
    )
    calls = []

    def prepare(*args: Any, **kwargs: Any) -> Any:
        assert kwargs["min_cluster_cells"] == 1
        return SimpleNamespace(candidate=kwargs["candidates"][0]), ["candidate"]

    def execute(deps: Any, candidate: str) -> Any:
        calls.append(candidate)
        return old.model_copy(
            update={
                "candidateId": deps.candidate.candidateId,
                "parameters": deps.candidate,
                "artifacts": dict(old.artifacts),
                "status": "failed" if len(calls) == 1 else "done",
                "error": "transient failure" if len(calls) == 1 else None,
            }
        )

    scored = []

    def doublets(store: Any, selected: Any, candidates: Any, **kwargs: Any) -> Any:
        assert len(candidates) == 1
        assert candidates[0].candidateId == selected.candidateId
        assert (
            candidates[0].artifacts["graphFeatures"].model_dump()
            == handoff.graphFeatures.model_dump()
        )
        scored.append(selected.candidateId)
        return None

    monkeypatch.setattr(rna_tuning, "prepare_parameter_tuning_dependencies", prepare)
    monkeypatch.setattr(rna_tuning, "execute_parameter_candidate", execute)
    monkeypatch.setattr(
        rna_tuning, "augment_pca_evaluations", lambda store, rows, **kw: rows
    )
    monkeypatch.setattr(
        rna_tuning, "augment_cluster_evaluations", lambda store, rows, **kw: rows
    )
    monkeypatch.setattr(rna_tuning, "score_advisory_doublets", doublets)
    with pytest.raises(RuntimeError, match="transient failure"):
        run.execute("full", run.cells, baseline)
    assert run.budget.summary()["scopes"]["full"]["reserved"]["partitions"] == 1
    assert run.budget.summary()["scopes"]["full"]["completed"]["partitions"] == 0
    assert not any(key.endswith("/complete") for key in checkpoints)
    completed = run.execute("full", run.cells, baseline)
    assert completed.status == "done"
    assert run.budget.summary()["scopes"]["full"]["reserved"]["partitions"] == 1
    assert run.budget.summary()["scopes"]["full"]["completed"]["partitions"] == 1
    assert len(scored) == expected_scores
    assert any("score_doublets=False" in warning for warning in completed.warnings) == (
        expected_scores == 0
    )
    if not expected_scores:
        assert completed.metrics.doubletHighScoreConcentration is None
        assert any(
            "contamination was not assessed" in warning
            for warning in completed.warnings
        )


@pytest.mark.parametrize(
    "protection",
    [{"protectFamilies": ["ribosomalProtein"]}, {"protectFeatures": ["RPS1"]}],
)
def test_feature_experiments_preserve_aliases_and_exact_protected_genes(
    checkpoints: dict[str, Any],
    protection: dict[str, Any],
) -> None:
    handoff = example(PreprocessedAssayHandoff)
    handoff.graphFeatureCandidates = {"eligibleDefault": handoff.graphFeatures}
    names = np.asarray(["RPS1", "RPL1", "ACTB", "CD3D"])
    feats = SimpleNamespace(fetch_all=lambda column: names)
    store = SimpleNamespace(
        load_artifact=lambda ref: {"values": np.ones(4, dtype=bool)},
        get_assay=lambda assay: SimpleNamespace(feats=feats),
    )
    plan = example(AutomatedPreprocessingPlan)
    plan.assays[0].featureParameters.update(protection)
    run = rna_tuning.RnaTuningRun(
        SimpleNamespace(model=object()),
        store,
        SimpleNamespace(workflowRunId="protection"),
        SimpleNamespace(config=AutomatedWorkflowConfig()),
        plan,
        handoff,
        StudyContract.get_blank(),
        {},
        {},
    )
    selected = example(ParameterCandidateEvaluation)
    run.settings[selected.candidateId] = run.baseline()
    run.family_patterns = {"ribosomal": "^RP[SL]"}
    with pytest.raises(ValueError, match="objective-protected"):
        run.apply_experiment(
            selected, {"parameter": "excludeFamily", "value": "ribosomal"}
        )


def test_inadequate_screens_preserve_evidence_without_unaffordable_full_panel(
    checkpoints: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handoff = example(PreprocessedAssayHandoff)
    handoff.nCells = 200_000
    handoff.graphFeatureCandidates = {"eligibleDefault": handoff.graphFeatures}
    run = rna_tuning.RnaTuningRun(
        SimpleNamespace(model=object()),
        SimpleNamespace(),
        SimpleNamespace(workflowRunId="fallback"),
        SimpleNamespace(config=AutomatedWorkflowConfig()),
        example(AutomatedPreprocessingPlan),
        handoff,
        StudyContract.get_blank(),
        {},
        {},
    )
    sampled = []
    assessed = []

    def sample(store: Any, cells: Any, *, size: int, seed: int) -> Any:
        sampled.append(size)
        return cells.__class__(
            scope=cells.scope,
            assay=cells.assay,
            kind=cells.kind,
            artifact_id=f"{size:064x}",
        )

    def assess(scope: str, cells: Any, initial: Any) -> Any:
        assessed.append((scope, cells, initial))
        return ("enlarge", None) if scope != "full" else ("defer", None)

    monkeypatch.setattr(rna_tuning, "uniform_screening_selection", sample)
    monkeypatch.setattr(run, "assess_scope", assess)
    report, summary = run.run()
    assert report.status == "needsInput"
    assert sampled == [20_000, 100_000]
    assert [scope for scope, _, _ in assessed] == ["sample0", "sample1"]
    assert summary["budget"]["scopes"]["full"]["reserved"]["partitions"] == 0

"""One faithful, read-only analysis report from the authoritative journal."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scarf.agent.orchestrator import journal
from scarf.agent.report import generator, plots
from scarf.agent.report.artifacts import (
    _local_root,
    report_directory,
    scientific_summary,
)
from scarf.agent.report.rendering import render_analysis_document
from scarf.agent.types import ArtifactReferenceModel
from scarf.storage.refs import ArtifactRef
from tests.test_agent_analysis_plots import display_store


def snapshot() -> dict[str, Any]:
    from scarf.agent.experimental_context import ExperimentalContextResult
    from scarf.agent.experimental_context.study import build_study_contract
    from tests.agent_comparison_examples import comparison_review

    request = {
        "studyContext": "Human RNA cells",
        "studyObjective": "Find stable populations <without> batch artifacts.",
    }
    context = ExperimentalContextResult.get_blank().model_dump(mode="json")
    context["status"] = "done"
    context["characterization"].update(
        status="done",
        columns=[{"name": "sample", "kind": "categorical", "domain": "design"}],
    )
    context["cellQc"]["profileId"] = "selected"
    context["qcProfiles"] = [
        {
            "profileId": "selected",
            "action": "globalGaussian",
            "driverAssay": "RNA2",
            "driverAssayType": "RNA",
            "attributes": ["RNA2_nCounts", "RNA2_nFeatures", "RNA2_percentMito"],
            "activeCells": 675218,
            "retainedCells": 621200,
            "retainedFraction": 621200 / 675218,
            "retainedCellsByColumn": {
                "condition": {"control": 320000, "treated": 301200}
            },
        }
    ]
    context = ExperimentalContextResult.model_validate(context)
    study = build_study_contract(
        study_context=request["studyContext"],
        study_objective=request["studyObjective"],
        experimental_result=context,
    )
    review = comparison_review()
    return {
        "runId": "exact-analysis",
        "status": "completed",
        "request": request,
        "finalAnalysis": {
            "primaryAssay": "RNA2",
            "limitations": ["Condition and batch are confounded."],
        },
        "stages": [
            {
                "stage": "parameter_tuning",
                "status": "done",
                "report": {
                    "recommendedCandidateId": "candidate-two",
                    "evaluations": review["candidates"],
                },
                "decisions": [],
            },
            {
                "stage": "experimental_context",
                "status": "done",
                "report": context.model_dump(mode="json"),
                "outputs": {"studyContract": study.model_dump(mode="json")},
                "decisions": [
                    {
                        "record": {
                            "decisionId": "cellQuality",
                            "rationale": "The selected quality policy retains supported study groups.",
                        },
                    }
                ],
            },
        ],
        "analysisReviews": [review],
    }


def display_payload() -> dict[str, Any]:
    return {
        "clusterCounts": {"0": 620_000, "1": 1_200},
        "markers": [{"cluster": "1", "feature": "MS4A1", "score": 0.84}],
        "umap": "plots/final_umap.png",
        "displayedCells": 50_000,
        "displayNotes": [],
    }


@pytest.mark.parametrize("mode", ["visual", "structured"])
def test_one_page_shows_recorded_choices_evidence_and_qualitative_findings(
    mode,
) -> None:
    state = snapshot()
    state["analysisReviews"][0]["evidenceMode"] = mode
    original = copy.deepcopy(state)
    payload = scientific_summary(state) | display_payload()
    document = render_analysis_document(payload)
    assert state == original
    assert "621,200 cells" in document and "2 clusters" in document
    assert "50,000 of 621,200" in document
    assert "QC retained 621,200 cells (92.0%)" in document
    assert "Resolution 0.75 retains a small population with clear markers." in document
    assert (
        "Repeat agreement was 0.92 and 84% of clusters had qualifying markers."
        in document
    )
    assert "Original detailed model reasoning retained in the journal." not in document
    assert "small population with clear markers" in document
    assert "Condition and batch are confounded." in document
    assert "The selected quality policy retains supported study groups." in document
    assert document.index(
        "The selected quality policy retains supported study groups."
    ) < document.index("Compared policy")
    assert "&lt;without&gt;" in document and "<without>" not in document
    assert "hidden-record-id" not in document and "not-in-report" not in document
    assert "technical.html" not in document and "decision-tree" not in document
    assert "added noise" not in document and "weaker" not in document
    assert ("No plots were supplied for visual inspection" in document) == (
        mode == "structured"
    )
    assert (
        document.index("final_umap.png")
        < document.index("Cell quality")
        < document.index("Selected methods and evidence")
        < document.index("Limits of this analysis")
        < document.index("<footer>")
    )


def test_report_uses_nygen_page_styles() -> None:
    document = render_analysis_document(
        scientific_summary(snapshot()) | display_payload()
    )
    header = document.split("<header>", 1)[1].split("</header>", 1)[0]
    assert '<a href="https://www.nygen.io/">Nygen Analytics</a>' in header
    assert '<a href="https://www.nygen.io/products/scarfweb">ScarfWeb</a>' in header
    assert '<a href="https://www.nygen.io/products/cytetype">CyteType</a>' in header
    assert "Distributed, secure infrastructure" not in header
    assert "justify-content:space-between" in document
    assert "family=Inter:wght@300;400" in document
    assert "font-family:Inter,sans-serif" in document
    assert "line-height:1.2" in document
    assert "background:#ffffff" in document
    assert "#0077fc" in document
    assert "color:#b4b4b4" in document
    assert "font-weight:300" in document and "font-weight:400" in document
    assert "letter-spacing:-.04em" in document
    assert "border-radius:999px" in document
    assert all(color not in document for color in ("#237e6a", "#f4f6f5", "#bd8b22"))


def test_all_untrusted_scientific_text_is_escaped() -> None:
    state = snapshot()
    injection = '<img src=x onerror="alert(1)">'
    state["stages"][1]["decisions"][0]["record"]["rationale"] = injection
    state["finalAnalysis"]["limitations"] = [injection]
    payload = scientific_summary(state) | display_payload()
    payload["markers"][0]["feature"] = injection
    document = render_analysis_document(payload)
    assert injection not in document
    assert document.count("&lt;img src=x onerror=&quot;alert(1)&quot;&gt;") == 3


def test_missing_evidence_rejects_regeneration_without_inventing_reasons() -> None:
    state = snapshot()
    state["analysisReviews"] = []
    with pytest.raises(ValueError, match="mandatory objective and comparison evidence"):
        scientific_summary(state)


def test_report_distinguishes_actual_comparisons_and_unavailable_choices() -> None:
    state = snapshot()
    document = render_analysis_document(scientific_summary(state) | display_payload())
    assert "Number of variable genes" in document and "4,000" in document
    assert "No technical grouping supports a batch-specific ranking." in document
    assert "Variable-gene ranking: not compared" in document
    assert "Genes and representation" in document and "Clustering" in document
    assert "Clusters with qualifying markers" in document and "84.0%" in document
    assert "candidate-two" not in document and "hvgCount:4000" not in document
    assert "Full cohort (621,200 cells)" in document


def test_repeated_reviews_do_not_repeat_candidate_inventory_or_raw_history() -> None:
    state = snapshot()
    earlier = copy.deepcopy(state["analysisReviews"][0])
    earlier.update(
        action="combine",
        rationale="Historical model claim about an unexecuted correction.",
    )
    state["analysisReviews"].insert(0, earlier)
    document = render_analysis_document(scientific_summary(state) | display_payload())
    assert document.count("<td>41</td>") == 1
    assert document.count("<td>1.25</td>") == 1
    assert "Historical model claim" not in document
    assert "Complete recorded reasoning" not in document
    assert document.index("Populations and markers") < document.index(
        "Genes and representation"
    )


def population_snapshot() -> dict[str, Any]:
    state = snapshot()
    review = state["analysisReviews"][0]
    cells = review["candidates"][0]["cellSelection"]
    clusters = {
        "scope": "assay",
        "assay": "RNA2",
        "kind": "cluster_labels",
        "artifactId": "3" * 64,
    }
    state["finalAnalysis"].update(cellSelection=cells, clusters=clusters)
    state["stages"][1]["outputs"]["studyContract"]["independentUnitColumns"] = ["donor"]
    review["populationSupport"] = {
        "candidate-two": {
            "candidateId": "candidate-two",
            "cellSelection": ArtifactRef(
                scope="datastore",
                kind="cell_selection",
                artifact_id=cells["artifactId"],
            ).to_dict(),
            "clusters": ArtifactRef(
                scope="assay",
                assay="RNA2",
                kind="cluster_labels",
                artifact_id=clusters["artifactId"],
            ).to_dict(),
            "columns": {
                "donor": {
                    "status": "computed",
                    "observedGroups": 19,
                    "missingCells": 200,
                    "omittedPopulations": 1,
                    "populations": [
                        {
                            "cluster": "1",
                            "cells": 1200,
                            "groupsWithAtLeast5Cells": 4,
                            "largestGroupFraction": 0.938,
                        }
                    ],
                }
            },
        }
    }
    return state


def test_population_support_is_descriptive_and_missing_rows_are_unavailable() -> None:
    document = render_analysis_document(
        scientific_summary(population_snapshot()) | display_payload()
    )
    assert "93.8%" in document and "<td>4</td>" in document
    assert "five cells is not a replication threshold" in document
    assert "200 cells lack" in document
    assert "not saved for 1 populations" in document
    assert "Unavailable" in document
    assert "not validated cell identities" in document
    assert '<progress value="0.938000"' in document
    assert '<progress value="0.001932"' in document
    assert '<div class="population-overview">' in document
    assert "@media(max-width:800px)" in document


def test_report_consumes_population_diagnostics_without_recomputing_or_mutating(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from scarf.agent.parameter_tuning.diagnostics import population_support_evidence
    from tests.test_agent_population_support import _setup

    store, evaluation, _ = _setup(
        monkeypatch,
        np.repeat([1, 2], 6),
        {"donor": np.tile(["d1", "d2"], 6)},
    )
    evaluation.candidateId = "candidate-two"
    population = population_support_evidence(store, evaluation, ["donor"])
    state = population_snapshot()
    review = state["analysisReviews"][0]
    review["populationSupport"] = {evaluation.candidateId: population}
    for key in ("cellSelection", "clusters"):
        state["finalAnalysis"][key] = ArtifactReferenceModel.from_artifact_ref(
            ArtifactRef.from_dict(population[key])
        ).model_dump(mode="json")
    original = copy.deepcopy(state)
    calls = store.calls.copy()
    monkeypatch.setattr(
        generator,
        "collect_analysis_artifacts",
        lambda *_: display_payload() | {"clusterCounts": {"1": 6, "2": 6}},
    )

    path = generator.render_analysis_report(store, state, tmp_path)

    assert path == tmp_path / "index.html"
    assert "12 cells" in path.read_text()
    assert "50.0%" in path.read_text()
    assert state == original
    assert store.calls == calls
    assert "artifact_id" in population["clusters"]
    assert "artifactId" in state["finalAnalysis"]["clusters"]


def test_selected_population_concerns_and_study_limits_remain_prominent() -> None:
    state = snapshot()
    explanation = "Population 1 lacks qualifying markers and remains unclassified."
    state["analysisReviews"][0]["populationConcerns"] = [
        {
            "candidateId": "candidate-two",
            "clusterId": "1",
            "status": "nonEssentialLimitation",
            "evidenceIds": ["candidate:candidate-two:clusters"],
            "explanation": explanation,
        }
    ]
    study_limit = "The study contains only one independent donor."
    state["stages"][1]["outputs"]["studyContract"]["limitations"].append(study_limit)
    state["finalAnalysis"]["limitations"].append(explanation)
    document = render_analysis_document(scientific_summary(state) | display_payload())
    assert document.count(explanation) == 1
    assert document.index("final_umap.png") < document.index(explanation)
    assert document.index("final_umap.png") < document.index(study_limit)
    assert document.index(explanation) < document.index("<footer>")
    assert document.index(study_limit) < document.index("<footer>")


def test_report_keeps_the_recorded_tradeoff_beside_its_comparison() -> None:
    payload = scientific_summary(snapshot()) | display_payload()
    payload["assessments"][0]["comparisonConclusions"][0]["tradeoffs"] = [
        {"interpretation": "The preferred setting loses some repeat agreement."}
    ]
    document = render_analysis_document(payload)
    assert "The preferred setting loses some repeat agreement." in document
    assert document.index("Genes and representation") < document.index(
        "The preferred setting loses some repeat agreement."
    )


@pytest.mark.parametrize("field", ["clusters", "cellSelection", "candidateId"])
def test_population_support_must_match_the_final_candidate_and_artifacts(field) -> None:
    state = population_snapshot()
    population = state["analysisReviews"][0]["populationSupport"]["candidate-two"]
    population[field] = (
        "different-candidate"
        if field == "candidateId"
        else {**population[field], "artifact_id": "f" * 64}
    )
    with pytest.raises(ValueError, match="Reported population support"):
        scientific_summary(state)


@pytest.mark.parametrize("field", ["clusters", "cellSelection"])
@pytest.mark.parametrize("value", [None, "unbound-reference"])
def test_report_requires_population_artifact_reference_mappings(field, value) -> None:
    state = population_snapshot()
    state["analysisReviews"][0]["populationSupport"]["candidate-two"][field] = value
    original = copy.deepcopy(state)
    with pytest.raises(ValueError, match=f"lacks its {field} reference"):
        scientific_summary(state)
    assert state == original


@pytest.mark.parametrize("field", ["clusters", "cellSelection"])
@pytest.mark.parametrize(
    "mutation, message",
    [
        ({"artifact_id": "malformed"}, "64-character lowercase hex"),
        ({"artifactId": "f" * 64}, "fields do not match"),
        ({"type": "external_artifact"}, "type must be 'artifact'"),
    ],
)
def test_report_rejects_malformed_population_artifact_references(
    field: str, mutation: dict[str, str], message: str
) -> None:
    state = population_snapshot()
    population = state["analysisReviews"][0]["populationSupport"]["candidate-two"]
    population[field].update(mutation)
    original = copy.deepcopy(state)
    with pytest.raises(ValueError, match=message):
        scientific_summary(state)
    assert state == original


@pytest.mark.parametrize(
    "missing", ["evidenceRequirements", "comparisonCoverage", "comparisonConclusions"]
)
def test_incompatible_report_evidence_fails_before_rendering_or_replacing_files(
    monkeypatch, tmp_path, missing
) -> None:
    state = snapshot()
    if missing == "evidenceRequirements":
        state["stages"][1]["outputs"]["studyContract"].pop(missing)
    else:
        state["analysisReviews"][0].pop(missing)
    old_page = tmp_path / "index.html"
    old_page.write_text("Existing historical report")
    numerical = tmp_path / "saved-artifact"
    numerical.write_bytes(b"original numerical values")

    def unexpected(*args, **kwargs):
        pytest.fail("An incompatible report must fail before artifact display reads")

    monkeypatch.setattr(generator, "collect_analysis_artifacts", unexpected)
    with pytest.raises(ValueError, match="start a new workflow"):
        generator.render_analysis_report(SimpleNamespace(), state, tmp_path)
    assert old_page.read_text() == "Existing historical report"
    assert numerical.read_bytes() == b"original numerical values"


def test_invalid_or_missing_fractions_never_render_as_zero() -> None:
    state = population_snapshot()
    population = state["analysisReviews"][0]["populationSupport"]["candidate-two"][
        "columns"
    ]["donor"]["populations"][0]
    population["largestGroupFraction"] = float("nan")
    payload = scientific_summary(state) | display_payload()
    payload["qcProfiles"][0]["retainedFraction"] = None
    document = render_analysis_document(payload)
    assert "0.0%" not in document
    assert '<progress value="nan"' not in document


def test_comparison_scopes_use_candidate_evidence_not_the_latest_review_scope() -> None:
    from tests.agent_comparison_examples import comparison_review

    state = snapshot()
    subset = comparison_review("sample0")
    payload = scientific_summary(state) | display_payload()
    payload["assessments"].insert(0, subset)
    # A later full review can also cite earlier screening comparisons.
    for setting in subset["comparisonCoverage"]["candidateSettings"].values():
        assert setting["scope"] == "sample0"
    subset["scope"] = "full"
    payload["assessments"].insert(
        0, {"scope": "sample0", "coverage": {"screeningCells": 50000}}
    )
    document = render_analysis_document(payload)
    assert "Screening sample (50,000 cells)" in document
    assert "Full cohort (621,200 cells)" in document
    assert "Screening sample (621,200 cells)" not in document


def test_screening_that_uses_all_cells_is_not_labeled_as_a_sample() -> None:
    from tests.agent_comparison_examples import comparison_review

    payload = scientific_summary(snapshot()) | display_payload()
    all_cells = comparison_review("sample0")
    all_cells["coverage"]["screeningCells"] = 621200
    all_cells["comparisonCoverage"]["population"] = "allCells"
    payload["assessments"] = [all_cells]
    document = render_analysis_document(payload)
    assert "Full cohort (621,200 cells)" in document
    assert "Screening sample" not in document


@pytest.mark.parametrize("mode", ["visual", "structured"])
@pytest.mark.parametrize("damage", [None, "digest", "scope", "action", "genes", "mode"])
@pytest.mark.parametrize("revised", [False, True])
def test_review_view_requires_exact_checkpoint_bindings(
    monkeypatch: pytest.MonkeyPatch, damage: str | None, mode: str, revised: bool
) -> None:
    import hashlib

    from scarf.agent import record_io
    from scarf.agent.orchestrator.models import AutomatedWorkflowConfig
    from scarf.agent.orchestrator.rna_tuning import TuningAction

    state = snapshot()
    view = state["analysisReviews"][0]
    candidate = copy.deepcopy(
        next(
            item
            for item in view["candidates"]
            if item["candidateId"] == "candidate-two"
        )
    )
    features = ArtifactReferenceModel(
        assay="RNA2", kind="feature_selection", artifactId="4" * 64
    ).model_dump(mode="json")
    candidate["artifacts"] = {"graphFeatures": features}
    action = {
        key: value for key, value in view.items() if key in TuningAction.model_fields
    }
    payload = {
        "inputs": {
            "scope": "full",
            "imageHashes": {"observed": "image-digest"} if mode == "visual" else {},
            "evidenceMode": mode,
            "visualInspection": "available" if mode == "visual" else "unavailable",
            "candidates": [candidate],
            "settings": {
                "candidate-two": {
                    **view["settings"]["candidate-two"],
                    "parameters": candidate["parameters"],
                    "features": features,
                }
            },
            "featureEvidence": {
                "candidate-two": view["featureEvidence"]["candidate-two"]
            },
            "comparisonCoverage": view["comparisonCoverage"],
            "coverage": view["coverage"],
        },
        "outputs": {"action": action},
    }
    digest = hashlib.sha256(record_io.canonical_json_bytes(payload)).hexdigest()
    entry = {
        "scope": "full",
        "review": action,
        "checkpointKey": (
            f"parameter_tuning/evidence_revisions/{'a' * 64}/full/review0"
            if revised
            else "parameter_tuning/full/review0"
        ),
        "checkpointSha256": digest,
        "imageHashes": payload["inputs"]["imageHashes"],
        "evidenceMode": mode,
        "visualInspection": payload["inputs"]["visualInspection"],
    }
    if damage == "digest":
        entry["checkpointSha256"] = "bad-digest"
    elif damage == "scope":
        entry["scope"] = "sample0"
    elif damage == "action":
        entry["review"] = {**action, "rationale": "Unrecorded reasoning"}
    elif damage == "mode":
        entry["evidenceMode"] = "structured" if mode == "visual" else "visual"
    elif damage == "genes":
        payload["inputs"]["settings"]["candidate-two"]["features"] = {
            **features,
            "artifactId": "5" * 64,
        }
        digest = hashlib.sha256(record_io.canonical_json_bytes(payload)).hexdigest()
        entry["checkpointSha256"] = digest
    record = record_io.canonical_json_bytes({**payload, "contentSha256": digest})
    monkeypatch.setattr(record_io, "read_key", lambda *_args: record)
    stages = [
        {
            "stage": "parameter_tuning",
            "outputs": {"tuningEvidence": {"history": [entry]}},
        }
    ]
    if damage is not None:
        with pytest.raises(ValueError, match="Analysis review"):
            journal._analysis_review_views(
                SimpleNamespace(zw=object()),
                "agents/orchestrations",
                "workflow",
                stages,
                AutomatedWorkflowConfig(),
            )
    else:
        result = journal._analysis_review_views(
            SimpleNamespace(zw=object()),
            "agents/orchestrations",
            "workflow",
            stages,
            AutomatedWorkflowConfig(),
        )
        assert result[0]["rationale"] == action["rationale"]
        assert result[0]["evidenceMode"] == mode
        assert result[0]["settings"]["candidate-two"]["hvgCount"] == 1000
        assert result[0]["candidates"][0]["artifacts"] == candidate["artifacts"]
        assert result[0]["comparisonCoverage"] == view["comparisonCoverage"]
        assert result[0]["coverage"] == view["coverage"]


def test_report_regeneration_only_replaces_derived_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "data.zarr"
    root.mkdir()
    frozen = root / "numerical-artifact"
    frozen.write_bytes(b"immutable")
    output = report_directory(root, "exact-analysis", "workspace")
    monkeypatch.setattr(
        generator, "collect_analysis_artifacts", lambda *_: display_payload()
    )
    result = generator.render_analysis_report(SimpleNamespace(), snapshot(), output)
    assert result == output / "index.html"
    assert list(output.glob("*.html")) == [result]
    before = result.read_text()
    assert (
        generator.render_analysis_report(SimpleNamespace(), snapshot(), output)
        == result
    )
    assert result.read_text() == before
    assert frozen.read_bytes() == b"immutable"
    assert not list(output.glob(".*.tmp"))


def test_public_report_opens_exact_journal_and_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[Any] = []
    store = SimpleNamespace(workspace="workspace")

    def open_store(target: Path, run_id: str, *, workspace: str | None) -> Any:
        calls.append((target, run_id, workspace))
        return store

    def load_snapshot(target: Any, run_id: str) -> dict[str, Any]:
        assert target is store and run_id == "exact-analysis"
        return snapshot()

    monkeypatch.setattr(journal, "open_analysis_store", open_store, raising=False)
    monkeypatch.setattr(journal, "analysis_snapshot", load_snapshot, raising=False)
    monkeypatch.setattr(
        generator, "collect_analysis_artifacts", lambda *_: display_payload()
    )
    path = generator.generate_agent_report(
        tmp_path, "exact-analysis", workspace="workspace"
    )
    assert calls == [(tmp_path, "exact-analysis", "workspace")]
    assert (
        path
        == tmp_path / "workspace/agents/orchestrations/exact-analysis/report/index.html"
    )


def test_report_does_not_replace_existing_page_after_render_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    old = tmp_path / "index.html"
    old.write_text("previous report")
    monkeypatch.setattr(
        generator, "collect_analysis_artifacts", lambda *_: display_payload()
    )

    def fail(_payload: Any) -> str:
        raise RuntimeError("render failed")

    monkeypatch.setattr(generator, "render_analysis_document", fail)
    with pytest.raises(RuntimeError, match="render failed"):
        generator.render_analysis_report(SimpleNamespace(), snapshot(), tmp_path)
    assert old.read_text() == "previous report"
    assert not list(tmp_path.glob(".*.tmp"))


def test_report_rejects_remote_paths_escaping_workspaces_and_incomplete_runs(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="local filesystem"):
        _local_root("s3://example/data.zarr")
    with pytest.raises(FileNotFoundError):
        _local_root(tmp_path / "missing")
    assert _local_root(f"file://{tmp_path}") == tmp_path
    with pytest.raises(ValueError, match="outside"):
        report_directory(tmp_path, "run", "../outside")
    with pytest.raises(ValueError, match="identifier"):
        report_directory(tmp_path, "../escape", None)
    state = snapshot()
    state["status"] = "needsInput"
    with pytest.raises(ValueError, match="completed"):
        generator.render_analysis_report(SimpleNamespace(), state, tmp_path / "report")
    assert not (tmp_path / "report").exists()


def test_large_report_reads_saved_map_and_marker_table_without_analysis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store, refs, _ = display_store(monkeypatch)
    marker_calls: list[str] = []

    def markers(_ref: Any, *, group_id: str) -> pd.DataFrame:
        marker_calls.append(group_id)
        return pd.DataFrame(
            {
                "feature_name": ["CD79A", "MS4A1", "CD74", "HLA-DRA"],
                "score": [0.7, 0.9, 0.8, 0.6],
            }
        )

    store.get_markers = markers
    inspect = store.inspect_artifact

    def marker_status(ref: Any) -> Any:
        status = inspect(ref)
        if ref.kind == "marker_table":
            status.inputs["clusters"] = refs["clusters"].to_dict()
        return status

    store.inspect_artifact = marker_status
    final = {
        "umap": ArtifactReferenceModel.from_artifact_ref(refs["umap"]).model_dump(),
        "clusters": ArtifactReferenceModel.from_artifact_ref(
            refs["clusters"]
        ).model_dump(),
        "cellSelection": ArtifactReferenceModel.from_artifact_ref(
            refs["cell_selection"]
        ).model_dump(),
        "graph": ArtifactReferenceModel.from_artifact_ref(refs["graph"]).model_dump(),
        "markers": {
            "scope": "assay",
            "assay": "RNA2",
            "kind": "marker_table",
            "artifactId": "5" * 64,
        },
    }
    data = plots.collect_analysis_artifacts(store, final, tmp_path)
    assert data["displayNotes"] == []
    assert data["displayedCells"] == 50_000
    assert sum(data["clusterCounts"].values()) == 621_200
    assert len(data["markers"]) == 12 and len(marker_calls) == 4
    assert data["markers"][0]["feature"] == "MS4A1"
    assert (tmp_path / "plots/final_umap.png").is_file()
    provenance = json.loads((tmp_path / "plots/final_umap.png.json").read_text())
    assert provenance["provenance"]["extras"]["input_n_cells"] == 621_200
    assert list((tmp_path / "plots").glob("*.png")) == [
        tmp_path / "plots/final_umap.png"
    ]


def test_optional_map_failure_preserves_counts_and_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store, refs, _ = display_store(monkeypatch, n=12)

    def unavailable(*_args: Any, **_kwargs: Any) -> Any:
        raise ImportError("matplotlib unavailable")

    monkeypatch.setattr(plots, "plot_final_umap", unavailable)
    state = snapshot()
    state["finalAnalysis"]["clusters"] = ArtifactReferenceModel.from_artifact_ref(
        refs["clusters"]
    ).model_dump()
    for name, ref_name in (
        ("umap", "umap"),
        ("cellSelection", "cell_selection"),
        ("graph", "graph"),
    ):
        state["finalAnalysis"][name] = ArtifactReferenceModel.from_artifact_ref(
            refs[ref_name]
        ).model_dump()
    path = generator.render_analysis_report(store, state, tmp_path)
    document = path.read_text()
    assert "12 cells" in document
    assert "matplotlib unavailable" in document
    assert "Resolution 0.75 retains a small population with clear markers." in document
    assert 'src="plots/final_umap.png"' not in document


def test_invalid_map_lineage_is_not_hidden_as_optional_display_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store, refs, _ = display_store(monkeypatch, n=12)
    final = {
        name: ArtifactReferenceModel.from_artifact_ref(refs[ref_name]).model_dump()
        for name, ref_name in (
            ("umap", "umap"),
            ("clusters", "clusters"),
            ("graph", "graph"),
            ("cellSelection", "cell_selection"),
        )
    }

    def mismatch(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("Final artifacts must share the exact frozen cell selection")

    monkeypatch.setattr(plots, "plot_final_umap", mismatch)
    with pytest.raises(ValueError, match="exact frozen cell selection"):
        plots.collect_analysis_artifacts(store, final, tmp_path)
    assert not list(tmp_path.iterdir())

"""RNA selection, early rejection, and immutable resume boundaries."""

from tests.agent_examples import example

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import zarr

from scarf.agent.data_enrichment.contracts import DataEnrichmentReport
from scarf.agent.experimental_context.characterization import _SelectionBoundCells
from scarf.agent.experimental_context.contracts import ExperimentalContextDependencies
from scarf.agent.experimental_context.qc_evidence import (
    _offered_qc_profiles,
    _qc_driver,
)
from scarf.agent.orchestrator import (
    AgentOrchestrator,
    AutomatedWorkflowRequest,
    AutomatedWorkflowResumeRequest,
)
from scarf.agent.orchestrator import context as context_module
from scarf.agent.orchestrator import journal, main as main_module
from scarf.agent.orchestrator.models import (
    OrchestrationRequestRecord,
    PreprocessedAssayHandoff,
    WorkflowIdentity,
    WorkflowStageName,
)
from scarf.agent.orchestrator.rna import selected_rna_assay
from scarf.datastore.datastore import DataStore
from scarf.storage.budget import ResourceBudget
from scarf.storage.schema import create_zarr_count_assay
from scarf.storage.sharding import write_counts_t
from tests.agent_orchestrator_store import create_store


def _request(path: str = "study.zarr", **values: Any) -> AutomatedWorkflowRequest:
    return AutomatedWorkflowRequest(
        sourcePath=path,
        studyContext="A human RNA study with independent donors.",
        studyObjective="Find stable populations while preserving condition.",
        **values,
    )


def _add_assay(path: Path, name: str, assay_type: str) -> None:
    root = zarr.open_group(str(path), mode="r+")
    values = np.asarray([[1, 3, 0], [0, 2, 5], [3, 1, 0], [1, 4, 2]], dtype=np.uint32)
    ids = np.asarray([f"{name}-{index}" for index in range(3)])
    counts = create_zarr_count_assay(
        root,
        name,
        None,
        len(values),
        feat_ids=ids,
        feat_names=np.asarray(["MT-CO1", "GENE1", "GENE2"]),
        dtype="uint32",
        profile="fast_local",
    )
    counts[:] = values
    write_counts_t(counts, root[name], resources=ResourceBudget(1024**3, 2))
    root.attrs["assayTypes"] = {**dict(root.attrs["assayTypes"]), name: assay_type}
    root[name].attrs["dataset_fingerprint"] = f"dataset-{name.lower()}"


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"analysisAssays": ["RNA", "ADT"]}, "at most one"),
        ({"pairedAssays": ["RNA", "ADT"]}, "pairedAssays"),
        ({"analysisAssays": ["RNA2"], "primaryAssay": "RNA"}, "primaryAssay"),
        ({"primaryAssay": "RNA", "markerAssay": "ADT"}, "markerAssay"),
        ({"experimentalDirections": {"hypothesisTesting": {}}}, "hypothesis testing"),
    ],
)
def test_request_rejects_unsupported_routes(
    fields: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _request(**fields)


def test_selected_rna_uses_persisted_type_and_requires_unambiguous_selection() -> None:
    types = {"protein": "ADT", "transcriptome": "RNA", "tags": "HTO"}
    assert selected_rna_assay(_request(), types) == "transcriptome"
    types["RNA2"] = "RNA"
    with pytest.raises(ValueError, match="found 2"):
        selected_rna_assay(_request(), types)
    assert selected_rna_assay(_request(primaryAssay="RNA2"), types) == "RNA2"
    assert selected_rna_assay(_request(analysisAssays=["RNA2"]), types) == "RNA2"
    with pytest.raises(ValueError, match="RNA only"):
        selected_rna_assay(_request(primaryAssay="protein"), types)
    with pytest.raises(ValueError, match="markerAssay"):
        selected_rna_assay(_request(markerAssay="RNA2"), {"transcriptome": "RNA"})


def test_mixed_store_enriches_only_selected_second_rna(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = create_store(tmp_path / "mixed.zarr")
    _add_assay(path, "RNA2", "RNA")
    _add_assay(path, "HTO", "HTO")
    inspected: list[list[str]] = []

    def stop_after_selection(
        _agent: Any, _store: Any, **kwargs: Any
    ) -> DataEnrichmentReport:
        inspected.append(kwargs["assays"])
        return DataEnrichmentReport(
            status="failed", limitations=["Stopped after selection"]
        )

    monkeypatch.setattr(context_module.DataEnrichmentAgent, "run", stop_after_selection)
    result = AgentOrchestrator(object()).run(_request(str(path), primaryAssay="RNA2"))
    assert result.status == "failed"
    assert inspected == [["RNA2"]]
    assert result.workflowRunId is not None
    root = zarr.open_group(str(path), mode="r")
    prefix = "agents/orchestrations"
    record = journal._read_model(
        root,
        journal._request_key(prefix, result.workflowRunId),
        OrchestrationRequestRecord,
    )
    assert record.request.analysisAssays == ["RNA2"]
    assert record.request.primaryAssay == record.request.markerAssay == "RNA2"
    assert not journal._stage_outcomes(
        root, prefix, result.workflowRunId, "rna_quality_metrics"
    )


def test_ambiguous_rna_store_is_rejected_before_writable_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = create_store(tmp_path / "ambiguous.zarr")
    _add_assay(path, "RNA2", "RNA")
    orchestrator = AgentOrchestrator(object())

    def unexpected_open(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("Unsupported RNA selection must not open the store for writes")

    monkeypatch.setattr(orchestrator, "open_store", unexpected_open)
    result = orchestrator.run(_request(str(path)))
    assert result.status == "failed"
    assert "found 2" in result.notes[0]
    assert "agents" not in zarr.open_group(str(path), mode="r")


def test_explicit_second_rna_drives_qc_profiles(tmp_path: Path) -> None:
    path = create_store(tmp_path / "qc.zarr")
    _add_assay(path, "RNA2", "RNA")
    store = DataStore(str(path), default_assay="RNA", min_features_per_cell=-1)
    assert _qc_driver(store, "RNA2") == ("RNA2", "RNA")
    selection = store.snapshot_cell_selection()
    deps = ExperimentalContextDependencies(
        store=store,
        cells=_SelectionBoundCells(store.zw, store.cells, selection),
        qcAssay="RNA2",
        cellSelection=selection,
    )
    profiles = _offered_qc_profiles(deps)
    assert profiles
    assert {profile.driverAssay for profile in profiles} == {"RNA2"}
    assert all(
        not name.startswith("RNA_")
        for profile in profiles
        for name in profile.attributes
    )


def test_qc_driver_rejects_an_explicit_non_rna_non_atac_assay() -> None:
    store = SimpleNamespace(
        assay_names=["ADT"], zw=SimpleNamespace(attrs={"assayTypes": {"ADT": "ADT"}})
    )
    with pytest.raises(ValueError, match="RNA or ATAC"):
        _qc_driver(store, "ADT")


def test_missing_h5ad_logs_failure_and_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    messages: list[str] = []
    monkeypatch.setattr(main_module.logger, "error", messages.append)
    path = tmp_path / "missing.h5ad"
    result = AgentOrchestrator(object()).run(_request(str(path)))
    assert result.status == "failed"
    assert any("RNA analysis failed during ingest" in message for message in messages)
    assert any("missing.h5ad" in message for message in messages)


@pytest.mark.parametrize(
    ("route", "message"),
    [
        ("hto", "Saved automatic HTO"),
        ("handoff_assay", "Saved preprocessing"),
        ("handoff_type", "Saved preprocessing"),
        ("enrichment_modality", "Saved enrichment"),
    ],
)
def test_resume_rejects_unsupported_saved_route_before_writable_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: str, message: str
) -> None:
    path = create_store(tmp_path / "resume.zarr")
    store = DataStore(str(path), default_assay="RNA", min_features_per_cell=-1)
    orchestrator = AgentOrchestrator(object())
    workflow = WorkflowIdentity("resume-rna")
    record = orchestrator.initialize_request(
        store,
        workflow,
        _request(
            str(path),
            zarrPath=str(path),
            primaryAssay="RNA",
            markerAssay="RNA",
            analysisAssays=["RNA"],
        ),
    )
    prefix = journal._ensure_orchestration_store(store)
    stage: WorkflowStageName = (
        "rna_quality_metrics"
        if route == "hto"
        else "data_enrichment"
        if route == "enrichment_modality"
        else "preprocessing"
    )
    started = journal._start_attempt(
        store.zw, prefix, workflow.workflowRunId, stage, record, []
    )
    references = []
    outputs: dict[str, Any] = {}
    if route == "hto":
        outputs["htoIdentityArtifacts"] = [{"name": "HTO_identity"}]
    elif route == "enrichment_modality":
        report = example(DataEnrichmentReport)
        report.policies[0].assayModality = "ADT"
        _, reference = journal._save_stage_report(
            store,
            started,
            report,
            expected_type=DataEnrichmentReport,
        )
        references.append(reference)
    else:
        handoff = PreprocessedAssayHandoff(
            assay="RNA2" if route == "handoff_assay" else "RNA",
            assayType="ADT" if route == "handoff_type" else "RNA",
        )
        outputs["assays"] = [handoff.model_dump(mode="json")]
    outcome = journal._complete_attempt(
        started,
        status="done",
        outputs=outputs,
        report_references=references,
    )
    journal._save_outcome(store.zw, prefix, outcome)

    def unexpected_open(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("Unsupported saved analysis state must not open for writes")

    monkeypatch.setattr(orchestrator, "open_store", unexpected_open)
    with pytest.raises(ValueError, match=message):
        orchestrator.load_request_for_resume(
            AutomatedWorkflowResumeRequest(
                zarrPath=str(path), workflowRunId=workflow.workflowRunId
            )
        )

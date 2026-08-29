"""Tests for immutable JSON agent reports embedded in a Scarf store."""

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pytest
import zarr
from pydantic import ValidationError
from zarr.errors import ZarrUserWarning

import scarf.agent as agent_api
import scarf.agent.record_io as record_io
from scarf.agent.biological_interpretation import BiologicalInterpretationReport
from scarf.agent.config import AgentRunConfig
from scarf.agent.data_enrichment import DataEnrichmentReport
from scarf.agent.experimental_context import ExperimentalContextResult
from scarf.agent.parameter_tuning import ArtifactRecord, ParameterTuningReport
from scarf.agent.persistence import (
    AgentInvocation,
    AgentReportLink,
    AgentReportRecord,
    AgentReportReference,
    AgentWorkflowRun,
    create_agent_workflow,
    finalize_agent_workflow,
    list_agent_workflows,
    load_agent_record,
    load_agent_report,
    load_agent_workflow,
    save_agent_report,
)
from scarf.agent.types import (
    AgentDataModel,
    ArtifactReferenceModel,
    ExperimentalTuningHandoff,
    TuningBiologyHandoff,
)
from scarf.datastore.datastore import DataStore
from scarf.storage.schema import create_cell_data, create_zarr_count_assay


REPORT_CASES = (
    (DataEnrichmentReport, "data_enrichment"),
    (ExperimentalContextResult, "experimental_context"),
    (ParameterTuningReport, "parameter_tuning"),
    (BiologicalInterpretationReport, "biological_interpretation"),
)


class AnyReport(AgentDataModel):
    value: str = ""

    @classmethod
    def get_example(cls) -> "AnyReport":
        return cls(value="not an agent report")


def _populate_scarf_group(
    group: zarr.Group,
    *,
    fingerprints: dict[str, str | None],
) -> None:
    values = np.asarray(
        [
            [4, 0, 1, 0],
            [0, 3, 0, 2],
            [2, 1, 0, 0],
            [0, 0, 5, 1],
        ],
        dtype=np.uint32,
    )
    cell_ids = np.asarray([f"cell-{index}" for index in range(values.shape[0])])
    feature_ids = np.asarray([f"feature-{index}" for index in range(values.shape[1])])
    feature_names = np.asarray(["MT-CO1", "RPS3", "GENE1", "GENE2"])
    create_cell_data(
        group,
        None,
        ids=cell_ids,
        names=cell_ids,
        profile="fast_local",
    )
    for assay_name, fingerprint in fingerprints.items():
        counts = create_zarr_count_assay(
            group,
            assay_name,
            None,
            values.shape[0],
            feat_ids=feature_ids,
            feat_names=feature_names,
            dtype="uint32",
            profile="fast_local",
        )
        counts[:] = values
        if fingerprint is not None:
            group[assay_name].attrs["dataset_fingerprint"] = fingerprint
    group.attrs["assayTypes"] = {assay_name: "Assay" for assay_name in fingerprints}


def _create_scarf_store(
    tmp_path: Path,
    *,
    fingerprints: dict[str, str | None] | None = None,
) -> Path:
    path = tmp_path / "data.zarr"
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    _populate_scarf_group(
        root,
        fingerprints=fingerprints or {"RNA": "dataset-rna"},
    )
    return path


def _create_workspace_store(tmp_path: Path) -> Path:
    path = tmp_path / "data.zarr"
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    for workspace in ("workspace_a", "workspace_b"):
        group = root.create_group(workspace)
        _populate_scarf_group(
            group,
            fingerprints={"RNA": f"dataset-{workspace}"},
        )
    return path


def _report(
    report_type: type[AgentDataModel],
    execution_run_id: str,
) -> AgentDataModel:
    report = report_type.get_example()
    return report.model_copy(
        update={
            "runInfo": report.runInfo.model_copy(
                update={
                    "agentName": "intentionally_misleading",
                    "modelName": "modèle-α",
                    "runId": execution_run_id,
                }
            )
        }
    )


def _experimental_report(execution_run_id: str) -> ExperimentalContextResult:
    report = _report(ExperimentalContextResult, execution_run_id)
    assert isinstance(report, ExperimentalContextResult)
    characterization = report.characterization.model_copy(
        update={
            "coefficients": [
                {
                    "name": "treatment",
                    "observationUnit": "sample",
                    "independentUnit": "donor",
                    "scope": "between",
                }
            ]
        }
    )
    return report.model_copy(update={"characterization": characterization})


def _tuning_report(execution_run_id: str) -> ParameterTuningReport:
    report = _report(ParameterTuningReport, execution_run_id)
    assert isinstance(report, ParameterTuningReport)
    evaluation = report.evaluations[0]
    artifacts = {
        **evaluation.artifacts,
        "clusters": ArtifactRecord(
            scope="assay",
            kind="clusters",
            artifactId="c" * 64,
            assay="RNA",
        ),
    }
    evaluation = evaluation.model_copy(update={"artifacts": artifacts})
    return report.model_copy(
        update={
            "evaluations": [evaluation],
            "selectedArtifacts": artifacts,
        }
    )


def _invocation(
    agent_name: str,
    *,
    parents: list[AgentReportLink] | None = None,
    **updates: object,
) -> AgentInvocation:
    values: dict[str, object] = {
        "agentName": agent_name,
        "parentReports": parents or [],
        "inputs": {"fromAssay": "RNA"},
        "artifacts": {"input": ArtifactReferenceModel.get_example()},
        "runConfig": AgentRunConfig.get_example(),
    }
    values.update(updates)
    artifacts = dict(values["artifacts"])
    for handoff_name in (
        "experimentalTuningHandoff",
        "experimentalBiologyHandoff",
    ):
        handoff = values.get(handoff_name)
        selection = getattr(handoff, "cellSelection", None)
        if isinstance(selection, ArtifactReferenceModel):
            artifacts["cellSelection"] = selection
    values["artifacts"] = artifacts
    return AgentInvocation.model_validate(values)


def _report_path(
    path: Path,
    reference: AgentReportReference,
    *,
    workspace: str | None = None,
) -> Path:
    base = path if workspace is None else path / workspace
    return (
        base
        / "agents"
        / "runs"
        / reference.workflowRunId
        / reference.agentName
        / reference.agentRunId
        / "report.json"
    )


def test_public_persistence_models_have_factories_and_exports() -> None:
    model_types = (
        AgentReportLink,
        AgentInvocation,
        AgentReportReference,
        AgentReportRecord,
        AgentWorkflowRun,
    )
    for model_type in model_types:
        assert isinstance(model_type.get_blank(), model_type)
        assert isinstance(model_type.get_example(), model_type)
        assert all("_" not in field_name for field_name in model_type.model_fields)

    assert agent_api.AgentInvocation is AgentInvocation
    assert agent_api.AgentReportLink is AgentReportLink
    assert agent_api.AgentReportRecord is AgentReportRecord
    assert agent_api.create_agent_workflow is create_agent_workflow
    assert agent_api.finalize_agent_workflow is finalize_agent_workflow
    assert agent_api.load_agent_record is load_agent_record
    assert agent_api.save_agent_report is save_agent_report


def test_record_io_preserves_json_bytes_and_store_key_order(tmp_path: Path) -> None:
    value = {"z": ["é", 1], "a": True}
    assert record_io.canonical_json_bytes(value) == b'{"a":true,"z":["\xc3\xa9",1]}'
    assert record_io.display_json_bytes(value) == (
        '{\n  "a": true,\n  "z": [\n    "é",\n    1\n  ]\n}\n'.encode()
    )
    assert record_io.join_key("/agents/", "", "/runs/") == "agents/runs"

    path = _create_scarf_store(tmp_path)
    create_agent_workflow(path, workflow_run_id="workflow-1")
    root = zarr.open_group(str(path), mode="r")
    workflow_key = "agents/runs/workflow-1/workflow.json"

    assert record_io.list_keys(root, "agents/runs") == [workflow_key]
    assert record_io.read_key(root, workflow_key) == (path / workflow_key).read_bytes()
    assert record_io.read_key(root, "agents/missing.json") is None


def test_persistence_models_reject_invalid_identity_and_duplicate_parents() -> None:
    with pytest.raises(ValidationError, match="safe path component"):
        AgentReportLink(workflowRunId="../workflow")
    with pytest.raises(ValidationError, match="SHA-256"):
        AgentReportReference(contentSha256="not-a-digest")
    with pytest.raises(ValidationError, match="Workspace name"):
        AgentWorkflowRun(workspace="nested/workspace")
    with pytest.raises(ValidationError, match="cannot precede"):
        AgentWorkflowRun(
            workflowRunId="workflow-1",
            createdAtNs=2,
            finalizedAtNs=1,
            status="failed",
            datasetFingerprints={"RNA": "dataset-rna"},
        )

    parent = AgentReportLink.get_example()
    with pytest.raises(ValidationError, match="duplicate"):
        AgentInvocation(
            agentName="parameter_tuning",
            parentReports=[parent, parent],
        )


def test_reports_are_plain_json_under_metadata_only_zarr_group(
    tmp_path: Path,
) -> None:
    path = _create_scarf_store(tmp_path)
    workflow = create_agent_workflow(
        path,
        workflow_run_id="workflow-1",
        dataset_fingerprints={"RNA": "dataset-rna"},
    )
    expected_files = {
        "zarr.json",
        "store.json",
        "runs/workflow-1/workflow.json",
    }

    for report_type, agent_name in REPORT_CASES:
        report = _report(report_type, f"provider-{agent_name}")
        reference = save_agent_report(
            path,
            workflow.workflowRunId,
            report,  # type: ignore[arg-type]
            invocation=_invocation(agent_name),  # type: ignore[arg-type]
            agent_run_id=f"{agent_name}-run",
        )
        record = load_agent_record(path, reference)

        assert reference.agentName == agent_name
        assert record.invocation.inputs == {"fromAssay": "RNA"}
        assert record.invocation.runConfig == AgentRunConfig.get_example()
        assert load_agent_report(path, reference).model_dump(mode="json") == (
            report.model_dump(mode="json")
        )
        expected_files.add(f"runs/workflow-1/{agent_name}/{agent_name}-run/report.json")

    agents_path = path / "agents"
    actual_files = {
        item.relative_to(agents_path).as_posix()
        for item in agents_path.rglob("*")
        if item.is_file()
    }
    assert actual_files == expected_files
    assert not any(
        item.name in {".zarray"} or "c" in item.relative_to(agents_path).parts
        for item in agents_path.rglob("*")
    )
    root = zarr.open_group(str(path), mode="r")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ZarrUserWarning)
        assert "agents" in root.group_keys()
    agents = root["agents"]
    assert isinstance(agents, zarr.Group)
    assert agents.attrs.asdict() == {
        "format": "scarf_agent_reports",
        "format_version": 2,
    }
    metadata = json.loads((agents_path / "zarr.json").read_text(encoding="utf-8"))
    assert metadata["node_type"] == "group"
    assert root.attrs.get("format") != "scarf_agent_reports"


def test_datastore_target_generates_a_missing_dataset_fingerprint(
    tmp_path: Path,
) -> None:
    path = _create_scarf_store(tmp_path, fingerprints={"RNA": None})
    datastore = DataStore(
        str(path),
        assay_types={"RNA": "Assay"},
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=1,
        mem_budget="64M",
    )

    workflow = create_agent_workflow(datastore, workflow_run_id="workflow-1")

    fingerprint = datastore._get_assay("RNA").attrs["dataset_fingerprint"]
    assert workflow.datasetFingerprints == {"RNA": fingerprint}
    assert load_agent_workflow(datastore, workflow.workflowRunId) == workflow


@pytest.mark.parametrize(
    "supplied",
    [
        {"RNA": "dataset-rna"},
        {"RNA": "dataset-rna", "ATAC": "dataset-atac", "ADT": "unknown"},
        {"RNA": "wrong", "ATAC": "dataset-atac"},
        {"RNA": "dataset-rna", "ATAC": ""},
    ],
)
def test_workflow_creation_requires_exact_all_assay_fingerprints(
    tmp_path: Path,
    supplied: dict[str, str],
) -> None:
    path = _create_scarf_store(
        tmp_path,
        fingerprints={"RNA": "dataset-rna", "ATAC": "dataset-atac"},
    )

    with pytest.raises(ValueError, match="do not match"):
        create_agent_workflow(
            path,
            workflow_run_id="workflow-1",
            dataset_fingerprints=supplied,
        )

    assert not (path / "agents").exists()


def test_load_fails_closed_when_dataset_binding_changes(tmp_path: Path) -> None:
    path = _create_scarf_store(
        tmp_path,
        fingerprints={"RNA": "dataset-rna", "ATAC": "dataset-atac"},
    )
    workflow = create_agent_workflow(
        path,
        workflow_run_id="workflow-1",
        dataset_fingerprints={"ATAC": "dataset-atac", "RNA": "dataset-rna"},
    )
    assert workflow.datasetFingerprints == {
        "ATAC": "dataset-atac",
        "RNA": "dataset-rna",
    }

    root = zarr.open_group(str(path), mode="r+")
    root["ATAC"].attrs["dataset_fingerprint"] = "changed"

    with pytest.raises(ValueError, match="do not match"):
        load_agent_workflow(path, workflow.workflowRunId)
    with pytest.raises(ValueError, match="do not match"):
        list_agent_workflows(path, include_incomplete=True)


def test_path_target_rejects_missing_assay_fingerprints(tmp_path: Path) -> None:
    path = _create_scarf_store(
        tmp_path,
        fingerprints={"RNA": "dataset-rna", "ATAC": None},
    )

    with pytest.raises(ValueError, match="missing.*ATAC"):
        create_agent_workflow(path, workflow_run_id="workflow-1")


def test_lineage_distinguishes_parallel_reports_and_persists_typed_handoffs(
    tmp_path: Path,
) -> None:
    path = _create_scarf_store(tmp_path)
    create_agent_workflow(path, workflow_run_id="workflow-1")

    experimental_1 = _experimental_report("experimental-provider-1")
    experimental_2 = _experimental_report("experimental-provider-2")
    experimental_ref_1 = save_agent_report(
        path,
        "workflow-1",
        experimental_1,
        invocation=_invocation("experimental_context"),
        agent_run_id="e1",
    )
    experimental_ref_2 = save_agent_report(
        path,
        "workflow-1",
        experimental_2,
        invocation=_invocation("experimental_context"),
        agent_run_id="e2",
    )
    experimental_link_1 = AgentReportLink.from_reference(experimental_ref_1)
    experimental_link_2 = AgentReportLink.from_reference(experimental_ref_2)

    tuning_1 = _tuning_report("tuning-provider-1")
    tuning_2 = _tuning_report("tuning-provider-2")
    tuning_ref_1 = save_agent_report(
        path,
        "workflow-1",
        tuning_1,
        invocation=_invocation(
            "parameter_tuning",
            parents=[experimental_link_1],
            experimentalTuningHandoff=(experimental_1.to_parameter_tuning_handoff()),
        ),
        agent_run_id="t1",
    )
    tuning_ref_2 = save_agent_report(
        path,
        "workflow-1",
        tuning_2,
        invocation=_invocation(
            "parameter_tuning",
            parents=[experimental_link_2],
            experimentalTuningHandoff=(experimental_2.to_parameter_tuning_handoff()),
        ),
        agent_run_id="t2",
    )
    tuning_link_1 = AgentReportLink.from_reference(tuning_ref_1)

    biology_ref = save_agent_report(
        path,
        "workflow-1",
        BiologicalInterpretationReport.get_example(),
        invocation=_invocation(
            "biological_interpretation",
            parents=[experimental_link_1, tuning_link_1],
            experimentalBiologyHandoff=experimental_1.to_biological_handoff(),
            tuningBiologyHandoff=tuning_1.to_biological_handoff(),
        ),
        agent_run_id="b1",
    )

    record_t1 = load_agent_record(path, tuning_ref_1)
    record_t2 = load_agent_record(path, tuning_ref_2)
    record_b1 = load_agent_record(path, biology_ref)
    assert record_t1.invocation.parentReports == [experimental_link_1]
    assert record_t2.invocation.parentReports == [experimental_link_2]
    assert record_b1.invocation.parentReports == [
        experimental_link_1,
        tuning_link_1,
    ]
    assert record_b1.reference.parentReports == record_b1.invocation.parentReports
    assert (
        record_b1.invocation.experimentalBiologyHandoff
        == experimental_1.to_biological_handoff()
    )
    assert record_b1.invocation.tuningBiologyHandoff == tuning_1.to_biological_handoff()


def test_biology_may_cite_context_without_a_treatment_handoff(
    tmp_path: Path,
) -> None:
    path = _create_scarf_store(tmp_path)
    create_agent_workflow(path, workflow_run_id="workflow-1")
    experimental = _experimental_report("experimental-provider")
    experimental_ref = save_agent_report(
        path,
        "workflow-1",
        experimental,
        invocation=_invocation("experimental_context"),
        agent_run_id="e1",
    )
    experimental_link = AgentReportLink.from_reference(experimental_ref)
    tuning = _tuning_report("tuning-provider")
    tuning_ref = save_agent_report(
        path,
        "workflow-1",
        tuning,
        invocation=_invocation(
            "parameter_tuning",
            parents=[experimental_link],
            experimentalTuningHandoff=experimental.to_parameter_tuning_handoff(),
        ),
        agent_run_id="t1",
    )
    tuning_link = AgentReportLink.from_reference(tuning_ref)

    biology_ref = save_agent_report(
        path,
        "workflow-1",
        BiologicalInterpretationReport.get_example(),
        invocation=_invocation(
            "biological_interpretation",
            parents=[experimental_link, tuning_link],
            tuningBiologyHandoff=tuning.to_biological_handoff(),
        ),
        agent_run_id="b1",
    )

    record = load_agent_record(path, biology_ref)
    assert record.invocation.parentReports == [experimental_link, tuning_link]
    assert record.invocation.experimentalBiologyHandoff is None


def test_lineage_rejects_unknown_cross_workflow_and_changed_parent_links(
    tmp_path: Path,
) -> None:
    path = _create_scarf_store(tmp_path)
    create_agent_workflow(path, workflow_run_id="workflow-1")
    create_agent_workflow(path, workflow_run_id="workflow-2")
    experimental = _experimental_report("experimental-provider")
    reference = save_agent_report(
        path,
        "workflow-1",
        experimental,
        invocation=_invocation("experimental_context"),
        agent_run_id="e1",
    )
    link = AgentReportLink.from_reference(reference)
    handoff = experimental.to_parameter_tuning_handoff()

    invalid_links = (
        link.model_copy(update={"agentRunId": "unknown"}),
        link.model_copy(update={"workflowRunId": "workflow-2"}),
        link.model_copy(update={"contentSha256": "f" * 64}),
    )
    for index, invalid_link in enumerate(invalid_links):
        with pytest.raises(ValueError, match="parent|Parent"):
            save_agent_report(
                path,
                "workflow-1",
                _tuning_report(f"tuning-provider-{index}"),
                invocation=_invocation(
                    "parameter_tuning",
                    parents=[invalid_link],
                    experimentalTuningHandoff=handoff,
                ),
                agent_run_id=f"invalid-{index}",
            )


def test_typed_handoffs_must_match_their_cited_parent_reports(
    tmp_path: Path,
) -> None:
    path = _create_scarf_store(tmp_path)
    create_agent_workflow(path, workflow_run_id="workflow-1")
    experimental = _experimental_report("experimental-provider")
    experimental_ref = save_agent_report(
        path,
        "workflow-1",
        experimental,
        invocation=_invocation("experimental_context"),
        agent_run_id="e1",
    )
    experimental_link = AgentReportLink.from_reference(experimental_ref)

    with pytest.raises(ValueError, match="experimentalTuningHandoff.*required"):
        save_agent_report(
            path,
            "workflow-1",
            _tuning_report("missing-handoff"),
            invocation=_invocation(
                "parameter_tuning",
                parents=[experimental_link],
            ),
            agent_run_id="missing-handoff",
        )
    with pytest.raises(ValueError, match="does not descend"):
        save_agent_report(
            path,
            "workflow-1",
            _tuning_report("wrong-handoff"),
            invocation=_invocation(
                "parameter_tuning",
                parents=[experimental_link],
                experimentalTuningHandoff=ExperimentalTuningHandoff(batchAction="skip"),
            ),
            agent_run_id="wrong-handoff",
        )

    tuning = _tuning_report("tuning-provider")
    tuning_ref = save_agent_report(
        path,
        "workflow-1",
        tuning,
        invocation=_invocation("parameter_tuning"),
        agent_run_id="t1",
    )
    with pytest.raises(ValueError, match="tuningBiologyHandoff.*does not match"):
        save_agent_report(
            path,
            "workflow-1",
            BiologicalInterpretationReport.get_example(),
            invocation=_invocation(
                "biological_interpretation",
                parents=[AgentReportLink.from_reference(tuning_ref)],
                tuningBiologyHandoff=TuningBiologyHandoff(),
            ),
            agent_run_id="wrong-biology-handoff",
        )


def test_invocation_is_required_and_must_match_report_type(tmp_path: Path) -> None:
    path = _create_scarf_store(tmp_path)
    create_agent_workflow(path, workflow_run_id="workflow-1")

    with pytest.raises(TypeError, match="invocation"):
        save_agent_report(
            path,
            "workflow-1",
            DataEnrichmentReport.get_example(),
            invocation=None,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="agentName"):
        save_agent_report(
            path,
            "workflow-1",
            DataEnrichmentReport.get_example(),
            invocation=_invocation("experimental_context"),
        )
    with pytest.raises(ValueError, match="inputs"):
        save_agent_report(
            path,
            "workflow-1",
            DataEnrichmentReport.get_example(),
            invocation=AgentInvocation(agentName="data_enrichment"),
        )


def test_workspace_records_are_physically_and_logically_isolated(
    tmp_path: Path,
) -> None:
    path = _create_workspace_store(tmp_path)
    workflow_a = create_agent_workflow(
        path,
        workflow_run_id="workflow-a",
        workspace="workspace_a",
    )
    workflow_b = create_agent_workflow(
        path,
        workflow_run_id="workflow-b",
        workspace="workspace_b",
    )
    reference_a = save_agent_report(
        path,
        workflow_a.workflowRunId,
        DataEnrichmentReport.get_example(),
        invocation=_invocation("data_enrichment"),
        agent_run_id="run-a",
        workspace="workspace_a",
    )
    reference_b = save_agent_report(
        path,
        workflow_b.workflowRunId,
        DataEnrichmentReport.get_example(),
        invocation=_invocation("data_enrichment"),
        agent_run_id="run-b",
        workspace="workspace_b",
    )

    assert reference_a.workspace == "workspace_a"
    assert reference_b.workspace == "workspace_b"
    assert _report_path(path, reference_a, workspace="workspace_a").is_file()
    assert _report_path(path, reference_b, workspace="workspace_b").is_file()
    assert not (path / "agents").exists()
    assert [
        item.workflowRunId
        for item in list_agent_workflows(
            path,
            workspace="workspace_a",
            include_incomplete=True,
        )
    ] == ["workflow-a"]
    assert [
        item.workflowRunId
        for item in list_agent_workflows(
            path,
            workspace="workspace_b",
            include_incomplete=True,
        )
    ] == ["workflow-b"]
    with pytest.raises(KeyError, match="Unknown agent workflow"):
        load_agent_report(path, reference_a, workspace="workspace_b")


def test_datastore_rejects_an_explicit_workspace_mismatch(tmp_path: Path) -> None:
    path = _create_scarf_store(tmp_path)
    datastore = DataStore(
        str(path),
        assay_types={"RNA": "Assay"},
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=1,
        mem_budget="64M",
    )

    with pytest.raises(ValueError, match="workspace does not match"):
        create_agent_workflow(
            datastore,
            workflow_run_id="workflow-1",
            workspace="workspace_a",
        )


def test_workflow_lifecycle_controls_listing_and_future_writes(
    tmp_path: Path,
) -> None:
    path = _create_scarf_store(tmp_path)
    running = create_agent_workflow(path, workflow_run_id="running-workflow")
    completed = create_agent_workflow(path, workflow_run_id="completed-workflow")
    failed = create_agent_workflow(path, workflow_run_id="failed-workflow")
    abandoned = create_agent_workflow(path, workflow_run_id="abandoned-workflow")

    assert list_agent_workflows(path) == []
    assert {
        item.workflowRunId
        for item in list_agent_workflows(path, include_incomplete=True)
    } == {
        running.workflowRunId,
        completed.workflowRunId,
        failed.workflowRunId,
        abandoned.workflowRunId,
    }
    with pytest.raises(ValueError, match="at least one report"):
        finalize_agent_workflow(path, completed.workflowRunId, status="completed")

    reference = save_agent_report(
        path,
        completed.workflowRunId,
        DataEnrichmentReport.get_example(),
        invocation=_invocation("data_enrichment"),
        agent_run_id="run-1",
    )
    workflow_file = path / "agents/runs/completed-workflow/workflow.json"
    original_workflow = workflow_file.read_bytes()
    completed_result = finalize_agent_workflow(
        path,
        completed.workflowRunId,
        status="completed",
        message="analysis finished",
    )
    failed_result = finalize_agent_workflow(
        path,
        failed.workflowRunId,
        status="failed",
        message="provider failed",
    )
    abandoned_result = finalize_agent_workflow(
        path,
        abandoned.workflowRunId,
        status="abandoned",
    )

    assert workflow_file.read_bytes() == original_workflow
    assert completed_result.status == "completed"
    assert completed_result.finalizedAtNs > completed_result.createdAtNs
    assert completed_result.finalizationMessage == "analysis finished"
    assert failed_result.status == "failed"
    assert failed_result.finalizationMessage == "provider failed"
    assert abandoned_result.status == "abandoned"
    assert load_agent_workflow(path, running.workflowRunId).status == "running"
    assert {item.status for item in list_agent_workflows(path)} == {
        "completed",
        "failed",
        "abandoned",
    }
    finalization = json.loads(
        (path / "agents/runs/completed-workflow/finalization.json").read_text()
    )
    assert finalization["status"] == "completed"
    assert finalization["message"] == "analysis finished"

    with pytest.raises(FileExistsError, match="already 'completed'"):
        finalize_agent_workflow(path, completed.workflowRunId, status="failed")
    with pytest.raises(RuntimeError, match="completed"):
        save_agent_report(
            path,
            completed.workflowRunId,
            DataEnrichmentReport.get_example(),
            invocation=_invocation("data_enrichment"),
            agent_run_id="run-2",
        )
    assert load_agent_report(path, reference) == DataEnrichmentReport.get_example()

    finalization["finalizedAtNs"] = completed_result.createdAtNs - 1
    (path / "agents/runs/completed-workflow/finalization.json").write_text(
        json.dumps(finalization),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cannot precede"):
        load_agent_workflow(path, completed.workflowRunId)


def test_report_paths_are_immutable_and_corruption_fails_closed(
    tmp_path: Path,
) -> None:
    path = _create_scarf_store(tmp_path)
    create_agent_workflow(path, workflow_run_id="workflow-1")
    report = DataEnrichmentReport.get_example()
    reference = save_agent_report(
        path,
        "workflow-1",
        report,
        invocation=_invocation("data_enrichment"),
        agent_run_id="run-1",
    )
    report_path = _report_path(path, reference)
    original = report_path.read_bytes()

    with pytest.raises(FileExistsError, match="already exists"):
        save_agent_report(
            path,
            "workflow-1",
            report.model_copy(update={"limitations": ["different payload"]}),
            invocation=_invocation("data_enrichment"),
            agent_run_id="run-1",
        )
    assert report_path.read_bytes() == original

    payload = json.loads(original)
    payload["report"]["limitations"] = ["tampered"]
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_agent_report(path, reference)


def test_malformed_json_and_legacy_zarr_agents_group_are_rejected(
    tmp_path: Path,
) -> None:
    path = _create_scarf_store(tmp_path)
    create_agent_workflow(path, workflow_run_id="workflow-1")
    workflow_path = path / "agents/runs/workflow-1/workflow.json"
    workflow_path.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        load_agent_workflow(path, "workflow-1")

    legacy_path = tmp_path / "legacy.zarr"
    legacy_root = zarr.open_group(str(legacy_path), mode="w", zarr_format=3)
    _populate_scarf_group(legacy_root, fingerprints={"RNA": "dataset-rna"})
    legacy_root.create_group(
        "agents",
        attributes={"format": "scarf_agent_reports", "format_version": 1},
    )
    with pytest.raises(ValueError, match="version 1.*migration"):
        create_agent_workflow(legacy_path, workflow_run_id="workflow-1")


@pytest.mark.parametrize(
    "invalid_id",
    ["", ".", "..", "../x", "a/b", r"a\b", "UPPER", " leading", "x\x00"],
)
def test_workflow_run_ids_are_validated_before_writing(
    tmp_path: Path,
    invalid_id: str,
) -> None:
    path = _create_scarf_store(tmp_path)
    with pytest.raises(ValueError, match="safe path component"):
        create_agent_workflow(path, workflow_run_id=invalid_id)
    assert not (path / "agents").exists()


def test_save_rejects_unknown_report_model(tmp_path: Path) -> None:
    path = _create_scarf_store(tmp_path)
    create_agent_workflow(path, workflow_run_id="workflow-1")

    with pytest.raises(TypeError, match="four Scarf agent report models"):
        save_agent_report(
            path,
            "workflow-1",
            AnyReport.get_example(),  # type: ignore[arg-type]
            invocation=_invocation("data_enrichment"),
        )


def test_generated_workflow_and_agent_run_ids_are_safe(tmp_path: Path) -> None:
    path = _create_scarf_store(tmp_path)
    workflow = create_agent_workflow(path)
    reference = save_agent_report(
        path,
        workflow.workflowRunId,
        DataEnrichmentReport.get_example(),
        invocation=_invocation("data_enrichment"),
    )

    assert re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,127}", workflow.workflowRunId)
    assert re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,127}", reference.agentRunId)

---
description: Run Scarf's resumable automated agent orchestrator on a 5K PBMC dataset.
jupytext:
  cell_metadata_filter: tags
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(agent_workflow)=

# Run the automated agent workflow

This tutorial sends a 10x H5 dataset and one study-context paragraph to
`AgentOrchestrator`. The orchestrator runs Scarf's four bounded agents, owns the exact operation
order, persists every handoff, and returns exact final artifact references. The agents select from
executor-authorized operations and parameters. They do not write exploratory code.

```{mermaid}
flowchart LR
    A[Input dataset and study context] --> B[Ingest]
    B --> C[Data Enrichment]
    C --> D[HTO demultiplexing when present]
    D --> E[Experimental Context]
    E --> F[Preprocessing plan]
    F --> G[Modality preprocessing]
    G --> H[Parameter Tuning]
    H --> I[UMAP, clusters, and markers]
    I --> J[Biological Interpretation]
    J --> K[Persisted reports and local HTML]
```

The committed documentation build uses one scripted Pydantic AI `FunctionModel`. It exercises the
real tools, validators, preprocessing, candidate execution, finalization, persistence, and resume
path without an API key. Its biological labels remain low-confidence marker-linked hypotheses. A
live-provider configuration is shown at the end.

## 1. Download the raw teaching dataset

Install the optional agent dependencies before running this workflow outside the documentation
environment:

```console
pip install "scarf[agent]"
```

The documentation run converts a raw H5 file into a separate teaching store. The explicit
`overwrite` direction is safe here because `agent_workflow.zarr` is a disposable derived target
owned by this tutorial. Omit it in ordinary work unless replacing that exact destination is
intentional.

```{code-cell} ipython3
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import scarf
from scarf.agent import (
    AgentOrchestrator,
    AgentRunConfig,
    AutomatedWorkflowConfig,
    AutomatedWorkflowRequest,
    AutomatedWorkflowResumeRequest,
    generate_agent_report,
    load_agent_report,
)
from scarf.agent.orchestrator import artifact_model_to_ref

scarf.configure_output(level="WARNING", progress=False)

source_path = scarf.cytebase.connect("scarf_docs").download(
    "tenx_5K_pbmc_rnaseq/data.h5",
    destination="scarf_datasets",
)[0]
zarr_path = source_path.with_name("agent_workflow.zarr")

study_context = (
    "This is a human 10x Genomics 5K PBMC 3-prime gene-expression dataset "
    "from peripheral blood collected from one healthy donor. The goal is "
    "unsupervised identification and characterization of the major immune-cell "
    "populations. No treatment comparison, technical batch covariate, paired "
    "modality, or independent replication metadata is available. Do not invent "
    "absent design variables or report treatment effects."
)

{"source": source_path.name, "destination": zarr_path.name}
```

The hidden setup below routes each model request by its available tools. Every structured response
is assembled from the exact tool result, so a fabricated assay, feature family, candidate, cluster,
or evidence identifier still fails the production validator.

```{code-cell} ipython3
:tags: [remove-cell]

import re
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from scarf.agent.biological_interpretation import (
    BiologicalInterpretationReport,
    ClusterCompositionEvidence,
    ClusterInterpretation,
    ClusterMarkerBatchEvidence,
)
from scarf.agent.data_enrichment import (
    AssayFeatureInspectionBatch,
    DataEnrichmentReport,
    FeatureSelectionPolicy,
    StudyContextSummary,
)
from scarf.agent.experimental_context import (
    BatchCorrectionPlan,
    CellQcPlan,
    CovariateEvidence,
    ExperimentalContextDecision,
)
from scarf.agent.parameter_tuning import (
    FinalGraphSelection,
    ParameterTuningReport,
)


def _prompt_text(messages: list[ModelMessage]) -> str:
    return "\n".join(
        part.content
        for message in messages
        for part in message.parts
        if isinstance(getattr(part, "content", None), str)
    )


def _tool_result(
    messages: list[ModelMessage],
    tool_name: str,
    model_type: Any,
) -> Any:
    for message in reversed(messages):
        for part in reversed(message.parts):
            if isinstance(part, ToolReturnPart) and part.tool_name == tool_name:
                if isinstance(part.content, model_type):
                    return part.content
                if isinstance(part.content, str):
                    return model_type.model_validate_json(part.content)
                return model_type.model_validate(part.content)
    raise AssertionError(f"Missing tool return {tool_name!r}")


def _tool_call(name: str, args: dict[str, Any] | None = None) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(tool_name=name, args=args or {})])


def _structured_output(info: AgentInfo, value: Any) -> ModelResponse:
    payload = value.model_dump() if hasattr(value, "model_dump") else value
    return _tool_call(info.output_tools[0].name, payload)


def _scripted_workflow_model() -> tuple[FunctionModel, dict[str, int]]:
    state = {
        "enrichment": 0,
        "context": 0,
        "parameter": 0,
        "biology": 0,
        "requests": 0,
    }

    async def reply(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        state["requests"] += 1
        tools = {tool.name for tool in info.function_tools}

        if "inspect_assay_features_batch" in tools or state["enrichment"] == 1:
            if state["enrichment"] == 0:
                state["enrichment"] = 1
                return _tool_call("inspect_assay_features_batch")

            batch = _tool_result(
                messages,
                "inspect_assay_features_batch",
                AssayFeatureInspectionBatch,
            )
            policies = []
            for inspection in batch.inspections:
                species_observed = inspection.species != "unknown"
                policy_evidence = list(inspection.evidenceIds)
                if not species_observed:
                    policy_evidence.append("context:study")
                policies.append(
                    FeatureSelectionPolicy(
                        assay=inspection.assay,
                        species=(
                            inspection.species
                            if species_observed
                            else "homo_sapiens"
                        ),
                        speciesConfidence="high" if species_observed else "medium",
                        speciesRationale=(
                            inspection.speciesReason
                            or "The exact study paragraph identifies a human sample."
                        ),
                        excludeFamilies=[
                            family.family
                            for family in inspection.families
                            if family.count > 0 and family.defaultExclude is True
                        ],
                        protectFamilies=[
                            family.family
                            for family in inspection.families
                            if family.count > 0 and family.defaultExclude is False
                        ],
                        rationale=(
                            "Exclude observed technical families and preserve "
                            "observed protected families."
                        ),
                        evidenceIds=list(dict.fromkeys(policy_evidence)),
                    )
                )
            state["enrichment"] = 2
            return _structured_output(
                info,
                DataEnrichmentReport(
                    status="done",
                    studyContextSummary=StudyContextSummary(
                        organismReferences=["human"],
                        tissueReferences=["peripheral blood"],
                        experimentalReferences=[
                            "10x Genomics 5K PBMC 3-prime gene-expression dataset"
                        ],
                        analysisIntentReferences=[
                            "unsupervised identification and characterization of "
                            "the major immune-cell populations"
                        ],
                    ),
                    policies=policies,
                ),
            )

        if tools.intersection(
            {
                "inspect_cell_covariates",
                "analyze_experimental_design",
                "score_current_representation",
            }
        ) or state["context"] in {1, 2}:
            if state["context"] == 0:
                state["context"] = 1
                return _tool_call("inspect_cell_covariates")
            if state["context"] == 1:
                state["context"] = 2
                return _tool_call(
                    "analyze_experimental_design",
                    {
                        "column_domains": {},
                        "coefficients_of_interest": [],
                        "units_of_inference": {},
                        "batch_columns": [],
                    },
                )

            design = _tool_result(
                messages,
                "analyze_experimental_design",
                CovariateEvidence,
            )
            profile = next(
                value
                for value in design.qcProfiles
                if value.action == "globalGaussian"
            )
            evidence_id = profile.evidenceId
            state["context"] = 3
            return _structured_output(
                info,
                ExperimentalContextDecision(
                    batchCorrection=BatchCorrectionPlan(
                        action="skip",
                        rationale="No trusted technical batch column was supplied.",
                        evidenceIds=[evidence_id],
                    ),
                    cellQc=CellQcPlan(
                        action=profile.action,
                        profileId=profile.profileId,
                        driverAssay=profile.driverAssay,
                        driverAssayType=profile.driverAssayType,
                        attributes=profile.attributes,
                        artifactMetrics=profile.artifactMetrics,
                        rationale="Apply the bounded global RNA QC profile.",
                        evidenceIds=[evidence_id],
                    ),
                    rationale="No experimental covariates were supplied.",
                    evidenceIds=[evidence_id],
                ),
            )

        if tools.intersection(
            {"inspect_cluster_composition", "inspect_cluster_markers_batch"}
        ) or state["biology"]:
            if state["biology"] == 0:
                state["biology"] = 1
                return _tool_call("inspect_cluster_composition")
            if state["biology"] == 1:
                composition = _tool_result(
                    messages,
                    "inspect_cluster_composition",
                    ClusterCompositionEvidence,
                )
                state["biology"] = 2
                return _tool_call(
                    "inspect_cluster_markers_batch",
                    {"cluster_ids": list(composition.clusterCounts)},
                )

            marker_batch = _tool_result(
                messages,
                "inspect_cluster_markers_batch",
                ClusterMarkerBatchEvidence,
            )
            interpretations = []
            for cluster in marker_batch.clusters:
                if cluster.evidenceId and cluster.markers:
                    marker = cluster.markers[0]
                    marker_name = marker.featureName or marker.featureId
                    interpretations.append(
                        ClusterInterpretation(
                            clusterId=cluster.clusterId,
                            proposedIdentity=f"{marker_name}-high RNA state",
                            identityIsHypothesis=True,
                            confidence="low",
                            rationale=(
                                "The returned marker panel is led by "
                                f"{marker_name}."
                            ),
                            evidenceIds=[cluster.evidenceId],
                        )
                    )
            state["biology"] = 3
            return _structured_output(
                info,
                BiologicalInterpretationReport(
                    status="done",
                    clusterInterpretations=interpretations,
                    evidenceIds=[item.evidenceIds[0] for item in interpretations],
                    limitations=[
                        "The scripted documentation model returns marker-linked "
                        "hypotheses, not validated cell identities."
                    ],
                    stopReason=(
                        "Every cluster with returned marker evidence was reviewed."
                    ),
                ),
            )

        prompt = _prompt_text(messages)
        if state["parameter"] == 0:
            match = re.search(
                r'"candidateId"\s*:\s*"([A-Za-z0-9_]+)"',
                prompt,
            )
            if match is None:
                raise AssertionError("The parameter prompt lacks a candidate ID")
            candidate_id = match.group(1)
            evidence_id = f"candidate:{candidate_id}:clusters"
            assay_report = ParameterTuningReport(
                status="done",
                recommendedCandidateId=candidate_id,
                confidence="high",
                rationale="The only authorized native branch is eligible.",
                evidenceIds=[evidence_id],
                stopReason="The bounded one-candidate screen completed.",
            )
            state["parameter"] = 1
            return _structured_output(
                info,
                ParameterTuningReport(
                    status="done",
                    assayReports={"RNA": assay_report},
                    rationale="The RNA native screen completed.",
                    evidenceIds=[evidence_id],
                    stopReason="Native selection completed.",
                ),
            )

        match = re.search(
            r'"optionId"\s*:\s*"(native:RNA:([A-Za-z0-9_]+))"',
            prompt,
        )
        if match is None:
            raise AssertionError("The final-selection prompt lacks a native option")
        option_id, candidate_id = match.groups()
        evidence_id = f"native:RNA:candidate:{candidate_id}:clusters"
        state["parameter"] = 2
        return _structured_output(
            info,
            FinalGraphSelection(
                status="done",
                selectedOptionId=option_id,
                graphMethod="native",
                nativeAssay="RNA",
                nativeCandidateId=candidate_id,
                markerAssay="RNA",
                confidence="high",
                rationale="The sole eligible native graph is selected.",
                evidenceIds=[evidence_id],
            ),
        )

    return FunctionModel(reply), state

```

## 2. Configure one bounded teaching branch

The production defaults screen five candidates for the primary assay and may request one
refinement. This documentation run uses one native RNA candidate, no refinement, and no Harmony.
The smaller search exercises the same executor and persistence path while keeping the build
bounded. Harmony would be eligible only if Experimental Context returned exact safe batch evidence.

```{code-cell} ipython3
model, model_state = _scripted_workflow_model()
config = AutomatedWorkflowConfig(
    primaryInitialCandidates=1,
    secondaryInitialCandidates=1,
    maxRefinedCandidatesPerAssay=0,
    maxHarmonyCandidatesPerAssay=0,
    integrationResolutionCandidates=1,
    maxCandidateBranches=1,
    minClusterCells=2,
    agentRunConfig=AgentRunConfig(
        requestLimit=5,
        toolCallLimit=5,
    ),
)
orchestrator = AgentOrchestrator(model, config=config)
request = AutomatedWorkflowRequest(
    sourcePath=str(source_path),
    zarrPath=str(zarr_path),
    studyContext=study_context,
    allowAssumptions=False,
    primaryAssay="RNA",
    markerAssay="RNA",
    analysisAssays=["RNA"],
    ingestDirections={"overwrite": True, "defaultAssay": "RNA"},
)

{
    "initial_candidates": config.primaryInitialCandidates,
    "refinement_candidates": config.maxRefinedCandidatesPerAssay,
    "harmony_candidates": config.maxHarmonyCandidatesPerAssay,
    "allow_assumptions": request.allowAssumptions,
}
```

## 3. Run to the persisted approval checkpoint

With `allowAssumptions=False`, the orchestrator persists the exact proposed plan before asking the
caller to approve it. Data Enrichment and Experimental Context have already completed at this
point. The documentation captures the normal report-path printout so its output does not contain a
random workflow identifier.

```{code-cell} ipython3
with redirect_stdout(StringIO()):
    result = orchestrator.run(request)

if (
    result.status != "needsInput"
    or result.currentStage != "preprocessing_plan"
    or result.preprocessingPlan is None
    or result.workflowRun is None
    or result.zarrPath is None
):
    raise RuntimeError(f"Unexpected workflow result: {result.status}, {result.notes}")

question = result.needsInput.questions[0]
plan = result.preprocessingPlan
{
    "status": result.status,
    "stage": result.currentStage,
    "question_id": question.questionId,
    "primary_assay": plan.primaryAssay,
    "marker_assay": plan.markerAssay,
    "cell_qc": plan.cellQc.action,
    "routes": [
        {
            "assay": assay.assay,
            "features": assay.featureMethod,
            "reduction": assay.reductionMethod,
        }
        for assay in plan.assays
    ],
}
```

The plan checksum binds the approval to this exact plan. A different value is rejected rather than
approving whichever plan happens to be current.

## 4. Resume the same workflow

Only a running persisted workflow can resume. The answer uses the question identifier and checksum
returned above. Completed stages and artifacts are validated and reused rather than executed again.

```{code-cell} ipython3
with redirect_stdout(StringIO()):
    result = orchestrator.resume(
        AutomatedWorkflowResumeRequest(
            zarrPath=result.zarrPath,
            workflowRunId=result.workflowRun.workflowRunId,
            workspace=result.workflowRun.workspace,
            answers={"approvePlanChecksum": plan.planChecksum},
        )
    )

if result.status != "completed" or result.finalAnalysis is None:
    raise RuntimeError(f"Workflow stopped at {result.currentStage}: {result.notes}")

{
    "status": result.status,
    "stage": result.currentStage,
    "agent_reports": [ref.agentName for ref in result.reportReferences],
    "model_requests": model_state["requests"],
    "graph_method": result.finalAnalysis.graphMethod,
    "marker_assay": result.finalAnalysis.markerAssay,
}
```

The single scripted provider is called by all four agents. Deterministic operations, such as HTO
routing, preprocessing, candidate execution, promotion, UMAP, clustering, marker search, and
persistence, do not require separate model requests.

## 5. Review parameter evidence and agent reports

The parameter agent receives executor-produced metrics for candidates that have already run. It
does not generate Scarf code. Each candidate follows the explicit reduction, optional Harmony,
ANN, neighbours, connectivity, Leiden, and metric chain. The final selected branch is replayed with
state updates and checked against the evaluated immutable references.

```{code-cell} ipython3
reports = {
    reference.agentName: load_agent_report(result.zarrPath, reference)
    for reference in result.reportReferences
}
parameter_report = reports["parameter_tuning"]

candidate_metrics = []
for assay, assay_report in parameter_report.assayReports.items():
    for index, evaluation in enumerate(assay_report.evaluations, start=1):
        candidate_metrics.append(
            {
                "assay": assay,
                "candidate": index,
                "dimensions": evaluation.parameters.dimensions,
                "resolution": evaluation.parameters.leidenResolution,
                "neighbors": evaluation.parameters.neighborsK,
                "eligible": evaluation.eligible,
                "clusters": evaluation.metrics.nClusters,
                "smallest_cluster": evaluation.metrics.minClusterCells,
                "graph_silhouette": evaluation.metrics.graphSilhouetteMedian,
            }
        )

{
    "candidates": candidate_metrics,
    "stop_reason": parameter_report.stopReason,
    "report_statuses": {
        name: report.status for name, report in reports.items()
    },
}
```

This one-candidate teaching run demonstrates execution and selection, not a broad parameter search.
The default configuration evaluates more initial candidates and may execute one evidence-driven
refinement. Harmony is added only when the exact Experimental Context handoff authorizes a matched
comparison.

## 6. Plot the exact final UMAP and inspect markers

`FinalAnalysisHandoff` separates graph ownership from marker-assay ownership and contains the exact
selection, graph, clusters, UMAP, and marker references used by Biological Interpretation. The
plotting call consumes those references directly; no coordinates or labels are copied into live
metadata columns.

```{code-cell} ipython3
final = result.finalAnalysis
if (
    final.cellSelection is None
    or final.clusters is None
    or final.umap is None
    or final.markers is None
):
    raise RuntimeError("The completed final handoff is missing required artifacts")

final_store = scarf.DataStore(
    result.zarrPath,
    default_assay=final.primaryAssay,
    min_features_per_cell=-1,
    mito_pattern="",
    ribo_pattern="",
    zarr_mode="r",
    workspace=result.workflowRun.workspace,
    nthreads=2,
)
cell_selection_ref = artifact_model_to_ref(final.cellSelection)
cluster_ref = artifact_model_to_ref(final.clusters)
umap_ref = artifact_model_to_ref(final.umap)
marker_ref = artifact_model_to_ref(final.markers)

final_store.plots.embedding(
    layout=umap_ref,
    color_by=cluster_ref,
    legend_loc="on_data",
    frame="none",
)
```

UMAP is a presentation artifact. The tuning agent compares graph and metadata metrics, not visual
appearance, and the orchestrator does not train several UMAPs to choose the most attractive one.

```{code-cell} ipython3
marker_table = final_store.get_markers(
    marker=marker_ref,
    group_id=None,
    min_score=-1,
    min_frac_exp=-1,
)
marker_table.sort_values(
    ["group_id", "score"],
    ascending=[True, False],
).groupby("group_id", sort=True).head(2)[
    ["group_id", "feature_name", "score", "frac_exp"]
].head(12)
```

Marker scores are cell-level descriptive evidence. They are not replicate-aware differential
expression, and the scripted identities remain hypotheses.

## 7. Open or regenerate the local HTML report

A completed local workflow first persists its terminal result and then writes a replaceable HTML
view under `agents/runs/<workflowRunId>/report/index.html`. Calling
`generate_agent_report()` regenerates that view from the persisted workflow and existing analysis
artifacts. It does not train another UMAP.

```{code-cell} ipython3
report_path = generate_agent_report(
    result.zarrPath,
    result.workflowRun.workflowRunId,
    workspace=result.workflowRun.workspace,
)
display_path = str(report_path.relative_to(Path(result.zarrPath).parent)).replace(
    result.workflowRun.workflowRunId,
    "<workflowRunId>",
)

{
    "report": display_path,
    "exists": report_path.is_file(),
    "final_artifact_kinds": {
        "selection": cell_selection_ref.kind,
        "clusters": cluster_ref.kind,
        "umap": umap_ref.kind,
        "markers": marker_ref.kind,
    },
}
```

## Pauses, failures, and other input formats

`needsInput` keeps the workflow running. Inspect every returned question and supply only grounded
answers. `failed` and `abandoned` are terminal. An ingest question can occur before a persisted
workflow exists; update `ingestDirections` and call `run()` again in that case. A running workflow
can also be finalized as abandoned with `orchestrator.cancel()`.

For another new local H5 or H5AD input, provide a destination that does not yet exist:

```python
request = AutomatedWorkflowRequest(
    sourcePath="study.h5ad",
    zarrPath="study.zarr",
    studyContext="One paragraph describing the study, design, and analysis intent.",
    allowAssumptions=False,
)
result = AgentOrchestrator(model).run(request)
```

For an existing Zarr input, omit `zarrPath` or set it to the same location. Its current `I`
selection is preserved and snapshotted. A workspace may be supplied only for an existing Zarr
input.

## Use a live model

Replace the scripted model with one supported Pydantic AI model. Keep credentials in environment
variables and never place them in a notebook or datastore:

```python
import os

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIChatModel(
    os.environ["SCARF_AGENT_MODEL"],
    provider=OpenAIProvider(
        base_url=os.environ["SCARF_AGENT_BASE_URL"],
        api_key=os.environ["SCARF_AGENT_API_KEY"],
    ),
)

orchestrator = AgentOrchestrator(model)
result = orchestrator.run(
    AutomatedWorkflowRequest(
        sourcePath="study.h5ad",
        zarrPath="study.zarr",
        studyContext=(
            "Human single-cell study with three biological replicates per "
            "condition; donor is the unit of inference and library is technical."
        ),
        allowAssumptions=False,
    )
)
```

Provider output remains provisional. Scarf validates evidence identifiers, operations, artifact
lineage, and resume state, but it cannot establish that a biologically plausible interpretation is
true.

---
description: Run Scarf's four grounded analysis agents from a rebuilt 5K PBMC baseline.
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

# Run a grounded agent workflow

This tutorial runs Data Enrichment, Experimental Context, Parameter Tuning, and Biological
Interpretation against a current 5K PBMC store.
Each stage can inspect only its bounded tools, and each recommendation must cite evidence returned by those tools.

The committed build uses small scripted `FunctionModel` callbacks.
They exercise the real tool calls, validators, artifact creation, and handoffs without sending data to an external model or requiring an API key.
The scripted Biological Interpretation callback deliberately leaves the selected cluster unresolved: it demonstrates evidence grounding, not biological expertise.
A live-provider configuration is shown at the end.

## 1. Open the rebuilt teaching store

Install the optional agent dependencies before running this workflow outside the documentation environment:

```console
pip install "scarf[agent]"
```

```{code-cell} ipython3
import scarf
from scarf.agent import (
    BiologicalContext,
    BiologicalInterpretationAgent,
    DataEnrichmentAgent,
    DataEnrichmentContext,
    ExperimentalContextAgent,
    ParameterCandidate,
    ParameterTuningAgent,
)

scarf.configure_output(level="WARNING", progress=False)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "tenx_5K_pbmc_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
ds = scarf.DataStore(
    f"{dataset}/data.zarr",
    default_assay="RNA",
    nthreads=2,
)

{
    "active_cells": int(ds.cells.fetch_all("I").sum()),
    "total_cells": ds.cells.N,
    "assays": ds.assay_names,
}
```

The downloaded store was rebuilt with this version of Scarf. Its labelled pipeline run supplies
the frozen baseline below, while agent-created candidates remain separate immutable artifacts.

The hidden setup below defines deterministic model responses for this documentation build.
Every response is assembled from the actual tool return, so fabricated cluster IDs, feature families, or evidence IDs still fail the production validators.

```{code-cell} ipython3
:tags: [remove-cell]

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from scarf.agent.biological_interpretation import (
    ClusterCompositionEvidence,
    ClusterMarkerBatchEvidence,
)
from scarf.agent.data_enrichment import AssayFeatureInspectionBatch
from scarf.agent.experimental_context import CovariateEvidence


def _tool_returns(messages: list[ModelMessage]) -> list[ToolReturnPart]:
    return [
        part
        for message in messages
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]


def _tool_call(name: str, args: dict | None = None) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(tool_name=name, args=args or {})])


def _structured_output(info: AgentInfo, payload: dict) -> ModelResponse:
    return _tool_call(info.output_tools[0].name, payload)


async def _enrichment_reply(
    messages: list[ModelMessage],
    info: AgentInfo,
) -> ModelResponse:
    returns = _tool_returns(messages)
    if not returns:
        return _tool_call("inspect_assay_features_batch")

    batch = AssayFeatureInspectionBatch.model_validate(returns[-1].content)
    inspection = batch.inspections[0]
    species_observed = inspection.species != "unknown"
    species = inspection.species if species_observed else "homo_sapiens"
    evidence_ids = list(inspection.evidenceIds)
    if not species_observed:
        evidence_ids.append("context:organism")
    policy = {
        "assay": inspection.assay,
        "species": species,
        "speciesConfidence": "high" if species_observed else "medium",
        "speciesRationale": (
            inspection.speciesReason
            or "The inspected features and caller context support this species."
        ),
        "excludeFamilies": [
            family.family
            for family in inspection.families
            if family.count > 0 and family.defaultExclude is True
        ],
        "protectFamilies": [
            family.family
            for family in inspection.families
            if family.count > 0 and family.defaultExclude is False
        ],
        "rationale": "Exclude observed technical families and preserve protected ones.",
        "evidenceIds": evidence_ids,
    }
    return _structured_output(info, {"status": "done", "policies": [policy]})


async def _experimental_reply(
    messages: list[ModelMessage],
    info: AgentInfo,
) -> ModelResponse:
    returns = _tool_returns(messages)
    if not returns:
        return _tool_call("inspect_cell_covariates")
    if len(returns) == 1:
        return _tool_call(
            "analyze_experimental_design",
            {
                "column_domains": {},
                "coefficients_of_interest": [],
                "units_of_inference": {},
                "batch_columns": [],
            },
        )

    design = CovariateEvidence.model_validate(returns[-1].content)
    evidence_id = design.evidenceIds[0]
    return _structured_output(
        info,
        {
            "batchCorrection": {
                "action": "skip",
                "rationale": (
                    "No explicit technical batch or biological contrast was supplied."
                ),
                "evidenceIds": [evidence_id],
            },
            "rationale": "Continue with an uncorrected baseline.",
            "evidenceIds": [evidence_id],
        },
    )


async def _tuning_reply(
    _messages: list[ModelMessage],
    info: AgentInfo,
) -> ModelResponse:
    return _structured_output(
        info,
        {
            "status": "done",
            "recommendedCandidateId": "baseline",
            "confidence": "medium",
            "rationale": "The single authorized baseline completed successfully.",
            "evidenceIds": ["candidate:baseline:clusters"],
            "stopReason": "The authorized candidate was evaluated.",
        },
    )


async def _biology_reply(
    messages: list[ModelMessage],
    info: AgentInfo,
) -> ModelResponse:
    returns = _tool_returns(messages)
    if not returns:
        return _tool_call("inspect_cluster_composition")
    if len(returns) == 1:
        composition = ClusterCompositionEvidence.model_validate(returns[-1].content)
        cluster_id = sorted(
            composition.clusterCounts,
            key=lambda value: (-composition.clusterCounts[value], value),
        )[0]
        return _tool_call(
            "inspect_cluster_markers_batch",
            {"cluster_ids": [cluster_id]},
        )

    marker_batch = ClusterMarkerBatchEvidence.model_validate(returns[-1].content)
    marker = marker_batch.clusters[0]
    if not marker.evidenceId:
        return _structured_output(
            info,
            {
                "status": "needsInput",
                "needsInput": {
                    "question": "No markers passed the bounded search thresholds.",
                    "requiredInputs": ["markerArtifact"],
                },
                "limitations": marker.warnings,
                "stopReason": "Marker evidence was unavailable.",
            },
        )
    names = [item.featureName or item.featureId for item in marker.markers[:3]]
    return _structured_output(
        info,
        {
            "status": "done",
            "clusterInterpretations": [
                {
                    "clusterId": marker.clusterId,
                    "proposedIdentity": "unresolved marker-defined cluster",
                    "identityIsHypothesis": True,
                    "confidence": "low",
                    "rationale": f"Top returned marker features: {', '.join(names)}.",
                    "evidenceIds": [marker.evidenceId],
                }
            ],
            "evidenceIds": [marker.evidenceId],
            "limitations": [
                "The scripted documentation model does not assign cell identities."
            ],
            "stopReason": "One bounded cluster was reviewed.",
        },
    )

```

## 2. Inspect species and feature families

Data Enrichment is read-only.
It inspects the requested assays and returns a policy rather than changing the datastore.
The context below is caller-supplied study evidence, not a label inferred from expression.

```{code-cell} ipython3
enrichment = DataEnrichmentAgent(FunctionModel(_enrichment_reply)).run(
    ds,
    context=DataEnrichmentContext(
        studyContext=(
            "10x 5K PBMC RNA-seq from peripheral blood of a healthy human donor."
        ),
        organismHint="human",
        tissueReferences=["peripheral blood"],
        cellTypeReferences=["T cell", "B cell", "NK cell", "monocyte"],
        experimentalDetails=["10x 3 prime RNA-seq", "single donor"],
    ),
    assays=["RNA"],
)

policy = enrichment.policies[0]
{
    "status": enrichment.status,
    "species": policy.species,
    "exclude_families": policy.excludeFamilies,
    "protect_families": policy.protectFamilies,
    "tool_calls": [call.name for call in enrichment.toolCalls],
}
```

The family policy is advisory.
The feature-selection API accepts an exact selection or regular-expression blacklist, so do not silently translate family names into guessed feature names.

## 3. Prepare a frozen baseline

Open the durable pipeline run built with this dataset. It supplies frozen metadata,
normalization, and a current graph to Experimental Context. Parameter Tuning consumes the exact
normalized artifact and creates its own PCA, neighbour, graph, and clustering candidate without
changing the run.

```{code-cell} ipython3
run = ds.pipeline.open(label="docs_default")
normalized = run["normalized"]
hvg_ref = run["highly_variable_features"]

{
    "run_id": run.run_id,
    "active_cells": int(run.cells.fetch_all("I").sum()),
    "feature_selection": hvg_ref.artifact_id,
    "normalized": normalized.artifact_id,
}
```

## 4. Check the experimental context

Experimental Context classifies metadata and authorizes an exact Harmony batch-column set only when the design supports it.
Passing the completed run binds metadata and integration metrics to its frozen cell selection and
graph artifact.
This single-donor store has no treatment or batch labels, so the grounded action is to keep an uncorrected baseline.

```{code-cell} ipython3
experimental = ExperimentalContextAgent(FunctionModel(_experimental_reply)).run(
    ds,
    study_context=(
        "Healthy-donor 5K PBMC. No treatment or batch labels are available. "
        "Do not invent a technical batch or biological contrast."
    ),
    run=run,
)

{
    "status": experimental.status,
    "batch_action": experimental.decision.batchCorrection.action,
    "batch_columns": experimental.decision.batchCorrection.batchColumns,
    "coefficients": experimental.decision.coefficientsOfInterest,
}
```

The validated result becomes a narrow handoff.
Downstream tuning receives the exact batch action and columns, rather than reparsing prose.

## 5. Evaluate one authorized parameter branch

The original notebook screens five defaults.
For a bounded documentation run, authorize one explicit candidate and disable refinement.
The candidate executes PCA, neighbours, graph construction, Leiden clustering, and its available
diagnostics against the baseline run's exact normalized artifact.

```{code-cell} ipython3
if experimental.status != "done":
    raise RuntimeError(f"Experimental Context stopped with {experimental.status!r}")

tuning_handoff = experimental.to_parameter_tuning_handoff()
candidate = ParameterCandidate(
    candidateId="baseline",
    dimensions=15,
    leidenResolution=0.5,
    neighborsK=11,
    useHarmony=False,
)
tuning = ParameterTuningAgent(FunctionModel(_tuning_reply)).run(
    ds,
    normalized=normalized,
    candidates=[candidate],
    experimental_handoff=tuning_handoff,
    max_candidates=1,
    max_refined_candidates=0,
    min_cluster_cells=10,
)

evaluation = tuning.evaluations[0]
{
    "status": tuning.status,
    "recommended_candidate": tuning.recommendedCandidateId,
    "eligible": evaluation.eligible,
    "clusters": evaluation.metrics.nClusters,
    "smallest_cluster": evaluation.metrics.minClusterCells,
    "cluster_artifact": evaluation.artifacts["clusters"].artifactId,
    "cell_selection": evaluation.cellSelection.artifactId,
}
```

## 6. Inspect one cluster with marker evidence

Biological Interpretation consumes the exact selected cluster artifact.
Marker search is explicitly authorized because Parameter Tuning does not create a marker table.
The build reviews only the largest cluster and uses relaxed retrieval thresholds so the example remains bounded; the returned identity remains an unresolved hypothesis.

```{code-cell} ipython3
if tuning.status != "done":
    raise RuntimeError(f"Parameter Tuning stopped with {tuning.status!r}")

biology_handoff = tuning.to_biological_handoff()
biology = BiologicalInterpretationAgent(FunctionModel(_biology_reply)).run(
    ds,
    tuning_handoff=biology_handoff,
    biological_context=BiologicalContext(
        organism="Homo sapiens",
        tissue="peripheral blood",
        cellTypeReferences=["T cell", "B cell", "NK cell", "monocyte"],
        experimentalDetails=["healthy donor PBMC", "no treatment contrast"],
    ),
    allow_marker_search=True,
    marker_features=hvg_ref,
    max_clusters=1,
    max_markers=5,
    marker_min_score=0.01,
    marker_min_fraction=0.0,
)

{
    "status": biology.status,
    "interpretations": [
        {
            "cluster": item.clusterId,
            "identity": item.proposedIdentity,
            "rationale": item.rationale,
            "evidence": item.evidenceIds,
        }
        for item in biology.clusterInterpretations
    ],
    "tool_calls": [call.toolName for call in biology.runInfo.toolCalls],
    "treatment_observations": len(biology.treatmentObservations),
    "limitations": biology.limitations,
}
```

There are no replicated condition labels, so treatment observations remain empty.
Artifact provenance supports the computational chain, but a real biological identity still requires an appropriate model, study context, and independent validation.

## Use a live model

Replace the four scripted models with one supported Pydantic AI model in an interactive analysis.
For an OpenAI-compatible endpoint, keep credentials in environment variables and never place them in a notebook or datastore:

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

enrichment = DataEnrichmentAgent(model).run(
    ds,
    context=DataEnrichmentContext(organismHint="human"),
)
```

Use the same `model` for the other stages, retain the explicit handoffs, and set execution limits appropriate to the provider.
Model output remains provisional: Scarf's validators reject unknown evidence, but they cannot establish that a biologically plausible interpretation is true.

---
description: Run Scarf's four grounded analysis agents on a fresh 5K PBMC workspace.
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

This tutorial adapts the four-agent 5K PBMC notebook into a reproducible documentation build.
It runs Data Enrichment, Experimental Context, Parameter Tuning, and Biological Interpretation against a fresh Scarf workspace.
Each stage can inspect only its bounded tools, and each recommendation must cite evidence returned by those tools.

The committed build uses small scripted `FunctionModel` callbacks.
They exercise the real tool calls, validators, artifact creation, and handoffs without sending data to an external model or requiring an API key.
The scripted Biological Interpretation callback deliberately leaves the selected cluster unresolved: it demonstrates evidence grounding, not biological expertise.
A live-provider configuration is shown at the end.

## 1. Create a fresh workspace

Install the optional agent dependencies before running this workflow outside the documentation environment:

```console
pip install "scarf[agent]"
```

```{code-cell} ipython3
from pathlib import Path
from tempfile import TemporaryDirectory

import scarf
from scarf.agent import (
    AgentRunConfig,
    BiologicalContext,
    BiologicalInterpretationAgent,
    DataEnrichmentAgent,
    DataEnrichmentContext,
    ExperimentalContextAgent,
    ParameterCandidate,
    ParameterTuningAgent,
)

scarf.configure_output(level="WARNING", progress=True)

dataset = scarf.cytebase.connect("scarf_docs").download_dataset(
    "tenx_5K_pbmc_rnaseq",
    destination="scarf_datasets",
    zarr=True,
)
analysis_directory = TemporaryDirectory()
ds = scarf.mount_datastore(
    f"{dataset}/data.zarr",
    at=str(Path(analysis_directory.name) / "agent_analysis.zarr"),
    default_assay="RNA",
    nthreads=2,
    min_features_per_cell=10,
)

{
    "active_cells": int(ds.cells.fetch_all("I").sum()),
    "total_cells": ds.cells.N,
    "assays": ds.assay_names,
}
```

The temporary mount keeps the downloaded counts read-only and places every selection and result in a new analysis layer.
That prevents an earlier run's filtering, clustering, or marker table from changing the result.

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


def _structured_output(info: AgentInfo, payload: dict) -> ModelResponse:
    return ModelResponse(
        parts=[
            ToolCallPart(
                tool_name=info.output_tools[0].name,
                args=payload,
            )
        ]
    )


async def _enrichment_reply(
    messages: list[ModelMessage],
    info: AgentInfo,
) -> ModelResponse:
    returns = _tool_returns(messages)
    if not returns:
        return ModelResponse(
            parts=[ToolCallPart(tool_name="inspect_assay_features_batch", args={})]
        )

    batch = AssayFeatureInspectionBatch.model_validate(returns[-1].content)
    policies = []
    for inspection in batch.inspections:
        excluded = [
            family
            for family in inspection.families
            if family.count > 0 and family.defaultExclude is True
        ]
        protected = [
            family
            for family in inspection.families
            if family.count > 0 and family.defaultExclude is False
        ]
        species = inspection.species
        evidence_ids = list(inspection.evidenceIds)
        if species == "unknown":
            species = "homo_sapiens"
            evidence_ids.append("context:organism")
        policies.append(
            {
                "assay": inspection.assay,
                "species": species,
                "speciesConfidence": (
                    "high" if inspection.species != "unknown" else "medium"
                ),
                "speciesRationale": (
                    inspection.speciesReason
                    or "The inspected features and caller context support this species."
                ),
                "excludeFamilies": [family.family for family in excluded],
                "protectFamilies": [family.family for family in protected],
                "rationale": (
                    "Use observed technical families as feature-selection exclusions "
                    "while preserving protected biological families."
                ),
                "evidenceIds": list(dict.fromkeys(evidence_ids)),
            }
        )
    return _structured_output(info, {"status": "done", "policies": policies})


async def _experimental_reply(
    messages: list[ModelMessage],
    info: AgentInfo,
) -> ModelResponse:
    returns = _tool_returns(messages)
    if not returns:
        return ModelResponse(
            parts=[ToolCallPart(tool_name="inspect_cell_covariates", args={})]
        )
    if len(returns) == 1:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="analyze_experimental_design",
                    args={
                        "column_domains": {},
                        "coefficients_of_interest": [],
                        "units_of_inference": {},
                        "batch_columns": [],
                    },
                )
            ]
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
        return ModelResponse(
            parts=[ToolCallPart(tool_name="inspect_cluster_composition", args={})]
        )
    if len(returns) == 1:
        composition = ClusterCompositionEvidence.model_validate(returns[-1].content)
        cluster_id = sorted(
            composition.clusterCounts,
            key=lambda value: (-composition.clusterCounts[value], value),
        )[0]
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="inspect_cluster_markers_batch",
                    args={"cluster_ids": [cluster_id]},
                )
            ]
        )

    marker_batch = ClusterMarkerBatchEvidence.model_validate(returns[-1].content)
    marker = next(
        (item for item in marker_batch.clusters if item.evidenceId),
        None,
    )
    if marker is None:
        return _structured_output(
            info,
            {
                "status": "needsInput",
                "needsInput": {
                    "question": "No markers passed the bounded search thresholds.",
                    "requiredInputs": ["markerArtifact"],
                },
                "limitations": marker_batch.warnings,
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


enrichment_model = FunctionModel(_enrichment_reply)
experimental_model = FunctionModel(_experimental_reply)
tuning_model = FunctionModel(_tuning_reply)
biology_model = FunctionModel(_biology_reply)
config = AgentRunConfig(temperature=0.0, timeoutSeconds=600.0)
```

## 2. Inspect species and feature families

Data Enrichment is read-only.
It inspects the requested assays and returns a policy rather than changing the datastore.
The context below is caller-supplied study evidence, not a label inferred from expression.

```{code-cell} ipython3
enrichment = DataEnrichmentAgent(enrichment_model, config=config).run(
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

## 3. Preprocess through normalization

Parameter Tuning owns PCA, neighbours, the graph, and clustering in this workflow.
Create only the filtered-cell selection, highly variable feature selection, and normalized artifact first.

```{code-cell} ipython3
ds.filter_cells(
    attrs=["RNA_nCounts", "RNA_nFeatures", "RNA_percentMito"],
    highs=[15000, 4000, 15],
    lows=[1000, 500, 0],
    reset_previous=True,
)
hvg_ref = ds.mark_hvgs(
    from_assay="RNA",
    top_n=500,
    min_cells=20,
    show_plot=False,
)
normalized = ds.run_normalization(
    from_assay="RNA",
    features=hvg_ref,
    log_transform=True,
    renormalize_subset=True,
    update_state=False,
)

{
    "active_cells": int(ds.cells.fetch_all("I").sum()),
    "feature_selection": hvg_ref.artifact_id,
    "normalized": normalized.artifact_id,
}
```

## 4. Check the experimental context

Experimental Context classifies metadata and authorizes an exact Harmony batch-column set only when the design supports it.
This single-donor store has no treatment or batch labels, so the grounded action is to keep an uncorrected baseline.

```{code-cell} ipython3
experimental = ExperimentalContextAgent(
    experimental_model,
    config=config,
).run(
    ds,
    study_context=(
        "Healthy-donor 5K PBMC. No treatment or batch labels are available. "
        "Do not invent a technical batch or biological contrast."
    ),
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
For a bounded documentation run, authorize one explicit baseline and disable refinement.
The candidate still executes normalization consumers, PCA, neighbours, graph construction, Leiden clustering, and its available diagnostics against the fresh store.

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
tuning = ParameterTuningAgent(tuning_model, config=config).run(
    ds,
    normalized=normalized,
    from_assay="RNA",
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
    "cluster_column": evaluation.clusterColumn,
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
biology = BiologicalInterpretationAgent(
    biology_model,
    config=config,
).run(
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

```{code-cell} ipython3
cluster_column = biology_handoff.clusterColumn
ds.cells.to_pandas_dataframe(
    columns=[cluster_column],
    key="I",
)[cluster_column].value_counts().sort_index()
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

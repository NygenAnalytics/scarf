---
description: Run a bounded RNA analysis with structured agent decisions and inspect its evidence.
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

# Automate an RNA analysis

Scarf agents analyze one RNA assay in a prepared local Scarf store or Cytebase mount.
Scarf measures the data and executes a fixed pipeline. A configured language model interprets
study metadata, chooses among registered alternatives, selects measured finalists, and proposes
provisional cluster identities. Results include clustering, descriptive markers, UMAP,
annotations, and a report explaining the decisions and limitations.

This tutorial runs a small synthetic example without downloading data or contacting a model
provider. For a real Cytebase dataset with a configured provider, see
{doc}`garrido_trigo_agents`. Annotation quality still requires biological review.

## Prepare your input and model

Install the optional agent dependencies:

```bash
uv pip install "scarf[agent]"
```

Import or mount the data before calling the agent. Follow {doc}`import_and_export` for count
files and {doc}`remote_stores` for Cytebase mounts. The agent does not download, convert, mount,
or publish a dataset. A local mount may still read remote count bytes during numerical work.

During development, a new analysis rejects complete numerical artifacts from earlier analyses,
including artifacts inherited from a mount's source. Imported labels and embeddings are allowed.
Prepare a clean input separately if needed; `repack_store(..., data_only=True)` copies count data
and rebuilds preparation metadata without the old analysis artifacts. The agent does not delete
old results. Its own completed artifacts can be reused during the same run and explicit resume.

Pass a Pydantic AI model object or a supported `provider:model-name` identifier. Configure
credentials on the provider or in the environment, not in saved study text or runtime settings.
Provider calls may incur charges. The model receives no shell, web retrieval, scientific tools,
or permission to execute generated code.

## Start a real analysis

In a notebook, await the asynchronous entry point. Here `model` is your configured Pydantic AI
model, and `study.zarr` must already be initialized.

```python
from scarf.agent import AnalysisConfig, Study, analyze_rna_async

run = await analyze_rna_async(
    "study.zarr",
    model=model,
    study=Study(
        context="Human blood from one healthy donor; no treatment comparison.",
        objective="Describe the major immune populations and uncertain identities.",
        organism="Homo sapiens",
        tissue="blood",
        excludedColumns=["author_annotation"],
    ),
    config=AnalysisConfig(assay="RNA", scoreDoublets=False),
)
print(run.status)
print(run.run_dir)
print(run.report())
```

Use `analyze_rna` in a normal Python script. Both interfaces execute numerical stages sequentially.
By default, the complete audit and report live in `agent_runs/<runId>` under the current working
directory. Set `run_dir="analyses/my-study"` to choose another new directory outside the numerical
store. An existing directory is rejected; resume is a separate operation.

`Study` separates supplied facts from scientific configuration. Declare `sampleColumn`,
`captureColumn`, `technicalBatchColumns`, and `protectedColumns` when supported by the experimental
design. Repeated samples are not independent biological replicates. Hold evaluation labels out
with `excludedColumns`. Local UTF-8 excerpts in `referenceFiles` have a combined 16 KiB limit.
Missing replication does not block descriptive population discovery.

## What runs, and who decides?

```{mermaid}
flowchart TD
    A[Inspect prepared store and freeze input identity] --> B[Model interprets metadata roles]
    B --> C[Freeze cohort, exclusions, and correction eligibility]
    C --> D[Execute baseline on every retained cell]
    D --> E[Model chooses one feasible PC probe]
    E --> F[Execute independent HVG, PC, and neighbor probes]
    F --> G[Model shortlists measured partitions]
    G --> H{Eligible optional Harmony comparison?}
    H -->|Yes| I[Measure corrected partition and matched native control]
    H -->|No| J[Compute markers for at most two finalists]
    I --> J
    J --> K[Validate correction gates and model selects finalist]
    K --> L[Finalize pinned recipe with UMAP and reused markers]
    L --> M[Model annotates every cluster in batches of at most eight]
    M --> N[Save compact store result and external report]
```

The default baseline has 1,000 HVGs, 21 PCs, 11 neighbors, and Leiden resolutions
`0.5`, `0.75`, `1.0`, and `1.25`. Four native representations are planned: the baseline,
one HVG alternative, one PC alternative, and one neighbor alternative. Each probe changes one
parameter from the same baseline. Alternatives come from HVGs `2,000/4,000`, PCs `10/30`, and
neighbors `21/41`; registered fallback values handle explicitly changed baselines. The model
cannot skip feasible native probes by accepting the baseline early.

Scarf checks dimensional rank, neighbor feasibility, actual selected genes, shared cohort,
same-resolution comparisons, and artifact lineage. Identical HVG selections, infeasible probes,
and failed trials remain visible in exploration coverage. An admitted failed trial consumes its
slot. There is no grid search, combined-parameter search, or automatic scientific repair.

Four native screens, two marker assessments, and finalization permit at most seven normal
pipeline invocations. One eligible Harmony trial raises that limit to eight. Explicit recovery
can add invocations; these counts do not bound elapsed time or provider spending.

| Evidence or result | Cells used |
| --- | --- |
| PCA, graph, clustering, markers, final UMAP | The entire retained cohort |
| PC/covariate and correction diagnostics | At most 10,000 cells |
| Silhouette assessment | At most 2,000 cells |
| Finalist cluster sizes and group/QC summaries | The entire retained cohort |

Diagnostic sampling never substitutes a smaller discovery cohort. Parameter comparisons include
adjusted Rand index and directional overlap on aligned cells at the same resolution. Marker
support is the fraction of clusters with at least one qualifying measured marker, not a measure
of correct cell identity.

## QC, correction, and uncertainty

The default `qcPolicy="retain"` keeps the supplied cohort and records outlier flags. Projected
retention under other supported policies is evidence, not additional filtering. Global manual
thresholds or the gentle five-MAD profile require explicit configuration. High counts/features
remain flags under the gentle profile. Optional missing metrics stay unknown.

The workflow explicitly supplies its HVG blacklist, normally excluding mitochondrial names
matching `^mt-` case-insensitively. HLA/H2, sex-linked, cell-cycle, and reporter features are
preserved unless explicitly excluded. An organism name alone does not resolve gene identifiers.

Harmony requires declared technical batches, complete labels, protected biological variables,
and supplied evidence separating technical variation from biology. The current design check
requires protected groups across technical batches. Unknown or confounded roles retain native
analysis. A corrected finalist needs its exact native counterpart at the same resolution, plus
measured mixing, preservation, marker, and doublet evidence. Checks use a `0.05` tolerance and
require improvement in at least one mixing measure.

**Doublet scoring is always opt-in.** With `scoreDoublets=False`, correction requiring doublet
evidence is unavailable. The agent never enables scoring implicitly, and scoring never removes
cells automatically.

The default `interactionMode="lenient"` applies recorded conservative policies to supported
ambiguities. Optional unknown metadata stays unknown. A tie among model-declared acceptable
partitions prefers native analysis, then frozen trial and resolution order. This is a disclosed
tie rule, not evidence of biological superiority. Strict mode keeps such questions pending.
Essential missing facts produce `needsInput` in either mode. Provider failures, invalid output
after bounded repair, and unknown numerical failures still stop work.

## Read and continue a run

The returned status is `running`, `needsInput`, `completed`, `failed`, or `interrupted`.
Inspect it before using finalized numerical results. Every persisted outcome supports a report,
including questions and failures. A report error does not downgrade scientific status.

```python
from scarf.agent import open_analysis, resume_rna_async

run = open_analysis("analyses/my-study")
print(run.status, run.pending_questions)
print(run.exploration_coverage)
print(run.decision_resolutions)
run.report()  # Saved evidence only; no provider or numerical computation.

# Explicitly resume unfinished work after reviewing its recorded outcome.
run = await resume_rna_async(run.run_dir, model=model)
```

For a pending question, pass `answers={questionId: answer}` for every exact pending ID. Answers
cannot replace fixed scientific settings. Resume requires matching scientific inputs and Scarf
procedure/prompt identity. Operational settings or the provider may change for unfinished
choices. A relocated source must match the fingerprint and saved pipeline/artifact history.
Prototype histories cannot be resumed.

Completed results expose `run.pipeline`, `run.artifacts`, `run.get_markers()`,
`run.plot_embedding()`, `run.plot_markers()`, and `run.annotations`. Numerical access verifies
the source and exact final artifacts. Annotations remain provisional and do not overwrite cell
metadata. A named identity requires observed supporting markers, but this validation cannot
establish that the biological identity is correct. Explicit `unassigned` clusters are permitted.

The external directory retains `run.json`, immutable events, evidence, visible model exchanges,
annotations, reports, and previews. A compact summary in `agent_results/<runId>` inside the
local Zarr store links to the exact final core pipeline and its workspace, selected configuration,
rationale, and external audit location. Read it through `run.compact_result`. It does not duplicate the
full history or make the external audit disposable. See {doc}`../reference/api/agent` for details.

## Worked example without network access

This teaching fixture contains 120 synthetic cells and 2,102 features. Three planted expression
patterns provide numerical structure without pretending they are real cell types. It is small
enough to construct in memory; real analysis uses Scarf's bounded count access. No downloaded
dataset or live provider is used.

```{code-cell} ipython3
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import scarf
from scarf.agent import AnalysisConfig, RuntimeConfig, Study, analyze_rna_async
from scarf.writers import SparseToZarr

scarf.configure_output(level="WARNING", progress=False)
teaching_directory = TemporaryDirectory(prefix="scarf-agent-teaching-")
source = Path(teaching_directory.name) / "counts.zarr"
rng = np.random.default_rng(39)
counts = rng.poisson(1, size=(120, 2102)).astype(np.uint32)
for group in range(3):
    counts[group * 40:(group + 1) * 40, 2 + group * 40:42 + group * 40] += (
        rng.poisson(4, size=(40, 40)).astype(np.uint32)
    )
names = ["MT-CO1", "RPL3", *[f"GENE{i}" for i in range(2100)]]
writer = SparseToZarr(
    csr_matrix(counts), str(source), [f"cell{i}" for i in range(120)], names,
    mem_budget="512M", nthreads=1,
)
writer.dump()
prepared = scarf.DataStore(
    str(source), default_assay="RNA", min_features_per_cell=-1,
    nthreads=1, mem_budget="512M",
)
prepared.cells.insert("sample", np.tile(["sample_A", "sample_B"], 60))
prepared.cells.insert("author_annotation", np.repeat(["planted_A", "planted_B", "planted_C"], 40))
initial_selection = prepared.cells.fetch_all("I").copy()
initial_columns = list(prepared.cells.columns)
del prepared, counts
```

The local `FunctionModel` receives the same schemas and measured evidence as a provider. It
chooses the registered 30-PC probe, compares two measured baseline resolutions, selects the first
eligible finalist, and leaves every synthetic cluster unassigned. These scripted preferences
demonstrate the interface; they are not a scientific selection algorithm.

```{code-cell} ipython3
import json

from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

observed_decisions = []

async def teaching_decision(messages, info):
    payload = json.loads(messages[-1].parts[-1].content)
    evidence = payload["evidence"]
    schema = info.output_tools[0].parameters_json_schema["title"]
    observed_decisions.append({"stage": payload["stage"], "schema": schema})
    if schema == "ContextDecision":
        answer = {"rationale": "Retain the supplied cohort and declared sample role."}
    elif schema == "AnnotationDecision":
        answer = {
            "annotations": [
                {
                    "clusterId": row["clusterId"],
                    "identity": "unassigned",
                    "rationale": "Synthetic markers do not establish a biological cell identity.",
                }
                for row in evidence["clusters"]
            ]
        }
    elif evidence.get("decisionKind") == "pcProbe":
        chosen = next(
            key for key, value in evidence["experiments"].items()
            if value["pcaDims"] == 30
        )
        answer = {
            "action": "experiment", "optionIds": [chosen], "evidenceIds": ["c0"],
            "rationale": "Measure the registered higher-rank probe for this demonstration.",
        }
    elif "eligibleOptions" in evidence:
        chosen = evidence["eligibleOptions"][0]
        answer = {
            "action": "choose", "optionIds": [chosen], "evidenceIds": [chosen],
            "rationale": "Use the first measured eligible finalist for this demonstration.",
        }
    else:
        assert evidence["decisionKind"] == "nativeShortlist"
        chosen = [row["optionId"] for row in evidence["candidates"][0]["partitions"][:2]]
        answer = {
            "action": "shortlist", "optionIds": chosen, "evidenceIds": chosen,
            "rationale": "Compare two measured baseline resolutions after the independent probes.",
        }
    return ModelResponse(parts=[ToolCallPart("decision", answer)])

teaching_model = FunctionModel(teaching_decision)
```

All numerical operations, evidence preparation, validation, and persistence use the production
workflow. The external run directory is temporary here so the example leaves no dataset beside
the documentation source. Choose a durable location for real work.

```{code-cell} ipython3
result = await analyze_rna_async(
    source,
    run_dir=Path(teaching_directory.name) / "analysis",
    model=teaching_model,
    study=Study(
        context="Synthetic RNA with three planted expression patterns and two interleaved sample labels.",
        objective="Demonstrate bounded population discovery without biological identity claims.",
        sampleColumn="sample",
        excludedColumns=["author_annotation"],
    ),
    config=AnalysisConfig(assay="RNA", maxCandidates=4),
    runtime=RuntimeConfig(nthreads=1, memBudget="512M"),
)
assert result.status == "completed", result.status
pd.DataFrame(result.exploration_coverage["slots"])[["candidateId", "axis", "status", "reason"]]
```

Inspect actual selected-feature counts. Requested HVG counts alone do not show that two trials
used different genes.

```{code-cell} ipython3
pd.DataFrame([
    {
        "candidate": row["candidateId"],
        "HVGs requested": row["parameters"]["hvgCount"],
        "HVGs selected": row["actualHvgCount"],
        "PCs": row["parameters"]["pcaDims"],
        "neighbors": row["parameters"]["neighborsK"],
    }
    for row in result.candidates
])
```

These figures consume the final saved UMAP and marker statistics without another parameter
search or model request.

```{code-cell} ipython3
from IPython.display import display

embedding = result.plot_embedding(show=False)
display(embedding.figure)
embedding.close()
marker_plot = result.plot_markers(show=False)
display(marker_plot.figure)
marker_plot.close()
```

Every synthetic identity remains unassigned. Completion means that every required stage and
cluster record is present, not that every population received a named biological identity.

```{code-cell} ipython3
pd.DataFrame(result.annotations)[["clusterId", "identity", "confidence", "rationale"]]
```

The compact result points to the exact final pipeline and executed configuration. The full audit
remains external; report regeneration and decision replay use those saved files.

```{code-cell} ipython3
compact = result.compact_result
assert compact["finalPipelineRunId"] == result.pipeline.run_id
compact["selectedParameters"]
```

```{code-cell} ipython3
from scarf.agent import open_analysis

reopened = open_analysis(result.run_dir)
report_path = reopened.report()
replayed = reopened.replay_decisions()
assert all(row["valid"] for row in replayed)
after = scarf.DataStore(str(source), zarr_mode="r", nthreads=1, mem_budget="512M")
assert list(after.cells.columns) == initial_columns
np.testing.assert_array_equal(after.cells.fetch_all("I"), initial_selection)
{
    "status": reopened.status,
    "pipeline invocations": len(reopened.pipeline_runs),
    "model decisions": len(observed_decisions),
    "offline replay checks": len(replayed),
    "report": report_path.name,
}
```

Open the generated `report.html` during your own run to review its six steps: study and input,
quality and preparation, exploration, final selection, numerical results, and provisional
identities. Export aligned labels, coordinates, markers, and annotations with
`result.export("a-new-export-directory")`. Archive both the numerical store and external audit.

Every live decision requests reasoning off and has structured validation, one semantic repair,
one transient retry, and recorded request budgets. Provider support varies; always-on reasoning
cannot be disabled universally. The teaching provider demonstrates workflow mechanics without
evaluating live-model reliability or annotation accuracy.

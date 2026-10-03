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

Scarf agents run an RNA analysis, compare a small set of analysis settings, and propose cell
identities from the measured markers. The result includes clusters, UMAP, marker tables,
provisional annotations, and a report explaining the choices.

Start with a prepared store and a model provider. The first sections show how to run the analysis
and review its results; {doc}`garrido_trigo_agents` is a worked example on real data. An optional
developer example at the end runs without a provider. Annotation still needs biological review.

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
from scarf.agent import Study, analyze_rna_async

# Start a new RNA analysis with the supplied study and model.
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
)
# Show whether the analysis completed or needs attention.
print(run.status)
# Show where the analysis history was saved.
print(run.run_dir)
# Write the report and show its local path.
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

## Read and continue a run

The returned status is `running`, `needsInput`, `completed`, `failed`, or `interrupted`.
Inspect it before using finalized numerical results. Every persisted outcome supports a report,
including questions and failures. A report error does not downgrade scientific status.

```python
from scarf.agent import open_analysis, resume_rna_async

# Reopen the saved agent analysis for inspection.
run = open_analysis(run.run_dir)
# Inspect the saved status and any unresolved questions.
print(run.status, run.pending_questions)
# Inspect which parameter alternatives were measured.
print(run.exploration_coverage)
# Inspect how the recorded decisions were resolved.
print(run.decision_resolutions)
# Write the report from the saved analysis evidence.
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

## What the agent compares

The baseline uses 1,000 variable genes, 21 PCs, and 11 neighbors. Scarf then measures alternatives
that change one of these settings at a time. It compares clusterings on the same retained cells,
checks markers for up to two finalists, and makes a final UMAP.

The model interprets those measurements and proposes labels. Scarf executes the numerical
operations and checks the returned decisions. A clean UMAP or many marker genes does not, by
itself, establish that the chosen identities are correct.

The main analysis uses every retained cell. Some diagnostics use bounded samples: up to 10,000
cells for covariate checks and 2,000 for silhouette assessment. See the
{doc}`../reference/api/agent` reference for the full comparison rules and execution limits.

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

## Optional: a developer example without a provider

The user workflow above is complete. This optional section is for readers who want to inspect
how structured model decisions enter the agent. It uses a scripted provider and synthetic data;
it does not teach biological annotation or evaluate a live model.

The fixture has 120 cells and 2,102 features, with three planted expression patterns. It is small
enough to construct in memory and needs no dataset download or provider credentials.

```{code-cell} ipython3
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import scarf
from scarf.agent import AnalysisConfig, RuntimeConfig, Study, analyze_rna_async
from scarf.writers import SparseToZarr

# Keep routine progress messages out of the teaching output.
scarf.configure_output(level="WARNING", progress=False)
```

Create a small synthetic count matrix for the provider-free example.

```{code-cell} ipython3
# Keep the synthetic dataset and report in a temporary folder.
teaching_directory = TemporaryDirectory(prefix="scarf-agent-teaching-")
# Choose the local path for the synthetic count store.
source = Path(teaching_directory.name) / "counts.zarr"
# Seed the generator so this example is reproducible.
rng = np.random.default_rng(39)
# Generate low background counts for the synthetic cells and genes.
counts = rng.poisson(1, size=(120, 2102)).astype(np.uint32)
# Plant a different expression pattern in each synthetic group.
for group in range(3):
    # Generate counts for this group's planted expression pattern.
    planted_counts = rng.poisson(4, size=(40, 40)).astype(np.uint32)
    # Add that pattern to the group's cells and marker genes.
    counts[group * 40 : (group + 1) * 40, 2 + group * 40 : 42 + group * 40] += planted_counts
# Check the synthetic matrix dimensions before writing it.
{"cells": counts.shape[0], "genes": counts.shape[1]}
```

Write the counts and prepare the study metadata.

```{code-cell} ipython3
# Add recognizable QC genes and names for the synthetic features.
names = ["MT-CO1", "RPL3", *[f"GENE{i}" for i in range(2100)]]
# Prepare a writer for the selected counts and metadata.
writer = SparseToZarr(
    csr_matrix(counts),
    str(source),
    [f"cell{i}" for i in range(120)],
    names,
    mem_budget="512M",
    nthreads=1,
)
# Write the prepared counts and metadata to the new store.
writer.dump()
```

Open the synthetic store and record its initial selection.

```{code-cell} ipython3
# Open the synthetic count store without additional filtering.
prepared = scarf.DataStore(
    str(source),
    default_assay="RNA",
    min_features_per_cell=-1,
    nthreads=1,
    mem_budget="512M",
)
# Save the values in cell metadata using the stated selection.
prepared.cells.insert("sample", np.tile(["sample_A", "sample_B"], 60))
# Save the values in cell metadata using the stated selection.
prepared.cells.insert(
    "author_annotation", np.repeat(["planted_A", "planted_B", "planted_C"], 40)
)
# Record the input selection for the later preservation check.
initial_selection = prepared.cells.fetch_all("I").copy()
# Record the original metadata columns.
initial_columns = list(prepared.cells.columns)
# Release objects that are no longer needed.
del prepared, counts
# Check the original selection and metadata size.
{
    "selected cells": int(initial_selection.sum()),
    "metadata columns": len(initial_columns),
}
```

The local `FunctionModel` receives the same schemas and measured evidence as a provider. It
chooses the registered 30-PC probe, compares two measured baseline resolutions, selects the first
eligible finalist, and leaves every synthetic cluster unassigned. These scripted preferences
demonstrate the interface; they are not a scientific selection algorithm.

```{code-cell} ipython3
import json

from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

# Collect the decision stages seen by the scripted provider.
observed_decisions = []


# Return scripted decisions for the measured teaching example.
async def teaching_decision(messages, info):
    # Read the evidence payload supplied to the model.
    payload = json.loads(messages[-1].parts[-1].content)
    # Keep the measured evidence for the current decision.
    evidence = payload["evidence"]
    # Identify which structured decision the workflow requests.
    schema = info.output_tools[0].parameters_json_schema["title"]
    # Record the stage and schema requested by the workflow.
    observed_decisions.append({"stage": payload["stage"], "schema": schema})
    # Confirm the supplied study context without changing the cohort.
    if schema == "ContextDecision":
        # Explain why the supplied cohort and sample role are retained.
        answer = {"rationale": "Retain the supplied cohort and declared sample role."}
    # Leave synthetic clusters without biological identity claims.
    elif schema == "AnnotationDecision":
        # Return one provisional annotation for every measured cluster.
        answer = {
            "annotations": [
                {
                    "clusterId": row["clusterId"],
                    "identity": "unassigned",
                    "rationale": (
                        "Synthetic markers do not establish a biological cell identity."
                    ),
                }
                for row in evidence["clusters"]
            ]
        }
    # Request the registered higher-rank PCA experiment.
    elif evidence.get("decisionKind") == "pcProbe":
        # Find the option that measures 30 principal components.
        chosen = next(
            key
            for key, value in evidence["experiments"].items()
            if value["pcaDims"] == 30
        )
        # Link this experiment request to its baseline evidence.
        answer = {
            "action": "experiment",
            "optionIds": [chosen],
            "evidenceIds": ["c0"],
            "rationale": "Measure the registered higher-rank probe for this demonstration.",
        }
    # Choose a finalist only after the workflow has measured eligible options.
    elif "eligibleOptions" in evidence:
        # Use the first eligible finalist for this interface demonstration.
        chosen = evidence["eligibleOptions"][0]
        # Record the chosen option and the evidence supporting its eligibility.
        answer = {
            "action": "choose",
            "optionIds": [chosen],
            "evidenceIds": [chosen],
            "rationale": "Use the first measured eligible finalist for this demonstration.",
        }
    else:
        # Require the expected shortlist decision before selecting candidates.
        assert evidence["decisionKind"] == "nativeShortlist"
        # Shortlist the first two measured baseline partitions.
        chosen = [
            row["optionId"] for row in evidence["candidates"][0]["partitions"][:2]
        ]
        # Request marker review for both shortlisted partitions.
        answer = {
            "action": "shortlist",
            "optionIds": chosen,
            "evidenceIds": chosen,
            "rationale": (
                "Compare two measured baseline resolutions after the independent probes."
            ),
        }
    return ModelResponse(parts=[ToolCallPart("decision", answer)])
```

Wrap the scripted decisions as a model, then run the analysis.

```{code-cell} ipython3
# Expose the scripted decision function through the model interface.
teaching_model = FunctionModel(teaching_decision)
```

All numerical operations, evidence preparation, validation, and persistence use the production
workflow. The external run directory is temporary here so the example leaves no dataset beside
the documentation source. Choose a durable location for real work.

```{code-cell} ipython3
# Run the bounded analysis with the synthetic study and scripted model.
result = await analyze_rna_async(
    source,
    run_dir=Path(teaching_directory.name) / "analysis",
    model=teaching_model,
    study=Study(
        context=(
            "Synthetic RNA with three planted expression patterns "
            "and two interleaved sample labels."
        ),
        objective=(
            "Demonstrate bounded population discovery without biological identity claims."
        ),
        sampleColumn="sample",
        excludedColumns=["author_annotation"],
    ),
    config=AnalysisConfig(assay="RNA", maxCandidates=4),
    runtime=RuntimeConfig(nthreads=1, memBudget="512M"),
)
# Require a completed analysis before inspecting final results.
assert result.status == "completed", result.status
# Inspect which parameter alternatives were measured.
pd.DataFrame(result.exploration_coverage["slots"])[
    ["candidateId", "axis", "status", "reason"]
]
```

Inspect actual selected-feature counts. Requested HVG counts alone do not show that two trials
used different genes.

```{code-cell} ipython3
# Compare requested and actual feature counts for every candidate.
pd.DataFrame(
    [
        {
            "candidate": row["candidateId"],
            "HVGs requested": row["parameters"]["hvgCount"],
            "HVGs selected": row["actualHvgCount"],
            "PCs": row["parameters"]["pcaDims"],
            "neighbors": row["parameters"]["neighborsK"],
        }
        for row in result.candidates
    ]
)
```

These figures consume the final saved UMAP and marker statistics without another parameter
search or model request.

```{code-cell} ipython3
from IPython.display import display

# Create the final cluster figure from saved results.
embedding = result.plot_embedding(show=False)
# Display the final clustering on its saved UMAP.
display(embedding.figure)
# Close the displayed figure to release its resources.
embedding.close()
```

Inspect the markers supporting the final clustering.

```{code-cell} ipython3
# Create the marker figure from the same final clustering.
marker_plot = result.plot_markers(show=False)
# Display the marker evidence for the final clusters.
display(marker_plot.figure)
# Close the displayed figure to release its resources.
marker_plot.close()
```

Every synthetic identity remains unassigned. Completion means that every required stage and
cluster record is present, not that every population received a named biological identity.

```{code-cell} ipython3
# Inspect the provisional identities and their supporting rationale.
annotation_table = pd.DataFrame(result.annotations)[
    ["clusterId", "identity", "confidence", "rationale"]
]
# Show each explanation in full rather than truncating it with an ellipsis.
with pd.option_context("display.max_colwidth", None):
    display(annotation_table)
```

The compact result points to the exact final pipeline and executed configuration. The full audit
remains external; report regeneration and decision replay use those saved files.

```{code-cell} ipython3
# Read the compact summary linked to the final pipeline.
compact = result.compact_result
# Check that the compact summary points to the final pipeline.
assert compact["finalPipelineRunId"] == result.pipeline.run_id
# Read the settings selected by the scripted workflow.
selected = compact["selectedParameters"]
# Summarize the choices that determine the final analysis.
pd.Series(
    {
        "candidate": selected["candidateId"],
        "HVGs": selected["actualHvgCount"],
        "PCs": selected["pcaDims"],
        "neighbors": selected["neighborsK"],
        "resolution": selected["resolution"],
        "Harmony": selected["useHarmony"],
    },
    name="selected settings",
)
```

```{code-cell} ipython3
from scarf.agent import open_analysis

# Reopen the saved analysis without requesting a new model decision.
reopened = open_analysis(result.run_dir)
# Regenerate the report from saved evidence.
report_path = reopened.report()
# Recheck recorded decisions against their saved evidence.
replayed = reopened.replay_decisions()
# Check that every recorded decision passes offline replay.
assert all(row["valid"] for row in replayed)
# Reopen the original input to verify its selection and metadata.
after = scarf.DataStore(str(source), zarr_mode="r", nthreads=1, mem_budget="512M")
# Check that the agent did not add or remove input metadata columns.
assert list(after.cells.columns) == initial_columns
# Verify that the original selected-cell mask is unchanged.
np.testing.assert_array_equal(after.cells.fetch_all("I"), initial_selection)
# Summarize the reopened analysis and its replay checks.
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

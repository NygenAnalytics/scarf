---
description: Run an unattended Scarf agent analysis on a local copy of the Garrido-Trigo Cytebase RNA dataset.
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(garrido_trigo_agents)=

# Analyze Garrido-Trigo RNA with Scarf agents

Use Scarf agents to explore RNA profiles from an intestinal IBD and healthy-control study.
We will prepare a local copy, describe the study, run the analysis with default scientific
settings, and inspect the proposed identities.

The input preparation is longer than the analysis call because this published store needs
local preparation before the current agent can use it. Do this once; later visits can resume
the saved analysis. For the shorter general API example, start with {doc}`agent_workflow`.

{nb-download}`Download the notebook <garrido_trigo_agents.ipynb>`.
The general workflow and an example without provider credentials are in
{doc}`agent_workflow`.

## Set up the notebook folder

Use an environment with Scarf's `agent`, `cytebase`, `docs`, and `extra` extras.
Configure `BASETEN_API_KEY` in your environment or a local `.env` file. This example
uses the OpenAI-compatible Baseten endpoint and `deepseek-ai/DeepSeek-V4.1-Flash`;
model calls send supplied metadata and measured marker evidence to that provider.
Scarf requests disabled reasoning on every call. Provider calls incur usage, and
the configured request limits are not a universal spending guarantee.

Run the notebook from its own folder. `SCARF_AGENT_NOTEBOOK_DIR` can choose another
working folder, and `SCARF_AGENT_ENV_FILE` can point to a local environment file.
Neither credentials nor private source locations are displayed or saved in the
notebook outputs.

```{code-cell} ipython3
import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from IPython.display import Image, display

import scarf
from scarf.agent import AnalysisConfig, RuntimeConfig, Study, analyze_rna_async, resume_rna_async
```

Load provider settings and choose the notebook working folder.

```{code-cell} ipython3
# Read the optional local environment-file setting.
env_file = os.environ.get("SCARF_AGENT_ENV_FILE")
# Load provider credentials from the local environment file.
_ = load_dotenv(env_file) if env_file else load_dotenv()
# Choose the notebook's working folder.
work_dir = Path(os.environ.get("SCARF_AGENT_NOTEBOOK_DIR", Path.cwd())).resolve()
# Create the local folder if it does not already exist.
work_dir.mkdir(parents=True, exist_ok=True)
# Run subsequent relative paths from the chosen notebook folder.
os.chdir(work_dir)
# Keep routine progress messages out of the teaching output.
scarf.configure_output(level="WARNING", progress=False)
# Keep request URLs out of shared notebook output.
logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)

# Keep the dataset identifier returned by the catalog.
dataset_id = "garridotrigo_2023_ibd_hc_10x_single_cell_transcriptomics_data_9bfecd44"
# Keep the prepared local count store in the working folder.
data_path = work_dir / "data.zarr"
# Choose the directory for the saved agent analysis.
run_dir = work_dir / "agent_runs" / "garrido-trigo"
# Show whether an analysis already exists, without displaying private paths.
{"prepared input exists": data_path.exists(), "saved analysis exists": run_dir.exists()}
```

Check provider access before spending time on input preparation. A saved completed run can be
reopened without configuring a provider.

```{code-cell} ipython3
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# Read the provider key from the environment without displaying it.
api_key = os.environ.get("BASETEN_API_KEY")
# Configure the model used for new or resumed decisions.
model = None
# Configure a provider only when a key is available.
if api_key:
    # Configure the model used for new or resumed decisions.
    model = OpenAIChatModel(
        "deepseek-ai/DeepSeek-V4.1-Flash",
        provider=OpenAIProvider(base_url="https://inference.baseten.co/v1", api_key=api_key),
    )
# Require provider access before downloading a fresh analysis input.
if not run_dir.exists() and model is None:
    raise RuntimeError("Configure BASETEN_API_KEY before starting a fresh analysis")
# Show whether this session can start or resume an analysis.
{"saved analysis exists": run_dir.exists(), "provider configured": model is not None}
```

## Prepare the input once

`Catalog.mount_datastore()` creates a writable local mount whose counts remain
remote. It has no full-copy option. The published version used here also predates
the preparation metadata required by current Scarf mounts. To obtain a complete
local copy, resolve the source through Cytebase, download its files read-only, and
use the public `repack_store(data_only=True)` operation to prepare `data.zarr`.
This preparation leaves published data unchanged and excludes prior numerical
analysis artifacts from the working copy.

The downloaded snapshot stays in `scarf_datasets/`. On subsequent notebook runs,
the existing prepared `data.zarr` is retained and its saved analysis can be resumed.
A different source or scientific policy should use a new notebook folder.

```{code-cell} ipython3
from huggingface_hub import BucketFile, download_bucket_files, get_token, list_bucket_tree
from huggingface_hub.utils import disable_progress_bars

from scarf import cytebase
from scarf.tools.repack_zarr import repack_store

# Prepare counts only when there is no completed local input.
prepare_input = not data_path.exists()
if prepare_input:
    # Connect to the public Cytebase catalog.
    catalog = cytebase.Catalog()
    # Read the dataset's scientific metadata and citation.
    entry = catalog.dataset(dataset_id)
    # Resolve the published dataset's source location from its catalog entry.
    source_uri = entry.row["zarr_uri"]
    # Require the Hugging Face bucket format expected by this copy recipe.
    if not source_uri.startswith("hf://buckets/"):
        raise ValueError("This copy recipe expects a Cytebase HF bucket source")
    # Separate the bucket source into its path components.
    pieces = source_uri.removeprefix("hf://buckets/").split("/")
    # Separate the bucket name from the dataset's object prefix.
    bucket, prefix = "/".join(pieces[:2]), "/".join(pieces[2:]).rstrip("/") + "/"
    # Keep a local copy of the published input files.
    raw_path = work_dir / "scarf_datasets" / f"{dataset_id}.zarr"
    # Create the local folder if it does not already exist.
    raw_path.mkdir(parents=True, exist_ok=True)
    # Keep file-transfer progress out of the saved notebook.
    disable_progress_bars()
    # Use existing credentials when available, otherwise read anonymously.
    token = get_token() or False
    # List the downloadable files belonging to this dataset.
    objects = [
        item
        for item in list_bucket_tree(bucket, prefix=prefix, recursive=True, token=token)
        if isinstance(item, BucketFile)
    ]

# Show whether the input needs to be copied and prepared.
{"preparation required": prepare_input}
```

Validate the download destinations before transferring any files.

```{code-cell} ipython3
# Keep preparation steps conditional on the same initial input check.
if prepare_input:
    # Collect each source object and its local destination.
    transfers = []
    # Validate every source path before choosing a local destination.
    for item in objects:
        # Require every listed object to belong to this dataset.
        if not item.path.startswith(prefix):
            raise ValueError("A source object is outside the dataset prefix")
        # Resolve the object path relative to the dataset prefix.
        relative = Path(item.path.removeprefix(prefix))
        # Reject paths that could escape the local dataset folder.
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("Invalid source object path")
        # Pair the source object with its validated local destination.
        transfers.append((item, raw_path / relative))
    # Reject an empty published source instead of creating an empty store.
    if not transfers:
        raise ValueError("The published source contains no downloadable files")
    # Show the number of files in the validated transfer list.
    print({"files to download": len(transfers)})
```

Download the source files and prepare the local counts.

```{code-cell} ipython3
# Recheck the destination so rerunning this cell retains the completed input.
if not data_path.exists():
    # Download the validated dataset files to their local paths.
    download_bucket_files(bucket, transfers, token=token, raise_on_missing_files=True)
    # Prepare a local count-only store without prior numerical results.
    repack_store(
        str(raw_path),
        str(data_path),
        data_only=True,
        profile="fast_local",
        nthreads=4,
        mem_budget="4G",
    )
# Refresh the preparation flag for any later rerun of the validation cell.
prepare_input = not data_path.exists()
# Check that the local input is now available.
{"prepared input exists": data_path.exists()}
```

Inspect the prepared local counts and record the original axes.

```{code-cell} ipython3
# Open the prepared input read-only for inspection.
store = scarf.DataStore(
    str(data_path), zarr_mode="r", min_features_per_cell=-1, nthreads=4, mem_budget="4G"
)
# Record the original selected-cell mask.
input_cells = store.cells.fetch_all("I").copy()
# Record the original cell identifiers.
input_ids = store.cells.fetch_all("ids").copy()
# Record the original feature identifiers.
input_features = store.RNA.feats.fetch_all("ids").copy()
# Show the selected dataset and the prepared count dimensions.
print({"dataset": dataset_id, "cells": int(input_cells.sum()), "genes": len(input_features)})
```

## Supply the scientific context

This is descriptive population discovery in the supplied IBD and healthy-control cohort.
The dataset name contains `hc`, but its disease metadata includes Crohn disease,
ulcerative colitis, and healthy controls; the full supplied cohort is retained.
Author cell-type annotations are held out. Donor or sample names do not authorize
technical correction, and cells are not independent biological replicates.
The default QC policy retains the supplied cells with advisory outlier flags.
Doublet scoring is opt-in and is left disabled here.

```{code-cell} ipython3
# Record the available metadata fields before building study context.
columns = set(store.cells.columns)
# Collect the public study metadata for the analysis context.
metadata_facts = {}
# Summarize each available field used to describe the study.
for name in ("organism", "tissue", "disease", "donor_id", "sample_id"):
    # Summarize this field only when the source provides it.
    if name in columns:
        # Read this metadata field for the supplied cells.
        values = store.cells.to_pandas_dataframe([name])[name]
        # Count the recorded levels of this metadata field.
        counts = values.dropna().astype(str).value_counts()
        # Record distinct levels and missingness for this field.
        metadata_facts[name] = {
            "distinct": len(counts),
            "missing": int(values.isna().sum()),
            "levels": {str(key): int(value) for key, value in counts.head(20).items()},
        }

import json

# Inspect the available study metadata, missingness, and recorded levels.
pd.DataFrame(metadata_facts).T
```

Describe the study using the metadata just inspected.

```{code-cell} ipython3
# Describe the cohort, permitted analysis, and held-out metadata.
study = Study(
    context=(
        "Garrido-Trigo 2023 IBD and healthy-control intestinal RNA cohort from Cytebase. "
        "Describe the supplied populations with measured markers. Author labels are held out. "
        "No technical batch correction or doublet scoring is authorized. "
        "Donors are biological identities; missing capture or replication details do not "
        "block descriptive analysis. Do not make disease-comparison or causal claims.\n"
        "Observed source metadata: " + json.dumps(metadata_facts, ensure_ascii=False)
    ),
    objective="Identify defensible major cell populations and report provisional identities and uncertainty.",
    organism="Homo sapiens",
    sampleColumn="donor_id" if "donor_id" in columns else None,
    protectedColumns=[name for name in ("disease", "tissue", "sex") if name in columns],
    excludedColumns=[name for name in columns if name.lower().startswith(("skill_", "agent_"))],
)
# Release objects that are no longer needed.
del store
# Inspect the metadata roles before sending the study to the model.
{
    "sample column": study.sampleColumn,
    "protected columns": study.protectedColumns,
    "held-out columns": study.excludedColumns,
}
```

## Run the automated analysis

The first execution starts a fresh analysis. Re-executing this cell explicitly
resumes the same saved run; completed numerical work and accepted decisions are
reused. Required missing facts or exhausted provider repairs produce a saved
non-completed outcome rather than an invented answer.

This example supplies a memorable `run_dir`. Omitting it creates
`./agent_runs/<run-id>/` in the directory where the analysis call starts.

Start a new analysis or resume the saved run.

```{code-cell} ipython3
# Set the thread and memory limits for numerical work.
runtime = RuntimeConfig(nthreads=4, memBudget="4G")
# Reuse completed local work when it is already present.
if run_dir.exists():
    # Resume the saved analysis and reuse its completed numerical work.
    run = await resume_rna_async(run_dir, model=model, runtime=runtime)
else:
    # Start a fresh analysis of the prepared input with the configured provider.
    run = await analyze_rna_async(
        data_path,
        run_dir=run_dir,
        model=model,
        study=study,
        config=AnalysisConfig(assay="RNA"),
        runtime=runtime,
    )
# Require a completed analysis before reading finalized results.
if run.status != "completed":
    raise RuntimeError(
        f"Analysis returned {run.status}; inspect its saved report before continuing"
    )
# Show the completed status and numerical pipeline invocation count.
print({"status": run.status, "pipelineInvocations": len(run.pipeline_runs)})
```

## Inspect coverage and provisional identities

The native baseline and feasible HVG, PC, and neighbor probes use the full retained
cohort. Every probe is compared with its actual parent at matching resolutions.
Marker evidence is collected for at most two finalists; finalization reuses the
selected marker artifacts and adds UMAP. These are descriptive markers rather than
replicated differential-expression tests.

```{code-cell} ipython3
# Read the measured parameter alternatives.
coverage = pd.DataFrame(run.exploration_coverage["slots"])
# Inspect which parameter alternatives were measured.
coverage[["axis", "candidateId", "status", "reason"]]
```

Read the proposed identities and their supporting rationale.

```{code-cell} ipython3
# Keep identities, confidence, and rationale together for review.
annotations = pd.DataFrame(run.annotations)[["clusterId", "identity", "confidence", "rationale"]]
# Show the complete annotation rationale without truncating table cells.
with pd.option_context("display.max_colwidth", None):
    # Review each provisional identity, confidence, and supporting rationale.
    display(annotations)
```

```{code-cell} ipython3
# Display the saved cluster UMAP from the completed analysis.
display(Image(filename=str(run.run_dir / "umap_clusters.png")))
```

Compare the saved marker evidence with the cluster layout.

```{code-cell} ipython3
# Display the saved marker dot plot supporting the provisional identities.
display(Image(filename=str(run.run_dir / "marker_dotplot.png")))
```

An `unassigned` cluster is an explicit uncertainty outcome. Review its measured
markers and QC evidence before assigning a biological name. A separated UMAP
island alone does not establish a distinct lineage.

The saved execution retained all 46,700 cells and completed all four native
representations, two marker assessments, and finalization in one uninterrupted
invocation. It used seven model requests without rejected responses or pending
questions. The selected baseline used 1,000 HVGs, 21 PCs, 11 neighbors, and Leiden
resolution 0.5, producing 18 clusters with two explicitly unassigned. Selecting
baseline settings here followed the measured alternatives; it did not skip
exploration. These provisional identities are not an annotation accuracy benchmark.

## Keep the results and report

Keep both the local dataset and the agent run directory. The dataset contains the numerical
results; the run directory contains the report, figures, annotations, and decision history.
The checks below confirm that the saved result points to the selected analysis and that the
input cells and genes stayed unchanged.

```{code-cell} ipython3
import numpy as np

# Read the compact summary linked to the final pipeline.
compact = run.compact_result
# Require the compact result stored with the completed analysis.
assert compact is not None
# Check that the compact result points to the selected pipeline.
assert compact["finalPipelineRunId"] == run.pipeline.run_id
# Read the settings selected for the final analysis.
selected = compact["selectedParameters"]
# Choose the final settings relevant to this analysis.
setting_names = [
    "requestedHvgCount", "actualHvgCount", "pcaDims", "neighborsK", "resolution", "useHarmony"
]
# Show the settings as a compact table.
pd.Series(selected, name="Final settings").loc[setting_names].to_frame()
```

Verify the stored input axes and replay the saved decisions.

```{code-cell} ipython3
# Reopen the input to check that its axes and selection were preserved.
verification = scarf.DataStore(str(data_path), zarr_mode="r", min_features_per_cell=-1)
# Check that the original selected-cell mask is unchanged.
assert np.array_equal(verification.cells.fetch_all("I"), input_cells)
# Check that the original cell identifiers are unchanged.
assert np.array_equal(verification.cells.fetch_all("ids"), input_ids)
# Check that the original feature identifiers are unchanged.
assert np.array_equal(verification.RNA.feats.fetch_all("ids"), input_features)
# Recheck recorded decisions against their saved evidence.
replayed = run.replay_decisions()
# Check that every saved decision passes offline replay.
assert all(item["valid"] for item in replayed)
# Summarize the completed consistency and replay checks.
print(
    {
        "compactResultStored": True,
        "pipelineReferenceVerified": True,
        "inputCohortUnchanged": True,
        "decisionsReplayed": len(replayed),
    }
)
# Write the report and show its path relative to the notebook folder.
print("Report:", run.report().relative_to(work_dir))
```

Open that report in a browser for the complete six-step account, recorded
limitations, cluster sizes, and marker evidence. Its HTML embeds the saved figures.
Copy both the dataset and external run directory to retain numerical results and
the full decision history. A store-only copy keeps the compact result, but its
external-history locator may need rebinding. Data-only repacking omits analysis
records, and a fresh Cytebase mount does not inherit this local agent result.

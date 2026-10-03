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
Configure `AGENT_API_KEY` in your environment or a local `.env` file. This example
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
from scarf.agent import (
    AnalysisConfig,
    RuntimeConfig,
    Study,
    analyze_rna_async,
    resume_rna_async,
)

env_file = os.environ.get("SCARF_AGENT_ENV_FILE")
_ = load_dotenv(env_file) if env_file else load_dotenv()
work_dir = Path(os.environ.get("SCARF_AGENT_NOTEBOOK_DIR", Path.cwd())).resolve()
work_dir.mkdir(parents=True, exist_ok=True)
os.chdir(work_dir)
scarf.configure_output(level="WARNING", progress=False)
logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)

dataset_id = "garridotrigo_2023_ibd_hc_10x_single_cell_transcriptomics_data_9bfecd44"
data_path = work_dir / "data.zarr"
run_dir = work_dir / "agent_runs" / "garrido-trigo"
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

if not data_path.exists():
    catalog = cytebase.Catalog()
    entry = catalog.dataset(dataset_id)
    source_uri = entry.row["zarr_uri"]
    if not source_uri.startswith("hf://buckets/"):
        raise ValueError("This copy recipe expects a Cytebase HF bucket source")
    pieces = source_uri.removeprefix("hf://buckets/").split("/")
    bucket, prefix = "/".join(pieces[:2]), "/".join(pieces[2:]).rstrip("/") + "/"
    raw_path = work_dir / "scarf_datasets" / f"{dataset_id}.zarr"
    raw_path.mkdir(parents=True, exist_ok=True)
    disable_progress_bars()
    token = get_token() or False
    objects = [
        item for item in list_bucket_tree(bucket, prefix=prefix, recursive=True, token=token)
        if isinstance(item, BucketFile)
    ]
    transfers = []
    for item in objects:
        if not item.path.startswith(prefix):
            raise ValueError("A source object is outside the dataset prefix")
        relative = Path(item.path.removeprefix(prefix))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("Invalid source object path")
        transfers.append((item, raw_path / relative))
    if not transfers:
        raise ValueError("The published source contains no downloadable files")
    download_bucket_files(bucket, transfers, token=token, raise_on_missing_files=True)
    repack_store(
        str(raw_path), str(data_path), data_only=True,
        profile="fast_local", nthreads=4, mem_budget="4G",
    )

store = scarf.DataStore(
    str(data_path), zarr_mode="r", min_features_per_cell=-1,
    nthreads=4, mem_budget="4G",
)
input_cells = store.cells.fetch_all("I").copy()
input_ids = store.cells.fetch_all("ids").copy()
input_features = store.RNA.feats.fetch_all("ids").copy()
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
columns = set(store.cells.columns)
metadata_facts = {}
for name in ("organism", "tissue", "disease", "donor_id", "sample_id"):
    if name in columns:
        values = store.cells.to_pandas_dataframe([name])[name]
        counts = values.dropna().astype(str).value_counts()
        metadata_facts[name] = {
            "distinct": len(counts),
            "missing": int(values.isna().sum()),
            "levels": {str(key): int(value) for key, value in counts.head(20).items()},
        }

import json

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
del store
```

## Run the automated analysis

The first execution starts a fresh analysis. Re-executing this cell explicitly
resumes the same saved run; completed numerical work and accepted decisions are
reused. Required missing facts or exhausted provider repairs produce a saved
non-completed outcome rather than an invented answer.

This example supplies a memorable `run_dir`. Omitting it creates
`./agent_runs/<run-id>/` in the directory where the analysis call starts.

```{code-cell} ipython3
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

api_key = os.environ.get("AGENT_API_KEY")
model = None
if api_key:
    model = OpenAIChatModel(
        "deepseek-ai/DeepSeek-V4.1-Flash",
        provider=OpenAIProvider(base_url="https://inference.baseten.co/v1", api_key=api_key),
    )
runtime = RuntimeConfig(nthreads=4, memBudget="4G")
if run_dir.exists():
    run = await resume_rna_async(run_dir, model=model, runtime=runtime)
else:
    if model is None:
        raise RuntimeError("Configure AGENT_API_KEY before starting a fresh analysis")
    run = await analyze_rna_async(
        data_path, run_dir=run_dir, model=model, study=study,
        config=AnalysisConfig(assay="RNA"), runtime=runtime,
    )
if run.status != "completed":
    raise RuntimeError(f"Analysis returned {run.status}; inspect its saved report before continuing")
print({"status": run.status, "pipelineInvocations": len(run.pipeline_runs)})
```

## Inspect coverage and provisional identities

The native baseline and feasible HVG, PC, and neighbor probes use the full retained
cohort. Every probe is compared with its actual parent at matching resolutions.
Marker evidence is collected for at most two finalists; finalization reuses the
selected marker artifacts and adds UMAP. These are descriptive markers rather than
replicated differential-expression tests.

```{code-cell} ipython3
display(pd.DataFrame(run.exploration_coverage["slots"]))
display(pd.DataFrame(run.annotations)[["clusterId", "identity", "confidence", "rationale"]])
```

```{code-cell} ipython3
display(Image(filename=str(run.run_dir / "umap_clusters.png")))
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

compact = run.compact_result
assert compact is not None
assert compact["finalPipelineRunId"] == run.pipeline.run_id
selected = compact["selectedParameters"]
display(pd.Series({
    key: selected[key]
    for key in ("requestedHvgCount", "actualHvgCount", "pcaDims", "neighborsK", "resolution", "useHarmony")
}, name="Final settings").to_frame())
verification = scarf.DataStore(str(data_path), zarr_mode="r", min_features_per_cell=-1)
assert np.array_equal(verification.cells.fetch_all("I"), input_cells)
assert np.array_equal(verification.cells.fetch_all("ids"), input_ids)
assert np.array_equal(verification.RNA.feats.fetch_all("ids"), input_features)
replayed = run.replay_decisions()
assert all(item["valid"] for item in replayed)
print({
    "compactResultStored": True,
    "pipelineReferenceVerified": True,
    "inputCohortUnchanged": True,
    "decisionsReplayed": len(replayed),
})
print("Report:", run.report().relative_to(work_dir))
```

Open that report in a browser for the complete six-step account, recorded
limitations, cluster sizes, and marker evidence. Its HTML embeds the saved figures.
Copy both the dataset and external run directory to retain numerical results and
the full decision history. A store-only copy keeps the compact result, but its
external-history locator may need rebinding. Data-only repacking omits analysis
records, and a fresh Cytebase mount does not inherit this local agent result.

# Bounded RNA analysis agents

`scarf.agent` runs a fixed RNA analysis procedure over a prepared local Scarf
store. Scarf executes numerical operations; structured model decisions interpret
context, choose a bounded PC probe, select measured finalists, and propose
cluster annotations. The model cannot execute code or introduce arbitrary tools
or pipeline settings.

The replacement implementation is contained in `scarf/agent/`. It uses existing core
Scarf APIs, numerical artifacts, and pipeline behavior without changing them.
Documentation, prompts, and defaults for this implementation live here. Agent
tests live in the repository's `tests/` directory and use its normal discovery.
Runtime prompts and defaults are Python modules, so they do not require changes
to package resource configuration.

## Start an analysis

Prepare and mount the dataset separately. The source must already be an
initialized local Scarf directory with a finalized dataset fingerprint and an RNA
assay. A Cytebase mount can read count bytes remotely despite having a local
directory. The agent does not download, convert, hydrate, mount, or publish data.

During development, a new analysis requires the selected RNA assay to have no
complete cached numerical results from earlier analyses. The read-only input
check includes artifacts inherited from a mount's source. It rejects unsupported
inputs before model calls, and checks again before the first pipeline invocation
if the run was paused. Manual feature selections, metadata snapshots, and
imported labels or embeddings remain allowed. Incomplete artifacts cannot be
reused by the pipeline and do not block admission.

The accepted run can reuse the artifacts it creates, including when resuming
under unchanged Scarf code. Starting another run directory on an already
analyzed store does not establish compatible numerical provenance. A new mount
also inherits its source's artifacts. Prepare a clean input separately when
needed; the public `repack_store(..., data_only=True)` helper can rebuild one
without saved analysis artifacts, but it copies count data and rebuilds
preparation metadata. The agent never runs that preparation or deletes artifacts
automatically. This restriction avoids changing core cache behavior.

The examples assume `model` is a configured Pydantic AI model instance, or a model
identifier supported by the installed Pydantic AI version. Configure credentials
on the provider or in its environment, not in saved runtime settings. Calling
analysis or resume can contact that provider and incur its usual usage costs.

```python
from scarf.agent import AnalysisConfig, RuntimeConfig, Study, analyze_rna

run = analyze_rna(
    "/path/to/prepared/store.zarr",
    run_dir="/path/to/analyses/rna-001",
    model=model,
    study=Study(
        context="Published RNA counts from one study; supplied cells passed author QC.",
        objective="Describe the major cell populations and uncertainty in their identities.",
        organism="Homo sapiens",
        tissue="blood",
        excludedColumns=["author_annotation"],
    ),
    config=AnalysisConfig(assay="RNA"),
    runtime=RuntimeConfig(nthreads=4, memBudget="0.5"),
)

print(run.status)
print(run.report())
```

`run_dir` is optional. Omit it to create `agent_runs/<runId>` under the current
working directory, using the same generated run ID as the saved manifest.
Supply a new path outside the numerical store to select another location.
An existing directory is rejected, even if empty. Resuming is a separate,
explicit operation. The external directory always retains the complete audit;
it is not a disposable cache.

`Study` also accepts `technicalBatchColumns`, `protectedColumns`, `sampleColumn`,
`captureColumn`, `featureExclusions`, `referenceFiles`, and
`batchCorrectionEvidence`. Declare roles from experimental facts. A model cannot
authorize technical correction just by guessing a column's meaning. Missing
replication does not block descriptive population discovery.

References must be local UTF-8 text excerpts, with a combined 16 KiB limit.
Reference text is frozen into evidence with content hashes. PDF parsing, web
retrieval, and automatic bibliography lookup are outside the procedure. Context,
references, and metadata values are treated as evidence rather than instructions.

The default `cellKey` is `I`. It must be a complete boolean selection. Supply
`assay` explicitly if the store contains multiple RNA assays. The procedure needs
at least four retained cells and three eligible features; the configured PCA rank
and neighbor count must also be feasible for the actual data. Set smaller values
explicitly for tiny evaluation datasets. The agent does not silently clamp an
infeasible requested recipe.

### Notebooks and asynchronous callers

Use the asynchronous entry points inside an existing event loop:

```python
from scarf.agent import analyze_rna_async, resume_rna_async

run = await analyze_rna_async(
    source,
    run_dir=run_dir,
    model=model,
    study=study,
    config=config,
)
```

The synchronous entry points raise if called from an active event loop. The
asynchronous interface awaits provider calls; numerical pipeline calls still run
sequentially in the invoking process. It does not create a background worker or
make CPU-bound stages nonblocking.

## Procedure and scientific defaults

The stage order is fixed:

1. Inspect the source and its nullable metadata without writing.
2. Interpret supplied context and supported metadata roles.
3. Freeze preprocessing, feature exclusions, and correction eligibility.
4. Explore registered full-cohort representations and Leiden partitions.
5. Assess at most two finalists using markers and technical/biological evidence.
6. Finalize one pinned pipeline with UMAP, reusing completed artifacts.
7. Annotate every final cluster in batches of at most eight.
8. Derive the report from saved records.

Decisions occur between complete `DataStore.pipeline` invocations. The numerical
pipeline owns artifact provenance, reuse, and stage reports. The agent's events
describe its own decisions and transitions instead of copying that numerical
ledger.

| Setting | Default |
| --- | --- |
| Additional cell filtering | Disabled; retain the selected input cohort |
| HVGs | 1,000 |
| PCs | 21 |
| Neighbors | 11 |
| Leiden resolutions | 0.5, 0.75, 1.0, 1.25 |
| Candidate representations | Four planned native trials; at most one eligible Harmony trial |
| `maxCandidates` | 5; new runs require 4 or 5 |
| `interactionMode` | `"lenient"` |
| Marker-assessed finalists | At most 2 |
| Random seed | 4444 |
| Doublet scoring | Disabled |
| Cell-cycle scoring and Paris clustering | Disabled |

The frozen native plan has four named slots: the configured baseline, one HVG
probe, one PC probe, and one neighbor probe. All three alternatives change only
one setting from the same baseline. The HVG probe uses the first feasible distinct
value from 2,000, 4,000, and 1,000; the neighbor probe uses 21, 41, and 11 in that
order. The model selects a feasible PC probe from 10 or 30 using baseline
diagnostics. A sole feasible choice is deterministic; lenient unresolved choice
prefers 30. The model cannot stop before feasible native trials are attempted.
The optional fifth slot is an eligible Harmony comparison, not another native
search or a combination of the three probes.

Each slot records measured, infeasible, failed, or pending status with a reason.
An admitted failed trial occupies its slot. A proposed larger HVG count that
produces the same actual selected genes does not establish a distinct sensitivity
comparison. Incomplete coverage stays explicit even when a usable analysis
finishes. No Cartesian grid, sampled discovery, parameter-combination synthesis,
or automatic scientific repair phase is implemented.

There are at most **seven normal pipeline invocations without Harmony**, or
**eight with Harmony**: four or five candidate screens, two finalist assessments,
and one finalization. Infeasible trials can reduce that count. An explicit
recovery after a failed invocation may create another pipeline run; these limits
are not retry, wall-time, or financial budgets. Completed numerical artifacts
remain reusable through core Scarf.

Graphs and clusters use the whole retained cohort. Silhouette assessment uses at
most 2,000 cells. Correction diagnostics build a separate nearest-neighbor graph
on at most 10,000 deterministically sampled frozen coordinate rows; doublet and
sample-support summaries use those same rows. Both branches of a correction
comparison must have the same recorded sample/method identity. Rare populations
can be absent from this diagnostic sample and are reported as a limitation.
Sampling does not change the full-cohort numerical artifacts or marker searches.
The procedure does not materialize the complete expression matrix.

### QC and feature exclusions

`qcPolicy="retain"` preserves the selected cohort and records QC outlier counts
and thresholds where metrics exist. Flags do not remove cells. Missing optional
metrics remain explicit limitations.

The evidence also projects retention under the supported global policies using
the same nullable QC measurements. Projected removal is distinguished from the
configured executed policy. Group summaries are descriptive; no per-capture
filtering thresholds are introduced.

Two policies allow caller-selected additional filtering:

```python
manual = AnalysisConfig(
    assay="RNA",
    qcPolicy="manual",
    qcBounds={
        "RNA_nFeatures": (200, None),
        "RNA_percentMito": (None, 20),
    },
)

gentle = AnalysisConfig(assay="RNA", qcPolicy="gentleMad5")
```

The manual thresholds above illustrate the interface, not universal biological
cutoffs. Bounds are inclusive. The gentle policy uses global five-scaled-MAD
bounds, with counts/features evaluated in `log1p` space. It removes low
counts/features and high mitochondrial percentages; upper counts/features remain
flags. The resolved bounds are passed through the existing manual-filtering
pipeline interface and saved as evidence. A zero MAD does not invent a cutoff.
Neither policy estimates separate thresholds per capture. Missing/nonfinite
values in metrics actively used for filtering are handled by core filtering and
can exclude affected cells.

The HVG blacklist is always supplied explicitly. Its default mitochondrial match
is case-insensitive `^mt-`, which requires suitable feature names. A supplied
organism does not automatically map feature identifiers or configure species gene
definitions. Additional `featureExclusions` are exact supplied names. HLA/H2,
sex-linked, cell-cycle, and reporter features are preserved unless explicitly
excluded.

Gene-family audits compare actual name matches with the standard Scarf blacklist
and the executed exclusions. Candidate diagnostics show selected-HVG composition,
actual selected counts, leading PC genes and their families, and bounded
PC associations with QC and supported covariates. An inferred sample/capture role
is labelled diagnostic-only and cannot authorize correction or establish
independent replication. Actual label cross-tabs are used for design evidence;
equal marginal counts are not treated as proof of confounding.

Parent/alternative partitions are compared on aligned cells at the same
resolution using adjusted Rand index and directional overlap tables. These
describe parameter sensitivity and population splits/merges; they do not prove
biological correctness. Silhouette scores are compared within their own
representation. Marker support is the fraction of clusters with at least one
marker scoring at least 0.25 and expressed in at least 20% of cells. It does not
measure coherent lineage identity, and marker scores depend on the partition.

Recognized author-annotation columns and `excludedColumns` are held out from
decision evidence. Do not also assign them technical or protected roles. Models
receive measured summaries and are instructed to cite offered evidence IDs;
selection validators reject unknown options and evidence IDs.

### Correction, uncertainty, and annotation

Harmony is only offered with opt-in doublet scoring, explicit technical batch
columns, complete categorical labels, protected biological metadata, and supplied
experimental evidence separating technical batch from biology. The current
design check requires protected groups to occur across all technical batches.
Uncertain or confounded designs remain native.

A corrected finalist must have its exact native counterpart at the same
resolution. That pair uses both finalist slots. Acceptance requires matched
mixing, biological preservation, marker, and doublet evidence, with a `0.05`
tolerance for the measured changes. At least one mixing measure must improve by
more than `0.05`; missing mandatory evidence prevents acceptance. These checks
support a bounded decision, not a claim that correction is scientifically proven.
Doublet scoring never removes cells automatically.

`interactionMode="lenient"` applies a short, frozen set of conservative policies.
Uncertain optional metadata remains unknown and disables dependent operations.
When the model identifies multiple acceptable measured choices, a deterministic
tie preference selects native before corrected, then frozen trial order, then
configured resolution order. This preference is not evidence of superiority.
The original accepted model response remains unchanged; a separate
`decisionResolved` event records the policy, resolved action, and limitation.

`interactionMode="strict"` instead retains structured questions for optional
metadata or ambiguous acceptable choices. Missing essential input remains
`needsInput` in both modes. Unsupported objectives remain explicitly unsupported.
Lenient mode does not authorize filtering, invent study facts, infer technical
permission, enable doublet scoring, change feature exclusions, or enlarge budgets.

Only recognized PCA/Harmony convergence failures in optional candidates can be
skipped in lenient mode, after verifying the terminal numerical record, source
identity, and completed parent. Baseline failure, source/storage errors, missing
or corrupt artifacts, unknown numerical errors, provider failure, exhausted
request budgets, and invalid output after the bounded repair still stop the run.
There is no automatic parameter repair or loop that retries until success.

Cluster identities are provisional. A named identity must cite at least two
observed markers with score at least `0.25` and expression fraction at least `0.2`
in the supplied cluster evidence. Supporting and
contradicting markers must be distinct and observed. Unsupported identities use
`unassigned` with an explanation and low confidence. Confidence values are
qualitative `low`, `medium`, or `high`, not calibrated probabilities. Every final
cluster must have an annotation record before the analysis completes.

## Records, limits, and recovery

The run directory contains readable files:

```text
run.json
events/000001.json
evidence/
calls/
annotations.csv
report.md
report.html
```

`run.json` freezes the initial request, scientific configuration, source locator,
and procedure identity. Inspect/preprocessing evidence freezes source
fingerprints, cohort policy, relevant metadata, and actual feature matches.
Accepted resolutions of missing inputs are recorded as events; the manifest is
not rewritten. Runtime prompts and schemas are saved with model requests, along
with model/software identity and a safe subset of effective settings.

Events are immutable, numbered, and integrity-checked. Evidence and visible model
exchanges are written atomically and referenced by events. Reports and CSV files
are derived outputs that can be regenerated.

Completed analyses also publish one compact attribute record at
`agent_results/<runId>` in the local Zarr store. The `agent_results` root is marked
as owned by `scarf.agent`; this is an agent result summary, not a core numerical
artifact or another copy of the event history. It records the agent run ID,
final core pipeline ID, assay, workspace, source fingerprint, procedure identity, selected
parameters, selection rationale, and a relative locator for the full external
audit. Selected parameters include the exact final core `pipelineConfig`,
candidate ID, resolution, requested HVG count, PCs, neighbors, Harmony flag, and
actual selected HVG count when measured.

The workspace field is `null` for the default workspace or its explicit name,
so the referenced core run can be reopened in a nondefault workspace.

The compact record is immutable and idempotent, with no mutable latest pointer.
Core artifacts and pipeline records remain authoritative for numerical results.
Read the summary through `run.compact_result` after completion; this read-only
property verifies the source and exact final pipeline. A missing record returns
`None`. Opening, reporting, or reading an older run never publishes or migrates it.
Publication errors record `resultPublicationError` without downgrading completed
scientific work; explicit completed-run resume retries publication.

The external locator is advisory rather than scientific identity. Relocating a
completed store preserves the original compact payload and can record
`resultLocatorStale`; retain the known external directory to reopen or rebind the
run. Archive the numerical store and external audit together. Summaries do not
contain transcripts, annotations, full evidence, or credentials.

| Operational setting | Default |
| --- | --- |
| `maxRequests` | 30 observed model requests across the saved run |
| `maxRequestsPerDecision` | 3 per decision in one invocation |
| Semantic repair | At most 1 per decision invocation |
| Transient transport retry | At most 1 per decision invocation |
| `maxPromptBytes` | 65,536 including serialized instructions/schema/settings |
| `maxOutputTokens` | 4,096 |
| `decisionTimeout` | 120 seconds |
| `nthreads` | 4 |
| `memBudget` | `"0.5"`, interpreted by core Scarf |

Repeated identical invalid outputs stop early. Existing model settings are
preserved except explicit runtime overrides, output limits, and the agent-wide
reasoning-off policy.
Unknown usage remains unknown. SDK-internal retries and post-response usage
accounting mean these controls cannot guarantee a hard provider spending limit.
Saved records exclude credentials, hidden reasoning, and opaque provider state.

Every decision, annotation batch, and repair requests disabled reasoning. The
provider adapter supplies `thinking=False` and these exact `extra_body` fields:

```python
{
    "thinking": {"type": "disabled"},
    "reasoning_effort": "none",
    "chat_template_kwargs": {"thinking": False},
    "reasoning": {"enabled": False},
}
```

These fields override conflicting caller settings without modifying the caller's
objects. Unrelated request settings and extra body fields are preserved. The
controlled reasoning fields are saved with request provenance; arbitrary extra
body fields remain excluded from the saved settings to protect credentials.
Provider support varies: a model with mandatory reasoning may ignore disable
settings, and a provider may reject unsupported fields. Such rejection remains
an operational failure; the agent never retries with reasoning enabled.
Truncation receives explicit feedback within the existing repair allowance.
Token and request limits are not raised automatically.

### Inspect, resume, and rebind

```python
from scarf.agent import open_analysis, resume_rna

run = open_analysis("/path/to/analyses/rna-001")
print(run.status)
print(run.pending_questions)
print(run.pipeline_runs)
print(run.candidates)
print(run.exploration_coverage)
print(run.decision_resolutions)
```

Statuses are `running`, `needsInput`, `completed`, `failed`, and `interrupted`.
Opening a run and reading its local status, questions, annotations, or completed
pipeline references does not open a provider or numerical store. A hard process
kill may leave the last status as `running`; that alone does not prove the process
is still alive.

```python
run = resume_rna(
    run.run_dir,
    model=model,
    runtime=RuntimeConfig(maxRequests=40),
)
```

Resume reuses committed decisions and completed pipelines, including a pipeline
that completed before its agent event was written. A saved visible model response
can be validated without another provider request. An explicit resume permits a
fresh bounded attempt for an unfinished decision, while all observed requests
continue to count toward the run-wide limit. The provider and operational limits
may change for unfinished work; accepted scientific decisions are not rerun.

For `needsInput`, answer the exact pending question IDs:

```python
question = run.pending_questions[0]
run = resume_rna(
    run.run_dir,
    model=model,
    answers={question["questionId"]: "RNA"},  # For an unresolved assay question.
)
```

That example applies to an unresolved assay choice. Do not reuse it for another
question. Supply all pending IDs and no unrelated keys. Answers can fill supported
missing inputs; they cannot silently replace an already fixed scientific policy.
If the required correction changes configured QC, the cohort, or established
study facts, start a new run directory. Resume without answers leaves a
`needsInput` outcome pending.

Source fingerprints, relevant ordered metadata, and the procedure identity must
still match. The identity hashes all Scarf package Python sources, including core
numerical code and agent prompts, using relative paths. It excludes tests and
generated bytecode and does not depend on Git or the installed version string.
Any Scarf Python change requires a new run, even if it does not affect the chosen
recipe. Histories made before this broader identity check also require a new run;
there are no automatic migrations. Saved outcomes and reports remain readable
through `open_analysis`. Changing a path alone does not change scientific identity:

```python
run = open_analysis(run.run_dir, source="/new/path/to/the/same/store.zarr")
# Or: resume_rna(run.run_dir, model=model, source="/new/path/to/the/same/store.zarr")
```

Rebinding is verified before numerical access or resumed computation. An
`open_analysis` override applies to that handle. A verified override supplied to
`resume_rna` is persisted as a locator event for later reopening; the immutable
manifest is not rewritten. The replacement must contain every recorded completed
pipeline and its complete artifacts, with the final artifact mapping unchanged.
A fresh mount can expose artifacts while lacking the original pipeline history;
that is insufficient for relocation. For an already inspected source, failed
validation preserves the previous locator and does not record supplied answers.
If initial inspection is still pending, the replacement locator is recorded only
after inspection succeeds. Recovery
refuses a numerical retry while the previous process may still be alive. Linux
process identities distinguish PID reuse; uncertain identity on other platforms
is handled conservatively. There is no force-resume flag.

`run.replay_decisions()` revalidates accepted decisions and their frozen evidence
offline using the procedure's stage schema and validator registries. It returns
validation results without provider calls or numerical computation. The internal
`provider.replay_decisions` helper also accepts explicit registries for developer
checks. Replay does not regenerate answers or promise identical new model
responses or bitwise numerical equality across hardware.

### Numerical results, reporting, and export

```python
print(run.annotations)  # Saved annotation records; no store access.
report_path = run.report()  # Return report.html after regenerating reports/CSV.

pipeline = run.pipeline  # Verified, read-only final core PipelineRun.
print(run.artifacts)  # Exact final core ArtifactRef objects.
print(run.compact_result)  # Verified compact store summary, or None if absent.
markers = run.get_markers(min_score=0.25, min_frac_exp=0.2)
plot = run.plot_embedding(show=False)
plot.close()
markers = run.plot_markers()  # Saved means and expressing fractions; no count reads.
markers.save("/path/to/markers.png", dpi=300, exact_size=False)
markers.close()
run.save_plots()  # Refresh both 300-DPI previews in the analysis directory.
run.report()  # Embed the saved previews without opening the numerical store.
export_dir = run.export("/path/to/new-export-directory")
```

Reports work for every saved outcome using local evidence only, even if the source
has moved or is unavailable. Six expandable steps follow the analysis: study and
input data, quality and preparation, clustering exploration, final selection,
numerical results, and provisional cell identities. Each step shows its recorded
progress and a short outcome before expansion. Completed reports open the first
step; incomplete reports open the current step. Questions and failures remain
visible above the steps. Missing stage records are labelled explicitly.

Decision explanations accompany the step they informed. The results step shows
the saved UMAP followed by the marker dotplot; identities, marker evidence,
annotation downloads, and interpretation limitations come last. Raw identifiers,
earlier recovered issues, and links to full records remain in a collapsed,
unnumbered technical appendix. The Markdown companion contains all sections
without collapsing them.

The same six steps include exploration coverage, resolved metadata roles, QC
projections, gene-family audits, candidate comparisons, and conservative policy
resolutions when recorded. These additions do not reopen the numerical source or
recompute missing diagnostics. Older saved runs remain inspectable and renderable:
missing new evidence is labelled not recorded, and a presentation-only alias reads
the old `markerCoherence` value as the same marker-support definition. Saved JSON
is not migrated or rewritten. Changed procedure identities require a new run for
execution or semantic replay; reading results and regenerating reports remains
available, and numerical export still verifies the source and exact artifacts.

An `Observed source metadata:` JSON summary in supplied context is presented as
readable field summaries and expandable category counts, preserving the
surrounding prose. Invalid or unrecognized summaries remain escaped source text.
Cluster sizes use an annotated bar chart generated from the exact selected
finalist's saved counts. Reports embed this chart and save `cluster_sizes.svg`
for reuse. Missing counts are labelled N/A, distinct from recorded zeroes.

Completed analyses automatically save UMAP and marker dotplot previews before
rendering the report. A plot failure is recorded without changing scientific
status, and the report still renders. `save_plots()` regenerates both previews
from a verified source; `report()` itself remains offline. UMAP defaults use an
8-inch figure, borderless points, a side legend, and slightly transparent colors;
`plot_embedding(...)` accepts explicit overrides and keeps Scarf's automatic
point sizing. These settings do not recompute or change the UMAP coordinates.

The marker dotplot selects up to two markers per cluster with score at least
0.25 and expressing fraction at least 0.2, capped at 40 distinct genes. Selection
cycles through cluster ranks so that each cluster's first marker is considered
before second markers. Every final cluster is displayed. Dot area shows the
expressing fraction, and color shows `log(1 + mean normalized expression)` on a
common scale. Gray crosses mean missing measurements; zero expression has no
dot. Feature indices keep duplicate gene names distinct. Statistics come from
the final immutable marker table, with one cluster table read at a time; no
count matrix is read. `plot_markers(top_n=..., max_genes=...)` adjusts panel size.

Reports embed Inter, the supplied Nygen and Scarf logos, and the favicon. The
responsive layout uses Nygen's blue accent, regular headlines, light subheadlines,
and 1.2 line spacing, with print styles and keyboard navigation. No script,
stylesheet service, or font service is required. The Inter SIL Open Font License
is retained in an HTML source comment. The masthead links to Nygen's website;
the footer links to the Scarf repository and paper alongside its full citation.
These are ordinary navigation links and do not load remote report assets.
Narrative text supports paragraphs, line
breaks, lists, emphasis, and inline code, including newline escapes in saved model
responses. Inline code preserves literal identifiers and paths. In HTML, supplied
markup is escaped, and narrative links and images remain inactive text.

Only the exact selected finalist supplies displayed cluster counts and marker
measurements. Missing measurements remain unknown; recorded zeroes stay zero.
Each visible table or repeated section is limited to 100 entries; the annotation
CSV and underlying saved records remain complete. An existing regular local
`umap_clusters.png` or `marker_dotplot.png` of at most 8 MiB can be embedded as a
saved preview. Reporting derives the cluster-size chart from saved summaries;
it never opens the numerical store, recomputes UMAP or markers, or follows a
preview symlink. A rendering error cannot
change scientific completion; analysis/resume record the report error separately.

Numerical access requires an available source with a matching fingerprint and an
exact completed final pipeline. Marker reads and plots stay tied to that final
run. Plot options cannot replace the run or its UMAP identity; frozen run fields
can be used for coloring through the existing plotting accessor.

Export requires a new directory outside the numerical store. It writes
`summary.json`, `clusters.csv`, `umap.csv`, `markers.csv`, and `annotations.csv`.
Cluster and UMAP CSV files share the frozen selected-cell order and explicit cell
IDs. Annotation records are checked against the final clustering. Export does not
rerun numerical stages or contact a model. Files are published atomically one at
a time; an interrupted export can leave a partial directory, so use a new export
directory for a fresh attempt.

### Side effects and boundaries

Analysis and resume create ordinary core pipeline runs and immutable artifacts
inside the selected writable store, plus the compact completed agent result.
They do not change live `I`, feature
metadata, or annotation columns. Numerical access through `AnalysisRun` opens the
source read-only. The external run directory contains supplied study text,
metadata summaries, references, decisions, and results; nothing is committed or
published automatically.

Agent writers use persistent adjacent lock files named
`.<run-name>.scarf-agent-run.lock` and `.<source-name>.scarf-agent.lock`. Their
parent directories must be writable. The lock inode remains after release to
avoid races. Locks coordinate this agent implementation; unrelated programs that
ignore them are not prevented from writing the source.

Deferred capabilities include ingestion, multimodal analysis, automatic extra
filtering, doublet removal, per-capture QC policies, sampled discovery,
subclustering, automatic scientific repair, differential-expression hypothesis
testing, causal claims, and online reference retrieval. An unsupported capability
does not justify changing core Scarf or introducing private storage mutations.

## Code map, migration, and local verification

The public interface is `scarf.agent`; the root `scarf` facade is unchanged.
`api.py` owns entry points, `workflow.py` owns stage transitions, and `models.py`
defines inputs and structured choices. `evidence.py`, `choices.py`, and
`execution.py` prepare and validate evidence and adapt it to existing pipelines.
`diagnostics.py` computes bounded descriptive summaries from frozen metadata
and public artifacts.
`provider.py` handles model calls; `prompts.py` contains their instructions.
`records.py` persists local history. `result.py` and `rendering.py` expose results
and derived outputs. The compact store summary links the selected core pipeline
to the complete external audit without changing the core artifact model.

This is a clean break from the prototype. `AgentOrchestrator`,
`AutomatedWorkflowResult`, `AnalysisError`, old `orchestrator.*` imports, and old
standalone agent APIs are not compatibility entry points. Migrate callers to the
new functions and inspect `AnalysisRun.status`; a persisted incomplete outcome
usually returns a result rather than raising the prototype's beginner-API error.
Invalid entry arguments and integrity violations still raise, and interruption
exceptions propagate after recording the interrupted status.

Prototype agent histories are left untouched and cannot be resumed or migrated
by this implementation. Their numerical artifacts remain accessible through core
Scarf. The replacement tests now live in `tests/test_agent_*.py`; prototype-only
tests and their unused helpers were removed. Independent core coverage from old
agent test files was retained in the corresponding core test modules. Existing
external launchers and CI configuration were not rewritten. The agent tutorial,
API reference, and analysis guide now describe this interface, and the worked
tutorial uses an offline scripted provider. Remaining consumers of removed
prototype APIs need a separate migration.
The [validation and migration inventory](VALIDATION.md) lists the observed
validation results and affected external consumers.

Run the focused agent tests, or use the repository's standard test commands:

```bash
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/scarf-agent-uv \
  uv run --no-sync pytest -n 0 -o cache_dir=/tmp/scarf-agent-pytest tests/test_agent_*.py

uv run --no-sync pytest -m "not slow and not integration"
uv run --no-sync pytest
uv run --no-sync ruff check scarf profiling tests
uv run --no-sync ruff format --check scarf profiling tests
uv run --no-sync mypy scarf profiling
```

The agent quality gate requires at least **95% line coverage** across every
module in `scarf/agent/`. Run it with the installed coverage plugin:

```bash
COVERAGE_FILE=/tmp/scarf-agent.coverage PYTHONDONTWRITEBYTECODE=1 \
  UV_CACHE_DIR=/tmp/scarf-agent-uv \
  uv run --no-sync pytest -n 0 -o cache_dir=/tmp/scarf-agent-pytest \
  tests/test_agent_*.py --cov=scarf/agent --cov-report=term-missing \
  --cov-report=html:/tmp/scarf-agent-coverage-html \
  --cov-precision=2 --cov-fail-under=95
```

This command fails below the threshold. It measures package line coverage,
including modules that were not imported; it does not measure branch coverage.
No agent files or defensive paths are excluded. The repository CI configuration
is unchanged, so run this explicit gate when validating agent work.

The shared `tests/fixtures_agent.py` plugin disables real provider requests for
agent test modules without changing other tests' provider settings. Tests use
temporary stores and scripted models, including a small real numerical pipeline. Test success does
not establish real-provider reliability or annotation quality. Live evaluation
must be authorized separately and should measure completion/blockage reasons,
repair rates, usage, runtime, data-read volume, QC retention, and annotation
quality. Published annotations are evaluation references, not decision inputs or
unquestionable truth.

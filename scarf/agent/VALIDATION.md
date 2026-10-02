# Validation and external migration inventory

The agent implementation remains in `scarf/agent/`. Its replacement tests now
live in `tests/test_agent_*.py` and participate in normal repository discovery.
The test relocation and prototype-test cleanup were separately authorized after
the initial implementation. Core Scarf and external skills remain unchanged.

## External audit and compact result documentation

The storage contract keeps the full readable audit outside the numerical store.
Omitting `run_dir` selects `Path.cwd() / "agent_runs" / <runId>`; an explicit new
external directory remains supported. The same ID identifies the immutable
manifest and the compact completed result under local Zarr
`agent_results/<runId>`.

That compact attribute record links the exact final core pipeline and its
configuration to the selected candidate, measured HVG count when available,
selection rationale, workspace, source/procedure identities, and external audit
locator. A `null` workspace means the default; an explicit workspace name keeps
nondefault core pipeline references reopenable.
It does not move or remove events, prompts, measured evidence, annotations,
or reports. `AnalysisRun.compact_result` reads the record after source/final-run
verification and never creates or migrates it. Publication failures remain
separate from scientific completion; a completed explicit resume can retry.
Relocation preserves published payloads and treats the external locator as
advisory. Core pipeline/artifact formats are unchanged.

The agent tutorial, API reference, and relevant analysis-guide sections now use
the replacement public API. The executable teaching fixture uses a small
synthetic prepared store and a local `FunctionModel`, exercises the four native
representations, and leaves synthetic identities unassigned. It requires no
download or model credentials. Documentation execution and build results for this
revision are recorded separately from the earlier test results below.

## Test organization

- Nine replacement test modules moved out of the installed agent package, with
  their cross-module imports updated.
- `tests/fixtures_agent.py` supplies the temporary RNA source and disables live
  model requests for agent test modules. It is registered by `tests/conftest.py`.
  Other test modules retain their existing provider settings and thread limits.
- Seventy-five prototype test modules and four unused prototype fixture/helper
  modules were removed. The duplicated provider-request guard was consolidated.
- Independent H5AD scalar parsing and nullable metadata fingerprint coverage was
  retained in `tests/test_h5ad_inspect_columns.py` and
  `tests/test_metadata_blocks.py`.
- Dependency-isolation coverage was retained in
  `tests/test_import_architecture.py`, whose agent checks now describe the current
  small package. Core graph-consumer checks remain in place.
- The retired Loom import assertion also accepts an absent agent-ingestion parent
  package, so it works in a clean checkout without obsolete namespace directories.

## Bounded exploration and lenient policy validation

The exploration update preserves the six-step report, its embedded assets,
narrative formatting, source-metadata tables, UMAP/marker previews, annotated
cluster-size SVG, accessibility, and offline rendering contract. New scientific
evidence is displayed within the existing steps. Policy resolutions remain
separate from accepted model responses. Missing historical coverage stays unknown.

The report, context, cluster-size, and result suite passed **145 cases** during
this update: all 136 existing cases plus nine focused additions. New cases cover recorded exploration coverage,
conservative resolutions, the presentation-only legacy marker-support alias,
zero versus missing measurements, QC projections, inferred roles, gene-family
and PC diagnostics, aligned parent comparisons, and escaped evidence. These
checks use saved temporary fixtures and do not open live providers or datasets.
Ruff check/format and patch-whitespace checks passed for these changes. A focused
mypy invocation found no errors in the reporting/result files; workflow checks
remain part of integration validation.

Integration checks for the exploration update passed:

| Check | Result |
| --- | --- |
| Final targeted agent suite with the 95% package coverage gate | 584 passed; 98.57% line coverage; every module exceeds 95% |
| Unchanged pipeline, artifacts, frozen-store, Cytebase, and plotting regressions | 185 passed |
| Quick repository suite, `pytest -n 4 -m "not slow and not integration"` | 7,683 passed, 1 skipped |
| Final complete repository suite, `pytest -n 4` | 7,794 passed, 5 skipped |
| `ruff check scarf profiling tests` | Passed |
| `ruff format --check scarf profiling tests` | Passed; 570 files checked |
| `mypy scarf profiling` | Passed; 299 source files checked |

The real numerical fixture exercises four native representations, seven pipeline
invocations, exact marker reuse, unchanged live metadata, held-out labels, and a
separately recorded lenient selection resolution. Large-prompt regression cases
exercise the complete serialized provider request for four representations,
two finalists, and an eight-cluster annotation batch against the 65,536-byte limit.
Compaction preserves the complete original evidence and records omitted prompt
detail. The final targeted aggregate also covers edited replay histories,
misrouted deferrals, invalid artifacts, cohort/feature-axis mismatches, missing
measurements, and excessive metadata categories. Diagnostics and report modules
have 100% line coverage. The quick suite predates the final edge tests and live
provider contract fixes; the final complete suite includes them and the
reasoning-off policy. Branch coverage was not measured. The full suite also
exposed an agent plotting test that assumed no unrelated figures were open. Its
assertion now compares the complete pre-existing figure set before and after a
failed plotting call. The 18-case plotting module and a reproduction with two
unrelated figures passed before the final full-suite rerun; production plotting
behavior was unchanged.

Live evaluation exposed a context citation mismatch and an ambiguous PC action
contract. Supplied role IDs now share the context validator's registered ID set,
and requests expose their valid citations and actions explicitly. Truncated
provider output receives actionable feedback within the existing single repair.
These cases have offline regressions; retries, token limits, and scientific gates
were not enlarged to accommodate them.

The provider adapter now applies one reasoning-off request policy to every
decision, annotation, semantic repair, and transport retry. It supplies the exact
four controlled extra-body fields documented in the README, neutralizes supplied
native reasoning overrides, preserves unrelated settings, and leaves caller
objects unchanged. Mocked HTTP tests inspect the serialized provider payload;
saved-request checks ensure only the controlled body fields enter provenance.
This verifies what the agent requests, not internal reasoning in providers that
ignore disable controls or always require reasoning.

These offline checks do not establish biological quality or live-provider
reliability. The prior numerical and coverage results below are historical.

## Live evaluation with reasoning disabled

Five fresh analyses used DeepSeek V4.1 Flash through the configured Baseten
provider. Every one of the 38 saved requests contains the exact reasoning-off
extra body, unified `thinking=False`, and native OpenAI reasoning effort `none`.
Credentials were loaded without being displayed. Published sources were accessed
read-only; intact local snapshots were prepared into fresh stores with existing
public Scarf operations. Original mounts and prior analyses were preserved.

| Dataset | Retained cells | Clusters | Unassigned clusters | Requests | Rejected responses | Explicit resumes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Szabo | 3,726 | 11 | 0 | 9 | 3 | 1 |
| Tran | 11,202 | 19 | 9 | 7 | 0 | 0 |
| Solé-Boldo | 15,457 | 15 | 1 | 6 | 0 | 0 |
| Wilk | 44,721 | 16 | 0 | 9 | 3 | 1 |
| Garrido-Trigo | 46,700 | 19 | 3 | 7 | 0 | 0 |

All five completed with four measured native representations and seven pipeline
invocations each. Cohort and feature ordering/selections match the original
inputs. No QC removals, correction, or doublet scoring were requested. Final
marker artifacts were reused exactly, saved decisions replay successfully, and
exports and embedded report figures are aligned with their final clustering.

First-attempt completion was three of five. Szabo stopped at finalist selection
and Wilk at PC selection after exhausting semantic repair; each completed on one
explicit operational resume with unchanged policy and budgets. These resumes
were triggered by the evaluation operator, not automatically by the workflow.
All prior pipeline invocations were reused, and failure events remain saved.

The prior batch also required two explicit resumes. Compared with that batch,
observed requests fell from 46 to 38, rejected responses from 14 to 6, input tokens
from 522,249 to 373,243, and output tokens from 126,828 to 16,722. Token-limit
rejections fell from 13 to zero; the six new rejections were structured-choice
validation errors. First-attempt completion did not improve. Biological choices
and some final partitions changed; annotations remain provisional and no label
accuracy benchmark was run. This small evaluation covers one live provider.
Count-byte read volume during numerical execution was not instrumented.

## Previous report validation

The report and plotting checks include **104 report cases** and **18 marker-plot
cases**. The targeted agent suite passes **487 cases** with **99.31% line
coverage**: 2,440 of 2,457 executable statements across 18 modules. The reporting
and plotting modules have 100% line coverage; every agent module exceeds 95%.
Branch coverage was not measured.

Checks cover readable headings and tables, all saved outcome statuses, exact
selected-cluster evidence, missing measurements versus zeroes, retained
correction diagnostics, escaping in HTML and Markdown, safe bounded image reads,
embedded brand assets and the font license, complete CSV exports, numeric cluster
ordering, and offline regeneration without changes to scientific records.
Narrative regressions cover actual and escaped line breaks, paragraphs, nested
lists, emphasis, inline code, Unicode, literal gene patterns and file paths,
malformed markup, and inactive supplied links and images.
Chronological report checks cover all six steps, outcomes visible before
expansion, decisions beside the relevant evidence, plots before identities,
current-step expansion, and visible questions or failures. Missing completion
records stay explicit. A resumed failure takes precedence over an earlier stage
completion without hiding the saved results. Markdown retains all step content.
Source-context checks cover named JSON metadata summaries, readable field and
category tables, surrounding prose, malformed-input fallback, escaping, and
bounded output. Cluster-size checks verify annotated counts, relative bar sizes
on a zero-based scale, missing values, natural ordering, exact finalist selection,
deterministic standalone SVG output, offline generation, and symlink protection.
Plot checks cover exact final-artifact binding, read-only access, bounded marker
selection, duplicate gene names, missing statistics versus zero expression,
fraction-to-area scaling, 300-DPI atomic saves, figure cleanup, user overrides,
and plot failures that preserve completed scientific status. The real numerical
workflow produces both report figures without extra pipeline invocations.

| Report redesign check | Result |
| --- | --- |
| `pytest -n 0 tests/test_agent_*.py --cov=scarf/agent --cov-fail-under=95` | 487 passed; 99.31% line coverage |
| Ruff check and format check on the agent package and changed report tests | Passed |
| `mypy scarf/agent` | Passed; 18 source files checked |
| Local Chromium at 1440 px and 390 px widths | Embedded Inter loaded; no page overflow; readable tables, marker details, paragraphs, and lists |
| Keyboard access to wide tables | Focusable scroll regions with accessible names |
| Chronological steps in local Chromium | Keyboard Enter toggles sections; closed sections and nested evidence remain visible when printing |
| Source metadata and cluster-size figures | Category tables retain exact counts; annotated bars remain readable on desktop, scroll within the mobile report, and fit print output |
| Company and Scarf resource links | Correct masthead/footer destinations and complete citation; no external requests on load; desktop/mobile layout passes |

All six reports under `cytebase_analysis/outputs/agent_rerun_20261001_1852` were
regenerated. Hash comparisons confirmed that manifests, event histories,
measured evidence, original images, and annotation CSV contents stayed unchanged.
Each report also has a derived `cluster_sizes.svg` figure from saved counts.
No provider or numerical store was opened by report regeneration. The complete
repository suite was not run for this task, as requested. The results below
predate the report redesign.

The later attempt to refresh the six example figures could not open their
mounted backing stores: each raised `GroupNotFoundError`, including with the
original launcher's environment loaded. Their existing images were preserved;
HTML and Markdown were regenerated offline. New UMAP and marker figures were
validated against the local real numerical fixture. Once the backing stores are
available, `open_analysis(path).save_plots()` followed by `.report()` refreshes
the example figures.

## Master compatibility validation

After the master merge and agent compatibility fixes, the focused suite contained
**358 passing cases**. Line coverage is **99.07%**: 1,815 of 1,832 executable
statements across all 12 agent modules. The explicit `--cov-fail-under=95` gate
passes. No coverage exclusions, dependency settings, or CI configuration changed.
Branch coverage was not measured.

The 51 new regression cases cover the expanded Scarf source identity, rejection
of complete numerical artifacts from earlier analyses, permitted imported labels
and embeddings, cache inheritance through mounts, and validation before saving a
replacement locator. They also cover relocation during inspection, unanswered
questions, incomplete or missing pipeline history, and successful relocation of
a complete store copy. Existing same-run pipeline and marker reuse tests pass.

Additional cases exercise scientific input and QC validation, complete matched
correction selection and rejection, source relocation, resumable questions,
interrupted or corrupt records, provider budgets and replay, exact final artifact
identity, and offline reporting/export safeguards. Scripted decisions and bounded
numerical doubles supplement the existing real-store pipeline tests.

| Agent module | Line coverage |
| --- | ---: |
| `api`, `choices`, `evidence`, `models`, `rendering`, `result` | 100% |
| `__init__`, `prompts` | 100% |
| `execution` | 99.60% |
| `workflow` | 99.27% |
| `provider` | 98.79% |
| `records` | 95.55% |

All modules individually exceed 95%. Remaining misses include platform-specific
locking and defensive failure paths. Use the reproducible gate command in
[README.md](README.md); the terminal report lists each uncovered line.

| Check after the master compatibility fixes | Result |
| --- | --- |
| Agent tests with `--cov=scarf/agent --cov-fail-under=95` | 358 passed; 99.07% line coverage |
| Quick suite, `pytest -n 4 -m "not slow and not integration"` | 7,503 passed, 1 skipped |
| Complete suite, `pytest -n 4` | 7,568 passed, 5 skipped |
| `ruff check scarf profiling tests` | Passed |
| `ruff format --check scarf profiling tests` | Passed; 554 files checked |
| `mypy scarf profiling` | Passed; 292 source files checked |

The complete suite discovered 7,573 tests. Its five skips have the same
environment and opt-in causes described below; no tests were skipped or removed
for these fixes. All changed files remain under `scarf/agent/` or are agent test
modules under `tests/`.

Before the master merge, the earlier coverage expansion reached 307 passing
agent cases and 98.99% coverage, up from 86.47%. That revision passed 6,295 quick
suite cases and 6,356 complete suite cases. Those are historical results; the
compatibility checks above include the new upstream tests and agent regression cases.

## Test relocation validation

At relocation, normal collection discovered **6,170 tests without collection
errors**. Before that cleanup, the quick/full suites both stopped at 74 imports
of prototype modules. Those obsolete collection errors were removed rather than
hidden with skips or compatibility code. The following results precede the
additional coverage tests above.

| Check | Result |
| --- | --- |
| Relocated agent tests plus H5AD, metadata-block, and public-API tests | 158 passed, including all 116 replacement agent tests |
| Import architecture and graph-consumer tests | 46 passed |
| Normal repository collection | 6,170 tests, no collection errors |
| Quick suite, `pytest -n 0 -m "not slow and not integration"` | 6,104 passed, 1 skipped, 65 deselected |
| Complete suite, `pytest -n 4` | 6,165 passed, 5 skipped |
| `ruff check scarf profiling tests` | Passed |
| `ruff format --check scarf profiling tests` | Passed; 526 files checked |
| `mypy scarf profiling` | Passed; 290 source files checked |

The quick-suite skip reflects the installed AnnData version rejecting a column
name containing `/`. The complete suite also skips three checks that require an
external master CITE-seq corpus and the opt-in visual reference comparison.
Test stores and explicit caches use temporary directories, and no dependency
synchronization or live-provider evaluation is performed.

## Prior implementation checks

Before relocation, all 116 replacement tests passed in the agent-local directory.
Separately, 495 selected core pipeline/artifact/offline-Cytebase/plotting tests and
three frozen-store compatibility tests passed. Ruff and mypy also passed. These
are historical results; the relocated suite is validated independently above.

## External documentation and callers

The original inventory came from read-only source searches. The current
documentation update migrates the three agent-facing pages listed below; old
validation counts above remain historical. Archived reports remain historical
evidence, rather than migration work items.

| Existing consumer | Incompatibility and required later migration |
| --- | --- |
| [Cytebase launcher](../../cytebase_analysis/run_cytebase_agent.py), imports around lines 27 and 127; execution around line 244 | Uses `AutomatedWorkflowResult`, `AutomatedWorkflowConfig`, `FinalAnalysisHandoff`, the retired ingestion helper, and `AgentOrchestrator`. Replace launch/configuration/result/export handling with the new public API and prepared-store requirement. |
| [Study-specific launcher](../../cytebase_analysis/run_cytebase_agent_wilk2020.py), line 11 | Imports the old launcher's `main`; inherits that migration requirement. |
| [Agent API reference](../../docs/source/reference/api/agent.md) | Migrated to typed inputs, status/result access, explicit resume, optional external run directory, compact store result, and current provider limits. |
| [Agent analysis guide](../../docs/source/analysis_with_agents.md) | Migrated its automated-agent section to fixed full-cohort exploration, opt-in correction evidence, conservative policies, and current persistence. General guidance for agents using granular Scarf APIs remains distinct. |
| [Agent tutorial](../../docs/source/tutorials/agent_workflow.md) | Replaced prototype calls and inline retired-provider fixture with the current API and an offline synthetic numerical example. |
| [Architecture guide](../../docs/source/developers/architecture.md) | Updated the current facade, fixed pipeline orchestration, external audit, compact local result, and plotting ownership. Historical removed-symbol notes remain historical. |
| [API overview](../../docs/source/reference/api.md) | Updated the public surface to include explicit resume and inspectable incomplete statuses. |

The earlier targeted search found no separately maintained source `.ipynb`
notebook outside documentation execution/build caches. The later notebook and
documentation update is separately authorized from that original test cleanup.
Remaining prototype launchers need explicit migration. Scripted-provider tests
do not establish live-provider reliability or biological annotation quality.

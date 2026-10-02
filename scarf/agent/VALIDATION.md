# Validation and external migration inventory

The agent implementation remains in `scarf/agent/`. Its replacement tests now
live in `tests/test_agent_*.py` and participate in normal repository discovery.
The test relocation and prototype-test cleanup were separately authorized after
the initial implementation. Core Scarf and external skills remain unchanged.

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

## Current validation

The focused agent suite now contains **307 passing cases**, including **191
additional cases** for the coverage requirement. Line coverage across the package
increased from **86.47% to 98.99%**: 1,770 of 1,788 executable statements across
all 12 agent modules. The explicit `--cov-fail-under=95` gate passes. No production
code, coverage exclusions, dependency settings, or CI configuration changed to
reach this result. Branch coverage was not measured.

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
| `records` | 95.14% |

All modules individually exceed 95%. Remaining misses include platform-specific
locking and defensive failure paths. Use the reproducible gate command in
[README.md](README.md); the terminal report lists each uncovered line.

| Check after adding coverage tests | Result |
| --- | --- |
| Agent tests with `--cov=scarf/agent --cov-fail-under=95` | 307 passed; 98.99% line coverage |
| Quick suite, `pytest -n 4 -m "not slow and not integration"` | 6,295 passed, 1 skipped |
| Complete suite, `pytest -n 4` | 6,356 passed, 5 skipped |
| `ruff check scarf profiling tests` | Passed |
| `ruff format --check scarf profiling tests` | Passed; 529 files checked |
| `mypy scarf profiling` | Passed; 290 source files checked |

The complete suite discovered 6,361 tests. Its five skips have the same
environment and opt-in causes described below; none was added for coverage.

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

These findings came from read-only source searches. They were not rewritten or
executed. Archived reports remain historical evidence, rather than migration
work items.

| Existing consumer | Incompatibility and required later migration |
| --- | --- |
| [Cytebase launcher](../../cytebase_analysis/run_cytebase_agent.py), imports around lines 27 and 127; execution around line 244 | Uses `AutomatedWorkflowResult`, `AutomatedWorkflowConfig`, `FinalAnalysisHandoff`, the retired ingestion helper, and `AgentOrchestrator`. Replace launch/configuration/result/export handling with the new public API and prepared-store requirement. |
| [Study-specific launcher](../../cytebase_analysis/run_cytebase_agent_wilk2020.py), line 11 | Imports the old launcher's `main`; inherits that migration requirement. |
| [Agent API reference](../../docs/source/reference/api/agent.md), lines 4-12 and 78 | Documents the old three-export facade, retired result/error types, old `analyze_rna` arguments, and advanced orchestrator. Rewrite around the new entry points, typed inputs, statuses, and `AnalysisRun`. |
| [Agent analysis guide](../../docs/source/analysis_with_agents.md), lines 100-131 and 215-224 | Describes the retired orchestration journal, old facade, and standalone agent packages. Update procedure, recovery, and result ownership. |
| [Agent tutorial](../../docs/source/tutorials/agent_workflow.md), lines 57, 175, 213-223, and 671 | Calls old analysis signatures and imports removed enrichment/context/ingestion packages. Rewrite its executable examples before refreshing documentation. |
| [Architecture guide](../../docs/source/developers/architecture.md), lines 143 and 491 | Names retired public types and `AgentOrchestrator.initialize_request`. Update placement and lifecycle descriptions. |
| [API overview](../../docs/source/reference/api.md), line 15 | Describes analysis as returning only a completed result. Explain the persisted incomplete statuses and explicit resume. |

No separately maintained source `.ipynb` notebook was found by the targeted
search outside documentation execution/build caches. Cached notebooks were not
executed or rewritten.

These external consumers were not migrated as part of the test cleanup. No
documentation build or publication was performed. Scripted-provider tests do not
validate live-provider reliability or biological annotation quality.

# Bulk and doublet memory remediation

Target: approximately one million selected cells and 50,000 features on a machine
with 32 GiB RAM. Use a 24 GiB Scarf planning budget to leave space for Python,
native libraries, compression, and allocator overhead. The budget governs known
allocations; it is not a hard RSS limit.

Bulk retains the feature-by-group output and cell grouping vectors; expression
scratch is limited to one active feature/cell band. Doublets retain the sampled
CSR pool, linear pair/statistic vectors, and one score per reference cell.
Synthetic expression is limited to one batch. ANN and graph sizes remain
resident costs, and graph smoothing stores edges without materializing graph
powers. Density, cluster count, and output size can still make admission fail.

## Findings and review decisions

`max_cells_per_cluster=100` limits the sampled parents. It does not limit the
synthetic population, which is `round(simulation_ratio * selected_cell_count)`.
With 100 clusters, at most 10,000 parents can still produce one million doublets.
The previous implementation retained dense parents, their CSR conversion, the
complete synthetic CSR, a temporary query store, mapping artifacts, and a powered
diffusion matrix across parts of the workflow.

The external review correctly identified the synthetic matrix and graph powers
as avoidable peaks. Its claim that bulk necessarily decodes an entire
123,020-cell group into a 22 GiB block was not supported by the current reader.
The inspected store uses 10,982-row count bands. One full-width uint16 band is
about 0.93 GiB; its float64 normalization is about 3.73 GiB. Concurrent bands,
normalization temporaries, and retained reduction partials are the relevant bulk
risks. Existing axis-zero reductions retain partial vectors until their merge.

Two further reviewer suggestions need narrower treatment:

- Raw RNA sums can use countsT regardless of the configured normalizer. Only
  means require `lib_size_feature_stream_eligible(assay)`. Custom and log RNA
  means, ATAC, and ADT retain their current path and its memory limitations.
- Doublets come from the reference assay itself. Validate the frozen feature
  selection and exact feature order, then fail on inconsistency. There is no
  need for another feature alignment or missing-gene filling implementation.

## Implementation

1. Add one reducer in `scarf/features/aggregation.py`, called by `make_bulk`.
   Use `map_feature_cell_bands` over countsT, with ordered compute, one group-code
   vector, and shared output arrays. Preserve group ordering, selection,
   exclusions, seeded pseudo-replicates, and feature labels. Exclude skipped
   cells before streaming and use each band's selected destinations for codes
   and scalars. Process sums, means, and expression fractions in the same pass.
2. Preserve NumPy's promoted sum dtype, including integers above `2**53`.
   Convert mean inputs to float64 before multiplying by the size factor. Read
   stored library totals. Preserve bulk's existing zero-total and NaN behavior.
   Emit matching zero fraction columns for empty groups and pseudo-replicates.
   Reserve output and DataFrame copies, metadata/index vectors, read/decode
   buffers, casts, and expression masks before allocating the result.
3. Keep doublet algorithms in `scarf/quality_control/doublets.py`. Read sampled
   parents in budgeted row blocks, retain CSR parts, and concatenate once.
   Admission includes the growing pool, conversion scratch, and concatenation
   copy. Preserve the existing parent sampling and complete pair RNG sequence.
4. First synthesize bounded batches across all features to collect library
   totals and positive feature counts. Preserve the temporary datastore's
   filter: keep all rows if median detected features is below 10; otherwise
   keep rows with more than 10 detected features. Fail if no rows survive.
5. Restrict the parent pool to validated reference features and release the
   full-feature pool before loading ANN. Replay kept pairs in budgeted batches.
   Reuse normalization, `project_pca`, `zero_norm_rows`, `NeighborQueryStage`, and
   `mapping_score_weights`. Clamp positive `save_k` to the reference's available
   neighbors. Exclude zero-norm projections. Accumulate one float64 reference
   score vector, then globally scale by `1000 / (informative_count * k)` and
   apply `log1p`. Preserve checked integer addition and overflow errors.
6. Release simulation and ANN locals before loading the symmetric graph. Build
   `diffusion_operator(graph, power=1)` once and apply it to the score vector
   `smoothing_t` times. Preserve isolated-row and constant-score behavior.
   Admit graph loading, symmetrization, and transition construction from stored
   edge sizes before reading those arrays. Account for resident graph caches.

The only shared-code changes are moving the existing reference-neighbor helpers
into the mapping domain and extracting numerical normalization from
`AlignedFeatureStream`. Reuse the existing planners and profiling stages. Add no
dependencies, public tuning flags, alternate algorithms, or profiling framework.

## Contracts and callers

`make_bulk` callers in tutorials, tests, and `profiling/stages.py` retain the same
signature and table structure. The RNA reducer fixes integer mean overflow and
missing fractions for empty columns. Other normalizers retain existing behavior.

Both `DataStore.run_doublet_detection` and the private pipeline entry point in
`datastore/pipeline_accessor.py` use the same streamed core. Preserve
`DoubletScoreArguments`, `count_arithmetic="checked_integer_sum"`, score schema,
frozen selections, lineage, and cache identity. Final score comparisons use
floating-point tolerances because reduction order changes.

Doublet execution no longer creates a temporary query store, projection, or
diffusion artifact. It still builds or reuses the mapping reference. Explicit
diffusion and imputation APIs remain unchanged. Keep the existing temporary
datastore factory and synthetic-store writer; use the writer as the small-store
parity oracle, with no production fallback.

## Validation and scale gate

- Bulk: explicit-array parity for selection, groups, exclusions, pseudo-reps,
  fractions, labels, empty columns, stored totals, zero totals, large integers,
  custom/log normalizers, and insufficient-memory rejection.
- Doublets: actual ANN parity against the materialized small-store workflow,
  full-library and subset normalization, log transform, neighbor clamping,
  batch independence, filtering around 10 features, checked overflow, and
  insufficient-memory rejection.
- Smoothing: compare repeated multiplication to powered diffusion on sparse
  graphs with hubs and isolated rows; include constant scores and invalid powers.
- Integration: unchanged cell metadata, no query/projection/diffusion side
  effects, exact cache reuse, query failure without a completed score, mapping,
  pipeline, public/artifact contracts, and import architecture.
- Run the focused, quick, and complete suites plus Ruff and mypy.

After local validation, use the four existing bulk and doublet consume stages
on the readable R2 million-cell store. The inspected selection contains 889,974
cells, 45,525 features, and 43 clusters; it is not literally the nominal target.
Measure process RSS and cgroup peaks under the 32 GiB limit, with a 24 GiB Scarf
budget. Check output shapes, bulk group totals and gene spot checks, finite
non-degenerate scores, and the normalized score range. Test denser parent pools
and larger group counts separately: sparsity and output size still matter.

Local implementation and validation are complete:

- Quick suite: 5,239 passed.
- Complete suite: 5,301 passed; one opt-in visual regression test skipped.
- Ruff lint and formatting checks, mypy, and whitespace checks passed.
- A first-pass probe with one million pairs and a sparse 1,000-by-2,000 parent
  pool used about 16.2 MiB of additional traced allocations. This excludes the
  existing pool and pair arrays and does not measure full-pipeline RSS.

Cloud scale validation remains pending user authorization. Modal deployment is a
user action. No million-cell RSS guarantee should be claimed before measurement.

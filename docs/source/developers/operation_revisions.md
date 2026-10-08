(operation_revisions)=
# Operation revisions

This page is for contributors who change what an analysis operation computes. It explains how
Scarf stops reusing stored results that a fix makes wrong or outdated, and when a change needs
that.

## Why revisions exist

An artifact's identity is its scope, assay, kind, and canonical provenance: the producing
operation, its parameters, and its inputs. Planning reuses a complete artifact only when its
canonical provenance matches the request exactly. Artifact IDs are random, so an input recorded
in provenance is a specific stored result: recomputing an upstream artifact gives every
downstream artifact a new identity.

Exact reuse also means that a fix that changes the output of an operation, for unchanged
parameters and inputs, would keep returning results of the old code. The operation revision
registry in `scarf/storage/operation_revisions.py` is the one explicit way to stop that. The
Scarf version recorded in each artifact's `scarf_version` attribute is diagnostic only and never
affects reuse.

## How a revision changes reuse

An operation with revisions has an ordered tuple of `OperationRevision` entries; an operation
that is not listed is at revision 1. Revision 1 is everything the operation computed before its
first entry; entries are numbered 2, 3, and so on. An entry has:

- `revision`: its number.
- `release`: the first Scarf release that records it.
- `change`: one line that says what changed, shown in logs and lineage reports. A scoped
  revision also says which results it affects.
- `applies`: `None` when the change affects every artifact of the operation, or a predicate
  `(kind, parameters, inputs) -> bool` that selects the affected artifacts.

When Scarf plans an artifact, `effective_revision` returns the highest revision whose `applies`
holds for the artifact's kind and serialized parameters and inputs, or 1. Provenance records it
as `"revision": n` only when it is 2 or more. An operation without revisions therefore records
exactly the provenance it recorded before the registry existed, and adding a revision changes
only the identities of the artifacts it applies to.

Reuse stays exact. An artifact that records no revision is revision 1, so it is reused only while
the effective revision for its parameters is 1. During the same scan, a complete artifact whose
provenance differs from the request only in its revision is a superseded match:

- it is never reused, and when no artifact matches exactly, planning logs one INFO line for the
  newest one, such as
  `Recomputing build_connectivity_map: artifact 1a2b3c4d5e6f is revision 1, current 2: <change>`;
- it stays listable, loadable, and traceable. `ArtifactStatus.revision`, `current_revision`,
  `is_current`, and `superseded_by` describe it, and `DataStore.lineage` marks it `stale`. Nothing
  persisted changes.

A revision also reaches the results built from a superseded artifact, through their inputs rather
than their own revisions. A UMAP or a Leiden clustering built on a revision 1 connectivity map
records the current revision of its own operation, so `ArtifactStatus.is_current`, which judges
only an artifact's own revision, is True for it. Its input is never reused, though: it is computed
again with a new reference, which gives the result new inputs, so the result is computed again
too.

An artifact that records a revision newer than the running release knows is not reused either,
and its status is not current. A user may still pass a superseded artifact explicitly as an input;
Scarf accepts it without a warning, and lineage shows it as stale.

## When a change needs a revision

Decide with this ladder, in order:

1. The change only moves results by last-bit or reassociation noise, and old results stay equally
   valid: add no revision. Record the change under {ref}`stable_identity_result_changes`.
2. Results become materially different, or were wrong, for an identifiable subset of artifacts:
   add a scoped revision whose `applies` selects that subset from recorded provenance.
3. Otherwise: add a whole-operation revision with `applies=None`.
4. A new reuse requirement that stops existing artifacts from being reused, such as a new required
   array or attribute or a stricter reuse validator, also adds a revision, so that the log explains
   the recomputation.

Do not change recorded parameters or inputs to invalidate results. That changes the identity of
every artifact the operation writes without telling anyone why its stored results stopped
matching. New changes use the registry instead of new version parameters; see
{ref}`legacy_version_parameters` for the ones that predate it.

(legacy_version_parameters)=
## Legacy version parameters

Some operations record a version parameter that predates the registry: the `algorithm_version` of
`run_harmony`, embedding initialization, `calc_membership_strength`, `smart_label`, label
transfer, WAGGR, and AUCell, and the `parallel_threads` of `build_ann_index`, always `None`. These
are frozen recorded constants: every artifact of the operation records the same value, and it is
never bumped. A change to what these operations compute adds a revision, like any other change.

## Rules for entries

- The registry is append-only. A released entry is never edited, renumbered, or removed, and its
  predicate never changes, because every later release judges stored artifacts against it.
- A predicate must be pure and total over recorded provenance. It receives serialized values:
  lists rather than tuples, artifact inputs as reference mappings, and only the keys that the
  release that wrote the artifact recorded. It must return a bool for any such record, including
  records written before the parameters it reads existed, and must not raise.

## Add a revision

1. Append the entry to the operation's tuple in `scarf/storage/operation_revisions.py`, adding the
   operation if it has no revisions yet, with the next number and the release that will ship it.
2. Add a test that a new artifact of the operation records the revision, as
   `ArtifactStatus.revision`, and that an artifact of the earlier revision is recomputed, not
   reused. For a scoped revision, also test an artifact that its predicate leaves out.

(stable_identity_result_changes)=
## Results that change while identities stay the same

Changes that move results without a revision, under the first rule of the ladder, are listed here
with the release, the operations, and why old results stay valid. Earlier artifacts keep being
reused; to recompute one, run its operation with `invalidate_cache=True`. Release candidates
before the registry existed changed some results without any invalidation; their entries also
name the earlier results that need such a recompute.

- 1.0.0rc16, Dunn post hoc tests of `run_statistical_testing`: the tie correction cubes the size
  of each group of tied values in float64, where an int64 cube wrapped for a group of more than
  about 2.1 million equal values. Tests whose tie groups are smaller move by float64 rounding at
  most, so their results stay valid; a test over a larger tie group, such as the zeros of millions
  of cells, holds a wrapped correction and needs a recompute.
- 1.0.0rc16, Welch's t-test of `run_statistical_testing` (`test="welch"` or `"t_test"`): when
  neither group varies, `df` is `n1 + n2 - 2`, where it was SciPy's placeholder of 1. The
  statistic and p-value of such a test do not depend on `df`, and every other test is unchanged,
  so earlier results stay valid apart from that `df` value.
- 1.0.0rc16, `select_hvgs` with `bin_strategy="fixed"`: `fit_lowess` fits only the genes whose
  mean and variance are finite and positive, as the adaptive strategy already did, and gives every
  other gene a corrected variance of zero. Fixed fits whose genes all have such values are
  unchanged bit for bit, so their selections stay valid. A fixed fit that met a gene with a zero or
  negative variance took its logarithm, which made the corrected variances of other genes NaN, so
  that none of them was selected; such a selection needs a recompute.
- 1.0.0rc16, `run_leiden_clustering`, `run_umap`, and `run_tsne` with NumPy boolean graph flags:
  `symmetric_graph=np.True_` symmetrizes the graph, and `graph_upper_only=np.True_` keeps its
  upper triangle, as `True` does. Earlier releases recorded such a flag as `True` but ignored it,
  so the result shares the identity of a correct one. Results requested with Python booleans are
  unchanged. Every UMAP and t-SNE embedding, and every clustering of a connectivity map or an SNN
  graph, has a new identity in 1.0.0 anyway, so only a Leiden clustering of a WNN graph requested
  with a NumPy boolean flag before 1.0.0rc16 can still be reused; it needs a recompute.
- 1.0.0rc16, `run_feature_percentage`: a cell without counts gets NaN, the value that the
  percentages written at preparation already held, instead of 0. Every cell with counts keeps its
  value bit for bit, so earlier results stay valid for every cell that has a percentage.
- 1.0.0rc18, normalized values of float32 counts: `run_normalization`, `run_feature_percentage`,
  and the results derived from them. Library-size, CLR, and TF-IDF values are computed in float64
  and rounded once. Subset-renormalized library-size values move by at most one float32 ulp (a few
  ulps for non-integral counts or subset totals above 2**24), CLR values by the error of their
  float32 log sums (typically 3e-5 relative at 8,000 cells and 8e-4 at 200,000), and whole-library
  library-size values, feature percentages, and subset-renormalized TF-IDF values only for
  non-integral counts or totals and products that float32 cannot hold exactly. Pseudotime markers
  and aggregations that stream library-size values without subset renormalization also compute
  them in float64. The earlier values lost only float32 precision, so they stay valid; statistical
  tests compare the fingerprints of their values and recompute by themselves.
- 1.0.0rc18, `run_pseudotime_marker_search`: `find_markers_by_regression` scales the regressor and
  each feature's values by a power of two before it sums them. Ordinary inputs give bit-identical
  results. Values whose squared deviations overflowed float64 (magnitudes above about 1e150) now
  give their correlation instead of r = 0 with p = 1, or NaN, a regressor whose squared deviations
  underflowed (below about 1e-160) is tested instead of reported as untested, and two-cell searches
  report r of exactly 1 or -1. Earlier searches stay valid unless their values reached such
  magnitudes.
- 1.0.0, streamed variances: `summarize_rna_features`, `select_hvgs`, `run_normalization`,
  `calculate_feature_scaling`, and `run_pca`. Feature summary variances (`sigmas`), normalized
  feature statistics, and PCA feature scales merge each block's count, sum, and sum of squared
  deviations from its mean (`scarf.utils.moments`) instead of subtracting the squared mean from
  the mean of the squares, which lost the significant digits of features whose mean is large
  beside their spread and could leave a constant feature a tiny or negative variance. Normalized
  artifacts store `feature_m2` instead of `feature_squared_sum`
  ({ref}`normalized_feature_statistics`); earlier ones are still reused, and their feature scaling
  is streamed from their data. On the 1K PBMC and 500 PBMC ATAC fixtures, sums, means, HVG
  selections, and stored PCA coordinates are unchanged to the bit; variances, HVG corrected
  variances, and feature scales move by at most 3e-13 relative, and PCA loadings by at most 3e-14
  of the largest loading. A constant feature now has a variance of exactly zero and a PCA scale of
  1, where its scale could be about 1e-6; its scaled values are zero either way. The Gram solver
  of `run_pca` with `feat_scaling=False` accumulates deviations from the first block's means, so
  its covariance no longer cancels for features with large means. `ChunkedArray.mean` and `var`
  accumulate in float64 on every axis, so integer means cannot overflow and float32 means keep
  float64 precision; `score_features` of an assay without a normalization, over float32 counts
  and with `log_transform=False`, therefore averages in float64, which can move a control
  feature across an expression bin boundary.
- 1.0.0, `summarize_rna_features` with `log_transform`, and the `run_cell_cycle_scoring` results
  that bin features by it: the values of `norm_dummy` or of a custom RNA normalizer whose output
  is not float64 are logged in float64, as `normed(log_transform=True)` logs them, instead of in
  their dtype (float16 for uint8 values, float32 for uint16 and float32 values). Values move at
  the resolution of that dtype, so earlier summaries stay valid. Float64 outputs, which
  normalizers that divide by `assay.scalar` return, and library-size summaries are unchanged bit
  for bit.
- 1.0.0, `run_pseudotime_marker_search`: a feature counts as constant, and is not tested, when its
  values span at most float64 epsilon times their largest magnitude; the rule was an absolute span
  of epsilon. Features whose values span about 2.2e-16 or less, but more than rounding of their
  magnitude, are now tested, and features whose values differ only by rounding of a large
  magnitude are no longer tested. Every other feature keeps bit-identical results, so earlier
  searches stay valid unless they tested values at such scales.

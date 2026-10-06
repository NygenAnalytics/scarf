# Assays and metadata API reference

Each assay opens as the class of the preset that its store declares in `assayTypes`, such as
`RNA`, `ATAC`, `ADT`, `HTO`, `GeneActivity`, or `Assay` for a generic assay; see
{ref}`assay_types_attribute`. `DataStore(assay_types=...)` accepts only these presets and only
names of assays in the store, and a read-only open accepts only the types that the store records.
An open assay carries the preset that its `DataStore` resolved as `assayType`, which merge,
subset, HTO demultiplexing, and `DataStore.summary()` read. An assay that measured only some cells
of its store marks them in the Boolean cell column `<assay>_I` ({ref}`assay_membership`), which
`MetaData` refuses to write or, while it carries the membership role, to drop.

`Assay.score_features(feature_names, cell_key, ctrl_size, n_bins, rand_seed)` is computation-only.
It computes full-row-order averages blockwise in memory and is safe on read-only counts; it does
not plan artifacts or write metadata. With `log_transform=True` it scores the values of
`normed(log_transform=True)`, so only normalizers that take `log_transform` accept it (see the
table below). Persistent cell-cycle outputs belong to `DataStore.run_cell_cycle_scoring`.

Feature-count percentages belong to
`DataStore.run_feature_percentage(cell_selection, features)`. It derives the assay from the exact
feature-selection ref and returns an assay-scoped `quality_metric` ref whose `values` array is the
per-cell percentage. It does not add a metadata column. A read-only `DataStore` returns an
identical existing result but raises `PermissionError` instead of computing a new one.

Persisted normalization belongs to
`DataStore.run_normalization(cell_selection, features)`, which returns an immutable artifact ref.
It saves the values of `normed` with the configured normalizer. A flag left as None takes the
operation's default only for the RNA library-size normalizers `norm_lib_size` and
`norm_lib_size_log`, whose scale Scarf knows, as
`scarf.assay.normalization.default_normalization_flags` reports: `run_normalization` applies the
flags they take by default. The values of every other normalizer, `norm_dummy`, CLR, TF-IDF, and
custom normalizers, may be signed, already logged, or computed over chosen totals, so their flags
stay False unless passed. The artifact records the resolved flags. Assay normalization methods are
computation-only and require explicit feature indexes where they do not accept an `ArtifactRef`.

Every normalized path applies the assay's configured `normMethod`: `normed`, saved normalization,
HVG statistics, feature summaries, scores, marker search, pseudotime markers and aggregations,
and marker heatmaps. Paths that compute library-size values straight from the counts, such as the
subset writer of `run_normalization` and the RNA feature streams, run only when
`scarf.assay.normalization.uses_library_size_normalization(assay)` holds: the normalizer is
`norm_lib_size` and `sf` is set.

The two normalization flags have one meaning on every path. `log_transform=True` takes `log1p` of
the configured normalizer's own output, in float64. `renormalize_subset=True` hands the normalizer
each cell's total over the selected features instead of over every feature. A normalizer accepts
only the flags it can apply, as `scarf.assay.normalization.applicable_normalization_flags` reports;
a flag it cannot apply defaults to False, and passing True raises `ValueError`:

| Assay and normalizer | `log_transform` | `renormalize_subset` |
|---|---|---|
| RNA `norm_lib_size` (default) and custom normalizers | yes | yes, through `assay.scalar` |
| RNA `norm_lib_size_log` | no, its values are already logarithms | yes |
| `norm_dummy` (the generic default) of RNA, ADT, and generic assays, and custom ADT and generic normalizers | yes | no |
| ATAC `norm_tf_idf` (default) and custom normalizers | no, ATAC values are never logged | yes, as the term-frequency denominator |
| `norm_clr` (the ADT default) on any assay, built-in normalizers of another assay class (`norm_tf_idf` outside ATAC, library-size normalizers outside RNA), and `norm_dummy` of ATAC | no | no |

A custom RNA normalizer is a callable `(assay, counts)` that may read each cell's total from
`assay.scalar` while it builds its result; `normed(log_transform=True)` returns the `log1p` of what
it returns. Saved results need a module-level function or a callable with an `artifact_identity`
attribute, which provenance records. Pass `log_transform=True` to take logarithms of the counts
that `norm_dummy` keeps as they are.

RNA library-size normalization divides each cell by its total, the `<assay>_nCounts` value or, with
`renormalize_subset=True`, its sum over the selected features, and a cell whose total is zero
normalizes to zeros. `scarf.assay.normalization.library_size_divisors` holds this rule for every
library-size path: `normed` with `norm_lib_size` or `norm_lib_size_log`, normalization artifacts,
HVG statistics, feature-group scores, marker search, `make_bulk` means, WAGGR, and mapping and
doublet projections. Each raises `ValueError` naming the totals when a total of a selected cell is
negative or not finite. `normed` with a custom `normMethod`, which need not read the totals, does
not check them and replaces a zero total with 1.

```{eval-rst}
.. autoclass:: scarf.assay.Assay
    :members:
```

```{eval-rst}
.. autoclass:: scarf.assay.RNAassay
    :members:
```

```{eval-rst}
.. autoclass:: scarf.assay.ATACassay
    :members:
```

```{eval-rst}
.. autoclass:: scarf.assay.ADTassay
    :members:
```

```{eval-rst}
.. autofunction:: scarf.assay.norm_dummy
```

```{eval-rst}
.. autoclass:: scarf.metadata.MetaData
    :members:
```

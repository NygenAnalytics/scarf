# Data access

Open, inspect, import, and connect to Scarf DataStores, including Cytebase remote and mounted
stores. Docs: <https://scarf.readthedocs.io/en/latest/tutorials/data_organization.html>,
<https://scarf.readthedocs.io/en/latest/tutorials/import_and_export.html>, <https://scarf.readthedocs.io/en/latest/tutorials/cytebase.html>,
<https://scarf.readthedocs.io/en/latest/tutorials/remote_stores.html>, <https://scarf.readthedocs.io/en/latest/reference/api/datastore.html>, `import_export.html`, `cytebase.html` (same folder).

## When to use

- First contact with any dataset: choose read-only or writable, then inspect before computing.
- Converting 10x, Matrix Market, H5AD, or Seurat RDS input into a Scarf Zarr store.
- Finding a Cytebase dataset, exploring it remotely, or mounting it for analysis that saves results.
- Reopening a store or a mount from an earlier session.

## Key concepts

- A store is one Zarr directory: `cellData` (shared cell table, `ds.cells`), one group per assay
  (`ds.RNA`; feature table `ds.RNA.feats`; counts; assay artifacts), datastore-scoped artifacts,
  and `pipeline/runs`. Cell QC columns are assay-prefixed: `RNA_nCounts`, `RNA_nFeatures`,
  `RNA_percentMito`, `RNA_percentRibo`. Tables are `MetaData` objects, not DataFrames.
- `I` is the live Boolean cell key (feature tables have one too). `fetch(col)` returns rows where
  `I` is true; `fetch_all(col)` returns every row. Analytical filtering returns a cell-selection
  artifact and never edits `I`.
- `zarr_mode="r+"` is the default. A writable open WRITES: first-open preparation of a freshly
  written store (QC columns, feature `nCells`/`dropOuts`), the `min_features_per_cell` filter
  applied to `I`, and the `defaultAssay`/`assayTypes` attributes. `zarr_mode="r"` writes nothing;
  most producers return a matching existing artifact but raise `PermissionError` instead of
  computing. Feature selections, WAGGR, AUCell, `select_prevalent_peaks`, cell-cycle scoring, and
  `pipeline.run` raise it before any lookup, and `run_mapping` and `build_mapping_reference` raise
  `ValueError`.
- RNA assays hold cell-major `counts` plus gene-major `countsT`; other assays hold `counts` only.
  Stores from older releases (Zarr v2, no `countsT`, an `{assay}/state` group) fail to open:
  re-import the source.
- A mount is a local writable target whose counts resolve from a separate source recorded in the
  root `matrixSource` attribute (absolute path or URI). Metadata is copied once at mount time; new
  artifacts are written only to the target.
- Cytebase has three layers: `Catalog` (verified local DuckDB copy of the catalog, no counts),
  `open_datastore` (read-only, everything remote), `mount_datastore` (local writable target,
  counts remote, plus a `<target>.cytebase.json` receipt beside it).
- `workspace` names a group inside one Zarr with its own `cellData`, assays, and artifacts (counts
  under `matrices/`). Writers default to `None` (root layout); keep `None` unless the store was
  written with a workspace. A mount records one source workspace and rejects another.

## Recipes

### Open and inspect a store

Design and annotation inference starts with `scripts/inspect_store.py STORE.zarr` (relative to the
skill root). Read-only, it flags annotation-like columns (values hidden unless
`--show-annotation-values`), design-like, author-derived and unflagged columns, and checks
whether counts look raw, corrected or pre-filtered. Flags are name-based hints: review every
column. By hand, open read-only until you need to write; `summary()` is metadata-only.

```python
import scarf

scarf.configure_output(level="WARNING", progress=False)
ds = scarf.DataStore("analysis.zarr", zarr_mode="r")
print(ds)                                    # "N_active (N_total) cells", assays, column names
snap = ds.summary().to_dict()                # JSON-safe
print(snap["default_assay"], snap["total_cells"], snap["active_cells"], snap["resources"])
print(snap["pipeline_run_counts"], snap["labeled_pipeline_runs"])
print([(a["name"], a["assay_type"], a["total_features"], len(a["artifacts"])) for a in snap["assays"]])
print(ds.pipeline.list_runs())                                   # newest first
print(ds.list_artifacts(complete_only=True))                     # default-assay artifacts
print(ds.list_artifacts(scope="datastore", complete_only=True))  # cell selections, snapshots
ds.show_zarr_tree(start="RNA", depth=1)

# Metadata: request only the columns you need
meta = ds.cells.to_pandas_dataframe(["ids", "RNA_nCounts", "RNA_nFeatures"], key="I").set_index("ids")
active_ids = ds.cells.fetch("ids")        # rows where I is True
all_ids = ds.cells.fetch_all("ids")       # every row
genes = ds.RNA.feats.to_pandas_dataframe(["ids", "names", "nCells"])
by_upper = {str(n).upper(): str(n) for n in ds.RNA.feats.fetch_all("names")}  # case-safe lookup

# Per-cell values of a cell-aligned artifact (clusters, Paris cut, cell cycle, pseudotime, UMAP)
run = ds.pipeline.open(label="baseline")  # a completed run; any cell-aligned ref works the same
cut = ds.load_cell_values(run["paris"])   # reads the kind's canonical array: "labels" here
print(cut.value, cut.categorical, cut.values.shape)
labels = cut.to_pandas()                  # indexed by cell id; rows recorded missing show as NA
s_score = ds.load_cell_values(run["cell_cycle"], value="s_score")  # another per-cell array
xy = ds.load_cell_values(run["umap"]).to_pandas()  # DataFrame, one column per dimension
```

`load_cell_values` aligns rows to cell `ids` and flags rows the artifact records as missing, so
prefer it to `load_artifact(ref)["values"]`. `cell_selection=` reads a subset of the artifact's own
cells. Cell selections, reference labels, graphs, reductions, and Harmony corrections are refused;
open those with `load_artifact`. A run's `cell_snapshot` and `feature_snapshot` are outputs of its
`input_snapshot` stage, listed under `outputs` in `run.report()["stages"]`, not run outputs, so
`run["cell_snapshot"]` raises `KeyError`. They hold whole metadata columns, not cell-aligned
values, and are refused too: read their columns with `run.cells.fetch(column)` or
`run.features.fetch(column)`. The memory check runs before anything is
read and charges the values, their mask, the cell ids, and the row indexes the read builds; on
`MemoryError`, pass a smaller `cell_selection=` or raise `mem_budget`.

### Convert inputs to Zarr

Every writer follows reader then `*ToZarr(...).dump()`. Open each new store writable once.
Write each import to a new or empty path: writers raise `FileExistsError` for a path that holds
data. `overwrite=True` replaces only an earlier import that no `DataStore` has opened; after the
first open the store is prepared and never replaced, so delete it yourself before you import again.
Writers also refuse, with `ValueError`, a path inside another store (`s.zarr/RNA`,
`s.zarr/new.zarr`): put every store in its own directory outside other stores.

```python
# 10x HDF5: assays inferred from feature types (RNA, ADT, ATAC)
reader = scarf.CrH5Reader("filtered_feature_bc_matrix.h5")
print(reader.assayFeats)                  # assays and feature ranges that will be written
scarf.CrToZarr(reader, zarr_loc="pbmc.zarr", mem_budget="4G").dump()
# A default count layout that does not fit mem_budget raises MemoryError before writing and
# names a smaller policy=CountMatrixPolicy(...) (scarf.storage.count_matrix); raise
# mem_budget or pass that policy.
ds = scarf.DataStore("pbmc.zarr", default_assay="RNA")   # required when there are 2+ assays

# 10x directory (matrix.mtx, genes/features.tsv, barcodes.tsv; .gz accepted)
scarf.CrToZarr(scarf.CrDirReader("filtered_feature_bc_matrix"), zarr_loc="dir.zarr").dump()

# Matrix Market: inspect, then pick one candidate explicitly
candidates = scarf.inspect_mtx("mtx_dir")
reader = scarf.MtxReader(candidates[0])
print(reader.assayFeats)
scarf.MtxToZarr(reader, zarr_loc="mtx.zarr").dump()

# H5AD: inspect first; selected obsm/obs values become artifacts, not live columns
import h5py

insp = scarf.inspect_h5ad("data.h5ad")
print(insp.matrixKey, insp.matrixCandidates, insp.integerLike, insp.layers, insp.suggestedAssays)
with h5py.File("data.h5ad", "r") as h5:  # pass only keys the file has; a missing one is a KeyError
    obsm = set(h5["obsm"]) if "obsm" in h5 else set()
    obs = h5["obs"]
    obs_cols = set(obs) if isinstance(obs, h5py.Group) else set(obs.dtype.names)  # old files
print(sorted(obsm), sorted(obs_cols))
roles = {k: r for k, r in {"X_umap": "umap", "X_tsne": "tsne"}.items() if k in obsm}
kw = dict(embedding_roles=roles, cluster_keys=tuple(c for c in ("clusters",) if c in obs_cols))
reader = scarf.H5adReader.from_inspect(insp, **kw)
res = scarf.H5adToZarr(reader, zarr_loc="h5ad.zarr").dump()
print(dict(res.embeddingArtifacts), dict(res.clusterArtifacts))

# Seurat RDS (an on-disk .rds; .h5seurat is not read)
si = scarf.inspect_seurat("pbmc.rds")
reductions = [r.name for r in si.reductions if r.importable]
with scarf.SeuratReader("pbmc.rds", assays=[si.activeAssay], reductions=reductions) as sr:
    out = scarf.SeuratToZarr(sr, zarr_loc="seurat.zarr").dump()
print(out.defaultAssay, dict(out.reductionArtifacts), out.activeIdentity, len(out.notices))

for path in ("dir.zarr", "mtx.zarr", "h5ad.zarr", "seurat.zarr"):
    scarf.DataStore(path)                 # writable first open prepares the store
```

`CSVReader`/`CSVtoZarr` (small dense CSV) and `SparseToZarr` (SciPy CSR plus IDs) also exist.

Every converter refuses this way from 1.0.0rc18 on (1.0.0rc17 shrinks the layout to fit). The
error is a `MemoryError`; releases after 1.0.0rc19 raise its subclass `CountLayoutMemoryError`, so
catch `MemoryError`. Prefer a larger `mem_budget` when the host has
the memory. Otherwise copy the policy numbers from the message exactly and record that choice:
smaller layouts make every later gene-major read slower.

```python
from scarf.storage.count_matrix import CountMatrixPolicy

policy = CountMatrixPolicy(unitBytes=3906250, chunkBytes=390625)   # copied from the message
scarf.CrToZarr(reader, zarr_loc="pbmc.zarr", mem_budget="128M", policy=policy).dump()
```

The need follows genes per cell relative to the gene count, not the number of cells. The 1K PBMC
CITE-seq file refuses at `512M` and imports at `1G`.

### Find a Cytebase dataset

```python
from scarf import cytebase

catalog = cytebase.Catalog()              # public Nygen/cytebase, no credentials
hits = catalog.search("bladder immune", limit=5)      # every word, case-insensitive
ids = [row["cytebase_id"] for row in hits]            # take IDs from rows, not printed tables
blood = catalog.find_datasets(tissue="blood", organism="Homo sapiens")  # exact labels
assay_labels = [row["label"] for row in catalog.list_terms("assay")]    # exact facet labels
small = catalog.query(
    "SELECT cytebase_id, cell_count FROM datasets "
    "WHERE status = ? AND cell_count < ? ORDER BY cell_count LIMIT 3",
    parameters=["ready", 5000],
)
entry = catalog.dataset(ids[0])
print(entry.describe())                   # Markdown summary; does not open the store
print(entry.cell_count, entry.source_embeddings())    # source obsm keys, imported or not
```

### Explore a Cytebase dataset read-only

```python
ds = catalog.open_datastore(entry.id)     # zarr_mode="r", min_features_per_cell=-1
print(cytebase.embeddings(ds))            # {"X_umap": ArtifactRef}: keys actually imported
umap_ref = cytebase.embedding(ds, "X_umap")
coords = cytebase.embedding_coordinates(ds, umap_ref)  # DataFrame indexed by cell id
meta = ds.cells.to_pandas_dataframe(["ids", "donor_id"], key="I").set_index("ids")  # a design column
frame = coords.join(meta)                 # join on ids, never on row order
```

### Mount a Cytebase dataset for writable analysis

The target must be a new local path. Mounting copies metadata and verifies the source; expect a
few minutes for a few thousand cells.

```python
analysis = catalog.mount_datastore(entry.id, at="work/analysis.zarr")
# Writes work/analysis.zarr (copied metadata, future artifacts) and work/analysis.zarr.cytebase.json
print(analysis.zarr_mode, int(analysis.cells.fetch_all("I").sum()), analysis.cells.N)
print(cytebase.embeddings(analysis))      # the source's imported embeddings, resolved read only
```

### Reopen a mount, or mount any store that owns its counts

```python
# Verify against the current published build (needs the .cytebase.json sidecar beside it)
analysis = catalog.mount_datastore(entry.id, at="work/analysis.zarr")
# Or open it like any store: counts resolve from matrixSource; no catalog or sidecar needed
analysis = scarf.DataStore("work/analysis.zarr", min_features_per_cell=-1)
# Generic mount; for s3:// or gs:// sources also pass storage_options (not executed here)
mounted = scarf.mount_datastore("shared/data.zarr", at="my_analysis.zarr", default_assay="RNA")
```

### Download a documentation dataset

```python
repo = scarf.cytebase.connect("scarf_docs")
print(repo.list_datasets())
path = repo.download_dataset("tenx_5K_pbmc_rnaseq", destination="scarf_datasets", zarr=True)
ds = scarf.DataStore(f"{path}/data.zarr", zarr_mode="r")  # prepared; run label "docs_default"
raw = repo.download_dataset("xin_1K_pancreas_rnaseq", destination="scarf_datasets")  # source files
```

## Parameters that matter

| Parameter | Default | Change when |
|---|---|---|
| `zarr_mode` | `"r+"` | `"r"` to inspect, share, or protect a store; a fresh store needs one `"r+"` open |
| `default_assay` | stored value, or the only assay | first open of a multi-assay store (otherwise `ValueError`) |
| `min_features_per_cell` | `10` | `-1` to leave `I` untouched on writable opens |
| `mito_pattern` / `ribo_pattern` | `None` (`^MT-`, `^RPS\|^RPL\|^MRPS\|^MRPL`) | gene names that these prefixes miss, only on the FIRST writable open; patterns ignore case, so `^MT-` already matches mouse `mt-` |
| `assay_types` | inferred from assay names | custom names, e.g. `{"GEX": "RNA"}`; values must be presets (`RNA`, `ADT`, `HTO`, `Assay`, ...); a writable open records them, a read-only open must match the store |
| `nthreads`, `mem_budget` | env `SCARF_WORKERS`, `SCARF_MEM_BUDGET`, else detected | see `performance-and-export.md` |
| `workspace` | `None` | store written into a named workspace |
| `storage_options` | `None` | object-store credentials or endpoints (read from env vars) |
| `zarrProfile` | from location (`fast_local`/`cloud`) | affects newly written arrays only |
| `Catalog(bucket=, token=)` | `Nygen/cytebase`, `None` (HF login if present) | private bucket; `token=False` forces anonymous |
| `search(limit=)`, `ready_only=` | `50`, `True` | `limit=None` for all; `ready_only=False` for unfinished |

## Check before moving on

- `print(ds)`: active versus total cells, expected assays, QC columns present.
- `snap["default_assay"]` is the assay you intend; `snap["labeled_pipeline_runs"]` and
  `list_artifacts` show existing work to reuse (see `pipeline-runs-and-artifacts.md`).
- Imports: `reader.assayFeats` lists the assays to be written; afterwards check
  `ds.RNA.rawData.shape` and dtype. For H5AD, `insp.integerLike` should be `True` for raw counts,
  but SCT-corrected counts are integers too: run "Check the matrix first" in `quality-control.md`.
- Cytebase: `entry.row["status"] == "ready"`; check `cytebase.embeddings(ds)` before plotting an
  imported layout.
- Mounts: `<target>.cytebase.json` exists beside the target; `ds.RNA.rawData.shape` resolves.

## Pitfalls

- A writable open with `min_features_per_cell=k` silently removes cells with at most `k`
  default-assay features from `I` and persists it. When at least half of the active cells would
  go, as in a small ADT panel, it keeps `I` and logs a warning. `k` must be an integer of at least
  `-1`. Reopening with a lower value does not restore cells; `ds.cells.reset_key("I")` does.
- Stores just written by a writer or `SubsetZarr` raise `Assay 'RNA' is not prepared yet. Open the
  store once with zarr_mode='r+' ...` when opened with `zarr_mode="r"`. Open once with `"r+"`; no
  rebuild is needed.
- Percent patterns are fixed at first preparation; a different pattern later raises `ValueError`.
  Use `run_feature_percentage` (see `quality-control.md`) for another gene set.
- `open_datastore` rejects writes and any `min_features_per_cell` other than `-1`. Both mount
  functions reject `zarr_mode="r"`; `scarf.mount_datastore` refuses an existing target, while
  `catalog.mount_datastore` reopens one only when its matching sidecar is present.
- `SubsetZarr`, `DataStoreMerge`, `mount_datastore`, and repack refuse a source store that holds a
  pending derived assay (`... holds a pending derived assay and cannot be ...`). Once no process is
  writing it, run `ds.discard_interrupted_assay("<name>")` on a writable `DataStore` of its
  workspace, then retry.
- Catalog reopen needs the sidecar: copying a mount without `<name>.cytebase.json` makes
  `catalog.mount_datastore` raise `FileExistsError`; plain `scarf.DataStore` still opens it.
- The mount source must stay at its recorded path or URI. Mounting a mount raises; repack first.
  A changed Cytebase build requires a new mount directory.
- Printed `CatalogResults` tables HTML-escape text (`_` as `&#95;`, `'` as `&#39;`). Read IDs and
  facet labels from the row dicts (`row["cytebase_id"]`, `row["label"]`), never from the table.
- A mount resolves its source's imported embeddings read only, so `cytebase.embedding(analysis)`
  and plots work on it; new artifacts are written to the mount.
- `to_pandas_dataframe(columns)` defaults to `key=None` (all rows); pass `key="I"` for active cells.
- `to_mtx(..., compress=True)` writes the feature `feature_type` column as the 10x feature type.
  CELLxGENE stores keep gene biotypes there, so re-import split one assay into thousands. Inspect
  `reader.assayFeats` before `dump()` or export with `compress=False`.
- Source column names containing `/` or `\` are stored with `_`. Reserved columns `ids`, `names`,
  `I` from a source are skipped. H5AD multi-assay import needs `assay_split_key` on `H5adToZarr`.
- `<assay>_I` marks the cells an assay measured (after a merge of stores with different assays).
  Imports reserve it for every assay they write: an H5AD file that Scarf exported declares the
  exported assay's column in `uns["scarf"]["assayMembership"]` (other assays' membership columns
  are not exported) and imports it back as membership; any other `<assay>_I`
  column of an imported assay is skipped with a warning (Seurat import raises). Merge refuses a
  plain `<assay>_I` column (`Cell column 'RNA_I' is reserved ...`), as an earlier release's
  import of an exported file left it: import the file again with this release, or drop the plain
  column with `ds.cells.drop("RNA_I")` (the assay then counts every cell as measured).
- Unmeasured cells hold zero counts, which are no measurement. Operations that read an assay's
  values (normalization, HVGs, detected features, WAGGR, AUCell, markers, `make_bulk`, statistical
  tests of genes, prevalent peaks, cell cycle, feature percentages, HTO, doublets, pseudotime
  features, `get_imputed` of genes, `run_mapping`, `to_anndata(matrix="normed")`, `pipeline.run`,
  and QC filters on an assay's metrics) raise `UnmeasuredCellsError` (a `ValueError` with
  `.operation`, `.assay`, `.column`, `.unmeasured`, `.selected`) after their argument checks and
  before writing anything, also on read-only stores. Narrow first:
  `cells = ds.select_measured_cells("ADT", cell_selection=ds.snapshot_cell_selection("I"))`
  (the input comes back unchanged when the assay measured every cell, so identities stay);
  `make_bulk`, `run_statistical_testing`, `auto_filter_cells`, and `filter_cells` take it as
  `cell_selection=cells`; other labels: `ds.snapshot_cluster_labels(labels,
  cell_selection=cells)`; graphs: build them over `cells`. The pipeline and normalized export take
  a `cell_key` column of the cells of `I` that the pipeline's RNA assay measured, not
  `cell_key="RNA_I"`, which also holds cells a filter removed from `I`:
  `ds.cells.insert("RNA_measured", ds.cells.fetch_all("I") & ds.cells.fetch_all("RNA_I"))`, then
  `ds.pipeline.run(cell_key="RNA_measured")`. A QC metric is a column that an assay's preparation
  wrote (`<assay>_nCounts`, `<assay>_nFeatures`, its recorded `<assay>_percent*` columns) or a
  `quality_metric` artifact of that assay. Plots and `get_cell_vals` show unmeasured cells' gene
  values as NaN (missing), not 0, and normalize only measured cells; dot and matrix plots compute
  `fraction`, `mean`, and `n_cells` over measured cells, and plots record
  `provenance.extras["unmeasured_cells"]`. `to_mtx` and `to_anndata` layers of another assay,
  which cannot declare membership, refuse unmeasured cells: subset to measured cells
  (`SubsetZarr(..., cell_key="RNA_I")`) or use `to_h5ad`. Results an earlier release computed over
  unmeasured cells are not flagged; rebuild them over measured cells.
- Never write `<assay>_I` yourself: `ds.cells.insert`, `update_key`, and `reset_key` raise for the
  membership name of any assay, even when the column is absent, and `drop` raises for a real
  membership column. To analyze only measured cells, `ds.select_measured_cells("ADT")`.
  Derived assays (`add_grouped_assay`, `add_melded_assay`) copy their source's membership, and a
  grouped assay fits the source normalization on measured cells only, with zeros elsewhere;
  `SubsetZarr(assays=[...])` and `DataStoreMerge(assays=[...])` drop the membership columns of the
  assays they leave out.
- A read-only open with `assay_types` that differs from the store's recorded type raises
  `ValueError`; open once with `zarr_mode="r+"` and that `assay_types` to record it. Merge, subset
  and `run_hto_demultiplexing` use the type the open resolved (`ds.get_assay(name).assayType`).
- `download_dataset(..., zarr=True)` keeps `data.zarr.tar.gz` beside `data.zarr` (double disk).
- Public atlas matrices (CELLxGENE, Seurat submissions) can hold SCT-corrected counts in `raw.X`,
  with genes such as rRNA removed and cells already filtered. Check before QC or count models.

## See also

- `performance-and-export.md`: budgets, remote I/O cost, repacking a mount, exports.
- `pipeline-runs-and-artifacts.md`: runs, artifact inspection, lineage.
- `quality-control.md`: filtering cells without editing `I`.
- `plotting.md`: `ds.plots.embedding(layout=umap_ref, ...)` for imported layouts.

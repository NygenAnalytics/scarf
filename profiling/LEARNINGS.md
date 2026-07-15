# Scarf cloud profiling learnings (100k / 250k)

Date range: 2026-07-14 to 2026-07-15  
Environment: Modal `scarf_profiling`, app `scarf-profiling`, region `eu-west-1`, secret `scarf-r2`  
Data: `s3://scarf-tests/scarf-profiling/` (datasets / stores / results)  
Dataset source: nested CELLxGENE samples already prepared on R2

This note is the baseline for quantifying later changes (500k+, code defaults, orchestrator CPU/mem tables). Times are stage wall seconds from result JSON. Peaks are `peakCgroupBytes`.

## Objective

Minimize wall time without pointless overprovisioning. Prefer using memory and CPU in ways that actually cut stage time. Do not treat `workingCopies` as a speed dial.

## Code changes already wired

| Change | Location | Effect |
|--------|----------|--------|
| Cloud default `targetChunkBytes` = 128 MiB when remote and unset | `scarf/storage/zarr_store.py` (`DEFAULT_CLOUD_TARGET_CHUNK_BYTES`) | Matches layout-sweep winner |
| Auto marker batch when `gene_batch_size is None` | `scarf/markers.py` `resolve_marker_gene_batch_size` | `min(col_chunk, n_features, budgetCap)` with `budgetCap = (memoryBytes // workingCopies) // (n_cells * 32)` |
| Optional profiling override | `profiling` `markerGeneBatchSize` | Layout sweep forced `50`; later runs leave unset for auto |

## Machine sizes used (Modal hard limit vs Scarf budget)

Two different numbers matter:

| Concept | Meaning |
|---------|---------|
| Modal memory | Hard cgroup limit on the container (OOM kill if exceeded) |
| Scarf `memoryBytes` / `scarfMemoryBudget` | Software budget used for chunk geometry at write time and auto marker batch size. Usually ~75% of Modal in our configs |

| Run tag | Cells | Modal CPU | Modal RAM | Scarf budget | Notes |
|---------|------:|----------:|----------:|-------------:|-------|
| layout sweep (`baseline`, `chunk*`) | 100k | 4 | 32 GiB | ~24 GiB | Forced marker batch 50 |
| `auto_markers_c4_m32` | 100k | 4 | 32 GiB | ~24 GiB | Auto markers; UMAP parallel off |
| `auto_markers_c4_m32_scarf16` | 100k | 4 | 32 GiB | **16 GiB** | Budget mismatch vs `c4_m32` (Modal same, Scarf cut) |
| `auto_markers_c4_m16` | 100k | 4 | 16 GiB | ~12 GiB | Both Modal and Scarf cut together |
| `auto_markers_c8_m32` | 100k | 8 | 32 GiB | ~24 GiB | Speed pack (UMAP/ANN parallel) |
| `auto_markers_c8_m32_250k` | 250k | 8 | 32 GiB | ~24 GiB | Same speed pack |
| `auto_markers_c8_m48_500k` | 500k | 8 | 48 GiB | 36 GiB | In progress / planned |

Local layout TOMLs under `profiling/layouts/` are gitignored. Treat `LEARNINGS.md` as the durable record; recreate TOMLs from these rows when needed.

## Constants that must stay stable when comparing

| Knob | Value used in successful speed runs | Rule |
|------|-------------------------------------|------|
| `workingCopies` | 4 in profiling configs (library default is 8) | Model of concurrent in-memory copies. Change only if peaks/OOM show the model is wrong |
| Assay / workflow seeds | graph 4466, umap/leiden 4444, `topN=2000`, `k=11`, `dims=50` | Keep fixed across A/B |
| Modal vs Scarf coupling | Usually Scarf ≈ 75% of Modal | Decouple on purpose only for budget-mismatch experiments (below) |

## How to quantify a future change

1. Pick a fixed reference run tag (see tables below). Prefer `auto_markers_c8_m32` for 100k and `auto_markers_c8_m32_250k` for 250k.
2. Change one variable (size, CPU, mem, parallel flags, or code). Keep `workingCopies` and workflow seeds fixed unless the experiment is about the copy model.
3. Compare stage seconds and peak GiB. Also report total seconds and max peak.
4. Result URIs: `{resultsUri}/results/{runTag}/{nRows}/{stage}.json`
5. Spawn via deployed app bare `run_all_jobs.spawn(...)` (avoids stuck `with_options` queues from local `modal run`). Stage workers still apply config resources inside the deployed orchestrator.

Useful call IDs from this campaign (`/tmp/scarf_calib_calls.txt`):

```
auto_markers_c4_m32=fc-01KXJX3H7988EH0TZM6EGP7AQC
auto_markers_c4_m16=fc-01KXJZ3VFQKMZ0T77ET87AD7KD
auto_markers_c8_m32=fc-01KXK3JTNTC44X44K3NFQRTDQ5
auto_markers_c8_m32_250k=fc-01KXK8BRWK9P50XW0NY4FWE378
auto_markers_c8_m48_500k=fc-01KXKE1ZCG0FG7KFEQ1H0QX35K
auto_markers_c4_m32_scarf16=fc-01KXKEX6N0N44661M6ZX26NX9C
```

## Layout sweep @ 100k (4 CPU / 32 GiB)

All layouts: forced `markerGeneBatchSize=50`, `annParallel=false`, `umapParallel=false`, workers=4, workingCopies=4.

| Tag | Nominal chunk target | Total s | Max peak GiB | Markers s | Markers peak |
|-----|---------------------|--------:|-------------:|----------:|-------------:|
| baseline | memory-first / default | 2467 | 7.24 | 1742 | 3.51 |
| chunk64m | 64 MiB | 1189 | 6.85 | 366 | 1.13 |
| **chunk128m** | **128 MiB** | **1147** | **6.50** | **353** | **1.01** |
| chunk256m | 256 MiB | 1301 | 6.43 | 536 | 1.20 |
| chunk512m | 512 MiB | 1188 | 6.23 | 521 | 1.37 |

**Learning:** ~128 MiB cloud count chunks win on total time. Larger nominal targets did not help markers and often hurt them. Baseline markers were catastrophic (1742s) under the old layout.

Configs: `profiling/layouts/100k_{baseline,chunk64m,chunk128m,chunk256m,chunk512m}.toml`

### Stage detail: layout winner (`chunk128m`)

| Stage | Seconds | Peak GiB |
|-------|--------:|---------:|
| createStore | 91.3 | 3.44 |
| initializeStore | 122.9 | 6.50 |
| reopenStore | 5.8 | 0.48 |
| filterCells | 15.9 | 0.48 |
| markHvgs | 221.6 | 1.06 |
| makeGraph | 159.7 | 6.43 |
| runUmap | 154.8 | 0.73 |
| runLeiden | 22.4 | 0.71 |
| findMarkers | 352.7 | 1.01 |
| **total** | **1147.1** | **6.50** |

## Auto markers and machine experiments @ 100k

Cloud 128 MiB default (no forced batch 50). Scarf budget ~24 GiB on 32 GiB Modal boxes (~12 GiB on 16 GiB).

| Tag | CPU | Modal GiB | Parallel UMAP/ANN | Total s | Max peak | Markers s | Markers peak |
|-----|----:|----------:|-------------------|--------:|---------:|----------:|-------------:|
| auto_markers_c4_m32 | 4 | 32 | off | 1215 | 6.50 | 269 | 4.54 |
| auto_markers_c4_m16 | 4 | 16 | off | 1177 | 6.48 | 316 | 4.95 |
| **auto_markers_c8_m32** | **8** | **32** | **on** | **1095** | **7.11** | **331** | **7.11** |

### Stage detail vs control

| Stage | c4_m32 (control) | c4_m16 | c8_m32 (speed pack) |
|-------|-----------------:|-------:|--------------------:|
| createStore | 107s / 3.5G | 111s / 3.4G | 104s / 3.5G |
| initializeStore | 196s / 6.5G | 137s / 6.5G | 141s / 6.4G |
| reopenStore | 9s / 0.5G | 8s / 0.5G | 7s / 0.5G |
| filterCells | 23s / 0.5G | 22s / 0.5G | 20s / 0.5G |
| markHvgs | 255s / 1.1G | 225s / 1.4G | 241s / 1.1G |
| makeGraph | 183s / 6.3G | 175s / 5.9G | 168s / 6.3G |
| runUmap | 146s / 0.7G | 155s / 0.7G | **58s / 0.7G** |
| runLeiden | 27s / 0.7G | 27s / 0.7G | 25s / 0.7G |
| findMarkers | **269s / 4.5G** | 316s / 5.0G | 331s / 7.1G |
| **total** | **1215s** | **1177s** | **1095s** |

### Learnings (100k resources)

1. **Auto marker batches beat forced batch 50** on markers (353s → 269s at 4CPU/32GiB) and raise marker peak (1.0 → 4.5 GiB). That is productive use of Scarf budget.
2. **Shrinking Modal RAM alone is the wrong optimization for speed.** 16 GiB vs 32 GiB: totals similar; markers got slower (269 → 316s). Peaks stayed ~6.5 GiB on non-marker heavy stages.
3. **Extra Modal RAM does nothing unless Scarf `memoryBytes` grows** so auto batches can grow under fixed `workingCopies`.
4. **Parallel UMAP is the clearest speed win** (146s → 58s). Almost no RAM change.
5. **8 workers did not speed markers**; they slowed them (269 → 331s) even with higher peak (7.1 GiB). Treat marker thread/worker scaling as unresolved.
6. **HVG and markers dominate** after UMAP is fixed (~22% each of the 4CPU control total).

## Scale: 100k → 250k (same speed pack)

Config family: 8 CPU / 32 GiB, workers=8, workingCopies=4, `annParallel=true`, `umapParallel=true`, auto markers, cloud 128 MiB chunks.

| Stage | 100k (`c8_m32`) | 250k (`c8_m32_250k`) | Time scale | Peak 100k | Peak 250k |
|-------|----------------:|---------------------:|-----------:|----------:|----------:|
| createStore | 104s | 264s | 2.53× | 3.5G | 3.8G |
| initializeStore | 141s | 320s | 2.27× | 6.4G | 6.6G |
| reopenStore | 7s | 8s | 1.07× | 0.5G | 0.5G |
| filterCells | 20s | 21s | 1.06× | 0.5G | 0.5G |
| markHvgs | 241s | 585s | 2.42× | 1.1G | 1.3G |
| makeGraph | 168s | 307s | 1.83× | 6.3G | **14.6G** |
| runUmap | 58s | 101s | 1.74× | 0.7G | 0.9G |
| runLeiden | 25s | 51s | 2.04× | 0.7G | 1.1G |
| findMarkers | 331s | 688s | 2.08× | 7.1G | 8.8G |
| **total** | **1095s** | **2346s** | **2.14×** | **7.1G** | **14.6G** |

Cells ×2.5 → wall ×2.14 (slightly better than linear).  
**makeGraph peak roughly doubled** (6.3 → 14.6 GiB); still under 32 GiB. Markers remain the slowest stage at both sizes; HVG second.

Configs:

- `profiling/layouts/100k_auto_markers_c8_m32.toml`
- `profiling/layouts/250k_auto_markers_c8_m32.toml`

## What actually moves speed

| Lever | Helps? | Evidence |
|-------|--------|----------|
| Cloud ~128 MiB count chunks | Yes | Layout sweep totals |
| Auto marker batch (larger Scarf budget) | Yes for markers | 353s → 269s @ 4c/32G |
| `umapParallel` / more CPU for UMAP | Yes | 146s → 58s |
| More Modal RAM with same Scarf budget | No | Peaks unchanged |
| Cutting Modal RAM for its own sake | No for speed | 16G markers slower |
| Raising `workers` for markers | Not shown; can hurt | 4→8 workers, markers +62s |
| Tuning `workingCopies` for speed | Do not | It is a copy-count model, not a perf knob |

## Ops notes (Modal)

- Prefer bare `Function.from_name(...).spawn` on the deployed app for `run_all_jobs`.
- Tight `with_options` + `region=eu-west-1` + high memory often sat in queue; capacity messages mentioned 16.8 / 28.8 GiB.
- Never `modal deploy` from the agent; user deploys after code changes.
- Use `uv` for local Python / Modal CLI.

## Current best reference profiles

| Size | Recommended reference tag | Machine | Notes |
|------|---------------------------|---------|-------|
| 100k | `auto_markers_c8_m32` | 8 CPU / 32 GiB | Fastest complete funnel so far (1095s) |
| 250k | `auto_markers_c8_m32_250k` | 8 CPU / 32 GiB | Same settings; 2346s, max peak 14.6 GiB |

For a pure markers comparison at 100k without parallel UMAP noise, use `auto_markers_c4_m32` (269s markers).

## Proposed experiment: Modal 32 GiB, Scarf budget 16 GiB @ 100k

Yes. Isolates software budget from machine size.

| Run | Modal | Scarf budget | Role |
|-----|------:|-------------:|------|
| `auto_markers_c4_m32` | 32 | ~24 | Reference (same CPU/parallel/workers/wc) |
| `auto_markers_c4_m16` | 16 | ~12 | Both knobs cut (confounded) |
| `auto_markers_c4_m32_scarf16` | 32 | **16** | Mismatch: machine rich, Scarf poor |

Config: `profiling/layouts/100k_auto_markers_c4_m32_scarf16.toml`  
Call id: `fc-01KXKEX6N0N44661M6ZX26NX9C`  
Matched to `c4_m32`: 4 CPU, workers=4, workingCopies=4, UMAP/ANN parallel off. Only `scarfMemoryBudget` changes. Store is created under the 16 GiB budget so write geometry matches.

What to read when done: markers seconds/peak vs `c4_m32` (budget binding) and vs `c4_m16` (whether Modal size mattered once budget is ~16-ish).


## Open questions before / during 500k

1. Does makeGraph peak keep ~linear (~30 GiB at 500k)? If so, 32 GiB Modal may OOM; plan a higher mem ceiling from evidence, not from habit.
2. Do marker auto batches stay budget-capped usefully as `n_cells` grows (`budgetCap ∝ 1/n_cells`)?
3. Can HVG use memory/CPU more productively (still ~1–1.4 GiB peak while taking 20%+ of wall)?
4. Why did 8 workers slow markers vs 4 at 100k?
5. Modal 32 + Scarf 16 mismatch at 100k (above): does wall time follow Scarf budget when the machine is not the limit?

## Next / in progress: 500k

| Field | Value |
|-------|-------|
| Tag | `auto_markers_c8_m48_500k` |
| Config | `profiling/layouts/500k_auto_markers_c8_m48.toml` |
| Machine | 8 CPU / 48 GiB Modal (`49152` MiB), Scarf budget 36 GiB (75%) |
| Same as speed pack | workers=8, workingCopies=4, UMAP/ANN parallel on, auto markers, cloud 128 MiB chunks |
| Call id | `fc-01KXKE1ZCG0FG7KFEQ1H0QX35K` |
| Why 48 GiB | 250k makeGraph peaked at 14.6 GiB; ~linear cell scaling suggested ~30 GiB at 500k, so 32 GiB looked risky |

Compare against `auto_markers_c8_m32` (100k) and `auto_markers_c8_m32_250k` (250k) when complete. Update this section with stage table after the run finishes.

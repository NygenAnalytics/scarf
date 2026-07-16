# Scarf cloud profiling learnings (100k → 1M)

Date range: 2026-07-14 to 2026-07-16  
Environment: Modal `scarf_profiling`, app `scarf-profiling`, region `eu` (was `eu-west-1`; broadened for capacity), secret `scarf-r2`  
Data: `s3://scarf-tests/scarf-profiling/` (datasets / stores / results)  
Dataset source: nested CELLxGENE samples already prepared on R2

This note is the baseline for quantifying later changes (code defaults, orchestrator CPU/mem tables, layout ideas). Times are stage wall seconds from result JSON. Peaks are `peakCgroupBytes` unless noted.

## Objective

Minimize wall time without pointless overprovisioning. Prefer using memory and CPU in ways that actually cut stage time. Do not treat `workingCopies` as a speed dial.

## Agent / ops rules (mandatory)

Long cloud jobs must survive a dead laptop Wi-Fi or WSL network drop. Treat local connectivity as unreliable.

1. **Never drive long work with `Function.remote()` or a blocking `modal run` that waits on the result.** When the local Modal client loses its gRPC session, Modal can cancel the running input. That is what killed the 1M IO baseline mid-`markerBatches` after ~40 minutes of good progress (`TimeoutError` / `Deadline exceeded` on the client, then `Received a cancellation signal` on the container). `--detach` alone is not enough if the entrypoint still uses `.remote()`.
2. **Prefer spawn on the deployed app**, then disconnect: `Function.from_name(...).spawn(...)` (same pattern as `run_all_jobs`). Persist results to R2 (or another durable URI) from inside the container. Poll with `FunctionCall.from_id(...).get(timeout=0)` or by reading the result object; do not hold an open await for hours.
3. **One-off scripts must write a result JSON to R2 before returning**, and log enough that progress is recoverable from `modal app logs` alone if the client dies.
4. **Log well for long jobs.** Flush stdout. Emit pattern/stage start, a plan line (counts, chunk sizes, block totals), progress every N blocks (wall, bytes, rate), and a done line (wall, peak RSS/cgroup, bytes). Silent multi-hour runs are unacceptable.
5. Never `modal deploy` from the agent; user deploys. Use `uv` for local Python / Modal CLI.
6. Prefer broad Modal region `eu` over narrow `eu-west-1` for capacity.

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
| `auto_markers_c4_m32_scarf16` | 100k | 4 | 32 GiB | **16 GiB** | Done; markers 219s (faster than c4_m32) |
| `auto_markers_c4_m16` | 100k | 4 | 16 GiB | ~12 GiB | Both Modal and Scarf cut together |
| `auto_markers_c8_m32` | 100k | 8 | 32 GiB | ~24 GiB | Speed pack (UMAP/ANN parallel) |
| `auto_markers_c8_m32_250k` | 250k | 8 | 32 GiB | ~24 GiB | Same speed pack |
| `auto_markers_c8_m48_500k` | 500k | 8 | 48 GiB | 36 GiB | Done; 4339s, max peak ~10 GiB |
| `auto_markers_c8_m64_1m` | 1M | 8 | 64 GiB | 48 GiB | Done; 9156s, max peak 28.3 GiB (makeGraph) |

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
5. Spawn via deployed app bare `run_all_jobs.spawn(...)` (see Agent / ops rules). Stage workers still apply config resources inside the deployed orchestrator.

Useful call IDs from this campaign (`/tmp/scarf_calib_calls.txt`):

```
auto_markers_c4_m32=fc-01KXJX3H7988EH0TZM6EGP7AQC
auto_markers_c4_m16=fc-01KXJZ3VFQKMZ0T77ET87AD7KD
auto_markers_c8_m32=fc-01KXK3JTNTC44X44K3NFQRTDQ5
auto_markers_c8_m32_250k=fc-01KXK8BRWK9P50XW0NY4FWE378
auto_markers_c8_m48_500k=fc-01KXKE1ZCG0FG7KFEQ1H0QX35K
auto_markers_c4_m32_scarf16=fc-01KXKEX6N0N44661M6ZX26NX9C
auto_markers_c8_m64_1m=fc-01KXKR82TFWSJS9EP1E4P5ZXCJ
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

- See **Agent / ops rules** above (spawn, no long `.remote()`, logging, R2 results).
- Prefer broad Modal region `eu` over narrow `eu-west` / `eu-west-1` for capacity ([region selection](https://modal.com/docs/guide/region-selection); broad multiplier 1.5x vs narrow 1.75x).
- Tight region + high memory often sat in queue; capacity messages mentioned 16.8 / 28.8 / 48.8 GiB under `eu-west-1`.
- Stage result JSON currently stores wall + memory peaks only (no CPU utilization). Modal dashboard has live CPU charts; CLI billing is app-day cost, not per-stage idle cores.

## Current best reference profiles

| Size | Recommended reference tag | Machine | Notes |
|------|---------------------------|---------|-------|
| 100k | `auto_markers_c8_m32` | 8 CPU / 32 GiB | Fastest 100k funnel (1095s) |
| 250k | `auto_markers_c8_m32_250k` | 8 CPU / 32 GiB | Same settings; 2346s, max peak 14.6 GiB |
| 500k | `auto_markers_c8_m48_500k` | 8 CPU / 48 GiB | 4339s; Modal RAM was generous vs peaks |
| 1M | `auto_markers_c8_m64_1m` | 8 CPU / 64 GiB | 9156s; makeGraph peak 28.3 GiB justifies 64 GiB |

For a pure markers comparison at 100k without parallel UMAP noise, use `auto_markers_c4_m32` (269s markers).

## Budget mismatch @ 100k: Modal 32 GiB, Scarf 16 GiB (done)

Tag `auto_markers_c4_m32_scarf16`. Matched to `c4_m32` (4 CPU, workers=4, workingCopies=4, UMAP/ANN off). Only Scarf budget cut to 16 GiB; Modal stayed 32 GiB. Store built under that budget.

| Stage | c4_m32 (~24G Scarf) | scarf16 (16G Scarf) | Δ |
|-------|--------------------:|--------------------:|--:|
| createStore | 107s / 3.5G | 115s / 3.3G | +9s |
| initializeStore | 196s / 6.5G | 115s / 6.6G | −81s |
| markHvgs | 255s / 1.1G | 226s / 1.2G | −30s |
| makeGraph | 183s / 6.3G | 165s / 6.1G | −18s |
| runUmap | 146s / 0.7G | 164s / 0.7G | +19s |
| findMarkers | **269s / 4.5G** | **219s / 6.1G** | **−51s** |
| **total** | **1215s** | **1052s** | **−163s** |

**Reading:** Lower Scarf budget did **not** slow markers here. Markers were faster with a higher peak (6.1 vs 4.5 GiB). Contrast `c4_m16` (Modal 16 + Scarf ~12): markers 316s / 5.0G. So cutting Modal+budget together hurt markers; cutting only Scarf budget did not. Do not treat this single A/B as proof that budget is unbound; geometry/noise may dominate. Call `fc-01KXKMXS06BTJTBE7JVR5N8BWZ`.

## Open questions

1. Why scarf16 markers faster than c4_m32 despite smaller software budget?
2. Can HVG use memory/CPU more productively (still ~1–1.4 GiB peak while taking 20%+ of wall)? Partial answer: IO baseline `hvgTiles` matched most of markHvgs wall (see below).
3. Why did 8 workers slow markers vs 4 at 100k?
4. makeGraph cgroup peak 250k→500k drop: re-measure with clean A/B before trusting.
5. Feature-major secondary matrix for gene-wise stages: **tabled** until IO baseline finishes cleanly; revisit with 1M HVG/marker evidence.

## Scale: 500k (8 CPU / 48 GiB, region `eu`)

Tag `auto_markers_c8_m48_500k`. Same speed-pack knobs as 100k/250k (workers=8, workingCopies=4, UMAP/ANN parallel, auto markers). Modal 48 GiB / Scarf 36 GiB. Respawned late stages onto broad region `eu` after `eu-west-1` capacity stalls.

| Stage | 100k | 250k | 500k | 250→500 time |
|-------|-----:|-----:|-----:|-------------:|
| createStore | 104s / 3.5G | 264s / 3.8G | 474s / 4.5G | 1.80× |
| initializeStore | 141s / 6.4G | 320s / 6.6G | 592s / 7.0G | 1.85× |
| markHvgs | 241s / 1.1G | 585s / 1.3G | 1095s / 1.1G | 1.87× |
| makeGraph | 168s / 6.3G | 307s / 14.6G | 436s / 9.3G | 1.42× |
| runUmap | 58s / 0.7G | 101s / 0.9G | 143s / 1.2G | 1.42× |
| runLeiden | 25s / 0.7G | 51s / 1.1G | 100s / 1.4G | 1.96× |
| findMarkers | 331s / 7.1G | 688s / 8.8G | 1473s / 10.1G | 2.14× |
| **total** | **1095s** | **2346s** | **4339s** | **1.85×** |

Cells 250k→500k is 2×; wall time 1.85× (slightly better than linear). Max peak at 500k ~10.1 GiB (markers), so 48 GiB was generous. makeGraph cgroup peak 9.3 GiB is *below* 250k's 14.6 GiB; treat that drop cautiously (cgroup vs RSS gap on 250k, restart mid-run). See discussion in chat / prefer RSS for graph sizing.

Call ids: original `fc-01KXKE1ZCG0FG7KFEQ1H0QX35K` (cancelled); eu resume `fc-01KXKH7Z9V88NXFG2DJ204C8C8`.

## Scale: 1M (8 CPU / 64 GiB, region `eu`)

Tag `auto_markers_c8_m64_1m`. Same speed-pack knobs. Modal 64 GiB / Scarf 48 GiB. Call `fc-01KXKR82TFWSJS9EP1E4P5ZXCJ`.

| Stage | 100k | 250k | 500k | 1M | 500k→1M time | Peak 1M |
|-------|-----:|-----:|-----:|---:|-------------:|--------:|
| createStore | 104s | 264s | 474s | 974s | 2.05× | 5.4G |
| initializeStore | 141s | 320s | 592s | 1241s | 2.10× | 7.0G |
| markHvgs | 241s | 585s | 1095s | 2546s | 2.33× | 1.1G |
| makeGraph | 168s | 307s | 436s | 1487s | 3.41× | **28.3G** |
| runUmap | 58s | 101s | 143s | 313s | 2.19× | 1.6G |
| runLeiden | 25s | 51s | 100s | 185s | 1.85× | 2.3G |
| findMarkers | 331s | 688s | 1473s | 2379s | 1.62× | 10.1G |
| **total** | **1095s** | **2346s** | **4339s** | **9156s** | **2.11×** | **28.3G** |

Cells 500k→1M is 2×; wall ~2.11×. **makeGraph is the outlier** (~3.4× time, peak 28.3 GiB cgroup / 21.0 GiB RSS): 64 GiB Modal is justified here, unlike 500k where 48 GiB was generous. HVG + markers still dominate wall (~54%). HVG peak stays ~1 GiB while taking ~28% of wall.

### What `initializeStore` is

First cold open after `createStore` with QC thresholds. It is not a cheap open: it streams the raw matrix to compute per-gene `nCells` / dropOuts and per-cell `nCounts` / `nFeatures` / percentMito / percentRibo, then applies min-feature filters. Later stages use `initialize=False` and skip that work (`reopenStore` ~6s at 1M). Wall scales roughly with cells; peak stays ~7 GiB (IO/reduction bound).

## IO baseline (1M, same machine; partial)

Goal: no-compute stream of HVG / marker / makeGraph read patterns on
`s3://scarf-tests/scarf-profiling/stores/auto_markers_c8_m64_1m/1000000.zarr`
at 8 CPU / 64 GiB / `eu`, to separate R2+layout cost from Scarf compute.

Script: `profiling/io_baseline.py`. Must be spawned (not long `.remote()`); write result to
`{resultsUri}/io-baseline/{runTag}.json`.

| Pattern | Status | Wall | Peak RSS | Bytes read | Notes |
|---------|--------|-----:|---------:|-----------:|-------|
| openStore | done | 7.7s | 0.53 GiB | - | Like reopen; QC columns already present |
| hvgTiles | done | **2347s** | 1.0 GiB | 108.4 GiB | 7912 physical tiles, chunks `(11671, 500)` |
| markerBatches | cancelled | - | - | - | Died ~batch 12/73 after local client disconnect |
| makeGraphRawCellBands | not run | | | | |
| makeGraphNormedCellBands | not run | | | | |

**Reading so far:** HVG-style tile IO alone (~39 min) is already close to full `markHvgs` wall (2546s) at ~1 GiB peak. That strongly supports HVG being IO/layout bound, not RAM-starved. Feature-major secondary matrix remains tabled until marker + makeGraph IO baselines complete under a spawn-safe run.

Active cells after QC ~890k; auto marker batch 452; HVGs 2000.

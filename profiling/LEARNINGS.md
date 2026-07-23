# Scarf cloud profiling learnings (50k → 10M, countsT + Paris)

Date range: 2026-07-14 to 2026-07-23
Environment: Modal `scarf_profiling`, app `scarf-profiling`, region `eu` (was `eu-west-1`; broadened for capacity), secret `scarf-r2`
Data: `s3://scarf-tests/scarf-profiling/` (datasets / stores / results)
Dataset source: nested CELLxGENE samples already prepared on R2

This note is the baseline for quantifying later changes (code defaults, orchestrator CPU/mem tables, layout ideas). Times are stage wall seconds from result JSON. Peaks are `peakRssBytes` / `peakCgroupBytes` as reported (RSS unless noted).

## Objective

Minimize wall time without pointless overprovisioning. Prefer using memory and CPU in ways that actually cut stage time. Do not treat `workingCopies` as a speed dial.

**Default layout going forward: feature-major `countsT` (Zarr v3).** Row-major tags remain as historical controls. New funnel profiles should use countsT stores.

## Agent / ops rules (mandatory)

Long cloud jobs must survive a dead laptop Wi-Fi or WSL network drop. Treat local connectivity as unreliable.

1. **Never use `Function.remote()` in profiling.** Local entrypoints and in-app orchestrators must `.spawn(...)` only. Waiting is via short `FunctionCall.get(timeout=…)` polls and/or durable R2 result JSON (`profiling/spawn_wait.py`). A long `.remote()` or a blocking `modal run` that waits on the result can cancel the input when the client gRPC session dies (this killed the 1M IO baseline mid-`markerBatches`).
2. **Prefer spawn on the deployed app**, then disconnect: `Function.from_name(...).spawn(...)`. Persist results to R2 from inside the container. Laptop clients print the call id and exit; orchestrators poll R2 / short `get()` so their own heartbeats stay alive.
3. **One-off scripts must write a result JSON to R2 before returning**, and log enough that progress is recoverable from `modal app logs` alone if the client dies.
4. **Log well for long jobs.** Flush stdout. Emit pattern/stage start, a plan line (counts, chunk sizes, block totals), progress every N blocks (wall, bytes, rate), and a done line (wall, peak RSS/cgroup, bytes). Silent multi-hour runs are unacceptable.
5. Never `modal deploy` from the agent; user deploys. Use `uv` for local Python / Modal CLI.
6. Prefer broad Modal region `eu` over narrow `eu-west-1` for capacity.
7. Do **not** set Modal `cloud=` (e.g. aws). Provider pinning shrinks the pool; leave cloud unset so Modal can schedule any provider.
8. Coordinators (`run_all_jobs` / `run_size_jobs`) must use tiny resources (~2–4 GiB / 1 CPU) via `orchestrator_function_options` with `retries=0`. Never spawn them with stage RAM (32–64 GiB); that competes with real stage workers for scarce high-memory capacity.
9. Stage workers use Modal `Retries(max_retries=3)` via `modal_function_options`. `run_size_jobs` also re-spawns a stage up to 3 times if the call dies without an R2 result (covers `InternalFailure: Server has lost track of input`).
10. Right-size Modal RAM per stage from measured peaks. Do not put Leiden/UMAP/HVG on a 64 GiB queue when peaks are ~5–8 GiB; that worsens scheduling and lost-input risk. Keep 64 GiB for createStore / makeGraph at large N when peaks justify it.
11. When polling spawned calls, treat call-graph `FAILURE` / `INIT_FAILURE` / `TERMINATED` / `TIMEOUT` as terminal. Modal can raise an empty `TimeoutError` from `get(timeout=…)` on failed inputs; spinning on that alone leaves orchestrators stuck until the stage deadline.
12. Local-vs-remote store A/B: use `run-local` / `run_local_funnel_job` (one container, H5AD + Zarr on ephemeral disk, `fast_local`). Per-stage `run` / `run-all` always write the store to R2. Compare tags like `local_ephemeral_c8_m32_100k` vs `counts_t_c8_m32_100k_reorg`.
13. **Run Leiden and Paris in a child process for large N.** Modal's runner heartbeat (~900s) is not configurable. Historical `leidenalg` and the former scikit-network Paris path could block the parent heartbeat during long native calls. Paris now uses Scarf's native Numba implementation, but profiling keeps both stages isolated so the parent can report progress and failures consistently. `profiling/leiden_worker.py` and `profiling/paris_worker.py` log every 30s and warn at 1800s without killing the child. Do not try to raise the heartbeat threshold; keep the child-process path.
14. **`countsT` durability:** `complete=False` until every tile is written. A full-shaped array with `complete=False` is untrusted (interrupted or overlapping writers). Do not flip the attr; re-run `write_counts_t`. Overlapping createStore retries on the same store can leave this state. Use `retries=0` for repair-only rewrites.

## Code changes already wired

| Change | Location | Effect |
|--------|----------|--------|
| Cloud default `targetChunkBytes` = 128 MiB when remote and unset | `scarf/storage/zarr_store.py` (`DEFAULT_CLOUD_TARGET_CHUNK_BYTES`) | Matches layout-sweep winner |
| Auto marker batch when `gene_batch_size is None` | `scarf/features/markers/batching.py` `resolve_marker_gene_batch_size` | `min(col_chunk, n_features, budgetCap)` with `budgetCap = (memoryBytes // workingCopies) // (n_cells * 32)` |
| Optional profiling override | `profiling` `markerGeneBatchSize` | Layout sweep forced `50`; later runs leave unset for auto |
| Leiden child-process isolation | `profiling/stages.py`, `profiling/leiden_worker.py` | Parent keeps Modal heartbeats alive during long `leidenalg` GIL holds |
| Paris child-process isolation | `profiling/stages.py`, `profiling/paris_worker.py` | Same heartbeat pattern; fixed or adaptive Paris cut |
| Guarded adaptive Paris profiling | `profiling/paris_profile.py`, `profiling/paris_quality_gate.py` | Measures the same persistence plus modularity cut shipped by `run_paris_clustering` |
| `fixedResources` expands per stage | `profiling/config.py` `_normalize_raw_config` | One resource block applies to every selected stage |
| `--ephemeral` spawn | `profiling/modal_app.py` `run` / `run-all` | Spawn from `modal run` app without deploy; prefer `--detach` |

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
| `auto_markers_c8_m32` | 100k | 8 | 32 GiB | ~24 GiB | Row-major speed pack (UMAP/ANN parallel) |
| `auto_markers_c8_m32_250k` | 250k | 8 | 32 GiB | ~24 GiB | Same speed pack |
| `auto_markers_c8_m48_500k` | 500k | 8 | 48 GiB | 36 GiB | Done; 4339s; makeGraph peak 9.3G looks low vs re-run |
| `auto_markers_c8_m32_500k` | 500k | 8 | 32 GiB | 24 GiB | Row-major right-size; 3906s, max peak **23.0 GiB** (makeGraph) |
| `auto_markers_c8_m64_1m` | 1M | 8 | 64 GiB | 48 GiB | Row-major; 9156s, max peak 28.3 GiB (makeGraph) |
| `auto_markers_c8_m64_2_5m` | 2.5M | 8 | 64 GiB | 48 GiB | Row-major; **15292s**, makeGraph peak 24.5 GiB |
| `counts_t_c8_m32_100k` | 100k | 8 | 32 GiB | ~24 GiB | Feature-major `countsT`; **881s** (vs 1095s row-major) |
| `counts_t_c8_m32_100k_reorg` | 100k | 8 | 32 GiB | ~24 GiB | Post-reorg R2 control; **735s** (same knobs, fresh store) |
| `local_ephemeral_c8_m32_100k` | 100k | 8 | 32 GiB | ~24 GiB | Same knobs; Zarr on Modal `/tmp` (`fast_local`); **421s** |
| `counts_t_c8_m32_500k` | 500k | 8 | 32 GiB | 24 GiB | Feature-major `countsT`; **2825s** (vs 3906s row-major) |
| `counts_t_c8_m64_5m` | 5M | 8 (Leiden 2) | right-sized 8–64 GiB | ~75% of Modal | countsT core; **29465s (~8.2 h)**; max peak **33.0 GiB** (makeGraph cgroup) |
| `counts_t_c8_m64_10m` | 10M | 8 (Leiden 2) | right-sized 16–64 GiB | ~75% of Modal | countsT core; corrected wall **~22.8 h**; HVG JSON under-timed |
| `core_paris_c2_m16_50k` | 50k | 2 | 16 GiB | 12 GiB | Full core + Paris; **693s**; max peak **6.5 GiB** (initializeStore) |
| Paris-only (`paris_c2_m16_*`) | 100k–5M | 2 | 16 GiB | 12 GiB | Legacy balanced-cut runs retained as a historical baseline |

Local layout TOMLs under `profiling/layouts/` are gitignored. Treat `LEARNINGS.md` as the durable record; recreate TOMLs from these rows when needed.

## Constants that must stay stable when comparing

| Knob | Value used in successful speed runs | Rule |
|------|-------------------------------------|------|
| `workingCopies` | 4 in profiling configs (library default is 8) | Model of concurrent in-memory copies. Change only if peaks/OOM show the model is wrong |
| Assay / workflow seeds | graph 4466, umap/leiden 4444, `topN=2000`, `k=11`, `dims=50` | Keep fixed across A/B |
| Modal vs Scarf coupling | Usually Scarf ≈ 75% of Modal | Decouple on purpose only for budget-mismatch experiments (below) |

## How to quantify a future change

1. Pick a fixed reference run tag (see tables below). Prefer `counts_t_c8_m32_100k` / `counts_t_c8_m32_500k` when comparing gene-wise IO. Use `auto_markers_c8_m32*` for row-major baselines.
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
auto_markers_c8_m64_2_5m=fc-01KXPPVXKA34Z0KK7MCJ41990T
auto_markers_c8_m32_500k=fc-01KXPCBMZGBQ02YCB9W1EKPF92
counts_t_c8_m32_100k=fc-01KXPPVDDF9A5J8599QM7ZW65Q
counts_t_c8_m32_500k=fc-01KXPR232W53HQXN4QPPNAD8D6
counts_t_c8_m64_5m_leiden=fc-01KY2DMQ331ZVJ8YCBN634BTJS
counts_t_c8_m64_5m_markers=fc-01KY2JQDPFWK689ZSV5QYK9H5E
counts_t_c8_m64_10m_resume=fc-01KY5TK4AH8MK0GAPDBJ374547
core_paris_c2_m16_50k=fc-01KY4TJ4J4CAXQQ0T6PG5PTTAQ
paris_100k=fc-01KY4TJQ76E8K4AHKRJJF4DVKH
paris_250k=fc-01KY4TKA0690V7767RG5K10RYA
paris_500k=fc-01KY4TKXM62KND6Y54H3FYYZDW
paris_1m=fc-01KY4TMFKCCG7YP0D7QZ31AW6B
paris_2_5m=fc-01KY4TN2DECFKJHZ57GP48DV8H
paris_5m=fc-01KY4TNN5D4QTFHAY9DDYKRHJE
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
| Feature-major `countsT` (Zarr v3) | **Yes** for HVG/markers | 100k/500k A/B below; createStore pays write cost |

## Ops notes (Modal)

- See **Agent / ops rules** above (spawn, no long `.remote()`, logging, R2 results).
- Prefer broad Modal region `eu` over narrow `eu-west` / `eu-west-1` for capacity ([region selection](https://modal.com/docs/guide/region-selection); broad multiplier 1.5x vs narrow 1.75x).
- Tight region + high memory often sat in queue; capacity messages mentioned 16.8 / 28.8 / 48.8 GiB under `eu-west-1`.
- Stage result JSON currently stores wall + memory peaks only (no CPU utilization). Modal dashboard has live CPU charts; CLI billing is app-day cost, not per-stage idle cores.

## Current best reference profiles

| Size | Recommended reference tag | Machine | Notes |
|------|---------------------------|---------|-------|
| 50k | `core_paris_c2_m16_50k` | 2 CPU / 16 GiB | Full core + Paris; **693s**; max peak 6.5 GiB |
| 100k | `counts_t_c8_m32_100k` | 8 CPU / 32 GiB | Fastest measured funnel (**881s**); row-major control `auto_markers_c8_m32` (1095s); Paris 42s |
| 250k | `auto_markers_c8_m32_250k` | 8 CPU / 32 GiB | Row-major only so far; 2346s, max peak 14.6 GiB; Paris 68s |
| 500k | `counts_t_c8_m32_500k` | 8 CPU / 32 GiB | Fastest measured (**2825s**); row-major control `auto_markers_c8_m32_500k` (3906s); Paris 140s |
| 1M | `auto_markers_c8_m64_1m` | 8 CPU / 64 GiB | Row-major 9156s; countsT not measured yet; Paris 274s |
| 2.5M | `auto_markers_c8_m64_2_5m` | 8 CPU / 64 GiB | Row-major **15292s**; makeGraph peak 24.5 GiB; Paris 1408s |
| 5M | `counts_t_c8_m64_5m` | right-sized 8–64 GiB | countsT core **29465s**; Paris 4492s / 14.8 GiB RSS |
| 10M | `counts_t_c8_m64_10m` | right-sized 16–64 GiB | countsT core done; HVG JSON **under-timed** (see 10M section) |

For a pure markers comparison at 100k without parallel UMAP noise, use `auto_markers_c4_m32` (269s markers).

## Scale summary (wall + max peak, speed pack)

| Cells | Tag | Layout | Modal | Total wall | Max peak | Stage at max peak |
|------:|-----|--------|------:|-----------:|---------:|-------------------|
| 100k | `auto_markers_c8_m32` | row | 32 GiB | 1095s | 7.1 GiB | findMarkers |
| 100k | `counts_t_c8_m32_100k` | **countsT** | 32 GiB | **881s** | 6.7 GiB | initializeStore |
| 250k | `auto_markers_c8_m32_250k` | row | 32 GiB | 2346s | 14.6 GiB | makeGraph |
| 500k | `auto_markers_c8_m32_500k` | row | 32 GiB | 3906s | **23.0 GiB** | makeGraph |
| 500k | `counts_t_c8_m32_500k` | **countsT** | 32 GiB | **2825s** | 16.6 GiB | makeGraph |
| 1M | `auto_markers_c8_m64_1m` | row | 64 GiB | 9156s | **28.3 GiB** | makeGraph |
| 2.5M | `auto_markers_c8_m64_2_5m` | row | 64 GiB | **15292s** | 24.5 GiB | makeGraph |

Rough Modal RAM floor from peaks: 100k ≥8 GiB, 250k ≥16 GiB, 500k ≥24–32 GiB, 1M–2.5M ≥32–64 GiB.

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
2. Why did 8 workers slow markers vs 4 at 100k?
3. countsT at 1M / 2.5M: confirm createStore write cost vs gene-wise savings (estimates below; not measured).
4. Can createStore / finalize overlap or stream `countsT` write to cut the growing createStore share?

## Scale: 500k

### First run @ 48 GiB (superseded for peaks)

Tag `auto_markers_c8_m48_500k`. Modal 48 GiB / Scarf 36 GiB. Call resume `fc-01KXKH7Z9V88NXFG2DJ204C8C8`. Total **4339s**. Reported makeGraph peak 9.3 GiB is inconsistent with the 32 GiB re-run below; do not size machines from that 9.3 GiB figure.

### Right-size re-run @ 32 GiB (preferred)

Tag `auto_markers_c8_m32_500k`. Modal 32 GiB / Scarf 24 GiB. Call `fc-01KXPCBMZGBQ02YCB9W1EKPF92`.

| Stage | 100k | 250k | 500k@32G | 250→500 time | Peak 500k@32G |
|-------|-----:|-----:|---------:|-------------:|--------------:|
| createStore | 104s | 264s | 402s | 1.52× | 4.4G |
| initializeStore | 141s | 320s | 841s | 2.63× | 6.4G |
| markHvgs | 241s | 585s | 1062s | 1.82× | 1.3G |
| makeGraph | 168s | 307s | 585s | 1.91× | **23.0G** |
| runUmap | 58s | 101s | 166s | 1.64× | 1.1G |
| runLeiden | 25s | 51s | 88s | 1.73× | 1.4G |
| findMarkers | 331s | 688s | 732s | 1.06× | 6.4G |
| **total** | **1095s** | **2346s** | **3906s** | **1.66×** | **23.0G** |

32 GiB is tight but worked (makeGraph 23 GiB peak). Faster wall than the 48 GiB run (3906 vs 4339) despite less RAM; geometry/noise and queue effects may differ. Prefer **23 GiB** as the 500k makeGraph sizing number.

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

## IO baseline (1M, done)

No-compute stream of HVG / marker / makeGraph read patterns on
`s3://scarf-tests/scarf-profiling/stores/auto_markers_c8_m64_1m/1000000.zarr`
at 8 CPU / 64 GiB / `eu`. Spawn via deployed `io_baseline_job` (not long `.remote()`).
Result: `s3://scarf-tests/scarf-profiling/io-baseline/auto_markers_c8_m64_1m.json`.

Active cells after QC ~890k; feats ~32.7k; HVGs 2000; auto marker batch 452; raw chunks `(11671, 500)`.

| Pattern | Wall | Peak RSS | Bytes | Notes |
|---------|-----:|---------:|------:|-------|
| openStore | 6s | 0.52 GiB | - | QC columns already present |
| hvgTiles | **2510s** | 1.01 GiB | 108.4 GiB | 7912 physical tiles |
| markerBatches | **2158s** | 6.91 GiB | 108.4 GiB | 73 gene batches |
| makeGraphRawCellBands | 502s | 5.33 GiB | 6.6 GiB | cells × 2000 HVGs, 77 bands |
| makeGraphNormedCellBands | **29s** | 7.81 GiB | 6.6 GiB | dense float32, 4 bands, ~234 MiB/s |
| **total** | **5206s** | | | |

### IO vs full stage (same 1M store)

| Stage | Full wall | IO-only | Implied non-IO | IO share |
|-------|----------:|--------:|---------------:|---------:|
| markHvgs | 2546s | 2510s (hvgTiles) | ~36s | **~99%** |
| findMarkers | 2379s | 2158s (markerBatches) | ~221s | **~91%** |
| makeGraph | 1487s | 502s raw + 29s normed | mostly other work | not IO-dominated |

**Takeaways:**

1. Gene-wise stages on this row-sharded cloud layout are **R2/layout bound**. Extra Modal RAM or CPU will not fix HVG/markers.
2. Speeding HVG/markers means cheaper column access. Feature-major `countsT` delivered that (see A/B): HVG/markers cut to ~25–40% of row-major time at 100k/500k.
3. makeGraph’s expensive wall/peak is **not** explained by re-reading the dense HVG-normed matrix (29s); cost is normalize/materialize, PCA/ANN/kmeans, and multi-pass work.
4. Competing large-RAM jobs in `eu` queue each other (IO baseline delayed 2.5M; countsT 500k sat ~1.5h before createStore started).

## Scale: 2.5M row-major (8 CPU / 64 GiB, region `eu`)

Tag `auto_markers_c8_m64_2_5m`. Same speed-pack knobs as 1M. Call `fc-01KXPPVXKA34Z0KK7MCJ41990T` (after redeploy without `cloud=` pin).

| Stage | Seconds | Peak GiB |
|-------|--------:|---------:|
| createStore | 1778 | 6.3 |
| initializeStore | 9 | 0.6 |
| reopenStore | 10 | 0.6 |
| filterCells | 20 | 0.6 |
| markHvgs | **3019** | 2.1 |
| makeGraph | **3042** | **24.5** |
| runUmap | 742 | 3.2 |
| runLeiden | 681 | 5.2 |
| findMarkers | **5990** | 13.3 |
| **total** | **15292** (~4.2h) | **24.5** |

**Reading:** makeGraph fit comfortably in 64 GiB (24.5G peak, vs 28.3G at 1M). Gene-wise stages dominate: HVG + markers = 9009s (~59% of wall). Markers alone are 5990s with slow R2 block reads. initializeStore was cheap here because QC columns already existed on the store from the earlier createStore attempt.

## Feature-major `countsT` A/B (done)

Zarr v3 secondary matrix written on finalize (`write_counts_t` / `finalize_writer_counts`). RNA HVG/markers use `assay.rawDataT` when present. Same speed pack as row-major controls (8 CPU / 32 GiB, parallel UMAP/ANN, auto markers, cloud 128 MiB chunks). Fresh stores so createStore pays for `countsT` write.

| Stage | 100k row | 100k countsT | 500k row | 500k countsT |
|-------|---------:|-------------:|---------:|-------------:|
| createStore | 104s | **260s** | 402s | **1042s** |
| initializeStore | 141s | 166s | 841s | 418s |
| markHvgs | 241s | **82s** | 1062s | **261s** |
| makeGraph | 168s | 164s | 585s | 539s |
| runUmap | 58s | 57s | 166s | 159s |
| runLeiden | 25s | 24s | 88s | 94s |
| findMarkers | 331s | **102s** | 732s | **293s** |
| **total** | **1095s** | **881s** | **3906s** | **2825s** |

Calls: `fc-01KXPPVDDF9A5J8599QM7ZW65Q` (100k), `fc-01KXPR232W53HQXN4QPPNAD8D6` (500k).

**Learnings:**

1. **Gene-wise stages are transformed.** HVG ~34% / ~25% of row-major time at 100k / 500k. Markers ~31% / ~40% of row-major time.
2. **Net funnel wins despite slower createStore.** 100k: −214s (−20%). 500k: −1081s (−28%). createStore roughly 2.5× slower (countsT write), but HVG+markers savings more than pay for it.
3. **Gene-wise share of total drops** from ~52% (row 100k HVG+markers) to ~21% (countsT 100k). createStore becomes a larger fraction (~37% at countsT 500k).
4. makeGraph / UMAP / Leiden are essentially unchanged (layout-independent).

## Local ephemeral vs R2 store (100k, done)

Same knobs as the countsT speed pack (8 CPU / 32 GiB, parallel UMAP/ANN, auto markers). Control is the post-reorg R2 funnel (`counts_t_c8_m32_100k_reorg`, **735s**). Local run uses `run_local_funnel_job`: one Modal container, H5AD downloaded once from R2, Zarr written under `/tmp` with `fast_local` profile; stage result JSONs still land on R2 under tag `local_ephemeral_c8_m32_100k`. Call `fc-01KY0AQ54YCV6D210FG6KKWHH0`.

| Stage | Local `/tmp` | R2 reorg | Local / R2 | Local RSS | R2 RSS |
|-------|-------------:|---------:|-----------:|----------:|-------:|
| createStore | 130 | 230 | 0.56× | 4.8G | 3.4G |
| initializeStore | 52 | 99 | 0.53× | 7.9G | 6.4G |
| reopenStore | 0.1 | 6.0 | 0.02× | 0.8G | 0.5G |
| filterCells | 0.3 | 17 | 0.02× | 0.8G | 0.5G |
| markHvgs | 21 | 69 | 0.30× | 2.5G | 1.1G |
| makeGraph | 99 | 173 | 0.57× | 10.6G | 6.4G |
| runUmap | 25 | 42 | 0.59× | 1.3G | 0.8G |
| runLeiden | 13 | 24 | 0.52× | 1.5G | 0.7G |
| findMarkers | 82 | 74 | **1.10×** | 10.4G | 4.1G |
| **total** | **421** | **735** | **0.57×** | | |

Original pre-reorg R2 countsT total was **881s**; reorg alone cut that to 735s (−17%). Local then cut another −43% vs reorg.

**Learnings:**

1. **At 100k, store IO is still a large fraction of R2 wall.** Putting the Zarr on ephemeral disk nearly halves the funnel (735 → 421s). The biggest relative wins are cheap metadata/open stages (`reopenStore`, `filterCells`) and gene-wise HVG (0.30×), which matches earlier IO-baseline evidence that R2 block latency dominates gene-wise work.
2. **createStore / initialize / makeGraph / UMAP / Leiden all improve ~1.7–2× locally.** That is not "compute disappeared"; it is less waiting on remote reads/writes around the same algorithms. Local peaks are higher (makeGraph 10.6G vs 6.4G RSS): more of the working set stays hot in page cache instead of streaming from R2.
3. **findMarkers is the exception (~same or slightly slower locally).** Markers are already much less R2-bound after countsT; once gene-column access is fast, wall is closer to CPU/math. Small +7s is within noise / cache / batching differences, not a local regression signal.
4. **Do not treat 421s as the cloud product number.** Production Scarf-on-R2 remains the R2 figures (735s post-reorg). Local ephemeral is the ceiling for "how fast is this funnel when storage is free," useful for sizing how much further remote IO work can buy.
5. **Ops caveat:** local requires one long-lived container (`run-local`). Per-stage spawn cannot keep a `/tmp` store across workers. H5AD download time is outside stage timers (same as remote createStore).

## Scaling: countsT 100k / 500k / 5M / 10M

Early fit from only 100k and 500k totals (`T ≈ 0.211 · N^0.724`) projected **~4.5 h** at 5M. Measured 5M was **~8.2 h**. The two-point fit under-predicted large-N wall, especially gene-wise stages and createStore write.

| Cells | Wall (s) | Hours | createStore | markHvgs | makeGraph | findMarkers | Status |
|------:|---------:|------:|------------:|---------:|----------:|------------:|--------|
| 100k | **881** | **0.24** | 260 | 82 | 164 | 102 | **measured** |
| 500k | **2825** | **0.78** | 1042 | 261 | 539 | 293 | **measured** |
| 5M | **29465** | **8.18** | 11508 | 3809 | 6296 | 5215 | **measured** |
| 10M | **~82k*** | **~22.8*** | 22130 | **~16k*** | 10576 | 16404 | **measured*** |

\*10M JSON stage sum is **65871s (~18.3 h)** if you trust `markHvgs.json` (21.8s). That HVG number is a **cache hit** and must not be used for scaling. Use the corrected HVG estimate below; adjusted funnel wall is then **~22.8 h**.

500k → 5M is cells ×10 and wall ×10.4 (near-linear overall). 5M → 10M (corrected) is cells ×2 and wall ×~2.8 vs 5M.

**Practical read:** countsT still wins vs row-major at small/mid N, but do not trust the old 100k/500k power law past 1M. Prefer measured 5M / corrected 10M as large-N anchors.

## Peak RAM sizing (Modal cgroup)

countsT does **not** change makeGraph RAM much. Size machines from **makeGraph `peakCgroupBytes`** (Modal OOM signal); RSS is often lower.

### Measured peaks (countsT speed pack)

| Cells | Max peak | Stage | makeGraph RSS / cgroup | Markers | Leiden |
|------:|----------|-------|------------------------|--------:|-------:|
| 100k | 6.7G | initializeStore | 5.2 / 5.3G | 4.4G | 0.7G |
| 500k | 16.6G RSS | makeGraph | 16.6 / **24.5G** | 7.8G | 1.5G |
| 5M | **33.0G** | makeGraph | 26.5 / **33.0G** | 7.9G | **13.0G** |
| 10M | **36.2G** | makeGraph | 30.0 / **36.2G** | 15.2G | **24.5G** |

Row-major makeGraph cgroup for mid sizes: 1M **28.3G**, 2.5M **32.6G**. 5M/10M countsT makeGraph cgroup stays in the ~33–36G band.

### Suggested Modal RAM (countsT funnel, ~20% headroom)

| Cells | Peak driver | Est. need | Recommend |
|------:|-------------|----------:|----------:|
| 50k | init ~7G | ~7G | 16 GiB |
| 100k | init 6.7G | 6.7G | 16–32 GiB |
| 250k | makeGraph ~13G | ~13G | 16–32 GiB |
| 500k | makeGraph 24.5G cgroup | 24.5G | **32 GiB** |
| 1M | makeGraph 28.3G | ~28G | 48–64 GiB |
| 2.5M | makeGraph 32.6G | ~33G | 48–64 GiB |
| 5M | makeGraph **33.0G** | 33G | **64 GiB** createStore/makeGraph; Leiden ≥16–32 GiB; Paris ≥16 GiB |
| 10M | makeGraph **36.2G** | 36G | **64 GiB** makeGraph; Leiden ≥32 GiB; HVG ≥16–32 GiB (stream ~14G) |

Wall-time and RAM scale differently: gene-wise wall shrinks with countsT at small N; at 5M+, createStore + makeGraph + markers dominate wall, and makeGraph still sets the RAM floor.

## Native guarded Paris local profile (2026-07-23)

Synthetic directed 15-neighbor graphs, 8 threads, Python 3.14:

| Cells | Fit | Modularity guard | Guarded cut total | Estimated peak | Fit incremental RSS |
|------:|----:|-----------------:|------------------:|---------------:|--------------------:|
| 100k | 1.82s | 1.57s | 1.71s | 0.18 GiB | 0.15 GiB |
| 500k | 25.33s | 8.39s | 9.10s | 0.90 GiB | 0.71 GiB |

The guard adds a linear topology pass but remains below fit time at both sizes.
The RSS sampler covers fitting; the estimate covers fitting and guarded-cut
paths. These local synthetic measurements are not directly comparable to the
older cloud store timings below.

## Legacy Paris balanced-cut profile (2026-07-22)

These measurements predate the native Paris hierarchy and adaptive cut. The
balanced-cut settings below are no longer accepted by the profiling worker.
Current runs use `parisNClusters="auto"` with optional
`parisMinClusterSize`, or an integer `parisNClusters` for a fixed cut. The
older results remain useful only as a runtime and memory baseline.

Leiden size stats used for balanced cut (active cells under `I`):

| Store tag | Cells (active) | Leiden clusters | min size | max size |
|-----------|---------------:|----------------:|---------:|---------:|
| `core_paris_c2_m16_50k` | 44428 | 38 | 136 | 4593 |
| `counts_t_c8_m32_100k` | 88955 | 40 | 27 | 8500 |
| `auto_markers_c8_m32_250k` | 222443 | 43 | 20 | 19302 |
| `counts_t_c8_m32_500k` | 444909 | 53 | 63 | 33025 |
| `auto_markers_c8_m64_1m` | 889974 | 52 | 111 | 86697 |
| `auto_markers_c8_m64_2_5m` | 2377327 | 53 | 455 | 192883 |
| `counts_t_c8_m64_5m` | 4605638 | 58 | 283 | 319408 |

### Paris wall + peaks

| Cells | Store tag | Seconds | Peak RSS | Peak cgroup | vs Leiden (same store, prior run) |
|------:|-----------|--------:|---------:|------------:|-----------------------------------|
| 50k | `core_paris_c2_m16_50k` | **25** | 1.09 | 0.76 | Leiden 25s / 1.07G RSS |
| 100k | `counts_t_c8_m32_100k` | **42** | 1.21 | 0.88 | Leiden ~24s (R2 reorg) |
| 250k | `auto_markers_c8_m32_250k` | **68** | 1.59 | 1.26 | Leiden 51s |
| 500k | `counts_t_c8_m32_500k` | **140** | 2.24 | 1.89 | Leiden ~1.5G peak band |
| 1M | `auto_markers_c8_m64_1m` | **274** | 3.53 | 3.19 | Leiden 185s / 2.3G |
| 2.5M | `auto_markers_c8_m64_2_5m` | **1408** | 8.01 | 7.64 | Leiden 681s row-major |
| 5M | `counts_t_c8_m64_5m` | **4492 (~75 min)** | **14.77** | 14.39 | Leiden 1550s / 13.0G |

**Learnings:**

1. **16 GiB is enough for Paris through 5M** on these graphs. 5M peaked at 14.8 GiB RSS (tight vs 16 GiB Modal; little headroom).
2. **Paris is slower than Leiden at large N**, and the gap widens: ~1× at 50k, ~1.5× at 1M, ~2× at 2.5M, ~2.9× at 5M (4492s vs 1550s).
3. **Paris RAM grows with cells** (roughly with graph size), unlike gene-wise stages that stay flatter after countsT. Size Paris like a graph algorithm, not like markers.
4. **Keep child-process isolation for long Paris runs** so the parent can report heartbeats, progress, and failures independently of the native implementation.
5. Prefer reusing existing stores for Paris-only A/B; do not rebuild the funnel unless the graph is missing.

### 50k full core + Paris @ 2 CPU / 16 GiB (done)

Tag `core_paris_c2_m16_50k`. Fresh store. Call `fc-01KY4TJ4J4CAXQQ0T6PG5PTTAQ`.

| Stage | Seconds | Peak GiB (RSS / cgroup) |
|-------|--------:|------------------------:|
| createStore | 141 | 3.4 / 3.4 |
| initializeStore | 90 | **6.5 / 6.4** |
| reopenStore | 8 | 0.5 / 0.5 |
| filterCells | 18 | 0.5 / 0.5 |
| markHvgs | 45 | 1.1 / 1.1 |
| makeGraph | 166 | 4.4 / 4.9 |
| runUmap | 65 | 0.7 / 0.7 |
| runLeiden | 25 | 1.1 / 0.7 |
| findMarkers | 109 | 2.0 / 1.9 |
| runClustering | 25 | 1.1 / 0.8 |
| **total** | **693** | **6.5** |

**Reading:** 2 CPU / 16 GiB is comfortable for 50k end-to-end. Peak driver is `initializeStore`, not makeGraph. Paris and Leiden are both ~25s at this size.

## Non-core extras @ 250k (countsT, planned / in flight)

Tag `counts_t_extras_c8_m32_250k`. Config `profiling/layouts/250k_counts_t_extras_c8_m32.toml`. Same speed pack (8 CPU / 32 GiB) plus optional stages after the core funnel. Mapping query is nested **25k** on R2; `prepareMappingQuery` is timed separately; `runMapping` measures projection only (store opens in `inputSetupSeconds`).

| Stage | Purpose |
|-------|---------|
| core funnel | createStore…findMarkers on countsT 250k |
| `getImputed` | MAGIC operator + 25 genes (`cache_operator=True`) |
| `runClustering` | Paris, 20 clusters |
| `runPseudotime` | PBA using Leiden first/last cluster as source/sink |
| `prepareMappingQuery` | Build 25k query Zarr (prep cost) |
| `runMapping` | Project 25k onto 250k reference (mapping cost only) |
| `makeGraphHarmony` | Synthetic 4 batches + `make_graph(harmonize=True)` |
| `subsetZarr` | Export active cells to local Zarr |
| `toH5ad` | Export assay to local H5AD |

Spawned: `fc-01KXQYNZG3X6NZ5K8J08XNNR3G`

```bash
uv run --group profiling modal run --env scarf_profiling -m profiling.modal_app -- \
  run-all --config profiling/layouts/250k_counts_t_extras_c8_m32.toml
```

## Scale: 5M countsT core (done)

Tag `counts_t_c8_m64_5m`. Config `profiling/layouts/5m_counts_t_c8_m64.toml`. Core stages only; countsT via Zarr v3 finalize. Right-sized Modal RAM (64 GiB createStore/makeGraph; smaller elsewhere). Leiden finished after child-process isolation (`probe_leiden` `fc-01KY2DMQ331ZVJ8YCBN634BTJS`); markers `fc-01KY2JQDPFWK689ZSV5QYK9H5E`.

| Stage | Seconds | Peak GiB (RSS / cgroup) |
|-------|--------:|------------------------:|
| createStore | 11508 | 7.4 / 7.3 |
| initializeStore | 12 | 0.6 / 0.6 |
| reopenStore | 7 | 0.6 / 0.6 |
| filterCells | 31 | 0.6 / 0.6 |
| markHvgs | 3809 | 8.0 / 7.9 |
| makeGraph | 6296 | 26.5 / **33.0** |
| runUmap | 1036 | 5.5 / 5.5 |
| runLeiden | 1550 | **13.0** / 12.7 |
| findMarkers | 5215 | 7.9 / 7.8 |
| **total** | **29465 (~8.2 h)** | **33.0** |

Share of wall: createStore 39%, makeGraph 21%, markers 18%, HVG 13%, Leiden 5%, UMAP 4%.

### Leiden at 5M (ops lesson)

In-process `leidenalg` / native igraph both hit Modal **runner heartbeat timeout (~900s)** while holding the GIL on ~4.61M active cells / ~50.7M edges (~11 edges/cell, normal for k=11). Graph shape was not pathological. Scaling through 2.5M was smooth (Leiden 681s row-major); naive 5M extrapolate was ~20–25 min, and the successful child-process run took **1550s (~26 min)** at **13 GiB** RSS. Fix: keep historical `leidenalg` semantics, run it in `profiling.leiden_worker`, parent logs every 30s. Native igraph backend was removed from profiling.

### Markers at 5M

`findMarkers` **5215s (~87 min)**, peak ~7.9 GiB, 58 clusters. That is between the optimistic countsT 100k/500k extrapolate (~22 min) and row-major 1M→2.5M power-law (~4 h). countsT still helps, but large-N marker wall is no longer near the small-N curve.

## Scale: 10M countsT core (done; HVG caveat)

Tag `counts_t_c8_m64_10m`. Config `profiling/layouts/10m_counts_t_c8_m64.toml`. Resume orchestrator `fc-01KY5TK4AH8MK0GAPDBJ374547` after createStore/initializeStore. `countsT` repaired earlier (`complete=True`, ~4.2 h rewrite).

| Stage | Seconds (result JSON) | Peak GiB (RSS / cgroup) | Notes |
|-------|----------------------:|------------------------:|-------|
| createStore | 22130 | 7.7 / 7.7 | includes countsT write |
| initializeStore | 9551 | 7.4 / 7.3 | cell QC + `nCells`/`dropOuts` |
| reopenStore | 12 | 0.8 / 0.8 | |
| filterCells | 43 | 0.8 / 0.8 | |
| markHvgs | **21.8 (do not use)** | 0.7 / 0.7 | **under-timed; see below** |
| makeGraph | 10576 | 30.0 / **36.2** | |
| runUmap | 3817 | 10.2 / 10.1 | |
| runLeiden | 3316 | **24.5** / 24.1 | child-process path |
| findMarkers | 16404 | 15.2 / 14.9 | |
| **JSON total** | **65871 (~18.3 h)** | **36.2** | treats HVG as 22s |
| **Corrected total** | **~82k (~22.8 h)** | **36.2** | HVG ≈ **~16k s (~4.5 h)** |

### markHvgs at 10M (under-timing)

`markHvgs.json` reports **21.8s / ~0.7 GiB**. That is a **feature-stats cache hit**, not a full HVG compute.

What actually happened:

1. `initializeStore` does **not** mark HVGs. It only builds cell QC and per-feature `nCells`/`dropOuts`.
2. Real HVG cost is `set_feature_stats` (library-size-normalized mean/variance stream). Logs showed `feature stats block N/15824` at ~14 GiB RSS while `markHvgs` was still pending.
3. That first pass wrote `summary_stats_I` to the store, then the stage attempt failed to leave a durable result (preempt/retry).
4. The successful attempt reused the cache (`Using cached feature stats`), ran only dispersion fit + top-2000 selection, and wrote the 21.8s JSON.

**Accounting for LEARNINGS / scaling:** use an estimated full HVG wall of **~4.5 h (~16000s)** from the mid-pass progress (~44% of blocks after ~2 h wall). Peak for sizing: **~14 GiB**, not 0.7 GiB. Do **not** put 21.8s in scale plots or 5M→10M ratios.

Corrected funnel wall ≈ JSON total − 22 + 16000 ≈ **~22.8 h**.

### Other 10M notes

- makeGraph cgroup **36.2 GiB** under 64 GiB Modal: comfortable vs 5M’s 33.0 GiB.
- Leiden **3316s (~55 min)** / **24.5 GiB** RSS (about 2.1× 5M Leiden time, ~1.9× RSS).
- Markers **16404s (~4.6 h)** / 15.2 GiB (about 3.1× 5M marker wall).
- 16 GiB was tight for the real HVG stream (~14 GiB); prefer 32 GiB for 10M `markHvgs` if remeasuring.

## Landscape context (how to talk about scale)

This campaign does **not** claim Scarf is uniquely able to touch multi-million-cell data. Several mature stacks reach large N under different assumptions. The useful claim is narrower and measured:

**Scarf completed a full core funnel (createStore through findMarkers) at ~10M cells on CPU Modal with peak ~36 GiB, store on cloud R2 Zarr.** That combination (CPU, ~64 GiB class host, remote object store, full graph + markers on all cells) is uncommon in public writeups.

### What other stacks optimize for

| Stack | Strength at large N | Typical resource model | Notes for comparison |
|-------|---------------------|------------------------|----------------------|
| Scanpy (in-memory) | Proven ~1.3M demos | Often **~100+ GiB** host RAM for classic 1.3M workflows | Out-of-core / Dask / lazy Zarr paths exist for larger atlas work |
| Scanpy + Dask / lazy AnnData | Atlas-scale PCA and chunked pipelines | Multi-worker or chunked CPU/GPU | Scales via distributed or out-of-core design |
| rapids-singlecell | Very fast 1M on one GPU; 11M+ with multi-GPU | **GPU VRAM** (managed memory can spill to host) | Different hardware class than this Modal CPU campaign |
| ScaleSC and similar GPU pipelines | Reported ~10–20M-class runs | Large single-GPU or multi-GPU | Same: GPU-first, not 64 GiB CPU |
| BPCells (+ Seurat v5) | Excellent disk-backed norm / HVG / PCA; census-scale PCA on modest RAM | Local disk bitpacking; streaming C++ | Strong RAM story for matrix algebra; see below for end-to-end shape |

Sources consulted while drafting this note: Scanpy 1.3M usage notes and memory discussions; NVIDIA RAPIDS-singlecell blogs/papers; ScaleSC (Bioinformatics Advances); BPCells benchmarks and manuscript; Seurat v5 BPCells and sketch vignettes.

### BPCells / Seurat v5 (closest RAM peer)

BPCells is a strong reference for **low-RAM, disk-backed** RNA/ATAC matrix work. Public benchmarks show normalize + PCA and even **44M-cell census PCA** within laptop/server RAM envelopes that Scanpy in-memory cannot match.

For **end-to-end** Seurat analysis at millions of cells, the documented product path is often:

1. Keep the full matrix on disk with BPCells.
2. **Sketch** a representative subset into memory (commonly on the order of tens of thousands of cells).
3. Run neighbors / clustering / UMAP on the sketch.
4. **Project** labels and embeddings back to the full dataset.

That is a deliberate, well-supported design for interactive large-N work. It is not the same job as running graph construction, Leiden/UMAP, and markers on **every** cell under a fixed CPU memory cap.

Community issues also show integration friction that is normal for a fast-moving stack: Seurat steps that historically materialized on-disk layers into `dgCMatrix` (2^31 nnz limits), marker/logFC differences while BPCells marker paths matured, slower full-object plotting unless using the sketch assay, and some Seurat options (for example regress-out during `ScaleData`) that need BPCells-native helpers. Fixes and workarounds exist; the point for Scarf is only that "BPCells PCA at atlas scale" and "full-N cloud funnel at 10M / 64 GiB CPU" answer different questions.

### Positioning for Scarf (use this wording)

Prefer:

> Scarf can run an end-to-end analysis funnel at multi-million cell scale on modest CPU memory against cloud Zarr, with measured wall time and peaks through 10M.

Avoid:

> Scarf is the only / most scalable single-cell package.

Market pull for that narrower job is real (atlas-sized studies, shared CPU nodes, cloud object storage) alongside GPU-first and sketch/disk-backed R workflows that serve other needs well.

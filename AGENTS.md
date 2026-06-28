# AGENTS.md

## Cursor Cloud specific instructions

Scarf is a single Python library (no long-running services, no database, no web server) for memory-efficient single-cell genomics analysis. "Running the app" means importing `scarf` and exercising the analysis pipeline. This section targets the `v1_prep` branch, which uses `uv` + Python 3.14 (master still uses the legacy pip/`requirements*.txt` flow).

### Environment basics
- Dependency manager is `uv` (lockfile `uv.lock`, deps declared in `pyproject.toml`). The startup update script runs `uv sync --extra extra --extra test`, which creates `.venv/` with the core, `extra`, `test`, and `dev` (mypy/ruff) groups. Prefix commands with `uv run`.
- `uv` is installed at `~/.local/bin` and added to PATH via `~/.bashrc`. Non-interactive shells may not pick this up; use the full path `~/.local/bin/uv` if `uv` is not found.
- Native deps (`hnswlib`, `pcst-fast`, `scikit-network`) compile from source. The system `c++`/`cc` alternatives are pinned to GNU `g++`/`gcc`; clang is also installed but cannot find libstdc++ headers and breaks these builds. Do not switch the alternatives back to clang. If a rebuild ever fails with `'string' file not found`, set `CC=gcc CXX=g++` for that build.

### Test data (not part of dependency install)
- Tests need fixture datasets that are downloaded over the network, so they are not in the update script. Fetch once per fresh VM with: `uv run python -m tests.download_fixtures --with-h5ad` (writes to `tests/datasets/`). `--with-h5ad` also pulls a dataset from OSF; omit it to skip the OSF download.

### Test / lint / run
- Tests: `uv run pytest` (config in `pyproject.toml` runs in parallel via `-n auto` and excludes `-m 'not integration'`). `integration`-marked tests hit live network (OSF, Cloudflare R2) and are skipped by default; R2 tests also need the `SCARF_R2_*` credentials from `tests/.env`.
- Lint: `uv run ruff check scarf`, `uv run ruff format --check scarf`, `uv run mypy scarf`. CI itself only runs pytest.
- "Run" the library: there is no server. Build a `DataStore` from a reader/zarr and call `auto_filter_cells` -> `mark_hvgs` -> `make_graph` -> `run_leiden_clustering` -> `run_umap`. See `tests/fixtures_datastore.py` for the canonical workflow.

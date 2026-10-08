"""Analyze a small synthetic dataset with an installed Scarf wheel.

``tests/smoke_wheel.py`` runs this script in isolated mode with the interpreter
of a clean environment that holds only the wheel and its dependencies. The
argument states whether that environment installed the ``tsne`` extra, so a
t-SNE backend that is missing where it is expected fails the smoke instead of
being skipped, and one that is present where it is not expected fails too.
The environment has no ``anndata``, so the run's H5AD exports show that they
need none.
"""

import argparse
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

_GROUPS = 3
_CELLS_PER_GROUP = 100
_GENES = 200
_MARKERS_PER_GROUP = 30
_MITOCHONDRIAL = ("MT-CO1", "MT-CO2", "MT-ND1", "MT-ND4", "MT-ATP6")
_RIBOSOMAL = ("RPL3", "RPL11", "RPL13", "RPS3", "RPS6", "RPS18")
_TSNE_SETTINGS = {"max_iter": 100, "early_iter": 25}


def synthetic_counts(seed: int = 7) -> tuple[csr_matrix, list[str], list[str]]:
    """Return counts of three cell populations, each with its own marker genes.

    Every gene has a Poisson background, the mitochondrial and ribosomal genes
    a higher one, and each population 30 marker genes with extra counts.
    """
    rng = np.random.default_rng(seed)
    n_cells = _GROUPS * _CELLS_PER_GROUP
    genes = [*_MITOCHONDRIAL, *_RIBOSOMAL]
    genes += [f"GENE{index}" for index in range(_GENES - len(genes))]
    counts = rng.poisson(1.0, size=(n_cells, _GENES))
    n_mito = len(_MITOCHONDRIAL)
    n_ribo = len(_RIBOSOMAL)
    counts[:, :n_mito] += rng.poisson(1.0, size=(n_cells, n_mito))
    counts[:, n_mito : n_mito + n_ribo] += rng.poisson(2.0, size=(n_cells, n_ribo))
    first_marker = n_mito + n_ribo
    for group in range(_GROUPS):
        rows = slice(group * _CELLS_PER_GROUP, (group + 1) * _CELLS_PER_GROUP)
        start = first_marker + group * _MARKERS_PER_GROUP
        counts[rows, start : start + _MARKERS_PER_GROUP] += rng.poisson(
            6.0, size=(_CELLS_PER_GROUP, _MARKERS_PER_GROUP)
        )
    cells = [f"cell{index}" for index in range(n_cells)]
    return csr_matrix(counts.astype(np.uint32)), cells, genes


def _coordinates(run: Any, prefix: str) -> np.ndarray:
    values = np.column_stack(
        [
            np.asarray(run.cells.fetch(f"{prefix}_{axis}"), dtype=float)
            for axis in (1, 2)
        ]
    )
    assert values.ndim == 2 and values.shape[1] == 2, (prefix, values.shape)
    assert np.isfinite(values).all(), f"{prefix} coordinates are not finite"
    assert np.all(np.ptp(values, axis=0) > 0), f"{prefix} coordinates collapsed"
    return values


def _check_run_export(path: Path, run: Any, matrix: str, counts: np.dtype) -> None:
    """Read a run's H5AD export back with h5py, which needs no AnnData."""
    import h5py

    cells = np.asarray(run.cells.fetch("ids")).astype(str)
    universe = np.asarray(run.features.fetch("ids")).astype(str)
    hvg = np.asarray(run.features.fetch("highly_variable_features"), dtype=bool)
    features = universe if matrix == "raw" else universe[hvg]
    with h5py.File(path, "r") as h5:
        shape = tuple(int(value) for value in h5["X"].attrs["shape"])
        assert shape == (len(cells), len(features)), (matrix, shape)
        assert h5["obs/ids"].asstr()[:].tolist() == cells.tolist(), matrix
        assert h5["var/gene_ids"].asstr()[:].tolist() == features.tolist(), matrix
        assert h5["obsm/X_umap"].shape == (len(cells), 2), matrix
        assert int(h5["X/indptr"][-1]) == h5["X/data"].shape[0], matrix
        values = h5["X/data"][:]
    expected = np.dtype(np.float32) if matrix == "normed" else counts
    assert values.dtype == expected, (matrix, values.dtype)
    assert values.size and np.isfinite(values).all(), matrix


def run_workflow(directory: Path, *, expect_tsne: bool) -> str:
    """Import, analyze, and check a synthetic store; return a summary line."""
    from scarf import DataStore
    from scarf.writers import SparseToZarr, to_h5ad

    # A wheel must never install an sgtsne executable, and the backend is
    # present exactly when the environment installed the tsne extra. Only the
    # environment's own scripts are checked; PATH may hold unrelated software.
    scripts = str(Path(sys.executable).parent)
    assert shutil.which("sgtsne", path=scripts) is None, scripts
    has_backend = importlib.util.find_spec("sgtsnepi") is not None
    assert has_backend is expect_tsne, f"sgtsnepi installed: {has_backend}"

    counts, cells, genes = synthetic_counts()
    location = str(directory / "smoke.zarr")
    SparseToZarr(counts, location, cells, genes, nthreads=1).dump()
    store = DataStore(location, default_assay="RNA", nthreads=1)
    run = store.pipeline.run(
        hvg_count=100,
        pca_dims=10,
        neighbors_k=10,
        cell_cycle=False,
        paris=False,
        doublets=False,
        markers=False,
        params={"tsne": dict(_TSNE_SETTINGS)} if expect_tsne else None,
    )
    assert run.status == "completed", run.status
    umap = _coordinates(run, "umap")
    clusters = np.unique(run.cells.fetch("clusters"))
    assert len(clusters) >= 2, f"Leiden found {len(clusters)} clusters, not 2 or more"
    if expect_tsne:
        tsne = _coordinates(run, "tsne")
        assert tsne.shape == umap.shape, (tsne.shape, umap.shape)
        outcome = "t-SNE computed"
    else:
        assert "tsne" not in run
        try:
            store.run_tsne(
                run["connectivity_map"],
                run["embedding_initialization"],
                verbose=False,
                **_TSNE_SETTINGS,
            )
        except ImportError as error:
            message = str(error)
            assert "scarf[tsne]" in message, message
            assert "Linux x86_64" in message, message
        else:
            raise AssertionError("run_tsne computed t-SNE without sgtsnepi")
        outcome = "t-SNE raised the sgtsnepi installation guidance"
    for matrix in ("raw", "normed"):
        exported = directory / f"smoke_{matrix}.h5ad"
        to_h5ad(store.RNA, str(exported), run=run, matrix=matrix)
        _check_run_export(exported, run, matrix, store.RNA.rawData.dtype)
    return (
        f"Workflow smoke passed: {len(umap)} cells, {len(clusters)} Leiden "
        f"clusters, {outcome}, raw and normalized H5AD exports"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tsne",
        choices=("with-tsne", "without-tsne"),
        help="whether the environment installed Scarf's tsne extra",
    )
    args = parser.parse_args()
    # Windows cannot delete a file that is still open, and the cleanup is not
    # part of what the smoke checks.
    with tempfile.TemporaryDirectory(
        prefix="scarf-smoke-workflow-", ignore_cleanup_errors=True
    ) as directory:
        summary = run_workflow(Path(directory), expect_tsne=args.tsne == "with-tsne")
    print(summary)


if __name__ == "__main__":
    main()

"""Run-backed embedding plots on a real pipeline run.

Run plots read the run's frozen cell fields, its outputs, and, with an explicit
normalization, live gene values over the run's cells.
"""

from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from matplotlib.colors import to_rgba

import scarf.plotting as splt
from scarf import DataStore
from tests.storage_helpers import write_count_store

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def plotted_run(tmp_path_factory) -> SimpleNamespace:
    """A run with a UMAP, one Leiden partition, and two frozen live columns."""
    rng = np.random.default_rng(5)
    n_types, per_type, n_genes = 3, 40, 60
    types = np.repeat(np.arange(n_types), per_type)
    counts = rng.poisson(1.0, size=(len(types), n_genes)).astype(np.uint32)
    for index in range(n_types):
        block = slice(index * 15, (index + 1) * 15)
        counts[types == index, block] += rng.poisson(8.0, size=(per_type, 15)).astype(
            np.uint32
        )
    location = str(tmp_path_factory.mktemp("run_plotting") / "data.zarr")
    write_count_store(location, {"RNA": counts}, np.uint32)
    store = DataStore(location, default_assay="RNA", min_features_per_cell=-1)
    donor = np.where(np.arange(len(types)) % 2, "d1", "d2").astype(object)
    keep = np.arange(len(types)) % 3 != 0
    store.cells.insert("donor", donor)
    store.cells.insert("keep", keep)
    run = store.pipeline.run(
        label="plots",
        filtering=False,
        cell_cycle=False,
        hvg_count=40,
        pca_dims=5,
        neighbors_k=8,
        leiden={"partitions": [1.0]},
        paris=False,
        doublets=False,
        markers=False,
        snapshot_columns=("donor", "keep"),
        params={"umap": {"n_epochs": 30}},
    )
    return SimpleNamespace(store=store, run=run)


def _coordinates(run) -> np.ndarray:
    return np.column_stack([run.cells.fetch("umap_1"), run.cells.fetch("umap_2")])


def _facecolors(result, panel) -> np.ndarray:
    (points,) = result.axes[panel].collections
    return np.asarray(points.get_facecolors())


def test_run_embedding_draws_several_frozen_colors_in_facets_and_subsets(
    plotted_run,
):
    store, run = plotted_run.store, plotted_run.run
    coordinates = _coordinates(run)
    donor = run.cells.fetch("donor")
    keep = run.cells.fetch("keep")
    clusters = run.cells.fetch("clusters")

    result = store.plots.embedding(
        run=run,
        color_by=["clusters", splt.CellField("donor", kind="categorical")],
        facet_by="donor",
        subset_by="keep",
        show=False,
    )
    try:
        assert list(result.axes) == [
            ("clusters", "d1"),
            ("clusters", "d2"),
            ("donor", "d1"),
            ("donor", "d2"),
        ]
        scales = [scale for scale in result.scales if hasattr(scale, "palette")]
        cluster_scale, donor_scale = scales
        for facet in ("d1", "d2"):
            drawn = keep & (donor == facet)
            (points,) = result.axes[("clusters", facet)].collections
            np.testing.assert_allclose(points.get_offsets(), coordinates[drawn])
            np.testing.assert_allclose(
                _facecolors(result, ("clusters", facet)),
                [to_rgba(cluster_scale.palette[label]) for label in clusters[drawn]],
            )
            np.testing.assert_allclose(
                _facecolors(result, ("donor", facet)),
                [to_rgba(donor_scale.palette[facet])] * int(drawn.sum()),
            )
        assert result.provenance.n_cells == int(keep.sum())
        extras = result.provenance.extras
        assert (extras["facet_by"], extras["subset_by"]) == ("donor", "keep")
    finally:
        result.close()


def test_run_embedding_colors_by_run_outputs(plotted_run):
    store, run = plotted_run.store, plotted_run.run

    result = store.plots.embedding(
        run=run, color_by=[run["clusters"], run["leiden_1.0"]], show=False
    )
    reference = store.plots.embedding(
        layout=run["umap"], color_by=run["clusters"], show=False
    )
    try:
        assert result.provenance.extras["color_artifacts"] == [
            run["clusters"].to_dict(),
            run["leiden_1.0"].to_dict(),
        ]
        # A run output colors the run plot as it colors the ref-mode plot.
        assert list(result.axes) == [(0, "cluster_labels"), (1, "cluster_labels")]
        np.testing.assert_allclose(
            _facecolors(result, (0, "cluster_labels")),
            _facecolors(reference, "cluster_labels"),
        )
    finally:
        result.close()
        reference.close()


def test_run_embedding_colors_genes_with_an_explicit_normalization(plotted_run):
    store, run = plotted_run.store, plotted_run.run
    gene = "RNA3"
    normalization = splt.NormalizationSpec(transform="log1p")
    result = store.plots.embedding(
        run=run,
        color_by=[gene, "clusters"],
        normalization=normalization,
        show=False,
    )
    reference = store.plots.embedding(
        layout=run["umap"],
        color_by=gene,
        normalization=normalization,
        show=False,
    )
    try:
        # Gene values are the live normalized counts of the run's cells.
        cells = np.flatnonzero(run.cells.fetch_all("I"))
        expected = np.asarray(
            store.RNA.normed(
                cell_idx=cells, feat_idx=np.asarray([3]), log_transform=True
            ).compute(),
            dtype=np.float64,
        )[:, 0]
        limits = result.provenance.extras["color_limits"][gene]
        assert limits == pytest.approx((expected.min(), expected.max()))
        np.testing.assert_allclose(
            _facecolors(result, gene), _facecolors(reference, gene)
        )
        extras = result.provenance.extras
        assert extras["normalization"] == {"source": "assay", "transform": "log1p"}
        assert extras["assays"] == ["RNA"]
        assert extras["run"] == {"runId": run.run_id, "label": "plots"}
    finally:
        result.close()
        reference.close()

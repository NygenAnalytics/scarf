"""Guarded-source and parity tests for Milestone C raster path."""

import numpy as np
import pytest

import scarf.plotting as splt


class _GuardedZarrArray:
    """Minimal array stand-in that forbids full-column ``[:]`` reads."""

    def __init__(self, data: np.ndarray, chunk: int = 8):
        self._data = np.asarray(data)
        self.chunks = (chunk,)
        self.dtype = self._data.dtype
        self.shape = self._data.shape
        self.metadata = type("Metadata", (), {"shards": None})()

    def __getitem__(self, key):
        if key == slice(None) or key == ():
            raise AssertionError(
                "full-column read is forbidden in guarded raster tests"
            )
        if isinstance(key, slice):
            if key.start is None and key.stop is None and key.step is None:
                raise AssertionError("full-column read is forbidden")
            return self._data[key]
        return self._data[key]


class _GuardedMeta:
    """Duck-typed MetaData using guarded column arrays."""

    def __init__(
        self,
        columns: dict[str, np.ndarray],
        chunk: int = 8,
        missing: dict[str, np.ndarray] | None = None,
    ):
        self.N = len(next(iter(columns.values())))
        self.columns = list(columns)
        self._arrays = {
            k: _GuardedZarrArray(v, chunk=chunk) for k, v in columns.items()
        }
        self._missing = {
            k: _GuardedZarrArray(v, chunk=chunk) for k, v in (missing or {}).items()
        }
        self.index = np.arange(self.N)

    def _get_array(self, column: str):
        return self._arrays[column]

    def _get_missing_mask_array(self, column: str):
        return self._missing.get(column)

    def _verify_bool(self, key: str) -> bool:
        if self._arrays[key].dtype != bool:
            raise TypeError("key must be bool")
        return True

    def get_dtype(self, column: str) -> np.dtype:
        return self._arrays[column].dtype

    def default_block_rows(self, column: str = "I") -> int:
        return int(self._arrays[column].chunks[0])

    def active_index(self, key: str) -> np.ndarray:
        return np.flatnonzero(self._arrays[key]._data)

    def fetch(self, column: str, key: str = "I") -> np.ndarray:
        idx = self.active_index(key)
        return self._arrays[column]._data[idx]

    def iter_row_blocks(self, *, cell_key="I", columns=None, block_rows=None):
        from scarf.metadata import MetaDataRowBlock

        self._verify_bool(cell_key)
        if block_rows is None:
            block_rows = self.default_block_rows(cell_key)
        col_list = list(columns or [])
        key_arr = self._arrays[cell_key]
        for start in range(0, self.N, block_rows):
            stop = min(start + block_rows, self.N)
            key_slice = np.asarray(key_arr[start:stop], dtype=bool)
            local = np.flatnonzero(key_slice)
            active_global = (local + start).astype(np.int64, copy=False)
            values = {
                col: np.asarray(self._arrays[col][start:stop])[local]
                for col in col_list
            }
            yield MetaDataRowBlock(
                start=start,
                stop=stop,
                active_global_indices=active_global,
                values=values,
            )


def test_raster_from_metadata_rejects_full_slice_and_matches_mean():
    from scarf.plotting._raster import raster_from_metadata

    rng = np.random.default_rng(0)
    n = 40
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    c = rng.normal(size=n)
    active = np.ones(n, dtype=bool)
    active[::5] = False
    cells = _GuardedMeta(
        {"I": active, "UMAP1": x, "UMAP2": y, "score": c},
        chunk=7,
    )
    canvas = raster_from_metadata(
        cells,
        x_key="UMAP1",
        y_key="UMAP2",
        color_key="score",
        cell_key="I",
        pixels=32,
        block_rows=7,
        quantiles=None,
        seed=0,
    )
    assert canvas.n_cells == int(active.sum())
    assert canvas.n_blocks >= 2
    assert np.isfinite(canvas.vmin) and np.isfinite(canvas.vmax)
    # Pixel means only where counts > 0
    assert np.isnan(canvas.image[canvas.counts == 0]).all()
    assert np.isfinite(canvas.image[canvas.counts > 0]).all()


def test_embedding_raster_accepts_explicit_literal_metadata_layout():
    rng = np.random.default_rng(3)
    cells = _GuardedMeta(
        {
            "I": np.ones(24, dtype=bool),
            "literal1": rng.normal(size=24),
            "literal2": rng.normal(size=24),
            "score": rng.normal(size=24),
        },
        chunk=6,
    )
    store = type("Store", (), {"cells": cells})()

    result = splt.embedding_raster(
        store,
        layout_key="literal",
        color_by="score",
        pixels=16,
        block_rows=6,
        show=False,
    )

    assert "live_metadata_layout" in result.provenance.notes
    assert result.provenance.extras["layout"] is None
    result.close()


def test_embedding_lineage_accepts_datastore_scoped_native_layout(monkeypatch):
    import importlib
    from types import SimpleNamespace

    from scarf.storage import ArtifactRef

    data_module = importlib.import_module("scarf.plotting._data")
    graph_module = importlib.import_module("scarf.graph.feature_projection")
    selection = ArtifactRef(
        scope="datastore",
        kind="cell_selection",
        artifact_id="1" * 64,
    )
    graph = ArtifactRef(
        scope="datastore",
        kind="integrated_graph",
        artifact_id="2" * 64,
    )
    layout = ArtifactRef(
        scope="datastore",
        kind="embedding",
        artifact_id="3" * 64,
    )
    status = SimpleNamespace(
        complete=True,
        operation="run_umap",
        inputs={"graph": graph.to_dict(), "cell_selection": selection.to_dict()},
    )
    monkeypatch.setattr(data_module, "inspect_artifact", lambda root, ref: status)
    monkeypatch.setattr(
        graph_module,
        "graph_cell_selection",
        lambda root, ref: selection,
    )
    store = type("Store", (), {"zw": object()})()

    assert data_module._validated_embedding_selection(store, layout) == selection


def test_embedding_lineage_rejects_unknown_producer_and_graph_selection(
    monkeypatch,
):
    import importlib
    from types import SimpleNamespace

    from scarf.storage import ArtifactRef

    data_module = importlib.import_module("scarf.plotting._data")
    graph_module = importlib.import_module("scarf.graph.feature_projection")
    direct_selection = ArtifactRef(
        scope="datastore",
        kind="cell_selection",
        artifact_id="4" * 64,
    )
    graph_selection = ArtifactRef(
        scope="datastore",
        kind="cell_selection",
        artifact_id="5" * 64,
    )
    graph = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="connectivity_map",
        artifact_id="6" * 64,
    )
    layout = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="embedding",
        artifact_id="7" * 64,
    )
    store = type("Store", (), {"zw": object()})()

    monkeypatch.setattr(
        data_module,
        "inspect_artifact",
        lambda root, ref: SimpleNamespace(
            complete=True,
            operation="foreign_embedding",
            inputs={"cell_selection": direct_selection.to_dict()},
        ),
    )
    with pytest.raises(ValueError, match="must be produced"):
        data_module._validated_embedding_selection(store, layout)

    monkeypatch.setattr(
        data_module,
        "inspect_artifact",
        lambda root, ref: SimpleNamespace(
            complete=True,
            operation="run_tsne",
            inputs={
                "graph": graph.to_dict(),
                "cell_selection": direct_selection.to_dict(),
            },
        ),
    )
    monkeypatch.setattr(
        graph_module,
        "graph_cell_selection",
        lambda root, ref: graph_selection,
    )
    with pytest.raises(ValueError, match="share the same cell selection"):
        data_module._validated_embedding_selection(store, layout)


def test_artifact_raster_adapter_reads_only_compact_coordinate_slices(monkeypatch):
    import importlib

    from scarf.storage import ArtifactRef
    from scarf.storage.selections import StoredSelectionBlock

    raster_module = importlib.import_module("scarf.plotting.embedding_raster")

    class GuardedCoordinates:
        shape = (3, 2)
        dtype = np.dtype(np.float64)

        def __init__(self) -> None:
            self.reads: list[slice] = []
            self.values = np.arange(6, dtype=np.float64).reshape(3, 2)

        def __getitem__(self, key):
            if not isinstance(key, slice) or key.start is None or key.stop is None:
                raise AssertionError("embedding coordinates must use bounded slices")
            self.reads.append(key)
            return self.values[key]

    selection = ArtifactRef(
        scope="datastore",
        kind="cell_selection",
        artifact_id="a" * 64,
    )
    selection_blocks = (
        StoredSelectionBlock(
            start=0,
            stop=3,
            mask=np.array([True, False, True]),
            selected_indices=np.array([0, 2]),
            compact_start=0,
            compact_stop=2,
        ),
        StoredSelectionBlock(
            start=3,
            stop=5,
            mask=np.array([False, True]),
            selected_indices=np.array([4]),
            compact_start=2,
            compact_stop=3,
        ),
    )

    def fake_selection_blocks(*args, **kwargs):
        del args, kwargs
        yield from selection_blocks

    monkeypatch.setattr(
        raster_module,
        "iter_stored_selection_blocks",
        fake_selection_blocks,
    )
    coordinates = GuardedCoordinates()
    cells = _GuardedMeta(
        {
            "I": np.ones(5, dtype=bool),
            "score": np.arange(5, dtype=np.int64),
            "keep": np.ones(5, dtype=bool),
        },
        chunk=2,
        missing={
            "score": np.array([False, False, True, False, False]),
            "keep": np.array([False, False, False, False, True]),
        },
    )
    view = raster_module._ArtifactRasterCells(
        object(),
        cells,
        coordinates,
        selection,
    )

    blocks = list(
        view.iter_row_blocks(
            columns=[
                raster_module._ARTIFACT_X,
                raster_module._ARTIFACT_Y,
                "score",
                "keep",
            ],
            block_rows=2,
        )
    )

    assert [(item.start, item.stop) for item in coordinates.reads] == [(0, 2), (2, 3)]
    np.testing.assert_array_equal(
        np.concatenate([block.active_global_indices for block in blocks]),
        [0, 2, 4],
    )
    np.testing.assert_equal(
        np.concatenate([block.values["score"] for block in blocks]),
        [0.0, np.nan, 4.0],
    )
    np.testing.assert_array_equal(
        np.concatenate([block.values["keep"] for block in blocks]),
        [True, True, False],
    )


def test_embedding_raster_on_datastore(umap, datastore):
    result = splt.embedding_raster(
        datastore,
        layout=umap,
        color_by="RNA_nCounts",
        pixels=64,
        block_rows=32,
        show=False,
    )
    assert result.provenance.renderer == "matplotlib-raster"
    assert "two_pass" in result.provenance.notes
    assert "artifact_layout" in result.provenance.notes
    assert "live_metadata_fields" in result.provenance.notes
    assert result.provenance.extras["layout"] == umap.to_dict()
    assert result.provenance.extras["cell_selection"] is not None
    assert result.provenance.n_cells > 0
    result.close()


def test_embedding_raster_rejects_categorical_metadata(umap, datastore):
    with pytest.raises(NotImplementedError, match="continuous color"):
        splt.embedding_raster(
            datastore,
            layout=umap,
            color_by="ids",
            pixels=32,
            show=False,
        )


def test_embedding_raster_density_and_foreign_target(umap, datastore):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    result = splt.embedding_raster(
        datastore,
        layout=umap,
        pixels=32,
        target=ax,
        show=False,
    )
    assert result.owns_figure is False
    assert result.provenance.extras["color_mode"] == "density"
    assert result.legends[0].label == "log1p cell count"
    result.close()
    assert plt.fignum_exists(fig.number)
    plt.close(fig)


def test_embedding_raster_image_fills_square_axes(umap, datastore):
    result = splt.embedding_raster(
        datastore,
        layout=umap,
        color_by="RNA_nCounts",
        pixels=48,
        show=False,
    )
    ax = next(iter(result.axes.values()))
    images = ax.get_images()
    assert len(images) == 1
    extent = images[0].get_extent()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    assert extent[0] == pytest.approx(xlim[0])
    assert extent[1] == pytest.approx(xlim[1])
    assert extent[2] == pytest.approx(ylim[0])
    assert extent[3] == pytest.approx(ylim[1])
    assert (xlim[1] - xlim[0]) == pytest.approx(ylim[1] - ylim[0])
    result.close()


def test_embedding_raster_requires_one_coordinate_source(umap, datastore):
    with pytest.raises(ValueError, match="exactly one"):
        splt.embedding_raster(datastore, show=False)
    with pytest.raises(ValueError, match="exactly one"):
        splt.embedding_raster(
            datastore,
            layout_key="literal",
            layout=umap,
            show=False,
        )
    with pytest.raises(ValueError, match="cannot override"):
        splt.embedding_raster(
            datastore,
            layout=umap,
            cell_key="another_selection",
            show=False,
        )


def test_embedding_raster_rejects_non_embedding_artifact(paris_clustering, datastore):
    with pytest.raises(ValueError, match="embedding artifact"):
        splt.embedding_raster(
            datastore,
            layout=paris_clustering,
            show=False,
        )


def test_raster_validates_quantiles():
    from scarf.plotting._raster import raster_from_metadata

    cells = _GuardedMeta(
        {
            "I": np.ones(4, dtype=bool),
            "x": np.arange(4, dtype=float),
            "y": np.arange(4, dtype=float),
            "v": np.arange(4, dtype=float),
        }
    )
    with pytest.raises(ValueError, match="quantiles"):
        raster_from_metadata(
            cells,
            x_key="x",
            y_key="y",
            color_key="v",
            quantiles=(0.9, 0.1),
        )
    with pytest.raises(ValueError, match="sample_capacity"):
        raster_from_metadata(
            cells,
            x_key="x",
            y_key="y",
            color_key="v",
            sample_capacity=0,
        )


def test_raster_without_color_encodes_log_density():
    from scarf.plotting._raster import raster_from_metadata

    cells = _GuardedMeta(
        {
            "I": np.ones(4, dtype=bool),
            "x": np.zeros(4),
            "y": np.zeros(4),
        }
    )
    canvas = raster_from_metadata(
        cells,
        x_key="x",
        y_key="y",
        pixels=8,
    )
    assert canvas.counts.sum() == 4
    assert np.nanmax(canvas.image) == pytest.approx(np.log1p(4))
    assert canvas.vmax == pytest.approx(np.log1p(4))


def test_raster_squares_extent_before_binning_without_stretching_points():
    from scarf.plotting._raster import raster_from_metadata

    cells = _GuardedMeta(
        {
            "I": np.ones(4, dtype=bool),
            "x": np.array([0.0, 0.0, 10.0, 10.0]),
            "y": np.array([0.0, 1.0, 0.0, 1.0]),
        }
    )

    canvas = raster_from_metadata(
        cells,
        x_key="x",
        y_key="y",
        pixels=100,
    )

    x_span = canvas.extent[1] - canvas.extent[0]
    y_span = canvas.extent[3] - canvas.extent[2]
    assert x_span == pytest.approx(y_span)
    occupied_rows, occupied_columns = np.nonzero(canvas.counts)
    assert np.ptp(occupied_rows) < np.ptp(occupied_columns) / 5


def test_density_canvas_from_points_uses_raster_contract():
    from scarf.plotting._raster import density_canvas_from_points

    canvas = density_canvas_from_points(
        np.array([0.1, 0.1, 0.8]),
        np.array([0.1, 0.1, 0.8]),
        extent=(0, 1, 0, 1),
        pixels=10,
    )

    assert canvas.counts.sum() == 3
    assert canvas.counts.max() == 2
    assert canvas.n_cells == 3
    assert canvas.extent == (0, 1, 0, 1)
    low_y_row = int(np.flatnonzero(canvas.counts[:, 1])[0])
    high_y_row = int(np.flatnonzero(canvas.counts[:, 8])[0])
    assert high_y_row < low_y_row


def test_raster_is_block_size_invariant():
    from scarf.plotting._raster import raster_from_metadata

    rng = np.random.default_rng(4)
    cells = _GuardedMeta(
        {
            "I": np.ones(60, dtype=bool),
            "x": rng.normal(size=60),
            "y": rng.normal(size=60),
            "value": rng.normal(size=60),
        }
    )
    kwargs = {
        "x_key": "x",
        "y_key": "y",
        "color_key": "value",
        "pixels": 24,
        "sample_capacity": 15,
        "seed": 9,
    }
    first = raster_from_metadata(cells, block_rows=7, **kwargs)
    second = raster_from_metadata(cells, block_rows=13, **kwargs)
    assert first.extent == pytest.approx(second.extent)
    assert first.vmin == pytest.approx(second.vmin)
    assert first.vmax == pytest.approx(second.vmax)
    np.testing.assert_allclose(first.image, second.image, equal_nan=True)
    np.testing.assert_array_equal(first.counts, second.counts)


def test_raster_all_missing_color_has_finite_default_limits():
    from scarf.plotting._raster import raster_from_metadata

    cells = _GuardedMeta(
        {
            "I": np.ones(4, dtype=bool),
            "x": np.arange(4, dtype=float),
            "y": np.arange(4, dtype=float),
            "value": np.full(4, np.nan),
        }
    )
    canvas = raster_from_metadata(
        cells,
        x_key="x",
        y_key="y",
        color_key="value",
        pixels=8,
    )
    assert (canvas.vmin, canvas.vmax) == (0.0, 1.0)
    assert np.isnan(canvas.image).all()


def test_raster_honors_nullable_color_and_subset_masks():
    from scarf.plotting._raster import raster_from_metadata

    cells = _GuardedMeta(
        {
            "I": np.ones(3, dtype=bool),
            "x": np.zeros(3),
            "y": np.zeros(3),
            "value": np.array([1, 3, 0], dtype=np.int64),
            "keep": np.ones(3, dtype=bool),
        },
        missing={
            "value": np.array([False, False, True]),
            "keep": np.array([False, True, False]),
        },
    )

    colored = raster_from_metadata(
        cells,
        x_key="x",
        y_key="y",
        color_key="value",
        pixels=8,
        quantiles=None,
    )
    assert colored.counts.sum() == 2
    assert colored.image[colored.counts > 0].item() == pytest.approx(2.0)

    subset = raster_from_metadata(
        cells,
        x_key="x",
        y_key="y",
        subset_by="keep",
        pixels=8,
    )
    assert subset.n_cells == 2


def test_embedding_raster_uses_one_effective_missing_color(umap, datastore):
    from scarf.plotting import ColorScale

    result = splt.embedding_raster(
        datastore,
        layout=umap,
        color_scale=ColorScale(missing_color="pink"),
        missing_color="white",
        pixels=32,
        show=False,
    )

    assert result.scales[0].missing_color == "white"
    assert result.provenance.extras["missing_color"] == "white"
    result.close()


def test_embedding_raster_uses_frozen_continuous_display_defaults():
    cells = _GuardedMeta(
        {
            "I": np.ones(4, dtype=bool),
            "layout1": np.arange(4, dtype=float),
            "layout2": np.arange(4, dtype=float),
            "score": np.arange(4, dtype=np.int32),
        }
    )

    class Store:
        def __init__(self) -> None:
            self.cells = cells

        def _stored_display_metadata(self, column: str):
            assert column == "score"
            return {
                "kind": "continuous",
                "colormap": "magma",
                "minimum": 0.0,
                "maximum": 3.0,
                "scale": "linear",
            }

    result = splt.embedding_raster(
        Store(),
        layout_key="layout",
        color_by="score",
        pixels=16,
        show=False,
    )

    scale = result.scales[0]
    assert scale.cmap == "magma"
    assert scale.vmin == 0.0
    assert scale.vmax == 3.0
    assert scale.quantiles is None
    assert "approximate_quantiles" not in result.provenance.notes
    result.close()


def test_raster_missing_pixels_default_white():
    from scarf.plotting._raster import RasterCanvas, draw_raster_canvas

    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    canvas = RasterCanvas(
        image=np.full((8, 8), np.nan, dtype=np.float64),
        counts=np.zeros((8, 8), dtype=np.int64),
        extent=(0.0, 1.0, 0.0, 1.0),
        vmin=0.0,
        vmax=1.0,
        n_cells=0,
        n_blocks=0,
    )
    fig, ax = plt.subplots()
    im = draw_raster_canvas(ax, canvas)
    bad = np.asarray(im.cmap.get_bad())
    # RGBA for the bad/missing color should be opaque white.
    np.testing.assert_allclose(bad[:3], [1.0, 1.0, 1.0], atol=1e-5)
    face = np.asarray(matplotlib.colors.to_rgba(ax.get_facecolor()))
    np.testing.assert_allclose(face[:3], [1.0, 1.0, 1.0], atol=1e-5)
    plt.close(fig)


def test_raster_subset_by_reduces_cells():
    from scarf.plotting._raster import raster_from_metadata

    n = 20
    keep = np.zeros(n, dtype=bool)
    keep[:5] = True
    cells = _GuardedMeta(
        {
            "I": np.ones(n, dtype=bool),
            "x": np.linspace(0, 1, n),
            "y": np.linspace(0, 1, n),
            "value": np.arange(n, dtype=float),
            "keep": keep,
        }
    )
    full = raster_from_metadata(
        cells, x_key="x", y_key="y", color_key="value", pixels=16
    )
    subset = raster_from_metadata(
        cells,
        x_key="x",
        y_key="y",
        color_key="value",
        subset_by="keep",
        pixels=16,
    )
    assert subset.n_cells == 5
    assert subset.n_cells < full.n_cells


def test_specialized_plotting_exports_are_callable():
    assert callable(splt.qc)
    assert callable(splt.elbow)
    assert callable(splt.graph_qc)
    assert callable(splt.highly_variable_features)
    assert callable(splt.marker_heatmap)
    assert callable(splt.cluster_tree)
    assert callable(splt.pseudotime_heatmap)

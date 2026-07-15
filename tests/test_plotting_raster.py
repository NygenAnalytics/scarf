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

    def __init__(self, columns: dict[str, np.ndarray], chunk: int = 8):
        self.N = len(next(iter(columns.values())))
        self.columns = list(columns)
        self._arrays = {
            k: _GuardedZarrArray(v, chunk=chunk) for k, v in columns.items()
        }
        self.index = np.arange(self.N)

    def _verify_bool(self, key: str) -> bool:
        if self._arrays[key].dtype != bool:
            raise TypeError("key must be bool")
        return True

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


def test_embedding_raster_on_datastore(umap, leiden_clustering, datastore):
    result = splt.embedding_raster(
        datastore,
        layout_key="RNA_UMAP",
        color_by="RNA_nCounts",
        pixels=64,
        block_rows=32,
        show=False,
    )
    assert result.provenance.renderer == "matplotlib-raster"
    assert "two_pass" in result.provenance.notes
    assert result.provenance.n_cells > 0
    result.close()


def test_embedding_raster_rejects_categorical_metadata(
    umap, leiden_clustering, datastore
):
    with pytest.raises(NotImplementedError, match="continuous color"):
        splt.embedding_raster(
            datastore,
            layout_key="RNA_UMAP",
            color_by="RNA_leiden_cluster",
            pixels=32,
            show=False,
        )


def test_embedding_raster_density_and_foreign_target(umap, datastore):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    result = splt.embedding_raster(
        datastore,
        layout_key="RNA_UMAP",
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


def test_specialized_facades_are_callable():
    assert callable(splt.qc)
    assert callable(splt.elbow)
    assert callable(splt.graph_qc)
    assert callable(splt.highly_variable_features)
    assert callable(splt.marker_heatmap)
    assert callable(splt.cluster_tree)
    assert callable(splt.pseudotime_heatmap)

"""Guarded-source and parity tests for the blockwise raster path."""

from importlib import import_module
from types import SimpleNamespace

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
        # Compare only slices and tuples, so array (fancy) keys still work.
        full_slice = isinstance(key, slice) and key == slice(None)
        if full_slice or (isinstance(key, tuple) and key == ()):
            raise AssertionError(
                "full-column read is forbidden in guarded raster tests"
            )
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


raster_module = import_module("scarf.plotting.embedding_raster")


def _expected_extent(x, y):
    """Pad each axis by 1% of its span, then square the window around it."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    pad_x = 0.01 * np.ptp(x)
    pad_y = 0.01 * np.ptp(y)
    x0, x1 = x.min() - pad_x, x.max() + pad_x
    y0, y1 = y.min() - pad_y, y.max() + pad_y
    half = max(x1 - x0, y1 - y0) / 2
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return (cx - half, cx + half, cy - half, cy + half)


def _grid(x, y, extent, pixels, weights=None):
    """Bin points on the raster grid, with image row 0 at the top."""
    binned, _, _ = np.histogram2d(
        x,
        y,
        bins=pixels,
        range=((extent[0], extent[1]), (extent[2], extent[3])),
        weights=weights,
    )
    return np.flipud(binned.T)


def _layout_store(columns, chunk=8, missing=None, **attributes):
    """A live-metadata store whose columns forbid full-column reads."""
    return SimpleNamespace(
        cells=_GuardedMeta(columns, chunk=chunk, missing=missing),
        **attributes,
    )


def _layout_columns(n=12):
    rng = np.random.default_rng(11)
    return {
        "I": np.ones(n, dtype=bool),
        "U1": rng.normal(size=n),
        "U2": rng.normal(size=n),
        "score": rng.normal(size=n),
    }


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

    xs, ys, cs = x[active], y[active], c[active]
    extent = _expected_extent(xs, ys)
    counts = _grid(xs, ys, extent, 32)
    sums = _grid(xs, ys, extent, 32, weights=cs)
    assert canvas.n_cells == int(active.sum())
    assert canvas.n_blocks == 6
    assert canvas.extent == pytest.approx(extent)
    np.testing.assert_array_equal(canvas.counts, counts)
    np.testing.assert_allclose(
        canvas.image,
        np.where(counts > 0, sums / np.maximum(counts, 1), np.nan),
        equal_nan=True,
        atol=1e-12,
    )
    assert (canvas.vmin, canvas.vmax) == (cs.min(), cs.max())


def test_embedding_raster_accepts_explicit_literal_metadata_layout():
    rng = np.random.default_rng(3)
    literal1 = rng.normal(size=24)
    literal2 = rng.normal(size=24)
    score = rng.normal(size=24)
    store = _layout_store(
        {
            "I": np.ones(24, dtype=bool),
            "literal1": literal1,
            "literal2": literal2,
            "score": score,
        },
        chunk=6,
    )

    result = splt.embedding_raster(
        store,
        layout_key="literal",
        color_by="score",
        pixels=16,
        block_rows=6,
        show=False,
    )

    assert result.provenance.notes == (
        "embedding_raster",
        "two_pass",
        "live_metadata_layout",
        "live_metadata_fields",
        "approximate_quantiles",
    )
    assert result.provenance.extras["layout"] is None
    assert result.provenance.n_cells == 24
    assert result.provenance.extras["n_blocks"] == 4
    ax = result.axes["score"]
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("literal1", "literal2")
    image = ax.get_images()[0]
    assert image.get_extent() == pytest.approx(_expected_extent(literal1, literal2))
    # The 50k-value sample keeps all 24 scores, so the quantiles are exact.
    expected_limits = tuple(np.quantile(score, [0.01, 0.99]))
    assert image.get_clim() == pytest.approx(expected_limits)
    assert (
        result.legends[0].extras["vmin"],
        result.legends[0].extras["vmax"],
    ) == pytest.approx(expected_limits)
    result.close()


def _embedding_status(layout, operation: str, inputs: dict):
    from scarf.storage import ArtifactStatus

    return ArtifactStatus(
        ref=layout,
        path="embedding",
        exists=True,
        complete=True,
        provenance={"operation": operation, "parameters": {}, "inputs": inputs},
    )


def test_embedding_lineage_accepts_datastore_scoped_native_layout(monkeypatch):
    from scarf.storage import ArtifactRef

    data_module = import_module("scarf.plotting._data")
    graph_module = import_module("scarf.graph.feature_projection")
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
    status = _embedding_status(
        layout,
        "run_umap",
        {"graph": graph.to_dict(), "cell_selection": selection.to_dict()},
    )
    inspected = []
    graphs = []
    monkeypatch.setattr(
        data_module,
        "inspect_artifact",
        lambda root, ref: inspected.append(ref) or status,
    )
    monkeypatch.setattr(
        graph_module,
        "graph_cell_selection",
        lambda root, ref: graphs.append(ref) or selection,
    )
    store = SimpleNamespace(zw=object())

    assert data_module._validated_embedding_selection(store, layout) == selection
    assert inspected == [layout]
    assert graphs == [graph]


def test_embedding_lineage_rejects_unknown_producer_and_graph_selection(
    monkeypatch,
):
    from scarf.storage import ArtifactRef

    data_module = import_module("scarf.plotting._data")
    graph_module = import_module("scarf.graph.feature_projection")
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
    store = SimpleNamespace(zw=object())

    monkeypatch.setattr(
        data_module,
        "inspect_artifact",
        lambda root, ref: _embedding_status(
            layout,
            "foreign_embedding",
            {"cell_selection": direct_selection.to_dict()},
        ),
    )
    with pytest.raises(ValueError, match="must be produced"):
        data_module._validated_embedding_selection(store, layout)

    monkeypatch.setattr(
        data_module,
        "inspect_artifact",
        lambda root, ref: _embedding_status(
            layout,
            "run_tsne",
            {"graph": graph.to_dict(), "cell_selection": direct_selection.to_dict()},
        ),
    )
    monkeypatch.setattr(
        graph_module,
        "graph_cell_selection",
        lambda root, ref: graph_selection,
    )
    with pytest.raises(ValueError, match="share the same cell selection"):
        data_module._validated_embedding_selection(store, layout)


class _GuardedCoordinates:
    """Embedding coordinates that only answer bounded compact-row slices."""

    def __init__(self, values) -> None:
        self.values = np.asarray(values)
        self.shape = self.values.shape
        self.dtype = self.values.dtype
        self.reads: list[slice] = []

    def __getitem__(self, key):
        if not isinstance(key, slice) or key.start is None or key.stop is None:
            raise AssertionError("embedding coordinates must use bounded slices")
        self.reads.append(key)
        return self.values[key]


def _raster_selection():
    from scarf.storage import ArtifactRef

    return ArtifactRef(
        scope="datastore",
        kind="cell_selection",
        artifact_id="a" * 64,
    )


def _two_selection_blocks():
    """Stored selection blocks selecting rows 0, 2 and 4 of five cells."""
    from scarf.storage.selections import StoredSelectionBlock

    return (
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


def _patch_selection_blocks(monkeypatch, blocks):
    def fake_selection_blocks(*args, **kwargs):
        del args, kwargs
        yield from blocks

    monkeypatch.setattr(
        raster_module,
        "iter_stored_selection_blocks",
        fake_selection_blocks,
    )


def test_artifact_raster_adapter_reads_only_compact_coordinate_slices(monkeypatch):
    _patch_selection_blocks(monkeypatch, _two_selection_blocks())
    coordinates = _GuardedCoordinates(np.arange(6, dtype=np.float64).reshape(3, 2))
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
        _raster_selection(),
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
    assert [(block.start, block.stop) for block in blocks] == [(0, 3), (3, 5)]
    np.testing.assert_array_equal(
        np.concatenate([block.active_global_indices for block in blocks]),
        [0, 2, 4],
    )
    # Compact coordinate rows map onto the selected cells in order.
    np.testing.assert_array_equal(
        np.concatenate([block.values[raster_module._ARTIFACT_X] for block in blocks]),
        [0.0, 2.0, 4.0],
    )
    np.testing.assert_array_equal(
        np.concatenate([block.values[raster_module._ARTIFACT_Y] for block in blocks]),
        [1.0, 3.0, 5.0],
    )
    # Blocks carry stored values; the raster reader applies each live mask once.
    np.testing.assert_array_equal(
        np.concatenate([block.values["score"] for block in blocks]),
        [0, 2, 4],
    )
    assert view._get_missing_mask_array(raster_module._ARTIFACT_X) is None
    from scarf.plotting._raster import _MissingMaskRows, _raster_block_values

    masks = _MissingMaskRows(view)
    np.testing.assert_equal(
        np.concatenate(
            [_raster_block_values(masks, block, "score") for block in blocks]
        ),
        [0.0, np.nan, 4.0],
    )
    np.testing.assert_array_equal(
        np.concatenate(
            [_raster_block_values(masks, block, "keep") for block in blocks]
        ),
        [True, True, False],
    )


def test_artifact_raster_adapter_reports_dtypes_and_rejects_invalid_reads(
    monkeypatch,
):
    _patch_selection_blocks(monkeypatch, _two_selection_blocks())
    selection = _raster_selection()
    coordinates = np.arange(6, dtype=np.float32).reshape(3, 2)
    live = _GuardedMeta({"I": np.ones(5, dtype=bool), "score": np.arange(5)})
    view = raster_module._ArtifactRasterCells(object(), live, coordinates, selection)

    assert view.columns == [
        "I",
        "score",
        raster_module._ARTIFACT_X,
        raster_module._ARTIFACT_Y,
    ]
    assert view.get_dtype(raster_module._ARTIFACT_X) == np.dtype(np.float32)
    assert view.get_dtype("score") == live.get_dtype("score")

    class FrozenFields:
        columns = ("phase",)

        def _field_dtype(self, column):
            assert column == "phase"
            return np.dtype("U2")

    frozen = raster_module._ArtifactRasterCells(
        object(), FrozenFields(), coordinates, selection
    )
    assert frozen.get_dtype("phase") == np.dtype("U2")
    silent = raster_module._ArtifactRasterCells(
        object(), SimpleNamespace(columns=("phase",)), coordinates, selection
    )
    with pytest.raises(
        TypeError, match="^Raster cell view cannot report field dtypes$"
    ):
        silent.get_dtype("phase")

    with pytest.raises(ValueError, match="^cell_key cannot override"):
        next(view.iter_row_blocks(cell_key="filtered", columns=["score"]))
    with pytest.raises(KeyError, match=r"Raster fields were not found: \['nope'\]"):
        next(view.iter_row_blocks(columns=["score", "nope"]))
    with pytest.raises(ValueError, match="^block_rows must be >= 1$"):
        next(view.iter_row_blocks(columns=["score"], block_rows=0))

    short = raster_module._ArtifactRasterCells(
        object(), live, coordinates[:2], selection
    )
    with pytest.raises(
        ValueError,
        match="^Embedding coordinates changed while they were being read$",
    ):
        list(short.iter_row_blocks(columns=[raster_module._ARTIFACT_X]))
    unfinished = coordinates.copy()
    unfinished[1, 0] = np.nan
    broken = raster_module._ArtifactRasterCells(object(), live, unfinished, selection)
    with pytest.raises(ValueError, match="^Embedding coordinates must be finite$"):
        list(broken.iter_row_blocks(columns=[raster_module._ARTIFACT_X]))


def _artifact_payload_store(monkeypatch, *, values_factory, selected_count=3):
    import zarr
    from zarr.storage import MemoryStore

    from scarf.storage import ArtifactRef

    selection = _raster_selection()
    monkeypatch.setattr(
        raster_module,
        "_validated_embedding_selection",
        lambda _store, _layout: selection,
    )
    monkeypatch.setattr(
        raster_module,
        "validate_stored_selection_integrity",
        lambda *args, **kwargs: SimpleNamespace(selected_count=selected_count),
    )
    group = zarr.open_group(store=MemoryStore(), mode="w")
    values_factory(group)
    monkeypatch.setattr(raster_module, "artifact_group", lambda _root, _ref: group)
    layout = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="embedding",
        artifact_id="b" * 64,
    )
    store = SimpleNamespace(zw=object(), cells=SimpleNamespace(columns=["I"]))
    return store, layout, selection


@pytest.mark.parametrize(
    ("values_factory", "error", "message"),
    [
        (
            lambda group: None,
            ValueError,
            "Embedding artifact has no canonical values array",
        ),
        (
            lambda group: group.create_group("values"),
            ValueError,
            "Embedding artifact values are malformed",
        ),
        (
            lambda group: group.create_array("values", shape=(3, 3), dtype="f8"),
            ValueError,
            "Embedding must have two columns and one row per selected cell",
        ),
        (
            lambda group: group.create_array("values", shape=(3, 2), dtype="bool"),
            TypeError,
            "Embedding coordinates must be numeric",
        ),
    ],
)
def test_artifact_raster_cells_validate_the_embedding_payload(
    monkeypatch,
    values_factory,
    error,
    message,
):
    store, layout, _ = _artifact_payload_store(
        monkeypatch,
        values_factory=values_factory,
    )

    with pytest.raises(error) as raised:
        raster_module._resolve_artifact_raster_cells(store, layout)

    assert raised.value.args == (message,)


def test_artifact_raster_cells_share_the_frozen_view_selection(monkeypatch):
    from scarf.storage import ArtifactRef

    store, layout, selection = _artifact_payload_store(
        monkeypatch,
        values_factory=lambda group: group.create_array(
            "values", data=np.zeros((3, 2), dtype=np.float32)
        ),
    )

    view, resolved = raster_module._resolve_artifact_raster_cells(store, layout)
    assert resolved == selection
    assert view.get_dtype(raster_module._ARTIFACT_Y) == np.dtype(np.float32)

    store.cells = SimpleNamespace(
        columns=["I"],
        _selection_ref=ArtifactRef(
            scope="datastore",
            kind="cell_selection",
            artifact_id="c" * 64,
        ),
    )
    with pytest.raises(ValueError) as raised:
        raster_module._resolve_artifact_raster_cells(store, layout)
    assert raised.value.args == (
        "Frozen run fields and embedding must share the same cell selection",
    )


@pytest.fixture(scope="module")
def imported_layout(tmp_path_factory):
    from tests.test_plotting_foundation import _imported_plot_store

    rng = np.random.default_rng(5)
    coordinates = rng.normal(size=(40, 2)) * np.array([3.0, 1.0])
    store, imported, counts = _imported_plot_store(
        tmp_path_factory.mktemp("raster_layout"),
        coordinates=coordinates,
        clusters=np.arange(40) % 3,
    )
    return SimpleNamespace(
        store=store,
        imported=imported,
        layout=imported.embeddingArtifacts["X_umap"],
        coordinates=coordinates,
        n_counts=counts.sum(axis=1),
    )


def test_embedding_raster_on_an_imported_artifact_layout(imported_layout):
    data = imported_layout
    result = splt.embedding_raster(
        data.store,
        layout=data.layout,
        color_by="RNA_nCounts",
        pixels=16,
        block_rows=8,
        show=False,
    )

    assert result.provenance.renderer == "matplotlib-raster"
    assert result.provenance.notes == (
        "embedding_raster",
        "two_pass",
        "artifact_layout",
        "live_metadata_fields",
        "approximate_quantiles",
    )
    assert result.provenance.assay == "RNA"
    assert result.provenance.extras["layout"] == data.layout.to_dict()
    assert (
        result.provenance.extras["cell_selection"]
        == data.imported.cellSelection.to_dict()
    )
    assert result.provenance.n_cells == 40
    assert result.provenance.extras["n_blocks"] == 5
    ax = result.axes["RNA_nCounts"]
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("Embedding1", "Embedding2")
    image = ax.get_images()[0]
    x, y = data.coordinates[:, 0], data.coordinates[:, 1]
    extent = _expected_extent(x, y)
    assert image.get_extent() == pytest.approx(extent)
    counts = _grid(x, y, extent, 16)
    sums = _grid(x, y, extent, 16, weights=data.n_counts)
    np.testing.assert_allclose(
        np.ma.filled(image.get_array(), np.nan),
        np.where(counts > 0, sums / np.maximum(counts, 1), np.nan),
        equal_nan=True,
    )
    assert image.get_clim() == pytest.approx(
        tuple(np.quantile(data.n_counts, [0.01, 0.99]))
    )
    assert image.colorbar.ax.get_xlabel() == "RNA_nCounts"
    result.close()


@pytest.mark.parametrize(
    ("column", "values", "color_by"),
    [
        ("label", np.array(["a", "b"] * 30, dtype=object), "label"),
        ("flag", np.tile([True, False], 30), "flag"),
        ("phase", np.arange(60) % 3, "phase"),
        (
            "score",
            np.linspace(0.0, 1.0, 60),
            splt.CellField("score", kind="categorical"),
        ),
    ],
    ids=["object", "boolean", "few-integers", "declared-categorical"],
)
def test_embedding_raster_rejects_categorical_metadata(column, values, color_by):
    columns = {
        "I": np.ones(60, dtype=bool),
        "U1": np.linspace(0.0, 1.0, 60),
        "U2": np.linspace(1.0, 0.0, 60),
        column: values,
    }

    with pytest.raises(NotImplementedError) as raised:
        splt.embedding_raster(
            _layout_store(columns, chunk=16),
            layout_key="U",
            color_by=color_by,
            pixels=16,
            block_rows=16,
            show=False,
        )

    assert raised.value.args == (
        "embedding_raster supports continuous color values only; use "
        "embedding() for categorical cell metadata",
    )


def test_embedding_raster_rejects_stored_categorical_display():
    columns = _layout_columns()
    store = _layout_store(
        columns,
        _stored_display_metadata=lambda column: {
            "kind": "categorical",
            "categories": [{"value": 0, "label": "zero", "color": "#000000"}],
        },
    )

    with pytest.raises(NotImplementedError, match="continuous color values only"):
        splt.embedding_raster(store, layout_key="U", color_by="score", show=False)


def test_embedding_raster_colors_many_integer_values_and_declared_fields():
    many = np.arange(120)
    columns = {
        "I": np.ones(120, dtype=bool),
        "U1": np.linspace(0.0, 1.0, 120),
        "U2": np.linspace(0.0, 1.0, 120),
        "many": many,
        "few": many % 3,
    }
    store = _layout_store(columns, chunk=50)

    result = splt.embedding_raster(
        store,
        layout_key="U",
        color_by="many",
        pixels=16,
        block_rows=50,
        show=False,
    )
    # More than 100 integer values are continuous; limits are the 1% quantiles.
    expected_limits = tuple(np.quantile(many, [0.01, 0.99]))
    assert result.axes["many"].get_images()[0].get_clim() == pytest.approx(
        expected_limits
    )
    assert expected_limits == pytest.approx((1.19, 117.81))
    result.close()

    declared = splt.embedding_raster(
        store,
        layout_key="U",
        color_by=splt.CellField("few", kind="continuous", label="Few"),
        pixels=16,
        show=False,
    )
    assert list(declared.axes) == ["Few"]
    assert declared.legends[0].label == "Few"
    assert declared.provenance.extras["color_by"] == "few"
    image = declared.axes["Few"].get_images()[0]
    assert image.colorbar.ax.get_xlabel() == "Few"
    assert image.get_clim() == pytest.approx(tuple(np.quantile(many % 3, [0.01, 0.99])))
    declared.close()


def test_embedding_raster_density_and_foreign_target():
    import matplotlib.pyplot as plt

    columns = _layout_columns()
    fig, ax = plt.subplots()
    result = splt.embedding_raster(
        _layout_store(columns),
        layout_key="U",
        pixels=16,
        target=ax,
        show=False,
    )

    assert result.owns_figure is False
    assert result.figure is fig
    assert list(result.axes.values()) == [ax]
    assert result.provenance.extras["color_mode"] == "density"
    assert result.provenance.extras["vmin"] == 0.0
    assert result.legends[0].label == "log1p cell count"
    image = np.ma.filled(ax.get_images()[0].get_array(), np.nan)
    # Each occupied pixel stores log1p(count), so the counts sum to the cells.
    assert np.nansum(np.expm1(image)) == pytest.approx(12)
    result.close()
    assert plt.fignum_exists(fig.number)
    plt.close(fig)


def test_embedding_raster_image_fills_square_axes():
    columns = _layout_columns()
    columns["U1"] = columns["U1"] * 5.0
    result = splt.embedding_raster(
        _layout_store(columns),
        layout_key="U",
        color_by="score",
        pixels=16,
        show=False,
    )
    ax = next(iter(result.axes.values()))
    images = ax.get_images()
    assert len(images) == 1
    extent = images[0].get_extent()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    assert extent == pytest.approx(_expected_extent(columns["U1"], columns["U2"]))
    assert extent[0] == pytest.approx(xlim[0])
    assert extent[1] == pytest.approx(xlim[1])
    assert extent[2] == pytest.approx(ylim[0])
    assert extent[3] == pytest.approx(ylim[1])
    assert (xlim[1] - xlim[0]) == pytest.approx(ylim[1] - ylim[0])
    assert ax.get_box_aspect() == pytest.approx(1.0)
    result.close()


def test_embedding_raster_requires_one_coordinate_source():
    from scarf.storage import ArtifactRef

    layout = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="embedding",
        artifact_id="d" * 64,
    )
    store = object()
    with pytest.raises(
        ValueError, match="^Provide exactly one of layout_key or layout$"
    ):
        splt.embedding_raster(store, show=False)
    with pytest.raises(
        ValueError, match="^Provide exactly one of layout_key or layout$"
    ):
        splt.embedding_raster(
            store,
            layout_key="literal",
            layout=layout,
            show=False,
        )
    with pytest.raises(ValueError, match="cannot override"):
        splt.embedding_raster(
            store,
            layout=layout,
            cell_key="another_selection",
            show=False,
        )


def test_embedding_raster_rejects_non_embedding_artifact():
    from scarf.storage import ArtifactRef

    clusters = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="cluster_cut",
        artifact_id="e" * 64,
    )
    with pytest.raises(
        ValueError, match="^layout must identify an embedding artifact$"
    ):
        splt.embedding_raster(SimpleNamespace(), layout=clusters, show=False)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        (
            {"layout_key": "nope"},
            KeyError,
            "Layout column 'nope1' not found in cell metadata",
        ),
        (
            {"color_by": "CD3D"},
            KeyError,
            "color_by 'CD3D' must be a cell-metadata column for embedding_raster "
            "(gene coloring uses embedding() for now)",
        ),
        (
            {"color_by": "score", "color_scale": splt.ColorScale(scale="log")},
            NotImplementedError,
            "embedding_raster currently supports only linear color scales",
        ),
        (
            {"subset_by": "nope"},
            KeyError,
            "subset_by 'nope' not found in cell metadata",
        ),
        ({"pixels": 7}, ValueError, "pixels must be >= 8"),
        (
            {"subset_by": "count"},
            TypeError,
            "subset_by 'count' must be boolean; got int64",
        ),
    ],
)
def test_embedding_raster_rejects_unusable_inputs(kwargs, error, message):
    columns = _layout_columns()
    columns["count"] = np.arange(12, dtype=np.int64)
    options = {"layout_key": "U", **kwargs}

    with pytest.raises(error) as raised:
        splt.embedding_raster(_layout_store(columns), show=False, **options)

    assert raised.value.args == (message,)


def test_embedding_raster_shows_owned_results_by_default(monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr("IPython.get_ipython", lambda: None)

    result = splt.embedding_raster(
        _layout_store(_layout_columns()),
        layout_key="U",
        pixels=16,
    )

    assert not plt.fignum_exists(result.figure.number)
    assert "rendered=False" in repr(result)


def test_raster_validates_quantiles_pixels_and_subset_column():
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
    with pytest.raises(ValueError, match="^pixels must be >= 8$"):
        raster_from_metadata(cells, x_key="x", y_key="y", pixels=7)
    with pytest.raises(KeyError, match="subset_by 'nope' not found in cell metadata"):
        raster_from_metadata(cells, x_key="x", y_key="y", subset_by="nope")


def test_raster_without_color_encodes_log_density():
    from scarf.plotting._raster import raster_from_metadata

    cells = _GuardedMeta(
        {
            "I": np.ones(4, dtype=bool),
            "x": np.array([0.0, 0.0, 0.0, 1.0]),
            "y": np.array([0.0, 0.0, 0.0, 1.0]),
        }
    )
    canvas = raster_from_metadata(
        cells,
        x_key="x",
        y_key="y",
        pixels=8,
    )

    assert canvas.extent == pytest.approx((-0.01, 1.01, -0.01, 1.01))
    expected_counts = np.zeros((8, 8), dtype=np.int64)
    # Three cells sit at the bottom-left corner and one at the top-right corner.
    expected_counts[7, 0] = 3
    expected_counts[0, 7] = 1
    np.testing.assert_array_equal(canvas.counts, expected_counts)
    assert canvas.image[7, 0] == pytest.approx(np.log(4))
    assert canvas.image[0, 7] == pytest.approx(np.log(2))
    assert np.isnan(canvas.image).sum() == 62
    assert (canvas.vmin, canvas.vmax) == pytest.approx((0.0, np.log(4)))


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

    # The y window widens to the x span around its centre, 0.5.
    assert canvas.extent == pytest.approx((-0.1, 10.1, -4.6, 5.6))
    occupied = sorted(zip(*np.nonzero(canvas.counts)))
    assert occupied == [(45, 0), (45, 99), (54, 0), (54, 99)]


def test_density_canvas_from_points_uses_raster_contract():
    from scarf.plotting._raster import density_canvas_from_points

    canvas = density_canvas_from_points(
        np.array([0.15, 0.15, 0.85, np.nan]),
        np.array([0.35, 0.35, 0.65, 0.2]),
        extent=(0, 1, 0, 1),
        pixels=10,
    )

    expected_counts = np.zeros((10, 10), dtype=np.int64)
    # Rows count down from the top: y = 0.35 lands in row 6 and y = 0.65 in row 3.
    expected_counts[6, 1] = 2
    expected_counts[3, 8] = 1
    np.testing.assert_array_equal(canvas.counts, expected_counts)
    assert canvas.image[6, 1] == pytest.approx(np.log(3))
    assert canvas.image[3, 8] == pytest.approx(np.log(2))
    assert np.isnan(canvas.image).sum() == 98
    assert canvas.n_cells == 3
    assert canvas.n_blocks == 1
    assert (canvas.vmin, canvas.vmax) == pytest.approx((0.0, np.log(3)))
    assert canvas.extent == (0, 1, 0, 1)


@pytest.mark.parametrize(
    ("x", "y", "kwargs", "message"),
    [
        ([0.5], [0.5], {"pixels": 7}, "pixels must be at least 8"),
        ([0.5, 0.6], [0.5], {}, "x and y lengths must match"),
        ([0.5], [0.5], {"extent": (1, 0, 0, 1)}, "extent must have increasing"),
        ([0.5], [0.5], {"extent": (0, 1, 1, 1)}, "extent must have increasing"),
    ],
)
def test_density_canvas_from_points_validates_its_grid(x, y, kwargs, message):
    from scarf.plotting._raster import density_canvas_from_points

    options = {"extent": (0, 1, 0, 1), "pixels": 8, **kwargs}
    with pytest.raises(ValueError, match=message):
        density_canvas_from_points(np.asarray(x), np.asarray(y), **options)


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
    assert (first.n_blocks, second.n_blocks) == (9, 5)
    assert first.extent == pytest.approx(second.extent)
    assert first.vmin == pytest.approx(second.vmin)
    assert first.vmax == pytest.approx(second.vmax)
    np.testing.assert_allclose(first.image, second.image, equal_nan=True)
    np.testing.assert_array_equal(first.counts, second.counts)


@pytest.mark.parametrize("quantiles", [None, (0.01, 0.99)])
def test_raster_all_missing_color_has_finite_default_limits(quantiles):
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
        quantiles=quantiles,
    )
    assert (canvas.vmin, canvas.vmax) == (0.0, 1.0)
    assert canvas.n_cells == 4
    assert canvas.counts.sum() == 0
    assert np.isnan(canvas.image).all()


def test_raster_exact_limits_skip_blocks_without_color_values():
    from scarf.plotting._raster import raster_from_metadata

    cells = _GuardedMeta(
        {
            "I": np.ones(8, dtype=bool),
            "x": np.arange(8, dtype=float),
            "y": np.arange(8, dtype=float),
            "value": np.array([np.nan] * 4 + [1.0, 2.0, 3.0, 4.0]),
        },
        chunk=4,
    )
    canvas = raster_from_metadata(
        cells,
        x_key="x",
        y_key="y",
        color_key="value",
        pixels=8,
        block_rows=4,
        quantiles=None,
    )

    assert canvas.n_blocks == 2
    assert (canvas.vmin, canvas.vmax) == (1.0, 4.0)
    assert canvas.counts.sum() == 4


@pytest.mark.parametrize(
    "columns",
    [
        {"I": np.ones(6, dtype=bool), "keep": np.zeros(6, dtype=bool)},
        {"I": np.zeros(6, dtype=bool), "keep": np.ones(6, dtype=bool)},
    ],
    ids=["subset-excludes-all", "no-active-cells"],
)
def test_raster_without_drawable_cells_returns_an_empty_canvas(columns):
    from scarf.plotting._raster import raster_from_metadata

    cells = _GuardedMeta(
        {
            **columns,
            "x": np.arange(6, dtype=float),
            "y": np.arange(6, dtype=float),
            "value": np.arange(6, dtype=float),
        },
        chunk=4,
    )
    canvas = raster_from_metadata(
        cells,
        x_key="x",
        y_key="y",
        color_key="value",
        subset_by="keep",
        pixels=8,
        block_rows=4,
    )

    assert canvas.extent == (0.0, 1.0, 0.0, 1.0)
    assert (canvas.vmin, canvas.vmax, canvas.n_cells) == (0.0, 1.0, 0)
    assert canvas.n_blocks == 2
    np.testing.assert_array_equal(canvas.counts, np.zeros((8, 8), dtype=np.int64))
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
    # The masked placeholder 0 never reaches the color limits.
    assert (colored.vmin, colored.vmax) == (1.0, 3.0)

    subset = raster_from_metadata(
        cells,
        x_key="x",
        y_key="y",
        subset_by="keep",
        pixels=8,
    )
    assert subset.n_cells == 2
    assert subset.counts.sum() == 2
    assert np.nanmax(subset.image) == pytest.approx(np.log(3))


def test_embedding_raster_uses_one_effective_missing_color():
    from matplotlib.colors import to_hex

    result = splt.embedding_raster(
        _layout_store(_layout_columns()),
        layout_key="U",
        color_scale=splt.ColorScale(missing_color="pink"),
        missing_color="#123456",
        pixels=16,
        show=False,
    )

    image = result.axes["U"].get_images()[0]
    assert to_hex(image.cmap.get_bad()) == "#123456"
    assert result.scales[0].missing_color == "#123456"
    assert result.provenance.extras["missing_color"] == "#123456"
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
    image = result.axes["score"].get_images()[0]
    assert image.cmap.name == "magma"
    assert image.get_clim() == (0.0, 3.0)
    assert result.legends[0].extras == {"vmin": 0.0, "vmax": 3.0}
    assert result.provenance.extras["missing_color"] == "white"
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
    x = np.linspace(0, 1, n)
    value = np.arange(n, dtype=float)
    cells = _GuardedMeta(
        {
            "I": np.ones(n, dtype=bool),
            "x": x,
            "y": x,
            "value": value,
            "keep": keep,
        }
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
    assert subset.counts.sum() == 5
    # Only the kept cells set the window and the colors.
    assert subset.extent == pytest.approx(_expected_extent(x[:5], x[:5]))
    assert sorted(subset.image[subset.counts > 0]) == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert (subset.vmin, subset.vmax) == pytest.approx(
        tuple(np.quantile(value[:5], [0.01, 0.99]))
    )


def test_raster_keeps_a_drawable_window_for_constant_large_coordinates():
    from scarf.plotting._raster import raster_from_metadata

    cells = _GuardedMeta(
        {
            "I": np.ones(4, dtype=bool),
            "x": np.full(4, 1e17),
            "y": np.full(4, -1e17),
        }
    )

    canvas = raster_from_metadata(cells, x_key="x", y_key="y", pixels=8)

    # Float64 absorbs the unit padding at 1e17, so the window widens by
    # representable steps instead, and every cell still lands in one bin.
    xmin, xmax, ymin, ymax = canvas.extent
    assert xmin < 1e17 < xmax
    assert ymin < -1e17 < ymax
    assert int(canvas.counts.sum()) == 4
    assert np.count_nonzero(canvas.counts) == 1


def test_embedding_raster_hides_internal_artifact_coordinate_fields(monkeypatch):
    from scarf.storage import ArtifactRef

    _patch_selection_blocks(monkeypatch, _two_selection_blocks())
    selection = _raster_selection()
    view = raster_module._ArtifactRasterCells(
        object(),
        _GuardedMeta({"I": np.ones(5, dtype=bool), "score": np.arange(5.0)}),
        np.arange(6, dtype=np.float64).reshape(3, 2),
        selection,
    )
    monkeypatch.setattr(
        raster_module,
        "_resolve_artifact_raster_cells",
        lambda _store, _layout: (view, selection),
    )
    layout = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="embedding",
        artifact_id="f" * 64,
    )

    store = SimpleNamespace(cells=view, zw=object(), _stored_display_metadata=None)
    with pytest.raises(
        KeyError,
        match=f"color_by '{raster_module._ARTIFACT_X}' must be a cell-metadata column",
    ):
        splt.embedding_raster(
            store,
            layout=layout,
            color_by=raster_module._ARTIFACT_X,
            pixels=8,
            show=False,
        )
    with pytest.raises(
        KeyError,
        match=f"subset_by '{raster_module._ARTIFACT_Y}' not found in cell metadata",
    ):
        splt.embedding_raster(
            store,
            layout=layout,
            subset_by=raster_module._ARTIFACT_Y,
            pixels=8,
            show=False,
        )

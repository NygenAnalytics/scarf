"""Foundation and integration tests for scarf.plotting."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import scarf.plotting as splt
from scarf.assay.normalization import applicable_normalization_flags
from scarf.storage import ArtifactRef


class _ArrayCells:
    def __init__(self, columns):
        self._columns = {name: np.asarray(values) for name, values in columns.items()}
        lengths = {len(values) for values in self._columns.values()}
        if len(lengths) != 1:
            raise ValueError("Synthetic cell columns must have matching lengths")
        self.N = lengths.pop()
        self.columns = tuple(self._columns)

    def active_index(self, key):
        return np.flatnonzero(np.asarray(self._columns[key], dtype=bool))

    def fetch(self, column, key="I"):
        return self._columns[column][self.active_index(key)]

    def fetch_all(self, column):
        return self._columns[column]

    def _get_array(self, column):
        return self._columns[column]


class _ArrayFeatures:
    def __init__(self, names):
        self._names = np.asarray(names, dtype=object)
        self._ids = np.asarray(
            [f"feature-{index}" for index in range(len(names))],
            dtype=object,
        )
        self.N = len(names)

    def fetch_all(self, column):
        return self._names if column == "names" else self._ids

    def get_index_by(self, values, column):
        source = self._names if column == "names" else self._ids
        indices = []
        for value in values:
            indices.extend(np.flatnonzero(source == value).tolist())
        return np.asarray(indices, dtype=np.int64)


class _ArrayAssay:
    # A synthetic assay measured every cell: the cell table that feature
    # reads take its membership from holds no membership column.
    cells = SimpleNamespace(columns=())

    def __init__(self, values, names):
        self._values = np.asarray(values, dtype=np.float64)
        self.rawData = self._values
        self.feats = _ArrayFeatures(names)

    def normed(self, *, cell_idx, feat_idx, log_transform=False):
        values = self._values[
            np.ix_(
                np.asarray(cell_idx, dtype=np.int64),
                np.asarray(feat_idx, dtype=np.int64),
            )
        ]
        return np.log1p(values) if log_transform else values


class _ArrayStore:
    _defaultAssay = "RNA"
    nthreads = 1
    zw = None

    def __init__(self, columns, feature_values, names=("GeneA", "GeneB")):
        self.cells = _ArrayCells(columns)
        self.RNA = _ArrayAssay(feature_values, list(names))

    def _get_assay(self, name):
        if name != "RNA":
            raise KeyError(name)
        return self.RNA

    @staticmethod
    def _stored_display_metadata(_column):
        return None


def _imported_plot_store(
    directory,
    *,
    coordinates,
    clusters,
    genes=("CD3E", "MS4A1", "LYZ", "NKG7", "GNLY", "FCGR3A"),
    seed=7,
):
    """Import a small AnnData file with one embedding and one cluster labelling.

    The store holds real immutable artifacts (an imported embedding, cluster
    labels and their shared cell selection) without building a graph, so
    artifact-backed plots run in about a second. Returns the opened store, the
    import result and the imported counts.
    """
    anndata = pytest.importorskip("anndata")
    from scipy.sparse import csr_matrix

    from scarf.datastore.datastore import DataStore
    from scarf.readers import H5adReader
    from scarf.writers import H5adToZarr

    coordinates = np.asarray(coordinates, dtype=np.float64)
    n_cells = len(coordinates)
    counts = (
        np.random.default_rng(seed)
        .poisson(3.0, size=(n_cells, len(genes)))
        .astype(np.float32)
    )
    obs = pd.DataFrame(
        {"clusters": pd.array(np.asarray(clusters), dtype="Int64")},
        index=[f"c{index}" for index in range(n_cells)],
    )
    var = pd.DataFrame(
        {"gene_short_name": list(genes)},
        index=[f"f{index}" for index in range(len(genes))],
    )
    source = Path(directory) / "plot_inputs.h5ad"
    anndata.AnnData(
        X=csr_matrix(counts),
        obs=obs,
        var=var,
        obsm={"X_umap": coordinates},
    ).write_h5ad(source)
    reader = H5adReader(
        str(source),
        embedding_roles={"X_umap": "umap"},
        cluster_keys=("clusters",),
    )
    try:
        imported = H5adToZarr(
            reader,
            zarr_loc=str(Path(directory) / "plot_inputs.zarr"),
            nthreads=1,
        ).dump()
    finally:
        reader.h5.close()
    store = DataStore(
        str(Path(directory) / "plot_inputs.zarr"),
        default_assay="RNA",
        min_features_per_cell=0,
        nthreads=1,
    )
    return store, imported, counts


def _summarize(
    store,
    *,
    features,
    group_by=None,
    groups=None,
    from_assay=None,
    **options,
):
    """Aggregate features the way dotplot and matrixplot do, without drawing."""
    from scarf.plotting._data import (
        _check_feature_count,
        _resolve_grouping,
        _summarize_resolved_features,
        coerce_feature_list,
        resolve_feature,
    )

    pairs = coerce_feature_list(features)
    _check_feature_count(pairs)
    aggregate, per_sample, _unmeasured = _summarize_resolved_features(
        store,
        [
            resolve_feature(store, feature, from_assay=from_assay)
            for _, feature in pairs
        ],
        [group for group, _ in pairs],
        _resolve_grouping(store, group_by=group_by, groups=groups, cell_key="I"),
        **options,
    )
    return aggregate, per_sample


@pytest.fixture
def synthetic_plot_store():
    sample = np.repeat(["s1", "s2", "s3", "s4"], 3).astype(object)
    subject = np.repeat(["donor1", "donor2", "donor1", "donor2"], 3).astype(object)
    sample_with_missing = sample.copy()
    sample_with_missing[0] = None
    inconsistent_subject = subject.copy()
    inconsistent_subject[1] = "donor3"
    columns = {
        "I": np.ones(12, dtype=bool),
        "none_selected": np.zeros(12, dtype=bool),
        "metricA": np.array(
            [10.0, 11.0, 9.0, 10.5, 4.0, 5.0, 6.0, 5.5, 1.0, 2.0, 3.0, 2.5]
        ),
        "metricB": np.linspace(20.0, 31.0, 12),
        "group": np.repeat(["group10", "group2", "group1"], 4),
        "category": np.array(["B", "A", None] * 4, dtype=object),
        "category_complete": np.array(["B", "A"] * 6, dtype=object),
        "split": np.array(["left", "right"] * 6, dtype=object),
        "split3": np.array(["left", "middle", "right"] * 4, dtype=object),
        "sample": sample,
        "sample_with_missing": sample_with_missing,
        "invalid_sample": np.full(12, "", dtype=object),
        "condition": np.repeat(["control", "control", "treated", "treated"], 3),
        "invalid_condition": np.full(12, "", dtype=object),
        "subject": subject,
        "inconsistent_subject": inconsistent_subject,
        # A layout on x = 0..11 with y cycling 0, 1.5, 3, 4.5, a second layout,
        # and a continuous score that jumps between two conditions.
        "plot_layout1": np.arange(12, dtype=np.float64),
        "plot_layout2": (np.arange(12) % 4) * 1.5,
        "plot_mirror1": -np.arange(12, dtype=np.float64),
        "plot_mirror2": np.arange(12, dtype=np.float64)[::-1],
        "plot_condition": np.repeat(["low", "high"], 6).astype(object),
        "plot_score": np.concatenate(
            (np.linspace(0.0, 1.0, 6), np.linspace(10.0, 11.0, 6))
        ),
        "keep": np.arange(12) < 8,
        "paired_category": np.array(list("AABABBBBBAAA"), dtype=object),
    }
    columns["plot_score_scaled"] = columns["plot_score"] * 10
    feature_values = np.column_stack(
        (
            np.linspace(0.0, 5.5, 12),
            np.array([0.0, 1.0, 0.0, 2.0, 4.0, 2.0, 8.0, 4.0, 8.0, 16.0, 8.0, 4.0]),
        )
    )
    return _ArrayStore(columns, feature_values)


@pytest.mark.parametrize("name", ["mapping_correction", "unified_embedding"])
def test_retired_mapping_plots_are_absent(name):
    from scarf.plotting.recipes import ALLOWED_PLOTS

    assert name not in splt.__all__
    assert name not in ALLOWED_PLOTS
    with pytest.raises(AttributeError):
        getattr(splt, name)


def test_plotting_modules_import_without_optional_dependencies():
    script = """
import builtins
import pandas as pd

original_import = builtins.__import__

def block_plotting_dependencies(name, *args, **kwargs):
    if name.split(".", 1)[0] in {"matplotlib", "seaborn", "kneed"}:
        raise ModuleNotFoundError(name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = block_plotting_dependencies
import scarf.plotting as plotting
try:
    plotting.elbow([1.0, 0.5], show=False)
except ImportError as exc:
    assert "scarf[extra]" in str(exc)
else:
    raise AssertionError("plot use should require optional dependencies")
try:
    plotting.qc(
        pd.DataFrame({"groups": ["a"], "value": [1.0]}),
        show=False,
    )
except ImportError as exc:
    assert "scarf[extra]" in str(exc)
else:
    raise AssertionError("plot use should require matplotlib")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_study_design_allows_pairing_rejects_tech_rep():
    design = splt.StudyDesign(
        sample_by="sample", subject_by="donor", condition_by="time"
    )
    assert design.subject_by == "donor"
    with pytest.raises(NotImplementedError, match="technical_replicate_by"):
        splt.StudyDesign(sample_by="sample", technical_replicate_by="lane")


def test_size_scale_maps_fraction_to_area():
    scale = splt.SizeScale(vmin=0, vmax=1, size_min=10, size_max=200)
    areas = scale.areas(np.array([0.0, 0.5, 1.0]))
    assert areas[0] == pytest.approx(10)
    assert areas[1] == pytest.approx(105)
    assert areas[2] == pytest.approx(200)


def test_plotting_contracts_reject_invalid_values():
    with pytest.raises(ValueError, match="source"):
        splt.NormalizationSpec(source="scaled")
    with pytest.raises(ValueError, match="quantiles"):
        splt.ColorScale(quantiles=(0.9, 0.1))
    with pytest.raises(ValueError, match="vmax"):
        splt.ColorScale(vmin=2, vmax=1)
    with pytest.raises(ValueError, match="size range"):
        splt.SizeScale(size_min=20, size_max=10)
    with pytest.raises(ValueError, match="kind"):
        splt.CellField("group", kind="ordinal")


@pytest.mark.parametrize(
    ("factory", "kwargs", "message"),
    [
        (splt.ColorScale, {"vmin": 0, "vcenter": 2, "vmax": 1}, "vcenter"),
        (splt.ColorScale, {"scope": "global"}, "scope"),
        (splt.ColorScale, {"scale": "sqrt"}, "scale"),
        (splt.CategoricalScale, {"palette_name": "pastel"}, "palette_name"),
        (splt.DensityOverlay, {"kind": "dots"}, "kind"),
        (splt.DensityOverlay, {"statistic": "median"}, "statistic"),
        (splt.DensityOverlay, {"pixels": 15}, "pixels"),
        (splt.DensityOverlay, {"sigma": -1}, "sigma"),
        (splt.DensityOverlay, {"min_support": 0}, "min_support"),
        (splt.DensityOverlay, {"levels": 0}, "levels"),
        (splt.DensityOverlay, {"levels": (0.2, np.nan)}, "levels"),
        (splt.DensityOverlay, {"groups": ("a",)}, "groups requires"),
        (splt.DensityOverlay, {"alpha": 2}, "alpha"),
        (splt.DensityOverlay, {"linewidth": -1}, "linewidth"),
        (splt.DensityOverlay, {"halo_width": -1}, "halo_width"),
        (splt.Highlight, {}, "exactly one"),
        (splt.Highlight, {"by": ""}, "by must be non-empty"),
        (splt.Highlight, {"indices": (-1,)}, "indices must be non-negative"),
        (splt.Highlight, {"indices": (1,), "groups": ("a",)}, "groups requires"),
        (splt.Highlight, {"indices": (1,), "alpha": 2}, "alpha values"),
        (splt.Highlight, {"indices": (1,), "size_multiplier": 0}, "positive"),
        (splt.Highlight, {"indices": (1,), "halo_width": -1}, "halo_width"),
    ],
)
def test_plotting_contracts_cover_each_validation_invariant(factory, kwargs, message):
    with pytest.raises(ValueError, match=message):
        factory(**kwargs)


def test_size_scale_with_degenerate_domain_uses_the_minimum_area():
    scale = splt.SizeScale(vmin=2.0, vmax=2.0, size_min=7.0, size_max=20.0)

    np.testing.assert_array_equal(scale.areas(np.array([1.0, 2.0, 3.0])), 7.0)


def test_plot_provenance_falls_back_when_distribution_metadata_is_unavailable(
    monkeypatch,
):
    from scarf.plotting import _contracts

    _contracts.installed_scarf_version.cache_clear()
    monkeypatch.setattr(
        _contracts,
        "version",
        lambda _name: (_ for _ in ()).throw(LookupError("missing metadata")),
    )

    assert _contracts.installed_scarf_version() == "unknown"
    _contracts.installed_scarf_version.cache_clear()


@pytest.mark.parametrize("block_size", [1, 3, 7])
@pytest.mark.parametrize("sample_by", [None, "sample_with_missing"])
def test_feature_summary_matches_cell_table_across_blocks(
    synthetic_plot_store, block_size, sample_by
):
    import zarr

    from scarf.matrix import ChunkedArray

    store = synthetic_plot_store
    values = store.RNA._values.copy()
    values[1, 0] = np.nan
    store.RNA.rawData = ChunkedArray(zarr.array(values, chunks=(block_size, 2)))
    features = {
        "markers": [
            splt.FeatureRef(0, by="index", label="shared"),
            splt.FeatureRef(1, by="index", label="shared"),
        ],
        "control": [splt.FeatureRef(1, by="index", label="other")],
    }
    normalized = np.log1p(values)
    frames = []
    for feature_group, items in features.items():
        for feature in items:
            # Summary tables name the grouping columns by role.
            column = normalized[:, int(feature.value)]
            frame = pd.DataFrame(
                {
                    "group": store.cells.fetch_all("category"),
                    "subgroup": store.cells.fetch_all("split"),
                    "feature": feature.label,
                    "feature_group": feature_group,
                    "value": column,
                    # Every statistic covers the cells with a value, which
                    # n_cells counts: the NaN cell joins no denominator.
                    "detected": np.where(np.isnan(column), np.nan, column > 0.5),
                }
            )
            if sample_by is not None:
                frame["sample"] = store.cells.fetch_all(sample_by)
                frame = frame.loc[frame["sample"].notna()]
            frames.append(frame)
    keys = ["group", "subgroup", "feature", "feature_group"]
    expected = (
        pd.concat(frames)
        .groupby((["sample"] if sample_by else []) + keys, dropna=False)
        .agg(
            mean=("value", "mean"),
            fraction=("detected", "mean"),
            n_cells=("value", "count"),
            variance=("value", "var"),
        )
        .reset_index()
    )

    aggregate, per_sample = _summarize(
        store,
        features=features,
        group_by=("category", "split"),
        sample_by=sample_by,
        normalization=splt.NormalizationSpec(source="raw", transform="log1p"),
        expression_cutoff=0.5,
    )

    if sample_by is not None:
        pd.testing.assert_frame_equal(per_sample, expected)
        expected = (
            expected.groupby(keys, dropna=False)
            .agg(
                mean=("mean", "mean"),
                fraction=("fraction", "mean"),
                n_cells=("n_cells", "sum"),
                # Samples without a value of the feature are skipped.
                n_samples=("n_cells", lambda counts: int((counts > 0).sum())),
                variance=("variance", "mean"),
            )
            .reset_index()
        )
    else:
        assert per_sample is None
    pd.testing.assert_frame_equal(aggregate, expected)


# Row 11 with blocks of 4 puts the infinity in the third streamed block.
@pytest.mark.parametrize(("row", "block_size"), [(0, 1), (11, 4)])
@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_feature_summary_rejects_infinite_values(
    synthetic_plot_store, row, block_size, value
):
    import zarr

    from scarf.matrix import ChunkedArray

    store = synthetic_plot_store
    values = store.RNA._values.copy()
    values[row, 0] = value
    store.RNA.rawData = ChunkedArray(zarr.array(values, chunks=(block_size, 2)))

    with pytest.raises(ValueError, match="infinity after normalization"):
        _summarize(
            store,
            features=["GeneA"],
            group_by="group",
            sample_by="sample",
            normalization=splt.NormalizationSpec(source="raw"),
        )


def test_feature_summary_excludes_invalid_samples_before_infinity_check(
    synthetic_plot_store,
):
    store = synthetic_plot_store
    store.RNA.rawData[0, 0] = np.inf
    aggregate, _ = _summarize(
        store,
        features=["GeneA"],
        group_by="group",
        sample_by="sample_with_missing",
        normalization=splt.NormalizationSpec(source="raw"),
    )

    # Cell 0 holds the infinity but has no sample, so it never counts.
    by_group = aggregate.set_index("group").loc[["group1", "group2", "group10"]]
    np.testing.assert_allclose(by_group["mean"], [4.5, 2.75, 1.125])
    assert by_group["n_cells"].tolist() == [4, 4, 3]
    assert by_group["n_samples"].tolist() == [2, 2, 2]


def test_feature_summary_releases_source_blocks(synthetic_plot_store, monkeypatch):
    import weakref

    import zarr

    from scarf.matrix import ChunkedArray

    store = synthetic_plot_store
    store.RNA.rawData = ChunkedArray(zarr.array(store.RNA._values, chunks=(1, 2)))
    references = []
    peak_blocks = 0
    original = ChunkedArray._materialize_range

    def materialize(array, start, stop):
        nonlocal peak_blocks
        block = original(array, start, stop)
        references.append(weakref.ref(block))
        peak_blocks = max(peak_blocks, sum(ref() is not None for ref in references))
        return block

    monkeypatch.setattr(ChunkedArray, "_materialize_range", materialize)
    aggregate, _ = _summarize(
        store,
        features=["GeneA", "GeneB"],
        group_by="group",
        normalization=splt.NormalizationSpec(source="raw"),
    )

    assert len(references) == store.cells.N
    assert peak_blocks <= 3
    np.testing.assert_allclose(
        aggregate.groupby("feature")["mean"].mean(),
        store.RNA._values.mean(axis=0),
    )


@pytest.mark.parametrize(
    ("fixture_name", "assay_name"),
    [("datastore", "RNA"), ("toy_crdir_ds", "ADT"), ("atac_datastore", "ATAC")],
)
def test_feature_summary_preserves_assay_normalization(
    request, fixture_name, assay_name
):
    store = request.getfixturevalue(fixture_name)
    cell_idx = store.cells.active_index("I")
    expected = np.asarray(
        store._get_assay(assay_name)
        .normed(cell_idx=cell_idx, feat_idx=np.array([0, 1]))
        .compute(),
        dtype=np.float64,
    )
    # Only values that are not logarithms are logged: RNA library sizes are,
    # while CLR log ratios and ATAC TF-IDF values are not.
    logged = assay_name == "RNA"
    if logged:
        expected = np.log1p(expected)
    aggregate, per_sample = _summarize(
        store,
        features=[
            splt.FeatureRef(index, by="index", label=f"feature-{index}")
            for index in (1, 0)
        ],
        from_assay=assay_name,
        group_by="I",
        normalization=splt.NormalizationSpec(transform="log1p" if logged else "none"),
    )

    assert per_sample is None
    assert aggregate["feature"].tolist() == ["feature-0", "feature-1"]
    np.testing.assert_allclose(aggregate["mean"], expected.mean(axis=0))
    np.testing.assert_allclose(aggregate["variance"], expected.var(axis=0, ddof=1))
    np.testing.assert_allclose(aggregate["fraction"], (expected > 0).mean(axis=0))
    np.testing.assert_array_equal(aggregate["n_cells"], len(cell_idx))


def test_embedding_keeps_square_panel_with_side_legend():
    rng = np.random.default_rng(0)
    n_cells = 48
    store = _ArrayStore(
        {
            "I": np.ones(n_cells, dtype=bool),
            "plot_layout1": rng.normal(size=n_cells),
            "plot_layout2": rng.normal(size=n_cells),
            "plot_group": np.array(
                [str(index % 6) for index in range(n_cells)],
                dtype=object,
            ),
        },
        np.zeros((n_cells, 2)),
    )
    result = splt.embedding(
        store,
        layout_key="plot_layout",
        color_by="plot_group",
        legend_loc="right",
        show=False,
    )
    ax = next(iter(result.axes.values()))
    assert ax.get_box_aspect() == pytest.approx(1.0)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    assert (xlim[1] - xlim[0]) == pytest.approx(ylim[1] - ylim[0])
    result.figure.canvas.draw()
    bbox = ax.get_window_extent()
    assert bbox.width == pytest.approx(bbox.height, rel=1e-3)
    result.close()


@pytest.fixture(scope="module")
def plot_artifacts(tmp_path_factory):
    """Real layout and cluster artifacts on a small imported store.

    Twelve cells lie on x = 0..11 with y cycling through 0, 1.5, 3 and 4.5, in
    clusters 10, 2 and 1 of four cells each.
    """
    x = np.arange(12, dtype=np.float64)
    coordinates = np.column_stack((x, (x % 4) * 1.5))
    labels = np.repeat([10, 2, 1], 4)
    store, imported, counts = _imported_plot_store(
        tmp_path_factory.mktemp("plot_artifacts"),
        coordinates=coordinates,
        clusters=labels,
    )
    return SimpleNamespace(
        store=store,
        layout=imported.embeddingArtifacts["X_umap"],
        clusters=imported.clusterArtifacts["clusters"],
        selection=imported.cellSelection,
        coordinates=coordinates,
        labels=labels,
        # The RNA assay normalizes each cell's counts to 1000.
        normalized=counts / counts.sum(axis=1, keepdims=True) * 1000.0,
    )


def _facecolor_rgba(collection):
    collection.update_scalarmappable()
    return np.asarray(collection.get_facecolors())


def test_embedding_dotplot_matrixplot_on_artifacts(plot_artifacts):
    from matplotlib.colors import Normalize, to_rgba

    data = plot_artifacts
    sizes = np.linspace(5, 40, 12)
    emb = splt.embedding(
        data.store,
        layout=data.layout,
        color_by=data.clusters,
        point_sizes=sizes,
        sort_values=False,
        show=False,
    )
    assert emb.owns_figure
    (ax,) = emb.axes.values()
    (points,) = ax.collections
    # Unsorted cells keep layout order with their own sizes.
    np.testing.assert_allclose(points.get_offsets(), data.coordinates)
    np.testing.assert_allclose(points.get_sizes(), sizes)
    scale = emb.scales[1]
    assert scale.order == (1, 2, 10)
    np.testing.assert_allclose(
        _facecolor_rgba(points),
        [to_rgba(scale.palette[label]) for label in data.labels],
    )
    legend = emb.figure.legends[0]
    assert [text.get_text() for text in legend.get_texts()] == ["1", "2", "10"]
    emb.close()

    # Gene coloring with sort_values draws the highest expression last.
    expression = data.normalized[:, 0]
    emb2 = splt.embedding(
        data.store,
        layout=data.layout,
        color_by="CD3E",
        sort_values=True,
        show=False,
    )
    (points,) = emb2.axes["CD3E"].collections
    drawn = np.asarray(points.get_offsets())[:, 0].astype(int)
    assert sorted(drawn) == list(range(12))
    assert np.all(np.diff(expression[drawn]) >= -1e-4)
    limits = emb2.provenance.extras["color_limits"]["CD3E"]
    assert limits == pytest.approx((expression.min(), expression.max()), rel=1e-5)
    np.testing.assert_allclose(
        _facecolor_rgba(points),
        plt_colormap("viridis")(Normalize(*limits)(expression[drawn])),
        atol=1e-6,
    )
    emb2.close()

    # Group means of the normalized gene, and the share of expressing cells.
    groups = [1, 2, 10]
    means = [expression[data.labels == group].mean() for group in groups]
    fractions = [(expression[data.labels == group] > 0).mean() for group in groups]
    dp = splt.dotplot(
        data.store,
        features=["CD3E"],
        groups=data.clusters,
        show=False,
    )
    aggregate = dp.tables["aggregate"]
    assert aggregate["group"].tolist() == groups
    np.testing.assert_allclose(aggregate["mean"], means, rtol=1e-5)
    np.testing.assert_allclose(aggregate["fraction"], fractions)
    assert aggregate["n_cells"].tolist() == [4, 4, 4]
    dots = next(iter(dp.axes.values())).collections[0]
    np.testing.assert_allclose(dots.get_array(), means, rtol=1e-5)
    assert dp.provenance.n_cells == 12
    assert dp.figure.legends
    dp.close()

    mp = splt.matrixplot(
        data.store,
        features=["CD3E"],
        groups=data.clusters,
        show=False,
    )
    np.testing.assert_allclose(
        mp.axes["matrixplot"].images[0].get_array(), [means], rtol=1e-5
    )
    assert mp.provenance.n_cells == 12
    mp.close()


def plt_colormap(name):
    import matplotlib

    return matplotlib.colormaps[name]


def _artifact_color_cache(monkeypatch, values, *, kind, grouping_indices=None):
    import importlib

    embedding_module = importlib.import_module("scarf.plotting.embedding")
    ref = ArtifactRef(
        scope="datastore",
        kind=kind,
        artifact_id="a" * 64,
    )
    indices = np.asarray([0, 2, 4], dtype=np.int64)
    resolved_indices = indices if grouping_indices is None else grouping_indices
    monkeypatch.setattr(
        embedding_module,
        "_resolve_grouping",
        lambda *_args, **_kwargs: (("groups",), resolved_indices, [values], None),
    )
    cache = embedding_module._prefetch_colors(
        object(),
        [ref],
        metadata_columns=(),
        from_assay=None,
        cell_key="I",
        n_cells=len(indices),
        normalization=splt.NormalizationSpec(),
        cell_indices=indices,
    )
    return ref, indices, cache[0]


def test_embedding_classifies_float_artifact_values_as_continuous(monkeypatch):
    values = np.asarray([0.1, 0.5, 0.9], dtype=np.float64)
    ref, _indices, (loaded, label, is_categorical, is_uniform) = _artifact_color_cache(
        monkeypatch, values, kind="membership_strength"
    )

    np.testing.assert_array_equal(loaded, values)
    assert label == ref.kind
    assert is_categorical is False
    assert is_uniform is False


def test_embedding_artifact_color_requires_exact_layout_selection(monkeypatch):
    with pytest.raises(ValueError, match="different order"):
        _artifact_color_cache(
            monkeypatch,
            np.asarray([0.1, 0.5, 0.9], dtype=np.float64),
            kind="membership_strength",
            grouping_indices=np.asarray([0, 4, 2], dtype=np.int64),
        )


@pytest.mark.parametrize(
    ("values", "kind"),
    [
        (np.asarray([0, 1, 1], dtype=np.int64), "cluster_labels"),
        (np.asarray(["a", "b", "a"]), "hto_identity"),
    ],
)
def test_embedding_classifies_discrete_artifact_values_as_categorical(
    monkeypatch,
    values,
    kind,
):
    ref, _indices, (loaded, label, is_categorical, is_uniform) = _artifact_color_cache(
        monkeypatch, values, kind=kind
    )

    np.testing.assert_array_equal(loaded, values)
    assert label == ref.kind
    assert is_categorical is True
    assert is_uniform is False


def test_resolve_feature_reports_missing_and_ambiguous_names(synthetic_plot_store):
    from scarf.plotting._data import resolve_feature

    with pytest.raises(
        KeyError,
        match="Feature '___not_a_real_feature___' not found in assay 'RNA' by 'name'",
    ):
        resolve_feature(synthetic_plot_store, "___not_a_real_feature___")

    synthetic_plot_store.RNA = _ArrayAssay(
        synthetic_plot_store.RNA._values, ["GeneA", "GENEA"]
    )
    # Names match case-insensitively, so both features answer "genea".
    with pytest.raises(
        ValueError, match=r"matches 2 entries in assay 'RNA' at indices \[0, 1\]"
    ):
        resolve_feature(synthetic_plot_store, "genea")
    pooled = resolve_feature(
        synthetic_plot_store, splt.FeatureRef("genea", reduction="mean")
    )
    assert pooled.indices == (0, 1)
    assert pooled.label == "genea:mean"


def test_caller_owned_target(plot_artifacts):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    data = plot_artifacts
    fig, ax = plt.subplots()
    result = splt.embedding(
        data.store,
        layout=data.layout,
        target=ax,
        show=False,
    )
    assert result.owns_figure is False
    assert result.figure is fig
    assert list(result.axes.values()) == [ax]
    (points,) = ax.collections
    np.testing.assert_allclose(points.get_offsets(), data.coordinates)
    assert {to_hex(color) for color in points.get_facecolors()} == {"#4682b4"}
    result.close()  # must not close foreign figure
    assert plt.fignum_exists(fig.number)

    grouped = splt.embedding(
        data.store,
        layout=data.layout,
        color_by=data.clusters,
        target=ax,
        show=False,
    )
    legend = ax.get_legend()
    assert [text.get_text() for text in legend.get_texts()] == ["1", "2", "10"]
    assert fig.legends == []
    grouped.close()
    plt.close(fig)


def test_summary_and_composition_accept_foreign_targets(plot_artifacts):
    import matplotlib.pyplot as plt

    data = plot_artifacts
    fig, axes = plt.subplots(1, 3)
    results = [
        splt.dotplot(
            data.store,
            features=["CD3E"],
            groups=data.clusters,
            target=axes[0],
            show=False,
        ),
        splt.matrixplot(
            data.store,
            features=["CD3E"],
            groups=data.clusters,
            target=axes[1],
            show=False,
        ),
        splt.composition(
            data.store,
            categories=data.clusters,
            kind="stacked",
            target=axes[2],
            show=False,
        ),
    ]
    assert all(result.owns_figure is False for result in results)
    for result, axis in zip(results, axes, strict=True):
        assert result.figure is fig
        assert list(result.axes.values()) == [axis]
    # Each cluster holds a third of the cells in the one stacked bar.
    np.testing.assert_allclose(
        results[2].tables["aggregate"]["proportion"], [1 / 3] * 3
    )
    assert fig.legends == []
    for result in results:
        result.close()
    assert plt.fignum_exists(fig.number)
    plt.close(fig)


def test_dotplot_weights_each_sample_equally(synthetic_plot_store):
    result = splt.dotplot(
        synthetic_plot_store,
        features=["GeneA"],
        group_by="group",
        sample_by="sample_with_missing",
        show=False,
    )

    # GeneA is 0, 0.5, ..., 5.5 over the cells; cell 0 has no sample. Each
    # group averages its per-sample means: group1 = (4 + 5) / 2, group2 =
    # (2.25 + 3.25) / 2 and group10 = (0.75 + 1.5) / 2. Weighting by cells
    # would give group1 4.75 and group10 1.0 instead.
    aggregate = result.tables["aggregate"].set_index("group")
    np.testing.assert_allclose(
        aggregate.loc[["group1", "group2", "group10"], "mean"], [4.5, 2.75, 1.125]
    )
    assert aggregate.loc[["group1", "group2", "group10"], "n_samples"].tolist() == [
        2,
        2,
        2,
    ]
    assert result.provenance.n_samples == 4
    assert result.provenance.extras["dropped_sample_cells"] == 1
    result.close()


def _colorbar_limits(figure):
    return [
        (axis._colorbar.norm.vmin, axis._colorbar.norm.vmax)
        for axis in figure.axes
        if getattr(axis, "_colorbar", None) is not None
    ]


def test_facet_shared_color_limits(synthetic_plot_store):
    from matplotlib.colors import Normalize

    store = synthetic_plot_store
    score = store.cells.fetch_all("plot_score")
    low = store.cells.fetch_all("plot_condition") == "low"
    viridis = plt_colormap("viridis")
    result = splt.embedding(
        store,
        layout_key="plot_layout",
        color_by=splt.CellField("plot_score", kind="continuous"),
        facet_by="plot_condition",
        facet_order=["low", "high"],
        show=False,
    )
    assert result.provenance.extras["color_limits"]["plot_score"] == pytest.approx(
        (0.0, 11.0)
    )
    # Both facets share coordinate limits and one colorbar spanning both.
    assert len(result.axes) == 2
    assert len({ax.get_xlim() for ax in result.axes.values()}) == 1
    assert len({ax.get_ylim() for ax in result.axes.values()}) == 1
    assert _colorbar_limits(result.figure) == [pytest.approx((0.0, 11.0))]
    for axis, cells in zip(result.axes.values(), (low, ~low), strict=True):
        np.testing.assert_allclose(
            _facecolor_rgba(axis.collections[0]),
            viridis(Normalize(0.0, 11.0)(score[cells])),
        )
    result.close()

    panel_result = splt.embedding(
        store,
        layout_key="plot_layout",
        color_by=splt.CellField("plot_score", kind="continuous"),
        facet_by="plot_condition",
        facet_order=["low", "high"],
        color_scale=splt.ColorScale(scope="panel"),
        show=False,
    )
    panel_limits = list(panel_result.provenance.extras["color_limits"].values())
    assert panel_limits[0] == pytest.approx((0.0, 1.0))
    assert panel_limits[1] == pytest.approx((10.0, 11.0))
    assert len(panel_result.figure.axes) == 4
    for axis, cells, limits in zip(
        panel_result.axes.values(), (low, ~low), panel_limits, strict=True
    ):
        np.testing.assert_allclose(
            _facecolor_rgba(axis.collections[0]),
            viridis(Normalize(*limits)(score[cells])),
        )
    panel_result.close()

    shared_result = splt.embedding(
        store,
        layout_key="plot_layout",
        color_by=[
            splt.CellField("plot_score", kind="continuous"),
            splt.CellField("plot_score_scaled", kind="continuous"),
        ],
        color_scale=splt.ColorScale(scope="shared"),
        show=False,
    )
    shared_limits = list(shared_result.provenance.extras["color_limits"].values())
    assert shared_limits == [pytest.approx((0.0, 110.0))] * 2
    shared_result.close()


def test_composition_and_export(synthetic_plot_store, tmp_path):
    from PIL import Image

    result = splt.composition(
        synthetic_plot_store,
        category_by="category_complete",
        sample_by="sample",
        kind="per_sample",
        show=False,
    )
    # Samples hold categories B, A, B or A, B, A in turn.
    per_sample = result.tables["per_sample"]
    assert per_sample["sample"].tolist() == ["s1", "s2", "s3", "s4"] * 2
    assert per_sample["category"].tolist() == ["A"] * 4 + ["B"] * 4
    np.testing.assert_allclose(
        per_sample["proportion"],
        [1 / 3, 2 / 3, 1 / 3, 2 / 3, 2 / 3, 1 / 3, 2 / 3, 1 / 3],
    )
    assert per_sample["n_cells"].tolist() == [1, 2, 1, 2, 2, 1, 2, 1]
    width, height = result.figure.get_size_inches()
    out = result.save(tmp_path / "composition.png", dpi=100)
    with Image.open(out) as image:
        assert image.size == (round(width * 100), round(height * 100))
    result.close()

    stacked = splt.composition(
        synthetic_plot_store,
        category_by="category_complete",
        sample_by="sample",
        show=False,
    )
    width, height = stacked.figure.get_size_inches()
    pdf = stacked.save(tmp_path / "composition.pdf", exact_size=True)
    payload = pdf.read_bytes()
    assert payload.startswith(b"%PDF-")
    media_box = f"/MediaBox [ 0 0 {width * 72:g} {height * 72:g} ]".encode()
    assert media_box in payload
    stacked.close()


def test_feature_plotting_uses_assay_normalization_adapter(plot_artifacts, monkeypatch):
    from matplotlib.colors import Normalize

    data = plot_artifacts
    assay = data.store.RNA
    native_normed = assay.normed
    calls = []

    def tracked_normed(*args, **kwargs):
        calls.append(kwargs)
        return native_normed(*args, **kwargs)

    monkeypatch.setattr(assay, "normed", tracked_normed)
    result = splt.embedding(
        data.store,
        layout=data.layout,
        color_by="LYZ",
        normalization=splt.NormalizationSpec(transform="log1p"),
        sort_values=True,
        show=False,
    )
    (points,) = result.axes["LYZ"].collections
    values = np.log1p(data.normalized[:, 2])
    drawn = np.asarray(points.get_offsets())[:, 0].astype(int)
    np.testing.assert_allclose(
        _facecolor_rgba(points),
        plt_colormap("viridis")(Normalize(values.min(), values.max())(values[drawn])),
        atol=1e-5,
    )
    result.close()
    assert len(calls) == 1
    assert calls[0]["log_transform"] is True
    assert set(calls[0]) == {"cell_idx", "feat_idx", "log_transform"}


def _assert_plotting_fetch_matches_assay_normed(
    datastore, assay_name, requested_indices, cell_idx
):
    from scarf.plotting._data import (
        fetch_normalized_feature_matrix,
        resolve_feature,
    )
    from scarf.utils import controlled_compute

    assay = datastore._get_assay(assay_name)
    resolved = [
        resolve_feature(
            datastore,
            splt.FeatureRef(value=index, assay=assay_name, by="index"),
        )
        for index in requested_indices
    ]
    physical_indices = np.unique(np.asarray(requested_indices, dtype=np.int64))
    expected = controlled_compute(
        assay.normed(cell_idx=cell_idx, feat_idx=physical_indices),
        datastore.nthreads,
    ).astype(np.float64)
    local_order = np.searchsorted(
        physical_indices, np.asarray(requested_indices, dtype=np.int64)
    )
    expected = expected[:, local_order]
    fetched = fetch_normalized_feature_matrix(
        datastore,
        resolved,
        cell_idx,
        normalization=splt.NormalizationSpec(source="assay"),
    )
    np.testing.assert_allclose(fetched, expected)
    log_spec = splt.NormalizationSpec(source="assay", transform="log1p")
    if "log_transform" in applicable_normalization_flags(assay):
        logged = fetch_normalized_feature_matrix(
            datastore, resolved, cell_idx, normalization=log_spec
        )
        np.testing.assert_allclose(logged, np.log1p(expected))
    else:
        # CLR log ratios and ATAC values are never logged again.
        with pytest.raises(ValueError, match="does not support log_transform"):
            fetch_normalized_feature_matrix(
                datastore, resolved, cell_idx, normalization=log_spec
            )


def test_plotting_fetch_matches_rna_normed(datastore):
    cell_idx = datastore.cells.active_index("I")[:32]
    _assert_plotting_fetch_matches_assay_normed(
        datastore,
        "RNA",
        [1, 0],
        cell_idx,
    )


def test_plotting_fetch_matches_adt_normed(toy_crdir_ds):
    cell_idx = np.arange(toy_crdir_ds.cells.N, dtype=np.int64)
    _assert_plotting_fetch_matches_assay_normed(
        toy_crdir_ds,
        "ADT",
        [1, 0],
        cell_idx,
    )


def test_plotting_fetch_matches_atac_normed(atac_datastore):
    cell_idx = atac_datastore.cells.active_index("I")[:32]
    _assert_plotting_fetch_matches_assay_normed(
        atac_datastore,
        "ATAC",
        [1, 0],
        cell_idx,
    )


@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_plotting_fetch_preserves_assay_groups_order_and_reduction(
    toy_crdir_ds, reduction
):
    from dataclasses import replace

    from scarf.plotting._data import (
        fetch_normalized_feature_matrix,
        resolve_feature,
    )
    from scarf.utils import controlled_compute

    cell_idx = np.arange(toy_crdir_ds.cells.N, dtype=np.int64)
    rna = [
        resolve_feature(
            toy_crdir_ds,
            splt.FeatureRef(value=index, assay="RNA", by="index"),
        )
        for index in (0, 1)
    ]
    adt = [
        resolve_feature(
            toy_crdir_ds,
            splt.FeatureRef(value=index, assay="ADT", by="index"),
        )
        for index in (0, 1)
    ]
    reduced_rna = replace(
        rna[0],
        indices=(0, 1),
        ids=rna[0].ids + rna[1].ids,
        names=rna[0].names + rna[1].names,
        reduction=reduction,
    )
    fetched = fetch_normalized_feature_matrix(
        toy_crdir_ds,
        [adt[1], reduced_rna, adt[0]],
        cell_idx,
    )
    rna_native = controlled_compute(
        toy_crdir_ds.RNA.normed(
            cell_idx=cell_idx,
            feat_idx=np.asarray([0, 1], dtype=np.int64),
        ),
        toy_crdir_ds.nthreads,
    )
    adt_native = controlled_compute(
        toy_crdir_ds.ADT.normed(
            cell_idx=cell_idx,
            feat_idx=np.asarray([0, 1], dtype=np.int64),
        ),
        toy_crdir_ds.nthreads,
    )
    expected = np.column_stack(
        (adt_native[:, 1], getattr(rna_native, reduction)(axis=1), adt_native[:, 0])
    )
    np.testing.assert_allclose(fetched, expected)


def test_normalization_spec_supports_raw_and_log1p(datastore):
    from scarf.plotting._data import (
        fetch_normalized_feature_matrix,
        resolve_feature,
    )
    from scarf.utils import controlled_compute

    assay = datastore.RNA
    cell_idx = datastore.cells.active_index("I")
    gene = str(assay.feats.fetch_all("names")[0])
    resolved = [resolve_feature(datastore, gene)]
    raw = fetch_normalized_feature_matrix(
        datastore,
        resolved,
        cell_idx,
        normalization=splt.NormalizationSpec(source="raw"),
    )
    expected_raw = controlled_compute(
        assay.rawData[:, [resolved[0].indices[0]]][cell_idx, :],
        datastore.nthreads,
    )
    normalized = fetch_normalized_feature_matrix(
        datastore,
        resolved,
        cell_idx,
        normalization=splt.NormalizationSpec(),
    )
    raw_logged = fetch_normalized_feature_matrix(
        datastore,
        resolved,
        cell_idx,
        normalization=splt.NormalizationSpec(source="raw", transform="log1p"),
    )
    logged = fetch_normalized_feature_matrix(
        datastore,
        resolved,
        cell_idx,
        normalization=splt.NormalizationSpec(transform="log1p"),
    )
    assert np.array_equal(raw, expected_raw)
    assert np.allclose(raw_logged, np.log1p(raw))
    assert np.allclose(logged, np.log1p(normalized))


def test_figsize_rejected_with_caller_owned_target(synthetic_plot_store):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    with pytest.raises(
        ValueError, match="^figsize is invalid when a caller-owned target is provided$"
    ):
        splt.embedding(
            synthetic_plot_store,
            layout_key="plot_layout",
            target=ax,
            figsize=(3, 3),
            show=False,
        )
    assert len(ax.collections) == 0
    plt.close(fig)


def test_multi_gene_by_condition_embedding(synthetic_plot_store):
    result = splt.embedding(
        synthetic_plot_store,
        layout_key="plot_layout",
        color_by=["GeneA", "GeneB"],
        facet_by="plot_condition",
        facet_order=["low", "high"],
        sort_values=True,
        show=False,
    )
    assert result.provenance.extras["n_colors"] == 2
    assert result.provenance.extras["n_facets"] == 2
    # Panel keys are (gene, condition), one row per gene.
    assert list(result.axes) == [
        ("GeneA", "low"),
        ("GeneA", "high"),
        ("GeneB", "low"),
        ("GeneB", "high"),
    ]
    # Each gene's limits span its values over both conditions.
    assert result.provenance.extras["color_limits"] == {
        "GeneA": pytest.approx((0.0, 5.5)),
        "GeneB": pytest.approx((0.0, 16.0)),
    }
    for (gene, condition), axis in result.axes.items():
        assert len(axis.collections[0].get_offsets()) == 6
        assert axis.get_title() == f"{gene} | plot_condition={condition}"
    result.close()


def _add_synthetic_embeddings(datastore, assay_name):
    coordinates = np.linspace(-1.0, 1.0, datastore.cells.N)
    layouts = [
        f"plot_{assay_name.lower()}_native_a",
        f"plot_{assay_name.lower()}_native_b",
    ]
    datastore.cells.insert(f"{layouts[0]}1", coordinates, overwrite=True)
    datastore.cells.insert(f"{layouts[0]}2", coordinates**2, overwrite=True)
    datastore.cells.insert(f"{layouts[1]}1", -coordinates, overwrite=True)
    datastore.cells.insert(f"{layouts[1]}2", coordinates[::-1], overwrite=True)
    return layouts


def test_multi_layout_multi_color_embedding(synthetic_plot_store):
    layouts = ["plot_layout", "plot_mirror"]
    colors = ["metricA", "metricB"]
    expected_keys = [(layout, color) for layout in layouts for color in colors]

    result = splt.embedding(
        synthetic_plot_store,
        layout_key=layouts,
        color_by=colors,
        show=False,
    )

    assert result.owns_figure is True
    assert list(result.axes) == expected_keys
    assert result.provenance.extras["layouts"] == layouts
    assert result.provenance.extras["n_layouts"] == 2
    assert set(result.provenance.extras["layout_provenance"]) == set(layouts)
    assert "multi_layout" in result.provenance.notes
    assert len(result.scales) == 1
    assert isinstance(result.scales[0], splt.ColorScale)
    # One colorbar per color, spanning that column's values.
    assert [
        (legend.kind, legend.label, legend.extras) for legend in result.legends
    ] == [
        ("colorbar", "metricA", {"vmin": 1.0, "vmax": 11.0}),
        ("colorbar", "metricB", {"vmin": 20.0, "vmax": 31.0}),
    ]
    for (layout, _color), axis in result.axes.items():
        offsets = np.asarray(axis.collections[0].get_offsets())
        np.testing.assert_allclose(
            offsets[:, 0],
            synthetic_plot_store.cells.fetch_all(f"{layout}1"),
        )
    result.close()


@pytest.mark.parametrize(
    ("fixture_name", "assay_name"),
    [
        pytest.param("datastore", "RNA", id="rna"),
        pytest.param("toy_crdir_ds", "ADT", id="adt"),
        pytest.param("atac_datastore", "ATAC", id="atac"),
    ],
)
def test_multi_layout_embedding_uses_native_feature_values(
    request,
    fixture_name,
    assay_name,
):
    from scarf.utils import controlled_compute

    datastore = request.getfixturevalue(fixture_name)
    layouts = _add_synthetic_embeddings(datastore, assay_name)
    label = f"{assay_name} native feature"
    feature = splt.FeatureRef(
        value=0,
        assay=assay_name,
        by="index",
        label=label,
    )
    result = splt.embedding(
        datastore,
        layout_key=layouts,
        color_by=feature,
        show=False,
    )

    assay = datastore._get_assay(assay_name)
    cell_index = datastore.cells.active_index("I")
    native_values = controlled_compute(
        assay.normed(
            cell_idx=cell_index,
            feat_idx=np.asarray([0], dtype=np.int64),
        ),
        datastore.nthreads,
    ).reshape(-1)
    expected_limits = (float(native_values.min()), float(native_values.max()))
    if expected_limits[1] <= expected_limits[0]:
        expected_limits = (expected_limits[0], expected_limits[0] + 1.0)

    assert list(result.axes) == [(layout, label) for layout in layouts]
    assert result.provenance.assay == assay_name
    assert result.provenance.extras["assays"] == [assay_name]
    for layout in layouts:
        limits = result.provenance.extras["color_limits_by_layout"][layout]
        assert limits[label] == pytest.approx(expected_limits)
    result.close()


def test_multi_layout_embedding_accepts_matching_target_axes(synthetic_plot_store):
    import matplotlib.pyplot as plt

    layouts = ["plot_layout", "plot_mirror"]
    colors = ["metricA", "metricB"]
    panel_keys = [(layout, color) for layout in layouts for color in colors]
    figure, target_axes = plt.subplots(2, 2)
    target = dict(zip(panel_keys, target_axes.ravel(), strict=True))

    result = splt.embedding(
        synthetic_plot_store,
        layout_key=layouts,
        color_by=colors,
        target=target,
        show=False,
    )

    assert result.owns_figure is False
    assert result.figure is figure
    assert result.axes == target
    assert all(len(axis.collections[0].get_offsets()) == 12 for axis in target.values())
    result.close()
    assert plt.fignum_exists(figure.number)
    plt.close(figure)


def test_embedding_show_default_suppression_and_later_show(
    synthetic_plot_store,
    monkeypatch,
):
    shown = []

    def track_show(result):
        shown.append(result)

    monkeypatch.setattr(splt.PlotResult, "show", track_show)
    default_result = splt.embedding(
        synthetic_plot_store,
        layout_key="plot_layout",
        color_by="metricA",
    )
    suppressed_result = splt.embedding(
        synthetic_plot_store,
        layout_key="plot_layout",
        color_by="metricA",
        show=False,
    )

    assert shown == [default_result]
    suppressed_result.show()
    assert shown == [default_result, suppressed_result]
    default_result.close()
    suppressed_result.close()


def test_resolve_feature_by_index(synthetic_plot_store):
    from scarf.plotting._data import ResolvedFeature, resolve_feature

    ref = splt.FeatureRef(value=1, by="index", assay="RNA")
    assert resolve_feature(synthetic_plot_store, ref) == ResolvedFeature(
        assay="RNA",
        by="index",
        indices=(1,),
        ids=("feature-1",),
        names=("GeneB",),
        label="GeneB",
        reduction=None,
        raw=ref,
    )
    with pytest.raises(
        KeyError, match=r"Feature index 2 out of range for assay 'RNA' \(N=2\)"
    ):
        resolve_feature(
            synthetic_plot_store, splt.FeatureRef(value=2, by="index", assay="RNA")
        )


def test_paired_composition_draws_subject_lines(synthetic_plot_store):
    result = splt.composition(
        synthetic_plot_store,
        category_by="paired_category",
        study_design=splt.StudyDesign(
            sample_by="sample",
            subject_by="subject",
            condition_by="condition",
        ),
        kind="per_sample",
        show=False,
    )

    per_sample = result.tables["per_sample"].drop_duplicates("sample")
    assert per_sample.set_index("sample")["subject"].to_dict() == {
        "s1": "donor1",
        "s2": "donor2",
        "s3": "donor1",
        "s4": "donor2",
    }
    assert result.provenance.extras["n_pair_lines"] == 4
    assert "paired_by=subject" in result.provenance.notes
    # Category A falls from 2/3 to 0 for donor1 and rises from 1/3 to 1 for
    # donor2; category B mirrors it. Blocks hold (control, treated) per category.
    assert _pair_lines(result.axes["composition"]) == [
        ([0.0, 1.0], [pytest.approx(2 / 3), 0.0]),
        ([0.0, 1.0], [pytest.approx(1 / 3), 1.0]),
        ([2.0, 3.0], [pytest.approx(1 / 3), 1.0]),
        ([2.0, 3.0], [pytest.approx(2 / 3), 0.0]),
    ]
    result.close()


def test_embedding_clip_and_subset(synthetic_plot_store):
    result = splt.embedding(
        synthetic_plot_store,
        layout_key="plot_layout",
        color_by="GeneB",
        clip_fraction=0.1,
        subset_by="keep",
        show=False,
    )

    # Only the first eight cells pass subset_by.
    (points,) = result.axes["GeneB"].collections
    offsets = np.asarray(points.get_offsets())
    np.testing.assert_allclose(sorted(offsets[:, 0]), np.arange(8.0))
    assert result.provenance.n_cells == 8
    assert result.provenance.extras["clip_fraction"] == 0.1
    assert result.provenance.extras["subset_by"] == "keep"
    result.close()


def test_embedding_groups_filters_categories(synthetic_plot_store):
    from matplotlib.colors import to_hex

    palette = {"group1": "#3366cc", "group2": "#dc3912", "group10": "#109618"}
    result = splt.embedding(
        synthetic_plot_store,
        layout_key="plot_layout",
        color_by="group",
        groups=["group2", "group10"],
        categorical_scale=splt.CategoricalScale(palette=palette),
        show=False,
    )

    assert result.provenance.extras["groups"] == ["group2", "group10"]
    cat_scale = next(s for s in result.scales if isinstance(s, splt.CategoricalScale))
    assert list(cat_scale.order) == ["group2", "group10"]
    # Cells 0-3 are group10 and cells 4-7 group2; group1 is not drawn.
    (points,) = result.axes["group"].collections
    offsets = np.asarray(points.get_offsets())
    colors = [to_hex(color) for color in points.get_facecolors()]
    assert dict(zip(offsets[:, 0].astype(int), colors, strict=True)) == {
        **{cell: "#109618" for cell in range(4)},
        **{cell: "#dc3912" for cell in range(4, 8)},
    }
    legend = result.figure.legends[0]
    assert [text.get_text() for text in legend.get_texts()] == ["group2", "group10"]
    result.close()


def test_distribution_violin(synthetic_plot_store, plot_artifacts):
    from matplotlib.colors import to_rgba
    from seaborn.utils import desaturate

    result = splt.distribution(
        synthetic_plot_store,
        keys=["metricA", "metricB"],
        grouping=splt.CellField("group"),
        kind="violin",
        max_points=5,
        seed=1,
        show=False,
    )
    assert list(result.axes) == ["metricA", "metricB"]
    np.testing.assert_array_equal(
        result.tables["metricA"]["value"], synthetic_plot_store.cells.fetch("metricA")
    )
    assert result.provenance.extras["approximate"] is True
    assert "subsampled_display" in result.provenance.notes
    ax = result.axes["metricA"]
    assert [tick.get_rotation() for tick in ax.get_xticklabels()] == [45.0] * 3
    # Violins take the group palette, desaturated by seaborn and faded.
    palette = result.scales[0].palette
    bodies = ax.collections[:3]
    for body, group in zip(bodies, ("group1", "group2", "group10"), strict=True):
        np.testing.assert_allclose(
            body.get_facecolor()[0], to_rgba(desaturate(palette[group], 0.9), 0.9)
        )
    result.close()

    data = plot_artifacts
    boxes = splt.distribution(
        data.store,
        keys="CD3E",
        cell_selection=data.selection,
        kind="box",
        max_points=0,
        show=False,
    )
    assert list(boxes.axes) == ["CD3E"]
    np.testing.assert_allclose(
        boxes.tables["CD3E"]["value"], data.normalized[:, 0], rtol=1e-5
    )
    assert boxes.provenance.extras["cell_selection"] == data.selection.to_dict()
    boxes.close()


def test_distribution_without_cell_selection_includes_all_cells():
    values = np.arange(6, dtype=np.float64)
    store = _ArrayStore(
        {"I": np.array([True, False] * 3), "metric": values},
        np.zeros((6, 2)),
    )

    result = splt.distribution(
        store,
        keys="metric",
        max_points=0,
        show=False,
    )

    # Without a selection every stored cell is plotted, active or not.
    assert result.provenance.extras["cell_selection"] is None
    assert result.provenance.n_cells == 6
    np.testing.assert_array_equal(result.tables["metric"]["value"], values)
    result.close()


def test_distribution_subset_and_groups(synthetic_plot_store):
    result = splt.distribution(
        synthetic_plot_store,
        keys="metricA",
        grouping=splt.CellField("group"),
        groups=["group2", "group10"],
        subset_by="keep",
        kind="box",
        max_points=0,
        show=False,
    )
    assert result.provenance.extras["subset_by"] == "keep"
    assert result.provenance.extras["groups"] == ["group2", "group10"]
    # Kept cells 0-7 belong to group10 (0-3) and group2 (4-7).
    table = result.tables["metricA"]
    np.testing.assert_array_equal(
        table["value"], [10.0, 11.0, 9.0, 10.5, 4.0, 5.0, 6.0, 5.5]
    )
    assert table["group"].tolist() == ["group10"] * 4 + ["group2"] * 4
    assert [tick.get_text() for tick in result.axes["metricA"].get_xticklabels()] == [
        "group2",
        "group10",
    ]
    assert result.provenance.n_cells == 8
    result.close()


def test_distribution_hist_and_ecdf(synthetic_plot_store, plot_artifacts):
    store = synthetic_plot_store
    values = store.cells.fetch("metricA")
    groups = store.cells.fetch("group")
    hist = splt.distribution(
        store,
        keys="metricA",
        grouping=splt.CellField("group"),
        kind="hist",
        bins=4,
        show=False,
    )
    assert hist.provenance.extras["bins"] == 4
    assert hist.provenance.extras["approximate"] is False
    ax = hist.axes["metricA"]
    # Every group shares the bin edges of all values.
    edges = np.histogram_bin_edges(values, bins=4)
    assert len(ax.patches) == 12
    for index, group in enumerate(("group1", "group2", "group10")):
        patches = ax.patches[4 * index : 4 * index + 4]
        expected, _ = np.histogram(values[groups == group], bins=edges)
        assert [patch.get_height() for patch in patches] == expected.tolist()
        np.testing.assert_allclose([patch.get_x() for patch in patches], edges[:-1])
    hist.close()

    data = plot_artifacts
    ecdf = splt.distribution(
        data.store,
        keys="RNA_nCounts",
        cell_selection=data.selection,
        kind="ecdf",
        max_points=500,
        show=False,
    )
    assert "ecdf" in ecdf.provenance.notes
    (line,) = ecdf.axes["RNA_nCounts"].lines
    totals = np.sort(data.store.cells.fetch_all("RNA_nCounts").astype(float))
    np.testing.assert_allclose(line.get_xdata(), totals)
    np.testing.assert_allclose(line.get_ydata(), np.arange(1, 13) / 12)
    ecdf.close()

    duplicates = splt.distribution(
        store,
        keys=["metricA", "metricA"],
        kind="hist",
        bins=5,
        show=False,
    )
    assert set(duplicates.tables) == {"0:metricA", "1:metricA"}
    duplicates.close()


def _ordered_group_scale():
    return splt.CategoricalScale(
        order=("group2", "group1", "group10"),
        palette={
            "group1": "#3366cc",
            "group2": "#dc3912",
            "group10": "#109618",
        },
    )


def test_grouped_violin_and_horizontal_box_follow_explicit_order(
    synthetic_plot_store,
):
    scale = _ordered_group_scale()
    violin = splt.distribution(
        synthetic_plot_store,
        keys="metricA",
        grouping=splt.CellField("group"),
        categorical_scale=scale,
        kind="violin",
        max_points=0,
        show=False,
    )
    box = splt.distribution(
        synthetic_plot_store,
        keys="metricB",
        grouping=splt.CellField("group"),
        categorical_scale=scale,
        kind="box",
        orientation="horizontal",
        max_points=0,
        show=False,
    )

    assert [tick.get_text() for tick in violin.axes["metricA"].get_xticklabels()] == [
        "group2",
        "group1",
        "group10",
    ]
    assert [tick.get_text() for tick in box.axes["metricB"].get_yticklabels()] == [
        "group2",
        "group1",
        "group10",
    ]
    assert box.axes["metricB"].get_xlabel() == "value"
    assert box.axes["metricB"].get_ylabel() == "group"
    assert violin.scales == (scale,)
    assert box.scales == (scale,)
    violin.close()
    box.close()


def test_ungrouped_ecdf_draws_a_subsample_of_at_most_max_points(plot_artifacts):
    data = plot_artifacts
    totals = data.store.cells.fetch_all("RNA_nCounts").astype(float)
    result = splt.distribution(
        data.store,
        keys="RNA_nCounts",
        cell_selection=data.selection,
        kind="ecdf",
        max_points=5,
        show=False,
    )
    (line,) = result.axes["RNA_nCounts"].lines
    drawn = line.get_xdata()
    assert len(drawn) == 5
    assert np.all(np.diff(drawn) >= 0)
    # The drawn steps are a sample of the cells' own totals, without repeats.
    remaining = list(totals)
    for value in drawn:
        remaining.remove(value)
    np.testing.assert_allclose(line.get_ydata(), np.arange(1, 6) / 5)
    assert "subsampled_display" in result.provenance.notes
    assert result.provenance.extras["approximate"] is True
    result.close()


def test_split_violin_dodges_points_by_split_level(synthetic_plot_store):
    from matplotlib.collections import PathCollection

    store = synthetic_plot_store
    values = store.cells.fetch("metricA")
    groups = store.cells.fetch("group")
    split = store.cells.fetch("split")
    result = splt.distribution(
        store,
        keys="metricA",
        grouping=splt.CellField("group"),
        split_by="split",
        kind="violin",
        max_points=1000,
        show=False,
    )
    ax = result.axes["metricA"]
    points = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert len(points) == 6
    for position, group in enumerate(("group1", "group2", "group10")):
        for offset, (side, level) in enumerate(((-1, "left"), (1, "right"))):
            drawn = np.asarray(points[2 * position + offset].get_offsets())
            expected = np.sort(values[(groups == group) & (split == level)])
            np.testing.assert_allclose(np.sort(drawn[:, 1]), expected)
            # Dodging puts each level's points on its own side of the group.
            assert np.all(np.sign(drawn[:, 0] - position) == side)
    # The points add no legend entries of their own.
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        "left",
        "right",
    ]
    result.close()


def test_grouped_ecdf_uses_order_palette_and_probability_limits(
    synthetic_plot_store,
):
    scale = _ordered_group_scale()
    result = splt.distribution(
        synthetic_plot_store,
        keys="metricA",
        grouping=splt.CellField("group"),
        categorical_scale=scale,
        kind="ecdf",
        max_points=0,
        show=False,
    )

    axis = result.axes["metricA"]
    assert [line.get_label() for line in axis.lines] == [
        "group2",
        "group1",
        "group10",
    ]
    assert [matplotlib.colors.to_hex(line.get_color()) for line in axis.lines] == [
        scale.palette[group] for group in scale.order
    ]
    # Each group's four sorted values rise in steps of a quarter.
    assert [line.get_xdata().tolist() for line in axis.lines] == [
        [4.0, 5.0, 5.5, 6.0],
        [1.0, 2.0, 2.5, 3.0],
        [9.0, 10.0, 10.5, 11.0],
    ]
    for line in axis.lines:
        np.testing.assert_allclose(line.get_ydata(), [0.25, 0.5, 0.75, 1.0])
    assert axis.get_ylim() == pytest.approx((-0.02, 1.02))
    result.close()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        pytest.param({"kind": "density"}, "kind must be", id="kind"),
        pytest.param({"orientation": "diagonal"}, "orientation", id="orientation"),
        pytest.param(
            {"kind": "hist", "orientation": "horizontal"},
            "orientation applies",
            id="hist-orientation",
        ),
        pytest.param(
            {"row_standardize": True},
            "row_standardize",
            id="row-standardize",
        ),
        pytest.param(
            {"kind": "hist", "share_y": True},
            "share_y applies",
            id="share-y",
        ),
        pytest.param(
            {"violin_linewidth": -0.1},
            "violin_linewidth",
            id="linewidth",
        ),
        pytest.param({"violin_alpha": 1.1}, "alpha", id="violin-alpha"),
        pytest.param({"point_alpha": -0.1}, "alpha", id="point-alpha"),
        pytest.param({"groups": ["group1"]}, "groups requires", id="groups"),
        pytest.param({"split_by": "split"}, "split_by requires", id="split"),
        pytest.param(
            {
                "grouping": splt.CellField("group"),
                "split_by": "split",
                "kind": "box",
            },
            "only for violin",
            id="split-kind",
        ),
        pytest.param(
            {"grouping": splt.CellField("group"), "split_by": "group"},
            "different columns",
            id="split-same-column",
        ),
        pytest.param({"sample_stat": "sum"}, "sample_stat", id="sample-stat"),
        pytest.param({"bins": 0}, "bins", id="bins"),
        pytest.param({"keys": []}, "non-empty", id="empty-keys"),
        pytest.param(
            {"max_figure_width": 0},
            "max_figure_width",
            id="figure-width",
        ),
        pytest.param(
            {
                "sample_by": "sample",
                "study_design": splt.StudyDesign(sample_by="other_sample"),
            },
            "conflicts",
            id="study-design-conflict",
        ),
    ],
)
def test_distribution_rejects_invalid_grouped_options(
    synthetic_plot_store,
    kwargs,
    message,
):
    options = dict(kwargs)
    keys = options.pop("keys", "metricA")
    with pytest.raises(ValueError, match=message):
        splt.distribution(
            synthetic_plot_store,
            keys=keys,
            show=False,
            **options,
        )


_CELL_CYCLE_REF = ArtifactRef(
    scope="assay",
    assay="RNA",
    kind="cell_cycle",
    artifact_id="c" * 64,
)


@pytest.mark.parametrize(
    ("keys", "kwargs", "error", "message"),
    [
        pytest.param(
            _CELL_CYCLE_REF,
            {"from_assay": "RNA"},
            ValueError,
            "from_assay cannot be used with artifact-backed keys",
            id="artifact-from-assay",
        ),
        pytest.param(
            _CELL_CYCLE_REF,
            {"normalization": splt.NormalizationSpec()},
            ValueError,
            "normalization cannot be used with artifact-backed keys",
            id="artifact-normalization",
        ),
        pytest.param(
            "metricA",
            {"color_by": "median"},
            ValueError,
            "color_by must be 'group' or 'mean'",
            id="color-by",
        ),
        pytest.param(
            "metricA",
            {"color_by": "mean", "grouping": splt.CellField("group")},
            ValueError,
            "color_by='mean' is available only for stacked_violin",
            id="mean-color-kind",
        ),
        pytest.param(
            "metricA",
            {"color_by": "mean", "kind": "stacked_violin"},
            ValueError,
            "color_by='mean' requires grouping",
            id="mean-color-grouping",
        ),
        pytest.param(
            "metricA",
            {"grouping": "group"},
            TypeError,
            "grouping must be an ArtifactRef, CellField, or None",
            id="grouping-type",
        ),
        pytest.param(
            "metricA",
            {"cell_selection": "I"},
            TypeError,
            "cell_selection must be an ArtifactRef or None",
            id="cell-selection-type",
        ),
        pytest.param(
            "metricA",
            {"stats_results": object()},
            ValueError,
            "stats_results requires grouping",
            id="stats-grouping",
        ),
        pytest.param(
            "metricA",
            {
                "grouping": splt.CellField("group"),
                "kind": "hist",
                "stats_results": object(),
            },
            ValueError,
            "stats annotation applies only to violin, stacked_violin, and box plots",
            id="stats-kind",
        ),
        pytest.param(
            "metricA",
            {"sample_by": "invalid_sample"},
            ValueError,
            "No cells remain after distribution selections",
            id="no-valid-samples",
        ),
    ],
)
def test_distribution_rejects_incompatible_values_and_overlays(
    synthetic_plot_store,
    keys,
    kwargs,
    error,
    message,
):
    with pytest.raises(error) as raised:
        splt.distribution(synthetic_plot_store, keys=keys, show=False, **kwargs)

    assert raised.value.args == (message,)


def test_distribution_resolves_stored_stats_before_checking_grouping(
    synthetic_plot_store,
):
    stored = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="statistical_tests",
        artifact_id="d" * 64,
    )
    requested = []

    def get_statistical_tests(ref):
        requested.append(ref)
        return object()

    synthetic_plot_store.get_statistical_tests = get_statistical_tests

    with pytest.raises(ValueError) as raised:
        splt.distribution(
            synthetic_plot_store,
            keys="metricA",
            stats_results={"metricA": stored, "metricB": object()},
            show=False,
        )

    assert raised.value.args == ("stats_results requires grouping",)
    assert requested == [stored]


def test_distribution_requires_a_complete_cell_cycle_artifact(synthetic_plot_store):
    import zarr
    from zarr.storage import MemoryStore

    synthetic_plot_store.zw = zarr.open_group(store=MemoryStore(), mode="w")

    with pytest.raises(ValueError) as raised:
        splt.distribution(synthetic_plot_store, keys=_CELL_CYCLE_REF, show=False)

    assert raised.value.args == ("Cell-cycle artifact is unavailable or incomplete",)


def test_ungrouped_horizontal_distribution_hides_category_ticks_and_titles(
    synthetic_plot_store,
):
    result = splt.distribution(
        synthetic_plot_store,
        keys="metricA",
        kind="box",
        orientation="horizontal",
        max_points=0,
        title="QC metric",
        show=False,
    )

    assert result.figure._suptitle.get_text() == "QC metric"
    assert len(result.axes["metricA"].get_yticks()) == 0
    assert result.provenance.extras["orientation"] == "horizontal"
    result.close()


@pytest.mark.parametrize("kind", ["box", "violin"])
def test_ungrouped_horizontal_distribution_draws_one_value_axis_shape(
    synthetic_plot_store,
    kind,
):
    result = splt.distribution(
        synthetic_plot_store,
        keys="metricA",
        kind=kind,
        orientation="horizontal",
        max_points=0,
        show=False,
    )

    axis = result.axes["metricA"]
    shapes = axis.patches if kind == "box" else axis.collections
    try:
        assert len(shapes) == 1
        extent = shapes[0].get_paths()[0] if kind == "violin" else shapes[0].get_path()
        value_span = extent.vertices[:, 0]
        values = synthetic_plot_store.cells.fetch("metricA")
        # A box spans the quartiles and a cut=0 violin the data range, on x.
        expected = (
            np.percentile(values, [25, 75])
            if kind == "box"
            else [values.min(), values.max()]
        )
        assert (value_span.min(), value_span.max()) == pytest.approx(tuple(expected))
    finally:
        result.close()


def test_distribution_rejects_missing_groups_and_incomplete_scales(
    synthetic_plot_store,
):
    with pytest.raises(ValueError, match="not present"):
        splt.distribution(
            synthetic_plot_store,
            keys="metricA",
            grouping=splt.CellField("group"),
            groups=["group1", "absent"],
            show=False,
        )
    with pytest.raises(ValueError, match="order is missing observed values"):
        splt.distribution(
            synthetic_plot_store,
            keys="metricA",
            grouping=splt.CellField("group"),
            categorical_scale=splt.CategoricalScale(order=("group1", "group2")),
            show=False,
        )
    with pytest.raises(KeyError, match="group2.*missing from palette"):
        splt.distribution(
            synthetic_plot_store,
            keys="metricA",
            grouping=splt.CellField("group"),
            categorical_scale=splt.CategoricalScale(
                order=("group1", "group2", "group10"),
                palette={"group1": "#111111", "group10": "#333333"},
            ),
            show=False,
        )
    with pytest.raises(ValueError, match="split_scale.order"):
        splt.distribution(
            synthetic_plot_store,
            keys="metricA",
            grouping=splt.CellField("group"),
            split_by="split",
            split_scale=splt.CategoricalScale(order=("left",)),
            show=False,
        )
    with pytest.raises(ValueError, match="exactly two observed categories"):
        splt.distribution(
            synthetic_plot_store,
            keys="metricA",
            grouping=splt.CellField("group"),
            split_by="split3",
            show=False,
        )


def test_sample_fraction_distribution_drops_missing_sample_ids(
    synthetic_plot_store,
):
    result = splt.distribution(
        synthetic_plot_store,
        keys="metricA",
        grouping=splt.CellField("group"),
        sample_by="sample_with_missing",
        sample_stat="fraction",
        expression_cutoff=5.0,
        kind="box",
        max_points=0,
        show=False,
    )

    table = result.tables["metricA"]
    assert {"sample", "group", "value", "display_value", "nCells"} <= set(table)
    assert table["value"].between(0, 1).all()
    assert result.provenance.n_samples == 4
    assert result.provenance.extras["dropped_sample_cells"] == 1
    assert result.axes["metricA"].get_ylabel() == "Sample fraction > 5"
    result.close()


def test_sample_distribution_adapter_rejects_all_missing_sample_ids():
    from scarf.plotting.distribution import _sample_aggregate

    frame = pd.DataFrame(
        {
            "sample": [None, ""],
            "group": ["a", "a"],
            "raw_value": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="valid sample value"):
        _sample_aggregate(
            frame,
            statistic="mean",
            expression_cutoff=0.0,
            split=False,
        )


def test_distribution_rejects_malformed_cell_column_lengths(
    synthetic_plot_store,
    monkeypatch,
):
    original_get_array = synthetic_plot_store.cells._get_array

    class MalformedArray:
        def __init__(self, values):
            self.values = values
            self.shape = values.shape

        def __getitem__(self, item):
            return self.values[item][:-1]

    cases = [
        (
            "group",
            {"grouping": splt.CellField("group")},
            "Grouping metadata does not align",
        ),
        (
            "split",
            {"grouping": splt.CellField("group"), "split_by": "split"},
            "split_by length",
        ),
        (
            "sample",
            {"grouping": splt.CellField("group"), "sample_by": "sample"},
            "sample_by length",
        ),
        (
            "metricA",
            {},
            "cell selection index length does not match selected cells",
        ),
        (
            "subject",
            {
                "grouping": splt.CellField("group"),
                "study_design": splt.StudyDesign(
                    sample_by="sample",
                    subject_by="subject",
                ),
                "stats_results": SimpleNamespace(method="wilcoxon"),
            },
            "pair_by length does not match selected cells",
        ),
    ]
    for malformed_column, kwargs, message in cases:

        def malformed_get_array(column, *, _malformed=malformed_column):
            values = original_get_array(column)
            return MalformedArray(values) if column == _malformed else values

        monkeypatch.setattr(
            synthetic_plot_store.cells,
            "_get_array",
            malformed_get_array,
        )
        with pytest.raises(ValueError, match=message):
            splt.distribution(
                synthetic_plot_store,
                keys="metricA",
                max_points=0,
                show=False,
                **kwargs,
            )


def test_cell_selection_adapter_validates_masks_groups_and_natural_order():
    from scarf.plotting._data import resolve_cell_selection

    categories = np.array(["group10", "group2", "group1"], dtype=object)
    mask, order = resolve_cell_selection(3, category_values=categories)
    np.testing.assert_array_equal(mask, np.ones(3, dtype=bool))
    assert order == ["group1", "group2", "group10"]

    with pytest.raises(TypeError, match="must be boolean"):
        resolve_cell_selection(3, subset=np.array([1, 0, 1]))
    with pytest.raises(ValueError, match="length must match"):
        resolve_cell_selection(3, subset=np.array([True, False]))
    with pytest.raises(ValueError, match="category values length"):
        resolve_cell_selection(3, category_values=np.array(["a", "b"]))
    with pytest.raises(ValueError, match="groups must be non-empty"):
        resolve_cell_selection(3, category_values=categories, groups=[])
    with pytest.raises(ValueError, match="not present"):
        resolve_cell_selection(3, category_values=categories, groups=["absent"])
    with pytest.raises(ValueError, match="No cells remain"):
        resolve_cell_selection(3, subset=np.zeros(3, dtype=bool))


def test_summary_adapter_validates_group_sample_and_condition_inputs(
    synthetic_plot_store,
):
    import matplotlib.pyplot as plt

    from scarf.plotting import _data

    existing = set(plt.get_fignums())
    for plot in (splt.dotplot, splt.matrixplot):
        with pytest.raises(ValueError, match="At least one feature"):
            plot(synthetic_plot_store, features=[], group_by="group", show=False)
        for group_by in ((), ("group", "category", "condition")):
            with pytest.raises(ValueError, match="group_by must have 1 or 2 keys"):
                plot(
                    synthetic_plot_store,
                    features=["GeneA"],
                    group_by=group_by,
                    show=False,
                )
        with pytest.raises(ValueError, match="No cells with valid sample_by"):
            plot(
                synthetic_plot_store,
                features=["GeneA"],
                group_by="group",
                sample_by="invalid_sample",
                show=False,
            )
        with pytest.raises(ValueError, match="condition_by is not constant"):
            plot(
                synthetic_plot_store,
                features=["GeneA"],
                group_by="category",
                study_design=splt.StudyDesign(
                    sample_by="sample",
                    condition_by="group",
                ),
                show=False,
            )
        # Summary panels are bounded; the limits name what to reduce.
        for limit, options, message in (
            (
                "_MAX_SUMMARY_FEATURES",
                {"features": ["GeneA", "GeneB"], "group_by": "group"},
                "Too many features .*select fewer features",
            ),
            (
                "_MAX_SUMMARY_GROUPS",
                {"features": ["GeneA"], "group_by": "group"},
                "Too many groups .*group the cells more",
            ),
            (
                "_MAX_SUMMARY_SAMPLES",
                {"features": ["GeneA"], "group_by": "group", "sample_by": "sample"},
                "Too many samples .*aggregate",
            ),
        ):
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(
                    _data, limit, 1 if limit == "_MAX_SUMMARY_FEATURES" else 2
                )
                with pytest.raises(ValueError, match=message):
                    plot(synthetic_plot_store, show=False, **options)
    assert set(plt.get_fignums()) == existing


def test_composition_orders_missing_category_and_labels_segments(
    synthetic_plot_store,
):
    import matplotlib.pyplot as plt

    scale = splt.CategoricalScale(
        order=("B", "A"),
        palette={"A": "#3366cc", "B": "#dc3912"},
        missing_color="#777777",
        missing_label="Unknown",
    )
    existing_figures = set(plt.get_fignums())
    try:
        result = splt.composition(
            synthetic_plot_store,
            category_by="category",
            categorical_scale=scale,
            show_percent_labels=True,
            label_min_fraction=0.2,
            show=False,
        )

        aggregate = result.tables["aggregate"]
        assert aggregate["category"].tolist() == ["B", "A", None]
        np.testing.assert_allclose(aggregate["proportion"], np.full(3, 1 / 3))
        assert result.scales[0].order == ("B", "A")
        assert result.scales[0].missing_color == "#777777"
        assert [text.get_text() for text in result.axes["composition"].texts] == [
            "33%",
            "33%",
            "33%",
        ]
        assert [text.get_text() for text in result.figure.legends[0].get_texts()] == [
            "B",
            "A",
            "Unknown",
        ]
        figure_number = result.figure.number
        result.close()
        assert not plt.fignum_exists(figure_number)
    finally:
        for figure_number in set(plt.get_fignums()) - existing_figures:
            plt.close(figure_number)


def test_per_sample_composition_preserves_missing_categories(
    synthetic_plot_store,
):
    scale = splt.CategoricalScale(
        order=("B", "A"),
        palette={"A": "#3366cc", "B": "#dc3912"},
        missing_color="#777777",
        missing_label="Unknown",
    )

    result = splt.composition(
        synthetic_plot_store,
        category_by="category",
        sample_by="sample",
        categorical_scale=scale,
        show=False,
    )

    assert result.tables["aggregate"]["category"].tolist() == ["B", "A", None]
    per_sample = result.tables["per_sample"]
    for _, rows in per_sample.groupby("sample", sort=False):
        assert rows["category"].tolist() == ["B", "A", None]
        np.testing.assert_allclose(rows["proportion"], np.full(3, 1 / 3))
    assert [text.get_text() for text in result.figure.legends[0].get_texts()] == [
        "B",
        "A",
        "Unknown",
    ]
    result.close()


def test_per_sample_composition_uses_foreign_panel_and_condition_summary(
    synthetic_plot_store,
):
    import matplotlib.pyplot as plt
    from matplotlib.legend import Legend

    figure, axis = plt.subplots()
    result = splt.composition(
        synthetic_plot_store,
        category_by="category_complete",
        sample_by="sample",
        condition_by="condition",
        kind="per_sample",
        uncertainty="se",
        target=axis,
        show=False,
    )

    assert result.owns_figure is False
    assert result.figure is figure
    assert set(result.tables) == {"aggregate", "per_sample", "summary"}
    assert set(result.tables["summary"]["condition"]) == {"control", "treated"}
    assert result.provenance.extras["uncertainty"] == "se"
    assert any(legend.kind == "marker" for legend in result.legends)
    assert (
        len([artist for artist in axis.get_children() if isinstance(artist, Legend)])
        == 3
    )
    result.close()
    assert plt.fignum_exists(figure.number)
    plt.close(figure)


def test_composition_summary_uncertainty_handles_singleton_groups():
    from scarf.plotting.composition import _summarize_proportions

    per_sample = pd.DataFrame(
        {
            "sample": ["s1", "s2", "s3"],
            "category": ["A", "A", "B"],
            "proportion": [0.2, 0.6, 1.0],
        }
    )
    standard_deviation = _summarize_proportions(
        per_sample,
        by_condition=False,
        uncertainty="sd",
    ).set_index("category")
    standard_error = _summarize_proportions(
        per_sample,
        by_condition=False,
        uncertainty="se",
    ).set_index("category")
    no_interval = _summarize_proportions(
        per_sample,
        by_condition=False,
        uncertainty="none",
    ).set_index("category")

    assert standard_deviation.loc["A", "mean_proportion"] == pytest.approx(0.4)
    assert standard_deviation.loc["A", "lower"] < 0.4
    assert standard_deviation.loc["A", "upper"] > 0.4
    assert standard_error.loc["A", "lower"] == pytest.approx(0.2)
    assert standard_error.loc["A", "upper"] == pytest.approx(0.6)
    assert standard_error.loc["B", "lower"] == pytest.approx(1.0)
    assert standard_error.loc["B", "upper"] == pytest.approx(1.0)
    assert no_interval["lower"].tolist() == pytest.approx([0.4, 1.0])
    assert no_interval["upper"].tolist() == pytest.approx([0.4, 1.0])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        pytest.param({"kind": "pie"}, "kind must be", id="kind"),
        pytest.param({"uncertainty": "iqr"}, "uncertainty", id="uncertainty"),
        pytest.param(
            {"kind": "stacked", "uncertainty": "sd"},
            "only for kind='per_sample'",
            id="stacked-uncertainty",
        ),
        pytest.param({"bar_width": 0}, "bar_width", id="bar-width"),
        pytest.param({"bar_gap": -0.1}, "bar_width", id="bar-gap"),
        pytest.param(
            {"segment_linewidth": -0.1},
            "segment_linewidth",
            id="segment-linewidth",
        ),
        pytest.param(
            {"label_min_fraction": 1.1},
            "label_min_fraction",
            id="label-min-fraction",
        ),
        pytest.param(
            {"kind": "per_sample"},
            "requires sample_by",
            id="per-sample-needs-sample",
        ),
        pytest.param(
            {"subject_by": "subject", "condition_by": "condition"},
            "require sample_by",
            id="subject-needs-sample",
        ),
        pytest.param(
            {"sample_by": "sample", "subject_by": "subject", "kind": "per_sample"},
            "requires condition_by",
            id="subject-needs-condition",
        ),
        pytest.param(
            {"cell_key": "none_selected"},
            "No cells selected",
            id="empty-selection",
        ),
        pytest.param(
            {"sample_by": "invalid_sample"},
            "No cells have valid values",
            id="invalid-samples",
        ),
        pytest.param(
            {
                "sample_by": "sample",
                "subject_by": "inconsistent_subject",
                "condition_by": "condition",
            },
            "not constant within sample",
            id="inconsistent-subject",
        ),
    ],
)
def test_composition_rejects_invalid_panel_inputs(
    synthetic_plot_store,
    kwargs,
    message,
):
    with pytest.raises(ValueError, match=message):
        splt.composition(
            synthetic_plot_store,
            category_by="category",
            show=False,
            **kwargs,
        )


def test_composition_rejects_category_order_missing_observed_value(
    synthetic_plot_store,
):
    with pytest.raises(ValueError, match="order is missing observed values"):
        splt.composition(
            synthetic_plot_store,
            category_by="category",
            categorical_scale=splt.CategoricalScale(order=("A",)),
            show=False,
        )


def _paired_design_store(subjects, conditions):
    """Two cells per sample: one of category A and one of category B or A."""
    n_samples = len(subjects)
    categories = np.array(
        [
            value
            for index in range(n_samples)
            for value in ("A", "B" if index % 2 else "A")
        ],
        dtype=object,
    )
    columns = {
        "I": np.ones(2 * n_samples, dtype=bool),
        "category": categories,
        "sample": np.repeat([f"s{index}" for index in range(n_samples)], 2).astype(
            object
        ),
        "subject": np.repeat(np.asarray(subjects, dtype=object), 2),
        "condition": np.repeat(np.asarray(conditions, dtype=object), 2),
    }
    return _ArrayStore(columns, np.zeros((2 * n_samples, 2)))


def _pair_lines(axis):
    from matplotlib.colors import to_rgba

    return [
        (line.get_xdata().tolist(), line.get_ydata().tolist())
        for line in axis.lines
        if to_rgba(line.get_color()) == to_rgba("#757575")
    ]


def test_paired_composition_by_pair_id_connects_only_complete_pairs():
    # s0 and s1 pair d1 across conditions; d2 and d3 each lack one condition.
    store = _paired_design_store(
        subjects=["d1", "d1", "d2", "d3"],
        conditions=["control", "treated", "control", "treated"],
    )

    result = splt.composition(
        store,
        category_by="category",
        sample_by="sample",
        pair_by="subject",
        condition_by="condition",
        kind="per_sample",
        show_summary=False,
        show=False,
    )

    per_sample = result.tables["per_sample"]
    assert per_sample.drop_duplicates("sample").set_index("sample")[
        "pair"
    ].to_dict() == {"s0": "d1", "s1": "d1", "s2": "d2", "s3": "d3"}
    assert result.provenance.notes == ("composition", "per_sample", "paired_by=pair")
    assert result.provenance.extras["n_pair_lines"] == 2
    assert result.provenance.extras["n_unpaired_samples"] == 0
    # Samples s0 (A, A) and s1 (A, B) give d1 A: 1 -> 0.5 and B: 0 -> 0.5.
    # Category blocks hold (control, treated) at x = (0, 1) for A, (2, 3) for B.
    assert _pair_lines(result.axes["composition"]) == [
        ([0.0, 1.0], [1.0, 0.5]),
        ([2.0, 3.0], [0.0, 0.5]),
    ]
    result.close()


@pytest.mark.parametrize(
    ("conditions", "subject_by", "message"),
    [
        (
            ["control", "control"],
            "subject",
            "Paired composition requires at least two condition values",
        ),
        (["", ""], None, "condition_by has no valid sample values"),
    ],
)
def test_per_sample_composition_rejects_unusable_condition_designs(
    conditions, subject_by, message
):
    store = _paired_design_store(subjects=["d1", "d2"], conditions=conditions)

    with pytest.raises(ValueError) as raised:
        splt.composition(
            store,
            category_by="category",
            sample_by="sample",
            subject_by=subject_by,
            condition_by="condition",
            kind="per_sample",
            show=False,
        )

    assert raised.value.args == (message,)


def test_per_sample_composition_places_legends_on_a_foreign_panel(
    synthetic_plot_store,
):
    import matplotlib.pyplot as plt
    from matplotlib.legend import Legend

    figure, axis = plt.subplots()
    result = splt.composition(
        synthetic_plot_store,
        category_by="category_complete",
        sample_by="sample",
        kind="per_sample",
        target=axis,
        show=False,
    )

    legends = [artist for artist in axis.get_children() if isinstance(artist, Legend)]
    assert [legend.get_title().get_text() for legend in legends] == [
        "category_complete",
        "Summary",
    ]
    assert [text.get_text() for text in legends[0].get_texts()] == ["A", "B"]
    assert figure.legends == []
    # Each of the four samples holds categories B, A, B or A, B, A.
    per_sample = result.tables["per_sample"].set_index(["sample", "category"])
    np.testing.assert_allclose(
        per_sample.loc[(["s1", "s2", "s3", "s4"], "A"), "proportion"],
        [1 / 3, 2 / 3, 1 / 3, 2 / 3],
    )
    result.close()
    plt.close(figure)


_COMPOSITION_GROUPING = ArtifactRef(
    scope="assay",
    assay="RNA",
    kind="cluster_cut",
    artifact_id="5" * 64,
)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"category_by": "category", "subject_by": "subject"},
            "grouping cannot be combined with subject, pair, or condition metadata",
        ),
        (
            {"category_by": "category", "condition_by": "condition"},
            "grouping cannot be combined with subject, pair, or condition metadata",
        ),
        (
            {"category_by": "category", "cell_key": "none_selected"},
            "cell_key cannot override an artifact's stored cell selection",
        ),
        ({}, "Provide exactly one of category_by or categories"),
    ],
)
def test_grouped_composition_rejects_conflicting_inputs(
    synthetic_plot_store, kwargs, message
):
    with pytest.raises(ValueError) as raised:
        splt.composition(
            synthetic_plot_store,
            grouping=_COMPOSITION_GROUPING,
            show=False,
            **kwargs,
        )

    assert raised.value.args == (message,)


def test_artifact_composition_rejects_inputs_it_cannot_align(
    synthetic_plot_store, monkeypatch
):
    from importlib import import_module

    module = import_module("scarf.plotting.composition")
    categories = ArtifactRef(
        scope="assay",
        assay="RNA",
        kind="cluster_cut",
        artifact_id="6" * 64,
    )

    with pytest.raises(ValueError) as raised:
        splt.composition(
            synthetic_plot_store,
            categories=categories,
            cell_key="none_selected",
            show=False,
        )
    assert raised.value.args == (
        "cell_key cannot override an artifact's stored cell selection",
    )
    with pytest.raises(ValueError) as raised:
        splt.composition(
            synthetic_plot_store,
            category_by="category",
            categories=categories,
            show=False,
        )
    assert raised.value.args == ("Provide exactly one of category_by or categories",)

    selections = {
        _COMPOSITION_GROUPING: ArtifactRef(
            scope="datastore", kind="cell_selection", artifact_id="7" * 64
        ),
        categories: ArtifactRef(
            scope="datastore", kind="cell_selection", artifact_id="7" * 64
        ),
    }
    indices = {_COMPOSITION_GROUPING: [0, 1, 2], categories: [0, 1, 2]}

    def resolve_grouping(_store, *, group_by, groups, cell_key):
        assert (group_by, cell_key) == (None, "I")
        values = np.array(["x", "y", "x"], dtype=object)
        return ("groups",), np.asarray(indices[groups]), [values], None

    monkeypatch.setattr(module, "_resolve_grouping", resolve_grouping)
    monkeypatch.setattr(
        module, "_artifact_cell_selection", lambda _store, ref: selections[ref]
    )
    result = splt.composition(
        synthetic_plot_store,
        categories=categories,
        grouping=_COMPOSITION_GROUPING,
        show=False,
    )
    # Group "x" holds cells 0 and 2 (both category "x"); group "y" holds cell 1.
    per_group = result.tables["per_group"].set_index(["sample", "category"])
    assert per_group.loc[("x", "x"), "proportion"] == 1.0
    assert per_group.loc[("y", "y"), "proportion"] == 1.0
    assert per_group.loc[("x", "y"), "proportion"] == 0.0
    result.close()

    indices[categories] = [0, 2, 1]
    with pytest.raises(ValueError) as raised:
        splt.composition(
            synthetic_plot_store,
            categories=categories,
            grouping=_COMPOSITION_GROUPING,
            show=False,
        )
    assert raised.value.args == (
        "categories and grouping select cells in a different order",
    )

    selections[categories] = ArtifactRef(
        scope="datastore", kind="cell_selection", artifact_id="8" * 64
    )
    with pytest.raises(ValueError) as raised:
        splt.composition(
            synthetic_plot_store,
            categories=categories,
            grouping=_COMPOSITION_GROUPING,
            show=False,
        )
    assert raised.value.args == (
        "categories and grouping must share the same cell selection",
    )


def test_summary_panels_use_explicit_feature_group_orders(
    synthetic_plot_store,
):
    group_order = ["group2", "group1", "group10"]
    feature_order = ["GeneA", "GeneB"]
    dot = splt.dotplot(
        synthetic_plot_store,
        features=["GeneB", "GeneA"],
        group_by="group",
        group_order=group_order,
        feature_order=feature_order,
        standardize="feature",
        color_scale=splt.ColorScale(cmap="magma", vmin=-2, vmax=2),
        size_scale=splt.SizeScale(size_min=5, size_max=50),
        show_legend=False,
        show=False,
    )
    matrix = splt.matrixplot(
        synthetic_plot_store,
        features=["GeneA", "GeneB"],
        group_by="group",
        group_order=group_order,
        feature_order=list(reversed(feature_order)),
        value="fraction",
        color_scale=splt.ColorScale(cmap="viridis", vmin=0, vmax=1),
        show_legend=False,
        show=False,
    )

    dot_axis = dot.axes["dotplot"]
    assert [tick.get_text() for tick in dot_axis.get_xticklabels()] == group_order
    assert [tick.get_text() for tick in dot_axis.get_yticklabels()] == feature_order
    assert dot.provenance.extras["group_order"] == group_order
    assert dot.provenance.extras["feature_order"] == feature_order
    # Each feature's group means are z-scored across groups (ddof=1).
    cells = pd.DataFrame(
        synthetic_plot_store.RNA._values, columns=["GeneA", "GeneB"]
    ).assign(group=synthetic_plot_store.cells.fetch("group"))
    means = cells.groupby("group")[["GeneA", "GeneB"]].mean()
    z_scores = (means - means.mean()) / means.std()
    fractions = (cells[["GeneA", "GeneB"]] > 0).groupby(cells["group"]).mean()
    dots = dot_axis.collections[0]
    drawn = {
        (group_order[int(x)], feature_order[int(y)]): (value, size)
        for (x, y), value, size in zip(
            dots.get_offsets(), dots.get_array(), dots.get_sizes(), strict=True
        )
    }
    assert set(drawn) == {(g, f) for g in group_order for f in feature_order}
    for (group, feature), (value, size) in drawn.items():
        assert value == pytest.approx(z_scores.loc[group, feature])
        assert size == pytest.approx(5 + fractions.loc[group, feature] * 45)
    assert (dots.norm.vmin, dots.norm.vmax) == (-2.0, 2.0)
    assert dots.get_cmap().name == "magma"

    matrix_table = matrix.tables["matrix"]
    assert matrix_table.index.tolist() == ["GeneB", "GeneA"]
    assert matrix_table.columns.tolist() == group_order
    expected_fractions = fractions.loc[group_order, ["GeneB", "GeneA"]].T.to_numpy()
    np.testing.assert_allclose(
        matrix_table[group_order].to_numpy(dtype=float), expected_fractions
    )
    np.testing.assert_allclose(
        matrix.axes["matrixplot"].images[0].get_array(), expected_fractions
    )
    dot.close()
    matrix.close()


def test_summary_helpers_validate_labels_and_standardization():
    from scarf.plotting.summary import (
        _group_axis_labels,
        _standardize_feature,
        _wrap_tick_labels,
    )

    assert _wrap_tick_labels(["long label"], 4) == ["long\nlabe\nl"]
    with pytest.raises(ValueError, match="label_wrap"):
        _wrap_tick_labels(["value"], 0)

    grouped = pd.DataFrame({"group": ["a"], "subgroup": ["b"]})
    assert _group_axis_labels(grouped, ("group", "subgroup")).tolist() == ["a | b"]

    values = pd.DataFrame(
        {
            "feature": ["a", "a", "b", "b", "a", "a"],
            "feature_group": ["T", "T", "T", "T", "B", "B"],
            "mean": [1.0, 3.0, 2.0, 2.0, 1.0, 3.0],
        }
    )
    standardized = _standardize_feature(values)
    # The means stay raw; the z-scores are a column of their own.
    pd.testing.assert_frame_equal(standardized[values.columns], values)
    first = standardized.loc[standardized["feature_group"] == "T"]
    # Means 1 and 3 have sample standard deviation sqrt(2).
    np.testing.assert_allclose(
        first.loc[first["feature"] == "a", "zscore"], [-(0.5**0.5), 0.5**0.5]
    )
    assert first.loc[first["feature"] == "b", "zscore"].isna().all()
    # A feature listed under two bracket groups is standardized once per group,
    # so both rows carry the same values.
    np.testing.assert_allclose(
        standardized.loc[
            (standardized["feature"] == "a") & (standardized["feature_group"] == "T"),
            "zscore",
        ],
        standardized.loc[
            (standardized["feature"] == "a") & (standardized["feature_group"] == "B"),
            "zscore",
        ],
    )


def test_summary_panels_reject_incomplete_orders_and_unsupported_scales(
    synthetic_plot_store,
):
    with pytest.raises(NotImplementedError, match="linear color scales"):
        splt.dotplot(
            synthetic_plot_store,
            features=["GeneA"],
            group_by="group",
            color_scale=splt.ColorScale(scale="log"),
            show=False,
        )
    with pytest.raises(NotImplementedError, match="linear color scales"):
        splt.matrixplot(
            synthetic_plot_store,
            features=["GeneA"],
            group_by="group",
            color_scale=splt.ColorScale(scale="symlog"),
            show=False,
        )
    with pytest.raises(ValueError, match="marker_linewidth"):
        splt.dotplot(
            synthetic_plot_store,
            features=["GeneA"],
            group_by="group",
            marker_linewidth=-0.1,
            show=False,
        )
    with pytest.raises(ValueError, match="feature_order is missing"):
        splt.dotplot(
            synthetic_plot_store,
            features=["GeneA", "GeneB"],
            group_by="group",
            feature_order=["GeneA"],
            show=False,
        )
    with pytest.raises(ValueError, match="group order is missing"):
        splt.dotplot(
            synthetic_plot_store,
            features=["GeneA"],
            group_by="group",
            group_order=["group1", "group2"],
            show=False,
        )
    with pytest.raises(ValueError, match="standardize"):
        splt.dotplot(
            synthetic_plot_store,
            features=["GeneA"],
            group_by="group",
            standardize="group",
            show=False,
        )
    with pytest.raises(ValueError, match="value must be"):
        splt.matrixplot(
            synthetic_plot_store,
            features=["GeneA"],
            group_by="group",
            value="median",
            show=False,
        )
    with pytest.raises(ValueError, match="standardize"):
        splt.matrixplot(
            synthetic_plot_store,
            features=["GeneA"],
            group_by="group",
            standardize="group",
            show=False,
        )


def test_summary_orders_reject_duplicate_labels(synthetic_plot_store):
    with pytest.raises(ValueError, match="^feature_order cannot contain duplicates$"):
        splt.matrixplot(
            synthetic_plot_store,
            features=["GeneA", "GeneB"],
            group_by="group",
            feature_order=["GeneA", "GeneA", "GeneB"],
            show=False,
        )


def test_matrixplot_on_a_caller_axis_keeps_annotations_and_sample_table(
    synthetic_plot_store,
):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    figure, ax = plt.subplots()
    result = splt.matrixplot(
        synthetic_plot_store,
        features=["GeneA", "GeneB"],
        group_by="group",
        sample_by="sample_with_missing",
        row_annotations={"panel": {"GeneA": "early", "GeneB": "late"}},
        annotation_scales={
            "panel": splt.CategoricalScale(
                order=("early", "late"),
                palette={"early": "#111111", "late": "#eeeeee"},
            )
        },
        target=ax,
        show=False,
    )

    # Annotation legends stay on the caller's axes beside the heatmap.
    legend = ax.get_legend()
    assert legend.get_title().get_text() == "Annotations"
    assert [text.get_text() for text in legend.get_texts()] == [
        "panel: early",
        "panel: late",
    ]
    assert [
        to_hex(handle.get_markerfacecolor()) for handle in legend.legend_handles
    ] == ["#111111", "#eeeeee"]
    assert figure.legends == []
    # Group means average the per-sample means, as in the dotplot.
    per_sample = result.tables["per_sample"]
    assert sorted(per_sample["sample"].unique()) == ["s1", "s2", "s3", "s4"]
    matrix = result.tables["matrix"]
    np.testing.assert_allclose(
        matrix.loc["GeneA", ["group1", "group2", "group10"]].to_numpy(dtype=float),
        [4.5, 2.75, 1.125],
    )
    result.close()
    plt.close(figure)


def test_owned_distribution_composition_and_summary_results_close_after_show(
    synthetic_plot_store,
):
    import matplotlib.pyplot as plt

    results = [
        splt.distribution(
            synthetic_plot_store,
            keys="metricA",
            grouping=splt.CellField("group"),
            kind="box",
            max_points=0,
        ),
        splt.composition(
            synthetic_plot_store,
            category_by="category_complete",
            show_legend=False,
        ),
        splt.dotplot(
            synthetic_plot_store,
            features=["GeneA"],
            group_by="group",
            show_legend=False,
        ),
        splt.matrixplot(
            synthetic_plot_store,
            features=["GeneA"],
            group_by="group",
            show_legend=False,
        ),
    ]

    assert all(result.owns_figure for result in results)
    assert all(not plt.fignum_exists(result.figure.number) for result in results)
    assert [result.provenance.notes[0] for result in results] == [
        "distribution",
        "composition",
        "dotplot",
        "matrixplot",
    ]

"""Regression tests for what plots display: order, colors, labels, and legends."""

from importlib import import_module
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import scarf.plotting as splt
from scarf.plotting._style import default_point_size, panel_area_inches
from tests.test_plotting_foundation import _ArrayAssay, _ArrayFeatures, _ArrayStore
from tests.test_plotting_modernization import _synthetic_plot_store
from tests.test_plotting_raster import _GuardedMeta

embedding_module = import_module("scarf.plotting.embedding")


def _summary_store(groups) -> _ArrayStore:
    rng = np.random.default_rng(0)
    groups = np.asarray(groups)
    return _ArrayStore(
        {"I": np.ones(len(groups), dtype=bool), "group": groups},
        rng.random((len(groups), 2)),
    )


def test_summaries_reject_implicitly_shared_feature_labels():
    store = _summary_store(np.repeat(["a", "b"], 4))
    store.ADT = _ArrayAssay(np.full((8, 2), 100.0), ["GeneA", "Other"])
    rna = store._get_assay
    store._get_assay = lambda name: store.ADT if name == "ADT" else rna(name)

    # One gene name measured in two assays must never be blended into one row,
    # whether its labels come from the feature name or are set explicitly.
    for label in (None, "shared"):
        features = [
            splt.FeatureRef("GeneA", assay=assay, label=label)
            for assay in ("RNA", "ADT")
        ]
        for plot in (splt.dotplot, splt.matrixplot):
            with pytest.raises(ValueError, match="different assays share the label"):
                plot(store, features=features, group_by="group", show=False)

    distinct = splt.dotplot(
        store,
        features=[
            splt.FeatureRef("GeneA", assay="RNA", label="GeneA RNA"),
            splt.FeatureRef("GeneA", assay="ADT", label="GeneA protein"),
        ],
        group_by="group",
        show=False,
    )
    try:
        means = distinct.tables["aggregate"].groupby("feature")["mean"].mean()
        assert means["GeneA protein"] == pytest.approx(100.0)
        assert means["GeneA RNA"] < 1.0
    finally:
        distinct.close()

    # Two features that share a gene symbol only pool when a caller labels them.
    store.RNA.feats = _ArrayFeatures(["CD4", "CD4"])
    with pytest.raises(ValueError, match="Different features share the label"):
        splt.matrixplot(
            store,
            features=[splt.FeatureRef(index, by="index") for index in (0, 1)],
            group_by="group",
            show=False,
        )


def test_matrixplot_orders_groups_like_dotplot():
    store = _summary_store(np.repeat(["group10", "group2", "group1"], 4))
    matrix = splt.matrixplot(store, features=["GeneA"], group_by="group", show=False)
    dot = splt.dotplot(store, features=["GeneA"], group_by="group", show=False)
    try:
        expected = ["group1", "group2", "group10"]
        assert list(matrix.tables["matrix"].columns[1:]) == expected
        assert dot.provenance.extras["group_order"] == expected
    finally:
        matrix.close()
        dot.close()

    numeric = _summary_store(np.repeat([10, 2, 1], 4))
    matrix = splt.matrixplot(
        numeric,
        features=["GeneA"],
        group_by="group",
        group_order=[2, 10, 1],
        show=False,
    )
    try:
        assert list(matrix.tables["matrix"].columns[1:]) == ["2", "10", "1"]
    finally:
        matrix.close()
    with pytest.raises(ValueError, match="applies only to value='mean'"):
        splt.matrixplot(
            numeric,
            features=["GeneA"],
            group_by="group",
            value="fraction",
            standardize="feature",
            show=False,
        )


def test_dotplot_keeps_a_row_for_each_bracket_of_a_shared_feature():
    store = _summary_store(np.repeat(["a", "b"], 4))
    result = splt.dotplot(
        store,
        features={"T": ["GeneA", "GeneB"], "B": ["GeneB"]},
        group_by="group",
        show=False,
    )
    try:
        ax = result.axes["dotplot"]
        assert [text.get_text() for text in ax.get_yticklabels()] == [
            "GeneA",
            "GeneB",
            "GeneB",
        ]
        labels = [text for text in ax.texts if text.get_gid() == "feature-group-label"]
        brackets = [
            line for line in ax.lines if line.get_gid() == "feature-group-bracket"
        ]
        assert [text.get_text() for text in labels] == ["T", "B"]
        assert [tuple(np.round(line.get_ydata(), 2)) for line in brackets] == [
            (-0.35, 1.35),
            (1.65, 2.35),
        ]
    finally:
        result.close()


def test_composition_orders_samples_and_conditions_naturally():
    samples = np.asarray([f"S{index}" for index in (1, 2, 3, 10, 11)])
    store = _synthetic_plot_store(
        I=np.ones(30, dtype=bool),
        sample=np.repeat(samples, 6),
        cluster=np.tile(np.repeat(["c1", "c2", "c10"], 2), 5),
        condition=np.repeat(["day2", "day10", "day2", "day10", "day2"], 6),
    )
    stacked = splt.composition(
        store, category_by="cluster", sample_by="sample", show=False
    )
    paired = splt.composition(
        store,
        category_by="cluster",
        sample_by="sample",
        condition_by="condition",
        kind="per_sample",
        show=False,
    )
    try:
        ticks = stacked.axes["composition"].get_xticklabels()
        assert [text.get_text() for text in ticks] == ["S1", "S2", "S3", "S10", "S11"]
        assert paired.legends[1].extras["values"] == ["day2", "day10"]
    finally:
        stacked.close()
        paired.close()


def test_stacked_violins_share_one_category_axis():
    rng = np.random.default_rng(0)
    store = _synthetic_plot_store(
        I=np.ones(60, dtype=bool),
        group=np.repeat(["a", "b", "c"], 20),
        m1=rng.normal(size=60),
        m2=rng.normal(size=60),
        m3=rng.normal(size=60),
    )
    options = {
        "grouping": splt.CellField("group"),
        "kind": "stacked_violin",
        "max_points": 0,
        "show": False,
    }
    vertical = splt.distribution(store, ["m1", "m2", "m3"], figsize=(4, 5), **options)
    horizontal = splt.distribution(
        store,
        ["m1", "m2"],
        orientation="horizontal",
        **options,
    )
    try:
        vertical.figure.canvas.draw()
        # A caller's figure size keeps the stacked rows in one column.
        assert [
            ax.get_subplotspec().colspan.start for ax in vertical.axes.values()
        ] == [0, 0, 0]
        assert [
            [text.get_text() for text in ax.get_xticklabels() if text.get_visible()]
            for ax in vertical.axes.values()
        ][-1] == ["a", "b", "c"]
        horizontal.figure.canvas.draw()
        axes = list(horizontal.axes.values())
        assert [ax.get_subplotspec().rowspan.start for ax in axes] == [0, 0]
        assert axes[0].yaxis.get_tick_params()["labelleft"] is True
        assert axes[1].yaxis.get_tick_params()["labelleft"] is False
    finally:
        vertical.close()
        horizontal.close()


def test_distribution_returns_the_scales_it_draws():
    rng = np.random.default_rng(0)
    store = _synthetic_plot_store(
        I=np.ones(40, dtype=bool),
        group=np.repeat(["g10", "g2", "g1", "g3"], 10),
        side=np.tile(["x", "y"], 20),
        metric=rng.normal(size=40),
    )
    options = {"grouping": splt.CellField("group"), "max_points": 0, "show": False}
    grouped = splt.distribution(store, "metric", **options)
    split = splt.distribution(store, "metric", split_by="side", **options)
    try:
        # Without a caller scale, the result still reports the drawn colors.
        (group_scale,) = grouped.scales
        ticks = grouped.axes["metric"].get_xticklabels()
        assert group_scale.order == ("g1", "g2", "g3", "g10")
        assert [text.get_text() for text in ticks] == list(group_scale.order)
        assert list(group_scale.palette) == list(group_scale.order)
        assert len(set(group_scale.palette.values())) == 4
        (split_scale,) = split.scales
        assert split_scale.order == ("x", "y")
        assert list(split_scale.palette) == ["x", "y"]
    finally:
        grouped.close()
        split.close()


def test_plot_errors_close_the_figures_they_created():
    rng = np.random.default_rng(0)
    store = _synthetic_plot_store(
        I=np.ones(30, dtype=bool),
        UMAP1=rng.normal(size=30),
        UMAP2=rng.normal(size=30),
        group=np.repeat(["a", "b", "c"], 10),
        counts=np.r_[np.zeros(5), rng.uniform(1, 10, size=25)],
        empty=np.full(30, np.nan),
        ok=rng.normal(size=30),
    )
    before = set(plt.get_fignums())
    with pytest.raises(ValueError, match="positive values"):
        splt.embedding(
            store,
            layout_key="UMAP",
            color_by="counts",
            color_scale=splt.ColorScale(scale="log", scope="panel"),
            show=False,
        )
    with pytest.raises(ValueError, match="No finite values"):
        splt.distribution(
            store,
            ["ok", "empty"],
            grouping=splt.CellField("group"),
            show=False,
        )
    assert set(plt.get_fignums()) == before


def test_embedding_draws_label_artifacts_as_categories(monkeypatch):
    from scarf.storage.artifacts import ArtifactRef

    labels = np.arange(300) % 150
    monkeypatch.setattr(
        embedding_module,
        "_resolve_grouping",
        lambda store, **_: (("groups",), np.arange(300), [labels], None),
    )
    for kind, categorical in (("cluster_labels", True), ("embedding", False)):
        ref = ArtifactRef(scope="assay", assay="RNA", kind=kind, artifact_id="a" * 64)
        ((_, _, is_categorical, _),) = embedding_module._prefetch_colors(
            None,
            [ref],
            metadata_columns=frozenset(),
            from_assay=None,
            cell_key="I",
            n_cells=300,
            normalization=splt.NormalizationSpec(),
            cell_indices=np.arange(300),
        )
        assert is_categorical is categorical


def test_mean_contours_skip_panels_without_signal():
    rng = np.random.default_rng(0)
    x = rng.normal(size=20_000)
    y = rng.normal(size=20_000)
    xlim = (float(x.min()), float(x.max()))
    ylim = (float(y.min()), float(y.max()))
    figure, ax = plt.subplots()
    try:
        embedding_module._draw_density_overlay(
            ax,
            x,
            y,
            overlay=splt.DensityOverlay(statistic="mean"),
            values=np.zeros(20_000),
            xlim=xlim,
            ylim=ylim,
            theme="notebook",
        )
        assert not ax.collections
    finally:
        plt.close(figure)


def test_multi_layout_names_layouts_and_shares_category_colors():
    rng = np.random.default_rng(0)
    clusters = np.repeat([f"c{index}" for index in range(11)], 10)
    tsne1 = rng.normal(size=110)
    tsne1[clusters == "c10"] = np.nan
    store = _synthetic_plot_store(
        I=np.ones(110, dtype=bool),
        UMAP1=rng.normal(size=110),
        UMAP2=rng.normal(size=110),
        TSNE1=tsne1,
        TSNE2=rng.normal(size=110),
        cluster=clusters,
    )
    options = {"layout_key": ["UMAP", "TSNE"], "color_by": "cluster", "show": False}
    titled = splt.embedding(store, **options)
    untitled = splt.embedding(store, show_titles=False, **options)
    try:
        assert [ax.get_title() for ax in titled.axes.values()] == [
            "UMAP | cluster",
            "TSNE | cluster",
        ]
        assert [ax.get_title() for ax in untitled.axes.values()] == ["", ""]
        palettes = [
            scale.palette
            for scale in titled.scales
            if isinstance(scale, splt.CategoricalScale)
        ]
        assert len(palettes) == 1
        assert list(palettes[0]) == [f"c{index}" for index in range(11)]
    finally:
        titled.close()
        untitled.close()


def test_composed_colorbars_keep_panel_limits_scales_and_raster_limits():
    rng = np.random.default_rng(0)
    store = _synthetic_plot_store(
        I=np.ones(80, dtype=bool),
        UMAP1=rng.normal(size=80),
        UMAP2=rng.normal(size=80),
        facet=np.repeat(["x", "y"], 40),
        score=rng.uniform(1.0, 1000.0, size=80),
    )

    figure, axes = plt.subplots(1, 2, layout="constrained")
    child = splt.embedding(
        store,
        layout_key="UMAP",
        color_by="score",
        facet_by="facet",
        color_scale=splt.ColorScale(scope="panel"),
        target=list(axes),
        show=False,
    )
    splt.compose_results(figure, [child], panel_labels=False)
    colorbars = [ax for ax in figure.axes if ax.get_label() == "<colorbar>"]
    assert len(colorbars) == 2
    plt.close(figure)

    figure, axis = plt.subplots(layout="constrained")
    child = splt.embedding(
        store,
        layout_key="UMAP",
        color_by="score",
        color_scale=splt.ColorScale(scale="log"),
        target=axis,
        show=False,
    )
    splt.compose_results(figure, [child], panel_labels=False)
    (colorbar_axis,) = [ax for ax in figure.axes if ax.get_label() == "<colorbar>"]
    assert isinstance(colorbar_axis._colorbar.norm, matplotlib.colors.LogNorm)
    plt.close(figure)

    raster_store = SimpleNamespace(
        cells=_GuardedMeta(
            {
                "I": np.ones(80, dtype=bool),
                "UMAP1": rng.normal(size=80),
                "UMAP2": rng.normal(size=80),
                "score": rng.normal(size=80),
            }
        ),
        _stored_display_metadata=lambda _column: None,
    )
    figure, axis = plt.subplots(layout="constrained")
    child = splt.embedding_raster(
        raster_store,
        layout_key="UMAP",
        color_by="score",
        pixels=16,
        target=axis,
        show=False,
    )
    splt.compose_results(figure, [child], panel_labels=False)
    assert sum(ax.get_label() == "<colorbar>" for ax in figure.axes) == 1
    plt.close(figure)


def test_composition_refreshes_embedding_point_sizes():
    rng = np.random.default_rng(0)
    store = _synthetic_plot_store(
        I=np.ones(200, dtype=bool),
        UMAP1=rng.normal(size=200),
        UMAP2=rng.normal(size=200),
        score=rng.normal(size=200),
    )
    figure, axis = plt.subplots(figsize=(4, 4), layout="constrained")
    child = splt.embedding(
        store,
        layout_key="UMAP",
        color_by="score",
        target=axis,
        show=False,
    )
    splt.compose_results(figure, [child], panel_labels=False)
    points = axis.collections[0]
    assert points.get_sizes()[0] == pytest.approx(
        default_point_size(200, panel_area=panel_area_inches(axis))
    )
    plt.close(figure)


def test_mapping_confusion_rejects_text_only_label_matches():
    evidence = pd.DataFrame(
        {
            "label": np.asarray([1, 1, 2, 2], dtype=object),
            "voteFraction": [0.9, 0.8, 0.95, 0.7],
            "isUnknown": [False] * 4,
        }
    )
    store = SimpleNamespace(get_target_label_evidence=lambda *a, **k: evidence)
    with pytest.raises(ValueError, match="only equal their transferred labels"):
        splt.mapping_confusion(
            store,
            SimpleNamespace(assay="RNA"),
            reference=None,
            reference_class_group="cell_type",
            known_labels=np.asarray(["1", "1", "2", "2"], dtype=object),
            show=False,
        )


def test_theme_applies_when_the_figure_is_created():
    rng = np.random.default_rng(0)
    store = _synthetic_plot_store(
        I=np.ones(30, dtype=bool),
        group=np.repeat(["a", "b", "c"], 10),
        metric=rng.normal(size=30),
    )
    raster_store = SimpleNamespace(
        cells=_GuardedMeta(
            {
                "I": np.ones(30, dtype=bool),
                "UMAP1": rng.normal(size=30),
                "UMAP2": rng.normal(size=30),
                "metric": rng.normal(size=30),
            }
        ),
        _stored_display_metadata=lambda _column: None,
    )
    results = [
        splt.distribution(
            store,
            "metric",
            grouping=splt.CellField("group"),
            theme="paper",
            show=False,
        ),
        splt.embedding_raster(
            raster_store,
            layout_key="UMAP",
            color_by="metric",
            pixels=16,
            theme="paper",
            show=False,
        ),
    ]
    try:
        for result in results:
            axis = next(iter(result.axes.values()))
            assert result.figure.dpi == splt.THEMES["paper"]["figure.dpi"]
            assert axis.spines["left"].get_linewidth() == pytest.approx(
                splt.THEMES["paper"]["axes.linewidth"]
            )
    finally:
        for result in results:
            result.close()

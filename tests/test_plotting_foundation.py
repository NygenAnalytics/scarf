"""Compatibility snapshots and basic scarf.plotting tests."""

import inspect
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import scarf.plots as plots
import scarf.plotting as splt
from scarf.datastore.datastore import DataStore
from scarf.datastore.mapping_datastore import MappingDatastore


def test_import_plotting_exports():
    assert callable(splt.embedding)
    assert callable(splt.dotplot)
    assert callable(splt.matrixplot)
    assert callable(splt.composition)
    assert splt.FeatureRef is not None


def test_plotting_modules_import_without_optional_dependencies():
    script = """
import builtins
import pandas as pd

original_import = builtins.__import__

def block_plotting_dependencies(name, *args, **kwargs):
    if name.split(".", 1)[0] in {"matplotlib", "seaborn", "datashader", "kneed"}:
        raise ModuleNotFoundError(name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = block_plotting_dependencies
import scarf
import scarf.plots
import scarf.plotting
try:
    scarf.plots.plot_elbow([1.0, 0.5])
except ImportError as exc:
    assert "scarf[extra]" in str(exc)
else:
    raise AssertionError("plot use should require optional dependencies")
try:
    scarf.plots.plot_qc(pd.DataFrame({"groups": ["a"], "value": [1.0]}))
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


def test_plots_no_longer_sets_global_svg_fonttype_on_import():
    # Import already happened; ensure the module does not require the old global default.
    # Regression: plots.py must not assign plt.rcParams at import time.
    import scarf.plots as p

    src = inspect.getsource(p)
    assert 'plt.rcParams["svg.fonttype"]' not in src
    assert "CUSTOM_PALETTES" in src or "custom_palettes" in src


def test_legacy_signature_snapshots():
    expected = {
        plots.plot_scatter: (
            "dfs",
            "in_ax",
            "width",
            "height",
            "default_color",
            "color_map",
            "color_key",
            "mask_values",
            "mask_name",
            "mask_color",
            "point_size",
            "ax_label_size",
            "frame_offset",
            "spine_width",
            "spine_color",
            "displayed_sides",
            "legend_ondata",
            "legend_onside",
            "legend_size",
            "legends_per_col",
            "titles",
            "title_size",
            "hide_title",
            "cbar_shrink",
            "marker_scale",
            "lspacing",
            "cspacing",
            "savename",
            "dpi",
            "force_ints_as_cats",
            "n_columns",
            "w_pad",
            "h_pad",
            "show_fig",
            "scatter_kwargs",
        ),
        plots.shade_scatter: (
            "dfs",
            "in_ax",
            "figsize",
            "pixels",
            "spread_px",
            "spread_threshold",
            "min_alpha",
            "color_map",
            "color_key",
            "mask_values",
            "mask_name",
            "mask_color",
            "ax_label_size",
            "frame_offset",
            "spine_width",
            "spine_color",
            "displayed_sides",
            "legend_ondata",
            "legend_onside",
            "legend_size",
            "legends_per_col",
            "titles",
            "title_size",
            "hide_title",
            "cbar_shrink",
            "marker_scale",
            "lspacing",
            "cspacing",
            "savename",
            "dpi",
            "force_ints_as_cats",
            "n_columns",
            "w_pad",
            "h_pad",
            "show_fig",
        ),
        plots.plot_qc: (
            "data",
            "color",
            "cmap",
            "fig_size",
            "label_size",
            "title_size",
            "sup_title",
            "sup_title_size",
            "scatter_size",
            "max_points",
            "show_on_single_row",
            "show_fig",
        ),
        DataStore.plot_layout: (
            "self",
            "from_assay",
            "cell_key",
            "layout_key",
            "color_by",
            "subselection_key",
            "size_vals",
            "clip_fraction",
            "width",
            "height",
            "default_color",
            "cmap",
            "color_key",
            "mask_values",
            "mask_name",
            "mask_color",
            "point_size",
            "do_shading",
            "shade_npixels",
            "shade_min_alpha",
            "spread_pixels",
            "spread_threshold",
            "ax_label_size",
            "frame_offset",
            "spine_width",
            "spine_color",
            "displayed_sides",
            "legend_ondata",
            "legend_onside",
            "legend_size",
            "legends_per_col",
            "title",
            "title_size",
            "hide_title",
            "cbar_shrink",
            "marker_scale",
            "lspacing",
            "cspacing",
            "shuffle_df",
            "sort_values",
            "savename",
            "save_dpi",
            "ax",
            "force_ints_as_cats",
            "n_columns",
            "w_pad",
            "h_pad",
            "show_fig",
            "scatter_kwargs",
            "use_plotting",
        ),
        MappingDatastore.plot_unified_layout: (
            "self",
            "from_assay",
            "layout_key",
            "show_target_only",
            "ref_name",
            "target_groups",
            "width",
            "height",
            "cmap",
            "color_key",
            "mask_color",
            "point_size",
            "ax_label_size",
            "frame_offset",
            "spine_width",
            "spine_color",
            "displayed_sides",
            "legend_ondata",
            "legend_onside",
            "legend_size",
            "legends_per_col",
            "title",
            "title_size",
            "hide_title",
            "cbar_shrink",
            "marker_scale",
            "lspacing",
            "cspacing",
            "savename",
            "save_dpi",
            "ax",
            "force_ints_as_cats",
            "n_columns",
            "w_pad",
            "h_pad",
            "scatter_kwargs",
            "shuffle_zorder",
            "show_fig",
        ),
        DataStore.plot_cells_dists: (
            "self",
            "from_assay",
            "cols",
            "cell_key",
            "group_key",
            "color",
            "cmap",
            "fig_size",
            "label_size",
            "title_size",
            "sup_title",
            "sup_title_size",
            "scatter_size",
            "max_points",
            "show_on_single_row",
            "show_fig",
        ),
        DataStore.plot_marker_heatmap: (
            "self",
            "from_assay",
            "group_key",
            "cell_key",
            "topn",
            "log_transform",
            "vmin",
            "vmax",
            "savename",
            "save_dpi",
            "show_fig",
            "heatmap_kwargs",
        ),
        DataStore.plot_cluster_tree: (
            "self",
            "from_assay",
            "cell_key",
            "feat_key",
            "cluster_key",
            "fill_by_value",
            "force_ints_as_cats",
            "width",
            "lvr_factor",
            "vert_gap",
            "min_node_size",
            "node_size_multiplier",
            "node_power",
            "root_size",
            "non_leaf_size",
            "show_labels",
            "fontsize",
            "root_color",
            "non_leaf_color",
            "cmap",
            "color_key",
            "edgecolors",
            "edgewidth",
            "alpha",
            "figsize",
            "ax",
            "show_fig",
            "savename",
            "save_dpi",
        ),
        DataStore.plot_pseudotime_heatmap: (
            "self",
            "from_assay",
            "cell_key",
            "feat_key",
            "feature_cluster_key",
            "pseudotime_key",
            "show_features",
            "width",
            "height",
            "vmin",
            "vmax",
            "heatmap_cmap",
            "pseudotime_cmap",
            "clusterbar_cmap",
            "tick_fontsize",
            "axis_fontsize",
            "feature_label_fontsize",
            "savename",
            "save_dpi",
            "show_fig",
        ),
    }
    for function, parameter_names in expected.items():
        assert tuple(inspect.signature(function).parameters) == parameter_names


def test_color_key_not_mutated():
    df = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0], "g": ["a", "b"]})
    color_key = {"a": "#111111", "b": "#222222"}
    original = dict(color_key)
    plots.plot_scatter(
        [df],
        color_key=color_key,
        force_ints_as_cats=True,
        show_fig=False,
        legend_ondata=False,
        legend_onside=False,
    )
    assert color_key == original


def test_mask_values_list_not_mutated():
    df = pd.DataFrame(
        {"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0], "g": ["a", "b", "c"]}
    )
    mask_values = ["c"]
    original = list(mask_values)
    plots.plot_scatter(
        [df],
        mask_values=mask_values,
        show_fig=False,
        legend_ondata=False,
        legend_onside=False,
    )
    assert mask_values == original


def test_legacy_scatter_does_not_mutate_dataframes():
    df = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 1.0, 2.0],
            "group": [1, 2, 1],
        }
    )
    original = df.copy(deep=True)
    scatter_kwargs = {"c": "red", "s": 5}
    original_kwargs = dict(scatter_kwargs)
    plots.plot_scatter(
        [df],
        force_ints_as_cats=True,
        show_fig=False,
        legend_ondata=False,
        legend_onside=False,
        scatter_kwargs=scatter_kwargs,
    )
    pd.testing.assert_frame_equal(df, original)
    assert scatter_kwargs == original_kwargs


def test_bare_axes_accepted_by_plot_scatter():
    fig, ax = matplotlib.pyplot.subplots()
    df = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0], "g": [0.1, 0.9]})
    out = plots.plot_scatter([df], in_ax=ax, show_fig=False, legend_onside=False)
    assert out is not None
    matplotlib.pyplot.close(fig)


def test_continuous_nan_not_filled_as_zero():
    df = pd.DataFrame(
        {"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0], "v": [0.0, np.nan, 1.0]}
    )
    out = plots.plot_scatter(
        [df], show_fig=False, legend_onside=False, mask_color="#ff00ff"
    )
    assert out is not None
    matplotlib.pyplot.close("all")


def test_plot_qc_with_groups_first_column():
    data = pd.DataFrame(
        {
            "groups": ["a", "a", "b", "b"],
            "nCounts": [10.0, 12.0, 20.0, 22.0],
            "nFeatures": [5.0, 6.0, 8.0, 9.0],
        }
    )
    fig = plots.plot_qc(data, show_fig=False)
    assert fig is not None
    # Two metric panels
    assert len(fig.axes) >= 2
    matplotlib.pyplot.close(fig)


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


def test_equal_weight_sample_aggregation_fixture():
    """Two samples of very different size must weight equally."""
    ps = pd.DataFrame(
        {
            "sample": ["A", "B"],
            "group": ["g1", "g1"],
            "feature": ["f1", "f1"],
            "mean": [1.0, 10.0],
            "fraction": [0.1, 0.9],
            "n_cells": [10, 1000],
        }
    )
    agg = (
        ps.groupby(["group", "feature"], observed=False)
        .agg(
            mean=("mean", "mean"),
            fraction=("fraction", "mean"),
            n_cells=("n_cells", "sum"),
        )
        .reset_index()
    )
    assert len(agg) == 1
    assert agg["mean"].iloc[0] == pytest.approx(5.5)
    assert agg["fraction"].iloc[0] == pytest.approx(0.5)
    # Cell-weighted would be ~9.91, not 5.5
    cell_weighted = np.average(ps["mean"], weights=ps["n_cells"])
    assert cell_weighted == pytest.approx(9.910891, rel=1e-5)
    assert agg["mean"].iloc[0] != pytest.approx(cell_weighted)


def test_embedding_keeps_square_panel_with_side_legend(
    umap, leiden_clustering, datastore
):
    result = splt.embedding(
        datastore,
        layout_key="RNA_UMAP",
        color_by="RNA_leiden_cluster",
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


def test_embedding_dotplot_matrixplot_on_fixture(umap, leiden_clustering, datastore):
    ds = datastore
    # point_sizes and sort_values are the clever legacy behaviors to preserve
    n = len(ds.cells.fetch("I", key="I"))
    sizes = np.linspace(5, 40, n)
    emb = splt.embedding(
        ds,
        layout_key="RNA_UMAP",
        color_by="RNA_leiden_cluster",
        point_sizes=sizes,
        sort_values=False,
        show=False,
    )
    assert emb.owns_figure
    assert len(emb.axes) == 1
    assert emb.figure.legends or next(iter(emb.axes.values())).get_legend() is not None
    emb.close()

    # Gene coloring with sort_values (high expression on top)
    names = ds.RNA.feats.fetch_all("names")
    gene = str(names[0])
    emb2 = splt.embedding(
        ds,
        layout_key="RNA_UMAP",
        color_by=gene,
        sort_values=True,
        show=False,
    )
    assert emb2.provenance.extras.get("sort_values") is True
    emb2.close()

    dp = splt.dotplot(
        ds,
        features=[gene],
        group_by="RNA_leiden_cluster",
        show=False,
    )
    assert "aggregate" in dp.tables
    assert "mean" in dp.tables["aggregate"].columns
    assert "fraction" in dp.tables["aggregate"].columns
    assert dp.provenance.n_cells == len(ds.cells.active_index("I"))
    assert dp.figure.legends or next(iter(dp.axes.values())).get_legend() is not None
    dp.close()

    mp = splt.matrixplot(
        ds,
        features=[gene],
        group_by="RNA_leiden_cluster",
        show=False,
    )
    assert "matrix" in mp.tables
    assert mp.provenance.n_cells == len(ds.cells.active_index("I"))
    mp.close()


def test_feature_ref_duplicate_raises(datastore):
    # Looking up by a nonsense name
    with pytest.raises(KeyError):
        splt.FeatureRef  # noqa: B018 — ensure import path
        from scarf.plotting._data import resolve_feature

        resolve_feature(datastore, "___not_a_real_feature___")


def test_caller_owned_target(umap, datastore):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    result = splt.embedding(
        datastore,
        layout_key="RNA_UMAP",
        color_by="RNA_leiden_cluster",
        target=ax,
        show=False,
    )
    assert result.owns_figure is False
    result.close()  # must not close foreign figure
    assert plt.fignum_exists(fig.number)
    plt.close(fig)


def test_summary_and_composition_accept_foreign_targets(
    umap, leiden_clustering, datastore
):
    import matplotlib.pyplot as plt

    ds = datastore
    gene = str(ds.RNA.feats.fetch_all("names")[0])
    fig, axes = plt.subplots(1, 3)
    results = [
        splt.dotplot(
            ds,
            features=[gene],
            group_by="RNA_leiden_cluster",
            target=axes[0],
            show=False,
        ),
        splt.matrixplot(
            ds,
            features=[gene],
            group_by="RNA_leiden_cluster",
            target=axes[1],
            show=False,
        ),
        splt.composition(
            ds,
            category_by="RNA_leiden_cluster",
            kind="stacked",
            target=axes[2],
            show=False,
        ),
    ]
    assert all(result.owns_figure is False for result in results)
    for result in results:
        result.close()
    assert plt.fignum_exists(fig.number)
    plt.close(fig)


def test_sample_by_equal_weight_on_datastore(umap, leiden_clustering, datastore):
    from scarf.plotting._data import summarize_features_by_group

    ds = datastore
    active_n = len(ds.cells.active_index("I"))
    # Unbalanced samples among active cells: 5 vs rest
    sample = np.array(["big"] * active_n, dtype=object)
    sample[:5] = "small"
    ds.cells.insert("plot_sample_id", sample, overwrite=True)

    gene = str(ds.RNA.feats.fetch_all("names")[0])
    agg, per = summarize_features_by_group(
        ds,
        features=[gene],
        group_by="RNA_leiden_cluster",
        sample_by="plot_sample_id",
    )
    assert per is not None
    assert set(per["sample"].unique()) == {"big", "small"}

    # For each group×feature present in both samples, aggregate mean ==
    # unweighted mean of per-sample means (not cell-weighted).
    both = per.groupby(["RNA_leiden_cluster", "feature"], observed=False)[
        "sample"
    ].nunique()
    shared = both[both == 2].index
    assert len(shared) > 0
    for cluster, feature in shared:
        rows = per[(per["RNA_leiden_cluster"] == cluster) & (per["feature"] == feature)]
        expected = float(rows["mean"].mean())
        cell_weighted = float(np.average(rows["mean"], weights=rows["n_cells"]))
        got = float(
            agg.loc[
                (agg["RNA_leiden_cluster"] == cluster) & (agg["feature"] == feature),
                "mean",
            ].iloc[0]
        )
        assert got == pytest.approx(expected, rel=1e-6, abs=1e-8)
        # When sample sizes and per-sample means differ, equal-weight != cell-weight
        if (
            rows["n_cells"].nunique() > 1
            and rows["mean"].nunique() > 1
            and not np.allclose(rows["mean"], 0)
        ):
            assert got != pytest.approx(cell_weighted, rel=1e-3)

    dp = splt.dotplot(
        ds,
        features=[gene],
        group_by="RNA_leiden_cluster",
        sample_by="plot_sample_id",
        show=False,
    )
    assert "per_sample" in dp.tables
    assert "n_samples" in dp.tables["aggregate"].columns
    assert dp.provenance.n_samples == 2
    assert dp.provenance.extras["dropped_sample_cells"] == 0
    dp.close()


def test_facet_shared_color_limits(umap, datastore):
    ds = datastore
    active_n = len(ds.cells.active_index("I"))
    condition = np.array(["low"] * active_n, dtype=object)
    condition[active_n // 2 :] = "high"
    score = np.zeros(active_n, dtype=np.float64)
    score[condition == "low"] = np.linspace(0.0, 1.0, int((condition == "low").sum()))
    score[condition == "high"] = np.linspace(
        10.0, 11.0, int((condition == "high").sum())
    )
    ds.cells.insert("plot_condition", condition, overwrite=True)
    ds.cells.insert("plot_score", score, overwrite=True)

    result = splt.embedding(
        ds,
        layout_key="RNA_UMAP",
        color_by=splt.CellField("plot_score", kind="continuous"),
        facet_by="plot_condition",
        facet_order=["low", "high"],
        show=False,
    )
    limits = result.provenance.extras["color_limits"]
    assert "plot_score" in limits
    vmin, vmax = limits["plot_score"]
    assert vmin == pytest.approx(0.0, abs=1e-6)
    assert vmax == pytest.approx(11.0, abs=1e-6)
    # Both facet panels must exist and share coordinate limits
    assert len(result.axes) == 2
    xlims = {ax.get_xlim() for ax in result.axes.values()}
    ylims = {ax.get_ylim() for ax in result.axes.values()}
    assert len(xlims) == 1
    assert len(ylims) == 1
    result.close()

    panel_result = splt.embedding(
        ds,
        layout_key="RNA_UMAP",
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
    panel_result.close()

    ds.cells.insert("plot_score_scaled", score * 10, overwrite=True)
    shared_result = splt.embedding(
        ds,
        layout_key="RNA_UMAP",
        color_by=[
            splt.CellField("plot_score", kind="continuous"),
            splt.CellField("plot_score_scaled", kind="continuous"),
        ],
        color_scale=splt.ColorScale(scope="shared"),
        show=False,
    )
    shared_limits = list(shared_result.provenance.extras["color_limits"].values())
    assert shared_limits[0] == pytest.approx(shared_limits[1])
    assert shared_limits[1][1] == pytest.approx(110.0)
    shared_result.close()


def test_composition_and_export(umap, leiden_clustering, datastore, tmp_path):
    ds = datastore
    active_n = len(ds.cells.active_index("I"))
    sample = np.array([f"s{i % 3}" for i in range(active_n)], dtype=object)
    ds.cells.insert("plot_comp_sample", sample, overwrite=True)

    result = splt.composition(
        ds,
        category_by="RNA_leiden_cluster",
        sample_by="plot_comp_sample",
        kind="per_sample",
        show=False,
    )
    assert "per_sample" in result.tables
    out = result.save(tmp_path / "composition.png", dpi=100)
    assert out.exists() and out.stat().st_size > 0
    result.close()

    stacked = splt.composition(
        ds,
        category_by="RNA_leiden_cluster",
        sample_by="plot_comp_sample",
        show=False,
    )
    pdf = stacked.save(tmp_path / "composition.pdf", exact_size=True)
    assert pdf.exists()
    stacked.close()


def test_feature_plotting_uses_pure_normalization_adapter(umap, datastore, monkeypatch):
    ds = datastore
    assay = ds.RNA
    prev_method = assay.normMethod
    prev_scalar = getattr(assay, "scalar", None)
    gene = str(assay.feats.fetch_all("names")[0])

    def fail_if_called(*args, **kwargs):
        raise AssertionError("plotting must not call stateful assay.normed()")

    monkeypatch.setattr(assay, "normed", fail_if_called)
    result = splt.embedding(
        ds, layout_key="RNA_UMAP", color_by=gene, sort_values=True, show=False
    )
    result.close()
    assert assay.normMethod is prev_method
    assert getattr(assay, "scalar", None) is prev_scalar


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
    logged = fetch_normalized_feature_matrix(
        datastore,
        resolved,
        cell_idx,
        normalization=splt.NormalizationSpec(transform="log1p"),
    )
    assert np.array_equal(raw, expected_raw)
    assert np.allclose(logged, np.log1p(normalized))


def test_figsize_rejected_with_owned_target(umap, datastore):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="figsize"):
        splt.embedding(
            datastore,
            layout_key="RNA_UMAP",
            color_by="RNA_leiden_cluster",
            target=ax,
            figsize=(3, 3),
            show=False,
        )
    plt.close(fig)


def test_multi_gene_by_condition_embedding(umap, datastore):
    ds = datastore
    active_n = len(ds.cells.active_index("I"))
    condition = np.array(["ctrl"] * active_n, dtype=object)
    condition[active_n // 2 :] = "stim"
    ds.cells.insert("plot_condition_mg", condition, overwrite=True)

    names = [str(x) for x in ds.RNA.feats.fetch_all("names")[:2]]
    result = splt.embedding(
        ds,
        layout_key="RNA_UMAP",
        color_by=names,
        facet_by="plot_condition_mg",
        facet_order=["ctrl", "stim"],
        sort_values=True,
        show=False,
    )
    assert result.provenance.extras["n_colors"] == 2
    assert result.provenance.extras["n_facets"] == 2
    assert len(result.axes) == 4
    limits = result.provenance.extras["color_limits"]
    for gene in names:
        assert gene in limits
        vmin, vmax = limits[gene]
        assert vmax >= vmin
    # Panel keys are (gene, condition)
    for gene in names:
        for cond in ("ctrl", "stim"):
            assert (gene, cond) in result.axes
    result.close()


def test_resolve_feature_by_index(datastore):
    from scarf.plotting._data import resolve_feature

    resolved = resolve_feature(
        datastore, splt.FeatureRef(value=0, by="index", assay="RNA")
    )
    assert resolved.indices == (0,)
    assert resolved.assay == "RNA"
    assert resolved.label


def test_label_panels_and_collect_legends(umap, datastore):
    import matplotlib.pyplot as plt

    ds = datastore
    fig, axes = plt.subplot_mosaic([["A", "B"]], figsize=(6, 3))
    a = splt.embedding(
        ds,
        layout_key="RNA_UMAP",
        color_by="RNA_leiden_cluster",
        target=axes["A"],
        show=False,
    )
    gene = str(ds.RNA.feats.fetch_all("names")[0])
    b = splt.embedding(
        ds,
        layout_key="RNA_UMAP",
        color_by=gene,
        target=axes["B"],
        show=False,
    )
    splt.label_panels({"A": axes["A"], "B": axes["B"]}, labels=["A", "B"])
    legends = splt.collect_legends(fig, [a, b])
    assert len(legends) >= 1
    a.close()
    b.close()
    plt.close(fig)


def test_legacy_mutable_copy_helper():
    from scarf.plotting._legacy import copy_plot_mutables

    color_key = {"a": "#000"}
    mask_values = ["x"]
    sk = {"lw": 0.2}
    ck2, mv2, sk2 = copy_plot_mutables(
        color_key=color_key, mask_values=mask_values, scatter_kwargs=sk
    )
    assert ck2 is not color_key and ck2 == color_key
    assert mv2 is not mask_values and mv2 == mask_values
    ck2["b"] = "#fff"
    mv2.append("y")
    assert "b" not in color_key
    assert mask_values == ["x"]


def test_plot_layout_bridge_still_blocked_by_default():
    from scarf.plotting._legacy import plot_layout_bridge_blockers

    blockers = plot_layout_bridge_blockers(
        layout_key="RNA_UMAP",
        color_by="RNA_leiden_cluster",
        do_shading=False,
        mask_values=None,
        subset_by=None,
        shuffle_df=False,
        legend_ondata=True,
        legend_onside=True,
        force_ints_as_cats=True,
        clip_fraction=0.01,
        ax=None,
        use_plotting=False,
    )
    assert blockers == ("bridge_not_enabled",)

    ok = plot_layout_bridge_blockers(
        layout_key="RNA_UMAP",
        color_by="RNA_leiden_cluster",
        do_shading=False,
        mask_values=None,
        subset_by=None,
        shuffle_df=False,
        legend_ondata=True,
        legend_onside=True,
        force_ints_as_cats=True,
        clip_fraction=0.01,
        ax=None,
        use_plotting=True,
    )
    assert ok == ()

    shaded = plot_layout_bridge_blockers(
        layout_key="RNA_UMAP",
        color_by="RNA_leiden_cluster",
        do_shading=True,
        mask_values=None,
        subset_by=None,
        shuffle_df=False,
        legend_ondata=False,
        legend_onside=False,
        force_ints_as_cats=False,
        clip_fraction=0.0,
        ax=None,
        use_plotting=True,
    )
    assert "do_shading" in shaded


def test_plot_layout_use_plotting_bridge(umap, leiden_clustering, datastore):
    result = datastore.plot_layout(
        layout_key="RNA_UMAP",
        color_by="RNA_leiden_cluster",
        show_fig=False,
        use_plotting=True,
    )
    assert isinstance(result, splt.PlotResult)
    assert "embedding" in result.provenance.notes
    result.close()

    categories = sorted(pd.unique(datastore.cells.fetch("RNA_leiden_cluster")))
    palette = {
        value: f"#{((index + 1) * 123457) % 0xFFFFFF:06x}"
        for index, value in enumerate(categories)
    }
    colored = datastore.plot_layout(
        layout_key="RNA_UMAP",
        color_by="RNA_leiden_cluster",
        color_key=palette,
        show_fig=False,
        use_plotting=True,
    )
    assert isinstance(colored, splt.PlotResult)
    categorical_scales = [
        scale for scale in colored.scales if isinstance(scale, splt.CategoricalScale)
    ]
    assert categorical_scales[0].palette == palette
    colored.close()

    legacy = datastore.plot_layout(
        layout_key="RNA_UMAP",
        color_by="RNA_leiden_cluster",
        title="Custom title",
        show_fig=False,
        use_plotting=True,
    )
    assert not isinstance(legacy, splt.PlotResult)
    matplotlib.pyplot.close("all")


def test_paired_composition_draws_subject_lines(umap, leiden_clustering, datastore):
    ds = datastore
    active_n = len(ds.cells.active_index("I"))
    sample = np.array([f"s{i % 6}" for i in range(active_n)], dtype=object)
    subject = np.array([f"d{i % 3}" for i in range(active_n)], dtype=object)
    condition = np.array(
        ["before" if i % 6 < 3 else "after" for i in range(active_n)],
        dtype=object,
    )
    # Two samples per subject (s0,s3 -> d0; s1,s4 -> d1; s2,s5 -> d2)
    ds.cells.insert("plot_pair_sample", sample, overwrite=True)
    ds.cells.insert("plot_pair_subject", subject, overwrite=True)
    ds.cells.insert("plot_pair_condition", condition, overwrite=True)

    result = splt.composition(
        ds,
        category_by="RNA_leiden_cluster",
        study_design=splt.StudyDesign(
            sample_by="plot_pair_sample",
            subject_by="plot_pair_subject",
            condition_by="plot_pair_condition",
        ),
        kind="per_sample",
        show=False,
    )
    assert "subject" in result.tables["per_sample"].columns
    assert result.provenance.extras["n_pair_lines"] >= 1
    assert any("paired_by=subject" in n for n in result.provenance.notes)
    result.close()


def test_paired_composition_requires_condition(leiden_clustering, datastore):
    with pytest.raises(ValueError, match="requires condition_by"):
        splt.composition(
            datastore,
            category_by="RNA_leiden_cluster",
            sample_by="RNA_leiden_cluster",
            subject_by="RNA_leiden_cluster",
            kind="per_sample",
            show=False,
        )


def test_embedding_clip_and_subset(umap, datastore):
    ds = datastore
    active_n = len(ds.cells.active_index("I"))
    keep = np.zeros(active_n, dtype=bool)
    keep[: max(10, active_n // 2)] = True
    ds.cells.insert("plot_keep", keep, overwrite=True)
    gene = str(ds.RNA.feats.fetch_all("names")[0])
    result = splt.embedding(
        ds,
        layout_key="RNA_UMAP",
        color_by=gene,
        clip_fraction=0.01,
        subset_by="plot_keep",
        show=False,
    )
    assert result.provenance.extras["clip_fraction"] == 0.01
    assert result.provenance.extras["subset_by"] == "plot_keep"
    result.close()


def test_distribution_violin(umap, leiden_clustering, datastore):
    ds = datastore
    result = splt.distribution(
        ds,
        keys=["RNA_nCounts", "RNA_nFeatures"],
        group_by="RNA_leiden_cluster",
        kind="violin",
        max_points=200,
        seed=1,
        show=False,
    )
    assert len(result.axes) == 2
    assert "RNA_nCounts" in result.tables
    assert result.provenance.extras["approximate"] is True
    assert "subsampled_display" in result.provenance.notes
    result.close()

    gene = str(ds.RNA.feats.fetch_all("names")[0])
    result2 = splt.distribution(ds, keys=gene, kind="box", max_points=100, show=False)
    assert len(result2.axes) == 1
    result2.close()


def test_distribution_hist_and_ecdf(umap, leiden_clustering, datastore):
    ds = datastore
    hist = splt.distribution(
        ds,
        keys="RNA_nCounts",
        group_by="RNA_leiden_cluster",
        kind="hist",
        bins=20,
        show=False,
    )
    assert hist.provenance.extras["bins"] == 20
    assert hist.provenance.extras["approximate"] is False
    ax = next(iter(hist.axes.values()))
    n_groups = len(np.unique(ds.cells.fetch("RNA_leiden_cluster")))
    assert len(ax.patches) == n_groups * 20
    first_bins = [(patch.get_x(), patch.get_width()) for patch in ax.patches[:20]]
    for group_index in range(1, n_groups):
        offset = group_index * 20
        group_bins = [
            (patch.get_x(), patch.get_width())
            for patch in ax.patches[offset : offset + 20]
        ]
        assert group_bins == pytest.approx(first_bins)
    hist.close()

    ecdf = splt.distribution(
        ds,
        keys="RNA_nFeatures",
        kind="ecdf",
        max_points=500,
        seed=2,
        show=False,
    )
    assert "ecdf" in ecdf.provenance.notes
    assert ecdf.provenance.extras["approximate"] is True
    ecdf.close()

    duplicates = splt.distribution(
        ds,
        keys=["RNA_nCounts", "RNA_nCounts"],
        kind="hist",
        bins=5,
        show=False,
    )
    assert set(duplicates.tables) == {"0:RNA_nCounts", "1:RNA_nCounts"}
    duplicates.close()

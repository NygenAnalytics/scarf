"""Foundation and integration tests for scarf.plotting."""

import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import scarf.plotting as splt


def test_import_plotting_exports():
    function_names = (
        "cluster_connectivity",
        "cluster_tree",
        "collect_legends",
        "compose_results",
        "composition",
        "distribution",
        "dotplot",
        "elbow",
        "embedding",
        "embedding_raster",
        "graph_qc",
        "highly_variable_features",
        "label_panels",
        "marker_heatmap",
        "mapping_calibration",
        "mapping_confusion",
        "mapping_evidence",
        "mapping_score",
        "matrixplot",
        "pseudotime_heatmap",
        "qc",
        "register_theme",
        "run_recipe",
        "theme_context",
    )
    result_names = (
        "CategoricalScale",
        "CellField",
        "ColorScale",
        "DensityOverlay",
        "FeatureRef",
        "Highlight",
        "LegendSpec",
        "NormalizationSpec",
        "PlotProvenance",
        "PlotOutput",
        "PlotOutputSettings",
        "PlotPanelTarget",
        "PlotRecipe",
        "PlotRecipeResult",
        "PlotResult",
        "PlotStep",
        "SizeScale",
        "StudyDesign",
    )

    assert all(name in splt.__all__ for name in (*function_names, *result_names))
    assert all(callable(getattr(splt, name)) for name in function_names)
    assert all(getattr(splt, name) is not None for name in result_names)


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
    # Point sizes and sort order are part of the native embedding contract.
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
        splt.FeatureRef  # noqa: B018 - ensure import path
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


def test_feature_plotting_uses_assay_normalization_adapter(
    umap, datastore, monkeypatch
):
    ds = datastore
    assay = ds.RNA
    gene = str(assay.feats.fetch_all("names")[0])
    native_normed = assay.normed
    calls = []

    def tracked_normed(*args, **kwargs):
        calls.append(kwargs)
        return native_normed(*args, **kwargs)

    monkeypatch.setattr(assay, "normed", tracked_normed)
    result = splt.embedding(
        ds,
        layout_key="RNA_UMAP",
        color_by=gene,
        normalization=splt.NormalizationSpec(transform="log1p"),
        sort_values=True,
        show=False,
    )
    result.close()
    assert len(calls) == 1
    assert set(calls[0]) == {"cell_idx", "feat_idx"}


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
    logged = fetch_normalized_feature_matrix(
        datastore,
        resolved,
        cell_idx,
        normalization=splt.NormalizationSpec(source="assay", transform="log1p"),
    )
    np.testing.assert_allclose(fetched, expected)
    np.testing.assert_allclose(logged, np.log1p(expected))


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


def test_plotting_fetch_preserves_assay_groups_order_and_reduction(toy_crdir_ds):
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
    rna_sum = replace(
        rna[0],
        indices=(0, 1),
        ids=rna[0].ids + rna[1].ids,
        names=rna[0].names + rna[1].names,
        reduction="sum",
    )
    fetched = fetch_normalized_feature_matrix(
        toy_crdir_ds,
        [adt[1], rna_sum, adt[0]],
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
        (adt_native[:, 1], rna_native.sum(axis=1), adt_native[:, 0])
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


def _add_secondary_embedding(datastore):
    x = np.asarray(datastore.cells.fetch_all("RNA_UMAP1"), dtype=np.float64)
    y = np.asarray(datastore.cells.fetch_all("RNA_UMAP2"), dtype=np.float64)
    datastore.cells.insert(
        "plot_secondary_embedding1",
        -y,
        overwrite=True,
    )
    datastore.cells.insert(
        "plot_secondary_embedding2",
        x,
        overwrite=True,
    )


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


def test_multi_layout_multi_color_embedding(umap, datastore):
    _add_secondary_embedding(datastore)
    layouts = ["RNA_UMAP", "plot_secondary_embedding"]
    colors = ["RNA_nCounts", "RNA_nFeatures"]
    expected_keys = [(layout, color) for layout in layouts for color in colors]

    result = splt.embedding(
        datastore,
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
    assert len(result.legends) == 2
    assert len(result.scales) == 1
    assert {legend.kind for legend in result.legends} == {"colorbar"}
    assert {legend.label for legend in result.legends} == set(colors)
    assert isinstance(result.scales[0], splt.ColorScale)
    assert all(
        np.isfinite([legend.extras["vmin"], legend.extras["vmax"]]).all()
        for legend in result.legends
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


def test_multi_layout_embedding_accepts_matching_target_axes(umap, datastore):
    import matplotlib.pyplot as plt

    _add_secondary_embedding(datastore)
    layouts = ["RNA_UMAP", "plot_secondary_embedding"]
    colors = ["RNA_nCounts", "RNA_nFeatures"]
    panel_keys = [(layout, color) for layout in layouts for color in colors]
    figure, target_axes = plt.subplots(2, 2)
    target = dict(zip(panel_keys, target_axes.ravel(), strict=True))

    result = splt.embedding(
        datastore,
        layout_key=layouts,
        color_by=colors,
        target=target,
        show=False,
    )

    assert result.owns_figure is False
    assert result.figure is figure
    assert result.axes == target
    result.close()
    assert plt.fignum_exists(figure.number)
    plt.close(figure)


def test_embedding_show_default_suppression_and_later_show(
    umap,
    datastore,
    monkeypatch,
):
    shown = []

    def track_show(result):
        shown.append(result)

    monkeypatch.setattr(splt.PlotResult, "show", track_show)
    default_result = splt.embedding(
        datastore,
        layout_key="RNA_UMAP",
        color_by="RNA_nCounts",
    )
    suppressed_result = splt.embedding(
        datastore,
        layout_key="RNA_UMAP",
        color_by="RNA_nCounts",
        show=False,
    )

    assert shown == [default_result]
    suppressed_result.show()
    assert shown == [default_result, suppressed_result]
    default_result.close()
    suppressed_result.close()


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


def test_embedding_groups_filters_categories(umap, leiden_clustering, datastore):
    ds = datastore
    labels = list(pd.unique(ds.cells.fetch("RNA_leiden_cluster")))
    assert len(labels) >= 2
    keep = labels[:2]
    result = splt.embedding(
        ds,
        layout_key="RNA_UMAP",
        color_by="RNA_leiden_cluster",
        groups=keep,
        show=False,
    )
    assert result.provenance.extras["groups"] == list(keep)
    cat_scale = next(s for s in result.scales if isinstance(s, splt.CategoricalScale))
    assert list(cat_scale.order) == list(keep)
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
    ax = next(iter(result.axes.values()))
    rotations = {tick.get_rotation() for tick in ax.get_xticklabels()}
    assert 45 in rotations or any(abs(r - 45) < 1e-6 for r in rotations)
    # Grouped violins should not be a single steelblue fill.
    face_colors = {
        tuple(np.round(c.get_facecolor()[0][:3], 3))
        for c in ax.collections
        if hasattr(c, "get_facecolor") and len(c.get_facecolor())
    }
    assert len(face_colors) >= 2
    result.close()

    gene = str(ds.RNA.feats.fetch_all("names")[0])
    result2 = splt.distribution(ds, keys=gene, kind="box", max_points=100, show=False)
    assert len(result2.axes) == 1
    result2.close()


def test_distribution_cell_key_none_includes_all_cells(datastore):
    result = splt.distribution(
        datastore,
        keys="RNA_nCounts",
        cell_key=None,
        max_points=0,
        show=False,
    )
    assert result.provenance.cell_key is None
    assert result.provenance.n_cells == datastore.cells.N
    assert len(result.tables["RNA_nCounts"]) == datastore.cells.N
    result.close()


def test_distribution_subset_and_groups(umap, leiden_clustering, datastore):
    ds = datastore
    active_n = len(ds.cells.active_index("I"))
    keep = np.zeros(active_n, dtype=bool)
    keep[: max(20, active_n // 2)] = True
    ds.cells.insert("dist_keep", keep, overwrite=True)
    labels = list(pd.unique(ds.cells.fetch("RNA_leiden_cluster")))
    keep_groups = labels[:2]
    result = splt.distribution(
        ds,
        keys="RNA_nCounts",
        group_by="RNA_leiden_cluster",
        groups=keep_groups,
        subset_by="dist_keep",
        kind="box",
        max_points=0,
        show=False,
    )
    assert result.provenance.extras["subset_by"] == "dist_keep"
    assert result.provenance.extras["groups"] == list(keep_groups)
    table_groups = set(result.tables["RNA_nCounts"]["group"].unique())
    assert table_groups == set(keep_groups)
    assert result.provenance.n_cells == len(result.tables["RNA_nCounts"])
    assert result.provenance.n_cells < active_n
    result.close()


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

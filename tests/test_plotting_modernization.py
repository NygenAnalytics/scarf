"""Focused behavior tests for publication plotting features."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import scarf.plotting as splt
from scarf.plotting._style import default_point_size, resolve_legend_loc


def test_point_size_uses_population_and_panel_area():
    assert default_point_size(500, panel_area=16) > default_point_size(
        5_000,
        panel_area=16,
    )
    assert default_point_size(5_000, panel_area=16) > default_point_size(
        5_000,
        panel_area=4,
    )
    assert resolve_legend_loc(80) == "right"


def test_register_theme_extends_rcparams():
    name = "test-publication-theme"
    splt.register_theme(name, {"font.size": 7.5}, base="paper")
    with pytest.raises(ValueError, match="already exists"):
        splt.register_theme(name, {"font.size": 8})
    with splt.theme_context(name):
        assert matplotlib.rcParams["font.size"] == pytest.approx(7.5)


def test_stored_display_metadata_does_not_hide_malformed_stores():
    from scarf.plotting._display import stored_display_metadata

    class MissingPlotStore:
        pass

    class MalformedPlotStore:
        zw = {}

    assert stored_display_metadata(MissingPlotStore(), "group") is None
    with pytest.raises(KeyError, match="cellData"):
        stored_display_metadata(MalformedPlotStore(), "group")


def test_embedding_density_highlight_and_labeled_colorbar(umap, datastore):
    n = len(datastore.cells.active_index("I"))
    highlighted = np.zeros(n, dtype=bool)
    highlighted[: max(3, n // 10)] = True
    datastore.cells.insert("plot_highlight", highlighted, overwrite=True)

    result = splt.embedding(
        datastore,
        layout_key="RNA_UMAP",
        color_by="RNA_nCounts",
        density_overlay=splt.DensityOverlay(pixels=32, levels=3, sigma=1),
        highlight=splt.Highlight(by="plot_highlight"),
        point_size_range=(2, 20),
        show_titles=False,
        show=False,
    )

    ax = next(iter(result.axes.values()))
    assert len(ax.collections) >= 3
    assert result.provenance.extras["highlight"]["n_highlighted"] == highlighted.sum()
    assert all(
        2 <= value <= 20
        for value in result.provenance.extras["point_size_by_panel"].values()
    )
    auxiliary_labels = [
        axis.get_xlabel() for axis in result.figure.axes if axis is not ax
    ]
    assert "RNA_nCounts" in auxiliary_labels
    assert ax.get_title() == ""
    result.close()


def test_embedding_mean_contours_require_and_use_continuous_values(
    umap,
    datastore,
):
    result = splt.embedding(
        datastore,
        layout_key="RNA_UMAP",
        color_by="RNA_nCounts",
        density_overlay=splt.DensityOverlay(
            statistic="mean",
            pixels=32,
            sigma=1.5,
            min_support=0.05,
            levels=(0.6, 0.85),
            max_hotspots=1,
        ),
        show=False,
    )

    ax = next(iter(result.axes.values()))
    assert len(ax.collections) > 1
    assert result.provenance.extras["density_overlay"]["statistic"] == "mean"
    assert result.provenance.extras["density_overlay"]["max_hotspots"] == 1
    result.close()

    with pytest.raises(ValueError, match="continuous color_by"):
        splt.embedding(
            datastore,
            layout_key="RNA_UMAP",
            color_by=splt.CellField("I", kind="categorical"),
            density_overlay=splt.DensityOverlay(statistic="mean"),
            show=False,
        )
    with pytest.raises(ValueError, match="positive integer"):
        splt.DensityOverlay(max_hotspots=0)


def test_contour_hotspot_limit_keeps_the_strongest_region():
    from scipy.ndimage import label

    from scarf.plotting.embedding import _retain_strongest_hotspots

    surface = np.zeros((12, 12), dtype=np.float64)
    support = np.ones_like(surface)
    surface[1:4, 1:4] = 2.0
    surface[7:11, 7:11] = 3.0
    surface[8:10, 8:10] = 0.0
    support[7:11, 7:11] = 2.0

    filtered = _retain_strongest_hotspots(
        surface,
        support,
        level=1.0,
        max_hotspots=1,
    )
    _, n_hotspots = label(filtered >= 1.0)

    assert n_hotspots == 1
    assert np.all(filtered[1:4, 1:4] < 1.0)
    assert np.all(filtered[7:11, 7:11] >= 1.0)


def test_embedding_point_size_uses_final_panel_area(umap, datastore):
    result = splt.embedding(
        datastore,
        layout_key="RNA_UMAP",
        color_by="RNA_nCounts",
        point_size_range=(2, 20),
        show=False,
    )

    result.figure.canvas.draw()
    ax = next(iter(result.axes.values()))
    bbox = ax.get_position()
    width, height = result.figure.get_size_inches()
    panel_area = float(bbox.width * width * bbox.height * height)
    expected = default_point_size(
        len(datastore.cells.active_index("I")),
        panel_area=panel_area,
        size_min=2,
        size_max=20,
    )
    observed = next(iter(result.provenance.extras["point_size_by_panel"].values()))
    assert observed == pytest.approx(expected)
    result.close()


def test_dotplot_feature_brackets_and_axis_swap(umap, leiden_clustering, datastore):
    genes = [str(value) for value in datastore.RNA.feats.fetch_all("names")[:4]]
    features = {"Lineage": genes[:2], "State": genes[2:]}
    result = splt.dotplot(
        datastore,
        features=features,
        group_by="RNA_leiden_cluster",
        swap_axes=True,
        marker_linewidth=0.6,
        show=False,
    )

    ax = result.axes["dotplot"]
    assert result.provenance.extras["feature_group_brackets"] == 2
    assert sum(line.get_gid() == "feature-group-bracket" for line in ax.lines) == 2
    assert ax.get_xlabel() == ""
    result.close()


def test_dotplot_marker_sizes_follow_physical_grid_cells(
    umap,
    leiden_clustering,
    datastore,
):
    genes = [str(value) for value in datastore.RNA.feats.fetch_all("names")[:4]]
    features = {"Lineage": genes[:2], "State": genes[2:]}
    compact_figure, compact_ax = plt.subplots(figsize=(3, 2))
    large_figure, large_ax = plt.subplots(figsize=(8, 6))

    compact = splt.dotplot(
        datastore,
        features=features,
        group_by="RNA_leiden_cluster",
        target=compact_ax,
        show_legend=False,
        show=False,
    )
    large = splt.dotplot(
        datastore,
        features=features,
        group_by="RNA_leiden_cluster",
        target=large_ax,
        show_legend=False,
        show=False,
    )

    compact_sizes = compact_ax.collections[0].get_sizes()
    large_sizes = large_ax.collections[0].get_sizes()
    assert large_sizes.max() > compact_sizes.max()
    assert compact.provenance.extras["size_scale_source"] == "panel"

    compact_figure.canvas.draw()
    renderer = compact_figure.canvas.get_renderer()
    label_left = min(
        label.get_window_extent(renderer).x0
        for label in compact_ax.get_yticklabels()
        if label.get_text()
    )
    bracket_right = max(
        line.get_transform().transform(line.get_xydata())[:, 0].max()
        for line in compact_ax.lines
        if line.get_gid() == "feature-group-bracket"
    )
    assert bracket_right < label_left

    compact.close()
    large.close()
    plt.close(compact_figure)
    plt.close(large_figure)


def test_dotplot_left_group_labels_clear_feature_tick_labels(
    umap,
    leiden_clustering,
    datastore,
):
    genes = [str(value) for value in datastore.RNA.feats.fetch_all("names")[:4]]
    result = splt.dotplot(
        datastore,
        features={"Myeloid lineage": genes[:2], "Cell state": genes[2:]},
        group_by="RNA_leiden_cluster",
        show_legend=False,
        show=False,
    )

    figure = result.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    feature_label_left = min(
        label.get_window_extent(renderer).x0
        for label in result.axes["dotplot"].get_yticklabels()
        if label.get_text()
    )
    group_label_right = max(
        label.get_window_extent(renderer).x1
        for label in result.axes["dotplot"].texts
        if label.get_gid() == "feature-group-label"
    )

    assert feature_label_left - group_label_right >= figure.dpi * 6.0 / 72.0
    result.close()


def test_stacked_violin_standardizes_rows(umap, leiden_clustering, datastore):
    genes = [str(value) for value in datastore.RNA.feats.fetch_all("names")[:2]]
    result = splt.distribution(
        datastore,
        keys=genes,
        group_by="RNA_leiden_cluster",
        kind="stacked_violin",
        row_standardize=True,
        max_points=0,
        show=False,
    )

    assert len(result.axes) == 2
    for table in result.tables.values():
        assert np.nanmean(table["display_value"]) == pytest.approx(0, abs=1e-7)
    assert result.provenance.extras["row_standardize"] is True
    result.close()


def test_stacked_violin_can_share_value_scale(umap, leiden_clustering, datastore):
    genes = [str(value) for value in datastore.RNA.feats.fetch_all("names")[:2]]
    result = splt.distribution(
        datastore,
        keys=genes,
        group_by="RNA_leiden_cluster",
        kind="stacked_violin",
        share_y=True,
        max_points=0,
        show=False,
    )

    limits = [axis.get_ylim() for axis in result.axes.values()]
    assert all(limit == pytest.approx(limits[0]) for limit in limits[1:])
    assert result.provenance.extras["share_y"] is True
    result.close()


def test_distribution_aggregates_biological_samples(
    umap,
    leiden_clustering,
    datastore,
):
    n = len(datastore.cells.active_index("I"))
    samples = np.asarray([f"sample_{index % 8}" for index in range(n)])
    datastore.cells.insert("plot_distribution_sample", samples, overwrite=True)

    result = splt.distribution(
        datastore,
        keys="RNA_nCounts",
        group_by="RNA_leiden_cluster",
        sample_by="plot_distribution_sample",
        sample_stat="median",
        kind="box",
        max_points=0,
        show=False,
    )

    table = result.tables["RNA_nCounts"]
    assert {"sample", "group", "value", "display_value", "nCells"} <= set(table)
    assert not table.duplicated(["sample", "group"]).any()
    assert result.provenance.n_samples == 8
    assert result.provenance.extras["sample_stat"] == "median"
    result.close()


def test_distribution_draws_sample_aware_split_violins(
    umap,
    leiden_clustering,
    datastore,
):
    n = len(datastore.cells.active_index("I"))
    sample_index = np.arange(n) % 8
    samples = np.asarray([f"sample_{index}" for index in sample_index])
    conditions = np.where(sample_index % 2 == 0, "control", "stimulated")
    datastore.cells.insert("plot_split_sample", samples, overwrite=True)
    datastore.cells.insert("plot_split_condition", conditions, overwrite=True)

    result = splt.distribution(
        datastore,
        keys="RNA_nCounts",
        group_by="RNA_leiden_cluster",
        split_by="plot_split_condition",
        study_design=splt.StudyDesign(sample_by="plot_split_sample"),
        kind="violin",
        max_points=30,
        show=False,
    )

    table = result.tables["RNA_nCounts"]
    assert set(table["split"]) == {"control", "stimulated"}
    scale = result.scales[0]
    assert isinstance(scale, splt.CategoricalScale)
    assert scale.order == ("control", "stimulated")
    assert result.legends[0].label == "plot_split_condition"
    result.close()


def test_distribution_seed_repeats_point_jitter(
    umap,
    leiden_clustering,
    datastore,
):
    kwargs = {
        "keys": "RNA_nCounts",
        "group_by": "RNA_leiden_cluster",
        "kind": "box",
        "max_points": 80,
        "seed": 17,
        "show": False,
    }
    first = splt.distribution(datastore, **kwargs)
    second = splt.distribution(datastore, **kwargs)

    first_offsets = [
        np.asarray(collection.get_offsets())
        for collection in next(iter(first.axes.values())).collections
    ]
    second_offsets = [
        np.asarray(collection.get_offsets())
        for collection in next(iter(second.axes.values())).collections
    ]
    assert len(first_offsets) == len(second_offsets)
    for observed, repeated in zip(first_offsets, second_offsets, strict=True):
        np.testing.assert_array_equal(observed, repeated)
    first.close()
    second.close()


def test_sample_aggregated_feature_axis_retains_requested_italics(
    umap,
    datastore,
):
    n = len(datastore.cells.active_index("I"))
    samples = np.asarray([f"sample_{index % 8}" for index in range(n)])
    datastore.cells.insert("plot_italic_sample", samples, overwrite=True)
    gene = str(datastore.RNA.feats.fetch_all("names")[0])

    result = splt.distribution(
        datastore,
        keys=gene,
        sample_by="plot_italic_sample",
        kind="box",
        italicize_features=True,
        show=False,
    )

    axis = next(iter(result.axes.values()))
    assert axis.get_ylabel() == f"Sample mean {gene}"
    assert axis.yaxis.label.get_fontstyle() == "italic"
    result.close()


def test_per_sample_composition_reports_uncertainty(
    umap,
    leiden_clustering,
    datastore,
):
    n = len(datastore.cells.active_index("I"))
    sample_index = np.arange(n) % 8
    samples = np.asarray([f"sample_{index}" for index in sample_index])
    conditions = np.where(sample_index < 4, "control", "stimulated")
    subjects = np.asarray([f"subject_{index % 4}" for index in sample_index])
    datastore.cells.insert("plot_ci_sample", samples, overwrite=True)
    datastore.cells.insert("plot_ci_condition", conditions, overwrite=True)
    datastore.cells.insert("plot_ci_subject", subjects, overwrite=True)

    result = splt.composition(
        datastore,
        category_by="RNA_leiden_cluster",
        study_design=splt.StudyDesign(
            sample_by="plot_ci_sample",
            condition_by="plot_ci_condition",
            subject_by="plot_ci_subject",
        ),
        kind="per_sample",
        uncertainty="ci95",
        show=False,
    )

    summary = result.tables["summary"]
    assert {"mean_proportion", "lower", "upper", "n_samples"} <= set(summary)
    assert np.all(summary["lower"] <= summary["mean_proportion"])
    assert np.all(summary["mean_proportion"] <= summary["upper"])
    assert result.provenance.extras["uncertainty"] == "ci95"
    assert result.provenance.extras["n_pair_lines"] > 0
    assert any(legend.kind == "marker" for legend in result.legends)
    axis = next(iter(result.axes.values()))
    assert axis.get_ylim()[1] >= float(summary["upper"].max())
    for legend in result.figure.legends:
        if legend.get_title().get_text() == "Condition":
            assert all(
                not text.get_text().startswith("mean") for text in legend.get_texts()
            )
    rendered_labels = [
        text.get_text()
        for legend in result.figure.legends
        for text in legend.get_texts()
    ]
    assert any(
        label.startswith("mean") or label.startswith("Summary: mean")
        for label in rendered_labels
    )
    result.close()


def test_embedding_side_legend_stays_page_sized_for_many_categories(
    umap,
    datastore,
):
    n = len(datastore.cells.active_index("I"))
    labels = np.full(n, "type_149", dtype=object)
    labels[:149] = [f"type_{index:03d}" for index in range(149)]
    datastore.cells.insert("plot_many_types", labels, overwrite=True)

    result = splt.embedding(
        datastore,
        layout_key="RNA_UMAP",
        color_by="plot_many_types",
        legend_loc="right",
        show=False,
    )

    width, _ = result.figure.get_size_inches()
    assert width <= 12
    legend = result.figure.legends[0]
    assert len(legend.get_texts()) == 80
    assert legend.get_title().get_text() == "plot_many_types (80 of 150)"
    assert "type_149" in [text.get_text() for text in legend.get_texts()]
    omitted = result.provenance.extras["omitted_legend_entries"]
    panel_key = str(next(iter(result.axes)))
    assert len(omitted[panel_key]) == 70
    result.close()


def test_cluster_connectivity_runs_on_real_datastore_graph(
    umap,
    leiden_clustering,
    datastore,
):
    result = datastore.plots.cluster_connectivity(
        group_by="RNA_leiden_cluster",
        layout_key="RNA_UMAP",
        feat_key="hvgs",
        minimum_edge_weight=0,
        max_edges_per_node=3,
        show_cells=True,
        show=False,
    )

    observed = len(np.unique(datastore.cells.fetch("RNA_leiden_cluster", key="I")))
    assert len(result.tables["nodes"]) == observed
    assert set(result.tables["edges"]) == {
        "source",
        "target",
        "rawWeight",
        "normalizedWeight",
    }
    assert result.provenance.extras["n_edges"] <= 3 * observed // 2
    background = result.axes["cluster_connectivity"].collections[0]
    assert background.get_alpha() == pytest.approx(0.3)
    assert background.get_sizes()[0] >= 4
    assert not hasattr(background, "_scarf_layout_point_size")
    assert len(np.unique(background.get_facecolors(), axis=0)) > 1
    assert result.provenance.extras["cell_size_source"] == "panel"
    result.close()


def test_composition_borders_labels_and_stored_palette(
    umap,
    leiden_clustering,
    datastore,
):
    n = len(datastore.cells.active_index("I"))
    samples = np.asarray([f"s{index % 3}" for index in range(n)], dtype=object)
    datastore.cells.insert("plot_modern_samples", samples, overwrite=True)

    embedding = splt.embedding(
        datastore,
        layout_key="RNA_UMAP",
        color_by="RNA_leiden_cluster",
        show_legend=False,
        show=False,
    )
    composition = splt.composition(
        datastore,
        category_by="RNA_leiden_cluster",
        sample_by="plot_modern_samples",
        segment_linewidth=0.8,
        show_percent_labels=True,
        label_min_fraction=0.01,
        show=False,
    )

    bars = composition.axes["composition"].patches
    assert bars
    assert all(bar.get_linewidth() == pytest.approx(0.8) for bar in bars)
    assert composition.axes["composition"].texts
    embedding_scale = next(
        scale for scale in embedding.scales if isinstance(scale, splt.CategoricalScale)
    )
    composition_scale = next(
        scale
        for scale in composition.scales
        if isinstance(scale, splt.CategoricalScale)
    )
    assert embedding_scale.palette == composition_scale.palette
    embedding.close()
    composition.close()


def test_compose_results_namespaces_tables_and_renders_shared_legend(
    umap,
    leiden_clustering,
    datastore,
):
    n = len(datastore.cells.active_index("I"))
    phases = np.asarray(["G1", "S", "G2M"] * (n // 3 + 1), dtype=object)[:n]
    datastore.cells.insert("plot_composite_phase", phases, overwrite=True)
    figure, axes = plt.subplot_mosaic(
        [["embedding", "composition"]],
        figsize=(7, 3),
        layout="constrained",
    )
    first = splt.embedding(
        datastore,
        layout_key="RNA_UMAP",
        color_by="RNA_leiden_cluster",
        target=axes["embedding"],
        show_legend=True,
        theme="paper",
        show=False,
    )
    second = splt.composition(
        datastore,
        category_by="plot_composite_phase",
        target=axes["composition"],
        show_legend=True,
        theme="paper",
        show=False,
    )

    result = splt.compose_results(
        figure,
        {"embedding": first, "composition": second},
        theme="paper",
    )

    assert result.owns_figure is False
    assert result.provenance.notes == ("composite",)
    assert "composition:aggregate" in result.tables
    assert len(figure.legends) == 2
    assert all(axis.get_legend() is None for axis in axes.values())
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    legend_boxes = [legend.get_window_extent(renderer) for legend in figure.legends]
    assert not legend_boxes[0].overlaps(legend_boxes[1])
    panel_labels = [
        text
        for axis in axes.values()
        for text in axis.texts
        if text.get_text() in {"A", "B"}
    ]
    assert {text.get_fontsize() for text in panel_labels} == {
        splt.THEMES["paper"]["axes.titlesize"]
    }
    assert {
        text.get_fontsize() for legend in figure.legends for text in legend.get_texts()
    } == {splt.THEMES["paper"]["legend.fontsize"]}
    result.close()
    assert plt.fignum_exists(figure.number)
    plt.close(figure)


def test_recipe_execution_is_headless_and_preserves_analysis_state(umap, datastore):
    before = datastore.get_assay_state("RNA")
    recipe = splt.PlotRecipe(
        (
            splt.PlotStep(
                name="overview",
                plot="embedding",
                kwargs={
                    "layout_key": "RNA_UMAP",
                    "color_by": "RNA_nCounts",
                },
            ),
        )
    )

    execution = datastore.plots.run_recipe(recipe)

    assert not execution.written_paths
    assert not execution.failures
    assert len(execution.results) == 1
    assert datastore.get_assay_state("RNA") == before
    result = execution.results[0]
    assert plt.fignum_exists(result.figure.number)
    result.close()


def test_recipe_batch_output_closes_owned_figure(umap, datastore, tmp_path):
    recipe = splt.PlotRecipe(
        (
            splt.PlotStep(
                name="overview",
                plot="embedding",
                kwargs={
                    "layout_key": "RNA_UMAP",
                    "color_by": "RNA_nCounts",
                },
                output=splt.PlotOutputSettings(
                    filename="overview.png",
                    dpi=90,
                ),
            ),
        )
    )

    execution = splt.run_recipe(datastore, recipe, output_dir=tmp_path)
    plot_result = execution.results[0]

    assert execution.written_paths == (tmp_path / "overview.png",)
    assert execution.written_paths[0].stat().st_size > 0
    assert not plt.fignum_exists(plot_result.figure.number)

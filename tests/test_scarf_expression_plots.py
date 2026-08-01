"""Tests for scarf.plotting.violinplot and scarf.plotting.boxplot."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import scarf.plotting as splt


def _gene_names(datastore, n=2):
    names = datastore.RNA.feats.fetch_all("names")
    counts = np.asarray(
        datastore.RNA.rawData[:, : len(names)].sum(axis=0).compute(),
        dtype=np.float64,
    ).ravel()
    order = np.argsort(-counts)
    chosen = []
    for index in order:
        if counts[index] > 0:
            chosen.append(str(names[index]))
        if len(chosen) == n:
            break
    return chosen


def _header_texts(figure):
    return [
        text.get_text()
        for ax in figure.axes
        if not ax.axison
        for text in ax.texts
    ]


def _facet_values(n_cells, n_facets=2):
    return np.array(
        [f"Dataset {i % n_facets + 1}" for i in range(n_cells)], dtype=object
    )


def test_expression_plot_exports():
    assert "violinplot" in splt.__all__
    assert "boxplot" in splt.__all__
    assert "stacked_violin" in splt.__all__
    assert callable(splt.violinplot)
    assert callable(splt.boxplot)
    assert callable(splt.stacked_violin)


def test_violinplot_basic(leiden_clustering, datastore):
    genes = _gene_names(datastore)
    result = splt.violinplot(
        datastore,
        genes,
        group_by="RNA_leiden_cluster",
        max_points=200,
        seed=1,
        show=False,
    )
    try:
        assert len(result.axes) == len(genes)
        assert result.provenance.n_cells > 0
        assert result.provenance.cell_key == "I"
        assert {"value", "group"}.issubset(result.tables[genes[0]].columns)
        # Full-width dark header block carries the white title.
        assert _header_texts(result.figure) == ["Expression violin plot"]
        header = next(ax for ax in result.figure.axes if not ax.axison)
        assert tuple(np.round(header.get_facecolor()[:3], 3)) == tuple(
            np.round(matplotlib.colors.to_rgb("#1F2933"), 3)
        )
        # One violin per group plus jittered point collections per panel.
        ax = next(iter(result.axes.values()))
        n_groups = len(np.unique(datastore.cells.fetch("RNA_leiden_cluster")))
        assert len(ax.collections) > n_groups
        # Group labels are rotated 45 degrees by default.
        rotations = {tick.get_rotation() for tick in ax.get_xticklabels()}
        assert any(abs(r - 45) < 1e-6 for r in rotations)
        assert result.provenance.extras["approximate"] is True
        assert result.provenance.extras["group_by"] == "RNA_leiden_cluster"
        assert result.legends[0].label == "RNA_leiden_cluster"
    finally:
        result.close()


def test_violinplot_parameters_and_save(leiden_clustering, datastore, tmp_path):
    gene = _gene_names(datastore, n=1)[0]
    result = splt.violinplot(
        datastore,
        gene,
        group_by="RNA_leiden_cluster",
        title="Custom banner",
        x_label_rotation=90,
        jitter=False,
        max_points=0,
        normalization=("raw", "log1p"),
        title_bg_color="#123456",
        figsize=(6, 4),
        show=False,
    )
    try:
        assert _header_texts(result.figure) == ["Custom banner"]
        ax = next(iter(result.axes.values()))
        n_groups = len(np.unique(datastore.cells.fetch("RNA_leiden_cluster")))
        assert len(ax.collections) == n_groups
        rotations = {tick.get_rotation() for tick in ax.get_xticklabels()}
        assert any(abs(r - 90) < 1e-6 for r in rotations)
        assert result.provenance.extras["jitter"] is False
        assert result.provenance.extras["approximate"] is False
        assert result.provenance.extras["normalization"] == {
            "source": "raw",
            "transform": "log1p",
        }
        out = tmp_path / "violin.png"
        result.save(out, dpi=80)
        assert out.is_file()
    finally:
        result.close()


def test_violinplot_subset_and_groups(leiden_clustering, datastore):
    gene = _gene_names(datastore, n=1)[0]
    active_n = len(datastore.cells.active_index("I"))
    keep = np.zeros(active_n, dtype=bool)
    keep[: max(20, active_n // 2)] = True
    datastore.cells.insert("vln_keep", keep, overwrite=True)
    labels = list(pd.unique(datastore.cells.fetch("RNA_leiden_cluster")))
    keep_groups = labels[:2]
    result = splt.violinplot(
        datastore,
        gene,
        group_by="RNA_leiden_cluster",
        groups=keep_groups,
        subset_by="vln_keep",
        max_points=0,
        show=False,
    )
    try:
        assert result.provenance.extras["groups"] == list(keep_groups)
        assert result.provenance.extras["subset_by"] == "vln_keep"
        table_groups = set(result.tables[gene]["group"].unique())
        assert table_groups == set(keep_groups)
        assert result.provenance.n_cells < active_n
    finally:
        result.close()


def test_boxplot_basic(leiden_clustering, datastore):
    n_cells = datastore.cells.N
    datastore.cells.insert(
        "box_dataset",
        _facet_values(n_cells, n_facets=2),
        overwrite=True,
    )
    genes = _gene_names(datastore)
    result = splt.boxplot(
        datastore,
        genes,
        group_by="RNA_leiden_cluster",
        facet_by="box_dataset",
        show=False,
    )
    try:
        assert len(result.axes) == len(genes) * 2
        assert result.provenance.extras["facet_by"] == "box_dataset"
        assert result.provenance.extras["n_cols"] == 2
        assert result.provenance.extras["show_outliers"] is True
        # Each gene block contributes one table with the facet column.
        for gene in genes:
            table = result.tables[gene]
            assert {"value", "group", "facet"}.issubset(table.columns)
        # No legend by default: groups are labelled on the x-axis.
        assert len(result.figure.legends) == 0
        assert result.legends == ()
        assert result.provenance.extras["show_legend"] is False
        # Lettered facet labels sit beneath the facets.
        labels = [
            text.get_text()
            for ax in result.axes.values()
            for text in ax.texts
            if text.get_text().startswith("(")
        ]
        assert "(a) Dataset 1" in labels
        assert "(b) Dataset 2" in labels
        # Boxes with a widened median line.
        ax = next(iter(result.axes.values()))
        median_widths = {
            line.get_linewidth()
            for line in ax.lines
            if len(line.get_xdata()) >= 2
            and len(line.get_ydata()) >= 2
            and line.get_xdata()[0] != line.get_xdata()[-1]
            and line.get_ydata()[0] == line.get_ydata()[-1]
        }
        assert median_widths == {2.5}
    finally:
        result.close()


def test_boxplot_no_outliers_facet_titles_and_save(leiden_clustering, datastore, tmp_path):
    n_cells = datastore.cells.N
    datastore.cells.insert(
        "box_dataset3",
        _facet_values(n_cells, n_facets=3),
        overwrite=True,
    )
    gene = _gene_names(datastore, n=1)[0]
    result = splt.boxplot(
        datastore,
        gene,
        group_by="RNA_leiden_cluster",
        facet_by="box_dataset3",
        facet_titles={"Dataset 1": "Group A", "Dataset 2": "Group B", "Dataset 3": "Group C"},
        panel_label_format="{letter}. {title}",
        n_cols=2,
        show_outliers=False,
        show_legend=True,
        legend_loc="outside right upper",
        legend_title="clusters",
        title="Across datasets",
        show=False,
    )
    try:
        assert len(result.axes) == 3
        assert result.provenance.extras["show_outliers"] is False
        assert result.provenance.extras["n_cols"] == 2
        assert result.provenance.extras["show_legend"] is True
        assert result.provenance.extras["legend_loc"] == "outside right upper"
        labels = [
            text.get_text()
            for ax in result.axes.values()
            for text in ax.texts
        ]
        assert "a. Group A" in labels
        assert "b. Group B" in labels
        assert "c. Group C" in labels
        # No flier markers when outliers are disabled.
        ax = next(iter(result.axes.values()))
        fliers = [line for line in ax.lines if line.get_marker() == "o"]
        assert len(fliers) == 0
        assert len(result.figure.legends) == 1
        assert result.legends[0].kind == "categorical"
        assert result.legends[0].label == "RNA_leiden_cluster"
        out = tmp_path / "boxplot.png"
        result.save(out, dpi=80)
        assert out.is_file()
    finally:
        result.close()


def test_boxplot_invalid_facet_order(leiden_clustering, datastore):
    n_cells = datastore.cells.N
    datastore.cells.insert(
        "box_dataset_ao",
        _facet_values(n_cells, n_facets=2),
        overwrite=True,
    )
    gene = _gene_names(datastore, n=1)[0]
    with pytest.raises(ValueError, match="facet_order"):
        splt.boxplot(
            datastore,
            gene,
            group_by="RNA_leiden_cluster",
            facet_by="box_dataset_ao",
            facet_order=["Dataset 9"],
            show=False,
        )


def test_missing_metadata_column_raises_helpful_error(datastore):
    gene = _gene_names(datastore, n=1)[0]
    with pytest.raises(KeyError, match="not a metadata column"):
        splt.violinplot(datastore, gene, group_by="missing_column", show=False)
    with pytest.raises(KeyError, match="not a metadata column"):
        splt.boxplot(
            datastore,
            gene,
            group_by="RNA_leiden_cluster",
            facet_by="missing_facet",
            show=False,
        )


def test_normalization_argument_changes_values(datastore):
    gene = _gene_names(datastore, n=1)[0]
    n_cells = datastore.cells.N
    datastore.cells.insert(
        "norm_grp",
        np.array([f"C{i % 3 + 1}" for i in range(n_cells)], dtype=object),
        overwrite=True,
    )
    raw = splt.violinplot(
        datastore,
        gene,
        group_by="norm_grp",
        normalization=("raw", "none"),
        max_points=0,
        show=False,
    )
    log1p = splt.violinplot(
        datastore,
        gene,
        group_by="norm_grp",
        normalization=("raw", "log1p"),
        max_points=0,
        show=False,
    )
    try:
        raw_values = raw.tables[gene]["value"].to_numpy(dtype=np.float64)
        log_values = log1p.tables[gene]["value"].to_numpy(dtype=np.float64)
        assert np.allclose(log_values, np.log1p(raw_values))
        assert raw.provenance.extras["normalization"]["source"] == "raw"
        assert log1p.provenance.extras["normalization"]["transform"] == "log1p"
    finally:
        raw.close()
        log1p.close()


def test_boxplot_single_panel_without_facet(leiden_clustering, datastore):
    genes = _gene_names(datastore)
    result = splt.boxplot(
        datastore,
        genes,
        group_by="RNA_leiden_cluster",
        normalization=("raw", "log1p"),
        show=False,
    )
    try:
        assert len(result.axes) == len(genes)
        ax = next(iter(result.axes.values()))
        n_groups = len(np.unique(datastore.cells.fetch("RNA_leiden_cluster")))
        assert len(ax.get_xticklabels()) == n_groups
        median_widths = {
            line.get_linewidth()
            for line in ax.lines
            if len(line.get_xdata()) >= 2
            and len(line.get_ydata()) >= 2
            and line.get_xdata()[0] != line.get_xdata()[-1]
            and line.get_ydata()[0] == line.get_ydata()[-1]
        }
        assert median_widths == {2.5}
        assert ax.get_ylabel() == "expression"
        assert result.provenance.extras["facet_by"] is None
        assert result.provenance.extras["normalization"]["transform"] == "log1p"
        assert "faceted" not in result.provenance.notes
        assert len(result.figure.legends) == 0
        assert result.legends == ()
    finally:
        result.close()


def test_boxplot_show_legend(leiden_clustering, datastore):
    gene = _gene_names(datastore, n=1)[0]
    result = splt.boxplot(
        datastore,
        gene,
        group_by="RNA_leiden_cluster",
        show_legend=True,
        legend_title="clusters",
        show=False,
    )
    try:
        assert len(result.figure.legends) == 1
        assert result.legends[0].kind == "categorical"
        assert result.legends[0].label == "RNA_leiden_cluster"
        legend = result.figure.legends[0]
        assert legend.get_title().get_text() == "clusters"
        assert result.provenance.extras["show_legend"] is True
    finally:
        result.close()


def test_datastore_plots_boxplot_single_panel(leiden_clustering, datastore):
    gene = _gene_names(datastore, n=1)[0]
    result = datastore.plots.boxplot(
        gene,
        group_by="RNA_leiden_cluster",
        show=False,
    )
    try:
        assert len(result.axes) == 1
        assert result.provenance.extras["facet_by"] is None
    finally:
        result.close()


def test_stacked_violin_layout(leiden_clustering, datastore):
    genes = _gene_names(datastore)
    result = splt.stacked_violin(
        datastore,
        genes,
        group_by="RNA_leiden_cluster",
        show=False,
    )
    try:
        n_genes = len(genes)
        assert len(result.axes) == n_genes
        assert result.provenance.notes == ("stacked_violin",)
        assert result.provenance.extras["scale"] == "width"
        assert result.provenance.extras["color_by"] == "mean"
        axes = list(result.axes.values())
        # Zero vertical spacing between rows.
        heights = [ax.get_position().height for ax in axes]
        assert max(heights) - min(heights) < 1e-9
        for index, ax in enumerate(axes):
            # No numeric y-ticks; gene name is a horizontal y-label.
            assert len(ax.get_yticks()) == 0
            assert ax.get_ylabel() == genes[index]
            # Only the bottom row keeps x ticks and the bottom spine.
            if index < n_genes - 1:
                assert len(ax.get_xticklabels()) == 0
                assert not ax.spines["bottom"].get_visible()
            else:
                assert len(ax.get_xticklabels()) > 0
                assert ax.spines["bottom"].get_visible()
            assert not ax.spines["top"].get_visible()
            assert not ax.spines["right"].get_visible()
            assert ax.spines["left"].get_visible()
        # Mean-expression coloring: continuous colorbar legend + ColorScale.
        assert result.legends[0].kind == "colorbar"
        assert result.legends[0].label == "mean expression"
        assert any(isinstance(scale, splt.ColorScale) for scale in result.scales)
        assert any(
            ax.get_label().startswith("<colorbar") for ax in result.figure.axes
        )
    finally:
        result.close()


def test_stacked_violin_color_by_group(leiden_clustering, datastore):
    genes = _gene_names(datastore)
    result = splt.stacked_violin(
        datastore,
        genes,
        group_by="RNA_leiden_cluster",
        color_by="group",
        show=False,
    )
    try:
        assert result.provenance.extras["color_by"] == "group"
        assert result.legends[0].kind == "categorical"
        assert result.legends[0].label == "RNA_leiden_cluster"
        assert any(isinstance(scale, splt.CategoricalScale) for scale in result.scales)
        assert len(result.figure.legends) == 1
        # Every row shares the same group colours down the stack.
        axes = list(result.axes.values())
        def color_sets(ax):
            return {
                tuple(np.round(c.get_facecolor()[0][:3], 3))
                for c in ax.collections
                if hasattr(c, "get_facecolor") and len(c.get_facecolor())
            }

        color_sets_list = [color_sets(ax) for ax in axes]
        assert all(colors == color_sets_list[0] for colors in color_sets_list)
    finally:
        result.close()


def test_stacked_violin_mean_color_scale(leiden_clustering, datastore):
    gene = _gene_names(datastore, n=1)[0]
    result = splt.stacked_violin(
        datastore,
        gene,
        group_by="RNA_leiden_cluster",
        color_by="mean",
        cmap="magma",
        vmin=0.0,
        vmax=5.0,
        show=False,
    )
    try:
        assert result.provenance.extras["cmap"] == "magma"
        assert result.provenance.extras["vmin"] == 0.0
        assert result.provenance.extras["vmax"] == 5.0
        color_scale = next(
            scale for scale in result.scales if isinstance(scale, splt.ColorScale)
        )
        assert color_scale.cmap == "magma"
        assert color_scale.vmin == 0.0
        assert color_scale.vmax == 5.0
        assert result.legends[0].kind == "colorbar"
        assert result.legends[0].extras["vmin"] == 0.0
        assert result.legends[0].extras["vmax"] == 5.0
    finally:
        result.close()


def test_stacked_violin_invalid_color_by(datastore):
    gene = _gene_names(datastore, n=1)[0]
    with pytest.raises(ValueError, match="color_by"):
        splt.stacked_violin(
            datastore,
            gene,
            group_by="RNA_leiden_cluster",
            color_by="bogus",
            show=False,
        )


def test_stacked_violin_rotation_row_standardize_and_save(
    leiden_clustering, datastore, tmp_path
):
    genes = _gene_names(datastore)
    result = splt.stacked_violin(
        datastore,
        genes,
        group_by="RNA_leiden_cluster",
        x_label_rotation=45,
        row_standardize=True,
        scale="area",
        cmap="viridis",
        row_height=0.6,
        title="Stacked",
        show=False,
    )
    try:
        assert result.provenance.extras["row_standardize"] is True
        assert result.provenance.extras["scale"] == "area"
        assert result.provenance.extras["x_label_rotation"] == 45
        bottom = list(result.axes.values())[-1]
        rotations = {tick.get_rotation() for tick in bottom.get_xticklabels()}
        assert any(abs(r - 45) < 1e-6 for r in rotations)
        # Standardized rows have near-zero mean.
        for gene in genes:
            values = result.tables[gene]["value"].to_numpy(dtype=np.float64)
            assert abs(np.nanmean(values)) < 1e-9
        out = tmp_path / "stacked.png"
        result.save(out, dpi=80)
        assert out.is_file()
    finally:
        result.close()


def test_stacked_violin_invalid_scale(datastore):
    gene = _gene_names(datastore, n=1)[0]
    with pytest.raises(ValueError, match="scale"):
        splt.stacked_violin(
            datastore,
            gene,
            group_by="RNA_leiden_cluster",
            scale="bogus",
            show=False,
        )


def test_stacked_violin_unknown_cmap(datastore):
    gene = _gene_names(datastore, n=1)[0]
    with pytest.raises(ValueError, match="colormap"):
        splt.stacked_violin(
            datastore,
            gene,
            group_by="RNA_leiden_cluster",
            cmap="not_a_real_cmap",
            show=False,
        )


def test_datastore_plots_stacked_violin(leiden_clustering, datastore):
    genes = _gene_names(datastore, n=2)
    result = datastore.plots.stacked_violin(
        genes,
        group_by="RNA_leiden_cluster",
        show=False,
    )
    try:
        assert len(result.axes) == 2
        assert result.provenance.notes == ("stacked_violin",)
    finally:
        result.close()




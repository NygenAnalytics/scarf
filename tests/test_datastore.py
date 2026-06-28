import numpy as np
import pandas as pd

from . import full_path


class TestToyDataStore:
    def test_toy_crdir_metadata(self, toy_crdir_ds):
        assert np.all(
            toy_crdir_ds.RNA.feats.fetch_all("ids") == ["g1", "g2", "g3", "g4"]
        )
        assert np.all(toy_crdir_ds.ADT.feats.fetch_all("ids") == ["a1", "a2"])
        assert np.all(toy_crdir_ds.HTO.feats.fetch_all("ids") == ["h1"])
        assert np.all(toy_crdir_ds.cells.fetch_all("ids") == ["b1", "b2", "b3"])

    def test_toy_crdir_rawdata(self, toy_crdir_ds):
        assert np.all(
            toy_crdir_ds.RNA.rawData.compute()
            == [[5, 0, 0, 2], [3, 3, 0, 7], [3, 3, 0, 7]]
        )
        assert np.all(
            toy_crdir_ds.ADT.rawData.compute() == [[30, 40], [30, 50], [0, 50]]
        )
        assert np.all(toy_crdir_ds.HTO.rawData.compute() == [[200], [100], [100]])


class TestDataStore:
    def test_init_wrong_zarr_mode(self, tmp_path):
        import pytest
        import tarfile

        from scarf.datastore.datastore import DataStore

        fn = full_path("1K_pbmc_citeseq.zarr.tar.gz")
        out_fn = tmp_path / "1K_pbmc_citeseq.zarr"
        with tarfile.open(fn, "r:gz") as tar:
            tar.extractall(out_fn, filter="data")
        with pytest.raises(ValueError):
            DataStore(str(out_fn), zarr_mode="wrong", default_assay="RNA")

    def test_auto_filter_cells(self, datastore_ephemeral):
        assert (
            datastore_ephemeral.auto_filter_cells(
                attrs=["nCounts", "nFeatures", "non_existing_column"],
                show_qc_plots=False,
            )
            is None
        )
        # show_qc_plots=True
        #  howto test plots?

    def test_filter_cells(self, datastore_ephemeral):
        assert (
            datastore_ephemeral.filter_cells(
                attrs=["nCounts", "nFeatures", "non_existing_column"],
                lows=[None, None, None],
                highs=[None, None, None],
                reset_previous=True,
            )
            is None
        )
        # still doesn't access `if j is None:` cases for j and k

    def test_graph_indices(self, make_graph, datastore):
        a = np.load(full_path("knn_indices.npy"))
        b = datastore.z[make_graph]["indices"][:]
        assert np.array_equal(a, b)

    def test_graph_distances(self, make_graph, datastore):
        a = np.load(full_path("knn_distances.npy"))
        b = datastore.z[make_graph]["distances"][:]
        assert np.all((a - b) < 1e-3)

    def test_graph_weights(self, make_graph, datastore):
        a = np.load(full_path("knn_weights.npy"))
        b = datastore.z[make_graph]["graph__1.0__1.5"]["weights"][:]
        assert np.all((a - b) < 1e-5)

    def test_atac_graph_indices(self, make_atac_graph, atac_datastore):
        a = np.load(full_path("atac_knn_indices.npy"))
        b = atac_datastore.z[make_atac_graph]["indices"][:]
        assert a.shape == b.shape

        # TODO: activate this when this PR is merged and released in gensim
        # https://github.com/RaRe-Technologies/gensim/pull/3194
        # assert np.array_equal(a, b)

    def test_atac_graph_distances(self, make_atac_graph, atac_datastore):
        a = np.load(full_path("atac_knn_distances.npy"))
        b = atac_datastore.z[make_atac_graph]["distances"][:]
        assert a.shape == b.shape

        # TODO: activate this when this PR is merged and released in gensim
        # https://github.com/RaRe-Technologies/gensim/pull/3194
        # assert np.all((a - b) < 1e-5)

    def test_leiden_values(self, leiden_clustering, cell_attrs):
        assert len(set(leiden_clustering)) == 10
        # Disabled the following test because failing on CI
        # assert np.array_equal(leiden_clustering, cell_attrs['RNA_leiden_cluster'].values)

    def test_paris_values(self, paris_clustering, cell_attrs):
        assert np.array_equal(paris_clustering, cell_attrs["RNA_cluster"].values)

    def test_paris_balanced_values(self, paris_clustering_balanced, cell_attrs):
        assert np.array_equal(
            paris_clustering_balanced, cell_attrs["RNA_balanced_clusters"].values
        )

    def test_run_cell_cycle_scoring(self, cell_cycle_scoring, cell_attrs):
        assert np.array_equal(
            cell_cycle_scoring, cell_attrs["RNA_cell_cycle_phase"].values
        )

    def test_umap_values(self, umap, cell_attrs):
        precalc_umap = cell_attrs[["RNA_UMAP1", "RNA_UMAP2"]].values
        assert umap.shape == precalc_umap.shape
        # Disabled the following test because failing on CI
        # assert np.all((umap - precalc_umap) < 0.1)

    def test_get_markers(self, marker_search, paris_clustering, datastore):
        precalc_markers = pd.read_csv(full_path("markers_cluster1.csv"), index_col=0)
        markers = datastore.get_markers(group_key="RNA_cluster", group_id=1)

        # Check feature names and scores (always required)
        assert markers.feature_name.equals(precalc_markers.feature_name)
        diff = (markers.score - precalc_markers.score).values
        assert np.all(diff < 1e-3)

        # Check p_values only if they exist in reference data (backward compatible)
        if "p_value" in precalc_markers.columns:
            assert "p_value" in markers.columns, "p_value column missing in output"
            # P-values should match within reasonable tolerance
            p_diff = (markers.p_value - precalc_markers.p_value).values
            assert np.all(np.abs(p_diff) < 1e-3), "p_values differ from reference"

    def test_export_markers_to_csv(
        self, marker_search, paris_clustering, datastore, tmp_path
    ):
        precalc_markers = pd.read_csv(full_path("markers_all_clusters.csv"))
        out_file = str(tmp_path / "test_values_markers.csv")
        datastore.export_markers_to_csv(group_key="RNA_cluster", csv_filename=out_file)
        markers = pd.read_csv(out_file)
        assert markers.equals(precalc_markers)

    def test_run_unified_umap(self, run_unified_umap, datastore):
        coords = datastore.z["RNA"]["projections"]["unified_UMAP"][:]
        precalc_coords = np.load(full_path("unified_UMAP_coords.npy"))
        assert coords.shape == precalc_coords.shape

    def test_get_target_classes(
        self, run_mapping, paris_clustering, cell_attrs, datastore
    ):
        classes = datastore.get_target_classes(
            target_name="selfmap", reference_class_group="RNA_cluster"
        )
        assert np.array_equal(classes.values, cell_attrs["target_classes"].values)

    def test_get_mapping_score(self, run_mapping, cell_attrs, datastore):
        scores = next(datastore.get_mapping_score(target_name="selfmap"))[1]
        diff = scores - cell_attrs["mapping_scores"].values
        assert np.all(diff < 1e-2)

    def test_coral_mapping_score(self, run_mapping_coral, cell_attrs, datastore):
        # TODO: add test values for coral
        assert 1 == 1

    def test_repr(self, datastore):
        # TODO: Test if the expected values are printed
        print(datastore)

    def test_get_imputed(self, datastore):
        # TODO: Test the output values
        values = datastore.get_imputed(feature_name="CD4")
        assert values.shape == datastore.cells.fetch("I").shape

    def test_run_doublet_detection(self, make_graph, paris_clustering, datastore):
        datastore.run_doublet_detection(
            cluster_key="RNA_cluster", simulation_ratio=0.5, random_seed=1
        )
        col = "RNA_doublet_score"
        assert col in datastore.cells.columns
        scores = datastore.cells.fetch(col)
        n_active = datastore.cells.active_index("I").shape[0]
        assert scores.shape[0] == n_active
        assert not np.isnan(scores).any()
        assert scores.min() >= 0.0 and scores.max() <= 1.0
        # scratch column used for smoothing should be removed
        assert "RNA_doublet_score__raw" not in datastore.cells.columns
        # temporary simulated-doublet projection should be cleaned up
        projections = datastore.z["RNA"].get("projections", None)
        if projections is not None:
            assert "_doublet_sim_RNA" not in projections

    def test_run_doublet_detection_bad_cluster_key(self, make_graph, datastore):
        import pytest

        with pytest.raises(ValueError):
            datastore.run_doublet_detection(cluster_key="not_a_column")

    def test_run_pseudotime_scoring(self, pseudotime_scoring, cell_attrs):
        diff = pseudotime_scoring - cell_attrs["RNA_pseudotime"].values
        assert np.all(np.abs(diff) < 0.08)

    def test_run_pseudotime_marker_search(self, pseudotime_markers):
        precalc_markers = pd.read_csv(
            full_path("pseudotime_markers_r_values.csv"), index_col=0
        )
        assert np.all(precalc_markers.index == pseudotime_markers.index)
        assert np.all(precalc_markers.names.values == pseudotime_markers.names.values)
        assert np.allclose(
            precalc_markers.I__RNA_pseudotime__r.values,
            pseudotime_markers.I__RNA_pseudotime__r.values,
            rtol=0.15,
            atol=0.15,
        )

    def test_run_pseudotime_aggregation(self, pseudotime_aggregation, datastore):
        precalc_values = np.load(full_path("aggregated_feat_idx.npy"))
        agg_group = datastore.z["RNA"]["aggregated_I_I_RNA_pseudotime"]
        test_values = agg_group["feature_indices"][:]
        assert np.array_equal(
            precalc_values.astype(np.int64), test_values.astype(np.int64)
        )

        precalc_values = np.load(full_path("aggregated_df_top_10.npy"))
        test_values = agg_group["data"][:10]
        assert np.max(np.abs(precalc_values - test_values)) < 3.0

        precalc_values = np.load(full_path("pseudotime_clusters.npy"))
        test_values = datastore.RNA.feats.fetch_all("pseudotime_clusters")
        assert len(test_values) == len(precalc_values)
        assert len(np.unique(test_values)) == len(np.unique(precalc_values))

    def test_add_grouped_assay(self, grouped_assay, datastore):
        precalc_values = np.load(full_path("ptime_modules_group_1.npy"))
        test_values = datastore.get_cell_vals(
            from_assay="PTIME_MODULES", cell_key="I", k="group_1"
        )
        assert np.allclose(precalc_values, test_values, rtol=0.5, atol=0.5)

    def test_make_bulk(self, leiden_clustering, datastore):
        df = datastore.make_bulk(group_key="RNA_leiden_cluster")
        assert df.shape == (18850, 10)
        assert hash(tuple((df.values.flatten()))) == -3925915741848261436

    def test_to_anndata(self, datastore):
        # TODO: Check if all the attributes copied to anndata
        datastore.to_anndata()

    def test_run_topacedo_sampler(self, cell_attrs, topacedo_sampler):
        assert np.all(topacedo_sampler == cell_attrs["RNA_sketched"])

    def test_plot_cells_dists(self, datastore):
        datastore.plot_cells_dists(show_fig=False)

    def test_plot_layout(self, umap, paris_clustering, datastore):
        datastore.plot_layout(
            layout_key="RNA_UMAP", color_by="RNA_cluster", show_fig=False
        )

    # def test_plot_layout_shade(self, umap, paris_clustering, datastore):
    #     datastore.plot_layout(
    #         layout_key="RNA_UMAP",
    #         color_by="RNA_cluster",
    #         show_fig=False,
    #         do_shading=True,
    #     )

    def test_plot_cluster_tree(self, datastore):
        datastore.plot_cluster_tree(cluster_key="RNA_cluster", show_fig=False)

    def test_plot_marker_heatmap(self, marker_search, datastore):
        datastore.plot_marker_heatmap(group_key="RNA_cluster", show_fig=False)

    def test_plot_unified_layout(self, run_unified_umap, datastore):
        datastore.plot_unified_layout(layout_key="unified_UMAP", show_fig=False)

    def test_plot_unified_layout_target_groups(
        self, run_unified_umap, paris_clustering, datastore
    ):
        from scarf._types import as_zarr_array, as_zarr_group

        projections = as_zarr_group(
            as_zarr_group(datastore.zw["RNA"], name="RNA")["projections"],
            name="projections",
        )
        layout = as_zarr_array(projections["unified_UMAP"], name="unified_UMAP")
        n_target_cells = int(layout.attrs["n_cells"][1])
        target_groups = paris_clustering[:n_target_cells]
        datastore.plot_unified_layout(
            layout_key="unified_UMAP",
            show_target_only=True,
            legend_ondata=True,
            target_groups=target_groups,
            show_fig=False,
        )

    def test_plot_pseudotime_heatmap(self, pseudotime_aggregation, datastore):
        datastore.plot_pseudotime_heatmap(
            cell_key="I",
            feat_key="I",
            feature_cluster_key="pseudotime_clusters",
            pseudotime_key="RNA_pseudotime",
            show_features=["Wsb1", "Rest"],
            show_fig=False,
        )

    def test_mark_hvgs_with_atac_assay(self, atac_datastore):
        import pytest

        with pytest.raises(TypeError):
            atac_datastore.mark_hvgs()

    def test_mark_hvgs_default_max_cells_unbounded(self, auto_filter_cells, datastore):
        from unittest.mock import patch

        with patch.object(datastore.RNA, "mark_hvgs") as mock:
            datastore.mark_hvgs(show_plot=False, top_n=10)
        assert mock.call_args.kwargs["max_cells"] == np.inf

    def test_mark_prevalent_peaks_with_rna_assay(self, datastore):
        import pytest

        with pytest.raises(TypeError):
            datastore.mark_prevalent_peaks()

    def test_run_marker_search_with_no_groupkey(self, datastore):
        import pytest

        with pytest.raises(ValueError):
            datastore.run_marker_search(group_key=None)

    def test_run_marker_search_with_cellkey(self, datastore, paris_clustering):
        datastore.run_marker_search(group_key="RNA_cluster", cell_key="I")

    def test_streaming_feature_stats_matches_three_pass(self, datastore):
        from scarf.utils import controlled_compute

        assay = datastore.RNA
        cell_idx, feat_idx = assay._get_cell_feat_idx("I", "I")

        tiled = assay._streaming_feature_stats(cell_idx, feat_idx)

        normed = assay.normed(cell_idx, feat_idx)
        legacy_n = controlled_compute((normed > 0).sum(axis=0), assay.nthreads)
        legacy_tot = controlled_compute(normed.sum(axis=0), assay.nthreads)
        legacy_sigmas = controlled_compute(normed.var(axis=0), assay.nthreads)

        np.testing.assert_allclose(tiled["normed_n"], legacy_n, rtol=0, atol=1e-6)
        np.testing.assert_allclose(
            tiled["normed_tot"], legacy_tot, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(tiled["sigmas"], legacy_sigmas, rtol=1e-5, atol=1e-6)

    def test_streaming_feature_stats_column_block_branch(self, datastore):
        # A tiny budget forces the chunk-aligned column-block path; results must
        # still match the full-width reductions.
        from scarf.storage.budget import ResourceBudget, set_resource_budget
        from scarf.utils import controlled_compute

        assay = datastore.RNA
        cell_idx, feat_idx = assay._get_cell_feat_idx("I", "I")
        normed = assay.normed(cell_idx, feat_idx)
        legacy_tot = controlled_compute(normed.sum(axis=0), assay.nthreads)
        legacy_sigmas = controlled_compute(normed.var(axis=0), assay.nthreads)

        try:
            set_resource_budget(ResourceBudget(memoryBytes=1 * 1024 * 1024, workers=2))
            tiled = assay._streaming_feature_stats(cell_idx, feat_idx)
        finally:
            set_resource_budget(None)

        np.testing.assert_allclose(
            tiled["normed_tot"], legacy_tot, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(tiled["sigmas"], legacy_sigmas, rtol=1e-5, atol=1e-6)

    def test_streaming_feature_stats_requires_sf(self, datastore):
        import pytest

        assay = datastore.RNA
        cell_idx, feat_idx = assay._get_cell_feat_idx("I", "I")
        original = assay.sf
        try:
            assay.sf = None
            with pytest.raises(ValueError):
                assay._streaming_feature_stats(cell_idx, feat_idx)
        finally:
            assay.sf = original

    def test_read_block_preserves_order_and_selection(self, datastore):
        from scarf.assay import _read_block

        backing = datastore.RNA.rawData._backing

        rows = np.array([3, 4, 5, 6])
        cols = np.array([0, 1, 2])
        contiguous = _read_block(backing, rows, cols)
        reference = np.asarray(backing.get_orthogonal_selection((rows, cols)))
        np.testing.assert_array_equal(contiguous, reference)

        scattered_rows = np.array([7, 2, 9])
        scattered_cols = np.array([4, 0, 2])
        scattered = _read_block(backing, scattered_rows, scattered_cols)
        ref2 = np.asarray(
            backing.get_orthogonal_selection((scattered_rows, scattered_cols))
        )
        np.testing.assert_array_equal(scattered, ref2)

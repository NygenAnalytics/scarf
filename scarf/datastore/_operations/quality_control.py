from collections.abc import Iterable
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from ...assay import ATACassay, RNAassay
from ...quality_control.cell_cycle import assign_cell_cycle_phase
from ...quality_control.filtering import gaussian_quantile_bounds
from ...quality_control.hto import hto_demux
from ...utils.compute import controlled_compute
from ...utils.logging import logger

if TYPE_CHECKING:
    from ...storage.stores import ZARRLOC
    from ..mapping_datastore import MappingDatastore as _QualityControlOperationsBase
else:
    _QualityControlOperationsBase = object


class _QualityControlOperationsMixin(_QualityControlOperationsBase):
    if TYPE_CHECKING:

        @staticmethod
        def _create_temporary_datastore(
            zarr_loc: ZARRLOC,
            *,
            default_assay: str,
            assay_types: dict[str, str],
            nthreads: int,
        ) -> _QualityControlOperationsBase: ...

    def filter_cells(
        self,
        attrs: Iterable[str],
        lows: Iterable[int],
        highs: Iterable[int],
        reset_previous: bool = False,
        keep_bounds: bool = False,
    ) -> None:
        """Filter cells based on the cell metadata column values. Filtering
        triggers `update` method on  'I' column of cell metadata which uses
        'and' operation. This means that cells that are not within the
        filtering thresholds will have value set as False in 'I' column of cell
        metadata table. When performing filtering repeatedly, the cells that
        were previously filtered out remain filtered out and 'I' column is
        updated only for those cells that are filtered out due to the latest
        filtering attempt.

        Args:
            attrs: Names of columns to be used for filtering
            lows: Lower bounds of thresholds for filtering. Should be in same order as the names in `attrs` parameter
            highs: Upper bounds of thresholds for filtering. Should be in same order as the names in `attrs` parameter
            reset_previous: If True, then results of previous filtering will be undone completely.
                            (Default value: False)
            keep_bounds: If True, then the boundary values are retained and not filtered out (Default value: False)

        Returns:
        """
        new_bool = np.ones(self.cells.N).astype(bool)
        for i, j, k in zip(attrs, lows, highs):
            # Checking here to avoid hard error from metadata class
            if i not in self.cells.columns:
                logger.warning(
                    f"{i} not found in cell metadata. Will ignore {i} for filtering"
                )
                continue
            if j is None:
                j = -np.inf
            if k is None:
                k = np.inf
            x = self.cells.sift(i, j, k, keep_bounds=keep_bounds)
            logger.info(
                f"{len(x) - x.sum()} cells flagged for filtering out using attribute {i}"
            )
            new_bool = new_bool & x
        if reset_previous:
            self.cells.reset_key(key="I")
        self.cells.update_key(new_bool, key="I")

    def auto_filter_cells(
        self,
        attrs: Iterable[str] | None = None,
        min_p: float = 0.01,
        max_p: float = 0.99,
        show_qc_plots: bool = True,
    ) -> None:
        """Automatically filter cells based on columns of the cell metadata
        table.

        This is a wrapper around ``filter_cells`` that determines the thresholds
        for each column. It models a normal distribution centered on the column
        median and using the column standard deviation, then evaluates its
        quantiles at ``min_p`` and ``max_p``.

        Args:
            attrs: Column names to be used for filtering.
            min_p: Quantile used for the lower threshold.
            max_p: Quantile used for the upper threshold.
            show_qc_plots: Show pre-filtering and post-filtering distributions
                for the columns used.

        Returns:
            None
        """
        from ...plotting import distribution

        if attrs is None:
            attrs = []
            for i in ["nCounts", "nFeatures", "percentMito", "percentRibo"]:
                i = f"{self._defaultAssay}_{i}"
                if i in self.cells.columns:
                    attrs.append(i)

        attrs_used: list[str] = []
        for i in attrs:
            if i not in self.cells.columns:
                logger.warning(
                    f"{i} not found in cell metadata. Will ignore {i} for filtering"
                )
                continue
            a = self.cells.fetch_all(i)
            low, high = gaussian_quantile_bounds(a, min_p, max_p)
            self.filter_cells(
                attrs=[i],
                lows=[cast(int, low)],
                highs=[cast(int, high)],
            )
            attrs_used.append(i)

        if show_qc_plots and attrs_used:
            # Match the previous plot_cells_dists contract: pre uses every cell,
            # post uses the filtered active set under cell key I.
            distribution(
                self,
                keys=attrs_used,
                cell_key=None,
                color="steelblue",
                title="Pre-filtering distribution",
                show=True,
            )
            distribution(
                self,
                keys=attrs_used,
                cell_key="I",
                color="coral",
                title="Post-filtering distribution",
                show=True,
            )

    def mark_hto_identities(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        label: str = "Hashtag_identity",
    ) -> None:
        """Assign HTO hashtag identities to cells using demultiplexing.

        Args:
            from_assay: HTO assay name (default: ``'HTO'``).
            cell_key: Boolean cell metadata column selecting cells (default: latest for assay).
            label: Column name to store identities in cell metadata.

        Returns:
            None
        """
        if from_assay is None:
            from_assay = "HTO"
        if cell_key is None:
            cell_key = self._get_latest_cell_key(from_assay)
        assay = self._get_assay(from_assay)
        counts = controlled_compute(
            assay.rawData[self.cells.fetch_all(cell_key)], self.nthreads
        )
        hto_idents = hto_demux(
            pd.DataFrame(counts, columns=assay.feats.fetch_all("ids"))
        )
        self.cells.insert(
            column_name=label,
            values=np.array(hto_idents.values),
            overwrite=True,
            key=cell_key,
        )

    def run_doublet_detection(
        self,
        cluster_key: str,
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        cluster_sample_fraction: float = 0.05,
        max_cells_per_cluster: int = 100,
        simulation_ratio: float = 1.0,
        heterotypic_fraction: float = 0.8,
        save_k: int = 5,
        smoothing_t: int = 2,
        normalize_scores: bool = True,
        label: str = "doublet_score",
        batch_size: int = 1000,
        random_seed: int = 4444,
    ) -> str:
        """Flag potential doublets by simulating and mapping synthetic doublets.

        Synthetic doublets are simulated by summing the raw counts of pairs of
        observed cells drawn from a per-cluster subsample, with a tunable bias
        toward cross-cluster (heterotypic) pairs. The simulated profiles are
        projected onto the existing reference graph with `run_mapping`, and each
        reference cell is scored by how frequently it appears among the nearest
        neighbours of the simulated doublets (`get_mapping_score`). The score is
        then diffused over the KNN graph using the same operator as
        `get_imputed`. The final per-cell score is written to cell metadata so
        that users can threshold and filter doublets themselves. This is a
        graph-native adaptation of the Scrublet and DoubletFinder approach.

        Args:
            cluster_key: Cell metadata column with cluster or group labels used
                to stratify the candidate pool (for example ``'RNA_cluster'``).
            from_assay: Assay to use. Defaults to the latest used assay. Only
                RNAassay type assays are supported.
            cell_key: Cell key matching the desired graph (default: ``'I'``).
            feat_key: Feature key matching the desired graph. Defaults to the
                latest used feature key for the assay.
            cluster_sample_fraction: Fraction of cells sampled from each cluster
                to build the candidate pool. (Default value: 0.05)
            max_cells_per_cluster: Cap on the number of cells sampled per cluster.
                (Default value: 100)
            simulation_ratio: Number of simulated doublets expressed as a
                multiple of the number of reference cells. (Default value: 1.0)
            heterotypic_fraction: Fraction of simulated doublets forced to be
                cross-cluster. Set to 0 to disable the bias. (Default value: 0.8)
            save_k: Number of reference neighbours stored per simulated doublet.
                (Default value: 5)
            smoothing_t: Diffusion power used to smoothen scores over the graph,
                same as the ``t`` parameter of `get_imputed`. (Default value: 2)
            normalize_scores: If True, the final score is min-max scaled to the
                0-1 range for interpretability. (Default value: True)
            label: Base name for the score column in cell metadata. The assay
                name (and cell key when not ``'I'``) is prepended.
                (Default value: 'doublet_score')
            batch_size: Number of simulated doublets written per batch.
                (Default value: 1000)
            random_seed: Seed for reproducible sampling. (Default value: 4444)

        Returns:
            Name of the cell-metadata column containing the final scores.
        """
        import shutil
        import tempfile

        from scipy.sparse import csr_matrix

        from ...quality_control.doublets import (
            sample_cluster_pool,
            simulate_doublet_pairs,
            write_doublet_target_zarr,
        )

        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay, cell_key, feat_key
        )
        source_assay = self._get_assay(from_assay)
        if type(source_assay) != RNAassay:  # noqa: E721
            raise TypeError(
                "ERROR: Doublet detection is only supported for RNAassay type assays. "
                f"The provided assay is {type(source_assay)} type"
            )
        if cluster_key not in self.cells.columns:
            raise ValueError(
                f"ERROR: `cluster_key` {cluster_key} not found in cell metadata. Provide a column "
                f"with cluster or group labels, for example '{from_assay}_cluster'"
            )

        rng = np.random.default_rng(random_seed)
        active_idx = self.cells.active_index(cell_key)
        n_active = len(active_idx)
        clusters = self.cells.fetch(cluster_key, key=cell_key)

        pool_positions = sample_cluster_pool(
            clusters, cluster_sample_fraction, max_cells_per_cluster, rng
        )
        pool_clusters = np.asarray(clusters)[pool_positions]
        pool_raw_rows = np.asarray(active_idx)[pool_positions]
        logger.info(
            f"Sampled {len(pool_positions)} cells across "
            f"{len(np.unique(pool_clusters))} clusters to seed doublet simulation"
        )

        pool_counts = controlled_compute(
            source_assay.rawData[pool_raw_rows, :], self.nthreads
        )
        pool_csr = csr_matrix(pool_counts)

        n_sim = max(1, int(round(simulation_ratio * n_active)))
        left, right = simulate_doublet_pairs(
            pool_clusters, n_sim, heterotypic_fraction, rng
        )
        sim_counts = (pool_csr[left] + pool_csr[right]).tocsr()
        logger.info(f"Simulated {n_sim} synthetic doublets")

        temp_dir = tempfile.mkdtemp(prefix="scarf_doublet_")
        target_name = f"_doublet_sim_{from_assay}"
        target_feat_key = f"{feat_key}_doublet"
        final_col: str | None = None
        try:
            write_doublet_target_zarr(
                zarr_loc=temp_dir,
                assay_name=from_assay,
                sim_counts=sim_counts,
                feat_ids=source_assay.feats.fetch_all("ids"),
                feat_names=source_assay.feats.fetch_all("names"),
                dtype=str(source_assay.rawData.dtype),
                batch_size=batch_size,
            )
            target_ds = self._create_temporary_datastore(
                temp_dir,
                default_assay=from_assay,
                assay_types={from_assay: "RNA"},
                nthreads=self.nthreads,
            )
            self.run_mapping(
                target_assay=target_ds._get_assay(from_assay),
                target_name=target_name,
                target_feat_key=target_feat_key,
                from_assay=from_assay,
                cell_key=cell_key,
                feat_key=feat_key,
                save_k=save_k,
                batch_size=batch_size,
            )

            raw_scores = None
            for _, score in self.get_mapping_score(
                target_name=target_name,
                from_assay=from_assay,
                cell_key=cell_key,
                log_transform=True,
            ):
                raw_scores = score
            if raw_scores is None:
                raise RuntimeError(
                    "ERROR: Mapping scores could not be computed for simulated doublets"
                )

            temp_col = self._col_renamer(from_assay, cell_key, f"{label}__raw")
            self.cells.insert(temp_col, raw_scores, key=cell_key, overwrite=True)
            smoothed = self.get_imputed(
                from_assay=from_assay,
                cell_key=cell_key,
                feature_name=temp_col,
                feat_key=feat_key,
                t=smoothing_t,
            )
            self.cells.drop(temp_col)

            scores = np.asarray(smoothed, dtype=float)
            if normalize_scores:
                lo, hi = scores.min(), scores.max()
                scores = (scores - lo) / (hi - lo) if hi > lo else np.zeros_like(scores)
            final_col = self._col_renamer(from_assay, cell_key, label)
            self.cells.insert(final_col, scores, key=cell_key, overwrite=True)
            logger.info(f"Doublet scores stored in cell metadata column '{final_col}'")
        finally:
            store_loc = f"{from_assay}/projections/{target_name}"
            try:
                if store_loc in self.zw:
                    del self.zw[store_loc]
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Could not remove temporary projection group: {e}")
            shutil.rmtree(temp_dir, ignore_errors=True)
        if final_col is None:
            raise RuntimeError("Doublet score column was not created")
        return final_col

    def mark_prevalent_peaks(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        top_n: int = 10000,
        prevalence_key_name: str = "prevalent_peaks",
    ) -> None:
        """Feature selection method for ATACassay type assays.

        This method first calculates prevalence of each peak by computing sum of TF-IDF normalized values for each peak
        and then marks `top_n` peaks with the highest prevalence as prevalent peaks.

        Args:
            from_assay: Assay to use for graph creation. If no value is provided then `defaultAssay` will be used
            cell_key: Cells to use for selection of most prevalent peaks. By default, all cells with True value in
                      'I' will be used. The provided value for `cell_key` should be a column in cell metadata table
                      with boolean values.
            top_n: Number of top prevalent peaks to be selected. (Default: 10000)
            prevalence_key_name: Base label for marking prevalent peaks in the features metadata column. The value for
                                'cell_key' parameter is prepended to this value. (Default value: 'prevalent_peaks')

        Returns:
            None
        """
        if cell_key is None:
            cell_key = "I"
        assay = self._get_assay(from_assay)
        if type(assay) != ATACassay:  # noqa: E721
            raise TypeError(
                f"ERROR: This method of feature selection can only be applied to ATACassay type of assay. "
                f"The provided assay is {type(assay)} type"
            )
        assay.mark_prevalent_peaks(cell_key, top_n, prevalence_key_name)

    def run_cell_cycle_scoring(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        s_genes: list[str] | None = None,
        g2m_genes: list[str] | None = None,
        n_bins: int = 50,
        rand_seed: int = 4466,
        s_score_label: str = "S_score",
        g2m_score_label: str = "G2M_score",
        phase_label: str = "cell_cycle_phase",
    ) -> None:
        """Computes S and G2M phase scores by taking into account the average
        expression of S and G2M phase genes respectively. Following steps are
        taken for each phase:

        - Average expression of all the genes in across `cell_key` cells is calculated
        - The log average expression is divided in `n_bins` bins
        - A control set of genes is identified by sampling genes from same expression bins where phase's genes are present.
        - The average expression of phase genes (Ep) and control genes (Ec) is calculated per cell.
        - A phase score is calculated as ``Ep - Ec``.
        - G1 is assigned when both scores are negative.
        - G2M is assigned when the G2M score exceeds the S score.
        - S is assigned otherwise, including tied non-negative scores.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            s_genes: A list of S phase genes. If not provided then Scarf loads pre-saved genes accessible at
                     `scarf.quality_control.s_phase_genes`
            g2m_genes: A list of G2M phase genes. If not provided then Scarf loads pre-saved genes accessible at
                     `scarf.quality_control.g2m_phase_genes`
            n_bins: Number of bins into which average expression of genes is divided.
            rand_seed: A random values to set seed while sampling cells from a cluster randomly. (Default value: 4466)
            s_score_label: A base label for saving the S phase scores into a cell metadata column
                           (Default value: 'S_score')
            g2m_score_label: A base label for saving the G2M phase scores into a cell metadata column
                           (Default value: 'G2M_score')
            phase_label: A base label for saving the inferred cell cycle phase into a cell metadata column
                           (Default value: 'cell_cycle_phase')

        Returns: None
        """
        if from_assay is None:
            from_assay = self._defaultAssay
        assay = self._get_assay(from_assay)
        if cell_key is None:
            cell_key = "I"
        if s_genes is None:
            from ...quality_control.cell_cycle_genes import s_phase_genes

            s_genes = list(s_phase_genes)
        if g2m_genes is None:
            from ...quality_control.cell_cycle_genes import g2m_phase_genes

            g2m_genes = list(g2m_phase_genes)
        control_size = min(len(s_genes), len(g2m_genes))

        s_score = assay.score_features(
            s_genes, cell_key, control_size, n_bins, rand_seed
        )
        s_score_label = self._col_renamer(from_assay, cell_key, s_score_label)
        self.cells.insert(s_score_label, s_score, key=cell_key, overwrite=True)

        g2m_score = assay.score_features(
            g2m_genes, cell_key, control_size, n_bins, rand_seed
        )
        g2m_score_label = self._col_renamer(from_assay, cell_key, g2m_score_label)
        self.cells.insert(g2m_score_label, g2m_score, key=cell_key, overwrite=True)

        phase = assign_cell_cycle_phase(s_score, g2m_score)
        phase_label = self._col_renamer(from_assay, cell_key, phase_label)
        self.cells.insert(phase_label, np.asarray(phase), key=cell_key, overwrite=True)

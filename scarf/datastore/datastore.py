from typing import Literal
from collections.abc import Iterable

import numpy as np
import pandas as pd
from ..chunked import ChunkedArray
from loguru import logger

from ..assay import Assay, ATACassay, RNAassay
from ..feat_utils import hto_demux
from ..utils import ZARRLOC, controlled_compute, tqdmbar
from ..writers import create_zarr_dataset, create_zarr_obj_array
from .mapping_datastore import MappingDatastore

__all__ = ["DataStore"]


class DataStore(MappingDatastore):
    """This class extends MappingDatastore and consequently inherits methods of
    all the other DataStore classes.

    This class is the main user facing class as it provides most of the plotting functions.
    It also contains methods for cell filtering, feature selection, marker features identification,
    subsetting and aggregating cells. This class also contains methods that perform in-memory data exports.
    In other words, DataStore objects provide the primary interface to interact with the data.

    Args:
        zarr_loc: Path to Zarr file created using one of writer functions of Scarf.
        assay_types: A dictionary with keys as assay names present in the Zarr file and values as either one of:
                     'RNA', 'ADT', 'ATAC' or 'GeneActivity'.
        default_assay: Name of assay that should be considered as default. It is mandatory to provide this value
                       when DataStore loads a Zarr file for the first time.
        min_features_per_cell: Minimum number of non-zero features in a cell. If lower than this then the cell
                               will be filtered out.
        min_cells_per_feature: Minimum number of cells where a feature has a non-zero value. Genes with values
                               less than this will be filtered out.
        mito_pattern: Regex pattern to capture mitochondrial genes. (default: 'MT-')
        ribo_pattern: Regex pattern to capture ribosomal genes. (default: 'RPS|RPL|MRPS|MRPL')
        nthreads: Number of maximum threads to use in all multi-threaded functions
        zarr_mode: For read-write mode use r+' or for read-only use 'r'. (Default value: 'r+')
        workspace: Workspace for the data
        synchronizer: Used as `synchronizer` parameter when opening the Zarr file. Please refer to this page for
                      more details: https://zarr.readthedocs.io/en/stable/api/sync.html. By default
                      ThreadSynchronizer will be used.
    """

    def __init__(
        self,
        zarr_loc: ZARRLOC,
        assay_types: dict | None = None,
        default_assay: str | None = None,
        min_features_per_cell: int = 10,
        min_cells_per_feature: int = 20,
        mito_pattern: str | None = None,
        ribo_pattern: str | None = None,
        nthreads: int = 2,
        zarr_mode: str = "r+",
        workspace: str | None = None,
        synchronizer=None,
        zarrProfile: Literal["fast_local", "cloud"] | None = None,
        storage_options: dict | None = None,
    ):
        from ..storage.zarr_store import set_storage_profile

        set_storage_profile(zarrProfile)
        if zarr_mode not in ["r", "r+"]:
            raise ValueError(
                "ERROR: Zarr file can only be accessed using either 'r' or 'r+' mode"
            )
        super().__init__(
            zarr_loc=zarr_loc,
            assay_types=assay_types,
            default_assay=default_assay,
            min_features_per_cell=min_features_per_cell,
            min_cells_per_feature=min_cells_per_feature,
            mito_pattern=mito_pattern,
            ribo_pattern=ribo_pattern,
            nthreads=nthreads,
            zarr_mode=zarr_mode,
            workspace=workspace,
            synchronizer=synchronizer,
            storage_options=storage_options,
        )

    def get_assay(self, assay_name: str) -> Assay:
        """Returns the assay object for the given assay name.

        Args:
            assay_name: Name of the assay to be returned.

        Returns:
            Assay object
        """
        if assay_name not in self.assay_names:
            raise ValueError(f"ERROR: Assay {assay_name} not found in the Zarr file")
        else:
            return getattr(self, assay_name)

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

        This is a wrapper function for `filer_cells` and determines the threshold values to be used for each column.
        For each cell metadata column, the function models a normal distribution using the median value and standard
        deviation of the column and then determines the point estimates of values at `min_p` and `max_p`
        fraction of densities.

        Args:
            attrs: Column names to be used for filtering.
            min_p: Fractional density point to be used for calculating lower bounds of threshold.
            max_p: Fractional density point to be used for calculating lower bounds of threshold.
            show_qc_plots: If True then violin plots with per cell distribution of features will be shown. This does
                       not have an effect if `auto_filter` is False.

        Returns:
            None
        """
        from scipy.stats import norm

        if attrs is None:
            attrs = []
            for i in ["nCounts", "nFeatures", "percentMito", "percentRibo"]:
                i = f"{self._defaultAssay}_{i}"
                if i in self.cells.columns:
                    attrs.append(i)

        attrs_used = []
        for i in attrs:
            if i not in self.cells.columns:
                logger.warning(
                    f"{i} not found in cell metadata. Will ignore {i} for filtering"
                )
                continue
            a = self.cells.fetch_all(i)
            dist = norm(np.median(a), np.std(a))
            self.filter_cells(
                attrs=[i], lows=[dist.ppf(min_p)], highs=[dist.ppf(max_p)]
            )
            attrs_used.append(i)

        if show_qc_plots:
            self.plot_cells_dists(
                cols=attrs_used, sup_title="Pre-filtering distribution"
            )
            self.plot_cells_dists(
                cols=attrs_used,
                cell_key="I",
                color="coral",
                sup_title="Post-filtering distribution",
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

    def mark_hvgs(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        min_cells: int | None = None,
        top_n: int = 500,
        min_var: float = -np.inf,
        max_var: float = np.inf,
        min_mean: float = -np.inf,
        max_mean: float = np.inf,
        n_bins: int = 200,
        lowess_frac: float = 0.1,
        blacklist: str = "^MT-|^RPS|^RPL|^MRPS|^MRPL|^CCN|^HLA-|^H2-|^HIST",
        keep_bounds: bool = False,
        show_plot: bool = True,
        hvg_key_name: str = "hvgs",
        max_cells: int | None = np.inf,
        **plot_kwargs,
    ) -> None:
        """Identify and mark genes as highly variable genes (HVGs). This is a
        critical and required feature selection step and is only applicable to
        RNAassay type of assays.

        Args:
            from_assay: Assay to use for graph creation. If no value is provided then `defaultAssay` will be used
            cell_key: Cells to use for HVG selection. By default, all cells with True value in 'I' will be used.
                      The provided value for `cell_key` should be a column in cell metadata table with boolean values.
            min_cells: Minimum number of cells where a gene should have non-zero expression values for it to be
                       considered a candidate for HVG selection. Large values for this parameter might make it difficult
                       to identify rare populations of cells. Very small values might lead to a higher signal-to-noise
                       ratio in the selected features. By default, a value is set assuming smallest population has no
                       less than 1% of all cells. So for example, if you have 1000 cells (as per cell_key parameter)
                       then `min_cells` will be set to 10.
            max_cells: Maximum number of cells where a gene should have non-zero expression values for it to be
                       considered a candidate for HVG selection. This can be useful to filter out genes that are
                       expressed in too many cells. Default value is infinity, meaning no upper limit.
            top_n: Number of top most variable genes to be set as HVGs. This value is ignored if a value is provided
                   for `min_var` parameter. (Default: 500)
            min_var: Minimum variance threshold for HVG selection. (Default: -Infinity)
            max_var: Maximum variance threshold for HVG selection. (Default: Infinity)
            min_mean: Minimum mean value of expression threshold for HVG selection. (Default: -Infinity)
            max_mean: Maximum mean value of expression threshold for HVG selection. (Default: Infinity)
            n_bins: Number of bins into which the mean expression is binned. (Default: 200)
            lowess_frac: Between 0 and 1. The fraction of the data used when estimating the fit between mean and
                         variance. This is same as `frac` in statsmodels.nonparametric.smoothers_lowess.lowess
                         (Default: 0.1)
            blacklist: This is a regular expression (regex) string that can be used to exclude genes from being marked
                       as HVGs. By default, we exclude mitochondrial, ribosomal, some cell-cycle related, histone and
                       HLA genes. (Default: '^MT- | ^RPS | ^RPL | ^MRPS | ^MRPL | ^CCN | ^HLA- | ^H2- | ^HIST' )
            keep_bounds: If True, then the boundary values are retained and not filtered out (Default value: False)
            show_plot: If True then a diagnostic scatter plot is shown with HVGs highlighted. (Default: True)
            hvg_key_name: Base label for HVGs in the features metadata column. The value for
                          'cell_key' parameter is prepended to this value. (Default value: 'hvgs')
            plot_kwargs: These named parameters are passed to plotting.plot_mean_var

        Returns:
            None
        """

        if cell_key is None:
            cell_key = "I"
        assay = self._get_assay(from_assay)
        if type(assay) != RNAassay:  # noqa: E721
            raise TypeError(
                f"ERROR: This method of feature selection can only be applied to RNAassay type of assay. "
                f"The provided assay is {type(assay)} type"
            )
        if min_cells is None:
            min_cells = int(0.01 * self.cells.N)
            logger.info(
                f"Setting `min_cells` to {min_cells}. Only those genes that are present in atleast this number "
                f"of cells will be considered HVGs."
            )
        if max_cells is None:
            max_cells = np.inf
        assay.mark_hvgs(
            cell_key=cell_key,
            min_cells=min_cells,
            max_cells=max_cells,
            top_n=top_n,
            min_var=min_var,
            max_var=max_var,
            min_mean=min_mean,
            max_mean=max_mean,
            n_bins=n_bins,
            lowess_frac=lowess_frac,
            blacklist=blacklist,
            hvg_key_name=hvg_key_name,
            keep_bounds=keep_bounds,
            show_plot=show_plot,
            **plot_kwargs,
        )

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

    def run_marker_search(
        self,
        from_assay: str | None = None,
        group_key: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        gene_batch_size: int = 50,
        use_prenormed: bool = False,
        prenormed_store: str | None = None,
        n_threads: int | None = None,
        skip_save: bool = False,
        **norm_params,
    ) -> dict | None:
        """Identifies group specific features for a given assay.

        Please check out the ``find_markers_by_rank`` function for further details of how marker features for groups
        are identified. The results are saved into the Zarr hierarchy under `markers` group.

        Args:
            from_assay: Name of the assay to be used. If no value is provided then the default assay will be used.
            group_key: Required parameter. This has to be a column name from cell metadata table. This column dictates
                       how the cells will be grouped. Usually this would be a column denoting cell clusters.
            cell_key: To run the test on specific subset of cells, provide the name of a boolean column in
                        the cell metadata table. (Default value: 'I')
            feat_key: Boolean feature metadata column selecting features (default: ``'I'``).
            gene_batch_size: Number of genes loaded per batch; all selected cells are loaded for each batch.
            use_prenormed: If True, use prenormalized cache from ``Assay.save_normed_for_query``.
                           (Default value: False)
            prenormed_store: Custom Zarr group with prenormalized values (default: None).
            n_threads: Threads for marker search when ``use_prenormed`` is True.
            skip_save: If True, return results without writing to Zarr.
            **norm_params: Extra keyword arguments forwarded to ``normed``.

        Returns:
            Marker dict if ``skip_save`` is True, else None.
        """
        from ..markers import find_markers_by_rank

        if group_key is None:
            raise ValueError(
                "ERROR: Please provide a value for `group_key`. This should be the name of a column from "
                "cell metadata object that has information on how cells should be grouped."
            )
        from_assay, cell_key, _ = self._get_latest_keys(from_assay, cell_key, None)
        if feat_key is None:
            feat_key = "I"
        if n_threads is None:
            n_threads = self.nthreads
        assay = self._get_assay(from_assay)

        slot_name = f"{cell_key}__{group_key}"
        z = self.zw[assay.name]
        if "markers" not in z:
            z.create_group("markers")
        group = z["markers"].create_group(slot_name, overwrite=True)

        markers = find_markers_by_rank(
            assay=assay,
            group_key=group_key,
            cell_key=cell_key,
            feat_key=feat_key,
            batch_size=gene_batch_size,
            use_prenormed=use_prenormed,
            prenormed_store=prenormed_store,
            n_threads=n_threads,
            **norm_params,
        )

        if skip_save:
            return markers
        else:
            for i in markers:
                g = group.create_group(str(i))
                vals = markers[i]
                if len(vals) != 0:
                    for j in vals.columns:
                        create_zarr_obj_array(g, j, vals[j].values, dtype=vals[j].dtype)
            return None

    def run_pseudotime_marker_search(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        pseudotime_key: str | None = None,
        min_cells: int = 10,
        gene_batch_size: int = 50,
        **norm_params,
    ) -> None:
        """Identify genes that a correlated with a given pseudotime ordering of
        cells. The results are saved in feature attribute tables. For example,
        the r value can be found under, 'I__RNA_pseudotime__r' and the
        corresponding p values can be found under 'I__RNA_pseudotime__p' The
        values are saved with patten {cell_key}__{regressor_key}__r/p.

        Args:
            from_assay: Name of the assay to be used. If no value is provided then the default assay will be used.
            cell_key: To run the test on specific subset of cells, provide the name of a boolean column in
                        the cell metadata table. (Default value: 'I')
            feat_key: Boolean feature metadata column selecting features (default: ``'I'``).
            pseudotime_key: Required parameter. This has to be a column name from cell metadata table. This column
                            contains values for pseudotime ordering of the cells.
            min_cells: Minimum number of cells where a gene should have non-zero value to be considered for test.
                       (Default: 10)
            gene_batch_size: Number of genes to be loaded in memory at a time. (Default value: 50).
            **norm_params: Extra keyword arguments forwarded to ``normed``.

        Returns: None
        """

        from ..markers import find_markers_by_regression

        if pseudotime_key is None:
            raise ValueError(
                "ERROR: Please provide a value for `pseudotime_key`. This should be the name of a column from "
                "cell metadata object where pseudotime values are stored. If you ran `run_pseudotime_scoring` then "
                "the values are stored under `RNA_pseudotime` by default."
            )
        if cell_key is None:
            cell_key = "I"
        if feat_key is None:
            feat_key = "I"
        assay = self._get_assay(from_assay)
        ptime = assay.cells.fetch(pseudotime_key, key=cell_key)
        markers = find_markers_by_regression(
            assay=assay,
            cell_key=cell_key,
            feat_key=feat_key,
            regressor=ptime,
            min_cells=min_cells,
            batch_size=gene_batch_size,
            **norm_params,
        )
        assay.feats.insert(
            f"{cell_key}__{pseudotime_key}__r",
            np.array(markers["r_value"].values),
            overwrite=True,
        )
        assay.feats.insert(
            f"{cell_key}__{pseudotime_key}__p",
            np.array(markers["p_value"].values),
            overwrite=True,
        )

    def run_pseudotime_aggregation(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        pseudotime_key: str | None = None,
        cluster_label: str | None = None,
        min_exp: float = 1e-3,
        window_size: int = 200,
        chunk_size: int = 50,
        smoothen: bool = True,
        z_scale: bool = True,
        n_neighbours: int = 11,
        n_clusters: int = 10,
        batch_size: int = 100,
        ann_params: dict | None = None,
        nan_cluster_value: int | str = -1,
    ) -> None:
        """This method performs clustering of features based on pseudotime
        ordered cells. The values from the pseudotime ordered cells are
        smoothened, scaled and binned. The resulting binned matrix is used to
        perform a KNN-Paris clustering of the features. This function can be
        used an alternative to `run_marker_search` and
        `run_pseudotime_marker_search`

        Args:
            from_assay: Name of the assay to be used. If no value is provided then the default assay will be used.
            cell_key: To run the test on specific subset of cells, provide the name of a boolean column in
                      the cell metadata table. (Default value: The cell key that was used to generate the latest graph)
            feat_key: To use only a subset of features, provide the name of a boolean column in the feature
                      metadata/attribute table. Default value: The cell key that was used to generate the latest graph)
            pseudotime_key: Required parameter. This has to be a column name from cell attribute table. This
                            column contains values for pseudotime ordering of the cells.
            cluster_label: Required parameter. Name of the column under which the feature cluster identity will be
                           saved in the feature attribute table.
            min_exp: Features with mean normalized expression than this value are dropped and hence not assigned
                     a cluster identity (Default value: 10)
            window_size: The window for calculating rolling mean of feature values along pseudotime ordering. Larger
                         values will slow down processing but produce more smoothened. The choice of value here depends
                         on the number of cells in the analysis. Larger value will be useful to produce smooth profiles
                         when number of cells are large. (Default value: 200)
            chunk_size: Number of bins of cells to create. Larger values will increase memory consumption but will
                        provide improved resolution (Default value: 50)
            smoothen: Whether to perform the rolling window averaging (Default value: True)
            z_scale: Whether to perform standard scaling of each feature. Turning this off may not be a good choice.
                     (Default value: True)
            n_neighbours: Number of neighbours to save in the KNN graph of features(Default value: 11)
            n_clusters: Number of feature clusters to create. (Default value: 10)
            batch_size: Number of features to load at a time when processing the data. Larger values will increase
                        memory consumption (Default value: 100)
            ann_params: The parameter to forward to HNSWlib index instantiation step. (Default value: {})
            nan_cluster_value: The value to use for features that are not assigned a cluster identity.
                               (Default value: -1)

        Returns: None
        """
        from ..markers import knn_clustering

        from_assay, cell_key, _ = self._get_latest_keys(from_assay, cell_key, feat_key)
        if feat_key is None:
            feat_key = "I"
        assay = self._get_assay(from_assay)

        if pseudotime_key is None:
            raise ValueError(
                "ERROR: Please provide a value for `pseudotime_key` parameter. This is the column in "
                "the cell attribute table that contains the pseudotime values."
            )
        if cluster_label is None:
            raise ValueError(
                "ERROR: Please provide a value for cluster_label. "
                "It will be used to create new column in feature attribute table. The module identity "
                "of each feature will be saved under this column name. If this column already exists "
                "then it will be overwritten."
            )

        df, feat_ids = assay.save_aggregated_ordering(
            cell_key=cell_key,
            feat_key=feat_key,
            ordering_key=pseudotime_key,
            min_exp=min_exp,
            window_size=window_size,
            chunk_size=chunk_size,
            smoothen=smoothen,
            z_scale=z_scale,
            batch_size=batch_size,
        )
        if ann_params is None:
            ann_params = {}
        clusts = knn_clustering(
            d_array=df,
            n_neighbours=n_neighbours,
            n_clusters=n_clusters,
            n_threads=self.nthreads,
            ann_params=ann_params,
        )
        temp = np.ones(assay.feats.N) * -1
        temp[feat_ids] = clusts
        assay.feats.insert(
            cluster_label, temp.astype(int), fill_value=-1, overwrite=True
        )
        return None

    def get_markers(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        group_key: str | None = None,
        group_id: str | int | None = None,
        min_score: float = 0.25,
        min_frac_exp: float = 0.2,
    ) -> pd.DataFrame:
        """Returns a table of markers features obtained through
        `run_marker_search` for a given group.

        The table contains names of marker features and feature ids are used as table index.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: To run the test on specific subset of cells, provide the name of a boolean column in
                        the cell metadata table.
            group_key: Required parameter. This has to be a column name from cell metadata table.
                       Usually this would be a column denoting cell clusters. Please use the same value as used
                       when ran `run_marker_search`
            group_id: This is one of the value in `group_key` column of cell metadata.
                      Results are returned for this group
            min_score: This value dictates how specific the feature value has to be in a group before it is
                       considered a marker for that group. The value has to be greater than 0 but less than or equal to
                       1 (Default value: 0.25)
            min_frac_exp: Minimum fraction of cells in a group that must have a non-zero value for a gene to be
                          considered a marker for that group.

        Returns:
            Pandas dataframe
        """

        if cell_key is None:
            from_assay, cell_key, _ = self._get_latest_keys(from_assay, cell_key, None)
        if group_key is None:
            raise ValueError(
                "ERROR: Please provide a value for group_key. "
                "This should be same as used for `run_marker_search`"
            )
        assay = self._get_assay(from_assay)
        try:
            g = assay.z["markers"][f"{cell_key}__{group_key}"]  # type: ignore
        except KeyError:
            raise KeyError(
                "ERROR: Couldn't find the location of markers. Please make sure that you have already called "
                "`run_marker_search` method with same value of `cell_key` and `group_key`"
            )
        out_cols = [
            "feature_index",
            "score",
            "mean",
            "mean_rest",
            "frac_exp",
            "frac_exp_rest",
            "fold_change",
            "p_value",
        ]
        gids = sorted(set(assay.cells.fetch(group_key, key=cell_key)))
        if group_id is not None:
            gids = [group_id]

        dfs = []
        for gid in gids:
            group_name = str(gid)
            if group_name in g:
                available_cols = [col for col in out_cols if col in g[group_name]]
                cols = [g[group_name][x][:] for x in available_cols]  # type: ignore
                df = pd.DataFrame(
                    cols,
                    index=available_cols,
                ).T
                df["group_id"] = gid
                df["feature_name"] = assay.feats.fetch_all("names")[
                    df.feature_index.astype("int")
                ]
                df = df[["group_id", "feature_name"] + available_cols]
            else:
                logger.debug(f"No markers found for {gid} returning empty dataframe")
                df = pd.DataFrame([[] for _ in out_cols], index=out_cols).T
                df["group_id"] = []
                df["feature_name"] = []
                df = df[["group_id", "feature_name"] + out_cols]
            dfs.append(df)
        dfs = pd.concat(dfs)
        return dfs[
            (dfs.score >= min_score) & (dfs.frac_exp >= min_frac_exp)
        ].reset_index(drop=True)

    def export_markers_to_csv(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        group_key: str | None = None,
        csv_filename: str | None = None,
        min_score: float = 0.25,
        min_frac_exp: float = 0.2,
    ) -> None:
        """Export markers of each cluster/group to a CSV file where each column
        contains the marker names sorted by score (descending order, highest
        first). This function does not export the scores of markers as they can
        be obtained using `get_markers` function.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: To run the test on specific subset of cells, provide the name of a boolean column in
                        the cell metadata table.
            group_key: Required parameter. This has to be a column name from cell metadata table.
                       Usually this would be a column denoting cell clusters. Please use the same value as used
                       when ran `run_marker_search`
            csv_filename: Required parameter. Name, with path, of CSV file where the marker table is to be saved.
            min_score: This value dictates how specific the feature value has to be in a group before it is
                       considered a marker for that group. The value has to be greater than 0 but less than or equal to
                       1 (Default value: 0.25)
            min_frac_exp: Minimum fraction of cells in a group that must have a non-zero value for a gene to be
                          considered a marker for that group.

        Returns:
        """
        # Not testing the values of from_assay and cell_key because they will be tested in `get_markers`
        if group_key is None:
            raise ValueError(
                "ERROR: Please provide a value for group_key. "
                "This should be same as used for `run_marker_search`"
            )
        if csv_filename is None:
            raise ValueError(
                "ERROR: Please provide a value for parameter `csv_filename`"
            )
        from_assay, cell_key, _ = self._get_latest_keys(from_assay, cell_key, None)
        clusters = self.cells.fetch(group_key, key=cell_key)
        markers_table = {}
        for group_id in sorted(set(clusters)):
            m = self.get_markers(
                from_assay=from_assay,
                cell_key=cell_key,
                group_key=group_key,
                group_id=group_id,
                min_score=min_score,
                min_frac_exp=min_frac_exp,
            )
            if len(m) > 0:
                markers_table[group_id] = m["feature_name"].reset_index(drop=True)
            else:
                markers_table[group_id] = pd.Series([])
        pd.DataFrame(markers_table).fillna("").to_csv(csv_filename, index=False)
        return None

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
        - A phase score is calculated as: Ep-Ec Cell cycle phase is assigned to each cell based on following rule set:
        - G1 phase: S score < -1 > G2M sore
        - S phase: S score > G2M score
        - G2M phase: G2M score > S score

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            s_genes: A list of S phase genes. If not provided then Scarf loads pre-saved genes accessible at
                     `scarf.bio_data.s_phase_genes`
            g2m_genes: A list of G2M phase genes. If not provided then Scarf loads pre-saved genes accessible at
                     `scarf.bio_data.g2m_phase_genes`
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
            from ..bio_data import s_phase_genes

            s_genes = list(s_phase_genes)
        if g2m_genes is None:
            from ..bio_data import g2m_phase_genes

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

        phase = pd.Series(
            ["S" for _ in range(self.cells.fetch(cell_key, key=cell_key).sum())]
        )
        phase[g2m_score > s_score] = "G2M"
        phase[(g2m_score < 0) & (s_score < 0)] = "G1"
        phase_label = self._col_renamer(from_assay, cell_key, phase_label)
        self.cells.insert(
            phase_label, np.array(phase.values), key=cell_key, overwrite=True
        )

    def add_grouped_assay(
        self,
        from_assay: str | None = None,
        group_key: str | None = None,
        assay_label: str | None = None,
        exclude_values: list | None = None,
    ) -> None:
        """Add a new assay to the DataStore by grouping together multiple
        features and taking their means. This method requires that the features
        are already assigned a group/cluster identity. The new assay will have
        all the cells but only features that marked by 'feat_key' and contain a
        group identity not present in `exclude_values`.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            group_key: This is mandatory parameter. Name of the column in feature metadata table to be used for
                       grouping features.
            assay_label: This is mandatory parameter. A name for the new assay.
            exclude_values: These groups/clusters will be ignored and not added to new assay. By default, it is set to
                            [-1], this means that all the features that have the group identity of -1 are not used.

        Returns: None
        """

        from ..writers import create_zarr_count_assay

        if assay_label is None:
            raise ValueError(
                "ERROR: Please provide a value for `assay_label`. "
                "It will be used to create a new assay"
            )
        if group_key is None:
            raise ValueError(
                "ERROR: Please provide a value for `group_key`. "
                "This should be name of the column in the feature attribute table that contains the group/cluster "
                "identity of each feature."
            )

        assay = self._get_assay(from_assay)
        groups = assay.feats.fetch_all(group_key)
        if exclude_values is None:
            exclude_values = [-1]
        group_set = sorted(set(groups).difference(exclude_values))

        module_ids = [f"group_{x}" for x in group_set]
        g = create_zarr_count_assay(
            z=self.zw,
            assay_name=assay_label,
            workspace=self.workspace,
            chunk_size=assay.rawData.chunksize,  # type: ignore
            n_cells=assay.cells.N,
            feat_ids=module_ids,
            feat_names=module_ids,
            dtype="float",
        )

        cell_idx = np.array(list(range(assay.cells.N)))
        for n, i in tqdmbar(
            enumerate(group_set), desc="Writing to Zarr", total=len(group_set)
        ):
            feat_idx = np.where(groups == i)[0]
            temp = np.zeros(assay.cells.N)
            temp[cell_idx] = (
                assay.normed(cell_idx=cell_idx, feat_idx=feat_idx)
                .mean(axis=1)
                .compute()
            )
            g[:, n] = temp

        self._load_assays(min_cells=0, custom_assay_types={assay_label: "Assay"})
        self._ini_cell_props(min_features=0, mito_pattern="", ribo_pattern="")

    def add_melded_assay(
        self,
        from_assay: str | None = None,
        external_bed_fn: str | None = None,
        assay_label: str | None = None,
        peaks_col: str = "ids",
        scalar_coeff: float = 1e5,
        renormalization: bool = True,
        assay_type: str = "Assay",
    ) -> None:
        """This method performs "assay melding" and can be only be used for
        assay's wherein features have genomic coordinates. In the process of
        melding the input genomic coordinates from `external_bed_fn` are
        intersected with the assay's features. Based on this intersection a
        mapping is created wherein each coordinate interval maps to one or more
        feature coordinates from the assay.

        This method has been designed for snATAC-Seq data and can be used to quantify accessibility of specific
        genomic loci such as gene bodies, promoters, enhancers, motifs, etc.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            external_bed_fn: This is mandatory parameter. This file should be a BED format file with at least five
                             columns containing: chromosome, start position, end position, feature id and feature name.
                             Coordinates should be in half open format. That means that actual end position is -1
            assay_label: This is mandatory parameter. A name for the new assay.
            peaks_col: The column in feature metadata table that contains the genomic coordinate information of each
                       feature. The genomic coordinates are represented as strings in this format: chr:start-end
                       (Default value: 'ids')
            scalar_coeff: An arbitrary scalar multiplier. Only used when renormalization is True (Default value: 1e5)
            renormalization: Whether to rescale the sum of feature values for each cell to `scalar_coeff`
                         (Default value: True)
            assay_type: The new assay (melded assay) is saved as this type. This can be any type of Assay class from
                        `assay` module. Please provide string representation of class. By default, the assay is assigned
                        a generic class and has a dummy normalization function (Default value: 'Assay')

        Returns:
            None
        """

        from ..meld_assay import coordinate_melding

        if assay_label is None:
            raise ValueError(
                "ERROR: Please provide a value for `assay_label`. "
                "It will be used to create a new assay"
            )
        if external_bed_fn is None:
            raise ValueError(
                "ERROR: Please provide a value for `feature_bed_fn`. "
                "This should be a BED format file with atleast 5 columns."
            )

        assay = self._get_assay(from_assay)
        feature_bed = pd.read_csv(external_bed_fn, header=None, sep="\t").sort_values(
            by=[0, 1]  # type: ignore
        )

        peaks_coords = assay.feats.fetch_all(peaks_col)
        for n, i in enumerate(peaks_coords):
            error_msg = (
                f"ERROR: Coordinate format check failed for element: {i} (position {n}). The format should "
                f"be chr:start-end. Please note the colon and hyphen position"
            )
            if len(i.split(":")) != 2:
                raise ValueError(error_msg)
            if len(i.split(":")[1].split("-")) != 2:
                raise ValueError(error_msg)

        coordinate_melding(
            assay,
            workspace=self.workspace,
            feature_bed=feature_bed,
            new_assay_name=assay_label,
            peaks_col=peaks_col,
            scalar_coeff=scalar_coeff,
            renormalization=renormalization,
        )

        self._load_assays(min_cells=10, custom_assay_types={assay_label: assay_type})
        self._ini_cell_props(min_features=0, mito_pattern=None, ribo_pattern=None)

    def make_bulk(
        self,
        from_assay: str | None = None,
        cell_key: str = "I",
        group_key: str | None = None,
        secondary_group_key: str | None = None,
        aggr_type: Literal["mean", "sum"] = "mean",
        return_fraction: bool = False,
        feature_label: Literal["index", "id", "name"] = "index",
        remove_empty_features: bool = True,
        pseudo_reps: int = 1,
        null_vals: list | None = None,
        secondary_null_vals: list | None = None,
        random_seed: int = 4466,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """Merge data from cells to create a bulk profile.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Name of the column in cell metadata table to be used for selecting cells.
            group_key: Name of the column in cell metadata table to be used for grouping cells.
            secondary_group_key: Name of the column in cell metadata table to be used for sub-grouping cells.
            aggr_type: Type of aggregation to be used. Can be either 'mean' or 'sum'. (Default value: 'mean')
            return_fraction: Return the fraction of cells expressing a gene in each group. (Default value: False)
            feature_label: The column in feature metadata table to use as row labels. (Default value: 'index')
            pseudo_reps: Within each group, cells will randomly be split into `pseudo_reps` partitions. Each partition
                         is considered a pseudo-replicate. (Default value: 3)
            remove_empty_features: Remove features that are not expressed in any cell. (Default value: True)
            null_vals: Values to be considered as missing values in the `group_key` column. These values will be skipped.
            secondary_null_vals: Values to be considered as missing values in the `secondary_group_key` column.
                                 These values will be skipped.
            random_seed: A random values to set seed while creating `pseudo_reps` partitions cells randomly.

        Returns:
            A pandas dataframe containing the bulk profile. If `return_fraction` is True, then a tuple of two dataframes
            is returned. The second dataframe contains the fraction of cells expressing each feature in each group.
        """

        def make_reps(v, n_reps: int, seed: int) -> list[np.ndarray]:
            v = list(v)
            random_state = np.random.RandomState(seed)
            shuffled_idx = random_state.choice(v, len(v), replace=False)
            rep_idx = np.array_split(shuffled_idx, n_reps)
            return [np.array(sorted(x)) for x in rep_idx]

        if pseudo_reps < 1:
            pseudo_reps = 1
        if null_vals is None:
            null_vals = []
        if secondary_null_vals is None:
            secondary_null_vals = []
        if group_key is None:
            raise ValueError("ERROR: Please provide a value for `group_key` parameter")
        else:
            groups = self.cells.fetch_all(group_key)
            groups_set = sorted(set(self.cells.fetch(group_key, key=cell_key)))
        if secondary_group_key is None:
            sec_groups = [None]
            sec_groups_set = [None]
        else:
            sec_groups = self.cells.fetch_all(secondary_group_key)
            sec_groups_set = sorted(
                set(self.cells.fetch(secondary_group_key, key=cell_key))
            )

        assay = self._get_assay(from_assay)

        vals = {}
        fracs = {}
        all_feat_idx = np.arange(assay.feats.N)
        for g in tqdmbar(groups_set):
            if g in null_vals:
                continue
            for sg in sec_groups_set:  # type: ignore
                if sg in secondary_null_vals:
                    continue
                if sg is None and len(sec_groups) == 1:
                    g_idx = np.where(groups == g)[0]
                else:
                    g_idx = np.where((groups == g) & (sec_groups == sg))[0]
                rep_indices = make_reps(g_idx, pseudo_reps, random_seed)
                for n, idx in enumerate(rep_indices):
                    if sg is None and len(sec_groups) == 1:
                        col_name = f"{g}"
                    else:
                        col_name = f"{g}_{sg}"
                    if pseudo_reps > 1:
                        col_name += f"_Rep{n + 1}"
                    if len(idx) == 0:
                        vals[col_name] = np.zeros(assay.feats.N)
                        continue
                    if aggr_type == "sum":
                        vals[col_name] = controlled_compute(
                            assay.rawData[idx].sum(axis=0), self.nthreads
                        )
                    elif aggr_type == "mean":
                        vals[col_name] = controlled_compute(
                            assay.normed(cell_idx=idx, feat_idx=all_feat_idx).mean(
                                axis=0
                            ),
                            self.nthreads,
                        )
                    else:
                        raise ValueError(
                            "ERROR: `aggr_type` can only be either 'sum' or 'mean'"
                        )
                    if return_fraction:
                        fracs[col_name] = (
                            (assay.rawData[idx] > 0).mean(axis=0).compute()
                        )

        vals = pd.DataFrame(vals).fillna(0)

        empty_idx = None
        if remove_empty_features:
            empty_idx = vals.sum(axis=1) != 0
            vals = vals.loc[empty_idx]

        if feature_label == "id":
            vals.set_index(
                pd.Series(assay.feats.fetch_all("ids")).reindex(vals.index).values,
                inplace=True,
                drop=True,
            )
        elif feature_label == "name":
            vals.set_index(
                pd.Series(assay.feats.fetch_all("names")).reindex(vals.index).values,
                inplace=True,
                drop=True,
            )

        if return_fraction:
            fracs = pd.DataFrame(fracs).fillna(0)
            if empty_idx is not None:
                fracs = fracs[empty_idx]
            fracs.set_index(vals.index, inplace=True, drop=True)
            return vals, fracs
        else:
            return vals

    def to_anndata(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        layers: dict | None = None,
    ):
        """Writes an assay of the Zarr hierarchy to AnnData file format.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Name of column from cell metadata that has boolean values. This is used to subset cells
            layers: A mapping of layer names to assay names. Ex. {'spliced': 'RNA', 'unspliced': 'URNA'}. The raw data
                    from the assays will be stored as sparse arrays in the corresponding layer in anndata.

        Returns: anndata object
        """
        try:
            # noinspection PyPackageRequirements
            from anndata import AnnData  # type: ignore
        except ImportError:
            logger.error(
                "Package anndata is not installed because its an optional dependency. "
                "Install via `pip install anndata` or `conda install anndata -c conda-forge`"
            )
            return None

        if cell_key is None:
            cell_key = "I"
        assay = self._get_assay(from_assay)
        df = self.cells.to_pandas_dataframe(self.cells.columns, key=cell_key)
        obs = df.reset_index(drop=True).set_index("ids")
        df = assay.feats.to_pandas_dataframe(assay.feats.columns)
        var = df.rename(columns={"ids": "gene_ids"}).set_index("gene_ids")
        adata = AnnData(assay.to_raw_sparse(cell_key), obs=obs, var=var)
        if layers is not None:
            for layer, assay_name in layers.items():
                adata.layers[layer] = self._get_assay(assay_name).to_raw_sparse(
                    cell_key
                )
        return adata

    def show_zarr_tree(self, start: str = "/", depth: int = 2) -> None:
        """Prints the Zarr hierarchy of the DataStore.

        Args:
            start: Location in Zarr hierarchy to be used as the root for display
            depth: Depth of Zarr hierarchy to be displayed.

        Returns:
            None
        """
        from ..storage.zarr_store import array_info

        root = start.strip("/")
        node = self.zw if root == "" else self.zw[root]
        print(node.tree(level=depth))
        for key in node.array_keys():
            print(f"  {key}: {array_info(node[key])}")

    def calc_membership_strength(
        self, from_assay: str, cell_key: str, feat_key: str, clust_key: str
    ) -> None:
        """Store per-cell cluster membership strength from the latest KNN graph.

        For each cell, computes the fraction of KNN neighbors sharing the most
        common cluster label and saves it in cell metadata.

        Args:
            from_assay: Assay used to locate the KNN graph.
            cell_key: Boolean column selecting cells.
            feat_key: Feature key used when the graph was built.
            clust_key: Cell metadata column with cluster assignments.

        Returns:
            None
        """
        loc = self._get_latest_graph_loc(
            from_assay=from_assay, cell_key=cell_key, feat_key=feat_key
        )
        n_cells, k = self._get_graph_ncells_k(graph_loc=loc)
        clusts = self.cells.fetch(clust_key, key=cell_key)
        v = pd.DataFrame(clusts[self.zw[loc]["edges"][:, 1].reshape(k, n_cells)])
        x = np.array([v[x].value_counts().index[0] for x in v])
        self.cells.insert(
            f"{from_assay}_{cell_key}_cluster_membership_strength",
            (np.array((v == x).sum().values) / k).round(3),
            key=cell_key,
            overwrite=True,
        )
        return None

    def smart_label(
        self,
        to_relabel: str,
        base_label: str,
        cell_key: str = "I",
        new_col_name: str | None = None,
    ) -> None | list[str]:
        """A convenience function to relabel the values in a cell attribute
        column (A) based on the values in another cell attribute column (B).
        For each unique value in A, the most frequently occurring value in B is
        found. If two or more values in A have maximum overlap with the same
        value in B, then they all get the same label as B along with different
        suffixes like, 'a', 'b', etc. The suffixes are ordered based on where
        the largest fraction of the B label lies. If one label from A takes up
        multiple labels from B then all the labels from B are included, and they
        are delimited by hyphens.

        Args:
            to_relabel: Cell attributes column to relabel
            base_label: Cell attributes column to relabel
            cell_key: Cell key fetching column values
            new_col_name: Name of new column where relabeled values will be saved. If None then values
                          are returned and not saved in cell attributes table

        Returns: None or a list of relabelled values
        """
        df = pd.crosstab(
            self.cells.fetch(base_label, key=cell_key),
            self.cells.fetch(to_relabel, key=cell_key),
        )
        normed_frac = df.divide(df.sum(axis=1), axis="index")
        idxmax = df.idxmax()
        new_names = {}
        for i in sorted(idxmax.unique()):
            j = normed_frac[idxmax[idxmax == i].index].loc[i]
            j = j.sort_values(ascending=False).index
            for n, k in enumerate(j, start=1):
                a = chr(ord("@") + n)
                new_names[k] = f"{i}{a.lower()}"

        missing_vals = list(set(df.index).difference(idxmax.unique()))
        if len(missing_vals) > 0:
            miss_idxmax = df.loc[missing_vals].idxmax(axis=1).to_dict()
            for k, v in miss_idxmax.items():
                new_names[v] = f"{new_names[v][:-1]}-{k}{new_names[v][-1]}"

        ret_val = [new_names[x] for x in self.cells.fetch(to_relabel, key=cell_key)]
        if new_col_name is None:
            return ret_val
        else:
            self.cells.insert(new_col_name, ret_val, overwrite=True)

    def plot_cells_dists(
        self,
        from_assay: str | None = None,
        cols: list[str] | None = None,
        cell_key: str | None = None,
        group_key: str | None = None,
        color: str = "steelblue",
        cmap: str = "tab20",
        fig_size: tuple | None = None,
        label_size: float = 10.0,
        title_size: float = 10.0,
        sup_title: str | None = None,
        sup_title_size: float = 12.0,
        scatter_size: float = 1.0,
        max_points: int = 10000,
        show_on_single_row: bool = True,
        show_fig: bool = True,
    ) -> None:
        """Makes violin plots of the distribution of values present in cell
        metadata. This method is designed to distribution of nCounts,
        nFeatures, percentMito and percentRibo cell attributes.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cols: Column names from cell metadata table to be used for plotting. Be default, nCounts, nFeatures,
                  percentMito and percentRibo columns are chosen.
            cell_key: One of the columns from cell metadata table that indicates the cells to be used for plotting.
                      The values in the chosen column should be boolean (Default value: 'I')
            group_key: A column name from cell metadata table that indicates how cells should be grouped. This can be
                       any column that has either boolean or categorical values. By default, no grouping will be
                       performed (Default value: None)
            color: Face color of the violin plots. The value can be valid matplotlib named colour. This is used only
                   when there is a single group. (Default value: 'steelblue')
            cmap: A matplotlib colormap to be used to color different groups. (Default value: 'tab20')
            fig_size: A tuple of figure width and figure height (Default value:  Automatically determined by `plot_qc`)
            label_size: The font size of y-axis labels (Default value: 10.0)
            title_size: The font size of title. Median value is printed as title of each violin plot
                        (Default value: 10.0)
            sup_title: The title for complete figure panel (Default value: 12.0 )
            sup_title_size: The font size of title for complete figure panel (Default value: 12.0 )
            scatter_size: Size of each point in the violin plot (Default value: 1.0)
            max_points: Maximum number of points to display over violin plot. Random uniform sampling will be performed
                        to bring down the number of datapoints to this value. This does not affect the violin plot.
                        (Default value: 10000)
            show_on_single_row: Show all subplots in a single row. It might be useful to set this to False if you have
                                too many groups within each subplot (Default value: True)
            show_fig: Whether to render the figure and display it using plt.show() (Default value: True)

        Returns:
            None
        """

        from ..plots import plot_qc

        if from_assay is None:
            from_assay = self._defaultAssay
        if cell_key is None:
            # Show all cells
            pass

        if cols is not None:
            if type(cols) != list:  # noqa: E721
                raise ValueError("ERROR: 'cols' argument must be of type list")
            plot_cols = []
            for i in cols:
                if i in self.cells.columns:
                    if i not in plot_cols:
                        plot_cols.append(i)
                else:
                    logger.warning(f"{i} not found in cell metadata")
        else:
            cols = ["nCounts", "nFeatures", "percentMito", "percentRibo"]
            cols = [f"{from_assay}_{x}" for x in cols]
            plot_cols = [x for x in cols if x in self.cells.columns]

        debug_print_cols = "\n".join(plot_cols)
        logger.debug(
            f"(plot_cells_dists): Will plot following columns: {debug_print_cols}"
        )

        df = self.cells.to_pandas_dataframe(plot_cols)
        if group_key is not None:
            df["groups"] = self.cells.to_pandas_dataframe([group_key])
        else:
            df["groups"] = np.zeros(len(df))
        if cell_key is not None:
            idx = self.cells.active_index(cell_key)
            df = df.reindex(idx)

        plot_qc(
            df,
            color=color,
            cmap=cmap,
            fig_size=fig_size,
            label_size=label_size,
            title_size=title_size,
            sup_title=sup_title,
            sup_title_size=sup_title_size,
            scatter_size=scatter_size,
            max_points=max_points,
            show_on_single_row=show_on_single_row,
            show_fig=show_fig,
        )
        return None

    def plot_layout(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        layout_key: str | None = None,
        color_by: str | None = None,
        subselection_key: str | None = None,
        size_vals: np.ndarray | list[float] | None = None,
        clip_fraction: float = 0.01,
        width: float = 6,
        height: float = 6,
        default_color: str = "steelblue",
        cmap: str | None = None,
        color_key: dict | None = None,
        mask_values: list | None = None,
        mask_name: str = "NA",
        mask_color: str = "k",
        point_size: float = 10,
        do_shading: bool = False,
        shade_npixels: int = 1000,
        shade_min_alpha: int = 10,
        spread_pixels: int = 1,
        spread_threshold: float = 0.2,
        ax_label_size: float = 12,
        frame_offset: float = 0.05,
        spine_width: float = 0.5,
        spine_color: str = "k",
        displayed_sides: tuple = ("bottom", "left"),
        legend_ondata: bool = True,
        legend_onside: bool = True,
        legend_size: float = 12,
        legends_per_col: int = 20,
        title: str | list[str] | None = None,
        title_size: int = 12,
        hide_title: bool = False,
        cbar_shrink: float = 0.6,
        marker_scale: float = 70,
        lspacing: float = 0.1,
        cspacing: float = 1,
        shuffle_df: bool = False,
        sort_values: bool = False,
        savename: str | None = None,
        save_dpi: int = 300,
        ax=None,
        force_ints_as_cats: bool = True,
        n_columns: int = 4,
        w_pad: float = 1,
        h_pad: float = 1,
        show_fig: bool = True,
        scatter_kwargs: dict | None = None,
    ):
        """Create a scatter plot with a chosen layout. The method fetches the
        coordinates based from the cell metadata columns with `layout_key`
        prefix. DataShader library is used to draw fast rasterized image is
        `do_shading` is True. This can be useful when large number of cells are
        present to quickly render the plot and avoid over-plotting. The
        description of shading parameters has mostly been copied from the
        Datashader API that can be found here:
        https://holoviews.org/_modules/holoviews/operation/datashader.html.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: One of the columns from cell metadata table that indicates the cells to be used.
                      The values in the chosen column should be boolean (Default value: 'I')
            layout_key: A prefix to cell metadata columns that contains the coordinates for the 2D layout of the cells.
                        For example, 'RNA_UMAP' or 'RNA_tSNE'. If a list of prefixes is provided a grid of plots will be
                        made.
            color_by: One (or a list) of the columns of the metadata table or a feature name (for example gene, GATA2).
                      If a list of names is provided a grid of plots will be made.
                      (Default: None)
            subselection_key: A column from cell metadata table to be used to show only a sub-selection of cells. This
                              key can be used to hide certain cells from a 2D layout. (Default value: None)
            size_vals: An array of values to be used to set sizes of each cell's datapoint in the layout.
                       By default, all cells are of same size determined by `point_size` parameter.
                       Has no effect if `do_shading` is True (Default value: None)
            clip_fraction: Same as `clip_fraction` parameter of 'get_cell_vals' method. This value is multiplied by 100
                           and the percentiles are soft-clipped from either end. (Default value: 0)
            width: Figure width (Default value: 6)
            height: Figure height (Default value: 6)
            default_color: A default color for the cells. (Default value: steelblue)
            cmap: A matplotlib colourmap to be used to colour categorical or continuous values plotted on the cells.
                  (Default value: tab20 for categorical variables and cmocean.deep for continuous variables)
            color_key: A custom colour map for cells. These can be used for categorical variables only. The keys in this
                       dictionary should be the category label as present in the `color_by` column and values should be
                       valid matplotlib colour names or hex codes of colours. (Default value: None)
            mask_values: These can a subset of categorical variables that are present in `color_by` which you would like
                         to mask away. These values would be combined under a same label (`mask_name`) and will be given
                         same colour (`mask_color`)
            mask_name: Label to replace the masked value labels. (Default value : None)
            mask_color: Color to be used for masked values. This should be a valid matplotlib named colour or a hexcode
                        of a colour. (Default value: 'k')
            point_size: Size of each scatter point. This is overridden if `size_vals` is provided. Has no effect if
                        `do_shading` is True. (Default value: 10)
            do_shading: Sets shading mode on/off. If shading mode is off (default) then matplotlib's scatter function is
                        is used otherwise a rasterized image is generated using datashader library. Turn this on if you
                        have more than 100K cells to improve render time and also to avoid issues with over-plotting.
                        (Default value: False)
            shade_npixels: Number of pixels to rasterize (for both height and width). This controls the resolution of
                           the figure. Adjust this according to the size of the image you want to generate.
                           (Default value: 1000)
            shade_min_alpha: The minimum alpha value to use for non-empty pixels when doing color-mapping, in [0, 255].
                             Use a higher value to avoid under-saturation, i.e. poorly visible low-value datapoints, at
                             the expense of the overall dynamic range. (Default value: 10)
            spread_pixels: Maximum number of pixels to spread on all sides (Default value: 1)
            spread_threshold:  When spreading, determines how far to spread. Spreading starts at 1 pixel, and stops
                               when the fraction of adjacent non-empty pixels reaches this threshold. Higher values
                               give more spreading, up to the `spread_pixels` allowed. (Default value: 0.2)
            ax_label_size: Font size for the x and y-axis labels. (Default value: 12)
            frame_offset: Extend the x and y-axis limits by this fraction (Default value: 0.05)
            spine_width: Line width of the displayed spines (Default value: 0.5)
            spine_color: Colour of the displayed spines.  (Default value: 'k')
            displayed_sides: Determines which figure spines are chosen. The spines to be shown can be supplied as a
                             tuple. The options are: top, bottom, left and right. (Default value: ('bottom', 'left) )
            legend_ondata: Whether to show category labels on the data (scatter points). The position of the label is
                           the centroid of the corresponding values. Has no effect if `color_by` has continuous values.
                           (Default value: True)
            legend_onside: Whether to draw a legend table on the side of the figure. (Default value: True)
            legend_size: Font size of the legend text. (Default value: 12)
            legends_per_col: Number of legends to be used on each legend column. This value determines how many
                             legend columns will be drawn (Default value: 20)
            title: Title to be used for plot/plots. If more than one plot are being plotted then the value should be a
                   list of strings. By default, the titles are automatically inferred from color_by parameter
                   (Default value: None)
            title_size: Size of each axis/subplots title (Default value: 12)
            hide_title: If True, then the title of the sublots is not shown (Default value: False)
            cbar_shrink: Shrinking factor for the width of color bar (Default value: 0.6)
            marker_scale: The relative size of legend markers compared with the originally drawn ones.
                          (Default value: 70)
            lspacing: The vertical space between the legend entries. Measured in font-size units. (Default value: 0.1)
            cspacing: The spacing between columns. Measured in font-size units. (Default value: 1)
            savename: Path where the rendered figure is to be saved. The format of the saved image depends on the
                      the extension present in the parameter value. (Default value: None)
            save_dpi: DPI when saving figure (Default value: 300)
            shuffle_df: Shuffle the order of cells in the plot (Default value: False)
            sort_values: Sort the values before plotting. Setting True will cause the datapoints with
                         (cells) with larger values to be plotted over the ones with lower values.
                         (Default value: False)
            ax: An instance of Matplotlib's Axes object. This can be used to plot the figure into an already
                created axes. It is ignored if `do_shading` is set to True. (Default value: None)
            force_ints_as_cats: Force integer labels in `color_by` as categories. If False, then integer will be
                                treated as continuous variables otherwise as categories. This effects how colormaps
                                are chosen and how legends are rendered. Set this to False if you are large number of
                                unique integer entries (Default: True)
            n_columns: If plotting several plots in a grid this argument decides the layout by how many columns in the
                       grid. Defaults to 4 but if the total number of plots is less than 4 it will default to that
                       number.
            w_pad: When plotting in multiple plots in a grid this decides the width padding between the plots.
                   If None is provided the padding will be automatically added to avoid overlap.
                   Ignored if only plotting one scatterplot.
            h_pad: When plotting in multiple plots in a grid this decides the height padding between the plots.
                   If None is provided the padding will be automatically added to avoid overlap.
                   Ignored if only plotting one scatterplot.
            show_fig: Whether to render the figure and display it using plt.show() (Default value: True)
            scatter_kwargs: Keyword argument to be passed to matplotlib's scatter command

        Returns:
            None
        """

        # TODO: add support for providing a list of subselections, from_assay and cell_keys
        # TODO: add support for different kinds of point markers

        from ..plots import plot_scatter, shade_scatter

        if from_assay is None:
            from_assay = self._defaultAssay
        if cell_key is None:
            cell_key = self._get_latest_cell_key(from_assay)
        if layout_key is None:
            raise ValueError("Please provide a value for `layout_key` parameter.")
        if clip_fraction >= 0.5:
            raise ValueError(
                "ERROR: clip_fraction cannot be larger than or equal to 0.5"
            )
        if isinstance(layout_key, str):
            layout_key: list[str] = [layout_key]
        # If a list of layout keys and color_by (e.g. layout_key=['UMAP', 'tSNE'], color_by=['gene1', 'gene2'] the
        # grid layout will be: plot1: UMAP + gene1, plot2: UMAP + gene2, plot3: tSNE + gene1, plot4: tSNE + gene2
        dfs = []
        for lk in layout_key:
            x = self.cells.fetch(f"{lk}1", key=cell_key)
            y = self.cells.fetch(f"{lk}2", key=cell_key)
            if color_by is None:
                color_by = "vc"
            if isinstance(color_by, str):
                color_by: list[str] = [color_by]
            for c in color_by:
                if c == "vc":
                    v = np.ones(len(x)).astype(int)
                else:
                    v = self.get_cell_vals(
                        from_assay=from_assay,
                        cell_key=cell_key,
                        k=c,
                        clip_fraction=clip_fraction,
                    )
                df = pd.DataFrame({f"{lk} 1": x, f"{lk} 2": y, c: v})
                if size_vals is not None:
                    if len(size_vals) != len(x):
                        raise ValueError(
                            "ERROR: `size_vals` is not of same size as layout_key"
                        )
                    df["s"] = size_vals
                if subselection_key is not None:
                    idx = self.cells.fetch(subselection_key, cell_key)
                    if idx.dtype != bool:
                        logger.warning(
                            f"`subselection_key` {subselection_key} is not bool type. Will not sub-select"
                        )
                    else:
                        df = df[idx]
                if shuffle_df:
                    df = df.sample(frac=1)
                if sort_values:
                    df = df.sort_values(by=c)
                dfs.append(df)

        if n_columns > len(dfs):
            n_columns = len(dfs)

        if do_shading:
            return shade_scatter(
                dfs,
                ax,
                width,
                shade_npixels,
                spread_pixels,
                spread_threshold,
                shade_min_alpha,
                cmap,
                color_key,
                mask_values,
                mask_name,
                mask_color,
                ax_label_size,
                frame_offset,
                spine_width,
                spine_color,
                displayed_sides,
                legend_ondata,
                legend_onside,
                legend_size,
                legends_per_col,
                title,
                title_size,
                hide_title,
                cbar_shrink,
                marker_scale,
                lspacing,
                cspacing,
                savename,
                save_dpi,
                force_ints_as_cats,
                n_columns,
                w_pad,
                h_pad,
                show_fig,
            )
        else:
            return plot_scatter(
                dfs,
                ax,
                width,
                height,
                default_color,
                cmap,
                color_key,
                mask_values,
                mask_name,
                mask_color,
                point_size,
                ax_label_size,
                frame_offset,
                spine_width,
                spine_color,
                displayed_sides,
                legend_ondata,
                legend_onside,
                legend_size,
                legends_per_col,
                title,
                title_size,
                hide_title,
                cbar_shrink,
                marker_scale,
                lspacing,
                cspacing,
                savename,
                save_dpi,
                force_ints_as_cats,
                n_columns,
                w_pad,
                h_pad,
                show_fig,
                scatter_kwargs,
            )

    def plot_cluster_tree(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        cluster_key: str | None = None,
        fill_by_value: str | None = None,
        force_ints_as_cats: bool = True,
        width: float = 1,
        lvr_factor: float = 0.5,
        vert_gap: float = 0.2,
        min_node_size: float = 10,
        node_size_multiplier: float = 1e4,
        node_power: float = 1.2,
        root_size: float = 100,
        non_leaf_size: float = 10,
        show_labels: bool = True,
        fontsize: float = 10,
        root_color: str = "#C0C0C0",
        non_leaf_color: str = "k",
        cmap="tab20",
        color_key: dict | None = None,
        edgecolors: str = "k",
        edgewidth: float = 1,
        alpha: float = 0.7,
        figsize=(5, 5),
        ax=None,
        show_fig: bool = True,
        savename: str | None = None,
        save_dpi: int = 300,
    ):
        """Plots a hierarchical layout of the clusters detected using
        `run_clustering` in a binary tree form. This helps evaluate the
        relationships between the clusters. This figure can complement
        embeddings likes tSNE where global distances are not preserved. The
        plot shows clusters as coloured nodes and the nodes are sized
        proportionally to the number of cells within the clusters. Root and
        branching nodes are shown to visually track the branching pattern of
        the tree. This figure is not scaled, i.e. the distances between the
        nodes are meaningless and only the branching pattern of the nodes must
        be evaluated.

        https://epidemicsonnetworks.readthedocs.io/en/latest/functions/EoN.hierarchy_pos.html

        Args:
            color_key: A custom colour map for cells. These can be used for categorical variables only. The keys in this
                       dictionary should be the category label as present in the `color_by` column and values should be
                       valid matplotlib colour names or hex codes of colours. (Default value: None)
            force_ints_as_cats: Force integer labels in `color_by` as categories. If False, then integer will be
                                treated as continuous variables otherwise as categories. This effects how colourmaps
                                are chosen and how legends are rendered. Set this to False if you are large number of
                                unique integer entries (Default: True)
            fill_by_value: ..
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: One of the columns from cell metadata table that indicates the cells to be used.
                      Should be same as the one that was used in one of the `run_clustering` calls for the given assay.
                      The values in the chosen column should be boolean (Default value: 'I')
            feat_key: Feature key. Should be same as the one that was used in `run_clustering` calls for the
                      given assay. By default, the latest used feature for the given assay will be used.
            cluster_key: Should be one of the columns from cell metadata table that contains the output of
                         `run_clustering` method. For example if chosen assay is `RNA` and default value for `label`
                         parameter was used in `run_clustering` then `cluster_key` can be 'RNA_cluster'
            width: Horizontal space allocated for the branches. Larger values may disrupt the hierarchical layout of
                   the cells (Default value: 1)
            lvr_factor: Leaf vs root factor. Controls the relative nodes horizontal spacing between as one moves up or
                        down the tree. Higher values will cause terminal nodes to be more spread out at cost of nodes
                        closer to the root and vice versa. (Default value: 0.5)
            vert_gap: Gap between levels of hierarchy (Default value: 0.2)
            min_node_size: Minimum size of a node (Default value: 10 )
            node_size_multiplier: Size of each leaf node is increased by this factor (Default value: 1e4)
            node_power: The number of cells within each cluster is raised to this value to scale up the node size.
                        (Default value: 1.2)
            root_size: Size of the root node (Default value: 100)
            non_leaf_size: Size of the nodes that represent branch points in the tree (Default value: 10)
            show_labels: Whether to show the cluster labels on the cluster nodes (Default value: True)
            fontsize: Font size of cluster labels. Only used when `do_label` is True (Default value: 10)
            root_color: Colour for root node. Acceptable values are  Matplotlib named colours or hexcodes for colours.
                        (Default value: '#C0C0C0')
            non_leaf_color: Colour for branch-point nodes. Acceptable values are  Matplotlib named colours or hexcodes
                            for colours. (Default value: 'k')
            cmap: A colormap to be used to colour cluster nodes. Should be one of Matplotlib colormaps.
                  (Default value: 'tab20')
            edgecolors: Edge colour of circles representing nodes in the hierarchical tree (Default value: 'k)
            edgewidth:  Line width of the edges circles representing nodes in the hierarchical tree  (Default value: 1)
            alpha: Alpha level (Opacity) of the displayed nodes in the figure. (Default value: 0.7)
            figsize: A tuple with describing figure width and height (Default value: (5, 5))
            ax: An instance of Matplotlib's Axes object. This can be used to plot the figure into an already
                created axes. (Default value: None)
            show_fig: If, False then axes object is returned rather than rendering the plot (Default value: True)
            savename: Path where the rendered figure is to be saved. The format of the saved image depends on
                      the extension present in the parameter value. (Default value: None)
            save_dpi: DPI when saving figure (Default value: 300)

        Returns:
            None
        """

        from networkx import DiGraph, to_pandas_edgelist

        from ..dendrogram import CoalesceTree, make_digraph
        from ..plots import plot_cluster_hierarchy

        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay, cell_key, feat_key
        )

        if cluster_key is None:
            raise ValueError(
                "ERROR: Please provide a value for `cluster_key` parameter"
            )
        clusts = self.cells.fetch(cluster_key, key=cell_key)
        graph_loc = self._get_latest_graph_loc(from_assay, cell_key, feat_key)
        dendrogram_loc = self.zw[graph_loc].attrs["latest_dendrogram"]
        n_clusts = len(set(clusts))
        coalesced_loc = dendrogram_loc + f"_coalesced_{n_clusts}"
        if coalesced_loc in self.zw:
            subgraph = DiGraph()
            subgraph.add_edges_from(self.zw[coalesced_loc + "/edgelist"][:])
            for i, j in zip(
                self.zw[coalesced_loc + "/nodelist"][:],
                self.zw[coalesced_loc + "/partition_id"][:],
            ):
                node = int(i[0])
                subgraph.nodes[node]["nleaves"] = int(i[1])
                if j != "-1":
                    subgraph.nodes[node]["partition_id"] = j
        else:
            subgraph = CoalesceTree(make_digraph(self.zw[dendrogram_loc][:]), clusts)
            edge_list = to_pandas_edgelist(subgraph).values
            store = create_zarr_dataset(
                self.zw, coalesced_loc + "/edgelist", (100000,), "u8", edge_list.shape
            )
            store[:] = edge_list
            node_list = []
            partition_id_list = []
            for i in subgraph.nodes():
                d = subgraph.nodes[i]
                p = d["partition_id"] if "partition_id" in d else -1
                node_list.append((i, d["nleaves"]))
                partition_id_list.append(str(p))

            node_list = np.array(node_list)
            store = create_zarr_dataset(
                self.zw,
                coalesced_loc + "/nodelist",
                (100000,),
                node_list.dtype,
                node_list.shape,
            )
            store[:] = node_list

            store = create_zarr_dataset(
                self.zw,
                coalesced_loc + "/partition_id",
                (100000,),
                str,
                (len(partition_id_list),),
            )
            store[:] = partition_id_list

        if fill_by_value is not None:
            color_values = self.get_cell_vals(
                from_assay=from_assay, cell_key=cell_key, k=fill_by_value
            )
        else:
            color_values = None
        plot_cluster_hierarchy(
            subgraph,
            clusts,
            color_values,
            force_ints_as_cats=force_ints_as_cats,
            width=width,
            lvr_factor=lvr_factor,
            vert_gap=vert_gap,
            min_node_size=min_node_size,
            node_size_multiplier=node_size_multiplier,
            node_power=node_power,
            root_size=root_size,
            non_leaf_size=non_leaf_size,
            show_labels=show_labels,
            fontsize=fontsize,
            root_color=root_color,
            non_leaf_color=non_leaf_color,
            cmap=cmap,
            color_key=color_key,
            edgecolors=edgecolors,
            edgewidth=edgewidth,
            alpha=alpha,
            figsize=figsize,
            ax=ax,
            show_fig=show_fig,
            savename=savename,
            save_dpi=save_dpi,
        )

    def plot_marker_heatmap(
        self,
        from_assay: str | None = None,
        group_key: str | None = None,
        cell_key: str | None = None,
        topn: int = 5,
        log_transform: bool = True,
        vmin: float = -1,
        vmax: float = 2,
        savename: str | None = None,
        save_dpi: int = 300,
        show_fig: bool = True,
        **heatmap_kwargs,
    ):
        """Displays a heatmap of top marker gene expression for the chosen
        groups (usually cell clusters).

        Z-scores are calculated for each marker gene before plotting them. The groups are subjected to hierarchical
        clustering to bring groups with similar expression pattern in proximity.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            group_key: Required parameter. This has to be a column name from cell metadata table. This column dictates
                       how the cells will be grouped. This value should be same as used for `run_marker_search`
            cell_key: One of the columns from cell metadata table that indicates the cells to be used.
                     Should be same as the one that was used in one of the `run_marker_search` calls for the given
                     assay. The values in the chosen column should be boolean (Default value: 'I')
            topn: Number of markers to be displayed for each group in `group_key` column. The markers are sorted based
                  on obtained scores by `run_marker_search`. (Default value: 5)
            log_transform: Whether to log-transform the values before displaying them in the heatmap.
                           (Default value: True)
            vmin: z-scores lower than this value are ceiled to this value. (Default value: -1)
            vmax: z-scores higher than this value are floored to this value. (Default value: 2)
            savename: Path where the rendered figure is to be saved. The format of the saved image depends on
                      the extension present in the parameter value. (Default value: None)
            save_dpi: DPI when saving figure. (Default value: 300)
            show_fig: Whether to render the figure and display it using plt.show() (Default value: True)
            **heatmap_kwargs: Keyword arguments to be forwarded to seaborn.clustermap.

        Returns:
            None
        """
        from ..plots import plot_heatmap

        assay = self._get_assay(from_assay)
        if group_key is None:
            raise ValueError("ERROR: Please provide a value for `group_key`")
        if cell_key is None:
            cell_key = "I"
        if "markers" not in self.zw[assay.name]:
            raise KeyError("ERROR: Please run `run_marker_search` first")
        slot_name = f"{cell_key}__{group_key}"
        if slot_name not in self.zw[assay.name]["markers"]:
            raise KeyError(
                f"ERROR: Please run `run_marker_search` first with {group_key} as `group_key` and "
                f"{cell_key} as `cell_key`"
            )
        g = self.zw[assay.name]["markers"][slot_name]
        feat_idx = []
        for i in g.keys():
            if "feature_index" in g[i]:
                feat_idx.extend(g[i]["feature_index"][:][:topn])
        if len(feat_idx) == 0:
            raise ValueError("ERROR: Marker list is empty for all the groups")
        feat_idx = np.array(sorted(set(feat_idx))).astype(int)
        cell_idx = np.array(assay.cells.active_index(cell_key))
        normed_data = assay.normed(
            cell_idx=cell_idx,
            feat_idx=feat_idx,
            log_transform=log_transform,
        )
        groups = assay.cells.fetch(group_key, cell_key)
        # Streaming per-group mean: accumulate per-block group sums and counts,
        # then divide. Groups are ordered like a pandas groupby (sorted labels).
        group_sums: dict = {}
        group_counts: dict = {}
        row_start = 0
        for block in tqdmbar(
            normed_data.blocks,
            total=normed_data.numblocks[0],
            desc="Aggregating marker values per group",
        ):
            a = controlled_compute(block, self.nthreads)
            block_groups = groups[row_start : row_start + a.shape[0]]
            row_start += a.shape[0]
            bdf = pd.DataFrame(a)
            bdf["__group__"] = block_groups
            grouped = bdf.groupby("__group__")
            block_sum = grouped.sum()
            block_count = grouped.size()
            for label in block_sum.index:
                vals = block_sum.loc[label].to_numpy(dtype=np.float64)
                if label not in group_sums:
                    group_sums[label] = vals
                    group_counts[label] = int(block_count.loc[label])
                else:
                    group_sums[label] += vals
                    group_counts[label] += int(block_count.loc[label])
        labels = sorted(group_sums.keys())
        df = pd.DataFrame(
            np.vstack([group_sums[label] / group_counts[label] for label in labels]),
            index=labels,
        )
        df = df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
        df.columns = assay.feats.fetch_all("names")[feat_idx]
        df = df.T
        # noinspection PyTypeChecker
        df[df < vmin] = vmin
        # noinspection PyTypeChecker
        df[df > vmax] = vmax
        plot_heatmap(
            df,
            savename=savename,
            save_dpi=save_dpi,
            show_fig=show_fig,
            **heatmap_kwargs,
        )

    def plot_pseudotime_heatmap(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        feature_cluster_key: str | None = None,
        pseudotime_key: str | None = None,
        show_features: list | None = None,
        width: int = 5,
        height: int = 10,
        vmin: float = -2.0,
        vmax: float = 2.0,
        heatmap_cmap: str | None = None,
        pseudotime_cmap: str | None = None,
        clusterbar_cmap: str | None = None,
        tick_fontsize: int = 10,
        axis_fontsize: int = 12,
        feature_label_fontsize: int = 12,
        savename: str | None = None,
        save_dpi: int = 300,
        show_fig: bool = True,
    ) -> None:
        """Plot heatmap for the matrix calculated by running
        `run_pseudotime_aggregation`. The heatmap shows the cell bins ordered
        as per pseudotime values and features ordered by clusters. The clusters
        themselves are ordered in a fashion such that features that have mean
        maximum expression in early pseudotime appear first and the feature
        cluster that has mean maxima in the later pseudotime appears last.

        CAUTION: This make take a long time to render and consume large amount of memory if your data has too many
                 features or if you create too many bins for cell ordering.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Required parameter. One of the columns from cell attribute table that indicates the cells to be
                      used. The values in the chosen column should be boolean. This value should be same as used for
                      `run_pseudotime_aggregation`. (Default value: The cell key used for latest graph created)
            feat_key: Required parameter. One of the columns from feature attribute table that indicates the cells to be
                      used. The values in the chosen column should be boolean. This value should be same as used for
                      `run_pseudotime_aggregation`. (Default value: The cell key used for latest graph created)
            feature_cluster_key: Required parameter. The name of column from feature attribute table that contains
                                 information about feature clusters.
            pseudotime_key: Required parameter. The name of the column from cell attribute table that contains the
                            pseudotime values. This should be same as the one used from the relevant run of
                            `run_pseudotime_aggregation`.
            show_features: A list of feature names to be highlighted/labelled on the heatmap.
            width: Width of the heatmap (Default value: 5)
            height: Height of the heatmap (Default value: 10)
            vmin: The minimum value to be displayed on the heatmap. The values lower than this will ceiled to this
                  value. (Default value: -2.0)
            vmax: The maximum value to be displayed on the heatmap. The values higher than this will floored to this
                  value. (Default value: 2.0)
            heatmap_cmap: Colormap for the heatmap (Default value: coolwarm)
            pseudotime_cmap: Colormap for the pseudotime bar. It should be some kind of continuous colormap.
                             (Default value: cmocean.deep)
            clusterbar_cmap: Colormap for the cluster bar showing the span of each feature cluster.
                             (Default value: tab20)
            tick_fontsize: Font size for cbar ticks (Default value: 10)
            axis_fontsize: Font size for labels along each axis(Default value: 12)
            feature_label_fontsize: Font size for feature labels on the heatmap (Default value: 12)
            savename: Path where the rendered figure is to be saved. The format of the saved image depends on
                      the extension present in the parameter value. (Default value: None)
            save_dpi: DPI when saving figure (Default value: 300)
            show_fig: If, False then axes object is returned rather than rendering the plot (Default value: True)

        Returns: None
        """

        from ..plots import plot_annotated_heatmap

        assay = self._get_assay(from_assay)
        for i in [cell_key, feat_key, feature_cluster_key, feature_cluster_key]:
            if i is None:
                var_name = list(dict(i=i).keys())[0]  # Trick to get variables own name
                raise ValueError(
                    f"ERROR: Please provide a value for parameter `{var_name}`"
                )

        cell_ordering = assay.cells.fetch(pseudotime_key, key=cell_key)
        # noinspection PyProtectedMember
        cell_idx, feat_idx = assay._get_cell_feat_idx(cell_key, feat_key)
        hashes = [hash(tuple(x)) for x in (cell_idx, feat_idx, cell_ordering)]
        location = f"aggregated_{cell_key}_{feat_key}_{pseudotime_key}"
        if location not in assay.z:
            raise KeyError(
                f"ERROR: Could not find aggregated feature values at location '{location}' "
                f"Please make sure that you have run `run_pseudotime_aggregation` with the same values for "
                f"parameters: `cell_key`, `feat_key` and `pseudotime_key`"
            )
        else:
            if hashes != assay.z[location].attrs["hashes"]:
                raise ValueError(
                    "ERROR: The values under one or more of these columns: `cell_key`, `feat_key` or/and "
                    "`pseudotime_key have been updated after running `run_pseudotime_aggregation`"
                )

        da = ChunkedArray(assay.z[location + "/data"], nthreads=self.nthreads)
        feature_indices = assay.z[location + "/feature_indices"][:]
        da = da[: feature_indices.shape[0]]

        feature_clusters = assay.feats.fetch_all(feature_cluster_key)[feature_indices]
        feature_labels = assay.feats.fetch_all("names")[feature_indices]

        idx = np.argsort(feature_clusters)
        feature_clusters = feature_clusters[idx]
        feature_labels = feature_labels[idx]
        da = da.compute()[idx]

        ordering = assay.cells.fetch(pseudotime_key, key=cell_key)

        plot_annotated_heatmap(
            df=da,
            xbar_values=ordering,
            ybar_values=feature_clusters,
            display_row_labels=show_features,
            row_labels=feature_labels,
            width=width,
            height=height,
            vmin=vmin,
            vmax=vmax,
            heatmap_cmap=heatmap_cmap,
            xbar_cmap=pseudotime_cmap,
            ybar_cmap=clusterbar_cmap,
            tick_fontsize=tick_fontsize,
            axis_fontsize=axis_fontsize,
            row_label_fontsize=feature_label_fontsize,
            savename=savename,
            save_dpi=save_dpi,
            show_fig=show_fig,
        )

    def metric_lisi(
        self,
        label_colnames: Iterable[str],
        use_latest_knn: bool = True,
        from_assay: str | None = None,
        knn_loc: str | None = None,
        save_result: bool = False,
        return_lisi: bool = True,
    ) -> list[tuple[str, np.ndarray]] | None:
        """Calculate Local Inverse Simpson Index (LISI) scores for cell populations.

        LISI measures how well mixed different cell populations are in the local neighborhood
        of each cell. Higher scores indicate better mixing of different populations.

        Args:
            label_colnames: Column names from cell metadata containing population labels
            use_latest_knn: Whether to use the most recent KNN graph (default: True)
            from_assay: Name of assay to use if not using latest KNN
            knn_loc: Location of KNN graph if not using latest (default: None)
            save_result: Whether to save LISI scores to cell metadata (default: True)
            return_lisi: Whether to return LISI scores (default: False)

        Returns:
            If return_lisi is True, returns list of tuples containing:
            - Label column name
            - numpy array of LISI scores for that label
            If return_lisi is False, returns None

        Raises:
            ValueError: If using custom KNN graph but required parameters are missing
            KeyError: If label columns not found in cell metadata

        Notes:
            LISI scores are computed for each label column separately.
            Scores near 1 indicate cells grouped with similar labels.
            Higher scores indicate more mixing between different labels.
        """

        if from_assay is None:
            from_assay = self._load_default_assay()

        if use_latest_knn and knn_loc is None:
            knn_loc = self._get_latest_knn_loc(from_assay)
            cell_key = self._get_latest_cell_key(from_assay)
            logger.info(f"Using the latest knn graph at location: {knn_loc}")

        else:
            if knn_loc is None:
                raise ValueError("Please provide values for the KNN graph location.")
            if knn_loc not in self.zw:
                raise ValueError(f"Could not find the knn graph at location: {knn_loc}")

            normed_part = knn_loc.split("/")[1]
            _, cell_key, _ = normed_part.split("__")
            logger.info(f"Using the knn graph at location: {knn_loc}")

        knn = self.zw[knn_loc]

        distances = knn["distances"]
        indices = knn["indices"]

        try:
            metadata = self.cells.to_pandas_dataframe(
                columns=label_colnames + [cell_key]
            )
            metadata = metadata[metadata[cell_key]]
        except KeyError:
            raise KeyError(
                f"Could not find the column(s) {label_colnames} in the cell metadata table."
            )

        from ..metrics import compute_lisi

        lisi_scores = compute_lisi(distances, indices, metadata, label_colnames)
        # lisi_scores Shape -> (n_cells, n_labels)
        if save_result:
            for col, vals in zip(label_colnames, lisi_scores.T):
                col_name = f"lisi__{col}__{knn_loc.split('/')[-1]}"
                self.cells.insert(
                    column_name=col_name, values=vals, overwrite=True, key=cell_key
                )

        if return_lisi:
            return list(zip(label_colnames, lisi_scores.T))
        else:
            return None

    def metric_silhouette(
        self,
        use_latest_knn: bool = True,
        res_label: str = "leiden_cluster",
        from_assay: str | None = None,
        knn_loc: str | None = None,
    ) -> np.ndarray | None:
        """Calculate modified silhouette scores for evaluating cluster separation.

        This implements a graph-based silhouette score that measures how similar cells
        are to their own cluster compared to the nearest neighboring cluster.

        Args:
            use_latest_knn: Whether to use most recent KNN graph (default: True)
            res_label: Column name containing cluster labels (default: "leiden_cluster")
            from_assay: Name of assay to use if not using latest KNN (default: None)
            knn_loc: Location of KNN graph if not using latest (default: None)

        Returns:
            numpy array of silhouette scores for each cluster, or None if computation fails

        Raises:
            ValueError: If using custom KNN graph but required parameters are missing

        Notes:
            Scores range from -1 to 1:
            - Near 1: Cluster is well-separated from neighboring clusters
            - Near 0: Cluster overlaps with neighboring clusters
            - Near -1: Cluster may be incorrectly assigned

            Implementation uses sampling for efficiency with large datasets.
            NaN values indicate clusters that couldn't be scored due to size constraints.
        """

        def compute_graph_feats(knn_loc: str):
            k = knn_loc.rsplit("/", 1)[-1].split("__")[-1]
            dims = knn_loc.rsplit("/", 2)[0].split("__")[-2]
            feat_key = knn_loc.split("/")[1].split("__")[-1]
            return k, dims, feat_key

        if from_assay is None:
            from_assay = self._load_default_assay()

        if use_latest_knn and knn_loc is None:
            knn_loc = self._get_latest_knn_loc(from_assay)
            k, dims, feat_key = compute_graph_feats(knn_loc)
            logger.info(
                f"Using the latest knn graph at location: {knn_loc} for assay: {from_assay}"
            )

        else:
            if knn_loc is None:
                raise ValueError("Please provide values for the KNN graph location.")
            if knn_loc not in self.zw:
                raise ValueError(f"Could not find the knn graph at location: {knn_loc}")
            k, dims, feat_key, from_assay = compute_graph_feats(knn_loc)
            logger.info(f"Using the knn graph at location: {knn_loc}")

        from ..metrics import knn_to_csr_matrix, silhouette_scoring

        ann_loc = knn_loc.rsplit("/", 1)[0]
        isHarmonized = self.zw[ann_loc].attrs["isHarmonized"]

        normed_part = knn_loc.split("/")[1]
        _, cell_key, feat_key_parsed = normed_part.split("__")
        ann_obj = self._load_ann_stream(
            from_assay=from_assay,
            cell_key=cell_key,
            feat_key=feat_key_parsed,
            knn_loc=knn_loc,
        )

        knn_group = self.z[knn_loc]
        graph = knn_to_csr_matrix(knn_group["indices"], knn_group["distances"])
        hvg_data = self.z[knn_loc.rsplit("/", 3)[0] + "/data"]
        scores = silhouette_scoring(
            self, ann_obj, graph, hvg_data, from_assay, res_label
        )
        return scores

    def metric_integration(
        self, batch_labels: list[str], metric: Literal["ari", "nmi"] = "ari"
    ) -> float | None:
        """Calculate integration score between different batch labels.

        Measures how well aligned different batches are after integration by comparing
        their cluster assignments using standard clustering metrics.

        Args:
            batch_labels: List of column names containing batch labels to compare
            metric: Metric to use for comparison (default: "ari")
                - "ari": Adjusted Rand Index
                - "nmi": Normalized Mutual Information

        Returns:
            Integration score between 0 and 1, or None if metric is not recognized
            Higher scores indicate better alignment between batches

        Notes:
            ARI and NMI measure the agreement between different batch labelings:
            - Score near 1: Batches are well integrated
            - Score near 0: Batches show poor integration

            ARI is adjusted for chance and generally more stringent than NMI.
        """
        from ..metrics import integration_score

        batch_labels_vals = []
        for batch in batch_labels:
            vals = np.array(self.cells.fetch_all(batch))
            batch_labels_vals.append(vals)

        scores = integration_score(batch_labels_vals, metric)
        return scores

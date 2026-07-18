import time
from collections.abc import Iterable, Sequence
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import zarr
from loguru import logger
from numpy.typing import NDArray

from .._types import ZarrMode, as_zarr_array, as_zarr_group
from ..assay import Assay, ATACassay, RNAassay
from ..feat_utils import hto_demux
from ..markers import resolve_marker_gene_batch_size, sort_marker_results
from ..results import PseudotimeAggregationResult, PseudotimeMarkerResult
from ..utils import ZARRLOC, array_digest, controlled_compute, tqdmbar
from ..writers import create_zarr_dataset
from .mapping_datastore import MappingDatastore

__all__ = ["DataStore"]

_MARKER_LAYOUT_V2 = "compact_v2"
_MARKER_STAT_COLUMNS = (
    "score",
    "mean",
    "mean_rest",
    "frac_exp",
    "frac_exp_rest",
    "fold_change",
    "p_value",
)
_MARKER_OUT_COLUMNS = ("feature_index", *_MARKER_STAT_COLUMNS)


def _feature_column_chunk(assay: Assay, n_features: int) -> int:
    # RNA feature-column streams (markers, HVG, pseudotime) prefer countsT
    # when present; other assays keep cell-major batch sizing.
    if isinstance(assay, RNAassay):
        counts_t = getattr(assay, "rawDataT", None)
        if counts_t is not None:
            chunks = getattr(counts_t, "chunks", None)
            if chunks and len(chunks) > 0:
                return max(1, int(chunks[0]))
    backing = getattr(assay.rawData, "_backing", None)
    chunks = getattr(backing, "chunks", None)
    if chunks and len(chunks) > 1:
        return max(1, int(chunks[1]))
    return max(1, int(n_features))


def _shared_marker_feature_index(markers: dict[Any, pd.DataFrame]) -> np.ndarray:
    for vals in markers.values():
        if len(vals) != 0:
            return np.sort(np.asarray(vals.index.values, dtype=np.int32))
    raise ValueError("Cannot save empty marker results")


def _marker_stats_matrix(vals: pd.DataFrame, feature_index: np.ndarray) -> np.ndarray:
    aligned = vals.reindex(feature_index)
    return np.asarray(
        aligned.loc[:, list(_MARKER_STAT_COLUMNS)].to_numpy(dtype=np.float64)
    )


def _write_compact_marker_stats(
    cluster_group: zarr.Group,
    stats: np.ndarray,
) -> None:
    from ..storage.zarr_store import ZarrArraySpec, create_numeric_array

    n_features = int(stats.shape[0])
    spec = ZarrArraySpec(
        shape=(n_features, len(_MARKER_STAT_COLUMNS)),
        chunks=(n_features, len(_MARKER_STAT_COLUMNS)),
        dtype="float64",
        overwrite=True,
    )
    arr = create_numeric_array(cluster_group, "stats", spec)
    arr[:] = stats


def _load_marker_cluster_frame(
    slot_group: zarr.Group,
    cluster_group: zarr.Group,
    feature_names: np.ndarray,
    *,
    group_id: Any,
) -> pd.DataFrame:
    out_cols = list(_MARKER_OUT_COLUMNS)
    if slot_group.attrs.get("layout") == _MARKER_LAYOUT_V2 and "stats" in cluster_group:
        feature_index = np.asarray(
            as_zarr_array(slot_group["feature_index"], name="feature_index")[:]
        )
        stats = np.asarray(as_zarr_array(cluster_group["stats"], name="stats")[:])
        df = pd.DataFrame(stats, columns=list(_MARKER_STAT_COLUMNS))
        df["feature_index"] = feature_index
        df["feature_name"] = feature_names[feature_index.astype(int)]
        df["group_id"] = group_id
        return sort_marker_results(df[["group_id", "feature_name", *out_cols[1:]]])

    available_cols = [col for col in out_cols if col in cluster_group]
    if not available_cols:
        return pd.DataFrame([[] for _ in out_cols], index=out_cols).T
    cols = [
        np.asarray(as_zarr_array(cluster_group[x], name=x)[:]) for x in available_cols
    ]
    df = pd.DataFrame(cols, index=available_cols).T
    df["group_id"] = group_id
    df["feature_name"] = feature_names[df.feature_index.astype("int")]
    return df[["group_id", "feature_name", *available_cols[1:]]]


def _validated_pseudotime_regressor(
    assay: Assay,
    cell_key: str,
    pseudotime_key: str,
) -> np.ndarray:
    try:
        pseudotime = np.asarray(
            assay.cells.fetch(pseudotime_key, key=cell_key),
            dtype=float,
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Pseudotime column '{pseudotime_key}' must be numeric"
        ) from exc

    if pseudotime.ndim != 1:
        raise ValueError(
            f"Pseudotime column '{pseudotime_key}' must be one-dimensional"
        )
    expected_size = assay.cells.active_index(cell_key).shape[0]
    if pseudotime.shape[0] != expected_size:
        raise ValueError(
            f"Pseudotime column '{pseudotime_key}' has {pseudotime.shape[0]} values, "
            f"but cell_key '{cell_key}' selects {expected_size} cells"
        )
    if not np.isfinite(pseudotime).all():
        validity_key = f"{pseudotime_key}__valid"
        if validity_key in assay.cells.columns:
            raise ValueError(
                f"Pseudotime column '{pseudotime_key}' contains unscored cells. "
                f"Use cell_key='{validity_key}' for downstream analysis"
            )
        raise ValueError(
            f"Pseudotime column '{pseudotime_key}' contains non-finite values"
        )
    if pseudotime.size < 2 or np.unique(pseudotime).size < 2:
        raise ValueError(
            f"Pseudotime column '{pseudotime_key}' must contain at least two distinct values"
        )
    return pseudotime


def _group_assignment_digest(values: np.ndarray) -> str:
    return array_digest(np.asarray(values).astype(str))


def _scatter_feature_clusters(
    n_features: int,
    feature_indices: np.ndarray,
    clusters: np.ndarray,
    unassigned_value: int,
) -> np.ndarray:
    feature_indices = np.asarray(feature_indices, dtype=int)
    clusters = np.asarray(clusters, dtype=int)
    if feature_indices.shape != clusters.shape:
        raise ValueError("Feature indices and cluster assignments are misaligned")
    if unassigned_value in clusters:
        raise ValueError("unassigned_value conflicts with an assigned feature cluster")
    values = np.full(n_features, unassigned_value, dtype=int)
    values[feature_indices] = clusters
    return values


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
        mem_budget: Memory budget bounding streaming and concurrency. Accepts bytes, a suffixed size
                    (e.g. '8G'), or a fraction of total system memory (e.g. '0.6'). When None, it is
                    auto-detected (SCARF_MEM_BUDGET env var, else total system memory). Override it to
                    simulate reading on a machine with a different memory size than the writer.
        working_copies: Number of concurrent in-memory working copies the memory budget is divided
                        across. When None, uses SCARF_WORKING_COPIES env var or the default.
    """

    def __init__(
        self,
        zarr_loc: ZARRLOC,
        assay_types: dict[str, str] | None = None,
        default_assay: str | None = None,
        min_features_per_cell: int = 10,
        min_cells_per_feature: int = 20,
        mito_pattern: str | None = None,
        ribo_pattern: str | None = None,
        nthreads: int = 2,
        zarr_mode: ZarrMode = "r+",
        workspace: str | None = None,
        synchronizer: Any = None,
        zarrProfile: Literal["fast_local", "cloud"] | None = None,
        storage_options: dict[str, Any] | None = None,
        mem_budget: int | str | None = None,
        working_copies: int | None = None,
    ) -> None:
        from ..storage.budget import resolve_budget, set_resource_budget
        from ..storage.zarr_store import (
            configure_zarr_io_for_profile,
            set_storage_profile,
        )

        set_storage_profile(zarrProfile)
        set_resource_budget(
            resolve_budget(
                memory=mem_budget, workers=nthreads, working_copies=working_copies
            )
        )
        configure_zarr_io_for_profile()
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
            return cast(Assay, getattr(self, assay_name))

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

        from ..plotting import distribution

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
            dist = norm(np.median(a), np.std(a))
            self.filter_cells(
                attrs=[i], lows=[dist.ppf(min_p)], highs=[dist.ppf(max_p)]
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

        from ..doublet_utils import (
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
            target_ds = DataStore(
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
        max_cells: float | None = None,
        **plot_kwargs: Any,
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
            plot_kwargs: Named parameters forwarded to ``plotting.highly_variable_features``
                         (``figsize``, ``label_size``, ``point_sizes``, ``colormaps``).

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
        if max_cells is None or max_cells == np.inf:
            max_cells_int: int | float = np.inf
        else:
            max_cells_int = int(max_cells)
        assay.mark_hvgs(
            cell_key=cell_key,
            min_cells=min_cells,
            max_cells=max_cells_int,
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
        gene_batch_size: int | None = None,
        n_threads: int | None = None,
        skip_save: bool = False,
        **norm_params: Any,
    ) -> dict[str, Any] | None:
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
                             When None (default), the batch size is the minimum of the on-disk feature chunk
                             width and a budget-safe cap derived from the active memory budget.
            n_threads: Threads for marker search.
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

        n_features = len(assay.feats.active_index(feat_key))
        if gene_batch_size is None:
            gene_batch_size = resolve_marker_gene_batch_size(
                n_features=n_features,
                n_cells=len(assay.cells.active_index(cell_key)),
                column_chunk=_feature_column_chunk(assay, n_features),
            )

        slot_name = f"{cell_key}__{group_key}"
        logger.debug(
            f"Running marker search for {from_assay}/{slot_name} "
            f"(feat_key={feat_key}, batch_size={gene_batch_size})"
        )
        assay_grp = as_zarr_group(self.zw[assay.name], name=assay.name)
        if "markers" not in assay_grp:
            assay_grp.create_group("markers")
        markers_grp = as_zarr_group(assay_grp["markers"], name="markers")

        markers = find_markers_by_rank(
            assay=assay,
            group_key=group_key,
            cell_key=cell_key,
            feat_key=feat_key,
            batch_size=gene_batch_size,
            n_threads=n_threads,
            **norm_params,
        )

        if skip_save:
            return markers

        from ..storage.zarr_store import is_remote_datastore

        remote = is_remote_datastore(self.zarr_loc, self.z)
        t_save = time.perf_counter()
        remote_slot = markers_grp.create_group(slot_name, overwrite=True)
        workers = max(1, int(n_threads or self.nthreads))
        self._write_marker_slot(
            remote_slot,
            markers,
            workers=workers if remote else 1,
        )
        logger.info(
            f"Saved marker results to {assay.name}/markers/{slot_name} "
            f"in {time.perf_counter() - t_save:.1f}s "
            f"({len(markers)} clusters, layout={_MARKER_LAYOUT_V2})"
        )
        return None

    @staticmethod
    def _write_marker_slot(
        group: zarr.Group,
        markers: dict[Any, pd.DataFrame],
        *,
        workers: int = 1,
    ) -> None:
        from ..storage.zarr_store import create_metadata_column

        feature_index = _shared_marker_feature_index(markers)
        group.attrs["layout"] = _MARKER_LAYOUT_V2
        group.attrs["statColumns"] = list(_MARKER_STAT_COLUMNS)
        create_metadata_column(
            group,
            "feature_index",
            data=feature_index,
            dtype=np.int32,
            overwrite=True,
        )

        def write_cluster(item: tuple[Any, pd.DataFrame]) -> None:
            cluster_id, vals = item
            if len(vals) == 0:
                return
            cluster_group = group.create_group(str(cluster_id))
            stats = _marker_stats_matrix(vals, feature_index)
            _write_compact_marker_stats(
                cluster_group,
                stats,
            )

        items = list(markers.items())
        if workers <= 1:
            for item in items:
                write_cluster(item)
        else:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(write_cluster, items))

    def run_pseudotime_marker_search(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        pseudotime_key: str | None = None,
        min_cells: int = 10,
        gene_batch_size: int = 50,
        **norm_params: Any,
    ) -> PseudotimeMarkerResult:
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

        Returns:
            Correlation table and the feature-metadata keys where it was saved.
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
        ptime = _validated_pseudotime_regressor(assay, cell_key, pseudotime_key)
        n_cells = len(assay.cells.active_index(cell_key))
        n_feats = len(assay.feats.active_index(feat_key))
        logger.info(
            f"Pseudotime markers: correlating features "
            f"(cells={n_cells}, features={n_feats}, batch_size={gene_batch_size})"
        )
        markers = find_markers_by_regression(
            assay=assay,
            cell_key=cell_key,
            feat_key=feat_key,
            regressor=ptime,
            min_cells=min_cells,
            batch_size=gene_batch_size,
            **norm_params,
        )
        feature_index = assay.feats.active_index(feat_key)
        markers = markers.reindex(feature_index)
        if markers.isna().any(axis=None):
            raise ValueError("Pseudotime marker results are not aligned to feat_key")
        logger.info("Pseudotime markers: saving marker scores")
        correlation_key = f"{cell_key}__{pseudotime_key}__r"
        p_value_key = f"{cell_key}__{pseudotime_key}__p"
        assay.feats.insert(
            correlation_key,
            np.array(markers["r_value"].values),
            key=feat_key,
            overwrite=True,
        )
        assay.feats.insert(
            p_value_key,
            np.array(markers["p_value"].values),
            key=feat_key,
            overwrite=True,
        )
        table = markers.rename_axis("feature_index").reset_index()
        feature_names = np.asarray(assay.feats.fetch_all("names"), dtype=object)
        table.insert(
            1,
            "feature_name",
            feature_names[table["feature_index"].to_numpy(dtype=np.int64)],
        )
        return PseudotimeMarkerResult(
            table=table,
            correlation_key=correlation_key,
            p_value_key=p_value_key,
            assay=assay.name,
            cell_key=cell_key,
            feature_key=feat_key,
            pseudotime_key=pseudotime_key,
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
        nan_cluster_value: int = -1,
        **norm_params: Any,
    ) -> PseudotimeAggregationResult:
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
                      metadata/attribute table. (Default value: 'I')
            pseudotime_key: Required parameter. This has to be a column name from cell attribute table. This
                            column contains values for pseudotime ordering of the cells.
            cluster_label: Required parameter. Name of the column under which the feature cluster identity will be
                           saved in the feature attribute table.
            min_exp: Features with mean normalized expression below this value are dropped and not assigned
                     a cluster identity. (Default value: 1e-3)
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
            **norm_params: Extra keyword arguments forwarded to normalized expression calculation.

        Returns:
            Lazy aggregated matrix with its aligned feature indices and clusters.
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
        if not isinstance(nan_cluster_value, (int, np.integer)) or isinstance(
            nan_cluster_value, (bool, np.bool_)
        ):
            raise TypeError("nan_cluster_value must be an integer")
        nan_cluster_value = int(nan_cluster_value)
        _validated_pseudotime_regressor(assay, cell_key, pseudotime_key)

        logger.info("Pseudotime modules: aggregating feature profiles")
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
            **norm_params,
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
        temp = _scatter_feature_clusters(
            assay.feats.N,
            feat_ids,
            clusts,
            nan_cluster_value,
        )
        logger.info("Pseudotime modules: saving module labels")
        assay.feats.insert(
            cluster_label,
            temp,
            fill_value=nan_cluster_value,
            overwrite=True,
        )

        location = f"aggregated_{cell_key}_{feat_key}_{pseudotime_key}"
        aggregation_group = as_zarr_group(assay.z[location], name=location)
        cluster_digest = _group_assignment_digest(temp)
        aggregation_group.attrs["cluster_label"] = cluster_label
        aggregation_group.attrs["cluster_digest"] = cluster_digest
        aggregation_group.attrs["nan_cluster_value"] = nan_cluster_value

        for assay_name in self.assay_names:
            grouped_assay = self._get_assay(assay_name)
            if (
                grouped_assay.attrs.get("grouped_from_assay") == assay.name
                and grouped_assay.attrs.get("grouped_group_key") == cluster_label
                and grouped_assay.attrs.get("grouped_group_digest") != cluster_digest
            ):
                logger.warning(
                    f"Grouped assay '{assay_name}' is stale after updating "
                    f"feature groups in '{cluster_label}'. Rerun add_grouped_assay"
                )
        return PseudotimeAggregationResult(
            data=df,
            feature_indices=np.asarray(feat_ids),
            feature_clusters=np.asarray(clusts),
            cluster_key=cluster_label,
            storage_path=str(aggregation_group.path),
            assay=assay.name,
            cell_key=cell_key,
            feature_key=feat_key,
            pseudotime_key=pseudotime_key,
        )

    def get_markers(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        group_key: str | None = None,
        group_id: str | int | None = None,
        min_score: float = 0.25,
        min_frac_exp: float = 0.2,
    ) -> pd.DataFrame:
        """Return marker features from `run_marker_search`.

        When ``group_id`` is ``None`` (default), markers for every group under
        ``group_key`` are returned in one long table with a ``group_id`` column.
        Pass a specific ``group_id`` to return markers for that group only.
        For a wide export of marker names only, use ``export_markers_to_csv``.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: To run the test on specific subset of cells, provide the name of a boolean column in
                        the cell metadata table.
            group_key: Required parameter. This has to be a column name from cell metadata table.
                       Usually this would be a column denoting cell clusters. Please use the same value as used
                       when ran `run_marker_search`
            group_id: One value from the ``group_key`` column, or ``None`` for all groups.
            min_score: This value dictates how specific the feature value has to be in a group before it is
                       considered a marker for that group. The value has to be greater than 0 but less than or equal to
                       1 (Default value: 0.25)
            min_frac_exp: Minimum fraction of cells in a group that must have a non-zero value for a gene to be
                          considered a marker for that group.

        Returns:
            Pandas dataframe with marker statistics. All-group results include a ``group_id`` column.
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
            markers_grp = as_zarr_group(assay.z["markers"], name="markers")
            g = as_zarr_group(
                markers_grp[f"{cell_key}__{group_key}"],
                name=f"{cell_key}__{group_key}",
            )
        except KeyError:
            raise KeyError(
                "ERROR: Couldn't find the location of markers. Please make sure that you have already called "
                "`run_marker_search` method with same value of `cell_key` and `group_key`"
            )
        out_cols = list(_MARKER_OUT_COLUMNS)
        gids = sorted(set(assay.cells.fetch(group_key, key=cell_key)))
        if group_id is not None:
            gids = [group_id]

        feature_names = assay.feats.fetch_all("names")
        dfs = []
        for gid in gids:
            group_name = str(gid)
            if group_name in g:
                marker_grp = as_zarr_group(g[group_name], name=group_name)
                df = _load_marker_cluster_frame(
                    g,
                    marker_grp,
                    feature_names,
                    group_id=gid,
                )
            else:
                logger.debug(f"No markers found for {gid} returning empty dataframe")
                df = pd.DataFrame([[] for _ in out_cols], index=out_cols).T
                df["group_id"] = []
                df["feature_name"] = []
                df = df[["group_id", "feature_name"] + list(out_cols[1:])]
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

        from ..storage.zarr_store import write_dense_in_shard_rows

        from ..writers import create_zarr_count_assay, finalize_writer_counts

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
        n_groups = len(group_set)
        matrix = np.zeros((assay.cells.N, n_groups), dtype=np.float64)
        for n, i in tqdmbar(
            enumerate(group_set), desc="Computing grouped means", total=len(group_set)
        ):
            feat_idx = np.where(groups == i)[0]
            matrix[:, n] = (
                assay.normed(cell_idx=cell_idx, feat_idx=feat_idx)
                .mean(axis=1)
                .compute()
            )
        write_dense_in_shard_rows(
            g,
            lambda start, end: matrix[start:end, :],
            msg="Writing grouped assay",
        )
        finalize_writer_counts(self.zw, assay_label, self.workspace)

        self._load_assays(min_cells=0, custom_assay_types={assay_label: "Assay"})
        self._ini_cell_props(min_features=0, mito_pattern="", ribo_pattern="")
        grouped_assay = self._get_assay(assay_label)
        grouped_assay.attrs["grouped_from_assay"] = assay.name
        grouped_assay.attrs["grouped_group_key"] = group_key
        grouped_assay.attrs["grouped_group_digest"] = _group_assignment_digest(groups)

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
        Features from the BED file are retained even when they do not overlap any peak; those zero-count features
        are marked invalid during assay initialization.

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
        coords_ser = pd.Series(peaks_coords, dtype="object")
        string_mask = coords_ser.map(lambda x: isinstance(x, str))
        colon_counts = coords_ser.str.count(":")
        hyphen_counts = coords_ser.str.split(":").str[-1].str.count("-")
        invalid_mask = (
            ~string_mask
            | colon_counts.ne(1).fillna(True)
            | hyphen_counts.ne(1).fillna(True)
        )
        invalid_coords = invalid_mask.to_numpy(dtype=bool)
        if invalid_coords.any():
            n = int(np.flatnonzero(invalid_coords)[0])
            raise ValueError(
                f"ERROR: Coordinate format check failed for element: {peaks_coords[n]} (position {n}). "
                f"The format should be chr:start-end. Please note the colon and hyphen position"
            )

        coordinate_melding(
            assay,
            workspace=self.workspace,
            feature_bed=feature_bed,
            new_assay_name=assay_label,
            peaks_col=peaks_col,
            scalar_coeff=scalar_coeff,
            renormalization=renormalization,
            peaks_coords=peaks_coords,
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
        null_vals: list[Any] | None = None,
        secondary_null_vals: list[Any] | None = None,
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

        def make_reps(v: NDArray[Any], n_reps: int, seed: int) -> list[NDArray[Any]]:
            v_list = list(v)
            random_state = np.random.RandomState(seed)
            shuffled_idx = random_state.choice(v_list, len(v_list), replace=False)
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
            active_idx = self.cells.active_index(cell_key)
            groups_set = sorted(set(groups[active_idx]))
        if secondary_group_key is None:
            sec_groups: NDArray[Any] = np.array([None], dtype=object)
            sec_groups_set: list[Any] = [None]
        else:
            sec_groups = self.cells.fetch_all(secondary_group_key)
            sec_groups_set = sorted(set(sec_groups[active_idx]))

        assay = self._get_assay(from_assay)

        vals: dict[str, NDArray[Any]] = {}
        fracs: dict[str, NDArray[Any]] = {}
        all_feat_idx = np.arange(assay.feats.N)
        active_mask = np.zeros(self.cells.N, dtype=bool)
        active_mask[active_idx] = True
        for g in tqdmbar(groups_set):
            if g in null_vals:
                continue
            for sg in sec_groups_set:  # type: ignore
                if sg in secondary_null_vals:
                    continue
                if sg is None and len(sec_groups) == 1:
                    g_idx = np.where((groups == g) & active_mask)[0]
                else:
                    g_idx = np.where((groups == g) & (sec_groups == sg) & active_mask)[
                        0
                    ]
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

        vals_df = pd.DataFrame(vals).fillna(0)

        empty_idx = None
        if remove_empty_features:
            empty_idx = vals_df.sum(axis=1) != 0
            vals_df = vals_df.loc[empty_idx]

        if feature_label == "id":
            vals_df.set_index(
                pd.Series(assay.feats.fetch_all("ids")).reindex(vals_df.index).values,
                inplace=True,
                drop=True,
            )
        elif feature_label == "name":
            vals_df.set_index(
                pd.Series(assay.feats.fetch_all("names")).reindex(vals_df.index).values,
                inplace=True,
                drop=True,
            )

        if return_fraction:
            fracs_df = pd.DataFrame(fracs).fillna(0)
            if empty_idx is not None:
                fracs_df = fracs_df[empty_idx]
            fracs_df.set_index(vals_df.index, inplace=True, drop=True)
            return vals_df, fracs_df
        return vals_df

    def to_anndata(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        layers: dict[str, str] | None = None,
    ) -> Any:
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
        node: zarr.Group = (
            self.zw if root == "" else as_zarr_group(self.zw[root], name=root)
        )
        print(node.tree(level=depth))
        for key in node.array_keys():
            print(f"  {key}: {array_info(as_zarr_array(node[key], name=key))}")

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
        graph_grp = as_zarr_group(self.zw[loc], name=loc)
        edges = np.asarray(as_zarr_array(graph_grp["edges"], name="edges")[:])
        v = pd.DataFrame(clusts[edges[:, 1].reshape(k, n_cells)])
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
            return None

    def _prepare_cluster_tree(
        self,
        *,
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        cluster_key: str | None = None,
        fill_by_value: str | None = None,
    ) -> dict[str, Any]:
        from networkx import DiGraph, to_pandas_edgelist

        from ..dendrogram import CoalesceTree, make_digraph

        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay, cell_key, feat_key
        )
        if cluster_key is None:
            raise ValueError(
                "ERROR: Please provide a value for `cluster_key` parameter"
            )

        clusters = np.asarray(self.cells.fetch(cluster_key, key=cell_key))
        graph_loc = self._get_latest_graph_loc(from_assay, cell_key, feat_key)
        graph_grp = as_zarr_group(self.zw[graph_loc], name=graph_loc)
        dendrogram_loc = cast(str, graph_grp.attrs["latest_dendrogram"])
        coalesced_loc = dendrogram_loc + f"_coalesced_{len(set(clusters))}"

        if coalesced_loc in self.zw:
            subgraph = DiGraph()
            subgraph.add_edges_from(
                np.asarray(
                    as_zarr_array(
                        self.zw[coalesced_loc + "/edgelist"],
                        name=f"{coalesced_loc}/edgelist",
                    )[:]
                )
            )
            nodelist = np.asarray(
                as_zarr_array(
                    self.zw[coalesced_loc + "/nodelist"],
                    name=f"{coalesced_loc}/nodelist",
                )[:]
            )
            partition_ids = np.asarray(
                as_zarr_array(
                    self.zw[coalesced_loc + "/partition_id"],
                    name=f"{coalesced_loc}/partition_id",
                )[:]
            )
            cluster_labels = {str(value): value for value in set(clusters)}
            for node_data, partition_id in zip(nodelist, partition_ids):
                node = int(node_data[0])
                subgraph.nodes[node]["nleaves"] = int(node_data[1])
                partition_text = str(partition_id)
                if partition_text != "-1":
                    subgraph.nodes[node]["partition_id"] = cluster_labels.get(
                        partition_text, partition_id
                    )
        else:
            dendrogram = np.asarray(
                as_zarr_array(self.zw[dendrogram_loc], name=dendrogram_loc)[:]
            )
            subgraph = CoalesceTree(make_digraph(dendrogram), clusters)
            edge_list = to_pandas_edgelist(subgraph).values
            store = create_zarr_dataset(
                self.zw,
                f"{coalesced_loc}/edgelist",
                (100000,),
                "u8",
                edge_list.shape,
            )
            store[:] = edge_list

            node_list = []
            partition_id_list = []
            for node in subgraph.nodes():
                node_data = subgraph.nodes[node]
                partition_id = node_data.get("partition_id", -1)
                node_list.append((node, node_data["nleaves"]))
                partition_id_list.append(str(partition_id))

            node_list_arr = np.asarray(node_list)
            store = create_zarr_dataset(
                self.zw,
                f"{coalesced_loc}/nodelist",
                (100000,),
                node_list_arr.dtype,
                node_list_arr.shape,
            )
            store[:] = node_list_arr

            store = create_zarr_dataset(
                self.zw,
                f"{coalesced_loc}/partition_id",
                (100000,),
                str,
                (len(partition_id_list),),
            )
            store[:] = partition_id_list

        color_values = (
            self.get_cell_vals(
                from_assay=from_assay,
                cell_key=cell_key,
                k=fill_by_value,
            )
            if fill_by_value is not None
            else None
        )
        return {
            "graph": subgraph,
            "clusters": clusters,
            "color_values": color_values,
            "from_assay": from_assay,
            "cell_key": cell_key,
            "feat_key": feat_key,
            "cluster_key": cluster_key,
            "coalesced_location": coalesced_loc,
        }

    def metric_lisi(
        self,
        label_colnames: Iterable[str],
        use_latest_knn: bool = True,
        from_assay: str | None = None,
        knn_loc: str | None = None,
        save_result: bool = False,
        return_lisi: bool = True,
        perplexity: float = 30,
    ) -> list[tuple[str, np.ndarray]] | None:
        """Calculate Local Inverse Simpson Index (LISI) scores for cell populations.

        LISI measures how well mixed different cell populations are in the local neighborhood
        of each cell. Higher scores indicate better mixing of different populations.

        Args:
            label_colnames: Column names from cell metadata containing population labels
            use_latest_knn: Whether to use the most recent KNN graph (default: True)
            from_assay: Name of assay to use if not using latest KNN
            knn_loc: Location of KNN graph if not using latest (default: None)
            save_result: Whether to save LISI scores to cell metadata (default: False)
            return_lisi: Whether to return LISI scores (default: True)
            perplexity: Effective neighborhood size used by LISI. It is reduced
                with a warning when the graph has fewer than three times this
                many neighbors.

        Returns:
            If return_lisi is True, returns list of tuples containing:

            - Label column name
            - numpy array of LISI scores for that label

            If return_lisi is False, returns None

        Raises:
            ValueError: If KNN inputs, perplexity, or labels are invalid
            KeyError: If label columns not found in cell metadata

        Notes:
            LISI scores are computed for each label column separately.
            Scores near 1 indicate cells grouped with similar labels.
            Higher scores indicate more mixing between different labels.
        """

        label_cols = list(label_colnames)
        if from_assay is None:
            from_assay = self._load_default_assay()

        if use_latest_knn and knn_loc is None:
            knn_loc = self._get_latest_knn_loc(from_assay)
            logger.info(f"Using the latest knn graph at location: {knn_loc}")

        else:
            if knn_loc is None:
                raise ValueError("Please provide values for the KNN graph location.")
            if knn_loc not in self.zw:
                raise ValueError(f"Could not find the knn graph at location: {knn_loc}")

            logger.info(f"Using the knn graph at location: {knn_loc}")

        normed_part = knn_loc.split("/")[1]
        _, cell_key, _ = normed_part.split("__")
        knn_grp = as_zarr_group(self.zw[knn_loc], name=knn_loc)

        distances = as_zarr_array(knn_grp["distances"], name="distances")
        indices = as_zarr_array(knn_grp["indices"], name="indices")

        try:
            metadata = self.cells.to_pandas_dataframe(columns=label_cols + [cell_key])
            metadata = metadata[metadata[cell_key]]
        except KeyError:
            raise KeyError(
                f"Could not find the column(s) {label_cols} in the cell metadata table."
            )

        from ..metrics import compute_lisi

        lisi_scores = compute_lisi(
            distances,
            indices,
            metadata,
            label_cols,
            perplexity=perplexity,
        )
        # lisi_scores Shape -> (n_cells, n_labels)
        if save_result:
            for col, vals in zip(label_cols, lisi_scores.T):
                col_name = f"lisi__{col}__{knn_loc.split('/')[-1]}"
                self.cells.insert(
                    column_name=col_name, values=vals, overwrite=True, key=cell_key
                )

        if return_lisi:
            return list(zip(label_cols, lisi_scores.T))
        else:
            return None

    def metric_silhouette(
        self,
        use_latest_knn: bool = True,
        res_label: str = "leiden_cluster",
        from_assay: str | None = None,
        knn_loc: str | None = None,
        random_seed: int = 4444,
        sample_size: int = 11,
    ) -> np.ndarray | None:
        """Calculate modified silhouette scores for evaluating cluster separation.

        This implements a graph-based silhouette score that measures how similar cells
        are to their own cluster compared to the nearest neighboring cluster.

        Args:
            use_latest_knn: Whether to use most recent KNN graph (default: True)
            res_label: Base or full column name containing cluster labels
                (default: "leiden_cluster")
            from_assay: Name of assay to use if not using latest KNN (default: None)
            knn_loc: Location of KNN graph if not using latest (default: None)
            random_seed: Seed used for cluster sampling.
            sample_size: Maximum size of each sampled cluster group.

        Returns:
            numpy array of silhouette scores for each cluster, or None if computation fails

        Raises:
            ValueError: If graph, labels, sampling, or embedding data are invalid

        Notes:
            Scores range from -1 to 1:
            - Near 1: Cluster is well-separated from neighboring clusters
            - Near 0: Cluster overlaps with neighboring clusters
            - Near -1: Cluster may be incorrectly assigned

            Implementation uses sampling for efficiency with large datasets.
            NaN values indicate clusters that couldn't be scored due to size constraints.
        """

        if from_assay is None:
            from_assay = self._load_default_assay()

        if use_latest_knn and knn_loc is None:
            knn_loc = self._get_latest_knn_loc(from_assay)
            logger.info(
                f"Using the latest knn graph at location: {knn_loc} for assay: {from_assay}"
            )

        else:
            if knn_loc is None:
                raise ValueError("Please provide values for the KNN graph location.")
            if knn_loc not in self.zw:
                raise ValueError(f"Could not find the knn graph at location: {knn_loc}")
            logger.info(f"Using the knn graph at location: {knn_loc}")

        from ..metrics import silhouette_scoring

        normed_part = knn_loc.split("/")[1]
        _, cell_key, feat_key_parsed = normed_part.split("__")
        ann_obj = self._load_ann_stream(
            from_assay=from_assay,
            cell_key=cell_key,
            feat_key=feat_key_parsed,
            knn_loc=knn_loc,
        )

        knn_grp = as_zarr_group(self.z[knn_loc], name=knn_loc)
        neighbor_indices = as_zarr_array(knn_grp["indices"], name="indices")
        neighbor_distances = as_zarr_array(knn_grp["distances"], name="distances")
        if ann_obj.harmonizedData is not None:
            metric_data = ann_obj.harmonizedData
            data_is_reduced = True
        else:
            if ann_obj.harmonize:
                raise ValueError("Harmony coordinates are missing for this KNN graph")
            metric_data = ann_obj.data
            data_is_reduced = False
        scores = silhouette_scoring(
            self,
            ann_obj,
            None,
            metric_data,
            from_assay,
            res_label,
            cell_key=cell_key,
            random_seed=random_seed,
            sample_size=sample_size,
            data_is_reduced=data_is_reduced,
            distance_metric=cast(Any, ann_obj.annMetric),
            neighbor_indices=neighbor_indices,
            neighbor_distances=neighbor_distances,
        )
        return scores

    def metric_label_concordance(
        self,
        label_columns: Sequence[str],
        metric: Literal["ari", "nmi"] = "ari",
    ) -> float:
        """Compare two metadata label partitions using ARI or NMI.

        This measures whether two labelings of the same cells agree, for
        example predicted clusters against imported reference annotations. It
        does not measure batch mixing; use :meth:`metric_batch_mixing` or
        :meth:`metric_lisi` for that.

        Args:
            label_columns: Exactly two cell metadata column names to compare.
            metric: ``"ari"`` for the adjusted Rand index or ``"nmi"`` for
                normalized mutual information.

        Returns:
            Agreement between the two partitions. ARI ranges from -1 to 1 and
            NMI from 0 to 1, with higher values meaning stronger agreement.

        Raises:
            ValueError: If the number of columns or the metric name is invalid.
        """
        from ..metrics import label_concordance_score

        label_values = [
            np.asarray(self.cells.fetch_all(column)) for column in label_columns
        ]
        return label_concordance_score(label_values, metric)

    def metric_integration(
        self,
        batch_labels: list[str],
        metric: Literal["ari", "nmi"] = "ari",
    ) -> float:
        """Backward-compatible alias for :meth:`metric_label_concordance`.

        This method compares label agreement and does not measure neighborhood
        mixing. Use :meth:`metric_batch_mixing` to evaluate batch integration.
        """
        logger.warning(
            "`metric_integration` measures label concordance. Use "
            "`metric_label_concordance` for ARI/NMI or `metric_batch_mixing` "
            "for neighborhood integration quality."
        )
        return self.metric_label_concordance(batch_labels, metric)

    def metric_batch_mixing(
        self,
        label_colname: str,
        use_latest_knn: bool = True,
        from_assay: str | None = None,
        knn_loc: str | None = None,
        perplexity: float = 30,
    ) -> float:
        """Summarize batch LISI as a normalized neighborhood-mixing score.

        This computes batch LISI on the current KNN graph and rescales its mean
        against the mixing that perfectly integrated data would reach given the
        dataset's batch sizes. Unlike raw LISI, the result is bounded in
        ``[0, 1]``, which makes it easier to compare across graphs and datasets.

        Args:
            label_colname: Cell metadata column holding the batch assignment.
            use_latest_knn: Whether to use the most recent KNN graph
                (default: True).
            from_assay: Name of assay to use if not using the latest KNN.
            knn_loc: Location of the KNN graph if not using the latest.
            perplexity: Effective neighborhood size passed to LISI.

        Returns:
            A value in ``[0, 1]``. Scores near 1 indicate that neighborhoods mix
            batches as well as the global composition allows, and scores near 0
            indicate poorly mixed batches.

        Raises:
            ValueError: If KNN inputs are invalid or the column has fewer than
                two batches.
        """
        from ..metrics import lisi_batch_mixing_score

        if from_assay is None:
            from_assay = self._load_default_assay()
        resolved_knn_loc = knn_loc
        if use_latest_knn and resolved_knn_loc is None:
            resolved_knn_loc = self._get_latest_knn_loc(from_assay)
        if resolved_knn_loc is None:
            raise ValueError("Please provide values for the KNN graph location.")

        lisi_result = self.metric_lisi(
            label_colnames=[label_colname],
            use_latest_knn=use_latest_knn,
            from_assay=from_assay,
            knn_loc=resolved_knn_loc,
            save_result=False,
            return_lisi=True,
            perplexity=perplexity,
        )
        if lisi_result is None:
            raise RuntimeError("LISI computation did not return scores")

        normed_part = resolved_knn_loc.split("/")[1]
        _, cell_key, _ = normed_part.split("__")
        batch_labels = self.cells.fetch(label_colname, key=cell_key)
        return lisi_batch_mixing_score(lisi_result[0][1], batch_labels)

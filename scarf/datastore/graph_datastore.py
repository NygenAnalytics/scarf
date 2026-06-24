import os
from collections.abc import Callable

import numpy as np
import pandas as pd
from ..chunked import ChunkedArray
from loguru import logger
from scipy.sparse import coo_matrix, csr_matrix

from ..assay import Assay
from ..storage.zarr_store import zarr_root_path
from ..utils import clean_array, show_dask_progress, system_call, tqdmbar
from ..writers import create_zarr_dataset
from .base_datastore import BaseDataStore


class GraphDataStore(BaseDataStore):
    """This class extends BaseDataStore by providing methods required to
    generate a cell-cell neighbourhood graph.

    It also contains all the methods that use the KNN graphs as primary input like UMAP/tSNE embedding calculation,
    clustering, down-sampling etc.

    Attributes:
        cells: List of cell barcodes.
        nthreads: Number of threads to use for this datastore instance.
        z: The Zarr file (directory) used for this datastore instance.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _choose_reduction_method(assay: Assay, reduction_method: str) -> str:
        """This is a convenience function to determine the linear dimension
        reduction method to be used for a given assay. It is uses a
        predetermined rule to make this determination.

        Args:
            assay: Assay object.
            reduction_method: Name of reduction method to use. It can be one from either: 'pca', 'lsi', 'auto'.

        Returns:
            The name of dimension reduction method to be used. Either 'pca' or 'lsi'

        Raises:
            ValueError: If `reduction_method` is not one of either 'pca', 'lsi', 'auto'
        """
        reduction_method = reduction_method.lower()
        if reduction_method not in ["pca", "lsi", "auto", "custom"]:
            raise ValueError(
                "ERROR: Please choose either 'pca' or 'lsi' as reduction method"
            )
        if reduction_method == "auto":
            assay_type = str(assay.__class__).split(".")[-1][:-2]
            if assay_type == "ATACassay":
                logger.debug("Using LSI for dimension reduction")
                reduction_method = "lsi"
            else:
                logger.debug("Using PCA for dimension reduction")
                reduction_method = "pca"
        return reduction_method

    def _set_graph_params(
        self,
        from_assay,
        cell_key,
        feat_key,
        log_transform=None,
        renormalize_subset=None,
        reduction_method="auto",
        dims=None,
        pca_cell_key=None,
        ann_metric=None,
        ann_efc=None,
        ann_ef=None,
        ann_m=None,
        rand_state=None,
        k=None,
        n_centroids=None,
        local_connectivity=None,
        bandwidth=None,
    ) -> tuple:
        """This function allows determination of values for the parameters of
        `make_graph` function. This function harbours the default values for
        each parameter.  If parameter value is None, then before choosing the
        default, it tries to use the values from the latest iteration of the step
        within the same hierarchy tree. Find details for parameters in the
        `make_graph` method.

        Args:
            from_assay: Same as from_assay in make_graph
            cell_key: Same as cell_key in make_graph
            feat_key: Same as feat_key in make_graph
            log_transform: Same as log_transform in make_graph
            renormalize_subset: Same as renormalize_subset in make_graph
            reduction_method: Same as reduction_method in make_graph
            dims: Same as dims in make_graph
            pca_cell_key: Same as pca_cell_key in make_graph
            ann_metric: Same as ann_metric in make_graph
            ann_efc: Same as ann_efc in make_graph
            ann_ef: Same as ann_ef in make_graph
            ann_m: Same as ann_m in make_graph
            rand_state: Same as rand_state in make_graph
            k: Same as k in make_graph
            n_centroids: Same as n_centroids in make_graph
            local_connectivity: Same as local_connectivity in make_graph
            bandwidth: Same as bandwidth in make_graph

        Returns:
            Finalized values for the all the optional parameters in the same order
        """

        def log_message(category, name, value, custom_msg=None):
            """Convenience function to log variable usage messages for
            make_graph.

            Args:
                category:
                name:
                value:
                custom_msg:

            Returns:
            """
            msg = f"No value provided for parameter `{name}`. "
            if category == "default":
                msg += f"Will use default value: {value}"
                logger.debug(msg)
            elif category == "cached":
                msg += f"Will use previously used value: {value}"
                logger.debug(msg)
            else:
                if custom_msg is None:
                    return False
                else:
                    logger.info(custom_msg)
            return True

        default_values = {
            "log_transform": True,
            "renormalize_subset": True,
            "dims": 11,
            "ann_metric": "l2",
            "rand_state": 4466,
            "k": 11,
            "n_centroids": 1000,
            "local_connectivity": 1.0,
            "bandwidth": 1.5,
        }

        normed_loc = f"{from_assay}/normed__{cell_key}__{feat_key}"
        if log_transform is None or renormalize_subset is None:
            if normed_loc in self.zw and "subset_params" in self.zw[normed_loc].attrs:
                # This works in coordination with save_normalized_data
                subset_params = self.zw[normed_loc].attrs["subset_params"]
                c_log_transform, c_renormalize_subset = (
                    subset_params["log_transform"],
                    subset_params["renormalize_subset"],
                )
            else:
                c_log_transform, c_renormalize_subset = None, None
            if log_transform is None:
                if c_log_transform is not None:
                    log_transform = bool(c_log_transform)
                    log_message("cached", "log_transform", log_transform)
                else:
                    log_transform = default_values["log_transform"]
                    log_message("default", "log_transform", log_transform)
            if renormalize_subset is None:
                if c_renormalize_subset is not None:
                    renormalize_subset = bool(c_renormalize_subset)
                    log_message("cached", "renormalize_subset", renormalize_subset)
                else:
                    renormalize_subset = default_values["renormalize_subset"]
                    log_message("default", "renormalize_subset", renormalize_subset)
        log_transform = bool(log_transform)
        renormalize_subset = bool(renormalize_subset)

        if dims is None or pca_cell_key is None:
            if (
                normed_loc in self.zw
                and "latest_reduction" in self.zw[normed_loc].attrs
            ):
                reduction_loc = self.zw[normed_loc].attrs["latest_reduction"]
                c_dims, c_pca_cell_key = reduction_loc.rsplit("__", 2)[1:]
            else:
                c_dims, c_pca_cell_key = None, None
            if dims is None:
                if c_dims is not None:
                    dims = int(c_dims)
                    log_message("cached", "dims", dims)
                else:
                    dims = default_values["dims"]
                    log_message("default", "dims", dims)
            if pca_cell_key is None:
                if c_pca_cell_key is not None:
                    pca_cell_key = c_pca_cell_key
                    log_message("cached", "pca_cell_key", pca_cell_key)
                else:
                    pca_cell_key = cell_key
                    log_message("default", "pca_cell_key", pca_cell_key)
            else:
                if pca_cell_key not in self.cells.columns:
                    raise ValueError(
                        f"ERROR: `pca_use_cell_key` {pca_cell_key} does not exist in cell metadata"
                    )
                if self.cells.get_dtype(pca_cell_key) != bool:
                    raise TypeError(
                        "ERROR: Type of `pca_use_cell_key` column in cell metadata should be `bool`"
                    )
        dims = int(dims)
        reduction_method = self._choose_reduction_method(
            self._get_assay(from_assay), reduction_method
        )
        reduction_loc = (
            f"{normed_loc}/reduction__{reduction_method}__{dims}__{pca_cell_key}"
        )

        if (
            ann_metric is None
            or ann_efc is None
            or ann_ef is None
            or ann_m is None
            or rand_state is None
        ):
            if (
                reduction_loc in self.zw
                and "latest_ann" in self.zw[reduction_loc].attrs
            ):
                ann_loc = self.zw[reduction_loc].attrs["latest_ann"]
                (
                    c_ann_metric,
                    c_ann_efc,
                    c_ann_ef,
                    c_ann_m,
                    c_rand_state,
                ) = ann_loc.rsplit("/", 1)[1].split("__")[1:]
            else:
                c_ann_metric, c_ann_efc, c_ann_ef, c_ann_m, c_rand_state = (
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            if ann_metric is None:
                if c_ann_metric is not None:
                    ann_metric = c_ann_metric
                    log_message("cached", "ann_metric", ann_metric)
                else:
                    ann_metric = default_values["ann_metric"]
                    log_message("default", "ann_metric", ann_metric)
            if ann_efc is None:
                if c_ann_efc is not None:
                    ann_efc = int(c_ann_efc)
                    log_message("cached", "ann_efc", ann_efc)
                else:
                    ann_efc = None  # Will be set after value for k is determined
                    log_message("default", "ann_efc", f"min(100, max(k * 3, 50))")
            if ann_ef is None:
                if c_ann_ef is not None:
                    ann_ef = int(c_ann_ef)
                    log_message("cached", "ann_ef", ann_ef)
                else:
                    ann_ef = None  # Will be set after value for k is determined
                    log_message("default", "ann_ef", f"min(100, max(k * 3, 50))")
            if ann_m is None:
                if c_ann_m is not None:
                    ann_m = int(c_ann_m)
                    log_message("cached", "ann_m", ann_m)
                else:
                    ann_m = min(max(48, int(dims * 1.5)), 64)
                    log_message("default", "ann_m", ann_m)
            if rand_state is None:
                if c_rand_state is not None:
                    rand_state = int(c_rand_state)
                    log_message("cached", "rand_state", rand_state)
                else:
                    rand_state = default_values["rand_state"]
                    log_message("default", "rand_state", rand_state)
        ann_metric = str(ann_metric)
        ann_m = int(ann_m)
        rand_state = int(rand_state)

        if k is None:
            if (
                reduction_loc in self.zw
                and "latest_ann" in self.zw[reduction_loc].attrs
            ):
                ann_loc = self.zw[reduction_loc].attrs["latest_ann"]
                knn_loc = self.zw[ann_loc].attrs["latest_knn"]
                k = int(knn_loc.rsplit("__", 1)[1])
                log_message("cached", "k", k)
            else:
                k = default_values["k"]
                log_message("default", "k", k)
        k = int(k)
        if ann_ef is None:
            ann_ef = min(100, max(k * 3, 50))
        ann_ef = int(ann_ef)
        if ann_efc is None:
            ann_efc = min(100, max(k * 3, 50))
        ann_efc = int(ann_efc)
        ann_loc = f"{reduction_loc}/ann__{ann_metric}__{ann_efc}__{ann_ef}__{ann_m}__{rand_state}"
        knn_loc = f"{ann_loc}/knn__{k}"

        if n_centroids is None:
            if (
                reduction_loc in self.zw
                and "latest_kmeans" in self.zw[reduction_loc].attrs
            ):
                kmeans_loc = self.zw[reduction_loc].attrs["latest_kmeans"]
                n_centroids = int(
                    kmeans_loc.split("/")[-1].split("__")[1]
                )  # depends on param_joiner
                log_message("default", "n_centroids", n_centroids)
            else:
                # n_centroids = min(data.shape[0]/10, max(500, data.shape[0]/100))
                n_centroids = default_values["n_centroids"]
                log_message("default", "n_centroids", n_centroids)
        n_centroids = int(n_centroids)

        if local_connectivity is None or bandwidth is None:
            if knn_loc in self.zw and "latest_graph" in self.zw[knn_loc].attrs:
                graph_loc = self.zw[knn_loc].attrs["latest_graph"]
                c_local_connectivity, c_bandwidth = map(
                    float, graph_loc.rsplit("/")[-1].split("__")[1:]
                )
            else:
                c_local_connectivity, c_bandwidth = None, None
            if local_connectivity is None:
                if c_local_connectivity is not None:
                    local_connectivity = c_local_connectivity
                    log_message("cached", "local_connectivity", local_connectivity)
                else:
                    local_connectivity = default_values["local_connectivity"]
                    log_message("default", "local_connectivity", local_connectivity)
            if bandwidth is None:
                if c_bandwidth is not None:
                    bandwidth = c_bandwidth
                    log_message("cached", "bandwidth", bandwidth)
                else:
                    bandwidth = default_values["bandwidth"]
                    log_message("default", "bandwidth", bandwidth)
        local_connectivity = float(local_connectivity)
        bandwidth = float(bandwidth)

        return (
            log_transform,
            renormalize_subset,
            reduction_method,
            dims,
            pca_cell_key,
            ann_metric,
            ann_efc,
            ann_ef,
            ann_m,
            rand_state,
            k,
            n_centroids,
            local_connectivity,
            bandwidth,
        )

    def _get_latest_keys(
        self,
        from_assay: str | None,
        cell_key: str | None,
        feat_key: str | None,
    ) -> tuple[str, str, str]:
        if from_assay is None:
            from_assay = self._defaultAssay
        if cell_key is None:
            cell_key = self._get_latest_cell_key(from_assay)
        if feat_key is None:
            feat_key = self._get_latest_feat_key(from_assay)
        return from_assay, cell_key, feat_key

    def _get_latest_graph_loc(
        self, from_assay: str, cell_key: str, feat_key: str
    ) -> str:
        """Convenience function to identify location of the latest graph in the
        Zarr hierarchy.

        Args:
            from_assay: Name of the assay.
            cell_key: Cell key used to create the graph.
            feat_key: Feature key used to create the graph.

        Returns:
            Path of graph in the Zarr hierarchy
        """
        normed_loc = f"{from_assay}/normed__{cell_key}__{feat_key}"
        reduction_loc = self.zw[normed_loc].attrs["latest_reduction"]
        ann_loc = self.zw[reduction_loc].attrs["latest_ann"]
        knn_loc = self.zw[ann_loc].attrs["latest_knn"]
        return self.zw[knn_loc].attrs["latest_graph"]

    def _get_latest_knn_loc(self, from_assay: str = None) -> str:
        """Convenience function to identify location of the latest KNN graph in
        the Zarr hierarchy.

        Args:
            from_assay: Name of the assay.

        Returns:
            Path of KNN graph in the Zarr hierarchy
        """
        if from_assay is None:
            logger.info("Using default assay for KNN graph.")
            from_assay = self._load_default_assay()

        if from_assay not in self.assay_names:
            raise ValueError(f"ERROR: Assay {from_assay} does not exist")

        latest_cell_key = self.zw[from_assay].attrs["latest_cell_key"]
        latest_feat_key = self.zw[from_assay].attrs["latest_feat_key"]
        normed_loc = f"{from_assay}/normed__{latest_cell_key}__{latest_feat_key}"
        reduction_loc = self.zw[normed_loc].attrs["latest_reduction"]
        if "reduction" not in self.zw[reduction_loc]:
            raise ValueError(f"ERROR: PCA Reduction not found in {reduction_loc}")
        latest_ann = self.zw[reduction_loc].attrs["latest_ann"]
        ann_loc = self.zw[latest_ann]
        latest_knn = ann_loc.attrs["latest_knn"]
        return latest_knn

    def _has_ann_stream_cache(
        self,
        from_assay: str,
        cell_key: str,
        feat_key: str,
        knn_loc: str | None = None,
    ) -> bool:
        try:
            if knn_loc is None:
                normed_loc = f"{from_assay}/normed__{cell_key}__{feat_key}"
                if normed_loc not in self.zw:
                    return False
                reduction_loc = self.zw[normed_loc].attrs["latest_reduction"]
                ann_loc = self.zw[reduction_loc].attrs["latest_ann"]
                knn_loc = self.zw[ann_loc].attrs["latest_knn"]
            else:
                ann_loc = knn_loc.rsplit("/", 1)[0]

            if knn_loc not in self.zw:
                return False
            zw_root = zarr_root_path(self.zw)
            if zw_root is None:
                return False
            return os.path.exists(os.path.join(zw_root, ann_loc, "ann_idx"))
        except KeyError:
            return False

    def _load_ann_stream(
        self,
        from_assay: str,
        cell_key: str,
        feat_key: str,
        feat_scaling: bool = True,
        knn_loc: str | None = None,
    ):
        """Load an AnnStream from an existing graph without recomputing KNN."""
        from ..ann import AnnStream

        if knn_loc is None:
            normed_loc = f"{from_assay}/normed__{cell_key}__{feat_key}"
            if normed_loc not in self.zw:
                raise KeyError(f"No normalized data at {normed_loc}")
            reduction_loc = self.zw[normed_loc].attrs["latest_reduction"]
            ann_loc = self.zw[reduction_loc].attrs["latest_ann"]
            knn_loc = self.zw[ann_loc].attrs["latest_knn"]
        else:
            ann_loc = knn_loc.rsplit("/", 1)[0]
            reduction_loc = ann_loc.rsplit("/ann__", 1)[0]
            normed_loc = reduction_loc.rsplit("/reduction__", 1)[0]

        if knn_loc not in self.zw:
            raise KeyError(f"KNN graph not found at {knn_loc}")

        ann_parts = ann_loc.rsplit("/", 1)[-1].split("__")
        ann_metric = ann_parts[1]
        ann_efc, ann_ef, ann_m, rand_state = map(int, ann_parts[2:6])
        red_parts = reduction_loc.rsplit("/", 1)[-1].split("__")
        reduction_method, dims, pca_cell_key = red_parts[1], int(red_parts[2]), red_parts[3]
        k = int(knn_loc.rsplit("/", 1)[-1].split("__")[-1])

        data = ChunkedArray(self.zw[normed_loc + "/data"], nthreads=self.nthreads)
        mu, sigma = np.ndarray([]), np.ndarray([])
        if reduction_method in ["pca", "manual"]:
            mu = self.zw[normed_loc]["mu"][:]
            sigma = self.zw[normed_loc]["sigma"][:]

        loadings = None
        if "reduction" in self.zw[reduction_loc]:
            loadings = self.zw[reduction_loc]["reduction"][:]

        harmonize = self.zw[ann_loc].attrs.get("isHarmonized", False)
        harmonized_data = None
        batches = None
        if harmonize and "harmonizedData" in self.zw[reduction_loc]:
            harmonized_data = ChunkedArray(
                self.zw[reduction_loc]["harmonizedData"], nthreads=self.nthreads
            )
            batch_columns = self.zw[reduction_loc]["harmonizedData"].attrs.get(
                "batches"
            )
            if batch_columns:
                batches = pd.DataFrame(
                    {
                        x: self.cells.fetch(x, key=cell_key).astype(object)
                        for x in batch_columns
                    }
                )

        zw_root = zarr_root_path(self.zw)
        if zw_root is None:
            raise FileNotFoundError("Cannot load ANN index without a local zarr path")
        ann_index_fn = os.path.join(zw_root, ann_loc, "ann_idx")
        if not os.path.exists(ann_index_fn):
            raise FileNotFoundError(f"ANN index not found at {ann_index_fn}")

        import hnswlib

        temp_dim = dims if dims > 0 else data.shape[1]
        ann_idx = hnswlib.Index(space=ann_metric, dim=temp_dim)
        ann_idx.load_index(ann_index_fn)

        use_for_pca = self.cells.fetch(pca_cell_key, key=cell_key)
        logger.info(f"Loaded existing ANN stream from {ann_loc}")
        return AnnStream(
            data=data,
            k=k,
            n_cluster=2,
            reduction_method=reduction_method,
            dims=dims,
            loadings=loadings,
            use_for_pca=use_for_pca,
            mu=mu,
            sigma=sigma,
            ann_metric=ann_metric,
            ann_efc=ann_efc,
            ann_ef=ann_ef,
            ann_m=ann_m,
            nthreads=self.nthreads,
            ann_parallel=False,
            rand_state=rand_state,
            do_kmeans_fit=False,
            disable_scaling=not feat_scaling,
            ann_idx=ann_idx,
            lsi_skip_first=True,
            lsi_params={},
            harmonize=harmonize,
            harmonized_data=harmonized_data,
            batches=batches,
        )

    def _get_ini_embed(
        self, from_assay: str, cell_key: str, feat_key: str, n_comps: int
    ) -> np.ndarray:
        """Runs PCA on kmeans cluster centers and ascribes the PC values to
        individual cells based on their cluster labels. This is used in
        `run_umap` and `run_tsne` for initial embedding of cells. Uses
        `rescale_array` to reduce the magnitude of extreme values.

        Args:
            from_assay: Name fo the assay for which Kmeans was fit.
            cell_key: Cell key used.
            feat_key: Feature key used.
            n_comps: Number of PC components to use

        Returns:
            Matrix with n_comps dimensions representing initial embedding of cells.
        """
        from sklearn.decomposition import PCA

        from ..utils import rescale_array

        normed_loc = f"{from_assay}/normed__{cell_key}__{feat_key}"
        reduction_loc = self.zw[normed_loc].attrs["latest_reduction"]
        kmeans_loc = self.zw[reduction_loc].attrs["latest_kmeans"]
        pc = PCA(n_components=n_comps).fit_transform(
            self.zw[kmeans_loc]["cluster_centers"][:]
        )
        for i in range(n_comps):
            pc[:, i] = rescale_array(pc[:, i])
        clusters = self.zw[kmeans_loc]["cluster_labels"][:].astype(np.uint32)
        return np.array([pc[x] for x in clusters]).astype(np.float32, order="C")

    def _get_graph_ncells_k(self, graph_loc: str) -> tuple[int, int]:
        """

        Args:
            graph_loc:

        Returns:

        """
        if graph_loc.startswith(self._integratedGraphsLoc):
            attrs = self.zw[graph_loc].attrs
            return attrs["n_cells"], attrs["n_neighbors"]
        knn_loc = self.zw[graph_loc.rsplit("/", 1)[0]]
        return knn_loc["indices"].shape

    def _store_to_sparse(
        self, graph_loc: str, sparse_format: str = "csr", use_k: int | None = None
    ) -> tuple:
        """

        Args:
            graph_loc:
            sparse_format:
            use_k:

        Returns:

        """
        logger.debug(f"Loading graph from location: {graph_loc}")
        store = self.zw[graph_loc]
        n_cells, k = self._get_graph_ncells_k(graph_loc)
        # TODO: can we have a progress bar for graph loading. Append to coo matrix?
        if use_k is None:
            use_k = k
        if use_k > k:
            use_k = k
        if use_k < 1:
            use_k = 1
        if use_k != k:
            indexer = np.tile([True] * use_k + [False] * (k - use_k), n_cells)
        else:
            indexer = None
        w, e = store["weights"][:], store["edges"][:]
        if indexer is not None:
            w, e = w[indexer], e[indexer]
        if sparse_format == "csr":
            return n_cells, csr_matrix(
                (w, (e[:, 0], e[:, 1])), shape=(n_cells, n_cells)
            )
        else:
            return n_cells, coo_matrix(
                (w, (e[:, 0], e[:, 1])), shape=(n_cells, n_cells)
            )

    def make_graph(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        pca_cell_key: str | None = None,
        reduction_method: str = "auto",
        dims: int | None = None,
        k: int | None = None,
        ann_metric: str | None = None,
        ann_efc: int | None = None,
        ann_ef: int | None = None,
        ann_m: int | None = None,
        ann_parallel: bool = False,
        rand_state: int | None = None,
        n_centroids: int | None = None,
        batch_size: int | None = None,
        log_transform: bool | None = None,
        renormalize_subset: bool | None = None,
        local_connectivity: float | None = None,
        bandwidth: float | None = None,
        update_keys: bool = True,
        return_ann_object: bool = False,
        custom_loadings: np.ndarray | None = None,
        feat_scaling: bool = True,
        lsi_skip_first: bool = True,
        harmonize: bool = False,
        batch_columns: list[str] | None = None,
        show_elbow_plot: bool = False,
        ann_index_fetcher: Callable | None = None,
        ann_index_saver: Callable | None = None,
    ):
        """Creates a cell neighbourhood graph. Performs following steps in the
        process:

        - Normalizes the data calling the `save_normalized_data` for the assay
        - instantiates ANNStream class which perform dimension reduction, feature scaling (optional) and fits ANN index
        - queries ANN index for nearest neighbours and saves the distances and indices of the neighbours
        - recalculates the distances to bound them into 0 and 1 (check out knn_utils module for details)
        - saves the indices and distances in sparse graph friendly form
        - fits a MiniBatch kmeans on the data

        The data for all the steps is saved in the Zarr in the following hierarchy which is organized based on data
        dependency. Parameter values for each step are incorporated into group names in the hierarchy::

            RNA
            ├── normed__I__hvgs
            │   ├── data (7648, 2000) float64                 # Normalized data
            │   ├── mu (2000,) float64                    # Means of normalized feature values
            │   ├── sigma (2000,) float64                 # Std dev. of normalized feature values
            │   └── reduction__pca__31__I                     # Dimension reduction group
            │       ├── reduction (2000, 31) float64          # PCA loadings matrix
            │       ├── ann__l2__63__63__48__4466             # ANN group named with ANN parameters
            │       │   └── knn__21                           # KNN group with value of k in name
            │       │       ├── distances (7648, 21) float64  # Raw distance matrix for k neighbours
            │       │       ├── indices (7648, 21) uint64     # Indices for k neighbours
            │       │       └── graph__1.0__1.5               # sparse graph with continuous form distance values
            │       │           ├── edges (160608, 2) uint64
            │       │           └── weights (160608,) float64
            │       └── kmeans__100__4466                     # Kmeans groups
            │           ├── cluster_centers (100, 31) float64 # Centroid matrix
            │           └── cluster_labels (7648,) float64    # Cluster labels for cells
            ...

        The most recent child of each hierarchy node is noted for quick retrieval and in cases where multiple child
        nodes exist. Parameters starting with `ann` are forwarded to HNSWlib. More details about these parameters can
        be found here: https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md

        Args:
            from_assay: Assay to use for graph creation. If no value is provided then `defaultAssay` will be used
            cell_key: Cells to use for graph creation. By default all cells with True value in 'I' will be used.
                      The provided value for `cell_key` should be a column in cell metadata table with boolean values.
            feat_key: Features to use for graph creation. It is a required parameter. We have chosen not to set this
                      to 'I' by default because this might lead to usage of too many features and may lead to poor
                      results. The value for `feat_key` should be a column in feature metadata from the `from_assay`
                      assay and should be boolean type.
            pca_cell_key: Name of a column from cell metadata table. This column should be boolean type. If no value is
                          provided then the value is set to same as `cell_key` which means all the cells in the
                          normalized data will be used for fitting the pca. This parameter, hence, basically provides a
                          mechanism to subset the normalized data only for PCA fitting step. This parameter can be
                          useful, for example, the data has cells from multiple replicates which wont merge together, in
                          which case the `pca_cell_key` can be used to fit PCA on cells from only one of the replicate.
            reduction_method: Method to use for linear dimension reduction. Could be either 'pca', 'lsi' or 'auto'. In
                              case of 'auto' `_choose_reduction_method` will be used to determine the best reduction
                              type for the assay.
            dims: Number of top reduced dimensions to use (Default value: 11)
            k: Number of nearest neighbours to query for each cell (Default value: 11)
            ann_metric: Refer to HNSWlib link above (Default value: 'l2')
            ann_efc: Refer to HNSWlib link above (Default value: min(100, max(k * 3, 50)))
            ann_ef: Refer to HNSWlib link above (Default value: min(100, max(k * 3, 50)))
            ann_m: Refer to HNSWlib link above (Default value: min(max(48, int(dims * 1.5)), 64) )
            ann_parallel: If True, then ANN graph is created in parallel mode using DataStore.nthreads number of
                          threads. Results obtained in parallel mode will not be reproducible. (Default: False)
            rand_state: Random seed number (Default value: 4466)
            n_centroids: Number of centroids for Kmeans clustering. As a general indication, have a value of 1+ for
                         every 100 cells. Small (<2000 cells) and very small (<500 cells) use a ballpark number for max
                         expected number of clusters (Default value: 500). The results of kmeans clustering are only
                         used to provide initial embedding for UMAP and tSNE. (Default value: 500)
            batch_size: Number of cells in a batch. This number is guided by number of features being used and the
                        amount of available free memory. Though the full data is already divided into chunks, however,
                        if only a fraction of features is being used in the normalized dataset, then the chunk size
                        can be increased to speed up the computation (i.e. PCA fitting and ANN index building).
                        (Default value: 1000)
            log_transform: If True, then the normalized data is log-transformed (only affects RNAassay type assays).
                           (Default value: True)
            renormalize_subset: If True, then the data is normalized using only those features that are True in
                                `feat_key` column rather using total expression of all features in a cell (only affects
                                RNAassay type assays). (Default value: True)
            local_connectivity: This parameter is forwarded to `smooth_knn_dist` function from UMAP package. Higher
                                value will push distribution of edge weights towards terminal values (binary like).
                                Lower values will accumulate edge weights around the mean produced by `bandwidth`
                                parameter. (Default value: 1.0)
            bandwidth: This parameter is forwarded to `smooth_knn_dist` function from UMAP package. Higher value will
                       push the mean of distribution of graph edge weights towards right.  (Default value: 1.5). Read
                       more about `smooth_knn_dist` function here:
                       https://umap-learn.readthedocs.io/en/latest/api.html#umap.umap_.smooth_knn_dist
            update_keys: If True (default) then `latest_feat_key` zarr attribute of the assay will be updated.
                         Choose False if you are experimenting with a `feat_key` do not want to override existing
                         `latest_feat_key` and by extension `latest_graph`.
            return_ann_object: If True then returns the ANNStream object. This allows one to directly interact with the
                               PCA transformer and HNSWlib index. Check out ANNStream documentation to know more.
                               (Default: False)
            custom_loadings: Custom loadings/transformer for linear dimension reduction. If provided, should have a form
                             (d x p) where d is same the number of active features in feat_key and p is the number of
                             reduced dimensions. `dims` parameter is ignored when this is provided.
                             (Default value: None)
            feat_scaling: If True (default) then the feature will be z-scaled otherwise not. It is highly recommended
                          that this is kept as True unless you know what you are doing. `feat_scaling` is internally
                          turned off when during cross sample mapping using CORAL normalized values are being used.
                          Read more about this in `run_mapping` method.
            lsi_skip_first: Whether to remove the first LSI dimension when using ATAC-Seq data.
            harmonize: Run Harmony to perform batch integration
            batch_columns: Columns in cell metadata table to use for batch integration
            show_elbow_plot: If True, then an elbow plot is shown when PCA is fitted to the data. Not shown when using
                            existing PCA loadings or custom loadings. (Default value: False)
            ann_index_fetcher:
            ann_index_saver:

        Returns:
            Either None or `AnnStream` object
        """
        from ..ann import AnnStream

        if from_assay is None:
            from_assay = self._defaultAssay
        assay = self._get_assay(from_assay)
        if batch_size is None:
            batch_size = assay.rawData.chunksize[0]
        if cell_key is None:
            cell_key = "I"
        if feat_key is None:
            bool_cols = [
                x.split("__", 1)
                for x in assay.feats.columns
                if assay.feats.get_dtype(x) == bool and x != "I"
            ]
            bool_cols = [f"{x[1]}({x[0]})" for x in bool_cols]
            bool_cols = " ".join(map(str, bool_cols))
            raise ValueError(
                "ERROR: You have to choose which features that should be used for graph construction. "
                "Ideally you should have performed a feature selection step before making this graph. "
                "Feature selection step adds a column to your feature table. \n"
                "You have following boolean columns in the feature "
                f"metadata of assay {from_assay} which you can choose from: {bool_cols}\n The values in "
                f"brackets indicate the cell_key for which the feat_key is available. Choosing 'I' "
                f"as `feat_key` means that you will use all the genes for graph creation."
            )
        if custom_loadings is not None:
            reduction_method = "custom"
            dims = custom_loadings.shape[1]
            logger.info(
                f"`dims` parameter and its default value ignored as using custom loadings "
                f"with {dims} dims"
            )

        (
            log_transform,
            renormalize_subset,
            reduction_method,
            dims,
            pca_cell_key,
            ann_metric,
            ann_efc,
            ann_ef,
            ann_m,
            rand_state,
            k,
            n_centroids,
            local_connectivity,
            bandwidth,
        ) = self._set_graph_params(
            from_assay,
            cell_key,
            feat_key,
            log_transform,
            renormalize_subset,
            reduction_method,
            dims,
            pca_cell_key,
            ann_metric,
            ann_efc,
            ann_ef,
            ann_m,
            rand_state,
            k,
            n_centroids,
            local_connectivity,
            bandwidth,
        )
        batches = None
        if harmonize:
            if batch_columns is None:
                raise ValueError(f"Harmonization requested but no batches provided")
            else:
                if isinstance(batch_columns, list) is False:
                    raise ValueError(
                        f"batches must be a list of columns in cell metadata that represent batches"
                    )
                batches = pd.DataFrame(
                    {
                        x: self.cells.fetch(x, key=cell_key).astype(object)
                        for x in batch_columns
                    }
                )

        normed_loc = f"{from_assay}/normed__{cell_key}__{feat_key}"
        reduction_loc = (
            f"{normed_loc}/reduction__{reduction_method}__{dims}__{pca_cell_key}"
        )
        ann_loc = f"{reduction_loc}/ann__{ann_metric}__{ann_efc}__{ann_ef}__{ann_m}__{rand_state}"
        knn_loc = f"{ann_loc}/knn__{k}"
        kmeans_loc = f"{reduction_loc}/kmeans__{n_centroids}__{rand_state}"
        graph_loc = f"{knn_loc}/graph__{local_connectivity}__{bandwidth}"

        data = assay.save_normalized_data(
            cell_key,
            feat_key,
            batch_size,
            normed_loc.split("/")[-1],
            log_transform,
            renormalize_subset,
            update_keys,
        )
        if custom_loadings is not None and data.shape[1] != custom_loadings.shape[0]:
            raise ValueError(
                f"Provided custom loadings has {custom_loadings.shape[0]} features while the data "
                f"has {data.shape[1]} features."
            )
        loadings = None
        fit_kmeans = True
        mu, sigma = np.ndarray([]), np.ndarray([])
        use_for_pca = self.cells.fetch(pca_cell_key, key=cell_key)
        harmonized_data = None

        if reduction_method in ["pca", "manual"]:
            if "mu" in self.zw[normed_loc]:
                mu = self.zw[normed_loc]["mu"][:]
            else:
                mu = clean_array(
                    show_dask_progress(
                        data.mean(axis=0),
                        "Calculating mean of norm. data",
                        self.nthreads,
                    )
                )
                g = create_zarr_dataset(
                    self.zw[normed_loc], "mu", (100000,), "f8", mu.shape
                )
                g[:] = mu
            if "sigma" in self.zw[normed_loc]:
                sigma = self.zw[normed_loc]["sigma"][:]
            else:
                sigma = clean_array(
                    show_dask_progress(
                        data.std(axis=0),
                        "Calculating std. dev. of norm. data",
                        self.nthreads,
                    ),
                    1,
                )
                g = create_zarr_dataset(
                    self.zw[normed_loc], "sigma", (100000,), "f8", sigma.shape
                )
                g[:] = sigma
        if reduction_loc in self.zw:
            if "reduction" in self.zw[reduction_loc]:
                loadings = self.zw[reduction_loc]["reduction"][:]
                if data.shape[1] != loadings.shape[0]:
                    logger.warning(
                        "Consistency breached in loading pre-cached loadings. Will perform fresh reduction."
                    )
                    loadings = None
                    del self.zw[reduction_loc]
            if harmonize and "harmonizedData" in self.zw[reduction_loc]:
                if "batches" in self.zw[reduction_loc]["harmonizedData"].attrs:
                    if (
                        self.zw[reduction_loc]["harmonizedData"].attrs["batches"]
                        == batch_columns
                    ):
                        harmonized_data = ChunkedArray(
                            self.zw[reduction_loc]["harmonizedData"], nthreads=self.nthreads
                        )

        if custom_loadings is None:
            if loadings is not None:
                logger.info(
                    f"Using existing loadings for {reduction_method} with {dims} dims"
                )
        else:
            if loadings is not None and np.array_equal(loadings, custom_loadings):
                logger.info("Custom loadings same as used before. Loading from cache")
            else:
                loadings = custom_loadings
                logger.info(
                    f"Using custom loadings with {dims} dims. Will overwrite any "
                    f"previously used custom loadings"
                )
                if reduction_loc in self.zw:
                    del self.zw[reduction_loc]
        if harmonized_data is not None:
            # Moved this statement here to print after the status of dim loadings
            logger.info(f"Using existing harmonized data with {dims} dims")

        ann_idx = None
        if ann_loc in self.zw:
            reset_ann = False
            if "isHarmonized" in self.zw[ann_loc].attrs:
                if self.zw[ann_loc].attrs["isHarmonized"]:
                    if harmonize is False:
                        reset_ann = True
                    if harmonized_data is None:
                        reset_ann = True
                else:
                    if harmonize:
                        reset_ann = True
            else:
                if harmonize:  # Mostly for backward compatibility
                    reset_ann = True

            if reset_ann:
                del self.zw[ann_loc]
            else:
                if ann_index_fetcher is None:
                    zwRoot = zarr_root_path(self.zw)
                    if zwRoot is not None:
                        ann_index_fn = os.path.join(zwRoot, ann_loc, "ann_idx")
                    else:
                        ann_index_fn = None
                        logger.warning(
                            f"No custom `ann_index_fetcher` provided and zarr path is not local"
                        )
                else:
                    # noinspection PyBroadException
                    try:
                        ann_index_fn = ann_index_fetcher(ann_loc)
                    except:
                        ann_index_fn = None
                        logger.warning(f"Custom `ann_index_fetcher` failed")
                if ann_index_fn is None or os.path.exists(ann_index_fn) is False:
                    logger.warning(f"Ann index file expected but could not be found")
                else:
                    import hnswlib

                    temp = dims if dims > 0 else data.shape[1]
                    ann_idx = hnswlib.Index(space=ann_metric, dim=temp)
                    ann_idx.load_index(ann_index_fn)
                    # TODO: check if ANN is index is trained with expected number of cells.
                    logger.info(f"Using existing ANN index")

        if kmeans_loc in self.zw:
            fit_kmeans = False
            logger.info(f"using existing kmeans cluster centers")
        disable_scaling = True if feat_scaling is False else False

        # TODO: expose LSImodel parameters
        ann_obj = AnnStream(
            data=data,
            k=k,
            n_cluster=n_centroids,
            reduction_method=reduction_method,
            dims=dims,
            loadings=loadings,
            use_for_pca=use_for_pca,
            mu=mu,
            sigma=sigma,
            ann_metric=ann_metric,
            ann_efc=ann_efc,
            ann_ef=ann_ef,
            ann_m=ann_m,
            nthreads=self.nthreads,
            ann_parallel=ann_parallel,
            rand_state=rand_state,
            do_kmeans_fit=fit_kmeans,
            disable_scaling=disable_scaling,
            ann_idx=ann_idx,
            lsi_skip_first=lsi_skip_first,
            lsi_params={},
            harmonize=harmonize,
            harmonized_data=harmonized_data,
            batches=batches,
        )

        if reduction_loc not in self.zw:
            logger.debug(f"Saving loadings to {reduction_loc}")
            self.zw.create_group(reduction_loc, overwrite=True)
            if ann_obj.loadings is not None:
                # can be None when no dimred is performed
                g = create_zarr_dataset(
                    self.zw[reduction_loc],
                    "reduction",
                    data.chunksize,
                    "f8",
                    ann_obj.loadings.shape,
                )
                g[:, :] = ann_obj.loadings
        if harmonize and harmonized_data is None and ann_obj.harmonizedData is not None:
            g = create_zarr_dataset(
                self.zw[reduction_loc],
                "harmonizedData",
                data.chunksize,
                "f8",
                ann_obj.harmonizedData.shape,
            )
            g[:] = ann_obj.harmonizedData
            g.attrs["batches"] = batch_columns
        if ann_loc not in self.zw:
            logger.debug(f"Saving ANN index to {ann_loc}")
            self.zw.create_group(ann_loc, overwrite=True)
        if ann_idx is None:
            if ann_index_saver is None:
                zwRoot = zarr_root_path(self.zw)
                if zwRoot is not None:
                    ann_obj.annIdx.save_index(
                        os.path.join(zwRoot, ann_loc, "ann_idx")
                    )
                else:
                    logger.warning(
                        "No custom `ann_index_saver` provided and local path is unknown"
                    )
            if ann_index_saver is not None:
                try:
                    ann_index_saver(ann_obj.annIdx, ann_loc)
                except:
                    logger.warning("Custom `ann_index_saver` failed")

        if fit_kmeans:
            logger.debug(f"Saving kmeans clusters to {kmeans_loc}")
            self.zw.create_group(kmeans_loc, overwrite=True)
            g = create_zarr_dataset(
                self.zw[kmeans_loc],
                "cluster_centers",
                (1000, 1000),
                "f8",
                ann_obj.kmeans.cluster_centers_.shape,
            )
            g[:, :] = ann_obj.kmeans.cluster_centers_
            g = create_zarr_dataset(
                self.zw[kmeans_loc],
                "cluster_labels",
                (100000,),
                "f8",
                ann_obj.clusterLabels.shape,
            )
            g[:] = ann_obj.clusterLabels
        if knn_loc in self.zw and graph_loc in self.zw:
            logger.info(f"KNN graph already exists will not recompute.")
        else:
            from ..knn_utils import self_query_knn, smoothen_dists

            recall = None
            if knn_loc not in self.zw:
                recall = self_query_knn(
                    ann_obj=ann_obj,
                    store=self.zw.create_group(knn_loc, overwrite=True),
                    chunk_size=batch_size,
                    nthreads=self.nthreads,
                )
                recall = "%.2f" % recall

            smoothen_dists(
                self.zw.create_group(graph_loc, overwrite=True),
                self.zw[knn_loc]["indices"],
                self.zw[knn_loc]["distances"],
                local_connectivity,
                bandwidth,
                batch_size,
            )
            if recall is not None:
                logger.info(f"ANN recall: {recall}%")

        self.zw[normed_loc].attrs["latest_reduction"] = reduction_loc
        self.zw[reduction_loc].attrs["latest_ann"] = ann_loc
        self.zw[reduction_loc].attrs["latest_kmeans"] = kmeans_loc
        self.zw[ann_loc].attrs["isHarmonized"] = harmonize
        self.zw[ann_loc].attrs["latest_knn"] = knn_loc
        self.zw[knn_loc].attrs["latest_graph"] = graph_loc
        if return_ann_object:
            return ann_obj
        if show_elbow_plot:
            from ..plots import plot_elbow

            try:
                var_exp = 100 * ann_obj._pca.explained_variance_ratio_
            except AttributeError:
                logger.warning("PCA was not fitted so not showing an Elbow plot")
            else:
                plot_elbow(var_exp)
        return None

    def load_graph(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        symmetric: bool | None = None,
        upper_only: bool | None = None,
        use_k: int | None = None,
        graph_loc: str | None = None,
    ) -> csr_matrix:
        """Load the cell neighbourhood as a scipy sparse matrix.

        Args:
            from_assay: Name of the assay. If None then the default assay is used.
            cell_key: Cell key used to create the graph. If None then the latest feature key used for creating a
                      KNN graph is used.
            feat_key: Feature key used to create the graph. If None then the latest feature key used for creating a
                      KNN graph is used.
            symmetric: If True, makes the graph symmetric by adding it to its transpose.
            upper_only: If True, then only the values from upper triangular of the matrix are returned. This is only
                       used when symmetric is True.
            use_k: Number of top k-nearest neighbours to keep in the graph. This value must be greater than 0 and less
                   the parameter k used. By default, all neighbours are used. (Default value: None)
            graph_loc: Zarr hierarchy where the graph is stored. If no value is provided then graph location is
                       obtained from `_get_latest_graph_loc` method.

        Returns:
            A scipy sparse matrix representing cell neighbourhood graph.
        """

        def symmetrize(g):
            t = g + g.T
            t = t - g.multiply(g.T)
            return t

        from scipy.sparse import triu

        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay, cell_key, feat_key
        )

        if graph_loc is None:
            graph_loc = self._get_latest_graph_loc(from_assay, cell_key, feat_key)
        if graph_loc not in self.zw:
            raise ValueError(
                f"{graph_loc} not found in zarr location. "
                f"Run `make_graph` for assay {from_assay}"
            )
        n_cells, graph = self._store_to_sparse(graph_loc, "csr", use_k)
        if symmetric is True:
            graph = symmetrize(graph)
            if upper_only is True:
                graph = triu(graph)
        return graph
        # idx = None
        # if min_edge_weight > 0:
        #     idx = graph.data > min_edge_weight
        # # Following if-else block is for purpose for improving performance when no filtering is performed.
        # if idx is None:
        #     return graph
        # elif idx.sum() == graph.data.shape[0]:
        #     return graph
        # else:
        #     graph = graph.tocoo()
        #     return csr_matrix((graph.data[idx], (graph.row[idx], graph.col[idx])), shape=(n_cells, n_cells))

    def run_tsne(
        self,
        from_assay: str = None,
        cell_key: str = None,
        feat_key: str = None,
        symmetric_graph: bool = False,
        graph_upper_only: bool = False,
        ini_embed: np.ndarray = None,
        tsne_dims: int = 2,
        lambda_scale: float = 1.0,
        max_iter: int = 500,
        early_iter: int = 200,
        alpha: int = 10,
        box_h: float = 0.7,
        temp_file_loc: str = ".",
        label: str = "tSNE",
        verbose: bool = True,
        parallel: bool = False,
        nthreads: int = None,
    ) -> None:
        """Run SGtSNE-pi (Read more here:
        https://github.com/fcdimitr/sgtsnepi/tree/v1.0.1). This is an
        implementation of tSNE that runs directly on graph structures. We use
        the graphs generated by `make_graph` method to create a layout of cells
        using tSNE algorithm. This function makes a system call to sgtSNE
        binary. To get a better understanding of how the parameters affect the
        embedding, check this out: http://t-sne-pi.cs.duke.edu/

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            feat_key:  Feature key. Should be same as the one that was used in the desired graph. By default, the latest
                       used feature for the given assay will be used.
            symmetric_graph: This parameter is forwarded to `load_graph` and is same as there. (Default value: False)
            graph_upper_only: This parameter is forwarded to `load_graph` and is same as there. (Default value: False)
            ini_embed: Initial embedding coordinates for the cells in cell_key. Should have the same number of columns
                       as tsne_dims. If not value is provided then the initial embedding is obtained using
                       `get_ini_embed`.
            tsne_dims: Number of tSNE dimensions to compute (Default value: 2)
            lambda_scale: λ rescaling parameter (Default value: 1.0)
            max_iter: Maximum number of iterations (Default value: 500)
            early_iter: Number of early exaggeration iterations (Default value: 200)
            alpha: Early exaggeration multiplier (Default value: 10)
            box_h: Grid side length (accuracy control). Lower values might drastically slow down
                   the algorithm (Default value: 0.7)
            temp_file_loc: Location of temporary file. By default, these files will be created in the current working
                           directory. These files are deleted before the method returns.
            label: base label for tSNE dimensions in the cell metadata column (Default value: 'tSNE')
            verbose: If True (default) then the full log from SGtSNEpi algorithm is shown.
            parallel: Whether to run tSNE in parallel mode. Setting value to True will use `nthreads` threads.
                      The results are not reproducible in parallel mode. (Default value: False)
            nthreads: If parallel=True then this number of threads will be used to run tSNE. By default the `nthreads`
                      attribute of the class is used. (Default value: None)

        Returns:
        """
        from uuid import uuid4
        from ..knn_utils import export_knn_to_mtx
        from pathlib import Path
        import sys

        if sys.platform not in ["posix", "linux"]:
            logger.error(f"{sys.platform} operating system is currently not supported.")
            return None

        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay, cell_key, feat_key
        )

        uid = str(uuid4())
        knn_mtx_fn = Path(temp_file_loc, f"{uid}.mtx").resolve()
        graph = self.load_graph(
            from_assay=from_assay,
            cell_key=cell_key,
            feat_key=feat_key,
            symmetric=symmetric_graph,
            upper_only=graph_upper_only,
        )
        export_knn_to_mtx(str(knn_mtx_fn), graph)

        ini_emb_fn = Path(temp_file_loc, f"{uid}.txt").resolve()
        with open(ini_emb_fn, "w") as h:
            if ini_embed is None:
                ini_embed = self._get_ini_embed(
                    from_assay, cell_key, feat_key, tsne_dims
                ).flatten()
            else:
                if ini_embed.shape != (graph.shape[0], tsne_dims):
                    raise ValueError(
                        "ERROR: Provided initial embedding does not shape required shape: "
                        f"{(graph.shape[0], tsne_dims)}"
                    )
            h.write("\n".join(map(str, ini_embed)))
        out_fn = Path(temp_file_loc, f"{uid}_output.txt").resolve()
        if parallel:
            if nthreads is None:
                nthreads = self.nthreads
            else:
                assert type(nthreads) == int
        else:
            nthreads = 1
        cmd = (
            f"sgtsne -m {max_iter} -l {lambda_scale} -d {tsne_dims} -e {early_iter} -p {nthreads} -a {alpha}"
            f" -h {box_h} -i {ini_emb_fn} -o {out_fn} {knn_mtx_fn}"
        )
        if verbose:
            system_call(cmd)
        else:
            os.system(cmd)
        try:
            emb = pd.read_csv(out_fn, header=None, sep=" ")[
                list(range(tsne_dims))
            ].values.T
            for i in range(tsne_dims):
                self.cells.insert(
                    self._col_renamer(from_assay, cell_key, f"{label}{i + 1}"),
                    emb[i],
                    key=cell_key,
                    overwrite=True,
                )
            for fn in [out_fn, knn_mtx_fn, ini_emb_fn]:
                Path.unlink(fn)
        except FileNotFoundError:
            logger.error(
                "SG-tSNE failed, possibly due to missing libraries or file permissions. SG-tSNE currently "
                "fails on readthedocs"
            )
            for fn in [knn_mtx_fn, ini_emb_fn]:
                Path.unlink(fn)

    def run_umap(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        symmetric_graph: bool | None = False,
        graph_upper_only: bool | None = False,
        ini_embed: np.ndarray | None = None,
        umap_dims: int = 2,
        spread: float = 2.0,
        min_dist: float = 1,
        n_epochs: int = 300,
        repulsion_strength: float = 1.0,
        initial_alpha: float = 1.0,
        negative_sample_rate: float = 5,
        use_density_map: bool = False,
        dens_lambda: float = 2.0,
        dens_frac: float = 0.3,
        dens_var_shift: float = 0.1,
        random_seed: int = 4444,
        label: str = "UMAP",
        integrated_graph: str | None = None,
        parallel: bool = False,
        nthreads: int | None = None,
    ) -> None:
        """Runs UMAP algorithm using the precomputed cell-neighbourhood graph.
        The calculated UMAP coordinates are saved in the cell metadata table.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            feat_key: Feature key. Should be same as the one that was used in the desired graph. By default, the latest
                      used feature for the given assay will be used.
            symmetric_graph: This parameter is forwarded to `load_graph` and is same as there. (Default value: False)
            graph_upper_only: This parameter is forwarded to `load_graph` and is same as there. (Default value: False)
            ini_embed: Initial embedding coordinates for the cells in cell_key. Should have the same number of columns
                       as umap_dims. If not value is provided then the initial embedding is obtained using
                       `get_ini_embed`.
            umap_dims: Number of dimensions of UMAP embedding (Default value: 2)
            spread: Same as spread in UMAP package.  The effective scale of embedded points. In combination with
                    ``min_dist`` this determines how clustered/clumped the embedded points are.
            min_dist: Same as min_dist in UMAP package. The effective minimum distance between embedded points.
                      Smaller values will result in a more clustered/clumped embedding where nearby points on the
                      manifold are drawn closer together, while larger values will result on a more even dispersal of
                      points. The value should be set relative to the ``spread`` value, which determines the scale at
                      which embedded points will be spread out. (Default value: 1)
            n_epochs: Same as n_epochs in UMAP package. The number of epochs to be used in optimizing the
                      low dimensional embedding. Larger values may result in more accurate embeddings.
                      (Default value: 300)
            repulsion_strength: Same as repulsion_strength in UMAP package. Weighting applied to negative samples in
                                low dimensional embedding optimization. Values higher than one will result in greater
                                weight being given to negative samples. (Default value: 1.0)
            initial_alpha: Same as learning_rate in UMAP package. The initial learning rate for the embedding
                           optimization. (Default value: 1.0)
            negative_sample_rate: Same as negative_sample_rate in UMAP package. The number of negative samples to
                                  select per positive sample in the optimization process. Increasing this value will
                                  result in greater repulsive force being applied, greater optimization cost, but
                                  slightly more accuracy. (Default value: 5)
            use_density_map:
            dens_lambda:
            dens_frac:
            dens_var_shift:
            random_seed: (Default value: 4444)
            label: base label for UMAP dimensions in the cell metadata column (Default value: 'UMAP')
            integrated_graph:
            parallel: Whether to run UMAP in parallel mode. Setting value to True will use `nthreads` threads.
                      The results are not reproducible in parallel mode. (Default value: False)
            nthreads: If parallel=True then this number of threads will be used to run UMAP. By default, the `nthreads`
                      attribute of the class is used. (Default value: None)

        Returns:
        """
        from ..umap import fit_transform
        from ..utils import get_log_level

        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay, cell_key, feat_key
        )
        graph_loc = None
        if integrated_graph is not None:
            graph_loc = f"{self._integratedGraphsLoc}/{integrated_graph}"
            if graph_loc not in self.zw:
                raise KeyError(
                    f"ERROR: An integrated graph with label: {integrated_graph} does not exist"
                )
        graph = self.load_graph(
            from_assay=from_assay,
            cell_key=cell_key,
            feat_key=feat_key,
            symmetric=symmetric_graph,
            upper_only=graph_upper_only,
            graph_loc=graph_loc,
        )

        if ini_embed is None:
            ini_embed = self._get_ini_embed(from_assay, cell_key, feat_key, umap_dims)
        if nthreads is None:
            nthreads = self.nthreads
        verbose = False
        if get_log_level() <= 20:
            verbose = True

        if use_density_map:
            if integrated_graph is not None:
                logger.warning(
                    "DensMap is not available for integrated graphs. Will run without UMAP without DensMap"
                )
            graph_loc = self._get_latest_graph_loc(from_assay, cell_key, feat_key)
            knn_loc = graph_loc.rsplit("/", 1)[0]
            logger.trace(f"Loading KNN dists and indices from {knn_loc}")
            knn_group = self.zw[knn_loc]
            dists = knn_group["distances"][:]
            indices = knn_group["indices"][:]
            dmat = csr_matrix(
                (
                    dists.flatten(),
                    (
                        np.repeat(range(indices.shape[0]), indices.shape[1]),
                        indices.flatten(),
                    ),
                ),
                shape=(indices.shape[0], indices.shape[0]),
            )
            # dmat = dmat.maximum(dmat.transpose()).todok()
            logger.trace(f"Created sparse KNN dists and indices")
            densmap_kwds = {
                "lambda": dens_lambda,
                "frac": dens_frac,
                "var_shift": dens_var_shift,
                "n_neighbors": dists.shape[1],
                "knn_dists": dmat,
            }
        else:
            densmap_kwds = {}

        t, a, b = fit_transform(
            graph=graph.tocoo(),
            ini_embed=ini_embed,
            spread=spread,
            min_dist=min_dist,
            n_epochs=n_epochs,
            random_seed=random_seed,
            repulsion_strength=repulsion_strength,
            initial_alpha=initial_alpha,
            negative_sample_rate=negative_sample_rate,
            densmap_kwds=densmap_kwds,
            parallel=parallel,
            nthreads=nthreads,
            verbose=verbose,
        )

        if integrated_graph is not None:
            from_assay = integrated_graph
        for i in range(umap_dims):
            self.cells.insert(
                self._col_renamer(from_assay, cell_key, f"{label}{i + 1}"),
                t[:, i],
                key=cell_key,
                overwrite=True,
            )
        return None

    def run_leiden_clustering(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        resolution: float = 1.0,
        integrated_graph: str | None = None,
        symmetric_graph: bool = False,
        graph_upper_only: bool = False,
        label: str = "leiden_cluster",
        random_seed: int = 4444,
    ) -> None:
        """Executes Leiden graph clustering algorithm on the cell-neighbourhood
        graph and saves cluster identities in the cell metadata column.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            feat_key:  Feature key. Should be same as the one that was used in the desired graph. By default, the latest
                       used feature for the given assay will be used.
            resolution: Resolution parameter for `RBConfigurationVertexPartition` configuration
            integrated_graph:
            symmetric_graph: This parameter is forwarded to `load_graph` and is same as there. (Default value: True)
            graph_upper_only: This parameter is forwarded to `load_graph` and is same as there. (Default value: True)
            label: base label for cluster identity in the cell metadata column (Default value: 'leiden_cluster')
            random_seed: (Default value: 4444)

        Returns:
        """
        try:
            # noinspection PyPackageRequirements
            import leidenalg
        except ImportError:
            raise ImportError(
                "ERROR: 'leidenalg' package is not installed. Please find the installation instructions "
                "here: https://github.com/vtraag/leidenalg#installation. Also, consider running Paris "
                "instead of Leiden clustering using `run_clustering` method"
            )
        # noinspection PyPackageRequirements
        import igraph  # python-igraph

        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay, cell_key, feat_key
        )
        graph_loc = None
        if integrated_graph is not None:
            graph_loc = f"{self._integratedGraphsLoc}/{integrated_graph}"
            if graph_loc not in self.zw:
                raise KeyError(
                    f"ERROR: An integrated graph with label: {integrated_graph} does not exist"
                )
        graph = self.load_graph(
            from_assay=from_assay,
            cell_key=cell_key,
            feat_key=feat_key,
            symmetric=symmetric_graph,
            upper_only=graph_upper_only,
            graph_loc=graph_loc,
        )
        sources, targets = graph.nonzero()
        g = igraph.Graph()
        g.add_vertices(graph.shape[0])
        g.add_edges(list(zip(sources, targets)))
        g.es["weight"] = graph[sources, targets].A1
        part = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            seed=random_seed,
        )

        if integrated_graph is not None:
            from_assay = integrated_graph
        self.cells.insert(
            self._col_renamer(from_assay, cell_key, label),
            np.array(part.membership) + 1,
            fill_value=-1,
            key=cell_key,
            overwrite=True,
        )
        return None

    def run_clustering(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        n_clusters: int | None = None,
        integrated_graph: str | None = None,
        symmetric_graph: bool = False,
        graph_upper_only: bool = False,
        balanced_cut: bool = False,
        max_size: int | None = None,
        min_size: int | None = None,
        max_distance_fc: float = 2,
        force_recalc: bool = False,
        label: str = "cluster",
    ) -> None:
        """Executes Paris clustering algorithm
        (https://arxiv.org/pdf/1806.01664.pdf) on the cell-neighbourhood graph.
        The algorithm captures the multiscale structure of the graph in to an
        ordinary dendrogram structure. The distances in the dendrogram are
        based on probability of sampling node (aka cell) pairs. These methods
        create this dendrogram if it doesn't already exist for the graph and
        induces either a straight cut or balanced cut to obtain clusters of
        cells.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            feat_key:  Feature key. Should be same as the one that was used in the desired graph. By default, the latest
                       used feature for the given assay will be used.
            n_clusters: Number of desired clusters (required if balanced_cut is False)
            integrated_graph:
            symmetric_graph: This parameter is forwarded to `load_graph` and is same as there. (Default value: True)
            graph_upper_only: This parameter is forwarded to `load_graph` and is same as there. (Default value: True)
            balanced_cut: If True, then uses the balanced cut algorithm as implemented in ``BalancedCut`` to obtain
                          clusters (Default value: False)
            max_size: Same as `max_size` in ``BalancedCut``. The limit for a maximum number of cells in a cluster.
                      This parameter value is required if `balanced_cut` is True.
            min_size: Same as `min_size` in ``BalancedCut``. The limit for a minimum number of cells in a cluster.
                      This parameter value is required if `balanced_cut` is True.
            max_distance_fc:  Same as `max_distance_fc` in ``BalancedCut``. The threshold of ratio of distance between
                              two clusters beyond which they will not be merged. (Default value: 2)
            force_recalc: Forces recalculation of dendrogram even if one already exists for the graph
            label: Base label for cluster identity in the cell metadata column (Default value: 'cluster')

        Returns:
            None
        """
        import sknetwork as skn

        if balanced_cut is False:
            if n_clusters is None:
                raise ValueError(
                    "ERROR: Please provide a value for n_clusters parameter. We are working on making "
                    "this parameter free"
                )
        else:
            if n_clusters is not None:
                logger.info(
                    "Using balanced cut method for cutting dendrogram. `n_clusters` will be ignored."
                )
            if max_size is None or min_size is None:
                raise ValueError(
                    "ERROR: Please provide value for max_size and min_size"
                )

        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay, cell_key, feat_key
        )

        graph_loc = self._get_latest_graph_loc(from_assay, cell_key, feat_key)
        if integrated_graph is not None:
            graph_loc = f"{self._integratedGraphsLoc}/{integrated_graph}"
            if graph_loc not in self.zw:
                raise KeyError(
                    f"ERROR: An integrated graph with label: {integrated_graph} does not exist"
                )

        dendrogram_loc = f"{graph_loc}/dendrogram"
        # tuple are changed to list when saved as zarr attrs
        if dendrogram_loc in self.zw and force_recalc is False:
            dendrogram = self.zw[dendrogram_loc][:]
            logger.info("Using existing dendrogram")
        else:
            paris = skn.hierarchy.Paris(reorder=False)
            graph = self.load_graph(
                from_assay=from_assay,
                cell_key=cell_key,
                feat_key=feat_key,
                symmetric=symmetric_graph,
                upper_only=graph_upper_only,
                graph_loc=graph_loc,
            )
            dendrogram = paris.fit_transform(graph)
            dendrogram[dendrogram == np.inf] = 0
            g = create_zarr_dataset(
                self.zw[graph_loc],
                dendrogram_loc.rsplit("/", 1)[1],
                (5000,),
                "f8",
                (graph.shape[0] - 1, 4),
            )
            g[:] = dendrogram
        self.zw[graph_loc].attrs["latest_dendrogram"] = dendrogram_loc

        if balanced_cut:
            from ..dendrogram import BalancedCut

            labels = BalancedCut(
                dendrogram, max_size, min_size, max_distance_fc
            ).get_clusters()
            logger.info(f"{len(set(labels))} clusters found")
        else:
            labels = skn.hierarchy.cut_straight(dendrogram, n_clusters=n_clusters) + 1

        if integrated_graph is not None:
            from_assay = integrated_graph
        self.cells.insert(
            self._col_renamer(from_assay, cell_key, label),
            labels,
            fill_value=-1,
            key=cell_key,
            overwrite=True,
        )

    def run_topacedo_sampler(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        cluster_key: str | None = None,
        use_k: int | None = None,
        density_depth: int = 2,
        density_bandwidth: float = 5.0,
        max_sampling_rate: float = 0.05,
        min_sampling_rate: float = 0.01,
        min_cells_per_group: int = 3,
        snn_bandwidth: float = 5.0,
        seed_reward: float = 3.0,
        non_seed_reward: float = 0,
        edge_cost_multiplier: float = 1.0,
        edge_cost_bandwidth: float = 10.0,
        save_sampling_key: str = "sketched",
        save_density_key: str = "cell_density",
        save_mean_snn_key: str = "snn_value",
        save_seeds_key: str = "sketch_seeds",
        rand_state: int = 4466,
        return_edges: bool = False,
    ) -> None | list:
        """Perform sub-sampling (aka sketching) of cells using TopACeDo
        algorithm. Sub-sampling required that cells are partitioned in cluster
        already. Since, sub-sampling is dependent on cluster information,
        having, large number of homogeneous and even sized cluster improves
        sub-sampling results.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            feat_key: Feature key. Should be same as the one that was used in the desired graph. By default, the latest
                       used feature for the given assay will be used.
            cluster_key: Name of the column in cell metadata table where cluster information is stored.
            use_k: Number of top k-nearest neighbours to retain in the graph over which downsampling is performed.
                   BY default all neighbours are used. (Default value: None)
            density_depth: Same as 'search_depth' parameter in `calc_neighbourhood_density`. (Default value: 2)
            density_bandwidth: This value is used to scale the penalty affected by neighbourhood density. Higher values
                               will lead to a larger penalty. (Default value: 5.0)
            max_sampling_rate: Maximum fraction of cells to sample from each group. The effective sampling rate is lower
                               than this value depending on the neighbourhood degree and SNN density of cells.
                               Should be greater than 0 and less than 1. (Default value: 0.1)
            min_sampling_rate: Minimum sampling rate. Effective sampling rate is not allowed to be lower than this
                               value. Should be greater than 0 and less than 1. (Default value: 0.01)
            min_cells_per_group: Minimum number of cells to sample from each group. (Default value: 3)
            snn_bandwidth: Bandwidth for the shared nearest neighbour award. Clusters with higher mean SNN values get
                           lower sampling penalty. This value, is raised to mean SNN value of the cluster to obtain
                           sampling reward of the cluster. (Default value: 5.0)
            seed_reward: Reward/prize value for seed nodes. (Default value: 3.0)
            non_seed_reward: Reward/prize for non-seed nodes. (Default value: 0.1)
            edge_cost_multiplier: This value is multiplier to each edge's cost. Higher values will make graph traversal
                                  costly and might lead to removal of poorly connected nodes (Default value: 1.0)
            edge_cost_bandwidth: This value is raised to edge cost to get an adjusted edge cost (Default value: 1.0)
            save_sampling_key: base label for marking the cells that were sampled into a cell metadata column
                               (Default value: 'sketched')
            save_density_key: base label for saving the cell neighbourhood densities into a cell metadata column
                              (Default value: 'cell_density')
            save_mean_snn_key: base label for saving the SNN value for each cell (identified by topacedo sampler) into
                               a cell metadata column (Default value: 'snn_value')
            save_seeds_key: base label for saving the seed cells (identified by topacedo sampler) into a cell
                            metadata column (Default value: 'sketch_seeds')
            rand_state: A random values to set seed while sampling cells from a cluster randomly. (Default value: 4466)
            return_edges: If True, then steiner nodes and edges are returned. (Default value: False)

        Returns:
        """

        try:
            from topacedo import TopacedoSampler
        except ImportError:
            logger.error("Could not find topacedo package")
            return None

        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay, cell_key, feat_key
        )
        if cluster_key is None:
            raise ValueError("ERROR: Please provide a value for cluster key")
        clusters = pd.Series(self.cells.fetch(cluster_key, key=cell_key))
        graph = self.load_graph(
            from_assay=from_assay,
            cell_key=cell_key,
            feat_key=feat_key,
            symmetric=False,
            upper_only=False,
            use_k=use_k,
        )
        graph_loc = self._get_latest_graph_loc(from_assay, cell_key, feat_key)
        try:
            dendrogram = self.zw[f"{graph_loc}/dendrogram"][:]
        except KeyError:
            raise KeyError(
                "ERROR: Couldn't find the dendrogram for clustering. Please note that "
                "TopACeDo requires a dendrogram from Paris clustering."
            )

        if len(clusters) != graph.shape[0]:
            raise ValueError(
                f"ERROR: cluster information exists for {len(clusters)} cells while graph has "
                f"{graph.shape[0]} cells."
            )
        sampler = TopacedoSampler(
            graph,
            clusters.values,
            dendrogram,
            density_depth,
            density_bandwidth,
            max_sampling_rate,
            min_sampling_rate,
            min_cells_per_group,
            snn_bandwidth,
            seed_reward,
            non_seed_reward,
            edge_cost_multiplier,
            edge_cost_bandwidth,
            rand_state,
        )
        nodes, edges = sampler.run()
        a = np.zeros(self.cells.fetch_all(cell_key).sum()).astype(bool)
        a[nodes] = True
        key = self._col_renamer(from_assay, cell_key, save_sampling_key)
        self.cells.insert(key, a, fill_value=False, key=cell_key, overwrite=True)
        logger.debug(f"Sketched cells saved under column '{key}'")

        key = self._col_renamer(from_assay, cell_key, save_density_key)
        self.cells.insert(key, sampler.densities, key=cell_key, overwrite=True)
        logger.debug(f"Cell neighbourhood densities saved under column: '{key}'")

        key = self._col_renamer(from_assay, cell_key, save_mean_snn_key)
        self.cells.insert(key, sampler.meanSnn, key=cell_key, overwrite=True)
        logger.debug(f"Mean SNN values saved under column: '{key}'")

        a = np.zeros(self.cells.fetch_all(cell_key).sum()).astype(bool)
        a[sampler.seeds] = True
        key = self._col_renamer(from_assay, cell_key, save_seeds_key)
        self.cells.insert(key, a, fill_value=False, key=cell_key, overwrite=True)
        logger.debug(f"Seed cells saved under column: '{key}'")

        if return_edges:
            return edges

    def get_imputed(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        feature_name: str | None = None,
        feat_key: str | None = None,
        t: int = 2,
        cache_operator: bool = True,
    ) -> np.ndarray:
        """

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            feature_name: Name of the feature to be imputed
            feat_key: Feature key. Should be same as the one that was used in the desired graph. By default, the latest
                       used feature for the given assay will be used.
            t: Same as the t parameter in MAGIC. Higher values lead to larger diffusion of values. Too large values
               can slow down the algorithm and cause over-smoothening. (Default value: 2)
            cache_operator: Whether to keep the diffusion operator in memory after the method returns. Can be useful
                            to set to True if many features are to imputed in a batch but can lead to increased memory
                            usage. (Default value: True)

        Returns:
            An array of imputed values for the given feature

        """

        def calc_diff_operator(g: csr_matrix, to_power: int) -> coo_matrix:
            d = np.ravel(g.sum(axis=1))
            d[d != 0] = 1 / d[d != 0]
            n = g.shape[0]
            d = csr_matrix((d, (range(n), range(n))), shape=[n, n])
            return d.dot(g).__pow__(to_power).tocoo()

        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay, cell_key, feat_key
        )
        if feature_name is None:
            raise ValueError(
                "ERROR: Please provide name for the feature to be imputed. It can, for example, "
                "be a gene name."
            )
        data = self.get_cell_vals(
            from_assay=from_assay, cell_key=cell_key, k=feature_name
        )

        graph_loc = self._get_latest_graph_loc(from_assay, cell_key, feat_key)
        magic_loc = f"{graph_loc}/magic_{t}"
        if magic_loc in self.zw:
            logger.info("Using existing MAGIC diffusion operator")
            if self._cachedMagicOperatorLoc == magic_loc:
                diff_op = self._cachedMagicOperator
            else:
                n_cells, _ = self._get_graph_ncells_k(graph_loc)
                store = self.zw[magic_loc]
                diff_op = coo_matrix(
                    (store["data"][:], (store["row"][:], store["col"][:])),
                    shape=(n_cells, n_cells),
                )
                if cache_operator:
                    self._cachedMagicOperator = diff_op
                    self._cachedMagicOperatorLoc = magic_loc
                else:
                    self._cachedMagicOperator = None
                    self._cachedMagicOperatorLoc = None
        else:
            graph = self.load_graph(
                from_assay=from_assay,
                cell_key=cell_key,
                feat_key=feat_key,
                symmetric=True,
                upper_only=False,
            )
            diff_op = calc_diff_operator(graph, t)
            shape = diff_op.data.shape
            store = self.zw.create_group(magic_loc, overwrite=True)
            for i, j in zip(["row", "col", "data"], ["uint32", "uint32", "float32"]):
                zg = create_zarr_dataset(store, i, (1000000,), j, shape)
                zg[:] = diff_op.__getattribute__(i)
            self.zw[graph_loc].attrs["latest_magic"] = magic_loc
            if cache_operator:
                self._cachedMagicOperator = diff_op
                self._cachedMagicOperatorLoc = magic_loc
            else:
                self._cachedMagicOperator = None
                self._cachedMagicOperatorLoc = None
        return diff_op.dot(data)

    def run_pseudotime_scoring(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        subset_cell_key: str | None = None,
        feat_key: str | None = None,
        n_singular_vals: int = 30,
        source_sink_key: str | None = None,
        sources: list | None = None,
        sinks: list | None = None,
        ss_vec: np.ndarray | None = None,
        min_max_norm_ptime: bool = True,
        random_seed: int = 4444,
        label: str = "pseudotime",
    ) -> None:
        """
        Calculate differentiation potential of cells. This function is a reimplementation of population balance
        analysis (PBA) approach published in Weinreb et al. 2017, PNAS. This function computes the random walk
        normalized Laplacian matrix of the reference graph, L_rw = I-A/D and then calculates a Moore-Penrose
        pseudoinverse of L_rw.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            subset_cell_key: Cell key for the subset of cells for which pseudotime scoring is to be performed.
            feat_key: Feature key. Should be same as the one that was used in the desired graph. By default, the latest
                        used feature for the given assay will be used.
            n_singular_vals: Number of the smallest singular values to save.
            source_sink_key: Name of a column from cell attributes table that shall be used for fetching source and
                             sink groups. Usually this will a column containing cell cluster/group identities.
            sources: A list of group/clusters ids from `source_sink_key` column to be treated as sources. Sources are
                     usually progenitor/precursor or other actively dividing cell states.
            sinks: A list of group/clusters ids from `source_sink_key` column to be treated as sinks. Sinks are usually
                   more differentiated (or terminally differentiated) cell states.
            ss_vec: A vector that contains source sink values for each cell. If not provided then, this vector is
                    internally computed using the `sources` and `sinks` parameter. This vector should add up to 0 and
                    should have negative values for source cells and positive values for sink cells.
            min_max_norm_ptime: Whether to perform min-max normalization on the final pseudotime values so that values
                                are in 0 to 1 range. (Default: True)
            random_seed: A random seed for svds (Default: 4444)
            label: label: Base label for pseudotime in the cell metadata column (Default value: 'pseudotime')

        Returns:

        """

        from scipy.sparse.linalg import svds

        def inverse_degree(g):
            d = np.ravel(g.sum(axis=1))
            n = g.shape[0]
            d[d != 0] = 1 / d[d != 0]
            return csr_matrix((d, (range(n), range(n))), shape=[n, n])

        def laplacian(g, inv_deg):
            n = g.shape[0]
            identity = csr_matrix((np.ones(n), (range(n), range(n))), shape=[n, n])
            return identity - g.dot(inv_deg)

        def make_source_sink_vector(c, source, sink):
            ss = list(source) + list(sink)

            r = np.zeros(c.shape[0])
            r[c.isin(sink)] = 1
            r[c.isin(source)] = -1

            n = c.isin(ss).sum()
            v = (0 - r.sum()) / (r.shape[0] - n)
            r[~c.isin(ss)] = v
            return r

        def pseudo_inverse(lap, k, rseed, r):
            random_state = np.random.RandomState(rseed)
            # noinspection PyArgumentList
            v0 = random_state.rand(lap.shape[0])
            # TODO: add thread management here
            logger.info(
                "Calculating SVD of graph laplacian. This might take a while...",
            )
            u, s, vt = svds(lap, k=k, which="SM", v0=v0)
            # Because the order of singular values is not guaranteed
            idx = np.argsort(s)
            # Extracting the second smallest values
            s = s[idx][1:].T
            s = 1 / s
            u = u[:, idx][:, 1:]
            vt = vt[idx, :][1:, :].T
            # Computing matmul in an iterative way to save memory
            n = u.shape[0]
            # TODO: Use numba for this part
            ilap = np.zeros(n)
            for i in tqdmbar(range(n), desc="Calculating pseudotime"):
                ilap[i] = (vt * u[i, :] * s * r).sum()
            return ilap

        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay, cell_key, feat_key
        )
        if subset_cell_key is None:
            subset_cell_key = cell_key
            cell_idx = self.cells.fetch(subset_cell_key, key=cell_key)
        else:
            cell_idx = self.cells.fetch(subset_cell_key, key=cell_key)
            if cell_idx.sum() != self.cells.fetch_all(subset_cell_key).sum():
                raise ValueError("subset_cell_key is not a complete subset of cell_key")

        graph = self.load_graph(
            from_assay=from_assay,
            cell_key=cell_key,
            feat_key=feat_key,
            symmetric=True,
            upper_only=False,
        )
        if cell_idx.shape[0] != cell_idx.sum():
            graph = graph[cell_idx][:, cell_idx]

        if source_sink_key is None:
            if sources is not None or sinks is not None:
                logger.warning(
                    "Provide `sources` and `sinks` will not be used because `source_sink_key` has not been "
                    "provided"
                )
        if ss_vec is None:
            if source_sink_key is None:
                logger.warning(
                    "No source/sink info or custom source sink vector provided. The results might not be "
                    "reflect true pseudotime."
                )
                ss_vec = np.ones(graph.shape[0])
            else:
                clusts = pd.Series(
                    self.cells.fetch(source_sink_key, key=cell_key)[cell_idx]
                )
                if sources is None:
                    sources = []
                else:
                    if isinstance(sources, list) is False:
                        raise ValueError(
                            "ERROR: Parameter `sources` should be of 'list' type"
                        )
                if sinks is None:
                    sinks = []
                else:
                    if isinstance(sinks, list) is False:
                        raise ValueError(
                            "ERROR: Parameter `sinks` should be of 'list' type"
                        )
                ss_vec = make_source_sink_vector(clusts, sources, sinks)
        else:
            if source_sink_key is not None:
                logger.warning(
                    "Sources/sinks from `source_sink_key` will not be because custom vector `ss_vec` is "
                    "provided"
                )
            ss_vec = np.array(ss_vec)
            if ss_vec.shape[0] != graph.shape[0]:
                raise ValueError(
                    f"ERROR: Size mismatch between `ss_vec` ({ss_vec.shape[0]}) and "
                    f"graph ({graph.shape[0]:})"
                )
            if ss_vec.sum() > 1e-10:
                raise ValueError(
                    f"ERROR: The sum of all the values in `ss_vec` should be zero. Here we test if the sum is less"
                    f" 1e-10"
                )

        ss_vec = ss_vec.reshape(-1, 1)

        ptime = pseudo_inverse(
            laplacian(graph, inverse_degree(graph)),
            n_singular_vals,
            random_seed,
            ss_vec,
        )
        if min_max_norm_ptime:
            # noinspection PyArgumentList
            ptime = ptime - ptime.min()
            ptime = ptime / ptime.max()

        self.cells.insert(
            self._col_renamer(from_assay, subset_cell_key, label),
            ptime,
            key=subset_cell_key,
            overwrite=True,
        )
        return None

    def integrate_assays(
        self,
        assays: list[str],
        label: str,
        method: str = "snn",
        chunk_size: int = 10000,
    ) -> None:
        """Merges KNN graphs of two or more assays from within the same
        DataStore. The input KNN graphs should have been constructed on the
        same set of cells and should each have been constructed with equal
        number of neighbours (parameter: k) The merged KNN graph has the same
        size and shape as the input graphs.

        Args:
            assays: Name of the input assays. The latest constructed graph from each assay is used.
            label: Label for integrated graph
            method: Choose a method for modality integration. Available options: 'snn': Shared nearest neighbour
                    approach and 'wnn': Weighted nearest neighbor approach based on Hao et. alm Cell 2022.
            chunk_size: number of cells to be loaded at a time while reading and writing the graph

        Returns: None
        """
        from ..knn_utils import merge_graphs, wnn_integration

        def load_pca_knn(assay_name):
            g = self.load_graph(
                from_assay=assay_name,
                cell_key=None,
                feat_key=None,
                symmetric=False,
                upper_only=False,
            ).tocsr()
            ao = self.make_graph(
                from_assay=assay_name,
                feat_key=self._get_latest_feat_key(assay_name),
                return_ann_object=True,
                update_keys=False,
            )
            if ao.loadings is None:
                logger.warning(
                    f"No dimension reduction was user for {assay_name} data. "
                    f"Memory consumption will be high."
                )
                return g, ao.data.compute()
            else:
                return g, ao.data.dot(ao.loadings).compute()

        if method == "snn":
            merged_graph = []
            for assay in assays:
                if assay not in self.assay_names:
                    raise ValueError(f"ERROR: Assay {assay} was not found.")
                merged_graph.append(
                    self.load_graph(
                        from_assay=assay,
                        cell_key=None,
                        feat_key=None,
                        symmetric=False,
                        upper_only=False,
                    ).tocsr()
                )
            merged_graph = merge_graphs(merged_graph)
        elif method == "wnn":
            if len(assays) != 2:
                raise ValueError(
                    "WNN integration in Scarf can currently be performed using only two assays"
                )
            g1, ld1 = load_pca_knn(assays[0])
            g2, ld2 = load_pca_knn(assays[1])
            merged_graph = wnn_integration(
                assays[0], g1, ld1, assays[1], g2, ld2, self.nthreads
            )
        else:
            raise ValueError(
                f"Method {method} not supported, choose one of these: 'snn', 'wnn'"
            )

        n_cells = merged_graph.shape[0]
        n_neighbors = int(merged_graph.size / n_cells)

        ig_loc = self._integratedGraphsLoc
        if ig_loc not in self.zw:
            self.zw.create_group(ig_loc)
        if label in self.zw[ig_loc]:
            del self.zw[f"{ig_loc}/{label}"]
        store = self.zw.create_group(f"{ig_loc}/{label}")
        store.attrs["n_cells"] = n_cells
        store.attrs["n_neighbors"] = n_neighbors

        zge = create_zarr_dataset(
            store,
            f"edges",
            (chunk_size,),
            (np.uint32, np.uint32),
            (n_cells * n_neighbors, 2),
        )
        zgw = create_zarr_dataset(
            store, f"weights", (chunk_size,), np.float64, (n_cells * n_neighbors)
        )

        zge[:, 0] = merged_graph.row
        zge[:, 1] = merged_graph.col
        zgw[:] = merged_graph.data

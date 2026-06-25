from collections.abc import Callable, Generator
from typing import Any, cast

import numpy as np
import pandas as pd
from loguru import logger
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from .._types import as_zarr_array, as_zarr_group
from ..ann import AnnStream
from ..assay import Assay, RNAassay
from ..chunked import ChunkedArray
from .graph_datastore import GraphDataStore
from ..utils import (
    show_dask_progress,
    clean_array,
    tqdmbar,
    controlled_compute,
)
from ..writers import create_zarr_dataset


class MappingDatastore(GraphDataStore):
    """This class extends GraphDataStore by providing methods for mapping/
    projection of cells from one DataStore onto another. It also contains the methods
    required for label transfer, mapping score generation and co-embedding.
    """

    def run_mapping(
        self,
        target_assay: Assay,
        target_name: str,
        target_feat_key: str,
        target_cell_key: str = "I",
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        save_k: int = 3,
        batch_size: int = 1000,
        ref_mu: bool = True,
        ref_sigma: bool = True,
        run_coral: bool = False,
        exclude_missing: bool = False,
        filter_null: bool = False,
        feat_scaling: bool = True,
        ann_index_fetcher: Callable | None = None,
        ann_index_saver: Callable | None = None,
    ) -> None:
        """Projects cells from external assays into the cell-neighbourhood
        graph using existing PCA loadings and ANN index. For each external cell
        (target) nearest neighbours are identified and save within the Zarr
        hierarchy group `projections`.

        Args:
            target_assay: Assay object of the target dataset.
            target_name: Name of target data. This used to keep track of projections in the Zarr hierarchy
            target_feat_key: This will be used to name wherein the normalized target data will be saved in its own
                             zarr hierarchy.
            target_cell_key: Cell key for the target data. (Default value: 'I')
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            feat_key:  Feature key. Should be same as the one that was used in the desired graph. By default, the latest
                       used feature for the given assay will be used.
            save_k: Number of the nearest neighbours to identify for each target cell (Default value: 3)
            batch_size: Number of cells that will be projected as a batch. This used to decide the chunk size when
                        normalized data for the target cells is saved to disk.
            ref_mu: If True (default), Then mean values of features as in the reference are used,
                    otherwise mean is calculated using target cells. Turning this to False is not recommended.
            ref_sigma: If True (default), Then standard deviation values of features as present in the reference are
                       used, otherwise std. dev. is calculated using target cells. Turning this to False is not
                       recommended.
            run_coral: If True then CORAL feature rescaling algorithm is used to correct for domain shift in target
                       cells. Read more about CORAL algorithm in function ``coral``. This algorithm creates a m by m
                       matrix where m is the number of features being used for mapping; so it is not advised to use this
                       in a case where a large number of features are being used (>10k for example).
                       (Default value: False)
            exclude_missing: If set to True then only those features that are present in both reference and
                             target are used. If not all reference features from `feat_key` are present in target data
                             then a new graph will be created for reference and mapping will be done onto that graph.
                             (Default value: False)
            filter_null: If True then those features that have a total sum of 0 in the target cells are removed.
                         This has an affect only when `exclude_missing` is True. (Default value: False)
            feat_scaling: If False then features from target cells are not scaled. This is automatically set to False
                          if `run_coral` is True (Default value: True). Setting this to False is not recommended.
            ann_index_fetcher:
            ann_index_saver:

        Returns:
            None
        """
        from ..mapping_utils import align_features, coral

        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay, cell_key, feat_key
        )
        source_assay = self._get_assay(from_assay)

        if type(target_assay) is not type(source_assay):
            raise TypeError(
                f"ERROR: Source assay ({type(source_assay)}) and target assay "
                f"({type(target_assay)}) are of different types. "
                f"Mapping can only be performed between same assay types"
            )
        if isinstance(target_assay, RNAassay):
            if target_assay.sf != source_assay.sf:
                logger.info(
                    f"Resetting target assay's size factor from {target_assay.sf} to {source_assay.sf}"
                )
                target_assay.sf = source_assay.sf

        if target_feat_key == feat_key:
            raise ValueError(
                f"ERROR: `target_feat_key` cannot be sample as `feat_key`: {feat_key}"
            )

        feat_idx = align_features(
            source_assay,
            target_assay,
            cell_key,
            feat_key,
            target_feat_key,
            target_cell_key,
            filter_null,
            exclude_missing,
            self.nthreads,
        )
        logger.debug(f"{len(feat_idx)} features being used for mapping")
        if np.all(
            source_assay.feats.active_index(cell_key + "__" + feat_key) == feat_idx
        ):
            ann_feat_key = feat_key
        else:
            ann_feat_key = f"{feat_key}_common_{target_name}"
            feat_mask = np.zeros(source_assay.feats.N).astype(bool)
            feat_mask[feat_idx] = True
            source_assay.feats.insert(
                cell_key + "__" + ann_feat_key,
                feat_mask,
                fill_value=False,
                overwrite=True,
            )
        if run_coral:
            feat_scaling = False
        if (
            ann_feat_key == feat_key
            and ann_index_fetcher is None
            and ann_index_saver is None
            and self._has_ann_stream_cache(from_assay, cell_key, ann_feat_key)
        ):
            ann_obj: AnnStream | None = self._load_ann_stream(
                from_assay,
                cell_key,
                ann_feat_key,
                feat_scaling=feat_scaling,
            )
        else:
            ann_obj = self.make_graph(
                from_assay=from_assay,
                cell_key=cell_key,
                feat_key=ann_feat_key,
                return_ann_object=True,
                update_keys=False,
                feat_scaling=feat_scaling,
                ann_index_fetcher=ann_index_fetcher,
                ann_index_saver=ann_index_saver,
            )
        if ann_obj is None:
            raise ValueError("ERROR: AnnStream could not be created for mapping")
        if save_k > ann_obj.k:
            logger.warning(f"`save_k` was decreased to {ann_obj.k}")
            save_k = ann_obj.k
        target_data = ChunkedArray(
            as_zarr_array(
                target_assay.z[
                    f"normed__{target_cell_key}__{target_feat_key}/data"
                ],
                name=f"normed__{target_cell_key}__{target_feat_key}/data",
            ),
            nthreads=self.nthreads,
        )
        if run_coral is True:
            # Reversing coral here to correct target data
            coral(
                target_data,
                ann_obj.data,
                target_assay,
                target_feat_key,
                target_cell_key,
                self.nthreads,
            )
            target_data = ChunkedArray(
                as_zarr_array(
                    target_assay.z[
                        f"normed__{target_cell_key}__{target_feat_key}/data_coral"
                    ],
                    name=(
                        f"normed__{target_cell_key}__{target_feat_key}/data_coral"
                    ),
                ),
                nthreads=self.nthreads,
            )
        if ann_obj.method == "pca" and run_coral is False:
            if ref_mu is False:
                mu = show_dask_progress(
                    target_data.mean(axis=0),
                    "Calculating mean of target norm. data",
                    self.nthreads,
                )
                ann_obj.mu = clean_array(mu)
            if ref_sigma is False:
                sigma = show_dask_progress(
                    target_data.std(axis=0),
                    "Calculating std. dev. of target norm. data",
                    self.nthreads,
                )
                ann_obj.sigma = clean_array(sigma, 1)
        if "projections" not in source_assay.z:
            source_assay.z.create_group("projections")
        projections = as_zarr_group(
            source_assay.z["projections"], name="projections"
        )
        store = projections.create_group(target_name, overwrite=True)
        nc = target_assay.cells.active_index(target_cell_key).shape[0]
        nk = save_k
        zi = create_zarr_dataset(store, "indices", (batch_size,), "u8", (nc, nk))
        zd = create_zarr_dataset(store, "distances", (batch_size,), "f8", (nc, nk))
        entry_start = 0
        for i in tqdmbar(
            target_data.blocks,
            desc=f"Mapping cells from {target_name}",
            total=target_data.numblocks[0],
        ):
            block: NDArray[Any] = controlled_compute(i, self.nthreads)
            knn_query = ann_obj.transform_ann(ann_obj.reducer(block), k=save_k)
            ki, kd = knn_query[0], knn_query[1]
            entry_end = entry_start + len(ki)
            zi[entry_start:entry_end, :] = ki
            zd[entry_start:entry_end, :] = kd
            entry_start = entry_end
        return None

    def get_mapping_score(
        self,
        target_name: str,
        target_groups: np.ndarray | None = None,
        from_assay: str | None = None,
        cell_key: str | None = None,
        log_transform: bool = True,
        multiplier: float = 1000,
        weighted: bool = True,
        fixed_weight: float = 0.1,
    ) -> Generator[tuple[str, np.ndarray], None, None]:
        """Yields the mapping scores that were a result of a mapping.

        Mapping scores are an indication of degree of similarity of reference cells in the graph to the target cells.
        The more often a reference cell is found in the nearest neighbour list of the target cells, the higher
        the mapping score will be for that cell.

        Args:
            target_name: Name of target data. This used to keep track of projections in the Zarr hierarchy
            target_groups: Group/cluster identity of target cells. This will then be used to calculate mapping score
                           for each group separately.
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            log_transform: If True (default) then the mapping scores will be log transformed
            multiplier: A scaling factor for mapping scores. All scores al multiplied this value. This mostly intended
                        for visualization of mapping scores (Default: 1000)
            weighted: Use distance weights when calculating mapping scores (default: True). If False then the actual
                      distances between the reference and target cells are ignored.
            fixed_weight: Used when `weighted` is False. This is the value that is added to mapping score of each
                          reference cell for every projected target cell. Can be any value >0.

        Yields:
            A tuple of group name and mapping score of reference cells for that target group.
        """
        from_assay, cell_key, _ = self._get_latest_keys(from_assay, cell_key, None)
        store_loc = f"{from_assay}/projections/{target_name}"
        if store_loc not in self.zw:
            raise KeyError(
                f"ERROR: Projections have not been computed for {target_name} in th latest graph. Please"
                f" run `run_mapping` or update latest_graph by running `make_graph` with desired parameters"
            )
        store = as_zarr_group(self.zw[store_loc], name=store_loc)

        indices = np.asarray(
            as_zarr_array(store["indices"], name="indices")[:]
        )
        dists = np.asarray(
            as_zarr_array(store["distances"], name="distances")[:]
        )
        # TODO: add more robust options for distance calculation here
        dists = 1 / (np.log1p(dists) + 1)
        n_cells, n_k = indices.shape

        if target_groups is not None:
            if len(target_groups) != n_cells:
                raise ValueError(
                    f"ERROR: Length of target_groups {len(target_groups)} not same as number of target "
                    f"cells in the projection {n_cells}"
                )
            groups = pd.Series(target_groups)
        else:
            groups = pd.Series(np.zeros(n_cells))

        ref_n_cells = self.cells.active_index(cell_key).shape[0]
        for group in sorted(groups.unique()):
            coi = {x: None for x in groups[groups == group].index.values}
            ms = np.zeros(ref_n_cells)
            for n, i, j in zip(range(len(indices)), indices, dists):
                if n in coi:
                    for x, y in zip(i, j):
                        if weighted:
                            ms[x] += y
                        else:
                            ms[x] += fixed_weight
            ms = multiplier * ms / (len(coi) * n_k)
            if log_transform:
                ms = np.log1p(ms)
            yield group, ms

    def get_target_classes(
        self,
        target_name: str,
        from_assay: str | None = None,
        cell_key: str | None = None,
        reference_class_group: str | None = None,
        threshold_fraction: float = 0.5,
        target_subset: list[int] | None = None,
        na_val: str = "NA",
    ) -> pd.Series:
        """Perform classification of target cells using a reference group.

        Args:
            target_name: Name of target data. This value should be the same as that used for `run_mapping` earlier.
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            reference_class_group: Group/cluster identity of the reference cells. These are the target labels for the
                                   classifier. The value here should be a column from cell metadata table. For
                                   example, to use default clustering identity one could use `RNA_cluster`
            threshold_fraction: The threshold for deciding if a cell belongs to a group or not.
                                Constrained between 0 and 1. (Default value: 0.5)
            target_subset: Choose only a subset of target cells to be classified. The value should be a list of
                           indices of the target cells. (Default: None)
            na_val: Value to be used if a cell is not classified to any of the `reference_class_group`.
                    (Default value: 'NA')

        Returns: A pandas Series containing predicted class for each cell in the projected sample (`target_name`).
        """
        from_assay, cell_key, _ = self._get_latest_keys(from_assay, cell_key, None)
        store_loc = f"{from_assay}/projections/{target_name}"
        if store_loc not in self.zw:
            raise KeyError(
                f"ERROR: Projections have not been computed for {target_name} in th latest graph. Please"
                f" run `run_mapping` or update latest_graph by running `make_graph` with desired parameters"
            )
        if reference_class_group is None:
            raise ValueError(
                "ERROR: A value is required for the parameter `reference_class_group`. "
                "This can be any cell metadata column. Please choose the value that contains cluster or "
                "group information"
            )
        ref_groups = self.cells.fetch(reference_class_group, key=cell_key)
        if threshold_fraction < 0 or threshold_fraction > 1:
            raise ValueError(
                "ERROR: `threshold_fraction` should have a value between 0 and 1"
            )
        target_subset_set: dict[int, None] | None = None
        if target_subset is not None:
            if not isinstance(target_subset, list):
                raise TypeError("ERROR:  `target_subset` should be <list> type")
            target_subset_set = {x: None for x in target_subset}

        store = as_zarr_group(self.zw[store_loc], name=store_loc)
        indices = np.asarray(
            as_zarr_array(store["indices"], name="indices")[:]
        )
        dists = np.asarray(
            as_zarr_array(store["distances"], name="distances")[:]
        )

        preds = []
        weights = 1 - (dists / dists.max(axis=1).reshape(-1, 1))
        for n in range(indices.shape[0]):
            if target_subset_set is not None and n not in target_subset_set:
                continue
            wd = {}
            for i, j in zip(indices[n, :-1], weights[n, :-1]):
                k = ref_groups[i]
                if k not in wd:
                    wd[k] = 0
                wd[k] += j
            temp = na_val
            s = weights[n, :-1].sum()
            for i, j in wd.items():
                if j / s > threshold_fraction:
                    if temp == na_val:
                        temp = i
                    else:
                        temp = na_val
                        break
            preds.append(temp)
        return pd.Series(preds)

    def load_unified_graph(
        self,
        from_assay: str | None,
        cell_key: str | None,
        feat_key: str | None,
        target_names: list[str],
        use_k: int,
        target_weight: float,
    ) -> tuple[list[int], csr_matrix]:
        """This is similar to ``load_graph`` but includes projected cells and
        their edges.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            feat_key: Feature key. Should be same as the one that was used in the desired graph. By default, the latest
                       used feature for the given assay will be used.
            target_names: Name of target datasets to be included in the unified graph
            use_k: Number of nearest neighbour edges of each projected cell to be included. If this value is larger
                   than `save_k` parameter while running mapping for the `target_name` target then `use_k` is reset to
                   'save_k'
            target_weight: A constant uniform weight to be ascribed to each target-reference edge.

        Returns:
        """
        # TODO:  allow loading multiple targets

        if from_assay is None:
            from_assay = self._defaultAssay
        if cell_key is None:
            cell_key = self._get_latest_cell_key(from_assay)
        if feat_key is None:
            feat_key = self._get_latest_feat_key(from_assay)
        graph_loc = self._get_latest_graph_loc(from_assay, cell_key, feat_key)
        graph_group = as_zarr_group(self.zw[graph_loc], name=graph_loc)
        edges = np.asarray(
            as_zarr_array(graph_group["edges"], name="edges")[:]
        )
        weights = np.asarray(
            as_zarr_array(graph_group["weights"], name="weights")[:]
        )
        ref_n_cells = self.cells.active_index(cell_key).shape[0]
        projections = as_zarr_group(
            as_zarr_group(self.zw[from_assay], name=from_assay)["projections"],
            name="projections",
        )
        pidx = np.vstack(
            [
                np.asarray(
                    as_zarr_array(
                        as_zarr_group(projections[x], name=x)["indices"],
                        name="indices",
                    )[:, :use_k]
                )
                for x in target_names
            ]
        )
        n_cells = [ref_n_cells] + [
            as_zarr_array(
                as_zarr_group(projections[x], name=x)["indices"],
                name="indices",
            ).shape[0]
            for x in target_names
        ]
        ne = []
        nw = []
        for n, i in enumerate(pidx):
            for j in i:
                ne.append([ref_n_cells + n, j])
                # TODO: Better way to weigh the target edges
                nw.append(target_weight)
        me = np.vstack([edges, ne]).astype(int)
        mw = np.hstack([weights, nw])
        tot_cells = ref_n_cells + pidx.shape[0]
        graph = csr_matrix((mw, (me[:, 0], me[:, 1])), shape=(tot_cells, tot_cells))
        return n_cells, graph

    def _get_uni_ini_embed(
        self,
        from_assay: str,
        cell_key: str,
        feat_key: str,
        graph: csr_matrix,
        ini_embed_with: str,
        ref_n_cells: int,
    ) -> np.ndarray:
        if ini_embed_with == "kmeans":
            ini_embed = self._get_ini_embed(from_assay, cell_key, feat_key, 2)
        else:
            x = self.cells.fetch(f"{ini_embed_with}1", key=cell_key)
            y = self.cells.fetch(f"{ini_embed_with}2", key=cell_key)
            ini_embed = np.array([x, y]).T.astype(np.float32, order="C")
        targets_best_nn = np.array(np.argmax(graph, axis=1)).reshape(1, -1)[0][
            ref_n_cells:
        ]
        return np.vstack([ini_embed, ini_embed[targets_best_nn]])

    def _save_embedding(
        self,
        from_assay: str,
        cell_key: str,
        label: str,
        embedding: np.ndarray,
        n_cells: list[int],
        target_names: list[str],
    ) -> None:
        g = create_zarr_dataset(
            as_zarr_group(
                as_zarr_group(self.zw[from_assay], name=from_assay)["projections"],
                name="projections",
            ),
            label,
            (1000, 2),
            "float64",
            embedding.shape,
        )
        g[:] = embedding
        g.attrs["n_cells"] = [
            int(x) for x in n_cells
        ]  # forcing int type here otherwise json raises TypeError
        g.attrs["target_names"] = target_names
        for i in range(2):
            self.cells.insert(
                self._col_renamer(from_assay, cell_key, f"{label}{i + 1}"),
                embedding[: n_cells[0], i],
                key=cell_key,
                overwrite=True,
            )
        return None

    def run_unified_umap(
        self,
        target_names: list[str],
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        use_k: int = 3,
        target_weight: float = 0.1,
        spread: float = 2.0,
        min_dist: float = 1,
        n_epochs: int = 200,
        repulsion_strength: float = 1.0,
        initial_alpha: float = 1.0,
        negative_sample_rate: float = 5,
        random_seed: int = 4444,
        ini_embed_with: str = "kmeans",
        label: str = "unified_UMAP",
        parallel: bool = False,
        nthreads: int | None = None,
    ) -> None:
        """Calculates the UMAP embedding for graph obtained using
        ``load_unified_graph``.

        The loaded graph is processed the same way as the graph as in ``run_umap``.

        Args:
            target_names: Names of target datasets to be included in the unified UMAP.
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            feat_key: Feature key. Should be same as the one that was used in the desired graph. By default, the latest
                       used feature for the given assay will be used.
            use_k: Number of nearest neighbour edges of each projected cell to be included. If this value is larger
                   than `save_k` parameter while running mapping for the `target_name` target then `use_k` is reset to
                   'save_k'
            target_weight: A constant uniform weight to be ascribed to each target-reference edge.
            spread: Same as spread in UMAP package.  The effective scale of embedded points. In combination with
                    ``min_dist`` this determines how clustered/clumped the embedded points are.
            min_dist: Same as min_dist in UMAP package. The effective minimum distance between embedded points.
                      Smaller values will result in a more clustered/clumped embedding where nearby points on the
                      manifold are drawn closer together, while larger values will result on a more even dispersal of
                      points. The value should be set relative to the ``spread`` value, which determines the scale at
                      which embedded points will be spread out. (Default value: 1)
            n_epochs: Same as n_epochs in UMAP package. The number of training epochs to be used in optimizing the
                      low dimensional embedding. Larger values result in more accurate embeddings.
                      (Default value: 200)
            repulsion_strength: Same as repulsion_strength in UMAP package. Weighting applied to negative samples in
                                low dimensional embedding optimization. Values higher than one will result in greater
                                weight being given to negative samples. (Default value: 1.0)
            initial_alpha: Same as learning_rate in UMAP package. The initial learning rate for the embedding
                           optimization. (Default value: 1.0)
            negative_sample_rate: Same as negative_sample_rate in UMAP package. The number of negative samples to
                                  select per positive sample in the optimization process. Increasing this value will
                                  result in greater repulsive force being applied, greater optimization cost, but
                                  slightly more accuracy. (Default value: 5)
            random_seed: (Default value: 4444)
            ini_embed_with: either 'kmeans' or a column from cell metadata to be used as initial embedding coordinates
            label: base label for UMAP dimensions in the cell metadata column (Default value: 'UMAP')
            parallel: Whether to run UMAP in parallel mode. Setting value to True will use `nthreads` threads.
                      The results are not reproducible in parallel mode. (Default value: False)
            nthreads: If parallel=True then this number of threads will be used to run UMAP. By default, the `nthreads`
                      attribute of the class is used. (Default value: None)

        Returns:
            None
        """
        from ..umap import fit_transform
        from ..utils import get_log_level

        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay, cell_key, feat_key
        )
        n_cells, graph = self.load_unified_graph(
            from_assay=from_assay,
            cell_key=cell_key,
            feat_key=feat_key,
            target_names=target_names,
            use_k=use_k,
            target_weight=target_weight,
        )
        ini_embed = self._get_uni_ini_embed(
            from_assay, cell_key, feat_key, graph, ini_embed_with, n_cells[0]
        )
        if nthreads is None:
            nthreads = self.nthreads
        verbose = False
        if get_log_level() <= 20:
            verbose = True
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
            densmap_kwds={},
            parallel=parallel,
            nthreads=nthreads,
            verbose=verbose,
        )
        self._save_embedding(from_assay, cell_key, label, t, n_cells, target_names)
        return None

    def run_unified_tsne(
        self,
        target_names: list[str],
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        use_k: int = 3,
        target_weight: float = 0.5,
        lambda_scale: float = 1.0,
        max_iter: int = 500,
        early_iter: int = 200,
        alpha: int = 10,
        box_h: float = 0.7,
        temp_file_loc: str = ".",
        verbose: bool = True,
        ini_embed_with: str = "kmeans",
        label: str = "unified_tSNE",
    ) -> None:
        """Calculates the tSNE embedding for graph obtained using
        ``load_unified_graph``. The loaded graph is processed the same way as
        the graph as in ``run_tsne``.

        Args:
            target_names: Names of target datasets to be included in the unified tSNE.
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            feat_key: Feature key. Should be same as the one that was used in the desired graph. By default, the latest
                       used feature for the given assay will be used.
            use_k: Number of nearest neighbour edges of each projected cell to be included. If this value is larger
                   than `save_k` parameter while running mapping for the `target_name` target then `use_k` is reset to
                   'save_k'.
            target_weight: A constant uniform weight to be ascribed to each target-reference edge.
            lambda_scale: λ rescaling parameter. (Default value: 1.0)
            max_iter: Maximum number of iterations. (Default value: 500)
            early_iter: Number of early exaggeration iterations. (Default value: 200)
            alpha: Early exaggeration multiplier. (Default value: 10)
            box_h: Grid side length (accuracy control). Lower values might drastically slow down
                   the algorithm (Default value: 0.7)
            temp_file_loc: Location of temporary file. By default, these files will be created in the current working
                           directory. These files are deleted before the method returns.
            verbose: If True (default) then the full log from SGtSNEpi algorithm is shown.
            ini_embed_with: Initial embedding coordinates for the cells in cell_key. Should have the same number of
                            columns as tsne_dims. If not value is provided then the initial embedding is obtained using
                            `get_ini_embed`.
            label: Base label for tSNE dimensions in the cell metadata column. (Default value: 'tSNE')

        Returns:
        """
        from ..knn_utils import run_sgtsne

        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay, cell_key, feat_key
        )
        n_cells, graph = self.load_unified_graph(
            from_assay=from_assay,
            cell_key=cell_key,
            feat_key=feat_key,
            target_names=target_names,
            use_k=use_k,
            target_weight=target_weight,
        )
        ini_embed = self._get_uni_ini_embed(
            from_assay, cell_key, feat_key, graph, ini_embed_with, n_cells[0]
        )
        emb = run_sgtsne(
            graph,
            ini_embed,
            tsne_dims=2,
            max_iter=max_iter,
            early_iter=early_iter,
            alpha=alpha,
            lambda_scale=lambda_scale,
            box_h=box_h,
            temp_file_loc=temp_file_loc,
            verbose=verbose,
        )
        self._save_embedding(from_assay, cell_key, label, emb.T, n_cells, target_names)
        return None

    def plot_unified_layout(
        self,
        from_assay: str | None = None,
        layout_key: str | None = None,
        show_target_only: bool = False,
        ref_name: str = "reference",
        target_groups: list[str] | None = None,
        width: float = 6,
        height: float = 6,
        cmap: str | None = None,
        color_key: dict[Any, Any] | None = None,
        mask_color: str = "k",
        point_size: float = 10,
        ax_label_size: float = 12,
        frame_offset: float = 0.05,
        spine_width: float = 0.5,
        spine_color: str = "k",
        displayed_sides: tuple[str, ...] = ("bottom", "left"),
        legend_ondata: bool = False,
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
        savename: str | None = None,
        save_dpi: int = 300,
        ax: Any = None,
        force_ints_as_cats: bool = True,
        n_columns: int = 1,
        w_pad: float = 1,
        h_pad: float = 1,
        scatter_kwargs: dict[Any, Any] | None = None,
        shuffle_zorder: bool = True,
        show_fig: bool = True,
    ) -> Any:
        """Plots the reference and target cells in their unified space.

        This function helps to plot the reference and target cells, the coordinates for which were obtained from
        either `run_unified_tsne` or `run_unified_umap`. Since the coordinates are not saved in the cell metadata
        but rather in the projections slot of the Zarr hierarchy, this function is needed to correctly fetch the values
        for reference and target cells. Additionally, this function provides a way to colour target cells by bringing in
        external annotations for those cells.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            layout_key: Should be same as the parameter value for `label` in `run_unified_umap` or `run_unified_tsne`
                        (Default value: 'UMAP')
            show_target_only: If True then the reference cells are not shown (Default value: False)
            ref_name: A label for reference cells to be used in the legend. (Default value: 'reference')
            target_groups: Categorical values to be used to colourmap target cells. (Default value: None)
            width: Figure width (Default value: 6)
            height: Figure height (Default value: 6)
            cmap: A matplotlib colourmap to be used to colour categorical or continuous values plotted on the cells.
                  (Default value: tab20 for categorical variables and cmocean.deep for continuous variables)
            color_key: A custom colour map for cells. These can be used for categorical variables only. The keys in this
                       dictionary should be the category label as present in the `color_by` column and values should be
                       valid matplotlib colour names or hex codes of colours. (Default value: None)
            mask_color: Color to be used for masked values. This should be a valid matplotlib named colour or a hexcode
                        of a colour. (Default value: 'k')
            point_size: Size of each scatter point. This is overridden if `size_vals` is provided. Has no effect if
                        `do_shading` is True. (Default value: 10)
            ax_label_size: Font size for the x and y-axis labels. (Default value: 12)
            frame_offset: Extend the x and y-axis limits by this fraction (Default value: 0.05)
            spine_width: Line width of the displayed spines (Default value: 0.5)
            spine_color: Colour of the displayed spines.  (Default value: 'k')
            displayed_sides: Determines which figure spines are chosen. The spines to be shown can be supplied as a
                             tuple. The options are: top, bottom, left and right. (Default value: ('bottom', 'left) )
            legend_ondata: Whether to show category labels on the data (scatter points). The position of the label is
                           the centroid of the corresponding values.
                           (Default value: True)
            legend_onside: Whether to draw a legend table on the side of the figure. (Default value: True)
            legend_size: Font size of the legend text. (Default value: 12)
            legends_per_col: Number of legends to be used on each legend column. This value determines how many
                             legend columns will be drawn (Default value: 20)
            title: Title to be used for plot. (Default value: None)
            title_size: Size of each axis/subplots title (Default value: 12)
            hide_title: If True, then the title of the sublots is not shown (Default value: False)
            cbar_shrink: Shrinking factor for the width of color bar (Default value: 0.6)
            marker_scale: The relative size of legend markers compared with the originally drawn ones.
                          (Default value: 70)
            lspacing: The vertical space between the legend entries. Measured in font-size units. (Default value: 0.1)
            cspacing: The spacing between columns. Measured in font-size units. (Default value: 1)
            savename: Path where the rendered figure is to be saved. The format of the saved image depends on
                      the extension present in the parameter value. (Default value: None)
            save_dpi: DPI when saving figure (Default value: 300)
            ax: An instance of Matplotlib's Axes object. This can be used to plot the figure into an already
                created axes. (Default value: None)
            force_ints_as_cats: Force integer labels in `color_by` as categories. If False, then integer will be
                                treated as continuous variables otherwise as categories. This effects how colormaps
                                are chosen and how legends are rendered. Set this to False if you are large number of
                                unique integer entries (Default: True)
            n_columns: Number of columns in the grid
            w_pad: When plotting in multiple plots in a grid this decides the width padding between the plots.
                   If None is provided the padding will be automatically added to avoid overlap.
                   Ignored if only plotting one scatterplot.
            h_pad: When plotting in multiple plots in a grid this decides the height padding between the plots.
                   If None is provided the padding will be automatically added to avoid overlap.
                   Ignored if only plotting one scatterplot.
            scatter_kwargs: Keyword argument to be passed to matplotlib's scatter command
            shuffle_zorder: Whether to shuffle the plot order of data points in the figure. (Default value: True)
            show_fig: Whether to render the figure and display it using plt.show() (Default value: True)

        Returns:
            None
        """

        from ..plots import plot_scatter

        if from_assay is None:
            from_assay = self._defaultAssay
        if layout_key is None:
            raise ValueError(
                "ERROR: Please provide a value for the `layout_key` parameter. This should be same as "
                "that for either `run_unified_umap` or `run_unified_tsne`. Please see the default values "
                "for `label` parameter in those functions if unsure."
            )
        projections = as_zarr_group(
            as_zarr_group(self.zw[from_assay], name=from_assay)["projections"],
            name="projections",
        )
        layout = as_zarr_array(projections[layout_key], name=layout_key)
        t = np.asarray(layout[:])
        attrs = dict(layout.attrs)
        t_names = cast(list[str], attrs["target_names"])
        n_cells_attr = cast(list[int], attrs["n_cells"])
        ref_n_cells = n_cells_attr[0]
        t_n_cells = n_cells_attr[1:]
        x = t[:, 0]
        y = t[:, 1]
        df = pd.DataFrame({f"{layout_key}1": x, f"{layout_key}2": y})
        plot_color_key = color_key
        plot_mask_values: list[Any] | None
        mask_name: str
        group_col: NDArray[Any]
        if target_groups is None:
            if plot_color_key is not None:
                temp_raise_error = False
                if ref_name not in plot_color_key:
                    temp_raise_error = True
                for i in t_names:
                    if i not in plot_color_key:
                        temp_raise_error = True
                if temp_raise_error:
                    temp = " ".join(t_names)
                    raise KeyError(
                        f"ERROR: `color_key` must contain these keys: '{ref_name}' and '{temp}'"
                    )
            else:
                import seaborn as sns

                temp_cmap = sns.color_palette("hls", n_colors=len(t_names) + 1).as_hex()
                plot_color_key = {k: v for k, v in zip(t_names, temp_cmap[1:])}
                plot_color_key[ref_name] = temp_cmap[0]
            group_labels: list[str] = []
            for i, j in zip(t_names, t_n_cells):
                group_labels.extend([i for _ in range(j)])
            group_col = np.array(group_labels, dtype=object)
            plot_mask_values = None
            mask_name = "NA"
        else:
            group_labels = list(target_groups)
            if len(group_labels) == len(t_names):
                flattened: list[str] = []
                for entry in group_labels:
                    flattened.extend(list(entry))
                group_labels = flattened
            plot_color_key = None
            plot_mask_values = [ref_name]
            mask_name = ref_name
            group_col = np.array(group_labels, dtype=object)
        if len(group_col) != sum(t_n_cells):
            raise ValueError(
                "ERROR: Number of values in `target_groups` should be same as no. of target cells"
            )
        # Turning array to object forces np.NaN to 'nan'
        if np.any(group_col == "nan"):
            raise ValueError("ERROR: `target_groups` cannot contain nan values")
        df["vc"] = np.hstack(
            [[ref_name for _ in range(ref_n_cells)], group_col]
        ).astype(object)
        if show_target_only:
            df = df[ref_n_cells:]
        if shuffle_zorder:
            df = df.sample(frac=1)
        plot_title = title if isinstance(title, str) else None
        return plot_scatter(
            [df],
            ax,
            width,
            height,
            mask_color,
            cmap,
            plot_color_key or {},
            plot_mask_values or [],
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
            plot_title,
            title_size,
            hide_title,
            cbar_shrink,
            marker_scale,
            lspacing,
            cspacing,
            savename or "",
            save_dpi,
            force_ints_as_cats,
            n_columns,
            w_pad,
            h_pad,
            show_fig,
            scatter_kwargs or {},
        )

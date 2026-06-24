"""
Methods and classes for merging datasets

"""

import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import zarr
from scipy.sparse import coo_matrix

from .chunked import ChunkedArray
from .assay import Assay
from .datastore.datastore import DataStore
from .metadata import MetaData
from .utils import (
    ZARRLOC,
    controlled_compute,
    load_zarr,
    logger,
    permute_into_chunks,
    tqdmbar,
)
from .writers import create_zarr_count_assay, create_zarr_obj_array, create_zarr_dataset

__all__ = [
    "DatasetMerge",
    "AssayMerge",
    "ZarrMerge",
]


# Creating a dummy Assay object
class DummyAssay:
    """
    A dummy assay object to be used in the AssayMerge class when an assay is missing in a dataset.
    """

    def __init__(
        self,
        ds: DataStore,
        counts: ChunkedArray,
        feats: MetaData,
        name: str,
    ):
        self.rawData = counts
        self.feats = feats
        self.cells = ds.cells
        self.name = name


class AssayMerge:
    # class ZarrMerge is renamed to AssayMerge for better understanding
    """Merge multiple Zarr files into a single Zarr file.

    Args:
        zarr_path: Name of the new, merged Zarr file with path.
        assays: List of assay objects to be merged. For example, [ds1.RNA, ds2.RNA].
        names: Names of each of the assay objects in the `assays` parameter. They should be in the same order as in
               `assays` parameter.
        merge_assay_name: Name of assay in the merged Zarr file. For example, for scRNA-Seq it could be simply,
                          'RNA'.
        chunk_size: Tuple of cell and feature chunk size. (Default value: (1000, 1000)).
        dtype: Dtype of the raw values in the assay. Dtype is automatically inferred from the provided assays. If
               assays have different dtypes then a float type is used.
        overwrite: If True, then overwrites previously created assay in the Zarr file. (Default value: False).
        prepend_text: This text is pre-appended to each column name (Default value: 'orig').
        reset_cell_filter: If True, then the cell filtering information is removed, i.e. even the filtered out cells
                           are set as True as in the 'I' column. To keep the filtering information set the value for
                           this parameter to False. (Default value: True)
        seed: Seed for randomization of rows in the assays.
        feat_name_ids_same: If True, then feature names and feature ids are same in the assays. (Default value: False)

    Attributes:
        assays: List of assay objects to be merged. For example, [ds1.RNA, ds2.RNA].
        names: Names of each assay objects in the `assays` parameter.
        mergedCells:
        nCells: Number of cells in dataset.
        featCollection:
        mergedFeats:
        nFeats: Number of features in the dataset.
        featOrder:
        z: The merged Zarr file.
        assayGroup:
    """

    def __init__(
        self,
        zarr_path: ZARRLOC,
        assays: list[Assay],
        names: list[str],
        merge_assay_name: str,
        in_workspaces: list[str] | None = None,
        out_workspace: str | None = None,
        chunk_size=(1000, 1000),
        dtype: str | None = None,
        overwrite: bool = False,
        prepend_text: str | None = "orig",
        reset_cell_filter: bool = True,
        seed: int | None = 42,
    ):
        self.assays = assays
        self.names = names
        self.inWorkspaces = in_workspaces
        self.outWorkspace = out_workspace
        self.merge_assay_name = merge_assay_name
        self.chunk_size = chunk_size
        (
            self.permutations_rows,
            self.permutations_rows_offset,
            self.coordinates_permutations,
        ) = self.perform_randomization_rows(seed)
        self.mergedCells: pd.DataFrame = self._merge_cell_table(
            reset_cell_filter, prepend_text
        )
        self.nCells: int = self.mergedCells.shape[0]
        self.featCollection: list[dict[str, str]] = self._get_feat_ids(assays)
        self.feat_name_ids_same: bool = self.check_feat_ids(self.featCollection)

        if self.feat_name_ids_same is True:
            self.feat_suffix: dict[int, int] = self.get_feat_suffix()
            self.featCollection = self.update_feat_ids()
            self.featCollection_map: list[dict[str, str]] = (
                self.update_feat_ids_for_map()
            )
        else:
            self.featCollection_map: list[dict[str, str]] = self.featCollection.copy()

        self.mergedFeats: pd.DataFrame = self._merge_order_feats(self.featCollection)
        self.mergedFeats_map: pd.DataFrame = self._merge_order_feats(
            self.featCollection_map
        )
        self.nFeats: int = self.mergedFeats_map.shape[0]
        self.featOrder: list[np.ndarray] = self._ref_order_feat_idx()

        if self.feat_name_ids_same is True:
            self.featOrder_map: list[np.ndarray] = self._ref_order_feat_idx_map()
        else:
            self.featOrder_map: list[np.ndarray] = self.featOrder.copy()

        self.cellOrder: dict[int, dict[int, np.ndarray]] = self._ref_order_cell_idx()
        self.z: zarr.Group = self._use_existing_zarr(
            zarr_path, merge_assay_name, overwrite
        )
        self._ini_cell_data(overwrite)
        if dtype is None:
            if len(set([str(x.rawData.dtype) for x in self.assays])) == 1:
                dtype = str(self.assays[0].rawData.dtype)
            else:
                dtype = "float"

        self.assayGroup = create_zarr_count_assay(
            z=self.z,
            assay_name=merge_assay_name,
            workspace=self.outWorkspace,
            chunk_size=chunk_size,
            n_cells=self.nCells,
            feat_ids=np.array(self.mergedFeats_map["ids"]),
            feat_names=np.array(self.mergedFeats_map["names"]),
            dtype=dtype,
        )

    def perform_randomization_rows(
        self, seed: int | None = 42
    ) -> tuple[
        dict[int, dict[int, np.ndarray]], dict[int, dict[int, np.ndarray]], np.ndarray
    ]:
        """
        Perform randomization of rows in the assays.
        Args:
            seed: Seed for randomization
        Returns:
        """
        rng = np.random.default_rng(seed=seed)
        chunkSize = np.array([x.rawData.chunksize[0] for x in self.assays])
        nCells = np.array([x.rawData.shape[0] for x in self.assays])
        permutations = {
            i: permute_into_chunks(nCells[i], chunkSize[i])
            for i in range(len(self.assays))
        }  # Randomize the rows in chunks

        # Create a dictionary of arrays. This is the same data in `permutations` but in a different format. We index the arrays by the chunk number.
        # Example:
        # permutation = {0: [array([2, 0, 1]), array([3, 4, 5]), array([8, 7, 6]), array([9])], 1: [array([2, 0, 1]), array([3, 4, 5]), array([8, 7, 6]), array([9])]}
        # permutations_rows = {0: {0: array([2, 0, 1]), 1: array([3, 4, 5]), 2: array([8, 7, 6]), 3: array([9])}, 1: {0: array([2, 0, 1]), 1: array([3, 4, 5]), 2: array([8, 7, 6]), 3: array([9])}}
        permutations_rows = {}
        for key, arrays in permutations.items():
            in_dict = {i: x for i, x in enumerate(arrays)}
            permutations_rows[key] = in_dict

        # Set the offset for each chunk. Offset calculated by adding the number of cells in the previous chunks. This will be helpful when we merge the cells metadata in the end.
        # Example:
        # {0: {0: array([2, 0, 1]), 1: array([3, 4, 5]), 2: array([8, 7, 6]), 3: array([9])}, 1: {0: array([12, 10, 11]), 1: array([13, 14, 15]), 2: array([18, 17, 16]), 3: array([19])}}
        permutations_rows_offset = {}
        offset = 0
        for key, val_dict in permutations_rows.items():
            in__dict: dict[int, np.ndarray] = {}
            for in_key, arrs in val_dict.items():
                in__dict[in_key] = arrs + offset
            permutations_rows_offset[key] = in__dict
            offset += nCells[key]

        # Set the random order in which the rows will be merged. The last chunk of each assay is appended at the end of the list to account for potential incomplete chunks.
        # Example:
        # coordinates_permutations = [[0, 0], [0, 1], [1, 2], [0, 2], [1, 1], [1, 0], [0, 3], [1, 3]]
        # Here [0, 0] means the first chunk of the first assay, [0, 1] means the second chunk of the first assay, [1, 2] means the third chunk of the second assay, and so on will be the order in which the rows will be merged.
        coordinates = []
        extra = []
        for i in range(len(self.assays)):
            for j in range(len(permutations[i])):
                if j == len(permutations[i]) - 1:  # if j is last, append extra
                    extra.append([i, j])
                    continue
                coordinates.append([i, j])
        coordinates_permutations = rng.permutation(
            coordinates
        )  # Randomize the order of the coordinates
        if len(coordinates_permutations) > 0:
            coordinates_permutations = np.concatenate(
                [coordinates_permutations, extra], axis=0
            )
        else:
            coordinates_permutations = np.array(extra)

        try:
            assert permutations_rows_offset[0][0].min() == 0
        except AssertionError:
            raise AssertionError(
                "ERROR: Randomization of rows failed. The first row should be at 0.",
                "Please report this issue.",
            )
        try:
            assert (
                permutations_rows_offset[list(permutations_rows_offset.keys())[-1]][
                    list(
                        permutations_rows_offset[
                            list(permutations_rows_offset.keys())[-1]
                        ].keys()
                    )[-1]
                ].max()
                == nCells.sum() - 1
            )
        except AssertionError:
            raise AssertionError(
                "ERROR: Randomization of rows failed. The last row should be at the end of the dataset.",
                "Please report this issue.",
            )
        return permutations_rows, permutations_rows_offset, coordinates_permutations

    def _ref_order_cell_idx(self) -> dict[int, dict[int, np.ndarray]]:
        """
        Calculate the order of the cells in the merged assay.
        """
        # We calculate the order of the cells in the merged assay by using the permutations_rows and coordinates_permutations. This is essentially the one-to-one mapping of the cells in the assays to the cells in the merged assay.
        # Example:
        # cellOrder = {0: {0: array([0, 1, 2]), 1: array([3, 4, 5]), 2: array([ 9, 10, 11]), 3: array([18])}, 1: {0: array([15, 16, 17]), 1: array([12, 13, 14]), 2: array([6, 7, 8]), 3: array([19])}}
        # Here we see that the cells [2, 0, 1] from the first chunk of the first assay are mapped to [0, 1, 2] in the merged assay. Similarly, the cells [2, 0, 1] from the first chunk of the second assay are mapped to [15, 16, 17] in the merged assay.
        new_cells = {}
        for i in range(len(self.assays)):
            in_dict: dict[int, np.ndarray] = {}
            for j in range(len(self.permutations_rows[i])):
                in_dict[j] = np.array([])
            new_cells[i] = in_dict
        offset = 0
        for i, (x, y) in enumerate(self.coordinates_permutations):
            arr = self.permutations_rows[x][y]
            arr = np.array(range(len(arr)))
            arr = arr + offset
            new_cells[x][y] = arr
            offset = arr.max() + 1
        return new_cells

    def _merge_cell_table(
        self, reset: bool, prepend_text: str | None = None
    ) -> pd.DataFrame:
        """Merges the cell metadata table for each sample.

        Args:
            reset: whether to remove filtering information
            prepend_text: string to add as prefix for each cell column

        Returns:
        """
        if len(self.assays) != len(set(self.names)):
            raise ValueError(
                "ERROR: A unique name should be provided for each of the assay"
            )
        if prepend_text == "":
            prepend_text = None
        ret_val = []
        for assay, name in zip(self.assays, self.names):
            a = assay.cells.to_pandas_dataframe(assay.cells.columns)
            a["ids"] = np.array([f"{name}__{x}" for x in a["ids"]])
            for i in list(a.columns):
                if i not in ["ids", "I", "names"] and prepend_text is not None:
                    a[f"{prepend_text}_{i}"] = assay.cells.fetch_all(i)
                    a = a.drop(columns=[i])
            if reset:
                a["I"] = np.ones(len(a["ids"])).astype(bool)
            ret_val.append(a)

        # Here we merge the cell metadata tables for each sample. We simply concatenate the tables and reset the index.
        ret_val_df = pd.concat(ret_val, axis=0).reset_index(drop=True)
        # Now we use the offsets stored in permutations_rows_offset along with the coordinates_permutations to reorder the cells in the merged assay. The offsets are used to bring the cells in the same order as the rows in the merged assay.
        compiled_idx = [
            self.permutations_rows_offset[i][j]
            for i, j in self.coordinates_permutations
        ]
        compiled_idx = np.concatenate(compiled_idx)
        # Index the merged cell metadata table with the compiled_idx to get the final randomized merged cell metadata table.
        ret_val_df = ret_val_df.iloc[compiled_idx]
        if sum([x.cells.N for x in self.assays]) != ret_val_df.shape[0]:
            raise AssertionError(
                "Unexpected number of cells in the merged table. This is unexpected, "
                " please report this bug"
            )
        return ret_val_df

    @staticmethod
    def _get_feat_ids(assays) -> list[dict[str, str]]:
        """Fetches ID->names mapping of features from each assay.

        Args:
            assays: List of Assay objects

        Returns:
            A list of dictionaries. Each dictionary is a id to name
            mapping for each feature in the corresponding assay
        """
        ret_val = []
        for i in assays:
            df = i.feats.to_pandas_dataframe(["names", "ids"])
            ret_val.append(dict(zip(df["ids"].to_numpy(), df["names"].to_numpy())))
        return ret_val

    def check_feat_ids(self, featCollection: list[dict[str, str]]) -> bool:
        """
        Check if feature names and feature ids are different in the assays.
        """
        isSame = False
        for i, dict_ in enumerate(featCollection):
            keys = np.array(list(dict_.keys()))
            values = np.array(list(dict_.values()))
            if np.equal(keys, values).all():
                logger.info(
                    f"Encountered same feature names and ids for feature {self.assays[i].name} in dataset {self.names[i]}. The feature ids will be updated with feature names."
                )
                isSame = True
                break
        return isSame

    def get_feat_suffix(self) -> dict[int, int]:
        """
        Get the suffix of the feature ids.
        """
        feat_suffix = {}
        for i, dict_ in enumerate(self.featCollection):
            keys = np.array(list(dict_.keys()))
            ends_0 = np.array([x.endswith("_0") for x in keys]).sum()
            ends_1 = np.array([x.endswith("_1") for x in keys]).sum()
            ends_2 = np.array([x.endswith("_2") for x in keys]).sum()
            if ends_0 > 0:
                feat_suffix[i] = 0
            elif ends_1 > 0:
                feat_suffix[i] = 1
            elif ends_0 > 0 and ends_1 > 0:
                feat_suffix[i] = 0
            elif ends_2 > 0:
                raise ValueError(
                    "Feature Numbering starts with 2, this is erroneous. Kindly check the data"
                )
            else:
                feat_suffix[i] = -1
        return feat_suffix

    def update_feat_ids(self) -> list[dict[str, str]]:
        """
        Update the feature ids in case of same feature names and ids.

        Returns:
            `list[dict[str, str]]`: List of dictionaries containing the updated feature ids for the merged assay.

        This function updates the feature ids for the merged assay in case the feature names and ids are the same in the assays.
        This function will generate a new feature id and name for the duplicate feature names and ids.
        We will append a numeric suffix to the feature ids to make them unique. We use this later to map multiple feature ids to a single feature id.
        """
        pattern = re.compile(r"_\d+$")
        # feat_suffix = self.get_feat_suffix()
        vals = np.array(list(self.feat_suffix.values()))
        vals = vals[vals > -1]
        min_val = vals.min() if len(vals) > 0 else 0
        new_featCollection = []
        for i, dict_ in enumerate(self.featCollection):
            in_dict = {}
            counter = Counter(dict_.values())
            if self.feat_suffix[i] == -1:
                sum_counter = {x: 0 for x in np.unique(list(dict_.values()))}
                # Update all values from 'val' to 'val_{min}'
                for _, val in dict_.items():
                    if counter[val] == 1:  # Unique value
                        in_dict[val] = val
                    else:  # Multiple values -- update
                        updated_val = f"{val}_{min_val+sum_counter[val]}"
                        in_dict[updated_val] = updated_val
                    sum_counter[val] += 1
            else:
                for _, val in dict_.items():
                    # check if the value ends with a number
                    if pattern.search(val):
                        num = int(val.split("_")[-1])
                        # replace the number with min_val
                        updated_val = pattern.sub(
                            f"_{min_val-self.feat_suffix[i]+num}", val
                        )
                        in_dict[updated_val] = updated_val
                    else:
                        updated_val = f"{val}"  # _{min_val}"
                        in_dict[updated_val] = updated_val
            new_featCollection.append(in_dict)
        return new_featCollection

    def update_feat_ids_for_map(self) -> list[dict[str, str]]:
        """
        Get the updated feature ids mapping for the merged assay in case of same feature names and ids.

        Returns:
            `list[dict[str, str]]`: List of dictionaries containing the updated feature ids for the merged assay.

        This function updates the feature ids for the merged assay in case the feature names and ids are the same in the assays.
        This function will remove the numeric suffix from the feature ids and update them with the feature names.
        """
        pattern = re.compile(r"_\d+$")
        new_featCollection = []
        for dict_ in self.featCollection:
            in_dict = {}
            for x in dict_.values():
                # check if the value ends with a number
                if pattern.search(x):
                    val = x.split("_")[:-1]
                    val = "_".join(val)
                    if val not in in_dict:
                        in_dict[val] = val
                else:
                    in_dict[x] = x
            new_featCollection.append(in_dict)
        return new_featCollection

    def _merge_order_feats(self, FeatCollection) -> pd.DataFrame:
        """Merge features from all the assays and determine their order.

        Returns:
        """
        union_set = {}
        for ids in FeatCollection:
            for i in ids:
                if i not in union_set:
                    union_set[i] = ids[i]
        ret_val = pd.DataFrame(
            {
                "idx": list(range(len(union_set))),
                "names": list(union_set.values()),
                "ids": list(union_set.keys()),
            }
        )

        r = ret_val.shape[0] / sum([x.feats.N for x in self.assays])
        if r == 1:
            raise ValueError(
                "No overlapping features found! Will not merge the files. Please check the features ids "
                " are comparable across the assays"
            )
        if r > 0.9:
            logger.warning("The number overlapping features is very low.")
        return ret_val

    def _ref_order_feat_idx(self) -> list[np.ndarray]:
        ret_val = []
        for ids in self.featCollection:
            ordered_ids = pd.DataFrame({"ids": list(ids.keys())})
            vals = ordered_ids.merge(self.mergedFeats, on="ids", how="left")[
                "idx"
            ].to_numpy()
            ret_val.append(np.array(vals))
        return ret_val

    def _ref_order_feat_idx_map(self) -> list[np.ndarray]:
        """
        Get the order of the features in the merged assay.

        Returns:
            `list[np.ndarray]`: List of numpy arrays containing the order of the features in the merged assay.

        This function returns the order of the features in the merged assay. The order is determined by the feature
        """
        featorder = []
        name_to_idx_dict = dict(
            zip(self.mergedFeats_map["names"], self.mergedFeats_map["idx"])
        )
        pattern = re.compile(r"_\d+$")
        for dict_ in self.featCollection:
            vals = []
            values_list = []
            for val in dict_.values():
                if pattern.search(val):
                    val = "_".join(val.split("_")[:-1])  # Remove the numeric suffix.
                values_list.append(val)
            vals = [name_to_idx_dict[name] for name in values_list]
            featorder.append(np.array(vals))
        return featorder

    def _use_existing_zarr(
        self, zarr_loc: ZARRLOC, merge_assay_name, overwrite
    ) -> zarr.Group:
        if self.outWorkspace is None:
            cell_slot = "cellData"
            assay_slot = merge_assay_name
        else:
            cell_slot = f"{self.outWorkspace}/cellData"
            assay_slot = f"{self.outWorkspace}/merge_assay_name"

        try:
            z = load_zarr(zarr_loc, mode="r")
            if cell_slot not in z:
                raise ValueError(
                    f"ERROR: Zarr file exists but seems corrupted. Either delete the "  # noqa: F541
                    "existing file or choose another path"
                )
            if assay_slot in z:
                if overwrite is False:
                    raise ValueError(
                        f"ERROR: Zarr file already contains {merge_assay_name} assay. Choose "
                        "a different zarr path or a different assay name. Otherwise set overwrite to True"
                    )
            try:
                if not all(
                    z[cell_slot]["ids"][:]
                    == np.array(np.array(self.mergedCells["ids"]))  # type: ignore
                ):
                    raise ValueError(
                        f"ERROR: order of cells does not match the one in existing file"  # noqa: F541
                    )
            except KeyError:
                raise ValueError(
                    f"ERROR: 'cell data seems corrupted. Either delete the "  # noqa: F541
                    "existing file or choose another path"
                )
            return load_zarr(zarr_loc, mode="r+")
        except (ValueError, FileNotFoundError):
            # So no zarr file with same name exists. Check if a non zarr folder with the same name exists
            if isinstance(zarr_loc, str) and os.path.exists(zarr_loc):
                raise ValueError(
                    f"ERROR: Directory/file with name `{zarr_loc}`exists. "
                    f"Either delete it or use another name"
                )
            # creating a new zarr file
            return load_zarr(zarr_loc, mode="w")

    def _ini_cell_data(self, overwrite) -> None:
        """Save cell attributes to Zarr.

        Returns:
            None
        """
        if self.outWorkspace is None:
            cell_slot = "cellData"
        else:
            cell_slot = f"{self.outWorkspace}/cellData"

        if (cell_slot in self.z and overwrite is True) or cell_slot not in self.z:
            g = self.z.create_group(cell_slot, overwrite=True)
            for i in self.mergedCells.columns:
                vals = np.array(self.mergedCells[i])
                create_zarr_obj_array(g, str(i), vals, vals.dtype, overwrite=True)
        else:
            logger.info(
                f"cellData already exists so skipping _ini_cell_data"  # noqa: F541
            )

    def _dask_to_coo(
        self, d_arr, order: np.ndarray, order_map: np.ndarray, n_threads: int
    ) -> coo_matrix:
        """
        Convert a chunked array block to a sparse COO matrix.
        Args:
            d_arr: Chunked array block to be converted
            order: Original feature indices
            order_map: Consolidated feature indices
            n_threads: Number of threads to use for computation
        Returns:
            Sparse COO matrix

        This function takes a chunked array block and converts it to a sparse COO matrix.
        The `order` is the original feature indices and `order_map` is the consolidated feature indices
        i.e. the indices of the features in the merged assay. If the `order` and `order_map` are the same,
        then the function will directly convert the block to a COO matrix. If they are different,
        then the function will consolidate the data from the block to the COO matrix using the `order_map`.
        For multiple indices mapping to the same consolidated index, the data is summed up.
        """
        if np.array_equal(order, order_map):
            return coo_matrix(controlled_compute(d_arr, n_threads))

        computed_data = controlled_compute(d_arr, n_threads)
        consolidation_map = {orig: cons for orig, cons in zip(order, order_map)}
        rows, cols, data = [], [], []
        for i, col_data in enumerate(computed_data.T):
            consolidated_idx = consolidation_map[order[i]]
            nz = np.flatnonzero(col_data)
            if nz.size == 0:
                continue
            rows.extend(nz)
            cols.extend([consolidated_idx] * nz.size)
            data.extend(col_data[nz])
        return coo_matrix(
            (data, (rows, cols)), shape=(d_arr.shape[0], self.nFeats)
        )

    def dump(self, nthreads=4):
        """Copy the values from individual assays to the merged assay.

        Args:
            nthreads: Number of compute threads to use. (Default value: 2)

        Returns:
        """
        counter = 0
        for i, (assay, feat_order, feat_order_map) in enumerate(
            zip(self.assays, self.featOrder, self.featOrder_map)
        ):
            for j, block in tqdmbar(
                enumerate(assay.rawData.blocks),
                total=assay.rawData.numblocks[0],
                desc=f"Writing data from assay {i+1}/{len(self.assays)} to merged file",
            ):
                # Perform the inter-chunk permutation of the rows
                perm_order = self.permutations_rows[i][j]
                perm_order = perm_order - perm_order.min()
                block = block[perm_order, :]
                a = self._dask_to_coo(block, feat_order, feat_order_map, nthreads)
                # Here we use the one-to-one mapping of the chunks in the assays to the chunks in the merged assay to bring the data in the same order.
                row_idx = self.cellOrder[i][j]
                self.assayGroup.set_coordinate_selection(
                    (a.row + row_idx.min(), a.col), a.data.astype(self.assayGroup.dtype)
                )
                counter += a.shape[0]
        try:
            assert counter == self.nCells
        except AssertionError:
            raise AssertionError(
                "ERROR: Mismatch in number of cells in the merged assay. Please report this issue."
            )


# Alias for ZarrMerge
class ZarrMerge(AssayMerge):
    """
    Alias for AssayMerge for backward compatibility.
    """

    def __init__(self, *args, **kwargs):
        logger.warning(
            "The 'ZarrMerge' class is deprecated and will be removed in a future release. Please use 'AssayMerge' instead."
        )
        super().__init__(*args, **kwargs)


class DatasetMerge:
    """
    Merge multiple datastores, handling different assay types and generating missing assays on the fly.

    Args:
        datasets: List of DataStore objects to be merged.
        zarr_path: Name of the new, merged Zarr file with path.
        names: Names of each of the dataset objects in the `datasets` parameter. They should be in the same order as in
               `datasets` parameter.
        in_workspaces: List of workspaces to be merged. If None, all workspaces are merged.
        out_workspace: Name of the workspace in the merged Zarr file. If None, the name of the first workspace is used.
        chunk_size: Tuple of cell and feature chunk size. (Default value: (1000, 1000)).
        dtype: Dtype of the raw values in the assay. Dtype is automatically inferred from the provided assays. If
               assays have different dtypes then a float type is used.
        overwrite: If True, then overwrites previously created assay in the Zarr file. (Default value: False).
        prepend_text: This text is pre-appended to each column name (Default value: 'orig').
        reset_cell_filter: If True, then the cell filtering information is removed, i.e. even the filtered out cells
                           are set as True as in the 'I' column. To keep the filtering information set the value for
                           this parameter to False. (Default value: True)
        seed: Seed for randomization of rows in the assays.

    Example:
        >>> # Assuming ds1, ds2 and ds3 are DataStore objects
        >>> # ds1 has RNA and ADT assays. ds2 has RNA assay. ds3 has ADT assay.
        >>> # Merge RNA and ADT assays from all the datastores
        >>> merge = DatasetMerge(
        >>>     datasets=[ds1, ds2, ds3],
        >>>     zarr_path="merged.zarr",
        >>>     names=["ds1", "ds2", "ds3"],
        >>>     overwrite = True
        >>> )
        >>> merge.dump()
        >>> # The merged.zarr file will have RNA and ADT assays from all the datastores
    """

    def __init__(
        self,
        datasets: list[DataStore],
        zarr_path: ZARRLOC,
        names: list[str],
        in_workspaces: list[str] | None = None,
        out_workspace: str | None = None,
        chunk_size=(1000, 1000),
        dtype: str | None = None,
        overwrite: bool = False,
        prepend_text: str | None = "orig",
        reset_cell_filter: bool = True,
        seed: int | None = 42,
    ):
        self.datasets = datasets
        self.names = names
        self.zarr_path = zarr_path
        self.in_workspaces = in_workspaces
        self.out_workspace = out_workspace
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.overwrite = overwrite
        self.prepend_text = prepend_text
        self.reset_cell_filter = reset_cell_filter
        self.seed = seed
        self.unique_assays = self.get_unique_assays()
        self.n_unique_assays = len(self.unique_assays)
        self.merge_generators = self.create_merge_generators()

    def get_unique_assays(self) -> list[str]:
        """
        Get unique assays from both datasets
        """
        unique_assays = set()
        for ds in self.datasets:
            unique_assays.update(ds.assay_names)
        return list(unique_assays)

    def create_merge_generators(self) -> list[AssayMerge]:
        """
        Create AssayMerge objects for each unique assay
        """
        gens = []
        for assay in self.unique_assays:
            assay_list = []
            for ds in self.datasets:
                if assay in ds.assay_names:
                    assay_list.append(ds.get_assay(assay))
                else:
                    # Generate a dummy assay on the fly
                    dummy_assay: DummyAssay = self.generate_dummy_assay(ds, assay)
                    assay_list.append(dummy_assay)
            gens.append(
                AssayMerge(
                    zarr_path=self.zarr_path,
                    assays=assay_list,
                    names=self.names,
                    merge_assay_name=assay,
                    in_workspaces=self.in_workspaces,
                    out_workspace=self.out_workspace,
                    chunk_size=self.chunk_size,
                    dtype=self.dtype,
                    overwrite=self.overwrite,
                    prepend_text=self.prepend_text,
                    reset_cell_filter=self.reset_cell_filter,
                    seed=self.seed,
                )
            )
        return gens

    def generate_dummy_assay(self, ds: DataStore, assay_name: str) -> DummyAssay:
        """
        Generate a dummy assay for a datastore that doesn't have the specified assay
        """
        # Find a datastore that has this assay to get feature information
        reference_ds = next(
            ds_ for ds_ in self.datasets if assay_name in ds_.assay_names
        )
        reference_assay = reference_ds.get_assay(assay_name)

        reference_chunk = [
            ds.get_assay(ref).rawData.chunksize[0] for ref in ds.assay_names
        ]
        # check if entries in reference_chunk are the same
        if not all(x == reference_chunk[0] for x in reference_chunk):
            rowChunkShape = reference_chunk[0]
        else:
            rowChunkShape = max(reference_chunk)
        colChunk = reference_assay.rawData.chunksize[1]
        chunkShape = (rowChunkShape, colChunk)

        # Create a dummy assay with zero counts and matching features
        dummy_shape = (ds.cells.N, reference_assay.feats.N)
        mem_store = zarr.storage.MemoryStore()
        mem_group = zarr.open_group(store=mem_store, mode="w")
        dummy_array = create_zarr_dataset(
            mem_group,
            "counts",
            chunkShape,
            reference_assay.rawData.dtype,
            dummy_shape,
        )
        dummy_counts = ChunkedArray(dummy_array, nthreads=1)
        dummy_assay = DummyAssay(
            ds, dummy_counts, reference_assay.feats, reference_assay.name
        )
        logger.info(f"Generated dummy {assay_name} assay for datastore {ds}")
        return dummy_assay

    def dump(self, nthreads=4) -> None:
        """
        Dump the merged data to the zarr file
        """
        for gen in self.merge_generators:
            logger.info(f"Dumping {gen.merge_assay_name}")
            gen.dump(nthreads)
        logger.info("Merging complete")
        return None

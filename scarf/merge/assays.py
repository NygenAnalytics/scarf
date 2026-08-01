import os
import re
import sys
from collections import Counter
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
import scarf
import zarr
from scipy.sparse import coo_matrix

from ..storage.types import as_zarr_array, as_zarr_group
from ..assay import Assay
from ..matrix import Block, ChunkedArray
from ..metadata import MetaData
from ..storage.arrays import create_zarr_obj_array
from ..storage.budget import admitted_worker_count, resolve_budget
from ..storage.layout import array_shard_rows
from ..storage.profiles import (
    StorageProfile,
    ZarrLocation,
    is_local_zarr_path,
    resolve_storage_profile,
)
from ..storage.schema import create_zarr_count_assay, validate_assay_name
from ..storage.sharding import (
    accumulate_sparse_to_shards,
    sparse_producer_peak_bytes,
)
from ..storage.stores import load_zarr as load_zarr
from ..utils.arrays import canonicalize_sparse, permute_into_chunks
from ..utils.compute import controlled_compute as controlled_compute
from ..utils.logging import logger
from ..utils.progress import iter_progress


# Creating a dummy Assay object
class DummyAssay:
    """
    A dummy assay object to be used in the AssayMerge class when an assay is missing in a dataset.
    """

    def __init__(
        self,
        ds: "scarf.DataStore",
        counts: ChunkedArray,
        feats: MetaData,
        name: str,
    ):
        self.rawData = counts
        self.feats = feats
        self.cells = ds.cells
        self.name = name


type MergeAssay = Assay | DummyAssay

type _RowPlan = tuple[
    dict[int, dict[int, np.ndarray]],
    dict[int, dict[int, np.ndarray]],
    np.ndarray,
]


def _dtype_for_integer_sum(dtype: np.dtype[Any], copies: int) -> np.dtype[Any]:
    if copies <= 1 or dtype.kind not in "biu":
        return dtype
    if dtype.kind in "bu":
        lower = 0
        upper = (1 if dtype.kind == "b" else np.iinfo(dtype).max) * copies
        for candidate in (np.uint8, np.uint16, np.uint32, np.uint64):
            candidate_info = np.iinfo(candidate)
            if lower >= candidate_info.min and upper <= candidate_info.max:
                return np.dtype(candidate)
    else:
        info = np.iinfo(dtype)
        lower = info.min * copies
        upper = info.max * copies
        for signed_candidate in (np.int8, np.int16, np.int32, np.int64):
            candidate_info = np.iinfo(signed_candidate)
            if lower >= candidate_info.min and upper <= candidate_info.max:
                return np.dtype(signed_candidate)
    return np.dtype(np.uint64 if dtype.kind in "bu" else np.int64)


class AssayMerge:
    """Merge multiple Zarr files into a single Zarr file.

    Args:
        zarr_path: Name of the new, merged Zarr file with path.
        assays: List of assay objects to be merged. For example, [ds1.RNA, ds2.RNA].
        names: Names of each of the assay objects in the `assays` parameter. They should be in the same order as in
               `assays` parameter.
        merge_assay_name: Name of assay in the merged Zarr file. For example, for scRNA-Seq it could be simply,
                          'RNA'.
        in_workspaces: Source workspace per assay (None uses each assay's default layout).
        out_workspace: Target workspace name in the merged Zarr file.
        dtype: Dtype of the raw values in the assay. Dtype is automatically inferred from the provided assays. If
               assays have different dtypes then a float type is used.
        overwrite: If True, then overwrites previously created assay in the Zarr file. (Default value: False).
        prepend_text: This text is pre-appended to each column name (Default value: 'orig').
        reset_cell_filter: If True, then the cell filtering information is removed, i.e. even the filtered out cells
                           are set as True as in the 'I' column. To keep the filtering information set the value for
                           this parameter to False. (Default value: True)
        seed: Seed for randomization of rows in the assays.
        source_column: Optional cell-metadata column populated from each entry in ``names``.

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
        zarr_path: ZarrLocation,
        assays: list[MergeAssay],
        names: list[str],
        merge_assay_name: str,
        in_workspaces: list[str] | None = None,
        out_workspace: str | None = None,
        dtype: str | None = None,
        overwrite: bool = False,
        prepend_text: str | None = "orig",
        reset_cell_filter: bool = True,
        seed: int | None = 42,
        storage_options: dict[str, Any] | None = None,
        source_column: str | None = None,
        mem_budget: int | str | None = None,
        nthreads: int | None = None,
        profile: StorageProfile | None = None,
        targetChunkBytes: int | None = None,
        targetShardBytes: int | None = None,
        _row_plan: _RowPlan | None = None,
        _row_chunk_sizes: list[int] | None = None,
    ) -> None:
        validate_assay_name(merge_assay_name)
        self.assays = assays
        self.names = names
        self.inWorkspaces = in_workspaces
        self.outWorkspace = out_workspace
        self.merge_assay_name = merge_assay_name
        assay_resources = [
            assay.resources for assay in assays if hasattr(assay, "resources")
        ]
        resolved_memory = (
            mem_budget
            if mem_budget is not None
            else (
                min(resource.memoryBytes for resource in assay_resources)
                if assay_resources
                else None
            )
        )
        resolved_workers = (
            nthreads
            if nthreads is not None
            else (
                min(resource.workers for resource in assay_resources)
                if assay_resources
                else None
            )
        )
        self.resources = resolve_budget(resolved_memory, resolved_workers)
        if _row_chunk_sizes is not None and len(_row_chunk_sizes) != len(assays):
            raise ValueError("Row chunk sizes must match the number of assays")
        if _row_chunk_sizes is not None and any(rows <= 0 for rows in _row_chunk_sizes):
            raise ValueError("Row chunk sizes must be positive")
        plan_rows = sum(int(assay.rawData.shape[0]) for assay in assays)
        chunk_rows = (
            [int(assay.rawData.chunksize[0]) for assay in assays]
            if _row_chunk_sizes is None
            else [int(rows) for rows in _row_chunk_sizes]
        )
        plan_chunks = sum(
            (int(assay.rawData.shape[0]) + rows - 1) // rows
            for assay, rows in zip(assays, chunk_rows, strict=True)
        )
        plan_features = sum(int(assay.rawData.shape[1]) for assay in assays)
        index_bytes = np.dtype(np.int64).itemsize
        row_plan_bytes = (
            3 * plan_rows * index_bytes
            + 2 * plan_chunks * index_bytes
            + 2 * plan_features * index_bytes
        )
        admitted_worker_count(
            self.resources,
            taskBytes=max(1, row_plan_bytes),
            requested=1,
        )
        self.profile = resolve_storage_profile(zarr_path, profile)
        self.storage_options = storage_options
        self._usesSharedRowPlan = _row_plan is not None or _row_chunk_sizes is not None
        row_plan = (
            self.perform_randomization_rows(seed, _row_chunk_sizes)
            if _row_plan is None
            else _row_plan
        )
        (
            self.permutations_rows,
            self.permutations_rows_offset,
            self.coordinates_permutations,
        ) = row_plan
        for assay_idx, assay in enumerate(self.assays):
            planned_rows = sum(
                rows.size for rows in self.permutations_rows[assay_idx].values()
            )
            if planned_rows != assay.rawData.shape[0]:
                raise ValueError(
                    "All assays from a datastore must contain the same number of cells"
                )
        self.mergedCells: pd.DataFrame = self._merge_cell_table(
            reset_cell_filter, prepend_text, source_column
        )
        self.nCells: int = self.mergedCells.shape[0]
        self.featCollection: list[dict[str, str]] = self._get_feat_ids(assays)
        self.feat_name_ids_same: bool = self.check_feat_ids(self.featCollection)
        self.featCollection_map: list[dict[str, str]]

        if self.feat_name_ids_same is True:
            self.feat_suffix: dict[int, int] = self.get_feat_suffix()
            self.featCollection = self.update_feat_ids()
            self.featCollection_map = self.update_feat_ids_for_map()
        else:
            self.featCollection_map = self.featCollection.copy()

        self.mergedFeats: pd.DataFrame = self._merge_order_feats(self.featCollection)
        self.mergedFeats_map: pd.DataFrame = self._merge_order_feats(
            self.featCollection_map
        )
        self.nFeats: int = self.mergedFeats_map.shape[0]
        self.featOrder: list[np.ndarray] = self._ref_order_feat_idx()
        self.featOrder_map: list[np.ndarray]

        if self.feat_name_ids_same is True:
            self.featOrder_map = self._ref_order_feat_idx_map()
        else:
            self.featOrder_map = self.featOrder.copy()

        self.cellOrder: dict[int, dict[int, np.ndarray]] = self._ref_order_cell_idx()
        admitted_worker_count(
            self.resources,
            taskBytes=1,
            residentBytes=(
                self._metadata_resident_bytes() + self._row_plan_resident_bytes()
            ),
            requested=1,
        )
        self.z: zarr.Group = self._use_existing_zarr(
            zarr_path, merge_assay_name, overwrite
        )
        self._ini_cell_data(overwrite)
        if dtype is None:
            if len(set([str(x.rawData.dtype) for x in self.assays])) == 1:
                max_copies = max(
                    (
                        int(np.unique(order_map, return_counts=True)[1].max())
                        for order_map in self.featOrder_map
                        if order_map.size
                    ),
                    default=1,
                )
                dtype = str(
                    _dtype_for_integer_sum(
                        np.dtype(self.assays[0].rawData.dtype),
                        max_copies,
                    )
                )
            else:
                dtype = "float"

        self.assayGroup = create_zarr_count_assay(
            z=self.z,
            assay_name=merge_assay_name,
            workspace=self.outWorkspace,
            n_cells=self.nCells,
            feat_ids=np.array(self.mergedFeats_map["ids"]),
            feat_names=np.array(self.mergedFeats_map["names"]),
            dtype=dtype,
            profile=self.profile,
            targetChunkBytes=targetChunkBytes,
            targetShardBytes=targetShardBytes,
        )

    def _metadata_resident_bytes(self) -> int:
        frames = (self.mergedCells, self.mergedFeats, self.mergedFeats_map)
        frame_bytes = sum(
            int(frame.memory_usage(index=True, deep=True).sum()) for frame in frames
        )
        feature_bytes = 0
        seen: set[int] = set()
        for collection in (self.featCollection, self.featCollection_map):
            feature_bytes += sys.getsizeof(collection)
            for mapping in collection:
                if id(mapping) in seen:
                    continue
                seen.add(id(mapping))
                feature_bytes += sys.getsizeof(mapping)
                for key, value in mapping.items():
                    for item in (key, value):
                        if id(item) not in seen:
                            seen.add(id(item))
                            feature_bytes += sys.getsizeof(item)
        return frame_bytes + feature_bytes

    def _row_plan_resident_bytes(self) -> int:
        arrays = [
            self.coordinates_permutations,
            *(
                rows
                for chunks in self.permutations_rows.values()
                for rows in chunks.values()
            ),
            *(
                rows
                for chunks in self.permutations_rows_offset.values()
                for rows in chunks.values()
            ),
            *(rows for chunks in self.cellOrder.values() for rows in chunks.values()),
            *self.featOrder,
            *self.featOrder_map,
        ]
        array_bytes = sum(
            array.nbytes for array in {id(array): array for array in arrays}.values()
        )
        mappings = (
            self.permutations_rows,
            self.permutations_rows_offset,
            self.cellOrder,
        )
        container_bytes = (
            sys.getsizeof(self.featOrder)
            + sys.getsizeof(self.featOrder_map)
            + sum(
                sys.getsizeof(mapping)
                + sum(sys.getsizeof(chunks) for chunks in mapping.values())
                for mapping in mappings
            )
        )
        return array_bytes + container_bytes

    def perform_randomization_rows(
        self,
        seed: int | None = 42,
        row_chunk_sizes: list[int] | None = None,
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
        if row_chunk_sizes is None:
            chunkSize = np.array([x.rawData.chunksize[0] for x in self.assays])
        else:
            if len(row_chunk_sizes) != len(self.assays):
                raise ValueError("Row chunk sizes must match the number of assays")
            chunkSize = np.asarray(row_chunk_sizes, dtype=int)
            if np.any(chunkSize <= 0):
                raise ValueError("Row chunk sizes must be positive")
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
        new_cells: dict[int, dict[int, np.ndarray]] = {}
        for i in range(len(self.assays)):
            new_cells[i] = {}
        offset = 0
        for x, y in self.coordinates_permutations:
            size = self.permutations_rows[x][y].size
            new_cells[x][y] = np.arange(
                offset,
                offset + size,
                dtype=np.int64,
            )
            offset += size
        return new_cells

    def _merge_cell_table(
        self,
        reset: bool,
        prepend_text: str | None = None,
        source_column: str | None = None,
    ) -> pd.DataFrame:
        """Merges the cell metadata table for each sample.

        Args:
            reset: whether to remove filtering information
            prepend_text: string to add as prefix for each cell column
            source_column: optional column populated with each source name

        Returns:
        """
        if len(self.assays) != len(set(self.names)):
            raise ValueError(
                "ERROR: A unique name should be provided for each of the assay"
            )
        if prepend_text == "":
            prepend_text = None
        if source_column is not None and (
            not isinstance(source_column, str)
            or not source_column.strip()
            or source_column in {"ids", "I", "names"}
        ):
            raise ValueError(
                "source_column must be a non-empty string that is not ids, I, or names"
            )
        ret_val = []
        for assay, name in zip(self.assays, self.names):
            a = assay.cells.to_pandas_dataframe(assay.cells.columns)
            a["ids"] = np.array([f"{name}__{x}" for x in a["ids"]])
            for i in list(a.columns):
                if i not in ["ids", "I", "names"] and prepend_text is not None:
                    a[f"{prepend_text}_{i}"] = assay.cells.fetch_all(i)
                    a = a.drop(columns=[i])
            if source_column is not None:
                if source_column in a.columns:
                    raise ValueError(
                        f"source_column {source_column!r} conflicts with merged metadata"
                    )
                a[source_column] = np.repeat(name, len(a))
            if reset:
                a["I"] = np.ones(len(a["ids"])).astype(bool)
            ret_val.append(a)

        # Here we merge the cell metadata tables for each sample. We simply concatenate the tables and reset the index.
        ret_val_df = pd.concat(ret_val, axis=0).reset_index(drop=True)
        # Now we use the offsets stored in permutations_rows_offset along with the coordinates_permutations to reorder the cells in the merged assay. The offsets are used to bring the cells in the same order as the rows in the merged assay.
        compiled_idx = np.concatenate(
            [
                self.permutations_rows_offset[i][j]
                for i, j in self.coordinates_permutations
            ]
        )
        # Index the merged cell metadata table with the compiled_idx to get the final randomized merged cell metadata table.
        ret_val_df = ret_val_df.iloc[compiled_idx]
        if sum([x.cells.N for x in self.assays]) != ret_val_df.shape[0]:
            raise AssertionError(
                "Unexpected number of cells in the merged table. This is unexpected, "
                " please report this bug"
            )
        return ret_val_df

    @staticmethod
    def _get_feat_ids(assays: list[MergeAssay]) -> list[dict[str, str]]:
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
                logger.warning(
                    f"Feature names and IDs are identical for assay "
                    f"{self.assays[i].name} in dataset {self.names[i]}; "
                    "feature names will be used as IDs"
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
                        updated_val = f"{val}_{min_val + sum_counter[val]}"
                        in_dict[updated_val] = updated_val
                    sum_counter[val] += 1
            else:
                for _, val in dict_.items():
                    # check if the value ends with a number
                    if pattern.search(val):
                        num = int(val.split("_")[-1])
                        # replace the number with min_val
                        updated_val = pattern.sub(
                            f"_{min_val - self.feat_suffix[i] + num}", val
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
            in_dict: dict[str, str] = {}
            for feat_val in dict_.values():
                if pattern.search(feat_val):
                    base_val = "_".join(feat_val.split("_")[:-1])
                    if base_val not in in_dict:
                        in_dict[base_val] = base_val
                else:
                    in_dict[feat_val] = feat_val
            new_featCollection.append(in_dict)
        return new_featCollection

    def _merge_order_feats(self, feat_collection: list[dict[str, str]]) -> pd.DataFrame:
        """Merge features from all the assays and determine their order.

        Returns:
        """
        union_set: dict[str, str] = {}
        for ids in feat_collection:
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
            logger.warning("Fewer than 10% of features overlap across the assays")
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
        self, zarr_loc: ZarrLocation, merge_assay_name: str, overwrite: bool
    ) -> zarr.Group:
        if self.outWorkspace is None:
            cell_slot = "cellData"
            assay_slot = merge_assay_name
        else:
            cell_slot = f"{self.outWorkspace}/cellData"
            assay_slot = f"{self.outWorkspace}/{merge_assay_name}"

        try:
            z = load_zarr(zarr_loc, mode="r", storage_options=self.storage_options)
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
                cell_data = as_zarr_group(z[cell_slot], name=cell_slot)
                cell_ids = as_zarr_array(cell_data["ids"], name="ids")
                if not all(
                    np.asarray(cell_ids[:]) == np.array(self.mergedCells["ids"])
                ):
                    raise ValueError(
                        f"ERROR: order of cells does not match the one in existing file"  # noqa: F541
                    )
            except KeyError:
                raise ValueError(
                    f"ERROR: 'cell data seems corrupted. Either delete the "  # noqa: F541
                    "existing file or choose another path"
                )
            return load_zarr(zarr_loc, mode="r+", storage_options=self.storage_options)
        except FileNotFoundError:
            # So no zarr file with same name exists. Check if a non zarr folder with the same name exists
            if (
                is_local_zarr_path(zarr_loc)
                and isinstance(zarr_loc, str)
                and os.path.exists(zarr_loc)
            ):
                raise ValueError(
                    f"ERROR: Directory/file with name `{zarr_loc}`exists. "
                    f"Either delete it or use another name"
                )
            # creating a new zarr file
            return load_zarr(zarr_loc, mode="w", storage_options=self.storage_options)

    def _ini_cell_data(self, overwrite: bool) -> None:
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
            logger.debug("Cell metadata already exists; skipping initialization")

    def _dask_to_coo(
        self,
        d_arr: Any,
        order: np.ndarray,
        order_map: np.ndarray,
        n_threads: int,
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

        Each source column is remapped through `order_map` to its merged feature
        index. If multiple source columns map to the same merged feature, their
        values are summed.
        """
        from . import controlled_compute

        computed_data = controlled_compute(d_arr, n_threads)
        if (
            order.shape != order_map.shape
            or order_map.shape[0] != computed_data.shape[1]
        ):
            raise ValueError("Feature order does not match the source matrix width")

        source = coo_matrix(computed_data)
        mapped = coo_matrix(
            (source.data, (source.row, order_map[source.col])),
            shape=(computed_data.shape[0], self.nFeats),
        )
        if not bool(mapped.has_canonical_format):
            destination = getattr(self, "assayGroup", None)
            mapped = canonicalize_sparse(
                mapped,
                None if destination is None else destination.dtype,
            )
        return mapped

    def dump(self) -> None:
        """Copy the values from individual assays to the merged assay.

        Returns:
        """
        assay_blocks = (
            []
            if self._usesSharedRowPlan
            else [list(assay.rawData.blocks) for assay in self.assays]
        )
        coordinates = [
            (int(assay_idx), int(block_idx))
            for assay_idx, block_idx in self.coordinates_permutations
        ]

        expected_start = 0
        for assay_idx, block_idx in coordinates:
            row_idx = self.cellOrder[assay_idx][block_idx]
            if row_idx.size == 0 or int(row_idx[0]) != expected_start:
                raise AssertionError(
                    "ERROR: Merged block order does not match the cell metadata order."
                )
            expected_start += row_idx.size

        def convert_block(coordinate: tuple[int, int]) -> coo_matrix:
            assay_idx, block_idx = coordinate
            perm_order = self.permutations_rows[assay_idx][block_idx]
            if self._usesSharedRowPlan:
                row_start = int(perm_order.min())
                row_end = int(perm_order.max()) + 1
                block = Block(
                    self.assays[assay_idx].rawData,
                    row_start,
                    row_end,
                    row_perm=perm_order - row_start,
                )
            else:
                local_order = perm_order - perm_order.min()
                block = assay_blocks[assay_idx][block_idx][local_order, :]
            return self._dask_to_coo(
                block,
                self.featOrder[assay_idx],
                self.featOrder_map[assay_idx],
                self.resources.workers,
            )

        def block_stream() -> Iterator[coo_matrix]:
            blocks = map(convert_block, coordinates)
            yield from iter_progress(
                blocks,
                total=len(coordinates),
                desc="Writing merged assay",
            )

        source_rows = max(
            (
                int(rows.size)
                for chunks in self.permutations_rows.values()
                for rows in chunks.values()
            ),
            default=0,
        )
        buffered_rows = min(
            self.nCells,
            source_rows + array_shard_rows(self.assayGroup),
        )
        row_nnz: list[np.ndarray] = []
        for assay in self.assays:
            column = f"{assay.name}_nFeatures"
            if column in assay.cells.columns:
                counts = np.asarray(
                    assay.cells.fetch_all(column),
                    dtype=np.int64,
                )
            else:
                counts = np.full(
                    int(assay.rawData.shape[0]),
                    int(assay.rawData.shape[1]),
                    dtype=np.int64,
                )
            row_nnz.append(counts)
        ordered_nnz = np.concatenate(
            [
                row_nnz[assay_idx][self.permutations_rows[assay_idx][block_idx]]
                for assay_idx, block_idx in coordinates
            ]
        )
        source_nnz = max(
            (
                int(
                    row_nnz[assay_idx][
                        self.permutations_rows[assay_idx][block_idx]
                    ].sum()
                )
                for assay_idx, block_idx in coordinates
            ),
            default=0,
        )
        if ordered_nnz.size:
            width = min(buffered_rows, ordered_nnz.size)
            cumulative = np.empty(ordered_nnz.size + 1, dtype=np.int64)
            cumulative[0] = 0
            np.cumsum(ordered_nnz, dtype=np.int64, out=cumulative[1:])
            buffered_nnz = int(np.max(cumulative[width:] - cumulative[:-width]))
            del cumulative
        else:
            buffered_nnz = 0
        dense_source_elements = max(
            (
                rows.size * int(self.assays[assay_idx].rawData.shape[1])
                for assay_idx, chunks in self.permutations_rows.items()
                for rows in chunks.values()
            ),
            default=0,
        )
        del row_nnz, ordered_nnz
        value_bytes = max(
            np.dtype(self.assayGroup.dtype).itemsize,
            *(np.dtype(assay.rawData.dtype).itemsize for assay in self.assays),
        )
        resident_bytes = (
            self._row_plan_resident_bytes() + self._metadata_resident_bytes()
        )
        counter = accumulate_sparse_to_shards(
            self.assayGroup,
            block_stream(),
            resources=self.resources,
            residentBytes=resident_bytes,
            producerReserveBytes=sparse_producer_peak_bytes(
                buffered_nnz,
                source_nnz,
                value_bytes,
            )
            + dense_source_elements * value_bytes,
        )
        if counter != self.nCells or expected_start != self.nCells:
            raise AssertionError(
                "ERROR: Mismatch in number of cells in the merged assay. Please report this issue."
            )

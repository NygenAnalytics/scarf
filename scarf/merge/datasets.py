from typing import Any

import scarf
import zarr

from ..matrix import ChunkedArray
from ..storage.arrays import create_zarr_dataset
from ..storage.profiles import StorageProfile, ZarrLocation
from ..utils.logging import logger
from .assays import AssayMerge, DummyAssay, MergeAssay, _RowPlan


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
        dtype: Dtype of the raw values in the assay. Dtype is automatically inferred from the provided assays. If
               assays have different dtypes then a float type is used.
        overwrite: If True, then overwrites previously created assay in the Zarr file. (Default value: False).
        prepend_text: This text is pre-appended to each column name (Default value: 'orig').
        reset_cell_filter: If True, then the cell filtering information is removed, i.e. even the filtered out cells
                           are set as True as in the 'I' column. To keep the filtering information set the value for
                           this parameter to False. (Default value: True)
        seed: Seed for randomization of rows in the assays.
        source_column: Optional cell-metadata column populated from each entry in ``names``.

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
        datasets: list["scarf.DataStore"],
        zarr_path: ZarrLocation,
        names: list[str],
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
    ) -> None:
        self.datasets = datasets
        self.names = names
        self.zarr_path = zarr_path
        self.in_workspaces = in_workspaces
        self.out_workspace = out_workspace
        self.dtype = dtype
        self.overwrite = overwrite
        self.prepend_text = prepend_text
        self.reset_cell_filter = reset_cell_filter
        self.seed = seed
        self.storage_options = storage_options
        self.source_column = source_column
        self.memBudget = (
            mem_budget
            if mem_budget is not None
            else min(ds.memoryBytes for ds in datasets)
        )
        self.nthreads = (
            nthreads if nthreads is not None else min(ds.nthreads for ds in datasets)
        )
        self.profile = profile
        self.targetChunkBytes = targetChunkBytes
        self.targetShardBytes = targetShardBytes
        self.unique_assays = self.get_unique_assays()
        self.n_unique_assays = len(self.unique_assays)
        self.merge_generators = self.create_merge_generators()

    def get_unique_assays(self) -> list[str]:
        """
        Get unique assays from both datasets
        """
        unique_assays = []
        seen = set()
        for ds in self.datasets:
            for assay_name in ds.assay_names:
                if assay_name not in seen:
                    seen.add(assay_name)
                    unique_assays.append(assay_name)
        return unique_assays

    def create_merge_generators(self) -> list[AssayMerge]:
        """
        Create AssayMerge objects for each unique assay
        """
        from . import AssayMerge

        gens: list[AssayMerge] = []
        shared_row_plan: _RowPlan | None = None
        row_chunk_sizes = [
            min(
                int(ds.get_assay(assay_name).rawData.chunksize[0])
                for assay_name in ds.assay_names
            )
            for ds in self.datasets
        ]
        for assay in self.unique_assays:
            assay_list: list[MergeAssay] = []
            for ds in self.datasets:
                if assay in ds.assay_names:
                    assay_list.append(ds.get_assay(assay))
                else:
                    assay_list.append(self.generate_dummy_assay(ds, assay))
            generator = AssayMerge(
                zarr_path=self.zarr_path,
                assays=assay_list,
                names=self.names,
                merge_assay_name=assay,
                in_workspaces=self.in_workspaces,
                out_workspace=self.out_workspace,
                dtype=self.dtype,
                overwrite=self.overwrite,
                prepend_text=self.prepend_text,
                reset_cell_filter=self.reset_cell_filter,
                seed=self.seed,
                storage_options=self.storage_options,
                source_column=self.source_column,
                mem_budget=self.memBudget,
                nthreads=self.nthreads,
                profile=self.profile,
                targetChunkBytes=self.targetChunkBytes,
                targetShardBytes=self.targetShardBytes,
                _row_plan=shared_row_plan,
                _row_chunk_sizes=row_chunk_sizes,
            )
            gens.append(generator)
            if shared_row_plan is None:
                shared_row_plan = (
                    generator.permutations_rows,
                    generator.permutations_rows_offset,
                    generator.coordinates_permutations,
                )
        return gens

    def generate_dummy_assay(
        self,
        ds: "scarf.DataStore",
        assay_name: str,
    ) -> DummyAssay:
        """
        Generate a dummy assay for a datastore that doesn't have the specified assay
        """
        from . import DummyAssay

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
        dummy_counts = ChunkedArray(
            dummy_array,
            nthreads=ds.nthreads,
            resources=ds.resources,
        )
        dummy_assay = DummyAssay(
            ds, dummy_counts, reference_assay.feats, reference_assay.name
        )
        logger.warning(
            f"Generated an empty {assay_name} assay for a dataset missing that assay"
        )
        return dummy_assay

    def dump(self) -> None:
        """
        Dump the merged data to the zarr file
        """
        logger.info(f"Merging {len(self.merge_generators)} assays")
        for gen in self.merge_generators:
            logger.debug(f"Writing merged assay {gen.merge_assay_name}")
            gen.dump()
        logger.info(f"Merged {len(self.merge_generators)} assays")
        return None

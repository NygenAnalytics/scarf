from typing import Any

import numpy as np

from ..storage.types import as_zarr_group
from ..readers import H5adReader
from ..readers.h5ad import _H5adAssayFeatures
from ..storage.stores import ZARRLOC
from ..utils.logging import logger
from ..utils.progress import tqdmbar

# Root group names the datastore layout reserves for cell metadata, matrices,
# and plots. Assays must not collide with these or the store is corrupted.
_RESERVED_ASSAY_NAMES = frozenset({"cellData", "matrices", "plots"})


def _validate_assay_names(names: tuple[str, ...]) -> None:
    for name in names:
        if not name or not name.strip():
            raise ValueError("Assay names must be non-empty")
        if "/" in name or "\\" in name:
            raise ValueError(f"Assay name {name!r} must not contain path separators")
        if name in _RESERVED_ASSAY_NAMES:
            raise ValueError(
                f"Assay name {name!r} is reserved by the datastore layout. "
                "Choose another name or provide a different assay_name_map."
            )


class H5adToZarr:
    """A class for converting data in anndata's H5ad format to Zarr hierarchy.

    Args:
        h5ad: Reader for the source H5AD file.
        zarr_loc: The file name for the Zarr hierarchy or a store
        assay_name: the name of the assay (e. g. 'RNA')
        assay_split_key: A var column used to split features into assays.
        assay_name_map: Feature type to assay name overrides.
        workspace: An optional workspace id.
        chunk_size: The requested size of chunks to load into memory and process.
        mem_budget: Memory budget driving write-time chunk and shard geometry. Accepts bytes, a
                    suffixed size (e.g. '8G'), or a fraction of total system memory (e.g. '0.6').
                    Set it to simulate writing on a machine with a different memory size.
        nthreads: Worker count for write-time concurrency. When None, auto-detected.
        working_copies: Number of concurrent in-memory working copies the memory budget is divided
                        across. When None, uses SCARF_WORKING_COPIES env var or the default.

    Attributes:
        h5ad: A h5ad object (h5 file with added AnnData structure).
        chunkSizes: The requested size of chunks to load into memory and process.
        assayName: The Zarr hierarchy (array or group).
        z: The Zarr hierarchy (array or group).
    """

    def __init__(
        self,
        h5ad: H5adReader,
        zarr_loc: ZARRLOC,
        assay_name: str | None = None,
        workspace: str | None = None,
        chunk_size: tuple[int, int] = (1000, 1000),
        storage_options: dict[str, Any] | None = None,
        mem_budget: int | str | None = None,
        nthreads: int | None = None,
        working_copies: int | None = None,
        targetChunkBytes: int | None = None,
        minFeatureChunk: int | None = None,
        maxFeatureChunk: int | None = None,
        assay_split_key: str | None = None,
        assay_name_map: dict[str, str] | None = None,
    ) -> None:
        from . import (
            _apply_budget_override,
            create_zarr_count_assay,
            load_zarr,
        )

        _apply_budget_override(mem_budget, nthreads, working_copies)
        self.h5ad = h5ad
        self.chunkSizes = chunk_size
        self.workspace = workspace
        self.storage_options = storage_options
        self.assaySplitKey = assay_split_key
        self.assayNameMap = assay_name_map
        self.assayFeatures: dict[str, _H5adAssayFeatures] | None
        if assay_split_key is not None:
            if assay_name is not None:
                logger.warning(
                    "`assay_name` is ignored when `assay_split_key` is provided"
                )
            self.assayName = None
            self.assayFeatures = self.h5ad.assay_feature_slices(
                assay_split_key,
                assay_name_map,
            )
            self.assayNames = tuple(self.assayFeatures)
        elif assay_name is None:
            logger.info(
                "No value provided for assay names. Will use default value: 'RNA'"
            )
            self.assayName = "RNA"
            self.assayFeatures = None
            self.assayNames = (self.assayName,)
        else:
            self.assayName = assay_name
            self.assayFeatures = None
            self.assayNames = (self.assayName,)
        _validate_assay_names(self.assayNames)
        self.z = load_zarr(zarr_loc=zarr_loc, mode="w", storage_options=storage_options)
        self._ini_cell_data()
        for resolved_assay_name in self.assayNames:
            if self.assayFeatures is None:
                feature_ids = self.h5ad.feat_ids()
                feature_names = self.h5ad.feat_names()
            else:
                features = self.assayFeatures[resolved_assay_name]
                feature_ids = features.featureIds
                feature_names = features.featureNames
            create_zarr_count_assay(
                z=self.z,
                assay_name=resolved_assay_name,
                workspace=workspace,
                chunk_size=chunk_size,
                n_cells=self.h5ad.nCells,
                feat_ids=feature_ids,
                feat_names=feature_names,
                dtype=self.h5ad.matrixDtype,
                targetChunkBytes=targetChunkBytes,
                minFeatureChunk=minFeatureChunk,
                maxFeatureChunk=maxFeatureChunk,
            )
        self._ini_feature_data()

    def _ini_cell_data(self) -> None:
        from . import create_cell_data, create_zarr_obj_array

        ids = self.h5ad.cell_ids()
        g = create_cell_data(
            z=self.z,
            workspace=self.workspace,
            ids=ids,
            names=ids,
        )
        for i, j in self.h5ad.get_cell_columns():
            create_zarr_obj_array(g, i, j, j.dtype)

    def _ini_feature_data(self) -> None:
        from . import create_zarr_obj_array

        targets: list[tuple[Any, np.ndarray | None]] = []
        for assay_name in self.assayNames:
            if self.workspace is None:
                group_path = f"{assay_name}/featureData"
            else:
                group_path = f"{self.workspace}/{assay_name}/featureData"
            feat_group = as_zarr_group(self.z[group_path], name=group_path)
            feature_indexes = (
                None
                if self.assayFeatures is None
                else self.assayFeatures[assay_name].featureIndexes
            )
            targets.append((feat_group, feature_indexes))

        # Stream one column at a time so a single decoded var column is held in
        # memory rather than every column for the full feature axis at once.
        for column_name, values in self.h5ad.get_feat_columns():
            for feat_group, feature_indexes in targets:
                if column_name in feat_group:
                    continue
                selected = (
                    values if feature_indexes is None else values[feature_indexes]
                )
                create_zarr_obj_array(
                    feat_group,
                    column_name,
                    selected,
                    selected.dtype,
                )

    def dump(self, batch_size: int = 1000) -> None:
        """Write h5ad matrix data into the Zarr counts array.

        Args:
            batch_size: Number of cells written per sparse_writer batch.

        Raises:
            AssertionError: If written cell count does not match expected nCells.

        Returns:
            None
        """
        from . import finalize_writer_counts, load_count_store, sparse_writer

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.assayFeatures is not None:
            self._dump_multi_assay(batch_size)
            return

        assay_name = self.assayNames[0]
        store = load_count_store(self.z, assay_name, self.workspace)
        total_cells_written = sparse_writer(
            store=store,
            data_stream=self.h5ad.consume(batch_size),
            n_cells=self.h5ad.nCells,
            batch_size=batch_size,
        )
        if total_cells_written != self.h5ad.nCells:
            raise AssertionError(
                "ERROR: This is a bug in H5adToZarr. All cells might not have been successfully "
                "written into the zarr file. Please report this issue"
            )
        finalize_writer_counts(self.z, assay_name, self.workspace)

    def _dump_multi_assay(self, batch_size: int) -> None:
        from . import finalize_writer_counts, load_count_store

        if self.assayFeatures is None:
            raise RuntimeError("Multi-assay features have not been initialized")
        stores = {
            assay_name: load_count_store(self.z, assay_name, self.workspace)
            for assay_name in self.assayNames
        }
        offsets: dict[str, tuple[int, ...]] = {}
        for assay_name, features in self.assayFeatures.items():
            current = 0
            assay_offsets = []
            for start, end in features.ranges:
                assay_offsets.append(-start + current)
                current += end - start
            offsets[assay_name] = tuple(assay_offsets)

        cell_start = 0
        n_chunks = (self.h5ad.nCells + batch_size - 1) // batch_size
        for matrix in tqdmbar(
            self.h5ad.consume(batch_size),
            total=n_chunks,
        ):
            chunk = matrix.tocoo(copy=False)
            # SciPy treats repeated coordinates additively; Zarr coordinate
            # assignment keeps only the last write, so collapse duplicates first.
            chunk.sum_duplicates()
            for assay_name, features in self.assayFeatures.items():
                selected = np.zeros(chunk.nnz, dtype=bool)
                feature_coordinates = chunk.col.copy()
                for feature_range, offset in zip(
                    features.ranges,
                    offsets[assay_name],
                    strict=True,
                ):
                    start, end = feature_range
                    in_range = (chunk.col >= start) & (chunk.col < end)
                    feature_coordinates[in_range] += offset
                    selected |= in_range
                if np.any(selected):
                    stores[assay_name].set_coordinate_selection(
                        (
                            cell_start + chunk.row[selected],
                            feature_coordinates[selected],
                        ),
                        chunk.data[selected],
                    )
            cell_start += chunk.shape[0]

        if cell_start != self.h5ad.nCells:
            raise AssertionError(
                "ERROR: This is a bug in H5adToZarr. All cells might not have "
                "been successfully written into the zarr file. Please report this issue"
            )
        for assay_name in self.assayNames:
            finalize_writer_counts(self.z, assay_name, self.workspace)

from typing import Any

import numpy as np
import pandas as pd
import zarr

from ..matrix import ChunkedArray
from ..metadata import MetaData
from ..storage.types import as_zarr_array, as_zarr_group
from ..utils.arrays import array_digest
from ..utils.compute import show_dask_progress
from ..utils.logging import logger
from .base import Assay
from .normalization import norm_tf_idf

_CELL_INDEX_DIGEST_ATTR = "cell_index_digest"
_NORMALIZATION_IDENTITY_ATTR = "normalization_identity"
_DOCUMENT_FREQUENCY_COLUMN = "document_frequency"


class ATACassay(Assay):
    """This subclass of Assay is designed for feature selection and
    normalization of scATAC-Seq data."""

    def __init__(
        self,
        z: zarr.Group,
        name: str,
        cell_data: MetaData,
        *,
        workspace: str | None = None,
        nthreads: int = 1,
        min_cells_per_feature: int = 10,
        **kwargs: Any,
    ) -> None:
        """This Assay subclass is designed for feature selection and
        normalization of scATAC-Seq data.

        Args:
            z (zarr.Group): Zarr hierarchy where raw data is located
            name (str): A label/name for assay.
            cell_data: Metadata class object for the cell attributes.
            **kwargs:

        Attributes:
            normMethod: Pointer to the function to be used for normalization of the raw data
            n_term_per_doc: Number of features per cell. Used for TF-IDF normalization
            n_docs: Number of cells. Used for TF-IDF normalization
            n_docs_per_term: Number of cells per feature. Used for TF-IDF normalization
        """
        super().__init__(
            z=z,
            workspace=workspace,
            name=name,
            cell_data=cell_data,
            nthreads=nthreads,
            min_cells_per_feature=min_cells_per_feature,
            **kwargs,
        )
        self.normMethod = norm_tf_idf
        self.n_term_per_doc: np.ndarray | None = None
        self.n_docs: int | None = None
        self.n_docs_per_term: np.ndarray | None = None

    def normed(
        self,
        cell_idx: np.ndarray | None = None,
        feat_idx: np.ndarray | None = None,
        **kwargs: Any,
    ) -> ChunkedArray:
        """This function normalizes the raw and returns a delayed chunked array of
        the normalized data. Unlike the `normed` method in the generic Assay
        class this method is optimized for scATAC-Seq data. This method uses
        the normalization indicated by attribute self.normMethod which by
        default is set to `norm_tf_idf`. Document frequency is learned from
        `cell_idx`. The returned matrix contains `feat_idx`, while term frequency
        uses total ATAC counts unless subset renormalization is requested.

        Args:
            cell_idx: Indices of cells to be included in the normalized matrix
                      (Default value: All those marked True in 'I' column of cell
                      attribute table)
            feat_idx: Indices of features to be included in the normalized matrix
                      (Default value: All those marked True in 'I' column of
                      feature attribute table)
            **kwargs: `log_transform` must be false. `renormalize_subset` uses
                      counts among `feat_idx` as the term-frequency denominator.

        Returns: A chunked array (delayed matrix) containing normalized data.
        """
        if cell_idx is None:
            cell_idx = self.cells.active_index("I")
        if feat_idx is None:
            feat_idx = self.feats.active_index("I")
        log_transform = kwargs.get("log_transform", False)
        renormalize_subset = kwargs.get("renormalize_subset", False)
        if not isinstance(log_transform, (bool, np.bool_)):
            raise TypeError("log_transform must be a boolean")
        if not isinstance(renormalize_subset, (bool, np.bool_)):
            raise TypeError("renormalize_subset must be a boolean")
        if log_transform:
            raise ValueError("ATAC TF-IDF does not support log_transform; use False")
        cell_idx = np.asarray(cell_idx, dtype=np.int64)
        feat_idx = np.asarray(feat_idx, dtype=np.int64)
        counts: ChunkedArray = self.rawData[:, feat_idx][cell_idx, :]
        self.n_term_per_doc = self._terms_per_document(
            cell_idx,
            counts=counts,
            renormalize_subset=bool(renormalize_subset),
        )
        self.n_docs = len(cell_idx)
        if self.normMethod is not norm_tf_idf:
            self.n_docs_per_term = self.feats.fetch_all("nCells")[feat_idx]
        elif self.n_docs == 0:
            self.n_docs_per_term = np.zeros(len(feat_idx), dtype=np.int64)
        else:
            document_frequency = self._cached_document_frequency(cell_idx, feat_idx)
            if document_frequency is None:
                document_frequency = show_dask_progress(
                    counts.count_nonzero(axis=0),
                    f"({self.name}) Computing document frequency across selected cells",
                    self.nthreads,
                )
            self.n_docs_per_term = np.asarray(document_frequency)
        return self.normMethod(self, counts)

    def _terms_per_document(
        self,
        cell_idx: np.ndarray,
        *,
        counts: ChunkedArray | None = None,
        renormalize_subset: bool = False,
    ) -> np.ndarray:
        """Return total ATAC counts used as each cell's TF denominator."""
        if renormalize_subset:
            if counts is None:
                raise ValueError(
                    "Selected counts are required for subset renormalization"
                )
            if len(cell_idx) == 0:
                terms = np.zeros(0, dtype=np.float64)
            else:
                terms = np.asarray(
                    show_dask_progress(
                        counts.sum(axis=1),
                        f"({self.name}) Recomputing counts across selected peaks",
                        self.nthreads,
                    ),
                    dtype=np.float64,
                )
        else:
            terms = np.asarray(
                self.cells.fetch_all(self.name + "_nCounts")[cell_idx],
                dtype=np.float64,
            )
        terms[terms == 0] = 1
        return terms

    @staticmethod
    def _normalization_identity() -> str:
        identity = getattr(norm_tf_idf, "artifact_identity", None)
        if identity is None:
            raise RuntimeError("norm_tf_idf must define artifact_identity")
        return str(identity)

    @staticmethod
    def _cell_index_digest(cell_idx: np.ndarray) -> str:
        return array_digest(np.asarray(cell_idx, dtype=np.int64))

    def _cached_document_frequency(
        self,
        cell_idx: np.ndarray,
        feat_idx: np.ndarray,
    ) -> np.ndarray | None:
        """Return cached document frequency for an identical ordered cell corpus."""
        if self.normMethod is not norm_tf_idf:
            return None
        expected_digest = self._cell_index_digest(cell_idx)
        expected_identity = self._normalization_identity()
        for stats_loc in sorted(
            str(key) for key in self.z.keys() if str(key).startswith("summary_stats_")
        ):
            try:
                stats_group = as_zarr_group(self.z[stats_loc], name=stats_loc)
            except TypeError:
                continue
            if (
                stats_group.attrs.get(_CELL_INDEX_DIGEST_ATTR) != expected_digest
                or stats_group.attrs.get(_NORMALIZATION_IDENTITY_ATTR)
                != expected_identity
                or _DOCUMENT_FREQUENCY_COLUMN not in stats_group
            ):
                continue
            try:
                stored = as_zarr_array(
                    stats_group[_DOCUMENT_FREQUENCY_COLUMN],
                    name=f"{stats_loc}/{_DOCUMENT_FREQUENCY_COLUMN}",
                )
            except TypeError:
                continue
            if stored.shape != (self.feats.N,):
                continue
            try:
                values = np.asarray(stored[:], dtype=np.float64)[feat_idx]
            except (IndexError, TypeError, ValueError):
                continue
            if (
                np.isfinite(values).all()
                and np.all(values >= 0)
                and np.all(values <= len(cell_idx))
            ):
                return np.asarray(values)
        return None

    def _valid_tfidf_stats(
        self,
        stats_loc: str,
        cell_idx: np.ndarray,
        feat_idx: np.ndarray,
    ) -> bool:
        if not self._validate_stats_loc(
            stats_loc,
            cell_idx,
            feat_idx,
            delete_on_fail=False,
        ):
            return False
        try:
            stats_group = as_zarr_group(self.z[stats_loc], name=stats_loc)
        except (KeyError, TypeError):
            return False
        if (
            stats_group.attrs.get(_CELL_INDEX_DIGEST_ATTR)
            != self._cell_index_digest(cell_idx)
            or stats_group.attrs.get(_NORMALIZATION_IDENTITY_ATTR)
            != self._normalization_identity()
        ):
            return False
        for column in ("prevalence", _DOCUMENT_FREQUENCY_COLUMN):
            if column not in stats_group:
                return False
            try:
                stored = as_zarr_array(
                    stats_group[column],
                    name=f"{stats_loc}/{column}",
                )
            except TypeError:
                return False
            if stored.shape != (self.feats.N,):
                return False
            try:
                values = np.asarray(stored[:], dtype=np.float64)[feat_idx]
            except (IndexError, TypeError, ValueError):
                return False
            if not np.isfinite(values).all():
                return False
            if column == _DOCUMENT_FREQUENCY_COLUMN and (
                np.any(values < 0) or np.any(values > len(cell_idx))
            ):
                return False
        return True

    def _streaming_tfidf_feature_stats(
        self,
        cell_idx: np.ndarray,
        feat_idx: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute document frequency and TF-IDF prevalence in one raw-data pass."""
        n_docs = len(cell_idx)
        n_features = len(feat_idx)
        if n_docs == 0 or n_features == 0:
            return (
                np.zeros(n_features, dtype=np.int64),
                np.zeros(n_features, dtype=np.float64),
            )

        counts = self.rawData[:, feat_idx][cell_idx, :]
        terms_per_document = self._terms_per_document(cell_idx)
        document_frequency = np.zeros(n_features, dtype=np.int64)
        term_frequency_sum = np.zeros(n_features, dtype=np.float64)
        float_itemsize = np.dtype(np.float64).itemsize
        feature_temporaries = 2 * n_features * float_itemsize
        static_bytes = (
            terms_per_document.nbytes
            + document_frequency.nbytes
            + term_frequency_sum.nbytes
            + feature_temporaries
        )
        decode_bytes = counts._max_decode_bytes()
        current_rows = min(int(counts.chunksize[0]), n_docs)
        owned_bytes_per_row = counts._block_owned_bytes() // current_rows
        working_bytes_per_row = owned_bytes_per_row + n_features * float_itemsize
        available_bytes = int(self.resources.memoryBytes) - static_bytes - decode_bytes
        if available_bytes < working_bytes_per_row:
            required_bytes = static_bytes + decode_bytes + working_bytes_per_row
            raise MemoryError(
                "ATAC peak prevalence needs about "
                f"{required_bytes} bytes for one row, but the operation limit is "
                f"{self.resources.memoryBytes} bytes"
            )
        block_rows = min(current_rows, available_bytes // working_bytes_per_row)
        counts = counts._with_block_size(block_rows)
        resident_bytes = static_bytes + block_rows * n_features * float_itemsize
        row_offset = 0
        for raw in counts._stream_blocks(
            nthreads=self.nthreads,
            msg=f"({self.name}) Calculating peak prevalence across cells",
            prefetch=1,
            row_mask=None,
            resident_bytes=resident_bytes,
        ):
            row_stop = row_offset + raw.shape[0]
            document_frequency += np.count_nonzero(raw, axis=0)
            scaled = np.asarray(raw, dtype=np.float64)
            scaled /= terms_per_document[row_offset:row_stop].reshape(-1, 1)
            term_frequency_sum += scaled.sum(axis=0)
            row_offset = row_stop
        if row_offset != n_docs:
            raise RuntimeError(
                f"({self.name}) Feature-stat stream produced {row_offset} rows; "
                f"expected {n_docs}"
            )
        idf = np.log2(1 + (n_docs / (document_frequency + 1)))
        return document_frequency, term_frequency_sum * idf

    def set_feature_stats(self, cell_key: str) -> None:
        """Calculates prevalence of each valid feature of the assay using only
        cells that are marked True by the 'cell_key' parameter. Prevalence of a
        feature is the sum of all its TF-IDF normalized values across cells.

        Args:
            cell_key: Name of the key (column) from cell attribute table.

        Returns: None
        """
        feat_key = "I"  # Here we choose to calculate stats for all the features
        cell_idx, feat_idx = self._get_cell_feat_idx(cell_key, feat_key)
        identifier, stats_loc = self._get_summary_stats_loc(cell_key)
        if self.normMethod is norm_tf_idf:
            cache_is_valid = self._valid_tfidf_stats(
                stats_loc,
                cell_idx,
                feat_idx,
            )
        else:
            cache_is_valid = self._validate_stats_loc(
                stats_loc,
                cell_idx,
                feat_idx,
                delete_on_fail=False,
            )
        if cache_is_valid:
            logger.debug(f"Using cached feature stats for cell_key {cell_key}")
            return None
        if identifier in self.feats.locations:
            self.feats.unmount_location(identifier)
        if stats_loc in self.z:
            del self.z[stats_loc]
        if len(cell_idx) == 0 or len(feat_idx) == 0:
            raise ValueError("Peak prevalence requires selected cells and features")
        document_frequency: np.ndarray | None = None
        if self.normMethod is norm_tf_idf:
            document_frequency, prevalence = self._streaming_tfidf_feature_stats(
                cell_idx,
                feat_idx,
            )
        else:
            prevalence = show_dask_progress(
                self.normed(cell_idx, feat_idx).sum(axis=0),
                f"({self.name}) Calculating peak prevalence across cells",
                self.nthreads,
            )
        self.z.create_group(stats_loc, overwrite=True)
        self.feats.mount_location(
            as_zarr_group(self.z[stats_loc], name=stats_loc), identifier
        )
        self.feats.insert(
            "prevalence", prevalence.astype(float), overwrite=True, location=identifier
        )
        if document_frequency is not None:
            self.feats.insert(
                _DOCUMENT_FREQUENCY_COLUMN,
                document_frequency.astype(float),
                overwrite=True,
                location=identifier,
            )
        self.z[stats_loc].attrs["subset_hash"] = self._create_subset_hash(
            cell_idx, feat_idx
        )
        if document_frequency is not None:
            self.z[stats_loc].attrs[_CELL_INDEX_DIGEST_ATTR] = self._cell_index_digest(
                cell_idx
            )
            self.z[stats_loc].attrs[_NORMALIZATION_IDENTITY_ATTR] = (
                self._normalization_identity()
            )
        self.feats.unmount_location(identifier)
        return None

    def mark_prevalent_peaks(
        self, cell_key: str, top_n: int, prevalence_key_name: str
    ) -> None:
        """Marks `top_n` peaks with highest prevalence as prevalent peaks.

        Args:
           cell_key: Cells to use for selection of most prevalent peaks. The provided value for `cell_key` should be a
                     column in cell attributes table with boolean values.
           top_n: Number of top prevalent peaks to be selected. (Default: 500)
           prevalence_key_name: Base label for marking prevalent peaks in the features attributes column. The value for
                                'cell_key' parameter is prepended to this value.

        Returns: None
        """
        import warnings

        warnings.warn(
            "ATACassay.mark_prevalent_peaks writes legacy metadata directly; "
            "use DataStore.mark_prevalent_peaks for artifact-backed persistence.",
            DeprecationWarning,
            stacklevel=2,
        )
        values = self._prevalent_peak_mask(cell_key, top_n)
        prevalence_key_name = cell_key + "__" + prevalence_key_name
        self.feats.insert(
            prevalence_key_name,
            values,
            fill_value=False,
            overwrite=True,
        )
        return None

    def _prevalent_peak_mask(
        self,
        cell_key: str,
        top_n: int,
    ) -> np.ndarray:
        if top_n >= self.feats.N:
            raise ValueError(
                f"ERROR: n_top should be less than total number of features ({self.feats.N})]"
            )
        if isinstance(top_n, int) is False or top_n < 1:
            raise TypeError("ERROR: n_top must a positive integer value")
        self.set_feature_stats(cell_key)
        identifier = self._load_stats_loc(cell_key)
        idx = (
            pd.Series(self.feats.fetch_all(f"{identifier}_prevalence"))
            .sort_values(ascending=False)
            .index.values[:top_n]
        )
        return np.asarray(self.feats.index_to_bool(idx), dtype=bool)

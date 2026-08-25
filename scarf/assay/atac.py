from typing import Any

import numpy as np
import pandas as pd
import zarr

from ..matrix import ChunkedArray
from ..metadata import MetaData
from ..utils.compute import compute_with_progress
from .base import Assay
from .normalization import norm_tf_idf


class ATACassay(Assay):
    """This subclass of Assay is designed for feature selection and
    normalization of scATAC-Seq data."""

    _feature_summary_operation = "summarize_atac_features"

    def __init__(
        self,
        z: zarr.Group,
        name: str,
        cell_data: MetaData,
        *,
        workspace: str | None = None,
        nthreads: int = 1,
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
            feat_idx: Indices of features to be included in the normalized matrix.
                      Defaults to the complete physical feature axis.
            **kwargs: `log_transform` must be false. `renormalize_subset` uses
                      counts among `feat_idx` as the term-frequency denominator.

        Returns: A chunked array (delayed matrix) containing normalized data.
        """
        if cell_idx is None:
            cell_idx = self.cells.active_index("I")
        if feat_idx is None:
            feat_idx = np.arange(self.feats.N, dtype=np.int64)
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
            document_frequency = compute_with_progress(
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
                    compute_with_progress(
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

    def _compute_feature_summary(
        self,
        cell_idx: np.ndarray,
        feat_idx: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Compute peak sufficient statistics without persisting metadata."""
        cell_idx = np.asarray(cell_idx, dtype=np.int64)
        feat_idx = np.asarray(feat_idx, dtype=np.int64)
        if len(cell_idx) == 0 or len(feat_idx) == 0:
            document_frequency = np.zeros(len(feat_idx), dtype=np.float64)
            prevalence = np.zeros(len(feat_idx), dtype=np.float64)
        elif self.normMethod is norm_tf_idf:
            document_frequency, prevalence = self._streaming_tfidf_feature_stats(
                cell_idx,
                feat_idx,
            )
        else:
            normed = self.normed(cell_idx, feat_idx)
            document_frequency = compute_with_progress(
                (normed > 0).sum(axis=0),
                f"({self.name}) Computing document frequency",
                self.nthreads,
            )
            prevalence = compute_with_progress(
                normed.sum(axis=0),
                f"({self.name}) Calculating peak prevalence across cells",
                self.nthreads,
            )
        return {
            "prevalence": np.asarray(prevalence, dtype=np.float64),
            "document_frequency": np.asarray(
                document_frequency,
                dtype=np.float64,
            ),
        }

    def _prevalent_peak_mask(
        self,
        prevalence: np.ndarray,
        top_n: int,
    ) -> np.ndarray:
        prevalence = np.asarray(prevalence, dtype=np.float64)
        if prevalence.shape != (self.feats.N,):
            raise ValueError(
                f"prevalence must have shape ({self.feats.N},), got {prevalence.shape}"
            )
        if top_n >= self.feats.N:
            raise ValueError(
                f"ERROR: n_top should be less than total number of features ({self.feats.N})]"
            )
        if isinstance(top_n, int) is False or top_n < 1:
            raise TypeError("ERROR: n_top must a positive integer value")
        idx = pd.Series(prevalence).sort_values(ascending=False).index.values[:top_n]
        return np.asarray(self.feats.index_to_bool(idx), dtype=bool)

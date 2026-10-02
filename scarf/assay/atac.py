from typing import Any

import numpy as np
import pandas as pd
import zarr

from ..matrix import ChunkedArray
from ..metadata import MetaData
from ..utils.compute import compute_with_progress
from .base import Assay
from .normalization import (
    inverse_document_frequency,
    norm_tf_idf,
    stream_document_frequency,
)


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
            n_docs_per_term: Number of cells per feature. Used for TF-IDF normalization.
                ``normed`` sets these three only while it calls ``normMethod``
                and then restores them.
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
        counts, state = self._fit_tf_idf(cell_idx, feat_idx, **kwargs)
        # The method reads the fitted state from these attributes while it
        # builds the lazy result, so concurrent calls must not interleave here.
        with self._normalization_lock:
            previous = (self.n_term_per_doc, self.n_docs, self.n_docs_per_term)
            self.n_term_per_doc, self.n_docs, self.n_docs_per_term = state
            try:
                return self.normMethod(self, counts)
            finally:
                self.n_term_per_doc, self.n_docs, self.n_docs_per_term = previous

    def _fit_tf_idf(
        self,
        cell_idx: np.ndarray | None = None,
        feat_idx: np.ndarray | None = None,
        **kwargs: Any,
    ) -> tuple[ChunkedArray, tuple[np.ndarray, int, np.ndarray]]:
        """Return the selected counts and the TF-IDF state fitted on them.

        Arguments are those of ``normed``. The state holds each cell's
        term-frequency denominator, the number of cells, and each feature's
        document frequency, which ``normed`` hands to the normalization method
        as ``n_term_per_doc``, ``n_docs``, and ``n_docs_per_term``.
        """
        from ..storage.identity import read_dataset_fingerprint

        read_dataset_fingerprint(self.z)
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
        n_term_per_doc = self._terms_per_document(
            cell_idx,
            counts=counts,
            renormalize_subset=bool(renormalize_subset),
        )
        n_docs = len(cell_idx)
        if self.normMethod is not norm_tf_idf:
            n_docs_per_term = self.feats.fetch_all("nCells")[feat_idx]
        elif n_docs == 0:
            n_docs_per_term = np.zeros(len(feat_idx), dtype=np.int64)
        else:
            n_docs_per_term = np.asarray(
                compute_with_progress(
                    counts.count_nonzero(axis=0),
                    f"({self.name}) Computing document frequency across selected cells",
                    self.nthreads,
                )
            )
        return counts, (n_term_per_doc, n_docs, n_docs_per_term)

    def _terms_per_document(
        self,
        cell_idx: np.ndarray,
        *,
        counts: ChunkedArray | None = None,
        renormalize_subset: bool = False,
    ) -> np.ndarray:
        """Return the float64 total ATAC counts used as each cell's TF denominator."""
        if renormalize_subset:
            if counts is None:
                raise ValueError(
                    "Selected counts are required for subset renormalization"
                )
            if len(cell_idx) == 0:
                terms = np.zeros(0, dtype=np.float64)
            else:
                terms = compute_with_progress(
                    counts.sum(axis=1, dtype=np.float64),
                    f"({self.name}) Recomputing counts across selected peaks",
                    self.nthreads,
                )
        else:
            terms = self._cell_count_totals(cell_idx)
        terms[terms == 0] = 1
        return terms

    def _streaming_tfidf_feature_stats(
        self,
        cell_idx: np.ndarray,
        feat_idx: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute document frequency and TF-IDF prevalence in one raw-data pass."""
        n_docs = len(cell_idx)
        document_frequency, term_frequency_sum = stream_document_frequency(
            self.rawData[:, feat_idx][cell_idx, :],
            memory_bytes=int(self.resources.memoryBytes),
            nthreads=self.nthreads,
            msg=f"({self.name}) Calculating peak prevalence across cells",
            operation="ATAC peak prevalence",
            term_totals=self._terms_per_document(cell_idx),
        )
        assert term_frequency_sum is not None
        idf = inverse_document_frequency(n_docs, document_frequency)
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
        """Mark the ``top_n`` most prevalent peaks.

        ``select_prevalent_peaks`` passes the full-axis prevalence of a
        feature summary and a ``top_n`` from 1 to one fewer than the peaks.
        """
        prevalence = np.asarray(prevalence, dtype=np.float64)
        idx = pd.Series(prevalence).sort_values(ascending=False).index.values[:top_n]
        return np.asarray(self.feats.index_to_bool(idx), dtype=bool)

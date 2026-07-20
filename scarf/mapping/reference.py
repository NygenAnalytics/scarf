"""Persistent handles for immutable Symphony-style mapping references."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import zarr

from .models import MappingResult, SymphonyReferenceModel

if TYPE_CHECKING:
    import pandas as pd

    from ..assay import Assay


@dataclass(frozen=True)
class MappingReference:
    """An immutable Symphony-style reference loaded from a SCARF Zarr store."""

    datastore: Any
    assay_name: str
    cell_key: str
    feature_key: str
    reduction_path: str
    ann_path: str
    artifact_path: str
    model: SymphonyReferenceModel
    feature_ids: np.ndarray
    metadata: dict[str, Any]
    reference_distance_quantiles: np.ndarray | None = None
    reference_distance_values: np.ndarray | None = None

    def map_query(
        self,
        target_assay: "Assay",
        target_name: str,
        target_feat_key: str,
        target_cell_key: str = "I",
        save_k: int = 3,
        query_batches: "pd.DataFrame | None" = None,
        correction_method: str = "symphony",
        missing_feature_policy: str = "reference_mean",
        result_store: zarr.Group | None = None,
    ) -> MappingResult:
        """Map a query into the fixed reference coordinate system.

        A writable reference stores results under its projection group. A
        read-only reference returns arrays in memory unless ``result_store`` is
        supplied.
        """
        return cast(
            MappingResult,
            self.datastore._map_with_mapping_reference(
                self,
                target_assay=target_assay,
                target_name=target_name,
                target_feat_key=target_feat_key,
                target_cell_key=target_cell_key,
                save_k=save_k,
                query_batches=query_batches,
                correction_method=correction_method,
                missing_feature_policy=missing_feature_policy,
                result_store=result_store,
            ),
        )

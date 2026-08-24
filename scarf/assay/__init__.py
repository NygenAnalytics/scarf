"""
- Classes:
    - Assay: A generic Assay class that contains methods to calculate feature level statistics.
    - RNAassay: This assay is designed for feature selection and normalization of scRNA-Seq data.
    - ATACassay: This assay is designed for ATAC-Seq data. It uses TF-IDF normalization and
                 performs feature selection by marking most prevalent peaks.
    - ADTassay: This assay is designed for ADT data (surface antibodies) obtained from CITE-Seq
                experiments. It performs CLR normalization of the data but does not have any
                method for feature selection.
"""

from .adt import ADTassay as ADTassay
from .atac import ATACassay as ATACassay
from .base import Assay as Assay
from .base import PercentFeatures as PercentFeatures
from .classification import is_rna_assay_type as is_rna_assay_type
from .classification import lookup_persisted_assay_type as lookup_persisted_assay_type
from .classification import preset_assay_types as preset_assay_types
from .classification import resolve_persisted_assay_type as resolve_persisted_assay_type
from .classification import rna_assay_type_names as rna_assay_type_names
from .normalization import NormMethod as NormMethod
from .normalization import (
    lib_size_feature_stream_eligible as lib_size_feature_stream_eligible,
)
from .normalization import norm_clr as norm_clr
from .normalization import norm_dummy as norm_dummy
from .normalization import norm_lib_size as norm_lib_size
from .normalization import norm_lib_size_log as norm_lib_size_log
from .normalization import norm_tf_idf as norm_tf_idf
from .persistence import _read_block as _read_block
from .rna import RNAassay as RNAassay

__all__ = [
    "Assay",
    "RNAassay",
    "ATACassay",
    "ADTassay",
    "is_rna_assay_type",
    "lookup_persisted_assay_type",
    "preset_assay_types",
    "resolve_persisted_assay_type",
    "rna_assay_type_names",
]


def _normalize_public_metadata() -> None:
    norm_lib_size.__globals__["Assay"] = Assay

    for function in (
        _read_block,
        lib_size_feature_stream_eligible,
        norm_clr,
        norm_dummy,
        norm_lib_size,
        norm_lib_size_log,
        norm_tf_idf,
    ):
        function.__module__ = __name__

    for assay_class in (Assay, RNAassay, ATACassay, ADTassay):
        assay_class.__module__ = __name__
        for descriptor in assay_class.__dict__.values():
            if isinstance(descriptor, staticmethod):
                descriptor.__func__.__module__ = __name__
            elif callable(descriptor) and hasattr(descriptor, "__module__"):
                descriptor.__module__ = __name__


_normalize_public_metadata()
del _normalize_public_metadata

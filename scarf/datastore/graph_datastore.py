from threading import RLock
from typing import Any

from ._operations.clustering import _ClusteringOperationsMixin
from ._operations.embeddings import _EmbeddingOperationsMixin
from ._operations.graph import _GraphOperationsMixin
from ._operations.mapping_reference import _MappingReferenceOperationsMixin
from ._operations.trajectory import _TrajectoryOperationsMixin
from .base_datastore import BaseDataStore


class GraphDataStore(
    _EmbeddingOperationsMixin,
    _ClusteringOperationsMixin,
    _TrajectoryOperationsMixin,
    _MappingReferenceOperationsMixin,
    _GraphOperationsMixin,
    BaseDataStore,
):
    """This class extends BaseDataStore by providing methods required to
    generate a cell-cell neighbourhood graph.

    It also contains all the methods that use the KNN graphs as primary input like UMAP/tSNE embedding calculation,
    clustering, down-sampling etc.

    Attributes:
        cells: List of cell barcodes.
        nthreads: Number of threads to use for this datastore instance.
        z: The Zarr file (directory) used for this datastore instance.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._graphMemoryCache: dict[tuple[str, bool, bool, int | None], Any] | None = (
            None
        )
        self._graphMemoryCacheLock = RLock()

from ._operations.mapping import _MappingOperationsMixin
from .graph_datastore import GraphDataStore


class MappingDatastore(_MappingOperationsMixin, GraphDataStore):
    """This class extends GraphDataStore by providing methods for mapping/
    projection of cells from one DataStore onto another. It also contains the methods
    required for label transfer, mapping score generation and co-embedding.
    """

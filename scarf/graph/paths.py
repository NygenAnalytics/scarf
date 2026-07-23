"""Value types for assay and integrated graph locations.

These dataclasses carry Zarr group paths and cheaply parsed parameters only.
They do not compute, write, or import mapping code.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AssayNearestNeighborPaths:
    """Exact current Zarr group locations through nearest neighbors."""

    normalized_group_path: str
    reduction_group_path: str
    neighbor_index_group_path: str
    nearest_neighbors_group_path: str


@dataclass(frozen=True, slots=True)
class AssayGraphPaths:
    """Exact current Zarr group locations for one assay graph chain."""

    normalized_group_path: str
    reduction_group_path: str
    neighbor_index_group_path: str
    nearest_neighbors_group_path: str
    cell_graph_group_path: str
    kmeans_initialization_group_path: str | None = None


@dataclass(frozen=True, slots=True)
class StoredAssayGraph:
    """A graph built from one assay, with paths and parsed parameters."""

    paths: AssayGraphPaths
    from_assay: str
    cell_key: str
    feat_key: str
    reduction_method: str | None = None
    dims: int | None = None
    pca_cell_key: str | None = None
    ann_metric: str | None = None
    ann_efc: int | None = None
    ann_ef: int | None = None
    ann_m: int | None = None
    rand_state: int | None = None
    k: int | None = None
    local_connectivity: float | None = None
    bandwidth: float | None = None
    feat_scaling: bool | None = None
    harmony_contract_hash: str | None = None
    n_centroids: int | None = None


@dataclass(frozen=True, slots=True)
class StoredIntegratedGraph:
    """An integrated graph group and the metadata actually stored on it."""

    cell_graph_group_path: str
    n_cells: int | None = None
    n_neighbors: int | None = None


StoredGraph = StoredAssayGraph | StoredIntegratedGraph

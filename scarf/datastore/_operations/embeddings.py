from typing import TYPE_CHECKING, Any

import numpy as np

from ...graph.arguments import graph_flag
from ...graph.distances import validate_distance_provenance
from ...graph.feature_projection import (
    graph_cell_selection,
    resolve_coordinate_inputs,
    resolve_native_graph_inputs,
)
from ...metadata.artifacts import (
    plan_cell_data_artifact,
    write_cell_data_artifact,
)
from ...metadata.arguments import TsneArguments, UmapArguments
from ...storage.artifacts import (
    ArtifactRef,
    artifact_group,
    fingerprint_array,
    inspect_artifact,
)
from ...storage.types import as_zarr_array, as_zarr_group
from ...utils.arguments import float_argument, integer_argument
from ...utils.logging import logger, progress_enabled
from ...utils.shutdown import shutdown_checkpoint

if TYPE_CHECKING:
    from .graph import _GraphOperationsMixin as _EmbeddingOperationsBase
else:
    _EmbeddingOperationsBase = object


def _checked_initialization(
    values: np.ndarray,
    dtype: type[np.floating[Any]] | None = None,
) -> np.ndarray:
    """Return a private C-ordered copy of finite, real initial coordinates."""
    if values.dtype.kind not in "iuf":
        raise TypeError("Initial embedding must contain real numbers")
    # Values that overflow ``dtype`` become infinite and are rejected below.
    with np.errstate(over="ignore"):
        copy = np.array(values, dtype=dtype, order="C", copy=True)
    if not np.all(np.isfinite(copy)):
        raise ValueError("Initial embedding must contain only finite values")
    return copy


def _check_initial_shape(ini_embed: np.ndarray, n_cells: int, n_comps: int) -> None:
    if ini_embed.shape != (n_cells, n_comps):
        raise ValueError(
            f"Initial embedding has an invalid shape; expected {(n_cells, n_comps)}"
        )


class _EmbeddingOperationsMixin(_EmbeddingOperationsBase):
    def _get_ini_embed(
        self,
        initialization: ArtifactRef,
        graph: ArtifactRef,
        n_comps: int,
    ) -> np.ndarray:
        """Runs PCA on kmeans cluster centers and ascribes the PC values to
        individual cells based on their cluster labels. This is used in
        `run_umap` and `run_tsne` for initial embedding of cells. Uses
        `rescale_array` to reduce the magnitude of extreme values.

        Args:
            initialization: Explicit embedding-initialization artifact.
            graph: Graph whose rows the initialization must match.
            n_comps: Number of PC components to use

        Returns:
            Matrix with n_comps dimensions representing initial embedding of cells.
        """
        from ...embeddings.initialization import initial_embedding

        if initialization.kind != "embedding_initialization":
            raise ValueError("initialization must be an embedding_initialization ref")
        if initialization.scope != "assay" or initialization.assay is None:
            raise ValueError("initialization must be an assay-scoped artifact")
        initialization_status = inspect_artifact(self.zw, initialization)
        if not initialization_status.complete:
            raise ValueError("Embedding initialization is unavailable or incomplete")
        if initialization_status.operation != "build_embedding_initialization":
            raise ValueError("Embedding initialization has an invalid operation")
        raw_source = (initialization_status.inputs or {}).get("coordinates")
        if not isinstance(raw_source, dict):
            raise ValueError("Embedding initialization has no coordinate source")
        source = ArtifactRef.from_dict(raw_source)
        if source.assay != initialization.assay:
            raise ValueError("Embedding initialization source belongs to another assay")
        source_inputs = resolve_coordinate_inputs(self.zw, source)
        graph_selection = graph_cell_selection(self.zw, graph)
        if source_inputs.cell_selection != graph_selection:
            raise ValueError(
                "Embedding initialization does not match the graph cell selection"
            )
        if graph.kind in {"connectivity_map", "neighbors"}:
            lineage = resolve_native_graph_inputs(self.zw, graph)
            if source != lineage.coordinates:
                raise ValueError(
                    "Embedding initialization does not belong to the graph coordinates"
                )
        kmeans_grp = artifact_group(self.zw, initialization)
        cluster_centers = np.asarray(
            as_zarr_array(
                kmeans_grp["cluster_centers"],
                name="cluster_centers",
            )[:]
        )
        clusters = np.asarray(
            as_zarr_array(kmeans_grp["cluster_labels"], name="cluster_labels")[:]
        )
        return initial_embedding(cluster_centers, clusters, n_comps)

    def _embedding_inputs(
        self,
        graph: ArtifactRef,
        initialization: ArtifactRef | np.ndarray,
        n_comps: int,
        dtype: type[np.floating[Any]] | None,
    ) -> tuple[ArtifactRef, int, object, np.ndarray | None]:
        """Validate embedding inputs without loading the graph.

        Returns the graph cell selection, the graph cell count, the
        initialization identity input, and an array initialization when one
        was given. An initialization artifact is expanded only when needed.
        """
        if not isinstance(graph, ArtifactRef):
            raise TypeError("graph must be an ArtifactRef")
        cell_selection = graph_cell_selection(self.zw, graph)
        n_cells = self._get_graph_ncells_k(self._graph_location(graph))[0]
        if isinstance(initialization, ArtifactRef):
            return cell_selection, n_cells, initialization, None
        if not isinstance(initialization, np.ndarray):
            raise TypeError("initialization must be an ArtifactRef or numpy array")
        ini_embed = _checked_initialization(initialization, dtype)
        _check_initial_shape(ini_embed, n_cells, n_comps)
        identity = {"value_fingerprint": fingerprint_array(ini_embed)}
        return cell_selection, n_cells, identity, ini_embed

    def run_tsne(
        self,
        graph: ArtifactRef,
        initialization: ArtifactRef | np.ndarray,
        *,
        symmetric_graph: bool = False,
        graph_upper_only: bool = False,
        tsne_dims: int = 2,
        lambda_scale: float = 1.0,
        max_iter: int = 500,
        early_iter: int = 200,
        alpha: int = 10,
        box_h: float = 0.7,
        temp_file_loc: str = ".",
        verbose: bool = True,
        parallel: bool = False,
        nthreads: int | None = None,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Run SGtSNE-pi (Read more here:
        https://github.com/fcdimitr/sgtsnepi/tree/v1.0.1). This is an
        implementation of tSNE that runs directly on graph structures. We use
        connectivity graphs generated by graph-construction methods to create a
        layout of cells using tSNE algorithm. This function makes a system call to sgtSNE
        binary. To get a better understanding of how the parameters affect the
        embedding, check this out: http://t-sne-pi.cs.duke.edu/

        Args:
            graph: Explicit connectivity map or integrated graph to embed.
            initialization: Explicit initialization artifact or coordinate array. An array must hold finite real
                            values.
            symmetric_graph: This parameter is forwarded to `load_graph` and is same as there. (Default value: False)
            graph_upper_only: This parameter is forwarded to `load_graph` and is same as there. (Default value: False)
            tsne_dims: Number of tSNE dimensions to compute (Default value: 2)
            lambda_scale: λ rescaling parameter (Default value: 1.0)
            max_iter: Maximum number of iterations (Default value: 500)
            early_iter: Number of early exaggeration iterations (Default value: 200)
            alpha: Early exaggeration multiplier (Default value: 10)
            box_h: Grid side length (accuracy control). Lower values might drastically slow down
                   the algorithm (Default value: 0.7)
            temp_file_loc: Location of temporary file. By default, these files will be created in the current working
                           directory. These files are deleted before the method returns.
            verbose: If True (default) then the full log from SGtSNEpi algorithm is shown. If False, the log is
                     discarded and the process's standard output and error are restored afterwards.
            parallel: Whether to run tSNE in parallel mode. Setting value to True will use `nthreads` threads.
                      The results are not reproducible in parallel mode. (Default value: False)
            nthreads: If parallel=True then this number of threads will be used to run tSNE. By default the `nthreads`
                      attribute of the class is used. (Default value: None)

        Returns:
            Reference to the immutable embedding artifact.
        """
        symmetric_graph = graph_flag(symmetric_graph, "symmetric_graph")
        graph_upper_only = graph_flag(graph_upper_only, "graph_upper_only")
        cell_selection, n_cells, initialization_input, ini_embed = (
            self._embedding_inputs(graph, initialization, tsne_dims, None)
        )
        if parallel:
            if nthreads is None:
                nthreads = self.nthreads
            else:
                assert isinstance(nthreads, int)
        else:
            nthreads = 1
        arguments = TsneArguments(
            graph=graph,
            initialization=initialization_input,
            symmetric_graph=symmetric_graph,
            graph_upper_only=graph_upper_only,
            tsne_dims=tsne_dims,
            lambda_scale=lambda_scale,
            max_iter=max_iter,
            early_iter=early_iter,
            alpha=alpha,
            box_h=box_h,
            parallel=parallel,
            parallel_threads=nthreads,
            temp_file_loc=temp_file_loc,
            verbose=verbose,
            invalidate_cache=invalidate_cache,
        )
        record = arguments.to_record()
        planned = plan_cell_data_artifact(
            self.zw,
            scope=graph.scope,
            assay=(graph.assay if graph.scope == "assay" else None),
            kind=arguments.artifact_kind,
            operation=arguments.operation,
            parameters=record.parameters,
            inputs=record.inputs,
            execution_options=record.execution_options,
            cell_selection=cell_selection,
            arrays={"values": ((n_cells, tsne_dims), "f")},
            invalidate_cache=invalidate_cache,
        )
        if planned.reused:
            logger.info(
                f"Reused {tsne_dims}-dimensional t-SNE embedding for {n_cells} cells"
            )
            return planned.ref
        import sys

        if sys.platform != "linux":
            raise RuntimeError(
                f"{sys.platform} operating system is currently not supported."
            )
        from ...embeddings.sgtsne import run_sgtsne

        graph_matrix = self._load_graph_artifact(
            graph,
            symmetric=symmetric_graph,
            upper_only=graph_upper_only,
            use_k=None,
        )
        if ini_embed is None:
            assert isinstance(initialization, ArtifactRef)
            ini_embed = self._get_ini_embed(initialization, graph, tsne_dims)
            _check_initial_shape(ini_embed, n_cells, tsne_dims)
        try:
            shutdown_checkpoint()
            raw_embedding = np.asarray(
                run_sgtsne(
                    graph_matrix,
                    ini_embed,
                    tsne_dims=tsne_dims,
                    max_iter=max_iter,
                    early_iter=early_iter,
                    alpha=alpha,
                    lambda_scale=lambda_scale,
                    box_h=box_h,
                    temp_file_loc=temp_file_loc,
                    verbose=verbose,
                    parallel=parallel,
                    nthreads=nthreads,
                )
            )
            shutdown_checkpoint()
        except (FileNotFoundError, ImportError) as exc:
            raise RuntimeError(
                "SG-tSNE failed, possibly due to missing sgtsne executable or "
                f"sgtsnepi package: {exc}"
            ) from exc
        if raw_embedding.shape != (tsne_dims, n_cells):
            raise ValueError(
                "SG-tSNE returned an embedding with shape "
                f"{raw_embedding.shape}; expected {(tsne_dims, n_cells)}"
            )
        write_cell_data_artifact(self.zw, planned, {"values": raw_embedding.T})
        logger.info(
            f"Stored {tsne_dims}-dimensional t-SNE embedding for {n_cells} cells"
        )
        return planned.ref

    def _run_umap_artifact(
        self,
        graph: ArtifactRef,
        initialization: ArtifactRef | np.ndarray,
        *,
        symmetric_graph: bool | None = False,
        graph_upper_only: bool | None = False,
        umap_dims: int = 2,
        spread: float = 2.0,
        min_dist: float = 1,
        n_epochs: int = 300,
        repulsion_strength: float = 1.0,
        initial_alpha: float = 1.0,
        negative_sample_rate: float = 5,
        use_density_map: bool = False,
        dens_lambda: float = 2.0,
        dens_frac: float = 0.3,
        dens_var_shift: float = 0.1,
        random_seed: int = 4444,
        parallel: bool = False,
        nthreads: int | None = None,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Run UMAP and store only its immutable coordinate artifact.

        Args:
            graph: Explicit connectivity map or integrated graph to embed.
            initialization: Explicit initialization artifact or coordinate array. An array must hold finite real
                            values; UMAP optimizes a C-ordered float32 copy and leaves the caller's array unchanged.
            symmetric_graph: This parameter is forwarded to `load_graph` and is same as there. (Default value: False)
            graph_upper_only: This parameter is forwarded to `load_graph` and is same as there. (Default value: False)
            umap_dims: Number of dimensions of UMAP embedding (Default value: 2)
            spread: Same as spread in UMAP package.  The effective scale of embedded points. In combination with
                    ``min_dist`` this determines how clustered/clumped the embedded points are.
            min_dist: Same as min_dist in UMAP package. The effective minimum distance between embedded points.
                      Smaller values will result in a more clustered/clumped embedding where nearby points on the
                      manifold are drawn closer together, while larger values will result on a more even dispersal of
                      points. The value should be set relative to the ``spread`` value, which determines the scale at
                      which embedded points will be spread out. (Default value: 1)
            n_epochs: Same as n_epochs in UMAP package. The number of epochs to be used in optimizing the
                      low dimensional embedding. Larger values may result in more accurate embeddings.
                      (Default value: 300)
            repulsion_strength: Same as repulsion_strength in UMAP package. Weighting applied to negative samples in
                                low dimensional embedding optimization. Values higher than one will result in greater
                                weight being given to negative samples. (Default value: 1.0)
            initial_alpha: Same as learning_rate in UMAP package. The initial learning rate for the embedding
                           optimization. (Default value: 1.0)
            negative_sample_rate: Same as negative_sample_rate in UMAP package. The number of negative samples to
                                  select per positive sample in the optimization process. Increasing this value will
                                  result in greater repulsive force being applied, greater optimization cost, but
                                  slightly more accuracy. (Default value: 5)
            use_density_map: If True, run densMAP (density-preserving UMAP) instead of standard UMAP.
            dens_lambda: densMAP density preservation strength (Default value: 2.0).
            dens_frac: Fraction of nearest neighbors used for local density estimation (Default value: 0.3).
            dens_var_shift: Variance shift for density correction (Default value: 0.1).
            random_seed: (Default value: 4444)
            parallel: Whether to run UMAP in parallel mode. Setting value to True will use `nthreads` threads.
                      The results are not reproducible in parallel mode. (Default value: False)
            nthreads: If parallel=True then this number of threads will be used to run UMAP. By default, the `nthreads`
                      attribute of the class is used. (Default value: None)

        Returns:
            Reference to the immutable embedding artifact.
        """
        from ...embeddings.umap import (
            DENSMAP_ALGORITHM_VERSION,
            densmap_distance_graph,
            fit_transform,
        )

        if symmetric_graph is not None:
            symmetric_graph = graph_flag(symmetric_graph, "symmetric_graph")
        if graph_upper_only is not None:
            graph_upper_only = graph_flag(graph_upper_only, "graph_upper_only")
        # Parameters are recorded canonically, so ``min_dist=1`` and
        # ``min_dist=1.0`` identify the same embedding.
        umap_dims = integer_argument(umap_dims, "umap_dims", minimum=1)
        n_epochs = integer_argument(n_epochs, "n_epochs", minimum=1)
        random_seed = integer_argument(random_seed, "random_seed", minimum=0)
        spread = float_argument(spread, "spread")
        min_dist = float_argument(min_dist, "min_dist")
        repulsion_strength = float_argument(repulsion_strength, "repulsion_strength")
        initial_alpha = float_argument(initial_alpha, "initial_alpha")
        negative_sample_rate = float_argument(
            negative_sample_rate, "negative_sample_rate"
        )
        dens_lambda = float_argument(dens_lambda, "dens_lambda")
        dens_frac = float_argument(dens_frac, "dens_frac")
        dens_var_shift = float_argument(dens_var_shift, "dens_var_shift")
        use_density_map = graph_flag(use_density_map, "use_density_map")
        parallel = graph_flag(parallel, "parallel")
        # UMAP optimizes its initialization in place and requires C-contiguous
        # float32 coordinates, so an array initialization becomes a private copy.
        cell_selection, n_cells, initialization_input, ini_embed = (
            self._embedding_inputs(graph, initialization, umap_dims, np.float32)
        )
        if nthreads is None:
            nthreads = self.nthreads
        nthreads = integer_argument(nthreads, "nthreads", minimum=1)
        effective_density_map = use_density_map and graph.kind != "integrated_graph"
        arguments = UmapArguments(
            graph=graph,
            initialization=initialization_input,
            symmetric_graph=symmetric_graph,
            graph_upper_only=graph_upper_only,
            umap_dims=umap_dims,
            spread=spread,
            min_dist=min_dist,
            n_epochs=n_epochs,
            repulsion_strength=repulsion_strength,
            initial_alpha=initial_alpha,
            negative_sample_rate=negative_sample_rate,
            use_density_map=effective_density_map,
            dens_lambda=dens_lambda,
            dens_frac=dens_frac,
            dens_var_shift=dens_var_shift,
            random_seed=random_seed,
            parallel=parallel,
            parallel_threads=nthreads if parallel else None,
            invalidate_cache=invalidate_cache,
            densmap_algorithm_version=(
                DENSMAP_ALGORITHM_VERSION if effective_density_map else None
            ),
        )
        record = arguments.to_record()
        planned = plan_cell_data_artifact(
            self.zw,
            scope=graph.scope,
            assay=(graph.assay if graph.scope == "assay" else None),
            kind=arguments.artifact_kind,
            operation=arguments.operation,
            parameters=record.parameters,
            inputs=record.inputs,
            execution_options=record.execution_options,
            cell_selection=cell_selection,
            arrays={"values": ((n_cells, umap_dims), "f")},
            invalidate_cache=invalidate_cache,
        )
        if use_density_map and graph.kind == "integrated_graph":
            logger.warning(
                "DensMap is not available for integrated graphs. Running standard UMAP."
            )
        if planned.reused:
            logger.info(
                f"Reused {umap_dims}-dimensional UMAP embedding for {n_cells} cells"
            )
            return planned.ref
        graph_matrix = self._load_graph_artifact(
            graph,
            symmetric=symmetric_graph,
            upper_only=graph_upper_only,
            use_k=None,
        )
        if ini_embed is None:
            assert isinstance(initialization, ArtifactRef)
            ini_embed = self._get_ini_embed(initialization, graph, umap_dims)
            _check_initial_shape(ini_embed, n_cells, umap_dims)
        densmap_kwds: dict[str, Any] = {}
        if effective_density_map:
            lineage = resolve_native_graph_inputs(self.zw, graph)
            knn_loc = inspect_artifact(self.zw, lineage.neighbors).path
            logger.trace(f"Loading KNN dists and indices from {knn_loc}")
            validate_distance_provenance(self.zw, lineage.neighbors)
            knn_group = as_zarr_group(self.zw[knn_loc], name=knn_loc)
            dists = np.asarray(
                as_zarr_array(knn_group["distances"], name="distances")[:]
            )
            indices = np.asarray(as_zarr_array(knn_group["indices"], name="indices")[:])
            densmap_kwds = {
                "lambda": dens_lambda,
                "frac": dens_frac,
                "var_shift": dens_var_shift,
                "n_neighbors": dists.shape[1],
                "knn_dists": densmap_distance_graph(indices, dists),
            }
            logger.trace("Created symmetric sparse KNN distances")
        shutdown_checkpoint()
        t, _a, _b = fit_transform(
            graph=graph_matrix.tocoo(),
            ini_embed=ini_embed,
            spread=spread,
            min_dist=min_dist,
            n_epochs=n_epochs,
            random_seed=random_seed,
            repulsion_strength=repulsion_strength,
            initial_alpha=initial_alpha,
            negative_sample_rate=negative_sample_rate,
            densmap_kwds=densmap_kwds,
            parallel=parallel,
            nthreads=nthreads,
            verbose=progress_enabled(),
        )
        shutdown_checkpoint()
        write_cell_data_artifact(self.zw, planned, {"values": t})
        logger.info(
            f"Stored {umap_dims}-dimensional UMAP embedding for {n_cells} cells"
        )
        return planned.ref

    def run_umap(
        self,
        graph: ArtifactRef,
        initialization: ArtifactRef | np.ndarray,
        *,
        symmetric_graph: bool | None = False,
        graph_upper_only: bool | None = False,
        umap_dims: int = 2,
        spread: float = 2.0,
        min_dist: float = 1,
        n_epochs: int = 300,
        repulsion_strength: float = 1.0,
        initial_alpha: float = 1.0,
        negative_sample_rate: float = 5,
        use_density_map: bool = False,
        dens_lambda: float = 2.0,
        dens_frac: float = 0.3,
        dens_var_shift: float = 0.1,
        random_seed: int = 4444,
        parallel: bool = False,
        nthreads: int | None = None,
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Build and return an immutable UMAP embedding artifact."""
        return self._run_umap_artifact(
            graph,
            initialization,
            symmetric_graph=symmetric_graph,
            graph_upper_only=graph_upper_only,
            umap_dims=umap_dims,
            spread=spread,
            min_dist=min_dist,
            n_epochs=n_epochs,
            repulsion_strength=repulsion_strength,
            initial_alpha=initial_alpha,
            negative_sample_rate=negative_sample_rate,
            use_density_map=use_density_map,
            dens_lambda=dens_lambda,
            dens_frac=dens_frac,
            dens_var_shift=dens_var_shift,
            random_seed=random_seed,
            parallel=parallel,
            nthreads=nthreads,
            invalidate_cache=invalidate_cache,
        )

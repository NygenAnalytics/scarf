from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ...assay import Assay
from ...graph.build import (
    GraphBuildPlan,
    GraphDataInputs,
    GraphExecutionOptions,
    ResolvedGraphParameters,
)
from ...graph.encoded_paths import (
    lookup_latest_cell_graph_group_path,
    lookup_latest_kmeans_group_path,
    lookup_latest_nearest_neighbors_group_path,
    lookup_latest_neighbor_index_group_path,
    lookup_latest_reduction_group_path,
    make_nearest_neighbors_group_path,
    make_neighbor_index_group_path,
    make_normalized_group_path,
    make_reduction_group_path,
    parse_cell_graph_group_path,
    parse_kmeans_group_path,
    parse_nearest_neighbors_group_path,
    parse_neighbor_index_group_path,
    parse_reduction_group_path,
)
from ...graph.state import read_assay_state
from ...neighbors.stream import AnnStream
from ...storage.artifacts import ArtifactRef, inspect_artifact
from ...storage.types import as_zarr_group
from ...utils.logging import logger

if TYPE_CHECKING:
    from .graph import _GraphOperationsMixin as _GraphLegacyParamsBase
else:
    _GraphLegacyParamsBase = object

_UNSET = object()
_ANN_EF_DEFAULT_LOG = "min(100, max(k * 3, 50))"


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


@dataclass(frozen=True, slots=True)
class _GraphParamRule:
    """One graph parameter: explicit → state → legacy → default."""

    name: str
    cast: Callable[[Any], Any]
    default: Any
    default_log: Any = _UNSET
    validate: Callable[[Any], None] | None = None


def _log_graph_param_source(source: str, name: str, value: Any) -> None:
    msg = f"No value provided for parameter `{name}`. "
    if source == "default":
        msg += f"Will use default value: {value}"
    else:
        msg += f"Will use previously used value: {value}"
    logger.debug(msg)


def _resolve_graph_param(
    rule: _GraphParamRule,
    explicit: Any,
    *,
    state_value: Any = None,
    legacy_value: Any = None,
) -> Any:
    if explicit is not None:
        value = rule.cast(explicit)
        if rule.validate is not None:
            rule.validate(value)
        return value
    if state_value is not None:
        value = rule.cast(state_value)
        _log_graph_param_source("cached", rule.name, value)
        return value
    if legacy_value is not None:
        value = rule.cast(legacy_value)
        _log_graph_param_source("cached", rule.name, value)
        return value
    default = rule.default() if callable(rule.default) else rule.default
    if default is None and rule.default_log is not _UNSET:
        _log_graph_param_source("default", rule.name, rule.default_log)
        return None
    value = rule.cast(default)
    _log_graph_param_source(
        "default",
        rule.name,
        rule.default_log if rule.default_log is not _UNSET else value,
    )
    return value


def _resolve_graph_param_group(
    rules: list[_GraphParamRule],
    explicits: Mapping[str, Any],
    resolved: dict[str, Any],
    *,
    state: Mapping[str, Any] | None,
    legacy: Mapping[str, Any] | None,
    allow_legacy: bool,
) -> None:
    for rule in rules:
        resolved[rule.name] = _resolve_graph_param(
            rule,
            explicits[rule.name],
            state_value=None if state is None else state.get(rule.name),
            legacy_value=(
                None if not allow_legacy or legacy is None else legacy.get(rule.name)
            ),
        )


def _artifact_params(status: Any) -> dict[str, Any]:
    return {} if status is None else dict(status.parameters or {})


def _inspect_state_params(root: Any, ref: ArtifactRef | None) -> dict[str, Any]:
    if ref is None:
        return {}
    return _artifact_params(inspect_artifact(root, ref))


def _legacy_subset_params(root: Any, normed_loc: str) -> dict[str, Any]:
    if normed_loc not in root:
        return {}
    group = as_zarr_group(root[normed_loc], name=normed_loc)
    if "subset_params" not in group.attrs:
        return {}
    return cast(dict[str, Any], group.attrs["subset_params"])


def _legacy_reduction_params(root: Any, normed_loc: str) -> dict[str, Any]:
    if normed_loc not in root:
        return {}
    try:
        path = lookup_latest_reduction_group_path(root, normed_loc)
    except KeyError:
        return {}
    try:
        _, dims_value, pca_key = parse_reduction_group_path(path)
    except ValueError:
        return {}
    return {"dims": dims_value, "pca_cell_key": pca_key}


def _legacy_ann_params(
    root: Any, reduction_loc: str
) -> tuple[dict[str, Any], str | None]:
    if reduction_loc not in root:
        return {}, None
    try:
        ann_loc = lookup_latest_neighbor_index_group_path(root, reduction_loc)
    except KeyError:
        return {}, None
    try:
        parsed = parse_neighbor_index_group_path(ann_loc)
    except ValueError:
        return {}, ann_loc
    return {
        "ann_metric": parsed[0],
        "ann_efc": parsed[1],
        "ann_ef": parsed[2],
        "ann_m": parsed[3],
        "rand_state": parsed[4],
    }, ann_loc


def _legacy_knn_k(root: Any, ann_loc: str | None) -> tuple[dict[str, Any], str | None]:
    if ann_loc is None:
        return {}, None
    try:
        knn_loc = lookup_latest_nearest_neighbors_group_path(root, ann_loc)
    except KeyError:
        return {}, None
    try:
        return {"k": parse_nearest_neighbors_group_path(knn_loc)}, knn_loc
    except ValueError:
        return {}, knn_loc


def _legacy_n_centroids(root: Any, reduction_loc: str) -> dict[str, Any]:
    if reduction_loc not in root:
        return {}
    path = lookup_latest_kmeans_group_path(root, reduction_loc)
    if path is None:
        return {}
    try:
        return {"n_centroids": parse_kmeans_group_path(path)[0]}
    except ValueError:
        return {}


def _legacy_graph_params(root: Any, knn_loc: str) -> dict[str, Any]:
    if knn_loc not in root:
        return {}
    try:
        graph_loc = lookup_latest_cell_graph_group_path(root, knn_loc)
    except KeyError:
        return {}
    try:
        local_connectivity, bandwidth = parse_cell_graph_group_path(graph_loc)
    except ValueError:
        return {}
    return {
        "local_connectivity": local_connectivity,
        "bandwidth": bandwidth,
    }


class _GraphLegacyParamsMixin(_GraphLegacyParamsBase):
    @staticmethod
    def _choose_reduction_method(assay: Assay, reduction_method: str) -> str:
        """This is a convenience function to determine the linear dimension
        reduction method to be used for a given assay. It is uses a
        predetermined rule to make this determination.

        Args:
            assay: Assay object.
            reduction_method: Name of reduction method to use. It can be one from either: 'pca', 'lsi', 'auto'.

        Returns:
            The name of dimension reduction method to be used. Either 'pca' or 'lsi'

        Raises:
            ValueError: If `reduction_method` is not one of either 'pca', 'lsi', 'auto'
        """
        reduction_method = reduction_method.lower()
        if reduction_method not in ["pca", "lsi", "auto", "custom"]:
            raise ValueError(
                "ERROR: Please choose either 'pca' or 'lsi' as reduction method"
            )
        if reduction_method == "auto":
            assay_type = str(assay.__class__).split(".")[-1][:-2]
            if assay_type == "ATACassay":
                logger.debug("Using LSI for dimension reduction")
                reduction_method = "lsi"
            else:
                logger.debug("Using PCA for dimension reduction")
                reduction_method = "pca"
        return reduction_method

    def _resolve_graph_parameters(
        self,
        from_assay: str,
        cell_key: str,
        feat_key: str,
        log_transform: bool | None = None,
        renormalize_subset: bool | None = None,
        reduction_method: str = "auto",
        dims: int | None = None,
        pca_cell_key: str | None = None,
        ann_metric: str | None = None,
        ann_efc: int | None = None,
        ann_ef: int | None = None,
        ann_m: int | None = None,
        rand_state: int | None = None,
        k: int | None = None,
        n_centroids: int | None = None,
        local_connectivity: float | None = None,
        bandwidth: float | None = None,
        feat_scaling: bool = True,
        lsi_skip_first: bool = True,
        harmonize: bool = False,
        batch_columns: list[str] | None = None,
        harmony_params: dict[str, Any] | None = None,
    ) -> ResolvedGraphParameters:
        """Resolve graph parameters from explicit, AssayState, legacy, and defaults."""
        state = read_assay_state(self.zw, from_assay)
        state_normalized: dict[str, Any] = {}
        state_reduction: dict[str, Any] = {}
        state_ann: dict[str, Any] = {}
        state_neighbors: dict[str, Any] = {}
        state_connectivity: dict[str, Any] = {}
        state_initialization: dict[str, Any] = {}
        state_correction: dict[str, Any] = {}
        state_pca_cell_key: str | None = None
        if state is not None and state.matches(cell_key, feat_key):
            reduction_status = (
                inspect_artifact(self.zw, state.reduction)
                if state.reduction is not None
                else None
            )
            state_normalized = _inspect_state_params(self.zw, state.normalized)
            state_reduction = _artifact_params(reduction_status)
            if reduction_status is not None:
                state_pca_cell_key = (reduction_status.execution_options or {}).get(
                    "pca_cell_key"
                )
            state_ann = _inspect_state_params(self.zw, state.ann_index)
            state_neighbors = _inspect_state_params(self.zw, state.neighbors)
            state_connectivity = _inspect_state_params(self.zw, state.connectivity_map)
            state_initialization = _inspect_state_params(
                self.zw, state.embedding_initialization
            )
            state_correction = _inspect_state_params(self.zw, state.batch_correction)

        normed_loc = make_normalized_group_path(from_assay, cell_key, feat_key)
        legacy_norm = (
            {} if state_normalized else _legacy_subset_params(self.zw, normed_loc)
        )
        legacy_reduction = (
            {} if state_reduction else _legacy_reduction_params(self.zw, normed_loc)
        )

        def validate_pca_cell_key(value: Any) -> None:
            if dims is not None:
                return
            key = str(value)
            if key not in self.cells.columns:
                raise ValueError(
                    f"ERROR: `pca_use_cell_key` {key} does not exist in cell metadata"
                )
            if self.cells.get_dtype(key) != bool:  # noqa: E721
                raise TypeError(
                    "ERROR: Type of `pca_use_cell_key` column in cell metadata "
                    "should be `bool`"
                )

        explicits = {
            "log_transform": log_transform,
            "renormalize_subset": renormalize_subset,
            "dims": dims,
            "pca_cell_key": pca_cell_key,
            "ann_metric": ann_metric,
            "ann_efc": ann_efc,
            "ann_ef": ann_ef,
            "ann_m": ann_m,
            "rand_state": rand_state,
            "k": k,
            "n_centroids": n_centroids,
            "local_connectivity": local_connectivity,
            "bandwidth": bandwidth,
        }
        resolved: dict[str, Any] = {}
        _resolve_graph_param_group(
            [
                _GraphParamRule("log_transform", bool, True),
                _GraphParamRule("renormalize_subset", bool, True),
            ],
            explicits,
            resolved,
            state=state_normalized,
            legacy=legacy_norm,
            allow_legacy=not state_normalized,
        )
        _resolve_graph_param_group(
            [
                _GraphParamRule("dims", int, 11),
                _GraphParamRule(
                    "pca_cell_key", str, cell_key, validate=validate_pca_cell_key
                ),
            ],
            explicits,
            resolved,
            state={**state_reduction, "pca_cell_key": state_pca_cell_key},
            legacy=legacy_reduction,
            allow_legacy=not state_reduction,
        )

        if reduction_method == "auto" and state_reduction.get("reduction_method"):
            reduction_method = str(state_reduction["reduction_method"])
        reduction_method = self._choose_reduction_method(
            self._get_assay(from_assay), reduction_method
        )
        reduction_loc = make_reduction_group_path(
            normed_loc, reduction_method, resolved["dims"], resolved["pca_cell_key"]
        )

        legacy_ann, latest_ann_loc = (
            ({}, None) if state_ann else _legacy_ann_params(self.zw, reduction_loc)
        )
        cached_ann = state_ann or legacy_ann
        _resolve_graph_param_group(
            [
                _GraphParamRule("ann_metric", str, "l2"),
                _GraphParamRule(
                    "ann_efc", _optional_int, None, default_log=_ANN_EF_DEFAULT_LOG
                ),
                _GraphParamRule(
                    "ann_ef", _optional_int, None, default_log=_ANN_EF_DEFAULT_LOG
                ),
                _GraphParamRule(
                    "ann_m",
                    int,
                    lambda: min(max(48, int(resolved["dims"] * 1.5)), 64),
                ),
                _GraphParamRule("rand_state", int, 4466),
            ],
            explicits,
            resolved,
            state=state_ann,
            legacy=legacy_ann,
            allow_legacy=not state_ann,
        )

        legacy_neighbors, latest_knn_loc = _legacy_knn_k(self.zw, latest_ann_loc)
        _resolve_graph_param_group(
            [_GraphParamRule("k", int, 11)],
            explicits,
            resolved,
            state=state_neighbors,
            legacy=legacy_neighbors,
            allow_legacy=True,
        )

        ann_keys = ("ann_metric", "ann_efc", "ann_ef", "ann_m", "rand_state")
        ann_matches = all(cached_ann.get(key) == resolved[key] for key in ann_keys)
        for key in ("ann_ef", "ann_efc"):
            if resolved[key] is None:
                resolved[key] = min(100, max(resolved["k"] * 3, 50))
            resolved[key] = int(resolved[key])

        ann_loc = make_neighbor_index_group_path(
            reduction_loc,
            resolved["ann_metric"],
            resolved["ann_efc"],
            resolved["ann_ef"],
            resolved["ann_m"],
            resolved["rand_state"],
        )
        graph_params_knn_loc = make_nearest_neighbors_group_path(ann_loc, resolved["k"])
        if ann_matches and latest_knn_loc is not None:
            try:
                if parse_nearest_neighbors_group_path(latest_knn_loc) == resolved["k"]:
                    graph_params_knn_loc = latest_knn_loc
            except ValueError:
                pass

        _resolve_graph_param_group(
            [_GraphParamRule("n_centroids", int, 1000)],
            explicits,
            resolved,
            state=state_initialization,
            legacy=_legacy_n_centroids(self.zw, reduction_loc),
            allow_legacy=True,
        )
        _resolve_graph_param_group(
            [
                _GraphParamRule("local_connectivity", float, 1.0),
                _GraphParamRule("bandwidth", float, 1.5),
            ],
            explicits,
            resolved,
            state=state_connectivity,
            legacy=(
                {}
                if state_connectivity
                else _legacy_graph_params(self.zw, graph_params_knn_loc)
            ),
            allow_legacy=not state_connectivity,
        )

        if feat_scaling is None:
            feat_scaling = bool(state_reduction.get("feat_scaling", True))
        if lsi_skip_first is None:
            lsi_skip_first = bool(state_reduction.get("lsi_skip_first", True))
        if harmonize is None:
            harmonize = bool(state_correction)
        if harmonize and state_correction:
            if batch_columns is None:
                stored = state_correction.get("batch_columns")
                if isinstance(stored, list):
                    batch_columns = [str(column) for column in stored]
            if harmony_params is None:
                stored = state_correction.get("harmony_parameters")
                if isinstance(stored, dict):
                    harmony_params = dict(stored)

        return ResolvedGraphParameters(
            reduction_method=reduction_method,
            feat_scaling=feat_scaling,
            lsi_skip_first=lsi_skip_first,
            harmonize=harmonize,
            batch_columns=batch_columns,
            harmony_params=harmony_params,
            **{key: resolved[key] for key in explicits},
        )

    def _resolve_graph_plan(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        pca_cell_key: str | None = None,
        reduction_method: str = "auto",
        dims: int | None = None,
        k: int | None = None,
        ann_metric: str | None = None,
        ann_efc: int | None = None,
        ann_ef: int | None = None,
        ann_m: int | None = None,
        ann_parallel: bool = False,
        rand_state: int | None = None,
        n_centroids: int | None = None,
        batch_size: int | None = None,
        log_transform: bool | None = None,
        renormalize_subset: bool | None = None,
        local_connectivity: float | None = None,
        bandwidth: float | None = None,
        update_keys: bool = True,
        return_ann_object: bool = False,
        custom_loadings: np.ndarray | None = None,
        feat_scaling: bool = True,
        lsi_skip_first: bool = True,
        harmonize: bool = False,
        batch_columns: list[str] | None = None,
        show_elbow_plot: bool = False,
        ann_index_fetcher: Callable | None = None,
        ann_index_saver: Callable | None = None,
        local_cache: bool | str = "auto",
        harmony_params: dict[str, Any] | None = None,
        force_harmony_refit: bool = False,
        invalidate_cache: bool = False,
    ) -> GraphBuildPlan:
        if from_assay is None:
            from_assay = self._defaultAssay
        assay = self._get_assay(from_assay)
        if cell_key is None:
            cell_key = "I"
        if feat_key is None:
            bool_col_parts = [
                x.split("__", 1)
                for x in assay.feats.columns
                if assay.feats.get_dtype(x) == bool and x != "I"  # noqa: E721
            ]
            bool_cols_msg = " ".join(f"{part[1]}({part[0]})" for part in bool_col_parts)
            raise ValueError(
                "ERROR: You have to choose which features that should be used for graph construction. "
                "Ideally you should have performed a feature selection step before making this graph. "
                "Feature selection step adds a column to your feature table. \n"
                "You have following boolean columns in the feature "
                f"metadata of assay {from_assay} which you can choose from: {bool_cols_msg}\n The values in "
                f"brackets indicate the cell_key for which the feat_key is available. Choosing 'I' "
                f"as `feat_key` means that you will use all the genes for graph creation."
            )
        if batch_size is None:
            state = read_assay_state(self.zw, from_assay)
            if (
                state is not None
                and state.matches(cell_key, feat_key)
                and state.reduction is not None
            ):
                state_reduction = (
                    inspect_artifact(self.zw, state.reduction).parameters or {}
                )
                stored_batch_size = state_reduction.get("batch_size")
                batch_size = (
                    int(cast(int | float | str, stored_batch_size))
                    if stored_batch_size is not None
                    else None
                )
            if batch_size is None:
                batch_size = assay.rawData.chunksize[0]
        if custom_loadings is not None:
            reduction_method = "custom"
            dims = custom_loadings.shape[1]
            logger.info(
                f"`dims` parameter and its default value ignored as using custom loadings "
                f"with {dims} dims"
            )

        parameters = self._resolve_graph_parameters(
            from_assay=from_assay,
            cell_key=cell_key,
            feat_key=feat_key,
            log_transform=log_transform,
            renormalize_subset=renormalize_subset,
            reduction_method=reduction_method,
            dims=dims,
            pca_cell_key=pca_cell_key,
            ann_metric=ann_metric,
            ann_efc=ann_efc,
            ann_ef=ann_ef,
            ann_m=ann_m,
            rand_state=rand_state,
            k=k,
            n_centroids=n_centroids,
            local_connectivity=local_connectivity,
            bandwidth=bandwidth,
            feat_scaling=feat_scaling,
            lsi_skip_first=lsi_skip_first,
            harmonize=harmonize,
            batch_columns=batch_columns,
            harmony_params=harmony_params,
        )
        n_active_cells = len(self.cells.active_index(cell_key))
        effective_batch_size = min(int(batch_size), n_active_cells)
        if effective_batch_size < 1:
            raise ValueError("Graph construction requires at least one active cell")
        if n_active_cells < 2:
            raise ValueError("Graph construction requires at least two active cells")
        effective_dims = parameters.dims
        if custom_loadings is None:
            effective_dims = min(effective_dims, len(self.cells.active_index(cell_key)))
            if effective_dims >= effective_batch_size:
                effective_dims = max(effective_batch_size - 1, 0)
        effective_centroids = min(
            max(parameters.n_centroids, 2),
            effective_batch_size,
        )
        parameters = replace(
            parameters,
            dims=effective_dims,
            n_centroids=effective_centroids,
            k=min(parameters.k, n_active_cells - 1),
        )
        if parameters.harmonize:
            if parameters.batch_columns is None:
                raise ValueError("Harmonization requested but no batches provided")
            if isinstance(parameters.batch_columns, list) is False:
                raise ValueError(
                    "batches must be a list of columns in cell metadata that represent batches"
                )
            for column in parameters.batch_columns:
                self.cells.fetch(column, key=cell_key)
        return GraphBuildPlan(
            data_inputs=GraphDataInputs(
                assay=assay,
                from_assay=from_assay,
                cell_key=cell_key,
                feat_key=feat_key,
                custom_loadings=custom_loadings,
            ),
            parameters=parameters,
            options=GraphExecutionOptions(
                batch_size=effective_batch_size,
                update_keys=update_keys,
                return_ann_object=return_ann_object,
                show_elbow_plot=show_elbow_plot,
                ann_parallel=ann_parallel,
                ann_index_fetcher=ann_index_fetcher,
                ann_index_saver=ann_index_saver,
                local_cache=local_cache,
                force_harmony_refit=force_harmony_refit,
                invalidate_cache=invalidate_cache,
            ),
        )

    def make_graph(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        feat_key: str | None = None,
        pca_cell_key: str | None = None,
        reduction_method: str = "auto",
        dims: int | None = None,
        k: int | None = None,
        ann_metric: str | None = None,
        ann_efc: int | None = None,
        ann_ef: int | None = None,
        ann_m: int | None = None,
        ann_parallel: bool = False,
        rand_state: int | None = None,
        n_centroids: int | None = None,
        batch_size: int | None = None,
        log_transform: bool | None = None,
        renormalize_subset: bool | None = None,
        local_connectivity: float | None = None,
        bandwidth: float | None = None,
        update_keys: bool = True,
        return_ann_object: bool = False,
        custom_loadings: np.ndarray | None = None,
        feat_scaling: bool = True,
        lsi_skip_first: bool = True,
        harmonize: bool = False,
        batch_columns: list[str] | None = None,
        show_elbow_plot: bool = False,
        ann_index_fetcher: Callable | None = None,
        ann_index_saver: Callable | None = None,
        local_cache: bool | str = "auto",
        harmony_params: dict[str, Any] | None = None,
        _force_harmony_refit: bool = False,
    ) -> AnnStream | None:
        """Compatibility facade for the atomic graph operations.

        .. deprecated:: 1.0
           Use ``DataStore.pipeline`` for the standard RNA workflow or call
           ``run_normalization``, ``run_pca`` or ``run_lsi``, ``run_harmony``,
           ``build_ann_index``, ``query_neighbors``, and
           ``build_connectivity_map`` directly.

        - Normalizes the data calling the `save_normalized_data` for the assay
        - runs reduction and optional Harmony correction
        - builds the ANN index and embedding initialization
        - queries ANN index for nearest neighbours and saves the distances and indices of the neighbours
        - recalculates distances into graph weights
        - publishes the completed refs through `AssayState`

        Args:
            from_assay: Assay to use for graph creation. If no value is provided then `defaultAssay` will be used
            cell_key: Cells to use for graph creation. By default all cells with True value in 'I' will be used.
                      The provided value for `cell_key` should be a column in cell metadata table with boolean values.
            feat_key: Features to use for graph creation. It is a required parameter. We have chosen not to set this
                      to 'I' by default because this might lead to usage of too many features and may lead to poor
                      results. The value for `feat_key` should be a column in feature metadata from the `from_assay`
                      assay and should be boolean type.
            pca_cell_key: Name of a column from cell metadata table. This column should be boolean type. If no value is
                          provided then the value is set to same as `cell_key` which means all the cells in the
                          normalized data will be used for fitting the pca. This parameter, hence, basically provides a
                          mechanism to subset the normalized data only for PCA fitting step. This parameter can be
                          useful, for example, the data has cells from multiple replicates which wont merge together, in
                          which case the `pca_cell_key` can be used to fit PCA on cells from only one of the replicate.
            reduction_method: Method to use for linear dimension reduction. Could be either 'pca', 'lsi' or 'auto'. In
                              case of 'auto' `_choose_reduction_method` will be used to determine the best reduction
                              type for the assay.
            dims: Number of top reduced dimensions to use (Default value: 11)
            k: Number of nearest neighbours to query for each cell (Default value: 11)
            ann_metric: HNSW distance metric (Default value: 'l2')
            ann_efc: HNSW construction search breadth (Default value: min(100, max(k * 3, 50)))
            ann_ef: HNSW query search breadth (Default value: min(100, max(k * 3, 50)))
            ann_m: Maximum HNSW graph degree (Default value: min(max(48, int(dims * 1.5)), 64) )
            ann_parallel: If True, then ANN graph is created in parallel mode using DataStore.nthreads number of
                          threads. Results obtained in parallel mode will not be reproducible. (Default: False)
            rand_state: Random seed number (Default value: 4466)
            n_centroids: Number of centroids for Kmeans clustering. As a general indication, have a value of 1+ for
                         every 100 cells. Small (<2000 cells) and very small (<500 cells) use a ballpark number for max
                         expected number of clusters. The results of kmeans clustering are only used to provide initial
                         embedding for UMAP and tSNE. (Default value: 1000)
            batch_size: Number of cells in a batch. This number is guided by number of features being used and the
                        amount of available free memory. Though the full data is already divided into chunks, however,
                        if only a fraction of features is being used in the normalized dataset, then the chunk size
                        can be increased to speed up the computation (i.e. PCA fitting and ANN index building).
                        (Default value: 1000)
            log_transform: If True, then the normalized data is log-transformed (only affects RNAassay type assays).
                           (Default value: True)
            renormalize_subset: If True, then the data is normalized using only those features that are True in
                                `feat_key` column rather using total expression of all features in a cell (only affects
                                RNAassay type assays). (Default value: True)
            local_connectivity: This parameter is forwarded to `smooth_knn_dist` function from UMAP package. Higher
                                value will push distribution of edge weights towards terminal values (binary like).
                                Lower values will accumulate edge weights around the mean produced by `bandwidth`
                                parameter. (Default value: 1.0)
            bandwidth: This parameter is forwarded to `smooth_knn_dist` function from UMAP package. Higher value will
                       push the mean of distribution of graph edge weights towards right.  (Default value: 1.5). Read
                       more about `smooth_knn_dist` function here:
                       https://umap-learn.readthedocs.io/en/latest/api.html#umap.umap_.smooth_knn_dist
            update_keys: If True, publish the completed chain through
                         `AssayState`. The name is retained for compatibility.
            return_ann_object: If True then returns the ANNStream object. This allows one to directly interact with the
                               PCA transformer and HNSWlib index. Check out ANNStream documentation to know more.
                               (Default: False)
            custom_loadings: Custom loadings/transformer for linear dimension reduction. If provided, should have a form
                             (d x p) where d is same the number of active features in feat_key and p is the number of
                             reduced dimensions. `dims` parameter is ignored when this is provided.
                             (Default value: None)
            feat_scaling: If True (default) then the feature will be z-scaled otherwise not. It is highly recommended
                          that this is kept as True unless you know what you are doing.
            lsi_skip_first: Whether to remove the first LSI dimension when using ATAC-Seq data.
            harmonize: If True, run Harmony batch correction on the PCA embedding before
                       building the KNN graph. Requires ``batch_columns``.
            batch_columns: Cell metadata columns defining batch variables for Harmony.
            harmony_params: Optional keyword arguments forwarded to ``fit_harmony``.
            show_elbow_plot: If True, then an elbow plot is shown when PCA is fitted to the data. Not shown when using
                            existing PCA loadings or custom loadings. (Default value: False)
            ann_index_fetcher: Optional callable to load a pre-built ANN index instead of fitting one.
            ann_index_saver: Optional callable to persist a fitted ANN index for reuse.
            local_cache: When ``'auto'`` or ``True``, remote stores copy the normalized
                         matrix to a local scratch Zarr before PCA/ANN/kmeans/KNN so
                         multi-pass reads hit local disk instead of object storage.
                         A string value is treated as a persistent scratch base path
                         keyed by artifact ID (~8 GB for 1M cells x 2000 HVGs in
                         float32). ``False`` disables staging.

        Returns:
            Either None or `AnnStream` object
        """
        import warnings

        warnings.warn(
            "make_graph is deprecated; call the atomic graph methods instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        plan = self._resolve_graph_plan(
            from_assay=from_assay,
            cell_key=cell_key,
            feat_key=feat_key,
            pca_cell_key=pca_cell_key,
            reduction_method=reduction_method,
            dims=dims,
            k=k,
            ann_metric=ann_metric,
            ann_efc=ann_efc,
            ann_ef=ann_ef,
            ann_m=ann_m,
            ann_parallel=ann_parallel,
            rand_state=rand_state,
            n_centroids=n_centroids,
            batch_size=batch_size,
            log_transform=log_transform,
            renormalize_subset=renormalize_subset,
            local_connectivity=local_connectivity,
            bandwidth=bandwidth,
            update_keys=update_keys,
            return_ann_object=return_ann_object,
            custom_loadings=custom_loadings,
            feat_scaling=feat_scaling,
            lsi_skip_first=lsi_skip_first,
            harmonize=harmonize,
            batch_columns=batch_columns,
            show_elbow_plot=show_elbow_plot,
            ann_index_fetcher=ann_index_fetcher,
            ann_index_saver=ann_index_saver,
            local_cache=local_cache,
            harmony_params=harmony_params,
            force_harmony_refit=_force_harmony_refit,
            invalidate_cache=False,
        )
        return self._run_resolved_graph_plan(plan)

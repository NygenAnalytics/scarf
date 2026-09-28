import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Any

import numpy as np

from ..assay import RNAassay
from ..clustering.leiden import canonical_resolution
from ..features.gene_reference import species_registry
from ..quality_control.filtering import validate_filter_bounds
from ..utils.arguments import integer_argument


@dataclass(frozen=True, slots=True)
class ResolvedPipelineRecipe:
    assay: str
    label: str | None
    cell_key: str
    filtering: dict[str, Any]
    harmony_batch_columns: tuple[str, ...]
    hvg_count: int
    pca_dims: int
    neighbors_k: int
    umap: bool
    leiden_partitions: tuple[tuple[str, float], ...]
    cell_cycle: bool
    paris: bool
    doublets: bool
    markers: bool
    snapshot_columns: tuple[str, ...]
    cell_snapshot_columns: tuple[str, ...]
    stage_order: tuple[str, ...]
    tsne: bool
    membership_strength: bool
    leiden_selected: str | None
    species: str | None
    stage_params: Mapping[str, Mapping[str, Any]]

    def params_for(self, stage: str) -> dict[str, Any]:
        """Return the keyword arguments ``params`` forwards to one stage."""
        return dict(self.stage_params.get(stage, {}))

    def to_config(self) -> dict[str, Any]:
        return {
            "cellKey": self.cell_key,
            "filtering": self.filtering,
            "harmonyBatchColumns": list(self.harmony_batch_columns),
            "hvgCount": self.hvg_count,
            "pcaDims": self.pca_dims,
            "neighborsK": self.neighbors_k,
            "umap": self.umap,
            "tsne": self.tsne,
            "leiden": {
                "partitions": [value for _key, value in self.leiden_partitions],
                "selected": (
                    None
                    if self.leiden_selected is None
                    else dict(self.leiden_partitions)[self.leiden_selected]
                ),
            },
            "cellCycle": self.cell_cycle,
            "paris": self.paris,
            "doublets": self.doublets,
            "markers": self.markers,
            "snapshotColumns": list(self.snapshot_columns),
            "membershipStrength": self.membership_strength,
            "species": self.species,
            "params": {
                stage: dict(values) for stage, values in self.stage_params.items()
            },
        }


def _column_sequence(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of column names")
    columns = tuple(value)
    if any(not isinstance(column, str) or not column for column in columns):
        raise TypeError(f"{name} must contain non-empty strings")
    if len(columns) != len(set(columns)):
        raise ValueError(f"{name} must not contain duplicates")
    return columns


def _resolve_leiden(
    value: Mapping[str, object] | bool,
) -> tuple[tuple[str, float], ...]:
    if value is False:
        return ()
    if value is True:
        return (
            ("0.5", 0.5),
            ("0.75", 0.75),
            ("1.0", 1.0),
            ("1.25", 1.25),
        )
    if not isinstance(value, Mapping):
        raise TypeError("leiden must be a mapping or bool")
    if set(value) != {"partitions"}:
        raise ValueError("leiden must contain exactly 'partitions'")
    raw_partitions = value["partitions"]
    if isinstance(raw_partitions, str | bytes) or not isinstance(
        raw_partitions,
        Sequence,
    ):
        raise TypeError("leiden partitions must be a non-empty sequence")
    partitions = tuple(
        (str(resolution), resolution)
        for resolution in map(canonical_resolution, raw_partitions)
    )
    if not partitions:
        raise ValueError("leiden partitions must not be empty")
    keys = [key for key, _resolution in partitions]
    if len(keys) != len(set(keys)):
        raise ValueError("leiden partitions contain duplicate resolutions")
    return partitions


# Keyword arguments that ``params`` may forward to each stage. The pipeline
# supplies the artifacts a stage consumes, so only its settings appear here.
_STAGE_PARAMETERS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "cell_cycle": frozenset(
            {
                "s_genes",
                "g2m_genes",
                "ctrl_size",
                "log_transform",
                "n_bins",
                "rand_seed",
            }
        ),
        "hvg": frozenset(
            {
                "top_n",
                "min_cells",
                "max_cells",
                "min_mean",
                "max_mean",
                "min_var",
                "max_var",
                "n_bins",
                "lowess_frac",
                "blacklist",
                "keep_bounds",
                "bin_strategy",
            }
        ),
        "normalization": frozenset({"log_transform", "renormalize_subset"}),
        "pca": frozenset({"dims", "feat_scaling", "batch_size"}),
        "harmony": frozenset({"batch_columns", "harmony_params", "batch_size"}),
        "ann_index": frozenset(
            {
                "ann_metric",
                "ann_efc",
                "ann_ef",
                "ann_m",
                "ann_parallel",
                "rand_state",
                "batch_size",
            }
        ),
        "neighbors": frozenset({"k", "batch_size"}),
        "connectivity": frozenset({"local_connectivity", "bandwidth"}),
        "embedding_initialization": frozenset(
            {
                "n_centroids",
                "rand_state",
                "batch_size",
                "kmeans_sampling",
                "kmeans_batch_size",
            }
        ),
        "umap": frozenset(
            {
                "symmetric_graph",
                "graph_upper_only",
                "umap_dims",
                "spread",
                "min_dist",
                "n_epochs",
                "repulsion_strength",
                "initial_alpha",
                "negative_sample_rate",
                "use_density_map",
                "dens_lambda",
                "dens_frac",
                "dens_var_shift",
                "random_seed",
                "parallel",
            }
        ),
        "tsne": frozenset(
            {
                "symmetric_graph",
                "graph_upper_only",
                "tsne_dims",
                "lambda_scale",
                "max_iter",
                "early_iter",
                "alpha",
                "box_h",
                "parallel",
            }
        ),
        "leiden": frozenset(
            {
                "partitions",
                "selected",
                "backend",
                "symmetric_graph",
                "graph_upper_only",
                "random_seed",
            }
        ),
        "membership_strength": frozenset(),
        "paris": frozenset({"n_clusters", "min_cluster_size"}),
        "doublets": frozenset(
            {
                "cluster_sample_fraction",
                "max_cells_per_cluster",
                "simulation_ratio",
                "heterotypic_fraction",
                "save_k",
                "smoothing_t",
                "normalize_scores",
                "random_seed",
            }
        ),
        "markers": frozenset(),
    }
)
# Stages that ``params`` can switch off with False, or on with True.
_OPTIONAL_STAGES = frozenset(
    {
        "filtering",
        "cell_cycle",
        "harmony",
        "umap",
        "tsne",
        "leiden",
        "membership_strength",
        "paris",
        "doublets",
        "markers",
    }
)
_ParamSection = bool | dict[str, Any]


def _parameter_value(value: Any, name: str) -> Any:
    """Return a JSON-compatible copy of one stage setting for the run record."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite; omit it to use the default")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"{name} keys must be strings")
        return {
            key: _parameter_value(item, f"{name}[{key!r}]")
            for key, item in value.items()
        }
    if isinstance(value, Sequence | np.ndarray) and not isinstance(value, bytes):
        return [
            _parameter_value(item, f"{name}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{name} must be a number, string, boolean, None, list, or mapping")


def _resolve_params(
    params: Mapping[str, object] | None,
) -> tuple[dict[str, _ParamSection], str | None]:
    """Validate ``params`` into per-stage switches and settings, plus species."""
    if params is None:
        return {}, None
    if not isinstance(params, Mapping):
        raise TypeError("params must be a mapping of stage names to settings")
    sections: dict[str, _ParamSection] = {}
    species: str | None = None
    for stage, value in params.items():
        name = f"params[{stage!r}]"
        if stage == "species":
            known = species_registry()
            if not isinstance(value, str) or value not in known:
                raise ValueError(f"{name} must be one of {sorted(known)!r}")
            species = value
        elif stage == "filtering":
            if not isinstance(value, bool | Mapping):
                raise TypeError(f"{name} must be a mapping or bool")
            sections[stage] = value if isinstance(value, bool) else dict(value)
        elif stage not in _STAGE_PARAMETERS:
            known_stages = sorted({*_STAGE_PARAMETERS, "filtering", "species"})
            raise ValueError(
                f"Unknown params section {stage!r}; expected one of {known_stages!r}"
            )
        elif isinstance(value, bool):
            if stage not in _OPTIONAL_STAGES:
                raise TypeError(
                    f"{name} must be a mapping because the stage always runs"
                )
            if stage == "harmony" and value:
                raise ValueError(f"{name} needs a mapping with batch_columns")
            sections[stage] = value
        elif isinstance(value, Mapping):
            allowed = _STAGE_PARAMETERS[stage]
            unknown = sorted(set(value) - allowed)
            if unknown:
                raise ValueError(
                    f"Unknown {stage} parameters {unknown!r}; allowed: "
                    f"{sorted(allowed)!r}"
                )
            sections[stage] = {
                key: _parameter_value(item, f"{name}[{key!r}]")
                for key, item in value.items()
            }
        else:
            raise TypeError(f"{name} must be a mapping or bool")
    return sections, species


def _default_filter_columns(store: Any, assay: str) -> tuple[str, ...]:
    return tuple(
        column
        for suffix in ("nCounts", "nFeatures", "percentMito", "percentRibo")
        if (column := f"{assay}_{suffix}") in store.cells.columns
    )


def _manual_bound(value: Any, name: str) -> float | int | None:
    """Return one validated pipeline bound in canonical form.

    The pipeline filters numeric QC columns, so text bounds are rejected.
    """
    if isinstance(value, str):
        raise TypeError(f"{name} values must be finite numbers or None")
    return value if value is None or isinstance(value, int) else float(value)


def _finite_real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be a finite number")
    return resolved


def _resolve_filtering(
    store: Any,
    assay: str,
    value: bool | Mapping[str, object],
) -> dict[str, Any]:
    if value is False:
        return {"enabled": False}
    if value is True:
        options: dict[str, Any] = {}
    elif isinstance(value, Mapping):
        options = dict(value)
    else:
        raise TypeError("filtering must be a mapping or bool")
    method = options.pop("method", "auto")
    if method == "auto":
        method = "mad"
    if method not in {"mad", "gaussian", "manual"}:
        raise ValueError(
            "filtering method must be 'auto', 'mad', 'gaussian', or 'manual'"
        )
    attrs = _column_sequence(
        options.pop("attrs", _default_filter_columns(store, assay)),
        "filtering attrs",
    )
    missing = [column for column in attrs if column not in store.cells.columns]
    if missing:
        raise KeyError(f"Filtering columns were not found: {missing!r}")
    if not attrs:
        raise ValueError(
            "Filtering was requested, but no QC columns were found; "
            "pass filtering=False to analyze the unfiltered cell selection"
        )
    if method == "manual":
        allowed = {"lows", "highs", "keep_bounds"}
        unknown = set(options) - allowed
        if unknown:
            raise ValueError(f"Unknown manual filtering options: {sorted(unknown)!r}")
        if "lows" not in options or "highs" not in options:
            raise ValueError("Manual filtering requires lows and highs")
        lows = list(options["lows"])
        highs = list(options["highs"])
        if len(lows) != len(attrs) or len(highs) != len(attrs):
            raise ValueError("Manual filtering bounds must align with attrs")
        keep_bounds = options.get("keep_bounds", False)
        validate_filter_bounds(lows, highs, keep_bounds=keep_bounds)
        return {
            "enabled": True,
            "method": "manual",
            "attrs": list(attrs),
            "lows": [_manual_bound(bound, "lows") for bound in lows],
            "highs": [_manual_bound(bound, "highs") for bound in highs],
            "keepBounds": keep_bounds,
        }
    allowed = {
        "min_p",
        "max_p",
        "sample_column",
        "n_mads",
        "min_cells_per_sample",
    }
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"Unknown automatic filtering options: {sorted(unknown)!r}")
    min_p = _finite_real(options.get("min_p", 0.01), "min_p")
    max_p = _finite_real(options.get("max_p", 0.99), "max_p")
    if not 0 < min_p < max_p < 1:
        raise ValueError("Automatic filtering requires 0 < min_p < max_p < 1")
    sample_column = options.get("sample_column")
    if sample_column is not None and (
        not isinstance(sample_column, str) or not sample_column
    ):
        raise TypeError("sample_column must be a non-empty string or None")
    if sample_column is not None and sample_column not in store.cells.columns:
        raise KeyError(f"Sample column {sample_column!r} was not found")
    n_mads = _finite_real(options.get("n_mads", 3.0), "n_mads")
    if n_mads <= 0:
        raise ValueError("n_mads must be finite and positive")
    min_cells = integer_argument(
        options.get("min_cells_per_sample", 20),
        "min_cells_per_sample",
        minimum=2,
    )
    if method == "mad" and (min_p != 0.01 or max_p != 0.99):
        raise ValueError(
            "min_p and max_p cannot be changed with method='mad'; "
            "use method='gaussian' for quantile bounds"
        )
    if method == "gaussian":
        if sample_column is not None:
            raise ValueError("Gaussian filtering does not support a sample source")
        if n_mads != 3.0 or min_cells != 20:
            raise ValueError(
                "n_mads and min_cells_per_sample apply only to method='mad'"
            )
    return {
        "enabled": True,
        "method": method,
        "attrs": list(attrs),
        "minP": min_p,
        "maxP": max_p,
        "sampleColumn": sample_column,
        "nMads": n_mads,
        "minCellsPerSample": min_cells,
    }


def resolve_pipeline_recipe(
    store: Any,
    *,
    assay: str | None,
    label: str | None,
    cell_key: str,
    filtering: bool | Mapping[str, object],
    harmony_batch_columns: Sequence[str] | None,
    hvg_count: int,
    pca_dims: int,
    neighbors_k: int,
    umap: bool,
    leiden: Mapping[str, object] | bool,
    cell_cycle: bool,
    paris: bool,
    doublets: bool,
    markers: bool,
    snapshot_columns: Sequence[str],
    params: Mapping[str, object] | None = None,
) -> ResolvedPipelineRecipe:
    assay_name = assay or store._defaultAssay
    if not isinstance(assay_name, str) or not assay_name:
        raise ValueError("No assay was provided and no default is configured")
    resolved_assay = store._get_assay(assay_name)
    if not isinstance(resolved_assay, RNAassay):
        raise TypeError("The basic pipeline requires an RNA assay")
    if label is not None and (not isinstance(label, str) or not label):
        raise TypeError("label must be a non-empty string or None")
    if not isinstance(cell_key, str) or not cell_key:
        raise TypeError("cell_key must be a non-empty string")
    if cell_key not in store.cells.columns:
        raise KeyError(f"Cell selection column {cell_key!r} was not found")
    if np.dtype(store.cells.get_dtype(cell_key)) != np.dtype(bool):
        raise TypeError("cell_key must identify a boolean metadata column")
    for flag, name in (
        (umap, "umap"),
        (cell_cycle, "cell_cycle"),
        (paris, "paris"),
        (doublets, "doublets"),
        (markers, "markers"),
    ):
        if not isinstance(flag, bool):
            raise TypeError(f"{name} must be a boolean")
    sections, species = _resolve_params(params)

    def section(stage: str, shortcut: str, changed: bool) -> _ParamSection | None:
        value = sections.get(stage)
        if value is not None and changed:
            raise ValueError(
                f"Set the {stage} stage with {shortcut}= or params[{stage!r}], not both"
            )
        return value

    def switch(value: _ParamSection | None, default: bool) -> tuple[bool, dict]:
        if value is None:
            return default, {}
        if isinstance(value, bool):
            return value, {}
        return True, dict(value)

    def setting(
        options: dict[str, Any], key: str, shortcut: str, value: Any, default: Any
    ) -> Any:
        if key not in options:
            return value
        if value != default:
            raise ValueError(f"Set {key} with {shortcut}= or params, not both")
        return options.pop(key)

    def settings(stage: str) -> dict[str, Any]:
        # Stages that always run accept only a settings mapping.
        value = sections.get(stage, {})
        assert isinstance(value, dict)
        return dict(value)

    filtering_section = section("filtering", "filtering", filtering is not True)
    if filtering_section is not None:
        filtering = filtering_section
    cell_cycle, cell_cycle_params = switch(
        section("cell_cycle", "cell_cycle", cell_cycle is not True), cell_cycle
    )
    umap, umap_params = switch(section("umap", "umap", umap is not True), umap)
    tsne, tsne_params = switch(sections.get("tsne"), False)
    paris, paris_params = switch(section("paris", "paris", paris is not True), paris)
    doublets, doublet_params = switch(
        section("doublets", "doublets", doublets is not True), doublets
    )
    markers, _marker_params = switch(
        section("markers", "markers", markers is not True), markers
    )
    membership_strength, _membership_params = switch(
        sections.get("membership_strength"), False
    )
    harmony_section = section(
        "harmony", "harmony_batch_columns", harmony_batch_columns is not None
    )
    harmony_params: dict[str, Any] = {}
    if harmony_section is False:
        harmony_batch_columns = None
    elif isinstance(harmony_section, dict):
        harmony_params = dict(harmony_section)
        if "batch_columns" not in harmony_params:
            raise ValueError("params['harmony'] needs batch_columns")
        harmony_batch_columns = harmony_params.pop("batch_columns")
    hvg_params = settings("hvg")
    hvg_count = setting(hvg_params, "top_n", "hvg_count", hvg_count, 1000)
    pca_params = settings("pca")
    pca_dims = setting(pca_params, "dims", "pca_dims", pca_dims, 21)
    neighbor_params = settings("neighbors")
    neighbors_k = setting(neighbor_params, "k", "neighbors_k", neighbors_k, 11)
    pca_dims = integer_argument(pca_dims, "pca_dims", minimum=0)
    if pca_dims == 0 and pca_params:
        raise ValueError("PCA settings need dims above 0; dims=0 skips PCA")

    leiden_section = section("leiden", "leiden", leiden is not True)
    leiden_params: dict[str, Any] = {}
    leiden_selected: str | None = None
    if isinstance(leiden_section, dict):
        leiden_params = dict(leiden_section)
        raw_partitions = leiden_params.pop("partitions", None)
        partitions = _resolve_leiden(
            True if raw_partitions is None else {"partitions": raw_partitions}
        )
        raw_selected = leiden_params.pop("selected", None)
        if raw_selected is not None:
            leiden_selected = str(canonical_resolution(raw_selected))
            if leiden_selected not in dict(partitions):
                raise ValueError("leiden selected must be one of the partitions")
    else:
        partitions = _resolve_leiden(
            leiden if leiden_section is None else leiden_section
        )
    if not partitions and (doublets or markers):
        raise ValueError("doublets and markers require at least one Leiden candidate")
    if not partitions and membership_strength:
        raise ValueError("membership_strength requires at least one Leiden candidate")
    umap_dims = integer_argument(
        umap_params.get("umap_dims", 2), "umap_dims", minimum=1
    )
    tsne_dims = integer_argument(
        tsne_params.get("tsne_dims", 2), "tsne_dims", minimum=1
    )
    snapshots = _column_sequence(snapshot_columns, "snapshot_columns")
    result_fields = {
        "highly_variable_features",
        "s_score",
        "g2m_score",
        "cell_cycle_phase",
        "umap_1",
        "umap_2",
        *(f"umap_{index}" for index in range(1, umap_dims + 1)),
        *(f"tsne_{index}" for index in range(1, tsne_dims + 1)),
        "paris",
        "clusters",
        "membership_strength",
        "doublet_score",
        *(f"leiden_{key}" for key, _value in partitions),
    }
    collisions = set(snapshots) & ({"I", "ids", "names"} | result_fields)
    if collisions:
        raise ValueError(
            f"snapshot_columns collide with reserved run fields: {sorted(collisions)!r}"
        )
    missing_snapshots = [
        column for column in snapshots if column not in store.cells.columns
    ]
    if missing_snapshots:
        raise KeyError(f"Snapshot columns were not found: {missing_snapshots!r}")
    harmony_columns = (
        ()
        if harmony_batch_columns is None
        else _column_sequence(harmony_batch_columns, "harmony_batch_columns")
    )
    if harmony_batch_columns is not None and not harmony_columns:
        raise ValueError("harmony_batch_columns must not be empty")
    missing_harmony = [
        column for column in harmony_columns if column not in store.cells.columns
    ]
    if missing_harmony:
        raise KeyError(f"Harmony columns were not found: {missing_harmony!r}")
    filtering_config = _resolve_filtering(store, assay_name, filtering)
    filter_columns = tuple(filtering_config.get("attrs", ()))
    sample_column = filtering_config.get("sampleColumn")
    if isinstance(sample_column, str):
        filter_columns = (*filter_columns, sample_column)
    cell_snapshot_columns = tuple(
        dict.fromkeys(("names", *filter_columns, *harmony_columns, *snapshots))
    )
    stage_order = (
        "input_snapshot",
        "filtering",
        "cell_cycle",
        "highly_variable_features",
        "normalization",
        "pca",
        "harmony",
        "ann_index",
        "neighbors",
        "connectivity",
        "embedding_initialization",
        "umap",
        *(f"leiden_{key}" for key, _value in partitions),
        "paris",
        "cluster_selection",
        "membership_strength",
        "tsne",
        "doublet_graph",
        "doublets",
        "markers",
    )
    return ResolvedPipelineRecipe(
        assay=assay_name,
        label=label,
        cell_key=cell_key,
        filtering=filtering_config,
        harmony_batch_columns=harmony_columns,
        hvg_count=integer_argument(hvg_count, "hvg_count", minimum=1),
        pca_dims=pca_dims,
        neighbors_k=integer_argument(neighbors_k, "neighbors_k", minimum=1),
        umap=umap,
        leiden_partitions=partitions,
        cell_cycle=cell_cycle,
        paris=paris,
        doublets=doublets,
        markers=markers,
        snapshot_columns=snapshots,
        cell_snapshot_columns=cell_snapshot_columns,
        stage_order=stage_order,
        tsne=tsne,
        membership_strength=membership_strength,
        leiden_selected=leiden_selected,
        species=species,
        stage_params=MappingProxyType(
            {
                stage: MappingProxyType(values)
                for stage, values in (
                    ("cell_cycle", cell_cycle_params),
                    ("hvg", hvg_params),
                    ("normalization", settings("normalization")),
                    ("pca", pca_params),
                    ("harmony", harmony_params),
                    ("ann_index", settings("ann_index")),
                    ("neighbors", neighbor_params),
                    ("connectivity", settings("connectivity")),
                    (
                        "embedding_initialization",
                        settings("embedding_initialization"),
                    ),
                    ("umap", umap_params),
                    ("tsne", tsne_params),
                    ("leiden", leiden_params),
                    ("paris", paris_params),
                    ("doublets", doublet_params),
                )
                if values
            }
        ),
    )

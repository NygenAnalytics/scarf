from collections.abc import Iterable
from numbers import Real
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from ...assay import ATACassay, RNAassay
from ...assay.feature_summary import (
    ensure_feature_summary,
    feature_summary_selected_count,
    feature_summary_values,
)
from ...graph.feature_projection import resolve_graph_assay_inputs
from ...graph.state import read_assay_state, resolve_graph_selection
from ...quality_control.cell_cycle import assign_cell_cycle_phase
from ...quality_control.filtering import (
    _metric_policy,
    _sample_aware_mad_mask,
    _validated_sample_labels,
    _validated_work_scale,
    gaussian_quantile_bounds,
)
from ...quality_control.hto import _hto_demux_method, hto_demux
from ...metadata.artifacts import (
    artifact_values,
    categorical_display,
    column_display,
    continuous_display,
    link_cell_data_column,
    plan_cell_data_artifact,
    write_cell_data_artifact,
)
from ...metadata.arguments import (
    CellCycleArguments,
    DoubletScoreArguments,
    HtoIdentityArguments,
    PrevalentPeakArguments,
)
from ...storage.artifacts import (
    artifact_path,
    canonical_bytes,
    fingerprint_array,
    fingerprint_strings,
)
from ...storage.feature_selection import (
    publish_feature_selection_alias,
    validate_feature_selection_label,
)
from ...storage.refs import ArtifactRef
from ...storage.types import as_zarr_group
from ...utils.compute import controlled_compute
from ...utils.logging import logger
from ...storage.feature_selection import (
    _feature_selection_plan,
    _ordered_feature_ids_fingerprint,
    _write_feature_selection,
)

if TYPE_CHECKING:
    from ...storage.profiles import ZarrLocation
    from ..mapping_datastore import MappingDatastore as _QualityControlOperationsBase
else:
    _QualityControlOperationsBase = object


class _QualityControlOperationsMixin(_QualityControlOperationsBase):
    if TYPE_CHECKING:

        def _create_temporary_datastore(
            self,
            zarr_loc: ZarrLocation,
            *,
            default_assay: str,
            assay_types: dict[str, str],
            nthreads: int,
        ) -> _QualityControlOperationsBase: ...

    def filter_cells(
        self,
        attrs: Iterable[str],
        lows: Iterable[int],
        highs: Iterable[int],
        reset_previous: bool = False,
        keep_bounds: bool = False,
        invalidate_cache: bool = False,
    ) -> None:
        """Filter cells based on the cell metadata column values. Filtering
        triggers `update` method on  'I' column of cell metadata which uses
        'and' operation. This means that cells that are not within the
        filtering thresholds will have value set as False in 'I' column of cell
        metadata table. When performing filtering repeatedly, the cells that
        were previously filtered out remain filtered out and 'I' column is
        updated only for those cells that are filtered out due to the latest
        filtering attempt.

        Args:
            attrs: Names of columns to be used for filtering
            lows: Lower bounds of thresholds for filtering. Should be in same order as the names in `attrs` parameter
            highs: Upper bounds of thresholds for filtering. Should be in same order as the names in `attrs` parameter
            reset_previous: If True, then results of previous filtering will be undone completely.
                            (Default value: False)
            keep_bounds: If True, then the boundary values are retained and not filtered out (Default value: False)

        Returns:
        """
        attrs = list(attrs)
        lows = list(lows)
        highs = list(highs)
        input_fingerprints = {}
        new_bool = np.ones(self.cells.N).astype(bool)
        for i, j, k in zip(attrs, lows, highs, strict=False):
            # Checking here to avoid hard error from metadata class
            if i not in self.cells.columns:
                logger.warning(
                    f"{i} not found in cell metadata. Will ignore {i} for filtering"
                )
                continue
            if j is None:
                j = -np.inf
            if k is None:
                k = np.inf
            x = self.cells.sift(i, j, k, keep_bounds=keep_bounds)
            values = np.asarray(self.cells.fetch_all(i))
            input_fingerprints[i] = (
                fingerprint_strings(values)
                if values.dtype.kind in {"O", "S", "U"}
                else fingerprint_array(values)
            )
            new_bool = new_bool & x
        if reset_previous:
            self.cells.reset_key(key="I")
        self.cells.update_key(new_bool, key="I")
        self._record_cell_selection(
            column="I",
            operation="filter_cells",
            parameters={
                "attrs": attrs,
                "lows": lows,
                "highs": highs,
                "reset_previous": reset_previous,
                "keep_bounds": keep_bounds,
            },
            inputs={"metadata_fingerprints": input_fingerprints},
            invalidate_cache=invalidate_cache,
        )
        remaining = int(np.asarray(self.cells.fetch_all("I"), dtype=bool).sum())
        logger.info(f"Cell filtering retained {remaining}/{self.cells.N} cells")

    def auto_filter_cells(
        self,
        attrs: Iterable[str] | None = None,
        min_p: float = 0.01,
        max_p: float = 0.99,
        show_qc_plots: bool = True,
        invalidate_cache: bool = False,
        sample_column: str | None = None,
        n_mads: float = 3.0,
        min_cells_per_sample: int = 20,
    ) -> None:
        """Automatically filter cells based on columns of the cell metadata
        table.

        By default this is a wrapper around ``filter_cells`` that determines the
        thresholds for each column. It models a normal distribution centered on
        the column median and using the column standard deviation, then
        evaluates its quantiles at ``min_p`` and ``max_p``.

        When ``sample_column`` is supplied, thresholds are instead calculated
        independently within each sample using median absolute deviation (MAD).
        ``n_mads`` controls that path. ``min_p`` and ``max_p`` remain
        global-Gaussian parameters and must stay at their defaults when
        ``sample_column`` is set.

        Args:
            attrs: Column names to be used for filtering.
            min_p: Quantile used for the lower threshold (Gaussian path only).
            max_p: Quantile used for the upper threshold (Gaussian path only).
            show_qc_plots: Show pre-filtering and post-filtering distributions
                for the columns used.
            sample_column: Optional cell-metadata column with sample labels.
                When set, MAD bounds are calculated within each sample.
            n_mads: Number of scaled MADs used for per-sample bounds.
            min_cells_per_sample: Samples with fewer active cells than this are
                retained without MAD filtering and emit a warning.

        Returns:
            None
        """
        from ...plotting import distribution

        if attrs is None:
            attrs = []
            for i in ["nCounts", "nFeatures", "percentMito", "percentRibo"]:
                i = f"{self._defaultAssay}_{i}"
                if i in self.cells.columns:
                    attrs.append(i)

        attrs_list = list(attrs)
        if sample_column is not None:
            self._auto_filter_cells_sample_mad(
                attrs=attrs_list,
                min_p=min_p,
                max_p=max_p,
                show_qc_plots=show_qc_plots,
                invalidate_cache=invalidate_cache,
                sample_column=sample_column,
                n_mads=n_mads,
                min_cells_per_sample=min_cells_per_sample,
            )
            return

        attrs_used: list[str] = []
        lower_bounds: list[int] = []
        upper_bounds: list[int] = []
        resolved_bounds: dict[str, dict[str, float]] = {}
        for i in attrs_list:
            if i not in self.cells.columns:
                logger.warning(
                    f"{i} not found in cell metadata. Will ignore {i} for filtering"
                )
                continue
            a = self.cells.fetch_all(i)
            low, high = gaussian_quantile_bounds(a, min_p, max_p)
            resolved_bounds[i] = {"low": float(low), "high": float(high)}
            attrs_used.append(i)
            lower_bounds.append(cast(int, low))
            upper_bounds.append(cast(int, high))

        if attrs_used:
            self.filter_cells(
                attrs=attrs_used,
                lows=lower_bounds,
                highs=upper_bounds,
                invalidate_cache=False,
            )

        self._record_cell_selection(
            column="I",
            operation="auto_filter_cells",
            parameters={
                "attrs": attrs_used,
                "min_p": min_p,
                "max_p": max_p,
                "resolved_bounds": resolved_bounds,
            },
            inputs={},
            invalidate_cache=invalidate_cache,
        )

        if show_qc_plots and attrs_used:
            # Match the previous plot_cells_dists contract: pre uses every cell,
            # post uses the filtered active set under cell key I.
            distribution(
                self,
                keys=attrs_used,
                cell_key=None,
                color="steelblue",
                title="Pre-filtering distribution",
                show=True,
            )
            distribution(
                self,
                keys=attrs_used,
                cell_key="I",
                color="coral",
                title="Post-filtering distribution",
                show=True,
            )

    def _auto_filter_cells_sample_mad(
        self,
        *,
        attrs: list[str],
        min_p: float,
        max_p: float,
        show_qc_plots: bool,
        invalidate_cache: bool,
        sample_column: str,
        n_mads: float,
        min_cells_per_sample: int,
    ) -> None:
        from ...plotting import distribution

        if min_p != 0.01 or max_p != 0.99:
            raise ValueError(
                "min_p and max_p apply only to the global Gaussian path. "
                "Leave them at their defaults (0.01 and 0.99) when "
                "sample_column is set, and use n_mads to control MAD bounds"
            )
        if isinstance(n_mads, bool) or not isinstance(n_mads, Real):
            raise TypeError("n_mads must be a positive number")
        resolved_n_mads = float(n_mads)
        if not np.isfinite(resolved_n_mads) or resolved_n_mads <= 0:
            raise ValueError("n_mads must be finite and greater than 0")
        if (
            not isinstance(min_cells_per_sample, int)
            or isinstance(min_cells_per_sample, bool)
            or min_cells_per_sample < 2
        ):
            raise ValueError("min_cells_per_sample must be an integer >= 2")
        if sample_column not in self.cells.columns:
            raise ValueError(
                f"sample_column '{sample_column}' not found in cell metadata"
            )

        active = np.asarray(self.cells.fetch_all("I"), dtype=bool)
        sample_labels = np.asarray(self.cells.fetch_all(sample_column))
        sample_labels = _validated_sample_labels(
            sample_labels,
            active,
            label_name=f"sample_column '{sample_column}'",
        )
        active_labels = sample_labels[active]
        if active_labels.size == 0:
            raise ValueError("No active cells are available for sample-aware filtering")

        attrs_used: list[str] = []
        values_by_attr: dict[str, np.ndarray] = {}
        for attr in attrs:
            if attr not in self.cells.columns:
                logger.warning(
                    f"{attr} not found in cell metadata. Will ignore {attr} for filtering"
                )
                continue
            values = np.asarray(self.cells.fetch_all(attr), dtype=float)
            _validated_work_scale(
                values[active],
                attr=attr,
                transform=_metric_policy(attr)["transform"],
            )
            attrs_used.append(attr)
            values_by_attr[attr] = values

        parameters: dict[str, Any] = {
            "attrs": attrs_used,
            "sample_column": sample_column,
            "n_mads": resolved_n_mads,
            "min_cells_per_sample": int(min_cells_per_sample),
            "resolved_bounds": {},
        }

        mad_provenance = None
        if attrs_used:
            keep, mad_provenance = _sample_aware_mad_mask(
                values_by_attr=values_by_attr,
                sample_labels=sample_labels,
                active=active,
                n_mads=resolved_n_mads,
                min_cells_per_sample=int(min_cells_per_sample),
                attrs=attrs_used,
            )
            parameters.update(
                {
                    "mad_scale": mad_provenance["mad_scale"],
                    "metric_policies": mad_provenance["metric_policies"],
                    "sample_sizes": mad_provenance["sample_sizes"],
                    "skip_reasons": mad_provenance["skip_reasons"],
                    "resolved_bounds": mad_provenance["resolved_bounds"],
                }
            )
        fingerprint_inputs: dict[str, Any] = {
            "sample_assignments_fingerprint": fingerprint_strings(active_labels),
            "qc_metric_fingerprints": {
                attr: fingerprint_array(values_by_attr[attr][active])
                for attr in attrs_used
            },
        }
        canonical_bytes(
            {
                "operation": "auto_filter_cells",
                "parameters": parameters,
                "inputs": fingerprint_inputs,
            }
        )
        prior_selection = self._ensure_cell_selection("I")
        inputs: dict[str, Any] = {
            "prior_cell_selection": prior_selection,
            **fingerprint_inputs,
        }
        canonical_bytes(
            {
                "operation": "auto_filter_cells",
                "parameters": parameters,
                "inputs": inputs,
            }
        )

        if attrs_used:
            assert mad_provenance is not None
            for message in mad_provenance["warnings"]:
                logger.warning(message)
            self.cells.update_key(keep, key="I")
            remaining = int(np.asarray(self.cells.fetch_all("I"), dtype=bool).sum())
            logger.info(f"Cell filtering retained {remaining}/{self.cells.N} cells")

        self._record_cell_selection(
            column="I",
            operation="auto_filter_cells",
            parameters=parameters,
            inputs=inputs,
            invalidate_cache=invalidate_cache,
        )

        if show_qc_plots and attrs_used:
            distribution(
                self,
                keys=attrs_used,
                cell_key=None,
                color="steelblue",
                title="Pre-filtering distribution",
                show=True,
            )
            distribution(
                self,
                keys=attrs_used,
                cell_key="I",
                color="coral",
                title="Post-filtering distribution",
                show=True,
            )

    def mark_hto_identities(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        label: str = "Hashtag_identity",
        random_seed: int = 0,
        invalidate_cache: bool = False,
    ) -> str:
        """Assign HTO hashtag identities to cells using demultiplexing.

        Args:
            from_assay: HTO assay name (default: ``'HTO'``).
            cell_key: Boolean cell metadata column selecting cells (default: latest for assay).
            label: Column name to store identities in cell metadata.
            random_seed: Seed used for HTO demultiplexing.
            invalidate_cache: Recompute even when matching provenance exists.

        Returns:
            Name of the cell metadata column containing HTO identities.
        """
        if from_assay is None:
            from_assay = "HTO"
        if cell_key is None:
            cell_key = self._get_latest_cell_key(from_assay)
        assay = self._get_assay(from_assay)
        if isinstance(random_seed, bool) or not isinstance(random_seed, int):
            raise TypeError("random_seed must be an integer")
        n_cells = len(self.cells.active_index(cell_key))
        required_cells = assay.feats.N + 1
        if n_cells < required_cells:
            raise ValueError(
                f"HTO demultiplexing requires at least {required_cells} selected cells"
            )
        selection = self._ensure_cell_selection(cell_key)
        arguments = HtoIdentityArguments(
            cell_selection=selection,
            feature_ids_fingerprint=fingerprint_strings(
                np.asarray(assay.feats.fetch_all("ids"))
            ),
            method=_hto_demux_method(),
            random_seed=random_seed,
            from_assay=from_assay,
            cell_key=cell_key,
            label=label,
            invalidate_cache=invalidate_cache,
        )
        record = arguments.to_record()
        planned = plan_cell_data_artifact(
            self.zw,
            scope="assay",
            assay=from_assay,
            kind=arguments.artifact_kind,
            operation=arguments.operation,
            parameters=record.parameters,
            inputs=record.inputs,
            execution_options=record.execution_options,
            cell_selection=selection,
            arrays={"values": ((n_cells,), None)},
            invalidate_cache=invalidate_cache,
        )
        preserved_display = column_display(self.zw, label)
        if planned.reused:
            artifact_group = as_zarr_group(
                self.zw[artifact_path(planned.ref)],
                name=planned.ref.artifact_id,
            )
            values = artifact_values(artifact_group, "values")
            self.cells.insert(
                column_name=label,
                values=values,
                overwrite=True,
                key=cell_key,
            )
            link_cell_data_column(
                self.zw,
                label,
                planned.ref,
                value_name="values",
                default_display=categorical_display(values),
                preserved_display=preserved_display,
            )
            return label
        counts = controlled_compute(
            assay.rawData[self.cells.fetch_all(cell_key)], self.nthreads
        )
        hto_idents = hto_demux(
            pd.DataFrame(counts, columns=assay.feats.fetch_all("ids")),
            random_seed=random_seed,
        )
        values = np.asarray(hto_idents.values)
        write_cell_data_artifact(
            self.zw,
            planned,
            {"values": values},
        )
        self.cells.insert(
            column_name=label,
            values=values,
            overwrite=True,
            key=cell_key,
        )
        link_cell_data_column(
            self.zw,
            label,
            planned.ref,
            value_name="values",
            default_display=categorical_display(values),
            preserved_display=preserved_display,
        )
        return label

    def run_doublet_detection(
        self,
        cluster_key: str,
        from_assay: str | None = None,
        cell_key: str | None = None,
        cluster_sample_fraction: float = 0.05,
        max_cells_per_cluster: int = 100,
        simulation_ratio: float = 1.0,
        heterotypic_fraction: float = 0.8,
        save_k: int = 5,
        smoothing_t: int = 2,
        normalize_scores: bool = True,
        label: str = "doublet_score",
        random_seed: int = 4444,
        invalidate_cache: bool = False,
        *,
        graph: ArtifactRef | None = None,
    ) -> str:
        """Flag potential doublets by simulating and mapping synthetic doublets.

        Synthetic doublets are simulated by summing the raw counts of pairs of
        observed cells drawn from a per-cluster subsample, with a tunable bias
        toward cross-cluster (heterotypic) pairs. The simulated profiles are
        projected onto the existing reference graph with `run_mapping`, and each
        reference cell is scored by how frequently it appears among the nearest
        neighbours of the simulated doublets (`get_mapping_score`). The score is
        then diffused over the KNN graph using the same operator as
        `get_imputed`. The final per-cell score is written to cell metadata so
        that users can threshold and filter doublets themselves. This is a
        graph-native adaptation of the Scrublet and DoubletFinder approach.

        Args:
            cluster_key: Cell metadata column with cluster or group labels used
                to stratify the candidate pool (for example ``'RNA_cluster'``).
            from_assay: Assay to use. Defaults to the configured default assay. Only
                RNAassay type assays are supported.
            cell_key: Cell key matching the desired graph (default: ``'I'``).
            graph: Connectivity-map or integrated-graph artifact. The assay's
                current connectivity map is used when omitted.
            cluster_sample_fraction: Fraction of cells sampled from each cluster
                to build the candidate pool. (Default value: 0.05)
            max_cells_per_cluster: Cap on the number of cells sampled per cluster.
                (Default value: 100)
            simulation_ratio: Number of simulated doublets expressed as a
                multiple of the number of reference cells. (Default value: 1.0)
            heterotypic_fraction: Fraction of simulated doublets forced to be
                cross-cluster. Set to 0 to disable the bias. (Default value: 0.8)
            save_k: Number of reference neighbours stored per simulated doublet.
                (Default value: 5)
            smoothing_t: Diffusion power used to smoothen scores over the graph,
                same as the ``t`` parameter of `get_imputed`. (Default value: 2)
            normalize_scores: If True, the final score is min-max scaled to the
                0-1 range for interpretability. (Default value: True)
            label: Base name for the score column in cell metadata. The assay
                name (and cell key when not ``'I'``) is prepended.
                (Default value: 'doublet_score')
            random_seed: Seed for reproducible sampling. (Default value: 4444)

        Returns:
            Name of the cell-metadata column containing the final scores.
        """
        import shutil
        import tempfile

        from scipy.sparse import csr_matrix

        from ...quality_control.doublets import (
            sample_cluster_pool,
            simulate_doublet_pairs,
            write_doublet_target_zarr,
        )

        graph_selection = resolve_graph_selection(
            self,
            graph,
            from_assay=from_assay,
            cell_key=cell_key,
        )
        from_assay = graph_selection.from_assay
        cell_key = graph_selection.cell_key
        connectivity = graph_selection.graph_ref
        source_assay = self._get_assay(from_assay)
        if type(source_assay) != RNAassay:  # noqa: E721
            raise TypeError(
                "ERROR: Doublet detection is only supported for RNAassay type assays. "
                f"The provided assay is {type(source_assay)} type"
            )
        if cluster_key not in self.cells.columns:
            raise ValueError(
                f"ERROR: `cluster_key` {cluster_key} not found in cell metadata. Provide a column "
                f"with cluster or group labels, for example '{from_assay}_cluster'"
            )
        selection = self._ensure_cell_selection(cell_key)
        cluster_input = self._resolve_cell_data_provenance_input(
            cluster_key,
            cell_key=cell_key,
        )
        state = read_assay_state(self.zw, from_assay)
        connectivity_status = self._require_complete_artifact(
            connectivity,
            connectivity.kind,
            assay=(from_assay if connectivity.scope == "assay" else None),
        )
        if connectivity.kind == "connectivity_map" and (
            connectivity_status.operation != "build_connectivity_map"
        ):
            raise ValueError(
                "Doublet detection requires a build_connectivity_map artifact. "
                "Rebuild and select the RNA graph before running doublet detection."
            )
        lineage = resolve_graph_assay_inputs(
            self.zw,
            connectivity,
            from_assay,
        )
        neighbors = lineage.neighbors
        graph_cell_selection = self._graph_cell_selection(connectivity)
        if not self._selection_artifacts_match(graph_cell_selection, selection):
            raise ValueError("cell_key does not match the graph cell selection")
        coordinates = lineage.coordinates
        if coordinates.kind != "reduction":
            raise ValueError(
                "Doublet detection requires a plain scaled-PCA mapping reference. "
                "The selected neighbors use batch-corrected or Symphony coordinates. "
                "Rebuild and select an uncorrected scaled-PCA connectivity graph first."
            )
        n_active = len(self.cells.active_index(cell_key))
        arguments = DoubletScoreArguments(
            clusters=cluster_input,
            connectivity_map=connectivity,
            neighbors=neighbors,
            cluster_sample_fraction=cluster_sample_fraction,
            max_cells_per_cluster=max_cells_per_cluster,
            simulation_ratio=simulation_ratio,
            heterotypic_fraction=heterotypic_fraction,
            save_k=save_k,
            smoothing_t=smoothing_t,
            normalize_scores=normalize_scores,
            random_seed=random_seed,
            from_assay=from_assay,
            cell_key=cell_key,
            label=label,
            invalidate_cache=invalidate_cache,
        )
        record = arguments.to_record()
        planned = plan_cell_data_artifact(
            self.zw,
            scope="assay",
            assay=from_assay,
            kind=arguments.artifact_kind,
            operation=arguments.operation,
            parameters=record.parameters,
            inputs=record.inputs,
            execution_options=record.execution_options,
            cell_selection=selection,
            arrays={"values": ((n_active,), "f")},
            invalidate_cache=invalidate_cache,
        )
        final_col = self._col_renamer(from_assay, cell_key, label)
        preserved_display = column_display(self.zw, final_col)
        if planned.reused:
            artifact_group = as_zarr_group(
                self.zw[artifact_path(planned.ref)],
                name=planned.ref.artifact_id,
            )
            scores = artifact_values(artifact_group, "values")
            self.cells.insert(
                final_col,
                scores,
                key=cell_key,
                overwrite=True,
            )
            link_cell_data_column(
                self.zw,
                final_col,
                planned.ref,
                value_name="values",
                default_display=continuous_display(scores),
                preserved_display=preserved_display,
            )
            return final_col

        reference = None
        feature_selection = lineage.feature_selection
        if feature_selection is None:
            raise ValueError(
                "Doublet detection requires normalized feature-selection ancestry"
            )
        named_reference = (
            state.named_results.get("mapping_reference") if state is not None else None
        )
        if named_reference is not None:
            try:
                candidate = self.get_mapping_reference(
                    named_reference,
                    from_assay=from_assay,
                )
            except (KeyError, RuntimeError, TypeError, ValueError):
                pass
            else:
                if (
                    candidate.neighbors == neighbors
                    and candidate.assay_name == from_assay
                    and candidate.cell_key == cell_key
                    and candidate.feature_selection == feature_selection
                    and candidate.method == "pca"
                    and candidate.batch_correction is None
                    and candidate.symphony_state is None
                ):
                    reference = candidate
        if reference is None:
            reference = self.build_mapping_reference(neighbors)
            if state is not None and state.neighbors == neighbors:
                named_results = dict(state.named_results)
                named_results["mapping_reference"] = reference.ref
                self._publish_current_artifact(
                    state.connectivity_map or neighbors,
                    update_state=True,
                    named_results=named_results,
                )
        if (
            reference.neighbors != neighbors
            or reference.assay_name != from_assay
            or reference.cell_key != cell_key
            or reference.feature_selection != feature_selection
            or reference.method != "pca"
            or reference.batch_correction is not None
            or reference.symphony_state is not None
        ):
            raise RuntimeError(
                "The prepared plain mapping reference does not match the selected "
                "RNA graph chain"
            )

        rng = np.random.default_rng(random_seed)
        active_idx = self.cells.active_index(cell_key)
        clusters = self.cells.fetch(cluster_key, key=cell_key)

        pool_positions = sample_cluster_pool(
            clusters, cluster_sample_fraction, max_cells_per_cluster, rng
        )
        pool_clusters = np.asarray(clusters)[pool_positions]
        pool_raw_rows = np.asarray(active_idx)[pool_positions]
        logger.debug(
            f"Sampled {len(pool_positions)} cells across "
            f"{len(np.unique(pool_clusters))} clusters to seed doublet simulation"
        )

        pool_counts = controlled_compute(
            source_assay.rawData[pool_raw_rows, :], self.nthreads
        )
        pool_csr = csr_matrix(pool_counts)

        n_sim = max(1, int(round(simulation_ratio * n_active)))
        left, right = simulate_doublet_pairs(
            pool_clusters, n_sim, heterotypic_fraction, rng
        )
        sim_counts = (pool_csr[left] + pool_csr[right]).tocsr()
        logger.debug(f"Simulated {n_sim} synthetic doublets")

        temp_dir = tempfile.mkdtemp(prefix="scarf_doublet_")
        target_name = f"_doublet_sim_{from_assay}"
        try:
            write_doublet_target_zarr(
                zarr_loc=temp_dir,
                assay_name=from_assay,
                sim_counts=sim_counts,
                feat_ids=source_assay.feats.fetch_all("ids"),
                feat_names=source_assay.feats.fetch_all("names"),
                dtype=str(source_assay.rawData.dtype),
                mem_budget=self.memoryBytes,
                nthreads=self.nthreads,
                profile="fast_local",
            )
            target_ds = self._create_temporary_datastore(
                temp_dir,
                default_assay=from_assay,
                assay_types={from_assay: "RNA"},
                nthreads=self.nthreads,
            )
            result = target_ds.run_mapping(
                reference,
                mapping_name=target_name,
                query_assay=from_assay,
                save_k=save_k,
            )

            try:
                _, raw_scores = next(
                    target_ds.get_mapping_score(
                        result,
                        reference=reference,
                        log_transform=True,
                    )
                )
            except StopIteration:
                raise RuntimeError(
                    "ERROR: Mapping scores could not be computed for simulated doublets"
                ) from None
            raw_scores = np.asarray(raw_scores)
            if raw_scores.shape != (n_active,):
                raise RuntimeError(
                    "Doublet mapping scores do not match the selected reference cells"
                )

            temp_col = self._col_renamer(from_assay, cell_key, f"{label}__raw")
            self.cells.insert(temp_col, raw_scores, key=cell_key, overwrite=True)
            try:
                smoothed = self.get_imputed(
                    temp_col,
                    connectivity,
                    from_assay=from_assay,
                    cell_key=cell_key,
                    t=smoothing_t,
                )
            finally:
                self.cells.drop(temp_col)

            scores = np.asarray(smoothed, dtype=float)
            if normalize_scores:
                lo, hi = scores.min(), scores.max()
                scores = (scores - lo) / (hi - lo) if hi > lo else np.zeros_like(scores)
            write_cell_data_artifact(
                self.zw,
                planned,
                {"values": scores},
            )
            self.cells.insert(final_col, scores, key=cell_key, overwrite=True)
            link_cell_data_column(
                self.zw,
                final_col,
                planned.ref,
                value_name="values",
                default_display=continuous_display(scores),
                preserved_display=preserved_display,
            )
            logger.info(
                f"Stored doublet scores in '{final_col}' using "
                f"{n_sim} synthetic doublets"
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        return final_col

    def mark_prevalent_peaks(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        top_n: int = 10000,
        label: str = "prevalent_peaks",
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        """Feature selection method for ATACassay type assays.

        This method first calculates prevalence of each peak by computing sum of TF-IDF normalized values for each peak
        and then marks `top_n` peaks with the highest prevalence as prevalent peaks.

        Args:
            from_assay: Assay to use for graph creation. If no value is provided then `defaultAssay` will be used
            cell_key: Cells to use for selection of most prevalent peaks. By default, all cells with True value in
                      'I' will be used. The provided value for `cell_key` should be a column in cell metadata table
                      with boolean values.
            top_n: Number of top prevalent peaks to be selected. (Default: 10000)
            label: Feature-selection label to publish. (Default: 'prevalent_peaks')

        Returns:
            The persisted prevalent-peak feature-selection artifact.
        """
        if self.zarr_mode != "r+":
            raise PermissionError(
                "mark_prevalent_peaks requires a DataStore opened with zarr_mode='r+'"
            )
        validate_feature_selection_label(label)
        if cell_key is None:
            cell_key = "I"
        assay = self._get_assay(from_assay)
        if type(assay) != ATACassay:  # noqa: E721
            raise TypeError(
                f"ERROR: This method of feature selection can only be applied to ATACassay type of assay. "
                f"The provided assay is {type(assay)} type"
            )
        cast(Any, self)._ensure_all_features(assay)
        cell_selection = self._ensure_cell_selection(cell_key)
        summary_ref = ensure_feature_summary(
            self.zw,
            assay,
            cell_selection,
            invalidate_cache=invalidate_cache,
        )
        arguments = PrevalentPeakArguments(
            feature_summary=summary_ref,
            top_n=top_n,
            from_assay=assay.name,
            cell_key=cell_key,
            label=label,
            invalidate_cache=invalidate_cache,
        )
        record = arguments.to_record()
        feature_ids_fingerprint = _ordered_feature_ids_fingerprint(assay)
        planned = _feature_selection_plan(
            self.zw,
            assay=assay.name,
            n_features=assay.feats.N,
            ordered_feature_ids_fingerprint=feature_ids_fingerprint,
            operation=arguments.operation,
            parameters=record.parameters,
            inputs=record.inputs,
            execution_options=record.execution_options,
            invalidate_cache=invalidate_cache,
        )
        if not planned.reused:
            n_selected = feature_summary_selected_count(
                self.zw,
                cell_selection,
                n_cells=assay.cells.N,
            )
            summary = feature_summary_values(
                self.zw,
                summary_ref,
                n_selected=n_selected,
            )
            values = assay._prevalent_peak_mask(summary["prevalence"], top_n)
            _write_feature_selection(
                self.zw,
                planned,
                ordered_feature_ids_fingerprint=feature_ids_fingerprint,
                payload={"values": values},
            )
        publish_feature_selection_alias(
            self.zw,
            assay.name,
            label,
            planned.ref,
        )
        return planned.ref

    def run_cell_cycle_scoring(
        self,
        from_assay: str | None = None,
        cell_key: str | None = None,
        s_genes: list[str] | None = None,
        g2m_genes: list[str] | None = None,
        n_bins: int = 50,
        rand_seed: int = 4466,
        s_score_label: str = "S_score",
        g2m_score_label: str = "G2M_score",
        phase_label: str = "cell_cycle_phase",
        invalidate_cache: bool = False,
    ) -> None:
        """Computes S and G2M phase scores by taking into account the average
        expression of S and G2M phase genes respectively. Following steps are
        taken for each phase:

        - Average expression of all the genes in across `cell_key` cells is calculated
        - The log average expression is divided in `n_bins` bins
        - A control set of genes is identified by sampling genes from same expression bins where phase's genes are present.
        - The average expression of phase genes (Ep) and control genes (Ec) is calculated per cell.
        - A phase score is calculated as ``Ep - Ec``.
        - G1 is assigned when both scores are negative.
        - G2M is assigned when the G2M score exceeds the S score.
        - S is assigned otherwise, including tied non-negative scores.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            s_genes: A list of S phase genes. If not provided then Scarf loads pre-saved genes accessible at
                     `scarf.quality_control.s_phase_genes`
            g2m_genes: A list of G2M phase genes. If not provided then Scarf loads pre-saved genes accessible at
                     `scarf.quality_control.g2m_phase_genes`
            n_bins: Number of bins into which average expression of genes is divided.
            rand_seed: A random values to set seed while sampling cells from a cluster randomly. (Default value: 4466)
            s_score_label: A base label for saving the S phase scores into a cell metadata column
                           (Default value: 'S_score')
            g2m_score_label: A base label for saving the G2M phase scores into a cell metadata column
                           (Default value: 'G2M_score')
            phase_label: A base label for saving the inferred cell cycle phase into a cell metadata column
                           (Default value: 'cell_cycle_phase')

        Returns: None
        """
        if self.zarr_mode != "r+":
            raise PermissionError(
                "Cell-cycle scoring requires a DataStore opened with zarr_mode='r+'"
            )
        if from_assay is None:
            from_assay = self._defaultAssay
        assay = self._get_assay(from_assay)
        if not isinstance(assay, RNAassay):
            raise TypeError(
                "Cell-cycle scoring can only be applied to an RNAassay; "
                f"received {type(assay).__name__}"
            )
        from ...graph.state import read_assay_state_document

        read_assay_state_document(self.zw, assay.name)
        if cell_key is None:
            cell_key = "I"
        if s_genes is None:
            from ...quality_control.cell_cycle_genes import s_phase_genes

            s_genes = list(s_phase_genes)
        if g2m_genes is None:
            from ...quality_control.cell_cycle_genes import g2m_phase_genes

            g2m_genes = list(g2m_phase_genes)
        control_size = min(len(s_genes), len(g2m_genes))
        s_gene_indices = assay.feats.get_index_by(
            s_genes,
            "names",
            None,
        ).tolist()
        g2m_gene_indices = assay.feats.get_index_by(
            g2m_genes,
            "names",
            None,
        ).tolist()
        selection = self._ensure_cell_selection(cell_key)
        summary_ref = ensure_feature_summary(
            self.zw,
            assay,
            selection,
            invalidate_cache=invalidate_cache,
        )
        n_cells = feature_summary_selected_count(
            self.zw,
            selection,
            n_cells=assay.cells.N,
        )
        arguments = CellCycleArguments(
            feature_summary=summary_ref,
            cell_selection=selection,
            s_gene_indices=tuple(s_gene_indices),
            g2m_gene_indices=tuple(g2m_gene_indices),
            control_size=control_size,
            n_bins=n_bins,
            rand_seed=rand_seed,
            from_assay=from_assay,
            cell_key=cell_key,
            s_score_label=s_score_label,
            g2m_score_label=g2m_score_label,
            phase_label=phase_label,
            invalidate_cache=invalidate_cache,
        )
        record = arguments.to_record()
        planned = plan_cell_data_artifact(
            self.zw,
            scope="assay",
            assay=from_assay,
            kind=arguments.artifact_kind,
            operation=arguments.operation,
            parameters=record.parameters,
            inputs=record.inputs,
            execution_options=record.execution_options,
            cell_selection=selection,
            arrays={
                "s_score": ((n_cells,), "f"),
                "g2m_score": ((n_cells,), "f"),
                "phase": ((n_cells,), None),
            },
            invalidate_cache=invalidate_cache,
        )
        if planned.reused:
            artifact_group = as_zarr_group(
                self.zw[artifact_path(planned.ref)],
                name=planned.ref.artifact_id,
            )
            s_score = artifact_values(artifact_group, "s_score")
            g2m_score = artifact_values(artifact_group, "g2m_score")
            phase = artifact_values(artifact_group, "phase")
        else:
            summary = feature_summary_values(
                self.zw,
                summary_ref,
                n_selected=n_cells,
            )
            cell_idx = np.asarray(self.cells.active_index(cell_key), dtype=np.int64)
            s_score = assay._score_feature_indices(
                np.asarray(s_gene_indices, dtype=np.int64),
                cell_idx,
                summary["avg"],
                ctrl_size=control_size,
                n_bins=n_bins,
                rand_seed=rand_seed,
            )
            g2m_score = assay._score_feature_indices(
                np.asarray(g2m_gene_indices, dtype=np.int64),
                cell_idx,
                summary["avg"],
                ctrl_size=control_size,
                n_bins=n_bins,
                rand_seed=rand_seed,
            )
            phase = np.asarray(assign_cell_cycle_phase(s_score, g2m_score))
            write_cell_data_artifact(
                self.zw,
                planned,
                {
                    "s_score": np.asarray(s_score),
                    "g2m_score": np.asarray(g2m_score),
                    "phase": phase,
                },
            )
        s_score_label = self._col_renamer(
            from_assay,
            cell_key,
            s_score_label,
        )
        g2m_score_label = self._col_renamer(
            from_assay,
            cell_key,
            g2m_score_label,
        )
        phase_label = self._col_renamer(
            from_assay,
            cell_key,
            phase_label,
        )
        s_display = column_display(self.zw, s_score_label)
        g2m_display = column_display(self.zw, g2m_score_label)
        phase_display = column_display(self.zw, phase_label)
        self.cells.insert(s_score_label, s_score, key=cell_key, overwrite=True)
        link_cell_data_column(
            self.zw,
            s_score_label,
            planned.ref,
            value_name="s_score",
            default_display=continuous_display(s_score),
            preserved_display=s_display,
        )
        self.cells.insert(g2m_score_label, g2m_score, key=cell_key, overwrite=True)
        link_cell_data_column(
            self.zw,
            g2m_score_label,
            planned.ref,
            value_name="g2m_score",
            default_display=continuous_display(g2m_score),
            preserved_display=g2m_display,
        )
        self.cells.insert(phase_label, np.asarray(phase), key=cell_key, overwrite=True)
        link_cell_data_column(
            self.zw,
            phase_label,
            planned.ref,
            value_name="phase",
            default_display=categorical_display(np.asarray(phase)),
            preserved_display=phase_display,
        )

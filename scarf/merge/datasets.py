import os
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import zarr

from ..storage.budget import resolve_budget
from ..storage.layout import _group_zarr_format, count_array_spec
from ..storage.profiles import (
    StorageProfile,
    ZarrLocation,
    is_local_zarr_path,
    resolve_storage_profile,
)
from ..storage.schema import validate_assay_name, validate_workspace_name
from ..storage.sharding import preflight_counts_t_spec, row_band_task_count
from ..storage.stores import MATRIX_SOURCE_ATTR, load_zarr, zarr_root_path
from ..storage.types import as_zarr_group
from ..utils.logging import logger
from .features import align_features, resolve_merge_dtype
from .metadata import (
    CellMetadataPlan,
    admit_cell_metadata_plan,
    metadata_chunk_rows,
    plan_cell_metadata,
    resolve_metadata_schema_scan_rows,
    validate_cell_metadata,
    write_cell_metadata,
)
from .models import (
    AssayMergePlan,
    ComponentAction,
    ComponentResult,
    CountsTPolicy,
    MergePlan,
    MergeResult,
    MissingAssayPolicy,
)
from .row_plan import RowPlan, build_row_plan
from .writer import (
    MissingAssay,
    _assay_metadata_path,
    _cell_data_path,
    _matrix_group_path,
    counts_t_complete,
    create_assay_counts,
    matrix_group_complete,
    preflight_assay_counts,
    validate_assay_counts,
    validate_counts_t,
    write_assay_counts,
    write_assay_counts_t,
)

_IMPORT_SOURCE = "DataStoreMerge"
_MANIFEST_ATTR = "scarf:merge_manifest"


def _normalized_store_location(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    location = value.rstrip("/")
    if location.startswith("file://"):
        location = location[7:]
    if not location:
        return os.path.sep
    if "://" in location:
        return location
    return os.path.realpath(os.path.abspath(os.path.expanduser(location)))


def _location_tokens(loc: object) -> set[tuple[str, str | int]]:
    tokens: set[tuple[str, str | int]] = set()
    normalized = _normalized_store_location(loc)
    if normalized is not None:
        tokens.add(("location", normalized))
    if isinstance(loc, str) or loc is None:
        return tokens

    store = getattr(loc, "store", loc)
    tokens.add(("object", id(store)))
    store_root = getattr(store, "root", None)
    if store_root is not None:
        normalized_root = _normalized_store_location(str(store_root))
        if normalized_root is not None:
            tokens.add(("location", normalized_root))
    return tokens


def _dataset_tokens(ds: Any) -> set[tuple[str, str | int]]:
    tokens = _location_tokens(getattr(ds, "zarr_loc", None))
    root = getattr(ds, "z", None)
    if isinstance(root, zarr.Group | zarr.Array):
        tokens.update(_location_tokens(root.store))
        tokens.update(_location_tokens(zarr_root_path(root)))
    return tokens


@dataclass(frozen=True, slots=True)
class _DestinationInspection:
    actions: dict[str, ComponentAction]
    canDump: bool = True
    blockedReason: str | None = None
    needsFinalization: bool = False
    restart: bool = False


class DataStoreMerge:
    """Merge multiple DataStores into one Zarr store.

    Construction is side-effect free. Call :meth:`plan` to inspect the resolved
    merge, then :meth:`dump` to write or resume it.
    """

    def __init__(
        self,
        datasets: list[Any],
        zarr_path: ZarrLocation,
        names: list[str],
        *,
        assays: list[str] | None = None,
        out_workspace: str | None = None,
        dtype: str | None = None,
        overwrite: bool = False,
        prepend_text: str | None = "orig",
        reset_cell_filter: bool = True,
        seed: int | None = 42,
        storage_options: dict[str, Any] | None = None,
        source_column: str | None = None,
        mem_budget: int | str | None = None,
        nthreads: int | None = None,
        profile: StorageProfile | None = None,
        targetChunkBytes: int | None = None,
        targetShardBytes: int | None = None,
        counts_t: CountsTPolicy = "rna",
        missing_assay_policy: MissingAssayPolicy = "zero_fill",
    ) -> None:
        validate_workspace_name(out_workspace)
        if len(datasets) < 2:
            raise ValueError("DataStoreMerge requires at least two source DataStores")
        if len(datasets) != len(names):
            raise ValueError("datasets and names must have the same length")
        if len(names) != len(set(names)):
            raise ValueError("A unique name must be provided for each source DataStore")
        if assays is not None and len(assays) != len(set(assays)):
            raise ValueError("assays must not contain duplicate assay names")
        if counts_t not in {"rna", "all", "none"}:
            raise ValueError("counts_t must be one of 'rna', 'all', or 'none'")
        if missing_assay_policy not in {"zero_fill", "error"}:
            raise ValueError(
                "missing_assay_policy must be one of 'zero_fill' or 'error'"
            )
        self.datasets = datasets
        self.names = list(names)
        self.zarr_path = zarr_path
        self.assayFilter = None if assays is None else list(assays)
        self.outWorkspace = out_workspace
        self.dtype = dtype
        self.overwrite = overwrite
        self.prependText = prepend_text
        self.resetCellFilter = reset_cell_filter
        self.seed = seed
        self.storageOptions = storage_options
        self.sourceColumn = source_column
        self.countsT = counts_t
        self.missingAssayPolicy = missing_assay_policy
        self.targetChunkBytes = targetChunkBytes
        self.targetShardBytes = targetShardBytes
        self.resources = resolve_budget(
            mem_budget
            if mem_budget is not None
            else min(int(ds.memoryBytes) for ds in datasets),
            nthreads
            if nthreads is not None
            else min(int(ds.nthreads) for ds in datasets),
        )
        self.profile = resolve_storage_profile(zarr_path, profile)
        self.uniqueAssays = self._resolve_assays()
        self._rowPlan: RowPlan | None = None
        self._alignments: dict[str, Any] = {}
        self._assaySources: dict[str, list[Any]] = {}
        self._metadataPlan: CellMetadataPlan | None = None

    def _resolve_assays(self) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for ds in self.datasets:
            for assay_name in ds.assay_names:
                if assay_name in seen:
                    continue
                if self.assayFilter is not None and assay_name not in self.assayFilter:
                    continue
                seen.add(assay_name)
                unique.append(assay_name)
        if self.assayFilter is not None:
            missing = [name for name in self.assayFilter if name not in seen]
            if missing:
                raise ValueError(
                    f"Requested assays were not found in any source: {missing}"
                )
            # Preserve caller order when filtering.
            unique = [name for name in self.assayFilter if name in seen]
        if not unique:
            raise ValueError("No assays available to merge")
        for assay_name in unique:
            validate_assay_name(assay_name)
        return unique

    def _source_cell_counts(self) -> list[int]:
        return [int(ds.cells.N) for ds in self.datasets]

    def _row_chunk_sizes(self) -> list[int]:
        sizes: list[int] = []
        for ds in self.datasets:
            selected = [
                name
                for name in ds.assay_names
                if self.assayFilter is None or name in self.assayFilter
            ]
            if not selected:
                sizes.append(max(1, int(ds.cells.default_block_rows("ids"))))
                continue
            chunk_rows = [
                max(1, int(ds.get_assay(name).rawData.chunksize[0]))
                for name in selected
            ]
            sizes.append(min(chunk_rows))
        return sizes

    def _membership_by_source(self) -> dict[str, set[str]]:
        return {
            name: set(ds.assay_names)
            for ds, name in zip(self.datasets, self.names, strict=True)
        }

    def _prepare_sources(self) -> None:
        if self._rowPlan is not None:
            return
        self._rowPlan = build_row_plan(
            self._source_cell_counts(),
            self._row_chunk_sizes(),
            self.names,
            seed=self.seed,
        )
        reference_feats: dict[str, Any] = {}
        reference_dtype: dict[str, Any] = {}
        for assay_name in self.uniqueAssays:
            for ds in self.datasets:
                if assay_name in ds.assay_names:
                    assay = ds.get_assay(assay_name)
                    reference_feats[assay_name] = assay.feats
                    reference_dtype[assay_name] = np.dtype(assay.rawData.dtype)
                    break
            if assay_name not in reference_feats:
                raise ValueError(f"No source provides assay {assay_name!r}")

        for assay_name in self.uniqueAssays:
            sources: list[Any] = []
            missing_names: list[str] = []
            for ds, name in zip(self.datasets, self.names, strict=True):
                if assay_name in ds.assay_names:
                    assay = ds.get_assay(assay_name)
                    if not bool(getattr(assay, "isMissing", False)):
                        raw_shape = tuple(int(value) for value in assay.rawData.shape)
                        if len(raw_shape) != 2:
                            raise ValueError(
                                f"Source {name!r} assay {assay_name!r} rawData "
                                "must be two-dimensional"
                            )
                        raw_rows, raw_features = raw_shape
                        source_cells = int(ds.cells.N)
                        source_features = int(assay.feats.N)
                        if raw_rows != source_cells:
                            raise ValueError(
                                f"Source {name!r} assay {assay_name!r} rawData has "
                                f"{raw_rows} rows, but source cells has "
                                f"{source_cells}"
                            )
                        if raw_features != source_features:
                            raise ValueError(
                                f"Source {name!r} assay {assay_name!r} rawData has "
                                f"{raw_features} columns, but assay features has "
                                f"{source_features}"
                            )
                    sources.append(assay)
                else:
                    missing_names.append(name)
                    if self.missingAssayPolicy == "error":
                        raise ValueError(
                            f"Source {name!r} is missing assay {assay_name!r}"
                        )
                    logger.warning(
                        f"Source {name!r} is missing assay {assay_name!r}; "
                        "writing zeros and marking assay membership false"
                    )
                    sources.append(
                        MissingAssay(
                            cells=ds.cells,
                            feats=reference_feats[assay_name],
                            name=assay_name,
                            n_cells=int(ds.cells.N),
                            dtype=reference_dtype[assay_name],
                        )
                    )
            self._assaySources[assay_name] = sources
            self._alignments[assay_name] = align_features(sources, self.names)

    def _should_write_counts_t(self, assay_name: str, sources: list[Any]) -> bool:
        if self.countsT == "none":
            logger.debug(f"countsT disabled for assay {assay_name} by policy")
            return False
        if self.countsT == "all":
            logger.debug(f"countsT enabled for assay {assay_name} by policy")
            return True
        from ..assay import RNAassay

        source_is_rna = [
            isinstance(source, RNAassay)
            for source in sources
            if not isinstance(source, MissingAssay)
        ]
        decision = any(source_is_rna)
        logger.debug(
            f"countsT {'enabled' if decision else 'disabled'} for assay "
            f"{assay_name} from source assay types"
        )
        return decision

    def _build_manifest(self) -> dict[str, Any]:
        self._prepare_sources()
        assert self._rowPlan is not None
        return {
            "sourceNames": list(self.names),
            "sourceWorkspaces": [
                getattr(ds, "workspace", None) for ds in self.datasets
            ],
            "sourceCellCounts": self._source_cell_counts(),
            "sourceFeatureCounts": {
                assay_name: [
                    0 if isinstance(source, MissingAssay) else int(source.feats.N)
                    for source in self._assaySources[assay_name]
                ]
                for assay_name in self.uniqueAssays
            },
            "assays": list(self.uniqueAssays),
            "seed": self.seed,
            "prependText": self.prependText,
            "resetCellFilter": self.resetCellFilter,
            "sourceColumn": self.sourceColumn,
            "dtype": self.dtype,
            "profile": self.profile,
            "targetChunkBytes": self.targetChunkBytes,
            "targetShardBytes": self.targetShardBytes,
            "countsT": self.countsT,
            "missingAssayPolicy": self.missingAssayPolicy,
            "outWorkspace": self.outWorkspace,
            "nCells": self._rowPlan.nCells,
        }

    def _attr_root(self, root: zarr.Group) -> zarr.Group:
        if self.outWorkspace is None:
            return root
        if self.outWorkspace not in root:
            return root.create_group(self.outWorkspace)
        return as_zarr_group(root[self.outWorkspace], name=self.outWorkspace)

    def _cell_slot(self) -> str:
        return _cell_data_path(self.outWorkspace)

    def _source_destination_alias_reason(self) -> str | None:
        destination = _location_tokens(self.zarr_path)
        if not destination:
            return None
        for source, name in zip(self.datasets, self.names, strict=True):
            if destination & _dataset_tokens(source):
                return (
                    f"Destination aliases source DataStore {name!r}. "
                    "Choose a distinct destination store."
                )
        return None

    def _existing_attr_root(self, root: zarr.Group) -> zarr.Group | None:
        if self.outWorkspace is None:
            return root
        if self.outWorkspace not in root:
            return None
        return as_zarr_group(root[self.outWorkspace], name=self.outWorkspace)

    def _workspaces_claiming_assay(
        self,
        root: zarr.Group,
        assay_name: str,
    ) -> list[str] | None:
        if assay_name in root:
            return None
        claimed: list[str] = []
        for workspace in sorted(root.group_keys()):
            if workspace == "matrices":
                continue
            group = root[workspace]
            if isinstance(group, zarr.Group) and assay_name in group:
                claimed.append(workspace)
        return claimed

    def _containment_reason(self, root: zarr.Group | None) -> str | None:
        alias_reason = self._source_destination_alias_reason()
        if alias_reason is not None:
            return alias_reason
        if root is None:
            return None
        if MATRIX_SOURCE_ATTR in root.attrs:
            return (
                "Destination is a mounted matrix-source store and cannot be used "
                "for DataStoreMerge."
            )
        if (
            self.outWorkspace is not None
            and self.outWorkspace in root
            and not isinstance(root[self.outWorkspace], zarr.Group)
        ):
            return f"Destination workspace {self.outWorkspace!r} is not a Zarr group."

        attr_root = self._existing_attr_root(root)
        if attr_root is not None:
            import_source = attr_root.attrs.get("scarf:import_source")
            if import_source is not None and import_source != _IMPORT_SOURCE:
                return (
                    f"Destination workspace has foreign import source "
                    f"{import_source!r}, not {_IMPORT_SOURCE!r}."
                )

        if self.outWorkspace is None:
            if "matrices" in root:
                return (
                    "Cannot merge into the legacy layout because the destination "
                    "already uses workspace matrix storage."
                )
            return None

        for assay_name in self.uniqueAssays:
            matrix_path = _matrix_group_path(assay_name, self.outWorkspace)
            if matrix_path not in root:
                continue
            claimers = self._workspaces_claiming_assay(root, assay_name)
            if claimers is None:
                return (
                    f"Destination matrix {matrix_path!r} is claimed by the legacy "
                    "assay layout."
                )
            if not claimers:
                return (
                    f"Destination matrix {matrix_path!r} is orphaned and cannot be "
                    "claimed by a workspace merge."
                )
            other_workspaces = [
                workspace for workspace in claimers if workspace != self.outWorkspace
            ]
            if other_workspaces:
                return (
                    f"Destination matrix {matrix_path!r} is claimed by workspace "
                    f"{other_workspaces!r}, not {self.outWorkspace!r}."
                )
        return None

    def _is_fresh_destination_shell(
        self,
        root: zarr.Group,
        attr_root: zarr.Group | None,
    ) -> bool:
        if attr_root is None:
            return self.outWorkspace is not None
        return not (
            attr_root.attrs
            or tuple(attr_root.group_keys())
            or tuple(attr_root.array_keys())
        )

    def _open_existing(self) -> zarr.Group | None:
        try:
            return load_zarr(
                self.zarr_path,
                mode="r",
                storage_options=self.storageOptions,
            )
        except FileNotFoundError:
            return None

    @staticmethod
    def _initial_actions(
        assay_plans: tuple[AssayMergePlan, ...] | list[AssayMergePlan],
        action: ComponentAction,
    ) -> dict[str, ComponentAction]:
        actions: dict[str, ComponentAction] = {"cellData": action}
        for assay_plan in assay_plans:
            assay_name = assay_plan.assayName
            actions[f"counts:{assay_name}"] = action
            actions[f"countsT:{assay_name}"] = (
                action if assay_plan.writeCountsT else "skip"
            )
        return actions

    def _blocked_inspection(
        self,
        assay_plans: tuple[AssayMergePlan, ...] | list[AssayMergePlan],
        reason: str,
    ) -> _DestinationInspection:
        return _DestinationInspection(
            actions=self._initial_actions(assay_plans, "blocked"),
            canDump=False,
            blockedReason=reason,
        )

    def _inspect_existing(
        self,
        manifest: dict[str, Any],
        assay_plans: tuple[AssayMergePlan, ...] | list[AssayMergePlan],
    ) -> _DestinationInspection:
        actions = self._initial_actions(assay_plans, "write")
        containment_reason = self._containment_reason(None)
        if containment_reason is not None:
            return self._blocked_inspection(assay_plans, containment_reason)
        existing = self._open_existing()
        containment_reason = self._containment_reason(existing)
        if containment_reason is not None:
            return self._blocked_inspection(assay_plans, containment_reason)
        if existing is None:
            return _DestinationInspection(actions)
        attr_root = self._existing_attr_root(existing)
        if self._is_fresh_destination_shell(existing, attr_root):
            return _DestinationInspection(actions)
        assert attr_root is not None
        stored_manifest = attr_root.attrs.get(_MANIFEST_ATTR)
        import_source = attr_root.attrs.get("scarf:import_source")
        import_complete = attr_root.attrs.get("scarf:import_complete") is True
        complete = attr_root.attrs.get("complete") is True
        if self.overwrite:
            if import_source == _IMPORT_SOURCE:
                return _DestinationInspection(actions, restart=True)
            return self._blocked_inspection(
                assay_plans,
                "overwrite=True can replace only a merge-owned destination or a "
                "fresh workspace shell.",
            )
        if import_source != _IMPORT_SOURCE or not isinstance(stored_manifest, dict):
            return self._blocked_inspection(
                assay_plans,
                "Destination already exists and does not contain a matching "
                "DataStoreMerge manifest.",
            )
        if stored_manifest != manifest:
            return self._blocked_inspection(
                assay_plans,
                "Destination contains an incomplete or completed merge with a "
                "different configuration. Set overwrite=True to restart the "
                "merge-owned components.",
            )

        assert self._rowPlan is not None
        assert self._metadataPlan is not None
        cell_slot = self._cell_slot()
        if cell_slot in existing:
            cell_group = as_zarr_group(existing[cell_slot], name=cell_slot)
            if cell_group.attrs.get("complete") is True:
                invalid = validate_cell_metadata(
                    existing,
                    self.outWorkspace,
                    self._rowPlan,
                    [ds.cells for ds in self.datasets],
                    self._metadataPlan,
                    resources=self.resources,
                    resident_bytes=(
                        self._rowPlan.resident_bytes()
                        + sum(
                            alignment.resident_bytes()
                            for alignment in self._alignments.values()
                        )
                    ),
                )
                if invalid is not None:
                    return self._blocked_inspection(
                        assay_plans,
                        f"Completed cellData cannot be reused: {invalid}. "
                        "Set overwrite=True to rebuild it.",
                    )
                actions["cellData"] = "skip"
            else:
                actions["cellData"] = "resume"

        for assay_plan in assay_plans:
            assay_name = assay_plan.assayName
            if matrix_group_complete(existing, assay_name, self.outWorkspace):
                invalid = validate_assay_counts(
                    existing,
                    assay_name,
                    self.outWorkspace,
                    n_cells=self._rowPlan.nCells,
                    alignment=self._alignments[assay_name],
                    dtype=assay_plan.dtype,
                    chunks=assay_plan.chunks,
                    shards=assay_plan.shards,
                )
                if invalid is not None:
                    return self._blocked_inspection(
                        assay_plans,
                        f"Completed counts for {assay_name!r} cannot be reused: "
                        f"{invalid}. Set overwrite=True to rebuild it.",
                    )
                actions[f"counts:{assay_name}"] = "skip"
            else:
                actions[f"counts:{assay_name}"] = "resume"
            if not assay_plan.writeCountsT:
                actions[f"countsT:{assay_name}"] = "skip"
            elif actions[f"counts:{assay_name}"] != "skip":
                actions[f"countsT:{assay_name}"] = "resume"
            elif counts_t_complete(existing, assay_name, self.outWorkspace):
                invalid = validate_counts_t(
                    existing,
                    assay_name,
                    self.outWorkspace,
                    n_cells=self._rowPlan.nCells,
                    n_features=assay_plan.nFeatures,
                    dtype=assay_plan.dtype,
                )
                if invalid is not None:
                    return self._blocked_inspection(
                        assay_plans,
                        f"Completed countsT for {assay_name!r} cannot be reused: "
                        f"{invalid}. Set overwrite=True to rebuild it.",
                    )
                actions[f"countsT:{assay_name}"] = "skip"
            else:
                actions[f"countsT:{assay_name}"] = "resume"

        all_complete = all(action == "skip" for action in actions.values())
        if (import_complete or complete) and not all_complete:
            return self._blocked_inspection(
                assay_plans,
                "Destination is marked complete but one or more planned "
                "components are incomplete. Set overwrite=True to rebuild the "
                "merge-owned components.",
            )
        return _DestinationInspection(
            actions,
            needsFinalization=all_complete and not (import_complete and complete),
        )

    def plan(self) -> MergePlan:
        """Return a side-effect-free merge plan."""
        self._prepare_sources()
        assert self._rowPlan is not None
        if self._metadataPlan is None:
            preferred_rows = metadata_chunk_rows(self._rowPlan)
            resident_bytes = self._rowPlan.resident_bytes() + sum(
                alignment.resident_bytes() for alignment in self._alignments.values()
            )
            scan_rows = resolve_metadata_schema_scan_rows(
                [ds.cells for ds in self.datasets],
                self._rowPlan,
                self.resources,
                resident_bytes=resident_bytes,
                preferred_rows=preferred_rows,
            )
            self._metadataPlan = plan_cell_metadata(
                [ds.cells for ds in self.datasets],
                self.names,
                prepend_text=self.prependText,
                reset_cell_filter=self.resetCellFilter,
                source_column=self.sourceColumn,
                membership_assays=self.uniqueAssays,
                block_rows=preferred_rows,
                scan_rows=scan_rows,
            )
        assert self._metadataPlan is not None
        manifest = self._build_manifest()
        existing = self._open_existing()
        zarr_format = 3 if existing is None else _group_zarr_format(existing)
        preliminary_plans: list[AssayMergePlan] = []
        count_specs = {}
        for assay_name in self.uniqueAssays:
            sources = self._assaySources[assay_name]
            alignment = self._alignments[assay_name]
            present = tuple(not isinstance(source, MissingAssay) for source in sources)
            missing = tuple(
                name
                for name, is_present in zip(self.names, present, strict=True)
                if not is_present
            )
            dtype = resolve_merge_dtype(
                [source for source in sources if not isinstance(source, MissingAssay)],
                alignment.featOrderMap,
                self.dtype,
            )
            count_spec = count_array_spec(
                self._rowPlan.nCells,
                alignment.nFeats,
                dtype=dtype,
                profile=self.profile,
                targetChunkBytes=self.targetChunkBytes,
                targetShardBytes=self.targetShardBytes,
                zarrFormat=zarr_format,
            )
            count_specs[assay_name] = count_spec
            chunks = count_spec.chunks
            shards = count_spec.shards
            write_t = self._should_write_counts_t(assay_name, sources)
            shard_rows = chunks[0] if shards is None else shards[0]
            preliminary_plans.append(
                AssayMergePlan(
                    assayName=assay_name,
                    sourcePresent=present,
                    missingSources=missing,
                    nFeatures=alignment.nFeats,
                    featureOverlapFraction=alignment.overlapFraction,
                    dtype=dtype,
                    chunks=chunks,  # type: ignore[arg-type]
                    shards=shards,  # type: ignore[arg-type]
                    writeCountsT=write_t,
                    estimatedWriteTasks=row_band_task_count(
                        self._rowPlan.nCells,
                        int(shard_rows),
                    ),
                    countsAction="write",
                    countsTAction="write" if write_t else "skip",
                )
            )
        inspection = self._inspect_existing(manifest, preliminary_plans)
        if any(item.writeCountsT for item in preliminary_plans) and zarr_format < 3:
            inspection = self._blocked_inspection(
                preliminary_plans,
                "countsT requires a Zarr v3 destination. Repack the store or "
                "choose counts_t='none'.",
            )
        if inspection.canDump:
            alignment_bytes = {
                assay_name: alignment.resident_bytes()
                for assay_name, alignment in self._alignments.items()
            }
            total_alignment_bytes = sum(alignment_bytes.values())
            if inspection.actions["cellData"] != "skip":
                metadata_plan = self._metadataPlan
                assert metadata_plan is not None
                self._metadataPlan = admit_cell_metadata_plan(
                    metadata_plan,
                    self._rowPlan,
                    self.resources,
                    resident_bytes=(
                        self._rowPlan.resident_bytes() + total_alignment_bytes
                    ),
                )
            for assay_plan in preliminary_plans:
                assay_name = assay_plan.assayName
                if inspection.actions[f"counts:{assay_name}"] != "skip":
                    preflight_assay_counts(
                        count_specs[assay_name],
                        self._assaySources[assay_name],
                        self._rowPlan,
                        self._alignments[assay_name],
                        resources=self.resources,
                        additionalResidentBytes=(
                            total_alignment_bytes - alignment_bytes[assay_name]
                        ),
                    )
                if inspection.actions[f"countsT:{assay_name}"] != "skip":
                    preflight_counts_t_spec(
                        count_specs[assay_name],
                        profile=self.profile,
                        resources=self.resources,
                        residentBytes=(
                            self._rowPlan.resident_bytes() + total_alignment_bytes
                        ),
                    )
        assay_plans = tuple(
            replace(
                assay_plan,
                countsAction=inspection.actions[f"counts:{assay_plan.assayName}"],
                countsTAction=inspection.actions[f"countsT:{assay_plan.assayName}"],
            )
            for assay_plan in preliminary_plans
        )
        will_resume = inspection.needsFinalization or any(
            action == "resume" for action in inspection.actions.values()
        )
        plan = MergePlan(
            zarrPath=str(self.zarr_path),
            outWorkspace=self.outWorkspace,
            sourceNames=tuple(self.names),
            nCells=self._rowPlan.nCells,
            assays=tuple(assay_plans),
            profile=self.profile,
            seed=self.seed,
            countsT=self.countsT,
            missingAssayPolicy=self.missingAssayPolicy,
            willResume=will_resume,
            canDump=inspection.canDump,
            blockedReason=inspection.blockedReason,
            cellDataAction=inspection.actions["cellData"],
            manifest=manifest,
        )
        return plan

    def _clear_merge_components(
        self,
        root: zarr.Group,
        stored_manifest: dict[str, Any] | None,
    ) -> None:
        assay_names = set(self.uniqueAssays)
        if stored_manifest is not None:
            stored_assays = stored_manifest.get("assays")
            if isinstance(stored_assays, list):
                assay_names.update(
                    name for name in stored_assays if isinstance(name, str)
                )
        paths = {self._cell_slot()}
        for assay_name in assay_names:
            paths.add(_assay_metadata_path(assay_name, self.outWorkspace))
            paths.add(_matrix_group_path(assay_name, self.outWorkspace))
        for path in sorted(paths, key=lambda value: value.count("/"), reverse=True):
            if path in root:
                del root[path]

    def _open_destination(
        self,
        manifest: dict[str, Any],
        inspection: _DestinationInspection,
    ) -> zarr.Group:
        existing = self._open_existing()
        containment_reason = self._containment_reason(existing)
        if containment_reason is not None:
            raise ValueError(containment_reason)
        if existing is None:
            if (
                is_local_zarr_path(self.zarr_path)
                and isinstance(self.zarr_path, str)
                and os.path.exists(self.zarr_path)
            ):
                raise ValueError(
                    f"ERROR: Directory/file with name `{self.zarr_path}`exists. "
                    f"Either delete it or use another name"
                )
            root = load_zarr(
                self.zarr_path,
                mode="w",
                storage_options=self.storageOptions,
            )
        else:
            root = load_zarr(
                self.zarr_path,
                mode="r+",
                storage_options=self.storageOptions,
            )
            if inspection.restart:
                current_attr_root = self._existing_attr_root(root)
                if not self._is_fresh_destination_shell(root, current_attr_root):
                    if (
                        current_attr_root is None
                        or current_attr_root.attrs.get("scarf:import_source")
                        != _IMPORT_SOURCE
                    ):
                        raise ValueError(
                            "Destination changed after planning and is no longer "
                            "safe to overwrite."
                        )
                stored_manifest = (
                    current_attr_root.attrs.get(_MANIFEST_ATTR)
                    if current_attr_root is not None
                    else None
                )
                self._clear_merge_components(
                    root,
                    stored_manifest if isinstance(stored_manifest, dict) else None,
                )
        containment_reason = self._containment_reason(root)
        if containment_reason is not None:
            raise ValueError(containment_reason)
        attr_root = self._attr_root(root)
        attr_root.attrs["scarf:import_source"] = _IMPORT_SOURCE
        attr_root.attrs["scarf:import_complete"] = False
        attr_root.attrs["complete"] = False
        attr_root.attrs[_MANIFEST_ATTR] = manifest
        return root

    def dump(self) -> MergeResult:
        """Write or resume the merge and return a component-level result."""
        plan = self.plan()
        try:
            return self._dump_prepared(plan)
        finally:
            self._reset_prepared_state()

    def _dump_prepared(self, plan: MergePlan) -> MergeResult:
        assert self._rowPlan is not None
        assert self._metadataPlan is not None
        containment_reason = self._containment_reason(self._open_existing())
        if containment_reason is not None:
            raise ValueError(containment_reason)
        inspection = self._inspect_existing(plan.manifest, plan.assays)
        if not inspection.canDump:
            raise ValueError(inspection.blockedReason)
        actions = inspection.actions
        if all(action == "skip" for action in actions.values()):
            if inspection.needsFinalization:
                root = load_zarr(
                    self.zarr_path,
                    mode="r+",
                    storage_options=self.storageOptions,
                )
                containment_reason = self._containment_reason(root)
                if containment_reason is not None:
                    raise ValueError(containment_reason)
                attr_root = self._attr_root(root)
                attr_root.attrs["scarf:import_complete"] = True
                attr_root.attrs["complete"] = True
            return MergeResult(
                zarrPath=str(self.zarr_path),
                nCells=plan.nCells,
                assayNames=tuple(self.uniqueAssays),
                components=tuple(ComponentResult(name, "skip") for name in actions),
                resumed=inspection.needsFinalization,
            )
        root = self._open_destination(plan.manifest, inspection)
        components: list[ComponentResult] = []
        resumed = any(action == "resume" for action in actions.values())

        cell_action = actions["cellData"]
        if cell_action != "skip":
            write_cell_metadata(
                root,
                self.outWorkspace,
                self._rowPlan,
                [ds.cells for ds in self.datasets],
                self._metadataPlan,
                profile=self.profile,
                prepend_text=self.prependText,
                reset_cell_filter=self.resetCellFilter,
                source_column=self.sourceColumn,
                membership_by_source=self._membership_by_source(),
                overwrite=True,
            )
            components.append(ComponentResult("cellData", cell_action))
        else:
            components.append(ComponentResult("cellData", "skip"))

        for assay_plan in plan.assays:
            assay_name = assay_plan.assayName
            sources = self._assaySources[assay_name]
            alignment = self._alignments[assay_name]
            counts_action = actions[f"counts:{assay_name}"]
            if counts_action != "skip":
                create_assay_counts(
                    root,
                    assay_name,
                    self.outWorkspace,
                    self._rowPlan.nCells,
                    alignment,
                    assay_plan.dtype,
                    profile=self.profile,
                    targetChunkBytes=self.targetChunkBytes,
                    targetShardBytes=self.targetShardBytes,
                )
                write_assay_counts(
                    root,
                    assay_name,
                    self.outWorkspace,
                    sources,
                    self._rowPlan,
                    alignment,
                    resources=self.resources,
                    profile=self.profile,
                    additionalResidentBytes=sum(
                        item.resident_bytes()
                        for name, item in self._alignments.items()
                        if name != assay_name
                    ),
                )
                components.append(
                    ComponentResult(f"counts:{assay_name}", counts_action)
                )
            else:
                components.append(ComponentResult(f"counts:{assay_name}", "skip"))

            counts_t_action = actions[f"countsT:{assay_name}"]
            if assay_plan.writeCountsT and counts_t_action != "skip":
                write_assay_counts_t(
                    root,
                    assay_name,
                    self.outWorkspace,
                    profile=self.profile,
                    resources=self.resources,
                    residentBytes=(
                        self._rowPlan.resident_bytes()
                        + sum(
                            item.resident_bytes() for item in self._alignments.values()
                        )
                    ),
                )
                components.append(
                    ComponentResult(f"countsT:{assay_name}", counts_t_action)
                )
            else:
                components.append(
                    ComponentResult(
                        f"countsT:{assay_name}",
                        "skip",
                    )
                )
            # Release per-assay alignment state.
            self._alignments.pop(assay_name, None)

        attr_root = self._attr_root(root)
        attr_root.attrs["scarf:import_complete"] = True
        attr_root.attrs["complete"] = True
        return MergeResult(
            zarrPath=str(self.zarr_path),
            nCells=plan.nCells,
            assayNames=tuple(self.uniqueAssays),
            components=tuple(components),
            resumed=resumed,
        )

    def _reset_prepared_state(self) -> None:
        self._rowPlan = None
        self._alignments.clear()
        self._assaySources.clear()
        self._metadataPlan = None

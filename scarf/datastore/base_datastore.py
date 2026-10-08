import numpy as np
import zarr
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from ..storage.artifacts import (
    ArtifactRef,
    ArtifactScope,
    ArtifactStatus,
    inspect_artifact,
    list_artifacts as list_artifact_refs,
)
from ..storage.types import ZarrMode, as_zarr_group
from ..storage.validation_scope import validation_scoped
from ..storage.budget import ResourceBudget
from ..features.gene_families import DEFAULT_PERCENT_PATTERNS
from ..features.values import measured_feature_means
from ..assay import RNAassay, ATACassay, ADTassay, Assay, preset_assay_types
from ..metadata import MetaData
from ..metadata.membership import (
    MeasuredCellsRemedy,
    require_measured_cells,
    resolve_assay_membership,
)
from ..metadata.rows import (
    apply_missing_mask,
    read_metadata_missing_rows,
    read_metadata_rows_chunkwise,
)
from ..metadata.selection import (
    CellValues,
    cell_value_array,
    resolve_cell_aligned_artifact,
)
from ..storage.schema import validate_assay_name
from ..storage.profiles import StorageProfile, ZarrLocation
from ..storage.stores import (
    load_zarr,
    metadata_workers,
    mount_artifact_namespace,
    resolve_matrix_source,
    run_concurrently,
)
from ..storage.selections import (
    ValidatedStoredSelection,
    resolve_stored_selection,
    validate_cell_selection,
    validate_stored_selection_integrity,
)
from ..utils.logging import logger

if TYPE_CHECKING:
    from ..storage.lineage import ArtifactLineage
    from ..mapping.reference import MappingReference
    from .summary import DataStoreSummary

# Kinds with a row per cell whose cells follow their lineage rather than a
# recorded cell selection, so load_cell_values points to load_artifact.
_LINEAGE_ALIGNED_KINDS = frozenset({"batch_correction", "reduction"})


def sanitize_hierarchy(
    z: zarr.Group,
    assay_name: str,
    workspace: str | None,
    matrix_root: zarr.Group | None = None,
) -> bool:
    """Test if an assay node in zarr object was created properly.

    Args:
        z: Zarr hierarchy object
        assay_name: String value with name of assay.
        workspace: Workspace name (None for legacy layout without ``matrices/``).
        matrix_root: Optional root that owns count matrices. Defaults to ``z``.

    Returns:
        True if assay_name is present in z and contains `counts` and `featureData` child nodes else raises error
    """
    matrix_root = z if matrix_root is None else matrix_root
    zw = z if workspace is None else as_zarr_group(z[workspace], name=workspace)
    assay_zw = as_zarr_group(
        _child(zw, assay_name, f"ERROR: {assay_name} not found in zarr file"),
        name=assay_name,
    )
    _child(assay_zw, "featureData", f"ERROR: 'featureData' not found in {assay_name}")
    if workspace is None:
        matrix_assay = (
            assay_zw
            if matrix_root is z
            else as_zarr_group(matrix_root[assay_name], name=assay_name)
        )
        _child(matrix_assay, "counts", f"ERROR: 'counts' not found in {assay_name}")
    else:
        matrices = as_zarr_group(
            _child(
                matrix_root,
                "matrices",
                "ERROR: Workspace defined but no 'matrices' slot found",
            ),
            name="matrices",
        )
        matrix_assay = as_zarr_group(
            _child(
                matrices,
                assay_name,
                f"ERROR: {assay_name} not found in workspace matrices slot",
            ),
            name=assay_name,
        )
        _child(
            matrix_assay,
            "counts",
            f"ERROR: 'counts' not found in {assay_name} in workspace matrices slot",
        )
    return True


def _child(group: zarr.Group, key: str, error: str) -> zarr.Group | zarr.Array:
    # Indexing reads the child once; a membership test first reads it twice.
    try:
        return group[key]
    except KeyError:
        raise KeyError(error) from None


def validate_min_features_per_cell(value: Any) -> int:
    """Validate a ``min_features_per_cell`` value and return it as a Python integer."""
    from ..utils.arguments import integer_argument

    return integer_argument(value, "min_features_per_cell", minimum=-1)


class BaseDataStore:
    """This is the base datastore class that deals with loading of assays from
    Zarr files and generating basic cell statistics like nCounts and nFeatures.
    Superclass of the other DataStores.

    Args:
        zarr_loc: Path to Zarr file created using one of writer functions of Scarf
        assay_types: A dictionary with keys as assay names present in the Zarr file and
                     values as either one of: 'RNA', 'ATAC', 'ADT', 'HTO', 'CRISPR',
                     'ANTIGEN', 'CUSTOM', 'GeneActivity', 'GeneScores', 'URNA' or
                     'Assay'. A read-only open cannot change an assay's type.
        default_assay: Name of assay that should be considered as default. It is mandatory to provide this value
                       when DataStore loads a Zarr file for the first time
        min_features_per_cell: Writable opens remove from ``I`` every cell whose
                               default-assay feature count is not greater than
                               this value, unless that would remove at least half
                               of the active cells.
        mito_pattern: Feature-name pattern for the ``{assay}_percentMito`` column of each RNA assay.
                      The first writable open replaces any existing column with values computed from
                      this pattern, or ``^MT-`` when None. Later opens keep the stored values when
                      None and reject a pattern that differs from the recorded one.
        ribo_pattern: The same for ``{assay}_percentRibo``, using ``^RPS|^RPL|^MRPS|^MRPL`` when None.
        zarr_mode: For read-write mode use ``r+`` or for read-only use ``r``.
                   (Default value: ``r+``)
        workspace: Workspace name within the Zarr store (None for legacy single-workspace layout).
        resources: Resolved memory and worker budget for this datastore.
        storage_profile: Zarr encoding profile used for new arrays written
                         through this datastore.
        storage_options: Backend options passed when opening the Zarr store.
        storageIo: Optional explicit read, compute, and write widths for storage
                   work. Unset values stay under automatic planning.

    Attributes:
        cells: MetaData object with cells and info about each cell (e. g. RNA_nCounts ids).
        nthreads: Number of threads to use for this datastore instance.
        z: The Zarr file (directory) used for this datastore instance.
    """

    @validation_scoped
    def __init__(
        self,
        zarr_loc: ZarrLocation,
        assay_types: dict[str, str],
        default_assay: str,
        min_features_per_cell: int,
        mito_pattern: str,
        ribo_pattern: str,
        zarr_mode: ZarrMode,
        workspace: str | None,
        resources: ResourceBudget,
        storage_profile: StorageProfile,
        storage_options: dict[str, Any] | None = None,
        storageIo: Any | None = None,
    ):
        # Checked before the store is opened, so an invalid value writes nothing.
        min_features_per_cell = validate_min_features_per_cell(min_features_per_cell)
        self.zarr_mode = zarr_mode
        self.zarr_loc = zarr_loc
        self.z = load_zarr(
            zarr_loc=zarr_loc,
            mode=zarr_mode,
            storage_options=storage_options,
        )
        resolved = resolve_matrix_source(
            self.z,
            storage_options=storage_options,
        )
        if resolved is None:
            self._matrix_z = None
            self.workspace = workspace
        else:
            self._matrix_z, source_workspace = resolved
            if workspace is not None and workspace != source_workspace:
                raise ValueError(
                    "workspace does not match the mounted matrixSource workspace"
                )
            self.workspace = source_workspace
            # A mount resolves its source's artifacts read only, after its own.
            self.z = mount_artifact_namespace(self.z, self._matrix_z, self.workspace)
        import_source = self.zw.attrs.get("scarf:import_source")
        if import_source is not None and not bool(
            self.zw.attrs.get("scarf:import_complete", False)
        ):
            raise RuntimeError(f"{import_source} import is incomplete")
        self.resources = resources
        self.nthreads = self.resources.workers
        self.memoryBytes = self.resources.memoryBytes
        self.storageProfile = storage_profile
        self.storageIo = storageIo
        from ..storage.identity import REBUILD_REQUIRED

        assay_groups = self._scan_assays()
        self._assayNames = tuple(assay_groups)
        for name, group in assay_groups.items():
            state = group.attrs.get("prepared")
            if state is not True and state is not False:
                raise ValueError(f"Assay {name!r} is not prepared. {REBUILD_REQUIRED}")
            if state is False and self.zw.read_only:
                # A fresh import is prepared by its first writable open.
                raise ValueError(
                    f"Assay {name!r} is not prepared yet. Open the store once with "
                    "zarr_mode='r+' to prepare it before opening it read-only or "
                    "mounting it; a store that cannot be written must be rebuilt "
                    "into a fresh destination with repack_store(..., data_only=True)."
                )
        legacy_state_paths = [
            f"{assay_name}/state"
            for assay_name, group in assay_groups.items()
            if "state" in group
        ]
        if legacy_state_paths:
            paths = ", ".join(legacy_state_paths)
            raise ValueError(
                f"Legacy assay state is unsupported: {paths}. This release never "
                "reads or migrates {assay}/state; rebuild the dataset with this "
                "Scarf version."
            )
        # The order is critical here:
        self.cells = self._load_cells()
        self._defaultAssay = self._load_default_assay(default_assay)
        self._load_assays(assay_types)
        # TODO: Reset all attrs, pca, dendrogram etc
        self._ini_cell_props(mito_pattern, ribo_pattern)
        if not self.zw.read_only:
            self._filter_cells(min_features_per_cell)
        if (
            self.zarr_mode == "r+"
            and self.zw.attrs.get("defaultAssay") != self._defaultAssay
        ):
            self.zw.attrs["defaultAssay"] = self._defaultAssay
        # TODO: Implement _caches to hold are cached data
        # TODO: Implement _defaults to hold default parameters for methods

    @property
    def zw(self) -> zarr.Group:
        """Return the active root or workspace Zarr group."""
        if self.workspace is None:
            ret_val: zarr.Group = self.z
        else:
            ret_val: zarr.Group = self.z[self.workspace]  # type: ignore
        return ret_val

    def inspect_artifact(self, ref: ArtifactRef) -> ArtifactStatus:
        """Inspect a logical artifact without mutating the store."""
        return inspect_artifact(self.zw, ref)

    def lineage(
        self,
        target: ArtifactRef | Mapping[str, ArtifactRef],
        *,
        references: "MappingReference | Sequence[MappingReference] | None" = None,
    ) -> "ArtifactLineage":
        """Build a read-only upstream lineage report for artifact outputs."""
        from ..storage.lineage import ArtifactLineage
        from ..mapping.artifact import validate_mapping_reference_binding
        from ..mapping.reference import MappingReference

        if references is None:
            resolved_references: Sequence[MappingReference] = ()
        elif isinstance(references, MappingReference):
            resolved_references = (references,)
        elif isinstance(references, Sequence) and not isinstance(
            references, str | bytes
        ):
            resolved_references = references
        else:
            raise TypeError(
                "references must be a MappingReference, a sequence of "
                "MappingReference values, or None"
            )

        external_roots: dict[str, zarr.Group] = {}
        for index, reference in enumerate(resolved_references):
            if not isinstance(reference, MappingReference):
                raise TypeError(f"references[{index}] must be a MappingReference")
            validate_mapping_reference_binding(reference)
            reference.validate_dataset_fingerprint()
            fingerprint = reference.external_ref.dataset_fingerprint
            root = reference.datastore.zw
            existing = external_roots.get(fingerprint)
            if existing is not None and str(existing.store_path) != str(
                root.store_path
            ):
                raise ValueError(
                    "References contain duplicate dataset fingerprint "
                    f"{fingerprint!r} for conflicting roots"
                )
            external_roots[fingerprint] = root

        return ArtifactLineage.from_store(
            self.zw,
            target,
            external_roots=external_roots,
        )

    def load_artifact(self, ref: ArtifactRef) -> zarr.Group:
        """Open a complete artifact through a read-only Zarr group."""
        status = self.inspect_artifact(ref)
        if not status.exists:
            raise KeyError(f"Artifact does not exist: {status.path}")
        if not status.complete:
            raise RuntimeError(f"Artifact is incomplete: {status.path}")
        workspace_path = str(getattr(self.zw, "path", "")).strip("/")
        store_path = (
            f"{workspace_path}/{status.path}" if workspace_path else status.path
        )
        return zarr.open_group(
            store=self.zw.store,
            path=store_path,
            mode="r",
            zarr_format=self.zw.metadata.zarr_format,
        )

    def load_cell_values(
        self,
        ref: ArtifactRef,
        *,
        value: str | None = None,
        cell_selection: ArtifactRef | None = None,
    ) -> CellValues:
        """Read the per-cell values of a cell-aligned artifact.

        Args:
            ref: A cell-aligned artifact, such as a clustering or an embedding.
            value: Name of the per-cell array to read, or None for the canonical one.
            cell_selection: A subset of the artifact's cell selection, or None for all.

        Returns:
            The values with their cell ids and missing mask, as ``CellValues``.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``ref`` is not a complete cell-aligned artifact.
            MemoryError: If the read does not fit the datastore memory budget.
        """
        if not isinstance(ref, ArtifactRef):
            raise TypeError("ref must be an ArtifactRef")
        if cell_selection is not None and not isinstance(cell_selection, ArtifactRef):
            raise TypeError("cell_selection must be an ArtifactRef")
        try:
            value_name, categorical = cell_value_array(ref.kind, value)
        except ValueError as error:
            if ref.kind not in _LINEAGE_ALIGNED_KINDS:
                raise
            raise ValueError(
                f"{error}. {ref.kind} artifacts take their cells from their "
                "lineage rather than a cell_selection input; open them with "
                "load_artifact"
            ) from None
        resolved = resolve_cell_aligned_artifact(
            self.zw,
            ref,
            cell_selection=cell_selection,
            value_name=value_name,
            ndim=None,
            max_bytes=self.memoryBytes,
            # The cell ids are read at the same rows, so the budget charges
            # them with the values.
            caller_reads=(self.cells._get_array("ids"),),
        )
        cell_ids = np.asarray(
            read_metadata_rows_chunkwise(self.cells, "ids", resolved.cell_idx)
        )
        return CellValues(
            source=ref,
            value=value_name,
            values=resolved.values,
            cell_ids=cell_ids,
            cell_idx=resolved.cell_idx,
            cell_selection=resolved.cell_selection,
            missing=resolved.missing_mask,
            categorical=categorical,
        )

    def list_artifacts(
        self,
        *,
        kind: str | None = None,
        from_assay: str | None = None,
        scope: ArtifactScope = "assay",
        complete_only: bool = False,
        operation: str | None = None,
        parameters: Mapping[str, Any] | None = None,
        inputs: Mapping[str, Any] | None = None,
    ) -> list[ArtifactRef]:
        """List every artifact ref matching the exact provenance predicates."""
        if scope == "assay" and from_assay is None:
            from_assay = self._defaultAssay
        return list_artifact_refs(
            self.zw,
            scope=scope,
            assay=from_assay,
            kind=kind,
            complete_only=complete_only,
            operation=operation,
            parameters=parameters,
            inputs=inputs,
        )

    def summary(self) -> "DataStoreSummary":
        """Return a read-only, metadata-only summary of this datastore."""
        from .. import __version__
        from .summary import build_datastore_summary

        return build_datastore_summary(self, scarf_version=__version__)

    def _load_cells(self) -> MetaData:
        """This convenience function loads cellData level from the Zarr
        hierarchy.

        Returns:
            Metadata object
        """
        try:
            cell_data = as_zarr_group(self.zw["cellData"], name="cellData")
        except KeyError as e:
            raise KeyError(
                f"cellData not found in zarr file at {self.zw.store_path}"
            ) from e
        return MetaData(cell_data)

    @property
    def assay_names(self) -> list[str]:
        """Names of the assays present in the Zarr file. Zarr writers create
        an 'is_assay' attribute in the assay level and the hierarchy is
        scanned for those attributes when the store opens and when an assay
        is added.

        Returns:
            Names of assays present in a Zarr file
        """
        return list(self._assayNames)

    def _scan_assays(self) -> dict[str, zarr.Group]:
        """Find and validate every assay group in one pass over the hierarchy."""
        # Object-store listings can repeat a group and may not preserve order
        # across calls, so keep unique names in sorted order.
        assays = {
            name: group
            for name, group in sorted(dict(self.zw.groups()).items())
            if "is_assay" in group.attrs
        }

        def check(name: str) -> Callable[[], bool]:
            def run() -> bool:
                validate_assay_name(name)
                return sanitize_hierarchy(
                    self.z,
                    name,
                    self.workspace,
                    matrix_root=self._matrix_root_for_assay(name),
                )

            return run

        run_concurrently(
            [check(name) for name in assays], workers=metadata_workers(self.zw)
        )
        return assays

    def _matrix_root_for_assay(self, assay_name: str) -> zarr.Group | None:
        """Return the local or mounted root that owns one assay's counts."""
        prefix = "" if self.workspace is None else "matrices/"
        try:
            self.z[f"{prefix}{assay_name}/counts"]
        except KeyError:
            return self._matrix_z
        return self.z

    def _load_default_assay(self, assay_name: str | None = None) -> str:
        """This function sets a given assay name as defaultAssay attribute. If
        `assay_name` value is None then the top-level directory attributes in
        the Zarr file are looked up for presence of previously used default
        assay.

        Args:
            assay_name: Name of the assay to be considered for setting as default.

        Returns:
            Name of the assay to be set as default assay
        """
        if assay_name is None:
            if "defaultAssay" in self.zw.attrs:
                assay_name = cast(str, self.zw.attrs["defaultAssay"])
                if assay_name not in self.assay_names:
                    raise ValueError(
                        f"ERROR: The stored default assay {assay_name!r} was not "
                        f"found. Choose one from: {' '.join(self.assay_names)}\n "
                        "using 'default_assay' parameter."
                    )
            else:
                if len(self.assay_names) == 1:
                    assay_name = self.assay_names[0]
                else:
                    raise ValueError(
                        "ERROR: You have more than one assay data. "
                        f"Choose one from: {' '.join(self.assay_names)}\n using 'default_assay' parameter. "
                        "Please note that names are case-sensitive."
                    )
        else:
            if assay_name in self.assay_names:
                if "defaultAssay" in self.zw.attrs:
                    if assay_name != self.zw.attrs["defaultAssay"]:
                        logger.info(
                            f"Default assay changed from {self.zw.attrs['defaultAssay']} to {assay_name}"
                        )
            else:
                raise ValueError(
                    f"ERROR: The provided default assay name: {assay_name} was not found. "
                    f"Please Choose one from: {' '.join(self.assay_names)}\n"
                    "Please note that the names are case-sensitive."
                )
        assert assay_name is not None
        return assay_name

    def _load_assays(self, custom_assay_types: Mapping[str, Any] | None = None) -> None:
        """Create the assay object of every assay in the store.

        Each assay takes its type from ``custom_assay_types``, then from the
        persisted ``assayTypes`` attribute, then from its own name when that
        name is a preset (see :func:`~scarf.assay.preset_assay_types`). An
        assay with none of these opens as the generic ``Assay`` with a warning.
        A writable store records the resolved types in ``assayTypes``. Each
        assay carries its resolved type as ``assayType``.

        A read-only open cannot record a type, so an explicit type must equal
        the type that the store declares. A record that is not a mapping is
        replaced only by a writable open that declares every assay explicitly.

        Args:
            custom_assay_types: Explicit preset type per assay name.

        Raises:
            ValueError: If ``custom_assay_types`` names an assay that is not in
                the store or a type that is not a preset, if a read-only open
                declares a type that differs from the store's, if the
                ``assayTypes`` record is not a mapping, or if a persisted type
                that an assay would use is not a preset. Nothing is written in
                that case.
        """
        from ..assay.classification import (
            lookup_persisted_assay_type,
            recorded_assay_types,
            validate_assay_types,
        )

        explicit = validate_assay_types(custom_assay_types, self._assayNames)
        writable = not self.zw.read_only
        presets = preset_assay_types()
        raw_types = self.zw.attrs.get("assayTypes")
        if (
            writable
            and raw_types is not None
            and not isinstance(raw_types, Mapping)
            and set(self._assayNames) <= set(explicit)
        ):
            # Every assay is declared explicitly, so this open replaces the
            # malformed record below.
            recorded: dict[str, Any] = {}
        else:
            recorded = recorded_assay_types(raw_types, self._assayNames)
        caution_statement = (
            "%s was set as a generic Assay with no normalization. If this is unintended "
            "then please make sure that you provide a correct assay type for this assay using "
            "'assay_types' parameter."
            "\nIf you have more than one assay in the dataset then you can set "
            "assay_types={'assay1': 'RNA', 'assay2': 'ADT'} "
            "Just replace with actual assay names instead of assay1 and assay2"
        )

        def stored_type(name: str) -> str:
            # The type that the store declares without an explicit one.
            if name in recorded or name in presets:
                return lookup_persisted_assay_type(name, recorded)
            return "Assay"

        # Every type is resolved, and so validated, before any assay is built.
        resolved: dict[str, str] = {}
        for name in self._assayNames:
            if name in explicit:
                resolved[name] = explicit[name]
                if not writable and explicit[name] != stored_type(name):
                    raise ValueError(
                        f"assay_types declares assay {name!r} as "
                        f"{explicit[name]!r}, but the store declares it as "
                        f"{stored_type(name)!r}. A read-only open cannot record "
                        "a type, so its assay_types must match the store. Open "
                        "the store once with zarr_mode='r+' and "
                        f"assay_types={{{name!r}: {explicit[name]!r}}} to record "
                        "the type, or omit assay_types to use the recorded one."
                    )
            else:
                if name not in recorded and name not in presets:
                    logger.warning(caution_statement % name)
                resolved[name] = stored_type(name)
            logger.debug(f"Setting assay {name} to assay type: {resolved[name]}")
        assays: dict[str, Assay] = {
            name: presets[type_name](
                z=self.z,
                workspace=self.workspace,
                name=name,
                cell_data=self.cells,
                nthreads=self.nthreads,
                matrix_root=self._matrix_root_for_assay(name),
                resources=self.resources,
                storageIo=self.storageIo,
                assay_type=type_name,
            )
            for name, type_name in resolved.items()
        }
        # Assays are kept apart from the datastore's own attributes, so no
        # assay name can replace one.
        self._assays = assays
        z_attrs = {name: str(value) for name, value in recorded.items()}
        z_attrs.update(resolved)
        if writable and self.zw.attrs.get("assayTypes") != z_attrs:
            self.zw.attrs["assayTypes"] = z_attrs
        return None

    def _get_assay(
        self,
        from_assay: str | None,
    ) -> Assay | RNAassay | ADTassay | ATACassay:
        """This is a convenience function used internally to quickly obtain the
        assay object that is linked to an assay name.

        Args:
            from_assay: Name of the assay whose object is to be returned.

        Returns:

        Raises:
            ValueError: if ``from_assay`` names no assay in this datastore.
        """
        if from_assay is None or from_assay == "":
            from_assay = self._defaultAssay
        # Only scanned assay names resolve; other attributes such as ``cells``
        # are not assays.
        if from_assay not in self._assayNames:
            available = ", ".join(self._assayNames)
            raise ValueError(
                f"Assay {from_assay!r} not found. Available assays: {available}"
            )
        return self._assays[from_assay]

    if not TYPE_CHECKING:

        def __getattr__(self, name: str) -> Assay:
            # Python calls this only after normal lookup fails, so the
            # datastore's own attributes always win over an assay of the same
            # name. A class attribute reaches here only when its getter raised
            # AttributeError; looking it up again surfaces that error. Reading
            # ``self._assays`` here would recurse on a store that has no
            # assays yet, such as one being copied or unpickled.
            if hasattr(type(self), name):
                return object.__getattribute__(self, name)
            assays = self.__dict__.get("_assays", {})
            if not name.startswith("_") and name in assays:
                return assays[name]
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}",
                name=name,
                obj=self,
            )

    def __dir__(self) -> list[str]:
        assays = self.__dict__.get("_assays", {})
        names = (name for name in assays if not name.startswith("_"))
        return sorted(set(super().__dir__()).union(names))

    def _require_writable(self, operation: str) -> None:
        """Refuse an operation that writes to a store opened read-only."""
        if self.zarr_mode != "r+":
            raise PermissionError(
                f"{operation} requires a DataStore opened with zarr_mode='r+'"
            )

    def _require_measured_cells(
        self,
        assay: str | None,
        cells: ArtifactRef | np.ndarray | str,
        *,
        operation: str,
        remedy: MeasuredCellsRemedy | None = None,
    ) -> None:
        """Refuse an operation that reads ``assay`` over cells it did not measure.

        Operations call this after their own argument checks and before they
        plan, reuse, or write a result, so a read-only store refuses such a
        request as a writable one does. It checks nothing when ``assay`` has
        no membership column, which means that it measured every cell, or
        names no assay of this store, which the operation reports itself;
        None names the default assay. A ``cells`` value of another type is
        left to the operation's own argument checks.

        Args:
            assay: Assay that the operation reads.
            cells: The cells that it reads: a datastore cell-selection
                artifact, whose stored mask is read in bounded blocks, the
                name of a boolean cell column, which the error names as its
                ``cell_key``, a boolean mask with one entry per cell, or
                their integer rows.
            operation: Name of the operation, which the error names.
            remedy: The input that the error tells the caller to narrow
                (see :func:`~scarf.metadata.membership.require_measured_cells`).

        Raises:
            UnmeasuredCellsError: If ``assay`` did not measure one of the cells.
        """
        name = assay or self._defaultAssay
        if name is None or name not in self.assay_names:
            return
        rows: Any
        if isinstance(cells, ArtifactRef):
            # A selection is validated only when there is membership to check.
            if resolve_assay_membership(self.cells, name) is None:
                return
            rows = validate_cell_selection(self.zw, cells).values
        elif isinstance(cells, np.ndarray | str):
            rows = cells
        else:
            return
        require_measured_cells(
            self.cells, name, rows, operation=operation, remedy=remedy
        )

    def _ensure_dataset_fingerprint(self, from_assay: str) -> str:
        from ..storage.identity import validate_preparation

        assay = self._get_assay(from_assay)
        fingerprint = validate_preparation(
            assay.z,
            self.cells.locations["primary"],
            assay.matrixGroup,
            require_transpose=assay.requiresCountsT,
        )
        assert fingerprint is not None
        return fingerprint

    def snapshot_cell_selection(self, cell_key: str = "I") -> ArtifactRef:
        """Capture a live boolean cell column as an immutable selection.

        The returned reference is the explicit starting input for atomic graph
        construction methods such as :meth:`run_normalization`. A complete
        matching snapshot is reused when possible.

        Args:
            cell_key: Boolean cell metadata column to capture.

        Returns:
            A complete datastore-scoped cell-selection artifact.
        """
        return self._snapshot_cell_selection(cell_key).ref

    def _snapshot_cell_selection(self, cell_key: str) -> ValidatedStoredSelection:
        return resolve_stored_selection(
            self.zw,
            table_path="cellData",
            id_column="ids",
            source_column=cell_key,
            scope="datastore",
            kind="cell_selection",
            operation="manual_selection",
            parameters={},
            inputs={},
        )

    def snapshot_cluster_labels(
        self,
        labels: str | ArtifactRef,
        *,
        cell_selection: ArtifactRef,
    ) -> ArtifactRef:
        """Freeze one label per selected cell as an immutable label artifact.

        Label consumers such as ``run_marker_search``, ``make_bulk``,
        ``select_cells``, ``smart_label``, and ``metric_label_concordance``
        take exact label artifacts. This method makes one from a cell
        metadata column, such as an annotation or a condition, or narrows an
        existing label artifact to some of its cells, such as the labelled
        cells of an imported clustering or a few of its clusters. A consumer
        that pairs the labels with another input, such as a graph in
        ``calc_membership_strength`` or a second label artifact in
        ``smart_label``, needs both inputs over one cell selection.

        The labels are read for exactly the cells of ``cell_selection``. A
        label artifact, such as ``cluster_labels``, ``cluster_cut``,
        ``smart_label``, or ``label_transfer``, must have a cell selection
        that contains ``cell_selection``. Every selected cell needs a label:
        a missing label, including a row that a linked missing mask flags,
        or a blank label is rejected. Select the labelled cells first, with
        ``select_cells(labels, include=[...])`` for an artifact or with
        ``snapshot_cell_selection`` of a boolean column that marks them.
        Text labels are stored at the width of the selected labels, and
        integer and boolean labels keep their dtype. Floating-point labels
        that are whole numbers within the int64 range, such as the float64
        ids that pandas writes for integer ids with missing values, are
        stored as int64, and other floating-point labels are rejected.

        The result is a datastore-scoped ``cluster_labels`` artifact over
        ``cell_selection``. Its identity holds the source column name or the
        source artifact, the cell selection, and a fingerprint of the stored
        labels. The same labels reuse one artifact, also from a store opened
        with ``zarr_mode='r'``, and changed labels create a new one while
        earlier snapshots keep their values. A mounted store writes the
        artifact to its target, never to the matrix source. The labels are
        not checked against the marker-group naming rule;
        ``run_marker_search`` rejects labels that cannot name a stored marker
        group before it writes anything.

        Args:
            labels: Cell metadata column name, or a cell-label artifact.
            cell_selection: Complete cell-selection artifact of the cells to
                label, for example from ``snapshot_cell_selection`` or
                ``select_cells``.

        Returns:
            A complete datastore-scoped ``cluster_labels`` artifact with one
            label per cell of ``cell_selection``.

        Raises:
            TypeError: If ``labels`` is neither a column name nor an
                ``ArtifactRef``, if ``cell_selection`` is not an
                ``ArtifactRef``, or if a label is not a text, integer,
                boolean, or whole-number value.
            KeyError: If no cell metadata column is named ``labels``.
            ValueError: If ``cell_selection`` is not a complete cell
                selection, if the ``labels`` artifact holds no cell labels or
                its cell selection does not contain ``cell_selection``, or if
                a selected cell has no label.
            PermissionError: If no matching snapshot exists and the store is
                not opened with ``zarr_mode='r+'``.
        """
        from ..metadata.artifacts import snapshot_cluster_labels

        return snapshot_cluster_labels(
            self.zw, self.cells, labels, cell_selection=cell_selection
        )

    def _selection_artifacts_match(
        self,
        first: ArtifactRef,
        second: ArtifactRef,
    ) -> bool:
        table_path = "cellData"
        try:
            first_status = inspect_artifact(self.zw, first)
            second_status = inspect_artifact(self.zw, second)
            validate_stored_selection_integrity(
                self.zw,
                first,
                kind=first.kind,
                scope=first.scope,
                assay=first.assay,
                table_path=table_path,
            )
            if first == second:
                return True
            validate_stored_selection_integrity(
                self.zw,
                second,
                kind=second.kind,
                scope=second.scope,
                assay=second.assay,
                table_path=table_path,
            )
            # Both are complete and fingerprint the live cell rows, so equal
            # values fingerprints select the same cells.
            first_inputs = first_status.inputs or {}
            second_inputs = second_status.inputs or {}
        except (KeyError, TypeError, ValueError):
            return False
        return first_inputs.get("values_fingerprint") == second_inputs.get(
            "values_fingerprint"
        )

    def _ini_cell_props(
        self,
        mito_pattern: str | None,
        ribo_pattern: str | None,
    ) -> None:
        """Prepare the cell and feature statistics of every assay.

        This never changes ``I``. Only a writable open filters it, once, with
        its ``min_features_per_cell`` through ``_filter_cells``.
        """
        for from_assay in self._assayNames:
            # _load_assays opened every assay group, so its attributes are current.
            assay = self._get_assay(from_assay)
            prepared = assay.z.attrs.get("prepared") is True
            patterns: dict[str, str | None] = {}
            if isinstance(assay, RNAassay):
                patterns = {
                    f"{from_assay}_percentMito": mito_pattern
                    if prepared or mito_pattern is not None
                    else DEFAULT_PERCENT_PATTERNS["percentMito"],
                    f"{from_assay}_percentRibo": ribo_pattern
                    if prepared or ribo_pattern is not None
                    else DEFAULT_PERCENT_PATTERNS["percentRibo"],
                }
            assay.prepare(patterns)

    def _filter_cells(self, min_features: int) -> None:
        """Remove low-feature cells of the default assay from ``I``.

        Active cells whose feature count is not greater than ``min_features``
        are removed, unless they are at least half of the active cells: then
        ``I`` is kept and a warning is logged.
        """
        from_assay = self._defaultAssay
        active = self.cells.fetch_all("I")
        n_active = int(np.count_nonzero(active))
        if n_active == 0:
            return
        n_features = self.cells.fetch_all(from_assay + "_nFeatures")
        removed = active & (n_features <= min_features)
        n_removed = int(np.count_nonzero(removed))
        # Write only when filtering changes the active cells.
        if n_removed == 0:
            return
        if 2 * n_removed >= n_active:
            logger.warning(
                f"{n_removed} of {n_active} active cells have at most {min_features} "
                f"features in assay {from_assay!r}. Will not remove low quality cells "
                "automatically, because that would remove at least half of the active "
                "cells."
            )
            return
        self.cells.update_key(~removed, key="I")

    def get_cell_vals(
        self,
        from_assay: str,
        cell_key: str,
        k: str,
        clip_fraction: float = 0,
    ) -> np.ndarray:
        """Fetches data from the Zarr file.

        This convenience function allows fetching values for cells from either cell metadata table or values of a
        given feature from normalized matrix.

        Rows that a nullable metadata column's linked missing mask flags are
        returned as missing values, as in run-aware plotting views: NaN for
        integer and float columns, which are then returned as float64, NaT for
        datetime and timedelta columns, None for other non-boolean columns, and
        False for boolean columns. Unclipped columns without masked rows keep
        their stored dtype.

        Args:
            from_assay: Name of assay to be used.
            cell_key: Boolean column in cell metadata selecting cells. Required; pass ``'I'``
                      for the default active-cell key.
            k: Cell metadata column name or feature name whose values are fetched.
            clip_fraction: Fraction in [0, 0.5) for soft percentile clipping of numeric
                           values. Missing values are ignored when the percentiles are
                           computed. Clipped integer columns are returned as float64.

        Returns:
            The requested values; a feature is NaN in cells its assay did not measure.
        """
        from ..utils.arguments import clip_fraction_argument

        clip_fraction = clip_fraction_argument(clip_fraction)
        cell_idx = self.cells.active_index(cell_key)
        if k not in self.cells.columns:
            assay = self._get_assay(from_assay)
            feat_idx = assay.feats.get_index_by([k], "names")
            if len(feat_idx) == 0:
                raise ValueError(f"ERROR: {k} not found in {from_assay} assay.")
            else:
                if len(feat_idx) > 1:
                    logger.warning(
                        f"Plotting mean of {len(feat_idx)} features because {k} is not unique."
                    )
            vals = measured_feature_means(
                assay, feat_idx, cell_idx, nthreads=self.nthreads
            )
        else:
            vals = apply_missing_mask(
                self.cells.fetch(k, key=cell_key),
                read_metadata_missing_rows(self.cells, k, cell_idx),
            )
        if clip_fraction > 0 and vals.dtype.kind in "iuf":
            values = vals.astype(np.float64, copy=False)
            present = values[~np.isnan(values)]
            if present.size:
                low, high = np.percentile(
                    present, [100 * clip_fraction, 100 - 100 * clip_fraction]
                )
                values = np.clip(values, low, high)
            vals = (
                values
                if vals.dtype.kind in "iu"
                else values.astype(vals.dtype, copy=False)
            )
        return vals

    def __repr__(self) -> str:
        def formatter(iter_vals: Iterable[str]) -> str:
            values = [f"'{x}'" for x in iter_vals]
            rows = [", ".join(values[i : i + 5]) for i in range(0, len(values), 5)]
            return f"\n{dtabs}" + f", \n{dtabs}".join(rows) if rows else ""

        htabs = " " * 3
        stabs = htabs * 2
        dtabs = stabs * 2

        res = (
            f"DataStore has {self.cells.active_index('I').shape[0]} ({self.cells.N}) cells with"
            f" {len(self.assay_names)} assays: {' '.join(self.assay_names)}"
        )
        res = res + f"\n{htabs}Cell metadata:"
        res += formatter(self.cells.columns)
        for i in self.assay_names:
            assay = self._get_assay(i)
            res += (
                f"\n{htabs}{i} assay has {assay.feats.N} "
                f"features and following metadata:"
            )
            res += formatter(assay.feats.columns)
        return res

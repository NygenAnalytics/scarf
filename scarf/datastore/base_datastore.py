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
from ..assay.classification import DEFAULT_PERCENT_PATTERNS
from ..assay import RNAassay, ATACassay, ADTassay, Assay, preset_assay_types
from ..metadata import MetaData
from ..storage.schema import validate_assay_name
from ..storage.profiles import StorageProfile, ZarrLocation
from ..storage.stores import (
    load_zarr,
    metadata_workers,
    resolve_matrix_source,
    run_concurrently,
)
from ..storage.selections import (
    ValidatedStoredSelection,
    resolve_stored_selection,
    validate_stored_selection_integrity,
)
from ..utils.compute import controlled_compute
from ..utils.logging import logger

if TYPE_CHECKING:
    from ..storage.lineage import ArtifactLineage
    from ..mapping.reference import MappingReference
    from .summary import DataStoreSummary


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


class BaseDataStore:
    """This is the base datastore class that deals with loading of assays from
    Zarr files and generating basic cell statistics like nCounts and nFeatures.
    Superclass of the other DataStores.

    Args:
        zarr_loc: Path to Zarr file created using one of writer functions of Scarf
        assay_types: A dictionary with keys as assay names present in the Zarr file and values as either one of:
                     'RNA', 'ADT', 'ATAC' or 'GeneActivity'
        default_assay: Name of assay that should be considered as default. It is mandatory to provide this value
                       when DataStore loads a Zarr file for the first time
        min_features_per_cell: Minimum number of non-zero features in a cell. If lower than this then the cell
                               will be filtered out.
        mito_pattern: Pattern for missing mitochondrial percentages. None preserves existing values
                      and uses ``^MT-`` for new values. Explicit patterns must match existing provenance.
        ribo_pattern: Pattern for missing ribosomal percentages. None preserves existing values
                      and uses ``RPS|RPL|MRPS|MRPL`` for new values.
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
            if workspace is None:
                self.workspace = source_workspace
            elif workspace != source_workspace:
                raise ValueError(
                    "workspace does not match the mounted matrixSource workspace"
                )
            else:
                self.workspace = workspace
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
            if (state is not True and state is not False) or (
                self.zw.read_only and state is not True
            ):
                raise ValueError(f"Assay {name!r} is not prepared. {REBUILD_REQUIRED}")
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
        self._ini_cell_props(min_features_per_cell, mito_pattern, ribo_pattern)
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

    @property
    def last_execution_report(self) -> Any:
        """Return the most recent storage execution report, if any."""
        from ..storage.execution import last_execution_report

        return last_execution_report()

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
            raise KeyError(f"cellData not found in zarr file at {self.z.path}") from e
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

    def _load_assays(self, custom_assay_types: dict | None = None) -> None:
        """This function loads all the assay names present in attribute
        `assayNames` as Assay objects. An attempt is made to automatically
        determine the most appropriate Assay class for each assay based on
        following mapping:

        literal_blocks::
            {'RNA': RNAassay, 'ATAC': ATACassay, 'ADT': ADTassay, 'GeneActivity': RNAassay, 'URNA': RNAassay}

        If an assay name does not match any of the keys above then it is assigned as generic assay class. This can be
        overridden using `predefined_assays` parameter

        Args:
            custom_assay_types: A mapping of assay names to Assay class type to associated with.

        Returns:
        """

        preset_assay_types_map = preset_assay_types()
        caution_statement = (
            "%s was set as a generic Assay with no normalization. If this is unintended "
            "then please make sure that you provide a correct assay type for this assay using "
            "'assay_types' parameter."
        )
        caution_statement = (
            caution_statement
            + "\nIf you have more than one assay in the dataset then you can set "
            "assay_types={'assay1': 'RNA', 'assay2': 'ADT'} "
            "Just replace with actual assay names instead of assay1 and assay2"
        )
        raw_types = self.zw.attrs.get("assayTypes", {})
        z_attrs: dict[str, str] = (
            {str(k): str(v) for k, v in raw_types.items()}
            if isinstance(raw_types, dict)
            else {}
        )
        if custom_assay_types is None:
            custom_assay_types = {}
        for i in self._assayNames:
            if i in custom_assay_types:
                if custom_assay_types[i] in preset_assay_types_map:
                    assay = preset_assay_types_map[custom_assay_types[i]]
                    assay_name = custom_assay_types[i]
                else:
                    logger.warning(
                        f"{custom_assay_types[i]} is not a recognized assay type. Has to be one of "
                        f"{', '.join(list(preset_assay_types_map.keys()))}\nPLease note that the names are"
                        f" case-sensitive."
                    )
                    logger.warning(caution_statement % i)
                    assay = Assay
                    assay_name = "Assay"
                if i in z_attrs and assay_name == z_attrs[i]:
                    pass
                else:
                    z_attrs[i] = assay_name
                    logger.debug(f"Setting assay {i} to assay type: {assay.__name__}")
            elif i in z_attrs:
                assay = preset_assay_types_map[z_attrs[i]]
            else:
                if i in preset_assay_types_map:
                    assay = preset_assay_types_map[i]
                    assay_name = i
                else:
                    logger.warning(caution_statement % i)
                    assay = Assay
                    assay_name = "Assay"
                if i in z_attrs and assay_name == z_attrs[i]:
                    pass
                else:
                    z_attrs[i] = assay_name
                    logger.debug(f"Setting assay {i} to assay type: {assay.__name__}")
            loaded_assay = assay(
                z=self.z,
                workspace=self.workspace,
                name=i,
                cell_data=self.cells,
                nthreads=self.nthreads,
                matrix_root=self._matrix_root_for_assay(i),
                resources=self.resources,
                storageIo=self.storageIo,
            )
            setattr(self, i, loaded_assay)
        if not self.zw.read_only and self.zw.attrs.get("assayTypes") != z_attrs:
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
        """
        if from_assay is None or from_assay == "":
            from_assay = self._defaultAssay
        return cast(
            Assay | RNAassay | ADTassay | ATACassay, self.__getattribute__(from_assay)
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

    def _selection_artifacts_match(
        self,
        first: ArtifactRef,
        second: ArtifactRef,
    ) -> bool:
        if (
            first.kind != second.kind
            or first.scope != second.scope
            or first.assay != second.assay
        ):
            return False
        if first.kind != "cell_selection" or first.scope != "datastore":
            return False
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
            first_inputs = first_status.inputs or {}
            second_inputs = second_status.inputs or {}
            if (
                not first_status.complete
                or not second_status.complete
                or first_inputs.get("ordered_row_ids_fingerprint")
                != second_inputs.get("ordered_row_ids_fingerprint")
            ):
                return False
        except (KeyError, TypeError, ValueError):
            return False
        return first_inputs.get("values_fingerprint") == second_inputs.get(
            "values_fingerprint"
        )

    def _ini_cell_props(
        self,
        min_features: int,
        mito_pattern: str | None,
        ribo_pattern: str | None,
    ) -> None:
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
        if not self.zw.read_only:
            self._filter_cells(min_features)

    def _filter_cells(self, min_features: int) -> None:
        from_assay = self._defaultAssay
        n_features = self.cells.fetch_all(from_assay + "_nFeatures")
        active = self.cells.fetch_all("I")
        if min_features > np.median(n_features[active]):
            logger.warning(
                f"More than half of the cells have fewer than {min_features} features "
                f"for assay: {from_assay}. Will not remove low quality cells automatically."
            )
            return
        keep = (n_features > min_features) & (n_features < np.inf)
        # Write only when filtering changes the active cells.
        if not np.array_equal(keep & active, active):
            self.cells.update_key(keep, key="I")

    @staticmethod
    def _col_renamer(from_assay: str, cell_key: str, suffix: str) -> str:
        """A convenience function for internal usage that creates naming rule
        for the metadata columns.

        Args:
            from_assay: Name of the assay.
            cell_key: Cell key to use.
            suffix: Base name for the column.

        Returns:
            column name updated as per the naming rule
        """
        if cell_key == "I":
            ret_val = "_".join(list(map(str, [from_assay, suffix])))
        else:
            ret_val = "_".join(list(map(str, [from_assay, cell_key, suffix])))
        return ret_val

    def set_default_assay(self, assay_name: str) -> None:
        """Override assigning of default assay.

        Args:
            assay_name: Name of the assay that should be set as default.

        Returns:

        Raises:
            ValueError: if `assay_name` is not found in attribute `assayNames`
        """
        if assay_name not in self.assay_names:
            available = ", ".join(self.assay_names)
            raise ValueError(
                f"Assay '{assay_name}' not found. Available assays: {available}"
            )
        self._defaultAssay = assay_name
        self.zw.attrs["defaultAssay"] = assay_name

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

        Args:
            from_assay: Name of assay to be used.
            cell_key: Boolean column in cell metadata selecting cells. Required; pass ``'I'``
                      for the default active-cell key.
            k: Cell metadata column name or feature name whose values are fetched.
            clip_fraction: Fraction (0-1) for soft percentile clipping of numeric values.

        Returns:
            The requested values
        """
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
            vals = controlled_compute(
                assay.normed(cell_idx, feat_idx).mean(axis=1), self.nthreads
            ).astype(np.float64)
        else:
            vals = self.cells.fetch(k, key=cell_key)
        if clip_fraction < 0 or clip_fraction > 1:
            raise ValueError(
                "ERROR: Value for `clip_fraction` parameter should be between 0 and 1"
            )
        if clip_fraction > 0:
            if vals.dtype in [np.float64, np.uint64]:
                min_v = np.percentile(vals, 100 * clip_fraction)
                max_v = np.percentile(vals, 100 - 100 * clip_fraction)
                vals[vals < min_v] = min_v
                vals[vals > max_v] = max_v
        return vals

    def __repr__(self) -> str:
        def formatter(label: str | None, iter_vals: Iterable[str]) -> str:
            if label is None:
                line = ""
            else:
                line = f"\n{stabs}{label}:"
            line += (
                "\n"
                + dtabs
                + "".join(
                    [
                        f"'{x}', " if n % 5 != 0 else f"'{x}', \n{dtabs}"
                        for n, x in enumerate(iter_vals, start=1)
                    ]
                )
            )
            return line.rstrip("\n\t")[:-2]

        htabs = " " * 3
        stabs = htabs * 2
        dtabs = stabs * 2

        res = (
            f"DataStore has {self.cells.active_index('I').shape[0]} ({self.cells.N}) cells with"
            f" {len(self.assay_names)} assays: {' '.join(self.assay_names)}"
        )
        res = res + f"\n{htabs}Cell metadata:"
        res += formatter(None, self.cells.columns)
        for i in self.assay_names:
            assay = self._get_assay(i)
            res += (
                f"\n{htabs}{i} assay has {assay.feats.N} "
                f"features and following metadata:"
            )
            res += formatter(None, assay.feats.columns)
            assay_group = as_zarr_group(self.zw[i], name=i)
            if "projections" in assay_group:
                targets: list[str] = []
                layouts: list[str] = []
                projections = as_zarr_group(
                    assay_group["projections"], name="projections"
                )
                for j in projections:
                    if isinstance(projections[j], zarr.Group):
                        targets.append(j)
                    else:
                        layouts.append(j)
                if len(targets) > 0:
                    res += formatter("Projected samples", targets)
                if len(layouts) > 0:
                    res += formatter("Co-embeddings", layouts)
        return res

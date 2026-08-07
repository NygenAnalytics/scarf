import numpy as np
import zarr
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from ..storage.artifacts import (
    ArtifactRef,
    ArtifactScope,
    ArtifactStatus,
    ValueFingerprintBuilder,
    artifact_path,
    canonical_bytes,
    fingerprint_array,
    fingerprint_strings,
    inspect_artifact,
    list_artifacts as list_artifact_refs,
)
from ..storage.types import ZarrMode, as_zarr_array, as_zarr_group
from ..storage.budget import ResourceBudget
from ..assay import RNAassay, ATACassay, ADTassay, Assay
from ..assay.base import _defer_feature_props
from ..metadata import MetaData
from ..metadata.artifacts import (
    artifact_values,
    link_cell_data_column,
    plan_cell_data_artifact,
    write_cell_data_artifact,
)
from ..storage.schema import validate_assay_name
from ..storage.profiles import StorageProfile, ZarrLocation
from ..storage.stores import load_zarr, resolve_matrix_source
from ..storage.selections import resolve_selection_artifact
from ..utils.compute import controlled_compute
from ..utils.logging import logger

if TYPE_CHECKING:
    from ..graph.state import AssayState
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
    if workspace is None:
        zw = z
    else:
        zw = as_zarr_group(z[workspace], name=workspace)
    if assay_name not in zw:
        raise KeyError(f"ERROR: {assay_name} not found in zarr file")
    assay_zw = as_zarr_group(zw[assay_name], name=assay_name)
    if "featureData" not in assay_zw:
        raise KeyError(f"ERROR: 'featureData' not found in {assay_name}")
    if workspace is None:
        matrix_assay = as_zarr_group(matrix_root[assay_name], name=assay_name)
        if "counts" not in matrix_assay:
            raise KeyError(f"ERROR: 'counts' not found in {assay_name}")
    else:
        if "matrices" not in matrix_root:
            raise KeyError("ERROR: Workspace defined but no 'matrices' slot found")
        matrices = as_zarr_group(matrix_root["matrices"], name="matrices")
        if assay_name not in matrices:
            raise KeyError(f"ERROR: {assay_name} not found in workspace matrices slot")
        matrix_assay = as_zarr_group(matrices[assay_name], name=assay_name)
        if "counts" not in matrix_assay:
            raise KeyError(
                f"ERROR: 'counts' not found in {assay_name} in workspace matrices slot"
            )
    return True


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
        min_cells_per_feature: Minimum number of cells where a feature has a non-zero value. Genes with values
                               less than this will be filtered out
        mito_pattern: Regex pattern to capture mitochondrial genes. When None, uses ``MT-|mt``.
        ribo_pattern: Regex pattern to capture ribosomal genes. When None, uses
                      ``RPS|RPL|MRPS|MRPL``.
        nthreads: Number of maximum threads to use in all multi-threaded functions
        zarr_mode: For read-write mode use r+' or for read-only use 'r' (Default value: 'r+')
        workspace: Workspace name within the Zarr store (None for legacy single-workspace layout).

    Attributes:
        cells: MetaData object with cells and info about each cell (e. g. RNA_nCounts ids).
        nthreads: Number of threads to use for this datastore instance.
        z: The Zarr file (directory) used for this datastore instance.
    """

    def __init__(
        self,
        zarr_loc: ZarrLocation,
        assay_types: dict[str, str],
        default_assay: str,
        min_features_per_cell: int,
        min_cells_per_feature: int,
        mito_pattern: str,
        ribo_pattern: str,
        zarr_mode: ZarrMode,
        workspace: str | None,
        resources: ResourceBudget,
        storage_profile: StorageProfile,
        storage_options: dict[str, Any] | None = None,
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
        _ = self.assay_names
        # The order is critical here:
        self.cells = self._load_cells()
        self._defaultAssay = self._load_default_assay(default_assay)
        self._load_assays(min_cells_per_feature, assay_types)
        # TODO: Reset all attrs, pca, dendrogram etc
        self._ini_cell_props(min_features_per_cell, mito_pattern, ribo_pattern)
        self._cachedMagicOperator = None
        self._cachedMagicOperatorLoc = None
        self._integratedGraphsLoc = "integratedGraphs"
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
        )

    def get_assay_state(self, from_assay: str | None = None) -> "AssayState | None":
        """Return the selected artifact state for one assay."""
        from ..graph.state import read_assay_state

        assay = from_assay or self._defaultAssay
        if assay is None:
            raise ValueError("No assay was provided and no default is configured")
        return read_assay_state(self.zw, assay)

    def list_artifacts(
        self,
        *,
        kind: str | None = None,
        from_assay: str | None = None,
        scope: ArtifactScope = "assay",
        complete_only: bool = False,
    ) -> list[ArtifactRef]:
        """List logical artifact references in one scope."""
        if scope == "assay" and from_assay is None:
            from_assay = self._defaultAssay
        return list_artifact_refs(
            self.zw,
            scope=scope,
            assay=from_assay,
            kind=kind,
            complete_only=complete_only,
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
        """Load all assay names present in the Zarr file. Zarr writers create
        an 'is_assay' attribute in the assay level and this function looks for
        presence of those attributes to load assay names.

        Returns:
            Names of assays present in a Zarr file
        """
        assays = []
        # Object-store listings can repeat a group and may not preserve order
        # across calls, so keep unique names in sorted order.
        for i in sorted(dict.fromkeys(self.zw.group_keys())):
            if "is_assay" in self.zw[i].attrs.keys():
                validate_assay_name(i)
                sanitize_hierarchy(
                    self.z,
                    i,
                    self.workspace,
                    matrix_root=self._matrix_z,
                )
                assays.append(i)
        return assays

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
                    if self.zarr_mode == "r+":
                        self.zw.attrs["defaultAssay"] = assay_name
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
                if self.zarr_mode == "r+":
                    self.zw.attrs["defaultAssay"] = assay_name
            else:
                raise ValueError(
                    f"ERROR: The provided default assay name: {assay_name} was not found. "
                    f"Please Choose one from: {' '.join(self.assay_names)}\n"
                    "Please note that the names are case-sensitive."
                )
        assert assay_name is not None
        return assay_name

    def _load_assays(
        self, min_cells: int, custom_assay_types: dict | None = None
    ) -> None:
        """This function loads all the assay names present in attribute
        `assayNames` as Assay objects. An attempt is made to automatically
        determine the most appropriate Assay class for each assay based on
        following mapping:

        literal_blocks::
            {'RNA': RNAassay, 'ATAC': ATACassay, 'ADT': ADTassay, 'GeneActivity': RNAassay, 'URNA': RNAassay}

        If an assay name does not match any of the keys above then it is assigned as generic assay class. This can be
        overridden using `predefined_assays` parameter

        Args:
            min_cells: Minimum number of cells that a feature in each assay must be present to not be discarded (i.e.
                       receive False value in `I` column)
            custom_assay_types: A mapping of assay names to Assay class type to associated with.

        Returns:
        """

        preset_assay_types = {
            "RNA": RNAassay,
            "ATAC": ATACassay,
            "ADT": ADTassay,
            "HTO": ADTassay,
            "CRISPR": Assay,
            "ANTIGEN": Assay,
            "CUSTOM": Assay,
            "GeneActivity": RNAassay,
            "GeneScores": RNAassay,
            "URNA": RNAassay,
            "Assay": Assay,
        }
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
        if "assayTypes" not in self.zw.attrs:
            self.zw.attrs["assayTypes"] = {}
        raw_types = self.zw.attrs["assayTypes"]
        z_attrs: dict[str, str] = (
            {str(k): str(v) for k, v in raw_types.items()}
            if isinstance(raw_types, dict)
            else {}
        )
        if custom_assay_types is None:
            custom_assay_types = {}
        for i in self.assay_names:
            if i in custom_assay_types:
                if custom_assay_types[i] in preset_assay_types:
                    assay = preset_assay_types[custom_assay_types[i]]
                    assay_name = custom_assay_types[i]
                else:
                    logger.warning(
                        f"{custom_assay_types[i]} is not a recognized assay type. Has to be one of "
                        f"{', '.join(list(preset_assay_types.keys()))}\nPLease note that the names are"
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
                assay = preset_assay_types[z_attrs[i]]
            else:
                if i in preset_assay_types:
                    assay = preset_assay_types[i]
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
            with _defer_feature_props():
                loaded_assay = assay(
                    z=self.z,
                    workspace=self.workspace,
                    name=i,
                    cell_data=self.cells,
                    min_cells_per_feature=min_cells,
                    nthreads=self.nthreads,
                    matrix_root=self._matrix_z,
                    resources=self.resources,
                )
            setattr(self, i, loaded_assay)
        if self.zw.attrs["assayTypes"] != z_attrs:
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

    def _get_latest_feat_key(self, from_assay: str) -> str:
        """Looks up the value in assay level attributes for key
        'latest_feat_key'.

        Args:
            from_assay: Assay whose latest feature is to be returned.

        Returns:
            Name of the latest feature that was used to run `save_normalized_data`
        """
        from ..graph.state import read_assay_state

        state = read_assay_state(self.zw, from_assay)
        if state is not None:
            return state.feat_key
        assay = self._get_assay(from_assay)
        return cast(str, assay.attrs["latest_feat_key"])

    def _get_latest_cell_key(self, from_assay: str) -> str:
        """Looks up the value in assay level attributes for key
        'latest_cell_key'.

        Args:
            from_assay: Assay whose latest feature is to be returned.

        Returns:
            Name of the latest feature that was used to run `save_normalized_data`
        """
        from ..graph.state import read_assay_state

        state = read_assay_state(self.zw, from_assay)
        if state is not None:
            return state.cell_key
        assay = self._get_assay(from_assay)
        return cast(str, assay.attrs.get("latest_cell_key", "I"))

    def _ensure_dataset_fingerprint(self, from_assay: str) -> str:
        assay = self._get_assay(from_assay)
        existing = assay.attrs.get("dataset_fingerprint")
        if existing is not None:
            return str(existing)
        if self.zarr_mode != "r+":
            raise PermissionError(
                "dataset_fingerprint is missing and cannot be stored read-only"
            )
        fingerprint = self._calculate_dataset_fingerprint(from_assay)
        assay.attrs["dataset_fingerprint"] = fingerprint
        return fingerprint

    def _calculate_dataset_fingerprint(self, from_assay: str) -> str:
        assay = self._get_assay(from_assay)
        builder = ValueFingerprintBuilder()
        builder.update_bytes(
            "dataset",
            canonical_bytes(
                {
                    "assay": from_assay,
                    "shape": list(assay.rawData.shape),
                    "dtype": np.dtype(assay.rawData.dtype).str,
                }
            ),
        )
        builder.update_array(
            "cell_ids",
            np.asarray(self.cells.fetch_all("ids")).astype(str),
        )
        builder.update_array(
            "feature_ids",
            np.asarray(assay.feats.fetch_all("ids")).astype(str),
        )
        builder.update_array(
            "cell_n_counts",
            np.asarray(self.cells.fetch_all(f"{from_assay}_nCounts")),
        )
        builder.update_array(
            "cell_n_features",
            np.asarray(self.cells.fetch_all(f"{from_assay}_nFeatures")),
        )
        builder.update_array(
            "feature_n_cells",
            np.asarray(assay.feats.fetch_all("nCells")),
        )
        return builder.hexdigest()

    def _record_cell_selection(
        self,
        *,
        column: str,
        operation: str,
        parameters: dict[str, Any],
        inputs: dict[str, Any],
        invalidate_cache: bool = False,
    ) -> ArtifactRef:
        ref = resolve_selection_artifact(
            self.zw,
            scope="datastore",
            kind="cell_selection",
            values=np.asarray(self.cells.fetch_all(column)),
            row_ids=np.asarray(self.cells.fetch_all("ids")),
            operation=operation,
            parameters=parameters,
            inputs=inputs,
            source_column=column,
            invalidate_cache=invalidate_cache,
        )
        column_array = as_zarr_array(
            as_zarr_group(self.zw["cellData"], name="cellData")[column],
            name=column,
        )
        column_array.attrs["source_artifact"] = ref.to_dict()
        column_array.attrs["source_value"] = "values"
        return ref

    def _linked_cell_selection(self, column: str) -> ArtifactRef | None:
        cell_data = as_zarr_group(self.zw["cellData"], name="cellData")
        column_array = as_zarr_array(cell_data[column], name=column)
        raw_ref = column_array.attrs.get("source_artifact")
        if not isinstance(raw_ref, dict):
            return None
        try:
            ref = ArtifactRef.from_dict(raw_ref)
            status = self.inspect_artifact(ref)
        except (KeyError, TypeError, ValueError):
            return None
        if (
            ref.kind != "cell_selection"
            or ref.scope != "datastore"
            or not status.complete
        ):
            return None
        inputs = status.inputs or {}
        if inputs.get("ordered_row_ids_fingerprint") != fingerprint_strings(
            np.asarray(self.cells.fetch_all("ids"))
        ):
            return None
        group = as_zarr_group(self.zw[status.path], name=status.path)
        if "values" not in group:
            return None
        stored = np.asarray(as_zarr_array(group["values"], name="values")[:])
        current = np.asarray(self.cells.fetch_all(column))
        return (
            ref
            if stored.ndim == 1
            and stored.dtype == np.dtype(bool)
            and current.ndim == 1
            and current.dtype == np.dtype(bool)
            and stored.shape == current.shape
            and np.array_equal(stored, current)
            else None
        )

    def _ensure_cell_selection(self, column: str) -> ArtifactRef:
        ref = self._linked_cell_selection(column)
        if ref is not None:
            return ref
        return self._record_cell_selection(
            column=column,
            operation="manual_selection",
            parameters={},
            inputs={},
        )

    def _selection_artifacts_match(
        self,
        first: ArtifactRef,
        second: ArtifactRef,
    ) -> bool:
        if first == second:
            return True
        if (
            first.kind != second.kind
            or first.scope != second.scope
            or first.assay != second.assay
        ):
            return False
        try:
            first_status = inspect_artifact(self.zw, first)
            second_status = inspect_artifact(self.zw, second)
            if (
                not first_status.complete
                or not second_status.complete
                or (first_status.inputs or {}).get("ordered_row_ids_fingerprint")
                != (second_status.inputs or {}).get("ordered_row_ids_fingerprint")
            ):
                return False
            first_group = as_zarr_group(
                self.zw[artifact_path(first)],
                name=artifact_path(first),
            )
            second_group = as_zarr_group(
                self.zw[artifact_path(second)],
                name=artifact_path(second),
            )
            first_values = np.asarray(
                as_zarr_array(
                    first_group["values"],
                    name="values",
                )[:]
            )
            second_values = np.asarray(
                as_zarr_array(
                    second_group["values"],
                    name="values",
                )[:]
            )
        except (KeyError, TypeError, ValueError):
            return False
        return (
            first_values.ndim == 1
            and second_values.ndim == 1
            and first_values.dtype == np.dtype(bool)
            and second_values.dtype == np.dtype(bool)
            and first_values.shape == second_values.shape
            and np.array_equal(first_values, second_values)
        )

    def _resolve_cell_data_input(
        self,
        column: str,
        *,
        cell_key: str,
    ) -> ArtifactRef:
        selection = self._ensure_cell_selection(cell_key)
        current = np.asarray(self.cells.fetch(column, key=cell_key))
        cell_data = as_zarr_group(self.zw["cellData"], name="cellData")
        column_array = as_zarr_array(cell_data[column], name=column)
        raw_ref = column_array.attrs.get("source_artifact")
        if isinstance(raw_ref, dict):
            try:
                ref = ArtifactRef.from_dict(raw_ref)
                status = self.inspect_artifact(ref)
            except (KeyError, TypeError, ValueError):
                pass
            else:
                inputs = status.inputs or {}
                source_value = column_array.attrs.get(
                    "source_value",
                    "values",
                )
                value_index = column_array.attrs.get("value_index")
                if (
                    status.complete
                    and inputs.get("cell_selection") == selection.to_dict()
                    and isinstance(source_value, str)
                ):
                    group = as_zarr_group(
                        self.zw[status.path],
                        name=status.path,
                    )
                    if source_value in group:
                        stored = artifact_values(
                            group,
                            source_value,
                            (
                                int(value_index)
                                if isinstance(value_index, int)
                                else None
                            ),
                        )
                        if np.array_equal(stored, current):
                            return ref
        values_fingerprint = (
            fingerprint_strings(current)
            if current.dtype.kind in {"O", "S", "U"}
            else fingerprint_array(current)
        )
        planned = plan_cell_data_artifact(
            self.zw,
            scope="datastore",
            kind="metadata_snapshot",
            operation="manual_cell_data",
            parameters={"dtype": str(current.dtype)},
            inputs={"values_fingerprint": values_fingerprint},
            execution_options={"source_column": column},
            cell_selection=selection,
            arrays={
                "values": (
                    current.shape,
                    (
                        None
                        if current.dtype.kind in {"O", "S", "U"}
                        else current.dtype.kind
                    ),
                )
            },
        )
        write_cell_data_artifact(
            self.zw,
            planned,
            {"values": current},
        )
        link_cell_data_column(
            self.zw,
            column,
            planned.ref,
            value_name="values",
        )
        return planned.ref

    def _resolve_cell_data_provenance_input(
        self,
        column: str,
        *,
        cell_key: str,
    ) -> dict[str, Any]:
        ref = self._resolve_cell_data_input(column, cell_key=cell_key)
        column_array = as_zarr_array(
            as_zarr_group(self.zw["cellData"], name="cellData")[column],
            name=column,
        )
        source_value = column_array.attrs.get("source_value", "values")
        if not isinstance(source_value, str):
            raise TypeError("Cell-data source_value must be a string")
        raw_index = column_array.attrs.get("value_index")
        if raw_index is not None and (
            isinstance(raw_index, bool) or not isinstance(raw_index, int | np.integer)
        ):
            raise TypeError("Cell-data value_index must be an integer")
        return {
            "artifact": ref.to_dict(),
            "source_value": source_value,
            "value_index": int(raw_index) if raw_index is not None else None,
        }

    def _ini_cell_props(
        self,
        min_features: int,
        mito_pattern: str | None,
        ribo_pattern: str | None,
    ) -> None:
        """This function is called on class initialization. For each assay, it
        calculates per-cell statistics i.e. nCounts, nFeatures, percentMito and
        percentRibo. These statistics are then populated into the cell metadata
        table.

        Args:
            min_features: Minimum features that a cell must have non-zero value before being filtered out.
            mito_pattern: Regex pattern for identification of mitochondrial genes.
            ribo_pattern: Regex pattern for identification of ribosomal genes.

        Returns:
        """
        for from_assay in self.assay_names:
            assay = self._get_assay(from_assay)

            n_counts_name = from_assay + "_nCounts"
            compute_n_counts = n_counts_name not in self.cells.columns
            n_features_name = from_assay + "_nFeatures"
            compute_n_features = n_features_name not in self.cells.columns
            pending_min_cells = assay._deferred_min_cells_per_feature
            compute_n_cells = pending_min_cells is not None

            percent_feature_indices: dict[str, np.ndarray] = {}
            if isinstance(assay, RNAassay):
                if mito_pattern != "":
                    resolved_mito_pattern = (
                        "MT-|mt" if mito_pattern is None else mito_pattern
                    )
                    percent_mito_name = from_assay + "_percentMito"
                    mito_idx = assay._plan_percent_feature(
                        resolved_mito_pattern,
                        percent_mito_name,
                    )
                    if mito_idx is not None:
                        percent_feature_indices[percent_mito_name] = mito_idx

                if ribo_pattern != "":
                    resolved_ribo_pattern = (
                        "RPS|RPL|MRPS|MRPL" if ribo_pattern is None else ribo_pattern
                    )
                    percent_ribo_name = from_assay + "_percentRibo"
                    ribo_idx = assay._plan_percent_feature(
                        resolved_ribo_pattern,
                        percent_ribo_name,
                    )
                    if ribo_idx is not None:
                        percent_feature_indices[percent_ribo_name] = ribo_idx

            stats: dict[str, np.ndarray] = {}
            if (
                compute_n_counts
                or compute_n_features
                or compute_n_cells
                or percent_feature_indices
            ):
                stats = assay._stream_initialization_stats(
                    compute_n_counts=compute_n_counts,
                    compute_n_features=compute_n_features,
                    compute_n_cells=compute_n_cells,
                    percent_feature_indices=percent_feature_indices,
                )

            if pending_min_cells is not None:
                assay._store_feature_props(stats["nCells"], pending_min_cells)

            computed_n_counts: np.ndarray | None = None
            if compute_n_counts:
                n_c = stats["nCounts"]
                computed_n_counts = n_c.astype(np.float64)
                self.cells.insert(
                    n_counts_name,
                    computed_n_counts,
                    overwrite=True,
                )
                if isinstance(assay, RNAassay):
                    min_nc = min(n_c)
                    if min(n_c) < assay.sf:
                        logger.warning(
                            f"Minimum cell count ({min_nc}) is lower than "
                            f"size factor multiplier ({assay.sf})"
                        )

            if compute_n_features:
                self.cells.insert(
                    n_features_name,
                    stats["nFeatures"].astype(np.float64),
                    overwrite=True,
                )

            for name in percent_feature_indices:
                assay._write_percent_feature(
                    name,
                    stats[name],
                    n_counts=computed_n_counts,
                )

            if assay._deferred_min_cells_per_feature is not None:
                raise RuntimeError(
                    f"({from_assay}) Deferred feature initialization was not completed"
                )

            if from_assay == self._defaultAssay:
                v = self.cells.fetch(from_assay + "_nFeatures", key="I")
                if min_features > np.median(v):
                    logger.warning(
                        f"More than half of the cells have fewer than {min_features} features "
                        f"for assay: {from_assay}. Will not remove low quality cells automatically."
                    )
                else:
                    bv = self.cells.sift(
                        from_assay + "_nFeatures", min_features, np.inf
                    )
                    # Making sure that the write operation is only done if the filtering results have changed
                    cur_index = self.cells.fetch_all("I")
                    nbv = bv & cur_index
                    if all(nbv == cur_index) is False:
                        self.cells.update_key(bv, key="I")

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
                f"\n{htabs}{i} assay has {assay.feats.fetch_all('I').sum()} ({assay.feats.N}) "
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

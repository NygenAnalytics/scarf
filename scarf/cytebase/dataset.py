"""One catalog dataset: details, read-only access, and precomputed CELLxGENE results."""

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._storage import dataset_prefix

if TYPE_CHECKING:
    import pandas as pd

    from scarf import DataStore
    from scarf.plotting import PlotResult
    from scarf.storage.artifacts import ArtifactRef

    from .catalog import Catalog

_ASSAY = "RNA"
_SUMMARY_FACETS = (
    ("Organisms", "organism_labels"),
    ("Assays", "assay_labels"),
    ("Tissues", "tissue_labels"),
    ("Diseases", "disease_labels"),
    ("Suspension", "suspension_types"),
)
# CELLxGENE citations join the publication, dataset version, and collection on one line.
_CITATION_BREAKS = re.compile(r"\s+(?=Dataset Version:|curated and distributed by)")


class CytebaseDataset:
    """Handle to one catalog dataset that opens its remote store only when needed.

    The Scarf store is opened read-only through :meth:`Catalog.open_dataset`, so
    provenance is checked before any array is read. Embeddings and cell metadata
    come from the CELLxGENE source as imported by the Cytebase pipeline.
    """

    def __init__(self, catalog: "Catalog", row: dict[str, Any]) -> None:
        self._catalog = catalog
        self.row = dict(row)
        self._record: dict | None = None
        self._datastore: DataStore | None = None

    @property
    def id(self) -> str:
        return str(self.row["cytebase_id"])

    @property
    def title(self) -> str | None:
        return self.row.get("title")

    @property
    def citation(self) -> str | None:
        return self.row.get("citation")

    @property
    def cell_count(self) -> int | None:
        return self.row.get("cell_count")

    def record(self) -> dict:
        """Return the published ``dataset.json`` record, cached after the first read."""
        if self._record is None:
            path = f"{dataset_prefix(self.id)}/dataset.json"
            record = self._catalog._storage.read_json(path)
            if record is None:
                raise KeyError(f"No dataset record is published for {self.id!r}")
            self._record = record
        return self._record

    def source_embeddings(self) -> list[str]:
        """List every ``obsm`` key in the source H5AD, imported or not."""
        inspection = self.record().get("inspection") or {}
        return list(inspection.get("embeddings") or [])

    def describe(self) -> str:
        """Summarize the dataset as Markdown without opening its Zarr store."""
        row = self.row
        lines = [f"### {self.title or self.id}", "", f"- **Cytebase ID:** `{self.id}`"]
        if self.citation:
            parts = _CITATION_BREAKS.split(self.citation.strip())
            if len(parts) == 1:
                lines.append(f"- **Citation:** {parts[0]}")
            else:
                lines.append("- **Citation:**")
                lines.extend(f"    - {part}" for part in parts)
        if row.get("doi"):
            lines.append(f"- **DOI:** {row['doi']}")
        for label, key in (
            ("CELLxGENE", "cellxgene_url"),
            ("Explorer", "explorer_url"),
        ):
            if row.get(key):
                lines.append(f"- **{label}:** {row[key]}")
        counts = []
        if row.get("cell_count") is not None:
            counts.append(f"{row['cell_count']:,} cells")
        if row.get("n_genes") is not None:
            counts.append(f"{row['n_genes']:,} genes")
        if counts:
            lines.append(f"- **Size:** {', '.join(counts)}")
        lines.append(f"- **Status:** {row.get('status')}")
        for label, key in _SUMMARY_FACETS:
            values = row.get(key) or []
            if values:
                lines.append(f"- **{label}:** {', '.join(map(str, values))}")
        try:
            embeddings = self.source_embeddings()
        except Exception:  # noqa: BLE001 - the summary must render without the record
            embeddings = []
        if embeddings:
            lines.append(f"- **Source embeddings:** {', '.join(embeddings)}")
        return "\n".join(lines)

    def _repr_markdown_(self) -> str:
        return self.describe()

    def __repr__(self) -> str:
        return f"CytebaseDataset({self.id!r})"

    def open(self, **datastore_options: Any) -> "DataStore":
        """Open the remote store read-only, reusing the store on later calls."""
        if self._datastore is None:
            self._datastore = self._catalog.open_dataset(self.id, **datastore_options)
        elif datastore_options:
            raise ValueError(
                "This dataset is already open; create a new handle to use other options"
            )
        return self._datastore

    def mount(self, at: str | Path, **datastore_options: Any) -> "DataStore":
        """Create or reopen a writable local analysis over the remote counts."""
        return self._catalog.mount_dataset(self.id, at, **datastore_options)

    def cell_metadata(self, columns: list[str] | None = None) -> "pd.DataFrame":
        """Return CELLxGENE ``obs`` annotations and other cell columns as a DataFrame."""
        cells = self.open().cells
        return cells.to_pandas_dataframe(
            list(columns) if columns is not None else list(cells.columns), key="I"
        )

    def embeddings(self) -> dict[str, "ArtifactRef"]:
        """Map each imported source embedding key, such as ``X_umap``, to its artifact."""
        datastore = self.open()
        found = {}
        for key in self.source_embeddings():
            refs = datastore.list_artifacts(
                kind="embedding",
                from_assay=_ASSAY,
                operation="import_dimreduc",
                complete_only=True,
                parameters={"dimreduc_key": key},
            )
            if len(refs) == 1:
                found[key] = refs[0]
        return found

    def embedding(self, key: str = "X_umap") -> "ArtifactRef":
        """Return the artifact reference for one imported embedding."""
        available = self.embeddings()
        if key not in available:
            names = ", ".join(available) or "none"
            raise KeyError(
                f"Embedding {key!r} was not imported for {self.id!r}; available: {names}"
            )
        return available[key]

    def embedding_coordinates(self, key: str = "X_umap") -> "pd.DataFrame":
        """Return embedding coordinates indexed by cell ID for custom plotting."""
        import numpy as np
        import pandas as pd

        from scarf.storage.artifacts import ArtifactRef
        from scarf.storage.selections import read_stored_selection_indices
        from scarf.storage.types import as_zarr_array

        datastore = self.open()
        ref = self.embedding(key)
        status = datastore.inspect_artifact(ref)
        selection = (status.inputs or {}).get("cell_selection")
        if selection is None:
            raise ValueError(f"Embedding {key!r} has no cell-selection input")
        cell_idx = read_stored_selection_indices(
            datastore.zw,
            ArtifactRef.from_dict(selection),
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        )
        values = np.asarray(
            as_zarr_array(datastore.load_artifact(ref)["values"], name="values")[:]
        )
        if values.ndim != 2 or values.shape[0] != len(cell_idx):
            raise ValueError(f"Embedding {key!r} does not match its cell selection")
        role = (status.parameters or {}).get("role") or key.removeprefix("X_")
        ids = datastore.cells.fetch_all("ids")[cell_idx]
        return pd.DataFrame(
            values,
            index=pd.Index(ids, name="ids"),
            columns=[f"{role}_{i + 1}" for i in range(values.shape[1])],
        )

    def plot_embedding(
        self,
        color_by: Any = None,
        *,
        key: str = "X_umap",
        **plot_options: Any,
    ) -> "PlotResult":
        """Plot an imported embedding colored by a cell column or gene name.

        Remaining keyword arguments are passed to ``DataStore.plots.embedding``.
        """
        datastore = self.open()
        return datastore.plots.embedding(
            layout=self.embedding(key), color_by=color_by, **plot_options
        )

"""Descriptive catalog entries, separate from opened Scarf DataStores."""

import re
from typing import TYPE_CHECKING, Any

from ._storage import dataset_prefix

if TYPE_CHECKING:
    from .catalog import Catalog

_SUMMARY_FACETS = (
    ("Organisms", "organism_labels"),
    ("Assays", "assay_labels"),
    ("Tissues", "tissue_labels"),
    ("Diseases", "disease_labels"),
    ("Suspension", "suspension_types"),
)
# CELLxGENE citations join the publication, dataset version, and collection on one line.
_CITATION_BREAKS = re.compile(r"\s+(?=Dataset Version:|curated and distributed by)")


class DatasetEntry:
    """Describe a registered dataset without opening its Zarr hierarchy.

    Catalog fields and source provenance belong to this entry. Open or mount a
    DataStore through the catalog to read cells, assays, and analysis artifacts.
    """

    def __init__(self, catalog: "Catalog", row: dict[str, Any]) -> None:
        self._catalog = catalog
        self.row = dict(row)
        self._record: dict | None = None

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
        return f"DatasetEntry({self.id!r})"

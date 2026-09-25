"""One verified local copy of the latest self-contained DuckDB catalog."""

import hashlib
import os
import re
import sys
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any

from scarf.utils.logging import logger

from ._storage import Bucket
from .display import CatalogResults

if TYPE_CHECKING:
    import duckdb

    from scarf import DataStore

    from .dataset import CytebaseDataset

_CATALOG_PATH = "catalog/cytebase.duckdb"
_HASH_PATH = f"{_CATALOG_PATH}.sha256"
_HASH_PATTERN = re.compile(rb"([0-9a-fA-F]{64})  cytebase\.duckdb(?:\r?\n)?")
_DOWNLOAD_ATTEMPTS = 3
FACETS = (
    "tissue",
    "organ",
    "disease",
    "assay",
    "organism",
    "cell_type",
    "sex",
    "development_stage",
    "suspension_type",
)
_FACET_COLUMNS = {
    facet: f"{facet}_labels" if facet != "suspension_type" else "suspension_types"
    for facet in FACETS
}
_READY_PREDICATE = (
    "status = 'ready' AND processed_version_id = latest_version_id "
    "AND zarr_uri IS NOT NULL"
)
_DATASET_ORDER = " ORDER BY cell_count DESC NULLS LAST, cytebase_id"
_SEARCH_TEXT = " || ' ' || ".join(
    [
        "coalesce(cytebase_id, '')",
        "coalesce(title, '')",
        "coalesce(citation, '')",
        "coalesce(first_author, '')",
        *(
            f"coalesce(array_to_string({column}, ' '), '')"
            for column in dict.fromkeys(_FACET_COLUMNS.values())
        ),
    ]
)
_SEARCH_COLUMNS = (
    "cytebase_id",
    "title",
    "cell_count",
    "tissue_labels",
    "disease_labels",
)


def _cache_directory() -> Path:
    if sys.platform == "win32":
        base = (
            Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
            / "scarf"
        )
    else:
        base = Path.home() / ".scarf"
    return base


def _parse_hash(raw: bytes) -> str:
    match = _HASH_PATTERN.fullmatch(raw)
    if match is None:
        raise ValueError("Expected a SHA-256 digest followed by '  cytebase.duckdb'")
    return match[1].decode("ascii").lower()


def _remote_hash(storage: Bucket) -> str:
    raw = storage.read_bytes(_HASH_PATH)
    if raw is None:
        raise RuntimeError(
            f"Missing {_HASH_PATH}. Ask the publisher to rebuild the catalog "
            "before querying it."
        )
    try:
        return _parse_hash(raw)
    except ValueError as error:
        raise RuntimeError(
            f"Malformed {_HASH_PATH}. Ask the publisher to rebuild the catalog "
            "before querying it."
        ) from error


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _save_hash(path: Path, digest: str) -> None:
    """Publish the computed checksum only after its database has been verified."""
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="wb", prefix=".checksum-", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(f"{digest}  cytebase.duckdb\n".encode("ascii"))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            temporary.replace(path)
        except PermissionError as error:
            # A concurrent Windows reader may briefly hold this sidecar open.
            # Another writer's identical committed checksum is sufficient.
            try:
                saved = _parse_hash(path.read_bytes())
            except (FileNotFoundError, ValueError):
                raise error
            if saved != digest:
                raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _verified_cache(path: Path, expected: str) -> bool:
    """Verify bytes even when a saved checksum exists; repair only the sidecar."""
    if not path.is_file():
        return False
    try:
        actual = _file_hash(path)
    except FileNotFoundError:
        return False
    if actual != expected:
        return False
    hash_path = path.with_suffix(".duckdb.sha256")
    try:
        saved = _parse_hash(hash_path.read_bytes())
    except (FileNotFoundError, ValueError):
        saved = None
    if saved != actual:
        _save_hash(hash_path, actual)
    return True


def _install(temporary: Path, destination: Path, digest: str) -> None:
    # Another process can complete the same download while this one is running.
    if _verified_cache(destination, digest):
        return
    try:
        temporary.replace(destination)
    except PermissionError as error:
        # Windows can reject replacement if another reader opened the catalog
        # between the check above and the rename. A verified copy is sufficient.
        if _verified_cache(destination, digest):
            return
        raise RuntimeError(
            f"Cannot refresh catalog cache at {destination}. Close any "
            "connections using this catalog and retry."
        ) from error
    _save_hash(destination.with_suffix(".duckdb.sha256"), digest)


def _cached_catalog(storage: Bucket) -> Path:
    """Check the remote hash on every access and return the verified local catalog.

    Authentication or remote-check failures propagate instead of serving stale
    data. Each successful return has a locally computed, saved SHA-256 checksum.
    Catalog publication races allow three download attempts with bounded backoff.
    Only the latest verified catalog is kept, shared across configured buckets.
    """
    root = _cache_directory()
    destination = root / "cytebase.duckdb"
    for attempt in range(_DOWNLOAD_ATTEMPTS):
        expected = _remote_hash(storage)
        if _verified_cache(destination, expected):
            return destination

        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if attempt == 0 and destination.is_file():
            logger.info("Catalog checksum mismatch; refreshing local cache.")
        temporary = None
        try:
            with NamedTemporaryFile(
                prefix=".catalog-", suffix=".download", dir=root, delete=False
            ) as handle:
                temporary = Path(handle.name)
            storage.download(_CATALOG_PATH, temporary)
            temporary.chmod(0o600)
            actual = _file_hash(temporary)
            current = _remote_hash(storage)
            if actual == expected == current:
                _install(temporary, destination, actual)
                return destination
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

        if attempt + 1 < _DOWNLOAD_ATTEMPTS:
            logger.debug("Catalog changed during download; checking again")
            time.sleep(attempt + 1)

    raise RuntimeError(
        "The downloaded catalog did not match its published SHA-256 after three "
        "attempts. Retry when catalog publication has finished; if the mismatch "
        "persists, ask the publisher to rebuild the catalog. The previous local "
        "catalog, if present, has been retained."
    )


class Catalog:
    """Discover and open datasets in the public Cytebase catalog or another bucket.

    ``bucket`` accepts ``namespace/name`` or an HF bucket URI and otherwise uses
    ``CYTEBASE_BUCKET``, falling back to the public ``Nygen/cytebase`` bucket.
    ``token=None`` uses standard Hugging Face authentication when available;
    ``token=False`` explicitly selects anonymous access. Construction prepares a
    verified local catalog. Each new catalog query checks the published checksum
    again and opens a verified local snapshot read-only.
    """

    def __init__(
        self, bucket: str | None = None, token: str | bool | None = None
    ) -> None:
        selected_bucket = (
            bucket
            if bucket is not None
            else os.environ.get("CYTEBASE_BUCKET", "Nygen/cytebase")
        )
        self._storage = Bucket(selected_bucket, token)
        _cached_catalog(self._storage)

    def connect_catalog(self) -> "duckdb.DuckDBPyConnection":
        """Open a verified local snapshot; callers must close the connection."""
        try:
            import duckdb
        except ImportError as error:
            raise ImportError(
                "Catalog queries require the scarf[cytebase] extra"
            ) from error
        return duckdb.connect(str(_cached_catalog(self._storage)), read_only=True)

    def query(
        self,
        sql: str,
        parameters: list | dict | None = None,
        *,
        max_cell_chars: int | None = 100,
    ) -> CatalogResults:
        """Query the local catalog; ``max_cell_chars=None`` shows full cell values."""
        with self.connect_catalog() as connection:
            result = connection.execute(sql, parameters)
            columns = [column[0] for column in result.description]
            return CatalogResults(
                [dict(zip(columns, row, strict=True)) for row in result.fetchall()],
                columns=columns,
                max_cell_chars=max_cell_chars,
            )

    def find_datasets(
        self,
        *,
        ready_only: bool = True,
        max_cell_chars: int | None = 100,
        **facets: str | list[str],
    ) -> CatalogResults:
        """Match exact labels: every facet must match, with OR within label lists.

        Discovery includes all registered datasets unless ``ready_only=True``.
        Row dictionaries retain every catalog column, including ``zarr_uri``.
        Set ``max_cell_chars=None`` to display full cell values.
        """
        predicates = []
        parameters = []
        if ready_only:
            predicates.append(_READY_PREDICATE)
        for facet, values in facets.items():
            if facet not in _FACET_COLUMNS:
                raise ValueError(
                    f"Unknown facet {facet!r}; choose from {', '.join(FACETS)}"
                )
            labels = [values] if isinstance(values, str) else values
            if not isinstance(labels, list) or not all(
                isinstance(value, str) for value in labels
            ):
                raise TypeError("Facet filters must be strings or lists of strings")
            predicates.append(f"list_has_any({_FACET_COLUMNS[facet]}, ?)")
            parameters.append(labels)
        where = " WHERE " + " AND ".join(predicates) if predicates else ""
        rows = self.query(
            "SELECT * FROM datasets" + where + _DATASET_ORDER,
            parameters,
            max_cell_chars=max_cell_chars,
        )
        return CatalogResults(
            rows,
            columns=(
                "cytebase_id",
                "status",
                "cell_count",
                "n_genes",
                "tissue_labels",
                "disease_labels",
            ),
            max_cell_chars=max_cell_chars,
        )

    def search(
        self,
        text: str,
        *,
        ready_only: bool = True,
        limit: int | None = 50,
        max_cell_chars: int | None = 100,
    ) -> CatalogResults:
        """Find datasets whose text matches every word in ``text``, ignoring case.

        Words are matched against the ID, title, citation, first author, and all
        facet labels. Use :meth:`find_datasets` for exact ontology labels.
        Set ``max_cell_chars=None`` to display full cell values.
        """
        if not isinstance(text, str) or not text.split():
            raise ValueError("Provide at least one search word")
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
        ):
            raise ValueError("limit must be a positive integer or None")
        words = text.split()
        predicates = [f"({_SEARCH_TEXT}) ILIKE ?" for _ in words]
        parameters: list[Any] = [f"%{word}%" for word in words]
        if ready_only:
            predicates.append(_READY_PREDICATE)
        sql = "SELECT * FROM datasets WHERE " + " AND ".join(predicates)
        sql += _DATASET_ORDER
        if limit is not None:
            sql += " LIMIT ?"
            parameters.append(limit)
        return CatalogResults(
            self.query(sql, parameters, max_cell_chars=max_cell_chars),
            columns=_SEARCH_COLUMNS,
            max_cell_chars=max_cell_chars,
        )

    def dataset(self, cytebase_id: str) -> "CytebaseDataset":
        """Return a handle for one catalog dataset without opening its store."""
        from .dataset import CytebaseDataset

        rows = self.query("SELECT * FROM datasets WHERE cytebase_id = ?", [cytebase_id])
        if not rows:
            raise KeyError(f"No catalog dataset is registered as {cytebase_id!r}")
        return CytebaseDataset(self, rows[0])

    def list_terms(
        self, facet: str | None = None, *, max_cell_chars: int | None = 100
    ) -> CatalogResults:
        """List naturally sorted terms; ``max_cell_chars=None`` shows full values."""
        if facet is not None and facet not in FACETS:
            raise ValueError(
                f"Unknown facet {facet!r}; choose from {', '.join(FACETS)}"
            )
        where = " WHERE facet = ? " if facet is not None else " "
        rows = self.query(
            "SELECT facet, label, term_id, count(DISTINCT cytebase_id) AS n_datasets "
            "FROM dataset_terms"
            + where
            + "GROUP BY facet, label, term_id, label_rank ORDER BY facet, label_rank, term_id",
            [facet] if facet is not None else [],
            max_cell_chars=max_cell_chars,
        )
        columns = ("label", "term_id", "n_datasets")
        return CatalogResults(
            rows,
            columns=("facet", *columns) if facet is None else columns,
            max_cell_chars=max_cell_chars,
        )

    def open_dataset(self, cytebase_id: str, **datastore_options: Any) -> "DataStore":
        """Open committed remote counts read-only after checking current provenance."""
        from .connector import open_dataset

        _cached_catalog(self._storage)
        return open_dataset(self._storage, cytebase_id, **datastore_options)

    def mount_dataset(
        self, cytebase_id: str, at: str | Path, **datastore_options: Any
    ) -> "DataStore":
        """Create or reopen a local analysis with remote counts pinned to one build."""
        from .connector import mount_dataset

        _cached_catalog(self._storage)
        return mount_dataset(self._storage, cytebase_id, at, **datastore_options)

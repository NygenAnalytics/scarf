"""Write compact gene-reference tables for tests."""

from collections.abc import Sequence
from pathlib import Path

from scarf.features.gene_reference import GeneReference, _write_reference


def write_reference_fixture(
    path: Path,
    *,
    species: str,
    release: str,
    rows: Sequence[tuple[str, str, str]],
) -> GeneReference:
    """Write a cached reference table and return the matching reference."""
    table_path = path if path.suffix == ".tsv" else path.with_suffix(".tsv")
    _write_reference(
        table_path,
        table_path.with_suffix(".release"),
        release=release,
        rows=rows,
    )
    return GeneReference(
        species=species,
        release=release,
        geneId=tuple(row[0] for row in rows),
        symbol=tuple(row[1] for row in rows),
        chromosome=tuple(row[2] for row in rows),
    )

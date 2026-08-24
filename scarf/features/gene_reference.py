"""Per-species Ensembl gene reference download and local lookup."""

import gzip
import os
import re
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

__all__ = [
    "GeneReference",
    "SpeciesSpec",
    "cached_species",
    "default_cache_dir",
    "ensure_reference",
    "load_reference",
    "parse_gff3_genes",
    "prefix_species",
    "reference_summary",
    "species_registry",
    "write_reference_fixture",
]

_ENSEMBL_GFF3 = "https://ftp.ensembl.org/pub/current_gff3/{species}/"
_ENSEMBL_GENOMES_GFF3 = (
    "https://ftp.ensemblgenomes.ebi.ac.uk/pub/{division}/current/gff3/{species}/"
)
_GENE_TYPES = frozenset({"gene", "ncRNA_gene", "pseudogene"})
_GFF_NAME = re.compile(
    r"^(?P<label>[^.]+)\.(?P<assembly>[^.]+)\.(?P<release>\d+)\.gff3\.gz$"
)


@dataclass(frozen=True, slots=True)
class SpeciesSpec:
    """One downloadable species entry in the gene-reference registry."""

    key: str
    label: str
    idPrefix: str
    source: str
    division: str | None = None


@dataclass
class GeneReference:
    """Local geneId / symbol / chromosome table for one species."""

    species: str
    release: str
    geneId: tuple[str, ...]
    symbol: tuple[str, ...]
    chromosome: tuple[str, ...]

    def __post_init__(self) -> None:
        n = len(self.geneId)
        if not (n == len(self.symbol) == len(self.chromosome)):
            raise ValueError("geneId, symbol, and chromosome must have equal length")
        self._by_id = {gene_id: index for index, gene_id in enumerate(self.geneId)}
        self._by_symbol: dict[str, list[int]] = {}
        for index, name in enumerate(self.symbol):
            if not name:
                continue
            self._by_symbol.setdefault(name, []).append(index)

    @property
    def nGenes(self) -> int:
        return len(self.geneId)

    def has_gene_id(self, gene_id: str) -> bool:
        return gene_id in self._by_id or _strip_version(gene_id) in self._by_id

    def symbol_for(self, gene_id: str) -> str | None:
        index = self._by_id.get(gene_id)
        if index is None:
            index = self._by_id.get(_strip_version(gene_id))
        if index is None:
            return None
        symbol = self.symbol[index]
        return symbol or None

    def chromosome_for(self, gene_id: str) -> str | None:
        index = self._by_id.get(gene_id)
        if index is None:
            index = self._by_id.get(_strip_version(gene_id))
        if index is None:
            return None
        return self.chromosome[index]

    def has_symbol(self, symbol: str) -> bool:
        return symbol in self._by_symbol

    def symbols(self) -> set[str]:
        return set(self._by_symbol)

    def mitochondrial_gene_ids(self) -> list[str]:
        return [
            gene_id
            for gene_id, chrom in zip(self.geneId, self.chromosome, strict=True)
            if chrom.upper() == "MT"
        ]


def species_registry() -> dict[str, SpeciesSpec]:
    """Supported species and their Ensembl resource locations."""
    main = [
        ("homo_sapiens", "human", "ENSG"),
        ("mus_musculus", "mouse", "ENSMUSG"),
        ("rattus_norvegicus", "rat", "ENSRNOG"),
        ("danio_rerio", "zebrafish", "ENSDARG"),
        ("drosophila_melanogaster", "fly", "FBgn"),
        ("caenorhabditis_elegans", "worm", "WBGene"),
    ]
    return {
        key: SpeciesSpec(key=key, label=label, idPrefix=prefix, source="ensembl")
        for key, label, prefix in main
    }


def default_cache_dir() -> Path:
    override = os.environ.get("SCARF_GENE_REFERENCE_CACHE")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    root = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return root / "scarf" / "gene_reference"


def _strip_version(gene_id: str) -> str:
    if "." in gene_id and gene_id.rsplit(".", 1)[-1].isdigit():
        return gene_id.rsplit(".", 1)[0]
    return gene_id


def _parse_attributes(raw: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for part in raw.split(";"):
        if not part or "=" not in part:
            continue
        key, value = part.split("=", 1)
        attrs[key] = value
    return attrs


def parse_gff3_genes(lines: Iterable[str]) -> list[tuple[str, str, str]]:
    """Extract ``(geneId, symbol, chromosome)`` from Ensembl GFF3 gene lines."""
    rows: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for line in lines:
        if not line or line.startswith("#"):
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 9 or parts[2] not in _GENE_TYPES:
            continue
        attrs = _parse_attributes(parts[8])
        gene_id = attrs.get("gene_id") or attrs.get("ID", "").removeprefix("gene:")
        if not gene_id or gene_id in seen:
            continue
        symbol = attrs.get("Name") or attrs.get("gene_name") or ""
        rows.append((gene_id, symbol, parts[0]))
        seen.add(gene_id)
    return rows


def _cache_paths(cache_dir: Path, species: str) -> tuple[Path, Path]:
    base = cache_dir / species
    return Path(f"{base}.tsv"), Path(f"{base}.release")


def _write_reference(
    path: Path,
    release_path: Path,
    *,
    release: str,
    rows: Sequence[tuple[str, str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write("geneId\tsymbol\tchromosome\n")
        for gene_id, symbol, chromosome in rows:
            handle.write(f"{gene_id}\t{symbol}\t{chromosome}\n")
    tmp.replace(path)
    release_path.write_text(f"{release}\n", encoding="utf-8")


def load_reference(
    species: str,
    *,
    cacheDir: Path | None = None,
) -> GeneReference | None:
    """Load a cached reference table, or return None when missing."""
    cache_dir = cacheDir or default_cache_dir()
    table_path, release_path = _cache_paths(cache_dir, species)
    if not table_path.is_file():
        return None
    release = (
        release_path.read_text(encoding="utf-8").strip()
        if release_path.is_file()
        else "unknown"
    )
    gene_ids: list[str] = []
    symbols: list[str] = []
    chromosomes: list[str] = []
    with table_path.open(encoding="utf-8") as handle:
        header = handle.readline()
        if not header.startswith("geneId"):
            raise ValueError(f"unexpected gene reference header in {table_path}")
        for line in handle:
            gene_id, symbol, chromosome = line.rstrip("\n").split("\t")
            gene_ids.append(gene_id)
            symbols.append(symbol)
            chromosomes.append(chromosome)
    return GeneReference(
        species=species,
        release=release,
        geneId=tuple(gene_ids),
        symbol=tuple(symbols),
        chromosome=tuple(chromosomes),
    )


def write_reference_fixture(
    path: Path,
    *,
    species: str,
    release: str,
    rows: Sequence[tuple[str, str, str]],
) -> GeneReference:
    """Write a compact reference table for tests or offline prep."""
    table_path = path if path.suffix == ".tsv" else path.with_suffix(".tsv")
    release_path = table_path.with_suffix(".release")
    _write_reference(table_path, release_path, release=release, rows=rows)
    return GeneReference(
        species=species,
        release=release,
        geneId=tuple(row[0] for row in rows),
        symbol=tuple(row[1] for row in rows),
        chromosome=tuple(row[2] for row in rows),
    )


def _directory_listing(url: str, *, timeout: float) -> str:
    request = urllib.request.Request(url, headers={"Accept": "text/html,text/plain"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    return bytes(payload).decode("utf-8", errors="replace")


def _pick_gff3_name(listing: str) -> tuple[str, str]:
    candidates: list[tuple[str, str]] = []
    for match in re.finditer(r'href="([^"]+\.gff3\.gz)"', listing, flags=re.IGNORECASE):
        name = match.group(1).rsplit("/", 1)[-1]
        if "chromosome" in name.lower() or "abinitio" in name.lower():
            continue
        parsed = _GFF_NAME.match(name)
        if parsed is None:
            continue
        candidates.append((name, parsed.group("release")))
    if not candidates:
        for match in re.finditer(r"([A-Za-z0-9._-]+\.gff3\.gz)", listing):
            name = match.group(1)
            if "chromosome" in name.lower() or "abinitio" in name.lower():
                continue
            parsed = _GFF_NAME.match(name)
            if parsed is None:
                continue
            candidates.append((name, parsed.group("release")))
    if not candidates:
        raise FileNotFoundError("no top-level Ensembl GFF3 file found in listing")
    candidates.sort(key=lambda item: int(item[1]), reverse=True)
    return candidates[0]


def _species_url(spec: SpeciesSpec) -> str:
    if spec.source == "ensembl":
        return _ENSEMBL_GFF3.format(species=spec.key)
    if spec.source == "ensemblgenomes":
        if not spec.division:
            raise ValueError(f"ensemblgenomes species {spec.key} needs a division")
        return _ENSEMBL_GENOMES_GFF3.format(division=spec.division, species=spec.key)
    raise ValueError(f"unsupported gene-reference source {spec.source!r}")


def _download_gff3(url: str, destination: Path, *, timeout: float) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    request = urllib.request.Request(url)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        with tmp.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    tmp.replace(destination)


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz" or path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open(encoding="utf-8", errors="replace")


def ensure_reference(
    species: str,
    *,
    cacheDir: Path | None = None,
    timeout: float = 60.0,
    force: bool = False,
) -> GeneReference:
    """Return a cached reference, downloading from Ensembl when needed."""
    registry = species_registry()
    if species not in registry:
        raise KeyError(f"species {species!r} is not in the gene-reference registry")
    cache_dir = cacheDir or default_cache_dir()
    if not force:
        cached = load_reference(species, cacheDir=cache_dir)
        if cached is not None:
            return cached

    spec = registry[species]
    base_url = _species_url(spec)
    listing = _directory_listing(base_url, timeout=timeout)
    filename, release = _pick_gff3_name(listing)
    gff_url = f"{base_url.rstrip('/')}/{filename}"
    raw_path = cache_dir / species / filename
    _download_gff3(gff_url, raw_path, timeout=timeout)
    with _open_text(raw_path) as handle:
        rows = parse_gff3_genes(handle)
    if not rows:
        raise ValueError(f"no gene lines parsed from {gff_url}")
    table_path, release_path = _cache_paths(cache_dir, species)
    _write_reference(table_path, release_path, release=release, rows=rows)
    return GeneReference(
        species=species,
        release=release,
        geneId=tuple(row[0] for row in rows),
        symbol=tuple(row[1] for row in rows),
        chromosome=tuple(row[2] for row in rows),
    )


def cached_species(cacheDir: Path | None = None) -> list[str]:
    """Species keys that already have a local reference table."""
    cache_dir = cacheDir or default_cache_dir()
    if not cache_dir.is_dir():
        return []
    known = set(species_registry())
    return sorted(path.stem for path in cache_dir.glob("*.tsv") if path.stem in known)


def prefix_species(gene_ids: Sequence[str]) -> dict[str, int]:
    """Count Ensembl-style ID prefixes against the registry."""
    counts = {spec.key: 0 for spec in species_registry().values()}
    for raw in gene_ids:
        gene_id = _strip_version(str(raw))
        for spec in species_registry().values():
            if gene_id.startswith(spec.idPrefix):
                counts[spec.key] += 1
                break
    return {key: count for key, count in counts.items() if count}


def reference_summary(reference: GeneReference) -> dict[str, Any]:
    return {
        "species": reference.species,
        "release": reference.release,
        "nGenes": reference.nGenes,
        "nSymbols": sum(1 for symbol in reference.symbol if symbol),
        "nMitochondrial": len(reference.mitochondrial_gene_ids()),
    }

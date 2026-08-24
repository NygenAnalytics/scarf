"""Tests for Ensembl gene-reference helpers."""

import gzip
from pathlib import Path

import pytest

from scarf.features import gene_reference as gene_reference_module
from scarf.features.gene_reference import (
    GeneReference,
    SpeciesSpec,
    cached_species,
    default_cache_dir,
    ensure_reference,
    load_reference,
    parse_gff3_genes,
    prefix_species,
    reference_summary,
    write_reference_fixture,
)


_GFF_SNIPPET = """\
##gff-version 3
#!genome-build GRCh38.p14
1	ensembl	gene	1	100	.	+	.	ID=gene:ENSG00000000001;Name=GENEA;gene_id=ENSG00000000001
1	ensembl	exon	1	50	.	+	.	Parent=transcript:ENST1
MT	insdc	ncRNA_gene	1	70	.	+	.	ID=gene:ENSG00000210156;Name=MT-TK;gene_id=ENSG00000210156
MT	insdc	gene	80	200	.	+	.	ID=gene:ENSG00000198727;Name=MT-CYB;gene_id=ENSG00000198727
2	ensembl	pseudogene	1	20	.	+	.	ID=gene:ENSG00000000002;Name=PSEUD;gene_id=ENSG00000000002
"""


class _BytesResponse:
    def __init__(self, *chunks: bytes) -> None:
        self._chunks = list(chunks)

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            payload = b"".join(self._chunks)
            self._chunks.clear()
            return payload
        if not self._chunks:
            return b""
        return self._chunks.pop(0)

    def __enter__(self) -> "_BytesResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def _gff_line(chromosome: str, feature: str, attributes: str) -> str:
    return f"{chromosome}\tensembl\t{feature}\t1\t100\t.\t+\t.\t{attributes}"


def test_gene_reference_rejects_unequal_columns() -> None:
    with pytest.raises(
        ValueError, match="geneId, symbol, and chromosome must have equal length"
    ):
        GeneReference(
            species="homo_sapiens",
            release="113",
            geneId=("ENSG00000000001",),
            symbol=(),
            chromosome=("1",),
        )


def test_gene_reference_lookups_handle_versions_blanks_and_mt() -> None:
    reference = GeneReference(
        species="homo_sapiens",
        release="113",
        geneId=(
            "ENSG00000000001",
            "ENSG00000000002",
            "ENSG00000000003",
        ),
        symbol=("GENEA", "GENEA", ""),
        chromosome=("1", "mt", "X"),
    )

    assert reference.nGenes == 3
    assert reference.has_gene_id("ENSG00000000001")
    assert reference.has_gene_id("ENSG00000000001.12")
    assert not reference.has_gene_id("ENSG00000000001.alpha")
    assert reference.symbol_for("ENSG00000000001.12") == "GENEA"
    assert reference.symbol_for("ENSG00000000003") is None
    assert reference.symbol_for("ENSG00000000999") is None
    assert reference.chromosome_for("ENSG00000000002.3") == "mt"
    assert reference.chromosome_for("ENSG00000000999") is None
    assert reference.has_symbol("GENEA")
    assert not reference.has_symbol("")
    assert reference.symbols() == {"GENEA"}
    assert reference.mitochondrial_gene_ids() == ["ENSG00000000002"]


def test_parse_gff3_genes_reads_name_and_mt() -> None:
    rows = parse_gff3_genes(_GFF_SNIPPET.splitlines())
    by_id = {gene_id: (symbol, chrom) for gene_id, symbol, chrom in rows}
    assert by_id["ENSG00000000001"] == ("GENEA", "1")
    assert by_id["ENSG00000210156"] == ("MT-TK", "MT")
    assert by_id["ENSG00000198727"] == ("MT-CYB", "MT")
    assert by_id["ENSG00000000002"] == ("PSEUD", "2")


def test_parse_gff3_genes_skips_malformed_and_duplicate_rows() -> None:
    rows = parse_gff3_genes(
        [
            "",
            "# a comment",
            "not\tenough\tcolumns",
            _gff_line("1", "exon", "ID=exon:ENSE000001"),
            _gff_line("1", "gene", "Name=NO_ID;malformed"),
            _gff_line(
                "1",
                "gene",
                "ID=gene:ENSG00000000010;gene_name=FIRST",
            ),
            _gff_line(
                "2",
                "gene",
                "gene_id=ENSG00000000010;Name=DUPLICATE",
            ),
            _gff_line(
                "X",
                "pseudogene",
                "malformed;gene_id=ENSG00000000011;gene_name=PSEUD",
            ),
            _gff_line(
                "MT",
                "ncRNA_gene",
                "ID=gene:ENSG00000000012;Name=MT=GENE",
            ),
        ]
    )

    assert rows == [
        ("ENSG00000000010", "FIRST", "1"),
        ("ENSG00000000011", "PSEUD", "X"),
        ("ENSG00000000012", "MT=GENE", "MT"),
    ]


def test_write_and_load_reference_roundtrip(tmp_path: Path) -> None:
    rows = parse_gff3_genes(_GFF_SNIPPET.splitlines())
    written = write_reference_fixture(
        tmp_path / "homo_sapiens",
        species="homo_sapiens",
        release="113",
        rows=rows,
    )
    loaded = load_reference("homo_sapiens", cacheDir=tmp_path)

    assert loaded == written
    assert loaded is not None
    assert loaded.release == "113"
    assert loaded.has_gene_id("ENSG00000198727.1")
    assert loaded.symbol_for("ENSG00000198727") == "MT-CYB"
    assert loaded.chromosome_for("ENSG00000198727") == "MT"
    assert set(loaded.mitochondrial_gene_ids()) == {
        "ENSG00000210156",
        "ENSG00000198727",
    }


def test_load_reference_returns_none_when_table_is_missing(tmp_path: Path) -> None:
    (tmp_path / "homo_sapiens.release").write_text("113\n", encoding="utf-8")

    assert load_reference("homo_sapiens", cacheDir=tmp_path) is None


def test_load_reference_uses_unknown_without_release_sidecar(
    tmp_path: Path,
) -> None:
    (tmp_path / "homo_sapiens.tsv").write_text(
        "geneId\tsymbol\tchromosome\nENSG00000000001\tGENEA\t1\n",
        encoding="utf-8",
    )

    reference = load_reference("homo_sapiens", cacheDir=tmp_path)

    assert reference is not None
    assert reference.release == "unknown"
    assert reference.geneId == ("ENSG00000000001",)


def test_load_reference_rejects_unexpected_header(tmp_path: Path) -> None:
    (tmp_path / "homo_sapiens.tsv").write_text(
        "id\tsymbol\tchromosome\nENSG00000000001\tGENEA\t1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unexpected gene reference header"):
        load_reference("homo_sapiens", cacheDir=tmp_path)


@pytest.mark.parametrize(
    "row",
    [
        "ENSG00000000001\tGENEA\n",
        "ENSG00000000001\tGENEA\t1\textra\n",
    ],
    ids=["missing-column", "extra-column"],
)
def test_load_reference_rejects_malformed_rows(tmp_path: Path, row: str) -> None:
    (tmp_path / "homo_sapiens.tsv").write_text(
        f"geneId\tsymbol\tchromosome\n{row}",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_reference("homo_sapiens", cacheDir=tmp_path)


def test_prefix_species_strips_only_numeric_versions() -> None:
    counts = prefix_species(
        [
            "ENSG000001.1",
            "ENSG000002",
            "ENSMUSG000001",
            "FBgn0000001.alpha",
            "not-an-id",
        ]
    )
    assert counts == {
        "homo_sapiens": 2,
        "mus_musculus": 1,
        "drosophila_melanogaster": 1,
    }


def test_default_cache_dir_respects_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SCARF_GENE_REFERENCE_CACHE", str(tmp_path / "custom"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))

    assert default_cache_dir() == tmp_path / "custom"


def test_default_cache_dir_respects_xdg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("SCARF_GENE_REFERENCE_CACHE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CACHE_HOME", "~/.xdg-cache")

    assert default_cache_dir() == (
        tmp_path / "home" / ".xdg-cache" / "scarf" / "gene_reference"
    )


def test_default_cache_dir_uses_home_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("SCARF_GENE_REFERENCE_CACHE", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    assert default_cache_dir() == (
        tmp_path / "home" / ".cache" / "scarf" / "gene_reference"
    )


def test_pick_gff3_name_prefers_newest_top_level_release() -> None:
    listing = """
    <a href="Homo_sapiens.GRCh38.99.gff3.gz">old</a>
    <a href="/pub/Homo_sapiens.GRCh38.113.gff3.gz">current</a>
    <a href="Homo_sapiens.GRCh38.114.chromosome.1.gff3.gz">chromosome</a>
    <a href="Homo_sapiens.GRCh38.115.abinitio.gff3.gz">ab initio</a>
    <a href="README.gff3.gz">invalid</a>
    """

    assert gene_reference_module._pick_gff3_name(listing) == (
        "Homo_sapiens.GRCh38.113.gff3.gz",
        "113",
    )


def test_pick_gff3_name_supports_plain_text_listing() -> None:
    listing = (
        "README.gff3.gz\n"
        "Mus_musculus.GRCm39.111.chromosome.1.gff3.gz\n"
        "Mus_musculus.GRCm39.112.gff3.gz\n"
    )

    assert gene_reference_module._pick_gff3_name(listing) == (
        "Mus_musculus.GRCm39.112.gff3.gz",
        "112",
    )


def test_pick_gff3_name_rejects_listing_without_top_level_file() -> None:
    listing = '<a href="Homo_sapiens.GRCh38.113.chromosome.1.gff3.gz">chromosome</a>'

    with pytest.raises(FileNotFoundError, match="no top-level Ensembl GFF3 file"):
        gene_reference_module._pick_gff3_name(listing)


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (
            SpeciesSpec(
                key="homo_sapiens",
                label="human",
                idPrefix="ENSG",
                source="ensembl",
            ),
            "https://ftp.ensembl.org/pub/current_gff3/homo_sapiens/",
        ),
        (
            SpeciesSpec(
                key="arabidopsis_thaliana",
                label="arabidopsis",
                idPrefix="AT",
                source="ensemblgenomes",
                division="plants",
            ),
            (
                "https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/current/"
                "gff3/arabidopsis_thaliana/"
            ),
        ),
    ],
    ids=["ensembl", "ensembl-genomes"],
)
def test_species_url_selects_source(spec: SpeciesSpec, expected: str) -> None:
    assert gene_reference_module._species_url(spec) == expected


def test_species_url_requires_ensembl_genomes_division() -> None:
    spec = SpeciesSpec(
        key="arabidopsis_thaliana",
        label="arabidopsis",
        idPrefix="AT",
        source="ensemblgenomes",
    )

    with pytest.raises(ValueError, match="needs a division"):
        gene_reference_module._species_url(spec)


def test_species_url_rejects_unknown_source() -> None:
    spec = SpeciesSpec(
        key="homo_sapiens",
        label="human",
        idPrefix="ENSG",
        source="other",
    )

    with pytest.raises(ValueError, match="unsupported gene-reference source 'other'"):
        gene_reference_module._species_url(spec)


def test_directory_listing_uses_request_headers_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_urlopen(request: object, timeout: float = 0.0) -> _BytesResponse:
        assert isinstance(request, gene_reference_module.urllib.request.Request)
        observed["url"] = request.full_url
        observed["accept"] = request.get_header("Accept")
        observed["timeout"] = timeout
        return _BytesResponse(b"listing \xff")

    monkeypatch.setattr(gene_reference_module.urllib.request, "urlopen", fake_urlopen)

    listing = gene_reference_module._directory_listing(
        "https://example.test/gff3/", timeout=7.5
    )

    assert listing == "listing \ufffd"
    assert observed == {
        "url": "https://example.test/gff3/",
        "accept": "text/html,text/plain",
        "timeout": 7.5,
    }


def test_download_gff3_streams_to_destination(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: dict[str, object] = {}

    def fake_urlopen(request: object, timeout: float = 0.0) -> _BytesResponse:
        assert isinstance(request, gene_reference_module.urllib.request.Request)
        observed["url"] = request.full_url
        observed["timeout"] = timeout
        return _BytesResponse(b"first-", b"second")

    monkeypatch.setattr(gene_reference_module.urllib.request, "urlopen", fake_urlopen)
    destination = tmp_path / "nested" / "genes.gff3.gz"

    gene_reference_module._download_gff3(
        "https://example.test/genes.gff3.gz",
        destination,
        timeout=3.25,
    )

    assert destination.read_bytes() == b"first-second"
    assert not destination.with_suffix(".gz.tmp").exists()
    assert observed == {
        "url": "https://example.test/genes.gff3.gz",
        "timeout": 3.25,
    }


@pytest.mark.parametrize("compressed", [False, True], ids=["plain", "gzip"])
def test_open_text_reads_plain_and_gzip(tmp_path: Path, compressed: bool) -> None:
    path = tmp_path / ("genes.gff3.gz" if compressed else "genes.gff3")
    if compressed:
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(_GFF_SNIPPET)
    else:
        path.write_text(_GFF_SNIPPET, encoding="utf-8")

    with gene_reference_module._open_text(path) as handle:
        assert handle.read() == _GFF_SNIPPET


def test_ensure_reference_rejects_unknown_species(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="'unknown_species'"):
        ensure_reference("unknown_species", cacheDir=tmp_path)


def test_ensure_reference_returns_cache_without_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected = write_reference_fixture(
        tmp_path / "homo_sapiens.tsv",
        species="homo_sapiens",
        release="112",
        rows=[("ENSG00000000001", "GENEA", "1")],
    )

    def fail_listing(_url: str, *, timeout: float) -> str:
        pytest.fail(f"network listing requested with timeout {timeout}")

    monkeypatch.setattr(gene_reference_module, "_directory_listing", fail_listing)

    assert ensure_reference("homo_sapiens", cacheDir=tmp_path) == expected


@pytest.mark.parametrize(
    ("preexisting", "force"),
    [(False, False), (True, True)],
    ids=["cache-miss", "forced-refresh"],
)
def test_ensure_reference_downloads_and_caches_reference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    preexisting: bool,
    force: bool,
) -> None:
    filename = "Homo_sapiens.GRCh38.113.gff3.gz"
    base_url = "https://ftp.ensembl.org/pub/current_gff3/homo_sapiens/"
    observed: dict[str, object] = {}
    if preexisting:
        write_reference_fixture(
            tmp_path / "homo_sapiens.tsv",
            species="homo_sapiens",
            release="112",
            rows=[("ENSG_STALE", "STALE", "1")],
        )

    def fake_listing(url: str, *, timeout: float) -> str:
        observed["listing"] = (url, timeout)
        return f'<a href="{filename}">{filename}</a>'

    def fake_download(url: str, destination: Path, *, timeout: float) -> None:
        observed["download"] = (url, destination, timeout)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(destination, "wt", encoding="utf-8") as handle:
            handle.write(_GFF_SNIPPET)

    monkeypatch.setattr(gene_reference_module, "_directory_listing", fake_listing)
    monkeypatch.setattr(gene_reference_module, "_download_gff3", fake_download)

    reference = ensure_reference(
        "homo_sapiens",
        cacheDir=tmp_path,
        timeout=12.5,
        force=force,
    )

    raw_path = tmp_path / "homo_sapiens" / filename
    assert observed == {
        "listing": (base_url, 12.5),
        "download": (
            f"{base_url}{filename}",
            raw_path,
            12.5,
        ),
    }
    assert reference.release == "113"
    assert reference.nGenes == 4
    assert reference.geneId[0] == "ENSG00000000001"
    assert "ENSG_STALE" not in reference.geneId
    assert raw_path.is_file()
    assert (tmp_path / "homo_sapiens.release").read_text(encoding="utf-8") == "113\n"
    assert load_reference("homo_sapiens", cacheDir=tmp_path) == reference


def test_ensure_reference_rejects_download_without_gene_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    filename = "Homo_sapiens.GRCh38.113.gff3.gz"

    def fake_listing(_url: str, *, timeout: float) -> str:
        assert timeout == 4.0
        return f'<a href="{filename}">{filename}</a>'

    def fake_download(_url: str, destination: Path, *, timeout: float) -> None:
        assert timeout == 4.0
        destination.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(destination, "wt", encoding="utf-8") as handle:
            handle.write(
                "##gff-version 3\n"
                + _gff_line("1", "exon", "ID=exon:ENSE000001")
                + "\n"
            )

    monkeypatch.setattr(gene_reference_module, "_directory_listing", fake_listing)
    monkeypatch.setattr(gene_reference_module, "_download_gff3", fake_download)

    with pytest.raises(ValueError, match="no gene lines parsed"):
        ensure_reference("homo_sapiens", cacheDir=tmp_path, timeout=4.0)

    assert not (tmp_path / "homo_sapiens.tsv").exists()


def test_cached_species_lists_only_known_tables(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    assert cached_species(cache_dir) == []

    cache_dir.mkdir()
    for filename in [
        "mus_musculus.tsv",
        "homo_sapiens.tsv",
        "unsupported_species.tsv",
    ]:
        (cache_dir / filename).write_text("", encoding="utf-8")
    (cache_dir / "rattus_norvegicus.release").write_text("113\n", encoding="utf-8")

    assert cached_species(cache_dir) == ["homo_sapiens", "mus_musculus"]


def test_reference_summary_counts_symbols_and_mitochondrial_genes() -> None:
    reference = GeneReference(
        species="homo_sapiens",
        release="113",
        geneId=("ENSG1", "ENSG2", "ENSG3", "ENSG4"),
        symbol=("GENEA", "", "MT-A", "GENEA"),
        chromosome=("1", "X", "MT", "mt"),
    )

    assert reference_summary(reference) == {
        "species": "homo_sapiens",
        "release": "113",
        "nGenes": 4,
        "nSymbols": 3,
        "nMitochondrial": 2,
    }

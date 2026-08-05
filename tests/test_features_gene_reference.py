"""Tests for Ensembl gene-reference helpers."""

from pathlib import Path

from scarf.features.gene_reference import (
    default_cache_dir,
    load_reference,
    parse_gff3_genes,
    prefix_species,
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


def test_parse_gff3_genes_reads_name_and_mt() -> None:
    rows = parse_gff3_genes(_GFF_SNIPPET.splitlines())
    by_id = {gene_id: (symbol, chrom) for gene_id, symbol, chrom in rows}
    assert by_id["ENSG00000000001"] == ("GENEA", "1")
    assert by_id["ENSG00000210156"] == ("MT-TK", "MT")
    assert by_id["ENSG00000198727"] == ("MT-CYB", "MT")
    assert "ENSG00000000002" in by_id


def test_write_and_load_reference_roundtrip(tmp_path: Path) -> None:
    rows = parse_gff3_genes(_GFF_SNIPPET.splitlines())
    write_reference_fixture(
        tmp_path / "homo_sapiens.tsv",
        species="homo_sapiens",
        release="113",
        rows=rows,
    )
    loaded = load_reference("homo_sapiens", cacheDir=tmp_path)
    assert loaded is not None
    assert loaded.release == "113"
    assert loaded.has_gene_id("ENSG00000198727.1")
    assert loaded.symbol_for("ENSG00000198727") == "MT-CYB"
    assert loaded.chromosome_for("ENSG00000198727") == "MT"
    assert set(loaded.mitochondrial_gene_ids()) == {
        "ENSG00000210156",
        "ENSG00000198727",
    }


def test_prefix_species_strips_version() -> None:
    counts = prefix_species(
        ["ENSG000001.1", "ENSG000002", "ENSMUSG000001", "not-an-id"]
    )
    assert counts == {"homo_sapiens": 2, "mus_musculus": 1}


def test_default_cache_dir_respects_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SCARF_GENE_REFERENCE_CACHE", str(tmp_path / "custom"))
    assert default_cache_dir() == tmp_path / "custom"

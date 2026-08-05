"""Tests for feature identity and species-aware family helpers."""

from pathlib import Path

from scarf.features.gene_reference import write_reference_fixture
from scarf.features.identity import (
    audit_feature_identity,
    backfill_symbols,
    exogenous_candidates,
    observe_families,
    reference_misses,
    resolve_species,
    score_symbol_overlap,
)
from scarf.quality_control.cell_cycle_genes import (
    g2m_phase_genes,
    g2m_phase_genes_mouse,
    s_phase_genes,
    s_phase_genes_mouse,
)


def _human_reference(tmp_path: Path):
    return write_reference_fixture(
        tmp_path / "homo_sapiens.tsv",
        species="homo_sapiens",
        release="test",
        rows=[
            ("ENSG00000075624", "ACTB", "7"),
            ("ENSG00000111640", "GAPDH", "12"),
            ("ENSG00000198727", "MT-CYB", "MT"),
            ("ENSG00000198712", "MT-CO2", "MT"),
            ("ENSG00000071082", "RPL31", "2"),
            ("ENSG00000198034", "RPS4X", "X"),
            ("ENSG00000229807", "XIST", "X"),
            ("ENSG00000129824", "RPS4Y1", "Y"),
            ("ENSG00000184640", "HIST1H4A", "6"),
            ("ENSG00000012048", "BRCA1", "17"),
        ],
    )


def _mouse_reference(tmp_path: Path):
    return write_reference_fixture(
        tmp_path / "mus_musculus.tsv",
        species="mus_musculus",
        release="test",
        rows=[
            ("ENSMUSG00000029580", "Actb", "5"),
            ("ENSMUSG00000057654", "Gapdh", "6"),
            ("ENSMUSG00000064345", "mt-Cytb", "MT"),
            ("ENSMUSG00000064354", "mt-Co2", "MT"),
            ("ENSMUSG00000044533", "Rpl31", "1"),
            ("ENSMUSG00000031329", "Xist", "X"),
        ],
    )


def test_audit_feature_identity_flags_duplicates_and_versions() -> None:
    report = audit_feature_identity(
        ["ENSG1.1", "ENSG1.1", "", "ENSG2"],
        ["A", "A", "B", "ENSG2"],
    )
    assert report["nFeatures"] == 4
    assert report["nEmptyIds"] == 1
    assert report["nDuplicateIds"] == 1
    assert report["nDuplicateNames"] == 1
    assert report["nVersionSuffixes"] == 2
    assert report["prefixCounts"]["homo_sapiens"] == 3


def test_resolve_species_prefix_path_offline() -> None:
    result = resolve_species(
        ["ENSG0001", "ENSG0002", "ENSG0003", "ENSG0004"],
        ["A", "B", "C", "D"],
        allowDownload=False,
    )
    assert result["species"] == "homo_sapiens"
    assert result["method"] == "ensemblPrefix"


def test_resolve_species_symbol_overlap_case_sensitive(tmp_path: Path) -> None:
    human = _human_reference(tmp_path)
    mouse = _mouse_reference(tmp_path)
    # Human uppercase symbols beat mouse title-case on the same genes.
    overlap = score_symbol_overlap(
        ["ACTB", "GAPDH", "BRCA1", "RPL31", "XIST"],
        {"homo_sapiens": human, "mus_musculus": mouse},
    )
    assert overlap["best"] == "homo_sapiens"
    assert overlap["clear"] is True
    mouse_overlap = score_symbol_overlap(
        ["Actb", "Gapdh", "Rpl31", "Xist", "mt-Cytb"],
        {"homo_sapiens": human, "mus_musculus": mouse},
    )
    assert mouse_overlap["best"] == "mus_musculus"
    assert mouse_overlap["clear"] is True


def test_resolve_species_directions_and_unknown(tmp_path: Path) -> None:
    directed = resolve_species(
        ["g1", "g2"],
        ["foo", "bar"],
        directed="mus_musculus",
    )
    assert directed == {
        "species": "mus_musculus",
        "method": "directions",
        "reason": "caller override",
    }
    unknown = resolve_species(
        ["g1", "g2"],
        ["foo", "bar"],
        cacheDir=tmp_path,
        allowDownload=False,
    )
    assert unknown["species"] == "unknown"
    assert unknown["method"] == "inconclusive"


def test_backfill_symbols_from_reference(tmp_path: Path) -> None:
    reference = _human_reference(tmp_path)
    ids = ["ENSG00000075624", "ENSG00000198727", "ERCC-00001"]
    names = list(ids)
    result = backfill_symbols(ids, names, reference)
    assert result["symbols"] == ["ACTB", "MT-CYB", "ERCC-00001"]
    assert result["nRecovered"] == 2
    assert result["joinRate"] == 2 / 3


def test_observe_families_uses_chromosome_for_mito_and_sex(tmp_path: Path) -> None:
    reference = _human_reference(tmp_path)
    ids = [
        "ENSG00000198727",
        "ENSG00000071082",
        "ENSG00000229807",
        "ENSG00000129824",
        "ENSG00000184640",
        "ENSG00000075624",
    ]
    symbols = ["MT-CYB", "RPL31", "XIST", "RPS4Y1", "HIST1H4A", "ACTB"]
    families = {
        item["family"]: item
        for item in observe_families(
            species="homo_sapiens",
            ids=ids,
            symbols=symbols,
            reference=reference,
            cellCycleGenes={
                "homo_sapiens": {"s": s_phase_genes, "g2m": g2m_phase_genes},
            },
        )
    }
    assert families["mitochondrial"]["method"] == "chromosome"
    assert families["mitochondrial"]["examples"] == ["MT-CYB"]
    assert families["mitochondrial"]["defaultExclude"] is True
    assert "RPL31" in families["ribosomal"]["examples"]
    assert families["sex"]["method"] == "chromosome"
    assert set(families["sex"]["examples"]) == {"RPS4Y1", "XIST"}
    assert families["sex"]["defaultExclude"] is False
    assert families["histone"]["examples"] == ["HIST1H4A"]
    assert "catalogSuspect" not in families["histone"]
    assert families["cellCycle"]["defaultExclude"] is False
    assert "catalogJoinRate" in families["cellCycle"]
    assert families["cellCycle"]["catalogSize"] == len(
        set(s_phase_genes) | set(g2m_phase_genes)
    )


def test_backfill_symbols_recovers_blank_names(tmp_path: Path) -> None:
    reference = _human_reference(tmp_path)
    result = backfill_symbols(
        ["ENSG00000075624", "ENSG00000198727", "ERCC-00001"],
        ["", "ENSG00000198727", "ERCC-00001"],
        reference,
    )
    assert result["symbols"] == ["ACTB", "MT-CYB", "ERCC-00001"]
    assert result["nRecovered"] == 2


def test_exogenous_candidates_skip_ensembl_reference_misses(tmp_path: Path) -> None:
    reference = _human_reference(tmp_path)
    ranked = exogenous_candidates(
        ["ENSG00000075624", "ENSG00000999999", "ERCC-00002", "GFP"],
        ["ACTB", "ENSG00000999999", "ERCC-00002", "GFP"],
        reference=reference,
        maxCandidates=10,
    )
    ids = {item["id"] for item in ranked}
    assert "ENSG00000075624" not in ids
    assert "ENSG00000999999" not in ids
    assert "ERCC-00002" in ids
    assert "GFP" in ids
    misses = reference_misses(
        ["ENSG00000075624", "ENSG00000999999", "ERCC-00002"],
        ["ACTB", "ENSG00000999999", "ERCC-00002"],
        reference,
    )
    assert misses["count"] == 1
    assert misses["examples"] == ["ENSG00000999999"]


def test_observe_families_flags_zero_prefix_matches_as_catalog_suspect(
    tmp_path: Path,
) -> None:
    """Empty prefix hits against a real symbol set are a catalog problem, not biology."""
    reference = write_reference_fixture(
        tmp_path / "homo_sapiens.tsv",
        species="homo_sapiens",
        release="test",
        rows=[
            ("ENSG00000075624", "ACTB", "7"),
            ("ENSG00000275713", "H4C1", "6"),
            ("ENSG00000168298", "H3-3A", "1"),
        ],
    )
    families = {
        item["family"]: item
        for item in observe_families(
            species="homo_sapiens",
            ids=["ENSG00000075624", "ENSG00000275713", "ENSG00000168298"],
            symbols=["ACTB", "H4C1", "H3-3A"],
            reference=reference,
        )
    }
    assert families["histone"]["count"] == 0
    assert families["histone"]["catalogSuspect"] == "zeroMatches"


def test_exogenous_candidates_skip_reference_hits(tmp_path: Path) -> None:
    reference = _human_reference(tmp_path)
    ranked = exogenous_candidates(
        ["ENSG00000075624", "ERCC-00002", "GFP", "custom_guide"],
        ["ACTB", "ERCC-00002", "GFP", "custom_guide"],
        reference=reference,
        maxCandidates=10,
    )
    names = [item["name"] for item in ranked]
    assert "ACTB" not in names
    assert names[0] in {"ERCC-00002", "GFP"}


def test_mouse_cell_cycle_lists_are_title_case() -> None:
    assert "Mcm5" in s_phase_genes_mouse
    assert "Top2a" in g2m_phase_genes_mouse
    assert len(s_phase_genes_mouse) == len(s_phase_genes)
    assert len(g2m_phase_genes_mouse) == len(g2m_phase_genes)

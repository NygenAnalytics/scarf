from scarf.quality_control.cell_cycle_genes import (
    _mouse_symbol,
    g2m_phase_genes,
    g2m_phase_genes_mouse,
    s_phase_genes,
    s_phase_genes_mouse,
)


def test_cell_cycle_gene_lists_are_nonempty_and_unique():
    # Seurat's cc.genes.updated.2019 lists 43 S and 54 G2M genes.
    assert len(s_phase_genes) == 43
    assert len(g2m_phase_genes) == 54
    assert len(s_phase_genes) == len(set(s_phase_genes))
    assert len(g2m_phase_genes) == len(set(g2m_phase_genes))
    assert set(s_phase_genes).isdisjoint(g2m_phase_genes)
    assert (s_phase_genes[0], s_phase_genes[-1]) == ("MCM5", "E2F8")
    assert (g2m_phase_genes[0], g2m_phase_genes[-1]) == ("HMGB2", "CENPA")


def test_cell_cycle_gene_lists_use_current_symbols():
    # MLF1IP, FAM64A, and HN1 were renamed; old symbols miss current annotations.
    assert "CENPU" in s_phase_genes
    assert {"PIMREG", "JPT1"} <= set(g2m_phase_genes)
    assert not {"MLF1IP", "FAM64A", "HN1"} & {*s_phase_genes, *g2m_phase_genes}
    assert "Cenpu" in s_phase_genes_mouse
    assert {"Pimreg", "Jpt1"} <= set(g2m_phase_genes_mouse)


def test_mouse_lists_hold_the_title_case_ortholog_of_each_human_gene():
    assert len(s_phase_genes_mouse) == len(s_phase_genes)
    assert len(g2m_phase_genes_mouse) == len(g2m_phase_genes)
    # MGI symbols keep the human order and their digits and letters.
    assert s_phase_genes_mouse[:3] == ["Mcm5", "Pcna", "Tyms"]
    assert {"Rad51ap1", "Casp8ap2", "E2f8", "Chaf1b"} <= set(s_phase_genes_mouse)
    assert {"Cdk1", "Top2a", "Mki67", "Cks1b", "Tubb4b"} <= set(g2m_phase_genes_mouse)
    for human, mouse in zip(
        [*s_phase_genes, *g2m_phase_genes],
        [*s_phase_genes_mouse, *g2m_phase_genes_mouse],
        strict=True,
    ):
        assert mouse.upper() == human
        assert mouse[0] == human[0]
        assert mouse[1:] == mouse[1:].lower()
    assert _mouse_symbol("") == ""
    assert _mouse_symbol("X") == "X"

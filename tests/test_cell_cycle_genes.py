from scarf.quality_control.cell_cycle_genes import (
    g2m_phase_genes,
    g2m_phase_genes_mouse,
    s_phase_genes,
    s_phase_genes_mouse,
)


def test_cell_cycle_gene_lists_are_nonempty_and_unique():
    assert len(s_phase_genes) > 20
    assert len(g2m_phase_genes) > 20
    assert len(s_phase_genes) == len(set(s_phase_genes))
    assert len(g2m_phase_genes) == len(set(g2m_phase_genes))
    assert set(s_phase_genes).isdisjoint(g2m_phase_genes)


def test_cell_cycle_gene_lists_use_current_symbols():
    # MLF1IP, FAM64A, and HN1 were renamed; old symbols miss current annotations.
    assert "CENPU" in s_phase_genes
    assert {"PIMREG", "JPT1"} <= set(g2m_phase_genes)
    assert not {"MLF1IP", "FAM64A", "HN1"} & {*s_phase_genes, *g2m_phase_genes}
    assert "Cenpu" in s_phase_genes_mouse
    assert {"Pimreg", "Jpt1"} <= set(g2m_phase_genes_mouse)

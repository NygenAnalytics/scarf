from scarf.quality_control.cell_cycle_genes import g2m_phase_genes, s_phase_genes


def test_cell_cycle_gene_lists_are_nonempty_and_unique():
    assert len(s_phase_genes) > 20
    assert len(g2m_phase_genes) > 20
    assert len(s_phase_genes) == len(set(s_phase_genes))
    assert len(g2m_phase_genes) == len(set(g2m_phase_genes))
    assert set(s_phase_genes).isdisjoint(g2m_phase_genes)

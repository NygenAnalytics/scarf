import numpy as np

__all__ = ["assign_cell_cycle_phase"]


def assign_cell_cycle_phase(
    s_score: np.ndarray,
    g2m_score: np.ndarray,
) -> np.ndarray:
    phase = np.full(np.asarray(s_score).shape, "S", dtype=object)
    phase[np.asarray(g2m_score) > np.asarray(s_score)] = "G2M"
    phase[(np.asarray(g2m_score) < 0) & (np.asarray(s_score) < 0)] = "G1"
    return phase

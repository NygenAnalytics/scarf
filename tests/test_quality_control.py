import numpy as np

from scarf.quality_control.cell_cycle import assign_cell_cycle_phase
from scarf.quality_control.doublets import simulate_doublet_pairs
from scarf.quality_control.filtering import gaussian_quantile_bounds


def test_simulate_doublet_pairs_is_seeded_and_heterotypic():
    clusters = np.array([0, 0, 1, 1])
    left, right = simulate_doublet_pairs(
        clusters,
        n_sim=12,
        heterotypic_fraction=1.0,
        rng=np.random.default_rng(11),
        max_tries=100,
    )

    np.testing.assert_array_equal(left, [0, 0, 3, 1, 2, 2, 2, 0, 1, 0, 1, 3])
    np.testing.assert_array_equal(right, [2, 3, 1, 3, 1, 1, 0, 2, 3, 2, 3, 0])
    assert np.all(clusters[left] != clusters[right])


def test_assign_cell_cycle_phase_preserves_rule_precedence():
    phases = assign_cell_cycle_phase(
        s_score=np.array([1.0, 0.1, -2.0, 0.0, -1.0]),
        g2m_score=np.array([0.5, 0.2, -1.0, 0.0, 0.5]),
    )

    np.testing.assert_array_equal(phases, ["S", "G2M", "G1", "S", "G2M"])


def test_gaussian_quantile_bounds_uses_median_and_population_deviation():
    bounds = gaussian_quantile_bounds(
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        min_p=0.1,
        max_p=0.9,
    )

    np.testing.assert_allclose(
        bounds,
        (1.1876123951263535, 4.8123876048736465),
        rtol=0,
        atol=1e-12,
    )

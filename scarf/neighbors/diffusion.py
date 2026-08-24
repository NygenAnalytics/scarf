import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


def diffusion_operator(graph: csr_matrix, power: int) -> coo_matrix:
    """Construct a powered row-normalized graph diffusion operator."""
    inverse_degree = np.ravel(graph.sum(axis=1))
    inverse_degree[inverse_degree != 0] = 1 / inverse_degree[inverse_degree != 0]
    n_cells = graph.shape[0]
    diagonal = csr_matrix(
        (
            inverse_degree,
            (range(n_cells), range(n_cells)),
        ),
        shape=[n_cells, n_cells],
    )
    return diagonal.dot(graph).__pow__(power).tocoo()

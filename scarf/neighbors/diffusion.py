import numpy as np
from numba import njit
from scipy.sparse import coo_matrix, csr_matrix

_INT32_MAX = int(np.iinfo(np.int32).max)
_FLOAT64_SIZE = int(np.dtype(np.float64).itemsize)


def _inverse_degree_diagonal(graph: csr_matrix) -> csr_matrix:
    inverse_degree = np.ravel(graph.sum(axis=1))
    inverse_degree[inverse_degree != 0] = 1 / inverse_degree[inverse_degree != 0]
    n_cells = graph.shape[0]
    return csr_matrix(
        (
            inverse_degree,
            (range(n_cells), range(n_cells)),
        ),
        shape=[n_cells, n_cells],
    )


def diffusion_operator(graph: csr_matrix, power: int) -> coo_matrix:
    """Construct a powered row-normalized graph diffusion operator."""
    diagonal = _inverse_degree_diagonal(graph)
    return diagonal.dot(graph).__pow__(power).tocoo()


@njit(cache=True)
def _product_nnz_bound(
    left_indptr: np.ndarray,
    left_indices: np.ndarray,
    right_row_nnz: np.ndarray,
    n_cols: int,
) -> int:
    """Sum per-row upper bounds on the entries of a sparse product."""
    total = 0
    for row in range(left_indptr.shape[0] - 1):
        count = 0
        for offset in range(left_indptr[row], left_indptr[row + 1]):
            count += right_row_nnz[left_indices[offset]]
        total += min(count, n_cols)
    return total


@njit(cache=True)
def _product_nnz(
    left_indptr: np.ndarray,
    left_indices: np.ndarray,
    right_indptr: np.ndarray,
    right_indices: np.ndarray,
    n_cols: int,
) -> int:
    """Count the structural entries of a sparse product without forming it."""
    last_row = np.full(n_cols, -1, dtype=np.int64)
    total = 0
    for row in range(left_indptr.shape[0] - 1):
        for offset in range(left_indptr[row], left_indptr[row + 1]):
            middle = left_indices[offset]
            for inner in range(right_indptr[middle], right_indptr[middle + 1]):
                col = right_indices[inner]
                if last_row[col] != row:
                    last_row[col] = row
                    total += 1
    return total


def _csr_bytes(*matrices: csr_matrix) -> int:
    """Return the bytes of distinct CSR matrices, counting an aliased one once."""
    unique = {id(matrix): matrix for matrix in matrices}
    return sum(
        int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)
        for matrix in unique.values()
    )


def _product_bytes(left: csr_matrix, right: csr_matrix, nnz: int) -> int:
    """Bytes SciPy allocates to multiply two CSR matrices with ``nnz`` outputs."""
    n_rows, n_cols = int(left.shape[0]), int(right.shape[1])
    narrow_inputs = all(
        np.dtype(array.dtype).itemsize <= 4
        for matrix in (left, right)
        for array in (matrix.indptr, matrix.indices)
    )
    wide = max(n_rows, n_cols, nnz) > _INT32_MAX or not narrow_inputs
    index_size = 8 if wide else 4
    output = nnz * (_FLOAT64_SIZE + index_size) + (n_rows + 1) * index_size
    # The product keeps one value and two links per output column.
    scratch = n_cols * (_FLOAT64_SIZE + 2 * index_size)
    if wide and narrow_inputs:
        # Operand index arrays are widened before the product.
        scratch += 8 * (int(left.nnz) + int(right.nnz) + n_rows + n_cols + 2)
    return output + scratch


def bounded_diffusion_operator(
    graph: csr_matrix,
    power: int,
    *,
    memory_bytes: int,
) -> csr_matrix:
    """Return ``diffusion_operator(graph, power)`` in CSR form within a budget.

    The same sparse products are formed, so the values are equal. Before each
    product, its output entries and the operands still held, including
    ``graph``, are checked against ``memory_bytes``, and ``MemoryError`` is
    raised before a product that does not fit is allocated.
    """
    if isinstance(power, bool) or not isinstance(power, int | np.integer):
        raise TypeError("power must be a positive integer")
    if power < 1:
        raise ValueError("power must be a positive integer")

    def reserve(
        left: csr_matrix,
        right: csr_matrix,
        held: tuple[csr_matrix, ...],
    ) -> None:
        held_bytes = _csr_bytes(graph, *held)
        n_cols = int(right.shape[1])
        bound = int(
            _product_nnz_bound(
                left.indptr,
                left.indices,
                np.diff(right.indptr),
                n_cols,
            )
        )
        if held_bytes + _product_bytes(left, right, bound) < memory_bytes:
            return
        exact = int(
            _product_nnz(
                left.indptr,
                left.indices,
                right.indptr,
                right.indices,
                n_cols,
            )
        )
        needed = held_bytes + _product_bytes(left, right, exact)
        if needed >= memory_bytes:
            raise MemoryError(
                f"Diffusion step with {exact} operator entries needs about "
                f"{needed} bytes, but the memory budget is {memory_bytes} bytes; "
                "increase the memory budget or use a smaller diffusion power t."
            )

    diagonal = _inverse_degree_diagonal(graph)
    reserve(diagonal, graph, (diagonal,))
    transition = diagonal.dot(graph)
    del diagonal

    def raised(exponent: int) -> csr_matrix:
        # Mirror scipy.sparse.linalg.matrix_power so values match exactly.
        if exponent == 1:
            return transition
        half = raised(exponent // 2)
        if exponent % 2:
            reserve(transition, half, (transition, half))
            partial = transition @ half
            reserve(partial, half, (transition, half, partial))
            return partial @ half
        reserve(half, half, (transition, half))
        return half @ half

    return csr_matrix(raised(int(power)))

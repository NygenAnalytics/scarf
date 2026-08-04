import numpy as np
from scipy.sparse import coo_matrix

from ..utils.logging import logger
from ..utils.progress import iter_progress


def _validate_neighbor_indices(
    name: str,
    values: np.ndarray,
    expected_cells: int | None = None,
) -> np.ndarray:
    indices = np.asarray(values)
    if indices.ndim != 2 or indices.shape[0] == 0:
        raise ValueError(f"WNN neighbors for {name} must be a non-empty matrix")
    if indices.shape[1] < 2:
        raise ValueError(
            f"WNN neighbors for {name} must contain at least two neighbors per cell"
        )
    if expected_cells is not None and indices.shape[0] != expected_cells:
        raise ValueError("WNN neighbor matrices must have the same number of cells")
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError(f"WNN neighbors for {name} must contain integer indices")

    n_cells = indices.shape[0]
    if int(indices.min()) < 0 or int(indices.max()) >= n_cells:
        raise ValueError(f"WNN neighbors for {name} contain indices outside cell range")
    for start in range(0, n_cells, 100_000):
        stop = min(start + 100_000, n_cells)
        block = indices[start:stop]
        rows = np.arange(start, stop)[:, np.newaxis]
        if np.any(block == rows):
            raise ValueError(f"WNN neighbors for {name} must exclude self")
        ordered = np.sort(block, axis=1)
        if np.any(ordered[:, 1:] == ordered[:, :-1]):
            raise ValueError(f"WNN neighbors for {name} must be unique within each row")
    return indices


def _validate_embedding(
    name: str,
    values: np.ndarray,
    n_cells: int,
) -> np.ndarray:
    embedding = np.asarray(values)
    if embedding.ndim != 2 or embedding.shape[1] == 0:
        raise ValueError(f"WNN embedding for {name} must be a non-empty matrix")
    if embedding.shape[0] != n_cells:
        raise ValueError(f"WNN embedding for {name} must have one row per graph cell")
    if not np.issubdtype(embedding.dtype, np.number) or np.issubdtype(
        embedding.dtype,
        np.complexfloating,
    ):
        raise TypeError(f"WNN embedding for {name} must contain real numeric values")
    if not np.issubdtype(embedding.dtype, np.floating):
        embedding = embedding.astype(np.float64)
    for start in range(0, n_cells, 100_000):
        if not np.all(np.isfinite(embedding[start : start + 100_000])):
            raise ValueError(f"WNN embedding for {name} contains non-finite values")
    return embedding


def _inverse_row_norms(values: np.ndarray) -> np.ndarray:
    norms = np.sqrt(
        np.einsum(
            "ij,ij->i",
            values,
            values,
            dtype=np.float64,
            optimize=True,
        )
    )
    inverse = np.zeros(norms.shape, dtype=np.float64)
    np.divide(1.0, norms, out=inverse, where=norms > 0)
    return inverse


def _kernel_affinity(
    distances: np.ndarray,
    nearest_distance: float,
    bandwidth: float,
) -> np.ndarray:
    adjusted = np.maximum(distances - nearest_distance, 0.0)
    # A bandwidth this close to zero is indistinguishable from rounding noise in
    # the distance reduction, so the tolerance tracks the local distance scale
    # to keep the result invariant to a rescaling of the embedding.
    tolerance = 8.0 * np.finfo(np.float64).eps * nearest_distance
    if bandwidth <= tolerance:
        return (adjusted <= tolerance).astype(np.float64)
    return np.exp(-(adjusted / bandwidth))


def _prediction_affinity(
    point: np.ndarray,
    estimate: np.ndarray,
    nearest_distance: float,
    bandwidth: float,
) -> float:
    estimate_distance = float(np.linalg.norm(point - estimate))
    return float(
        _kernel_affinity(
            np.asarray([estimate_distance]),
            nearest_distance,
            bandwidth,
        )[0]
    )


def _wnn_integration_many(
    modalities: list[tuple[str, np.ndarray, np.ndarray]],
    n_threads: int,
    *,
    l2_normalize: bool = True,
) -> tuple[coo_matrix, np.ndarray]:
    """Build an N-modality WNN graph using Scarf's bounded candidate pool."""
    if not isinstance(l2_normalize, bool | np.bool_):
        raise TypeError("l2_normalize must be a boolean")
    if len(modalities) < 2:
        raise ValueError("WNN integration requires at least two modalities")

    names = [name for name, _indices, _embedding in modalities]
    if len(set(names)) != len(names):
        raise ValueError("WNN modality names must be unique")

    first_name, first_indices, first_embedding = modalities[0]
    validated_indices = [_validate_neighbor_indices(first_name, first_indices)]
    n_cells = validated_indices[0].shape[0]
    validated_embeddings = [_validate_embedding(first_name, first_embedding, n_cells)]
    for name, indices, embedding in modalities[1:]:
        validated_indices.append(
            _validate_neighbor_indices(name, indices, expected_cells=n_cells)
        )
        validated_embeddings.append(_validate_embedding(name, embedding, n_cells))

    neighbor_counts = [indices.shape[1] for indices in validated_indices]
    nk = min(neighbor_counts)
    if len(set(neighbor_counts)) != 1:
        counts = ", ".join(
            f"{name}: {count}"
            for name, count in zip(names, neighbor_counts, strict=True)
        )
        logger.warning(
            f"WNN graphs have different neighbor counts ({counts}). "
            f"The integrated graph will retain {nk} "
            "neighbors per cell."
        )

    from threadpoolctl import threadpool_limits

    inverse_norms = (
        [_inverse_row_norms(embedding) for embedding in validated_embeddings]
        if l2_normalize
        else [None] * len(modalities)
    )
    index_dtype = np.uint32 if n_cells < 2**32 else np.uint64
    output_size = n_cells * nk
    merged_columns = np.empty(output_size, dtype=index_dtype)
    merged_data = np.empty(output_size, dtype=np.float32)
    modality_weights = np.empty((n_cells, len(modalities)), dtype=np.float32)

    with threadpool_limits(limits=n_threads):
        for cell_idx in iter_progress(
            range(n_cells),
            desc="Building WNN graph",
            total=n_cells,
        ):
            neighbor_rows = [indices[cell_idx] for indices in validated_indices]
            mixed_k = neighbor_rows[0]
            for neighbors in neighbor_rows[1:]:
                mixed_k = np.union1d(mixed_k, neighbors)

            candidate_data: list[np.ndarray] = []
            points: list[np.ndarray] = []
            distances: list[np.ndarray] = []
            positions = [
                np.searchsorted(mixed_k, neighbors) for neighbors in neighbor_rows
            ]
            nearest_distances: list[float] = []
            bandwidths: list[float] = []
            for embedding, inverse, own_positions in zip(
                validated_embeddings,
                inverse_norms,
                positions,
                strict=True,
            ):
                candidates = embedding[mixed_k]
                point = embedding[cell_idx]
                if inverse is not None:
                    candidates = candidates * inverse[mixed_k, np.newaxis]
                    point = point * inverse[cell_idx]
                modality_distances = np.linalg.norm(point - candidates, axis=1)
                ranked_distances = np.sort(modality_distances[own_positions])
                nearest_distance = float(ranked_distances[0])
                candidate_data.append(candidates)
                points.append(point)
                distances.append(modality_distances)
                nearest_distances.append(nearest_distance)
                bandwidths.append(float(ranked_distances[-1] - nearest_distance))

            within_affinities = [
                _prediction_affinity(
                    point,
                    candidates[own_positions].mean(axis=0),
                    nearest_distance,
                    bandwidth,
                )
                for point, candidates, own_positions, nearest_distance, bandwidth in zip(
                    points,
                    candidate_data,
                    positions,
                    nearest_distances,
                    bandwidths,
                    strict=True,
                )
            ]
            scores = np.full(
                (len(modalities), len(modalities)),
                -np.inf,
                dtype=np.float64,
            )
            for target_idx, (
                point,
                candidates,
                nearest_distance,
                bandwidth,
                within_affinity,
            ) in enumerate(
                zip(
                    points,
                    candidate_data,
                    nearest_distances,
                    bandwidths,
                    within_affinities,
                    strict=True,
                )
            ):
                for source_idx, source_positions in enumerate(positions):
                    if source_idx == target_idx:
                        continue
                    cross_affinity = _prediction_affinity(
                        point,
                        candidates[source_positions].mean(axis=0),
                        nearest_distance,
                        bandwidth,
                    )
                    scores[target_idx, source_idx] = np.clip(
                        within_affinity / (cross_affinity + 1e-4),
                        0,
                        200,
                    )

            finite_scores = scores[np.isfinite(scores)]
            max_score = float(finite_scores.max())
            pairwise_strengths = np.zeros(scores.shape, dtype=np.float64)
            finite = np.isfinite(scores)
            pairwise_strengths[finite] = np.exp(scores[finite] - max_score)
            modality_strengths = pairwise_strengths.sum(axis=1)
            weights = modality_strengths / modality_strengths.sum()
            modality_weights[cell_idx] = weights

            combined_affinity = weights[0] * _kernel_affinity(
                distances[0],
                nearest_distances[0],
                bandwidths[0],
            )
            for weight, modality_distances, nearest_distance, bandwidth in zip(
                weights[1:],
                distances[1:],
                nearest_distances[1:],
                bandwidths[1:],
                strict=True,
            ):
                combined_affinity += weight * _kernel_affinity(
                    modality_distances,
                    nearest_distance,
                    bandwidth,
                )
            selected = np.lexsort((mixed_k, -combined_affinity))[:nk]
            output_slice = slice(cell_idx * nk, (cell_idx + 1) * nk)
            merged_columns[output_slice] = mixed_k[selected]
            merged_data[output_slice] = np.maximum(
                np.clip(combined_affinity[selected], 0.0, 1.0),
                np.finfo(np.float32).tiny,
            )

    if not np.all(np.isfinite(merged_data)):
        raise FloatingPointError("WNN integration produced non-finite graph weights")
    if not np.all(np.isfinite(modality_weights)):
        raise FloatingPointError("WNN integration produced non-finite modality weights")
    rows = np.repeat(np.arange(n_cells, dtype=index_dtype), nk)
    graph = coo_matrix(
        (merged_data, (rows, merged_columns)),
        shape=(n_cells, n_cells),
    )
    return graph, modality_weights


def wnn_integration(
    name1: str,
    indices1: np.ndarray,
    ld1: np.ndarray,
    name2: str,
    indices2: np.ndarray,
    ld2: np.ndarray,
    n_threads: int,
    *,
    l2_normalize: bool = True,
) -> tuple[coo_matrix, np.ndarray]:
    """Build a two-modality WNN graph and per-cell modality weights.

    Candidates are the union of two self-free neighbour-index rows. Each
    modality uses its own nearest and k-th-neighbour distances to convert
    distances into affinities. Rows are L2-normalized during scoring by default.
    The returned COO graph stores blended affinity as float32 edge weights, and
    the second array stores two float32 modality weights per cell.

    This bounded candidate pool and simple bandwidth differ from Seurat's
    default wider search and SNN-far bandwidth. This adapter is the
    two-modality special case of Scarf's private N-modality implementation.
    """
    private_names = (
        (name1, name2) if name1 != name2 else (f"{name1} [first]", f"{name2} [second]")
    )
    return _wnn_integration_many(
        [
            (private_names[0], indices1, ld1),
            (private_names[1], indices2, ld2),
        ],
        n_threads,
        l2_normalize=l2_normalize,
    )

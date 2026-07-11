"""Benchmark two-pass Symphony query correction and fixed-reference projection.

Run with:
    uv run python -m tests.performance.bench_mapping
"""

import argparse
import math

import numpy as np

from scarf.symphony import (
    SymphonyReferenceModel,
    accumulate_sufficient_statistics,
    apply_query_correction,
    initialize_sufficient_statistics,
    project_pca,
    soft_cluster_assignments,
    solve_query_correction,
)

from ._common import (
    DEFAULT_DIMS,
    add_size_args,
    best_of,
    local_zarr_array,
    print_row,
    section,
)


def _reference(n_features: int, dims: int, clusters: int) -> SymphonyReferenceModel:
    rng = np.random.default_rng(0)
    return SymphonyReferenceModel(
        feature_means=rng.normal(size=n_features),
        feature_scales=np.abs(rng.normal(size=n_features)) + 0.1,
        loadings=rng.normal(size=(n_features, dims)),
        centroids=rng.normal(size=(clusters, dims)),
        raw_centroids=rng.normal(size=(clusters, dims)),
        corrected_centroids=rng.normal(size=(clusters, dims)),
        cluster_mass=np.full(clusters, 100.0),
        sigma=np.full(clusters, 0.1),
        correction_ridge=1.0,
    )


def run(args: argparse.Namespace) -> None:
    n_batches = 4
    clusters = 30
    model = _reference(args.n_cols, args.dims, clusters)
    batch_codes = np.arange(args.n_cells, dtype=np.int64) % n_batches
    n_blocks = math.ceil(args.n_cells / args.shard_rows)
    print(
        f"dataset: {args.n_cells} x {args.n_cols} -> {args.dims} dims, "
        f"{n_batches} query batches, {n_blocks} shards",
        flush=True,
    )

    with local_zarr_array(args.n_cells, args.n_cols, args.shard_rows) as data:
        section("two-pass Symphony-style correction")

        def map_query() -> np.ndarray:
            counts, sums = initialize_sufficient_statistics(n_batches, model)
            for start in range(0, args.n_cells, args.shard_rows):
                stop = min(start + args.shard_rows, args.n_cells)
                coordinates = project_pca(np.asarray(data[start:stop]), model)
                assignments = soft_cluster_assignments(coordinates, model)
                accumulate_sufficient_statistics(
                    counts,
                    sums,
                    coordinates,
                    assignments,
                    batch_codes[start:stop],
                )
            correction = solve_query_correction(counts, sums, model)
            corrected = []
            for start in range(0, args.n_cells, args.shard_rows):
                stop = min(start + args.shard_rows, args.n_cells)
                coordinates = project_pca(np.asarray(data[start:stop]), model)
                assignments = soft_cluster_assignments(coordinates, model)
                corrected.append(
                    apply_query_correction(
                        coordinates,
                        assignments,
                        batch_codes[start:stop],
                        model,
                        correction,
                    )
                )
            return np.vstack(corrected)

        result, seconds = best_of(map_query, n=args.repeats)
        print_row("two-pass correction", seconds)
        if not np.all(np.isfinite(result)):
            raise AssertionError("mapping benchmark produced non-finite coordinates")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_size_args(parser)
    parser.add_argument("--dims", type=int, default=DEFAULT_DIMS)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

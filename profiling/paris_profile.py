import argparse
import json
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import zarr
from scipy.sparse import csr_matrix

from profiling.metrics import ResourceSampler
from scarf.clustering._paris_core import ParisHierarchy
from scarf.clustering._paris_modularity import modularity_split_gains
from scarf.clustering.paris_multiscale import (
    PlateauForest,
    adaptive_cut,
    collapse_equal_height_plateaus,
)
from scarf.clustering.paris import fit_paris_hierarchy, hierarchy_to_dendrogram
from scarf.datastore._operations.paris_persistence import (
    adaptive_config_digest,
    estimate_paris_peak_bytes,
    load_adaptive_result,
    persist_adaptive_result,
    write_hierarchy_generation,
)
from scarf.storage.arrays import create_zarr_dataset


@dataclass(frozen=True)
class ThreadProfile:
    threads: int
    wallSeconds: float
    speedup: float
    reciprocalRounds: int
    rowScanSeconds: float
    sortSeconds: float
    contractionSeconds: float
    contractionRemapSeconds: float
    contractionFilterSeconds: float
    contractionBuildSeconds: float
    contractionCleanupSeconds: float
    meanMergeFraction: float
    initialActiveVertices: int
    initialActiveEdges: int
    finalActiveVertices: int
    finalActiveEdges: int
    peakRssBytes: int | None
    incrementalPeakRssBytes: int | None


@dataclass(frozen=True)
class ParisProfile:
    cells: int
    directedEdges: int
    estimatedPeakBytes: int
    preprocessingSeconds: float
    componentSeconds: float
    plateauSeconds: float
    modularityGuardSeconds: float
    inMemoryCutSeconds: float
    linkageSeconds: float
    hierarchyWriteSeconds: float
    adaptiveCacheWriteSeconds: float
    adaptiveCacheLoadSeconds: float
    metadataWriteSeconds: float
    hierarchyBytes: int
    eventBytes: int
    linkageBytes: int
    storedBytes: int
    adaptiveClusters: int
    threadProfiles: tuple[ThreadProfile, ...]


def random_directed_graph(
    n_cells: int,
    neighbors: int,
    seed: int,
) -> csr_matrix:
    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(n_cells, dtype=np.int64), neighbors)
    columns = rng.integers(0, n_cells, size=rows.size, dtype=np.int64)
    diagonal = rows == columns
    columns[diagonal] = (columns[diagonal] + 1) % n_cells
    weights = rng.uniform(0.1, 1.0, rows.size).astype(np.float32)
    graph = csr_matrix((weights, (rows, columns)), shape=(n_cells, n_cells))
    graph.sum_duplicates()
    graph.sort_indices()
    return graph


def _array_bytes(values: tuple[np.ndarray, ...]) -> int:
    return sum(array.nbytes for array in values)


def _hierarchy_bytes(hierarchy: ParisHierarchy) -> int:
    return _array_bytes(
        (
            hierarchy.children,
            hierarchy.heights,
            hierarchy.sizes,
            hierarchy.component_roots,
            hierarchy.synthetic_joins,
        )
    )


def _event_bytes(forest: PlateauForest) -> int:
    return _array_bytes(
        (
            forest.representatives,
            forest.heights,
            forest.sizes,
            forest.parent_events,
            forest.child_offsets,
            forest.child_refs,
            forest.min_leaves,
            forest.component_roots,
        )
    )


def _stored_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def profile_paris(
    *,
    n_cells: int,
    neighbors: int,
    seed: int,
    thread_counts: tuple[int, ...],
    min_cluster_size: int,
) -> ParisProfile:
    graph = random_directed_graph(n_cells, neighbors, seed)
    warmup_graph = random_directed_graph(64, 6, seed)
    fit_paris_hierarchy(warmup_graph, n_threads=1)

    measured: list[tuple[int, float, ParisHierarchy, int | None, int | None]] = []
    for threads in thread_counts:
        sampler = ResourceSampler(sampleIntervalSeconds=0.01)
        sampler.start()
        start = time.perf_counter()
        hierarchy = fit_paris_hierarchy(graph, n_threads=threads)
        wall_seconds = time.perf_counter() - start
        resources = sampler.stop()
        measured.append(
            (
                threads,
                wall_seconds,
                hierarchy,
                resources.processTreeRssPeakBytes,
                resources.processTreeRssIncrementalPeakBytes,
            )
        )

    baseline = measured[0][1]
    thread_profiles: list[ThreadProfile] = []
    for threads, wall_seconds, hierarchy, peak_rss, incremental_peak in measured:
        diagnostics = hierarchy.diagnostics
        if diagnostics is None:
            raise RuntimeError("Paris fit diagnostics are unavailable")
        rounds = diagnostics.rounds
        active_total = sum(round_.active_vertices for round_ in rounds)
        merge_fraction = (
            sum(round_.merges for round_ in rounds) / active_total
            if active_total
            else 0.0
        )
        thread_profiles.append(
            ThreadProfile(
                threads=threads,
                wallSeconds=wall_seconds,
                speedup=baseline / wall_seconds,
                reciprocalRounds=len(rounds),
                rowScanSeconds=sum(round_.scan_seconds for round_ in rounds),
                sortSeconds=sum(round_.sort_seconds for round_ in rounds),
                contractionSeconds=sum(round_.contraction_seconds for round_ in rounds),
                contractionRemapSeconds=sum(
                    round_.contraction_remap_seconds for round_ in rounds
                ),
                contractionFilterSeconds=sum(
                    round_.contraction_filter_seconds for round_ in rounds
                ),
                contractionBuildSeconds=sum(
                    round_.contraction_build_seconds for round_ in rounds
                ),
                contractionCleanupSeconds=sum(
                    round_.contraction_cleanup_seconds for round_ in rounds
                ),
                meanMergeFraction=merge_fraction,
                initialActiveVertices=rounds[0].active_vertices,
                initialActiveEdges=rounds[0].active_edges,
                finalActiveVertices=rounds[-1].active_vertices,
                finalActiveEdges=rounds[-1].active_edges,
                peakRssBytes=peak_rss,
                incrementalPeakRssBytes=incremental_peak,
            )
        )

    hierarchy = measured[-1][2]
    diagnostics = hierarchy.diagnostics
    if diagnostics is None:
        raise RuntimeError("Paris fit diagnostics are unavailable")
    start = time.perf_counter()
    forest = collapse_equal_height_plateaus(hierarchy)
    plateau_seconds = time.perf_counter() - start
    start = time.perf_counter()
    split_gate = modularity_split_gains(hierarchy, forest, graph)
    modularity_guard_seconds = time.perf_counter() - start
    start = time.perf_counter()
    result = adaptive_cut(
        hierarchy,
        min_cluster_size=min_cluster_size,
        plateau_forest=forest,
        split_gate=split_gate,
    )
    in_memory_cut_seconds = modularity_guard_seconds + time.perf_counter() - start
    start = time.perf_counter()
    linkage = hierarchy_to_dendrogram(hierarchy, compatibility=True)
    linkage_seconds = time.perf_counter() - start

    with tempfile.TemporaryDirectory(prefix="scarf-paris-profile-") as temp_dir:
        path = Path(temp_dir)
        root = zarr.open_group(store=str(path), mode="w")
        root.create_group("graph")
        start = time.perf_counter()
        generation_id, _location = write_hierarchy_generation(
            root,
            "graph",
            hierarchy,
            forest,
        )
        hierarchy_write_seconds = time.perf_counter() - start
        digest = adaptive_config_digest(generation_id, min_cluster_size)
        start = time.perf_counter()
        persist_adaptive_result(
            root,
            "graph",
            "paris_cluster",
            digest,
            result,
            generation_id=generation_id,
            final_label_key="RNA_paris_cluster",
            hierarchy_cache_hit=False,
            cut_seconds=in_memory_cut_seconds,
        )
        adaptive_cache_write_seconds = time.perf_counter() - start
        start = time.perf_counter()
        cached_result = load_adaptive_result(
            root,
            "graph",
            "paris_cluster",
            digest,
            hierarchy,
        )
        adaptive_cache_load_seconds = time.perf_counter() - start
        if cached_result is None:
            raise RuntimeError("Adaptive cache could not be reloaded")
        metadata = root.create_group("cells")
        start = time.perf_counter()
        labels = create_zarr_dataset(
            metadata,
            "RNA_paris_cluster",
            (min(100_000, n_cells),),
            np.int32,
            (n_cells,),
        )
        labels[:] = cached_result.labels
        metadata_write_seconds = time.perf_counter() - start
        stored_bytes = _stored_bytes(path)

    estimated_peak = estimate_paris_peak_bytes(
        n_cells,
        graph.nnz,
        graph.indices.dtype.itemsize,
        graph.data.dtype.itemsize,
        n_threads=max(thread_counts, default=1),
    )
    return ParisProfile(
        cells=n_cells,
        directedEdges=graph.nnz,
        estimatedPeakBytes=estimated_peak,
        preprocessingSeconds=diagnostics.preprocessing_seconds,
        componentSeconds=diagnostics.component_seconds,
        plateauSeconds=plateau_seconds,
        modularityGuardSeconds=modularity_guard_seconds,
        inMemoryCutSeconds=in_memory_cut_seconds,
        linkageSeconds=linkage_seconds,
        hierarchyWriteSeconds=hierarchy_write_seconds,
        adaptiveCacheWriteSeconds=adaptive_cache_write_seconds,
        adaptiveCacheLoadSeconds=adaptive_cache_load_seconds,
        metadataWriteSeconds=metadata_write_seconds,
        hierarchyBytes=_hierarchy_bytes(hierarchy),
        eventBytes=_event_bytes(forest),
        linkageBytes=linkage.nbytes,
        storedBytes=stored_bytes,
        adaptiveClusters=result.n_clusters,
        threadProfiles=tuple(thread_profiles),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=100_000)
    parser.add_argument("--neighbors", type=int, default=15)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--threads", default="1,2,4,8")
    parser.add_argument("--min-cluster-size", type=int, default=16)
    arguments = parser.parse_args()
    thread_counts = tuple(int(value) for value in arguments.threads.split(","))
    report = profile_paris(
        n_cells=arguments.cells,
        neighbors=arguments.neighbors,
        seed=arguments.seed,
        thread_counts=thread_counts,
        min_cluster_size=arguments.min_cluster_size,
    )
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()

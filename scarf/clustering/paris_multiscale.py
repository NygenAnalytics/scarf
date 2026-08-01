from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..storage.refs import ArtifactRef
from ._paris_core import ParisHierarchy

_INFEASIBLE = np.uint8(0)
_KEEP_SCORE = np.uint8(1)
_KEEP_CONSTRAINT = np.uint8(2)
_SPLIT = np.uint8(3)
_FORCED_ROOT = np.uint8(4)


@dataclass(frozen=True)
class ParisClusterDiagnostic:
    """Structural diagnostic for one selected Paris cluster."""

    label: int
    selected_node: int
    parent_event: int
    component: int
    size: int
    resolution_lower: float | None
    resolution_upper: float | None
    persistence: float | None
    forced: bool
    blocking_child_count: int
    folded_cell_count: int
    decision_margin: float | None


@dataclass(frozen=True)
class ParisClusteringResult:
    """Labels and diagnostics returned by a Paris cut.

    ``ref`` and ``label_key`` are set when the cut was written to a store, and
    stay ``None`` for cuts produced directly from a hierarchy.
    """

    labels: np.ndarray
    mode: Literal["auto", "fixed"]
    n_clusters: int
    diagnostics: tuple[ParisClusterDiagnostic, ...] = ()
    min_cluster_size: int | None = None
    label_key: str | None = None
    hierarchy_generation_id: str | None = None
    ref: ArtifactRef | None = None

    def __post_init__(self) -> None:
        if self.labels.ndim != 1 or self.labels.dtype != np.int32:
            raise ValueError("labels must be a one-dimensional int32 array")
        if self.n_clusters < 1:
            raise ValueError("n_clusters must be positive")
        if self.mode == "fixed" and self.diagnostics:
            raise ValueError("fixed cuts cannot contain adaptive diagnostics")
        self.labels.setflags(write=False)


@dataclass(frozen=True)
class PlateauForest:
    """Compact equal-height event forest."""

    representatives: np.ndarray
    heights: np.ndarray
    sizes: np.ndarray
    parent_events: np.ndarray
    child_offsets: np.ndarray
    child_refs: np.ndarray
    min_leaves: np.ndarray
    component_roots: np.ndarray
    n_leaves: int

    def __post_init__(self) -> None:
        n_events = self.representatives.size
        for name, values in (
            ("heights", self.heights),
            ("sizes", self.sizes),
            ("parent_events", self.parent_events),
            ("min_leaves", self.min_leaves),
        ):
            if values.shape != (n_events,):
                raise ValueError(f"{name} must have one value per event")
        if self.child_offsets.shape != (n_events + 1,):
            raise ValueError("child_offsets must have one more item than events")
        if self.child_offsets[0] != 0 or self.child_offsets[-1] != len(self.child_refs):
            raise ValueError("child offsets do not span child_refs")
        for values in (
            self.representatives,
            self.heights,
            self.sizes,
            self.parent_events,
            self.child_offsets,
            self.child_refs,
            self.min_leaves,
            self.component_roots,
        ):
            values.setflags(write=False)


def _validate_hierarchy(
    hierarchy: ParisHierarchy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_leaves = hierarchy.n_leaves
    if n_leaves < 2:
        raise ValueError("Paris hierarchy must contain at least two leaves")
    if not np.issubdtype(hierarchy.children.dtype, np.integer):
        raise TypeError("hierarchy child references must be integers")
    if not np.issubdtype(hierarchy.sizes.dtype, np.integer):
        raise TypeError("hierarchy subtree sizes must be integers")

    n_nodes = 2 * n_leaves - 1
    index_dtype = hierarchy.children.dtype
    parents = np.full(n_nodes, -1, dtype=index_dtype)
    subtree_sizes = np.ones(n_nodes, dtype=hierarchy.sizes.dtype)
    min_leaves = np.arange(n_nodes, dtype=index_dtype)
    heights = hierarchy.heights

    if np.isnan(heights).any() or np.any(heights < 0):
        raise ValueError("Paris merge distances must be non-negative and not NaN")
    if np.any(~hierarchy.synthetic_joins & ~np.isfinite(heights)):
        raise ValueError("Only synthetic component joins may have infinite distance")
    if np.any(hierarchy.synthetic_joins & ~np.isinf(heights)):
        raise ValueError("Synthetic component joins must have infinite distance")

    for merge_index in range(n_leaves - 1):
        node = n_leaves + merge_index
        left = int(hierarchy.children[merge_index, 0])
        right = int(hierarchy.children[merge_index, 1])
        if left == right or left < 0 or right < 0 or left >= node or right >= node:
            raise ValueError(
                "Hierarchy children must be distinct topological references"
            )
        if parents[left] != -1 or parents[right] != -1:
            raise ValueError("Hierarchy node has more than one parent")
        parents[left] = node
        parents[right] = node
        expected_size = int(subtree_sizes[left]) + int(subtree_sizes[right])
        if int(hierarchy.sizes[merge_index]) != expected_size:
            raise ValueError("Hierarchy subtree size does not match its children")
        subtree_sizes[node] = expected_size
        min_leaves[node] = min(int(min_leaves[left]), int(min_leaves[right]))

        parent_height = heights[merge_index]
        if hierarchy.synthetic_joins[merge_index]:
            continue
        for child in (left, right):
            if child < n_leaves:
                continue
            child_index = child - n_leaves
            if hierarchy.synthetic_joins[child_index]:
                raise ValueError("A finite merge cannot contain a synthetic join")
            if heights[child_index] > parent_height:
                raise ValueError("Paris merge distances must be monotone")

    candidate_roots: list[int] = []
    for node in range(n_nodes):
        if node >= n_leaves and hierarchy.synthetic_joins[node - n_leaves]:
            continue
        parent = int(parents[node])
        if parent == -1 or (
            parent >= n_leaves and hierarchy.synthetic_joins[parent - n_leaves]
        ):
            candidate_roots.append(node)
    expected_roots = np.asarray(candidate_roots, dtype=index_dtype)
    if not np.array_equal(
        np.sort(expected_roots),
        np.sort(hierarchy.component_roots),
    ):
        raise ValueError("Hierarchy component roots do not match synthetic joins")
    return parents, subtree_sizes, min_leaves


def collapse_equal_height_plateaus(hierarchy: ParisHierarchy) -> PlateauForest:
    """Collapse exact equal-height parent-child regions into multiway events."""
    parents, _subtree_sizes, raw_min_leaves = _validate_hierarchy(hierarchy)
    n_leaves = hierarchy.n_leaves
    n_nodes = 2 * n_leaves - 1
    finite = ~hierarchy.synthetic_joins
    event_tops = np.zeros(n_leaves - 1, dtype=bool)

    for merge_index in range(n_leaves - 1):
        if not finite[merge_index]:
            continue
        node = n_leaves + merge_index
        parent = int(parents[node])
        event_tops[merge_index] = (
            parent == -1
            or hierarchy.synthetic_joins[parent - n_leaves]
            or hierarchy.heights[parent - n_leaves] != hierarchy.heights[merge_index]
        )

    representatives = (
        np.flatnonzero(event_tops).astype(hierarchy.children.dtype) + n_leaves
    )
    n_events = representatives.size
    event_by_raw = np.full(n_nodes, -1, dtype=hierarchy.children.dtype)
    event_by_raw[representatives] = np.arange(
        n_events,
        dtype=hierarchy.children.dtype,
    )

    for merge_index in range(n_leaves - 2, -1, -1):
        if not finite[merge_index]:
            continue
        node = n_leaves + merge_index
        event = int(event_by_raw[node])
        if event < 0:
            raise RuntimeError("Plateau event propagation failed")
        height = hierarchy.heights[merge_index]
        for child in hierarchy.children[merge_index]:
            child_node = int(child)
            if (
                child_node >= n_leaves
                and finite[child_node - n_leaves]
                and hierarchy.heights[child_node - n_leaves] == height
            ):
                event_by_raw[child_node] = event

    child_counts = np.zeros(n_events, dtype=hierarchy.children.dtype)
    for merge_index in range(n_leaves - 1):
        if not finite[merge_index]:
            continue
        event = int(event_by_raw[n_leaves + merge_index])
        height = hierarchy.heights[merge_index]
        for child in hierarchy.children[merge_index]:
            child_node = int(child)
            same_plateau = (
                child_node >= n_leaves
                and finite[child_node - n_leaves]
                and hierarchy.heights[child_node - n_leaves] == height
            )
            if not same_plateau:
                child_counts[event] += 1

    child_offsets = np.empty(n_events + 1, dtype=hierarchy.children.dtype)
    child_offsets[0] = 0
    np.cumsum(child_counts, out=child_offsets[1:])
    child_refs = np.empty(int(child_offsets[-1]), dtype=hierarchy.children.dtype)
    write_offsets = child_offsets[:-1].copy()
    for merge_index in range(n_leaves - 1):
        if not finite[merge_index]:
            continue
        event = int(event_by_raw[n_leaves + merge_index])
        height = hierarchy.heights[merge_index]
        for child in hierarchy.children[merge_index]:
            child_node = int(child)
            same_plateau = (
                child_node >= n_leaves
                and finite[child_node - n_leaves]
                and hierarchy.heights[child_node - n_leaves] == height
            )
            if same_plateau:
                continue
            offset = int(write_offsets[event])
            if child_node < n_leaves:
                child_refs[offset] = -child_node - 1
            else:
                child_event = int(event_by_raw[child_node])
                if child_event < 0:
                    raise RuntimeError("Plateau boundary does not reference an event")
                child_refs[offset] = child_event
            write_offsets[event] += 1

    parent_events = np.full(n_events, -1, dtype=hierarchy.children.dtype)
    for event, representative in enumerate(representatives):
        parent = int(parents[int(representative)])
        if parent != -1 and not hierarchy.synthetic_joins[parent - n_leaves]:
            parent_event = int(event_by_raw[parent])
            if parent_event == event or parent_event < 0:
                raise RuntimeError("Plateau parent mapping failed")
            parent_events[event] = parent_event

    component_roots = np.empty(
        len(hierarchy.component_roots),
        dtype=hierarchy.children.dtype,
    )
    for component, root in enumerate(hierarchy.component_roots):
        raw_root = int(root)
        component_roots[component] = (
            -raw_root - 1 if raw_root < n_leaves else event_by_raw[raw_root]
        )

    representative_indices = representatives - n_leaves
    return PlateauForest(
        representatives=representatives,
        heights=hierarchy.heights[representative_indices].copy(),
        sizes=hierarchy.sizes[representative_indices].copy(),
        parent_events=parent_events,
        child_offsets=child_offsets,
        child_refs=child_refs,
        min_leaves=raw_min_leaves[representatives].copy(),
        component_roots=component_roots,
        n_leaves=n_leaves,
    )


def _event_keep_score(forest: PlateauForest, event: int) -> float:
    parent = int(forest.parent_events[event])
    if parent < 0:
        return 0.0
    creation_distance = float(forest.heights[event])
    parent_distance = float(forest.heights[parent])
    if parent_distance <= creation_distance:
        return 0.0
    # A logistic measure on log resolution is finite and centered at gamma=1.
    return int(forest.sizes[event]) * (
        1.0 / (1.0 + creation_distance) - 1.0 / (1.0 + parent_distance)
    )


def _score_events(
    forest: PlateauForest,
    min_cluster_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_events = forest.representatives.size
    best_scores = np.full(n_events, -np.inf, dtype=np.float64)
    states = np.full(n_events, _INFEASIBLE, dtype=np.uint8)
    blocking_counts = np.zeros(n_events, dtype=np.int32)
    folded_cells = np.zeros(n_events, dtype=forest.sizes.dtype)

    for event in range(n_events):
        split_score = 0.0
        blocking_count = 0
        folded_count = 0
        for offset in range(
            int(forest.child_offsets[event]),
            int(forest.child_offsets[event + 1]),
        ):
            child_ref = int(forest.child_refs[offset])
            if child_ref < 0:
                blocking_count += 1
                folded_count += 1
            elif states[child_ref] == _INFEASIBLE:
                blocking_count += 1
                folded_count += int(forest.sizes[child_ref])
            else:
                split_score += best_scores[child_ref]
        blocking_counts[event] = blocking_count
        folded_cells[event] = folded_count

        size = int(forest.sizes[event])
        is_root = forest.parent_events[event] < 0
        if size < min_cluster_size:
            if is_root:
                states[event] = _FORCED_ROOT
                best_scores[event] = 0.0
            continue

        split_feasible = blocking_count == 0
        keep_score = _event_keep_score(forest, event)
        if is_root:
            if split_feasible and split_score > 0:
                states[event] = _SPLIT
                best_scores[event] = split_score
            else:
                states[event] = _FORCED_ROOT
                best_scores[event] = 0.0
        elif not split_feasible:
            states[event] = _KEEP_CONSTRAINT
            best_scores[event] = keep_score
        elif keep_score >= split_score:
            states[event] = _KEEP_SCORE
            best_scores[event] = keep_score
        else:
            states[event] = _SPLIT
            best_scores[event] = split_score

    return best_scores, states, blocking_counts, folded_cells


def _selected_events(
    forest: PlateauForest,
    states: np.ndarray,
    split_gate: np.ndarray | None = None,
) -> list[tuple[int, int, bool]]:
    selected: list[tuple[int, int, bool]] = []
    stack = np.empty(max(1, forest.representatives.size), dtype=np.int64)
    for component, root_ref in enumerate(forest.component_roots):
        root = int(root_ref)
        if root < 0:
            selected.append((root, component, True))
            continue
        stack_size = 1
        stack[0] = root
        while stack_size:
            stack_size -= 1
            event = int(stack[stack_size])
            vetoed = split_gate is not None and float(split_gate[event]) <= 0.0
            if states[event] != _SPLIT or vetoed:
                selected.append((event, component, bool(states[event] == _FORCED_ROOT)))
                continue
            for offset in range(
                int(forest.child_offsets[event]),
                int(forest.child_offsets[event + 1]),
            ):
                child_ref = int(forest.child_refs[offset])
                if child_ref < 0:
                    raise RuntimeError("A feasible event split contains a leaf")
                stack[stack_size] = child_ref
                stack_size += 1
    return selected


def _fill_labels(
    hierarchy: ParisHierarchy,
    forest: PlateauForest,
    selected: list[tuple[int, int, bool]],
) -> tuple[np.ndarray, list[tuple[int, int, bool]]]:
    selected.sort(
        key=lambda item: (
            -item[0] - 1 if item[0] < 0 else int(forest.min_leaves[item[0]])
        )
    )
    selected_nodes = np.asarray(
        [
            -selected_ref - 1
            if selected_ref < 0
            else int(forest.representatives[selected_ref])
            for selected_ref, _component, _forced in selected
        ],
        dtype=hierarchy.children.dtype,
    )
    return labels_from_selected_nodes(hierarchy, selected_nodes), selected


def labels_from_selected_nodes(
    hierarchy: ParisHierarchy,
    selected_nodes: np.ndarray,
) -> np.ndarray:
    """Regenerate labels from disjoint selected hierarchy nodes."""
    selected = np.asarray(selected_nodes)
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("selected_nodes must be a non-empty one-dimensional array")
    if not np.issubdtype(selected.dtype, np.integer):
        raise TypeError("selected_nodes must contain integers")
    if len(np.unique(selected)) != len(selected):
        raise ValueError("selected_nodes must not contain duplicates")

    n_leaves = hierarchy.n_leaves
    labels = np.zeros(n_leaves, dtype=np.int32)
    stack = np.empty(n_leaves, dtype=hierarchy.children.dtype)
    for label, selected_node in enumerate(selected, start=1):
        raw_node = int(selected_node)
        if raw_node < 0 or raw_node >= 2 * n_leaves - 1:
            raise ValueError("selected node is outside the hierarchy")
        if raw_node >= n_leaves and hierarchy.synthetic_joins[raw_node - n_leaves]:
            raise ValueError("synthetic component joins cannot be selected")
        stack_size = 1
        stack[0] = raw_node
        while stack_size:
            stack_size -= 1
            node = int(stack[stack_size])
            if node < n_leaves:
                if labels[node] != 0:
                    raise RuntimeError("Adaptive Paris clusters overlap")
                labels[node] = label
                continue
            children = hierarchy.children[node - n_leaves]
            stack[stack_size] = children[0]
            stack[stack_size + 1] = children[1]
            stack_size += 2
    if np.any(labels == 0):
        raise RuntimeError("Adaptive Paris cut did not cover every leaf")
    return labels


def _diagnostics(
    forest: PlateauForest,
    selected: list[tuple[int, int, bool]],
    best_scores: np.ndarray,
    states: np.ndarray,
    blocking_counts: np.ndarray,
    folded_cells: np.ndarray,
) -> tuple[ParisClusterDiagnostic, ...]:
    diagnostics: list[ParisClusterDiagnostic] = []
    for label, (selected_ref, component, forced) in enumerate(selected, start=1):
        if selected_ref < 0:
            leaf = -selected_ref - 1
            diagnostics.append(
                ParisClusterDiagnostic(
                    label=label,
                    selected_node=leaf,
                    parent_event=-1,
                    component=component,
                    size=1,
                    resolution_lower=None,
                    resolution_upper=None,
                    persistence=None,
                    forced=True,
                    blocking_child_count=0,
                    folded_cell_count=0,
                    decision_margin=None,
                )
            )
            continue

        event = selected_ref
        parent = int(forest.parent_events[event])
        parent_representative = (
            -1 if parent < 0 else int(forest.representatives[parent])
        )
        size = int(forest.sizes[event])
        if forced or parent < 0:
            lower = None
            upper = None
            persistence = None
            margin = None
        else:
            lower = float(forest.heights[event])
            upper = float(forest.heights[parent])
            persistence = _event_keep_score(forest, event)
            if states[event] != _KEEP_SCORE:
                margin = None
            else:
                split_score = 0.0
                for offset in range(
                    int(forest.child_offsets[event]),
                    int(forest.child_offsets[event + 1]),
                ):
                    child_ref = int(forest.child_refs[offset])
                    if child_ref < 0:
                        raise RuntimeError(
                            "Score-selected event has an infeasible split"
                        )
                    split_score += best_scores[child_ref]
                margin = (persistence - split_score) / size

        diagnostics.append(
            ParisClusterDiagnostic(
                label=label,
                selected_node=int(forest.representatives[event]),
                parent_event=parent_representative,
                component=component,
                size=size,
                resolution_lower=lower,
                resolution_upper=upper,
                persistence=persistence,
                forced=forced,
                blocking_child_count=int(blocking_counts[event]),
                folded_cell_count=int(folded_cells[event]),
                decision_margin=margin,
            )
        )
    return tuple(diagnostics)


def adaptive_cut(
    hierarchy: ParisHierarchy,
    min_cluster_size: int = 2,
    *,
    plateau_forest: PlateauForest | None = None,
    split_gate: np.ndarray | None = None,
) -> ParisClusteringResult:
    """Select a deterministic branch-adaptive antichain from a Paris hierarchy.

    When ``split_gate`` is given it must hold one value per plateau event. A
    split the persistence score would otherwise accept is vetoed wherever the
    gate value is not strictly positive, coarsening the cut. This layers a
    configuration-null modularity guard on top of persistence without changing
    the underlying scoring; passing ``None`` reproduces the unguarded cut.
    """
    if isinstance(min_cluster_size, (bool, np.bool_)) or not isinstance(
        min_cluster_size,
        (int, np.integer),
    ):
        raise TypeError("min_cluster_size must be an integer")
    if min_cluster_size < 2:
        raise ValueError("min_cluster_size must be at least 2")
    min_cluster_size = int(min_cluster_size)
    forest = (
        collapse_equal_height_plateaus(hierarchy)
        if plateau_forest is None
        else plateau_forest
    )
    if forest.n_leaves != hierarchy.n_leaves:
        raise ValueError("Plateau forest and hierarchy have different leaf counts")
    if split_gate is not None:
        if split_gate.shape != forest.representatives.shape:
            raise ValueError("split_gate must have one value per plateau event")
        if not (
            np.issubdtype(split_gate.dtype, np.integer)
            or np.issubdtype(split_gate.dtype, np.floating)
        ):
            raise TypeError("split_gate must contain real numbers")
        if not np.isfinite(split_gate).all():
            raise ValueError("split_gate must contain only finite values")

    best_scores, states, blocking_counts, folded_cells = _score_events(
        forest,
        min_cluster_size,
    )
    selected = _selected_events(forest, states, split_gate)
    labels, selected = _fill_labels(hierarchy, forest, selected)
    diagnostics = _diagnostics(
        forest,
        selected,
        best_scores,
        states,
        blocking_counts,
        folded_cells,
    )
    return ParisClusteringResult(
        labels=labels,
        mode="auto",
        n_clusters=len(diagnostics),
        diagnostics=diagnostics,
        min_cluster_size=min_cluster_size,
    )

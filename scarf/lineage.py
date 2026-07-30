import json
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import Any, cast

import networkx as nx
import zarr

from .storage.artifacts import ArtifactRef, ArtifactStatus, inspect_artifact

type LineageTarget = ArtifactRef | Mapping[str, ArtifactRef]

_DETAIL_ITEM_LIMIT = 12
_DETAIL_VALUE_LIMIT = 160
_OMITTED = object()


def _ref_sort_key(ref: ArtifactRef) -> tuple[str, str, str, str]:
    return (ref.scope, ref.assay or "", ref.kind, ref.artifact_id)


def _input_path(parts: tuple[str, ...]) -> str:
    value = ""
    for part in parts:
        if part.startswith("["):
            value += part
        elif value:
            value += f".{part}"
        else:
            value = part
    return value or "input"


def _artifact_inputs(
    value: Any,
    path: tuple[str, ...] = (),
) -> Iterator[tuple[str, ArtifactRef]]:
    if isinstance(value, ArtifactRef):
        yield _input_path(path), value
        return
    if isinstance(value, Mapping):
        if value.get("type") == "artifact":
            try:
                ref = ArtifactRef.from_dict(value)
            except (TypeError, ValueError) as exc:
                location = _input_path(path)
                raise ValueError(
                    f"Invalid artifact reference at lineage input {location!r}"
                ) from exc
            yield _input_path(path), ref
            return
        for key in sorted(value):
            yield from _artifact_inputs(value[key], (*path, str(key)))
        return
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for index, item in enumerate(value):
            yield from _artifact_inputs(item, (*path, f"[{index}]"))


def _external_value(value: Any) -> Any:
    if isinstance(value, ArtifactRef):
        return _OMITTED
    if isinstance(value, Mapping):
        if value.get("type") == "artifact":
            return _OMITTED
        result = {}
        for key in sorted(value):
            item = _external_value(value[key])
            if item is not _OMITTED:
                result[str(key)] = item
        return result if result else _OMITTED
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        result = {}
        for index, raw_item in enumerate(value):
            item = _external_value(raw_item)
            if item is not _OMITTED:
                result[f"[{index}]"] = item
        return result if result else _OMITTED
    return value


def _external_inputs(inputs: Mapping[str, Any] | None) -> dict[str, Any]:
    if inputs is None:
        return {}
    value = _external_value(inputs)
    return {} if value is _OMITTED else cast(dict[str, Any], value)


def _normalize_outputs(target: LineageTarget) -> dict[str, ArtifactRef]:
    if isinstance(target, ArtifactRef):
        return {"output": target}
    if not isinstance(target, Mapping):
        raise TypeError("lineage target must be an ArtifactRef or a named mapping")
    if not target:
        raise ValueError("lineage output mapping cannot be empty")
    outputs = {}
    for name, ref in target.items():
        if not isinstance(name, str) or not name:
            raise TypeError("lineage output names must be non-empty strings")
        if not isinstance(ref, ArtifactRef):
            raise TypeError(f"lineage output {name!r} must be an ArtifactRef")
        outputs[name] = ref
    return dict(sorted(outputs.items()))


def _build_graph(
    root: zarr.Group,
    outputs: Mapping[str, ArtifactRef],
) -> nx.DiGraph:
    graph = nx.DiGraph()
    visited: set[ArtifactRef] = set()

    def visit(ref: ArtifactRef) -> None:
        if ref in visited:
            return
        visited.add(ref)
        status = inspect_artifact(root, ref)
        graph.add_node(ref, status=status, outputs=())
        dependencies = sorted(
            _artifact_inputs(status.inputs),
            key=lambda item: (item[0], _ref_sort_key(item[1])),
        )
        for input_name, input_ref in dependencies:
            if graph.has_edge(input_ref, ref):
                labels = set(graph.edges[input_ref, ref]["inputs"])
                labels.add(input_name)
                graph.edges[input_ref, ref]["inputs"] = tuple(sorted(labels))
            else:
                graph.add_edge(input_ref, ref, inputs=(input_name,))
            visit(input_ref)

    for ref in sorted(set(outputs.values()), key=_ref_sort_key):
        visit(ref)

    output_names: dict[ArtifactRef, list[str]] = defaultdict(list)
    for name, ref in outputs.items():
        output_names[ref].append(name)
    for ref in graph:
        graph.nodes[ref]["outputs"] = tuple(sorted(output_names[ref]))

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Artifact provenance contains a dependency cycle")
    return nx.freeze(graph)


def _status_label(status: ArtifactStatus) -> str:
    if not status.exists:
        return "missing"
    if not status.complete:
        return "incomplete"
    return "complete"


def _display_scope(ref: ArtifactRef) -> str:
    return ref.assay if ref.assay is not None else "datastore"


def _mermaid_text(value: str) -> str:
    return " ".join(value.replace('"', "'").replace("\\", "/").split())


def _detail_value(value: Any) -> str:
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    text = " ".join(text.replace("`", "'").split())
    if len(text) > _DETAIL_VALUE_LIMIT:
        return f"{text[: _DETAIL_VALUE_LIMIT - 3]}..."
    return text


def _detail_mapping(values: Mapping[str, Any]) -> str:
    items = sorted(values.items())
    shown = [
        f"{key}={_detail_value(value)}" for key, value in items[:_DETAIL_ITEM_LIMIT]
    ]
    hidden = len(items) - len(shown)
    if hidden:
        shown.append(f"... {hidden} more")
    return "; ".join(shown)


class ArtifactLineage:
    __slots__ = ("_graph", "_outputs")

    def __init__(
        self,
        graph: nx.DiGraph,
        outputs: Mapping[str, ArtifactRef],
    ) -> None:
        self._graph = graph
        self._outputs = MappingProxyType(dict(outputs))

    @classmethod
    def from_store(
        cls,
        root: zarr.Group,
        target: LineageTarget,
    ) -> "ArtifactLineage":
        outputs = _normalize_outputs(target)
        return cls(_build_graph(root, outputs), outputs)

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph

    @property
    def outputs(self) -> Mapping[str, ArtifactRef]:
        return self._outputs

    def _ordered_refs(self) -> list[ArtifactRef]:
        return list(
            nx.lexicographical_topological_sort(
                self._graph,
                key=_ref_sort_key,
            )
        )

    def to_mermaid(self) -> str:
        refs = self._ordered_refs()
        node_ids = {ref: f"artifact{index}" for index, ref in enumerate(refs)}
        lines = ["flowchart LR"]
        for ref in refs:
            status = cast(ArtifactStatus, self._graph.nodes[ref]["status"])
            label_parts = [
                f"{_display_scope(ref)} / {ref.kind}",
                status.operation or "operation unavailable",
                ref.artifact_id[:12],
            ]
            output_names = cast(tuple[str, ...], self._graph.nodes[ref]["outputs"])
            if output_names:
                label_parts.append(f"outputs: {', '.join(output_names)}")
            status_name = _status_label(status)
            if status_name != "complete":
                label_parts.append(f"status: {status_name}")
            label = _mermaid_text(" | ".join(label_parts))
            lines.append(f'    {node_ids[ref]}["{label}"]')

        edges = sorted(
            self._graph.edges,
            key=lambda edge: (node_ids[edge[1]], node_ids[edge[0]]),
        )
        for source, target in edges:
            input_names = cast(
                tuple[str, ...],
                self._graph.edges[source, target]["inputs"],
            )
            label = _mermaid_text(", ".join(input_names))
            lines.append(f'    {node_ids[source]} -->|"{label}"| {node_ids[target]}')
        return "\n".join(lines)

    def to_markdown(self) -> str:
        lines = [
            "```mermaid",
            self.to_mermaid(),
            "```",
            "",
            "### Artifact details",
        ]
        for ref in self._ordered_refs():
            status = cast(ArtifactStatus, self._graph.nodes[ref]["status"])
            lines.extend(
                [
                    "",
                    (
                        f"#### {_display_scope(ref)} / {ref.kind} / "
                        f"{ref.artifact_id[:12]}"
                    ),
                    f"- Status: `{_status_label(status)}`",
                    f"- Path: `{status.path}`",
                ]
            )
            if status.operation is not None:
                lines.append(f"- Operation: `{status.operation}`")
            output_names = cast(tuple[str, ...], self._graph.nodes[ref]["outputs"])
            if output_names:
                lines.append(
                    "- Outputs: "
                    + ", ".join(
                        f"`{name.replace('`', chr(39))}`" for name in output_names
                    )
                )
            parameters = status.parameters or {}
            if parameters:
                lines.append(f"- Parameters: `{_detail_mapping(parameters)}`")
            execution_options = status.execution_options or {}
            if execution_options:
                lines.append(
                    f"- Execution options: `{_detail_mapping(execution_options)}`"
                )
            external_inputs = _external_inputs(status.inputs)
            if external_inputs:
                lines.append(f"- Other inputs: `{_detail_mapping(external_inputs)}`")
        return "\n".join(lines)

    def _repr_markdown_(self) -> str:
        return self.to_markdown()

    def __repr__(self) -> str:
        return (
            f"ArtifactLineage(outputs={len(self._outputs)}, "
            f"artifacts={self._graph.number_of_nodes()}, "
            f"dependencies={self._graph.number_of_edges()})"
        )

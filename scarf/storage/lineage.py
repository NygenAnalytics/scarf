import json
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import networkx as nx
import zarr

from .artifacts import ArtifactStatus, inspect_artifact
from .refs import ArtifactLocator, ArtifactRef, ExternalArtifactRef

type LineageTarget = ArtifactRef | Mapping[str, ArtifactRef]


@dataclass(frozen=True, slots=True)
class _ExternalLineageRef:
    dataset_fingerprint: str
    source_assay: str
    ref: ArtifactRef


type _LineageLocator = ArtifactLocator | _ExternalLineageRef


_DETAIL_ITEM_LIMIT = 12
_DETAIL_VALUE_LIMIT = 160
_OMITTED = object()


def _ref_sort_key(ref: ArtifactRef) -> tuple[str, str, str, str]:
    return (ref.scope, ref.assay or "", ref.kind, ref.artifact_id)


def _locator_sort_key(
    locator: _LineageLocator,
) -> tuple[str, str, str, str, str, str]:
    if isinstance(locator, ExternalArtifactRef | _ExternalLineageRef):
        return (
            "external",
            locator.dataset_fingerprint,
            *_ref_sort_key(locator.ref),
        )
    return ("local", "", *_ref_sort_key(locator))


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
) -> Iterator[tuple[str, ArtifactLocator]]:
    if isinstance(value, ExternalArtifactRef):
        yield _input_path(path), value
        return
    if isinstance(value, ArtifactRef):
        yield _input_path(path), value
        return
    if isinstance(value, Mapping):
        if value.get("type") == "external_artifact":
            try:
                external_ref = ExternalArtifactRef.from_dict(value)
            except (TypeError, ValueError) as exc:
                location = _input_path(path)
                raise ValueError(
                    f"Invalid external artifact reference at lineage input {location!r}"
                ) from exc
            yield _input_path(path), external_ref
            return
        if value.get("type") == "artifact":
            try:
                artifact_ref = ArtifactRef.from_dict(value)
            except (TypeError, ValueError) as exc:
                location = _input_path(path)
                raise ValueError(
                    f"Invalid artifact reference at lineage input {location!r}"
                ) from exc
            yield _input_path(path), artifact_ref
            return
        for key in sorted(value):
            yield from _artifact_inputs(value[key], (*path, str(key)))
        return
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for index, item in enumerate(value):
            yield from _artifact_inputs(item, (*path, f"[{index}]"))


def _external_value(value: Any) -> Any:
    if isinstance(value, ArtifactRef | ExternalArtifactRef):
        return _OMITTED
    if isinstance(value, Mapping):
        if value.get("type") in ("artifact", "external_artifact"):
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


def _normalize_external_roots(
    external_roots: Mapping[str, zarr.Group] | None,
) -> dict[str, zarr.Group]:
    if external_roots is None:
        return {}
    if not isinstance(external_roots, Mapping):
        raise TypeError("external_roots must be a mapping or None")
    roots = {}
    for fingerprint, root in external_roots.items():
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError("External root fingerprints must be non-empty strings")
        if not isinstance(root, zarr.Group):
            raise TypeError(
                f"External root for fingerprint {fingerprint!r} must be a Zarr group"
            )
        roots[fingerprint] = root
    return roots


def _inspect_external_artifact(
    root: zarr.Group,
    locator: ExternalArtifactRef | _ExternalLineageRef,
    *,
    dataset_fingerprint_validated: bool = False,
) -> ArtifactStatus:
    assay_name = (
        locator.ref.assay
        if isinstance(locator, ExternalArtifactRef)
        else locator.source_assay
    )
    assert assay_name is not None
    if assay_name not in root or not isinstance(root[assay_name], zarr.Group):
        raise ValueError(
            f"External root for dataset fingerprint "
            f"{locator.dataset_fingerprint!r} is missing assay group "
            f"{assay_name!r}"
        )
    assay = root[assay_name]
    received = assay.attrs.get("dataset_fingerprint")
    if received is None and not dataset_fingerprint_validated:
        raise ValueError(
            f"External root assay {assay_name!r} has no stored "
            f"dataset_fingerprint; expected {locator.dataset_fingerprint!r}"
        )
    if received is not None and received != locator.dataset_fingerprint:
        raise ValueError(
            f"External root assay {assay_name!r} dataset fingerprint mismatch. "
            f"Expected {locator.dataset_fingerprint!r}, received {received!r}"
        )
    return inspect_artifact(root, locator.ref)


def _build_graph(
    root: zarr.Group,
    outputs: Mapping[str, ArtifactRef],
    external_roots: Mapping[str, zarr.Group],
    *,
    validated_external_fingerprints: frozenset[str] = frozenset(),
) -> nx.DiGraph:
    graph = nx.DiGraph()
    visited: set[_LineageLocator] = set()

    def visit(locator: _LineageLocator) -> None:
        if locator in visited:
            return
        visited.add(locator)
        if isinstance(locator, ExternalArtifactRef | _ExternalLineageRef):
            external_root = external_roots.get(locator.dataset_fingerprint)
            status = (
                None
                if external_root is None
                else _inspect_external_artifact(
                    external_root,
                    locator,
                    dataset_fingerprint_validated=(
                        locator.dataset_fingerprint in validated_external_fingerprints
                    ),
                )
            )
        else:
            status = inspect_artifact(root, locator)
        graph.add_node(locator, status=status, outputs=())
        if status is None:
            return
        dependencies = sorted(
            _artifact_inputs(status.inputs),
            key=lambda item: (item[0], _locator_sort_key(item[1])),
        )
        for input_name, input_locator in dependencies:
            dependency: _LineageLocator = input_locator
            if isinstance(
                locator,
                ExternalArtifactRef | _ExternalLineageRef,
            ) and isinstance(input_locator, ArtifactRef):
                source_assay = (
                    locator.ref.assay
                    if isinstance(locator, ExternalArtifactRef)
                    else locator.source_assay
                )
                assert source_assay is not None
                if (
                    input_locator.scope == "assay"
                    and input_locator.assay == source_assay
                ):
                    dependency = ExternalArtifactRef(
                        dataset_fingerprint=locator.dataset_fingerprint,
                        ref=input_locator,
                    )
                else:
                    dependency = _ExternalLineageRef(
                        dataset_fingerprint=locator.dataset_fingerprint,
                        source_assay=source_assay,
                        ref=input_locator,
                    )
            if graph.has_edge(dependency, locator):
                labels = set(graph.edges[dependency, locator]["inputs"])
                labels.add(input_name)
                graph.edges[dependency, locator]["inputs"] = tuple(sorted(labels))
            else:
                graph.add_edge(dependency, locator, inputs=(input_name,))
            visit(dependency)

    for ref in sorted(set(outputs.values()), key=_ref_sort_key):
        visit(ref)

    output_names: dict[ArtifactRef, list[str]] = defaultdict(list)
    for name, ref in outputs.items():
        output_names[ref].append(name)
    for locator in graph:
        names = output_names[locator] if isinstance(locator, ArtifactRef) else []
        graph.nodes[locator]["outputs"] = tuple(sorted(names))

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Artifact provenance contains a dependency cycle")
    return nx.freeze(graph)


def _status_label(status: ArtifactStatus | None) -> str:
    if status is None:
        return "unresolved external"
    if not status.exists:
        return "missing"
    if not status.complete:
        return "incomplete"
    return "complete"


def _located_ref(locator: _LineageLocator) -> ArtifactRef:
    if isinstance(locator, ExternalArtifactRef | _ExternalLineageRef):
        return locator.ref
    return locator


def _display_scope(locator: _LineageLocator) -> str:
    ref = _located_ref(locator)
    scope = ref.assay if ref.assay is not None else "datastore"
    if isinstance(locator, ExternalArtifactRef | _ExternalLineageRef):
        return f"external {locator.dataset_fingerprint[:12]} / {scope}"
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
        *,
        external_roots: Mapping[str, zarr.Group] | None = None,
    ) -> "ArtifactLineage":
        outputs = _normalize_outputs(target)
        roots = _normalize_external_roots(external_roots)
        return cls(_build_graph(root, outputs, roots), outputs)

    @classmethod
    def _from_validated_external_roots(
        cls,
        root: zarr.Group,
        target: LineageTarget,
        *,
        external_roots: Mapping[str, zarr.Group],
    ) -> "ArtifactLineage":
        """Build lineage after each external root fingerprint was validated."""
        outputs = _normalize_outputs(target)
        roots = _normalize_external_roots(external_roots)
        return cls(
            _build_graph(
                root,
                outputs,
                roots,
                validated_external_fingerprints=frozenset(roots),
            ),
            outputs,
        )

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph

    @property
    def outputs(self) -> Mapping[str, ArtifactRef]:
        return self._outputs

    def _ordered_refs(self) -> list[_LineageLocator]:
        return list(
            nx.lexicographical_topological_sort(
                self._graph,
                key=_locator_sort_key,
            )
        )

    def to_mermaid(self) -> str:
        locators = self._ordered_refs()
        node_ids = {
            locator: f"artifact{index}" for index, locator in enumerate(locators)
        }
        lines = ["flowchart LR"]
        for locator in locators:
            status = cast(
                ArtifactStatus | None,
                self._graph.nodes[locator]["status"],
            )
            ref = _located_ref(locator)
            label_parts = [
                f"{_display_scope(locator)} / {ref.kind}",
                (
                    status.operation or "operation unavailable"
                    if status is not None
                    else "external source unresolved"
                ),
                ref.artifact_id[:12],
            ]
            output_names = cast(
                tuple[str, ...],
                self._graph.nodes[locator]["outputs"],
            )
            if output_names:
                label_parts.append(f"outputs: {', '.join(output_names)}")
            status_name = _status_label(status)
            if status_name != "complete":
                label_parts.append(f"status: {status_name}")
            label = _mermaid_text(" | ".join(label_parts))
            lines.append(f'    {node_ids[locator]}["{label}"]')

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
        for locator in self._ordered_refs():
            status = cast(
                ArtifactStatus | None,
                self._graph.nodes[locator]["status"],
            )
            ref = _located_ref(locator)
            lines.extend(
                [
                    "",
                    (
                        f"#### {_display_scope(locator)} / {ref.kind} / "
                        f"{ref.artifact_id[:12]}"
                    ),
                    f"- Status: `{_status_label(status)}`",
                ]
            )
            if isinstance(locator, ExternalArtifactRef | _ExternalLineageRef):
                lines.append(f"- Dataset fingerprint: `{locator.dataset_fingerprint}`")
            if status is None:
                lines.append("- Resolution: `No matching external root was supplied`")
                continue
            lines.append(f"- Path: `{status.path}`")
            if status.operation is not None:
                lines.append(f"- Operation: `{status.operation}`")
            output_names = cast(
                tuple[str, ...],
                self._graph.nodes[locator]["outputs"],
            )
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

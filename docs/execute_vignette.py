"""Parse and execute one MyST notebook into an isolated output cache."""

import argparse
import hashlib
import html
import json
import math
import os
import platform
import re
import shutil
import sqlite3
import sys
from collections.abc import Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import nbformat
from jupyter_cache import __version__ as jupyter_cache_version
from jupyter_cache import get_cache
from jupyter_cache.base import CacheBundleIn
from myst_nb.core.config import NbParserConfig
from myst_nb.core.execute import create_client
from myst_nb.core.read import create_nb_reader
from myst_parser.config.main import MdParserConfig

DOCS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = DOCS_ROOT.parent
SOURCE_DIR = DOCS_ROOT / "source"
DEFAULT_CACHE = DOCS_ROOT / ".jupyter_cache"
DEFAULT_PAGE = "scrna_seq"
CACHE_VERSION = jupyter_cache_version
FINGERPRINT_VERSION = 2

_SKIP_DIR_NAMES = {
    "_build",
    "_static",
    "_templates",
    "dev",
    "scarf_datasets",
}
_JOURNAL_SUFFIXES = ("-journal", "-shm", "-wal")
_WIDGET_VIEW_MIME = "application/vnd.jupyter.widget-view+json"
_WIDGET_STATE_MIME = "application/vnd.jupyter.widget-state+json"
_MODEL_REFERENCE_PREFIX = "IPY_MODEL_"
_STATIC_PROGRESS_MARKER = 'class="scarf-static-progress"'
_HTML_TAG_RE = re.compile(r"<[^>]*>")
_ANSI_ESCAPE_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
_TQDM_PROGRESS_RE = re.compile(
    r"^(?P<label>.*?)\s*(?P<percent>\d{1,3}(?:\.\d+)?)%\|"
    r".*?(?P<value>\d[\d.,]*(?:[kMGTPE]?))\s*/\s*"
    r"(?P<total>\d[\d.,]*(?:[kMGTPE]?))",
    re.IGNORECASE,
)
_DASK_PROGRESS_RE = re.compile(
    r"^(?P<label>.*?)\[[#=\-\s]+\]\s*\|\s*"
    r"(?P<percent>\d{1,3}(?:\.\d+)?)%\s+(?:Completed|complete)\b",
    re.IGNORECASE,
)
_PROGRESS_LIKE_RE = re.compile(
    r"(?:\d{1,3}(?:\.\d+)?%\||\|\s*\d{1,3}(?:\.\d+)?%\s+"
    r"(?:Completed|complete)\b)",
    re.IGNORECASE,
)
_PROGRESS_PERCENT_RE = re.compile(r"(?P<percent>\d{1,3}(?:\.\d+)?)%")
_LONG_HEX_RE = re.compile(r"(?<![0-9a-f])[0-9a-f]{32,}(?![0-9a-f])", re.IGNORECASE)
_OUTPUT_CONTROL_RE = re.compile(r"[\r\b\x1b]")
_ALLOWED_TQDM_WIDGET_MODELS = frozenset(
    {
        "FloatProgressModel",
        "HBoxModel",
        "HTMLModel",
        "HTMLStyleModel",
        "IntProgressModel",
        "LayoutModel",
        "ProgressStyleModel",
        "VBoxModel",
    }
)

os.environ.setdefault("IPYTHONDIR", str(DOCS_ROOT / ".ipython"))


class CacheToolError(RuntimeError):
    pass


class CacheValidationError(CacheToolError):
    pass


class CacheBuildError(CacheToolError):
    pass


class ProgressRenderError(CacheToolError):
    pass


@dataclass(frozen=True, slots=True)
class ParsedSource:
    path: Path
    uri: str
    notebook: nbformat.NotebookNode
    hashkey: str


@dataclass(frozen=True, slots=True)
class ValidationReport:
    source_count: int
    record_count: int
    unique_hash_count: int


@dataclass(frozen=True, slots=True)
class ProgressSnapshot:
    label: str
    value: float
    total: float
    value_text: str
    total_text: str


def configure_doc_execution_env(
    env: MutableMapping[str, str] | None = None,
) -> MutableMapping[str, str]:
    target = os.environ if env is None else env
    target.update(
        {
            "SCARF_MEM_BUDGET": "4G",
            "SCARF_WORKERS": "2",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "NUMEXPR_NUM_THREADS": "2",
        }
    )
    target.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
    return target


configure_doc_execution_env()


class _Logger:
    prefix = ""

    def info(self, msg: str, subtype: str | None = None) -> None:
        print(f"{self.prefix}{msg}", flush=True)

    def warning(self, msg: str, subtype: str | None = None) -> None:
        print(f"{self.prefix}WARNING: {msg}", flush=True)

    def debug(self, msg: str, subtype: str | None = None) -> None:
        return None


def close_cache(cache: Any) -> None:
    engine = getattr(cache, "_db", None)
    if engine is not None:
        engine.dispose()
        cache._db = None


def notebook_hash(notebook: nbformat.NotebookNode) -> str:
    cache = get_cache(DEFAULT_CACHE)
    _, hashkey = cache.create_hashed_notebook(notebook)
    return str(hashkey)


def _coerce_output_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return "".join(value)
    raise ProgressRenderError("Progress output text is not a string")


def _clean_progress_label(value: str) -> str:
    label = html.unescape(_HTML_TAG_RE.sub("", value))
    label = label.replace("\u2007", " ").replace("\u200b", " ")
    label = re.sub(r"\s+", " ", label).strip()
    label = re.sub(
        r"\s*:?\s*\d{1,3}(?:\.\d+)?%\s*(?:\|.*)?$",
        "",
        label,
    )
    label = re.sub(
        r"\s+to\s+(?:(?:/|[A-Za-z]:[\\/])\S+|artifacts(?:[/\\]\S+)?)$",
        "",
        label,
        flags=re.IGNORECASE,
    )
    label = _LONG_HEX_RE.sub("", label)
    label = re.sub(r"\s+", " ", label).strip(" :|")
    if not label:
        return "Progress"
    return label[:160].rstrip()


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ProgressRenderError(f"Progress {field} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ProgressRenderError(f"Progress {field} is not finite")
    return result


def _format_progress_number(value: float) -> str:
    if value == 0:
        return "0"
    return format(value, ".12g")


def _completed_snapshot(
    *,
    label: str,
    value: float,
    total: float,
    value_text: str | None = None,
    total_text: str | None = None,
) -> ProgressSnapshot:
    if total <= 0:
        raise ProgressRenderError("Progress total must be positive")
    tolerance = max(1.0, abs(total)) * 1e-9
    if abs(value - total) > tolerance:
        raise ProgressRenderError(
            f"Progress output is incomplete: {_format_progress_number(value)} "
            f"of {_format_progress_number(total)}"
        )
    return ProgressSnapshot(
        label=_clean_progress_label(label),
        value=total,
        total=total,
        value_text=value_text or _format_progress_number(total),
        total_text=total_text or _format_progress_number(total),
    )


def _static_progress_output(
    snapshot: ProgressSnapshot,
) -> nbformat.NotebookNode:
    value = _format_progress_number(snapshot.value)
    total = _format_progress_number(snapshot.total)
    detail = f"{snapshot.value_text} / {snapshot.total_text} complete"
    plain = f"{snapshot.label}: {detail}"
    escaped_label = html.escape(snapshot.label, quote=True)
    escaped_detail = html.escape(detail, quote=True)
    escaped_plain = html.escape(plain, quote=True)
    rendered = (
        '<div class="scarf-static-progress">'
        f'<span class="scarf-static-progress__label">{escaped_label}</span>'
        f'<progress value="{value}" max="{total}" '
        f'aria-label="{escaped_plain}">{escaped_detail}</progress>'
        f'<span class="scarf-static-progress__detail">{escaped_detail}</span>'
        "</div>"
    )
    return nbformat.v4.new_output(
        "display_data",
        data={
            "text/html": rendered,
            "text/plain": plain,
        },
        metadata={},
    )


def _widget_state_models(
    notebook: nbformat.NotebookNode,
) -> Mapping[str, object] | None:
    widgets = notebook.metadata.get("widgets")
    if widgets is None:
        return None
    if not isinstance(widgets, Mapping):
        raise ProgressRenderError("Notebook widget metadata is invalid")
    if set(widgets) != {_WIDGET_STATE_MIME}:
        raise ProgressRenderError("Notebook contains unsupported widget metadata")
    payload = widgets.get(_WIDGET_STATE_MIME)
    if not isinstance(payload, Mapping):
        raise ProgressRenderError("Notebook widget state payload is invalid")
    state = payload.get("state")
    if not isinstance(state, Mapping) or not all(
        isinstance(model_id, str) for model_id in state
    ):
        raise ProgressRenderError("Notebook widget model state is invalid")
    return state


def _iter_model_references(value: object) -> Iterator[str]:
    if isinstance(value, str):
        if value.startswith(_MODEL_REFERENCE_PREFIX):
            yield value.removeprefix(_MODEL_REFERENCE_PREFIX)
        return
    if isinstance(value, Mapping):
        for nested in value.values():
            yield from _iter_model_references(nested)
        return
    if isinstance(value, list | tuple):
        for nested in value:
            yield from _iter_model_references(nested)


def _reachable_widget_models(
    root_id: str,
    models: Mapping[str, object],
) -> list[str]:
    ordered: list[str] = []
    visited: set[str] = set()

    def visit(model_id: str) -> None:
        if model_id in visited:
            return
        entry = models.get(model_id)
        if not isinstance(entry, Mapping):
            raise ProgressRenderError(
                f"Widget model {model_id!r} is missing from notebook state"
            )
        visited.add(model_id)
        ordered.append(model_id)
        for reference in _iter_model_references(entry):
            visit(reference)

    visit(root_id)
    return ordered


def _widget_model_parts(
    model_id: str,
    models: Mapping[str, object],
) -> tuple[str, Mapping[str, object]]:
    entry = models.get(model_id)
    if not isinstance(entry, Mapping):
        raise ProgressRenderError(f"Widget model {model_id!r} is invalid")
    if entry.get("buffers"):
        raise ProgressRenderError("Buffered widget state is not supported")
    state = entry.get("state")
    if not isinstance(state, Mapping):
        raise ProgressRenderError(f"Widget model {model_id!r} has no state")
    model_name = entry.get("model_name", state.get("_model_name"))
    if not isinstance(model_name, str):
        raise ProgressRenderError(f"Widget model {model_id!r} has no model name")
    return model_name, state


def _widget_progress_snapshot(
    root_id: str,
    models: Mapping[str, object],
) -> tuple[ProgressSnapshot, set[str]]:
    reachable = _reachable_widget_models(root_id, models)
    root_name, _ = _widget_model_parts(root_id, models)
    if root_name not in {"HBoxModel", "VBoxModel"}:
        raise ProgressRenderError(
            f"Widget output uses unsupported root model {root_name!r}"
        )

    progress_states: list[Mapping[str, object]] = []
    html_values: list[str] = []
    for model_id in reachable:
        model_name, state = _widget_model_parts(model_id, models)
        if model_name not in _ALLOWED_TQDM_WIDGET_MODELS:
            raise ProgressRenderError(
                f"Widget output uses unsupported model {model_name!r}"
            )
        if model_name in {"FloatProgressModel", "IntProgressModel"}:
            progress_states.append(state)
        elif model_name == "HTMLModel":
            value = state.get("value")
            if isinstance(value, str) and value.strip():
                html_values.append(value)

    if len(progress_states) != 1:
        raise ProgressRenderError(
            "Progress widget must contain exactly one progress model"
        )
    progress = progress_states[0]
    minimum = _finite_number(progress.get("min"), "minimum")
    value = _finite_number(progress.get("value"), "value")
    maximum = _finite_number(progress.get("max"), "maximum")
    if minimum != 0:
        raise ProgressRenderError("Progress widget minimum must be zero")

    label_source = next(
        (candidate for candidate in html_values if "%" in candidate),
        html_values[0] if html_values else "",
    )
    if not label_source:
        raise ProgressRenderError("Progress widget has no readable label")
    bar_style = progress.get("bar_style")
    if bar_style == "danger":
        label = _clean_progress_label(label_source)
        raise ProgressRenderError(
            f"Progress widget {label!r} finished in an error state at "
            f"{_format_progress_number(value)} of "
            f"{_format_progress_number(maximum)}"
        )
    if bar_style == "success":
        value = maximum
    return (
        _completed_snapshot(
            label=label_source,
            value=value,
            total=maximum,
        ),
        set(reachable),
    )


def _terminal_lines(text: str) -> list[tuple[str, bool]]:
    text = _ANSI_ESCAPE_RE.sub("", text)
    lines: list[tuple[str, bool]] = []
    current: list[str] = []
    cursor = 0
    for character in text:
        if character == "\r":
            cursor = 0
            continue
        if character == "\b":
            cursor = max(0, cursor - 1)
            continue
        if character == "\n":
            lines.append(("".join(current), True))
            current = []
            cursor = 0
            continue
        if ord(character) < 32 and character != "\t":
            raise ProgressRenderError(
                f"Progress stream contains unsupported control U+{ord(character):04X}"
            )
        if cursor < len(current):
            current[cursor] = character
        else:
            current.extend(" " * (cursor - len(current)))
            current.append(character)
        cursor += 1
    if current or (text and not text.endswith("\n")):
        lines.append(("".join(current), False))
    return lines


def _terminal_progress_snapshot(line: str) -> ProgressSnapshot | None:
    match = _TQDM_PROGRESS_RE.match(line.strip())
    if match is not None:
        percent = float(match.group("percent"))
        return _completed_snapshot(
            label=match.group("label"),
            value=percent,
            total=100.0,
            value_text=match.group("value"),
            total_text=match.group("total"),
        )

    match = _DASK_PROGRESS_RE.match(line.strip())
    if match is not None:
        percent = float(match.group("percent"))
        return _completed_snapshot(
            label=match.group("label"),
            value=percent,
            total=100.0,
            value_text="100",
            total_text="100",
        )

    if not _PROGRESS_LIKE_RE.search(line):
        return None
    percent_match = _PROGRESS_PERCENT_RE.search(line)
    if percent_match is None:
        raise ProgressRenderError("Progress stream has no readable percentage")
    percent = float(percent_match.group("percent"))
    return _completed_snapshot(
        label=line[: percent_match.start()],
        value=percent,
        total=100.0,
        value_text="100",
        total_text="100",
    )


def _render_terminal_stream(name: str, text: str) -> tuple[list[object], int]:
    rendered: list[object] = []
    stream_parts: list[str] = []
    progress_count = 0
    previous_snapshot: ProgressSnapshot | None = None

    def flush_stream() -> None:
        if not stream_parts:
            return
        rendered.append(
            nbformat.v4.new_output(
                "stream",
                name=name,
                text="".join(stream_parts),
            )
        )
        stream_parts.clear()

    for line, terminated in _terminal_lines(text):
        snapshot = _terminal_progress_snapshot(line)
        if snapshot is None:
            stream_parts.append(line)
            if terminated:
                stream_parts.append("\n")
            previous_snapshot = None
            continue
        flush_stream()
        if snapshot != previous_snapshot:
            rendered.append(_static_progress_output(snapshot))
            progress_count += 1
        previous_snapshot = snapshot
    flush_stream()
    return rendered, progress_count


def _freeze_cell_streams(cell: nbformat.NotebookNode) -> int:
    outputs = list(cell.get("outputs", []))
    grouped: dict[str, list[str]] = {}
    last_index: dict[str, int] = {}
    for index, output in enumerate(outputs):
        if output.get("output_type") != "stream":
            continue
        name = output.get("name")
        if name not in {"stdout", "stderr"}:
            raise ProgressRenderError(f"Unsupported notebook stream name: {name!r}")
        grouped.setdefault(name, []).append(_coerce_output_text(output.get("text")))
        last_index[name] = index

    transformed: dict[str, list[object]] = {}
    progress_count = 0
    for name, fragments in grouped.items():
        combined = "".join(fragments)
        if not _OUTPUT_CONTROL_RE.search(combined) and not _PROGRESS_LIKE_RE.search(
            combined
        ):
            continue
        replacement, count = _render_terminal_stream(name, combined)
        transformed[name] = replacement
        progress_count += count

    if not transformed:
        return 0
    rewritten: list[object] = []
    for index, output in enumerate(outputs):
        if output.get("output_type") != "stream":
            rewritten.append(output)
            continue
        name = output.get("name")
        if name not in transformed:
            rewritten.append(output)
        elif index == last_index[name]:
            rewritten.extend(transformed[name])
    cell.outputs = rewritten
    return progress_count


def _progress_integrity_issue(
    notebook: nbformat.NotebookNode,
) -> str | None:
    if "widgets" in notebook.metadata:
        return "notebook widget state remains after progress rendering"
    for cell_index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            data = output.get("data")
            if isinstance(data, Mapping):
                widget_mimes = [
                    mime
                    for mime in data
                    if isinstance(mime, str)
                    and mime.startswith("application/vnd.jupyter.widget")
                ]
                if widget_mimes:
                    return f"code cell {cell_index} contains a raw widget output"
                rendered_html = data.get("text/html")
                if rendered_html is not None:
                    try:
                        rendered_html = _coerce_output_text(rendered_html)
                    except ProgressRenderError as exc:
                        return f"code cell {cell_index}: {exc}"
                    if _STATIC_PROGRESS_MARKER in rendered_html and (
                        "<progress " not in rendered_html
                        or "</progress>" not in rendered_html
                        or "<script" in rendered_html.lower()
                        or "text/plain" not in data
                    ):
                        return (
                            f"code cell {cell_index} has invalid static progress HTML"
                        )
            if output.get("output_type") != "stream":
                continue
            try:
                text = _coerce_output_text(output.get("text"))
            except ProgressRenderError as exc:
                return f"code cell {cell_index}: {exc}"
            if _OUTPUT_CONTROL_RE.search(text):
                return f"code cell {cell_index} contains terminal controls"
            if _PROGRESS_LIKE_RE.search(text):
                return f"code cell {cell_index} contains raw progress output"
    return None


def validate_progress_outputs(
    notebook: nbformat.NotebookNode,
    source_uri: str = "<notebook>",
) -> None:
    issue = _progress_integrity_issue(notebook)
    if issue is not None:
        raise ProgressRenderError(f"{source_uri}: {issue}")


def freeze_progress_outputs(
    notebook: nbformat.NotebookNode,
    source_uri: str = "<notebook>",
) -> int:
    try:
        models = _widget_state_models(notebook)
        used_model_ids: set[str] = set()
        progress_count = 0
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            rewritten: list[object] = []
            for output in cell.get("outputs", []):
                data = output.get("data")
                if not isinstance(data, Mapping):
                    rewritten.append(output)
                    continue
                widget_mimes = {
                    mime
                    for mime in data
                    if isinstance(mime, str)
                    and mime.startswith("application/vnd.jupyter.widget")
                }
                if not widget_mimes:
                    rewritten.append(output)
                    continue
                if widget_mimes != {_WIDGET_VIEW_MIME}:
                    raise ProgressRenderError(
                        "Notebook contains an unsupported widget output"
                    )
                if output.get("output_type") != "display_data":
                    raise ProgressRenderError(
                        "Progress widget is not a display-data output"
                    )
                if set(data) - {_WIDGET_VIEW_MIME, "text/plain"}:
                    raise ProgressRenderError(
                        "Progress widget contains unsupported MIME data"
                    )
                view = data.get(_WIDGET_VIEW_MIME)
                if not isinstance(view, Mapping):
                    raise ProgressRenderError("Progress widget view is invalid")
                root_id = view.get("model_id")
                if not isinstance(root_id, str) or not root_id:
                    raise ProgressRenderError("Progress widget has no model ID")
                if models is None:
                    raise ProgressRenderError(
                        "Progress widget has no saved notebook state"
                    )
                snapshot, reachable = _widget_progress_snapshot(root_id, models)
                used_model_ids.update(reachable)
                rewritten.append(_static_progress_output(snapshot))
                progress_count += 1
            cell.outputs = rewritten
            progress_count += _freeze_cell_streams(cell)

        if models is not None:
            if not used_model_ids:
                raise ProgressRenderError(
                    "Notebook widget state has no rendered progress view"
                )
            unused = set(models) - used_model_ids
            if unused:
                raise ProgressRenderError(
                    "Notebook contains widget state unrelated to progress"
                )
            notebook.metadata.pop("widgets", None)
        validate_progress_outputs(notebook, source_uri)
        return progress_count
    except ProgressRenderError as exc:
        message = str(exc)
        if message.startswith(f"{source_uri}:"):
            raise
        raise ProgressRenderError(f"{source_uri}: {message}") from exc


def _parser_configs() -> tuple[MdParserConfig, NbParserConfig]:
    return (
        MdParserConfig(enable_extensions={"colon_fence"}),
        NbParserConfig(execution_mode="off"),
    )


def discover_sources(
    source_dir: Path = SOURCE_DIR,
    docs_root: Path = DOCS_ROOT,
) -> list[ParsedSource]:
    source_dir = source_dir.resolve()
    docs_root = docs_root.resolve()
    md_config, nb_config = _parser_configs()
    sources: list[ParsedSource] = []

    for path in sorted(source_dir.rglob("*.md")):
        relative_parts = path.relative_to(source_dir).parts
        if any(part in _SKIP_DIR_NAMES for part in relative_parts):
            continue
        content = path.read_text(encoding="utf-8")
        reader = create_nb_reader(str(path), md_config, nb_config, content)
        if reader is None:
            continue
        notebook = reader.read(content)
        uri = path.resolve().relative_to(docs_root).as_posix()
        sources.append(
            ParsedSource(
                path=path.resolve(),
                uri=uri,
                notebook=notebook,
                hashkey=notebook_hash(notebook),
            )
        )
    return sources


def iter_executable_paths() -> list[Path]:
    return [source.path for source in discover_sources()]


def _source_aliases(source: ParsedSource, source_dir: Path) -> set[str]:
    relative = source.path.relative_to(source_dir.resolve()).as_posix()
    without_suffix = relative.removesuffix(".md")
    return {
        source.path.stem,
        without_suffix,
        relative,
        source.uri,
        source.uri.removesuffix(".md"),
    }


def list_executable_docs() -> list[str]:
    sources = discover_sources()
    stem_counts: dict[str, int] = {}
    for source in sources:
        stem_counts[source.path.stem] = stem_counts.get(source.path.stem, 0) + 1
    return [
        source.path.stem
        if stem_counts[source.path.stem] == 1
        else source.path.relative_to(SOURCE_DIR).as_posix().removesuffix(".md")
        for source in sources
    ]


def list_vignettes() -> list[str]:
    return list_executable_docs()


def resolve_doc_source(
    name: str,
    sources: list[ParsedSource] | None = None,
    source_dir: Path = SOURCE_DIR,
) -> ParsedSource:
    sources = (
        discover_sources(source_dir, source_dir.parent) if sources is None else sources
    )
    cleaned = name.strip().removeprefix("./")
    if (
        not cleaned
        or PurePosixPath(cleaned).is_absolute()
        or ".." in PurePosixPath(cleaned).parts
    ):
        raise FileNotFoundError(f"Invalid executable doc name: {name!r}")

    matches = [
        source
        for source in sources
        if cleaned in _source_aliases(source, source_dir)
        or cleaned.removesuffix(".md") in _source_aliases(source, source_dir)
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        choices = ", ".join(
            source.path.relative_to(source_dir).as_posix().removesuffix(".md")
            for source in matches
        )
        raise FileNotFoundError(
            f"Ambiguous executable doc {name!r}; use one of: {choices}"
        )
    raise FileNotFoundError(f"Executable doc not found: {name}")


def resolve_doc_path(name: str) -> Path:
    return resolve_doc_source(name).path


def transfer_source_bundle(
    source_cache: Any,
    target_cache: Any,
    source: ParsedSource,
) -> None:
    try:
        target_cache.match_cache_notebook(source.notebook)
        return
    except KeyError:
        pass

    try:
        record = source_cache.match_cache_notebook(source.notebook)
    except KeyError as exc:
        raise CacheBuildError(f"No cache output matches {source.uri}") from exc

    bundle = source_cache.get_cache_bundle(record.pk)
    target_cache.cache_notebook_bundle(
        CacheBundleIn(
            bundle.nb,
            source.uri,
            artifacts=bundle.artifacts,
            data=dict(bundle.record.data or {}),
        ),
        check_validity=True,
        overwrite=False,
        description=bundle.record.description,
    )


def build_candidate(
    sources: list[ParsedSource],
    requested_uris: set[str],
    result_cache_path: Path,
    snapshot_cache_path: Path | None,
    candidate_path: Path,
) -> None:
    if candidate_path.exists():
        raise CacheBuildError(f"Candidate already exists: {candidate_path}")

    result_cache = get_cache(result_cache_path)
    snapshot_cache = (
        get_cache(snapshot_cache_path) if snapshot_cache_path is not None else None
    )
    candidate_cache = get_cache(candidate_path)
    requested = [source for source in sources if source.uri in requested_uris]
    preserved = [source for source in sources if source.uri not in requested_uris]

    try:
        for source in requested:
            transfer_source_bundle(result_cache, candidate_cache, source)
        for source in preserved:
            if snapshot_cache is None:
                raise CacheBuildError(
                    f"Cannot preserve {source.uri}: the existing cache is empty"
                )
            transfer_source_bundle(snapshot_cache, candidate_cache, source)
    finally:
        close_cache(result_cache)
        if snapshot_cache is not None:
            close_cache(snapshot_cache)
        close_cache(candidate_cache)


def _canonical_uri(uri: str, current_uris: set[str]) -> bool:
    pure = PurePosixPath(uri)
    return (
        uri.startswith("source/")
        and "\\" not in uri
        and not pure.is_absolute()
        and "." not in pure.parts
        and ".." not in pure.parts
        and pure.as_posix() == uri
        and uri in current_uris
    )


def _error_output_cells(notebook: nbformat.NotebookNode) -> set[int]:
    error_cells: set[int] = set()
    code_index = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        if any(
            output.get("output_type") == "error" for output in cell.get("outputs", [])
        ):
            error_cells.add(code_index)
        code_index += 1
    return error_cells


def _expected_error_cells(source: ParsedSource) -> set[int]:
    expected: set[int] = set()
    code_index = 0
    for cell in source.notebook.cells:
        if cell.cell_type != "code":
            continue
        tags = cell.get("metadata", {}).get("tags", [])
        if isinstance(tags, str):
            tags = [tags]
        if "raises-exception" in tags:
            expected.add(code_index)
        code_index += 1
    return expected


def _read_cache_rows_read_only(
    db_path: Path,
) -> tuple[list[tuple[str, str]], int]:
    db_uri = f"{db_path.resolve().as_uri()}?mode=ro&immutable=1"
    try:
        connection = sqlite3.connect(db_uri, uri=True)
    except sqlite3.Error as exc:
        raise CacheValidationError(f"Cannot open cache database: {exc}") from exc

    try:
        connection.execute("PRAGMA query_only = ON")
        integrity = [row[0] for row in connection.execute("PRAGMA integrity_check")]
        if integrity != ["ok"]:
            raise CacheValidationError(
                f"SQLite integrity check failed: {', '.join(integrity)}"
            )
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        missing = {"nbcache", "nbproject"} - tables
        if missing:
            names = ", ".join(sorted(missing))
            raise CacheValidationError(f"Cache database is missing tables: {names}")
        rows = [
            (str(hashkey), str(uri))
            for hashkey, uri in connection.execute(
                "SELECT hashkey, uri FROM nbcache ORDER BY hashkey"
            )
        ]
        project_count = int(
            connection.execute("SELECT COUNT(*) FROM nbproject").fetchone()[0]
        )
    except sqlite3.Error as exc:
        raise CacheValidationError(f"Cannot read cache database: {exc}") from exc
    finally:
        connection.close()
    return rows, project_count


def validate_cache(
    cache_path: Path = DEFAULT_CACHE,
    *,
    source_dir: Path = SOURCE_DIR,
    docs_root: Path = DOCS_ROOT,
) -> ValidationReport:
    cache_path = cache_path.resolve()
    if not cache_path.is_dir():
        raise CacheValidationError(f"Cache directory does not exist: {cache_path}")

    journal_files = [
        path
        for path in cache_path.rglob("*")
        if path.is_file() and path.name.endswith(_JOURNAL_SUFFIXES)
    ]
    if journal_files:
        names = ", ".join(
            path.relative_to(cache_path).as_posix() for path in journal_files
        )
        raise CacheValidationError(f"Cache contains SQLite journal files: {names}")

    version_path = cache_path / "__version__.txt"
    if not version_path.is_file():
        raise CacheValidationError("Cache version file is missing")
    version = version_path.read_text(encoding="utf-8").strip()
    if version != CACHE_VERSION:
        raise CacheValidationError(
            f"Unsupported cache version {version!r}; expected {CACHE_VERSION!r}"
        )

    db_path = cache_path / "global.db"
    if not db_path.is_file():
        raise CacheValidationError("Cache database is missing")

    sources = discover_sources(source_dir, docs_root)
    sources_by_uri = {source.uri: source for source in sources}
    sources_by_hash: dict[str, list[ParsedSource]] = {}
    for source in sources:
        sources_by_hash.setdefault(source.hashkey, []).append(source)
    current_hashes = set(sources_by_hash)
    current_uris = set(sources_by_uri)

    rows, project_count = _read_cache_rows_read_only(db_path)
    if project_count:
        raise CacheValidationError(
            f"Committed cache contains {project_count} project row(s)"
        )

    row_hashes = {hashkey for hashkey, _ in rows}
    stale_hashes = row_hashes - current_hashes
    if stale_hashes:
        raise CacheValidationError(
            f"Cache contains stale record hashes: {', '.join(sorted(stale_hashes))}"
        )
    missing_hashes = current_hashes - row_hashes
    if missing_hashes:
        missing_uris = sorted(
            source.uri
            for hashkey in missing_hashes
            for source in sources_by_hash[hashkey]
        )
        raise CacheValidationError(
            f"Current sources have no matching cache output: {', '.join(missing_uris)}"
        )

    executed_root = cache_path / "executed"
    entries = list(executed_root.iterdir()) if executed_root.is_dir() else []
    non_directories = [entry.name for entry in entries if not entry.is_dir()]
    if non_directories:
        raise CacheValidationError(
            f"Executed cache contains orphan files: {', '.join(sorted(non_directories))}"
        )
    directory_hashes = {entry.name for entry in entries if entry.is_dir()}
    missing_directories = row_hashes - directory_hashes
    orphan_directories = directory_hashes - row_hashes
    if missing_directories:
        raise CacheValidationError(
            "Cache records have no executed directory: "
            + ", ".join(sorted(missing_directories))
        )
    if orphan_directories:
        raise CacheValidationError(
            "Cache contains orphan executed directories: "
            + ", ".join(sorted(orphan_directories))
        )

    for hashkey, uri in rows:
        if not _canonical_uri(uri, current_uris):
            raise CacheValidationError(f"Cache URI is not canonical: {uri!r}")
        if sources_by_uri[uri].hashkey != hashkey:
            raise CacheValidationError(
                f"Cache URI {uri!r} does not agree with hash {hashkey}"
            )

        notebook_path = executed_root / hashkey / "base.ipynb"
        if not notebook_path.is_file():
            raise CacheValidationError(f"Cached notebook is missing: {notebook_path}")
        try:
            executed = nbformat.read(  # type: ignore[no-untyped-call]
                notebook_path,
                nbformat.NO_CONVERT,
            )
        except Exception as exc:
            raise CacheValidationError(
                f"Cached notebook is not parseable: {notebook_path}"
            ) from exc
        actual_hash = notebook_hash(executed)
        if actual_hash != hashkey:
            raise CacheValidationError(
                f"Cached notebook hash {actual_hash} disagrees with directory {hashkey}"
            )

        source = sources_by_hash[hashkey][0]
        source_code_count = sum(
            cell.cell_type == "code" for cell in source.notebook.cells
        )
        executed_code_count = sum(cell.cell_type == "code" for cell in executed.cells)
        if source_code_count != executed_code_count:
            raise CacheValidationError(
                f"Cached notebook code-cell count disagrees for {source.uri}"
            )
        expected_execution_count = 1
        for cell in executed.cells:
            if cell.cell_type != "code":
                continue
            if cell.execution_count != expected_execution_count:
                raise CacheValidationError(
                    f"Cached notebook execution counts are incomplete for {source.uri}"
                )
            expected_execution_count += 1
        unexpected_errors = _error_output_cells(executed) - _expected_error_cells(
            source
        )
        if unexpected_errors:
            cells = ", ".join(str(index) for index in sorted(unexpected_errors))
            raise CacheValidationError(
                f"Unexpected error output in {source.uri} code cell(s): {cells}"
            )
        progress_issue = _progress_integrity_issue(executed)
        if progress_issue is not None:
            raise CacheValidationError(f"{source.uri}: {progress_issue}")

    return ValidationReport(
        source_count=len(sources),
        record_count=len(rows),
        unique_hash_count=len(current_hashes),
    )


def _hash_file(hasher: Any, path: Path, label: str) -> None:
    hasher.update(label.encode())
    hasher.update(b"\0")
    hasher.update(path.read_bytes())
    hasher.update(b"\0")


def execution_fingerprint(
    repo_root: Path = REPO_ROOT,
    docs_root: Path = DOCS_ROOT,
) -> str:
    hasher = hashlib.sha256()
    runtime = {
        "fingerprintVersion": FINGERPRINT_VERSION,
        "implementation": platform.python_implementation(),
        "python": platform.python_version(),
        "cacheTag": sys.implementation.cache_tag,
    }
    hasher.update(json.dumps(runtime, sort_keys=True).encode())

    fixed_inputs = [
        repo_root / "pyproject.toml",
        repo_root / "uv.lock",
        docs_root / "execute_vignette.py",
        docs_root / "execute_all_vignettes.py",
        docs_root / "modal_cache.py",
        docs_root / "modal_docs.py",
        docs_root / "Makefile",
        docs_root / "source" / "conf.py",
    ]
    for path in fixed_inputs:
        if path.is_file():
            _hash_file(hasher, path, path.relative_to(repo_root).as_posix())

    scarf_root = repo_root / "scarf"
    if scarf_root.is_dir():
        for path in sorted(scarf_root.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            _hash_file(hasher, path, path.relative_to(repo_root).as_posix())
    return hasher.hexdigest()


@contextmanager
def serialization_lock(target_path: Path) -> Iterator[None]:
    lock_path = target_path.with_name(f"{target_path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(  # type: ignore[attr-defined]
                handle.fileno(),
                msvcrt.LK_LOCK,  # type: ignore[attr-defined]
                1,
            )
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(  # type: ignore[attr-defined]
                handle.fileno(),
                msvcrt.LK_UNLCK,  # type: ignore[attr-defined]
                1,
            )
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def backup_path(target_path: Path) -> Path:
    return target_path.with_name(f"{target_path.name}.backup")


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _rename_path(source: Path, destination: Path) -> None:
    os.rename(source, destination)


def recover_interrupted_swap(target_path: Path) -> bool:
    backup = backup_path(target_path)
    if not backup.exists():
        return False
    if target_path.exists():
        _remove_path(backup)
    else:
        _rename_path(backup, target_path)
    return True


def _publish_candidate_locked(candidate_path: Path, target_path: Path) -> None:
    backup = backup_path(target_path)
    recover_interrupted_swap(target_path)
    moved_target = False
    if target_path.exists():
        _rename_path(target_path, backup)
        moved_target = True
    try:
        _rename_path(candidate_path, target_path)
    except BaseException:
        if target_path.exists():
            _remove_path(target_path)
        if moved_target and backup.exists():
            _rename_path(backup, target_path)
        raise
    if backup.exists():
        try:
            _remove_path(backup)
        except OSError:
            pass


def publish_candidate(candidate_path: Path, target_path: Path = DEFAULT_CACHE) -> None:
    with serialization_lock(target_path):
        _publish_candidate_locked(candidate_path, target_path)


def execute_page(
    name: str,
    *,
    cache_path: Path,
    execution_in_temp: bool = True,
    source_dir: Path = SOURCE_DIR,
    docs_root: Path = DOCS_ROOT,
) -> dict[str, object]:
    sources = discover_sources(source_dir, docs_root)
    source = resolve_doc_source(name, sources, source_dir)
    cache_path = cache_path.resolve()
    execution_cache_path = cache_path.with_name(f"{cache_path.name}.execution")
    if cache_path.exists() or execution_cache_path.exists():
        raise CacheToolError(
            f"Isolated page cache paths must not already exist: {cache_path}"
        )

    nb_config = NbParserConfig(
        execution_mode="cache",
        execution_cache_path=str(execution_cache_path),
        execution_timeout=600,
        execution_allow_errors=False,
        execution_raise_on_error=True,
        execution_show_tb=True,
        execution_in_temp=execution_in_temp,
    )
    md_config = MdParserConfig(enable_extensions={"colon_fence"})
    content = source.path.read_text(encoding="utf-8")
    reader = create_nb_reader(str(source.path), md_config, nb_config, content)
    if reader is None:
        raise CacheToolError(f"Not a MyST notebook page: {source.path}")
    notebook = reader.read(content)
    if notebook_hash(notebook) != source.hashkey:
        raise CacheToolError(f"Source changed while preparing execution: {source.uri}")

    logger = _Logger()
    logger.prefix = f"[{source.uri}] "
    execution_cache = None
    output_cache = None
    try:
        with create_client(
            notebook,
            str(source.path),
            nb_config,
            logger,
            reader.read_fmt,
        ) as client:
            metadata = client.exec_metadata or {}

        execution_cache = get_cache(execution_cache_path)
        record = execution_cache.match_cache_notebook(notebook)
        bundle = execution_cache.get_cache_bundle(record.pk)
        freeze_progress_outputs(bundle.nb, source.uri)
        output_cache = get_cache(cache_path)
        output_cache.cache_notebook_bundle(
            CacheBundleIn(
                bundle.nb,
                source.uri,
                artifacts=bundle.artifacts,
                data=dict(bundle.record.data or {}),
            ),
            check_validity=True,
            overwrite=False,
        )
        output_cache.match_cache_notebook(source.notebook)
    except BaseException:
        close_cache(execution_cache)
        close_cache(output_cache)
        _remove_path(cache_path)
        raise
    finally:
        close_cache(execution_cache)
        close_cache(output_cache)
        _remove_path(execution_cache_path)

    print(f"[{source.uri}] finished", flush=True)
    return {
        "name": name,
        "uri": source.uri,
        "hashkey": source.hashkey,
        "metadata": metadata,
    }


def execute_vignette(
    name: str,
    *,
    cache_path: Path,
    execution_in_temp: bool = True,
) -> dict[str, object]:
    return execute_page(
        name,
        cache_path=cache_path,
        execution_in_temp=execution_in_temp,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "page",
        nargs="?",
        default=DEFAULT_PAGE,
        help="Page stem, source-relative path, or canonical source URI",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        help="New isolated output cache directory",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List executable page names and exit",
    )
    cli = parser.parse_args()
    if cli.list:
        for page_name in list_executable_docs():
            print(page_name)
        return
    if cli.cache_path is None:
        parser.error(
            "--cache-path is required; use execute_all_vignettes.py to publish"
        )
    execute_page(cli.page, cache_path=cli.cache_path)


if __name__ == "__main__":
    main()

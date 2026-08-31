import copy
import io
import shutil
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

try:
    import nbformat
    from jupyter_cache import get_cache
    from jupyter_cache.base import CacheBundleIn
    from jupyter_cache.cache.main import NbArtifacts

    import docs.execute_vignette as cache_tools
    import docs.modal_cache as modal_cache
    from docs.execute_all_vignettes import (
        ExecutionBatchError,
        _prepare_resume,
        _record_result,
        _valid_resume_uris,
        execute_and_publish,
        prune_and_publish,
    )
    from docs.execute_vignette import (
        CacheBuildError,
        CacheValidationError,
        ProgressRenderError,
        build_candidate,
        close_cache,
        discover_sources,
        execution_fingerprint,
        freeze_progress_outputs,
        publish_candidate,
        validate_cache,
        validate_progress_outputs,
    )
    from docs.modal_cache import (
        CacheTransportError,
        SpawnedPageRunner,
        await_page_cache,
        pack_page_cache,
        restore_page_cache,
    )
except ImportError:
    pytest.skip("documentation dependencies are not installed", allow_module_level=True)


def _write_source(
    docs_root: Path,
    name: str,
    *,
    code: str = "value = 1",
    raises_exception: bool = False,
) -> Path:
    path = docs_root / "source" / f"{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    tags = ":tags: [raises-exception]\n\n" if raises_exception else ""
    path.write_text(
        "---\n"
        "jupytext:\n"
        "  text_representation:\n"
        "    extension: .md\n"
        "    format_name: myst\n"
        "kernelspec:\n"
        "  display_name: Python 3\n"
        "  language: python\n"
        "  name: python3\n"
        "---\n\n"
        f"# {name}\n\n"
        "```{code-cell} ipython3\n"
        f"{tags}"
        f"{code}\n"
        "```\n",
        encoding="utf-8",
    )
    return path


def test_doc_execution_env_uses_shared_dataset_directory() -> None:
    env = cache_tools.configure_doc_execution_env({})

    assert env["SCARF_DOCS_DATA_DIR"] == str(cache_tools.SOURCE_DIR / "scarf_datasets")


def _source(docs_root: Path, name: str):
    return next(
        source
        for source in discover_sources(docs_root / "source", docs_root)
        if source.path.stem == name
    )


def _executed_notebook(
    source,
    *,
    text: str = "ok\n",
    error: bool = False,
) -> nbformat.NotebookNode:
    notebook = copy.deepcopy(source.notebook)
    execution_count = 1
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        cell.execution_count = execution_count
        execution_count += 1
        if error:
            cell.outputs = [
                nbformat.v4.new_output(
                    "error",
                    ename="ValueError",
                    evalue="expected",
                    traceback=["ValueError: expected"],
                )
            ]
        else:
            cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text=text)]
    return notebook


def _first_code_cell(notebook: nbformat.NotebookNode) -> nbformat.NotebookNode:
    return next(cell for cell in notebook.cells if cell.cell_type == "code")


def _widget_progress_notebook(
    source,
    *,
    value: float = 1.0,
    maximum: float = 1.0,
    root_model: str = "HBoxModel",
    bar_style: str = "success",
) -> nbformat.NotebookNode:
    notebook = _executed_notebook(source)
    root_id = "root"
    label_id = "label"
    progress_id = "progress"
    detail_id = "detail"
    digest = "a" * 64
    label = f"<b>Writing data</b> to artifacts/normalized/{digest}/data: 100%| "
    _first_code_cell(notebook).outputs = [
        nbformat.v4.new_output(
            "display_data",
            data={
                "application/vnd.jupyter.widget-view+json": {
                    "model_id": root_id,
                    "version_major": 2,
                    "version_minor": 0,
                },
                "text/plain": "widget progress",
            },
        )
    ]
    notebook.metadata["widgets"] = {
        "application/vnd.jupyter.widget-state+json": {
            "state": {
                root_id: {
                    "model_name": root_model,
                    "state": {
                        "children": [
                            f"IPY_MODEL_{label_id}",
                            f"IPY_MODEL_{progress_id}",
                            f"IPY_MODEL_{detail_id}",
                        ]
                    },
                },
                label_id: {
                    "model_name": "HTMLModel",
                    "state": {"value": label},
                },
                progress_id: {
                    "model_name": "FloatProgressModel",
                    "state": {
                        "bar_style": bar_style,
                        "max": maximum,
                        "min": 0.0,
                        "value": value,
                    },
                },
                detail_id: {
                    "model_name": "HTMLModel",
                    "state": {"value": "1/1"},
                },
            },
            "version_major": 2,
            "version_minor": 0,
        }
    }
    return notebook


def _cache_output(
    cache_path: Path,
    source,
    *,
    text: str = "ok\n",
    error: bool = False,
    uri: str | None = None,
    artifact_root: Path | None = None,
    notebook: nbformat.NotebookNode | None = None,
) -> None:
    artifacts = None
    if artifact_root is not None:
        artifact = artifact_root / "nested" / "result.txt"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("artifact", encoding="utf-8")
        artifacts = NbArtifacts([artifact], in_folder=artifact_root)

    cache = get_cache(cache_path)
    try:
        cache.cache_notebook_bundle(
            CacheBundleIn(
                notebook or _executed_notebook(source, text=text, error=error),
                uri or source.uri,
                artifacts=artifacts,
                data={"execution_seconds": 0.01},
            ),
            check_validity=True,
            overwrite=True,
        )
    finally:
        close_cache(cache)


def _output_text(cache_path: Path, source) -> str:
    cache = get_cache(cache_path)
    try:
        record = cache.match_cache_notebook(source.notebook)
        bundle = cache.get_cache_bundle(record.pk)
        return bundle.nb.cells[0].outputs[0].get("text", "")
    finally:
        close_cache(cache)


def _tree_bytes(path: Path) -> dict[str, bytes]:
    return {
        file.relative_to(path).as_posix(): file.read_bytes()
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }


def _initialize_cache(path: Path) -> None:
    cache = get_cache(path)
    cache.db
    close_cache(cache)


def test_discovery_uses_myst_notebook_parser(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    _write_source(docs_root, "executable")
    prose = docs_root / "source" / "prose.md"
    prose.write_text(
        "# Prose\n\nThis mentions ``{code-cell}`` but is not a MyST notebook.\n",
        encoding="utf-8",
    )

    sources = discover_sources(docs_root / "source", docs_root)

    assert [source.uri for source in sources] == ["source/executable.md"]


def test_empty_cache_does_not_match_current_source(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    _write_source(docs_root, "page")
    cache_path = docs_root / ".jupyter_cache"
    _initialize_cache(cache_path)

    with pytest.raises(CacheValidationError, match="no matching cache output"):
        validate_cache(
            cache_path,
            source_dir=docs_root / "source",
            docs_root=docs_root,
        )


def test_relocated_cache_uses_relative_uri(tmp_path: Path) -> None:
    original = tmp_path / "original" / "docs"
    _write_source(original, "page")
    source = _source(original, "page")
    _cache_output(original / ".jupyter_cache", source)

    relocated = tmp_path / "relocated" / "docs"
    shutil.copytree(original, relocated)
    report = validate_cache(
        relocated / ".jupyter_cache",
        source_dir=relocated / "source",
        docs_root=relocated,
    )
    cache = get_cache(relocated / ".jupyter_cache")
    try:
        assert cache.list_cache_records()[0].uri == "source/page.md"
    finally:
        close_cache(cache)

    assert report.source_count == 1


def test_validation_rejects_absolute_uri_and_project_rows(tmp_path: Path) -> None:
    absolute_root = tmp_path / "absolute" / "docs"
    _write_source(absolute_root, "page")
    absolute_source = _source(absolute_root, "page")
    absolute_cache = absolute_root / ".jupyter_cache"
    _cache_output(
        absolute_cache,
        absolute_source,
        uri=str(absolute_source.path),
    )
    with pytest.raises(CacheValidationError, match="not canonical"):
        validate_cache(
            absolute_cache,
            source_dir=absolute_root / "source",
            docs_root=absolute_root,
        )

    project_root = tmp_path / "project" / "docs"
    _write_source(project_root, "page")
    project_source = _source(project_root, "page")
    project_cache = project_root / ".jupyter_cache"
    _cache_output(project_cache, project_source)
    project_notebook = tmp_path / "project.ipynb"
    nbformat.write(project_source.notebook, project_notebook)
    cache = get_cache(project_cache)
    try:
        cache.add_nb_to_project(str(project_notebook))
    finally:
        close_cache(cache)
    with pytest.raises(CacheValidationError, match="project row"):
        validate_cache(
            project_cache,
            source_dir=project_root / "source",
            docs_root=project_root,
        )


def test_subset_update_preserves_unrequested_output(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    _write_source(docs_root, "first", code="first = 1")
    _write_source(docs_root, "second", code="second = 2")
    sources = discover_sources(docs_root / "source", docs_root)
    first, second = sources
    snapshot = tmp_path / "snapshot"
    results = tmp_path / "results"
    candidate = tmp_path / "candidate"
    _cache_output(snapshot, first, text="old first\n")
    _cache_output(snapshot, second, text="old second\n")
    _cache_output(results, first, text="new first\n")

    build_candidate(
        sources,
        {first.uri},
        results,
        snapshot,
        candidate,
    )
    validate_cache(
        candidate,
        source_dir=docs_root / "source",
        docs_root=docs_root,
    )

    assert _output_text(candidate, first) == "new first\n"
    assert _output_text(candidate, second) == "old second\n"


def test_shared_source_hash_needs_one_record(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    _write_source(docs_root, "first")
    _write_source(docs_root, "second")
    sources = discover_sources(docs_root / "source", docs_root)
    assert sources[0].hashkey == sources[1].hashkey
    cache_path = docs_root / ".jupyter_cache"
    _cache_output(cache_path, sources[0])

    report = validate_cache(
        cache_path,
        source_dir=docs_root / "source",
        docs_root=docs_root,
    )

    assert report.source_count == 2
    assert report.record_count == 1


def test_bundle_import_preserves_nested_artifacts(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    _write_source(docs_root, "page")
    source = _source(docs_root, "page")
    snapshot = tmp_path / "snapshot"
    candidate = tmp_path / "candidate"
    _cache_output(
        snapshot,
        source,
        artifact_root=tmp_path / "artifact-source",
    )

    build_candidate(
        [source],
        set(),
        tmp_path / "unused",
        snapshot,
        candidate,
    )

    artifact = (
        candidate / "executed" / source.hashkey / "artifacts" / "nested" / "result.txt"
    )
    assert artifact.read_text(encoding="utf-8") == "artifact"


def test_completed_widget_progress_freezes_to_static_html(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    _write_source(docs_root, "page")
    source = _source(docs_root, "page")
    notebook = _widget_progress_notebook(source, value=0.8)

    assert freeze_progress_outputs(notebook, source.uri) == 1

    output = _first_code_cell(notebook).outputs[0]
    assert "widgets" not in notebook.metadata
    assert "application/vnd.jupyter.widget-view+json" not in output.data
    assert '<progress value="1" max="1"' in output.data["text/html"]
    assert "Writing data: 1 / 1 complete" == output.data["text/plain"]
    assert "artifacts/" not in output.data["text/html"]
    assert "a" * 64 not in output.data["text/html"]
    validate_progress_outputs(notebook, source.uri)


def test_incomplete_or_unknown_widget_progress_is_rejected(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    _write_source(docs_root, "page")
    source = _source(docs_root, "page")

    with pytest.raises(ProgressRenderError, match="incomplete"):
        freeze_progress_outputs(
            _widget_progress_notebook(
                source,
                value=9,
                maximum=10,
                bar_style="",
            ),
            source.uri,
        )
    with pytest.raises(ProgressRenderError, match="unsupported root model"):
        freeze_progress_outputs(
            _widget_progress_notebook(source, root_model="ButtonModel"),
            source.uri,
        )


def test_fragmented_terminal_progress_preserves_regular_streams(
    tmp_path: Path,
) -> None:
    docs_root = tmp_path / "docs"
    _write_source(docs_root, "page")
    source = _source(docs_root, "page")
    notebook = _executed_notebook(source)
    _first_code_cell(notebook).outputs = [
        nbformat.v4.new_output(
            "stream",
            name="stderr",
            text="Working: 0%| 0/2 [00:00]\rWork",
        ),
        nbformat.v4.new_output("stream", name="stdout", text="keep this\n"),
        nbformat.v4.new_output(
            "stream",
            name="stderr",
            text="ing: 100%|##| 2/2 [00:01]\n",
        ),
    ]

    assert freeze_progress_outputs(notebook, source.uri) == 1

    outputs = _first_code_cell(notebook).outputs
    assert outputs[0].output_type == "stream"
    assert outputs[0].name == "stdout"
    assert outputs[0].text == "keep this\n"
    assert outputs[1].output_type == "display_data"
    assert outputs[1].data["text/plain"] == "Working: 2 / 2 complete"


def test_terminal_progress_applies_backspace_and_ansi_controls(
    tmp_path: Path,
) -> None:
    docs_root = tmp_path / "docs"
    _write_source(docs_root, "page")
    source = _source(docs_root, "page")
    notebook = _executed_notebook(source)
    _first_code_cell(notebook).outputs = [
        nbformat.v4.new_output(
            "stream",
            name="stderr",
            text="\x1b[32mTask: 10X\b0%|##| 2/2\x1b[0m\n",
        )
    ]

    assert freeze_progress_outputs(notebook, source.uri) == 1
    assert _first_code_cell(notebook).outputs[0].data["text/plain"] == (
        "Task: 2 / 2 complete"
    )


def test_hash_bar_terminal_progress_freezes_to_static_output(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    _write_source(docs_root, "page")
    source = _source(docs_root, "page")
    notebook = _executed_notebook(source)
    _first_code_cell(notebook).outputs = [
        nbformat.v4.new_output(
            "stream",
            name="stderr",
            text="Legacy task [##########] | 100% Completed\n",
        )
    ]

    assert freeze_progress_outputs(notebook, source.uri) == 1
    assert _first_code_cell(notebook).outputs[0].data["text/plain"] == (
        "Legacy task: 100 / 100 complete"
    )


def test_incomplete_terminal_progress_is_rejected(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    _write_source(docs_root, "page")
    source = _source(docs_root, "page")
    notebook = _executed_notebook(source)
    _first_code_cell(notebook).outputs = [
        nbformat.v4.new_output(
            "stream",
            name="stderr",
            text="Working: 75%|###| 3/4\r",
        )
    ]

    with pytest.raises(ProgressRenderError, match="incomplete"):
        freeze_progress_outputs(notebook, source.uri)


def test_progress_validation_rejects_raw_widget_state(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    _write_source(docs_root, "page")
    source = _source(docs_root, "page")

    with pytest.raises(ProgressRenderError, match="widget state remains"):
        validate_progress_outputs(_widget_progress_notebook(source), source.uri)


def test_frozen_progress_survives_modal_cache_transport(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    _write_source(docs_root, "page")
    source = _source(docs_root, "page")
    notebook = _widget_progress_notebook(source)
    freeze_progress_outputs(notebook, source.uri)
    original = tmp_path / "original"
    restored = docs_root / ".jupyter_cache"
    _cache_output(original, source, notebook=notebook)

    restore_page_cache(
        (source.uri, source.hashkey, pack_page_cache(original)),
        restored,
        expected_uri=source.uri,
        expected_hashkey=source.hashkey,
    )

    validate_cache(
        restored,
        source_dir=docs_root / "source",
        docs_root=docs_root,
    )


def test_modal_page_cache_archive_round_trip(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    _write_source(docs_root, "page")
    source = _source(docs_root, "page")
    original = tmp_path / "original"
    restored = docs_root / ".jupyter_cache"
    _cache_output(
        original,
        source,
        artifact_root=tmp_path / "artifact-source",
    )

    payload = (source.uri, source.hashkey, pack_page_cache(original))
    restore_page_cache(
        payload,
        restored,
        expected_uri=source.uri,
        expected_hashkey=source.hashkey,
    )

    validate_cache(
        restored,
        source_dir=docs_root / "source",
        docs_root=docs_root,
    )
    artifact = (
        restored / "executed" / source.hashkey / "artifacts" / "nested" / "result.txt"
    )
    assert artifact.read_text(encoding="utf-8") == "artifact"


def test_modal_page_cache_rejects_mismatched_identity(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    _write_source(docs_root, "page")
    source = _source(docs_root, "page")
    original = tmp_path / "original"
    _cache_output(original, source)
    archive = pack_page_cache(original)

    with pytest.raises(CacheTransportError, match="does not match"):
        restore_page_cache(
            ("source/other.md", source.hashkey, archive),
            tmp_path / "wrong-uri",
            expected_uri=source.uri,
            expected_hashkey=source.hashkey,
        )
    with pytest.raises(CacheTransportError, match="does not match"):
        restore_page_cache(
            (source.uri, "wrong-hash", archive),
            tmp_path / "wrong-hash",
            expected_uri=source.uri,
            expected_hashkey=source.hashkey,
        )

    assert not (tmp_path / "wrong-uri").exists()
    assert not (tmp_path / "wrong-hash").exists()


def test_modal_page_cache_rejects_unsafe_archive_member(tmp_path: Path) -> None:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        member = tarfile.TarInfo("../outside")
        member.size = 1
        archive.addfile(member, io.BytesIO(b"x"))

    with pytest.raises(CacheTransportError, match="unsafe path"):
        restore_page_cache(
            ("source/page.md", "hash", stream.getvalue()),
            tmp_path / "restored",
            expected_uri="source/page.md",
            expected_hashkey="hash",
        )

    assert not (tmp_path / "outside").exists()
    assert not (tmp_path / "restored").exists()


@pytest.mark.parametrize(
    "status",
    ["FAILURE", "INIT_FAILURE", "TERMINATED", "TIMEOUT"],
)
def test_modal_wait_rejects_terminal_failure_status(status: str) -> None:
    class FailedCall:
        object_id = "fc-test"

        def get(self, timeout: float):
            raise TimeoutError

        def get_call_graph(self):
            return [
                SimpleNamespace(
                    function_call_id=self.object_id,
                    status=SimpleNamespace(name=status),
                    children=[],
                )
            ]

    with pytest.raises(RuntimeError, match=f"status={status}"):
        await_page_cache(
            FailedCall(),
            poll_seconds=1,
            deadline_seconds=10,
        )


def test_modal_wait_honors_polling_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_timeouts: list[float] = []

    class PendingCall:
        def get(self, timeout: float):
            observed_timeouts.append(timeout)
            raise TimeoutError

    times = iter([0.0, 0.0, 0.0, 2.0])
    monkeypatch.setattr(modal_cache.time, "monotonic", lambda: next(times))

    with pytest.raises(TimeoutError, match="within 1 seconds"):
        await_page_cache(
            PendingCall(),
            poll_seconds=0.25,
            deadline_seconds=1,
        )

    assert observed_timeouts == [0.25]


@pytest.mark.parametrize("corruption", ["stale", "orphan", "journal"])
def test_validation_detects_stale_and_orphan_state(
    tmp_path: Path,
    corruption: str,
) -> None:
    docs_root = tmp_path / "docs"
    _write_source(docs_root, "page")
    source = _source(docs_root, "page")
    cache_path = docs_root / ".jupyter_cache"
    _cache_output(cache_path, source)

    if corruption == "stale":
        _write_source(docs_root, "stale", code="stale = True")
        stale = _source(docs_root, "stale")
        _cache_output(cache_path, stale)
        (docs_root / "source" / "stale.md").unlink()
    elif corruption == "orphan":
        (cache_path / "executed" / ("f" * 32)).mkdir()
    else:
        (cache_path / "global.db-journal").write_bytes(b"journal")

    with pytest.raises(CacheValidationError):
        validate_cache(
            cache_path,
            source_dir=docs_root / "source",
            docs_root=docs_root,
        )


def test_only_tagged_cells_may_contain_error_outputs(tmp_path: Path) -> None:
    expected_root = tmp_path / "expected" / "docs"
    _write_source(expected_root, "page", raises_exception=True)
    expected = _source(expected_root, "page")
    _cache_output(expected_root / ".jupyter_cache", expected, error=True)
    validate_cache(
        expected_root / ".jupyter_cache",
        source_dir=expected_root / "source",
        docs_root=expected_root,
    )

    unexpected_root = tmp_path / "unexpected" / "docs"
    _write_source(unexpected_root, "page")
    unexpected = _source(unexpected_root, "page")
    _cache_output(unexpected_root / ".jupyter_cache", unexpected, error=True)
    with pytest.raises(CacheValidationError, match="Unexpected error output"):
        validate_cache(
            unexpected_root / ".jupyter_cache",
            source_dir=unexpected_root / "source",
            docs_root=unexpected_root,
        )


def test_validation_rejects_incomplete_execution_counts(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    _write_source(docs_root, "page")
    source = _source(docs_root, "page")
    cache_path = docs_root / ".jupyter_cache"
    _cache_output(cache_path, source)
    notebook_path = cache_path / "executed" / source.hashkey / "base.ipynb"
    notebook = nbformat.read(notebook_path, nbformat.NO_CONVERT)
    notebook.cells[0].execution_count = None
    nbformat.write(notebook, notebook_path)

    with pytest.raises(CacheValidationError, match="execution counts are incomplete"):
        validate_cache(
            cache_path,
            source_dir=docs_root / "source",
            docs_root=docs_root,
        )


def test_execution_failure_preserves_target(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    _write_source(docs_root, "page")
    source = _source(docs_root, "page")
    target = docs_root / ".jupyter_cache"
    _cache_output(target, source, text="old\n")
    before = _tree_bytes(target)

    def fail_runner(source, cache_path):
        raise RuntimeError("execution failed")

    with pytest.raises(ExecutionBatchError):
        execute_and_publish(
            ["page"],
            docs_root=docs_root,
            page_runner=fail_runner,
        )

    assert _tree_bytes(target) == before


def test_import_failure_preserves_target(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    _write_source(docs_root, "page")
    source = _source(docs_root, "page")
    target = docs_root / ".jupyter_cache"
    _cache_output(target, source, text="old\n")
    before = _tree_bytes(target)

    def empty_runner(source, cache_path):
        _initialize_cache(cache_path)
        return cache_path

    with pytest.raises(ExecutionBatchError):
        execute_and_publish(
            ["page"],
            docs_root=docs_root,
            page_runner=empty_runner,
        )

    assert _tree_bytes(target) == before


def test_candidate_validation_failure_preserves_target(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    _write_source(docs_root, "page")
    source = _source(docs_root, "page")
    target = docs_root / ".jupyter_cache"
    _cache_output(target, source, text="old\n")
    before = _tree_bytes(target)

    def invalid_runner(source, cache_path):
        _cache_output(cache_path, source, error=True)
        return cache_path

    with pytest.raises(CacheValidationError):
        execute_and_publish(
            ["page"],
            docs_root=docs_root,
            page_runner=invalid_runner,
        )

    assert _tree_bytes(target) == before


def test_publication_failure_restores_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs_root = tmp_path / "docs"
    _write_source(docs_root, "page")
    source = _source(docs_root, "page")
    target = docs_root / ".jupyter_cache"
    candidate = docs_root / ".jupyter_cache.candidate-test"
    _cache_output(target, source, text="old\n")
    _cache_output(candidate, source, text="new\n")
    before = _tree_bytes(target)
    original_rename = cache_tools._rename_path

    def fail_candidate_rename(source_path: Path, destination: Path) -> None:
        if source_path == candidate and destination == target:
            raise OSError("replacement failed")
        original_rename(source_path, destination)

    monkeypatch.setattr(cache_tools, "_rename_path", fail_candidate_rename)

    with pytest.raises(OSError, match="replacement failed"):
        publish_candidate(candidate, target)

    assert _tree_bytes(target) == before
    assert not cache_tools.backup_path(target).exists()


def test_next_run_recovers_interrupted_swap(tmp_path: Path) -> None:
    target = tmp_path / ".jupyter_cache"
    target.mkdir()
    (target / "sentinel").write_text("old", encoding="utf-8")
    backup = cache_tools.backup_path(target)
    cache_tools._rename_path(target, backup)

    assert cache_tools.recover_interrupted_swap(target)
    assert (target / "sentinel").read_text(encoding="utf-8") == "old"
    assert not backup.exists()


def test_resume_invalidates_source_and_execution_fingerprints(
    tmp_path: Path,
) -> None:
    docs_root = tmp_path / "docs"
    _write_source(docs_root, "page")
    source = _source(docs_root, "page")
    resume_dir = docs_root / ".jupyter_cache.resume"
    page_cache = tmp_path / "page-cache"
    _cache_output(page_cache, source)
    manifest = _prepare_resume(
        resume_dir,
        "fingerprint-one",
        {source.uri},
        use_resume=False,
        runner_identity="local",
    )
    _record_result(source, page_cache, resume_dir, manifest)
    assert _valid_resume_uris(
        [source],
        {source.uri},
        resume_dir,
        manifest,
    ) == {source.uri}

    _write_source(docs_root, "page", code="value = 2")
    changed = _source(docs_root, "page")
    assert not _valid_resume_uris(
        [changed],
        {changed.uri},
        resume_dir,
        manifest,
    )

    replaced = _prepare_resume(
        resume_dir,
        "fingerprint-two",
        {changed.uri},
        use_resume=True,
        runner_identity="local",
    )
    assert replaced["entries"] == {}
    assert not (resume_dir / "cache").exists()


def test_resume_invalidates_runner_identity(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    _write_source(docs_root, "page")
    source = _source(docs_root, "page")
    resume_dir = docs_root / ".jupyter_cache.resume"
    page_cache = tmp_path / "page-cache"
    _cache_output(page_cache, source)
    manifest = _prepare_resume(
        resume_dir,
        "shared-fingerprint",
        {source.uri},
        use_resume=False,
        runner_identity="local",
    )
    _record_result(source, page_cache, resume_dir, manifest)

    replaced = _prepare_resume(
        resume_dir,
        "shared-fingerprint",
        {source.uri},
        use_resume=True,
        runner_identity="modal-python-3.14",
    )

    assert replaced["runnerIdentity"] == "modal-python-3.14"
    assert replaced["entries"] == {}
    assert not (resume_dir / "cache").exists()


@pytest.mark.parametrize("filename", ["modal_cache.py", "modal_docs.py"])
def test_modal_runner_files_participate_in_execution_fingerprint(
    tmp_path: Path,
    filename: str,
) -> None:
    repo_root = tmp_path / "repo"
    docs_root = repo_root / "docs"
    docs_root.mkdir(parents=True)
    for runner_file in ("modal_cache.py", "modal_docs.py"):
        (docs_root / runner_file).write_text("original\n", encoding="utf-8")
    before = execution_fingerprint(repo_root, docs_root)

    (docs_root / filename).write_text("changed\n", encoding="utf-8")

    assert execution_fingerprint(repo_root, docs_root) != before


def test_explicit_resume_reuses_matching_successes(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    _write_source(docs_root, "first", code="first = 1")
    _write_source(docs_root, "second", code="second = 2")
    first, second = discover_sources(docs_root / "source", docs_root)
    target = docs_root / ".jupyter_cache"
    _cache_output(target, first, text="old first\n")
    _cache_output(target, second, text="old second\n")

    def interrupted_runner(source, cache_path):
        if source.uri == second.uri:
            raise RuntimeError("second failed")
        _cache_output(cache_path, source, text="new first\n")
        return cache_path

    with pytest.raises(ExecutionBatchError):
        execute_and_publish(
            ["first", "second"],
            docs_root=docs_root,
            page_runner=interrupted_runner,
        )

    resumed_calls: list[str] = []

    def resumed_runner(source, cache_path):
        resumed_calls.append(source.uri)
        _cache_output(cache_path, source, text="new second\n")
        return cache_path

    execute_and_publish(
        ["first", "second"],
        docs_root=docs_root,
        resume=True,
        page_runner=resumed_runner,
    )

    assert resumed_calls == [second.uri]
    assert _output_text(target, first) == "new first\n"
    assert _output_text(target, second) == "new second\n"


def test_modal_fanout_spawns_before_wait_and_resumes_failures(
    tmp_path: Path,
) -> None:
    docs_root = tmp_path / "docs"
    _write_source(docs_root, "first", code="first = 1")
    _write_source(docs_root, "second", code="second = 2")
    first, second = discover_sources(docs_root / "source", docs_root)
    target = docs_root / ".jupyter_cache"
    _cache_output(target, first, text="old first\n")
    _cache_output(target, second, text="old second\n")
    before = _tree_bytes(target)

    first_cache = tmp_path / "first-cache"
    second_cache = tmp_path / "second-cache"
    _cache_output(first_cache, first, text="new first\n")
    _cache_output(second_cache, second, text="new second\n")
    payloads = {
        first.uri: (first.uri, first.hashkey, pack_page_cache(first_cache)),
        second.uri: (second.uri, second.hashkey, pack_page_cache(second_cache)),
    }

    class FakeCall:
        def __init__(
            self,
            outcome,
            spawned: list[str],
            observed_spawn_counts: list[int],
        ) -> None:
            self.outcome = outcome
            self.spawned = spawned
            self.observed_spawn_counts = observed_spawn_counts
            self.cancelled = False

        def get(self, timeout: float):
            assert timeout > 0
            self.observed_spawn_counts.append(len(self.spawned))
            if isinstance(self.outcome, BaseException):
                raise self.outcome
            return self.outcome

        def cancel(self) -> None:
            self.cancelled = True

    first_run_spawned: list[str] = []
    observed_spawn_counts: list[int] = []

    def first_run_spawn(source):
        first_run_spawned.append(source.uri)
        outcome = (
            RuntimeError("second failed")
            if source.uri == second.uri
            else payloads[source.uri]
        )
        return FakeCall(outcome, first_run_spawned, observed_spawn_counts)

    first_launcher = SpawnedPageRunner(first_run_spawn)
    try:
        with pytest.raises(ExecutionBatchError):
            execute_and_publish(
                ["first", "second"],
                jobs=2,
                docs_root=docs_root,
                page_runner_factory=first_launcher.prepare,
                warn_parallel_memory=False,
            )
    finally:
        first_launcher.cancel_unclaimed()

    assert first_run_spawned == [first.uri, second.uri]
    assert observed_spawn_counts == [2, 2]
    assert _tree_bytes(target) == before

    resumed_spawned: list[str] = []
    resumed_observed_counts: list[int] = []

    def resumed_spawn(source):
        resumed_spawned.append(source.uri)
        return FakeCall(
            payloads[source.uri],
            resumed_spawned,
            resumed_observed_counts,
        )

    resumed_launcher = SpawnedPageRunner(resumed_spawn)
    try:
        execute_and_publish(
            ["first", "second"],
            jobs=2,
            resume=True,
            docs_root=docs_root,
            page_runner_factory=resumed_launcher.prepare,
            warn_parallel_memory=False,
        )
    finally:
        resumed_launcher.cancel_unclaimed()

    assert resumed_spawned == [second.uri]
    assert resumed_observed_counts == [1]
    assert _output_text(target, first) == "new first\n"
    assert _output_text(target, second) == "new second\n"


def test_modal_submission_failure_cancels_submitted_calls(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    _write_source(docs_root, "first", code="first = 1")
    _write_source(docs_root, "second", code="second = 2")
    first, second = discover_sources(docs_root / "source", docs_root)

    class FakeCall:
        def __init__(self) -> None:
            self.cancelled = False

        def cancel(self) -> None:
            self.cancelled = True

    first_call = FakeCall()

    def spawn(source):
        if source.uri == second.uri:
            raise RuntimeError("submission failed")
        return first_call

    launcher = SpawnedPageRunner(spawn)
    with pytest.raises(RuntimeError, match="submission failed"):
        launcher.prepare([first, second])

    assert first_call.cancelled


def test_resume_without_a_staged_run_fails(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    _write_source(docs_root, "page")

    with pytest.raises(CacheBuildError, match="No resumable documentation run"):
        execute_and_publish(
            resume=True,
            docs_root=docs_root,
            page_runner=lambda source, cache_path: cache_path,
        )


def test_partial_run_fails_when_unrequested_source_has_no_match(
    tmp_path: Path,
) -> None:
    docs_root = tmp_path / "docs"
    _write_source(docs_root, "first", code="first = 1")
    _write_source(docs_root, "second", code="second = 2")
    first, second = discover_sources(docs_root / "source", docs_root)
    target = docs_root / ".jupyter_cache"
    _cache_output(target, first, text="old\n")
    before = _tree_bytes(target)

    def runner(source, cache_path):
        _cache_output(cache_path, source, text="new\n")
        return cache_path

    with pytest.raises(CacheBuildError, match=second.uri):
        execute_and_publish(
            ["first"],
            docs_root=docs_root,
            page_runner=runner,
        )

    assert _tree_bytes(target) == before


def test_prune_publishes_only_matching_current_sources(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    _write_source(docs_root, "current", code="current = True")
    _write_source(docs_root, "stale", code="stale = True")
    current, stale = discover_sources(docs_root / "source", docs_root)
    target = docs_root / ".jupyter_cache"
    _cache_output(target, current)
    _cache_output(target, stale)
    (docs_root / "source" / "stale.md").unlink()
    (target / "executed" / ("f" * 32)).mkdir()

    report = prune_and_publish(docs_root)

    assert report.source_count == 1
    assert report.record_count == 1
    validate_cache(
        target,
        source_dir=docs_root / "source",
        docs_root=docs_root,
    )


def test_full_run_replaces_invalid_old_cache(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    _write_source(docs_root, "page")
    source = _source(docs_root, "page")
    target = docs_root / ".jupyter_cache"
    _cache_output(target, source, text="old\n")
    (target / "executed" / ("f" * 32)).mkdir()
    calls: list[str] = []

    def runner(source, cache_path):
        calls.append(source.uri)
        _cache_output(cache_path, source, text="new\n")
        return cache_path

    execute_and_publish(
        full=True,
        docs_root=docs_root,
        page_runner=runner,
    )

    assert calls == [source.uri]
    assert _output_text(target, source) == "new\n"
    validate_cache(
        target,
        source_dir=docs_root / "source",
        docs_root=docs_root,
    )


def test_validation_does_not_change_cache_bytes(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    _write_source(docs_root, "page")
    source = _source(docs_root, "page")
    cache_path = docs_root / ".jupyter_cache"
    _cache_output(cache_path, source)
    before = _tree_bytes(cache_path)

    validate_cache(
        cache_path,
        source_dir=docs_root / "source",
        docs_root=docs_root,
    )

    assert _tree_bytes(cache_path) == before

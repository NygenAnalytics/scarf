import json
import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.io import mmread

import scarf.embeddings.sgtsne as sgtsne_module
from tests.test_utils_process import _null_descriptors, _standard_descriptors


def _graph() -> csr_matrix:
    return csr_matrix(
        ([1.0, 1.0, 1.0, 1.0, 0.0], ([0, 1, 1, 2, 0], [1, 0, 2, 1, 2])),
        shape=(3, 3),
    )


def test_run_sgtsne_validates_initial_embedding_shape():
    with pytest.raises(ValueError, match=r"must have shape \(3, 2\)"):
        sgtsne_module.run_sgtsne(
            _graph(),
            np.zeros((3, 3)),
            tsne_dims=2,
        )


_FAKE_SGTSNE = """
import json
import os
import shutil
import sys
from pathlib import Path

arguments = sys.argv[1:]
record = Path(os.environ["SCARF_FAKE_SGTSNE_RECORD"])
record.mkdir(parents=True, exist_ok=True)
(record / "argv.json").write_text(json.dumps(arguments), encoding="utf-8")
shutil.copyfile(arguments[-1], record / "graph.mtx")
shutil.copyfile(arguments[arguments.index("-i") + 1], record / "initial.txt")
print("fake sgtsne progress", flush=True)
mode = os.environ.get("SCARF_FAKE_SGTSNE_MODE", "ok")
if mode == "fail":
    sys.stderr.write("\\n".join(f"diagnostic {index}" for index in range(30)))
    sys.exit(3)
if mode == "ok":
    Path(arguments[arguments.index("-o") + 1]).write_text(
        "1 10\\n2 20\\n3 30\\n",
        encoding="utf-8",
    )
"""


def _install_fake_sgtsne(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    mode: str = "ok",
) -> Path:
    """Put a real ``sgtsne`` executable that records its argv on PATH."""
    bin_dir = tmp_path / "fake bin"
    bin_dir.mkdir()
    script = bin_dir / "fake_sgtsne.py"
    script.write_text(_FAKE_SGTSNE, encoding="utf-8")
    executable = bin_dir / "sgtsne"
    executable.write_text(
        f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n',
        encoding="utf-8",
    )
    executable.chmod(0o755)
    record = tmp_path / "record"
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setenv("SCARF_FAKE_SGTSNE_RECORD", str(record))
    monkeypatch.setenv("SCARF_FAKE_SGTSNE_MODE", mode)
    return record


@pytest.mark.parametrize("sparse_format", ["csr", "coo"])
@pytest.mark.parametrize(
    ("verbose", "parallel", "expected_threads"),
    [
        (True, True, 4),
        (False, False, 1),
    ],
)
def test_run_sgtsne_cli_backend_passes_argv_and_cleans_temporary_files(
    monkeypatch,
    tmp_path,
    verbose,
    parallel,
    expected_threads,
    sparse_format,
):
    graph = _graph().asformat(sparse_format)
    record = _install_fake_sgtsne(monkeypatch, tmp_path)
    work_dir = tmp_path / "work dir; $(touch injected)"
    work_dir.mkdir()
    monkeypatch.setattr(sgtsne_module, "uuid4", lambda: "fixed")
    logged: list[str] = []
    sink = sgtsne_module.logger.add(
        lambda message: logged.append(message.record["message"]),
        level="DEBUG",
    )
    try:
        embedding = sgtsne_module.run_sgtsne(
            graph,
            np.arange(6),
            tsne_dims=2,
            max_iter=11,
            early_iter=3,
            alpha=7,
            lambda_scale=0.5,
            box_h=0.2,
            temp_file_loc=str(work_dir),
            verbose=verbose,
            parallel=parallel,
            nthreads=4,
        )
    finally:
        sgtsne_module.logger.remove(sink)

    arguments = json.loads((record / "argv.json").read_text(encoding="utf-8"))
    assert arguments == [
        "-m",
        "11",
        "-l",
        "0.5",
        "-d",
        "2",
        "-e",
        "3",
        "-p",
        str(expected_threads),
        "-a",
        "7",
        "-h",
        "0.2",
        "-i",
        str((work_dir / "fixed.txt").resolve()),
        "-o",
        str((work_dir / "fixed_output.txt").resolve()),
        str((work_dir / "fixed.mtx").resolve()),
    ]
    exported = mmread(record / "graph.mtx")
    assert exported.nnz == 4
    assert np.all(exported.data > 0)
    np.testing.assert_array_equal(exported.toarray(), graph.toarray())
    assert graph.nnz == 5
    assert (record / "initial.txt").read_text(encoding="utf-8") == ("0\n1\n2\n3\n4\n5")
    np.testing.assert_array_equal(
        embedding,
        np.array([[1, 2, 3], [10, 20, 30]]),
    )
    assert ("fake sgtsne progress" in logged) is verbose
    assert list(work_dir.iterdir()) == []
    assert not (tmp_path / "injected").exists()


def test_run_sgtsne_cli_backend_raises_on_nonzero_exit(monkeypatch, tmp_path):
    _install_fake_sgtsne(monkeypatch, tmp_path, mode="fail")
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    with pytest.raises(RuntimeError, match="exited with status 3") as caught:
        sgtsne_module.run_sgtsne(
            _graph(),
            np.zeros((3, 2)),
            temp_file_loc=str(work_dir),
            verbose=False,
        )

    message = str(caught.value)
    assert "diagnostic 29" in message
    assert "diagnostic 10" in message
    assert "diagnostic 9\n" not in message
    assert list(work_dir.iterdir()) == []


def test_run_sgtsne_cli_backend_cleans_inputs_when_output_is_missing(
    monkeypatch,
    tmp_path,
):
    _install_fake_sgtsne(monkeypatch, tmp_path, mode="no-output")
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        sgtsne_module.run_sgtsne(
            _graph(),
            np.zeros((3, 2)),
            temp_file_loc=str(work_dir),
        )

    assert list(work_dir.iterdir()) == []


@pytest.mark.parametrize("sparse_format", ["csr", "coo"])
@pytest.mark.parametrize(
    ("parallel", "expected_warnings"),
    [
        (
            True,
            [
                "parallel=True is not supported by the sgtsnepi Python backend; "
                "running single-threaded"
            ],
        ),
        (False, []),
    ],
)
@pytest.mark.parametrize("verbose", [True, False])
def test_run_sgtsne_python_backend_forwards_parameters(
    monkeypatch,
    parallel,
    expected_warnings,
    sparse_format,
    verbose,
):
    graph = _graph().asformat(sparse_format)
    initial = np.arange(6, dtype=np.float64).reshape(3, 2)
    captured = {}
    warnings = []
    fake_module = types.ModuleType("sgtsnepi")
    before = _standard_descriptors()

    def fake_sgtsnepi(received_graph, **kwargs):
        captured["graph"] = received_graph
        captured["kwargs"] = kwargs
        captured["descriptors"] = _standard_descriptors()
        return [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    fake_module.sgtsnepi = fake_sgtsnepi
    monkeypatch.setattr(sgtsne_module.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        sgtsne_module,
        "logger",
        SimpleNamespace(warning=warnings.append),
    )
    monkeypatch.setitem(sys.modules, "sgtsnepi", fake_module)

    embedding = sgtsne_module.run_sgtsne(
        graph,
        initial,
        tsne_dims=2,
        max_iter=17,
        early_iter=5,
        alpha=8,
        lambda_scale=0.25,
        box_h=0.4,
        verbose=verbose,
        parallel=parallel,
        nthreads=12,
    )

    assert captured["graph"].nnz == 4
    assert np.all(captured["graph"].data > 0)
    np.testing.assert_array_equal(captured["graph"].toarray(), graph.toarray())
    assert graph.nnz == 5
    assert captured["kwargs"] == {
        "y0": pytest.approx(initial.T),
        "d": 2,
        "max_iter": 17,
        "early_exag": 5,
        "lambda_par": 0.25,
        "h": 0.4,
        "alpha": 8,
        "silent": False,
    }
    assert captured["descriptors"] == (before if verbose else _null_descriptors())
    assert _standard_descriptors() == before
    np.testing.assert_array_equal(
        embedding,
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    )
    assert warnings == expected_warnings


_QUIET_SGTSNEPI_PROBE = textwrap.dedent(
    """
    import json
    import os
    import sys

    import numpy as np
    from scipy.sparse import coo_matrix

    from scarf.embeddings.sgtsne import run_sgtsne


    def identity(fd):
        status = os.fstat(fd)
        return [status.st_dev, status.st_ino]


    n_cells = 64
    rows = np.repeat(np.arange(n_cells), 2)
    cols = np.stack(
        [(np.arange(n_cells) - 1) % n_cells, (np.arange(n_cells) + 1) % n_cells],
        axis=1,
    ).ravel()
    graph = coo_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(n_cells, n_cells),
    )
    initial = np.random.default_rng(0).normal(size=(n_cells, 2))
    before = {fd: identity(fd) for fd in (1, 2)}
    embedding = run_sgtsne(
        graph,
        initial,
        max_iter=20,
        early_iter=5,
        verbose=False,
    )
    after = {fd: identity(fd) for fd in (1, 2)}
    with open(os.devnull, "w") as later:
        later_fd = later.fileno()
    print("probe-stdout", flush=True)
    sys.stderr.write("probe-stderr\\n")
    sys.stderr.flush()
    with open(sys.argv[1], "w", encoding="utf-8") as handle:
        json.dump(
            {
                "before": before,
                "after": after,
                "later_fd": later_fd,
                "shape": list(embedding.shape),
                "finite": bool(np.isfinite(embedding).all()),
            },
            handle,
        )
    """
)


@pytest.mark.slow
def test_quiet_sgtsnepi_backend_keeps_standard_descriptors(tmp_path):
    pytest.importorskip("sgtsnepi")
    probe = tmp_path / "probe.py"
    probe.write_text(_QUIET_SGTSNEPI_PROBE, encoding="utf-8")
    result_path = tmp_path / "result.json"
    empty_path = tmp_path / "empty-path"
    empty_path.mkdir()
    environment = {**os.environ, "PATH": str(empty_path)}

    completed = subprocess.run(
        [sys.executable, str(probe), str(result_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        timeout=300,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["after"] == result["before"]
    assert result["later_fd"] > 2
    assert result["shape"] == [2, 64]
    assert result["finite"] is True
    assert completed.stdout == "probe-stdout\n"
    assert completed.stderr.endswith("probe-stderr\n")
    assert "Number of vertices" not in completed.stderr


def test_run_sgtsne_requires_an_available_backend(monkeypatch):
    monkeypatch.setattr(sgtsne_module.shutil, "which", lambda _name: None)
    monkeypatch.setitem(sys.modules, "sgtsnepi", None)

    with pytest.raises(ImportError, match="executable on PATH or the sgtsnepi package"):
        sgtsne_module.run_sgtsne(
            csr_matrix((1, 1), dtype=np.float64),
            np.zeros((1, 2)),
        )

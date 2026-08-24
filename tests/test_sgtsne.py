import shlex
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import scarf.embeddings.sgtsne as sgtsne_module


def _graph() -> csr_matrix:
    return csr_matrix(
        np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
    )


def test_run_sgtsne_validates_initial_embedding_shape():
    with pytest.raises(ValueError, match=r"must have shape \(3, 2\)"):
        sgtsne_module.run_sgtsne(
            _graph(),
            np.zeros((3, 3)),
            tsne_dims=2,
        )


@pytest.mark.parametrize(
    ("verbose", "parallel", "expected_threads", "expected_runner"),
    [
        (True, True, 4, "system_call"),
        (False, False, 1, "os.system"),
    ],
)
def test_run_sgtsne_cli_backend_builds_command_and_cleans_temporary_files(
    monkeypatch,
    tmp_path,
    verbose,
    parallel,
    expected_threads,
    expected_runner,
):
    graph = _graph()
    captured = {}

    def fake_export(path, received_graph):
        captured["graph"] = received_graph
        Path(path).write_text("mock matrix", encoding="utf-8")

    def execute(command, runner):
        captured["command"] = command
        captured["runner"] = runner
        arguments = shlex.split(command)
        initial_path = Path(arguments[arguments.index("-i") + 1])
        output_path = Path(arguments[arguments.index("-o") + 1])
        captured["initial_values"] = initial_path.read_text(encoding="utf-8")
        output_path.write_text("1 10\n2 20\n3 30\n", encoding="utf-8")

    monkeypatch.setattr(sgtsne_module.shutil, "which", lambda _name: "/mock/sgtsne")
    monkeypatch.setattr(sgtsne_module, "uuid4", lambda: "fixed")
    monkeypatch.setattr(sgtsne_module, "export_knn_to_mtx", fake_export)
    monkeypatch.setattr(
        sgtsne_module,
        "system_call",
        lambda command: execute(command, "system_call"),
    )
    monkeypatch.setattr(
        sgtsne_module.os,
        "system",
        lambda command: execute(command, "os.system"),
    )

    embedding = sgtsne_module.run_sgtsne(
        graph,
        np.arange(6),
        tsne_dims=2,
        max_iter=11,
        early_iter=3,
        alpha=7,
        lambda_scale=0.5,
        box_h=0.2,
        temp_file_loc=str(tmp_path),
        verbose=verbose,
        parallel=parallel,
        nthreads=4,
    )

    arguments = shlex.split(captured["command"])
    assert arguments == [
        "sgtsne",
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
        str((tmp_path / "fixed.txt").resolve()),
        "-o",
        str((tmp_path / "fixed_output.txt").resolve()),
        str((tmp_path / "fixed.mtx").resolve()),
    ]
    assert captured["runner"] == expected_runner
    assert captured["graph"] is graph
    assert captured["initial_values"] == "0\n1\n2\n3\n4\n5"
    np.testing.assert_array_equal(
        embedding,
        np.array([[1, 2, 3], [10, 20, 30]]),
    )
    assert list(tmp_path.iterdir()) == []


def test_run_sgtsne_cli_backend_cleans_inputs_when_output_is_missing(
    monkeypatch,
    tmp_path,
):
    def fake_export(path, _graph):
        Path(path).write_text("mock matrix", encoding="utf-8")

    monkeypatch.setattr(sgtsne_module.shutil, "which", lambda _name: "/mock/sgtsne")
    monkeypatch.setattr(sgtsne_module, "uuid4", lambda: "failed")
    monkeypatch.setattr(sgtsne_module, "export_knn_to_mtx", fake_export)
    monkeypatch.setattr(sgtsne_module, "system_call", lambda _command: None)

    with pytest.raises(FileNotFoundError):
        sgtsne_module.run_sgtsne(
            _graph(),
            np.zeros((3, 2)),
            temp_file_loc=str(tmp_path),
        )

    assert list(tmp_path.iterdir()) == []


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
def test_run_sgtsne_python_backend_forwards_parameters(
    monkeypatch,
    parallel,
    expected_warnings,
):
    graph = _graph()
    initial = np.arange(6, dtype=np.float64).reshape(3, 2)
    captured = {}
    warnings = []
    fake_module = types.ModuleType("sgtsnepi")

    def fake_sgtsnepi(received_graph, **kwargs):
        captured["graph"] = received_graph
        captured["kwargs"] = kwargs
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
        verbose=False,
        parallel=parallel,
        nthreads=12,
    )

    assert captured["graph"] is graph
    assert captured["kwargs"] == {
        "y0": pytest.approx(initial.T),
        "d": 2,
        "max_iter": 17,
        "early_exag": 5,
        "lambda_par": 0.25,
        "h": 0.4,
        "alpha": 8,
        "silent": True,
    }
    np.testing.assert_array_equal(
        embedding,
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    )
    assert warnings == expected_warnings


def test_run_sgtsne_requires_an_available_backend(monkeypatch):
    monkeypatch.setattr(sgtsne_module.shutil, "which", lambda _name: None)
    monkeypatch.setitem(sys.modules, "sgtsnepi", None)

    with pytest.raises(ImportError, match="executable on PATH or the sgtsnepi package"):
        sgtsne_module.run_sgtsne(
            csr_matrix((1, 1), dtype=np.float64),
            np.zeros((1, 2)),
        )

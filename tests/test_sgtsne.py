import json
import os
import shutil
import subprocess
import sys
import textwrap
import types

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import scarf.embeddings.sgtsne as sgtsne_module
from tests.test_utils_process import _null_descriptors, _standard_descriptors


def _graph() -> csr_matrix:
    return csr_matrix(
        ([1.0, 1.0, 1.0, 1.0, 0.0], ([0, 1, 1, 2, 0], [1, 0, 2, 1, 2])),
        shape=(3, 3),
    )


@pytest.mark.parametrize("initial", [np.zeros((3, 3)), np.zeros(6)])
def test_run_sgtsne_validates_initial_embedding_shape(initial):
    with pytest.raises(ValueError, match=r"must have shape \(3, 2\)"):
        sgtsne_module.run_sgtsne(
            _graph(),
            initial,
            tsne_dims=2,
        )


def _fake_sgtsnepi(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[dict[str, object]],
) -> None:
    """Install an ``sgtsnepi`` module that records its calls."""
    fake_module = types.ModuleType("sgtsnepi")

    def fake_sgtsnepi(received_graph, **kwargs):
        calls.append(
            {
                "graph": received_graph,
                "kwargs": kwargs,
                "descriptors": _standard_descriptors(),
            }
        )
        return np.arange(2 * received_graph.shape[0], dtype=np.float64).reshape(
            2, received_graph.shape[0]
        )

    fake_module.sgtsnepi = fake_sgtsnepi
    monkeypatch.setitem(sys.modules, "sgtsnepi", fake_module)


@pytest.mark.parametrize("sparse_format", ["csr", "coo"])
@pytest.mark.parametrize("verbose", [True, False])
def test_run_sgtsne_forwards_parameters_to_sgtsnepi(
    monkeypatch,
    sparse_format,
    verbose,
):
    graph = _graph().asformat(sparse_format)
    initial = np.arange(6, dtype=np.float64).reshape(3, 2)
    calls: list[dict[str, object]] = []
    before = _standard_descriptors()
    _fake_sgtsnepi(monkeypatch, calls)

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
    )

    (call,) = calls
    received = call["graph"]
    assert received.nnz == 4
    assert np.all(received.data > 0)
    np.testing.assert_array_equal(received.toarray(), graph.toarray())
    assert graph.nnz == 5
    assert call["kwargs"] == {
        "y0": pytest.approx(initial.T),
        "d": 2,
        "max_iter": 17,
        "early_exag": 5,
        "lambda_par": 0.25,
        "h": 0.4,
        "alpha": 8,
        "silent": False,
    }
    assert call["descriptors"] == (before if verbose else _null_descriptors())
    assert _standard_descriptors() == before
    np.testing.assert_array_equal(
        embedding,
        np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]),
    )


def test_run_sgtsne_ignores_an_sgtsne_executable_on_path(monkeypatch, tmp_path):
    # An earlier release preferred any sgtsne executable on PATH over sgtsnepi
    # and recorded nothing about which one ran.
    marker = tmp_path / "executable-ran"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable = bin_dir / "sgtsne"
    # A shell redirection marks the run, since PATH holds no touch command.
    executable.write_text(f'#!/bin/sh\n: > "{marker}"\nexit 3\n', encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    assert shutil.which("sgtsne") == str(executable)
    calls: list[dict[str, object]] = []
    _fake_sgtsnepi(monkeypatch, calls)

    embedding = sgtsne_module.run_sgtsne(_graph(), np.zeros((3, 2)), verbose=False)

    assert len(calls) == 1
    assert embedding.shape == (2, 3)
    assert not marker.exists()


_SETTINGS = {
    "tsne_dims": 2,
    "lambda_scale": 1.0,
    "max_iter": 500,
    "early_iter": 200,
    "alpha": 10,
    "box_h": 0.7,
}
# Each setting's bound. The shared argument validators have their own tests of
# types and finiteness.
_INVALID_SETTINGS = [
    ({"tsne_dims": 0}, ValueError, "tsne_dims must be at least 1"),
    ({"max_iter": 0}, ValueError, "max_iter must be at least 1"),
    ({"max_iter": True}, TypeError, "max_iter must be an integer"),
    ({"early_iter": -1}, ValueError, "early_iter must be at least 0"),
    ({"alpha": 0}, ValueError, "alpha must be at least 1"),
    ({"lambda_scale": 0.0}, ValueError, "lambda_scale must be positive"),
    ({"box_h": 0}, ValueError, "box_h must be positive"),
]


@pytest.mark.parametrize(("change", "error", "message"), _INVALID_SETTINGS)
def test_sgtsne_settings_reject_invalid_values(change, error, message):
    with pytest.raises(error, match=message):
        sgtsne_module.sgtsne_settings(**{**_SETTINGS, **change})


def test_sgtsne_settings_are_canonical_python_numbers():
    settings = sgtsne_module.sgtsne_settings(
        tsne_dims=np.int64(3),
        lambda_scale=1,
        max_iter=np.int32(20),
        early_iter=0,
        alpha=np.uint8(1),
        box_h=np.float32(0.5),
    )

    assert settings == sgtsne_module.SgtsneSettings(
        tsne_dims=3,
        lambda_scale=1.0,
        max_iter=20,
        early_iter=0,
        alpha=1,
        box_h=0.5,
    )
    for name in ("tsne_dims", "max_iter", "early_iter", "alpha"):
        assert type(getattr(settings, name)) is int
    for name in ("lambda_scale", "box_h"):
        assert type(getattr(settings, name)) is float


def test_run_sgtsne_validates_settings_before_calling_the_backend(monkeypatch):
    calls: list[dict[str, object]] = []
    _fake_sgtsnepi(monkeypatch, calls)

    with pytest.raises(ValueError, match="max_iter must be at least 1"):
        sgtsne_module.run_sgtsne(
            _graph(), np.zeros((3, 2)), **{**_SETTINGS, "max_iter": 0}
        )
    assert calls == []


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

    completed = subprocess.run(
        [sys.executable, str(probe), str(result_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=dict(os.environ),
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


def test_quiet_sgtsnepi_backend_suppresses_notebook_streams(capsys):
    pytest.importorskip("sgtsnepi")
    n_cells = 40
    rows = np.repeat(np.arange(n_cells), 2)
    columns = np.column_stack(
        ((np.arange(n_cells) + 1) % n_cells, (np.arange(n_cells) - 1) % n_cells)
    ).ravel()
    graph = csr_matrix((np.ones(2 * n_cells) / 2, (rows, columns)))
    initial = np.random.default_rng(0).normal(scale=1e-4, size=(n_cells, 2))
    streams = sys.stdout, sys.stderr

    embedding = sgtsne_module.run_sgtsne(
        graph, initial, max_iter=10, early_iter=5, verbose=False
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert (sys.stdout, sys.stderr) == streams
    assert embedding.shape == (2, n_cells)
    assert np.isfinite(embedding).all()


def test_run_sgtsne_names_the_tsne_extra_when_sgtsnepi_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "sgtsnepi", None)

    with pytest.raises(ImportError) as caught:
        sgtsne_module.run_sgtsne(
            csr_matrix((1, 1), dtype=np.float64),
            np.zeros((1, 2)),
        )

    message = str(caught.value)
    assert message == sgtsne_module.SGTSNEPI_GUIDANCE
    assert 'pip install "scarf[tsne]"' in message
    assert "Linux x86_64" in message
    assert "macOS 26 or newer on arm64" in message
    assert isinstance(caught.value.__cause__, ImportError)

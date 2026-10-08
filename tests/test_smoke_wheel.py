import re
import sys
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import pytest

import tests.smoke_wheel as smoke_wheel
from tests.smoke_wheel import _REQUIRED_MODULES, validate_wheel_contents
from tests.smoke_workflow import run_workflow


_DIST_INFO = "scarf-1.0.0.dist-info"
_PURE_WHEEL = (
    "Wheel-Version: 1.0\n"
    "Generator: setuptools (84.0.0)\n"
    "Root-Is-Purelib: true\n"
    "Tag: py3-none-any\n"
)
# The 64-byte DOS header of a PE file points at its "PE\0\0" signature.
_PE_IMAGE = b"MZ" + bytes(58) + (64).to_bytes(4, "little") + b"PE\0\0" + bytes(16)


def _write_wheel(
    directory: Path,
    *,
    tag: str = "py3-none-any",
    wheel_metadata: str | None = _PURE_WHEEL,
    extra: dict[str, bytes] | None = None,
    omit: frozenset[str] = frozenset(),
) -> Path:
    path = directory / f"scarf-1.0.0-{tag}.whl"
    with ZipFile(path, "w") as archive:
        for module in sorted(_REQUIRED_MODULES - omit):
            archive.writestr(module, "VALUE = 1\n")
        archive.writestr(f"{_DIST_INFO}/METADATA", "Name: scarf\nVersion: 1.0.0\n")
        if wheel_metadata is not None:
            archive.writestr(f"{_DIST_INFO}/WHEEL", wheel_metadata)
        for name, data in (extra or {}).items():
            info = ZipInfo(name)
            info.external_attr = 0o755 << 16
            archive.writestr(info, data)
    return path


def test_accepts_a_pure_python_wheel(tmp_path: Path) -> None:
    # Text that merely starts like a DOS header is not a native binary.
    wheel = _write_wheel(tmp_path, extra={"scarf/_mz.txt": b"MZ is plain text\n"})

    validate_wheel_contents(wheel)


@pytest.mark.parametrize(
    ("member", "data", "message"),
    [
        ("scarf-1.0.0.data/scripts/sgtsne", b"#!/bin/sh\n", "install-time data"),
        ("scarf/native.so", b"\x7fELF\x02\x01\x01" + bytes(9), "ELF"),
        ("scarf/native.dylib", b"\xcf\xfa\xed\xfe" + bytes(12), "Mach-O"),
        ("scarf/native.pyd", _PE_IMAGE, "PE"),
    ],
    ids=["data_script", "elf", "mach_o", "pe"],
)
def test_rejects_install_data_and_native_binaries(
    tmp_path: Path, member: str, data: bytes, message: str
) -> None:
    wheel = _write_wheel(tmp_path, extra={member: data})

    with pytest.raises(RuntimeError, match=message) as caught:
        validate_wheel_contents(wheel)
    assert member in str(caught.value)


@pytest.mark.parametrize(
    ("tag", "wheel_metadata", "message"),
    [
        (
            "cp312-cp312-linux_x86_64",
            _PURE_WHEEL,
            "file name tag 'cp312-cp312-linux_x86_64', expected 'py3-none-any'",
        ),
        (
            "py3-none-any",
            _PURE_WHEEL.replace("py3-none-any", "cp312-cp312-linux_x86_64"),
            "tags ['cp312-cp312-linux_x86_64'], expected ['py3-none-any']",
        ),
        (
            "py3-none-any",
            _PURE_WHEEL.replace("Root-Is-Purelib: true", "Root-Is-Purelib: false"),
            "Root-Is-Purelib: true",
        ),
        ("py3-none-any", None, "exactly one .dist-info/WHEEL file"),
    ],
    ids=[
        "platform_file_name",
        "platform_tag",
        "platlib_root",
        "missing_wheel_file",
    ],
)
def test_rejects_wheels_that_are_not_pure_python(
    tmp_path: Path, tag: str, wheel_metadata: str | None, message: str
) -> None:
    wheel = _write_wheel(tmp_path, tag=tag, wheel_metadata=wheel_metadata)

    with pytest.raises(RuntimeError, match=re.escape(message)):
        validate_wheel_contents(wheel)


def test_reports_retired_and_missing_modules_together(tmp_path: Path) -> None:
    required = sorted(_REQUIRED_MODULES)[0]
    wheel = _write_wheel(
        tmp_path,
        extra={"scarf/knn_utils.py": b"VALUE = 1\n", "bin/sgtsne": b"\x7fELF"},
        omit=frozenset({required}),
    )

    with pytest.raises(RuntimeError) as caught:
        validate_wheel_contents(wheel)
    problems = str(caught.value).splitlines()
    assert problems[0] == "Wheel contents violate the pure-Python contract:"
    assert any("scarf/knn_utils.py" in problem for problem in problems)
    assert any(required in problem for problem in problems)
    assert any("bin/sgtsne" in problem and "ELF" in problem for problem in problems)


@pytest.mark.parametrize(
    ("platform_name", "machine", "expected"),
    [
        ("linux", "x86_64", True),
        ("linux", "aarch64", False),
        ("win32", "AMD64", False),
    ],
)
def test_the_tsne_extra_is_installed_exactly_where_sgtsnepi_is_tested(
    monkeypatch: pytest.MonkeyPatch, platform_name: str, machine: str, expected: bool
) -> None:
    monkeypatch.setattr(smoke_wheel.sys, "platform", platform_name)
    monkeypatch.setattr(smoke_wheel.platform, "machine", lambda: machine)

    assert smoke_wheel.installs_tsne_extra() is expected


@pytest.mark.parametrize("with_tsne", [True, False])
def test_smoke_installs_the_wheel_into_a_clean_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, with_tsne: bool
) -> None:
    wheel = tmp_path / "scarf-1.0.0-py3-none-any.whl"
    commands: list[tuple[list[str], dict[str, str]]] = []

    def run(command, *, cwd, env, check):
        assert check is True
        assert Path(cwd).is_dir()
        commands.append((list(command), env))

    monkeypatch.setattr(smoke_wheel, "installs_tsne_extra", lambda: with_tsne)
    monkeypatch.setattr(smoke_wheel.subprocess, "run", run)
    smoke_wheel.smoke_installed_wheel(wheel)

    (create, _), (install, _), (imports, env), (workflow, _) = commands
    assert create[:2] == ["uv", "venv"]
    environment = Path(create[-1])
    python = Path(install[-2])
    assert python.parent.parent == environment
    requirement = f"scarf[tsne] @ {wheel.as_uri()}" if with_tsne else str(wheel)
    assert install == ["uv", "pip", "install", "--python", str(python), requirement]
    # Isolated mode keeps PYTHONPATH and the source checkout off sys.path.
    assert imports[:3] == [str(python), "-I", "-c"]
    assert workflow == [
        str(python),
        "-I",
        str(Path(smoke_wheel.__file__).with_name("smoke_workflow.py")),
        "with-tsne" if with_tsne else "without-tsne",
    ]
    assert env["HNSWLIB_NO_NATIVE"] == "1"


@pytest.mark.slow
def test_smoke_workflow_analyzes_the_synthetic_dataset_without_tsne(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # CI runs the workflow with t-SNE from the installed wheel on every push;
    # the branch without sgtsnepi otherwise runs only in the Windows release smoke.
    monkeypatch.setitem(sys.modules, "sgtsnepi", None)

    summary = run_workflow(tmp_path, expect_tsne=False)

    assert summary.startswith("Workflow smoke passed: ")
    assert "t-SNE raised the sgtsnepi installation guidance" in summary

import re
import tarfile
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest

from tests.validate_release_artifacts import (
    ArtifactValidationError,
    validate_release_artifacts,
)


def _metadata(name: str, version: str) -> bytes:
    return f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n".encode()


def _write_wheel(path: Path, *, name: str, version: str) -> None:
    with ZipFile(path, "w") as archive:
        archive.writestr(
            f"{name}-{version}.dist-info/METADATA",
            _metadata(name, version),
        )


def _write_sdist(path: Path, *, name: str, version: str) -> None:
    metadata = _metadata(name, version)
    info = tarfile.TarInfo(f"{name}-{version}/PKG-INFO")
    info.size = len(metadata)
    with tarfile.open(path, "w:gz") as archive:
        archive.addfile(info, BytesIO(metadata))


def _artifacts(tmp_path: Path, *, name: str, version: str) -> list[Path]:
    wheel = tmp_path / f"{name}-{version}-py3-none-any.whl"
    sdist = tmp_path / f"{name}-{version}.tar.gz"
    _write_wheel(wheel, name=name, version=version)
    _write_sdist(sdist, name=name, version=version)
    return [wheel, sdist]


def test_accepts_artifacts_matching_release_tag(tmp_path: Path, capsys) -> None:
    artifacts = _artifacts(tmp_path, name="scarf", version="1.0.0rc5")

    validate_release_artifacts(
        artifacts,
        release_tag="1.0.0rc5",
        distribution="scarf",
    )

    assert capsys.readouterr().out.splitlines() == [
        f"Checked {artifacts[0].name}: scarf 1.0.0rc5",
        f"Checked {artifacts[1].name}: scarf 1.0.0rc5",
    ]


def test_compares_canonical_names_and_normalized_versions(
    tmp_path: Path, capsys
) -> None:
    artifacts = _artifacts(tmp_path, name="Scarf", version="1.0.0-RC5")

    validate_release_artifacts(
        artifacts,
        release_tag="1.0.0rc5",
        distribution="scarf",
    )

    assert capsys.readouterr().out.splitlines() == [
        f"Checked {artifacts[0].name}: Scarf 1.0.0rc5",
        f"Checked {artifacts[1].name}: Scarf 1.0.0rc5",
    ]


@pytest.mark.parametrize(
    ("release_tag", "message"),
    [
        ("v-next", "Release tag 'v-next' is not a valid package version"),
        ("1.0.0+local", "Release tag '1.0.0+local' contains a local version"),
    ],
)
def test_rejects_an_unusable_release_tag(
    tmp_path: Path, release_tag: str, message: str
) -> None:
    artifacts = _artifacts(tmp_path, name="scarf", version="1.0.0")

    with pytest.raises(ArtifactValidationError, match=f"^{re.escape(message)}$"):
        validate_release_artifacts(
            artifacts, release_tag=release_tag, distribution="scarf"
        )


def test_reports_every_problem_together(tmp_path: Path) -> None:
    wheel, sdist = _artifacts(tmp_path, name="other", version="2.0.0")
    notes = tmp_path / "notes.txt"

    with pytest.raises(ArtifactValidationError) as raised:
        validate_release_artifacts(
            [notes, wheel, wheel, sdist],
            release_tag="1.0.0",
            distribution="scarf",
        )

    assert str(raised.value).splitlines() == [
        "Unsupported distribution file: notes.txt",
        f"{wheel.name} contains distribution 'other', expected 'scarf'",
        f"{wheel.name} contains version 2.0.0, expected release tag 1.0.0",
        f"{wheel.name} contains distribution 'other', expected 'scarf'",
        f"{wheel.name} contains version 2.0.0, expected release tag 1.0.0",
        f"{sdist.name} contains distribution 'other', expected 'scarf'",
        f"{sdist.name} contains version 2.0.0, expected release tag 1.0.0",
        "Expected exactly one wheel, found 2",
    ]


def _bare_wheel(path: Path, files: dict[str, bytes]) -> Path:
    with ZipFile(path, "w") as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
    return path


def _bare_sdist(path: Path, files: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, BytesIO(payload))
    return path


@pytest.mark.parametrize(
    ("make_artifact", "message"),
    [
        pytest.param(
            lambda path: _bare_wheel(path / "scarf-1.0.0-py3-none-any.whl", {}),
            "scarf-1.0.0-py3-none-any.whl must contain exactly one "
            ".dist-info/METADATA file",
            id="wheel-without-metadata",
        ),
        pytest.param(
            lambda path: _bare_sdist(
                path / "scarf-1.0.0.tar.gz",
                {"scarf-1.0.0/nested/PKG-INFO": _metadata("scarf", "1.0.0")},
            ),
            "scarf-1.0.0.tar.gz must contain exactly one top-level PKG-INFO file",
            id="sdist-without-top-level-metadata",
        ),
        pytest.param(
            lambda path: _bare_wheel(
                path / "scarf-1.0.0-py3-none-any.whl",
                {"scarf-1.0.0.dist-info/METADATA": b"Metadata-Version: 2.4\n"},
            ),
            "scarf-1.0.0-py3-none-any.whl metadata must contain Name and Version",
            id="metadata-without-version",
        ),
        pytest.param(
            lambda path: _bare_sdist(
                path / "scarf-1.0.0.tar.gz",
                {"scarf-1.0.0/PKG-INFO": _metadata("scarf", "banana")},
            ),
            "scarf-1.0.0.tar.gz has invalid version 'banana'",
            id="invalid-version",
        ),
    ],
)
def test_rejects_unreadable_package_metadata(
    tmp_path: Path, make_artifact, message: str
) -> None:
    with pytest.raises(ArtifactValidationError, match=f"^{re.escape(message)}$"):
        validate_release_artifacts(
            [make_artifact(tmp_path)], release_tag="1.0.0", distribution="scarf"
        )


def test_rejects_local_artifact_version(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path, name="scarf", version="1.0.0+local")

    with pytest.raises(ArtifactValidationError, match="forbidden local version"):
        validate_release_artifacts(
            artifacts,
            release_tag="1.0.0",
            distribution="scarf",
        )


def test_rejects_version_that_does_not_match_release_tag(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path, name="scarf", version="1.0.0rc4")

    with pytest.raises(ArtifactValidationError, match="expected release tag 1.0.0rc5"):
        validate_release_artifacts(
            artifacts,
            release_tag="1.0.0rc5",
            distribution="scarf",
        )


def test_requires_one_wheel_and_one_sdist(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path, name="scarf", version="1.0.0")

    with pytest.raises(ArtifactValidationError, match="exactly one source"):
        validate_release_artifacts(
            artifacts[:1],
            release_tag="1.0.0",
            distribution="scarf",
        )


def test_rejects_wrong_distribution_name(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path, name="other", version="1.0.0")

    with pytest.raises(ArtifactValidationError, match="expected 'scarf'"):
        validate_release_artifacts(
            artifacts,
            release_tag="1.0.0",
            distribution="scarf",
        )

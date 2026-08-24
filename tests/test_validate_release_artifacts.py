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


def test_accepts_artifacts_matching_release_tag(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path, name="scarf", version="1.0.0rc5")

    validate_release_artifacts(
        artifacts,
        release_tag="1.0.0rc5",
        distribution="scarf",
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

import argparse
import sys
import tarfile
from email.parser import BytesParser
from email.policy import default
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version


class ArtifactValidationError(RuntimeError):
    pass


def _parse_metadata(raw: bytes, artifact: Path) -> tuple[str, Version]:
    metadata = BytesParser(policy=default).parsebytes(raw)
    name = metadata.get("Name")
    version_text = metadata.get("Version")
    if not name or not version_text:
        raise ArtifactValidationError(
            f"{artifact.name} metadata must contain Name and Version"
        )
    try:
        version = Version(str(version_text))
    except InvalidVersion as exc:
        raise ArtifactValidationError(
            f"{artifact.name} has invalid version {version_text!r}"
        ) from exc
    return str(name), version


def _read_wheel_metadata(artifact: Path) -> tuple[str, Version]:
    with ZipFile(artifact) as archive:
        metadata_files = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_files) != 1:
            raise ArtifactValidationError(
                f"{artifact.name} must contain exactly one .dist-info/METADATA file"
            )
        return _parse_metadata(archive.read(metadata_files[0]), artifact)


def _read_sdist_metadata(artifact: Path) -> tuple[str, Version]:
    with tarfile.open(artifact, "r:gz") as archive:
        metadata_files = [
            member
            for member in archive.getmembers()
            if member.isfile()
            and PurePosixPath(member.name).name == "PKG-INFO"
            and len(PurePosixPath(member.name).parts) == 2
        ]
        if len(metadata_files) != 1:
            raise ArtifactValidationError(
                f"{artifact.name} must contain exactly one top-level PKG-INFO file"
            )
        extracted = archive.extractfile(metadata_files[0])
        if extracted is None:
            raise ArtifactValidationError(
                f"Could not read PKG-INFO from {artifact.name}"
            )
        return _parse_metadata(extracted.read(), artifact)


def validate_release_artifacts(
    artifacts: list[Path],
    *,
    release_tag: str,
    distribution: str,
) -> None:
    try:
        expected_version = Version(release_tag)
    except InvalidVersion as exc:
        raise ArtifactValidationError(
            f"Release tag {release_tag!r} is not a valid package version"
        ) from exc
    if expected_version.local is not None:
        raise ArtifactValidationError(
            f"Release tag {release_tag!r} contains a local version"
        )

    expected_name = canonicalize_name(distribution)
    wheel_count = 0
    sdist_count = 0
    errors: list[str] = []

    for artifact in artifacts:
        if artifact.name.endswith(".whl"):
            wheel_count += 1
            name, version = _read_wheel_metadata(artifact)
        elif artifact.name.endswith(".tar.gz"):
            sdist_count += 1
            name, version = _read_sdist_metadata(artifact)
        else:
            errors.append(f"Unsupported distribution file: {artifact.name}")
            continue

        if canonicalize_name(name) != expected_name:
            errors.append(
                f"{artifact.name} contains distribution {name!r}, expected {distribution!r}"
            )
        if version.local is not None:
            errors.append(f"{artifact.name} contains forbidden local version {version}")
        if version != expected_version:
            errors.append(
                f"{artifact.name} contains version {version}, "
                f"expected release tag {expected_version}"
            )
        print(f"Checked {artifact.name}: {name} {version}")

    if wheel_count != 1:
        errors.append(f"Expected exactly one wheel, found {wheel_count}")
    if sdist_count != 1:
        errors.append(f"Expected exactly one source distribution, found {sdist_count}")
    if errors:
        raise ArtifactValidationError("\n".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--distribution", required=True)
    parser.add_argument("artifacts", nargs="+", type=Path)
    args = parser.parse_args()

    try:
        validate_release_artifacts(
            args.artifacts,
            release_tag=args.release_tag,
            distribution=args.distribution,
        )
    except (ArtifactValidationError, BadZipFile, OSError, tarfile.TarError) as exc:
        print(f"Release artifact validation failed:\n{exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print("Release artifacts match the release tag")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Publish unit-test fixtures to the Cytebase scarf_tests repository.

Run without `--apply` to print the plan. Uploading needs a write token for the
bucket.

Example:
    uv run python scripts/publish_test_fixtures.py
    uv run python scripts/publish_test_fixtures.py --apply
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import BucketFile, batch_bucket_files, list_bucket_tree

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.download_fixtures import (  # noqa: E402
    _BUCKET_ID,
    _CYTEBASE_FIXTURES,
    _REPOSITORY,
    _sha256,
    datasets_dir,
)


def _published_paths() -> set[str]:
    paths: set[str] = set()
    for item in list_bucket_tree(
        _BUCKET_ID,
        prefix=_REPOSITORY,
        recursive=True,
        token=False,
    ):
        if isinstance(item, BucketFile):
            paths.add(item.path)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Upload fixtures instead of only printing the plan.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-upload fixtures that already exist remotely.",
    )
    args = parser.parse_args(argv)

    source_root = datasets_dir()
    remote_paths = _published_paths()
    uploads: list[tuple[Path, str]] = []
    skipped: list[str] = []

    for name, expected in _CYTEBASE_FIXTURES.items():
        local = source_root / name
        remote = f"{_REPOSITORY}/{name}"
        if not local.is_file():
            raise FileNotFoundError(f"Missing local fixture: {local}")
        digest = _sha256(local)
        if digest != expected:
            raise RuntimeError(
                f"{local.name} has SHA-256 {digest}, expected {expected}. "
                "Update _CYTEBASE_FIXTURES before publishing."
            )
        if remote in remote_paths and not args.force:
            skipped.append(remote)
            continue
        uploads.append((local, remote))

    for remote in skipped:
        print(f"  skip     {remote}")
    for local, remote in uploads:
        print(f"  upload   {remote} ({local.stat().st_size / 1e6:.1f} MB)")

    if not args.apply:
        print("Dry run. Pass --apply to publish, which needs a bucket write token.")
        return 0

    if uploads:
        batch_bucket_files(
            _BUCKET_ID,
            add=[(str(local), remote) for local, remote in uploads],
        )
        print(f"Uploaded {len(uploads)} fixture(s).")
    else:
        print("Nothing to upload.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

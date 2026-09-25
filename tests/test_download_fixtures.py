"""Offline checks for the checksum gate used before CI starts pytest."""

import hashlib

import pytest

from tests import download_fixtures


@pytest.mark.parametrize(
    ("cached", "force", "downloads"),
    [(b"verified", False, 0), (b"old archive", False, 1), (b"verified", True, 1)],
)
def test_fixture_cache_requires_current_digest(
    monkeypatch, tmp_path, cached, force, downloads
):
    payload = b"verified"
    fixture = tmp_path / "fixture.tar.gz"
    fixture.write_bytes(cached)
    monkeypatch.setattr(
        download_fixtures,
        "_CYTEBASE_FIXTURES",
        {fixture.name: hashlib.sha256(payload).hexdigest()},
    )
    calls = []

    def download(bucket_id, *, files, token, raise_on_missing_files):
        calls.append(files)
        assert token is False
        assert raise_on_missing_files is True
        assert files == [(f"scarf_tests/{fixture.name}", fixture)]
        fixture.write_bytes(payload)

    monkeypatch.setattr(download_fixtures, "download_bucket_files", download)
    download_fixtures._download_cytebase_fixtures(tmp_path, force=force)

    assert len(calls) == downloads
    assert fixture.read_bytes() == payload


@pytest.mark.parametrize("payload", [b"different archive", None])
def test_downloaded_fixture_must_exist_and_match_digest(monkeypatch, tmp_path, payload):
    fixture = tmp_path / "fixture.tar.gz"
    monkeypatch.setattr(
        download_fixtures,
        "_CYTEBASE_FIXTURES",
        {fixture.name: hashlib.sha256(b"verified").hexdigest()},
    )

    def download(*args, **kwargs):
        if payload is not None:
            fixture.write_bytes(payload)

    monkeypatch.setattr(download_fixtures, "download_bucket_files", download)
    message = "SHA-256" if payload is not None else "did not download"
    with pytest.raises(RuntimeError, match=message):
        download_fixtures._download_cytebase_fixtures(tmp_path, force=False)

    assert not fixture.exists()

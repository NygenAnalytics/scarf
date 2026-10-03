import hashlib
import platform
from pathlib import Path

import pytest

from profiling import provenance
from profiling.provenance import collect_run_provenance, source_tree_digest


def _tree(root: Path) -> Path:
    for name, text in {
        "scarf/__init__.py": "VERSION = 1\n",
        "scarf/storage/io.py": "def read(): ...\n",
        "scarf/storage/data.json": "{}",
        "profiling/stages.py": "STAGES = ()\n",
        "profiling/config.toml": "secret = 1\n",
        "tests/test_x.py": "def test(): ...\n",
        "pyproject.toml": "[project]\n",
        "uv.lock": "version = 1\n",
    }.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


_SHIPPED = [
    "profiling/stages.py",
    "pyproject.toml",
    "scarf/__init__.py",
    "scarf/storage/io.py",
    "uv.lock",
]


def _expected_digest(root: Path, names: list[str]) -> str:
    """Digest each shipped path and its bytes, NUL separated, in sorted order."""
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode() + b"\0" + (root / name).read_bytes() + b"\0")
    return digest.hexdigest()


@pytest.fixture
def executed_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Run provenance against a small fixed tree instead of the live checkout.

    Digesting the live checkout races with any concurrent edit, so the executed
    tree is a temporary one that nothing else changes.
    """
    root = _tree(tmp_path)
    monkeypatch.setattr(provenance, "_REPO_ROOT", root)
    provenance._executed_source_tree_digest.cache_clear()
    provenance._lockfile_digest.cache_clear()
    yield root
    # Later callers must digest the real checkout again, not this tree.
    provenance._executed_source_tree_digest.cache_clear()
    provenance._lockfile_digest.cache_clear()


def test_has_client_code_identity_requires_source_tree_digest(executed_tree):
    git_only = collect_run_provenance(clientProvenance={"gitSha": "abc123"})
    assert git_only["hasClientCodeIdentity"] is False
    assert git_only["gitSha"] == "abc123"
    assert git_only["sourceTreeMatchesClient"] is None

    with_tree = collect_run_provenance(
        clientProvenance={"sourceTreeSha256": "deadbeef"}
    )
    assert with_tree["hasClientCodeIdentity"] is True
    assert with_tree["sourceTreeSha256"] == "deadbeef"
    assert with_tree["sourceTreeMatchesClient"] is False


def test_client_identity_is_merged_without_collecting_it_again(
    executed_tree, monkeypatch
):
    def unexpected(**_kwargs):
        raise AssertionError("client identity must not be collected again")

    monkeypatch.setattr("profiling.provenance.collect_client_code_identity", unexpected)
    result = collect_run_provenance(
        clientProvenance={"gitSha": "abc123", "capturedOn": "client"},
        configDigestValue="config-digest",
    )

    assert result["gitSha"] == "abc123"
    assert result["configSha256"] == "config-digest"
    assert result["clientCapturedOn"] == "client"
    assert result["pythonVersion"] == platform.python_version()


def test_identity_is_collected_here_without_client_provenance(
    executed_tree, monkeypatch
):
    calls = []

    def local_identity(**_kwargs):
        calls.append(True)
        return {"gitSha": "local", "capturedOn": "client"}

    monkeypatch.setattr(
        "profiling.provenance.collect_client_code_identity", local_identity
    )
    result = collect_run_provenance()

    assert calls == [True]
    assert result["gitSha"] == "local"
    assert result["hasClientCodeIdentity"] is False
    assert "clientCapturedOn" not in result


def test_source_tree_digest_covers_exactly_the_shipped_code(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    assert provenance.source_tree_files(root) == _SHIPPED
    before = source_tree_digest(root)
    assert before == _expected_digest(root, _SHIPPED)

    # Files the image never ships do not change the identity.
    (root / "profiling" / "config.toml").write_text("secret = 2\n", encoding="utf-8")
    (root / "tests" / "test_x.py").write_text("changed\n", encoding="utf-8")
    assert source_tree_digest(root) == before

    # An uncommitted edit or a new source file does.
    (root / "scarf" / "storage" / "io.py").write_text("changed\n", encoding="utf-8")
    edited = source_tree_digest(root)
    assert edited != before
    (root / "scarf" / "new.py").write_text("\n", encoding="utf-8")
    assert source_tree_digest(root) != edited


def test_run_provenance_compares_client_and_executed_code(executed_tree: Path) -> None:
    executed = _expected_digest(executed_tree, _SHIPPED)

    matching = collect_run_provenance(clientProvenance={"sourceTreeSha256": executed})
    stale = collect_run_provenance(clientProvenance={"sourceTreeSha256": "0" * 64})
    local = collect_run_provenance(
        clientProvenance=None, configDigestValue="config-digest"
    )

    assert matching["executedSourceTreeSha256"] == executed
    assert matching["sourceTreeMatchesClient"] is True
    assert matching["hasClientCodeIdentity"] is True
    assert stale["executedSourceTreeSha256"] == executed
    assert stale["sourceTreeSha256"] == "0" * 64
    assert stale["sourceTreeMatchesClient"] is False
    # Identity collected here digests the executed tree, but without a client
    # digest there is nothing to compare.
    assert local["sourceTreeSha256"] == executed
    assert local["hasClientCodeIdentity"] is False
    assert local["sourceTreeMatchesClient"] is None
    assert local["executedSourceTreeSha256"] == executed
    assert local["configSha256"] == "config-digest"

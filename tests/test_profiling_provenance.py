from pathlib import Path

from profiling import provenance
from profiling.provenance import collect_run_provenance, source_tree_digest


def test_has_client_code_identity_requires_source_tree_digest():
    git_only = collect_run_provenance(clientProvenance={"gitSha": "abc123"})
    assert git_only["hasClientCodeIdentity"] is False
    assert git_only["gitSha"] == "abc123"

    with_tree = collect_run_provenance(
        clientProvenance={"sourceTreeSha256": "deadbeef"}
    )
    assert with_tree["hasClientCodeIdentity"] is True
    assert with_tree["sourceTreeSha256"] == "deadbeef"


def test_client_identity_is_merged_without_collecting_it_again(monkeypatch):
    def unexpected(**_kwargs):
        raise AssertionError("client identity must not be collected again")

    monkeypatch.setattr("profiling.provenance.collect_client_code_identity", unexpected)
    provenance = collect_run_provenance(
        clientProvenance={"gitSha": "abc123", "capturedOn": "client"},
        configDigestValue="config-digest",
    )

    assert provenance["gitSha"] == "abc123"
    assert provenance["configSha256"] == "config-digest"
    assert provenance["clientCapturedOn"] == "client"
    assert provenance["pythonVersion"]


def test_identity_is_collected_here_without_client_provenance(monkeypatch):
    calls = []

    def local_identity(**_kwargs):
        calls.append(True)
        return {"gitSha": "local", "capturedOn": "client"}

    monkeypatch.setattr(
        "profiling.provenance.collect_client_code_identity", local_identity
    )
    provenance = collect_run_provenance()

    assert calls == [True]
    assert provenance["gitSha"] == "local"
    assert provenance["hasClientCodeIdentity"] is False
    assert "clientCapturedOn" not in provenance


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


def test_source_tree_digest_covers_exactly_the_shipped_code(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    assert provenance.source_tree_files(root) == [
        "profiling/stages.py",
        "pyproject.toml",
        "scarf/__init__.py",
        "scarf/storage/io.py",
        "uv.lock",
    ]
    before = source_tree_digest(root)

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


def test_run_provenance_compares_client_and_executed_code() -> None:
    executed = source_tree_digest(provenance._REPO_ROOT)

    matching = collect_run_provenance(clientProvenance={"sourceTreeSha256": executed})
    stale = collect_run_provenance(clientProvenance={"sourceTreeSha256": "0" * 64})
    local = collect_run_provenance(
        clientProvenance=None, configDigestValue="config-digest"
    )

    assert matching["executedSourceTreeSha256"] == executed
    assert matching["sourceTreeMatchesClient"] is True
    assert stale["sourceTreeMatchesClient"] is False
    assert local["sourceTreeMatchesClient"] is None
    assert local["executedSourceTreeSha256"] == executed

from profiling.provenance import collect_run_provenance


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

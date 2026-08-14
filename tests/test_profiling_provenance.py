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

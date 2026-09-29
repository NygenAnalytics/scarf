"""Offline tests for the Cytebase deployment command."""

import os
import sys
from types import SimpleNamespace

import pytest

from scarf.cytebase.pipeline import deploy

pytestmark = pytest.mark.usefixtures("cytebase_offline")

APP_MODULE = "scarf.cytebase.pipeline.app"
MODAL_DEPLOY = [sys.executable, "-m", "modal", "deploy"]


@pytest.fixture
def runs(monkeypatch) -> list[tuple[list[str], dict[str, str]]]:
    """Record deployments instead of running ``modal deploy``."""
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(command, *, env, check):
        assert check is False
        calls.append((command, env))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(deploy.subprocess, "run", fake_run)
    return calls


@pytest.mark.parametrize(
    ("argv", "env_args", "bucket_key"),
    [
        pytest.param([], [], "CYTEBASE_BUCKET", id="default"),
        pytest.param(["--dev"], [], "CYTEBASE_BUCKET_DEV", id="dev"),
        pytest.param(
            ["--env", "staging"], ["--env", "staging"], "CYTEBASE_BUCKET", id="env"
        ),
        pytest.param(
            ["-e", "staging", "--dev"],
            ["--env", "staging"],
            "CYTEBASE_BUCKET_DEV",
            id="short-env-dev",
        ),
    ],
)
def test_deploy_runs_modal_with_the_selected_bucket_key(
    runs, argv, env_args, bucket_key
):
    assert deploy.main(argv) == 0
    assert runs == [
        (
            [*MODAL_DEPLOY, *env_args, "-m", APP_MODULE],
            {**os.environ, "CYTEBASE_BUCKET_KEY": bucket_key},
        )
    ]


def test_production_deploy_overrides_an_inherited_dev_bucket_key(runs, monkeypatch):
    monkeypatch.setenv("CYTEBASE_BUCKET_KEY", "CYTEBASE_BUCKET_DEV")
    deploy.main([])
    [(_, env)] = runs
    assert env["CYTEBASE_BUCKET_KEY"] == "CYTEBASE_BUCKET"


def test_deploy_passes_container_and_connection_limits(runs):
    deploy.main(["--process-containers", "6", "--download-connections", "3"])
    [(_, env)] = runs
    assert env["CYTEBASE_PROCESS_CONTAINERS"] == "6"
    assert env["CYTEBASE_DOWNLOAD_CONNECTIONS"] == "3"


def test_deploy_passes_the_hub_api_quota(runs):
    deploy.main(["--hub-api-quota", "3000"])
    [(_, env)] = runs
    assert env["CYTEBASE_HUB_API_QUOTA"] == "3000"


def test_deploy_leaves_the_hub_api_quota_to_the_app_default(runs):
    deploy.main([])
    [(_, env)] = runs
    assert "CYTEBASE_HUB_API_QUOTA" not in env


def test_dry_run_prints_the_deployment_without_running_it(runs, capsys):
    assert deploy.main(["--dry-run", "--dev", "-e", "staging"]) == 0
    assert runs == []
    printed = capsys.readouterr().err
    assert printed.startswith("CYTEBASE_BUCKET_KEY=CYTEBASE_BUCKET_DEV ")
    assert printed.rstrip().endswith(f"deploy --env staging -m {APP_MODULE}")


def test_deploy_returns_the_modal_exit_code(monkeypatch):
    monkeypatch.setattr(
        deploy.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=2),
    )
    assert deploy.main([]) == 2


def test_deploy_command_matches_the_modal_cli():
    pytest.importorskip("modal")
    from modal.cli.entry_point import entrypoint_cli

    command = deploy.deploy_command("staging")
    assert command[: len(MODAL_DEPLOY)] == MODAL_DEPLOY
    context = entrypoint_cli.commands["deploy"].make_context(
        "deploy", command[len(MODAL_DEPLOY) :]
    )
    assert {
        key: context.params[key] for key in ("app_ref", "env", "use_module_mode")
    } == {"app_ref": APP_MODULE, "env": "staging", "use_module_mode": True}

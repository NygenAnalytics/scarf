"""Deploy the Cytebase Modal app to a chosen Modal environment and bucket.

The target environment needs two Modal secrets: ``scarf-env`` with ``HF_TOKEN``
and the bucket key, and ``cytebase-api`` with ``CYTEBASE_API_TOKEN``, at least
32 visible ASCII characters without whitespace, which only the web function
receives. The deployment fails while either is missing. Every environment
needs both, including one whose operators never call the HTTP API and start
work only with ``python -m scarf.cytebase.pipeline reset-run`` or the Modal
SDK: the app deploys its web function with the workers, so ``modal deploy``
requires the ``cytebase-api`` secret. Every request to the deployed HTTP API
sends a workspace proxy auth token, created for example with
``modal workspace proxy-tokens create``, in the ``Modal-Key`` and
``Modal-Secret`` headers, and the API token in the ``Cytebase-Token`` header.
A proxy token alone is valid for every proxy-authenticated endpoint of the
workspace, so the app answers 401 without the API token. ``GET /jobs/{call_id}``
reports only calls that this deployment's API started, until Modal expires the
API's record of the call 7 days after the call started or was last polled. The
``scarf.cytebase.pipeline.app`` module describes how to create the secret and
which jobs the API reports.
"""

import argparse
import os
import shlex
import subprocess
import sys

APP_MODULE = "scarf.cytebase.pipeline.app"


def deploy_command(environment: str | None) -> list[str]:
    """Return the ``modal deploy`` command for the Cytebase app."""
    command = [sys.executable, "-m", "modal", "deploy"]
    if environment is not None:
        command += ["--env", environment]
    return [*command, "-m", APP_MODULE]


def deploy_settings(args: argparse.Namespace) -> dict[str, str]:
    """Return the variables ``app.py`` reads while defining the deployment.

    The bucket key is always set, so a key left in the shell never selects the
    development bucket for a production deployment.
    """
    settings = {
        "CYTEBASE_BUCKET_KEY": "CYTEBASE_BUCKET_DEV" if args.dev else "CYTEBASE_BUCKET"
    }
    if args.process_containers is not None:
        settings["CYTEBASE_PROCESS_CONTAINERS"] = str(args.process_containers)
    if args.download_connections is not None:
        settings["CYTEBASE_DOWNLOAD_CONNECTIONS"] = str(args.download_connections)
    if args.hub_api_quota is not None:
        settings["CYTEBASE_HUB_API_QUOTA"] = str(args.hub_api_quota)
    return settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-e", "--env", help="Modal environment; Modal's default when omitted"
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Use the bucket named by CYTEBASE_BUCKET_DEV in the scarf-env secret",
    )
    parser.add_argument(
        "--process-containers", type=int, help="Dataset worker container limit"
    )
    parser.add_argument(
        "--download-connections", type=int, help="Source connections, 1 to 4"
    )
    parser.add_argument(
        "--hub-api-quota",
        type=int,
        help=(
            "Hub API calls the token's account may make per five minutes "
            "(default 1000); dataset starts are paced to stay under it"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the deployment without running it"
    )
    args = parser.parse_args(argv)

    settings = deploy_settings(args)
    command = deploy_command(args.env)
    print(
        shlex.join([f"{key}={value}" for key, value in settings.items()] + command),
        file=sys.stderr,
    )
    if args.dry_run:
        return 0
    return subprocess.run(
        command, env={**os.environ, **settings}, check=False
    ).returncode


if __name__ == "__main__":
    sys.exit(main())

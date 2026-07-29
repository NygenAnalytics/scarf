"""Modal entrypoints for Scanpy dask e2e profiling.

Peer to ``profiling.modal_app`` run-e2e, tuned for an 8 CPU / 32 GiB box with
lazy H5AD + Dask through HVG/PCA (Scanpy docs recipe).

Deploy once (you run this), then spawn:

  modal deploy --env scarf_profiling -m profiling.scanpy_modal_app

  modal run --env scarf_profiling -m profiling.scanpy_modal_app -- \\
    run-e2e --config profiling/config.scanpy.toml --size 1000000

Watch: modal app logs scarf-profiling-scanpy --env scarf_profiling
"""

import argparse
import os
from pathlib import Path
from typing import Any

import modal

from profiling.modal_resources import (
    BASE_EPHEMERAL_DISK_MB,
    resolve_ephemeral_disk_mb,
)
from profiling.r2 import object_exists
from profiling.scanpy_config import (
    MAX_TIMEOUT_SECONDS,
    SCANPY_STAGE_ORDER,
    ScanpyProfilingConfig,
    load_scanpy_profiling_config,
)
from profiling.scanpy_modal_image import COMMON_FUNCTION_OPTIONS, app
from profiling.scanpy_stages import run_scanpy_e2e_funnel_body

_WORK = Path("/tmp/scarf-profiling-scanpy")


def validate_scanpy_modal_environment(config: ScanpyProfilingConfig) -> None:
    if config.modalEnvironmentName != "scarf_profiling":
        raise ValueError("Modal environment must be scarf_profiling")
    environment = modal.Environment.from_name(
        config.modalEnvironmentName,
        create_if_missing=False,
    )
    environment.hydrate()


def scanpy_modal_function_options(config: ScanpyProfilingConfig) -> dict[str, Any]:
    resources = config.resources
    _ = resolve_ephemeral_disk_mb(resources.ephemeralDiskMb)
    secret = modal.Secret.from_name(
        config.modalSecretName,
        environment_name=config.modalEnvironmentName,
        required_keys=["R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"],
    )
    return {
        "cpu": (resources.modalCpuRequest, resources.modalCpuLimit),
        "memory": (resources.modalMemoryRequestMb, resources.modalMemoryLimitMb),
        "env": {"R2_ENDPOINT": config.r2EndpointUrl},
        "secrets": [secret],
        "retries": 0,
        "max_containers": 1,
        "buffer_containers": 0,
        "timeout": resources.timeoutSeconds,
        "region": config.modalRegion,
    }


def _e2e_conflicting_uris(
    config: ScanpyProfilingConfig,
    nRows: int,
) -> list[str]:
    candidates = [
        config.e2eClaimUri(),
        config.funnelResultUri(nRows),
        *(config.resultUri(nRows, stage) for stage in SCANPY_STAGE_ORDER),
    ]
    return [uri for uri in candidates if object_exists(uri)]


def run_scanpy_e2e_entry(
    configDict: dict,
    nRows: int,
) -> dict:
    config = ScanpyProfilingConfig.model_validate(configDict)
    os.environ.setdefault("R2_ENDPOINT", config.r2EndpointUrl)
    conflicts = _e2e_conflicting_uris(config, nRows)
    if conflicts:
        raise FileExistsError(
            "run-e2e requires a fresh runTag; existing R2 objects: "
            + ", ".join(conflicts)
        )
    work = _WORK / f"e2e-{config.runTag}-{nRows}"
    return run_scanpy_e2e_funnel_body(config, nRows, workDir=work)


@app.function(
    **COMMON_FUNCTION_OPTIONS,
    timeout=MAX_TIMEOUT_SECONDS,
    memory=65_536,
    cpu=8.0,
    ephemeral_disk=BASE_EPHEMERAL_DISK_MB,
)
def run_scanpy_e2e_funnel_job(
    configDict: dict,
    nRows: int,
) -> dict:
    return run_scanpy_e2e_entry(configDict, nRows)


def _deployed_function(config: ScanpyProfilingConfig, name: str):
    return modal.Function.from_name(
        config.modalAppName,
        name,
        environment_name=config.modalEnvironmentName,
    )


def _print_spawned(label: str, call) -> None:
    print({"spawned": label, "functionCallId": call.object_id})
    print("disconnect is safe; watch with:")
    print("  modal app logs scarf-profiling-scanpy --env scarf_profiling")


@app.local_entrypoint()
def main(*arg_list: str) -> None:
    parser = argparse.ArgumentParser(prog="profiling.scanpy_modal_app")
    sub = parser.add_subparsers(dest="command", required=True)

    e2e_parser = sub.add_parser("run-e2e")
    e2e_parser.add_argument("--config", required=True)
    e2e_parser.add_argument("--size", type=int, required=True)
    e2e_parser.add_argument(
        "--ephemeral",
        action="store_true",
        help="Spawn from this modal run app without a deploy",
    )

    args = parser.parse_args(arg_list)
    config = load_scanpy_profiling_config(args.config)
    validate_scanpy_modal_environment(config)
    payload = config.model_dump(mode="python")

    if args.command == "run-e2e":
        if args.size not in config.targetSizes:
            raise SystemExit(f"size {args.size} is not in config.targetSizes")
        if not config.runTag.strip():
            raise SystemExit("run-e2e requires a non-empty runTag")
        options = scanpy_modal_function_options(config)
        target = (
            run_scanpy_e2e_funnel_job
            if args.ephemeral
            else _deployed_function(config, "run_scanpy_e2e_funnel_job")
        )
        call = target.with_options(**options).spawn(payload, args.size)
        _print_spawned(f"run_scanpy_e2e_funnel_job {args.size}", call)
        print(f"result URI (when done): {config.funnelResultUri(args.size)}")
        return

    raise SystemExit(f"unknown command: {args.command}")

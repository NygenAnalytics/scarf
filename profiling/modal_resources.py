from typing import Any

import modal

from profiling.config import PrepareResources, ProfilingConfig, StageResources

# Modal currently requires ephemeral disk in this range (MiB).
MIN_EPHEMERAL_DISK_MB = 524_288
MAX_EPHEMERAL_DISK_MB = 3_145_728
BASE_EPHEMERAL_DISK_MB = MIN_EPHEMERAL_DISK_MB

# Stage workers: retry after runner loss / InternalFailure. Orchestrators stay at 0.
STAGE_JOB_RETRIES = modal.Retries(
    max_retries=3,
    backoff_coefficient=2.0,
    initial_delay=5.0,
)


def resolve_ephemeral_disk_mb(requestedMb: int) -> int:
    if requestedMb > MAX_EPHEMERAL_DISK_MB:
        raise ValueError(
            "Requested ephemeral disk is "
            f"{requestedMb} MiB, but Modal allows at most "
            f"{MAX_EPHEMERAL_DISK_MB} MiB"
        )
    if requestedMb < MIN_EPHEMERAL_DISK_MB:
        return MIN_EPHEMERAL_DISK_MB
    return requestedMb


def validate_modal_environment(config: ProfilingConfig) -> None:
    if config.modalEnvironmentName != "scarf_profiling":
        raise ValueError("Modal environment must be scarf_profiling")
    environment = modal.Environment.from_name(
        config.modalEnvironmentName,
        create_if_missing=False,
    )
    environment.hydrate()


def modal_function_options(
    config: ProfilingConfig,
    resources: StageResources | PrepareResources,
    *,
    maxContainers: int = 1,
    retries: int | modal.Retries = STAGE_JOB_RETRIES,
) -> dict[str, Any]:
    if maxContainers <= 0:
        raise ValueError("maxContainers must be positive")
    secret = modal.Secret.from_name(
        config.modalSecretName,
        environment_name=config.modalEnvironmentName,
        required_keys=["R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"],
    )
    # ephemeral_disk is fixed on @app.function; with_options does not accept it.
    _ = resolve_ephemeral_disk_mb(resources.ephemeralDiskMb)
    # Do not pass cloud=; pinning aws (or any provider) shrinks Modal capacity.
    return {
        "cpu": (resources.modalCpuRequest, resources.modalCpuLimit),
        "memory": (resources.modalMemoryRequestMb, resources.modalMemoryLimitMb),
        "env": {"R2_ENDPOINT": config.r2EndpointUrl},
        "secrets": [secret],
        "retries": retries,
        "max_containers": maxContainers,
        "buffer_containers": 0,
        "timeout": resources.timeoutSeconds,
        "region": config.modalRegion,
    }


def orchestrator_function_options(
    config: ProfilingConfig,
    *,
    maxContainers: int = 1,
) -> dict[str, Any]:
    """Tiny options for run_all_jobs / run_size_jobs coordinators.

    These only spawn/wait; they must not request stage RAM (32–64 GiB) or they
    compete with the real stage workers for scarce high-memory capacity.
    """
    # Borrow secrets/region/env from a stage resource block, then shrink compute.
    options = modal_function_options(
        config,
        config.resourcesFor("reopenStore"),
        maxContainers=maxContainers,
        retries=0,
    )
    options["memory"] = (2048, 4096)
    options["cpu"] = (1.0, 1.0)
    options["timeout"] = 86_400
    return options

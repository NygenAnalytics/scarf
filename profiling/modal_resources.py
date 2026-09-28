from typing import Any

import modal

from profiling.config import PrepareResources, ProfilingConfig, StageResources
from profiling.modal_support import MODAL_ENVIRONMENT_NAME

# Every function declares this ephemeral disk (MiB), the smallest Modal allows, and
# with_options cannot change it.
BASE_EPHEMERAL_DISK_MB = 524_288

# Mutating workers never retry. A retry can overlap an in-flight write.
STAGE_JOB_RETRIES = 0


def require_base_ephemeral_disk(requestedMb: int) -> None:
    """Refuse an ephemeralDiskMb the fixed function disk cannot provide."""
    if requestedMb > BASE_EPHEMERAL_DISK_MB:
        raise ValueError(
            f"ephemeralDiskMb {requestedMb} exceeds the {BASE_EPHEMERAL_DISK_MB} MiB "
            "every profiling function declares; Modal does not allow a dynamic "
            "ephemeral_disk override"
        )


def validate_modal_environment(config: ProfilingConfig) -> None:
    if config.modalEnvironmentName != MODAL_ENVIRONMENT_NAME:
        raise ValueError(f"Modal environment must be {MODAL_ENVIRONMENT_NAME}")
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
    require_base_ephemeral_disk(resources.ephemeralDiskMb)
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
    donor = config.stageResources.get("reopenStore")
    if donor is None:
        if not config.stageResources:
            raise ValueError("orchestrator needs at least one stageResources block")
        donor = next(iter(config.stageResources.values()))
    options = modal_function_options(
        config,
        donor,
        maxContainers=maxContainers,
        retries=0,
    )
    options["memory"] = (2048, 4096)
    options["cpu"] = (1.0, 1.0)
    options["timeout"] = 86_400
    return options

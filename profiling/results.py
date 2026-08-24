from profiling.config import ProfilingConfig, StageName
from profiling.r2 import get_json, object_exists, put_json, put_json_if_absent
from profiling.stages import StageRunResult


def result_exists(config: ProfilingConfig, nRows: int, stage: StageName) -> bool:
    return object_exists(config.resultUri(nRows, stage))


def load_result(
    config: ProfilingConfig, nRows: int, stage: StageName
) -> dict[str, object] | None:
    uri = config.resultUri(nRows, stage)
    if not object_exists(uri):
        return None
    return get_json(uri)


def existing_error_result(
    config: ProfilingConfig, nRows: int, stage: StageName
) -> dict[str, object] | None:
    payload = load_result(config, nRows, stage)
    if payload is None:
        return None
    if payload.get("status") == "error":
        return payload
    return None


def write_result(
    config: ProfilingConfig,
    result: StageRunResult,
    *,
    overwrite: bool = False,
) -> str:
    uri = config.resultUri(result.nRows, result.stage)
    payload = result.to_json()
    if overwrite:
        put_json(uri, payload)
        return uri
    if not put_json_if_absent(uri, payload):
        raise FileExistsError(f"Refusing to overwrite existing stage result at {uri}")
    return uri


def write_funnel_result(
    config: ProfilingConfig,
    nRows: int,
    payload: dict[str, object],
) -> str:
    uri = config.funnelResultUri(nRows)
    if not put_json_if_absent(uri, payload):
        raise FileExistsError(f"Refusing to overwrite existing funnel result at {uri}")
    return uri

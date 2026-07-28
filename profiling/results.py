from profiling.config import ProfilingConfig, StageName
from profiling.r2 import object_exists, put_json
from profiling.stages import StageRunResult


def result_exists(config: ProfilingConfig, nRows: int, stage: StageName) -> bool:
    return object_exists(config.resultUri(nRows, stage))


def write_result(config: ProfilingConfig, result: StageRunResult) -> str:
    uri = config.resultUri(result.nRows, result.stage)
    put_json(uri, result.to_json())
    return uri


def write_funnel_result(
    config: ProfilingConfig,
    nRows: int,
    payload: dict[str, object],
) -> str:
    uri = config.funnelResultUri(nRows)
    put_json(uri, payload)
    return uri

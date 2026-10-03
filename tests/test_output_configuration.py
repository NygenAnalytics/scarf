import json
import re
import subprocess
import sys

import pytest

from scarf.utils import configure_output, logger, set_verbosity
from scarf.utils.logging import _config, progress_enabled


@pytest.fixture(autouse=True)
def _restore_output_configuration():
    yield
    set_verbosity("INFO")
    configure_output(progress=False, timestamps=False)


def test_import_defaults_are_notebook_first_in_isolated_process():
    code = """
import json
import scarf
from scarf.utils.logging import _config, progress_enabled

scarf.logger.info("default output")
print(json.dumps({"level": _config.level, "progress": progress_enabled()}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = result.stdout.splitlines()

    assert lines[0] == "INFO: default output"
    assert json.loads(lines[-1]) == {"level": "INFO", "progress": True}
    assert "\x1b" not in result.stdout


def test_configure_output_updates_settings_independently(capsys):
    configure_output(level="WARNING", progress=True)
    configure_output(timestamps=True)
    logger.warning("timestamped")
    output = capsys.readouterr().out

    assert _config.level == "WARNING"
    assert progress_enabled()
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}"
        r"[+-]\d{2}:\d{2} \| WARNING \| timestamped\n",
        output,
    )


def test_set_verbosity_file_output_has_no_ansi_codes(tmp_path):
    path = tmp_path / "scarf.log"
    set_verbosity("INFO", str(path))
    logger.warning("file output")
    output = path.read_text(encoding="utf-8")

    assert output == "WARNING: file output\n"
    assert "\x1b" not in output


def test_reconfiguration_preserves_caller_sink_and_configured_level():
    captured: list[str] = []
    sink = logger.add(
        lambda message: captured.append(message.record["message"]),
        level="TRACE",
    )
    try:
        configure_output(level="WARNING")
        logger.debug("caller only")
        logger.warning("visible warning")
    finally:
        logger.remove(sink)

    assert captured == ["caller only", "visible warning"]
    assert _config.level == "WARNING"


def test_reconfiguration_tolerates_a_stale_scarf_handler(capsys):
    logger.remove()
    configure_output(level="DEBUG")
    logger.debug("restored")

    assert capsys.readouterr().out == "DEBUG: restored\n"


@pytest.mark.parametrize("setting", ["progress", "timestamps"])
def test_configure_output_requires_boolean_settings(setting):
    with pytest.raises(TypeError, match=setting):
        configure_output(**{setting: 1})


def test_loading_tolerates_an_already_removed_default_handler():
    import importlib.util

    import scarf.utils.logging as scarf_logging

    # An application, like this test session, may remove Loguru's default
    # handler 0 before Scarf's logging module loads.
    with pytest.raises(ValueError):
        logger.remove(0)
    spec = importlib.util.spec_from_file_location(
        "scarf_logging_copy", scarf_logging.__file__
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module._config == module._OutputConfig()
    assert module._handler_id is not None
    # Removing the copy's handler succeeds only because loading installed it.
    logger.remove(module._handler_id)

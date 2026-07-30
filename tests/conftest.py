import sys

import pytest

from scarf.utils import configure_output, logger

pytest_plugins = [
    "tests.fixtures_downloader",
    "tests.fixtures_readers",
    "tests.fixtures_datastore",
]


@pytest.fixture(scope="session", autouse=True)
def _quiet_test_logs() -> None:
    """Keep pytest output readable while standalone CLIs retain INFO logs."""
    configure_output(progress=False)
    logger.remove()
    logger.add(sys.stderr, level="ERROR")

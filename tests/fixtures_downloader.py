import os

import pytest

from . import full_path


@pytest.fixture(scope="session")
def bastidas_ponce_data():
    sample = "bastidas-ponce_4K_pancreas-d15_rnaseq"
    local_h5ad = full_path(sample, "data.h5ad")
    if not os.path.isfile(local_h5ad):
        pytest.skip(
            f"Bundled test data missing at {local_h5ad}. "
            "Download via `python -m tests.download_fixtures --with-h5ad` "
            "(transient Hugging Face failures are skipped after retries)."
        )
    yield local_h5ad

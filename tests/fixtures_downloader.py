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
            "Add data.h5ad under tests/datasets/ to run h5ad reader tests."
        )
    yield local_h5ad

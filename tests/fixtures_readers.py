import os
import tarfile

import pytest

from . import full_path


@pytest.fixture(scope="session")
def toy_crdir_reader(tmp_path_factory):
    from scarf.readers import CrDirReader

    out_fn = tmp_path_factory.mktemp("toy_cr_dir")
    with tarfile.open(full_path("toy_cr_dir.tar.gz"), "r:gz") as tar:
        tar.extractall(out_fn, filter="data")
    reader = CrDirReader(str(out_fn))
    reader.rename_assays({"ASSAY4": "HTO"})
    yield reader


@pytest.fixture(scope="session")
def toy_crdir_empty(tmp_path_factory):
    from scarf.readers import CrDirReader

    out_fn = tmp_path_factory.mktemp("toy_cr_dir_empty")
    with tarfile.open(full_path("toy_cr_dir_empty.tar.gz"), "r:gz") as tar:
        tar.extractall(out_fn, filter="data")
    reader = CrDirReader(str(out_fn))
    yield reader


@pytest.fixture(scope="session")
def crh5_reader():
    from scarf.readers import CrH5Reader

    return CrH5Reader(full_path("1K_pbmc_citeseq.h5"))


@pytest.fixture(scope="session")
def mtx_dir(tmp_path_factory):
    fn = full_path("1K_pbmc_citeseq_dir.tar.gz")
    if not os.path.isfile(fn):
        pytest.skip(
            f"Bundled MTX fixture missing at {fn}. "
            "Add 1K_pbmc_citeseq_dir.tar.gz under tests/datasets/."
        )
    base = tmp_path_factory.mktemp("mtx_dir")
    with tarfile.open(fn, "r:gz") as tar:
        tar.extractall(base, filter="data")
    yield str(base / "1K_pbmc_citeseq_dir")


@pytest.fixture(scope="session")
def crdir_reader(mtx_dir):
    from scarf.readers import CrDirReader

    reader = CrDirReader(mtx_dir)
    yield reader


@pytest.fixture(scope="session")
def h5ad_reader(bastidas_ponce_data):
    from scarf.readers import H5adReader

    reader = H5adReader(bastidas_ponce_data)
    yield reader
    reader.h5.close()


@pytest.fixture(scope="session")
def loom_reader():
    from scarf.readers import LoomReader

    reader = LoomReader(
        full_path("sympathetic.loom"),
        cell_names_key="Cell_id",
        feature_names_key="Gene",
    )
    yield reader
    reader.h5.close()

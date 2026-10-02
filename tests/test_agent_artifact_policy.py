"""A fresh agent run never inherits prior numerical analysis through a store."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from pydantic_ai.models.function import FunctionModel
from scipy.sparse import csr_matrix

from scarf import DataStore
from scarf.agent import analyze_rna
from scarf.agent.evidence import (
    inspect_source,
    require_clean_analysis,
    verify_source,
)
from scarf.agent.models import AnalysisConfig, AnalysisInputError
from scarf.datastore.datastore import mount_datastore
from scarf.readers import H5adReader
from scarf.writers import H5adToZarr
from tests.test_agent_evidence import _files, runtime, study


@pytest.mark.parametrize(
    ("kind", "operation"),
    [
        ("feature_summary", "summarize_rna_features"),
        ("feature_selection", "select_detected_features"),
        ("feature_selection", "select_hvgs"),
        ("normalized", "run_normalization"),
        ("feature_scaling", "calculate_feature_scaling"),
        ("reduction", "run_pca"),
        ("batch_correction", "run_harmony"),
        ("ann_index", "build_ann_index"),
        ("neighbors", "query_neighbors"),
        ("connectivity_map", "build_connectivity_map"),
        ("embedding_initialization", "build_embedding_initialization"),
        ("embedding", "run_umap"),
        ("cluster_labels", "run_leiden_clustering"),
        ("cluster_cut", "cut_paris_hierarchy"),
        ("cluster_selection", "select_clusters"),
        ("marker_table", "run_marker_search"),
        ("doublet_score", "run_doublet_detection"),
    ],
)
def test_complete_numerical_results_are_unsupported_inputs(
    kind: str, operation: str
) -> None:
    queries = []
    ref = SimpleNamespace(kind=kind)

    def list_artifacts(**kwargs: Any) -> list[Any]:
        queries.append(kwargs)
        return [ref]

    store = SimpleNamespace(
        list_artifacts=list_artifacts,
        inspect_artifact=lambda supplied: SimpleNamespace(operation=operation),
    )
    with pytest.raises(AnalysisInputError, match="Prior complete numerical") as error:
        require_clean_analysis(store, "RNA")
    assert "Prepare a clean store" in str(error.value)
    assert "Mounting an analyzed source" in str(error.value)
    assert kind in str(error.value)
    assert operation in str(error.value)
    assert queries == [{"from_assay": "RNA", "complete_only": True}]


def test_only_selected_assay_complete_numerical_results_are_considered() -> None:
    rows = [
        SimpleNamespace(kind="normalized", assay="ATAC", complete=True),
        SimpleNamespace(kind="normalized", assay="RNA", complete=False),
    ]

    def list_artifacts(*, from_assay: str, complete_only: bool) -> list[Any]:
        assert complete_only
        return [row for row in rows if row.assay == from_assay and row.complete]

    require_clean_analysis(SimpleNamespace(list_artifacts=list_artifacts), "RNA")


@pytest.mark.parametrize(
    ("kind", "operation"),
    [
        ("feature_selection", "create_all_features"),
        ("feature_selection", "set_feature_selection"),
        ("embedding", "import_dimreduc"),
        ("cluster_labels", "import_cluster_labels"),
        ("cluster_labels", "import_active_identity"),
        ("cluster_labels", "snapshot_cluster_labels"),
        ("metadata_snapshot", "snapshot_run_metadata"),
    ],
)
def test_imports_and_frozen_inputs_are_allowed(kind: str, operation: str) -> None:
    store = SimpleNamespace(
        list_artifacts=lambda **kwargs: [SimpleNamespace(kind=kind)],
        inspect_artifact=lambda ref: SimpleNamespace(operation=operation),
    )
    require_clean_analysis(store, "RNA")


def test_prior_analysis_fails_read_only_before_model_or_new_pipeline(
    agent_rna_source: Path, tmp_path: Path
) -> None:
    source = agent_rna_source
    store = DataStore(
        str(source), min_features_per_cell=-1, nthreads=2, mem_budget="256M"
    )
    cells = store.snapshot_cell_selection()
    features = store.select_all_features(from_assay="RNA")
    prior = store.run_normalization(cells, features)
    assert store.inspect_artifact(prior).complete
    before = _files(source)

    async def forbidden_model(*args: Any) -> Any:
        pytest.fail("A prior numerical artifact must be rejected before a model call")

    result = analyze_rna(
        source,
        run_dir=tmp_path / "rejected",
        model=FunctionModel(forbidden_model),
        study=study(),
        runtime=runtime(),
    )
    assert result.status == "failed"
    assert "Prior complete numerical" in (result.run_dir / "report.md").read_text()
    assert store.pipeline.list_runs() == ()
    assert _files(source) == before


def test_saved_source_verification_allows_current_run_artifacts(
    agent_rna_source: Path,
) -> None:
    source = agent_rna_source
    prepared = inspect_source(source, study(), AnalysisConfig(), runtime())
    store = DataStore(
        str(source), min_features_per_cell=-1, nthreads=2, mem_budget="256M"
    )
    store.run_normalization(
        store.snapshot_cell_selection(), store.select_all_features(from_assay="RNA")
    )
    verify_source(source, prepared, study(), AnalysisConfig(), runtime())
    with pytest.raises(AnalysisInputError, match="Prior complete numerical"):
        inspect_source(source, study(), AnalysisConfig(), runtime())


def test_imported_annotations_are_safe_but_a_mount_inherits_numerical_results(
    tmp_path: Path,
) -> None:
    h5ad = tmp_path / "source.h5ad"
    source = tmp_path / "source.zarr"
    counts = np.arange(32, dtype=np.float32).reshape(8, 4) % 5
    ad.AnnData(
        X=csr_matrix(counts),
        obs=pd.DataFrame(
            {"cell_type": pd.Categorical(["author_a", "author_b"] * 4)},
            index=[f"cell{i}" for i in range(8)],
        ),
        var=pd.DataFrame(index=[f"gene{i}" for i in range(4)]),
        obsm={"X_umap": np.arange(16, dtype=np.float32).reshape(8, 2)},
    ).write_h5ad(h5ad)
    reader = H5adReader(
        str(h5ad), cluster_keys=["cell_type"], embedding_roles={"X_umap": "umap"}
    )
    try:
        H5adToZarr(
            reader,
            zarr_loc=str(source),
            assay_name="RNA",
            nthreads=2,
            mem_budget="256M",
        ).dump()
    finally:
        reader.h5.close()
    store = DataStore(
        str(source), min_features_per_cell=-1, nthreads=2, mem_budget="256M"
    )
    prepared = inspect_source(source, study(), AnalysisConfig(), runtime())
    assert "author_a" not in repr(prepared["contextEvidence"])
    imported = store.list_artifacts(from_assay="RNA", complete_only=True)
    assert {ref.kind for ref in imported} == {"cluster_labels", "embedding"}
    clean_mount = tmp_path / "clean_mount.zarr"
    mount_datastore(
        str(source),
        at=str(clean_mount),
        min_features_per_cell=-1,
        nthreads=2,
        mem_budget="256M",
    )
    inspect_source(clean_mount, study(), AnalysisConfig(), runtime())

    prior = store.run_normalization(
        store.snapshot_cell_selection(), store.select_all_features(from_assay="RNA")
    )
    analyzed_mount = tmp_path / "analyzed_mount.zarr"
    mounted = mount_datastore(
        str(source),
        at=str(analyzed_mount),
        min_features_per_cell=-1,
        nthreads=2,
        mem_budget="256M",
    )
    assert prior in mounted.list_artifacts(from_assay="RNA", complete_only=True)
    before = _files(source), _files(analyzed_mount)
    with pytest.raises(AnalysisInputError, match="Mounting an analyzed source"):
        inspect_source(analyzed_mount, study(), AnalysisConfig(), runtime())
    assert (_files(source), _files(analyzed_mount)) == before

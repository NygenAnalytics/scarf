import hashlib
import json
import os
from pathlib import Path

import pytest

from scarf.datastore.datastore import DataStore


_DEFAULT_CORPUS = Path("/home/parashar/data/scarf_master_compat/1k_citeseq")
_MASTER_COMMIT = "1ce016ed17710b7daebcf187c34c6f9b23aae0b4"
_FINAL_TREE_DIGEST = "b27aee365d89d6653d0ac5912f0044d6fe78e5f3a5e20bd164824085c441fedd"


def _corpus_root() -> Path:
    return Path(os.environ.get("SCARF_MASTER_CITESEQ_CORPUS", _DEFAULT_CORPUS))


def _tree_digest(root: Path) -> str:
    digest = hashlib.blake2b(digest_size=32)
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        relative_path = str(path.relative_to(root))
        payload = path.read_bytes()
        encoded_path = relative_path.encode()
        digest.update(len(encoded_path).to_bytes(8, "little"))
        digest.update(encoded_path)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def master_citeseq_corpus() -> tuple[Path, dict]:
    root = _corpus_root()
    store = root / "enriched"
    manifest_path = root / "master_manifest.json"
    if not store.is_dir() or not manifest_path.is_file():
        pytest.skip(
            "Master CITE-seq corpus is unavailable. Set "
            "SCARF_MASTER_CITESEQ_CORPUS to its external directory."
        )
    return store, json.loads(manifest_path.read_text())


@pytest.mark.integration
def test_master_citeseq_manifest_covers_requested_artifacts(
    master_citeseq_corpus: tuple[Path, dict],
) -> None:
    store, manifest = master_citeseq_corpus
    successful = {
        step["name"] for step in manifest["steps"] if step["status"] == "success"
    }
    steps = {step["name"]: step for step in manifest["steps"]}
    required = {
        "mark_hvgs_100",
        "mark_hvgs_200",
        "rna_graph_small",
        "rna_graph_small_repeat",
        "rna_graph_alternate",
        "rna_graph_unscaled",
        "rna_graph_harmony",
        "adt_graph",
        "integrate_assays_snn",
        "integrate_assays_wnn",
        "run_umap_a",
        "run_umap_b",
        "run_leiden_0_5",
        "run_leiden_1_0",
        "run_leiden_1_5",
        "run_paris_fixed",
        "run_paris_balanced",
        "run_marker_search",
        "get_imputed_t2",
        "get_imputed_t3",
        "run_pseudotime_scoring",
        "run_pseudotime_marker_search",
        "run_pseudotime_aggregation",
        "run_mapping",
        "run_mapping_coral",
        "run_unified_umap",
        "run_topacedo_sampler",
        "metric_lisi",
    }

    assert manifest["master_commit"] == _MASTER_COMMIT
    assert required <= successful
    assert manifest["summary"] == {"failed": 2, "skipped": 1, "success": 39}
    assert len(manifest["layout"]) == 308
    assert manifest["final_tree_digest"] == _FINAL_TREE_DIGEST
    assert _tree_digest(store) == _FINAL_TREE_DIGEST
    assert (
        steps["rna_graph_small"]["result"] == steps["rna_graph_small_repeat"]["result"]
    )
    assert steps["rna_graph_small_repeat"]["changed_file_count"] > 0
    assert steps["rna_graph_unscaled"]["result"] == steps["rna_graph_harmony"]["result"]
    assert any(
        path.endswith("/harmonizedData")
        for path in steps["rna_graph_harmony"]["changed_nodes"]
    )
    assert any(
        path.startswith("RNA/normed__")
        for path in steps["integrate_assays_wnn"]["changed_nodes"]
    )


@pytest.mark.integration
def test_frozen_citeseq_rna_open_fails_closed(
    master_citeseq_corpus: tuple[Path, dict],
) -> None:
    store, _manifest = master_citeseq_corpus
    with pytest.raises(ValueError, match="countsT|Zarr v3|Rebuild|repack"):
        DataStore(
            str(store),
            default_assay="RNA",
            assay_types={"assay2": "ADT"},
            min_features_per_cell=0,
            zarr_mode="r",
        )


@pytest.mark.integration
def test_current_reader_does_not_mutate_enriched_master_store(
    master_citeseq_corpus: tuple[Path, dict],
) -> None:
    store, manifest = master_citeseq_corpus
    before = _tree_digest(store)
    with pytest.raises(ValueError, match="countsT|Zarr v3|Rebuild|repack"):
        DataStore(
            str(store),
            default_assay="RNA",
            assay_types={"assay2": "ADT"},
            min_features_per_cell=0,
            zarr_mode="r",
        )

    after = _tree_digest(store)
    assert before == manifest["final_tree_digest"]
    assert after == before

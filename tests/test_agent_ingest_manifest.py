"""Tests for read-only H5AD decision manifests."""

from hashlib import sha256
from pathlib import Path

import h5py
import numpy as np
from scipy.sparse import csr_matrix

from scarf.agent.ingest import ingest
from scarf.agent.ingest.manifest import inspect_h5ad_manifest


def _write_sparse_group(
    h5: h5py.File | h5py.Group,
    key: str,
    values: np.ndarray,
) -> None:
    matrix = csr_matrix(values)
    group = h5.create_group(key)
    group.attrs["encoding-type"] = "csr_matrix"
    group.attrs["shape"] = values.shape
    group.create_dataset("data", data=matrix.data)
    group.create_dataset("indices", data=matrix.indices)
    group.create_dataset("indptr", data=matrix.indptr)


def _write_categorical(
    table: h5py.Group,
    name: str,
    *,
    codes: list[int],
    categories: list[str],
) -> None:
    group = table.create_group(name)
    group.attrs["encoding-type"] = "categorical"
    group.create_dataset("codes", data=np.asarray(codes, dtype=np.int16))
    group.create_dataset(
        "categories",
        data=np.asarray([value.encode() for value in categories]),
    )


def _write_metadata_table(
    h5: h5py.File,
    key: str,
    *,
    n_rows: int,
    feature_types: list[str] | None = None,
    filtered: list[bool] | None = None,
) -> h5py.Group:
    table = h5.create_group(key)
    table.attrs["_index"] = "_index"
    table.create_dataset(
        "_index",
        data=np.asarray(
            [f"{key.replace('/', '_')}-{i}".encode() for i in range(n_rows)]
        ),
    )
    if feature_types is not None:
        table.create_dataset(
            "feature_name",
            data=np.asarray([f"gene-{i}".encode() for i in range(n_rows)]),
        )
        table.create_dataset(
            "feature_types",
            data=np.asarray([value.encode() for value in feature_types]),
        )
    if filtered is not None:
        table.create_dataset(
            "feature_is_filtered",
            data=np.asarray(filtered, dtype=bool),
        )
    return table


def _write_cellxgene_h5ad(path: Path) -> None:
    with h5py.File(path, mode="w") as h5:
        _write_sparse_group(
            h5,
            "X",
            np.asarray(
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
                dtype=np.float32,
            ),
        )
        _write_sparse_group(
            h5,
            "raw/X",
            np.asarray(
                [[1, 0, 3, 0], [0, 2, 4, 1], [5, 0, 0, 2]],
                dtype=np.uint16,
            ),
        )
        layers = h5.create_group("layers")
        _write_sparse_group(
            layers,
            "log1p",
            np.asarray(
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
                dtype=np.float32,
            ),
        )

        obs = _write_metadata_table(h5, "obs", n_rows=3)
        _write_categorical(
            obs,
            "assay_ontology_term_id",
            codes=[0, 0, 0],
            categories=["EFO:0009922"],
        )
        _write_categorical(
            obs,
            "suspension_type",
            codes=[0, 0, 0],
            categories=["cell"],
        )
        _write_categorical(
            obs,
            "organism_ontology_term_id",
            codes=[0, 0, 0],
            categories=["NCBITaxon:9606"],
        )
        _write_categorical(
            obs,
            "donor_id",
            codes=[0, 0, 1],
            categories=["donor-a", "donor-b"],
        )
        _write_categorical(
            obs,
            "batch",
            codes=[0, 0, 1],
            categories=["batch-a", "batch-b"],
        )
        _write_categorical(
            obs,
            "cell_type",
            codes=[0, 1, 1],
            categories=["secret-author-label-a", "secret-author-label-b"],
        )
        _write_categorical(
            obs,
            "cell_type_ontology_term_id",
            codes=[0, 1, 1],
            categories=["CL:0000001", "CL:0000002"],
        )
        _write_categorical(
            obs,
            "leiden",
            codes=[0, 1, 1],
            categories=["cluster-zero", "cluster-one"],
        )
        obs.create_dataset(
            "is_primary_data",
            data=np.asarray([True, False, True], dtype=bool),
        )

        _write_metadata_table(
            h5,
            "var",
            n_rows=3,
            feature_types=["Gene Expression"] * 3,
            filtered=[False, False, True],
        )
        _write_metadata_table(
            h5,
            "raw/var",
            n_rows=4,
            feature_types=["Gene Expression"] * 4,
            filtered=[False, True, False, True],
        )
        obsm = h5.create_group("obsm")
        obsm.create_dataset("X_umap", data=np.zeros((3, 2), dtype=np.float32))
        uns = h5.create_group("uns")
        uns.create_dataset("schema_version", data=np.bytes_("7.1.0"))
        uns.create_dataset(
            "schema_reference",
            data=np.bytes_("https://example.invalid/schema/7.1.0"),
        )
        uns.create_dataset("batch_condition", data=np.asarray([b"batch"]))
        uns.create_dataset(
            "leiden_colors",
            data=np.asarray([b"#000000", b"#ffffff"]),
        )
        uns.create_group("neighbors")


def _write_simple_h5ad(
    path: Path,
    x: np.ndarray,
    *,
    raw: np.ndarray | None = None,
    feature_type: str = "Gene Expression",
) -> None:
    with h5py.File(path, mode="w") as h5:
        _write_sparse_group(h5, "X", x)
        obs = _write_metadata_table(h5, "obs", n_rows=x.shape[0])
        _write_categorical(
            obs,
            "donor_id",
            codes=list(range(x.shape[0])),
            categories=[f"d{i}" for i in range(x.shape[0])],
        )
        _write_metadata_table(
            h5,
            "var",
            n_rows=x.shape[1],
            feature_types=[feature_type] * x.shape[1],
        )
        if raw is not None:
            _write_sparse_group(h5, "raw/X", raw)
            _write_metadata_table(
                h5,
                "raw/var",
                n_rows=raw.shape[1],
                feature_types=[feature_type] * raw.shape[1],
            )


def test_h5ad_manifest_selects_one_raw_matrix_and_holds_out_labels(
    tmp_path: Path,
) -> None:
    path = tmp_path / "discover.h5ad"
    _write_cellxgene_h5ad(path)
    before = path.stat()

    manifest = inspect_h5ad_manifest(
        path,
        source_uri="cxg://collection/dataset@7.1.0",
        chunk_values=2,
    )

    assert manifest.decision.status == "supported"
    assert manifest.decision.selectedMatrixKey == "raw/X"
    assert manifest.selectedFeatureMetadataKey == "raw/var"
    assert manifest.nCells == 3
    assert manifest.nFeatures == 4
    assert manifest.sourceSha256 == sha256(path.read_bytes()).hexdigest()
    assert manifest.sourceSizeBytes == path.stat().st_size
    assert manifest.sourceUri == "cxg://collection/dataset@7.1.0"
    assert [item.key for item in manifest.matrixCandidates if item.selected] == [
        "raw/X"
    ]
    assert {item.key for item in manifest.matrixCandidates} == {
        "X",
        "layers/log1p",
        "raw/X",
    }

    visible_obs = {column.name: column for column in manifest.obs.columns}
    assert "cell_type" not in visible_obs
    assert "cell_type_ontology_term_id" not in visible_obs
    assert "leiden" not in visible_obs
    assert manifest.obs.heldOutAuthorColumnCount == 3
    assert manifest.labelBenchmarkEligible is True
    assert manifest.assayMetadata is not None
    assert manifest.assayMetadata.domainValues == ["EFO:0009922"]
    assert manifest.suspensionMetadata is not None
    assert manifest.suspensionMetadata.domainValues == ["cell"]
    assert manifest.organismMetadata is not None
    assert manifest.organismMetadata.domainValues == ["NCBITaxon:9606"]
    assert manifest.declaredBatchColumns == ["batch"]
    serialized = manifest.model_dump_json()
    assert "secret-author-label" not in serialized
    assert "cluster-zero" not in serialized

    assert manifest.inventory.layers == ["log1p"]
    assert manifest.inventory.obsm == ["X_umap"]
    assert manifest.inventory.priorEmbeddings == ["obsm/X_umap"]
    assert "leiden_colors" not in manifest.inventory.uns
    assert manifest.inventory.heldOutUnsItemCount == 1
    assert manifest.priorFiltering.cellXGeneSchemaDetected is True
    assert manifest.priorFiltering.cellXGeneSchemaVersion == "7.1.0"
    assert manifest.priorFiltering.cellSetStatus == "publishedCellsOnly"
    assert manifest.priorFiltering.rawCountsAvailable is True
    assert manifest.priorFiltering.originalDropletsAvailable is False
    assert manifest.priorFiltering.filteredFeatureCount == 2
    assert manifest.priorFiltering.nonPrimaryCellCount == 1
    assert path.stat().st_mtime_ns == before.st_mtime_ns


def test_h5ad_manifest_preservation_policy_exposes_labels_but_disables_benchmark(
    tmp_path: Path,
) -> None:
    path = tmp_path / "discover.h5ad"
    _write_cellxgene_h5ad(path)

    manifest = inspect_h5ad_manifest(path, author_label_policy="preservation")

    columns = {column.name: column for column in manifest.obs.columns}
    assert columns["cell_type"].domainValues == [
        "secret-author-label-a",
        "secret-author-label-b",
    ]
    assert columns["leiden"].domainValues == ["cluster-zero", "cluster-one"]
    assert manifest.obs.heldOutAuthorColumnCount == 0
    assert manifest.labelBenchmarkEligible is False
    assert "leiden_colors" in manifest.inventory.uns


def test_h5ad_manifest_abstains_for_normalized_only_source(tmp_path: Path) -> None:
    path = tmp_path / "normalized.h5ad"
    _write_simple_h5ad(
        path,
        np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
    )

    manifest = inspect_h5ad_manifest(path)

    assert manifest.decision.status == "abstained"
    assert manifest.decision.reasonCode == "normalizedOnly"
    assert manifest.decision.selectedMatrixKey is None
    assert manifest.selectedFeatureMetadataKey is None
    assert manifest.priorFiltering.rawCountsAvailable is False
    assert not any(candidate.selected for candidate in manifest.matrixCandidates)


def test_h5ad_manifest_prefers_authoritative_raw_x_over_integer_x(
    tmp_path: Path,
) -> None:
    path = tmp_path / "raw-authoritative.h5ad"
    _write_simple_h5ad(
        path,
        np.asarray([[1, 2], [3, 4]], dtype=np.uint16),
        raw=np.asarray([[5, 6, 7], [8, 9, 10]], dtype=np.uint16),
    )

    manifest = inspect_h5ad_manifest(path)

    assert manifest.decision.status == "supported"
    assert manifest.decision.selectedMatrixKey == "raw/X"
    assert [
        candidate.key for candidate in manifest.matrixCandidates if candidate.selected
    ] == ["raw/X"]


def test_h5ad_manifest_requires_selection_for_conflicting_count_layer(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ambiguous.h5ad"
    _write_simple_h5ad(
        path,
        np.asarray([[1, 2], [3, 4]], dtype=np.uint16),
        raw=np.asarray([[5, 6, 7], [8, 9, 10]], dtype=np.uint16),
    )
    with h5py.File(path, mode="a") as h5:
        layers = h5.create_group("layers")
        _write_sparse_group(
            layers,
            "counts",
            np.asarray([[11, 12], [13, 14]], dtype=np.uint16),
        )

    unresolved = inspect_h5ad_manifest(path)
    resolved = inspect_h5ad_manifest(path, matrix_key="raw/X")

    assert unresolved.decision.status == "needsInput"
    assert unresolved.decision.reasonCode == "ambiguousCountMatrices"
    assert unresolved.decision.options == ["X", "layers/counts", "raw/X"]
    assert not any(candidate.selected for candidate in unresolved.matrixCandidates)
    assert resolved.decision.status == "supported"
    assert resolved.decision.selectedMatrixKey == "raw/X"


def test_manifest_selected_matrix_key_is_accepted_by_ingest(tmp_path: Path) -> None:
    path = tmp_path / "raw-authoritative.h5ad"
    _write_simple_h5ad(
        path,
        np.asarray([[1, 2], [3, 4]], dtype=np.uint16),
        raw=np.asarray([[5, 6, 7], [8, 9, 10]], dtype=np.uint16),
    )
    manifest = inspect_h5ad_manifest(path)

    result = ingest(
        path=path,
        zarrPath=tmp_path / "converted.zarr",
        directions={"matrixKey": manifest.decision.selectedMatrixKey},
    )

    assert result.status == "done"
    assert result.acceptedActions[0] == {
        "op": "inspect_h5ad",
        "path": str(path),
        "matrixKey": "raw/X",
    }


def test_h5ad_manifest_abstains_when_selected_features_have_no_rna(
    tmp_path: Path,
) -> None:
    path = tmp_path / "adt.h5ad"
    _write_simple_h5ad(
        path,
        np.asarray([[1, 2], [3, 4]], dtype=np.uint16),
        feature_type="Antibody Capture",
    )

    manifest = inspect_h5ad_manifest(path)

    assert manifest.decision.status == "abstained"
    assert manifest.decision.reasonCode == "rnaModalityUnavailable"
    assert manifest.decision.options == ["ADT"]
    assert not any(candidate.selected for candidate in manifest.matrixCandidates)


def test_h5ad_manifest_caps_domains_while_counting_missing_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "domains.h5ad"
    with h5py.File(path, mode="w") as h5:
        _write_sparse_group(
            h5,
            "X",
            np.asarray([[1], [2], [3], [4]], dtype=np.uint16),
        )
        obs = _write_metadata_table(h5, "obs", n_rows=4)
        obs.create_dataset(
            "free_text_group",
            data=np.asarray([b"a", b"b", b"", b"c"]),
        )
        _write_metadata_table(
            h5,
            "var",
            n_rows=1,
            feature_types=["Gene Expression"],
        )

    manifest = inspect_h5ad_manifest(
        path,
        chunk_values=1,
        max_domain_values=2,
    )

    column = next(
        column for column in manifest.obs.columns if column.name == "free_text_group"
    )
    assert column.missingCount == 1
    assert column.missingFraction == 0.25
    assert column.domainTruncated is True
    assert column.domainSize is None
    assert column.domainValues == []
    assert column.valueCounts == {}

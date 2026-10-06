"""Parity between write-time RNA classification and DataStore presets."""

import pytest

from scarf.assay import (
    ADTassay,
    ATACassay,
    Assay,
    RNAassay,
    is_rna_assay_type,
    preset_assay_types,
    resolve_persisted_assay_type,
)


def test_preset_map_names_every_assay_class():
    assert preset_assay_types() == {
        "RNA": RNAassay,
        "ATAC": ATACassay,
        "ADT": ADTassay,
        "HTO": ADTassay,
        "CRISPR": Assay,
        "ANTIGEN": Assay,
        "CUSTOM": Assay,
        "GeneActivity": RNAassay,
        "GeneScores": RNAassay,
        "URNA": RNAassay,
        "Assay": Assay,
    }


def test_rna_classifier_accepts_classes_and_instances():
    class GeneModules(RNAassay):
        pass

    assert is_rna_assay_type(RNAassay)
    assert is_rna_assay_type(GeneModules)
    assert not is_rna_assay_type(ATACassay)
    assert is_rna_assay_type(RNAassay.__new__(GeneModules))
    assert not is_rna_assay_type(ADTassay.__new__(ADTassay))
    assert not is_rna_assay_type(3)


def test_rna_classifier_aliases():
    assert is_rna_assay_type("RNA")
    assert is_rna_assay_type("GeneActivity")
    assert is_rna_assay_type("GeneScores")
    assert is_rna_assay_type("URNA")
    assert not is_rna_assay_type("ATAC")
    assert not is_rna_assay_type("ADT")
    assert not is_rna_assay_type("Assay")
    assert not is_rna_assay_type("CUSTOM_NAME")


def test_resolve_persisted_assay_type_keeps_only_presets():
    assert resolve_persisted_assay_type("RNA") == "RNA"
    assert resolve_persisted_assay_type("CUSTOM_NAME") == "Assay"
    assert resolve_persisted_assay_type("CUSTOM_NAME", "RNA") == "RNA"
    assert resolve_persisted_assay_type("CUSTOM_NAME", "Assay") == "Assay"
    # An explicit type is never replaced; "Assay" is the explicit generic choice.
    with pytest.raises(
        ValueError,
        match=r"assay_type 'not-a-preset' of assay 'CUSTOM_NAME' is not a preset"
        r".*'HTO'.*'Assay' for a generic assay",
    ):
        resolve_persisted_assay_type("CUSTOM_NAME", "not-a-preset")
    assert resolve_persisted_assay_type("GeneActivity") == "GeneActivity"
    assert not is_rna_assay_type(Assay)


def test_lookup_persisted_assay_type_rejects_an_unknown_recorded_type():
    from scarf.assay import lookup_persisted_assay_type

    with pytest.raises(
        ValueError,
        match=r"Assay 'Spatial' is recorded in assayTypes as 'Imaging', which is "
        r"not a preset.*assay_types=\{'Spatial': ",
    ):
        lookup_persisted_assay_type("Spatial", {"Spatial": "Imaging"})
    with pytest.raises(ValueError, match="recorded in assayTypes as 3"):
        lookup_persisted_assay_type("Spatial", {"Spatial": 3})
    # An explicit preset wins over the recorded value it replaces.
    assert (
        lookup_persisted_assay_type("Spatial", {"Spatial": "Imaging"}, assay_type="RNA")
        == "RNA"
    )


def test_lookup_persisted_assay_type_prefers_map_and_explicit():
    from scarf.assay import lookup_persisted_assay_type

    assert lookup_persisted_assay_type("GeneActivity") == "GeneActivity"
    assert (
        lookup_persisted_assay_type(
            "CUSTOM",
            {"CUSTOM": "GeneScores"},
        )
        == "GeneScores"
    )
    assert (
        lookup_persisted_assay_type(
            "CUSTOM",
            {"CUSTOM": "Assay"},
            assay_type="URNA",
        )
        == "URNA"
    )


def test_declared_assay_type_is_the_type_the_assay_carries():
    from types import SimpleNamespace

    from scarf.assay.classification import declared_assay_type

    # Declarations are kept even when they share an assay class, and the
    # store attributes are never read again.
    tags = SimpleNamespace(
        name="tags",
        assayType="HTO",
        _artifact_root=SimpleNamespace(attrs={"assayTypes": {"tags": "Assay"}}),
    )
    assert declared_assay_type(tags) == "HTO"
    gene_activity = SimpleNamespace(name="GeneActivity", assayType="GeneActivity")
    assert declared_assay_type(gene_activity) == "GeneActivity"
    with pytest.raises(ValueError, match="Assay 'protein' carries no declared type"):
        declared_assay_type(SimpleNamespace(name="protein", assayType=None))


def test_an_assay_declares_only_a_preset_of_its_own_class(tmp_path):
    import numpy as np

    from scarf.metadata import MetaData
    from scarf.storage.schema import create_cell_data, create_zarr_count_assay
    from scarf.storage.stores import load_zarr
    from scarf.writers.counts_t import finalize_writer_counts_t
    from tests.storage_helpers import finalize_test_counts

    root = load_zarr(str(tmp_path / "data.zarr"), mode="w")
    ids = np.asarray(["c0", "c1"])
    create_cell_data(root, None, ids=ids, names=ids)
    counts = create_zarr_count_assay(
        root, "tags", None, 2, np.asarray(["t0"]), np.asarray(["t0"]), dtype="uint8"
    )
    counts[:] = [[1], [2]]
    finalize_test_counts(counts)
    finalize_writer_counts_t(root, "tags", None)

    def build(cls, assay_type):
        return cls(
            z=root,
            workspace=None,
            name="tags",
            cell_data=MetaData(root["cellData"]),
            nthreads=1,
            assay_type=assay_type,
        )

    assert build(ADTassay, "HTO").assayType == "HTO"
    assert build(Assay, "CRISPR").assayType == "CRISPR"
    assert build(Assay, None).assayType is None
    with pytest.raises(ValueError, match="declared as 'HTO', which opens as ADTassay"):
        build(Assay, "HTO")
    with pytest.raises(ValueError, match="assay_type 'hto' of assay 'tags' is not a"):
        build(ADTassay, "hto")


def test_recorded_assay_types_reject_a_record_that_is_not_a_mapping():
    from scarf.assay.classification import recorded_assay_types

    assert recorded_assay_types(None, ["RNA"]) == {}
    assert recorded_assay_types({"RNA": "RNA", "x": 3}, ["RNA"]) == {
        "RNA": "RNA",
        "x": 3,
    }
    # A record that is not a mapping is malformed, not empty.
    with pytest.raises(
        ValueError,
        match=r"The assayTypes attribute of the store is \['ADT'\], which is not a "
        r"mapping.*zarr_mode='r\+'.*assay_types=\{'RNA': <preset>, 'tags': <preset>\}",
    ):
        recorded_assay_types(["ADT"], ["RNA", "tags"])
    with pytest.raises(ValueError, match="names a preset for every assay of the"):
        recorded_assay_types("RNA")


def test_validate_assay_types_names_the_store_assays():
    from scarf.assay.classification import validate_assay_types

    assert validate_assay_types(None, ["RNA"]) == {}
    assert validate_assay_types({"RNA": "URNA"}, ["RNA", "ADT"]) == {"RNA": "URNA"}
    with pytest.raises(
        ValueError,
        match=r"assay_types names assays that are not in the store: 'RNAX'\. "
        r"Assays in the store: 'RNA', 'ADT'",
    ):
        validate_assay_types({"RNAX": "RNA"}, ["RNA", "ADT"])
    with pytest.raises(ValueError, match="assay_type 'rna' of assay 'RNA' is not a"):
        validate_assay_types({"RNA": "rna"}, ["RNA"])
    with pytest.raises(TypeError, match="assay_types must be a mapping"):
        validate_assay_types([("RNA", "RNA")], ["RNA"])  # type: ignore[arg-type]

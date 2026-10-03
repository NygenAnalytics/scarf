"""Parity between write-time RNA classification and DataStore presets."""

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
    assert resolve_persisted_assay_type("CUSTOM_NAME", "not-a-preset") == "Assay"
    assert resolve_persisted_assay_type("GeneActivity") == "GeneActivity"
    assert not is_rna_assay_type(Assay)


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

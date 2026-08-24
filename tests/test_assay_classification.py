"""Parity between write-time RNA classification and DataStore presets."""

from scarf.assay import (
    Assay,
    RNAassay,
    is_rna_assay_type,
    preset_assay_types,
    resolve_persisted_assay_type,
    rna_assay_type_names,
)


def test_rna_classifier_matches_preset_map():
    presets = preset_assay_types()
    for name, assay_cls in presets.items():
        expected = issubclass(assay_cls, RNAassay)
        assert is_rna_assay_type(name) is expected
        assert is_rna_assay_type(assay_cls) is expected
    assert rna_assay_type_names() == frozenset(
        name for name, cls in presets.items() if issubclass(cls, RNAassay)
    )


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

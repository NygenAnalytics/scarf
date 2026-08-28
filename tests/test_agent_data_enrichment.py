"""Tests for the read-only data enrichment agent."""

from types import SimpleNamespace

import pytest
from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from scarf.agent.characterize_features import FeatureCharacterization
from scarf.agent.data_enrichment import (
    AssayFeatureInspection,
    AssayFeatureInspectionBatch,
    DataEnrichmentAgent,
    DataEnrichmentContext,
    DataEnrichmentDependencies,
    DataEnrichmentReport,
    DataEnrichmentToolCall,
    ExogenousFeatureEvidence,
    FeatureFamilyEvidence,
    FeatureLookupResult,
    FeatureLookupBatch,
    FeatureMatch,
    FeatureReference,
    FeatureSelectionPolicy,
    validate_data_enrichment_report,
)


class FeatureTable:
    def __init__(self) -> None:
        self.fetches: list[str] = []

    def fetch_all(self, column: str) -> list[str]:
        self.fetches.append(column)
        values = {
            "ids": ["ENSG00000198727", "ERCC-00002", "ENSG00000111640"],
            "names": ["MT-CYB", "ERCC-00002", "GAPDH"],
        }
        return values[column]


class ReadOnlyStore:
    assay_names = ["RNA"]

    def __init__(self) -> None:
        self.features = FeatureTable()
        self.assay = SimpleNamespace(feats=self.features)

    def get_assay(self, assay_name: str) -> SimpleNamespace:
        assert assay_name == "RNA"
        return self.assay


def characterization() -> FeatureCharacterization:
    return FeatureCharacterization(
        status="done",
        assays=[
            {
                "assay": "RNA",
                "assayKind": "RNAassay",
                "identity": {"nFeatures": 3, "nDuplicateIds": 0},
                "species": "unknown",
                "speciesMethod": "inconclusive",
                "speciesResolution": {
                    "reason": "Feature identifiers alone are inconclusive"
                },
                "families": [
                    {
                        "family": "mitochondrial",
                        "species": "unknown",
                        "method": "symbolPrefix",
                        "count": 1,
                        "examples": ["MT-CYB"],
                        "defaultExclude": True,
                    },
                    {
                        "family": "sex",
                        "species": "unknown",
                        "method": "skipped",
                        "count": 0,
                        "examples": [],
                        "defaultExclude": False,
                        "skipped": "speciesUnknown",
                    },
                ],
                "exogenous": [
                    {
                        "id": "ERCC-00002",
                        "name": "ERCC-00002",
                        "score": 4,
                        "class": "unresolved",
                    }
                ],
            }
        ],
    )


def test_data_enrichment_models_have_factories_and_camelcase_fields() -> None:
    model_types: list[type[BaseModel]] = [
        DataEnrichmentContext,
        FeatureFamilyEvidence,
        ExogenousFeatureEvidence,
        AssayFeatureInspection,
        FeatureReference,
        FeatureMatch,
        FeatureLookupResult,
        FeatureLookupBatch,
        AssayFeatureInspectionBatch,
        FeatureSelectionPolicy,
        DataEnrichmentToolCall,
        DataEnrichmentReport,
        DataEnrichmentDependencies,
    ]

    for model_type in model_types:
        assert isinstance(model_type.get_blank(), model_type)
        assert isinstance(model_type.get_example(), model_type)
        assert all("_" not in field_name for field_name in model_type.model_fields)


def test_data_enrichment_agent_uses_only_read_tools_and_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scarf.agent import data_enrichment as module

    store = ReadOnlyStore()
    tool_names: set[str] = set()
    settings: list[dict] = []
    characterization_calls: list[dict] = []

    def inspect_characterization(_store, **kwargs):
        characterization_calls.append(kwargs)
        return characterization()

    monkeypatch.setattr(module, "characterize_features", inspect_characterization)
    state = {"request": 0}

    async def reply(
        _messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        tool_names.update(tool.name for tool in info.function_tools)
        settings.append(dict(info.model_settings or {}))
        request = state["request"]
        state["request"] += 1
        if request == 0:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="inspect_assay_features_batch",
                        args={},
                    )
                ]
            )
        if request == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="find_present_features_batch",
                        args={
                            "queries_by_assay": {"RNA": ["MT-CYB", "ERCC-00002"]},
                        },
                    )
                ]
            )
        report = DataEnrichmentReport(
            status="done",
            policies=[
                FeatureSelectionPolicy(
                    assay="RNA",
                    species="homo_sapiens",
                    organismName="human",
                    speciesConfidence="medium",
                    speciesRationale=(
                        "Human tissue and cell-type context resolves ambiguous IDs"
                    ),
                    excludeFamilies=["mitochondrial"],
                    protectFamilies=["sex"],
                    artificialFeatures=["ERCC-00002"],
                    tissueReferences=["lung"],
                    cellTypeReferences=["alveolar macrophage"],
                    experimentalReferences=["ERCC spike-in"],
                    rationale="Exclude technical signals while preserving biology",
                    evidenceIds=[
                        "context:organism",
                        "context:tissue:0",
                        "context:cellType:0",
                        "context:experiment:0",
                        "assay:RNA:family:mitochondrial",
                        "assay:RNA:feature:ERCC-00002",
                        "assay:RNA:exogenous:ERCC-00002",
                    ],
                )
            ],
        )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args=report.model_dump(),
                )
            ]
        )

    result = DataEnrichmentAgent(FunctionModel(reply)).run(
        store,
        context=DataEnrichmentContext(
            studyContext="Treated human lung samples with an ERCC spike-in",
            organismHint="human",
            tissueReferences=["lung"],
            cellTypeReferences=["alveolar macrophage"],
            experimentalDetails=["ERCC spike-in"],
        ),
        allow_download=False,
    )

    assert result.status == "done"
    assert result.policies[0].species == "homo_sapiens"
    assert result.policies[0].artificialFeatures == ["ERCC-00002"]
    assert [call.name for call in result.toolCalls] == [
        "inspect_assay_features_batch",
        "find_present_features_batch",
    ]
    assert result.runInfo.agentName == "data_enrichment"
    assert result.runInfo.modelName.startswith("function:")
    assert [call.toolName for call in result.runInfo.toolCalls] == [
        "inspect_assay_features_batch",
        "find_present_features_batch",
    ]
    assert tool_names == {
        "inspect_assay_features_batch",
        "find_present_features_batch",
    }
    assert store.features.fetches == ["ids", "names"]
    assert characterization_calls[0]["model"] is None
    assert characterization_calls[0]["studyContext"].startswith("Treated human")
    assert settings[0]["parallel_tool_calls"] is False
    assert settings[0]["extra_body"]["reasoning_effort"] == "none"


def test_data_enrichment_retries_hallucinated_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scarf.agent import data_enrichment as module

    store = ReadOnlyStore()
    monkeypatch.setattr(
        module,
        "characterize_features",
        lambda *_args, **_kwargs: characterization(),
    )
    state = {"request": 0}

    async def reply(
        _messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        request = state["request"]
        state["request"] += 1
        if request == 0:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="inspect_assay_features_batch",
                        args={},
                    )
                ]
            )
        if request == 1:
            bad = DataEnrichmentReport(
                status="done",
                policies=[
                    FeatureSelectionPolicy(
                        assay="RNA",
                        species="unknown",
                        excludeFeatures=["NOT_A_GENE"],
                        evidenceIds=["assay:RNA:species"],
                    )
                ],
            )
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=info.output_tools[0].name,
                        args=bad.model_dump(),
                    )
                ]
            )
        if request == 2:
            corrected = DataEnrichmentReport(
                status="done",
                policies=[
                    FeatureSelectionPolicy(
                        assay="RNA",
                        species="unknown",
                        rationale="Use only observed evidence.",
                        evidenceIds=["assay:RNA:species"],
                    )
                ],
            )
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=info.output_tools[0].name,
                        args=corrected.model_dump(),
                    )
                ]
            )
        raise AssertionError(
            "The grounded validator should accept the corrected report"
        )

    result = DataEnrichmentAgent(FunctionModel(reply)).run(store)

    assert result.status == "done"
    assert result.policies[0].excludeFeatures == []
    assert state["request"] == 3


def test_data_enrichment_validates_policy_and_assay() -> None:
    with pytest.raises(ValueError, match="both excluded and protected"):
        FeatureSelectionPolicy(
            assay="RNA",
            excludeFamilies=["cellCycle"],
            protectFamilies=["cellCycle"],
        )

    agent = DataEnrichmentAgent(FunctionModel(lambda _messages, _info: ModelResponse()))
    with pytest.raises(ValueError, match="unknown assays"):
        agent.run(ReadOnlyStore(), assays=["ADT"])


def test_enrichment_copies_caller_context_instead_of_model_paraphrases() -> None:
    inspection = AssayFeatureInspection(
        assay="RNA",
        species="unknown",
        evidenceIds=["assay:RNA:species"],
    )
    deps = DataEnrichmentDependencies(
        store=ReadOnlyStore(),
        context=DataEnrichmentContext(
            tissueReferences=["peripheral blood"],
            cellTypeReferences=["T cell"],
            experimentalDetails=["10x 3 prime RNA-seq", "single donor"],
        ),
        assays=["RNA"],
        inspections={"RNA": inspection},
        evidenceIds={"assay:RNA:species"},
    )
    report = DataEnrichmentReport(
        status="done",
        policies=[
            FeatureSelectionPolicy(
                assay="RNA",
                tissueReferences=[],
                cellTypeReferences=[],
                experimentalReferences=["10x 5K PBMC"],
                evidenceIds=["assay:RNA:species"],
            )
        ],
    )

    validated = validate_data_enrichment_report(deps, report)

    assert validated.policies[0].tissueReferences == ["peripheral blood"]
    assert validated.policies[0].cellTypeReferences == ["T cell"]
    assert validated.policies[0].experimentalReferences == [
        "10x 3 prime RNA-seq",
        "single donor",
    ]


def test_enrichment_rejects_protected_family_exclusion() -> None:
    family = FeatureFamilyEvidence(
        family="cellCycle",
        defaultExclude=False,
        evidenceId="assay:RNA:family:cellCycle",
    )
    inspection = AssayFeatureInspection(
        assay="RNA",
        species="unknown",
        families=[family],
        evidenceIds=["assay:RNA:species", family.evidenceId],
    )
    deps = DataEnrichmentDependencies(
        store=ReadOnlyStore(),
        assays=["RNA"],
        inspections={"RNA": inspection},
        evidenceIds={"assay:RNA:species", family.evidenceId},
    )
    report = DataEnrichmentReport(
        status="done",
        policies=[
            FeatureSelectionPolicy(
                assay="RNA",
                excludeFamilies=["cellCycle"],
                evidenceIds=[family.evidenceId],
            )
        ],
    )

    with pytest.raises(ValueError, match="protects by default"):
        validate_data_enrichment_report(deps, report)


def test_artificial_feature_requires_feature_specific_evidence() -> None:
    inspection = AssayFeatureInspection(
        assay="RNA",
        species="unknown",
        evidenceIds=["assay:RNA:species"],
    )
    feature_evidence = "assay:RNA:feature:ENSG00000111640"
    deps = DataEnrichmentDependencies(
        store=ReadOnlyStore(),
        assays=["RNA"],
        inspections={"RNA": inspection},
        confirmedFeatures={"RNA": {"GAPDH", "ENSG00000111640"}},
        evidenceIds={"assay:RNA:species", feature_evidence},
    )
    report = DataEnrichmentReport(
        status="done",
        policies=[
            FeatureSelectionPolicy(
                assay="RNA",
                artificialFeatures=["GAPDH"],
                evidenceIds=[feature_evidence],
            )
        ],
    )

    with pytest.raises(ValueError, match="feature-specific"):
        validate_data_enrichment_report(deps, report)

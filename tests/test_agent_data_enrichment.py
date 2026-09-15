"""Tests for the read-only data enrichment agent."""

from tests.agent_examples import example

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import BaseModel
from pydantic_ai import ModelRetry, UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

import scarf.agent.data_enrichment.agent as data_enrichment_agent_module
import scarf.agent.data_enrichment.tools as data_enrichment_tools
import scarf.agent.data_enrichment.validation as data_enrichment_validation
from scarf.agent.data_enrichment.characterization import FeatureCharacterization
from scarf.agent.data_enrichment import (
    AdtControlEvidence,
    AssayFeatureInspection,
    AssayFeatureInspectionBatch,
    AssayModalityEvidence,
    AtacCoordinateEvidence,
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
    HtoTagEvidence,
    StudyContextSummary,
    find_present_features,
    find_present_features_batch,
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
        StudyContextSummary,
        AdtControlEvidence,
        HtoTagEvidence,
        AtacCoordinateEvidence,
        AssayModalityEvidence,
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
        assert isinstance(example(model_type), model_type)
        assert all("_" not in field_name for field_name in model_type.model_fields)


def test_data_enrichment_agent_uses_only_read_tools_and_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scarf.agent.data_enrichment import tools as module

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


def test_data_enrichment_batches_grounded_multimodal_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scarf.agent.data_enrichment import tools as module

    assay_features = {
        "RNA": (
            ["ENSG00000198727", "ENSG00000111640"],
            ["MT-CYB", "GAPDH"],
        ),
        "peaks": (
            ["chr1:100-200", "not-a-coordinate"],
            ["chr1:100-200", "not-a-coordinate"],
        ),
        "proteins": (
            ["CD3", "mouse-igg1-control", "IgG1"],
            ["CD3", "Mouse IgG1 isotype control", "IgG1 antibody"],
        ),
        "hashes": (["HTO-1", "HTO-2"], ["sample one", "sample two"]),
        "guides": (["guide-1"], ["guide one"]),
    }

    class Table:
        def __init__(self, ids: list[str], names: list[str]) -> None:
            self.values = {"ids": ids, "names": names}

        def fetch_all(self, column: str) -> list[str]:
            return self.values[column]

    class Store:
        assay_names = list(assay_features)

        def __init__(self) -> None:
            self.assays = {
                name: SimpleNamespace(feats=Table(ids, names))
                for name, (ids, names) in assay_features.items()
            }

        def get_assay(self, assay_name: str) -> SimpleNamespace:
            return self.assays[assay_name]

        def summary(self) -> SimpleNamespace:
            assay_types = {
                "RNA": "RNA",
                "peaks": "ATAC",
                "proteins": "ADT",
                "hashes": "HTO",
                "guides": "CRISPR",
            }
            return SimpleNamespace(
                assays=[
                    SimpleNamespace(name=name, assay_type=assay_type)
                    for name, assay_type in assay_types.items()
                ]
            )

    def inspect_characterization(_store, **kwargs):
        assay_name = kwargs["assays"][0]
        ids, _names = assay_features[assay_name]
        assay_kind = {
            "RNA": "RNAassay",
            "peaks": "ATACassay",
            "proteins": "ADTassay",
            "hashes": "ADTassay",
            "guides": "Assay",
        }[assay_name]
        return FeatureCharacterization(
            status="done",
            assays=[
                {
                    "assay": assay_name,
                    "assayKind": assay_kind,
                    "identity": {"nFeatures": len(ids)},
                    "species": "unknown",
                    "speciesMethod": "notApplicable",
                    "speciesResolution": {"reason": "not an RNA decision"},
                    "families": [],
                    "exogenous": [],
                }
            ],
        )

    monkeypatch.setattr(module, "characterize_features", inspect_characterization)
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
        report = DataEnrichmentReport(
            status="done",
            studyContextSummary=StudyContextSummary(
                tissueReferences=["blood"],
                hypothesisReferences=["treatment changes immune states"],
            ),
            policies=[
                FeatureSelectionPolicy(
                    assay=assay_name,
                    evidenceIds=[f"assay:{assay_name}:species"],
                )
                for assay_name in assay_features
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

    context = "Human blood study tests whether treatment changes immune states."
    result = DataEnrichmentAgent(FunctionModel(reply)).run(
        Store(),
        context=DataEnrichmentContext(studyContext=context),
    )

    policies = {policy.assay: policy for policy in result.policies}
    assert policies["RNA"].assayModality == "RNA"
    assert policies["RNA"].graphEligible is True
    assert policies["peaks"].assayModality == "ATAC"
    assert policies["peaks"].peakCoordinateStatus == "partial"
    assert [item.featureId for item in policies["proteins"].exactControlFeatures] == [
        "mouse-igg1-control"
    ]
    assert [item.featureId for item in policies["hashes"].exactTagFeatures] == [
        "HTO-1",
        "HTO-2",
    ]
    assert policies["hashes"].demultiplexEligible is True
    assert policies["hashes"].graphEligible is False
    assert policies["guides"].assayModality == "unsupported"
    assert result.studyContextSummary.studyContext == context
    assert result.studyContextSummary.organismReferences == ["Human"]
    assert result.studyContextSummary.tissueReferences == ["blood"]
    assert result.studyContextSummary.hypothesisReferences == [
        "treatment changes immune states"
    ]
    assert policies["RNA"].tissueReferences == ["blood"]
    assert [call.name for call in result.toolCalls] == ["inspect_assay_features_batch"]


def test_data_enrichment_retries_hallucinated_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scarf.agent.data_enrichment import tools as module

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


def test_data_enrichment_fails_after_completed_inspection_without_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scarf.agent.data_enrichment import tools as module

    store = ReadOnlyStore()
    monkeypatch.setattr(
        module,
        "characterize_features",
        lambda *_args, **_kwargs: characterization(),
    )
    tool_retries: dict[str, int] = {}

    def unavailable_structured_output(**kwargs: object) -> None:
        deps = kwargs["deps"]
        assert isinstance(deps, DataEnrichmentDependencies)
        for tool in kwargs["tools"]:
            tool_retries[tool.name] = tool.max_retries
        asyncio.run(module.inspect_assay_features_batch(SimpleNamespace(deps=deps)))
        raise UnexpectedModelBehavior("structured output unavailable")

    monkeypatch.setattr(
        data_enrichment_agent_module,
        "run_agent_sync",
        unavailable_structured_output,
    )
    result = DataEnrichmentAgent(object()).run(
        store,
        context=DataEnrichmentContext(organismHint="human"),
    )

    assert result.status == "failed"
    assert result.runInfo.agentName == "data_enrichment_failed"
    assert result.policies == []
    assert result.unresolvedQuestions == []
    assert result.inspections[0].species == "unknown"
    assert "No scientific feature policy was selected" in result.limitations[0]
    assert tool_retries == {
        "inspect_assay_features_batch": 1,
        "find_present_features_batch": 1,
    }


def test_data_enrichment_preserves_validated_policy_uncertainty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scarf.agent.data_enrichment import tools as module
    from scarf.agent.types import AgentRunInfo

    store = ReadOnlyStore()
    monkeypatch.setattr(
        module,
        "characterize_features",
        lambda *_args, **_kwargs: characterization(),
    )

    def unresolved_structured_output(**kwargs: object) -> SimpleNamespace:
        deps = kwargs["deps"]
        assert isinstance(deps, DataEnrichmentDependencies)
        asyncio.run(module.inspect_assay_features_batch(SimpleNamespace(deps=deps)))
        return SimpleNamespace(
            output=DataEnrichmentReport(
                status="needsInput",
                unresolvedQuestions=[
                    "The objective does not resolve which response genes must be protected."
                ],
            ),
            runInfo=AgentRunInfo(agentName="data_enrichment", modelName="test-model"),
        )

    monkeypatch.setattr(
        data_enrichment_agent_module,
        "run_agent_sync",
        unresolved_structured_output,
    )
    result = DataEnrichmentAgent(object()).run(
        store,
        context=DataEnrichmentContext(organismHint="human"),
    )

    assert result.status == "needsInput"
    assert result.policies == []
    assert result.unresolvedQuestions == [
        "The objective does not resolve which response genes must be protected."
    ]
    assert result.runInfo.agentName == "data_enrichment"


def test_feature_lookup_cache_rejects_different_arguments() -> None:
    deps = DataEnrichmentDependencies(
        store=ReadOnlyStore(),
        assays=["RNA"],
    )
    run_context = SimpleNamespace(deps=deps)

    first = asyncio.run(
        find_present_features_batch(
            run_context,
            queries_by_assay={"RNA": ["MT-CYB"]},
        )
    )
    repeated = asyncio.run(
        find_present_features_batch(
            run_context,
            queries_by_assay={"RNA": ["MT-CYB"]},
        )
    )

    assert repeated is first
    with pytest.raises(ModelRetry, match="already completed"):
        asyncio.run(
            find_present_features_batch(
                run_context,
                queries_by_assay={"RNA": ["GAPDH"]},
            )
        )


def test_data_enrichment_tool_helpers_cover_resolution_edges() -> None:
    assert data_enrichment_tools._assay_modality(None, "RNAassay") == (
        "RNA",
        "RNAassay",
        "assayClass",
    )
    assert data_enrichment_tools._assay_modality(None, "ATACassay") == (
        "ATAC",
        "ATACassay",
        "assayClass",
    )
    assert data_enrichment_tools._assay_modality(None, "CustomAssay") == (
        "unsupported",
        "CustomAssay",
        "assayClass",
    )
    assert data_enrichment_tools._assay_modality(None, "") == (
        "unsupported",
        "Assay",
        "unknown",
    )
    assert data_enrichment_tools._valid_peak_coordinate("chr1:10") is False
    assert data_enrichment_tools._valid_peak_coordinate("chr1:start-20") is False
    controls = data_enrichment_tools._inspect_adt_features(
        "ADT",
        [("control-1", "IgG control")],
    )
    assert controls[0].matchedToken == "control"
    assert data_enrichment_tools._inspect_atac_features("ATAC", []).status == "invalid"
    assert (
        data_enrichment_tools._inspect_atac_features(
            "ATAC",
            ["chr1:10-20"],
        ).status
        == "valid"
    )
    future_tool = SimpleNamespace(name="future_tool")
    assert (
        data_enrichment_tools._prepare_data_enrichment_tool(
            SimpleNamespace(deps=DataEnrichmentDependencies()),
            future_tool,
        )
        is future_tool
    )


def test_find_present_features_reports_casefold_ambiguity_and_absence() -> None:
    class LookupFeatures:
        @staticmethod
        def fetch_all(column: str) -> list[str]:
            return {
                "ids": ["GENE1", "GENE2", "GAPDH"],
                "names": ["shared", "shared", "GAPDH"],
            }[column]

    store = SimpleNamespace(
        get_assay=lambda _name: SimpleNamespace(feats=LookupFeatures())
    )
    deps = DataEnrichmentDependencies(store=store, assays=["RNA"])

    result = asyncio.run(
        find_present_features(
            SimpleNamespace(deps=deps),
            assay_name="RNA",
            queries=["gapdh", "shared", "missing"],
        )
    )

    assert [item.status for item in result.results] == [
        "present",
        "ambiguous",
        "absent",
    ]
    assert deps.confirmedFeatures["RNA"] == {"GAPDH"}
    assert result.results[1].evidenceIds == []
    assert result.results[2].matches == []


def test_feature_lookup_tools_reject_invalid_requests() -> None:
    with pytest.raises(ModelRetry, match="datastore is unavailable"):
        asyncio.run(
            find_present_features(
                SimpleNamespace(
                    deps=DataEnrichmentDependencies(store=None, assays=["RNA"])
                ),
                assay_name="RNA",
                queries=["GAPDH"],
            )
        )
    deps = DataEnrichmentDependencies(store=ReadOnlyStore(), assays=["RNA"])
    context = SimpleNamespace(deps=deps)
    with pytest.raises(ModelRetry, match="requested assays"):
        asyncio.run(
            find_present_features(
                context,
                assay_name="ADT",
                queries=["CD3"],
            )
        )
    with pytest.raises(ModelRetry, match="between 1 and 50"):
        asyncio.run(
            find_present_features(
                context,
                assay_name="RNA",
                queries=[],
            )
        )
    with pytest.raises(ModelRetry, match="Unknown requested assays"):
        asyncio.run(
            find_present_features_batch(
                context,
                queries_by_assay={"ADT": ["CD3"]},
            )
        )
    with pytest.raises(ModelRetry, match="at least one assay"):
        asyncio.run(find_present_features_batch(context, queries_by_assay={}))
    with pytest.raises(ModelRetry, match="cannot be empty"):
        asyncio.run(
            find_present_features_batch(
                context,
                queries_by_assay={"RNA": []},
            )
        )
    with pytest.raises(ModelRetry, match="at most 50"):
        asyncio.run(
            find_present_features_batch(
                context,
                queries_by_assay={
                    "RNA": [f"gene-{index}" for index in range(51)],
                },
            )
        )
    with pytest.raises(ModelRetry, match="requested assays"):
        asyncio.run(
            data_enrichment_tools.inspect_assay_features(
                context,
                assay_name="ADT",
            )
        )
    empty = SimpleNamespace(
        deps=DataEnrichmentDependencies(store=ReadOnlyStore(), assays=[])
    )
    with pytest.raises(ModelRetry, match="No assays were requested"):
        asyncio.run(data_enrichment_tools.inspect_assay_features_batch(empty))


def test_assay_inspection_rejects_failed_or_modified_characterization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = SimpleNamespace(
        deps=DataEnrichmentDependencies(store=ReadOnlyStore(), assays=["RNA"])
    )
    monkeypatch.setattr(
        data_enrichment_tools,
        "characterize_features",
        lambda *_args, **_kwargs: FeatureCharacterization(
            status="failed",
            notes=["feature inspection failed"],
        ),
    )
    with pytest.raises(ModelRetry, match="feature inspection failed"):
        asyncio.run(
            data_enrichment_tools.inspect_assay_features(
                context,
                assay_name="RNA",
            )
        )

    modified = characterization()
    modified.assays[0]["defaultFeatureInventory"] = {"blacklist": "modified"}
    monkeypatch.setattr(
        data_enrichment_tools,
        "characterize_features",
        lambda *_args, **_kwargs: modified,
    )
    with pytest.raises(ModelRetry, match="exact default HVG blacklist"):
        asyncio.run(
            data_enrichment_tools.inspect_assay_features(
                context,
                assay_name="RNA",
            )
        )


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


def test_enrichment_rejects_duplicate_assay_policies() -> None:
    inspection = AssayFeatureInspection(
        assay="RNA",
        species="unknown",
        evidenceIds=["assay:RNA:species"],
    )
    deps = DataEnrichmentDependencies(
        store=ReadOnlyStore(),
        assays=["RNA"],
        inspections={"RNA": inspection},
        evidenceIds={"assay:RNA:species"},
    )
    policy = FeatureSelectionPolicy(
        assay="RNA",
        evidenceIds=["assay:RNA:species"],
    )

    with pytest.raises(ValueError, match="one policy for each assay"):
        validate_data_enrichment_report(
            deps,
            DataEnrichmentReport(status="done", policies=[policy, policy.model_copy()]),
        )


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


def test_study_context_summary_rejects_unbounded_or_ungrounded_references() -> None:
    context = DataEnrichmentContext(studyContext="Human lung study")
    with pytest.raises(ValueError, match="verbatim caller text"):
        data_enrichment_validation._ground_study_context_summary(
            context,
            StudyContextSummary(hypothesisReferences=["invented hypothesis"]),
        )
    with pytest.raises(ValueError, match="at most 12"):
        data_enrichment_validation._ground_study_context_summary(
            context,
            StudyContextSummary(
                tissueReferences=[f"tissue-{index}" for index in range(13)]
            ),
        )
    long_reference = "x" * 241
    with pytest.raises(ValueError, match="may not exceed 240"):
        data_enrichment_validation._ground_study_context_summary(
            DataEnrichmentContext(studyContext=long_reference),
            StudyContextSummary(analysisIntentReferences=[long_reference]),
        )


def test_feature_policy_validation_rejects_ungrounded_contracts() -> None:
    inspection = AssayFeatureInspection(
        assay="RNA",
        species="unknown",
        families=[
            FeatureFamilyEvidence(
                family="mitochondrial",
                defaultExclude=True,
                evidenceId="assay:RNA:family:mitochondrial",
            )
        ],
        evidenceIds=["assay:RNA:species"],
    )
    deps = DataEnrichmentDependencies(
        store=ReadOnlyStore(),
        context=DataEnrichmentContext(),
        assays=["RNA"],
        inspections={"RNA": inspection},
        evidenceIds={"assay:RNA:species"},
    )
    grounded = StudyContextSummary()
    validate = data_enrichment_validation._validate_feature_policy

    unsupported = FeatureSelectionPolicy.model_construct(
        assay="RNA",
        species="unsupported",
        evidenceIds=["assay:RNA:species"],
    )
    with pytest.raises(ValueError, match="unsupported species"):
        validate(deps, unsupported, grounded)
    with pytest.raises(ValueError, match="requires evidence IDs"):
        validate(deps, FeatureSelectionPolicy(assay="RNA"), grounded)
    with pytest.raises(ValueError, match="was not inspected"):
        validate(
            deps.model_copy(update={"inspections": {}}),
            FeatureSelectionPolicy(
                assay="RNA",
                evidenceIds=["assay:RNA:species"],
            ),
            grounded,
        )
    with pytest.raises(ValueError, match="must cite context evidence"):
        validate(
            deps,
            FeatureSelectionPolicy(
                assay="RNA",
                species="homo_sapiens",
                evidenceIds=["assay:RNA:species"],
            ),
            grounded,
        )
    with pytest.raises(ValueError, match="conflicts with inspected"):
        validate(
            deps.model_copy(
                update={
                    "inspections": {
                        "RNA": inspection.model_copy(update={"species": "homo_sapiens"})
                    }
                }
            ),
            FeatureSelectionPolicy(
                assay="RNA",
                species="mus_musculus",
                evidenceIds=["assay:RNA:species"],
            ),
            grounded,
        )
    with pytest.raises(ValueError, match="unobserved families"):
        validate(
            deps,
            FeatureSelectionPolicy(
                assay="RNA",
                protectFamilies=["cellCycle"],
                evidenceIds=["assay:RNA:species"],
            ),
            grounded,
        )
    with pytest.raises(ValueError, match="unknown evidence IDs"):
        validate(
            deps,
            FeatureSelectionPolicy(
                assay="RNA",
                evidenceIds=["evidence:unknown"],
            ),
            grounded,
        )


def test_feature_policy_accepts_feature_specific_context_evidence() -> None:
    inspection = AssayFeatureInspection(
        assay="RNA",
        species="unknown",
        evidenceIds=["assay:RNA:species"],
    )
    deps = DataEnrichmentDependencies(
        store=ReadOnlyStore(),
        context=DataEnrichmentContext(
            experimentalDetails=["ERCC-00002 spike-in"],
        ),
        assays=["RNA"],
        inspections={"RNA": inspection},
        confirmedFeatures={"RNA": {"ERCC-00002"}},
        evidenceIds={"assay:RNA:species", "context:experiment:0"},
    )
    policy = FeatureSelectionPolicy(
        assay="RNA",
        artificialFeatures=["ERCC-00002"],
        evidenceIds=["assay:RNA:species", "context:experiment:0"],
    )

    data_enrichment_validation._validate_feature_policy(
        deps,
        policy,
        StudyContextSummary(),
    )

    assert policy.artificialFeatures == ["ERCC-00002"]


def test_data_enrichment_report_rejects_incomplete_assay_inventory() -> None:
    inspection = AssayFeatureInspection(
        assay="RNA",
        species="unknown",
        evidenceIds=["assay:RNA:species"],
    )
    with pytest.raises(ValueError, match="Inspect every requested assay"):
        validate_data_enrichment_report(
            DataEnrichmentDependencies(store=ReadOnlyStore(), assays=["RNA"]),
            DataEnrichmentReport(
                status="needsInput",
                unresolvedQuestions=["Inspect the requested assays."],
            ),
        )
    deps = DataEnrichmentDependencies(
        store=ReadOnlyStore(),
        assays=["RNA"],
        inspections={"RNA": inspection},
        evidenceIds={"assay:RNA:species"},
    )
    with pytest.raises(ValueError, match="outside the requested set"):
        validate_data_enrichment_report(
            deps,
            DataEnrichmentReport(
                status="needsInput",
                unresolvedQuestions=["Resolve the assay mismatch."],
                policies=[
                    FeatureSelectionPolicy(
                        assay="ADT",
                        evidenceIds=["assay:RNA:species"],
                    )
                ],
            ),
        )
    with pytest.raises(ValueError, match="one policy for every requested assay"):
        validate_data_enrichment_report(
            deps.model_copy(update={"assays": ["RNA", "ADT"]}),
            DataEnrichmentReport(
                status="done",
                policies=[
                    FeatureSelectionPolicy(
                        assay="RNA",
                        evidenceIds=["assay:RNA:species"],
                    )
                ],
            ),
        )


def test_failed_enrichment_retains_partial_evidence_without_inventing_policy() -> None:
    incomplete = DataEnrichmentDependencies(store=ReadOnlyStore(), assays=["RNA"])
    for inspections in (
        {},
        {"RNA": AssayFeatureInspection(assay="RNA", species="unknown")},
    ):
        failed = data_enrichment_validation.failed_data_enrichment_report(
            incomplete.model_copy(update={"inspections": inspections}),
            error=RuntimeError("model failed"),
            model_name="test",
        )
        assert failed.status == "failed"
        assert failed.policies == []
        assert failed.inspections == list(inspections.values())
        assert "RuntimeError: model failed" in failed.limitations

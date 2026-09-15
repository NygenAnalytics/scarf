"""Objective-led metadata comparisons remain bounded and respect study units."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import Tool

from scarf.agent.experimental_context import tools
from scarf.agent.experimental_context import qc_evidence
from scarf.agent.cell_quality.profiles import project_registered_qc_profile
from scarf.agent.experimental_context.comparisons import (
    accept_capture_proposal,
    canonical_design_choices,
    combination_labels,
    compare_covariates,
    evaluate_proposals,
)
from scarf.agent.experimental_context.contracts import (
    CaptureProposal,
    CovariateCharacterization,
    CovariateProposal,
    ExperimentalContextDecision,
    ExperimentalContextDependencies,
)
from scarf.agent.experimental_context.study import build_study_contract
from scarf.agent.parameter_tuning import diagnostics, execution
from scarf.storage.refs import ArtifactRef


class _Cells:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.columns = list(frame.columns)

    def fetch(self, column: str) -> np.ndarray:
        return self.frame[column].to_numpy()


def _design() -> tuple[_Cells, CovariateCharacterization]:
    frame = pd.DataFrame(
        {
            "sample": [f"s{i}" for i in range(8)],
            "treatment": ["a", "a", "b", "b"] * 2,
            "time": ["early", "late", "early", "late"] * 2,
            "response": ["same", "different", "different", "same"] * 2,
            "age": np.arange(8, dtype=float),
        }
    )
    cells = _Cells(frame.loc[frame.index.repeat(3)].reset_index(drop=True))
    characterization = CovariateCharacterization(
        status="done",
        columns=[
            {
                "name": name,
                "kind": "continuous" if name == "age" else "categorical",
                "domain": "design" if name == "sample" else "biological",
            }
            for name in cells.columns
        ],
    )
    return cells, characterization


def _proposal(**changes: object) -> CovariateProposal:
    return CovariateProposal.model_validate(
        {
            "response": "response",
            "explanatoryColumns": ["treatment", "time"],
            "observationUnit": "sample",
            "rationale": "Preserve treatment-by-time structure relevant to the objective.",
            **changes,
        }
    )


def _deps(cells: _Cells) -> ExperimentalContextDependencies:
    return ExperimentalContextDependencies(
        cells=cells,
        store=SimpleNamespace(cells=cells),
        cellSelection=ArtifactRef(
            scope="datastore", kind="cell_selection", artifact_id="c" * 64
        ),
    )


def test_joint_explanation_detects_interaction_without_additive_alias() -> None:
    cells, characterization = _design()
    result = compare_covariates(
        cells, characterization, _proposal(), selection_identity={"id": "one"}
    )
    assert result.status == "computed"
    assert result.evidence["independentUnits"] == 8
    assert result.evidence["jointEstimability"]["coefficientEstimable"] is True
    assert result.evidence["jointGroupEstimability"]["coefficientEstimable"] is False
    assert all(
        item["value"] == 0 for item in result.evidence["singleAssociations"].values()
    )
    assert (
        result.evidence["jointAssociation"]["directionalMapping"]["nesting"] != "none"
    )


def test_conditional_comparison_keeps_each_stratum_and_support() -> None:
    cells, characterization = _design()
    result = compare_covariates(
        cells,
        characterization,
        _proposal(explanatoryColumns=["treatment"], conditionedOn="time"),
        selection_identity={},
    )
    assert result.status == "computed"
    assert len(result.evidence["strata"]) == 2
    assert all(row["independentUnits"] == 4 for row in result.evidence["strata"])


@pytest.mark.parametrize(
    "change,reason",
    [
        (
            {"explanatoryColumns": ["treatment"], "conditionedOn": "age"},
            "continuousConditioningIsUnsupported",
        ),
        (
            {"observationUnit": "response"},
            "observationAndIndependentUnitsMustBeDesignOrTechnical",
        ),
        ({"response": "missing"}, "unknownObservedColumn"),
    ],
)
def test_unsupported_explanations_are_explicit(
    change: dict[str, object], reason: str
) -> None:
    cells, characterization = _design()
    result = compare_covariates(
        cells, characterization, _proposal(**change), selection_identity={}
    )
    assert result.status == "unsupported"
    assert reason in result.reasons


def test_missingness_does_not_become_a_categorical_group() -> None:
    cells, characterization = _design()
    cells.frame.loc[:2, "response"] = None
    result = compare_covariates(
        cells, characterization, _proposal(), selection_identity={}
    )
    assert result.evidence["missingCells"] == 3
    assert result.evidence["missingCellsByColumn"]["response"] == 3
    assert result.evidence["independentUnits"] == 7
    assert result.status == "unsupported"


@pytest.mark.parametrize("declared", [False, True])
def test_biological_donor_can_be_an_explicit_independent_unit(declared: bool) -> None:
    cells, characterization = _design()
    cells.frame["donor"] = cells.frame["sample"].map(
        {f"s{i}": f"d{i % 4}" for i in range(8)}
    )
    cells.columns.append("donor")
    characterization.columns.append(
        {"name": "donor", "kind": "categorical", "domain": "biological"}
    )
    if declared:
        characterization.coefficients = [
            {
                "name": "response",
                "observationUnit": "sample",
                "independentUnit": "donor",
            }
        ]
    result = compare_covariates(
        cells,
        characterization,
        _proposal(explanatoryColumns=["treatment"], independentUnit="donor"),
        selection_identity={},
    )
    if declared:
        assert result.status == "computed"
        assert result.evidence["independentUnits"] == 4
        assert result.evidence["observationUnits"] == 8
        assert result.evidence["columnDomains"]["donor"] == "biological"
    else:
        assert result.status == "unsupported"
        assert "observationAndIndependentUnitsMustBeDesignOrTechnical" in result.reasons


@pytest.mark.parametrize("invalid", ["continuous", "withinDonor"])
def test_declared_biological_unit_keeps_kind_and_repeated_measure_guards(
    invalid: str,
) -> None:
    cells, characterization = _design()
    cells.frame["donor"] = cells.frame["sample"].map(
        {f"s{i}": f"d{i % 4}" for i in range(8)}
    )
    cells.columns.append("donor")
    characterization.columns.append(
        {
            "name": "donor",
            "kind": "continuous" if invalid == "continuous" else "categorical",
            "domain": "biological",
        }
    )
    characterization.coefficients = [
        {"name": "response", "observationUnit": "sample", "independentUnit": "donor"}
    ]
    if invalid == "withinDonor":
        cells.frame.loc[cells.frame["sample"] == "s4", "response"] = "different"
    result = compare_covariates(
        cells,
        characterization,
        _proposal(explanatoryColumns=["treatment"], independentUnit="donor"),
        selection_identity={},
    )
    assert result.status == "unsupported"
    assert (
        "observationAndIndependentUnitsMustBeCategorical"
        if invalid == "continuous"
        else "withinIndependentUnitComparisonsAreUnsupported"
    ) in result.reasons


def test_study_contract_preserves_unsupported_comparison_limitations() -> None:
    cells, characterization = _design()
    comparison = compare_covariates(
        cells,
        characterization,
        _proposal(observationUnit="response"),
        selection_identity={},
    )
    characterization.comparisons = [comparison]
    original = characterization.model_dump(mode="json")
    contract = build_study_contract(
        study_context="Independent donors with technical captures.",
        study_objective="Describe populations without unsupported associations.",
        experimental_result=SimpleNamespace(
            status="done",
            decision=ExperimentalContextDecision.get_blank(),
            batchSafety=[],
            characterization=characterization,
            notes=[],
        ),
    )
    limitation = next(
        value for value in contract.limitations if comparison.evidenceId in value
    )
    assert comparison.reasons[0] in limitation
    assert "no supported association or absence finding" in limitation
    assert characterization.model_dump(mode="json") == original


def test_two_round_limit_counts_retries_and_reuses_identical_proposals() -> None:
    cells, characterization = _design()
    deps = _deps(cells)
    proposal = _proposal(protectCombination=True)
    evaluate_proposals(deps, characterization, [proposal] * 8)
    assert len(deps.comparisons) == 1
    assert deps.protectedCombinations == [["time", "treatment"]]
    with pytest.raises(ValueError, match="eight initial and four"):
        evaluate_proposals(deps, characterization, [proposal] * 5)
    evaluate_proposals(deps, characterization, [proposal])
    assert len(deps.comparisons) == 1
    with pytest.raises(ValueError, match="two evidence rounds"):
        evaluate_proposals(deps, characterization, [])


def test_objective_questions_require_explicit_grounded_purpose_before_a_round() -> None:
    cells, characterization = _design()
    deps = _deps(cells)
    deps.studyObjective = "Explain treatment and time design coverage"
    with pytest.raises(ValueError, match="explicit purpose"):
        evaluate_proposals(deps, characterization, [_proposal()])
    assert deps.designRounds == 0
    proposal = _proposal(purpose="designCoverage", objectiveQuote=deps.studyObjective)
    with pytest.raises(ValueError, match="must remain essential"):
        evaluate_proposals(
            deps, characterization, [proposal.model_copy(update={"essential": False})]
        )
    assert deps.designRounds == 0
    evaluate_proposals(deps, characterization, [proposal])
    assert deps.designRounds == 1


def test_changed_declared_unit_pair_recomputes_one_current_comparison() -> None:
    cells, characterization = _design()
    cells.frame["donor"] = cells.frame["sample"]
    cells.columns.append("donor")
    characterization.columns.append(
        {"name": "donor", "domain": "biological", "kind": "categorical"}
    )
    deps = _deps(cells)
    proposal = _proposal(independentUnit="donor")
    evaluate_proposals(deps, characterization, [proposal])
    assert deps.comparisons[0].status == "unsupported"
    prior_id = deps.comparisons[0].evidenceId
    characterization.coefficients = [
        {"name": "response", "observationUnit": "sample", "independentUnit": "donor"}
    ]
    evaluate_proposals(deps, characterization, [proposal])
    assert len(deps.comparisons) == 1
    assert deps.comparisons[0].status == "computed"
    assert deps.comparisons[0].evidenceId != prior_id


def test_unsupported_explanation_does_not_discard_protected_biology() -> None:
    cells, characterization = _design()
    cells.frame.loc[:2, "response"] = None
    deps = _deps(cells)
    evaluate_proposals(deps, characterization, [_proposal(protectCombination=True)])
    assert deps.comparisons[0].status == "unsupported"
    assert deps.protectedCombinations == [["time", "treatment"]]


def test_proposals_cannot_exceed_three_columns() -> None:
    with pytest.raises(ValidationError, match="three distinct measured") as caught:
        _proposal(conditionedOn="age")
    message = str(caught.value)
    assert "response='response'" in message
    assert "explanatoryColumns=['treatment', 'time']" in message
    assert "conditionedOn='age'" in message
    assert "set conditionedOn=null" in message
    assert "joint explanation within strata is unsupported" in message


@pytest.mark.parametrize(
    ("explanatory", "condition"),
    [(["treatment"], None), (["treatment", "time"], None), (["treatment"], "time")],
)
def test_proposal_measurement_limit_excludes_observation_and_independent_units(
    explanatory: list[str], condition: str | None
) -> None:
    proposal = _proposal(
        explanatoryColumns=explanatory,
        conditionedOn=condition,
        observationUnit="sample",
        independentUnit="donor",
    )
    assert proposal.explanatoryColumns == explanatory
    assert proposal.conditionedOn == condition
    assert proposal.independentUnit == "donor"
    assert _proposal(independentUnit="sample").independentUnit == "sample"


@pytest.mark.parametrize(
    ("changes", "repeated"),
    [
        ({"explanatoryColumns": ["treatment", "treatment"]}, "treatment"),
        ({"explanatoryColumns": ["response"]}, "response"),
        (
            {"explanatoryColumns": ["treatment"], "conditionedOn": "response"},
            "response",
        ),
        (
            {"explanatoryColumns": ["treatment"], "conditionedOn": "treatment"},
            "treatment",
        ),
    ],
)
def test_proposal_duplicate_errors_identify_the_repeated_measurement(
    changes: dict[str, object], repeated: str
) -> None:
    with pytest.raises(
        ValidationError, match="Comparison columns must be distinct"
    ) as caught:
        _proposal(**changes)
    message = str(caught.value)
    assert f"repeated columns ['{repeated}']" in message
    assert "A column cannot explain itself" in message
    assert "at most three" not in message


@pytest.mark.parametrize(
    ("invalid_changes", "correction_hint"),
    [
        ({"conditionedOn": "age"}, "set conditionedOn=null"),
        ({"explanatoryColumns": ["response", "time"]}, "repeated columns ['response']"),
    ],
)
def test_design_tool_schema_and_retry_correct_proposals_before_computation(
    monkeypatch: pytest.MonkeyPatch,
    invalid_changes: dict[str, object],
    correction_hint: str,
) -> None:
    cells, characterization = _design()
    batch_columns = ["batch_a", "batch_b"]
    for batch, values in zip(batch_columns, ["treatment", "time"], strict=True):
        cells.frame[batch] = cells.frame[values]
        cells.columns.append(batch)
        characterization.columns.append(
            {"name": batch, "kind": "categorical", "domain": "technical"}
        )
    deps = _deps(cells)
    deps.characterization = characterization
    scans: list[dict[str, object]] = []
    safety_columns: list[list[str]] = []
    batch_safety = tools._batch_safety_evidence

    def characterize(*_args: object, **kwargs: object) -> CovariateCharacterization:
        scans.append(kwargs)
        return characterization

    def record_batch_safety(*args: object, **kwargs: object) -> object:
        safety_columns.append(kwargs["batch_columns"])
        return batch_safety(*args, **kwargs)

    monkeypatch.setattr(tools, "characterize_covariates", characterize)
    monkeypatch.setattr(tools, "_offered_qc_profiles", lambda *_args: [])
    monkeypatch.setattr(tools, "_batch_safety_evidence", record_batch_safety)
    requests = 0

    async def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal requests
        schema = info.function_tools[0].parameters_json_schema
        proposal_schema = schema["$defs"]["CovariateProposal"]
        properties = proposal_schema["properties"]
        assert (
            "Two explanatory columns plus conditioning are unsupported"
            in proposal_schema["description"]
        )
        assert (
            "conditionedOn must be null"
            in properties["explanatoryColumns"]["description"]
        )
        assert "does not count" in properties["observationUnit"]["description"]
        assert "does not count" in properties["independentUnit"]["description"]
        assert "must differ" in properties["response"]["description"]
        request = requests
        requests += 1
        if request == 1:
            retry_parts = [
                part
                for message in messages
                for part in message.parts
                if isinstance(part, RetryPromptPart)
            ]
            assert correction_hint in str(retry_parts[-1].content)
            assert scans == []
            assert deps.designRounds == 0
        if request < 2:
            proposal = _proposal().model_dump()
            if request == 0:
                proposal.update(invalid_changes)
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="analyze_experimental_design",
                        args={
                            "column_domains": {
                                batch: "technical" for batch in batch_columns
                            },
                            "coefficients_of_interest": [],
                            "units_of_inference": {},
                            "batch_columns": batch_columns,
                            "proposals": [proposal],
                        },
                    )
                ]
            )
        return ModelResponse(parts=[TextPart("Comparison evidence computed.")])

    agent = Agent(
        FunctionModel(reply),
        deps_type=ExperimentalContextDependencies,
        tools=[
            Tool(
                tools.model_evidence_tool(tools.analyze_experimental_design),
                max_retries=3,
            )
        ],
    )
    result = agent.run_sync("Compare the observed study design.", deps=deps)
    assert result.output == "Comparison evidence computed."
    assert requests == 3
    assert len(scans) == 1
    assert deps.designRounds == 1
    assert deps.toolCalls == ["analyze_experimental_design"]
    assert safety_columns == [batch_columns]
    assert len(deps.comparisons) == 1
    assert deps.comparisons[0].status == "computed"
    assert deps.comparisons[0].proposal == _proposal()


def test_combinations_preserve_typed_values_and_reject_missing() -> None:
    cells = _Cells(
        pd.DataFrame({"a": ["x,y", "x", 1, "1"], "b": ["z", "y,z", "a", "a"]})
    )
    assert len(np.unique(combination_labels(cells, ["a", "b"]))) == 4
    cells.frame.loc[0, "b"] = None
    with pytest.raises(ValueError, match="missing"):
        combination_labels(cells, ["a", "b"])


def test_capture_and_reference_selection_requires_verbatim_provenance() -> None:
    cells, characterization = _design()
    deps = _deps(cells)
    deps.studyContext = (
        "sample identifies the physical capture. s0 and s1 are reference captures."
    )
    proposal = CaptureProposal(
        column="sample",
        provenanceQuote="sample identifies the physical capture.",
        referenceCaptures=["s0", "s1"],
        referenceProvenanceQuote="s0 and s1 are reference captures.",
    )
    accept_capture_proposal(deps, characterization, proposal)
    assert deps.directions == {}
    choices = canonical_design_choices(deps, ExperimentalContextDecision())
    assert choices["physicalCaptureColumn"] == "sample"
    assert choices["pooledReferenceCaptures"] == ["s0", "s1"]
    with pytest.raises(ValueError, match="exact study quote"):
        accept_capture_proposal(_deps(cells), characterization, proposal)
    with pytest.raises(ValueError, match="Reference captures require"):
        accept_capture_proposal(
            deps,
            characterization,
            proposal.model_copy(
                update={"referenceProvenanceQuote": "Made up controls"}
            ),
        )


def test_final_decision_cannot_invent_a_protected_combination() -> None:
    cells, _ = _design()
    deps = _deps(cells)
    decision = ExperimentalContextDecision(
        protectedCombinations=[["time", "treatment"]]
    )
    with pytest.raises(ValueError, match="evaluated"):
        canonical_design_choices(deps, decision)


def test_continuous_protection_is_explicit_without_changing_design_license() -> None:
    cells, characterization = _design()
    deps = _deps(cells)
    deps.characterization = characterization
    decision = ExperimentalContextDecision(coefficientsOfInterest=["age"])
    choices = canonical_design_choices(deps, decision)
    assert choices["unsupportedProtection"] == ["age"]
    assert "batchCorrection" not in choices


def test_qc_retention_checks_joint_groups_beside_marginal_groups() -> None:
    cells, characterization = _design()
    characterization.coefficients = [{"name": "treatment"}, {"name": "time"}]
    deps = _deps(cells)
    deps.protectedCombinations = [["treatment", "time"]]
    keep = ~((cells.fetch("treatment") == "a") & (cells.fetch("time") == "early"))
    mito = np.where(keep, np.linspace(1, 3, len(keep)), 80.0)
    projection = project_registered_qc_profile(
        "globalMad5",
        values_by_metric={"RNA_percentMito": mito},
        active=np.ones(len(keep), dtype=bool),
    )
    result = qc_evidence._registered_profile_evidence(
        projection,
        deps=deps,
        characterization=characterization,
        driver=("RNA", "RNA"),
        active=np.ones(len(keep), dtype=bool),
        values_by_attr={"RNA_percentMito": mito},
        metadata_attributes=["RNA_percentMito"],
        artifact_metrics=[],
        metric_sources=[],
        source_concordance=[],
        sample_column=None,
        sample_artifact=None,
        capture_column=None,
        capture_artifact=None,
        capture_labels=None,
        pooled_reference_captures=None,
        active_cells=len(keep),
        comparison_source=None,
    )
    assert all(
        count > 0
        for counts in result.retainedCellsByColumn.values()
        for count in counts.values()
    )
    assert any(
        count == 0
        for counts in result.retainedCellsByCombination.values()
        for count in counts.values()
    )
    assert any(
        reason.startswith("combination:") for reason in result.unsafeRetentionGroups
    )


@pytest.mark.parametrize("independent_units", [5, 8])
def test_interaction_confounding_blocks_only_the_matching_batch_set(
    monkeypatch: pytest.MonkeyPatch,
    independent_units: int,
) -> None:
    cells, characterization = _design()
    cells.frame = cells.frame.iloc[: independent_units * 3].copy()
    for record in characterization.columns:
        if record["name"] in {"treatment", "time"}:
            record["domain"] = "technical"
    characterization.coefficients = [
        {
            "name": "response",
            "kind": "categorical",
            "observationUnit": "sample",
            "scope": "betweenUnit",
        }
    ]
    characterization.confounding = [
        {
            "coefficient": "response",
            "observationUnit": "sample",
            "pairs": [{"technical": "treatment"}, {"technical": "time"}],
        }
    ]
    deps = _deps(cells)
    evaluate_proposals(deps, characterization, [_proposal()])
    if independent_units == 5:
        assert deps.comparisons[0].status == "unsupported"
        assert deps.comparisons[0].reasons == ["unsupportedJointAssociation"]
        assert (
            deps.comparisons[0].evidence["jointGroupEstimability"][
                "coefficientEstimable"
            ]
            is False
        )
    monkeypatch.setattr(
        tools,
        "reduce_observation_units",
        lambda _cells, unit, columns, **_kwargs: cells.frame.groupby(unit).first()[
            columns
        ],
    )
    joint = tools._batch_safety_evidence(
        deps,
        characterization,
        coefficients=["response"],
        batch_columns=["treatment", "time"],
    )
    single = tools._batch_safety_evidence(
        deps, characterization, coefficients=["response"], batch_columns=["treatment"]
    )
    assert joint[0].status == "unsafe"
    assert single[0].status == "safe"


def test_tool_rejects_oversized_batch_before_scanning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cells, _ = _design()
    deps = _deps(cells)
    monkeypatch.setattr(
        tools,
        "characterize_covariates",
        lambda *_args, **_kwargs: pytest.fail("must reject before metadata scan"),
    )
    with pytest.raises(ModelRetry, match="eight initial"):
        asyncio.run(
            tools.analyze_experimental_design(
                SimpleNamespace(deps=deps),
                column_domains={},
                coefficients_of_interest=[],
                units_of_inference={},
                batch_columns=[],
                proposals=[_proposal()] * 9,
            )
        )


def test_pca_associations_use_covariate_kind_independently_of_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = np.arange(12, dtype=float)
    coordinates = np.column_stack([values, np.tile([1.0, -1.0], 6)])
    monkeypatch.setattr(
        diagnostics,
        "_aligned_metadata_values",
        lambda _store, _selection, column: (
            values % 2 if column == "qc_code" else values
        ),
    )
    support: dict[str, object] = {}
    scores = diagnostics._covariate_associations(
        None,
        None,
        coordinates,
        ["age", "technical_age", "qc_code"],
        ["protected", "technical", "qc"],
        {"age": "continuous", "technical_age": "continuous", "qc_code": "categorical"},
        support,
    )
    assert scores[0, 0] == pytest.approx(1.0)
    assert scores[0, 1] < 0.2
    np.testing.assert_array_equal(scores[0], scores[1])
    assert scores[2, 1] == pytest.approx(1.0)
    assert support["age"]["kind"] == "continuous"


def test_pca_numeric_association_uses_complete_rows_without_loading_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinates = np.arange(10, dtype=float).reshape(5, 2)

    class BoundedCoordinates:
        shape = coordinates.shape

        def __array__(self, *_args: object, **_kwargs: object) -> np.ndarray:
            raise AssertionError("Coordinates must be read in bounded slices")

        def __getitem__(self, rows: slice) -> np.ndarray:
            assert rows.stop - rows.start <= 65_536
            return coordinates[rows]

    values = np.asarray([0.0, 1.0, np.nan, 3.0, np.inf])
    monkeypatch.setattr(diagnostics, "_aligned_metadata_values", lambda *_args: values)
    support: dict[str, object] = {}
    scores = diagnostics._covariate_associations(
        None,
        None,
        BoundedCoordinates(),
        ["age"],
        ["protected"],
        {"age": "continuous"},
        support,
    )
    np.testing.assert_allclose(scores, 1.0)
    assert support["age"]["completeRows"] == 3
    assert support["age"]["missingRows"] == 2


def test_metric_fingerprint_changes_when_only_missing_mask_changes() -> None:
    values = np.asarray([1.0, 2.0, 3.0])
    missing = np.asarray([False, False, False])
    metadata = SimpleNamespace(
        N=3,
        _get_array=lambda _column: values,
        default_block_rows=lambda _column: 2,
        _get_missing_mask_array=lambda _column: missing,
    )
    first = execution._metadata_column_fingerprint(metadata, "age")
    missing[1] = True
    assert execution._metadata_column_fingerprint(metadata, "age") != first
    missing[1] = False
    assert execution._metadata_column_fingerprint(metadata, "age") == first


def test_single_cluster_keeps_diagnostics_without_running_marker_contrasts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scarf.agent.parameter_tuning.contracts import (
        ArtifactRecord,
        ParameterCandidateEvaluation,
    )
    from scarf.agent.types import ArtifactReferenceModel

    graph = ArtifactRef(
        scope="assay", assay="RNA", kind="connectivity_map", artifact_id="a" * 64
    )
    clusters = ArtifactRef(
        scope="assay", assay="RNA", kind="cluster_labels", artifact_id="b" * 64
    )
    selection = ArtifactRef(
        scope="datastore", kind="cell_selection", artifact_id="c" * 64
    )
    labels = np.zeros(20, dtype=int)
    evaluation = ParameterCandidateEvaluation(
        candidateId="single",
        status="done",
        eligible=True,
        cellSelection=ArtifactReferenceModel.from_artifact_ref(selection),
        artifacts={
            "connectivityMap": ArtifactRecord.from_ref(graph),
            "clusters": ArtifactRecord.from_ref(clusters),
        },
    )
    monkeypatch.setattr(
        diagnostics,
        "_selected_feature_names",
        lambda *_args: (np.arange(3), np.asarray(["A", "B", "C"])),
    )
    monkeypatch.setattr(diagnostics, "_cluster_labels", lambda *_args: labels)
    monkeypatch.setattr(
        diagnostics, "_subsample_partition_stability", lambda *_args: 1.0
    )

    def no_markers(*_args, **_kwargs):
        raise AssertionError("A single cluster has no marker contrast")

    store = SimpleNamespace(
        run_leiden_clustering=lambda *_args, **_kwargs: clusters,
        load_graph=lambda *_args: object(),
        run_marker_search=no_markers,
        cells=SimpleNamespace(columns=[]),
    )
    (result,) = diagnostics.augment_cluster_evaluations(
        store,
        [evaluation],
        marker_assay="RNA",
        marker_features=selection,
        independent_unit_columns=[],
        technical_columns=[],
    )
    assert not result.eligible
    assert result.metrics.markerCoherence is None
    assert result.metrics.markerSpecificityMedian is None
    assert result.metrics.markerFamilyEnrichment == {}
    assert "markerTable" not in result.artifacts
    assert result.metrics.seedStability == 1.0
    assert result.metrics.subsampleStability == 1.0
    assert (
        "Marker contrasts require at least two populated clusters"
        in result.eligibilityReasons
    )

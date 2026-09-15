"""Representative agent values belong to test fixtures, not production APIs."""


def example(model):
    """Construct a fixture through its closest registered model factory."""
    for base in model.__mro__:
        factory = _FACTORIES.get(f"{base.__module__}.{base.__name__}")
        if factory is not None:
            return factory(model)
    return model.get_blank()


def _example_0_BiologicalContext(cls):
    return cls(
        organism="Homo sapiens",
        studyContext="Human lung samples were profiled after drug or vehicle treatment.",
        tissue="lung",
        cellTypeReferences=["alveolar macrophage", "T cell"],
        experimentalDetails=["drug and vehicle groups"],
        treatmentQuestion="Which populations respond selectively to treatment?",
    )


def _example_1_ConditionClusterSummary(cls):
    return cls(
        condition="treated",
        clusterId="3",
        nSamples=4,
        meanFraction=0.18,
        minFraction=0.12,
        maxFraction=0.25,
        cellCount=180,
        evidenceId="composition:RNA_cluster:condition:treated:cluster:3",
    )


def _example_2_ClusterCompositionEvidence(cls):
    from scarf.agent.biological_interpretation.contracts import (
        ArtifactReferenceModel,
        ConditionClusterSummary,
    )

    summary = example(ConditionClusterSummary)
    reference_summary = ConditionClusterSummary(
        condition="control",
        clusterId=summary.clusterId,
        nSamples=4,
        meanFraction=0.11,
        minFraction=0.08,
        maxFraction=0.15,
        cellCount=110,
        evidenceId="composition:RNA_cluster:condition:control:cluster:3",
    )
    return cls(
        clusterArtifact=ArtifactReferenceModel(
            assay="RNA", kind="cluster_labels", artifactId="b" * 64
        ),
        cellSelection=ArtifactReferenceModel(
            scope="datastore", assay=None, kind="cell_selection", artifactId="c" * 64
        ),
        totalCells=1000,
        clusterCounts={"0": 520, "1": 300, "3": 180},
        sampleColumn="sample",
        conditionColumn="treatment",
        conditionSummaries=[reference_summary, summary],
        evidenceIds=[
            "composition:RNA_cluster:counts",
            reference_summary.evidenceId,
            summary.evidenceId,
        ],
    )


def _example_3_MarkerFeature(cls):
    return cls(
        featureId="ENSG00000173372",
        featureName="C1QA",
        featureIndex=123,
        score=0.83,
        foldChange=3.4,
        fractionExpressed=0.76,
        fractionExpressedRest=0.18,
        auc=0.91,
        adjustedPvalue=0.001,
    )


def _example_4_ClusterMarkerEvidence(cls):
    from scarf.agent.biological_interpretation.contracts import (
        ArtifactReferenceModel,
        MarkerFeature,
    )

    return cls(
        clusterId="3",
        markers=[example(MarkerFeature)],
        markerArtifact=ArtifactReferenceModel(
            assay="RNA", kind="marker_table", artifactId="a" * 64
        ),
        evidenceId="markers:RNA_cluster:cluster:3",
    )


def _example_5_ClusterMarkerBatchEvidence(cls):
    from scarf.agent.biological_interpretation.contracts import ClusterMarkerEvidence

    cluster = example(ClusterMarkerEvidence)
    return cls(clusters=[cluster], evidenceIds=[cluster.evidenceId])


def _example_6_ClusterInterpretation(cls):
    return cls(
        clusterId="3",
        proposedIdentity="alveolar macrophage-like",
        identityIsHypothesis=True,
        confidence="medium",
        rationale="Observed marker pattern is consistent with the proposed identity.",
        evidenceIds=["markers:RNA_cluster:cluster:3"],
    )


def _example_7_TreatmentObservation(cls):
    return cls(
        clusterId="3",
        referenceCondition="control",
        comparisonCondition="treated",
        direction="higher",
        observation="Cluster 3 has a higher mean fraction in treated samples.",
        evidenceIds=[
            "composition:RNA_cluster:condition:control:cluster:3",
            "composition:RNA_cluster:condition:treated:cluster:3",
        ],
    )


def _example_8_FollowUpRecommendation(cls):
    return cls(
        question="Is the abundance difference reproducible across donors?",
        operation="sample-level differential abundance",
        rationale="Current evidence is descriptive and requires independent replicates.",
        requiredInputs=["sample", "condition", "donor"],
        evidenceIds=[
            "composition:RNA_cluster:condition:control:cluster:3",
            "composition:RNA_cluster:condition:treated:cluster:3",
        ],
    )


def _example_9_BiologicalInterpretationNeedsInput(cls):
    return cls(
        question="Provide an exact marker artifact or authorize marker search.",
        requiredInputs=["markerArtifact"],
    )


def _example_10_BiologicalInterpretationReport(cls):
    from scarf.agent.biological_interpretation.contracts import (
        ClusterCompositionEvidence,
        ClusterInterpretation,
        ClusterMarkerEvidence,
        FollowUpRecommendation,
        TreatmentObservation,
    )

    interpretation = example(ClusterInterpretation)
    observation = example(TreatmentObservation)
    follow_up = example(FollowUpRecommendation)
    return cls(
        status="done",
        clusterInterpretations=[interpretation],
        treatmentObservations=[observation],
        followUps=[follow_up],
        clusterArtifact=example(ClusterCompositionEvidence).clusterArtifact,
        markerArtifact=example(ClusterMarkerEvidence).markerArtifact,
        graphAssay="RNA",
        markerAssay="RNA",
        evidenceIds=sorted(
            {
                *interpretation.evidenceIds,
                *observation.evidenceIds,
                *follow_up.evidenceIds,
            }
        ),
        limitations=[
            "Cell identities remain hypotheses until independently validated."
        ],
        stopReason="The requested clusters were reviewed.",
    )


def _example_11_BiologicalInterpretationDependencies(cls):
    return cls(
        cluster=object(),
        fromAssay="RNA",
        graphAssay="RNA",
        markerAssay="RNA",
        markerAssayType="RNA",
        sampleColumn="sample",
        conditionColumn="treatment",
    )


def _example_12_AgentRunConfig(cls):
    return cls(requestLimit=9, toolCallLimit=5, outputTokenLimit=2048)


def _example_13_FeatureCharacterization(cls):
    return cls(
        status="done",
        notes=["Feature identity and families were characterized."],
        assays=[{"assay": "RNA", "species": "homo_sapiens"}],
    )


def _example_14_DataEnrichmentContext(cls):
    return cls(
        studyContext="Single-cell profiling of treated lung tissue",
        studyObjective="Discover stable populations while preserving treatment effects.",
        organismHint="human",
        tissueReferences=["lung"],
        cellTypeReferences=["alveolar macrophage", "T cell"],
        experimentalDetails=["CRISPR perturbation", "10x 3 prime RNA-seq"],
    )


def _example_15_StudyContextSummary(cls):
    return cls(
        studyContext="Single-cell profiling of treated human lung tests whether treatment changes alveolar macrophage states.",
        studyObjective="Discover populations while preserving the treatment comparison.",
        organismReferences=["human"],
        tissueReferences=["lung"],
        cellTypeReferences=["alveolar macrophage"],
        experimentalReferences=["treated"],
        hypothesisReferences=["treatment changes alveolar macrophage states"],
        analysisIntentReferences=["Single-cell profiling"],
        evidenceIds=["context:study"],
    )


def _example_16_AdtControlEvidence(cls):
    return cls(
        featureId="Mouse-IgG1-Control",
        featureName="Mouse IgG1 isotype control",
        matchedToken="isotype",
        evidenceId="assay:ADT:adtControl:Mouse-IgG1-Control",
    )


def _example_17_HtoTagEvidence(cls):
    return cls(
        featureId="HTO-1",
        featureName="Sample tag 1",
        evidenceId="assay:HTO:htoTag:HTO-1",
    )


def _example_18_AtacCoordinateEvidence(cls):
    return cls(
        status="valid",
        totalFeatures=2,
        validFeatures=2,
        validExamples=["chr1:100-200", "chr2:300-450"],
        evidenceId="assay:ATAC:atacCoordinates",
    )


def _example_19_AssayModalityEvidence(cls):
    from scarf.agent.data_enrichment.contracts import AdtControlEvidence

    control = example(AdtControlEvidence)
    return cls(
        assayType="ADT",
        modality="ADT",
        typeSource="persisted",
        graphEligible=True,
        markerEligible=True,
        adtControls=[control],
        totalObservedFeatures=20,
        reportedFeatures=1,
        evidenceIds=["assay:ADT:modality", control.evidenceId],
    )


def _example_20_FeatureFamilyEvidence(cls):
    return cls(
        family="mitochondrial",
        species="homo_sapiens",
        method="chromosome",
        count=2,
        examples=["MT-CO1", "MT-CYB"],
        defaultExclude=True,
        evidenceId="assay:RNA:family:mitochondrial",
    )


def _example_21_DefaultHvgFamilyEvidence(cls):
    return cls(
        family="mitochondrial",
        pattern="^MT-",
        count=2,
        examples=["MT-CO1", "MT-CYB"],
        evidenceId="assay:RNA:scarfDefaultHvg:family:mitochondrial",
    )


def _example_22_RnaFeatureInventoryEvidence(cls):
    from scarf.features.variability import DEFAULT_HVG_BLACKLIST
    from scarf.agent.data_enrichment.contracts import DefaultHvgFamilyEvidence

    family = example(DefaultHvgFamilyEvidence)
    evidence_id = "assay:RNA:scarfDefaultHvg:combined"
    return cls(
        totalFeatures=20000,
        blacklist=DEFAULT_HVG_BLACKLIST,
        matchCount=2,
        examples=["MT-CO1", "MT-CYB"],
        families=[family],
        evidenceId=evidence_id,
        evidenceIds=[evidence_id, family.evidenceId],
    )


def _example_23_ExogenousFeatureEvidence(cls):
    return cls(
        featureId="ERCC-00002",
        featureName="ERCC-00002",
        score=4,
        classification="potentialExogenous",
        evidenceId="assay:RNA:exogenous:ERCC-00002",
    )


def _example_24_AssayFeatureInspection(cls):
    from scarf.agent.data_enrichment.contracts import (
        AssayModalityEvidence,
        FeatureFamilyEvidence,
        RnaFeatureInventoryEvidence,
    )

    family = example(FeatureFamilyEvidence)
    default_inventory = example(RnaFeatureInventoryEvidence)
    modality = AssayModalityEvidence(
        assayType="RNA",
        modality="RNA",
        typeSource="persisted",
        graphEligible=True,
        markerEligible=True,
        totalObservedFeatures=20000,
        evidenceIds=["assay:RNA:modality"],
    )
    return cls(
        assay="RNA",
        assayKind="RNAassay",
        identity={"nFeatures": 20000, "nDuplicateIds": 0},
        species="homo_sapiens",
        speciesMethod="ensemblPrefix",
        speciesReason="Most feature IDs carry the ENSG prefix",
        families=[family],
        defaultFeatureInventory=default_inventory,
        modalityEvidence=modality,
        evidenceIds=[
            "assay:RNA:identity",
            "assay:RNA:species",
            family.evidenceId,
            *default_inventory.evidenceIds,
            *modality.evidenceIds,
        ],
    )


def _example_25_AssayFeatureInspectionBatch(cls):
    from scarf.agent.data_enrichment.contracts import AssayFeatureInspection

    inspection = example(AssayFeatureInspection)
    return cls(inspections=[inspection], evidenceIds=list(inspection.evidenceIds))


def _example_26_FeatureReference(cls):
    return cls(featureId="ENSG00000198727", featureName="MT-CYB")


def _example_27_FeatureMatch(cls):
    from scarf.agent.data_enrichment.contracts import FeatureReference

    return cls(
        query="MT-CYB",
        status="present",
        matches=[example(FeatureReference)],
        evidenceIds=["assay:RNA:feature:ENSG00000198727"],
    )


def _example_28_FeatureLookupResult(cls):
    from scarf.agent.data_enrichment.contracts import FeatureMatch

    match = example(FeatureMatch)
    return cls(assay="RNA", results=[match], evidenceIds=list(match.evidenceIds))


def _example_29_FeatureLookupBatch(cls):
    from scarf.agent.data_enrichment.contracts import FeatureLookupResult

    lookup = example(FeatureLookupResult)
    return cls(lookups=[lookup], evidenceIds=list(lookup.evidenceIds))


def _example_30_FeatureSelectionPolicy(cls):
    return cls(
        assay="RNA",
        species="homo_sapiens",
        organismName="human",
        speciesConfidence="high",
        speciesRationale="Gene IDs and study context agree",
        excludeFamilies=["mitochondrial", "ribosomal"],
        protectFamilies=["cellCycle", "sex"],
        artificialFeatures=["ERCC-00002"],
        tissueReferences=["lung"],
        cellTypeReferences=["alveolar macrophage"],
        experimentalReferences=["ERCC spike-in"],
        assayType="RNA",
        assayModality="RNA",
        graphEligible=True,
        markerEligible=True,
        rationale="Use technical families for feature-selection exclusions",
        evidenceIds=["assay:RNA:species", "assay:RNA:family:mitochondrial"],
    )


def _example_31_DataEnrichmentToolCall(cls):
    return cls(
        name="inspect_assay_features",
        assay="RNA",
        evidenceIds=["assay:RNA:identity", "assay:RNA:species"],
    )


def _example_32_DataEnrichmentReport(cls):
    from scarf.agent.data_enrichment.contracts import (
        AgentRunInfo,
        AssayFeatureInspection,
        DataEnrichmentToolCall,
        FeatureSelectionPolicy,
        StudyContextSummary,
    )

    policy = example(FeatureSelectionPolicy)
    inspection = example(AssayFeatureInspection)
    return cls(
        status="done",
        policies=[policy],
        inspections=[inspection],
        studyContextSummary=example(StudyContextSummary),
        evidenceIds=list(policy.evidenceIds),
        toolCalls=[example(DataEnrichmentToolCall)],
        runInfo=example(AgentRunInfo),
    )


def _example_33_DataEnrichmentDependencies(cls):
    from scarf.agent.data_enrichment.contracts import DataEnrichmentContext, Path

    return cls(
        context=example(DataEnrichmentContext),
        assays=["RNA"],
        cacheDir=Path("/tmp/scarf-gene-reference"),
        allowDownload=False,
        evidenceIds={"context:organism", "context:tissue:0"},
    )


def _example_34_CovariateCharacterization(cls):
    from scarf.agent.experimental_context.contracts import ArtifactReferenceModel

    return cls(
        status="done",
        cellSelection=ArtifactReferenceModel(
            scope="datastore", kind="cell_selection", artifactId="c" * 64
        ),
        notes=["Cell covariates and confounding were characterized."],
        columns=[{"name": "batch", "domain": "technical"}],
    )


def _example_35_InferenceUnit(cls):
    return cls(observationUnit="sample", independentUnit="donor")


def _example_36_BatchCorrectionPlan(cls):
    return cls(
        action="evaluateHarmony",
        batchColumns=["batch"],
        preserveColumns=["cell_type", "treatment"],
        metricsRequired=["iLISI", "cLISI", "graphConnectivity"],
        rationale="Batch is technical and crossed with treatment, so compare an exact Harmony candidate while protecting biological labels.",
        evidenceIds=[
            "column:batch",
            "estimability:treatment",
            "batchEstimability:treatment:batch",
        ],
    )


def _example_37_NamedArtifactSource(cls):
    from scarf.agent.experimental_context.contracts import ArtifactReferenceModel

    return cls(
        name="RNA_percentMito",
        artifact=ArtifactReferenceModel(
            assay="RNA", kind="quality_metric", artifactId="1" * 64
        ),
    )


def _example_38_CellQcProfileEvidence(cls):
    from scarf.agent.experimental_context.contracts import NamedArtifactSource

    return cls(
        profileId="cellQc:RNA:globalMad5",
        action="registeredMad",
        registeredProfile="globalMad5",
        driverAssay="RNA",
        driverAssayType="RNA",
        attributes=["RNA_nCounts", "RNA_nFeatures"],
        artifactMetrics=[example(NamedArtifactSource)],
        parameters={"nMads": 5.0},
        activeCells=100,
        retainedCells=96,
        retainedFraction=0.96,
        evidenceId="qcProfile:cellQc:RNA:globalMad5",
    )


def _example_39_CellQcPlan(cls):
    from scarf.agent.experimental_context.contracts import CellQcProfileEvidence

    evidence = example(CellQcProfileEvidence)
    return cls(
        action=evidence.action,
        registeredProfile=evidence.registeredProfile,
        profileId=evidence.profileId,
        driverAssay=evidence.driverAssay,
        driverAssayType=evidence.driverAssayType,
        sampleColumn=evidence.sampleColumn,
        sampleArtifact=evidence.sampleArtifact,
        attributes=evidence.attributes,
        artifactMetrics=evidence.artifactMetrics,
        rationale="Use the bounded global profile for the RNA assay.",
        evidenceIds=[evidence.evidenceId],
    )


def _example_40_ExperimentalContextDecision(cls):
    from scarf.agent.experimental_context.contracts import (
        BatchCorrectionPlan,
        InferenceUnit,
    )

    return cls(
        columnDomains={
            "batch": "technical",
            "sample": "design",
            "donor": "design",
            "treatment": "biological",
        },
        coefficientsOfInterest=["treatment"],
        unitsOfInference={"treatment": example(InferenceUnit)},
        batchCorrection=example(BatchCorrectionPlan),
        rationale="Treatment is the primary between-sample contrast.",
        evidenceIds=[
            "column:batch",
            "column:donor",
            "column:sample",
            "column:treatment",
        ],
    )


def _example_41_RepresentationEvaluation(cls):
    from scarf.agent.experimental_context.contracts import ArtifactReferenceModel

    return cls(
        available=True,
        assay="RNA",
        cellSelection=ArtifactReferenceModel(
            scope="datastore", kind="cell_selection", artifactId="c" * 64
        ),
        neighbors=ArtifactReferenceModel(
            assay="RNA", kind="neighbors", artifactId="a" * 64
        ),
        connectivityMap=ArtifactReferenceModel(
            assay="RNA", kind="connectivity_map", artifactId="b" * 64
        ),
        metrics={"iLISI:batch": 0.71, "cLISI:cell_type": 0.94},
        evidenceIds=[
            "metric:iLISI:batch:assay:RNA:neighbors:example-neighbors",
            "metric:cLISI:cell_type:assay:RNA:neighbors:example-neighbors",
        ],
    )


def _example_42_CovariateEvidence(cls):
    from scarf.agent.experimental_context.contracts import (
        ArtifactReferenceModel,
        CellQcProfileEvidence,
        CovariateCharacterization,
        NamedArtifactSource,
    )

    return cls(
        characterization=CovariateCharacterization(
            status="done", notes=["Example deterministic covariate characterization"]
        ),
        qcProfiles=[example(CellQcProfileEvidence)],
        htoIdentityColumns=["sample_id"],
        htoIdentityArtifacts=[
            NamedArtifactSource(
                name="HTO_htoIdentity",
                artifact=ArtifactReferenceModel(
                    assay="HTO", kind="hto_identity", artifactId="2" * 64
                ),
            )
        ],
        evidenceIds=[
            "column:batch",
            example(CellQcProfileEvidence).evidenceId,
            "htoIdentity:sample_id",
            f"htoIdentityArtifact:HTO_htoIdentity:{'2' * 64}",
        ],
    )


def _example_43_ExperimentalContextResult(cls):
    from scarf.agent.experimental_context.contracts import (
        AgentRunInfo,
        ArtifactReferenceModel,
        BatchSafetyEvidence,
        CellQcProfileEvidence,
        CovariateCharacterization,
        ExperimentalContextDecision,
        NamedArtifactSource,
        RepresentationEvaluation,
    )

    representation = example(RepresentationEvaluation)
    return cls(
        status="done",
        decision=example(ExperimentalContextDecision),
        characterization=CovariateCharacterization(
            status="done", notes=["Example deterministic design characterization"]
        ),
        cellSelection=representation.cellSelection,
        qcProfiles=[example(CellQcProfileEvidence)],
        qualityMetricArtifacts=[example(NamedArtifactSource)],
        htoIdentityColumns=["sample_id"],
        htoIdentityArtifacts=[
            NamedArtifactSource(
                name="HTO_htoIdentity",
                artifact=ArtifactReferenceModel(
                    assay="HTO", kind="hto_identity", artifactId="2" * 64
                ),
            )
        ],
        batchSafety=[example(BatchSafetyEvidence)],
        currentRepresentation=representation,
        runInfo=example(AgentRunInfo),
    )


def _example_44_ExperimentalContextDependencies(cls):
    return cls(
        studyContext="Case-control study with samples nested in donors.",
        studyObjective="Discover populations while preserving the case-control contrast.",
        directions={"columnDomains": {"batch": "technical"}},
    )


def _example_45_StudyContract(cls):
    return cls(
        studyContext="Treated and control blood samples from multiple donors.",
        studyObjective="Discover stable populations while preserving treatment-associated structure.",
        processingGoal="conditionPreservingDiscovery",
        scientificQuestions=[
            "Discover stable populations while preserving treatment-associated structure."
        ],
        physicalCaptureColumn="sample",
        independentUnitColumns=["donor"],
        conditionColumns=["treatment"],
        technicalBatchColumns=["batch"],
        protectedColumns=["treatment", "donor"],
        correctionLicense="safe",
        allowedClaims=["Describe reproducible population structure."],
        unsupportedClaims=[
            "This workflow does not test differential-expression hypotheses."
        ],
        evidenceIds=["column:batch", "column:donor", "column:treatment"],
        evidenceRequirements=[
            {
                "requirementId": "studyDesign",
                "question": "Which independent units and design constraints support population discovery?",
                "objectiveQuote": "Discover stable populations while preserving treatment-associated structure.",
                "kind": "studyDesign",
                "columns": ["batch", "donor", "treatment"],
                "observationUnit": "sample",
                "independentUnit": "donor",
                "essential": True,
            }
        ],
        evidenceCoverage=[
            {
                "requirementId": "studyDesign",
                "status": "computed",
                "evidenceIds": ["column:batch", "column:donor", "column:treatment"],
                "reasons": [],
            }
        ],
    )


def _example_46_IngestResult(cls):
    return cls(
        status="done", format="h5ad", zarrPath="dataset.zarr", assayNames=["RNA"]
    )


def _example_47_WorkflowQuestion(cls):
    return cls(
        questionId="approvePlanChecksum",
        question="Approve this preprocessing plan?",
        planChecksum="0" * 64,
    )


def _example_48_WorkflowNeedsInput(cls):
    from scarf.agent.orchestrator.models import WorkflowQuestion

    return cls(questions=[example(WorkflowQuestion)])


def _example_49_WorkflowStageLink(cls):
    return cls(stage="ingest", attemptId="attempt-1", contentSha256="0" * 64)


def _example_50_WorkflowStageAttempt(cls):
    return cls(
        workflowRunId="workflow-1",
        stage="ingest",
        attemptId="attempt-1",
        status="done",
        startedAtNs=1,
        completedAtNs=2,
        requestSha256="0" * 64,
        configSha256="1" * 64,
        contentSha256="2" * 64,
    )


def _example_51_AssayPreprocessingPlan(cls):
    return cls(
        assay="RNA",
        assayType="RNA",
        role="graph",
        graphEligible=True,
        markerEligible=True,
        featureMethod="hvg",
        reductionMethod="pca",
        featureParameters={"topN": 1000, "minCells": 20},
    )


def _example_52_AutomatedPreprocessingPlan(cls):
    from scarf.agent.orchestrator.models import (
        ArtifactReferenceModel,
        AssayPreprocessingPlan,
    )

    return cls(
        primaryAssay="RNA",
        markerAssay="RNA",
        cellSelection=ArtifactReferenceModel(
            scope="datastore", kind="cell_selection", artifactId="c" * 64
        ),
        assays=[example(AssayPreprocessingPlan)],
        planChecksum="0" * 64,
    )


def _example_53_PreprocessedAssayHandoff(cls):
    from scarf.agent.orchestrator.models import ArtifactReferenceModel

    return cls(
        assay="RNA",
        assayType="RNA",
        cellSelection=ArtifactReferenceModel(
            scope="datastore", kind="cell_selection", artifactId="c" * 64
        ),
        reductionMethod="pca",
        graphFeatures=example(ArtifactReferenceModel),
        markerFeatures=example(ArtifactReferenceModel),
        normalized=ArtifactReferenceModel(
            assay="RNA", kind="normalized", artifactId="1" * 64
        ),
        nCells=100,
        nFeatures=1000,
    )


def _example_55_FinalAnalysisHandoff(cls):
    from scarf.agent.orchestrator.models import ArtifactReferenceModel

    return cls(
        workflowRunId="workflow-1",
        primaryAssay="RNA",
        markerAssay="RNA",
        cellSelection=ArtifactReferenceModel(
            scope="datastore", kind="cell_selection", artifactId="c" * 64
        ),
        graph=ArtifactReferenceModel(
            assay="RNA", kind="connectivity_map", artifactId="2" * 64
        ),
        embeddingInitialization=ArtifactReferenceModel(
            assay="RNA", kind="embedding_initialization", artifactId="5" * 64
        ),
        clusters=ArtifactReferenceModel(
            assay="RNA", kind="cluster_labels", artifactId="3" * 64
        ),
    )


def _example_56_AutomatedWorkflowConfig(cls):
    return cls()


def _example_57_AutomatedWorkflowRequest(cls):
    return cls(
        sourcePath="dataset.h5ad",
        zarrPath="dataset.zarr",
        studyContext="Single-cell profiling of treated human blood.",
        studyObjective="Discover stable populations while preserving treatment structure.",
    )


def _example_58_AutomatedWorkflowResumeRequest(cls):
    return cls(
        zarrPath="dataset.zarr",
        workflowRunId="workflow-1",
        answers={"approvePlanChecksum": "0" * 64},
    )


def _example_59_OrchestrationResumeRecord(cls):
    return cls(workflowRunId="workflow-1", answers={"approvePlanChecksum": "0" * 64})


def _example_60_ArtifactRecord(cls):
    return cls(scope="assay", kind="connectivity_map", artifactId="a" * 64, assay="RNA")


def _example_61_ParameterCandidate(cls):
    return cls(
        candidateId="baseline",
        reductionMethod="pca",
        dimensions=21,
        leidenResolution=1.0,
        neighborsK=11,
        useHarmony=False,
    )


def _example_62_ParameterMetrics(cls):
    return cls(
        nClusters=8,
        minClusterCells=42,
        minClusterFraction=0.021,
        graphSilhouetteMedian=0.41,
        pcaSilhouette=0.36,
        macroF1=0.82,
        weightedF1=0.86,
        batchMixing={"batch": 0.73},
        biologicalPreservation={
            "cell_type": {"clisi": 0.88, "graphConnectivity": 0.91}
        },
    )


def _example_63_ParameterCandidateEvaluation(cls):
    from scarf.agent.parameter_tuning.contracts import (
        ArtifactRecord,
        ArtifactReferenceModel,
        ParameterCandidate,
        ParameterMetrics,
    )

    candidate = example(ParameterCandidate)
    return cls(
        candidateId=candidate.candidateId,
        status="done",
        eligible=True,
        parameters=candidate,
        artifacts={
            "connectivityMap": example(ArtifactRecord),
            "clusters": ArtifactRecord(
                assay="RNA", kind="cluster_labels", artifactId="b" * 64
            ),
        },
        cellSelection=ArtifactReferenceModel(
            scope="datastore", assay=None, kind="cell_selection", artifactId="c" * 64
        ),
        clusterColumn="RNA_agent_tuning_baseline",
        clusterLabel="agent_tuning_baseline",
        effectiveDimensions=21,
        metrics=example(ParameterMetrics),
        evidenceIds=["candidate:baseline:clusters"],
    )


def _example_64_IntegrationMetrics(cls):
    return cls(
        nClusters=8,
        minClusterCells=37,
        minClusterFraction=0.0185,
        adjustedRandByAssay={"RNA": 0.71, "ADT": 0.63},
        normalizedMutualInformationByAssay={"RNA": 0.76, "ADT": 0.69},
        modalityWeightsValid=True,
    )


def _example_65_IntegrationCandidateEvaluation(cls):
    from scarf.agent.parameter_tuning.contracts import (
        ArtifactRecord,
        ArtifactReferenceModel,
        IntegrationMetrics,
    )

    return cls(
        integrationId="wnn_resolution_1",
        method="wnn",
        assays=["RNA", "ADT"],
        status="done",
        eligible=True,
        cellSelection=ArtifactReferenceModel(
            scope="datastore", assay=None, kind="cell_selection", artifactId="c" * 64
        ),
        graphArtifact=ArtifactRecord(
            scope="datastore", kind="integrated_graph", artifactId="2" * 64
        ),
        clusterArtifact=ArtifactRecord(
            scope="datastore", kind="cluster_labels", artifactId="3" * 64
        ),
        clusterColumn="agent_wnn_cluster",
        metrics=example(IntegrationMetrics),
        evidenceIds=["integration:wnn_resolution_1:clusters"],
    )


def _example_66_FinalGraphComparison(cls):
    return cls(
        optionId="native:ADT:baseline",
        summary="The RNA-native option better preserves the requested labels.",
        evidenceIds=[
            "native:RNA:candidate:baseline:clusters",
            "native:ADT:candidate:baseline:clusters",
        ],
    )


def _example_67_FinalGraphNeedsInput(cls):
    return cls(
        question="Which biological signal must the final graph preserve?",
        options=["cell_type", "condition"],
    )


def _example_68_FinalGraphSelection(cls):
    from scarf.agent.parameter_tuning.contracts import AgentRunInfo

    return cls(
        status="done",
        selectedOptionId="native:RNA:baseline",
        graphMethod="native",
        nativeAssay="RNA",
        nativeCandidateId="baseline",
        markerAssay="RNA",
        confidence="medium",
        rationale="The selected native graph has the strongest supported balance.",
        evidenceIds=["native:RNA:candidate:baseline:clusters"],
        runInfo=example(AgentRunInfo),
    )


def _example_69_CandidateComparison(cls):
    return cls(
        candidateId="pca_15",
        summary="The selected baseline retains larger minimum clusters.",
        evidenceIds=["candidate:baseline:clusters", "candidate:pca_15:clusters"],
    )


def _example_70_ParameterSearchPlan(cls):
    from scarf.agent.parameter_tuning.contracts import AgentRunInfo, ParameterCandidate

    return cls(
        status="refine",
        candidates=[
            ParameterCandidate(
                candidateId="refined_pca_18",
                dimensions=18,
                leidenResolution=1.0,
                neighborsK=11,
                useHarmony=False,
            )
        ],
        basedOnCandidateIds=["baseline", "pca_15"],
        harmonyBatchColumns=[],
        objectives=["Resolve the dimension tradeoff."],
        rationale="The initial screen brackets a narrower dimension range.",
        evidenceIds=["candidate:baseline:clusters", "candidate:pca_15:clusters"],
        stoppingCriteria=["Run the proposed candidate once."],
        runInfo=example(AgentRunInfo),
    )


def _example_71_ParameterTuningBatchSearchPlan(cls):
    from scarf.agent.parameter_tuning.contracts import ParameterSearchPlan

    return cls(assayPlans={"RNA": example(ParameterSearchPlan)})


def _example_72_ParameterTuningNeedsInput(cls):
    return cls(
        question="Which trusted biological label should be preserved?",
        options=["cell_type", "none"],
        evidenceIds=["candidate:baseline:batchMixing:batch"],
    )


def _example_73_ParameterTuningReport(cls):
    from scarf.agent.parameter_tuning.contracts import (
        AgentRunInfo,
        FinalGraphSelection,
        ParameterCandidateEvaluation,
    )

    evaluation = example(ParameterCandidateEvaluation)
    return cls(
        status="done",
        fromAssay="RNA",
        cellSelection=evaluation.cellSelection,
        evaluations=[evaluation],
        recommendedCandidateId=evaluation.candidateId,
        selectedArtifacts=dict(evaluation.artifacts),
        confidence="medium",
        rationale="The baseline balances separation and cluster size.",
        evidenceIds=["candidate:baseline:clusters"],
        tradeoffs=["Higher resolutions produced smaller clusters."],
        limitations=["No trusted biological preservation label was supplied."],
        stopReason="All authorized candidates were evaluated.",
        recommendedByAssay={"RNA": evaluation.candidateId},
        totalCandidates=1,
        graphAssay="RNA",
        markerAssay="RNA",
        finalSelection=example(FinalGraphSelection),
        runInfo=example(AgentRunInfo),
    )


def _example_74_ParameterTuningDependencies(cls):
    from scarf.agent.parameter_tuning.contracts import ParameterCandidate

    candidate = example(ParameterCandidate)
    return cls(
        fromAssay="RNA",
        normalizedShape=(1000, 2000),
        candidates={candidate.candidateId: candidate},
        batchColumns=("batch",),
        preservationColumns=("cell_type",),
    )


def _example_75_ParameterTuningAssayInput(cls):
    from scarf.agent.parameter_tuning.contracts import (
        ArtifactRecord,
        ExperimentalTuningHandoff,
        _default_parameter_candidates,
    )

    return cls(
        normalized=ArtifactRecord(assay="RNA", kind="normalized", artifactId="4" * 64),
        candidates=_default_parameter_candidates(),
        experimentalHandoff=ExperimentalTuningHandoff(batchAction="skip"),
    )


def _example_76_AgentDataModel(cls):
    """Return a small representative value for tests and fixtures."""
    return cls.get_blank()


def _example_77_ArtifactReferenceModel(cls):
    return cls(assay="RNA", kind="reduction", artifactId="0" * 64)


def _example_78_BatchSafetyEvidence(cls):
    return cls(
        coefficient="treatment",
        coefficientKind="categorical",
        observationUnit="sample",
        batchColumns=["batch"],
        unitConstantBatchColumns=["batch"],
        status="safe",
        estimability={
            "status": "ok",
            "coefficientEstimable": True,
            "rankDeficient": False,
        },
        evidenceId="batchEstimability:treatment:batch",
    )


def _example_79_TuningBiologyHandoff(cls):
    from scarf.agent.types import ArtifactReferenceModel

    return cls(
        cellSelection=ArtifactReferenceModel(
            scope="datastore", assay=None, kind="cell_selection", artifactId="c" * 64
        ),
        fromAssay="RNA",
        graphAssay="RNA",
        markerAssay="RNA",
        recommendedCandidateId="baseline",
        clusterArtifact=ArtifactReferenceModel(
            assay="RNA", kind="cluster_labels", artifactId="1" * 64
        ),
        evidenceIds=["candidate:baseline:clusters"],
    )


def _example_80_ToolCallInfo(cls):
    return cls(toolName="inspect_store", callId="tool-call-1")


def _example_81_AgentUsageInfo(cls):
    return cls(
        inputTokens=100, outputTokens=50, totalTokens=150, requests=2, toolCalls=1
    )


def _example_82_AgentRunInfo(cls):
    from scarf.agent.types import AgentUsageInfo, ToolCallInfo

    return cls(
        agentName="data_enrichment",
        modelName="example-model",
        runId="example-run",
        durationSeconds=0.1,
        usage=example(AgentUsageInfo),
        toolCalls=[example(ToolCallInfo)],
    )


def _example_83_AgentExecutionResult(cls):
    from scarf.agent.types import AgentRunInfo

    return cls(output={}, runInfo=example(AgentRunInfo))


def _example_84_EvidenceItem(cls):
    return cls(
        id="evidence:example", label="example", summary="A bounded observed fact."
    )


def _example_85_Decision(cls):
    return cls(
        selectedId="evidence:example",
        rationale="The evidence directly answers the question.",
        evidenceIds=["evidence:example"],
    )


def _example_86_NeedsInput(cls):
    return cls(
        question="Which condition column should be used?",
        options=["condition", "treatment"],
    )


def _example_87_StageResult(cls):
    from scarf.agent.types import Decision

    return cls(status="done", decision=example(Decision))


_FACTORIES = {
    "scarf.agent.biological_interpretation.contracts.BiologicalContext": _example_0_BiologicalContext,
    "scarf.agent.biological_interpretation.contracts.ConditionClusterSummary": _example_1_ConditionClusterSummary,
    "scarf.agent.biological_interpretation.contracts.ClusterCompositionEvidence": _example_2_ClusterCompositionEvidence,
    "scarf.agent.biological_interpretation.contracts.MarkerFeature": _example_3_MarkerFeature,
    "scarf.agent.biological_interpretation.contracts.ClusterMarkerEvidence": _example_4_ClusterMarkerEvidence,
    "scarf.agent.biological_interpretation.contracts.ClusterMarkerBatchEvidence": _example_5_ClusterMarkerBatchEvidence,
    "scarf.agent.biological_interpretation.contracts.ClusterInterpretation": _example_6_ClusterInterpretation,
    "scarf.agent.biological_interpretation.contracts.TreatmentObservation": _example_7_TreatmentObservation,
    "scarf.agent.biological_interpretation.contracts.FollowUpRecommendation": _example_8_FollowUpRecommendation,
    "scarf.agent.biological_interpretation.contracts.BiologicalInterpretationNeedsInput": _example_9_BiologicalInterpretationNeedsInput,
    "scarf.agent.biological_interpretation.contracts.BiologicalInterpretationReport": _example_10_BiologicalInterpretationReport,
    "scarf.agent.biological_interpretation.contracts.BiologicalInterpretationDependencies": _example_11_BiologicalInterpretationDependencies,
    "scarf.agent.config.AgentRunConfig": _example_12_AgentRunConfig,
    "scarf.agent.data_enrichment.characterization.FeatureCharacterization": _example_13_FeatureCharacterization,
    "scarf.agent.data_enrichment.contracts.DataEnrichmentContext": _example_14_DataEnrichmentContext,
    "scarf.agent.data_enrichment.contracts.StudyContextSummary": _example_15_StudyContextSummary,
    "scarf.agent.data_enrichment.contracts.AdtControlEvidence": _example_16_AdtControlEvidence,
    "scarf.agent.data_enrichment.contracts.HtoTagEvidence": _example_17_HtoTagEvidence,
    "scarf.agent.data_enrichment.contracts.AtacCoordinateEvidence": _example_18_AtacCoordinateEvidence,
    "scarf.agent.data_enrichment.contracts.AssayModalityEvidence": _example_19_AssayModalityEvidence,
    "scarf.agent.data_enrichment.contracts.FeatureFamilyEvidence": _example_20_FeatureFamilyEvidence,
    "scarf.agent.data_enrichment.contracts.DefaultHvgFamilyEvidence": _example_21_DefaultHvgFamilyEvidence,
    "scarf.agent.data_enrichment.contracts.RnaFeatureInventoryEvidence": _example_22_RnaFeatureInventoryEvidence,
    "scarf.agent.data_enrichment.contracts.ExogenousFeatureEvidence": _example_23_ExogenousFeatureEvidence,
    "scarf.agent.data_enrichment.contracts.AssayFeatureInspection": _example_24_AssayFeatureInspection,
    "scarf.agent.data_enrichment.contracts.AssayFeatureInspectionBatch": _example_25_AssayFeatureInspectionBatch,
    "scarf.agent.data_enrichment.contracts.FeatureReference": _example_26_FeatureReference,
    "scarf.agent.data_enrichment.contracts.FeatureMatch": _example_27_FeatureMatch,
    "scarf.agent.data_enrichment.contracts.FeatureLookupResult": _example_28_FeatureLookupResult,
    "scarf.agent.data_enrichment.contracts.FeatureLookupBatch": _example_29_FeatureLookupBatch,
    "scarf.agent.data_enrichment.contracts.FeatureSelectionPolicy": _example_30_FeatureSelectionPolicy,
    "scarf.agent.data_enrichment.contracts.DataEnrichmentToolCall": _example_31_DataEnrichmentToolCall,
    "scarf.agent.data_enrichment.contracts.DataEnrichmentReport": _example_32_DataEnrichmentReport,
    "scarf.agent.data_enrichment.contracts.DataEnrichmentDependencies": _example_33_DataEnrichmentDependencies,
    "scarf.agent.experimental_context.contracts.CovariateCharacterization": _example_34_CovariateCharacterization,
    "scarf.agent.experimental_context.contracts.InferenceUnit": _example_35_InferenceUnit,
    "scarf.agent.experimental_context.contracts.BatchCorrectionPlan": _example_36_BatchCorrectionPlan,
    "scarf.agent.experimental_context.contracts.NamedArtifactSource": _example_37_NamedArtifactSource,
    "scarf.agent.experimental_context.contracts.CellQcProfileEvidence": _example_38_CellQcProfileEvidence,
    "scarf.agent.experimental_context.contracts.CellQcPlan": _example_39_CellQcPlan,
    "scarf.agent.experimental_context.contracts.ExperimentalContextDecision": _example_40_ExperimentalContextDecision,
    "scarf.agent.experimental_context.contracts.RepresentationEvaluation": _example_41_RepresentationEvaluation,
    "scarf.agent.experimental_context.contracts.CovariateEvidence": _example_42_CovariateEvidence,
    "scarf.agent.experimental_context.contracts.ExperimentalContextResult": _example_43_ExperimentalContextResult,
    "scarf.agent.experimental_context.contracts.ExperimentalContextDependencies": _example_44_ExperimentalContextDependencies,
    "scarf.agent.experimental_context.study.StudyContract": _example_45_StudyContract,
    "scarf.agent.ingest.result.IngestResult": _example_46_IngestResult,
    "scarf.agent.orchestrator.models.WorkflowQuestion": _example_47_WorkflowQuestion,
    "scarf.agent.orchestrator.models.WorkflowNeedsInput": _example_48_WorkflowNeedsInput,
    "scarf.agent.orchestrator.models.WorkflowStageLink": _example_49_WorkflowStageLink,
    "scarf.agent.orchestrator.models.WorkflowStageAttempt": _example_50_WorkflowStageAttempt,
    "scarf.agent.orchestrator.models.AssayPreprocessingPlan": _example_51_AssayPreprocessingPlan,
    "scarf.agent.orchestrator.models.AutomatedPreprocessingPlan": _example_52_AutomatedPreprocessingPlan,
    "scarf.agent.orchestrator.models.PreprocessedAssayHandoff": _example_53_PreprocessedAssayHandoff,
    "scarf.agent.orchestrator.models.FinalAnalysisHandoff": _example_55_FinalAnalysisHandoff,
    "scarf.agent.orchestrator.models.AutomatedWorkflowConfig": _example_56_AutomatedWorkflowConfig,
    "scarf.agent.orchestrator.models.AutomatedWorkflowRequest": _example_57_AutomatedWorkflowRequest,
    "scarf.agent.orchestrator.models.AutomatedWorkflowResumeRequest": _example_58_AutomatedWorkflowResumeRequest,
    "scarf.agent.orchestrator.models.OrchestrationResumeRecord": _example_59_OrchestrationResumeRecord,
    "scarf.agent.parameter_tuning.contracts.ArtifactRecord": _example_60_ArtifactRecord,
    "scarf.agent.parameter_tuning.contracts.ParameterCandidate": _example_61_ParameterCandidate,
    "scarf.agent.parameter_tuning.contracts.ParameterMetrics": _example_62_ParameterMetrics,
    "scarf.agent.parameter_tuning.contracts.ParameterCandidateEvaluation": _example_63_ParameterCandidateEvaluation,
    "scarf.agent.parameter_tuning.contracts.IntegrationMetrics": _example_64_IntegrationMetrics,
    "scarf.agent.parameter_tuning.contracts.IntegrationCandidateEvaluation": _example_65_IntegrationCandidateEvaluation,
    "scarf.agent.parameter_tuning.contracts.FinalGraphComparison": _example_66_FinalGraphComparison,
    "scarf.agent.parameter_tuning.contracts.FinalGraphNeedsInput": _example_67_FinalGraphNeedsInput,
    "scarf.agent.parameter_tuning.contracts.FinalGraphSelection": _example_68_FinalGraphSelection,
    "scarf.agent.parameter_tuning.contracts.CandidateComparison": _example_69_CandidateComparison,
    "scarf.agent.parameter_tuning.contracts.ParameterSearchPlan": _example_70_ParameterSearchPlan,
    "scarf.agent.parameter_tuning.contracts.ParameterTuningBatchSearchPlan": _example_71_ParameterTuningBatchSearchPlan,
    "scarf.agent.parameter_tuning.contracts.ParameterTuningNeedsInput": _example_72_ParameterTuningNeedsInput,
    "scarf.agent.parameter_tuning.contracts.ParameterTuningReport": _example_73_ParameterTuningReport,
    "scarf.agent.parameter_tuning.contracts.ParameterTuningDependencies": _example_74_ParameterTuningDependencies,
    "scarf.agent.parameter_tuning.contracts.ParameterTuningAssayInput": _example_75_ParameterTuningAssayInput,
    "scarf.agent.types.AgentDataModel": _example_76_AgentDataModel,
    "scarf.agent.types.ArtifactReferenceModel": _example_77_ArtifactReferenceModel,
    "scarf.agent.types.BatchSafetyEvidence": _example_78_BatchSafetyEvidence,
    "scarf.agent.types.TuningBiologyHandoff": _example_79_TuningBiologyHandoff,
    "scarf.agent.types.ToolCallInfo": _example_80_ToolCallInfo,
    "scarf.agent.types.AgentUsageInfo": _example_81_AgentUsageInfo,
    "scarf.agent.types.AgentRunInfo": _example_82_AgentRunInfo,
    "scarf.agent.types.AgentExecutionResult": _example_83_AgentExecutionResult,
    "scarf.agent.types.EvidenceItem": _example_84_EvidenceItem,
    "scarf.agent.types.Decision": _example_85_Decision,
    "scarf.agent.types.NeedsInput": _example_86_NeedsInput,
    "scarf.agent.types.StageResult": _example_87_StageResult,
}

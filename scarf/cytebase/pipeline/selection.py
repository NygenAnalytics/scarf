"""Shared, offline primary-RNA selection for discovery and registration.

The Census list is copied from its literal ``RNA_SEQ`` assignment, extracted with
``ast.literal_eval``. Refresh the pinned lookup deliberately; unfamiliar assays
require review instead of relying on labels or importing an ontology at runtime.
"""

from urllib.parse import urlsplit
from uuid import UUID


SELECTION_SOURCES = [
    {
        "name": "CELLxGENE Census RNA_SEQ",
        "url": (
            "https://raw.githubusercontent.com/chanzuckerberg/cellxgene-census/"
            "e8a7e01ec89395edeb4a7235dee7a19200812813/tools/"
            "cellxgene_census_builder/src/cellxgene_census_builder/"
            "build_soma/globals.py"
        ),
        "commit": "e8a7e01ec89395edeb4a7235dee7a19200812813",
        "sha256": "7e7b15586ce10b45de8762e73377cdaf30d50ec49d38dfe7d6908faae9eff6e2",
        "reviewedAt": "2026-09-24",
    },
    {
        "name": "EBI EFO reviewed RNA additions and non-RNA assays",
        "url": "https://www.ebi.ac.uk/ols4/api/ontologies/efo/terms?obo_id={termId}",
        "reviewedAt": "2026-09-24",
        "scope": "Explicit term IDs below; no descendant or label inference",
    },
]

_CENSUS_RNA = frozenset(
    {
        "EFO:0003755",
        "EFO:0008640",
        "EFO:0008641",
        "EFO:0008643",
        "EFO:0008661",
        "EFO:0008669",
        "EFO:0008673",
        "EFO:0008675",
        "EFO:0008679",
        "EFO:0008694",
        "EFO:0008697",
        "EFO:0008703",
        "EFO:0008708",
        "EFO:0008710",
        "EFO:0008718",
        "EFO:0008720",
        "EFO:0008722",
        "EFO:0008735",
        "EFO:0008747",
        "EFO:0008748",
        "EFO:0008752",
        "EFO:0008753",
        "EFO:0008756",
        "EFO:0008763",
        "EFO:0008780",
        "EFO:0008796",
        "EFO:0008797",
        "EFO:0008824",
        "EFO:0008825",
        "EFO:0008826",
        "EFO:0008850",
        "EFO:0008859",
        "EFO:0008863",
        "EFO:0008868",
        "EFO:0008869",
        "EFO:0008877",
        "EFO:0008896",
        "EFO:0008897",
        "EFO:0008898",
        "EFO:0008903",
        "EFO:0008919",
        "EFO:0008929",
        "EFO:0008930",
        "EFO:0008931",
        "EFO:0008937",
        "EFO:0008941",
        "EFO:0008945",
        "EFO:0008953",
        "EFO:0008954",
        "EFO:0008956",
        "EFO:0008962",
        "EFO:0008966",
        "EFO:0008967",
        "EFO:0008972",
        "EFO:0008974",
        "EFO:0008975",
        "EFO:0008978",
        "EFO:0008980",
        "EFO:0009309",
        "EFO:0009899",
        "EFO:0009900",
        "EFO:0009901",
        "EFO:0009919",
        "EFO:0009922",
        "EFO:0009991",
        "EFO:0009999",
        "EFO:0010003",
        "EFO:0010004",
        "EFO:0010005",
        "EFO:0010006",
        "EFO:0010007",
        "EFO:0010010",
        "EFO:0010022",
        "EFO:0010034",
        "EFO:0010041",
        "EFO:0010058",
        "EFO:0010184",
        "EFO:0010550",
        "EFO:0011025",
        "EFO:0022396",
        "EFO:0022488",
        "EFO:0022490",
        "EFO:0022600",
        "EFO:0022601",
        "EFO:0022602",
        "EFO:0022604",
        "EFO:0022605",
        "EFO:0022606",
        "EFO:0022839",
        "EFO:0022845",
        "EFO:0022846",
        "EFO:0022962",
        "EFO:0030001",
        "EFO:0030002",
        "EFO:0030003",
        "EFO:0030004",
        "EFO:0030019",
        "EFO:0030021",
        "EFO:0030026",
        "EFO:0030028",
        "EFO:0030030",
        "EFO:0030031",
        "EFO:0030059",  # 10x multiome includes RNA.
        "EFO:0030060",
        "EFO:0030061",
        "EFO:0030074",
        "EFO:0700003",
        "EFO:0700004",
        "EFO:0700010",
        "EFO:0700011",
        "EFO:0700016",
        "EFO:0900000",
        "EFO:0900001",
        "EFO:0900002",
    }
)

_REVIEWED_RNA = {
    "EFO:0008913": "single-cell RNA sequencing",
    "EFO:0009809": "single nucleus RNA sequencing",
    "EFO:0008992": "MERFISH",
    "EFO:0008991": "seqFISH",
    "EFO:0920126": "Xenium",
    "EFO:0022615": "10x Xenium",
    "EFO:0022994": "CosMx",
    "EFO:0010961": "Visium Spatial Gene Expression",
    "EFO:0022857": "Visium Spatial Gene Expression V1",
    "EFO:0022858": "Visium CytAssist Spatial Gene Expression V2",
    "EFO:0022859": "Visium CytAssist Spatial Gene Expression, 6.5mm",
    "EFO:0022860": "Visium CytAssist Spatial Gene Expression, 11mm",
    "EFO:0920058": "Visium HD",
    "EFO:0009920": "Slide-seq",
    "EFO:0030062": "Slide-seqV2",
    "EFO:0920125": "Stereo-seq",
    "EFO:0920122": "Seq-Scope",
    "EFO:0008990": "FISSEQ",
    "EFO:0008989": "in situ sequencing",
    "EFO:0700006": "ExSeq",
    "EFO:0700007": "targeted ExSeq",
    "EFO:0700008": "untargeted ExSeq",
}

_NON_RNA = {
    "EFO:0010891": "scATAC-seq",
    "EFO:0030007": "10x scATAC-seq",
}

_RNA_IDS = _CENSUS_RNA | _REVIEWED_RNA.keys()


def is_main_dataset(dataset: dict) -> bool:
    """Keep any-primary datasets; reject missing or contradictory primary metadata."""
    dataset_id = dataset.get("dataset_id", "unknown")
    count = dataset.get("primary_cell_count")
    cells = dataset.get("cell_count")
    flags = dataset.get("is_primary_data")
    if count is not None and (type(count) is not int or count < 0):
        raise ValueError(f"Invalid primary_cell_count for dataset {dataset_id}")
    if cells is not None and (type(cells) is not int or cells < 0):
        raise ValueError(f"Invalid cell_count for dataset {dataset_id}")
    if flags is not None and (
        not isinstance(flags, list) or any(type(flag) is not bool for flag in flags)
    ):
        raise ValueError(f"Invalid is_primary_data list for dataset {dataset_id}")
    primary_flags = set(flags or [])
    if count is None and not primary_flags:
        raise ValueError(f"Primary-data metadata is missing for dataset {dataset_id}")
    contradictory = count is not None and (
        (cells is not None and count > cells)
        or (bool(primary_flags) and (count > 0) != (True in primary_flags))
        or (cells is not None and primary_flags == {True} and count != cells)
        or (cells is not None and primary_flags == {True, False} and count == cells)
    )
    if contradictory:
        raise ValueError(
            f"Primary-data metadata contradicts itself for dataset {dataset_id}"
        )
    return count > 0 if count is not None else True in primary_flags


def _asset_issue(dataset: dict) -> str | None:
    try:
        UUID(dataset["dataset_version_id"])
    except (AttributeError, KeyError, TypeError, ValueError):
        return "Missing or invalid dataset_version_id; review the CELLxGENE record"
    assets = dataset.get("assets")
    if not isinstance(assets, list) or any(
        not isinstance(asset, dict) for asset in assets
    ):
        return "Missing or invalid assets list; review the H5AD download asset"
    h5ad = [
        asset for asset in assets if str(asset.get("filetype", "")).upper() == "H5AD"
    ]
    if len(h5ad) != 1:
        return "Expected exactly one H5AD download asset; review the CELLxGENE record"
    asset = h5ad[0]
    url = asset.get("url")
    try:
        parsed = urlsplit(url) if isinstance(url, str) else None
        if (
            parsed is None
            or parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or any(char.isspace() for char in url)
        ):
            return "Missing or invalid H5AD HTTP URL; review the download asset"
        parsed.port  # Validate a present port without changing the source URL.
    except ValueError:
        return "Invalid H5AD HTTP URL; review the download asset"
    size = asset.get("filesize")
    if size is not None and (type(size) is not int or size < 0):
        return "Invalid H5AD filesize; expected a nonnegative integer or no value"
    return None


def classify_dataset(dataset: dict) -> dict:
    """Select primary RNA datasets across organisms, with explicit review reasons."""
    primary = None

    def result(selection: str, reason: str) -> dict:
        return {"selection": selection, "reason": reason, "primary": primary}

    if not isinstance(dataset, dict):
        return result("needsReview", "Dataset metadata must be an object")
    try:
        primary = is_main_dataset(dataset)
    except ValueError as error:
        return result("needsReview", str(error))
    try:
        UUID(dataset["dataset_id"])
    except (AttributeError, KeyError, TypeError, ValueError):
        return result("needsReview", "Missing or invalid stable dataset_id")
    if not primary:
        return result("skipped", "All cells are secondary")
    assays = dataset.get("assay")
    if not isinstance(assays, list) or not assays:
        return result("needsReview", "Missing assay metadata; review RNA content")
    if any(
        not isinstance(assay, dict)
        or not isinstance(assay.get("ontology_term_id"), str)
        or not assay["ontology_term_id"]
        for assay in assays
    ):
        return result("needsReview", "Missing or invalid assay ontology term IDs")
    ids = {assay["ontology_term_id"] for assay in assays}
    unknown = ids - _RNA_IDS - _NON_RNA.keys()
    if unknown:
        return result(
            "needsReview",
            "Unreviewed assay IDs: "
            + ", ".join(sorted(unknown))
            + "; review RNA content",
        )
    rna = ids & _RNA_IDS
    non_rna = ids & _NON_RNA.keys()
    if rna and non_rna:
        return result(
            "needsReview",
            "Conflicting RNA and non-RNA assay IDs: "
            + ", ".join(sorted(ids))
            + "; review which measurement the H5AD contains",
        )
    if non_rna:
        return result("skipped", "Known non-RNA assays: " + ", ".join(sorted(non_rna)))
    issue = _asset_issue(dataset)
    if issue:
        return result("needsReview", issue)
    return result("selected", "Contains primary cells and reviewed RNA assays")

from collections.abc import Mapping, Sequence

import pandas as pd


AUTO_ASSAY_NAMES: dict[str, str] = {
    "Gene Expression": "RNA",
    "mRNA": "RNA",
    "Peaks": "ATAC",
    "Antibody Capture": "ADT",
    "AbSeq": "ADT",
    "CRISPR Guide Capture": "CRISPR",
    "CRISPR": "CRISPR",
    "CRISPR Direct Capture": "CRISPR",
    "Multiplexing Capture": "HTO",
    "Antigen Capture": "ANTIGEN",
    "ANTIGEN": "ANTIGEN",
    "Custom": "CUSTOM",
    "CUSTOM": "CUSTOM",
    "RNA": "RNA",
    "ADT": "ADT",
    "HTO": "HTO",
}


def make_feat_table_from_types(feature_types: Sequence[str]) -> pd.DataFrame:
    if len(feature_types) == 0:
        raise ValueError("Cannot build an assay table without features")

    spans: list[tuple[str, int, int]] = []
    previous = feature_types[0]
    start = 0
    for index, feature_type in enumerate(feature_types[1:], 1):
        if feature_type != previous:
            spans.append((previous, start, index))
            start = index
        previous = feature_type
    spans.append((previous, start, len(feature_types)))

    table = pd.DataFrame(spans, columns=["type", "start", "end"])
    table.index = [f"ASSAY{index + 1}" for index in table.index]
    table["nFeatures"] = table.end - table.start
    return table.T


def auto_name_feat_table(
    assay_feats: pd.DataFrame,
    name_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    mapping = dict(AUTO_ASSAY_NAMES)
    if name_map is not None:
        mapping.update(name_map)

    renamed = assay_feats.copy()
    renamed.columns = [
        mapping.get(str(feature_type), mapping.get(str(key), str(key)))
        for key, feature_type in renamed.T["type"].items()
    ]
    return renamed

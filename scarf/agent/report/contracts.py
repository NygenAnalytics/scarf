"""Small value adapters for the saved analysis summary."""

import re
from collections.abc import Mapping, Sequence
from typing import Any


def mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def mappings(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def texts(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return []
    return list(
        dict.fromkeys(
            item.strip() for item in value if isinstance(item, str) and item.strip()
        )
    )


def label(value: str) -> str:
    names = {
        "donor_id": "Donor",
        "sample_id": "Sample",
        "library_id": "Capture",
        "T2D": "T2D status",
        "hvgCount": "Number of variable genes",
        "hvgRanking": "Variable-gene ranking",
        "observationAndIndependentUnitsMustBeDesignOrTechnical": "The saved analysis did not support the declared study-unit roles",
        "withinIndependentUnitComparisonsAreUnsupported": "Association testing for repeated observations from one study unit is unavailable",
        "continuousConditioningIsUnsupported": "Conditioning on a continuous variable is unavailable",
        "fewerThanFourIndependentUnits": "Fewer than four independent study units",
        "unsupportedConditioningStrata": "Some conditioned groups do not support the requested comparison",
    }
    if value in names:
        return names[value]
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value.replace("_", " ")).capitalize()


def scalar(value: Any) -> str:
    if value is None:
        return "Unavailable"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)

import hashlib
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import DTypeLike

__all__ = ["PreparedNetwork", "prepare_network", "read_gmt"]


def _owned_readonly(values: np.ndarray, dtype: DTypeLike) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).copy()
    array.setflags(write=False)
    return array


def _update_text(digest: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little", signed=False))
    digest.update(encoded)


def _network_digest(
    source_names: np.ndarray,
    edge_source_index: np.ndarray,
    edge_feature_index: np.ndarray,
    edge_weight: np.ndarray,
    *,
    weighted: bool,
) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(b"weighted\0" if weighted else b"unweighted\0")
    digest.update(len(source_names).to_bytes(8, "little", signed=False))
    for source in source_names:
        _update_text(digest, str(source))
    for values, dtype in (
        (edge_source_index, "<i8"),
        (edge_feature_index, "<i8"),
    ):
        array = np.ascontiguousarray(values, dtype=dtype)
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    if weighted:
        weights = np.ascontiguousarray(edge_weight, dtype="<f8")
        digest.update(np.asarray(weights.shape, dtype="<i8").tobytes())
        digest.update(weights.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PreparedNetwork:
    source_names: np.ndarray
    source_sizes: np.ndarray
    matched_feature_index: np.ndarray
    edge_source_index: np.ndarray
    edge_feature_index: np.ndarray
    edge_weight: np.ndarray
    network_digest: str

    def __post_init__(self) -> None:
        one_dimensional = (
            self.source_names,
            self.source_sizes,
            self.matched_feature_index,
            self.edge_source_index,
            self.edge_feature_index,
            self.edge_weight,
        )
        if any(values.ndim != 1 for values in one_dimensional):
            raise ValueError("Prepared network arrays must be one-dimensional")
        if len(self.source_names) == 0:
            raise ValueError("Prepared network must contain at least one source")
        if len(self.source_names) != len(self.source_sizes):
            raise ValueError("Source names and sizes must be aligned")
        n_edges = len(self.edge_source_index)
        if len(self.edge_feature_index) != n_edges or len(self.edge_weight) != n_edges:
            raise ValueError("Prepared network edge arrays must be aligned")
        if n_edges == 0 or len(self.matched_feature_index) == 0:
            raise ValueError("Prepared network must contain at least one matched edge")
        if np.any(self.source_sizes <= 0):
            raise ValueError("Prepared network source sizes must be positive")
        if np.unique(self.source_names).size != len(self.source_names):
            raise ValueError("Prepared network source names must be unique")
        if not all(isinstance(source, str) and source for source in self.source_names):
            raise ValueError("Prepared network source names must be non-empty strings")
        if not np.array_equal(self.source_names, np.sort(self.source_names)):
            raise ValueError("Prepared network source names must be sorted")
        integer_arrays = (
            self.source_sizes,
            self.matched_feature_index,
            self.edge_source_index,
            self.edge_feature_index,
        )
        if any(
            not np.issubdtype(values.dtype, np.integer) for values in integer_arrays
        ):
            raise ValueError("Prepared network index arrays must have integer dtypes")
        if np.any(self.matched_feature_index < 0) or not np.array_equal(
            self.matched_feature_index, np.unique(self.matched_feature_index)
        ):
            raise ValueError(
                "Prepared network matched features must be sorted and unique"
            )
        if np.any(self.edge_source_index < 0) or np.any(
            self.edge_source_index >= len(self.source_names)
        ):
            raise ValueError("Prepared network source indices are out of bounds")
        if (
            np.any(self.edge_feature_index < 0)
            or not np.isin(self.edge_feature_index, self.matched_feature_index).all()
        ):
            raise ValueError("Prepared network feature indices are out of bounds")
        expected_sizes = np.bincount(
            self.edge_source_index,
            minlength=len(self.source_names),
        )
        if not np.array_equal(self.source_sizes, expected_sizes):
            raise ValueError("Prepared network source sizes do not match its edges")
        edge_order = np.lexsort((self.edge_feature_index, self.edge_source_index))
        if not np.array_equal(edge_order, np.arange(n_edges)):
            raise ValueError("Prepared network edges must be canonically sorted")
        if not np.isfinite(self.edge_weight).all():
            raise ValueError("Prepared network weights must be finite")
        if not isinstance(self.network_digest, str) or not self.network_digest:
            raise ValueError("Prepared network digest must be a non-empty string")


def read_gmt(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Read gene sets from a tab-separated GMT file.

    The first field is used as the source name, the second description field is
    ignored, and each remaining non-empty field becomes one target row.

    Args:
        path: Path to a GMT file with at least three fields per non-empty line.

    Returns:
        A DataFrame with ``source`` and ``target`` columns.
    """
    records: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                continue
            fields = line.split("\t")
            if len(fields) < 3:
                raise ValueError(
                    f"Malformed GMT line {line_number}: expected at least 3 tab-separated fields"
                )
            source = fields[0].strip()
            targets = [target.strip() for target in fields[2:] if target.strip()]
            if not source:
                raise ValueError(f"Malformed GMT line {line_number}: source is empty")
            if not targets:
                raise ValueError(f"Malformed GMT line {line_number}: no targets found")
            records.extend((source, target) for target in targets)
    if not records:
        raise ValueError("GMT file contains no gene sets")
    return pd.DataFrame.from_records(records, columns=["source", "target"])


def prepare_network(
    net: pd.DataFrame,
    *,
    active_feature_names: np.ndarray,
    active_feature_index: np.ndarray,
    tmin: int,
    weighted: bool,
) -> PreparedNetwork:
    """Validate, match, prune, and canonically order a gene-set network."""
    if not isinstance(net, pd.DataFrame):
        raise TypeError("net must be a pandas DataFrame")
    if isinstance(tmin, bool) or not isinstance(tmin, int) or tmin < 1:
        raise ValueError("tmin must be an integer greater than or equal to 1")
    if not isinstance(weighted, bool):
        raise TypeError("weighted must be a boolean")
    if not {"source", "target"}.issubset(net.columns):
        raise ValueError("net must contain 'source' and 'target' columns")
    if any(
        list(net.columns).count(column) > 1 for column in ("source", "target", "weight")
    ):
        raise ValueError("Network columns must have unique names")

    feature_names = np.asarray(active_feature_names)
    raw_feature_index = np.asarray(active_feature_index)
    if feature_names.ndim != 1 or raw_feature_index.ndim != 1:
        raise ValueError("Active feature names and indices must be one-dimensional")
    if len(feature_names) == 0:
        raise ValueError("Feature selection contains no active features")
    if len(feature_names) != len(raw_feature_index):
        raise ValueError("Active feature names and indices must be aligned")
    if not np.issubdtype(raw_feature_index.dtype, np.integer):
        raise ValueError("Active feature indices must have an integer dtype")
    feature_index = np.asarray(raw_feature_index, dtype=np.int64)
    if np.any(feature_index < 0) or np.unique(feature_index).size != len(feature_index):
        raise ValueError("Active feature indices must be non-negative and unique")

    columns = ["source", "target"]
    if "weight" in net.columns:
        columns.append("weight")
    frame = net.loc[:, columns].copy(deep=True).reset_index(drop=True)
    if frame[["source", "target"]].isna().any().any():
        raise ValueError("Network source and target values must not be missing")
    frame["source"] = frame["source"].map(lambda value: str(value).strip())
    frame["target"] = frame["target"].map(lambda value: str(value).strip())
    if (frame["source"] == "").any() or (frame["target"] == "").any():
        raise ValueError("Network source and target values must be non-empty")
    if frame.duplicated(subset=["source", "target"]).any():
        raise ValueError("Network contains duplicate source-target edges")

    if "weight" not in frame.columns:
        frame["weight"] = 1.0
    if frame["weight"].map(lambda value: isinstance(value, (bool, np.bool_))).any():
        raise ValueError("Network weights must be numeric and not boolean")
    try:
        weights = pd.to_numeric(frame["weight"], errors="raise").to_numpy(
            dtype=np.float64
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Network weights must be numeric") from exc
    if not np.isfinite(weights).all():
        raise ValueError("Network weights must be finite")
    frame["weight"] = weights
    if weighted:
        frame = frame.loc[frame["weight"] != 0.0].copy()
        if frame.empty:
            raise ValueError("Weighted network contains no non-zero edges")
    else:
        frame["weight"] = 1.0

    name_to_indices: dict[str, list[int]] = {}
    for name, index in zip(feature_names, feature_index, strict=True):
        name_to_indices.setdefault(str(name).upper(), []).append(int(index))

    matched_rows: list[int] = []
    matched_indices: list[int] = []
    for row_index, target in zip(frame.index, frame["target"], strict=True):
        matches = name_to_indices.get(str(target).upper(), [])
        if not matches:
            continue
        if len(matches) > 1:
            raise ValueError(
                f"Network target {target!r} matches multiple active assay features"
            )
        matched_rows.append(int(row_index))
        matched_indices.append(matches[0])

    if not matched_rows:
        raise ValueError("Network has no targets overlapping the active assay features")
    frame = frame.loc[matched_rows].copy()
    frame["feature_index"] = np.asarray(matched_indices, dtype=np.int64)
    if frame.duplicated(subset=["source", "feature_index"]).any():
        raise ValueError(
            "Network contains duplicate edges after case-insensitive feature matching"
        )

    counts = frame.groupby("source", sort=False, observed=True).size()
    kept_sources = sorted(str(source) for source in counts[counts >= tmin].index)
    if not kept_sources:
        raise ValueError(
            "No network sources remain after feature overlap and "
            f"tmin={tmin} pruning ({len(frame)} matched edges)"
        )
    frame = frame.loc[frame["source"].isin(kept_sources)].copy()

    source_names = np.asarray(kept_sources, dtype="U")
    source_to_index = {
        source: index for index, source in enumerate(source_names.tolist())
    }
    frame["source_index"] = frame["source"].map(source_to_index).astype(np.int64)
    frame = frame.sort_values(
        ["source_index", "feature_index"], kind="mergesort"
    ).reset_index(drop=True)

    edge_source_index = frame["source_index"].to_numpy(dtype=np.int64)
    edge_feature_index = frame["feature_index"].to_numpy(dtype=np.int64)
    edge_weight = frame["weight"].to_numpy(dtype=np.float64)
    matched_feature_index = np.unique(edge_feature_index).astype(np.int64)
    source_sizes = np.bincount(edge_source_index, minlength=len(source_names)).astype(
        np.int64
    )
    digest = _network_digest(
        source_names,
        edge_source_index,
        edge_feature_index,
        edge_weight,
        weighted=weighted,
    )

    return PreparedNetwork(
        source_names=_owned_readonly(source_names, source_names.dtype),
        source_sizes=_owned_readonly(source_sizes, np.int64),
        matched_feature_index=_owned_readonly(matched_feature_index, np.int64),
        edge_source_index=_owned_readonly(edge_source_index, np.int64),
        edge_feature_index=_owned_readonly(edge_feature_index, np.int64),
        edge_weight=_owned_readonly(edge_weight, np.float64),
        network_digest=digest,
    )

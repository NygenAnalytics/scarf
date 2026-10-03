"""Benchmarks of the bookkeeping inside a bounded RNA agent analysis.

A completed four-probe analysis with two finalists verifies the fingerprint of
its frozen source about 15 times (before every pipeline call, every candidate
measurement and every final-result read), appends roughly 75 hash-chained
journal events (150 or more with large populations, repairs and resumes), and
compares each of its three native probes with the baseline at every registered
Leiden resolution. Each benchmark times the agent function those steps call and
checks its value against an independent reference implementation.
"""

import hashlib
import json
import shutil
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from .harness import Ladder

pytestmark = pytest.mark.benchmark

SEED = 4466
# A verified analysis calls verify_source once per pipeline call, once per
# candidate measurement and twice when publishing and plotting the result.
VERIFICATIONS_PER_ANALYSIS = 15.0
# Opening the store and reading every metadata column one at a time costs
# about a second per call, comparable to the per-cell hashing up to 32k cells,
# so the fit separates the two and keeps the best of five repeats.
FINGERPRINT = Ladder(
    sizes=(4_000, 8_000, 16_000, 32_000),
    smoke=300,
    work=VERIFICATIONS_PER_ANALYSIS,
    model="linear",
)
# A completed analysis writes about 75 events; large populations, semantic
# repairs and resumes take a run directory to 150 and beyond.
JOURNAL = Ladder(sizes=(16, 32, 64, 128), smoke=12, unit="events", targets=(150, 500))
# compare_candidates runs once per native probe and covers every resolution.
# Its per-cluster pandas calls cost about 0.1 s per call whatever the cohort
# size, so a linear fit separates them from the per-cell scans.
COMPARISON = Ladder(
    sizes=(16_000, 32_000, 64_000, 128_000), smoke=600, work=3.0, model="linear"
)

N_FEATURES = 2_000
HELD_OUT = "author_cell_type"
RESOLUTIONS = (0.5, 0.75, 1.0, 1.25)
CLUSTERS = (12, 18, 24, 30)


def _barcodes(n_cells: int) -> np.ndarray:
    return np.array([f"{index:016d}-1" for index in range(n_cells)])


def _cell_metadata(n_cells: int) -> dict[str, np.ndarray]:
    """Return CELLxGENE-style cell metadata; one author label is held out."""
    rng = np.random.default_rng(SEED)
    donor = rng.integers(0, 8, n_cells)
    sample = donor * 4 + rng.integers(0, 4, n_cells)
    samples = np.array([f"sample_{index:03d}" for index in range(32)])
    return {
        "donor_id": np.array([f"donor_{index:02d}" for index in range(8)])[donor],
        "sample_id": samples[sample],
        "library_id": np.char.add("library_", samples[sample]),
        "assay": np.full(n_cells, "10x 3' v3"),
        "tissue": np.array(["blood", "bone marrow", "spleen"])[
            rng.integers(0, 3, n_cells)
        ],
        "disease": np.array(["normal", "COVID-19"])[rng.integers(0, 2, n_cells)],
        "sex": np.array(["female", "male"])[rng.integers(0, 2, n_cells)],
        "development_stage": np.array(["adult", "aged"])[rng.integers(0, 2, n_cells)],
        "suspension_type": np.full(n_cells, "cell"),
        "is_primary_data": rng.random(n_cells) < 0.9,
        "n_genes": rng.integers(200, 6_000, n_cells),
        "total_counts": rng.lognormal(8.0, 0.5, n_cells),
        "pct_counts_mt": rng.random(n_cells) * 20.0,
        HELD_OUT: np.array(["T cell", "B cell", "NK cell"])[
            rng.integers(0, 3, n_cells)
        ],
    }


def _write_source(path: Path, n_cells: int) -> None:
    """Write sparse counts and metadata as a prepared local agent source."""
    from scipy import sparse

    from scarf import DataStore
    from scarf.writers import SparseToZarr

    rng = np.random.default_rng(SEED)
    counts = sparse.random(
        n_cells,
        N_FEATURES,
        density=0.02,
        format="csr",
        random_state=rng,
        data_rvs=lambda size: rng.integers(1, 20, size),
    ).astype(np.uint32)
    names = ["MT-CO1", "MT-ND1", *[f"GENE{index}" for index in range(N_FEATURES - 2)]]
    SparseToZarr(
        counts,
        str(path),
        list(_barcodes(n_cells)),
        names,
        mem_budget="512M",
        nthreads=1,
    ).dump()
    store = DataStore(
        str(path),
        default_assay="RNA",
        min_features_per_cell=-1,
        nthreads=1,
        mem_budget="512M",
    )
    for column, values in _cell_metadata(n_cells).items():
        store.cells.insert(column, values)


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()
        + b"\n"
    )


def _reference_fingerprint(
    store: Any, columns: list[str], *, replace: tuple[str, int, Any] | None = None
) -> str:
    """Hash the documented canonical stream from whole-column reads.

    The header binds the assay's finalized count identity and axis sizes; every
    column then contributes its axis, name and dtype, followed by each value in
    row order. ``replace`` substitutes one cell value, as a source edit would.
    """
    summary = store.summary()
    descriptor = next(item for item in summary.assays if item.name == "RNA")
    digest = hashlib.sha256()
    digest.update(
        _canonical(
            [
                "RNA",
                descriptor.dataset_fingerprint,
                summary.total_cells,
                descriptor.total_features,
            ]
        )
    )
    for axis, table, names in (
        ("cells", store.cells, columns),
        ("features", store.get_assay("RNA").feats, ["ids", "names", "I"]),
    ):
        for column in names:
            digest.update(_canonical([axis, column, str(table.get_dtype(column))]))
            values = table.fetch_all(column).tolist()
            if replace is not None and (axis, column) == ("cells", replace[0]):
                values[replace[1]] = replace[2]
            for value in values:
                digest.update(_canonical(value))
    return digest.hexdigest()


@pytest.fixture(scope="module")
def agent_sources(tmp_path_factory) -> Callable[[int], tuple[Path, dict[str, Any]]]:
    """Return a prepared source of a size and its frozen verification evidence."""
    from scarf import DataStore

    root = tmp_path_factory.mktemp("agent_sources")
    built: dict[int, tuple[Path, dict[str, Any]]] = {}

    def get(n_cells: int) -> tuple[Path, dict[str, Any]]:
        if n_cells not in built:
            path = root / f"cells_{n_cells}.zarr"
            _write_source(path, n_cells)
            store = DataStore(str(path), zarr_mode="r", nthreads=1, mem_budget="512M")
            # Every visible column is frozen; held-out annotations never are.
            columns = sorted(name for name in store.cells.columns if name != HELD_OUT)
            prepared = {
                "assay": "RNA",
                "fingerprintColumns": columns,
                "fingerprint": _reference_fingerprint(store, columns),
                "contextEvidence": {"references": []},
            }
            built[n_cells] = path, prepared
        return built[n_cells]

    return get


def test_agent_source_verification(bench, agent_sources) -> None:
    from scarf.agent.evidence import verify_source
    from scarf.agent.models import (
        AnalysisConfig,
        AnalysisInputError,
        RuntimeConfig,
        Study,
    )

    study = Study(context="Benchmark", objective="Describe populations")
    config = AnalysisConfig()
    runtime = RuntimeConfig(nthreads=1, memBudget="512M")

    def make(n_cells: int):
        path, prepared = agent_sources(n_cells)
        return lambda: verify_source(path, prepared, study, config, runtime)

    def check(n_cells: int, value: object) -> None:
        from scarf import DataStore

        # verify_source returns only when the store hashes to the digest that
        # the reference implementation computed from whole-column reads.
        assert value is None
        path, prepared = agent_sources(n_cells)
        assert len(prepared["fingerprintColumns"]) == 19
        store = DataStore(str(path), zarr_mode="r", nthreads=1, mem_budget="512M")
        edited = _reference_fingerprint(
            store, prepared["fingerprintColumns"], replace=("donor_id", 0, "donor_99")
        )
        assert edited != prepared["fingerprint"]
        with pytest.raises(AnalysisInputError, match="Source fingerprint changed"):
            verify_source(
                path, {**prepared, "fingerprint": edited}, study, config, runtime
            )

    bench("agent.source_verification", make, FINGERPRINT, check=check, min_repeats=5)


_DIGEST = "0123456789abcdef" * 4
_CANDIDATE = {
    "candidateId": "c3",
    "parentId": "c0",
    "hvgCount": 1000,
    "pcaDims": 21,
    "neighborsK": 21,
    "useHarmony": False,
}


def _journal_events(n_events: int) -> list[tuple[str, dict[str, Any]]]:
    """Return ``n_events`` events cycling through a completed analysis's mix."""
    events = []
    for index in range(n_events):
        call = f"{index:06d}-0123456789ab"
        provenance = {
            "decisionId": f"annotations_{index}",
            "stage": "annotate",
            "identityDigest": _DIGEST,
            "evidenceDigest": _DIGEST,
            "promptDigest": _DIGEST,
            "schemaDigest": _DIGEST,
            "invocationId": "f" * 32,
            "callId": call,
        }
        label = f"agent_0123456789abcdef_screen_c{index % 4}_0"
        templates = [
            ("status", {"status": "running", "stage": "explore"}),
            (
                "modelRequest",
                {
                    **provenance,
                    "requestPath": f"calls/{call}.request.json",
                    "responsePath": f"calls/{call}.response.json",
                },
            ),
            (
                "modelResponse",
                {
                    **provenance,
                    "responsePath": f"calls/{call}.response.json",
                    "elapsedSeconds": 1.25,
                    "usage": {
                        "inputTokens": 21_000,
                        "outputTokens": 900,
                        "cacheReadTokens": 0,
                        "cacheWriteTokens": 0,
                    },
                },
            ),
            (
                "decisionAccepted",
                {
                    **provenance,
                    "recovered": False,
                    "output": {
                        "annotations": [
                            {
                                "clusterId": str(cluster),
                                "identity": "unassigned",
                                "confidence": "low",
                                "supportingMarkers": [],
                                "contradictingMarkers": [],
                                "rationale": "Markers do not establish a lineage.",
                            }
                            for cluster in range(8)
                        ]
                    },
                },
            ),
            (
                "pipelinePlanned",
                {
                    "operation": f"screen_c{index % 4}",
                    "label": label,
                    "candidate": _CANDIDATE,
                    "resolution": None,
                    "markers": False,
                    "final": False,
                    "process": {
                        "pid": 12345,
                        "platform": "linux",
                        "startTicks": "123456789",
                        "bootId": "0123abcd-0123-abcd-0123-0123456789ab",
                    },
                },
            ),
            (
                "pipelineCompleted",
                {
                    "operation": f"screen_c{index % 4}",
                    "runId": _DIGEST[:20],
                    "label": label,
                    "recovered": False,
                },
            ),
            ("candidateAdmitted", {"candidateId": "c3", "candidate": _CANDIDATE}),
            (
                "explorationCoverage",
                {
                    "coverage": {
                        "slots": [
                            {
                                "candidateId": f"c{slot}",
                                "axis": axis,
                                "parentId": None if slot == 0 else "c0",
                                "status": "measured",
                                "reason": None,
                                "parameters": {
                                    **_CANDIDATE,
                                    "candidateId": f"c{slot}",
                                },
                            }
                            for slot, axis in enumerate(
                                ("baseline", "hvgCount", "pcaDims", "neighborsK")
                            )
                        ],
                        "nativeComplete": True,
                    }
                },
            ),
            (
                "stageCompleted",
                {"stage": "explore", "evidence": "evidence/explore.json"},
            ),
        ]
        events.append(templates[index % len(templates)])
    return events


def test_agent_journal_appends(bench, tmp_path) -> None:
    from scarf.agent.records import RunRecords

    created: list[Path] = []

    def make(n_events: int):
        for previous in created:
            shutil.rmtree(previous, ignore_errors=True)
        path = tmp_path / f"journal_{len(created)}"
        created.append(path)
        records = RunRecords.create(path, {"runId": "benchmark"})
        events = _journal_events(n_events)

        def call() -> Path:
            for kind, payload in events:
                records.append(kind, **payload)
            return records.path

        return call

    def check(n_events: int, path: Path) -> None:
        # Verify the hash chain from the raw files with the documented
        # canonical form, independently of RunRecords.events().
        files = sorted((path / "events").iterdir())
        assert [item.name for item in files] == [
            f"{sequence:06d}.json" for sequence in range(1, n_events + 1)
        ]
        previous = None
        for sequence, (item, (kind, payload)) in enumerate(
            zip(files, _journal_events(n_events), strict=True), start=1
        ):
            event = json.loads(item.read_text(encoding="utf-8"))
            record_hash = event.pop("recordHash")
            assert (event["sequence"], event["kind"]) == (sequence, kind)
            assert event["previousHash"] == previous
            assert {key: event[key] for key in payload} == payload
            canonical = json.dumps(
                event, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2
            )
            assert (
                record_hash == hashlib.sha256((canonical + "\n").encode()).hexdigest()
            )
            previous = record_hash

    bench("agent.journal_append", make, JOURNAL, check=check, fresh=True)


class _Run(dict[str, Any]):
    """The completed-run interface compare_candidates reads, served from memory."""

    status = "completed"

    def __init__(self, run_id: str, selection: Any, columns: dict[str, Any]) -> None:
        super().__init__(
            {"analysis_cell_selection": selection}
            | {key: f"{run_id}/{key}" for key in columns if key.startswith("leiden_")}
        )
        self.run_id = run_id
        self.cells = SimpleNamespace(fetch=columns.__getitem__)


@lru_cache(maxsize=2)
def _comparison_runs(n_cells: int) -> tuple[_Run, _Run]:
    """Return a baseline and a probe whose partitions mostly agree.

    Cluster sizes are skewed, the probe numbers its clusters differently, and
    5% of cells move to another cluster, as a neighbor probe would move them.
    """
    rng = np.random.default_rng(SEED)
    ids = _barcodes(n_cells)
    selection = SimpleNamespace(to_dict=lambda: {"artifactId": "frozen-cells"})
    parent: dict[str, np.ndarray] = {"ids": ids}
    probe: dict[str, np.ndarray] = {"ids": ids.copy()}
    for resolution, clusters in zip(RESOLUTIONS, CLUSTERS, strict=True):
        weights = 1.0 / np.arange(1, clusters + 1)
        labels = rng.choice(
            np.arange(1, clusters + 1), n_cells, p=weights / weights.sum()
        )
        moved = rng.random(n_cells) < 0.05
        changed = labels.copy()
        changed[moved] = rng.integers(1, clusters + 1, int(moved.sum()))
        renumbered = rng.permutation(clusters) + 1
        parent[f"leiden_{resolution}"] = labels
        probe[f"leiden_{resolution}"] = renumbered[changed - 1]
    return _Run("baseline", selection, parent), _Run("probe", selection, probe)


def _reference_overlaps(
    source: np.ndarray, target: np.ndarray
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Best matches from an integer contingency table over string labels."""
    sources, source_codes = np.unique(source, return_inverse=True)
    targets, target_codes = np.unique(target, return_inverse=True)
    table = np.zeros((sources.size, targets.size), dtype=np.int64)
    np.add.at(table, (source_codes, target_codes), 1)
    rows = []
    for index, label in enumerate(sources):
        best = int(np.argmax(table[index]))
        size = int(table[index].sum())
        rows.append(
            {
                "clusterId": str(label),
                "matchedClusterId": str(targets[best]),
                "sourceCells": size,
                "intersectionCells": int(table[index, best]),
                "fraction": int(table[index, best]) / size,
            }
        )
    return rows, table


def _reference_ari(table: np.ndarray) -> float:
    def pairs(values: np.ndarray) -> float:
        return float((values * (values - 1) // 2).sum())

    joint = pairs(table)
    rows, columns = pairs(table.sum(axis=1)), pairs(table.sum(axis=0))
    expected = rows * columns / pairs(np.array([table.sum()]))
    return (joint - expected) / ((rows + columns) / 2 - expected)


def test_agent_probe_comparison(bench) -> None:
    from scarf.agent.execution import compare_candidates
    from scarf.agent.models import AnalysisConfig, Candidate

    baseline = Candidate(candidateId="c0", hvgCount=1000, pcaDims=21, neighborsK=11)
    probe = baseline.model_copy(
        update={"candidateId": "c3", "parentId": "c0", "neighborsK": 21}
    )
    config = AnalysisConfig(resolutions=RESOLUTIONS)

    def make(n_cells: int):
        parent, run = _comparison_runs(n_cells)
        return lambda: compare_candidates(None, parent, run, baseline, probe, config)

    def check(n_cells: int, comparisons: Any) -> None:
        parent, run = _comparison_runs(n_cells)
        assert [row["resolution"] for row in comparisons] == list(RESOLUTIONS)
        for row in comparisons:
            key = f"leiden_{row['resolution']}"
            left = parent.cells.fetch(key).astype(str)
            right = run.cells.fetch(key).astype(str)
            forward, table = _reference_overlaps(left, right)
            backward, _ = _reference_overlaps(right, left)
            assert row["cellCount"] == n_cells
            assert (row["parentRunId"], row["runId"]) == ("baseline", "probe")
            assert row["parentToCandidate"] == forward
            assert row["candidateToParent"] == backward
            assert row["adjustedRandIndex"] == pytest.approx(
                _reference_ari(table), rel=1e-12
            )
            # Renumbered clusters with 5% of cells moved still mostly agree.
            assert 0.8 < row["adjustedRandIndex"] < 1.0

    bench("agent.probe_comparison", make, COMPARISON, check=check)

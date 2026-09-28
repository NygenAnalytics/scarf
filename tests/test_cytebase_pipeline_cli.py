"""Offline tests for the Cytebase pipeline command line."""

import inspect
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scarf.cytebase.pipeline import build
from scarf.cytebase.pipeline.__main__ import main
from tests.fixtures_cytebase import (
    BUCKET_ID,
    COLLECTION_ID,
    COUNTS,
    DATASET_ID,
    full_manifest,
    write_h5ad,
)

pytestmark = pytest.mark.usefixtures("cytebase_offline")

RUN_ID = "fc-orchestrator-1"
RESET_CALL_ID = "fc-reset-1"


class FakeModal:
    """Stands in for the ``modal`` module and records lookups and spawns."""

    def __init__(self) -> None:
        self.lookups: list[tuple[tuple, dict]] = []
        self.spawns: list[tuple[tuple, dict]] = []
        self.Function = SimpleNamespace(from_name=self._from_name)

    def _from_name(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
        self.lookups.append((args, kwargs))
        return SimpleNamespace(spawn=self._spawn)

    def _spawn(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
        self.spawns.append((args, kwargs))
        return SimpleNamespace(object_id=RESET_CALL_ID)


@pytest.fixture
def fake_modal(monkeypatch) -> FakeModal:
    """Resolve ``import modal`` to a recorder so no deployment is contacted."""
    fake = FakeModal()
    monkeypatch.setitem(sys.modules, "modal", fake)
    return fake


@pytest.fixture
def catalog_module():
    pytest.importorskip("duckdb")
    pytest.importorskip("natsort")
    from scarf.cytebase.pipeline import catalog

    return catalog


@pytest.fixture
def inventory_module(catalog_module):
    from scarf.cytebase.pipeline import inventory

    return inventory


def _set_argv(monkeypatch, *args: Any) -> None:
    monkeypatch.setattr(sys, "argv", ["cytebase-pipeline", *map(str, args)])


def _run(monkeypatch, capsys, *args: Any) -> Any:
    """Run the CLI and return its printed JSON result."""
    _set_argv(monkeypatch, *args)
    main()
    return json.loads(capsys.readouterr().out)


def _write_json(path: Path, value: Any) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _receipt(output: Path) -> Any:
    text = (output / "scarf_ingest.json").read_text(encoding="utf-8")
    assert text.endswith("}\n")
    return json.loads(text)


@pytest.mark.parametrize(
    ("flag", "counts", "selection_source", "feature_table"),
    [
        pytest.param([], "raw/X", "inspection", "raw/var", id="discovered"),
        pytest.param(
            ["--raw-data-location", "raw.X"],
            "raw/X",
            "curation_api",
            "raw/var",
            id="declared-raw",
        ),
        pytest.param(
            ["--raw-data-location", "X"],
            "none",
            "curation_api",
            None,
            id="declared-normalized",
        ),
    ],
)
def test_inspect_prints_the_count_selection_manifest(
    tmp_path, monkeypatch, capsys, flag, counts, selection_source, feature_table
):
    source = write_h5ad(
        tmp_path / "source.h5ad", (COUNTS / 2).astype(np.float32), raw_counts=COUNTS
    )
    location = flag[1] if flag else None
    printed = _run(monkeypatch, capsys, "inspect", source, *flag)
    expected = build.inspect_file(source, location)["manifest"]
    assert printed == json.loads(json.dumps(expected))
    assert printed["countsLocation"] == counts
    assert printed["apiRawDataLocation"] == location
    assert printed["countsSelectionSource"] == selection_source
    assert printed["featureAttrsKey"] == feature_table
    # A declared location that fails validation asks for input, never falls back.
    assert (printed["selectionNeedsInput"] is None) == (counts != "none")


def test_convert_writes_a_verified_receipt(tmp_path, monkeypatch, capsys):
    source = write_h5ad(tmp_path / "source.h5ad")
    manifest = _write_json(tmp_path / "manifest.json", full_manifest(source))
    output = tmp_path / "out"
    printed = _run(
        monkeypatch,
        capsys,
        "convert",
        source,
        "--manifest",
        manifest,
        "--output",
        output,
    )
    receipt = _receipt(output)
    assert printed == receipt
    assert receipt["status"] == "done"
    assert receipt["zarrPath"] == str(output / "data.zarr")
    assert (receipt["nObs"], receipt["nVars"]) == COUNTS.shape
    verification = receipt["verification"]
    assert verification["countsTMatches"] is True
    assert verification["countsTShape"] == [COUNTS.shape[1], COUNTS.shape[0]]
    assert verification["countsBlock"] == COUNTS[:3].tolist()
    assert (output / "data.zarr").is_dir()


def test_convert_records_needs_input_without_verifying(tmp_path, monkeypatch, capsys):
    def unexpected_verify(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Only completed conversions are verified")

    monkeypatch.setattr(build, "verify_store", unexpected_verify)
    source = write_h5ad(tmp_path / "source.h5ad")  # Has no raw/X matrix.
    manifest = _write_json(
        tmp_path / "manifest.json", full_manifest(source, raw_data_location="raw.X")
    )
    output = tmp_path / "runs" / "first"
    printed = _run(
        monkeypatch,
        capsys,
        "convert",
        source,
        "--manifest",
        manifest,
        "--output",
        output,
    )
    receipt = _receipt(output)
    assert printed == receipt
    assert receipt["status"] == "needsInput"
    assert receipt["needsInput"]["options"] == ["raw/X"]
    assert "CELLxGENE specifies raw.X" in receipt["needsInput"]["question"]
    assert "verification" not in receipt
    assert not (output / "data.zarr").exists()


def test_convert_rejects_an_inspection_without_provenance(
    tmp_path, monkeypatch, capsys
):
    source = write_h5ad(tmp_path / "source.h5ad")
    manifest = _write_json(
        tmp_path / "manifest.json", build.inspect_file(source)["manifest"]
    )
    output = tmp_path / "out"
    _set_argv(
        monkeypatch, "convert", source, "--manifest", manifest, "--output", output
    )
    with pytest.raises(ValueError, match="sourceSha256"):
        main()
    assert not output.exists()
    assert capsys.readouterr().out == ""


def test_convert_refuses_to_write_non_standard_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        build,
        "convert_local",
        lambda source, store, manifest: {"status": "needsInput", "countsMax": np.nan},
    )
    manifest = _write_json(tmp_path / "manifest.json", {})
    output = tmp_path / "out"
    _set_argv(
        monkeypatch,
        "convert",
        "source.h5ad",
        "--manifest",
        manifest,
        "--output",
        output,
    )
    with pytest.raises(ValueError, match="not JSON compliant"):
        main()
    assert not (output / "scarf_ingest.json").exists()
    assert capsys.readouterr().out == ""


def test_collection_ids_prints_public_collections(monkeypatch, capsys, catalog_module):
    ids = [COLLECTION_ID, "55555555-5555-4555-8555-555555555555"]
    monkeypatch.setattr(catalog_module, "list_collection_ids", lambda: ids)
    assert _run(monkeypatch, capsys, "collection-ids") == {"collectionIds": ids}


def test_command_diagnostics_use_stderr(monkeypatch, capsys, catalog_module):
    from scarf.utils import logger

    def list_collection_ids():
        logger.info("Fetching collection metadata")
        print("Resolving collection previews")
        return [COLLECTION_ID]

    monkeypatch.setattr(catalog_module, "list_collection_ids", list_collection_ids)
    # Exercise stdout logging even when the test session silences INFO logs.
    handler = logger.add(lambda message: sys.stdout.write(str(message)), level="INFO")
    try:
        _set_argv(monkeypatch, "collection-ids")
        main()
    finally:
        logger.remove(handler)
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"collectionIds": [COLLECTION_ID]}
    assert "Fetching collection metadata" in captured.err
    assert "Resolving collection previews" in captured.err


@pytest.mark.parametrize(
    ("bucket_args", "bucket"),
    [
        pytest.param([], None, id="public-only"),
        pytest.param(["--bucket", BUCKET_ID], BUCKET_ID, id="with-catalog"),
    ],
)
def test_inventory_prints_its_summary_with_a_large_dataset_count(
    tmp_path, monkeypatch, capsys, inventory_module, bucket_args, bucket
):
    calls = []
    large = [
        {
            "collectionId": COLLECTION_ID,
            "datasetId": DATASET_ID,
            "cellCount": 1_500_000,
        },
        {
            "collectionId": COLLECTION_ID,
            "datasetId": "55555555-5555-4555-8555-555555555555",
            "cellCount": 2_000_000,
        },
    ]

    def build_inventory(output: Path, *, bucket: str | None = None) -> dict:
        calls.append((output, bucket))
        return {
            "collections": {},
            "summary": {
                "collectionCount": 1,
                "selectedDatasetCount": 2,
                "selectedDatasetsOverMillionCells": large,
            },
        }

    monkeypatch.setattr(inventory_module, "build_inventory", build_inventory)
    output = tmp_path / "inventory.json"
    printed = _run(monkeypatch, capsys, "inventory", "--output", output, *bucket_args)
    assert calls == [(output, bucket)]
    assert printed == {
        "output": str(output),
        "collectionCount": 1,
        "selectedDatasetCount": 2,
        "selectedDatasetsOverMillionCells": 2,
    }


@pytest.mark.parametrize(
    ("env_args", "environment"),
    [
        pytest.param([], None, id="default-env"),
        pytest.param(["--env", "staging"], "staging", id="named-env"),
    ],
)
def test_reset_run_spawns_a_reset_on_the_deployed_pipeline(
    monkeypatch, capsys, fake_modal, env_args, environment
):
    printed = _run(
        monkeypatch,
        capsys,
        "reset-run",
        "--expected-run-id",
        RUN_ID,
        "--workers-drained",
        *env_args,
    )
    assert printed == {"callId": RESET_CALL_ID}
    assert fake_modal.lookups == [
        (("cytebase", "run_pipeline"), {"environment_name": environment})
    ]
    assert fake_modal.spawns == [
        (("reset", {"expectedRunId": RUN_ID, "workersDrained": True}), {})
    ]


def test_reset_run_lookup_matches_the_modal_api(monkeypatch, capsys):
    modal = pytest.importorskip("modal")
    lookup = inspect.signature(modal.Function.from_name)
    fake = FakeModal()
    monkeypatch.setitem(sys.modules, "modal", fake)
    _run(
        monkeypatch,
        capsys,
        "reset-run",
        "--expected-run-id",
        RUN_ID,
        "--workers-drained",
        "--env",
        "staging",
    )
    [(args, kwargs)] = fake.lookups
    assert lookup.bind(*args, **kwargs).arguments == {
        "app_name": "cytebase",
        "name": "run_pipeline",
        "environment_name": "staging",
    }


@pytest.mark.parametrize(
    ("args", "message"),
    [
        pytest.param(
            [], "the following arguments are required: command", id="no-command"
        ),
        pytest.param(["publish"], "invalid choice: 'publish'", id="unknown-command"),
        pytest.param(
            ["inspect"], "the following arguments are required: source", id="no-source"
        ),
        pytest.param(
            ["inspect", "a.h5ad", "--raw-data-location", "layers/counts"],
            "invalid choice: 'layers/counts'",
            id="unsupported-raw-location",
        ),
        pytest.param(
            ["convert", "a.h5ad", "--output", "out"],
            "the following arguments are required: --manifest",
            id="convert-without-manifest",
        ),
        pytest.param(
            ["convert", "a.h5ad", "--manifest", "manifest.json"],
            "the following arguments are required: --output",
            id="convert-without-output",
        ),
        pytest.param(
            ["collection-ids", "--bucket", BUCKET_ID],
            "unrecognized arguments: --bucket",
            id="collection-ids-takes-no-options",
        ),
        pytest.param(
            ["inventory"],
            "the following arguments are required: --output",
            id="inventory-without-output",
        ),
        pytest.param(
            ["reset-run", "--workers-drained"],
            "the following arguments are required: --expected-run-id",
            id="reset-without-run-id",
        ),
        pytest.param(
            ["reset-run", "--expected-run-id", RUN_ID],
            "the following arguments are required: --workers-drained",
            id="reset-without-drained-workers",
        ),
    ],
)
def test_cli_rejects_invalid_arguments(monkeypatch, capsys, fake_modal, args, message):
    _set_argv(monkeypatch, *args)
    with pytest.raises(SystemExit) as exited:
        main()
    assert exited.value.code == 2
    assert message in capsys.readouterr().err
    assert fake_modal.lookups == []


def test_module_entry_point_runs_the_cli(monkeypatch, capsys, fake_modal):
    monkeypatch.delitem(sys.modules, "scarf.cytebase.pipeline.__main__", raising=False)
    _set_argv(
        monkeypatch, "reset-run", "--expected-run-id", RUN_ID, "--workers-drained"
    )
    runpy.run_module("scarf.cytebase.pipeline", run_name="__main__")
    assert json.loads(capsys.readouterr().out) == {"callId": RESET_CALL_ID}
    assert len(fake_modal.spawns) == 1

#!/usr/bin/env python3
"""Regenerate current-format Scarf tutorial stores with artifact provenance.

This builds a small analyzed Zarr store using the atomic / pipeline API so
documentation and local demos can show list_artifacts / inspect_artifact
without relying on master-era archives.

Publishing the resulting store to Cytebase / Hugging Face is intentionally
manual: upload the unpacked directory (for Repository.open_zarr) and optionally
a .zarr.tar.gz (for download_dataset(zarr=True)).

Example:
    uv run python scripts/regenerate_docs_datasets.py \\
        --dataset tenx_5K_pbmc_rnaseq \\
        --destination /tmp/scarf_docs_datasets
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _artifact_inventory(datastore) -> list[dict[str, object]]:
    inventory: list[dict[str, object]] = []
    for ref in datastore.list_artifacts(complete_only=True):
        status = datastore.inspect_artifact(ref)
        inventory.append(
            {
                "kind": ref.kind,
                "artifactId": ref.artifact_id,
                "assay": ref.assay,
                "scope": ref.scope,
                "operation": status.operation,
                "path": status.path,
            }
        )
    return inventory


def build_analyzed_store(
    *,
    dataset: str,
    destination: Path,
    repository: str,
) -> Path:
    import scarf
    from scarf import DataStore

    work = destination / "_download"
    work.mkdir(parents=True, exist_ok=True)
    scarf.cytebase.connect(repository).download_dataset(
        dataset,
        destination=str(work),
        zarr=True,
    )
    preferred = (
        work / dataset / "data.zarr",
        work / f"{dataset}.zarr",
        work / dataset,
    )
    source = next((path for path in preferred if path.exists()), None)
    if source is None:
        matches = [
            path
            for path in work.rglob("*")
            if path.is_dir()
            and ((path / "zarr.json").exists() or (path / ".zgroup").exists())
        ]
        if not matches:
            raise FileNotFoundError(f"No extracted Zarr found under {work}")
        source = matches[0]

    output = destination / f"{dataset}_analyzed.zarr"
    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(source, output)

    ds = DataStore(str(output), default_assay="RNA")
    refs = ds.pipeline.run(
        filtering={"method": "auto"},
        cell_cycle_scoring=False,
        highly_variable_features={"top_n": 100, "show_plot": False},
        paris=False,
        doublet_scoring=False,
        markers=False,
        leiden={1.0: {}},
    )
    state = ds.get_assay_state("RNA")
    if state is None or state.connectivity_map is None:
        raise RuntimeError("Pipeline did not publish a connectivity map")

    inventory = _artifact_inventory(ds)
    manifest = {
        "dataset": dataset,
        "repository": repository,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "generatorCommit": _git_commit(Path(__file__).resolve().parents[1]),
        "scarfVersion": getattr(scarf, "__version__", "unknown"),
        "outputPath": str(output),
        "nCellsActive": int(ds.cells.active_index("I").size),
        "pipelineRefs": {name: ref.to_dict() for name, ref in refs.items()},
        "artifacts": inventory,
        "publishNotes": [
            "Upload the unpacked *.zarr directory for Repository.open_zarr demos.",
            "Optionally pack a .zarr.tar.gz for download_dataset(zarr=True).",
            "Do not overwrite the frozen 0.32.3 compatibility corpus with this store.",
        ],
    }
    manifest_path = destination / f"{dataset}_analyzed_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    archive_path = destination / f"{dataset}_analyzed.zarr.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(output, arcname=output.name)

    print(f"Wrote analyzed store: {output}")
    print(f"Wrote archive:        {archive_path}")
    print(f"Wrote manifest:       {manifest_path}")
    print(f"Complete artifacts:   {len(inventory)}")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="tenx_5K_pbmc_rnaseq",
        help="Cytebase dataset name to regenerate",
    )
    parser.add_argument(
        "--repository",
        default="scarf_docs",
        help="Cytebase repository name",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("docs/source/scarf_datasets/regenerated"),
        help="Output directory for analyzed store, archive, and manifest",
    )
    args = parser.parse_args(argv)
    args.destination.mkdir(parents=True, exist_ok=True)
    build_analyzed_store(
        dataset=args.dataset,
        destination=args.destination,
        repository=args.repository,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

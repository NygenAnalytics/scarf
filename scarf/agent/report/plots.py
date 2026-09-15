"""One saved map and a compact marker table for the analysis report."""

import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .._plots import cluster_counts, plot_final_umap
from .artifacts import artifact_ref

if TYPE_CHECKING:
    from ...datastore.datastore import DataStore
    from ...plotting import PlotResult


def _save_plot(plot: "PlotResult", path: Path) -> None:
    """Atomically replace the derived image and its provenance sidecar."""
    token = uuid.uuid4().hex
    temporary = path.with_name(f".{path.stem}.{token}{path.suffix}")
    sidecar = path.with_suffix(path.suffix + ".json")
    temporary_sidecar = sidecar.with_name(f".{sidecar.stem}.{token}{sidecar.suffix}")
    try:
        plot.save(temporary, dpi=150)
        plot.save_provenance(temporary_sidecar, figure_path=path, dpi=150)
        os.replace(temporary, path)
        os.replace(temporary_sidecar, sidecar)
    finally:
        temporary.unlink(missing_ok=True)
        temporary_sidecar.unlink(missing_ok=True)
        plot.close()


def collect_analysis_artifacts(
    store: "DataStore", final: Mapping[str, Any], report_dir: Path
) -> dict[str, Any]:
    """Read final numerical artifacts without launching any analysis operation."""
    clusters = artifact_ref(final.get("clusters"))
    notes: list[str] = []
    plot_path: str | None = None
    displayed: int | None = None
    counts: dict[str, int]
    try:
        plot = plot_final_umap(
            store,
            umap=artifact_ref(final.get("umap")),
            clusters=clusters,
            cell_selection=artifact_ref(final.get("cellSelection")),
            graph=artifact_ref(final.get("graph")),
            show=False,
        )
        counts = dict(
            zip(
                plot.tables["cluster_counts"]["cluster"],
                plot.tables["cluster_counts"]["cells"],
                strict=True,
            )
        )
        displayed = plot.provenance.n_cells
        plot_dir = report_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        _save_plot(plot, plot_dir / "final_umap.png")
        plot_path = "plots/final_umap.png"
    except (ImportError, OSError, RuntimeError) as exc:
        counts = cluster_counts(store, clusters)
        notes.append(f"UMAP display unavailable: {type(exc).__name__}: {exc}")
    marker_rows: list[dict[str, Any]] = []
    if final.get("markers") is not None:
        marker = artifact_ref(final["markers"])
        marker_inputs = store.inspect_artifact(marker).inputs or {}
        if marker_inputs.get("clusters") != clusters.to_dict():
            raise ValueError("Final marker statistics must use the selected clusters")
        if (
            marker_inputs.get("cell_selection")
            != artifact_ref(final.get("cellSelection")).to_dict()
        ):
            raise ValueError("Final markers must use the selected cells")
        if marker.assay != clusters.assay:
            raise ValueError("Final markers must use the selected RNA assay")
        for cluster in counts:
            try:
                table = store.get_markers(marker, group_id=cluster)
                if "score" in table:
                    table = table.sort_values("score", ascending=False, kind="stable")
                for row in table.head(3).to_dict(orient="records"):
                    marker_rows.append(
                        {
                            "cluster": cluster,
                            "feature": row.get(
                                "feature_name", row.get("feature_id", "")
                            ),
                            "score": row.get("score"),
                        }
                    )
            except Exception as exc:
                notes.append(
                    f"Markers unavailable for cluster {cluster}: {type(exc).__name__}: {exc}"
                )
    return {
        "clusterCounts": counts,
        "markers": marker_rows,
        "umap": plot_path,
        "displayedCells": displayed,
        "displayNotes": notes,
    }

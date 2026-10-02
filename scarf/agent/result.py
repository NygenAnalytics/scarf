"""Public results that keep frozen numerical identity separate from local records."""

import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from .models import AnalysisConfig, RuntimeConfig, Study
from .records import RecordError, RunRecords
from .rendering import (
    annotation_csv,
    annotations,
    atomic_text,
    decision_resolutions,
    exploration_coverage,
    render_report,
    stage_data,
    summary,
)

if TYPE_CHECKING:
    import pandas as pd

    from scarf.datastore.pipeline_run import PipelineRun
    from scarf.storage.artifacts import ArtifactRef


class AnalysisRun:
    """A saved analysis; opening and reporting never require the source or a model."""

    def __init__(self, run_dir: str | Path, source: str | Path | None = None) -> None:
        self._records = RunRecords(run_dir)
        self.run_dir = self._records.path
        self._source = (
            Path(source).expanduser().resolve() if source is not None else None
        )

    @property
    def source(self) -> Path:
        if self._source is not None:
            return self._source
        rebound = self._records.latest("sourceRebound")
        if rebound:
            return (self.run_dir / str(rebound["source"])).resolve()
        return (self.run_dir / str(self._records.manifest["source"])).resolve()

    @property
    def status(self) -> str:
        current = self._records.latest("status")
        return str(current["status"]) if current else "running"

    @property
    def pending_questions(self) -> list[dict[str, Any]]:
        current = self._records.latest("status")
        return (
            list(current.get("questions", []))
            if current and current["status"] == "needsInput"
            else []
        )

    @property
    def pipeline_runs(self) -> list[dict[str, Any]]:
        """Return saved completed-run references without opening the source."""
        return [
            {
                key: event[key]
                for key in ("operation", "runId", "label", "recovered")
                if key in event
            }
            for event in self._records.events()
            if event["kind"] == "pipelineCompleted"
        ]

    @property
    def candidates(self) -> list[dict[str, Any]]:
        """Saved candidate measurements and their exact numerical run references."""
        return [
            self._records.read_json(event["evidence"])
            for event in self._records.events()
            if event["kind"] == "candidateMeasured"
        ]

    @property
    def exploration_coverage(self) -> dict[str, Any] | None:
        """Recorded trial coverage, or None when the saved procedure omitted it."""
        return exploration_coverage(self._records)

    @property
    def decision_resolutions(self) -> list[dict[str, Any]]:
        """Saved conservative policy resolutions, distinct from model choices."""
        return decision_resolutions(self._records)

    def replay_decisions(self) -> list[dict[str, Any]]:
        """Revalidate accepted decisions without a provider or numerical store."""
        from .workflow import replay

        return replay(self._records)

    def _bound_pipeline(self) -> tuple[Any, "PipelineRun"]:
        from .evidence import open_store, verify_source

        final = stage_data(self._records, "finalize")
        if final is None:
            raise ValueError("This analysis has no finalized numerical pipeline")
        manifest = self._records.manifest
        supplied = self._records.latest("inputsResolved") or manifest
        study = Study.model_validate(supplied["study"])
        config = AnalysisConfig.model_validate(supplied["config"])
        invocation = self._records.latest("invocationStarted")
        runtime = RuntimeConfig.model_validate(
            invocation["runtime"] if invocation else {}
        )
        prepared = stage_data(self._records, "preprocess") or stage_data(
            self._records, "inspect"
        )
        if prepared is None:
            raise RecordError("Finalized analysis is missing its source fingerprint")
        verify_source(self.source, prepared, study, config, runtime)
        store = open_store(self.source, config, runtime, writable=False)
        run = store.pipeline.open(run_id=final["runId"])
        if run.status != "completed":
            raise ValueError("The saved final pipeline is no longer complete")
        actual = {key: ref.to_dict() for key, ref in run.items()}
        if actual != final["artifacts"]:
            raise ValueError(
                "Final pipeline artifacts do not match the accepted analysis"
            )
        if any(not store.inspect_artifact(ref).complete for ref in run.values()):
            raise ValueError("The saved final pipeline has incomplete artifacts")
        return store, run

    @property
    def pipeline(self) -> "PipelineRun":
        """Open the exact final core run after verifying the source fingerprint."""
        return self._bound_pipeline()[1]

    @property
    def artifacts(self) -> dict[str, "ArtifactRef"]:
        return dict(self.pipeline)

    @property
    def compact_result(self) -> dict[str, Any] | None:
        """Read the verified local-store summary; absent summaries stay absent."""
        from .compact_result import read_result

        return read_result(self)

    @property
    def annotations(self) -> list[dict[str, Any]]:
        return annotations(self._records)

    def get_markers(self, **kwargs: Any) -> "pd.DataFrame":
        """Read the final immutable marker artifact using core filtering options."""
        if "marker" in kwargs:
            raise ValueError("Marker identity is pinned to the final pipeline")
        store, run = self._bound_pipeline()
        return cast("pd.DataFrame", store.get_markers(marker=run["markers"], **kwargs))

    def plot_embedding(self, **kwargs: Any) -> Any:
        """Plot the frozen UMAP with readable defaults; explicit options override them."""
        if {"run", "layout", "layout_key"}.intersection(kwargs):
            raise ValueError("Plot identity is pinned to the final pipeline UMAP")
        store, run = self._bound_pipeline()
        options = {**_embedding_options(), **kwargs}
        return store.plots.embedding(run=run, layout="umap", **options)

    def plot_markers(
        self, *, top_n: int = 2, max_genes: int = 40, show: bool = False
    ) -> Any:
        """Plot saved marker means and expressing fractions for the final clusters."""
        from .plots import marker_dotplot

        store, run = self._bound_pipeline()
        return marker_dotplot(store, run, top_n=top_n, max_genes=max_genes, show=show)

    def save_plots(self) -> dict[str, Path]:
        """Save 300-DPI UMAP and marker previews from the verified final artifacts.

        This requires the source store, but never reads the count matrix or runs
        a numerical pipeline. Call report() afterwards to embed the saved images.
        """
        from .plots import marker_dotplot

        store, run = self._bound_pipeline()
        outputs = {}
        for name in ("umap_clusters", "marker_dotplot"):
            path = self.run_dir / f"{name}.png"
            if path.is_symlink():
                raise RecordError(f"Refusing to replace a symlink plot: {path.name}")
            plot = (
                store.plots.embedding(run=run, layout="umap", **_embedding_options())
                if name == "umap_clusters"
                else marker_dotplot(store, run)
            )
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=self.run_dir, prefix=f".{name}-", suffix=".png", delete=False
                ) as stream:
                    temporary = Path(stream.name)
                plot.save(temporary, dpi=300, exact_size=False)
                os.replace(temporary, path)
                outputs[name] = path
            finally:
                plot.close()
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        return outputs

    def report(self) -> Path:
        """Regenerate Markdown, HTML, and annotation CSV using only saved evidence."""
        return render_report(self._records)

    def export(self, output_dir: str | Path) -> Path:
        """Export frozen aligned results into a new directory without recomputation."""
        store, run = self._bound_pipeline()
        directory = Path(output_dir).expanduser().resolve()
        if directory == self.source or directory.is_relative_to(self.source):
            raise ValueError("Export directory must be outside the numerical store")
        frame = run.cells.to_pandas_dataframe(["ids", "clusters", "umap_1", "umap_2"])
        rows = self.annotations
        if rows and {str(value) for value in frame["clusters"]} != {
            row["clusterId"] for row in rows
        }:
            raise ValueError("Annotations do not cover the exported frozen clusters")
        markers = store.get_markers(marker=run["markers"], min_score=0, min_frac_exp=0)
        result = summary(self._records)
        result["exportedCells"] = len(frame)
        result["exportedMarkers"] = len(markers)
        directory.mkdir(parents=True, exist_ok=False)
        atomic_text(
            directory / "clusters.csv", frame[["ids", "clusters"]].to_csv(index=False)
        )
        atomic_text(
            directory / "umap.csv",
            frame[["ids", "umap_1", "umap_2"]].to_csv(index=False),
        )
        atomic_text(directory / "markers.csv", markers.to_csv(index=False))
        atomic_text(directory / "annotations.csv", annotation_csv(rows))
        atomic_text(
            directory / "summary.json",
            json.dumps(
                result, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2
            )
            + "\n",
        )
        return directory

    def __repr__(self) -> str:
        return f"AnalysisRun(run_dir={str(self.run_dir)!r}, status={self.status!r})"


def _embedding_options() -> dict[str, Any]:
    return {
        "color_by": "clusters",
        "figsize": (8, 8),
        "theme": "paper",
        "point_edgewidth": 0,
        "point_alpha": 0.85,
        "legend_loc": "right",
        "show_titles": False,
        "show": False,
    }

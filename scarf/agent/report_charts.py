"""Small report figures made from saved measurements, without store access."""

import io
import math
from typing import Any
from xml.etree import ElementTree


def cluster_size_svg(rows: list[dict[str, Any]]) -> str | None:
    """Draw up to 100 final clusters, with exact counts and explicit missing values."""
    if not rows:
        return None
    from matplotlib import rc_context
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    def order(row: dict[str, Any]) -> tuple[int, int | str]:
        label = str(row["clusterId"])
        return (0, int(label)) if label.isdecimal() else (1, label.casefold())

    selected = sorted(rows, key=order)[:100]
    labels = [str(row["clusterId"]) for row in selected]
    counts = [
        int(count)
        if isinstance(count, (int, float))
        and not isinstance(count, bool)
        and math.isfinite(count)
        and count >= 0
        and float(count).is_integer()
        else None
        for row in selected
        for count in [row.get("count")]
    ]
    annotations = [f"{count:,}" if count is not None else "N/A" for count in counts]
    width = max(6.5, len(labels) * max(0.48, max(map(len, annotations)) * 0.07) + 0.8)
    rotate_labels = any(len(label) > 4 for label in labels)
    with rc_context(
        {
            "svg.fonttype": "none",
            "svg.hashsalt": "scarf-cluster-sizes",
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "text.usetex": False,
            "text.parse_math": False,
        }
    ):
        figure = Figure(figsize=(width, 3.1))
        ax = figure.subplots()
        figure.subplots_adjust(
            left=0.6 / width,
            right=0.99,
            bottom=0.34 if rotate_labels else 0.20,
            top=0.91,
        )
        bars = ax.bar(
            range(len(labels)),
            [count or 0 for count in counts],
            color="#0077fc",
            width=0.68,
        )
        ax.bar_label(bars, labels=annotations, padding=4, fontsize=8)
        ax.set_xticks(
            range(len(labels)),
            labels=[label if len(label) <= 10 else label[:9] + "…" for label in labels],
        )
        ax.tick_params(
            axis="x",
            length=0,
            labelrotation=90 if rotate_labels else 0,
        )
        ax.tick_params(axis="y", length=0, colors="#636973")
        ax.set(
            xlabel="Cluster",
            ylabel="Cells",
            ylim=(0, max(count or 0 for count in counts) * 1.18 or 1),
        )
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e3e7ec", linewidth=0.6)
        for spine in ax.spines.values():
            spine.set_visible(False)
        buffer = io.StringIO()
        figure.savefig(buffer, format="svg", metadata={"Date": None})
    root = ElementTree.fromstring(buffer.getvalue())
    namespace = "http://www.w3.org/2000/svg"
    ElementTree.register_namespace("", namespace)
    root.set("role", "img")
    root.set("aria-labelledby", "cluster-size-title cluster-size-description")
    title = ElementTree.Element(f"{{{namespace}}}title", {"id": "cluster-size-title"})
    title.text = "Cell counts by final cluster"
    description = ElementTree.Element(
        f"{{{namespace}}}desc", {"id": "cluster-size-description"}
    )
    description.text = "; ".join(
        f"Cluster {label}: {count:,} cells"
        if count is not None
        else f"Cluster {label}: Not recorded"
        for label, count in zip(labels, counts, strict=True)
    )
    root.insert(0, title)
    root.insert(1, description)
    return ElementTree.tostring(root, encoding="unicode")

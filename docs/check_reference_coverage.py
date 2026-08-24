"""Check that the built reference documents every public symbol exactly once."""

import inspect
import sys
import zlib
from pathlib import Path

DOCS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = DOCS_ROOT.parent
DEFAULT_INVENTORY = DOCS_ROOT / "build" / "html" / "objects.inv"


def read_inventory(path: Path) -> list[str]:
    raw = path.read_bytes()
    header = raw.split(b"\n", 4)
    if len(header) < 5:
        raise SystemExit(f"Malformed inventory: {path}")
    lines = zlib.decompress(header[4]).decode("utf-8").splitlines()
    return [line.split(" ", 1)[0] for line in lines if line.strip()]


def datastore_method_report(names: list[str]) -> tuple[list[str], list[str]]:
    from scarf import DataStore

    missing: list[str] = []
    duplicated: list[str] = []
    for name, _ in inspect.getmembers(DataStore, callable):
        if name.startswith("_"):
            continue
        hits = [entry for entry in names if entry.endswith(f".DataStore.{name}")]
        if not hits:
            missing.append(name)
        elif len(hits) > 1:
            duplicated.append(f"{name}: {', '.join(sorted(hits))}")
    return missing, duplicated


def top_level_report(names: list[str]) -> list[str]:
    import scarf

    documented = set(names)
    missing: list[str] = []
    for name in getattr(scarf, "__all__", []):
        if f"scarf.{name}" in documented:
            continue
        if any(entry.endswith(f".{name}") for entry in documented):
            continue
        missing.append(name)
    return missing


def main() -> None:
    inventory = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INVENTORY
    if not inventory.is_file():
        raise SystemExit(f"Build the HTML docs first; no inventory at {inventory}")
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    names = read_inventory(inventory)
    missing_methods, duplicated_methods = datastore_method_report(names)
    missing_exports = top_level_report(names)

    problems: list[str] = []
    if missing_methods:
        problems.append(
            "DataStore methods missing from the reference: "
            + ", ".join(sorted(missing_methods))
        )
    if duplicated_methods:
        problems.append(
            "DataStore methods documented more than once:\n  "
            + "\n  ".join(sorted(duplicated_methods))
        )
    if missing_exports:
        problems.append(
            "scarf.__all__ symbols missing from the reference: "
            + ", ".join(sorted(missing_exports))
        )
    if problems:
        raise SystemExit("\n".join(problems))

    print(
        "Reference coverage complete: every public DataStore method is documented "
        "exactly once and every scarf.__all__ symbol is documented"
    )


if __name__ == "__main__":
    main()

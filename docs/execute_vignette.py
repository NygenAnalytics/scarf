"""Execute myst-nb documentation pages into docs/.jupyter_cache."""

import os
import shutil
import sqlite3
from pathlib import Path

from myst_parser.config.main import MdParserConfig
from myst_nb.core.config import NbParserConfig
from myst_nb.core.execute import create_client
from myst_nb.core.read import create_nb_reader

DOCS_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = DOCS_ROOT / "source"
DEFAULT_CACHE = DOCS_ROOT / ".jupyter_cache"
DEFAULT_PAGE = "scrna_seq"

# Non-executable prose trees (still under source/) are skipped.
_SKIP_DIR_NAMES = {
    "_static",
    "_templates",
    "_build",
    "scarf_datasets",
    "dev",
}

os.environ.setdefault("IPYTHONDIR", str(DOCS_ROOT / ".ipython"))


def configure_doc_execution_env() -> None:
    """Cap Scarf and BLAS memory for notebook doc builds."""
    os.environ["SCARF_MEM_BUDGET"] = "4G"
    os.environ["SCARF_WORKING_COPIES"] = "1"
    os.environ["SCARF_WORKERS"] = "2"
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = "2"


configure_doc_execution_env()


class _Logger:
    prefix: str = ""

    def info(self, msg: str, subtype: str | None = None) -> None:
        print(f"{self.prefix}{msg}", flush=True)

    def warning(self, msg: str, subtype: str | None = None) -> None:
        print(f"{self.prefix}WARNING: {msg}", flush=True)

    def debug(self, msg: str, subtype: str | None = None) -> None:
        pass


def _is_executable_myst(path: Path) -> bool:
    """Return True when the markdown file is a myst-nb notebook source."""
    text = path.read_text(encoding="utf-8", errors="replace")
    # Require notebook frontmatter or a real MyST code-cell fence, not prose mentions.
    if "kernelspec:" in text:
        return True
    return "```{code-cell}" in text or "```{code-cell} " in text


def iter_executable_paths() -> list[Path]:
    """Discover executable MyST pages under docs/source recursively."""
    paths: list[Path] = []
    for path in sorted(SOURCE_DIR.rglob("*.md")):
        if any(part in _SKIP_DIR_NAMES for part in path.parts):
            continue
        if not _is_executable_myst(path):
            continue
        paths.append(path)
    return paths


def _page_aliases(path: Path) -> list[str]:
    """Names that resolve to this page: stem and source-relative path without suffix."""
    rel = path.relative_to(SOURCE_DIR)
    aliases = [path.stem, rel.as_posix().removesuffix(".md")]
    # Deduplicate while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for alias in aliases:
        if alias not in seen:
            seen.add(alias)
            out.append(alias)
    return out


def list_executable_docs() -> list[str]:
    """Return unique page stems (and conflict-safe relative paths) for CLI use."""
    stem_counts: dict[str, int] = {}
    paths = iter_executable_paths()
    for path in paths:
        stem_counts[path.stem] = stem_counts.get(path.stem, 0) + 1

    names: list[str] = []
    for path in paths:
        if stem_counts[path.stem] == 1:
            names.append(path.stem)
        else:
            names.append(path.relative_to(SOURCE_DIR).as_posix().removesuffix(".md"))
    return names


def list_vignettes() -> list[str]:
    """Backward-compatible alias for list_executable_docs."""
    return list_executable_docs()


def resolve_doc_path(name: str) -> Path:
    """Resolve a page name or source-relative path to an executable .md file."""
    cleaned = name.removesuffix(".md").lstrip("./")
    candidates = [
        SOURCE_DIR / f"{cleaned}.md",
        SOURCE_DIR / "tutorials" / f"{cleaned}.md",
        SOURCE_DIR / "vignettes" / f"{cleaned}.md",
    ]
    for candidate in candidates:
        if candidate.exists() and _is_executable_myst(candidate):
            return candidate

    matches = [
        path
        for path in iter_executable_paths()
        if cleaned in _page_aliases(path) or path.stem == Path(cleaned).stem
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        opts = ", ".join(
            p.relative_to(SOURCE_DIR).as_posix().removesuffix(".md") for p in matches
        )
        raise FileNotFoundError(
            f"Ambiguous executable doc {name!r}; use one of: {opts}"
        )
    raise FileNotFoundError(f"Executable doc not found: {name}")


def prune_stale_cache(cache_path: Path = DEFAULT_CACHE) -> list[str]:
    """Drop cache rows and executed notebooks whose source .md no longer exists."""
    db_path = cache_path / "global.db"
    if not db_path.exists():
        return []

    removed: list[str] = []
    db = sqlite3.connect(db_path)
    try:
        for (uri,) in db.execute("SELECT uri FROM nbproject").fetchall():
            if Path(uri).exists():
                continue
            row = db.execute(
                "SELECT hashkey FROM nbcache WHERE uri = ?", (uri,)
            ).fetchone()
            if row is not None:
                executed_dir = cache_path / "executed" / row[0]
                if executed_dir.exists():
                    shutil.rmtree(executed_dir)
            db.execute("DELETE FROM nbproject WHERE uri = ?", (uri,))
            db.execute("DELETE FROM nbcache WHERE uri = ?", (uri,))
            removed.append(uri)

        valid_hashkeys = {
            row[0] for row in db.execute("SELECT hashkey FROM nbcache").fetchall()
        }
        executed_root = cache_path / "executed"
        if executed_root.exists():
            for entry in executed_root.iterdir():
                if entry.is_dir() and entry.name not in valid_hashkeys:
                    shutil.rmtree(entry)
                    removed.append(f"orphan:{entry.name}")

        db.commit()
    finally:
        db.close()

    removed.extend(prune_failed_cache_without_notebook(cache_path))
    return removed


def prune_failed_cache_without_notebook(
    cache_path: Path = DEFAULT_CACHE,
) -> list[str]:
    """Drop nbproject error rows that never produced a cached notebook."""
    db_path = cache_path / "global.db"
    if not db_path.exists():
        return []

    removed: list[str] = []
    db = sqlite3.connect(db_path)
    try:
        for (uri,) in db.execute(
            "SELECT uri FROM nbproject WHERE traceback IS NOT NULL"
        ).fetchall():
            row = db.execute(
                "SELECT hashkey FROM nbcache WHERE uri = ?", (uri,)
            ).fetchone()
            if row is not None:
                continue
            db.execute("DELETE FROM nbproject WHERE uri = ?", (uri,))
            removed.append(uri)
        db.commit()
    finally:
        db.close()
    return removed


def _invalidate_cache_entry(cache_path: Path, uri: str) -> None:
    db_path = cache_path / "global.db"
    if not db_path.exists():
        return
    db = sqlite3.connect(db_path)
    try:
        row = db.execute("SELECT hashkey FROM nbcache WHERE uri = ?", (uri,)).fetchone()
        if row is not None:
            executed_dir = cache_path / "executed" / row[0]
            if executed_dir.exists():
                shutil.rmtree(executed_dir)
        db.execute("DELETE FROM nbproject WHERE uri = ?", (uri,))
        db.execute("DELETE FROM nbcache WHERE uri = ?", (uri,))
        db.commit()
    finally:
        db.close()


def _cached_notebook_hash(cache_path: Path, uri: str) -> str | None:
    db_path = cache_path / "global.db"
    if not db_path.exists():
        return None
    db = sqlite3.connect(db_path)
    try:
        row = db.execute("SELECT hashkey FROM nbcache WHERE uri = ?", (uri,)).fetchone()
        return None if row is None else str(row[0])
    finally:
        db.close()


def _stale_failure_traceback(cache_path: Path, uri: str) -> str | None:
    """Return traceback only when a prior failure has no successful notebook cache."""
    if _cached_notebook_hash(cache_path, uri) is not None:
        return None
    db_path = cache_path / "global.db"
    if not db_path.exists():
        return None
    db = sqlite3.connect(db_path)
    try:
        row = db.execute(
            "SELECT traceback FROM nbproject WHERE uri = ?", (uri,)
        ).fetchone()
        if row is None or not row[0]:
            return None
        return str(row[0])
    finally:
        db.close()


def execute_page(
    name: str,
    *,
    cache_path: Path | None = None,
    execution_in_temp: bool = False,
    force: bool = False,
) -> dict:
    path = resolve_doc_path(name)

    cache_path = cache_path or DEFAULT_CACHE
    cache_path.mkdir(parents=True, exist_ok=True)
    uri = str(path.resolve())

    if force:
        _invalidate_cache_entry(cache_path, uri)
    elif (traceback := _stale_failure_traceback(cache_path, uri)) is not None:
        raise RuntimeError(
            f"Page {name!r} has a cached execution error; re-run with --force:\n"
            f"{traceback}"
        )

    nb_config = NbParserConfig(
        execution_mode="cache",
        execution_cache_path=str(cache_path),
        execution_timeout=600,
        execution_allow_errors=False,
        execution_raise_on_error=True,
        execution_show_tb=True,
        execution_in_temp=execution_in_temp,
    )
    md_config = MdParserConfig(enable_extensions={"colon_fence"})
    content = path.read_text(encoding="utf-8")
    nb_reader = create_nb_reader(str(path), md_config, nb_config, content)
    if nb_reader is None:
        raise ValueError(f"Not a myst-nb page: {path}")

    notebook = nb_reader.read(content)
    logger = _Logger()
    logger.prefix = f"[{name}] "
    with create_client(
        notebook, str(path), nb_config, logger, nb_reader.read_fmt
    ) as client:
        metadata = client.exec_metadata or {}

    if _cached_notebook_hash(cache_path, uri) is None:
        raise RuntimeError(f"Page {name!r} finished without writing a cached notebook")

    print(f"[{name}] finished", flush=True)
    return {"name": name, "metadata": metadata}


def execute_vignette(
    name: str,
    *,
    cache_path: Path | None = None,
    execution_in_temp: bool = False,
    force: bool = False,
) -> dict:
    """Backward-compatible alias for execute_page."""
    return execute_page(
        name,
        cache_path=cache_path,
        execution_in_temp=execution_in_temp,
        force=force,
    )


def merge_caches(source_dirs: list[Path], target_dir: Path = DEFAULT_CACHE) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    target_executed = target_dir / "executed"
    target_executed.mkdir(parents=True, exist_ok=True)
    version_file = target_dir / "__version__.txt"
    if not version_file.exists() and source_dirs:
        first_version = next(
            (
                d / "__version__.txt"
                for d in source_dirs
                if (d / "__version__.txt").exists()
            ),
            None,
        )
        if first_version is not None:
            shutil.copy2(first_version, version_file)

    target_db = sqlite3.connect(target_dir / "global.db")
    try:
        for source_dir in source_dirs:
            source_db_path = source_dir / "global.db"
            if not source_db_path.exists():
                continue
            source_db = sqlite3.connect(source_db_path)
            try:
                projects = source_db.execute("SELECT * FROM nbproject").fetchall()
                caches = source_db.execute("SELECT * FROM nbcache").fetchall()
            finally:
                source_db.close()

            for row in projects:
                uri = row[1]
                target_db.execute("DELETE FROM nbproject WHERE uri = ?", (uri,))
                target_db.execute(
                    "INSERT INTO nbproject (uri, read_data, assets, exec_data, created, traceback) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    row[1:],
                )

            for row in caches:
                hashkey = row[1]
                uri = row[2]
                src = source_dir / "executed" / hashkey
                dst = target_executed / hashkey
                if dst.exists():
                    shutil.rmtree(dst)
                if src.exists():
                    shutil.copytree(src, dst)
                target_db.execute("DELETE FROM nbcache WHERE uri = ?", (uri,))
                target_db.execute(
                    "INSERT INTO nbcache (hashkey, uri, description, data, created, accessed) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    row[1:],
                )
        target_db.commit()
    finally:
        target_db.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "page",
        nargs="?",
        default=DEFAULT_PAGE,
        help="Page stem or source-relative path without .md (default: scrna_seq)",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=None,
        help="Override jupyter cache directory",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-execute even when a cached notebook exists",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List executable page names and exit",
    )
    cli = parser.parse_args()
    if cli.list:
        for name in list_executable_docs():
            print(name)
        raise SystemExit(0)
    execute_page(
        cli.page,
        cache_path=cli.cache_path,
        execution_in_temp=True,
        force=cli.force,
    )

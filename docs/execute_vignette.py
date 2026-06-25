"""Execute myst-nb vignettes into docs/.jupyter_cache."""

from __future__ import annotations

import shutil
import sqlite3
import sys
from pathlib import Path

from myst_parser.config.main import MdParserConfig
from myst_nb.core.config import NbParserConfig
from myst_nb.core.execute import create_client
from myst_nb.core.read import create_nb_reader

DOCS_ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = DOCS_ROOT / ".jupyter_cache"
VIGNETTES_DIR = DOCS_ROOT / "source" / "vignettes"


class _Logger:
    prefix: str = ""

    def info(self, msg: str, subtype: str | None = None) -> None:
        print(f"{self.prefix}{msg}", flush=True)

    def warning(self, msg: str, subtype: str | None = None) -> None:
        print(f"{self.prefix}WARNING: {msg}", flush=True)

    def debug(self, msg: str, subtype: str | None = None) -> None:
        pass


def list_vignettes() -> list[str]:
    return sorted(p.stem for p in VIGNETTES_DIR.glob("*.md"))


def execute_vignette(
    name: str,
    *,
    cache_path: Path | None = None,
    execution_in_temp: bool = False,
) -> dict:
    path = VIGNETTES_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Vignette not found: {path}")

    cache_path = cache_path or DEFAULT_CACHE
    cache_path.mkdir(parents=True, exist_ok=True)

    nb_config = NbParserConfig(
        execution_mode="cache",
        execution_cache_path=str(cache_path),
        execution_timeout=200,
        execution_allow_errors=True,
        execution_in_temp=execution_in_temp,
    )
    md_config = MdParserConfig(enable_extensions={"colon_fence"})
    content = path.read_text(encoding="utf-8")
    nb_reader = create_nb_reader(str(path), md_config, nb_config, content)
    if nb_reader is None:
        raise ValueError(f"Not a myst-nb vignette: {path}")

    notebook = nb_reader.read(content)
    logger = _Logger()
    logger.prefix = f"[{name}] "
    with create_client(
        notebook, str(path), nb_config, logger, nb_reader.read_fmt
    ) as client:
        metadata = client.exec_metadata or {}

    print(f"[{name}] finished", flush=True)
    return {"name": name, "metadata": metadata}


def merge_caches(
    source_dirs: list[Path], target_dir: Path = DEFAULT_CACHE
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    target_executed = target_dir / "executed"
    target_executed.mkdir(parents=True, exist_ok=True)
    version_file = target_dir / "__version__.txt"
    if not version_file.exists() and source_dirs:
        first_version = next(
            (d / "__version__.txt" for d in source_dirs if (d / "__version__.txt").exists()),
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
    vignette = sys.argv[1] if len(sys.argv) > 1 else "basic_tutorial_scRNAseq"
    execute_vignette(vignette, execution_in_temp=True)

import gzip
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from ...utils.logging import logger
from ...utils.progress import iter_progress

__all__ = ["GffReader"]


class GffReader:
    """Reader for a GFF3 format file."""

    def __init__(
        self,
        gff_fn: str,
        up_offset: int = 1000,
        down_offset: int = 500,
        chunk_size: int = 100000,
    ):
        self.gffFn = gff_fn
        self.header = self.fetch_header_lines()
        self.nHeaderLines = len(self.header)
        self.up = up_offset
        self.down = down_offset
        self.chunksize = chunk_size

    def fetch_header_lines(self) -> list[str]:
        """Fetch header lines starting with `#` from a GFF file."""
        temp = []
        if self.gffFn.endswith("gz"):
            h = gzip.open(self.gffFn, "rt")
        else:
            h = open(self.gffFn)
        for line in h:
            if line[0] != "#":
                break
            temp.append(line.rstrip())
        h.close()
        return temp

    def stream(self) -> pd.DataFrame:
        """Stream the GFF file in chunks as pandas dataframes."""
        stream = pd.read_csv(
            self.gffFn,
            skiprows=self.nHeaderLines,
            chunksize=int(self.chunksize),
            sep="\t",
            header=None,
        )
        for df in stream:
            yield df

    def get_promoter(self, v: pd.Series) -> tuple[int, int]:
        """Create strand-aware promoter coordinates."""
        if v[6] == "+":
            return max(0, v[3] - self.up), v[3] + self.down
        if v[6] == "-":
            return v[4] - 1 - self.down, v[4] + self.up
        raise ValueError(f"ERROR: Unknown symbol for strand: {v[6]}")

    def get_body(self, v: pd.Series) -> tuple[int, int]:
        """Create strand-aware gene body and promoter coordinates."""
        if v[6] == "+":
            return max(v[3] - self.up, 0), v[4]
        if v[6] == "-":
            return v[3], v[4] + self.up
        raise ValueError(f"ERROR: Unknown symbol for strand: {v[6]}")

    @staticmethod
    def get_ids_names(v: pd.Series) -> tuple[str | None, str | None]:
        """Extract gene ID and gene name values from a GFF record."""
        gid, name = None, None
        for i in v[8].split(";"):
            j, k = i.split("=")
            if j == "gene_id":
                gid = k
            elif j == "gene_name":
                name = k
        return gid, name

    @staticmethod
    def d_apply(d: pd.DataFrame, func: Callable[[pd.Series], Any]) -> np.ndarray:
        """Apply a function over dataframe rows and return an array."""
        values = d.apply(func, axis=1)
        return np.array(list(values.values))

    def to_bed(
        self,
        out_bed_fn: str,
        flavour: str = "body",
    ) -> None:
        """Convert gene annotations from GFF to a six-column BED file."""
        bed = []
        if flavour not in ["body", "promoter"]:
            raise ValueError(
                "ERROR: The value of flavour must be one of either 'body' or 'promoter'"
            )
        for df in iter_progress(
            self.stream(),
            desc="Reading gene annotations",
        ):
            df = df[df[2] == "gene"]
            if flavour == "promoter":
                coords = self.d_apply(df, self.get_promoter)
            else:
                coords = self.d_apply(df, self.get_body)

            anno = self.d_apply(df, self.get_ids_names)
            odf = pd.DataFrame(
                {
                    0: df[0].values,
                    1: coords[:, 0],
                    2: coords[:, 1],
                    3: anno[:, 0],
                    4: anno[:, 1],
                    5: df[6].values,
                }
            )
            bed.append(odf)
        merged_bed = pd.concat(bed)
        merged_bed.to_csv(out_bed_fn, sep="\t", header=False, index=False)
        logger.info(f"{merged_bed.shape[0]} genes saved to BED file")
        return None

"""
- Classes:
    - GffReader: A class for reading and GFF3 format files and convert to BED format files

- Methods:
    - create_bed_from_coord_ids:
    - binary_search:
    - get_feature_mappings:
    - create_counts_mat:
    - coordinate_melding:
"""

import gzip
import logging

import numpy as np
import pandas as pd
from numba import jit
from scipy.sparse import coo_matrix
import zarr

from .utils import controlled_compute, logger, tqdmbar
from .writers import create_zarr_count_assay

__all__ = ["GffReader", "coordinate_melding"]


class GffReader:
    """Reader for a GFF3 format file."""

    def __init__(
        self,
        gff_fn: str,
        up_offset: int = 1000,
        down_offset: int = 500,
        chunk_size=100000,
    ):
        """

        Args:
            gff_fn: Path to GFF file to be read
            up_offset: Upstream shift from TSS. (Default value: 1000 bases)
            down_offset: Only used when flavour='body' in `to_bed`.
                         Downstream shift from TSS (Default value: 500 bases)
            chunk_size: Number of lines to read at a time
        """

        self.gffFn = gff_fn
        self.header = self.fetch_header_lines()
        self.nHeaderLines = len(self.header)
        self.up = up_offset
        self.down = down_offset
        self.chunksize = chunk_size

    def fetch_header_lines(self) -> list[str]:
        """Fetch header lines (starting with '#') from GFF file.

        Returns: A list of all the header lines
        """
        temp = []
        if self.gffFn.endswith("gz"):
            h = gzip.open(self.gffFn, "rt")
        else:
            h = open(self.gffFn)
        for line in h:
            if line[0] != "#":
                break
            else:
                temp.append(line.rstrip())
        h.close()
        return temp

    def stream(self) -> pd.DataFrame:
        """Stream the GFF file in chunks as Pandas DataFrame.

        Returns:
            Pandas DataFrame of the GFF file using \t as separator
        """
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
        """Create strand-aware promoter coordinates using gene start and end
        coordinates.

        Args:
            v: A row from the GFF file in Pandas Series format

        Returns:
            A Tuple of start and end coordinates for the promoter
        """
        if v[6] == "+":
            return max(0, v[3] - self.up), v[3] + self.down
        elif v[6] == "-":
            return v[4] - 1 - self.down, v[4] + self.up
        else:
            raise ValueError(f"ERROR: Unknown symbol for strand: {v[6]}")

    def get_body(self, v: pd.Series) -> tuple[int, int]:
        """Create strand-aware gene body + promoter coordinates using gene
        start and end coordinates.

        Args:
            v: A row from the GFF file in Pandas Series format

        Returns:
            A Tuple of start and end coordinates
        """
        if v[6] == "+":
            return max(v[3] - self.up, 0), v[4]
        elif v[6] == "-":
            return v[3], v[4] + self.up
        else:
            raise ValueError(f"ERROR: Unknown symbol for strand: {v[6]}")

    @staticmethod
    def get_ids_names(v: pd.Series) -> tuple[str, str]:
        """Extracts gene_id and gene_name values from last (9th) column of GFF
        file record.

        Args:
            v: A Pandas Series representing a row from GFF file

        Returns:
            Tuple of gene ID and gene name
        """
        gid, name = None, None
        for i in v[8].split(";"):
            j, k = i.split("=")
            if j == "gene_id":
                gid = k
            elif j == "gene_name":
                name = k
        return gid, name

    @staticmethod
    def d_apply(d: pd.DataFrame, func) -> np.ndarray:
        """A convenience method to apply arbitrary functions over a dataframe.

        Args:
            d: A pandas dataframe over which function is to applied over axis 1
            func: Function to be applied

        Returns:
            Numpy array of values returned by func
        """
        v = d.apply(func, axis=1)
        return np.array(list(v.values))

    def to_bed(
        self,
        out_bed_fn: str,
        flavour: str = "body",
    ) -> None:
        """Converts the 'gene' annotations from the GFF file to a 6-column BED
        file. The columns '3' and '4' contain te gene names and gene IDs
        respectively.

        Args:
            out_bed_fn: Path of output BED file.
            flavour: Should be either 'promoter' or 'body' (Default value: 'body')

        Returns:
            None
        """
        bed = []
        if flavour not in ["body", "promoter"]:
            raise ValueError(
                "ERROR: The value of flavour must be one of either 'body' or 'promoter'"
            )
        for df in tqdmbar(self.stream()):
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
        bed = pd.concat(bed)
        bed.to_csv(out_bed_fn, sep="\t", header=False, index=False)
        logger.info(f"{bed.shape[0]} genes saved to BED file")
        return None


def create_bed_from_coord_ids(ids: list) -> pd.DataFrame:
    """Creates a 3 column BED file from list of strings in format: <chr:start-
    end>

    Args:
        ids: List of strings in format: <chr:start-end>

    Returns:
        A 3 column Pandas dataframe sorted by chromosome and start position
    """

    out = []
    for i in ids:
        j = i.split(":")
        o = [j[0], int(j[1].split("-")[0]), int(j[1].split("-")[1])]
        out.append(o)
    return pd.DataFrame(out).sort_values(by=[0, 1])


@jit(nopython=True)
def binary_search(ranges: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Identify the position of intervals in `queries` in the `ranges` interval
    list using binary search algorithm.

    Args:
        ranges: A sorted numpy array of shape (n, 2)
        queries: A sorted numpy array of shape (m, 2)

    Returns:
        A numpy array of shape (m, 2). The values are indices of ranges which overlapped the query
        intervals.
    """

    max_len = (ranges[:, 1] - ranges[:, 0]).max()
    n = queries.shape[0]
    ret_val = np.full((n, 2), 0)
    for i in range(n):
        start = max(0, queries[i][0] - max_len)
        end = queries[i][1]

        # left search
        lo = 0
        hi = ranges.shape[0]
        while lo < hi:
            mid = (lo + hi) // 2
            if ranges[mid][0] < start:
                lo = mid + 1
            else:
                hi = mid
        starts_after = lo

        # right search
        lo = 0
        hi = ranges.shape[0]
        while lo < hi:
            mid = (lo + hi) // 2
            if end < ranges[mid][0]:
                hi = mid
            else:
                lo = mid + 1
        ends_before = lo

        if starts_after == ends_before:
            ret_val[i][0] = -1
            ret_val[i][1] = -1

        else:
            start = queries[i][0]
            m_pos_s, m_pos_e = -1, -1
            for j in range(starts_after, ends_before):
                if start < ranges[j][1] and end > ranges[j][0]:
                    if m_pos_s == -1:
                        m_pos_s = j
                    m_pos_e = j + 1
            ret_val[i][0] = m_pos_s
            ret_val[i][1] = m_pos_e

    return ret_val


def get_ranges(df: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
    """Convenience function to extract column 1 and 2 of the dataframe df and
    return int type values.

    Args:
        df: A pandas dataframe with minimum three columns.
        idx: Boolean indexer for the dataframe

    Returns:
        A numpy array with start and end positions
    """

    return df[[1, 2]][idx].values.astype(int)


def get_feature_mappings(
    peaks_bed_df: pd.DataFrame, features_bed_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Identify which intervals from `features_bed_df` overlap with those from
    `peaks_bed_df`.

    Args:
        peaks_bed_df: DataFrame containing reference intervals. Must have atleast 3 columns:
                      'chrom', 'start', 'end'. It should be sorted by chromosome and start
        features_bed_df: DataFrame containing reference intervals. Must have at least 5 columns:
                         'chrom', 'start', 'end', 'ids', 'names'.
                         It should be sorted by chromosome and start

    Returns:
        A tuple containing three numpy arrays, the first two are simply the 'ids' and 'names'
        columns from `features_bed_df`. These are returned to ensure that they are in same order
        as the indices in the third array. The third array has shape (features_bed_df.shape[0], 2).
        The values are indices of the overlapping intervals from peaks_bed_df. If no overlap is found
        then that row has value [-1, -1].
    """

    cross_indices = []
    feats_ids, feats_names = [], []
    id_counter = {}
    n_no_match = 0
    for chrom in peaks_bed_df[0].unique():
        feats_chrom_idx = np.array(features_bed_df[0] == chrom)
        if feats_chrom_idx.shape[0] == 0:
            logger.warning(f"Chromosome {chrom} not in the input feature BED")
            continue

        feats_names.extend(features_bed_df[4][feats_chrom_idx].values)
        # Making IDs unique
        for i in features_bed_df[3][feats_chrom_idx].values:
            if i not in id_counter:
                id_counter[i] = 0
            id_counter[i] += 1
            if id_counter[i] > 1:
                i = i + f"_{id_counter[i]}"
            feats_ids.append(i)

        peaks_chrom_idx = np.array(peaks_bed_df[0] == chrom)

        match_indices = binary_search(
            get_ranges(peaks_bed_df, peaks_chrom_idx),
            get_ranges(features_bed_df, feats_chrom_idx),
        ).astype(int)

        # Now this is the main trick. Since the peak_bed_df is a sorted dataframe.
        # The dataframe index might itself not be in sorted order.
        peak_idx = np.array(peaks_bed_df.index[peaks_chrom_idx])
        for i in match_indices:
            if i[0] == -1:
                assert i[1] == -1
                cross_indices.append(None)
                n_no_match += 1
            else:
                cross_indices.append(list(peak_idx[i[0] : i[1]]))

    if len(feats_ids) == 0:
        raise ValueError(
            "ERROR: None of the features were found in the assay. Melding failed"
        )
    feats_ids = np.array(feats_ids)
    feats_names = np.array(feats_names)
    cross_indices = np.array(cross_indices, dtype=object)
    if n_no_match == len(cross_indices):
        logging.critical(
            "None of the provided features overlap with the peak coordinates. "
            "Melding has possibly failed."
        )
    else:
        logger.info(
            f"{n_no_match}/{feats_ids.shape[0]} features did not overlap with any peak"
        )
    if len(set(feats_ids)) != feats_ids.shape[0]:
        raise ValueError(
            "ERROR: encountered an unexpected error. Somehow the feature ids are not unique "
            "despite our attempt to make them unique by appending a suffix. Please report this "
            "bug on Github"
        )
    assert feats_ids.shape[0] == feats_names.shape[0] == cross_indices.shape[0]
    return feats_ids, feats_names, cross_indices


def create_counts_mat(
    assay,
    store: zarr.Array,
    cross_map: np.ndarray,
    scalar_coeff: float,
    renormalization: bool,
) -> None:
    """Populate the count matrix in the Zarr store.

    Args:
        assay: Scarf Assay object which contains the rawData attribute representing a chunked array of count matrix
        store: Output Zarr Dataset
        cross_map: Mapping of indices. as obtained from get_feature_mappings function
        scalar_coeff: An arbitrary scalar multiplier. Only used when renormalization is True.
        renormalization: Whether to rescale the sum of feature values for each cell to `scalar_coeff`

    Returns:
        None
    """

    idx = np.where(cross_map)[0]
    feat_idx = np.repeat(idx, list(map(len, cross_map[idx])))
    peak_idx = np.array(
        sum(list(cross_map[idx]), [])
    )  # There is no guarantee that these are in sorted order
    assert feat_idx.shape == peak_idx.shape

    n_term_per_doc = assay.cells.fetch_all(assay.name + "_nFeatures")
    n_docs = n_term_per_doc.shape[0]
    n_docs_per_term = assay.feats.fetch_all("nCells")

    s = 0
    for a in tqdmbar(assay.rawData.blocks, total=assay.rawData.numblocks[0]):
        a = controlled_compute(a, assay.nthreads)
        tf = a / n_term_per_doc[s : s + a.shape[0]].reshape(-1, 1)
        idf = np.log2(1 + (n_docs / (n_docs_per_term + 1)))
        a = tf * idf

        df = pd.DataFrame(a[:, peak_idx]).T
        df["fidx"] = feat_idx
        df = df.groupby("fidx").sum().T
        if renormalization:
            df = (scalar_coeff * df) / df.sum(axis=1).values.reshape(-1, 1)
        assert df.shape[1] == idx.shape[0]

        coord_renamer = dict(enumerate(df.columns))
        coo = coo_matrix(df.values)
        coo.col = np.array([coord_renamer[x] for x in coo.col])
        store.set_coordinate_selection((s + coo.row, coo.col), coo.data)
        s += a.shape[0]


def coordinate_melding(
    assay,
    workspace: str | None,
    feature_bed: pd.DataFrame,
    new_assay_name: str,
    peaks_col: str = "ids",
    scalar_coeff: float = 1e5,
    renormalization: bool = True,
) -> None:
    """This function the coordinates of the features of the given assay and
    overlaps (genomics intersection) them with a user provided set of external
    features. Based on the overlap the original feature values are transferred
    to the new features and the new feature set is saved as a new assay. If the
    new feature overlaps with multiple original features than we sum the values
    from original feature. Similarly, a single original feature may overlap
    with multiple new features and hence its value will used for multiple new
    features. The values are TF-IDF normalized before they are saved into the
    new assay.

    Args:
        assay: Scarf Assay object which contains the rawData attribute representing a chunked array of count matrix.
        workspace:
        feature_bed: DataFrame containing reference intervals. Must have at least 5 columns representing
                    'chrom', 'start', 'end', 'ids', 'names'. But the column names should 0,1,2,3,4
        new_assay_name: Name of the new melded assay
        peaks_col: The name of the column in the feature metadata that contains the coordinate information. This
                   column should have coordinates in this format <chrom:start-end>. The values from this
                   column will be processed into BED format dataframe using `create_bed_from_coord_ids`
        scalar_coeff: An arbitrary scalar multiplier. Only used when renormalization is True (Default value: 1e5)
        renormalization: Whether to rescale the sum of feature values for each cell to `scalar_coeff`
                         (Default value: True)

    Returns:
        None
    """

    peaks_bed = create_bed_from_coord_ids(assay.feats.fetch_all(peaks_col))
    feat_ids, feat_names, mappings = get_feature_mappings(peaks_bed, feature_bed)

    from .storage.zarr_store import zarr_root_path
    import zarr

    store_root = zarr.open_group(zarr_root_path(assay.z), mode="r+")
    g = create_zarr_count_assay(
        z=store_root,
        assay_name=new_assay_name,
        workspace=workspace,
        chunk_size=assay.rawData.chunksize,
        n_cells=assay.rawData.shape[0],
        feat_ids=feat_ids,
        feat_names=feat_names,
        dtype="float",
    )

    create_counts_mat(
        assay=assay,
        store=g,
        cross_map=mappings,
        scalar_coeff=scalar_coeff,
        renormalization=renormalization,
    )
    return None

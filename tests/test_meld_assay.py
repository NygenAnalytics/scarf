import numpy as np
import pandas as pd
import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.chunked import ChunkedArray
from scarf.meld_assay import (
    GffReader,
    binary_search,
    create_bed_from_coord_ids,
    create_counts_mat,
    get_feature_mappings,
)
from scarf.writers import create_zarr_dataset


def _features_bed(rows):
    return pd.DataFrame({i: [r[i] for r in rows] for i in range(6)}).sort_values(
        by=[0, 1]
    )


class _FakeMeta:
    def __init__(self, columns):
        self._columns = columns

    def fetch_all(self, name):
        return np.asarray(self._columns[name])


class _FakeRawData:
    def __init__(self, blocks):
        self.blocks = blocks
        self.numblocks = (len(blocks),)
        self.shape = (sum(b.shape[0] for b in blocks), blocks[0].shape[1])

    def stream_blocks(self, nthreads=None, msg=None, prefetch=None):
        # Mirrors ChunkedArray.stream_blocks: yields materialized, in-order blocks.
        yield from (np.asarray(b) for b in self.blocks)


class _FakeAssay:
    name = "ATAC"
    nthreads = 1

    def __init__(self, cells, feats, raw_data):
        self.cells = cells
        self.feats = feats
        self.rawData = raw_data


def _reference_melded(
    raw, n_features_per_cell, n_cells_per_peak, mapping_dense, scalar_coeff, renorm
):
    n_docs = raw.shape[0]
    idf = np.log2(1 + (n_docs / (n_cells_per_peak + 1)))
    tf = raw / n_features_per_cell.reshape(-1, 1)
    tfidf = tf * idf
    melded = tfidf @ mapping_dense
    if not renorm:
        return melded
    row_sums = melded.sum(axis=1)
    out = np.zeros_like(melded)
    nz = row_sums != 0
    out[nz] = (scalar_coeff * melded[nz]) / row_sums[nz].reshape(-1, 1)
    return out


def _old_style_melded(
    raw,
    n_features_per_cell,
    n_cells_per_peak,
    feature_to_peaks,
    scalar_coeff,
    renorm,
):
    n_docs = raw.shape[0]
    idf = np.log2(1 + (n_docs / (n_cells_per_peak + 1)))
    tf = raw / n_features_per_cell.reshape(-1, 1)
    tfidf = tf * idf

    idx = np.array([i for i, peaks in enumerate(feature_to_peaks) if len(peaks)])
    if idx.size == 0:
        melded = np.zeros((raw.shape[0], len(feature_to_peaks)))
    else:
        feat_idx = np.repeat(idx, [len(feature_to_peaks[i]) for i in idx])
        peak_idx = np.array(sum((feature_to_peaks[i] for i in idx), []))
        df = pd.DataFrame(tfidf[:, peak_idx]).T
        df["fidx"] = feat_idx
        df = df.groupby("fidx").sum().T
        melded = np.zeros((raw.shape[0], len(feature_to_peaks)))
        melded[:, df.columns.to_numpy(dtype=int)] = df.values

    if not renorm:
        return melded
    row_sums = melded.sum(axis=1)
    out = np.zeros_like(melded)
    nz = row_sums != 0
    out[nz] = (scalar_coeff * melded[nz]) / row_sums[nz].reshape(-1, 1)
    return out


def _feature_to_peaks_from_intervals(peaks, features):
    feature_to_peaks = []
    for _, feat in features.iterrows():
        peak_hits = []
        for peak_pos, peak in peaks.iterrows():
            if feat[0] == peak[0] and feat[1] < peak[2] and feat[2] > peak[1]:
                peak_hits.append(int(peak_pos))
        feature_to_peaks.append(peak_hits)
    return feature_to_peaks


def _build_melding_scenario():
    peaks = create_bed_from_coord_ids(
        ["chr1:100-200", "chr1:150-250", "chr1:400-500", "chr2:100-200"]
    )
    features = _features_bed(
        [
            ("chr1", 120, 160, "a", "A", "+"),  # overlaps peaks 0, 1
            ("chr1", 300, 350, "b", "B", "+"),  # overlaps nothing
            ("chr2", 150, 180, "c", "C", "+"),  # overlaps peak 3
        ]
    )
    _, _, mapping = get_feature_mappings(peaks, features)

    # Rows are cells, columns are peaks (assay feature order).
    raw = np.array(
        [
            [1.0, 0.0, 2.0, 3.0],
            [0.0, 4.0, 0.0, 0.0],
            [
                0.0,
                0.0,
                5.0,
                0.0,
            ],  # only peak 2, which maps to no feature -> melded row is all zero
        ]
    )
    n_features_per_cell = np.array([3.0, 1.0, 1.0])
    n_cells_per_peak = np.array([1.0, 1.0, 2.0, 1.0])

    cells = _FakeMeta({"ATAC_nFeatures": n_features_per_cell})
    feats = _FakeMeta({"nCells": n_cells_per_peak})
    raw_data = _FakeRawData([raw[0:2], raw[2:3]])
    assay = _FakeAssay(cells, feats, raw_data)
    return assay, mapping, raw, n_features_per_cell, n_cells_per_peak


def _run_create_counts_mat(assay, mapping, scalar_coeff, renormalization):
    n_features = mapping.shape[1]
    n_cells = assay.rawData.shape[0]
    group = zarr.open_group(store=MemoryStore(), mode="w")
    store = create_zarr_dataset(
        group, "counts", (2, n_features), "float", (n_cells, n_features)
    )
    create_counts_mat(
        assay=assay,
        store=store,
        mapping=mapping,
        scalar_coeff=scalar_coeff,
        renormalization=renormalization,
    )
    return store[:]


GFF_CONTENT = """##gff-version 3
# test annotation
chr1\t.\tgene\t1000\t2000\t.\t+\t.\tgene_id=gene_a;gene_name=GENE_A
chr1\t.\tgene\t3000\t4500\t.\t-\t.\tgene_id=gene_b;gene_name=GENE_B
chr2\t.\tgene\t500\t1500\t.\t+\t.\tgene_id=gene_c;gene_name=GENE_C
"""


def test_gff_reader_parses_header_and_streams(tmp_path):
    gff_path = tmp_path / "test.gff"
    gff_path.write_text(GFF_CONTENT)

    reader = GffReader(str(gff_path), up_offset=500, down_offset=200, chunk_size=2)
    assert len(reader.header) == 2
    chunks = list(reader.stream())
    assert len(chunks) == 2
    assert chunks[0].shape[0] == 2
    assert set(chunks[0][2]) == {"gene"}


def test_gff_reader_promoter_and_body_coordinates(tmp_path):
    gff_path = tmp_path / "coords.gff"
    gff_path.write_text(GFF_CONTENT)
    reader = GffReader(str(gff_path), up_offset=500, down_offset=200)
    plus_row = pd.Series([None] * 9)
    plus_row[3] = 1000
    plus_row[4] = 2000
    plus_row[6] = "+"

    promoter_start, promoter_end = reader.get_promoter(plus_row)
    assert promoter_start == 500
    assert promoter_end == 1200

    body_start, body_end = reader.get_body(plus_row)
    assert body_start == 500
    assert body_end == 2000

    minus_row = plus_row.copy()
    minus_row[3] = 3000
    minus_row[4] = 4500
    minus_row[6] = "-"
    m_start, m_end = reader.get_promoter(minus_row)
    assert m_start == 4499 - reader.down
    assert m_end == 4500 + reader.up


def test_gff_reader_get_ids_names():
    row = pd.Series([None] * 9)
    row[8] = "gene_id=abc;gene_name=XYZ"
    gene_id, gene_name = GffReader.get_ids_names(row)
    assert gene_id == "abc"
    assert gene_name == "XYZ"


def test_gff_reader_to_bed_promoter(tmp_path):
    gff_path = tmp_path / "genes.gff"
    gff_path.write_text(GFF_CONTENT)
    bed_path = tmp_path / "out.bed"

    reader = GffReader(str(gff_path))
    reader.to_bed(str(bed_path), flavour="promoter")

    bed = pd.read_csv(bed_path, sep="\t", header=None)
    assert bed.shape[0] == 3
    assert bed.shape[1] == 6


def test_gff_reader_to_bed_rejects_unknown_flavour(tmp_path):
    gff_path = tmp_path / "genes.gff"
    gff_path.write_text(GFF_CONTENT)
    reader = GffReader(str(gff_path))
    with pytest.raises(ValueError, match="flavour"):
        reader.to_bed(str(tmp_path / "out.bed"), flavour="exon")


def test_create_bed_from_coord_ids_sorts_intervals():
    bed = create_bed_from_coord_ids(["chr2:100-200", "chr1:50-150", "chr1:300-400"])
    assert bed.iloc[0, 0] == "chr1"
    assert bed.iloc[0, 1] == 50
    assert bed.iloc[-1, 0] == "chr2"


def test_binary_search_finds_overlapping_ranges():
    ranges = np.array([[0, 10], [10, 20], [20, 30], [30, 40]], dtype=np.int64)
    queries = np.array([[5, 8], [15, 18], [25, 28]], dtype=np.int64)
    hits = binary_search(ranges, queries)
    assert hits.shape == (3, 2)
    assert hits[0, 0] <= hits[0, 1]


def test_get_feature_mappings_builds_sparse_overlap_matrix():
    peaks = create_bed_from_coord_ids(
        ["chr1:100-200", "chr1:150-250", "chr1:400-500", "chr2:100-200"]
    )
    features = _features_bed(
        [
            ("chr1", 120, 160, "a", "A", "+"),
            ("chr1", 300, 350, "b", "B", "+"),
            ("chr2", 150, 180, "c", "C", "+"),
        ]
    )
    feat_ids, feat_names, mapping = get_feature_mappings(peaks, features)

    assert list(feat_ids) == ["a", "b", "c"]
    assert list(feat_names) == ["A", "B", "C"]
    assert mapping.shape == (4, 3)

    expected = np.zeros((4, 3))
    expected[[0, 1], 0] = 1  # feature A overlaps peaks at original positions 0 and 1
    expected[3, 2] = 1  # feature C overlaps peak at original position 3
    np.testing.assert_array_equal(mapping.toarray(), expected)


def test_get_feature_mappings_keeps_features_without_peaks():
    peaks = create_bed_from_coord_ids(["chr1:100-200"])
    features = _features_bed(
        [
            ("chr1", 120, 160, "a", "A", "+"),
            ("chr3", 100, 200, "d", "D", "+"),
        ]
    )
    feat_ids, _, mapping = get_feature_mappings(peaks, features)

    # Feature on peak-less chromosome must be retained (not silently dropped)
    assert list(feat_ids) == ["a", "d"]
    assert mapping.shape[1] == 2
    assert mapping.getcol(1).nnz == 0


def test_get_feature_mappings_uniquifies_duplicate_ids():
    peaks = create_bed_from_coord_ids(["chr1:100-200"])
    features = _features_bed(
        [
            ("chr1", 120, 160, "dup", "A", "+"),
            ("chr1", 500, 600, "dup", "B", "+"),
        ]
    )
    feat_ids, _, _ = get_feature_mappings(peaks, features)
    assert list(feat_ids) == ["dup", "dup_2"]


def test_get_feature_mappings_follows_feature_bed_order():
    peaks = create_bed_from_coord_ids(["chr1:100-200", "chr2:100-200"])
    features = _features_bed(
        [
            ("chr2", 120, 160, "b", "B", "+"),
            ("chr3", 100, 200, "c", "C", "+"),
            ("chr1", 120, 160, "a", "A", "+"),
        ]
    )
    feat_ids, _, mapping = get_feature_mappings(peaks, features)

    assert list(feat_ids) == ["a", "b", "c"]
    assert mapping.getcol(2).nnz == 0


def test_mapping_matmul_sums_overlapping_peaks():
    peaks = create_bed_from_coord_ids(
        ["chr1:100-200", "chr1:150-250", "chr1:400-500", "chr2:100-200"]
    )
    features = _features_bed(
        [
            ("chr1", 120, 160, "a", "A", "+"),
            ("chr1", 300, 350, "b", "B", "+"),
            ("chr2", 150, 180, "c", "C", "+"),
        ]
    )
    _, _, mapping = get_feature_mappings(peaks, features)

    tfidf = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [0.5, 0.0, 0.0, 1.5],
        ]
    )
    melded = tfidf @ mapping.toarray()

    # Feature A melds peaks 0 and 1, feature B nothing, feature C peak 3
    np.testing.assert_allclose(melded[:, 0], tfidf[:, 0] + tfidf[:, 1])
    np.testing.assert_allclose(melded[:, 1], np.zeros(2))
    np.testing.assert_allclose(melded[:, 2], tfidf[:, 3])


def test_create_counts_mat_matches_old_groupby_oracle():
    peaks = create_bed_from_coord_ids(
        ["chr1:100-200", "chr1:150-250", "chr1:210-260", "chr2:50-120"]
    )
    features = _features_bed(
        [
            ("chr1", 120, 240, "a", "A", "+"),
            ("chr1", 500, 600, "b", "B", "+"),
            ("chr2", 70, 100, "c", "C", "+"),
        ]
    )
    _, _, mapping = get_feature_mappings(peaks, features)
    raw = np.array(
        [
            [3.0, 0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0, 5.0],
            [4.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 7.0, 3.0],
        ]
    )
    n_feat = np.array([2.0, 2.0, 2.0, 2.0])
    n_cells_peak = np.array([2.0, 2.0, 2.0, 2.0])
    assay = _FakeAssay(
        _FakeMeta({"ATAC_nFeatures": n_feat}),
        _FakeMeta({"nCells": n_cells_peak}),
        _FakeRawData([raw[:2], raw[2:]]),
    )
    old_oracle = _old_style_melded(
        raw,
        n_feat,
        n_cells_peak,
        _feature_to_peaks_from_intervals(peaks, features),
        scalar_coeff=1e4,
        renorm=True,
    )
    written = _run_create_counts_mat(
        assay, mapping, scalar_coeff=1e4, renormalization=True
    )
    np.testing.assert_allclose(written, old_oracle)


def test_create_counts_mat_reads_real_chunked_array_stream_blocks():
    assay, mapping, raw, n_feat, n_cells_peak = _build_melding_scenario()
    group = zarr.open_group(store=MemoryStore(), mode="w")
    counts = create_zarr_dataset(group, "raw", (2, raw.shape[1]), "float", raw.shape)
    counts[:] = raw
    assay.rawData = ChunkedArray(counts, nthreads=2)

    written = _run_create_counts_mat(
        assay, mapping, scalar_coeff=1e4, renormalization=True
    )
    expected = _reference_melded(
        raw, n_feat, n_cells_peak, mapping.toarray(), scalar_coeff=1e4, renorm=True
    )
    np.testing.assert_allclose(written, expected)


def test_create_counts_mat_without_renormalization():
    assay, mapping, raw, n_feat, n_cells_peak = _build_melding_scenario()
    written = _run_create_counts_mat(
        assay, mapping, scalar_coeff=1e4, renormalization=False
    )
    expected = _reference_melded(
        raw, n_feat, n_cells_peak, mapping.toarray(), scalar_coeff=1e4, renorm=False
    )
    np.testing.assert_allclose(written, expected)


def test_create_counts_mat_with_renormalization_and_zero_sum_cell():
    assay, mapping, raw, n_feat, n_cells_peak = _build_melding_scenario()
    written = _run_create_counts_mat(
        assay, mapping, scalar_coeff=1e4, renormalization=True
    )
    expected = _reference_melded(
        raw, n_feat, n_cells_peak, mapping.toarray(), scalar_coeff=1e4, renorm=True
    )
    np.testing.assert_allclose(written, expected)

    # The cell whose only signal maps to no feature must be all zeros, not NaN.
    assert np.all(np.isfinite(written))
    np.testing.assert_array_equal(written[2], np.zeros(mapping.shape[1]))
    # Non-empty cells are rescaled to sum to scalar_coeff.
    np.testing.assert_allclose(written[0].sum(), 1e4)
    np.testing.assert_allclose(written[1].sum(), 1e4)

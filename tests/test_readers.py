import numpy as np


def test_toy_crdir_assay_feats_table(toy_crdir_reader):
    assert np.all(
        toy_crdir_reader.assayFeats.columns
        == np.array(["RNA", "ADT", "RNA", "HTO", "RNA"])
    )
    assert np.all(
        toy_crdir_reader.assayFeats.values[1:]
        == [[0, 1, 3, 5, 6], [1, 3, 5, 6, 7], [1, 2, 2, 1, 1]]
    )


def test_toy_crdir_reader_cells_feats(toy_crdir_reader):
    assert toy_crdir_reader.nCells == 3
    assert toy_crdir_reader.nFeatures == 8
    assert toy_crdir_reader.cell_names() == ["b1", "b2", "b3"]
    assert toy_crdir_reader.feature_names() == [
        "g1",
        "a1",
        "a2",
        "g2",
        "g3",
        "h1",
        "g4",
        "a3",
    ]
    assert toy_crdir_reader.feature_ids() == [
        "g1",
        "a1",
        "a2",
        "g2",
        "g3",
        "h1",
        "g4",
        "a3",
    ]

def test_toy_crdir_empty(toy_crdir_empty):
    assert toy_crdir_empty.nCells == 0
    assert toy_crdir_empty.nFeatures == 4
    assert toy_crdir_empty.feature_names() == [
        "g1",
        "a1",
        "a2",
        "g2",
    ]
    assert toy_crdir_empty.feature_ids() == [
        "g1",
        "a1",
        "a2",
        "g2",
    ]
    # check for raise ValueError
    try:
        toy_crdir_empty.read_header()
    except ValueError:
        pass

def test_crh5reader(crh5_reader):
    assert crh5_reader.nCells == 892
    assert crh5_reader.nFeatures == 36611
    n_assay_feats = list(crh5_reader.assayFeats.T.nFeatures.values)
    assert n_assay_feats == [36601, 10]


def test_crdir_reader(crdir_reader):
    assert crdir_reader.nCells == 892
    assert crdir_reader.nFeatures == 36601  # Does not contain 10 ADTs


def test_h5ad_reader(h5ad_reader):
    assert h5ad_reader.nCells == 3696 == len(h5ad_reader.cell_ids())
    assert h5ad_reader.nFeatures == 27998 == len(h5ad_reader.feat_names())


def test_loom_reader(loom_reader):
    assert loom_reader.nCells == 298 == len(loom_reader.cell_ids())
    assert loom_reader.nFeatures == 16892 == len(loom_reader.feature_names())

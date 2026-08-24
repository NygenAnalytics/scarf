import numpy as np
import pandas as pd
import pytest

from scarf.features.enrichment.net import prepare_network, read_gmt


def test_read_gmt_parses_sets_and_ignores_descriptions(tmp_path):
    path = tmp_path / "sets.gmt"
    path.write_text(
        "\nSet B\tdescription\tGene3\tGene4\nSet A\tna\tGene1\tGene2\n",
        encoding="utf-8",
    )

    result = read_gmt(path)

    pd.testing.assert_frame_equal(
        result,
        pd.DataFrame(
            {
                "source": ["Set B", "Set B", "Set A", "Set A"],
                "target": ["Gene3", "Gene4", "Gene1", "Gene2"],
            }
        ),
    )


@pytest.mark.parametrize(
    "contents, match",
    [
        ("set\tdescription\n", "at least 3"),
        ("\tdescription\tgene\n", "source is empty"),
        ("set\tdescription\t\n", "no targets"),
        ("\n", "no gene sets"),
    ],
)
def test_read_gmt_rejects_malformed_files(tmp_path, contents, match):
    path = tmp_path / "bad.gmt"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        read_gmt(path)


def test_prepare_network_matches_prunes_and_is_canonical():
    original = pd.DataFrame(
        {
            "source": ["Beta", "Alpha", "Alpha", "Beta", "Dropped"],
            "target": ["geneD", "geneA", "GENEC", "GENEB", "missing"],
            "weight": [4.0, -1.0, 2.0, 3.0, 8.0],
        }
    )
    original.index = ["edge-a", "edge-b", "edge-c", "edge-d", "edge-e"]
    unchanged = original.copy(deep=True)
    kwargs = {
        "active_feature_names": np.array(["GeneA", "GeneB", "GeneC", "GeneD"]),
        "active_feature_index": np.array([8, 2, 5, 9]),
        "tmin": 2,
        "weighted": True,
    }

    prepared = prepare_network(original, **kwargs)
    shuffled = prepare_network(
        original.sample(frac=1.0, random_state=7).reset_index(drop=True),
        **kwargs,
    )

    pd.testing.assert_frame_equal(original, unchanged)
    np.testing.assert_array_equal(prepared.source_names, ["Alpha", "Beta"])
    np.testing.assert_array_equal(prepared.source_sizes, [2, 2])
    np.testing.assert_array_equal(prepared.matched_feature_index, [2, 5, 8, 9])
    for field in (
        "source_names",
        "source_sizes",
        "matched_feature_index",
        "edge_source_index",
        "edge_feature_index",
        "edge_weight",
    ):
        np.testing.assert_array_equal(
            getattr(prepared, field),
            getattr(shuffled, field),
        )
    assert prepared.network_digest == shuffled.network_digest


def test_prepare_network_detects_duplicate_and_ambiguous_matches():
    features = np.array(["GeneA", "GeneB"])
    indices = np.array([0, 1])
    duplicate = pd.DataFrame({"source": ["Set", "Set"], "target": ["GeneA", "GeneA"]})
    collapsed_duplicate = pd.DataFrame(
        {"source": ["Set", "Set"], "target": ["GeneA", "GENEA"]}
    )

    with pytest.raises(ValueError, match="duplicate source-target"):
        prepare_network(
            duplicate,
            active_feature_names=features,
            active_feature_index=indices,
            tmin=1,
            weighted=False,
        )
    with pytest.raises(ValueError, match="after case-insensitive"):
        prepare_network(
            collapsed_duplicate,
            active_feature_names=features,
            active_feature_index=indices,
            tmin=1,
            weighted=False,
        )
    with pytest.raises(ValueError, match="multiple active assay features"):
        prepare_network(
            pd.DataFrame({"source": ["Set"], "target": ["genea"]}),
            active_feature_names=np.array(["GeneA", "GENEA"]),
            active_feature_index=indices,
            tmin=1,
            weighted=False,
        )


def test_network_digest_only_uses_score_affecting_weights():
    first = pd.DataFrame(
        {
            "source": ["Set", "Set"],
            "target": ["GeneA", "GeneB"],
            "weight": [1.0, 2.0],
        }
    )
    second = first.assign(weight=[3.0, 4.0])
    kwargs = {
        "active_feature_names": np.array(["GeneA", "GeneB"]),
        "active_feature_index": np.array([0, 1]),
        "tmin": 2,
    }

    weighted_first = prepare_network(first, weighted=True, **kwargs)
    weighted_second = prepare_network(second, weighted=True, **kwargs)
    unweighted_first = prepare_network(first, weighted=False, **kwargs)
    unweighted_second = prepare_network(second, weighted=False, **kwargs)

    assert weighted_first.network_digest != weighted_second.network_digest
    assert unweighted_first.network_digest == unweighted_second.network_digest
    np.testing.assert_array_equal(unweighted_first.edge_weight, [1.0, 1.0])


def test_weighted_network_drops_zero_edges_before_tmin():
    net = pd.DataFrame(
        {
            "source": ["Set", "Set"],
            "target": ["GeneA", "GeneB"],
            "weight": [1.0, 0.0],
        }
    )

    with pytest.raises(ValueError, match="tmin=2"):
        prepare_network(
            net,
            active_feature_names=np.array(["GeneA", "GeneB"]),
            active_feature_index=np.array([0, 1]),
            tmin=2,
            weighted=True,
        )


def test_network_rejects_boolean_weights():
    net = pd.DataFrame(
        {
            "source": ["Set", "Set"],
            "target": ["GeneA", "GeneB"],
            "weight": [True, False],
        }
    )

    with pytest.raises(ValueError, match="not boolean"):
        prepare_network(
            net,
            active_feature_names=np.array(["GeneA", "GeneB"]),
            active_feature_index=np.array([0, 1]),
            tmin=1,
            weighted=True,
        )

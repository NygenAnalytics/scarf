import numpy as np
import pytest

from scarf.features.statistical import StatisticalTestResult
from scarf.plotting import StudyDesign


def _insert_group_columns(ds):
    n = len(ds.cells.active_index("I"))
    groups2 = np.array([f"g{i % 2}" for i in range(n)], dtype=object)
    groups3 = np.array([f"g{i % 3}" for i in range(n)], dtype=object)
    samples = np.array([f"s{i % 6}" for i in range(n)], dtype=object)
    subjects = np.array([f"d{i % 3}" for i in range(n)], dtype=object)
    ds.cells.insert("stat_group2", groups2, overwrite=True)
    ds.cells.insert("stat_group3", groups3, overwrite=True)
    ds.cells.insert("stat_sample", samples, overwrite=True)
    ds.cells.insert("stat_subject", subjects, overwrite=True)
    return groups2, groups3, samples, subjects


def test_run_statistical_testing_mann_whitney(datastore_ephemeral):
    ds = datastore_ephemeral
    _insert_group_columns(ds)
    result = ds.run_statistical_testing(
        ["MALAT1", "B2M"],
        group_by="stat_group2",
    )
    assert isinstance(result, StatisticalTestResult)
    assert result.method == "mann_whitney"
    assert result.posthoc is None
    assert result.adjustment_method == "fdr_bh"
    assert result.n_groups == 2
    assert result.cell_key == "I"
    assert set(result.tables) == {"MALAT1", "B2M"}
    for table in result.tables.values():
        assert {
            "group_1",
            "group_2",
            "n_1",
            "n_2",
            "u_statistic",
            "mean_1",
            "mean_2",
            "mean_difference",
            "p_value",
            "p_value_adjusted",
        } <= set(table.columns)
        assert table["p_value"].between(0, 1).all()
        assert table["p_value_adjusted"].between(0, 1).all()
        assert table["n_1"].iloc[0] + table["n_2"].iloc[0] == result.n_cells
    assert "statistical_tests" in ds.zw["RNA"]


def test_get_statistical_tests_round_trip(datastore_ephemeral):
    ds = datastore_ephemeral
    _insert_group_columns(ds)
    result = ds.run_statistical_testing(
        ["MALAT1", "B2M"],
        group_by="stat_group2",
    )
    loaded = ds.get_statistical_tests(
        group_key="stat_group2",
        method="mann_whitney",
    )
    assert isinstance(loaded, StatisticalTestResult)
    assert loaded.method == result.method
    assert loaded.adjustment_method == result.adjustment_method
    assert loaded.n_cells == result.n_cells
    assert loaded.cell_key == "I"
    assert set(loaded.tables) == set(result.tables)
    for key in result.tables:
        assert loaded.tables[key].to_dict("records") == result.tables[key].to_dict(
            "records"
        )


def test_run_statistical_testing_reuses_artifact(datastore_ephemeral):
    ds = datastore_ephemeral
    _insert_group_columns(ds)
    ds.run_statistical_testing(["MALAT1"], group_by="stat_group2")
    stats_group = ds.zw["RNA"]["statistical_tests"]
    index_before = dict(stats_group.attrs["artifacts"])
    ds.run_statistical_testing(["MALAT1"], group_by="stat_group2")
    index_after = dict(ds.zw["RNA"]["statistical_tests"].attrs["artifacts"])
    assert index_after == index_before


def test_run_statistical_testing_groups_restriction(datastore_ephemeral):
    ds = datastore_ephemeral
    _insert_group_columns(ds)
    result = ds.run_statistical_testing(
        ["MALAT1"],
        group_by="stat_group3",
        groups=["g0", "g2"],
        test="mann_whitney",
    )
    assert result.method == "mann_whitney"
    assert result.n_groups == 2
    row = result.tables["MALAT1"].iloc[0]
    assert row["group_1"] == "g0"
    assert row["group_2"] == "g2"


def test_run_statistical_testing_kruskal_dunn(datastore_ephemeral):
    ds = datastore_ephemeral
    _insert_group_columns(ds)
    result = ds.run_statistical_testing(
        ["MALAT1"],
        group_by="stat_group3",
        test="kruskal_wallis",
        posthoc="dunn",
        comparisons=[("g0", "g2")],
    )
    assert result.method == "kruskal_wallis"
    assert result.posthoc == "dunn"
    table = result.tables["MALAT1"]
    assert list(table["group_1"]) == ["g0"]
    assert list(table["group_2"]) == ["g2"]
    assert {"z", "p_value", "p_value_adjusted"} <= set(table.columns)
    assert table["p_value"].between(0, 1).all()
    loaded = ds.get_statistical_tests(
        group_key="stat_group3",
        method="kruskal_wallis",
        posthoc="dunn",
    )
    assert loaded.tables["MALAT1"].to_dict("records") == table.to_dict("records")


def test_run_statistical_testing_wilcoxon_paired(datastore_ephemeral):
    ds = datastore_ephemeral
    _insert_group_columns(ds)
    result = ds.run_statistical_testing(
        ["MALAT1"],
        group_by="stat_group2",
        test="wilcoxon",
        study_design=StudyDesign(
            sample_by="stat_sample",
            subject_by="stat_subject",
        ),
    )
    assert result.method == "wilcoxon"
    assert result.sample_by == "stat_sample"
    assert result.pair_by == "stat_subject"
    table = result.tables["MALAT1"]
    assert {"group_1", "group_2", "n_pairs", "statistic", "p_value"} <= set(
        table.columns
    )
    assert table["n_pairs"].iloc[0] >= 2
    loaded = ds.get_statistical_tests(
        group_key="stat_group2",
        method="wilcoxon",
    )
    assert loaded.tables["MALAT1"].to_dict("records") == table.to_dict("records")


def test_run_statistical_testing_cell_field_key(datastore_ephemeral):
    ds = datastore_ephemeral
    _insert_group_columns(ds)
    result = ds.run_statistical_testing(
        "RNA_nCounts",
        group_by="stat_group2",
    )
    assert result.method == "mann_whitney"
    assert "RNA_nCounts" in result.tables
    assert result.tables["RNA_nCounts"]["p_value"].notna().all()


def test_run_statistical_testing_normalization_option(datastore_ephemeral):
    ds = datastore_ephemeral
    _insert_group_columns(ds)
    from scarf.plotting import NormalizationSpec

    result = ds.run_statistical_testing(
        ["MALAT1"],
        group_by="stat_group2",
        normalization=NormalizationSpec(source="assay", transform="log1p"),
    )
    assert result.tables["MALAT1"]["p_value"].notna().all()
    assert result.tables["MALAT1"]["mean_1"].iloc[0] >= 0


def test_run_statistical_testing_skip_save(datastore_ephemeral):
    ds = datastore_ephemeral
    _insert_group_columns(ds)
    result = ds.run_statistical_testing(
        ["MALAT1"],
        group_by="stat_group2",
        skip_save=True,
    )
    assert result.method == "mann_whitney"
    assert "statistical_tests" not in ds.zw["RNA"]


def test_run_statistical_testing_errors(datastore_ephemeral):
    ds = datastore_ephemeral
    _insert_group_columns(ds)
    with pytest.raises(ValueError, match="group_by"):
        ds.run_statistical_testing(["MALAT1"], group_by=None)
    with pytest.raises(ValueError, match="requires sample aggregation"):
        ds.run_statistical_testing(
            ["MALAT1"],
            group_by="stat_group2",
            test="wilcoxon",
        )
    with pytest.raises(ValueError, match="at least three groups"):
        ds.run_statistical_testing(
            ["MALAT1"],
            group_by="stat_group2",
            test="kruskal_wallis",
        )
    with pytest.raises(ValueError, match="requires test"):
        ds.run_statistical_testing(
            ["MALAT1"],
            group_by="stat_group2",
            test="mann_whitney",
            posthoc="dunn",
        )
    with pytest.raises(ValueError, match="adjustment must be"):
        ds.run_statistical_testing(
            ["MALAT1"],
            group_by="stat_group2",
            adjustment="bogus",
        )
    with pytest.raises(KeyError, match="not found"):
        ds.run_statistical_testing(
            ["NOT_A_REAL_GENE_123"],
            group_by="stat_group2",
        )


def test_get_statistical_tests_errors(datastore_ephemeral):
    ds = datastore_ephemeral
    _insert_group_columns(ds)
    with pytest.raises(ValueError, match="group_key"):
        ds.get_statistical_tests(group_key=None)
    with pytest.raises(ValueError, match="test method"):
        ds.get_statistical_tests(group_key="stat_group2", method=None)
    with pytest.raises(KeyError, match="Couldn't find"):
        ds.get_statistical_tests(
            group_key="stat_group2",
            method="wilcoxon",
        )

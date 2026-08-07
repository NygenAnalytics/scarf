import numpy as np
import pandas as pd
import pytest

from scarf.features.statistical import StatisticalTestResult
from scarf.plotting import StudyDesign

pytestmark = pytest.mark.filterwarnings("ignore:Cell-level statistical testing")


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
    assert len(result.tested_features) == 2
    assert all(
        isinstance(identity, str) and identity for identity in result.tested_features
    )
    assert set(result.tables) == {"MALAT1", "B2M"}
    assert result.posthoc_tables == {}
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
        keys=["MALAT1", "B2M"],
    )
    assert isinstance(loaded, StatisticalTestResult)
    assert loaded.method == result.method
    assert loaded.adjustment_method == result.adjustment_method
    assert loaded.n_cells == result.n_cells
    assert loaded.cell_key == "I"
    assert loaded.tested_features == result.tested_features
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


def test_reuse_returns_persisted_without_recompute(
    datastore_ephemeral,
    monkeypatch,
):
    ds = datastore_ephemeral
    _insert_group_columns(ds)
    features_module = pytest.importorskip("scarf.datastore._operations.features")
    calls = {"n": 0}
    original = features_module.compare_group_distributions

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(features_module, "compare_group_distributions", counting)
    ds.run_statistical_testing(["MALAT1"], group_by="stat_group2")
    first = calls["n"]
    assert first >= 1
    ds.run_statistical_testing(["MALAT1"], group_by="stat_group2")
    assert calls["n"] == first


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
    loaded = ds.get_statistical_tests(
        group_key="stat_group3",
        method="mann_whitney",
        keys=["MALAT1"],
        groups=["g0", "g2"],
    )
    assert loaded.tables["MALAT1"].to_dict("records") == result.tables[
        "MALAT1"
    ].to_dict("records")


def test_groups_control_contrast_direction(datastore_ephemeral):
    ds = datastore_ephemeral
    _insert_group_columns(ds)
    forward = ds.run_statistical_testing(
        ["MALAT1"],
        group_by="stat_group3",
        groups=["g2", "g0"],
        test="mann_whitney",
    )
    backward = ds.run_statistical_testing(
        ["MALAT1"],
        group_by="stat_group3",
        groups=["g0", "g2"],
        test="mann_whitney",
    )
    assert forward.tables["MALAT1"]["group_1"].iloc[0] == "g2"
    assert backward.tables["MALAT1"]["group_1"].iloc[0] == "g0"
    assert np.isclose(
        forward.tables["MALAT1"]["mean_difference"].iloc[0],
        -backward.tables["MALAT1"]["mean_difference"].iloc[0],
    )


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
    omnibus = result.tables["MALAT1"]
    assert {"kruskal_statistic", "df", "p_value", "p_value_adjusted"} <= set(
        omnibus.columns
    )
    posthoc = result.posthoc_tables["MALAT1"]
    assert list(posthoc["group_1"]) == ["g0"]
    assert list(posthoc["group_2"]) == ["g2"]
    assert {"z", "p_value", "p_value_adjusted"} <= set(posthoc.columns)
    assert posthoc["p_value"].between(0, 1).all()
    loaded = ds.get_statistical_tests(
        group_key="stat_group3",
        method="kruskal_wallis",
        posthoc="dunn",
        keys=["MALAT1"],
        comparisons=[("g0", "g2")],
    )
    assert loaded.tables["MALAT1"].to_dict("records") == omnibus.to_dict("records")
    assert loaded.posthoc_tables["MALAT1"].to_dict("records") == posthoc.to_dict(
        "records"
    )


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
        keys=["MALAT1"],
        sample_by="stat_sample",
        pair_by="stat_subject",
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
    loaded = ds.get_statistical_tests(
        group_key="stat_group2",
        method="mann_whitney",
        keys="RNA_nCounts",
    )
    assert loaded.tables["RNA_nCounts"].to_dict("records") == result.tables[
        "RNA_nCounts"
    ].to_dict("records")


def test_summary_scope_is_persisted(datastore_ephemeral):
    ds = datastore_ephemeral
    _insert_group_columns(ds)
    cell_result = ds.run_statistical_testing(
        ["MALAT1"],
        group_by="stat_group2",
    )
    assert cell_result.summary_scope == "cell"
    sample_result = ds.run_statistical_testing(
        ["MALAT1"],
        group_by="stat_group2",
        sample_by="stat_sample",
    )
    assert sample_result.summary_scope == "sample"
    loaded_cell = ds.get_statistical_tests(
        group_key="stat_group2",
        method="mann_whitney",
        keys=["MALAT1"],
    )
    assert loaded_cell.summary_scope == "cell"
    loaded_sample = ds.get_statistical_tests(
        group_key="stat_group2",
        method="mann_whitney",
        keys=["MALAT1"],
        sample_by="stat_sample",
    )
    assert loaded_sample.summary_scope == "sample"


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


def test_variant_slots_are_distinct_and_retrievable(datastore_ephemeral):
    ds = datastore_ephemeral
    _insert_group_columns(ds)
    ds.run_statistical_testing(["MALAT1"], group_by="stat_group2")
    ds.run_statistical_testing(["MALAT1", "B2M"], group_by="stat_group2")
    index = ds.zw["RNA"]["statistical_tests"].attrs["artifacts"]
    assert len(index) == 2
    single = ds.get_statistical_tests(
        group_key="stat_group2",
        method="mann_whitney",
        keys=["MALAT1"],
    )
    double = ds.get_statistical_tests(
        group_key="stat_group2",
        method="mann_whitney",
        keys=["MALAT1", "B2M"],
    )
    assert set(single.tables) == {"MALAT1"}
    assert set(double.tables) == {"MALAT1", "B2M"}


def test_subset_by_change_recomputes(datastore_ephemeral):
    ds = datastore_ephemeral
    _insert_group_columns(ds)
    n = len(ds.cells.active_index("I"))
    subset = np.ones(n, dtype=bool)
    subset[::3] = False
    ds.cells.insert("stat_subset", subset, overwrite=True)
    ds.run_statistical_testing(
        ["MALAT1"],
        group_by="stat_group2",
        subset_by="stat_subset",
    )
    stats_group = ds.zw["RNA"]["statistical_tests"]
    slot = next(iter(stats_group.attrs["artifacts"]))
    first_id = stats_group.attrs["artifacts"][slot]["artifact_id"]
    ds.cells.insert("stat_subset", ~subset, overwrite=True)
    ds.run_statistical_testing(
        ["MALAT1"],
        group_by="stat_group2",
        subset_by="stat_subset",
    )
    second_id = ds.zw["RNA"]["statistical_tests"].attrs["artifacts"][slot][
        "artifact_id"
    ]
    assert first_id != second_id


def test_cell_key_none_uses_all_cells(datastore_ephemeral):
    ds = datastore_ephemeral
    _insert_group_columns(ds)
    total = ds.cells.N
    result = ds.run_statistical_testing(
        ["MALAT1"],
        group_by="stat_group2",
        cell_key=None,
    )
    assert result.cell_key is None
    assert result.n_cells == total
    loaded = ds.get_statistical_tests(
        cell_key=None,
        group_key="stat_group2",
        method="mann_whitney",
        keys=["MALAT1"],
    )
    assert loaded.cell_key is None
    assert loaded.n_cells == total


def test_int_and_bool_group_labels_roundtrip(datastore_ephemeral):
    ds = datastore_ephemeral
    n = len(ds.cells.active_index("I"))
    int_groups = np.array([i % 2 for i in range(n)], dtype=np.int64)
    bool_groups = np.array([i % 2 == 0 for i in range(n)], dtype=bool)
    ds.cells.insert("stat_int_group", int_groups, overwrite=True)
    ds.cells.insert("stat_bool_group", bool_groups, overwrite=True)
    ds.run_statistical_testing(["MALAT1"], group_by="stat_int_group")
    ds.run_statistical_testing(["MALAT1"], group_by="stat_bool_group")
    loaded_int = ds.get_statistical_tests(
        group_key="stat_int_group",
        method="mann_whitney",
        keys=["MALAT1"],
    )
    loaded_bool = ds.get_statistical_tests(
        group_key="stat_bool_group",
        method="mann_whitney",
        keys=["MALAT1"],
    )
    int_group = loaded_int.tables["MALAT1"]["group_1"]
    bool_group = loaded_bool.tables["MALAT1"]["group_1"]
    assert pd.api.types.is_integer_dtype(int_group)
    assert pd.api.types.is_bool_dtype(bool_group)
    assert int_group.iloc[0] in (0, 1)
    assert bool_group.iloc[0] in (True, False)


def test_auto_counts_surviving_groups_after_dropout(datastore_ephemeral):
    ds = datastore_ephemeral
    n = len(ds.cells.active_index("I"))
    groups3 = np.array([f"g{i % 3}" for i in range(n)], dtype=object)
    samples = np.array([f"s{i % 6}" for i in range(n)], dtype=object)
    samples = np.where(groups3 == "g2", "", samples).astype(object)
    ds.cells.insert("stat_grp_drop", groups3, overwrite=True)
    ds.cells.insert("stat_samp_drop", samples, overwrite=True)
    result = ds.run_statistical_testing(
        ["MALAT1"],
        group_by="stat_grp_drop",
        sample_by="stat_samp_drop",
    )
    assert result.method == "mann_whitney"
    assert result.n_groups == 2
    assert set(result.tables["MALAT1"]["group_1"]) <= {"g0", "g1"}


def test_explicit_test_rejects_filtered_requested_group(datastore_ephemeral):
    ds = datastore_ephemeral
    n = len(ds.cells.active_index("I"))
    groups3 = np.array([f"g{i % 3}" for i in range(n)], dtype=object)
    samples = np.array([f"s{i % 6}" for i in range(n)], dtype=object)
    samples = np.where(groups3 == "g2", "", samples).astype(object)
    ds.cells.insert("stat_grp_drop", groups3, overwrite=True)
    ds.cells.insert("stat_samp_drop", samples, overwrite=True)
    with pytest.warns(UserWarning, match="removed because all of its cells"):
        with pytest.raises(ValueError, match="must all retain at least one valid cell"):
            ds.run_statistical_testing(
                ["MALAT1"],
                group_by="stat_grp_drop",
                groups=["g0", "g1", "g2"],
                test="kruskal_wallis",
                sample_by="stat_samp_drop",
            )


def test_run_statistical_testing_errors(datastore_ephemeral):
    ds = datastore_ephemeral
    _insert_group_columns(ds)
    with pytest.raises(ValueError, match="group_by"):
        ds.run_statistical_testing(["MALAT1"], group_by=None)
    with pytest.raises(NotImplementedError, match="non-parametric"):
        ds.run_statistical_testing(["MALAT1"], group_by="stat_group2", test="anova")
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
        ds.get_statistical_tests(group_key=None, keys=["MALAT1"])
    with pytest.raises(ValueError, match="test method"):
        ds.get_statistical_tests(
            group_key="stat_group2",
            method=None,
            keys=["MALAT1"],
        )
    with pytest.raises(KeyError, match="Couldn't find"):
        ds.get_statistical_tests(
            group_key="stat_group2",
            method="wilcoxon",
            keys=["MALAT1"],
        )

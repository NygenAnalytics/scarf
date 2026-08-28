from typing import get_type_hints

import numpy as np
import pandas as pd
import pytest

from scarf import DataStore
from scarf.features.statistical import (
    ANOVA_COLUMNS,
    StatisticalTestResult,
    WELCH_COLUMNS,
)
from scarf.plotting import CellField, FeatureRef, NormalizationSpec, StudyDesign

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
    assert result.artifact is not None
    pinned = ds.get_statistical_tests(artifact=result.artifact)
    assert isinstance(loaded, StatisticalTestResult)
    assert loaded.method == result.method
    assert loaded.adjustment_method == result.adjustment_method
    assert loaded.n_cells == result.n_cells
    assert loaded.cell_key == "I"
    assert loaded.tested_features == result.tested_features
    assert loaded.artifact == result.artifact
    assert loaded.cell_selection == result.cell_selection
    assert loaded.cell_selection_fingerprint == result.cell_selection_fingerprint
    assert loaded.group_fingerprint == result.group_fingerprint
    assert loaded.group_order == result.group_order
    assert loaded.normalization == result.normalization
    assert loaded.source_assays == result.source_assays == ("RNA", "RNA")
    assert loaded.source_dataset_fingerprint == result.source_dataset_fingerprint
    assert loaded.source_dataset_fingerprint
    assert loaded.value_fingerprints == result.value_fingerprints
    assert len(loaded.value_fingerprints) == 2
    assert loaded.normalization_method == result.normalization_method
    assert loaded.size_factor == result.size_factor
    assert pinned.artifact == result.artifact
    assert pinned.tables.keys() == result.tables.keys()
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


def test_statistical_value_identity_rejects_stale_raw_reuse(datastore_ephemeral):
    ds = datastore_ephemeral
    _insert_group_columns(ds)
    key = FeatureRef(value=0, by="index", label="raw_feature")
    normalization = NormalizationSpec(source="raw", transform="none")
    first = ds.run_statistical_testing(
        key,
        group_by="stat_group2",
        normalization=normalization,
    )
    assert first.artifact is not None
    assert len(first.value_fingerprints) == 1

    active = np.asarray(ds.cells.active_index("I"), dtype=np.int64)
    backing = ds.RNA.rawData._backing
    original = np.asarray(backing[int(active[0]), 0]).item()
    backing[int(active[0]), 0] = original + 1

    with pytest.raises(ValueError, match="stale for the current data"):
        ds.get_statistical_tests(
            group_key="stat_group2",
            method="mann_whitney",
            keys=key,
            normalization=normalization,
        )

    historical = ds.get_statistical_tests(artifact=first.artifact)
    assert historical.artifact == first.artifact
    assert historical.value_fingerprints == first.value_fingerprints

    second = ds.run_statistical_testing(
        key,
        group_by="stat_group2",
        normalization=normalization,
    )
    assert second.artifact != first.artifact
    assert second.value_fingerprints != first.value_fingerprints


def test_reuse_returns_persisted_without_recompute(
    datastore_ephemeral,
    monkeypatch,
):
    ds = datastore_ephemeral
    _insert_group_columns(ds)
    features_module = pytest.importorskip("scarf.datastore._operations.features")
    calls = {"n": 0}
    original = features_module.compare_group_distributions
    original_fetch = ds.RNA.normed
    fetch_calls = {"n": 0}

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    def counting_fetch(*args, **kwargs):
        fetch_calls["n"] += 1
        return original_fetch(*args, **kwargs)

    monkeypatch.setattr(features_module, "compare_group_distributions", counting)
    monkeypatch.setattr(ds.RNA, "normed", counting_fetch)
    ds.run_statistical_testing(["MALAT1"], group_by="stat_group2")
    first = calls["n"]
    first_fetch = fetch_calls["n"]
    assert first >= 1
    assert first_fetch == 1
    ds.run_statistical_testing(["MALAT1"], group_by="stat_group2")
    assert calls["n"] == first
    assert fetch_calls["n"] == first_fetch + 1


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


def test_statistical_testing_accepts_numpy_group_and_comparison_sequences(
    datastore_ephemeral,
):
    ds = datastore_ephemeral
    _insert_group_columns(ds)
    groups = np.array(["g2", "g0"], dtype=object)
    comparisons = np.array([["g2", "g0"]], dtype=object)

    result = ds.run_statistical_testing(
        ["MALAT1"],
        group_by="stat_group3",
        groups=groups,
        comparisons=comparisons,
        test="mann_whitney",
    )
    loaded = ds.get_statistical_tests(
        group_key="stat_group3",
        method="mann_whitney",
        keys=["MALAT1"],
        groups=groups,
        comparisons=comparisons,
    )

    assert result.tables["MALAT1"].loc[0, "group_1"] == "g2"
    assert loaded.artifact == result.artifact


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
    assert loaded.sample_fingerprint == result.sample_fingerprint
    assert loaded.pair_fingerprint == result.pair_fingerprint


def test_run_statistical_testing_rejects_missing_pair_in_selection(
    datastore_ephemeral,
):
    ds = datastore_ephemeral
    _groups, _groups3, _samples, subjects = _insert_group_columns(ds)
    subjects = subjects.copy()
    subjects[0] = ""
    ds.cells.insert("stat_subject_missing", subjects, overwrite=True)

    with pytest.raises(ValueError, match="valid pair value for every cell"):
        ds.run_statistical_testing(
            ["MALAT1"],
            group_by="stat_group2",
            test="wilcoxon",
            sample_by="stat_sample",
            pair_by="stat_subject_missing",
        )


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
    assert result.artifact is not None
    assert result.artifact.scope == "datastore"
    assert result.source_assays == (None,)
    assert result.normalization == {}
    assert "statistical_tests" in ds.zw
    loaded = ds.get_statistical_tests(
        group_key="stat_group2",
        method="mann_whitney",
        keys="RNA_nCounts",
    )
    assert loaded.tables["RNA_nCounts"].to_dict("records") == result.tables[
        "RNA_nCounts"
    ].to_dict("records")


def test_statistical_key_labels_define_distinct_slots(datastore_ephemeral):
    ds = datastore_ephemeral
    _insert_group_columns(ds)
    first_key = CellField("RNA_nCounts", label="first")
    second_key = CellField("RNA_nCounts", label="second")

    first = ds.run_statistical_testing(first_key, group_by="stat_group2")
    second = ds.run_statistical_testing(second_key, group_by="stat_group2")
    loaded_first = ds.get_statistical_tests(
        group_key="stat_group2",
        method="mann_whitney",
        keys=first_key,
    )
    loaded_second = ds.get_statistical_tests(
        group_key="stat_group2",
        method="mann_whitney",
        keys=second_key,
    )

    assert first.artifact != second.artifact
    assert set(first.tables) == set(loaded_first.tables) == {"first"}
    assert set(second.tables) == set(loaded_second.tables) == {"second"}
    assert len(ds.zw["statistical_tests"].attrs["artifacts"]) == 2


def test_statistical_tested_metadata_rejects_effective_missing_mask(
    datastore_ephemeral,
):
    ds = datastore_ephemeral
    _insert_group_columns(ds)
    active = np.asarray(ds.cells.active_index("I"), dtype=np.int64)
    values = np.arange(len(active), dtype=np.float64)
    ds.cells.insert("stat_tested_masked", values, overwrite=True)
    missing = np.zeros(ds.cells.N, dtype=bool)
    missing[active[0]] = True
    cell_data = ds.cells.locations["primary"]
    missing_name = "__scarf_missing__stat_tested_masked"
    cell_data.create_array(
        missing_name,
        data=missing,
        chunks=(min(ds.cells.N, 100_000),),
        overwrite=True,
    )
    cell_data["stat_tested_masked"].attrs["missing_mask"] = missing_name

    with pytest.raises(ValueError, match="effective cell selection"):
        ds.run_statistical_testing(
            CellField("stat_tested_masked"),
            group_by="stat_group2",
        )

    subset = np.ones(len(active), dtype=bool)
    subset[0] = False
    ds.cells.insert("stat_excludes_tested_missing", subset, overwrite=True)
    result = ds.run_statistical_testing(
        CellField("stat_tested_masked"),
        group_by="stat_group2",
        subset_by="stat_excludes_tested_missing",
    )
    assert result.n_cells == len(active) - 1


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
    assert result.artifact is None
    assert result.cell_selection is None
    assert result.normalization_method is not None
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
    first_result = ds.run_statistical_testing(
        ["MALAT1"],
        group_by="stat_group2",
        subset_by="stat_subset",
    )
    stats_group = ds.zw["RNA"]["statistical_tests"]
    slot = next(iter(stats_group.attrs["artifacts"]))
    first_id = stats_group.attrs["artifacts"][slot]["artifact_id"]
    ds.cells.insert("stat_subset", ~subset, overwrite=True)
    second_result = ds.run_statistical_testing(
        ["MALAT1"],
        group_by="stat_group2",
        subset_by="stat_subset",
    )
    second_id = ds.zw["RNA"]["statistical_tests"].attrs["artifacts"][slot][
        "artifact_id"
    ]
    assert first_id != second_id
    assert first_result.artifact is not None
    assert second_result.artifact is not None
    pinned_first = ds.get_statistical_tests(artifact=first_result.artifact)
    assert pinned_first.artifact == first_result.artifact
    assert (
        pinned_first.cell_selection_fingerprint
        == first_result.cell_selection_fingerprint
        != second_result.cell_selection_fingerprint
    )


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
    with pytest.raises(ValueError, match="requires.*sample"):
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
    with pytest.raises(ValueError, match="alternative is only supported"):
        ds.run_statistical_testing(
            ["MALAT1"],
            group_by="stat_group2",
            alternative="greater",
        )
    with pytest.raises(ValueError, match="pair_by is only supported"):
        ds.run_statistical_testing(
            ["MALAT1"],
            group_by="stat_group2",
            pair_by="stat_subject",
            sample_by="stat_sample",
            test="mann_whitney",
        )
    with pytest.raises(ValueError, match="sample_stat requires sample_by"):
        ds.run_statistical_testing(
            ["MALAT1"],
            group_by="stat_group2",
            sample_stat="median",
        )
    with pytest.raises(ValueError, match="only used.*fraction"):
        ds.run_statistical_testing(
            ["MALAT1"],
            group_by="stat_group2",
            sample_by="stat_sample",
            expression_cutoff=1.0,
        )
    with pytest.raises(ValueError, match="comparisons requires a pairwise test"):
        ds.run_statistical_testing(
            ["MALAT1"],
            group_by="stat_group3",
            test="kruskal_wallis",
            comparisons=[("g0", "g1")],
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


def test_run_statistical_testing_welch_cell_level(datastore_ephemeral):
    ds = datastore_ephemeral
    _insert_group_columns(ds)
    result = ds.run_statistical_testing(
        ["MALAT1"],
        group_by="stat_group2",
        test="welch",
        alternative="less",
    )
    assert result.method == "welch"
    assert result.alternative == "less"
    assert result.equal_var is False
    table = result.tables["MALAT1"]
    assert tuple(table.columns) == (*WELCH_COLUMNS, "p_value_adjusted")
    assert table["p_value"].between(0, 1).all()
    assert bool((table["df"] > 0).all())
    loaded = ds.get_statistical_tests(
        group_key="stat_group2",
        method="welch",
        keys=["MALAT1"],
        alternative="less",
    )
    assert loaded.alternative == "less"
    assert loaded.tables["MALAT1"].to_dict("records") == table.to_dict("records")


def test_welch_groups_restriction_and_contrast_direction(datastore_ephemeral):
    ds = datastore_ephemeral
    _insert_group_columns(ds)
    forward = ds.run_statistical_testing(
        ["MALAT1"],
        group_by="stat_group3",
        groups=["g2", "g0"],
        test="welch",
    )
    assert forward.n_groups == 2
    row = forward.tables["MALAT1"].iloc[0]
    assert row["group_1"] == "g2"
    assert row["group_2"] == "g0"
    assert row["mean_difference"] == row["mean_1"] - row["mean_2"]
    with pytest.raises(ValueError, match="exactly two groups"):
        ds.run_statistical_testing(
            ["MALAT1"],
            group_by="stat_group3",
            test="welch",
        )


def test_one_way_anova_omnibus_roundtrip(datastore_ephemeral):
    ds = datastore_ephemeral
    _insert_group_columns(ds)
    result = ds.run_statistical_testing(
        ["MALAT1", "B2M"],
        group_by="stat_group3",
        test="one_way_anova",
    )
    assert result.method == "one_way_anova"
    assert result.n_groups == 3
    for key in ("MALAT1", "B2M"):
        table = result.tables[key]
        assert set(ANOVA_COLUMNS) <= set(table.columns)
        assert table.loc[0, "df_between"] == 2
        assert table["p_value"].between(0, 1).all()
    loaded = ds.get_statistical_tests(
        group_key="stat_group3",
        method="one_way_anova",
        keys=["MALAT1", "B2M"],
    )
    assert loaded.equal_var is None
    for key in result.tables:
        assert loaded.tables[key].to_dict("records") == result.tables[key].to_dict(
            "records"
        )


def test_welch_variant_slots_distinct_by_alternative(datastore_ephemeral):
    ds = datastore_ephemeral
    _insert_group_columns(ds)
    ds.run_statistical_testing(
        ["MALAT1"],
        group_by="stat_group2",
        test="welch",
        alternative="less",
    )
    artifacts_after_less = dict(ds.zw["RNA"]["statistical_tests"].attrs["artifacts"])
    ds.run_statistical_testing(
        ["MALAT1"],
        group_by="stat_group2",
        test="welch",
        alternative="greater",
    )
    artifacts_after_greater = dict(ds.zw["RNA"]["statistical_tests"].attrs["artifacts"])
    new_slots = set(artifacts_after_greater) - set(artifacts_after_less)
    assert len(new_slots) == 1
    distinct_ids = {
        artifacts_after_less[key]["artifact_id"] for key in artifacts_after_less
    } | {artifacts_after_greater[key]["artifact_id"] for key in new_slots}
    assert len(distinct_ids) >= 2


def test_welch_alternative_roundtrip_dtypes(datastore_ephemeral):
    ds = datastore_ephemeral
    _insert_group_columns(ds)
    ds.run_statistical_testing(
        ["MALAT1"],
        group_by="stat_group2",
        test="welch",
        alternative="greater",
    )
    loaded = ds.get_statistical_tests(
        group_key="stat_group2",
        method="welch",
        keys=["MALAT1"],
        alternative="greater",
    )
    table = loaded.tables["MALAT1"]
    assert table["group_1"].dtype.kind in {"O", "U"}
    assert table["group_2"].dtype.kind in {"O", "U"}
    assert table["t_statistic"].dtype.kind == "f"
    assert str(table["t_statistic"].dtype) == "float64"
    assert str(table["df"].dtype) == "float64"


def test_plotting_distribution_annotates_welch_brackets(datastore_ephemeral):
    ds = datastore_ephemeral
    _insert_group_columns(ds)
    baseline = ds.plots.distribution(["MALAT1"], group_by="stat_group2", show=False)
    baseline_lines = len(baseline.axes["MALAT1"].lines)
    result = ds.run_statistical_testing(
        ["MALAT1"],
        group_by="stat_group2",
        test="welch",
    )
    annotated = ds.plots.distribution(
        ["MALAT1"],
        group_by="stat_group2",
        stats_results=result,
        show=False,
    )
    texts = [text.get_text() for text in annotated.axes["MALAT1"].texts]
    assert any(text.startswith("p=") for text in texts)
    assert len(annotated.axes["MALAT1"].lines) > baseline_lines
    assert annotated.owns_figure == baseline.owns_figure
    assert annotated.provenance.extras.get("stats_annotated") is True


def test_statistical_index_distinguishes_selection_and_group_aliases(
    datastore_ephemeral,
):
    ds = datastore_ephemeral
    groups, *_ = _insert_group_columns(ds)
    ds.cells.insert("stat_group_alias", groups.copy(), overwrite=True)
    base_selection = np.asarray(ds.cells.fetch_all("I"), dtype=bool)
    ds.cells.insert("all_cells", base_selection, overwrite=True)
    ds.cells.insert("stat_selection_alias", base_selection, overwrite=True)

    group_a = ds.run_statistical_testing(["MALAT1"], group_by="stat_group2")
    group_b = ds.run_statistical_testing(["MALAT1"], group_by="stat_group_alias")
    select_a = ds.run_statistical_testing(
        ["MALAT1"],
        group_by="stat_group2",
        cell_key="all_cells",
    )
    select_b = ds.run_statistical_testing(
        ["MALAT1"],
        group_by="stat_group2",
        cell_key="stat_selection_alias",
    )
    all_cells = ds.run_statistical_testing(
        ["MALAT1"],
        group_by="stat_group2",
        cell_key=None,
    )

    refs = {
        result.artifact for result in (group_a, group_b, select_a, select_b, all_cells)
    }
    assert None not in refs
    assert len(refs) == 5
    index = ds.zw["RNA"]["statistical_tests"].attrs["artifacts"]
    assert len(index) == 5
    assert all(len(slot) == 64 and "__" not in slot for slot in index)
    assert group_b.group_key == "stat_group_alias"
    assert select_a.cell_key == "all_cells"
    assert all_cells.cell_key is None


def test_statistical_source_assay_scope_and_mixed_assay_policy(
    datastore_ephemeral,
):
    ds = datastore_ephemeral
    _insert_group_columns(ds)
    assay2 = ds.assay2.name
    ref = FeatureRef(value=0, assay=assay2, by="index")
    result = ds.run_statistical_testing(ref, group_by="stat_group2")
    assert result.artifact is not None
    assert result.artifact.scope == "assay"
    assert result.artifact.assay == assay2
    assert result.source_assays == (assay2,)
    assert "statistical_tests" in ds.zw[assay2]
    loaded = ds.get_statistical_tests(
        group_key="stat_group2",
        method="mann_whitney",
        keys=ref,
    )
    assert loaded.artifact == result.artifact
    with pytest.raises(ValueError, match="multiple assays"):
        ds.run_statistical_testing(
            ["MALAT1", ref],
            group_by="stat_group2",
        )


def test_statistical_effective_selection_identity_and_n_cells(
    datastore_ephemeral,
):
    ds = datastore_ephemeral
    n = len(ds.cells.active_index("I"))
    groups = np.array(["g0" if i % 2 == 0 else "g1" for i in range(n)], dtype=object)
    groups[::7] = None
    groups[1::11] = ""
    ds.cells.insert("stat_groups_missing", groups, overwrite=True)
    result = ds.run_statistical_testing(
        ["MALAT1"],
        group_by="stat_groups_missing",
    )
    expected = int(sum(value not in (None, "") for value in groups))
    assert result.n_cells == expected
    assert result.cell_selection_fingerprint
    assert result.group_fingerprint
    assert result.group_order == ("g0", "g1")
    row = result.tables["MALAT1"].iloc[0]
    assert int(row["n_1"] + row["n_2"]) == expected


def test_statistical_selection_honors_metadata_missing_mask(datastore_ephemeral):
    ds = datastore_ephemeral
    active = np.asarray(ds.cells.active_index("I"), dtype=np.int64)
    groups = np.arange(ds.cells.N, dtype=np.int64) % 2
    missing = np.zeros(ds.cells.N, dtype=bool)
    missing[active[::7]] = True
    ds.cells.insert("stat_groups_masked", groups, overwrite=True)
    cell_data = ds.cells.locations["primary"]
    missing_name = "__scarf_missing__stat_groups_masked"
    cell_data.create_array(
        missing_name,
        data=missing,
        chunks=(min(ds.cells.N, 100_000),),
        overwrite=True,
    )
    cell_data["stat_groups_masked"].attrs["missing_mask"] = missing_name

    result = ds.run_statistical_testing(
        ["MALAT1"],
        group_by="stat_groups_masked",
    )
    figure = ds.plots.distribution(
        ["MALAT1"],
        group_by="stat_groups_masked",
        stats_results=result,
        show=False,
    )

    assert result.n_cells == int((~missing[active]).sum())
    assert any(text.get_text().startswith("p=") for text in figure.axes["MALAT1"].texts)
    figure.close()


def test_statistical_numeric_dtypes_and_infinite_statistic_roundtrip(
    datastore_ephemeral,
):
    ds = datastore_ephemeral
    n = len(ds.cells.active_index("I"))
    groups = np.array(["a" if i < n // 2 else "b" for i in range(n)], dtype=object)
    values = np.array([0.0 if value == "a" else 1.0 for value in groups])
    ds.cells.insert("stat_constant_group", groups, overwrite=True)
    ds.cells.insert("stat_constant_value", values, overwrite=True)
    result = ds.run_statistical_testing(
        "stat_constant_value",
        group_by="stat_constant_group",
        test="welch",
    )
    loaded = ds.get_statistical_tests(
        group_key="stat_constant_group",
        method="welch",
        keys="stat_constant_value",
    )
    original = result.tables["stat_constant_value"]
    restored = loaded.tables["stat_constant_value"]
    assert np.isinf(restored["t_statistic"]).all()
    assert restored["p_value"].eq(0.0).all()
    assert restored.dtypes.to_dict() == original.dtypes.to_dict()


def test_statistical_public_annotations_resolve_at_runtime():
    assert get_type_hints(DataStore.run_statistical_testing)
    assert get_type_hints(DataStore.get_statistical_tests)


def test_statistical_slot_key_is_structured_and_unambiguous():
    from scarf.datastore._operations.features import _statistical_slot_key

    variants = {
        _statistical_slot_key(None, "group__name", "mann_whitney", None, "digest"),
        _statistical_slot_key(
            "all_cells", "group__name", "mann_whitney", None, "digest"
        ),
        _statistical_slot_key(None, "group", "name__mann_whitney", None, "digest"),
        _statistical_slot_key(None, "group__name", "mann_whitney", "dunn", "digest"),
    }
    assert len(variants) == 4
    assert all(len(slot) == 64 and "__" not in slot for slot in variants)

"""
tests/test_profiler.py
======================
DataInvest AI — Phase 1 Unit Tests

Tests cover:
- Row and column counts
- Missing-value detection (count and percentage)
- Duplicate-row detection
- Numerical summary statistics (mean, median, std, min, max, Q1, Q3, IQR, skewness)
- IQR-based potential outlier detection
- Constant-column detection
- High-cardinality column detection
- Datetime column detection
- Empty / invalid DataFrame handling
- Edge cases: all-NaN column, single-row dataset, too-few-obs column
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

# Adjust import path for direct pytest invocation from the project root.
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.profiler import (
    profile_dataset,
    _get_overview,
    _get_missing_stats,
    _get_duplicate_stats,
    _get_numerical_stats,
    _get_outlier_stats,
    _get_cardinality,
    _get_column_types,
    _detect_datetime_columns,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def simple_df() -> pd.DataFrame:
    """Basic DataFrame with known properties."""
    return pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45],
            "salary": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
            "name": ["Alice", "Bob", "Carol", "David", "Eve"],
            "department": ["Eng", "Eng", "HR", "Eng", "HR"],
        }
    )


@pytest.fixture()
def df_with_missing() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, None, 4.0, 5.0],
            "b": [None, None, None, None, None],
            "c": ["x", "y", "z", None, "w"],
        }
    )


@pytest.fixture()
def df_with_duplicates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [1, 2, 2, 3, 3],
            "y": ["a", "b", "b", "c", "c"],
        }
    )


@pytest.fixture()
def df_with_outliers() -> pd.DataFrame:
    """Normal values with two obvious outliers."""
    values = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 1000, -500]
    return pd.DataFrame({"val": values})


@pytest.fixture()
def df_constant() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "const_num": [5, 5, 5, 5],
            "const_str": ["X", "X", "X", "X"],
            "normal": [1, 2, 3, 4],
        }
    )


@pytest.fixture()
def df_datetime() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date_str": ["2021-01-01", "2021-06-15", "2022-03-10", "2023-07-22"],
            "not_date": ["apple", "banana", "cherry", "date"],
            "value": [10, 20, 30, 40],
        }
    )


@pytest.fixture()
def df_high_cardinality() -> pd.DataFrame:
    """A categorical column with nearly all unique values."""
    return pd.DataFrame(
        {
            "id_col": [f"user_{i}" for i in range(100)],
            "category": ["A", "B", "A", "C"] * 25,
        }
    )


# ──────────────────────────────────────────────────────────────────────────────
# Row and column count tests
# ──────────────────────────────────────────────────────────────────────────────


class TestOverview:
    def test_row_count(self, simple_df):
        ov = _get_overview(simple_df)
        assert ov["n_rows"] == 5

    def test_column_count(self, simple_df):
        ov = _get_overview(simple_df)
        assert ov["n_cols"] == 4

    def test_column_names(self, simple_df):
        ov = _get_overview(simple_df)
        assert ov["column_names"] == ["age", "salary", "name", "department"]

    def test_dtypes_present(self, simple_df):
        ov = _get_overview(simple_df)
        assert set(ov["dtypes"].keys()) == {"age", "salary", "name", "department"}

    def test_memory_usage_positive(self, simple_df):
        ov = _get_overview(simple_df)
        assert ov["memory_usage_bytes"] > 0

    def test_single_row(self):
        df = pd.DataFrame({"a": [1], "b": ["x"]})
        ov = _get_overview(df)
        assert ov["n_rows"] == 1
        assert ov["n_cols"] == 2


# ──────────────────────────────────────────────────────────────────────────────
# Missing-value tests
# ──────────────────────────────────────────────────────────────────────────────


class TestMissingValues:
    def test_no_missing(self, simple_df):
        miss = _get_missing_stats(simple_df)
        assert miss["total_missing"] == 0
        assert miss["total_missing_pct"] == 0.0
        for col_info in miss["per_column"].values():
            assert col_info["missing_count"] == 0

    def test_partial_missing(self, df_with_missing):
        miss = _get_missing_stats(df_with_missing)
        assert miss["per_column"]["a"]["missing_count"] == 1
        assert miss["per_column"]["b"]["missing_count"] == 5
        assert miss["per_column"]["c"]["missing_count"] == 1

    def test_total_missing_count(self, df_with_missing):
        miss = _get_missing_stats(df_with_missing)
        # a=1, b=5, c=1
        assert miss["total_missing"] == 7

    def test_missing_percentage_column(self, df_with_missing):
        miss = _get_missing_stats(df_with_missing)
        assert miss["per_column"]["b"]["missing_pct"] == pytest.approx(100.0)

    def test_missing_pct_partial(self, df_with_missing):
        miss = _get_missing_stats(df_with_missing)
        assert miss["per_column"]["a"]["missing_pct"] == pytest.approx(20.0)

    def test_total_missing_pct(self, df_with_missing):
        miss = _get_missing_stats(df_with_missing)
        # 7 missing out of 15 cells
        assert miss["total_missing_pct"] == pytest.approx(7 / 15 * 100, rel=1e-3)

    def test_all_null_column_pct(self):
        df = pd.DataFrame({"x": [None, None, None]})
        miss = _get_missing_stats(df)
        assert miss["per_column"]["x"]["missing_pct"] == pytest.approx(100.0)


# ──────────────────────────────────────────────────────────────────────────────
# Duplicate-row tests
# ──────────────────────────────────────────────────────────────────────────────


class TestDuplicates:
    def test_no_duplicates(self, simple_df):
        dups = _get_duplicate_stats(simple_df)
        assert dups["n_duplicates"] == 0
        assert dups["duplicate_pct"] == 0.0

    def test_with_duplicates(self, df_with_duplicates):
        dups = _get_duplicate_stats(df_with_duplicates)
        # Rows (2,b) and (3,c) each appear twice → 2 duplicate rows
        assert dups["n_duplicates"] == 2

    def test_duplicate_pct(self, df_with_duplicates):
        dups = _get_duplicate_stats(df_with_duplicates)
        assert dups["duplicate_pct"] == pytest.approx(2 / 5 * 100)

    def test_all_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 1], "b": ["x", "x", "x"]})
        dups = _get_duplicate_stats(df)
        assert dups["n_duplicates"] == 2  # first row is original


# ──────────────────────────────────────────────────────────────────────────────
# Numerical statistics tests
# ──────────────────────────────────────────────────────────────────────────────


class TestNumericalStats:
    def test_mean(self, simple_df):
        stats = _get_numerical_stats(simple_df)
        assert stats["age"]["mean"] == pytest.approx(35.0)

    def test_median(self, simple_df):
        stats = _get_numerical_stats(simple_df)
        assert stats["age"]["median"] == pytest.approx(35.0)

    def test_std(self, simple_df):
        stats = _get_numerical_stats(simple_df)
        expected_std = pd.Series([25, 30, 35, 40, 45], dtype=float).std()
        assert stats["age"]["std"] == pytest.approx(expected_std)

    def test_min(self, simple_df):
        stats = _get_numerical_stats(simple_df)
        assert stats["age"]["min"] == pytest.approx(25.0)

    def test_max(self, simple_df):
        stats = _get_numerical_stats(simple_df)
        assert stats["age"]["max"] == pytest.approx(45.0)

    def test_q1_q3(self, simple_df):
        stats = _get_numerical_stats(simple_df)
        series = pd.Series([25, 30, 35, 40, 45], dtype=float)
        assert stats["age"]["q1"] == pytest.approx(series.quantile(0.25))
        assert stats["age"]["q3"] == pytest.approx(series.quantile(0.75))

    def test_iqr(self, simple_df):
        stats = _get_numerical_stats(simple_df)
        series = pd.Series([25, 30, 35, 40, 45], dtype=float)
        expected_iqr = series.quantile(0.75) - series.quantile(0.25)
        assert stats["age"]["iqr"] == pytest.approx(expected_iqr)

    def test_skewness_symmetric(self, simple_df):
        # age = [25,30,35,40,45] is symmetric — skewness near 0
        stats = _get_numerical_stats(simple_df)
        assert stats["age"]["skewness"] == pytest.approx(0.0, abs=0.01)

    def test_all_nan_column(self):
        df = pd.DataFrame({"x": [np.nan, np.nan, np.nan]})
        stats = _get_numerical_stats(df)
        assert stats["x"]["mean"] is None
        assert stats["x"]["std"] is None

    def test_too_few_obs(self):
        df = pd.DataFrame({"x": [1.0, 2.0, None, None, None]})
        stats = _get_numerical_stats(df)
        # n_valid=2 < _MIN_OBS_FOR_STATS=3
        assert stats["x"]["mean"] == pytest.approx(1.5)
        assert stats["x"]["std"] is None

    def test_constant_column(self):
        df = pd.DataFrame({"x": [7.0, 7.0, 7.0, 7.0, 7.0]})
        stats = _get_numerical_stats(df)
        assert stats["x"]["mean"] == pytest.approx(7.0)
        assert stats["x"]["std"] == pytest.approx(0.0)
        assert stats["x"]["iqr"] == pytest.approx(0.0)

    def test_salary_mean(self, simple_df):
        stats = _get_numerical_stats(simple_df)
        assert stats["salary"]["mean"] == pytest.approx(70000.0)

    def test_non_numerical_columns_excluded(self, simple_df):
        stats = _get_numerical_stats(simple_df)
        assert "name" not in stats
        assert "department" not in stats


# ──────────────────────────────────────────────────────────────────────────────
# Outlier detection tests
# ──────────────────────────────────────────────────────────────────────────────


class TestOutlierDetection:
    def test_outliers_detected(self, df_with_outliers):
        outliers = _get_outlier_stats(df_with_outliers)
        info = outliers["val"]
        assert info["n_potential_outliers"] >= 1

    def test_outlier_fences_set(self, df_with_outliers):
        outliers = _get_outlier_stats(df_with_outliers)
        info = outliers["val"]
        assert info["lower_fence"] is not None
        assert info["upper_fence"] is not None
        # Upper fence must be less than the obvious outlier (1000)
        assert info["upper_fence"] < 1000

    def test_no_outliers_in_uniform(self):
        df = pd.DataFrame({"x": [10, 11, 12, 13, 14, 15]})
        outliers = _get_outlier_stats(df)
        assert outliers["x"]["n_potential_outliers"] == 0

    def test_constant_column_iqr_zero(self):
        df = pd.DataFrame({"x": [5.0, 5.0, 5.0, 5.0, 5.0]})
        outliers = _get_outlier_stats(df)
        assert outliers["x"]["n_potential_outliers"] == 0
        assert "IQR is 0" in outliers["x"]["note"]

    def test_too_few_obs_returns_none(self):
        df = pd.DataFrame({"x": [1.0, 2.0]})
        outliers = _get_outlier_stats(df)
        assert outliers["x"]["n_potential_outliers"] is None

    def test_outlier_pct_in_range(self, df_with_outliers):
        outliers = _get_outlier_stats(df_with_outliers)
        pct = outliers["val"]["outlier_pct"]
        assert 0.0 <= pct <= 100.0

    def test_note_field_present(self, df_with_outliers):
        outliers = _get_outlier_stats(df_with_outliers)
        assert "note" in outliers["val"]


# ──────────────────────────────────────────────────────────────────────────────
# Cardinality tests
# ──────────────────────────────────────────────────────────────────────────────


class TestCardinality:
    def test_constant_column_detected(self, df_constant):
        card = _get_cardinality(df_constant)
        assert "const_num" in card["constant_columns"]
        assert "const_str" in card["constant_columns"]

    def test_non_constant_not_flagged(self, df_constant):
        card = _get_cardinality(df_constant)
        assert "normal" not in card["constant_columns"]

    def test_unique_counts(self, simple_df):
        card = _get_cardinality(simple_df)
        assert card["per_column"]["department"]["n_unique"] == 2
        assert card["per_column"]["name"]["n_unique"] == 5

    def test_high_cardinality_detected(self, df_high_cardinality):
        card = _get_cardinality(df_high_cardinality)
        assert "id_col" in card["high_cardinality_columns"]

    def test_low_cardinality_not_flagged(self, df_high_cardinality):
        card = _get_cardinality(df_high_cardinality)
        assert "category" not in card["high_cardinality_columns"]

    def test_all_null_unique_count(self):
        df = pd.DataFrame({"x": [None, None, None]})
        card = _get_cardinality(df)
        assert card["per_column"]["x"]["n_unique"] == 0


# ──────────────────────────────────────────────────────────────────────────────
# Datetime detection tests
# ──────────────────────────────────────────────────────────────────────────────


class TestDatetimeDetection:
    def test_date_string_column_detected(self, df_datetime):
        dt_cols = _detect_datetime_columns(df_datetime)
        assert "date_str" in dt_cols

    def test_non_date_column_not_detected(self, df_datetime):
        dt_cols = _detect_datetime_columns(df_datetime)
        assert "not_date" not in dt_cols

    def test_numeric_column_not_datetime(self, df_datetime):
        dt_cols = _detect_datetime_columns(df_datetime)
        assert "value" not in dt_cols

    def test_already_datetime_dtype(self):
        df = pd.DataFrame({"ts": pd.to_datetime(["2021-01-01", "2022-06-15"])})
        dt_cols = _detect_datetime_columns(df)
        assert "ts" in dt_cols

    def test_no_datetime_if_mostly_unparseable(self):
        df = pd.DataFrame({"text": ["hello", "world", "foo", "bar", "baz"]})
        dt_cols = _detect_datetime_columns(df)
        assert "text" not in dt_cols


# ──────────────────────────────────────────────────────────────────────────────
# Column type classification tests
# ──────────────────────────────────────────────────────────────────────────────


class TestColumnTypes:
    def test_numerical_classified(self, simple_df):
        dt_cols = _detect_datetime_columns(simple_df)
        types = _get_column_types(simple_df, dt_cols)
        assert types["age"] == "numerical"
        assert types["salary"] == "numerical"

    def test_categorical_classified(self, simple_df):
        dt_cols = _detect_datetime_columns(simple_df)
        types = _get_column_types(simple_df, dt_cols)
        assert types["name"] == "categorical"
        assert types["department"] == "categorical"

    def test_datetime_classified(self, df_datetime):
        dt_cols = _detect_datetime_columns(df_datetime)
        types = _get_column_types(df_datetime, dt_cols)
        assert types["date_str"] == "datetime"


# ──────────────────────────────────────────────────────────────────────────────
# Integration: profile_dataset
# ──────────────────────────────────────────────────────────────────────────────


class TestProfileDataset:
    def test_returns_all_keys(self, simple_df):
        profile = profile_dataset(simple_df)
        expected = {"overview", "missing", "duplicates", "column_types", "cardinality", "numerical_stats", "outliers", "datetime_cols"}
        assert expected == set(profile.keys())

    def test_original_df_not_mutated(self, simple_df):
        original_cols = list(simple_df.columns)
        original_shape = simple_df.shape
        _ = profile_dataset(simple_df)
        assert list(simple_df.columns) == original_cols
        assert simple_df.shape == original_shape

    def test_invalid_input_raises_typeerror(self):
        with pytest.raises(TypeError):
            profile_dataset(None)

    def test_invalid_input_dict_raises(self):
        with pytest.raises(TypeError):
            profile_dataset({"a": [1, 2, 3]})

    def test_deterministic(self, simple_df):
        """Same input must produce identical output."""
        p1 = profile_dataset(simple_df)
        p2 = profile_dataset(simple_df)
        assert p1["overview"] == p2["overview"]
        assert p1["missing"] == p2["missing"]
        assert p1["duplicates"] == p2["duplicates"]
        for col in p1["numerical_stats"]:
            for key, v1 in p1["numerical_stats"][col].items():
                v2 = p2["numerical_stats"][col][key]
                if v1 is None:
                    assert v2 is None
                else:
                    assert v1 == pytest.approx(v2)

    def test_single_column_df(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        profile = profile_dataset(df)
        assert profile["overview"]["n_cols"] == 1

    def test_single_row_df(self):
        df = pd.DataFrame({"a": [42], "b": ["hello"]})
        profile = profile_dataset(df)
        assert profile["overview"]["n_rows"] == 1

    def test_all_missing_df(self):
        df = pd.DataFrame({"a": [None, None], "b": [None, None]})
        profile = profile_dataset(df)
        assert profile["missing"]["total_missing"] == 4

"""
backend/tests/test_profiler.py
================================
DataInvest AI — Phase 1 Backend Statistical & Quality Validation Tests

Comprehensive validation for:
- Exact mathematical calculation of all statistics
- Row, column, and dtype detection
- Missing-value count and percentages
- Duplicate row detection
- Unique-value counts & cardinality
- Constant-column detection (numeric and string)
- High-cardinality detection
- IQR-based outlier detection (1.5x rule)
- Non-destructive datetime detection
- Problematic columns (<3 obs, constant, all-NaN, Inf values)
- Strict JSON RFC 8259 compliance (allow_nan=False)
- Pydantic v2 ProfileResponse schema validation
- Immutability and determinism
- Upload service parsing and multi-encoding
"""

from __future__ import annotations

import io
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

# Ensure backend/app is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from app.modules.profiler import (
    profile_dataset,
    prepare_preview_rows,
    _get_overview,
    _get_missing_stats,
    _get_duplicate_stats,
    _get_numerical_stats,
    _get_outlier_stats,
    _get_cardinality,
    _get_column_types,
    _detect_datetime_columns,
)
from app.schemas.profile import ProfileResponse
from app.services.upload_service import parse_csv


# ──────────────────────────────────────────────────────────────────────────────
# Test Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def simple_df() -> pd.DataFrame:
    """Standard controlled tabular dataset with known properties."""
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 45],
        "salary": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
        "name": ["Alice", "Bob", "Carol", "David", "Eve"],
        "department": ["Eng", "Eng", "HR", "Eng", "HR"],
    })


@pytest.fixture()
def controlled_stats_df() -> pd.DataFrame:
    """Exact controlled dataset: [10, 20, 30, 40, 50]."""
    return pd.DataFrame({
        "vals": [10.0, 20.0, 30.0, 40.0, 50.0]
    })


@pytest.fixture()
def df_with_missing() -> pd.DataFrame:
    return pd.DataFrame({
        "a": [1.0, 2.0, None, 4.0, 5.0],
        "b": [None, None, None, None, None],
        "c": ["x", "y", "z", None, "w"],
    })


@pytest.fixture()
def df_with_duplicates() -> pd.DataFrame:
    return pd.DataFrame({
        "x": [1, 2, 2, 3, 3],
        "y": ["a", "b", "b", "c", "c"],
    })


@pytest.fixture()
def df_with_outliers() -> pd.DataFrame:
    return pd.DataFrame({
        "val": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 1000.0, -500.0]
    })


@pytest.fixture()
def df_constant() -> pd.DataFrame:
    return pd.DataFrame({
        "const_num": [5, 5, 5, 5],
        "const_str": ["X", "X", "X", "X"],
        "normal": [1, 2, 3, 4],
    })


@pytest.fixture()
def df_datetime() -> pd.DataFrame:
    return pd.DataFrame({
        "date_str": ["2021-01-01", "2021-06-15", "2022-03-10", "2023-07-22"],
        "not_date": ["apple", "banana", "cherry", "mango"],
        "value": [10, 20, 30, 40],
    })


@pytest.fixture()
def df_high_cardinality() -> pd.DataFrame:
    return pd.DataFrame({
        "id_col": [f"user_{i}" for i in range(100)],
        "category": ["A", "B", "A", "C"] * 25,
    })


# ──────────────────────────────────────────────────────────────────────────────
# 1. Dataset Dimensions & Overview
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
        dtypes = _get_overview(simple_df)["dtypes"]
        assert set(dtypes.keys()) == {"age", "salary", "name", "department"}

    def test_memory_usage_positive(self, simple_df):
        assert _get_overview(simple_df)["memory_usage_bytes"] > 0

    def test_values_are_native_types(self, simple_df):
        ov = _get_overview(simple_df)
        assert isinstance(ov["n_rows"], int)
        assert isinstance(ov["n_cols"], int)
        assert isinstance(ov["memory_usage_bytes"], int)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Missing-Value Diagnostics
# ──────────────────────────────────────────────────────────────────────────────


class TestMissingValues:
    def test_no_missing(self, simple_df):
        m = _get_missing_stats(simple_df)
        assert m["total_missing"] == 0
        assert m["total_missing_pct"] == 0.0
        assert all(v["missing_count"] == 0 for v in m["per_column"].values())

    def test_partial_missing_col(self, df_with_missing):
        m = _get_missing_stats(df_with_missing)
        assert m["per_column"]["a"]["missing_count"] == 1
        assert m["per_column"]["a"]["missing_pct"] == pytest.approx(20.0)

    def test_all_missing_col(self, df_with_missing):
        m = _get_missing_stats(df_with_missing)
        assert m["per_column"]["b"]["missing_count"] == 5
        assert m["per_column"]["b"]["missing_pct"] == pytest.approx(100.0)

    def test_total_missing(self, df_with_missing):
        m = _get_missing_stats(df_with_missing)
        assert m["total_missing"] == 7
        assert m["total_cells"] == 15
        assert m["total_missing_pct"] == pytest.approx(7 / 15 * 100, rel=1e-3)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Duplicate-Row Counting
# ──────────────────────────────────────────────────────────────────────────────


class TestDuplicates:
    def test_no_duplicates(self, simple_df):
        d = _get_duplicate_stats(simple_df)
        assert d["n_duplicates"] == 0
        assert d["duplicate_pct"] == 0.0

    def test_with_duplicates(self, df_with_duplicates):
        d = _get_duplicate_stats(df_with_duplicates)
        assert d["n_duplicates"] == 2
        assert d["duplicate_pct"] == pytest.approx(40.0)

    def test_all_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 1]})
        assert _get_duplicate_stats(df)["n_duplicates"] == 2


# ──────────────────────────────────────────────────────────────────────────────
# 4. Exact Mathematical Verification of Numerical Statistics
# ──────────────────────────────────────────────────────────────────────────────


class TestExactNumericalCalculations:
    """
    Dataset: [10, 20, 30, 40, 50]
    Exact stats:
      - mean: 30.0
      - median: 30.0
      - std: sqrt(250) = 15.811388...
      - min: 10.0, max: 50.0
      - q1: 20.0, q3: 40.0, iqr: 20.0
      - skewness: 0.0
    """

    def test_exact_mean(self, controlled_stats_df):
        stats = _get_numerical_stats(controlled_stats_df)["vals"]
        assert stats["mean"] == pytest.approx(30.0)

    def test_exact_median(self, controlled_stats_df):
        stats = _get_numerical_stats(controlled_stats_df)["vals"]
        assert stats["median"] == pytest.approx(30.0)

    def test_exact_std(self, controlled_stats_df):
        stats = _get_numerical_stats(controlled_stats_df)["vals"]
        assert stats["std"] == pytest.approx(math.sqrt(250.0), rel=1e-5)

    def test_exact_min_max(self, controlled_stats_df):
        stats = _get_numerical_stats(controlled_stats_df)["vals"]
        assert stats["min"] == pytest.approx(10.0)
        assert stats["max"] == pytest.approx(50.0)

    def test_exact_quartiles_and_iqr(self, controlled_stats_df):
        stats = _get_numerical_stats(controlled_stats_df)["vals"]
        assert stats["q1"] == pytest.approx(20.0)
        assert stats["q3"] == pytest.approx(40.0)
        assert stats["iqr"] == pytest.approx(20.0)

    def test_exact_skewness_symmetric(self, controlled_stats_df):
        stats = _get_numerical_stats(controlled_stats_df)["vals"]
        assert stats["skewness"] == pytest.approx(0.0, abs=0.001)

    def test_non_numeric_excluded(self, simple_df):
        s = _get_numerical_stats(simple_df)
        assert "name" not in s
        assert "department" not in s

    def test_values_are_native_types(self, simple_df):
        s = _get_numerical_stats(simple_df)["age"]
        for k, v in s.items():
            if v is not None:
                assert isinstance(v, (int, float)), f"{k}={v!r} is not a Python native type"


# ──────────────────────────────────────────────────────────────────────────────
# 5. Outlier Detection (1.5x IQR Rule)
# ──────────────────────────────────────────────────────────────────────────────


class TestOutlierDetection:
    def test_controlled_no_outliers(self, controlled_stats_df):
        # [10, 20, 30, 40, 50]: Q1=20, Q3=40, IQR=20 => lower = -10, upper = 70
        o = _get_outlier_stats(controlled_stats_df)["vals"]
        assert o["n_potential_outliers"] == 0
        assert o["lower_fence"] == pytest.approx(-10.0)
        assert o["upper_fence"] == pytest.approx(70.0)
        assert o["outlier_pct"] == 0.0

    def test_outliers_detected(self, df_with_outliers):
        o = _get_outlier_stats(df_with_outliers)["val"]
        assert o["n_potential_outliers"] >= 2  # 1000 and -500
        assert o["lower_fence"] is not None
        assert o["upper_fence"] is not None
        assert o["upper_fence"] < 1000.0
        assert o["lower_fence"] > -500.0

    def test_constant_col_iqr_zero(self, df_constant):
        o = _get_outlier_stats(df_constant)["const_num"]
        assert o["n_potential_outliers"] == 0
        assert o["outlier_pct"] == 0.0
        assert "IQR is 0" in o["note"]


# ──────────────────────────────────────────────────────────────────────────────
# 6. Cardinality, Constant Columns & High-Cardinality
# ──────────────────────────────────────────────────────────────────────────────


class TestCardinality:
    def test_constant_detected(self, df_constant):
        c = _get_cardinality(df_constant)
        assert "const_num" in c["constant_columns"]
        assert "const_str" in c["constant_columns"]
        assert "normal" not in c["constant_columns"]

    def test_unique_counts(self, simple_df):
        c = _get_cardinality(simple_df)
        assert c["per_column"]["department"]["n_unique"] == 2
        assert c["per_column"]["name"]["n_unique"] == 5

    def test_high_cardinality_detected(self, df_high_cardinality):
        c = _get_cardinality(df_high_cardinality)
        assert "id_col" in c["high_cardinality_columns"]
        assert "category" not in c["high_cardinality_columns"]

    def test_all_null_unique_zero(self):
        df = pd.DataFrame({"x": [None, None, None]})
        c = _get_cardinality(df)
        assert c["per_column"]["x"]["n_unique"] == 0


# ──────────────────────────────────────────────────────────────────────────────
# 7. Datetime Column Detection
# ──────────────────────────────────────────────────────────────────────────────


class TestDatetimeDetection:
    def test_date_string_detected(self, df_datetime):
        dt_cols = _detect_datetime_columns(df_datetime)
        assert "date_str" in dt_cols
        assert "not_date" not in dt_cols
        assert "value" not in dt_cols

    def test_native_datetime_dtype(self):
        df = pd.DataFrame({"ts": pd.to_datetime(["2021-01-01", "2022-06-15"])})
        assert "ts" in _detect_datetime_columns(df)

    def test_numeric_string_not_date(self):
        df = pd.DataFrame({"zip_codes": ["90210", "10001", "94105", "30301"]})
        assert "zip_codes" not in _detect_datetime_columns(df)

    def test_column_types_mapping(self, df_datetime):
        dt_cols = _detect_datetime_columns(df_datetime)
        types = _get_column_types(df_datetime, dt_cols)
        assert types["date_str"] == "datetime"
        assert types["not_date"] == "categorical"
        assert types["value"] == "numerical"


# ──────────────────────────────────────────────────────────────────────────────
# 8. Problematic & Edge-Case Datasets
# ──────────────────────────────────────────────────────────────────────────────


class TestProblematicData:
    def test_all_nan_column(self):
        df = pd.DataFrame({"x": [np.nan, np.nan, np.nan]})
        stats = _get_numerical_stats(df)["x"]
        assert stats["n_valid"] == 0
        assert stats["mean"] is None
        assert stats["std"] is None
        assert stats["median"] is None

    def test_single_observation_column(self):
        df = pd.DataFrame({"x": [42.0, np.nan, np.nan]})
        stats = _get_numerical_stats(df)["x"]
        assert stats["n_valid"] == 1
        assert stats["mean"] == pytest.approx(42.0)
        assert stats["median"] == pytest.approx(42.0)
        assert stats["std"] is None
        assert stats["q1"] is None

    def test_two_observations_column(self):
        df = pd.DataFrame({"x": [10.0, 20.0, np.nan]})
        stats = _get_numerical_stats(df)["x"]
        assert stats["n_valid"] == 2
        assert stats["mean"] == pytest.approx(15.0)
        assert stats["median"] == pytest.approx(15.0)
        assert stats["std"] is None

    def test_single_row_dataset(self):
        df = pd.DataFrame({"a": [10], "b": ["x"]})
        p = profile_dataset(df)
        assert p["overview"]["n_rows"] == 1
        assert p["overview"]["n_cols"] == 2

    def test_single_column_dataset(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        p = profile_dataset(df)
        assert p["overview"]["n_cols"] == 1

    def test_all_missing_dataframe(self):
        df = pd.DataFrame({"a": [None, None], "b": [None, None]})
        p = profile_dataset(df)
        assert p["missing"]["total_missing"] == 4
        assert p["missing"]["total_missing_pct"] == 100.0

    def test_infinity_values_handled_safely(self):
        df = pd.DataFrame({"num": [1.0, np.inf, -np.inf, 2.0]})
        p = profile_dataset(df)
        # Must serialize cleanly to strict JSON
        dumped = json.dumps(p, allow_nan=False)
        assert isinstance(dumped, str)

    def test_prepare_preview_rows_strict_json(self):
        df = pd.DataFrame({
            "a": [1.5, np.nan, np.inf, 4.0],
            "b": ["x", None, "z", "w"],
            "dt": pd.to_datetime(["2021-01-01", pd.NaT, "2021-03-01", "2021-04-01"])
        })
        rows = prepare_preview_rows(df, n=10)
        assert len(rows) == 4
        # Assert strict JSON serialization without NaN or Infinity tokens
        dumped = json.dumps(rows, allow_nan=False)
        assert isinstance(dumped, str)


# ──────────────────────────────────────────────────────────────────────────────
# 9. Schema Validation, Immutability & Determinism
# ──────────────────────────────────────────────────────────────────────────────


class TestQualityAndContracts:
    def test_input_dataframe_immutability(self, simple_df):
        orig_cols = list(simple_df.columns)
        orig_shape = simple_df.shape
        orig_values = simple_df.values.copy()

        _ = profile_dataset(simple_df)

        assert list(simple_df.columns) == orig_cols
        assert simple_df.shape == orig_shape
        np.testing.assert_array_equal(simple_df.values, orig_values)

    def test_profiler_determinism(self, simple_df):
        p1 = profile_dataset(simple_df)
        p2 = profile_dataset(simple_df)
        assert p1 == p2

    def test_pydantic_profile_response_validation(self, simple_df):
        p = profile_dataset(simple_df)
        p["filename"] = "test.csv"
        p["preview_rows"] = prepare_preview_rows(simple_df, n=10)

        # Validate against the Pydantic schema
        model = ProfileResponse.model_validate(p)
        assert model.overview.n_rows == 5
        assert model.overview.n_cols == 4

    def test_pydantic_validation_with_outliers(self, df_with_outliers):
        p = profile_dataset(df_with_outliers)
        p["filename"] = "outliers.csv"
        p["preview_rows"] = prepare_preview_rows(df_with_outliers, n=10)

        model = ProfileResponse.model_validate(p)
        assert model.outliers["val"].n_potential_outliers >= 2

    def test_strict_json_serialization(self, simple_df, df_with_outliers, df_with_missing):
        for df in (simple_df, df_with_outliers, df_with_missing):
            p = profile_dataset(df)
            p["preview_rows"] = prepare_preview_rows(df, n=10)
            # allow_nan=False strictly fails if any NaN, Inf, -Inf is present
            dumped = json.dumps(p, allow_nan=False)
            assert isinstance(dumped, str)


# ──────────────────────────────────────────────────────────────────────────────
# 10. CSV Parsing & Upload Service
# ──────────────────────────────────────────────────────────────────────────────


class TestUploadService:
    def _to_bytes(self, text: str, encoding: str = "utf-8") -> bytes:
        return text.encode(encoding)

    def test_valid_csv(self):
        content = self._to_bytes("a,b,c\n1,2,3\n4,5,6\n")
        df, err = parse_csv(content)
        assert err is None
        assert df is not None
        assert len(df) == 2
        assert list(df.columns) == ["a", "b", "c"]

    def test_empty_bytes_rejected(self):
        df, err = parse_csv(b"")
        assert df is None
        assert err is not None
        assert "empty" in err.lower()

    def test_whitespace_only_rejected(self):
        df, err = parse_csv(b"   \n   \n")
        assert df is None
        assert err is not None

    def test_header_only_rejected(self):
        df, err = parse_csv(self._to_bytes("col1,col2,col3\n"))
        assert df is None
        assert err is not None
        assert "no data rows" in err.lower()

    def test_utf8_sig_encoding(self):
        content = self._to_bytes("col_a,col_b\n1,2\n3,4\n", encoding="utf-8-sig")
        df, err = parse_csv(content)
        assert err is None
        assert df is not None
        assert "col_a" in df.columns

    def test_latin1_encoding(self):
        content = "name,location\nFrançois,Montréal\nJürgen,München\n".encode("latin-1")
        df, err = parse_csv(content)
        assert err is None
        assert df is not None
        assert len(df) == 2

    def test_cp1252_encoding(self):
        content = "item,price\nItem1,€10\nItem2,€20\n".encode("cp1252")
        df, err = parse_csv(content)
        assert err is None
        assert df is not None
        assert len(df) == 2

    def test_corrupted_bytes_handled_cleanly(self):
        # Corrupted non-text binary garbage
        content = b"\x00\x01\x02\x03\x04\xff\xfe\xfd\x00\x00\x00"
        df, err = parse_csv(content)
        # Must fail gracefully without unhandled exception
        assert df is None or len(df) >= 0

"""
backend/app/modules/profiler.py
================================
DataInvest AI — Phase 1: Dataset Profiler Engine

Deterministic, pure Python / Pandas / NumPy statistical profiling.
All output values are strictly JSON-serializable Python native types.
No raw NumPy scalars, NaNs, or Infinities escape to the caller.
Original input DataFrames are never mutated.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Profiler Configuration Constants
# ──────────────────────────────────────────────────────────────────────────────

_DATETIME_PARSE_THRESHOLD: float = 0.80
_HIGH_CARDINALITY_THRESHOLD: float = 0.50
_IQR_FENCE_MULTIPLIER: float = 1.5
_MIN_OBS_FOR_STATS: int = 3
_NUMERIC_STRING_THRESHOLD: float = 0.80
_MONTH_ONLY_NAMES: set[str] = {
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
    "january", "february", "march", "april", "june",
    "july", "august", "september", "october", "november", "december"
}


# ──────────────────────────────────────────────────────────────────────────────
# JSON-Safe Scalar Converters
# ──────────────────────────────────────────────────────────────────────────────


def _safe(value: Any) -> float | None:
    """Convert any numeric/numpy scalar to float; return None for NaN/Inf/None."""
    try:
        if value is None:
            return None
        f = float(value)
        return None if math.isnan(f) or math.isinf(f) else round(f, 6)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    """Convert value to int; return None on failure."""
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Public Profiling API
# ──────────────────────────────────────────────────────────────────────────────


def profile_dataset(
    df: pd.DataFrame,
    dataset_name: Optional[str] = None,
) -> dict[str, Any]:
    """
    Generate a complete, deterministic, JSON-serializable profile of *df*.

    Parameters
    ----------
    df : pd.DataFrame
        The uploaded tabular dataset. Never modified by this function.
    dataset_name : str, optional
        Dataset filename or identifier to resolve verified benchmark schemas.

    Returns
    -------
    dict
        Structured profile dictionary adhering to the ProfileResponse schema.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise TypeError("profile_dataset requires a pandas DataFrame.")

    # Work strictly on an isolated copy to guarantee input immutability
    work = df.copy(deep=True)

    # 1. Identify datetime columns before category assignment
    datetime_cols = _detect_datetime_columns(work)

    profile_dict = {
        "overview": _get_overview(work),
        "missing": _get_missing_stats(work),
        "duplicates": _get_duplicate_stats(work),
        "column_types": _get_column_types(work, datetime_cols, dataset_name=dataset_name),
        "cardinality": _get_cardinality(work),
        "numerical_stats": _get_numerical_stats(work),
        "outliers": _get_outlier_stats(work),
        "datetime_cols": list(datetime_cols),
    }
    prep_rules = getattr(work, "attrs", {}).get("preprocessing_rules") or getattr(df, "attrs", {}).get("preprocessing_rules")
    if prep_rules:
        profile_dict["preprocessing_rules"] = prep_rules

    return profile_dict


def prepare_preview_rows(df: pd.DataFrame, n: int = 100) -> list[dict[str, Any]]:
    """
    Extract first *n* rows of *df* as clean JSON-serializable records (null for NaN/Inf).
    """
    subset = df.head(n)
    rows: list[dict[str, Any]] = []
    for _, row in subset.iterrows():
        row_dict: dict[str, Any] = {}
        for col, val in row.items():
            col_str = str(col)
            if pd.isna(val):
                row_dict[col_str] = None
            elif isinstance(val, (np.floating, float)):
                if math.isnan(val) or math.isinf(val):
                    row_dict[col_str] = None
                else:
                    row_dict[col_str] = round(float(val), 6)
            elif isinstance(val, (np.integer, int)):
                row_dict[col_str] = int(val)
            elif isinstance(val, (pd.Timestamp, np.datetime64)):
                row_dict[col_str] = str(val)
            elif isinstance(val, (bool, np.bool_)):
                row_dict[col_str] = bool(val)
            else:
                row_dict[col_str] = str(val)
        rows.append(row_dict)
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Internal Statistical Profiling Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _get_overview(df: pd.DataFrame) -> dict[str, Any]:
    """Compute dataset dimensions, column names, dtypes, and memory footprint."""
    return {
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
        "column_names": [str(c) for c in df.columns],
        "dtypes": {str(col): str(df[col].dtype) for col in df.columns},
        "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
    }


def _get_missing_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Calculate per-column and aggregate missing-value statistics."""
    n_rows = int(len(df))
    per_col: dict[str, dict[str, Any]] = {}
    for col in df.columns:
        mc = int(df[col].isna().sum())
        pct = round(mc / n_rows * 100, 2) if n_rows > 0 else 0.0
        per_col[str(col)] = {"missing_count": mc, "missing_pct": pct}

    total_cells = n_rows * int(len(df.columns))
    total_missing = int(df.isna().sum().sum())
    res = {
        "per_column": per_col,
        "total_missing": total_missing,
        "total_cells": total_cells,
        "total_missing_pct": round(total_missing / total_cells * 100, 2)
        if total_cells > 0
        else 0.0,
    }
    prov = getattr(df, "attrs", {}).get("missing_value_provenance")
    if prov:
        res["provenance"] = prov
    return res


def _get_duplicate_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Calculate duplicate row count and percentage."""
    n_dup = int(df.duplicated().sum())
    n_rows = int(len(df))
    return {
        "n_duplicates": n_dup,
        "duplicate_pct": round(n_dup / n_rows * 100, 2) if n_rows > 0 else 0.0,
    }


def _is_string_like(series: pd.Series) -> bool:
    """Return True if series has object or Pandas 3+ StringDtype."""
    if series.dtype == object:
        return True
    if pd.api.types.is_string_dtype(series):
        return True
    return False


def _detect_datetime_columns(df: pd.DataFrame) -> list[str]:
    """
    Identify columns that parse consistently as dates without mutating data.
    Skips numeric columns and purely numeric string sequences (e.g. zip codes).
    """
    dt_cols: list[str] = []
    for col in df.columns:
        series = df[col]
        try:
            # Native datetime dtype
            if pd.api.types.is_datetime64_any_dtype(series):
                dt_cols.append(str(col))
                continue

            # Skip numerics and booleans
            if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
                continue

            # Only inspect string-like columns
            if not _is_string_like(series):
                continue

            non_null = series.dropna()
            if len(non_null) == 0:
                continue

            # If column is purely numeric strings (like IDs), don't treat as date
            num_parsed = pd.to_numeric(non_null, errors="coerce")
            if (num_parsed.notna().sum() / len(non_null)) >= _NUMERIC_STRING_THRESHOLD:
                continue

            # If column values are solely month-only labels (without year or day), treat as categorical,
            # avoiding accidental conversion to arbitrary current-year timestamps.
            clean_str = non_null.astype(str).str.strip().str.lower()
            if (clean_str.isin(_MONTH_ONLY_NAMES).sum() / len(non_null)) >= 0.8:
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(non_null, errors="coerce")

            rate = float(parsed.notna().sum()) / len(non_null)
            if rate >= _DATETIME_PARSE_THRESHOLD:
                dt_cols.append(str(col))
        except Exception:
            continue
    return dt_cols


def _get_column_types(
    df: pd.DataFrame,
    datetime_cols: list[str],
    dataset_name: Optional[str] = None,
) -> dict[str, str]:
    """Classify columns into 'numerical', 'categorical', or 'datetime'."""
    ds_name = (dataset_name or getattr(df, "attrs", {}).get("dataset_name") or "").lower()
    schema_types: dict[str, str] = {}
    if ds_name:
        try:
            from app.services.upload_service import VERIFIED_BENCHMARK_SCHEMAS
            schema = VERIFIED_BENCHMARK_SCHEMAS.get(ds_name)
            if schema and "column_types" in schema:
                schema_types = schema["column_types"]
        except ImportError:
            pass

    result: dict[str, str] = {}
    for col in df.columns:
        s = str(col)
        if s in schema_types:
            result[s] = schema_types[s]
        elif s in datetime_cols:
            result[s] = "datetime"
        elif pd.api.types.is_numeric_dtype(df[col]) and not isinstance(df[col].dtype, pd.CategoricalDtype):
            result[s] = "numerical"
        else:
            result[s] = "categorical"
    return result


def _get_cardinality(df: pd.DataFrame) -> dict[str, Any]:
    """
    Compute distinct count, constant-column flags, and high-cardinality flags.
    """
    per_col: dict[str, dict[str, Any]] = {}
    constant_cols: list[str] = []
    high_card_cols: list[str] = []

    for col in df.columns:
        s = str(col)
        series = df[col]
        try:
            n_unique = int(series.nunique(dropna=True))
            n_non_null = int(series.notna().sum())
            is_constant = n_unique <= 1

            is_high_card = False
            if not pd.api.types.is_numeric_dtype(series) and n_non_null > 0:
                is_high_card = (n_unique / n_non_null) > _HIGH_CARDINALITY_THRESHOLD

            per_col[s] = {
                "n_unique": n_unique,
                "is_constant": bool(is_constant),
                "is_high_cardinality": bool(is_high_card),
            }
            if is_constant:
                constant_cols.append(s)
            if is_high_card:
                high_card_cols.append(s)
        except Exception:
            per_col[s] = {"n_unique": None, "is_constant": False, "is_high_cardinality": False}

    return {
        "per_column": per_col,
        "constant_columns": constant_cols,
        "high_cardinality_columns": high_card_cols,
    }


def _get_numerical_stats(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """
    Compute descriptive statistics for all numerical columns safely.
    Handles edge cases: empty data, <3 observations, constant columns, all-NaN.
    """
    result: dict[str, dict[str, Any]] = {}
    numerical_cols = df.select_dtypes(include=[np.number]).columns

    for col in numerical_cols:
        s = str(col)
        try:
            series = df[col].dropna()
            n_valid = int(len(series))
            stats: dict[str, Any] = {"n_valid": n_valid}

            # 1. No valid observations
            if n_valid == 0:
                for k in ("mean", "median", "std", "min", "max", "q1", "q3", "iqr", "skewness"):
                    stats[k] = None
                result[s] = stats
                continue

            # 2. Too few observations for variance/quartiles
            if n_valid < _MIN_OBS_FOR_STATS:
                stats["mean"] = _safe(series.mean())
                stats["median"] = _safe(series.median())
                stats["min"] = _safe(series.min())
                stats["max"] = _safe(series.max())
                for k in ("std", "q1", "q3", "iqr", "skewness"):
                    stats[k] = None
                result[s] = stats
                continue

            # 3. Standard statistics
            q1 = _safe(series.quantile(0.25))
            q3 = _safe(series.quantile(0.75))
            iqr = (q3 - q1) if (q1 is not None and q3 is not None) else None

            try:
                skew = _safe(series.skew())
            except Exception:
                skew = None

            stats.update({
                "mean": _safe(series.mean()),
                "median": _safe(series.median()),
                "std": _safe(series.std()),
                "min": _safe(series.min()),
                "max": _safe(series.max()),
                "q1": q1,
                "q3": q3,
                "iqr": _safe(iqr),
                "skewness": skew,
            })
            result[s] = stats
        except Exception:
            result[s] = {"n_valid": 0, **{k: None for k in ("mean", "median", "std", "min", "max", "q1", "q3", "iqr", "skewness")}}

    return result


def _get_outlier_stats(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """
    Detect potential outliers in numerical columns using 1.5 * IQR rule.
    Values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] are flagged.
    """
    result: dict[str, dict[str, Any]] = {}
    numerical_cols = df.select_dtypes(include=[np.number]).columns

    for col in numerical_cols:
        s = str(col)
        try:
            series = df[col].dropna()
            n_valid = int(len(series))

            if n_valid < _MIN_OBS_FOR_STATS:
                result[s] = {
                    "n_potential_outliers": None,
                    "outlier_pct": None,
                    "lower_fence": None,
                    "upper_fence": None,
                    "note": "Insufficient observations for IQR analysis.",
                }
                continue

            q1_val = _safe(series.quantile(0.25))
            q3_val = _safe(series.quantile(0.75))

            if q1_val is None or q3_val is None:
                result[s] = {
                    "n_potential_outliers": None,
                    "outlier_pct": None,
                    "lower_fence": None,
                    "upper_fence": None,
                    "note": "Could not determine quartiles.",
                }
                continue

            iqr = q3_val - q1_val

            if iqr == 0:
                result[s] = {
                    "n_potential_outliers": 0,
                    "outlier_pct": 0.0,
                    "lower_fence": q1_val,
                    "upper_fence": q3_val,
                    "note": "IQR is 0; column may be constant or near-constant.",
                }
                continue

            lower = q1_val - _IQR_FENCE_MULTIPLIER * iqr
            upper = q3_val + _IQR_FENCE_MULTIPLIER * iqr
            n_out = int(((series < lower) | (series > upper)).sum())

            result[s] = {
                "n_potential_outliers": n_out,
                "outlier_pct": round(n_out / n_valid * 100, 2),
                "lower_fence": round(lower, 4),
                "upper_fence": round(upper, 4),
                "note": "Values outside fences are potential outliers requiring further investigation.",
            }
        except Exception:
            result[s] = {
                "n_potential_outliers": None,
                "outlier_pct": None,
                "lower_fence": None,
                "upper_fence": None,
                "note": "Could not compute outlier statistics for this column.",
            }

    return result

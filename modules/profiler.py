"""
modules/profiler.py
===================
DataInvest AI — Phase 1: Dataset Profiler

All statistical computations are deterministic and performed exclusively by
Python, Pandas, and NumPy.  No LLM or external service is used here.

The public API is:
    profile_dataset(df: pd.DataFrame) -> dict
        Returns a structured dictionary consumed by the Streamlit UI and,
        in later phases, by the AI agent pipeline.

Internal helpers are prefixed with an underscore and are not part of the
public API.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Fraction of non-null values that must parse as datetime for us to treat the
# column as datetime-like.
_DATETIME_PARSE_THRESHOLD: float = 0.80

# A column is "high cardinality" when the ratio of unique values to non-null
# rows exceeds this threshold (and the column is categorical).
_HIGH_CARDINALITY_THRESHOLD: float = 0.50

# IQR multiplier for potential-outlier detection.
_IQR_FENCE_MULTIPLIER: float = 1.5

# Minimum non-null observations required to compute meaningful statistics.
_MIN_OBS_FOR_STATS: int = 3


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """
    Generate a comprehensive statistical profile of *df*.

    Parameters
    ----------
    df : pd.DataFrame
        The raw dataset loaded from a CSV upload.  This function **never**
        modifies *df*.

    Returns
    -------
    dict
        A structured dictionary with the following top-level keys:

        ``overview``       — basic shape / column metadata
        ``missing``        — per-column and aggregate missing-value stats
        ``duplicates``     — duplicate-row information
        ``column_types``   — type classification per column
        ``cardinality``    — unique-value counts and flags
        ``numerical_stats``— descriptive statistics for numerical columns
        ``outliers``       — IQR-based potential-outlier information
        ``datetime_cols``  — columns detected as datetime-like
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise TypeError("profile_dataset requires a pandas DataFrame.")

    # Work on a copy so the caller's DataFrame is never mutated.
    work = df.copy(deep=True)

    # Attempt datetime parsing *before* column-type classification.
    datetime_cols = _detect_datetime_columns(work)

    return {
        "overview": _get_overview(work),
        "missing": _get_missing_stats(work),
        "duplicates": _get_duplicate_stats(work),
        "column_types": _get_column_types(work, datetime_cols),
        "cardinality": _get_cardinality(work),
        "numerical_stats": _get_numerical_stats(work),
        "outliers": _get_outlier_stats(work),
        "datetime_cols": datetime_cols,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────


def _get_overview(df: pd.DataFrame) -> dict[str, Any]:
    """Return basic shape and column-level dtype metadata."""
    return {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "column_names": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
    }


def _get_missing_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Per-column and aggregate missing-value statistics."""
    n_rows = len(df)
    per_col: dict[str, dict] = {}
    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        pct = round(missing_count / n_rows * 100, 2) if n_rows > 0 else 0.0
        per_col[col] = {
            "missing_count": missing_count,
            "missing_pct": pct,
        }

    total_cells = n_rows * len(df.columns)
    total_missing = int(df.isna().sum().sum())
    return {
        "per_column": per_col,
        "total_missing": total_missing,
        "total_cells": total_cells,
        "total_missing_pct": round(total_missing / total_cells * 100, 2)
        if total_cells > 0
        else 0.0,
    }


def _get_duplicate_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Count fully duplicate rows (all columns identical)."""
    n_duplicates = int(df.duplicated().sum())
    return {
        "n_duplicates": n_duplicates,
        "duplicate_pct": round(n_duplicates / len(df) * 100, 2)
        if len(df) > 0
        else 0.0,
    }


def _detect_datetime_columns(df: pd.DataFrame) -> list[str]:
    """
    Identify columns that are likely datetime-like without mutating *df*.

    A column is considered datetime-like when:
    - Its Pandas dtype is already a datetime type, **or**
    - At least ``_DATETIME_PARSE_THRESHOLD`` fraction of its non-null values
      successfully parse as dates with ``pd.to_datetime(errors='coerce')``.

    Returns a list of column names.
    """
    dt_cols: list[str] = []
    for col in df.columns:
        series = df[col]

        # Already a datetime dtype → include immediately.
        if pd.api.types.is_datetime64_any_dtype(series):
            dt_cols.append(col)
            continue

        # Only attempt parsing on string/object columns — skip numerics and booleans.
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        # Skip if the dtype is definitely not string-like (e.g. categorical with non-str)
        if not (series.dtype == object or pd.api.types.is_string_dtype(series)):
            continue

        non_null = series.dropna()
        if len(non_null) == 0:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                parsed = pd.to_datetime(non_null, errors="coerce")
            except Exception:
                continue

        parse_rate = parsed.notna().sum() / len(non_null)
        if parse_rate >= _DATETIME_PARSE_THRESHOLD:
            dt_cols.append(col)

    return dt_cols


def _get_column_types(df: pd.DataFrame, datetime_cols: list[str]) -> dict[str, str]:
    """
    Classify each column as ``'numerical'``, ``'categorical'``, or
    ``'datetime'``.
    """
    classification: dict[str, str] = {}
    for col in df.columns:
        if col in datetime_cols:
            classification[col] = "datetime"
        elif pd.api.types.is_numeric_dtype(df[col]):
            classification[col] = "numerical"
        else:
            classification[col] = "categorical"
    return classification


def _get_cardinality(df: pd.DataFrame) -> dict[str, Any]:
    """
    Compute unique-value counts, constant-column flags, and high-cardinality
    flags for every column.
    """
    per_col: dict[str, dict] = {}
    constant_cols: list[str] = []
    high_cardinality_cols: list[str] = []

    for col in df.columns:
        series = df[col]
        n_unique = int(series.nunique(dropna=True))
        n_non_null = int(series.notna().sum())

        is_constant = n_unique <= 1
        # High cardinality: only flag non-numeric columns with many unique vals.
        is_high_cardinality = False
        if not pd.api.types.is_numeric_dtype(series) and n_non_null > 0:
            ratio = n_unique / n_non_null
            is_high_cardinality = ratio > _HIGH_CARDINALITY_THRESHOLD

        per_col[col] = {
            "n_unique": n_unique,
            "is_constant": is_constant,
            "is_high_cardinality": is_high_cardinality,
        }

        if is_constant:
            constant_cols.append(col)
        if is_high_cardinality:
            high_cardinality_cols.append(col)

    return {
        "per_column": per_col,
        "constant_columns": constant_cols,
        "high_cardinality_columns": high_cardinality_cols,
    }


def _safe_scalar(value: Any) -> Any:
    """Convert numpy scalar to a plain Python type; return None on failure."""
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return float(value)
    except Exception:
        return None


def _get_numerical_stats(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """
    Compute descriptive statistics for every numerical column.

    Handles edge cases:
    - Columns with fewer than ``_MIN_OBS_FOR_STATS`` valid observations.
    - Constant columns (std = 0, skewness = 0 by definition).
    - All-NaN columns.
    """
    result: dict[str, dict[str, Any]] = {}

    numerical_cols = df.select_dtypes(include=[np.number]).columns

    for col in numerical_cols:
        series = df[col].dropna()
        n_valid = len(series)

        stats: dict[str, Any] = {"n_valid": n_valid}

        if n_valid == 0:
            # All values are missing — no meaningful statistics possible.
            for key in ("mean", "median", "std", "min", "max", "q1", "q3", "iqr", "skewness"):
                stats[key] = None
            result[col] = stats
            continue

        if n_valid < _MIN_OBS_FOR_STATS:
            # Too few observations — compute only safe statistics.
            stats["mean"] = _safe_scalar(series.mean())
            stats["median"] = _safe_scalar(series.median())
            stats["min"] = _safe_scalar(series.min())
            stats["max"] = _safe_scalar(series.max())
            for key in ("std", "q1", "q3", "iqr", "skewness"):
                stats[key] = None
            result[col] = stats
            continue

        q1 = _safe_scalar(series.quantile(0.25))
        q3 = _safe_scalar(series.quantile(0.75))
        iqr = (q3 - q1) if (q1 is not None and q3 is not None) else None

        # Skewness: suppress warnings for constant / near-constant columns.
        try:
            skew_val = _safe_scalar(series.skew())
        except Exception:
            skew_val = None

        stats.update(
            {
                "mean": _safe_scalar(series.mean()),
                "median": _safe_scalar(series.median()),
                "std": _safe_scalar(series.std()),
                "min": _safe_scalar(series.min()),
                "max": _safe_scalar(series.max()),
                "q1": q1,
                "q3": q3,
                "iqr": _safe_scalar(iqr) if iqr is not None else None,
                "skewness": skew_val,
            }
        )
        result[col] = stats

    return result


def _get_outlier_stats(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """
    Detect *potential* outliers in numerical columns using the 1.5 × IQR rule.

    Values below  Q1 − 1.5 × IQR  or above  Q3 + 1.5 × IQR  are flagged.
    These are labelled "potential outliers" — they may be legitimate extreme
    values, data entry errors, or genuinely anomalous observations.

    Returns an empty dict for columns that cannot be analysed safely.
    """
    result: dict[str, dict[str, Any]] = {}

    numerical_cols = df.select_dtypes(include=[np.number]).columns

    for col in numerical_cols:
        series = df[col].dropna()
        n_valid = len(series)

        if n_valid < _MIN_OBS_FOR_STATS:
            result[col] = {
                "n_potential_outliers": None,
                "outlier_pct": None,
                "lower_fence": None,
                "upper_fence": None,
                "note": "Insufficient observations for IQR analysis.",
            }
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            # Constant or near-constant column — IQR fence is degenerate.
            result[col] = {
                "n_potential_outliers": 0,
                "outlier_pct": 0.0,
                "lower_fence": float(q1),
                "upper_fence": float(q3),
                "note": "IQR is 0; column may be constant or near-constant.",
            }
            continue

        lower_fence = q1 - _IQR_FENCE_MULTIPLIER * iqr
        upper_fence = q3 + _IQR_FENCE_MULTIPLIER * iqr

        mask = (series < lower_fence) | (series > upper_fence)
        n_outliers = int(mask.sum())
        outlier_pct = round(n_outliers / n_valid * 100, 2)

        result[col] = {
            "n_potential_outliers": n_outliers,
            "outlier_pct": outlier_pct,
            "lower_fence": round(float(lower_fence), 4),
            "upper_fence": round(float(upper_fence), 4),
            "note": (
                "Values outside fences are potential outliers and may require "
                "further investigation."
            ),
        }

    return result

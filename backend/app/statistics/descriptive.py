"""
backend/app/statistics/descriptive.py
=====================================
Pure Python & SciPy implementation for distribution analysis and missingness audits.
Computes descriptive shape parameters (skewness, kurtosis, normality tests) and
dataset completeness statistics.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats


def execute_distribution_analysis(
    df: pd.DataFrame,
    column_a: str,
    method: str = "Descriptive summary & distribution assessment",
) -> Tuple[bool, dict[str, Any], list[str], Optional[dict[str, str]]]:
    """
    Compute distribution properties, moments, quantiles, and normality test
    for numerical columns, or frequency counts, proportions, mode, and
    goodness-of-fit / entropy metrics for categorical columns.
    """
    warnings: list[str] = []

    # 1. Validate inputs
    if df is None or not isinstance(df, pd.DataFrame):
        return False, {}, warnings, {
            "code": "INVALID_INPUT",
            "message": "Input 'df' must be a valid pandas DataFrame."
        }
    if not isinstance(column_a, str):
        return False, {}, warnings, {
            "code": "INVALID_COLUMN",
            "message": "Column name must be a string."
        }
    if column_a not in df.columns:
        return False, {}, warnings, {
            "code": "INVALID_COLUMN",
            "message": f"Column '{column_a}' does not exist in dataset."
        }

    col_series = df[column_a]

    # 2. Determine variable nature: numeric vs categorical
    is_bool = pd.api.types.is_bool_dtype(col_series)
    is_cat_dtype = isinstance(col_series.dtype, pd.CategoricalDtype)
    is_numeric_type = pd.api.types.is_numeric_dtype(col_series) and not is_bool and not is_cat_dtype

    is_numeric = False
    if is_numeric_type:
        is_numeric = True
    elif not is_bool and not is_cat_dtype:
        # Check if object/string column contains predominantly numeric data
        non_null = col_series.dropna()
        if len(non_null) > 0:
            coerced = pd.to_numeric(non_null, errors="coerce")
            if float(coerced.notna().sum()) / len(non_null) >= 0.80:
                is_numeric = True

    # =========================================================================
    # A. NUMERICAL DISTRIBUTION ANALYSIS (Preserved Existing Functionality)
    # =========================================================================
    if is_numeric:
        s = pd.to_numeric(df[column_a], errors="coerce")
        valid_mask = s.notna() & ~np.isinf(s)
        clean_vals = s[valid_mask].to_numpy(dtype=float)

        n_valid = int(len(clean_vals))
        if n_valid < int(len(df)):
            warnings.append(f"Missing observations excluded: {len(df) - n_valid} rows omitted.")

        if n_valid < 3:
            return False, {}, warnings, {
                "code": "INSUFFICIENT_OBSERVATIONS",
                "message": f"Insufficient observations to assess distribution (N={n_valid}, minimum is 3)."
            }

        # Central tendency & dispersion
        mean_val = float(np.mean(clean_vals))
        med_val = float(np.median(clean_vals))
        std_val = float(np.std(clean_vals, ddof=1)) if n_valid > 1 else 0.0
        min_val = float(np.min(clean_vals))
        max_val = float(np.max(clean_vals))

        q1 = float(np.percentile(clean_vals, 25))
        q3 = float(np.percentile(clean_vals, 75))
        iqr = q3 - q1

        # Shape parameters (skewness & kurtosis)
        if std_val > 0:
            skew_val = float(stats.skew(clean_vals, bias=False)) if n_valid >= 3 else float(stats.skew(clean_vals))
            kurt_val = float(stats.kurtosis(clean_vals, bias=False)) if n_valid >= 4 else float(stats.kurtosis(clean_vals))
        else:
            skew_val = 0.0
            kurt_val = 0.0

        # Normality test
        normality_test_name = "None"
        norm_stat = None
        norm_pval = None

        if std_val == 0.0:
            warnings.append(f"Variable '{column_a}' has zero variance (constant value).")
        elif n_valid >= 8:
            try:
                stat_res, pval_res = stats.normaltest(clean_vals)
                normality_test_name = "D'Agostino-Pearson omnibus test"
                norm_stat = round(float(stat_res), 4) if not math.isnan(stat_res) else None
                norm_pval = float(pval_res) if not math.isnan(pval_res) else None
            except Exception:
                pass
        elif n_valid >= 3:
            try:
                stat_res, pval_res = stats.shapiro(clean_vals)
                normality_test_name = "Shapiro-Wilk test"
                norm_stat = round(float(stat_res), 4) if not math.isnan(stat_res) else None
                norm_pval = float(pval_res) if not math.isnan(pval_res) else None
            except Exception:
                pass

        # Qualitative shape assessment
        if abs(skew_val) < 0.5:
            symmetry = "approximately symmetric"
        elif skew_val >= 0.5:
            symmetry = "moderately right-skewed" if skew_val < 1.0 else "highly right-skewed"
        else:
            symmetry = "moderately left-skewed" if skew_val > -1.0 else "highly left-skewed"

        # Check for documented sentinel values (e.g. pdays = -1 indicating previous contact unavailable)
        is_sentinel_col = column_a == "pdays" or (getattr(df, "attrs", {}).get("sentinels", {}).get(column_a) is not None)
        sentinel_info = None
        if is_sentinel_col and -1 in clean_vals:
            sentinel_cnt = int(np.sum(clean_vals == -1))
            sentinel_pct = round((sentinel_cnt / n_valid) * 100.0, 2)
            warnings.append(
                f"Sentinel value -1 detected ({sentinel_cnt} observations, {sentinel_pct}%): "
                "represents special status (previous contact unavailable), not an ordinary negative measurement."
            )
            contacted_subset = clean_vals[clean_vals >= 0]
            sentinel_info = {
                "sentinel_value": -1,
                "count": sentinel_cnt,
                "percentage": sentinel_pct,
                "description": "Previous contact unavailable",
                "valid_contacted_count": int(len(contacted_subset)),
                "valid_contacted_mean": round(float(np.mean(contacted_subset)), 4) if len(contacted_subset) > 0 else None,
                "valid_contacted_median": round(float(np.median(contacted_subset)), 4) if len(contacted_subset) > 0 else None,
            }

        results = {
            "column": column_a,
            "variable_type": "numeric",
            "sample_size": n_valid,
            "mean": round(mean_val, 4),
            "median": round(med_val, 4),
            "std": round(std_val, 4),
            "min": round(min_val, 4),
            "max": round(max_val, 4),
            "q1": round(q1, 4),
            "q3": round(q3, 4),
            "iqr": round(iqr, 4),
            "skewness": round(skew_val, 4),
            "kurtosis": round(kurt_val, 4),
            "symmetry": symmetry,
            "statistic": norm_stat,
            "p_value": norm_pval,
            "normality_test": {
                "method": normality_test_name,
                "statistic": norm_stat,
                "p_value": norm_pval,
                "is_normally_distributed": bool(norm_pval > 0.05) if norm_pval is not None else None,
            }
        }
        if sentinel_info is not None:
            results["sentinel_observations"] = sentinel_info

        return True, results, warnings, None

    # =========================================================================
    # B. CATEGORICAL FREQUENCY DISTRIBUTION ANALYSIS
    # =========================================================================
    series = col_series.dropna()
    n_total = int(len(df))
    n_valid = int(len(series))
    n_missing = n_total - n_valid

    if n_missing > 0:
        warnings.append(f"Missing observations excluded: {n_missing} rows omitted.")

    if n_valid < 3:
        return False, {}, warnings, {
            "code": "INSUFFICIENT_OBSERVATIONS",
            "message": f"Insufficient observations to assess categorical distribution (N={n_valid}, minimum is 3)."
        }

    val_series = series.astype(str).str.strip()
    non_empty_mask = val_series != ""
    if not non_empty_mask.all():
        empty_count = int((~non_empty_mask).sum())
        warnings.append(f"Empty string observations excluded: {empty_count} rows omitted.")
        val_series = val_series[non_empty_mask]
        n_valid = int(len(val_series))
        if n_valid < 3:
            return False, {}, warnings, {
                "code": "INSUFFICIENT_OBSERVATIONS",
                "message": f"Insufficient non-empty observations to assess categorical distribution (N={n_valid}, minimum is 3)."
            }

    counts = val_series.value_counts()
    num_categories = int(len(counts))

    frequencies = {str(k): int(v) for k, v in counts.items()}
    proportions = {str(k): round(float(v) / n_valid, 4) for k, v in counts.items()}
    percentages = {str(k): round(float(v) / n_valid * 100.0, 2) for k, v in counts.items()}

    mode_val = str(counts.index[0])
    mode_count = int(counts.iloc[0])
    mode_pct = round(mode_count / n_valid * 100.0, 2)

    # Statistical test: One-sample Chi-Square goodness-of-fit against uniform null hypothesis
    chi2_stat = None
    chi2_pval = None
    entropy_val = 0.0

    if num_categories >= 2:
        try:
            chi2_res, pval_res = stats.chisquare(counts.to_numpy())
            if not math.isnan(chi2_res):
                chi2_stat = round(float(chi2_res), 4)
            if not math.isnan(pval_res):
                chi2_pval = float(pval_res)
        except Exception:
            pass

        try:
            entropy_raw = float(stats.entropy(counts.to_numpy()))
            if not math.isnan(entropy_raw):
                entropy_val = round(entropy_raw, 4)
        except Exception:
            pass
    else:
        warnings.append(f"Variable '{column_a}' has zero variability (only one distinct category: '{mode_val}').")

    is_imbalanced = bool(num_categories >= 2 and mode_pct >= 75.0)
    if is_imbalanced:
        warnings.append(
            f"Class imbalance detected: dominant category '{mode_val}' represents {mode_pct:.1f}% of observations."
        )

    results = {
        "column": column_a,
        "variable_type": "categorical",
        "sample_size": n_valid,
        "num_categories": num_categories,
        "mode": mode_val,
        "mode_frequency": mode_count,
        "mode_percentage": mode_pct,
        "frequencies": frequencies,
        "proportions": proportions,
        "percentages": percentages,
        "entropy": entropy_val,
        "is_imbalanced": is_imbalanced,
        "statistic": chi2_stat,
        "p_value": chi2_pval,
        "goodness_of_fit_test": {
            "method": "Chi-square goodness-of-fit test (uniform null)" if num_categories >= 2 else "None",
            "statistic": chi2_stat,
            "p_value": chi2_pval,
            "degrees_of_freedom": max(0, num_categories - 1),
            "is_uniformly_distributed": bool(chi2_pval > 0.05) if chi2_pval is not None else None,
        },
    }

    return True, results, warnings, None


def execute_missingness_audit(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    method: str = "Missingness pattern analysis & completeness audit",
) -> Tuple[bool, dict[str, Any], list[str], Optional[dict[str, str]]]:
    """
    Perform audit on missingness patterns across targeted columns or entire dataframe.
    """
    warnings: list[str] = []

    target_cols = [c for c in (columns or list(df.columns)) if c in df.columns]
    if not target_cols:
        return False, {}, warnings, {
            "code": "NO_VALID_COLUMNS",
            "message": "No valid columns provided for missingness audit."
        }

    sub_df = df[target_cols]
    n_rows = int(len(sub_df))
    total_cells = n_rows * len(target_cols)

    per_column = {}
    total_missing = 0
    for col in target_cols:
        col_missing = int(sub_df[col].isna().sum())
        total_missing += col_missing
        col_pct = (col_missing / n_rows * 100.0) if n_rows > 0 else 0.0
        per_column[col] = {
            "missing_count": col_missing,
            "missing_percentage": round(col_pct, 2),
            "valid_count": n_rows - col_missing,
        }

    complete_cases = int(sub_df.dropna().shape[0])
    complete_cases_pct = (complete_cases / n_rows * 100.0) if n_rows > 0 else 100.0
    overall_missing_pct = (total_missing / total_cells * 100.0) if total_cells > 0 else 0.0

    results = {
        "evaluated_columns": target_cols,
        "total_rows": n_rows,
        "total_cells": total_cells,
        "total_missing_cells": total_missing,
        "overall_missing_percentage": round(overall_missing_pct, 2),
        "complete_cases": complete_cases,
        "complete_cases_percentage": round(complete_cases_pct, 2),
        "per_column_breakdown": per_column,
        "has_missing_data": bool(total_missing > 0),
    }

    return True, results, warnings, None

"""
backend/app/statistics/sensitivity.py
=====================================
Pure Python & SciPy implementation for IQR-based outlier sensitivity analysis.
Compares baseline full-sample metrics against outlier-trimmed metrics to quantify
the influence of extreme values without mutating the underlying dataset.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple
import numpy as np
import pandas as pd


def execute_outlier_sensitivity(
    df: pd.DataFrame,
    column_a: str,
    method: str = "IQR-based sensitivity analysis",
) -> Tuple[bool, dict[str, Any], list[str], Optional[dict[str, str]]]:
    """
    Perform IQR-based sensitivity analysis on a numerical column.

    Compares descriptive statistics on the complete valid dataset vs.
    the subset after excluding observations outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].

    Returns
    -------
    (success, results_dict, warnings_list, error_dict)
    """
    warnings: list[str] = []

    # 1. Validate column existence
    if column_a not in df.columns:
        return False, {}, warnings, {
            "code": "INVALID_COLUMN",
            "message": f"Column '{column_a}' does not exist in dataset."
        }

    # 2. Extract numeric observations
    s = pd.to_numeric(df[column_a], errors="coerce")
    valid_mask = s.notna() & ~np.isinf(s)
    clean_vals = s[valid_mask].to_numpy(dtype=float)

    n_original = int(len(clean_vals))
    if n_original < int(len(df)):
        warnings.append(f"Missing observations excluded: {len(df) - n_original} rows omitted.")

    if n_original < 4:
        return False, {}, warnings, {
            "code": "INSUFFICIENT_OBSERVATIONS",
            "message": f"Insufficient observations for quartile calculations (N={n_original}, minimum required is 4)."
        }

    # 3. Baseline metrics
    base_mean = float(np.mean(clean_vals))
    base_median = float(np.median(clean_vals))
    base_std = float(np.std(clean_vals, ddof=1)) if n_original > 1 else 0.0
    base_min = float(np.min(clean_vals))
    base_max = float(np.max(clean_vals))

    # Quartiles and IQR
    q1 = float(np.percentile(clean_vals, 25))
    q3 = float(np.percentile(clean_vals, 75))
    iqr = q3 - q1

    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    # 4. Outlier identification
    outlier_mask = (clean_vals < lower_fence) | (clean_vals > upper_fence)
    outlier_count = int(np.sum(outlier_mask))
    outlier_pct = float(outlier_count / n_original * 100.0)

    filtered_vals = clean_vals[~outlier_mask]
    n_filtered = int(len(filtered_vals))

    if n_filtered > 0:
        filt_mean = float(np.mean(filtered_vals))
        filt_median = float(np.median(filtered_vals))
        filt_std = float(np.std(filtered_vals, ddof=1)) if n_filtered > 1 else 0.0
        filt_min = float(np.min(filtered_vals))
        filt_max = float(np.max(filtered_vals))
    else:
        filt_mean = base_mean
        filt_median = base_median
        filt_std = base_std
        filt_min = base_min
        filt_max = base_max

    # 5. Sensitivity shifts (deltas)
    mean_delta = filt_mean - base_mean
    mean_delta_pct = (mean_delta / abs(base_mean) * 100.0) if abs(base_mean) > 1e-9 else 0.0
    std_delta = filt_std - base_std

    # Outlier values preview (up to 10)
    outliers_list = [round(float(v), 4) for v in clean_vals[outlier_mask][:10]]

    results = {
        "column": column_a,
        "sample_size": n_original,
        "original_sample_size": n_original,
        "filtered_sample_size": n_filtered,
        "outlier_count": outlier_count,
        "outlier_percentage": round(outlier_pct, 2),
        "q1": round(q1, 4),
        "q3": round(q3, 4),
        "iqr": round(iqr, 4),
        "lower_fence": round(lower_fence, 4),
        "upper_fence": round(upper_fence, 4),
        "baseline_metrics": {
            "mean": round(base_mean, 4),
            "median": round(base_median, 4),
            "std": round(base_std, 4),
            "min": round(base_min, 4),
            "max": round(base_max, 4),
        },
        "filtered_metrics": {
            "mean": round(filt_mean, 4),
            "median": round(filt_median, 4),
            "std": round(filt_std, 4),
            "min": round(filt_min, 4),
            "max": round(filt_max, 4),
        },
        "mean_shift": round(mean_delta, 4),
        "mean_shift_pct": round(mean_delta_pct, 2),
        "std_shift": round(std_delta, 4),
        "outliers_sample": outliers_list,
        "substantial_influence": bool(abs(mean_delta_pct) >= 5.0 and outlier_count > 0),
    }

    return True, results, warnings, None

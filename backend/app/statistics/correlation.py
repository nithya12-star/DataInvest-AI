"""
backend/app/statistics/correlation.py
======================================
Pure Python & SciPy implementation for bivariate numerical correlation analysis.
Supports Pearson and Spearman rank correlation with complete edge-case handling.
"""

from __future__ import annotations

import math
from typing import Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats


def execute_correlation(
    df: pd.DataFrame,
    column_a: str,
    column_b: str,
    method: str = "Pearson correlation",
) -> Tuple[bool, dict[str, Any], list[str], Optional[dict[str, str]]]:
    """
    Execute Pearson or Spearman correlation on pairwise complete observations.

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
    if column_b not in df.columns:
        return False, {}, warnings, {
            "code": "INVALID_COLUMN",
            "message": f"Column '{column_b}' does not exist in dataset."
        }

    # 2. Extract and cast to numeric
    s_a = pd.to_numeric(df[column_a], errors="coerce")
    s_b = pd.to_numeric(df[column_b], errors="coerce")

    # Pairwise complete cases
    valid_mask = s_a.notna() & s_b.notna() & ~np.isinf(s_a) & ~np.isinf(s_b)
    clean_a = s_a[valid_mask].to_numpy(dtype=float)
    clean_b = s_b[valid_mask].to_numpy(dtype=float)

    n_valid = int(len(clean_a))
    original_n = int(len(df))

    if n_valid < original_n:
        missing_count = original_n - n_valid
        warnings.append(f"Pairwise missing observations excluded: {missing_count} rows omitted.")

    # 3. Validate sample size
    if n_valid < 3:
        return False, {}, warnings, {
            "code": "INSUFFICIENT_OBSERVATIONS",
            "message": f"Insufficient valid paired observations (N={n_valid}, minimum required is 3)."
        }

    # 4. Check for constant columns (zero variance)
    std_a = np.std(clean_a, ddof=1)
    std_b = np.std(clean_b, ddof=1)

    if std_a == 0.0:
        return False, {}, warnings, {
            "code": "CONSTANT_VARIABLE",
            "message": f"Statistical analysis cannot be performed because variable '{column_a}' is constant."
        }
    if std_b == 0.0:
        return False, {}, warnings, {
            "code": "CONSTANT_VARIABLE",
            "message": f"Statistical analysis cannot be performed because variable '{column_b}' is constant."
        }

    # 5. Execute SciPy correlation
    method_clean = method.lower()
    is_spearman = ("spearman" in method_clean) and ("pearson" not in method_clean)
    ci_low = None
    ci_high = None
    try:
        if is_spearman:
            res = stats.spearmanr(clean_a, clean_b)
            r_val = float(res.statistic)
            p_val = float(res.pvalue)
            actual_method = "Spearman rank correlation"
            # Fisher z-transformation approximation for Spearman 95% CI
            if n_valid > 3 and not (math.isnan(r_val) or math.isinf(r_val)):
                r_clipped = max(-0.999999, min(0.999999, r_val))
                z = np.arctanh(r_clipped)
                se = 1.0 / np.sqrt(max(1, n_valid - 3))
                ci_low = float(np.tanh(z - 1.95996 * se))
                ci_high = float(np.tanh(z + 1.95996 * se))
        else:
            res = stats.pearsonr(clean_a, clean_b)
            r_val = float(res.statistic)
            p_val = float(res.pvalue)
            actual_method = "Pearson correlation"
            if n_valid > 3 and not (math.isnan(r_val) or math.isinf(r_val)):
                try:
                    ci = res.confidence_interval(confidence_level=0.95)
                    ci_low = float(ci.low)
                    ci_high = float(ci.high)
                except Exception:
                    r_clipped = max(-0.999999, min(0.999999, r_val))
                    z = np.arctanh(r_clipped)
                    se = 1.0 / np.sqrt(max(1, n_valid - 3))
                    ci_low = float(np.tanh(z - 1.95996 * se))
                    ci_high = float(np.tanh(z + 1.95996 * se))
    except Exception as e:
        return False, {}, warnings, {
            "code": "CALCULATION_ERROR",
            "message": f"Statistical calculation failed: {str(e)}"
        }

    # 6. Sanitize NaN or Inf outputs
    if math.isnan(r_val) or math.isinf(r_val) or math.isnan(p_val) or math.isinf(p_val):
        return False, {}, warnings, {
            "code": "DEGENERATE_OUTPUT",
            "message": "Calculated correlation produced a non-finite statistic or p-value."
        }

    abs_r = abs(r_val)
    if abs_r >= 0.7:
        strength = "strong"
    elif abs_r >= 0.4:
        strength = "moderate"
    elif abs_r >= 0.1:
        strength = "weak"
    else:
        strength = "negligible"

    r_sq = round(float(r_val ** 2), 6)

    results = {
        "method": actual_method,
        "actual_method": actual_method,
        "statistic": round(r_val, 6),
        "correlation_coefficient": round(r_val, 6),
        "p_value": float(p_val),
        "p_value_formatted": f"{p_val:.4e}" if p_val < 0.0001 else f"{p_val:.4f}",
        "sample_size": n_valid,
        "degrees_of_freedom": n_valid - 2,
        "p_value_below_0_05": bool(p_val < 0.05),
        "p_value_below_0_01": bool(p_val < 0.01),
        "association_strength": strength,
        "direction": "positive" if r_val > 0 else "negative" if r_val < 0 else "none",
        "effect_size": round(abs_r, 4),
        "effect_size_type": "Correlation coefficient (|r|)",
        "r_squared": r_sq,
        "ci_95_low": round(ci_low, 4) if ci_low is not None and not math.isnan(ci_low) else None,
        "ci_95_high": round(ci_high, 4) if ci_high is not None and not math.isnan(ci_high) else None,
        "confidence_interval": {
            "low": round(ci_low, 4) if ci_low is not None and not math.isnan(ci_low) else None,
            "high": round(ci_high, 4) if ci_high is not None and not math.isnan(ci_high) else None,
            "confidence_level": 0.95,
        } if ci_low is not None else None,
        "column_a": column_a,
        "column_b": column_b,
        "mean_a": round(float(np.mean(clean_a)), 4),
        "mean_b": round(float(np.mean(clean_b)), 4),
        "std_a": round(float(std_a), 4),
        "std_b": round(float(std_b), 4),
    }

    return True, results, warnings, None

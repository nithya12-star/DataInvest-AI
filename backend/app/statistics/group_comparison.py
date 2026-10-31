"""
backend/app/statistics/group_comparison.py
===========================================
Pure Python & SciPy implementation for Two-Group and Multi-Group comparisons.
Supports Welch's t-test, Mann-Whitney U, One-way ANOVA, and Kruskal-Wallis.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats


def execute_two_group_comparison(
    df: pd.DataFrame,
    group_column: str,
    value_column: str,
    method: str = "Welch's t-test",
) -> Tuple[bool, dict[str, Any], list[str], Optional[dict[str, str]]]:
    """
    Execute Welch's t-test (equal_var=False) or Mann-Whitney U test comparing
    a numerical variable across 2 distinct categories.
    """
    warnings: list[str] = []

    # 1. Validate columns
    if group_column not in df.columns:
        return False, {}, warnings, {"code": "INVALID_COLUMN", "message": f"Group column '{group_column}' does not exist."}
    if value_column not in df.columns:
        return False, {}, warnings, {"code": "INVALID_COLUMN", "message": f"Value column '{value_column}' does not exist."}

    # 2. Extract valid pairs
    s_val = pd.to_numeric(df[value_column], errors="coerce")
    s_grp = df[group_column].astype(str)

    valid_mask = s_val.notna() & df[group_column].notna() & ~np.isinf(s_val)
    clean_val = s_val[valid_mask]
    clean_grp = s_grp[valid_mask]

    n_valid = int(len(clean_val))
    if n_valid < len(df):
        warnings.append(f"Missing observations excluded: {len(df) - n_valid} rows omitted.")

    # 3. Identify distinct groups
    unique_groups = list(clean_grp.unique())
    if len(unique_groups) < 2:
        return False, {}, warnings, {
            "code": "INSUFFICIENT_GROUPS",
            "message": f"Grouping column '{group_column}' contains fewer than 2 distinct valid groups (found {len(unique_groups)})."
        }

    # Select the two primary groups if more exist
    g1_name, g2_name = unique_groups[0], unique_groups[1]
    g1_vals = clean_val[clean_grp == g1_name].to_numpy(dtype=float)
    g2_vals = clean_val[clean_grp == g2_name].to_numpy(dtype=float)

    if len(unique_groups) > 2:
        warnings.append(f"Grouping column has {len(unique_groups)} groups; testing primary pair '{g1_name}' vs '{g2_name}'.")

    n1, n2 = len(g1_vals), len(g2_vals)
    if n1 < 2 or n2 < 2:
        return False, {}, warnings, {
            "code": "INSUFFICIENT_OBSERVATIONS",
            "message": f"Insufficient observations in groups (group '{g1_name}': n={n1}, group '{g2_name}': n={n2}, minimum required is 2 per group)."
        }

    # 4. Group Descriptives
    mean1, mean2 = float(np.mean(g1_vals)), float(np.mean(g2_vals))
    med1, med2 = float(np.median(g1_vals)), float(np.median(g2_vals))
    std1 = float(np.std(g1_vals, ddof=1)) if n1 > 1 else 0.0
    std2 = float(np.std(g2_vals, ddof=1)) if n2 > 1 else 0.0

    if std1 == 0.0 and std2 == 0.0 and mean1 == mean2:
        return False, {}, warnings, {
            "code": "CONSTANT_VARIABLE",
            "message": "Both groups contain constant identical values; no variation to test."
        }

    # 5. Execute Test
    method_clean = method.lower()
    is_mann_whitney = ("mann" in method_clean or "whitney" in method_clean) and ("t-test" not in method_clean and "ttest" not in method_clean)
    ci_low = None
    ci_high = None
    effect_size = None
    effect_size_type = None
    try:
        if is_mann_whitney:
            res = stats.mannwhitneyu(g1_vals, g2_vals, alternative="two-sided")
            stat_val = float(res.statistic)
            p_val = float(res.pvalue)
            actual_method = "Mann-Whitney U test"
            # Rank-biserial correlation: r = 1 - (2*U)/(n1*n2)
            if n1 * n2 > 0:
                u_val = stat_val
                r_rb = 1.0 - (2.0 * u_val) / (n1 * n2)
                effect_size = round(float(r_rb), 4)
                effect_size_type = "Rank-biserial correlation"
        else:
            res = stats.ttest_ind(g1_vals, g2_vals, equal_var=False)
            stat_val = float(res.statistic)
            p_val = float(res.pvalue)
            actual_method = "Welch's t-test"
            # Confidence interval for mean difference
            try:
                ci = res.confidence_interval(confidence_level=0.95)
                ci_low = float(ci.low)
                ci_high = float(ci.high)
            except Exception:
                # Welch-Satterthwaite approximation fallback
                se_diff = np.sqrt((std1 ** 2) / n1 + (std2 ** 2) / n2)
                ci_low = float((mean1 - mean2) - 1.95996 * se_diff)
                ci_high = float((mean1 - mean2) + 1.95996 * se_diff)
            # Cohen's d (pooled standard deviation)
            pooled_denom = ((n1 - 1) * (std1 ** 2) + (n2 - 1) * (std2 ** 2)) / max(1, (n1 + n2 - 2))
            s_pooled = np.sqrt(pooled_denom) if pooled_denom > 0 else 0.0
            if s_pooled > 0:
                d_val = (mean1 - mean2) / s_pooled
                effect_size = round(float(d_val), 4)
                effect_size_type = "Cohen's d"
    except Exception as e:
        return False, {}, warnings, {"code": "CALCULATION_ERROR", "message": f"Test calculation failed: {str(e)}"}

    if math.isnan(stat_val) or math.isnan(p_val):
        return False, {}, warnings, {"code": "DEGENERATE_OUTPUT", "message": "Test resulted in NaN statistic or p-value."}

    results = {
        "method": actual_method,
        "actual_method": actual_method,
        "statistic": round(stat_val, 6),
        "test_statistic": round(stat_val, 6),
        "p_value": float(p_val),
        "p_value_formatted": f"{p_val:.4e}" if p_val < 0.0001 else f"{p_val:.4f}",
        "sample_size": n1 + n2,
        "group_column": group_column,
        "value_column": value_column,
        "group_1": {
            "name": str(g1_name),
            "n": n1,
            "mean": round(mean1, 4),
            "median": round(med1, 4),
            "std": round(std1, 4),
        },
        "group_2": {
            "name": str(g2_name),
            "n": n2,
            "mean": round(mean2, 4),
            "median": round(med2, 4),
            "std": round(std2, 4),
        },
        "mean_difference": round(mean1 - mean2, 4),
        "effect_size": effect_size,
        "effect_size_type": effect_size_type,
        "ci_95_low": round(ci_low, 4) if ci_low is not None and not math.isnan(ci_low) else None,
        "ci_95_high": round(ci_high, 4) if ci_high is not None and not math.isnan(ci_high) else None,
        "confidence_interval": {
            "low": round(ci_low, 4) if ci_low is not None and not math.isnan(ci_low) else None,
            "high": round(ci_high, 4) if ci_high is not None and not math.isnan(ci_high) else None,
            "confidence_level": 0.95,
        } if ci_low is not None else None,
        "p_value_below_0_05": bool(p_val < 0.05),
        "p_value_below_0_01": bool(p_val < 0.01),
    }

    return True, results, warnings, None


def execute_multi_group_comparison(
    df: pd.DataFrame,
    group_column: str,
    value_column: str,
    method: str = "One-way ANOVA",
) -> Tuple[bool, dict[str, Any], list[str], Optional[dict[str, str]]]:
    """
    Execute One-way ANOVA or Kruskal-Wallis test comparing a numerical variable
    across multiple distinct categories.
    """
    warnings: list[str] = []

    # 1. Validate columns
    if group_column not in df.columns:
        return False, {}, warnings, {"code": "INVALID_COLUMN", "message": f"Group column '{group_column}' does not exist."}
    if value_column not in df.columns:
        return False, {}, warnings, {"code": "INVALID_COLUMN", "message": f"Value column '{value_column}' does not exist."}

    # 2. Extract clean pairs
    s_val = pd.to_numeric(df[value_column], errors="coerce")
    s_grp = df[group_column].astype(str)

    valid_mask = s_val.notna() & df[group_column].notna() & ~np.isinf(s_val)
    clean_val = s_val[valid_mask]
    clean_grp = s_grp[valid_mask]

    n_valid = int(len(clean_val))
    if n_valid < len(df):
        warnings.append(f"Missing observations excluded: {len(df) - n_valid} rows omitted.")

    # 3. Partition by group
    grouped_data = {}
    for grp_name, group_vals in clean_val.groupby(clean_grp):
        arr = group_vals.to_numpy(dtype=float)
        if len(arr) >= 2:  # Only groups with >= 2 observations
            grouped_data[str(grp_name)] = arr

    k = len(grouped_data)
    if k < 2:
        return False, {}, warnings, {
            "code": "INSUFFICIENT_GROUPS",
            "message": f"Grouping column '{group_column}' has fewer than 2 groups with adequate observations (found {k})."
        }

    # 4. Group Descriptives
    group_stats_list = []
    arrays_to_test = []
    for gname, gvals in grouped_data.items():
        arrays_to_test.append(gvals)
        group_stats_list.append({
            "name": gname,
            "n": int(len(gvals)),
            "mean": round(float(np.mean(gvals)), 4),
            "median": round(float(np.median(gvals)), 4),
            "std": round(float(np.std(gvals, ddof=1)), 4),
        })

    # 5. Execute Test
    method_clean = method.lower()
    is_kruskal = ("kruskal" in method_clean) and ("anova" not in method_clean)
    try:
        if is_kruskal:
            res = stats.kruskal(*arrays_to_test)
            stat_val = float(res.statistic)
            p_val = float(res.pvalue)
            actual_method = "Kruskal-Wallis test"
            stat_name = "H_statistic"
        else:
            res = stats.f_oneway(*arrays_to_test)
            stat_val = float(res.statistic)
            p_val = float(res.pvalue)
            actual_method = "One-way ANOVA"
            stat_name = "F_statistic"
    except Exception as e:
        return False, {}, warnings, {"code": "CALCULATION_ERROR", "message": f"Multi-group test failed: {str(e)}"}

    if math.isnan(stat_val) or math.isnan(p_val):
        return False, {}, warnings, {"code": "DEGENERATE_OUTPUT", "message": "Test resulted in NaN statistic or p-value."}

    total_tested_n = sum(len(a) for a in arrays_to_test)
    effect_size = None
    effect_size_type = None

    if is_kruskal:
        # Epsilon-squared: ε² = (H - k + 1) / (N - k)
        if total_tested_n > k:
            eps2 = max(0.0, min(1.0, (stat_val - k + 1.0) / (total_tested_n - k)))
            effect_size = round(float(eps2), 4)
            effect_size_type = "Epsilon-squared (ε²)"
    else:
        # Eta-squared for ANOVA: η² = SS_between / SS_total
        all_vals = np.concatenate(arrays_to_test)
        grand_mean = float(np.mean(all_vals))
        ss_total = float(np.sum((all_vals - grand_mean) ** 2))
        ss_between = float(sum(len(arr) * ((np.mean(arr) - grand_mean) ** 2) for arr in arrays_to_test))
        if ss_total > 0:
            eta2 = max(0.0, min(1.0, ss_between / ss_total))
            effect_size = round(float(eta2), 4)
            effect_size_type = "Eta-squared (η²)"

    results = {
        "method": actual_method,
        "actual_method": actual_method,
        "statistic": round(stat_val, 6),
        stat_name: round(stat_val, 6),
        "p_value": float(p_val),
        "p_value_formatted": f"{p_val:.4e}" if p_val < 0.0001 else f"{p_val:.4f}",
        "sample_size": total_tested_n,
        "num_groups": k,
        "degrees_of_freedom_between": k - 1,
        "degrees_of_freedom_within": total_tested_n - k,
        "group_column": group_column,
        "value_column": value_column,
        "groups": group_stats_list,
        "effect_size": effect_size,
        "effect_size_type": effect_size_type,
        "p_value_below_0_05": bool(p_val < 0.05),
        "p_value_below_0_01": bool(p_val < 0.01),
    }

    return True, results, warnings, None

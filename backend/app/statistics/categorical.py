"""
backend/app/statistics/categorical.py
======================================
Pure Python & SciPy implementation for categorical association analysis.
Computes Chi-square test of independence and Cramér's V effect size.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats


def execute_categorical_association(
    df: pd.DataFrame,
    column_a: str,
    column_b: str,
    method: str = "Chi-square test of independence",
) -> Tuple[bool, dict[str, Any], list[str], Optional[dict[str, str]]]:
    """
    Execute Chi-square test of independence and calculate Cramér's V effect size.
    """
    warnings: list[str] = []

    # 1. Validate columns
    if column_a not in df.columns:
        return False, {}, warnings, {"code": "INVALID_COLUMN", "message": f"Column '{column_a}' does not exist."}
    if column_b not in df.columns:
        return False, {}, warnings, {"code": "INVALID_COLUMN", "message": f"Column '{column_b}' does not exist."}

    # 2. Extract non-null pairs
    valid_mask = df[column_a].notna() & df[column_b].notna()
    clean_a = df.loc[valid_mask, column_a].astype(str)
    clean_b = df.loc[valid_mask, column_b].astype(str)

    n_valid = int(len(clean_a))
    if n_valid < len(df):
        warnings.append(f"Missing observations excluded: {len(df) - n_valid} rows omitted.")

    if n_valid < 5:
        return False, {}, warnings, {
            "code": "INSUFFICIENT_OBSERVATIONS",
            "message": f"Insufficient valid observation pairs for contingency analysis (N={n_valid}, minimum is 5)."
        }

    # 3. Build contingency table
    contingency = pd.crosstab(clean_a, clean_b)
    rows, cols = contingency.shape

    if rows < 2 or cols < 2:
        return False, {}, warnings, {
            "code": "INSUFFICIENT_DIMENSIONS",
            "message": f"Contingency table must be at least 2x2 (found {rows}x{cols})."
        }

    # 4. Execute Chi-Square
    try:
        chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency.to_numpy())
        chi2_val = float(chi2_stat)
        p_val = float(p_val)
        dof_val = int(dof)
    except Exception as e:
        return False, {}, warnings, {"code": "CALCULATION_ERROR", "message": f"Chi-square calculation failed: {str(e)}"}

    if math.isnan(chi2_val) or math.isnan(p_val):
        return False, {}, warnings, {"code": "DEGENERATE_OUTPUT", "message": "Chi-square resulted in NaN values."}

    # 5. Compute Cramér's V: sqrt(chi2 / (N * (min(r, c) - 1)))
    min_dim = min(rows, cols)
    if min_dim > 1 and n_valid > 0:
        cramer_v = math.sqrt(chi2_val / (n_valid * (min_dim - 1)))
        cramer_v = min(max(cramer_v, 0.0), 1.0)  # Bound to [0, 1]
    else:
        cramer_v = 0.0

    # Format contingency table for JSON
    table_dict = {
        "row_labels": [str(r) for r in contingency.index],
        "col_labels": [str(c) for c in contingency.columns],
        "matrix": [[int(v) for v in row] for row in contingency.to_numpy()],
    }

    # Check expected frequencies according to Cochran's rule (1954):
    # 1. No expected frequency should be < 1.0.
    # 2. No more than 20% of expected frequencies should be < 5.0.
    min_expected = float(np.min(expected)) if expected.size > 0 else 0.0
    cells_below_5 = int(np.sum(expected < 5))
    cells_below_1 = int(np.sum(expected < 1))
    small_expected_pct = (float(cells_below_5) / expected.size * 100.0) if expected.size > 0 else 0.0

    if min_expected < 1.0:
        warnings.append(
            f"Expected cell frequency < 1.0 detected (minimum expected count = {min_expected:.2f}); "
            f"Cochran's condition violated and Chi-square asymptotic approximation is unreliable."
        )
    if small_expected_pct > 20.0:
        warnings.append(
            f"{small_expected_pct:.1f}% of cells have expected count < 5 (exceeds 20% Cochran threshold); "
            f"interpret chi-square with caution."
        )

    results = {
        "statistic": round(chi2_val, 4),
        "chi2_statistic": round(chi2_val, 4),
        "column_a": column_a,
        "column_b": column_b,
        "p_value": float(p_val),
        "p_value_formatted": f"{p_val:.4e}" if p_val < 0.0001 else f"{p_val:.4f}",
        "degrees_of_freedom": dof_val,
        "sample_size": n_valid,
        "cramer_v": round(cramer_v, 4),
        "cramer_v_interpretation": (
            "strong association" if cramer_v >= 0.3 else
            "moderate association" if cramer_v >= 0.15 else
            "weak association" if cramer_v >= 0.05 else "negligible association"
        ),
        "table_dimensions": f"{rows}x{cols}",
        "contingency_table": table_dict,
        "expected_cell_counts": {
            "min_expected": round(min_expected, 4),
            "cells_below_5_count": cells_below_5,
            "cells_below_1_count": cells_below_1,
            "cells_below_5_pct": round(small_expected_pct, 2),
            "cochran_criterion_satisfied": bool(min_expected >= 1.0 and small_expected_pct <= 20.0),
        },
        "p_value_below_0_05": bool(p_val < 0.05),
        "p_value_below_0_01": bool(p_val < 0.01),
    }

    return True, results, warnings, None

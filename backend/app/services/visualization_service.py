"""
backend/app/services/visualization_service.py
=============================================
Phase 4: Dynamic Visualization & Programmatic Visual Evidence Service.

Purpose:
    Generate dynamic, publication-grade visualizations from actual dataset observations
    and SciPy statistical outputs using existing environment libraries (Altair / Vega-Lite).
    Produces defensible programmatic visual diagnostics without ever claiming or simulating
    visual image perception by an LLM.

Key Principles:
    - Pure empirical derivation from Pandas/NumPy/Altair.
    - Explicit limitation notice: No vision-capable model is claimed.
    - Zero fabrication of visual evidence.
    - Robust error isolation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

try:
    import altair as alt
    alt.data_transformers.disable_max_rows()
    _HAS_ALTAIR = True
except ImportError:
    _HAS_ALTAIR = False

from app.schemas.visualization_schema import (
    FigureType,
    VisualDiagnostics,
    VisualEvidence,
)

logger = logging.getLogger("datainvest.visualization_service")

_PRIMARY_COLOR = "#18C8FF"      # Cyan brand color
_SECONDARY_COLOR = "#4F8EF7"    # Navy / blue
_ACCENT_COLOR = "#F7944F"       # Orange accent
_DANGER_COLOR = "#E05252"       # Red outlier/warning


class VisualizationService:
    """Service generating dynamic visual evidence for analytical findings."""

    def generate_finding_visualization(
        self,
        task_id: str,
        analysis_type: str,
        method: str,
        columns: List[str],
        df: pd.DataFrame,
        results: Dict[str, Any],
    ) -> Optional[VisualEvidence]:
        """
        Dynamically construct chart specification and programmatic visual diagnostics
        associated with an empirical finding.
        """
        if df is None or not isinstance(df, pd.DataFrame) or df.empty or not columns:
            return None

        clean_type = analysis_type.strip().lower()

        try:
            if clean_type == "numerical_vs_numerical":
                return self._plot_correlation_scatter(task_id, method, columns, df, results)
            elif clean_type == "two_groups_vs_numerical":
                return self._plot_two_group_box(task_id, method, columns, df, results)
            elif clean_type == "multiple_groups_vs_numerical":
                return self._plot_multi_group_box(task_id, method, columns, df, results)
            elif clean_type == "categorical_vs_categorical":
                return self._plot_categorical_bar(task_id, method, columns, df, results)
            elif clean_type in ("outlier_sensitivity", "distribution_analysis"):
                return self._plot_distribution_histogram(task_id, method, columns, df, results)
            else:
                return self._plot_generic_distribution(task_id, method, columns, df, results)
        except Exception as exc:
            logger.warning("Failed to generate dynamic visualization for task %s: %s", task_id, exc)
            return None

    # ------------------------------------------------------------------
    # 1. SCATTER PLOT (Numerical vs Numerical)
    # ------------------------------------------------------------------
    def _plot_correlation_scatter(
        self,
        task_id: str,
        method: str,
        columns: List[str],
        df: pd.DataFrame,
        results: Dict[str, Any],
    ) -> VisualEvidence:
        res_a = results.get("column_a") if results else None
        res_b = results.get("column_b") if results else None
        col_a = (res_a if res_a and res_a in df.columns else None) or (columns[0] if len(columns) > 0 else "")
        col_b = (res_b if res_b and res_b in df.columns else None) or (columns[1] if len(columns) > 1 else (columns[0] if len(columns) > 0 else ""))

        s_a = pd.to_numeric(df[col_a], errors="coerce")
        s_b = pd.to_numeric(df[col_b], errors="coerce")
        valid_mask = s_a.notna() & s_b.notna() & ~np.isinf(s_a) & ~np.isinf(s_b)

        arr_a = s_a[valid_mask].to_numpy(dtype=float)
        arr_b = s_b[valid_mask].to_numpy(dtype=float)
        n = len(arr_a)

        method_clean = (method or "").lower()
        is_spearman = ("spearman" in method_clean) and ("pearson" not in method_clean)

        outlier_count = 0
        notes: List[str] = [f"Plotted {n} pairwise valid observations for '{col_a}' vs '{col_b}'."]

        if n >= 3:
            if is_spearman:
                rank_a = stats.rankdata(arr_a)
                rank_b = stats.rankdata(arr_b)
                slope, intercept, r_val, p_val, std_err = stats.linregress(rank_a, rank_b)
                notes.append("Rank-transformed scatter plot for Spearman rank evaluation (monotonic fit).")
                notes.append(f"Rank-space fit slope: {slope:.4f}, intercept: {intercept:.4f}.")
            else:
                slope, intercept, r_val, p_val, std_err = stats.linregress(arr_a, arr_b)
                residuals = arr_b - (slope * arr_a + intercept)
                res_std = np.std(residuals) if len(residuals) > 1 else 0.0
                if res_std > 0:
                    z_res = np.abs(residuals / res_std)
                    outlier_count = int(np.sum(z_res > 3.0))
                    if outlier_count > 0:
                        notes.append(f"Programmatic diagnostics detected {outlier_count} potential leverage/residual outlier points (|z| > 3.0).")
                notes.append(f"Linear fit slope: {slope:.4f}, intercept: {intercept:.4f}.")

        title = f"Spearman Rank Correlation: {col_a} vs {col_b}" if is_spearman else f"Correlation: {col_a} vs {col_b} ({method})"

        # Build declarative chart spec
        fig_spec: Dict[str, Any] = {}
        if _HAS_ALTAIR and n > 0:
            if is_spearman:
                rank_col_a = f"Rank_{col_a}"
                rank_col_b = f"Rank_{col_b}"
                plot_df = pd.DataFrame({rank_col_a: rank_a, rank_col_b: rank_b})
                chart = alt.Chart(plot_df, title=title).mark_circle(size=60, color="#8b5cf6", opacity=0.75).encode(
                    x=alt.X(rank_col_a, type="quantitative", title=f"Rank({col_a})"),
                    y=alt.Y(rank_col_b, type="quantitative", title=f"Rank({col_b})"),
                )
            else:
                plot_df = pd.DataFrame({col_a: arr_a, col_b: arr_b})
                chart = alt.Chart(plot_df, title=title).mark_circle(size=60, color=_PRIMARY_COLOR, opacity=0.75).encode(
                    x=alt.X(col_a, type="quantitative"),
                    y=alt.Y(col_b, type="quantitative"),
                )
            fig_spec = chart.to_dict()

        diagnostics = VisualDiagnostics(
            outlier_points_count=outlier_count,
            linearity_indicator="Linear" if outlier_count <= 2 else "Non-linear / contaminated",
            visual_notes=notes,
            vision_model_used=False,
            visual_inspection_claimed=False,
        )

        return VisualEvidence(
            graph_id=f"graph_{task_id}_scatter",
            task_id=task_id,
            figure_type="scatter",
            title=title,
            x_label=col_a,
            y_label=col_b,
            columns=[col_a, col_b],
            method=method,
            data_summary={"sample_size": n, "bivariate_outliers": outlier_count},
            figure_spec=fig_spec,
            visual_diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------
    # 2. TWO-GROUP BOX PLOT (Two Groups vs Numerical)
    # ------------------------------------------------------------------
    def _plot_two_group_box(
        self,
        task_id: str,
        method: str,
        columns: List[str],
        df: pd.DataFrame,
        results: Dict[str, Any],
    ) -> VisualEvidence:
        res_grp = results.get("group_column") if results else None
        res_val = results.get("value_column") if results else None
        grp_col = (res_grp if res_grp and res_grp in df.columns else None) or (columns[0] if len(columns) > 0 else "")
        val_col = (res_val if res_val and res_val in df.columns else None) or (columns[1] if len(columns) > 1 else (columns[0] if len(columns) > 0 else ""))

        s_val = pd.to_numeric(df[val_col], errors="coerce")
        s_grp = df[grp_col].astype(str)
        valid_mask = s_val.notna() & df[grp_col].notna() & ~np.isinf(s_val)

        clean_val = s_val[valid_mask]
        clean_grp = s_grp[valid_mask]

        notes: List[str] = []
        outlier_count = 0
        iqrs: List[tuple[float, float]] = []

        unique_groups = list(clean_grp.unique())[:2]
        for gname in unique_groups:
            gvals = clean_val[clean_grp == gname].to_numpy(dtype=float)
            if len(gvals) >= 4:
                q1 = float(np.percentile(gvals, 25))
                q3 = float(np.percentile(gvals, 75))
            else:
                q1, q3 = float(np.min(gvals)), float(np.max(gvals))
            iqr = q3 - q1
            iqrs.append((q1, q3))
            f_low, f_high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outlier_count += int(np.sum((gvals < f_low) | (gvals > f_high)))

        iqr_overlap = None
        if len(iqrs) == 2:
            overlap_min = max(iqrs[0][0], iqrs[1][0])
            overlap_max = min(iqrs[0][1], iqrs[1][1])
            if overlap_max > overlap_min:
                span = max(iqrs[0][1], iqrs[1][1]) - min(iqrs[0][0], iqrs[1][0])
                iqr_overlap = round(float((overlap_max - overlap_min) / max(span, 1e-6)), 3)
                notes.append(f"Interquartile overlap ratio between groups: {iqr_overlap:.2f}.")
            else:
                iqr_overlap = 0.0
                notes.append("Zero visual overlap between group IQRs (strong separation).")

        title = f"Group Comparison: {val_col} by {grp_col} ({method})"

        fig_spec: Dict[str, Any] = {}
        if _HAS_ALTAIR:
            plot_df = pd.DataFrame({grp_col: clean_grp, val_col: clean_val})
            plot_df = plot_df[plot_df[grp_col].isin(unique_groups)]
            chart = alt.Chart(plot_df, title=title).mark_boxplot().encode(
                x=alt.X(grp_col, type="nominal"),
                y=alt.Y(val_col, type="quantitative"),
                color=alt.Color(grp_col, type="nominal", legend=None),
            )
            fig_spec = chart.to_dict()

        diagnostics = VisualDiagnostics(
            outlier_points_count=outlier_count,
            iqr_overlap_ratio=iqr_overlap,
            visual_notes=notes,
            vision_model_used=False,
            visual_inspection_claimed=False,
        )

        return VisualEvidence(
            graph_id=f"graph_{task_id}_twogroup_box",
            task_id=task_id,
            figure_type="box",
            title=title,
            x_label=grp_col,
            y_label=val_col,
            columns=[grp_col, val_col],
            method=method,
            data_summary={"groups": unique_groups, "outliers_detected": outlier_count},
            figure_spec=fig_spec,
            visual_diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------
    # 3. MULTI-GROUP BOX PLOT (ANOVA / Kruskal-Wallis)
    # ------------------------------------------------------------------
    def _plot_multi_group_box(
        self,
        task_id: str,
        method: str,
        columns: List[str],
        df: pd.DataFrame,
        results: Dict[str, Any],
    ) -> VisualEvidence:
        res_grp = results.get("group_column") if results else None
        res_val = results.get("value_column") if results else None
        grp_col = (res_grp if res_grp and res_grp in df.columns else None) or (columns[0] if len(columns) > 0 else "")
        val_col = (res_val if res_val and res_val in df.columns else None) or (columns[1] if len(columns) > 1 else (columns[0] if len(columns) > 0 else ""))

        s_val = pd.to_numeric(df[val_col], errors="coerce")
        s_grp = df[grp_col].astype(str)
        valid_mask = s_val.notna() & df[grp_col].notna() & ~np.isinf(s_val)

        clean_val = s_val[valid_mask]
        clean_grp = s_grp[valid_mask]
        unique_groups = list(clean_grp.unique())

        total_outliers = 0
        for gname in unique_groups:
            gvals = clean_val[clean_grp == gname].to_numpy(dtype=float)
            if len(gvals) >= 4:
                q1, q3 = float(np.percentile(gvals, 25)), float(np.percentile(gvals, 75))
                iqr = q3 - q1
                total_outliers += int(np.sum((gvals < q1 - 1.5 * iqr) | (gvals > q3 + 1.5 * iqr)))

        title = f"Multi-Group Distribution: {val_col} across {grp_col} ({method})"

        fig_spec: Dict[str, Any] = {}
        if _HAS_ALTAIR:
            plot_df = pd.DataFrame({grp_col: clean_grp, val_col: clean_val})
            chart = alt.Chart(plot_df, title=title).mark_boxplot().encode(
                x=alt.X(grp_col, type="nominal"),
                y=alt.Y(val_col, type="quantitative"),
                color=alt.Color(grp_col, type="nominal", legend=None),
            )
            fig_spec = chart.to_dict()

        notes = [f"Depicted {len(unique_groups)} group distributions with {total_outliers} total visual outliers."]

        diagnostics = VisualDiagnostics(
            outlier_points_count=total_outliers,
            visual_notes=notes,
            vision_model_used=False,
            visual_inspection_claimed=False,
        )

        return VisualEvidence(
            graph_id=f"graph_{task_id}_multigroup_box",
            task_id=task_id,
            figure_type="multi_box",
            title=title,
            x_label=grp_col,
            y_label=val_col,
            columns=[grp_col, val_col],
            method=method,
            data_summary={"k_groups": len(unique_groups), "total_outliers": total_outliers},
            figure_spec=fig_spec,
            visual_diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------
    # 4. CATEGORICAL GROUPED BAR CHART (Chi-Square)
    # ------------------------------------------------------------------
    def _plot_categorical_bar(
        self,
        task_id: str,
        method: str,
        columns: List[str],
        df: pd.DataFrame,
        results: Dict[str, Any],
    ) -> VisualEvidence:
        res_a = results.get("column_a") if results else None
        res_b = results.get("column_b") if results else None
        col_a = (res_a if res_a and res_a in df.columns else None) or (columns[0] if len(columns) > 0 else "")
        col_b = (res_b if res_b and res_b in df.columns else None) or (columns[1] if len(columns) > 1 else (columns[0] if len(columns) > 0 else ""))

        valid_df = df[[col_a, col_b]].dropna()
        contingency = pd.crosstab(valid_df[col_a], valid_df[col_b])
        sparse_cells = int(np.sum(contingency.to_numpy() < 5))

        title = f"Categorical Contingency: {col_a} vs {col_b} ({method})"

        fig_spec: Dict[str, Any] = {}
        if _HAS_ALTAIR:
            plot_df = valid_df.copy()
            chart = alt.Chart(plot_df, title=title).mark_bar().encode(
                x=alt.X(col_a, type="nominal"),
                y=alt.Y("count()", type="quantitative"),
                color=alt.Color(col_b, type="nominal"),
                xOffset=alt.X(col_b, type="nominal"),
            )
            fig_spec = chart.to_dict()

        notes = [f"Cross-tabulated {contingency.shape[0]}x{contingency.shape[1]} contingency matrix."]
        if sparse_cells > 0:
            notes.append(f"Visual diagnostics flagged {sparse_cells} sparse contingency cells with count < 5.")

        diagnostics = VisualDiagnostics(
            sparse_categories_count=sparse_cells,
            visual_notes=notes,
            vision_model_used=False,
            visual_inspection_claimed=False,
        )

        return VisualEvidence(
            graph_id=f"graph_{task_id}_cat_bar",
            task_id=task_id,
            figure_type="grouped_bar",
            title=title,
            x_label=col_a,
            y_label="Count",
            columns=[col_a, col_b],
            method=method,
            data_summary={"table_shape": list(contingency.shape), "sparse_cells": sparse_cells},
            figure_spec=fig_spec,
            visual_diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------
    # 5. DISTRIBUTION & OUTLIER HISTOGRAM (Outlier Sensitivity)
    # ------------------------------------------------------------------
    def _plot_distribution_histogram(
        self,
        task_id: str,
        method: str,
        columns: List[str],
        df: pd.DataFrame,
        results: Dict[str, Any],
    ) -> VisualEvidence:
        res_col = (results.get("column") or results.get("column_a")) if results else None
        col = (res_col if res_col and res_col in df.columns else None) or (columns[0] if len(columns) > 0 else "")
        if results.get("variable_type") == "categorical" or "frequencies" in results:
            return self._plot_categorical_frequency_bar(task_id, method, columns, df, results)

        s = pd.to_numeric(df[col], errors="coerce").dropna()
        vals = s.to_numpy(dtype=float)
        if len(vals) == 0:
            return self._plot_categorical_frequency_bar(task_id, method, columns, df, results)

        q1 = float(np.percentile(vals, 25)) if len(vals) >= 4 else float(np.min(vals))
        q3 = float(np.percentile(vals, 75)) if len(vals) >= 4 else float(np.max(vals))
        iqr = q3 - q1
        lower_fence, upper_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_count = int(np.sum((vals < lower_fence) | (vals > upper_fence)))
        if outlier_count == 0 and ("outlier_count" in results or "n_outliers" in results):
            outlier_count = int(results.get("outlier_count", results.get("n_outliers", 0)))

        title = f"Outlier Sensitivity & Distribution: {col} ({method})"

        fig_spec: Dict[str, Any] = {}
        if _HAS_ALTAIR:
            plot_df = pd.DataFrame({col: vals})
            chart = alt.Chart(plot_df, title=title).mark_bar(color=_PRIMARY_COLOR, opacity=0.8).encode(
                x=alt.X(col, type="quantitative", bin=alt.Bin(maxbins=25)),
                y=alt.Y("count()", type="quantitative"),
            )
            fig_spec = chart.to_dict()

        notes = [
            f"Fences: [{lower_fence:.2f}, {upper_fence:.2f}], IQR: {iqr:.2f}.",
            f"Detected {outlier_count} potential outlier observations outside 1.5*IQR whiskers.",
        ]

        diagnostics = VisualDiagnostics(
            outlier_points_count=outlier_count,
            visual_notes=notes,
            vision_model_used=False,
            visual_inspection_claimed=False,
        )

        return VisualEvidence(
            graph_id=f"graph_{task_id}_distribution",
            task_id=task_id,
            figure_type="distribution_histogram",
            title=title,
            x_label=col,
            y_label="Frequency",
            columns=[col],
            method=method,
            data_summary={"outlier_count": outlier_count, "iqr": round(iqr, 4)},
            figure_spec=fig_spec,
            visual_diagnostics=diagnostics,
        )

    def _plot_categorical_frequency_bar(
        self,
        task_id: str,
        method: str,
        columns: List[str],
        df: pd.DataFrame,
        results: Dict[str, Any],
    ) -> VisualEvidence:
        col = columns[0]
        freqs = dict(results.get("frequencies") or {})
        if not freqs and col in df.columns:
            counts = df[col].dropna().astype(str).str.strip().value_counts()
            freqs = {str(k): int(v) for k, v in counts.items()}

        mode_val = str(results.get("mode") or (list(freqs.keys())[0] if freqs else "None"))
        n_cats = len(freqs)
        title = f"Categorical Distribution: {col} ({method})"

        fig_spec: Dict[str, Any] = {}
        if _HAS_ALTAIR and freqs:
            plot_df = pd.DataFrame({"category": list(freqs.keys()), "count": list(freqs.values())})
            chart = alt.Chart(plot_df, title=title).mark_bar(color=_PRIMARY_COLOR, opacity=0.85).encode(
                x=alt.X("category:N", sort="-y", title=col),
                y=alt.Y("count:Q", title="Frequency"),
            )
            fig_spec = chart.to_dict()

        notes = [
            f"Plotted categorical frequency distribution across {n_cats} categories.",
            f"Dominant mode is '{mode_val}' with count={freqs.get(mode_val, 0)}.",
        ]

        diagnostics = VisualDiagnostics(
            sparse_categories_count=sum(1 for v in freqs.values() if v < 5),
            visual_notes=notes,
            vision_model_used=False,
            visual_inspection_claimed=False,
        )

        return VisualEvidence(
            graph_id=f"graph_{task_id}_cat_dist",
            task_id=task_id,
            figure_type="distribution_histogram",
            title=title,
            x_label=col,
            y_label="Frequency",
            columns=[col],
            method=method,
            data_summary={"num_categories": n_cats, "mode": mode_val},
            figure_spec=fig_spec,
            visual_diagnostics=diagnostics,
        )

    def _plot_generic_distribution(
        self,
        task_id: str,
        method: str,
        columns: List[str],
        df: pd.DataFrame,
        results: Dict[str, Any],
    ) -> VisualEvidence:
        return self._plot_distribution_histogram(task_id, method, columns, df, results)


_VISUALIZATION_SERVICE = VisualizationService()


def get_visualization_service() -> VisualizationService:
    """Provide global VisualizationService instance."""
    return _VISUALIZATION_SERVICE

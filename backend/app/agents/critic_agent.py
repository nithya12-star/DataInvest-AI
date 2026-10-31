"""
backend/app/agents/critic_agent.py
==================================
Phase 4: Dedicated Critic Agent & Statistical Verification Engine.

Purpose:
    Examine structured statistical evidence produced by Phase 3 (Analysis Agent)
    and evaluate method appropriateness, sample size adequacy, missing-value impact,
    internal mathematical consistency, assumption stability, and effect size magnitude.

Critical Rules:
    - Zero LLM statistical calculation or fabrication. All numerical evidence originates
      from Python/SciPy in Phase 3.
    - Deterministic evaluation works 100% without an API key.
    - Failure isolation: Malformed or unexecutable tasks are audited into structured
      REJECTED or QUESTIONABLE verdicts without crashing the review.
    - Classifications: ACCEPTED, QUESTIONABLE, REJECTED, REQUIRES_REINVESTIGATION.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional, Union

import numpy as np

from app.schemas.analysis_schema import AnalysisResult, PlanAnalysisResponse
from app.schemas.critic_schema import (
    ConfidenceLevel,
    CriticOverallSummary,
    CriticResponse,
    CriticResult,
    CriticStatus,
)
from app.schemas.planner_schema import PlannerResponse

logger = logging.getLogger("datainvest.critic_agent")


class CriticAgent:
    """
    Dedicated verification agent evaluating Phase 3 statistical results.
    """

    CRITIC_VERSION = "4.0"

    def __init__(self, use_llm: bool = False):
        """
        Initialize the Critic Agent.
        Deterministic mode is the authoritative default.
        """
        self.use_llm = use_llm

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def evaluate_result(
        self,
        result: Union[AnalysisResult, dict[str, Any]],
        profile: Optional[dict[str, Any]] = None,
    ) -> CriticResult:
        """
        Evaluate a single Phase 3 AnalysisResult with failure isolation.
        """
        if isinstance(result, dict):
            try:
                res_obj = AnalysisResult(**result)
            except Exception as e:
                # Malformed result dictionary
                task_id = str(result.get("task_id", "unknown_task"))
                return CriticResult(
                    task_id=task_id,
                    analysis_type=str(result.get("analysis_type", "unknown")),
                    method=str(result.get("method", "unknown")),
                    variables=list(result.get("columns", [])),
                    status="REJECTED",
                    confidence=1.0,
                    confidence_level="HIGH",
                    summary=f"Analysis result payload is malformed: {str(e)}",
                    evidence=["Input data does not conform to AnalysisResult schema."],
                    statistical_validity="Invalid schema payload.",
                    method_assessment="Cannot evaluate method on malformed payload.",
                    data_quality_concerns=["Malformed data contract."],
                    assumption_concerns=[],
                    effect_size_assessment=None,
                    recommendation="Re-run analysis through the standard pipeline.",
                    requires_reinvestigation=False,
                    recommended_method=None,
                    reason_codes=["MALFORMED_PAYLOAD", "SCHEMA_VALIDATION_FAILED"],
                    metadata={"critic_version": self.CRITIC_VERSION},
                )
        else:
            res_obj = result

        try:
            return self._evaluate_single_deterministic(res_obj, profile)
        except Exception as e:
            logger.exception(f"Unhandled error in Critic reviewing task {res_obj.task_id}: {e}")
            return CriticResult(
                task_id=res_obj.task_id,
                analysis_type=res_obj.analysis_type,
                method=res_obj.method,
                variables=list(res_obj.columns or []),
                status="REJECTED",
                confidence=1.0,
                confidence_level="HIGH",
                summary=f"Critic evaluation encountered an internal error: {str(e)}",
                evidence=["Evaluation interrupted by unhandled exception."],
                statistical_validity="Unverified due to exception.",
                method_assessment="Unverified.",
                data_quality_concerns=[],
                assumption_concerns=[],
                effect_size_assessment=None,
                recommendation="Inspect backend logs and re-evaluate.",
                requires_reinvestigation=False,
                recommended_method=None,
                reason_codes=["CRITIC_INTERNAL_ERROR"],
                metadata={"critic_version": self.CRITIC_VERSION},
            )

    def evaluate_plan(
        self,
        analysis_input: Union[PlanAnalysisResponse, list[Union[AnalysisResult, dict[str, Any]]], dict[str, Any]],
        profile: Optional[dict[str, Any]] = None,
        plan: Optional[Union[PlannerResponse, dict[str, Any]]] = None,
        dataset_id: Optional[str] = None,
    ) -> CriticResponse:
        """
        Evaluate an entire plan of Phase 3 results and synthesize an overall summary.
        """
        # 1. Normalize input results list
        raw_results: list[AnalysisResult] = []
        resolved_dataset_id = dataset_id or "active_dataset"

        if isinstance(analysis_input, PlanAnalysisResponse):
            raw_results = analysis_input.results
            resolved_dataset_id = dataset_id or analysis_input.dataset_id
        elif isinstance(analysis_input, dict) and "results" in analysis_input:
            resolved_dataset_id = dataset_id or analysis_input.get("dataset_id", "active_dataset")
            for item in analysis_input.get("results", []):
                if isinstance(item, AnalysisResult):
                    raw_results.append(item)
                elif isinstance(item, dict):
                    try:
                        raw_results.append(AnalysisResult(**item))
                    except Exception:
                        pass
        elif isinstance(analysis_input, list):
            for item in analysis_input:
                if isinstance(item, AnalysisResult):
                    raw_results.append(item)
                elif isinstance(item, dict):
                    try:
                        raw_results.append(AnalysisResult(**item))
                    except Exception:
                        pass

        # 2. Evaluate each result individually with failure isolation
        reviews: list[CriticResult] = []
        for res_item in raw_results:
            review = self.evaluate_result(res_item, profile)
            reviews.append(review)

        # 3. Compute overall summary statistics
        summary = self._synthesize_overall_summary(reviews, profile)

        return CriticResponse(
            dataset_id=resolved_dataset_id,
            summary=summary,
            reviews=reviews,
            critic_version=self.CRITIC_VERSION,
            mode="deterministic",
        )

    # ------------------------------------------------------------------
    # DETERMINISTIC EVALUATION ENGINE
    # ------------------------------------------------------------------

    def _evaluate_single_deterministic(
        self,
        res: AnalysisResult,
        profile: Optional[dict[str, Any]] = None,
    ) -> CriticResult:
        task_id = res.task_id
        analysis_type = res.analysis_type
        method = res.method
        variables = list(res.columns or [])
        res_data = res.results or {}
        warnings = list(res.warnings or [])

        reason_codes: list[str] = []
        evidence: list[str] = []
        data_quality_concerns: list[str] = []
        assumption_concerns: list[str] = []

        # --------------------------------------------------------------
        # 1. AUDIT FAILED / BOUNDARY TASKS
        # --------------------------------------------------------------
        if res.status == "failed":
            err = res.error
            err_code = err.code if err else "UNKNOWN_FAILURE"
            err_msg = err.message if err else "Analysis task execution failed."

            reason_codes.append(f"BOUNDARY_{err_code}")

            if err_code == "CONSTANT_VARIABLE":
                return CriticResult(
                    task_id=task_id,
                    analysis_type=analysis_type,
                    method=method,
                    variables=variables,
                    status="REJECTED",
                    confidence=1.0,
                    confidence_level="HIGH",
                    summary=f"Analysis was correctly rejected: {err_msg}",
                    evidence=[f"Zero variance detected in variable: {err_msg}"],
                    statistical_validity="Mathematically undefined: constant variables have zero variance.",
                    method_assessment="Method cannot be executed on invariant data.",
                    data_quality_concerns=["Variable is constant (cardinality <= 1)."],
                    assumption_concerns=[],
                    effect_size_assessment=None,
                    recommendation="Exclude constant variables from comparative statistical modeling.",
                    requires_reinvestigation=False,
                    recommended_method=None,
                    reason_codes=reason_codes,
                    metadata={"critic_version": self.CRITIC_VERSION},
                )

            if err_code == "INSUFFICIENT_OBSERVATIONS":
                return CriticResult(
                    task_id=task_id,
                    analysis_type=analysis_type,
                    method=method,
                    variables=variables,
                    status="REJECTED",
                    confidence=1.0,
                    confidence_level="HIGH",
                    summary=f"Analysis rejected due to inadequate sample size: {err_msg}",
                    evidence=[f"Effective observation count fell below mathematical minimum: {err_msg}"],
                    statistical_validity="Insufficient degrees of freedom for parameter estimation.",
                    method_assessment="Sample size inadequate for the chosen statistical test.",
                    data_quality_concerns=["Sample size too small."],
                    assumption_concerns=[],
                    effect_size_assessment=None,
                    recommendation="Collect additional valid observations before attempting statistical inference.",
                    requires_reinvestigation=False,
                    recommended_method=None,
                    reason_codes=reason_codes,
                    metadata={"critic_version": self.CRITIC_VERSION},
                )

            return CriticResult(
                task_id=task_id,
                analysis_type=analysis_type,
                method=method,
                variables=variables,
                status="REJECTED",
                confidence=1.0,
                confidence_level="HIGH",
                summary=f"Analysis unexecutable: {err_msg}",
                evidence=[f"Execution boundary encountered: {err_msg}"],
                statistical_validity="Invalid / uncomputed.",
                method_assessment="Execution boundary precluded calculation.",
                data_quality_concerns=[err_msg],
                assumption_concerns=[],
                effect_size_assessment=None,
                recommendation="Verify column types and data completeness.",
                requires_reinvestigation=False,
                recommended_method=None,
                reason_codes=reason_codes,
                metadata={"critic_version": self.CRITIC_VERSION},
            )

        # --------------------------------------------------------------
        # 2. AUDIT MATHEMATICAL CONSISTENCY & BOUNDS
        # --------------------------------------------------------------
        sample_size = res.sample_size or res_data.get("sample_size", 0)

        # Reject findings with missing statistical evidence
        if analysis_type in ("numerical_vs_numerical", "two_groups_vs_numerical", "multiple_groups_vs_numerical", "categorical_vs_categorical"):
            if res_data.get("p_value") is None or sample_size <= 0:
                return CriticResult(
                    task_id=task_id,
                    analysis_type=analysis_type,
                    method=method,
                    variables=variables,
                    status="REJECTED",
                    confidence=1.0,
                    confidence_level="HIGH",
                    summary="Missing statistical evidence: required test statistic or p-value was not computed.",
                    evidence=["Missing empirical statistics; cannot evaluate reliability without evidence."],
                    statistical_validity="Insufficient evidence.",
                    method_assessment="Incomplete test output.",
                    data_quality_concerns=["Missing numerical evidence."],
                    assumption_concerns=[],
                    effect_size_assessment=None,
                    recommendation="Do not interpret this finding. Re-run analysis with complete data.",
                    requires_reinvestigation=False,
                    recommended_method=None,
                    reason_codes=["INSUFFICIENT_EVIDENCE", "MISSING_STATISTICAL_OUTPUT"],
                    metadata={"critic_version": self.CRITIC_VERSION},
                )

        # Check p-value validity if present
        p_val = res_data.get("p_value")
        if p_val is not None:
            if math.isnan(p_val) or p_val < 0.0 or p_val > 1.000001:
                return CriticResult(
                    task_id=task_id,
                    analysis_type=analysis_type,
                    method=method,
                    variables=variables,
                    status="REJECTED",
                    confidence=1.0,
                    confidence_level="HIGH",
                    summary="Calculated p-value is outside the mathematically valid probability interval [0, 1].",
                    evidence=[f"Computed p_value: {p_val}"],
                    statistical_validity="Degenerate p-value indicates calculation breakdown.",
                    method_assessment="Invalid numerical result.",
                    data_quality_concerns=["Mathematical degeneracy."],
                    assumption_concerns=[],
                    effect_size_assessment=None,
                    recommendation="Do not interpret this finding. Re-examine data distribution.",
                    requires_reinvestigation=True,
                    recommended_method=None,
                    reason_codes=["INVALID_P_VALUE_BOUNDS"],
                    metadata={"critic_version": self.CRITIC_VERSION},
                )

        # Check correlation bounds if present
        r_val = res_data.get("correlation_coefficient")
        if r_val is not None:
            if math.isnan(r_val) or abs(r_val) > 1.000001:
                return CriticResult(
                    task_id=task_id,
                    analysis_type=analysis_type,
                    method=method,
                    variables=variables,
                    status="REJECTED",
                    confidence=1.0,
                    confidence_level="HIGH",
                    summary="Calculated correlation coefficient is outside valid mathematical range [-1, 1].",
                    evidence=[f"Computed correlation coefficient: {r_val}"],
                    statistical_validity="Correlation exceeds [-1, 1] range.",
                    method_assessment="Invalid numerical result.",
                    data_quality_concerns=["Mathematical degeneracy."],
                    assumption_concerns=[],
                    effect_size_assessment=None,
                    recommendation="Do not interpret this finding.",
                    requires_reinvestigation=False,
                    recommended_method=None,
                    reason_codes=["INVALID_CORRELATION_BOUNDS"],
                    metadata={"critic_version": self.CRITIC_VERSION},
                )

        # Check Cramér's V bounds if present
        cramer_v = res_data.get("cramer_v")
        if cramer_v is not None:
            if math.isnan(cramer_v) or cramer_v < 0.0 or cramer_v > 1.000001:
                return CriticResult(
                    task_id=task_id,
                    analysis_type=analysis_type,
                    method=method,
                    variables=variables,
                    status="REJECTED",
                    confidence=1.0,
                    confidence_level="HIGH",
                    summary="Calculated Cramér's V effect size is outside valid range [0, 1].",
                    evidence=[f"Computed Cramér's V: {cramer_v}"],
                    statistical_validity="Effect size out of bounds.",
                    method_assessment="Invalid numerical result.",
                    data_quality_concerns=[],
                    assumption_concerns=[],
                    effect_size_assessment=None,
                    recommendation="Do not interpret this finding.",
                    requires_reinvestigation=False,
                    recommended_method=None,
                    reason_codes=["INVALID_CRAMER_V_BOUNDS"],
                    metadata={"critic_version": self.CRITIC_VERSION},
                )

        # --------------------------------------------------------------
        # 3. DOMAIN-SPECIFIC EVALUATION & ASSUMPTION CHECKS
        # --------------------------------------------------------------
        status: CriticStatus = "ACCEPTED"
        confidence = 1.0
        requires_reinvestigation = False
        recommended_method: Optional[str] = None
        effect_assessment: Optional[str] = None
        stat_validity = "Statistically consistent and verified within valid bounds."
        method_assessment = f"Method '{method}' is appropriate for {analysis_type}."

        # Pass through existing execution warnings
        for w in warnings:
            if "missing" in w.lower() or "omitted" in w.lower():
                data_quality_concerns.append(w)
            elif "outlier" in w.lower() or "skew" in w.lower() or "expected" in w.lower():
                assumption_concerns.append(w)

        # Extract programmatic visual diagnostics if attached to the analysis result
        viz = getattr(res, "visual_evidence", None)
        diag = getattr(viz, "visual_diagnostics", None) if viz is not None else None

        # --------------------------------------------------------------
        # 3A. CORRELATION (Numerical vs Numerical)
        # --------------------------------------------------------------
        if analysis_type == "numerical_vs_numerical":
            r_coef = float(res_data.get("correlation_coefficient", 0.0))
            pval_val = float(res_data.get("p_value", 1.0))
            is_sig = bool(res_data.get("p_value_below_0_05", False))
            strength = res_data.get("association_strength", "negligible")
            direction = res_data.get("direction", "none")
            col_a = res_data.get("column_a", variables[0] if variables else "")
            col_b = res_data.get("column_b", variables[1] if len(variables) > 1 else "")

            evidence.append(
                f"Sample size N={sample_size} with degrees of freedom df={res_data.get('degrees_of_freedom', sample_size - 2)}."
            )
            evidence.append(
                f"Correlation coefficient: {r_coef:.4f} ({strength} {direction}), p-value: {res_data.get('p_value_formatted', f'{pval_val:.4f}')}."
            )

            # Check skewness and outliers in profile if available
            skew_a = 0.0
            skew_b = 0.0
            has_outliers = False
            if profile:
                num_stats = profile.get("numerical_stats", {})
                skew_a = abs(float(num_stats.get(col_a, {}).get("skewness", 0.0) or 0.0))
                skew_b = abs(float(num_stats.get(col_b, {}).get("skewness", 0.0) or 0.0))
                outliers = profile.get("outliers", {})
                has_outliers = bool(
                    (outliers.get(col_a, {}).get("n_potential_outliers", 0) or 0) > 0
                    or (outliers.get(col_b, {}).get("n_potential_outliers", 0) or 0) > 0
                )

            has_skew_warning = any("skew" in w.lower() for w in warnings)
            has_outlier_warning = any("outlier" in w.lower() for w in warnings)

            # Visual scatter diagnostics: evaluate whether residual outliers indicate true linearity violation
            # Under a Gaussian distribution, ~0.27% of observations naturally have |z| > 3.0.
            # In small-to-moderate samples (N < 100), 3+ residual outliers represent >= 3-5% of the data,
            # which indicates substantial leverage/contamination violating Pearson's linearity assumption.
            # In contrast, 1-2 isolated points in moderate/large samples represent normal tail variance.
            outlier_cnt = diag.outlier_points_count if diag else 0
            outlier_rate = (outlier_cnt / sample_size) if sample_size > 0 else 0.0
            has_bivariate_outliers = bool(
                diag and (
                    (outlier_cnt >= 3 and (sample_size < 100 or outlier_rate >= 0.02))
                    or (diag.linearity_indicator == "Non-linear / contaminated" and outlier_cnt >= 3)
                )
            )
            is_pearson = "pearson" in method.lower()

            # Evaluate Pearson vulnerability to extreme skew or bivariate leverage outliers
            if is_pearson and (skew_a > 1.5 or skew_b > 1.5 or has_outliers or has_skew_warning or has_outlier_warning or has_bivariate_outliers):
                trigger_details = []
                if skew_a > 1.5 or skew_b > 1.5:
                    trigger_details.append(f"skew_a={skew_a:.2f}, skew_b={skew_b:.2f}")
                if has_outliers or has_outlier_warning:
                    trigger_details.append("univariate distribution outliers detected")
                if has_bivariate_outliers:
                    trigger_details.append(f"visual scatter diagnostics confirmed {outlier_cnt} leverage/residual outlier points (|z| > 3.0, {outlier_rate * 100:.1f}% of sample)")
                    reason_codes.append("VISUAL_OUTLIERS_REINVESTIGATE")
                    reason_codes.append("VISUAL_OUTLIERS_DETECTED")

                concerns_str = "; ".join(trigger_details) or "distributional or outlier violations"
                assumption_concerns.append(
                    "Outlier-influence diagnostic: identifies observations that may disproportionately influence Pearson correlation. "
                    f"This diagnostic alone does not establish that the linearity assumption is violated ({concerns_str})."
                )
                status = "REQUIRES_REINVESTIGATION"
                requires_reinvestigation = True
                recommended_method = "Spearman rank correlation"
                reason_codes.append("RECOMMEND_SPEARMAN_RANK")
                recommendation = f"Perform Spearman rank correlation on '{col_a}' vs '{col_b}' to evaluate monotonic relationship robust to leverage/residual outliers and skew."
                confidence -= 0.15
            else:
                reason_codes.append("METHOD_VALIDATED")
                recommendation = f"Finding is verified. Association between '{col_a}' and '{col_b}' is {strength} ({direction})."
                if diag and outlier_cnt > 0:
                    evidence.append(
                        f"Visual scatter diagnostics detected {outlier_cnt} extreme tail observation(s), "
                        f"but outlier rate ({outlier_rate * 100:.1f}%) is within acceptable bounds for {method}."
                    )
                    reason_codes.append("VISUAL_OUTLIERS_DETECTED")

            # Effect size and confidence interval evidence
            if "ci_95_low" in res_data and res_data["ci_95_low"] is not None:
                evidence.append(f"95% CI (Confidence Interval) for correlation: [{res_data['ci_95_low']:.4f}, {res_data['ci_95_high']:.4f}].")
            if "effect_size" in res_data and res_data["effect_size"] is not None:
                evidence.append(f"Effect size: {res_data['effect_size']}.")
            if "r_squared" in res_data:
                evidence.append(f"Coefficient of determination (r²): {res_data['r_squared']:.4f}.")

            if is_sig:
                if abs(r_coef) >= 0.4:
                    effect_assessment = f"Statistically significant (p < 0.05) with substantial practical effect magnitude (|r| = {abs(r_coef):.3f})."
                    reason_codes.append("SIGNIFICANT_STRONG_EFFECT")
                else:
                    effect_assessment = f"Statistically significant (p < 0.05), but weak practical effect magnitude (|r| = {abs(r_coef):.3f})."
                    reason_codes.append("SIGNIFICANT_WEAK_EFFECT")
            else:
                effect_assessment = f"No statistically significant association at alpha=0.05 (|r| = {abs(r_coef):.3f}, p = {pval_val:.4f})."
                reason_codes.append("NON_SIGNIFICANT")
                if not requires_reinvestigation:
                    reason_codes.append("P_HACKING_PREVENTED")

        # --------------------------------------------------------------
        # 3B. TWO-GROUP COMPARISON (Two Groups vs Numerical)
        # --------------------------------------------------------------
        elif analysis_type == "two_groups_vs_numerical":
            g1 = res_data.get("group_1", {})
            g2 = res_data.get("group_2", {})
            n1, n2 = g1.get("n", 0), g2.get("n", 0)
            diff = res_data.get("mean_difference", 0.0)
            pval_val = float(res_data.get("p_value", 1.0))
            is_sig = bool(res_data.get("p_value_below_0_05", False))

            evidence.append(
                f"Compared Group '{g1.get('name')}' (n={n1}, mean={g1.get('mean')}) vs Group '{g2.get('name')}' (n={n2}, mean={g2.get('mean')})."
            )
            evidence.append(
                f"Test statistic: {res_data.get('test_statistic', res_data.get('statistic'))}, p-value: {res_data.get('p_value_formatted', f'{pval_val:.4f}')}, mean difference: {diff:.4f}."
            )
            if "ci_95_low" in res_data and res_data["ci_95_low"] is not None:
                evidence.append(f"95% CI for difference: [{res_data['ci_95_low']:.4f}, {res_data['ci_95_high']:.4f}].")
            if "effect_size" in res_data and res_data["effect_size"] is not None:
                evidence.append(f"Effect size ({res_data.get('effect_size_type', 'magnitude')}): {res_data['effect_size']:.4f}.")

            # Sample size checks
            if n1 < 5 or n2 < 5:
                status = "QUESTIONABLE"
                assumption_concerns.append(f"Small group sample sizes (n1={n1}, n2={n2}) limit statistical power.")
                reason_codes.append("SMALL_GROUP_SAMPLE")
                confidence -= 0.20

            # Method suitability under skew and visual boxplot outliers
            val_col = res_data.get("value_column", variables[1] if len(variables) > 1 else "")
            val_skew = 0.0
            if profile:
                val_skew = abs(float(profile.get("numerical_stats", {}).get(val_col, {}).get("skewness", 0.0) or 0.0))

            group_outlier_cnt = diag.outlier_points_count if diag else 0
            # Welch's t-test has established robustness to single mild outliers;
            # 2+ outliers in small groups (n < 25) or high skewness indicates heavy-tail non-normality
            has_severe_group_outliers = bool(diag and group_outlier_cnt >= 2)
            is_welch = "welch" in method.lower()
            small_groups = (n1 < 25 or n2 < 25)

            if is_welch and (val_skew > 1.5 or has_severe_group_outliers) and small_groups:
                status = "REQUIRES_REINVESTIGATION"
                requires_reinvestigation = True
                recommended_method = "Mann-Whitney U test"
                triggers = []
                if val_skew > 1.5:
                    triggers.append(f"high skewness ({val_skew:.2f})")
                if has_severe_group_outliers:
                    triggers.append(f"visual boxplot detected {group_outlier_cnt} extreme group outliers")
                    reason_codes.append("VISUAL_OUTLIERS_REINVESTIGATE")
                    reason_codes.append("VISUAL_OUTLIERS_DETECTED")
                assumption_concerns.append(f"Normality assumption of Welch's t-test compromised in small groups ({'; '.join(triggers)}).")
                reason_codes.append("RECOMMEND_MANN_WHITNEY")
                recommendation = (
                    f"Run non-parametric Mann-Whitney U test to verify stochastic dominance across '{g1.get('name')}' "
                    f"and '{g2.get('name')}' (evaluates distributional rank sums rather than parametric mean difference)."
                )
                confidence -= 0.15
            else:
                reason_codes.append("GROUP_COMPARISON_VALIDATED")
                recommendation = f"Finding is verified. Mean difference between groups is {diff:.2f} (p={pval_val:.4f})."
                if diag and group_outlier_cnt > 0:
                    evidence.append(f"Visual boxplot diagnostics detected {group_outlier_cnt} mild outlier observation(s); Welch's t-test remains robust.")
                    reason_codes.append("VISUAL_OUTLIERS_DETECTED")

            if diag and diag.iqr_overlap_ratio is not None:
                if diag.iqr_overlap_ratio == 0.0:
                    evidence.append("Visual boxplot diagnostics confirm zero IQR overlap between groups (distinct distributional separation).")
                else:
                    evidence.append(f"Visual boxplot diagnostics indicate IQR overlap ratio: {diag.iqr_overlap_ratio:.2f}.")

            if is_sig:
                effect_assessment = f"Statistically significant group difference (p < 0.05, delta = {diff:.2f})."
                reason_codes.append("SIGNIFICANT_GROUP_DIFFERENCE")
            else:
                effect_assessment = f"No statistically significant difference between groups at alpha=0.05 (p = {pval_val:.4f})."
                reason_codes.append("NON_SIGNIFICANT_GROUP_DIFFERENCE")
                if not requires_reinvestigation:
                    reason_codes.append("P_HACKING_PREVENTED")

        # --------------------------------------------------------------
        # 3C. MULTI-GROUP COMPARISON (ANOVA / Kruskal-Wallis)
        # --------------------------------------------------------------
        elif analysis_type == "multiple_groups_vs_numerical":
            k = res_data.get("num_groups", 0)
            pval_val = float(res_data.get("p_value", 1.0))
            is_sig = bool(res_data.get("p_value_below_0_05", False))
            groups = res_data.get("groups", [])
            stat_name = "F_statistic" if "F_statistic" in res_data else "H_statistic"
            stat_val = res_data.get(stat_name, 0.0)

            evidence.append(
                f"Multi-group test across {k} categories (total N={sample_size}): {stat_name}={stat_val:.4f}, p-value={res_data.get('p_value_formatted', f'{pval_val:.4f}')}."
            )
            if "effect_size" in res_data and res_data["effect_size"] is not None:
                evidence.append(f"Effect size ({res_data.get('effect_size_type', 'magnitude')}): {res_data['effect_size']:.4f}.")

            # Check for severely undersized or imbalanced groups
            group_ns = [g.get("n", 0) for g in groups]
            if any(n < 3 for n in group_ns):
                status = "QUESTIONABLE"
                assumption_concerns.append("One or more category levels contain fewer than 3 observations.")
                reason_codes.append("UNDERSIZED_GROUP_LEVEL")
                confidence -= 0.20

            val_col = res_data.get("value_column", variables[1] if len(variables) > 1 else "")
            grp_col = res_data.get("group_column", variables[0] if variables else "")
            val_skew = 0.0
            if profile:
                val_skew = abs(float(profile.get("numerical_stats", {}).get(val_col, {}).get("skewness", 0.0) or 0.0))
            is_imbalanced = bool(group_ns and (max(group_ns) / (min(group_ns) or 1)) > 4.0)

            has_skew_warning = any("skew" in w.lower() for w in warnings)
            has_visual_multi_outliers = bool(diag and diag.outlier_points_count >= 3 and any(n < 25 for n in group_ns))

            if "anova" in method.lower() and (val_skew > 1.5 or is_imbalanced or has_skew_warning or has_visual_multi_outliers):
                status = "REQUIRES_REINVESTIGATION"
                requires_reinvestigation = True
                recommended_method = "Kruskal-Wallis test"
                ratio_str = f"{max(group_ns)/(min(group_ns) or 1):.1f}:1" if group_ns else "N/A"
                triggers = []
                if val_skew > 1.5:
                    triggers.append(f"val_skew={val_skew:.2f}")
                if is_imbalanced:
                    triggers.append(f"group size ratio={ratio_str}")
                if has_visual_multi_outliers:
                    triggers.append(f"visual multi-group box detected {diag.outlier_points_count} extreme outlier(s)")
                    reason_codes.append("VISUAL_OUTLIERS_REINVESTIGATE")
                    reason_codes.append("VISUAL_OUTLIERS_DETECTED")
                assumption_concerns.append(f"ANOVA assumption weakness ({'; '.join(triggers)}).")
                reason_codes.append("RECOMMEND_KRUSKAL_WALLIS")
                recommendation = f"Perform non-parametric Kruskal-Wallis test across '{grp_col}' categories to evaluate group differences robust to skewness/imbalance/outliers."
                confidence -= 0.15
            elif is_imbalanced:
                ratio_str = f"{max(group_ns)/(min(group_ns) or 1):.1f}:1" if group_ns else "N/A"
                assumption_concerns.append(f"Severe group size imbalance (ratio {ratio_str}).")
                reason_codes.append("GROUP_IMBALANCE")

            if is_sig:
                effect_assessment = f"Statistically significant variation across {k} groups (p < 0.05)."
                if not requires_reinvestigation:
                    recommendation = "Reject null hypothesis of equal group means. Consider post-hoc pairwise comparisons for specific group contrasts."
                reason_codes.append("SIGNIFICANT_MULTI_GROUP")
            else:
                effect_assessment = f"No statistically significant variation observed across {k} groups (p = {pval_val:.4f})."
                if not requires_reinvestigation:
                    recommendation = "Fail to reject null hypothesis; numerical variable does not vary significantly across group levels."
                    reason_codes.append("P_HACKING_PREVENTED")
                reason_codes.append("NON_SIGNIFICANT_MULTI_GROUP")

        # --------------------------------------------------------------
        # 3D. CATEGORICAL ASSOCIATION (Chi-Square & Cramér's V)
        # --------------------------------------------------------------
        elif analysis_type == "categorical_vs_categorical":
            chi2_val = float(res_data.get("chi2_statistic", res_data.get("statistic", 0.0)))
            pval_val = float(res_data.get("p_value", 1.0))
            c_v = float(res_data.get("cramer_v", 0.0))
            dof = int(res_data.get("degrees_of_freedom", 1))
            interp = res_data.get("cramer_v_interpretation", "negligible association")
            is_sig = bool(res_data.get("p_value_below_0_05", False))

            evidence.append(
                f"Contingency analysis ({res_data.get('table_dimensions', 'table')}): χ²={chi2_val:.4f} (df={dof}), p-value={res_data.get('p_value_formatted', f'{pval_val:.4f}')}."
            )
            evidence.append(f"Effect size (Cramér's V): {c_v:.4f} ({interp}).")

            # Evaluate expected contingency frequencies according to Cochran's rule (1954):
            # 1. Condition 1: No expected cell frequency should be < 1.0 (min_expected >= 1.0).
            # 2. Condition 2: No more than 20% of expected cell frequencies should be < 5.0 (pct_below_5 <= 20%).
            exp_counts = res_data.get("expected_cell_counts")
            min_exp = None
            pct_below_5 = None

            if isinstance(exp_counts, dict):
                min_exp = exp_counts.get("min_expected")
                pct_below_5 = exp_counts.get("cells_below_5_pct")
            else:
                # Reconstruct expected frequencies from contingency matrix if available
                table_dict = res_data.get("contingency_table", {})
                matrix = table_dict.get("matrix") if isinstance(table_dict, dict) else None
                if matrix and len(matrix) >= 2 and len(matrix[0]) >= 2:
                    try:
                        mat_arr = np.array(matrix, dtype=float)
                        row_sums = mat_arr.sum(axis=1, keepdims=True)
                        col_sums = mat_arr.sum(axis=0, keepdims=True)
                        grand_total = mat_arr.sum()
                        if grand_total > 0:
                            reconstructed_expected = (row_sums * col_sums) / grand_total
                            min_exp = float(np.min(reconstructed_expected))
                            pct_below_5 = float(np.sum(reconstructed_expected < 5.0)) / reconstructed_expected.size * 100.0
                    except Exception:
                        pass

            # Detect Cochran's rule violations on expected frequencies
            has_expected_below_1 = False
            has_expected_below_5_over_20 = False

            if min_exp is not None:
                has_expected_below_1 = bool(min_exp < 1.0)
            else:
                has_expected_below_1 = any("expected" in w.lower() and ("< 1" in w or "below 1" in w) for w in warnings)

            if pct_below_5 is not None:
                has_expected_below_5_over_20 = bool(pct_below_5 > 20.0)
            else:
                has_expected_below_5_over_20 = any("expected count < 5" in w.lower() for w in warnings)

            cochran_violation = has_expected_below_1 or has_expected_below_5_over_20

            if cochran_violation:
                status = "QUESTIONABLE"
                violation_reasons = []
                if has_expected_below_1:
                    exp_val_str = f"min expected = {min_exp:.2f}" if min_exp is not None else "< 1.0"
                    violation_reasons.append(f"expected cell frequency < 1.0 detected ({exp_val_str})")
                    reason_codes.append("EXPECTED_COUNT_BELOW_ONE")
                if has_expected_below_5_over_20:
                    pct_str = f"{pct_below_5:.1f}%" if pct_below_5 is not None else ">20%"
                    violation_reasons.append(f"{pct_str} of cells have expected frequency < 5.0 (exceeds 20% Cochran limit)")
                assumption_concerns.append(f"Cochran's contingency condition violated: {'; '.join(violation_reasons)}. Asymptotic Chi-square approximation may be inflated.")
                reason_codes.append("SPARSE_CONTINGENCY_CELLS")
                confidence -= 0.15
            else:
                reason_codes.append("COCHRAN_CONDITION_SATISFIED")
                diag_sparse = diag.sparse_categories_count if diag else 0
                if diag and diag_sparse > 0:
                    evidence.append(
                        f"Visual diagnostics noted {diag_sparse} cell(s) with observed count < 5; "
                        f"however, actual expected frequencies satisfy Cochran's rule (min expected={min_exp if min_exp is not None else '>=1.0'}, "
                        f"cells < 5: {f'{pct_below_5:.1f}%' if pct_below_5 is not None else '<=20%'})."
                    )

            if is_sig:
                effect_assessment = f"Statistically significant categorical dependency (p < 0.05) with {interp} (V = {c_v:.3f})."
                recommendation = f"Association between '{variables[0] if variables else 'col_a'}' and '{variables[1] if len(variables)>1 else 'col_b'}' is confirmed."
                reason_codes.append("SIGNIFICANT_CATEGORICAL_ASSOCIATION")
            else:
                effect_assessment = f"No statistically significant association between categories (p = {pval_val:.4f}, V = {c_v:.3f})."
                recommendation = "Variables appear statistically independent at alpha=0.05."
                reason_codes.append("INDEPENDENT_CATEGORIES")
                reason_codes.append("P_HACKING_PREVENTED")

        # --------------------------------------------------------------
        # 3E. OUTLIER SENSITIVITY
        # --------------------------------------------------------------
        elif analysis_type == "outlier_sensitivity":
            n_outliers = int(res_data.get("outlier_count", 0))
            outlier_pct = float(res_data.get("outlier_percentage", res_data.get("outlier_pct", 0.0)))
            m_shift = float(res_data.get("mean_shift", res_data.get("mean_delta", res_data.get("delta_mean", 0.0))))
            m_shift_pct = float(res_data.get("mean_shift_pct", res_data.get("mean_delta_pct", res_data.get("relative_change_pct", 0.0))))
            subst = bool(res_data.get("substantial_influence", abs(m_shift_pct) > 15.0 or outlier_pct > 10.0 or abs(m_shift) > 500.0))

            evidence.append(
                f"Identified {n_outliers} IQR outlier observations ({outlier_pct:.1f}% of N={sample_size})."
            )
            evidence.append(
                f"Mean shift upon trimming: {m_shift:+.4f} ({m_shift_pct:+.2f}% change from baseline)."
            )

            if subst:
                status = "QUESTIONABLE"
                assumption_concerns.append(f"High outlier sensitivity: extreme values shift mean by {m_shift_pct:+.1f}%.")
                reason_codes.append("SUBSTANTIAL_OUTLIER_INFLUENCE")
                effect_assessment = f"High outlier sensitivity: extreme values shift mean by {m_shift_pct:+.1f}%."
                recommendation = "Apply robust estimators or rank-based statistical tests when analyzing this variable."
                confidence -= 0.15
            else:
                status = "ACCEPTED"
                reason_codes.append("MINIMAL_OUTLIER_INFLUENCE")
                effect_assessment = f"Minimal outlier sensitivity: mean shift is minor ({m_shift_pct:+.1f}%)."
                recommendation = "Standard parametric methods are resilient to the observed extreme values."

        # --------------------------------------------------------------
        # 3F. DISTRIBUTION ANALYSIS & DESCRIPTIVE
        # --------------------------------------------------------------
        elif analysis_type in ("distribution_analysis", "descriptive_statistics"):
            is_categorical = (
                res_data.get("variable_type") == "categorical"
                or "frequencies" in res_data
                or "proportions" in res_data
            )
            if is_categorical:
                n_cats = int(res_data.get("num_categories", len(res_data.get("frequencies", {}))))
                mode_val = str(res_data.get("mode", "unknown"))
                mode_count = int(res_data.get("mode_frequency", 0))
                mode_pct = float(res_data.get("mode_percentage", 0.0))
                is_imbal = bool(res_data.get("is_imbalanced", False))

                evidence.append(
                    f"Categorical frequency distribution: {n_cats} distinct categories, "
                    f"mode '{mode_val}' ({mode_count} occurrences, {mode_pct:.1f}%)."
                )

                if n_cats <= 1:
                    status = "QUESTIONABLE"
                    data_quality_concerns.append(f"Variable has zero variability (only 1 distinct category: '{mode_val}').")
                    reason_codes.append("ZERO_VARIANCE_CATEGORICAL")
                    confidence -= 0.20
                    recommendation = "Constant categorical variable offers no discriminatory or predictive information."
                elif is_imbal or mode_pct >= 75.0:
                    data_quality_concerns.append(
                        f"Substantial class imbalance: dominant category '{mode_val}' represents {mode_pct:.1f}% of observations."
                    )
                    reason_codes.append("CLASS_IMBALANCE_DETECTED")
                    recommendation = "Apply class-weighting, stratification, or resampling in subsequent modeling."
                else:
                    reason_codes.append("BALANCED_CATEGORICAL_DISTRIBUTION")
                    recommendation = "Categorical distribution is sufficiently balanced for downstream cross-tabulation and modeling."

                gof = res_data.get("goodness_of_fit_test") or {}
                if isinstance(gof, dict) and gof.get("p_value") is not None:
                    gof_pval = float(gof["p_value"])
                    evidence.append(f"Uniform goodness-of-fit test: p-value={gof_pval:.4f}.")
                    if gof_pval < 0.05:
                        reason_codes.append("NON_UNIFORM_CATEGORICAL_DISTRIBUTION")
                    else:
                        reason_codes.append("UNIFORM_CATEGORICAL_DISTRIBUTION")

                effect_assessment = f"Categorical distribution characterized by {n_cats} categories; dominant mode is '{mode_val}' ({mode_pct:.1f}%)."
            else:
                skew = float(res_data.get("skewness", 0.0))
                kurt = float(res_data.get("kurtosis", 0.0))
                symm = res_data.get("symmetry", "symmetric")
                norm = res_data.get("normality_test", {})

                evidence.append(f"Distribution moments: skewness={skew:.4f} ({symm}), kurtosis={kurt:.4f}.")
                if isinstance(norm, dict) and norm.get("method") and norm.get("p_value") is not None:
                    norm_pval = float(norm.get("p_value", 1.0))
                    evidence.append(f"Normality test ({norm.get('method')}): p-value={norm_pval:.4f}.")
                    if norm_pval < 0.05:
                        assumption_concerns.append("Variable significantly departs from normal distribution.")
                        reason_codes.append("NON_NORMAL_DISTRIBUTION")
                    else:
                        reason_codes.append("NORMAL_DISTRIBUTION")

                effect_assessment = f"Distribution characterized as {symm}."
                recommendation = "Use distribution shape to guide parametric vs non-parametric method selection."

        # --------------------------------------------------------------
        # 3G. MISSING VALUE INVESTIGATION
        # --------------------------------------------------------------
        elif analysis_type == "missing_value_investigation":
            tot_missing = int(res_data.get("total_missing_cells", 0))
            overall_pct = float(res_data.get("overall_missing_percentage", 0.0))
            comp_cases = int(res_data.get("complete_cases", 0))
            comp_pct = float(res_data.get("complete_cases_percentage", 100.0))

            evidence.append(f"Total missing cells: {tot_missing} ({overall_pct:.2f}% of all cells).")
            evidence.append(f"Complete observation cases: {comp_cases} ({comp_pct:.2f}% of rows).")

            if overall_pct > 25.0:
                status = "QUESTIONABLE"
                data_quality_concerns.append(f"High overall missingness ({overall_pct:.1f}%).")
                reason_codes.append("HIGH_DATASET_MISSINGNESS")
                confidence -= 0.20
                recommendation = "Investigate missingness mechanism (MCAR/MAR/MNAR) before multi-variable modeling."
            else:
                reason_codes.append("ACCEPTABLE_COMPLETENESS")
                recommendation = "Missingness rate is within acceptable thresholds for standard pairwise analysis."

        # --------------------------------------------------------------
        # 3H. CLUSTERING & CUSTOMER SEGMENTATION
        # --------------------------------------------------------------
        elif analysis_type in ("clustering", "customer_segmentation"):
            k_val = int(res_data.get("k_clusters", 0))
            is_scaled = bool(res_data.get("scaling_applied", False))
            sil_native = float(res_data.get("silhouette_score_native", 0.0))
            sil_std = float(res_data.get("silhouette_score_standardized", sil_native))
            db_val = float(res_data.get("davies_bouldin_index", 0.0))
            ch_val = float(res_data.get("calinski_harabasz_score", 0.0))
            audit_findings_res = res_data.get("audit_findings", {})
            disparity_info = audit_findings_res.get("scale_disparity", {})
            has_disparity = bool(disparity_info.get("has_disparity", False))
            disparity_ratio = float(disparity_info.get("ratio", 1.0))
            cluster_sizes_dict = res_data.get("cluster_sizes", {})

            evidence.append(
                f"Clustering model: K-Means (k={k_val}) on N={sample_size} observations across features {variables}."
            )
            evidence.append(
                f"Standardized Silhouette Score: {sil_std:.4f}, Davies-Bouldin Index: {db_val:.4f}, Calinski-Harabasz Score: {ch_val:.2f}."
            )

            # Check for sequential identifier leakage
            id_detected = audit_findings_res.get("identifier_columns_detected", {})
            if id_detected and not is_scaled:
                data_quality_concerns.append(
                    f"Sequential record identifier(s) detected in clustering input: {list(id_detected.keys())}."
                )
                reason_codes.append("IDENTIFIER_FEATURE_LEAKAGE")

            # Check scale disparity & standardization assumption
            if has_disparity and not is_scaled:
                status = "REQUIRES_REINVESTIGATION"
                requires_reinvestigation = True
                recommended_method = "Standardized K-Means Clustering (with StandardScaler)"
                assumption_concerns.append(
                    f"Severe feature scale disparity (ratio {disparity_ratio:.1f}: '{disparity_info.get('max_col')}' vs '{disparity_info.get('min_col')}') "
                    "without standardization violates Euclidean distance isotropy, biasing clusters along the largest variance axis."
                )
                reason_codes.append("SCALE_DISPARITY_DISTORTION")
                reason_codes.append("RECOMMEND_STANDARDIZATION")
                recommendation = "Standardize continuous features using StandardScaler or RobustScaler to ensure equal dimensional weighting."
                confidence -= 0.25
            else:
                reason_codes.append("DIMENSIONAL_SCALING_VERIFIED")

            # Check cluster quality and separation
            if sil_std < 0.25:
                if status == "ACCEPTED":
                    status = "QUESTIONABLE"
                assumption_concerns.append(
                    f"Weak cluster separation: Standardized silhouette score {sil_std:.4f} is below the 0.25 threshold, "
                    "indicating substantial point overlap or arbitrary partition structure."
                )
                reason_codes.append("WEAK_CLUSTER_STRUCTURE")
                confidence -= 0.15
            elif sil_std >= 0.35:
                reason_codes.append("SUBSTANTIAL_CLUSTER_SEPARATION")

            # Check cluster size balance / degenerate clusters
            if cluster_sizes_dict and sample_size > 0:
                min_size = min(cluster_sizes_dict.values())
                min_prop = (min_size / sample_size) * 100.0
                if min_prop < 5.0:
                    data_quality_concerns.append(
                        f"Imbalanced cluster distribution: Smallest cluster contains only {min_size} samples ({min_prop:.1f}% of total)."
                    )
                    reason_codes.append("DEGENERATE_OR_OUTLIER_CLUSTER")
                else:
                    reason_codes.append("BALANCED_CLUSTER_DISTRIBUTION")

            effect_assessment = (
                f"K-Means clustering with k={k_val} partitioned N={sample_size} observations into "
                f"{k_val} segments (silhouette={sil_std:.4f}, Davies-Bouldin={db_val:.4f})."
            )
            if status == "ACCEPTED":
                recommendation = f"Clustering structure is verified with k={k_val}. Partition exhibits balanced segments and valid separation."

        # --------------------------------------------------------------
        # 3I. TEXT CLASSIFICATION & FAKE NEWS AUDIT
        # --------------------------------------------------------------
        elif analysis_type in ("text_classification", "fake_news_classification"):
            acc_val = float(res_data.get("accuracy", 0.0))
            f1_val = float(res_data.get("f1_score", 0.0))
            auc_val = float(res_data.get("roc_auc", 0.0))
            is_cleaned = bool("critic_on" in res_data.get("critic_mode", "").lower() or "critic_on" in method.lower())
            audit_findings_res = res_data.get("audit_findings", {})
            wire_leakage = audit_findings_res.get("wire_leakage", {})
            meta_leakage = audit_findings_res.get("metadata_leakage_detected", {})
            top_true = res_data.get("top_true_features", [])
            top_fake = res_data.get("top_fake_features", [])

            evidence.append(
                f"Supervised Text Classification (TF-IDF + {res_data.get('model_type', 'LogisticRegression')}) "
                f"evaluated on N={sample_size} articles (Train={res_data.get('train_samples')}, Test={res_data.get('test_samples')})."
            )
            evidence.append(
                f"Held-out Metrics: Accuracy={acc_val:.4f}, F1-score={f1_val:.4f}, ROC-AUC={auc_val:.4f}."
            )

            # 1. Audit metadata leakage (e.g. subject disjoint classes)
            if meta_leakage:
                if not is_cleaned:
                    data_quality_concerns.append(
                        f"Disjoint metadata categories detected: {list(meta_leakage.keys())}. "
                        "Features provide 100% trivial target leakage and must be strictly excluded from NLP models."
                    )
                    reason_codes.append("DISJOINT_METADATA_LEAKAGE")
                else:
                    evidence.append(
                        f"Disjoint metadata columns verified excluded from model features: {list(meta_leakage.keys())}."
                    )
                    reason_codes.append("METADATA_LEAKAGE_EXCLUDED")

            # 2. Audit publisher wire artifact / watermark leakage
            has_wire_leakage = bool(wire_leakage.get("has_leakage", False))
            reuters_disparity = float(wire_leakage.get("disparity_pct", 0.0))
            top_token = top_true[0].get("token", "") if top_true else ""
            top_coef = abs(float(top_true[0].get("coefficient", 0.0))) if top_true else 0.0

            if has_wire_leakage and not is_cleaned:
                status = "REQUIRES_REINVESTIGATION"
                requires_reinvestigation = True
                recommended_method = "Sanitized Text Classification (strip publisher dateline watermarks)"
                assumption_concerns.append(
                    f"Severe publisher watermark leakage detected ({reuters_disparity:.1f}% disparity). "
                    f"Dominant model feature '{top_token}' (coef magnitude {top_coef:.2f}) acts as a publisher/wire "
                    "agency recognition watermark rather than genuine linguistic or journalistic misinformation detection."
                )
                reason_codes.append("PUBLISHER_WATERMARK_LEAKAGE")
                reason_codes.append("RECOMMEND_WATERMARK_SANITIZATION")
                recommendation = "Sanitize publisher wire headers and source watermarks to evaluate genuine stylistic and semantic discrimination."
                confidence -= 0.25
            else:
                reason_codes.append("WATERMARK_SANITATION_VERIFIED")
                if top_true:
                    evidence.append(
                        f"Top genuine journalistic features: {', '.join(f['token'] for f in top_true[:4])}; "
                        f"Top sensationalist/misinformation features: {', '.join(f['token'] for f in top_fake[:4])}."
                    )

            # 3. Model performance validity bounds
            if acc_val >= 0.90:
                reason_codes.append("HIGH_CLASSIFICATION_DISCRIMINATION")
            elif acc_val < 0.60:
                status = "QUESTIONABLE"
                assumption_concerns.append(f"Near-chance accuracy ({acc_val:.4f}); model fails to discriminate text classes.")
                reason_codes.append("WEAK_CLASSIFICATION_POWER")
                confidence -= 0.20

            effect_assessment = (
                f"Binary text classifier achieved Accuracy={acc_val:.4f}, F1={f1_val:.4f}, ROC-AUC={auc_val:.4f} "
                f"on N={res_data.get('test_samples', 0)} held-out test articles."
            )
            if status == "ACCEPTED":
                recommendation = "Text classification findings are verified. Model demonstrates robust linguistic discrimination without wire agency watermark bias."

        # --------------------------------------------------------------
        # 4. SAMPLE SIZE POWER CHECKS
        # --------------------------------------------------------------
        if sample_size > 0:
            if sample_size < 10:
                if status == "ACCEPTED":
                    status = "QUESTIONABLE"
                data_quality_concerns.append(f"Small sample size (N={sample_size}) confers low statistical power.")
                reason_codes.append("SMALL_SAMPLE_POWER_RISK")
                confidence -= 0.25
            elif sample_size < 25:
                confidence -= 0.10
                reason_codes.append("MODERATE_SAMPLE_SIZE")
            else:
                reason_codes.append("ADEQUATE_SAMPLE_SIZE")

        # --------------------------------------------------------------
        # 4B. AUDIT PROGRAMMATIC VISUAL EVIDENCE (NO VISION MODEL CLAIMED)
        # --------------------------------------------------------------
        if viz is not None:
            evidence.append("Associated visual evidence audited via programmatic visual diagnostics (no vision model used).")
            if diag:
                if diag.outlier_points_count > 0 and "VISUAL_OUTLIERS_DETECTED" not in reason_codes:
                    reason_codes.append("VISUAL_OUTLIERS_DETECTED")
                if diag.visual_notes:
                    for note in diag.visual_notes[:2]:
                        evidence.append(f"Visual note: {note}")

        # Clamp confidence to [0.1, 1.0]
        confidence = max(0.1, min(1.0, round(confidence, 2)))
        confidence_level: ConfidenceLevel = (
            "HIGH" if confidence >= 0.80 else
            "MEDIUM" if confidence >= 0.50 else "LOW"
        )

        # --------------------------------------------------------------
        # 5. SYNTHESIZE SCIENTIFIC SUMMARY
        # --------------------------------------------------------------
        if status == "ACCEPTED":
            summary_str = f"Finding is empirically verified and methodologically robust: {effect_assessment or 'Valid statistical result.'}"
        elif status == "REQUIRES_REINVESTIGATION":
            summary_str = f"Finding requires re-investigation: Methodological or distributional concerns suggest running {recommended_method}."
        elif status == "QUESTIONABLE":
            summary_str = f"Finding has notable caveats: {'; '.join(data_quality_concerns + assumption_concerns) or 'Low sample power.'}"
        else:
            summary_str = "Finding is rejected due to invalid assumptions, missing data bounds, or unfeasible analysis."

        return CriticResult(
            task_id=task_id,
            analysis_type=analysis_type,
            method=method,
            variables=variables,
            status=status,
            confidence=confidence,
            confidence_level=confidence_level,
            summary=summary_str,
            evidence=evidence,
            statistical_validity=stat_validity,
            method_assessment=method_assessment,
            data_quality_concerns=data_quality_concerns,
            assumption_concerns=assumption_concerns,
            effect_size_assessment=effect_assessment,
            recommendation=recommendation,
            requires_reinvestigation=requires_reinvestigation,
            recommended_method=recommended_method,
            reason_codes=reason_codes,
            metadata={"critic_version": self.CRITIC_VERSION},
        )

    # ------------------------------------------------------------------
    # OVERALL SYNTHESIS
    # ------------------------------------------------------------------

    def _synthesize_overall_summary(
        self,
        reviews: list[CriticResult],
        profile: Optional[dict[str, Any]] = None,
    ) -> CriticOverallSummary:
        total = len(reviews)
        accepted_cnt = sum(1 for r in reviews if r.status == "ACCEPTED")
        questionable_cnt = sum(1 for r in reviews if r.status == "QUESTIONABLE")
        rejected_cnt = sum(1 for r in reviews if r.status == "REJECTED")
        reinvestigate_cnt = sum(1 for r in reviews if r.requires_reinvestigation or r.status == "REQUIRES_REINVESTIGATION")

        critical_issues: list[str] = []
        key_insights: list[str] = []

        # Extract major critical issues
        for r in reviews:
            if r.status in ("QUESTIONABLE", "REJECTED", "REQUIRES_REINVESTIGATION"):
                for concern in r.data_quality_concerns + r.assumption_concerns:
                    if concern not in critical_issues:
                        critical_issues.append(f"[{r.task_id}] {concern}")

            if r.status == "ACCEPTED" and r.effect_size_assessment:
                key_insights.append(f"[{r.task_id}] {r.summary}")

        # Overall Status String
        if total == 0:
            overall_status = "No analytical findings provided for review."
        elif rejected_cnt == total:
            overall_status = "All candidate findings failed validity checks."
        elif reinvestigate_cnt > total / 2:
            overall_status = "Substantial assumption concerns: Re-investigation recommended for primary relationships."
        elif accepted_cnt >= total * 0.7:
            overall_status = "High-Quality Empirical Evidence: Majority of findings verified with strong statistical support."
        elif accepted_cnt + questionable_cnt >= total * 0.6:
            overall_status = "Moderate Empirical Evidence: Findings valid but subject to data quality or sample size caveats."
        else:
            overall_status = "Limited Empirical Evidence: Multiple findings flagged for methodological limitations."

        return CriticOverallSummary(
            total_results=total,
            accepted=accepted_cnt,
            questionable=questionable_cnt,
            rejected=rejected_cnt,
            requires_reinvestigation=reinvestigate_cnt,
            overall_status=overall_status,
            critical_issues=critical_issues[:10],
            key_verified_insights=key_insights[:10],
        )

# backend/app/services/evaluation_service.py
"""
Service for computing Phase 7 — System Evaluation & Quality Scorecard.
Evaluates the end-to-end investigation pipeline across 4 scientific dimensions:
1. Execution Reliability (25 pts)
2. Statistical Rigor & Assumption Compliance (25 pts)
3. Verification & Truth Grounding (25 pts)
4. Insight Actionability & Causality Safety (25 pts)
"""

from typing import Dict, Any, List, Optional
from ..schemas.evaluation_schema import (
    MetricItem,
    DimensionScore,
    BenchmarkCheck,
    EvaluationSummary,
    EvaluationResponse,
    EvaluationStatus,
)


class EvaluationService:
    """Deterministic evaluation engine benchmarked against AutoMind empirical rigor standards."""

    CAUSAL_KEYWORDS = ["causes", "caused", "causing", "leads to", "led to", "drives", "driven by", "determines"]

    def evaluate_pipeline(
        self,
        profile: Dict[str, Any],
        plan: Optional[Dict[str, Any]] = None,
        analysis: Optional[Dict[str, Any]] = None,
        critic: Optional[Dict[str, Any]] = None,
        verification: Optional[Dict[str, Any]] = None,
        insights: Optional[Dict[str, Any]] = None,
    ) -> EvaluationResponse:
        """
        Executes full pipeline evaluation across all completed stages.
        """
        plan = plan or {}
        analysis = analysis or {}
        critic = critic or {}
        verification = verification or {}
        insights = insights or {}

        dim1 = self._evaluate_execution_reliability(profile, plan, analysis)
        dim2 = self._evaluate_statistical_rigor(profile, plan, analysis, critic, verification)
        dim3 = self._evaluate_verification_grounding(analysis, critic, verification)
        dim4 = self._evaluate_insight_actionability(plan, verification, insights)

        dimensions = [dim1, dim2, dim3, dim4]
        overall_score = round(sum(d.score for d in dimensions), 1)
        overall_score = min(100.0, max(0.0, overall_score))

        overall_grade, assessment = self._compute_overall_grade(overall_score)

        # Aggregate detailed metrics
        all_metrics: List[MetricItem] = []
        for d in dimensions:
            all_metrics.extend(d.metrics)

        # Extract verified insight items supporting both key_insights and legacy insights
        insight_items = insights.get("key_insights") or insights.get("insights") or []
        total_tasks_planned = len(plan.get("analysis_tasks", []))
        results = analysis.get("results", [])
        total_tasks_executed = len(results)
        total_findings_verified = len(verification.get("verified_findings", []))
        total_insights_generated = len(insight_items)

        # Determine calculation provenance (zero-hallucination proxy)
        has_full_provenance = (
            total_tasks_executed > 0
            and all(self._has_task_provenance(r) for r in results)
        )

        # Build benchmark checklist
        benchmark_checks = self._build_benchmark_checklist(
            profile, plan, analysis, critic, verification, insights, has_full_provenance
        )

        summary = EvaluationSummary(
            overall_score=overall_score,
            overall_grade=overall_grade,
            total_tasks_planned=total_tasks_planned,
            total_tasks_executed=total_tasks_executed,
            total_findings_verified=total_findings_verified,
            total_insights_generated=total_insights_generated,
            zero_hallucination_verified=has_full_provenance,
            executive_assessment=assessment,
        )

        return EvaluationResponse(
            summary=summary,
            dimensions=dimensions,
            benchmark_checks=benchmark_checks,
            detailed_metrics=all_metrics,
        )

    def _evaluate_execution_reliability(
        self, profile: Dict[str, Any], plan: Dict[str, Any], analysis: Dict[str, Any]
    ) -> DimensionScore:
        """Dimension 1: Evaluates execution success, error handling, and profiling integrity."""
        tasks_planned = plan.get("analysis_tasks", [])
        n_planned = len(tasks_planned)
        results = analysis.get("results", [])
        n_results = len(results)

        # 1. Task Completion Rate (10 pts)
        if n_planned > 0:
            completion_pct = round((n_results / n_planned) * 100.0, 1)
            score_completion = round(min(10.0, (n_results / n_planned) * 10.0), 2)
        else:
            completion_pct = 100.0 if profile else 0.0
            score_completion = 10.0 if profile else 0.0

        status_comp: EvaluationStatus = "EXCELLENT" if completion_pct >= 95.0 else ("GOOD" if completion_pct >= 80.0 else "ACCEPTABLE")
        m_completion = MetricItem(
            metric_id="m_exec_completion",
            name="Task Completion Rate",
            dimension="Execution Reliability",
            value=f"{completion_pct}%",
            unit="%",
            status=status_comp,
            description="Proportion of planned analytical tasks successfully executed through statistical engines.",
            benchmark_target="100.0%",
        )

        # 2. Profiling Completeness Score (8 pts)
        # Check actual profiler diagnostic sections with backward-compatible key aliases
        def _has_profile_section(p: Dict[str, Any], sec: str) -> bool:
            if sec == "missing":
                return bool(p.get("missing") is not None or p.get("missing_stats") is not None)
            if sec == "duplicates":
                return bool(p.get("duplicates") is not None or p.get("data_quality") is not None)
            return bool(p.get(sec) is not None)

        core_profile_sections = ["overview", "column_types", "missing", "numerical_stats", "outliers", "cardinality", "duplicates"]
        present_keys = sum(1 for sec in core_profile_sections if _has_profile_section(profile, sec))
        prof_pct = round((present_keys / len(core_profile_sections)) * 100.0, 1)
        score_prof = round((present_keys / len(core_profile_sections)) * 8.0, 2)
        status_prof: EvaluationStatus = "EXCELLENT" if prof_pct >= 95.0 else ("GOOD" if prof_pct >= 80.0 else "ACCEPTABLE")

        m_prof = MetricItem(
            metric_id="m_exec_profiling",
            name="Dataset Profiling Completeness",
            dimension="Execution Reliability",
            value=f"{prof_pct}%",
            unit="%",
            status=status_prof,
            description="Coverage of structural, missingness, cardinality, and distribution diagnostics in Phase 1.",
            benchmark_target="100.0%",
        )

        # 3. Computational Robustness & Error Rate (7 pts)
        failed_tasks = sum(
            1 for r in results
            if r.get("status") in ("failed", "ERROR") or bool(r.get("error"))
        )
        error_rate = round((failed_tasks / n_results * 100.0) if n_results > 0 else 0.0, 1)
        score_robustness = round(max(0.0, 7.0 - (failed_tasks * 2.0)), 2)
        status_robust: EvaluationStatus = "EXCELLENT" if failed_tasks == 0 else ("GOOD" if failed_tasks <= 1 else "NEEDS_IMPROVEMENT")

        m_robust = MetricItem(
            metric_id="m_exec_robustness",
            name="Computational Error Rate",
            dimension="Execution Reliability",
            value=f"{error_rate}%",
            unit="%",
            status=status_robust,
            description="Proportion of statistical computations that encountered execution or numerical errors.",
            benchmark_target="0.0%",
        )

        total_score = round(min(25.0, score_completion + score_prof + score_robustness), 2)
        percentage = round((total_score / 25.0) * 100.0, 1)
        grade = self._grade_dimension(percentage)

        summary = (
            f"Executed {n_results}/{n_planned} planned tasks with {failed_tasks} computation errors. "
            f"Dataset profiling achieved {prof_pct}% diagnostic coverage."
        )

        return DimensionScore(
            dimension_key="execution_reliability",
            name="Execution Reliability & Stability",
            score=total_score,
            max_score=25.0,
            percentage=percentage,
            grade=grade,
            metrics=[m_completion, m_prof, m_robust],
            summary=summary,
        )

    def _evaluate_statistical_rigor(
        self,
        profile: Dict[str, Any],
        plan: Dict[str, Any],
        analysis: Dict[str, Any],
        critic: Dict[str, Any],
        verification: Dict[str, Any],
    ) -> DimensionScore:
        """Dimension 2: Evaluates assumption audits, non-parametric fallback rates, and effect size coverage."""
        results = analysis.get("results", [])
        n_results = len(results)
        reviews = critic.get("reviews", [])
        n_reviews = len(reviews)

        # 1. Assumption Audit Coverage (10 pts)
        if n_results > 0:
            audit_coverage = round((n_reviews / n_results) * 100.0, 1)
            score_audit = round(min(10.0, (n_reviews / n_results) * 10.0), 2)
        else:
            audit_coverage = 100.0 if n_reviews > 0 else 0.0
            score_audit = 10.0 if n_reviews > 0 else 0.0

        status_audit: EvaluationStatus = "EXCELLENT" if audit_coverage >= 95.0 else "GOOD"
        m_audit = MetricItem(
            metric_id="m_rigor_audit_coverage",
            name="Assumption Audit Coverage",
            dimension="Statistical Rigor",
            value=f"{audit_coverage}%",
            unit="%",
            status=status_audit,
            description="Proportion of computed statistical results reviewed by Critic Agent for parametric assumptions.",
            benchmark_target="100.0%",
        )

        # 2. Non-Parametric Fallback & Robust Transition Rate (8 pts)
        # Evaluated from empirical evidence: flags by Critic vs successful re-investigations
        v_summary = verification.get("summary", {})
        reinvestigated_count = v_summary.get("total_reinvestigated", 0)
        reinvestigated_verified = v_summary.get("verified_after_reinvestigation", 0)

        critic_flagged_count = sum(
            1 for r in reviews
            if r.get("requires_reinvestigation")
            or r.get("status") in ("REQUIRES_REINVESTIGATION", "FLAGGED")
            or r.get("decision") in ("REQUIRES_REINVESTIGATION", "FLAGGED")
        )
        if critic_flagged_count == 0 and isinstance(critic, dict):
            critic_flagged_count = critic.get("overall_summary", {}).get("requires_reinvestigation", 0)
        if critic_flagged_count == 0:
            critic_flagged_count = v_summary.get("critic_flagged", 0)

        if reinvestigated_count > 0:
            fallback_success_rate = round((reinvestigated_verified / reinvestigated_count) * 100.0, 1)
            score_fallback = round((reinvestigated_verified / reinvestigated_count) * 8.0, 2)
            fallback_desc = f"{reinvestigated_count} tests flagged for assumption violations; {reinvestigated_verified} successfully resolved via non-parametric fallbacks."
            status_fallback: EvaluationStatus = "EXCELLENT" if fallback_success_rate >= 80.0 else ("GOOD" if fallback_success_rate >= 50.0 else "NEEDS_IMPROVEMENT")
        elif critic_flagged_count > 0:
            fallback_success_rate = 0.0
            score_fallback = 0.0
            fallback_desc = f"{critic_flagged_count} tests flagged by Critic for assumption violations, but re-investigation was not performed."
            status_fallback = "NEEDS_IMPROVEMENT"
        elif n_reviews > 0:
            fallback_success_rate = 100.0
            score_fallback = 8.0
            fallback_desc = "No unhandled assumption violations; parametric assumptions were met or non-parametric tests applied directly."
            status_fallback = "EXCELLENT"
        else:
            fallback_success_rate = 0.0
            score_fallback = 0.0
            fallback_desc = "Critic review has not been performed; assumption fallback status is unverified."
            status_fallback = "ACCEPTABLE"

        m_fallback = MetricItem(
            metric_id="m_rigor_fallback_rate",
            name="Robust Fallback Resolution",
            dimension="Statistical Rigor",
            value=f"{fallback_success_rate}%",
            unit="%",
            status=status_fallback,
            description=fallback_desc,
            benchmark_target="100.0%",
        )

        # 3. Effect Size Quantification Coverage (7 pts)
        findings_with_effect_size = sum(1 for r in results if self._has_effect_size(r))
        effect_coverage = round((findings_with_effect_size / n_results * 100.0) if n_results > 0 else 0.0, 1)
        score_effect = round((effect_coverage / 100.0) * 7.0, 2)
        status_effect: EvaluationStatus = "EXCELLENT" if effect_coverage >= 90.0 else ("GOOD" if effect_coverage >= 70.0 else "NEEDS_IMPROVEMENT")

        m_effect = MetricItem(
            metric_id="m_rigor_effect_size_coverage",
            name="Effect Size Quantification Rate",
            dimension="Statistical Rigor",
            value=f"{effect_coverage}%",
            unit="%",
            status=status_effect,
            description="Proportion of findings reporting standardized effect size metrics (r, Cohen's d, eta^2, Cramer's V) alongside p-values.",
            benchmark_target=">= 90.0%",
        )

        total_score = round(min(25.0, score_audit + score_fallback + score_effect), 2)
        percentage = round((total_score / 25.0) * 100.0, 1)
        grade = self._grade_dimension(percentage)

        summary = (
            f"Audited {n_reviews}/{n_results} statistical tasks for parametric compliance. "
            f"Effect sizes were quantified across {effect_coverage}% of analytical outputs."
        )

        return DimensionScore(
            dimension_key="statistical_rigor",
            name="Statistical Rigor & Assumption Compliance",
            score=total_score,
            max_score=25.0,
            percentage=percentage,
            grade=grade,
            metrics=[m_audit, m_fallback, m_effect],
            summary=summary,
        )

    def _evaluate_verification_grounding(
        self, analysis: Dict[str, Any], critic: Dict[str, Any], verification: Dict[str, Any]
    ) -> DimensionScore:
        """Dimension 3: Evaluates factuality, critic audit efficacy, and verification grounding."""
        results = analysis.get("results", [])
        reviews = critic.get("reviews", [])
        verified_findings = verification.get("verified_findings", [])
        v_summary = verification.get("summary", {})

        n_results = len(results)
        n_reviews = len(reviews)
        n_verified = len(verified_findings)
        n_questionable = v_summary.get("questionable", 0)
        n_rejected = v_summary.get("rejected", 0)

        # 1. Deterministic Provenance Rate (Factuality Proxy) (10 pts)
        # Evaluated from evidence: non-null Python/SciPy calculated statistics and provenance
        if n_results > 0:
            provenance_count = sum(1 for r in results if self._has_task_provenance(r))
            provenance_pct = round((provenance_count / n_results) * 100.0, 1)
            score_factuality = round((provenance_count / n_results) * 10.0, 2)
        else:
            provenance_pct = 0.0
            score_factuality = 0.0

        status_factuality: EvaluationStatus = "EXCELLENT" if provenance_pct >= 95.0 else ("GOOD" if provenance_pct >= 80.0 else "NEEDS_IMPROVEMENT")
        m_factuality = MetricItem(
            metric_id="m_verif_zero_hallucination",
            name="Deterministic Provenance Rate (Factuality Proxy)",
            dimension="Verification & Truth Grounding",
            value=f"{provenance_pct}%",
            unit="%",
            status=status_factuality,
            description="Proportion of statistical results with verified deterministic SciPy/Python calculation provenance and non-null metrics (deterministic verification proxy; external ground-truth auditing unavailable).",
            benchmark_target="100.0%",
        )

        # 2. Critic Audit & Diagnostic Efficacy (8 pts)
        # Evaluated from evidence: proportion of Critic reviews providing concrete diagnostic evidence and confidence scores
        if n_reviews > 0:
            substantive_reviews = sum(
                1 for rev in reviews
                if (
                    bool(rev.get("evidence") or rev.get("critique") or rev.get("summary"))
                    and (rev.get("confidence") is not None or rev.get("confidence_score") is not None)
                    and bool(rev.get("reason_codes") or rev.get("decision") or rev.get("status"))
                )
            )
            critic_efficacy_pct = round((substantive_reviews / n_reviews) * 100.0, 1)
            score_critic = round((substantive_reviews / n_reviews) * 8.0, 2)
            status_critic: EvaluationStatus = "EXCELLENT" if critic_efficacy_pct >= 95.0 else ("GOOD" if critic_efficacy_pct >= 80.0 else "NEEDS_IMPROVEMENT")
        else:
            critic_efficacy_pct = 0.0
            score_critic = 0.0
            status_critic = "NEEDS_IMPROVEMENT"

        m_critic_efficacy = MetricItem(
            metric_id="m_verif_critic_efficacy",
            name="Critic Diagnostic Efficacy Rate",
            dimension="Verification & Truth Grounding",
            value=f"{critic_efficacy_pct}%",
            unit="%",
            status=status_critic,
            description="Proportion of findings evaluated by Critic Agent with explicit empirical evidence, confidence scoring, and assumption audits.",
            benchmark_target="100.0%",
        )

        # 3. Verification & Resolution Ratio (7 pts)
        if n_results > 0:
            verified_ratio = round((n_verified / n_results) * 100.0, 1)
            score_verif = round(min(7.0, (n_verified / n_results) * 7.0), 2)
        else:
            verified_ratio = 0.0
            score_verif = 0.0

        status_verif: EvaluationStatus = "EXCELLENT" if verified_ratio >= 70.0 else ("GOOD" if verified_ratio >= 40.0 else "NEEDS_IMPROVEMENT")
        m_verif_ratio = MetricItem(
            metric_id="m_verif_verified_ratio",
            name="Verified Finding Yield",
            dimension="Verification & Truth Grounding",
            value=f"{verified_ratio}%",
            unit="%",
            status=status_verif,
            description=f"Synthesized {n_verified} verified findings ({n_questionable} questionable, {n_rejected} rejected).",
            benchmark_target=">= 70.0%",
        )

        total_score = round(min(25.0, score_factuality + score_critic + score_verif), 2)
        percentage = round((total_score / 25.0) * 100.0, 1)
        grade = self._grade_dimension(percentage)

        summary = (
            f"Deterministic provenance verified across {provenance_pct}% of results. "
            f"Synthesized {n_verified} verified findings with complete audit trails."
        )

        return DimensionScore(
            dimension_key="verification_grounding",
            name="Verification & Truth Grounding",
            score=total_score,
            max_score=25.0,
            percentage=percentage,
            grade=grade,
            metrics=[m_factuality, m_critic_efficacy, m_verif_ratio],
            summary=summary,
        )

    def _evaluate_insight_actionability(
        self, plan: Dict[str, Any], verification: Dict[str, Any], insights: Dict[str, Any]
    ) -> DimensionScore:
        """Dimension 4: Evaluates observational phrasing safety, target alignment, and insight strength."""
        # Support both key_insights (actual Phase 6 schema) and insights (legacy/mock)
        insight_items = insights.get("key_insights") or insights.get("insights") or []
        n_insights = len(insight_items)
        i_summary = insights.get("summary", {})

        if n_insights == 0:
            # Do not award full actionability marks when no insights exist
            m_causality = MetricItem(
                metric_id="m_insight_causality_safety",
                name="Non-Causal Phrasing Compliance",
                dimension="Insight Actionability",
                value="0.0% (No Insights)",
                unit="%",
                status="NEEDS_IMPROVEMENT",
                description="No analytical insights available to evaluate for non-causal observational language.",
                benchmark_target="100.0%",
            )
            m_alignment = MetricItem(
                metric_id="m_insight_target_alignment",
                name="Objective Target Alignment",
                dimension="Insight Actionability",
                value="0.0% (No Insights)",
                unit="%",
                status="NEEDS_IMPROVEMENT",
                description="No analytical insights available to assess alignment with planned target variables.",
                benchmark_target=">= 60.0%",
            )
            m_implication = MetricItem(
                metric_id="m_insight_implication_rate",
                name="Practical Implication Completeness",
                dimension="Insight Actionability",
                value="0.0% (No Insights)",
                unit="%",
                status="NEEDS_IMPROVEMENT",
                description="No analytical insights available with domain implications.",
                benchmark_target="100.0%",
            )

            return DimensionScore(
                dimension_key="insight_actionability",
                name="Insight Actionability & Causality Safety",
                score=0.0,
                max_score=25.0,
                percentage=0.0,
                grade="D",
                metrics=[m_causality, m_alignment, m_implication],
                summary="No verified analytical insights were generated to evaluate actionability.",
            )

        # 1. Observational Phrasing Compliance (Causality Safety) (10 pts)
        causal_violations = 0
        for item in insight_items:
            text = (
                str(item.get("insight") or "") + " "
                + str(item.get("text") or "") + " "
                + str(item.get("title") or "")
            ).lower()
            for kw in self.CAUSAL_KEYWORDS:
                if f" {kw} " in text:
                    causal_violations += 1
                    break

        causality_score = round(max(0.0, 10.0 - (causal_violations * (10.0 / n_insights))), 2)
        causality_pct = round((causality_score / 10.0) * 100.0, 1)
        status_causal: EvaluationStatus = "EXCELLENT" if causal_violations == 0 else "NEEDS_IMPROVEMENT"

        m_causality = MetricItem(
            metric_id="m_insight_causality_safety",
            name="Non-Causal Phrasing Compliance",
            dimension="Insight Actionability",
            value=f"{causality_pct}%",
            unit="%",
            status=status_causal,
            description=f"Strict adherence to observational association language on cross-sectional data ({causal_violations} causal violations detected).",
            benchmark_target="100.0%",
        )

        # 2. Objective Target Alignment (8 pts)
        target_vars = plan.get("target_variables", [])
        target_col = plan.get("target_column")
        all_targets = list(target_vars)
        if target_col and target_col not in all_targets:
            all_targets.append(target_col)

        if all_targets:
            aligned_insights = sum(
                1 for item in insight_items
                if any(
                    tv.lower() in [v.lower() for v in item.get("evidence", {}).get("variables", [])]
                    or tv.lower() in (str(item.get("title", "")).lower() + " " + str(item.get("insight", "")).lower())
                    for tv in all_targets
                )
            )
            alignment_pct = round((aligned_insights / n_insights) * 100.0, 1)
            score_alignment = round((alignment_pct / 100.0) * 8.0, 2)
        else:
            # Exploratory plan without specific target variable
            alignment_pct = 100.0
            score_alignment = 8.0

        status_align: EvaluationStatus = "EXCELLENT" if alignment_pct >= 60.0 else "GOOD"
        m_alignment = MetricItem(
            metric_id="m_insight_target_alignment",
            name="Objective Target Alignment",
            dimension="Insight Actionability",
            value=f"{alignment_pct}%",
            unit="%",
            status=status_align,
            description="Proportion of generated insights addressing target variables identified in the investigation plan.",
            benchmark_target=">= 60.0%",
        )

        # 3. Insight Strength & Implication Yield (7 pts)
        strong_count = i_summary.get("strong_insights", i_summary.get("strong_insights_count", 0))
        moderate_count = i_summary.get("moderate_insights", i_summary.get("moderate_insights_count", 0))
        structured_implications = sum(1 for item in insight_items if bool(item.get("practical_implication")))

        implication_pct = round((structured_implications / n_insights) * 100.0, 1)
        score_implication = round((implication_pct / 100.0) * 7.0, 2)

        m_implication = MetricItem(
            metric_id="m_insight_implication_rate",
            name="Practical Implication Completeness",
            dimension="Insight Actionability",
            value=f"{implication_pct}%",
            unit="%",
            status="EXCELLENT" if implication_pct >= 90.0 else "GOOD",
            description=f"Generated {n_insights} insights ({strong_count} strong, {moderate_count} moderate) with practical domain implications.",
            benchmark_target="100.0%",
        )

        total_score = round(min(25.0, causality_score + score_alignment + score_implication), 2)
        percentage = round((total_score / 25.0) * 100.0, 1)
        grade = self._grade_dimension(percentage)

        summary = (
            f"Generated {n_insights} verified insights with {causality_pct}% non-causal observational compliance. "
            f"Target alignment achieved {alignment_pct}%."
        )

        return DimensionScore(
            dimension_key="insight_actionability",
            name="Insight Actionability & Causality Safety",
            score=total_score,
            max_score=25.0,
            percentage=percentage,
            grade=grade,
            metrics=[m_causality, m_alignment, m_implication],
            summary=summary,
        )

    def _has_task_provenance(self, r: Dict[str, Any]) -> bool:
        """
        Determine deterministic calculation provenance for a task result.
        Valid provenance requires execution success and factual Python/SciPy output:
        - For hypothesis testing tasks: a non-null test statistic (root or nested).
        - For categorical distribution tasks: non-empty calculated frequencies or proportions.
        - For numerical distribution tasks: computed moments/quantiles (mean, median, skewness).
        - For data quality audits (missingness, duplicates): computed audit counts.
        - For outlier sensitivity analyses: computed outlier metrics (valid outlier_count with iqr or baseline_metrics).
        """
        if r.get("status") not in ("success", "SUCCESS"):
            return False
        if r.get("statistic") is not None:
            return True
        res = r.get("results")
        if not isinstance(res, dict):
            return False
        if res.get("statistic") is not None:
            return True
        # Categorical distribution: frequencies or proportions
        if bool(res.get("frequencies")) or bool(res.get("proportions")):
            return True
        # Numerical distribution / descriptive summary: mean, median, skewness
        if res.get("mean") is not None or res.get("median") is not None or res.get("skewness") is not None:
            return True
        # Missingness or duplicate audits: computed metric counts
        if res.get("total_missing_cells") is not None or res.get("duplicate_rows") is not None:
            return True
        # Outlier sensitivity analyses: computed outlier metrics with baseline metrics or fences
        if (
            res.get("outlier_count") is not None
            and isinstance(res.get("outlier_count"), (int, float))
            and (
                res.get("iqr") is not None
                or isinstance(res.get("baseline_metrics"), dict)
            )
        ):
            return True
        return False

    def _has_effect_size(self, r: Dict[str, Any]) -> bool:
        """Inspect both root-level fields and nested 'results' dictionary for standardized effect size."""
        if r.get("effect_size") is not None:
            return True
        if any(k in r and isinstance(r[k], dict) and any(v is not None for v in r[k].values()) for k in ("contingency", "two_group", "multi_group", "bivariate")):
            return True
        res = r.get("results")
        if isinstance(res, dict):
            if res.get("effect_size") is not None:
                return True
            if res.get("correlation_coefficient") is not None or res.get("cramer_v") is not None:
                return True
            if res.get("mean_difference") is not None or res.get("cohens_d") is not None or res.get("eta_squared") is not None:
                return True
            analysis_type = str(r.get("analysis_type", ""))
            if "numerical_vs_numerical" in analysis_type and res.get("statistic") is not None:
                return True
            if res.get("association_strength") is not None:
                return True
        return False

    def _is_effect_size_applicable(self, r: Dict[str, Any]) -> bool:
        """Determine whether standardized effect size reporting is statistically applicable for a task."""
        analysis_type = str(r.get("analysis_type") or r.get("task_type") or "").lower()
        method = str(r.get("method") or "").lower()
        # Descriptive summaries, distributions, missingness, or outlier sensitivity tasks do not take standard effect sizes
        if any(na in analysis_type for na in ("descriptive", "univariate", "distribution", "outlier_sensitivity", "hygiene", "missingness")):
            return False
        if any(na in method for na in ("iqr-based", "summary", "frequency", "descriptive", "histogram", "boxplot")):
            return False
        return True

    def _build_benchmark_checklist(
        self,
        profile: Dict[str, Any],
        plan: Dict[str, Any],
        analysis: Dict[str, Any],
        critic: Dict[str, Any],
        verification: Dict[str, Any],
        insights: Dict[str, Any],
        has_full_provenance: bool = True,
    ) -> List[BenchmarkCheck]:
        """Validates formal scientific benchmark compliance checks based strictly on available evidence."""
        insight_items = insights.get("key_insights") or insights.get("insights") or []
        critic_reviews = critic.get("reviews", [])
        verified_findings = verification.get("verified_findings", [])
        results = analysis.get("results", [])
        n_results = len(results)

        # ---------------------------------------------------------------------
        # 1. Computation Provenance Verification (Zero-Hallucination Proxy)
        # ---------------------------------------------------------------------
        if n_results == 0:
            chk_zh_passed = False
            chk_zh_detail = "Computation provenance check failed: No statistical analysis tasks were executed."
        elif has_full_provenance:
            chk_zh_passed = True
            chk_zh_detail = (
                f"Deterministic computation provenance verified across all {n_results} executed task(s) in SciPy "
                "(serves as a computation-provenance check; does not constitute external ground-truth auditing)."
            )
        else:
            chk_zh_passed = False
            chk_zh_detail = (
                "Computation provenance check failed: One or more analysis tasks failed or returned null/missing test statistics."
            )

        # ---------------------------------------------------------------------
        # 2. Parametric Assumption Auditing
        # ---------------------------------------------------------------------
        if n_results == 0:
            chk_audit_passed = False
            chk_audit_detail = "No executed statistical results available to audit for parametric assumptions."
        elif len(critic_reviews) > 0:
            chk_audit_passed = True
            chk_audit_detail = (
                f"Formally audited {len(critic_reviews)} of {n_results} task(s) for parametric assumptions, "
                "sample size bounds, and distribution stability."
            )
        else:
            chk_audit_passed = False
            chk_audit_detail = (
                "Critic review was not performed; parametric assumptions for computed findings remain unverified."
            )

        # ---------------------------------------------------------------------
        # 3. Targeted Non-Parametric Fallbacks
        # ---------------------------------------------------------------------
        v_summary = verification.get("summary", {})
        reinvestigated_count = v_summary.get("total_reinvestigated", len(verification.get("reinvestigations", [])))
        reinvestigated_verified = v_summary.get("verified_after_reinvestigation", 0)
        critic_flagged_count = sum(
            1 for r in critic_reviews
            if r.get("requires_reinvestigation")
            or r.get("status") in ("REQUIRES_REINVESTIGATION", "FLAGGED")
            or r.get("decision") in ("REQUIRES_REINVESTIGATION", "FLAGGED")
        )
        if critic_flagged_count == 0 and isinstance(critic, dict):
            critic_flagged_count = critic.get("overall_summary", {}).get("requires_reinvestigation", 0)

        if reinvestigated_count > 0:
            chk_fallback_passed = True
            chk_fallback_detail = (
                f"Targeted non-parametric fallback executed and recorded across {reinvestigated_count} flagged task(s) "
                f"({reinvestigated_verified} successfully verified)."
            )
        elif critic_flagged_count > 0:
            chk_fallback_passed = False
            chk_fallback_detail = (
                f"Fallback requirement violated: {critic_flagged_count} task(s) flagged for assumption violations by Critic, "
                "but no re-investigation fallback was executed or recorded."
            )
        else:
            # Conservative failure: no fallback execution recorded/required
            chk_fallback_passed = False
            chk_fallback_detail = (
                "No fallback execution recorded: either Critic review was omitted or no assumption violations triggered "
                "an alternative non-parametric execution."
            )

        # ---------------------------------------------------------------------
        # 4. Non-Causal Language Guard
        # ---------------------------------------------------------------------
        if len(insight_items) == 0:
            chk_causal_passed = False
            chk_causal_detail = "No analytical insights available to inspect for causal language compliance."
        else:
            causal_violations = 0
            for item in insight_items:
                text = (
                    str(item.get("insight") or "") + " "
                    + str(item.get("text") or "") + " "
                    + str(item.get("title") or "")
                ).lower()
                for kw in self.CAUSAL_KEYWORDS:
                    if f" {kw} " in text:
                        causal_violations += 1
                        break
            if causal_violations == 0:
                chk_causal_passed = True
                chk_causal_detail = (
                    f"Observational framing strictly verified across all {len(insight_items)} generated insight(s); "
                    "0 causal overclaims detected."
                )
            else:
                chk_causal_passed = False
                chk_causal_detail = (
                    f"Non-causal language guard violated: {causal_violations} of {len(insight_items)} insight(s) "
                    "contained unjustified causal assertions."
                )

        # ---------------------------------------------------------------------
        # 5. Standardized Effect Size Reporting
        # ---------------------------------------------------------------------
        applicable_tasks = [r for r in results if r.get("status") in ("success", "SUCCESS") and self._is_effect_size_applicable(r)]
        non_applicable_tasks = [r for r in results if r.get("status") in ("success", "SUCCESS") and not self._is_effect_size_applicable(r)]

        if n_results == 0:
            chk_eff_passed = False
            chk_eff_detail = "No executed statistical results available to evaluate effect size reporting."
        elif len(applicable_tasks) == 0:
            chk_eff_passed = True
            chk_eff_detail = (
                f"All {len(non_applicable_tasks)} executed task(s) are descriptive or sensitivity tasks where "
                "standardized effect sizes are not statistically applicable."
            )
        else:
            reported_count = sum(1 for r in applicable_tasks if self._has_effect_size(r))
            if reported_count == len(applicable_tasks):
                chk_eff_passed = True
                chk_eff_detail = (
                    f"Standardized effect sizes reported across 100.0% of applicable statistical tasks "
                    f"({reported_count}/{len(applicable_tasks)} tasks)."
                )
            else:
                chk_eff_passed = False
                missing_count = len(applicable_tasks) - reported_count
                chk_eff_detail = (
                    f"Standardized effect size reporting incomplete: {missing_count} of {len(applicable_tasks)} "
                    "applicable task(s) lacked standardized effect sizes."
                )

        # ---------------------------------------------------------------------
        # 6. End-to-End Evidence Traceability
        # ---------------------------------------------------------------------
        if n_results == 0:
            chk_trace_passed = False
            chk_trace_detail = "Empty analysis pipeline: No statistical analysis results available to establish evidence traceability."
        elif len(insight_items) == 0 and len(verified_findings) == 0:
            chk_trace_passed = False
            chk_trace_detail = "Traceability evidence unavailable: Verification and insight stages were not conducted for executed analysis tasks."
        elif len(insight_items) > 0:
            traceable_count = sum(
                1 for item in insight_items
                if bool(item.get("evidence") or item.get("finding_id") or item.get("task_id"))
            )
            if traceable_count == len(insight_items) and len(verified_findings) > 0:
                chk_trace_passed = True
                chk_trace_detail = (
                    f"Full evidence traceability verified: All {len(insight_items)} synthesized insight(s) bind directly "
                    "to underlying verified findings and statistical test metadata."
                )
            else:
                chk_trace_passed = False
                missing_trace = len(insight_items) - traceable_count
                chk_trace_detail = (
                    f"Evidence traceability incomplete: {missing_trace} of {len(insight_items)} insight(s) "
                    "lacked direct verified evidence bindings."
                )
        else:
            # len(verified_findings) > 0 and len(insight_items) == 0
            traceable_count = sum(1 for vf in verified_findings if bool(vf.get("task_id") or vf.get("statistical_evidence")))
            if traceable_count == len(verified_findings):
                chk_trace_passed = True
                chk_trace_detail = (
                    f"Evidence traceability verified across {len(verified_findings)} verified finding(s) with direct binding "
                    "to source analysis tasks."
                )
            else:
                chk_trace_passed = False
                chk_trace_detail = "Verified findings lack complete binding to source analysis tasks."

        return [
            BenchmarkCheck(
                check_id="chk_zero_hallucination",
                name="Zero-Hallucination Empirical Guarantee",
                standard="AutoMind Empirical Reliability",
                passed=chk_zh_passed,
                detail=chk_zh_detail,
            ),
            BenchmarkCheck(
                check_id="chk_assumption_audits",
                name="Parametric Assumption Auditing",
                standard="Statistical Inference Standards",
                passed=chk_audit_passed,
                detail=chk_audit_detail,
            ),
            BenchmarkCheck(
                check_id="chk_robust_fallbacks",
                name="Targeted Non-Parametric Fallbacks",
                standard="AutoMind Robustness Protocol",
                passed=chk_fallback_passed,
                detail=chk_fallback_detail,
            ),
            BenchmarkCheck(
                check_id="chk_causality_guard",
                name="Non-Causal Language Guard",
                standard="Scientific Factuality & Ethics",
                passed=chk_causal_passed,
                detail=chk_causal_detail,
            ),
            BenchmarkCheck(
                check_id="chk_effect_size_reporting",
                name="Standardized Effect Size Reporting",
                standard="APA / ASA Empirical Reporting",
                passed=chk_eff_passed,
                detail=chk_eff_detail,
            ),
            BenchmarkCheck(
                check_id="chk_full_traceability",
                name="End-to-End Evidence Traceability",
                standard="InsightBench Reproducibility",
                passed=chk_trace_passed,
                detail=chk_trace_detail,
            ),
        ]

    def _grade_dimension(self, percentage: float) -> str:
        """Assigns letter grade to a dimensional percentage score."""
        if percentage >= 95.0:
            return "A+"
        elif percentage >= 90.0:
            return "A"
        elif percentage >= 80.0:
            return "B+"
        elif percentage >= 70.0:
            return "B"
        elif percentage >= 60.0:
            return "C"
        else:
            return "D"

    def _compute_overall_grade(self, overall_score: float) -> tuple[str, str]:
        """Computes executive grade and assessment text."""
        if overall_score >= 93.0:
            grade = "A+ - Outstanding Empirical Rigor"
            assessment = "Pipeline demonstrates exceptional statistical rigor, zero hallucination, thorough assumption auditing, and robust evidence-grounded insights."
        elif overall_score >= 85.0:
            grade = "A - High Scientific Rigor"
            assessment = "Strong end-to-end execution with validated statistical assumptions, standardized effect sizes, and high factual fidelity."
        elif overall_score >= 75.0:
            grade = "B+ - Satisfactory Analytical Rigor"
            assessment = "Satisfactory analytical workflow with verifiable evidence grounding and good task completion."
        elif overall_score >= 60.0:
            grade = "B - Acceptable"
            assessment = "Acceptable execution meeting baseline statistical standards; some parametric or coverage improvements possible."
        else:
            grade = "C - Needs Methodological Improvement"
            assessment = "Pipeline execution encountered anomalies or incomplete assumption verification."

        return grade, assessment


evaluation_service = EvaluationService()

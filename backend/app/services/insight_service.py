"""
backend/app/services/insight_service.py
=======================================
Service layer for DataInvest AI — Phase 6: Insight Generation.
Converts Phase 5 verified statistical findings into clear, human-readable,
non-causal analytical insights with full evidence traceability and deterministic
strength classification.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from app.schemas.insight_schema import (
    Insight,
    InsightEvidence,
    InsightRequest,
    InsightResponse,
    InsightStrength,
    InsightSummary,
    InterpretationType,
    UnresolvedFinding,
)
from app.schemas.verification_schema import VerifiedFinding

logger = logging.getLogger(__name__)


class InsightService:
    """
    Deterministic Insight Generation Service.
    Translates structured empirical findings into clear, non-causal insights
    without recalculating or fabricating any statistical numbers.
    """

    def generate_insights(self, request: InsightRequest) -> InsightResponse:
        """
        Main entrypoint: extracts verified findings from Phase 5 response,
        applies deterministic translation templates, deduplicates overlapping
        variable relationships, orders key insights by empirical strength,
        and segregates questionable/rejected findings.
        """
        dataset_id = request.dataset_id or "dataset.csv"
        verified_findings: list[VerifiedFinding] = []
        questionable_findings: list[VerifiedFinding] = []
        rejected_findings: list[VerifiedFinding] = []

        # 1. Extract findings from request payload
        if request.verification_response:
            v_res = request.verification_response
            dataset_id = v_res.dataset_id or dataset_id
            verified_findings = list(v_res.verified_findings or [])
            questionable_findings = list(v_res.questionable_findings or [])
            rejected_findings = list(v_res.rejected_findings or [])

            # Fallback: inspect all_findings if verified_findings list was empty
            if not verified_findings and v_res.all_findings:
                for f in v_res.all_findings:
                    if f.status in ("VERIFIED", "VERIFIED_AFTER_REINVESTIGATION"):
                        verified_findings.append(f)
                    elif f.status == "QUESTIONABLE":
                        questionable_findings.append(f)
                    elif f.status == "REJECTED":
                        rejected_findings.append(f)

        elif request.verified_findings:
            for f in request.verified_findings:
                if f.status in ("VERIFIED", "VERIFIED_AFTER_REINVESTIGATION"):
                    verified_findings.append(f)
                elif f.status == "QUESTIONABLE":
                    questionable_findings.append(f)
                elif f.status == "REJECTED":
                    rejected_findings.append(f)

        # 2. Process Unresolved Findings (Questionable & Rejected)
        unresolved_list: list[UnresolvedFinding] = []
        for qf in questionable_findings:
            unresolved_list.append(
                UnresolvedFinding(
                    finding_id=qf.finding_id,
                    task_id=qf.task_id,
                    status=qf.status,
                    title=f"Questionable Finding: {', '.join(qf.variables) if qf.variables else qf.task_id}",
                    summary=qf.summary or qf.critic_assessment or "Retained with methodological caveats.",
                    variables=qf.variables,
                    limitations=qf.limitations or [qf.verification_reason],
                )
            )

        for rf in rejected_findings:
            unresolved_list.append(
                UnresolvedFinding(
                    finding_id=rf.finding_id,
                    task_id=rf.task_id,
                    status=rf.status,
                    title=f"Excluded Finding: {', '.join(rf.variables) if rf.variables else rf.task_id}",
                    summary=rf.summary or rf.critic_assessment or "Excluded due to critical assumption violations or failure.",
                    variables=rf.variables,
                    limitations=rf.limitations or [rf.verification_reason],
                )
            )

        # 3. Deduplicate and Generate Verified Insights
        key_insights: list[Insight] = []
        processed_pairs: set[str] = set()

        for idx, vf in enumerate(verified_findings, start=1):
            try:
                # Deduplication check for identical variable sets
                var_key = "_".join(sorted(vf.variables)) if vf.variables else vf.task_id
                if var_key and var_key in processed_pairs:
                    # Duplicate variable relationship already covered by a higher/primary test
                    continue
                if var_key:
                    processed_pairs.add(var_key)

                insight_obj = self._translate_finding_to_insight(vf, insight_index=len(key_insights) + 1)
                if insight_obj:
                    key_insights.append(insight_obj)

            except Exception as e:
                logger.warning(f"Failed to generate insight for finding {vf.finding_id}: {e}", exc_info=True)
                # Failure isolation: continue processing remaining findings

        # 4. Rank Key Insights by Priority Score
        key_insights.sort(key=lambda x: x.priority_score, reverse=True)

        # 5. Compute Dynamic Summary Metrics
        strong_count = sum(1 for i in key_insights if i.strength == "STRONG")
        moderate_count = sum(1 for i in key_insights if i.strength == "MODERATE")
        weak_count = sum(1 for i in key_insights if i.strength == "WEAK")
        not_assessed_count = sum(1 for i in key_insights if i.strength == "NOT_ASSESSED")
        top_title = key_insights[0].title if key_insights else None

        summary = InsightSummary(
            total_verified_findings=len(verified_findings),
            total_insights=len(key_insights),
            strong_insights=strong_count,
            moderate_insights=moderate_count,
            weak_insights=weak_count,
            not_assessed_insights=not_assessed_count,
            unresolved_count=len(unresolved_list),
            top_insight_title=top_title,
        )

        return InsightResponse(
            dataset_id=dataset_id,
            summary=summary,
            key_insights=key_insights,
            unresolved_findings=unresolved_list,
            phase_version="6.0",
        )

    def _translate_finding_to_insight(self, vf: VerifiedFinding, insight_index: int) -> Optional[Insight]:
        """Translates a single VerifiedFinding into a strongly typed Insight."""
        method = (vf.method or "").lower()
        evidence_dict = vf.statistical_evidence or {}
        variables = vf.variables or []

        # Determine method family
        if any(m in method for m in ("correlation", "pearson", "spearman")):
            return self._build_correlation_insight(vf, evidence_dict, variables, insight_index)

        elif any(m in method for m in ("t_test", "mann_whitney", "students_t", "welch")):
            return self._build_two_group_insight(vf, evidence_dict, variables, insight_index)

        elif any(m in method for m in ("anova", "kruskal")):
            return self._build_multi_group_insight(vf, evidence_dict, variables, insight_index)

        elif "chi_square" in method or "contingency" in method:
            return self._build_contingency_insight(vf, evidence_dict, variables, insight_index)

        elif "outlier" in method:
            return self._build_outlier_insight(vf, evidence_dict, variables, insight_index)

        elif "distribution" in method:
            return self._build_distribution_insight(vf, evidence_dict, variables, insight_index)

        else:
            return self._build_generic_insight(vf, evidence_dict, variables, insight_index)

    # ──────────────────────────────────────────────────────────────────────────
    # Specific Translation Handlers
    # ──────────────────────────────────────────────────────────────────────────

    def _build_correlation_insight(
        self, vf: VerifiedFinding, ev: dict[str, Any], variables: list[str], idx: int
    ) -> Insight:
        var_a = ev.get("column_a") or (variables[0] if len(variables) > 0 else "Variable A")
        var_b = ev.get("column_b") or (variables[1] if len(variables) > 1 else "Variable B")

        r_val = ev.get("statistic") or ev.get("correlation") or ev.get("r")
        p_val = ev.get("p_value")
        n_val = ev.get("sample_size") or ev.get("n")

        # Format p-value string
        p_str = self._format_p_str(p_val)
        stat_name = "Spearman rho" if "spearman" in vf.method.lower() else "Pearson r"

        r_num = float(r_val) if r_val is not None else 0.0
        is_sig = p_val is not None and p_val < 0.05

        # Direction and Strength
        direction = "positive" if r_num > 0.05 else ("negative" if r_num < -0.05 else "neutral")
        abs_r = abs(r_num)

        if abs_r >= 0.50:
            strength: InsightStrength = "STRONG"
            strength_desc = "strong"
        elif abs_r >= 0.30:
            strength = "MODERATE"
            strength_desc = "moderate"
        elif abs_r >= 0.10:
            strength = "WEAK"
            strength_desc = "weak"
        else:
            strength = "WEAK"
            strength_desc = "negligible"

        # Phrasing
        if is_sig and direction != "neutral":
            title = f"{strength_desc.capitalize()} {direction.capitalize()} Association: {var_a} and {var_b}"
            trend_note = (
                f"Higher observed values of {var_a} tend to correspond with higher observed values of {var_b}."
                if direction == "positive"
                else f"Higher observed values of {var_a} tend to correspond with lower observed values of {var_b}."
            )
            insight_text = (
                f"{var_a} and {var_b} demonstrate a statistically significant {strength_desc} {direction} "
                f"association in the analyzed data ({stat_name} = {r_num:.3f}, {p_str}, n = {n_val or 'N/A'}). {trend_note}"
            )
        elif is_sig and direction == "neutral":
            title = f"Negligible Correlation: {var_a} and {var_b}"
            insight_text = (
                f"{var_a} and {var_b} show a statistically significant but practically negligible association "
                f"({stat_name} = {r_num:.3f}, {p_str}, n = {n_val or 'N/A'})."
            )
        else:
            title = f"No Significant Association: {var_a} and {var_b}"
            insight_text = (
                f"{var_a} and {var_b} do not show a statistically significant linear association "
                f"({stat_name} = {r_num:.3f}, {p_str}, n = {n_val or 'N/A'})."
            )


        # Practical Implication
        if is_sig and abs_r < 0.30:
            implication = "While statistically significant due to sample size, the empirical correlation magnitude is weak, indicating substantial variance remains unexplained."
        elif is_sig and abs_r >= 0.50:
            implication = "The substantial correlation magnitude indicates a strong empirical alignment between these two features."
        else:
            implication = "The observed relationship should be interpreted in context with other interacting features."

        # Re-investigation info
        reinvestigated_method = vf.reinvestigation_record.reinvestigated_method if vf.reinvestigation_record else None
        comparison_summary = vf.reinvestigation_record.comparison.summary if (vf.reinvestigation_record and vf.reinvestigation_record.comparison) else None

        limitations = list(vf.limitations or [])
        limitations.append("Observational data only; established correlation does not imply causal relationship or direction.")
        if n_val and n_val < 30:
            limitations.append(f"Small sample size (n = {n_val}) limits generalization power.")

        priority_score = self._compute_priority_score(vf.confidence, is_sig, strength, vf.reinvestigated)

        evidence = InsightEvidence(
            method=vf.method,
            variables=variables,
            statistic=r_num,
            statistic_name=stat_name,
            p_value=p_val,
            effect_size=abs_r,
            effect_size_label=strength_desc,
            sample_size=n_val,
            reinvestigated=vf.reinvestigated,
            reinvestigated_method=reinvestigated_method,
            comparison_summary=comparison_summary,
        )

        return Insight(
            insight_id=f"insight_{idx}",
            source_task_id=vf.task_id,
            title=title,
            insight=insight_text,
            interpretation_type="ASSOCIATION",
            strength=strength,
            verification_status=vf.status,
            confidence_score=vf.confidence,
            is_statistically_significant=is_sig,
            evidence=evidence,
            limitations=limitations,
            practical_implication=implication,
            priority_score=priority_score,
        )

    def _build_two_group_insight(
        self, vf: VerifiedFinding, ev: dict[str, Any], variables: list[str], idx: int
    ) -> Insight:
        group_col = ev.get("group_column") or (variables[0] if len(variables) > 0 else "Group Variable")
        val_col = ev.get("value_column") or (variables[1] if len(variables) > 1 else "Value Variable")

        p_val = ev.get("p_value")
        stat_val = ev.get("statistic") or ev.get("t_statistic") or ev.get("u_statistic")
        n_val = ev.get("sample_size") or ev.get("n")
        d_val = ev.get("cohens_d") or ev.get("effect_size")
        group_stats = ev.get("group_metrics") or ev.get("group_comparison") or {}

        p_str = self._format_p_str(p_val)
        stat_name = "Mann-Whitney U" if "mann_whitney" in vf.method else "Welch t"
        is_sig = p_val is not None and p_val < 0.05

        # Strength from Cohen's d
        d_num = abs(float(d_val)) if d_val is not None else None
        if d_num is not None:
            if d_num >= 0.80:
                strength: InsightStrength = "STRONG"
                strength_desc = "large"
            elif d_num >= 0.50:
                strength = "MODERATE"
                strength_desc = "moderate"
            elif d_num >= 0.20:
                strength = "WEAK"
                strength_desc = "small"
            else:
                strength = "WEAK"
                strength_desc = "negligible"
        else:
            strength = "NOT_ASSESSED"
            strength_desc = "unassessed"

        # Check group comparison details
        g1_name = ev.get("group1") or "Group 1"
        g2_name = ev.get("group2") or "Group 2"
        g1_mean = ev.get("group1_mean")
        g2_mean = ev.get("group2_mean")

        if is_sig:
            if g1_mean is not None and g2_mean is not None:
                higher_g = g1_name if g1_mean >= g2_mean else g2_name
                higher_v = max(g1_mean, g2_mean)
                lower_g = g2_name if g1_mean >= g2_mean else g1_name
                lower_v = min(g1_mean, g2_mean)
                title = f"Significant Difference in {val_col} between {higher_g} and {lower_g}"
                insight_text = (
                    f"Observed values of {val_col} differ significantly across {group_col} groups ({stat_name}, {p_str}, n = {n_val or 'N/A'}). "
                    f"Group '{higher_g}' shows a higher observed mean ({higher_v:.2f}) compared to '{lower_g}' ({lower_v:.2f})."
                )
            else:
                title = f"Significant Group Difference in {val_col} across {group_col}"
                insight_text = (
                    f"Observed values of {val_col} exhibit a statistically significant difference across {group_col} groups "
                    f"({stat_name}, {p_str}, n = {n_val or 'N/A'})."
                )
        else:
            title = f"No Significant Difference in {val_col} across {group_col}"
            insight_text = (
                f"No statistically significant difference in {val_col} was observed between {group_col} groups "
                f"({stat_name}, {p_str}, n = {n_val or 'N/A'})."
            )

        limitations = list(vf.limitations or [])
        limitations.append("Group differences reflect observed distributions and do not prove causal interventions.")

        priority_score = self._compute_priority_score(vf.confidence, is_sig, strength, vf.reinvestigated)

        evidence = InsightEvidence(
            method=vf.method,
            variables=variables,
            statistic=float(stat_val) if stat_val is not None else None,
            statistic_name=stat_name,
            p_value=p_val,
            effect_size=d_num,
            effect_size_label=strength_desc,
            sample_size=n_val,
            group_metrics=group_stats or {"group1": g1_name, "group2": g2_name, "g1_mean": g1_mean, "g2_mean": g2_mean},
            reinvestigated=vf.reinvestigated,
            reinvestigated_method=vf.reinvestigation_record.reinvestigated_method if vf.reinvestigation_record else None,
            comparison_summary=vf.reinvestigation_record.comparison.summary if (vf.reinvestigation_record and vf.reinvestigation_record.comparison) else None,
        )

        return Insight(
            insight_id=f"insight_{idx}",
            source_task_id=vf.task_id,
            title=title,
            insight=insight_text,
            interpretation_type="GROUP_DIFFERENCE",
            strength=strength,
            verification_status=vf.status,
            confidence_score=vf.confidence,
            is_statistically_significant=is_sig,
            evidence=evidence,
            limitations=limitations,
            practical_implication=f"Effect size indicates a {strength_desc} distributional shift between compared groups.",
            priority_score=priority_score,
        )

    def _build_multi_group_insight(
        self, vf: VerifiedFinding, ev: dict[str, Any], variables: list[str], idx: int
    ) -> Insight:
        group_col = variables[0] if len(variables) > 0 else "Category"
        val_col = variables[1] if len(variables) > 1 else "Value"

        p_val = ev.get("p_value")
        stat_val = ev.get("statistic") or ev.get("f_statistic") or ev.get("h_statistic")
        n_val = ev.get("sample_size") or ev.get("n")
        eta_val = ev.get("eta_squared") or ev.get("effect_size")
        n_groups = ev.get("n_groups") or ev.get("num_groups") or "multiple"

        p_str = self._format_p_str(p_val)
        stat_name = "Kruskal-Wallis H" if "kruskal" in vf.method else "ANOVA F"
        is_sig = p_val is not None and p_val < 0.05

        eta_num = float(eta_val) if eta_val is not None else None
        if eta_num is not None:
            if eta_num >= 0.14:
                strength: InsightStrength = "STRONG"
                strength_desc = "large"
            elif eta_num >= 0.06:
                strength = "MODERATE"
                strength_desc = "moderate"
            elif eta_num >= 0.01:
                strength = "WEAK"
                strength_desc = "small"
            else:
                strength = "WEAK"
                strength_desc = "negligible"
        else:
            strength = "NOT_ASSESSED"
            strength_desc = "unassessed"

        if is_sig:
            title = f"Significant Variance in {val_col} across {group_col} Categories"
            insight_text = (
                f"Statistically significant variance in {val_col} was detected across the {n_groups} categories of {group_col} "
                f"({stat_name} = {float(stat_val):.2f} if stat_val is not None else 'N/A', {p_str}, n = {n_val or 'N/A'})."
            )
        else:
            title = f"Homogeneous Distribution of {val_col} across {group_col}"
            insight_text = (
                f"No statistically significant difference in {val_col} was found across categories of {group_col} "
                f"({stat_name}, {p_str}, n = {n_val or 'N/A'})."
            )

        limitations = list(vf.limitations or [])
        limitations.append("Multi-group ANOVA/Kruskal-Wallis confirms overall between-group variance; pairwise post-hoc tests recommended for specific pairwise contrasts.")

        priority_score = self._compute_priority_score(vf.confidence, is_sig, strength, vf.reinvestigated)

        evidence = InsightEvidence(
            method=vf.method,
            variables=variables,
            statistic=float(stat_val) if stat_val is not None else None,
            statistic_name=stat_name,
            p_value=p_val,
            effect_size=eta_num,
            effect_size_label=strength_desc,
            sample_size=n_val,
            reinvestigated=vf.reinvestigated,
            reinvestigated_method=vf.reinvestigation_record.reinvestigated_method if vf.reinvestigation_record else None,
            comparison_summary=vf.reinvestigation_record.comparison.summary if (vf.reinvestigation_record and vf.reinvestigation_record.comparison) else None,
        )

        return Insight(
            insight_id=f"insight_{idx}",
            source_task_id=vf.task_id,
            title=title,
            insight=insight_text,
            interpretation_type="GROUP_DIFFERENCE",
            strength=strength,
            verification_status=vf.status,
            confidence_score=vf.confidence,
            is_statistically_significant=is_sig,
            evidence=evidence,
            limitations=limitations,
            practical_implication=f"Category membership accounts for a {strength_desc} portion of variance in {val_col}.",
            priority_score=priority_score,
        )

    def _build_contingency_insight(
        self, vf: VerifiedFinding, ev: dict[str, Any], variables: list[str], idx: int
    ) -> Insight:
        var_a = variables[0] if len(variables) > 0 else "Variable A"
        var_b = variables[1] if len(variables) > 1 else "Variable B"

        p_val = ev.get("p_value")
        chi2_val = ev.get("statistic") or ev.get("chi2_statistic") or ev.get("chi2")
        v_val = ev.get("cramers_v") or ev.get("effect_size")
        n_val = ev.get("sample_size") or ev.get("n")

        p_str = self._format_p_str(p_val)
        is_sig = p_val is not None and p_val < 0.05

        v_num = float(v_val) if v_val is not None else None
        if v_num is not None:
            if v_num >= 0.35:
                strength: InsightStrength = "STRONG"
                strength_desc = "strong"
            elif v_num >= 0.15:
                strength = "MODERATE"
                strength_desc = "moderate"
            elif v_num >= 0.05:
                strength = "WEAK"
                strength_desc = "weak"
            else:
                strength = "WEAK"
                strength_desc = "negligible"
        else:
            strength = "NOT_ASSESSED"
            strength_desc = "unassessed"

        if is_sig:
            title = f"Categorical Association: {var_a} and {var_b}"
            insight_text = (
                f"{var_a} and {var_b} demonstrate a statistically significant categorical association "
                f"(Chi-Square = {float(chi2_val):.2f} if chi2_val is not None else 'N/A', {p_str}, Cramer's V = {v_num:.3f} if v_num is not None else 'N/A', n = {n_val or 'N/A'}). "
                f"The distribution of {var_a} categories varies substantially according to {var_b}."
            )
        else:
            title = f"Categorical Independence: {var_a} and {var_b}"
            insight_text = (
                f"{var_a} and {var_b} appear statistically independent in the analyzed dataset "
                f"({p_str}, n = {n_val or 'N/A'})."
            )


        limitations = list(vf.limitations or [])
        limitations.append("Contingency tables identify co-occurrence frequencies without directional dependence.")

        priority_score = self._compute_priority_score(vf.confidence, is_sig, strength, vf.reinvestigated)

        evidence = InsightEvidence(
            method=vf.method,
            variables=variables,
            statistic=float(chi2_val) if chi2_val is not None else None,
            statistic_name="Chi-Square",
            p_value=p_val,
            effect_size=v_num,
            effect_size_label=strength_desc,
            sample_size=n_val,
            reinvestigated=vf.reinvestigated,
            reinvestigated_method=vf.reinvestigation_record.reinvestigated_method if vf.reinvestigation_record else None,
            comparison_summary=vf.reinvestigation_record.comparison.summary if (vf.reinvestigation_record and vf.reinvestigation_record.comparison) else None,
        )

        return Insight(
            insight_id=f"insight_{idx}",
            source_task_id=vf.task_id,
            title=title,
            insight=insight_text,
            interpretation_type="CATEGORICAL_ASSOCIATION",
            strength=strength,
            verification_status=vf.status,
            confidence_score=vf.confidence,
            is_statistically_significant=is_sig,
            evidence=evidence,
            limitations=limitations,
            practical_implication=f"The cross-tabulation reveals {strength_desc} categorical dependency across groups.",
            priority_score=priority_score,
        )

    def _build_outlier_insight(
        self, vf: VerifiedFinding, ev: dict[str, Any], variables: list[str], idx: int
    ) -> Insight:
        var_a = variables[0] if variables else "Analyzed Feature"
        count = ev.get("outlier_count") or ev.get("count") or 0
        pct = ev.get("outlier_percentage") or ev.get("pct") or 0.0
        raw_m = ev.get("raw_mean") or ev.get("mean")
        trim_m = ev.get("trimmed_mean")

        title = f"Outlier Sensitivity Profile: {var_a}"
        if raw_m is not None and trim_m is not None:
            insight_text = (
                f"Identified {count} extreme values ({float(pct):.1f}% of observations) in {var_a}. "
                f"Comparing the raw mean ({float(raw_m):.2f}) with the trimmed mean ({float(trim_m):.2f}) "
                f"indicates the central tendency remains stable after adjusting for extreme values."
            )
        else:
            insight_text = (
                f"Identified {count} extreme values ({float(pct):.1f}% of observations) in {var_a}. "
                f"The overall distribution remains robust to extreme observation filtering."
            )

        limitations = list(vf.limitations or [])
        limitations.append("Outlier thresholds computed using the standard 1.5x IQR rule.")

        priority_score = self._compute_priority_score(vf.confidence, True, "NOT_ASSESSED", vf.reinvestigated)

        evidence = InsightEvidence(
            method=vf.method,
            variables=variables,
            statistic=float(count),
            statistic_name="Outlier Count",
            sample_size=ev.get("sample_size"),
            reinvestigated=vf.reinvestigated,
        )

        return Insight(
            insight_id=f"insight_{idx}",
            source_task_id=vf.task_id,
            title=title,
            insight=insight_text,
            interpretation_type="SENSITIVITY",
            strength="NOT_ASSESSED",
            verification_status=vf.status,
            confidence_score=vf.confidence,
            is_statistically_significant=True,
            evidence=evidence,
            limitations=limitations,
            practical_implication="Extreme observations should be monitored but do not invalidate primary central tendency metrics.",
            priority_score=priority_score,
        )

    def _build_distribution_insight(
        self, vf: VerifiedFinding, ev: dict[str, Any], variables: list[str], idx: int
    ) -> Insight:
        var_a = variables[0] if variables else "Feature"
        mean_val = ev.get("mean")
        med_val = ev.get("median")
        std_val = ev.get("std")
        skew_val = ev.get("skewness")
        n_val = ev.get("sample_size") or ev.get("n")

        skew_num = float(skew_val) if skew_val is not None else 0.0
        if skew_num > 1.0:
            skew_label = "positively skewed (right-tailed)"
        elif skew_num < -1.0:
            skew_label = "negatively skewed (left-tailed)"
        else:
            skew_label = "relatively symmetric"

        mean_str = f"{float(mean_val):.2f}" if mean_val is not None else "N/A"
        med_str = f"{float(med_val):.2f}" if med_val is not None else "N/A"
        std_str = f"{float(std_val):.2f}" if std_val is not None else "N/A"

        title = f"Distributional Characteristics: {var_a}"
        insight_text = (
            f"{var_a} exhibits a {skew_label} distribution "
            f"(mean = {mean_str}, "
            f"median = {med_str}, "
            f"std = {std_str}, "
            f"skewness = {skew_num:.2f}, n = {n_val or 'N/A'})."
        )

        limitations = list(vf.limitations or [])
        priority_score = self._compute_priority_score(vf.confidence, True, "NOT_ASSESSED", vf.reinvestigated)

        evidence = InsightEvidence(
            method=vf.method,
            variables=variables,
            statistic=skew_num,
            statistic_name="Skewness",
            sample_size=n_val,
            reinvestigated=vf.reinvestigated,
        )

        return Insight(
            insight_id=f"insight_{idx}",
            source_task_id=vf.task_id,
            title=title,
            insight=insight_text,
            interpretation_type="DESCRIPTIVE",
            strength="NOT_ASSESSED",
            verification_status=vf.status,
            confidence_score=vf.confidence,
            is_statistically_significant=True,
            evidence=evidence,
            limitations=limitations,
            practical_implication=f"For highly skewed distributions, median values provide a more representative measure of location.",
            priority_score=priority_score,
        )

    def _build_generic_insight(
        self, vf: VerifiedFinding, ev: dict[str, Any], variables: list[str], idx: int
    ) -> Insight:
        title = f"Verified Finding: {', '.join(variables) if variables else vf.task_id}"
        insight_text = vf.summary or f"Empirical analysis using {vf.method} verified successfully."

        priority_score = self._compute_priority_score(vf.confidence, True, "NOT_ASSESSED", vf.reinvestigated)

        evidence = InsightEvidence(
            method=vf.method,
            variables=variables,
            p_value=ev.get("p_value"),
            statistic=ev.get("statistic"),
            sample_size=ev.get("sample_size"),
            reinvestigated=vf.reinvestigated,
        )

        return Insight(
            insight_id=f"insight_{idx}",
            source_task_id=vf.task_id,
            title=title,
            insight=insight_text,
            interpretation_type="DESCRIPTIVE",
            strength="NOT_ASSESSED",
            verification_status=vf.status,
            confidence_score=vf.confidence,
            is_statistically_significant=True,
            evidence=evidence,
            limitations=list(vf.limitations or []),
            priority_score=priority_score,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Helper utilities
    # ──────────────────────────────────────────────────────────────────────────

    def _format_p_str(self, p: Optional[float]) -> str:
        if p is None:
            return "p = N/A"
        if p < 0.001:
            return "p < 0.001"
        return f"p = {p:.4f}"

    def _compute_priority_score(
        self, confidence: float, is_sig: bool, strength: InsightStrength, reinvestigated: bool
    ) -> float:
        """
        Deterministic ranking score to order key insights:
        - Confidence: up to 100 points
        - Significance: +50 points
        - Strength: STRONG (+40), MODERATE (+20), WEAK (+5), NOT_ASSESSED (0)
        - Re-investigated & verified: +15 points (high robustness)
        """
        score = confidence * 100.0
        if is_sig:
            score += 50.0

        if strength == "STRONG":
            score += 40.0
        elif strength == "MODERATE":
            score += 20.0
        elif strength == "WEAK":
            score += 5.0

        if reinvestigated:
            score += 15.0

        return round(score, 2)


_insight_service_instance: Optional[InsightService] = None


def get_insight_service() -> InsightService:
    global _insight_service_instance
    if _insight_service_instance is None:
        _insight_service_instance = InsightService()
    return _insight_service_instance

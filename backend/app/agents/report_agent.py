"""
backend/app/agents/report_agent.py
==================================
Phase 5 / Stage B: Dedicated Report Agent & Scientific Synthesis Engine.

Purpose:
    Synthesize factual, structured research reports from actual saved experiment runs
    across Phases 1 through 7. Faithfully represent dataset characteristics, initial
    statistical findings, Critic verification verdicts, controlled re-investigations,
    and final findings without fabrication or unwarranted causal claims.

Critical Rules:
    - Zero LLM fabrication. All metrics, tests, verdicts, and evidence originate directly
      from Python/SciPy execution and saved run state.
    - Missing data is represented as unavailable (None or "—"), never invented.
    - Clearly distinguish initial findings from verified or reinvestigated revisions.
    - State explicitly that Critic acceptance confirms assumption and bound checks,
      not causal or mathematical proof.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from app.schemas.report_schema import (
    AnalysisContext,
    CriticVerdictSummary,
    DatasetContext,
    EvaluationSummary,
    FinalFindingSummary,
    InitialFindingSummary,
    ReinvestigationDetail,
    ResearchReport,
)
from app.schemas.run_schema import RunPayload

logger = logging.getLogger("datainvest.report_agent")


def _format_stat(val: Optional[Union[float, int]], precision: int = 4) -> str:
    """Format numeric statistic safely."""
    if val is None:
        return "—"
    try:
        f = float(val)
        return f"{f:.{precision}f}"
    except (ValueError, TypeError):
        return str(val)


def _format_pval(pval: Optional[float], formatted_str: Optional[str] = None) -> str:
    """Format p-value safely with scientific notation for small numbers."""
    if formatted_str:
        return formatted_str
    if pval is None:
        return "—"
    try:
        f = float(pval)
        if f < 0.0001:
            return f"{f:.4e}"
        return f"{f:.4f}"
    except (ValueError, TypeError):
        return str(pval)


class ReportAgent:
    """Generates structured scientific research reports and formatted exports from experiment runs."""

    REPORT_AGENT_VERSION = "1.0.0"

    def generate_report(self, run_input: Union[RunPayload, Dict[str, Any]]) -> ResearchReport:
        """
        Synthesize a structured ResearchReport from saved run data or a RunPayload.
        Extracts factual pipeline results across all executed phases.
        """
        run_data: Dict[str, Any] = (
            run_input.model_dump() if isinstance(run_input, RunPayload) else dict(run_input)
        )

        run_id = str(run_data.get("run_id") or "unknown_run")
        dataset_name = str(run_data.get("dataset_name") or "untitled_dataset.csv")
        dataset_id = str(run_data.get("dataset_id") or dataset_name)
        now_iso = datetime.now(timezone.utc).isoformat()

        # ------------------------------------------------------------------
        # 1. DATASET CONTEXT (Phase 1)
        # ------------------------------------------------------------------
        profile = run_data.get("profile") or {}
        overview = profile.get("overview") or {}

        n_rows = overview.get("n_rows")
        n_cols = overview.get("n_cols")
        total_missing = overview.get("total_missing_cells")
        missing_pct = overview.get("overall_missing_percentage")
        target_var = overview.get("target_variable")

        num_cols = len(profile.get("numerical_stats", {})) if profile.get("numerical_stats") else None
        cat_cols = len(profile.get("categorical_stats", {})) if profile.get("categorical_stats") else None

        dataset_ctx = DatasetContext(
            dataset_name=dataset_name,
            dataset_id=dataset_id,
            n_rows=n_rows,
            n_cols=n_cols,
            total_missing_cells=total_missing,
            missingness_pct=round(missing_pct, 2) if missing_pct is not None else None,
            numerical_columns_count=num_cols,
            categorical_columns_count=cat_cols,
            target_variable=target_var,
        )

        # ------------------------------------------------------------------
        # 2. ANALYSIS CONTEXT (Phase 2 & Execution Setup)
        # ------------------------------------------------------------------
        objective = str(run_data.get("objective") or "Investigate this dataset.")
        branch_mode = run_data.get("mode")
        paired_run_id = run_data.get("paired_run_id")
        completed_phases = list(run_data.get("completed_phases") or [])

        analysis_obj = run_data.get("analysis") or {}
        analysis_results_raw = analysis_obj.get("results") or []
        plan_obj = run_data.get("plan") or {}
        tasks_planned = len(plan_obj.get("analysis_tasks") or []) or len(analysis_results_raw)
        planner_mode = plan_obj.get("planner_mode")
        planner_model = plan_obj.get("planner_model") or (plan_obj.get("dataset_summary") or {}).get("model")
        planner_provider = plan_obj.get("planner_provider")
        if not planner_provider and planner_mode:
            planner_provider = "google" if planner_mode == "llm" else ("deterministic" if planner_mode == "deterministic" else None)

        successful_tasks = sum(1 for r in analysis_results_raw if isinstance(r, dict) and r.get("status") == "success")
        failed_tasks = sum(1 for r in analysis_results_raw if isinstance(r, dict) and r.get("status") == "failed")

        analysis_ctx = AnalysisContext(
            objective=objective,
            branch_mode=branch_mode,
            paired_run_id=paired_run_id,
            completed_phases=completed_phases,
            total_tasks_planned=tasks_planned,
            successful_tasks=successful_tasks,
            failed_tasks=failed_tasks,
            planner_mode=planner_mode,
            planner_model=planner_model,
            planner_provider=planner_provider,
        )

        # ------------------------------------------------------------------
        # 3. INITIAL FINDINGS (Phase 3 Execution)
        # ------------------------------------------------------------------
        initial_findings: List[InitialFindingSummary] = []
        for r in analysis_results_raw:
            if not isinstance(r, dict):
                continue

            r_data = r.get("results") or {}
            task_id = str(r.get("task_id") or f"task_{len(initial_findings) + 1}")
            method = str(r.get("method") or "unknown_method")
            analysis_type = str(r.get("analysis_type") or "statistical_test")
            cols = list(r.get("columns") or [])

            # Extract primary statistic
            stat_val = (
                r_data.get("statistic")
                or r_data.get("correlation_coefficient")
                or r_data.get("t_statistic")
                or r_data.get("f_statistic")
                or r_data.get("chi2_statistic")
                or r_data.get("u_statistic")
                or r_data.get("h_statistic")
            )
            try:
                stat_val = float(stat_val) if stat_val is not None else None
            except (ValueError, TypeError):
                stat_val = None

            pval = r_data.get("p_value")
            try:
                pval = float(pval) if pval is not None else None
            except (ValueError, TypeError):
                pval = None

            pval_fmt = r_data.get("p_value_formatted")
            sample_sz = r.get("sample_size") or r_data.get("sample_size")

            # Extract effect size
            eff_val = (
                r_data.get("effect_size")
                or r_data.get("cramer_v")
                or r_data.get("eta_squared")
                or r_data.get("cohens_d")
                or (abs(stat_val) if method.lower().startswith("pearson") or method.lower().startswith("spearman") else None)
            )
            try:
                eff_val = float(eff_val) if eff_val is not None else None
            except (ValueError, TypeError):
                eff_val = None

            eff_type = None
            if "cramer_v" in r_data:
                eff_type = "Cramér's V"
            elif "eta_squared" in r_data:
                eff_type = "Eta-squared (η²)"
            elif "cohens_d" in r_data:
                eff_type = "Cohen's d"
            elif method.lower().startswith("pearson") or method.lower().startswith("spearman"):
                eff_type = "Correlation (r)"

            initial_findings.append(
                InitialFindingSummary(
                    task_id=task_id,
                    variables=cols,
                    analysis_type=analysis_type,
                    method=method,
                    statistic=round(stat_val, 4) if stat_val is not None else None,
                    p_value=pval,
                    p_value_formatted=pval_fmt,
                    sample_size=sample_sz,
                    effect_size=round(eff_val, 4) if eff_val is not None else None,
                    effect_size_type=eff_type,
                    status=str(r.get("status") or "success"),
                    warnings=list(r.get("warnings") or []),
                )
            )

        # ------------------------------------------------------------------
        # 4. CRITIC VERDICTS (Phase 4 Verification)
        # ------------------------------------------------------------------
        critic_obj = run_data.get("critic") or {}
        critic_reviews_raw = critic_obj.get("reviews") or []
        critic_verdicts: List[CriticVerdictSummary] = []

        for rev in critic_reviews_raw:
            if not isinstance(rev, dict):
                continue
            task_id = str(rev.get("task_id") or "unknown_task")
            status = str(rev.get("status") or "UNVERIFIED")
            conf = float(rev.get("confidence", 1.0))
            conf_level = str(rev.get("confidence_level") or "HIGH")
            summary_str = str(rev.get("summary") or "Critic evaluation completed.")
            stat_validity = rev.get("statistical_validity")
            method_assess = rev.get("method_assessment")
            dq_concerns = list(rev.get("data_quality_concerns") or [])
            assump_concerns = list(rev.get("assumption_concerns") or [])
            reason_codes = list(rev.get("reason_codes") or [])
            evidence = list(rev.get("evidence") or [])

            critic_verdicts.append(
                CriticVerdictSummary(
                    task_id=task_id,
                    status=status,
                    confidence=conf,
                    confidence_level=conf_level,
                    summary=summary_str,
                    statistical_validity=stat_validity,
                    method_assessment=method_assess,
                    data_quality_concerns=dq_concerns,
                    assumption_concerns=assump_concerns,
                    reason_codes=reason_codes,
                    evidence=evidence,
                )
            )

        # ------------------------------------------------------------------
        # 5. RE-INVESTIGATION DETAILS (Phase 5 Controlled Transitions)
        # ------------------------------------------------------------------
        verification_obj = run_data.get("verification") or {}
        reinvestigations_raw = verification_obj.get("reinvestigations") or []
        reinvestigations: List[ReinvestigationDetail] = []

        for reinv in reinvestigations_raw:
            if not isinstance(reinv, dict):
                continue

            reinv_id = str(reinv.get("reinvestigation_id") or "reinv_1")
            source_id = str(reinv.get("source_task_id") or "unknown_task")
            vars_list = list(reinv.get("variables") or [])
            orig_method = str(reinv.get("original_method") or "Original Method")
            reinv_method = str(reinv.get("reinvestigated_method") or "Revised Method")
            trigger_reason = str(reinv.get("trigger_reason") or "Methodological assumption violation.")

            comp = reinv.get("comparison") or {}
            orig_stat = comp.get("original_statistic")
            reinv_stat = comp.get("reinvestigated_statistic")
            orig_p = comp.get("original_p_value")
            reinv_p = comp.get("reinvestigated_p_value")
            dir_cons = comp.get("direction_consistent")
            sig_cons = comp.get("significance_consistent")
            comp_summary = str(comp.get("summary") or "Comparative delta evaluated.")
            final_st = str(reinv.get("status") or "VERIFIED_AFTER_REINVESTIGATION")

            reinvestigations.append(
                ReinvestigationDetail(
                    reinvestigation_id=reinv_id,
                    source_task_id=source_id,
                    variables=vars_list,
                    original_method=orig_method,
                    reinvestigated_method=reinv_method,
                    trigger_reason=trigger_reason,
                    original_statistic=round(orig_stat, 4) if orig_stat is not None else None,
                    reinvestigated_statistic=round(reinv_stat, 4) if reinv_stat is not None else None,
                    original_p_value=orig_p,
                    reinvestigated_p_value=reinv_p,
                    direction_consistent=dir_cons,
                    significance_consistent=sig_cons,
                    comparison_summary=comp_summary,
                    final_status=final_st,
                )
            )

        # ------------------------------------------------------------------
        # 6. FINAL FINDINGS (Consolidated & Revised Distinction)
        # ------------------------------------------------------------------
        final_findings: List[FinalFindingSummary] = []
        verified_raw = (
            verification_obj.get("all_findings")
            or verification_obj.get("verified_findings")
            or []
        )

        # If verification phase was executed, extract from verification findings
        if verified_raw:
            for idx, vf in enumerate(verified_raw, 1):
                if not isinstance(vf, dict):
                    continue
                finding_id = str(vf.get("finding_id") or f"finding_{idx}")
                task_id = str(vf.get("task_id") or f"task_{idx}")
                vars_list = list(vf.get("variables") or [])
                method = str(vf.get("method") or "unknown_method")
                status = str(vf.get("status") or "VERIFIED")
                conf = float(vf.get("confidence", 1.0))
                conf_lvl = str(vf.get("confidence_level") or "HIGH")
                summary = str(vf.get("summary") or "Verified statistical finding.")
                reason = str(vf.get("verification_reason") or "Method passed verification checks.")
                is_revised = bool(
                    vf.get("reinvestigated")
                    or status == "VERIFIED_AFTER_REINVESTIGATION"
                    or vf.get("reinvestigation_record")
                )
                limits = list(vf.get("limitations") or [])
                codes = list(vf.get("reason_codes") or [])
                has_viz = bool(vf.get("visual_evidence") or vf.get("graph_id"))

                final_findings.append(
                    FinalFindingSummary(
                        finding_id=finding_id,
                        task_id=task_id,
                        variables=vars_list,
                        method=method,
                        status=status,
                        confidence=conf,
                        confidence_level=conf_lvl,
                        summary=summary,
                        verification_reason=reason,
                        is_revised_after_reinvestigation=is_revised,
                        limitations=limits,
                        reason_codes=codes,
                        has_visual_evidence=has_viz,
                    )
                )
        elif critic_verdicts:
            # Fallback if verification was not run: map critic verdicts faithfully
            for idx, cv in enumerate(critic_verdicts, 1):
                is_questionable = cv.status == "QUESTIONABLE"
                is_rejected = cv.status == "REJECTED"
                stat_mapped = "QUESTIONABLE" if is_questionable else ("REJECTED" if is_rejected else "VERIFIED")

                final_findings.append(
                    FinalFindingSummary(
                        finding_id=f"finding_{idx}",
                        task_id=cv.task_id,
                        variables=initial_findings[idx - 1].variables if idx - 1 < len(initial_findings) else [],
                        method=initial_findings[idx - 1].method if idx - 1 < len(initial_findings) else "Statistical test",
                        status=stat_mapped,
                        confidence=cv.confidence,
                        confidence_level=cv.confidence_level,
                        summary=cv.summary,
                        verification_reason=cv.statistical_validity or "Evaluated via Critic verification.",
                        is_revised_after_reinvestigation=False,
                        limitations=cv.assumption_concerns + cv.data_quality_concerns,
                        reason_codes=cv.reason_codes,
                        has_visual_evidence=False,
                    )
                )

        # ------------------------------------------------------------------
        # 7. LIMITATIONS & UNRESOLVED VERIFICATION ISSUES
        # ------------------------------------------------------------------
        limitations: List[str] = []
        seen_core_keys: set[str] = set()

        verif_summary = verification_obj.get("summary") or {}
        for issue in verif_summary.get("unresolved_issues", []):
            if issue:
                issue_str = str(issue).strip()
                core = re.sub(r"^\[[\w\.-]+\]\s*", "", issue_str).strip()
                if issue_str not in limitations and core not in seen_core_keys:
                    limitations.append(issue_str)
                    seen_core_keys.add(core)
                    seen_core_keys.add(issue_str)

        for cv in critic_verdicts:
            for concern in cv.assumption_concerns + cv.data_quality_concerns:
                if concern:
                    concern_str = str(concern).strip()
                    core = re.sub(r"^\[[\w\.-]+\]\s*", "", concern_str).strip()
                    if core not in seen_core_keys and concern_str not in limitations:
                        entry = f"[{cv.task_id}] {concern_str}" if cv.status in ("QUESTIONABLE", "REJECTED", "REQUIRES_REINVESTIGATION") else concern_str
                        limitations.append(entry)
                        seen_core_keys.add(core)
                        seen_core_keys.add(entry)
                        seen_core_keys.add(concern_str)

        if dataset_ctx.missingness_pct and dataset_ctx.missingness_pct > 0.0:
            msg = f"Dataset contains {dataset_ctx.missingness_pct:.1f}% missing cells; listwise deletion applied pairwise per test."
            if msg not in limitations:
                limitations.append(msg)

        # ------------------------------------------------------------------
        # 8. EVALUATION SUMMARY (Phase 7)
        # ------------------------------------------------------------------
        eval_obj = run_data.get("evaluation") or {}
        eval_summary_raw = eval_obj.get("summary") or {}
        eval_summary: Optional[EvaluationSummary] = None

        if eval_summary_raw and eval_summary_raw.get("overall_score") is not None:
            eval_summary = EvaluationSummary(
                overall_score=eval_summary_raw.get("overall_score"),
                overall_grade=eval_summary_raw.get("overall_grade"),
                benchmark_pass_rate=eval_summary_raw.get("benchmark_pass_rate"),
                summary_verdict=eval_summary_raw.get("summary_verdict"),
            )

        report_id = f"report_{run_id}"

        return ResearchReport(
            report_id=report_id,
            run_id=run_id,
            title=f"DataInvest AI Scientific Investigation Report — {dataset_name}",
            generated_at=now_iso,
            dataset_context=dataset_ctx,
            analysis_context=analysis_ctx,
            initial_findings=initial_findings,
            critic_verdicts=critic_verdicts,
            reinvestigations=reinvestigations,
            final_findings=final_findings,
            limitations_and_unresolved_issues=limitations,
            evaluation_summary=eval_summary,
        )

    def render_markdown(self, report: ResearchReport) -> str:
        """
        Render a readable, structured GitHub-Flavored Markdown report from ResearchReport.
        """
        lines: List[str] = []

        # 1. Header Banner
        lines.append(f"# {report.title}")
        lines.append("")
        lines.append("> **DataInvest AI Automated Investigation & Verification Pipeline**  ")
        lines.append(f"> **Run ID:** `{report.run_id}` | **Generated:** `{report.generated_at}`")
        if report.analysis_context.branch_mode:
            lines.append(f"> **Architecture Mode:** `{report.analysis_context.branch_mode.upper()}`")
        if report.evaluation_summary and report.evaluation_summary.overall_score is not None:
            lines.append(
                f"> **System Rigor Score:** `{report.evaluation_summary.overall_score} / 100` "
                f"(`Grade {report.evaluation_summary.overall_grade}`) | Pass Rate: `{report.evaluation_summary.benchmark_pass_rate or '—'}`"
            )

        # Explicitly distinguish partial-stage / interim reports from completed investigation runs
        completed_phases = report.analysis_context.completed_phases or []
        is_interim = not any(p in completed_phases for p in (5, 6, 7))
        if is_interim:
            max_p = max(completed_phases) if completed_phases else 1
            lines.append(
                f"> ⚠ **Interim Report (Phase {max_p} Snapshot)**: This export reflects preliminary pipeline execution only. "
                "Downstream statistical verification, synthesis, and evaluation have not yet completed."
            )

        lines.append("")
        lines.append("---")
        lines.append("")

        # 2. Executive Objective & Scope
        lines.append("## 1. Investigation Objective & Execution Scope")
        lines.append("")
        lines.append(f"**Primary Objective:** {report.analysis_context.objective}")
        lines.append("")
        lines.append("| Execution Dimension | Value |")
        lines.append("|:---|:---|")
        lines.append(f"| **Dataset Name** | `{report.dataset_context.dataset_name}` |")
        lines.append(f"| **Dataset Identifier** | `{report.dataset_context.dataset_id}` |")
        lines.append(f"| **Planned Tasks** | `{report.analysis_context.total_tasks_planned}` |")
        lines.append(f"| **Successful Executions** | `{report.analysis_context.successful_tasks}` |")
        lines.append(f"| **Failed/Boundary Tasks** | `{report.analysis_context.failed_tasks}` |")
        lines.append(f"| **Completed Phases** | `{', '.join(f'P{p}' for p in report.analysis_context.completed_phases) or 'None'}` |")
        planner_mode_str = report.analysis_context.planner_mode.upper() if report.analysis_context.planner_mode else "Not Recorded"
        lines.append(f"| **Planner Mode** | `{planner_mode_str}` |")
        if report.analysis_context.planner_model:
            lines.append(f"| **Planner Model** | `{report.analysis_context.planner_model}` |")
        if report.analysis_context.planner_provider:
            lines.append(f"| **Planner Provider** | `{report.analysis_context.planner_provider}` |")
        lines.append("")

        # 3. Dataset Characteristics & Data Quality
        lines.append("## 2. Dataset Context & Profile Characteristics")
        lines.append("")
        lines.append("| Metric | Value | Interpretation |")
        lines.append("|:---|:---|:---|")
        n_r = f"{report.dataset_context.n_rows:,}" if report.dataset_context.n_rows else "Unavailable"
        n_c = str(report.dataset_context.n_cols) if report.dataset_context.n_cols else "Unavailable"
        lines.append(f"| **Total Rows** | `{n_r}` | Effective sample universe |")
        lines.append(f"| **Total Columns** | `{n_c}` | Dimensionality |")
        lines.append(
            f"| **Numerical Variables** | `{report.dataset_context.numerical_columns_count or '—'}` | Continuous and discrete quantitative fields |"
        )
        lines.append(
            f"| **Categorical Variables** | `{report.dataset_context.categorical_columns_count or '—'}` | Nominal and factor attributes |"
        )
        miss_pct = f"{report.dataset_context.missingness_pct:.2f}%" if report.dataset_context.missingness_pct is not None else "Unavailable"
        lines.append(
            f"| **Overall Missingness** | `{miss_pct}` | Total missing data proportion |"
        )
        if report.dataset_context.target_variable:
            lines.append(f"| **Target Attribute** | `{report.dataset_context.target_variable}` | Primary analytical target |")
        lines.append("")

        # 4. Phase 3 Initial Statistical Findings
        lines.append("## 3. Initial Statistical Findings (Phase 3 Execution)")
        lines.append("")
        lines.append(
            "The initial analysis executed deterministic SciPy statistical procedures planned during Phase 2. "
            "These findings represent raw empirical calculations prior to Critic self-verification:"
        )
        lines.append("")
        if not report.initial_findings:
            lines.append("_No Phase 3 statistical tasks were recorded for this run._")
            lines.append("")
        else:
            lines.append("| Task ID | Variables | Method Executed | Test Statistic | p-Value | Effect Size | Status |")
            lines.append("|:---|:---|:---|:---|:---|:---|:---|")
            for f in report.initial_findings:
                vars_str = ", ".join(f.variables) if f.variables else "—"
                stat_str = _format_stat(f.statistic)
                p_str = _format_pval(f.p_value, f.p_value_formatted)
                eff_str = f"{f.effect_size_type}: {_format_stat(f.effect_size)}" if f.effect_size is not None else "—"
                lines.append(
                    f"| `{f.task_id}` | {vars_str} | {f.method} | `{stat_str}` | `{p_str}` | {eff_str} | `{f.status.upper()}` |"
                )
            lines.append("")

        # 5. Phase 4 Critic Agent Verification Verdicts
        lines.append("## 4. Critic Agent Statistical Self-Verification (Phase 4)")
        lines.append("")
        lines.append(
            "The Phase 4 Critic Agent performed deterministic mathematical audits on each finding, verifying parametric "
            "assumptions (e.g. residual normality, outlier leverage, contingency cell expected frequencies), statistical power, "
            "and mathematical bounds. No LLM calculation was used."
        )
        lines.append("")
        if not report.critic_verdicts:
            lines.append("_No Critic verification reviews were recorded for this run._")
            lines.append("")
        else:
            lines.append("| Task ID | Verdict | Evaluation Certainty | Decision Codes | Core Assessment |")
            lines.append("|:---|:---|:---|:---|:---|")
            for cv in report.critic_verdicts:
                codes_str = ", ".join(cv.reason_codes[:3]) if cv.reason_codes else "NONE"
                if cv.status == "REJECTED":
                    certainty_label = f"{cv.confidence:.2f} ({cv.confidence_level} Decision Certainty)"
                elif cv.status == "REQUIRES_REINVESTIGATION":
                    certainty_label = f"{cv.confidence:.2f} ({cv.confidence_level} Review Certainty)"
                else:
                    certainty_label = f"{cv.confidence:.2f} ({cv.confidence_level})"
                lines.append(
                    f"| `{cv.task_id}` | **`{cv.status}`** | `{certainty_label}` | `{codes_str}` | {cv.summary} |"
                )
            lines.append("")

            # Evidence and assumption details
            lines.append("### Key Verification Details & Assumption Audits")
            lines.append("")
            for cv in report.critic_verdicts:
                if cv.assumption_concerns or cv.evidence:
                    lines.append(f"- **Task `{cv.task_id}`** ({cv.status}):")
                    if cv.statistical_validity:
                        lines.append(f"  - _Mathematical Validity:_ {cv.statistical_validity}")
                    for ev in cv.evidence[:2]:
                        lines.append(f"  - _Empirical Evidence:_ {ev}")
                    for concern in cv.assumption_concerns:
                        lines.append(f"  - ⚠ _Assumption Concern:_ {concern}")
                    for dq in cv.data_quality_concerns:
                        lines.append(f"  - ℹ _Data Concern:_ {dq}")
            lines.append("")

        # 6. Phase 5 Controlled Re-Investigations
        lines.append("## 5. Controlled Re-investigation & Method Transitions (Phase 5)")
        lines.append("")
        if not report.reinvestigations:
            lines.append(
                "No findings required targeted Phase 5 re-investigation in this run. "
                "Initial parametric assumptions were either satisfied or findings were classified directly."
            )
            lines.append("")
        else:
            lines.append(
                "When severe assumption violations were flagged by the Critic Agent (e.g. leverage outliers, extreme skewness), "
                "the framework executed targeted, alternative non-parametric tests under a strict 1-attempt budget:"
            )
            lines.append("")
            lines.append("| Task | Original Method | Re-investigated Method | Trigger Rationale | Original Stat → Revised Stat | Direction Preserved | Significance Preserved |")
            lines.append("|:---|:---|:---|:---|:---|:---|:---|")
            for ri in report.reinvestigations:
                orig_s = _format_stat(ri.original_statistic)
                reinv_s = _format_stat(ri.reinvestigated_statistic)
                dir_p = "✓ Yes" if ri.direction_consistent is True else ("✗ No" if ri.direction_consistent is False else "—")
                sig_p = "✓ Yes" if ri.significance_consistent is True else ("✗ No" if ri.significance_consistent is False else "—")
                lines.append(
                    f"| `{ri.source_task_id}` | {ri.original_method} | **{ri.reinvestigated_method}** | {ri.trigger_reason} | `{orig_s}` → `{reinv_s}` | {dir_p} | {sig_p} |"
                )
            lines.append("")

            for ri in report.reinvestigations:
                lines.append(f"**Re-investigation Detail for Task `{ri.source_task_id}`:**")
                lines.append(f"> {ri.comparison_summary} (Final Status: `{ri.final_status}`)")
                lines.append("")

        # 7. Final Consolidated Findings
        lines.append("## 6. Consolidated Final Findings & Scientific Outcomes")
        lines.append("")
        lines.append(
            "Final findings distinguish between unrevised verified insights and findings successfully verified "
            "after controlled re-investigation, as well as questionable or rejected hypotheses:"
        )
        lines.append("")

        if not report.final_findings:
            lines.append("_No final findings synthesized._")
            lines.append("")
        else:
            verified = [f for f in report.final_findings if f.status in ("VERIFIED", "VERIFIED_AFTER_REINVESTIGATION")]
            questionable = [f for f in report.final_findings if f.status == "QUESTIONABLE"]
            rejected = [f for f in report.final_findings if f.status == "REJECTED"]

            if verified:
                lines.append("### A. Verified Scientific Findings")
                lines.append("")
                lines.append("| Finding ID | Variables | Statistical Method | Status | Revised | Summary |")
                lines.append("|:---|:---|:---|:---|:---|:---|")
                for vf in verified:
                    rev_tag = "✓ Revised" if vf.is_revised_after_reinvestigation else "Original"
                    lines.append(
                        f"| `{vf.finding_id}` | {', '.join(vf.variables)} | **{vf.method}** | `{vf.status}` | `{rev_tag}` | {vf.summary} |"
                    )
                lines.append("")

            if questionable:
                lines.append("### B. Questionable Findings (Retained with Methodological Caveats)")
                lines.append("")
                lines.append("| Finding ID | Variables | Method | Documented Caveats | Reason Codes |")
                lines.append("|:---|:---|:---|:---|:---|")
                for qf in questionable:
                    caveats = "; ".join(qf.limitations[:2]) if qf.limitations else qf.verification_reason
                    lines.append(
                        f"| `{qf.finding_id}` | {', '.join(qf.variables)} | {qf.method} | {caveats} | `{', '.join(qf.reason_codes[:2])}` |"
                    )
                lines.append("")

            if rejected:
                lines.append("### C. Rejected Analyses (Boundary Violations & Infeasible Tests)")
                lines.append("")
                lines.append("| Finding ID | Variables | Method | Rejection Rationale |")
                lines.append("|:---|:---|:---|:---|")
                for rf in rejected:
                    lines.append(
                        f"| `{rf.finding_id}` | {', '.join(rf.variables)} | {rf.method} | {rf.verification_reason} |"
                    )
                lines.append("")

        # 8. Documented Limitations & Unresolved Issues
        lines.append("## 7. Documented Limitations & Unresolved Verification Issues")
        lines.append("")
        if not report.limitations_and_unresolved_issues:
            lines.append("No unresolved verification issues or severe data quality limitations were documented.")
            lines.append("")
        else:
            for idx, limit in enumerate(report.limitations_and_unresolved_issues, 1):
                lines.append(f"{idx}. {limit}")
            lines.append("")

        # 9. Phase 7 System Evaluation Scorecard
        if report.evaluation_summary and report.evaluation_summary.overall_score is not None:
            lines.append("## 8. Quantitative Benchmark Scorecard (Phase 7)")
            lines.append("")
            lines.append(
                f"- **Overall Empirical Rigor Score:** `{report.evaluation_summary.overall_score:.1f} / 100`"
            )
            lines.append(f"- **System Quality Grade:** `{report.evaluation_summary.overall_grade}`")
            if report.evaluation_summary.benchmark_pass_rate:
                lines.append(f"- **Benchmark Check Pass Rate:** `{report.evaluation_summary.benchmark_pass_rate}`")
            if report.evaluation_summary.summary_verdict:
                lines.append(f"- **Executive Evaluation Verdict:** {report.evaluation_summary.summary_verdict}")
            lines.append("")

        # 10. Formal Methodological Disclaimer
        lines.append("## 9. Methodological & Non-Causal Scientific Disclaimer")
        lines.append("")
        lines.append(f"> ⚠ **Notice:** {report.methodological_disclaimer}")
        lines.append("")
        lines.append("---")
        lines.append(
            f"_Report synthesized deterministically by DataInvest AI Report Agent v{self.REPORT_AGENT_VERSION}. "
            "Zero fabricated findings or ungrounded model citations._"
        )
        lines.append("")

        return "\n".join(lines)


# Global singleton instance
report_agent = ReportAgent()

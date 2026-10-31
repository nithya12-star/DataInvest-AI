"""
backend/app/services/verification_service.py
============================================
Phase 5: Re-investigation & Verified Findings Orchestration Engine.

Purpose:
    Orchestrate the empirical verification loop between the existing Phase 3
    Analysis Agent and Phase 4 Critic Agent. Re-investigates eligible findings
    using targeted alternative methods, compares before/after results, and
    constructs an auditable collection of Verified Findings.

Key Rules:
    - Zero statistical fabrication. All new calculations use Python/Pandas/SciPy.
    - Configurable MAX_REINVESTIGATIONS (0 to 3 attempts) with cycle and p-hacking prevention.
    - Re-uses existing Analysis Agent and Critic Agent instances.
    - Failure isolation: Failed individual re-analyses do not crash the batch.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Union
import numpy as np
import pandas as pd

from app.agents.analysis_agent import AnalysisAgent
from app.agents.critic_agent import CriticAgent
from app.schemas.analysis_schema import AnalysisResult, PlanAnalysisResponse
from app.schemas.critic_schema import CriticResponse, CriticResult
from app.schemas.planner_schema import InvestigationTask, PlannerResponse
from app.schemas.verification_schema import (
    FinalVerificationStatus,
    ReinvestigationRecord,
    ResultComparison,
    VerificationRequest,
    VerificationResponse,
    VerificationSummary,
    VerifiedFinding,
)
from app.services.dataset_store import resolve_dataset

logger = logging.getLogger("datainvest.verification_service")

# Standardized mapping of Critic recommended methods to AnalysisAgent capabilities
SUPPORTED_METHODS: dict[str, dict[str, str]] = {
    "spearman rank correlation": {
        "task_type": "numerical_vs_numerical",
        "method": "Spearman rank correlation",
    },
    "spearman correlation": {
        "task_type": "numerical_vs_numerical",
        "method": "Spearman rank correlation",
    },
    "pearson correlation": {
        "task_type": "numerical_vs_numerical",
        "method": "Pearson correlation",
    },
    "mann-whitney u test": {
        "task_type": "two_groups_vs_numerical",
        "method": "Mann-Whitney U test",
    },
    "mann-whitney u": {
        "task_type": "two_groups_vs_numerical",
        "method": "Mann-Whitney U test",
    },
    "welch's t-test": {
        "task_type": "two_groups_vs_numerical",
        "method": "Welch's t-test",
    },
    "kruskal-wallis test": {
        "task_type": "multiple_groups_vs_numerical",
        "method": "Kruskal-Wallis test",
    },
    "kruskal-wallis": {
        "task_type": "multiple_groups_vs_numerical",
        "method": "Kruskal-Wallis test",
    },
    "one-way anova": {
        "task_type": "multiple_groups_vs_numerical",
        "method": "One-way ANOVA",
    },
    "chi-square test of independence": {
        "task_type": "categorical_vs_categorical",
        "method": "Chi-square test of independence",
    },
    "iqr-based sensitivity analysis": {
        "task_type": "outlier_sensitivity",
        "method": "IQR-based sensitivity analysis",
    },
}


class VerificationService:
    """Orchestrates Phase 5 re-investigation execution and final finding verification."""

    def __init__(
        self,
        analysis_agent: Optional[AnalysisAgent] = None,
        critic_agent: Optional[CriticAgent] = None,
    ):
        self.analysis_agent = analysis_agent or AnalysisAgent()
        self.critic_agent = critic_agent or CriticAgent()

    def process_verification(self, request: VerificationRequest) -> VerificationResponse:
        """
        Execute controlled re-investigation on eligible findings and synthesize verified findings.
        """
        dataset_id = request.dataset_id or (
            request.analysis_response.dataset_id if request.analysis_response else "active_dataset"
        )
        max_attempts = request.max_reinvestigations if request.max_reinvestigations is not None else 1

        # 1. Resolve dataset DataFrame
        df = resolve_dataset(dataset_id)
        if (df is None or df.empty) and request.dataset_profile:
            profile_filename = request.dataset_profile.get("overview", {}).get("filename")
            if profile_filename and profile_filename != dataset_id:
                df = resolve_dataset(profile_filename)
        if df is None:
            # Fallback empty dataframe
            df = pd.DataFrame()

        # 2. Extract Phase 3 Analysis results map
        analysis_map: dict[str, AnalysisResult] = {}
        if request.analysis_response:
            for item in request.analysis_response.results:
                analysis_map[item.task_id] = item

        # 2b. Map Investigation Tasks from plan if available
        task_map: dict[str, Any] = {}
        if request.investigation_plan:
            plan_tasks = (
                getattr(request.investigation_plan, "analysis_tasks", None)
                or getattr(request.investigation_plan, "tasks", None)
                or (request.investigation_plan.get("analysis_tasks") if isinstance(request.investigation_plan, dict) else None)
                or (request.investigation_plan.get("tasks") if isinstance(request.investigation_plan, dict) else None)
                or []
            )
            for t in plan_tasks:
                tid = getattr(t, "task_id", None) or (t.get("task_id") if isinstance(t, dict) else None)
                if tid:
                    task_map[tid] = t

        # 3. Extract or compute Phase 4 Critic reviews
        critic_map: dict[str, CriticResult] = {}
        if request.critic_response:
            for rev in request.critic_response.reviews:
                critic_map[rev.task_id] = rev
        elif request.analysis_response:
            # Dynamically compute critic review if not pre-computed
            critic_resp = self.critic_agent.evaluate_plan(
                analysis_input=request.analysis_response,
                profile=request.dataset_profile,
                plan=request.investigation_plan,
                dataset_id=dataset_id,
            )
            for rev in critic_resp.reviews:
                critic_map[rev.task_id] = rev

        # 4. Iterate through each finding and apply Phase 5 verification logic
        all_findings: list[VerifiedFinding] = []
        verified_list: list[VerifiedFinding] = []
        questionable_list: list[VerifiedFinding] = []
        rejected_list: list[VerifiedFinding] = []
        reinvestigations_list: list[ReinvestigationRecord] = []

        finding_counter = 1

        for task_id, critic_rev in critic_map.items():
            orig_analysis = analysis_map.get(task_id)
            if not orig_analysis:
                continue

            orig_task = task_map.get(task_id)
            orig_results = orig_analysis.results or {}

            # Determine the exact variables investigated in the original analysis
            orig_task_col_a = (
                getattr(orig_task, "column_a", None)
                if hasattr(orig_task, "column_a")
                else (orig_task.get("column_a") if isinstance(orig_task, dict) else None)
            )
            orig_task_col_b = (
                getattr(orig_task, "column_b", None)
                if hasattr(orig_task, "column_b")
                else (orig_task.get("column_b") if isinstance(orig_task, dict) else None)
            )
            orig_task_grp = (
                getattr(orig_task, "group_column", None)
                if hasattr(orig_task, "group_column")
                else (orig_task.get("group_column") if isinstance(orig_task, dict) else None)
            )
            orig_task_val = (
                getattr(orig_task, "value_column", None)
                if hasattr(orig_task, "value_column")
                else (orig_task.get("value_column") if isinstance(orig_task, dict) else None)
            )

            orig_col_a = (
                orig_results.get("column_a")
                or orig_results.get("column")
                or orig_task_col_a
                or (orig_analysis.columns[0] if len(orig_analysis.columns) > 0 else None)
            )
            orig_col_b = (
                orig_results.get("column_b")
                or orig_task_col_b
                or (orig_analysis.columns[1] if len(orig_analysis.columns) > 1 else None)
            )
            orig_grp = (
                orig_results.get("group_column")
                or orig_task_grp
                or orig_col_a
            )
            orig_val = (
                orig_results.get("value_column")
                or orig_task_val
                or orig_col_b
            )
            if "group" in (orig_analysis.analysis_type or "") and orig_grp and orig_val:
                orig_finding_vars = [orig_grp, orig_val]
            else:
                orig_finding_vars = [c for c in [orig_col_a, orig_col_b] if c] if orig_col_b else ([orig_col_a] if orig_col_a else list(orig_analysis.columns))

            # Ensure visual evidence is attached to original analysis if not already present
            if orig_analysis.status == "success" and not orig_analysis.visual_evidence and not df.empty:
                try:
                    from app.services.visualization_service import get_visualization_service
                    viz_svc = get_visualization_service()
                    orig_analysis.visual_evidence = viz_svc.generate_finding_visualization(
                        task_id=orig_analysis.task_id,
                        analysis_type=orig_analysis.analysis_type,
                        method=orig_analysis.method,
                        columns=orig_analysis.columns,
                        df=df,
                        results=orig_analysis.results,
                    )
                except Exception as e:
                    logger.warning("Could not generate visual evidence for %s: %s", orig_analysis.task_id, e)

            finding_id = f"finding_{finding_counter}"
            finding_counter += 1

            # Check Reinvestigation Eligibility
            critic_wants_reinvestigation = (
                (critic_rev.status == "REQUIRES_REINVESTIGATION" or critic_rev.requires_reinvestigation)
                and bool(critic_rev.recommended_method)
            )
            should_reinvestigate = critic_wants_reinvestigation and not df.empty

            # ----------------------------------------------------------
            # CASE 1: TARGETED RE-INVESTIGATION EXECUTION (max_attempts >= 1)
            # ----------------------------------------------------------
            if should_reinvestigate and max_attempts >= 1:
                current_analysis = orig_analysis
                current_critic = critic_rev
                attempted_methods = {orig_analysis.method.strip().lower()}
                last_reinvest_record: Optional[ReinvestigationRecord] = None
                attempts_run = 0

                for attempt in range(1, max_attempts + 1):
                    # Check if re-investigation is still requested
                    if not (current_critic.status == "REQUIRES_REINVESTIGATION" or current_critic.requires_reinvestigation):
                        break

                    rec_raw = current_critic.recommended_method
                    if not rec_raw:
                        break

                    rec_key = rec_raw.strip().lower()
                    # Prevent method cycling, duplicate methods, and p-hacking
                    if rec_key in attempted_methods:
                        logger.info("Re-investigation loop stopped for %s: method '%s' already attempted (cycle prevention)", task_id, rec_raw)
                        break

                    method_spec = SUPPORTED_METHODS.get(rec_key)
                    if not method_spec:
                        logger.info("Re-investigation loop stopped for %s: method '%s' not supported", task_id, rec_raw)
                        break

                    # Valid alternative found - execute attempt
                    attempts_run += 1
                    attempted_methods.add(rec_key)
                    rec_method = method_spec["method"]
                    rec_task_type = method_spec["task_type"]
                    reinvest_task_id = f"{task_id}_reinvest_{attempt}" if attempt > 1 else f"{task_id}_reinvest"

                    # Build targeted re-investigation task preserving original variable pair
                    is_group_task = "group" in rec_task_type
                    target_col_a = orig_grp if is_group_task else orig_col_a
                    target_col_b = orig_val if is_group_task else orig_col_b
                    reinvest_pair = [c for c in [target_col_a, target_col_b] if c]

                    reinvest_task = InvestigationTask(
                        task_id=reinvest_task_id,
                        task_type=rec_task_type,
                        task=f"Re-investigation (Attempt {attempt}): {rec_method} on {', '.join(reinvest_pair if reinvest_pair else orig_analysis.columns)}",
                        description=f"Follow-up re-analysis recommended by Critic: {current_critic.recommendation or current_critic.summary}",
                        method=rec_method,
                        columns=reinvest_pair if reinvest_pair else orig_analysis.columns,
                        column_a=target_col_a,
                        column_b=target_col_b,
                        group_column=orig_grp if is_group_task else None,
                        value_column=orig_val if is_group_task else None,
                        reason=f"Targeted re-analysis addressing: {current_critic.summary}",
                        priority="high",
                        notes=[f"Source task: {task_id}", f"Attempt: {attempt}", f"Trigger: {current_critic.summary}"],
                    )

                    # Execute re-investigation using existing Analysis Agent
                    try:
                        reinvest_analysis = self.analysis_agent.execute_task(reinvest_task, df)
                    except Exception as e:
                        logger.exception(f"Error executing re-investigation attempt {attempt} for {task_id}: {e}")
                        reinvest_analysis = AnalysisResult(
                            task_id=reinvest_task_id,
                            status="failed",
                            analysis_type=rec_task_type,
                            method=rec_method,
                            columns=orig_analysis.columns,
                            sample_size=0,
                            results={},
                            error={"code": "REINVESTIGATION_FAILED", "message": str(e)},
                        )

                    # Evaluate new result using existing Critic Agent
                    next_critic = self.critic_agent.evaluate_result(reinvest_analysis, request.dataset_profile)

                    # Compute structured comparison between previous analysis and this follow-up
                    comparison = self._compare_results(current_analysis, reinvest_analysis)

                    # Determine Status for this attempt
                    has_var_mismatch = bool(
                        orig_finding_vars and reinvest_pair and set(orig_finding_vars) != set(reinvest_pair)
                    ) or ("Warning: Variable pair changed" in (comparison.effect_size_shift or ""))
                    direction_conflict = comparison.direction_consistent is False
                    sig_lost = bool(
                        comparison.original_p_value is not None
                        and comparison.original_p_value < 0.05
                        and comparison.reinvestigated_p_value is not None
                        and comparison.reinvestigated_p_value >= 0.05
                    )

                    if has_var_mismatch:
                        attempt_status: FinalVerificationStatus = "QUESTIONABLE"
                    elif direction_conflict:
                        attempt_status = "QUESTIONABLE"
                    elif sig_lost:
                        attempt_status = "QUESTIONABLE"
                    elif next_critic.status == "ACCEPTED":
                        attempt_status = "VERIFIED_AFTER_REINVESTIGATION"
                    elif next_critic.status == "QUESTIONABLE":
                        attempt_status = "QUESTIONABLE"
                    else:
                        attempt_status = "REJECTED"

                    # Generate visual evidence for original analysis if not present
                    if current_analysis.status == "success" and not current_analysis.visual_evidence and not df.empty:
                        try:
                            from app.services.visualization_service import get_visualization_service
                            viz_svc = get_visualization_service()
                            current_analysis.visual_evidence = viz_svc.generate_finding_visualization(
                                task_id=current_analysis.task_id,
                                analysis_type=current_analysis.analysis_type,
                                method=current_analysis.method,
                                columns=current_analysis.columns,
                                df=df,
                                results=current_analysis.results,
                            )
                        except Exception as e:
                            logger.warning("Could not generate visual evidence for %s: %s", current_analysis.task_id, e)

                    # Generate visual evidence for follow-up analysis if not present
                    if reinvest_analysis.status == "success" and not reinvest_analysis.visual_evidence and not df.empty:
                        try:
                            from app.services.visualization_service import get_visualization_service
                            viz_svc = get_visualization_service()
                            reinvest_analysis.visual_evidence = viz_svc.generate_finding_visualization(
                                task_id=reinvest_task_id,
                                analysis_type=rec_task_type,
                                method=rec_method,
                                columns=reinvest_pair if reinvest_pair else orig_analysis.columns,
                                df=df,
                                results=reinvest_analysis.results,
                            )
                        except Exception as e:
                            logger.warning("Could not generate follow-up visual evidence for %s: %s", reinvest_task_id, e)

                    orig_graph_id = current_analysis.visual_evidence.graph_id if current_analysis.visual_evidence else None
                    reinvest_graph_id = reinvest_analysis.visual_evidence.graph_id if reinvest_analysis.visual_evidence else None

                    # Construct ReinvestigationRecord
                    reinvest_record_id = f"reinvest_{task_id}_attempt_{attempt}" if attempt > 1 else f"reinvest_{task_id}"
                    reinvest_record = ReinvestigationRecord(
                        reinvestigation_id=reinvest_record_id,
                        source_task_id=task_id,
                        original_method=current_analysis.method,
                        reinvestigated_method=rec_method,
                        variables=reinvest_pair if reinvest_pair else list(orig_analysis.columns),
                        trigger_reason=current_critic.summary,
                        original_analysis=current_analysis,
                        original_critic_review=current_critic,
                        reinvestigated_analysis=reinvest_analysis,
                        second_critic_review=next_critic,
                        comparison=comparison,
                        status=attempt_status,
                        attempt_number=attempt,
                        original_statistic=comparison.original_statistic,
                        reinvestigated_statistic=comparison.reinvestigated_statistic,
                        original_p_value=comparison.original_p_value,
                        reinvestigated_p_value=comparison.reinvestigated_p_value,
                        original_effect_size=comparison.original_effect_size,
                        reinvestigated_effect_size=comparison.reinvestigated_effect_size,
                        original_sample_size=comparison.original_sample_size,
                        reinvestigated_sample_size=comparison.reinvestigated_sample_size,
                        original_graph_id=orig_graph_id,
                        reinvestigated_graph_id=reinvest_graph_id,
                        original_visual_evidence=current_analysis.visual_evidence,
                        reinvestigated_visual_evidence=reinvest_analysis.visual_evidence,
                    )
                    reinvestigations_list.append(reinvest_record)
                    last_reinvest_record = reinvest_record
                    current_analysis = reinvest_analysis
                    current_critic = next_critic

                    # If this attempt succeeded or definitively failed, stop further attempts
                    if next_critic.status in ("ACCEPTED", "REJECTED"):
                        break

                # After loop: if at least 1 attempt ran
                if attempts_run > 0 and last_reinvest_record:
                    last_comp = last_reinvest_record.comparison
                    has_var_mismatch = (
                        "Warning: Variable pair changed" in (last_comp.effect_size_shift or "")
                        or (orig_finding_vars and last_reinvest_record.variables and set(orig_finding_vars) != set(last_reinvest_record.variables))
                    )
                    direction_conflict = last_comp.direction_consistent is False
                    sig_lost = bool(
                        last_comp.original_p_value is not None
                        and last_comp.original_p_value < 0.05
                        and last_comp.reinvestigated_p_value is not None
                        and last_comp.reinvestigated_p_value >= 0.05
                    )

                    extra_codes = []
                    if has_var_mismatch:
                        final_status = "QUESTIONABLE"
                        verif_reason = f"Re-investigation evaluated a mismatched variable pair ({last_comp.effect_size_shift}). Original research question remains unverified."
                        final_summary = f"Questionable finding: Re-investigation evaluated a mismatched variable pair ({last_comp.effect_size_shift})."
                        extra_codes = ["VARIABLE_PAIR_MISMATCH", "QUESTIONABLE_FINDING"]
                    elif direction_conflict:
                        final_status = "QUESTIONABLE"
                        verif_reason = f"Re-investigation via '{current_analysis.method}' contradicted the original direction of effect. Relationship is non-monotonic or unstable."
                        final_summary = f"Questionable finding: Re-investigation contradicted original relationship direction."
                        extra_codes = ["CONFLICTING_DIRECTION", "QUESTIONABLE_FINDING"]
                    elif sig_lost:
                        final_status = "QUESTIONABLE"
                        verif_reason = f"Re-investigation via '{current_analysis.method}' did not replicate statistical significance (original p={last_comp.original_p_value:.4f}, robust p={last_comp.reinvestigated_p_value:.4f}). Finding is sensitive to outlier or distributional violations."
                        final_summary = f"Questionable finding: Statistical significance was not replicated under robust alternative method."
                        extra_codes = ["METHOD_SENSITIVITY_DISCREPANCY", "QUESTIONABLE_FINDING"]
                    elif current_critic.status == "ACCEPTED":
                        final_status = "VERIFIED_AFTER_REINVESTIGATION"
                        verif_reason = f"Re-investigation via '{current_analysis.method}' successfully resolved prior concerns in {attempts_run} attempt(s). Validated by Critic."
                        final_summary = f"Verified after re-investigation: {current_critic.summary}"
                        extra_codes = ["REINVESTIGATION_RESOLVED"]
                    elif current_critic.status == "QUESTIONABLE":
                        final_status = "QUESTIONABLE"
                        verif_reason = f"Re-investigation via '{current_analysis.method}' completed ({attempts_run} attempt(s)) but caveats remain: {current_critic.summary}"
                        final_summary = f"Questionable finding: {current_critic.summary}"
                        extra_codes = ["QUESTIONABLE_FINDING"]
                    else:
                        final_status = "REJECTED"
                        verif_reason = f"Re-investigated test was rejected by Critic after {attempts_run} attempt(s): {current_critic.summary}"
                        final_summary = f"Rejected finding: {current_critic.summary}"
                        extra_codes = ["REJECTED_AFTER_REINVESTIGATION"]

                    eff_visual = current_analysis.visual_evidence if final_status == "VERIFIED_AFTER_REINVESTIGATION" else orig_analysis.visual_evidence
                    eff_graph_id = eff_visual.graph_id if eff_visual else None

                    final_codes = list(current_critic.reason_codes or [])
                    for c in extra_codes:
                        if c not in final_codes:
                            final_codes.append(c)

                    finding = VerifiedFinding(
                        finding_id=finding_id,
                        task_id=task_id,
                        variables=reinvest_pair if reinvest_pair else orig_finding_vars,
                        method=current_analysis.method,
                        status=final_status,
                        confidence=current_critic.confidence,
                        confidence_level=current_critic.confidence_level,
                        summary=final_summary,
                        statistical_evidence=current_analysis.results,
                        critic_assessment=current_critic.summary,
                        reinvestigated=True,
                        reinvestigation_record=last_reinvest_record,
                        graph_id=eff_graph_id,
                        visual_evidence=eff_visual,
                        verification_reason=verif_reason,
                        limitations=current_critic.data_quality_concerns + current_critic.assumption_concerns,
                        reason_codes=final_codes,
                    )
                else:
                    # 0 attempts ran because method was unsupported or duplicate
                    direct_graph_id = orig_analysis.visual_evidence.graph_id if orig_analysis.visual_evidence else None
                    unsupported_note = f" (Recommended method '{critic_rev.recommended_method}' could not be executed: not supported or duplicate)."
                    finding = VerifiedFinding(
                        finding_id=finding_id,
                        task_id=task_id,
                        variables=orig_finding_vars,
                        method=orig_analysis.method,
                        status="QUESTIONABLE",
                        confidence=critic_rev.confidence,
                        confidence_level=critic_rev.confidence_level,
                        summary=f"{critic_rev.summary}{unsupported_note}",
                        statistical_evidence=orig_analysis.results,
                        critic_assessment=critic_rev.summary,
                        reinvestigated=False,
                        reinvestigation_record=None,
                        graph_id=direct_graph_id,
                        visual_evidence=orig_analysis.visual_evidence,
                        verification_reason=f"Finding retained with documented caveats: alternative method '{critic_rev.recommended_method}' could not be executed.",
                        limitations=critic_rev.data_quality_concerns + critic_rev.assumption_concerns,
                        reason_codes=critic_rev.reason_codes + ["QUESTIONABLE_FINDING"],
                    )

            # ----------------------------------------------------------
            # CASE 2: RE-INVESTIGATION DISABLED (max_attempts == 0)
            # ----------------------------------------------------------
            elif should_reinvestigate and max_attempts == 0:
                direct_graph_id = orig_analysis.visual_evidence.graph_id if orig_analysis.visual_evidence else None
                finding = VerifiedFinding(
                    finding_id=finding_id,
                    task_id=task_id,
                    variables=orig_finding_vars,
                    method=orig_analysis.method,
                    status="QUESTIONABLE",
                    confidence=critic_rev.confidence,
                    confidence_level=critic_rev.confidence_level,
                    summary=f"{critic_rev.summary} (Re-investigation disabled: max_reinvestigations set to 0).",
                    statistical_evidence=orig_analysis.results,
                    critic_assessment=critic_rev.summary,
                    reinvestigated=False,
                    reinvestigation_record=None,
                    graph_id=direct_graph_id,
                    visual_evidence=orig_analysis.visual_evidence,
                    verification_reason="Re-investigation disabled (max_reinvestigations=0); finding retained as questionable with unverified caveats.",
                    limitations=critic_rev.data_quality_concerns + critic_rev.assumption_concerns,
                    reason_codes=critic_rev.reason_codes + ["REINVESTIGATION_DISABLED", "QUESTIONABLE_FINDING"],
                )

            # ----------------------------------------------------------
            # CASE 2B: DATASET UNAVAILABLE FOR REQUIRED RE-INVESTIGATION
            # ----------------------------------------------------------
            elif critic_wants_reinvestigation and df.empty:
                direct_graph_id = orig_analysis.visual_evidence.graph_id if orig_analysis.visual_evidence else None
                rec_m = critic_rev.recommended_method
                verif_reason = (
                    f"Re-investigation via '{rec_m}' could not be completed: original dataset '{dataset_id}' "
                    f"could not be resolved from memory cache or storage. Finding retained as questionable with unverified caveats."
                )
                finding = VerifiedFinding(
                    finding_id=finding_id,
                    task_id=task_id,
                    variables=orig_finding_vars,
                    method=orig_analysis.method,
                    status="QUESTIONABLE",
                    confidence=critic_rev.confidence,
                    confidence_level=critic_rev.confidence_level,
                    summary=f"{critic_rev.summary} (Re-investigation via '{rec_m}' precluded: dataset unavailable).",
                    statistical_evidence=orig_analysis.results,
                    critic_assessment=critic_rev.summary,
                    reinvestigated=False,
                    reinvestigation_record=None,
                    graph_id=direct_graph_id,
                    visual_evidence=orig_analysis.visual_evidence,
                    verification_reason=verif_reason,
                    limitations=critic_rev.data_quality_concerns + critic_rev.assumption_concerns + [
                        f"Re-investigation precluded because dataset '{dataset_id}' was unavailable in memory or storage."
                    ],
                    reason_codes=critic_rev.reason_codes + [
                        "DATASET_UNAVAILABLE_FOR_REINVESTIGATION",
                        "QUESTIONABLE_FINDING",
                    ],
                )

            # ----------------------------------------------------------
            # CASE 3: DIRECT VERIFICATION (No Re-investigation)
            # ----------------------------------------------------------
            elif critic_rev.status == "ACCEPTED":
                direct_graph_id = orig_analysis.visual_evidence.graph_id if orig_analysis.visual_evidence else None
                finding = VerifiedFinding(
                    finding_id=finding_id,
                    task_id=task_id,
                    variables=orig_finding_vars,
                    method=orig_analysis.method,
                    status="VERIFIED",
                    confidence=critic_rev.confidence,
                    confidence_level=critic_rev.confidence_level,
                    summary=critic_rev.summary,
                    statistical_evidence=orig_analysis.results,
                    critic_assessment=critic_rev.summary,
                    reinvestigated=False,
                    reinvestigation_record=None,
                    graph_id=direct_graph_id,
                    visual_evidence=orig_analysis.visual_evidence,
                    verification_reason="Analysis completed cleanly and was directly accepted by the Critic Agent with zero unresolved issues.",
                    limitations=critic_rev.data_quality_concerns + critic_rev.assumption_concerns,
                    reason_codes=critic_rev.reason_codes + ["DIRECTLY_VERIFIED"],
                )

            # ----------------------------------------------------------
            # CASE 3: REJECTED FINDINGS (Audited & Excluded from Insights)
            # ----------------------------------------------------------
            elif critic_rev.status == "REJECTED":
                direct_graph_id = orig_analysis.visual_evidence.graph_id if orig_analysis.visual_evidence else None
                finding = VerifiedFinding(
                    finding_id=finding_id,
                    task_id=task_id,
                    variables=orig_finding_vars,
                    method=orig_analysis.method,
                    status="REJECTED",
                    confidence=critic_rev.confidence,
                    confidence_level=critic_rev.confidence_level,
                    summary=f"Rejected: {critic_rev.summary}",
                    statistical_evidence=orig_analysis.results,
                    critic_assessment=critic_rev.summary,
                    reinvestigated=False,
                    reinvestigation_record=None,
                    graph_id=direct_graph_id,
                    visual_evidence=orig_analysis.visual_evidence,
                    verification_reason=f"Finding rejected by Critic: {critic_rev.summary}",
                    limitations=critic_rev.data_quality_concerns + critic_rev.assumption_concerns,
                    reason_codes=critic_rev.reason_codes + ["VERIFICATION_REJECTED"],
                )

            # ----------------------------------------------------------
            # CASE 4: QUESTIONABLE FINDINGS (Retained with Caveats)
            # ----------------------------------------------------------
            else:
                unsupported_note = ""
                if should_reinvestigate and not method_spec:
                    unsupported_note = f" (Recommended method '{critic_rev.recommended_method}' is not supported for automated execution; manual review advised)."

                direct_graph_id = orig_analysis.visual_evidence.graph_id if orig_analysis.visual_evidence else None
                finding = VerifiedFinding(
                    finding_id=finding_id,
                    task_id=task_id,
                    variables=orig_finding_vars,
                    method=orig_analysis.method,
                    status="QUESTIONABLE",
                    confidence=critic_rev.confidence,
                    confidence_level=critic_rev.confidence_level,
                    summary=f"{critic_rev.summary}{unsupported_note}",
                    statistical_evidence=orig_analysis.results,
                    critic_assessment=critic_rev.summary,
                    reinvestigated=False,
                    reinvestigation_record=None,
                    graph_id=direct_graph_id,
                    visual_evidence=orig_analysis.visual_evidence,
                    verification_reason=f"Finding retained with documented caveats: {'; '.join(critic_rev.data_quality_concerns + critic_rev.assumption_concerns) or 'Power limitations.'}",
                    limitations=critic_rev.data_quality_concerns + critic_rev.assumption_concerns,
                    reason_codes=critic_rev.reason_codes + ["QUESTIONABLE_FINDING"],
                )

            # Categorize into bucket lists
            all_findings.append(finding)
            if finding.status in ("VERIFIED", "VERIFIED_AFTER_REINVESTIGATION"):
                verified_list.append(finding)
            elif finding.status == "QUESTIONABLE":
                questionable_list.append(finding)
            elif finding.status == "REJECTED":
                rejected_list.append(finding)

        # 5. Dynamic Summary Synthesis
        total_f = len(all_findings)
        ver_direct_cnt = sum(1 for f in all_findings if f.status == "VERIFIED")
        ver_reinvest_cnt = sum(1 for f in all_findings if f.status == "VERIFIED_AFTER_REINVESTIGATION")
        quest_cnt = len(questionable_list)
        rej_cnt = len(rejected_list)
        reinvest_total = len(reinvestigations_list)

        unresolved: list[str] = []
        for f in questionable_list + rejected_list:
            for lim in f.limitations:
                if lim not in unresolved:
                    unresolved.append(f"[{f.task_id}] {lim}")

        if total_f == 0:
            overall_integ = "No findings available for verification."
        elif ver_direct_cnt + ver_reinvest_cnt >= total_f * 0.75:
            overall_integ = "High Empirical Confidence: Substantial majority of findings verified and validated."
        elif ver_direct_cnt + ver_reinvest_cnt + quest_cnt >= total_f * 0.60:
            overall_integ = "Moderate Empirical Confidence: Verified findings supported with documented limitations."
        else:
            overall_integ = "Limited Empirical Confidence: Analytical constraints affect several planned findings."

        summary = VerificationSummary(
            total_findings=total_f,
            verified=ver_direct_cnt,
            verified_after_reinvestigation=ver_reinvest_cnt,
            questionable=quest_cnt,
            rejected=rej_cnt,
            total_reinvestigated=reinvest_total,
            overall_integrity=overall_integ,
            unresolved_issues=unresolved[:10],
        )

        return VerificationResponse(
            dataset_id=dataset_id,
            summary=summary,
            verified_findings=verified_list,
            questionable_findings=questionable_list,
            rejected_findings=rejected_list,
            reinvestigations=reinvestigations_list,
            all_findings=all_findings,
            phase_version="5.0",
        )

    # ------------------------------------------------------------------
    # INTERNAL BEFORE / AFTER COMPARISON
    # ------------------------------------------------------------------

    def _compare_results(
        self,
        orig: AnalysisResult,
        reinvest: AnalysisResult,
    ) -> ResultComparison:
        """
        Compare original vs re-investigated statistical evidence strictly from Python outputs.
        """
        orig_data = orig.results or {}
        new_data = reinvest.results or {}

        # Extract primary test statistics
        orig_stat = orig_data.get("correlation_coefficient", orig_data.get("statistic"))
        new_stat = new_data.get("correlation_coefficient", new_data.get("statistic"))

        # Extract p-values
        orig_p = orig_data.get("p_value")
        new_p = new_data.get("p_value")

        # Extract sample sizes
        orig_n = orig.sample_size or orig_data.get("sample_size")
        new_n = reinvest.sample_size or new_data.get("sample_size")

        # Extract effect sizes
        orig_eff = orig_data.get("effect_size")
        new_eff = new_data.get("effect_size")
        if orig_eff is None and orig_stat is not None and "correlation" in orig.method.lower():
            orig_eff = round(abs(float(orig_stat)), 4)
        if new_eff is None and new_stat is not None and "correlation" in reinvest.method.lower():
            new_eff = round(abs(float(new_stat)), 4)

        # Check variable pair alignment
        orig_var_a = orig_data.get("column_a") or orig_data.get("group_column") or (orig.columns[0] if len(orig.columns) > 0 else None)
        orig_var_b = orig_data.get("column_b") or orig_data.get("value_column") or (orig.columns[1] if len(orig.columns) > 1 else None)
        new_var_a = new_data.get("column_a") or new_data.get("group_column") or (reinvest.columns[0] if len(reinvest.columns) > 0 else None)
        new_var_b = new_data.get("column_b") or new_data.get("value_column") or (reinvest.columns[1] if len(reinvest.columns) > 1 else None)

        orig_pair = {orig_var_a, orig_var_b} - {None}
        new_pair = {new_var_a, new_var_b} - {None}
        pair_matched = True
        if orig_pair and new_pair:
            pair_matched = bool(orig_pair == new_pair)

        # Direction consistency
        direction_consistent: Optional[bool] = None
        orig_dir = orig_data.get("direction")
        new_dir = new_data.get("direction")
        if orig_dir and new_dir:
            direction_consistent = bool(orig_dir == new_dir)
        elif orig_stat is not None and new_stat is not None:
            direction_consistent = bool((orig_stat > 0) == (new_stat > 0))

        # Significance consistency
        significance_consistent: Optional[bool] = None
        if orig_p is not None and new_p is not None:
            significance_consistent = bool((orig_p < 0.05) == (new_p < 0.05))

        # Magnitude delta
        magnitude_delta: Optional[float] = None
        if orig_stat is not None and new_stat is not None:
            try:
                magnitude_delta = round(abs(float(orig_stat) - float(new_stat)), 4)
            except Exception:
                pass

        # Qualitative shift
        effect_shift = "Consistent relationship"
        if not pair_matched and orig_pair and new_pair:
            effect_shift = f"Warning: Variable pair changed ({orig_var_a} vs {orig_var_b} -> {new_var_a} vs {new_var_b})"
            direction_consistent = False
            significance_consistent = False
        elif significance_consistent is False:
            effect_shift = (
                "Significance changed (parametric vs non-parametric)"
            )
        elif magnitude_delta is not None and magnitude_delta > 0.2:
            effect_shift = f"Magnitude shifted by {magnitude_delta:.3f}"

        # Factual summary
        stat_desc_orig = f"stat={orig_stat:.4f}" if orig_stat is not None else "uncomputed"
        stat_desc_new = f"stat={new_stat:.4f}" if new_stat is not None else "uncomputed"
        p_desc_orig = f"p={orig_p:.4f}" if orig_p is not None else "p=N/A"
        p_desc_new = f"p={new_p:.4f}" if new_p is not None else "p=N/A"

        summary_text = (
            f"Original ({orig.method}): {stat_desc_orig}, {p_desc_orig} (N={orig_n}). "
            f"Re-investigation ({reinvest.method}): {stat_desc_new}, {p_desc_new} (N={new_n}). "
            f"Significance consistent: {significance_consistent}."
        )

        return ResultComparison(
            original_method=orig.method,
            reinvestigated_method=reinvest.method,
            original_statistic=round(float(orig_stat), 4) if orig_stat is not None else None,
            reinvestigated_statistic=round(float(new_stat), 4) if new_stat is not None else None,
            original_p_value=float(orig_p) if orig_p is not None else None,
            reinvestigated_p_value=float(new_p) if new_p is not None else None,
            original_sample_size=int(orig_n) if orig_n is not None else None,
            reinvestigated_sample_size=int(new_n) if new_n is not None else None,
            original_effect_size=round(float(orig_eff), 4) if orig_eff is not None else None,
            reinvestigated_effect_size=round(float(new_eff), 4) if new_eff is not None else None,
            direction_consistent=direction_consistent,
            significance_consistent=significance_consistent,
            magnitude_delta=magnitude_delta,
            effect_size_shift=effect_shift,
            summary=summary_text,
        )


_VERIFICATION_SERVICE = VerificationService()


def get_verification_service() -> VerificationService:
    """Provide the global VerificationService instance."""
    return _VERIFICATION_SERVICE

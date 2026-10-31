"""
backend/app/services/experiment_runner.py
=========================================
Paired Experimentation Engine for DataInvest AI.
Executes paired scientific evaluations on datasets:
- Generates exactly ONE investigation plan using the genuine LLM Planner (is_research_run=True).
- Executes Python/SciPy statistical analysis tasks once.
- Baseline branch: Plan -> Analysis -> Evaluation (Critic, Verification, Insights bypassed).
- Proposed branch: Same Plan -> Same Analysis -> Critic -> Verification -> Insights -> Evaluation.
- Compares empirical metrics and scorecards solely derived from actual service outputs.
- Persists both runs with distinct, linked IDs (or skips persistence when persist_runs=False).
"""

from __future__ import annotations

import logging
from pathlib import Path
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Union

from app.schemas.analysis_schema import PlanAnalysisRequest, PlanAnalysisResponse
from app.schemas.critic_schema import CriticRequest, CriticResponse
from app.schemas.evaluation_schema import EvaluationResponse
from app.schemas.experiment_schema import (
    BranchRunSummary,
    MetricDeltaSummary,
    PairedExperimentRequest,
    PairedExperimentResponse,
)
from app.schemas.insight_schema import InsightRequest, InsightResponse
from app.schemas.planner_schema import PlannerResponse
from app.schemas.verification_schema import VerificationRequest, VerificationResponse
from app.services.analysis_service import AnalysisService, get_analysis_service
from app.services.critic_service import CriticService, get_critic_service
from app.services.evaluation_service import EvaluationService, evaluation_service
from app.services.gemini_client import GeminiClient
from app.services.insight_service import InsightService, get_insight_service
from app.services.planner_service import create_analysis_plan
from app.services.experiment_logger import ExperimentLogger, experiment_logger
from app.services.run_store import RunStoreService
from app.services.verification_service import VerificationService, get_verification_service

logger = logging.getLogger("datainvest.experiment_runner")


class ExperimentRunnerError(Exception):
    """Base exception for paired experiment execution failures."""
    pass


class ExperimentPersistenceError(ExperimentRunnerError):
    """Raised when one or both experiment branches fail to persist."""

    def __init__(
        self,
        message: str,
        partial_persistence: bool = False,
        saved_run_id: Optional[str] = None,
    ):
        super().__init__(message)
        self.partial_persistence = partial_persistence
        self.saved_run_id = saved_run_id


def _sanitize_slug(name: str) -> str:
    """Convert filename or dataset identifier to safe alphanumeric slug."""
    clean = re.sub(r"[^\w\s-]", "", name).strip().lower()
    clean = re.sub(r"[-\s]+", "_", clean)
    return clean[:30] if clean else "dataset"


def _sanitize_error_message(msg: str) -> str:
    """Mask potential API keys or secrets in error messages."""
    return re.sub(r"AIza[0-9A-Za-z-_]{35}", "[REDACTED_API_KEY]", msg)


def run_paired_experiment(
    profile: Optional[Dict[str, Any]] = None,
    dataset_id: Optional[str] = None,
    objective: str = "Investigate this dataset.",
    max_reinvestigations: int = 1,
    persist_runs: bool = True,
    request: Optional[PairedExperimentRequest] = None,
    gemini_client: Optional[GeminiClient] = None,
    planner_fn: Optional[Callable[..., PlannerResponse]] = None,
    analysis_service: Optional[AnalysisService] = None,
    critic_service: Optional[CriticService] = None,
    verification_service: Optional[VerificationService] = None,
    insight_service: Optional[InsightService] = None,
    evaluation_service_inst: Optional[EvaluationService] = None,
    run_store_service: Optional[RunStoreService] = None,
    is_offline_fixture: bool = False,
    execution_mode: Optional[str] = None,
    experiment_logger_inst: Optional[ExperimentLogger] = None,
    log_dir: Optional[Union[str, Path]] = None,
) -> PairedExperimentResponse:
    """
    Execute a paired experiment comparing Baseline vs Proposed architectures.

    Research Design:
    1. Generates exactly ONE plan using the genuine LLM Planner (is_research_run=True)
       or deterministic fixture replay if is_offline_fixture=True.
    2. Fails immediately if the LLM plan fails in research mode (zero fallback).
    3. Executes Python-based statistical analysis for that plan once.
    4. Passes the exact same plan and raw analysis results to both branches.
    5. Baseline: Evaluates Plan + Analysis with Critic=None, Verification=None, Insights=None.
    6. Proposed: Evaluates Plan + Analysis with Critic, Re-investigation, Insights, and Evaluation.
    7. Calculates comparison metrics purely from actual service outputs.
    8. Persists linked records using RunStoreService if persist_runs=True.
    9. Instruments monotonic wall-clock durations and rule-level Critic metrics.

    Parameters
    ----------
    profile : dict, optional
        Phase 1 Dataset Profile dictionary.
    dataset_id : str, optional
        Dataset identifier or filename for resolution and tracking.
    objective : str
        User's analytical objective in natural language.
    max_reinvestigations : int
        Maximum re-investigation loops for flagged findings (0 to 3, default 1).
    persist_runs : bool
        Whether to save both runs to backend/data/runs/ on disk.
    request : PairedExperimentRequest, optional
        Pre-validated Pydantic request model.
    gemini_client : GeminiClient, optional
        Injected Gemini client (e.g., with MockTransport for offline tests).
    planner_fn : Callable, optional
        Planning function (defaults to app.services.planner_service.create_analysis_plan).
    analysis_service : AnalysisService, optional
        Injected AnalysisService instance.
    critic_service : CriticService, optional
        Injected CriticService instance.
    verification_service : VerificationService, optional
        Injected VerificationService instance.
    insight_service : InsightService, optional
        Injected InsightService instance.
    evaluation_service_inst : EvaluationService, optional
        Injected EvaluationService instance.
    run_store_service : RunStoreService, optional
        Injected RunStoreService instance.
    is_offline_fixture : bool, optional
        Whether this is an offline fixture replay run (defaults to False).
    execution_mode : str, optional
        Explicit label for execution mode ('research_run' or 'offline_fixture_replay').

    Returns
    -------
    PairedExperimentResponse
        Strongly typed comparison results with shared plan and branch summaries.
    """
    # Task 1: Start total monotonic wall-clock timer
    t_exp_start = time.perf_counter()

    # 1. Resolve and validate request parameters
    if request is not None:
        prof = request.profile
        ds_id = request.dataset_id
        obj = request.objective
        max_reinv = request.max_reinvestigations
        persist = request.persist_runs
    else:
        # Validate through schema
        req_obj = PairedExperimentRequest(
            profile=profile or {},
            dataset_id=dataset_id,
            objective=objective,
            max_reinvestigations=max_reinvestigations,
            persist_runs=persist_runs,
        )
        prof = req_obj.profile
        ds_id = req_obj.dataset_id
        obj = req_obj.objective
        max_reinv = req_obj.max_reinvestigations
        persist = req_obj.persist_runs

    overview = prof.get("overview") or {}
    dataset_name = ds_id or overview.get("filename") or "dataset.csv"
    resolved_dataset_id = ds_id or dataset_name

    now_utc = datetime.now(timezone.utc)
    ts = now_utc.strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    slug = _sanitize_slug(dataset_name)
    experiment_id = f"exp_{ts}_{slug}_{suffix}"
    baseline_run_id = f"run_{ts}_{slug}_baseline_{suffix}"
    proposed_run_id = f"run_{ts}_{slug}_proposed_{suffix}"

    logger.info(
        "Initiating paired experiment %s for dataset '%s' (max_reinvestigations=%d, persist=%s, offline_fixture=%s)",
        experiment_id,
        resolved_dataset_id,
        max_reinv,
        persist,
        is_offline_fixture,
    )

    # 2. Resolve services
    plan_func = planner_fn or create_analysis_plan
    analysis_svc = analysis_service or get_analysis_service()
    critic_svc = critic_service or get_critic_service()
    verif_svc = verification_service or get_verification_service()
    insight_svc = insight_service or get_insight_service()
    eval_svc = evaluation_service_inst or evaluation_service
    run_store = run_store_service or RunStoreService()

    # Resolve event logger: custom instance > explicit log_dir > default historical logger
    if experiment_logger_inst is not None:
        exp_logger = experiment_logger_inst
    elif log_dir is not None:
        # Explicit log directory requested: validate and instantiate dedicated logger.
        # Fails loudly via validate_safe_log_dir without falling back to default historical logs.
        exp_logger = ExperimentLogger(logs_dir=log_dir)
    else:
        exp_logger = experiment_logger

    # 3. Stage 1: Generate exactly ONE plan (shared across both branches)
    # In research mode: must fail immediately if LLM call or validation fails. Zero deterministic fallback.
    # In offline fixture mode: deterministic or fixture replay is explicitly used and labeled.
    t_plan_start = time.perf_counter()
    try:
        import inspect
        sig = inspect.signature(plan_func)
        has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        plan_kwargs: dict[str, Any] = {
            "profile": prof,
            "objective": obj,
            "dataset_id": resolved_dataset_id,
            "gemini_client": gemini_client,
            "is_research_run": not is_offline_fixture,
            "use_llm": not is_offline_fixture,
        }

        if not has_var_keyword:
            plan_kwargs = {k: v for k, v in plan_kwargs.items() if k in sig.parameters}

        shared_plan: PlannerResponse = plan_func(**plan_kwargs)
    except Exception as exc:
        sanitized_err = _sanitize_error_message(str(exc))
        logger.error(
            "Paired experiment %s aborted at planning stage: %s: %s",
            experiment_id,
            type(exc).__name__,
            sanitized_err,
        )
        # Propagate genuine planning / Gemini error directly without swallowing
        raise
    finally:
        t_plan_end = time.perf_counter()

    planning_duration = max(0.0, round(t_plan_end - t_plan_start, 4))

    if not shared_plan or not shared_plan.analysis_tasks:
        raise ExperimentRunnerError(
            f"Planner returned an empty plan with no analysis tasks for experiment {experiment_id}."
        )

    # Log Planner completion
    exp_logger.log_event(
        run_id=proposed_run_id,
        event_type="PLANNER_COMPLETED",
        phase=2,
        outcome="SUCCESS",
        details={"tasks_planned": len(shared_plan.analysis_tasks), "objective": obj},
        experiment_id=experiment_id,
    )
    exp_logger.log_event(
        run_id=baseline_run_id,
        event_type="PLANNER_COMPLETED",
        phase=2,
        outcome="SUCCESS",
        details={"tasks_planned": len(shared_plan.analysis_tasks), "objective": obj},
        experiment_id=experiment_id,
    )

    # 4. Stage 2: Execute Python-based statistical analysis once
    t_analysis_start = time.perf_counter()
    try:
        analysis_request = PlanAnalysisRequest(
            dataset_id=resolved_dataset_id,
            tasks=shared_plan.analysis_tasks,
        )
        raw_analysis_response: PlanAnalysisResponse = analysis_svc.execute_plan(analysis_request)
    except Exception as exc:
        sanitized_err = _sanitize_error_message(str(exc))
        logger.error(
            "Paired experiment %s failed during statistical analysis: %s: %s",
            experiment_id,
            type(exc).__name__,
            sanitized_err,
        )
        raise ExperimentRunnerError(f"Statistical analysis stage failed: {sanitized_err}") from exc
    finally:
        t_analysis_end = time.perf_counter()

    analysis_duration = max(0.0, round(t_analysis_end - t_analysis_start, 4))
    shared_duration = round(planning_duration + analysis_duration, 4)

    # Log Analysis completion
    exp_logger.log_event(
        run_id=proposed_run_id,
        event_type="ANALYSIS_COMPLETED",
        phase=3,
        outcome="SUCCESS",
        details={"tasks_executed": len(raw_analysis_response.results)},
        experiment_id=experiment_id,
    )
    exp_logger.log_event(
        run_id=baseline_run_id,
        event_type="ANALYSIS_COMPLETED",
        phase=3,
        outcome="SUCCESS",
        details={"tasks_executed": len(raw_analysis_response.results)},
        experiment_id=experiment_id,
    )

    # Serialize shared artifacts once for exact reuse across branches
    plan_dict = shared_plan.model_dump()
    analysis_dict = raw_analysis_response.model_dump()

    # 5. Stage 3: Baseline Branch Execution
    # Baseline skips Critic (Phase 4), Verification (Phase 5), Insights (Phase 6)
    # Stored critic=None, verification=None, insights=None
    t_base_eval_start = time.perf_counter()
    try:
        baseline_eval: EvaluationResponse = eval_svc.evaluate_pipeline(
            profile=prof,
            plan=plan_dict,
            analysis=analysis_dict,
            critic=None,
            verification=None,
            insights=None,
        )
    except Exception as exc:
        sanitized_err = _sanitize_error_message(str(exc))
        logger.error(
            "Paired experiment %s baseline evaluation failed: %s: %s",
            experiment_id,
            type(exc).__name__,
            sanitized_err,
        )
        raise ExperimentRunnerError(f"Baseline evaluation stage failed: {sanitized_err}") from exc
    finally:
        t_base_eval_end = time.perf_counter()

    baseline_duration = max(0.0, round(t_base_eval_end - t_base_eval_start, 4))

    # 6. Stage 4: Proposed Branch Execution
    # Phase 4: Critic review
    t_critic_start = time.perf_counter()
    try:
        critic_req = CriticRequest(
            analysis_response=raw_analysis_response,
            dataset_profile=prof,
            investigation_plan=shared_plan,
            dataset_id=resolved_dataset_id,
        )
        critic_resp: CriticResponse = critic_svc.run_review(critic_req)
        critic_dict = critic_resp.model_dump()
    except Exception as exc:
        sanitized_err = _sanitize_error_message(str(exc))
        logger.error(
            "Paired experiment %s critic review failed: %s: %s",
            experiment_id,
            type(exc).__name__,
            sanitized_err,
        )
        raise ExperimentRunnerError(f"Critic review stage failed: {sanitized_err}") from exc
    finally:
        t_critic_end = time.perf_counter()

    critic_duration = max(0.0, round(t_critic_end - t_critic_start, 4))

    # Phase 5: Verification & Controlled Re-investigation
    t_verif_start = time.perf_counter()
    try:
        verification_req = VerificationRequest(
            analysis_response=raw_analysis_response,
            critic_response=critic_resp,
            dataset_profile=prof,
            investigation_plan=shared_plan,
            dataset_id=resolved_dataset_id,
            max_reinvestigations=max_reinv,
        )
        verif_resp: VerificationResponse = verif_svc.process_verification(verification_req)
        verif_dict = verif_resp.model_dump()
    except Exception as exc:
        sanitized_err = _sanitize_error_message(str(exc))
        logger.error(
            "Paired experiment %s verification/reinvestigation failed: %s: %s",
            experiment_id,
            type(exc).__name__,
            sanitized_err,
        )
        raise ExperimentRunnerError(f"Verification stage failed: {sanitized_err}") from exc
    finally:
        t_verif_end = time.perf_counter()

    verif_duration = max(0.0, round(t_verif_end - t_verif_start, 4))

    # Phase 6: Verified Insight Generation
    t_insight_start = time.perf_counter()
    try:
        insight_req = InsightRequest(
            verification_response=verif_resp,
            dataset_profile=prof,
            investigation_plan=shared_plan,
            dataset_id=resolved_dataset_id,
        )
        insight_resp: InsightResponse = insight_svc.generate_insights(insight_req)
        insight_dict = insight_resp.model_dump()
    except Exception as exc:
        sanitized_err = _sanitize_error_message(str(exc))
        logger.error(
            "Paired experiment %s insight generation failed: %s: %s",
            experiment_id,
            type(exc).__name__,
            sanitized_err,
        )
        raise ExperimentRunnerError(f"Insight generation stage failed: {sanitized_err}") from exc
    finally:
        t_insight_end = time.perf_counter()

    insight_duration = max(0.0, round(t_insight_end - t_insight_start, 4))

    # Phase 7: Proposed Pipeline System Evaluation
    t_prop_eval_start = time.perf_counter()
    try:
        proposed_eval: EvaluationResponse = eval_svc.evaluate_pipeline(
            profile=prof,
            plan=plan_dict,
            analysis=analysis_dict,
            critic=critic_dict,
            verification=verif_dict,
            insights=insight_dict,
        )
    except Exception as exc:
        sanitized_err = _sanitize_error_message(str(exc))
        logger.error(
            "Paired experiment %s proposed evaluation failed: %s: %s",
            experiment_id,
            type(exc).__name__,
            sanitized_err,
        )
        raise ExperimentRunnerError(f"Proposed evaluation stage failed: {sanitized_err}") from exc
    finally:
        t_prop_eval_end = time.perf_counter()

    prop_eval_duration = max(0.0, round(t_prop_eval_end - t_prop_eval_start, 4))

    # Duration aggregations
    # Fairness guarantee: shared_duration is strictly isolated and not claimed by either branch.
    # proposed_overhead directly isolates Critic self-verification and reinvestigation overhead.
    proposed_overhead = round(critic_duration + verif_duration, 4)
    proposed_duration = round(critic_duration + verif_duration + insight_duration + prop_eval_duration, 4)

    # 7. Stage 5: Calculate comparison metrics strictly from actual service outputs
    baseline_score = float(baseline_eval.summary.overall_score)
    proposed_score = float(proposed_eval.summary.overall_score)
    score_delta = round(proposed_score - baseline_score, 2)

    baseline_dims = {d.dimension_key: float(d.score) for d in baseline_eval.dimensions}
    proposed_dims = {d.dimension_key: float(d.score) for d in proposed_eval.dimensions}
    all_dimension_keys = sorted(list(set(baseline_dims.keys()) | set(proposed_dims.keys())))
    dimension_deltas = {
        k: round(proposed_dims.get(k, 0.0) - baseline_dims.get(k, 0.0), 2)
        for k in all_dimension_keys
    }

    rigor_delta = dimension_deltas.get("statistical_rigor", 0.0)
    verif_delta = dimension_deltas.get("verification_grounding", 0.0)

    # Extract factual verification counts from Critic and Verification summaries
    violations_flagged = int(critic_resp.summary.requires_reinvestigation)
    violations_resolved = int(verif_resp.summary.verified_after_reinvestigation)
    total_reinvestigations = int(verif_resp.summary.total_reinvestigated)
    verified_count = int(verif_resp.summary.verified + verif_resp.summary.verified_after_reinvestigation)
    questionable_count = int(verif_resp.summary.questionable)
    rejected_count = int(verif_resp.summary.rejected)

    resolution_rate: Optional[float] = None
    if violations_flagged > 0:
        resolution_rate = round(violations_resolved / violations_flagged, 4)

    # Task 2: Extract structured rule-level Critic metrics and reason code counts
    rule_breakdown: dict[str, int] = {}
    cochran_applicable = False
    normality_applicable = False
    leverage_applicable = False

    cochran_viol_set: set[str] = set()
    normality_viol_set: set[str] = set()
    leverage_viol_set: set[str] = set()

    for rev in critic_resp.reviews:
        codes = list(rev.reason_codes or [])
        for c in codes:
            rule_breakdown[c] = rule_breakdown.get(c, 0) + 1

        # Check Cochran applicability and violations
        is_categorical = (
            rev.analysis_type == "categorical_vs_categorical"
            or "chi" in rev.method.lower()
            or any(c in codes for c in ("COCHRAN_CONDITION_SATISFIED", "EXPECTED_COUNT_BELOW_ONE", "SPARSE_CONTINGENCY_CELLS"))
        )
        if is_categorical:
            cochran_applicable = True
            if any(c in codes for c in ("EXPECTED_COUNT_BELOW_ONE", "SPARSE_CONTINGENCY_CELLS")):
                cochran_viol_set.add(rev.task_id)

        # Check Normality applicability and violations
        is_normality_relevant = (
            rev.analysis_type in ("numerical_vs_numerical", "two_groups_vs_numerical", "multiple_groups_vs_numerical")
            or any(c in codes for c in ("NORMAL_DISTRIBUTION", "NON_NORMAL_DISTRIBUTION", "RECOMMEND_MANN_WHITNEY", "RECOMMEND_KRUSKAL_WALLIS", "RECOMMEND_SPEARMAN_RANK"))
            or any("normality" in conc.lower() or "skew" in conc.lower() for conc in rev.assumption_concerns)
        )
        if is_normality_relevant:
            normality_applicable = True
            if any(c in codes for c in ("NON_NORMAL_DISTRIBUTION", "RECOMMEND_MANN_WHITNEY", "RECOMMEND_KRUSKAL_WALLIS", "RECOMMEND_SPEARMAN_RANK")) or any(
                "normality" in conc.lower() or "skew" in conc.lower() for conc in rev.assumption_concerns
            ):
                normality_viol_set.add(rev.task_id)

        # Check Leverage / Outlier applicability and violations
        is_leverage_relevant = (
            any(c in codes for c in ("MINIMAL_OUTLIER_INFLUENCE", "SUBSTANTIAL_OUTLIER_INFLUENCE", "VISUAL_OUTLIERS_DETECTED", "VISUAL_OUTLIERS_REINVESTIGATE"))
            or any("outlier" in conc.lower() or "leverage" in conc.lower() for conc in rev.assumption_concerns)
        )
        if is_leverage_relevant:
            leverage_applicable = True
            if any(c in codes for c in ("SUBSTANTIAL_OUTLIER_INFLUENCE", "VISUAL_OUTLIERS_REINVESTIGATE")) or any(
                "outlier" in conc.lower() or "leverage" in conc.lower() for conc in rev.assumption_concerns
            ):
                leverage_viol_set.add(rev.task_id)

    cochran_violations_count = len(cochran_viol_set) if cochran_applicable else None
    normality_violations_count = len(normality_viol_set) if normality_applicable else None
    leverage_outlier_violations_count = len(leverage_viol_set) if leverage_applicable else None

    # Dynamically formulate scientific verdict based on actual results (never hardcoded)
    if score_delta > 0:
        verdict = (
            f"Proposed pipeline improved overall score by +{score_delta:.1f} pts "
            f"({baseline_score:.1f} -> {proposed_score:.1f}). Critic audited {critic_resp.summary.total_results} "
            f"findings, flagged {violations_flagged} assumption violation(s), and resolved "
            f"{violations_resolved} via controlled re-investigation."
        )
    elif score_delta < 0:
        verdict = (
            f"Proposed pipeline score was lower by {abs(score_delta):.1f} pts "
            f"({baseline_score:.1f} -> {proposed_score:.1f}) due to strict penalization of questionable findings."
        )
    else:
        verdict = (
            f"Proposed and baseline pipelines achieved equal overall score ({baseline_score:.1f} pts). "
            f"No score delta detected."
        )

    comparison_summary = MetricDeltaSummary(
        baseline_score=baseline_score,
        proposed_score=proposed_score,
        overall_score_delta=score_delta,
        rigor_score_delta=rigor_delta,
        verification_score_delta=verif_delta,
        dimension_deltas=dimension_deltas,
        assumption_violations_flagged=violations_flagged,
        violations_resolved_via_reinvestigation=violations_resolved,
        resolution_rate=resolution_rate,
        total_reinvestigations=total_reinvestigations,
        verified_findings_count=verified_count,
        questionable_findings_count=questionable_count,
        rejected_findings_count=rejected_count,
        initial_findings_count=len(raw_analysis_response.results),
        findings_accepted_directly=int(verif_resp.summary.verified),
        findings_requiring_reinvestigation=violations_flagged,
        findings_revised=violations_resolved,
        findings_unresolved=questionable_count + rejected_count,
        scientific_verdict=verdict,
        total_experiment_duration_seconds=None,  # Populated right before return
        shared_duration_seconds=shared_duration,
        planning_duration_seconds=planning_duration,
        analysis_duration_seconds=analysis_duration,
        critic_duration_seconds=critic_duration,
        verification_duration_seconds=verif_duration,
        proposed_overhead_seconds=proposed_overhead,
        baseline_duration_seconds=baseline_duration,
        proposed_duration_seconds=proposed_duration,
        cochran_violations_count=cochran_violations_count,
        normality_violations_count=normality_violations_count,
        leverage_outlier_violations_count=leverage_outlier_violations_count,
        rule_breakdown=rule_breakdown,
    )

    # 8. Stage 6: Persistence and Linking
    # Log Critic, Reinvestigation, and Evaluation outcomes
    exp_logger.log_event(
        run_id=proposed_run_id,
        event_type="CRITIC_COMPLETED",
        phase=4,
        outcome="FLAGGED" if violations_flagged > 0 else "PASSED",
        details={
            "audited": critic_resp.summary.total_results,
            "flagged": violations_flagged,
            "accepted": critic_resp.summary.accepted,
        },
        experiment_id=experiment_id,
    )
    exp_logger.log_event(
        run_id=proposed_run_id,
        event_type="REINVESTIGATION_COMPLETED",
        phase=5,
        outcome="SUCCESS",
        details={
            "total_reinvestigated": total_reinvestigations,
            "verified_after_reinvestigation": violations_resolved,
            "questionable": questionable_count,
            "rejected": rejected_count,
        },
        experiment_id=experiment_id,
    )
    exp_logger.log_event(
        run_id=baseline_run_id,
        event_type="EVALUATION_COMPLETED",
        phase=7,
        outcome="SUCCESS",
        details={"overall_score": baseline_score, "overall_grade": baseline_eval.summary.overall_grade},
        experiment_id=experiment_id,
    )
    exp_logger.log_event(
        run_id=proposed_run_id,
        event_type="EVALUATION_COMPLETED",
        phase=7,
        outcome="SUCCESS",
        details={"overall_score": proposed_score, "overall_grade": proposed_eval.summary.overall_grade},
        experiment_id=experiment_id,
    )

    baseline_summary = BranchRunSummary(
        run_id=baseline_run_id,
        mode="baseline",
        dataset_id=resolved_dataset_id,
        total_tasks_executed=baseline_eval.summary.total_tasks_executed,
        overall_score=baseline_score,
        overall_grade=baseline_eval.summary.overall_grade,
        dimension_scores=baseline_dims,
        completed_phases=[1, 2, 3, 7],
        has_critic_review=False,
        has_verification=False,
        has_insights=False,
        persisted=False,
        paired_run_id=proposed_run_id,
        duration_seconds=baseline_duration,
    )

    proposed_summary = BranchRunSummary(
        run_id=proposed_run_id,
        mode="proposed",
        dataset_id=resolved_dataset_id,
        total_tasks_executed=proposed_eval.summary.total_tasks_executed,
        overall_score=proposed_score,
        overall_grade=proposed_eval.summary.overall_grade,
        dimension_scores=proposed_dims,
        completed_phases=[1, 2, 3, 4, 5, 6, 7],
        has_critic_review=True,
        has_verification=True,
        has_insights=True,
        persisted=False,
        paired_run_id=baseline_run_id,
        duration_seconds=proposed_duration,
    )

    persisted_successfully = False
    if persist:
        created_at_iso = now_utc.isoformat()

        # Build baseline RunPayload: critic, verification, insights honestly set to None
        baseline_payload = {
            "run_id": baseline_run_id,
            "mode": "baseline",
            "experiment_id": experiment_id,
            "paired_run_id": proposed_run_id,
            "dataset_name": dataset_name,
            "dataset_id": resolved_dataset_id,
            "created_at": created_at_iso,
            "updated_at": created_at_iso,
            "active_phase": 7,
            "completed_phases": [1, 2, 3, 7],
            "objective": obj,
            "profile": prof,
            "plan": plan_dict,
            "analysis": analysis_dict,
            "critic": None,
            "verification": None,
            "insights": None,
            "evaluation": baseline_eval.model_dump(),
        }

        # Build proposed RunPayload: full phases populated
        proposed_payload = {
            "run_id": proposed_run_id,
            "mode": "proposed",
            "experiment_id": experiment_id,
            "paired_run_id": baseline_run_id,
            "dataset_name": dataset_name,
            "dataset_id": resolved_dataset_id,
            "created_at": created_at_iso,
            "updated_at": created_at_iso,
            "active_phase": 7,
            "completed_phases": [1, 2, 3, 4, 5, 6, 7],
            "objective": obj,
            "profile": prof,
            "plan": plan_dict,
            "analysis": analysis_dict,
            "critic": critic_dict,
            "verification": verif_dict,
            "insights": insight_dict,
            "evaluation": proposed_eval.model_dump(),
        }

        try:
            run_store.save_run(baseline_payload)
            baseline_summary.persisted = True
            exp_logger.log_event(baseline_run_id, "RUN_SAVED", outcome="SUCCESS", experiment_id=experiment_id)
        except Exception as exc:
            sanitized_err = _sanitize_error_message(str(exc))
            logger.error(
                "Failed to save baseline run %s for experiment %s: %s",
                baseline_run_id,
                experiment_id,
                sanitized_err,
            )
            raise ExperimentPersistenceError(
                f"Failed to persist baseline branch '{baseline_run_id}': {sanitized_err}",
                partial_persistence=False,
                saved_run_id=None,
            ) from exc

        try:
            run_store.save_run(proposed_payload)
            proposed_summary.persisted = True
            persisted_successfully = True
            exp_logger.log_event(proposed_run_id, "RUN_SAVED", outcome="SUCCESS", experiment_id=experiment_id)
        except Exception as exc:
            sanitized_err = _sanitize_error_message(str(exc))
            logger.error(
                "Partial persistence failure: baseline run %s was saved, but saving proposed run %s failed: %s",
                baseline_run_id,
                proposed_run_id,
                sanitized_err,
            )
            raise ExperimentPersistenceError(
                f"Partial persistence failure: baseline branch '{baseline_run_id}' was persisted, "
                f"but saving proposed branch '{proposed_run_id}' failed: {sanitized_err}",
                partial_persistence=True,
                saved_run_id=baseline_run_id,
            ) from exc

    # Finalize monotonic experiment duration
    t_exp_end = time.perf_counter()
    total_exp_duration = max(0.0, round(t_exp_end - t_exp_start, 4))
    comparison_summary.total_experiment_duration_seconds = total_exp_duration

    resolved_execution_mode = "offline_fixture_replay" if is_offline_fixture else (execution_mode or "research_run")

    return PairedExperimentResponse(
        experiment_id=experiment_id,
        dataset_id=resolved_dataset_id,
        objective=obj,
        created_at=now_utc.isoformat(),
        shared_plan=shared_plan,
        baseline=baseline_summary,
        proposed=proposed_summary,
        comparison=comparison_summary,
        persisted=persisted_successfully,
        partial_persistence=False,
        execution_mode=resolved_execution_mode,
        is_offline_fixture=is_offline_fixture,
    )

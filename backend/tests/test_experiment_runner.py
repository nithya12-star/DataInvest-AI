"""
backend/tests/test_experiment_runner.py
=======================================
Offline unit tests for Paired Experimentation Engine.

CRITICAL REQUIREMENTS:
- 100% offline, zero network calls, zero real Gemini API keys (httpx.MockTransport).
- Exactly one Planner call per paired experiment.
- Research mode requires the genuine LLM Planner.
- Missing API key or Planner errors stop execution before statistical analysis.
- Both branches receive the exact same plan and raw analysis results.
- Baseline contains no fabricated Critic or verification records.
- Proposed branch invokes Critic and controlled re-investigation using existing services.
- Comparison deltas match actual service outputs (including non-improving proposed scores).
- persist_runs=False writes no records.
- Persistence uses distinct linked IDs and preserves pre-existing run files.
- Service failures are not represented as successful completed experiments.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, call

import httpx
import pytest

from app.schemas.analysis_schema import (
    AnalysisResult,
    PlanAnalysisRequest,
    PlanAnalysisResponse,
)
from app.schemas.critic_schema import CriticOverallSummary, CriticResponse, CriticResult
from app.schemas.evaluation_schema import (
    DimensionScore,
    EvaluationResponse,
    EvaluationSummary,
)
from app.schemas.experiment_schema import (
    PairedExperimentRequest,
    PairedExperimentResponse,
)
from app.schemas.insight_schema import InsightResponse, InsightSummary
from app.schemas.planner_schema import InvestigationTask, PlannerResponse
from app.schemas.verification_schema import (
    VerificationResponse,
    VerificationSummary,
    VerifiedFinding,
)
from app.services.experiment_runner import (
    ExperimentPersistenceError,
    ExperimentRunnerError,
    run_paired_experiment,
)
from app.services.gemini_client import (
    GeminiClient,
    GeminiMissingKeyError,
    GeminiNetworkError,
)
from app.services.run_store import RunStoreService


# ============================================================================
# Fixtures & Mock Builders
# ============================================================================

@pytest.fixture
def sample_profile() -> Dict[str, Any]:
    """Standard Phase 1 dataset profile."""
    return {
        "dataset_id": "test_salaries.csv",
        "overview": {
            "n_rows": 120,
            "n_cols": 4,
            "column_names": ["salary", "experience", "department", "rating"],
            "filename": "test_salaries.csv",
        },
        "column_types": {
            "salary": "numerical",
            "experience": "numerical",
            "rating": "numerical",
            "department": "categorical",
        },
        "missing": {"total_missing": 0, "total_missing_pct": 0.0},
        "cardinality": {"department": 3},
    }


@pytest.fixture
def sample_planner_response() -> PlannerResponse:
    """Pre-formed valid PlannerResponse model."""
    task = InvestigationTask(
        task_id="task_1",
        task_type="numerical_vs_numerical",
        task="Analyze salary vs experience",
        description="Assess correlation between experience and salary.",
        method="Pearson correlation",
        columns=["salary", "experience"],
        column_a="salary",
        column_b="experience",
        reason="Both continuous numerical features.",
        priority="high",
    )
    return PlannerResponse(
        objective="Investigate salary determinants.",
        interpreted_goal="Investigate drivers of salary.",
        target_column="salary",
        selected_columns=[
            {"column": "salary", "data_type": "numerical", "reason": "Target", "is_target": True},
            {"column": "experience", "data_type": "numerical", "reason": "Predictor", "is_target": False},
        ],
        analysis_tasks=[task],
        tasks=[task],
        reasoning=["Single high-priority correlation task."],
        warnings=[],
        expected_outputs=["Pearson r and p-value."],
    )


def build_mock_gemini_client(response_plan: Dict[str, Any], status_code: int = 200) -> GeminiClient:
    """Offline GeminiClient backed by httpx.MockTransport."""
    gemini_body = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": json.dumps(response_plan)}],
                },
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 100,
            "candidatesTokenCount": 100,
            "totalTokenCount": 200,
        },
        "modelVersion": "gemini-3.5-flash-lite",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=status_code,
            headers={"Content-Type": "application/json"},
            content=json.dumps(gemini_body).encode("utf-8"),
        )

    transport = httpx.MockTransport(handler)
    client = GeminiClient(api_key="mock_test_key_ai_test", transport=transport)
    return client


def build_sample_analysis_response() -> PlanAnalysisResponse:
    """Sample Phase 3 Analysis Response."""
    return PlanAnalysisResponse(
        dataset_id="test_salaries.csv",
        total_tasks=1,
        successful_tasks=1,
        failed_tasks=0,
        results=[
            AnalysisResult(
                task_id="task_1",
                status="success",
                analysis_type="numerical_vs_numerical",
                method="Pearson correlation",
                columns=["salary", "experience"],
                sample_size=120,
                results={"statistic": 0.72, "p_value": 0.0001, "correlation": 0.72},
                warnings=[],
            )
        ],
    )


def build_sample_critic_response(requires_reinvestigation: bool = False) -> CriticResponse:
    """Sample Phase 4 Critic Response."""
    status = "REQUIRES_REINVESTIGATION" if requires_reinvestigation else "ACCEPTED"
    return CriticResponse(
        dataset_id="test_salaries.csv",
        total_evaluated=1,
        reviews=[
            CriticResult(
                task_id="task_1",
                analysis_type="numerical_vs_numerical",
                method="Pearson correlation",
                variables=["salary", "experience"],
                status=status,
                confidence=0.95,
                confidence_level="HIGH",
                summary="Assessed linearity and normality.",
                evidence=["p-value=0.0001"],
                statistical_validity="Valid test statistic.",
                method_assessment="Appropriate correlation method.",
                data_quality_concerns=[],
                assumption_concerns=["Mild non-normality"] if requires_reinvestigation else [],
                recommendation="Reinvestigate with Spearman" if requires_reinvestigation else "Accept finding.",
                requires_reinvestigation=requires_reinvestigation,
                recommended_method="Spearman rank correlation" if requires_reinvestigation else None,
            )
        ],
        summary=CriticOverallSummary(
            total_results=1,
            accepted=0 if requires_reinvestigation else 1,
            questionable=0,
            rejected=0,
            requires_reinvestigation=1 if requires_reinvestigation else 0,
            overall_status="ROBUST" if not requires_reinvestigation else "METHODOLOGICAL_CONCERNS",
            critical_issues=[],
            key_verified_insights=["Salary correlates with experience."],
        ),
    )


def build_sample_verification_response(reinvestigated: bool = False) -> VerificationResponse:
    """Sample Phase 5 Verification Response."""
    status = "VERIFIED_AFTER_REINVESTIGATION" if reinvestigated else "VERIFIED"
    return VerificationResponse(
        dataset_id="test_salaries.csv",
        total_verified=1,
        all_findings=[
            VerifiedFinding(
                finding_id="finding_1",
                task_id="task_1",
                variables=["salary", "experience"],
                method="Spearman rank correlation" if reinvestigated else "Pearson correlation",
                status=status,
                confidence=0.98,
                confidence_level="HIGH",
                summary="Correlation verified across methods.",
                statistical_evidence={"statistic": 0.70, "p_value": 0.0001},
                critic_assessment="Robust relationship confirmed.",
                reinvestigated=reinvestigated,
                verification_reason="Consistent effect size and significance.",
                limitations=[],
                reason_codes=["CONFIRMED"],
            )
        ],
        verified_findings=[
            VerifiedFinding(
                finding_id="finding_1",
                task_id="task_1",
                variables=["salary", "experience"],
                method="Spearman rank correlation" if reinvestigated else "Pearson correlation",
                status=status,
                confidence=0.98,
                confidence_level="HIGH",
                summary="Correlation verified across methods.",
                statistical_evidence={"statistic": 0.70, "p_value": 0.0001},
                critic_assessment="Robust relationship confirmed.",
                reinvestigated=reinvestigated,
                verification_reason="Consistent effect size and significance.",
                limitations=[],
                reason_codes=["CONFIRMED"],
            )
        ],
        questionable_findings=[],
        rejected_findings=[],
        reinvestigations=[],
        summary=VerificationSummary(
            total_findings=1,
            verified=0 if reinvestigated else 1,
            verified_after_reinvestigation=1 if reinvestigated else 0,
            questionable=0,
            rejected=0,
            total_reinvestigated=1 if reinvestigated else 0,
            overall_integrity="HIGH",
            unresolved_issues=[],
        ),
    )


def build_sample_insight_response() -> InsightResponse:
    """Sample Phase 6 Insight Response."""
    return InsightResponse(
        dataset_id="test_salaries.csv",
        summary=InsightSummary(
            total_verified_findings=1,
            total_insights=1,
            strong_insights=1,
            moderate_insights=0,
            weak_insights=0,
            not_assessed_insights=0,
            unresolved_count=0,
            top_insight_title="Experience Correlates with Salary",
        ),
        key_insights=[],
        unresolved_findings=[],
    )


def build_mock_evaluation_response(score: float, grade: str) -> EvaluationResponse:
    """Build mocked Phase 7 EvaluationResponse with arbitrary scores."""
    return EvaluationResponse(
        summary=EvaluationSummary(
            overall_score=score,
            overall_grade=grade,
            total_tasks_planned=1,
            total_tasks_executed=1,
            total_findings_verified=1 if score > 70 else 0,
            total_insights_generated=1 if score > 70 else 0,
            zero_hallucination_verified=True,
            executive_assessment="Mocked evaluation result.",
        ),
        dimensions=[
            DimensionScore(
                dimension_key="execution_reliability",
                name="Execution Reliability",
                score=min(25.0, score * 0.25),
                max_score=25.0,
                percentage=100.0,
                grade="A",
                summary="Executed cleanly.",
            ),
            DimensionScore(
                dimension_key="statistical_rigor",
                name="Statistical Rigor",
                score=min(25.0, score * 0.25),
                max_score=25.0,
                percentage=100.0,
                grade="A",
                summary="Evaluated assumptions.",
            ),
            DimensionScore(
                dimension_key="verification_grounding",
                name="Verification Truth Grounding",
                score=min(25.0, score * 0.25),
                max_score=25.0,
                percentage=100.0,
                grade="A",
                summary="Grounded evidence.",
            ),
            DimensionScore(
                dimension_key="insight_actionability",
                name="Insight Actionability",
                score=min(25.0, score * 0.25),
                max_score=25.0,
                percentage=100.0,
                grade="A",
                summary="Actionable insights.",
            ),
        ],
        benchmark_checks=[],
        detailed_metrics=[],
    )


@pytest.fixture
def mock_services():
    """Bundle of mock services for controlled experiment testing."""
    mock_analysis = MagicMock()
    mock_analysis.execute_plan.return_value = build_sample_analysis_response()

    mock_critic = MagicMock()
    mock_critic.run_review.return_value = build_sample_critic_response(requires_reinvestigation=True)

    mock_verif = MagicMock()
    mock_verif.process_verification.return_value = build_sample_verification_response(reinvestigated=True)

    mock_insight = MagicMock()
    mock_insight.generate_insights.return_value = build_sample_insight_response()

    mock_eval = MagicMock()
    # Baseline eval = 65.0 pts, Proposed eval = 88.0 pts
    mock_eval.evaluate_pipeline.side_effect = [
        build_mock_evaluation_response(65.0, "B - Acceptable"),
        build_mock_evaluation_response(88.0, "A - High Scientific Rigor"),
    ]

    return {
        "analysis": mock_analysis,
        "critic": mock_critic,
        "verification": mock_verif,
        "insight": mock_insight,
        "evaluation": mock_eval,
    }


# ============================================================================
# Focused Unit Tests
# ============================================================================

def test_single_planner_call_per_paired_experiment(sample_profile, sample_planner_response, mock_services):
    """Requirement 1: Exactly ONE Planner call per paired experiment; both branches share it."""
    mock_planner = MagicMock(return_value=sample_planner_response)

    response = run_paired_experiment(
        profile=sample_profile,
        planner_fn=mock_planner,
        analysis_service=mock_services["analysis"],
        critic_service=mock_services["critic"],
        verification_service=mock_services["verification"],
        insight_service=mock_services["insight"],
        evaluation_service_inst=mock_services["evaluation"],
        persist_runs=False,
    )

    # Assert Planner was invoked exactly once
    assert mock_planner.call_count == 1
    # Verify both branches received the exact same plan
    assert response.shared_plan == sample_planner_response
    assert response.baseline.total_tasks_executed == 1
    assert response.proposed.total_tasks_executed == 1


def test_research_mode_requires_genuine_llm_planner(sample_profile, sample_planner_response, mock_services):
    """Requirement 2: Research mode requires the genuine LLM Planner (is_research_run=True)."""
    mock_planner = MagicMock(return_value=sample_planner_response)

    run_paired_experiment(
        profile=sample_profile,
        planner_fn=mock_planner,
        analysis_service=mock_services["analysis"],
        critic_service=mock_services["critic"],
        verification_service=mock_services["verification"],
        insight_service=mock_services["insight"],
        evaluation_service_inst=mock_services["evaluation"],
        persist_runs=False,
    )

    call_kwargs = mock_planner.call_args.kwargs
    assert call_kwargs.get("is_research_run") is True


def test_missing_api_key_or_planner_errors_stop_before_analysis(sample_profile, mock_services):
    """Requirement 3: Missing API key or Planner errors stop execution immediately before statistical analysis."""
    # Planner raises GeminiMissingKeyError
    def failing_planner(**kwargs):
        raise GeminiMissingKeyError("GEMINI_API_KEY environment variable is not set.")

    with pytest.raises(GeminiMissingKeyError) as exc_info:
        run_paired_experiment(
            profile=sample_profile,
            planner_fn=failing_planner,
            analysis_service=mock_services["analysis"],
            critic_service=mock_services["critic"],
            verification_service=mock_services["verification"],
            insight_service=mock_services["insight"],
            evaluation_service_inst=mock_services["evaluation"],
            persist_runs=False,
        )

    assert "GEMINI_API_KEY" in str(exc_info.value)
    # Analysis service must NEVER have been called
    mock_services["analysis"].execute_plan.assert_not_called()
    mock_services["critic"].run_review.assert_not_called()


def test_both_branches_receive_same_plan_and_analysis_results(sample_profile, sample_planner_response, mock_services):
    """Requirement 4: Both branches receive the exact same plan and raw analysis results."""
    mock_planner = MagicMock(return_value=sample_planner_response)

    run_paired_experiment(
        profile=sample_profile,
        planner_fn=mock_planner,
        analysis_service=mock_services["analysis"],
        critic_service=mock_services["critic"],
        verification_service=mock_services["verification"],
        insight_service=mock_services["insight"],
        evaluation_service_inst=mock_services["evaluation"],
        persist_runs=False,
    )

    # Analysis executed once
    assert mock_services["analysis"].execute_plan.call_count == 1

    # Evaluation service called twice: call 0 for baseline, call 1 for proposed
    assert mock_services["evaluation"].evaluate_pipeline.call_count == 2
    baseline_eval_call = mock_services["evaluation"].evaluate_pipeline.call_args_list[0].kwargs
    proposed_eval_call = mock_services["evaluation"].evaluate_pipeline.call_args_list[1].kwargs

    # Both calls must share the identical plan and analysis dicts
    assert baseline_eval_call["plan"] == sample_planner_response.model_dump()
    assert proposed_eval_call["plan"] == sample_planner_response.model_dump()
    assert baseline_eval_call["analysis"] == proposed_eval_call["analysis"]


def test_baseline_contains_no_fabricated_critic_or_verification(sample_profile, sample_planner_response, mock_services):
    """Requirement 5: Baseline branch skips Critic and verification; critic=None, verification=None, insights=None."""
    mock_planner = MagicMock(return_value=sample_planner_response)

    response = run_paired_experiment(
        profile=sample_profile,
        planner_fn=mock_planner,
        analysis_service=mock_services["analysis"],
        critic_service=mock_services["critic"],
        verification_service=mock_services["verification"],
        insight_service=mock_services["insight"],
        evaluation_service_inst=mock_services["evaluation"],
        persist_runs=False,
    )

    # Baseline evaluation received None for critic, verification, insights
    baseline_eval_call = mock_services["evaluation"].evaluate_pipeline.call_args_list[0].kwargs
    assert baseline_eval_call["critic"] is None
    assert baseline_eval_call["verification"] is None
    assert baseline_eval_call["insights"] is None

    # Baseline summary flags are all False
    assert response.baseline.has_critic_review is False
    assert response.baseline.has_verification is False
    assert response.baseline.has_insights is False
    assert response.baseline.completed_phases == [1, 2, 3, 7]


def test_proposed_branch_invokes_critic_and_verification(sample_profile, sample_planner_response, mock_services):
    """Requirement 6: Proposed branch invokes Critic, controlled re-investigation, and insights."""
    mock_planner = MagicMock(return_value=sample_planner_response)

    response = run_paired_experiment(
        profile=sample_profile,
        planner_fn=mock_planner,
        max_reinvestigations=2,
        analysis_service=mock_services["analysis"],
        critic_service=mock_services["critic"],
        verification_service=mock_services["verification"],
        insight_service=mock_services["insight"],
        evaluation_service_inst=mock_services["evaluation"],
        persist_runs=False,
    )

    # Verify Critic and Verification were invoked
    assert mock_services["critic"].run_review.call_count == 1
    assert mock_services["verification"].process_verification.call_count == 1
    assert mock_services["insight"].generate_insights.call_count == 1

    # Verify max_reinvestigations passed to verification request
    verif_call_req = mock_services["verification"].process_verification.call_args[0][0]
    assert verif_call_req.max_reinvestigations == 2

    # Proposed summary flags
    assert response.proposed.has_critic_review is True
    assert response.proposed.has_verification is True
    assert response.proposed.has_insights is True
    assert response.proposed.completed_phases == [1, 2, 3, 4, 5, 6, 7]


def test_comparison_deltas_match_mocked_outputs_and_handles_non_improving_score(sample_profile, sample_planner_response, mock_services):
    """Requirement 7: Comparison deltas match actual service outputs, including negative or zero deltas."""
    mock_planner = MagicMock(return_value=sample_planner_response)

    # Case A: Proposed improves (65 -> 88 = +23.0)
    resp_positive = run_paired_experiment(
        profile=sample_profile,
        planner_fn=mock_planner,
        analysis_service=mock_services["analysis"],
        critic_service=mock_services["critic"],
        verification_service=mock_services["verification"],
        insight_service=mock_services["insight"],
        evaluation_service_inst=mock_services["evaluation"],
        persist_runs=False,
    )
    assert resp_positive.comparison.overall_score_delta == 23.0
    assert "improved" in resp_positive.comparison.scientific_verdict

    # Case B: Proposed does NOT improve (Baseline 80, Proposed 70 = -10.0)
    mock_eval_lower = MagicMock()
    mock_eval_lower.evaluate_pipeline.side_effect = [
        build_mock_evaluation_response(80.0, "A - High Scientific Rigor"),
        build_mock_evaluation_response(70.0, "B - Acceptable"),
    ]
    resp_negative = run_paired_experiment(
        profile=sample_profile,
        planner_fn=mock_planner,
        analysis_service=mock_services["analysis"],
        critic_service=mock_services["critic"],
        verification_service=mock_services["verification"],
        insight_service=mock_services["insight"],
        evaluation_service_inst=mock_eval_lower,
        persist_runs=False,
    )
    assert resp_negative.comparison.overall_score_delta == -10.0
    assert "lower" in resp_negative.comparison.scientific_verdict

    # Case C: Equal scores (Baseline 75, Proposed 75 = 0.0)
    mock_eval_equal = MagicMock()
    mock_eval_equal.evaluate_pipeline.side_effect = [
        build_mock_evaluation_response(75.0, "B+ - Satisfactory"),
        build_mock_evaluation_response(75.0, "B+ - Satisfactory"),
    ]
    resp_equal = run_paired_experiment(
        profile=sample_profile,
        planner_fn=mock_planner,
        analysis_service=mock_services["analysis"],
        critic_service=mock_services["critic"],
        verification_service=mock_services["verification"],
        insight_service=mock_services["insight"],
        evaluation_service_inst=mock_eval_equal,
        persist_runs=False,
    )
    assert resp_equal.comparison.overall_score_delta == 0.0
    assert "equal" in resp_equal.comparison.scientific_verdict


def test_persist_runs_false_writes_no_records(sample_profile, sample_planner_response, mock_services):
    """Requirement 8: persist_runs=False writes no records to disk."""
    mock_planner = MagicMock(return_value=sample_planner_response)
    mock_run_store = MagicMock()

    response = run_paired_experiment(
        profile=sample_profile,
        planner_fn=mock_planner,
        persist_runs=False,
        analysis_service=mock_services["analysis"],
        critic_service=mock_services["critic"],
        verification_service=mock_services["verification"],
        insight_service=mock_services["insight"],
        evaluation_service_inst=mock_services["evaluation"],
        run_store_service=mock_run_store,
    )

    mock_run_store.save_run.assert_not_called()
    assert response.persisted is False
    assert response.baseline.persisted is False
    assert response.proposed.persisted is False


def test_persistence_uses_distinct_linked_ids_and_preserves_existing_runs(sample_profile, sample_planner_response, mock_services):
    """Requirement 9: Persistence uses distinct linked IDs and preserves pre-existing run files."""
    mock_planner = MagicMock(return_value=sample_planner_response)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        runs_dir = Path(tmpdir)
        store = RunStoreService(runs_dir=runs_dir)

        # Pre-populate an existing historical run file
        existing_run_file = runs_dir / "run_20260101_000000_historical_abc123.json"
        existing_content = {"run_id": "run_20260101_000000_historical_abc123", "data": "preserved"}
        with open(existing_run_file, "w", encoding="utf-8") as f:
            json.dump(existing_content, f)

        response = run_paired_experiment(
            profile=sample_profile,
            planner_fn=mock_planner,
            persist_runs=True,
            analysis_service=mock_services["analysis"],
            critic_service=mock_services["critic"],
            verification_service=mock_services["verification"],
            insight_service=mock_services["insight"],
            evaluation_service_inst=mock_services["evaluation"],
            run_store_service=store,
        )

        assert response.persisted is True
        assert response.baseline.persisted is True
        assert response.proposed.persisted is True

        # Check distinct run IDs
        assert response.baseline.run_id != response.proposed.run_id
        assert "baseline" in response.baseline.run_id
        assert "proposed" in response.proposed.run_id

        # Check linked IDs
        assert response.baseline.paired_run_id == response.proposed.run_id
        assert response.proposed.paired_run_id == response.baseline.run_id

        # Verify saved files on disk
        baseline_file = runs_dir / f"{response.baseline.run_id}.json"
        proposed_file = runs_dir / f"{response.proposed.run_id}.json"
        assert baseline_file.exists()
        assert proposed_file.exists()

        with open(baseline_file, "r", encoding="utf-8") as f:
            baseline_data = json.load(f)
            assert baseline_data["mode"] == "baseline"
            assert baseline_data["critic"] is None
            assert baseline_data["verification"] is None
            assert baseline_data["insights"] is None

        with open(proposed_file, "r", encoding="utf-8") as f:
            proposed_data = json.load(f)
            assert proposed_data["mode"] == "proposed"
            assert proposed_data["critic"] is not None
            assert proposed_data["verification"] is not None
            assert proposed_data["insights"] is not None

        # Verify pre-existing historical file remains completely untouched
        assert existing_run_file.exists()
        with open(existing_run_file, "r", encoding="utf-8") as f:
            loaded_hist = json.load(f)
            assert loaded_hist == existing_content


def test_service_failures_not_represented_as_successful_experiments(sample_profile, sample_planner_response, mock_services):
    """Requirement 10: Service failures are not represented as successful completed experiments."""
    mock_planner = MagicMock(return_value=sample_planner_response)

    # 10a. Analysis failure propagates sanitized error
    mock_failing_analysis = MagicMock()
    mock_failing_analysis.execute_plan.side_effect = RuntimeError("Dataset resolution failure.")

    with pytest.raises(ExperimentRunnerError) as exc_info:
        run_paired_experiment(
            profile=sample_profile,
            planner_fn=mock_planner,
            analysis_service=mock_failing_analysis,
            critic_service=mock_services["critic"],
            verification_service=mock_services["verification"],
            insight_service=mock_services["insight"],
            evaluation_service_inst=mock_services["evaluation"],
            persist_runs=False,
        )
    assert "Statistical analysis stage failed" in str(exc_info.value)

    # 10b. Partial persistence failure explicitly reported
    mock_partial_store = MagicMock()
    # Baseline save succeeds, Proposed save fails
    mock_partial_store.save_run.side_effect = [{"status": "saved"}, IOError("Disk quota exceeded")]

    with pytest.raises(ExperimentPersistenceError) as exc_info:
        run_paired_experiment(
            profile=sample_profile,
            planner_fn=mock_planner,
            persist_runs=True,
            analysis_service=mock_services["analysis"],
            critic_service=mock_services["critic"],
            verification_service=mock_services["verification"],
            insight_service=mock_services["insight"],
            evaluation_service_inst=mock_services["evaluation"],
            run_store_service=mock_partial_store,
        )

    err = exc_info.value
    assert err.partial_persistence is True
    assert err.saved_run_id is not None
    assert "Partial persistence failure" in str(err)


def test_paired_experiment_request_schema_validation(sample_profile):
    """Verify input validation for PairedExperimentRequest."""
    # Valid request
    req = PairedExperimentRequest(
        profile=sample_profile,
        dataset_id="test.csv",
        objective="Analyze factors.",
        max_reinvestigations=2,
        persist_runs=True,
    )
    assert req.max_reinvestigations == 2

    # Invalid max_reinvestigations (> 3)
    with pytest.raises(ValueError):
        PairedExperimentRequest(
            profile=sample_profile,
            max_reinvestigations=5,
        )

    # Invalid profile (0 rows)
    with pytest.raises(ValueError):
        PairedExperimentRequest(
            profile={"overview": {"n_rows": 0, "n_cols": 0}},
        )


def test_end_to_end_with_mock_transport_gemini_client(sample_profile, mock_services):
    """
    Verify full end-to-end execution through create_analysis_plan using
    a MockTransport-backed GeminiClient (100% offline).
    """
    plan_payload = {
        "objective": "Investigate salary determinants.",
        "interpreted_goal": "Investigate drivers of salary.",
        "target_column": "salary",
        "selected_columns": [
            {"column": "salary", "data_type": "numerical", "reason": "Target outcome", "is_target": True},
            {"column": "experience", "data_type": "numerical", "reason": "Potential driver", "is_target": False},
        ],
        "analysis_tasks": [
            {
                "task_id": "task_1",
                "task_type": "numerical_vs_numerical",
                "task": "Analyze salary vs experience",
                "description": "Assess correlation between experience and salary.",
                "method": "Pearson correlation",
                "columns": ["salary", "experience"],
                "column_a": "salary",
                "column_b": "experience",
                "reason": "Both continuous numerical features.",
                "priority": "high",
            }
        ],
        "reasoning": ["Investigating relationship between experience and salary."],
        "warnings": [],
        "expected_outputs": ["Pearson correlation results."],
    }
    client = build_mock_gemini_client(plan_payload)

    # Note: planner_fn is NOT provided, so real create_analysis_plan is invoked
    response = run_paired_experiment(
        profile=sample_profile,
        gemini_client=client,
        persist_runs=False,
        analysis_service=mock_services["analysis"],
        critic_service=mock_services["critic"],
        verification_service=mock_services["verification"],
        insight_service=mock_services["insight"],
        evaluation_service_inst=mock_services["evaluation"],
    )

    assert isinstance(response, PairedExperimentResponse)
    assert response.shared_plan.planner_mode == "llm"
    assert response.shared_plan.analysis_tasks[0].task_id == "task_1"
    assert response.baseline.total_tasks_executed == 1
    assert response.proposed.total_tasks_executed == 1
    assert response.comparison.overall_score_delta == 23.0


"""
backend/tests/test_planner_llm.py
==================================
Offline unit and integration tests for Gemini LLM Planner integration.

CRITICAL CONSTRAINTS:
- No real API keys required.
- Zero network requests / no external network sockets used (httpx.MockTransport).
- Comprehensive coverage of:
  - Valid plan generation & Pydantic schema validation
  - Hallucinated column detection & rejection
  - Invalid task_type rejection
  - Anti-fabrication enforcement (no invented statistics)
  - Explicit error propagation: missing keys, auth failures, rate limits, network errors
  - Strict absence of silent fallback to deterministic planning in LLM mode
"""

import json
import httpx
import pytest

from app.agents.planner_agent import PlannerAgent
from app.schemas.planner_schema import PlannerResponse, InvestigationTask
from app.services.gemini_client import (
    GeminiAuthenticationError,
    GeminiClient,
    GeminiMalformedResponseError,
    GeminiMissingKeyError,
    GeminiNetworkError,
    GeminiRateLimitError,
)
from app.services.planner_service import create_analysis_plan


# ============================================================================
# Helpers & Fixtures
# ============================================================================

@pytest.fixture
def sample_profile() -> dict:
    """Standard dataset profile with 4 columns."""
    return {
        "dataset_id": "test_emp.csv",
        "overview": {
            "n_rows": 100,
            "n_cols": 4,
            "column_names": ["salary", "experience", "department", "age"],
            "filename": "test_emp.csv",
        },
        "column_types": {
            "salary": "numerical",
            "experience": "numerical",
            "age": "numerical",
            "department": "categorical",
        },
        "missing": {
            "total_missing": 0,
            "total_missing_pct": 0.0,
        },
        "cardinality": {
            "constant_columns": [],
            "high_cardinality_columns": [],
        },
        "numerical_stats": {},
        "outliers": {},
    }


def make_valid_gemini_plan_payload(objective: str = "Analyze salary drivers") -> dict:
    """Return valid plan JSON conforming to PlannerResponse schema."""
    return {
        "objective": objective,
        "interpreted_goal": "Investigate factors influencing employee salary.",
        "target_column": "salary",
        "selected_columns": [
            {"column": "salary", "data_type": "numerical", "reason": "Target outcome", "is_target": True},
            {"column": "experience", "data_type": "numerical", "reason": "Potential numerical driver", "is_target": False},
            {"column": "department", "data_type": "categorical", "reason": "Grouping variable", "is_target": False},
        ],
        "analysis_tasks": [
            {
                "task_id": "task_1",
                "task_type": "numerical_vs_numerical",
                "task": "Analyze salary vs experience",
                "description": "Assess linear and monotonic relationship between experience and salary.",
                "method": "Pearson correlation",
                "columns": ["salary", "experience"],
                "column_a": "salary",
                "column_b": "experience",
                "reason": "Both variables are continuous numerical features.",
                "priority": "high",
                "notes": ["Check for non-linear patterns."],
            },
            {
                "task_id": "task_2",
                "task_type": "multiple_groups_vs_numerical",
                "task": "Compare salary across departments",
                "description": "Evaluate if average salary varies significantly across departments.",
                "method": "One-way ANOVA",
                "columns": ["department", "salary"],
                "group_column": "department",
                "value_column": "salary",
                "reason": "Department is a multi-class categorical grouping.",
                "priority": "medium",
                "notes": ["Assess variance homogeneity."],
            },
        ],
        "reasoning": [
            "Dataset has 100 rows and 4 columns.",
            "Salary identified as primary target from user objective.",
            "Selected experience for correlation and department for ANOVA.",
        ],
        "warnings": [],
        "expected_outputs": [
            "Correlation coefficient and p-value from Phase 3 Analysis.",
            "ANOVA F-statistic and p-value from Phase 3 Analysis.",
        ],
    }


def build_mock_gemini_client(response_payload: dict, status_code: int = 200, headers: dict = None) -> GeminiClient:
    """Construct an offline GeminiClient backed by httpx.MockTransport."""
    gemini_body = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": json.dumps(response_payload)}],
                },
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 200,
            "candidatesTokenCount": 150,
            "totalTokenCount": 350,
        },
        "modelVersion": "gemini-3.5-flash-lite",
    }

    resp_headers = {"Content-Type": "application/json"}
    if headers:
        resp_headers.update(headers)

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=status_code,
            headers=resp_headers,
            content=json.dumps(gemini_body if status_code == 200 else response_payload).encode("utf-8"),
        )

    transport = httpx.MockTransport(mock_handler)
    return GeminiClient(api_key="mock_secret_key", transport=transport)


# ============================================================================
# 1. Successful Plan Generation & Schema Validation
# ============================================================================

def test_llm_planner_generates_valid_plan(sample_profile):
    """PlannerAgent in LLM mode parses valid Gemini response into PlannerResponse."""
    payload = make_valid_gemini_plan_payload()
    client = build_mock_gemini_client(payload)

    agent = PlannerAgent(use_llm=True, gemini_client=client)
    plan = agent.plan(sample_profile, "Analyze salary drivers")

    assert isinstance(plan, PlannerResponse)
    assert plan.planner_mode == "llm"
    assert plan.target_column == "salary"
    assert len(plan.analysis_tasks) == 2
    assert plan.tasks == plan.analysis_tasks
    assert plan.analysis_tasks[0].task_type == "numerical_vs_numerical"
    assert plan.analysis_tasks[1].task_type == "multiple_groups_vs_numerical"
    assert plan.dataset_summary["planner_mode"] == "llm"
    assert plan.dataset_summary["n_rows"] == 100


def test_planner_service_enforces_llm_by_default_and_isolates_explicit_offline(sample_profile, monkeypatch):
    """
    Normal intended Planner workflow requires Gemini LLM by default,
    never silently falling back to deterministic planning.
    Explicit offline mode is strictly isolated via use_llm=False.
    """
    payload = make_valid_gemini_plan_payload()
    mock_client = build_mock_gemini_client(payload)

    # 1. Normal intended application workflow defaults to LLM planning (no use_llm passed)
    plan_default = create_analysis_plan(
        sample_profile,
        "Analyze salary",
        gemini_client=mock_client,
    )
    assert plan_default.planner_mode == "llm"
    assert plan_default.planner_provider == "google"

    # 2. When credentials are missing and use_llm is omitted, must fail explicitly (no silent fallback)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(GeminiMissingKeyError):
        create_analysis_plan(sample_profile, "Analyze salary")

    # 3. Explicit offline mode (use_llm=False) produces deterministic planning
    plan_offline = create_analysis_plan(
        sample_profile,
        "Analyze salary",
        use_llm=False,
    )
    assert plan_offline.planner_mode == "deterministic"
    assert plan_offline.planner_provider == "deterministic"

    # 4. Explicit use_llm=True with injected mock client
    plan_llm = create_analysis_plan(
        sample_profile,
        "Analyze salary",
        use_llm=True,
        gemini_client=mock_client,
    )
    assert plan_llm.planner_mode == "llm"

    # 5. Via preferences dict
    plan_pref = create_analysis_plan(
        sample_profile,
        "Analyze salary",
        preferences={"use_llm": True},
        gemini_client=mock_client,
    )
    assert plan_pref.planner_mode == "llm"



# ============================================================================
# 2. Strict Semantic & Anti-Hallucination Rejection Tests
# ============================================================================

def test_llm_planner_rejects_hallucinated_column_in_task(sample_profile):
    """Plan must be rejected if a task references a column not in the profile."""
    payload = make_valid_gemini_plan_payload()
    # Inject hallucinated column
    payload["analysis_tasks"][0]["columns"] = ["salary", "non_existent_column_xyz"]

    client = build_mock_gemini_client(payload)
    agent = PlannerAgent(use_llm=True, gemini_client=client)

    with pytest.raises(GeminiMalformedResponseError) as exc_info:
        agent.plan(sample_profile, "Analyze salary")

    assert "hallucinated column 'non_existent_column_xyz'" in str(exc_info.value)


def test_llm_planner_rejects_hallucinated_target_column(sample_profile):
    """Plan must be rejected if identified target column is not in the dataset."""
    payload = make_valid_gemini_plan_payload()
    payload["target_column"] = "phantom_target"

    client = build_mock_gemini_client(payload)
    agent = PlannerAgent(use_llm=True, gemini_client=client)

    with pytest.raises(GeminiMalformedResponseError) as exc_info:
        agent.plan(sample_profile, "Analyze salary")

    assert "target_column 'phantom_target' does not exist" in str(exc_info.value)


def test_llm_planner_rejects_invalid_task_type(sample_profile):
    """Plan must be rejected if an unknown task_type is provided."""
    payload = make_valid_gemini_plan_payload()
    payload["analysis_tasks"][0]["task_type"] = "quantum_neural_regression"

    client = build_mock_gemini_client(payload)
    agent = PlannerAgent(use_llm=True, gemini_client=client)

    with pytest.raises(GeminiMalformedResponseError) as exc_info:
        agent.plan(sample_profile, "Analyze salary")

    assert "invalid task_type 'quantum_neural_regression'" in str(exc_info.value)


def test_llm_planner_rejects_fabricated_statistics(sample_profile):
    """Plan must be rejected if the LLM claims calculated statistics (e.g. p < 0.05)."""
    payload = make_valid_gemini_plan_payload()
    payload["analysis_tasks"][0]["reason"] = "We found strong correlation with p = 0.001."

    client = build_mock_gemini_client(payload)
    agent = PlannerAgent(use_llm=True, gemini_client=client)

    with pytest.raises(GeminiMalformedResponseError) as exc_info:
        agent.plan(sample_profile, "Analyze salary")

    assert "violates scientific boundaries" in str(exc_info.value)


# ============================================================================
# 3. Explicit Failure Propagation (NO Silent Fallback)
# ============================================================================

def test_missing_api_key_raises_explicit_error_without_fallback(sample_profile, monkeypatch):
    """When LLM mode is active and API key is missing, raise GeminiMissingKeyError explicitly."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    # Agent instantiated with use_llm=True and default client resolution
    agent = PlannerAgent(use_llm=True, gemini_client=None)

    with pytest.raises(GeminiMissingKeyError) as exc_info:
        agent.plan(sample_profile, "Analyze salary")

    assert "GEMINI_API_KEY" in str(exc_info.value)


def test_auth_error_propagates_without_fallback(sample_profile):
    """HTTP 401 must raise GeminiAuthenticationError explicitly, never silent fallback."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=401,
            json={"error": {"code": 401, "message": "API key not valid."}},
        )

    transport = httpx.MockTransport(mock_handler)
    client = GeminiClient(api_key="bad_key", transport=transport)
    agent = PlannerAgent(use_llm=True, gemini_client=client)

    with pytest.raises(GeminiAuthenticationError) as exc_info:
        agent.plan(sample_profile, "Analyze salary")

    assert exc_info.value.status_code == 401
    assert "API key not valid" in str(exc_info.value)


def test_rate_limit_propagates_without_fallback(sample_profile):
    """HTTP 429 must raise GeminiRateLimitError explicitly, never silent fallback."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=429,
            headers={"Retry-After": "30"},
            json={"error": {"code": 429, "message": "ResourceExhausted"}},
        )

    transport = httpx.MockTransport(mock_handler)
    client = GeminiClient(api_key="test_key", transport=transport)
    agent = PlannerAgent(use_llm=True, gemini_client=client)

    with pytest.raises(GeminiRateLimitError) as exc_info:
        agent.plan(sample_profile, "Analyze salary")

    assert exc_info.value.status_code == 429
    assert exc_info.value.retry_after == 30.0


def test_network_timeout_propagates_without_fallback(sample_profile):
    """Network timeout must raise GeminiNetworkError explicitly, never silent fallback."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Connection timed out to Gemini API.")

    transport = httpx.MockTransport(mock_handler)
    client = GeminiClient(api_key="test_key", transport=transport)
    agent = PlannerAgent(use_llm=True, gemini_client=client)

    with pytest.raises(GeminiNetworkError) as exc_info:
        agent.plan(sample_profile, "Analyze salary")

    assert "Connection timed out" in str(exc_info.value)


def test_malformed_json_propagates_without_fallback(sample_profile):
    """Malformed non-JSON LLM response must raise GeminiMalformedResponseError, never silent fallback."""
    gemini_body = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": "I am unable to generate a plan."}],
                },
                "finishReason": "STOP",
            }
        ]
    }
    transport = httpx.MockTransport(
        lambda req: httpx.Response(status_code=200, json=gemini_body)
    )
    client = GeminiClient(api_key="test_key", transport=transport)
    agent = PlannerAgent(use_llm=True, gemini_client=client)

    with pytest.raises(GeminiMalformedResponseError):
        agent.plan(sample_profile, "Analyze salary")


# ============================================================================
# 4. Mandatory Genuine LLM in Research Runs (Step 4 Enforcement)
# ============================================================================

def test_research_run_forces_llm_even_if_use_llm_false_supplied(sample_profile):
    """
    When is_research_run=True (or preferences['research_run']=True),
    use_llm=True is enforced unconditionally even if use_llm=False was supplied.
    """
    payload = make_valid_gemini_plan_payload()
    mock_client = build_mock_gemini_client(payload)

    # 1. Via explicit is_research_run parameter with conflicting use_llm=False
    plan1 = create_analysis_plan(
        profile=sample_profile,
        objective="Analyze salary",
        use_llm=False,
        is_research_run=True,
        gemini_client=mock_client,
    )
    assert plan1.planner_mode == "llm"
    assert len(plan1.analysis_tasks) == 2

    # 2. Via preferences dict with conflicting use_llm=False
    plan2 = create_analysis_plan(
        profile=sample_profile,
        objective="Analyze salary",
        preferences={"research_run": True, "use_llm": False},
        gemini_client=mock_client,
    )
    assert plan2.planner_mode == "llm"


def test_research_run_missing_api_key_fails_without_deterministic_fallback(sample_profile, monkeypatch):
    """
    If the API key is missing during an actual research run, the run must fail
    with GeminiMissingKeyError. It must NEVER silently fall back to the deterministic planner.
    """
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(GeminiMissingKeyError) as exc_info:
        create_analysis_plan(
            profile=sample_profile,
            objective="Analyze salary",
            is_research_run=True,
            gemini_client=None,
        )

    assert "GEMINI_API_KEY" in str(exc_info.value)


def test_research_run_mocked_llm_success_returns_llm_mode(sample_profile):
    """
    Successful genuine LLM execution in research run produces a validated
    PlannerResponse with planner_mode='llm'.
    """
    payload = make_valid_gemini_plan_payload("Investigate pay equity")
    mock_client = build_mock_gemini_client(payload)

    plan = create_analysis_plan(
        profile=sample_profile,
        objective="Investigate pay equity",
        preferences={"research_run": True},
        gemini_client=mock_client,
    )

    assert isinstance(plan, PlannerResponse)
    assert plan.planner_mode == "llm"
    assert plan.objective == "Investigate pay equity"
    assert plan.dataset_summary["planner_mode"] == "llm"


def test_research_run_llm_errors_propagate_without_fallback(sample_profile):
    """
    In research runs, network/auth/rate-limit failures must raise explicit exceptions
    and NEVER silently switch to deterministic heuristics.
    """
    # 1. Auth failure
    auth_client = build_mock_gemini_client({"error": "unauthorized"}, status_code=401)
    with pytest.raises(GeminiAuthenticationError):
        create_analysis_plan(
            profile=sample_profile,
            objective="Analyze salary",
            is_research_run=True,
            gemini_client=auth_client,
        )

    # 2. Rate limit failure
    rate_client = build_mock_gemini_client({"error": "quota exceeded"}, status_code=429)
    with pytest.raises(GeminiRateLimitError):
        create_analysis_plan(
            profile=sample_profile,
            objective="Analyze salary",
            is_research_run=True,
            gemini_client=rate_client,
        )

    # 3. Network timeout failure
    def timeout_handler(req):
        raise httpx.ReadTimeout("Gemini connection timed out.")

    net_client = GeminiClient(api_key="test_key", transport=httpx.MockTransport(timeout_handler))
    with pytest.raises(GeminiNetworkError):
        create_analysis_plan(
            profile=sample_profile,
            objective="Analyze salary",
            is_research_run=True,
            gemini_client=net_client,
        )


# ============================================================================
# 5. Comprehensive Regression Tests for Enforced LLM Planning
# ============================================================================

def test_research_mode_forces_llm_even_when_env_disables_it(sample_profile, monkeypatch):
    """
    1. Research mode forces LLM planning even when USE_LLM_PLANNER=false or DATAINVEST_RESEARCH_RUN=false in env.
    """
    monkeypatch.setenv("USE_LLM_PLANNER", "false")
    payload = make_valid_gemini_plan_payload()
    mock_client = build_mock_gemini_client(payload)

    plan = create_analysis_plan(
        profile=sample_profile,
        objective="Analyze salary",
        is_research_run=True,
        gemini_client=mock_client,
    )
    assert plan.planner_mode == "llm"
    assert plan.planner_provider == "google"


def test_normal_planner_workflow_cannot_silently_fallback(sample_profile, monkeypatch):
    """
    2. The normal intended Planner workflow cannot silently fall back to deterministic planning.
    When use_llm is omitted and key is missing, it fails explicitly with GeminiMissingKeyError.
    """
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(GeminiMissingKeyError):
        create_analysis_plan(
            profile=sample_profile,
            objective="Analyze salary",
        )


def test_missing_credentials_cause_clear_failure_not_deterministic_plan(sample_profile, monkeypatch):
    """
    3. Missing credentials cause a clear failure, not a deterministic plan.
    """
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(GeminiMissingKeyError) as exc_info:
        create_analysis_plan(sample_profile, "Analyze salary", is_research_run=True)
    assert "GEMINI_API_KEY" in str(exc_info.value)


def test_gemini_api_errors_propagate_without_fallback(sample_profile):
    """
    4. Gemini HTTP/API errors propagate as clear failures without fallback.
    """
    failing_client = build_mock_gemini_client({"error": "rate limit"}, status_code=429)
    with pytest.raises(GeminiRateLimitError):
        create_analysis_plan(
            profile=sample_profile,
            objective="Analyze salary",
            gemini_client=failing_client,
        )


def test_invalid_json_or_schema_invalid_planner_response_fails_safely(sample_profile):
    """
    5. Invalid JSON or schema-invalid planner responses fail safely with GeminiMalformedResponseError.
    """
    # 1. Non-JSON response
    non_json_client = GeminiClient(
        api_key="mock_key",
        transport=httpx.MockTransport(lambda req: httpx.Response(status_code=200, json={"candidates": [{"content": {"role": "model", "parts": [{"text": "NOT_JSON"}]}}]}))
    )
    with pytest.raises(GeminiMalformedResponseError):
        create_analysis_plan(profile=sample_profile, objective="Analyze", gemini_client=non_json_client)

    # 2. Schema-invalid response (missing analysis_tasks)
    invalid_schema_client = build_mock_gemini_client({"objective": "Test", "tasks": []})
    with pytest.raises(GeminiMalformedResponseError):
        create_analysis_plan(profile=sample_profile, objective="Analyze", gemini_client=invalid_schema_client)


def test_mocked_gemini_produces_validated_plan_with_accurate_provenance(sample_profile):
    """
    6. Successful mocked Gemini responses produce a validated LLM plan with accurate provenance.
    Only marks 'llm' and 'google' upon actual successful provider call.
    """
    payload = make_valid_gemini_plan_payload("Analyze compensation")
    mock_client = build_mock_gemini_client(payload)

    plan = create_analysis_plan(
        profile=sample_profile,
        objective="Analyze compensation",
        gemini_client=mock_client,
    )
    assert plan.planner_mode == "llm"
    assert plan.planner_status == "success"
    assert plan.planner_provider == "google"
    assert plan.planner_model == "gemini-3.5-flash-lite"
    assert plan.dataset_summary["planner_mode"] == "llm"


def test_paired_experiment_calls_planner_once_and_passes_identical_plan_to_branches(sample_profile):
    """
    7. Each paired experiment generates the plan exactly once and passes the identical plan to Baseline and Proposed.
    """
    from unittest.mock import MagicMock
    from app.services.experiment_runner import run_paired_experiment
    from tests.test_experiment_runner import (
        build_sample_analysis_response,
        build_sample_critic_response,
        build_sample_verification_response,
        build_sample_insight_response,
        build_mock_evaluation_response,
    )

    payload = make_valid_gemini_plan_payload()
    mock_client = build_mock_gemini_client(payload)
    mock_planner = MagicMock(side_effect=create_analysis_plan)

    mock_analysis_svc = MagicMock()
    mock_analysis_svc.execute_plan.return_value = build_sample_analysis_response()

    mock_eval_svc = MagicMock()
    mock_eval_svc.evaluate_pipeline.return_value = build_mock_evaluation_response(75.0, "B")

    mock_critic_svc = MagicMock()
    mock_critic_svc.run_review.return_value = build_sample_critic_response(requires_reinvestigation=False)

    mock_verif_svc = MagicMock()
    mock_verif_svc.process_verification.return_value = build_sample_verification_response(reinvestigated=False)

    mock_insight_svc = MagicMock()
    mock_insight_svc.generate_insights.return_value = build_sample_insight_response()

    response = run_paired_experiment(
        profile=sample_profile,
        planner_fn=mock_planner,
        gemini_client=mock_client,
        analysis_service=mock_analysis_svc,
        critic_service=mock_critic_svc,
        verification_service=mock_verif_svc,
        insight_service=mock_insight_svc,
        evaluation_service_inst=mock_eval_svc,
        persist_runs=False,
    )

    # 1. Planner called exactly ONCE
    assert mock_planner.call_count == 1
    # 2. Both branches receive the exact same plan
    assert response.shared_plan.planner_mode == "llm"
    assert response.shared_plan.planner_provider == "google"
    assert len(response.shared_plan.analysis_tasks) == 2
    # 3. Baseline and Proposed receive identical shared plan and task metrics
    assert response.baseline.total_tasks_executed == response.proposed.total_tasks_executed == 1


def test_paired_experiment_branch_differences_intact(sample_profile):
    """
    8. Existing paired-experiment branch differences remain intact:
    Baseline skips Critic, Verification, and Insights (stored as None).
    Proposed includes Critic, Verification, and Insights.
    """
    from unittest.mock import MagicMock
    from app.services.experiment_runner import run_paired_experiment
    from tests.test_experiment_runner import (
        build_sample_analysis_response,
        build_sample_critic_response,
        build_sample_verification_response,
        build_sample_insight_response,
        build_mock_evaluation_response,
    )

    payload = make_valid_gemini_plan_payload()
    mock_client = build_mock_gemini_client(payload)

    eval_calls = []
    def record_eval(**kwargs):
        eval_calls.append(kwargs)
        return build_mock_evaluation_response(80.0, "A")

    mock_eval_svc = MagicMock()
    mock_eval_svc.evaluate_pipeline.side_effect = record_eval

    mock_analysis_svc = MagicMock()
    mock_analysis_svc.execute_plan.return_value = build_sample_analysis_response()

    mock_critic_svc = MagicMock()
    mock_critic_svc.run_review.return_value = build_sample_critic_response(requires_reinvestigation=False)

    mock_verif_svc = MagicMock()
    mock_verif_svc.process_verification.return_value = build_sample_verification_response(reinvestigated=False)

    mock_insight_svc = MagicMock()
    mock_insight_svc.generate_insights.return_value = build_sample_insight_response()

    run_paired_experiment(
        profile=sample_profile,
        gemini_client=mock_client,
        analysis_service=mock_analysis_svc,
        critic_service=mock_critic_svc,
        verification_service=mock_verif_svc,
        insight_service=mock_insight_svc,
        evaluation_service_inst=mock_eval_svc,
        persist_runs=False,
    )

    # 2 evaluation calls: baseline first, then proposed
    assert len(eval_calls) == 2
    baseline_call, proposed_call = eval_calls[0], eval_calls[1]

    # Baseline has critic=None, verification=None, insights=None
    assert baseline_call["critic"] is None
    assert baseline_call["verification"] is None
    assert baseline_call["insights"] is None

    # Proposed has active critic, verification, insights
    assert baseline_call["plan"] == proposed_call["plan"]  # identical shared plan!
    assert proposed_call["critic"] is not None
    assert proposed_call["verification"] is not None
    assert proposed_call["insights"] is not None


def test_offline_demo_mode_isolated_from_research_mode():
    """
    9. Existing explicitly labelled offline/demo mode remains isolated from research mode.
    Never calls Gemini and stamps output as deterministic_simulated with disclaimer.
    """
    from app.services.demo_service import DemoService, DEMO_DISCLAIMER

    demo_svc = DemoService()
    result = demo_svc.run_offline_demo()

    assert result["is_offline_demo"] is True
    assert result["plan"]["planner_mode"] == "deterministic_simulated"
    assert result["plan"]["is_offline_demo"] is True
    assert result["plan"]["disclaimer"] == DEMO_DISCLAIMER


def test_api_and_frontend_request_compatibility(sample_profile):
    """
    10. Existing API and frontend request behaviour remains compatible with the enforced requirement.
    Default POST /api/plan uses LLM planning; explicit use_llm=False permits deterministic testing.
    """
    from fastapi.testclient import TestClient
    from app.main import app
    from app.services import planner_service

    client = TestClient(app)
    mock_gemini = build_mock_gemini_client(make_valid_gemini_plan_payload())

    # When request omits use_llm, it uses LLM planning (server-side enforced default)
    orig_create = planner_service.create_analysis_plan
    def mock_create(*args, **kwargs):
        kwargs["gemini_client"] = mock_gemini
        return orig_create(*args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(planner_service, "create_analysis_plan", mock_create)
        resp = client.post(
            "/api/plan",
            json={
                "profile": sample_profile,
                "objective": "Analyze salary drivers",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["planner_mode"] == "llm"
        assert data["planner_provider"] == "google"

    # Explicit use_llm=False remains supported for isolated offline testing
    resp_offline = client.post(
        "/api/plan",
        json={
            "profile": sample_profile,
            "objective": "Analyze salary drivers",
            "use_llm": False,
        },
    )
    assert resp_offline.status_code == 200
    assert resp_offline.json()["planner_mode"] == "deterministic"


def test_llm_planner_normalizes_missing_task_title_field(sample_profile):
    """
    Test that the LLM planner defensively normalizes missing or alternative 'task' fields:
    1. Task with 'title' instead of 'task' adopts the title.
    2. Task with 'task_name' instead of 'task' adopts task_name.
    3. Task with neither title nor task derives short readable title from description.
    4. Task with empty/whitespace task and no description derives fallback from task_type and columns.
    Ensures PlannerResponse.model_validate succeeds without 'Field required' validation errors.
    """
    payload = make_valid_gemini_plan_payload()

    # Task 1: has 'title' instead of 'task'
    t1 = payload["analysis_tasks"][0]
    del t1["task"]
    t1["title"] = "Analyze salary vs experience via correlation"

    # Task 2: has 'task_name' instead of 'task'
    t2 = payload["analysis_tasks"][1]
    del t2["task"]
    t2["task_name"] = "Department Salary ANOVA Comparison"

    # Task 3: omits 'task' and 'title', but has 'description'
    t3 = {
        "task_id": "task_3",
        "task_type": "distribution_analysis",
        "description": "Assess distribution and skewness of salary. Secondary exploratory review.",
        "method": "Histogram and summary statistics",
        "columns": ["salary"],
        "column_a": "salary",
        "reason": "Examine normality and distribution characteristics.",
        "priority": "low",
    }

    # Task 4: has whitespace 'task' and empty description, falls back to task_type and columns
    t4 = {
        "task_id": "task_4",
        "task_type": "numerical_vs_numerical",
        "task": "   ",
        "description": "",
        "method": "Spearman rank correlation",
        "columns": ["age", "salary"],
        "column_a": "age",
        "column_b": "salary",
        "reason": "Assess monotonic association.",
        "priority": "medium",
    }

    payload["analysis_tasks"].extend([t3, t4])

    mock_client = build_mock_gemini_client(payload)
    agent = PlannerAgent(use_llm=True, gemini_client=mock_client)
    plan = agent.plan(sample_profile, objective="Analyze salary drivers")

    assert isinstance(plan, PlannerResponse)
    assert len(plan.tasks) == 4

    # 1. 'title' preferred
    assert plan.tasks[0].task == "Analyze salary vs experience via correlation"

    # 2. 'task_name' preferred
    assert plan.tasks[1].task == "Department Salary ANOVA Comparison"

    # 3. Derived from first sentence of description
    assert plan.tasks[2].task == "Assess distribution and skewness of salary"

    # 4. Fallback constructed from task_type and columns
    assert plan.tasks[3].task == "Numerical Vs Numerical: age, salary"



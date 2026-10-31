"""
backend/tests/test_experiment_api.py
====================================
Focused offline tests for Paired Experiment API endpoint (POST /api/experiments/paired).

Requirements:
- 100% offline, zero network requests, zero real Gemini API calls.
- Successful endpoint response using a mocked runner.
- Unknown dataset handling (404).
- Invalid request handling (422).
- Missing API key (503).
- Authentication failure (502).
- Rate limit failure (429).
- Upstream Gemini network/malformed failure (502).
- Runner failure (500).
- Partial-persistence failure preserves explicit partial persistence metadata (500).
- Runner is invoked exactly once with authoritative dataset profile and correct research parameters.
- Client-supplied profile cannot override authoritative profile derived from stored dataset.
- Existing API endpoints continue to behave as before.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.experiment_schema import (
    BranchRunSummary,
    MetricDeltaSummary,
    PairedExperimentResponse,
)
from app.schemas.planner_schema import InvestigationTask, PlannerResponse
from app.services import dataset_store
from app.services.experiment_runner import (
    ExperimentPersistenceError,
    ExperimentRunnerError,
)
from app.services.gemini_client import (
    GeminiAuthenticationError,
    GeminiMissingKeyError,
    GeminiNetworkError,
    GeminiRateLimitError,
)

client = TestClient(app)


# ============================================================================
# Helpers & Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def clean_dataset_store():
    """Ensure dataset store is reset before and after each test."""
    dataset_store.clear_cache()
    yield
    dataset_store.clear_cache()


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Sample DataFrame cached in dataset_store for API tests."""
    return pd.DataFrame({
        "salary": [50000, 60000, 75000, 90000, 110000],
        "experience": [1, 3, 5, 8, 12],
        "department": ["Sales", "Sales", "Engineering", "Engineering", "Management"],
    })


def build_dummy_experiment_response(dataset_id: str = "test_data.csv") -> PairedExperimentResponse:
    """Build a valid dummy PairedExperimentResponse for mocking."""
    task = InvestigationTask(
        task_id="task_1",
        task_type="numerical_vs_numerical",
        task="Analyze salary vs experience",
        description="Correlation test",
        method="Pearson correlation",
        columns=["salary", "experience"],
        column_a="salary",
        column_b="experience",
        reason="Numerical features",
        priority="high",
    )
    plan = PlannerResponse(
        objective="Investigate salary determinants.",
        interpreted_goal="Examine salary relationships.",
        target_column="salary",
        selected_columns=[
            {"column": "salary", "data_type": "numerical", "reason": "Target", "is_target": True},
            {"column": "experience", "data_type": "numerical", "reason": "Predictor", "is_target": False},
        ],
        analysis_tasks=[task],
        tasks=[task],
        reasoning=["Planned analysis."],
        warnings=[],
        expected_outputs=["Pearson r."],
        planner_mode="llm",
    )
    baseline_summary = BranchRunSummary(
        run_id="run_20261021_baseline_abc123",
        mode="baseline",
        dataset_id=dataset_id,
        total_tasks_executed=1,
        overall_score=68.0,
        overall_grade="B - Acceptable",
        dimension_scores={"execution_reliability": 25.0, "statistical_rigor": 18.0, "verification_grounding": 15.0, "insight_actionability": 10.0},
        completed_phases=[1, 2, 3, 7],
        has_critic_review=False,
        has_verification=False,
        has_insights=False,
        persisted=True,
        paired_run_id="run_20261021_proposed_xyz789",
    )
    proposed_summary = BranchRunSummary(
        run_id="run_20261021_proposed_xyz789",
        mode="proposed",
        dataset_id=dataset_id,
        total_tasks_executed=1,
        overall_score=89.5,
        overall_grade="A - High Scientific Rigor",
        dimension_scores={"execution_reliability": 25.0, "statistical_rigor": 23.5, "verification_grounding": 21.0, "insight_actionability": 20.0},
        completed_phases=[1, 2, 3, 4, 5, 6, 7],
        has_critic_review=True,
        has_verification=True,
        has_insights=True,
        persisted=True,
        paired_run_id="run_20261021_baseline_abc123",
    )
    comparison_summary = MetricDeltaSummary(
        baseline_score=68.0,
        proposed_score=89.5,
        overall_score_delta=21.5,
        rigor_score_delta=5.5,
        verification_score_delta=6.0,
        dimension_deltas={"execution_reliability": 0.0, "statistical_rigor": 5.5, "verification_grounding": 6.0, "insight_actionability": 10.0},
        assumption_violations_flagged=1,
        violations_resolved_via_reinvestigation=1,
        resolution_rate=1.0,
        total_reinvestigations=1,
        verified_findings_count=1,
        questionable_findings_count=0,
        rejected_findings_count=0,
        scientific_verdict="Proposed pipeline improved overall score by +21.5 pts.",
    )
    return PairedExperimentResponse(
        experiment_id="exp_20261021_test_exp_123456",
        dataset_id=dataset_id,
        objective="Investigate salary determinants.",
        created_at="2026-10-21T10:00:00Z",
        shared_plan=plan,
        baseline=baseline_summary,
        proposed=proposed_summary,
        comparison=comparison_summary,
        persisted=True,
        partial_persistence=False,
    )


# ============================================================================
# API Endpoint Tests
# ============================================================================

def test_paired_experiment_success_with_mocked_runner(sample_df):
    """Verify successful 200 response when dataset is resolved and runner completes."""
    dataset_id = "employees.csv"
    dataset_store.store_dataset(dataset_id, sample_df)

    dummy_response = build_dummy_experiment_response(dataset_id)

    with patch("app.main.run_paired_experiment", return_value=dummy_response) as mock_runner:
        response = client.post(
            "/api/experiments/paired",
            json={
                "dataset_id": dataset_id,
                "objective": "Investigate salary determinants.",
                "max_reinvestigations": 2,
                "persist_runs": True,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["experiment_id"] == dummy_response.experiment_id
        assert data["baseline"]["mode"] == "baseline"
        assert data["proposed"]["mode"] == "proposed"
        assert data["comparison"]["overall_score_delta"] == 21.5

        # Verify runner was invoked exactly once with authoritative profile and correct parameters
        assert mock_runner.call_count == 1
        kwargs = mock_runner.call_args.kwargs
        assert kwargs["dataset_id"] == dataset_id
        assert kwargs["objective"] == "Investigate salary determinants."
        assert kwargs["max_reinvestigations"] == 2
        assert kwargs["persist_runs"] is True

        # Profile was computed authoritatively from stored DataFrame
        assert kwargs["profile"]["overview"]["n_rows"] == 5
        assert kwargs["profile"]["overview"]["n_cols"] == 3


def test_authoritative_profile_overrides_client_supplied_profile(sample_df):
    """Verify client-supplied bogus profile is not trusted when authoritative profile can be computed."""
    dataset_id = "test_override.csv"
    dataset_store.store_dataset(dataset_id, sample_df)

    dummy_response = build_dummy_experiment_response(dataset_id)

    bogus_profile = {
        "overview": {
            "n_rows": 999999,
            "n_cols": 999,
            "column_names": ["fake_col"],
            "filename": dataset_id,
        }
    }

    with patch("app.main.run_paired_experiment", return_value=dummy_response) as mock_runner:
        response = client.post(
            "/api/experiments/paired",
            json={
                "dataset_id": dataset_id,
                "profile": bogus_profile,
                "objective": "Investigate salary determinants.",
            },
        )

        assert response.status_code == 200
        kwargs = mock_runner.call_args.kwargs
        # The runner received the authoritative 5 rows, NOT the client's 999999 rows
        assert kwargs["profile"]["overview"]["n_rows"] == 5


def test_paired_experiment_unknown_dataset_returns_404():
    """Verify 404 response when dataset is not in cache or disk."""
    response = client.post(
        "/api/experiments/paired",
        json={
            "dataset_id": "nonexistent_unknown_dataset.csv",
            "objective": "Investigate factors.",
        },
    )
    assert response.status_code == 404
    data = response.json()
    assert "error" in data
    assert "could not be resolved" in data["error"]


def test_paired_experiment_invalid_request_validation():
    """Verify 422 responses on schema constraint violations."""
    # 1. Neither dataset_id nor profile
    resp_empty = client.post("/api/experiments/paired", json={})
    assert resp_empty.status_code == 422

    # 2. Invalid max_reinvestigations (> 3)
    resp_invalid_reinv = client.post(
        "/api/experiments/paired",
        json={"dataset_id": "some.csv", "max_reinvestigations": 5},
    )
    assert resp_invalid_reinv.status_code == 422

    # 3. Objective too long (> 1000 characters)
    resp_long_obj = client.post(
        "/api/experiments/paired",
        json={"dataset_id": "some.csv", "objective": "x" * 1005},
    )
    assert resp_long_obj.status_code == 422


def test_paired_experiment_missing_api_key_returns_503(sample_df):
    """Verify 503 response and sanitized error when Gemini API key is missing."""
    dataset_id = "test_key.csv"
    dataset_store.store_dataset(dataset_id, sample_df)

    with patch("app.main.run_paired_experiment", side_effect=GeminiMissingKeyError("GEMINI_API_KEY environment variable is not set.")):
        response = client.post(
            "/api/experiments/paired",
            json={"dataset_id": dataset_id},
        )
        assert response.status_code == 503
        data = response.json()
        assert "error" in data
        assert "credentials not configured" in data["error"]
        # Verify no secret or environment leakage
        assert "AIza" not in data["error"]


def test_paired_experiment_gemini_auth_failure_returns_502(sample_df):
    """Verify 502 response when Gemini authentication fails."""
    dataset_id = "test_auth.csv"
    dataset_store.store_dataset(dataset_id, sample_df)

    with patch("app.main.run_paired_experiment", side_effect=GeminiAuthenticationError("Unauthorized")):
        response = client.post(
            "/api/experiments/paired",
            json={"dataset_id": dataset_id},
        )
        assert response.status_code == 502
        data = response.json()
        assert "authentication failed" in data["error"]


def test_paired_experiment_gemini_rate_limit_returns_429(sample_df):
    """Verify 429 response when Gemini rate limit is hit."""
    dataset_id = "test_quota.csv"
    dataset_store.store_dataset(dataset_id, sample_df)

    with patch("app.main.run_paired_experiment", side_effect=GeminiRateLimitError("Rate limit exceeded")):
        response = client.post(
            "/api/experiments/paired",
            json={"dataset_id": dataset_id},
        )
        assert response.status_code == 429
        data = response.json()
        assert "rate limit exceeded" in data["error"]


def test_paired_experiment_upstream_network_error_returns_502(sample_df):
    """Verify 502 response on upstream Gemini network failure."""
    dataset_id = "test_net.csv"
    dataset_store.store_dataset(dataset_id, sample_df)

    with patch("app.main.run_paired_experiment", side_effect=GeminiNetworkError("Connection timed out")):
        response = client.post(
            "/api/experiments/paired",
            json={"dataset_id": dataset_id},
        )
        assert response.status_code == 502
        data = response.json()
        assert "Upstream error communicating with Gemini" in data["error"]


def test_paired_experiment_runner_failure_returns_500(sample_df):
    """Verify 500 response when analysis or verification stage fails."""
    dataset_id = "test_fail.csv"
    dataset_store.store_dataset(dataset_id, sample_df)

    with patch("app.main.run_paired_experiment", side_effect=ExperimentRunnerError("Analysis stage failed.")):
        response = client.post(
            "/api/experiments/paired",
            json={"dataset_id": dataset_id},
        )
        assert response.status_code == 500
        data = response.json()
        assert "Analysis stage failed" in data["error"]


def test_paired_experiment_partial_persistence_failure_preserves_metadata(sample_df):
    """Verify 500 response preserves explicit partial persistence metadata."""
    dataset_id = "test_persist.csv"
    dataset_store.store_dataset(dataset_id, sample_df)

    exc = ExperimentPersistenceError(
        message="Partial persistence failure: baseline run 'run_baseline_123' saved, proposed failed.",
        partial_persistence=True,
        saved_run_id="run_baseline_123",
    )

    with patch("app.main.run_paired_experiment", side_effect=exc):
        response = client.post(
            "/api/experiments/paired",
            json={"dataset_id": dataset_id},
        )
        assert response.status_code == 500
        data = response.json()
        assert data["error"] == "Failed to persist paired experiment run records."
        assert data["partial_persistence"] is True
        assert data["saved_run_id"] == "run_baseline_123"
        assert "Partial persistence failure" in data["detail"]


def test_existing_api_endpoints_continue_to_behave():
    """Verify existing root, health, and profile endpoints remain intact."""
    # 1. Root
    resp_root = client.get("/")
    assert resp_root.status_code == 200
    assert resp_root.json()["status"] == "ok"

    # 2. Health
    resp_health = client.get("/health")
    assert resp_health.status_code == 200
    assert resp_health.json()["status"] == "healthy"

    # 3. Profile upload
    csv_bytes = b"col_a,col_b\n10,20\n30,40\n"
    resp_profile = client.post(
        "/api/profile",
        files={"file": ("test_upload.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert resp_profile.status_code == 200
    profile_data = resp_profile.json()
    assert profile_data["overview"]["n_rows"] == 2
    assert profile_data["overview"]["n_cols"] == 2

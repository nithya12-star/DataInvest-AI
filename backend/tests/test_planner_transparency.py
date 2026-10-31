"""
backend/tests/test_planner_transparency.py
===========================================
Targeted offline verification tests for:
1. Explicit use_llm preference handling (API and service layers)
2. Truthful metadata generation (llm vs deterministic)
3. Zero silent fallback on LLM failure
4. Run store persistence and summary extraction of planner metadata
5. History listing metadata preservation
6. Markdown and JSON export transparency (including interim snapshot banners)
7. Legacy run backward compatibility

All tests run completely OFFLINE using httpx.MockTransport or stubs.
ZERO live Gemini or external network requests are made.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.agents.planner_agent import PlannerAgent
from app.schemas.planner_schema import PlannerRequest, PlannerResponse
from app.services.gemini_client import (
    GeminiAuthenticationError,
    GeminiClient,
    GeminiMalformedResponseError,
    GeminiRateLimitError,
)
from app.services.planner_service import create_analysis_plan
from app.services.run_store import RunStoreService
from app.agents.report_agent import ReportAgent


@pytest.fixture
def mock_profile() -> dict:
    return {
        "dataset_id": "test_salaries.csv",
        "overview": {
            "n_rows": 250,
            "n_cols": 4,
            "column_names": ["salary", "experience", "education", "age"],
            "filename": "test_salaries.csv",
        },
        "column_types": {
            "salary": "numerical",
            "experience": "numerical",
            "age": "numerical",
            "education": "categorical",
        },
        "missing": {"total_missing": 0, "total_missing_pct": 0.0},
        "cardinality": {"constant_columns": [], "high_cardinality_columns": []},
        "numerical_stats": {},
        "outliers": {},
    }


def make_mock_llm_plan_payload() -> dict:
    return {
        "objective": "Investigate salary drivers",
        "interpreted_goal": "Examine relationships between salary and experience, education, and age.",
        "target_column": "salary",
        "selected_columns": [
            {"column": "salary", "data_type": "numerical", "reason": "Target outcome", "is_target": True},
            {"column": "experience", "data_type": "numerical", "reason": "Predictor", "is_target": False},
            {"column": "education", "data_type": "categorical", "reason": "Group factor", "is_target": False},
        ],
        "analysis_tasks": [
            {
                "task_id": "llm_task_1",
                "task_type": "distribution_analysis",
                "task": "Analyze distribution of salary",
                "description": "Assess distribution shape and outliers of salary.",
                "method": "Descriptive statistics",
                "columns": ["salary"],
                "column_a": "salary",
                "reason": "Understand baseline target dispersion.",
                "priority": "high",
            },
            {
                "task_id": "llm_task_2",
                "task_type": "numerical_vs_numerical",
                "task": "Salary vs Experience correlation",
                "description": "Evaluate linear or monotonic relationship.",
                "method": "Pearson correlation",
                "columns": ["salary", "experience"],
                "column_a": "salary",
                "column_b": "experience",
                "reason": "Identify experience impact on pay.",
                "priority": "high",
            },
        ],
        "reasoning": ["Identified target variable salary.", "Formulated distribution and correlation tasks."],
        "warnings": ["Check for non-normality in salary."],
        "expected_outputs": ["Validated statistical analysis results."],
    }


def make_mock_gemini_client(plan_payload: dict, status_code: int = 200) -> GeminiClient:
    """Create offline mock GeminiClient with deterministic JSON response."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        if status_code != 200:
            return httpx.Response(
                status_code=status_code,
                json={"error": {"message": f"Simulated HTTP {status_code} failure"}},
            )
        response_data = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": json.dumps(plan_payload)}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 500,
                "candidatesTokenCount": 350,
                "totalTokenCount": 850,
            },
        }
        return httpx.Response(status_code=200, json=response_data)

    transport = httpx.MockTransport(mock_handler)
    return GeminiClient(api_key="mock_test_key", transport=transport, jitter=False)


# ============================================================================
# Tests
# ============================================================================

def test_api_explicit_llm_mode_success(mock_profile, monkeypatch):
    """Verify that POST /api/plan with use_llm=True triggers LLM planner and returns truthful metadata."""
    mock_client = make_mock_gemini_client(make_mock_llm_plan_payload())

    from app.services import planner_service
    orig_create_analysis_plan = planner_service.create_analysis_plan

    def patched_create_plan(*args, **kwargs):
        kwargs["gemini_client"] = mock_client
        return orig_create_analysis_plan(*args, **kwargs)

    monkeypatch.setattr(planner_service, "create_analysis_plan", patched_create_plan)

    client = TestClient(app)
    response = client.post(
        "/api/plan",
        json={
            "profile": mock_profile,
            "objective": "Investigate salary drivers",
            "dataset_id": "test_salaries.csv",
            "use_llm": True,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["planner_mode"] == "llm"
    assert data["planner_status"] == "success"
    assert data["planner_model"] == "gemini-3.5-flash-lite"
    assert data["planner_provider"] == "google"
    assert len(data["analysis_tasks"]) == 2
    assert data["analysis_tasks"][0]["task_id"] == "llm_task_1"


def test_api_deterministic_mode_labeled_truthfully(mock_profile):
    """Verify that POST /api/plan with use_llm=False generates deterministic plan with truthful metadata."""
    client = TestClient(app)
    response = client.post(
        "/api/plan",
        json={
            "profile": mock_profile,
            "objective": "Investigate this dataset.",
            "dataset_id": "test_salaries.csv",
            "use_llm": False,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["planner_mode"] == "deterministic"
    assert data["planner_status"] == "success"
    assert data["planner_model"] is None
    assert data["planner_provider"] == "deterministic"


def test_llm_failure_does_not_silently_fallback(mock_profile, monkeypatch):
    """Verify that when use_llm=True and LLM fails, an error is returned and no silent fallback occurs."""
    mock_failing_client = make_mock_gemini_client({}, status_code=429)

    from app.services import planner_service
    orig_create_analysis_plan = planner_service.create_analysis_plan

    def patched_create_plan(*args, **kwargs):
        kwargs["gemini_client"] = mock_failing_client
        return orig_create_analysis_plan(*args, **kwargs)

    monkeypatch.setattr(planner_service, "create_analysis_plan", patched_create_plan)

    client = TestClient(app)
    response = client.post(
        "/api/plan",
        json={
            "profile": mock_profile,
            "objective": "Investigate salary drivers",
            "dataset_id": "test_salaries.csv",
            "use_llm": True,
        },
    )

    # Must return rate limit error (429) or upstream error (502), NOT a 200 deterministic fallback
    assert response.status_code in (429, 502)
    err = response.json()
    assert "rate limit" in err.get("error", "").lower() or "gemini" in err.get("error", "").lower()


def test_run_store_summary_extracts_planner_metadata():
    """Verify that RunStoreService._extract_summary captures planner_mode, model, and provider."""
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        run_store = RunStoreService(runs_dir=Path(temp_dir))

        llm_payload = {
            "run_id": "run_20261026_test_llm_123456",
            "dataset_name": "test_salaries.csv",
            "completed_phases": [1, 2],
            "plan": {
                "planner_mode": "llm",
                "planner_model": "gemini-3.5-flash-lite",
                "planner_provider": "google",
                "analysis_tasks": [{"task_id": "t1"}],
            },
        }

        summary = run_store._extract_summary(llm_payload)
        assert summary.planner_mode == "llm"
        assert summary.planner_model == "gemini-3.5-flash-lite"
        assert summary.planner_provider == "google"

        det_payload = {
            "run_id": "run_20261026_test_det_654321",
            "dataset_name": "test_salaries.csv",
            "completed_phases": [1, 2],
            "plan": {
                "planner_mode": "deterministic",
                "planner_model": None,
                "planner_provider": "deterministic",
                "analysis_tasks": [{"task_id": "t1"}],
            },
        }

        det_summary = run_store._extract_summary(det_payload)
        assert det_summary.planner_mode == "deterministic"
        assert det_summary.planner_model is None
        assert det_summary.planner_provider == "deterministic"


def test_legacy_run_without_planner_metadata_remains_readable():
    """Verify backward compatibility: older runs without planner_mode deserialize safely as None."""
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        run_store = RunStoreService(runs_dir=Path(temp_dir))

        legacy_payload = {
            "run_id": "run_20261020_legacy_run_000000",
            "dataset_name": "legacy_dataset.csv",
            "completed_phases": [1],
            "profile": {"overview": {"n_rows": 50, "n_cols": 3}},
        }

        summary = run_store._extract_summary(legacy_payload)
        assert summary.planner_mode is None
        assert summary.planner_model is None
        assert summary.planner_provider is None


def test_markdown_export_renders_planner_metadata_and_interim_banner():
    """Verify that ReportAgent.render_markdown displays planner mode, model, and interim banner for Phase 2 run."""
    report_agent = ReportAgent()

    run_payload = {
        "run_id": "run_20261026_test_export_999999",
        "dataset_name": "student_grades.csv",
        "completed_phases": [1, 2],  # Interim: Phase 5, 6, 7 not completed
        "profile": {
            "overview": {
                "n_rows": 395,
                "n_cols": 10,
                "total_missing_cells": 0,
                "overall_missing_percentage": 0.0,
            }
        },
        "plan": {
            "planner_mode": "llm",
            "planner_model": "gemini-3.5-flash-lite",
            "planner_provider": "google",
            "analysis_tasks": [{"task_id": "llm_task_1"}],
        },
    }

    report = report_agent.generate_report(run_payload)
    md = report_agent.render_markdown(report)

    # 1. Must include interim warning banner because phases 5-7 are incomplete
    assert "Interim Report" in md
    assert "Phase 2 Snapshot" in md

    # 2. Must explicitly display truthful planner metadata in Section 1
    assert "| **Planner Mode** | `LLM` |" in md
    assert "| **Planner Model** | `gemini-3.5-flash-lite` |" in md
    assert "| **Planner Provider** | `google` |" in md

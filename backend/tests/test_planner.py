"""
backend/tests/test_planner.py
==============================
DataInvest AI — Phase 2: Planner Agent Tests

Covers:
- Default and custom objective interpretation
- Target column detection
- Numerical vs Numerical planning (Pearson / Spearman selection)
- Two Groups vs Numerical planning (Welch's t-test / Mann-Whitney U)
- Multiple Groups vs Numerical planning (ANOVA / Kruskal-Wallis)
- Categorical vs Categorical planning (Chi-square test of independence)
- Outlier sensitivity task generation
- Constant column exclusion
- High-cardinality handling
- Missing-value data quality awareness
- Task deduplication & non-symmetric pairing
- Pydantic schema validation for PlannerResponse
- API endpoint integration (POST /api/plan)
- Clean error handling (missing profile, empty profile, invalid inputs)
"""

from __future__ import annotations

import os
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure backend/app is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from app.main import app
from app.modules.profiler import profile_dataset
from app.agents.planner_agent import PlannerAgent
from app.services.planner_service import create_analysis_plan
from app.schemas.planner_schema import PlannerResponse, InvestigationTask

import pandas as pd
import numpy as np

client = TestClient(app)


# ──────────────────────────────────────────────────────────────────────────────
# Test Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def mixed_df() -> pd.DataFrame:
    """Rich mixed dataset for multi-type planning."""
    return pd.DataFrame({
        "age": [22, 28, 35, 45, 52, 60, 29, 38],
        "salary": [45000.0, 52000.0, 75000.0, 88000.0, 110000.0, 135000.0, 62000.0, 80000.0],
        "score": [85.5, 90.0, 78.0, 92.5, 88.0, 95.0, 82.0, 89.0],
        "gender": ["F", "M", "F", "M", "F", "M", "F", "M"],  # 2 groups
        "department": ["Eng", "Eng", "HR", "Sales", "Sales", "Eng", "HR", "Sales"],  # 3 groups
        "city": ["NY", "LA", "NY", "CHI", "LA", "NY", "CHI", "LA"],  # 3 groups
        "constant_col": [1, 1, 1, 1, 1, 1, 1, 1],  # Constant
        "user_id": [f"USR_{i}" for i in range(8)],  # High cardinality ID
    })


@pytest.fixture()
def mixed_profile(mixed_df) -> dict:
    return profile_dataset(mixed_df)


@pytest.fixture()
def numerical_only_df() -> pd.DataFrame:
    return pd.DataFrame({
        "height": [160, 165, 170, 175, 180, 185],
        "weight": [55, 62, 68, 74, 82, 90],
        "shoe_size": [38, 39, 41, 42, 44, 45],
    })


@pytest.fixture()
def categorical_only_df() -> pd.DataFrame:
    return pd.DataFrame({
        "device": ["iOS", "Android", "iOS", "Android", "Web", "iOS"],
        "browser": ["Safari", "Chrome", "Safari", "Chrome", "Firefox", "Chrome"],
        "subscribed": ["Yes", "No", "Yes", "No", "No", "Yes"],
    })


@pytest.fixture()
def outlier_df() -> pd.DataFrame:
    return pd.DataFrame({
        "price": [10.0, 12.0, 11.5, 13.0, 12.5, 11.0, 10.5, 12.0, 1000.0],
        "units": [100, 110, 105, 95, 102, 108, 112, 104, 98],
    })


# ──────────────────────────────────────────────────────────────────────────────
# 1. Objective Interpretation & Target Variable Detection
# ──────────────────────────────────────────────────────────────────────────────


class TestObjectiveInterpretation:
    def test_default_objective(self, mixed_profile):
        agent = PlannerAgent()
        plan = agent.plan(mixed_profile, "Investigate this dataset.")
        assert plan.objective == "Investigate this dataset."
        assert len(plan.analysis_tasks) > 0
        assert plan.planner_mode == "deterministic"

    def test_empty_objective_uses_default(self, mixed_profile):
        agent = PlannerAgent()
        plan = agent.plan(mixed_profile, "")
        assert plan.objective == "Investigate this dataset."
        assert len(plan.analysis_tasks) > 0

    def test_target_variable_detected(self, mixed_profile):
        agent = PlannerAgent()
        plan = agent.plan(mixed_profile, "Find factors affecting salary")
        assert plan.target_column == "salary"
        assert any(t.is_target for t in plan.selected_columns if t.column == "salary")

    def test_bivariate_relationship_objective(self, mixed_profile):
        agent = PlannerAgent()
        plan = agent.plan(mixed_profile, "Understand relationship between age and score")
        rel_tasks = [t for t in plan.analysis_tasks if t.task_type == "numerical_vs_numerical"]
        assert len(rel_tasks) > 0
        # Age vs Score should be present and high priority
        target_task = [t for t in rel_tasks if set(t.columns) == {"age", "score"}]
        assert len(target_task) == 1
        assert target_task[0].priority == "high"


# ──────────────────────────────────────────────────────────────────────────────
# 2. Planning by Analysis Category
# ──────────────────────────────────────────────────────────────────────────────


class TestAnalysisCategories:
    def test_numerical_vs_numerical_planning(self, numerical_only_df):
        profile = profile_dataset(numerical_only_df)
        plan = PlannerAgent().plan(profile, "Explore correlations")
        rel_tasks = [t for t in plan.analysis_tasks if t.task_type == "numerical_vs_numerical"]
        assert len(rel_tasks) >= 1
        for t in rel_tasks:
            assert t.method in ("Pearson correlation", "Spearman rank correlation")
            assert t.column_a in numerical_only_df.columns
            assert t.column_b in numerical_only_df.columns
            assert t.column_a != t.column_b

    def test_two_groups_vs_numerical_planning(self, mixed_profile):
        plan = PlannerAgent().plan(mixed_profile, "Compare gender differences")
        two_group_tasks = [t for t in plan.analysis_tasks if t.task_type == "two_groups_vs_numerical"]
        assert len(two_group_tasks) >= 1
        task = two_group_tasks[0]
        assert task.group_column == "gender"
        assert task.method in ("Welch's t-test", "Mann-Whitney U test")

    def test_multiple_groups_vs_numerical_planning(self, mixed_profile):
        plan = PlannerAgent().plan(mixed_profile, "Compare departments")
        multi_group_tasks = [t for t in plan.analysis_tasks if t.task_type == "multiple_groups_vs_numerical"]
        assert len(multi_group_tasks) >= 1
        task = [t for t in multi_group_tasks if t.group_column == "department"][0]
        assert task.group_column == "department"
        assert task.method in ("One-way ANOVA", "Kruskal-Wallis test")

    def test_categorical_vs_categorical_planning(self, categorical_only_df):
        profile = profile_dataset(categorical_only_df)
        plan = PlannerAgent().plan(profile, "Analyze categorical associations")
        cat_tasks = [t for t in plan.analysis_tasks if t.task_type == "categorical_vs_categorical"]
        assert len(cat_tasks) >= 1
        task = cat_tasks[0]
        assert task.method == "Chi-square test of independence"
        assert any("Cramér's V" in n for n in (task.notes or []))

    def test_outlier_sensitivity_task(self, outlier_df):
        profile = profile_dataset(outlier_df)
        plan = PlannerAgent().plan(profile, "Investigate price outliers")
        outlier_tasks = [t for t in plan.analysis_tasks if t.task_type == "outlier_sensitivity"]
        assert len(outlier_tasks) >= 1
        assert outlier_tasks[0].column_a == "price"
        assert outlier_tasks[0].method == "IQR-based sensitivity analysis"


# ──────────────────────────────────────────────────────────────────────────────
# 3. Data Quality & Constraints Awareness
# ──────────────────────────────────────────────────────────────────────────────


class TestDataQualityAwareness:
    def test_constant_column_excluded(self, mixed_profile):
        plan = PlannerAgent().plan(mixed_profile, "Investigate all columns")
        # constant_col should not be compared in any bivariate task
        for task in plan.analysis_tasks:
            assert "constant_col" not in (task.columns or [])
            assert task.column_a != "constant_col"
            assert task.column_b != "constant_col"
            assert task.group_column != "constant_col"

    def test_high_cardinality_id_excluded_from_grouping(self, mixed_profile):
        plan = PlannerAgent().plan(mixed_profile, "Investigate dataset")
        group_tasks = [t for t in plan.analysis_tasks if t.task_type in ("two_groups_vs_numerical", "multiple_groups_vs_numerical")]
        for t in group_tasks:
            assert t.group_column != "user_id"

    def test_missing_values_warning_and_task(self):
        df_missing = pd.DataFrame({
            "a": [1, 2, None, 4, 5, None, 7, 8],
            "b": [10, 20, 30, 40, 50, 60, 70, 80],
        })
        profile = profile_dataset(df_missing)
        plan = PlannerAgent().plan(profile, "Analyze data")
        assert any("missing" in w.lower() for w in plan.warnings)
        assert any(t.task_type == "missing_value_investigation" for t in plan.analysis_tasks)

    def test_no_symmetric_duplicate_pairs(self, numerical_only_df):
        profile = profile_dataset(numerical_only_df)
        plan = PlannerAgent().plan(profile, "Investigate correlations")
        rel_tasks = [t for t in plan.analysis_tasks if t.task_type == "numerical_vs_numerical"]
        seen_pairs = set()
        for t in rel_tasks:
            pair = frozenset([t.column_a, t.column_b])
            assert pair not in seen_pairs, f"Duplicate symmetric pair found: {pair}"
            seen_pairs.add(pair)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Strict Non-Fabrication Rule (No Statistics Computed)
# ──────────────────────────────────────────────────────────────────────────────


class TestNoStatisticalFabrication:
    def test_no_statistical_results_in_plan(self, mixed_profile):
        plan = PlannerAgent().plan(mixed_profile, "Analyze everything")
        dumped = plan.model_dump_json()

        # Confirm no p-values or correlation numbers are fabricated
        assert "p-value =" not in dumped
        assert "correlation =" not in dumped
        assert "F-statistic =" not in dumped
        assert "t-statistic =" not in dumped
        assert "strongly affects" not in dumped

        for task in plan.analysis_tasks:
            # Tasks should state analytical descriptions and methods, not conclusions
            assert not hasattr(task, "p_value")
            assert not hasattr(task, "correlation_coefficient")
            assert not hasattr(task, "t_statistic")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Planner Service & Pydantic Contracts
# ──────────────────────────────────────────────────────────────────────────────


class TestPlannerService:
    def test_service_creates_valid_plan(self, mixed_profile):
        response = create_analysis_plan(
            profile=mixed_profile,
            objective="Analyze factors related to salary",
            dataset_id="employee_data.csv",
            use_llm=False,
        )
        assert isinstance(response, PlannerResponse)
        assert response.target_column == "salary"
        assert len(response.analysis_tasks) > 0
        assert response.tasks == response.analysis_tasks

    def test_service_rejects_empty_profile(self):
        with pytest.raises(ValueError, match="valid dataset profile"):
            create_analysis_plan(profile=None)

        with pytest.raises(ValueError, match="valid dataset profile"):
            create_analysis_plan(profile={})


# ──────────────────────────────────────────────────────────────────────────────
# 6. API Endpoint Integration (POST /api/plan)
# ──────────────────────────────────────────────────────────────────────────────


class TestPlannerApiEndpoint:
    def test_api_plan_endpoint_explicit_deterministic_success(self, mixed_profile):
        """Explicit use_llm=False permits deterministic planning for isolated offline/testing."""
        response = client.post(
            "/api/plan",
            json={
                "profile": mixed_profile,
                "objective": "Understand salary and department relationships",
                "dataset_id": "test_dataset.csv",
                "use_llm": False,
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "objective" in data
        assert "interpreted_goal" in data
        assert "analysis_tasks" in data
        assert len(data["analysis_tasks"]) > 0

        # Validate against Pydantic schema
        model = PlannerResponse.model_validate(data)
        assert model.planner_mode == "deterministic"

    def test_api_plan_endpoint_enforces_llm_when_unspecified(self, mixed_profile, monkeypatch):
        """When use_llm is unspecified, /api/plan enforces Gemini LLM and fails if key is missing (no silent fallback)."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        response = client.post(
            "/api/plan",
            json={
                "profile": mixed_profile,
                "objective": "Understand salary and department relationships",
                "dataset_id": "test_dataset.csv",
            }
        )
        assert response.status_code == 503
        data = response.json()
        assert "Gemini API credentials not configured" in data.get("error", "")

    def test_api_plan_default_objective(self, mixed_profile):
        response = client.post(
            "/api/plan",
            json={"profile": mixed_profile, "use_llm": False}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["objective"] == "Investigate this dataset."

    def test_api_plan_missing_profile_error(self):
        response = client.post(
            "/api/plan",
            json={"objective": "Explore data"}
        )
        assert response.status_code == 422
        data = response.json()
        assert "error" in data

    def test_api_plan_too_long_objective(self, mixed_profile):
        long_obj = "A" * 1500
        response = client.post(
            "/api/plan",
            json={"profile": mixed_profile, "objective": long_obj}
        )
        assert response.status_code == 422
        assert "too long" in response.json()["error"].lower()

    def test_api_plan_with_real_sample_csv(self):
        sample_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample.csv")
        df = pd.read_csv(sample_path)
        profile = profile_dataset(df)

        response = client.post(
            "/api/plan",
            json={
                "profile": profile,
                "objective": "Find factors associated with salary and performance score",
                "use_llm": False,
            }
        )
        assert response.status_code == 200
        data = response.json()
        model = PlannerResponse.model_validate(data)
        assert model.target_column in ("salary", "score")
        assert len(model.analysis_tasks) >= 4

    def test_api_plan_missing_api_key_returns_503(self, mixed_profile):
        from unittest.mock import patch
        from app.services.gemini_client import GeminiMissingKeyError

        with patch("app.main.planner_service.create_analysis_plan", side_effect=GeminiMissingKeyError()):
            response = client.post("/api/plan", json={"profile": mixed_profile})
            assert response.status_code == 503
            data = response.json()
            assert "error" in data
            assert "credentials not configured" in data["error"]

    def test_api_plan_gemini_auth_failure_returns_502(self, mixed_profile):
        from unittest.mock import patch
        from app.services.gemini_client import GeminiAuthenticationError

        with patch("app.main.planner_service.create_analysis_plan", side_effect=GeminiAuthenticationError("Unauthorized")):
            response = client.post("/api/plan", json={"profile": mixed_profile})
            assert response.status_code == 502
            data = response.json()
            assert "authentication failed" in data["error"]

    def test_api_plan_gemini_rate_limit_returns_429(self, mixed_profile):
        from unittest.mock import patch
        from app.services.gemini_client import GeminiRateLimitError

        with patch("app.main.planner_service.create_analysis_plan", side_effect=GeminiRateLimitError("Quota exceeded")):
            response = client.post("/api/plan", json={"profile": mixed_profile})
            assert response.status_code == 429
            data = response.json()
            assert "rate limit exceeded" in data["error"]

    def test_api_plan_gemini_timeout_returns_504(self, mixed_profile):
        from unittest.mock import patch
        from app.services.gemini_client import GeminiTimeoutError

        with patch("app.main.planner_service.create_analysis_plan", side_effect=GeminiTimeoutError("Read timed out")):
            response = client.post("/api/plan", json={"profile": mixed_profile})
            assert response.status_code == 504
            data = response.json()
            assert "timed out" in data["error"].lower()

    def test_api_plan_upstream_network_error_returns_502(self, mixed_profile):
        from unittest.mock import patch
        from app.services.gemini_client import GeminiNetworkError

        with patch("app.main.planner_service.create_analysis_plan", side_effect=GeminiNetworkError("DNS lookup failed")):
            response = client.post("/api/plan", json={"profile": mixed_profile})
            assert response.status_code == 502
            data = response.json()
            assert "Upstream error communicating with Gemini" in data["error"]

    def test_api_plan_internal_error_returns_500_with_sanitized_message(self, mixed_profile):
        from unittest.mock import patch

        with patch("app.main.planner_service.create_analysis_plan", side_effect=RuntimeError("Unexpected internal planner crash")):
            response = client.post("/api/plan", json={"profile": mixed_profile})
            assert response.status_code == 500
            data = response.json()
            assert "Planning service encountered an internal error" in data["error"]
            assert "dataset profile" not in data["error"]

"""
backend/tests/test_analysis.py
==============================
Unit and integration test suite for DataInvest AI — Phase 3: Analysis Agent & Statistical Execution.

Verifies:
1. Pure Python / SciPy mathematical correctness (Pearson, Spearman, Welch's t-test,
   Mann-Whitney, ANOVA, Kruskal-Wallis, Chi-square, Cramér's V, IQR Outlier Sensitivity).
2. Complete edge-case handling (constant variables, insufficient N, missing values).
3. Failure isolation in multi-task plans.
4. FastAPI endpoint contracts (/api/analyze, /api/analyze-plan).
5. Zero hallucinated or fabricated statistics.
"""

import pytest
import numpy as np
import pandas as pd
from scipy import stats
from fastapi.testclient import TestClient

from app.main import app
from app.agents.analysis_agent import AnalysisAgent
from app.schemas.planner_schema import InvestigationTask
from app.schemas.analysis_schema import SingleAnalysisRequest, PlanAnalysisRequest
from app.services.dataset_store import store_dataset, clear_cache
from app.statistics.correlation import execute_correlation
from app.statistics.group_comparison import (
    execute_two_group_comparison,
    execute_multi_group_comparison,
)
from app.statistics.categorical import execute_categorical_association
from app.statistics.sensitivity import execute_outlier_sensitivity
from app.statistics.descriptive import (
    execute_distribution_analysis,
    execute_missingness_audit,
)


@pytest.fixture(autouse=True)
def cleanup():
    """Clear cache before each test."""
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_df():
    """Standard multi-type DataFrame for testing."""
    np.random.seed(42)
    n = 100
    x = np.linspace(1, 10, n)
    y = 2.5 * x + np.random.normal(0, 1, n)
    cat1 = np.random.choice(["Alpha", "Beta"], size=n)
    cat2 = np.random.choice(["North", "South", "East"], size=n)
    val_group = np.where(cat1 == "Alpha", np.random.normal(10, 2, n), np.random.normal(20, 2, n))

    df = pd.DataFrame({
        "x": x,
        "y": y,
        "cat1": cat1,
        "cat2": cat2,
        "val_group": val_group,
        "const_col": [42.0] * n,
    })
    # Add a couple outliers to y
    df.loc[0, "y"] = 250.0
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 1. CORRELATION TESTS
# ──────────────────────────────────────────────────────────────────────────────


def test_pearson_correlation_linear():
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [2.0, 4.0, 6.0, 8.0, 10.0],
    })
    success, results, warnings, error = execute_correlation(df, "a", "b", "Pearson correlation")
    assert success is True
    assert error is None
    assert pytest.approx(results["correlation_coefficient"], 0.0001) == 1.0
    assert results["p_value"] < 0.001
    assert results["association_strength"] == "strong"
    assert results["direction"] == "positive"
    assert results["sample_size"] == 5


def test_spearman_correlation_monotonic():
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [1.0, 10.0, 100.0, 1000.0, 10000.0],
    })
    success, results, warnings, error = execute_correlation(df, "a", "b", "Spearman rank correlation")
    assert success is True
    assert pytest.approx(results["correlation_coefficient"], 0.0001) == 1.0
    assert results["p_value"] < 0.01


def test_correlation_constant_variable():
    df = pd.DataFrame({
        "a": [5.0, 5.0, 5.0, 5.0, 5.0],
        "b": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    success, results, warnings, error = execute_correlation(df, "a", "b")
    assert success is False
    assert error is not None
    assert error["code"] == "CONSTANT_VARIABLE"


def test_correlation_insufficient_rows():
    df = pd.DataFrame({
        "a": [1.0, 2.0],
        "b": [3.0, 4.0],
    })
    success, results, warnings, error = execute_correlation(df, "a", "b")
    assert success is False
    assert error["code"] == "INSUFFICIENT_OBSERVATIONS"


def test_correlation_missing_data():
    df = pd.DataFrame({
        "a": [1.0, 2.0, np.nan, 4.0, 5.0],
        "b": [2.0, np.nan, 6.0, 8.0, 10.0],
    })
    success, results, warnings, error = execute_correlation(df, "a", "b")
    assert success is True
    assert results["sample_size"] == 3
    assert len(warnings) > 0


# ──────────────────────────────────────────────────────────────────────────────
# 2. GROUP COMPARISON TESTS
# ──────────────────────────────────────────────────────────────────────────────


def test_welch_ttest_two_groups():
    np.random.seed(42)
    g1 = np.random.normal(10, 1, 30)
    g2 = np.random.normal(15, 1, 30)
    df = pd.DataFrame({
        "group": ["A"] * 30 + ["B"] * 30,
        "score": np.concatenate([g1, g2]),
    })
    success, results, warnings, error = execute_two_group_comparison(df, "group", "score", "Welch's t-test")
    assert success is True
    assert results["p_value"] < 0.0001
    assert results["p_value_below_0_05"] is True
    assert results["group_1"]["n"] == 30
    assert results["group_2"]["n"] == 30
    assert abs(results["mean_difference"]) > 4.0


def test_mann_whitney_u_test():
    df = pd.DataFrame({
        "group": ["A"] * 5 + ["B"] * 5,
        "val": [1, 2, 3, 4, 5, 10, 20, 30, 40, 50],
    })
    success, results, warnings, error = execute_two_group_comparison(df, "group", "val", "Mann-Whitney U test")
    assert success is True
    assert results["p_value"] < 0.05
    assert results["sample_size"] == 10


def test_one_way_anova_multi_groups():
    np.random.seed(42)
    df = pd.DataFrame({
        "cat": ["G1"] * 20 + ["G2"] * 20 + ["G3"] * 20,
        "score": np.concatenate([
            np.random.normal(10, 1, 20),
            np.random.normal(15, 1, 20),
            np.random.normal(20, 1, 20),
        ]),
    })
    success, results, warnings, error = execute_multi_group_comparison(df, "cat", "score", "One-way ANOVA")
    assert success is True
    assert results["p_value"] < 0.001
    assert results["num_groups"] == 3
    assert len(results["groups"]) == 3
    assert "F_statistic" in results


def test_kruskal_wallis_multi_groups():
    df = pd.DataFrame({
        "cat": ["G1"] * 5 + ["G2"] * 5 + ["G3"] * 5,
        "score": [1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24],
    })
    success, results, warnings, error = execute_multi_group_comparison(df, "cat", "score", "Kruskal-Wallis test")
    assert success is True
    assert results["p_value"] < 0.05
    assert "H_statistic" in results


# ──────────────────────────────────────────────────────────────────────────────
# 3. CATEGORICAL ASSOCIATION TESTS
# ──────────────────────────────────────────────────────────────────────────────


def test_chi_square_and_cramer_v():
    df = pd.DataFrame({
        "gender": ["M"] * 40 + ["F"] * 40,
        "preference": ["Yes"] * 30 + ["No"] * 10 + ["Yes"] * 10 + ["No"] * 30,
    })
    success, results, warnings, error = execute_categorical_association(df, "gender", "preference")
    assert success is True
    assert results["p_value"] < 0.001
    assert results["degrees_of_freedom"] == 1
    assert results["cramer_v"] > 0.4
    assert "contingency_table" in results
    assert results["sample_size"] == 80


# ──────────────────────────────────────────────────────────────────────────────
# 4. SENSITIVITY AND DESCRIPTIVE TESTS
# ──────────────────────────────────────────────────────────────────────────────


def test_outlier_sensitivity():
    # 20 points around 10, plus 2 severe outliers (100, 120)
    vals = [10.0 + (i % 3) for i in range(20)] + [100.0, 120.0]
    df = pd.DataFrame({"metric": vals})

    success, results, warnings, error = execute_outlier_sensitivity(df, "metric")
    assert success is True
    assert results["outlier_count"] == 2
    assert results["original_sample_size"] == 22
    assert results["filtered_sample_size"] == 20
    assert results["baseline_metrics"]["mean"] > results["filtered_metrics"]["mean"]
    assert results["substantial_influence"] is True


def test_distribution_analysis():
    np.random.seed(42)
    df = pd.DataFrame({"normal_var": np.random.normal(50, 10, 100)})
    success, results, warnings, error = execute_distribution_analysis(df, "normal_var")
    assert success is True
    assert results["variable_type"] == "numeric"
    assert pytest.approx(results["mean"], 2.0) == 50.0
    assert results["sample_size"] == 100
    assert "normality_test" in results
    assert "skewness" in results
    assert "kurtosis" in results


def test_distribution_analysis_categorical_yes_no():
    """Verify categorical distribution analysis on yes/no target column (e.g., bank-full.csv 'y')."""
    df = pd.DataFrame({
        "y": ["no"] * 88 + ["yes"] * 12
    })
    success, results, warnings, error = execute_distribution_analysis(df, "y")
    assert success is True
    assert error is None
    assert results["variable_type"] == "categorical"
    assert results["column"] == "y"
    assert results["sample_size"] == 100
    assert results["num_categories"] == 2
    assert results["frequencies"] == {"no": 88, "yes": 12}
    assert pytest.approx(results["proportions"]["no"], 0.001) == 0.88
    assert pytest.approx(results["proportions"]["yes"], 0.001) == 0.12
    assert results["percentages"]["no"] == 88.0
    assert results["percentages"]["yes"] == 12.0
    assert results["mode"] == "no"
    assert results["mode_frequency"] == 88
    assert results["mode_percentage"] == 88.0
    assert results["is_imbalanced"] is True
    assert results["entropy"] > 0.0
    assert results["goodness_of_fit_test"] is not None
    assert results["goodness_of_fit_test"]["degrees_of_freedom"] == 1
    assert any("Class imbalance detected" in w for w in warnings)


def test_distribution_analysis_categorical_missing_values():
    """Verify sensible missing-value and empty-string handling for categorical columns."""
    df = pd.DataFrame({
        "status": ["yes", "no", None, np.nan, "   ", "", "yes", "no", "yes"]
    })
    success, results, warnings, error = execute_distribution_analysis(df, "status")
    assert success is True
    assert error is None
    assert results["variable_type"] == "categorical"
    assert results["sample_size"] == 5  # 3 'yes', 2 'no'; 2 nulls, 2 empty strings excluded
    assert results["frequencies"] == {"yes": 3, "no": 2}
    assert results["mode"] == "yes"
    assert any("Missing observations excluded" in w for w in warnings)
    assert any("Empty string observations excluded" in w for w in warnings)


def test_distribution_analysis_categorical_zero_variability():
    """Verify handling when categorical column has only one distinct value."""
    df = pd.DataFrame({"flag": ["yes"] * 10})
    success, results, warnings, error = execute_distribution_analysis(df, "flag")
    assert success is True
    assert error is None
    assert results["variable_type"] == "categorical"
    assert results["num_categories"] == 1
    assert results["mode"] == "yes"
    assert results["frequencies"] == {"yes": 10}
    assert results["proportions"] == {"yes": 1.0}
    assert any("zero variability" in w for w in warnings)


def test_distribution_analysis_insufficient_observations():
    """Verify clean rejection when observations are below minimum threshold (N < 3)."""
    # 1. Only 2 rows
    df_small = pd.DataFrame({"y": ["yes", "no"]})
    success, results, warnings, error = execute_distribution_analysis(df_small, "y")
    assert success is False
    assert error is not None
    assert error["code"] == "INSUFFICIENT_OBSERVATIONS"

    # 2. All missing values
    df_missing = pd.DataFrame({"y": [None, np.nan, None]})
    success_m, _, _, error_m = execute_distribution_analysis(df_missing, "y")
    assert success_m is False
    assert error_m["code"] == "INSUFFICIENT_OBSERVATIONS"

    # 3. All empty strings
    df_empty = pd.DataFrame({"y": ["", "  ", "   "]})
    success_e, _, _, error_e = execute_distribution_analysis(df_empty, "y")
    assert success_e is False
    assert error_e["code"] == "INSUFFICIENT_OBSERVATIONS"


def test_distribution_analysis_invalid_inputs():
    """Verify clean error responses for invalid columns or non-DataFrame inputs."""
    df = pd.DataFrame({"col_a": [1, 2, 3, 4]})

    # Non-existent column
    s1, _, _, err1 = execute_distribution_analysis(df, "non_existent")
    assert s1 is False
    assert err1["code"] == "INVALID_COLUMN"

    # Non-string column name
    s2, _, _, err2 = execute_distribution_analysis(df, 123)  # type: ignore
    assert s2 is False
    assert err2["code"] == "INVALID_COLUMN"

    # Non-DataFrame input
    s3, _, _, err3 = execute_distribution_analysis("not_a_df", "col_a")  # type: ignore
    assert s3 is False
    assert err3["code"] == "INVALID_INPUT"


def test_distribution_analysis_agent_categorical_task():
    """Verify AnalysisAgent end-to-end execution of distribution_analysis on categorical column."""
    agent = AnalysisAgent()
    df = pd.DataFrame({
        "y": ["yes", "no", "yes", "no", "no", "no", "yes", "no"]
    })
    task = InvestigationTask(
        task_id="dist_cat_1",
        task_type="distribution_analysis",
        task="Analyze target variable distribution",
        description="Inspect frequency counts and proportions of 'y'",
        method="Descriptive summary & distribution assessment",
        columns=["y"],
        column_a="y",
        reason="Target distribution assessment",
        priority="high",
    )
    result = agent.execute_task(task, df)
    assert result.status == "success"
    assert result.sample_size == 8
    assert result.results["variable_type"] == "categorical"
    assert result.results["frequencies"] == {"no": 5, "yes": 3}
    assert result.visual_evidence is not None
    assert result.visual_evidence.figure_type == "distribution_histogram"


def test_missingness_audit():
    df = pd.DataFrame({
        "a": [1, 2, np.nan, 4],
        "b": [np.nan, np.nan, 3, 4],
        "c": [1, 2, 3, 4],
    })
    success, results, warnings, error = execute_missingness_audit(df)
    assert success is True
    assert results["total_missing_cells"] == 3
    assert results["complete_cases"] == 1


# ──────────────────────────────────────────────────────────────────────────────
# 5. ANALYSIS AGENT & FAILURE ISOLATION TESTS
# ──────────────────────────────────────────────────────────────────────────────


def test_analysis_agent_plan_failure_isolation(sample_df):
    agent = AnalysisAgent()
    store_dataset("test_data", sample_df)

    tasks = [
        InvestigationTask(
            task_id="t1",
            task_type="numerical_vs_numerical",
            task="Valid Pearson Correlation",
            description="Correlation of x and y",
            method="Pearson correlation",
            columns=["x", "y"],
            column_a="x",
            column_b="y",
            reason="Numerical association",
            priority="high",
        ),
        InvestigationTask(
            task_id="t2",
            task_type="numerical_vs_numerical",
            task="Invalid Constant Variable",
            description="Correlation with constant column",
            method="Pearson correlation",
            columns=["x", "const_col"],
            column_a="x",
            column_b="const_col",
            reason="Should fail gracefully",
            priority="low",
        ),
        InvestigationTask(
            task_id="t3",
            task_type="two_groups_vs_numerical",
            task="Valid Two-Group Comparison",
            description="Compare cat1 on val_group",
            method="Welch's t-test",
            columns=["cat1", "val_group"],
            group_column="cat1",
            value_column="val_group",
            reason="Group comparison",
            priority="high",
        ),
    ]

    response = agent.execute_plan(tasks, sample_df, dataset_id="test_data")
    assert response.total_tasks == 3
    assert response.successful_tasks == 2
    assert response.failed_tasks == 1

    # Check that task 1 succeeded
    assert response.results[0].task_id == "t1"
    assert response.results[0].status == "success"
    assert "correlation_coefficient" in response.results[0].results

    # Check that task 2 failed cleanly with structured error
    assert response.results[1].task_id == "t2"
    assert response.results[1].status == "failed"
    assert response.results[1].error is not None
    assert response.results[1].error.code == "CONSTANT_VARIABLE"

    # Check that task 3 succeeded despite task 2 failing
    assert response.results[2].task_id == "t3"
    assert response.results[2].status == "success"
    assert "test_statistic" in response.results[2].results


# ──────────────────────────────────────────────────────────────────────────────
# 6. FASTAPI ENDPOINT INTEGRATION TESTS
# ──────────────────────────────────────────────────────────────────────────────


def test_api_analyze_single_task(client, sample_df):
    store_dataset("api_dataset", sample_df)

    task = {
        "task_id": "api_t1",
        "task_type": "numerical_vs_numerical",
        "task": "Test Pearson",
        "description": "Desc",
        "method": "Pearson correlation",
        "columns": ["x", "y"],
        "column_a": "x",
        "column_b": "y",
        "reason": "Testing",
        "priority": "high",
    }

    payload = {
        "dataset_id": "api_dataset",
        "task": task,
    }

    res = client.post("/api/analyze", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["task_id"] == "api_t1"
    assert data["status"] == "success"
    assert "correlation_coefficient" in data["results"]


def test_api_analyze_plan(client, sample_df):
    store_dataset("api_dataset", sample_df)

    tasks = [
        {
            "task_id": "p1",
            "task_type": "numerical_vs_numerical",
            "task": "Test Pearson",
            "description": "Desc",
            "method": "Pearson correlation",
            "columns": ["x", "y"],
            "column_a": "x",
            "column_b": "y",
            "reason": "Testing",
            "priority": "high",
        },
        {
            "task_id": "p2",
            "task_type": "categorical_vs_categorical",
            "task": "Test Chi-Square",
            "description": "Desc",
            "method": "Chi-square test of independence",
            "columns": ["cat1", "cat2"],
            "column_a": "cat1",
            "column_b": "cat2",
            "reason": "Testing cat",
            "priority": "medium",
        }
    ]

    payload = {
        "dataset_id": "api_dataset",
        "tasks": tasks,
    }

    res = client.post("/api/analyze-plan", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["total_tasks"] == 2
    assert data["successful_tasks"] == 2
    assert len(data["results"]) == 2

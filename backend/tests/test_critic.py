"""
backend/tests/test_critic.py
============================
Unit and integration test suite for DataInvest AI — Phase 4: Critic Agent & Statistical Verification.

Verifies:
1. Deterministic empirical verification without an API key.
2. Correct classification of findings (ACCEPTED, QUESTIONABLE, REJECTED, REQUIRES_REINVESTIGATION).
3. Method appropriateness and re-investigation recommendations (e.g. Pearson -> Spearman under skew).
4. Boundary and failed task auditing (zero variance, insufficient sample size).
5. Mathematical bounds verification (p-value, correlation coefficient, Cramér's V).
6. Failure isolation across multi-finding plans.
7. FastAPI endpoint contract (/api/critic).
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.agents.critic_agent import CriticAgent
from app.schemas.analysis_schema import AnalysisError, AnalysisResult, PlanAnalysisResponse
from app.schemas.critic_schema import CriticRequest, CriticResponse


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def critic():
    return CriticAgent()


# ──────────────────────────────────────────────────────────────────────────────
# 1. ACCEPTED FINDINGS TESTS
# ──────────────────────────────────────────────────────────────────────────────


def test_critic_accepts_valid_pearson(critic):
    result = AnalysisResult(
        task_id="t1",
        status="success",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        columns=["experience", "salary"],
        sample_size=60,
        results={
            "statistic": 0.8245,
            "correlation_coefficient": 0.8245,
            "p_value": 0.00001,
            "p_value_formatted": "1.0000e-05",
            "sample_size": 60,
            "degrees_of_freedom": 58,
            "p_value_below_0_05": True,
            "association_strength": "strong",
            "direction": "positive",
            "column_a": "experience",
            "column_b": "salary",
        },
        warnings=[],
    )

    review = critic.evaluate_result(result)
    assert review.task_id == "t1"
    assert review.status == "ACCEPTED"
    assert review.confidence >= 0.80
    assert review.confidence_level == "HIGH"
    assert review.requires_reinvestigation is False
    assert "SIGNIFICANT_STRONG_EFFECT" in review.reason_codes
    assert len(review.evidence) > 0


def test_critic_accepts_valid_spearman(critic):
    result = AnalysisResult(
        task_id="t2",
        status="success",
        analysis_type="numerical_vs_numerical",
        method="Spearman rank correlation",
        columns=["ranking", "score"],
        sample_size=45,
        results={
            "statistic": -0.654,
            "correlation_coefficient": -0.654,
            "p_value": 0.0002,
            "p_value_formatted": "0.0002",
            "sample_size": 45,
            "degrees_of_freedom": 43,
            "p_value_below_0_05": True,
            "association_strength": "moderate",
            "direction": "negative",
            "column_a": "ranking",
            "column_b": "score",
        },
        warnings=[],
    )

    review = critic.evaluate_result(result)
    assert review.status == "ACCEPTED"
    assert review.requires_reinvestigation is False


def test_critic_accepts_valid_two_group_comparison(critic):
    result = AnalysisResult(
        task_id="t3",
        status="success",
        analysis_type="two_groups_vs_numerical",
        method="Welch's t-test",
        columns=["department", "salary"],
        sample_size=60,
        results={
            "statistic": 3.452,
            "test_statistic": 3.452,
            "p_value": 0.0012,
            "p_value_formatted": "0.0012",
            "sample_size": 60,
            "group_column": "department",
            "value_column": "salary",
            "group_1": {"name": "Engineering", "n": 30, "mean": 85000.0, "median": 84000.0, "std": 5000.0},
            "group_2": {"name": "Marketing", "n": 30, "mean": 65000.0, "median": 64000.0, "std": 6000.0},
            "mean_difference": 20000.0,
            "p_value_below_0_05": True,
        },
        warnings=[],
    )

    review = critic.evaluate_result(result)
    assert review.status == "ACCEPTED"
    assert review.confidence_level == "HIGH"
    assert "SIGNIFICANT_GROUP_DIFFERENCE" in review.reason_codes


def test_critic_accepts_valid_chi_square(critic):
    result = AnalysisResult(
        task_id="t4",
        status="success",
        analysis_type="categorical_vs_categorical",
        method="Chi-square test of independence",
        columns=["tier", "status"],
        sample_size=100,
        results={
            "statistic": 18.45,
            "chi2_statistic": 18.45,
            "p_value": 0.0001,
            "p_value_formatted": "0.0001",
            "degrees_of_freedom": 2,
            "sample_size": 100,
            "cramer_v": 0.4295,
            "cramer_v_interpretation": "strong association",
            "p_value_below_0_05": True,
            "table_dimensions": "2x3",
            "contingency_table": {"row_labels": ["A", "B"], "col_labels": ["X", "Y", "Z"], "matrix": [[20, 20, 10], [5, 15, 30]]},
        },
        warnings=[],
    )

    review = critic.evaluate_result(result)
    assert review.status == "ACCEPTED"
    assert review.requires_reinvestigation is False
    assert "SIGNIFICANT_CATEGORICAL_ASSOCIATION" in review.reason_codes


# ──────────────────────────────────────────────────────────────────────────────
# 2. REINVESTIGATION RECOMMENDATION TESTS
# ──────────────────────────────────────────────────────────────────────────────


def test_critic_recommends_spearman_when_pearson_has_skew_or_outliers(critic):
    # Pearson executed on data with high skew and outliers in profile
    result = AnalysisResult(
        task_id="t5",
        status="success",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        columns=["skewed_a", "skewed_b"],
        sample_size=50,
        results={
            "statistic": 0.45,
            "correlation_coefficient": 0.45,
            "p_value": 0.01,
            "p_value_formatted": "0.0100",
            "sample_size": 50,
            "column_a": "skewed_a",
            "column_b": "skewed_b",
        },
        warnings=[],
    )

    profile_with_skew = {
        "numerical_stats": {
            "skewed_a": {"skewness": 2.45},
            "skewed_b": {"skewness": 0.30},
        },
        "outliers": {
            "skewed_a": {"n_potential_outliers": 4},
        },
    }

    review = critic.evaluate_result(result, profile=profile_with_skew)
    assert review.status == "REQUIRES_REINVESTIGATION"
    assert review.requires_reinvestigation is True
    assert review.recommended_method == "Spearman rank correlation"
    assert "RECOMMEND_SPEARMAN_RANK" in review.reason_codes


def test_critic_recommends_mann_whitney_when_welch_has_high_skew_and_small_sample(critic):
    result = AnalysisResult(
        task_id="t6",
        status="success",
        analysis_type="two_groups_vs_numerical",
        method="Welch's t-test",
        columns=["segment", "spend"],
        sample_size=20,
        results={
            "statistic": 2.1,
            "test_statistic": 2.1,
            "p_value": 0.04,
            "p_value_formatted": "0.0400",
            "sample_size": 20,
            "value_column": "spend",
            "group_1": {"name": "S1", "n": 10, "mean": 100.0, "median": 80.0, "std": 50.0},
            "group_2": {"name": "S2", "n": 10, "mean": 60.0, "median": 55.0, "std": 20.0},
            "mean_difference": 40.0,
            "p_value_below_0_05": True,
        },
        warnings=[],
    )

    profile = {
        "numerical_stats": {
            "spend": {"skewness": 2.8},
        },
    }

    review = critic.evaluate_result(result, profile=profile)
    assert review.status == "REQUIRES_REINVESTIGATION"
    assert review.requires_reinvestigation is True
    assert review.recommended_method == "Mann-Whitney U test"


# ──────────────────────────────────────────────────────────────────────────────
# 3. QUESTIONABLE FINDINGS TESTS
# ──────────────────────────────────────────────────────────────────────────────


def test_critic_flags_small_sample_size(critic):
    result = AnalysisResult(
        task_id="t7",
        status="success",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        columns=["x", "y"],
        sample_size=8,
        results={
            "statistic": 0.70,
            "correlation_coefficient": 0.70,
            "p_value": 0.05,
            "p_value_formatted": "0.0500",
            "sample_size": 8,
            "column_a": "x",
            "column_b": "y",
        },
        warnings=[],
    )

    review = critic.evaluate_result(result)
    assert review.status == "QUESTIONABLE"
    assert "SMALL_SAMPLE_POWER_RISK" in review.reason_codes
    assert review.confidence_level in ("MEDIUM", "LOW")


def test_critic_flags_sparse_chi_square_frequencies(critic):
    result = AnalysisResult(
        task_id="t8",
        status="success",
        analysis_type="categorical_vs_categorical",
        method="Chi-square test of independence",
        columns=["rare_a", "rare_b"],
        sample_size=30,
        results={
            "statistic": 4.5,
            "chi2_statistic": 4.5,
            "p_value": 0.04,
            "p_value_formatted": "0.0400",
            "degrees_of_freedom": 1,
            "sample_size": 30,
            "cramer_v": 0.38,
            "cramer_v_interpretation": "moderate association",
            "p_value_below_0_05": True,
        },
        warnings=["33.3% of cells have expected count < 5; interpret chi-square with caution."],
    )

    review = critic.evaluate_result(result)
    assert review.status == "QUESTIONABLE"
    assert "SPARSE_CONTINGENCY_CELLS" in review.reason_codes


# ──────────────────────────────────────────────────────────────────────────────
# 4. REJECTED FINDINGS TESTS (Boundaries, Failures & Mathematical Violations)
# ------------------------------------------------------------------------------


def test_critic_rejects_constant_variable_failure(critic):
    result = AnalysisResult(
        task_id="t9",
        status="failed",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        columns=["var_a", "const_col"],
        sample_size=0,
        results={},
        warnings=[],
        error=AnalysisError(
            code="CONSTANT_VARIABLE",
            message="Statistical analysis cannot be performed because variable 'const_col' is constant.",
        ),
    )

    review = critic.evaluate_result(result)
    assert review.status == "REJECTED"
    assert "BOUNDARY_CONSTANT_VARIABLE" in review.reason_codes
    assert review.requires_reinvestigation is False
    assert "constant" in review.summary.lower()


def test_critic_rejects_out_of_bounds_p_value(critic):
    result = AnalysisResult(
        task_id="t10",
        status="success",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        columns=["a", "b"],
        sample_size=50,
        results={
            "statistic": 0.5,
            "correlation_coefficient": 0.5,
            "p_value": 1.45,  # Out of bounds (>1)
            "sample_size": 50,
        },
    )

    review = critic.evaluate_result(result)
    assert review.status == "REJECTED"
    assert "INVALID_P_VALUE_BOUNDS" in review.reason_codes


def test_critic_rejects_out_of_bounds_correlation(critic):
    result = AnalysisResult(
        task_id="t11",
        status="success",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        columns=["a", "b"],
        sample_size=50,
        results={
            "statistic": 2.5,  # Out of bounds (>1)
            "correlation_coefficient": 2.5,
            "p_value": 0.01,
            "sample_size": 50,
        },
    )

    review = critic.evaluate_result(result)
    assert review.status == "REJECTED"
    assert "INVALID_CORRELATION_BOUNDS" in review.reason_codes


# ──────────────────────────────────────────────────────────────────────────────
# 5. PLAN EVALUATION & FAILURE ISOLATION TESTS
# ──────────────────────────────────────────────────────────────────────────────


def test_critic_evaluates_entire_plan_with_isolation(critic):
    results = [
        # 1. Valid Pearson
        AnalysisResult(
            task_id="p1",
            status="success",
            analysis_type="numerical_vs_numerical",
            method="Pearson correlation",
            columns=["x", "y"],
            sample_size=50,
            results={"correlation_coefficient": 0.85, "p_value": 0.0001, "sample_size": 50, "p_value_below_0_05": True},
        ),
        # 2. Failed Constant Variable
        AnalysisResult(
            task_id="p2",
            status="failed",
            analysis_type="numerical_vs_numerical",
            method="Pearson correlation",
            columns=["x", "const"],
            sample_size=0,
            results={},
            error=AnalysisError(code="CONSTANT_VARIABLE", message="Variable 'const' is constant."),
        ),
        # 3. Small sample size
        AnalysisResult(
            task_id="p3",
            status="success",
            analysis_type="numerical_vs_numerical",
            method="Spearman rank correlation",
            columns=["u", "v"],
            sample_size=6,
            results={"correlation_coefficient": 0.4, "p_value": 0.20, "sample_size": 6, "p_value_below_0_05": False},
        ),
    ]

    response = critic.evaluate_plan(results, dataset_id="test_dataset.csv")
    assert isinstance(response, CriticResponse)
    assert response.dataset_id == "test_dataset.csv"
    assert response.summary.total_results == 3
    assert response.summary.accepted == 1
    assert response.summary.rejected == 1
    assert response.summary.questionable == 1
    assert len(response.reviews) == 3

    # Check reviews mapped properly
    assert response.reviews[0].task_id == "p1"
    assert response.reviews[0].status == "ACCEPTED"
    assert response.reviews[1].task_id == "p2"
    assert response.reviews[1].status == "REJECTED"
    assert response.reviews[2].task_id == "p3"
    assert response.reviews[2].status == "QUESTIONABLE"


# ──────────────────────────────────────────────────────────────────────────────
# 6. FASTAPI ENDPOINT INTEGRATION TESTS
# ──────────────────────────────────────────────────────────────────────────────


def test_api_critic_endpoint(client):
    plan_response = {
        "dataset_id": "sample.csv",
        "total_tasks": 2,
        "successful_tasks": 2,
        "failed_tasks": 0,
        "results": [
            {
                "task_id": "api_t1",
                "status": "success",
                "analysis_type": "numerical_vs_numerical",
                "method": "Pearson correlation",
                "columns": ["age", "salary"],
                "sample_size": 60,
                "results": {
                    "correlation_coefficient": 0.75,
                    "p_value": 0.0001,
                    "sample_size": 60,
                    "p_value_below_0_05": True,
                    "column_a": "age",
                    "column_b": "salary",
                },
                "warnings": [],
            },
            {
                "task_id": "api_t2",
                "status": "failed",
                "analysis_type": "numerical_vs_numerical",
                "method": "Pearson correlation",
                "columns": ["age", "zero_col"],
                "sample_size": 0,
                "results": {},
                "warnings": [],
                "error": {
                    "code": "CONSTANT_VARIABLE",
                    "message": "Column 'zero_col' is constant.",
                },
            },
        ],
    }

    payload = {
        "analysis_response": plan_response,
        "dataset_id": "sample.csv",
    }

    res = client.post("/api/critic", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["dataset_id"] == "sample.csv"
    assert data["summary"]["total_results"] == 2
    assert data["summary"]["accepted"] == 1
    assert data["summary"]["rejected"] == 1
    assert len(data["reviews"]) == 2
    assert data["reviews"][0]["status"] == "ACCEPTED"
    assert data["reviews"][1]["status"] == "REJECTED"


# ──────────────────────────────────────────────────────────────────────────────
# 7. PHASE 3 & 4 EVIDENCE, CI, EFFECT SIZE, OUTLIER, AND P-HACKING TESTS
# ──────────────────────────────────────────────────────────────────────────────


def test_critic_rejects_missing_evidence(critic):
    """Missing or empty statistical evidence must be rejected as INSUFFICIENT_EVIDENCE, not assumed reliable."""
    result = AnalysisResult(
        task_id="t_missing",
        status="success",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        columns=["x", "y"],
        sample_size=50,
        results={},  # Missing p_value, statistic, etc.
        warnings=[],
    )

    review = critic.evaluate_result(result)
    assert review.status == "REJECTED"
    assert "INSUFFICIENT_EVIDENCE" in review.reason_codes
    assert review.confidence == 1.0
    assert review.confidence_level == "HIGH"
    assert any("missing" in ev.lower() for ev in review.evidence)


def test_critic_records_ci_and_effect_size(critic):
    """Verified findings must record 95% confidence intervals and effect sizes from Python statistical calculations."""
    result = AnalysisResult(
        task_id="t_ci",
        status="success",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        columns=["experience", "salary"],
        sample_size=50,
        results={
            "correlation_coefficient": 0.72,
            "statistic": 0.72,
            "p_value": 0.0001,
            "sample_size": 50,
            "degrees_of_freedom": 48,
            "p_value_below_0_05": True,
            "ci_95_low": 0.55,
            "ci_95_high": 0.83,
            "confidence_interval": [0.55, 0.83],
            "r_squared": 0.5184,
            "effect_size": {"type": "r_squared", "value": 0.5184},
        },
        warnings=[],
    )

    review = critic.evaluate_result(result)
    assert review.status == "ACCEPTED"
    assert any("95% CI" in ev for ev in review.evidence)
    assert any("Effect size" in ev for ev in review.evidence)


def test_critic_evaluates_outlier_sensitivity_task(critic):
    """Outlier sensitivity analyses are evaluated for substantial vs minimal outlier influence."""
    # Substantial influence case
    result_high = AnalysisResult(
        task_id="t_out_high",
        status="success",
        analysis_type="outlier_sensitivity",
        method="IQR-based sensitivity analysis",
        columns=["salary"],
        sample_size=60,
        results={
            "delta_mean": 12500.0,
            "relative_change_pct": 22.5,
            "outlier_count": 4,
            "sample_size": 60,
        },
        warnings=[],
    )
    rev_high = critic.evaluate_result(result_high)
    assert "SUBSTANTIAL_OUTLIER_INFLUENCE" in rev_high.reason_codes

    # Minimal influence case
    result_low = AnalysisResult(
        task_id="t_out_low",
        status="success",
        analysis_type="outlier_sensitivity",
        method="IQR-based sensitivity analysis",
        columns=["salary"],
        sample_size=60,
        results={
            "delta_mean": 200.0,
            "relative_change_pct": 1.2,
            "outlier_count": 0,
            "sample_size": 60,
        },
        warnings=[],
    )
    rev_low = critic.evaluate_result(result_low)
    assert "MINIMAL_OUTLIER_INFLUENCE" in rev_low.reason_codes


def test_critic_prevents_p_hacking(critic):
    """Non-significant finding with valid assumptions should NOT be recommended for alternative test to fish for significance."""
    result = AnalysisResult(
        task_id="t_nohack",
        status="success",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        columns=["x", "y"],
        sample_size=80,
        results={
            "correlation_coefficient": 0.12,
            "statistic": 0.12,
            "p_value": 0.28,
            "sample_size": 80,
            "p_value_below_0_05": False,
        },
        warnings=[],  # No skew, no outlier warnings
    )

    review = critic.evaluate_result(result)
    assert review.requires_reinvestigation is False
    assert "P_HACKING_PREVENTED" in review.reason_codes


def test_critic_recommends_kruskal_wallis_for_skewed_anova(critic):
    """One-way ANOVA with severe skewness or unequal group sizes must recommend Kruskal-Wallis."""
    result = AnalysisResult(
        task_id="t_anova_skew",
        status="success",
        analysis_type="multiple_groups_vs_numerical",
        method="One-way ANOVA",
        columns=["dept", "salary"],
        sample_size=75,
        results={
            "F_statistic": 4.12,
            "p_value": 0.02,
            "sample_size": 75,
            "p_value_below_0_05": True,
        },
        warnings=["Severe positive skewness detected in group 'Sales' (skew = 2.45)."],
    )

    review = critic.evaluate_result(result)
    assert review.requires_reinvestigation is True
    assert "RECOMMEND_KRUSKAL_WALLIS" in review.reason_codes
    assert review.recommended_method == "Kruskal-Wallis test"


def test_critic_audits_visual_diagnostics(critic):
    """Programmatic visual diagnostics with detected bivariate outliers affect Critic status."""
    from app.schemas.visualization_schema import VisualEvidence, VisualDiagnostics

    viz = VisualEvidence(
        graph_id="g1",
        task_id="t_viz",
        figure_type="scatter",
        title="Visual Test",
        x_label="x",
        y_label="y",
        columns=["x", "y"],
        method="Pearson correlation",
        visual_diagnostics=VisualDiagnostics(
            outlier_points_count=5,
            linearity_indicator="Non-linear / contaminated",
            visual_notes=["Detected 5 visual outlier points."],
            vision_model_used=False,
            visual_inspection_claimed=False,
        ),
    )

    result = AnalysisResult(
        task_id="t_viz",
        status="success",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        columns=["a", "b"],
        sample_size=60,
        results={
            "correlation_coefficient": 0.70,
            "p_value": 0.001,
            "sample_size": 60,
            "p_value_below_0_05": True,
        },
        visual_evidence=viz,
    )

    review = critic.evaluate_result(result)
    assert review.status == "REQUIRES_REINVESTIGATION"
    assert review.requires_reinvestigation is True
    assert review.recommended_method == "Spearman rank correlation"
    assert "VISUAL_OUTLIERS_REINVESTIGATE" in review.reason_codes
    assert "VISUAL_OUTLIERS_DETECTED" in review.reason_codes
    assert any("visual scatter diagnostics" in ev.lower() or "visual diagnostics" in ev.lower() for ev in review.evidence)


def test_critic_spearman_robust_to_visual_outliers(critic):
    """Spearman rank correlation is robust to monotonic outliers and is ACCEPTED despite visual outliers."""
    from app.schemas.visualization_schema import VisualEvidence, VisualDiagnostics

    viz = VisualEvidence(
        graph_id="g_spearman",
        task_id="t_spearman_viz",
        figure_type="scatter",
        title="Spearman Visual Test",
        x_label="x",
        y_label="y",
        columns=["x", "y"],
        method="Spearman rank correlation",
        visual_diagnostics=VisualDiagnostics(
            outlier_points_count=4,
            visual_notes=["Detected 4 visual outlier points."],
            vision_model_used=False,
            visual_inspection_claimed=False,
        ),
    )

    result = AnalysisResult(
        task_id="t_spearman_viz",
        status="success",
        analysis_type="numerical_vs_numerical",
        method="Spearman rank correlation",
        columns=["x", "y"],
        sample_size=55,
        results={
            "correlation_coefficient": 0.76,
            "p_value": 0.0001,
            "sample_size": 55,
            "p_value_below_0_05": True,
        },
        visual_evidence=viz,
    )

    review = critic.evaluate_result(result)
    assert review.status == "ACCEPTED"
    assert review.requires_reinvestigation is False
    assert "METHOD_VALIDATED" in review.reason_codes
    assert "VISUAL_OUTLIERS_DETECTED" in review.reason_codes


def test_critic_welch_small_group_visual_outliers_recommends_mann_whitney(critic):
    """Welch t-test with small groups and boxplot outliers violates normality and recommends Mann-Whitney."""
    from app.schemas.visualization_schema import VisualEvidence, VisualDiagnostics

    viz = VisualEvidence(
        graph_id="g_welch",
        task_id="t_welch_viz",
        figure_type="box",
        title="Welch Boxplot Test",
        x_label="group",
        y_label="value",
        columns=["group", "value"],
        method="Welch's t-test",
        visual_diagnostics=VisualDiagnostics(
            outlier_points_count=3,
            iqr_overlap_ratio=0.15,
            visual_notes=["Detected 3 group boxplot outlier observations."],
            vision_model_used=False,
            visual_inspection_claimed=False,
        ),
    )

    result = AnalysisResult(
        task_id="t_welch_viz",
        status="success",
        analysis_type="two_groups_vs_numerical",
        method="Welch's t-test",
        columns=["group", "value"],
        sample_size=30,
        results={
            "test_statistic": 2.45,
            "p_value": 0.025,
            "p_value_below_0_05": True,
            "mean_difference": 4.5,
            "group_1": {"name": "A", "n": 15, "mean": 12.5},
            "group_2": {"name": "B", "n": 15, "mean": 8.0},
        },
        visual_evidence=viz,
    )

    review = critic.evaluate_result(result)
    assert review.status == "REQUIRES_REINVESTIGATION"
    assert review.requires_reinvestigation is True
    assert review.recommended_method == "Mann-Whitney U test"
    assert "RECOMMEND_MANN_WHITNEY" in review.reason_codes
    assert "VISUAL_OUTLIERS_REINVESTIGATE" in review.reason_codes


def test_critic_chi_square_visual_sparse_cells_marks_questionable(critic):
    """Chi-square contingency test with visual sparse cells (<5) is marked QUESTIONABLE."""
    from app.schemas.visualization_schema import VisualEvidence, VisualDiagnostics

    viz = VisualEvidence(
        graph_id="g_chi2",
        task_id="t_chi2_viz",
        figure_type="grouped_bar",
        title="Contingency Bar Test",
        x_label="cat_a",
        y_label="Count",
        columns=["cat_a", "cat_b"],
        method="Chi-square test of independence",
        visual_diagnostics=VisualDiagnostics(
            sparse_categories_count=2,
            visual_notes=["Visual diagnostics flagged 2 sparse contingency cells with count < 5."],
            vision_model_used=False,
            visual_inspection_claimed=False,
        ),
    )

    result = AnalysisResult(
        task_id="t_chi2_viz",
        status="success",
        analysis_type="categorical_vs_categorical",
        method="Chi-square test of independence",
        columns=["cat_a", "cat_b"],
        sample_size=40,
        results={
            "chi2_statistic": 6.8,
            "p_value": 0.03,
            "p_value_below_0_05": True,
            "cramer_v": 0.32,
            "degrees_of_freedom": 1,
            "table_dimensions": "2x2",
            "contingency_table": {
                "matrix": [[15, 2], [3, 20]],
            },
            "expected_cell_counts": {
                "min_expected": 3.61,
                "cells_below_5_count": 2,
                "cells_below_1_count": 0,
                "cells_below_5_pct": 50.0,
                "cochran_criterion_satisfied": False,
            },
        },
        visual_evidence=viz,
    )

    review = critic.evaluate_result(result)
    assert review.status == "QUESTIONABLE"
    assert "SPARSE_CONTINGENCY_CELLS" in review.reason_codes
    assert any("sparse" in concern.lower() or "cochran" in concern.lower() for concern in review.assumption_concerns)


def test_critic_chi_square_expected_below_one_marks_questionable(critic):
    """Cochran Condition 1: If any expected cell frequency is < 1.0, the finding is marked QUESTIONABLE."""
    result = AnalysisResult(
        task_id="t_chi2_below_one",
        status="success",
        analysis_type="categorical_vs_categorical",
        method="Chi-square test of independence",
        columns=["cat_a", "cat_b"],
        sample_size=50,
        results={
            "chi2_statistic": 4.2,
            "p_value": 0.04,
            "p_value_below_0_05": True,
            "cramer_v": 0.29,
            "degrees_of_freedom": 2,
            "table_dimensions": "2x3",
            "expected_cell_counts": {
                "min_expected": 0.75,
                "cells_below_5_count": 1,
                "cells_below_1_count": 1,
                "cells_below_5_pct": 16.7,  # <= 20%, but min_expected < 1.0 violates Condition 1
                "cochran_criterion_satisfied": False,
            },
        },
    )

    review = critic.evaluate_result(result)
    assert review.status == "QUESTIONABLE"
    assert "EXPECTED_COUNT_BELOW_ONE" in review.reason_codes
    assert "SPARSE_CONTINGENCY_CELLS" in review.reason_codes
    assert any("expected cell frequency < 1.0" in concern.lower() for concern in review.assumption_concerns)


def test_critic_pearson_mild_tail_outliers_remains_accepted(critic):
    """Pearson correlation with 1-2 isolated residual outliers (|z| > 3.0) represents normal tail variance and remains ACCEPTED."""
    from app.schemas.visualization_schema import VisualEvidence, VisualDiagnostics

    viz = VisualEvidence(
        graph_id="g_mild_outlier",
        task_id="t_mild_pearson",
        figure_type="scatter",
        title="Pearson Mild Outlier Test",
        x_label="x",
        y_label="y",
        columns=["x", "y"],
        method="Pearson correlation",
        visual_diagnostics=VisualDiagnostics(
            outlier_points_count=1,
            linearity_indicator="Linear",
            visual_notes=["1 isolated point (|z| = 3.05) detected."],
            vision_model_used=False,
            visual_inspection_claimed=False,
        ),
    )

    result = AnalysisResult(
        task_id="t_mild_pearson",
        status="success",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        columns=["x", "y"],
        sample_size=60,
        results={
            "correlation_coefficient": 0.82,
            "p_value": 0.0001,
            "sample_size": 60,
            "p_value_below_0_05": True,
            "column_a": "x",
            "column_b": "y",
        },
        visual_evidence=viz,
    )

    review = critic.evaluate_result(result)
    assert review.status == "ACCEPTED"
    assert review.requires_reinvestigation is False
    assert "METHOD_VALIDATED" in review.reason_codes
    assert "VISUAL_OUTLIERS_DETECTED" in review.reason_codes
    assert any("acceptable bounds" in ev.lower() for ev in review.evidence)


def test_critic_welch_single_mild_outlier_remains_accepted(critic):
    """Welch's t-test with a single mild boxplot outlier in moderate groups remains ACCEPTED."""
    from app.schemas.visualization_schema import VisualEvidence, VisualDiagnostics

    viz = VisualEvidence(
        graph_id="g_welch_mild",
        task_id="t_welch_mild",
        figure_type="box",
        title="Welch Mild Outlier Boxplot",
        x_label="group",
        y_label="value",
        columns=["group", "value"],
        method="Welch's t-test",
        visual_diagnostics=VisualDiagnostics(
            outlier_points_count=1,
            iqr_overlap_ratio=0.10,
            visual_notes=["1 observation outside 1.5*IQR fence."],
            vision_model_used=False,
            visual_inspection_claimed=False,
        ),
    )

    result = AnalysisResult(
        task_id="t_welch_mild",
        status="success",
        analysis_type="two_groups_vs_numerical",
        method="Welch's t-test",
        columns=["group", "value"],
        sample_size=40,
        results={
            "test_statistic": 3.12,
            "p_value": 0.004,
            "p_value_below_0_05": True,
            "mean_difference": 5.2,
            "group_1": {"name": "Treatment", "n": 20, "mean": 15.0},
            "group_2": {"name": "Control", "n": 20, "mean": 9.8},
        },
        visual_evidence=viz,
    )

    review = critic.evaluate_result(result)
    assert review.status == "ACCEPTED"
    assert review.requires_reinvestigation is False
    assert "GROUP_COMPARISON_VALIDATED" in review.reason_codes
    assert "VISUAL_OUTLIERS_DETECTED" in review.reason_codes
    assert any("welch's t-test remains robust" in ev.lower() for ev in review.evidence)


def test_critic_chi_square_mild_single_cell_below_five_remains_accepted(critic):
    """Chi-square with only 1 low-count cell in a 3x3 table (11.1% <= 20% Cochran bound) remains ACCEPTED."""
    from app.schemas.visualization_schema import VisualEvidence, VisualDiagnostics

    viz = VisualEvidence(
        graph_id="g_chi2_cochran",
        task_id="t_chi2_cochran",
        figure_type="grouped_bar",
        title="3x3 Contingency Table",
        x_label="education",
        y_label="income_bracket",
        columns=["education", "income_bracket"],
        method="Chi-square test of independence",
        visual_diagnostics=VisualDiagnostics(
            sparse_categories_count=1,  # 1 cell out of 9 = 11.1%
            visual_notes=["1 contingency cell with count = 4."],
            vision_model_used=False,
            visual_inspection_claimed=False,
        ),
    )

    result = AnalysisResult(
        task_id="t_chi2_cochran",
        status="success",
        analysis_type="categorical_vs_categorical",
        method="Chi-square test of independence",
        columns=["education", "income_bracket"],
        sample_size=120,
        results={
            "chi2_statistic": 18.4,
            "p_value": 0.002,
            "p_value_below_0_05": True,
            "cramer_v": 0.28,
            "degrees_of_freedom": 4,
            "table_dimensions": "3x3",
            "contingency_table": {
                "matrix": [[15, 15, 10], [15, 15, 10], [10, 26, 4]],
            },
            "expected_cell_counts": {
                "min_expected": 8.0,
                "cells_below_5_count": 0,
                "cells_below_1_count": 0,
                "cells_below_5_pct": 0.0,
                "cochran_criterion_satisfied": True,
            },
        },
        visual_evidence=viz,
    )

    review = critic.evaluate_result(result)
    assert review.status == "ACCEPTED"
    assert review.requires_reinvestigation is False
    assert "SIGNIFICANT_CATEGORICAL_ASSOCIATION" in review.reason_codes
    assert any("satisfy cochran's rule" in ev.lower() or "satisfies cochran's rule" in ev.lower() for ev in review.evidence)


def test_critic_chi_square_reconstructs_expected_from_contingency_matrix(critic):
    """When expected_cell_counts is omitted, Critic reconstructs expected counts from contingency_table['matrix']."""
    # 2x2 table with counts [[35, 1], [30, 4]]:
    # Row totals: R1 = 36, R2 = 34. Col totals: C1 = 65, C2 = 5. Grand total N = 70.
    # Expected cell [1, 1] = (34 * 5) / 70 = 2.43 (< 5.0)
    # Expected cell [0, 1] = (36 * 5) / 70 = 2.57 (< 5.0)
    # Cells below 5 = 2 of 4 = 50.0% (> 20% limit)
    result = AnalysisResult(
        task_id="t_chi2_reconstruct",
        status="success",
        analysis_type="categorical_vs_categorical",
        method="Chi-square test of independence",
        columns=["feature_a", "feature_b"],
        sample_size=70,
        results={
            "chi2_statistic": 1.25,
            "p_value": 0.26,
            "cramer_v": 0.13,
            "degrees_of_freedom": 1,
            "table_dimensions": "2x2",
            "contingency_table": {
                "matrix": [[35, 1], [30, 4]],
            },
            # Note: expected_cell_counts intentionally omitted to test Critic reconstruction
        },
    )

    review = critic.evaluate_result(result)
    assert review.status == "QUESTIONABLE"
    assert "SPARSE_CONTINGENCY_CELLS" in review.reason_codes
    assert any("cochran's contingency condition violated" in concern.lower() for concern in review.assumption_concerns)


def test_critic_distribution_analysis_uses_distribution_rule_not_outliers(critic):
    """Distribution analysis should route to Branch 3F and evaluate skewness/normality,
    not Branch 3E (outlier sensitivity) which would default missing outlier counts to 0."""
    result = AnalysisResult(
        task_id="t_dist_12",
        status="success",
        analysis_type="distribution_analysis",
        method="Distribution analysis",
        columns=["absences"],
        sample_size=395,
        results={
            "mean": 5.7089,
            "median": 4.0,
            "std": 8.0031,
            "variance": 64.0495,
            "skewness": 3.6657,
            "kurtosis": 22.9501,
            "q1": 0.0,
            "q3": 8.0,
            "iqr": 8.0,
            "symmetry": "highly skewed (right-tailed)",
            "normality_test": {
                "method": "Shapiro-Wilk test (first 5000 rows)",
                "statistic": 0.6698,
                "p_value": 0.0,
            },
        },
    )

    review = critic.evaluate_result(result)
    # Must NOT use outlier sensitivity rule or reason codes
    assert "MINIMAL_OUTLIER_INFLUENCE" not in review.reason_codes
    assert "SUBSTANTIAL_OUTLIER_INFLUENCE" not in review.reason_codes
    assert not any("0 IQR outlier observations" in ev for ev in review.evidence)
    assert not any("Mean shift upon trimming" in ev for ev in review.evidence)

    # Must use distribution rule and reason codes
    assert "NON_NORMAL_DISTRIBUTION" in review.reason_codes
    assert any("Distribution moments: skewness=3.6657" in ev for ev in review.evidence)
    assert any("Normality test" in ev for ev in review.evidence)
    assert "highly skewed (right-tailed)" in review.effect_size_assessment


def test_critic_outlier_sensitivity_preserves_task14_behavior(critic):
    """Outlier sensitivity should route to Branch 3E and identify substantial influence."""
    result = AnalysisResult(
        task_id="t_outlier_14",
        status="success",
        analysis_type="outlier_sensitivity",
        method="Outlier Sensitivity (IQR Trimming)",
        columns=["absences"],
        sample_size=395,
        results={
            "outlier_count": 15,
            "outlier_percentage": 3.8,
            "outlier_pct": 3.7975,
            "baseline_mean": 5.7089,
            "trimmed_mean": 4.6026,
            "mean_shift": -1.1062,
            "mean_shift_pct": -19.38,
            "substantial_influence": True,
        },
    )

    review = critic.evaluate_result(result)
    assert review.status == "QUESTIONABLE"
    assert "SUBSTANTIAL_OUTLIER_INFLUENCE" in review.reason_codes
    assert review.confidence == pytest.approx(0.85, abs=0.01)
    assert any("Identified 15 IQR outlier observations (3.8% of N=395)" in ev for ev in review.evidence)
    assert any("Mean shift upon trimming: -1.1062 (-19.38% change from baseline)" in ev for ev in review.evidence)


def test_critic_evaluates_categorical_distribution_imbalanced(critic):
    """Verify Critic Agent evaluates categorical distributions and detects class imbalance."""
    result = AnalysisResult(
        task_id="t_cat_dist_imbal",
        status="success",
        analysis_type="distribution_analysis",
        method="Descriptive summary & distribution assessment",
        columns=["y"],
        sample_size=1000,
        results={
            "column": "y",
            "variable_type": "categorical",
            "sample_size": 1000,
            "num_categories": 2,
            "mode": "no",
            "mode_frequency": 883,
            "mode_percentage": 88.3,
            "frequencies": {"no": 883, "yes": 117},
            "proportions": {"no": 0.883, "yes": 0.117},
            "percentages": {"no": 88.3, "yes": 11.7},
            "is_imbalanced": True,
            "goodness_of_fit_test": {
                "method": "Chi-square goodness-of-fit test (uniform null)",
                "statistic": 586.76,
                "p_value": 0.00001,
            },
        },
        warnings=[],
    )

    review = critic.evaluate_result(result)
    assert review.task_id == "t_cat_dist_imbal"
    assert "CLASS_IMBALANCE_DETECTED" in review.reason_codes
    assert "NON_NORMAL_DISTRIBUTION" not in review.reason_codes
    assert not any("Distribution moments: skewness=0.0000" in ev for ev in review.evidence)
    assert any("Categorical frequency distribution: 2 distinct categories" in ev for ev in review.evidence)
    assert any("mode 'no' (883 occurrences, 88.3%)" in ev for ev in review.evidence)
    assert any("class imbalance" in c.lower() for c in review.data_quality_concerns)


def test_critic_evaluates_categorical_distribution_balanced(critic):
    """Verify Critic Agent evaluates balanced categorical distributions without flagging imbalance."""
    result = AnalysisResult(
        task_id="t_cat_dist_bal",
        status="success",
        analysis_type="distribution_analysis",
        method="Descriptive summary & distribution assessment",
        columns=["variant"],
        sample_size=200,
        results={
            "column": "variant",
            "variable_type": "categorical",
            "sample_size": 200,
            "num_categories": 2,
            "mode": "control",
            "mode_frequency": 105,
            "mode_percentage": 52.5,
            "frequencies": {"control": 105, "treatment": 95},
            "proportions": {"control": 0.525, "treatment": 0.475},
            "is_imbalanced": False,
        },
        warnings=[],
    )

    review = critic.evaluate_result(result)
    assert review.status == "ACCEPTED"
    assert "BALANCED_CATEGORICAL_DISTRIBUTION" in review.reason_codes
    assert "CLASS_IMBALANCE_DETECTED" not in review.reason_codes


def test_critic_evaluates_categorical_distribution_constant(critic):
    """Verify Critic Agent flags zero variability when categorical column has only one category."""
    result = AnalysisResult(
        task_id="t_cat_dist_const",
        status="success",
        analysis_type="distribution_analysis",
        method="Descriptive summary & distribution assessment",
        columns=["status"],
        sample_size=100,
        results={
            "column": "status",
            "variable_type": "categorical",
            "sample_size": 100,
            "num_categories": 1,
            "mode": "active",
            "mode_frequency": 100,
            "mode_percentage": 100.0,
            "frequencies": {"active": 100},
            "proportions": {"active": 1.0},
            "is_imbalanced": False,
        },
        warnings=[],
    )

    review = critic.evaluate_result(result)
    assert review.status == "QUESTIONABLE"
    assert "ZERO_VARIANCE_CATEGORICAL" in review.reason_codes



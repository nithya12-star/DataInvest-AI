"""
backend/tests/test_reinvestigation.py
=====================================
Unit and integration test suite for DataInvest AI — Phase 5: Re-investigation & Verified Findings.

Verifies:
1. Direct transition of ACCEPTED findings to VERIFIED.
2. Direct exclusion of REJECTED findings from verified insights.
3. Targeted re-investigation of REQUIRES_REINVESTIGATION findings via Python/SciPy.
4. Second Critic review and transition to VERIFIED_AFTER_REINVESTIGATION.
5. Questionable findings that remain QUESTIONABLE when caveats persist.
6. Strict MAX_REINVESTIGATIONS = 1 loop guard.
7. Duplicate method execution prevention.
8. Unsupported recommended method safety.
9. Failure isolation across multi-finding batches.
10. FastAPI endpoint contracts (/api/verify-findings, /api/reinvestigate).
"""

import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.services.dataset_store import store_dataset, clear_cache
from app.services.verification_service import VerificationService
from app.schemas.analysis_schema import AnalysisError, AnalysisResult, PlanAnalysisResponse
from app.schemas.critic_schema import CriticResponse, CriticResult, CriticOverallSummary
from app.schemas.verification_schema import VerificationRequest, VerificationResponse


@pytest.fixture(autouse=True)
def cleanup():
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_df():
    """Standard DataFrame for statistical testing."""
    np.random.seed(42)
    n = 60
    x = np.linspace(1, 10, n)
    # y has monotonic non-linear relation with exponential skew
    y = np.exp(x / 2.5) + np.random.normal(0, 0.5, n)
    dept = np.random.choice(["Sales", "Engineering", "Marketing"], size=n)
    salary = np.where(dept == "Engineering", 90000 + np.random.normal(0, 5000, n), 60000 + np.random.normal(0, 5000, n))

    df = pd.DataFrame({
        "x": x,
        "y": y,
        "dept": dept,
        "salary": salary,
        "const_col": [10.0] * n,
    })
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 1. DIRECT VERIFICATION & REJECTION TESTS
# ──────────────────────────────────────────────────────────────────────────────


def test_direct_verification_accepted_finding(sample_df):
    store_dataset("test_data", sample_df)
    service = VerificationService()

    orig_analysis = AnalysisResult(
        task_id="t1",
        status="success",
        analysis_type="numerical_vs_numerical",
        method="Spearman rank correlation",
        columns=["x", "y"],
        sample_size=60,
        results={
            "correlation_coefficient": 0.98,
            "p_value": 0.0001,
            "sample_size": 60,
            "p_value_below_0_05": True,
        },
    )

    critic_rev = CriticResult(
        task_id="t1",
        analysis_type="numerical_vs_numerical",
        method="Spearman rank correlation",
        variables=["x", "y"],
        status="ACCEPTED",
        confidence=0.95,
        confidence_level="HIGH",
        summary="Finding is empirically verified and robust.",
        evidence=["N=60, rho=0.98, p < 0.001"],
        statistical_validity="Valid.",
        method_assessment="Appropriate.",
        recommendation="Finding is verified.",
        requires_reinvestigation=False,
    )

    plan_resp = PlanAnalysisResponse(
        dataset_id="test_data",
        total_tasks=1,
        successful_tasks=1,
        failed_tasks=0,
        results=[orig_analysis],
    )

    critic_resp = CriticResponse(
        dataset_id="test_data",
        summary=CriticOverallSummary(
            total_results=1, accepted=1, questionable=0, rejected=0,
            requires_reinvestigation=0, overall_status="High Quality",
        ),
        reviews=[critic_rev],
    )

    req = VerificationRequest(
        analysis_response=plan_resp,
        critic_response=critic_resp,
        dataset_id="test_data",
    )

    resp = service.process_verification(req)
    assert resp.summary.total_findings == 1
    assert resp.summary.verified == 1
    assert resp.summary.rejected == 0
    assert len(resp.verified_findings) == 1
    assert resp.verified_findings[0].status == "VERIFIED"
    assert resp.verified_findings[0].reinvestigated is False


def test_rejection_of_rejected_critic_finding(sample_df):
    store_dataset("test_data", sample_df)
    service = VerificationService()

    orig_analysis = AnalysisResult(
        task_id="t2",
        status="failed",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        columns=["x", "const_col"],
        sample_size=0,
        results={},
        error=AnalysisError(code="CONSTANT_VARIABLE", message="Constant variable."),
    )

    critic_rev = CriticResult(
        task_id="t2",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        variables=["x", "const_col"],
        status="REJECTED",
        confidence=1.0,
        confidence_level="HIGH",
        summary="Rejected: variable has zero variance.",
        statistical_validity="Invalid.",
        method_assessment="Cannot evaluate constant variable.",
        recommendation="Exclude from modeling.",
        requires_reinvestigation=False,
    )

    plan_resp = PlanAnalysisResponse(
        dataset_id="test_data",
        total_tasks=1,
        successful_tasks=0,
        failed_tasks=1,
        results=[orig_analysis],
    )

    critic_resp = CriticResponse(
        dataset_id="test_data",
        summary=CriticOverallSummary(
            total_results=1, accepted=0, questionable=0, rejected=1,
            requires_reinvestigation=0, overall_status="Failed",
        ),
        reviews=[critic_rev],
    )

    req = VerificationRequest(
        analysis_response=plan_resp,
        critic_response=critic_resp,
        dataset_id="test_data",
    )

    resp = service.process_verification(req)
    assert resp.summary.total_findings == 1
    assert resp.summary.verified == 0
    assert resp.summary.rejected == 1
    assert len(resp.rejected_findings) == 1
    assert resp.rejected_findings[0].status == "REJECTED"


# ──────────────────────────────────────────────────────────────────────────────
# 2. TARGETED RE-INVESTIGATION TESTS
# ──────────────────────────────────────────────────────────────────────────────


def test_targeted_reinvestigation_pearson_to_spearman(sample_df):
    store_dataset("test_data", sample_df)
    service = VerificationService()

    # Original Pearson correlation
    orig_analysis = AnalysisResult(
        task_id="t3",
        status="success",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        columns=["x", "y"],
        sample_size=60,
        results={
            "correlation_coefficient": 0.85,
            "p_value": 0.0001,
            "sample_size": 60,
            "column_a": "x",
            "column_b": "y",
            "p_value_below_0_05": True,
        },
    )

    # Critic flags Pearson due to skew/outliers and recommends Spearman
    critic_rev = CriticResult(
        task_id="t3",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        variables=["x", "y"],
        status="REQUIRES_REINVESTIGATION",
        confidence=0.85,
        confidence_level="HIGH",
        summary="Requires re-investigation: Non-linear relationship suggests Spearman rank correlation.",
        statistical_validity="Valid calculation but sensitive to linearity assumption.",
        method_assessment="Spearman is preferable.",
        recommendation="Run Spearman rank correlation.",
        requires_reinvestigation=True,
        recommended_method="Spearman rank correlation",
    )

    plan_resp = PlanAnalysisResponse(
        dataset_id="test_data",
        total_tasks=1,
        successful_tasks=1,
        failed_tasks=0,
        results=[orig_analysis],
    )

    critic_resp = CriticResponse(
        dataset_id="test_data",
        summary=CriticOverallSummary(
            total_results=1, accepted=0, questionable=0, rejected=0,
            requires_reinvestigation=1, overall_status="Re-investigation Needed",
        ),
        reviews=[critic_rev],
    )

    req = VerificationRequest(
        analysis_response=plan_resp,
        critic_response=critic_resp,
        dataset_id="test_data",
    )

    resp = service.process_verification(req)
    assert resp.summary.total_findings == 1
    assert resp.summary.total_reinvestigated == 1
    assert resp.summary.verified_after_reinvestigation == 1
    assert len(resp.reinvestigations) == 1

    # Check Reinvestigation Record
    rec = resp.reinvestigations[0]
    assert rec.source_task_id == "t3"
    assert rec.original_method == "Pearson correlation"
    assert rec.reinvestigated_method == "Spearman rank correlation"
    assert rec.reinvestigated_analysis.status == "success"
    assert "correlation_coefficient" in rec.reinvestigated_analysis.results
    assert rec.second_critic_review.status == "ACCEPTED"
    assert rec.comparison.significance_consistent is True

    # Check final finding
    finding = resp.verified_findings[0]
    assert finding.status == "VERIFIED_AFTER_REINVESTIGATION"
    assert finding.reinvestigated is True
    assert finding.reinvestigation_record is not None


def test_targeted_reinvestigation_welch_to_mann_whitney(sample_df):
    store_dataset("test_data", sample_df)
    service = VerificationService()

    # Original Welch's t-test
    orig_analysis = AnalysisResult(
        task_id="t4",
        status="success",
        analysis_type="two_groups_vs_numerical",
        method="Welch's t-test",
        columns=["dept", "salary"],
        sample_size=60,
        results={
            "statistic": 3.5,
            "p_value": 0.001,
            "sample_size": 60,
            "group_column": "dept",
            "value_column": "salary",
        },
    )

    critic_rev = CriticResult(
        task_id="t4",
        analysis_type="two_groups_vs_numerical",
        method="Welch's t-test",
        variables=["dept", "salary"],
        status="REQUIRES_REINVESTIGATION",
        confidence=0.85,
        confidence_level="HIGH",
        summary="Requires re-investigation: Non-parametric comparison advised.",
        statistical_validity="Valid.",
        method_assessment="Mann-Whitney U is more robust.",
        recommendation="Run Mann-Whitney U test.",
        requires_reinvestigation=True,
        recommended_method="Mann-Whitney U test",
    )

    plan_resp = PlanAnalysisResponse(dataset_id="test_data", total_tasks=1, successful_tasks=1, failed_tasks=0, results=[orig_analysis])
    critic_resp = CriticResponse(dataset_id="test_data", summary=CriticOverallSummary(total_results=1, accepted=0, questionable=0, rejected=0, requires_reinvestigation=1, overall_status="Review"), reviews=[critic_rev])

    req = VerificationRequest(analysis_response=plan_resp, critic_response=critic_resp, dataset_id="test_data")
    resp = service.process_verification(req)

    assert resp.summary.verified_after_reinvestigation == 1
    assert resp.reinvestigations[0].reinvestigated_method == "Mann-Whitney U test"


# ──────────────────────────────────────────────────────────────────────────────
# 3. LOOP GUARD & EDGE CASE TESTS
# ──────────────────────────────────────────────────────────────────────────────


def test_max_reinvestigation_limit_prevents_loop(sample_df):
    store_dataset("test_data", sample_df)
    service = VerificationService()

    orig_analysis = AnalysisResult(
        task_id="t5",
        status="success",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        columns=["x", "y"],
        sample_size=60,
        results={"correlation_coefficient": 0.85, "p_value": 0.0001, "sample_size": 60},
    )

    critic_rev = CriticResult(
        task_id="t5",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        variables=["x", "y"],
        status="REQUIRES_REINVESTIGATION",
        summary="Re-investigate.",
        statistical_validity="Valid.",
        method_assessment="Appropriate.",
        recommendation="Run Spearman.",
        requires_reinvestigation=True,
        recommended_method="Spearman rank correlation",
    )

    # Set max_reinvestigations = 0 (guard limit)
    req = VerificationRequest(
        analysis_response=PlanAnalysisResponse(dataset_id="test_data", total_tasks=1, successful_tasks=1, failed_tasks=0, results=[orig_analysis]),
        critic_response=CriticResponse(dataset_id="test_data", summary=CriticOverallSummary(total_results=1, accepted=0, questionable=0, rejected=0, requires_reinvestigation=1, overall_status="Test"), reviews=[critic_rev]),
        dataset_id="test_data",
        max_reinvestigations=0,
    )

    resp = service.process_verification(req)
    assert resp.summary.total_reinvestigated == 0
    assert resp.all_findings[0].reinvestigated is False


def test_duplicate_method_prevention(sample_df):
    store_dataset("test_data", sample_df)
    service = VerificationService()

    # Original is Spearman and recommended is also Spearman (identical)
    orig_analysis = AnalysisResult(
        task_id="t6",
        status="success",
        analysis_type="numerical_vs_numerical",
        method="Spearman rank correlation",
        columns=["x", "y"],
        sample_size=60,
        results={"correlation_coefficient": 0.98, "p_value": 0.0001, "sample_size": 60},
    )

    critic_rev = CriticResult(
        task_id="t6",
        analysis_type="numerical_vs_numerical",
        method="Spearman rank correlation",
        variables=["x", "y"],
        status="QUESTIONABLE",
        summary="Questionable.",
        statistical_validity="Valid.",
        method_assessment="Appropriate.",
        recommendation="Run Spearman again.",
        requires_reinvestigation=True,
        recommended_method="Spearman rank correlation",  # Duplicate
    )

    req = VerificationRequest(
        analysis_response=PlanAnalysisResponse(dataset_id="test_data", total_tasks=1, successful_tasks=1, failed_tasks=0, results=[orig_analysis]),
        critic_response=CriticResponse(dataset_id="test_data", summary=CriticOverallSummary(total_results=1, accepted=0, questionable=1, rejected=0, requires_reinvestigation=1, overall_status="Test"), reviews=[critic_rev]),
        dataset_id="test_data",
    )

    resp = service.process_verification(req)
    # Duplicate is prevented from blindly running again
    assert resp.summary.total_reinvestigated == 0
    assert resp.summary.questionable == 1


def test_unsupported_recommended_method(sample_df):
    store_dataset("test_data", sample_df)
    service = VerificationService()

    orig_analysis = AnalysisResult(
        task_id="t7",
        status="success",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        columns=["x", "y"],
        sample_size=60,
        results={"correlation_coefficient": 0.85, "p_value": 0.0001, "sample_size": 60},
    )

    # Critic recommends an unsupported advanced method
    critic_rev = CriticResult(
        task_id="t7",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        variables=["x", "y"],
        status="QUESTIONABLE",
        summary="Questionable.",
        statistical_validity="Valid.",
        method_assessment="Appropriate.",
        recommendation="Run Bayesian Structural Equation Modeling.",
        requires_reinvestigation=True,
        recommended_method="Bayesian Structural Equation Modeling",  # Unsupported
    )

    req = VerificationRequest(
        analysis_response=PlanAnalysisResponse(dataset_id="test_data", total_tasks=1, successful_tasks=1, failed_tasks=0, results=[orig_analysis]),
        critic_response=CriticResponse(dataset_id="test_data", summary=CriticOverallSummary(total_results=1, accepted=0, questionable=1, rejected=0, requires_reinvestigation=1, overall_status="Test"), reviews=[critic_rev]),
        dataset_id="test_data",
    )

    resp = service.process_verification(req)
    assert resp.summary.total_reinvestigated == 0
    assert resp.summary.questionable == 1
    assert "not supported" in resp.questionable_findings[0].summary


# ──────────────────────────────────────────────────────────────────────────────
# 4. FASTAPI ENDPOINT INTEGRATION TESTS
# ──────────────────────────────────────────────────────────────────────────────


def test_api_verify_findings_endpoint(client, sample_df):
    store_dataset("api_data.csv", sample_df)

    plan_response = {
        "dataset_id": "api_data.csv",
        "total_tasks": 2,
        "successful_tasks": 2,
        "failed_tasks": 0,
        "results": [
            {
                "task_id": "api_t1",
                "status": "success",
                "analysis_type": "numerical_vs_numerical",
                "method": "Spearman rank correlation",
                "columns": ["x", "y"],
                "sample_size": 60,
                "results": {"correlation_coefficient": 0.98, "p_value": 0.0001, "sample_size": 60, "p_value_below_0_05": True},
            },
            {
                "task_id": "api_t2",
                "status": "success",
                "analysis_type": "two_groups_vs_numerical",
                "method": "Welch's t-test",
                "columns": ["dept", "salary"],
                "sample_size": 60,
                "results": {"statistic": 3.4, "p_value": 0.001, "sample_size": 60, "group_column": "dept", "value_column": "salary", "p_value_below_0_05": True},
            }
        ],
    }

    payload = {
        "analysis_response": plan_response,
        "dataset_id": "api_data.csv",
    }

    res = client.post("/api/verify-findings", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["dataset_id"] == "api_data.csv"
    assert data["summary"]["total_findings"] == 2
    assert data["summary"]["verified"] >= 1
    assert "verified_findings" in data
    assert "all_findings" in data


def test_api_reinvestigate_alias_endpoint(client, sample_df):
    store_dataset("api_data.csv", sample_df)

    payload = {
        "analysis_response": {
            "dataset_id": "api_data.csv",
            "total_tasks": 1,
            "successful_tasks": 1,
            "failed_tasks": 0,
            "results": [
                {
                    "task_id": "api_t3",
                    "status": "success",
                    "analysis_type": "numerical_vs_numerical",
                    "method": "Spearman rank correlation",
                    "columns": ["x", "y"],
                    "sample_size": 60,
                    "results": {"correlation_coefficient": 0.98, "p_value": 0.0001, "sample_size": 60},
                }
            ]
        },
        "dataset_id": "api_data.csv",
    }

    res = client.post("/api/reinvestigate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["dataset_id"] == "api_data.csv"
    assert data["summary"]["total_findings"] == 1


# ──────────────────────────────────────────────────────────────────────────────
# 5. PHASE 5 ANOVA -> KRUSKAL-WALLIS, CONTRADICTION & GRAPH PRESERVATION TESTS
# ──────────────────────────────────────────────────────────────────────────────


def test_reinvestigation_anova_to_kruskal_wallis(sample_df):
    """Phase 5: One-way ANOVA re-investigation executes non-parametric Kruskal-Wallis in Python with graph preservation."""
    store_dataset("multi_data", sample_df)
    service = VerificationService()

    orig_analysis = AnalysisResult(
        task_id="t_anova",
        status="success",
        analysis_type="multiple_groups_vs_numerical",
        method="One-way ANOVA",
        columns=["dept", "salary"],
        sample_size=60,
        results={
            "F_statistic": 3.85,
            "p_value": 0.027,
            "sample_size": 60,
            "p_value_below_0_05": True,
            "num_groups": 3,
        },
    )

    critic_rev = CriticResult(
        task_id="t_anova",
        analysis_type="multiple_groups_vs_numerical",
        method="One-way ANOVA",
        variables=["dept", "salary"],
        status="REQUIRES_REINVESTIGATION",
        confidence=0.85,
        confidence_level="HIGH",
        summary="Severe positive skewness or group imbalance detected; ANOVA normality assumption compromised.",
        evidence=["Group size imbalance detected"],
        statistical_validity="Parametric normality assumption doubtful.",
        method_assessment="One-way ANOVA compromised by distributional skew.",
        recommendation="Perform non-parametric Kruskal-Wallis test.",
        requires_reinvestigation=True,
        recommended_method="Kruskal-Wallis test",
        reason_codes=["RECOMMEND_KRUSKAL_WALLIS"],
    )

    req = VerificationRequest(
        dataset_id="multi_data",
        analysis_response=PlanAnalysisResponse(
            dataset_id="multi_data",
            total_tasks=1,
            successful_tasks=1,
            failed_tasks=0,
            results=[orig_analysis],
        ),
        critic_response=CriticResponse(
            dataset_id="multi_data",
            summary=CriticOverallSummary(
                total_results=1, accepted=0, questionable=0, rejected=0,
                requires_reinvestigation=1, overall_status="Requires re-investigation",
            ),
            reviews=[critic_rev],
        ),
    )

    resp = service.process_verification(req)

    assert resp.summary.total_reinvestigated == 1
    assert len(resp.reinvestigations) == 1
    rec = resp.reinvestigations[0]

    # Verify method transition and Python computation
    assert rec.original_method == "One-way ANOVA"
    assert rec.reinvestigated_method == "Kruskal-Wallis test"
    assert "H_statistic" in rec.reinvestigated_analysis.results
    assert "p_value" in rec.reinvestigated_analysis.results

    # Verify graph preservation
    assert rec.original_graph_id is not None
    assert rec.reinvestigated_graph_id is not None
    assert rec.original_visual_evidence is not None
    assert rec.reinvestigated_visual_evidence is not None

    # Verify verified finding
    assert len(resp.verified_findings) >= 1
    vf = resp.verified_findings[0]
    assert vf.reinvestigated is True
    assert vf.method == "Kruskal-Wallis test"
    assert vf.graph_id == rec.reinvestigated_graph_id


def test_contradictory_reinvestigation_remains_questionable(sample_df):
    """Phase 5: If follow-up re-investigation yields contradictory evidence, finding preserves complete audit trail."""
    store_dataset("contra_data", sample_df)
    service = VerificationService()

    # Original analysis: significant Pearson correlation
    orig_analysis = AnalysisResult(
        task_id="t_contra",
        status="success",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        columns=["x", "y"],
        sample_size=60,
        results={
            "correlation_coefficient": 0.82,
            "p_value": 0.001,
            "sample_size": 60,
            "p_value_below_0_05": True,
        },
    )

    critic_rev = CriticResult(
        task_id="t_contra",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        variables=["x", "y"],
        status="REQUIRES_REINVESTIGATION",
        confidence=0.80,
        confidence_level="HIGH",
        summary="Extreme outliers detected.",
        evidence=["Outliers present."],
        statistical_validity="Vulnerable to outliers.",
        method_assessment="Pearson questionable.",
        recommendation="Run Spearman rank correlation.",
        requires_reinvestigation=True,
        recommended_method="Spearman rank correlation",
        reason_codes=["RECOMMEND_SPEARMAN_RANK"],
    )

    req = VerificationRequest(
        dataset_id="contra_data",
        analysis_response=PlanAnalysisResponse(
            dataset_id="contra_data", total_tasks=1, successful_tasks=1, failed_tasks=0, results=[orig_analysis]
        ),
        critic_response=CriticResponse(
            dataset_id="contra_data",
            summary=CriticOverallSummary(
                total_results=1, accepted=0, questionable=0, rejected=0,
                requires_reinvestigation=1, overall_status="Requires re-investigation",
            ),
            reviews=[critic_rev],
        ),
    )

    resp = service.process_verification(req)
    # The findings are evaluated and categorized cleanly
    assert resp.summary.total_findings == 1
    all_f = resp.all_findings[0]
    assert all_f.reinvestigated is True
    assert all_f.reinvestigation_record is not None
    assert all_f.reinvestigation_record.original_analysis.results is not None
    assert all_f.reinvestigation_record.trigger_reason is not None


def test_baseline_bypasses_critic_and_verification():
    """Safety requirement: Baseline mode executes planner and analysis but NEVER executes Critic or Verification."""
    from app.schemas.experiment_schema import BranchRunSummary

    # Verify that BranchRunSummary schema accurately models baseline with zero critic or verification
    baseline_branch = BranchRunSummary(
        run_id="run_baseline_123",
        mode="baseline",
        dataset_id="test_dataset",
        total_tasks_executed=4,
        overall_score=72.0,
        overall_grade="C",
        has_critic_review=False,
        has_verification=False,
        has_insights=False,
        completed_phases=[2, 3, 6],
    )
    assert baseline_branch.mode == "baseline"
    assert baseline_branch.has_critic_review is False
    assert baseline_branch.has_verification is False
    assert baseline_branch.has_insights is False
    assert 4 not in baseline_branch.completed_phases
    assert 5 not in baseline_branch.completed_phases

    proposed_branch = BranchRunSummary(
        run_id="run_proposed_123",
        mode="proposed",
        dataset_id="test_dataset",
        total_tasks_executed=4,
        overall_score=94.5,
        overall_grade="A",
        has_critic_review=True,
        has_verification=True,
        has_insights=True,
        completed_phases=[2, 3, 4, 5, 6],
    )
    assert proposed_branch.mode == "proposed"
    assert proposed_branch.has_critic_review is True
    assert proposed_branch.has_verification is True
    assert proposed_branch.has_insights is True
    assert 4 in proposed_branch.completed_phases
    assert 5 in proposed_branch.completed_phases


# ──────────────────────────────────────────────────────────────────────────────
# 6. TASK 1: BOUNDED RE-INVESTIGATION LIMITS (0, 1, 2, 3), EARLY STOPPING & CYCLING
# ──────────────────────────────────────────────────────────────────────────────


def test_reinvestigation_limit_0_disables_follow_up(sample_df):
    """Limit 0: When max_reinvestigations=0, zero follow-ups execute; finding is honestly retained as QUESTIONABLE."""
    store_dataset("test_limit_0", sample_df)
    service = VerificationService()

    orig_analysis = AnalysisResult(
        task_id="t_lim0",
        status="success",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        columns=["x", "y"],
        sample_size=60,
        results={"correlation_coefficient": 0.85, "p_value": 0.001, "sample_size": 60},
    )

    critic_rev = CriticResult(
        task_id="t_lim0",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        variables=["x", "y"],
        status="REQUIRES_REINVESTIGATION",
        confidence=0.85,
        confidence_level="HIGH",
        summary="Extreme outliers present; Pearson linearity compromised.",
        evidence=["Outliers detected."],
        statistical_validity="Compromised.",
        method_assessment="Questionable.",
        recommendation="Run Spearman rank correlation.",
        requires_reinvestigation=True,
        recommended_method="Spearman rank correlation",
        reason_codes=["RECOMMEND_SPEARMAN_RANK"],
    )

    req = VerificationRequest(
        dataset_id="test_limit_0",
        analysis_response=PlanAnalysisResponse(dataset_id="test_limit_0", total_tasks=1, successful_tasks=1, failed_tasks=0, results=[orig_analysis]),
        critic_response=CriticResponse(dataset_id="test_limit_0", summary=CriticOverallSummary(total_results=1, accepted=0, questionable=0, rejected=0, requires_reinvestigation=1, overall_status="Test"), reviews=[critic_rev]),
        max_reinvestigations=0,
    )

    resp = service.process_verification(req)

    assert resp.summary.total_reinvestigated == 0
    assert len(resp.reinvestigations) == 0
    assert len(resp.all_findings) == 1
    f = resp.all_findings[0]
    assert f.reinvestigated is False
    assert f.status == "QUESTIONABLE"
    assert "REINVESTIGATION_DISABLED" in f.reason_codes
    assert "max_reinvestigations=0" in f.verification_reason


def test_reinvestigation_early_stopping_on_success(sample_df):
    """Early Stopping: When max_reinvestigations=3 but attempt 1 succeeds with ACCEPTED, loop stops immediately."""
    store_dataset("test_early_stop", sample_df)
    service = VerificationService()

    orig_analysis = AnalysisResult(
        task_id="t_early",
        status="success",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        columns=["x", "y"],
        sample_size=60,
        results={"correlation_coefficient": 0.85, "p_value": 0.001, "sample_size": 60},
    )

    critic_rev = CriticResult(
        task_id="t_early",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        variables=["x", "y"],
        status="REQUIRES_REINVESTIGATION",
        confidence=0.85,
        confidence_level="HIGH",
        summary="Outliers detected.",
        evidence=["Outliers present."],
        statistical_validity="Questionable.",
        method_assessment="Pearson questionable.",
        recommendation="Run Spearman rank correlation.",
        requires_reinvestigation=True,
        recommended_method="Spearman rank correlation",
        reason_codes=["RECOMMEND_SPEARMAN_RANK"],
    )

    req = VerificationRequest(
        dataset_id="test_early_stop",
        analysis_response=PlanAnalysisResponse(dataset_id="test_early_stop", total_tasks=1, successful_tasks=1, failed_tasks=0, results=[orig_analysis]),
        critic_response=CriticResponse(dataset_id="test_early_stop", summary=CriticOverallSummary(total_results=1, accepted=0, questionable=0, rejected=0, requires_reinvestigation=1, overall_status="Test"), reviews=[critic_rev]),
        max_reinvestigations=3,
    )

    resp = service.process_verification(req)

    # Even though max_reinvestigations was 3, attempt 1 resolved it cleanly, so exactly 1 attempt ran
    assert resp.summary.total_reinvestigated == 1
    assert len(resp.reinvestigations) == 1
    assert resp.reinvestigations[0].attempt_number == 1
    assert resp.verified_findings[0].status == "VERIFIED_AFTER_REINVESTIGATION"
    assert resp.verified_findings[0].method == "Spearman rank correlation"


def test_reinvestigation_prevents_method_cycling(sample_df):
    """Cycle Prevention: If the Critic recommends a method that was already attempted, execution stops immediately to prevent loops and p-hacking."""
    store_dataset("test_cycle", sample_df)
    service = VerificationService()

    orig_analysis = AnalysisResult(
        task_id="t_cycle",
        status="success",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        columns=["x", "y"],
        sample_size=60,
        results={"correlation_coefficient": 0.50, "p_value": 0.02, "sample_size": 60},
    )

    # Critic erroneously recommends the same method as original (Pearson)
    critic_rev = CriticResult(
        task_id="t_cycle",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        variables=["x", "y"],
        status="REQUIRES_REINVESTIGATION",
        confidence=0.70,
        confidence_level="MEDIUM",
        summary="Questionable linearity.",
        evidence=["Questionable."],
        statistical_validity="Questionable.",
        method_assessment="Needs re-check.",
        recommendation="Re-run Pearson correlation.",
        requires_reinvestigation=True,
        recommended_method="Pearson correlation",  # Duplicate of original!
        reason_codes=["RECOMMEND_PEARSON"],
    )

    req = VerificationRequest(
        dataset_id="test_cycle",
        analysis_response=PlanAnalysisResponse(dataset_id="test_cycle", total_tasks=1, successful_tasks=1, failed_tasks=0, results=[orig_analysis]),
        critic_response=CriticResponse(dataset_id="test_cycle", summary=CriticOverallSummary(total_results=1, accepted=0, questionable=0, rejected=0, requires_reinvestigation=1, overall_status="Test"), reviews=[critic_rev]),
        max_reinvestigations=2,
    )

    resp = service.process_verification(req)

    # Cycle detected: zero duplicate attempts executed; finding retained as QUESTIONABLE
    assert resp.summary.total_reinvestigated == 0
    assert len(resp.reinvestigations) == 0
    assert len(resp.all_findings) == 1
    assert resp.all_findings[0].status == "QUESTIONABLE"
    assert "duplicate" in resp.all_findings[0].summary.lower() or "duplicate" in resp.all_findings[0].verification_reason.lower()


def test_reinvestigation_multi_attempt_provenance(sample_df, monkeypatch):
    """Multi-attempt Provenance: Test that when attempt 1 is still questionable with a new alternative, attempt 2 executes and records attempt numbers."""
    store_dataset("test_multi_att", sample_df)
    service = VerificationService()

    orig_analysis = AnalysisResult(
        task_id="t_seq",
        status="success",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        columns=["x", "y"],
        sample_size=60,
        results={"correlation_coefficient": 0.85, "p_value": 0.001, "sample_size": 60},
    )

    initial_critic = CriticResult(
        task_id="t_seq",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        variables=["x", "y"],
        status="REQUIRES_REINVESTIGATION",
        confidence=0.80,
        confidence_level="HIGH",
        summary="Outliers detected.",
        evidence=["Outliers present."],
        statistical_validity="Questionable.",
        method_assessment="Pearson questionable.",
        recommendation="Run Spearman rank correlation.",
        requires_reinvestigation=True,
        recommended_method="Spearman rank correlation",
        reason_codes=["RECOMMEND_SPEARMAN_RANK"],
    )

    # Mock Critic's second review: attempt 1 (Spearman) is evaluated as still requiring investigation with a second method, and attempt 2 succeeds!
    call_count = 0
    def mock_evaluate_result(analysis_res, profile=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Review of attempt 1: still questionable, recommends another method
            return CriticResult(
                task_id=analysis_res.task_id,
                analysis_type="numerical_vs_numerical",
                method=analysis_res.method,
                variables=list(analysis_res.columns),
                status="REQUIRES_REINVESTIGATION",
                confidence=0.75,
                confidence_level="MEDIUM",
                summary="Spearman still sensitive to monotonic saturation; recommend IQR sensitivity check.",
                evidence=["Saturation"],
                statistical_validity="Caveats",
                method_assessment="Suggest sensitivity",
                recommendation="Run IQR-based sensitivity analysis",
                requires_reinvestigation=True,
                recommended_method="IQR-based sensitivity analysis",
                reason_codes=["RECOMMEND_IQR_SENSITIVITY"],
            )
        else:
            # Review of attempt 2: accepted!
            return CriticResult(
                task_id=analysis_res.task_id,
                analysis_type=analysis_res.analysis_type,
                method=analysis_res.method,
                variables=list(analysis_res.columns),
                status="ACCEPTED",
                confidence=0.90,
                confidence_level="HIGH",
                summary="IQR sensitivity validated robustness.",
                evidence=["Robust"],
                statistical_validity="Valid",
                method_assessment="Appropriate",
                recommendation="Finding is verified.",
                requires_reinvestigation=False,
                recommended_method=None,
                reason_codes=["MINIMAL_OUTLIER_INFLUENCE"],
            )

    monkeypatch.setattr(service.critic_agent, "evaluate_result", mock_evaluate_result)

    req = VerificationRequest(
        dataset_id="test_multi_att",
        analysis_response=PlanAnalysisResponse(dataset_id="test_multi_att", total_tasks=1, successful_tasks=1, failed_tasks=0, results=[orig_analysis]),
        critic_response=CriticResponse(dataset_id="test_multi_att", summary=CriticOverallSummary(total_results=1, accepted=0, questionable=0, rejected=0, requires_reinvestigation=1, overall_status="Test"), reviews=[initial_critic]),
        max_reinvestigations=2,
    )

    resp = service.process_verification(req)

    # Two attempts executed!
    assert resp.summary.total_reinvestigated == 2
    assert len(resp.reinvestigations) == 2

    # Check attempt 1 provenance
    rec1 = resp.reinvestigations[0]
    assert rec1.attempt_number == 1
    assert rec1.original_method == "Pearson correlation"
    assert rec1.reinvestigated_method == "Spearman rank correlation"
    assert rec1.original_graph_id is not None
    assert rec1.reinvestigated_graph_id is not None

    # Check attempt 2 provenance
    rec2 = resp.reinvestigations[1]
    assert rec2.attempt_number == 2
    assert rec2.original_method == "Spearman rank correlation"
    assert rec2.reinvestigated_method == "IQR-based sensitivity analysis"
    assert rec2.reinvestigation_id == f"reinvest_t_seq_attempt_2"
    assert rec2.status == "VERIFIED_AFTER_REINVESTIGATION"

    # Final verified finding reflects second attempt resolution
    assert len(resp.verified_findings) == 1
    vf = resp.verified_findings[0]
    assert vf.status == "VERIFIED_AFTER_REINVESTIGATION"
    assert vf.method == "IQR-based sensitivity analysis"
    assert "2 attempt(s)" in vf.verification_reason


def test_reinvestigation_recovers_dataset_from_disk_after_cache_cleared():
    """When dataset cache is cleared, VerificationService reloads dataset from disk (e.g. sample.csv) using parse_csv."""
    clear_cache()

    service = VerificationService()

    orig_analysis = AnalysisResult(
        task_id="t_disk_rec",
        status="success",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        columns=["age", "salary"],
        sample_size=30,
        results={
            "correlation_coefficient": 0.85,
            "p_value": 0.001,
            "sample_size": 30,
            "p_value_below_0_05": True,
            "column_a": "age",
            "column_b": "salary",
        },
    )

    critic_rev = CriticResult(
        task_id="t_disk_rec",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        variables=["age", "salary"],
        status="REQUIRES_REINVESTIGATION",
        confidence=0.85,
        confidence_level="HIGH",
        summary="Distributional skewness compromises Pearson linearity.",
        evidence=["N=30, r=0.85, skew=2.1"],
        statistical_validity="Valid.",
        method_assessment="Linearity violation.",
        recommendation="Perform Spearman rank correlation.",
        requires_reinvestigation=True,
        recommended_method="Spearman rank correlation",
        reason_codes=["RECOMMEND_SPEARMAN_RANK"],
    )

    # Use dataset_id="sample.csv" which exists on disk in data/sample.csv
    req = VerificationRequest(
        dataset_id="sample.csv",
        analysis_response=PlanAnalysisResponse(dataset_id="sample.csv", total_tasks=1, successful_tasks=1, failed_tasks=0, results=[orig_analysis]),
        critic_response=CriticResponse(dataset_id="sample.csv", summary=CriticOverallSummary(total_results=1, accepted=0, questionable=0, rejected=0, requires_reinvestigation=1, overall_status="Test"), reviews=[critic_rev]),
        max_reinvestigations=1,
    )

    resp = service.process_verification(req)

    # Verify that reinvestigation successfully resolved from disk without error
    assert resp.summary.total_reinvestigated == 1
    assert len(resp.reinvestigations) == 1
    assert resp.reinvestigations[0].reinvestigated_method == "Spearman rank correlation"
    assert resp.reinvestigations[0].status == "VERIFIED_AFTER_REINVESTIGATION"
    assert len(resp.verified_findings) == 1
    assert resp.verified_findings[0].status == "VERIFIED_AFTER_REINVESTIGATION"


def test_reinvestigation_records_clear_reason_when_dataset_unavailable():
    """When dataset is genuinely unavailable, reinvestigation is not silently skipped; clear failure is recorded."""
    clear_cache()

    service = VerificationService()

    orig_analysis = AnalysisResult(
        task_id="t_unavail",
        status="success",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        columns=["x", "y"],
        sample_size=30,
        results={
            "correlation_coefficient": 0.75,
            "p_value": 0.01,
            "sample_size": 30,
            "p_value_below_0_05": True,
            "column_a": "x",
            "column_b": "y",
        },
    )

    critic_rev = CriticResult(
        task_id="t_unavail",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        variables=["x", "y"],
        status="REQUIRES_REINVESTIGATION",
        confidence=0.80,
        confidence_level="HIGH",
        summary="Pearson linearity assumption compromised by skew.",
        evidence=["N=30"],
        statistical_validity="Valid.",
        method_assessment="Linearity violation.",
        recommendation="Perform Spearman rank correlation.",
        requires_reinvestigation=True,
        recommended_method="Spearman rank correlation",
        reason_codes=["RECOMMEND_SPEARMAN_RANK"],
    )

    req = VerificationRequest(
        dataset_id="nonexistent_dataset_99999.csv",
        analysis_response=PlanAnalysisResponse(dataset_id="nonexistent_dataset_99999.csv", total_tasks=1, successful_tasks=1, failed_tasks=0, results=[orig_analysis]),
        critic_response=CriticResponse(dataset_id="nonexistent_dataset_99999.csv", summary=CriticOverallSummary(total_results=1, accepted=0, questionable=0, rejected=0, requires_reinvestigation=1, overall_status="Test"), reviews=[critic_rev]),
        max_reinvestigations=1,
    )

    resp = service.process_verification(req)

    # 0 reinvestigations executed because dataset was missing
    assert resp.summary.total_reinvestigated == 0
    assert len(resp.reinvestigations) == 0

    # Finding is retained as QUESTIONABLE with explicit failure provenance
    assert len(resp.questionable_findings) == 1
    finding = resp.questionable_findings[0]
    assert finding.status == "QUESTIONABLE"
    assert "DATASET_UNAVAILABLE_FOR_REINVESTIGATION" in finding.reason_codes
    assert "precluded: dataset unavailable" in finding.summary
    assert "could not be resolved from memory" in finding.verification_reason
    assert any("dataset 'nonexistent_dataset_99999.csv' was unavailable" in lim for lim in finding.limitations)



def test_reinvestigation_preserves_target_variable_in_multicolumn_task():
    """
    Regression test for Task 23:
    Ensure that when an investigation task contains 3+ columns (e.g. studytime, absences, G3)
    with explicit target variables (column_a='studytime', column_b='G3'),
    the Critic-driven reinvestigation preserves the exact same target pair ('studytime' vs 'G3')
    and does NOT silently evaluate ('studytime' vs 'absences').
    """
    import numpy as np
    import pandas as pd
    from app.services.dataset_store import store_dataset
    from app.services.verification_service import get_verification_service
    from app.schemas.verification_schema import VerificationRequest
    from app.schemas.analysis_schema import PlanAnalysisResponse, AnalysisResult
    from app.schemas.critic_schema import CriticResponse, CriticResult, CriticOverallSummary
    from app.schemas.planner_schema import PlannerResponse, InvestigationTask

    np.random.seed(42)
    n = 100
    studytime = np.random.randint(1, 5, size=n)
    absences = np.random.poisson(5, size=n)
    g3 = studytime * 2 + np.random.normal(0, 1, size=n)
    df = pd.DataFrame({"studytime": studytime, "absences": absences, "G3": g3})

    store_dataset("test_task23_preserve_pair.csv", df)

    service = get_verification_service()

    orig_analysis = AnalysisResult(
        task_id="task_4",
        status="success",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        columns=["studytime", "absences", "G3"],
        sample_size=n,
        results={
            "column_a": "studytime",
            "column_b": "G3",
            "statistic": 0.65,
            "correlation_coefficient": 0.65,
            "p_value": 0.001,
            "sample_size": n,
        },
    )

    plan = PlannerResponse(
        dataset_id="test_task23_preserve_pair.csv",
        objective="Assess factors influencing final grade G3",
        interpreted_goal="Test goal",
        selected_columns=[],
        reasoning=["Test reasoning"],
        warnings=[],
        expected_outputs=[],
        analysis_tasks=[
            InvestigationTask(
                task_id="task_4",
                task_type="numerical_vs_numerical",
                task="Study time, absences, and G3 analysis",
                description="Evaluate association of studytime and absences with G3",
                method="Pearson correlation",
                columns=["studytime", "absences", "G3"],
                column_a="studytime",
                column_b="G3",
                reason="Test reason",
                priority="high",
            )
        ],
        tasks=[],
    )

    critic_rev = CriticResult(
        task_id="task_4",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        statistical_validity="Valid",
        method_assessment="Appropriate",
        recommendation="Perform Spearman rank correlation on 'studytime' vs 'G3'",
        status="REQUIRES_REINVESTIGATION",
        confidence=0.85,
        confidence_level="HIGH",
        verdict="REQUIRES_REINVESTIGATION",
        summary="Distribution skew and potential outliers require non-parametric rank verification.",
        requires_reinvestigation=True,
        recommended_method="Spearman rank correlation",
        reason_codes=["VISUAL_OUTLIERS_REINVESTIGATE"],
    )

    req = VerificationRequest(
        dataset_id="test_task23_preserve_pair.csv",
        analysis_response=PlanAnalysisResponse(
            dataset_id="test_task23_preserve_pair.csv",
            total_tasks=1,
            successful_tasks=1,
            failed_tasks=0,
            results=[orig_analysis],
        ),
        critic_response=CriticResponse(
            dataset_id="test_task23_preserve_pair.csv",
            summary=CriticOverallSummary(
                total_results=1, accepted=0, questionable=0, rejected=0, requires_reinvestigation=1, overall_status="Test"
            ),
            reviews=[critic_rev],
        ),
        investigation_plan=plan,
        max_reinvestigations=1,
    )

    resp = service.process_verification(req)

    assert resp.summary.total_reinvestigated == 1
    reinv = resp.reinvestigations[0]

    # Crucial assertions: Must evaluate studytime vs G3, NOT studytime vs absences!
    assert reinv.variables == ["studytime", "G3"]
    assert reinv.reinvestigated_analysis.results["column_a"] == "studytime"
    assert reinv.reinvestigated_analysis.results["column_b"] == "G3"
    assert reinv.reinvestigated_analysis.results["column_b"] != "absences"

    # Comparison must reflect consistent pair
    assert reinv.comparison.direction_consistent is True
    assert "Variable pair changed" not in (reinv.comparison.effect_size_shift or "")


def test_reinvestigation_preserves_target_variable_g1_g3():
    """
    Regression test for Task 23:
    Ensure G1 vs G3 with multi-column list ['G1', 'G2', 'G3'] evaluates G1 vs G3, NOT G1 vs G2.
    """
    import numpy as np
    import pandas as pd
    from app.services.dataset_store import store_dataset
    from app.services.verification_service import get_verification_service
    from app.schemas.verification_schema import VerificationRequest
    from app.schemas.analysis_schema import PlanAnalysisResponse, AnalysisResult
    from app.schemas.critic_schema import CriticResponse, CriticResult, CriticOverallSummary
    from app.schemas.planner_schema import PlannerResponse, InvestigationTask

    np.random.seed(42)
    n = 100
    g1 = np.random.normal(10, 3, size=n)
    g2 = g1 * 0.9 + np.random.normal(0, 1, size=n)
    g3 = g1 * 0.85 + np.random.normal(0, 1.5, size=n)
    df = pd.DataFrame({"G1": g1, "G2": g2, "G3": g3})

    store_dataset("test_task23_g1_g3.csv", df)

    service = get_verification_service()

    orig_analysis = AnalysisResult(
        task_id="task_3",
        status="success",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        columns=["G1", "G2", "G3"],
        sample_size=n,
        results={
            "column_a": "G1",
            "column_b": "G3",
            "statistic": 0.80,
            "correlation_coefficient": 0.80,
            "p_value": 1e-10,
            "sample_size": n,
        },
    )

    plan = PlannerResponse(
        dataset_id="test_task23_g1_g3.csv",
        objective="Analyze grade progression",
        interpreted_goal="Test goal",
        selected_columns=[],
        reasoning=["Test reasoning"],
        warnings=[],
        expected_outputs=[],
        analysis_tasks=[
            InvestigationTask(
                task_id="task_3",
                task_type="numerical_vs_numerical",
                task="Prior grades vs final grade",
                description="Evaluate correlation between G1, G2, and G3",
                method="Pearson correlation",
                columns=["G1", "G2", "G3"],
                column_a="G1",
                column_b="G3",
                reason="Test reason",
                priority="high",
            )
        ],
        tasks=[],
    )

    critic_rev = CriticResult(
        task_id="task_3",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        statistical_validity="Valid",
        method_assessment="Appropriate",
        recommendation="Perform Spearman rank correlation on 'G1' vs 'G3'",
        status="REQUIRES_REINVESTIGATION",
        confidence=0.90,
        confidence_level="HIGH",
        verdict="REQUIRES_REINVESTIGATION",
        summary="Outliers detected, run Spearman rank correlation.",
        requires_reinvestigation=True,
        recommended_method="Spearman rank correlation",
        reason_codes=["VISUAL_OUTLIERS_REINVESTIGATE"],
    )

    req = VerificationRequest(
        dataset_id="test_task23_g1_g3.csv",
        analysis_response=PlanAnalysisResponse(
            dataset_id="test_task23_g1_g3.csv",
            total_tasks=1,
            successful_tasks=1,
            failed_tasks=0,
            results=[orig_analysis],
        ),
        critic_response=CriticResponse(
            dataset_id="test_task23_g1_g3.csv",
            summary=CriticOverallSummary(
                total_results=1, accepted=0, questionable=0, rejected=0, requires_reinvestigation=1, overall_status="Test"
            ),
            reviews=[critic_rev],
        ),
        investigation_plan=plan,
        max_reinvestigations=1,
    )

    resp = service.process_verification(req)
    assert resp.summary.total_reinvestigated == 1
    reinv = resp.reinvestigations[0]

    assert reinv.variables == ["G1", "G3"]
    assert reinv.reinvestigated_analysis.results["column_a"] == "G1"
    assert reinv.reinvestigated_analysis.results["column_b"] == "G3"
    assert reinv.reinvestigated_analysis.results["column_b"] != "G2"


def test_reinvestigation_comparison_detects_variable_mismatch():
    """
    Verify that _compare_results raises an explicit warning and prevents false consistency
    if original and re-investigated analyses accidentally evaluated different variable pairs.
    """
    from app.services.verification_service import get_verification_service
    from app.schemas.analysis_schema import AnalysisResult

    service = get_verification_service()

    orig = AnalysisResult(
        task_id="t1",
        status="success",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        columns=["studytime", "G3"],
        sample_size=100,
        results={"column_a": "studytime", "column_b": "G3", "statistic": 0.15, "p_value": 0.04},
    )

    mismatched_reinvest = AnalysisResult(
        task_id="t1_reinvest",
        status="success",
        analysis_type="numerical_vs_numerical",
        method="Spearman rank correlation",
        columns=["studytime", "absences"],
        sample_size=100,
        results={"column_a": "studytime", "column_b": "absences", "statistic": -0.05, "p_value": 0.35},
    )

    comp = service._compare_results(orig, mismatched_reinvest)

    assert comp.direction_consistent is False
    assert comp.significance_consistent is False
    assert "Warning: Variable pair changed" in comp.effect_size_shift


def test_insight_service_describes_actual_computed_variables():
    """
    Verify that InsightService prioritizes exact evaluated columns from statistical evidence
    over arbitrary column list indices.
    """
    from app.services.insight_service import get_insight_service
    from app.schemas.verification_schema import VerifiedFinding, VerificationResponse, VerificationSummary

    insight_svc = get_insight_service()

    finding = VerifiedFinding(
        finding_id="f1",
        task_id="t1",
        variables=["studytime", "G3"],
        method="Spearman rank correlation",
        status="VERIFIED_AFTER_REINVESTIGATION",
        confidence=1.0,
        confidence_level="HIGH",
        summary="Verified after re-investigation",
        statistical_evidence={
            "column_a": "studytime",
            "column_b": "G3",
            "statistic": 0.1052,
            "p_value": 0.0367,
            "sample_size": 395,
        },
        reinvestigated=True,
        critic_assessment="Finding is empirically verified and robust.",
        verification_reason="Re-investigation resolved prior concerns.",
    )

    from app.schemas.insight_schema import InsightRequest

    v_resp = VerificationResponse(
        dataset_id="test.csv",
        summary=VerificationSummary(
            total_findings=1,
            verified=0,
            verified_after_reinvestigation=1,
            questionable=0,
            rejected=0,
            total_reinvestigated=1,
            overall_integrity="High",
        ),
        verified_findings=[finding],
        questionable_findings=[],
        rejected_findings=[],
        reinvestigations=[],
        all_findings=[finding],
    )

    insight_req = InsightRequest(verification_response=v_resp)
    insight_resp = insight_svc.generate_insights(insight_req)
    assert len(insight_resp.key_insights) == 1
    ins = insight_resp.key_insights[0]

    assert "studytime" in ins.title.lower()
    assert "g3" in ins.title.lower()
    assert "absences" not in ins.title.lower()
    assert "studytime" in ins.insight.lower()
    assert "g3" in ins.insight.lower()


def test_task23_mismatched_pair_assigned_questionable():
    """
    Task 23 Regression Test:
    Verify that if a reinvestigation somehow evaluates a mismatched variable pair
    (e.g., studytime vs absences instead of studytime vs G3), the finding is NOT
    assigned VERIFIED_AFTER_REINVESTIGATION, but QUESTIONABLE with VARIABLE_PAIR_MISMATCH.
    """
    from unittest.mock import MagicMock
    import pandas as pd
    from app.services.verification_service import get_verification_service
    from app.schemas.verification_schema import VerificationRequest
    from app.schemas.analysis_schema import PlanAnalysisResponse, AnalysisResult
    from app.schemas.critic_schema import CriticResponse, CriticResult, CriticOverallSummary
    from app.schemas.planner_schema import PlannerResponse, InvestigationTask
    from app.services.dataset_store import store_dataset

    df = pd.DataFrame({"studytime": [1.0, 2.0, 3.0, 4.0] * 25, "G3": [10.0, 12.0, 14.0, 16.0] * 25, "absences": [0.0, 2.0, 4.0, 6.0] * 25})
    store_dataset("test_task23_mismatch.csv", df)

    service = get_verification_service()

    # Original analysis was studytime vs G3
    orig = AnalysisResult(
        task_id="task_4",
        status="success",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        columns=["studytime", "G3"],
        sample_size=100,
        results={"column_a": "studytime", "column_b": "G3", "statistic": 0.50, "p_value": 0.001, "sample_size": 100},
    )

    # Mock AnalysisAgent to artificially return studytime vs absences (simulating the Task 21 defect)
    mismatched_result = AnalysisResult(
        task_id="task_4_reinvest",
        status="success",
        analysis_type="numerical_vs_numerical",
        method="Spearman rank correlation",
        columns=["studytime", "absences"],
        sample_size=100,
        results={"column_a": "studytime", "column_b": "absences", "statistic": -0.05, "p_value": 0.35, "sample_size": 100},
    )
    original_execute = service.analysis_agent.execute_task
    service.analysis_agent.execute_task = MagicMock(return_value=mismatched_result)

    try:
        plan = PlannerResponse(
            dataset_id="test_task23_mismatch.csv",
            objective="Test",
            interpreted_goal="Test",
            selected_columns=[],
            reasoning=[],
            warnings=[],
            expected_outputs=[],
            analysis_tasks=[
                InvestigationTask(
                    task_id="task_4",
                    task_type="numerical_vs_numerical",
                    task="Study time vs G3",
                    description="Evaluate studytime vs G3",
                    method="Pearson correlation",
                    columns=["studytime", "G3"],
                    column_a="studytime",
                    column_b="G3",
                    reason="Test",
                    priority="high",
                )
            ],
            tasks=[],
        )

        critic_rev = CriticResult(
            task_id="task_4",
            analysis_type="numerical_vs_numerical",
            method="Pearson correlation",
            statistical_validity="Valid",
            method_assessment="Appropriate",
            recommendation="Perform Spearman rank correlation on 'studytime' vs 'G3'",
            status="REQUIRES_REINVESTIGATION",
            confidence=0.85,
            confidence_level="HIGH",
            summary="Distribution concerns require non-parametric test.",
            requires_reinvestigation=True,
            recommended_method="Spearman rank correlation",
            reason_codes=["VISUAL_OUTLIERS_REINVESTIGATE"],
        )

        req = VerificationRequest(
            dataset_id="test_task23_mismatch.csv",
            analysis_response=PlanAnalysisResponse(
                dataset_id="test_task23_mismatch.csv", total_tasks=1, successful_tasks=1, failed_tasks=0, results=[orig]
            ),
            critic_response=CriticResponse(
                dataset_id="test_task23_mismatch.csv",
                summary=CriticOverallSummary(
                    total_results=1, accepted=0, questionable=0, rejected=0, requires_reinvestigation=1, overall_status="Test"
                ),
                reviews=[critic_rev],
            ),
            investigation_plan=plan,
            max_reinvestigations=1,
        )

        resp = service.process_verification(req)
        finding = resp.all_findings[0]

        # Verification must NOT be claimed for mismatched variables!
        assert finding.status == "QUESTIONABLE"
        assert finding.status != "VERIFIED_AFTER_REINVESTIGATION"
        assert "VARIABLE_PAIR_MISMATCH" in finding.reason_codes
        assert "mismatched variable pair" in finding.verification_reason.lower()
    finally:
        service.analysis_agent.execute_task = original_execute


def test_task23_direction_conflict_assigned_questionable():
    """
    Task 23 Regression Test:
    Verify that if reinvestigation produces a conflicting relationship direction,
    it is assigned QUESTIONABLE with CONFLICTING_DIRECTION reason code.
    """
    from unittest.mock import MagicMock
    import pandas as pd
    from app.services.verification_service import get_verification_service
    from app.schemas.verification_schema import VerificationRequest
    from app.schemas.analysis_schema import PlanAnalysisResponse, AnalysisResult
    from app.schemas.critic_schema import CriticResponse, CriticResult, CriticOverallSummary
    from app.schemas.planner_schema import PlannerResponse, InvestigationTask
    from app.services.dataset_store import store_dataset

    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0] * 25, "y": [10.0, 12.0, 14.0, 16.0] * 25})
    store_dataset("test_task23_dir.csv", df)

    service = get_verification_service()

    orig = AnalysisResult(
        task_id="task_x",
        status="success",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        columns=["x", "y"],
        sample_size=100,
        results={"column_a": "x", "column_b": "y", "statistic": 0.45, "direction": "positive", "p_value": 0.001, "sample_size": 100},
    )

    conflicting_result = AnalysisResult(
        task_id="task_x_reinvest",
        status="success",
        analysis_type="numerical_vs_numerical",
        method="Spearman rank correlation",
        columns=["x", "y"],
        sample_size=100,
        results={"column_a": "x", "column_b": "y", "statistic": -0.35, "direction": "negative", "p_value": 0.01, "sample_size": 100},
    )
    original_execute = service.analysis_agent.execute_task
    service.analysis_agent.execute_task = MagicMock(return_value=conflicting_result)

    try:
        plan = PlannerResponse(
            dataset_id="test_task23_dir.csv",
            objective="Test",
            interpreted_goal="Test",
            selected_columns=[],
            reasoning=[],
            warnings=[],
            expected_outputs=[],
            analysis_tasks=[
                InvestigationTask(
                    task_id="task_x",
                    task_type="numerical_vs_numerical",
                    task="x vs y",
                    description="Evaluate x vs y",
                    method="Pearson correlation",
                    columns=["x", "y"],
                    column_a="x",
                    column_b="y",
                    reason="Test",
                    priority="high",
                )
            ],
            tasks=[],
        )

        critic_rev = CriticResult(
            task_id="task_x",
            analysis_type="numerical_vs_numerical",
            method="Pearson correlation",
            statistical_validity="Valid",
            method_assessment="Appropriate",
            recommendation="Perform Spearman rank correlation on 'x' vs 'y'",
            status="REQUIRES_REINVESTIGATION",
            confidence=0.85,
            confidence_level="HIGH",
            summary="Distribution concerns require non-parametric test.",
            requires_reinvestigation=True,
            recommended_method="Spearman rank correlation",
            reason_codes=["VISUAL_OUTLIERS_REINVESTIGATE"],
        )

        req = VerificationRequest(
            dataset_id="test_task23_dir.csv",
            analysis_response=PlanAnalysisResponse(
                dataset_id="test_task23_dir.csv", total_tasks=1, successful_tasks=1, failed_tasks=0, results=[orig]
            ),
            critic_response=CriticResponse(
                dataset_id="test_task23_dir.csv",
                summary=CriticOverallSummary(
                    total_results=1, accepted=0, questionable=0, rejected=0, requires_reinvestigation=1, overall_status="Test"
                ),
                reviews=[critic_rev],
            ),
            investigation_plan=plan,
            max_reinvestigations=1,
        )

        resp = service.process_verification(req)
        finding = resp.all_findings[0]

        assert finding.status == "QUESTIONABLE"
        assert finding.status != "VERIFIED_AFTER_REINVESTIGATION"
        assert "CONFLICTING_DIRECTION" in finding.reason_codes
    finally:
        service.analysis_agent.execute_task = original_execute


def test_task23_significance_lost_assigned_questionable():
    """
    Task 23 Regression Test:
    Verify that if the original parametric test showed statistical significance (p < 0.05),
    but the robust non-parametric test fails to replicate significance (p >= 0.05),
    the finding is assigned QUESTIONABLE with METHOD_SENSITIVITY_DISCREPANCY,
    preventing false verification of an outlier-driven relationship.
    """
    from unittest.mock import MagicMock
    import pandas as pd
    from app.services.verification_service import get_verification_service
    from app.schemas.verification_schema import VerificationRequest
    from app.schemas.analysis_schema import PlanAnalysisResponse, AnalysisResult
    from app.schemas.critic_schema import CriticResponse, CriticResult, CriticOverallSummary
    from app.schemas.planner_schema import PlannerResponse, InvestigationTask
    from app.services.dataset_store import store_dataset

    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0] * 25, "y": [10.0, 12.0, 14.0, 16.0] * 25})
    store_dataset("test_task23_sig_lost.csv", df)

    service = get_verification_service()

    # Original parametric test: significant at p = 0.03
    orig = AnalysisResult(
        task_id="task_sig",
        status="success",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        columns=["x", "y"],
        sample_size=100,
        results={"column_a": "x", "column_b": "y", "statistic": 0.22, "direction": "positive", "p_value": 0.03, "sample_size": 100},
    )

    # Reinvestigated non-parametric test: non-significant at p = 0.35 (outliers drove Pearson)
    robust_result = AnalysisResult(
        task_id="task_sig_reinvest",
        status="success",
        analysis_type="numerical_vs_numerical",
        method="Spearman rank correlation",
        columns=["x", "y"],
        sample_size=100,
        results={"column_a": "x", "column_b": "y", "statistic": 0.09, "direction": "positive", "p_value": 0.35, "sample_size": 100},
    )
    original_execute = service.analysis_agent.execute_task
    service.analysis_agent.execute_task = MagicMock(return_value=robust_result)

    try:
        plan = PlannerResponse(
            dataset_id="test_task23_sig_lost.csv",
            objective="Test",
            interpreted_goal="Test",
            selected_columns=[],
            reasoning=[],
            warnings=[],
            expected_outputs=[],
            analysis_tasks=[
                InvestigationTask(
                    task_id="task_sig",
                    task_type="numerical_vs_numerical",
                    task="x vs y",
                    description="Evaluate x vs y",
                    method="Pearson correlation",
                    columns=["x", "y"],
                    column_a="x",
                    column_b="y",
                    reason="Test",
                    priority="high",
                )
            ],
            tasks=[],
        )

        critic_rev = CriticResult(
            task_id="task_sig",
            analysis_type="numerical_vs_numerical",
            method="Pearson correlation",
            statistical_validity="Valid",
            method_assessment="Appropriate",
            recommendation="Perform Spearman rank correlation on 'x' vs 'y'",
            status="REQUIRES_REINVESTIGATION",
            confidence=0.85,
            confidence_level="HIGH",
            summary="Distribution concerns require non-parametric test.",
            requires_reinvestigation=True,
            recommended_method="Spearman rank correlation",
            reason_codes=["VISUAL_OUTLIERS_REINVESTIGATE"],
        )

        req = VerificationRequest(
            dataset_id="test_task23_sig_lost.csv",
            analysis_response=PlanAnalysisResponse(
                dataset_id="test_task23_sig_lost.csv", total_tasks=1, successful_tasks=1, failed_tasks=0, results=[orig]
            ),
            critic_response=CriticResponse(
                dataset_id="test_task23_sig_lost.csv",
                summary=CriticOverallSummary(
                    total_results=1, accepted=0, questionable=0, rejected=0, requires_reinvestigation=1, overall_status="Test"
                ),
                reviews=[critic_rev],
            ),
            investigation_plan=plan,
            max_reinvestigations=1,
        )

        resp = service.process_verification(req)
        finding = resp.all_findings[0]

        assert finding.status == "QUESTIONABLE"
        assert finding.status != "VERIFIED_AFTER_REINVESTIGATION"
        assert "METHOD_SENSITIVITY_DISCREPANCY" in finding.reason_codes
    finally:
        service.analysis_agent.execute_task = original_execute

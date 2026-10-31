"""
backend/tests/test_group_reinvestigation_fix.py
================================================
Focused offline unit tests for Critic-driven re-investigation & provenance fixes:
1. Multi-group and two-group deterministic method selection (Fix 1).
   - Composite strings like "One-way ANOVA or Kruskal-Wallis test" select ANOVA.
   - Pure "Kruskal-Wallis test" selects Kruskal-Wallis.
   - Composite strings like "Independent samples t-test or Mann-Whitney U test" select Welch's t-test.
   - Pure "Mann-Whitney U test" selects Mann-Whitney U.
   - Actual executed method is accurately recorded in result metadata.
2. Multi-group variable-pair validation (Fix 2).
   - Multi-group task with group_column and value_column does not falsely trigger VARIABLE_PAIR_MISMATCH.
3. Outlier sensitivity computational provenance (Fix 3).
   - _has_task_provenance() accepts valid outlier_sensitivity structures with outlier_count and baseline_metrics/iqr.
   - Rejects failed, empty, or uncomputed outlier results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.schemas.analysis_schema import AnalysisResult, PlanAnalysisResponse
from app.schemas.critic_schema import (
    CriticOverallSummary,
    CriticResponse,
    CriticResult,
)
from app.schemas.planner_schema import InvestigationTask
from app.schemas.verification_schema import VerificationRequest
from app.services.dataset_store import store_dataset, clear_cache
from app.services.evaluation_service import EvaluationService
from app.services.verification_service import VerificationService
from app.statistics.group_comparison import (
    execute_multi_group_comparison,
    execute_two_group_comparison,
)


# ==============================================================================
# Fix 1: Statistical Method Dispatch Tests
# ==============================================================================

def test_multi_group_dispatch_composite_vs_pure_kruskal():
    """Verify composite method selects One-way ANOVA while pure Kruskal selects Kruskal-Wallis."""
    np.random.seed(42)
    df = pd.DataFrame({
        "group": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
        "val": np.concatenate([
            np.random.normal(10, 2, 20),
            np.random.normal(15, 2, 20),
            np.random.normal(20, 2, 20),
        ]),
    })

    # Composite Planner method string -> must execute One-way ANOVA
    ok_comp, res_comp, _, _ = execute_multi_group_comparison(
        df, "group", "val", method="One-way ANOVA or Kruskal-Wallis test"
    )
    assert ok_comp is True
    assert res_comp["method"] == "One-way ANOVA"
    assert res_comp["actual_method"] == "One-way ANOVA"
    assert "F_statistic" in res_comp
    assert "H_statistic" not in res_comp
    assert res_comp["effect_size_type"] == "Eta-squared (η²)"

    # Pure Kruskal-Wallis recommendation -> must execute Kruskal-Wallis test
    ok_pure, res_pure, _, _ = execute_multi_group_comparison(
        df, "group", "val", method="Kruskal-Wallis test"
    )
    assert ok_pure is True
    assert res_pure["method"] == "Kruskal-Wallis test"
    assert res_pure["actual_method"] == "Kruskal-Wallis test"
    assert "H_statistic" in res_pure
    assert "F_statistic" not in res_pure
    assert res_pure["effect_size_type"] == "Epsilon-squared (ε²)"


def test_two_group_dispatch_composite_vs_pure_mann_whitney():
    """Verify composite method selects Welch's t-test while pure Mann-Whitney selects Mann-Whitney U."""
    np.random.seed(42)
    df = pd.DataFrame({
        "group": ["A"] * 20 + ["B"] * 20,
        "val": np.concatenate([
            np.random.normal(10, 2, 20),
            np.random.normal(15, 2, 20),
        ]),
    })

    # Composite Planner method string -> must execute Welch's t-test
    ok_comp, res_comp, _, _ = execute_two_group_comparison(
        df, "group", "val", method="Independent samples t-test or Mann-Whitney U test"
    )
    assert ok_comp is True
    assert res_comp["method"] == "Welch's t-test"
    assert res_comp["actual_method"] == "Welch's t-test"
    assert res_comp["effect_size_type"] == "Cohen's d"

    # Pure Mann-Whitney U recommendation -> must execute Mann-Whitney U test
    ok_pure, res_pure, _, _ = execute_two_group_comparison(
        df, "group", "val", method="Mann-Whitney U test"
    )
    assert ok_pure is True
    assert res_pure["method"] == "Mann-Whitney U test"
    assert res_pure["actual_method"] == "Mann-Whitney U test"
    assert res_pure["effect_size_type"] == "Rank-biserial correlation"


# ==============================================================================
# Fix 2: Group Comparison Variable-Pair Validation Test
# ==============================================================================

def test_multi_group_reinvestigation_does_not_trigger_false_variable_mismatch():
    """Verify multi-group tasks with group_column & value_column do not trigger false VARIABLE_PAIR_MISMATCH."""
    clear_cache()
    df = pd.DataFrame({
        "profession": ["Doctor"] * 20 + ["Artist"] * 20 + ["Engineer"] * 20,
        "spending_score": [50.0 + i for i in range(20)] + [30.0 + i for i in range(20)] + [40.0 + i for i in range(20)],
    })
    store_dataset("test_multigroup_reinvest.csv", df)

    service = VerificationService()

    # Initial ANOVA analysis where Planner specified group_column and value_column
    initial_analysis = AnalysisResult(
        task_id="t_multigroup",
        status="success",
        analysis_type="multiple_groups_vs_numerical",
        method="One-way ANOVA",
        columns=["profession", "spending_score"],
        sample_size=60,
        results={
            "statistic": 5.4321,
            "F_statistic": 5.4321,
            "p_value": 0.007,
            "p_value_formatted": "0.0070",
            "sample_size": 60,
            "group_column": "profession",
            "value_column": "spending_score",
            "effect_size": 0.16,
            "effect_size_type": "Eta-squared (η²)",
            "p_value_below_0_05": True,
        },
    )

    # Critic flags for Kruskal-Wallis re-investigation
    critic_rev = CriticResult(
        task_id="t_multigroup",
        analysis_type="multiple_groups_vs_numerical",
        method="One-way ANOVA",
        variables=["profession", "spending_score"],
        status="REQUIRES_REINVESTIGATION",
        confidence=0.85,
        confidence_level="HIGH",
        summary="ANOVA assumption concerns. Recommend Kruskal-Wallis.",
        evidence=["Group imbalance detected."],
        statistical_validity="Valid.",
        method_assessment="Appropriate.",
        recommendation="Perform Kruskal-Wallis test across 'profession' categories.",
        requires_reinvestigation=True,
        recommended_method="Kruskal-Wallis test",
    )

    req = VerificationRequest(
        dataset_id="test_multigroup_reinvest.csv",
        analysis_response=PlanAnalysisResponse(
            dataset_id="test_multigroup_reinvest.csv",
            total_tasks=1,
            successful_tasks=1,
            failed_tasks=0,
            results=[initial_analysis],
        ),
        critic_response=CriticResponse(
            dataset_id="test_multigroup_reinvest.csv",
            summary=CriticOverallSummary(
                total_results=1, accepted=0, questionable=0, rejected=0,
                requires_reinvestigation=1, overall_status="NEEDS_REVIEW"
            ),
            reviews=[critic_rev],
        ),
        max_reinvestigations=1,
    )

    resp = service.process_verification(req)

    assert resp.summary.total_reinvestigated == 1
    finding = resp.all_findings[0]

    # Crucial check: Must NOT have VARIABLE_PAIR_MISMATCH
    assert "VARIABLE_PAIR_MISMATCH" not in finding.reason_codes
    assert "mismatched variable pair" not in (finding.verification_reason or "").lower()
    assert finding.reinvestigated is True
    assert finding.method == "Kruskal-Wallis test"
    assert finding.status in ("VERIFIED", "VERIFIED_AFTER_REINVESTIGATION")


# ==============================================================================
# Fix 3: Outlier Sensitivity Provenance Tests
# ==============================================================================

def test_outlier_sensitivity_provenance_valid_metrics():
    """Verify _has_task_provenance accepts valid outlier sensitivity results with factual outputs."""
    eval_svc = EvaluationService()

    valid_outlier_result = {
        "task_id": "t_outlier",
        "status": "success",
        "analysis_type": "outlier_sensitivity",
        "method": "IQR fence rule and Z-score thresholding",
        "columns": ["income"],
        "sample_size": 100,
        "results": {
            "column": "income",
            "sample_size": 100,
            "outlier_count": 5,
            "outlier_percentage": 5.0,
            "q1": 50000.0,
            "q3": 90000.0,
            "iqr": 40000.0,
            "lower_fence": -10000.0,
            "upper_fence": 150000.0,
            "baseline_metrics": {"mean": 70000.0, "median": 68000.0, "std": 25000.0},
            "filtered_metrics": {"mean": 68000.0, "median": 67000.0, "std": 21000.0},
            "mean_shift": -2000.0,
            "mean_shift_pct": -2.86,
        },
    }

    assert eval_svc._has_task_provenance(valid_outlier_result) is True


def test_outlier_sensitivity_provenance_rejects_invalid_or_missing_metrics():
    """Verify _has_task_provenance rejects failed or uncomputed outlier sensitivity tasks."""
    eval_svc = EvaluationService()

    # 1. Failed task status -> False
    failed_task = {
        "task_id": "t_failed",
        "status": "failed",
        "analysis_type": "outlier_sensitivity",
        "results": {"outlier_count": 5, "iqr": 10.0},
    }
    assert eval_svc._has_task_provenance(failed_task) is False

    # 2. Results is not a dict -> False
    not_dict_res = {
        "task_id": "t_none",
        "status": "success",
        "analysis_type": "outlier_sensitivity",
        "results": None,
    }
    assert eval_svc._has_task_provenance(not_dict_res) is False

    # 3. outlier_count is None (uncomputed) -> False
    uncomputed_outliers = {
        "task_id": "t_uncomp",
        "status": "success",
        "analysis_type": "outlier_sensitivity",
        "results": {"column": "income", "outlier_count": None},
    }
    assert eval_svc._has_task_provenance(uncomputed_outliers) is False


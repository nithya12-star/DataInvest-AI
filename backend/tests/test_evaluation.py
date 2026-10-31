# backend/tests/test_evaluation.py
"""
Unit and integration tests for Phase 7: System Evaluation & Quality Scorecard.
Validates scoring calculations, dimensional bounds, benchmark compliance,
zero-hallucination guarantees, and FastAPI /api/evaluate integration.
"""

import json
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.evaluation_service import evaluation_service
from app.schemas.evaluation_schema import EvaluationResponse, EvaluationRequest


@pytest.fixture
def sample_pipeline_data():
    """Mock complete pipeline state across Phases 1 to 6."""
    profile = {
        "dataset_id": "test_dataset.csv",
        "overview": {"n_rows": 100, "n_cols": 4, "memory_bytes": 4000},
        "column_types": {"numerical": ["salary", "experience", "age"], "categorical": ["department"], "datetime": []},
        "missing_stats": {"columns": {"salary": 0, "experience": 0, "age": 0, "department": 0}, "total_missing": 0, "missing_cells_pct": 0.0},
        "numerical_stats": {"salary": {"mean": 75000, "median": 70000, "std": 15000, "skewness": 0.3}},
        "outliers": {"salary": {"count": 2, "outlier_percentage": 2.0}},
        "data_quality": {"completeness_score": 100.0, "duplicate_rows": 0, "duplicate_rows_pct": 0.0},
    }

    plan = {
        "dataset_id": "test_dataset.csv",
        "objective": "Investigate salary determinants.",
        "target_variables": ["salary"],
        "analysis_tasks": [
            {"task_id": "task_1", "method": "pearson_correlation", "variables": ["salary", "experience"]},
            {"task_id": "task_2", "method": "two_group_comparison", "variables": ["department", "salary"]},
        ],
    }

    analysis = {
        "dataset_id": "test_dataset.csv",
        "results": [
            {
                "task_id": "task_1",
                "method": "pearson_correlation",
                "variables": ["salary", "experience"],
                "status": "SUCCESS",
                "statistic": 0.85,
                "statistic_name": "r",
                "p_value": 0.0001,
                "effect_size": 0.85,
                "sample_size": 100,
                "bivariate": {"correlation": 0.85, "p_value": 0.0001},
            },
            {
                "task_id": "task_2",
                "method": "two_group_comparison",
                "variables": ["department", "salary"],
                "status": "SUCCESS",
                "statistic": 3.42,
                "statistic_name": "t",
                "p_value": 0.002,
                "effect_size": 0.72,
                "sample_size": 100,
                "two_group": {"cohens_d": 0.72, "p_value": 0.002},
            },
        ],
    }

    critic = {
        "reviews": [
            {
                "task_id": "task_1",
                "decision": "ACCEPTED",
                "confidence_score": 0.95,
                "critique": "Assumptions met.",
            },
            {
                "task_id": "task_2",
                "decision": "ACCEPTED",
                "confidence_score": 0.90,
                "critique": "Adequate sample size.",
            },
        ]
    }

    verification = {
        "verified_findings": [
            {
                "finding_id": "vf_1",
                "task_id": "task_1",
                "verification_status": "VERIFIED",
                "confidence_score": 0.95,
                "method": "pearson_correlation",
                "variables": ["salary", "experience"],
                "statistic": 0.85,
                "statistic_name": "r",
                "p_value": 0.0001,
                "effect_size": 0.85,
                "sample_size": 100,
            },
            {
                "finding_id": "vf_2",
                "task_id": "task_2",
                "verification_status": "VERIFIED",
                "confidence_score": 0.90,
                "method": "two_group_comparison",
                "variables": ["department", "salary"],
                "statistic": 3.42,
                "statistic_name": "t",
                "p_value": 0.002,
                "effect_size": 0.72,
                "sample_size": 100,
            },
        ],
        "summary": {
            "total_findings": 2,
            "verified": 2,
            "verified_after_reinvestigation": 0,
            "questionable": 0,
            "rejected": 0,
            "total_reinvestigated": 0,
        },
    }

    insights = {
        "summary": {
            "total_verified_findings": 2,
            "total_insights": 2,
            "strong_insights_count": 2,
            "moderate_insights_count": 0,
            "weak_insights_count": 0,
        },
        "insights": [
            {
                "insight_id": "ins_1",
                "title": "Strong Positive Association: salary and experience",
                "text": "A strong positive linear association was observed between salary and experience (r = 0.850, p < 0.001).",
                "strength": "STRONG",
                "verification_status": "VERIFIED",
                "confidence_score": 0.95,
                "practical_implication": "Experience is strongly correlated with salary levels.",
                "evidence": {"method": "Pearson correlation", "variables": ["salary", "experience"]},
            },
            {
                "insight_id": "ins_2",
                "title": "Significant Group Difference: department on salary",
                "text": "Statistically significant differences in salary were observed across department groups (d = 0.720, p = 0.002).",
                "strength": "STRONG",
                "verification_status": "VERIFIED",
                "confidence_score": 0.90,
                "practical_implication": "Department-level compensation variations observed.",
                "evidence": {"method": "Two-group comparison", "variables": ["department", "salary"]},
            },
        ],
    }

    return {
        "profile": profile,
        "plan": plan,
        "analysis": analysis,
        "critic": critic,
        "verification": verification,
        "insights": insights,
    }


def test_evaluation_service_full_pipeline(sample_pipeline_data):
    """Test deterministic evaluation calculation across all dimensions."""
    resp = evaluation_service.evaluate_pipeline(
        profile=sample_pipeline_data["profile"],
        plan=sample_pipeline_data["plan"],
        analysis=sample_pipeline_data["analysis"],
        critic=sample_pipeline_data["critic"],
        verification=sample_pipeline_data["verification"],
        insights=sample_pipeline_data["insights"],
    )

    assert isinstance(resp, EvaluationResponse)
    assert resp.summary.overall_score >= 90.0
    assert "A+" in resp.summary.overall_grade
    assert resp.summary.total_tasks_planned == 2
    assert resp.summary.total_tasks_executed == 2
    assert resp.summary.total_findings_verified == 2
    assert resp.summary.total_insights_generated == 2
    assert resp.summary.zero_hallucination_verified is True
    assert len(resp.dimensions) == 4
    assert len(resp.benchmark_checks) >= 6
    assert len(resp.detailed_metrics) >= 12


def test_dimension_scores_within_bounds(sample_pipeline_data):
    """Verify that each dimension score is bounded in [0, 25] and overall in [0, 100]."""
    resp = evaluation_service.evaluate_pipeline(**sample_pipeline_data)

    for dim in resp.dimensions:
        assert 0.0 <= dim.score <= 25.0
        assert 0.0 <= dim.percentage <= 100.0
        assert dim.max_score == 25.0
        assert dim.grade in ["A+", "A", "B+", "B", "C", "D"]
        assert len(dim.metrics) > 0

    assert 0.0 <= resp.summary.overall_score <= 100.0


def test_zero_hallucination_and_benchmark_checks(sample_pipeline_data):
    """Verify AutoMind benchmark compliance checks."""
    resp = evaluation_service.evaluate_pipeline(**sample_pipeline_data)

    check_names = [c.name for c in resp.benchmark_checks]
    assert "Zero-Hallucination Empirical Guarantee" in check_names
    assert "Parametric Assumption Auditing" in check_names
    assert "Targeted Non-Parametric Fallbacks" in check_names
    assert "Non-Causal Language Guard" in check_names
    assert "Standardized Effect Size Reporting" in check_names
    assert "End-to-End Evidence Traceability" in check_names

    checks_by_id = {c.check_id: c for c in resp.benchmark_checks}
    assert checks_by_id["chk_zero_hallucination"].passed is True
    assert checks_by_id["chk_assumption_audits"].passed is True
    # In sample_pipeline_data, no assumption violations occurred so no fallback was executed;
    # the conservative evidence-based rule correctly reports False rather than fabricating a pass.
    assert checks_by_id["chk_robust_fallbacks"].passed is False
    assert "No fallback execution recorded" in checks_by_id["chk_robust_fallbacks"].detail
    assert checks_by_id["chk_causality_guard"].passed is True
    assert checks_by_id["chk_effect_size_reporting"].passed is True
    assert checks_by_id["chk_full_traceability"].passed is True

    for check in resp.benchmark_checks:
        assert len(check.detail) > 10


def test_causal_keyword_penalty(sample_pipeline_data):
    """Verify that causal claims in insight text are penalized."""
    # Inject causal claims
    bad_insights = dict(sample_pipeline_data["insights"])
    bad_insights["insights"] = [
        {
            "insight_id": "ins_bad",
            "title": "Experience causes salary growth",
            "text": "Increasing experience drives higher compensation and causes promotions.",
            "strength": "STRONG",
            "verification_status": "VERIFIED",
            "confidence_score": 0.90,
            "practical_implication": "Experience leads to higher salary.",
            "evidence": {"method": "Pearson", "variables": ["salary", "experience"]},
        }
    ]

    resp = evaluation_service.evaluate_pipeline(
        profile=sample_pipeline_data["profile"],
        plan=sample_pipeline_data["plan"],
        analysis=sample_pipeline_data["analysis"],
        critic=sample_pipeline_data["critic"],
        verification=sample_pipeline_data["verification"],
        insights=bad_insights,
    )

    dim4 = next(d for d in resp.dimensions if d.dimension_key == "insight_actionability")
    m_causal = next(m for m in dim4.metrics if m.metric_id == "m_insight_causality_safety")

    assert m_causal.status == "NEEDS_IMPROVEMENT"
    assert dim4.score < 25.0


def test_evaluation_with_minimal_profile_only():
    """Verify graceful scoring when only profile is provided."""
    profile = {
        "overview": {"n_rows": 50, "n_cols": 2},
        "column_types": {"numerical": ["x"], "categorical": ["y"], "datetime": []},
        "missing_stats": {"columns": {"x": 0, "y": 0}, "total_missing": 0},
        "numerical_stats": {"x": {"mean": 10}},
        "outliers": {"x": {"count": 0}},
        "data_quality": {"completeness_score": 100.0},
    }

    resp = evaluation_service.evaluate_pipeline(profile=profile)

    assert isinstance(resp, EvaluationResponse)
    assert 0.0 <= resp.summary.overall_score <= 100.0
    assert len(resp.dimensions) == 4


def test_evaluation_with_reinvestigated_findings(sample_pipeline_data):
    """Verify that re-investigation tracking correctly scores statistical rigor."""
    verification = dict(sample_pipeline_data["verification"])
    verification["summary"] = {
        "total_findings": 4,
        "verified": 2,
        "verified_after_reinvestigation": 2,
        "questionable": 0,
        "rejected": 0,
        "total_reinvestigated": 2,
    }

    resp = evaluation_service.evaluate_pipeline(
        profile=sample_pipeline_data["profile"],
        plan=sample_pipeline_data["plan"],
        analysis=sample_pipeline_data["analysis"],
        critic=sample_pipeline_data["critic"],
        verification=verification,
        insights=sample_pipeline_data["insights"],
    )

    dim2 = next(d for d in resp.dimensions if d.dimension_key == "statistical_rigor")
    m_fallback = next(m for m in dim2.metrics if m.metric_id == "m_rigor_fallback_rate")
    assert "2 tests flagged for assumption violations; 2 successfully resolved" in m_fallback.description


def test_json_serializability(sample_pipeline_data):
    """Verify that the evaluation response serializes cleanly to JSON."""
    resp = evaluation_service.evaluate_pipeline(**sample_pipeline_data)
    json_str = resp.model_dump_json()
    parsed = json.loads(json_str)

    assert "summary" in parsed
    assert "dimensions" in parsed
    assert "benchmark_checks" in parsed
    assert "detailed_metrics" in parsed


def test_api_evaluate_endpoint_success(sample_pipeline_data):
    """Test FastAPI POST /api/evaluate endpoint."""
    client = TestClient(app)
    response = client.post("/api/evaluate", json=sample_pipeline_data)

    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert data["summary"]["overall_score"] >= 80.0
    assert len(data["dimensions"]) == 4
    assert len(data["benchmark_checks"]) >= 6


def test_api_evaluate_missing_profile():
    """Test FastAPI POST /api/evaluate rejects empty payload."""
    client = TestClient(app)
    response = client.post("/api/evaluate", json={})

    assert response.status_code == 422


def test_real_profiler_output_completeness():
    """Verify that actual profiler schema keys (missing, duplicates, cardinality) score 100%."""
    real_profile = {
        "dataset_id": "real_data.csv",
        "overview": {"n_rows": 500, "n_cols": 5},
        "column_types": {"numerical": ["age", "income"], "categorical": ["gender"], "datetime": []},
        "missing": {"missing_by_column": {"age": 0, "income": 5, "gender": 0}, "total_missing_cells": 5, "missing_cell_percentage": 0.2},
        "duplicates": {"duplicate_row_count": 0, "duplicate_row_percentage": 0.0},
        "cardinality": {"unique_counts": {"age": 45, "income": 300, "gender": 2}},
        "numerical_stats": {"age": {"mean": 35.0, "median": 34.0, "std": 10.0}},
        "outliers": {"age": {"count": 1, "outlier_percentage": 0.2}},
    }
    resp = evaluation_service.evaluate_pipeline(profile=real_profile)
    dim1 = next(d for d in resp.dimensions if d.dimension_key == "execution_reliability")
    m_prof = next(m for m in dim1.metrics if m.metric_id == "m_exec_profiling")
    assert m_prof.value == "100.0%"
    assert m_prof.status == "EXCELLENT"


def test_nested_analysis_results_effect_size(sample_pipeline_data):
    """Verify that effect-size coverage inspects nested results dict (Phase 3 schema)."""
    nested_analysis = {
        "dataset_id": "test_dataset.csv",
        "results": [
            {
                "task_id": "task_1",
                "method": "pearson_correlation",
                "variables": ["x", "y"],
                "status": "SUCCESS",
                "results": {
                    "correlation_coefficient": 0.78,
                    "p_value": 0.001,
                    "sample_size": 100,
                },
            },
            {
                "task_id": "task_2",
                "method": "chi_square",
                "variables": ["cat_a", "cat_b"],
                "status": "SUCCESS",
                "results": {
                    "cramer_v": 0.42,
                    "chi2_statistic": 15.6,
                    "p_value": 0.0005,
                },
            },
        ],
    }
    resp = evaluation_service.evaluate_pipeline(
        profile=sample_pipeline_data["profile"],
        plan=sample_pipeline_data["plan"],
        analysis=nested_analysis,
        critic=sample_pipeline_data["critic"],
        verification=sample_pipeline_data["verification"],
    )
    dim2 = next(d for d in resp.dimensions if d.dimension_key == "statistical_rigor")
    m_eff = next(m for m in dim2.metrics if m.metric_id == "m_rigor_effect_size_coverage")
    assert m_eff.value == "100.0%"
    assert m_eff.status == "EXCELLENT"


def test_key_insights_schema_support(sample_pipeline_data):
    """Verify that InsightResponse with 'key_insights' is correctly detected and scored."""
    key_insights_data = {
        "key_insights": [
            {
                "insight_id": "ins_1",
                "finding_id": "vf_1",
                "type": "CORRELATION",
                "title": "Positive correlation between salary and experience",
                "insight": "Salary associates positively with experience (r=0.85).",
                "importance": "HIGH",
                "recommendation": "Review compensation bands based on tenure.",
                "evidence": {"method": "Pearson correlation", "variables": ["salary", "experience"]},
            }
        ],
        "summary_statistics": {"total_insights": 1},
    }
    resp = evaluation_service.evaluate_pipeline(
        profile=sample_pipeline_data["profile"],
        plan=sample_pipeline_data["plan"],
        analysis=sample_pipeline_data["analysis"],
        critic=sample_pipeline_data["critic"],
        verification=sample_pipeline_data["verification"],
        insights=key_insights_data,
    )
    assert resp.summary.total_insights_generated == 1
    dim4 = next(d for d in resp.dimensions if d.dimension_key == "insight_actionability")
    assert dim4.score > 15.0


def test_empty_insights_no_free_marks(sample_pipeline_data):
    """Verify that empty insight lists receive 0.0 actionability marks rather than full 25.0."""
    resp = evaluation_service.evaluate_pipeline(
        profile=sample_pipeline_data["profile"],
        plan=sample_pipeline_data["plan"],
        analysis=sample_pipeline_data["analysis"],
        critic=sample_pipeline_data["critic"],
        verification=sample_pipeline_data["verification"],
        insights={"key_insights": []},
    )
    assert resp.summary.total_insights_generated == 0
    dim4 = next(d for d in resp.dimensions if d.dimension_key == "insight_actionability")
    assert dim4.score == 0.0
    for metric in dim4.metrics:
        assert "0.0%" in str(metric.value)
        assert metric.status == "NEEDS_IMPROVEMENT"


def test_fallback_resolution_evidence_based(sample_pipeline_data):
    """Verify that unresolved assumption violations are penalized in fallback score."""
    unresolved_verification = {
        "verified_findings": [],
        "summary": {
            "total_findings": 2,
            "verified": 0,
            "verified_after_reinvestigation": 0,
            "critic_flagged": 2,
            "total_reinvestigated": 0,
        },
    }
    critic_with_flags = {
        "reviews": [
            {"task_id": "t1", "decision": "FLAGGED", "critique": "Normality violated."},
            {"task_id": "t2", "decision": "FLAGGED", "critique": "Heteroscedasticity."},
        ]
    }
    resp = evaluation_service.evaluate_pipeline(
        profile=sample_pipeline_data["profile"],
        plan=sample_pipeline_data["plan"],
        analysis=sample_pipeline_data["analysis"],
        critic=critic_with_flags,
        verification=unresolved_verification,
    )
    dim2 = next(d for d in resp.dimensions if d.dimension_key == "statistical_rigor")
    m_fb = next(m for m in dim2.metrics if m.metric_id == "m_rigor_fallback_rate")
    assert m_fb.value == "0.0%"
    assert m_fb.status == "NEEDS_IMPROVEMENT"


def test_chk_robust_fallbacks_evidence_based(sample_pipeline_data):
    """Verify chk_robust_fallbacks passes only when fallback execution is actually recorded."""
    # 1. Fallback executed and recorded -> passes
    executed_verification = dict(sample_pipeline_data["verification"])
    executed_verification["summary"] = {
        "total_findings": 2,
        "verified": 2,
        "verified_after_reinvestigation": 1,
        "questionable": 0,
        "rejected": 0,
        "total_reinvestigated": 1,
    }
    resp_passed = evaluation_service.evaluate_pipeline(
        profile=sample_pipeline_data["profile"],
        plan=sample_pipeline_data["plan"],
        analysis=sample_pipeline_data["analysis"],
        critic=sample_pipeline_data["critic"],
        verification=executed_verification,
        insights=sample_pipeline_data["insights"],
    )
    chk = next(c for c in resp_passed.benchmark_checks if c.check_id == "chk_robust_fallbacks")
    assert chk.passed is True
    assert "Targeted non-parametric fallback executed and recorded" in chk.detail

    # 2. Critic flagged violations but no fallback executed -> fails with violation
    critic_with_flags = {
        "reviews": [
            {"task_id": "task_1", "decision": "FLAGGED", "requires_reinvestigation": True},
        ]
    }
    unresolved_verification = dict(sample_pipeline_data["verification"])
    unresolved_verification["summary"] = {
        "total_findings": 2,
        "verified": 1,
        "verified_after_reinvestigation": 0,
        "total_reinvestigated": 0,
    }
    resp_violated = evaluation_service.evaluate_pipeline(
        profile=sample_pipeline_data["profile"],
        plan=sample_pipeline_data["plan"],
        analysis=sample_pipeline_data["analysis"],
        critic=critic_with_flags,
        verification=unresolved_verification,
        insights=sample_pipeline_data["insights"],
    )
    chk_v = next(c for c in resp_violated.benchmark_checks if c.check_id == "chk_robust_fallbacks")
    assert chk_v.passed is False
    assert "Fallback requirement violated" in chk_v.detail

    # 3. No fallback needed or executed -> conservative failure documented
    resp_none = evaluation_service.evaluate_pipeline(**sample_pipeline_data)
    chk_none = next(c for c in resp_none.benchmark_checks if c.check_id == "chk_robust_fallbacks")
    assert chk_none.passed is False
    assert "No fallback execution recorded" in chk_none.detail


def test_chk_causality_guard_evidence_based(sample_pipeline_data):
    """Verify chk_causality_guard inspects actual text and does not pass on empty insights."""
    # 1. Clean observational insights -> passes
    resp_clean = evaluation_service.evaluate_pipeline(**sample_pipeline_data)
    chk_clean = next(c for c in resp_clean.benchmark_checks if c.check_id == "chk_causality_guard")
    assert chk_clean.passed is True
    assert "0 causal overclaims detected" in chk_clean.detail

    # 2. Causal overclaim keywords in insights -> fails
    causal_insights = {
        "insights": [
            {
                "insight_id": "ins_1",
                "title": "Salary cause",
                "text": "Experience causes an increase in salary.",
                "evidence": {"task_id": "task_1"},
            }
        ]
    }
    resp_causal = evaluation_service.evaluate_pipeline(
        profile=sample_pipeline_data["profile"],
        plan=sample_pipeline_data["plan"],
        analysis=sample_pipeline_data["analysis"],
        critic=sample_pipeline_data["critic"],
        verification=sample_pipeline_data["verification"],
        insights=causal_insights,
    )
    chk_causal = next(c for c in resp_causal.benchmark_checks if c.check_id == "chk_causality_guard")
    assert chk_causal.passed is False
    assert "Non-causal language guard violated" in chk_causal.detail

    # 3. Empty insights -> fails (no free pass for absent insights)
    resp_empty = evaluation_service.evaluate_pipeline(
        profile=sample_pipeline_data["profile"],
        plan=sample_pipeline_data["plan"],
        analysis=sample_pipeline_data["analysis"],
        critic=sample_pipeline_data["critic"],
        verification=sample_pipeline_data["verification"],
        insights={"insights": []},
    )
    chk_empty = next(c for c in resp_empty.benchmark_checks if c.check_id == "chk_causality_guard")
    assert chk_empty.passed is False
    assert "No analytical insights available" in chk_empty.detail


def test_chk_effect_size_reporting_applicability(sample_pipeline_data):
    """Verify effect size check distinguishes applicable tests from non-applicable descriptive tasks."""
    # 1. Applicable tasks report effect size -> passes
    resp_valid = evaluation_service.evaluate_pipeline(**sample_pipeline_data)
    chk_valid = next(c for c in resp_valid.benchmark_checks if c.check_id == "chk_effect_size_reporting")
    assert chk_valid.passed is True
    assert "100.0% of applicable statistical tasks" in chk_valid.detail

    # 2. Applicable task missing effect size -> fails
    missing_eff_analysis = {
        "results": [
            {
                "task_id": "task_1",
                "analysis_type": "bivariate_correlation",
                "method": "pearson_correlation",
                "variables": ["salary", "experience"],
                "status": "SUCCESS",
                "statistic": 0.85,
                "p_value": 0.001,
                # effect_size omitted
            }
        ]
    }
    resp_missing = evaluation_service.evaluate_pipeline(
        profile=sample_pipeline_data["profile"],
        plan=sample_pipeline_data["plan"],
        analysis=missing_eff_analysis,
        critic=sample_pipeline_data["critic"],
        verification=sample_pipeline_data["verification"],
        insights=sample_pipeline_data["insights"],
    )
    chk_missing = next(c for c in resp_missing.benchmark_checks if c.check_id == "chk_effect_size_reporting")
    assert chk_missing.passed is False
    assert "Standardized effect size reporting incomplete" in chk_missing.detail

    # 3. Non-applicable tasks (e.g. descriptive/distribution summary) -> passes with non-applicable documentation
    descriptive_analysis = {
        "results": [
            {
                "task_id": "desc_1",
                "analysis_type": "univariate_descriptive",
                "method": "summary_statistics",
                "variables": ["salary"],
                "status": "SUCCESS",
                "results": {"mean": 50000, "std": 10000},
            }
        ]
    }
    resp_desc = evaluation_service.evaluate_pipeline(
        profile=sample_pipeline_data["profile"],
        plan=sample_pipeline_data["plan"],
        analysis=descriptive_analysis,
        critic=sample_pipeline_data["critic"],
        verification=sample_pipeline_data["verification"],
        insights=sample_pipeline_data["insights"],
    )
    chk_desc = next(c for c in resp_desc.benchmark_checks if c.check_id == "chk_effect_size_reporting")
    assert chk_desc.passed is True
    assert "not statistically applicable" in chk_desc.detail

    # 4. Genuinely empty analysis -> fails
    resp_empty = evaluation_service.evaluate_pipeline(
        profile=sample_pipeline_data["profile"],
        plan=sample_pipeline_data["plan"],
        analysis={"results": []},
    )
    chk_empty = next(c for c in resp_empty.benchmark_checks if c.check_id == "chk_effect_size_reporting")
    assert chk_empty.passed is False
    assert "No executed statistical results available" in chk_empty.detail


def test_chk_full_traceability_evidence_based(sample_pipeline_data):
    """Verify chk_full_traceability distinguishes empty analysis, missing verification, and bound evidence."""
    # 1. Full evidence binding -> passes
    resp_bound = evaluation_service.evaluate_pipeline(**sample_pipeline_data)
    chk_bound = next(c for c in resp_bound.benchmark_checks if c.check_id == "chk_full_traceability")
    assert chk_bound.passed is True
    assert "Full evidence traceability verified" in chk_bound.detail

    # 2. Insights missing evidence or finding_id -> fails
    unbound_insights = {
        "insights": [
            {
                "insight_id": "ins_unbound",
                "title": "Unbound Claim",
                "text": "Some claim without any evidence field or finding link.",
            }
        ]
    }
    resp_unbound = evaluation_service.evaluate_pipeline(
        profile=sample_pipeline_data["profile"],
        plan=sample_pipeline_data["plan"],
        analysis=sample_pipeline_data["analysis"],
        critic=sample_pipeline_data["critic"],
        verification=sample_pipeline_data["verification"],
        insights=unbound_insights,
    )
    chk_unbound = next(c for c in resp_unbound.benchmark_checks if c.check_id == "chk_full_traceability")
    assert chk_unbound.passed is False
    assert "Evidence traceability incomplete" in chk_unbound.detail

    # 3. Missing verification and insight stages (e.g. baseline or omitted stages) -> fails
    resp_no_verif = evaluation_service.evaluate_pipeline(
        profile=sample_pipeline_data["profile"],
        plan=sample_pipeline_data["plan"],
        analysis=sample_pipeline_data["analysis"],
        critic={},
        verification={},
        insights={},
    )
    chk_no_verif = next(c for c in resp_no_verif.benchmark_checks if c.check_id == "chk_full_traceability")
    assert chk_no_verif.passed is False
    assert "Traceability evidence unavailable" in chk_no_verif.detail

    # 4. Genuinely empty analysis -> fails with empty pipeline message
    resp_empty = evaluation_service.evaluate_pipeline(
        profile=sample_pipeline_data["profile"],
        plan=sample_pipeline_data["plan"],
        analysis={"results": []},
        critic={},
        verification={},
        insights={},
    )
    chk_empty = next(c for c in resp_empty.benchmark_checks if c.check_id == "chk_full_traceability")
    assert chk_empty.passed is False
    assert "Empty analysis pipeline" in chk_empty.detail


def test_chk_zero_hallucination_computation_provenance(sample_pipeline_data):
    """Verify chk_zero_hallucination is represented as a computation-provenance check."""
    # 1. Full deterministic provenance -> passes and clearly documents provenance
    resp = evaluation_service.evaluate_pipeline(**sample_pipeline_data)
    chk = next(c for c in resp.benchmark_checks if c.check_id == "chk_zero_hallucination")
    assert chk.passed is True
    assert "Deterministic computation provenance verified" in chk.detail
    assert "computation-provenance check" in chk.detail
    assert "does not constitute external ground-truth auditing" in chk.detail

    # 2. Failed task or missing provenance -> fails
    failed_analysis = {
        "results": [
            {"task_id": "t1", "status": "failed", "error": "Singular matrix"}
        ]
    }
    resp_failed = evaluation_service.evaluate_pipeline(
        profile=sample_pipeline_data["profile"],
        plan=sample_pipeline_data["plan"],
        analysis=failed_analysis,
    )
    chk_failed = next(c for c in resp_failed.benchmark_checks if c.check_id == "chk_zero_hallucination")
    assert chk_failed.passed is False
    assert "Computation provenance check failed" in chk_failed.detail

    # 3. Genuinely empty results -> fails
    resp_empty = evaluation_service.evaluate_pipeline(
        profile=sample_pipeline_data["profile"],
        plan=sample_pipeline_data["plan"],
        analysis={"results": []},
    )
    chk_empty = next(c for c in resp_empty.benchmark_checks if c.check_id == "chk_zero_hallucination")
    assert chk_empty.passed is False


def test_chk_zero_hallucination_categorical_provenance(sample_pipeline_data):
    """Verify categorical distribution tasks with frequencies/proportions establish valid provenance without fabricated statistics."""
    categorical_analysis = {
        "dataset_id": "bank-full.csv",
        "results": [
            {
                "task_id": "task_dist_cat",
                "method": "Descriptive summary & distribution assessment",
                "analysis_type": "distribution_analysis",
                "variables": ["y"],
                "status": "SUCCESS",
                "sample_size": 45211,
                # No fabricated scalar hypothesis test statistic
                "statistic": None,
                "p_value": None,
                "results": {
                    "column": "y",
                    "variable_type": "categorical",
                    "sample_size": 45211,
                    "num_categories": 2,
                    "mode": "no",
                    "frequencies": {"no": 39922, "yes": 5289},
                    "proportions": {"no": 0.883, "yes": 0.117},
                    "percentages": {"no": 88.3, "yes": 11.7},
                    "is_imbalanced": True,
                },
            }
        ]
    }
    resp = evaluation_service.evaluate_pipeline(
        profile=sample_pipeline_data["profile"],
        plan={"analysis_tasks": [{"task_id": "task_dist_cat"}]},
        analysis=categorical_analysis,
    )
    assert resp.summary.zero_hallucination_verified is True
    dim3 = next(d for d in resp.dimensions if d.dimension_key == "verification_grounding")
    m_prov = next(m for m in dim3.metrics if m.metric_id == "m_verif_zero_hallucination")
    assert m_prov.value == "100.0%"
    assert m_prov.status == "EXCELLENT"
    chk_zh = next(c for c in resp.benchmark_checks if c.check_id == "chk_zero_hallucination")
    assert chk_zh.passed is True






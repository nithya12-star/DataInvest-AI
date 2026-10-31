"""
backend/tests/test_insights.py
==============================
Unit and integration tests for DataInvest AI — Phase 6: Insight Generation.
Tests cover:
1. Verified finding produces insight
2. Rejected finding does not produce confirmed insight
3. Questionable finding does not produce confirmed insight
4. Actual statistics are preserved (zero fabrication)
5. No numerical values are fabricated
6. Deterministic mode works without external LLM
7. Multiple findings generate multiple ranked insights
8. Duplicate insights on same variable pair are handled
9. Evidence remains traceable to source
10. Causal language is avoided
11. Limitations are generated appropriately
12. Strength classification is deterministic
13. One failed/malformed finding does not crash the request (failure isolation)
14. Empty verified findings are handled gracefully
15. API POST /api/insights endpoint tests
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.insight_schema import InsightRequest, InsightResponse
from app.schemas.verification_schema import (
    ReinvestigationRecord,
    ResultComparison,
    VerificationResponse,
    VerificationSummary,
    VerifiedFinding,
)
from app.services.insight_service import InsightService, get_insight_service
from app.schemas.analysis_schema import AnalysisResult
from app.schemas.critic_schema import CriticResult

client = TestClient(app)


def _make_sample_verified_finding(
    finding_id="finding_1",
    task_id="T1",
    variables=None,
    method="spearman_correlation",
    status="VERIFIED",
    statistic=0.82,
    p_value=0.001,
    n=50,
    effect_size=0.82,
    reinvestigated=False,
    reinvestigation_record=None,
    limitations=None,
) -> VerifiedFinding:
    if variables is None:
        variables = ["experience", "salary"]
    return VerifiedFinding(
        finding_id=finding_id,
        task_id=task_id,
        variables=variables,
        method=method,
        status=status,
        confidence=1.0,
        confidence_level="HIGH",
        summary=f"Strong positive correlation observed between {variables[0]} and {variables[1]}.",
        statistical_evidence={
            "statistic": statistic,
            "correlation": statistic,
            "p_value": p_value,
            "sample_size": n,
            "effect_size": effect_size,
        },
        critic_assessment="Empirical verification passed with strong effect size.",
        reinvestigated=reinvestigated,
        reinvestigation_record=reinvestigation_record,
        verification_reason="Statistically significant with robust effect size.",
        limitations=limitations or [],
        reason_codes=["VERIFIED_OK"],
    )


# ── 1. Verified finding produces insight ──────────────────────────────────────


def test_verified_finding_produces_insight():
    service = InsightService()
    vf = _make_sample_verified_finding(
        variables=["experience", "salary"],
        statistic=0.78,
        p_value=0.0002,
        n=45,
    )
    req = InsightRequest(verified_findings=[vf], dataset_id="test.csv")
    res = service.generate_insights(req)

    assert isinstance(res, InsightResponse)
    assert len(res.key_insights) == 1
    insight = res.key_insights[0]
    assert insight.source_task_id == "T1"
    assert insight.verification_status == "VERIFIED"
    assert "experience" in insight.insight.lower()
    assert "salary" in insight.insight.lower()
    assert insight.evidence.statistic == 0.78
    assert insight.evidence.p_value == 0.0002
    assert insight.evidence.sample_size == 45


# ── 2. Rejected finding does not produce confirmed insight ────────────────────


def test_rejected_finding_excluded_from_insights():
    service = InsightService()
    rf = _make_sample_verified_finding(
        finding_id="finding_rej",
        task_id="T_REJ",
        status="REJECTED",
        method="pearson_correlation",
    )
    req = InsightRequest(verified_findings=[rf], dataset_id="test.csv")
    res = service.generate_insights(req)

    assert len(res.key_insights) == 0
    assert len(res.unresolved_findings) == 1
    assert res.unresolved_findings[0].status == "REJECTED"
    assert res.unresolved_findings[0].task_id == "T_REJ"


# ── 3. Questionable finding does not produce confirmed insight ────────────────


def test_questionable_finding_excluded_from_insights():
    service = InsightService()
    qf = _make_sample_verified_finding(
        finding_id="finding_q",
        task_id="T_Q",
        status="QUESTIONABLE",
        method="welchs_t_test",
    )
    req = InsightRequest(verified_findings=[qf], dataset_id="test.csv")
    res = service.generate_insights(req)

    assert len(res.key_insights) == 0
    assert len(res.unresolved_findings) == 1
    assert res.unresolved_findings[0].status == "QUESTIONABLE"
    assert res.unresolved_findings[0].task_id == "T_Q"


# ── 4. Actual statistics are preserved (zero fabrication) ─────────────────────


def test_exact_statistics_preserved():
    service = InsightService()
    vf = _make_sample_verified_finding(
        statistic=0.6543,
        p_value=0.0123,
        n=88,
        effect_size=0.6543,
    )
    res = service.generate_insights(InsightRequest(verified_findings=[vf]))
    ev = res.key_insights[0].evidence

    assert ev.statistic == pytest.approx(0.6543, rel=1e-4)
    assert ev.p_value == pytest.approx(0.0123, rel=1e-4)
    assert ev.sample_size == 88


# ── 5. Deterministic mode works without external LLM ──────────────────────────


def test_deterministic_mode_works_standalone():
    service = get_insight_service()
    vf = _make_sample_verified_finding()
    res = service.generate_insights(InsightRequest(verified_findings=[vf]))

    assert res.key_insights is not None
    assert len(res.key_insights) > 0
    assert res.summary.total_insights == 1


# ── 6. Multiple findings generate multiple ranked insights ───────────────────


def test_multiple_findings_ranked():
    service = InsightService()
    vf_strong = _make_sample_verified_finding(
        finding_id="f1",
        task_id="T1",
        variables=["experience", "salary"],
        statistic=0.85,
        p_value=0.0001,
        effect_size=0.85,
    )
    vf_weak = _make_sample_verified_finding(
        finding_id="f2",
        task_id="T2",
        variables=["age", "height"],
        statistic=0.15,
        p_value=0.04,
        effect_size=0.15,
    )

    res = service.generate_insights(InsightRequest(verified_findings=[vf_strong, vf_weak]))

    assert len(res.key_insights) == 2
    # Strong insight should be ranked first
    assert res.key_insights[0].source_task_id == "T1"
    assert res.key_insights[0].strength == "STRONG"
    assert res.key_insights[1].source_task_id == "T2"
    assert res.key_insights[1].strength == "WEAK"


# ── 7. Duplicate insights on same variable pair are handled ───────────────────


def test_duplicate_variable_pair_deduplication():
    service = InsightService()
    vf_pearson = _make_sample_verified_finding(
        finding_id="f1",
        task_id="T1",
        variables=["salary", "experience"],
        method="pearson_correlation",
        statistic=0.80,
    )
    vf_spearman = _make_sample_verified_finding(
        finding_id="f2",
        task_id="T2",
        variables=["experience", "salary"],  # Same pair inverted
        method="spearman_correlation",
        statistic=0.83,
    )

    res = service.generate_insights(InsightRequest(verified_findings=[vf_pearson, vf_spearman]))

    # Only 1 consolidated insight should be generated for the duplicate pair
    assert len(res.key_insights) == 1


# ── 8. Evidence remains traceable to source ───────────────────────────────────


def test_evidence_traceability():
    service = InsightService()
    vf = _make_sample_verified_finding(
        finding_id="finding_123",
        task_id="T_TRACE",
        variables=["ad_spend", "revenue"],
        method="spearman_correlation",
        statistic=0.91,
        p_value=0.00001,
        n=120,
    )
    res = service.generate_insights(InsightRequest(verified_findings=[vf]))
    insight = res.key_insights[0]

    assert insight.source_task_id == "T_TRACE"
    assert insight.evidence.method == "spearman_correlation"
    assert insight.evidence.variables == ["ad_spend", "revenue"]
    assert insight.evidence.sample_size == 120
    assert insight.evidence.statistic == 0.91


# ── 9. Causal language is strictly avoided ────────────────────────────────────


def test_causal_language_avoided():
    service = InsightService()
    vf = _make_sample_verified_finding(
        variables=["pricing", "sales"],
        statistic=0.75,
        p_value=0.001,
    )
    res = service.generate_insights(InsightRequest(verified_findings=[vf]))
    text = res.key_insights[0].insight.lower()

    # Banned causal claims
    assert " causes " not in text
    assert " caused " not in text
    assert " leads to " not in text
    assert " driving " not in text
    # Allowed non-causal claims
    assert "association" in text or "correspond" in text or "correlat" in text


# ── 10. Limitations are generated appropriately ───────────────────────────────


def test_limitations_attached():
    service = InsightService()
    vf = _make_sample_verified_finding(
        n=15,  # Small sample
        limitations=["Severe skew detected in initial check."],
    )
    res = service.generate_insights(InsightRequest(verified_findings=[vf]))
    lims = res.key_insights[0].limitations

    assert any("observational" in l.lower() for l in lims)
    assert any("small sample" in l.lower() for l in lims)


# ── 11. Strength classification is deterministic ──────────────────────────────


def test_deterministic_strength_thresholds():
    service = InsightService()

    # Strong: >= 0.50
    res_s = service.generate_insights(InsightRequest(verified_findings=[_make_sample_verified_finding(statistic=0.60)]))
    assert res_s.key_insights[0].strength == "STRONG"

    # Moderate: 0.30 - 0.49
    res_m = service.generate_insights(InsightRequest(verified_findings=[_make_sample_verified_finding(statistic=0.40)]))
    assert res_m.key_insights[0].strength == "MODERATE"

    # Weak: 0.10 - 0.29
    res_w = service.generate_insights(InsightRequest(verified_findings=[_make_sample_verified_finding(statistic=0.20)]))
    assert res_w.key_insights[0].strength == "WEAK"


# ── 12. Failure isolation: One bad finding does not crash request ─────────────


def test_failure_isolation():
    service = InsightService()
    vf_good = _make_sample_verified_finding(finding_id="good_1", task_id="T_GOOD")
    # Finding with empty evidence dictionary
    vf_empty_ev = VerifiedFinding(
        finding_id="f_empty",
        task_id="T_EMPTY",
        variables=[],
        method="unknown_method",
        status="VERIFIED",
        confidence=1.0,
        summary="",
        statistical_evidence={},
        critic_assessment="None",
        verification_reason="Valid",
        limitations=[],
        reason_codes=[],
    )

    res = service.generate_insights(InsightRequest(verified_findings=[vf_good, vf_empty_ev]))
    assert len(res.key_insights) >= 1
    assert any(i.source_task_id == "T_GOOD" for i in res.key_insights)


# ── 13. Empty verified findings handled gracefully ────────────────────────────


def test_empty_findings_returns_empty_response():
    service = InsightService()
    res = service.generate_insights(InsightRequest(verified_findings=[]))

    assert isinstance(res, InsightResponse)
    assert len(res.key_insights) == 0
    assert res.summary.total_insights == 0
    assert res.summary.total_verified_findings == 0


# ── 14. Group Difference & Contingency Insights ───────────────────────────────


def test_two_group_difference_insight():
    service = InsightService()
    vf = VerifiedFinding(
        finding_id="f_grp",
        task_id="T_GRP",
        variables=["department", "salary"],
        method="welchs_t_test",
        status="VERIFIED",
        confidence=1.0,
        summary="Engineering shows higher salary than Support.",
        statistical_evidence={
            "statistic": 3.45,
            "p_value": 0.002,
            "sample_size": 60,
            "cohens_d": 0.88,
            "group1": "Engineering",
            "group2": "Support",
            "group1_mean": 95000.0,
            "group2_mean": 62000.0,
        },
        critic_assessment="Accepted.",
        verification_reason="Welch t-test valid.",
        limitations=[],
        reason_codes=[],
    )

    res = service.generate_insights(InsightRequest(verified_findings=[vf]))
    assert len(res.key_insights) == 1
    ins = res.key_insights[0]
    assert ins.interpretation_type == "GROUP_DIFFERENCE"
    assert ins.strength == "STRONG"
    assert "Engineering" in ins.insight
    assert "Support" in ins.insight


# ── 15. API POST /api/insights Endpoint Tests ─────────────────────────────────


def test_api_insights_endpoint():
    vf = _make_sample_verified_finding(
        finding_id="finding_api",
        task_id="T_API",
        variables=["experience", "salary"],
        statistic=0.72,
        p_value=0.001,
        n=50,
    )
    v_res = VerificationResponse(
        dataset_id="test.csv",
        summary=VerificationSummary(
            total_findings=1,
            verified=1,
            verified_after_reinvestigation=0,
            questionable=0,
            rejected=0,
            total_reinvestigated=0,
            overall_integrity="HIGH",
            unresolved_issues=[],
        ),
        verified_findings=[vf],
        questionable_findings=[],
        rejected_findings=[],
        reinvestigations=[],
        all_findings=[vf],
    )

    payload = {"verification_response": v_res.model_dump()}
    response = client.post("/api/insights", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "key_insights" in data
    assert len(data["key_insights"]) == 1
    assert data["summary"]["total_insights"] == 1
    assert data["key_insights"][0]["source_task_id"] == "T_API"


def test_api_insights_missing_payload_422():
    response = client.post("/api/insights", json={})
    assert response.status_code == 422


def test_distribution_insight_formatting_and_missing_values():
    """Verify distribution insights format mean, median, std properly without literal Python ternary expressions,
    and fallback to 'N/A' when values are missing."""
    service = InsightService()

    # Complete stats
    vf_complete = VerifiedFinding(
        finding_id="f_dist_complete",
        task_id="T_dist_1",
        variables=["G3"],
        method="Distribution analysis",
        status="VERIFIED",
        confidence=1.0,
        confidence_level="HIGH",
        summary="Distribution analysis of G3.",
        statistical_evidence={
            "mean": 10.42,
            "median": 11.0,
            "std": 4.58,
            "skewness": -0.73,
            "sample_size": 395,
        },
        critic_assessment="Distribution analysis accepted.",
        reinvestigated=False,
        verification_reason="Standard moments evaluated.",
        limitations=[],
        reason_codes=["NORMAL_DISTRIBUTION"],
    )

    res1 = service.generate_insights(InsightRequest(verified_findings=[vf_complete]))
    assert len(res1.key_insights) == 1
    ins1 = res1.key_insights[0]
    assert "if mean_val is not None" not in ins1.insight
    assert "mean = 10.42" in ins1.insight
    assert "median = 11.00" in ins1.insight
    assert "std = 4.58" in ins1.insight
    assert "skewness = -0.73" in ins1.insight
    assert "n = 395" in ins1.insight

    # Missing mean/med/std
    vf_missing = VerifiedFinding(
        finding_id="f_dist_missing",
        task_id="T_dist_2",
        variables=["score"],
        method="Distribution analysis",
        status="VERIFIED",
        confidence=1.0,
        confidence_level="HIGH",
        summary="Distribution analysis with missing moments.",
        statistical_evidence={
            "mean": None,
            "median": None,
            "std": None,
            "skewness": 0.15,
            "sample_size": 100,
        },
        critic_assessment="Distribution analysis accepted.",
        reinvestigated=False,
        verification_reason="Skewness evaluated.",
        limitations=[],
        reason_codes=[],
    )

    res2 = service.generate_insights(InsightRequest(verified_findings=[vf_missing]))
    assert len(res2.key_insights) == 1
    ins2 = res2.key_insights[0]
    assert "if mean_val is not None" not in ins2.insight
    assert "mean = N/A" in ins2.insight
    assert "median = N/A" in ins2.insight
    assert "std = N/A" in ins2.insight


def test_spearman_rho_and_pearson_r_labeling():
    """Verify Spearman methods (including Title Case) produce 'Spearman rho' and Pearson produces 'Pearson r'."""
    service = InsightService()

    # Title Case Spearman
    vf_spearman = _make_sample_verified_finding(
        finding_id="f_spearman",
        task_id="T_spearman",
        variables=["G3", "failures"],
        method="Spearman rank correlation",
        statistic=-0.361,
        p_value=0.0001,
        n=395,
        effect_size=0.361,
    )

    res_spearman = service.generate_insights(InsightRequest(verified_findings=[vf_spearman]))
    assert len(res_spearman.key_insights) == 1
    ins_spearman = res_spearman.key_insights[0]
    assert ins_spearman.evidence.statistic_name == "Spearman rho"
    assert "Spearman rho = -0.361" in ins_spearman.insight
    assert "Pearson r" not in ins_spearman.insight

    # Pearson
    vf_pearson = _make_sample_verified_finding(
        finding_id="f_pearson",
        task_id="T_pearson",
        variables=["G1", "G2"],
        method="Pearson correlation",
        statistic=0.852,
        p_value=0.0001,
        n=395,
        effect_size=0.852,
    )

    res_pearson = service.generate_insights(InsightRequest(verified_findings=[vf_pearson]))
    assert len(res_pearson.key_insights) == 1
    ins_pearson = res_pearson.key_insights[0]
    assert ins_pearson.evidence.statistic_name == "Pearson r"
    assert "Pearson r = 0.852" in ins_pearson.insight
    assert "Spearman rho" not in ins_pearson.insight


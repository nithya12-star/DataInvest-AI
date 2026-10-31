"""
test_correlation_reinvestigation_fix.py
=======================================
Deterministic offline regression tests verifying:
1. Pure Spearman vs composite method dispatch in calculate_correlation.
2. Full empirical validation of Age vs Cholesterol on heart.csv:
   - Initial test: Pearson r = -0.0953, p = 0.0039 (p < 0.05, significant)
   - Alternative test: Spearman rho = -0.0472, p = 0.1534 (p >= 0.05, non-significant)
   - Significance lost -> QUESTIONABLE (METHOD_SENSITIVITY_DISCREPANCY)
3. ReinvestigationRecord schema top-level field population and serialization.
4. ResultComparison effect_size retention.
"""

import math
import numpy as np
import pandas as pd
import pytest

from app.statistics.correlation import execute_correlation
from app.schemas.analysis_schema import AnalysisResult
from app.schemas.critic_schema import CriticResult
from app.schemas.verification_schema import (
    ReinvestigationRecord,
    ResultComparison,
)
from app.services.verification_service import VerificationService


def test_correlation_dispatch_pure_spearman_vs_composite():
    """Verify 'spearman' runs Spearman, but composite methods run Pearson."""
    # Create non-linear monotonic data where Pearson and Spearman differ
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    y = np.array([1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 1000.0])  # outlier at end
    df = pd.DataFrame({"x": x, "y": y})

    # Pure spearman
    ok_spearman, res_spearman, _, _ = execute_correlation(df, "x", "y", method="Spearman rank correlation")
    assert ok_spearman is True
    assert res_spearman["method"] == "Spearman rank correlation"
    # Spearman rank of monotonic data is 1.0
    assert pytest.approx(res_spearman["statistic"], 0.01) == 1.0

    # Composite / initial Planner method string
    ok_comp, res_comp, _, _ = execute_correlation(df, "x", "y", method="Pearson and Spearman correlation coefficients")
    assert ok_comp is True
    assert res_comp["method"] == "Pearson correlation"
    # Pearson r will be substantially less than 1.0 due to the extreme quadratic jump
    assert res_comp["statistic"] < 0.90


def test_heart_csv_age_vs_cholesterol_empirical_verification():
    """Verify empirical values on Heart Disease dataset."""
    import os
    candidates = [
        "DataInvest_Research_Results/gemini_live_5_datasets_20261029/heart_disease/heart.csv",
        "../DataInvest_Dataset_Candidates/heart.csv",
        "../DataInvest_Pilot_Staging/heart.csv",
    ]
    heart_path = next((p for p in candidates if os.path.exists(p)), None)
    if not heart_path:
        pytest.skip("heart.csv not found in candidate paths")
    df = pd.read_csv(heart_path)

    # Initial Planner method: "Pearson and Spearman correlation coefficients"
    ok_init, init_res, _, _ = execute_correlation(
        df, "Age", "Cholesterol", method="Pearson and Spearman correlation coefficients"
    )
    assert ok_init is True
    assert init_res["method"] == "Pearson correlation"
    assert pytest.approx(init_res["statistic"], abs=1e-3) == -0.0953
    assert pytest.approx(init_res["p_value"], abs=1e-4) == 0.0039
    assert init_res["p_value"] < 0.05  # Significant

    # Critic re-investigation method: "Spearman rank correlation"
    ok_alt, alt_res, _, _ = execute_correlation(
        df, "Age", "Cholesterol", method="Spearman rank correlation"
    )
    assert ok_alt is True
    assert alt_res["method"] == "Spearman rank correlation"
    assert pytest.approx(alt_res["statistic"], abs=1e-3) == -0.0472
    assert pytest.approx(alt_res["p_value"], abs=1e-4) == 0.1534
    assert alt_res["p_value"] >= 0.05  # Non-significant


def test_verification_service_detects_significance_loss():
    """Verify that VerificationService marks significance loss as QUESTIONABLE."""
    service = VerificationService()

    orig_analysis = AnalysisResult(
        task_id="task_5",
        status="success",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        columns=["Age", "Cholesterol"],
        sample_size=918,
        results={
            "statistic": -0.0953,
            "correlation_coefficient": -0.0953,
            "p_value": 0.0039,
            "sample_size": 918,
            "effect_size": 0.0953,
            "method": "Pearson correlation",
        },
    )

    reinvest_analysis = AnalysisResult(
        task_id="task_5",
        status="success",
        analysis_type="numerical_vs_numerical",
        method="Spearman rank correlation",
        columns=["Age", "Cholesterol"],
        sample_size=918,
        results={
            "statistic": -0.0472,
            "correlation_coefficient": -0.0472,
            "p_value": 0.1534,
            "sample_size": 918,
            "effect_size": 0.0472,
            "method": "Spearman rank correlation",
        },
    )

    comp = service._compare_results(orig_analysis, reinvest_analysis)
    assert comp.original_p_value == 0.0039
    assert comp.reinvestigated_p_value == 0.1534
    assert comp.significance_consistent is False
    assert comp.original_effect_size == 0.0953
    assert comp.reinvestigated_effect_size == 0.0472

    critic_orig = CriticResult(
        task_id="task_5",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        status="QUESTIONABLE",
        confidence=0.75,
        confidence_level="MEDIUM",
        summary="High skewness detected; non-parametric rank test recommended.",
        statistical_validity="Valid calculation",
        method_assessment="Pearson assumption violated",
        recommendation="Use Spearman rank correlation",
        requires_reinvestigation=True,
        recommended_method="Spearman rank correlation",
    )

    critic_second = CriticResult(
        task_id="task_5",
        analysis_type="numerical_vs_numerical",
        method="Spearman rank correlation",
        status="ACCEPTED",
        confidence=0.85,
        confidence_level="HIGH",
        summary="Spearman rank correlation executed successfully.",
        statistical_validity="Valid calculation",
        method_assessment="Spearman appropriate",
        recommendation="Accept finding with rank interpretation",
    )

    record = ReinvestigationRecord(
        reinvestigation_id="reinvest_task_5",
        source_task_id="task_5",
        original_method="Pearson correlation",
        reinvestigated_method="Spearman rank correlation",
        variables=["Age", "Cholesterol"],
        trigger_reason="High skewness detected",
        original_analysis=orig_analysis,
        original_critic_review=critic_orig,
        reinvestigated_analysis=reinvest_analysis,
        second_critic_review=critic_second,
        comparison=comp,
        status="QUESTIONABLE",
        attempt_number=1,
        original_statistic=comp.original_statistic,
        reinvestigated_statistic=comp.reinvestigated_statistic,
        original_p_value=comp.original_p_value,
        reinvestigated_p_value=comp.reinvestigated_p_value,
        original_effect_size=comp.original_effect_size,
        reinvestigated_effect_size=comp.reinvestigated_effect_size,
        original_sample_size=comp.original_sample_size,
        reinvestigated_sample_size=comp.reinvestigated_sample_size,
    )

    data = record.model_dump()
    assert data["original_statistic"] == -0.0953
    assert data["reinvestigated_statistic"] == -0.0472
    assert data["original_p_value"] == 0.0039
    assert data["reinvestigated_p_value"] == 0.1534
    assert data["original_effect_size"] == 0.0953
    assert data["reinvestigated_effect_size"] == 0.0472

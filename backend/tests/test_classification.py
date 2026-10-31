"""
backend/tests/test_classification.py
=====================================
Unit tests for the DataInvest AI supervised binary classification module,
reproducible train/test protocol, and paired McNemar statistical test.
"""

import numpy as np
import pandas as pd
import pytest

from app.statistics.classification import (
    compute_mcnemar_p_value,
    execute_binary_classification,
    compare_paired_classification,
    _detect_impossible_zeros,
)
from app.agents.analysis_agent import AnalysisAgent
from app.schemas.planner_schema import InvestigationTask


@pytest.fixture
def synthetic_heart_data():
    """Generates synthetic tabular heart-disease clinical data with impossible zeros."""
    rng = np.random.RandomState(42)
    n = 200
    age = rng.randint(30, 75, size=n)
    sex = rng.choice(["M", "F"], size=n)
    chest_pain = rng.choice(["ATA", "NAP", "ASY", "TA"], size=n)
    resting_bp = rng.randint(90, 180, size=n)
    # Inject 30 impossible zero values into cholesterol
    cholesterol = rng.randint(150, 350, size=n)
    cholesterol[:30] = 0
    max_hr = rng.randint(80, 200, size=n)
    # Target correlated with age and max_hr
    logits = 0.05 * age - 0.03 * max_hr + (cholesterol > 250) * 0.5 - 1.0
    probs = 1 / (1 + np.exp(-logits))
    target = (rng.rand(n) < probs).astype(int)

    return pd.DataFrame({
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "MaxHR": max_hr,
        "HeartDisease": target,
    })


def test_detect_impossible_zeros(synthetic_heart_data):
    """Test detection of physiologically impossible zeros in clinical metrics."""
    zeros = _detect_impossible_zeros(synthetic_heart_data, list(synthetic_heart_data.columns))
    assert "Cholesterol" in zeros
    assert zeros["Cholesterol"] == 30
    assert "Age" not in zeros


def test_execute_binary_classification_critic_off(synthetic_heart_data):
    """Test standard baseline classification with Critic OFF (retaining literal zeros)."""
    success, results, warnings, error = execute_binary_classification(
        df=synthetic_heart_data,
        target_column="HeartDisease",
        model_type="logistic_regression",
        apply_critic_cleaning=False,
        random_state=42,
    )

    assert success is True
    assert error is None
    assert results["critic_mode"] == "critic_off"
    assert 0.0 <= results["accuracy"] <= 1.0
    assert 0.0 <= results["precision"] <= 1.0
    assert 0.0 <= results["recall"] <= 1.0
    assert 0.0 <= results["f1_score"] <= 1.0
    assert len(results["test_predictions"]) == results["test_samples"]
    assert len(results["test_ground_truth"]) == results["test_samples"]
    # Check that literal zero retention warning was issued
    assert any("literal zero values" in w for w in warnings)


def test_execute_binary_classification_critic_on(synthetic_heart_data):
    """Test proposed classification with Critic ON (imputing impossible zeros)."""
    success, results, warnings, error = execute_binary_classification(
        df=synthetic_heart_data,
        target_column="HeartDisease",
        model_type="logistic_regression",
        apply_critic_cleaning=True,
        random_state=42,
    )

    assert success is True
    assert error is None
    assert results["critic_mode"] == "critic_on"
    assert len(results["reinvestigation_applied"]) > 0
    assert any("Imputed" in r for r in results["reinvestigation_applied"])


def test_compare_paired_classification(synthetic_heart_data):
    """Test paired comparison under identical train/test split."""
    comp = compare_paired_classification(
        df=synthetic_heart_data,
        target_column="HeartDisease",
        model_type="logistic_regression",
        random_state=42,
    )

    assert "baseline_critic_off" in comp
    assert "proposed_critic_on" in comp
    assert "deltas" in comp
    assert "statistical_significance" in comp

    sig = comp["statistical_significance"]
    assert "p_value" in sig
    assert 0.0 <= sig["p_value"] <= 1.0
    assert "table" in sig


def test_mcnemar_identical_predictions():
    """Test McNemar test when predictions are identical (discordance = 0)."""
    y_true = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    y_pred1 = np.array([1, 0, 1, 0, 0, 1, 0, 1])
    y_pred2 = np.array([1, 0, 1, 0, 0, 1, 0, 1])

    res = compute_mcnemar_p_value(y_true, y_pred1, y_pred2)
    assert res["p_value"] == 1.0
    assert res["statistic"] == 0.0


def test_mcnemar_discordant_predictions():
    """Test McNemar test with discordant predictions."""
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_pred1 = np.array([1, 0, 0, 0, 0, 0, 1, 1])  # 2 correct (index 0, 4, 5)
    y_pred2 = np.array([1, 1, 1, 0, 0, 0, 0, 0])  # 6 correct

    res = compute_mcnemar_p_value(y_true, y_pred1, y_pred2)
    assert 0.0 <= res["p_value"] <= 1.0
    assert res["discordant_pairs"] > 0


def test_analysis_agent_dispatch_binary_classification(synthetic_heart_data):
    """Test AnalysisAgent dispatches binary_classification task type properly."""
    agent = AnalysisAgent()
    task = InvestigationTask(
        task_id="task_class_1",
        task_type="binary_classification",
        task="Evaluate Heart Disease Classification",
        description="Predict HeartDisease target using clinical features",
        method="logistic_regression",
        columns=["HeartDisease", "Age", "Sex", "RestingBP", "Cholesterol", "MaxHR"],
        column_a="HeartDisease",
        reason="Reproduce classification benchmarks",
        priority="high",
        notes=["critic_on"],
    )

    res = agent.execute_task(task, synthetic_heart_data)
    assert res.status == "success"
    assert res.analysis_type == "binary_classification"
    assert "accuracy" in res.results
    assert "f1_score" in res.results


def test_classification_error_isolation():
    """Test failure isolation when target column is missing or dataset is empty."""
    agent = AnalysisAgent()
    task = InvestigationTask(
        task_id="task_err_1",
        task_type="binary_classification",
        task="Evaluate Missing Target",
        description="Predict non-existent target",
        method="logistic_regression",
        columns=["NonExistentTarget", "Age"],
        column_a="NonExistentTarget",
        reason="Testing error isolation",
        priority="low",
    )

    df = pd.DataFrame({"Age": [30, 40, 50]})
    res = agent.execute_task(task, df)
    assert res.status == "failed"
    assert res.error.code == "TARGET_COLUMN_NOT_FOUND"

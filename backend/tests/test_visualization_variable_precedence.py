"""
backend/tests/test_visualization_variable_precedence.py
======================================================
Focused offline regression tests verifying that visualization generation
strictly prioritizes actual computed result columns over positional planner
candidate column lists.
"""

import numpy as np
import pandas as pd
import pytest

from app.agents.analysis_agent import AnalysisAgent
from app.schemas.planner_schema import InvestigationTask
from app.services.visualization_service import VisualizationService


@pytest.fixture
def heart_sample_df():
    np.random.seed(42)
    n = 100
    age = np.random.uniform(30, 80, n)
    cholesterol = np.random.uniform(150, 400, n)
    max_hr = 220 - age + np.random.normal(0, 10, n)
    chest_pain = np.random.choice(["ATA", "NAP", "ASY"], size=n)
    heart_disease = np.random.choice([0, 1], size=n)

    return pd.DataFrame({
        "Age": age,
        "Cholesterol": cholesterol,
        "MaxHR": max_hr,
        "ChestPainType": chest_pain,
        "HeartDisease": heart_disease,
    })


def test_correlation_visualization_prioritizes_results_columns_over_candidate_list(heart_sample_df):
    """
    Regression test for the heart.csv finding 5 issue:
    Candidate columns list is ["Age", "Cholesterol", "MaxHR"].
    Actual computed results specify column_a="Age" and column_b="MaxHR".
    The resulting chart must display Age vs MaxHR, NOT Age vs Cholesterol.
    """
    viz_svc = VisualizationService()
    candidate_columns = ["Age", "Cholesterol", "MaxHR"]
    results = {
        "column_a": "Age",
        "column_b": "MaxHR",
        "correlation_coefficient": -0.3820,
        "statistic": -0.3820,
        "p_value": 2.84e-33,
        "sample_size": len(heart_sample_df),
    }

    evidence = viz_svc.generate_finding_visualization(
        task_id="task_5",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        columns=candidate_columns,
        df=heart_sample_df,
        results=results,
    )

    assert evidence is not None
    assert evidence.figure_type == "scatter"
    assert evidence.x_label == "Age"
    assert evidence.y_label == "MaxHR"
    assert evidence.columns == ["Age", "MaxHR"]
    assert "Age vs MaxHR" in evidence.title
    assert "Cholesterol" not in evidence.title


def test_analysis_agent_generates_correct_correlation_chart_with_candidate_columns(heart_sample_df):
    """
    End-to-end integration test through AnalysisAgent:
    When InvestigationTask has candidate columns ["Age", "Cholesterol", "MaxHR"]
    and targeted column_a="Age", column_b="MaxHR", the resulting visual evidence
    must reflect Age vs MaxHR.
    """
    agent = AnalysisAgent()
    task = InvestigationTask(
        task_id="task_5",
        task_type="numerical_vs_numerical",
        task="Correlation Analysis of Numerical Features",
        description="Evaluate pairwise linear correlations among key continuous clinical features.",
        method="Pearson correlation",
        columns=["Age", "Cholesterol", "MaxHR"],
        column_a="Age",
        column_b="MaxHR",
        reason="Checks for physiological predictors correlation.",
        priority="high",
    )

    result = agent.execute_task(task, heart_sample_df)

    assert result.status == "success"
    assert result.results["column_a"] == "Age"
    assert result.results["column_b"] == "MaxHR"
    assert result.visual_evidence is not None
    assert result.visual_evidence.x_label == "Age"
    assert result.visual_evidence.y_label == "MaxHR"
    assert "Age vs MaxHR" in result.visual_evidence.title
    assert "Cholesterol" not in result.visual_evidence.title


def test_two_group_box_plot_prioritizes_results_group_and_value(heart_sample_df):
    """
    Two-group box plot must prioritize results['group_column'] and results['value_column']
    over positional columns ordering.
    """
    viz_svc = VisualizationService()
    # Intentionally misordered candidate columns
    candidate_columns = ["Cholesterol", "HeartDisease", "Age"]
    results = {
        "group_column": "HeartDisease",
        "value_column": "Age",
        "statistic": 4.12,
        "p_value": 0.0001,
    }

    evidence = viz_svc.generate_finding_visualization(
        task_id="task_group",
        analysis_type="two_groups_vs_numerical",
        method="Welch's t-test",
        columns=candidate_columns,
        df=heart_sample_df,
        results=results,
    )

    assert evidence is not None
    assert evidence.figure_type == "box"
    assert evidence.x_label == "HeartDisease"
    assert evidence.y_label == "Age"
    assert evidence.columns == ["HeartDisease", "Age"]
    assert "Age by HeartDisease" in evidence.title


def test_multi_group_box_plot_prioritizes_results_group_and_value(heart_sample_df):
    """
    Multi-group box plot must prioritize results['group_column'] and results['value_column'].
    """
    viz_svc = VisualizationService()
    candidate_columns = ["Cholesterol", "ChestPainType", "MaxHR"]
    results = {
        "group_column": "ChestPainType",
        "value_column": "MaxHR",
        "statistic": 8.45,
        "p_value": 0.0002,
    }

    evidence = viz_svc.generate_finding_visualization(
        task_id="task_anova",
        analysis_type="multiple_groups_vs_numerical",
        method="One-way ANOVA",
        columns=candidate_columns,
        df=heart_sample_df,
        results=results,
    )

    assert evidence is not None
    assert evidence.figure_type == "multi_box"
    assert evidence.x_label == "ChestPainType"
    assert evidence.y_label == "MaxHR"
    assert evidence.columns == ["ChestPainType", "MaxHR"]
    assert "MaxHR across ChestPainType" in evidence.title


def test_fallback_to_positional_columns_when_results_keys_missing(heart_sample_df):
    """
    When results dictionary lacks column_a/column_b, gracefully fall back to positional columns[0], columns[1].
    """
    viz_svc = VisualizationService()
    evidence = viz_svc.generate_finding_visualization(
        task_id="t_fallback",
        analysis_type="numerical_vs_numerical",
        method="Pearson correlation",
        columns=["Age", "Cholesterol"],
        df=heart_sample_df,
        results={"statistic": 0.12},  # No column_a or column_b
    )

    assert evidence is not None
    assert evidence.x_label == "Age"
    assert evidence.y_label == "Cholesterol"
    assert "Age vs Cholesterol" in evidence.title


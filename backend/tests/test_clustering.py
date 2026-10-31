"""
backend/tests/test_clustering.py
================================
Targeted unit tests for Customer Segmentation and K-Means Clustering module,
AnalysisAgent dispatch, and CriticAgent statistical verification.
"""

from __future__ import annotations

import pathlib
import pytest
import pandas as pd
import numpy as np

from app.statistics.clustering import (
    detect_identifier_columns,
    detect_scale_disparity,
    execute_kmeans_clustering,
    compare_paired_clustering,
)
from app.agents.analysis_agent import AnalysisAgent
from app.agents.critic_agent import CriticAgent
from app.schemas.planner_schema import InvestigationTask


@pytest.fixture
def sample_customer_df() -> pd.DataFrame:
    """Load or generate a synthetic customer segmentation dataframe."""
    candidate_path = pathlib.Path(
        r"C:\Users\SPURGE\MAJOR_PROJECT_DOCS\REVIEW 2\DataInvest_Dataset_Candidates\Customers.csv"
    )
    if candidate_path.exists():
        return pd.read_csv(candidate_path)

    # Fallback synthetic dataframe
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "CustomerID": np.arange(1, n + 1),
        "Gender": np.random.choice(["Male", "Female"], size=n),
        "Age": np.random.randint(18, 70, size=n),
        "Annual Income ($)": np.random.randint(15000, 150000, size=n),
        "Spending Score (1-100)": np.random.randint(1, 100, size=n),
        "Profession": np.random.choice(["Artist", "Engineer", "Doctor"], size=n),
        "Work Experience": np.random.randint(0, 15, size=n),
        "Family Size": np.random.randint(1, 7, size=n),
    })


def test_detect_identifier_columns(sample_customer_df: pd.DataFrame):
    """Verify that record identifier columns like CustomerID are identified."""
    detected = detect_identifier_columns(sample_customer_df, sample_customer_df.columns)
    assert "CustomerID" in detected
    assert "Annual Income ($)" not in detected
    assert "Spending Score (1-100)" not in detected


def test_detect_scale_disparity(sample_customer_df: pd.DataFrame):
    """Verify detection of scale disparity between Annual Income and Spending Score."""
    cols = ["Annual Income ($)", "Spending Score (1-100)"]
    disparity = detect_scale_disparity(sample_customer_df, cols)
    assert disparity["has_disparity"] is True
    assert disparity["ratio"] > 10.0
    assert disparity["max_col"] == "Annual Income ($)"
    assert disparity["min_col"] == "Spending Score (1-100)"


def test_execute_kmeans_clustering_baseline_vs_proposed(sample_customer_df: pd.DataFrame):
    """Verify baseline unscaled execution vs proposed audited/standardized execution."""
    features = ["Annual Income ($)", "Spending Score (1-100)"]

    # 1. Baseline (Critic OFF): unscaled
    s1, r1, w1, e1 = execute_kmeans_clustering(
        df=sample_customer_df,
        feature_columns=features,
        k_clusters=3,
        random_state=42,
        apply_critic_cleaning=False,
        scale_features=False,
    )
    assert s1 is True
    assert e1 is None
    assert r1["scaling_applied"] is False
    assert any("scale disparity" in w.lower() for w in w1)
    assert r1["k_clusters"] == 3
    assert sum(r1["cluster_sizes"].values()) == len(sample_customer_df)

    # 2. Proposed (Critic ON): standardized
    s2, r2, w2, e2 = execute_kmeans_clustering(
        df=sample_customer_df,
        feature_columns=features,
        k_clusters=3,
        random_state=42,
        apply_critic_cleaning=True,
        scale_features=True,
    )
    assert s2 is True
    assert e2 is None
    assert r2["scaling_applied"] is True
    assert r2["silhouette_score_standardized"] > r1["silhouette_score_standardized"]
    assert any("standardscaler" in item.lower() for item in r2["reinvestigation_applied"])


def test_fixed_seed_reproducibility(sample_customer_df: pd.DataFrame):
    """Verify 100% deterministic reproducibility with fixed random_state."""
    features = ["Annual Income ($)", "Spending Score (1-100)"]

    s1, r1, _, _ = execute_kmeans_clustering(
        df=sample_customer_df,
        feature_columns=features,
        k_clusters=4,
        random_state=42,
        apply_critic_cleaning=True,
    )
    s2, r2, _, _ = execute_kmeans_clustering(
        df=sample_customer_df,
        feature_columns=features,
        k_clusters=4,
        random_state=42,
        apply_critic_cleaning=True,
    )

    assert s1 and s2
    assert r1["cluster_labels"] == r2["cluster_labels"]
    assert r1["silhouette_score_standardized"] == r2["silhouette_score_standardized"]
    assert r1["cluster_sizes"] == r2["cluster_sizes"]
    assert r1["cluster_centroids"] == r2["cluster_centroids"]


def test_compare_paired_clustering(sample_customer_df: pd.DataFrame):
    """Verify paired comparison computes valid ARI, NMI, and metric deltas."""
    features = ["Annual Income ($)", "Spending Score (1-100)"]
    comparison = compare_paired_clustering(
        df=sample_customer_df,
        feature_columns=features,
        k_baseline=3,
        random_state=42,
    )

    assert "error" not in comparison
    assert "baseline_critic_off" in comparison
    assert "proposed_critic_on_matched_k" in comparison
    assert "proposed_critic_on_optimal_k" in comparison

    comp_metrics = comparison["comparison_metrics"]
    assert 0.0 <= comp_metrics["adjusted_rand_index_matched_k"] <= 1.0
    assert 0.0 <= comp_metrics["normalized_mutual_info_matched_k"] <= 1.0
    assert comp_metrics["delta_silhouette_standardized_matched_k"] > 0.0
    assert len(comparison["methodological_conclusions"]) >= 3


def test_analysis_agent_dispatch_clustering(sample_customer_df: pd.DataFrame):
    """Verify AnalysisAgent correctly dispatches clustering and customer_segmentation tasks."""
    agent = AnalysisAgent()
    task = InvestigationTask(
        task_id="task_cluster_01",
        task_type="customer_segmentation",
        task="Customer Segmentation",
        description="Cluster customers based on income and spending",
        method="K-Means Clustering k=4 critic_on",
        columns=["Annual Income ($)", "Spending Score (1-100)"],
        reason="Discover customer behavioral cohorts",
        priority="high",
    )

    result = agent.execute_task(task, sample_customer_df)
    assert result.status == "success"
    assert result.sample_size == len(sample_customer_df)
    assert "silhouette_score_standardized" in result.results
    assert result.results["k_clusters"] == 4


def test_critic_agent_reinvestigation_trigger_on_unscaled(sample_customer_df: pd.DataFrame):
    """Verify CriticAgent flags unscaled clustering with REQUIRES_REINVESTIGATION."""
    analysis_agent = AnalysisAgent()
    critic_agent = CriticAgent()

    # Unscaled Baseline Task (Critic OFF)
    task_unscaled = InvestigationTask(
        task_id="task_unscaled",
        task_type="clustering",
        task="Unscaled Customer Segmentation",
        description="Raw clustering without standardization",
        method="K-Means k=3 critic_off",
        columns=["Annual Income ($)", "Spending Score (1-100)"],
        reason="Baseline segmentation",
        priority="high",
    )

    result_unscaled = analysis_agent.execute_task(task_unscaled, sample_customer_df)
    assert result_unscaled.status == "success"

    review_unscaled = critic_agent.evaluate_result(result_unscaled)
    assert review_unscaled.status == "REQUIRES_REINVESTIGATION"
    assert review_unscaled.requires_reinvestigation is True
    assert "SCALE_DISPARITY_DISTORTION" in review_unscaled.reason_codes
    assert "RECOMMEND_STANDARDIZATION" in review_unscaled.reason_codes

    # Audited / Standardized Task (Critic ON)
    task_scaled = InvestigationTask(
        task_id="task_scaled",
        task_type="clustering",
        task="Standardized Customer Segmentation",
        description="Clustering with standardization and audit",
        method="K-Means k=4 critic_on",
        columns=["Annual Income ($)", "Spending Score (1-100)"],
        reason="Audited segmentation",
        priority="high",
    )

    result_scaled = analysis_agent.execute_task(task_scaled, sample_customer_df)
    assert result_scaled.status == "success"

    review_scaled = critic_agent.evaluate_result(result_scaled)
    assert review_scaled.status in ("ACCEPTED", "QUESTIONABLE")
    assert review_scaled.requires_reinvestigation is False
    assert "DIMENSIONAL_SCALING_VERIFIED" in review_scaled.reason_codes


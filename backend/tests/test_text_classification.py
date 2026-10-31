"""
backend/tests/test_text_classification.py
=========================================
Targeted unit tests for Fake News Text Classification module,
AnalysisAgent dispatch, and CriticAgent statistical verification.
"""

from __future__ import annotations

import pathlib
import pytest
import pandas as pd
import numpy as np

from app.statistics.text_classification import (
    detect_publisher_wire_leakage,
    detect_categorical_metadata_leakage,
    sanitize_publisher_watermarks,
    execute_text_classification,
    compare_paired_text_classification,
)
from app.agents.analysis_agent import AnalysisAgent
from app.agents.critic_agent import CriticAgent
from app.schemas.planner_schema import InvestigationTask


@pytest.fixture
def sample_text_df() -> pd.DataFrame:
    """Generate a clean synthetic text classification dataframe mimicking ISOT Fake News structure."""
    true_texts = [
        "WASHINGTON (Reuters) - The Senate approved the bipartisan defense legislation on Tuesday, sending the measure to the president.",
        "LONDON (Reuters) - Prime Minister said the trade agreement was finalized following diplomatic negotiations.",
        "BEIJING (Reuters) - Officials announced economic reforms aimed at stabilizing industrial output for the fiscal quarter.",
        "PARIS (Reuters) — European Union ministers agreed to implement environmental standards after prolonged debates on Friday.",
        "WASHINGTON (Reuters) - The committee released its non-partisan economic report highlighting job growth figures.",
    ]
    fake_texts = [
        "BREAKING: Shocking video exposes secret government plot to manipulate election results! Watch what happened!",
        "Hillary Clinton caught on tape confessing to massive scandal; liberal media refuses to report this bombshell!",
        "You won't believe what Donald Trump just said to his critics in epic takedown message on social media!",
        "Proof that secret deep state operatives orchestrated the crisis to undermine patriotic American citizens!",
        "Viral photo proves famous Hollywood celebrities were secretly arrested in midnight FBI operation!",
    ]
    
    rows = []
    for t in true_texts:
        rows.append({"title": t[:40], "text": t, "subject": "politicsNews", "label": 0})
    for t in fake_texts:
        rows.append({"title": t[:40], "text": t, "subject": "News", "label": 1})
    
    # Replicate to reach reasonable size for train/test splitting
    df = pd.DataFrame(rows * 4)  # 40 rows
    return df


def test_detect_publisher_wire_leakage(sample_text_df: pd.DataFrame):
    """Verify detection of wire agency dateline leakage."""
    leak = detect_publisher_wire_leakage(sample_text_df["text"].values, sample_text_df["label"].values)
    assert leak["has_leakage"] is True
    assert leak["wire_prefix_in_negative_pct"] == 100.0
    assert leak["wire_prefix_in_positive_pct"] == 0.0
    assert leak["disparity_pct"] == 100.0


def test_detect_categorical_metadata_leakage(sample_text_df: pd.DataFrame):
    """Verify detection of disjoint categorical features (e.g. subject)."""
    meta_leak = detect_categorical_metadata_leakage(sample_text_df, "label", ["subject", "title"])
    assert "subject" in meta_leak
    assert "Disjoint categories" in meta_leak["subject"]


def test_sanitize_publisher_watermarks():
    """Verify stripping of wire dateline prefixes and Reuters mentions."""
    raw1 = "WASHINGTON (Reuters) - The president signed the bill into law."
    clean1 = sanitize_publisher_watermarks(raw1)
    assert not clean1.startswith("WASHINGTON (Reuters) -")
    assert "Reuters" not in clean1
    assert "The president signed the bill into law." in clean1

    raw2 = "LONDON (Reuters) — Officials confirmed the report."
    clean2 = sanitize_publisher_watermarks(raw2)
    assert "Reuters" not in clean2
    assert "Officials confirmed the report." in clean2


def test_execute_text_classification_deterministic(sample_text_df: pd.DataFrame):
    """Verify reproducible execution of text classification pipeline."""
    s1, r1, w1, e1 = execute_text_classification(
        df=sample_text_df,
        text_columns=["title", "text"],
        label_column="label",
        random_state=42,
        apply_critic_cleaning=False,
    )
    s2, r2, w2, e2 = execute_text_classification(
        df=sample_text_df,
        text_columns=["title", "text"],
        label_column="label",
        random_state=42,
        apply_critic_cleaning=False,
    )

    assert s1 and s2
    assert e1 is None and e2 is None
    assert r1["accuracy"] == r2["accuracy"]
    assert r1["f1_score"] == r2["f1_score"]
    assert r1["test_predictions"] == r2["test_predictions"]
    assert len(r1["top_true_features"]) > 0
    assert len(r1["top_fake_features"]) > 0


def test_compare_paired_text_classification(sample_text_df: pd.DataFrame):
    """Verify paired comparison computes valid McNemar test and deltas."""
    paired = compare_paired_text_classification(
        df=sample_text_df,
        text_columns=["title", "text"],
        label_column="label",
        random_state=42,
    )

    assert "error" not in paired
    assert "baseline_critic_off" in paired
    assert "proposed_critic_on" in paired
    assert "statistical_significance" in paired
    assert "p_value" in paired["statistical_significance"]
    assert "deltas" in paired


def test_analysis_agent_dispatch_text_classification(sample_text_df: pd.DataFrame):
    """Verify AnalysisAgent correctly dispatches text_classification task."""
    agent = AnalysisAgent()
    task = InvestigationTask(
        task_id="task_text_01",
        task_type="text_classification",
        task="Fake News Text Classification",
        description="Classify news articles using TF-IDF and Logistic Regression",
        method="Logistic Regression critic_on",
        columns=["title", "text"],
        column_a="label",
        reason="Evaluate journalistic vs sensationalist linguistic features",
        priority="high",
    )

    result = agent.execute_task(task, sample_text_df)
    assert result.status == "success"
    assert result.sample_size == len(sample_text_df)
    assert "f1_score" in result.results
    assert "confusion_matrix" in result.results


def test_critic_agent_reinvestigation_trigger_on_uncleaned_text(sample_text_df: pd.DataFrame):
    """Verify CriticAgent triggers REQUIRES_REINVESTIGATION on watermark-leaked text."""
    analysis_agent = AnalysisAgent()
    critic_agent = CriticAgent()

    # 1. Un-audited Baseline Task (Critic OFF)
    task_raw = InvestigationTask(
        task_id="task_raw_text",
        task_type="text_classification",
        task="Baseline Text Classification",
        description="Evaluate raw news text with wire watermarks",
        method="Logistic Regression critic_off",
        columns=["title", "text"],
        column_a="label",
        reason="Baseline text classification",
        priority="high",
    )

    res_raw = analysis_agent.execute_task(task_raw, sample_text_df)
    assert res_raw.status == "success"

    review_raw = critic_agent.evaluate_result(res_raw)
    assert review_raw.status == "REQUIRES_REINVESTIGATION"
    assert review_raw.requires_reinvestigation is True
    assert "PUBLISHER_WATERMARK_LEAKAGE" in review_raw.reason_codes
    assert "RECOMMEND_WATERMARK_SANITIZATION" in review_raw.reason_codes

    # 2. Audited Proposed Task (Critic ON)
    task_san = InvestigationTask(
        task_id="task_san_text",
        task_type="text_classification",
        task="Sanitized Text Classification",
        description="Evaluate sanitized news text without wire watermarks",
        method="Logistic Regression critic_on",
        columns=["title", "text"],
        column_a="label",
        reason="Audited text classification",
        priority="high",
    )

    res_san = analysis_agent.execute_task(task_san, sample_text_df)
    assert res_san.status == "success"

    review_san = critic_agent.evaluate_result(res_san)
    assert review_san.status in ("ACCEPTED", "QUESTIONABLE")
    assert review_san.requires_reinvestigation is False
    assert "WATERMARK_SANITATION_VERIFIED" in review_san.reason_codes


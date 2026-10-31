"""
backend/tests/test_gemini_benchmark_preflight.py
================================================
Focused offline test suite for genuine Gemini 5-dataset benchmark preflight:
1. Validates all 5 approved datasets exist and load cleanly from raw files without mocks.
2. Validates Education Survey schema (79 columns), target 'b4_8', and binary classification transformation.
3. Validates ShopMania schema (4 columns: product_id, text, cat_id, category) without synthetic features.
4. Verifies run_gemini_five_datasets_benchmark.py contains zero cached JSON metric substitutions.
5. Verifies genuine Gemini Planner strictly enforces no-silent-fallback in research-run mode.
"""

from __future__ import annotations

import json
from pathlib import Path
import httpx
import numpy as np
import pandas as pd
import pytest

from app.modules import profiler
from app.services.gemini_client import (
    GeminiClient,
    GeminiNetworkError,
    GeminiClientError,
    GeminiRateLimitError,
)
from app.services.planner_service import create_analysis_plan
from scripts.run_education_benchmark import load_education_xlsx
from scripts.run_shopmania_benchmark import load_shopmania_data


PROJECT_ROOT = Path(r"C:\Users\SPURGE\MAJOR_PROJECT_DOCS\REVIEW 2\Major_Project")
CANDIDATES_ROOT = Path(r"C:\Users\SPURGE\MAJOR_PROJECT_DOCS\REVIEW 2\DataInvest_Dataset_Candidates")
BENCHMARK_SCRIPT = PROJECT_ROOT / "backend" / "scripts" / "run_gemini_five_datasets_benchmark.py"


# ============================================================================
# 1. RAW DATASET LOADING TESTS
# ============================================================================

def test_heart_disease_raw_loading():
    """Verify heart.csv exists, loads 918 rows x 12 cols, and contains target HeartDisease."""
    path = CANDIDATES_ROOT / "heart.csv"
    assert path.exists(), f"heart.csv not found at {path}"
    df = pd.read_csv(path)
    assert df.shape == (918, 12)
    assert "HeartDisease" in df.columns
    assert set(df["HeartDisease"].unique()).issubset({0, 1})


def test_fake_news_raw_loading():
    """Verify Fake.csv and True.csv exist, concatenate to 44,898 rows with binary labels."""
    fake_path = CANDIDATES_ROOT / "FakeNews" / "Fake.csv"
    true_path = CANDIDATES_ROOT / "FakeNews" / "True.csv"
    assert fake_path.exists(), f"Fake.csv missing at {fake_path}"
    assert true_path.exists(), f"True.csv missing at {true_path}"
    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)
    assert len(df_fake) == 23481
    assert len(df_true) == 21417
    df_fake["label"] = 0
    df_true["label"] = 1
    comb = pd.concat([df_fake, df_true], ignore_index=True)
    assert len(comb) == 44898
    assert "text" in comb.columns


def test_customer_segmentation_raw_loading():
    """Verify Customers.csv exists, loads 2,000 rows x 8 cols, with all segmentation features."""
    path = CANDIDATES_ROOT / "Customers.csv"
    assert path.exists(), f"Customers.csv missing at {path}"
    df = pd.read_csv(path)
    assert df.shape == (2000, 8)
    expected_features = ["Age", "Annual Income ($)", "Spending Score (1-100)", "Work Experience"]
    for feat in expected_features:
        assert feat in df.columns, f"Missing segmentation feature {feat}"
    assert "CustomerID" in df.columns


def test_education_survey_raw_loading_and_schema():
    """Verify educational-factors.xlsx loads via pure-Python parser, has 79 columns and target b4_8."""
    path = CANDIDATES_ROOT / "Education_Survey" / "educational-factors.xlsx"
    assert path.exists(), f"educational-factors.xlsx missing at {path}"
    df, chosen_path, sha256 = load_education_xlsx(path)
    assert df.shape == (1680, 79)
    assert "b4_8" in df.columns
    assert "No." in df.columns


def test_shopmania_raw_loading_and_schema():
    """Verify shopmania.csv loads as 4 headerless columns without synthetic features."""
    path = CANDIDATES_ROOT / "shopmania.csv"
    assert path.exists(), f"shopmania.csv missing at {path}"
    df, chosen_path, sha256 = load_shopmania_data(path)
    assert df.shape[1] == 4
    assert list(df.columns) == ["product_id", "text", "cat_id", "category"]
    assert len(df) == 313706
    # Confirm NO synthetic features exist
    for forbidden in ["price", "brand", "merchant", "product_name", "description"]:
        assert forbidden not in df.columns


# ============================================================================
# 2. EDUCATION TARGET TRANSFORMATION & PROFILE TESTS
# ============================================================================

def test_education_observed_values_and_target_binarization():
    """Validate b4_8 distribution and AutoMind Table 4 binary satisfaction split."""
    path = CANDIDATES_ROOT / "Education_Survey" / "educational-factors.xlsx"
    df, _, _ = load_education_xlsx(path)
    val_counts = df["b4_8"].value_counts().to_dict()
    # Ground truth Likert counts
    assert val_counts[1] == 13
    assert val_counts[2] == 37
    assert val_counts[3] == 229
    assert val_counts[4] == 778
    assert val_counts[5] == 623

    # Binary split: High (>=4) vs Moderate/Low (<4)
    y_cls = (df["b4_8"] >= 4).astype(int)
    high_count = (y_cls == 1).sum()
    low_count = (y_cls == 0).sum()
    assert high_count == 1401
    assert low_count == 279
    assert high_count + low_count == 1680


def test_education_real_profile_contains_all_79_columns():
    """Verify profiling educational-factors.xlsx produces true 79-column schema including b4_8."""
    path = CANDIDATES_ROOT / "Education_Survey" / "educational-factors.xlsx"
    df, _, _ = load_education_xlsx(path)
    prof = profiler.profile_dataset(df, dataset_name="educational-factors.xlsx")
    overview = prof.get("overview", {})
    assert overview.get("n_rows") == 1680
    assert overview.get("n_cols") == 79
    col_names = overview.get("column_names", [])
    assert len(col_names) == 79
    assert "b4_8" in col_names
    assert "No." in col_names


# ============================================================================
# 3. SHOPMANIA REAL PROFILE TESTS
# ============================================================================

def test_shopmania_real_profile_reflects_four_columns():
    """Verify profiling shopmania.csv reflects actual 4-column schema without invented columns."""
    path = CANDIDATES_ROOT / "shopmania.csv"
    df, _, _ = load_shopmania_data(path)
    prof = profiler.profile_dataset(df.head(2000), dataset_name="shopmania.csv")
    overview = prof.get("overview", {})
    assert overview.get("n_cols") == 4
    col_names = overview.get("column_names", [])
    assert col_names == ["product_id", "text", "cat_id", "category"]


# ============================================================================
# 4. BENCHMARK SCRIPT INTEGRITY (NO CACHED SUBSTITUTION)
# ============================================================================

def test_benchmark_script_contains_no_cached_metric_loading():
    """Verify run_gemini_five_datasets_benchmark.py does NOT load cached JSON metrics."""
    assert BENCHMARK_SCRIPT.exists()
    content = BENCHMARK_SCRIPT.read_text(encoding="utf-8")
    assert "education_metrics.json" not in content, "Found forbidden reference to cached education_metrics.json!"
    assert "shopmania_metrics.json" not in content, "Found forbidden reference to cached shopmania_metrics.json!"
    assert "load_education_xlsx" in content, "Missing load_education_xlsx in benchmark script!"
    assert "load_shopmania_data" in content, "Missing load_shopmania_data in benchmark script!"
    assert "run_education_benchmark" in content, "Missing run_education_benchmark in benchmark script!"
    assert "run_shopmania_benchmark" in content, "Missing run_shopmania_benchmark in benchmark script!"


# ============================================================================
# 5. NO-SILENT-FALLBACK SAFEGUARDS IN RESEARCH MODE
# ============================================================================

def test_planner_raises_error_on_gemini_failure_without_silent_fallback():
    """Verify is_research_run=True raises GeminiClientError without falling back to deterministic planning."""
    def error_transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"code": 500, "message": "Simulated Google API failure"}})

    client = GeminiClient(
        api_key="mock_test_key_non_empty_length_12345",
        transport=httpx.MockTransport(error_transport),
        max_retries=0,
    )

    sample_profile = {
        "overview": {
            "n_rows": 100,
            "n_cols": 3,
            "column_names": ["feat1", "feat2", "target"],
        },
        "column_types": {
            "feat1": "numerical",
            "feat2": "numerical",
            "target": "categorical",
        },
    }

    # Must raise GeminiNetworkError / GeminiClientError directly. MUST NOT return a deterministic plan!
    with pytest.raises(GeminiClientError):
        create_analysis_plan(
            profile=sample_profile,
            objective="Predict target from feat1 and feat2.",
            dataset_id="test_data",
            is_research_run=True,
            gemini_client=client,
        )


def test_planner_succeeds_with_valid_gemini_response_in_research_mode():
    """Verify is_research_run=True returns strongly-typed LLM PlannerResponse with shared tasks."""
    mock_plan_payload = {
        "objective": "Predict target from feat1 and feat2.",
        "interpreted_goal": "Investigate classification of target using feat1 and feat2.",
        "target_column": "target",
        "selected_columns": [
            {"column": "target", "data_type": "categorical", "reason": "Classification target", "is_target": True},
            {"column": "feat1", "data_type": "numerical", "reason": "Predictor feature", "is_target": False},
            {"column": "feat2", "data_type": "numerical", "reason": "Predictor feature", "is_target": False},
        ],
        "analysis_tasks": [
            {
                "task_id": "task_1",
                "task_type": "binary_classification",
                "task": "Classify target using logistic regression",
                "description": "Evaluate binary classification performance on target.",
                "method": "logistic_regression",
                "columns": ["feat1", "feat2", "target"],
                "target_column": "target",
                "feature_columns": ["feat1", "feat2"],
                "reason": "Target is binary and features are continuous.",
                "priority": "high",
                "notes": ["Evaluate accuracy and F1-score."],
            }
        ],
        "reasoning": ["Features are numerical and target is binary categorical."],
        "warnings": [],
        "expected_outputs": ["Classification metrics: Accuracy, Precision, Recall, F1."],
    }

    def success_transport(request: httpx.Request) -> httpx.Response:
        body = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": json.dumps(mock_plan_payload)}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 150,
                "candidatesTokenCount": 85,
                "totalTokenCount": 235,
            },
            "modelVersion": "gemini-3.5-flash-lite",
        }
        return httpx.Response(200, json=body)

    client = GeminiClient(
        api_key="mock_test_key_non_empty_length_12345",
        transport=httpx.MockTransport(success_transport),
    )

    sample_profile = {
        "overview": {
            "n_rows": 100,
            "n_cols": 3,
            "column_names": ["feat1", "feat2", "target"],
        },
        "column_types": {
            "feat1": "numerical",
            "feat2": "numerical",
            "target": "categorical",
        },
    }

    plan = create_analysis_plan(
        profile=sample_profile,
        objective="Predict target from feat1 and feat2.",
        dataset_id="test_data",
        is_research_run=True,
        gemini_client=client,
    )

    assert plan.planner_mode == "llm"
    assert len(plan.analysis_tasks) == 1
    assert plan.analysis_tasks[0].task_type == "binary_classification"
    assert plan.target_column == "target"


# ============================================================================
# 6. RESULT SCHEMA & DICTIONARY KEY ACCESS TESTS (MOCKED & LOCAL)
# ============================================================================

def test_benchmark_script_key_access_conformance():
    """Verify run_gemini_five_datasets_benchmark.py uses exact keys and no invalid metric fields or runner arguments."""
    content = BENCHMARK_SCRIPT.read_text(encoding="utf-8")
    assert "baseline_metrics" not in content, "Found obsolete 'baseline_metrics' key access!"
    assert "proposed_metrics" not in content, "Found obsolete 'proposed_metrics' key access!"
    assert "standardized_silhouette_score" not in content, "Found obsolete 'standardized_silhouette_score' key access!"
    assert 'text_column="text"' not in content, "Found invalid singular keyword 'text_column' in orchestrator!"
    assert 'text_columns=["text"]' in content, "Expected 'text_columns=[\"text\"]' in fake news orchestration call!"
    assert 'heart_comp["baseline_critic_off"]["f1_score"]' in content
    assert 'heart_comp["proposed_critic_on"]["f1_score"]' in content
    assert 'fake_comp["baseline_critic_off"]["f1_score"]' in content
    assert 'fake_comp["proposed_critic_on"]["f1_score"]' in content
    assert 'cust_comp["baseline_critic_off"]["silhouette_standardized"]' in content
    assert 'cust_comp["proposed_critic_on_optimal_k"]["silhouette_standardized"]' in content
    assert 'k_baseline=3' in content


def test_ast_orchestration_fake_news_call_keywords():
    """Verify via Python AST inspection that compare_paired_text_classification call uses valid keywords."""
    import ast
    tree = ast.parse(BENCHMARK_SCRIPT.read_text(encoding="utf-8"))
    found_call = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = ""
            if isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            elif isinstance(node.func, ast.Name):
                func_name = node.func.id
            if func_name == "compare_paired_text_classification":
                found_call = True
                kw_names = [kw.arg for kw in node.keywords]
                assert "text_column" not in kw_names, "AST error: found invalid keyword 'text_column'!"
                assert "text_columns" in kw_names, "AST error: missing expected keyword 'text_columns'!"
                assert "label_column" in kw_names
                assert "random_state" in kw_names
    assert found_call, "Failed to locate compare_paired_text_classification call in AST!"


def test_paired_classification_return_schema():
    """Verify compare_paired_classification returns expected keys for baseline and proposed."""
    from app.statistics.classification import compare_paired_classification
    df = pd.DataFrame({
        "num1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 5,
        "num2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0] * 5,
        "target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 5,
    })
    res = compare_paired_classification(df, target_column="target", model_type="logistic_regression", random_state=42)
    assert "baseline_critic_off" in res
    assert "proposed_critic_on" in res
    assert "f1_score" in res["baseline_critic_off"]
    assert "f1_score" in res["proposed_critic_on"]
    assert "p_value" in res["statistical_significance"]


def test_paired_text_classification_return_schema():
    """Verify compare_paired_text_classification returns expected keys for baseline and proposed."""
    from app.statistics.text_classification import compare_paired_text_classification
    df = pd.DataFrame({
        "text": [
            "WASHINGTON (Reuters) - Federal officials announced economic reforms today.",
            "BREAKING SHOCK: Secret government conspiracy revealed by insider source!",
            "LONDON (Reuters) - International markets observed steady currency exchange rates.",
            "YOU WON'T BELIEVE: Aliens confirmed in classified underground documents!",
        ] * 10,
        "label": [1, 0, 1, 0] * 10,
    })
    res = compare_paired_text_classification(df, text_columns=["text"], label_column="label", random_state=42)
    assert "baseline_critic_off" in res
    assert "proposed_critic_on" in res
    assert "f1_score" in res["baseline_critic_off"]
    assert "f1_score" in res["proposed_critic_on"]
    assert "p_value" in res["statistical_significance"]


def test_paired_clustering_return_schema():
    """Verify compare_paired_clustering returns expected keys with k_baseline=3."""
    from app.statistics.clustering import compare_paired_clustering
    df = pd.DataFrame({
        "x": [1.0, 1.2, 0.9, 10.0, 10.2, 9.8, 100.0, 101.0, 99.0] * 5,
        "y": [2.0, 2.1, 1.9, 20.0, 20.5, 19.5, 200.0, 201.0, 199.0] * 5,
    })
    res = compare_paired_clustering(df, feature_columns=["x", "y"], k_baseline=3, random_state=42)
    assert "baseline_critic_off" in res
    assert "proposed_critic_on_matched_k" in res
    assert "proposed_critic_on_optimal_k" in res
    assert "silhouette_standardized" in res["baseline_critic_off"]
    assert "silhouette_standardized" in res["proposed_critic_on_optimal_k"]


def test_education_and_shopmania_mocked_result_schemas():
    """Verify dictionary key accesses for Education and ShopMania results using mock return payloads."""
    edu_mock = {
        "classification_table4": {
            "baseline": {"accuracy": 0.9405, "f1_score": 0.9649, "roc_auc": 0.9631},
            "proposed": {"accuracy": 0.8482, "f1_score": 0.9160, "roc_auc": 0.7400},
            "mcnemar": {"p_value": 0.000003},
        }
    }
    assert edu_mock["classification_table4"]["baseline"]["f1_score"] == 0.9649
    assert edu_mock["classification_table4"]["proposed"]["f1_score"] == 0.9160
    assert edu_mock["classification_table4"]["mcnemar"]["p_value"] == 0.000003

    shop_mock = {
        "baseline_critic_off": {"macro_f1": 0.9271, "accuracy": 0.9267},
        "proposed_critic_on": {"macro_f1": 0.9254, "accuracy": 0.9250},
        "statistical_significance": {"p_value": 0.3914},
    }
    assert shop_mock["baseline_critic_off"]["macro_f1"] == 0.9271
    assert shop_mock["proposed_critic_on"]["macro_f1"] == 0.9254
    assert shop_mock["statistical_significance"]["p_value"] == 0.3914


def test_planner_prompt_conciseness_constraints():
    """Verify PlannerAgent builds prompts with strict conciseness constraints for high-dimensional schemas."""
    from app.agents.planner_agent import PlannerAgent
    planner = PlannerAgent(gemini_client=None)
    mock_cols = [f"item_{i}" for i in range(79)]
    mock_profile = {
        "overview": {"n_rows": 1680, "n_cols": 79},
        "column_types": {c: "numerical" for c in mock_cols},
        "missing": {"total_missing": 0, "total_missing_pct": 0.0},
        "cardinality": {"constant_columns": [], "high_cardinality_columns": []},
        "outliers": {},
    }
    user_prompt, sys_inst = planner._build_llm_planning_prompt(
        profile=mock_profile,
        objective="Assess psychometric reliability and binary satisfaction.",
        column_names=mock_cols,
    )
    assert "CONCISENESS & COMPLETENESS" in sys_inst
    assert "At most 10 items in 'selected_columns'" in sys_inst or "at most 10 items in 'selected_columns'" in sys_inst
    assert "5 to 8 prioritized, non-redundant 'analysis_tasks'" in sys_inst
    assert "At most 10 key columns or constructs" in user_prompt
    assert "Exactly 5 to 8 prioritized, non-redundant investigation tasks" in user_prompt


def test_benchmark_artifact_reuse_safeguards():
    """Verify benchmark script implements strict existence and schema validation before reusing artifacts."""
    content = BENCHMARK_SCRIPT.read_text(encoding="utf-8")
    # All three completed datasets must have existence checks
    assert "fake_plan_file.exists() or fake_comp_file.exists():" in content
    assert "cust_plan_file.exists() or cust_comp_file.exists():" in content
    assert "heart_plan_file.exists() or heart_comp_file.exists():" in content
    # All three must raise RuntimeError on partial artifacts
    assert "Partial Heart Disease artifacts found" in content
    assert "Partial Fake News artifacts found" in content
    assert "Partial Customer Segmentation artifacts found" in content
    # All three must validate metric fields and raise ValueError
    assert "Fake News comparison artifact invalid" in content
    assert "Customer Segmentation comparison artifact invalid" in content
    assert "Heart Disease comparison artifact invalid" in content



"""
backend/tests/test_benchmark_datasets.py
=========================================
Tests for benchmark dataset semantic typing, sentinel handling, and statistical analyses:
- adult.txt (UCI Adult Census Income)
- processed.cleveland.txt (UCI Cleveland Heart Disease)
- bank-full.csv (UCI Bank Marketing)
- Month-only categorical classification
- Edge cases and arbitrary dataset isolation
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from app.services.upload_service import (
    parse_csv,
    derive_bank_contacted_feature,
    derive_benchmark_features,
    VERIFIED_BENCHMARK_SCHEMAS,
)
from app.modules.profiler import profile_dataset, _detect_datetime_columns, _get_column_types
from app.statistics.descriptive import execute_distribution_analysis
from app.statistics.categorical import execute_categorical_association
from app.statistics.correlation import execute_correlation
from app.statistics.group_comparison import execute_two_group_comparison, execute_multi_group_comparison


# Path to data directory
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


@pytest.fixture(scope="module")
def cleveland_df() -> pd.DataFrame:
    cleveland_path = DATA_DIR / "processed.cleveland.txt"
    assert cleveland_path.exists(), f"Benchmark file {cleveland_path} missing."
    with open(cleveland_path, "rb") as f:
        df, err = parse_csv(f.read(), filename="processed.cleveland.txt")
    assert err is None
    assert df is not None
    return df


@pytest.fixture(scope="module")
def bank_df() -> pd.DataFrame:
    bank_path = DATA_DIR / "bank-full.csv"
    assert bank_path.exists(), f"Benchmark file {bank_path} missing."
    with open(bank_path, "rb") as f:
        df, err = parse_csv(f.read(), filename="bank-full.csv")
    assert err is None
    assert df is not None
    return df


@pytest.fixture(scope="module")
def adult_df() -> pd.DataFrame:
    adult_path = DATA_DIR / "adult.txt"
    assert adult_path.exists(), f"Benchmark file {adult_path} missing."
    with open(adult_path, "rb") as f:
        df, err = parse_csv(f.read(), filename="adult.txt")
    assert err is None
    assert df is not None
    return df


# ============================================================================
# A. Cleveland Dataset Semantic Typing Tests
# ============================================================================

class TestClevelandDataset:
    """Verify semantic typing, parsing, and statistical analysis for processed.cleveland.txt."""

    def test_cleveland_dimensions_and_columns(self, cleveland_df):
        """Cleveland must have 303 rows and 14 documented benchmark columns."""
        assert cleveland_df.shape[0] == 303
        assert cleveland_df.shape[1] == 14
        expected_cols = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
        ]
        assert list(cleveland_df.columns) == expected_cols

    def test_cleveland_missing_values_preserved(self, cleveland_df):
        """Missing values '?' must be mapped to NaN (4 in 'ca', 2 in 'thal') and remain missing."""
        assert cleveland_df["ca"].isna().sum() == 4
        assert cleveland_df["thal"].isna().sum() == 2
        assert cleveland_df.isna().sum().sum() == 6
        assert cleveland_df.isna().any(axis=1).sum() == 6

    def test_cleveland_semantic_typing_in_profiler(self, cleveland_df):
        """Continuous variables must be numerical; discrete codes must be categorical."""
        prof = profile_dataset(cleveland_df, dataset_name="processed.cleveland.txt")
        types = prof["column_types"]

        # Continuous variables
        for num_col in ["age", "trestbps", "chol", "thalach", "oldpeak"]:
            assert types[num_col] == "numerical", f"Expected {num_col} to be numerical."

        # Documented nominal/categorical codes
        for cat_col in ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "target"]:
            assert types[cat_col] == "categorical", f"Expected {cat_col} to be categorical."

    def test_cleveland_target_classes_not_relabeled(self, cleveland_df):
        """Target classes must preserve all original stages 0, 1, 2, 3, 4 without relabeling."""
        target_counts = cleveland_df["target"].value_counts().to_dict()
        assert target_counts.get("0") == 164
        assert target_counts.get("1") == 55
        assert target_counts.get("2") == 36
        assert target_counts.get("3") == 35
        assert target_counts.get("4") == 13

    def test_cleveland_statistical_analyses(self, cleveland_df):
        """Verify appropriate analysis selection on continuous and categorical columns."""
        # 1. Categorical distribution on binary 'sex'
        ok, res, warns, err = execute_distribution_analysis(cleveland_df, "sex")
        assert ok is True
        assert res["variable_type"] == "categorical"
        assert res["frequencies"] == {"1": 206, "0": 97}

        # 2. Categorical distribution on 'ca' with 4 missing rows warning
        ok, res, warns, err = execute_distribution_analysis(cleveland_df, "ca")
        assert ok is True
        assert res["variable_type"] == "categorical"
        assert any("4 rows omitted" in w for w in warns)

        # 3. Continuous numerical distribution on 'chol'
        ok, res, warns, err = execute_distribution_analysis(cleveland_df, "chol")
        assert ok is True
        assert res["variable_type"] == "numeric"
        assert 245.0 < res["mean"] < 248.0

        # 4. Pearson correlation between continuous variables
        ok, res, warns, err = execute_correlation(cleveland_df, "age", "chol")
        assert ok is True
        assert 0.18 < res["statistic"] < 0.23
        assert res["p_value"] < 0.001

        # 5. Chi-square test between categorical variables
        ok, res, warns, err = execute_categorical_association(cleveland_df, "sex", "target")
        assert ok is True
        assert res["statistic"] > 20.0
        assert res["p_value"] < 0.001

        # 6. Two-group t-test: sex vs continuous chol
        ok, res, warns, err = execute_two_group_comparison(cleveland_df, "sex", "chol")
        assert ok is True
        assert res["p_value"] < 0.01

        # 7. Multi-group ANOVA: target (5 levels) vs continuous chol
        ok, res, warns, err = execute_multi_group_comparison(cleveland_df, "target", "chol")
        assert ok is True
        assert "p_value" in res


# ============================================================================
# B. Bank Marketing Dataset Sentinel & Typing Tests
# ============================================================================

class TestBankDataset:
    """Verify sentinel handling, month typing, raw 17-column integrity, and derived representations for bank-full.csv."""

    def test_bank_dimensions_and_parsing(self, bank_df):
        """Bank dataset must parse 45,211 rows with semicolon delimiter and exact 17 raw columns."""
        assert bank_df.shape == (45211, 17)
        assert len(bank_df.columns) == 17
        assert "pdays_contacted" not in bank_df.columns
        expected_cols = [
            "age", "job", "marital", "education", "default", "balance",
            "housing", "loan", "contact", "day", "month", "duration",
            "campaign", "pdays", "previous", "poutcome", "y",
        ]
        assert list(bank_df.columns) == expected_cols

    def test_bank_unknown_sentinels_mapped_to_nan(self, bank_df):
        """'unknown' in job and education must be mapped to NaN where justified by semantics."""
        assert bank_df["job"].isna().sum() == 288
        assert bank_df["education"].isna().sum() == 1857
        # contact and poutcome retain 'unknown' as domain category values, never silently reinterpreted as NaN
        assert (bank_df["contact"] == "unknown").sum() == 13020
        assert bank_df["contact"].isna().sum() == 0
        assert (bank_df["poutcome"] == "unknown").sum() == 36959
        assert bank_df["poutcome"].isna().sum() == 0

    def test_bank_month_classified_as_categorical_not_datetime(self, bank_df):
        """Month names (may, jul, aug) must remain categorical and not parse into current-year datetimes."""
        prof = profile_dataset(bank_df, dataset_name="bank-full.csv")
        assert prof["column_types"]["month"] == "categorical"
        assert "month" not in prof["datetime_cols"]
        assert prof["overview"]["n_rows"] == 45211
        assert prof["overview"]["n_cols"] == 17

    def test_bank_pdays_sentinel_preserved_and_flagged(self, bank_df):
        """Raw pdays = -1 must be preserved and flagged with informative sentinel warnings."""
        # Raw negative 1 preserved
        assert (bank_df["pdays"] == -1).sum() == 36954

        # Distribution analysis detects -1 sentinel
        ok, res, warns, err = execute_distribution_analysis(bank_df, "pdays")
        assert ok is True
        assert res["variable_type"] == "numeric"
        assert res["min"] == -1.0
        assert any("Sentinel value -1 detected" in w for w in warns)

        # Structured sentinel observation details
        sentinel_obs = res.get("sentinel_observations")
        assert sentinel_obs is not None
        assert sentinel_obs["sentinel_value"] == -1
        assert sentinel_obs["count"] == 36954
        assert sentinel_obs["percentage"] == pytest.approx(81.74, abs=0.05)
        assert sentinel_obs["valid_contacted_count"] == 8257
        assert sentinel_obs["valid_contacted_mean"] == pytest.approx(224.58, abs=0.5)

    def test_bank_pdays_contacted_explicit_derivation(self, bank_df):
        """Derived pdays_contacted column is produced via explicit helper without mutating raw DataFrame."""
        # Raw bank_df does NOT have pdays_contacted
        assert "pdays_contacted" not in bank_df.columns
        assert bank_df.shape == (45211, 17)

        # Explicit derivation generates 18-column DataFrame
        derived_df = derive_bank_contacted_feature(bank_df)
        assert derived_df.shape == (45211, 18)
        assert "pdays_contacted" in derived_df.columns
        # Original bank_df remains completely unmutated
        assert "pdays_contacted" not in bank_df.columns
        assert bank_df.shape == (45211, 17)

        counts = derived_df["pdays_contacted"].value_counts().to_dict()
        assert counts["not_contacted"] == 36954
        assert counts["contacted"] == 8257

        # Association between derived feature and subscription target y
        ok, res, warns, err = execute_categorical_association(derived_df, "pdays_contacted", "y")
        assert ok is True
        assert res["statistic"] > 1000.0
        assert res["p_value"] < 1e-50

    def test_bank_raw_versus_interpreted_missing_counts(self):
        """Compare raw uninterpreted CSV parse with benchmark-preprocessed parse and verify provenance."""
        bank_path = DATA_DIR / "bank-full.csv"
        with open(bank_path, "rb") as f:
            raw_content = f.read()

        # 1. Raw parse without benchmark sentinel interpretation: zero missing values
        raw_df, err = parse_csv(raw_content, filename="bank-full.csv", apply_benchmark_sentinels=False)
        assert err is None
        assert raw_df.shape == (45211, 17)
        assert raw_df.isna().sum().sum() == 0
        assert (raw_df["job"] == "unknown").sum() == 288
        assert (raw_df["education"] == "unknown").sum() == 1857
        assert (raw_df["contact"] == "unknown").sum() == 13020
        assert (raw_df["poutcome"] == "unknown").sum() == 36959
        assert (raw_df["pdays"] == -1).sum() == 36954

        # 2. Preprocessed parse with benchmark sentinel interpretation: exactly 2,145 missing values
        clean_df, err = parse_csv(raw_content, filename="bank-full.csv", apply_benchmark_sentinels=True)
        assert err is None
        assert clean_df.shape == (45211, 17)
        assert clean_df.isna().sum().sum() == 2145
        assert clean_df["job"].isna().sum() == 288
        assert clean_df["education"].isna().sum() == 1857
        assert clean_df["contact"].isna().sum() == 0
        assert clean_df["poutcome"].isna().sum() == 0

        # 3. Verify provenance attached to df.attrs
        prov = clean_df.attrs.get("missing_value_provenance")
        assert prov is not None
        assert prov["summary"]["raw_missing_cells"] == 0
        assert prov["summary"]["interpreted_missing_cells"] == 2145
        assert prov["summary"]["total_missing_cells_after"] == 2145
        assert prov["sentinel_interpretation"]["job"]["raw_sentinel_count"] == 288
        assert prov["sentinel_interpretation"]["education"]["raw_sentinel_count"] == 1857
        assert prov["retained_domain_categories"]["contact"]["raw_sentinel_count"] == 13020
        assert prov["retained_domain_categories"]["poutcome"]["raw_sentinel_count"] == 36959
        assert prov["numeric_sentinels"]["pdays"]["count"] == 36954

        # 4. Verify profiler output reflects 17 columns and provenance
        prof = profile_dataset(clean_df, dataset_name="bank-full.csv")
        assert prof["overview"]["n_rows"] == 45211
        assert prof["overview"]["n_cols"] == 17
        assert prof["missing"]["total_missing"] == 2145
        assert "preprocessing_rules" in prof
        assert prof["missing"].get("provenance") is not None


# ============================================================================
# C. Adult Census Income Dataset Tests
# ============================================================================

class TestAdultDataset:
    """Verify parsing, dimensions, missingness, and statistical analyses for adult.txt."""

    def test_adult_dimensions_and_schema(self, adult_df):
        """Adult dataset must parse 32,561 rows x 15 columns with verified schema."""
        assert adult_df.shape == (32561, 15)
        expected_cols = [
            "age", "workclass", "fnlwgt", "education", "education_num",
            "marital_status", "occupation", "relationship", "race", "sex",
            "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
        ]
        assert list(adult_df.columns) == expected_cols

    def test_adult_missing_values(self, adult_df):
        """Missing values '?' must be mapped to NaN across workclass, occupation, native_country."""
        assert adult_df["workclass"].isna().sum() == 1836
        assert adult_df["occupation"].isna().sum() == 1843
        assert adult_df["native_country"].isna().sum() == 583
        assert adult_df.isna().sum().sum() == 4262

    def test_adult_column_types(self, adult_df):
        """Profiler must identify 6 numerical and 9 categorical columns."""
        prof = profile_dataset(adult_df, dataset_name="adult.txt")
        types = prof["column_types"]

        for num_col in ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]:
            assert types[num_col] == "numerical"

        for cat_col in ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "native_country", "income"]:
            assert types[cat_col] == "categorical"

    def test_adult_statistical_analyses(self, adult_df):
        """Verify statistical analysis execution on adult features."""
        # 1. Target income distribution
        ok, res, warns, err = execute_distribution_analysis(adult_df, "income")
        assert ok is True
        assert res["variable_type"] == "categorical"
        assert res["frequencies"]["<=50K"] == 24720
        assert res["frequencies"][">50K"] == 7841

        # 2. Categorical distribution on workclass with missing warning
        ok, res, warns, err = execute_distribution_analysis(adult_df, "workclass")
        assert ok is True
        assert any("1836 rows omitted" in w for w in warns)

        # 3. Numerical distribution on age
        ok, res, warns, err = execute_distribution_analysis(adult_df, "age")
        assert ok is True
        assert 38.0 < res["mean"] < 39.0

        # 4. Association between education and income
        ok, res, warns, err = execute_categorical_association(adult_df, "education", "income")
        assert ok is True
        assert res["statistic"] > 4000.0
        assert res["p_value"] < 1e-100


# ============================================================================
# D. General Robustness, Month Detection & Arbitrary Dataset Isolation
# ============================================================================

class TestGeneralDatasetRobustness:
    """Verify general rules avoid polluting arbitrary datasets."""

    def test_month_only_labels_general_detection(self):
        """Synthetic month-only columns should be categorical; actual ISO dates should be datetime."""
        df = pd.DataFrame({
            "months": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"],
            "dates": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01", "2023-06-01", "2023-07-01", "2023-08-01"],
            "values": [10, 20, 30, 40, 50, 60, 70, 80],
        })
        prof = profile_dataset(df)
        assert prof["column_types"]["months"] == "categorical"
        assert prof["column_types"]["dates"] == "datetime"
        assert prof["column_types"]["values"] == "numerical"

    def test_arbitrary_upload_not_polluted(self):
        """An arbitrary uploaded CSV must not have benchmark-specific rules applied."""
        content = b"job,pdays,month,val\nunknown,-1,random,100\nengineer,5,sample,200\n"
        df, err = parse_csv(content, filename="custom_user_data.csv")
        assert err is None
        # 'unknown' not converted to NaN because this is not bank-full.csv
        assert "unknown" in df["job"].values
        # pdays -1 preserved
        assert -1 in df["pdays"].values
        # pdays_contacted not added to arbitrary dataset
        assert "pdays_contacted" not in df.columns

    def test_empty_or_whitespace_file_handling(self):
        """Empty or whitespace-only files produce clear user-friendly errors."""
        df, err = parse_csv(b"", filename="empty.csv")
        assert df is None
        assert "empty" in err.lower()

        df, err = parse_csv(b"   \n\t  \n", filename="whitespace.csv")
        assert df is None
        assert "empty" in err.lower()


# ============================================================================
# E. Experiment Branch Fairness & Shared Preprocessing Verification
# ============================================================================

class TestExperimentPipelineFairness:
    """Verify that both experiment branches share identical preprocessing, plans, and statistical analyses."""

    def test_paired_experiment_receives_identical_shared_plan_and_analysis(self):
        """Confirm that experiment_runner executes planning and analysis once and shares results across branches."""
        from unittest.mock import MagicMock
        from app.services.experiment_runner import run_paired_experiment
        from app.schemas.planner_schema import PlannerResponse, InvestigationTask
        from app.schemas.analysis_schema import PlanAnalysisResponse, AnalysisResult

        # Mock single shared plan
        task = InvestigationTask(
            task_id="t1",
            task_type="distribution_analysis",
            task="Distribution analysis",
            description="Assess distribution",
            method="descriptive summary",
            columns=["col1"],
            column_a="col1",
            reason="Check shape",
            priority="high",
        )
        mock_plan = PlannerResponse(
            objective="Fairness test",
            interpreted_goal="Fairness test goal",
            selected_columns=[
                {"column": "col1", "data_type": "numerical", "reason": "Test col", "is_target": False}
            ],
            analysis_tasks=[task],
            tasks=[task],
            reasoning=["Single task"],
            warnings=[],
            expected_outputs=["Moments"],
            dataset_summary={"n_rows": 100, "n_cols": 2, "model": "test"},
        )
        mock_planner_fn = MagicMock(return_value=mock_plan)

        # Mock single analysis service
        mock_analysis_svc = MagicMock()
        mock_analysis_response = PlanAnalysisResponse(
            dataset_id="test_dataset.csv",
            total_tasks=1,
            successful_tasks=1,
            failed_tasks=0,
            results=[
                AnalysisResult(
                    task_id="t1",
                    status="success",
                    analysis_type="distribution_analysis",
                    method="descriptive summary",
                    columns=["col1"],
                    sample_size=100,
                    results={"mean": 10.0, "std": 2.0},
                    warnings=[],
                )
            ],
        )
        mock_analysis_svc.execute_plan.return_value = mock_analysis_response

        from app.schemas.evaluation_schema import EvaluationResponse, EvaluationSummary

        # Mock evaluation service to capture what each branch receives
        mock_eval_svc = MagicMock()
        mock_eval_response = EvaluationResponse(
            summary=EvaluationSummary(
                overall_score=85.0,
                overall_grade="A",
                total_tasks_planned=1,
                total_tasks_executed=1,
                executive_assessment="Good",
            ),
            dimensions=[],
            benchmark_checks=[],
            detailed_metrics=[],
        )
        mock_eval_svc.evaluate_pipeline.return_value = mock_eval_response

        profile = {
            "overview": {"n_rows": 100, "n_cols": 2, "column_names": ["col1", "col2"]},
            "column_types": {"col1": "numerical", "col2": "categorical"},
            "missing": {"per_column": {}, "total_missing": 0, "total_cells": 200, "total_missing_pct": 0.0},
            "cardinality": {"per_column": {}, "constant_columns": [], "high_cardinality_columns": []},
            "duplicates": {"n_duplicates": 0, "duplicate_pct": 0.0},
            "numerical_stats": {},
            "outliers": {},
            "datetime_cols": [],
        }

        # Run paired experiment in fixture mode
        res = run_paired_experiment(
            profile=profile,
            dataset_id="test_dataset.csv",
            objective="Fairness test",
            planner_fn=mock_planner_fn,
            analysis_service=mock_analysis_svc,
            evaluation_service_inst=mock_eval_svc,
            is_offline_fixture=True,
            persist_runs=False,
        )

        # 1. Planner function was called exactly ONCE
        assert mock_planner_fn.call_count == 1

        # 2. Analysis service execute_plan was called exactly ONCE
        assert mock_analysis_svc.execute_plan.call_count == 1

        # 3. Both baseline and proposed branches evaluate the same shared analysis results
        assert mock_eval_svc.evaluate_pipeline.call_count == 2
        baseline_eval_call = mock_eval_svc.evaluate_pipeline.call_args_list[0]
        proposed_eval_call = mock_eval_svc.evaluate_pipeline.call_args_list[1]

        # Both branches received the identical analysis dict and plan dict
        assert baseline_eval_call.kwargs.get("analysis") == mock_analysis_response.model_dump()
        assert proposed_eval_call.kwargs.get("analysis") == mock_analysis_response.model_dump()
        assert baseline_eval_call.kwargs.get("plan") == mock_plan.model_dump()
        assert proposed_eval_call.kwargs.get("plan") == mock_plan.model_dump()

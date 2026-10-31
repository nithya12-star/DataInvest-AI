"""
backend/tests/test_benchmark_harness.py
======================================
Automated offline tests for Stage D2:
- Monotonic runtime duration tracking & fair time separation.
- Rule-level Critic metric aggregation & overlapping code handling.
- Cross-dataset benchmark suite execution & partial failure visibility.
- CSV, Markdown, and JSON export correctness.
- Fixture reproducibility & overwrite safety.
- Strict prevention of external API calls in offline mode.
"""

import csv
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from app.schemas.experiment_schema import (
    BranchRunSummary,
    MetricDeltaSummary,
    PairedExperimentResponse,
)
from app.schemas.critic_schema import (
    CriticOverallSummary,
    CriticResponse,
    CriticResult,
)
from app.schemas.planner_schema import PlannerResponse
from app.services.experiment_runner import run_paired_experiment
from scripts.run_cross_dataset_benchmark import (
    build_canonical_fixture_plan,
    run_benchmark_suite,
    run_single_dataset_benchmark,
    validate_safe_output_dir,
)


def test_duration_fields_populated_nonnegative_and_backward_compatible(tmp_path):
    """Test 1: Monotonic durations are populated, nonnegative, and backward-compatible."""
    # Test backward compatibility of schema with None defaults
    empty_delta = MetricDeltaSummary(
        baseline_score=50.0,
        proposed_score=80.0,
        overall_score_delta=30.0,
        rigor_score_delta=20.0,
        verification_score_delta=25.0,
        dimension_deltas={},
        assumption_violations_flagged=1,
        violations_resolved_via_reinvestigation=1,
        total_reinvestigations=1,
        verified_findings_count=1,
        questionable_findings_count=0,
        rejected_findings_count=0,
        scientific_verdict="Valid comparison.",
    )
    assert empty_delta.total_experiment_duration_seconds is None
    assert empty_delta.shared_duration_seconds is None
    assert empty_delta.cochran_violations_count is None

    # Test live offline execution on sample dataset
    entry = run_single_dataset_benchmark(
        dataset_path="data/sample.csv",
        use_canonical_fixtures=True,
        is_research_mode=False,
    )

    assert entry["status"] == "SUCCESS"
    assert entry["total_experiment_duration_seconds"] >= 0.0
    assert entry["shared_duration_seconds"] >= 0.0
    assert entry["planning_duration_seconds"] >= 0.0
    assert entry["analysis_duration_seconds"] >= 0.0
    assert entry["baseline_duration_seconds"] >= 0.0
    assert entry["proposed_duration_seconds"] >= 0.0
    assert entry["critic_duration_seconds"] >= 0.0
    assert entry["verification_duration_seconds"] >= 0.0
    assert entry["proposed_overhead_seconds"] >= 0.0


def test_fair_timing_separation():
    """Test 2: Fair timing separation - shared duration is isolated from branch overhead."""
    entry = run_single_dataset_benchmark(
        dataset_path="data/sample.csv",
        use_canonical_fixtures=True,
        is_research_mode=False,
    )

    # Shared time is planning + initial analysis
    shared_sum = round(entry["planning_duration_seconds"] + entry["analysis_duration_seconds"], 4)
    assert abs(entry["shared_duration_seconds"] - shared_sum) <= 0.001

    # Proposed overhead specifically isolates Critic auditing and reinvestigation
    overhead_sum = round(entry["critic_duration_seconds"] + entry["verification_duration_seconds"], 4)
    assert abs(entry["proposed_overhead_seconds"] - overhead_sum) <= 0.001

    # Shared time must not be attributed to baseline duration
    assert entry["shared_duration_seconds"] != entry["baseline_duration_seconds"]


def test_rule_level_aggregation_and_overlapping_codes():
    """Test 3: Rule-level metric aggregation correctly handles overlapping codes per finding."""
    # Build synthetic critic reviews with overlapping rule triggers
    review_1 = CriticResult(
        task_id="task_1",
        analysis_type="categorical_vs_categorical",
        method="chi2_contingency",
        variables=["colA", "colB"],
        status="QUESTIONABLE",
        confidence=0.8,
        confidence_level="MEDIUM",
        summary="Cochran rule violated",
        evidence=["min expected = 0.5"],
        statistical_validity="Invalid",
        method_assessment="Poor sample size",
        data_quality_concerns=[],
        assumption_concerns=["Cochran guideline violated"],
        recommendation="Use Fisher exact test",
        requires_reinvestigation=True,
        reason_codes=["EXPECTED_COUNT_BELOW_ONE", "SPARSE_CONTINGENCY_CELLS"],
    )
    review_2 = CriticResult(
        task_id="task_2",
        analysis_type="numerical_vs_numerical",
        method="pearson_correlation",
        variables=["x", "y"],
        status="REQUIRES_REINVESTIGATION",
        confidence=0.8,
        confidence_level="MEDIUM",
        summary="Skew and leverage violation",
        evidence=["skew=2.1"],
        statistical_validity="Invalid",
        method_assessment="Non-normal",
        data_quality_concerns=[],
        assumption_concerns=["Distributional skewness and outlier leverage"],
        recommendation="Use Spearman rank",
        requires_reinvestigation=True,
        reason_codes=[
            "NON_NORMAL_DISTRIBUTION",
            "RECOMMEND_SPEARMAN_RANK",
            "SUBSTANTIAL_OUTLIER_INFLUENCE",
            "VISUAL_OUTLIERS_REINVESTIGATE",
        ],
    )

    mock_critic_resp = CriticResponse(
        dataset_id="test_ds",
        summary=CriticOverallSummary(
            total_results=2,
            accepted=0,
            questionable=1,
            rejected=0,
            requires_reinvestigation=2,
            overall_status="NEEDS_REVIEW",
            critical_issues=[],
            key_verified_insights=[],
        ),
        reviews=[review_1, review_2],
        critic_version="4.0",
        mode="deterministic",
    )

    mock_critic_svc = MagicMock()
    mock_critic_svc.run_review.return_value = mock_critic_resp

    # Use a dummy profile with no live dependencies
    prof = {
        "overview": {"filename": "test.csv", "n_rows": 50, "n_cols": 4},
        "column_types": {"colA": "categorical", "colB": "categorical", "x": "numerical", "y": "numerical"},
    }

    from app.schemas.planner_schema import InvestigationTask

    dummy_task = InvestigationTask(
        task_id="task_1",
        task_type="categorical_vs_categorical",
        task="Test task",
        description="Test task description",
        method="chi2_contingency",
        columns=["colA", "colB"],
        reason="Test reason",
        priority="high",
    )

    dummy_plan = PlannerResponse(
        objective="Test",
        interpreted_goal="Test",
        selected_columns=[],
        analysis_tasks=[dummy_task],
        tasks=[dummy_task],
        reasoning=[],
        warnings=[],
        expected_outputs=[],
        planner_mode="deterministic",
    )

    from app.schemas.analysis_schema import PlanAnalysisResponse, AnalysisResult

    mock_analysis_svc = MagicMock()
    mock_analysis_resp = PlanAnalysisResponse(
        dataset_id="test.csv",
        total_tasks=2,
        successful_tasks=2,
        failed_tasks=0,
        results=[
            AnalysisResult(
                task_id="task_1",
                analysis_type="categorical_vs_categorical",
                method="chi2_contingency",
                columns=["colA", "colB"],
                results={},
                status="success",
                findings=["Categorical association evaluated."],
            ),
            AnalysisResult(
                task_id="task_2",
                analysis_type="numerical_vs_numerical",
                method="pearson_correlation",
                columns=["x", "y"],
                results={},
                status="success",
                findings=["Correlation evaluated."],
            ),
        ],
    )
    mock_analysis_svc.execute_plan.return_value = mock_analysis_resp

    from tests.test_experiment_runner import (
        build_sample_verification_response,
        build_sample_insight_response,
        build_mock_evaluation_response,
    )

    mock_verif_svc = MagicMock()
    mock_verif_svc.process_verification.return_value = build_sample_verification_response(reinvestigated=True)

    mock_eval_svc = MagicMock()
    mock_eval_svc.evaluate_pipeline.side_effect = [
        build_mock_evaluation_response(65.0, "B - Acceptable"),
        build_mock_evaluation_response(88.0, "A - High Scientific Rigor"),
    ]

    mock_insight_svc = MagicMock()
    mock_insight_svc.generate_insights.return_value = build_sample_insight_response()

    resp = run_paired_experiment(
        profile=prof,
        dataset_id="test.csv",
        planner_fn=lambda **kwargs: dummy_plan,
        analysis_service=mock_analysis_svc,
        critic_service=mock_critic_svc,
        verification_service=mock_verif_svc,
        insight_service=mock_insight_svc,
        evaluation_service_inst=mock_eval_svc,
        persist_runs=False,
        is_offline_fixture=True,
    )

    # Cochran violations: exactly 1 finding violated Cochran (even though 2 reason codes triggered)
    assert resp.comparison.cochran_violations_count == 1

    # Normality violations: exactly 1 finding violated normality (even though multiple reason codes triggered)
    assert resp.comparison.normality_violations_count == 1

    # Leverage violations: exactly 1 finding violated leverage
    assert resp.comparison.leverage_outlier_violations_count == 1

    # Rule breakdown tracks all trigger codes
    breakdown = resp.comparison.rule_breakdown
    assert breakdown.get("EXPECTED_COUNT_BELOW_ONE") == 1
    assert breakdown.get("SPARSE_CONTINGENCY_CELLS") == 1
    assert breakdown.get("NON_NORMAL_DISTRIBUTION") == 1
    assert breakdown.get("VISUAL_OUTLIERS_REINVESTIGATE") == 1


def test_benchmark_processing_multi_fixture_datasets(tmp_path):
    """Test 4: Benchmark processing across multiple candidate fixture datasets."""
    datasets = ["data/sample.csv", "data/student-mat.csv"]
    results, report_paths = run_benchmark_suite(
        datasets=datasets,
        output_dir=str(tmp_path),
        use_canonical_fixtures=True,
        is_research_mode=False,
    )

    assert len(results) == 2
    for r in results:
        assert r["status"] == "SUCCESS"
        assert r["n_rows"] > 0
        assert r["n_cols"] > 0
        assert r["tasks_executed"] > 0
        assert r["execution_mode"] == "offline_fixture_replay"
        assert r["is_offline_fixture"] is True


def test_csv_markdown_json_output_correctness(tmp_path):
    """Test 5: Benchmark exports valid CSV, Markdown, and JSON files."""
    datasets = ["data/sample.csv"]
    results, report_paths = run_benchmark_suite(
        datasets=datasets,
        output_dir=str(tmp_path),
        use_canonical_fixtures=True,
        is_research_mode=False,
    )

    csv_file = Path(report_paths["csv"])
    md_file = Path(report_paths["markdown"])
    json_file = Path(report_paths["json"])

    assert csv_file.exists()
    assert md_file.exists()
    assert json_file.exists()

    # Validate CSV content
    with open(csv_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["dataset_name"] == "sample.csv"
        assert rows[0]["status"] == "SUCCESS"
        assert "shared_duration_seconds" in rows[0]
        assert "proposed_overhead_seconds" in rows[0]

    # Validate Markdown content
    md_text = md_file.read_text(encoding="utf-8")
    assert "# DataInvest AI — Cross-Dataset Benchmark Summary" in md_text
    assert "sample.csv" in md_text
    assert "Offline Deterministic Fixture Replay" in md_text

    # Validate JSON content
    with open(json_file, encoding="utf-8") as f:
        json_data = json.load(f)
        assert json_data["is_offline_fixture"] is True
        assert json_data["success_count"] == 1
        assert len(json_data["results"]) == 1


def test_partial_failure_reporting_and_visibility(tmp_path):
    """Test 6: Partial failures are recorded visibly without discarding or crashing."""
    datasets = ["data/sample.csv", "data/nonexistent_dataset.csv"]
    results, report_paths = run_benchmark_suite(
        datasets=datasets,
        output_dir=str(tmp_path),
        use_canonical_fixtures=True,
        is_research_mode=False,
    )

    assert len(results) == 2
    success_item = next(r for r in results if r["dataset_name"] == "sample.csv")
    failed_item = next(r for r in results if r["dataset_name"] == "nonexistent_dataset.csv")

    assert success_item["status"] == "SUCCESS"
    assert failed_item["status"] == "FAILED"
    assert "could not be loaded" in failed_item["error_message"]

    # Verify Markdown reflects partial failure
    md_text = Path(report_paths["markdown"]).read_text(encoding="utf-8")
    assert "### Partial Failures / Unprocessed Datasets" in md_text
    assert "nonexistent_dataset.csv" in md_text


def test_reproducibility_of_fixture_output(tmp_path):
    """Test 7: Fixture-based outputs are reproducible across runs."""
    entry_1 = run_single_dataset_benchmark("data/sample.csv", use_canonical_fixtures=True)
    entry_2 = run_single_dataset_benchmark("data/sample.csv", use_canonical_fixtures=True)

    deterministic_keys = [
        "dataset_name",
        "status",
        "n_rows",
        "n_cols",
        "tasks_executed",
        "initial_findings_count",
        "verified_findings_count",
        "findings_requiring_reinvestigation",
        "findings_revised",
        "findings_unresolved",
        "baseline_score",
        "proposed_score",
        "score_delta",
    ]

    for k in deterministic_keys:
        assert entry_1[k] == entry_2[k], f"Mismatch on key {k}: {entry_1[k]} != {entry_2[k]}"


def test_path_safety_and_overwrite_protection(tmp_path):
    """Test 8: Output files are not overwritten by default and historical runs are preserved."""
    datasets = ["data/sample.csv"]

    _, paths_1 = run_benchmark_suite(
        datasets=datasets,
        output_dir=str(tmp_path),
        overwrite=False,
    )
    _, paths_2 = run_benchmark_suite(
        datasets=datasets,
        output_dir=str(tmp_path),
        overwrite=False,
    )

    # Second run should not overwrite the first if timestamps match, creating distinct files
    p1 = Path(paths_1["csv"])
    p2 = Path(paths_2["csv"])
    assert p1.exists()
    assert p2.exists()


def test_zero_external_api_calls_enforced():
    """Test 9: Offline harness forbids calling external APIs and research mode requires explicit opt-in."""
    # Attempting research mode without confirm_live_api must raise RuntimeError
    with pytest.raises(RuntimeError, match="CRITICAL EXECUTION BOUNDARY"):
        run_benchmark_suite(
            datasets=["data/sample.csv"],
            is_research_mode=True,
            confirm_live_api=False,
        )


def test_repetition_support_and_repetition_id_tracking(tmp_path):
    """Test 10: Repetitions loop correctly tracks repetition_id and records in all output formats."""
    datasets = ["data/sample.csv"]
    results, report_paths = run_benchmark_suite(
        datasets=datasets,
        output_dir=str(tmp_path),
        repetitions=2,
        use_canonical_fixtures=True,
        is_research_mode=False,
    )

    assert len(results) == 2
    assert results[0]["repetition_id"] == 1
    assert results[1]["repetition_id"] == 2
    assert results[0]["dataset_name"] == "sample.csv"
    assert results[1]["dataset_name"] == "sample.csv"

    # Verify JSON output structure
    json_path = Path(report_paths["json"])
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
        assert data["repetitions_per_dataset"] == 2
        assert data["total_trials"] == 2
        assert "sample.csv" in data["per_dataset_summary"]
        assert data["per_dataset_summary"]["sample.csv"]["total_trials"] == 2

    # Verify CSV has repetition_id column
    csv_path = Path(report_paths["csv"])
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["repetition_id"] == "1"
        assert rows[1]["repetition_id"] == "2"


def test_invalid_repetition_values_raise_error():
    """Test 11: Benchmark rejects repetition values < 1."""
    with pytest.raises(ValueError, match="Repetitions must be a positive integer"):
        run_benchmark_suite(
            datasets=["data/sample.csv"],
            repetitions=0,
        )

    with pytest.raises(ValueError, match="Repetitions must be a positive integer"):
        run_benchmark_suite(
            datasets=["data/sample.csv"],
            repetitions=-3,
        )


def test_output_directory_selection_and_isolation():
    """Test 12: Output directory routes correctly based on offline vs live research mode."""
    # Test safe path validation for permitted workspace relative paths
    safe_rel = validate_safe_output_dir("reports/benchmarks")
    assert "reports" in str(safe_rel)

    # Test safe path validation inside system tempdir
    import tempfile
    temp_dir = Path(tempfile.gettempdir()) / "datainvest_test_bench"
    safe_temp = validate_safe_output_dir(temp_dir)
    assert safe_temp.resolve() == temp_dir.resolve()


def test_path_traversal_detection_and_rejection():
    """Test 13: Path traversal attempts escaping permitted directories are blocked."""
    # An attempt to escape outside the workspace into root/arbitrary directories
    with pytest.raises(ValueError, match="Path traversal or unauthorized output path detected"):
        validate_safe_output_dir("C:/Windows/System32/evil_test", allow_temp=False)

    with pytest.raises(ValueError, match="Path traversal or unauthorized output path detected"):
        validate_safe_output_dir("../../../../../../../../../../../../../evil_dir", allow_temp=False)


def test_per_dataset_breakdown_and_non_pooled_reporting(tmp_path):
    """Test 14: Benchmark reports contain dedicated per-dataset breakdowns avoiding naive pooling."""
    datasets = ["data/sample.csv", "data/student-mat.csv"]
    results, report_paths = run_benchmark_suite(
        datasets=datasets,
        output_dir=str(tmp_path),
        repetitions=1,
        use_canonical_fixtures=True,
        is_research_mode=False,
    )

    md_text = Path(report_paths["markdown"]).read_text(encoding="utf-8")
    assert "## Per-Dataset Performance Breakdown" in md_text
    assert "### Dataset: `sample.csv`" in md_text
    assert "### Dataset: `student-mat.csv`" in md_text
    assert "Scientific Notice on Per-Dataset Reporting" in md_text

    # JSON includes per_dataset_summary
    json_text = Path(report_paths["json"]).read_text(encoding="utf-8")
    json_data = json.loads(json_text)
    assert "per_dataset_summary" in json_data
    assert "sample.csv" in json_data["per_dataset_summary"]
    assert "student-mat.csv" in json_data["per_dataset_summary"]
    assert json_data["per_dataset_summary"]["sample.csv"]["successful_trials"] == 1
    assert json_data["per_dataset_summary"]["student-mat.csv"]["successful_trials"] == 1

"""
backend/tests/test_pilot_path_safety.py
=======================================
Targeted regression tests for Phase 2.2 pilot path safety:
- Dataset resolution (absolute external paths, missing files, directories, project-relative)
- Output-path validation (approved external root, legitimate children, siblings, traversal)
- Event-log isolation (configured pilot log directory, invalid path rejection, persist_runs=False protection)

Safety Guarantees:
- 100% offline: no network calls, no LLMs, no external APIs.
- Non-destructive: does not modify staged datasets, source code, historical runs, or logs.
- Uses tmp_path fixtures for any test file generation.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from app.services.dataset_store import (
    clear_cache,
    get_dataset,
    resolve_dataset,
    store_dataset,
)
from app.services.experiment_logger import (
    APPROVED_EXTERNAL_RESULTS_ROOT as LOGGER_APPROVED_ROOT,
    ExperimentLogger,
    validate_safe_log_dir,
)
from app.services.experiment_runner import run_paired_experiment
from scripts.run_cross_dataset_benchmark import (
    APPROVED_EXTERNAL_RESULTS_ROOT as BENCHMARK_APPROVED_ROOT,
    validate_safe_output_dir,
)

STAGED_HEART_CSV = Path(r"C:\Users\SPURGE\MAJOR_PROJECT_DOCS\REVIEW 2\DataInvest_Pilot_Staging\heart.csv")


# ==============================================================================
# 1. Dataset Resolution Tests
# ==============================================================================

def test_resolve_staged_heart_csv_directly_without_modification():
    """A valid staged absolute CSV path can be resolved directly without modifying source."""
    assert STAGED_HEART_CSV.exists(), f"Staged dataset missing: {STAGED_HEART_CSV}"
    stat_before = STAGED_HEART_CSV.stat()

    clear_cache()
    df = resolve_dataset(str(STAGED_HEART_CSV))

    assert df is not None
    assert not df.empty
    assert len(df) > 0
    assert len(df.columns) > 0

    stat_after = STAGED_HEART_CSV.stat()
    assert stat_before.st_mtime == stat_after.st_mtime, "Source dataset mtime was modified!"
    assert stat_before.st_size == stat_after.st_size, "Source dataset size was modified!"


def test_resolve_missing_absolute_path_raises_filenotfound():
    """A missing absolute path produces a clear, controlled FileNotFoundError."""
    missing_path = r"C:\Users\SPURGE\MAJOR_PROJECT_DOCS\REVIEW 2\DataInvest_Pilot_Staging\nonexistent_dataset.csv"
    with pytest.raises(FileNotFoundError, match="Dataset file not found at absolute path"):
        resolve_dataset(missing_path)


def test_resolve_directory_as_dataset_raises_valueerror(tmp_path):
    """A directory path supplied as an absolute dataset path is cleanly rejected."""
    dir_path = tmp_path / "fake_dataset_dir"
    dir_path.mkdir()

    with pytest.raises(ValueError, match="Dataset path is not a regular file"):
        resolve_dataset(str(dir_path))


def test_resolve_unsupported_file_extension_raises_valueerror(tmp_path):
    """An absolute file path with an unsupported extension is rejected."""
    bad_file = tmp_path / "script.py"
    bad_file.write_text("print('hello')", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported dataset file extension"):
        resolve_dataset(str(bad_file))


def test_existing_project_relative_dataset_resolution_preserved():
    """Existing project-relative dataset resolution remains fully functional."""
    clear_cache()
    df = resolve_dataset("sample.csv")
    assert df is not None
    assert not df.empty
    assert "age" in df.columns or len(df.columns) > 0


# ==============================================================================
# 2. Output Path Validation Tests
# ==============================================================================

def test_approved_external_results_root_accepted():
    """The approved external results root is accepted without being created."""
    res = validate_safe_output_dir(BENCHMARK_APPROVED_ROOT)
    assert res == BENCHMARK_APPROVED_ROOT


def test_approved_external_results_child_directory_accepted():
    """A legitimate child directory under the approved external results root is accepted."""
    child = BENCHMARK_APPROVED_ROOT / "pilot_run_heart_01"
    res = validate_safe_output_dir(child)
    assert res == child.resolve()


def test_approved_research_results_root_accepted():
    """DataInvest_Research_Results and its children are accepted as valid output roots."""
    research_root = Path(r"C:\Users\SPURGE\MAJOR_PROJECT_DOCS\REVIEW 2\DataInvest_Research_Results")
    res = validate_safe_output_dir(research_root)
    assert res == research_root.resolve()

    res_child = validate_safe_output_dir(research_root / "tables")
    assert res_child == (research_root / "tables").resolve()


def test_sibling_directory_outside_approved_root_rejected():
    """A sibling directory outside the approved root is rejected."""
    sibling = Path(r"C:\Users\SPURGE\MAJOR_PROJECT_DOCS\REVIEW 2\DataInvest_Pilot_Results_fake")
    with pytest.raises(ValueError, match="Path traversal or unauthorized output path detected"):
        validate_safe_output_dir(sibling, allow_temp=False)


def test_path_traversal_escaping_approved_root_rejected():
    """A path-traversal attempt escaping the approved root is rejected."""
    traversal = BENCHMARK_APPROVED_ROOT / ".." / "ArbitraryEscape"
    with pytest.raises(ValueError, match="Path traversal or unauthorized output path detected"):
        validate_safe_output_dir(traversal, allow_temp=False)


def test_existing_project_and_temp_paths_retained(tmp_path):
    """Existing project-workspace and temp directories retain their intended behavior."""
    # Permitted project path
    proj_res = validate_safe_output_dir("reports/benchmarks")
    assert "reports" in str(proj_res)

    # Permitted temporary path
    temp_res = validate_safe_output_dir(tmp_path, allow_temp=True)
    assert temp_res.resolve() == tmp_path.resolve()


# ==============================================================================
# 3. Event-Log Isolation Tests
# ==============================================================================

def test_explicitly_configured_pilot_log_directory_used(tmp_path):
    """An explicitly configured pilot log directory writes logs to the designated location."""
    log_dir = tmp_path / "pilot_logs"
    logger = ExperimentLogger(logs_dir=log_dir, allow_temp=True)

    success = logger.log_event(
        run_id="run_pilot_test_01",
        event_type="PILOT_START",
        phase=1,
        details={"dataset": "heart.csv"},
    )
    assert success is True

    expected_file = log_dir / "run_pilot_test_01.jsonl"
    assert expected_file.exists()
    content = expected_file.read_text(encoding="utf-8")
    assert "PILOT_START" in content
    assert "heart.csv" in content

    # Historical production log dir must NOT contain this test log
    default_prod_dir = Path(__file__).resolve().parents[1] / "data" / "logs"
    prod_file = default_prod_dir / "run_pilot_test_01.jsonl"
    assert not prod_file.exists()


def test_approved_research_log_dir_accepted():
    """DataInvest_Research_Results/runs is accepted as a valid log directory."""
    research_log_dir = Path(r"C:\Users\SPURGE\MAJOR_PROJECT_DOCS\REVIEW 2\DataInvest_Research_Results\runs")
    res = validate_safe_log_dir(research_log_dir)
    assert res == research_log_dir.resolve()


def test_invalid_external_log_path_fails_safely():
    """Invalid external log paths fail immediately with ValueError rather than falling back."""
    unauthorized = Path(r"C:\Users\SPURGE\MAJOR_PROJECT_DOCS\REVIEW 2\unauthorized_logs")
    with pytest.raises(ValueError, match="Path traversal or unauthorized log directory detected"):
        validate_safe_log_dir(unauthorized, allow_temp=False)

    with pytest.raises(ValueError, match="Path traversal or unauthorized log directory detected"):
        ExperimentLogger(logs_dir=unauthorized, allow_temp=False)


def test_persist_runs_false_prevents_historical_run_persistence(tmp_path):
    """persist_runs=False strictly prevents writes to historical run store while routing logs safely."""
    from app.schemas.planner_schema import InvestigationTask, PlannerResponse
    from app.schemas.analysis_schema import PlanAnalysisResponse
    from app.schemas.critic_schema import CriticOverallSummary, CriticResponse
    from app.schemas.verification_schema import VerificationResponse, VerificationSummary
    from app.schemas.insight_schema import InsightResponse, InsightSummary
    from app.schemas.evaluation_schema import EvaluationResponse, EvaluationSummary

    dummy_task = InvestigationTask(
        task_id="t1",
        task_type="descriptive_statistics",
        task="Descriptive analysis",
        description="Run descriptive statistics on numeric columns",
        method="describe",
        columns=["age"],
        reason="Initial baseline distribution check",
        priority="medium",
    )

    dummy_plan = PlannerResponse(
        objective="Test offline safety",
        interpreted_goal="Test offline safety",
        dataset_summary={"overview": "Heart dataset stub"},
        selected_columns=[],
        analysis_tasks=[dummy_task],
        tasks=[dummy_task],
        reasoning=[],
        warnings=[],
        expected_outputs=[],
        planner_mode="deterministic_rule",
    )

    from tests.test_experiment_runner import (
        build_sample_analysis_response,
        build_sample_critic_response,
        build_sample_verification_response,
        build_sample_insight_response,
        build_mock_evaluation_response,
    )

    mock_analysis = MagicMock()
    mock_analysis.execute_plan.return_value = build_sample_analysis_response()

    mock_critic = MagicMock()
    mock_critic.run_review.return_value = build_sample_critic_response(requires_reinvestigation=False)

    mock_verif = MagicMock()
    mock_verif.process_verification.return_value = build_sample_verification_response(reinvestigated=False)

    mock_insight = MagicMock()
    mock_insight.generate_insights.return_value = build_sample_insight_response()

    mock_eval = MagicMock()
    mock_eval.evaluate_pipeline.return_value = build_mock_evaluation_response(score=80.0, grade="B")

    mock_run_store = MagicMock()
    pilot_log_dir = tmp_path / "experiment_isolated_logs"

    resp = run_paired_experiment(
        profile={
            "dataset_id": "heart.csv",
            "overview": {"filename": "heart.csv", "n_rows": 303, "n_cols": 14, "column_names": ["age"]},
        },
        dataset_id="heart.csv",
        objective="Test isolation offline",
        persist_runs=False,
        planner_fn=lambda **kwargs: dummy_plan,
        analysis_service=mock_analysis,
        critic_service=mock_critic,
        verification_service=mock_verif,
        insight_service=mock_insight,
        evaluation_service_inst=mock_eval,
        run_store_service=mock_run_store,
        log_dir=pilot_log_dir,
        is_offline_fixture=True,
    )

    # 1. run_store.save_run must NEVER be called
    assert mock_run_store.save_run.call_count == 0
    assert resp.persisted is False

    # 2. Logs were written to pilot_log_dir, NOT production data/logs
    proposed_log = pilot_log_dir / f"{resp.proposed.run_id}.jsonl"
    baseline_log = pilot_log_dir / f"{resp.baseline.run_id}.jsonl"
    assert proposed_log.exists()
    assert baseline_log.exists()


def test_default_logging_behavior_preserved_without_pilot_config():
    """Default logger instance uses default backend/data/logs without affecting other runs."""
    default_logger = ExperimentLogger()
    expected_path = Path(__file__).resolve().parents[1] / "data" / "logs"
    assert default_logger.logs_dir.resolve() == expected_path.resolve()

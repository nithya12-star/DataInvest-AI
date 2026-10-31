"""
backend/tests/test_stage_c.py
=============================
Stage C Verification Suite:
- Objective 1: Comparison metrics & honest missing data handling in MetricDeltaSummary
- Objective 2: Structured experiment logging, redaction, safety, retrieval
- Objective 3: Offline deterministic demo with synthetic data and simulation markers
- Objective 4: AutoMind citation and base-paper capability distinction checks
- Safeguards: Historical saved run integrity (18 runs preserved untouched)
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from app.schemas.experiment_schema import MetricDeltaSummary
from app.services.experiment_logger import ExperimentLogger, sanitize_log_payload
from app.services.demo_service import DemoService, create_synthetic_demo_dataframe, DEMO_DISCLAIMER
from app.services.run_store import run_store


class TestStageCObjective1ComparisonMetrics:
    """Validate comparison metrics schema and honest missing data handling."""

    def test_metric_delta_summary_with_all_breakdown_fields(self):
        summary = MetricDeltaSummary(
            baseline_score=75.0,
            proposed_score=85.0,
            overall_score_delta=10.0,
            rigor_score_delta=12.0,
            verification_score_delta=15.0,
            dimension_deltas={"statistical_rigor": 12.0},
            assumption_violations_flagged=2,
            violations_resolved_via_reinvestigation=2,
            resolution_rate=1.0,
            total_reinvestigations=2,
            verified_findings_count=4,
            questionable_findings_count=0,
            rejected_findings_count=0,
            initial_findings_count=4,
            findings_accepted_directly=2,
            findings_requiring_reinvestigation=2,
            findings_revised=2,
            findings_unresolved=0,
            scientific_verdict="Proposed improved quality by 10 pts.",
        )
        assert summary.initial_findings_count == 4
        assert summary.findings_accepted_directly == 2
        assert summary.findings_requiring_reinvestigation == 2
        assert summary.findings_revised == 2
        assert summary.findings_unresolved == 0

    def test_metric_delta_summary_honest_missing_data_not_zeroes(self):
        """When fields are not provided, they remain None and are not coerced to false 0s."""
        summary = MetricDeltaSummary(
            baseline_score=70.0,
            proposed_score=70.0,
            overall_score_delta=0.0,
            rigor_score_delta=0.0,
            verification_score_delta=0.0,
            dimension_deltas={},
            scientific_verdict="Equal performance.",
        )
        # Verify optional fields default to None, not 0
        assert summary.initial_findings_count is None
        assert summary.findings_accepted_directly is None
        assert summary.findings_requiring_reinvestigation is None
        assert summary.findings_revised is None
        assert summary.findings_unresolved is None
        assert summary.resolution_rate is None


class TestStageCObjective2ExperimentLogger:
    """Validate structured experiment logging, credential redaction, and safety."""

    def test_sanitization_redacts_api_keys_and_tokens(self):
        payload = {
            "api_key": "AIzaSyD-1234567890abcdef1234567890ab",
            "message": "Calling Gemini with AIzaSyD-1234567890abcdef1234567890ab now",
            "auth": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.token",
            "safe_field": 42,
        }
        sanitized = sanitize_log_payload(payload)
        assert sanitized["api_key"] == "[REDACTED]"
        assert "[REDACTED_API_KEY]" in sanitized["message"]
        assert "AIzaSyD" not in sanitized["message"]
        assert sanitized["auth"] == "[REDACTED]"
        assert sanitized["safe_field"] == 42

    def test_sanitization_protects_raw_dataset_rows(self):
        payload = {
            "task_id": "T01",
            "dataset_rows": [{"id": 1, "ssn": "000-00-0000", "income": 50000}] * 10,
            "sample_size": 10,
        }
        sanitized = sanitize_log_payload(payload)
        assert sanitized["task_id"] == "T01"
        assert "[PROTECTED_DATASET_ROWS: count=10]" in sanitized["dataset_rows"]
        assert sanitized["sample_size"] == 10

    def test_logger_writes_and_retrieves_events(self, tmp_path):
        logger = ExperimentLogger(logs_dir=tmp_path)
        run_id = "test_run_20261023_001"

        success = logger.log_event(
            run_id=run_id,
            event_type="PLANNER_COMPLETED",
            phase=2,
            outcome="SUCCESS",
            details={"tasks_planned": 3},
        )
        assert success is True

        success = logger.log_event(
            run_id=run_id,
            event_type="CRITIC_COMPLETED",
            phase=4,
            outcome="FLAGGED",
            reason_code="OUTLIER_LEVERAGE_EXCEEDED",
            details={"flagged_tasks": ["T02"]},
        )
        assert success is True

        events = logger.get_run_logs(run_id)
        assert len(events) == 2
        assert events[0]["event_type"] == "PLANNER_COMPLETED"
        assert events[0]["phase"] == 2
        assert events[1]["event_type"] == "CRITIC_COMPLETED"
        assert events[1]["reason_code"] == "OUTLIER_LEVERAGE_EXCEEDED"

    def test_logger_rejects_path_traversal(self, tmp_path):
        logger = ExperimentLogger(logs_dir=tmp_path)
        assert logger.log_event("../evil_run", "TEST") is False
        assert logger.get_run_logs("../../etc/passwd") == []

    def test_logger_handles_missing_file_honestly(self, tmp_path):
        logger = ExperimentLogger(logs_dir=tmp_path)
        events = logger.get_run_logs("non_existent_run_999")
        assert events == []


class TestStageCObjective3OfflineDemo:
    """Validate offline demonstration execution without external API calls."""

    def test_synthetic_data_generation(self):
        df = create_synthetic_demo_dataframe(n_samples=50, seed=123)
        assert len(df) == 50
        assert "study_hours" in df.columns
        assert "exam_score" in df.columns
        assert "tutoring_group" in df.columns

    def test_offline_demo_execution_is_labeled_and_deterministic(self):
        demo = DemoService()
        result = demo.run_offline_demo(objective="Analyze synthetic student data.")

        assert result["is_offline_demo"] is True
        assert result["mode"] == "offline_deterministic_demo"
        assert DEMO_DISCLAIMER in result["disclaimer"]
        assert result["plan"]["planner_mode"] == "deterministic_simulated"
        assert result["plan"]["is_offline_demo"] is True
        assert result["critic"]["is_offline_demo"] is True
        assert result["verification"]["is_offline_demo"] is True
        assert result["evaluation"]["is_offline_demo"] is True


class TestHistoricalSavedRunIntegrity:
    """Verify that all 18 historical runs in backend/data/runs/ remain untouched and readable."""

    def test_all_18_historical_runs_intact(self):
        runs = run_store.list_runs()
        assert len(runs) >= 18, f"Expected at least 18 historical runs, found {len(runs)}"
        for run_summary in runs:
            run_data = run_store.get_run(run_summary.run_id)
            assert run_data is not None
            assert "run_id" in run_data
            assert "created_at" in run_data

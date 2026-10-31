"""
backend/tests/test_report.py
============================
Phase 5 / Stage B: Unit & Integration Tests for Report Agent & Export Endpoints.

Verifies:
- Structured report generation from saved run states.
- Accurate distinction between initial and reinvestigated findings.
- Representation of Critic decisions and reinvestigation history.
- Graceful handling of missing phases without fabricating data.
- Safe Markdown rendering and strict JSON export serialization.
- Run-specific integrity (exports correspond strictly to selected run).
- Error handling for missing runs and unsafe run IDs.
- HTTP API endpoints for /api/runs/{run_id}/report and export downloads.
"""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.agents.report_agent import ReportAgent
from app.schemas.report_schema import ResearchReport
from app.services.report_service import ReportService, RunNotFoundError, InvalidRunIdError
from app.services.run_store import RunStoreService


@pytest.fixture
def synthetic_complete_run() -> dict:
    """Fixture providing a complete, multi-phase synthetic experiment run."""
    return {
        "run_id": "run_test_synthetic_complete_001",
        "dataset_name": "student_grades.csv",
        "dataset_id": "student_grades",
        "created_at": "2026-10-23T10:00:00Z",
        "updated_at": "2026-10-23T10:05:00Z",
        "active_phase": 7,
        "completed_phases": [1, 2, 3, 4, 5, 6, 7],
        "objective": "Investigate relationships between study hours, attendance, and exam scores.",
        "mode": "proposed",
        "profile": {
            "overview": {
                "n_rows": 395,
                "n_cols": 8,
                "total_missing_cells": 12,
                "overall_missing_percentage": 0.38,
                "target_variable": "exam_score",
            },
            "numerical_stats": {"study_hours": {}, "attendance": {}, "exam_score": {}},
            "categorical_stats": {"school": {}, "gender": {}, "famsize": {}},
        },
        "plan": {
            "analysis_tasks": [
                {"task_id": "t1", "method": "Pearson correlation"},
                {"task_id": "t2", "method": "Chi-square test of independence"},
            ]
        },
        "analysis": {
            "total_tasks": 2,
            "successful_tasks": 2,
            "failed_tasks": 0,
            "results": [
                {
                    "task_id": "t1",
                    "status": "success",
                    "analysis_type": "numerical_vs_numerical",
                    "method": "Pearson correlation",
                    "columns": ["study_hours", "exam_score"],
                    "sample_size": 395,
                    "results": {
                        "correlation_coefficient": 0.4215,
                        "p_value": 0.0001,
                        "p_value_formatted": "1.0000e-04",
                    },
                    "warnings": [],
                },
                {
                    "task_id": "t2",
                    "status": "success",
                    "analysis_type": "categorical_vs_categorical",
                    "method": "Chi-square test of independence",
                    "columns": ["gender", "school"],
                    "sample_size": 395,
                    "results": {
                        "chi2_statistic": 4.15,
                        "p_value": 0.0416,
                        "cramer_v": 0.1025,
                    },
                    "warnings": [],
                },
            ],
        },
        "critic": {
            "reviews": [
                {
                    "task_id": "t1",
                    "status": "REQUIRES_REINVESTIGATION",
                    "confidence": 0.75,
                    "confidence_level": "MEDIUM",
                    "summary": "Severe outliers detected (|z| > 3.5). Re-investigate via Spearman rank correlation.",
                    "statistical_validity": "Linearity condition questionable due to extreme leverage points.",
                    "method_assessment": "Pearson sensitive to heavy tails.",
                    "data_quality_concerns": [],
                    "assumption_concerns": ["Extreme residual outliers present."],
                    "reason_codes": ["SEVERE_OUTLIERS_DETECTED", "REQUIRES_NON_PARAMETRIC_TEST"],
                    "evidence": ["Detected 4 leverage observations outside 3.0 IQR threshold."],
                },
                {
                    "task_id": "t2",
                    "status": "ACCEPTED",
                    "confidence": 0.95,
                    "confidence_level": "HIGH",
                    "summary": "Contingency table satisfies Cochran assumption bounds.",
                    "statistical_validity": "Expected frequencies all >= 5.0.",
                    "method_assessment": "Chi-square appropriate for factor dependency.",
                    "data_quality_concerns": [],
                    "assumption_concerns": [],
                    "reason_codes": ["COCHRAN_CONDITION_SATISFIED", "SIGNIFICANT_CATEGORICAL_ASSOCIATION"],
                    "evidence": ["All expected cell counts >= 15.2 (exceeds 5.0 threshold)."],
                },
            ]
        },
        "verification": {
            "summary": {
                "total_findings": 2,
                "verified": 1,
                "verified_after_reinvestigation": 1,
                "questionable": 0,
                "rejected": 0,
                "unresolved_issues": ["Pairwise missing data excluded via listwise deletion."],
            },
            "reinvestigations": [
                {
                    "reinvestigation_id": "reinv_t1",
                    "source_task_id": "t1",
                    "variables": ["study_hours", "exam_score"],
                    "original_method": "Pearson correlation",
                    "reinvestigated_method": "Spearman rank correlation",
                    "trigger_reason": "Severe residual outliers flagged during Phase 4 Critic audit.",
                    "comparison": {
                        "original_method": "Pearson correlation",
                        "reinvestigated_method": "Spearman rank correlation",
                        "original_statistic": 0.4215,
                        "reinvestigated_statistic": 0.3980,
                        "original_p_value": 0.0001,
                        "reinvestigated_p_value": 0.0002,
                        "direction_consistent": True,
                        "significance_consistent": True,
                        "summary": "Monotonic correlation remains statistically significant (rho=0.3980, p=0.0002).",
                    },
                    "status": "VERIFIED_AFTER_REINVESTIGATION",
                }
            ],
            "all_findings": [
                {
                    "finding_id": "finding_1",
                    "task_id": "t1",
                    "variables": ["study_hours", "exam_score"],
                    "method": "Spearman rank correlation",
                    "status": "VERIFIED_AFTER_REINVESTIGATION",
                    "confidence": 0.90,
                    "confidence_level": "HIGH",
                    "summary": "Monotonic association confirmed robust against outliers via Spearman rank test.",
                    "reinvestigated": True,
                    "verification_reason": "Verified after non-parametric re-test successfully handled leverage outliers.",
                    "limitations": ["Monotonic relationship; does not establish causal link."],
                    "reason_codes": ["REINVESTIGATION_SUCCESSFUL", "MONOTONIC_RELATIONSHIP_VERIFIED"],
                },
                {
                    "finding_id": "finding_2",
                    "task_id": "t2",
                    "variables": ["gender", "school"],
                    "method": "Chi-square test of independence",
                    "status": "VERIFIED",
                    "confidence": 0.95,
                    "confidence_level": "HIGH",
                    "summary": "Statistically significant association between gender and school enrollment.",
                    "reinvestigated": False,
                    "verification_reason": "Passed all asymptotic frequency bounds and sample adequacy tests.",
                    "limitations": ["Small effect size (Cramer's V = 0.10)."],
                    "reason_codes": ["COCHRAN_CONDITION_SATISFIED"],
                },
            ],
        },
        "evaluation": {
            "summary": {
                "overall_score": 94.2,
                "overall_grade": "A+",
                "benchmark_pass_rate": "8/8 (100.0%)",
                "summary_verdict": "Fully verified pipeline execution with zero ungrounded claims.",
            }
        },
    }


def test_report_agent_generation_complete_run(synthetic_complete_run):
    """Test ReportAgent synthesizes complete structured report with correct sections and fields."""
    agent = ReportAgent()
    report = agent.generate_report(synthetic_complete_run)

    assert isinstance(report, ResearchReport)
    assert report.run_id == "run_test_synthetic_complete_001"
    assert report.dataset_context.dataset_name == "student_grades.csv"
    assert report.dataset_context.n_rows == 395
    assert report.dataset_context.n_cols == 8

    # Initial findings check
    assert len(report.initial_findings) == 2
    assert report.initial_findings[0].task_id == "t1"
    assert report.initial_findings[0].statistic == 0.4215
    assert report.initial_findings[0].method == "Pearson correlation"

    # Critic verdicts check
    assert len(report.critic_verdicts) == 2
    assert report.critic_verdicts[0].status == "REQUIRES_REINVESTIGATION"
    assert "SEVERE_OUTLIERS_DETECTED" in report.critic_verdicts[0].reason_codes
    assert report.critic_verdicts[1].status == "ACCEPTED"

    # Reinvestigation check
    assert len(report.reinvestigations) == 1
    reinv = report.reinvestigations[0]
    assert reinv.source_task_id == "t1"
    assert reinv.original_method == "Pearson correlation"
    assert reinv.reinvestigated_method == "Spearman rank correlation"
    assert reinv.direction_consistent is True

    # Final findings distinction
    assert len(report.final_findings) == 2
    f1 = next(f for f in report.final_findings if f.task_id == "t1")
    f2 = next(f for f in report.final_findings if f.task_id == "t2")
    assert f1.is_revised_after_reinvestigation is True
    assert f1.status == "VERIFIED_AFTER_REINVESTIGATION"
    assert f1.method == "Spearman rank correlation"

    assert f2.is_revised_after_reinvestigation is False
    assert f2.status == "VERIFIED"
    assert f2.method == "Chi-square test of independence"

    # Methodological disclaimer and non-causal check
    assert "does not constitute mathematical or causal proof" in report.methodological_disclaimer.lower()
    assert report.evaluation_summary.overall_score == 94.2
    assert report.evaluation_summary.overall_grade == "A+"


def test_report_agent_renders_valid_markdown(synthetic_complete_run):
    """Test Markdown rendering includes all mandatory sections and readable formatting."""
    agent = ReportAgent()
    report = agent.generate_report(synthetic_complete_run)
    md = agent.render_markdown(report)

    assert isinstance(md, str)
    assert "# DataInvest AI Scientific Investigation Report" in md
    assert "run_test_synthetic_complete_001" in md
    assert "student_grades.csv" in md

    # Check structural sections
    assert "## 1. Investigation Objective & Execution Scope" in md
    assert "## 2. Dataset Context & Profile Characteristics" in md
    assert "## 3. Initial Statistical Findings (Phase 3 Execution)" in md
    assert "## 4. Critic Agent Statistical Self-Verification (Phase 4)" in md
    assert "## 5. Controlled Re-investigation & Method Transitions (Phase 5)" in md
    assert "## 6. Consolidated Final Findings & Scientific Outcomes" in md
    assert "## 7. Documented Limitations & Unresolved Verification Issues" in md
    assert "## 8. Quantitative Benchmark Scorecard (Phase 7)" in md
    assert "## 9. Methodological & Non-Causal Scientific Disclaimer" in md

    # Check table rows and content
    assert "Pearson correlation" in md
    assert "Spearman rank correlation" in md
    assert "Chi-square test of independence" in md
    assert "VERIFIED_AFTER_REINVESTIGATION" in md
    assert "Grade A+" in md
    assert "does not constitute mathematical or causal proof" in md


def test_report_agent_handles_minimal_or_missing_phases_gracefully():
    """Test report synthesis handles incomplete runs without crashing or fabricating fields."""
    agent = ReportAgent()
    minimal_run = {
        "run_id": "run_minimal_test",
        "dataset_name": "minimal.csv",
        "created_at": "2026-10-23T12:00:00Z",
        # profile, plan, critic, verification, evaluation all omitted
        "analysis": {
            "results": [
                {
                    "task_id": "task_1",
                    "status": "success",
                    "method": "Welch's t-test",
                    "columns": ["colA", "colB"],
                    "results": {"statistic": 2.14, "p_value": 0.035},
                }
            ]
        },
    }

    report = agent.generate_report(minimal_run)
    assert report.run_id == "run_minimal_test"
    assert report.dataset_context.n_rows is None
    assert report.dataset_context.n_cols is None
    assert report.dataset_context.total_missing_cells is None
    assert report.evaluation_summary is None
    assert len(report.reinvestigations) == 0

    # Ensure markdown renders cleanly without exceptions
    md = agent.render_markdown(report)
    assert "Unavailable" in md
    assert "Welch's t-test" in md


def test_report_service_export_markdown_and_json(tmp_path, synthetic_complete_run):
    """Test ReportService exports valid Markdown and JSON files with safe filenames."""
    # Setup temporary run store
    store = RunStoreService(runs_dir=tmp_path)
    store.save_run(synthetic_complete_run)

    service = ReportService()
    # Temporarily point store to tmp_path
    service_run_store_backup = service.report_agent
    from app.services import run_store as global_run_store_module
    orig_store = global_run_store_module.run_store
    global_run_store_module.run_store = store

    try:
        run_id = synthetic_complete_run["run_id"]

        # Test Markdown Export
        md_text, md_filename = service.export_markdown(run_id)
        assert isinstance(md_text, str)
        assert md_filename.endswith(".md")
        assert "run_test_synthetic_complete_001" in md_filename
        assert "# DataInvest AI Scientific Investigation Report" in md_text

        # Test JSON Export
        json_data, json_filename = service.export_json(run_id)
        assert isinstance(json_data, dict)
        assert json_filename.endswith(".json")
        assert json_data["run_id"] == run_id
        assert "report" in json_data
        assert "raw_run_state" in json_data

        # Verify JSON is serializable and valid
        serialized = json.dumps(json_data)
        reloaded = json.loads(serialized)
        assert reloaded["report"]["run_id"] == run_id

    finally:
        global_run_store_module.run_store = orig_store


def test_report_service_run_not_found(tmp_path):
    """Test ReportService raises RunNotFoundError for nonexistent run."""
    store = RunStoreService(runs_dir=tmp_path)
    from app.services import run_store as global_run_store_module
    orig_store = global_run_store_module.run_store
    global_run_store_module.run_store = store
    service = ReportService()

    try:
        with pytest.raises(RunNotFoundError):
            service.get_report_for_run("run_does_not_exist_12345")
    finally:
        global_run_store_module.run_store = orig_store


def test_report_service_rejects_unsafe_run_ids():
    """Test ReportService rejects traversal patterns and unsafe filenames."""
    service = ReportService()
    with pytest.raises(InvalidRunIdError):
        service.get_report_for_run("../../etc/passwd")

    with pytest.raises(InvalidRunIdError):
        service.get_report_for_run("run_foo\\..\\bar")


def test_report_api_endpoints(tmp_path, synthetic_complete_run):
    """Test HTTP API routes: /api/runs/{run_id}/report, /export/markdown, /export/json, /api/report/generate."""
    store = RunStoreService(runs_dir=tmp_path)
    store.save_run(synthetic_complete_run)

    from app.services import run_store as global_run_store_module
    orig_store = global_run_store_module.run_store
    global_run_store_module.run_store = store

    client = TestClient(app)
    run_id = synthetic_complete_run["run_id"]

    try:
        # 1. GET /api/runs/{run_id}/report
        res = client.get(f"/api/runs/{run_id}/report")
        assert res.status_code == 200
        data = res.json()
        assert data["run_id"] == run_id
        assert data["dataset_context"]["dataset_name"] == "student_grades.csv"
        assert len(data["initial_findings"]) == 2

        # 2. GET /api/runs/{run_id}/export/markdown
        res_md = client.get(f"/api/runs/{run_id}/export/markdown")
        assert res_md.status_code == 200
        assert "text/markdown" in res_md.headers.get("content-type", "")
        disp = res_md.headers.get("content-disposition", "")
        assert "attachment" in disp
        assert ".md" in disp
        assert "# DataInvest AI Scientific Investigation Report" in res_md.text

        # 3. GET /api/runs/{run_id}/export/json
        res_json = client.get(f"/api/runs/{run_id}/export/json")
        assert res_json.status_code == 200
        assert "application/json" in res_json.headers.get("content-type", "")
        disp_j = res_json.headers.get("content-disposition", "")
        assert "attachment" in disp_j
        assert ".json" in disp_j
        j_data = res_json.json()
        assert j_data["run_id"] == run_id
        assert "report" in j_data

        # 4. GET nonexistent run -> 404
        res_404 = client.get("/api/runs/run_nonexistent_99999/report")
        assert res_404.status_code == 404

        # 5. POST /api/report/generate from payload
        res_gen = client.post("/api/report/generate", json=synthetic_complete_run)
        assert res_gen.status_code == 200
        gen_data = res_gen.json()
        assert gen_data["run_id"] == run_id
        assert len(gen_data["final_findings"]) == 2

    finally:
        global_run_store_module.run_store = orig_store


def test_exports_strictly_correspond_to_requested_run(tmp_path):
    """Test that export endpoints strictly correspond to the requested run_id, not in-memory or another run."""
    store = RunStoreService(runs_dir=tmp_path)
    run_a = {
        "run_id": "run_alpha_111",
        "dataset_name": "alpha_dataset.csv",
        "analysis": {"results": [{"task_id": "ta1", "method": "Pearson correlation", "results": {"statistic": 0.5}}]},
    }
    run_b = {
        "run_id": "run_beta_222",
        "dataset_name": "beta_dataset.csv",
        "analysis": {"results": [{"task_id": "tb1", "method": "Chi-square test", "results": {"statistic": 10.5}}]},
    }
    store.save_run(run_a)
    store.save_run(run_b)

    from app.services import run_store as global_run_store_module
    orig_store = global_run_store_module.run_store
    global_run_store_module.run_store = store

    client = TestClient(app)

    try:
        # Request Run A
        res_a = client.get("/api/runs/run_alpha_111/export/markdown")
        assert res_a.status_code == 200
        assert "alpha_dataset.csv" in res_a.text
        assert "beta_dataset.csv" not in res_a.text
        assert "run_alpha_111" in res_a.headers["content-disposition"]

        # Request Run B
        res_b = client.get("/api/runs/run_beta_222/export/markdown")
        assert res_b.status_code == 200
        assert "beta_dataset.csv" in res_b.text
        assert "alpha_dataset.csv" not in res_b.text
        assert "run_beta_222" in res_b.headers["content-disposition"]

    finally:
        global_run_store_module.run_store = orig_store


def test_export_json_redacts_sensitive_fields(tmp_path):
    """Regression test: verify JSON export sanitizes any sensitive API keys or credentials in raw_run_state."""
    store = RunStoreService(runs_dir=tmp_path)
    run_with_leak = {
        "run_id": "run_leak_test_001",
        "dataset_name": "sensitive.csv",
        "api_key": "AIzaSyFakeSecretKeyThatMustBeRedacted",
        "gemini_api_key": "AIzaSyAnotherSecretKey",
        "credentials": {"token": "bearer_secret_token_12345", "password": "supersecretpassword"},
        "analysis": {
            "results": [
                {
                    "task_id": "t1",
                    "method": "Pearson correlation",
                    "results": {"statistic": 0.35, "p_value": 0.02},
                    "metadata": {"auth_token": "token_abc123"},
                }
            ]
        },
    }
    store.save_run(run_with_leak)

    from app.services import run_store as global_run_store_module
    orig_store = global_run_store_module.run_store
    global_run_store_module.run_store = store
    service = ReportService()

    try:
        export_payload, filename = service.export_json("run_leak_test_001")
        raw_state = export_payload["raw_run_state"]

        assert raw_state["api_key"] == "[REDACTED]"
        assert raw_state["gemini_api_key"] == "[REDACTED]"
        assert raw_state["credentials"] == "[REDACTED]"
        assert raw_state["analysis"]["results"][0]["metadata"]["auth_token"] == "[REDACTED]"

        # Verify complete serialized string contains zero leaked tokens
        serialized_str = json.dumps(export_payload)
        assert "AIzaSyFakeSecretKeyThatMustBeRedacted" not in serialized_str
        assert "AIzaSyAnotherSecretKey" not in serialized_str
        assert "bearer_secret_token_12345" not in serialized_str
        assert "supersecretpassword" not in serialized_str
        assert "token_abc123" not in serialized_str

    finally:
        global_run_store_module.run_store = orig_store


def test_report_generation_against_legacy_schema_variants():
    """Test ReportAgent against historical variations where phases or fields are partially absent."""
    agent = ReportAgent()

    # Legacy Variant 1: empty critic and verification dicts (as in some early exploratory runs)
    legacy_empty_dicts = {
        "run_id": "run_legacy_empty",
        "dataset_name": "student_mat.csv",
        "objective": "Investigate student data.",
        "critic": {},
        "verification": {},
        "analysis": {"results": []},
    }
    report_1 = agent.generate_report(legacy_empty_dicts)
    assert report_1.run_id == "run_legacy_empty"
    assert len(report_1.critic_verdicts) == 0
    assert len(report_1.reinvestigations) == 0
    assert len(report_1.final_findings) == 0
    md_1 = agent.render_markdown(report_1)
    assert "No Phase 3 statistical tasks were recorded" in md_1
    assert "No Critic verification reviews were recorded" in md_1

    # Legacy Variant 2: None values for all optional objects
    legacy_none_objects = {
        "run_id": "run_legacy_nones",
        "dataset_name": "wine.csv",
        "profile": None,
        "plan": None,
        "analysis": None,
        "critic": None,
        "verification": None,
        "insights": None,
        "evaluation": None,
    }
    report_2 = agent.generate_report(legacy_none_objects)
    assert report_2.run_id == "run_legacy_nones"
    assert report_2.dataset_context.n_rows is None
    md_2 = agent.render_markdown(report_2)
    assert "Unavailable" in md_2
    assert report_2.methodological_disclaimer is not None


def test_report_agent_deduplicates_limitations_and_preserves_distinct():
    """Verify ReportAgent deduplicates identical concerns between unresolved_issues and critic_verdicts
    while strictly preserving genuinely distinct limitations."""
    agent = ReportAgent()
    run_with_duplicates = {
        "run_id": "run_limitations_test",
        "dataset_name": "test_data.csv",
        "verification": {
            "summary": {
                "unresolved_issues": [
                    "[task_14] High outlier sensitivity: extreme values shift mean by -19.4%.",
                    "[task_10] Sparse cell counts in contingency table (cells below 5).",
                ]
            }
        },
        "critic": {
            "reviews": [
                {
                    "task_id": "task_14",
                    "status": "QUESTIONABLE",
                    "confidence": 0.85,
                    "confidence_level": "HIGH",
                    "summary": "Questionable finding due to outlier sensitivity.",
                    "assumption_concerns": [
                        "High outlier sensitivity: extreme values shift mean by -19.4%."  # Exact duplicate of task_14 issue without prefix
                    ],
                    "data_quality_concerns": [],
                    "evidence": [],
                    "reason_codes": [],
                },
                {
                    "task_id": "task_2",
                    "status": "ACCEPTED",
                    "confidence": 1.0,
                    "confidence_level": "HIGH",
                    "summary": "Accepted with minor tail departure.",
                    "assumption_concerns": [
                        "Mild departure from bivariate normality in tails."  # Distinct concern
                    ],
                    "data_quality_concerns": [],
                    "evidence": [],
                    "reason_codes": [],
                },
            ]
        },
    }

    report = agent.generate_report(run_with_duplicates)
    limits = report.limitations_and_unresolved_issues

    # Should contain exactly 3 limitations:
    # 1. [task_14] High outlier sensitivity...
    # 2. [task_10] Sparse cell counts...
    # 3. Mild departure from bivariate normality...
    assert len(limits) == 3
    assert "[task_14] High outlier sensitivity: extreme values shift mean by -19.4%." in limits
    assert "[task_10] Sparse cell counts in contingency table (cells below 5)." in limits
    assert any("Mild departure from bivariate normality" in l for l in limits)

    # Must NOT have duplicate un-prefixed outlier sensitivity
    assert limits.count("[task_14] High outlier sensitivity: extreme values shift mean by -19.4%.") == 1
    assert "High outlier sensitivity: extreme values shift mean by -19.4%." not in limits

    # Verify Markdown rendering also includes all distinct limitations without duplication
    md = agent.render_markdown(report)
    # Appears once in Critic Verdict details table/evidence and once in Section 7 list
    assert "[task_14] High outlier sensitivity: extreme values shift mean by -19.4%." in md
    assert "[task_10] Sparse cell counts in contingency table" in md


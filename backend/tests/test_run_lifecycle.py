"""
backend/tests/test_run_lifecycle.py
===================================
End-to-end lifecycle verification for Requirement 6:
- Saving partial progress
- Completing a full run (Phases 1-7)
- Switching datasets & resetting workspace without losing history
- Reopening a saved run and restoring full results without rerunning computations
- Preserving separate unique runs when the same dataset filename is uploaded repeatedly
- Reloading from disk after server restart
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path

from app.main import app
from app.services.run_store import RunStoreService, run_store


@pytest.fixture
def api_client():
    return TestClient(app)


def test_full_experiment_lifecycle(api_client, tmp_path):
    """
    Test complete lifecycle:
    1. Partial progress saving
    2. Multi-phase execution through Phase 7
    3. Distinct runs for repeated uploads of the same dataset
    4. Workspace reset / dataset switching
    5. Reopening saved run without rerunning
    6. Surviving server restart (disk reload)
    """
    # ── Step 1: Save Partial Progress (Phase 1 Profiling) ──────────────
    phase1_payload = {
        "dataset_name": "winequality-red.csv",
        "dataset_id": "winequality-red.csv",
        "active_phase": 1,
        "completed_phases": [1],
        "profile": {
            "dataset_id": "winequality-red.csv",
            "overview": {"n_rows": 1599, "n_cols": 12},
            "column_types": {"numerical": ["fixed_acidity", "quality"], "categorical": []},
            "missing": {"total_missing": 0},
            "duplicates": {"n_duplicates": 240},
        },
    }

    res1 = api_client.post("/api/runs", json=phase1_payload)
    assert res1.status_code == 200
    run1_id = res1.json()["run_id"]
    assert run1_id.startswith("run_")
    assert "winequality_red" in run1_id

    # Verify partial progress is saved on disk
    loaded_p1 = api_client.get(f"/api/runs/{run1_id}").json()
    assert loaded_p1["completed_phases"] == [1]
    assert loaded_p1["active_phase"] == 1
    assert loaded_p1["profile"]["overview"]["n_rows"] == 1599

    # ── Step 2: Auto-save Progress Through All 7 Phases ───────────────
    full_payload = dict(loaded_p1)
    full_payload["plan"] = {
        "dataset_id": "winequality-red.csv",
        "objective": "Analyze acidity effects on wine quality",
        "analysis_tasks": [{"task_id": "t1", "method": "pearson_correlation", "variables": ["fixed_acidity", "quality"]}],
    }
    full_payload["analysis"] = {
        "results": [{"task_id": "t1", "status": "SUCCESS", "results": {"correlation_coefficient": 0.12, "statistic": 0.12}}]
    }
    full_payload["critic"] = {
        "reviews": [{"task_id": "t1", "status": "ACCEPTED", "confidence": 0.95, "evidence": ["n=1599"]}]
    }
    full_payload["verification"] = {
        "verified_findings": [{"finding_id": "vf_1", "task_id": "t1", "status": "VERIFIED"}],
        "summary": {"total_findings": 1, "verified": 1},
    }
    full_payload["insights"] = {
        "key_insights": [{"insight_id": "ins_1", "insight": "Fixed acidity exhibits weak positive correlation with quality."}],
        "summary": {"total_insights": 1},
    }
    full_payload["evaluation"] = {
        "summary": {
            "overall_score": 91.4,
            "overall_grade": "A - High Scientific Rigor",
            "total_insights_generated": 1,
            "zero_hallucination_verified": True,
        }
    }
    full_payload["active_phase"] = 7
    full_payload["completed_phases"] = [1, 2, 3, 4, 5, 6, 7]

    update_res = api_client.post("/api/runs", json=full_payload)
    assert update_res.status_code == 200
    assert update_res.json()["run_id"] == run1_id  # In-place update for same run

    # Verify summary fields extracted correctly
    run1_summary = update_res.json()["summary"]
    assert run1_summary["completed_phases"] == [1, 2, 3, 4, 5, 6, 7]
    assert run1_summary["overall_score"] == 91.4
    assert run1_summary["overall_grade"] == "A - High Scientific Rigor"

    # ── Step 3: Repeated Upload of the SAME dataset creates a NEW distinct run ─
    # Requirement 2: Do not overwrite an earlier run merely because the same dataset filename is uploaded again
    repeat_upload_payload = {
        "dataset_name": "winequality-red.csv",
        "dataset_id": "winequality-red.csv",
        "active_phase": 1,
        "completed_phases": [1],
        "profile": phase1_payload["profile"],
    }
    res_repeat = api_client.post("/api/runs", json=repeat_upload_payload)
    assert res_repeat.status_code == 200
    run2_id = res_repeat.json()["run_id"]

    assert run2_id != run1_id  # Safe unique ID, no overwrite!

    # ── Step 4: Upload a Different Dataset (Dataset Switching) ─────────
    dataset2_payload = {
        "dataset_name": "bank-full.csv",
        "dataset_id": "bank-full.csv",
        "active_phase": 1,
        "completed_phases": [1],
        "profile": {
            "dataset_id": "bank-full.csv",
            "overview": {"n_rows": 45211, "n_cols": 17},
        },
    }
    res_bank = api_client.post("/api/runs", json=dataset2_payload)
    assert res_bank.status_code == 200
    run3_id = res_bank.json()["run_id"]
    assert "bank_full" in run3_id

    # ── Step 5: Check History Listing Contains All 3 Runs ──────────────
    list_res = api_client.get("/api/runs")
    assert list_res.status_code == 200
    runs = list_res.json()["runs"]
    run_ids = [r["run_id"] for r in runs]
    assert run1_id in run_ids
    assert run2_id in run_ids
    assert run3_id in run_ids

    # ── Step 6: Reopen Saved Run 1 Without Rerunning (Requirement 5) ──
    reopened_res = api_client.get(f"/api/runs/{run1_id}")
    assert reopened_res.status_code == 200
    reopened = reopened_res.json()
    assert reopened["run_id"] == run1_id
    assert reopened["profile"]["overview"]["n_rows"] == 1599
    assert reopened["plan"]["objective"] == "Analyze acidity effects on wine quality"
    assert reopened["analysis"]["results"][0]["task_id"] == "t1"
    assert reopened["critic"]["reviews"][0]["status"] == "ACCEPTED"
    assert reopened["verification"]["verified_findings"][0]["finding_id"] == "vf_1"
    assert reopened["insights"]["key_insights"][0]["insight_id"] == "ins_1"
    assert reopened["evaluation"]["summary"]["overall_score"] == 91.4
    assert reopened["completed_phases"] == [1, 2, 3, 4, 5, 6, 7]

    # ── Step 7: Simulate Server Restart & Reload from Disk ──────────────
    fresh_service = RunStoreService(runs_dir=run_store.runs_dir)
    fresh_runs = fresh_service.list_runs()
    fresh_ids = [r.run_id for r in fresh_runs]
    assert run1_id in fresh_ids
    assert run2_id in fresh_ids
    assert run3_id in fresh_ids

    # Clean up test runs from backend/data/runs
    for r_id in [run1_id, run2_id, run3_id]:
        fresh_service.delete_run(r_id)

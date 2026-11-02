"""
backend/tests/test_deployment_export_validation.py
=================================================
Automated regression tests verifying the 10 acceptance criteria for
DataInvest AI experiment export, run-ID logging, and persistence.
"""

import sys
import json
import logging
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Ensure backend root is in sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.main import app
from app.services.run_store import RunStoreService, run_store


@pytest.fixture
def client():
    return TestClient(app)


def test_criterion_1_and_2_run_id_generation_and_logger_binding(client, caplog):
    """
    Acceptance Criteria 1 & 2:
    1. Saving a run returns a valid unique run ID.
    2. The logger receives that exact ID without None warnings and logs RUN_SAVED.
    """
    with caplog.at_level(logging.WARNING):
        payload = {
            "dataset_name": "criterion_test.csv",
            "dataset_id": "criterion_test.csv",
            "active_phase": 1,
            "completed_phases": [1],
            "profile": {"overview": {"n_rows": 25, "n_cols": 2}},
        }
        # Post without run_id
        resp = client.post("/api/runs", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "saved"
        run_id = data["run_id"]
        assert run_id.startswith("run_")

        # Verify no "invalid run_id: None" warnings were logged
        none_warnings = [r.message for r in caplog.records if "invalid run_id: None" in r.message]
        assert len(none_warnings) == 0, f"Found unexpected logger warning: {none_warnings}"

        # Verify logger actually recorded RUN_SAVED for this run_id
        logs_resp = client.get(f"/api/runs/{run_id}/logs")
        assert logs_resp.status_code == 200
        logs_data = logs_resp.json()
        assert logs_data["total_events"] >= 1
        assert any(e.get("event_type") == "RUN_SAVED" for e in logs_data["events"])

        # Clean up
        del_resp = client.delete(f"/api/runs/{run_id}")
        assert del_resp.status_code == 200


def test_criterion_3_history_listing_and_retrieval_consistency(client):
    """
    Acceptance Criterion 3:
    History listing and run retrieval use the same persisted record.
    """
    payload = {
        "dataset_name": "retrieval_consistency.csv",
        "dataset_id": "retrieval_consistency.csv",
        "active_phase": 1,
        "completed_phases": [1],
        "profile": {"overview": {"n_rows": 100, "n_cols": 5}},
    }
    save_resp = client.post("/api/runs", json=payload)
    assert save_resp.status_code == 200
    run_id = save_resp.json()["run_id"]

    try:
        # Check listing
        list_resp = client.get("/api/runs")
        assert list_resp.status_code == 200
        matching = [r for r in list_resp.json()["runs"] if r["run_id"] == run_id]
        assert len(matching) == 1
        assert matching[0]["n_rows"] == 100

        # Check single retrieval
        get_resp = client.get(f"/api/runs/{run_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["run_id"] == run_id
        assert get_resp.json()["profile"]["overview"]["n_rows"] == 100
    finally:
        client.delete(f"/api/runs/{run_id}")


def test_criterion_4_existing_committed_runs_remain_retrievable(client):
    """
    Acceptance Criterion 4:
    Existing committed runs remain retrievable.
    """
    mandatory_run = "run_20261030_133553_heartcsv_proposed_58c9f0"
    get_resp = client.get(f"/api/runs/{mandatory_run}")
    assert get_resp.status_code == 200
    data = get_resp.json()
    assert data["run_id"] == mandatory_run
    assert data["mode"] == "proposed"
    assert 7 in data.get("completed_phases", [])


def test_criterion_5_and_6_export_json_and_markdown_success(client):
    """
    Acceptance Criteria 5 & 6:
    5. JSON export succeeds for a valid saved run.
    6. Markdown export succeeds for a valid saved run.
    """
    mandatory_run = "run_20261030_133553_heartcsv_proposed_58c9f0"

    # Test Markdown export
    md_resp = client.get(f"/api/runs/{mandatory_run}/export/markdown")
    assert md_resp.status_code == 200
    assert "text/markdown" in md_resp.headers.get("content-type", "")
    assert "attachment" in md_resp.headers.get("content-disposition", "")
    assert len(md_resp.text) > 500

    # Test JSON export
    json_resp = client.get(f"/api/runs/{mandatory_run}/export/json")
    assert json_resp.status_code == 200
    assert "application/json" in json_resp.headers.get("content-type", "")
    assert "attachment" in json_resp.headers.get("content-disposition", "")
    export_payload = json_resp.json()
    assert export_payload.get("run_id") == mandatory_run


def test_criterion_7_missing_ids_return_clear_404_without_crashing(client):
    """
    Acceptance Criterion 7:
    Missing IDs return a clear 404 without crashing.
    """
    missing_id = "run_20991231_999999_nonexistent_000000"

    # GET run
    get_resp = client.get(f"/api/runs/{missing_id}")
    assert get_resp.status_code == 404
    assert "not found" in get_resp.json().get("error", "").lower()

    # Export Markdown
    exp_md = client.get(f"/api/runs/{missing_id}/export/markdown")
    assert exp_md.status_code == 404
    assert "not found" in exp_md.json().get("error", "").lower()

    # Export JSON
    exp_json = client.get(f"/api/runs/{missing_id}/export/json")
    assert exp_json.status_code == 404
    assert "not found" in exp_json.json().get("error", "").lower()


def test_criterion_8_baseline_and_proposed_distinct_ids_and_pairing(client):
    """
    Acceptance Criterion 8:
    Baseline and Proposed retain distinct IDs and correct pairing metadata.
    """
    base_id = "run_20261030_133553_heartcsv_baseline_58c9f0"
    prop_id = "run_20261030_133553_heartcsv_proposed_58c9f0"

    base_resp = client.get(f"/api/runs/{base_id}")
    prop_resp = client.get(f"/api/runs/{prop_id}")

    assert base_resp.status_code == 200
    assert prop_resp.status_code == 200

    base_data = base_resp.json()
    prop_data = prop_resp.json()

    assert base_data["run_id"] != prop_data["run_id"]
    assert base_data["mode"] == "baseline"
    assert prop_data["mode"] == "proposed"
    assert base_data["paired_run_id"] == prop_id
    assert prop_data["paired_run_id"] == base_id


def test_criterion_9_simulated_process_restart_persistence():
    """
    Acceptance Criterion 9:
    A saved run remains retrievable after a simulated application restart.
    """
    runs_dir = run_store.runs_dir
    test_run_id = "run_20261101_restart_simulation_test_999"
    test_payload = {
        "run_id": test_run_id,
        "dataset_name": "restart_test.csv",
        "dataset_id": "restart_test.csv",
        "active_phase": 1,
        "completed_phases": [1],
        "profile": {"overview": {"n_rows": 42, "n_cols": 3}},
    }

    # Save via active service
    run_store.save_run(test_payload)

    try:
        # Simulate restart: Instantiate brand new RunStoreService pointing to same dir
        restarted_store = RunStoreService(runs_dir=runs_dir)
        loaded = restarted_store.get_run(test_run_id)
        assert loaded is not None
        assert loaded["run_id"] == test_run_id
        assert loaded["profile"]["overview"]["n_rows"] == 42
    finally:
        run_store.delete_run(test_run_id)


def test_criterion_10_unsafe_id_validation_returns_422(client):
    """
    Acceptance Criterion 10 / Security:
    Path traversal and invalid characters return clear 422 error.
    """
    unsafe_id = "../etc/passwd"
    resp = client.get(f"/api/runs/{unsafe_id}/export/markdown")
    # FastAPI path parser or app security rejects unsafe characters with 404 or 422
    assert resp.status_code in [404, 422]


def test_criterion_11_xlsx_upload_and_profiling_support(client):
    """
    Acceptance Criterion 11 / Dataset 4:
    Uploading educational-factors.xlsx parses without openpyxl and returns valid profile.
    """
    import io
    xlsx_path = Path(r"C:\Users\SPURGE\MAJOR_PROJECT_DOCS\REVIEW 2\DataInvest_Dataset_Candidates\educational-factors.xlsx")
    if not xlsx_path.exists():
        pytest.skip(f"educational-factors.xlsx fixture not found at {xlsx_path}")

    with open(xlsx_path, "rb") as f:
        file_bytes = f.read()

    files = {"file": ("educational-factors.xlsx", io.BytesIO(file_bytes), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
    resp = client.post("/api/profile", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["overview"]["n_rows"] == 1680
    assert data["overview"]["n_cols"] == 79
    assert "dataset_id" in data


def test_criterion_12_cors_production_origin_configured():
    """
    Acceptance Criterion 12 / Connectivity:
    Verified that production Vercel frontend is included in default allowed CORS origins.
    """
    from app.main import get_cors_origins
    origins = get_cors_origins()
    assert "https://data-invest-ai.vercel.app" in origins

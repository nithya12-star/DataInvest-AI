"""
backend/tests/test_run_store.py
===============================
Unit and integration tests for Experiment Run Repository (RunStoreService and /api/runs).
Validates persistence, unique ID generation, no-overwrite guarantees,
restoration of full/partial states, and API endpoints.
"""

import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app
from app.services.run_store import RunStoreService, _sanitize_run_id, _generate_unique_run_id


@pytest.fixture
def temp_run_store(tmp_path):
    """Provides an isolated RunStoreService instance backed by a temporary directory."""
    return RunStoreService(runs_dir=tmp_path / "runs")


@pytest.fixture
def sample_run_payload():
    return {
        "dataset_name": "winequality-red.csv",
        "dataset_id": "winequality-red.csv",
        "active_phase": 2,
        "completed_phases": [1, 2],
        "objective": "Predict wine quality from physicochemical properties.",
        "profile": {
            "dataset_id": "winequality-red.csv",
            "overview": {"n_rows": 1599, "n_cols": 12},
            "column_types": {"numerical": ["fixed_acidity", "volatile_acidity", "quality"], "categorical": []},
        },
        "plan": {
            "dataset_id": "winequality-red.csv",
            "objective": "Predict wine quality",
            "analysis_tasks": [
                {"task_id": "task_1", "method": "pearson_correlation", "variables": ["volatile_acidity", "quality"]}
            ],
        },
        "analysis": None,
        "critic": None,
        "verification": None,
        "insights": None,
        "evaluation": None,
    }


def test_unique_run_id_generation():
    """Verify generated run IDs are timestamped, slugified, and collision-proof."""
    id1 = _generate_unique_run_id("winequality-red.csv")
    id2 = _generate_unique_run_id("winequality-red.csv")
    assert id1.startswith("run_")
    assert "winequality_red" in id1
    assert id1 != id2  # Collision-proof suffix


def test_sanitize_run_id():
    """Verify path traversal prevention."""
    assert _sanitize_run_id("run_20261019_wine_123") == "run_20261019_wine_123"
    assert _sanitize_run_id("../run_evil") is None
    assert _sanitize_run_id("evil_id") is None
    assert _sanitize_run_id("") is None


def test_save_and_retrieve_run(temp_run_store, sample_run_payload):
    """Verify run is saved to disk and can be retrieved with identical content."""
    res = temp_run_store.save_run(dict(sample_run_payload))
    assert res["status"] == "saved"
    run_id = res["run_id"]
    assert run_id is not None

    loaded = temp_run_store.get_run(run_id)
    assert loaded is not None
    assert loaded["run_id"] == run_id
    assert loaded["dataset_name"] == "winequality-red.csv"
    assert loaded["profile"]["overview"]["n_rows"] == 1599
    assert loaded["plan"]["analysis_tasks"][0]["task_id"] == "task_1"
    assert loaded["completed_phases"] == [1, 2]


def test_no_overwrite_same_dataset_upload(temp_run_store, sample_run_payload):
    """Uploading the same dataset multiple times must create distinct, non-overwriting runs."""
    payload1 = dict(sample_run_payload)
    payload2 = dict(sample_run_payload)

    res1 = temp_run_store.save_run(payload1)
    res2 = temp_run_store.save_run(payload2)

    assert res1["run_id"] != res2["run_id"]
    runs = temp_run_store.list_runs()
    assert len(runs) == 2
    ids = {r.run_id for r in runs}
    assert res1["run_id"] in ids
    assert res2["run_id"] in ids


def test_update_existing_run_in_place(temp_run_store, sample_run_payload):
    """Saving with an existing run_id must update that run rather than creating a duplicate."""
    res1 = temp_run_store.save_run(dict(sample_run_payload))
    run_id = res1["run_id"]

    # Now simulate Phase 3 completion
    updated_payload = dict(sample_run_payload)
    updated_payload["run_id"] = run_id
    updated_payload["active_phase"] = 3
    updated_payload["completed_phases"] = [1, 2, 3]
    updated_payload["analysis"] = {"results": [{"task_id": "task_1", "status": "SUCCESS"}]}

    res2 = temp_run_store.save_run(updated_payload)
    assert res2["run_id"] == run_id

    runs = temp_run_store.list_runs()
    assert len(runs) == 1
    assert runs[0].run_id == run_id
    assert runs[0].active_phase == 3
    assert runs[0].completed_phases == [1, 2, 3]

    loaded = temp_run_store.get_run(run_id)
    assert loaded["analysis"]["results"][0]["status"] == "SUCCESS"


def test_list_runs_summaries_and_sorting(temp_run_store):
    """Verify summaries extract dimensions and evaluation scores properly."""
    run1 = {
        "dataset_name": "first.csv",
        "profile": {"overview": {"n_rows": 100, "n_cols": 5}},
        "evaluation": {"summary": {"overall_score": 92.5, "overall_grade": "A+"}},
    }
    run2 = {
        "dataset_name": "second.csv",
        "profile": {"overview": {"n_rows": 200, "n_cols": 8}},
    }

    res1 = temp_run_store.save_run(run1)
    res2 = temp_run_store.save_run(run2)

    summaries = temp_run_store.list_runs()
    assert len(summaries) == 2
    # newest first
    assert summaries[0].run_id == res2["run_id"]
    assert summaries[0].n_rows == 200

    s1 = next(s for s in summaries if s.run_id == res1["run_id"])
    assert s1.n_rows == 100
    assert s1.overall_score == 92.5
    assert s1.overall_grade == "A+"


def test_delete_run(temp_run_store, sample_run_payload):
    """Verify deleting a run file."""
    res = temp_run_store.save_run(dict(sample_run_payload))
    run_id = res["run_id"]
    assert temp_run_store.get_run(run_id) is not None

    deleted = temp_run_store.delete_run(run_id)
    assert deleted is True
    assert temp_run_store.get_run(run_id) is None

    # Deleting non-existent run returns False
    assert temp_run_store.delete_run("run_non_existent") is False


def test_api_runs_endpoints():
    """Integration test of /api/runs endpoints via FastAPI TestClient."""
    client = TestClient(app)

    # 1. Save new run
    payload = {
        "dataset_name": "api_test.csv",
        "dataset_id": "api_test.csv",
        "active_phase": 1,
        "completed_phases": [1],
        "profile": {"overview": {"n_rows": 50, "n_cols": 3}},
    }
    post_resp = client.post("/api/runs", json=payload)
    assert post_resp.status_code == 200
    data = post_resp.json()
    assert data["status"] == "saved"
    run_id = data["run_id"]
    assert run_id.startswith("run_")

    # 2. List runs
    list_resp = client.get("/api/runs")
    assert list_resp.status_code == 200
    list_data = list_resp.json()
    assert list_data["total_runs"] >= 1
    matching = [r for r in list_data["runs"] if r["run_id"] == run_id]
    assert len(matching) == 1
    assert matching[0]["dataset_name"] == "api_test.csv"
    assert matching[0]["n_rows"] == 50

    # 3. Get single run
    get_resp = client.get(f"/api/runs/{run_id}")
    assert get_resp.status_code == 200
    run_data = get_resp.json()
    assert run_data["run_id"] == run_id
    assert run_data["profile"]["overview"]["n_rows"] == 50

    # 4. Delete run
    del_resp = client.delete(f"/api/runs/{run_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "deleted"

    # 5. Verify 404 after deletion
    get_again = client.get(f"/api/runs/{run_id}")
    assert get_again.status_code == 404

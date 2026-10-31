"""
backend/tests/test_run_visual_evidence_persistence.py
=====================================================
Automated tests for visualization evidence persistence and restoration.
Validates:
1. Serialization & atomic disk persistence of Phase 3 & Phase 5 visual evidence.
2. Extraction of `has_visual_evidence` flag and metadata in RunSummary.
3. Perfect preservation of Vega-Lite `figure_spec`, `visual_diagnostics`, and dual-graph references on reload.
4. Backward compatibility with older runs that lack visual evidence.
5. End-to-end API restoration via FastAPI TestClient.
"""

import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app
from app.services.run_store import RunStoreService


@pytest.fixture
def temp_run_store(tmp_path):
    """Provides an isolated RunStoreService instance backed by a temporary directory."""
    return RunStoreService(runs_dir=tmp_path / "runs")


@pytest.fixture
def synthetic_visual_evidence():
    """Synthetic Vega-Lite visual evidence fixture."""
    return {
        "graph_id": "graph_task_1_scatter",
        "task_id": "task_1",
        "figure_type": "scatter",
        "title": "Alcohol vs Quality Correlation",
        "x_label": "alcohol",
        "y_label": "quality",
        "columns": ["alcohol", "quality"],
        "method": "pearson_correlation",
        "data_summary": {"sample_size": 1599, "bivariate_outliers": 5},
        "figure_spec": {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "data": {
                "values": [
                    {"alcohol": 9.4, "quality": 5},
                    {"alcohol": 10.5, "quality": 6},
                    {"alcohol": 12.0, "quality": 7},
                ]
            },
            "mark": {"type": "circle", "color": "#00BFFF", "opacity": 0.75, "size": 60},
            "encoding": {
                "x": {"field": "alcohol", "type": "quantitative"},
                "y": {"field": "quality", "type": "quantitative"},
            },
        },
        "visual_diagnostics": {
            "outlier_points_count": 5,
            "linearity_indicator": "Linear",
            "distribution_shape": None,
            "visual_notes": [
                "Moderate positive correlation visible across dynamic range.",
                "Detected 5 bivariate outlier points via Mahalanobis distance at alpha=0.01.",
            ],
            "vision_model_used": False,
            "visual_inspection_claimed": False,
        },
    }


def test_save_and_reload_phase3_visual_evidence(temp_run_store, synthetic_visual_evidence):
    """Verify that visual evidence generated during Phase 3 is serialized and reloaded intact."""
    run_payload = {
        "dataset_name": "test_wine.csv",
        "dataset_id": "test_wine.csv",
        "active_phase": 3,
        "completed_phases": [1, 2, 3],
        "mode": "proposed",
        "profile": {
            "dataset_id": "test_wine.csv",
            "overview": {"n_rows": 1599, "n_cols": 12},
        },
        "analysis": {
            "results": [
                {
                    "task_id": "task_1",
                    "status": "success",
                    "analysis_type": "bivariate_correlation",
                    "method": "pearson_correlation",
                    "columns": ["alcohol", "quality"],
                    "results": {"correlation": 0.476, "p_value": 1.2e-90},
                    "visual_evidence": synthetic_visual_evidence,
                }
            ]
        },
    }

    # Save run
    save_result = temp_run_store.save_run(run_payload)
    assert save_result["status"] == "saved"
    run_id = save_result["run_id"]
    assert save_result["summary"]["has_visual_evidence"] is True
    assert save_result["summary"]["mode"] == "proposed"

    # Verify summary listing
    runs = temp_run_store.list_runs()
    assert len(runs) == 1
    assert runs[0].has_visual_evidence is True
    assert runs[0].mode == "proposed"

    # Reload full run and verify exact spec restoration
    reloaded = temp_run_store.get_run(run_id)
    assert reloaded is not None
    assert reloaded["has_visual_evidence"] is True
    assert reloaded["mode"] == "proposed"

    loaded_results = reloaded["analysis"]["results"]
    assert len(loaded_results) == 1
    loaded_viz = loaded_results[0]["visual_evidence"]
    assert loaded_viz is not None
    assert loaded_viz["figure_type"] == "scatter"
    assert loaded_viz["figure_spec"]["mark"]["type"] == "circle"
    assert loaded_viz["figure_spec"]["data"]["values"][0]["alcohol"] == 9.4
    assert loaded_viz["visual_diagnostics"]["vision_model_used"] is False
    assert loaded_viz["visual_diagnostics"]["outlier_points_count"] == 5


def test_save_and_reload_phase5_dual_visual_evidence(temp_run_store, synthetic_visual_evidence):
    """Verify that Phase 5 reinvestigation dual visual evidence (orig + follow-up) is persisted."""
    reinvest_viz = dict(synthetic_visual_evidence)
    reinvest_viz["graph_id"] = "graph_task_1_reinvest_scatter"
    reinvest_viz["title"] = "Alcohol vs Quality (Spearman Re-investigation)"
    reinvest_viz["method"] = "spearman_correlation"

    run_payload = {
        "dataset_name": "test_churn.csv",
        "active_phase": 5,
        "completed_phases": [1, 2, 3, 4, 5],
        "mode": "proposed",
        "paired_run_id": "run_baseline_123",
        "verification": {
            "verified_findings": [
                {
                    "finding_id": "finding_1",
                    "task_id": "task_1",
                    "status": "VERIFIED_AFTER_REINVESTIGATION",
                    "confidence": 0.95,
                    "visual_evidence": reinvest_viz,
                    "reinvestigated": True,
                    "reinvestigation_record": {
                        "reinvestigation_id": "reinv_task_1_1",
                        "original_visual_evidence": synthetic_visual_evidence,
                        "reinvestigated_visual_evidence": reinvest_viz,
                    },
                }
            ],
            "reinvestigations": [
                {
                    "reinvestigation_id": "reinv_task_1_1",
                    "original_visual_evidence": synthetic_visual_evidence,
                    "reinvestigated_visual_evidence": reinvest_viz,
                }
            ],
        },
    }

    save_result = temp_run_store.save_run(run_payload)
    run_id = save_result["run_id"]
    assert save_result["summary"]["has_visual_evidence"] is True
    assert save_result["summary"]["paired_run_id"] == "run_baseline_123"

    reloaded = temp_run_store.get_run(run_id)
    assert reloaded["has_visual_evidence"] is True
    assert reloaded["paired_run_id"] == "run_baseline_123"

    vf = reloaded["verification"]["verified_findings"][0]
    assert vf["visual_evidence"]["graph_id"] == "graph_task_1_reinvest_scatter"
    assert vf["reinvestigation_record"]["original_visual_evidence"]["graph_id"] == "graph_task_1_scatter"
    assert vf["reinvestigation_record"]["reinvestigated_visual_evidence"]["graph_id"] == "graph_task_1_reinvest_scatter"


def test_backward_compatibility_older_runs_lacking_visual_evidence(temp_run_store):
    """Verify that older runs created prior to Phase 4 load cleanly with has_visual_evidence=False."""
    older_run_payload = {
        "run_id": "run_20261019_wine_legacy",
        "dataset_name": "winequality-red.csv",
        "active_phase": 5,
        "completed_phases": [1, 2, 3, 4, 5],
        "profile": {"overview": {"n_rows": 1599, "n_cols": 12}},
        "analysis": {
            "results": [
                {
                    "task_id": "task_1",
                    "status": "success",
                    "results": {"mean": 8.32},
                    # No visual_evidence key
                }
            ]
        },
        "verification": {
            "verified_findings": [
                {
                    "finding_id": "finding_1",
                    "status": "VERIFIED",
                    "summary": "Historical finding without charts",
                    # No visual_evidence or reinvestigation_record
                }
            ]
        },
    }

    save_result = temp_run_store.save_run(older_run_payload)
    assert save_result["summary"]["has_visual_evidence"] is False

    summaries = temp_run_store.list_runs()
    assert len(summaries) == 1
    assert summaries[0].has_visual_evidence is False

    reloaded = temp_run_store.get_run("run_20261019_wine_legacy")
    assert reloaded is not None
    assert reloaded.get("has_visual_evidence") is not True
    assert reloaded["verification"]["verified_findings"][0]["summary"] == "Historical finding without charts"


def test_api_runs_endpoints_with_visual_evidence(synthetic_visual_evidence):
    """End-to-end API test: POST run with visual evidence, list it, reload it, verify it, and delete it."""
    client = TestClient(app)

    payload = {
        "dataset_name": "e2e_viz_test.csv",
        "dataset_id": "e2e_viz_test.csv",
        "active_phase": 3,
        "completed_phases": [1, 2, 3],
        "mode": "proposed",
        "paired_run_id": "run_paired_base_999",
        "profile": {"overview": {"n_rows": 200, "n_cols": 4}},
        "analysis": {
            "results": [
                {
                    "task_id": "task_e2e_1",
                    "status": "success",
                    "visual_evidence": synthetic_visual_evidence,
                }
            ]
        },
    }

    # 1. POST /api/runs
    post_resp = client.post("/api/runs", json=payload)
    assert post_resp.status_code == 200, post_resp.text
    save_data = post_resp.json()
    assert save_data["status"] == "saved"
    run_id = save_data["run_id"]
    assert save_data["summary"]["has_visual_evidence"] is True
    assert save_data["summary"]["mode"] == "proposed"

    try:
        # 2. GET /api/runs (History listing)
        list_resp = client.get("/api/runs")
        assert list_resp.status_code == 200
        runs_list = list_resp.json()["runs"]
        matching = [r for r in runs_list if r["run_id"] == run_id]
        assert len(matching) == 1
        assert matching[0]["has_visual_evidence"] is True
        assert matching[0]["mode"] == "proposed"
        assert matching[0]["paired_run_id"] == "run_paired_base_999"

        # 3. GET /api/runs/{run_id} (History drawer restoration)
        get_resp = client.get(f"/api/runs/{run_id}")
        assert get_resp.status_code == 200
        run_data = get_resp.json()
        assert run_data["run_id"] == run_id
        assert run_data["mode"] == "proposed"
        assert run_data["paired_run_id"] == "run_paired_base_999"

        # Assert visual evidence is fully restored
        restored_results = run_data["analysis"]["results"]
        assert len(restored_results) == 1
        restored_viz = restored_results[0]["visual_evidence"]
        assert restored_viz is not None
        assert restored_viz["graph_id"] == "graph_task_1_scatter"
        assert restored_viz["figure_type"] == "scatter"
        assert restored_viz["title"] == "Alcohol vs Quality Correlation"
        assert restored_viz["figure_spec"]["mark"]["type"] == "circle"
        assert restored_viz["visual_diagnostics"]["outlier_points_count"] == 5

    finally:
        # 4. Clean up test run file so persistent storage is untouched
        del_resp = client.delete(f"/api/runs/{run_id}")
        assert del_resp.status_code == 200

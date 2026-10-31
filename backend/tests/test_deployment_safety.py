"""
backend/tests/test_deployment_safety.py
======================================
Tests verifying deployment readiness and production security controls:
1. CORS origins parsing without wildcard relaxation
2. Production environment detection
3. Destructive DELETE /api/runs/{run_id} protection in production
4. In-memory sliding-window abuse rate limiter
5. Benchmark figure path-traversal sanitization
6. Benchmark data endpoints graceful availability handling
"""

import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import (
    app,
    get_cors_origins,
    is_production_environment,
    check_rate_limit,
    get_benchmark_results_root,
    _rate_limits,
)


@pytest.fixture
def client():
    return TestClient(app)


# ──────────────────────────────────────────────────────────────────────────────
# 1. CORS Configuration Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_cors_origins_defaults(monkeypatch):
    """Verify default development localhost origins are present when CORS_ORIGINS is unset."""
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    origins = get_cors_origins()
    assert "http://localhost:5173" in origins
    assert "http://127.0.0.1:5173" in origins
    assert "*" not in origins


def test_cors_origins_custom_domains(monkeypatch):
    """Verify custom production origins are parsed, normalized, and deduplicated."""
    monkeypatch.setenv(
        "CORS_ORIGINS",
        "https://datainvest-review.vercel.app, https://custom-domain.org/, http://localhost:5173",
    )
    origins = get_cors_origins()
    assert "https://datainvest-review.vercel.app" in origins
    assert "https://custom-domain.org" in origins  # Trailing slash stripped
    assert origins.count("http://localhost:5173") == 1  # Deduplicated


def test_cors_origins_rejects_wildcard(monkeypatch):
    """Verify wildcard '*' is strictly discarded to enforce security constraints."""
    monkeypatch.setenv("CORS_ORIGINS", "*, https://trusted.example.com")
    origins = get_cors_origins()
    assert "*" not in origins
    assert "https://trusted.example.com" in origins


# ──────────────────────────────────────────────────────────────────────────────
# 2. Production Environment Detection Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_is_production_environment(monkeypatch):
    """Verify accurate detection across various deployment environment flags."""
    monkeypatch.delenv("DATAINVEST_PRODUCTION", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.delenv("RENDER", raising=False)
    assert is_production_environment() is False

    monkeypatch.setenv("DATAINVEST_PRODUCTION", "true")
    assert is_production_environment() is True

    monkeypatch.setenv("DATAINVEST_PRODUCTION", "1")
    assert is_production_environment() is True

    monkeypatch.delenv("DATAINVEST_PRODUCTION", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "production")
    assert is_production_environment() is True

    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.setenv("RENDER", "true")
    assert is_production_environment() is True


# ──────────────────────────────────────────────────────────────────────────────
# 3. Production Delete Protection Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_delete_run_forbidden_in_production_without_auth(client, monkeypatch):
    """In production, DELETE /api/runs/{id} returns 403 Forbidden without admin credentials."""
    monkeypatch.setenv("DATAINVEST_PRODUCTION", "true")
    monkeypatch.delenv("ADMIN_API_TOKEN", raising=False)

    response = client.delete("/api/runs/test_run_123")
    assert response.status_code == 403
    assert "Run deletion is disabled in production" in response.json()["error"]


def test_delete_run_forbidden_with_invalid_token(client, monkeypatch):
    """In production, DELETE /api/runs/{id} returns 403 Forbidden with mismatched token."""
    monkeypatch.setenv("DATAINVEST_PRODUCTION", "true")
    monkeypatch.setenv("ADMIN_API_TOKEN", "secret-admin-pass-xyz")

    response = client.delete(
        "/api/runs/test_run_123",
        headers={"x-admin-token": "wrong-token"},
    )
    assert response.status_code == 403


def test_delete_run_allowed_with_valid_admin_token(client, monkeypatch):
    """In production, DELETE /api/runs/{id} proceeds when valid admin token is supplied."""
    monkeypatch.setenv("DATAINVEST_PRODUCTION", "true")
    monkeypatch.setenv("ADMIN_API_TOKEN", "secret-admin-pass-xyz")

    # Attempt delete on nonexistent ID; should pass auth guard and return 404 from run_store
    response = client.delete(
        "/api/runs/nonexistent_test_run",
        headers={"x-admin-token": "secret-admin-pass-xyz"},
    )
    assert response.status_code == 404
    assert "not found" in response.json()["error"]


# ──────────────────────────────────────────────────────────────────────────────
# 4. In-Memory Rate Limiting Abuse Protection Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_rate_limiter_allows_and_throttles():
    """Verify rate limiter permits requests up to max_requests and throttles thereafter."""
    test_ip = "192.0.2.100"
    _rate_limits[test_ip] = []

    # First 3 requests allowed under max_requests=3
    assert check_rate_limit(test_ip, max_requests=3, window_seconds=10) is True
    assert check_rate_limit(test_ip, max_requests=3, window_seconds=10) is True
    assert check_rate_limit(test_ip, max_requests=3, window_seconds=10) is True

    # 4th request rejected
    assert check_rate_limit(test_ip, max_requests=3, window_seconds=10) is False

    # Different IP is unaffected
    other_ip = "192.0.2.101"
    assert check_rate_limit(other_ip, max_requests=3, window_seconds=10) is True


# ──────────────────────────────────────────────────────────────────────────────
# 5. Figure Endpoint Path Traversal Sanitization Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_figure_endpoint_rejects_path_traversal(client):
    """Verify directory traversal attempts in figure endpoint return HTTP 400 Bad Request."""
    response = client.get("/api/benchmarks/figure/..%2F..%2Fsecret.env")
    assert response.status_code == 400
    assert "Invalid figure filename" in response.json()["error"]


def test_figure_endpoint_returns_404_for_missing(client):
    """Verify request for non-existent figure returns clean 404 without leaking paths."""
    response = client.get("/api/benchmarks/figure/nonexistent_plot.png")
    assert response.status_code == 404
    assert "not found" in response.json()["error"]


# ──────────────────────────────────────────────────────────────────────────────
# 6. Benchmark Results Root and Data Handling Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_benchmark_results_root_resolution():
    """Verify get_benchmark_results_root returns a valid Path object."""
    root = get_benchmark_results_root()
    assert isinstance(root, Path)


def test_all_datasets_benchmark_endpoint_structure(client):
    """Verify GET /api/benchmarks/all-datasets returns valid JSON without 500 error."""
    response = client.get("/api/benchmarks/all-datasets")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "benchmark_name" in data
    assert "provenance" in data
    assert "citation" in data
    assert "datasets" in data


def test_plan_rate_limiting_in_production(client, monkeypatch):
    """Verify POST /api/plan returns 429 when rate limit is exceeded in production."""
    monkeypatch.setenv("DATAINVEST_PRODUCTION", "true")
    # Reset limits for test client IP
    _rate_limits.clear()

    # Fill rate limit quota with empty/minimal requests
    for _ in range(20):
        client.post("/api/plan", json={"profile": {"test": 1}, "objective": "test"})

    # 21st request triggers rate limiter
    resp = client.post("/api/plan", json={"profile": {"test": 1}, "objective": "test"})
    assert resp.status_code == 429
    assert "Rate limit exceeded" in resp.json()["error"]


def test_plan_unthrottled_in_development(client, monkeypatch):
    """Verify POST /api/plan is never throttled in local development/testing environment."""
    monkeypatch.delenv("DATAINVEST_PRODUCTION", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.delenv("RENDER", raising=False)
    _rate_limits.clear()

    # Rapid requests in dev mode should not return 429
    for _ in range(25):
        resp = client.post("/api/plan", json={"profile": {}, "objective": "test"})
        # Should be 422 (validation error on empty profile), NOT 429
        assert resp.status_code != 429


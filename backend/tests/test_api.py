"""
backend/tests/test_api.py
==========================
DataInvest AI — FastAPI Integration & Endpoint Hardening Tests

Covers:
- Root & health metadata endpoints
- Valid CSV upload & response structure
- Extension validation (.csv only)
- Empty payload rejection
- Header-only CSV rejection
- Encoding resilience via endpoint (Latin-1 / UTF-8-sig)
- Pydantic response contract validation
- Real sample dataset upload & validation
"""

from __future__ import annotations

import io
import os
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.profile import ProfileResponse

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["app"] == "DataInvest AI"
    assert data["phase"] in (1, 2, 3, 4, 5, 6, 7)



def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_upload_valid_csv():
    csv_content = b"id,name,age,salary\n1,Alice,30,70000\n2,Bob,25,60000\n3,Charlie,35,80000\n"
    response = client.post(
        "/api/profile",
        files={"file": ("test.csv", io.BytesIO(csv_content), "text/csv")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test.csv"
    assert data["overview"]["n_rows"] == 3
    assert data["overview"]["n_cols"] == 4
    assert data["overview"]["column_names"] == ["id", "name", "age", "salary"]
    assert "preview_rows" in data
    assert len(data["preview_rows"]) == 3
    assert data["missing"]["total_missing"] == 0
    assert data["duplicates"]["n_duplicates"] == 0

    # Validate against Pydantic ProfileResponse model
    model = ProfileResponse.model_validate(data)
    assert model.overview.n_rows == 3


def test_upload_non_csv_file():
    response = client.post(
        "/api/profile",
        files={"file": ("test.pdf", io.BytesIO(b"Hello world"), "application/pdf")},
    )
    assert response.status_code == 422
    data = response.json()
    assert "error" in data
    assert "Only CSV and TXT tabular files are supported" in data["error"]



def test_upload_empty_csv():
    response = client.post(
        "/api/profile",
        files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")},
    )
    assert response.status_code == 422
    data = response.json()
    assert "error" in data
    assert "empty" in data["error"].lower()


def test_upload_csv_header_only():
    response = client.post(
        "/api/profile",
        files={"file": ("header_only.csv", io.BytesIO(b"col1,col2\n"), "text/csv")},
    )
    assert response.status_code == 422
    data = response.json()
    assert "error" in data


def test_upload_latin1_encoded_csv():
    content = "emp_name,city\nFrançois,Montréal\nJürgen,München\n".encode("latin-1")
    response = client.post(
        "/api/profile",
        files={"file": ("accents.csv", io.BytesIO(content), "text/csv")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["overview"]["n_rows"] == 2


def test_upload_sample_dataset():
    sample_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample.csv")
    with open(sample_path, "rb") as f:
        content = f.read()

    response = client.post(
        "/api/profile",
        files={"file": ("sample.csv", io.BytesIO(content), "text/csv")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["overview"]["n_rows"] == 60
    assert data["overview"]["n_cols"] == 8
    assert "salary" in data["numerical_stats"]
    assert data["numerical_stats"]["salary"]["mean"] > 0
    assert "notes" in data["missing"]["per_column"]

    # Verify Pydantic schema validation passes on real dataset
    model = ProfileResponse.model_validate(data)
    assert model.overview.n_rows == 60

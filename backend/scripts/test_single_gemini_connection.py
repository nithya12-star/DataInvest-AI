"""
backend/scripts/test_single_gemini_connection.py
================================================
Executes exactly ONE live Gemini API connection test.
Captures HTTP status, latency, response model, and token usage.
Saves results to DataInvest_Research_Results/gemini_connection_test_20261029/
Redacts all API keys and credentials.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add backend directory to sys.path
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from dotenv import load_dotenv
load_dotenv(_BACKEND_DIR / ".env")

import httpx

api_key = os.environ.get("GEMINI_API_KEY", "")


def sanitize(text: str) -> str:
    """Redact sensitive API keys from text."""
    if not text:
        return ""
    if api_key and len(api_key) > 5:
        text = text.replace(api_key, "[REDACTED_API_KEY]")
    text = re.sub(r"AIza[0-9A-Za-z-_]{35}", "[REDACTED_API_KEY]", text)
    return text


def run_single_connection_test() -> dict:
    model = "gemini-3.5-flash-lite"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "Respond in JSON with a single key 'connection' with value 'active'."}],
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 100,
            "responseMimeType": "application/json",
        },
    }

    t_start = time.perf_counter()
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, headers=headers, json=payload)
            latency = round(time.perf_counter() - t_start, 4)
            status_code = response.status_code
            response_headers = {
                k: sanitize(v)
                for k, v in response.headers.items()
                if k.lower() in ("content-type", "date", "server")
            }
            raw_text = sanitize(response.text)
            try:
                resp_json = response.json()
            except Exception:
                resp_json = None

            is_success = (status_code == 200)
            usage = resp_json.get("usageMetadata", {}) if isinstance(resp_json, dict) else {}

            candidate_text = ""
            response_model = ""
            if isinstance(resp_json, dict):
                response_model = resp_json.get("modelVersion", model)
                candidates = resp_json.get("candidates", [])
                if candidates and isinstance(candidates[0], dict):
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts and isinstance(parts[0], dict):
                        candidate_text = parts[0].get("text", "")

            result = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model_requested": model,
                "model_returned": response_model or model,
                "endpoint": sanitize(url),
                "http_status": status_code,
                "success": is_success,
                "latency_seconds": latency,
                "response_headers": response_headers,
                "usage_metadata": usage,
                "candidate_text": candidate_text,
                "raw_response": json.loads(raw_text) if raw_text.startswith("{") else raw_text,
            }
    except Exception as exc:
        latency = round(time.perf_counter() - t_start, 4)
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_requested": model,
            "endpoint": sanitize(url),
            "http_status": getattr(exc, "status_code", None),
            "success": False,
            "latency_seconds": latency,
            "error_type": type(exc).__name__,
            "error_message": sanitize(str(exc)),
        }

    out_dir = Path(r"C:\Users\SPURGE\MAJOR_PROJECT_DOCS\REVIEW 2\DataInvest_Research_Results\gemini_connection_test_20261029")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "connection_test_result.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    res = run_single_connection_test()
    print("=== LIVE GEMINI TEST OUTCOME ===")
    print("HTTP Status Code:", res.get("http_status"))
    print("Execution Success:", res.get("success"))
    print("Model Requested:", res.get("model_requested"))
    print("Model Returned:", res.get("model_returned"))
    print("Latency:", f"{res.get('latency_seconds')} seconds")
    if res.get("success"):
        print("Usage Metadata (Tokens):", json.dumps(res.get("usage_metadata"), indent=2))
        print("Generated Output Text:", res.get("candidate_text"))
    else:
        print("Raw Error Response:", json.dumps(res.get("raw_response") or res.get("error_message"), indent=2))


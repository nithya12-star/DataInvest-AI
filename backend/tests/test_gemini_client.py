"""
backend/tests/test_gemini_client.py
====================================
Offline unit tests for GeminiClient using httpx.MockTransport.

CRITICAL CONSTRAINTS:
- No real API keys required.
- Zero network requests / no external network sockets used.
- Complete coverage of security, missing keys, auth failures, rate limits,
  malformed responses, safety blocks, network errors, and JSON parsing.
"""

import json
from typing import Any
import httpx
import pytest

from app.services.gemini_client import (
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_MODEL,
    DEFAULT_POOL_TIMEOUT,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_WRITE_TIMEOUT,
    GeminiAuthenticationError,
    GeminiClient,
    GeminiClientError,
    GeminiMalformedResponseError,
    GeminiMissingKeyError,
    GeminiNetworkError,
    GeminiRateLimitError,
    GeminiResponse,
    GeminiTimeoutError,
)


# ============================================================================
# Helpers & Fixtures
# ============================================================================

def make_gemini_json_response(
    text: str,
    finish_reason: str = "STOP",
    prompt_tokens: int = 120,
    candidates_tokens: int = 45,
    status_code: int = 200,
) -> httpx.Response:
    """Build mock Gemini REST response conforming to Google Gemini specification."""
    payload = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": text}],
                },
                "finishReason": finish_reason,
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": prompt_tokens,
            "candidatesTokenCount": candidates_tokens,
            "totalTokenCount": prompt_tokens + candidates_tokens,
        },
        "modelVersion": "gemini-3.5-flash-lite",
    }
    return httpx.Response(
        status_code=status_code,
        headers={"Content-Type": "application/json"},
        content=json.dumps(payload).encode("utf-8"),
    )


# ============================================================================
# 1. API Key Security & Configuration Tests
# ============================================================================

def test_missing_api_key_raises_explicit_error(monkeypatch):
    """Client must raise GeminiMissingKeyError if no key is configured."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    client = GeminiClient(api_key=None)
    assert not client.is_configured

    with pytest.raises(GeminiMissingKeyError) as exc_info:
        client.generate_content("Hello")

    assert "GEMINI_API_KEY" in str(exc_info.value)


def test_empty_string_api_key_raises_error():
    """Client initialized with blank string must raise GeminiMissingKeyError."""
    client = GeminiClient(api_key="   ")
    assert not client.is_configured

    with pytest.raises(GeminiMissingKeyError):
        client.generate_content("Hello")


def test_api_key_never_exposed_in_repr_or_str():
    """API key value must NEVER appear in string representations or object inspects."""
    secret_key = "super_secret_test_key_abc123"
    client = GeminiClient(api_key=secret_key)

    repr_str = repr(client)
    str_str = str(client)

    assert secret_key not in repr_str
    assert secret_key not in str_str
    assert "configured=True" in repr_str
    assert DEFAULT_MODEL in repr_str


def test_model_defaults_to_gemini_3_5_flash_lite():
    """Default model must be verified gemini-3.5-flash-lite."""
    client = GeminiClient(api_key="test_key")
    assert client.model == "gemini-3.5-flash-lite"


# ============================================================================
# 2. Request Formatting & Successful Generation Tests
# ============================================================================

def test_generate_content_success_and_request_structure():
    """Verify request URL, headers, payload structure and response parsing."""
    captured_request: dict[str, Any] = {}

    def mock_handler(request: httpx.Request) -> httpx.Response:
        captured_request["method"] = request.method
        captured_request["url"] = str(request.url)
        captured_request["headers"] = dict(request.headers)
        captured_request["json"] = json.loads(request.content.decode("utf-8"))
        return make_gemini_json_response(
            text="Statistical analysis plan ready.",
            finish_reason="STOP",
            prompt_tokens=150,
            candidates_tokens=40,
        )

    transport = httpx.MockTransport(mock_handler)
    client = GeminiClient(api_key="mock_secret_key", transport=transport)

    response = client.generate_content(
        prompt="Propose 3 hypothesis tests",
        system_instruction="You are a data science assistant",
        temperature=0.3,
        max_output_tokens=2048,
    )

    # Validate Request
    assert captured_request["method"] == "POST"
    assert "models/gemini-3.5-flash-lite:generateContent" in captured_request["url"]
    assert captured_request["headers"]["x-goog-api-key"] == "mock_secret_key"
    assert captured_request["headers"]["content-type"] == "application/json"

    # Validate Payload
    body = captured_request["json"]
    assert body["contents"][0]["parts"][0]["text"] == "Propose 3 hypothesis tests"
    assert body["systemInstruction"]["parts"][0]["text"] == "You are a data science assistant"
    assert body["generationConfig"]["temperature"] == 0.3
    assert body["generationConfig"]["maxOutputTokens"] == 2048

    # Validate Response
    assert isinstance(response, GeminiResponse)
    assert response.text == "Statistical analysis plan ready."
    assert response.model == "gemini-3.5-flash-lite"
    assert response.finish_reason == "STOP"
    assert response.prompt_tokens == 150
    assert response.candidates_tokens == 40
    assert response.total_tokens == 190


def test_generate_json_success():
    """Verify generate_json sets responseMimeType and parses dictionary result."""
    expected_dict = {"mode": "bivariate", "tasks": ["t_test", "chi_square"]}

    def mock_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        return make_gemini_json_response(text=json.dumps(expected_dict))

    transport = httpx.MockTransport(mock_handler)
    client = GeminiClient(api_key="mock_key", transport=transport)

    result = client.generate_json("Generate JSON plan")
    assert result == expected_dict
    assert result["mode"] == "bivariate"
    assert len(result["tasks"]) == 2


def test_generate_json_strips_markdown_code_fences():
    """Verify that JSON wrapped in markdown ```json ... ``` blocks is cleaned properly."""
    expected_data = {"status": "ok", "count": 42}
    raw_markdown = f"```json\n{json.dumps(expected_data)}\n```"

    transport = httpx.MockTransport(lambda req: make_gemini_json_response(text=raw_markdown))
    client = GeminiClient(api_key="mock_key", transport=transport)

    result = client.generate_json("Return wrapped json")
    assert result == expected_data


# ============================================================================
# 3. Explicit Error Handling Tests (No Silent Fallback)
# ============================================================================

def test_authentication_error_on_401():
    """HTTP 401 must raise GeminiAuthenticationError explicitly."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=401,
            json={"error": {"code": 401, "message": "API_KEY_INVALID"}},
        )

    transport = httpx.MockTransport(mock_handler)
    client = GeminiClient(api_key="bad_key", transport=transport)

    with pytest.raises(GeminiAuthenticationError) as exc_info:
        client.generate_content("Test")

    assert exc_info.value.status_code == 401
    assert "API_KEY_INVALID" in str(exc_info.value)


def test_authentication_error_on_403():
    """HTTP 403 must raise GeminiAuthenticationError explicitly."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=403,
            json={"error": {"code": 403, "message": "PermissionDenied"}},
        )

    transport = httpx.MockTransport(mock_handler)
    client = GeminiClient(api_key="bad_key", transport=transport)

    with pytest.raises(GeminiAuthenticationError) as exc_info:
        client.generate_content("Test")

    assert exc_info.value.status_code == 403


def test_rate_limit_error_on_429():
    """HTTP 429 must raise GeminiRateLimitError and extract Retry-After."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=429,
            headers={"Retry-After": "15"},
            json={"error": {"code": 429, "message": "ResourceExhausted: Quota exceeded"}},
        )

    transport = httpx.MockTransport(mock_handler)
    client = GeminiClient(api_key="test_key", transport=transport)

    with pytest.raises(GeminiRateLimitError) as exc_info:
        client.generate_content("Test")

    assert exc_info.value.status_code == 429
    assert exc_info.value.retry_after == 15.0
    assert "ResourceExhausted" in str(exc_info.value)


def test_network_error_on_server_500():
    """HTTP 500 must raise GeminiNetworkError explicitly."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=500,
            json={"error": {"code": 500, "message": "Internal Server Error"}},
        )

    transport = httpx.MockTransport(mock_handler)
    client = GeminiClient(api_key="test_key", transport=transport)

    with pytest.raises(GeminiNetworkError) as exc_info:
        client.generate_content("Test")

    assert exc_info.value.status_code == 500


def test_network_error_on_connection_timeout():
    """httpx.TimeoutException must raise GeminiNetworkError."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Connection timed out after 30s")

    transport = httpx.MockTransport(mock_handler)
    client = GeminiClient(api_key="test_key", transport=transport)

    with pytest.raises(GeminiNetworkError) as exc_info:
        client.generate_content("Test")

    assert "Connection timed out" in str(exc_info.value)


def test_malformed_response_empty_candidates():
    """Empty candidates array must raise GeminiMalformedResponseError."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={"candidates": []},
        )

    transport = httpx.MockTransport(mock_handler)
    client = GeminiClient(api_key="test_key", transport=transport)

    with pytest.raises(GeminiMalformedResponseError) as exc_info:
        client.generate_content("Test")

    assert "no candidates" in str(exc_info.value)


def test_malformed_response_safety_blocked():
    """Safety filter block (finishReason='SAFETY') must raise GeminiMalformedResponseError."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "candidates": [
                    {
                        "finishReason": "SAFETY",
                        "index": 0,
                    }
                ]
            },
        )

    transport = httpx.MockTransport(mock_handler)
    client = GeminiClient(api_key="test_key", transport=transport)

    with pytest.raises(GeminiMalformedResponseError) as exc_info:
        client.generate_content("Test")

    assert "safety filters" in str(exc_info.value)


def test_generate_json_invalid_json_text_raises_error():
    """Non-JSON response text in generate_json must raise GeminiMalformedResponseError."""
    transport = httpx.MockTransport(
        lambda req: make_gemini_json_response(text="Not a valid json response")
    )
    client = GeminiClient(api_key="test_key", transport=transport)

    with pytest.raises(GeminiMalformedResponseError) as exc_info:
        client.generate_json("Test")

    assert "not valid JSON" in str(exc_info.value)


def test_empty_prompt_raises_error():
    """Calling generate_content with empty prompt must raise GeminiMalformedResponseError."""
    client = GeminiClient(api_key="test_key")
    with pytest.raises(GeminiMalformedResponseError):
        client.generate_content("   ")


def test_dotenv_loading_mechanism_offline(tmp_path, monkeypatch):
    """
    Offline automated test proving that the python-dotenv loading mechanism
    successfully loads a test API key from a .env file into the process environment
    and GeminiClient resolves it, without making any network requests.
    """
    from dotenv import load_dotenv

    # Ensure clean process environment
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    # Verify initially unconfigured
    unconfigured_client = GeminiClient()
    assert unconfigured_client.is_configured is False
    with pytest.raises(GeminiMissingKeyError):
        unconfigured_client._ensure_configured()

    # Create temporary .env with dummy test key
    temp_env = tmp_path / ".env"
    dummy_key = "test_offline_dummy_key_98765"
    temp_env.write_text(f"GEMINI_API_KEY={dummy_key}\n", encoding="utf-8")

    # Load from explicit path
    loaded = load_dotenv(dotenv_path=temp_env)
    assert loaded is True

    # GeminiClient should now read the key from the environment
    client = GeminiClient()
    assert client.is_configured is True
    assert client._api_key == dummy_key

    # Confirm representation never leaks the key
    assert dummy_key not in repr(client)
    assert dummy_key not in str(client)


def test_dotenv_google_api_key_fallback_offline(tmp_path, monkeypatch):
    """Verify GOOGLE_API_KEY is loaded from .env when GEMINI_API_KEY is absent."""
    from dotenv import load_dotenv

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    temp_env = tmp_path / ".env"
    dummy_fallback_key = "test_google_fallback_key_54321"
    temp_env.write_text(f"GOOGLE_API_KEY={dummy_fallback_key}\n", encoding="utf-8")

    load_dotenv(dotenv_path=temp_env)

    client = GeminiClient()
    assert client.is_configured is True
    assert client._api_key == dummy_fallback_key
    assert dummy_fallback_key not in repr(client)


def test_retry_on_503_eventual_success(monkeypatch):
    """Verify that an explicit HTTP 503 is retried and succeeds on a subsequent attempt."""
    sleep_calls = []
    monkeypatch.setattr("time.sleep", lambda s: sleep_calls.append(s))

    call_count = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(
                status_code=503,
                json={"error": {"message": "Model is experiencing high demand."}},
            )
        return make_gemini_json_response(text="GEMINI_CONNECTION_OK")

    transport = httpx.MockTransport(mock_handler)
    client = GeminiClient(api_key="test_key", transport=transport, jitter=False)

    resp = client.generate_content("Test prompt")
    assert resp.text == "GEMINI_CONNECTION_OK"
    assert call_count == 2
    assert sleep_calls == [2.0]


def test_retry_on_503_exhaustion_raises_network_error(monkeypatch):
    """Verify that 3 consecutive 503 responses exhaust retries and raise GeminiNetworkError with status_code 503."""
    sleep_calls = []
    monkeypatch.setattr("time.sleep", lambda s: sleep_calls.append(s))

    call_count = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(
            status_code=503,
            json={"error": {"message": "High demand spike."}},
        )

    transport = httpx.MockTransport(mock_handler)
    client = GeminiClient(api_key="test_key", transport=transport, jitter=False)

    with pytest.raises(GeminiNetworkError) as exc_info:
        client.generate_content("Test prompt")

    assert exc_info.value.status_code == 503
    assert "High demand spike" in str(exc_info.value)
    assert call_count == 3  # Initial + 2 retries
    assert sleep_calls == [2.0, 4.0]


def test_non_retryable_errors_fail_fast(monkeypatch):
    """Verify non-503 errors and exceptions are never retried and fail on attempt 1."""
    sleep_calls = []
    monkeypatch.setattr("time.sleep", lambda s: sleep_calls.append(s))

    # 1. HTTP 401 Unauthorized -> fails immediately
    call_count_401 = 0

    def handler_401(req):
        nonlocal call_count_401
        call_count_401 += 1
        return httpx.Response(status_code=401, json={"error": {"message": "Invalid API key."}})

    client_401 = GeminiClient(api_key="test_key", transport=httpx.MockTransport(handler_401))
    with pytest.raises(GeminiAuthenticationError):
        client_401.generate_content("Test")
    assert call_count_401 == 1

    # 2. HTTP 429 Rate Limit -> fails immediately
    call_count_429 = 0

    def handler_429(req):
        nonlocal call_count_429
        call_count_429 += 1
        return httpx.Response(status_code=429, json={"error": {"message": "Quota exceeded."}})

    client_429 = GeminiClient(api_key="test_key", transport=httpx.MockTransport(handler_429))
    with pytest.raises(GeminiRateLimitError):
        client_429.generate_content("Test")
    assert call_count_429 == 1

    # 3. HTTP 500 Server Error (non-503) -> fails immediately
    call_count_500 = 0

    def handler_500(req):
        nonlocal call_count_500
        call_count_500 += 1
        return httpx.Response(status_code=500, json={"error": {"message": "Internal error."}})

    client_500 = GeminiClient(api_key="test_key", transport=httpx.MockTransport(handler_500))
    with pytest.raises(GeminiNetworkError):
        client_500.generate_content("Test")
    assert call_count_500 == 1

    # Zero sleeps across all non-retryable errors
    assert len(sleep_calls) == 0


def test_retry_on_timeout_eventual_success(monkeypatch):
    """Verify that a first-attempt timeout followed by success succeeds on attempt 2."""
    sleep_calls = []
    monkeypatch.setattr("time.sleep", lambda s: sleep_calls.append(s))

    call_count = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise httpx.ReadTimeout("Read timed out")
        return make_gemini_json_response(text="GEMINI_TIMEOUT_RETRY_OK")

    transport = httpx.MockTransport(mock_handler)
    client = GeminiClient(api_key="test_key", transport=transport, jitter=False)

    resp = client.generate_content("Test prompt")
    assert resp.text == "GEMINI_TIMEOUT_RETRY_OK"
    assert call_count == 2
    assert sleep_calls == [2.0]


def test_retry_on_connection_error_eventual_success(monkeypatch):
    """Verify that a first-attempt connection failure followed by success succeeds on attempt 2."""
    sleep_calls = []
    monkeypatch.setattr("time.sleep", lambda s: sleep_calls.append(s))

    call_count = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise httpx.ConnectError("Connection refused")
        return make_gemini_json_response(text="GEMINI_CONNECT_RETRY_OK")

    transport = httpx.MockTransport(mock_handler)
    client = GeminiClient(api_key="test_key", transport=transport, jitter=False)

    resp = client.generate_content("Test prompt")
    assert resp.text == "GEMINI_CONNECT_RETRY_OK"
    assert call_count == 2
    assert sleep_calls == [2.0]


def test_retry_on_timeout_exhaustion_raises_gemini_timeout_error(monkeypatch):
    """Verify repeated timeout failures stop at the configured limit and raise GeminiTimeoutError."""
    sleep_calls = []
    monkeypatch.setattr("time.sleep", lambda s: sleep_calls.append(s))

    call_count = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        raise httpx.ConnectTimeout("Connect timed out")

    transport = httpx.MockTransport(mock_handler)
    client = GeminiClient(api_key="test_key", transport=transport, jitter=False)

    with pytest.raises(GeminiTimeoutError) as exc_info:
        client.generate_content("Test prompt")

    assert exc_info.value.status_code == 504
    assert "timeout" in str(exc_info.value).lower()
    assert call_count == 3  # Initial attempt + 2 retries
    assert sleep_calls == [2.0, 4.0]
    # Verify inheritance
    assert isinstance(exc_info.value, GeminiNetworkError)
    assert isinstance(exc_info.value, GeminiClientError)


def test_retry_on_connection_error_exhaustion_raises_gemini_network_error(monkeypatch):
    """Verify repeated connection failures stop at the configured limit and raise GeminiNetworkError."""
    sleep_calls = []
    monkeypatch.setattr("time.sleep", lambda s: sleep_calls.append(s))

    call_count = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        raise httpx.ConnectError("Network is unreachable")

    transport = httpx.MockTransport(mock_handler)
    client = GeminiClient(api_key="test_key", transport=transport, jitter=False)

    with pytest.raises(GeminiNetworkError):
        client.generate_content("Test prompt")

    assert call_count == 3
    assert sleep_calls == [2.0, 4.0]


def test_retry_after_header_handling(monkeypatch):
    """Verify Retry-After is respected when within bounds, and clamped to fallback when excessive."""
    sleep_calls = []
    monkeypatch.setattr("time.sleep", lambda s: sleep_calls.append(s))

    # Case 1: Valid short Retry-After (e.g. 1.5s <= 5.0s max)
    call_count_short = 0

    def handler_short(req):
        nonlocal call_count_short
        call_count_short += 1
        if call_count_short == 1:
            return httpx.Response(
                status_code=503,
                headers={"Retry-After": "1.5"},
                json={"error": {"message": "High demand."}},
            )
        return make_gemini_json_response(text="OK")

    client_short = GeminiClient(api_key="test_key", transport=httpx.MockTransport(handler_short), jitter=False)
    client_short.generate_content("Test")
    assert call_count_short == 2
    assert sleep_calls == [1.5]

    sleep_calls.clear()

    # Case 2: Excessive Retry-After (e.g. 60.0s > 10.0s max) -> uses fallback 2.0s
    call_count_excess = 0

    def handler_excess(req):
        nonlocal call_count_excess
        call_count_excess += 1
        if call_count_excess == 1:
            return httpx.Response(
                status_code=503,
                headers={"Retry-After": "60.0"},
                json={"error": {"message": "High demand."}},
            )
        return make_gemini_json_response(text="OK")

    client_excess = GeminiClient(api_key="test_key", transport=httpx.MockTransport(handler_excess), jitter=False)
    client_excess.generate_content("Test")
    assert call_count_excess == 2
    assert sleep_calls == [2.0]


def test_retry_exponential_backoff_with_jitter_bounds(monkeypatch):
    """Verify that when jitter is enabled, backoff delays fall within +/- 15% of exponential base."""
    sleep_calls = []
    monkeypatch.setattr("time.sleep", lambda s: sleep_calls.append(s))

    call_count = 0

    def mock_503(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(status_code=503, json={"error": {"message": "Server overloaded."}})

    transport = httpx.MockTransport(mock_503)
    client = GeminiClient(api_key="test_key", transport=transport, jitter=True)

    with pytest.raises(GeminiNetworkError) as exc_info:
        client.generate_content("Test jitter prompt")

    assert exc_info.value.status_code == 503
    assert call_count == 3  # Strictly 1 initial + 2 retries
    assert len(sleep_calls) == 2
    # Attempt 1: base 2.0s +/- 15% -> [1.70, 2.30]
    assert 1.65 <= sleep_calls[0] <= 2.35
    # Attempt 2: base 4.0s +/- 15% -> [3.40, 4.60]
    assert 3.35 <= sleep_calls[1] <= 4.65


def test_retry_after_header_within_expanded_bounds(monkeypatch):
    """Verify that a Retry-After header up to max_backoff (e.g. 7.5s <= 10.0s) is strictly honored."""
    sleep_calls = []
    monkeypatch.setattr("time.sleep", lambda s: sleep_calls.append(s))

    call_count = 0

    def mock_retry_after(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(
                status_code=503,
                headers={"Retry-After": "7.5"},
                json={"error": {"message": "Cooldown required."}},
            )
        return make_gemini_json_response(text="OK_AFTER_7_5")

    transport = httpx.MockTransport(mock_retry_after)
    client = GeminiClient(api_key="test_key", transport=transport, jitter=True)

    resp = client.generate_content("Test Retry-After")
    assert resp.text == "OK_AFTER_7_5"
    assert call_count == 2
    assert sleep_calls == [7.5]


# ============================================================================
# 8. Timeout Configuration & Validation Tests
# ============================================================================

def test_default_timeout_configuration(monkeypatch):
    """Client defaults to 60.0s read timeout and sensible connection limits when env is unset."""
    monkeypatch.delenv("GEMINI_TIMEOUT_SECONDS", raising=False)

    client = GeminiClient(api_key="test_key")
    assert client.timeout == 60.0
    assert client.timeout == DEFAULT_TIMEOUT_SECONDS
    assert client.connect_timeout == DEFAULT_CONNECT_TIMEOUT
    assert client.write_timeout == DEFAULT_WRITE_TIMEOUT
    assert client.pool_timeout == DEFAULT_POOL_TIMEOUT

    # Inspect httpx.Timeout internals
    assert client._httpx_timeout.read == 60.0
    assert client._httpx_timeout.connect == 10.0
    assert client._httpx_timeout.write == 10.0
    assert client._httpx_timeout.pool == 10.0
    assert client._http_client.timeout.read == 60.0
    assert client._http_client.timeout.connect == 10.0


def test_timeout_environment_variable_override(monkeypatch):
    """GEMINI_TIMEOUT_SECONDS environment variable overrides the default 60s timeout."""
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "45")
    client = GeminiClient(api_key="test_key")
    assert client.timeout == 45.0
    assert client._httpx_timeout.read == 45.0
    assert client._http_client.timeout.read == 45.0

    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "90.5")
    client_float = GeminiClient(api_key="test_key")
    assert client_float.timeout == 90.5
    assert client_float._httpx_timeout.read == 90.5
    assert client_float._http_client.timeout.read == 90.5


def test_timeout_explicit_constructor_override_precedence(monkeypatch):
    """Explicit constructor argument takes precedence over GEMINI_TIMEOUT_SECONDS."""
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "90.0")
    client = GeminiClient(api_key="test_key", timeout=30.0)
    assert client.timeout == 30.0
    assert client._httpx_timeout.read == 30.0
    assert client._http_client.timeout.read == 30.0


def test_invalid_timeout_configurations_rejected(monkeypatch):
    """Invalid, non-numeric, zero, negative, or boolean timeout values must be safely rejected with ValueError."""
    # 1. Non-numeric environment variable
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "not_a_number")
    with pytest.raises(ValueError, match="positive numeric value"):
        GeminiClient(api_key="test_key")

    # 2. Zero environment variable
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "0")
    with pytest.raises(ValueError, match="greater than 0 seconds"):
        GeminiClient(api_key="test_key")

    # 3. Negative environment variable
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "-15.0")
    with pytest.raises(ValueError, match="greater than 0 seconds"):
        GeminiClient(api_key="test_key")

    # 4. Explicit non-numeric constructor argument
    monkeypatch.delenv("GEMINI_TIMEOUT_SECONDS", raising=False)
    with pytest.raises(ValueError, match="positive numeric value"):
        GeminiClient(api_key="test_key", timeout="invalid")  # type: ignore[arg-type]

    # 5. Explicit zero constructor argument
    with pytest.raises(ValueError, match="greater than 0 seconds"):
        GeminiClient(api_key="test_key", timeout=0)

    # 6. Explicit negative constructor argument
    with pytest.raises(ValueError, match="greater than 0 seconds"):
        GeminiClient(api_key="test_key", timeout=-5.0)

    # 7. Boolean constructor argument
    with pytest.raises(ValueError, match="boolean values are not permitted"):
        GeminiClient(api_key="test_key", timeout=True)


def test_sensible_connection_and_write_limits_clamped_for_short_timeout(monkeypatch):
    """If a short timeout is configured (e.g. 5.0s), connect/write/pool limits are clamped so they do not exceed timeout."""
    monkeypatch.delenv("GEMINI_TIMEOUT_SECONDS", raising=False)
    client = GeminiClient(api_key="test_key", timeout=5.0)
    assert client.timeout == 5.0
    assert client._httpx_timeout.connect == 5.0
    assert client._httpx_timeout.read == 5.0
    assert client._httpx_timeout.write == 5.0
    assert client._httpx_timeout.pool == 5.0




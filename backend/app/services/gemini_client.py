"""
backend/app/services/gemini_client.py
======================================
Isolated, secure client for Google Gemini LLM API (gemini-3.5-flash-lite).

CRITICAL REQUIREMENTS:
- Read API key strictly from environment variable (GEMINI_API_KEY / GOOGLE_API_KEY).
- Never hardcode, expose, or log API keys.
- No silent fallback: Missing keys, auth failures, rate limits, network errors,
  and malformed responses explicitly raise typed domain exceptions.
- Zero mock or rule-based fallback during genuine runs.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

logger = logging.getLogger("datainvest.gemini")

DEFAULT_MODEL = "gemini-3.5-flash-lite"
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MAX_RETRIES = 2
DEFAULT_BASE_BACKOFF = 2.0
DEFAULT_MAX_BACKOFF = 10.0
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_CONNECT_TIMEOUT = 10.0
DEFAULT_WRITE_TIMEOUT = 10.0
DEFAULT_POOL_TIMEOUT = 10.0


def _validate_and_resolve_timeout(timeout: Optional[Any] = None) -> float:
    """
    Resolve and validate Gemini request timeout in seconds.

    Priority:
    1. Explicit constructor argument (if provided and not None)
    2. GEMINI_TIMEOUT_SECONDS environment variable
    3. DEFAULT_TIMEOUT_SECONDS (60.0)

    Raises
    ------
    ValueError
        If configured timeout is non-numeric, zero, or negative.
    """
    raw_val: Any = timeout
    source = "constructor argument 'timeout'"

    if raw_val is None:
        env_val = os.environ.get("GEMINI_TIMEOUT_SECONDS")
        if env_val is None or not env_val.strip():
            return DEFAULT_TIMEOUT_SECONDS
        raw_val = env_val.strip()
        source = "GEMINI_TIMEOUT_SECONDS environment variable"

    if isinstance(raw_val, bool):
        raise ValueError(
            f"Invalid timeout configured from {source}: boolean values are not permitted."
        )

    try:
        val = float(raw_val)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Invalid timeout configured from {source}: '{raw_val}'. Must be a positive numeric value in seconds."
        ) from exc

    if val <= 0:
        raise ValueError(
            f"Invalid timeout configured from {source}: {val}. Timeout must be greater than 0 seconds."
        )

    return val



# ============================================================================
# Domain Exceptions (Explicit Failure Hierarchy)
# ============================================================================

class GeminiClientError(Exception):
    """Base exception for all Gemini client errors."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code

    def __str__(self) -> str:
        if self.status_code:
            return f"[{self.status_code}] {self.message}"
        return self.message


class GeminiMissingKeyError(GeminiClientError):
    """Raised when the Gemini API key is missing or blank."""

    def __init__(self, message: str = "Gemini API key is missing. Set GEMINI_API_KEY environment variable."):
        super().__init__(message=message, status_code=None)


class GeminiAuthenticationError(GeminiClientError):
    """Raised on HTTP 401 or 403 (unauthorized or invalid API key)."""
    pass


class GeminiRateLimitError(GeminiClientError):
    """Raised on HTTP 429 (ResourceExhausted or quota breach)."""

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message=message, status_code=429)
        self.retry_after = retry_after


class GeminiNetworkError(GeminiClientError):
    """Raised on connection failures, DNS errors, or HTTP 5xx."""
    pass


class GeminiTimeoutError(GeminiNetworkError):
    """Raised specifically when a request to Gemini times out (connection, read, write, or pool timeout)."""

    def __init__(self, message: str = "Request to Gemini API timed out.", status_code: Optional[int] = 504):
        super().__init__(message=message, status_code=status_code)


class GeminiMalformedResponseError(GeminiClientError):
    """Raised when API returns invalid JSON, empty candidates, or safety blocks."""
    pass


# ============================================================================
# Response Model
# ============================================================================

@dataclass
class GeminiResponse:
    """Strongly typed response container for Gemini API generation."""
    text: str
    model: str
    finish_reason: str = "STOP"
    prompt_tokens: int = 0
    candidates_tokens: int = 0
    total_tokens: int = 0
    raw_response: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Gemini Client
# ============================================================================

class GeminiClient:
    """
    Isolated, production-grade Gemini LLM client.

    Parameters
    ----------
    api_key : str, optional
        API key. If not provided, read from GEMINI_API_KEY or GOOGLE_API_KEY environment variables.
    model : str, default 'gemini-3.5-flash-lite'
        Gemini model identifier verified against Google official documentation.
    timeout : float, optional
        Network timeout in seconds. If omitted, resolved from GEMINI_TIMEOUT_SECONDS
        environment variable or defaults to 60.0 seconds.
    transport : httpx.BaseTransport, optional
        Optional HTTP transport (e.g. httpx.MockTransport for offline unit testing).
    client : httpx.Client, optional
        Optional pre-configured httpx.Client instance.
    max_retries : int, default 2
        Maximum number of retry attempts on HTTP 503 responses.
    base_backoff : float, default 2.0
        Initial backoff delay in seconds for HTTP 503 retry.
    max_backoff : float, default 5.0
        Upper bound ceiling in seconds for backoff delay.
    connect_timeout : float, default 10.0
        Connection establishment timeout in seconds.
    write_timeout : float, default 10.0
        Network write timeout in seconds.
    pool_timeout : float, default 10.0
        Connection pool acquisition timeout in seconds.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        timeout: Optional[float] = None,
        transport: Optional[httpx.BaseTransport] = None,
        client: Optional[httpx.Client] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_backoff: float = DEFAULT_BASE_BACKOFF,
        max_backoff: float = DEFAULT_MAX_BACKOFF,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        write_timeout: float = DEFAULT_WRITE_TIMEOUT,
        pool_timeout: float = DEFAULT_POOL_TIMEOUT,
        jitter: bool = True,
    ):
        resolved_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        if not resolved_key or not resolved_key.strip():
            self._api_key = ""
        else:
            self._api_key = resolved_key.strip()

        self.model = model.strip() if model else DEFAULT_MODEL
        self.timeout = _validate_and_resolve_timeout(timeout)
        self.connect_timeout = max(0.0, float(connect_timeout))
        self.write_timeout = max(0.0, float(write_timeout))
        self.pool_timeout = max(0.0, float(pool_timeout))
        self.max_retries = max(0, int(max_retries))
        self.base_backoff = max(0.0, float(base_backoff))
        self.max_backoff = max(self.base_backoff, float(max_backoff))
        self.jitter = bool(jitter)

        effective_connect = min(self.connect_timeout, self.timeout)
        effective_write = min(self.write_timeout, self.timeout)
        effective_pool = min(self.pool_timeout, self.timeout)
        self._httpx_timeout = httpx.Timeout(
            self.timeout,
            connect=effective_connect,
            read=self.timeout,
            write=effective_write,
            pool=effective_pool,
        )

        if client is not None:
            self._http_client = client
        elif transport is not None:
            self._http_client = httpx.Client(transport=transport, timeout=self._httpx_timeout)
        else:
            self._http_client = httpx.Client(timeout=self._httpx_timeout)
        self.last_response: Optional[GeminiResponse] = None

    @property
    def is_configured(self) -> bool:
        """Check if an API key is available without exposing it."""
        return bool(self._api_key)

    def _ensure_configured(self) -> None:
        """Raise GeminiMissingKeyError if API key is not configured."""
        if not self.is_configured:
            raise GeminiMissingKeyError(
                "Gemini API key is not configured. "
                "Provide it via GEMINI_API_KEY environment variable or client constructor."
            )

    def __repr__(self) -> str:
        """Safe string representation that never exposes the API key."""
        return (
            f"GeminiClient(model='{self.model}', "
            f"configured={self.is_configured}, "
            f"timeout={self.timeout})"
        )

    def __str__(self) -> str:
        return self.__repr__()

    def close(self) -> None:
        """Close the underlying HTTP client session."""
        self._http_client.close()

    def __enter__(self) -> GeminiClient:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _calculate_backoff_delay(self, attempt: int, retry_after_header: Optional[str] = None) -> float:
        """
        Compute bounded exponential backoff delay with jitter.

        Rules:
        - If Retry-After header is provided and within (0, max_backoff], honor it.
        - Otherwise compute base exponential backoff: base_backoff * (2 ** (attempt - 1)).
        - If jitter is enabled, apply +/- 15% uniform randomization.
        - Clamp result strictly within [0.5, max_backoff].
        """
        if retry_after_header:
            try:
                retry_after_val = float(retry_after_header.strip())
                if 0.0 < retry_after_val <= self.max_backoff:
                    return retry_after_val
            except (ValueError, TypeError):
                pass

        base_delay = min(self.max_backoff, self.base_backoff * (2 ** (attempt - 1)))
        if self.jitter:
            factor = random.uniform(0.85, 1.15)
            jittered = base_delay * factor
            return max(0.5, min(self.max_backoff, round(jittered, 2)))
        return base_delay

    def generate_content(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        max_output_tokens: int = 4096,
        response_mime_type: Optional[str] = None,
    ) -> GeminiResponse:
        """
        Invoke gemini-3.5-flash-lite generateContent endpoint.

        Parameters
        ----------
        prompt : str
            The user prompt text.
        system_instruction : str, optional
            System prompt instructions.
        temperature : float, default 0.2
            Sampling temperature (0.0 to 2.0).
        max_output_tokens : int, default 4096
            Maximum tokens to generate.
        response_mime_type : str, optional
            e.g. 'application/json' to force structured JSON output.

        Returns
        -------
        GeminiResponse
            Structured response object containing text and usage metadata.

        Raises
        ------
        GeminiMissingKeyError
            If API key is missing.
        GeminiAuthenticationError
            If API key is invalid or unauthorized (HTTP 401/403).
        GeminiRateLimitError
            If rate limit or quota exceeded (HTTP 429).
        GeminiNetworkError
            If network connection fails, times out, or server errors occur.
        GeminiMalformedResponseError
            If response structure is invalid, empty, or blocked by safety filters.
        """
        self._ensure_configured()

        if not prompt or not prompt.strip():
            raise GeminiMalformedResponseError("Prompt cannot be empty.")

        url = f"{GEMINI_API_BASE_URL}/models/{self.model}:generateContent"

        # Build payload according to official Google Gemini REST specification
        payload: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": float(temperature),
                "maxOutputTokens": int(max_output_tokens),
            },
        }

        if response_mime_type:
            payload["generationConfig"]["responseMimeType"] = response_mime_type

        if system_instruction and system_instruction.strip():
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction.strip()}]
            }

        headers = {
            "x-goog-api-key": self._api_key,
            "Content-Type": "application/json",
        }

        logger.info(
            "Invoking Gemini model '%s' (prompt_len=%d, mime_type=%s)",
            self.model,
            len(prompt),
            response_mime_type or "text/plain",
        )

        max_attempts = 1 + self.max_retries
        for attempt in range(1, max_attempts + 1):
            try:
                response = self._http_client.post(
                    url,
                    headers=headers,
                    json=payload,
                )
            except (
                httpx.TimeoutException,
                httpx.ConnectError,
                httpx.NetworkError,
                httpx.ProtocolError,
                httpx.RemoteProtocolError,
            ) as exc:
                if attempt < max_attempts:
                    delay = self._calculate_backoff_delay(attempt)
                    logger.warning(
                        "Gemini transient network error (%s) on attempt %d/%d. Retrying in %.1fs...",
                        type(exc).__name__,
                        attempt,
                        max_attempts,
                        delay,
                    )
                    time.sleep(delay)
                    continue

                logger.error(
                    "Gemini network error during generateContent after %d attempt(s): %s",
                    attempt,
                    exc,
                )
                if isinstance(exc, httpx.TimeoutException):
                    raise GeminiTimeoutError(f"Network timeout connecting to Gemini API: {exc}") from exc
                raise GeminiNetworkError(f"Network error connecting to Gemini API: {exc}") from exc
            except Exception as exc:
                logger.error("Unexpected error during Gemini request: %s", exc)
                raise GeminiNetworkError(f"Failed to communicate with Gemini API: {exc}") from exc

            # Explicit HTTP 503 retry handling with bounded exponential backoff
            if response.status_code == 503 and attempt < max_attempts:
                retry_after_raw = response.headers.get("Retry-After")
                delay = self._calculate_backoff_delay(attempt, retry_after_header=retry_after_raw)

                logger.warning(
                    "Gemini 503 (high demand) on attempt %d/%d. Retrying in %.1fs...",
                    attempt,
                    max_attempts,
                    delay,
                )
                time.sleep(delay)
                continue

            break

        # Handle HTTP status codes explicitly
        if response.status_code in (401, 403):
            error_detail = self._extract_error_message(response)
            logger.error("Gemini authentication failed [%d]: %s", response.status_code, error_detail)
            raise GeminiAuthenticationError(
                f"Gemini API authentication failed [{response.status_code}]: {error_detail}",
                status_code=response.status_code,
            )

        if response.status_code == 429:
            error_detail = self._extract_error_message(response)
            retry_after = None
            if "Retry-After" in response.headers:
                try:
                    retry_after = float(response.headers["Retry-After"])
                except (ValueError, TypeError):
                    pass
            logger.error("Gemini rate limit exceeded (429): %s", error_detail)
            raise GeminiRateLimitError(
                f"Gemini API rate limit exceeded: {error_detail}",
                retry_after=retry_after,
            )

        if response.status_code >= 500:
            error_detail = self._extract_error_message(response)
            logger.error("Gemini server error [%d]: %s", response.status_code, error_detail)
            raise GeminiNetworkError(
                f"Gemini API server error [{response.status_code}]: {error_detail}",
                status_code=response.status_code,
            )

        if response.status_code != 200:
            error_detail = self._extract_error_message(response)
            logger.error("Gemini API call failed [%d]: %s", response.status_code, error_detail)
            raise GeminiClientError(
                f"Gemini API returned error [{response.status_code}]: {error_detail}",
                status_code=response.status_code,
            )

        # Parse JSON response
        try:
            data = response.json()
        except Exception as exc:
            logger.error("Gemini response is not valid JSON: %s", exc)
            raise GeminiMalformedResponseError(f"Failed to parse Gemini response as JSON: {exc}") from exc

        parsed_resp = self._parse_candidates_response(data)
        self.last_response = parsed_resp
        return parsed_resp

    def generate_json(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        max_output_tokens: int = 4096,
    ) -> dict[str, Any]:
        """
        Generate structured JSON output from Gemini and parse it into a Python dictionary.

        Raises
        ------
        GeminiMalformedResponseError
            If response text is missing or cannot be parsed as valid JSON.
        """
        response = self.generate_content(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
        )

        raw_text = response.text.strip()
        if not raw_text:
            raise GeminiMalformedResponseError("Gemini returned empty text for JSON generation.")

        # Clean potential markdown code fence markers if model wrapped JSON
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
        raw_text = raw_text.strip()

        try:
            parsed = json.loads(raw_text)
            if not isinstance(parsed, (dict, list)):
                raise GeminiMalformedResponseError(
                    f"Expected JSON object or array, got {type(parsed).__name__}."
                )
            return parsed
        except json.JSONDecodeError as exc:
            logger.error("Failed to decode JSON from Gemini output: %s", exc)
            raise GeminiMalformedResponseError(
                f"Gemini output is not valid JSON: {exc}. Response snippet: {raw_text[:200]}"
            ) from exc

    def _parse_candidates_response(self, data: dict[str, Any]) -> GeminiResponse:
        """Extract candidate content, finish reason, and usage statistics."""
        candidates = data.get("candidates")
        if not candidates or not isinstance(candidates, list) or len(candidates) == 0:
            prompt_feedback = data.get("promptFeedback", {})
            block_reason = prompt_feedback.get("blockReason")
            if block_reason:
                raise GeminiMalformedResponseError(
                    f"Gemini generation blocked: {block_reason}. Prompt feedback: {prompt_feedback}"
                )
            raise GeminiMalformedResponseError(
                f"Gemini response contains no candidates. Full response keys: {list(data.keys())}"
            )

        candidate = candidates[0]
        finish_reason = candidate.get("finishReason", "UNKNOWN")

        if finish_reason == "SAFETY":
            raise GeminiMalformedResponseError(
                "Gemini response candidate was blocked by safety filters (finishReason='SAFETY')."
            )

        content = candidate.get("content")
        if not content or not isinstance(content, dict):
            raise GeminiMalformedResponseError(
                f"Candidate missing content block (finishReason='{finish_reason}')."
            )

        parts = content.get("parts", [])
        if not parts or not isinstance(parts, list):
            raise GeminiMalformedResponseError(
                f"Candidate content parts list is empty (finishReason='{finish_reason}')."
            )

        # Concatenate text parts
        text_chunks = [p.get("text", "") for p in parts if isinstance(p, dict) and "text" in p]
        full_text = "".join(text_chunks)

        usage = data.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount", 0)
        candidates_tokens = usage.get("candidatesTokenCount", 0)
        total_tokens = usage.get("totalTokenCount", 0)

        return GeminiResponse(
            text=full_text,
            model=data.get("modelVersion", self.model),
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            candidates_tokens=candidates_tokens,
            total_tokens=total_tokens,
            raw_response=data,
        )

    @staticmethod
    def _extract_error_message(response: httpx.Response) -> str:
        """Safely extract error message without leaking sensitive request details."""
        try:
            data = response.json()
            if isinstance(data, dict):
                error_obj = data.get("error", {})
                if isinstance(error_obj, dict) and "message" in error_obj:
                    return str(error_obj["message"])
                return str(data)
        except Exception:
            pass
        return response.text[:200]

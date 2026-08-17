"""
AIVC Resilient Inference Client for OpenRouter & OpenAI Compatible APIs.

Provides a robust, production-grade LLM inference client featuring:
- Exponential backoff with full jitter
- HTTP Retry-After header parsing for 429 and 503 status codes
- Multi-model fallback configuration (models: [primary, fallback], allow_fallbacks: True)
- Centralized tool call and message payload sanitization to prevent HTTP 400 errors
- Explicit exception hierarchy (zero silent failures)
- OpenRouter and OpenAI API compatibility
"""

from __future__ import annotations

import http.client
import json
import logging
import os
import random
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Configure dedicated module logger
logger = logging.getLogger("aivc.inference_client")


# ---------------------------------------------------------------------------
# Exception Hierarchy
# ---------------------------------------------------------------------------

class InferenceError(Exception):
    """Base exception for all inference client errors."""
    pass


class InferenceAPIError(InferenceError):
    """Exception raised when an API returns an HTTP error response."""

    def __init__(
        self,
        status_code: int,
        response_body: str,
        message: Optional[str] = None,
    ):
        self.status_code = status_code
        self.response_body = response_body
        msg = message or f"API error with HTTP status {status_code}: {response_body}"
        super().__init__(msg)


class InferenceAuthError(InferenceAPIError):
    """Exception raised for authentication / authorization failures (HTTP 401, 403)."""
    pass


class InferenceBadRequestError(InferenceAPIError):
    """Exception raised for unrecoverable client errors (HTTP 400, 404)."""
    pass


class InferenceRateLimitError(InferenceAPIError):
    """Exception raised when rate limits (HTTP 429) persist beyond max retries."""
    pass


class InferenceTimeoutError(InferenceError):
    """Exception raised for network timeouts and connection drops."""
    pass


# ---------------------------------------------------------------------------
# Centralized Message & Tool Call Sanitizer
# ---------------------------------------------------------------------------

def sanitize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sanitize and normalize message history to prevent API validation errors.

    Guarantees:
    - Tool call arguments are valid serialized JSON strings
    - Tool call types are set to 'function'
    - Content fields for system, user, and tool messages are strings (not None)
    - All messages have valid role and content keys
    """
    sanitized: List[Dict[str, Any]] = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        m = dict(msg)
        role = m.get("role", "user")

        if role == "assistant":
            tool_calls = m.get("tool_calls")
            if tool_calls and isinstance(tool_calls, list):
                valid_tcs: List[Dict[str, Any]] = []
                for idx, tc in enumerate(tool_calls):
                    if not isinstance(tc, dict):
                        continue
                    tc_copy = dict(tc)
                    fn = dict(tc_copy.get("function", {}))

                    raw_args = fn.get("arguments", "{}")
                    if isinstance(raw_args, dict):
                        fn["arguments"] = json.dumps(raw_args, ensure_ascii=False)
                    elif isinstance(raw_args, str):
                        raw_args_trimmed = raw_args.strip()
                        if not raw_args_trimmed:
                            fn["arguments"] = "{}"
                        else:
                            try:
                                parsed = json.loads(raw_args_trimmed)
                                if isinstance(parsed, dict):
                                    fn["arguments"] = json.dumps(parsed, ensure_ascii=False)
                                else:
                                    fn["arguments"] = json.dumps({"value": parsed}, ensure_ascii=False)
                            except Exception:
                                fn["arguments"] = "{}"
                    else:
                        fn["arguments"] = "{}"

                    tc_copy["function"] = fn
                    if "type" not in tc_copy:
                        tc_copy["type"] = "function"
                    if "id" not in tc_copy:
                        tc_copy["id"] = f"call_sanitized_{idx}_{int(time.time()*1000)}"

                    valid_tcs.append(tc_copy)

                m["tool_calls"] = valid_tcs

            # Content in assistant message can be None or string
            if m.get("content") is None:
                m["content"] = ""
            elif not isinstance(m["content"], str):
                m["content"] = str(m["content"])

        elif role == "tool":
            if "content" not in m or m["content"] is None:
                m["content"] = ""
            elif not isinstance(m["content"], str):
                m["content"] = str(m["content"])

        elif role in ("system", "user"):
            if "content" not in m or m["content"] is None:
                m["content"] = ""
            elif not isinstance(m["content"], str):
                m["content"] = str(m["content"])

        sanitized.append(m)

    return sanitized


# ---------------------------------------------------------------------------
# Resilient Inference Client
# ---------------------------------------------------------------------------

class InferenceClient:
    """
    Production-grade LLM inference client for OpenRouter and OpenAI compatible endpoints.

    Features:
    - Full jitter exponential backoff
    - Automatic Retry-After header parsing for HTTP 429 and 503
    - Transparent multi-model fallback payload routing
    - Immediate exception raising for client errors (400, 401, 403, 404)
    - Zero silent failures: detailed logging and explicit exception typing
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "qwen/qwen3.7-flash",
        fallback_model: Optional[str] = "deepseek/deepseek-v4-flash-0731",
        max_retries: int = 5,
        base_delay: float = 1.5,
        max_delay: float = 30.0,
        timeout: float = 60.0,
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        app_referer: str = "https://github.com/aivc/aivc",
        app_title: str = "AIVC Benchmark Suite",
        headers: Optional[Dict[str, str]] = None,
    ):
        # Resolve API key from arguments or environment
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        self.default_model = default_model
        self.fallback_model = fallback_model
        self.max_retries = max(1, max_retries)
        self.base_delay = max(0.1, base_delay)
        self.max_delay = max(self.base_delay, max_delay)
        self.timeout = max(1.0, timeout)
        self.base_url = base_url
        self.app_referer = app_referer
        self.app_title = app_title
        self.custom_headers = headers or {}

    def _build_headers(self) -> Dict[str, str]:
        """Construct standard HTTP request headers."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.app_referer,
            "X-Title": self.app_title,
        }
        headers.update(self.custom_headers)
        return headers

    def _calculate_backoff(self, attempt: int, retry_after: Optional[str] = None) -> float:
        """
        Calculate backoff wait duration with full jitter and Retry-After header support.

        Formula: wait = min(max_delay, 2**(attempt-1) * base_delay + uniform(0, 1))
        Header extraction: wait = float(retry_after) + uniform(0.2, 1.0)
        """
        if retry_after:
            try:
                header_delay = float(retry_after.strip())
                jittered_delay = max(0.1, header_delay) + random.uniform(0.2, 1.0)
                logger.info(f"Respecting Retry-After header: {header_delay}s (+ jitter -> {jittered_delay:.2f}s)")
                return jittered_delay
            except (ValueError, TypeError):
                logger.debug(f"Could not parse Retry-After header '{retry_after}', using exponential backoff.")

        # Full jitter exponential backoff
        exp_factor = 2 ** (attempt - 1)
        raw_backoff = (exp_factor * self.base_delay) + random.uniform(0.0, 1.0)
        return min(self.max_delay, raw_backoff)

    def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        extra_body: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        fallback_model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Execute chat completion request with resilient retries, fallback routing, and validation.

        Args:
            messages: List of message dictionaries with roles (system, user, assistant, tool).
            tools: Optional tool/function schema definitions.
            max_tokens: Maximum completion tokens to generate.
            temperature: Sampling temperature (default 0.2).
            extra_body: Additional raw payload attributes.
            model: Primary model override (defaults to self.default_model).
            fallback_model: Fallback model override (defaults to self.fallback_model).
            **kwargs: Extra parameters passed to the request payload.

        Returns:
            Decoded JSON dictionary from the completion API.

        Raises:
            InferenceAuthError: When API key is missing, invalid, or forbidden (401/403).
            InferenceBadRequestError: When request format or payload is invalid (400/404).
            InferenceRateLimitError: When rate limit (429) persists after max_retries.
            InferenceAPIError: When server errors (500/502/503/504/529) persist.
            InferenceTimeoutError: When network connection drops or times out persistently.
        """
        if not self.api_key:
            raise InferenceAuthError(
                status_code=401,
                response_body="Missing API key",
                message="OPENROUTER_API_KEY / OPENAI_API_KEY is not set or empty. A valid API key is required.",
            )

        primary_model = model or self.default_model
        resolved_fallback = fallback_model if fallback_model is not None else self.fallback_model

        # Sanitize messages
        clean_messages = sanitize_messages(messages)

        # Build payload
        payload: Dict[str, Any] = {
            "model": primary_model,
            "messages": clean_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Multi-model fallback configuration for OpenRouter
        if resolved_fallback and resolved_fallback != primary_model:
            payload["models"] = [primary_model, resolved_fallback]
            payload["provider"] = {"allow_fallbacks": True}

        if tools:
            payload["tools"] = tools

        if extra_body:
            payload.update(extra_body)

        for k, v in kwargs.items():
            if v is not None:
                payload[k] = v

        encoded_data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = self._build_headers()

        last_exception: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            req = urllib.request.Request(
                self.base_url,
                data=encoded_data,
                headers=headers,
                method="POST",
            )

            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    status_code = resp.status if hasattr(resp, "status") else 200
                    body_bytes = resp.read()
                    body_text = body_bytes.decode("utf-8", errors="replace")

                    if 200 <= status_code < 300:
                        try:
                            parsed_json = json.loads(body_text)
                            return parsed_json
                        except json.JSONDecodeError as jde:
                            raise InferenceAPIError(
                                status_code=status_code,
                                response_body=body_text,
                                message=f"Failed to parse API JSON response: {jde}",
                            )

                    # Unexpected non-200 return inside urlopen block
                    raise InferenceAPIError(
                        status_code=status_code,
                        response_body=body_text,
                    )

            except urllib.error.HTTPError as http_err:
                status_code = http_err.code
                err_body = ""
                try:
                    err_body = http_err.read().decode("utf-8", errors="replace")
                except Exception:
                    err_body = str(http_err)

                retry_after = None
                if hasattr(http_err, "headers") and http_err.headers:
                    retry_after = http_err.headers.get("Retry-After") or http_err.headers.get("retry-after")

                # 1. Immediate failure codes (no retries)
                if status_code in (401, 403):
                    msg = f"Authentication/Authorization failure (HTTP {status_code}): {err_body}"
                    logger.error(f"[InferenceClient] {msg}")
                    raise InferenceAuthError(status_code=status_code, response_body=err_body, message=msg)

                if status_code in (400, 404):
                    if status_code == 400 and ("function.arguments" in err_body or "json" in err_body.lower() or "arguments" in err_body.lower()) and attempt < self.max_retries:
                        logger.warning(f"[InferenceClient] Detected provider argument format error on attempt {attempt}. Retrying with aggressive JSON sanitization...")
                        for m in payload.get("messages", []):
                            if m.get("role") == "assistant" and m.get("tool_calls"):
                                for tc in m["tool_calls"]:
                                    if "function" in tc and isinstance(tc["function"], dict):
                                        tc["function"]["arguments"] = "{}"
                        encoded_data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                        time.sleep(1.0)
                        continue

                    msg = f"Invalid client request (HTTP {status_code}): {err_body}"
                    logger.error(f"[InferenceClient] {msg}")
                    raise InferenceBadRequestError(status_code=status_code, response_body=err_body, message=msg)

                # 2. Retryable HTTP error codes (429, 500, 502, 503, 504, 529)
                last_exception = http_err
                if attempt == self.max_retries:
                    if status_code == 429:
                        msg = f"HTTP 429 Rate Limit exceeded after {self.max_retries} attempts: {err_body}"
                        logger.error(f"[InferenceClient] {msg}")
                        raise InferenceRateLimitError(status_code=429, response_body=err_body, message=msg)
                    else:
                        msg = f"HTTP {status_code} server error after {self.max_retries} attempts: {err_body}"
                        logger.error(f"[InferenceClient] {msg}")
                        raise InferenceAPIError(status_code=status_code, response_body=err_body, message=msg)

                wait_sec = self._calculate_backoff(attempt, retry_after=retry_after)
                logger.warning(
                    f"[InferenceClient] HTTP {status_code} (Attempt {attempt}/{self.max_retries}). "
                    f"Retrying in {wait_sec:.2f}s... Error: {err_body[:200]}"
                )
                time.sleep(wait_sec)

            except (
                urllib.error.URLError,
                socket.timeout,
                TimeoutError,
                http.client.HTTPException,
                ConnectionResetError,
                OSError,
            ) as net_err:
                last_exception = net_err
                if attempt == self.max_retries:
                    msg = f"Network connection failed after {self.max_retries} attempts: {net_err}"
                    logger.error(f"[InferenceClient] {msg}")
                    raise InferenceTimeoutError(msg) from net_err

                wait_sec = self._calculate_backoff(attempt)
                logger.warning(
                    f"[InferenceClient] Network/Timeout error (Attempt {attempt}/{self.max_retries}): {net_err}. "
                    f"Retrying in {wait_sec:.2f}s..."
                )
                time.sleep(wait_sec)

            except InferenceError:
                # Re-raise internal inference errors directly
                raise

            except Exception as unk_err:
                last_exception = unk_err
                if attempt == self.max_retries:
                    msg = f"Unexpected error during inference after {self.max_retries} attempts: {unk_err}"
                    logger.error(f"[InferenceClient] {msg}")
                    raise InferenceError(msg) from unk_err

                wait_sec = self._calculate_backoff(attempt)
                logger.warning(
                    f"[InferenceClient] Unexpected error (Attempt {attempt}/{self.max_retries}): {unk_err}. "
                    f"Retrying in {wait_sec:.2f}s..."
                )
                time.sleep(wait_sec)

        # Fallback if loop finishes unexpectedly
        raise InferenceError(f"Inference execution failed after {self.max_retries} retries: {last_exception}")


# ---------------------------------------------------------------------------
# Backward-Compatibility Alias
# ---------------------------------------------------------------------------

OpenRouterClient = InferenceClient

__all__ = [
    "InferenceClient",
    "OpenRouterClient",
    "InferenceError",
    "InferenceAPIError",
    "InferenceAuthError",
    "InferenceBadRequestError",
    "InferenceRateLimitError",
    "InferenceTimeoutError",
    "sanitize_messages",
]

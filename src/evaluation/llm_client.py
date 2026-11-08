"""Model-independent provider interface with Gemini and optional OpenAI clients."""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Protocol
from urllib.parse import quote

from src.evaluation.evaluation_prompts import (
    SYSTEM_PROMPT,
    build_candidate_prompt,
    build_openai_messages,
    build_proposal_messages,
    response_schema,
    proposal_response_schema,
)
from src.evaluation.evaluation_schema import (
    gemini_proposal_response_schema,
    gemini_response_schema,
)


GEMINI_FREE_TIER_MODEL_ALLOWLIST = frozenset({"gemini-3.5-flash-lite"})
GEMINI_RATE_WINDOW_SECONDS = 60.0


class LLMClientError(RuntimeError):
    """Provider request failure with a safe, user-facing message."""


class LLMQuotaExceededError(LLMClientError):
    """Provider quota/rate exhaustion; callers must stop safely and resume later."""


def _is_gemini_quota_error(status: int, response_error: str) -> bool:
    return (
        status == 429
        or "resource_exhausted" in response_error
        or "quota" in response_error
        or "rate limit" in response_error
    )


def _retry_delay(headers: Any, attempts: int) -> float:
    retry_after = headers.get("Retry-After")
    if retry_after:
        try:
            delay = float(retry_after)
            if math.isfinite(delay):
                return max(0.0, delay)
        except ValueError:
            try:
                retry_time = parsedate_to_datetime(retry_after)
                if retry_time.tzinfo is None:
                    retry_time = retry_time.replace(tzinfo=timezone.utc)
                return max(0.0, (retry_time - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                pass
    return min(2**attempts, 30)


def _sanitize_gemini_error_value(value: Any, api_key: str | None) -> str:
    if not isinstance(value, str):
        return ""
    sanitized = value.strip()
    if api_key:
        sanitized = sanitized.replace(api_key, "[REDACTED]")
    sanitized = re.sub(r"(?i)\bBearer\s+\S+", "Bearer [REDACTED]", sanitized)
    sanitized = re.sub(r"\bAIza[0-9A-Za-z_-]{20,}\b", "[REDACTED]", sanitized)
    return sanitized[:1000]


class _GeminiGenerationLimiter:
    """Track generation requests and input tokens in rolling time windows."""

    def __init__(
        self,
        *,
        max_rpm: int,
        max_input_tpm: int,
        sleeper: Any,
        clock: Any,
    ) -> None:
        if max_rpm < 1 or max_input_tpm < 1:
            raise ValueError("Gemini RPM and input TPM limits must be positive")
        self._max_rpm = max_rpm
        self._max_input_tpm = max_input_tpm
        self._sleep = sleeper
        self._clock = clock
        self._generation_requests: deque[float] = deque()
        self._input_tokens: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def acquire(self, input_tokens: int) -> None:
        """Wait for capacity and account for exactly one generation attempt."""
        if isinstance(input_tokens, bool) or not isinstance(input_tokens, int) or input_tokens < 0:
            raise LLMClientError("Gemini token count was not a non-negative integer")
        if input_tokens > self._max_input_tpm:
            raise LLMQuotaExceededError(
                f"Gemini proposal requires {input_tokens} input tokens, exceeding the "
                f"configured per-request input-token budget of {self._max_input_tpm}; "
                "no generation request was submitted."
            )

        while True:
            with self._lock:
                now = self._clock()
                self._discard_expired(now)
                wait_seconds = self._wait_seconds(now, input_tokens)
                if wait_seconds <= 0:
                    self._generation_requests.append(now)
                    self._input_tokens.append((now, input_tokens))
                    return
            self._sleep(wait_seconds)

    def _discard_expired(self, now: float) -> None:
        cutoff = now - GEMINI_RATE_WINDOW_SECONDS
        while self._generation_requests and self._generation_requests[0] <= cutoff:
            self._generation_requests.popleft()
        while self._input_tokens and self._input_tokens[0][0] <= cutoff:
            self._input_tokens.popleft()

    def _wait_seconds(self, now: float, input_tokens: int) -> float:
        waits = [0.0]
        if len(self._generation_requests) >= self._max_rpm:
            waits.append(
                self._generation_requests[0] + GEMINI_RATE_WINDOW_SECONDS - now
            )
        current_tokens = sum(count for _, count in self._input_tokens)
        if current_tokens + input_tokens > self._max_input_tpm:
            remaining = current_tokens
            for created_at, count in self._input_tokens:
                remaining -= count
                if remaining + input_tokens <= self._max_input_tpm:
                    waits.append(created_at + GEMINI_RATE_WINDOW_SECONDS - now)
                    break
        return max(waits)


class LLMProvider(Protocol):
    """Minimal interface consumed by the evaluator."""

    provider_name: str
    model_name: str

    def evaluate(self, candidate: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Return raw JSON text and non-secret response metadata."""

    def evaluate_proposal(self, proposal: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Return a structured assembled-promo response and safe metadata."""

    def test_connection(self) -> dict[str, Any]:
        """Verify API credentials and availability of the selected model."""


@dataclass(frozen=True)
class LLMConfig:
    """Provider configuration loaded from environment variables."""

    provider: str
    model: str
    api_key: str | None
    batch_size: int
    max_retries: int
    timeout_seconds: int
    free_tier_confirmed: bool = False
    gemini_max_rpm: int = 10
    gemini_max_input_tpm: int = 200_000

    @classmethod
    def from_environment(cls) -> LLMConfig:
        """Read configuration without logging or exposing the API key."""
        provider = os.getenv("PROMO_LLM_PROVIDER", "gemini").strip().lower()
        default_model = {
            "gemini": "gemini-3.5-flash-lite",
            "openai": "gpt-4o-mini",
            "claude": "claude-sonnet-5-5",
            "grok": "grok-4.7",
        }.get(provider, "gpt-4o-mini")
        model = os.getenv("PROMO_LLM_MODEL", default_model).strip()
        batch_raw = os.getenv("PROMO_EVAL_BATCH_SIZE", "1")
        retries_raw = os.getenv("PROMO_LLM_MAX_RETRIES", "3")
        timeout_raw = os.getenv("PROMO_LLM_TIMEOUT_SECONDS", "90")
        gemini_rpm_raw = os.getenv("PROMO_GEMINI_MAX_RPM", "10")
        gemini_input_tpm_raw = os.getenv("PROMO_GEMINI_MAX_INPUT_TPM", "200000")
        try:
            batch_size = int(batch_raw)
            max_retries = int(retries_raw)
            timeout_seconds = int(timeout_raw)
            gemini_max_rpm = int(gemini_rpm_raw)
            gemini_max_input_tpm = int(gemini_input_tpm_raw)
        except ValueError as error:
            raise ValueError(
                "Batch size, retry count, timeout, and Gemini rate limits must be integers"
            ) from error
        if batch_size < 1 or max_retries < 0 or timeout_seconds < 1:
            raise ValueError("Batch size/timeout must be positive and retries cannot be negative")
        if gemini_max_rpm < 1 or gemini_max_input_tpm < 1:
            raise ValueError("Gemini RPM and input TPM limits must be positive")
        return cls(
            provider=provider,
            model=model,
            api_key=os.getenv(
                {
                    "gemini": "GEMINI_API_KEY",
                    "openai": "OPENAI_API_KEY",
                    "claude": "ANTHROPIC_API_KEY",
                    "grok": "XAI_API_KEY",
                }.get(provider, "")
            ),
            batch_size=batch_size,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            free_tier_confirmed=os.getenv("PROMO_GEMINI_FREE_TIER_CONFIRMED", "").strip()
            == "YES",
            gemini_max_rpm=gemini_max_rpm,
            gemini_max_input_tpm=gemini_max_input_tpm,
        )


class OpenAIChatCompletionsClient:
    """OpenAI provider using the standard-library HTTP client."""

    provider_name = "openai"
    endpoint = "https://api.openai.com/v1/chat/completions"

    def __init__(
        self,
        config: LLMConfig,
        *,
        sleeper: Any = time.sleep,
        opener: Any = urllib.request.urlopen,
    ) -> None:
        if config.provider != self.provider_name:
            raise ValueError(f"Unsupported provider: {config.provider}")
        if not config.model:
            raise ValueError("PROMO_LLM_MODEL must not be empty")
        if not config.api_key:
            raise ValueError(
                "Missing OPENAI_API_KEY. Set it in the process environment or run with --dry-run."
            )
        self.config = config
        self.model_name = config.model
        self._sleep = sleeper
        self._open = opener

    def evaluate(self, candidate: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Request one candidate evaluation with bounded retries for transient errors."""
        return self._evaluate(candidate, proposal_mode=False)

    def evaluate_proposal(self, proposal: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Request one assembled-promo evaluation with bounded retries."""
        return self._evaluate(proposal, proposal_mode=True)

    def test_connection(self) -> dict[str, Any]:
        request = urllib.request.Request(
            f"https://api.openai.com/v1/models/{quote(self.config.model, safe='')}",
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            method="GET",
        )
        try:
            with self._open(request, timeout=self.config.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise _provider_http_error("OpenAI", error) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise LLMClientError(
                f"OpenAI connection test failed: {type(error).__name__}"
            ) from error
        if not isinstance(body, dict):
            raise LLMClientError("OpenAI returned a malformed model response.")
        return {"provider": self.provider_name, "model": body.get("id", self.config.model)}

    def _evaluate(
        self,
        evaluation_input: dict[str, Any],
        *,
        proposal_mode: bool,
    ) -> tuple[str, dict[str, Any]]:
        messages = (
            build_proposal_messages(evaluation_input)
            if proposal_mode
            else build_openai_messages(evaluation_input)
        )
        payload = {
            "model": self.config.model,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": (
                    proposal_response_schema() if proposal_mode else response_schema()
                ),
            },
            "temperature": 0,
            "max_completion_tokens": 1400,
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        attempts = 0
        while True:
            try:
                with self._open(request, timeout=self.config.timeout_seconds) as response:
                    body = json.loads(response.read().decode("utf-8"))
                if not isinstance(body, dict):
                    raise LLMClientError("OpenAI returned a malformed completion response.")
                choices = body.get("choices")
                if (
                    not isinstance(choices, list)
                    or not choices
                    or not isinstance(choices[0], dict)
                ):
                    raise LLMClientError("Provider response did not contain a completion choice")
                content = _text_from_chat_response(
                    choices[0].get("message"), "OpenAI"
                )
                metadata = {
                    "provider": self.provider_name,
                    "requested_model": self.config.model,
                    "returned_model": body.get("model"),
                    "response_id": body.get("id"),
                    "usage": body.get("usage"),
                }
                return content, metadata
            except urllib.error.HTTPError as error:
                status = error.code
                if status == 402:
                    raise LLMQuotaExceededError(
                        "OpenAI reports insufficient quota or credits (HTTP 402); "
                        "stopping evaluation. Completed evaluations are preserved for resume."
                    ) from error
                if status == 429:
                    error_body = _http_error_body(error)
                    if _is_insufficient_quota(error_body):
                        raise LLMQuotaExceededError(
                            "OpenAI reports insufficient quota or credits (HTTP 429); "
                            "stopping evaluation. Completed evaluations are preserved for resume."
                        ) from error
                    if attempts >= self.config.max_retries:
                        raise LLMQuotaExceededError(
                            "OpenAI rate limit retries were exhausted (HTTP 429); "
                            "stopping evaluation. Completed evaluations are preserved for resume."
                        ) from error
                retryable = status in {408, 409} or 500 <= status <= 599
                retryable = retryable or status == 429
                if not retryable or attempts >= self.config.max_retries:
                    raise LLMClientError(
                        _provider_error_message("OpenAI", status)
                        + f" after {attempts + 1} attempt(s)"
                    ) from error
                retry_after = error.headers.get("Retry-After")
                delay = min(float(retry_after), 30.0) if retry_after else min(2**attempts, 30)
                attempts += 1
                self._sleep(delay)
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                if attempts >= self.config.max_retries:
                    raise LLMClientError(
                        f"OpenAI request failed after {attempts + 1} attempt(s): {type(error).__name__}"
                    ) from error
                self._sleep(min(2**attempts, 30))
                attempts += 1
            except json.JSONDecodeError as error:
                raise LLMClientError("Provider returned invalid JSON") from error


def _provider_error_message(provider: str, status: int) -> str:
    if status in {401, 403}:
        return f"{provider} rejected the API key or its permissions (HTTP {status})."
    if status == 404:
        return f"{provider} does not recognize the selected model (HTTP 404)."
    if status == 400:
        return (
            f"{provider} rejected the request or structured-output schema (HTTP 400); "
            "the model may not support structured output."
        )
    if status >= 500:
        return f"{provider} is unavailable (HTTP {status})."
    return f"{provider} request failed (HTTP {status})."


def _http_error_body(error: urllib.error.HTTPError) -> str:
    try:
        body = json.loads(error.read().decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return ""
    return json.dumps(body, ensure_ascii=False).casefold()


def _is_insufficient_quota(error_body: str) -> bool:
    return any(
        phrase in error_body
        for phrase in ("insufficient_quota", "billing_hard_limit", "quota exceeded", "insufficient credits")
    )


def _provider_http_error(provider: str, error: urllib.error.HTTPError) -> LLMClientError:
    if error.code == 402:
        return LLMQuotaExceededError(
            f"{provider} reports insufficient quota or credits (HTTP 402); "
            "stopping evaluation. Completed evaluations are preserved for resume."
        )
    if error.code == 429:
        return LLMQuotaExceededError(
            f"{provider} rate limit reached (HTTP 429); stopping evaluation. "
            "Completed evaluations are preserved for resume."
        )
    return LLMClientError(_provider_error_message(provider, error.code))


def _text_from_chat_response(message: Any, provider: str) -> str:
    if not isinstance(message, dict):
        raise LLMClientError(f"{provider} response did not contain a message object.")
    if message.get("refusal"):
        raise LLMClientError(f"{provider} refused the structured evaluation request.")
    content = message.get("content")
    if isinstance(content, str) and content:
        return content
    if isinstance(content, list):
        text = "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
        if text:
            return text
    raise LLMClientError(f"{provider} returned an empty or non-text completion.")


def _validate_provider_config(config: LLMConfig, provider: str, key_name: str) -> None:
    if config.provider != provider:
        raise ValueError(f"Unsupported provider: {config.provider}")
    if not config.model.strip():
        raise ValueError(f"A {provider} model ID is required.")
    if not config.api_key:
        raise ValueError(
            f"Missing {key_name}. Configure it locally or enter it in the dashboard."
        )


def _inline_schema(value: Any) -> Any:
    """Inline local JSON Schema references for providers with limited ref support."""
    definitions = value.get("$defs", {}) if isinstance(value, dict) else {}

    def visit(item: Any) -> Any:
        if isinstance(item, dict):
            reference = item.get("$ref")
            if isinstance(reference, str) and reference.startswith("#/$defs/"):
                name = reference.rsplit("/", 1)[-1]
                if name not in definitions:
                    raise ValueError(f"Unknown JSON Schema reference: {reference}")
                return visit(definitions[name])
            return {
                key: visit(child)
                for key, child in item.items()
                if key not in {"$defs", "additionalProperties"}
            }
        if isinstance(item, list):
            return [visit(child) for child in item]
        return item

    return visit(value)


class AnthropicMessagesProvider:
    """Anthropic Messages API adapter with documented structured JSON output."""

    provider_name = "claude"
    endpoint = "https://api.anthropic.com/v1/messages"
    model_endpoint = "https://api.anthropic.com/v1/models/{model}"

    def __init__(
        self,
        config: LLMConfig,
        *,
        sleeper: Any = time.sleep,
        opener: Any = urllib.request.urlopen,
    ) -> None:
        _validate_provider_config(config, self.provider_name, "ANTHROPIC_API_KEY")
        self.config = config
        self.model_name = config.model
        self._sleep = sleeper
        self._open = opener

    def evaluate(self, candidate: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        return self._evaluate(candidate, proposal_mode=False)

    def evaluate_proposal(self, proposal: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        return self._evaluate(proposal, proposal_mode=True)

    def test_connection(self) -> dict[str, Any]:
        request = urllib.request.Request(
            self.model_endpoint.format(model=quote(self.config.model, safe="")),
            headers=self._headers(),
            method="GET",
        )
        body = self._request(request)
        return {"provider": self.provider_name, "model": body.get("id", self.config.model)}

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.config.api_key or "",
            "anthropic-version": "2026-10-21",
            "Content-Type": "application/json",
        }

    def _evaluate(
        self, value: dict[str, Any], *, proposal_mode: bool
    ) -> tuple[str, dict[str, Any]]:
        messages = (
            build_proposal_messages(value)
            if proposal_mode
            else build_openai_messages(value)
        )
        schema = (
            proposal_response_schema() if proposal_mode else response_schema()
        )["schema"]
        payload = {
            "model": self.config.model,
            "max_tokens": 1400,
            "system": messages[0]["content"],
            "messages": [{"role": "user", "content": messages[1]["content"]}],
            "output_config": {
                "format": {"type": "json_schema", "schema": _inline_schema(schema)}
            },
            "temperature": 0,
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        body = self._request(request)
        blocks = body.get("content")
        text = "".join(
            block["text"]
            for block in blocks or []
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        )
        if not text:
            raise LLMClientError("Claude returned no text block for the evaluation.")
        return text, {
            "provider": self.provider_name,
            "requested_model": self.config.model,
            "returned_model": body.get("model"),
            "response_id": body.get("id"),
            "usage": body.get("usage"),
        }

    def _request(self, request: urllib.request.Request) -> dict[str, Any]:
        attempts = 0
        while True:
            try:
                with self._open(request, timeout=self.config.timeout_seconds) as response:
                    body = json.loads(response.read().decode("utf-8"))
                if not isinstance(body, dict):
                    raise LLMClientError("Claude returned a malformed response object.")
                return body
            except urllib.error.HTTPError as error:
                if error.code in {429, 402}:
                    raise _provider_http_error("Claude", error) from error
                retryable = error.code in {408, 409} or 500 <= error.code <= 599
                if not retryable or attempts >= self.config.max_retries:
                    raise LLMClientError(
                        _provider_error_message("Claude", error.code)
                        + f" after {attempts + 1} attempt(s)"
                    ) from error
                attempts += 1
                self._sleep(min(2 ** (attempts - 1), 30))
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                if attempts >= self.config.max_retries:
                    raise LLMClientError(
                        f"Claude request failed after {attempts + 1} attempt(s): "
                        f"{type(error).__name__}"
                    ) from error
                attempts += 1
                self._sleep(min(2 ** (attempts - 1), 30))
            except json.JSONDecodeError as error:
                raise LLMClientError("Claude returned invalid JSON.") from error


class XAIChatCompletionsProvider:
    """xAI adapter using its documented structured Chat Completions API."""

    provider_name = "grok"
    endpoint = "https://api.x.ai/v1/chat/completions"

    def __init__(
        self,
        config: LLMConfig,
        *,
        sleeper: Any = time.sleep,
        opener: Any = urllib.request.urlopen,
    ) -> None:
        _validate_provider_config(config, self.provider_name, "XAI_API_KEY")
        self.config = config
        self.model_name = config.model
        self._sleep = sleeper
        self._open = opener

    def evaluate(self, candidate: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        return self._evaluate(candidate, proposal_mode=False)

    def evaluate_proposal(self, proposal: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        return self._evaluate(proposal, proposal_mode=True)

    def test_connection(self) -> dict[str, Any]:
        request = urllib.request.Request(
            f"https://api.x.ai/v1/models/{quote(self.config.model, safe='')}",
            headers=self._headers(),
            method="GET",
        )
        body = self._request(request)
        return {"provider": self.provider_name, "model": body.get("id", self.config.model)}

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def _evaluate(
        self, value: dict[str, Any], *, proposal_mode: bool
    ) -> tuple[str, dict[str, Any]]:
        messages = (
            build_proposal_messages(value)
            if proposal_mode
            else build_openai_messages(value)
        )
        schema = proposal_response_schema() if proposal_mode else response_schema()
        payload = {
            "model": self.config.model,
            "messages": messages,
            "response_format": {"type": "json_schema", "json_schema": schema},
            "temperature": 0,
            "max_completion_tokens": 1400,
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        body = self._request(request)
        choices = body.get("choices")
        if (
            not isinstance(choices, list)
            or not choices
            or not isinstance(choices[0], dict)
        ):
            raise LLMClientError("Grok response did not contain a completion choice.")
        text = _text_from_chat_response(choices[0].get("message"), "Grok")
        return text, {
            "provider": self.provider_name,
            "requested_model": self.config.model,
            "returned_model": body.get("model"),
            "response_id": body.get("id"),
            "usage": body.get("usage"),
        }

    def _request(self, request: urllib.request.Request) -> dict[str, Any]:
        attempts = 0
        while True:
            try:
                with self._open(request, timeout=self.config.timeout_seconds) as response:
                    body = json.loads(response.read().decode("utf-8"))
                if not isinstance(body, dict):
                    raise LLMClientError("Grok returned a malformed response object.")
                return body
            except urllib.error.HTTPError as error:
                if error.code in {429, 402}:
                    raise _provider_http_error("Grok", error) from error
                retryable = error.code in {408, 409} or 500 <= error.code <= 599
                if not retryable or attempts >= self.config.max_retries:
                    raise LLMClientError(
                        _provider_error_message("Grok", error.code)
                        + f" after {attempts + 1} attempt(s)"
                    ) from error
                attempts += 1
                self._sleep(min(2 ** (attempts - 1), 30))
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                if attempts >= self.config.max_retries:
                    raise LLMClientError(
                        f"Grok request failed after {attempts + 1} attempt(s): "
                        f"{type(error).__name__}"
                    ) from error
                attempts += 1
                self._sleep(min(2 ** (attempts - 1), 30))
            except json.JSONDecodeError as error:
                raise LLMClientError("Grok returned invalid JSON.") from error


class GeminiProvider:
    """Gemini Developer API client using the documented REST generateContent endpoint."""

    provider_name = "gemini"
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    count_tokens_endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/{model}:countTokens"
    )

    def __init__(
        self,
        config: LLMConfig,
        *,
        sleeper: Any = time.sleep,
        opener: Any = urllib.request.urlopen,
        clock: Any = time.monotonic,
    ) -> None:
        if config.provider != self.provider_name:
            raise ValueError(f"Unsupported provider: {config.provider}")
        if not config.model:
            raise ValueError("PROMO_LLM_MODEL must not be empty")
        if config.model not in GEMINI_FREE_TIER_MODEL_ALLOWLIST:
            raise ValueError(
                f"Model {config.model!r} is not in this project's verified Gemini "
                f"free-tier allowlist: {sorted(GEMINI_FREE_TIER_MODEL_ALLOWLIST)}"
            )
        if not config.api_key:
            raise ValueError(
                "Missing GEMINI_API_KEY. Create a key in Google AI Studio and run with "
                "--dry-run until ready."
            )
        if not config.free_tier_confirmed:
            raise ValueError(
                "Free-tier confirmation required. Verify the Google AI Studio project's "
                "free-tier eligibility and no-billing configuration, then set "
                "PROMO_GEMINI_FREE_TIER_CONFIRMED=YES."
            )
        self.config = config
        self.model_name = config.model
        self._sleep = sleeper
        self._open = opener
        self._clock = clock
        self._generation_limiter = _GeminiGenerationLimiter(
            max_rpm=config.gemini_max_rpm,
            max_input_tpm=config.gemini_max_input_tpm,
            sleeper=sleeper,
            clock=clock,
        )
        self._generation_lock = threading.Lock()

    def evaluate(self, candidate: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Request one candidate response and stop immediately on quota exhaustion."""
        return self._evaluate(candidate, proposal_mode=False)

    def evaluate_proposal(self, proposal: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Request one assembled-promo response with exact segment-aware evidence."""
        return self._evaluate(proposal, proposal_mode=True)

    def test_connection(self) -> dict[str, Any]:
        if not self.config.free_tier_confirmed:
            raise ValueError(
                "Gemini connection testing requires explicit free-tier confirmation."
            )
        request = urllib.request.Request(
            "https://generativelanguage.googleapis.com/v1beta/models/"
            + quote(self.config.model, safe=""),
            headers={"x-goog-api-key": self.config.api_key or ""},
            method="GET",
        )
        try:
            with self._open(request, timeout=self.config.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise _provider_http_error("Gemini", error) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise LLMClientError(
                f"Gemini connection test failed: {type(error).__name__}"
            ) from error
        if not isinstance(body, dict):
            raise LLMClientError("Gemini returned a malformed model response.")
        return {
            "provider": self.provider_name,
            "model": body.get("name", self.config.model),
        }

    def _evaluate(
        self,
        evaluation_input: dict[str, Any],
        *,
        proposal_mode: bool,
    ) -> tuple[str, dict[str, Any]]:
        if proposal_mode:
            proposal_messages = build_proposal_messages(evaluation_input)
            system_prompt = proposal_messages[0]["content"]
            prompt = proposal_messages[1]["content"]
            output_schema = gemini_proposal_response_schema()
        else:
            prompt = build_candidate_prompt(evaluation_input)
            system_prompt = SYSTEM_PROMPT
            output_schema = gemini_response_schema()
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseJsonSchema": output_schema,
                "temperature": 0,
                "maxOutputTokens": 1400,
            },
        }
        input_token_count = self._count_input_tokens(payload)
        url = self.endpoint.format(model=self.config.model)
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "x-goog-api-key": self.config.api_key or "",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        attempts = 0
        while True:
            try:
                with self._generation_lock:
                    self._generation_limiter.acquire(input_token_count)
                    with self._open(request, timeout=self.config.timeout_seconds) as response:
                        body = json.loads(response.read().decode("utf-8"))
                    if not isinstance(body, dict):
                        raise LLMClientError("Gemini returned a malformed response object.")
                    candidates = body.get("candidates")
                    if (
                        not isinstance(candidates, list)
                        or not candidates
                        or not isinstance(candidates[0], dict)
                    ):
                        feedback = body.get("promptFeedback", {})
                        block_reason = (
                            feedback.get("blockReason")
                            if isinstance(feedback, dict)
                            else None
                        )
                        raise LLMClientError(
                            "Gemini returned no candidate content "
                            f"(block reason: {block_reason or 'unspecified'})"
                        )
                model_candidate = candidates[0]
                finish_reason = model_candidate.get("finishReason")
                content = model_candidate.get("content", {})
                parts = content.get("parts", []) if isinstance(content, dict) else []
                text = "".join(
                    part["text"]
                    for part in parts
                    if isinstance(part, dict) and isinstance(part.get("text"), str)
                )
                if not text:
                    raise LLMClientError(
                        f"Gemini returned empty content (finish reason: {finish_reason or 'unspecified'})"
                    )
                metadata = {
                    "provider": self.provider_name,
                    "requested_model": self.config.model,
                    "returned_model": body.get("modelVersion"),
                    "response_id": body.get("responseId"),
                    "usage": body.get("usageMetadata"),
                    "finish_reason": finish_reason,
                    "counted_input_tokens": input_token_count,
                }
                return text, metadata
            except urllib.error.HTTPError as error:
                status = error.code
                response_error = ""
                try:
                    error_body = json.loads(error.read().decode("utf-8"))
                    error_object = error_body.get("error", {})
                    response_error = " ".join(
                        str(error_object.get(field, ""))
                        for field in ("status", "message", "reason")
                    ).casefold()
                except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
                    pass
                if _is_gemini_quota_error(status, response_error):
                    raise LLMQuotaExceededError(
                        f"Gemini free-tier quota/rate limit exhausted (HTTP {status}); "
                        "stopping now. Completed evaluations are preserved for resume."
                    ) from error
                retryable = status in {408, 409} or 500 <= status <= 599
                if not retryable or attempts >= self.config.max_retries:
                    raise LLMClientError(
                        f"Gemini request failed with HTTP {status} after {attempts + 1} attempt(s)"
                    ) from error
                delay = _retry_delay(error.headers, attempts)
                attempts += 1
                self._sleep(delay)
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                if attempts >= self.config.max_retries:
                    raise LLMClientError(
                        f"Gemini request failed after {attempts + 1} attempt(s): {type(error).__name__}"
                    ) from error
                self._sleep(min(2**attempts, 30))
                attempts += 1
            except json.JSONDecodeError as error:
                raise LLMClientError("Gemini returned invalid JSON") from error

    def _count_input_tokens(self, generation_payload: dict[str, Any]) -> int:
        """Count the exact generation request input before any generation attempt."""
        request = urllib.request.Request(
            self.count_tokens_endpoint.format(model=self.config.model),
            data=json.dumps(
                {
                    "generateContentRequest": {
                        **generation_payload,
                        "model": f"models/{self.config.model}",
                    }
                },
                ensure_ascii=False,
            ).encode("utf-8"),
            headers={
                "x-goog-api-key": self.config.api_key or "",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        attempts = 0
        while True:
            try:
                with self._open(request, timeout=self.config.timeout_seconds) as response:
                    body = json.loads(response.read().decode("utf-8"))
                token_count = body.get("totalTokens")
                if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 0:
                    raise LLMClientError(
                        "Gemini countTokens response did not contain a valid totalTokens value; "
                        "generation was not submitted."
                    )
                return token_count
            except urllib.error.HTTPError as error:
                status = error.code
                google_error: dict[str, Any] = {}
                try:
                    error_body = json.loads(error.read().decode("utf-8"))
                    if isinstance(error_body, dict) and isinstance(error_body.get("error"), dict):
                        google_error = error_body["error"]
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
                response_error = " ".join(
                    str(google_error.get(field, ""))
                    for field in ("status", "message", "reason")
                ).casefold()
                google_status = _sanitize_gemini_error_value(
                    google_error.get("status"), self.config.api_key
                )
                google_message = _sanitize_gemini_error_value(
                    google_error.get("message"), self.config.api_key
                )
                details: list[str] = []
                if google_status:
                    details.append(f"Google status: {google_status}")
                if google_message:
                    details.append(f"Google message: {google_message}")
                detail_suffix = f"; {'; '.join(details)}" if details else ""
                if _is_gemini_quota_error(status, response_error):
                    raise LLMQuotaExceededError(
                        f"Gemini countTokens quota/rate limit exhausted (HTTP {status}); "
                        f"generation was not submitted{detail_suffix}."
                    ) from error
                retryable = status in {408, 409} or 500 <= status <= 599
                if not retryable or attempts >= self.config.max_retries:
                    raise LLMClientError(
                        f"Gemini countTokens request failed with HTTP {status} after "
                        f"{attempts + 1} attempt(s){detail_suffix}; "
                        "generation was not submitted."
                    ) from error
                delay = _retry_delay(error.headers, attempts)
                attempts += 1
                self._sleep(delay)
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                if attempts >= self.config.max_retries:
                    raise LLMClientError(
                        "Gemini countTokens request failed after "
                        f"{attempts + 1} attempt(s): {type(error).__name__}; "
                        "generation was not submitted."
                    ) from error
                self._sleep(min(2**attempts, 30))
                attempts += 1
            except json.JSONDecodeError as error:
                raise LLMClientError(
                    "Gemini countTokens returned invalid JSON; generation was not submitted."
                ) from error


def create_provider(config: LLMConfig) -> LLMProvider:
    """Create the explicitly configured provider; never switch providers implicitly."""
    if config.provider == "gemini":
        return GeminiProvider(config)
    if config.provider == "openai":
        return OpenAIChatCompletionsClient(config)
    if config.provider == "claude":
        return AnthropicMessagesProvider(config)
    if config.provider == "grok":
        return XAIChatCompletionsProvider(config)
    raise ValueError(
        f"Unsupported provider {config.provider!r}. "
        "Supported providers: gemini, openai, claude, grok."
    )

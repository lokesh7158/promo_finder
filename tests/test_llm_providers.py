"""Mocked adapter tests; no provider credentials or network access are used."""

from __future__ import annotations

import json
import os
import unittest
from urllib.error import HTTPError
from unittest.mock import patch

from src.evaluation.llm_client import (
    AnthropicMessagesProvider,
    GeminiProvider,
    LLMClientError,
    LLMConfig,
    LLMQuotaExceededError,
    OpenAIChatCompletionsClient,
    XAIChatCompletionsProvider,
    create_provider,
)


class FakeResponse:
    def __init__(self, value: dict) -> None:
        self.value = value

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.value).encode("utf-8")


def _candidate() -> dict:
    return {
        "candidate_id": "episode-001:clip-0001",
        "episode": {"episode_id": "episode-001", "episode_number": 1},
        "time_range": {
            "start": {"raw": "00:00.00", "seconds": 0.0},
            "end": {"raw": "00:24.00", "seconds": 24.0},
            "duration_seconds": 24.0,
        },
        "events": [
            {
                "event": {
                    "id": "episode-001:event-001",
                    "start": {"raw": "00:00.00"},
                    "end": {"raw": "00:05.00"},
                    "content": {"raw": "A door opens."},
                }
            }
        ],
    }


class ProviderAdapterTests(unittest.TestCase):
    def test_environment_selects_each_provider_key_and_default_model(self) -> None:
        settings = (
            ("gemini", "GEMINI_API_KEY", "gemini-3.5-flash-lite"),
            ("openai", "OPENAI_API_KEY", "gpt-4o-mini"),
            ("claude", "ANTHROPIC_API_KEY", "claude-sonnet-5-5"),
            ("grok", "XAI_API_KEY", "grok-4.7"),
        )
        for provider, key_name, model in settings:
            with self.subTest(provider=provider):
                env = {
                    "PROMO_LLM_PROVIDER": provider,
                    key_name: f"key-for-{provider}",
                }
                with patch.dict(os.environ, env, clear=True):
                    config = LLMConfig.from_environment()
                self.assertEqual(config.provider, provider)
                self.assertEqual(config.model, model)
                self.assertEqual(config.api_key, f"key-for-{provider}")

    def test_factory_model_and_connection_probe_for_each_provider(self) -> None:
        cases = (
            ("gemini", "gemini-3.5-flash-lite", "secret-gemini", GeminiProvider),
            ("openai", "custom-openai-model", "secret-openai", OpenAIChatCompletionsClient),
            ("claude", "custom-claude-model", "secret-claude", AnthropicMessagesProvider),
            ("grok", "custom-grok-model", "secret-grok", XAIChatCompletionsProvider),
        )
        for name, model, key, expected_class in cases:
            with self.subTest(provider=name):
                requests = []

                def opener(request, timeout):
                    requests.append(request)
                    if name == "gemini":
                        return FakeResponse({"name": f"models/{model}"})
                    return FakeResponse({"id": model})

                config = LLMConfig(
                    name,
                    model,
                    key,
                    1,
                    0,
                    5,
                    free_tier_confirmed=name == "gemini",
                )
                provider = (
                    expected_class(config, opener=opener, sleeper=lambda _: None)
                    if name != "gemini"
                    else expected_class(
                        config,
                        opener=opener,
                        sleeper=lambda _: None,
                        clock=lambda: 0.0,
                    )
                )
                self.assertIsInstance(create_provider(config), expected_class)
                result = provider.test_connection()
                self.assertEqual(provider.model_name, model)
                self.assertEqual(result["provider"], name)
                self.assertIn(model, result["model"])
                self.assertEqual(len(requests), 1)
                request = requests[0]
                if name == "gemini":
                    self.assertEqual(request.get_header("X-goog-api-key"), key)
                elif name == "claude":
                    self.assertEqual(request.get_header("X-api-key"), key)
                else:
                    self.assertEqual(request.get_header("Authorization"), f"Bearer {key}")
                self.assertNotIn(key, repr(result))

    def test_all_adapters_normalize_their_documented_response_shapes(self) -> None:
        providers = ("gemini", "openai", "claude", "grok")
        for name in providers:
            with self.subTest(provider=name):
                model = "gemini-3.5-flash-lite" if name == "gemini" else "custom-model"
                config = LLMConfig(
                    name,
                    model,
                    f"secret-{name}",
                    1,
                    0,
                    5,
                    free_tier_confirmed=name == "gemini",
                )
                captured = []
                raw = '{"candidate_id":"episode-001:clip-0001"}'
                proposal_raw = '{"promo_id":"promo-test"}'
                generation_count = 0

                def opener(request, timeout):
                    nonlocal generation_count
                    captured.append(request)
                    if name == "gemini" and request.full_url.endswith(":countTokens"):
                        return FakeResponse({"totalTokens": 10})
                    if name == "gemini":
                        generation_count += 1
                        response_text = raw if generation_count == 1 else proposal_raw
                        return FakeResponse(
                            {
                                "modelVersion": model,
                                "candidates": [
                                    {
                                        "finishReason": "STOP",
                                        "content": {"parts": [{"text": response_text}]},
                                    }
                                ],
                                "usageMetadata": {"totalTokenCount": 17},
                            }
                        )
                    response_text = raw if generation_count == 0 else proposal_raw
                    generation_count += 1
                    if name == "claude":
                        return FakeResponse(
                            {
                                "id": "response-id",
                                "model": model,
                                "content": [{"type": "text", "text": response_text}],
                                "usage": {"input_tokens": 8, "output_tokens": 4},
                            }
                        )
                    return FakeResponse(
                        {
                            "id": "response-id",
                            "model": model,
                            "choices": [{"message": {"content": response_text}}],
                            "usage": {"prompt_tokens": 8, "completion_tokens": 4},
                        }
                    )

                provider = create_provider(config)
                provider._open = opener
                provider._sleep = lambda _: None
                response, metadata = provider.evaluate(_candidate())
                self.assertEqual(json.loads(response)["candidate_id"], _candidate()["candidate_id"])
                self.assertEqual(metadata["requested_model"], model)
                self.assertEqual(metadata["provider"], name)
                self.assertIn("usage", metadata)
                self.assertNotIn(f"secret-{name}", response + repr(metadata))
                self.assertTrue(captured)
                proposal = {
                    "promo_id": "promo-test",
                    "promo_category": "SINGLE_SEGMENT",
                    "total_duration_seconds": 20,
                    "segments": [],
                }
                proposal_response, _ = provider.evaluate_proposal(proposal)
                self.assertEqual(json.loads(proposal_response)["promo_id"], "promo-test")
                request_body = json.loads(captured[-1].data.decode("utf-8"))
                if name != "gemini":
                    self.assertEqual(request_body["model"], model)
                    if name == "claude":
                        schema = request_body["output_config"]["format"]["schema"]
                        self.assertIn("promo_id", schema["properties"])
                        self.assertNotIn("candidate_id", schema["properties"])
                    else:
                        self.assertEqual(
                            request_body["response_format"]["json_schema"]["name"],
                            "promo_proposal_evaluation",
                        )
                else:
                    schema = request_body["generationConfig"]["responseJsonSchema"]
                    self.assertIn("promo_id", schema["properties"])
                    self.assertNotIn("candidate_id", schema["properties"])

    def test_malformed_json_is_not_repaired_or_fabricated(self) -> None:
        config = LLMConfig("openai", "custom-model", "secret", 1, 0, 5)
        provider = OpenAIChatCompletionsClient(
            config,
            opener=lambda *_args, **_kwargs: FakeResponse(
                {"choices": [{"message": {"content": "not-json"}}]}
            ),
        )
        text, _ = provider.evaluate(_candidate())
        with self.assertRaises(json.JSONDecodeError):
            json.loads(text)

    def test_provider_errors_are_normalized_and_rate_limit_stops(self) -> None:
        for provider_type, name in (
            (OpenAIChatCompletionsClient, "openai"),
            (AnthropicMessagesProvider, "claude"),
            (XAIChatCompletionsProvider, "grok"),
        ):
            with self.subTest(provider=name):
                config = LLMConfig(name, "model", "secret-key", 1, 0, 1)

                def rate_limited(*_args, **_kwargs):
                    raise HTTPError("https://provider.invalid", 429, "limited", {}, None)

                client = provider_type(config, opener=rate_limited, sleeper=lambda _: None)
                with self.assertRaisesRegex(LLMQuotaExceededError, "rate limit"):
                    client.evaluate(_candidate())

                def unauthorized(*_args, **_kwargs):
                    raise HTTPError("https://provider.invalid", 401, "unauthorized", {}, None)

                client = provider_type(config, opener=unauthorized, sleeper=lambda _: None)
                with self.assertRaisesRegex(LLMClientError, "API key"):
                    client.test_connection()

    def test_timeout_is_bounded_and_does_not_expose_key(self) -> None:
        config = LLMConfig("grok", "model", "secret-never-log", 1, 0, 1)

        def timeout(*_args, **_kwargs):
            raise TimeoutError("network timeout")

        provider = XAIChatCompletionsProvider(config, opener=timeout)
        with self.assertRaisesRegex(LLMClientError, "TimeoutError") as raised:
            provider.evaluate(_candidate())
        self.assertNotIn("secret-never-log", str(raised.exception))

    def test_model_schema_and_insufficient_quota_errors_are_actionable(self) -> None:
        config = LLMConfig("grok", "custom-model", "secret", 1, 0, 1)

        def unsupported(*_args, **_kwargs):
            raise HTTPError("https://provider.invalid", 404, "missing", {}, None)

        provider = XAIChatCompletionsProvider(config, opener=unsupported)
        with self.assertRaisesRegex(LLMClientError, "selected model"):
            provider.test_connection()

        def invalid_schema(*_args, **_kwargs):
            raise HTTPError("https://provider.invalid", 400, "bad schema", {}, None)

        provider = XAIChatCompletionsProvider(config, opener=invalid_schema)
        with self.assertRaisesRegex(LLMClientError, "structured-output schema"):
            provider.evaluate(_candidate())

        def insufficient_quota(*_args, **_kwargs):
            raise HTTPError("https://provider.invalid", 402, "payment required", {}, None)

        provider = XAIChatCompletionsProvider(config, opener=insufficient_quota)
        with self.assertRaisesRegex(LLMQuotaExceededError, "insufficient quota"):
            provider.evaluate(_candidate())


if __name__ == "__main__":
    unittest.main()

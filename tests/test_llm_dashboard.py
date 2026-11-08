"""Dashboard smoke tests prove startup and provider changes make no API calls."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

import streamlit
from streamlit.testing.v1 import AppTest

from services.llm_credentials import clear_all_credentials


class LLMConfigurationUITests(unittest.TestCase):
    def test_hosted_flag_forces_demo_when_mode_requests_local(self) -> None:
        application = Path(__file__).resolve().parents[1] / "app.py"
        clear_all_credentials()
        try:
            with patch.dict(
                os.environ,
                {"PROMOFINDER_HOSTED": "1", "PROMOFINDER_MODE": "local"},
            ):
                with patch(
                    "urllib.request.urlopen",
                    side_effect=AssertionError("Hosted demo attempted a provider API call"),
                ) as urlopen:
                    app = AppTest.from_file(str(application), default_timeout=20).run()
                    self.assertFalse(app.exception)
                    app.radio[0].set_value("Upload & Run").run()
                    self.assertFalse(app.exception)
                    self.assertTrue(
                        any("DEMO MODE" in item.value for item in app.markdown)
                    )
                    self.assertTrue(
                        any(
                            item.label == "Download 50-file synthetic sample pack"
                            for item in app.get("download_button")
                        )
                    )
                    self.assertFalse(
                        any(item.label == "Provider" for item in app.selectbox)
                    )
                    self.assertFalse(
                        any(item.label == "Provider API key" for item in app.text_input)
                    )
                    self.assertFalse(
                        any(
                            item.label == "Run LLM Evaluation"
                            for item in app.form_submit_button
                        )
                    )
                    urlopen.assert_not_called()
        finally:
            clear_all_credentials()

    def test_hosted_flag_is_read_from_streamlit_secrets(self) -> None:
        application = Path(__file__).resolve().parents[1] / "app.py"
        clear_all_credentials()
        try:
            with patch.dict(
                os.environ,
                {"PROMOFINDER_HOSTED": "", "PROMOFINDER_MODE": "local"},
            ):
                with patch.object(streamlit, "secrets", {"PROMOFINDER_HOSTED": "1"}):
                    with patch(
                        "urllib.request.urlopen",
                        side_effect=AssertionError("Hosted demo attempted a provider API call"),
                    ) as urlopen:
                        app = AppTest.from_file(str(application), default_timeout=20).run()
                        self.assertFalse(app.exception)
                        app.radio[0].set_value("Upload & Run").run()
                        self.assertFalse(app.exception)
                        self.assertTrue(
                            any("DEMO MODE" in item.value for item in app.markdown)
                        )
                        self.assertFalse(
                            any(item.label == "Provider" for item in app.selectbox)
                        )
                        self.assertFalse(
                            any(
                                item.label == "Provider API key"
                                for item in app.text_input
                            )
                        )
                        self.assertFalse(
                            any(
                                item.label == "Run LLM Evaluation"
                                for item in app.form_submit_button
                            )
                        )
                        urlopen.assert_not_called()
        finally:
            clear_all_credentials()

    def test_startup_and_provider_selection_do_not_make_network_requests(self) -> None:
        application = Path(__file__).resolve().parents[1] / "app.py"
        clear_all_credentials()
        try:
            with patch.dict(os.environ, {"PROMOFINDER_MODE": "local"}):
                with patch(
                    "urllib.request.urlopen",
                    side_effect=AssertionError("API call without an explicit action"),
                ):
                    app = AppTest.from_file(str(application), default_timeout=20).run()
                    self.assertFalse(app.exception)
                    app.radio[0].set_value("Upload & Run").run()
                    self.assertFalse(app.exception)
                    provider = next(item for item in app.selectbox if item.label == "Provider")
                    provider.set_value("Anthropic Claude").run()
                    self.assertFalse(app.exception)
                    self.assertEqual(
                        next(item for item in app.selectbox if item.label == "Provider").value,
                        "Anthropic Claude",
                    )
                    app.text_input[0].set_value("mock-only-secret")
                    next(
                        item
                        for item in app.checkbox
                        if item.label.startswith("Save for this session")
                    ).set_value(True)
                    app.form_submit_button[1].click().run()
                    self.assertFalse(app.exception)
                    self.assertEqual(app.session_state.get("llm_api_key_input"), "")
                    self.assertNotIn("mock-only-secret", repr(app.session_state))
        finally:
            clear_all_credentials()


if __name__ == "__main__":
    unittest.main()

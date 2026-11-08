"""Tests for process-local dashboard credential lifetime."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from services.llm_credentials import (
    clear_all_credentials,
    clear_credential,
    clear_session_credentials,
    credential_for_action,
    has_saved_credential,
    store_credential,
)


class SessionCredentialTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_all_credentials()

    def tearDown(self) -> None:
        clear_all_credentials()

    def test_unsaved_key_is_available_for_one_action_only(self) -> None:
        store_credential("session-1", "openai", "secret", save_for_session=False)
        self.assertEqual(credential_for_action("session-1", "openai"), "secret")
        self.assertIsNone(credential_for_action("session-1", "openai"))
        self.assertFalse(has_saved_credential("session-1", "openai"))

    def test_saved_key_is_provider_scoped_and_can_be_cleared(self) -> None:
        store_credential("session-1", "claude", "secret", save_for_session=True)
        self.assertEqual(credential_for_action("session-1", "claude"), "secret")
        self.assertTrue(has_saved_credential("session-1", "claude"))
        self.assertIsNone(credential_for_action("session-1", "grok"))
        clear_credential("session-1", "claude")
        self.assertIsNone(credential_for_action("session-1", "claude"))

    def test_saved_credentials_expire_after_idle_ttl(self) -> None:
        with patch("services.llm_credentials.time.monotonic", return_value=100.0):
            store_credential("session-1", "gemini", "secret", save_for_session=True)
        with patch("services.llm_credentials.time.monotonic", return_value=100.0 + 20 * 60):
            self.assertFalse(has_saved_credential("session-1", "gemini"))
            self.assertIsNone(credential_for_action("session-1", "gemini"))

    def test_session_cleanup_removes_all_provider_keys(self) -> None:
        store_credential("session-1", "openai", "one", save_for_session=True)
        store_credential("session-1", "grok", "two", save_for_session=True)
        store_credential("session-2", "openai", "three", save_for_session=True)

        clear_session_credentials("session-1")

        self.assertIsNone(credential_for_action("session-1", "openai"))
        self.assertIsNone(credential_for_action("session-1", "grok"))
        self.assertEqual(credential_for_action("session-2", "openai"), "three")


if __name__ == "__main__":
    unittest.main()

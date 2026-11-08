"""Short-lived in-memory credential storage for Streamlit sessions."""

from __future__ import annotations

import threading
import time


_SESSION_TTL_SECONDS = 20 * 60
_SUBMISSION_TTL_SECONDS = 2 * 60
_lock = threading.Lock()
_credentials: dict[tuple[str, str], tuple[str, float, bool]] = {}


def store_credential(
    session_id: str,
    provider: str,
    api_key: str,
    *,
    save_for_session: bool,
) -> None:
    """Keep a submitted key in process memory, never in application session state."""
    if not session_id or not provider or not api_key:
        raise ValueError("A session ID, provider, and non-empty API key are required.")
    lifetime = _SESSION_TTL_SECONDS if save_for_session else _SUBMISSION_TTL_SECONDS
    with _lock:
        _expire_credentials()
        _credentials[(session_id, provider)] = (
            api_key,
            time.monotonic() + lifetime,
            save_for_session,
        )


def credential_for_action(session_id: str, provider: str) -> str | None:
    """Consume one unsaved submission or return an explicitly saved session key."""
    with _lock:
        _expire_credentials()
        credential_id = (session_id, provider)
        entry = _credentials.get(credential_id)
        if entry is None:
            return None
        api_key, _, saved = entry
        if not saved:
            del _credentials[credential_id]
        else:
            _credentials[credential_id] = (
                api_key,
                time.monotonic() + _SESSION_TTL_SECONDS,
                True,
            )
        return api_key


def has_saved_credential(session_id: str, provider: str) -> bool:
    with _lock:
        _expire_credentials()
        credential_id = (session_id, provider)
        entry = _credentials.get(credential_id)
        if entry is None or not entry[2]:
            return False
        _credentials[credential_id] = (
            entry[0],
            time.monotonic() + _SESSION_TTL_SECONDS,
            True,
        )
        return True


def clear_credential(session_id: str, provider: str) -> None:
    """Remove a session credential immediately."""
    with _lock:
        _credentials.pop((session_id, provider), None)


def clear_session_credentials(session_id: str) -> None:
    """Remove all credentials for one Streamlit session."""
    with _lock:
        for credential_id in [
            key for key in _credentials if key[0] == session_id
        ]:
            del _credentials[credential_id]


def clear_all_credentials() -> None:
    """Clear process-local credentials; intended for deterministic tests."""
    with _lock:
        _credentials.clear()


def _expire_credentials() -> None:
    now = time.monotonic()
    expired = [
        session_id
        for session_id, (_, expires_at, _) in _credentials.items()
        if expires_at <= now
    ]
    for session_id in expired:
        del _credentials[session_id]

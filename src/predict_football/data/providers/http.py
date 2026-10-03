"""Shared outbound HTTP helper for JSON API providers.

football-data.org and API-Football both need the same small piece of transport:
a GET with a timeout, an identifying User-Agent, per-request headers (the API
key), optional query parameters, and a status check that turns a non-200 into a
:class:`ProviderError` rather than letting a JSON decode fail confusingly later.

Keeping it in one place means the retry/timeout/key policy cannot drift between
providers, and the fake session used in tests only has to model one signature.
"""

from __future__ import annotations

from typing import Any

from predict_football.config.settings import HTTP_USER_AGENT, Settings
from predict_football.data.providers.base import ProviderError


def get_bytes(
    url: str,
    *,
    session: Any,
    settings: Settings,
    display_name: str,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
) -> bytes:
    """Fetch a URL and return the response body.

    Args:
        url: Absolute URL to fetch.
        session: Requests session, or a fake with the same ``get`` signature.
            When ``None`` a short-lived session is created and closed.
        settings: Resolved settings, used for the timeout.
        display_name: Provider name used in error messages.
        headers: Extra request headers. The caller adds its API key here.
        params: Query parameters. Kept separate from the URL so the raw cache
            can key on the logical resource rather than the rendered query, and
            so tests can assert what was requested.

    Returns:
        Response body bytes.

    Raises:
        ProviderError: If ``requests`` is unavailable, the request fails, or the
            status code is not 200.
    """
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - requests is a hard dependency
        raise ProviderError("requests is required for network access") from exc

    owned = session is None
    if owned:
        session = requests.Session()

    request_headers = {"User-Agent": HTTP_USER_AGENT}
    if headers:
        request_headers.update(headers)

    try:
        response = session.get(
            url,
            timeout=settings.http_timeout,
            headers=request_headers,
            params=params,
        )
        if response.status_code != 200:
            raise ProviderError(
                f"{display_name}: HTTP {response.status_code} for {url}. "
                f"Check the API key, the quota and that the competition and season exist."
            )
        return response.content
    except ProviderError:
        raise
    except Exception as exc:
        raise ProviderError(f"{display_name}: request to {url} failed: {exc}") from exc
    finally:
        if owned:
            session.close()


__all__ = ["get_bytes"]

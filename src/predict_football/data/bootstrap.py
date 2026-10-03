"""First-run data acquisition for a public deployment.

A deployed app arrives with an empty data directory: the SQLite store is
gitignored, so a fresh checkout has no matches, and the football-data.co.uk
history that powers a local checkout may not be served from a public app (see
:mod:`predict_football.config.licences`). This module closes that gap without
weakening the project's honesty rules:

* It fetches **only** a source whose licence permits public serving, and asks
  :func:`assert_can_serve_publicly` first so an inappropriate source fails loudly
  instead of quietly shipping.
* It reports the attribution the licence requires, so the app can display it.
* It raises an honest error when the API token is missing rather than inventing
  data or silently showing an empty app.

The bootstrap is idempotent: once the store holds matches for the league it
returns them without touching the network.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

from predict_football.config.licences import assert_can_serve_publicly, attribution_for
from predict_football.config.settings import Settings, get_settings
from predict_football.data.loaders import build_repository, ingest_season
from predict_football.data.providers.base import DataProvider, ProviderError, ProviderNotAvailable
from predict_football.data.providers.registry import get_provider

logger = logging.getLogger(__name__)

#: Provider used for a public deployment. Chosen because its licence explicitly
#: permits serving from a single deployed application; the richer
#: football-data.co.uk history grants no redistribution rights.
PUBLIC_PROVIDER = "football_data_org"

#: Competition bootstrapped by default. One league keeps the free tier's
#: ten-calls-per-minute budget comfortable.
PUBLIC_LEAGUE = "ENG_PL"

#: Number of completed seasons to pull before the season in progress. The free
#: tier serves three past seasons; older requests fail and are reported, not
#: guessed.
PUBLIC_HISTORY_SEASONS = 3


class BootstrapError(RuntimeError):
    """Raised when a public deployment cannot be given usable, compliant data."""


@dataclass(frozen=True)
class PublicDataStatus:
    """Outcome of the first-run bootstrap.

    Attributes:
        ready: Whether the store now holds matches that may be served publicly.
        provider: Provider key that was used or would have been used.
        league_key: Competition bootstrapped.
        matches: Number of matches now stored for the competition.
        seasons_ingested: Season codes successfully fetched in this run.
        attribution: Attribution strings required by the served source.
        message: Human-readable explanation, safe to show to a reader.
    """

    ready: bool
    provider: str
    league_key: str
    matches: int = 0
    seasons_ingested: tuple[str, ...] = ()
    attribution: tuple[str, ...] = ()
    message: str = ""


def default_seasons(today: date | None = None) -> list[str]:
    """Return the season codes to fetch for a public bootstrap.

    The three completed seasons before today's, plus the season in progress,
    give a first-run deployment roughly a thousand completed matches -- more than
    enough for the walk-forward warm-up -- within the free tier's quota.

    Args:
        today: Reference date. Defaults to the current date, injected so the
            result is testable without freezing the clock.

    Returns:
        Season codes, oldest first, e.g. ``["2324", "2425", "2526", "2627"]``.
    """
    moment = today or date.today()
    current = moment.year if moment.month >= 8 else moment.year - 1
    return [f"{year % 100:02d}{(year + 1) % 100:02d}" for year in range(current - PUBLIC_HISTORY_SEASONS, current + 1)]


def ensure_public_data(
    settings: Settings | None = None,
    *,
    provider_name: str = PUBLIC_PROVIDER,
    league_key: str = PUBLIC_LEAGUE,
    seasons: list[str] | None = None,
    api_key: str | None = None,
    provider: DataProvider | None = None,
) -> PublicDataStatus:
    """Ensure a compliant competition store exists, fetching it if necessary.

    Args:
        settings: Resolved settings. Defaults to the process settings, whose
            data directory is created if missing.
        provider_name: Provider key. Must be one whose licence permits public
            serving, otherwise :func:`assert_can_serve_publicly` raises.
        league_key: Competition to store.
        seasons: Season codes to fetch. Defaults to :func:`default_seasons`.
        api_key: API token for providers that need one. Read from the environment
            by the provider when omitted.
        provider: Pre-built provider, injected for tests so the network is never
            required.

    Returns:
        A :class:`PublicDataStatus` describing what is now stored.

    Raises:
        LicenceViolation: If ``provider_name`` may not be served publicly.
        BootstrapError: If no matches could be loaded. The message names the
            reason (usually a missing token or an unavailable season) so the app
            can tell the reader what to fix.
    """
    resolved = settings or get_settings()
    assert_can_serve_publicly(provider_name)
    resolved.ensure_directories()
    repository = build_repository(resolved)

    existing = repository.load_matches(league_key=league_key, include_stats=False)
    if not existing.empty:
        return PublicDataStatus(
            ready=True,
            provider=provider_name,
            league_key=league_key,
            matches=len(existing),
            attribution=(attribution_for(provider_name),),
            message=f"Using {len(existing)} stored matches for {league_key}.",
        )

    adapter = provider or get_provider(provider_name, settings=resolved, api_key=api_key)
    requested = seasons or default_seasons()
    ingested: list[str] = []
    warnings: list[str] = []
    for code in requested:
        try:
            result = ingest_season(adapter, repository, league_key=league_key, season_code=code, include_odds=False)
        except (ProviderNotAvailable, ProviderError, ValueError) as exc:
            warnings.append(f"{code}: {exc}")
            logger.warning("bootstrap could not ingest %s %s: %s", league_key, code, exc)
        else:
            ingested.append(code)
            warnings.extend(f"{code}: {warning}" for warning in result.warnings)

    matches = int(repository.match_count(league_key=league_key))
    if matches == 0:
        detail = "; ".join(warnings) if warnings else "the provider returned no matches"
        raise BootstrapError(
            f"Could not load any {league_key} matches from {provider_name}. {detail}"
        )

    logger.info("bootstrap loaded %d matches for %s from %s", matches, league_key, provider_name)
    return PublicDataStatus(
        ready=True,
        provider=provider_name,
        league_key=league_key,
        matches=matches,
        seasons_ingested=tuple(ingested),
        attribution=(attribution_for(provider_name),),
        message=f"Loaded {matches} matches for {league_key} from {provider_name}.",
    )


__all__ = [
    "PUBLIC_HISTORY_SEASONS",
    "PUBLIC_LEAGUE",
    "PUBLIC_PROVIDER",
    "BootstrapError",
    "PublicDataStatus",
    "default_seasons",
    "ensure_public_data",
]

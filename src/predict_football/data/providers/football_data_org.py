"""football-data.org v4 adapter.

A free-tier live source for results and fixtures. It is useful for topping up a
season in progress, but it is *not* a history or odds source:

* Free tier: 10 calls per minute, DELAYED scores only, no lineups, no odds, no
  live minute-by-minute.
* Mandatory attribution and single-application use (see the licence entry).

LICENCE WARNING -- READ BEFORE SHIPPING ANYTHING HERE
=====================================================
The free-tier terms (verified 2026-10-02) permit serving the data from a single
application with attribution, but forbid commercial use. ``may_commit_raw_data``
is False, so this provider's payloads are cached locally and gitignored. The
required attribution is in :data:`predict_football.config.licences.attribution_for`.

Design notes
------------
* Competition codes here are football-data.org's own short codes (``"PL"``,
  ``"PD"``, ...), stored in :attr:`League.source_codes`. They are published and
  human-readable, unlike the numeric IDs other providers use, so no manifest
  discovery is needed.
* The API wants a *season start year* (``2024``), while the rest of the project
  uses the four-character code (``"2425"``). :func:`season_start_year` bridges
  the two; this adapter still speaks the project convention on its public
  methods, so ingestion and storage keys stay uniform.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any

import pandas as pd

from predict_football.config.leagues import get_league, season_label, season_start_year
from predict_football.config.licences import licence_for
from predict_football.config.settings import Settings, get_settings
from predict_football.data.cache import RawCache
from predict_football.data.identifiers import make_match_id
from predict_football.data.providers.base import (
    DataProvider,
    ProviderError,
    ProviderNotAvailable,
    SeasonRef,
)
from predict_football.data.providers.http import get_bytes
from predict_football.data.schema import apply_dtypes
from predict_football.data.teams import TeamResolver

logger = logging.getLogger(__name__)

#: Root of the v4 API.
BASE_URL = "https://api.football-data.org/v4"

#: One document per competition season. The season is passed as a query
#: parameter so the cache key stays readable and free of a rendered query string.
MATCHES_URL_TEMPLATE = BASE_URL + "/competitions/{competition_code}/matches"

#: Environment variable holding the API token.
API_KEY_ENV = "FOOTBALL_DATA_ORG_API_KEY"

#: Statuses that mean the scoreline is final. Anything else is a fixture whose
#: result is not knowable yet, so goals are left null rather than guessed.
_FINISHED_STATUSES = {"FINISHED", "AWARDED"}

#: Earliest season the free tier serves for the competitions we map. Coverage is
#: uneven by competition, so a requested season that is unavailable surfaces as a
#: provider error rather than being silently skipped.
_FIRST_SEASON_START_YEAR = 2000


class FootballDataOrgProvider(DataProvider):
    """Adapter for the football-data.org v4 API.

    Args:
        settings: Resolved project settings.
        cache: Raw download cache.
        teams: Team name resolver.
        session: Optional requests session, injected for testability.
        api_key: API token. Defaults to ``FOOTBALL_DATA_ORG_API_KEY`` from the
            environment. Read lazily so importing the provider never requires a
            key.

    Raises:
        ProviderError: If no licence review is on record for this source.
    """

    name = "football_data_org"
    display_name = "football-data.org"

    def __init__(
        self,
        settings: Settings | None = None,
        cache: RawCache | None = None,
        teams: TeamResolver | None = None,
        session: Any | None = None,
        api_key: str | None = None,
    ) -> None:
        licence_for(self.name)
        self._settings = settings or get_settings()
        self._cache = cache or RawCache(self._settings)
        self._teams = teams or TeamResolver()
        self._session = session
        self._api_key = api_key if api_key is not None else self._settings.env.get(API_KEY_ENV, "")

    # --- coverage -----------------------------------------------------------

    def covers(self, league_key: str) -> bool:
        """Report whether a competition code is configured for this provider.

        Args:
            league_key: Internal competition key.

        Returns:
            True if a provider code exists.
        """
        try:
            league = get_league(league_key)
        except KeyError:
            return False
        return league.code_for(self.name) is not None

    def available_seasons(self, league_key: str) -> list[SeasonRef]:
        """List seasons from the earliest mapped year to the current one.

        Args:
            league_key: Internal competition key.

        Returns:
            Seasons ordered oldest to newest.

        Raises:
            ProviderNotAvailable: If the competition has no mapped code.
        """
        league = get_league(league_key)
        if league.code_for(self.name) is None:
            raise ProviderNotAvailable(f"{self.display_name} has no code configured for {league_key}")
        current = self._current_season_start_year()
        return [
            SeasonRef(
                league_key=league_key,
                season_code=f"{y % 100:02d}{(y + 1) % 100:02d}",
                season_label=season_label(f"{y % 100:02d}{(y + 1) % 100:02d}"),
            )
            for y in range(_FIRST_SEASON_START_YEAR, current + 1)
        ]

    @staticmethod
    def _current_season_start_year() -> int:
        """Return the start year of the current European season.

        Returns:
            The current season's starting year (August boundary), so a July run
            still refers to the season that began the previous year.
        """
        today = date.today()
        return today.year if today.month >= 8 else today.year - 1

    # --- fetching -----------------------------------------------------------

    def _competition_code(self, league_key: str) -> str:
        """Resolve the provider competition code for an internal key.

        Args:
            league_key: Internal competition key.

        Returns:
            The provider's short competition code.

        Raises:
            ProviderNotAvailable: If no code is mapped.
        """
        code = get_league(league_key).code_for(self.name)
        if code is None:
            raise ProviderNotAvailable(f"{self.display_name} does not cover {league_key}")
        return code

    def _require_key(self) -> str:
        """Return the API token, or raise a clear error if it is absent.

        Returns:
            The token.

        Raises:
            ProviderError: If no token is configured. The free tier refuses
                anonymous requests, and a blank key would otherwise surface as a
                confusing 403.
        """
        if not self._api_key:
            raise ProviderError(
                f"{self.display_name} requires an API token. Set {API_KEY_ENV} in the environment "
                f"(see .env.example). The free tier still needs a registered token."
            )
        return self._api_key

    def fetch_matches(self, league_key: str, season_code: str) -> pd.DataFrame:
        """Fetch one competition season and normalise it to the canonical schema.

        Args:
            league_key: Internal competition key.
            season_code: Four-character season code, e.g. ``"2425"``.

        Returns:
            Canonical match frame. Fixtures not yet finished carry a null
            ``result`` and null goals, and are kept because the current season
            legitimately contains future matches.

        Raises:
            ProviderError: If the request fails or the JSON shape is not what the
                API documents.
        """
        competition_code = self._competition_code(league_key)
        start_year = season_start_year(season_code)
        url = MATCHES_URL_TEMPLATE.format(competition_code=competition_code)
        payload = self._cache.get_or_fetch(
            self.name,
            f"matches:{league_key}:{season_code}",
            url,
            fetcher=lambda _url: self._request(url, params={"season": start_year}),
        )
        try:
            document = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"{self.display_name}: response was not JSON for {league_key} {season_code}") from exc

        matches = document.get("matches")
        if not isinstance(matches, list):
            raise ProviderError(
                f"{self.display_name}: response for {league_key} {season_code} had no 'matches' list. "
                f"Keys present: {sorted(document)[:8]}"
            )
        return self._matches_to_canonical(matches, league_key=league_key, season_code=season_code)

    def _request(self, url: str, *, params: dict[str, Any]) -> bytes:
        """Perform an authenticated GET.

        Args:
            url: Absolute URL.
            params: Query parameters.

        Returns:
            Response body bytes.

        Raises:
            ProviderError: If no token is set or the request fails.
        """
        headers = {"X-Auth-Token": self._require_key()}
        return get_bytes(
            url,
            session=self._session,
            settings=self._settings,
            display_name=self.display_name,
            headers=headers,
            params=params,
        )

    def _matches_to_canonical(
        self, matches: list[dict[str, Any]], *, league_key: str, season_code: str
    ) -> pd.DataFrame:
        """Map published match objects onto the canonical match schema.

        Args:
            matches: Raw match dictionaries from the API.
            league_key: Internal competition key.
            season_code: Four-character season code.

        Returns:
            Canonical match frame with a ``football_data_org_id`` column kept for
            future per-match lookups.
        """
        league = get_league(league_key)
        label = _season_label_for_code(season_code)
        records: list[dict[str, Any]] = []

        for match in matches:
            home_name = _team_name(match.get("homeTeam"))
            away_name = _team_name(match.get("awayTeam"))
            home = self._teams.resolve_or_pass_through(home_name) if home_name else None
            away = self._teams.resolve_or_pass_through(away_name) if away_name else None

            played_at = _parse_datetime(match.get("utcDate"))
            finished = str(match.get("status", "")).upper() in _FINISHED_STATUSES
            score = match.get("score") or {}
            home_goals, away_goals = _score_pair(score.get("fullTime"), only_if=finished)
            home_ht, away_ht = _score_pair(score.get("halfTime"), only_if=finished)
            pens_home, pens_away = _score_pair(score.get("penalties"), only_if=finished)

            result = _result_from_goals(home_goals, away_goals)
            records.append(
                {
                    "match_id": (
                        make_match_id(league_key, played_at.date().isoformat(), home, away)
                        if played_at is not None and home and away
                        else None
                    ),
                    "source": self.name,
                    "league_key": league_key,
                    "competition_type": league.competition_type.value,
                    "season": label,
                    "season_code": season_code,
                    "match_date": played_at,
                    "match_week": match.get("matchday"),
                    "home_team": home,
                    "away_team": away,
                    "venue": match.get("venue"),
                    "referee": _first_referee(match.get("referees")),
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "home_goals_ht": home_ht,
                    "away_goals_ht": away_ht,
                    "result": result,
                    "pens_home": pens_home,
                    "pens_away": pens_away,
                    "football_data_org_id": match.get("id"),
                }
            )

        frame = pd.DataFrame(records)
        logger.info(
            "%s: loaded %d matches for %s season=%s",
            self.display_name,
            len(frame),
            league_key,
            season_code,
        )
        return apply_dtypes(frame, table="matches", keep_extra=True)


def _season_label_for_code(season_code: str) -> str:
    """Return a human season label for either season-code convention.

    Args:
        season_code: Four-character split code or calendar year.

    Returns:
        ``"2024/25"`` for a split season, or the year itself for an annual one.
    """
    try:
        return season_label(season_code)
    except ValueError:
        return str(season_start_year(season_code))


def _team_name(team: Any) -> str | None:
    """Read a club name from a team object without inventing one.

    Args:
        team: The published team object, or ``None``.

    Returns:
        The full name, falling back to the short name, or ``None`` when absent.
    """
    if not isinstance(team, dict):
        return None
    for key in ("name", "shortName"):
        value = team.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _parse_datetime(value: Any) -> pd.Timestamp | None:
    """Parse an ISO timestamp to a timezone-naive UTC timestamp.

    Args:
        value: Published date string, e.g. ``"2024-08-16T19:00:00Z"``.

    Returns:
        Timestamp in naive UTC, or ``None`` when absent or unparseable. UTC is
        stripped rather than converted so dates sort consistently regardless of
        the machine's local zone.
    """
    if not value:
        return None
    try:
        stamp = pd.to_datetime(value, errors="coerce", utc=True)
    except (TypeError, ValueError):
        return None
    if pd.isna(stamp):
        return None
    return stamp.tz_convert(None)


def _score_pair(block: Any, *, only_if: bool) -> tuple[int | None, int | None]:
    """Read a ``{"home":, "away":}`` score block.

    Args:
        block: The score block, or ``None``.
        only_if: When False, return nulls regardless. Used to refuse to publish
            goals for a match that has not finished.

    Returns:
        Tuple of home and away goals, each ``int`` or ``None``.
    """
    if not only_if or not isinstance(block, dict):
        return None, None
    home = block.get("home")
    away = block.get("away")
    return (_as_int(home), _as_int(away))


def _as_int(value: Any) -> int | None:
    """Coerce a published numeric value to ``int`` or ``None``.

    Args:
        value: Raw value.

    Returns:
        The integer, or ``None`` when missing or not numeric. ``None`` is used
        rather than ``0`` because an unpublished score is unknown, not nil.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):  # pragma: no cover - non-scalar
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _result_from_goals(home_goals: int | None, away_goals: int | None) -> str | None:
    """Derive the 1X2 result from a scoreline.

    Args:
        home_goals: Full-time home goals, or ``None``.
        away_goals: Full-time away goals, or ``None``.

    Returns:
        ``"H"``, ``"D"``, ``"A"`` or ``None`` when the score is unknown.
    """
    if home_goals is None or away_goals is None:
        return None
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def _first_referee(referees: Any) -> str | None:
    """Read the first referee name from the published list.

    Args:
        referees: Published referees list, or ``None``.

    Returns:
        The first referee's name, or ``None``.
    """
    if not isinstance(referees, list) or not referees:
        return None
    first = referees[0]
    if isinstance(first, dict):
        name = first.get("name")
        return str(name) if name else None
    return None


__all__ = [
    "API_KEY_ENV",
    "BASE_URL",
    "MATCHES_URL_TEMPLATE",
    "FootballDataOrgProvider",
]

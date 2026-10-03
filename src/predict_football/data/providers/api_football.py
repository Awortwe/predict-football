"""API-Football (api-sports.io) v3 adapter.

A second live source, useful when football-data.org's competition codes or
season coverage are insufficient. It is deliberately treated as **unverified**:

* The licence terms could not be read -- the documentation site returned HTTP
  403 when checked (`config/licences.py`). ``may_serve_from_public_app`` is
  False, so this provider exists for internal research and reconciliation only.
* The free tier allows 100 requests per day, which cannot backfill a league.
  A single competition season is one request, so a daily poll of a handful of
  competitions fits, but a history import does not.

Design notes
------------
* The v3 API addresses competitions by **numeric** league id. We refuse to
  hard-code those ids (they are opaque and can be confused across countries), so
  :meth:`_resolve_league_id` discovers them from the ``/leagues`` manifest by
  matching the published name and country against :class:`League.provider_names`
  and :attr:`League.country`. If no confident match is found it raises rather
  than guessing, mirroring the team resolver's "a gap stays visible" rule.
* Seasons are exposed oldest to newest as the project's four-character code, but
  the API wants the season's start year, so :func:`season_start_year` does the
  translation.
"""

from __future__ import annotations

import json
import logging
import re
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

#: Root of the v3 API.
BASE_URL = "https://v3.football.api-sports.io"

#: Competition manifest, used to discover numeric league ids.
LEAGUES_URL = BASE_URL + "/leagues"

#: Fixtures endpoint.
FIXTURES_URL = BASE_URL + "/fixtures"

#: Environment variable holding the API key.
API_KEY_ENV = "API_FOOTBALL_KEY"

#: Statuses that mean a result is final. ``AET`` and ``PEN`` are included
#: because the goals block already folds in extra time; the shootout score is
#: read separately from ``score.penalty``.
_FINISHED_STATUSES = {"FT", "AET", "PEN"}

#: A trailing integer in a round label, e.g. ``"Regular Season - 34"`` -> 34.
_ROUND_NUMBER = re.compile(r"(\d+)\s*$")


class ApiFootballProvider(DataProvider):
    """Adapter for the API-Football v3 API.

    Args:
        settings: Resolved project settings.
        cache: Raw download cache.
        teams: Team name resolver.
        session: Optional requests session, injected for testability.
        api_key: API key. Defaults to ``API_FOOTBALL_KEY`` from the environment.

    Raises:
        ProviderError: If no licence review is on record for this source.
    """

    name = "api_football"
    display_name = "API-Football"

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
        """Report whether a published name is configured for this competition.

        Args:
            league_key: Internal competition key.

        Returns:
            True if :attr:`League.provider_names` names this provider. The
            numeric id is still discovered from the manifest at fetch time, but
            an entry here is the project's declaration that the competition is
            meant to come from this source.
        """
        try:
            league = get_league(league_key)
        except KeyError:
            return False
        return self.name in league.provider_names

    def available_seasons(self, league_key: str) -> list[SeasonRef]:
        """List the seasons the manifest offers for a competition.

        Args:
            league_key: Internal competition key.

        Returns:
            Seasons ordered oldest to newest.

        Raises:
            ProviderNotAvailable: If the competition cannot be matched, or the
                manifest lists no seasons for it.
        """
        league_id, entry = self._resolve_league_entry(league_key)
        seasons = entry.get("seasons") or []
        refs: list[SeasonRef] = []
        for season in seasons:
            year = _as_int(season.get("year"))
            if year is None:
                continue
            code = f"{year % 100:02d}{(year + 1) % 100:02d}"
            refs.append(
                SeasonRef(
                    league_key=league_key,
                    season_code=code,
                    season_label=season_label(code),
                    start_date=_parse_date(season.get("start")),
                    end_date=_parse_date(season.get("end")),
                    match_count=None,
                )
            )
        if not refs:
            raise ProviderNotAvailable(
                f"{self.display_name}: manifest listed no seasons for {league_key} (id {league_id})"
            )
        refs.sort(key=lambda ref: ref.season_code)
        return refs

    # --- manifest discovery -------------------------------------------------

    def _league_manifest(self) -> list[dict[str, Any]]:
        """Download and cache the competition manifest.

        Returns:
            The ``response`` list from the ``/leagues`` document.

        Raises:
            ProviderError: If the request fails or the JSON shape is unexpected.
        """
        payload = self._cache.get_or_fetch(
            self.name,
            "leagues",
            LEAGUES_URL,
            fetcher=lambda _url: self._request(LEAGUES_URL),
        )
        try:
            document = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"{self.display_name}: /leagues response was not JSON") from exc
        response = document.get("response")
        if not isinstance(response, list):
            raise ProviderError(f"{self.display_name}: /leagues response had no 'response' list")
        return response

    def _resolve_league_entry(self, league_key: str) -> tuple[int, dict[str, Any]]:
        """Find the manifest entry for a competition.

        Matching uses both the published name and the country, so "Serie A"
        (Italy) is not confused with "Serie A" (Brazil).

        Args:
            league_key: Internal competition key.

        Returns:
            Pair of the numeric league id and its manifest entry.

        Raises:
            ProviderNotAvailable: If nothing matches confidently.
            ProviderError: If the matched entry has no usable id.
        """
        league = get_league(league_key)
        wanted_name = league.provider_names.get(self.name, league.name).casefold()
        wanted_country = league.country.casefold()
        for entry in self._league_manifest():
            info = entry.get("league") or {}
            country = (entry.get("country") or {}).get("name", "")
            if str(info.get("name", "")).casefold() != wanted_name:
                continue
            if str(country).casefold() != wanted_country:
                continue
            league_id = _as_int(info.get("id"))
            if league_id is None:
                raise ProviderError(f"{self.display_name}: manifest entry for {league_key} had no numeric id")
            return league_id, entry
        raise ProviderNotAvailable(
            f"{self.display_name}: could not match {league_key} ({league.name}, {league.country}) "
            f"in the {len(self._league_manifest())}-entry competition manifest"
        )

    # --- fetching -----------------------------------------------------------

    def _require_key(self) -> str:
        """Return the API key, or raise a clear error if it is absent.

        Returns:
            The key.

        Raises:
            ProviderError: If no key is configured.
        """
        if not self._api_key:
            raise ProviderError(
                f"{self.display_name} requires an API key. Set {API_KEY_ENV} in the environment "
                f"(see .env.example)."
            )
        return self._api_key

    def _request(self, url: str, params: dict[str, Any] | None = None) -> bytes:
        """Perform an authenticated GET.

        Args:
            url: Absolute URL.
            params: Query parameters.

        Returns:
            Response body bytes.

        Raises:
            ProviderError: If no key is set or the request fails.
        """
        headers = {"x-apisports-key": self._require_key()}
        return get_bytes(
            url,
            session=self._session,
            settings=self._settings,
            display_name=self.display_name,
            headers=headers,
            params=params,
        )

    def fetch_matches(self, league_key: str, season_code: str) -> pd.DataFrame:
        """Fetch one competition season and normalise it to the canonical schema.

        Args:
            league_key: Internal competition key.
            season_code: Four-character season code, e.g. ``"2425"``.

        Returns:
            Canonical match frame. Unfinished fixtures keep null goals and a null
            ``result`` so they cannot be mistaken for played matches.

        Raises:
            ProviderError: If the request fails or the JSON shape is unexpected.
            ProviderNotAvailable: If the competition cannot be matched.
        """
        league_id, _ = self._resolve_league_entry(league_key)
        start_year = season_start_year(season_code)
        payload = self._cache.get_or_fetch(
            self.name,
            f"matches:{league_key}:{season_code}",
            FIXTURES_URL,
            fetcher=lambda _url: self._request(FIXTURES_URL, params={"league": league_id, "season": start_year}),
        )
        try:
            document = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"{self.display_name}: response was not JSON for {league_key} {season_code}") from exc

        response = document.get("response")
        if not isinstance(response, list):
            raise ProviderError(
                f"{self.display_name}: fixtures response for {league_key} {season_code} had no 'response' list. "
                f"Keys present: {sorted(document)[:8]}"
            )
        return self._fixtures_to_canonical(response, league_key=league_key, season_code=season_code)

    def _fixtures_to_canonical(
        self, fixtures: list[dict[str, Any]], *, league_key: str, season_code: str
    ) -> pd.DataFrame:
        """Map published fixture objects onto the canonical match schema.

        Args:
            fixtures: Raw fixture dictionaries from the API.
            league_key: Internal competition key.
            season_code: Four-character season code.

        Returns:
            Canonical match frame with an ``api_football_id`` column kept for
            future per-match lookups.
        """
        league = get_league(league_key)
        label = season_label(season_code)
        records: list[dict[str, Any]] = []

        for fixture in fixtures:
            block = fixture.get("fixture") or {}
            teams = fixture.get("teams") or {}
            score = fixture.get("score") or {}
            goals = fixture.get("goals") or {}

            home_name = _team_name(teams.get("home"))
            away_name = _team_name(teams.get("away"))
            home = self._teams.resolve_or_pass_through(home_name) if home_name else None
            away = self._teams.resolve_or_pass_through(away_name) if away_name else None

            played_at = _parse_datetime(block.get("date"))
            status = str((block.get("status") or {}).get("short", "")).upper()
            finished = status in _FINISHED_STATUSES

            home_goals = _as_int(goals.get("home")) if finished else None
            away_goals = _as_int(goals.get("away")) if finished else None
            home_ht, away_ht = _score_pair(score.get("halftime"), only_if=finished)
            pens_home, pens_away = _score_pair(score.get("penalty"), only_if=finished and status == "PEN")

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
                    "match_week": _parse_round((fixture.get("league") or {}).get("round")),
                    "home_team": home,
                    "away_team": away,
                    "venue": (block.get("venue") or {}).get("name"),
                    "referee": block.get("referee"),
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "home_goals_ht": home_ht,
                    "away_goals_ht": away_ht,
                    "result": _result_from_goals(home_goals, away_goals),
                    "pens_home": pens_home,
                    "pens_away": pens_away,
                    "api_football_id": block.get("id"),
                }
            )

        frame = pd.DataFrame(records)
        logger.info(
            "%s: loaded %d fixtures for %s season=%s",
            self.display_name,
            len(frame),
            league_key,
            season_code,
        )
        return apply_dtypes(frame, table="matches", keep_extra=True)


def _team_name(team: Any) -> str | None:
    """Read a club name from a team object without inventing one.

    Args:
        team: The published team object, or ``None``.

    Returns:
        The published name, or ``None``.
    """
    if not isinstance(team, dict):
        return None
    value = team.get("name")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _parse_datetime(value: Any) -> pd.Timestamp | None:
    """Parse an ISO timestamp to a timezone-naive UTC timestamp.

    Args:
        value: Published date string, e.g. ``"2024-08-16T19:00:00+00:00"``.

    Returns:
        Timestamp in naive UTC, or ``None`` when absent or unparseable.
    """
    if not value:
        return None
    stamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(stamp):
        return None
    return stamp.tz_convert(None)


def _parse_date(value: Any) -> Any:
    """Parse a plain ``YYYY-MM-DD`` date.

    Args:
        value: Published date string.

    Returns:
        A :class:`datetime.date`, or ``None``.
    """
    if not value:
        return None
    stamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(stamp):
        return None
    return stamp.date()


def _parse_round(value: Any) -> int | None:
    """Extract a matchday number from a round label.

    Args:
        value: Round label such as ``"Regular Season - 34"`` or ``"Round of 16"``.

    Returns:
        The trailing integer, or ``None`` when there is none. Knockout rounds
        have no matchday and must stay null rather than being numbered.
    """
    if not isinstance(value, str):
        return None
    match = _ROUND_NUMBER.search(value.strip())
    return int(match.group(1)) if match else None


def _score_pair(block: Any, *, only_if: bool) -> tuple[int | None, int | None]:
    """Read a ``{"home":, "away":}`` score block.

    Args:
        block: The score block, or ``None``.
        only_if: When False, return nulls regardless.

    Returns:
        Tuple of home and away values, each ``int`` or ``None``.
    """
    if not only_if or not isinstance(block, dict):
        return None, None
    return _as_int(block.get("home")), _as_int(block.get("away"))


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


__all__ = [
    "API_KEY_ENV",
    "BASE_URL",
    "FIXTURES_URL",
    "LEAGUES_URL",
    "ApiFootballProvider",
]

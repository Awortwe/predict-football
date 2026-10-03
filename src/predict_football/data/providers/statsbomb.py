"""StatsBomb open-data adapter.

The only freely available source of per-shot expected goals with per-event
timestamps, lineups, and starting formations. That combination is what makes an
offline replay feed for the live model possible, so this adapter is load-bearing
for phases 7 and 8 even though the pre-match work does not need it.

LICENCE WARNING -- READ BEFORE CHANGING ANYTHING HERE
=====================================================
This is **not** open source. It is the StatsBomb Public Data User Agreement
(standard terms last updated 2023-09-08), verified 2026-10-02:

* Clause 1.2.1 -- may not edit, distort, distribute, reproduce, sell or in any
  way provide the data to any external or third party.
* Clause 1.2.2 -- may not commercially exploit the data **or any analysis
  derived from it**.
* Clause 1.4 -- **the StatsBomb brand logo is required** on any publication of
  analysis formed from this data.

So: excellent for research and published analysis, provided the logo is
credited. NOT to be committed to a public repository, and NOT to be served from
a public deployment. The licence-clean route to commercial event data is
Wyscout soccer-logs (CC BY 4.0), which has no xG.

A second practical restriction: the terms request registration. Anonymous
downloading works from GitHub today, but access can be withdrawn at any time
(clause 2.1), so we cache aggressively and never depend on the data being
permanently available.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any

import pandas as pd

from predict_football.config.leagues import get_league
from predict_football.config.licences import licence_for
from predict_football.config.settings import HTTP_USER_AGENT, Settings, get_settings
from predict_football.data.cache import RawCache
from predict_football.data.identifiers import make_event_id, make_match_id
from predict_football.data.providers.base import (
    DataProvider,
    ProviderError,
    ProviderNotAvailable,
    SeasonRef,
)
from predict_football.data.schema import EventType, apply_dtypes
from predict_football.data.teams import TeamResolver

logger = logging.getLogger(__name__)

#: Base URL for the open-data repository. The project moved from the statsbomb
#: organisation to hudl, which is why the URL is not what older tutorials show.
REPO_BASE = "https://raw.githubusercontent.com/hudl/open-data/master/data"

COMPETITIONS_URL = f"{REPO_BASE}/competitions.json"
MATCHES_URL_TEMPLATE = f"{REPO_BASE}/matches/{{competition_id}}/{{season_id}}.json"
EVENTS_URL_TEMPLATE = f"{REPO_BASE}/events/{{match_id}}.json"
LINEUPS_URL_TEMPLATE = f"{REPO_BASE}/lineups/{{match_id}}.json"

#: StatsBomb event type -> canonical EventType. Anything unmapped becomes
#: UNKNOWN rather than being silently bucketed somewhere convenient.
EVENT_TYPE_MAP: dict[str, EventType] = {
    "Goal": EventType.GOAL,
    "Own Goal": EventType.OWN_GOAL,
    "Penalty": EventType.PENALTY_GOAL,
    "Missed Penalty": EventType.MISSED_PENALTY,
    "Shot": EventType.SHOT,
    "Save": EventType.UNKNOWN,
    "Post": EventType.SHOT_POST,
    "Block": EventType.BLOCK,
    "Clearance": EventType.CLEARANCE,
    "Pass": EventType.PASS,
    "Pass Offside": EventType.PASS,
    "Ball Receipt*": EventType.RECOVERY,
    "Carry": EventType.CARRY,
    "Cross": EventType.CROSS,
    "Duel": EventType.DUEL,
    "Tackle": EventType.TACKLE,
    "Interception": EventType.INTERCEPTION,
    "Pressure": EventType.PRESSURE,
    "Foul": EventType.FOUL,
    "Offside": EventType.OFFSIDE,
    "Substitution": EventType.SUBSTITUTION_ON,
    "Player On": EventType.SUBSTITUTION_ON,
    "Player Off": EventType.SUBSTITUTION_OFF,
    "Card": EventType.YELLOW_CARD,
    "Yellow Card": EventType.YELLOW_CARD,
    "Red Card": EventType.RED_CARD,
    "Second Yellow": EventType.SECOND_YELLOW,
    "2nd Yellow": EventType.SECOND_YELLOW,
    "Kick Off": EventType.KICK_OFF,
    "Starting XI": EventType.PERIOD_START,
    "Half End": EventType.HALF_TIME,
    "End": EventType.FULL_TIME,
    "Goal Kick": EventType.GOAL_KICK,
    "Corner": EventType.CORNER,
    "Throw-in": EventType.THROW_IN,
    "Recovery": EventType.RECOVERY,
    "Ball Recovery": EventType.RECOVERY,
}

#: Shot outcome -> canonical type refinement.
SHOT_OUTCOME_MAP: dict[str, EventType] = {
    "Goal": EventType.SHOT_ON_TARGET,
    "Saved": EventType.SHOT_ON_TARGET,
    "Blocked": EventType.SHOT_BLOCKED,
    "Off T": EventType.SHOT_OFF_TARGET,
    "Post": EventType.SHOT_POST,
    "Wayward": EventType.SHOT_OFF_TARGET,
}

#: StatisticsBomb pitch is 120 x 80 in its own coordinates. We pass event
#: locations through unchanged so published xG locations stay directly
#: comparable with the source; no rescaling is applied here.
_PITCH_LENGTH = 120.0
_PITCH_WIDTH = 80.0

#: Documented convention: substitutes carry this position id in lineup files.
_SUBSTITUTE_POSITION_ID = 16


class StatsBombOpenProvider(DataProvider):
    """Adapter for the StatsBomb open-data GitHub repository.

    Args:
        settings: Resolved project settings.
        cache: Raw download cache.
        teams: Team name resolver.
        session: Optional requests session, injected for testability.
    """

    name = "statsbomb_open"
    display_name = "StatsBomb Open Data"

    def __init__(
        self,
        settings: Settings | None = None,
        cache: RawCache | None = None,
        teams: TeamResolver | None = None,
        session: Any | None = None,
    ) -> None:
        licence_for(self.name)
        self._settings = settings or get_settings()
        self._cache = cache or RawCache(self._settings)
        self._teams = teams or TeamResolver()
        self._session = session
        self._competitions: pd.DataFrame | None = None

    # --- manifest -----------------------------------------------------------

    def competitions(self) -> pd.DataFrame:
        """Return the published competitions manifest.

        Returns:
            Frame with competition_id, season_id, country_name, competition_name
            and season_name as published by StatsBomb.

        Raises:
            ProviderError: If the manifest cannot be fetched or parsed.
        """
        if self._competitions is not None:
            return self._competitions

        payload = self._cache.get_or_fetch(
            self.name, "competitions", COMPETITIONS_URL, fetcher=self._http_get_json
        )
        data = json.loads(payload)
        rows = [
            {
                "competition_id": entry.get("competition_id"),
                "season_id": entry.get("season_id"),
                "country_name": entry.get("country_name"),
                "competition_name": entry.get("competition_name"),
                "season_name": entry.get("season_name"),
                "match_available_360": entry.get("match_available_360"),
                "match_available": entry.get("match_available"),
            }
            for entry in data
        ]
        self._competitions = pd.DataFrame(rows)
        return self._competitions

    def discover(self, competition_name: str) -> pd.DataFrame:
        """Find manifest entries for a competition by published name.

        We match on the provider's own competition name rather than hard-coding
        numeric IDs. Those IDs are renumbered without notice across releases,
        so a hard-coded ID silently returns the wrong competition -- the worst
        possible failure for a data source.

        Args:
            competition_name: Published competition name, e.g. ``"Premier League"``.

        Returns:
            Matching manifest rows.
        """
        manifest = self.competitions()
        mask = manifest["competition_name"].astype("string").str.casefold() == competition_name.casefold()
        return manifest.loc[mask].copy()

    def resolve_competition(self, league_key: str) -> pd.DataFrame:
        """Resolve an internal league key to StatsBomb manifest rows.

        Args:
            league_key: Internal competition key, e.g. ``"INT_AFCON"``.

        Returns:
            Manifest rows for the competition.

        Raises:
            ProviderNotAvailable: If no published name is configured, or the
                provider has no data for it.
        """
        league = get_league(league_key)
        published = league.provider_names.get(self.name)
        if published is None:
            raise ProviderNotAvailable(
                f"{self.display_name} has no published competition name configured for {league_key}. "
                f"Run competitions() to inspect what is available, then add the name to the "
                f"League.provider_names mapping."
            )
        rows = self.discover(published)
        if rows.empty:
            available = sorted(self.competitions()["competition_name"].dropna().unique())
            raise ProviderNotAvailable(
                f"{self.display_name} has no data for {league_key} (published name {published!r}). "
                f"Available competitions: {available}"
            )
        return rows

    def covers(self, league_key: str) -> bool:
        """Report whether StatsBomb publishes this competition.

        Args:
            league_key: Internal competition key.

        Returns:
            True if a manifest entry exists.
        """
        try:
            return not self.resolve_competition(league_key).empty
        except (ProviderNotAvailable, ProviderError, KeyError):
            return False

    def available_seasons(self, league_key: str) -> list[SeasonRef]:
        """List StatsBomb seasons available for a competition.

        Args:
            league_key: Internal competition key.

        Returns:
            Seasons ordered oldest to newest.

        Raises:
            ProviderNotAvailable: If the competition is not published.
        """
        rows = self.resolve_competition(league_key)
        refs: list[SeasonRef] = []
        for _, row in rows.iterrows():
            season_name = str(row.get("season_name") or "")
            start = _parse_season_name(season_name)
            refs.append(
                SeasonRef(
                    league_key=league_key,
                    season_code=str(row["season_id"]),
                    season_label=season_name or f"season_id_{row['season_id']}",
                    start_date=start,
                    match_count=int(row["match_available"]) if pd.notna(row.get("match_available")) else None,
                )
            )
        return sorted(refs, key=lambda r: r.season_label)

    def fetch_matches(self, league_key: str, season_code: str) -> pd.DataFrame:
        """Fetch match metadata for one competition season.

        Args:
            league_key: Internal competition key.
            season_code: StatsBomb ``season_id``, obtained from
                :meth:`available_seasons` rather than guessed.

        Returns:
            Canonical match frame built from the published match list.

        Raises:
            ProviderNotAvailable: If the competition is not published.
            ProviderError: If the match list cannot be parsed.
        """
        rows = self.resolve_competition(league_key)
        season_rows = rows[rows["season_id"].astype(str) == str(season_code)]
        if season_rows.empty:
            available = sorted(rows["season_id"].astype(str).unique())
            raise ProviderNotAvailable(
                f"{self.display_name}: season_id {season_code!r} not found for {league_key}. "
                f"Available season_ids: {available}"
            )

        competition_id = int(season_rows.iloc[0]["competition_id"])
        url = MATCHES_URL_TEMPLATE.format(competition_id=competition_id, season_id=season_code)
        payload = self._cache.get_or_fetch(
            self.name,
            f"matches:{league_key}:{season_code}",
            url,
            fetcher=self._http_get_json,
        )
        matches = json.loads(payload)
        return self._matches_to_canonical(
            matches,
            league_key=league_key,
            season_code=str(season_code),
            season_label=self._season_label_for(league_key, season_rows),
        )

    @staticmethod
    def _season_label_for(league_key: str, season_rows: pd.DataFrame) -> str:
        """Derive a display label from the manifest row for one season.

        Args:
            league_key: Internal competition key, used only in the fallback.
            season_rows: Manifest rows already filtered to a single season.

        Returns:
            The manifest's ``season_name`` when present, otherwise a readable
            label derived from ``season_id``.
        """
        if season_rows.empty:  # pragma: no cover - caller guarantees one row
            return league_key

        name = str(season_rows.iloc[0].get("season_name") or "").strip()
        if name:
            return name

        season_id = season_rows.iloc[0].get("season_id")
        return f"season_id_{season_id}"

    def _matches_to_canonical(
        self, matches: list[dict[str, Any]], *, league_key: str, season_code: str, season_label: str
    ) -> pd.DataFrame:
        """Map published match metadata onto the canonical schema.

        Args:
            matches: Raw match dictionaries.
            league_key: Internal competition key.
            season_code: StatsBomb ``season_id``, used as the natural key.
            season_label: Human-readable season label. This is taken from the
                manifest's ``season_name`` rather than derived from
                ``season_id``, because StatsBomb uses a calendar year for
                tournament competitions (the Africa Cup of Nations has
                ``season_id`` 2023) and a split-season code for domestic
                leagues. Only the manifest knows which convention applies.

        Returns:
            Canonical match frame with a ``statsbomb_match_id`` column preserved
            for event lookups.
        """
        league = get_league(league_key)
        records = []
        for match in matches:
            home = self._teams.resolve_or_pass_through(str(match.get("home_team", {}).get("home_team_name", "")))
            away = self._teams.resolve_or_pass_through(str(match.get("away_team", {}).get("away_team_name", "")))
            match_date = pd.to_datetime(match.get("match_date"), errors="coerce")
            home_score = match.get("home_score")
            away_score = match.get("away_score")
            result = None
            if isinstance(home_score, int) and isinstance(away_score, int):
                result = "H" if home_score > away_score else ("D" if home_score == away_score else "A")

            records.append(
                {
                    "match_id": (
                        make_match_id(league_key, match_date.date().isoformat(), home, away)
                        if pd.notna(match_date)
                        else None
                    ),
                    "source": self.name,
                    "league_key": league_key,
                    "competition_type": league.competition_type.value,
                    "season": season_label,
                    "season_code": season_code,
                    "match_date": match_date,
                    "home_team": home,
                    "away_team": away,
                    "venue": match.get("stadium", {}).get("name"),
                    "referee": match.get("referee_name"),
                    "home_goals": home_score if isinstance(home_score, int) else None,
                    "away_goals": away_score if isinstance(away_score, int) else None,
                    "result": result,
                    "result_et": match.get("competition_stage"),
                    "statsbomb_match_id": match.get("match_id"),
                    "match_week": match.get("match_week"),
                    "has_lineups": bool(match.get("home_team", {}).get("managers")),
                }
            )
        frame = pd.DataFrame(records)
        logger.info(
            "%s: loaded %d matches for %s season_id=%s",
            self.display_name,
            len(frame),
            league_key,
            season_code,
        )
        return apply_dtypes(frame, table="matches", keep_extra=True)

    def fetch_events(self, match_id: str) -> pd.DataFrame:
        """Fetch the ordered event stream for one match.

        Args:
            match_id: Canonical match identifier.

        Returns:
            Canonical event frame ordered by period, minute, second. Includes a
            ``statsbomb_match_id`` column when the frame carries one.

        Raises:
            ProviderError: If the event payload cannot be fetched or parsed.
        """
        sb_match_id = self._lookup_statsbomb_id(match_id)
        url = EVENTS_URL_TEMPLATE.format(match_id=sb_match_id)
        payload = self._cache.get_or_fetch(
            self.name, f"events:{sb_match_id}", url, fetcher=self._http_get_json
        )
        events = json.loads(payload)
        return self._events_to_canonical(events, canonical_match_id=match_id)

    def _events_to_canonical(
        self, events: list[dict[str, Any]], *, canonical_match_id: str | None
    ) -> pd.DataFrame:
        """Map published events onto the canonical schema.

        Args:
            events: Raw event dictionaries.
            canonical_match_id: Canonical match identifier to stamp on rows.

        Returns:
            Canonical event frame ordered by period, minute, second.

        Raises:
            ProviderError: If an event lacks the minimum fields (id, minute).
        """
        records: list[dict[str, Any]] = []
        for event in events:
            event_id = event.get("id")
            minute = event.get("minute")
            if event_id is None or minute is None:
                raise ProviderError(
                    f"{self.display_name}: event missing id or minute: "
                    f"{ {k: event.get(k) for k in ('id', 'type', 'minute')} }"
                )

            team_name = (event.get("team") or {}).get("name")
            team = self._teams.resolve_or_pass_through(str(team_name)) if team_name else None
            raw_type = str(event.get("type", {}).get("name", ""))
            period = int(event.get("period", 1))
            sec = int(event.get("second", 0) or 0)

            shot = event.get("shot") or {}
            outcome_name = str(shot.get("outcome", {}).get("name", "")) if shot.get("outcome") else ""
            canonical_type = SHOT_OUTCOME_MAP.get(outcome_name, EVENT_TYPE_MAP.get(raw_type, EventType.UNKNOWN))

            records.append(
                {
                    "event_id": make_event_id(
                        canonical_match_id or str(event_id), period, int(minute), sec, team or ""
                    ),
                    "match_id": canonical_match_id,
                    "statsbomb_match_id": event_id,
                    "period": period,
                    "minute": int(minute),
                    "second": sec,
                    "timestamp": event.get("timestamp"),
                    "team": team,
                    "player": (event.get("player") or {}).get("name"),
                    "event_type": canonical_type.value,
                    "raw_event_type": raw_type,
                    "xg": shot.get("statsbomb_xg"),
                    "shot_outcome": outcome_name or None,
                    "shot_body_part": (shot.get("body_part") or {}).get("name"),
                    "shot_type": (shot.get("type") or {}).get("name"),
                    "location_x": _coord(shot.get("location"), 0),
                    "location_y": _coord(shot.get("location"), 1),
                    "end_location_x": _coord(shot.get("end_location"), 0),
                    "end_location_y": _coord(shot.get("end_location"), 1),
                    "pass_length": (event.get("pass") or {}).get("length"),
                    "pass_angle": (event.get("pass") or {}).get("angle"),
                    "shot_assist": ((shot.get("assist") or {}).get("name")) if shot else None,
                    "related_event": (event.get("related_events") or [{}])[0].get("id")
                    if event.get("related_events")
                    else None,
                }
            )

        frame = pd.DataFrame(records)
        if len(frame):
            frame = frame.sort_values(["period", "minute", "second"], kind="mergesort").reset_index(drop=True)
        return apply_dtypes(frame, table="events", keep_extra=True)

    def fetch_lineups(self, match_id: str) -> pd.DataFrame:
        """Fetch full squads, including substitutes, for one match.

        Args:
            match_id: Canonical match identifier.

        Returns:
            Canonical lineup frame with one row per player per team.

        Raises:
            ProviderError: If the payload cannot be parsed.
        """
        sb_match_id = self._lookup_statsbomb_id(match_id)
        url = LINEUPS_URL_TEMPLATE.format(match_id=sb_match_id)
        payload = self._cache.get_or_fetch(
            self.name, f"lineups:{sb_match_id}", url, fetcher=self._http_get_json
        )
        lineups = json.loads(payload)

        records: list[dict[str, Any]] = []
        for team_block in lineups:
            team_name = team_block.get("team", {}).get("name")
            team = self._teams.resolve_or_pass_through(str(team_name)) if team_name else None
            formation = None
            for entry in team_block.get("lineup", []):
                if entry.get("formation"):
                    formation = entry["formation"]
            for entry in team_block.get("lineup", []):
                position = entry.get("position") or {}
                position_id = position.get("id")
                records.append(
                    {
                        "match_id": match_id,
                        "statsbomb_match_id": sb_match_id,
                        "team": team,
                        "player": (entry.get("player") or {}).get("name"),
                        "player_id": _optional_str((entry.get("player") or {}).get("id")),
                        "position": position.get("name"),
                        "position_id": position_id,
                        "jersey_number": entry.get("jersey_number"),
                        # StatsBomb flags substitutes with position id 16. This is
                        # their documented convention rather than a guarantee, so
                        # anything we cannot interpret stays null instead of
                        # defaulting to "starter".
                        "is_starter": False if position_id == 16 else (True if isinstance(position_id, int) else None),
                        # StatsBomb open data does NOT publish a captain flag.
                        # Left null on purpose: claiming False would be inventing
                        # a fact, and downstream code must handle the gap.
                        "is_captain": None,
                        "formation": formation,
                        "minutes_played": entry.get("minutes_played"),
                    }
                )
        return apply_dtypes(pd.DataFrame(records), table="lineups", keep_extra=True)

    def _lookup_statsbomb_id(self, match_id: str) -> str:
        """Translate a canonical match id to a provider match id.

        Args:
            match_id: Canonical match identifier.

        Returns:
            The provider's own match id.

        Raises:
            ProviderNotAvailable: If the match is not in any fetched manifest.
        """
        sb_id = getattr(self, "_id_overrides", {}).get(match_id)
        if sb_id:
            return str(sb_id)
        raise ProviderNotAvailable(
            f"{self.display_name}: no provider match id known for canonical id {match_id!r}. "
            f"Fetch the season first via fetch_matches() so the mapping is populated."
        )

    def register_match_ids(self, matches: pd.DataFrame) -> None:
        """Remember the canonical-to-provider match id mapping from a season fetch.

        Args:
            matches: A frame returned by :meth:`fetch_matches`.
        """
        if "statsbomb_match_id" not in matches.columns or "match_id" not in matches.columns:
            return
        mapping: dict[str, str] = {}
        for _, row in matches.iterrows():
            if pd.notna(row.get("match_id")) and pd.notna(row.get("statsbomb_match_id")):
                mapping[str(row["match_id"])] = str(row["statsbomb_match_id"])
        overrides = getattr(self, "_id_overrides", None)
        if overrides is None:
            overrides = {}
            self._id_overrides = overrides
        overrides.update(mapping)

    # --- transport ----------------------------------------------------------

    def _http_get_json(self, url: str) -> bytes:
        """Fetch a JSON document as bytes.

        Args:
            url: Absolute URL.

        Returns:
            Raw response bytes.

        Raises:
            ProviderError: If the request fails.
        """
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - hard dependency
            raise ProviderError("requests is required for network access") from exc

        session = self._session
        owned = False
        if session is None:
            session = requests.Session()
            owned = True
        try:
            response = session.get(url, timeout=self._settings.http_timeout, headers={"User-Agent": HTTP_USER_AGENT})
            if response.status_code != 200:
                raise ProviderError(f"{self.display_name}: HTTP {response.status_code} for {url}")
            return response.content
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"{self.display_name}: request to {url} failed: {exc}") from exc
        finally:
            if owned:
                session.close()


def _optional_str(value: Any) -> str | None:
    """Coerce an identifier to text without inventing a value.

    Args:
        value: Raw value from a published record.

    Returns:
        The text form, or ``None`` when absent. A falsy id of ``0`` is treated
        as absent, since no provider issues identifier zero.
    """
    if value is None or value == "":
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):  # pragma: no cover - non-scalar input
        return None
    text = str(value)
    return None if text == "0" else text


def _coord(location: Any, index: int) -> float | None:
    """Read one axis out of a StatsBomb location triple.

    StatsBomb publishes pitch coordinates as ``[x, y, z]``. Reading the
    position of a shot from the enclosing event is wrong: a shot's location
    lives under ``shot.location``, and the event itself has none, so the value
    silently came back null.

    Args:
        location: The published triple, or ``None``.
        index: Axis to read; 0 is x, 1 is y.

    Returns:
        The coordinate, or ``None`` when absent or not numeric.
    """
    if not isinstance(location, (list, tuple)) or len(location) <= index:
        return None
    value = location[index]
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _parse_season_name(season_name: str) -> date | None:
    """Parse a published season name into an approximate start date.

    Args:
        season_name: Name as published, e.g. ``"2015/2016"``.

    Returns:
        Approximate season start date, or ``None`` if unparseable.
    """
    digits = season_name.split("/")[0].strip()
    if len(digits) >= 4 and digits[:4].isdigit():
        year = int(digits[:4])
        # A "2015/2016" style label means the season began in August 2015.
        return date(year, 8, 1)
    return None


__all__ = [
    "COMPETITIONS_URL",
    "EVENTS_URL_TEMPLATE",
    "LINEUPS_URL_TEMPLATE",
    "MATCHES_URL_TEMPLATE",
    "REPO_BASE",
    "StatsBombOpenProvider",
]

"""football-data.co.uk adapter.

The richest free source of long-run match history *and* bookmaker odds, which
makes it the backbone of our pre-match work and the source of the market
baseline we have to beat.

LICENCE WARNING -- READ BEFORE CHANGING ANYTHING HERE
=====================================================
This provider publishes **no licence** and its operator states he holds
copyright in official league match data. Verified 2026-10-02. Consequences,
enforced in code by :mod:`predict_football.config.licences`:

* Raw CSVs are cached locally and **gitignored**. Never committed.
* A public deployment may not serve this data without written permission.
* Derived analysis is for personal and internal use.

The licence-clean alternative for anything shipped publicly is openfootball
(CC0) for results and Wyscout soccer-logs (CC BY 4.0) for events.
"""

from __future__ import annotations

import csv
import io
import logging
from datetime import date
from pathlib import Path

import pandas as pd

from predict_football.config.leagues import get_league, season_label
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
from predict_football.data.schema import apply_dtypes
from predict_football.data.teams import TeamResolver

logger = logging.getLogger(__name__)

#: CSV download pattern. Season code is the two-digit start year.
CSV_URL_TEMPLATE = "https://www.football-data.co.uk/mmz4281/{season}/{league_code}.csv"

#: The site publishes English club names with Spanish/French/German accents in
#: Latin-1. Decode permissively so a single odd byte cannot fail a whole season.
_ENCODINGS = ("utf-8", "cp1252", "latin-1")

#: Earliest season available, per competition code. football-data.co.uk
#: coverage is uneven: the Premier League reaches back to 1993/94, several other
#: leagues stop far later. Requesting an unavailable season gets a 404, which
#: we surface as a clear message rather than an opaque error.
def _season_resource(league_key: str, season_code: str) -> str:
    """Build the cache key for one season document.

    Results and odds come from the *same* season CSV, so both accessors must
    use one key. Giving them separate keys downloaded and stored the identical
    file twice, doubling network and disk use for no benefit.

    Args:
        league_key: Internal competition key.
        season_code: Four-character season code.

    Returns:
        A cache resource key.
    """
    return f"season:{league_key}:{season_code}"


_FIRST_SEASON_START_YEAR: dict[str, int] = {
    "E0": 1993,
    "E1": 1993,
    "E2": 2004,
    "EC": 2015,
    "SC0": 1998,
    "SC1": 1998,
    "D1": 1993,
    "D2": 1993,
    "I1": 1993,
    "I2": 1993,
    "SP1": 1993,
    "SP2": 1993,
    "F1": 1993,
    "F2": 2003,
    "N1": 1993,
    "P1": 1993,
    "T1": 1993,
}

#: Canonical match column -> provider column. Only columns the provider
#: actually publishes are mapped; anything absent stays null. Per-season column
#: sets differ, so every lookup is tolerant of absence.
MATCH_FIELD_MAP: dict[str, str] = {
    "home_team": "HomeTeam",
    "away_team": "AwayTeam",
    "home_goals": "FTHG",
    "away_goals": "FTAG",
    "result": "FTR",
    "home_goals_ht": "HTHG",
    "away_goals_ht": "HTAG",
    "match_time": "Time",
    "referee": "Referee",
    "home_shots": "HS",
    "away_shots": "AS",
    "home_shots_on_target": "HST",
    "away_shots_on_target": "AST",
    "home_corners": "HC",
    "away_corners": "AC",
    "home_fouls": "HF",
    "away_fouls": "AF",
    "home_yellow": "HY",
    "away_yellow": "AY",
    "home_red": "HR",
    "away_red": "AR",
    "home_xg": "HxG",
    "away_xg": "AxG",
}

#: Bookmaker columns we ingest, per bookmaker. The site publishes opening odds
#: without a marker and closing odds with "C" inserted before the outcome letter
#: (B365CH is bet365 closing home). "Avg" is the market consensus across
#: bookmakers, which is the more honest baseline than any single book.
ODDS_FIELD_MAP: dict[str, dict[str, str]] = {
    "avg": {
        "odds_home_open": "AvgH",
        "odds_draw_open": "AvgD",
        "odds_away_open": "AvgA",
        "odds_home_close": "AvgCH",
        "odds_draw_close": "AvgCD",
        "odds_away_close": "AvgCA",
    },
    "b365": {
        "odds_home_open": "B365H",
        "odds_draw_open": "B365D",
        "odds_away_open": "B365A",
        "odds_home_close": "B365CH",
        "odds_draw_close": "B365CD",
        "odds_away_close": "B365CA",
    },
    "max": {
        "odds_home_open": "MaxH",
        "odds_draw_open": "MaxD",
        "odds_away_open": "MaxA",
        "odds_home_close": "MaxCH",
        "odds_draw_close": "MaxCD",
        "odds_away_close": "MaxCA",
    },
    "ps": {
        "odds_home_open": "PSH",
        "odds_draw_open": "PSD",
        "odds_away_open": "PSA",
        "odds_home_close": "PSCH",
        "odds_draw_close": "PSCD",
        "odds_away_close": "PSCA",
    },
}

#: Over/under 2.5 goal columns, stored in a separate market row.
OVER_UNDER_MAP: dict[str, str] = {
    "avg": {"over": "Avg>2.5", "under": "Avg<2.5", "over_close": "AvgC>2.5", "under_close": "AvgC<2.5"},
    "b365": {"over": "B365>2.5", "under": "B365<2.5", "over_close": "B365C>2.5", "under_close": "B365C<2.5"},
}

#: Columns that mark the file as an odds-carrying result file.
_REQUIRED_SOURCE_COLUMNS = ("HomeTeam", "AwayTeam")


class FootballDataCoProvider(DataProvider):
    """Adapter for football-data.co.uk season CSV files.

    Args:
        settings: Resolved project settings.
        cache: Raw download cache.
        teams: Team name resolver.
        session: Optional requests session, injected for testability.

    Raises:
        ProviderError: If the provider is constructed with no licence review on
            record. This is a guard, not a formality: it means the data layer
            cannot be used for a source nobody has checked.
    """

    name = "football_data_co"
    display_name = "football-data.co.uk"

    def __init__(
        self,
        settings: Settings | None = None,
        cache: RawCache | None = None,
        teams: TeamResolver | None = None,
        session: object | None = None,
    ) -> None:
        # Fails loudly if nobody reviewed the terms.
        licence_for(self.name)
        self._settings = settings or get_settings()
        self._cache = cache or RawCache(self._settings)
        self._teams = teams or TeamResolver()
        self._session = session

    # --- coverage -----------------------------------------------------------

    def covers(self, league_key: str) -> bool:
        """Report whether football-data.co.uk carries this competition.

        Args:
            league_key: Internal competition key.

        Returns:
            True if a provider-specific code exists.
        """
        try:
            league = get_league(league_key)
        except KeyError:
            return False
        return league.code_for(self.name) is not None

    def available_seasons(self, league_key: str) -> list[SeasonRef]:
        """List available seasons for a competition.

        Args:
            league_key: Internal competition key, e.g. ``"ENG_PL"``.

        Returns:
            Seasons from the earliest published season to the current one.

        Raises:
            ProviderNotAvailable: If the competition is not covered.
        """
        league = get_league(league_key)
        code = league.code_for(self.name)
        if code is None:
            raise ProviderNotAvailable(
                f"{self.display_name} has no code configured for {league_key}. "
                f"Known: {sorted(_FIRST_SEASON_START_YEAR)}"
            )
        first = _FIRST_SEASON_START_YEAR.get(code, 2000)
        current = self._current_season_start_year()
        return [
            SeasonRef(
                league_key=league_key,
                season_code=f"{y % 100:02d}{(y + 1) % 100:02d}",
                season_label=season_label(f"{y % 100:02d}{(y + 1) % 100:02d}"),
            )
            for y in range(first, current + 1)
        ]

    @staticmethod
    def _current_season_start_year() -> int:
        """Return the starting year of the current European season.

        Returns:
            2025 for, say, 2026-03 (i.e. the 2025/26 season is in progress).

        Examples:
            >>> FootballDataCoProvider._current_season_start_year() >= 2024
            True
        """
        today = date.today()
        # European seasons start in August, so January-July still belongs to
        # the season that began the previous calendar year.
        return today.year if today.month >= 8 else today.year - 1

    # --- fetching -----------------------------------------------------------

    def _url_for(self, league_key: str, season_code: str) -> str:
        """Build the CSV download URL for a competition season.

        Args:
            league_key: Internal competition key.
            season_code: Four-character season code.

        Returns:
            The download URL.

        Raises:
            ProviderNotAvailable: If the competition has no provider code.
        """
        code = get_league(league_key).code_for(self.name)
        if code is None:
            raise ProviderNotAvailable(f"{self.display_name} does not cover {league_key}")
        return CSV_URL_TEMPLATE.format(season=season_code, league_code=code)

    def fetch_matches(self, league_key: str, season_code: str) -> pd.DataFrame:
        """Fetch and normalise one season of match results.

        Args:
            league_key: Internal competition key.
            season_code: Four-character season code, e.g. ``"2425"``.

        Returns:
            Canonical match frame. Matches not yet played carry a null
            ``result`` and are kept, because the current season file legitimately
            contains future fixtures.

        Raises:
            ProviderError: If the download fails or the file is unparseable.
        """
        get_league(league_key)  # raises for an unknown competition key
        url = self._url_for(league_key, season_code)
        payload = self._cache.get_or_fetch(
            self.name, _season_resource(league_key, season_code), url, fetcher=self._http_get
        )
        self._archive(payload, league_key, season_code)

        raw = self._decode_csv(payload)
        missing = [c for c in _REQUIRED_SOURCE_COLUMNS if c not in raw.columns]
        if missing:
            raise ProviderError(
                f"{self.display_name} {league_key} {season_code}: expected columns {missing} "
                f"absent. Got {list(raw.columns)[:12]}. The file layout may have changed."
            )

        return self._to_canonical(raw, league_key=league_key, season_code=season_code)

    def _decode_csv(self, payload: bytes) -> pd.DataFrame:
        """Decode a CSV payload, tolerating encoding differences.

        Args:
            payload: Raw CSV bytes.

        Returns:
            Parsed DataFrame with all columns as strings, so numeric coercion is
            explicit and inspectable rather than implicit in the reader.

        Raises:
            ProviderError: If no known encoding parses the payload.
        """
        for encoding in _ENCODINGS:
            try:
                text = payload.decode(encoding)
            except UnicodeDecodeError:
                continue
            try:
                return pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=True)
            except pd.errors.EmptyDataError as exc:
                raise ProviderError(f"{self.display_name}: empty CSV payload") from exc
            except (pd.errors.ParserError, csv.Error) as exc:
                raise ProviderError(f"{self.display_name}: CSV parse failed using {encoding}: {exc}") from exc
        raise ProviderError(
            f"{self.display_name}: could not decode payload with any of {_ENCODINGS}. "
            f"This suggests the response was not CSV (an error page, perhaps)."
        )

    def _to_canonical(self, raw: pd.DataFrame, *, league_key: str, season_code: str) -> pd.DataFrame:
        """Map raw provider columns onto the canonical match schema.

        Args:
            raw: Source frame as published.
            league_key: Internal competition key.
            season_code: Four-character season code.

        Returns:
            Canonical match frame.

        Raises:
            ProviderError: If the date column is missing or wholly unparseable,
                since without dates no time-ordered work is possible.
        """
        league = get_league(league_key)
        out = pd.DataFrame()

        for canonical, source in MATCH_FIELD_MAP.items():
            out[canonical] = raw[source] if source in raw.columns else pd.NA

        if "Date" not in raw.columns:
            raise ProviderError(f"{self.display_name}: no Date column for {league_key} {season_code}")
        # football-data.co.uk writes DD/MM/YYYY; dayfirst avoids the US default
        # silently shifting every autumn date by months.
        dates = pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce", format="mixed")
        if dates.isna().all():
            raise ProviderError(
                f"{self.display_name}: every Date value failed to parse for "
                f"{league_key} {season_code}. Sample: {raw['Date'].head(3).tolist()}"
            )
        out["match_date"] = dates

        out["source"] = self.name
        out["league_key"] = league_key
        out["competition_type"] = league.competition_type.value
        out["season"] = season_label(season_code)
        out["season_code"] = season_code

        out["home_team"] = [self._teams.resolve_or_pass_through(x) for x in raw["HomeTeam"]]
        out["away_team"] = [self._teams.resolve_or_pass_through(x) for x in raw["AwayTeam"]]

        out["match_id"] = [
            make_match_id(league_key, d.date().isoformat(), h, a)
            for d, h, a in zip(dates, out["home_team"], out["away_team"], strict=False)
        ]

        # Recompute the result rather than trusting the provider's label: a
        # single mislabelled row would otherwise silently corrupt the target.
        home_goals = pd.to_numeric(out["home_goals"], errors="coerce")
        away_goals = pd.to_numeric(out["away_goals"], errors="coerce")
        derived = pd.Series(pd.NA, index=out.index, dtype="string")
        derived = derived.mask(home_goals > away_goals, "H").mask(home_goals == away_goals, "D").mask(
            home_goals < away_goals, "A"
        )
        provider_label = out["result"].astype("string").str.strip().str.upper()
        disagree = derived.notna() & provider_label.notna() & (derived != provider_label)
        if disagree.any():
            logger.warning(
                "%d row(s) in %s %s disagree between goals and the published result; "
                "trusting the scoreline. Example match_id=%s",
                int(disagree.sum()),
                league_key,
                season_code,
                out.loc[disagree, "match_id"].iloc[0],
            )
        out["result"] = derived.where(derived.notna(), provider_label)
        out["home_goals"] = home_goals
        out["away_goals"] = away_goals

        # The season file contains future fixtures with blank scorelines.
        out.loc[out["result"].isna(), ["home_goals", "away_goals"]] = pd.NA

        # Return canonical dtypes, not the raw strings pandas read from CSV, so
        # the provider contract holds and downstream code never has to guess.
        return apply_dtypes(out, table="matches", keep_extra=True)

    def fetch_odds(self, league_key: str, season_code: str) -> pd.DataFrame:
        """Fetch bookmaker odds for one season in long format.

        Args:
            league_key: Internal competition key.
            season_code: Four-character season code.

        Returns:
            One row per match and bookmaker with opening and closing prices. Rows
            where the provider published no odds have null prices.

        Raises:
            ProviderError: If the download fails.
        """
        url = self._url_for(league_key, season_code)
        payload = self._cache.get_or_fetch(
            self.name, _season_resource(league_key, season_code), url, fetcher=self._http_get
        )
        raw = self._decode_csv(payload)
        return self._odds_to_canonical(raw, league_key=league_key, season_code=season_code)

    def _odds_to_canonical(self, raw: pd.DataFrame, *, league_key: str, season_code: str) -> pd.DataFrame:
        """Map raw odds columns onto the canonical long odds schema.

        Args:
            raw: Source frame as published.
            league_key: Internal competition key.
            season_code: Four-character season code.

        Returns:
            Long-format odds frame joined to matches on ``match_id``.

        Raises:
            ProviderError: If the date or team columns are absent.
        """
        if "Date" not in raw.columns or "HomeTeam" not in raw.columns:
            raise ProviderError(f"{self.display_name}: odds payload missing Date/HomeTeam columns")

        dates = pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce", format="mixed")
        home = [self._teams.resolve_or_pass_through(x) for x in raw["HomeTeam"]]
        away = [self._teams.resolve_or_pass_through(x) for x in raw["AwayTeam"]]
        match_ids = [
            make_match_id(league_key, d.date().isoformat(), h, a) if pd.notna(d) else None
            for d, h, a in zip(dates, home, away, strict=False)
        ]

        rows: list[dict[str, object]] = []
        for bookmaker, mapping in ODDS_FIELD_MAP.items():
            frame = pd.DataFrame({"match_id": match_ids, "bookmaker": bookmaker, "market": "1x2"})
            for canonical, source in mapping.items():
                frame[canonical] = (
                    pd.to_numeric(raw[source], errors="coerce") if source in raw.columns else pd.NA
                )
            rows.extend(frame.to_dict("records"))

        odds = pd.DataFrame(rows)
        if len(odds):
            odds = self._add_overround(odds)
        return odds

    @staticmethod
    def _add_overround(odds: pd.DataFrame) -> pd.DataFrame:
        """Compute the bookmaker margin for each closing 1X2 price set.

        The overround is the sum of raw implied probabilities ``1/odds``. It
        exceeds 1.0 because the bookmaker builds in a margin, which is why raw
        implied probabilities must be normalised before being used as a
        baseline. Where the provider published no closing price we fall back to
        opening prices, and record which was used.

        Args:
            odds: Long odds frame with closing and opening price columns.

        Returns:
            The frame with ``overround`` and ``price_basis`` columns added.
        """
        closing = odds[["odds_home_close", "odds_draw_close", "odds_away_close"]]
        opening = odds[["odds_home_open", "odds_draw_open", "odds_away_open"]]
        has_closing = closing.notna().all(axis=1)
        has_opening = opening.notna().all(axis=1)

        implied_closing = closing.pow(-1).sum(axis=1, min_count=3)
        implied_opening = opening.pow(-1).sum(axis=1, min_count=3)

        odds["overround"] = implied_closing.where(has_closing, implied_opening)
        odds["price_basis"] = pd.Series(
            pd.NA, index=odds.index, dtype="string"
        ).mask(has_closing, "close").mask(~has_closing & has_opening, "open")
        return odds

    # --- transport ----------------------------------------------------------

    def _http_get(self, url: str) -> bytes:
        """Fetch a URL as bytes with retry and timeout.

        Args:
            url: Absolute URL to fetch.

        Returns:
            Response body.

        Raises:
            ProviderError: If the request fails or returns an error status.
        """
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - requests is a hard dependency
            raise ProviderError("requests is required for network access") from exc

        session = self._session
        owned = False
        if session is None:
            session = requests.Session()
            owned = True
        try:
            response = session.get(
                url,
                timeout=self._settings.http_timeout,
                headers={"User-Agent": _user_agent()},
            )
            if response.status_code != 200:
                raise ProviderError(
                    f"{self.display_name}: HTTP {response.status_code} for {url}. "
                    f"A 404 usually means the season is not published for this competition."
                )
            return response.content
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"{self.display_name}: request to {url} failed: {exc}") from exc
        finally:
            if owned:
                session.close()

    def _archive(self, payload: bytes, league_key: str, season_code: str) -> Path:
        """Write a verbatim copy of a download into ``data/raw``.

        The cache is for programmatic replay; this tree is the archival copy of
        record a human can inspect when a number looks wrong.

        Args:
            payload: Raw downloaded bytes.
            league_key: Internal competition key.
            season_code: Four-character season code.

        Returns:
            Path to the archived file.
        """
        code = get_league(league_key).code_for(self.name) or league_key
        target_dir = self._settings.raw_dir / self.name / code
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{code}_{season_code}.csv"
        if not path.exists():
            path.write_bytes(payload)
        return path


def _user_agent() -> str:
    """Return the identifying user agent used for all outbound requests.

    Returns:
        The project user agent string.
    """
    from predict_football.config.settings import HTTP_USER_AGENT

    return HTTP_USER_AGENT

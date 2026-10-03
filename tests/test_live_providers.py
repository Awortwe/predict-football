"""Tests for the live JSON providers and the season-code helper. All offline.

Both adapters are exercised through :class:`FakeSession` so no request leaves the
machine. The payloads are synthetic and obviously fake (id 111, "Ref One"), which
matters because the real responses are covered by licences that forbid
redistribution.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
from tests.conftest import FakeSession

from predict_football.config.leagues import season_start_year
from predict_football.data.providers.api_football import (
    API_KEY_ENV as API_FOOTBALL_KEY_ENV,
)
from predict_football.data.providers.api_football import (
    ApiFootballProvider,
)
from predict_football.data.providers.base import ProviderError, ProviderNotAvailable, SeasonRef
from predict_football.data.providers.football_data_org import (
    API_KEY_ENV as FDORG_KEY_ENV,
)
from predict_football.data.providers.football_data_org import (
    FootballDataOrgProvider,
)
from predict_football.data.providers.registry import get_provider, list_providers


def _json_bytes(document: object) -> bytes:
    """Serialise a fake payload the way the real API would.

    Args:
        document: JSON-serialisable object.

    Returns:
        UTF-8 encoded JSON bytes.
    """
    return json.dumps(document).encode("utf-8")


FDORG_PAYLOAD = {
    "matches": [
        {
            "id": 111,
            "utcDate": "2024-08-16T19:00:00Z",
            "status": "FINISHED",
            "matchday": 1,
            "homeTeam": {"name": "Liverpool FC"},
            "awayTeam": {"name": "Ipswich Town"},
            "score": {
                "fullTime": {"home": 2, "away": 0},
                "halfTime": {"home": 1, "away": 0},
                "penalties": None,
            },
            "venue": "Anfield",
            "referees": [{"name": "Ref One"}],
        },
        {
            "id": 112,
            "utcDate": "2024-08-17T14:00:00Z",
            "status": "SCHEDULED",
            "matchday": 1,
            "homeTeam": {"name": "Arsenal FC"},
            "awayTeam": {"name": "Wolverhampton Wanderers FC"},
            "score": {
                "fullTime": {"home": None, "away": None},
                "halfTime": {"home": None, "away": None},
            },
            "venue": "Emirates Stadium",
            "referees": [],
        },
        {
            "id": 113,
            "utcDate": "2024-08-18T15:30:00Z",
            "status": "FINISHED",
            "matchday": 2,
            "homeTeam": {"name": "Manchester City FC"},
            "awayTeam": {"name": "Chelsea FC"},
            "score": {"fullTime": {"home": 1, "away": 1}, "halfTime": {"home": 0, "away": 1}},
            "venue": None,
            "referees": None,
        },
    ]
}


@pytest.fixture
def fdorg_session() -> FakeSession:
    """Fake session serving the synthetic football-data.org payload."""
    return FakeSession({"competitions/PL/matches": _json_bytes(FDORG_PAYLOAD)})


@pytest.fixture
def fdorg_provider(settings, fdorg_session, resolver) -> FootballDataOrgProvider:
    """football-data.org provider wired to a fake session and a known token."""
    return FootballDataOrgProvider(
        settings=settings, teams=resolver, session=fdorg_session, api_key="secret-token"
    )


# --- registry ---------------------------------------------------------------


def test_registry_includes_the_live_providers() -> None:
    """Both live adapters must be discoverable by their licence key."""
    names = list_providers()
    assert "football_data_org" in names
    assert "api_football" in names


def test_registry_constructs_the_live_providers() -> None:
    """Name-based construction must resolve to the right classes."""
    assert isinstance(get_provider("football_data_org"), FootballDataOrgProvider)
    assert isinstance(get_provider("api_football"), ApiFootballProvider)


# --- season code helper -----------------------------------------------------


def test_season_start_year_reads_split_and_annual_codes() -> None:
    """Split seasons map to their start year; annual codes map to themselves."""
    assert season_start_year("2425") == 2024
    assert season_start_year("9394") == 1993
    assert season_start_year("2022") == 2022


def test_season_start_year_treats_2021_as_a_split_season() -> None:
    """The ambiguous code stays consistent with season_label."""
    assert season_start_year("2021") == 2020


def test_season_start_year_rejects_nonsense() -> None:
    """A malformed code must fail loudly rather than pick a year."""
    with pytest.raises(ValueError):
        season_start_year("24")
    with pytest.raises(ValueError):
        season_start_year("abcd")


# --- football-data.org ------------------------------------------------------


def test_fdorg_maps_results_fixtures_and_half_time(fdorg_provider) -> None:
    """Finished scores, half-time scores and results must all be populated."""
    frame = fdorg_provider.fetch_matches("ENG_PL", "2425")

    assert len(frame) == 3
    assert list(frame["season"]) == ["2024/25"] * 3
    assert list(frame["season_code"]) == ["2425"] * 3
    assert list(frame["league_key"]) == ["ENG_PL"] * 3

    finished = frame[frame["football_data_org_id"] == 111].iloc[0]
    assert finished["result"] == "H"
    assert (finished["home_goals"], finished["away_goals"]) == (2, 0)
    assert (finished["home_goals_ht"], finished["away_goals_ht"]) == (1, 0)

    draw = frame[frame["football_data_org_id"] == 113].iloc[0]
    assert draw["result"] == "D"
    assert draw["venue"] is pd.NA or pd.isna(draw["venue"])


def test_fdorg_leaves_unplayed_fixtures_unknown_not_zero(fdorg_provider) -> None:
    """An unplayed match must have a null result and null goals, never 0."""
    frame = fdorg_provider.fetch_matches("ENG_PL", "2425")
    pending = frame[frame["football_data_org_id"] == 112].iloc[0]

    assert pd.isna(pending["result"])
    assert pd.isna(pending["home_goals"])
    assert pd.isna(pending["away_goals"])
    assert pd.isna(pending["home_goals_ht"])


def test_fdorg_resolves_team_aliases(fdorg_provider) -> None:
    """Published spellings must collapse onto canonical names."""
    frame = fdorg_provider.fetch_matches("ENG_PL", "2425")
    teams = set(frame["home_team"]) | set(frame["away_team"])
    assert "Liverpool" in teams
    assert "Wolverhampton Wanderers" in teams
    assert "Liverpool FC" not in teams


def test_fdorg_builds_stable_unique_match_ids(fdorg_provider) -> None:
    """Identity must be unique per row and reproducible across calls."""
    frame = fdorg_provider.fetch_matches("ENG_PL", "2425")
    assert frame["match_id"].is_unique
    again = fdorg_provider.fetch_matches("ENG_PL", "2425")
    assert list(frame["match_id"]) == list(again["match_id"])


def test_fdorg_sends_the_token_and_requests_the_season_year(fdorg_provider, fdorg_session) -> None:
    """The season must reach the query and the token must ride in the header."""
    fdorg_provider.fetch_matches("ENG_PL", "2425")
    call = fdorg_session.calls[0]
    assert call["headers"]["X-Auth-Token"] == "secret-token"
    assert call["params"]["season"] == 2024


def test_fdorg_rejects_an_unmapped_competition(fdorg_provider) -> None:
    """A competition with no provider code must not silently fetch nothing."""
    assert fdorg_provider.covers("INT_AFCON") is False
    with pytest.raises(ProviderNotAvailable):
        fdorg_provider.fetch_matches("INT_AFCON", "2425")


def test_fdorg_requires_a_token(settings, resolver) -> None:
    """A missing token must fail before any request is attempted."""
    provider = FootballDataOrgProvider(
        settings=settings, teams=resolver, session=FakeSession({}), api_key=""
    )
    with pytest.raises(ProviderError, match="requires an API token"):
        provider.fetch_matches("ENG_PL", "2425")


def test_fdorg_surfaces_http_failures(settings, resolver) -> None:
    """A non-200 must become a provider error, not a JSON decode error."""
    session = FakeSession({"competitions/PL/matches": 500})
    provider = FootballDataOrgProvider(settings=settings, teams=resolver, session=session, api_key="k")
    with pytest.raises(ProviderError, match="HTTP 500"):
        provider.fetch_matches("ENG_PL", "2425")


def test_fdorg_rejects_a_malformed_document(settings, resolver) -> None:
    """A payload without a matches list must be reported clearly."""
    session = FakeSession({"competitions/PL/matches": _json_bytes({"unexpected": []})})
    provider = FootballDataOrgProvider(settings=settings, teams=resolver, session=session, api_key="k")
    with pytest.raises(ProviderError, match="no 'matches' list"):
        provider.fetch_matches("ENG_PL", "2425")


def test_fdorg_reports_available_seasons(fdorg_provider) -> None:
    """Seasons must be ordered and include the requested split season."""
    refs = fdorg_provider.available_seasons("ENG_PL")
    codes = [ref.season_code for ref in refs]
    assert codes == sorted(codes)
    assert "2425" in codes
    assert season_start_year(refs[-1].season_code) >= 2025


def test_fdorg_does_not_claim_to_publish_odds(fdorg_provider) -> None:
    """The free tier has no odds, so the base class refusal must stand."""
    with pytest.raises(ProviderNotAvailable, match="does not provide bookmaker odds"):
        fdorg_provider.fetch_odds("ENG_PL", "2425")


# --- API-Football -----------------------------------------------------------

APIF_LEAGUES = {
    "response": [
        {
            "league": {"id": 39, "name": "Premier League", "type": "League"},
            "country": {"name": "England"},
            "seasons": [
                {"year": 2023, "start": "2023-08-11", "end": "2024-05-19"},
                {"year": 2024, "start": "2024-08-16", "end": "2025-05-25"},
            ],
        },
        {
            "league": {"id": 71, "name": "Serie A", "type": "League"},
            "country": {"name": "Brazil"},
            "seasons": [{"year": 2024, "start": "2024-01-01", "end": "2024-12-08"}],
        },
        {
            "league": {"id": 135, "name": "Serie A", "type": "League"},
            "country": {"name": "Italy"},
            "seasons": [{"year": 2024, "start": "2024-08-17", "end": "2025-05-25"}],
        },
    ]
}

APIF_FIXTURES = {
    "response": [
        {
            "fixture": {
                "id": 9001,
                "date": "2024-08-16T19:00:00+00:00",
                "venue": {"name": "Anfield"},
                "status": {"long": "Match Finished", "short": "FT"},
                "referee": "Ref One",
            },
            "league": {"id": 39, "season": 2024, "round": "Regular Season - 1"},
            "teams": {"home": {"id": 40, "name": "Liverpool"}, "away": {"id": 41, "name": "Ipswich Town"}},
            "goals": {"home": 2, "away": 0},
            "score": {
                "halftime": {"home": 1, "away": 0},
                "fulltime": {"home": 2, "away": 0},
                "penalty": {"home": None, "away": None},
            },
        },
        {
            "fixture": {
                "id": 9002,
                "date": "2024-08-25T15:30:00+00:00",
                "venue": {"name": "Etihad Stadium"},
                "status": {"long": "Not Started", "short": "NS"},
                "referee": None,
            },
            "league": {"id": 39, "season": 2024, "round": "Regular Season - 2"},
            "teams": {"home": {"id": 50, "name": "Manchester City"}, "away": {"id": 49, "name": "Chelsea"}},
            "goals": {"home": None, "away": None},
            "score": {"halftime": {"home": None, "away": None}, "fulltime": {"home": None, "away": None}},
        },
    ]
}


@pytest.fixture
def apif_session() -> FakeSession:
    """Fake session serving the synthetic API-Football manifest and fixtures."""
    return FakeSession(
        {
            "api-sports.io/leagues": _json_bytes(APIF_LEAGUES),
            "api-sports.io/fixtures": _json_bytes(APIF_FIXTURES),
        }
    )


@pytest.fixture
def apif_provider(settings, apif_session, resolver) -> ApiFootballProvider:
    """API-Football provider wired to a fake session and a known key."""
    return ApiFootballProvider(
        settings=settings, teams=resolver, session=apif_session, api_key="apif-key"
    )


def test_apif_discovers_the_numeric_league_id_by_name_and_country(apif_provider, apif_session) -> None:
    """The id must come from the manifest, matched on name *and* country."""
    apif_provider.fetch_matches("ENG_PL", "2425")
    fixtures_call = next(call for call in apif_session.calls if call["url"].endswith("/fixtures"))
    assert fixtures_call["params"]["league"] == 39
    assert fixtures_call["params"]["season"] == 2024
    assert fixtures_call["headers"]["x-apisports-key"] == "apif-key"


def test_apif_disambiguates_same_named_leagues_by_country(apif_provider) -> None:
    """Italian Serie A must not resolve to the Brazilian league id 71."""
    frame = apif_provider.fetch_matches("ITA_SERIE_A", "2425")
    assert frame["api_football_id"].tolist() == [9001, 9002]


def test_apif_maps_finished_and_upcoming_fixtures(apif_provider) -> None:
    """A finished fixture is scored; an unstarted one stays unknown."""
    frame = apif_provider.fetch_matches("ENG_PL", "2425")

    played = frame[frame["api_football_id"] == 9001].iloc[0]
    assert played["result"] == "H"
    assert (played["home_goals"], played["away_goals"]) == (2, 0)
    assert (played["home_goals_ht"], played["away_goals_ht"]) == (1, 0)
    assert played["match_week"] == 1

    upcoming = frame[frame["api_football_id"] == 9002].iloc[0]
    assert pd.isna(upcoming["result"])
    assert pd.isna(upcoming["home_goals"])
    assert upcoming["match_week"] == 2


def test_apif_available_seasons_come_from_the_manifest(apif_provider) -> None:
    """Seasons must be the ones the manifest offers, oldest first."""
    refs = apif_provider.available_seasons("ENG_PL")
    assert [ref.season_code for ref in refs] == ["2324", "2425"]
    assert refs[-1].season_label == "2024/25"


def test_apif_raises_when_no_league_matches(settings, resolver) -> None:
    """An unmatched competition must raise rather than guess an id."""
    session = FakeSession(
        {
            "api-sports.io/leagues": _json_bytes(
                {
                    "response": [
                        {
                            "league": {"id": 71, "name": "Serie A", "type": "League"},
                            "country": {"name": "Brazil"},
                            "seasons": [{"year": 2024}],
                        }
                    ]
                }
            )
        }
    )
    provider = ApiFootballProvider(settings=settings, teams=resolver, session=session, api_key="k")
    with pytest.raises(ProviderNotAvailable, match="could not match ENG_PL"):
        provider.fetch_matches("ENG_PL", "2425")


def test_apif_requires_a_key(settings, resolver) -> None:
    """A missing key must fail before any request is attempted."""
    provider = ApiFootballProvider(settings=settings, teams=resolver, session=FakeSession({}), api_key="")
    with pytest.raises(ProviderError, match="requires an API key"):
        provider.fetch_matches("ENG_PL", "2425")


def test_apif_covers_only_configured_competitions(apif_provider) -> None:
    """Coverage is the project's declaration, not a network guess."""
    assert apif_provider.covers("ENG_PL") is True
    assert apif_provider.covers("ITA_SERIE_A") is True
    assert apif_provider.covers("INT_AFCON") is False


def test_live_provider_key_env_names_are_stable() -> None:
    """The documented env var names must not drift from the code."""
    assert FDORG_KEY_ENV == "FOOTBALL_DATA_ORG_API_KEY"
    assert API_FOOTBALL_KEY_ENV == "API_FOOTBALL_KEY"


# --- poller -----------------------------------------------------------------


@pytest.fixture(scope="module")
def poll_live():
    """Load the polling script as a module, the way the CLI dispatcher does."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "poll_live.py"
    spec = importlib.util.spec_from_file_location("poll_live_under_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_poll_refuses_when_network_is_disabled(poll_live, monkeypatch) -> None:
    """The daily job must not report success while reading a stale cache."""
    monkeypatch.setenv("PREDICT_FOOTBALL_ALLOW_NETWORK", "0")
    assert poll_live.main(["--league", "ENG_PL"]) == 2


def test_poll_rejects_an_unknown_league(poll_live) -> None:
    """A typo must fail fast, before any network or settings work."""
    assert poll_live.main(["--league", "ENG_PL", "--league", "NOT_A_LEAGUE"]) == 2


def test_poll_current_season_is_the_latest(poll_live) -> None:
    """The poller must top up the newest season the provider offers."""

    class Stub:
        display_name = "Stub"

        def available_seasons(self, league_key):  # type: ignore[no-untyped-def]
            return [
                SeasonRef(league_key, "2324", "2023/24"),
                SeasonRef(league_key, "2425", "2024/25"),
            ]

    assert poll_live._current_season_code(Stub(), "ENG_PL") == "2425"


def test_poll_current_season_requires_coverage(poll_live) -> None:
    """No seasons must raise rather than silently pick none."""

    class Empty:
        display_name = "Empty"

        def available_seasons(self, league_key):  # type: ignore[no-untyped-def]
            return []

    with pytest.raises(ProviderError, match="offers no seasons"):
        poll_live._current_season_code(Empty(), "ENG_PL")

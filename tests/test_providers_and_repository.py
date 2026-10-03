"""Provider adapter and repository tests. All offline."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from tests.conftest import FDCO_CSV, FakeSession, fdco_bytes

from predict_football.config.settings import Settings
from predict_football.data.providers.base import (
    DataProvider,
    ProviderError,
    ProviderNotAvailable,
    RemoteDataDisabled,
    SeasonRef,
)
from predict_football.data.providers.football_data_co import FootballDataCoProvider
from predict_football.data.providers.registry import get_provider, list_providers
from predict_football.data.providers.statsbomb import StatsBombOpenProvider
from predict_football.data.repository import Database

# --- registry ---------------------------------------------------------------


def test_registry_lists_both_providers() -> None:
    """Both adapters must be discoverable by name."""
    names = list_providers()
    assert "football_data_co" in names
    assert "statsbomb_open" in names


def test_registry_rejects_unknown_provider() -> None:
    """An unknown provider name must list the valid options."""
    with pytest.raises(KeyError, match="Registered providers"):
        get_provider("not_a_provider")


def test_get_provider_returns_the_right_type() -> None:
    """Name-based construction must resolve to the right class."""
    assert isinstance(get_provider("football_data_co"), FootballDataCoProvider)
    assert isinstance(get_provider("statsbomb_open"), StatsBombOpenProvider)


# --- DataProvider contract --------------------------------------------------


def test_provider_contract_is_enforced() -> None:
    """A partial implementation must not be instantiable."""

    class Incomplete(DataProvider):
        name = "incomplete"

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


def test_optional_methods_raise_a_helpful_default() -> None:
    """Optional capabilities must explain themselves when unsupported."""

    class Minimal(DataProvider):
        name = "minimal"
        display_name = "Minimal"

        def available_seasons(self, league_key):  # type: ignore[no-untyped-def]
            return []

        def fetch_matches(self, league_key, season_code):  # type: ignore[no-untyped-def]
            return pd.DataFrame()

    provider = Minimal()
    with pytest.raises(ProviderNotAvailable, match="does not provide bookmaker odds"):
        provider.fetch_odds("X", "1")
    with pytest.raises(ProviderNotAvailable, match="does not provide event-level data"):
        provider.fetch_events("X")
    with pytest.raises(ProviderNotAvailable, match="does not provide lineup data"):
        provider.fetch_lineups("X")
    assert provider.covers("X") is False


# --- football-data.co.uk ----------------------------------------------------


def test_fdco_maps_columns_to_canonical_schema(fdco_provider) -> None:
    """Provider fields must land on canonical column names."""
    frame = fdco_provider.fetch_matches("ENG_PL", "2324")
    assert len(frame) == 5
    assert frame["home_team"].iloc[0] == "Liverpool"
    assert frame["away_team"].iloc[0] == "Bournemouth"
    assert frame["league_key"].iloc[0] == "ENG_PL"
    assert frame["competition_type"].iloc[0] == "league"
    assert frame["season"].iloc[0] == "2023/24"
    assert frame["season_code"].iloc[0] == "2324"


def test_fdco_resolves_team_aliases(fdco_provider) -> None:
    """Source spellings must resolve to canonical names."""
    frame = fdco_provider.fetch_matches("ENG_PL", "2324")
    teams = set(frame["home_team"]) | set(frame["away_team"])
    assert "Nottingham Forest" in teams
    assert "Wolverhampton Wanderers" in teams
    assert "Manchester United" in teams
    assert "Man City" not in teams


def test_fdco_parses_day_first_dates(fdco_provider) -> None:
    """DD/MM must not be read as MM/DD, or every autumn date shifts."""
    frame = fdco_provider.fetch_matches("ENG_PL", "2324")
    first = frame["match_date"].iloc[0]
    assert (first.month, first.day) == (8, 11)


def test_fdco_computes_match_ids_that_are_unique_and_stable(fdco_provider) -> None:
    """Match identity must be unique per row and reproducible across calls."""
    frame = fdco_provider.fetch_matches("ENG_PL", "2324")
    assert frame["match_id"].is_unique
    again = fdco_provider.fetch_matches("ENG_PL", "2324")
    assert list(frame["match_id"]) == list(again["match_id"])


def test_fdco_derives_result_and_keeps_statistics(fdco_provider) -> None:
    """Outcome labels and match statistics must survive the mapping."""
    frame = fdco_provider.fetch_matches("ENG_PL", "2324")
    assert list(frame["result"]) == ["D", "H", "D", "H", "H"]
    assert frame["home_shots"].iloc[0] == 15
    assert frame["away_shots"].iloc[0] == 7
    assert frame["home_shots_on_target"].iloc[0] == 5


def test_fdco_tolerates_missing_optional_columns(settings, resolver) -> None:
    """Older seasons lack several columns; absent fields must become null."""
    minimal = (
        "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,AvgH,AvgD,AvgA\n"
        "E0,11/08/2023,Liverpool,Arsenal,2,1,H,1.80,3.60,4.50\n"
    )
    session = FakeSession({"E0.csv": fdco_bytes(minimal)})
    provider = FootballDataCoProvider(settings=settings, teams=resolver, session=session)
    frame = provider.fetch_matches("ENG_PL", "2324")
    assert len(frame) == 1
    assert frame["home_shots"].isna().all()
    assert frame["home_xg"].isna().all()
    assert frame["result"].iloc[0] == "H"


def test_fdco_keeps_unplayed_fixtures_with_null_result(settings, resolver) -> None:
    """The live season file contains future fixtures with blank scorelines."""
    body = (
        "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
        "E0,11/08/2023,Liverpool,Arsenal,2,1,H\n"
        "E0,01/05/2026,Arsenal,Liverpool,,,\n"
    )
    session = FakeSession({"E0.csv": fdco_bytes(body)})
    provider = FootballDataCoProvider(settings=settings, teams=resolver, session=session)
    frame = provider.fetch_matches("ENG_PL", "2526")
    assert len(frame) == 2
    assert frame["result"].isna().sum() == 1
    assert frame["home_goals"].isna().sum() == 1


def test_fdco_rejects_a_payload_that_is_not_csv(settings, resolver) -> None:
    """An HTML error page must fail with a useful message, not a parser crash."""
    session = FakeSession({"E0.csv": b"<html><body>404 not found</body></html>"})
    provider = FootballDataCoProvider(settings=settings, teams=resolver, session=session)
    with pytest.raises(ProviderError, match=r"expected columns|EmptyDataError|CSV parse failed"):
        provider.fetch_matches("ENG_PL", "2324")


def test_fdco_reports_http_errors_clearly(settings, resolver) -> None:
    """A 404 must be explained as an unavailable season."""
    session = FakeSession({"E0.csv": 404})
    provider = FootballDataCoProvider(settings=settings, teams=resolver, session=session)
    with pytest.raises(ProviderError, match="HTTP 404"):
        provider.fetch_matches("ENG_PL", "2324")


def test_fdco_available_seasons_are_ordered_and_typed(fdco_provider) -> None:
    """Season listing must be usable without hitting the network."""
    seasons = fdco_provider.available_seasons("ENG_PL")
    assert len(seasons) > 25
    assert all(isinstance(s, SeasonRef) for s in seasons)
    assert seasons[0].season_code == "9394"
    assert seasons[-1].season_code >= f"{pd.Timestamp.now().year % 100:02d}"


def test_fdco_covers_only_configured_leagues(fdco_provider) -> None:
    """Coverage must reflect configuration, not optimism."""
    assert fdco_provider.covers("ENG_PL") is True
    assert fdco_provider.covers("ESP_LA_LIGA") is True
    assert fdco_provider.covers("INT_WORLD_CUP") is False
    assert fdco_provider.covers("NOT_A_LEAGUE") is False


def test_fdco_raises_for_uncovered_league(fdco_provider) -> None:
    """An uncovered competition must raise a specific error."""
    with pytest.raises(ProviderNotAvailable, match="does not cover"):
        fdco_provider.fetch_matches("INT_WORLD_CUP", "2324")


def test_fdco_extracts_odds_and_computes_overround(fdco_provider) -> None:
    """Odds must be extracted and the bookmaker margin computed."""
    odds = fdco_provider.fetch_odds("ENG_PL", "2324")
    assert not odds.empty
    assert set(odds["market"]) == {"1x2"}
    assert set(odds["bookmaker"]) == {"avg", "b365", "max", "ps"}

    avg_home = odds[(odds["bookmaker"] == "avg") & (odds["odds_home_close"] == 1.95)]
    assert len(avg_home) == 1
    # 1/1.95 + 1/3.65 + 1/3.55 = 1.0685, i.e. a 6.85% bookmaker margin.
    expected = 1 / 1.95 + 1 / 3.65 + 1 / 3.55
    assert avg_home.iloc[0]["overround"] == pytest.approx(expected, abs=1e-9)
    assert avg_home.iloc[0]["price_basis"] == "close"


def test_fdco_falls_back_to_opening_prices_and_records_the_basis(settings, resolver) -> None:
    """When closing prices are absent, opening prices must be used and labelled."""
    body = (
        "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,AvgH,AvgD,AvgA\n"
        "E0,11/08/2023,Liverpool,Arsenal,2,1,H,1.80,3.60,4.50\n"
    )
    session = FakeSession({"E0.csv": fdco_bytes(body)})
    provider = FootballDataCoProvider(settings=settings, teams=resolver, session=session)
    odds = provider.fetch_odds("ENG_PL", "2324")
    avg = odds[odds["bookmaker"] == "avg"].iloc[0]
    assert avg["price_basis"] == "open"
    assert avg["odds_home_close"] is None or pd.isna(avg["odds_home_close"])
    assert avg["overround"] == pytest.approx(1 / 1.80 + 1 / 3.60 + 1 / 4.50, abs=0.001)


def test_fdco_caches_downloads(settings, resolver) -> None:
    """A second fetch must be served from cache without a second request."""
    session = FakeSession({"E0.csv": fdco_bytes()})
    provider = FootballDataCoProvider(settings=settings, teams=resolver, session=session)
    provider.fetch_matches("ENG_PL", "2324")
    assert len(session.requested) == 1
    provider.fetch_matches("ENG_PL", "2324")
    assert len(session.requested) == 1, "the cached copy should have been reused"


def test_fdco_writes_an_archival_copy(settings, resolver) -> None:
    """Raw downloads must be archived for inspection."""
    session = FakeSession({"E0.csv": fdco_bytes()})
    provider = FootballDataCoProvider(settings=settings, teams=resolver, session=session)
    provider.fetch_matches("ENG_PL", "2324")
    archived = settings.raw_dir / "football_data_co" / "E0" / "E0_2324.csv"
    assert archived.is_file()
    assert "Liverpool" in archived.read_text(encoding="utf-8")


def test_fdco_downloads_the_season_once_for_results_and_odds(settings, resolver) -> None:
    """Results and odds come from one CSV, so one download must serve both.

    Separate cache keys for the same document would download and store the file
    twice, doubling network and disk use on every ingest.
    """
    session = FakeSession({"E0.csv": fdco_bytes()})
    provider = FootballDataCoProvider(settings=settings, teams=resolver, session=session)
    provider.fetch_matches("ENG_PL", "2324")
    provider.fetch_odds("ENG_PL", "2324")
    assert len(session.requested) == 1, "the season CSV should be fetched exactly once"


def test_cache_miss_offline_raises_rather_than_hanging(offline_settings, resolver) -> None:
    """With no cache and no network, fail loudly instead of blocking."""
    session = FakeSession({"E0.csv": fdco_bytes()})
    provider = FootballDataCoProvider(settings=offline_settings, teams=resolver, session=session)
    with pytest.raises(RemoteDataDisabled, match="network access is disabled"):
        provider.fetch_matches("ENG_PL", "2324")


def test_cached_data_is_readable_when_offline(settings, offline_settings, resolver) -> None:
    """Priming the cache online then going offline must still work.

    Both providers must share a cache directory, which is why the online phase
    uses ``settings`` and the offline phase ``offline_settings``: the two
    fixtures differ only in ``allow_network``, so the cache they point at is
    the same one.
    """
    online = FootballDataCoProvider(
        settings=settings, teams=resolver, session=FakeSession({"E0.csv": fdco_bytes()})
    )
    online.fetch_matches("ENG_PL", "2324")

    offline = FootballDataCoProvider(settings=offline_settings, teams=resolver, session=None)
    frame = offline.fetch_matches("ENG_PL", "2324")
    assert len(frame) == 5


def test_fixture_csv_is_labelled_synthetic() -> None:
    """Guard the honesty rule: the test fixture must not look like real data.

    Real football-data.co.uk CSVs are gitignored, so this fixture is synthetic.
    If someone swaps in real rows, this test is the tripwire.
    """
    assert "Div,Date" in FDCO_CSV
    # The fixture uses deliberately round, non-real scorelines and venues.
    assert "Old Trafford" not in FDCO_CSV


# --- StatsBomb --------------------------------------------------------------

SB_COMPETITIONS = json.dumps(
    [
        {
            "competition_id": 999,
            "season_id": 2023,
            "country_name": "International",
            "competition_name": "Africa Cup of Nations",
            "season_name": "2023",
            "match_available": 52,
        },
        {
            "competition_id": 998,
            "season_id": 2018,
            "country_name": "International",
            "competition_name": "FIFA World Cup",
            "season_name": "2018",
            "match_available": 64,
        },
    ]
).encode("utf-8")

SB_MATCHES = json.dumps(
    [
        {
            "match_id": 11111,
            "match_date": "2023-01-10",
            "kick_off": "20:00",
            "competition": {"competition_id": 999, "season_id": 2023},
            "home_team": {"home_team_name": "Nigeria"},
            "away_team": {"away_team_name": "Egypt"},
            "home_score": 1,
            "away_score": 0,
            "referee_name": "A. Referee",
            "stadium": {"name": "Test Stadium"},
        }
    ]
).encode("utf-8")

SB_EVENTS = json.dumps(
    [
        {
            "id": "e1",
            "index": 1,
            "period": 1,
            "timestamp": "00:00:00.000",
            "minute": 0,
            "second": 0,
            "type": {"id": 35, "name": "Starting XI"},
            "team": {"id": 1, "name": "Nigeria"},
            "tactics": {"formation": 433},
        },
        {
            "id": "e2",
            "period": 1,
            "timestamp": "00:12:30.000",
            "minute": 12,
            "second": 30,
            "type": {"id": 16, "name": "Shot"},
            "team": {"id": 1, "name": "Nigeria"},
            "player": {"id": 7, "name": "A. Player"},
            "shot": {
                "statsbomb_xg": 0.312,
                "location": [88.0, 40.5, 0.0],
                "end_location": [100.2, 6.1, 0.9],
                "outcome": {"id": 97, "name": "Goal"},
                "type": {"id": 87, "name": "Open Play"},
                "body_part": {"id": 40, "name": "Right Foot"},
                "assist": {"id": 9, "name": "B. Assister"},
            },
        },
        {
            "id": "e3",
            "period": 1,
            "timestamp": "00:40:05.000",
            "minute": 40,
            "second": 5,
            "type": {"id": 43, "name": "Card"},
            "team": {"id": 2, "name": "Egypt"},
            "player": {"id": 11, "name": "C. Foul"},
        },
        {
            "id": "e4",
            "period": 1,
            "timestamp": "00:50:00.000",
            "minute": 50,
            "second": 0,
            "type": {"id": 19, "name": "Substitution"},
            "team": {"id": 1, "name": "Nigeria"},
            "player": {"id": 12, "name": "D. Sub"},
            "substitution": {"outcome": True, "replacement": {"id": 13, "name": "E. Replacement"}},
        },
    ]
).encode("utf-8")

SB_LINEUPS = json.dumps(
    [
        {
            "team": {"id": 1, "name": "Nigeria"},
            "lineup": [
                {
                    "player": {"id": 7, "name": "A. Player"},
                    "position": {"id": 1, "name": "Goalkeeper"},
                    "jersey_number": 1,
                },
                {
                    "player": {"id": 12, "name": "D. Sub"},
                    "position": {"id": 16, "name": "Substitute"},
                    "jersey_number": 19,
                },
            ],
        }
    ]
).encode("utf-8")


def _sb_provider(settings: Settings) -> StatsBombOpenProvider:
    """Build a StatsBomb provider backed by canned JSON.

    Args:
        settings: Isolated settings.

    Returns:
        A configured provider.
    """
    session = FakeSession(
        {
            "competitions.json": SB_COMPETITIONS,
            "matches/999/2023.json": SB_MATCHES,
            "events/11111.json": SB_EVENTS,
            "lineups/11111.json": SB_LINEUPS,
        }
    )
    return StatsBombOpenProvider(settings=settings, session=session)


def test_statsbomb_discovers_competitions_by_name_not_hardcoded_id(settings) -> None:
    """Numeric IDs must come from the published manifest, not from our memory."""
    provider = _sb_provider(settings)
    rows = provider.resolve_competition("INT_AFCON")
    assert len(rows) == 1
    assert rows.iloc[0]["competition_id"] == 999
    assert rows.iloc[0]["season_id"] == 2023


def test_statsbomb_reports_uncovered_competition_clearly(settings) -> None:
    """A competition StatsBomb lacks must name what is actually available."""
    provider = _sb_provider(settings)
    with pytest.raises(ProviderNotAvailable, match="Available competitions"):
        provider.resolve_competition("ENG_PL")


def test_statsbomb_available_seasons_uses_manifest(settings) -> None:
    """Season ids must come from the manifest."""
    provider = _sb_provider(settings)
    seasons = provider.available_seasons("INT_AFCON")
    assert [s.season_code for s in seasons] == ["2023"]
    assert seasons[0].match_count == 52


def test_statsbomb_maps_matches_to_canonical_schema(settings) -> None:
    """Published match metadata must land on canonical columns."""
    provider = _sb_provider(settings)
    frame = provider.fetch_matches("INT_AFCON", "2023")
    assert len(frame) == 1
    assert frame["home_team"].iloc[0] == "Nigeria"
    assert frame["away_team"].iloc[0] == "Egypt"
    assert frame["result"].iloc[0] == "H"
    assert frame["competition_type"].iloc[0] == "cup"
    assert frame["statsbomb_match_id"].iloc[0] == 11111


def test_statsbomb_maps_events_with_xg_and_timestamps(settings) -> None:
    """Events must carry the fields the replay engine depends on."""
    provider = _sb_provider(settings)
    matches = provider.fetch_matches("INT_AFCON", "2023")
    provider.register_match_ids(matches)
    events = provider.fetch_events(str(matches.iloc[0]["match_id"]))

    assert len(events) == 4
    assert list(events["minute"]) == [0, 12, 40, 50]
    shot = events[events["event_type"] == "shot_on_target"].iloc[0]
    assert shot["xg"] == pytest.approx(0.312)
    assert shot["player"] == "A. Player"
    assert shot["location_x"] == pytest.approx(88.0)
    assert shot["shot_assist"] == "B. Assister"
    assert shot["shot_body_part"] == "Right Foot"


def test_statsbomb_events_are_ordered_by_time(settings) -> None:
    """Replay requires strict chronological order."""
    provider = _sb_provider(settings)
    matches = provider.fetch_matches("INT_AFCON", "2023")
    provider.register_match_ids(matches)
    events = provider.fetch_events(str(matches.iloc[0]["match_id"]))
    assert list(events["event_id"]) == list(events["event_id"]), "event ids must be stable"
    keys = list(zip(events["period"], events["minute"], events["second"], strict=True))
    assert keys == sorted(keys)


def test_statsbomb_events_require_match_registration(settings) -> None:
    """Fetching events for an unknown match must fail with guidance."""
    provider = _sb_provider(settings)
    with pytest.raises(ProviderNotAvailable, match="no provider match id known"):
        provider.fetch_events("unknown_match_id")


def test_statsbomb_lineups_mark_substitutes_and_leave_captain_unknown(settings) -> None:
    """Substitutes must be identified; captain must stay null, not False.

    StatsBomb open data publishes no captain flag. Claiming False would be
    inventing a fact.
    """
    provider = _sb_provider(settings)
    matches = provider.fetch_matches("INT_AFCON", "2023")
    provider.register_match_ids(matches)
    lineups = provider.fetch_lineups(str(matches.iloc[0]["match_id"]))

    keeper = lineups[lineups["player"] == "A. Player"].iloc[0]
    sub = lineups[lineups["player"] == "D. Sub"].iloc[0]
    assert keeper["is_starter"] is False or keeper["is_starter"] == 0 or keeper["is_starter"]
    assert sub["position"] == "Substitute"
    assert lineups["is_captain"].isna().all(), "captain must be null because the source does not publish it"


def test_statsbomb_event_mapping_covers_shot_outcomes(settings) -> None:
    """Shot outcomes must refine the generic event type, not overwrite blindly."""
    provider = _sb_provider(settings)
    matches = provider.fetch_matches("INT_AFCON", "2023")
    provider.register_match_ids(matches)
    events = provider.fetch_events(str(matches.iloc[0]["match_id"]))
    assert "shot_on_target" in set(events["event_type"])
    assert "yellow_card" in set(events["event_type"])
    assert "substitution_on" in set(events["event_type"])
    assert "raw_event_type" in events.columns, "the source label must be kept for debugging"


# --- repository -------------------------------------------------------------


def test_database_initialises_schema(tmp_path) -> None:
    """Schema creation must be idempotent and record its version."""
    db = Database(tmp_path / "test.sqlite")
    db.initialise()
    db.initialise()
    assert db.schema_version() == "1.0.0"
    counts = db.row_counts()
    for table in ("matches", "match_targets", "match_stats", "match_odds", "events", "lineups"):
        assert table in counts


def test_repository_roundtrips_matches(repository, sample_matches) -> None:
    """Written matches must come back with the same values."""
    written = repository.upsert_matches(sample_matches)
    assert written == 4

    loaded = repository.load_matches(league_key="ENG_PL")
    assert len(loaded) == 4
    assert set(loaded["match_id"]) == set(sample_matches["match_id"])
    assert loaded[loaded["match_id"] == "m1"].iloc[0]["result"] == "D"


def test_repository_upsert_is_idempotent(repository, sample_matches) -> None:
    """Re-ingesting a season must not duplicate rows."""
    repository.upsert_matches(sample_matches)
    repository.upsert_matches(sample_matches)
    assert repository.match_count(league_key="ENG_PL") == 4
    assert len(repository.load_matches(league_key="ENG_PL")) == 4


def test_repository_joins_statistics_back(repository, sample_matches) -> None:
    """Post-match stats must be reachable as home_/away_ columns."""
    repository.upsert_matches(sample_matches)
    loaded = repository.load_matches(league_key="ENG_PL")
    m1 = loaded[loaded["match_id"] == "m1"].iloc[0]
    assert m1["home_shots"] == 15
    assert m1["away_shots"] == 7
    assert m1["home_xg"] == pytest.approx(1.4)


def test_repository_filters_by_season_and_completed(repository, sample_matches) -> None:
    """Season and result filters must work together."""
    repository.upsert_matches(sample_matches)
    assert repository.available_seasons("ENG_PL") == ["2023/24"]
    completed = repository.load_matches(league_key="ENG_PL", with_result=True)
    assert len(completed) == 3
    assert completed["result"].notna().all()


def test_repository_stores_odds(repository, sample_matches) -> None:
    """Odds must persist and reload joined to match metadata."""
    repository.upsert_matches(sample_matches)
    odds = pd.DataFrame(
        {
            "match_id": ["m1", "m2"],
            "bookmaker": ["avg"] * 2,
            "market": ["1x2"] * 2,
            "odds_home_close": [1.95, 1.57],
            "odds_draw_close": [3.65, 4.00],
            "odds_away_close": [3.55, 5.50],
            "overround": [1.0448, 1.0489],
            "price_basis": ["close"] * 2,
        }
    )
    assert repository.upsert_odds(odds) == 2

    loaded = repository.load_odds(league_keys=["ENG_PL"], bookmaker="avg")
    assert len(loaded) == 2
    row = loaded[loaded["match_id"] == "m1"].iloc[0]
    assert row["odds_home_close"] == pytest.approx(1.95)
    assert row["home_team"] == "Liverpool"
    assert row["match_date"] is not None


def test_repository_requires_columns(repository) -> None:
    """An incomplete frame must be rejected with a clear message."""
    with pytest.raises(ValueError, match="missing"):
        repository.upsert_matches(pd.DataFrame({"match_id": ["m1"]}))


def test_repository_tracks_unresolved_teams(repository) -> None:
    """Unrecognised team names must be queryable for maintenance."""
    repository.upsert_matches(
        pd.DataFrame(
            {
                "match_id": ["m1"],
                "source": ["test"],
                "league_key": ["ENG_PL"],
                "competition_type": ["league"],
                "season": ["2023/24"],
                "match_date": pd.to_datetime(["2023-08-11"]),
                "home_team": ["Nonexistent FC"],
                "away_team": ["Liverpool"],
            }
        )
    )
    unresolved = repository.teams(unresolved_only=True)
    assert "Nonexistent FC" in set(unresolved["canonical_name"])

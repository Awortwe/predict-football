"""Loader orchestration and cache tests."""

from __future__ import annotations

import pandas as pd
import pytest
from tests.conftest import FakeSession, fdco_bytes

from predict_football.config.settings import Settings
from predict_football.data.cache import CacheEntry, RawCache
from predict_football.data.loaders import ingest_season
from predict_football.data.providers.base import DataProvider, ProviderNotAvailable
from predict_football.data.providers.registry import get_provider
from predict_football.data.repository import Database, MatchRepository


def test_cache_stores_and_reloads_payloads(settings: Settings) -> None:
    """A stored payload must be byte-identical on the way back out."""
    cache = RawCache(settings)
    payload = b"Div,Date\nE0,11/08/2023\n"
    entry = cache.store("test_provider", "matches:ENG_PL:2324", payload, url="https://example.invalid/x.csv")

    assert entry.size_bytes == len(payload)
    assert entry.source_kind == "http"
    assert cache.read("test_provider", "matches:ENG_PL:2324") == payload
    assert cache.read_meta("test_provider", "matches:ENG_PL:2324") is not None


def test_cache_detects_corrupted_payloads(settings: Settings) -> None:
    """A digest mismatch must be treated as absent, not returned as truth."""
    cache = RawCache(settings)
    cache.store("test_provider", "resource", b"original")
    (cache.entry_dir("test_provider", "resource") / "payload.bin").write_bytes(b"tampered")
    assert cache.read("test_provider", "resource") is None


def test_cache_records_local_provenance(settings: Settings) -> None:
    """Generated payloads must be marked local, not as downloads."""
    cache = RawCache(settings)
    entry = cache.put_local("test_provider", "generated", b"data")
    assert entry.url is None
    assert entry.source_kind == "local"


def test_cache_entry_age_uses_utc() -> None:
    """Freshness must be computed against UTC, not naive local time."""
    from datetime import datetime, timedelta, timezone

    fresh = CacheEntry("p", "r", None, "sha", 1, datetime.now(timezone.utc).isoformat(), "http")
    stale = CacheEntry(
        "p", "r", None, "sha", 1, (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(), "http"
    )
    assert fresh.is_stale(ttl_days=7) is False
    assert stale.is_stale(ttl_days=7) is True
    assert stale.age_days() > 29


def test_cache_sanitises_resource_keys(settings: Settings) -> None:
    """Path separators in a resource key must not escape the cache root."""
    cache = RawCache(settings)
    entry_dir = cache.entry_dir("provider", "matches:../escape/../../etc")
    assert cache.root.resolve() in entry_dir.resolve().parents
    assert ".." not in entry_dir.name


def test_cache_stats_and_prune(settings: Settings) -> None:
    """Cache bookkeeping must report and clean entries."""
    cache = RawCache(settings)
    cache.store("p1", "a", b"x")
    cache.store("p2", "b", b"yy")
    stats = cache.stats()
    assert stats["p1"]["entries"] == 1
    assert stats["p1"]["bytes"] == 1
    assert stats["p2"]["bytes"] == 2

    # Nothing is stale yet.
    assert cache.prune(max_age_days=365) == 0
    assert cache.prune(max_age_days=0) == 2


def test_ingest_season_writes_matches_and_odds(repository, sample_matches) -> None:
    """End-to-end ingest must populate the database."""

    class StubProvider(DataProvider):
        name = "football_data_co"
        display_name = "stub"

        def __init__(self, matches: pd.DataFrame, odds: pd.DataFrame | None) -> None:
            self._matches = matches
            self._odds = odds

        def available_seasons(self, league_key: str):
            return []

        def fetch_matches(self, league_key: str, season_code: str) -> pd.DataFrame:
            return self._matches

        def fetch_odds(self, league_key: str, season_code: str) -> pd.DataFrame:
            if self._odds is None:
                raise ProviderNotAvailable("stub provider has no odds")
            return self._odds

    odds = pd.DataFrame(
        {
            "match_id": ["m1"],
            "bookmaker": ["avg"],
            "market": ["1x2"],
            "odds_home_close": [1.95],
            "odds_draw_close": [3.65],
            "odds_away_close": [3.55],
            "overround": [1.0448],
            "price_basis": ["close"],
        }
    )
    provider = StubProvider(sample_matches, odds)
    result = ingest_season(provider, repository, league_key="ENG_PL", season_code="2324")

    assert result.matches_written == 4
    assert result.odds_written == 1
    assert result.coverage["matches_with_result"] == 3
    assert "ENG_PL 2324" in result.summary()
    assert repository.match_count(league_key="ENG_PL") == 4


def test_ingest_season_records_missing_odds_as_a_warning_not_a_failure(repository, sample_matches) -> None:
    """A provider without odds must not abort the ingest."""

    class NoOddsProvider(DataProvider):
        name = "football_data_co"
        display_name = "no-odds stub"

        def available_seasons(self, league_key: str):
            return []

        def fetch_matches(self, league_key: str, season_code: str) -> pd.DataFrame:
            return sample_matches

    result = ingest_season(NoOddsProvider(), repository, league_key="ENG_PL", season_code="2324")
    assert result.matches_written == 4
    assert result.odds_written == 0
    assert any("no odds available" in w for w in result.warnings)


def test_ingest_season_warns_about_unresolved_teams(repository) -> None:
    """Unrecognised team names must surface as a warning on the result."""

    class OddTeamsProvider(DataProvider):
        name = "football_data_co"
        display_name = "odd teams stub"

        def available_seasons(self, league_key: str):
            return []

        def fetch_matches(self, league_key: str, season_code: str) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "match_id": ["m1"],
                    "source": ["football_data_co"],
                    "league_key": ["ENG_PL"],
                    "competition_type": ["league"],
                    "season": ["2023/24"],
                    "match_date": pd.to_datetime(["2023-08-11"]),
                    "home_team": ["Nonexistent FC"],
                    "away_team": ["Liverpool"],
                    "result": ["H"],
                    "home_goals": [1],
                    "away_goals": [0],
                }
            )

    result = ingest_season(OddTeamsProvider(), repository, league_key="ENG_PL", season_code="2324")
    assert any("outside the registry" in w for w in result.warnings)


def test_ingest_season_raises_on_empty_response(repository) -> None:
    """An empty season is a provider problem and must surface."""

    class EmptyProvider(DataProvider):
        name = "football_data_co"
        display_name = "empty stub"

        def available_seasons(self, league_key: str):
            return []

        def fetch_matches(self, league_key: str, season_code: str) -> pd.DataFrame:
            return pd.DataFrame()

    with pytest.raises(ProviderNotAvailable, match="returned no matches"):
        ingest_season(EmptyProvider(), repository, league_key="ENG_PL", season_code="2324")


def test_real_provider_writes_through_the_loader(tmp_path, repository) -> None:
    """The real adapter and the loader must compose without a hand-off gap."""
    settings = Settings.from_env(project_root=tmp_path)
    object.__setattr__(settings, "allow_network", True)
    settings.ensure_directories()
    session = FakeSession({"E0.csv": fdco_bytes()})
    provider = get_provider("football_data_co", settings=settings, session=session)

    result = ingest_season(provider, repository, league_key="ENG_PL", season_code="2324", include_odds=True)
    assert result.matches_written == 5
    assert result.odds_written == 4 * 5  # four bookmakers x five matches

    loaded = repository.load_matches(league_key="ENG_PL", with_result=True)
    assert len(loaded) == 5
    assert set(loaded["result"]) == {"D", "H"}


def test_ingest_is_reproducible_across_runs(tmp_path) -> None:
    """Two ingests into separate databases must produce identical match ids.

    Reproducibility is a stated requirement, so it needs a test rather than a
    promise in a README.
    """
    settings = Settings.from_env(project_root=tmp_path)
    object.__setattr__(settings, "allow_network", True)
    settings.ensure_directories()

    frames = []
    for _run in range(2):
        db = MatchRepository(Database(":memory:"))
        provider = get_provider(
            "football_data_co", settings=settings, session=FakeSession({"E0.csv": fdco_bytes()})
        )
        ingest_season(provider, db, league_key="ENG_PL", season_code="2324", include_odds=False)
        frames.append(list(db.load_matches(league_key="ENG_PL")["match_id"]))

    assert frames[0] == frames[1]
    assert len(set(frames[0])) == len(frames[0]), "match ids must be unique"

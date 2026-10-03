"""Tests for the public-deployment bootstrap.

The bootstrap is the one place that decides what data a deployed app may hold,
so the tests concentrate on its two guarantees: it never fetches a source whose
licence forbids public serving, and it never returns a ready status without
matches. Everything runs offline through an injected provider.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from predict_football.config.licences import LicenceViolation
from predict_football.config.settings import Settings
from predict_football.data.bootstrap import (
    PUBLIC_LEAGUE,
    BootstrapError,
    default_seasons,
    ensure_public_data,
)
from predict_football.data.loaders import build_repository
from predict_football.data.providers.base import DataProvider, ProviderNotAvailable


class _StubProvider(DataProvider):
    """A provider that returns a fixed frame, or refuses every season."""

    name = "football_data_org"
    display_name = "stub"

    def __init__(self, matches: pd.DataFrame, *, fail: bool = False) -> None:
        self._matches = matches
        self._fail = fail

    def available_seasons(self, league_key: str) -> list[object]:
        return []

    def fetch_matches(self, league_key: str, season_code: str) -> pd.DataFrame:
        if self._fail:
            raise ProviderNotAvailable(f"no data for {season_code}")
        return self._matches


def test_bootstrap_ingests_a_compliant_source(settings: Settings, sample_matches: pd.DataFrame) -> None:
    """An empty store is filled from a source that may be served publicly."""
    status = ensure_public_data(settings, provider=_StubProvider(sample_matches), seasons=["2324"])

    assert status.ready
    assert status.matches == len(sample_matches)
    assert status.league_key == PUBLIC_LEAGUE
    assert "Football-Data.org" in status.attribution[0]

    store = build_repository(settings)
    assert store.match_count(league_key=PUBLIC_LEAGUE) == len(sample_matches)


def test_bootstrap_reuses_existing_matches_without_fetching(settings: Settings, sample_matches: pd.DataFrame) -> None:
    """A second run must not re-download a store that is already populated."""
    ensure_public_data(settings, provider=_StubProvider(sample_matches), seasons=["2324"])

    status = ensure_public_data(settings, provider=_StubProvider(sample_matches, fail=True), seasons=["2324"])

    assert status.ready
    assert status.matches == len(sample_matches)
    assert status.seasons_ingested == ()


def test_bootstrap_refuses_a_source_that_may_not_be_served_publicly(
    settings: Settings, sample_matches: pd.DataFrame
) -> None:
    """The licence guard must fire before anything is fetched."""
    with pytest.raises(LicenceViolation, match="football_data_co"):
        ensure_public_data(
            settings,
            provider_name="football_data_co",
            provider=_StubProvider(sample_matches),
            seasons=["2324"],
        )


def test_bootstrap_raises_when_no_season_could_be_loaded(
    settings: Settings, sample_matches: pd.DataFrame
) -> None:
    """Reporting every season as unavailable is an honest failure, not an empty app."""
    with pytest.raises(BootstrapError, match="Could not load any"):
        ensure_public_data(settings, provider=_StubProvider(sample_matches, fail=True), seasons=["2324"])


def test_bootstrap_writes_to_the_configured_data_directory(settings: Settings, sample_matches: pd.DataFrame) -> None:
    """The store must land in the data directory the settings name, not a default."""
    ensure_public_data(settings, provider=_StubProvider(sample_matches), seasons=["2324"])

    assert settings.database_path.exists()


def test_default_seasons_covers_three_prior_and_the_current_season() -> None:
    """A run in October 2026 pulls 2023/24 through 2026/27."""
    assert default_seasons(date(2026, 10, 3)) == ["2324", "2425", "2526", "2627"]


def test_default_seasons_uses_the_previous_start_year_before_august() -> None:
    """Before August the season in progress began the previous calendar year."""
    assert default_seasons(date(2026, 3, 1)) == ["2223", "2324", "2425", "2526"]

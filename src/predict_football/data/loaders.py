"""Orchestration: provider -> cache -> clean -> store.

This is the only module that knows the order of operations during ingestion, so
scripts and notebooks call :func:`ingest_season` rather than wiring the stages
up themselves.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from predict_football.config.leagues import get_league
from predict_football.config.settings import Settings, get_settings
from predict_football.data.cleaning import clean_matches, clean_odds, match_coverage
from predict_football.data.providers.base import DataProvider, ProviderNotAvailable
from predict_football.data.providers.registry import get_provider
from predict_football.data.repository import Database, MatchRepository
from predict_football.data.teams import TeamResolver

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """Outcome of ingesting one competition season.

    Attributes:
        league_key: Competition ingested.
        season_code: Season code requested.
        source: Provider key used.
        matches_written: Match rows written to the database.
        odds_written: Odds rows written.
        events_written: Event rows written.
        lineups_written: Lineup rows written.
        coverage: Coverage summary from :func:`match_coverage`.
        warnings: Human-readable notes gathered during ingestion.
    """

    league_key: str
    season_code: str
    source: str
    matches_written: int = 0
    odds_written: int = 0
    events_written: int = 0
    lineups_written: int = 0
    coverage: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Render a one-line human summary.

        Returns:
            A summary string suitable for a log line.
        """
        played = self.coverage.get("matches_with_result", "?")
        return (
            f"{self.league_key} {self.season_code} [{self.source}]: "
            f"{self.matches_written} rows ({played} with a result), "
            f"{self.odds_written} odds rows"
        )


def build_repository(settings: Settings | None = None) -> MatchRepository:
    """Open (or create) the project database.

    Args:
        settings: Resolved settings. Defaults to process settings.

    Returns:
        A ready :class:`MatchRepository`.
    """
    resolved = settings or get_settings()
    resolved.ensure_directories()
    return MatchRepository(Database(resolved.database_path))


def ingest_season(
    provider: DataProvider,
    repository: MatchRepository,
    *,
    league_key: str,
    season_code: str,
    include_odds: bool = True,
    teams: TeamResolver | None = None,
) -> IngestResult:
    """Ingest one competition season into the database.

    Args:
        provider: Source adapter.
        repository: Destination store.
        league_key: Internal competition key, e.g. ``"ENG_PL"``.
        season_code: Provider-specific season code.
        include_odds: Also fetch and store bookmaker odds. Providers without odds
            raise :class:`ProviderNotAvailable`, which is caught and reported as
            a warning rather than failing the whole ingest.
        teams: Team resolver used during cleaning.

    Returns:
        An :class:`IngestResult` describing what was written.

    Raises:
        ProviderNotAvailable: If the provider has no data for the season.
        ValueError: If the returned data fails validation.
    """
    league = get_league(league_key)
    resolver = teams or TeamResolver()
    result = IngestResult(league_key=league_key, season_code=season_code, source=provider.name)

    matches = provider.fetch_matches(league_key, season_code)
    if matches is None or matches.empty:
        raise ProviderNotAvailable(f"{provider.display_name} returned no matches for {league_key} {season_code}")

    # Register the league row so the database is self-describing.
    repository.upsert_league(
        league_key=league.key,
        name=league.name,
        country=league.country,
        competition_type=league.competition_type.value,
        tier=league.tier,
        notes=league.notes,
    )

    cleaned = clean_matches(matches, teams=resolver)
    result.matches_written = repository.upsert_matches(cleaned)
    result.coverage = match_coverage(cleaned, league_key=league_key, season=cleaned["season"].iloc[0])

    unresolved = cleaned[cleaned.get("teams_unresolved", pd.Series(0, index=cleaned.index)) > 0]
    if len(unresolved):
        result.warnings.append(
            f"{len(unresolved)} row(s) reference a team name outside the registry; "
            f"see repository.teams(unresolved_only=True)"
        )

    if include_odds:
        try:
            odds = provider.fetch_odds(league_key, season_code)
            if odds is not None and not odds.empty:
                valid_odds = clean_odds(odds, matches=cleaned)
                result.odds_written = repository.upsert_odds(valid_odds)
        except ProviderNotAvailable as exc:
            result.warnings.append(f"no odds available: {exc}")

    logger.info(result.summary())
    for warning in result.warnings:
        logger.warning("%s %s: %s", league_key, season_code, warning)
    return result


def ingest_league_seasons(
    provider_name: str,
    *,
    league_key: str,
    start_year: int,
    end_year: int,
    repository: MatchRepository | None = None,
    include_odds: bool = True,
    stop_on_error: bool = False,
) -> list[IngestResult]:
    """Ingest a run of seasons for one competition.

    Args:
        provider_name: Provider key, e.g. ``"football_data_co"``.
        league_key: Internal competition key.
        start_year: First season's starting year.
        end_year: Last season's starting year, inclusive.
        repository: Destination store. Defaults to the project database.
        include_odds: Also fetch bookmaker odds.
        stop_on_error: Abort on the first failure. When False, an unavailable
            season is recorded as a warning and ingestion continues, which is
            what we want because historical coverage is uneven.

    Returns:
        One :class:`IngestResult` per season that succeeded.
    """
    from predict_football.config.leagues import season_codes

    store = repository or build_repository()
    provider = get_provider(provider_name)

    results: list[IngestResult] = []
    for code in season_codes(start_year, end_year):
        try:
            results.append(
                ingest_season(
                    provider,
                    store,
                    league_key=league_key,
                    season_code=code,
                    include_odds=include_odds,
                )
            )
        except (ProviderNotAvailable, ValueError) as exc:
            logger.warning("Skipping %s %s: %s", league_key, code, exc)
            if stop_on_error:
                raise
    return results

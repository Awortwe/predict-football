"""Poll live providers for the current season and top up the database.

This is the daily job: it re-fetches the current season from a live source and
upserts it. Because :func:`ingest_season` writes through the canonical
``match_id``, running it repeatedly is idempotent -- an unchanged match is
rewritten in place, and a newly played fixture gains its result without creating
a duplicate row.

The poller is network-gated on purpose. A scheduled job that silently reads a
stale cache and reports success is worse than one that fails loudly, so unless
``PREDICT_FOOTBALL_ALLOW_NETWORK`` is enabled the command refuses to run and
exits 2. The cache TTL is forced to at most ``--max-age-days`` so a daily poll
actually re-downloads instead of replaying the seven-day research cache.

Usage:
    python -m scripts.poll_live --league ENG_PL
    python -m scripts.poll_live --provider api_football --league ESP_LA_LIGA --max-age-days 0
    python -m scripts.poll_live --league ENG_PL --season 2425 --include-odds
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

# Allow running the script directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from predict_football.config.leagues import get_league
from predict_football.config.licences import licence_for
from predict_football.config.settings import Settings
from predict_football.data.loaders import IngestResult, ingest_season
from predict_football.data.providers.base import DataProvider, ProviderError, RemoteDataDisabled
from predict_football.data.providers.registry import get_provider, list_providers

logger = logging.getLogger("predict_football.poll_live")


def _build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--league", "-l", action="append", dest="leagues", help="League key; repeat for several")
    parser.add_argument("--season", help="Season code, e.g. 2425. Defaults to each league's current season")
    parser.add_argument("--provider", default="football_data_org", choices=list_providers())
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=0,
        help="Refresh cached downloads older than this. 0 (default) always re-fetches",
    )
    parser.add_argument(
        "--include-odds", action="store_true", help="Also fetch odds, where the provider publishes them"
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one polling pass.

    Args:
        argv: Argument vector. Defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` when every requested league refreshed, ``1`` when at least one
        failed, ``2`` when the invocation is invalid or network access is off.
    """
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    leagues = args.leagues or ["ENG_PL"]
    for league_key in leagues:
        try:
            get_league(league_key)
        except KeyError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    settings = Settings.from_env()
    if not settings.allow_network:
        print(
            "error: network access is disabled; the poller has nothing to do. "
            "Set PREDICT_FOOTBALL_ALLOW_NETWORK=1 (see .env.example).",
            file=sys.stderr,
        )
        return 2

    # Force the research cache stale enough that a poll actually refreshes.
    max_age = max(0, args.max_age_days)
    fresh_settings = dataclasses.replace(settings, cache_ttl_days=max_age)
    fresh_settings.ensure_directories()

    from predict_football.data.loaders import build_repository

    repository = build_repository(fresh_settings)
    provider = get_provider(args.provider, settings=fresh_settings)

    _print_licence(args.provider)
    print(f"Polling {provider.display_name} for: {', '.join(leagues)} (cache TTL {max_age} day(s))")

    failures = 0
    for league_key in leagues:
        try:
            season_code = args.season or _current_season_code(provider, league_key)
            result = ingest_season(
                provider,
                repository,
                league_key=league_key,
                season_code=season_code,
                include_odds=args.include_odds,
            )
            _print_result(result)
        except RemoteDataDisabled as exc:
            print(f"  {league_key}: network disabled - {exc}", file=sys.stderr)
            return 2
        except (ProviderError, ValueError) as exc:
            failures += 1
            print(f"  {league_key}: FAILED - {exc}", file=sys.stderr)

    unresolved = repository.teams(unresolved_only=True)
    if len(unresolved):
        print(f"\n{len(unresolved)} team name(s) still outside the registry; review before they accumulate.")

    return 1 if failures else 0


def _current_season_code(provider: DataProvider, league_key: str) -> str:
    """Pick the provider's latest available season for a competition.

    Args:
        provider: Constructed provider.
        league_key: Internal competition key.

    Returns:
        The newest season code the provider offers.

    Raises:
        ProviderError: If the provider offers no seasons for the competition.
    """
    seasons = provider.available_seasons(league_key)
    if not seasons:
        raise ProviderError(f"{provider.display_name} offers no seasons for {league_key}")
    return seasons[-1].season_code


def _print_result(result: IngestResult) -> None:
    """Print one ingest outcome and its warnings.

    Args:
        result: Outcome to report.
    """
    print(f"  {result.summary()}")
    for warning in result.warnings:
        print(f"      ! {warning}")


def _print_licence(provider_name: str) -> None:
    """Print a one-line licence reminder for the chosen provider.

    Args:
        provider_name: Provider key.
    """
    policy = licence_for(provider_name)
    shipping = "OK to ship publicly" if policy.may_serve_from_public_app else "DO NOT ship publicly"
    print(f"Licence: {policy.name} - {shipping}.")


if __name__ == "__main__":
    raise SystemExit(main())

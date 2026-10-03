"""Download and clean historical data into SQLite.

Usage:
    python -m scripts.download_data --league ENG_PL --from 2018 --to 2025
    python -m scripts.download_data --league ENG_PL --provider statsbomb_open --list-seasons
    python -m scripts.download_data --status

Downloads are cached on first run and replayed from disk afterwards, so
re-running is cheap and never hammers a free community source.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running the script directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from predict_football.config.leagues import get_league, list_leagues
from predict_football.data.loaders import build_repository, ingest_season
from predict_football.data.providers.base import ProviderError
from predict_football.data.providers.registry import get_provider, list_providers

logger = logging.getLogger("predict_football.download")


def _build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--league", "-l", help="Internal league key, e.g. ENG_PL")
    parser.add_argument("--from", dest="from_year", type=int, help="First season's starting year")
    parser.add_argument("--to", dest="to_year", type=int, help="Last season's starting year")
    parser.add_argument("--season", help="Ingest a single season code, e.g. 2425")
    parser.add_argument("--provider", default="football_data_co", choices=list_providers())
    parser.add_argument("--no-odds", action="store_true", help="Skip bookmaker odds")
    parser.add_argument("--list-seasons", action="store_true", help="List available seasons and exit")
    parser.add_argument("--list-leagues", action="store_true", help="List known competitions and exit")
    parser.add_argument("--status", action="store_true", help="Show database row counts and exit")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the downloader.

    Args:
        argv: Argument vector. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.list_leagues:
        _print_leagues()
        return 0

    repository = build_repository()

    if args.status:
        _print_status(repository)
        return 0

    if not args.league:
        print("error: --league is required (see --list-leagues)", file=sys.stderr)
        return 2

    provider = get_provider(args.provider)

    if args.list_seasons:
        try:
            for season in provider.available_seasons(args.league):
                extra = f" ({season.match_count} matches)" if season.match_count else ""
                print(f"  {season.season_code}  {season.season_label}{extra}")
        except ProviderError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0

    from predict_football.config.leagues import season_codes

    if args.season:
        codes = [args.season]
    elif args.from_year and args.to_year:
        codes = season_codes(args.from_year, args.to_year)
    else:
        print("error: provide --season, or both --from and --to", file=sys.stderr)
        return 2

    league = get_league(args.league)
    print(f"Ingesting {league.name} ({args.league}) seasons {codes[0]}..{codes[-1]} from {provider.display_name}")
    print(f"  licence: {_licence_note(args.provider)}")

    failures = 0
    for code in codes:
        try:
            result = ingest_season(
                provider,
                repository,
                league_key=args.league,
                season_code=code,
                include_odds=not args.no_odds,
            )
            print(f"  {result.summary()}")
            for warning in result.warnings:
                print(f"      ! {warning}")
        except (ProviderError, ValueError) as exc:
            failures += 1
            print(f"  {code}: FAILED - {exc}", file=sys.stderr)

    print()
    _print_status(repository)
    return 1 if failures == len(codes) else 0


def _licence_note(provider_name: str) -> str:
    """Return a one-line licence reminder for a provider.

    Args:
        provider_name: Provider key.

    Returns:
        Human-readable licence note.
    """
    from predict_football.config.licences import licence_for

    policy = licence_for(provider_name)
    shipping = "OK to ship publicly" if policy.may_serve_from_public_app else "DO NOT ship publicly"
    return f"{policy.name} - {shipping}. {policy.summary}"


def _print_leagues() -> None:
    """Print the configured competition registry."""
    print("Known competitions:")
    for league in list_leagues():
        codes = ", ".join(f"{k}={v}" for k, v in sorted(league.source_codes.items()))
        print(f"  {league.key:18s} {league.competition_type.value:6s} {league.name}")
        if codes:
            print(f"  {'':18s} codes: {codes}")
        if league.notes:
            print(f"  {'':18s} note: {league.notes[:110]}")


def _print_status(repository: object) -> None:
    """Print database row counts and unresolved team names.

    Args:
        repository: A :class:`MatchRepository`.
    """
    from predict_football.data.repository import MatchRepository

    if not isinstance(repository, MatchRepository):
        return
    counts = repository._db.row_counts()
    print("Database status:")
    for table in sorted(counts):
        if counts[table]:
            print(f"  {table:18s} {counts[table]:>8,d} rows")
    unresolved = repository.teams(unresolved_only=True)
    if len(unresolved):
        print(f"\n{len(unresolved)} team name(s) not in the registry (review queue):")
        for _, row in unresolved.head(20).iterrows():
            print(f"  {row['canonical_name']}")


if __name__ == "__main__":
    raise SystemExit(main())

"""Command-line entry point.

Thin dispatcher so ``predict-football <command>`` works after install, without
argparse duplicated across scripts.

Usage:
    predict-football download --league ENG_PL --from 2018 --to 2025
    predict-football status
    predict-football leagues
    predict-football schema
    predict-football backtest --league ENG_PL
    predict-football benchmark --league ENG_PL
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    """Dispatch a subcommand.

    Args:
        argv: Argument vector. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(prog="predict-football", description=__doc__)
    sub = parser.add_subparsers(dest="command")

    download = sub.add_parser("download", help="Download and clean historical data")
    download.add_argument("--league", "-l", required=True)
    download.add_argument("--from", dest="from_year", type=int)
    download.add_argument("--to", dest="to_year", type=int)
    download.add_argument("--season")
    download.add_argument("--provider", default="football_data_co")
    download.add_argument("--no-odds", action="store_true")

    sub.add_parser("status", help="Show database row counts")
    sub.add_parser("leagues", help="List known competitions")
    sub.add_parser("schema", help="Print the canonical data schema")

    backtest = sub.add_parser("backtest", help="Run a chronological walk-forward backtest")
    backtest.add_argument("--league", "-l", default="ENG_PL")
    backtest.add_argument("--min-train", type=int, default=380)
    backtest.add_argument("--refit-every", type=int, default=5)
    backtest.add_argument("--ridge", type=float, default=0.05, help="Ridge penalty on team ratings")
    backtest.add_argument("--half-life", type=float, default=None, help="Recency half-life in matches")
    backtest.add_argument("--no-market", action="store_true")
    backtest.add_argument("--calibrate", action="store_true", help="Add a walk-forward calibrated comparison")
    backtest.add_argument("--verbose", "-v", action="store_true")

    tune = sub.add_parser("tune", help="Sweep Dixon-Coles hyperparameters out of sample")
    tune.add_argument("--league", "-l", default="ENG_PL")
    tune.add_argument("--half-lives", default="none,60,120,240")
    tune.add_argument("--ridges", default="0.02,0.1")
    tune.add_argument("--min-train", type=int, default=380)
    tune.add_argument("--refit-every", type=int, default=10)
    tune.add_argument("--output", default=None)
    tune.add_argument("--verbose", "-v", action="store_true")

    benchmark = sub.add_parser("benchmark", help="Compare models against each other and the market")
    benchmark.add_argument("--league", "-l", default="ENG_PL")
    benchmark.add_argument("--models", default="dixon_coles,lightgbm")
    benchmark.add_argument("--min-train", type=int, default=380)
    benchmark.add_argument("--refit-every", type=int, default=20)
    benchmark.add_argument("--ridge", type=float, default=0.05)
    benchmark.add_argument("--half-life", type=float, default=None)
    benchmark.add_argument("--no-market", action="store_true")
    benchmark.add_argument("--verbose", "-v", action="store_true")

    poll = sub.add_parser("poll", help="Poll a live provider for the current season")
    poll.add_argument("--league", "-l", action="append", dest="leagues")
    poll.add_argument("--season")
    poll.add_argument("--provider", default="football_data_org")
    poll.add_argument("--max-age-days", type=int, default=0)
    poll.add_argument("--include-odds", action="store_true")
    poll.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "download":
        forwarded = ["--league", args.league, "--provider", args.provider]
        if args.no_odds:
            forwarded.append("--no-odds")
        if args.season:
            forwarded += ["--season", args.season]
        elif args.from_year and args.to_year:
            forwarded += ["--from", str(args.from_year), "--to", str(args.to_year)]
        else:
            print("error: provide --season or both --from and --to", file=sys.stderr)
            return 2
        return _run_script("download_data.py", forwarded)

    if args.command == "status":
        return _run_script("download_data.py", ["--status"])

    if args.command == "leagues":
        return _run_script("download_data.py", ["--list-leagues"])

    if args.command == "schema":
        return _print_schema()

    if args.command == "backtest":
        forwarded = [
            "--league",
            args.league,
            "--min-train",
            str(args.min_train),
            "--refit-every",
            str(args.refit_every),
            "--ridge",
            str(args.ridge),
        ]
        if args.half_life is not None:
            forwarded += ["--half-life", str(args.half_life)]
        if args.no_market:
            forwarded.append("--no-market")
        if args.calibrate:
            forwarded.append("--calibrate")
        if args.verbose:
            forwarded.append("--verbose")
        return _run_script("backtest.py", forwarded)

    if args.command == "tune":
        forwarded = [
            "--league",
            args.league,
            "--half-lives",
            args.half_lives,
            "--ridges",
            args.ridges,
            "--min-train",
            str(args.min_train),
            "--refit-every",
            str(args.refit_every),
        ]
        if args.output:
            forwarded += ["--output", args.output]
        if args.verbose:
            forwarded.append("--verbose")
        return _run_script("tune.py", forwarded)

    if args.command == "benchmark":
        forwarded = [
            "--league",
            args.league,
            "--models",
            args.models,
            "--min-train",
            str(args.min_train),
            "--refit-every",
            str(args.refit_every),
            "--ridge",
            str(args.ridge),
        ]
        if args.half_life is not None:
            forwarded += ["--half-life", str(args.half_life)]
        if args.no_market:
            forwarded.append("--no-market")
        if args.verbose:
            forwarded.append("--verbose")
        return _run_script("benchmark.py", forwarded)

    if args.command == "poll":
        forwarded = ["--provider", args.provider, "--max-age-days", str(args.max_age_days)]
        for league_key in args.leagues or []:
            forwarded += ["--league", league_key]
        if args.season:
            forwarded += ["--season", args.season]
        if args.include_odds:
            forwarded.append("--include-odds")
        if args.verbose:
            forwarded.append("--verbose")
        return _run_script("poll_live.py", forwarded)

    parser.print_help()
    return 0


def _run_script(script: str, argv: list[str]) -> int:
    """Execute one of the repository scripts in-process.

    Args:
        script: Script filename inside ``scripts/``.
        argv: Arguments to pass through.

    Returns:
        The script's exit code.
    """
    from pathlib import Path

    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    if not (scripts_dir / script).is_file():
        print(f"error: script not found at {scripts_dir / script}", file=sys.stderr)
        return 2

    import importlib.util

    spec = importlib.util.spec_from_file_location(f"predict_football_script_{script[:-3]}", scripts_dir / script)
    if spec is None or spec.loader is None:
        print(f"error: could not load {script}", file=sys.stderr)
        return 2
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return int(module.main(argv))


def _print_schema() -> int:
    """Print the canonical match schema as a table.

    Returns:
        Zero on success.
    """
    from predict_football.data.schema import describe_schema

    frame = describe_schema()
    print(frame.to_string(index=False))
    print(f"\n{len(frame)} columns. Availability values:")
    for value, group in frame.groupby("availability"):
        print(f"  {value}: {len(group)} columns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

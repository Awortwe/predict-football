"""Sweep Dixon-Coles hyperparameters with the walk-forward backtest.

Usage:
    python -m scripts.tune --league ENG_PL
    python -m scripts.tune --league ENG_PL --half-lives none,60,120,240 --ridges 0.02,0.1
    python -m scripts.tune --league ENG_PL --refit-every 10 --output reports/tuning.csv

Every candidate is scored only on matches that come after the ones it was fitted
on, so the winning row is an out-of-sample result and not a fit to the test set.
The whole table is printed, because a winner by a hair is not a finding.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running the script directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from predict_football.config.settings import Settings
from predict_football.data.loaders import build_repository
from predict_football.tuning import explain_winner, sweep_dixon_coles

logger = logging.getLogger("predict_football.tune")


def _parse_half_lives(text: str) -> tuple[float | None, ...]:
    """Parse a comma-separated half-life list.

    Args:
        text: Values such as ``"none,60,120"``. The case-insensitive token
            ``none`` (or ``off``) disables recency weighting.

    Returns:
        Tuple of half-lives in matches, with ``None`` for "no weighting".

    Raises:
        ValueError: If a value is not a positive number or ``none``.
    """
    values: list[float | None] = []
    for token in text.split(","):
        cleaned = token.strip().lower()
        if not cleaned:
            continue
        if cleaned in {"none", "off", "null"}:
            values.append(None)
            continue
        try:
            number = float(cleaned)
        except ValueError as error:
            raise ValueError(f"could not parse half-life {token!r}") from error
        if number <= 0:
            raise ValueError(f"half-life must be positive, got {number}")
        values.append(number)
    if not values:
        raise ValueError("no half-lives provided")
    return tuple(values)


def _parse_floats(text: str) -> tuple[float, ...]:
    """Parse a comma-separated list of non-negative floats.

    Args:
        text: Values such as ``"0.02,0.1"``.

    Returns:
        Tuple of floats.

    Raises:
        ValueError: If a value is not a non-negative number.
    """
    values: list[float] = []
    for token in text.split(","):
        cleaned = token.strip()
        if not cleaned:
            continue
        try:
            number = float(cleaned)
        except ValueError as error:
            raise ValueError(f"could not parse value {token!r}") from error
        if number < 0:
            raise ValueError(f"value must be non-negative, got {number}")
        values.append(number)
    if not values:
        raise ValueError("no values provided")
    return tuple(values)


def _build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--league", "-l", default="ENG_PL", help="Internal league key")
    parser.add_argument("--half-lives", default="none,60,120,240", help="Comma list; 'none' disables weighting")
    parser.add_argument("--ridges", default="0.02,0.1", help="Comma list of ridge penalties")
    parser.add_argument("--min-train", type=int, default=380, help="Warm-up matches before the first forecast")
    parser.add_argument("--refit-every", type=int, default=10, help="Refit after this many prediction dates")
    parser.add_argument("--output", type=Path, default=None, help="Write the full table to this CSV path")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the hyperparameter sweep.

    Args:
        argv: Argument vector. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        half_lives = _parse_half_lives(args.half_lives)
        ridges = _parse_floats(args.ridges)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    settings = Settings.from_env()
    repository = build_repository(settings)
    matches = repository.load_matches(league_key=args.league, with_result=True, include_stats=False)
    if matches.empty:
        print(f"error: no matches stored for {args.league}; run download first", file=sys.stderr)
        return 2

    combinations = len(half_lives) * len(ridges)
    print(f"{args.league}: {len(matches)} matches, sweeping {combinations} combinations")
    print(f"warm-up {args.min_train} matches, refitting every {args.refit_every} dates\n")

    def progress(label: str, position: int, total: int) -> None:
        """Report which combination is running.

        Args:
            label: Human-readable combination.
            position: One-based combination number.
            total: Total number of combinations.
        """
        print(f"[{position}/{total}] {label}")

    table = sweep_dixon_coles(
        matches,
        half_lives=half_lives,
        ridges=ridges,
        min_train_matches=args.min_train,
        refit_every_dates=args.refit_every,
        progress=progress,
    )

    print()
    print(table.to_string(index=False))
    print()
    print(explain_winner(table))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
        print(f"\nwrote {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

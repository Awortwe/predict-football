"""Run a chronological walk-forward backtest and report the scores.

Usage:
    python -m scripts.backtest --league ENG_PL
    python -m scripts.backtest --league ENG_PL --min-train 380 --refit-every 5
    python -m scripts.backtest --league ENG_PL --no-market

The backtest trains only on matches that kicked off strictly before each
predicted match, so every number printed here is out of sample. When bookmaker
closing prices are available they are scored on exactly the same matches, because
a football model that cannot be compared honestly with the market is not telling
the reader anything useful.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running the script directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from predict_football.backtest import verify_no_overlap, walk_forward_backtest
from predict_football.calibration import walk_forward_calibrate
from predict_football.config.settings import Settings
from predict_football.data.loaders import build_repository
from predict_football.evaluation import (
    base_rate_probabilities,
    compare_reports,
    evaluate,
    implied_probabilities,
)
from predict_football.models.dixon_coles import PROBABILITY_COLUMNS, DixonColesModel

logger = logging.getLogger("predict_football.backtest")

_ODDS_COLUMNS = ["odds_home_close", "odds_draw_close", "odds_away_close"]


def _build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--league", "-l", default="ENG_PL", help="Internal league key")
    parser.add_argument("--min-train", type=int, default=380, help="Warm-up matches before the first forecast")
    parser.add_argument("--refit-every", type=int, default=5, help="Refit after this many prediction dates")
    parser.add_argument("--ridge", type=float, default=0.05, help="Ridge penalty on team ratings")
    parser.add_argument("--half-life", type=float, default=None, help="Recency half-life in matches")
    parser.add_argument("--no-market", action="store_true", help="Skip the closing-odds benchmark")
    parser.add_argument("--calibrate", action="store_true", help="Add a walk-forward calibrated comparison")
    parser.add_argument(
        "--min-calibration",
        type=int,
        default=300,
        help="Completed matches required before a calibration map is trusted",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def _as_probability_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the three probability columns.

    Args:
        frame: Frame carrying the canonical probability columns.

    Returns:
        Frame of just those columns.
    """
    return frame[list(PROBABILITY_COLUMNS)]


def main(argv: list[str] | None = None) -> int:
    """Run the backtest.

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

    settings = Settings.from_env()
    repository = build_repository(settings)
    matches = repository.load_matches(league_key=args.league, with_result=True, include_stats=False)
    if matches.empty:
        print(f"error: no matches stored for {args.league}; run download first", file=sys.stderr)
        return 2

    print(f"{args.league}: {len(matches)} matches, {matches['match_date'].min()} to {matches['match_date'].max()}")
    print(f"walk-forward from a {args.min_train}-match warm-up, refitting every {args.refit_every} dates\n")

    predictions = walk_forward_backtest(
        matches,
        model_factory=lambda: DixonColesModel(ridge=args.ridge, half_life_matches=args.half_life),
        min_train_matches=args.min_train,
        refit_every_dates=args.refit_every,
    )
    verify_no_overlap(predictions, matches)
    print(f"{len(predictions)} out-of-sample predictions verified against {len(matches)} stored matches\n")

    # Calibration is fitted only on earlier matchdays, so the coverage shrinks at
    # the start of the series. Everything below is scored on that covered subset
    # so the raw model, the calibrated model and the market are comparable.
    calibrated_columns = [f"{column}_calibrated" for column in PROBABILITY_COLUMNS]
    calibrate = args.calibrate
    if calibrate:
        predictions = walk_forward_calibrate(predictions, min_calibration=args.min_calibration)
        covered = int(predictions["is_calibrated"].sum())
        print(
            f"calibration covers {covered} of {len(predictions)} predictions "
            f"(min {args.min_calibration} prior matches per map)\n"
        )
        predictions = predictions[predictions["is_calibrated"]].copy()

    reports = [
        evaluate(predictions["outcome"], _as_probability_frame(predictions), label="Dixon-Coles"),
    ]
    if calibrate:
        reports.append(
            evaluate(
                predictions["outcome"],
                predictions[calibrated_columns].to_numpy(),
                label="Dixon-Coles calibrated",
            )
        )
    reports.append(
        evaluate(
            predictions["outcome"],
            base_rate_probabilities(predictions["outcome"]),
            label="Base rate",
        )
    )

    if not args.no_market:
        try:
            odds = repository.load_odds(league_keys=[args.league], market="1x2")
        except Exception as error:  # pragma: no cover - depends on stored data
            logger.warning("could not load odds for the market benchmark: %s", error)
            odds = pd.DataFrame()

        if not odds.empty and set(_ODDS_COLUMNS).issubset(odds.columns):
            prices = odds.dropna(subset=_ODDS_COLUMNS).copy()
            prices["match_date"] = pd.to_datetime(prices["match_date"])
            merged = predictions.merge(
                prices[["match_date", "home_team", "away_team", *_ODDS_COLUMNS]],
                on=["match_date", "home_team", "away_team"],
                how="inner",
            )
            if len(merged) < len(predictions):
                print(
                    f"note: closing prices cover {len(merged)} of {len(predictions)} predictions; "
                    "the market line below is scored only on those matches\n"
                )
            if len(merged):
                market = implied_probabilities(merged, columns=tuple(_ODDS_COLUMNS))
                reports.append(evaluate(merged["outcome"], market, label="Closing market"))
                # Re-score the model on the market's subset so the comparison is fair.
                reports[0] = evaluate(
                    merged["outcome"],
                    _as_probability_frame(merged),
                    label="Dixon-Coles",
                )
                if calibrate:
                    reports[1] = evaluate(
                        merged["outcome"],
                        merged[calibrated_columns].to_numpy(),
                        label="Dixon-Coles calibrated",
                    )
                predictions = merged

    print("=" * 70)
    for report in reports:
        print(report)
        print()
    print(compare_reports(reports).to_string(index=False))

    # Season by season, so a good average cannot conceal a bad year.
    if "season" in predictions.columns:
        print("\nby season (Brier, lower better):")
        encoded = predictions["outcome"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
        ideal = np.eye(3)[encoded]
        model_error = np.sum((predictions[list(PROBABILITY_COLUMNS)].to_numpy() - ideal) ** 2, axis=1)
        predictions = predictions.assign(_model_error=model_error)
        for season, group in predictions.groupby("season"):
            print(f"  {season}: n={len(group):<5} Brier {group['_model_error'].mean():.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Compare pre-match models against each other and the closing market.

Usage:
    python -m scripts.benchmark --league ENG_PL
    python -m scripts.benchmark --league ENG_PL --models lightgbm
    python -m scripts.benchmark --league ENG_PL --no-market --refit-every 10

Every model is run through the same chronological walk-forward and then re-scored
on the **intersection** of matches all of them covered, together with the base
rate and, when prices are stored, the de-vigged closing market. That intersection
is the whole point: a model cannot win by being measured on a different, easier
set of fixtures than its rival. Like every score in this project, each row states
the number of matches it was measured on.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from pathlib import Path

# Allow running the script directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from predict_football.benchmark import BenchmarkError, benchmark_models
from predict_football.config.settings import Settings
from predict_football.data.loaders import build_repository
from predict_football.inference import ModelOptions
from predict_football.models.base import PredictiveModel
from predict_football.models.feature_model import FeatureModelOptions

logger = logging.getLogger("predict_football.benchmark")

#: Registry of buildable models, keyed by the name a user types.
_MODEL_NAMES = ("dixon_coles", "lightgbm")

#: Display labels, kept stable because the report order follows them.
_DISPLAY = {"dixon_coles": "Dixon-Coles", "lightgbm": "LightGBM"}


def _build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--league", "-l", default="ENG_PL", help="Internal league key")
    parser.add_argument("--models", default="dixon_coles,lightgbm", help="Comma-separated models to compare")
    parser.add_argument("--min-train", type=int, default=380, help="Warm-up matches before the first forecast")
    parser.add_argument(
        "--refit-every",
        type=int,
        default=20,
        help="Refit after this many prediction dates (larger is faster and slightly staler)",
    )
    parser.add_argument("--ridge", type=float, default=0.05, help="Ridge penalty (Dixon-Coles only)")
    parser.add_argument("--half-life", type=float, default=None, help="Recency half-life (Dixon-Coles only)")
    parser.add_argument("--no-market", action="store_true", help="Skip the closing-odds benchmark")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def _model_factories(
    names: list[str],
    *,
    ridge: float,
    half_life: float | None,
) -> dict[str, Callable[[], PredictiveModel]]:
    """Turn model names into fresh factories.

    Args:
        names: Registry names to build.
        ridge: Ridge penalty, applied to Dixon-Coles.
        half_life: Recency half-life, applied to Dixon-Coles.

    Returns:
        Mapping of display label to a factory, in the requested order.

    Raises:
        ValueError: If a name is not in the registry.
    """
    factories: dict[str, Callable[[], PredictiveModel]] = {}
    for name in names:
        key = name.strip().lower()
        if key not in _MODEL_NAMES:
            raise ValueError(f"unknown model {name!r}; choose from {list(_MODEL_NAMES)}")
        if key == "lightgbm":
            factories[_DISPLAY[key]] = FeatureModelOptions().build
        else:
            factories[_DISPLAY[key]] = ModelOptions(ridge=ridge, half_life_matches=half_life).build
    return factories


def _season_breakdown(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute per-season Brier scores for each model.

    The average can conceal a bad year, so the breakdown is printed rather than
    left for a reader to assume the mean holds everywhere.

    Args:
        predictions: Aligned predictions carrying ``outcome``, ``season`` and one
            probability triple per model.

    Returns:
        Frame with a row per season and a Brier column per model.
    """
    if "season" not in predictions.columns:
        return pd.DataFrame()

    frame = predictions.copy()
    outcomes = frame["outcome"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    ideal = np.eye(3)[outcomes]
    labels: list[str] = []
    for column in frame.columns:
        if not column.startswith("prob_home__"):
            continue
        label = column.split("__", 1)[1]
        triple = frame[[f"prob_{outcome}__{label}" for outcome in ("home", "draw", "away")]].to_numpy()
        frame[f"_brier__{label}"] = np.sum((triple - ideal) ** 2, axis=1)
        labels.append(label)

    rows: list[dict[str, object]] = []
    for season, group in frame.groupby("season", sort=True):
        row: dict[str, object] = {"season": season, "n_matches": len(group)}
        for label in labels:
            row[label] = float(group[f"_brier__{label}"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark.

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
        factories = _model_factories(args.models.split(","), ridge=args.ridge, half_life=args.half_life)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    settings = Settings.from_env()
    repository = build_repository(settings)

    matches = repository.load_matches(league_key=args.league, with_result=True, include_stats=False)
    if matches is None or matches.empty:
        print(f"error: no completed matches stored for {args.league}", file=sys.stderr)
        return 2

    odds = None
    if not args.no_market:
        try:
            odds = repository.load_odds(league_keys=[args.league], market="1x2")
        except Exception as error:  # pragma: no cover - depends on stored data
            logger.warning("could not load odds for the market benchmark: %s", error)
            odds = None

    try:
        result = benchmark_models(
            matches,
            odds,
            league_key=args.league,
            models=factories,
            min_train_matches=args.min_train,
            refit_every_dates=args.refit_every,
            include_market=not args.no_market,
        )
    except BenchmarkError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(
        f"{args.league}: {result.n_matches} common out-of-sample matches, "
        f"warm-up {args.min_train}, refitting every {args.refit_every} dates"
    )
    if result.has_market:
        print(f"closing prices cover all {result.market_matches} scored matches")
    print()

    print("=" * 70)
    for report in result.reports:
        print(report)
        print()
    print(result.comparison().to_string(index=False))

    seasons = matches[["match_id", "season"]].drop_duplicates("match_id")
    scored = result.predictions.merge(seasons, on="match_id", how="left")
    breakdown = _season_breakdown(scored)
    if not breakdown.empty:
        print("\nby season (Brier, lower better):")
        metric_columns = [column for column in breakdown.columns if column not in {"season", "n_matches"}]
        for _, row in breakdown.iterrows():
            scores = "  ".join(f"{column} {row[column]:.4f}" for column in metric_columns)
            print(f"  {row['season']}: n={int(row['n_matches']):<5} {scores}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

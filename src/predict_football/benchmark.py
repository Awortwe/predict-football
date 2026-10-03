"""Head-to-head benchmarking of pre-match models on the same matches.

Comparing two models by running each on whatever matches it happens to predict
is the easiest way to manufacture a winner: the closing market only prices some
fixtures, a warm-up cut can exclude different matches, and a model that predicts
an easier subset looks better. This module removes all of that freedom.

Every model is scored with the same chronological walk-forward and then all of
them -- including the base rate and, when prices are stored, the de-vigged
closing market -- are re-scored on the **intersection** of matches every model
covered. The sample size is reported, because a win over 40 matches is not a win
over 380.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import pandas as pd

from predict_football.backtest import walk_forward_backtest
from predict_football.evaluation import (
    ForecastReport,
    base_rate_probabilities,
    compare_reports,
    evaluate,
    implied_probabilities,
)
from predict_football.inference import ModelOptions
from predict_football.models.base import PredictiveModel
from predict_football.models.dixon_coles import PROBABILITY_COLUMNS
from predict_football.models.feature_model import FeatureModelOptions

module_logger = logging.getLogger(__name__)

#: Closing-odds columns used for the market benchmark. Closing prices are set at
#: kickoff, so they are a benchmark only and never a model input.
_ODDS_COLUMNS = ("odds_home_close", "odds_draw_close", "odds_away_close")


class BenchmarkError(ValueError):
    """Raised when a benchmark cannot be run as specified."""


@dataclass(frozen=True)
class ModelBenchmark:
    """Scores for several models, all measured on the same matches.

    Attributes:
        league_key: Competition benchmarked, for labelling.
        n_matches: Matches every model is scored on.
        market_matches: Matches the closing market covered. Equal to
            ``n_matches`` when the market is present, since the whole comparison
            is restricted to it.
        reports: One report per model, then the base rate, then the market.
        predictions: One row per scored match, carrying the observed outcome and
            each model's probabilities as ``prob_home__<label>``.
        min_train_matches: Warm-up used for the walk.
    """

    league_key: str
    n_matches: int
    market_matches: int
    reports: tuple[ForecastReport, ...] = field(default_factory=tuple)
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    min_train_matches: int = 380

    @property
    def has_market(self) -> bool:
        """Whether a closing-market benchmark is included."""
        return self.market_matches > 0

    def comparison(self) -> pd.DataFrame:
        """Tabulate the reports side by side.

        Returns:
            Frame with one row per report, including the sample size each metric
            was computed on.
        """
        return compare_reports(list(self.reports))


def _default_models() -> dict[str, Callable[[], PredictiveModel]]:
    """Return the models benchmarked unless a caller overrides them.

    Returns:
        Mapping of display label to a model factory. Fresh factories are built
        per call so no fitted state is shared between runs.
    """
    return {
        "Dixon-Coles": ModelOptions().build,
        "LightGBM": FeatureModelOptions().build,
    }


def _walk_forward(
    matches: pd.DataFrame,
    models: Mapping[str, Callable[[], PredictiveModel]],
    min_train_matches: int,
    refit_every_dates: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Run every model's walk-forward and align the forecasts on match id.

    Args:
        matches: Completed canonical match frame.
        models: Mapping of display label to a model factory.
        min_train_matches: Warm-up forwarded to the backtest.
        refit_every_dates: Refit cadence forwarded to the backtest.

    Returns:
        Tuple of the aligned frame and the model labels in label order.

    Raises:
        BenchmarkError: If the walk-forward is impossible, or the models do not
            share a single predicted match.
    """
    labels = list(models)
    aligned: pd.DataFrame | None = None
    for label in labels:
        try:
            predictions = walk_forward_backtest(
                matches,
                models[label],
                min_train_matches=min_train_matches,
                refit_every_dates=refit_every_dates,
            )
        except ValueError as error:  # BacktestError subclasses ValueError
            raise BenchmarkError(f"benchmark_models: {label} could not be scored: {error}") from error

        renamed = predictions.rename(columns={column: f"{column}__{label}" for column in PROBABILITY_COLUMNS})
        keep = [
            "match_id",
            "match_date",
            "home_team",
            "away_team",
            "outcome",
            *(f"{column}__{label}" for column in PROBABILITY_COLUMNS),
        ]
        renamed = renamed[keep]
        if aligned is None:
            aligned = renamed
        else:
            aligned = aligned.merge(
                renamed.drop(columns=["match_date", "home_team", "away_team", "outcome"]),
                on="match_id",
                how="inner",
            )

    if aligned is None or aligned.empty:
        raise BenchmarkError("benchmark_models: the models share no predicted match")
    return aligned, labels


def benchmark_models(
    matches: pd.DataFrame,
    odds: pd.DataFrame | None = None,
    *,
    league_key: str = "",
    models: Mapping[str, Callable[[], PredictiveModel]] | None = None,
    min_train_matches: int = 380,
    refit_every_dates: int = 5,
    include_market: bool = True,
) -> ModelBenchmark:
    """Score several models out of sample on the same matches.

    Args:
        matches: Completed canonical match frame.
        odds: Closing-odds frame, or ``None`` when none were loaded.
        league_key: Competition key, carried through for labelling.
        models: Mapping of display label to a model factory. Defaults to
            Dixon-Coles and LightGBM.
        min_train_matches: Warm-up matches before the first forecast.
        refit_every_dates: Distinct matchdays predicted per fitted model.
        include_market: Score the closing-odds benchmark when prices are present.

    Returns:
        A :class:`ModelBenchmark` carrying the aligned predictions and reports.

    Raises:
        BenchmarkError: If ``matches`` is empty, or a model cannot be scored.
    """
    if matches is None or matches.empty:
        raise BenchmarkError("benchmark_models: no completed matches were supplied")

    specifications = dict(models) if models is not None else _default_models()
    if not specifications:
        raise BenchmarkError("benchmark_models: no models were supplied")

    aligned, labels = _walk_forward(matches, specifications, min_train_matches, refit_every_dates)

    market_matches = 0
    if include_market and odds is not None and not odds.empty and set(_ODDS_COLUMNS).issubset(odds.columns):
        prices = odds.dropna(subset=list(_ODDS_COLUMNS)).copy()
        prices["match_date"] = pd.to_datetime(prices["match_date"], errors="coerce")
        aligned["match_date"] = pd.to_datetime(aligned["match_date"], errors="coerce")
        aligned = aligned.merge(
            prices[["match_date", "home_team", "away_team", *_ODDS_COLUMNS]],
            on=["match_date", "home_team", "away_team"],
            how="inner",
        )
        market_matches = len(aligned)

    reports: list[ForecastReport] = []
    for label in labels:
        probabilities = pd.DataFrame(
            {column: aligned[f"{column}__{label}"].to_numpy() for column in PROBABILITY_COLUMNS}
        )
        reports.append(evaluate(aligned["outcome"], probabilities, label=label))
    reports.append(evaluate(aligned["outcome"], base_rate_probabilities(aligned["outcome"]), label="Base rate"))

    if market_matches:
        market = implied_probabilities(aligned, columns=_ODDS_COLUMNS)
        reports.append(evaluate(aligned["outcome"], market, label="Closing market"))

    module_logger.info(
        "Benchmarked %s across %d matches (market %d)",
        ", ".join(labels),
        len(aligned),
        market_matches,
    )
    return ModelBenchmark(
        league_key=league_key,
        n_matches=len(aligned),
        market_matches=market_matches,
        reports=tuple(reports),
        predictions=aligned,
        min_train_matches=min_train_matches,
    )


def benchmark_league(
    repository: object,
    league_key: str,
    *,
    models: Mapping[str, Callable[[], PredictiveModel]] | None = None,
    min_train_matches: int = 380,
    refit_every_dates: int = 5,
    include_market: bool = True,
) -> ModelBenchmark:
    """Benchmark a competition loaded from a repository.

    Args:
        repository: A :class:`~predict_football.data.repository.MatchRepository`.
        league_key: Internal competition key.
        models: Mapping of display label to a model factory, as in
            :func:`benchmark_models`.
        min_train_matches: Warm-up matches before the first forecast.
        refit_every_dates: Distinct matchdays predicted per fitted model.
        include_market: Score the closing-odds benchmark when prices are stored.

    Returns:
        A :class:`ModelBenchmark`.

    Raises:
        BenchmarkError: If the competition has no completed matches.
    """
    matches = repository.load_matches(league_key=league_key, with_result=True, include_stats=False)
    if matches is None or matches.empty:
        raise BenchmarkError(f"benchmark_league: no completed matches stored for {league_key}")

    odds: pd.DataFrame | None = None
    if include_market:
        try:
            odds = repository.load_odds(league_keys=[league_key], market="1x2")
        except Exception as error:  # pragma: no cover - depends on stored data
            module_logger.warning("could not load odds for the market benchmark: %s", error)
            odds = None

    return benchmark_models(
        matches,
        odds,
        league_key=league_key,
        models=models,
        min_train_matches=min_train_matches,
        refit_every_dates=refit_every_dates,
        include_market=include_market,
    )


__all__ = ["BenchmarkError", "ModelBenchmark", "benchmark_league", "benchmark_models"]

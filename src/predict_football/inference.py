"""Leakage-safe inference for fixtures that have not been played yet.

The model in :mod:`predict_football.models.dixon_coles` is deliberately ignorant
of dates; it fits whatever rows it is handed. This module is the boundary that
supplies those rows, and it is the only place that decides which matches are
allowed to inform a prediction. Its one rule is the same one the backtest uses:

    a fixture is forecast by a model fitted on matches whose kickoff is
    **strictly earlier** than the fixture's own date.

Two consequences are surfaced rather than hidden:

* A fixture without enough prior history is left unforecast, with the shortfall
  recorded, instead of being given a confident number from a thin fit.
* A club the training window has never seen still receives the league-average
  rating the model falls back to, but the fixture is flagged ``unknown_teams``
  so the reader knows the forecast rests on no evidence about that club.

Nothing here writes to the database or reaches the network; it is a pure
function of the frames a caller already holds, which keeps it testable offline
and auditable by reading one file.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from predict_football.backtest import verify_no_overlap, walk_forward_backtest
from predict_football.evaluation import (
    ForecastReport,
    base_rate_probabilities,
    compare_reports,
    evaluate,
    implied_probabilities,
)
from predict_football.models.base import PredictiveModel
from predict_football.models.dixon_coles import MAX_GOALS, PROBABILITY_COLUMNS, DixonColesModel

module_logger = logging.getLogger(__name__)

#: Columns a history frame must carry to fit a model.
_HISTORY_REQUIRED = ("match_date", "home_team", "away_team", "home_goals", "away_goals")

#: Columns a fixture frame must carry to be forecast.
_FIXTURE_REQUIRED = ("match_date", "home_team", "away_team")

#: Fixture columns carried through to the forecast, when present.
_IDENTIFIER_COLUMNS = (
    "match_id",
    "league_key",
    "season",
    "season_code",
    "match_date",
    "home_team",
    "away_team",
)

#: Closing-odds columns used for the market benchmark. Closing prices are set at
#: kickoff, so they are a benchmark only and never a model input.
_ODDS_COLUMNS = ["odds_home_close", "odds_draw_close", "odds_away_close"]


class InferenceError(ValueError):
    """Raised when a forecast cannot be produced as requested."""


@dataclass(frozen=True)
class ModelOptions:
    """Hyperparameters for the forecast model.

    Attributes:
        ridge: Ridge penalty on attack and defence. Mild shrinkage stops a club
            with a handful of matches receiving an extreme rating.
        half_life_matches: Recency half-life in matches, or ``None`` to weight
            every match equally.
        max_goals: Largest goals per side in the score matrix.
    """

    ridge: float = 0.05
    half_life_matches: float | None = None
    max_goals: int = MAX_GOALS

    def build(self) -> DixonColesModel:
        """Construct an unfitted model with these options.

        Returns:
            A fresh :class:`DixonColesModel`, so no state leaks between fits.
        """
        return DixonColesModel(
            ridge=self.ridge,
            half_life_matches=self.half_life_matches,
            max_goals=self.max_goals,
        )


def _coerce_dates(frame: pd.DataFrame, *, context: str) -> pd.DataFrame:
    """Return a copy of the frame with a real datetime column.

    Dates arrive from SQLite as ISO strings. Comparing a string to a
    ``Timestamp`` is either an error or a wrong answer depending on the format,
    so it is coerced once here and any unparseable value aborts rather than being
    guessed at.

    Args:
        frame: Frame carrying a ``match_date`` column.
        context: Caller name, used in the error message.

    Returns:
        Copy of the frame with ``match_date`` as ``datetime64``.

    Raises:
        InferenceError: If any ``match_date`` cannot be interpreted as a date.
    """
    result = frame.copy()
    result["match_date"] = pd.to_datetime(result["match_date"], errors="coerce")
    if result["match_date"].isna().any():
        unparsed = int(result["match_date"].isna().sum())
        raise InferenceError(f"{context}: could not interpret the match date on {unparsed} row(s)")
    return result


def predict_fixtures(
    history: pd.DataFrame,
    fixtures: pd.DataFrame,
    *,
    min_train_matches: int = 380,
    model_factory: Callable[[], PredictiveModel] | None = None,
) -> pd.DataFrame:
    """Forecast fixtures using only matches strictly before each fixture.

    Fixtures are grouped by matchday and each group is predicted by a model
    refitted on the history available before that date. Same-day matches never
    inform each other, which is the same discipline the walk-forward backtest
    applies.

    Args:
        history: Completed matches with dates and goals. Rows without a result
            are ignored, since a fixture cannot train on an unknown outcome.
        fixtures: Fixtures to forecast. Their results, if any are present, are
            ignored.
        min_train_matches: Prior matches required before a fixture is forecast.
            Below this the row is returned with null probabilities and an
            ``is_forecast`` flag of False, so a thin fit is never presented as a
            prediction.
        model_factory: Callable returning an unfitted model. Defaults to
            :class:`ModelOptions` with its standard hyperparameters, and is
            called once per matchday so folds share no state.

    Returns:
        One row per fixture in the caller's order, carrying the fixture
        identifiers, ``lambda_home``/``lambda_away``, the three probability
        columns, ``n_train_matches``, ``unknown_teams``, ``model_converged``,
        ``model_rho``, ``is_forecast`` and a human-readable ``note`` where a row
        was skipped. ``lambda_*`` and ``model_rho`` are null for a model that
        does not produce them.

    Raises:
        InferenceError: If either frame is empty, lacks a required column, holds
            an unparseable date, or the history contains no completed match.
    """
    if history is None or history.empty:
        raise InferenceError("predict_fixtures: history is empty, so nothing can be trained")
    if fixtures is None or fixtures.empty:
        raise InferenceError("predict_fixtures: no fixtures were supplied")

    missing_history = set(_HISTORY_REQUIRED) - set(history.columns)
    if missing_history:
        raise InferenceError(f"predict_fixtures: history is missing {sorted(missing_history)}")
    missing_fixtures = set(_FIXTURE_REQUIRED) - set(fixtures.columns)
    if missing_fixtures:
        raise InferenceError(f"predict_fixtures: fixtures are missing {sorted(missing_fixtures)}")

    train_all = _coerce_dates(history, context="predict_fixtures").dropna(
        subset=["home_goals", "away_goals"]
    )
    if train_all.empty:
        raise InferenceError("predict_fixtures: history has no completed match")

    frame = _coerce_dates(fixtures, context="predict_fixtures").reset_index(drop=True)
    identifiers = [column for column in _IDENTIFIER_COLUMNS if column in frame.columns]

    out = frame[identifiers].copy()
    out["lambda_home"] = np.nan
    out["lambda_away"] = np.nan
    for column in PROBABILITY_COLUMNS:
        out[column] = np.nan
    out["n_train_matches"] = 0
    out["unknown_teams"] = 0
    out["model_converged"] = pd.Series([pd.NA] * len(out), dtype="boolean")
    out["is_forecast"] = False
    out["note"] = ""
    # The low-score dependence parameter, when the model has one. It is carried
    # through so a caller can rebuild the full scoreline distribution from the
    # forecast without refitting (see models.dixon_coles.score_matrix).
    out["model_rho"] = np.nan

    factory = model_factory if model_factory is not None else ModelOptions().build

    for date in sorted(frame["match_date"].unique()):
        mask = (frame["match_date"] == date).to_numpy()
        train = train_all[train_all["match_date"] < date]
        n_train = len(train)
        if n_train < min_train_matches:
            out.loc[mask, "n_train_matches"] = n_train
            out.loc[mask, "note"] = f"needs {min_train_matches} prior matches; {n_train} available"
            continue

        model = factory()
        model.fit(train)
        test = frame.loc[mask]
        predicted = model.outcome_probabilities(test)

        train_teams = set(train["home_team"]) | set(train["away_team"])
        fixture_teams = set(test["home_team"]) | set(test["away_team"])
        unknown = sorted(fixture_teams - train_teams)

        # A rate-based model reports expected goals; a classifier need not, so
        # the columns are copied only when the model actually produced them.
        for column in ("lambda_home", "lambda_away"):
            if column in predicted.columns:
                out.loc[mask, column] = predicted[column].to_numpy()
        for column in PROBABILITY_COLUMNS:
            out.loc[mask, column] = predicted[column].to_numpy()
        out.loc[mask, "n_train_matches"] = n_train
        out.loc[mask, "unknown_teams"] = len(unknown)
        out.loc[mask, "model_converged"] = model.converged
        out.loc[mask, "is_forecast"] = True
        rho = getattr(model, "rho", None)
        if rho is not None:
            out.loc[mask, "model_rho"] = float(rho)
        if unknown:
            out.loc[mask, "note"] = "team(s) absent from training: " + ", ".join(unknown)

    forecast = int(out["is_forecast"].sum())
    module_logger.info(
        "Forecast %d of %d fixtures (warm-up min %d); %d fixture(s) lacked history",
        forecast,
        len(out),
        min_train_matches,
        len(out) - forecast,
    )
    return out


def forecast_league(
    repository: object,
    league_key: str,
    *,
    min_train_matches: int = 380,
    model_factory: Callable[[], PredictiveModel] | None = None,
) -> pd.DataFrame:
    """Forecast every stored fixture of a competition that lacks a result.

    Args:
        repository: A :class:`~predict_football.data.repository.MatchRepository`.
        league_key: Internal competition key, e.g. ``"ENG_PL"``.
        min_train_matches: Prior matches required before a fixture is forecast.
        model_factory: Optional model constructor, as in :func:`predict_fixtures`.

    Returns:
        Forecast frame as returned by :func:`predict_fixtures`, restricted to
        unplayed fixtures.

    Raises:
        InferenceError: If the competition has no stored matches, or every
            stored match already has a result so there is nothing to forecast.
    """
    matches = repository.load_matches(league_key=league_key, include_stats=False)
    if matches is None or matches.empty:
        raise InferenceError(f"forecast_league: no matches stored for {league_key}")

    complete = (
        matches["result"].notna()
        & matches["home_goals"].notna()
        & matches["away_goals"].notna()
    )
    history = matches[complete]
    fixtures = matches[~complete]
    if fixtures.empty:
        raise InferenceError(f"forecast_league: every stored match for {league_key} already has a result")
    if history.empty:
        raise InferenceError(f"forecast_league: {league_key} has no completed matches to train on")

    return predict_fixtures(
        history,
        fixtures,
        min_train_matches=min_train_matches,
        model_factory=model_factory,
    )


@dataclass(frozen=True)
class LeagueEvaluation:
    """Out-of-sample diagnostics for one competition.

    Attributes:
        league_key: Competition evaluated.
        n_played: Completed matches available.
        n_predictions: Out-of-sample forecasts produced.
        predictions: The forecast frame, carrying the observed outcome.
        reports: Scored reports. The raw model and base rate are always present;
            the closing market is appended when prices are stored.
        market_matches: Matches the market covered. When the market is present,
            every report is scored on this same subset so the comparison is fair.
        min_train_matches: Warm-up used for the walk.
    """

    league_key: str
    n_played: int
    n_predictions: int
    predictions: pd.DataFrame
    reports: tuple[ForecastReport, ...] = field(default_factory=tuple)
    market_matches: int = 0
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


def evaluate_frames(
    matches: pd.DataFrame,
    odds: pd.DataFrame | None = None,
    *,
    league_key: str = "",
    min_train_matches: int = 380,
    refit_every_dates: int = 5,
    model_factory: Callable[[], PredictiveModel] | None = None,
    include_market: bool = True,
) -> LeagueEvaluation:
    """Score completed matches out of sample from frames already in memory.

    This is the pure core of :func:`evaluate_league`, split out so a caller that
    holds the frames (such as the app's cache) does not need a repository. When
    closing prices exist every forecast is re-scored on the market's subset, so a
    model that looks better on a different set of matches cannot claim the win.

    Args:
        matches: Completed canonical match frame.
        odds: Closing-odds frame, or ``None`` when none were loaded.
        league_key: Competition key, carried through for labelling.
        min_train_matches: Warm-up matches before the first forecast.
        refit_every_dates: Distinct matchdays predicted per fitted model.
        model_factory: Optional model constructor. Defaults to
            :class:`ModelOptions` with standard hyperparameters.
        include_market: Score the closing-odds benchmark when prices are present.

    Returns:
        A :class:`LeagueEvaluation` carrying the predictions and reports.

    Raises:
        InferenceError: If there are too few completed matches to walk forward
            from the requested warm-up.
    """
    if matches is None or matches.empty:
        raise InferenceError("evaluate_frames: no completed matches were supplied")

    factory = model_factory if model_factory is not None else ModelOptions().build
    try:
        predictions = walk_forward_backtest(
            matches,
            model_factory=factory,
            min_train_matches=min_train_matches,
            refit_every_dates=refit_every_dates,
        )
    except ValueError as error:  # BacktestError subclasses ValueError
        raise InferenceError(str(error)) from error

    verify_no_overlap(predictions, matches)

    reports: list[ForecastReport] = [
        evaluate(predictions["outcome"], predictions[list(PROBABILITY_COLUMNS)], label="Dixon-Coles"),
        evaluate(
            predictions["outcome"],
            base_rate_probabilities(predictions["outcome"]),
            label="Base rate",
        ),
    ]
    market_matches = 0

    if include_market and odds is not None and not odds.empty and set(_ODDS_COLUMNS).issubset(odds.columns):
        prices = odds.dropna(subset=_ODDS_COLUMNS).copy()
        prices["match_date"] = pd.to_datetime(prices["match_date"], errors="coerce")
        scored = predictions.copy()
        scored["match_date"] = pd.to_datetime(scored["match_date"], errors="coerce")
        merged = scored.merge(
            prices[["match_date", "home_team", "away_team", *_ODDS_COLUMNS]],
            on=["match_date", "home_team", "away_team"],
            how="inner",
        )
        if len(merged):
            market = implied_probabilities(merged, columns=tuple(_ODDS_COLUMNS))
            # Re-score everything on the market's subset so raw, base rate and
            # market are all measured on exactly the same matches.
            reports = [
                evaluate(merged["outcome"], merged[list(PROBABILITY_COLUMNS)], label="Dixon-Coles"),
                evaluate(
                    merged["outcome"],
                    base_rate_probabilities(merged["outcome"]),
                    label="Base rate",
                ),
                evaluate(merged["outcome"], market, label="Closing market"),
            ]
            predictions = merged
            market_matches = len(merged)

    return LeagueEvaluation(
        league_key=league_key,
        n_played=len(matches),
        n_predictions=len(predictions),
        predictions=predictions,
        reports=tuple(reports),
        market_matches=market_matches,
        min_train_matches=min_train_matches,
    )


def evaluate_league(
    repository: object,
    league_key: str,
    *,
    min_train_matches: int = 380,
    refit_every_dates: int = 5,
    model_factory: Callable[[], PredictiveModel] | None = None,
    include_market: bool = True,
) -> LeagueEvaluation:
    """Score a competition out of sample, against the base rate and the market.

    Args:
        repository: A :class:`~predict_football.data.repository.MatchRepository`.
        league_key: Internal competition key.
        min_train_matches: Warm-up matches before the first forecast.
        refit_every_dates: Distinct matchdays predicted per fitted model.
        model_factory: Optional model constructor. Defaults to
            :class:`ModelOptions` with standard hyperparameters.
        include_market: Score the closing-odds benchmark when prices are stored.

    Returns:
        A :class:`LeagueEvaluation` carrying the predictions and reports.

    Raises:
        InferenceError: If the competition has no completed matches, or too few
            to walk forward from the requested warm-up.
    """
    matches = repository.load_matches(league_key=league_key, with_result=True, include_stats=False)
    if matches is None or matches.empty:
        raise InferenceError(f"evaluate_league: no completed matches stored for {league_key}")

    odds: pd.DataFrame | None = None
    if include_market:
        try:
            odds = repository.load_odds(league_keys=[league_key], market="1x2")
        except Exception as error:  # pragma: no cover - depends on stored data
            module_logger.warning("could not load odds for the market benchmark: %s", error)
            odds = None

    return evaluate_frames(
        matches,
        odds,
        league_key=league_key,
        min_train_matches=min_train_matches,
        refit_every_dates=refit_every_dates,
        model_factory=model_factory,
        include_market=include_market,
    )


__all__ = [
    "InferenceError",
    "LeagueEvaluation",
    "ModelOptions",
    "evaluate_frames",
    "evaluate_league",
    "forecast_league",
    "predict_fixtures",
]

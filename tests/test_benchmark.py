"""Tests for the head-to-head model benchmark.

The benchmark exists to stop a model winning by being scored on easier matches.
These tests pin down that property: every report covers the same matches, and,
when closing prices exist, the whole comparison is restricted to the market's
subset so the market cannot be compared on a different sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predict_football.benchmark import BenchmarkError, ModelBenchmark, benchmark_league, benchmark_models
from predict_football.evaluation import compare_reports
from predict_football.models.dixon_coles import PROBABILITY_COLUMNS


def _synthetic_league(n_teams: int = 6, rounds: int = 40, seed: int = 20260810) -> pd.DataFrame:
    """Build a synthetic league with a known quality ordering and home edge.

    Args:
        n_teams: Number of clubs.
        rounds: Number of matchdays.
        seed: Random seed, fixed so the tests never flake.

    Returns:
        Match frame with one match per club per matchday.
    """
    rng = np.random.default_rng(seed)
    strengths = np.linspace(0.9, 2.3, n_teams)
    teams = [f"T{i:02d}" for i in range(n_teams)]

    rows: list[dict[str, object]] = []
    for round_index in range(rounds):
        order = list(rng.permutation(n_teams))
        for a, b in zip(order[::2], order[1::2], strict=True):
            home_goals = int(rng.poisson(strengths[a] + 0.35))
            away_goals = int(rng.poisson(strengths[b]))
            result = "H" if home_goals > away_goals else "A" if home_goals < away_goals else "D"
            rows.append(
                {
                    "match_id": f"m{round_index:03d}_{a}_{b}",
                    "match_date": pd.Timestamp("2021-01-01") + pd.Timedelta(days=7 * round_index),
                    "home_team": teams[a],
                    "away_team": teams[b],
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "result": result,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def league() -> pd.DataFrame:
    """Return a synthetic league ready to backtest."""
    return _synthetic_league()


class _ConstantModel:
    """Stub model that predicts a fixed, valid distribution."""

    def __init__(self, home: float = 0.45, draw: float = 0.25, away: float = 0.30) -> None:
        self.weights = {"prob_home": home, "prob_draw": draw, "prob_away": away}
        self.trained_ids: set[str] = set()

    def fit(self, matches: pd.DataFrame) -> _ConstantModel:
        """Record the training identities without learning anything.

        Args:
            matches: Training frame.

        Returns:
            The stub, for chaining.
        """
        self.trained_ids = set(matches["match_id"])
        return self

    def outcome_probabilities(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Return a fixed forecast.

        Args:
            matches: Frame to predict.

        Returns:
            Forecast frame with the fixture identifiers and probabilities.
        """
        frame = pd.DataFrame(
            {
                "match_id": matches["match_id"].to_numpy(),
                "match_date": matches["match_date"].to_numpy(),
                "home_team": matches["home_team"].to_numpy(),
                "away_team": matches["away_team"].to_numpy(),
            }
        )
        for column, value in self.weights.items():
            frame[column] = value
        return frame


def _constant_factory(home: float = 0.45, draw: float = 0.25, away: float = 0.30):
    """Return a factory producing a fresh constant model.

    Args:
        home: Home-win probability.
        draw: Draw probability.
        away: Away-win probability.

    Returns:
        A zero-argument callable returning a new stub.
    """
    return lambda: _ConstantModel(home, draw, away)


def _odds_for(league: pd.DataFrame, *, drop_match_ids: set[str] | None = None) -> pd.DataFrame:
    """Build a closing-odds frame for the league.

    Args:
        league: Match frame supplying dates and clubs.
        drop_match_ids: Matches to omit, to force a proper subset.

    Returns:
        Odds frame with the canonical closing columns.
    """
    odds = league[["match_id", "match_date", "home_team", "away_team"]].copy()
    if drop_match_ids:
        odds = odds[~odds["match_id"].isin(drop_match_ids)]
    odds["odds_home_close"] = 2.0
    odds["odds_draw_close"] = 3.4
    odds["odds_away_close"] = 3.8
    return odds.reset_index(drop=True)


def test_benchmark_scores_every_model_on_the_same_matches(league: pd.DataFrame) -> None:
    """All reports must share one sample size, whatever the model."""
    result = benchmark_models(
        league,
        None,
        league_key="SYN",
        models={"Home bias": _constant_factory(), "Flat": _constant_factory(1 / 3, 1 / 3, 1 / 3)},
        min_train_matches=40,
        refit_every_dates=5,
    )

    assert isinstance(result, ModelBenchmark)
    assert [report.label for report in result.reports] == ["Home bias", "Flat", "Base rate"]
    assert {report.n_matches for report in result.reports} == {result.n_matches}
    assert result.n_matches > 0
    assert not result.has_market


def test_benchmark_reports_base_rate_and_market_in_order(league: pd.DataFrame) -> None:
    """The market benchmark must be last, after every model and the base rate."""
    result = benchmark_models(
        league,
        _odds_for(league),
        models={"Flat": _constant_factory(1 / 3, 1 / 3, 1 / 3)},
        min_train_matches=40,
        refit_every_dates=5,
    )

    assert [report.label for report in result.reports] == ["Flat", "Base rate", "Closing market"]
    assert result.has_market


def test_benchmark_restricts_the_comparison_to_the_market_subset(league: pd.DataFrame) -> None:
    """When prices cover only some matches, every report must shrink with them.

    Otherwise the market is scored on its own fixtures while the model is scored
    on all of them, and the two numbers are not comparable at all.
    """
    models = {"Flat": _constant_factory(1 / 3, 1 / 3, 1 / 3)}
    full = benchmark_models(league, None, models=models, min_train_matches=40, refit_every_dates=5)

    dropped = set(sorted(full.predictions["match_id"])[:5])
    result = benchmark_models(
        league,
        _odds_for(league, drop_match_ids=dropped),
        models=models,
        min_train_matches=40,
        refit_every_dates=5,
    )

    assert result.n_matches == full.n_matches - len(dropped)
    assert result.market_matches == result.n_matches
    assert {report.n_matches for report in result.reports} == {result.n_matches}


def test_benchmark_comparison_keeps_the_sample_size(league: pd.DataFrame) -> None:
    """The tabulated comparison must keep n_matches visible per row."""
    result = benchmark_models(
        league,
        None,
        models={"Flat": _constant_factory(1 / 3, 1 / 3, 1 / 3)},
        min_train_matches=40,
        refit_every_dates=5,
    )

    table = compare_reports(list(result.reports))

    assert list(table["label"]) == ["Flat", "Base rate"]
    assert (table["n_matches"] == result.n_matches).all()


def test_default_models_are_dixon_coles_then_lightgbm(league: pd.DataFrame) -> None:
    """The built-in benchmark must compare the baseline against the challenger."""
    result = benchmark_models(league, None, min_train_matches=40, refit_every_dates=10)

    assert [report.label for report in result.reports] == ["Dixon-Coles", "LightGBM", "Base rate"]


def test_benchmark_predictions_are_a_probability_frame(league: pd.DataFrame) -> None:
    """The aligned predictions must carry valid distributions per model."""
    result = benchmark_models(
        league,
        None,
        models={"Home bias": _constant_factory(), "Flat": _constant_factory(1 / 3, 1 / 3, 1 / 3)},
        min_train_matches=40,
        refit_every_dates=5,
    )

    for column in PROBABILITY_COLUMNS:
        assert f"{column}__Flat" in result.predictions.columns
        assert f"{column}__Home bias" in result.predictions.columns
    totals = sum(result.predictions[f"{column}__Flat"] for column in PROBABILITY_COLUMNS).to_numpy()
    np.testing.assert_allclose(totals, 1.0)


def test_benchmark_rejects_an_empty_frame() -> None:
    """No matches means nothing can be compared."""
    with pytest.raises(BenchmarkError, match="no completed matches"):
        benchmark_models(pd.DataFrame(), None)


def test_benchmark_rejects_an_empty_model_set(league: pd.DataFrame) -> None:
    """A benchmark with no models is a caller bug, not an empty result."""
    with pytest.raises(BenchmarkError, match="no models"):
        benchmark_models(league, None, models={}, min_train_matches=40)


class _StubRepository:
    """Repository double exposing only the two load methods the benchmark uses."""

    def __init__(self, matches: pd.DataFrame, odds: pd.DataFrame | None = None) -> None:
        self._matches = matches
        self._odds = odds if odds is not None else pd.DataFrame()

    def load_matches(self, *, with_result: bool = False, **_: object) -> pd.DataFrame:
        """Return stored matches, optionally only those with a result."""
        frame = self._matches.copy()
        if with_result and "result" in frame.columns:
            frame = frame[frame["result"].notna()]
        return frame

    def load_odds(self, **_: object) -> pd.DataFrame:
        """Return the stored odds frame."""
        return self._odds.copy()


def test_benchmark_league_scores_models_and_base_rate(league: pd.DataFrame) -> None:
    """The league loader must score a model and the base rate on one sample."""
    result = benchmark_league(
        _StubRepository(league),
        "SYN",
        models={"Flat": _constant_factory(1 / 3, 1 / 3, 1 / 3)},
        min_train_matches=40,
        refit_every_dates=5,
    )

    assert result.league_key == "SYN"
    assert [report.label for report in result.reports] == ["Flat", "Base rate"]
    assert result.n_matches > 0
    assert not result.has_market
    assert {report.n_matches for report in result.reports} == {result.n_matches}


def test_benchmark_league_uses_stored_odds_for_the_market(league: pd.DataFrame) -> None:
    """Stored closing prices must be loaded and used for the market benchmark."""
    result = benchmark_league(
        _StubRepository(league, odds=_odds_for(league)),
        "SYN",
        models={"Flat": _constant_factory(1 / 3, 1 / 3, 1 / 3)},
        min_train_matches=40,
        refit_every_dates=5,
    )

    assert [report.label for report in result.reports] == ["Flat", "Base rate", "Closing market"]
    assert result.has_market
    assert result.market_matches == result.n_matches


def test_benchmark_league_rejects_a_league_without_matches() -> None:
    """A competition with no completed matches is reported, not silently empty."""
    with pytest.raises(BenchmarkError, match="no completed matches"):
        benchmark_league(_StubRepository(pd.DataFrame()), "NOPE")

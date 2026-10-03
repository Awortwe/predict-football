"""Tests for leakage-safe inference.

The point of these tests is not that the model produces a particular number --
it will not, and copying a figure out of the implementation would only assert
consistency. It is that the module never trains on a match dated at or after the
fixture it is predicting, never forecasts without enough history, and says so
plainly when it declines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predict_football.inference import (
    InferenceError,
    ModelOptions,
    evaluate_league,
    forecast_league,
    predict_fixtures,
)
from predict_football.models.dixon_coles import PROBABILITY_COLUMNS


def _fixture(home: str, away: str, date: str) -> pd.DataFrame:
    """Build a one-row fixture frame for tests.

    Args:
        home: Home club.
        away: Away club.
        date: Kickoff date.

    Returns:
        A fixture frame with the minimal required columns.
    """
    return pd.DataFrame(
        {
            "match_id": [f"{home}-{away}-{date}"],
            "match_date": [date],
            "home_team": [home],
            "away_team": [away],
        }
    )


class _StubRepository:
    """Minimal repository double exposing only the two load methods used."""

    def __init__(self, matches: pd.DataFrame, odds: pd.DataFrame | None = None) -> None:
        self._matches = matches
        self._odds = odds if odds is not None else pd.DataFrame()

    def load_matches(self, *, with_result: bool = False, **_: object) -> pd.DataFrame:
        """Return the stored matches, optionally only completed ones."""
        frame = self._matches.copy()
        if with_result:
            frame = frame[frame["result"].notna()]
        return frame

    def load_odds(self, **_: object) -> pd.DataFrame:
        """Return the stored odds frame."""
        return self._odds.copy()


def test_predict_fixtures_returns_valid_distributions(synthetic_league: pd.DataFrame) -> None:
    """A forecast is a probability distribution over the three outcomes."""
    history = synthetic_league[synthetic_league["match_date"] <= pd.Timestamp("2021-01-29")]
    fixture = _fixture("T00", "T01", "2021-02-05")

    result = predict_fixtures(history, fixture, min_train_matches=15)

    row = result.iloc[0]
    assert bool(row["is_forecast"]) is True
    assert row["n_train_matches"] == len(history)
    probabilities = row[list(PROBABILITY_COLUMNS)].to_numpy(dtype="float64")
    assert np.all(probabilities > 0)
    assert np.isclose(probabilities.sum(), 1.0)
    assert row["lambda_home"] > 0
    assert row["lambda_away"] > 0


def test_predict_fixtures_excludes_same_day_matches(synthetic_league: pd.DataFrame) -> None:
    """A match played on the fixture's own date must not enter its training set."""
    earlier = synthetic_league[synthetic_league["match_date"] <= pd.Timestamp("2021-01-29")]
    same_day = pd.DataFrame(
        {
            "match_id": ["same-day"],
            "match_date": [pd.Timestamp("2021-02-05")],
            "season": ["2020/21"],
            "home_team": ["T02"],
            "away_team": ["T03"],
            "home_goals": [4],
            "away_goals": [0],
            "result": ["H"],
        }
    )
    history = pd.concat([earlier, same_day], ignore_index=True)
    fixture = _fixture("T00", "T01", "2021-02-05")

    result = predict_fixtures(history, fixture, min_train_matches=15)

    assert result.iloc[0]["n_train_matches"] == len(earlier)


def test_predict_fixtures_skips_without_enough_history(synthetic_league: pd.DataFrame) -> None:
    """Insufficient history yields a flagged non-forecast, not a confident guess."""
    fixture = _fixture("T00", "T01", "2021-02-05")

    result = predict_fixtures(synthetic_league, fixture, min_train_matches=1000)

    expected_prior = int((synthetic_league["match_date"] < pd.Timestamp("2021-02-05")).sum())
    row = result.iloc[0]
    assert bool(row["is_forecast"]) is False
    assert row["n_train_matches"] == expected_prior
    assert row[list(PROBABILITY_COLUMNS)].isna().all()
    assert "needs 1000 prior matches" in row["note"]


def test_predict_fixtures_flags_unknown_teams(synthetic_league: pd.DataFrame) -> None:
    """A club the training window has never seen is counted and named."""
    history = synthetic_league[synthetic_league["match_date"] <= pd.Timestamp("2021-01-29")]
    fixture = _fixture("T00", "Brand New FC", "2021-02-05")

    result = predict_fixtures(history, fixture, min_train_matches=15)

    row = result.iloc[0]
    assert bool(row["is_forecast"]) is True
    assert row["unknown_teams"] == 1
    assert "Brand New FC" in row["note"]


def test_predict_fixtures_reports_rho_for_a_rate_model(synthetic_league: pd.DataFrame) -> None:
    """Dixon-Coles reports its low-score dependence parameter for the scoreline chart."""
    history = synthetic_league[synthetic_league["match_date"] <= pd.Timestamp("2021-01-29")]
    fixture = _fixture("T00", "T01", "2021-02-05")

    result = predict_fixtures(history, fixture, min_train_matches=15)

    rho = result.iloc[0]["model_rho"]
    assert np.isfinite(rho)
    assert -0.2 <= rho <= 0.2


class _NoRhoModel:
    """A predictive model that predicts outcomes but has no scoreline parameters."""

    converged = True

    def fit(self, matches: pd.DataFrame) -> _NoRhoModel:
        """Pretend to fit, so the fixture is treated as forecastable."""
        self.n_matches = len(matches)
        return self

    def outcome_probabilities(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Return a fixed, honest-looking distribution without expected goals."""
        return pd.DataFrame(
            {
                "prob_home": [0.4] * len(matches),
                "prob_draw": [0.3] * len(matches),
                "prob_away": [0.3] * len(matches),
            }
        )


def test_predict_fixtures_leaves_rho_null_without_a_rate_model(synthetic_league: pd.DataFrame) -> None:
    """A classifier that exposes no expected goals must report null rho, not a guess."""
    history = synthetic_league[synthetic_league["match_date"] <= pd.Timestamp("2021-01-29")]
    fixture = _fixture("T00", "T01", "2021-02-05")

    result = predict_fixtures(history, fixture, min_train_matches=15, model_factory=_NoRhoModel)

    row = result.iloc[0]
    assert bool(row["is_forecast"]) is True
    assert pd.isna(row["model_rho"])


def test_predict_fixtures_accepts_string_dates(synthetic_league: pd.DataFrame) -> None:
    """Dates arriving from SQLite as ISO strings are coerced, not compared as text."""
    history = synthetic_league[synthetic_league["match_date"] <= pd.Timestamp("2021-01-29")].copy()
    history["match_date"] = history["match_date"].dt.strftime("%Y-%m-%d")
    fixture = _fixture("T00", "T01", "2021-02-05")

    result = predict_fixtures(history, fixture, min_train_matches=15)

    assert bool(result.iloc[0]["is_forecast"]) is True
    assert result.iloc[0]["n_train_matches"] == len(history)


@pytest.mark.parametrize(
    ("history", "fixtures", "message"),
    [
        (pd.DataFrame(), pd.DataFrame({"match_date": ["2021-01-01"], "home_team": ["A"], "away_team": ["B"]}), "history is empty"),
        (pd.DataFrame({"match_date": ["2021-01-01"], "home_team": ["A"], "away_team": ["B"], "home_goals": [1], "away_goals": [0]}), pd.DataFrame(), "no fixtures"),
    ],
)
def test_predict_fixtures_rejects_empty_inputs(history: pd.DataFrame, fixtures: pd.DataFrame, message: str) -> None:
    """Empty history or fixtures is an error, not an empty result."""
    with pytest.raises(InferenceError, match=message):
        predict_fixtures(history, fixtures, min_train_matches=1)


def test_predict_fixtures_rejects_missing_columns() -> None:
    """A frame missing a required column fails loudly."""
    history = pd.DataFrame({"match_date": ["2021-01-01"], "home_team": ["A"], "away_team": ["B"]})
    fixture = _fixture("A", "B", "2021-01-08")

    with pytest.raises(InferenceError, match="missing"):
        predict_fixtures(history, fixture, min_train_matches=1)


def test_predict_fixtures_rejects_unparseable_dates() -> None:
    """A corrupt date aborts rather than being silently dropped."""
    history = pd.DataFrame(
        {
            "match_date": ["not-a-date"],
            "home_team": ["A"],
            "away_team": ["B"],
            "home_goals": [1],
            "away_goals": [0],
        }
    )
    fixture = _fixture("A", "B", "2021-01-08")

    with pytest.raises(InferenceError, match="match date"):
        predict_fixtures(history, fixture, min_train_matches=1)


def test_model_options_build_is_independent() -> None:
    """Each call returns a fresh, unfitted model so folds do not share state."""
    options = ModelOptions(ridge=0.1, half_life_matches=120)

    first = options.build()
    second = options.build()

    assert first is not second
    assert first.ridge == 0.1
    assert first.half_life_matches == 120
    assert first.n_matches == 0


def test_forecast_league_forecasts_only_unplayed_fixtures() -> None:
    """Only fixtures lacking a result are forecast, with the stored history used."""
    frame = pd.DataFrame(
        {
            "match_id": ["m1", "m2", "m3", "m4"],
            "match_date": pd.to_datetime(["2021-01-01", "2021-01-08", "2021-01-15", "2021-01-22"]),
            "home_team": ["A", "B", "A", "B"],
            "away_team": ["B", "A", "B", "A"],
            "home_goals": [1, 2, 0, None],
            "away_goals": [0, 2, 1, None],
            "result": ["H", "D", "A", None],
        }
    )
    repository = _StubRepository(frame)

    result = forecast_league(repository, "ENG_PL", min_train_matches=3)

    assert len(result) == 1
    assert result.iloc[0]["match_id"] == "m4"
    assert bool(result.iloc[0]["is_forecast"]) is True


def test_forecast_league_errors_when_nothing_pending() -> None:
    """A league whose matches are all complete has nothing to forecast."""
    frame = pd.DataFrame(
        {
            "match_id": ["m1"],
            "match_date": pd.to_datetime(["2021-01-01"]),
            "home_team": ["A"],
            "away_team": ["B"],
            "home_goals": [1],
            "away_goals": [0],
            "result": ["H"],
        }
    )

    with pytest.raises(InferenceError, match="already has a result"):
        forecast_league(_StubRepository(frame), "ENG_PL", min_train_matches=1)


def test_evaluate_league_scores_model_and_base_rate() -> None:
    """Without stored odds the evaluation still reports model and base rate."""
    synthetic = _synthetic_played()
    evaluation = evaluate_league(_StubRepository(synthetic), "SYN", min_train_matches=4, refit_every_dates=3)

    labels = [report.label for report in evaluation.reports]
    assert labels == ["Dixon-Coles", "Base rate"]
    assert evaluation.has_market is False
    for report in evaluation.reports:
        assert report.n_matches == evaluation.n_predictions == len(evaluation.predictions)


def test_evaluate_league_rescores_everything_on_market_subset() -> None:
    """When the market is present all reports share the market's matches."""
    synthetic = _synthetic_played()
    odds = synthetic.copy()
    odds["match_date"] = pd.to_datetime(odds["match_date"])
    odds["odds_home_close"] = 2.10
    odds["odds_draw_close"] = 3.30
    odds["odds_away_close"] = 3.60

    evaluation = evaluate_league(
        _StubRepository(synthetic, odds=odds),
        "SYN",
        min_train_matches=4,
        refit_every_dates=3,
    )

    labels = [report.label for report in evaluation.reports]
    assert labels == ["Dixon-Coles", "Base rate", "Closing market"]
    assert evaluation.has_market is True
    assert evaluation.market_matches > 0
    for report in evaluation.reports:
        assert report.n_matches == evaluation.market_matches


def test_evaluate_league_errors_when_warm_up_too_large() -> None:
    """An impossible warm-up is reported, not swallowed into an empty result."""
    synthetic = _synthetic_played()

    with pytest.raises(InferenceError, match="warm-up"):
        evaluate_league(_StubRepository(synthetic), "SYN", min_train_matches=10_000)


def _synthetic_played() -> pd.DataFrame:
    """Build a small deterministic completed league for evaluation tests.

    Returns:
        Frame of matches with results and goals, two clubs over eight matchdays.
    """
    rows: list[dict[str, object]] = []
    for week in range(8):
        winner = 2 if week % 2 == 0 else 1
        loser = 0 if week % 2 == 0 else 1
        rows.append(
            {
                "match_id": f"m{week:02d}",
                "match_date": pd.Timestamp("2021-01-01") + pd.Timedelta(days=7 * week),
                "season": "2020/21",
                "home_team": "A" if week % 2 == 0 else "B",
                "away_team": "B" if week % 2 == 0 else "A",
                "home_goals": winner,
                "away_goals": loser,
                "result": "H" if winner > loser else "D",
            }
        )
    return pd.DataFrame(rows)

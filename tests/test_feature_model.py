"""Tests for the gradient-boosted engineered-feature model.

A boosted classifier is easy to make look good and hard to make honest, so these
tests do not ask whether its numbers are plausible. They ask the questions that
matter: is every forecast a distribution, does it ignore the fixture it is
predicting, and does it slot into the leakage-safe inference path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predict_football.inference import predict_fixtures
from predict_football.models.dixon_coles import PROBABILITY_COLUMNS
from predict_football.models.feature_model import FeatureModel, FeatureModelOptions


def _synthetic_league(n_teams: int = 8, rounds: int = 30, seed: int = 20260810) -> pd.DataFrame:
    """Build a synthetic league with a known quality ordering and home edge.

    Args:
        n_teams: Number of clubs.
        rounds: Number of matchdays.
        seed: Random seed, fixed so the tests never flake.

    Returns:
        Match frame where higher-indexed clubs are stronger and the home side
        scores more, so a fitted model has a real ordering to recover.
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
                    "season": "2020/21",
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
    """Return a synthetic league ready to fit."""
    return _synthetic_league()


@pytest.fixture
def fitted(league: pd.DataFrame) -> FeatureModel:
    """Return a model fitted to the synthetic league."""
    return FeatureModel().fit(league)


def test_options_build_returns_a_fresh_unfitted_model() -> None:
    """Each factory call must return its own model, so folds share no state."""
    first = FeatureModelOptions(n_estimators=10).build()
    second = FeatureModelOptions(n_estimators=10).build()

    assert isinstance(first, FeatureModel)
    assert first.converged is False
    assert first.n_matches == 0
    assert first.options.n_estimators == 10
    assert first is not second


def test_fit_records_the_played_matches(fitted: FeatureModel, league: pd.DataFrame) -> None:
    """The training size must reflect the rows actually used."""
    assert fitted.converged is True
    assert fitted.n_matches == len(league)


def test_probabilities_are_a_distribution(fitted: FeatureModel, league: pd.DataFrame) -> None:
    """Every forecast must be a genuine probability distribution."""
    probabilities = fitted.outcome_probabilities(league.head(8))

    values = probabilities[list(PROBABILITY_COLUMNS)].to_numpy(dtype="float64")
    np.testing.assert_allclose(values.sum(axis=1), 1.0, atol=1e-9)
    assert (values >= 0).all()
    assert (values <= 1).all()


def test_a_stronger_home_club_is_favoured(fitted: FeatureModel, league: pd.DataFrame) -> None:
    """T07 is constructed strongest and T00 weakest, so this asserts a direction.

    A sign error in the feature assembly would swap the two orderings and still
    produce valid-looking distributions, so only a directional check catches it.
    A fixture must carry a date, because rest is measured from the previous match.
    """
    when = league["match_date"].max() + pd.Timedelta(days=7)
    forward = fitted.outcome_probabilities(
        pd.DataFrame({"home_team": ["T07"], "away_team": ["T00"], "match_date": [when]})
    )
    reverse = fitted.outcome_probabilities(
        pd.DataFrame({"home_team": ["T00"], "away_team": ["T07"], "match_date": [when]})
    )

    assert forward["prob_home"].iloc[0] > forward["prob_away"].iloc[0]
    assert forward["prob_home"].iloc[0] > reverse["prob_home"].iloc[0]


def test_predicting_ignores_the_fixture_own_result(fitted: FeatureModel, league: pd.DataFrame) -> None:
    """The row being predicted must not feed its own features.

    The adapter builds features from the history it was fitted on, so a result
    attached to the prediction frame is inert. Rewriting it must change nothing.
    """
    target = league.iloc[[10]].copy()
    baseline = fitted.outcome_probabilities(target)
    flipped = fitted.outcome_probabilities(target.assign(result="A", home_goals=0, away_goals=9))

    np.testing.assert_allclose(
        baseline[list(PROBABILITY_COLUMNS)].to_numpy(dtype="float64"),
        flipped[list(PROBABILITY_COLUMNS)].to_numpy(dtype="float64"),
    )


def test_predictions_carry_the_fixture_identifiers(fitted: FeatureModel, league: pd.DataFrame) -> None:
    """A forecast nobody can attribute to a fixture is not usable."""
    sample = league.head(3)
    probabilities = fitted.outcome_probabilities(sample)

    for column in ("match_id", "match_date", "home_team", "away_team"):
        assert column in probabilities.columns
    assert probabilities["match_id"].tolist() == sample["match_id"].tolist()


def test_forecasts_are_reproducible(league: pd.DataFrame) -> None:
    """A fixed seed with one job must give the same forecast twice."""
    first = FeatureModel().fit(league).outcome_probabilities(league.head(5))
    second = FeatureModel().fit(league).outcome_probabilities(league.head(5))

    np.testing.assert_allclose(
        first[list(PROBABILITY_COLUMNS)].to_numpy(dtype="float64"),
        second[list(PROBABILITY_COLUMNS)].to_numpy(dtype="float64"),
    )


def test_predict_before_fitting_is_an_error(league: pd.DataFrame) -> None:
    """An unfitted model must say so rather than return zeros."""
    with pytest.raises(ValueError, match="not been fitted"):
        FeatureModel().outcome_probabilities(league.head(1))


def test_fit_rejects_a_frame_missing_columns(league: pd.DataFrame) -> None:
    """A missing column must be named, not surface later as a KeyError."""
    with pytest.raises(ValueError, match="home_goals"):
        FeatureModel().fit(league.drop(columns=["home_goals"]))


def test_fit_rejects_a_frame_with_no_results(league: pd.DataFrame) -> None:
    """Training on fixtures with no outcome is not a valid model."""
    with pytest.raises(ValueError, match="no row has a result"):
        FeatureModel().fit(league.assign(result=pd.NA))


def test_fit_rejects_an_unknown_outcome_label(league: pd.DataFrame) -> None:
    """An outcome outside H/D/A must be refused, not silently encoded."""
    broken = league.copy()
    broken.loc[broken.index[0], "result"] = "X"

    with pytest.raises(ValueError, match="unrecognised"):
        FeatureModel().fit(broken)


def test_predict_fixtures_accepts_a_classifier_factory(league: pd.DataFrame) -> None:
    """The inference path must work for a model that emits no expected goals.

    Dixon-Coles reports ``lambda_*``; this classifier does not. The adapter must
    leave those columns null rather than crash, and still produce distributions.
    """
    ordered = league.sort_values(["match_date", "match_id"])
    cutoff = ordered.iloc[80]["match_date"]
    history = league[league["match_date"] < cutoff]
    fixtures = league[league["match_date"] >= cutoff].drop(columns=["result", "home_goals", "away_goals"])

    forecast = predict_fixtures(
        history,
        fixtures,
        min_train_matches=60,
        model_factory=FeatureModelOptions(n_estimators=30).build,
    )
    scored = forecast[forecast["is_forecast"]]

    assert len(scored) > 0
    values = scored[list(PROBABILITY_COLUMNS)].to_numpy(dtype="float64")
    assert np.isfinite(values).all()
    np.testing.assert_allclose(values.sum(axis=1), 1.0, atol=1e-9)
    assert scored["model_converged"].all()
    assert scored["lambda_home"].isna().all()
    assert scored["lambda_away"].isna().all()

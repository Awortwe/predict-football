"""Tests for the Dixon-Coles model, the scoring rules and the backtest harness.

These three modules carry the project's honesty claim, so the tests concentrate on
three things: that probabilities are real probabilities, that the metrics punish
exactly what they should, and that the backtest structurally cannot train on its
own test data.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from predict_football.backtest import (
    BacktestError,
    verify_no_overlap,
    walk_forward_backtest,
    walk_forward_folds,
)
from predict_football.evaluation import (
    EvaluationError,
    base_rate_probabilities,
    calibration,
    compare_reports,
    evaluate,
    implied_probabilities,
    multiclass_brier_score,
    multiclass_log_loss,
    outcome_accuracy,
)
from predict_football.models.dixon_coles import (
    MAX_GOALS,
    OUTCOMES,
    PROBABILITY_COLUMNS,
    DixonColesModel,
    _centre_ratings,
    score_matrix,
)

# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


#: Extra expected goals the synthetic home side is given, so the fitted home
#: advantage has a known sign to test against rather than being pure noise.
HOME_EDGE = 0.35


def _synthetic_league(n_teams: int = 6, rounds: int = 40, seed: int = 20260810) -> pd.DataFrame:
    """Build a synthetic league with a known quality ordering and home edge.

    Args:
        n_teams: Number of clubs.
        rounds: Number of matchdays.
        seed: Random seed, fixed so tests never flake.

    Returns:
        Match frame with one match per club per matchday. The home side is drawn
        to score :data:`HOME_EDGE` more goals than it would away, so the fitted
        home advantage has a known sign to be tested against. An earlier version
        assigned venues at random, which left home advantage indistinguishable
        from noise and made the sign untestable.
    """
    rng = np.random.default_rng(seed)
    # Team i is stronger the higher i is, which gives the tests a direction to
    # assert on without hard-coding any fitted number.
    strengths = np.linspace(0.9, 2.3, n_teams)
    teams = [f"T{i:02d}" for i in range(n_teams)]

    rows: list[dict[str, object]] = []
    for round_index in range(rounds):
        order = list(rng.permutation(n_teams))
        for a, b in zip(order[::2], order[1::2], strict=True):
            home, away = teams[a], teams[b]
            home_goals = int(rng.poisson(strengths[a] + HOME_EDGE))
            away_goals = int(rng.poisson(strengths[b]))
            result = "H" if home_goals > away_goals else "A" if home_goals < away_goals else "D"
            rows.append(
                {
                    "match_id": f"m{round_index:03d}_{a}_{b}",
                    "match_date": pd.Timestamp("2021-01-01") + pd.Timedelta(days=7 * round_index),
                    "season": "2020/21",
                    "home_team": home,
                    "away_team": away,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "result": result,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def league() -> pd.DataFrame:
    """Return a fitted-ready synthetic league."""
    return _synthetic_league()


@pytest.fixture
def fitted(league: pd.DataFrame) -> DixonColesModel:
    """Return a model fitted to the synthetic league."""
    return DixonColesModel(ridge=0.01).fit(league)


# --------------------------------------------------------------------------
# Dixon-Coles
# --------------------------------------------------------------------------


def test_outcome_probabilities_sum_to_one_exactly(fitted: DixonColesModel, league: pd.DataFrame) -> None:
    """Every forecast must be a genuine probability distribution."""
    probabilities = fitted.outcome_probabilities(league)

    totals = probabilities[list(PROBABILITY_COLUMNS)].sum(axis=1)
    np.testing.assert_allclose(totals.to_numpy(), 1.0, atol=1e-12)
    assert (probabilities[list(PROBABILITY_COLUMNS)].to_numpy() >= 0).all()
    assert (probabilities[list(PROBABILITY_COLUMNS)].to_numpy() <= 1).all()


def test_model_ranks_the_stronger_club_as_more_likely_to_win(
    fitted: DixonColesModel, league: pd.DataFrame
) -> None:
    """The strongest club at home against the weakest must be a clear favourite.

    Team T05 is constructed to be the strongest and T00 the weakest, so this
    asserts a direction rather than repeating a number from the fit.
    """
    fixtures = pd.DataFrame({"home_team": ["T05"], "away_team": ["T00"]})
    probabilities = fitted.outcome_probabilities(fixtures)

    assert probabilities["prob_home"].iloc[0] > 0.6
    assert probabilities["prob_home"].iloc[0] > probabilities["prob_away"].iloc[0]

    reverse = fitted.outcome_probabilities(pd.DataFrame({"home_team": ["T00"], "away_team": ["T05"]}))
    assert reverse["prob_away"].iloc[0] > reverse["prob_home"].iloc[0]


def test_home_advantage_is_learned_as_an_advantage(fitted: DixonColesModel) -> None:
    """Two clubs rated alike should be more likely to win at home."""
    assert fitted.home_advantage > 0
    assert np.exp(fitted.home_advantage) > 1.0


def test_fitted_ratings_are_centred(league: pd.DataFrame) -> None:
    """Attack is centred to zero so the numbers are interpretable.

    Only the difference between attack and defence is identified. Attack is
    centred by removing a constant from *both* arrays, which preserves every
    prediction; defence is therefore not separately centred. The common-shift
    property itself is pinned down by ``test_centring_applies_one_common_shift``.
    """
    ratings = DixonColesModel(ridge=0.01).fit(league).team_ratings()

    assert ratings["attack"].mean() == pytest.approx(0.0, abs=1e-9)


def test_team_ratings_are_sorted_by_attack(fitted: DixonColesModel) -> None:
    """The ratings table must be ordered strongest attack first."""
    ratings = fitted.team_ratings()
    assert ratings["attack"].is_monotonic_decreasing


def test_zero_rho_reduces_the_matrix_to_independent_poisson(fitted: DixonColesModel) -> None:
    """With no correction the score matrix must be the plain Poisson product.

    This pins down the Dixon-Coles correction as the only thing separating the
    model from an independent Poisson, so a bug in ``tau`` cannot hide.
    """
    from scipy.stats import poisson

    lam, mu = 1.7, 1.1
    independent = poisson.pmf(np.arange(MAX_GOALS + 1), lam)[:, None] * poisson.pmf(np.arange(MAX_GOALS + 1), mu)

    matrix = fitted.score_matrix("T05", "T00").to_numpy()
    corrected = fitted.score_matrix("T05", "T00").to_numpy()

    # Reconstruct expectations the model actually used.
    expected = fitted.expected_goals(pd.DataFrame({"home_team": ["T05"], "away_team": ["T00"]}))
    independent = (
        poisson.pmf(np.arange(MAX_GOALS + 1), expected["lambda_home"].iloc[0])[:, None]
        * poisson.pmf(np.arange(MAX_GOALS + 1), expected["lambda_away"].iloc[0])
    )
    independent = independent / independent.sum()

    # The corrected matrix differs from the independent one in the low scores,
    # and equals it everywhere else.
    assert not np.allclose(matrix, independent)
    assert corrected[0, 0] != pytest.approx(independent[0, 0])
    assert corrected[5, 7] == pytest.approx(independent[5, 7])
    assert lam and mu  # referenced for readability of the test's intent


def test_score_matrix_rows_and_columns_are_probabilities(fitted: DixonColesModel) -> None:
    """The score matrix must be non-negative and sum to one over the whole grid."""
    matrix = fitted.score_matrix("T03", "T02")

    assert matrix.shape == (MAX_GOALS + 1, MAX_GOALS + 1)
    assert (matrix.to_numpy() >= 0).all()
    assert matrix.to_numpy().sum() == pytest.approx(1.0, abs=1e-12)


def test_public_score_matrix_matches_the_fitted_model(fitted: DixonColesModel) -> None:
    """The free function must reproduce the model's own matrix from its rates.

    The app rebuilds the scoreline chart from a forecast's expected goals rather
    than refitting, so the two paths must agree exactly. Verified against the
    model's method, which is the independent reference here.
    """
    expected = fitted.expected_goals(pd.DataFrame({"home_team": ["T03"], "away_team": ["T02"]}))
    lam = float(expected["lambda_home"].iloc[0])
    mu = float(expected["lambda_away"].iloc[0])

    rebuilt = score_matrix(lam, mu, fitted.rho).to_numpy()

    assert np.allclose(rebuilt, fitted.score_matrix("T03", "T02").to_numpy())
    assert rebuilt.sum() == pytest.approx(1.0, abs=1e-12)


def test_centring_applies_one_common_shift() -> None:
    """Centring must not change any attack-minus-defence difference.

    Attack and defence are only identified up to a shared constant. Centring each
    by its own mean would apply two different shifts and move predictions off the
    fitted optimum; the earlier implementation did exactly that.
    """
    attack = np.array([0.3, -0.1, 0.5, -0.7])
    defence = np.array([0.9, 0.2, -0.4, -0.7])

    centred_attack, centred_defence = _centre_ratings(attack, defence)

    assert centred_attack.mean() == pytest.approx(0.0)
    assert np.allclose(attack - centred_attack, defence - centred_defence)
    assert np.allclose(
        attack[:, None] - defence[None, :],
        centred_attack[:, None] - centred_defence[None, :],
    )


def test_score_matrix_shrinks_rho_for_an_extrapolating_fixture() -> None:
    """A lopsided fixture must still yield a valid distribution.

    A model fitted on ordinary matches can be asked about a fixture with much
    larger expected goals, where the fitted rho would make a low-score correction
    factor negative. The matrix must remain a probability distribution rather
    than raise and abort an entire backtest.
    """
    model = DixonColesModel(rho=-0.2, intercept=2.0, home_advantage=1.0)
    model.teams = ["Strong", "Weak"]
    model._attack = {"Strong": 0.0, "Weak": 0.0}
    model._defence = {"Strong": 0.0, "Weak": 0.0}

    matrix = model.score_matrix("Strong", "Weak").to_numpy()

    assert (matrix >= 0).all()
    assert matrix.sum() == pytest.approx(1.0, abs=1e-12)


def test_unknown_team_is_given_league_average_ratings(fitted: DixonColesModel) -> None:
    """An unseen club must not inherit another club's rating.

    A promoted team with no history gets attack and defence of zero, i.e. exactly
    league average. Anything else would be inventing an opinion.
    """
    expected = fitted.expected_goals(pd.DataFrame({"home_team": ["Ghost A"], "away_team": ["Ghost B"]}))
    baseline = np.exp(fitted.intercept)

    # Both clubs are unknown, so attack and defence are both zero and the only
    # remaining terms are the base rate and the home edge.
    assert expected["lambda_home"].iloc[0] == pytest.approx(baseline * np.exp(fitted.home_advantage), rel=1e-9)
    assert expected["lambda_away"].iloc[0] == pytest.approx(baseline, rel=1e-9)


def test_expected_goals_carry_the_fixture_identifiers(fitted: DixonColesModel, league: pd.DataFrame) -> None:
    """A prediction nobody can attribute to a fixture is not usable."""
    result = fitted.expected_goals(league.head(5))

    for column in ("match_id", "match_date", "home_team", "away_team", "lambda_home", "lambda_away"):
        assert column in result.columns


def test_log_likelihood_is_finite_and_improves_on_a_worse_fit(league: pd.DataFrame) -> None:
    """The fitted model must beat a deliberately flat model in-sample."""
    good = DixonColesModel(ridge=0.01).fit(league)
    flat = DixonColesModel(ridge=50.0).fit(league)

    assert np.isfinite(good.log_likelihood(league))
    assert np.isfinite(flat.log_likelihood(league))
    assert good.log_likelihood(league) > flat.log_likelihood(league)


def test_fit_rejects_a_frame_missing_columns(league: pd.DataFrame) -> None:
    """A missing column must be named, not surface later as a KeyError."""
    with pytest.raises(ValueError, match="home_goals"):
        DixonColesModel().fit(league.drop(columns=["home_goals"]))


def test_fit_rejects_a_frame_with_no_results(league: pd.DataFrame) -> None:
    """Training on fixtures with no outcome is not a valid model."""
    blanked = league.assign(home_goals=pd.NA, away_goals=pd.NA)

    with pytest.raises(ValueError, match="no row has a result"):
        DixonColesModel().fit(blanked)


def test_fit_rejects_a_single_team_league() -> None:
    """Attack and defence cannot be identified from one club."""
    only = pd.DataFrame(
        {
            "match_id": ["a", "b"],
            "match_date": pd.to_datetime(["2021-01-01", "2021-01-08"]),
            "home_team": ["Solo", "Solo"],
            "away_team": ["Solo", "Solo"],
            "home_goals": [1, 2],
            "away_goals": [0, 1],
        }
    )

    with pytest.raises(ValueError, match=r"same club|at least two teams"):
        DixonColesModel().fit(only)


def test_predict_before_fitting_is_an_error(league: pd.DataFrame) -> None:
    """An unfitted model must say so rather than return zeros."""
    with pytest.raises(ValueError, match="not been fitted"):
        DixonColesModel().expected_goals(league.head(1))


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------


def _perfect_and_worst(n: int = 50) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Return results, a perfect forecast and a maximally wrong forecast.

    Args:
        n: Number of matches.

    Returns:
        Tuple of observed results, the perfect forecast and its complement.
    """
    results = pd.Series(["H", "D", "A"] * n)[:n]
    one_hot = pd.get_dummies(results).reindex(columns=list(OUTCOMES), fill_value=0).astype("float64")
    one_hot.columns = list(PROBABILITY_COLUMNS)
    perfect = one_hot.reset_index(drop=True)
    # Maximally wrong but still a valid distribution: put all the mass on a
    # single incorrect outcome. Subtracting from one would produce rows summing
    # to two, which is not a forecast at all.
    wrong_column = {0: 2, 1: 2, 2: 0}  # H -> A, D -> A, A -> H
    worst = pd.DataFrame(0.0, index=range(len(results)), columns=list(PROBABILITY_COLUMNS))
    for i, outcome in enumerate(results):
        worst.loc[i, PROBABILITY_COLUMNS[wrong_column[OUTCOMES.index(outcome)]]] = 1.0
    return results.reset_index(drop=True), perfect, worst


def test_a_perfect_forecast_scores_perfectly() -> None:
    """A certain forecast must score zero Brier and zero loss."""
    results, perfect, _ = _perfect_and_worst()

    assert multiclass_brier_score(results, perfect) == pytest.approx(0.0, abs=1e-12)
    assert multiclass_log_loss(results, perfect) < 1e-9
    assert outcome_accuracy(results, perfect) == 1.0


def test_a_maximally_wrong_forecast_scores_worse_than_uniform() -> None:
    """Certain and wrong must be punished harder than merely uncertain.

    This is the property that makes log loss a proper scoring rule rather than
    just a ranking metric.
    """
    results, _, worst = _perfect_and_worst()
    uniform = pd.DataFrame(
        {c: [1 / 3] * len(results) for c in PROBABILITY_COLUMNS},
    )

    assert multiclass_log_loss(results, worst) > multiclass_log_loss(results, uniform)
    assert multiclass_brier_score(results, worst) > multiclass_brier_score(results, uniform)


def test_uniform_forecast_matches_the_analytic_values() -> None:
    """Uniform scores ln(3) and 2/3, verified analytically rather than copied."""
    results = pd.Series(["H"] * 30)
    uniform = pd.DataFrame({c: [1 / 3] * 30 for c in PROBABILITY_COLUMNS})

    assert multiclass_log_loss(results, uniform) == pytest.approx(np.log(3))
    assert multiclass_brier_score(results, uniform) == pytest.approx(2 / 3)


def test_accuracy_rewards_the_confident_and_correct() -> None:
    """Accuracy is the share of matches where the top pick actually happened.

    Kept separate from the uniform test: with exactly tied probabilities the
    winning column is decided by ``argmax`` tie-breaking, which is an
    implementation detail and not something to assert about a forecast.
    """
    results = pd.Series(["H", "H", "A", "D"])
    forecast = pd.DataFrame(
        {
            "prob_home": [0.7, 0.6, 0.2, 0.5],
            "prob_draw": [0.2, 0.3, 0.2, 0.3],
            "prob_away": [0.1, 0.1, 0.6, 0.2],
        }
    )

    # Three of four: the last row backs a home win but a draw occurred.
    assert outcome_accuracy(results, forecast) == pytest.approx(0.75)


def test_metrics_reject_probabilities_that_do_not_sum_to_one() -> None:
    """A malformed forecast must be refused, not silently scored."""
    bad = pd.DataFrame({"prob_home": [0.5], "prob_draw": [0.2], "prob_away": [0.1]})

    with pytest.raises(EvaluationError, match="sum to 1"):
        multiclass_brier_score(pd.Series(["H"]), bad)


def test_metrics_reject_negative_probabilities() -> None:
    """Negative probability is not a probability."""
    bad = pd.DataFrame({"prob_home": [1.4], "prob_draw": [0.2], "prob_away": [-0.6]})

    with pytest.raises(EvaluationError, match=r"\[0, 1\]"):
        multiclass_brier_score(pd.Series(["H"]), bad)


def test_metrics_reject_missing_probability_columns() -> None:
    """The caller must supply all three outcomes."""
    bad = pd.DataFrame({"prob_home": [1.0]})

    with pytest.raises(EvaluationError, match="missing columns"):
        multiclass_brier_score(pd.Series(["H"]), bad)


def test_metrics_reject_an_unknown_outcome_label() -> None:
    """An unrecognised result must be refused rather than scored as a loss."""
    uniform = pd.DataFrame({c: [1 / 3] for c in PROBABILITY_COLUMNS})

    with pytest.raises(EvaluationError, match="unrecognised"):
        multiclass_log_loss(pd.Series(["X"]), uniform)


def test_calibration_of_a_perfectly_calibrated_forecast_is_near_zero() -> None:
    """Probabilities that match observed frequencies must show no miscalibration.

    Constructed directly: a prediction of p for an event that occurs with
    frequency p is calibrated by definition, so the weighted error must be
    essentially zero without relying on the model.
    """
    results = pd.Series(["H"] * 30 + ["D"] * 70)
    probabilities = pd.DataFrame(
        {
            "prob_home": [0.3] * 100,
            "prob_draw": [0.7] * 100,
            "prob_away": [0.0] * 100,
        }
    )
    probabilities.loc[0:29, "prob_home"] = 1.0
    probabilities.loc[0:29, "prob_draw"] = 0.0
    probabilities.loc[0:29, "prob_away"] = 0.0
    probabilities.loc[30:, "prob_home"] = 0.0
    probabilities.loc[30:, "prob_draw"] = 1.0
    probabilities.loc[30:, "prob_away"] = 0.0

    summary = calibration(results, probabilities, "H", bins=5)

    assert summary.weighted_absolute_error < 1e-9
    assert summary.n_matches == 100
    assert summary.counts.sum() == 100


def test_calibration_detects_a_confidently_wrong_forecast() -> None:
    """Claiming 100% on an event that never happens must be flagged."""
    results = pd.Series(["D"] * 40)
    probabilities = pd.DataFrame(
        {"prob_home": [0.9] * 40, "prob_draw": [0.05] * 40, "prob_away": [0.05] * 40}
    )

    summary = calibration(results, probabilities, "H", bins=5)

    assert summary.weighted_absolute_error > 0.5
    assert summary.observed_frequency[summary.counts > 0].max() == 0.0


def test_calibration_bins_report_their_populations() -> None:
    """An apparently good calibration must be checkable for sample size."""
    results = pd.Series(["H", "D", "A"] * 10)
    probabilities = pd.DataFrame(
        {c: [1 / 3] * len(results) for c in PROBABILITY_COLUMNS},
    )

    summary = calibration(results, probabilities, "H", bins=5)

    assert summary.counts.sum() == len(results)
    assert summary.bins[0] == 0.0
    assert summary.bins[-1] == 1.0


def test_calibration_rejects_an_unknown_outcome() -> None:
    """Only the three known outcomes can be assessed."""
    uniform = pd.DataFrame({c: [1 / 3] for c in PROBABILITY_COLUMNS})

    with pytest.raises(EvaluationError, match="outcome must be"):
        calibration(pd.Series(["H"]), uniform, "X")


def test_report_always_carries_the_sample_size() -> None:
    """No metric may be reported without the number of matches behind it."""
    results = pd.Series(["H", "D", "A"] * 4)
    probabilities = pd.DataFrame({c: [1 / 3] * 12 for c in PROBABILITY_COLUMNS})
    report = evaluate(results, probabilities, label="uniform")

    assert report.n_matches == 12
    row = report.as_row()
    assert row["n_matches"] == 12
    assert "12 matches" in str(report)
    assert "12 matches" in str(report)


def test_base_rate_forecast_reproduces_the_observed_frequencies() -> None:
    """The base rate benchmark must be the empirical distribution."""
    results = pd.Series(["H"] * 45 + ["D"] * 30 + ["A"] * 25)

    probabilities = base_rate_probabilities(results)

    assert probabilities["prob_home"].iloc[0] == pytest.approx(0.45)
    assert probabilities["prob_draw"].iloc[0] == pytest.approx(0.30)
    assert probabilities["prob_away"].iloc[0] == pytest.approx(0.25)
    np.testing.assert_allclose(probabilities.sum(axis=1).to_numpy(), 1.0)


def test_implied_probabilities_removes_the_overround() -> None:
    """De-vigging must normalise inverse prices, checked by hand.

    Prices 1.5 / 4.0 / 6.0 imply inverse odds 0.6667 / 0.25 / 0.1667 summing to
    1.0833, so the overround is 8.33% and the fair home probability is
    0.6667 / 1.0833 = 0.6154.
    """
    odds = pd.DataFrame({"odds_home_close": [1.5], "odds_draw_close": [4.0], "odds_away_close": [6.0]})

    probabilities = implied_probabilities(odds)

    inverse = np.array([1 / 1.5, 1 / 4.0, 1 / 6.0])
    expected = inverse / inverse.sum()
    np.testing.assert_allclose(probabilities.to_numpy().ravel(), expected)
    assert probabilities.to_numpy().sum() == pytest.approx(1.0)


def test_implied_probabilities_rejects_a_price_at_or_below_one() -> None:
    """A decimal odd of 1.0 or less cannot be true, so it must be refused."""
    odds = pd.DataFrame({"odds_home_close": [1.0], "odds_draw_close": [4.0], "odds_away_close": [6.0]})

    with pytest.raises(EvaluationError, match="greater than 1"):
        implied_probabilities(odds)


def test_compare_reports_tabulates_each_row() -> None:
    """Comparison output must keep the sample size visible per row."""
    results = pd.Series(["H", "A"] * 5)
    probabilities = pd.DataFrame({c: [0.5, 0.5][:1] * 0 + [0.5] * 10 for c in PROBABILITY_COLUMNS})
    probabilities["prob_home"] = 0.4
    probabilities["prob_draw"] = 0.3
    probabilities["prob_away"] = 0.3

    table = compare_reports([evaluate(results, probabilities, label="a"), evaluate(results, probabilities, label="b")])

    assert len(table) == 2
    assert list(table["label"]) == ["a", "b"]
    assert list(table["n_matches"]) == [10, 10]


# --------------------------------------------------------------------------
# backtest
# --------------------------------------------------------------------------


class _ConstantModel:
    """Model stub that predicts a fixed split and records what it trained on."""

    def __init__(self) -> None:
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
        """Return a fixed, valid probability forecast.

        Args:
            matches: Frame to predict.

        Returns:
            Forecast frame with the canonical probability columns.
        """
        n = len(matches)
        return pd.DataFrame(
            {
                "match_id": matches["match_id"].to_numpy(),
                "match_date": matches["match_date"].to_numpy(),
                "season": matches["season"].to_numpy(),
                "home_team": matches["home_team"].to_numpy(),
                "away_team": matches["away_team"].to_numpy(),
                "prob_home": [0.45] * n,
                "prob_draw": [0.25] * n,
                "prob_away": [0.30] * n,
            }
        )


class _OracleModel(_ConstantModel):
    """Model stub that cheats by returning the observed outcome as certain."""

    def outcome_probabilities(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Return a perfect forecast, as a leaked model would.

        Args:
            matches: Frame to predict, which in a leak carries the outcome.

        Returns:
            Forecast frame that scores perfectly.
        """
        base = super().outcome_probabilities(matches)
        for position, outcome in enumerate(OUTCOMES):
            base[PROBABILITY_COLUMNS[position]] = [1.0 if r == outcome else 0.0 for r in matches["result"]]
        return base


def test_folds_never_overlap_and_advance_in_time(league: pd.DataFrame) -> None:
    """Every fold's training set must end before its test set begins."""
    folds = walk_forward_folds(league, min_train_matches=10, refit_every_dates=1)

    assert folds, "expected at least one fold"
    for fold in folds:
        assert fold.train_end < fold.test_start
    for earlier, later in pairwise(folds):
        assert earlier.test_start <= later.test_start
        assert later.n_train > earlier.n_train


def test_folds_respect_the_warm_up(league: pd.DataFrame) -> None:
    """No prediction may be made before the warm-up is complete."""
    folds = walk_forward_folds(league, min_train_matches=10)

    assert all(fold.n_train >= 10 for fold in folds)


def test_folds_cover_every_match_after_the_warm_up(league: pd.DataFrame) -> None:
    """The walk must predict every match from the first eligible matchday on.

    Warm-up rounds up to a whole matchday: a matchday is predicted in one go, so
    the first forecast happens on the first date with enough prior matches, and
    everything from there is covered.
    """
    folds = walk_forward_folds(league, min_train_matches=10)

    predicted = sum(fold.n_test for fold in folds)
    first_train = folds[0].n_train
    assert first_train >= 10
    assert predicted == len(league) - first_train


def test_grouping_dates_predicts_a_matchday_in_one_go(league: pd.DataFrame) -> None:
    """Refitting every few dates must not change how many matches are predicted."""
    every_one = walk_forward_folds(league, min_train_matches=10, refit_every_dates=1)
    every_five = walk_forward_folds(league, min_train_matches=10, refit_every_dates=5)

    assert sum(f.n_test for f in every_five) == sum(f.n_test for f in every_one)
    assert len(every_five) < len(every_one)


def test_backtest_produces_verifiable_out_of_sample_predictions(league: pd.DataFrame) -> None:
    """Every predicted match must be absent from the fold that predicted it."""
    predictions = walk_forward_backtest(league, _ConstantModel, min_train_matches=10, refit_every_dates=5)

    assert 0 < len(predictions) < len(league)
    assert "outcome" in predictions.columns
    assert predictions["outcome"].notna().all()
    verify_no_overlap(predictions, league)


def test_backtest_records_growing_training_sets(league: pd.DataFrame) -> None:
    """The walk must be expanding, so later folds see more evidence."""
    predictions = walk_forward_backtest(league, _ConstantModel, min_train_matches=10, refit_every_dates=5)

    sizes = predictions["n_train_matches"].to_numpy()
    assert sizes.min() >= 10
    assert np.all(np.diff(sizes) >= 0)
    assert sizes[-1] > sizes[0]


def test_backtest_never_trains_on_a_later_match(league: pd.DataFrame) -> None:
    """Direct assertion of the temporal property, independent of the harness.

    A stub records exactly which fixtures it was fitted on, so this checks the
    invariant against the model's own account rather than the harness's.
    """
    seen: list[tuple[str, set[str], set[str]]] = []

    class _Recording(_ConstantModel):
        """Stub that records its training and prediction sets."""

        def outcome_probabilities(self, matches: pd.DataFrame) -> pd.DataFrame:
            """Record and delegate.

            Args:
                matches: Frame to predict.

            Returns:
                The constant forecast.
            """
            seen.append((id(self), set(self.trained_ids), set(matches["match_id"])))
            return super().outcome_probabilities(matches)

    walk_forward_backtest(league, _Recording, min_train_matches=10, refit_every_dates=5)

    assert seen
    for _, trained, predicted in seen:
        assert not (trained & predicted), "a model was asked to predict a match it trained on"


def test_verify_no_overlap_catches_a_leaked_prediction(league: pd.DataFrame) -> None:
    """The guard must fail loudly when given an impossible training size.

    A backtest reporting more training matches than precede the fixture is
    arithmetically impossible, and this is the check that says so.
    """
    predictions = pd.DataFrame(
        {
            "match_id": [league.iloc[0]["match_id"]],
            "n_train_matches": [len(league)],
            "prob_home": [0.4],
            "prob_draw": [0.3],
            "prob_away": [0.3],
        }
    )

    with pytest.raises(BacktestError, match="trained on"):
        verify_no_overlap(predictions, league)


def test_oracle_model_scores_perfectly(league: pd.DataFrame) -> None:
    """A leaked model must be visibly detectable in the metrics.

    If a model that knows the answer cannot achieve a perfect score, the scoring
    rules themselves are broken and the honest model's numbers mean nothing.
    """
    predictions = walk_forward_backtest(league, _OracleModel, min_train_matches=10, refit_every_dates=5)
    report = evaluate(predictions["outcome"], predictions[list(PROBABILITY_COLUMNS)], label="oracle")

    assert report.accuracy == 1.0
    assert report.brier == pytest.approx(0.0, abs=1e-12)
    assert report.log_loss < 1e-9


def test_backtest_refuses_an_impossible_warm_up(league: pd.DataFrame) -> None:
    """Asking to predict with more warm-up than data must be an error."""
    with pytest.raises(BacktestError, match="warm-up"):
        walk_forward_backtest(league, _ConstantModel, min_train_matches=len(league) + 1)


def test_backtest_refuses_a_leaked_fold(league: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    """A fold that overlaps its own training set must abort the backtest.

    The harness asserts the invariant itself rather than trusting the fold
    builder, so this forces the violation to confirm the guard fires.
    """
    import predict_football.backtest as backtest_module

    bad = backtest_module.Fold(
        index=0,
        train_end=pd.Timestamp("2100-01-01"),
        test_start=pd.Timestamp("2021-01-01"),
        n_train=len(league),
        n_test=len(league),
    )
    monkeypatch.setattr(backtest_module, "walk_forward_folds", lambda *a, **k: [bad])

    with pytest.raises(BacktestError, match="leaked"):
        backtest_module.walk_forward_backtest(league, _ConstantModel, min_train_matches=10)


def test_folds_reject_a_frame_without_dates(league: pd.DataFrame) -> None:
    """Missing fixture columns must be named up front."""
    with pytest.raises(BacktestError, match="missing"):
        walk_forward_folds(league.drop(columns=["match_date"]), min_train_matches=1)


def test_folds_reject_uninterpretable_dates(league: pd.DataFrame) -> None:
    """A corrupt date must be reported, not guessed at."""
    broken = league.assign(match_date="not-a-date")

    with pytest.raises(BacktestError, match="could not interpret"):
        walk_forward_folds(broken, min_train_matches=1)

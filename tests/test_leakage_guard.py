"""Leakage guard tests.

These are the most important tests in the project. Every other component assumes
that pre-match features cannot contain post-match information; if this guard
regresses, every backtest number downstream becomes fiction without any visible
error.
"""

from __future__ import annotations

import pandas as pd
import pytest

from predict_football.data.schema import (
    ENGINEERED_FEATURE_COLUMNS_BY_NAME,
    MATCH_COLUMNS,
    ODDS_COLUMNS_BY_NAME,
    POST_MATCH_COLUMNS,
    PRE_KICKOFF_COLUMNS,
    PRE_MATCH_COLUMNS,
    TARGET_COLUMNS,
    Availability,
    LeakageError,
    assert_no_leakage,
    feature_plan,
)


def test_every_column_declares_availability() -> None:
    """Every schema column must declare when it becomes knowable."""
    assert len(MATCH_COLUMNS) > 20
    for column in MATCH_COLUMNS:
        assert isinstance(column.availability, Availability), f"{column.name} has no availability"


def test_availability_groups_are_disjoint_and_complete() -> None:
    """The three availability groups must partition the schema with no overlap."""
    groups = [set(PRE_MATCH_COLUMNS), set(TARGET_COLUMNS), set(POST_MATCH_COLUMNS)]
    assert not set.intersection(*groups), "a column appears in more than one availability group"
    union = set().union(*groups)
    assert union == {c.name for c in MATCH_COLUMNS}, "availability groups do not cover the schema"


def test_shots_and_xg_are_blocked_from_pre_match_features() -> None:
    """Shots, xG and half-time scores must never be pre-match features."""
    for column in ("home_shots", "away_shots", "home_xg", "away_xg", "home_goals_ht", "away_goals_ht"):
        assert column in POST_MATCH_COLUMNS, f"{column} is not marked POST_MATCH"
        with pytest.raises(LeakageError, match="final whistle"):
            assert_no_leakage([column])


def test_targets_are_blocked_from_features() -> None:
    """The scoreline and result are labels, not features."""
    for column in ("home_goals", "away_goals", "result"):
        assert column in TARGET_COLUMNS
        with pytest.raises(LeakageError, match="prediction target"):
            assert_no_leakage([column])


def test_bookmaker_closing_odds_are_not_pre_match_information() -> None:
    """Closing prices are set at kickoff, so a days-ahead model cannot use them.

    This is a deliberate correction. The schema previously classified closing
    odds as plain pre-match information on the reasoning that they are published
    before the final whistle. That is true but irrelevant: a prediction made on
    the morning of a match could not have seen them, so training on them measures
    the model against information it would not have had at prediction time.

    They remain usable, but only for a caller that states it predicts at kickoff.
    """
    for column in ("odds_home_close", "odds_draw_close", "odds_away_close", "overround"):
        assert ODDS_COLUMNS_BY_NAME[column].availability is Availability.PRE_KICKOFF
        assert column in PRE_KICKOFF_COLUMNS
        with pytest.raises(LeakageError, match="PRE_KICKOFF"):
            assert_no_leakage([column])


def test_closing_odds_are_admitted_when_predicting_at_kickoff() -> None:
    """The opt-in must work, so a kickoff-time model is not blocked."""
    assert_no_leakage(["odds_home_close", "overround"], allow_pre_kickoff=True)


def test_opening_odds_are_pre_match_information() -> None:
    """Opening prices are available days ahead, so they are legitimate features."""
    assert_no_leakage(["odds_home_open", "odds_draw_open", "odds_away_open"])
    for column in ("odds_home_open", "odds_draw_open", "odds_away_open"):
        assert ODDS_COLUMNS_BY_NAME[column].availability is Availability.PRE_MATCH


def test_engineered_features_are_classified_as_pre_match() -> None:
    """Derived features must be registered, not merely asserted safe.

    A feature nobody classified is a feature nobody checked. Each engineered
    column has to appear in the schema with an explicit availability before the
    guard will accept it.
    """
    assert_no_leakage(ENGINEERED_FEATURE_COLUMNS_BY_NAME)
    for name, spec in ENGINEERED_FEATURE_COLUMNS_BY_NAME.items():
        assert spec.availability is Availability.PRE_MATCH, f"{name} is not classified pre-match"


def test_unknown_column_is_rejected() -> None:
    """An unrecognised column must be refused rather than passed through.

    An unknown name is far more likely to be a mistake or an unregistered
    provider column than a deliberate post-match feature.
    """
    with pytest.raises(LeakageError, match="not a recognised canonical column"):
        assert_no_leakage(["home_shots_rolling_avg_mystery"])


def test_leakage_error_lists_every_problem_at_once() -> None:
    """A caller fixing several problems should see them all in one pass.

    Reporting one problem per run turns a three-column mistake into three
    debugging cycles, so all of them must appear together.

    ``league_key`` is deliberately *not* expected in the list: it is a
    legitimate pre-match column, and including it here would test that a valid
    column is flagged, which is the opposite of what we want.
    """
    with pytest.raises(LeakageError) as excinfo:
        assert_no_leakage(["home_xg", "result", "league_key", "who_knows"])
    message = str(excinfo.value)

    for token in ("home_xg", "result", "who_knows"):
        assert token in message, f"{token} missing from the error message"
    assert "3 problem(s)" in message
    assert "league_key" not in message, "a valid pre-match column must not be reported as a problem"


def test_leakage_error_names_the_context() -> None:
    """The error should say which model was being built."""
    with pytest.raises(LeakageError, match="halftime model"):
        assert_no_leakage(["away_corners"], context="halftime model")


def test_feature_plan_returns_validated_tuple() -> None:
    """A valid feature list is returned unchanged as a tuple."""
    plan = feature_plan(["league_key", "match_date", "home_team", "away_team"])
    assert plan == ("league_key", "match_date", "home_team", "away_team")
    assert isinstance(plan, tuple)


def test_empty_feature_list_is_valid() -> None:
    """An empty feature list is legal; it is a model bug, not a schema bug."""
    assert feature_plan([]) == ()


def test_post_match_columns_are_not_usable_as_pre_match_features_in_bulk() -> None:
    """Passing the whole frame's columns at once must still be refused."""
    all_columns = [c.name for c in MATCH_COLUMNS]
    with pytest.raises(LeakageError):
        assert_no_leakage(all_columns)


def test_shuffle_cannot_change_schema_semantics() -> None:
    """Column order must not affect which columns are considered legal.

    Guards against an implementation that checks only the first column.
    """
    columns = ["league_key", "home_goals", "away_team"]
    with pytest.raises(LeakageError):
        assert_no_leakage(columns)
    with pytest.raises(LeakageError):
        assert_no_leakage(list(reversed(columns)))


def test_leakage_detection_works_on_a_realistic_feature_frame() -> None:
    """A frame mixing legal and illegal columns must be rejected with context.

    Mirrors the realistic failure: someone selects every numeric column with
    ``df.select_dtypes('number')`` and accidentally picks up xG.
    """
    frame = pd.DataFrame(
        {
            "league_key": ["ENG_PL"],
            "home_team": ["Liverpool"],
            "away_team": ["Arsenal"],
            "home_xg": [1.4],
            "away_goals": [2],
        }
    )
    numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    assert "home_xg" in numeric, "the realistic mistake should reproduce in this fixture"
    with pytest.raises(LeakageError, match="final whistle"):
        assert_no_leakage(numeric)

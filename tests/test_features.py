"""Feature pipeline tests.

The central question for every test here is not "does the number look plausible"
but "could this number have been computed at kickoff?". A feature pipeline that
is subtly misaligned or draws on a fallback value built from future matches
produces perfectly reasonable-looking numbers and a quietly fictional backtest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predict_football.data.schema import Availability, LeakageError, assert_no_leakage
from predict_football.features import (
    FEATURE_COLUMNS,
    FORM_WINDOWS,
    build_fixture_features,
    build_pre_match_features,
    describe_features,
    feature_plan,
)


def _synthetic_matches() -> pd.DataFrame:
    """Build a synthetic eight-club league with two tiers of known strength.

    Synthetic by construction so it cannot resemble real provider rows. The
    schedule is a proper round robin built with the circle method: every club
    plays every other club exactly once, and each club plays at most one match
    per date. That constraint matters. An earlier version of this fixture put a
    club on the pitch twice in one round, which made the expected history
    ambiguous and produced failures that looked like implementation bugs but
    were really a broken fixture.

    Returns:
        Match frame with the canonical columns the feature builder needs.
    """
    rng = np.random.default_rng(20260810)
    strong = ["Alpha", "Bravo", "Charlie", "Delta"]
    weak = ["Echo", "Foxtrot", "Golf", "Hotel"]
    teams = strong + weak

    rows: list[dict[str, object]] = []
    cycle_length = len(teams) - 1
    for cycle in range(3):
        # Three cycles of the circle method, alternating venues, so a club
        # accumulates enough history for career averages to be stable. Seven
        # rounds was too noisy to assert the direction of a feature.
        rotator = list(teams[1:])
        rotator = rotator[cycle:] + rotator[:cycle]
        for round_index in range(cycle_length):
            rotator = rotator[-1:] + rotator[:-1]
            pairs = [(teams[0], rotator[0])]
            for i in range(1, len(teams) // 2):
                pairs.append((rotator[i], rotator[len(teams) - 1 - i]))
            if cycle % 2 == 1:
                pairs = [(away, home) for home, away in pairs]

            date = pd.Timestamp("2021-01-01") + pd.Timedelta(days=7 * (cycle * cycle_length + round_index))
            for home, away in pairs:
                home_rate = 2.4 if home in strong else 0.6
                away_rate = 2.4 if away in strong else 0.6
                home_goals = int(rng.poisson(home_rate))
                away_goals = int(rng.poisson(away_rate))
                if home_goals > away_goals:
                    result = "H"
                elif home_goals < away_goals:
                    result = "A"
                else:
                    result = "D"
                rows.append(
                    {
                        "match_id": f"m{len(rows):04d}",
                        "match_date": date,
                        "season_label": "2020/21",
                        "home_team": home,
                        "away_team": away,
                        "home_goals": home_goals,
                        "away_goals": away_goals,
                        "result": result,
                    }
                )

    frame = pd.DataFrame(rows)
    # Guard the fixture itself: the tests below assume a club never appears twice
    # on one date, and a broken fixture should fail loudly rather than produce
    # confusing downstream failures.
    per_day = frame.groupby("match_date")["home_team"].apply(list)
    assert all(len(set(v)) == len(v) for v in per_day), "fixture double-books a club"
    return frame


def test_features_are_produced_for_every_played_match() -> None:
    """Every match with a result must get a feature row, one per fixture."""
    matches = _synthetic_matches()
    features = build_pre_match_features(matches)

    assert len(features) == len(matches)
    for column in FEATURE_COLUMNS:
        assert column in features.columns, f"{column} was never produced"
        assert pd.api.types.is_float_dtype(features[column]), f"{column} is not float"


def test_all_declared_features_are_numeric_and_finite() -> None:
    """No inf values and no unexplained nulls once a league has started.

    An inf in a feature silently becomes a huge coefficient during fitting. Nulls
    are acceptable only where the league genuinely has no prior match yet.
    """
    features = build_pre_match_features(_synthetic_matches())
    later = features.iloc[1:]

    for column in FEATURE_COLUMNS:
        values = later[column].to_numpy(dtype="float64")
        assert not np.isinf(values).any(), f"{column} contains inf"
        assert not np.isnan(values).any(), f"{column} has nulls beyond the first match"


def test_a_match_never_sees_its_own_result() -> None:
    """Changing a match's own outcome must not change any of its features.

    This is the direct test for target leakage: the feature row for a fixture is
    built from its history, so it must be blind to the fixture being predicted.
    """
    matches = _synthetic_matches()
    baseline = build_pre_match_features(matches).set_index("match_id")

    tampered = matches.copy()
    row = tampered.index[7]
    tampered.loc[row, ["home_goals", "away_goals"]] = 9, 0
    tampered.loc[row, "result"] = "H"
    after = build_pre_match_features(tampered).set_index("match_id")

    target = matches.loc[row, "match_id"]
    before_values = baseline.loc[target, list(FEATURE_COLUMNS)].to_numpy(dtype="float64")
    after_values = after.loc[target, list(FEATURE_COLUMNS)].to_numpy(dtype="float64")

    np.testing.assert_allclose(before_values, after_values, equal_nan=True)


def test_earlier_features_do_not_depend_on_later_results() -> None:
    """Rewriting the end of the season must not move features at the start.

    A target-only test is not sufficient. Fallback values for a club's first
    appearance are built from a league mean, and if that mean is assembled in the
    wrong order it quietly includes results from later fixtures. That bug passed
    a target-only probe and moved four features on the opening match of the real
    dataset, so it gets its own test.
    """
    matches = _synthetic_matches()
    baseline = build_pre_match_features(matches)

    later = matches.sort_values(["match_date", "match_id"]).tail(4)
    tampered = matches.copy()
    tampered.loc[tampered["match_id"].isin(later["match_id"]), "home_goals"] = 8
    tampered.loc[tampered["match_id"].isin(later["match_id"]), "result"] = "H"
    after = build_pre_match_features(tampered)

    early = baseline.sort_values(["match_date", "match_id"]).head(3)
    early_ids = set(early["match_id"])
    early_after = after[after["match_id"].isin(early_ids)]

    np.testing.assert_allclose(
        early[list(FEATURE_COLUMNS)].to_numpy(dtype="float64"),
        early_after[list(FEATURE_COLUMNS)].to_numpy(dtype="float64"),
        equal_nan=True,
    )


def test_career_scoring_rate_matches_a_hand_computed_average() -> None:
    """The headline feature must equal a naive replay of the same history.

    The expectation is derived independently here by walking the fixtures in
    order and averaging each club's own previous results, rather than by reading
    a figure back out of the implementation.
    """
    matches = _synthetic_matches()
    features = build_pre_match_features(matches).set_index("match_id")

    history: dict[str, list[int]] = {}
    checked = 0
    for row in matches.sort_values(["match_date", "match_id"]).itertuples():
        previous_home = history.get(row.home_team, [])
        if previous_home:
            expected = sum(previous_home) / len(previous_home)
            actual = features.loc[row.match_id, "home_career_gf_avg"]
            assert actual == pytest.approx(expected), f"history misaligned for {row.match_id}"
            checked += 1
        history.setdefault(row.home_team, []).append(int(row.home_goals))
        history.setdefault(row.away_team, []).append(int(row.away_goals))

    assert checked > 0, "the synthetic schedule produced no club with prior history"


def test_a_stronger_club_gets_a_higher_attack_strength() -> None:
    """Sign convention check: the synthetic strong side must out-score the weak.

    A flipped sign would still produce finite, plausible, leakage-free numbers, so
    it can only be caught by an assertion about direction.
    """
    features = build_pre_match_features(_synthetic_matches())
    settled = features[features["games_played"] >= 6]

    strong_home = settled["home_team"].isin(["Alpha", "Bravo", "Charlie", "Delta"])
    weak_away = settled["away_team"].isin(["Echo", "Foxtrot", "Golf", "Hotel"])

    assert (settled.loc[strong_home & weak_away, "attack_strength"] > 0).all()
    assert (settled.loc[weak_away & ~strong_home, "attack_strength"] < 0).all()


def test_rest_advantage_rewards_the_side_that_rested_longer() -> None:
    """rest_advantage must be home rest days minus away rest days."""
    features = build_pre_match_features(_synthetic_matches())
    expected = features["home_days_since_last"] - features["away_days_since_last"]
    np.testing.assert_allclose(features["rest_advantage"], expected)


def test_recent_form_window_ignores_older_matches() -> None:
    """The shortest form window must be computable from the last few matches only.

    Recomputed independently: the mean of the trailing window should beat the
    career mean more often than not for a club whose results moved recently.
    """
    matches = _synthetic_matches()
    features = build_pre_match_features(matches).set_index("match_id")

    shortest = min(FORM_WINDOWS)
    history: dict[str, list[float]] = {}
    verified = 0
    for row in matches.sort_values(["match_date", "match_id"]).itertuples():
        for team, scored in (
            (row.home_team, row.home_goals),
            (row.away_team, row.away_goals),
        ):
            past = history.get(team, [])[-shortest:]
            if len(past) == shortest:
                expected = sum(past) / shortest
                actual = features.loc[row.match_id, f"{'home' if team == row.home_team else 'away'}_form_gf_w{shortest}"]
                assert actual == pytest.approx(expected)
                verified += 1
            history.setdefault(team, []).append(float(scored))

    assert verified > 0


def test_build_rejects_an_empty_frame() -> None:
    """An empty frame is a caller bug, not an empty result."""
    with pytest.raises(ValueError, match="empty frame"):
        build_pre_match_features(pd.DataFrame(columns=["match_id"]))


def test_build_rejects_a_frame_with_no_results() -> None:
    """With nothing played there is no history, and saying so beats returning junk."""
    matches = _synthetic_matches()
    unplayed = matches.assign(result=pd.NA, home_goals=pd.NA, away_goals=pd.NA)

    with pytest.raises(ValueError, match="no row has a result"):
        build_pre_match_features(unplayed)


def test_build_rejects_a_frame_missing_history_columns() -> None:
    """Missing columns must be reported by name, not surfaced as a later KeyError."""
    matches = _synthetic_matches().drop(columns=["away_goals"])

    with pytest.raises(ValueError, match="away_goals"):
        build_pre_match_features(matches)


def test_build_rejects_a_fixture_where_a_club_plays_itself() -> None:
    """A club cannot be both home and away; that breaks the per-side split."""
    matches = _synthetic_matches()
    matches.loc[3, "away_team"] = matches.loc[3, "home_team"]

    with pytest.raises(ValueError, match="same club"):
        build_pre_match_features(matches)


def test_unplayed_fixtures_contribute_history_but_get_no_features() -> None:
    """A scheduled match with no result must not appear as a feature row."""
    matches = _synthetic_matches()
    future = matches.iloc[[0]].copy()
    future["match_id"] = "future-1"
    future["match_date"] = matches["match_date"].max() + pd.Timedelta(days=21)
    future["result"] = pd.NA
    future["home_goals"] = pd.NA
    future["away_goals"] = pd.NA

    combined = pd.concat([matches, future], ignore_index=True)
    features = build_pre_match_features(combined)

    assert "future-1" not in set(features["match_id"])
    assert len(features) == len(matches)


def test_shuffling_the_input_does_not_change_any_feature() -> None:
    """Row order must not matter, because ordering is derived from dates.

    Without the internal sort, passing rows in arbitrary order would quietly
    produce different history and a different model.
    """
    matches = _synthetic_matches()
    ordered = build_pre_match_features(matches).set_index("match_id")
    shuffled = build_pre_match_features(matches.sample(frac=1.0, random_state=7)).set_index("match_id")

    np.testing.assert_allclose(
        ordered.loc[ordered.index, list(FEATURE_COLUMNS)].to_numpy(dtype="float64"),
        shuffled.loc[ordered.index, list(FEATURE_COLUMNS)].to_numpy(dtype="float64"),
        equal_nan=True,
    )


def test_feature_plan_declares_the_columns_it_returns() -> None:
    """The plan and the output must not drift apart."""
    planned = feature_plan()

    assert planned == FEATURE_COLUMNS


def test_declared_features_are_registered_pre_match_in_the_schema() -> None:
    """Each feature must be classified in the canonical schema.

    Asserting safety at runtime is not a substitute for registering the column,
    because the next contributor would then add an unclassified feature.
    """
    assert_no_leakage(FEATURE_COLUMNS)

    table = describe_features()
    assert len(table) == len(FEATURE_COLUMNS)
    assert list(table["feature"]) == list(FEATURE_COLUMNS)

    from predict_football.data.schema import ENGINEERED_FEATURE_COLUMNS_BY_NAME

    for column in FEATURE_COLUMNS:
        assert column in ENGINEERED_FEATURE_COLUMNS_BY_NAME, f"{column} is unregistered"
        assert ENGINEERED_FEATURE_COLUMNS_BY_NAME[column].availability is Availability.PRE_MATCH


def test_leakage_guard_still_rejects_a_forged_feature() -> None:
    """Guard the guard: the runtime check must still fail on a target column."""
    with pytest.raises(LeakageError):
        assert_no_leakage(["home_goals"])


# --------------------------------------------------------------------------
# fixture features (fixtures that may not have been played yet)
# --------------------------------------------------------------------------


def _future_fixture(matches: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    """Return a single unplayed fixture dated after every match in ``matches``.

    Args:
        matches: History frame used to place the fixture in time.
        home: Home club.
        away: Away club.

    Returns:
        One-row fixture frame carrying only the columns a forecast needs.
    """
    return pd.DataFrame(
        {
            "match_id": ["future-1"],
            "match_date": [matches["match_date"].max() + pd.Timedelta(days=21)],
            "home_team": [home],
            "away_team": [away],
        }
    )


def test_fixture_features_reproduce_the_training_table_for_played_matches() -> None:
    """A played fixture's features must equal its training row.

    The two builders are independent implementations of the same strictly-before
    rule, so agreement over every settled row is a strong check that the fast
    prefix path has not drifted from the replay path. Rows where neither club has
    any prior appearance are excluded, since both sides then fall back to a league
    mean that is itself undefined at that point.
    """
    matches = _synthetic_matches()
    training = build_pre_match_features(matches).set_index("match_id")
    fixtures = build_fixture_features(matches, matches).set_index("match_id")

    settled = training["games_played"] >= 1
    assert settled.any()
    np.testing.assert_allclose(
        training.loc[settled, list(FEATURE_COLUMNS)].to_numpy(dtype="float64"),
        fixtures.loc[settled, list(FEATURE_COLUMNS)].to_numpy(dtype="float64"),
        equal_nan=True,
    )


def test_fixture_features_use_only_matches_before_the_fixture() -> None:
    """Rewriting later results must not move an earlier fixture's features."""
    matches = _synthetic_matches()
    ordered = matches.sort_values(["match_date", "match_id"])
    fixture = ordered.iloc[[6]].drop(columns=["result", "home_goals", "away_goals"])
    baseline = build_fixture_features(matches, fixture).set_index("match_id")

    later = ordered.tail(5)["match_id"]
    tampered = matches.copy()
    tampered.loc[tampered["match_id"].isin(later), "home_goals"] = 9
    tampered.loc[tampered["match_id"].isin(later), "result"] = "H"
    after = build_fixture_features(tampered, fixture).set_index("match_id")

    np.testing.assert_allclose(
        baseline.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype="float64"),
        after.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype="float64"),
        equal_nan=True,
    )


def test_fixture_features_ignore_the_fixtures_own_result() -> None:
    """Changing a played fixture's own outcome must not change its features.

    This is the direct leakage test for the fixture path: the features come from
    the history passed in, and the fixture's own same-date row is strictly
    excluded, so nothing about the row being predicted can influence itself.
    """
    matches = _synthetic_matches()
    fixture = matches.sort_values(["match_date", "match_id"]).iloc[[6]].drop(
        columns=["result", "home_goals", "away_goals"]
    )
    baseline = build_fixture_features(matches, fixture)

    target = fixture["match_id"].iloc[0]
    tampered = matches.copy()
    tampered.loc[tampered["match_id"] == target, ["home_goals", "away_goals"]] = 9, 0
    tampered.loc[tampered["match_id"] == target, "result"] = "H"
    after = build_fixture_features(tampered, fixture)

    np.testing.assert_allclose(
        baseline[list(FEATURE_COLUMNS)].to_numpy(dtype="float64"),
        after[list(FEATURE_COLUMNS)].to_numpy(dtype="float64"),
        equal_nan=True,
    )


def test_fixture_features_count_only_earlier_appearances() -> None:
    """An unplayed fixture's history must be exactly the matches before it.

    The count is recomputed here from the raw schedule rather than read back from
    the implementation, so a shifted or inclusive date cut would be caught.
    """
    matches = _synthetic_matches()
    home, away = "Alpha", "Echo"
    features = build_fixture_features(matches, _future_fixture(matches, home, away)).iloc[0]

    home_prior = int((matches["home_team"].eq(home) | matches["away_team"].eq(home)).sum())
    away_prior = int((matches["home_team"].eq(away) | matches["away_team"].eq(away)).sum())

    assert features["games_played"] == min(home_prior, away_prior)


def test_fixture_features_preserve_the_fixture_index() -> None:
    """The output must be alignable back onto the caller's fixture frame."""
    matches = _synthetic_matches()
    fixture = matches.sort_values(["match_date", "match_id"]).iloc[[6, 7]].copy()
    fixture.index = [101, 202]

    features = build_fixture_features(matches, fixture)

    assert list(features.index) == [101, 202]


def test_fixture_features_reject_an_empty_history() -> None:
    """No completed match means no history to describe a fixture with."""
    with pytest.raises(ValueError, match="empty history"):
        build_fixture_features(pd.DataFrame(), _future_fixture(_synthetic_matches(), "Alpha", "Echo"))


def test_fixture_features_reject_an_empty_fixture_frame() -> None:
    """No fixtures means nothing to build, and saying so beats a silent blank."""
    with pytest.raises(ValueError, match="empty fixture"):
        build_fixture_features(_synthetic_matches(), pd.DataFrame())


def test_fixture_features_reject_a_history_with_no_result() -> None:
    """With nothing played there is no state to read off."""
    matches = _synthetic_matches().assign(result=pd.NA, home_goals=pd.NA, away_goals=pd.NA)

    with pytest.raises(ValueError, match="no row has a result"):
        build_fixture_features(matches, _future_fixture(_synthetic_matches(), "Alpha", "Echo"))


def test_fixture_features_name_a_missing_fixture_column() -> None:
    """Missing columns must be reported by name, not surface later as a KeyError."""
    fixture = _future_fixture(_synthetic_matches(), "Alpha", "Echo").drop(columns=["away_team"])

    with pytest.raises(ValueError, match="fixtures are missing"):
        build_fixture_features(_synthetic_matches(), fixture)


def test_fixture_features_reject_a_club_playing_itself() -> None:
    """A club cannot be both sides of one fixture."""
    matches = _synthetic_matches()
    fixture = _future_fixture(matches, "Alpha", "Alpha")

    with pytest.raises(ValueError, match="same club"):
        build_fixture_features(matches, fixture)

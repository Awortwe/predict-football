"""Pre-match feature engineering.

Every feature in this module is computed from matches that kicked off
*strictly before* the match being predicted. That single rule is what separates
a usable model from a flattering one, and it is enforced structurally rather
than by convention: the core trick is that a team's aggregate for a given match
is built with :meth:`pandas.core.groupby.DataFrameGroupBy.cumsum` minus the
match's own contribution, so the match can never see its own result.

The public entry point is :func:`build_pre_match_features`, which returns one
row per match with the home and away team's history attached.

A note on bookmaker prices
--------------------------
Opening prices are available days in advance and are safe to use. Closing prices
are set at kickoff, so they are *not* known at the moment a pre-match
prediction is made. They are therefore excluded from the feature set entirely
and used only as an evaluation benchmark in :mod:`predict_football.evaluation`.
Mixing them into the model would inflate measured skill with information the
model would not have had when it was asked to predict.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from predict_football.data.schema import assert_no_leakage

logger = logging.getLogger(__name__)

# Points awarded, and the counts we track per team appearance.
POINTS = {"H": 3, "D": 1, "A": 0}

# Windows for the short-form features. Three is the shortest span that still
# distinguishes a side in genuine form from one in a lucky run; ten is roughly a
# third of a league season and stops short of averaging away real strength.
FORM_WINDOWS: tuple[int, ...] = (3, 5, 10)


@dataclass(frozen=True)
class FeatureSpec:
    """One engineered feature.

    Attributes:
        name: Column name in the engineered frame.
        side: ``"home"``, ``"away"`` or ``"both"``. A ``"both"`` feature is a
            difference between the two sides.
        description: What the value means, for documentation and the app.
    """

    name: str
    side: str
    description: str


FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec("games_played", "both", "Prior matches played by the less-established club"),
    FeatureSpec("rest_advantage", "both", "Home rest days minus away rest days"),
    FeatureSpec("form_ppg_diff", "both", "Home minus away points per game over the last 5 prior matches"),
    FeatureSpec("form_ppg_long", "both", "Home minus away points per game over the last 10 prior matches"),
    FeatureSpec("form_gf_diff", "both", "Home minus away goals scored per game over the last 5 prior matches"),
    FeatureSpec("form_ga_diff", "both", "Away minus home goals conceded per game, last 5, so higher favours home"),
    FeatureSpec("career_ppg_diff", "both", "Home minus away points per game over all prior matches"),
    FeatureSpec("career_gf_diff", "both", "Home minus away goals scored per game over all prior matches"),
    FeatureSpec("career_ga_diff", "both", "Away minus home goals conceded per game, so higher favours home"),
    FeatureSpec("home_attack_vs_away_defence", "both", "Home attack at home minus away defence on their travels"),
    FeatureSpec("away_attack_vs_home_defence", "both", "Away attack away minus home defence at home"),
    FeatureSpec("attack_strength", "both", "Home scoring rate relative to the away concession rate"),
    FeatureSpec("defence_strength", "both", "Home concession rate relative to the away scoring rate"),
)

FEATURE_COLUMNS: tuple[str, ...] = tuple(spec.name for spec in FEATURE_SPECS)

# Columns that are safe to hand a pre-match model because they describe the
# fixture itself rather than any result.
FIXTURE_COLUMNS: tuple[str, ...] = ("match_id", "league_key", "season", "match_date", "home_team", "away_team")


def feature_plan() -> tuple[str, ...]:
    """Return the engineered feature list after the leakage check.

    Engineered columns are registered in the canonical schema, so this is a real
    check rather than a formality: adding an unclassified feature fails here
    instead of quietly reaching a model.

    Returns:
        The validated feature names, in declaration order.

    Raises:
        LeakageError: If a feature name is not pre-match safe, which would mean
            the schema and the pipeline disagree about what is knowable.
    """
    assert_no_leakage(FEATURE_COLUMNS, context="pre-match feature set")
    return FEATURE_COLUMNS


def _team_appearances(matches: pd.DataFrame) -> pd.DataFrame:
    """Explode match rows into one row per team appearance.

    Args:
        matches: Match frame with a result, sorted chronologically.

    Returns:
        A long frame with one row per (match, team), sorted so that each team's
        own history runs in kickoff order. The ``_row`` column carries the
        originating match's position, which is how per-side features find their
        way back to the right fixture.
    """
    required = {"match_id", "match_date", "home_team", "away_team", "home_goals", "away_goals", "result"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"build_pre_match_features: frame is missing {sorted(missing)}")

    played = matches[matches["result"].notna()].copy()
    # Carry the originating row's *label*, not a fresh position, so the summary
    # can be aligned back onto the caller's frame with a plain reindex.
    played["_row"] = played.index

    # Coerce here rather than trusting the caller. A frame assembled by
    # concatenating played and scheduled matches carries object dtype, and the
    # cumulative sums below reject that.
    played["home_goals"] = pd.to_numeric(played["home_goals"], errors="coerce").astype("float64")
    played["away_goals"] = pd.to_numeric(played["away_goals"], errors="coerce").astype("float64")

    self_fixtures = played[played["home_team"] == played["away_team"]]
    if not self_fixtures.empty:
        raise ValueError(
            "build_pre_match_features: a fixture lists the same club on both sides, which cannot "
            f"happen in a real league (first offender: {self_fixtures.iloc[0]['match_id']})"
        )

    home = pd.DataFrame(
        {
            "_row": played["_row"],
            "match_id": played["match_id"],
            "match_date": played["match_date"],
            "team": played["home_team"],
            "opponent": played["away_team"],
            "is_home": True,
            "goals_for": played["home_goals"],
            "goals_against": played["away_goals"],
            "result": played["result"],
        }
    )
    away = pd.DataFrame(
        {
            "_row": played["_row"],
            "match_id": played["match_id"],
            "match_date": played["match_date"],
            "team": played["away_team"],
            "opponent": played["home_team"],
            "is_home": False,
            "goals_for": played["away_goals"],
            "goals_against": played["home_goals"],
            "result": played["result"].map({"H": "A", "D": "D", "A": "H"}),
        }
    )

    long = pd.concat([home, away], ignore_index=True)
    long["points"] = long["result"].map(POINTS).astype("float64")
    long["match_date"] = pd.to_datetime(long["match_date"])

    # Sorting by match_id within a date matters: two matches for one club on one
    # date are ordered deterministically, so the aggregate for either can only
    # ever include matches earlier in that ordering. Never later ones.
    return long.sort_values(["team", "match_date", "match_id"], kind="mergesort").reset_index(drop=True)


def _prior_only(long: pd.DataFrame) -> pd.DataFrame:
    """Attach prior-match aggregates to every team appearance.

    Each column is computed as a cumulative total *including* the current match
    minus the current match's own value, which is algebraically the total over
    strictly earlier matches only. Getting that subtraction wrong in either
    direction is the classic way to leak a result into its own features.

    Args:
        long: Team-appearance frame from :func:`_team_appearances`.

    Returns:
        The frame with prior-history columns attached.
    """
    out = long.copy()
    grouped = out.groupby("team", sort=False)
    team = out["team"]

    out["games_played"] = grouped.cumcount().astype("float64")

    # Cumulative totals *excluding* the current match. Every window below is a
    # difference of these series, so getting this step wrong contaminates every
    # form feature with the match it is trying to predict.
    sources = ["goals_for", "goals_against", "points"]
    prior_cumulative = (
        out[sources].groupby(out["team"], sort=False).cumsum().groupby(team).shift(1, fill_value=0.0)
    )

    for source in sources:
        cumulative = grouped[source].cumsum()
        out[f"_cum_{source}"] = cumulative
        out[f"career_{source}_sum"] = cumulative - out[source]

        for window in FORM_WINDOWS:
            # prior_cumulative[i] - prior_cumulative[i - window] is the total over
            # the matches strictly before this one and within the trailing
            # window. Anchoring on prior_cumulative rather than the raw
            # cumulative is what keeps the current match out of its own window.
            earlier = prior_cumulative[source].groupby(team).shift(window, fill_value=0.0)
            out[f"_{source}_last{window}_sum"] = prior_cumulative[source] - earlier
            # The denominator is how many matches the window actually spans,
            # which is min(games_played, window) -- not games_played.
            out[f"_{source}_last{window}_n"] = out["games_played"].clip(upper=float(window))

    out["days_since_last"] = grouped["match_date"].diff().dt.total_seconds() / 86400.0

    # Venue split, computed from prior matches only.
    for venue, flag in (("home", True), ("away", False)):
        venue_rows = out["is_home"] == flag
        venue_cum = out["goals_for"].where(venue_rows).groupby(team).cumsum()
        venue_cum_against = out["goals_against"].where(venue_rows).groupby(team).cumsum()
        venue_n = venue_rows.astype("int64").groupby(team).cumsum()
        out[f"_{venue}_n"] = venue_n
        out[f"_{venue}_gf_sum"] = venue_cum - out["goals_for"].where(venue_rows)
        out[f"_{venue}_ga_sum"] = venue_cum_against - out["goals_against"].where(venue_rows)

    return out


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two series, returning null instead of inf or NaN.

    Args:
        numerator: Series numerator.
        denominator: Series denominator, which may contain zeros.

    Returns:
        The ratio, with non-finite values replaced by null so that a club's
        first ever appearance is null rather than infinite.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = numerator.astype("float64") / denominator.astype("float64").replace(0.0, np.nan)
    return ratio.replace([np.inf, -np.inf], np.nan).astype("float64")


def _causal_league_mean(played: pd.DataFrame) -> pd.DataFrame:
    """League scoring rates per match, using only strictly earlier matches.

    This exists because a club's first appearance has no history of its own and
    needs a fallback. That fallback must itself be causal. An earlier version
    computed it from the team-sorted cumulative sums, which meant the "league
    mean" attached to an August match depended on results from March for clubs
    that happened to sort early in the alphabet. Tiny numbers, but the same
    species of bug as any other leak, and it made the two sides of a match
    disagree about what the league mean was.

    Args:
        played: Match frame with a result, in any order.

    Returns:
        Frame indexed like ``played`` with ``league_gf`` and ``league_ga``, the
        goals scored and conceded per match across all matches that kicked off
        strictly before this one. Null for the first match of the dataset.
    """
    ordered = played.sort_values(["match_date", "match_id"], kind="mergesort")
    prior_matches = pd.Series(np.arange(len(ordered), dtype="float64"), index=ordered.index)
    match_goals = (ordered["home_goals"] + ordered["away_goals"]).astype("float64")
    prior_goals = (match_goals.cumsum() - match_goals).astype("float64")
    points = ordered["result"].map(POINTS).astype("float64")
    prior_points = (points.cumsum() - points).astype("float64")

    return pd.DataFrame(
        {
            "league_gf": _safe_ratio(prior_goals, prior_matches),
            "league_ga": _safe_ratio(prior_goals, prior_matches),
            "league_ppg": _safe_ratio(prior_points, prior_matches),
        },
        index=ordered.index,
    ).reindex(played.index)


def _summarise(
    history: pd.DataFrame,
    prefix: str,
    league: pd.DataFrame,
) -> pd.DataFrame:
    """Turn prior sums into per-match averages for one side of a fixture.

    Args:
        history: Team-appearance frame with prior sums attached.
        prefix: Either ``"home"`` or ``"away"``.

    Returns:
        Frame of summary columns indexed by ``_row``, i.e. by originating match
        position, so it can be aligned straight back onto the match frame.
    """
    played = history["games_played"]
    out = pd.DataFrame(index=history.index)

    out[f"{prefix}_games_played"] = played
    out[f"{prefix}_days_since_last"] = history["days_since_last"]

    for window in FORM_WINDOWS:
        n = history[f"_points_last{window}_n"]
        out[f"{prefix}_form_ppg_w{window}"] = _safe_ratio(history[f"_points_last{window}_sum"], n)
        out[f"{prefix}_form_gf_w{window}"] = _safe_ratio(history[f"_goals_for_last{window}_sum"], n)
        out[f"{prefix}_form_ga_w{window}"] = _safe_ratio(history[f"_goals_against_last{window}_sum"], n)

    out[f"{prefix}_career_gf_avg"] = _safe_ratio(history["career_goals_for_sum"], played)
    out[f"{prefix}_career_ga_avg"] = _safe_ratio(history["career_goals_against_sum"], played)
    out[f"{prefix}_career_ppg"] = _safe_ratio(history["career_points_sum"], played)

    out[f"{prefix}_home_gf_avg"] = _safe_ratio(history["_home_gf_sum"], history["_home_n"])
    out[f"{prefix}_home_ga_avg"] = _safe_ratio(history["_home_ga_sum"], history["_home_n"])
    out[f"{prefix}_away_gf_avg"] = _safe_ratio(history["_away_gf_sum"], history["_away_n"])
    out[f"{prefix}_away_ga_avg"] = _safe_ratio(history["_away_ga_sum"], history["_away_n"])

    # A club with no history yet cannot be summarised. Fill with the league mean
    # from strictly earlier matches rather than zero: a zero here would read as
    # "has scored nothing", which is a claim the data does not support. The same
    # league mean is used for both sides of a match, so an evenly matched pair
    # produces exactly zero.
    prior = league.reindex(history["_row"].to_numpy())
    league_gf = prior["league_gf"].astype("float64")
    league_ga = prior["league_ga"].astype("float64")
    league_ppg = prior["league_ppg"].astype("float64")
    for column, fallback in (
        (f"{prefix}_career_gf_avg", league_gf),
        (f"{prefix}_career_ga_avg", league_ga),
        (f"{prefix}_career_ppg", league_ppg),
    ):
        fallback = fallback.copy()
        fallback.index = out.index
        out[column] = out[column].fillna(fallback)

    for window in FORM_WINDOWS:
        for stat, career_column in (
            ("form_ppg_w", f"{prefix}_career_ppg"),
            ("form_gf_w", f"{prefix}_career_gf_avg"),
            ("form_ga_w", f"{prefix}_career_ga_avg"),
        ):
            column = f"{prefix}_{stat}{window}"
            if column in out:
                out[column] = out[column].fillna(out[career_column])

    out[f"{prefix}_days_since_last"] = out[f"{prefix}_days_since_last"].fillna(7.0)
    return out.set_axis(history["_row"].to_numpy(), axis=0)


def _symmetric_features(features: pd.DataFrame) -> None:
    """Add the home-versus-away comparison columns the models consume.

    These are the features that actually describe a fixture rather than a club,
    so both the training table and the fixture table must derive them the same
    way; this helper is the single definition.

    Args:
        features: Frame carrying the per-side summary columns. Mutated in place.
    """
    features["games_played"] = features[["home_games_played", "away_games_played"]].min(axis=1)
    features["rest_advantage"] = features["home_days_since_last"] - features["away_days_since_last"]
    features["form_ppg_diff"] = features["home_form_ppg_w5"] - features["away_form_ppg_w5"]
    features["form_ppg_long"] = features["home_form_ppg_w10"] - features["away_form_ppg_w10"]
    features["form_gf_diff"] = features["home_form_gf_w5"] - features["away_form_gf_w5"]
    features["form_ga_diff"] = features["away_form_ga_w5"] - features["home_form_ga_w5"]
    features["career_ppg_diff"] = features["home_career_ppg"] - features["away_career_ppg"]
    features["career_gf_diff"] = features["home_career_gf_avg"] - features["away_career_gf_avg"]
    features["career_ga_diff"] = features["away_career_ga_avg"] - features["home_career_ga_avg"]
    features["home_attack_vs_away_defence"] = features["home_career_gf_avg"] - features["away_career_ga_avg"]
    features["away_attack_vs_home_defence"] = features["away_career_gf_avg"] - features["home_career_ga_avg"]
    features["attack_strength"] = features["home_career_gf_avg"] - features["away_career_ga_avg"]
    features["defence_strength"] = features["home_career_ga_avg"] - features["away_career_gf_avg"]


#: Prior-aggregate columns a synthetic appearance row must carry so that
#: :func:`_summarise` can turn a fixture's history into the same features a
#: played match gets. Kept explicit so a divergence shows up as a KeyError here.
_STATE_COLUMNS: tuple[str, ...] = (
    "games_played",
    "days_since_last",
    "career_goals_for_sum",
    "career_goals_against_sum",
    "career_points_sum",
    "_home_gf_sum",
    "_home_ga_sum",
    "_home_n",
    "_away_gf_sum",
    "_away_ga_sum",
    "_away_n",
    *(
        column
        for window in FORM_WINDOWS
        for column in (
            f"_points_last{window}_sum",
            f"_points_last{window}_n",
            f"_goals_for_last{window}_sum",
            f"_goals_against_last{window}_sum",
        )
    ),
)


@dataclass(frozen=True)
class _TeamHistory:
    """Prefix aggregates for one club, readable at any point in its history.

    A fixture's features must be the state of both clubs *just before kickoff*.
    Rather than replay the appearance machinery for every fixture, each club's
    cumulative totals are stored once and indexed by an integer cut: the number
    of that club's earlier appearances. Because a club never plays twice on one
    date, the cut is found with a plain date search, which keeps the
    strictly-before rule exact.

    Attributes:
        dates: Kickoff dates of the appearances, oldest first.
        goals_for: Cumulative goals scored, length ``len(dates) + 1``.
        goals_against: Cumulative goals conceded.
        points: Cumulative league points.
        home_goals_for: Cumulative goals scored in home appearances.
        home_goals_against: Cumulative goals conceded in home appearances.
        home_count: Cumulative home appearances.
        away_goals_for: Cumulative goals scored in away appearances.
        away_goals_against: Cumulative goals conceded in away appearances.
        away_count: Cumulative away appearances.
    """

    dates: np.ndarray
    goals_for: np.ndarray
    goals_against: np.ndarray
    points: np.ndarray
    home_goals_for: np.ndarray
    home_goals_against: np.ndarray
    home_count: np.ndarray
    away_goals_for: np.ndarray
    away_goals_against: np.ndarray
    away_count: np.ndarray

    @classmethod
    def from_appearances(cls, group: pd.DataFrame) -> _TeamHistory:
        """Build the prefix tables from one club's appearances, oldest first.

        Args:
            group: Appearance rows for a single club, sorted chronologically.

        Returns:
            A populated :class:`_TeamHistory`.
        """

        def prefix(values: np.ndarray) -> np.ndarray:
            """Prepend a zero so ``prefix[k]`` is the total over the first k rows."""
            return np.concatenate(([0.0], np.cumsum(values, dtype="float64")))

        is_home = group["is_home"].to_numpy(dtype=bool)
        goals_for = group["goals_for"].to_numpy(dtype="float64")
        goals_against = group["goals_against"].to_numpy(dtype="float64")
        points = group["points"].to_numpy(dtype="float64")
        return cls(
            dates=pd.to_datetime(group["match_date"]).to_numpy(),
            goals_for=prefix(goals_for),
            goals_against=prefix(goals_against),
            points=prefix(points),
            home_goals_for=prefix(np.where(is_home, goals_for, 0.0)),
            home_goals_against=prefix(np.where(is_home, goals_against, 0.0)),
            home_count=prefix(is_home.astype("float64")),
            away_goals_for=prefix(np.where(~is_home, goals_for, 0.0)),
            away_goals_against=prefix(np.where(~is_home, goals_against, 0.0)),
            away_count=prefix((~is_home).astype("float64")),
        )

    def prior_state(self, cut: int) -> dict[str, float]:
        """Return the prior-aggregate columns as of ``cut`` earlier appearances.

        Args:
            cut: Number of appearances to include. Clamped to the club's history.

        Returns:
            Mapping of :data:`_STATE_COLUMNS` (except ``days_since_last``, which
            depends on the fixture's own date) to their value at that cut.
        """
        cut = max(0, min(int(cut), len(self.dates)))
        state: dict[str, float] = {
            "games_played": float(cut),
            "career_goals_for_sum": float(self.goals_for[cut]),
            "career_goals_against_sum": float(self.goals_against[cut]),
            "career_points_sum": float(self.points[cut]),
            "_home_gf_sum": float(self.home_goals_for[cut]),
            "_home_ga_sum": float(self.home_goals_against[cut]),
            "_home_n": float(self.home_count[cut]),
            "_away_gf_sum": float(self.away_goals_for[cut]),
            "_away_ga_sum": float(self.away_goals_against[cut]),
            "_away_n": float(self.away_count[cut]),
        }
        for window in FORM_WINDOWS:
            start = max(0, cut - window)
            n = float(min(cut, window))
            state[f"_points_last{window}_sum"] = float(self.points[cut] - self.points[start])
            state[f"_points_last{window}_n"] = n
            state[f"_goals_for_last{window}_sum"] = float(self.goals_for[cut] - self.goals_for[start])
            state[f"_goals_against_last{window}_sum"] = float(self.goals_against[cut] - self.goals_against[start])
        return state


def _team_histories(long: pd.DataFrame) -> dict[str, _TeamHistory]:
    """Build one :class:`_TeamHistory` per club.

    Args:
        long: Team-appearance frame from :func:`_team_appearances`.

    Returns:
        Mapping of club name to its prefix aggregates.
    """
    return {str(team): _TeamHistory.from_appearances(group) for team, group in long.groupby("team", sort=False)}


def _fixture_states(
    histories: dict[str, _TeamHistory],
    teams: list[object],
    dates: list[object],
) -> pd.DataFrame:
    """Compute prior-aggregate columns for one side of every fixture.

    Args:
        histories: Per-club prefix aggregates.
        teams: The club on this side of each fixture, in fixture order.
        dates: The fixture's kickoff date, in the same order.

    Returns:
        Frame of :data:`_STATE_COLUMNS`, one row per fixture, with
        ``days_since_last`` measuring the gap to the club's most recent earlier
        match. A club with no history gets zeros and a null rest gap, which
        :func:`_summarise` replaces with the league mean and the default rest.
    """
    n = len(teams)
    columns = {name: np.full(n, np.nan, dtype="float64") for name in _STATE_COLUMNS}
    for i, (team, date) in enumerate(zip(teams, dates, strict=True)):
        table = histories.get(str(team))
        when = pd.Timestamp(date)
        if table is None:
            state = {
                name: 0.0 for name in _STATE_COLUMNS if name != "days_since_last"
            }
        else:
            cut = int(np.searchsorted(table.dates, when.to_datetime64(), side="left"))
            state = table.prior_state(cut)
            if cut:
                columns["days_since_last"][i] = (when - pd.Timestamp(table.dates[cut - 1])).total_seconds() / 86400.0
        for name, value in state.items():
            columns[name][i] = value
    return pd.DataFrame(columns)


def _fixture_league_means(played: pd.DataFrame, dates: list[object]) -> pd.DataFrame:
    """League scoring and points averages before each fixture's date.

    Args:
        played: Completed matches used as history.
        dates: Fixture kickoff dates, in fixture order.

    Returns:
        Frame indexed positionally like the fixtures, with ``league_gf``,
        ``league_ga`` and ``league_ppg`` as of strictly earlier matches. Null
        before the first match, which :func:`_summarise` tolerates as a fallback
        of nothing.
    """
    ordered = played.sort_values(["match_date", "match_id"], kind="mergesort")
    match_goals = (ordered["home_goals"].astype("float64") + ordered["away_goals"].astype("float64")).to_numpy()
    points = ordered["result"].map(POINTS).astype("float64").to_numpy()
    goals_prefix = np.concatenate(([0.0], np.cumsum(match_goals)))
    points_prefix = np.concatenate(([0.0], np.cumsum(points)))
    ordered_dates = pd.to_datetime(ordered["match_date"]).to_numpy()

    rows: list[dict[str, float]] = []
    for date in dates:
        cut = int(np.searchsorted(ordered_dates, pd.Timestamp(date).to_datetime64(), side="left"))
        if cut:
            mean_goals = goals_prefix[cut] / cut
            mean_points = points_prefix[cut] / cut
        else:
            mean_goals = np.nan
            mean_points = np.nan
        rows.append({"league_gf": mean_goals, "league_ga": mean_goals, "league_ppg": mean_points})
    return pd.DataFrame(rows)


def build_fixture_features(history: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Build features for fixtures that may not have been played yet.

    :func:`build_pre_match_features` only emits a row for a match that already
    has a result. A forecast needs the other case, and it must obey the same
    rule: every feature for a fixture is the state of both clubs strictly before
    that fixture's kickoff. The fixture's own result, if it exists, is ignored,
    so this function is safe to call on played rows too and produces features
    identical to the training table for a settled fixture.

    Args:
        history: Completed canonical match frame. Only rows with a result
            contribute, and only to fixtures dated strictly after them.
        fixtures: Fixtures to describe. Results, if present, are ignored.

    Returns:
        Frame indexed like ``fixtures`` carrying :data:`FEATURE_COLUMNS`.

    Raises:
        ValueError: If either frame is empty, lacks a required column, holds an
            unparseable date, has no completed history, or lists a club on both
            sides of one fixture.
    """
    if history is None or history.empty:
        raise ValueError("build_fixture_features: received an empty history frame")
    if fixtures is None or fixtures.empty:
        raise ValueError("build_fixture_features: received an empty fixture frame")

    required = {"match_id", "match_date", "home_team", "away_team", "home_goals", "away_goals", "result"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"build_fixture_features: history is missing {sorted(missing)}")
    fixture_required = {"match_date", "home_team", "away_team"}
    missing_fixtures = fixture_required - set(fixtures.columns)
    if missing_fixtures:
        raise ValueError(f"build_fixture_features: fixtures are missing {sorted(missing_fixtures)}")

    ordered = history.sort_values(["match_date", "match_id"], kind="mergesort")
    played = ordered[ordered["result"].notna()]
    if played.empty:
        raise ValueError("build_fixture_features: no row has a result, so no history can be built")

    frame = fixtures.copy()
    frame["match_date"] = pd.to_datetime(frame["match_date"])
    self_fixtures = frame[frame["home_team"] == frame["away_team"]]
    if not self_fixtures.empty:
        raise ValueError(
            "build_fixture_features: a fixture lists the same club on both sides, which cannot "
            f"happen in a real league (first offender: {self_fixtures.iloc[0].get('match_id', '?')})"
        )

    histories = _team_histories(_team_appearances(ordered))
    dates = frame["match_date"].tolist()
    features = frame.reset_index(drop=True)
    league = _fixture_league_means(played, dates)

    home_states = _fixture_states(histories, frame["home_team"].tolist(), dates)
    away_states = _fixture_states(histories, frame["away_team"].tolist(), dates)
    home_states["_row"] = np.arange(len(frame))
    away_states["_row"] = np.arange(len(frame))

    for state, prefix in ((home_states, "home"), (away_states, "away")):
        summary = _summarise(state, prefix, league)
        for column in summary.columns:
            features[column] = summary[column].reindex(features.index).to_numpy()

    _symmetric_features(features)

    for column in FEATURE_COLUMNS:
        features[column] = pd.to_numeric(features[column], errors="coerce").astype("float64")

    assert_no_leakage(FEATURE_COLUMNS, context="engineered fixture features")
    logger.info("Built %d feature row(s) for fixtures across %d features", len(features), len(FEATURE_COLUMNS))
    features.index = fixtures.index
    return features


def build_pre_match_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Build the pre-match feature table.

    One row per match, carrying the fixture identifiers and the engineered
    features for both sides. Features describe only what was known before kickoff.

    Args:
        matches: Canonical match frame, as returned by
            :meth:`~predict_football.data.repository.MatchRepository.load_matches`.
            Rows without a result contribute history but get no features of
            their own, since a fixture's own outcome is not yet known.

    Returns:
        Feature frame indexed by the original match frame's index, with
        :data:`FEATURE_COLUMNS` plus fixture identifiers and the target.

    Raises:
        ValueError: If the frame is empty or lacks the columns needed to compute
            history.
    """
    if matches is None or matches.empty:
        raise ValueError("build_pre_match_features: received an empty frame")

    required = {"match_id", "match_date", "home_team", "away_team", "home_goals", "away_goals", "result"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"build_pre_match_features: frame is missing {sorted(missing)}")

    ordered = matches.sort_values(["match_date", "match_id"], kind="mergesort")
    played = ordered[ordered["result"].notna()]
    if played.empty:
        raise ValueError("build_pre_match_features: no row has a result, so no history can be built")

    long = _prior_only(_team_appearances(ordered))

    home_rows = long[long["is_home"]]
    away_rows = long[~long["is_home"]]
    if len(home_rows) != len(played) or len(away_rows) != len(played):
        raise ValueError(
            "build_pre_match_features: a match produced an uneven number of team rows, which "
            "means a fixture lists the same club twice on one side"
        )

    features = played.copy()
    league = _causal_league_mean(played)
    for rows, prefix in ((home_rows, "home"), (away_rows, "away")):
        summary = _summarise(rows, prefix, league)
        for column in summary.columns:
            features[column] = summary[column].reindex(features.index).to_numpy()

    _symmetric_features(features)

    for column in FEATURE_COLUMNS:
        features[column] = pd.to_numeric(features[column], errors="coerce").astype("float64")

    assert_no_leakage(FEATURE_COLUMNS, context="engineered pre-match features")
    logger.info("Built %d feature rows across %d features", len(features), len(FEATURE_COLUMNS))
    return features


def describe_features() -> pd.DataFrame:
    """Return the engineered features as a documentation table.

    Returns:
        DataFrame with name, side and description for each feature.
    """
    return pd.DataFrame(
        [{"feature": s.name, "side": s.side, "description": s.description} for s in FEATURE_SPECS]
    )


__all__ = [
    "FEATURE_COLUMNS",
    "FEATURE_SPECS",
    "FORM_WINDOWS",
    "FeatureSpec",
    "build_fixture_features",
    "build_pre_match_features",
    "describe_features",
    "feature_plan",
]

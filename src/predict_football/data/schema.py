"""Canonical data schema and the leakage guard that protects it.

The single most dangerous bug in sports modelling is not a broken formula, it is
subtle leakage: a feature that contains information only available *after* the
thing you are predicting. Shots, corners, cards, xG and half-time scores all
look harmless in a DataFrame and all destroy honest evaluation.

This module encodes the distinction once, as data, and makes every feature
builder go through it:

* :attr:`Availability.PRE_MATCH` -- knowable before kickoff, safe as a feature.
* :attr:`Availability.PRE_KICKOFF` -- knowable only in the minutes immediately
  before kickoff. Bookmaker *closing* prices are the motivating case: they are
  set at the close of betting, not days ahead. A prediction made on the morning
  of a match could not have used them, so a model that trains on them looks
  better than it is. They are rejected by default and admitted only where the
  caller states it is genuinely predicting at kickoff.
* :attr:`Availability.TARGET` -- the answer. Used as the label only.
* :attr:`Availability.POST_MATCH` -- only knowable after the final whistle.
  **Never** a feature for a pre-match model. (This is *not* leakage when used
  for post-match analysis, which is a different question.)

:func:`assert_no_leakage` is the enforcement point, and it is tested.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum

import pandas as pd


class Availability(str, Enum):
    """When a column's value becomes knowable relative to kickoff."""

    PRE_MATCH = "pre_match"
    PRE_KICKOFF = "pre_kickoff"
    TARGET = "target"
    POST_MATCH = "post_match"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass(frozen=True)
class ColumnSpec:
    """One column in the canonical match table.

    Attributes:
        name: Column name as it appears in the DataFrame.
        dtype: Canonical dtype string, e.g. ``"int64"`` or ``"float64"``.
        availability: Whether the value is knowable before kickoff.
        nullable: Whether missing values are expected and legitimate.
        description: What the column means, for data dictionaries and the UI.
    """

    name: str
    dtype: str
    availability: Availability
    nullable: bool
    description: str


def _spec(name: str, dtype: str, availability: Availability, nullable: bool, description: str) -> ColumnSpec:
    return ColumnSpec(name=name, dtype=dtype, availability=availability, nullable=nullable, description=description)


#: Version of the canonical schema. Bump on any breaking column change; stored
#: in the database ``meta`` table so a stale database is detectable.
SCHEMA_VERSION = "1.0.0"


MATCH_COLUMNS: tuple[ColumnSpec, ...] = (
    _spec("match_id", "string", Availability.PRE_MATCH, False, "Deterministic cross-provider match identifier"),
    _spec("source", "string", Availability.PRE_MATCH, False, "Provider the row was ingested from"),
    _spec("league_key", "string", Availability.PRE_MATCH, False, "Internal competition key, e.g. ENG_PL"),
    _spec("competition_type", "string", Availability.PRE_MATCH, False, "'league' or 'cup'"),
    _spec("season", "string", Availability.PRE_MATCH, False, "Season label, e.g. 2024/25"),
    _spec("season_code", "string", Availability.PRE_MATCH, True, "Four-character season code, e.g. 2425"),
    _spec("match_date", "datetime64[ns]", Availability.PRE_MATCH, False, "Local kickoff date"),
    _spec("match_week", "Int64", Availability.PRE_MATCH, True, "Matchday number within the season"),
    _spec("home_team", "string", Availability.PRE_MATCH, False, "Canonical home team name"),
    _spec("away_team", "string", Availability.PRE_MATCH, False, "Canonical away team name"),
    _spec("venue", "string", Availability.PRE_MATCH, True, "Stadium name when published in advance"),
    _spec("referee", "string", Availability.POST_MATCH, True, (
        "Match referee. Officials are actually appointed days in advance, so this is "
        "technically pre-match knowable, but the provider only publishes it once the "
        "match has been played. Treated as POST_MATCH so it cannot be used as a feature "
        "without an explicit, deliberate decision"
    )),
    # --- Target: the outcome we are trying to predict ---
    _spec("home_goals", "Int64", Availability.TARGET, True, "Full-time home goals"),
    _spec("away_goals", "Int64", Availability.TARGET, True, "Full-time away goals"),
    _spec("result", "string", Availability.TARGET, True, "Full-time result: H, D or A"),
    _spec("result_et", "string", Availability.TARGET, True, "Result after extra time, if played"),
    _spec("pens_home", "Int64", Availability.TARGET, True, "Penalty shootout home goals"),
    _spec("pens_away", "Int64", Availability.TARGET, True, "Penalty shootout away goals"),
    # --- Post-match: team match statistics. Legitimate for post-match
    #     analysis and for label construction, never a pre-match feature. ---
    _spec("home_goals_ht", "Int64", Availability.POST_MATCH, True, "Half-time home goals"),
    _spec("away_goals_ht", "Int64", Availability.POST_MATCH, True, "Half-time away goals"),
    _spec("home_shots", "Int64", Availability.POST_MATCH, True, "Home shots"),
    _spec("away_shots", "Int64", Availability.POST_MATCH, True, "Away shots"),
    _spec("home_shots_on_target", "Int64", Availability.POST_MATCH, True, "Home shots on target"),
    _spec("away_shots_on_target", "Int64", Availability.POST_MATCH, True, "Away shots on target"),
    _spec("home_corners", "Int64", Availability.POST_MATCH, True, "Home corners"),
    _spec("away_corners", "Int64", Availability.POST_MATCH, True, "Away corners"),
    _spec("home_fouls", "Int64", Availability.POST_MATCH, True, "Home fouls"),
    _spec("away_fouls", "Int64", Availability.POST_MATCH, True, "Away fouls"),
    _spec("home_yellow", "Int64", Availability.POST_MATCH, True, "Home yellow cards"),
    _spec("away_yellow", "Int64", Availability.POST_MATCH, True, "Away yellow cards"),
    _spec("home_red", "Int64", Availability.POST_MATCH, True, "Home red cards"),
    _spec("away_red", "Int64", Availability.POST_MATCH, True, "Away red cards"),
    _spec("home_xg", "float64", Availability.POST_MATCH, True, "Home expected goals (post-match xG)"),
    _spec("away_xg", "float64", Availability.POST_MATCH, True, "Away expected goals (post-match xG)"),
)


MATCH_COLUMNS_BY_NAME: dict[str, ColumnSpec] = {c.name: c for c in MATCH_COLUMNS}
MATCH_COLUMN_NAMES: tuple[str, ...] = tuple(c.name for c in MATCH_COLUMNS)

#: Counterpart (home_/away_) pairs for post-match team statistics.
TEAM_STAT_PAIRS: tuple[tuple[str, str], ...] = (
    ("home_goals", "away_goals"),
    ("home_goals_ht", "away_goals_ht"),
    ("home_shots", "away_shots"),
    ("home_shots_on_target", "away_shots_on_target"),
    ("home_corners", "away_corners"),
    ("home_fouls", "away_fouls"),
    ("home_yellow", "away_yellow"),
    ("home_red", "away_red"),
    ("home_xg", "away_xg"),
)


def _names_with(availability: Availability) -> tuple[str, ...]:
    return tuple(c.name for c in MATCH_COLUMNS if c.availability is availability)


#: Columns legitimately usable as pre-match features. Note that bookmaker
#: *closing* odds are absent: they are set at the close of betting and are
#: tracked separately as :data:`PRE_KICKOFF_COLUMNS`, defined below once the
#: odds specs exist.
PRE_MATCH_COLUMNS: tuple[str, ...] = _names_with(Availability.PRE_MATCH)

#: The outcome columns. These are labels, never features.
TARGET_COLUMNS: tuple[str, ...] = _names_with(Availability.TARGET)

#: Columns that exist only after the match is over. Using any of these as a
#: pre-match feature is leakage and :func:`assert_no_leakage` will stop it.
POST_MATCH_COLUMNS: tuple[str, ...] = _names_with(Availability.POST_MATCH)

#: Bookmaker odds live in their own table but are pre-match information.
ODDS_COLUMNS: tuple[str, ...] = (
    "bookmaker",
    "market",
    "odds_home_open",
    "odds_draw_open",
    "odds_away_open",
    "odds_home_close",
    "odds_draw_close",
    "odds_away_close",
    "overround",
)

ODDS_COLUMNS_BY_NAME: dict[str, ColumnSpec] = {
    "bookmaker": _spec("bookmaker", "string", Availability.PRE_MATCH, False, "Bookmaker name"),
    "market": _spec("market", "string", Availability.PRE_MATCH, False, "Market: 1x2, ou_2_5, btts"),
    "odds_home_open": _spec(
        "odds_home_open", "float64", Availability.PRE_MATCH, True, "Opening 1X2 home odds, published days ahead"
    ),
    "odds_draw_open": _spec("odds_draw_open", "float64", Availability.PRE_MATCH, True, "Opening 1X2 draw odds"),
    "odds_away_open": _spec("odds_away_open", "float64", Availability.PRE_MATCH, True, "Opening 1X2 away odds"),
    "odds_home_close": _spec("odds_home_close", "float64", Availability.PRE_KICKOFF, True, "Closing 1X2 home odds"),
    "odds_draw_close": _spec("odds_draw_close", "float64", Availability.PRE_KICKOFF, True, "Closing 1X2 draw odds"),
    "odds_away_close": _spec("odds_away_close", "float64", Availability.PRE_KICKOFF, True, "Closing 1X2 away odds"),
    "overround": _spec(
        "overround",
        "float64",
        Availability.PRE_KICKOFF,
        False,
        "Bookmaker margin: sum of implied probabilities from closing odds. Above 1.0 means "
        "the bookmaker takes a cut, so raw implied probabilities must be normalised before use",
    ),
}


# --- Engineered features ----------------------------------------------------

#: Features derived by :mod:`predict_football.features` from prior matches.
#: They are registered here, with an explicit availability, for the same reason
#: provider columns are: a feature that is not classified is a feature nobody
#: has checked. All of them are built from matches that kicked off strictly
#: earlier, so all are genuinely pre-match.
ENGINEERED_FEATURE_COLUMNS: tuple[ColumnSpec, ...] = tuple(
    _spec(name, "float64", Availability.PRE_MATCH, True, description)
    for name, description in (
        ("games_played", "Prior matches played by the less-established club"),
        ("rest_advantage", "Home rest days minus away rest days"),
        ("form_ppg_diff", "Home minus away points per game over the last 5 prior matches"),
        ("form_ppg_long", "Home minus away points per game over the last 10 prior matches"),
        ("form_gf_diff", "Home minus away goals scored per game over the last 5 prior matches"),
        ("form_ga_diff", "Away minus home goals conceded per game, last 5, so higher favours home"),
        ("career_ppg_diff", "Home minus away points per game over all prior matches"),
        ("career_gf_diff", "Home minus away goals scored per game over all prior matches"),
        ("career_ga_diff", "Away minus home goals conceded per game, so higher favours home"),
        ("home_attack_vs_away_defence", "Home attack at home minus away defence on their travels"),
        ("away_attack_vs_home_defence", "Away attack away minus home defence at home"),
        ("attack_strength", "Home scoring rate relative to the away concession rate"),
        ("defence_strength", "Home concession rate relative to the away scoring rate"),
    )
)

ENGINEERED_FEATURE_COLUMNS_BY_NAME: dict[str, ColumnSpec] = {
    c.name: c for c in ENGINEERED_FEATURE_COLUMNS
}

#: Knowable only in the minutes before kickoff. Legitimate for a model that
#: predicts at the close of betting, dishonest for one that predicts days ahead,
#: so :func:`assert_no_leakage` refuses them unless the caller opts in.
PRE_KICKOFF_COLUMNS: tuple[str, ...] = tuple(
    name for name, spec in ODDS_COLUMNS_BY_NAME.items() if spec.availability is Availability.PRE_KICKOFF
)


# --- Events -----------------------------------------------------------------

#: Canonical event types, mapped to equivalent StatsBomb type names. This is the
#: vocabulary the live engine reasons over, independent of any one provider.
class EventType(str, Enum):
    """Canonical on-pitch event categories."""

    GOAL = "goal"
    SHOT = "shot"
    SHOT_ON_TARGET = "shot_on_target"
    SHOT_BLOCKED = "shot_blocked"
    SHOT_OFF_TARGET = "shot_off_target"
    SHOT_POST = "shot_post"
    OWN_GOAL = "own_goal"
    PENALTY_GOAL = "penalty_goal"
    MISSED_PENALTY = "missed_penalty"
    PASS = "pass"
    PASS_COMPLETED = "pass_completed"
    CARRY = "carry"
    CROSS = "cross"
    TACKLE = "tackle"
    INTERCEPTION = "interception"
    CLEARANCE = "clearance"
    BLOCK = "block"
    PRESSURE = "pressure"
    DUEL = "duel"
    FOUL = "foul"
    YELLOW_CARD = "yellow_card"
    SECOND_YELLOW = "second_yellow"
    RED_CARD = "red_card"
    SUBSTITUTION_ON = "substitution_on"
    SUBSTITUTION_OFF = "substitution_off"
    KICK_OFF = "kick_off"
    PERIOD_START = "period_start"
    PERIOD_END = "period_end"
    HALF_TIME = "half_time"
    FULL_TIME = "full_time"
    GOAL_KICK = "goal_kick"
    CORNER = "corner"
    THROW_IN = "throw_in"
    OFFSIDE = "offside"
    RECOVERY = "recovery"
    UNKNOWN = "unknown"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


EVENT_COLUMNS: tuple[ColumnSpec, ...] = (
    _spec("event_id", "string", Availability.POST_MATCH, False, "Unique event identifier within a match"),
    _spec("match_id", "string", Availability.PRE_MATCH, False, "Owning match"),
    _spec("period", "Int64", Availability.POST_MATCH, False, "1, 2 or extra-time period number"),
    _spec("minute", "Int64", Availability.POST_MATCH, False, "Minute within the match"),
    _spec("second", "Int64", Availability.POST_MATCH, False, "Seconds within the minute"),
    _spec("timestamp", "string", Availability.POST_MATCH, True, "Provider timestamp string"),
    _spec("team", "string", Availability.POST_MATCH, True, "Team in possession / committing the event"),
    _spec("player", "string", Availability.POST_MATCH, True, "Primary player name"),
    _spec("event_type", "string", Availability.POST_MATCH, False, "Canonical EventType value"),
    _spec("xg", "float64", Availability.POST_MATCH, True, "Shot expected goals, where published"),
    _spec("shot_outcome", "string", Availability.POST_MATCH, True, "Shot outcome (goal, blocked, off target, post)"),
    _spec("shot_body_part", "string", Availability.POST_MATCH, True, "Head, left foot, right foot, other"),
    _spec("shot_type", "string", Availability.POST_MATCH, True, "Open play, penalty, free kick, corner"),
    _spec("location_x", "float64", Availability.POST_MATCH, True, "Event x coordinate (0-120), provider scaled"),
    _spec("location_y", "float64", Availability.POST_MATCH, True, "Event y coordinate (0-80), provider scaled"),
    _spec("end_location_x", "float64", Availability.POST_MATCH, True, "End x coordinate where published"),
    _spec("end_location_y", "float64", Availability.POST_MATCH, True, "End y coordinate where published"),
    _spec("pass_length", "float64", Availability.POST_MATCH, True, "Pass distance in provider units"),
    _spec("pass_angle", "float64", Availability.POST_MATCH, True, "Pass angle in radians"),
    _spec("shot_assist", "string", Availability.POST_MATCH, True, "Player credited with the assist"),
    _spec(
        "related_event", "string", Availability.POST_MATCH, True,
        "Identifier of the paired event, e.g. substitution",
    ),
)

EVENT_COLUMNS_BY_NAME: dict[str, ColumnSpec] = {c.name: c for c in EVENT_COLUMNS}
EVENT_COLUMN_NAMES: tuple[str, ...] = tuple(c.name for c in EVENT_COLUMNS)


LINEUP_COLUMNS: tuple[ColumnSpec, ...] = (
    _spec("match_id", "string", Availability.PRE_MATCH, False, "Owning match"),
    _spec("team", "string", Availability.POST_MATCH, True, "Team name"),
    _spec("player", "string", Availability.POST_MATCH, False, "Player name"),
    _spec("player_id", "string", Availability.PRE_MATCH, True, "Provider player identifier"),
    _spec("position", "string", Availability.POST_MATCH, True, "Nominal position, e.g. Goalkeeper, Centre-Back"),
    _spec("jersey_number", "Int64", Availability.POST_MATCH, True, "Shirt number"),
    _spec("is_starter", "boolean", Availability.POST_MATCH, False, "Started the match"),
    _spec("is_captain", "boolean", Availability.POST_MATCH, True, "Team captain"),
    _spec("formation", "string", Availability.POST_MATCH, True, "Starting formation when the provider publishes one"),
)

LINEUP_COLUMN_NAMES: tuple[str, ...] = tuple(c.name for c in LINEUP_COLUMNS)


class LeakageError(ValueError):
    """Raised when post-match information is requested as a model feature."""


def assert_no_leakage(
    feature_columns: Iterable[str],
    *,
    context: str = "pre-match model",
    allow_pre_kickoff: bool = False,
) -> None:
    """Reject any feature column that is not knowable before the prediction.

    This is the central guard against data leakage. It is deliberately strict
    and deliberately noisy: it refuses a column if the name is unknown as well
    as if it is known to be unsafe, because an unrecognised name is far more
    likely to be a mistake than a deliberate post-match feature.

    Args:
        feature_columns: Column names a model wants to use as features.
        context: Human-readable description used in the error message.
        allow_pre_kickoff: Admit columns that are only knowable in the minutes
            before kickoff, such as bookmaker closing prices. Only a caller that
            genuinely predicts at kickoff should set this; a model predicting
            days ahead that sets it is measuring itself against information it
            could not have had.

    Raises:
        LeakageError: If a column is post-match or target information, if it is
            pre-kickoff only and ``allow_pre_kickoff`` is False, or if a name is
            not recognised at all.
    """
    post_match = set(POST_MATCH_COLUMNS)
    pre_kickoff = set(PRE_KICKOFF_COLUMNS)
    targets = set(TARGET_COLUMNS)
    known = set(MATCH_COLUMNS_BY_NAME) | set(ODDS_COLUMNS_BY_NAME) | set(ENGINEERED_FEATURE_COLUMNS_BY_NAME)

    violations: list[str] = []
    for column in feature_columns:
        if column in post_match:
            violations.append(f"{column!r} is only knowable after the final whistle (POST_MATCH)")
        elif column in targets:
            violations.append(f"{column!r} is the prediction target, not a feature (TARGET)")
        elif column in pre_kickoff and not allow_pre_kickoff:
            violations.append(
                f"{column!r} is only knowable in the minutes before kickoff (PRE_KICKOFF). "
                f"Pass allow_pre_kickoff=True only if the prediction is genuinely made at kickoff"
            )
        elif column not in known:
            violations.append(
                f"{column!r} is not a recognised canonical column; if it is provider-specific, "
                f"register it in schema.py with an explicit availability before using it"
            )

    if violations:
        listed = "\n  - ".join(violations)
        raise LeakageError(
            f"Refusing to build features for the {context}. "
            f"{len(violations)} problem(s) found:\n  - {listed}\n"
            f"Post-match columns: {', '.join(sorted(post_match))}"
        )


def canonical_dtypes() -> dict[str, str]:
    """Return the canonical dtype map for the match table.

    Returns:
        Mapping of column name to dtype string.
    """
    return {c.name: c.dtype for c in MATCH_COLUMNS}


def apply_dtypes(df: pd.DataFrame, *, table: str = "matches", keep_extra: bool = False) -> pd.DataFrame:
    """Coerce a DataFrame to the canonical dtypes for a table.

    Missing values become ``pd.NA`` for nullable columns. Columns absent from
    the input are added and left null, so a partial provider response still
    satisfies the schema.

    Args:
        df: Input frame.
        table: One of ``"matches"``, ``"events"``, ``"lineups"``, ``"odds"``.
        keep_extra: Retain columns that are not part of the canonical schema.
            Providers set this because they carry provider-specific extras
            (``statsbomb_match_id`` and friends) that later stages need for
            child-record lookups. Cleaning leaves it False so unknown columns
            do not leak into modelling.

    Returns:
        A new frame with canonical columns and dtypes.

    Raises:
        ValueError: If ``table`` is not a recognised table name.
    """
    specs: tuple[ColumnSpec, ...]
    if table == "matches":
        specs = MATCH_COLUMNS
    elif table == "events":
        specs = EVENT_COLUMNS
    elif table == "lineups":
        specs = LINEUP_COLUMNS
    elif table == "odds":
        specs = tuple(ODDS_COLUMNS_BY_NAME[name] for name in ODDS_COLUMNS)
    else:
        raise ValueError(f"Unknown table {table!r}. Expected one of: matches, events, lineups, odds.")

    known = {c.name for c in specs}
    result = df.copy()
    if not keep_extra:
        result = result.drop(columns=[c for c in result.columns if c not in known])

    for spec in specs:
        if spec.name not in result.columns:
            result[spec.name] = pd.NA
        try:
            if spec.dtype in {"Int64", "boolean"}:
                result[spec.name] = pd.to_numeric(result[spec.name], errors="coerce").astype(spec.dtype)
            elif spec.dtype == "datetime64[ns]":
                result[spec.name] = pd.to_datetime(result[spec.name], errors="coerce")
            elif spec.dtype == "string":
                result[spec.name] = result[spec.name].astype("string")
            else:
                result[spec.name] = pd.to_numeric(result[spec.name], errors="coerce").astype(spec.dtype)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Column {spec.name!r} could not be coerced to {spec.dtype}: {exc}"
            ) from exc

    ordered = [c.name for c in specs]
    extras = [c for c in result.columns if c not in known]
    return result[ordered + extras]


def build_empty_matches() -> pd.DataFrame:
    """Return an empty match frame with the correct columns and dtypes.

    Returns:
        A zero-row DataFrame matching the canonical match schema.
    """
    return apply_dtypes(pd.DataFrame({name: pd.Series(dtype="object") for name in MATCH_COLUMN_NAMES}))


def describe_schema() -> pd.DataFrame:
    """Return the match schema as a tidy table for documentation and the app.

    Returns:
        DataFrame with columns name, dtype, availability, nullable, description.
    """
    return pd.DataFrame(
        [
            {
                "column": c.name,
                "dtype": c.dtype,
                "availability": c.availability.value,
                "nullable": c.nullable,
                "description": c.description,
            }
            for c in MATCH_COLUMNS
        ]
    )


def validate_match_frame(df: pd.DataFrame, *, require_targets: bool = False) -> None:
    """Check a match frame against the canonical schema.

    Validates structure only. Statistical sanity (probabilities summing to one,
    and so on) belongs in the model tests, not the schema layer.

    Args:
        df: Frame to validate.
        require_targets: Also require goals and result to be present.

    Raises:
        ValueError: If a required column is missing, if the schema version is
            wrong, or if a result value is not H, D or A.
    """
    missing = [name for name in MATCH_COLUMN_NAMES if name not in df.columns]
    if missing:
        raise ValueError(f"Match frame is missing canonical columns: {missing}")

    if require_targets:
        incomplete = df[df["result"].isna()]
        if len(incomplete):
            raise ValueError(
                f"{len(incomplete)} row(s) have no result. These must be excluded from training "
                f"or backtesting. First offending match_id: {incomplete['match_id'].iloc[0]!r}"
            )

    invalid = sorted(set(df["result"].dropna().unique()) - {"H", "D", "A"})
    if invalid:
        raise ValueError(f"Unexpected result values {invalid}; expected only H, D or A.")

    bad_ids = df["match_id"].duplicated().sum()
    if bad_ids:
        raise ValueError(f"{bad_ids} duplicate match_id row(s) present; the identifier must be unique.")


def sort_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Sort a match frame chronologically, the order every model consumes.

    Args:
        df: Match frame.

    Returns:
        Frame sorted by match_date then match_id, with a reset index.
    """
    return df.sort_values(["match_date", "match_id"], kind="mergesort").reset_index(drop=True)


def feature_plan(columns: Sequence[str], *, context: str = "pre-match model") -> tuple[str, ...]:
    """Validate a feature list and return it as a tuple.

    Args:
        columns: Requested feature names.
        context: Description used in any error message.

    Returns:
        The validated columns.

    Raises:
        LeakageError: If the list is not pre-match safe.
    """
    assert_no_leakage(columns, context=context)
    return tuple(columns)

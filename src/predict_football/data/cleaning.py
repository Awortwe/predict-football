"""Normalisation and integrity checks applied after a provider returns data.

Providers map source fields to canonical names. This module handles everything
that must be true of the data *regardless* of source: dtypes, duplicate rows,
team-name consistency, result labels, and odds sanity.

The recurring theme is refusing to repair data silently. Where something is
wrong we either fix it deterministically or raise, and in both cases the problem
is logged. A dataset that looks plausible but is subtly wrong is the failure mode
that destroys a modelling project, and it is invisible without these checks.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from predict_football.data.identifiers import slugify
from predict_football.data.schema import apply_dtypes, sort_matches
from predict_football.data.teams import TeamResolver

logger = logging.getLogger(__name__)

#: Goals above this in a league match indicate a data error, not a football
#: result. Set high deliberately: we would rather flag a suspicious row than
#: silently accept it.
_IMPLAUSIBLE_GOALS = 20

#: Decimal odds outside this range are errors. 1.01 is a certainty and 1000.0
#: is a joke; nothing legitimate sits between.
_ODDS_MIN = 1.01
_ODDS_MAX = 1000.0


def clean_matches(df: pd.DataFrame, *, teams: TeamResolver | None = None) -> pd.DataFrame:
    """Normalise and validate a canonical match frame.

    Args:
        df: Raw canonical match frame from a provider.
        teams: Resolver used to flag rows whose team names are unrecognised.
            Optional; when omitted, name checking is skipped.

    Returns:
        Cleaned frame, chronologically sorted, with canonical dtypes.

    Raises:
        ValueError: If the frame has no usable rows, or if a match has the same
            team on both sides, or if a scoreline is implausible.
    """
    if df is None or df.empty:
        raise ValueError("clean_matches received an empty frame")

    out = df.copy()

    for column in ("home_team", "away_team", "league_key", "source"):
        if column in out.columns:
            out[column] = out[column].astype("string").str.strip()

    out = _drop_duplicate_matches(out)
    out = _fix_duplicate_fixtures(out)
    out = _coerce_outcomes(out)
    out = _reject_implausible_scores(out)
    out = apply_dtypes(out, table="matches")
    out = sort_matches(out)

    if teams is not None:
        out = _flag_unrecognised_teams(out, teams)

    return out


def _drop_duplicate_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows sharing a ``match_id``.

    Re-ingesting the same season should be idempotent, so duplicates collapse
    rather than accumulate. When duplicate rows disagree on the scoreline we
    keep the first and warn loudly, because silent disagreement usually means
    two sources are describing different fixtures under one identity.

    Args:
        df: Match frame.

    Returns:
        Frame with at most one row per ``match_id``.
    """
    if "match_id" not in df.columns:
        return df
    duplicated = df["match_id"].duplicated()
    if not duplicated.any():
        return df

    conflicting = df.loc[df["match_id"].isin(df.loc[duplicated, "match_id"])]
    conflict_ids = set(conflicting["match_id"]) if "home_goals" in conflicting.columns else set()
    if conflict_ids:
        sample = df[df["match_id"].isin(conflict_ids)]
        disagreements = sample.groupby("match_id")["home_goals"].nunique(dropna=True)
        genuine = disagreements[disagreements > 1]
        if len(genuine):
            logger.error(
                "%d match_id(s) have conflicting scorelines across rows. This usually means two "
                "different fixtures collide on league+date+teams. Examples: %s",
                len(genuine),
                list(genuine.index[:5]),
            )
    logger.warning("Dropping %d duplicate match_id row(s)", int(duplicated.sum()))
    return df.loc[~duplicated].copy()


def _fix_duplicate_fixtures(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse rows describing the same fixture from different sources.

    Args:
        df: Match frame.

    Returns:
        Frame with one row per fixture.
    """
    key = ["league_key", "match_date", "home_team", "away_team"]
    if not all(column in df.columns for column in key):
        return df
    duplicated = df.duplicated(subset=key, keep=False)
    if not duplicated.any():
        return df
    logger.warning(
        "%d row(s) describe fixtures already present from another source; keeping the first",
        int(duplicated.sum()),
    )
    return df.drop_duplicates(subset=key, keep="first").copy()


def _coerce_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Derive and validate the result label from the scoreline.

    The scoreline is authoritative. A published result label that disagrees with
    it is a data error, and trusting it would put a mislabelled row straight
    into the training set.

    Args:
        df: Match frame.

    Returns:
        Frame with a validated ``result`` column.
    """
    if "result" not in df.columns:
        return df

    out = df.copy()
    out["result"] = out["result"].astype("string").str.strip().str.upper()

    # Validate the label *before* deriving from the scoreline. Derivation
    # overwrites it, so an unrecognised label would otherwise be silently
    # hidden -- and an unrecognised label usually means the provider changed
    # its output format, which is exactly what we want to fail loudly on.
    unexpected = sorted(set(out["result"].dropna().unique()) - {"H", "D", "A"})
    if unexpected:
        raise ValueError(
            f"Unexpected result values {unexpected}; expected only H, D or A. "
            f"This usually means the provider changed its output format."
        )

    if {"home_goals", "away_goals"}.issubset(out.columns):
        home = pd.to_numeric(out["home_goals"], errors="coerce")
        away = pd.to_numeric(out["away_goals"], errors="coerce")
        derived = pd.Series(pd.NA, index=out.index, dtype="string")
        derived = derived.mask(home > away, "H").mask(home == away, "D").mask(home < away, "A")
        derived = derived.mask(home.isna() | away.isna(), pd.NA)

        conflict = derived.notna() & out["result"].notna() & (derived != out["result"])
        if conflict.any():
            logger.error(
                "%d row(s) have a result label contradicting their scoreline; the scoreline wins. "
                "Examples: %s",
                int(conflict.sum()),
                out.loc[conflict, "match_id"].head(3).tolist(),
            )
        out["result"] = derived.where(derived.notna(), out["result"])

    return out


def _reject_implausible_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose scoreline is impossible or whose teams cannot be identical.

    Args:
        df: Match frame.

    Returns:
        Frame without the offending rows.

    Raises:
        ValueError: If more than a fifth of rows have impossible scorelines,
            which indicates a systematic parsing problem rather than a few
            bad rows. Self-matches are dropped with a logged error instead,
            because they stem from a different kind of error.
    """
    if not {"home_goals", "away_goals"}.issubset(df.columns):
        return df

    out = df.copy()
    home = pd.to_numeric(out["home_goals"], errors="coerce")
    away = pd.to_numeric(out["away_goals"], errors="coerce")
    impossible = (home > _IMPLAUSIBLE_GOALS) | (away > _IMPLAUSIBLE_GOALS) | (home < 0) | (away < 0)

    if impossible.any():
        fraction = float(impossible.sum()) / len(out)
        if fraction > 0.20:
            raise ValueError(
                f"{int(impossible.sum())} of {len(out)} rows ({fraction:.1%}) have impossible "
                f"scorelines. That is a systematic parsing error, not a few bad rows. "
                f"Inspect the provider output before proceeding."
            )
        logger.error(
            "Dropping %d row(s) with impossible scorelines. Examples: %s",
            int(impossible.sum()),
            out.loc[impossible, ["match_date", "home_team", "away_team"]].head(3).to_dict("records"),
        )
        out = out.loc[~impossible].copy()

    # A team cannot play itself. Handled separately from the scoreline guard:
    # self-matches come from a broken join or identifier rather than a parsing
    # error, they are far rarer, and a batch of them is a different kind of
    # systemic problem to a batch of nonsense scorelines.
    self_match = (out["home_team"].astype("string") == out["away_team"].astype("string")).fillna(False)
    if self_match.any():
        logger.error(
            "Dropping %d self-match row(s), which indicate a broken join or identifier. Examples: %s",
            int(self_match.sum()),
            out.loc[self_match, "match_id"].head(3).tolist(),
        )
        out = out.loc[~self_match].copy()

    return out


def _flag_unrecognised_teams(df: pd.DataFrame, teams: TeamResolver) -> pd.DataFrame:
    """Record which team names are missing from the registry.

    Args:
        df: Match frame.
        teams: Team resolver.

    Returns:
        The frame with a ``teams_unresolved`` integer column added.
    """
    home_ok = [teams.is_known(x) if pd.notna(x) else False for x in df["home_team"]]
    away_ok = [teams.is_known(x) if pd.notna(x) else False for x in df["away_team"]]
    unresolved = int(home_ok.count(False) + away_ok.count(False))
    out = df.copy()
    out["teams_unresolved"] = [int(not h) + int(not a) for h, a in zip(home_ok, away_ok, strict=False)]
    if unresolved:
        names = sorted(
            {
                str(name)
                for column in ("home_team", "away_team")
                for name, ok in zip(df[column], home_ok if column == "home_team" else away_ok, strict=False)
                if pd.notna(name) and not ok
            }
        )
        logger.warning(
            "%d team reference(s) across %d row(s) are not in the registry: %s. "
            "They are stored as-is; add them to predict_football/data/teams.py to resolve.",
            unresolved,
            int((out["teams_unresolved"] > 0).sum()),
            names[:10],
        )
    return out


def clean_odds(df: pd.DataFrame, *, matches: pd.DataFrame | None = None) -> pd.DataFrame:
    """Validate and normalise an odds frame.

    Args:
        df: Long odds frame with ``match_id``, ``bookmaker`` and price columns.
        matches: Optional match frame, used to check referential integrity and
            to verify that an odds row is genuinely pre-match information.

    Returns:
        Cleaned odds frame.

    Raises:
        ValueError: If any odds value is outside the plausible decimal range,
            or if every row in a price set is invalid.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    price_columns = [c for c in out.columns if c.startswith("odds_")]
    for column in price_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    invalid = pd.Series(False, index=out.index)
    for column in price_columns:
        values = out[column]
        invalid |= values.notna() & ((values < _ODDS_MIN) | (values > _ODDS_MAX))
    if invalid.any():
        logger.warning(
            "Dropping %d odds row(s) containing prices outside [%s, %s]",
            int(invalid.sum()),
            _ODDS_MIN,
            _ODDS_MAX,
        )
        out = out.loc[~invalid].copy()

    if "match_id" in out.columns and matches is not None and "match_id" in matches.columns:
        known = set(matches["match_id"].dropna())
        orphans = ~out["match_id"].isin(known)
        if orphans.any():
            logger.warning(
                "Dropping %d odds row(s) whose match_id is not in the matches table",
                int(orphans.sum()),
            )
            out = out.loc[~orphans].copy()

    if "overround" in out.columns and len(out):
        present = out[["overround"]].notna().all(axis=1)
        weird = present & ((out["overround"] <= 0) | (out["overround"] > 5))
        if weird.any():
            logger.warning(
                "%d odds row(s) have an overround outside (0, 5]; those prices are probably misaligned",
                int(weird.sum()),
            )

    return out.reset_index(drop=True)


def clean_events(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and order a canonical event frame.

    Args:
        df: Raw canonical event frame.

    Returns:
        Cleaned frame ordered by period, minute, second.

    Raises:
        ValueError: If the frame lacks the ordering columns, which would make a
            replay feed impossible to trust.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    required = {"period", "minute", "second"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Event frame is missing ordering columns {sorted(missing)}")

    out = df.copy()
    for column in ("period", "minute", "second"):
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")
    # A provider may omit xG entirely. Build a real all-null float column rather
    # than passing None to pd.to_numeric, which raises instead of yielding nulls.
    out["xg"] = (
        pd.Series(np.nan, index=out.index, dtype="float64")
        if "xg" not in out.columns
        else pd.to_numeric(out["xg"], errors="coerce").astype("float64")
    )

    stamps = out[["period", "minute", "second"]]
    if stamps.isna().any(axis=None):
        logger.warning(
            "Dropping %d event(s) with a missing timestamp",
            int(stamps.isna().any(axis=1).sum()),
        )
        out = out.dropna(subset=["period", "minute", "second"])

    negative = (out["minute"] < 0) | (out["second"] < 0)
    if negative.any():
        logger.warning("Dropping %d event(s) with a negative timestamp", int(negative.sum()))
        out = out.loc[~negative].copy()

    if out["xg"].notna().any():
        # Parentheses are required: `|` binds tighter than comparison in Python,
        # so `xg < 0 | xg > 10` parses as `xg < (0 | xg) > 10`.
        absurd = (out["xg"] < 0) | (out["xg"] > 10)
        if absurd.any():
            logger.warning("Clipping %d event xG value(s) outside [0, 10]", int(absurd.sum()))
            out.loc[absurd, "xg"] = out.loc[absurd, "xg"].clip(0.0, 10.0)

    return out.sort_values(["period", "minute", "second"], kind="mergesort").reset_index(drop=True)


def match_coverage(df: pd.DataFrame, *, league_key: str, season: str) -> dict[str, object]:
    """Summarise what a match frame actually contains.

    Coverage must be stated explicitly wherever we report a result. A model
    trained on 380 of 380 matches is a different claim from one trained on 120,
    and users cannot tell the difference unless we say so.

    Args:
        df: Match frame.
        league_key: Competition key.
        season: Season label.

    Returns:
        Summary dictionary with counts, date span, and result distribution.
    """
    played = df[df["result"].notna()] if "result" in df.columns else df
    distribution = played["result"].value_counts().to_dict() if "result" in played.columns else {}

    return {
        "league_key": league_key,
        "season": season,
        "rows": len(df),
        "matches_with_result": len(played),
        "teams": len(
            set(df.get("home_team", pd.Series(dtype=str)).dropna())
            | set(df.get("away_team", pd.Series(dtype=str)).dropna())
        ),
        "first_match": str(df["match_date"].min()) if "match_date" in df.columns else None,
        "last_match": str(df["match_date"].max()) if "match_date" in df.columns else None,
        "home_wins": int(distribution.get("H", 0)),
        "draws": int(distribution.get("D", 0)),
        "away_wins": int(distribution.get("A", 0)),
        "has_xg": bool({"home_xg", "away_xg"}.issubset(df.columns) and df["home_xg"].notna().any()),
        "has_odds_inputs": bool("home_shots" in df.columns and df["home_shots"].notna().any()),
    }


def describe_unknown_teams(df: pd.DataFrame) -> list[str]:
    """List team names present in a frame but absent from the registry.

    Args:
        df: Match frame.

    Returns:
        Sorted unresolved team names.
    """
    resolver = TeamResolver()
    names: set[str] = set()
    for column in ("home_team", "away_team"):
        if column in df.columns:
            names.update(
                str(name) for name in df[column].dropna().unique() if not resolver.is_known(str(name))
            )
    return sorted(names)


def infer_league_from_names(names: list[str]) -> str | None:
    """Guess which competition a list of team names belongs to.

    A diagnostic helper for registry maintenance, never used in production code
    paths, because a guessed competition in a data pipeline is a silent bug.

    Args:
        names: Team names.

    Returns:
        The league key with the most matching teams, or ``None`` if nothing
        matches confidently.
    """
    from predict_football.config.leagues import LEAGUES

    resolver = TeamResolver()
    tally: dict[str, int] = {}
    for name in names:
        resolved = resolver.resolve(name)
        if not resolved:
            continue
        for team in resolver._by_key.values():
            if team.canonical_name == resolved:
                for key in team.league_keys:
                    tally[key] = tally.get(key, 0) + 1
    if not tally:
        return None
    best = max(tally, key=lambda k: tally[k])
    if tally[best] < 2:
        return None
    return best if best in LEAGUES else None


def safe_ratio(numerator: float, denominator: float, *, default: float = np.nan) -> float:
    """Divide without raising or warning on a zero denominator.

    Args:
        numerator: Dividend.
        denominator: Divisor.
        default: Value returned when the denominator is zero or missing.

    Returns:
        The ratio, or ``default``.
    """
    if denominator is None or (isinstance(denominator, float) and np.isnan(denominator)) or denominator == 0:
        return default
    return numerator / denominator


def team_key(value: str) -> str:
    """Return the comparable slug for a team name.

    Args:
        value: Team name.

    Returns:
        Slug used for matching.
    """
    return slugify(value)

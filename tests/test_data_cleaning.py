"""Identifier, team resolver and cleaning tests."""

from __future__ import annotations

import pandas as pd
import pytest

from predict_football.data.cleaning import (
    clean_events,
    clean_matches,
    clean_odds,
    describe_unknown_teams,
    match_coverage,
)
from predict_football.data.identifiers import (
    make_event_id,
    make_match_id,
    make_team_id,
    slugify,
)
from predict_football.data.teams import TeamResolver


def test_slugify_handles_accents_punctuation_and_spacing() -> None:
    """Slugs must normalise the spellings that differ between sources."""
    assert slugify("Atlético Madrid") == "atletico madrid"
    assert slugify("  FC   Schalke 04!  ") == "fc schalke 04"
    assert slugify("Borussia Mönchengladbach") == "borussia monchengladbach"
    assert slugify("Nott'm Forest") == "nott m forest"


def test_match_id_is_deterministic_and_order_sensitive() -> None:
    """The same fixture must always hash the same; swapping teams must change it."""
    first = make_match_id("ENG_PL", "2024-08-16", "Liverpool", "Arsenal")
    again = make_match_id("ENG_PL", "2024-08-16", "Liverpool", "Arsenal")
    swapped = make_match_id("ENG_PL", "2024-08-16", "Arsenal", "Liverpool")
    other_date = make_match_id("ENG_PL", "2024-08-17", "Liverpool", "Arsenal")
    assert first == again
    assert first != swapped, "home and away must be distinguished"
    assert first != other_date
    assert len(first) == 16


def test_match_id_ignores_team_name_spelling() -> None:
    """Provider spelling differences must not split one match into two rows.

    This is the property that lets us join football-data.co.uk to StatsBomb.
    The resolver is what buys it, so this exercises the pair: resolve the two
    spellings, then hash. Hashing raw provider names would not work, and
    ``make_match_id`` deliberately keeps no registry dependency.
    """
    resolver = TeamResolver()
    canonical = make_match_id(
        "ENG_PL", "2024-08-16",
        resolver.resolve("Manchester United"), resolver.resolve("Wolverhampton Wanderers"),
    )
    variant = make_match_id(
        "ENG_PL", "2024-08-16", resolver.resolve("Man United"), resolver.resolve("Wolves")
    )
    assert canonical == variant


def test_match_id_is_case_and_whitespace_insensitive_on_names() -> None:
    """Casing and stray whitespace are cosmetic, not identity."""
    assert make_match_id("ENG_PL", "2024-08-16", "Liverpool", "Arsenal") == make_match_id(
        "ENG_PL", "2024-08-16", "  liverpool ", "ARSENAL"
    )


def test_match_id_is_case_insensitive_on_league() -> None:
    """League key casing must not change identity."""
    assert make_match_id("eng_pl", "2024-08-16", "A", "B") == make_match_id("ENG_PL", "2024-08-16", "A", "B")


def test_event_id_is_stable_within_a_match() -> None:
    """Event identity must survive re-parsing the same source file."""
    first = make_event_id("abc123", 1, 45, 30, "Liverpool")
    again = make_event_id("abc123", 1, 45, 30, "Liverpool")
    later = make_event_id("abc123", 1, 46, 0, "Liverpool")
    assert first == again
    assert first != later


def test_team_resolver_maps_known_aliases() -> None:
    """Provider spellings must resolve to one canonical identity."""
    resolver = TeamResolver()
    assert resolver.resolve("Man United") == "Manchester United"
    assert resolver.resolve("Man Utd") == "Manchester United"
    assert resolver.resolve("Spurs") == "Tottenham Hotspur"
    assert resolver.resolve("Ath Bilbao") in {"Athletic Club", "Athletic Bilbao"}


def test_team_resolver_is_case_and_spacing_insensitive() -> None:
    """Whitespace and case must not defeat a lookup."""
    resolver = TeamResolver()
    assert resolver.resolve("  LIVERPOOL ") == "Liverpool"
    assert resolver.resolve("manchester   united") == "Manchester United"


def test_team_resolver_returns_none_for_unknown_names() -> None:
    """Unknown names must return None rather than a guess.

    A wrong merge is far harder to detect than a visible gap.
    """
    resolver = TeamResolver()
    assert resolver.resolve("Tottenham Widget Rovers") is None
    assert resolver.is_known("Tottenham Widget Rovers") is False


def test_team_resolver_pass_through_is_flagged() -> None:
    """Unrecognised names are stored as-is so the gap stays visible."""
    resolver = TeamResolver()
    assert resolver.resolve_or_pass_through("  New  Club ") == "New Club"
    assert resolver.is_known("New Club") is False


def test_team_resolver_does_not_merge_different_clubs() -> None:
    """Two clubs sharing a word must stay distinct."""
    resolver = TeamResolver()
    assert resolver.resolve("Sheffield United") == "Sheffield United"
    # "Sheffield Wednesday" was listed as an alias by mistake; confirm the two
    # Yorkshire clubs cannot be confused.
    assert resolver.resolve("Sheffield Weds") != resolver.resolve("Sheffield United")


def test_team_resolver_covers_every_football_data_co_abbreviation() -> None:
    """Every short form football-data.co.uk publishes must be in the registry.

    These names were found by ingesting a real season and reading the unresolved
    review queue. Left unregistered they are stored as separate teams, so
    "Real Sociedad" and "Sociedad" become two clubs and every feature built on
    team strength is wrong. Pinning them here means a future registry edit that
    drops one fails a test instead of quietly corrupting the data.
    """
    resolver = TeamResolver()
    expected = {
        "Almeria": "UD Almería",
        "Ath Madrid": "Atlético Madrid",
        "Cadiz": "Cádiz",
        "Celta": "Celta Vigo",
        "Granada": "Granada",
        "Las Palmas": "Las Palmas",
        "Sociedad": "Real Sociedad",
        "Vallecano": "Rayo Vallecano",
    }
    for raw, canonical in expected.items():
        assert resolver.resolve(raw) == canonical, f"{raw!r} no longer resolves to {canonical!r}"


def test_team_resolver_separates_a_reserve_side_from_its_parent() -> None:
    """Real Betis Balompié is Real Betis B and must not merge with Real Betis."""
    resolver = TeamResolver()
    assert resolver.resolve("Betis Balompie") == "Real Betis Balompié"
    assert resolver.resolve("Betis Balompie") != resolver.resolve("Betis")


def test_clean_matches_rejects_empty_frames() -> None:
    """An empty frame is a caller bug, not an empty dataset."""
    with pytest.raises(ValueError, match="empty frame"):
        clean_matches(pd.DataFrame())


def test_clean_matches_is_idempotent_on_duplicate_match_ids() -> None:
    """Re-ingesting a season must not duplicate rows."""
    frame = pd.DataFrame(
        {
            "match_id": ["m1", "m1", "m2"],
            "league_key": ["ENG_PL"] * 3,
            "match_date": pd.to_datetime(["2023-08-11", "2023-08-11", "2023-08-12"]),
            "home_team": ["Liverpool", "Liverpool", "Arsenal"],
            "away_team": ["Bournemouth", "Bournemouth", "Chelsea"],
            "home_goals": [1, 1, 2],
            "away_goals": [1, 1, 0],
            "result": ["D", "D", "H"],
            "source": ["test"] * 3,
        }
    )
    cleaned = clean_matches(frame)
    assert len(cleaned) == 2
    assert cleaned["match_id"].is_unique


def test_clean_matches_derives_result_from_scoreline(sample_matches) -> None:
    """The scoreline is authoritative over a published label."""
    frame = sample_matches.copy()
    frame.loc[0, "result"] = "H"
    cleaned = clean_matches(frame)
    row = cleaned[cleaned["match_id"] == "m1"].iloc[0]
    assert row["result"] == "D", "a label contradicting the scoreline must be overruled"


def test_clean_matches_rejects_invalid_result_values() -> None:
    """An unrecognised result label must raise, not be silently dropped."""
    frame = pd.DataFrame(
        {
            "match_id": ["m1"],
            "league_key": ["ENG_PL"],
            "match_date": pd.to_datetime(["2023-08-11"]),
            "home_team": ["Liverpool"],
            "away_team": ["Bournemouth"],
            "home_goals": [1],
            "away_goals": [1],
            "result": ["X"],
            "source": ["test"],
        }
    )
    with pytest.raises(ValueError, match="Unexpected result values"):
        clean_matches(frame)


def test_clean_matches_drops_self_matches() -> None:
    """A team cannot play itself; such a row is corrupt."""
    frame = pd.DataFrame(
        {
            "match_id": ["m1", "m2"],
            "league_key": ["ENG_PL"] * 2,
            "match_date": pd.to_datetime(["2023-08-11", "2023-08-12"]),
            "home_team": ["Liverpool", "Arsenal"],
            "away_team": ["Liverpool", "Chelsea"],
            "home_goals": [1, 2],
            "away_goals": [1, 0],
            "result": ["D", "H"],
            "source": ["test"] * 2,
        }
    )
    cleaned = clean_matches(frame)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["match_id"] == "m2"


def test_clean_matches_raises_on_systematically_implausible_scores() -> None:
    """Many bad rows mean a systematic error, so we stop rather than trim."""
    rows = {
        "match_id": [f"m{i}" for i in range(10)],
        "league_key": ["ENG_PL"] * 10,
        "match_date": pd.to_datetime([f"2023-08-{11 + i:02d}" for i in range(10)]),
        "home_team": ["Liverpool"] * 10,
        "away_team": [f"Team{i}" for i in range(10)],
        "home_goals": [999] * 10,
        "away_goals": [1] * 10,
        "result": ["H"] * 10,
        "source": ["test"] * 10,
    }
    with pytest.raises(ValueError, match="systematic parsing error"):
        clean_matches(pd.DataFrame(rows))


def test_clean_matches_flags_unresolved_teams(sample_matches, resolver) -> None:
    """Unrecognised team names must be counted, not hidden."""
    frame = sample_matches.copy()
    frame.loc[0, "away_team"] = "Nonexistent FC"
    cleaned = clean_matches(frame, teams=resolver)
    unresolved = describe_unknown_teams(cleaned)
    assert "Nonexistent FC" in unresolved


def test_clean_matches_sorts_chronologically(sample_matches) -> None:
    """Every model consumes matches in time order; enforce it here."""
    shuffled = sample_matches.iloc[::-1].reset_index(drop=True)
    cleaned = clean_matches(shuffled)
    dates = pd.to_datetime(cleaned["match_date"])
    assert dates.is_monotonic_increasing


def test_clean_matches_preserves_unplayed_fixtures(sample_matches) -> None:
    """The current season file contains future fixtures; they must survive."""
    cleaned = clean_matches(sample_matches)
    assert cleaned["result"].isna().sum() == 1, "the pending fixture should be retained with a null result"


def test_clean_odds_drops_impossible_prices() -> None:
    """Decimal odds below 1.01 or above 1000 are errors, not prices."""
    frame = pd.DataFrame(
        {
            # m2's home price is below 1.01 and m3's away price is above 1000.
            # Both are malformed, so m1 is the only row that should survive.
            "match_id": ["m1", "m2", "m3"],
            "bookmaker": ["avg"] * 3,
            "market": ["1x2"] * 3,
            "odds_home_close": [2.0, 0.5, 3.0],
            "odds_draw_close": [3.5, 3.5, 3.5],
            "odds_away_close": [4.0, 4.0, 4000.0],
        }
    )
    cleaned = clean_odds(frame)
    assert cleaned["match_id"].tolist() == ["m1"]


def test_clean_odds_drops_orphaned_rows() -> None:
    """Odds for unknown matches must not enter the store."""
    odds = pd.DataFrame(
        {
            "match_id": ["m1", "ghost"],
            "bookmaker": ["avg"] * 2,
            "market": ["1x2"] * 2,
            "odds_home_close": [2.0, 2.0],
            "odds_draw_close": [3.5, 3.5],
            "odds_away_close": [4.0, 4.0],
        }
    )
    matches = pd.DataFrame({"match_id": ["m1"]})
    cleaned = clean_odds(odds, matches=matches)
    assert list(cleaned["match_id"]) == ["m1"]


def test_clean_odds_handles_empty_input() -> None:
    """An empty odds frame is normal for providers without odds."""
    assert clean_odds(pd.DataFrame()).empty


def test_clean_events_requires_ordering_columns() -> None:
    """Without timestamps a replay feed cannot be trusted."""
    with pytest.raises(ValueError, match="ordering columns"):
        clean_events(pd.DataFrame({"event_type": ["Pass"]}))


def test_clean_events_orders_by_time() -> None:
    """Out-of-order source events must be sorted before replay."""
    frame = pd.DataFrame(
        {
            "period": [2, 1, 1],
            "minute": [10, 45, 2],
            "second": [0, 30, 5],
            "event_type": ["pass"] * 3,
        }
    )
    cleaned = clean_events(frame)
    assert list(zip(cleaned["period"], cleaned["minute"], strict=True)) == [(1, 2), (1, 45), (2, 10)]


def test_clean_events_clips_absurd_xg() -> None:
    """xG outside a plausible range is a parsing bug and must be contained."""
    frame = pd.DataFrame(
        {
            "period": [1, 1],
            "minute": [5, 10],
            "second": [0, 0],
            "event_type": ["shot"] * 2,
            "xg": [0.05, 42.0],
        }
    )
    cleaned = clean_events(frame)
    assert cleaned["xg"].max() <= 10.0


def test_match_coverage_reports_what_is_actually_there(sample_matches) -> None:
    """Coverage must be explicit so no result is overstated."""
    cleaned = clean_matches(sample_matches)
    coverage = match_coverage(cleaned, league_key="ENG_PL", season="2023/24")
    assert coverage["rows"] == 4
    assert coverage["matches_with_result"] == 3
    assert coverage["home_wins"] == 2
    assert coverage["draws"] == 1
    assert coverage["away_wins"] == 0
    assert coverage["has_xg"] is True
    assert coverage["teams"] >= 6


def test_team_id_is_derived_from_canonical_name() -> None:
    """Team ids must be stable and readable for debugging."""
    assert make_team_id("Manchester United") == "manchester_united"
    assert make_team_id("Tottenham Hotspur") == "tottenham_hotspur"

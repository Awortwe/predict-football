"""Configuration and licence policy tests."""

from __future__ import annotations

from datetime import date

import pytest

from predict_football.config.leagues import (
    LEAGUES,
    CompetitionType,
    get_league,
    list_leagues,
    season_codes,
    season_label,
)
from predict_football.config.licences import (
    LICENCES,
    REVIEWED_ON,
    LicenceViolation,
    assert_can_commit_raw_data,
    assert_can_serve_publicly,
    assert_commercial_use,
    attribution_for,
    licence_for,
    redistribution_safe_sources,
)
from predict_football.config.settings import Settings


def test_every_league_has_required_fields() -> None:
    """Every configured competition must be internally complete."""
    assert LEAGUES, "the league registry is empty"
    for key, league in LEAGUES.items():
        assert league.key == key, f"registry key {key} disagrees with League.key {league.key}"
        assert league.name, f"{key} has no name"
        assert isinstance(league.competition_type, CompetitionType)
        if league.tier is not None:
            assert league.tier >= 1, f"{key} has a nonsensical tier"
        if key.startswith("INT_"):
            assert league.tier is None, f"{key} is an international competition and should have no tier"


def test_league_keys_are_uppercase_and_stable() -> None:
    """Keys are primary keys in the database, so their format must be uniform."""
    for key in LEAGUES:
        assert key == key.upper()
        assert " " not in key, f"{key} contains a space; use underscores"


def test_get_league_is_case_insensitive_and_typed() -> None:
    """Lookups should be forgiving about case but strict about validity."""
    assert get_league("eng_pl").key == "ENG_PL"
    with pytest.raises(KeyError, match="Valid keys"):
        get_league("NOT_A_LEAGUE")


def test_list_leagues_filters() -> None:
    """Filters should combine and return only registry entries."""
    top_flight = list_leagues(tier=1)
    assert top_flight, "expected at least one tier-1 competition"
    assert all(x.tier == 1 for x in top_flight)

    cups = list_leagues(competition_type=CompetitionType.CUP)
    assert all(x.competition_type is CompetitionType.CUP for x in cups)
    assert any(x.key == "INT_WORLD_CUP" for x in cups)

    assert list_leagues(country="England")
    assert list_leagues(country="england"), "country filter should be case-insensitive"
    assert list_leagues(country="Atlantis") == []


def test_ucl_carries_an_explicit_feasibility_warning() -> None:
    """Champions League has no usable free data, and that must stay documented.

    If this note is ever removed, someone will trust a UCL model built from
    domestic form without being told what it actually is.
    """
    ucl = get_league("INT_UCL")
    assert "NO USABLE FREE DATA" in ucl.notes
    assert "domestic" in ucl.notes.lower()


def test_season_codes_round_trip() -> None:
    """Season code generation and labelling must be inverses."""
    assert season_codes(2023, 2025) == ["2324", "2425", "2526"]
    assert season_label("2425") == "2024/25"
    assert season_label("9900") == "1999/00"
    assert season_label("0001") == "2000/01"
    assert season_label("2526") == "2025/26"


def test_season_codes_reject_reversed_ranges() -> None:
    """A reversed range is a user error worth catching immediately."""
    with pytest.raises(ValueError, match="must be"):
        season_codes(2025, 2020)


def test_season_label_rejects_bad_input() -> None:
    """Malformed season codes should fail loudly."""
    for bad in ("abcd", "24", "24a5"):
        with pytest.raises(ValueError, match="four digits"):
            season_label(bad)


def test_season_label_rejects_a_bare_year() -> None:
    """A year is not a season code, and silently decoding it would be worse.

    "2024" would otherwise become "2020/24", a season that does not exist. The
    genuine 2020/21 code "2021" must still work, so the check cannot be a naive
    "starts with 20" test.
    """
    with pytest.raises(ValueError, match="not a valid season code"):
        season_label("2024")
    assert season_label("2021") == "2020/21"


def test_season_label_wraps_at_the_century_boundary() -> None:
    """1999/00 is encoded 9900, and the wrap is valid rather than malformed."""
    assert season_label("9900") == "1999/00"
    assert season_label("0001") == "2000/01"


def test_settings_derive_every_path_from_the_project_root(tmp_path) -> None:
    """All managed paths must hang off the project root, not the cwd."""
    resolved = Settings.from_env(project_root=tmp_path)
    for path in (
        resolved.raw_dir,
        resolved.processed_dir,
        resolved.cache_dir,
        resolved.samples_dir,
        resolved.artifacts_dir,
        resolved.reports_dir,
    ):
        assert tmp_path in path.parents or path.parent == tmp_path
    assert resolved.database_path.parent == resolved.data_dir
    assert resolved.seed == 42


def test_settings_ensure_directories_is_idempotent(tmp_path) -> None:
    """Directory creation must be safe to call repeatedly."""
    resolved = Settings.from_env(project_root=tmp_path)
    resolved.ensure_directories()
    resolved.ensure_directories()
    assert resolved.data_dir.is_dir()
    assert resolved.raw_dir.is_dir()


def test_network_access_is_off_unless_opted_in(tmp_path, monkeypatch) -> None:
    """The documented default is offline, so a scheduled run cannot reach out."""
    monkeypatch.delenv("PREDICT_FOOTBALL_ALLOW_NETWORK", raising=False)
    assert Settings.from_env(project_root=tmp_path).allow_network is False
    monkeypatch.setenv("PREDICT_FOOTBALL_ALLOW_NETWORK", "1")
    assert Settings.from_env(project_root=tmp_path).allow_network is True
    monkeypatch.setenv("PREDICT_FOOTBALL_ALLOW_NETWORK", "off")
    assert Settings.from_env(project_root=tmp_path).allow_network is False


def test_every_reviewed_source_has_a_complete_licence_record() -> None:
    """A licence record missing its terms URL would be unusable in practice."""
    assert LICENCES
    for name, policy in LICENCES.items():
        assert policy.url.startswith("http"), f"{name} has no terms URL"
        assert policy.summary.strip(), f"{name} has an empty summary"
        assert isinstance(policy.may_commit_raw_data, bool)
        assert isinstance(policy.may_serve_from_public_app, bool)
        assert isinstance(policy.may_use_commercially, bool)


def test_restrictive_licences_block_the_actions_they_forbid() -> None:
    """football-data.co.uk and StatsBomb must fail the shipping checks."""
    for source in ("football_data_co", "statsbomb_open"):
        with pytest.raises(LicenceViolation, match="may not be committed"):
            assert_can_commit_raw_data(source)
        with pytest.raises(LicenceViolation, match="public app may not serve"):
            assert_can_serve_publicly(source)
        with pytest.raises(LicenceViolation, match="commercial use is not permitted"):
            assert_commercial_use(source)


def test_permissive_licences_pass_every_check() -> None:
    """openfootball and Wyscout are the licence-clean paths and must stay usable."""
    assert_can_commit_raw_data("openfootball")
    assert_can_serve_publicly("openfootball")
    assert_commercial_use("openfootball")
    assert_can_commit_raw_data("wyscout_open")
    assert_can_serve_publicly("wyscout_open")
    assert_commercial_use("wyscout_open")


def test_unreviewed_source_is_an_error_not_a_permissive_default() -> None:
    """Assuming permission for an unreviewed source is the mistake this prevents."""
    with pytest.raises(KeyError, match="No licence review"):
        licence_for("some_random_scraper")


def test_statsbomb_policy_records_the_license_requirement() -> None:
    """The StatsBomb logo obligation is easy to forget and legally binding."""
    policy = licence_for("statsbomb_open")
    assert "logo" in policy.summary.lower()
    assert "logo" in (policy.attribution or "").lower()
    assert "1.2.1" in policy.summary and "1.2.2" in policy.summary


def test_licence_review_date_is_recorded() -> None:
    """Licence findings go stale; the review date must be visible in code."""
    assert isinstance(REVIEWED_ON, date)
    assert date.today() >= REVIEWED_ON


def test_redistribution_safe_list_is_correct() -> None:
    """The helper used to pick a shippable source must match the policies."""
    safe = redistribution_safe_sources()
    assert "openfootball" in safe
    assert "wyscout_open" in safe
    assert "football_data_co" not in safe
    assert "statsbomb_open" not in safe


def test_attribution_is_available_for_shippable_sources() -> None:
    """Anything safe to ship must carry attribution text for published output."""
    for source in redistribution_safe_sources():
        assert not attribution_for(source).endswith("confirm terms."), f"{source} lacks attribution text"

"""Headless tests for the Streamlit app.

These exercise the app's UI behaviour -- that it renders, that pressing the run
button produces forecasts and diagnostics, and that it degrades gracefully when
the database is empty. They do not re-test the modelling logic, which has its own
unit tests in ``test_inference.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from predict_football.data.repository import Database, MatchRepository

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "streamlit_app.py"


def _synthetic_league(n_teams: int = 24, rounds: int = 36) -> pd.DataFrame:
    """Build a completed synthetic league plus one pending fixture.

    Args:
        n_teams: Number of clubs.
        rounds: Number of weekly matchdays.

    Returns:
        Canonical match frame with results, followed by one unplayed fixture
        dated after the last matchday.
    """
    rng = np.random.default_rng(7)
    strengths = np.linspace(0.8, 2.4, n_teams)
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
                    "source": "test",
                    "league_key": "ENG_PL",
                    "competition_type": "league",
                    "season": "2021/22",
                    "season_code": "2122",
                    "match_date": pd.Timestamp("2021-01-01") + pd.Timedelta(days=7 * round_index),
                    "home_team": teams[a],
                    "away_team": teams[b],
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "result": result,
                }
            )
    rows.append(
        {
            "match_id": "pending",
            "source": "test",
            "league_key": "ENG_PL",
            "competition_type": "league",
            "season": "2021/22",
            "season_code": "2122",
            "match_date": pd.Timestamp("2021-09-17"),
            "home_team": teams[0],
            "away_team": teams[1],
            "home_goals": None,
            "away_goals": None,
            "result": None,
        }
    )
    return pd.DataFrame(rows)


def _seed(database_path: Path) -> None:
    """Write the synthetic league into a fresh SQLite database.

    Args:
        database_path: Location of the database file to create.
    """
    repository = MatchRepository(Database(database_path))
    repository.upsert_matches(_synthetic_league())


def test_app_runs_forecast_and_diagnostics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pressing run renders forecasts, sample sizes and benchmark metrics."""
    _seed(tmp_path / "predict_football.sqlite")
    monkeypatch.setenv("PREDICT_FOOTBALL_DATA_DIR", str(tmp_path))

    at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "predict_football"

    at.button(key="run_button").click().run(timeout=60)
    assert not at.exception

    # Three headline metrics: Brier, log loss, accuracy.
    assert len(at.metric) >= 3
    # The fixtures table and the model/base-rate/market comparison are present.
    assert len(at.dataframe) >= 1
    captions = " ".join(node.value for node in at.caption)
    assert "out-of-sample" in captions
    # The challenger is benchmarked beside the baseline on the same matches.
    labels: set[str] = set()
    for node in at.dataframe:
        frame = node.value
        if hasattr(frame, "columns") and "label" in frame.columns:
            labels.update(str(value) for value in frame["label"])
    assert {"Dixon-Coles", "LightGBM", "Base rate"} <= labels


def test_app_handles_empty_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no stored matches the app explains how to fetch data instead of crashing."""
    monkeypatch.setenv("PREDICT_FOOTBALL_DATA_DIR", str(tmp_path))

    at = AppTest.from_file(str(APP_PATH), default_timeout=30).run()

    assert not at.exception
    assert at.warning
    assert "No matches are stored" in at.warning[0].value


def test_app_explains_a_public_deploy_without_an_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A public deployment with no token says what to configure, without a traceback."""
    monkeypatch.setenv("PREDICT_FOOTBALL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PREDICT_FOOTBALL_PUBLIC_DEPLOY", "1")
    monkeypatch.setenv("PREDICT_FOOTBALL_ALLOW_NETWORK", "1")
    monkeypatch.setenv("FOOTBALL_DATA_ORG_API_KEY", "")

    at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()

    assert not at.exception
    warnings = " ".join(node.value for node in at.warning).lower()
    assert "could not load its match data" in warnings
    infos = " ".join(node.value for node in at.info)
    assert "FOOTBALL_DATA_ORG_API_KEY" in infos


def test_app_reports_missing_fixtures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A competition whose matches are all complete still renders diagnostics."""
    frame = _synthetic_league()
    frame = frame[frame["result"].notna()]
    repository = MatchRepository(Database(tmp_path / "predict_football.sqlite"))
    repository.upsert_matches(frame)
    monkeypatch.setenv("PREDICT_FOOTBALL_DATA_DIR", str(tmp_path))

    at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    at.button(key="run_button").click().run(timeout=60)

    assert not at.exception
    warnings = " ".join(node.value for node in at.warning)
    assert "already has a result" in warnings


def test_app_predicts_a_matchday(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Selecting a date forecasts the stored fixtures on it with charts."""
    _seed(tmp_path / "predict_football.sqlite")
    monkeypatch.setenv("PREDICT_FOOTBALL_DATA_DIR", str(tmp_path))

    at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    assert not at.exception

    league = "ENG_PL"
    at.date_input(key=f"day_date_{league}").set_value(pd.Timestamp("2021-09-17").date()).run(timeout=60)
    assert not at.exception

    labels = {node.label for node in at.metric}
    assert {"Home win", "Draw", "Away win"} <= labels
    captions = " ".join(node.value for node in at.caption)
    assert "Trained on" in captions


def test_app_lists_every_match_on_a_selected_day(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The predictor surfaces every stored fixture for the chosen date."""
    _seed(tmp_path / "predict_football.sqlite")
    monkeypatch.setenv("PREDICT_FOOTBALL_DATA_DIR", str(tmp_path))

    at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    league = "ENG_PL"
    at.date_input(key=f"day_date_{league}").set_value(pd.Timestamp("2021-01-08").date()).run(timeout=60)

    assert not at.exception
    options = list(at.selectbox(key=f"day_match_{league}").options)
    assert len(options) == 12


def test_app_reports_a_date_without_matches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A date with no stored fixture says so instead of showing an empty forecast."""
    _seed(tmp_path / "predict_football.sqlite")
    monkeypatch.setenv("PREDICT_FOOTBALL_DATA_DIR", str(tmp_path))

    at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    league = "ENG_PL"
    at.date_input(key=f"day_date_{league}").set_value(pd.Timestamp("2021-06-02").date()).run()

    assert not at.exception
    infos = " ".join(node.value for node in at.info)
    assert "No stored matches" in infos

"""Tests for hyperparameter selection.

These tests do not assert which configuration is best -- that is an empirical
question about real football, not a property of the code. They assert the
mechanics: one row per grid point, every score attached to its sample size,
consistent coverage across the grid, and a stable sort so "best" is meaningful.
"""

from __future__ import annotations

import pandas as pd
import pytest

from predict_football.tuning import best_configuration, explain_winner, sweep_dixon_coles


def test_sweep_returns_one_row_per_combination(synthetic_league: pd.DataFrame) -> None:
    """A 2x2 grid must produce exactly four scored rows."""
    table = sweep_dixon_coles(
        synthetic_league,
        half_lives=(None, 30.0),
        ridges=(0.05, 0.2),
        min_train_matches=20,
        refit_every_dates=3,
    )

    assert len(table) == 4
    combos = {
        (None if pd.isna(half_life) else float(half_life), float(ridge))
        for half_life, ridge in zip(table["half_life_matches"], table["ridge"], strict=True)
    }
    assert combos == {
        (None, 0.05),
        (None, 0.2),
        (30.0, 0.05),
        (30.0, 0.2),
    }


def test_sweep_scores_are_sorted_and_bounded(synthetic_league: pd.DataFrame) -> None:
    """Brier must ascend (best first) and lie in the valid [0, 2] range."""
    table = sweep_dixon_coles(
        synthetic_league,
        half_lives=(None, 30.0),
        ridges=(0.05,),
        min_train_matches=20,
        refit_every_dates=3,
    )

    assert table["brier"].is_monotonic_increasing
    assert ((table["brier"] >= 0) & (table["brier"] <= 2)).all()
    assert (table["n_matches"] > 0).all()


def test_sweep_uses_the_same_matches_for_every_configuration(synthetic_league: pd.DataFrame) -> None:
    """Comparing configurations is only honest if they cover the same matches."""
    table = sweep_dixon_coles(
        synthetic_league,
        half_lives=(None, 30.0),
        ridges=(0.05, 0.2),
        min_train_matches=20,
        refit_every_dates=3,
    )

    assert table["n_matches"].nunique() == 1


def test_best_configuration_is_the_top_row(synthetic_league: pd.DataFrame) -> None:
    """`best` must be the first row, since the table is sorted by Brier."""
    table = sweep_dixon_coles(
        synthetic_league,
        half_lives=(None, 30.0),
        ridges=(0.05,),
        min_train_matches=20,
        refit_every_dates=3,
    )

    assert best_configuration(table)["brier"] == table.iloc[0]["brier"]


def test_explain_winner_mentions_the_gap(synthetic_league: pd.DataFrame) -> None:
    """The summary must state the runner-up gap, not sell a decisive win."""
    table = sweep_dixon_coles(
        synthetic_league,
        half_lives=(None, 30.0),
        ridges=(0.05,),
        min_train_matches=20,
        refit_every_dates=3,
    )

    summary = explain_winner(table)
    assert "best:" in summary
    assert "runner-up" in summary
    assert "gap" in summary


def test_empty_grid_is_rejected(synthetic_league: pd.DataFrame) -> None:
    """An empty grid is a caller error, not an empty result."""
    with pytest.raises(ValueError, match="grid is empty"):
        sweep_dixon_coles(synthetic_league, half_lives=(), ridges=(0.05,))


def test_empty_table_is_rejected() -> None:
    """Asking for the best of nothing must raise."""
    with pytest.raises(ValueError, match="sweep table is empty"):
        best_configuration(pd.DataFrame())

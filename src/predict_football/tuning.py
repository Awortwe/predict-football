"""Hyperparameter selection by out-of-sample walk-forward score.

Tuning a football model on the same matches it is later scored on is one of the
easiest ways to produce a number that is meaningless. Every candidate here is
therefore judged with :func:`walk_forward_backtest`: it is fitted only on earlier
matches and scored only on later ones, exactly like the final model. The chosen
configuration is the one with the best Brier score, but the full table is
returned so a reader can see how close the runner-up was and whether an edge of
the grid won.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable

import pandas as pd

from predict_football.backtest import walk_forward_backtest
from predict_football.evaluation import evaluate
from predict_football.models.dixon_coles import PROBABILITY_COLUMNS, DixonColesModel

module_logger = logging.getLogger(__name__)

#: A deliberately small default grid. Recency weighting is the hyperparameter
#: that matters most for football form, so it gets the widest sweep; the ridge
#: penalty mainly stabilises promoted clubs with few matches.
DEFAULT_HALF_LIVES: tuple[float | None, ...] = (None, 60.0, 120.0, 240.0)
DEFAULT_RIDGES: tuple[float, ...] = (0.02, 0.1)


def sweep_dixon_coles(
    matches: pd.DataFrame,
    *,
    half_lives: Iterable[float | None] = DEFAULT_HALF_LIVES,
    ridges: Iterable[float] = DEFAULT_RIDGES,
    min_train_matches: int = 380,
    refit_every_dates: int = 5,
    progress: Callable[[str, int, int], None] | None = None,
) -> pd.DataFrame:
    """Score every hyperparameter combination out of sample.

    Args:
        matches: Canonical match frame with results.
        half_lives: Candidate recency half-lives in matches; ``None`` disables
            weighting.
        ridges: Candidate ridge penalties on the team ratings.
        min_train_matches: Warm-up size, forwarded to the backtest.
        refit_every_dates: Refit cadence, forwarded to the backtest.
        progress: Optional callable ``(combination_label, position, total)`` used
            by the CLI to show which combination is running.

    Returns:
        One row per combination with the sample size and the three headline
        metrics, sorted by Brier score ascending (best first).

    Raises:
        ValueError: If the grid is empty.
    """
    half_life_options = list(half_lives)
    ridge_options = list(ridges)
    combinations = [(half_life, ridge) for half_life in half_life_options for ridge in ridge_options]
    if not combinations:
        raise ValueError("sweep_dixon_coles: the hyperparameter grid is empty")

    rows: list[dict[str, object]] = []
    for position, (half_life, ridge) in enumerate(combinations, start=1):
        label = f"half_life={half_life}, ridge={ridge}"
        if callable(progress):
            progress(label, position, len(combinations))

        predictions = walk_forward_backtest(
            matches,
            model_factory=lambda h=half_life, r=ridge: DixonColesModel(ridge=r, half_life_matches=h),
            min_train_matches=min_train_matches,
            refit_every_dates=refit_every_dates,
        )
        report = evaluate(predictions["outcome"], predictions[list(PROBABILITY_COLUMNS)], label=label)

        rows.append(
            {
                "half_life_matches": half_life,
                "ridge": ridge,
                "n_matches": report.n_matches,
                "brier": round(report.brier, 4),
                "log_loss": round(report.log_loss, 4),
                "accuracy": round(report.accuracy, 4),
            }
        )
        module_logger.info("%s -> Brier %.4f over %d matches", label, report.brier, report.n_matches)

    return pd.DataFrame(rows).sort_values(["brier", "log_loss"], ignore_index=True)


def best_configuration(table: pd.DataFrame) -> dict[str, object]:
    """Return the top row of a sweep as a plain dict.

    Args:
        table: Output of :func:`sweep_dixon_coles`.

    Returns:
        The best configuration with its scores.

    Raises:
        ValueError: If the table is empty.
    """
    if table.empty:
        raise ValueError("best_configuration: the sweep table is empty")
    return table.iloc[0].to_dict()


def explain_winner(table: pd.DataFrame) -> str:
    """Summarise the sweep in one line, including the runner-up gap.

    Args:
        table: Output of :func:`sweep_dixon_coles`.

    Returns:
        Human-readable summary. Stating the gap matters because a win by 0.001
        Brier is noise, and the reader should be told that rather than sold a
        decisive result.
    """
    best = best_configuration(table)
    line = (
        f"best: half_life={best['half_life_matches']}, ridge={best['ridge']} "
        f"-> Brier {best['brier']:.4f} over {best['n_matches']} matches"
    )
    if len(table) > 1:
        runner_up = table.iloc[1]
        line += f"; runner-up Brier {runner_up['brier']:.4f} (gap {runner_up['brier'] - best['brier']:+.4f})"
    return line


__all__ = [
    "DEFAULT_HALF_LIVES",
    "DEFAULT_RIDGES",
    "best_configuration",
    "explain_winner",
    "sweep_dixon_coles",
]

"""Chronological backtesting.

The whole purpose of this module is that a model must never see a match it is
asked to predict, or any match that came after it. That is enforced structurally
rather than by convention:

* Folds are cut on **dates**, never on rows, so every match on a given matchday is
  predicted by the same model trained on matches from strictly earlier dates.
* Every training set is verified to end before its test set begins. If that check
  ever fails the backtest aborts instead of returning a plausible number.
* A single fold's training data is all matches up to that date, so the walk is
  expanding rather than rolling: early predictions are made from less evidence,
  which is the honest depiction of what a live deployment would have had.

Random splitting is not offered. It would produce a better-looking score and a
meaningless one.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from predict_football.models.base import PredictiveModel
from predict_football.models.dixon_coles import PROBABILITY_COLUMNS

module_logger = logging.getLogger(__name__)

#: Columns a fold must be able to identify a fixture by.
_FIXTURE_COLUMNS = ("match_id", "match_date", "home_team", "away_team")


class BacktestError(ValueError):
    """Raised when a backtest cannot be run as specified."""


@dataclass(frozen=True)
class Fold:
    """One train/test split of the timeline.

    Attributes:
        index: Zero-based position of the fold in the walk.
        train_end: Last date in the training set.
        test_start: First date in the test set.
        n_train: Matches used for fitting.
        n_test: Matches predicted.
    """

    index: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    n_train: int
    n_test: int

    def describe(self) -> str:
        """Render the fold boundaries.

        Returns:
            Human-readable summary of the split.
        """
        return (
            f"fold {self.index}: train {self.n_train} matches to {self.train_end.date()}, "
            f"test {self.n_test} matches from {self.test_start.date()}"
        )


def _chronological(matches: pd.DataFrame) -> pd.DataFrame:
    """Return the frame with real datetimes, ordered oldest first.

    Dates arrive from the database as ISO strings. Comparing those to a
    ``Timestamp`` silently raises rather than sorting, and a string sort would
    be wrong anyway for any non-ISO format. Coercion happens once, here, so the
    fold arithmetic can rely on the dtype.

    Args:
        matches: Canonical match frame.

    Returns:
        Copy of the frame with ``match_date`` as ``datetime64`` and rows ordered
        by date then match id.

    Raises:
        BacktestError: If ``match_date`` cannot be interpreted as a date, which
            means the stored data is corrupt and should not be guessed at.
    """
    missing = set(_FIXTURE_COLUMNS) - set(matches.columns)
    if missing:
        raise BacktestError(f"frame is missing {sorted(missing)}")

    frame = matches.copy()
    frame["match_date"] = pd.to_datetime(frame["match_date"], errors="coerce")
    if frame["match_date"].isna().any():
        unparsed = int(frame["match_date"].isna().sum())
        raise BacktestError(f"could not interpret the match date on {unparsed} row(s)")

    return frame.sort_values(["match_date", "match_id"], kind="mergesort")


def walk_forward_folds(
    matches: pd.DataFrame,
    min_train_matches: int = 380,
    refit_every_dates: int = 1,
) -> list[Fold]:
    """Split a timeline into expanding-window folds.

    Args:
        matches: Canonical match frame with a result and a ``match_date``.
        min_train_matches: Matches required before the first prediction. This is
            the warm-up, and it is why a backtest covers fewer matches than the
            dataset: the first ``min_train_matches`` are evidence, not forecasts.
        refit_every_dates: Refit after this many distinct prediction dates. A
            value of one refits on every matchday, which is what a live system
            would do and is the most defensible choice; larger values are faster
            and slightly stale.

    Returns:
        List of folds in chronological order.

    Raises:
        BacktestError: If required columns are missing, no dates are available,
            or the frame holds too few matches to ever predict out of sample.
    """
    frame = _chronological(matches)
    if frame.empty:
        raise BacktestError("walk_forward_folds: no rows have a date")

    dates = sorted(pd.Timestamp(d) for d in frame["match_date"].unique())
    if len(frame) <= min_train_matches:
        raise BacktestError(
            f"walk_forward_folds: {len(frame)} matches cannot support a warm-up of {min_train_matches}"
        )

    folds: list[Fold] = []
    batch: list[pd.Timestamp] = []

    def flush(dates_in_batch: list[pd.Timestamp]) -> None:
        """Turn a group of prediction dates into a fold.

        Args:
            dates_in_batch: Distinct dates to predict with one fitted model.
        """
        train = frame[frame["match_date"] < dates_in_batch[0]]
        test = frame[frame["match_date"].isin(dates_in_batch)]
        folds.append(
            Fold(
                index=len(folds),
                train_end=pd.Timestamp(train["match_date"].max()),
                test_start=pd.Timestamp(dates_in_batch[0]),
                n_train=len(train),
                n_test=len(test),
            )
        )

    for test_date in dates:
        if len(frame[frame["match_date"] < test_date]) < min_train_matches:
            continue
        batch.append(test_date)
        if len(batch) >= refit_every_dates:
            flush(batch)
            batch = []

    # Flush the trailing partial batch. Without this, asking to refit every N
    # dates silently drops the final 1..N-1 matchdays of the season, which is
    # exactly the kind of quiet coverage loss this project is meant to prevent.
    if batch:
        flush(batch)

    return folds


def walk_forward_backtest(
    matches: pd.DataFrame,
    model_factory: Callable[[], PredictiveModel],
    min_train_matches: int = 380,
    refit_every_dates: int = 1,
    progress: Callable[[Fold, int], None] | None = None,
) -> pd.DataFrame:
    """Predict every eligible match out of sample, in date order.

    Args:
        matches: Canonical match frame with results.
        model_factory: Callable returning an unfitted model with a ``fit`` and
            ``outcome_probabilities`` method. Called once per fold so no state is
            shared between folds.
        min_train_matches: Warm-up size before the first prediction.
        refit_every_dates: Distinct prediction dates per fitted model.
        progress: Optional callback receiving each fold and its one-based
            position, for reporting on long runs.

    Returns:
        Frame of out-of-sample predictions, one row per match, carrying the
        fixture identifiers, the observed outcome and the forecast probabilities.

    Raises:
        BacktestError: If no folds could be produced, or if a fold's training set
            is not strictly earlier than its test set.
    """
    folds = walk_forward_folds(
        matches,
        min_train_matches=min_train_matches,
        refit_every_dates=refit_every_dates,
    )
    if not folds:
        raise BacktestError(
            "walk_forward_backtest: no fold qualified; lower min_train_matches or supply more matches"
        )

    frame = _chronological(matches).dropna(subset=["result"])

    predictions: list[pd.DataFrame] = []
    for position, fold in enumerate(folds, start=1):
        test_dates = frame.loc[frame["match_date"] >= fold.test_start, "match_date"]
        boundary = fold.test_start
        if not test_dates.empty:
            # Reconstruct the exact test slice for this fold: the dates from its
            # start up to the next fold's start.
            next_start = folds[position].test_start if position < len(folds) else None
            mask = frame["match_date"] >= fold.test_start
            if next_start is not None:
                mask &= frame["match_date"] < next_start
            test = frame[mask]
        else:  # pragma: no cover - defensive
            continue

        if not test.empty and pd.Timestamp(test["match_date"].min()) <= fold.train_end:
            raise BacktestError(
                f"walk_forward_backtest: fold {fold.index} would train on matches at or after its own "
                f"test date; refusing to produce a leaked result"
            )

        train = frame[frame["match_date"] < boundary]
        model = model_factory()
        model.fit(train)

        predicted = model.outcome_probabilities(test)
        predicted["outcome"] = test["result"].to_numpy()
        predicted["n_train_matches"] = len(train)
        predictions.append(predicted)

        if progress is not None:
            progress(fold, position)
        module_logger.debug("%s", fold.describe())

    if not predictions:
        raise BacktestError("walk_forward_backtest: folds produced no predictions")

    return pd.concat(predictions, ignore_index=True)


def verify_no_overlap(predictions: pd.DataFrame, matches: pd.DataFrame) -> None:
    """Assert that no predicted match was used to train its own model.

    Args:
        predictions: Output of :func:`walk_forward_backtest`.
        matches: The frame the backtest ran on.

    Raises:
        BacktestError: If a predicted match appears in the frame without a
            training-size annotation, or if any fold reported fewer training
            matches than the number of earlier matches in the dataset.
    """
    required = {"match_id", "n_train_matches", *PROBABILITY_COLUMNS}
    missing = required - set(predictions.columns)
    if missing:
        raise BacktestError(f"verify_no_overlap: predictions are missing {sorted(missing)}")

    dated = _chronological(matches)
    # Position of every match in the full chronology, which is the largest
    # training size any legitimate fold could have used for a given match.
    position = {match_id: rank for rank, match_id in enumerate(dated["match_id"])}

    for row in predictions.itertuples():
        rank = position.get(row.match_id)
        if rank is None:
            raise BacktestError(f"verify_no_overlap: predicted match {row.match_id} is not in the dataset")
        if row.n_train_matches > rank:
            raise BacktestError(
                f"verify_no_overlap: match {row.match_id} was predicted by a model trained on "
                f"{row.n_train_matches} matches but only {rank} precede it in the timeline"
            )


__all__ = ["BacktestError", "Fold", "verify_no_overlap", "walk_forward_backtest", "walk_forward_folds"]

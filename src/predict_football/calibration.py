"""Post-hoc recalibration of match-outcome probabilities.

A model can rank fixtures well and still be mis-scaled: it may say 0.70 when the
true frequency is 0.55. That is a calibration defect, and it is exactly what a
Brier score or log loss punishes. This module learns a correction from forecasts
whose matches have already finished and applies it to later forecasts.

The method is multinomial logistic regression on the log of the forecast
probabilities ("vector scaling", the multiclass form of Platt scaling). It is a
strict generalisation of temperature scaling: when forecasts are already
calibrated the learned transform is close to the identity, and when they are
over- or under-confident it sharpens or softens them. A feature is never added
here -- only the three existing probabilities are remapped -- so the correction
cannot smuggle in information the model did not have.

Honesty rules, enforced by :func:`walk_forward_calibrate`:

* The calibrator for a given matchday is fitted only on forecasts from
  **strictly earlier** matchdays, so it never sees the outcomes it is asked to
  correct. This is the same discipline the backtest applies to the model.
* Rows before enough calibration history exists are left untouched and flagged
  ``is_calibrated == False``, so a report can say plainly how many predictions
  the correction actually covered.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from predict_football.evaluation import _as_matrix, outcome_index
from predict_football.models.dixon_coles import OUTCOMES, PROBABILITY_COLUMNS

module_logger = logging.getLogger(__name__)

#: Floor applied before taking a logarithm of a probability. A forecast of
#: exactly zero would otherwise produce negative infinity.
CALIBRATION_EPSILON = 1e-6

#: Suffix appended to each probability column by the walk-forward calibrator.
CALIBRATED_SUFFIX = "_calibrated"


class CalibrationError(ValueError):
    """Raised when a calibrator cannot be fitted or applied as supplied."""


class ProbabilityCalibrator:
    """Map forecast probabilities onto calibrated ones.

    Args:
        strength: Inverse regularisation for the logistic fit. Larger values
            trust the calibration sample more; the default is deliberately mild
            so a small sample cannot produce wild corrections.

    Attributes:
        strength: Inverse regularisation strength.
        n_fit: Number of forecasts the transform was fitted on.
        classes_: Outcome indices seen during fitting, once fitted.
    """

    def __init__(self, strength: float = 1.0) -> None:
        if strength <= 0:
            raise CalibrationError("ProbabilityCalibrator: strength must be positive")
        self.strength = float(strength)
        self.n_fit = 0
        self._model: LogisticRegression | None = None

    @staticmethod
    def _features(probabilities: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Turn probabilities into the log-probability regressors.

        Args:
            probabilities: Forecast probabilities as a frame or array.

        Returns:
            Array of shape ``(n, 3)`` holding ``log(clip(p, epsilon, 1))``.
        """
        matrix = _as_matrix(probabilities)
        return np.log(np.clip(matrix, CALIBRATION_EPSILON, 1.0))

    def fit(self, results: pd.Series | np.ndarray, probabilities: pd.DataFrame | np.ndarray) -> ProbabilityCalibrator:
        """Fit the correction on forecasts whose outcomes are known.

        Args:
            results: Observed outcomes, ``"H"``/``"D"``/``"A"``.
            probabilities: The corresponding forecast probabilities.

        Returns:
            The fitted calibrator, for chaining.

        Raises:
            CalibrationError: If fewer than two outcomes are present, or the
                forecast and result lengths differ.
        """
        features = self._features(probabilities)
        observed = outcome_index(results)
        if len(features) != len(observed):
            raise CalibrationError(
                f"fit: {len(features)} forecast rows but {len(observed)} results"
            )
        if len(np.unique(observed)) < 2:
            raise CalibrationError("fit: need at least two distinct outcomes to calibrate")

        model = LogisticRegression(max_iter=1000, C=self.strength)
        model.fit(features, observed)
        self._model = model
        self.n_fit = len(observed)
        return self

    def transform(self, probabilities: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Apply the fitted correction.

        Args:
            probabilities: Forecast probabilities to recalibrate.

        Returns:
            Calibrated probabilities with the canonical column names. Rows are
            renormalised, because an outcome class absent from the calibration
            sample receives no mass from the classifier.

        Raises:
            CalibrationError: If called before :meth:`fit`.
        """
        if self._model is None:
            raise CalibrationError("transform: calibrator has not been fitted")

        features = self._features(probabilities)
        raw = self._model.predict_proba(features)

        out = np.zeros((len(features), len(OUTCOMES)), dtype="float64")
        for column, outcome_class in enumerate(self._model.classes_):
            out[:, int(outcome_class)] = raw[:, column]

        # A class missing from the training sample scores zero everywhere. Spread
        # a tiny mass over all three first so the row stays a valid distribution.
        out = np.clip(out, CALIBRATION_EPSILON, None)
        out = out / out.sum(axis=1, keepdims=True)
        return pd.DataFrame(out, columns=list(PROBABILITY_COLUMNS))


def walk_forward_calibrate(
    predictions: pd.DataFrame,
    *,
    min_calibration: int = 300,
    strength: float = 1.0,
    result_column: str = "outcome",
    date_column: str = "match_date",
) -> pd.DataFrame:
    """Recalibrate a chronological set of forecasts without look-ahead.

    For each matchday, a calibrator is fitted on every earlier forecast whose
    outcome is known and applied to that matchday. The first matchdays, before
    ``min_calibration`` forecasts exist, are left untouched.

    Args:
        predictions: Frame of out-of-sample forecasts, one row per match, with
            the outcome, the probability columns and a match date. Rows may be in
            any order; they are sorted internally by date.
        min_calibration: Earlier forecasts required before a calibrator is used.
            Below this the original probabilities are returned unchanged.
        strength: Inverse regularisation passed to :class:`ProbabilityCalibrator`.
        result_column: Name of the observed-outcome column.
        date_column: Name of the match-date column.

    Returns:
        A copy of ``predictions`` with three ``*_calibrated`` columns and an
        ``is_calibrated`` flag. Rows where the flag is False keep ``NaN`` in the
        calibrated columns, so an average over them cannot silently include
        uncorrected rows.

    Raises:
        CalibrationError: If required columns are missing or a date is
            unparseable.
    """
    required = {result_column, date_column, *PROBABILITY_COLUMNS}
    missing = required - set(predictions.columns)
    if missing:
        raise CalibrationError(f"walk_forward_calibrate: frame is missing {sorted(missing)}")

    frame = predictions.reset_index(drop=True).copy()
    parsed = pd.to_datetime(frame[date_column], errors="coerce", format="mixed")
    if parsed.isna().any():
        unparsed = int(parsed.isna().sum())
        raise CalibrationError(f"walk_forward_calibrate: {unparsed} row(s) have an unparseable date")

    work = frame.copy()
    work["_sort_date"] = parsed
    work["_position"] = np.arange(len(work))
    work = work.sort_values(["_sort_date", "_position"], kind="mergesort").reset_index(drop=True)

    n = len(work)
    calibrated = np.full((n, len(PROBABILITY_COLUMNS)), np.nan, dtype="float64")
    flags = np.zeros(n, dtype=bool)
    ordered_dates = work["_sort_date"].to_numpy()
    ordered_results = work[result_column].to_numpy()
    ordered_probs = work[list(PROBABILITY_COLUMNS)]

    for date in pd.unique(ordered_dates):
        prior = ordered_dates < date
        if int(prior.sum()) < min_calibration:
            continue
        try:
            calibrator = ProbabilityCalibrator(strength=strength).fit(
                ordered_results[prior], ordered_probs.loc[prior]
            )
        except CalibrationError as error:
            module_logger.debug("skipping calibration for %s: %s", date, error)
            continue

        current = ordered_dates == date
        calibrated[current] = calibrator.transform(ordered_probs.loc[current]).to_numpy()
        flags[current] = True

    # Restore the caller's original row order via the recorded position.
    order = np.argsort(work["_position"].to_numpy(), kind="mergesort")
    result = frame.copy()
    for index, column in enumerate(PROBABILITY_COLUMNS):
        result[f"{column}{CALIBRATED_SUFFIX}"] = calibrated[order, index]
    result["is_calibrated"] = flags[order]

    covered = int(flags.sum())
    module_logger.info(
        "Calibrated %d of %d forecasts (%d left raw; need %d of calibration history)",
        covered,
        n,
        n - covered,
        min_calibration,
    )
    return result


__all__ = [
    "CALIBRATED_SUFFIX",
    "CALIBRATION_EPSILON",
    "CalibrationError",
    "ProbabilityCalibrator",
    "walk_forward_calibrate",
]

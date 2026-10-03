"""Scoring rules for probability forecasts.

Two rules govern everything here. First, every metric is reported alongside the
number of matches it was computed from, because an accuracy of 55% on 40 matches
and on 3,000 matches are different claims and the difference has to be visible.
Second, discrimination is not calibration: a model can pick the right outcome
more often than a coin toss while assigning those outcomes badly-calibrated
probabilities. Both are reported.

The Brier score follows Gneiting and Raftery: the mean over matches of the sum
over the three outcomes of the squared probability error, so it ranges from 0 to
2. Some sources divide by the number of outcomes instead, giving a score in the
range 0 to 2/3. The convention is stated wherever a number appears because the
two are not comparable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from predict_football.models.dixon_coles import OUTCOMES, PROBABILITY_COLUMNS

module_logger = logging.getLogger(__name__)

#: Probability floor. Without clipping, a model that is confidently wrong pays an
#: unbounded log loss and a single match dominates the average.
EPSILON = 1e-15

#: Default number of equal-width bins for a reliability table.
DEFAULT_BINS = 10


class EvaluationError(ValueError):
    """Raised when a forecast cannot be scored as supplied."""


def outcome_index(results: pd.Series | np.ndarray) -> np.ndarray:
    """Convert outcome labels to column positions.

    Args:
        results: Series of ``"H"``, ``"D"`` or ``"A"``.

    Returns:
        Array of integer positions matching the column order of the probability
        matrix.

    Raises:
        EvaluationError: If any label is not a recognised outcome, which would
            otherwise be silently scored as a miss.
    """
    lookup = {label: position for position, label in enumerate(OUTCOMES)}
    labels = list(pd.Series(results).astype("string"))
    unknown = sorted({label for label in labels if label not in lookup})
    if unknown:
        raise EvaluationError(f"outcome_index: unrecognised outcome labels {unknown}")
    return np.array([lookup[label] for label in labels], dtype="int64")


def _as_matrix(probabilities: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Coerce probabilities into a validated ``(n, 3)`` matrix.

    Args:
        probabilities: Either the three probability columns or a raw array.

    Returns:
        Array of shape ``(n, 3)``.

    Raises:
        EvaluationError: If the shape is wrong, a probability is negative or
            greater than one, or a row does not sum to one.
    """
    if isinstance(probabilities, pd.DataFrame):
        missing = set(PROBABILITY_COLUMNS) - set(probabilities.columns)
        if missing:
            raise EvaluationError(f"probabilities: missing columns {sorted(missing)}")
        matrix = probabilities[list(PROBABILITY_COLUMNS)].to_numpy(dtype="float64")
    else:
        matrix = np.asarray(probabilities, dtype="float64")

    if matrix.ndim != 2 or matrix.shape[1] != len(OUTCOMES):
        raise EvaluationError(f"probabilities: expected shape (n, 3), got {matrix.shape}")

    if (matrix < 0).any() or (matrix > 1).any():
        raise EvaluationError("probabilities: values must lie in [0, 1]")

    sums = matrix.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-6):
        worst = float(np.abs(sums - 1.0).max())
        raise EvaluationError(f"probabilities: rows must sum to 1, worst deviation {worst:.3g}")

    return matrix


def multiclass_brier_score(results: pd.Series | np.ndarray, probabilities: pd.DataFrame | np.ndarray) -> float:
    """Mean squared error of the full probability vector.

    Args:
        results: Observed outcomes.
        probabilities: Forecast probabilities.

    Returns:
        Brier score in ``[0, 2]``, lower is better. The empirical base rate of
        always predicting the outcome frequencies gives roughly 0.66 for a
        typical football season, so a model must fall well below that to be
        worth anything.
    """
    matrix = _as_matrix(probabilities)
    observed = np.zeros_like(matrix)
    observed[np.arange(len(matrix)), outcome_index(results)] = 1.0
    return float(np.mean(np.sum((matrix - observed) ** 2, axis=1)))


def multiclass_log_loss(results: pd.Series | np.ndarray, probabilities: pd.DataFrame | np.ndarray) -> float:
    """Negative log likelihood of the observed outcomes.

    Args:
        results: Observed outcomes.
        probabilities: Forecast probabilities.

    Returns:
        Mean log loss in nats. Random guessing on three outcomes scores
        ``ln(3) = 1.0986``; anything above that is worse than uniform.
    """
    matrix = _as_matrix(probabilities)
    observed = outcome_index(results)
    picked = matrix[np.arange(len(matrix)), observed]
    return float(-np.mean(np.log(np.clip(picked, EPSILON, 1.0))))


def outcome_accuracy(results: pd.Series | np.ndarray, probabilities: pd.DataFrame | np.ndarray) -> float:
    """Share of matches where the most likely outcome was the one observed.

    Args:
        results: Observed outcomes.
        probabilities: Forecast probabilities.

    Returns:
        Accuracy in ``[0, 1]``. Always compare this against always predicting the
        single most common outcome, which is roughly 0.45 in the Premier League.
    """
    matrix = _as_matrix(probabilities)
    predicted = matrix.argmax(axis=1)
    return float(np.mean(predicted == outcome_index(results)))


@dataclass(frozen=True)
class CalibrationSummary:
    """How close predicted frequencies are to observed frequencies.

    Attributes:
        n_matches: Matches the calibration was computed from.
        bins: Bin edges on the probability axis.
        counts: Matches in each bin.
        mean_predicted: Mean predicted probability per bin.
        observed_frequency: Observed share of the event per bin.
        weighted_absolute_error: Probability-weighted mean absolute gap between
            predicted and observed. Zero is perfect; it is also known as the
            calibration error weighted by bin population.
        event: Which column of the probability matrix was assessed.
    """

    n_matches: int
    bins: np.ndarray
    counts: np.ndarray
    mean_predicted: np.ndarray
    observed_frequency: np.ndarray
    weighted_absolute_error: float
    event: str


def calibration(
    results: pd.Series | np.ndarray,
    probabilities: pd.DataFrame | np.ndarray,
    outcome: str = "H",
    bins: int = DEFAULT_BINS,
) -> CalibrationSummary:
    """Assess calibration of one outcome.

    Args:
        results: Observed outcomes.
        probabilities: Forecast probabilities.
        outcome: Which outcome to assess, one of ``"H"``, ``"D"``, ``"A"``.
        bins: Number of equal-width probability bins.

    Returns:
        Calibration summary including bin populations, so an apparently good
        calibration can be checked for whether it rests on a handful of matches.

    Raises:
        EvaluationError: If ``outcome`` is not recognised or ``bins`` is not
            positive.
    """
    if outcome not in OUTCOMES:
        raise EvaluationError(f"calibration: outcome must be one of {list(OUTCOMES)}")
    if bins < 1:
        raise EvaluationError("calibration: bins must be at least 1")

    matrix = _as_matrix(probabilities)
    position = OUTCOMES.index(outcome)
    predicted = matrix[:, position]
    observed = (outcome_index(results) == position).astype("float64")

    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(predicted, edges[1:-1], right=False), 0, bins - 1)

    counts = np.zeros(bins, dtype="int64")
    mean_predicted = np.full(bins, np.nan, dtype="float64")
    observed_frequency = np.full(bins, np.nan, dtype="float64")

    for b in range(bins):
        selected = index == b
        counts[b] = int(selected.sum())
        if counts[b]:
            mean_predicted[b] = float(predicted[selected].mean())
            observed_frequency[b] = float(observed[selected].mean())

    populated = counts > 0
    weights = counts[populated].astype("float64")
    gaps = np.abs(mean_predicted[populated] - observed_frequency[populated])
    weighted_error = float(np.sum(weights * gaps) / weights.sum()) if weights.sum() else float("nan")

    return CalibrationSummary(
        n_matches=len(matrix),
        bins=edges,
        counts=counts,
        mean_predicted=mean_predicted,
        observed_frequency=observed_frequency,
        weighted_absolute_error=weighted_error,
        event=outcome,
    )


@dataclass(frozen=True)
class ForecastReport:
    """Scores for one set of forecasts, with the sample size attached.

    Attributes:
        n_matches: Matches scored. Every metric below refers to this many.
        brier: Multiclass Brier score in ``[0, 2]``.
        log_loss: Multiclass log loss in nats.
        accuracy: Share of matches where the most likely outcome occurred.
        calibration: Per-outcome calibration summaries.
        label: Short description of whose forecasts these are.
    """

    n_matches: int
    brier: float
    log_loss: float
    accuracy: float
    calibration: dict[str, CalibrationSummary]
    label: str = ""

    def as_row(self) -> dict[str, object]:
        """Flatten the report for a comparison table.

        Returns:
            Dict of metric name to value, including the sample size and the
            calibration error so no figure travels without its coverage.
        """
        row: dict[str, object] = {
            "label": self.label,
            "n_matches": self.n_matches,
            "brier": round(self.brier, 4),
            "log_loss": round(self.log_loss, 4),
            "accuracy": round(self.accuracy, 4),
        }
        for outcome, summary in self.calibration.items():
            row[f"calibration_error_{outcome}"] = round(summary.weighted_absolute_error, 4)
        return row

    def __str__(self) -> str:
        """Render the report, always leading with the sample size."""
        header = f"{self.label or 'forecast'} over {self.n_matches} matches"
        lines = [
            header,
            f"  Brier score      {self.brier:.4f}   (lower better; base rate ~0.66)",
            f"  Log loss         {self.log_loss:.4f}   (lower better; uniform 1.0986)",
            f"  Outcome accuracy {self.accuracy:.4f}   (majority class ~0.45)",
        ]
        for outcome, summary in sorted(self.calibration.items()):
            lines.append(
                f"  Calibration {outcome}  {summary.weighted_absolute_error:.4f}   (0 is perfect)"
            )
        return "\n".join(lines)


def evaluate(
    results: pd.Series | np.ndarray,
    probabilities: pd.DataFrame | np.ndarray,
    label: str = "",
    outcomes: tuple[str, ...] = OUTCOMES,
) -> ForecastReport:
    """Score a set of forecasts.

    Args:
        results: Observed outcomes.
        probabilities: Forecast probabilities.
        label: Description used in the rendered report.
        outcomes: Which outcomes to assess for calibration.

    Returns:
        Report carrying every metric and the number of matches evaluated.
    """
    matrix = _as_matrix(probabilities)
    return ForecastReport(
        n_matches=len(matrix),
        brier=multiclass_brier_score(results, probabilities),
        log_loss=multiclass_log_loss(results, probabilities),
        accuracy=outcome_accuracy(results, probabilities),
        calibration={outcome: calibration(results, probabilities, outcome) for outcome in outcomes},
        label=label,
    )


def base_rate_probabilities(results: pd.Series | np.ndarray) -> pd.DataFrame:
    """Forecast the empirical outcome frequencies, ignoring who is playing.

    Args:
        results: Observed outcomes used to estimate the frequencies.

    Returns:
        Probability frame with one identical row per match. This is the
        benchmark a sophisticated model has to beat; a model that cannot manage
        it has learned nothing beyond the league's result distribution.
    """
    labels = pd.Series(results).reset_index(drop=True)
    frequencies = labels.value_counts(normalize=True)
    return pd.DataFrame(
        {
            column: [float(frequencies.get(outcome, 0.0))] * len(labels)
            for column, outcome in zip(PROBABILITY_COLUMNS, OUTCOMES, strict=True)
        }
    )


def implied_probabilities(
    odds: pd.DataFrame,
    columns: tuple[str, ...] = ("odds_home_close", "odds_draw_close", "odds_away_close"),
) -> pd.DataFrame:
    """Convert decimal odds to de-vigged outcome probabilities.

    Bookmaker prices sum to more than one; the excess is the overround, or the
    margin. Removing it proportionally (dividing each inverse price by the sum
    of inverse prices) is the standard, transparent de-vig and is the only
    method used here. It gives the market a fair, comparable forecast rather
    than quietly scoring raw inverse odds that sum to 1.05 and flatter
    themselves.

    Args:
        odds: Frame carrying the decimal odds columns.
        columns: The three odds columns, in home/draw/away order.

    Returns:
        Frame with the canonical probability columns.

    Raises:
        EvaluationError: If a required column is missing, a price is not a
            finite decimal odd greater than one, or a row is incomplete.
    """
    missing = [column for column in columns if column not in odds.columns]
    if missing:
        raise EvaluationError(f"implied_probabilities: missing odds columns {missing}")
    if len(columns) != len(PROBABILITY_COLUMNS):
        raise EvaluationError("implied_probabilities: expected exactly one price per outcome")

    prices = odds[list(columns)].to_numpy(dtype="float64")
    if np.isnan(prices).any():
        raise EvaluationError("implied_probabilities: odds contain missing values")
    if not np.isfinite(prices).all() or (prices <= 1.0).any():
        raise EvaluationError("implied_probabilities: decimal odds must be finite and greater than 1")

    inverse = 1.0 / prices
    return pd.DataFrame(inverse / inverse.sum(axis=1, keepdims=True), columns=list(PROBABILITY_COLUMNS))


def compare_reports(reports: list[ForecastReport]) -> pd.DataFrame:
    """Tabulate several reports for side-by-side reading.

    Args:
        reports: Reports to compare.

    Returns:
        DataFrame with one row per report.
    """
    return pd.DataFrame([report.as_row() for report in reports])


__all__ = [
    "DEFAULT_BINS",
    "EPSILON",
    "CalibrationSummary",
    "EvaluationError",
    "ForecastReport",
    "base_rate_probabilities",
    "calibration",
    "compare_reports",
    "evaluate",
    "implied_probabilities",
    "multiclass_brier_score",
    "multiclass_log_loss",
    "outcome_accuracy",
    "outcome_index",
]

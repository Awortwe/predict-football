"""Tests for post-hoc probability recalibration.

The synthetic forecasts are generated from a known ground truth, so the expected
direction of the correction is known independently of the implementation: a
deliberately over-sharpened forecast must be softened, and an already-calibrated
forecast must be left roughly alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predict_football.calibration import (
    CalibrationError,
    ProbabilityCalibrator,
    walk_forward_calibrate,
)
from predict_football.evaluation import multiclass_brier_score, multiclass_log_loss
from predict_football.models.dixon_coles import OUTCOMES, PROBABILITY_COLUMNS


def _draw_true_probabilities(rng: np.random.Generator, n: int) -> np.ndarray:
    """Draw ``n`` probability vectors from a Dirichlet distribution."""
    return rng.dirichlet([2.0, 2.0, 2.0], size=n)


def _sample_outcomes(probabilities: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample one outcome per row from the supplied probabilities."""
    cumulative = np.cumsum(probabilities, axis=1)
    draws = rng.random(len(probabilities))
    indices = (draws > cumulative[:, 0]).astype(int) + (draws > cumulative[:, 1]).astype(int)
    return np.array(OUTCOMES)[indices]


def _temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Sharpen (T < 1) or soften (T > 1) a set of probabilities."""
    adjusted = np.clip(probabilities, 1e-9, 1.0) ** (1.0 / temperature)
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def _frame(results: np.ndarray, probabilities: np.ndarray) -> pd.DataFrame:
    """Assemble a forecast frame with canonical column names."""
    frame = pd.DataFrame({column: probabilities[:, i] for i, column in enumerate(PROBABILITY_COLUMNS)})
    frame["outcome"] = results
    return frame


def test_calibration_reduces_error_on_an_over_sharp_forecast() -> None:
    """An over-sharpened forecast must be softened and score better."""
    rng = np.random.default_rng(20260810)
    truth = _draw_true_probabilities(rng, 4000)
    observed = _sample_outcomes(truth, rng)
    sharp = _temperature(truth, temperature=0.5)
    frame = _frame(observed, sharp)

    train, test = frame.iloc[:2000], frame.iloc[2000:]
    calibrator = ProbabilityCalibrator().fit(train["outcome"], train[list(PROBABILITY_COLUMNS)])
    calibrated = calibrator.transform(test[list(PROBABILITY_COLUMNS)])

    raw_brier = multiclass_brier_score(test["outcome"], test[list(PROBABILITY_COLUMNS)])
    raw_log_loss = multiclass_log_loss(test["outcome"], test[list(PROBABILITY_COLUMNS)])

    assert multiclass_brier_score(test["outcome"], calibrated) < raw_brier
    assert multiclass_log_loss(test["outcome"], calibrated) < raw_log_loss


def test_calibration_leaves_a_calibrated_forecast_nearly_unchanged() -> None:
    """When forecasts already match the truth, the correction must be small."""
    rng = np.random.default_rng(7)
    truth = _draw_true_probabilities(rng, 20000)
    observed = _sample_outcomes(truth, rng)
    frame = _frame(observed, truth)

    calibrator = ProbabilityCalibrator().fit(frame["outcome"], frame[list(PROBABILITY_COLUMNS)])
    recalibrated = calibrator.transform(frame[list(PROBABILITY_COLUMNS)].iloc[:5000])

    gap = np.abs(recalibrated.to_numpy() - truth[:5000]).mean()
    assert gap < 0.03


def test_transform_before_fit_is_an_error() -> None:
    """Using an unfitted calibrator must raise rather than pass data through."""
    with pytest.raises(CalibrationError, match="not been fitted"):
        ProbabilityCalibrator().transform(pd.DataFrame({c: [0.4, 0.3, 0.3] for c in PROBABILITY_COLUMNS}))


def test_fit_requires_more_than_one_outcome() -> None:
    """A single observed outcome cannot identify a correction."""
    probabilities = pd.DataFrame({"prob_home": [0.5, 0.6], "prob_draw": [0.3, 0.2], "prob_away": [0.2, 0.2]})

    with pytest.raises(CalibrationError, match="two distinct outcomes"):
        ProbabilityCalibrator().fit(pd.Series(["H", "H"]), probabilities)


def test_calibrated_rows_sum_to_one() -> None:
    """Whatever the fit, the output must remain a probability distribution."""
    rng = np.random.default_rng(11)
    truth = _draw_true_probabilities(rng, 2000)
    observed = _sample_outcomes(truth, rng)
    frame = _frame(observed, _temperature(truth, 0.6))

    calibrator = ProbabilityCalibrator().fit(frame["outcome"], frame[list(PROBABILITY_COLUMNS)])
    out = calibrator.transform(frame[list(PROBABILITY_COLUMNS)])

    assert np.allclose(out.to_numpy().sum(axis=1), 1.0)


def _walk_frame(seed: int = 3, matchdays: int = 40, per_day: int = 15) -> pd.DataFrame:
    """Build a dated forecast frame spanning several matchdays."""
    rng = np.random.default_rng(seed)
    total = matchdays * per_day
    truth = _draw_true_probabilities(rng, total)
    observed = _sample_outcomes(truth, rng)
    frame = _frame(observed, _temperature(truth, 0.5))
    frame["match_date"] = np.repeat(pd.date_range("2023-08-01", periods=matchdays, freq="7D"), per_day)
    return frame


def test_walk_forward_calibrate_flags_covered_rows() -> None:
    """Early matchdays stay raw; later ones are marked calibrated."""
    frame = _walk_frame()
    out = walk_forward_calibrate(frame, min_calibration=200)

    assert out["is_calibrated"].any()
    assert not out["is_calibrated"].all()
    # Uncalibrated rows must not carry a number that a naive average could include.
    raw_rows = out.loc[~out["is_calibrated"], [f"{c}_calibrated" for c in PROBABILITY_COLUMNS]]
    assert raw_rows.isna().all().all()


def test_walk_forward_calibrate_improves_the_covered_rows() -> None:
    """On the rows it covers, the correction must beat the raw forecast."""
    frame = _walk_frame()
    out = walk_forward_calibrate(frame, min_calibration=200)
    covered = out[out["is_calibrated"]]

    calibrated_columns = [f"{c}_calibrated" for c in PROBABILITY_COLUMNS]
    raw_brier = multiclass_brier_score(covered["outcome"], covered[list(PROBABILITY_COLUMNS)].to_numpy())
    calibrated_brier = multiclass_brier_score(covered["outcome"], covered[calibrated_columns].to_numpy())

    assert calibrated_brier < raw_brier


def test_walk_forward_calibrate_does_not_look_ahead() -> None:
    """Changing a later outcome must not alter an earlier calibrated row.

    This is the whole point of the walk-forward discipline: if it fails, the
    reported improvement is contaminated by outcomes that were not yet known.
    """
    frame = _walk_frame()
    baseline = walk_forward_calibrate(frame, min_calibration=200)

    tampered = frame.copy()
    last = tampered.index[-1]
    tampered.loc[last, "outcome"] = "A" if tampered.loc[last, "outcome"] != "A" else "H"
    changed = walk_forward_calibrate(tampered, min_calibration=200)

    calibrated_columns = [f"{c}_calibrated" for c in PROBABILITY_COLUMNS]
    earlier = baseline.loc[baseline["is_calibrated"]].index
    pd.testing.assert_frame_equal(
        baseline.loc[earlier, calibrated_columns],
        changed.loc[earlier, calibrated_columns],
    )


def test_walk_forward_calibrate_rejects_missing_columns() -> None:
    """A frame without probabilities cannot be calibrated."""
    with pytest.raises(CalibrationError, match="missing"):
        walk_forward_calibrate(pd.DataFrame({"outcome": ["H"], "match_date": ["2023-08-01"]}))


def test_walk_forward_calibrate_rejects_unparseable_dates() -> None:
    """A date that cannot be read must raise rather than be silently dropped."""
    frame = _walk_frame()
    frame["match_date"] = frame["match_date"].astype("object")
    frame.loc[0, "match_date"] = "not a date"

    with pytest.raises(CalibrationError, match="unparseable date"):
        walk_forward_calibrate(frame, min_calibration=200)

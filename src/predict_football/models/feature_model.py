"""Gradient-boosted pre-match model on the engineered features.

Dixon-Coles is the transparent baseline: it knows about attack, defence and home
advantage, and nothing else. This module spends the features from
:mod:`predict_football.features` -- recent form, rest, venue splits, career
scoring rates -- through a small gradient-boosted classifier and asks a plain
question: does that extra information beat the baseline out of sample, net of
overfitting?

Two properties are non-negotiable and are why the model routes every feature
through :func:`~predict_football.features.build_pre_match_features` and
:func:`~predict_football.features.build_fixture_features`:

* The training matrix for a match is built only from earlier matches.
* The matrix for a fixture to be forecast is also built only from matches dated
  strictly before it. The adapter keeps the training history it was fitted on
  and re-derives the fixture features from it, so a test match can never be
  described using its own result or any later one.

The classifier emits the same canonical probability columns as the baseline, so
it is a drop-in replacement anywhere a model factory is accepted. It has no
expected-goals interpretation, so it deliberately omits ``lambda_*``; the
forecast frame leaves those null.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from predict_football.features import (
    FEATURE_COLUMNS,
    build_fixture_features,
    build_pre_match_features,
)
from predict_football.models.dixon_coles import OUTCOMES, PROBABILITY_COLUMNS

module_logger = logging.getLogger(__name__)

#: Fixture columns carried through to a forecast, when present.
_IDENTIFIER_COLUMNS = (
    "match_id",
    "league_key",
    "season",
    "season_code",
    "match_date",
    "home_team",
    "away_team",
)

#: Columns a training frame must carry.
_REQUIRED = frozenset({"match_date", "home_team", "away_team", "home_goals", "away_goals", "result"})

_OUTCOME_POSITION = {outcome: position for position, outcome in enumerate(OUTCOMES)}


@dataclass(frozen=True)
class FeatureModelOptions:
    """Hyperparameters for the gradient-boosted classifier.

    The defaults are deliberately conservative. A football season is a few
    thousand rows at most, so an expressive forest memorises it; shallow trees,
    a slow learning rate and a large leaf minimum are what stop the walk-forward
    score from being an illusion of in-sample fit.

    Attributes:
        n_estimators: Number of boosting rounds.
        learning_rate: Shrinkage applied to each tree.
        num_leaves: Maximum leaves per tree.
        max_depth: Maximum tree depth, or ``-1`` for no limit.
        min_child_samples: Minimum rows in a leaf.
        subsample: Row subsampling fraction per tree.
        colsample_bytree: Feature subsampling fraction per tree.
        reg_lambda: L2 regularisation on leaf weights.
        random_state: Seed, fixed so a benchmark is reproducible.
    """

    n_estimators: int = 300
    learning_rate: float = 0.03
    num_leaves: int = 15
    max_depth: int = -1
    min_child_samples: int = 40
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    random_state: int = 0

    def build(self) -> FeatureModel:
        """Construct an unfitted model with these options.

        Returns:
            A fresh :class:`FeatureModel`, so no state leaks between folds.
        """
        return FeatureModel(options=self)


@dataclass
class FeatureModel:
    """Multiclass classifier over the engineered pre-match features.

    Attributes:
        options: Hyperparameters used for the next fit.
        converged: Whether a fit has completed. LightGBM has no scipy-style
            success flag, so this reports that training ran and produced a
            usable classifier.
        n_matches: Training rows used for the most recent fit.
    """

    options: FeatureModelOptions = field(default_factory=FeatureModelOptions)
    converged: bool = False
    n_matches: int = 0
    _history: pd.DataFrame | None = field(default=None, repr=False)
    _classifier: LGBMClassifier | None = field(default=None, repr=False)

    def fit(self, matches: pd.DataFrame) -> FeatureModel:
        """Fit the classifier on completed matches.

        Args:
            matches: Canonical match frame with results. Only played rows are
                used, and each row's features are built from strictly earlier
                matches.

        Returns:
            The fitted model, for chaining.

        Raises:
            ValueError: If required columns are missing, no row has a result, or
                an outcome label is not one of ``H``/``D``/``A``.
        """
        missing = _REQUIRED - set(matches.columns)
        if missing:
            raise ValueError(f"FeatureModel.fit: frame is missing {sorted(missing)}")

        played = matches[matches["result"].notna()].copy()
        if played.empty:
            raise ValueError("FeatureModel.fit: no row has a result")

        labels = played["result"].astype("string")
        unknown = sorted(set(labels) - set(OUTCOMES))
        if unknown:
            raise ValueError(f"FeatureModel.fit: unrecognised outcome labels {unknown}")

        features = build_pre_match_features(played)
        design = features[list(FEATURE_COLUMNS)].to_numpy(dtype="float64")
        target = features["result"].map(_OUTCOME_POSITION).to_numpy(dtype="int64")

        classifier = LGBMClassifier(
            objective="multiclass",
            num_class=len(OUTCOMES),
            n_estimators=self.options.n_estimators,
            learning_rate=self.options.learning_rate,
            num_leaves=self.options.num_leaves,
            max_depth=self.options.max_depth,
            min_child_samples=self.options.min_child_samples,
            subsample=self.options.subsample,
            colsample_bytree=self.options.colsample_bytree,
            reg_lambda=self.options.reg_lambda,
            random_state=self.options.random_state,
            n_jobs=1,
            verbosity=-1,
        )
        classifier.fit(design, target)

        self._classifier = classifier
        self._history = played
        self.n_matches = len(played)
        self.converged = True
        module_logger.info("Fitted FeatureModel on %d matches across %d features", self.n_matches, len(FEATURE_COLUMNS))
        return self

    def outcome_probabilities(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Forecast home, draw and away probabilities for each fixture.

        Args:
            matches: Fixtures to forecast. Their results, if any are present, are
                ignored; the features come from the history the model was fitted
                on, never from the prediction frame.

        Returns:
            Frame with the fixture identifiers and the three probability columns,
            each row summing to one.

        Raises:
            ValueError: If the model has not been fitted, or a required fixture
                column is missing.
        """
        if self._classifier is None or self._history is None:
            raise ValueError("FeatureModel.outcome_probabilities: model has not been fitted")
        missing = {"match_date", "home_team", "away_team"} - set(matches.columns)
        if missing:
            raise ValueError(f"FeatureModel.outcome_probabilities: frame is missing {sorted(missing)}")

        features = build_fixture_features(self._history, matches)
        design = features[list(FEATURE_COLUMNS)].to_numpy(dtype="float64")
        raw = self._classifier.predict_proba(design)

        # A class absent from training has no column; score it as impossible
        # rather than letting the columns shift under us. Real leagues produce
        # all three, and the caller sees the zeros if one is genuinely missing.
        probabilities = np.zeros((len(design), len(OUTCOMES)), dtype="float64")
        for column, label in enumerate(self._classifier.classes_):
            probabilities[:, int(label)] = raw[:, column]

        identifiers = [column for column in _IDENTIFIER_COLUMNS if column in matches.columns]
        result = matches[identifiers].reset_index(drop=True).copy()
        for position, column in enumerate(PROBABILITY_COLUMNS):
            result[column] = probabilities[:, position]
        return result


__all__ = ["FeatureModel", "FeatureModelOptions"]

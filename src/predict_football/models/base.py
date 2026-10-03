"""The interface every forecast model implements.

Dixon-Coles and the gradient-boosted classifier share no code, and they should
not be forced to: one is a generative Poisson model, the other a discriminative
classifier. What they must share is the contract the inference, backtest and
benchmark layers rely on, so those layers can be written once and tested against
a stub.

The contract is intentionally small:

* ``fit(matches)`` learns from completed matches and returns ``self`` so it can
  be chained.
* ``outcome_probabilities(fixtures)`` returns the home/draw/away distribution for
  each fixture, indexed back onto the frame it was handed.
* ``converged`` says whether a fit produced a usable model, so an adapter can
  surface that without knowing which optimiser ran.

A model that reports expected goals may add ``lambda_home``/``lambda_away`` to
the forecast frame, but the columns are optional: a classifier has no such
quantity and must not be forced to invent one.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

__all__ = ["PredictiveModel"]


@runtime_checkable
class PredictiveModel(Protocol):
    """Structural type for a pre-match outcome model.

    Implementations are not expected to subclass this; they only have to provide
    the methods. The protocol exists so the type checker can reject a model that
    forgets ``converged`` before the backtest discovers it at runtime.
    """

    #: Whether a fit has completed and produced a usable model.
    converged: bool

    def fit(self, matches: pd.DataFrame) -> PredictiveModel:
        """Fit the model on completed matches.

        Args:
            matches: Canonical match frame carrying results.

        Returns:
            The fitted model, for chaining.
        """
        ...

    def outcome_probabilities(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Forecast home, draw and away probabilities for each fixture.

        Args:
            matches: Fixtures to forecast.

        Returns:
            Frame carrying the fixture identifiers and the three canonical
            probability columns.
        """
        ...

"""predict_football -- football probability and intelligence system.

Design rules that apply package-wide:

* Probabilities, never guarantees. Nothing in this package may assert that an
  outcome "will" happen.
* No data leakage. Feature access is gated through
  :mod:`predict_football.data.schema` so post-match information cannot reach a
  pre-match model by accident.
* Everything works offline. No API key is required for any core code path.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]

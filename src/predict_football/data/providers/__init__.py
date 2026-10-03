"""Adapters for external football data sources."""

from __future__ import annotations

from predict_football.data.providers.base import DataProvider, ProviderError, SeasonRef
from predict_football.data.providers.registry import get_provider, list_providers

__all__ = ["DataProvider", "ProviderError", "SeasonRef", "get_provider", "list_providers"]

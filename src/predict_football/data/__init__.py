"""Data layer: providers, caching, cleaning, and the SQLite store."""

from __future__ import annotations

from predict_football.data.cache import RawCache
from predict_football.data.providers.base import (
    DataProvider,
    ProviderError,
    ProviderNotAvailable,
    RemoteDataDisabled,
    SeasonRef,
)
from predict_football.data.providers.football_data_co import FootballDataCoProvider
from predict_football.data.providers.registry import get_provider, list_providers
from predict_football.data.providers.statsbomb import StatsBombOpenProvider

__all__ = [
    "DataProvider",
    "FootballDataCoProvider",
    "ProviderError",
    "ProviderNotAvailable",
    "RawCache",
    "RemoteDataDisabled",
    "SeasonRef",
    "StatsBombOpenProvider",
    "get_provider",
    "list_providers",
]

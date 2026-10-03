"""Configuration: filesystem layout, environment settings, and source licences."""

from __future__ import annotations

from predict_football.config.leagues import (
    CompetitionType,
    League,
    get_league,
    list_leagues,
)
from predict_football.config.licences import LICENCES, Licence, licence_for
from predict_football.config.settings import Settings, get_settings

__all__ = [
    "LICENCES",
    "CompetitionType",
    "League",
    "Licence",
    "Settings",
    "get_league",
    "get_settings",
    "licence_for",
    "list_leagues",
]

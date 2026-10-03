"""The :class:`DataProvider` interface.

Every external data source is reached through this one interface, so swapping
providers -- or adding a new league from a new source -- never requires touching
modelling code.

Implementations must return frames conforming to the canonical schema in
:mod:`predict_football.data.schema`. Providers are responsible for *fetching and
mapping source fields to canonical names*; they are **not** responsible for
cross-provider consistency, which is the job of
:mod:`predict_football.data.cleaning`.

Contract every implementation must honour:

* **Never invent data.** If a field is not published by the source, leave it
  null. Do not impute, interpolate or guess. Fabricated columns are worse than
  missing ones because they are indistinguishable from real ones later.
* **Never hit the network when cached.** Fetch through :class:`RawCache`.
* **Label derived values.** A value the provider computed rather than published
  must be identified as derived in the code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date

import pandas as pd


class ProviderError(RuntimeError):
    """Base class for all data provider failures."""


class ProviderNotAvailable(ProviderError):
    """A provider is configured but cannot serve the request.

    Example: a competition the provider does not cover at all.
    """


class RemoteDataDisabled(ProviderError):
    """A network fetch was attempted while network access is disabled.

    Raised instead of attempting a request, so tests and offline runs fail
    loudly and immediately rather than silently hanging on a socket.
    """


@dataclass(frozen=True)
class SeasonRef:
    """A single season of a competition, as offered by a provider.

    Attributes:
        league_key: Internal competition key, e.g. ``"ENG_PL"``.
        season_code: Provider-specific four-character code, e.g. ``"2425"``.
        season_label: Human-readable label, e.g. ``"2024/25"``.
        start_date: Approximate first match date.
        end_date: Approximate last match date.
        match_count: Number of matches the provider expects to return, or
            ``None`` if not known in advance.
    """

    league_key: str
    season_code: str
    season_label: str
    start_date: date | None = None
    end_date: date | None = None
    match_count: int | None = None

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.league_key} {self.season_label} ({self.season_code})"


class DataProvider(ABC):
    """Abstract base for all football data sources.

    Subclasses must set :attr:`name` and implement every abstract method.

    Attributes:
        name: Stable provider key, matching a licence entry in
            :mod:`predict_football.config.licences`.
    """

    #: Stable provider key. Must match a key in ``LICENCES``.
    name: str = "abstract"

    #: Human-readable provider name for logs and the UI.
    display_name: str = "Abstract provider"

    @abstractmethod
    def available_seasons(self, league_key: str) -> list[SeasonRef]:
        """List the seasons this provider can supply for a competition.

        Args:
            league_key: Internal competition key, e.g. ``"ENG_PL"``.

        Returns:
            Seasons ordered oldest to newest.

        Raises:
            ProviderNotAvailable: If the provider does not cover the competition.
        """

    @abstractmethod
    def fetch_matches(self, league_key: str, season_code: str) -> pd.DataFrame:
        """Fetch match results for one competition season.

        Args:
            league_key: Internal competition key.
            season_code: Four-character season code, e.g. ``"2425"``.

        Returns:
            A frame conforming to the canonical match schema. Rows with no known
            final result must have a null ``result``.

        Raises:
            ProviderNotAvailable: If the combination is not available.
            RemoteDataDisabled: If a network fetch is needed but disabled.
        """

    def fetch_odds(self, league_key: str, season_code: str) -> pd.DataFrame:
        """Fetch bookmaker odds for one competition season.

        Args:
            league_key: Internal competition key.
            season_code: Four-character season code.

        Returns:
            A long-format odds frame with one row per match and bookmaker.

        Raises:
            ProviderNotAvailable: If the provider publishes no odds.
        """
        raise ProviderNotAvailable(f"{self.display_name} does not provide bookmaker odds.")

    def fetch_events(self, match_id: str) -> pd.DataFrame:
        """Fetch the ordered event stream for one match.

        Args:
            match_id: Canonical match identifier.

        Returns:
            A frame conforming to the canonical event schema, ordered by
            period, minute, second.

        Raises:
            ProviderNotAvailable: If the provider publishes no event data.
        """
        raise ProviderNotAvailable(f"{self.display_name} does not provide event-level data.")

    def fetch_lineups(self, match_id: str) -> pd.DataFrame:
        """Fetch the full squad list for one match, including substitutes.

        Args:
            match_id: Canonical match identifier.

        Returns:
            A frame conforming to the canonical lineup schema.

        Raises:
            ProviderNotAvailable: If the provider publishes no lineups.
        """
        raise ProviderNotAvailable(f"{self.display_name} does not provide lineup data.")

    def covers(self, league_key: str) -> bool:
        """Report whether this provider has any data for a competition.

        Args:
            league_key: Internal competition key.

        Returns:
            True if the competition is in this provider's coverage map.
        """
        return False

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"<{type(self).__name__} name={self.name!r}>"

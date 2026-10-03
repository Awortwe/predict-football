"""League and competition registry.

Multi-league expansion is a configuration change, not a code change: adding
La Liga, the Champions League, AFCON or the World Cup means adding a
:class:`League` entry here and, if needed, a data adapter that knows the new
source.

Note on identifiers: we deliberately do NOT hard-code third-party numeric
competition IDs. Those IDs are provider-specific, silently renumbered, and easy
to get wrong. Adapters discover them by matching human-readable competition
names against the provider's own published manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from predict_football.config.licences import Licence


class CompetitionType(str, Enum):
    """How a competition is played, which changes how we model it.

    ``LEAGUE`` is a double round-robin (home and away), so home advantage and
    form are meaningful and a team plays everyone. ``CUP`` is a knockout:
    single matches, possible extra time and penalties, no home/away balance. The
    distinction matters because a model trained on league results will be
    miscalibrated on cup ties.
    """

    LEAGUE = "league"
    CUP = "cup"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass(frozen=True)
class League:
    """Static description of a competition we can model.

    Attributes:
        key: Stable internal identifier, e.g. ``"ENG_PL"``. Used as the
            database primary key and in file paths, so never change one once
            data has been ingested under it.
        name: Human-readable competition name.
        country: Governing country or confederation scope.
        competition_type: League or cup, see :class:`CompetitionType`.
        tier: Competition level within its country (1 = top flight). ``None``
            for national-team tournaments.
        source_codes: Provider-specific short codes, keyed by provider name.
            e.g. ``{"football_data_co": "E0"}``.
        provider_names: Human-readable competition names as published by each
            provider, used for manifest-based discovery rather than guessing
            numeric IDs.
        licence: Licence governing redistribution of this competition's data.
        notes: Free-text caveats, coverage gaps, and modelling warnings.
    """

    key: str
    name: str
    country: str
    competition_type: CompetitionType
    tier: int | None
    source_codes: dict[str, str] = field(default_factory=dict)
    provider_names: dict[str, str] = field(default_factory=dict)
    licence: Licence = Licence.INTERNAL_RESEARCH_ONLY
    notes: str = ""

    def code_for(self, provider: str) -> str | None:
        """Return this competition's short code for a given provider.

        Args:
            provider: Provider name, e.g. ``"football_data_co"``.

        Returns:
            The provider-specific code, or ``None`` if we have no mapping.
        """
        return self.source_codes.get(provider)


#: Every competition the project knows about.
LEAGUES: dict[str, League] = {
    league.key: league
    for league in (
        League(
            key="ENG_PL",
            name="Premier League",
            country="England",
            competition_type=CompetitionType.LEAGUE,
            tier=1,
            source_codes={"football_data_co": "E0", "openfootball": "england/1", "football_data_org": "PL"},
            provider_names={"statsbomb_open": "Premier League", "api_football": "Premier League"},
            licence=Licence.INTERNAL_RESEARCH_ONLY,
            notes=(
                "Primary competition. football-data.co.uk has long-run history plus "
                "closing bookmaker odds (our strongest baseline). StatsBomb open data "
                "covers 2015/16 completely for in-match events."
            ),
        ),
        League(
            key="ENG_CHAMPIONSHIP",
            name="EFL Championship",
            country="England",
            competition_type=CompetitionType.LEAGUE,
            tier=2,
            source_codes={"football_data_co": "E1", "openfootball": "england/2", "football_data_org": "ELC"},
            provider_names={"api_football": "Championship"},
            licence=Licence.INTERNAL_RESEARCH_ONLY,
            notes="Useful as a promotion/relegation continuity check for tier-1 strength.",
        ),
        League(
            key="ESP_LA_LIGA",
            name="La Liga",
            country="Spain",
            competition_type=CompetitionType.LEAGUE,
            tier=1,
            source_codes={"football_data_co": "SP1", "openfootball": "spain/1", "football_data_org": "PD"},
            provider_names={"statsbomb_open": "La Liga", "api_football": "La Liga"},
            licence=Licence.INTERNAL_RESEARCH_ONLY,
            notes=(
                "Long history available. StatsBomb coverage is only scattered samples "
                "(roughly one match per matchday in recent seasons), so in-match event "
                "data is thin here compared with the Premier League."
            ),
        ),
        League(
            key="ITA_SERIE_A",
            name="Serie A",
            country="Italy",
            competition_type=CompetitionType.LEAGUE,
            tier=1,
            source_codes={"football_data_co": "I1", "openfootball": "italy/1", "football_data_org": "SA"},
            provider_names={"api_football": "Serie A"},
            licence=Licence.INTERNAL_RESEARCH_ONLY,
        ),
        League(
            key="GER_BUNDESLIGA",
            name="Bundesliga",
            country="Germany",
            competition_type=CompetitionType.LEAGUE,
            tier=1,
            source_codes={"football_data_co": "D1", "openfootball": "germany/1", "football_data_org": "BL1"},
            provider_names={"api_football": "Bundesliga"},
            licence=Licence.INTERNAL_RESEARCH_ONLY,
        ),
        League(
            key="FRA_LIGUE_1",
            name="Ligue 1",
            country="France",
            competition_type=CompetitionType.LEAGUE,
            tier=1,
            source_codes={"football_data_co": "F1", "openfootball": "france/1", "football_data_org": "FL1"},
            provider_names={"api_football": "Ligue 1"},
            licence=Licence.INTERNAL_RESEARCH_ONLY,
        ),
        League(
            key="INT_AFCON",
            name="Africa Cup of Nations",
            country="Africa (CAF)",
            competition_type=CompetitionType.CUP,
            tier=None,
            provider_names={"statsbomb_open": "Africa Cup of Nations"},
            licence=Licence.STATSBOMB_RESEARCH,
            notes=(
                "Cup tournament with a finite pool of teams. StatsBomb open data covers "
                "the 2023 edition completely (52 matches) including lineups and events."
            ),
        ),
        League(
            key="INT_WORLD_CUP",
            name="FIFA World Cup",
            country="International (FIFA)",
            competition_type=CompetitionType.CUP,
            tier=None,
            provider_names={"statsbomb_open": "FIFA World Cup"},
            licence=Licence.STATSBOMB_RESEARCH,
            notes=(
                "National-team knockout tournament. Team strength cannot be estimated "
                "from club ratings; a separate rating system is required. StatsBomb covers "
                "2018 and 2022 completely."
            ),
        ),
        League(
            key="INT_UCL",
            name="UEFA Champions League",
            country="Europe (UEFA)",
            competition_type=CompetitionType.CUP,
            tier=None,
            licence=Licence.INTERNAL_RESEARCH_ONLY,
            notes=(
                "NO USABLE FREE DATA. No free source provides a usable history of "
                "Champions League matches with team strength context. The plan is to "
                "infer strength from each club's DOMESTIC league record and apply generic "
                "competition adjustments (neutral venue, travel, two-legged ties, "
                "strength of schedule). This is inference from domestic form, not UCL "
                "data, and the app must label it as such."
            ),
        ),
    )
}


def get_league(key: str) -> League:
    """Look up a competition by its internal key.

    Args:
        key: Internal identifier, e.g. ``"ENG_PL"``. Case-insensitive.

    Returns:
        The matching :class:`League`.

    Raises:
        KeyError: If the key is unknown. The message lists valid keys, because a
            typo here is a common and easily-missed failure.
    """
    try:
        return LEAGUES[key.upper()]
    except KeyError as exc:
        valid = ", ".join(sorted(LEAGUES))
        raise KeyError(f"Unknown league key {key!r}. Valid keys: {valid}") from exc


def list_leagues(
    *,
    competition_type: CompetitionType | None = None,
    country: str | None = None,
    tier: int | None = None,
) -> list[League]:
    """List known competitions, optionally filtered.

    Args:
        competition_type: Restrict to league or cup competitions.
        country: Case-insensitive country match.
        tier: Restrict to a competition tier (1 = top flight).

    Returns:
        Competitions sorted by key.
    """
    results = list(LEAGUES.values())
    if competition_type is not None:
        results = [x for x in results if x.competition_type is competition_type]
    if country is not None:
        results = [x for x in results if x.country.lower() == country.lower()]
    if tier is not None:
        results = [x for x in results if x.tier == tier]
    return sorted(results, key=lambda x: x.key)


def season_codes(start_year: int, end_year: int) -> list[str]:
    """Generate football-data.co.uk season codes for an inclusive year range.

    Args:
        start_year: First season's starting year, e.g. ``2015`` for 2015/16.
        end_year: Last season's starting year, e.g. ``2025`` for 2025/26.

    Returns:
        Four-character codes like ``["1516", ..., "2526"]``.

    Raises:
        ValueError: If the range is empty or reversed.

    Examples:
        >>> season_codes(2023, 2025)
        ['2324', '2425', '2526']
    """
    if end_year < start_year:
        raise ValueError(f"end_year ({end_year}) must be >= start_year ({start_year})")
    return [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(start_year, end_year + 1)]


def season_start_year(code: str) -> int:
    """Return the starting calendar year for a season code.

    Two forms are accepted, because different competitions are addressed
    differently by the live providers:

    * The project's split-season code, ``"2425"`` -> ``2024``, used by domestic
      leagues. The second pair must be exactly one greater than the first.
    * A bare four-digit calendar year, ``"2022"`` -> ``2022``, used by annual
      tournaments such as the World Cup.

    A code such as ``"2021"`` is ambiguous (it is both the split season 2020/21
    and the calendar year 2021); it is read as the split season to stay
    consistent with :func:`season_label`.

    Args:
        code: Four-character code such as ``"2425"`` or ``"2022"``.

    Returns:
        The starting year as an integer.

    Raises:
        ValueError: If the code is not four digits or matches neither form.

    Examples:
        >>> season_start_year("2425")
        2024
        >>> season_start_year("2022")
        2022
    """
    if len(code) != 4 or not code.isdigit():
        raise ValueError(f"Season code must be four digits, got {code!r}")

    start_pair = int(code[:2])
    end_pair = int(code[2:])
    if end_pair == (start_pair + 1) % 100:
        century = 1900 if start_pair >= 90 else 2000
        return century + start_pair

    year = int(code)
    if 1900 <= year <= 2100:
        return year
    raise ValueError(f"{code!r} is neither a valid split season nor a plausible calendar year")


def season_label(code: str) -> str:
    """Convert a four-character season code to a ``YYYY/YY`` label.

    A season code is two two-digit years: the start year followed by the end
    year, so the second pair must be exactly one greater than the first
    (wrapping at century boundaries). That gives a precise validity rule and
    catches the realistic mistake of passing a bare year -- ``"2024"`` fails
    because 24 is not 20 + 1, whereas the genuine ``"2021"`` (2020/21) passes.

    Args:
        code: Four-character code such as ``"2425"``.

    Returns:
        Human-readable label such as ``"2024/25"``.

    Raises:
        ValueError: If the code is not four digits, or is not a valid season.

    Examples:
        >>> season_label("2425")
        '2024/25'
        >>> season_label("9900")
        '1999/00'
        >>> season_label("2024")
        Traceback (most recent call last):
            ...
        ValueError: '2024' is not a valid season code ...
    """
    if len(code) != 4 or not code.isdigit():
        raise ValueError(f"Season code must be four digits, got {code!r}")

    start_pair = int(code[:2])
    end_pair = int(code[2:])
    expected_end = (start_pair + 1) % 100
    if end_pair != expected_end:
        raise ValueError(
            f"{code!r} is not a valid season code: the second pair must be one greater than "
            f"the first (expected {expected_end:02d}, got {end_pair:02d}). A four-digit year "
            f"was probably passed where a season code was expected."
        )

    # Two-digit seasons are ambiguous across a century boundary; 97-99 are late
    # 1990s seasons, 00+ are 2000s onwards.
    century = 1900 if start_pair >= 90 else 2000
    return f"{century + start_pair}/{code[2:]}"

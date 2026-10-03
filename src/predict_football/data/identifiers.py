"""Deterministic identifiers shared by providers, cleaning and storage.

Match identity is the join key that lets us combine data from different sources
about the same fixture. It must therefore be derived only from information every
source agrees on: the competition, the date, and the two teams.

Deliberately *not* used: any provider's own match number, and any provider's
team-name spelling. Both differ between sources, which would silently split one
real match into two rows.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

#: Length of the hex digest used for match identifiers.
_MATCH_ID_LENGTH = 16

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def slugify(value: str) -> str:
    """Convert a name to a lowercase ASCII slug.

    Accents are stripped rather than transliterated aggressively, so "Atlético
    Madrid" becomes "atletico madrid". This is deliberate: consistency matters
    more than typographic fidelity for a join key.

    Args:
        value: Any human-readable name.

    Returns:
        A slug such as ``"manchester united"``.

    Examples:
        >>> slugify("Atlético Madrid")
        'atletico madrid'
        >>> slugify("  FC   Schalke 04!  ")
        'fc schalke 04'
    """
    # Lowercase BEFORE the character filter: the strip pattern is [a-z0-9], so
    # an uppercase "M" would otherwise be treated as a separator and deleted,
    # silently turning "Madrid" into "adrid".
    normalised = unicodedata.normalize("NFKD", value).lower()
    ascii_only = normalised.encode("ascii", "ignore").decode("ascii")
    slug = _SLUG_STRIP.sub(" ", ascii_only).strip()
    return re.sub(r"\s+", " ", slug)


def normalise_whitespace(value: str) -> str:
    """Collapse runs of whitespace and strip the ends of a string.

    Args:
        value: Input string.

    Returns:
        Trimmed string with single spaces between words.
    """
    return re.sub(r"\s+", " ", value).strip()


def make_team_id(team_name: str) -> str:
    """Build the canonical team identifier from a display name.

    Args:
        team_name: Canonical team name, e.g. ``"Manchester United"``.

    Returns:
        Slug identifier, e.g. ``"manchester_united"``.

    Examples:
        >>> make_team_id("Manchester United")
        'manchester_united'
    """
    return slugify(team_name).replace(" ", "_")


def make_match_id(
    league_key: str,
    match_date: str,
    home_team: str,
    away_team: str,
) -> str:
    """Build a deterministic, provider-independent match identifier.

    The identifier is a truncated SHA-256 digest of the natural key
    ``league|date|home|away|. It is stable across runs and across providers, so
    the same real-world match ingested from two sources produces one row rather
    than two.

    Names must already be canonical. Resolving provider-specific spellings
    ("Man United" to "Manchester United") is the job of
    :class:`~predict_football.data.teams.TeamResolver`, which providers apply
    before hashing. Doing it here instead would mean a registry lookup inside a
    pure function, and an unresolved name would silently produce an identity
    that looks valid.

    Args:
        league_key: Internal competition key, e.g. ``"ENG_PL"``.
        match_date: ISO ``YYYY-MM-DD`` match date.
        home_team: Canonical home team name.
        away_team: Canonical away team name.

    Returns:
        A 16-character hexadecimal identifier.

    Examples:
        >>> make_match_id("ENG_PL", "2024-08-16", "Liverpool", "Manchester United")
        '3b1f0c9a4d7e2568'
    """
    natural_key = "|".join(
        (
            league_key.strip().upper(),
            str(match_date).strip(),
            slugify(home_team),
            slugify(away_team),
        )
    )
    digest = hashlib.sha256(natural_key.encode("utf-8")).hexdigest()
    return digest[:_MATCH_ID_LENGTH]


def make_event_id(match_id: str, period: int, minute: int, second: int, team: str) -> str:
    """Build a deterministic event identifier.

    Event data has no natural primary key across providers, so we synthesise a
    stable one from the event's position in the match. Collisions are possible
    when one team records two events in the same second; those are tolerated
    because the identifier is only used for de-duplication within a match.

    Args:
        match_id: Owning match identifier.
        period: Period number, 1 or 2.
        minute: Match minute.
        second: Second within the minute.
        team: Team responsible for the event.

    Returns:
        A deterministic event identifier.
    """
    natural_key = f"{match_id}|{period}|{minute}|{second}|{slugify(team)}"
    return hashlib.sha256(natural_key.encode("utf-8")).hexdigest()[:_MATCH_ID_LENGTH]

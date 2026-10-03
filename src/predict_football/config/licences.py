"""Data source licences, encoded as code so the rules cannot be forgotten.

Licence restrictions are easy to state in a README and easy to violate in a
`SELECT *`. This module makes them executable:

* :data:`LICENCES` records, per source, whether raw data may be committed,
  served from a deployed app, or used commercially.
* :func:`assert_can_ship` raises when a caller tries to do something a licence
  forbids.

See ``docs/DATA_SOURCES.md`` for the full verified findings and source URLs.

HONEST LIMITATION: these records reflect a manual review of each provider's
published terms as of ``REVIEWED_ON``. Terms change silently. Re-verify before
any public release, and get written permission rather than relying on this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from urllib.parse import urlparse


class Licence(Enum):
    """Licence class governing how a source's data may be used.

    Attributes:
        CC0: Public domain dedication. No restrictions.
        MIT: Permissive licence. Commercial use allowed, notice retained.
        CC_BY_4: Creative Commons Attribution 4.0. Commercial use allowed with
            attribution.
        STATSBOMB_RESEARCH: StatsBomb Public Data User Agreement. Research use;
            redistribution of the raw data forbidden; commercial exploitation of
            derived analysis forbidden; the StatsBomb logo is required on any
            published analysis.
        INTERNAL_RESEARCH_ONLY: No published licence at all. Personal and
            internal research is the only defensible use.
    """

    CC0 = "CC0-1.0"
    MIT = "MIT"
    CC_BY_4 = "CC BY 4.0"
    STATSBOMB_RESEARCH = "StatsBomb Public Data User Agreement"
    INTERNAL_RESEARCH_ONLY = "No published licence"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass(frozen=True)
class LicencePolicy:
    """What a licence permits, in terms the code can enforce.

    Attributes:
        name: Short provider-agnostic licence label.
        url: Provider's terms or licence file.
        summary: Plain-English statement of the binding restrictions.
        may_commit_raw_data: Whether raw files may live in a public repository.
        may_serve_from_public_app: Whether a deployed app may return this data.
        may_use_commercially: Whether derived analysis may be monetised.
        attribution: Attribution string required in published output.
    """

    name: str
    url: str
    summary: str
    may_commit_raw_data: bool
    may_serve_from_public_app: bool
    may_use_commercially: bool
    attribution: str | None = None


#: Date on which every policy below was manually verified against source.
REVIEWED_ON = date(2026, 10, 2)

#: Provider key -> licence policy. Provider keys match those in
#: :mod:`predict_football.config.leagues` source-code mappings.
LICENCES: dict[str, LicencePolicy] = {
    "openfootball": LicencePolicy(
        name="CC0-1.0",
        url="https://github.com/openfootball/england/blob/master/LICENSE.md",
        summary="Public domain dedication. No restrictions on use or redistribution.",
        may_commit_raw_data=True,
        may_serve_from_public_app=True,
        may_use_commercially=True,
        attribution="Optional; courtesy of the openfootball contributors.",
    ),
    "wyscout_open": LicencePolicy(
        name="CC BY 4.0",
        url="https://figshare.com/collections/Soccer_match_event_dataset/4415000",
        summary=(
            "Creative Commons Attribution 4.0. Commercial use and redistribution are "
            "permitted provided the required attribution is given. Contains full event "
            "streams, lineups, benches and substitutions, but NO shot xG."
        ),
        may_commit_raw_data=True,
        may_serve_from_public_app=True,
        may_use_commercially=True,
        attribution=(
            "Pappalardo, L.; Massucco, E. (2019), figshare, Soccer match event dataset. "
            "https://doi.org/10.6084/m9.figshare.c.4415000.v5"
        ),
    ),
    "skillcorner_open": LicencePolicy(
        name="MIT",
        url="https://github.com/SkillCorner/opendata/blob/master/LICENSE",
        summary=(
            "Permissive MIT licence. Commercial use and redistribution permitted with "
            "the notice retained. Contains broadcast tracking plus EPV (expected "
            "possession value) for 10 matches. EPV is not shot xG."
        ),
        may_commit_raw_data=True,
        may_serve_from_public_app=True,
        may_use_commercially=True,
        attribution="SkillCorner open data, MIT licence.",
    ),
    "football_data_co": LicencePolicy(
        name="No published licence",
        url="https://www.football-data.co.uk/help_footballdata.php",
        summary=(
            "The site publishes no licence, terms page or redistribution permission for "
            "its CSV files, and the operator states he holds copyright in official "
            "league match data. Richest free source of history and bookmaker odds, but "
            "the absence of a licence means there is nothing granting us redistribution "
            "rights. Treat as personal and internal use only."
        ),
        may_commit_raw_data=False,
        may_serve_from_public_app=False,
        may_use_commercially=False,
        attribution="Data from football-data.co.uk (free, ad-supported).",
    ),
    "statsbomb_open": LicencePolicy(
        name="StatsBomb Public Data User Agreement",
        url="https://github.com/hudl/open-data/blob/master/LICENSE.pdf",
        summary=(
            "Custom agreement, NOT open source. Clause 1.2.1 forbids editing, "
            "distributing, reproducing, selling or otherwise providing the data to any "
            "third party. Clause 1.2.2 forbids commercially exploiting the data or any "
            "analysis derived from it. Clause 1.4 REQUIRES the StatsBomb brand logo on "
            "any publication of analysis. Research use and published analysis are "
            "permitted. The only free source of per-shot xG with timestamps and lineups."
        ),
        may_commit_raw_data=False,
        may_serve_from_public_app=False,
        may_use_commercially=False,
        attribution="Analysis based on StatsBomb Open Data. Requires the StatsBomb logo.",
    ),
    "clubelo": LicencePolicy(
        name="No published licence",
        url="https://clubelo.com/About",
        summary=(
            "No licence, attribution requirement or redistribution terms are published. "
            "The documented API endpoint also failed to respond when we tried it, so "
            "even the response schema is unverified. Usable as an internal feature; do "
            "not ship it."
        ),
        may_commit_raw_data=False,
        may_serve_from_public_app=False,
        may_use_commercially=False,
        attribution=None,
    ),
    "football_data_org": LicencePolicy(
        name="Free tier ToS",
        url="https://www.football-data.org/about",
        summary=(
            "Free tier is delayed scores only, 10 calls per minute, with mandatory "
            "attribution and single-application use. Clause 9.1 forbids referencing the "
            "data after cancelling a subscription. Live scores and lineups are paid tiers."
        ),
        may_commit_raw_data=False,
        may_serve_from_public_app=True,
        may_use_commercially=False,
        attribution="Football data provided by the Football-Data.org API",
    ),
    "api_football": LicencePolicy(
        name="Unverified",
        url="https://www.api-football.com/",
        summary=(
            "Terms could not be read: the documentation site returned HTTP 403 when we "
            "checked. Free tier is 100 requests per day, which is far too few to backfill "
            "a league. Treat the licence as unknown until the dashboard terms are read "
            "after registering."
        ),
        may_commit_raw_data=False,
        may_serve_from_public_app=False,
        may_use_commercially=False,
        attribution=None,
    ),
}


class LicenceViolation(RuntimeError):
    """Raised when an action would breach a source's licence."""


def licence_for(source: str) -> LicencePolicy:
    """Return the licence policy for a data source.

    Args:
        source: Provider key, e.g. ``"football_data_co"``.

    Returns:
        The matching :class:`LicencePolicy`.

    Raises:
        KeyError: If the source has not been reviewed. Unknown sources are an
            error rather than a default, because assuming permission is exactly
            the mistake this module exists to prevent.
    """
    try:
        return LICENCES[source]
    except KeyError as exc:
        known = ", ".join(sorted(LICENCES))
        raise KeyError(f"No licence review for source {source!r}. Reviewed sources: {known}") from exc


def assert_can_commit_raw_data(source: str) -> None:
    """Raise unless raw data from a source may be committed to the repository.

    Args:
        source: Provider key.

    Raises:
        LicenceViolation: If the licence forbids committing the raw data.
    """
    policy = licence_for(source)
    if not policy.may_commit_raw_data:
        raise LicenceViolation(
            f"{source}: raw data may not be committed to the repository. "
            f"Licence: {policy.name}. {policy.summary} Source terms: {policy.url}"
        )


def assert_can_serve_publicly(source: str) -> None:
    """Raise unless a deployed public app may return data from a source.

    Args:
        source: Provider key.

    Raises:
        LicenceViolation: If the licence forbids public serving.
    """
    policy = licence_for(source)
    if not policy.may_serve_from_public_app:
        raise LicenceViolation(
            f"{source}: a deployed public app may not serve this data. "
            f"Licence: {policy.name}. {policy.summary} Source terms: {policy.url}"
        )


def assert_commercial_use(source: str) -> None:
    """Raise unless derived analysis from a source may be used commercially.

    Args:
        source: Provider key.

    Raises:
        LicenceViolation: If commercial use is not permitted.
    """
    policy = licence_for(source)
    if not policy.may_use_commercially:
        raise LicenceViolation(
            f"{source}: commercial use is not permitted by the source licence "
            f"({policy.name}). {policy.summary} Source terms: {policy.url}"
        )


def attribution_for(source: str) -> str:
    """Return the attribution string required for a source.

    Args:
        source: Provider key.

    Returns:
        Attribution text, or a placeholder demanding one be supplied if the
        source has no recorded attribution requirement.
    """
    policy = licence_for(source)
    if policy.attribution:
        return policy.attribution
    return f"Source: {source} (https://{urlparse(policy.url).netloc}) - attribution required, confirm terms."


def redistribution_safe_sources() -> list[str]:
    """List sources whose data may legally be redistributed.

    Returns:
        Sorted provider keys that pass all three permission checks.
    """
    return sorted(
        key
        for key, policy in LICENCES.items()
        if policy.may_commit_raw_data and policy.may_serve_from_public_app
    )

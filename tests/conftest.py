"""Shared pytest fixtures.

All tests run offline. Where a provider needs HTTP, a fake session is injected,
so the suite never touches the network and never depends on a free source being
up. This is deliberate: a test that fails because a website is down teaches you
nothing about your code.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from predict_football.config.settings import Settings
from predict_football.data.providers.football_data_co import FootballDataCoProvider
from predict_football.data.repository import Database, MatchRepository
from predict_football.data.teams import TeamResolver


class FakeResponse:
    """Minimal stand-in for a ``requests`` response."""

    def __init__(self, content: bytes, status_code: int = 200) -> None:
        self.content = content
        self.status_code = status_code


class FakeSession:
    """Canned HTTP responses keyed by URL substring.

    Args:
        responses: Mapping of URL substring to response bytes or status code.
    """

    def __init__(self, responses: dict[str, Any]) -> None:
        self._responses = responses
        self.requested: list[str] = []
        self.calls: list[dict[str, Any]] = []

    def get(
        self,
        url: str,
        timeout: float | None = None,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> FakeResponse:
        """Return the canned response whose key appears in the URL.

        Args:
            url: Requested URL.
            timeout: Ignored.
            headers: Ignored, but recorded in :attr:`calls` so tests can assert
                an API key or auth header was sent.
            params: Query parameters, recorded for assertions. Canned responses
                are matched on the URL path only, so a provider that puts the
                season in ``params`` is still testable.
            **kwargs: Accepted and ignored, so a provider may pass extra
                arguments without breaking the fake.

        Returns:
            A :class:`FakeResponse`.

        Raises:
            AssertionError: If no canned response matches, so a test fails loudly
                rather than silently getting an empty frame.
        """
        self.requested.append(url)
        self.calls.append({"url": url, "headers": dict(headers or {}), "params": dict(params or {})})
        for fragment, payload in self._responses.items():
            if fragment in url:
                if isinstance(payload, int):
                    return FakeResponse(b"", payload)
                return FakeResponse(payload)
        raise AssertionError(f"FakeSession has no canned response for {url}")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Build isolated settings rooted in a temporary directory.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Settings with all directories created and network disabled.
    """
    resolved = Settings.from_env(project_root=tmp_path)
    object.__setattr__(resolved, "allow_network", True)
    resolved.ensure_directories()
    return resolved


@pytest.fixture
def offline_settings(tmp_path: Path) -> Settings:
    """Settings with network access disabled, for cache-miss tests.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Settings with ``allow_network`` False.
    """
    resolved = Settings.from_env(project_root=tmp_path)
    object.__setattr__(resolved, "allow_network", False)
    resolved.ensure_directories()
    return resolved


@pytest.fixture
def database(tmp_path: Path) -> Database:
    """In-memory database.

    Args:
        tmp_path: Pytest temporary directory, unused but keeps the fixture
            signature uniform with the others.

    Returns:
        An initialised in-memory :class:`Database`.
    """
    return Database(":memory:")


@pytest.fixture
def repository(database: Database) -> MatchRepository:
    """Repository backed by an in-memory database.

    Args:
        database: In-memory database fixture.

    Returns:
        A ready :class:`MatchRepository`.
    """
    return MatchRepository(database)


@pytest.fixture
def resolver() -> TeamResolver:
    """A fresh team resolver.

    Returns:
        A :class:`TeamResolver`.
    """
    return TeamResolver()


#: A realistic football-data.co.uk CSV body, SYNTHETIC and clearly labelled.
#: Real source CSVs are gitignored because the provider publishes no licence
#: permitting redistribution (see docs/DATA_SOURCES.md), so fixtures are built
#: from made-up rows that exercise the real column layout. No number here comes
#: from a real match.
FDCO_CSV = """Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,HS,AS,HST,AST,HF,AF,HC,AC,HY,AY,HR,AR,AvgH,AvgD,AvgA,AvgCH,AvgCD,AvgCA,B365CH,B365CD,B365CA
E0,11/08/2023,20:00,Liverpool,Bournemouth,1,1,D,1,0,N,15,7,5,2,7,8,6,4,1,2,0,0,2.05,3.55,3.62,1.95,3.65,3.55,1.95,3.60,3.60
E0,12/08/2023,12:30,Arsenal,Nott'm Forest,2,1,H,1,0,N,20,7,6,3,10,7,10,3,0,2,0,0,1.62,3.90,5.20,1.57,4.00,5.50,1.55,4.00,5.50
E0,12/08/2023,15:00,Bournemouth,West Ham,0,0,D,0,0,N,9,14,2,5,11,9,3,6,3,1,0,0,2.60,3.30,2.70,2.70,3.40,2.60,2.75,3.40,2.55
E0,19/08/2023,15:00,Man City,Newcastle,2,0,H,1,0,N,17,10,7,3,8,10,7,2,1,2,0,0,1.35,5.50,8.50,1.33,5.75,8.50,1.33,5.50,8.50
E0,19/08/2023,12:30,Man United, Wolverhampton Wanderers,1,0,H,0,0,N,14,8,5,2,9,11,5,3,2,1,0,0,1.45,4.75,6.50,1.44,4.75,6.75,1.44,4.60,6.75
"""


def fdco_bytes(body: str = FDCO_CSV) -> bytes:
    """Encode a fixture CSV body as bytes.

    Args:
        body: CSV text.

    Returns:
        UTF-8 encoded bytes.
    """
    return body.encode("utf-8")


@pytest.fixture
def fdco_session() -> FakeSession:
    """Fake session serving the synthetic football-data.co.uk CSV.

    Returns:
        A :class:`FakeSession` matching any E0 CSV URL.
    """
    return FakeSession({"E0.csv": fdco_bytes()})


@pytest.fixture
def fdco_provider(settings: Settings, fdco_session: FakeSession, resolver: TeamResolver) -> FootballDataCoProvider:
    """Provider wired to a fake session and a temporary cache.

    Args:
        settings: Isolated settings fixture.
        fdco_session: Fake session fixture.
        resolver: Team resolver fixture.

    Returns:
        A configured provider that never touches the network.
    """
    return FootballDataCoProvider(settings=settings, teams=resolver, session=fdco_session)


@pytest.fixture
def sample_matches() -> pd.DataFrame:
    """A small canonical match frame for repository tests.

    Returns:
        Frame with four matches, two completed and two pending.
    """
    return pd.DataFrame(
        {
            "match_id": ["m1", "m2", "m3", "m4"],
            "source": ["test"] * 4,
            "league_key": ["ENG_PL"] * 4,
            "competition_type": ["league"] * 4,
            "season": ["2023/24"] * 4,
            "season_code": ["2324"] * 4,
            "match_date": pd.to_datetime(["2023-08-11", "2023-08-12", "2023-08-12", "2023-08-19"]),
            "home_team": ["Liverpool", "Arsenal", "Manchester City", "Chelsea"],
            "away_team": ["Bournemouth", "Nott'm Forest", "Newcastle", "Wolverhampton Wanderers"],
            "result": ["D", "H", "H", None],
            "home_goals": [1, 2, 2, None],
            "away_goals": [1, 1, 0, None],
            "home_shots": [15, 20, 17, None],
            "away_shots": [7, 7, 10, None],
            "home_xg": [1.4, 2.1, 1.9, None],
            "away_xg": [0.9, 0.7, 0.4, None],
        }
    )


def read_fixture(name: str) -> str:
    """Read a text fixture from the tests directory.

    Args:
        name: Filename relative to ``tests/data``.

    Returns:
        File contents.

    Raises:
        FileNotFoundError: If the fixture is missing.
    """
    path = Path(__file__).parent / "data" / name
    if not path.is_file():
        raise FileNotFoundError(f"Test fixture not found: {path}")
    return path.read_text(encoding="utf-8")


def write_csv_fixture(df: pd.DataFrame) -> bytes:
    """Serialise a frame to CSV bytes for provider injection.

    Args:
        df: Frame to serialise.

    Returns:
        CSV bytes.
    """
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


#: Extra expected home goals in the synthetic league, giving home advantage a
#: known sign so model and tuning tests can assert a direction rather than noise.
SYNTHETIC_HOME_EDGE = 0.35


@pytest.fixture
def synthetic_league() -> pd.DataFrame:
    """A small synthetic round-robin league with a home edge.

    Team quality increases with the index, so a fitted model has a real ordering
    to recover. The home side is given :data:`SYNTHETIC_HOME_EDGE` extra expected
    goals, so home advantage has a sign the tests can check.

    Returns:
        Frame with six clubs over thirty matchdays, one row per match.
    """
    import numpy as np

    rng = np.random.default_rng(20260810)
    n_teams = 6
    rounds = 12
    strengths = np.linspace(0.9, 2.3, n_teams)
    teams = [f"T{i:02d}" for i in range(n_teams)]

    rows: list[dict[str, object]] = []
    for round_index in range(rounds):
        order = list(rng.permutation(n_teams))
        for a, b in zip(order[::2], order[1::2], strict=True):
            home_goals = int(rng.poisson(strengths[a] + SYNTHETIC_HOME_EDGE))
            away_goals = int(rng.poisson(strengths[b]))
            result = "H" if home_goals > away_goals else "A" if home_goals < away_goals else "D"
            rows.append(
                {
                    "match_id": f"m{round_index:03d}_{a}_{b}",
                    "match_date": pd.Timestamp("2021-01-01") + pd.Timedelta(days=7 * round_index),
                    "season": "2020/21",
                    "home_team": teams[a],
                    "away_team": teams[b],
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "result": result,
                }
            )
    return pd.DataFrame(rows)

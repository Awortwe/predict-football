"""SQLite storage layer.

Deliberately plain SQL over the standard-library ``sqlite3`` driver rather than
an ORM. Reasons:

* We store a handful of wide, append-mostly tables. There is no object graph to
  map and no lazy loading to optimise, so an ORM would add a dependency and a
  layer of indirection for nothing.
* Every query the project needs is visible in one place and easy to audit -- which
  matters when the correctness of a query is the difference between an honest
  backtest and a leaky one.
* Moving to Postgres later means swapping the ``INSERT``/``SELECT`` dialect in
  this module. The :class:`MatchRepository` interface above it stays put, and so
  does everything that consumes it.

The interface is deliberately narrow so that swap stays cheap.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

from predict_football.data.schema import MATCH_COLUMN_NAMES, SCHEMA_VERSION
from predict_football.data.teams import TeamResolver

logger = logging.getLogger(__name__)

#: Columns that are the natural key of a match row. Everything else is
#: upserted on conflict, which makes re-ingesting a season idempotent.
_MATCH_CONFLICT_KEY = ("source", "match_id")

#: Extra provider columns we keep alongside the canonical schema rather than
#: dropping, because they are needed to fetch child records later.
_MATCH_EXTRA_COLUMNS: tuple[str, ...] = ("statsbomb_match_id", "teams_unresolved", "has_lineups")

_TEAM_STAT_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("shots", "home_shots", "away_shots"),
    ("shots_on_target", "home_shots_on_target", "away_shots_on_target"),
    ("corners", "home_corners", "away_corners"),
    ("fouls", "home_fouls", "away_fouls"),
    ("yellow_cards", "home_yellow", "away_yellow"),
    ("red_cards", "home_red", "away_red"),
    ("xg", "home_xg", "away_xg"),
)


_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS leagues (
    league_key        TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    country           TEXT,
    competition_type  TEXT NOT NULL,
    tier              INTEGER,
    notes             TEXT
);

CREATE TABLE IF NOT EXISTS teams (
    team_id        TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    short_name     TEXT,
    country        TEXT,
    in_registry    INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS matches (
    match_id        TEXT PRIMARY KEY,
    source          TEXT NOT NULL,
    league_key      TEXT NOT NULL,
    competition_type TEXT NOT NULL,
    season          TEXT NOT NULL,
    season_code     TEXT,
    match_date      TEXT NOT NULL,
    home_team_id    TEXT NOT NULL,
    away_team_id    TEXT NOT NULL,
    home_team       TEXT NOT NULL,
    away_team       TEXT NOT NULL,
    result          TEXT,
    ingested_at     TEXT NOT NULL,
    FOREIGN KEY (home_team_id) REFERENCES teams(team_id),
    FOREIGN KEY (away_team_id) REFERENCES teams(team_id)
);

CREATE TABLE IF NOT EXISTS match_targets (
    match_id     TEXT PRIMARY KEY,
    home_goals   INTEGER,
    away_goals   INTEGER,
    result       TEXT,
    result_et    TEXT,
    pens_home    INTEGER,
    pens_away    INTEGER,
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS match_stats (
    match_id       TEXT NOT NULL,
    side           TEXT NOT NULL CHECK (side IN ('home','away')),
    team_id        TEXT NOT NULL,
    shots          REAL,
    shots_on_target REAL,
    corners        REAL,
    fouls          REAL,
    yellow_cards   REAL,
    red_cards      REAL,
    xg             REAL,
    PRIMARY KEY (match_id, side),
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS match_odds (
    match_id        TEXT NOT NULL,
    bookmaker       TEXT NOT NULL,
    market          TEXT NOT NULL,
    odds_home_open  REAL,
    odds_draw_open  REAL,
    odds_away_open  REAL,
    odds_home_close REAL,
    odds_draw_close REAL,
    odds_away_close REAL,
    overround       REAL,
    price_basis     TEXT,
    PRIMARY KEY (match_id, bookmaker, market),
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS match_metadata (
    match_id  TEXT PRIMARY KEY,
    venue     TEXT,
    referee   TEXT,
    match_week INTEGER,
    provider_match_id TEXT,
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS events (
    event_id   TEXT PRIMARY KEY,
    match_id   TEXT NOT NULL,
    period     INTEGER NOT NULL,
    minute     INTEGER NOT NULL,
    second     INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    team       TEXT,
    player     TEXT,
    xg         REAL,
    shot_outcome TEXT,
    shot_body_part TEXT,
    shot_type  TEXT,
    location_x REAL,
    location_y REAL,
    end_location_x REAL,
    end_location_y REAL,
    pass_length REAL,
    pass_angle  REAL,
    shot_assist TEXT,
    related_event TEXT,
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS lineups (
    match_id      TEXT NOT NULL,
    team_id       TEXT NOT NULL,
    player        TEXT NOT NULL,
    player_id     TEXT,
    position      TEXT,
    jersey_number INTEGER,
    is_starter    INTEGER,
    is_captain    INTEGER,
    formation     TEXT,
    PRIMARY KEY (match_id, team_id, player),
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS ix_matches_league_date ON matches (league_key, match_date);
CREATE INDEX IF NOT EXISTS ix_matches_season      ON matches (season);
CREATE INDEX IF NOT EXISTS ix_matches_teams      ON matches (home_team_id, away_team_id);
CREATE INDEX IF NOT EXISTS ix_matches_result     ON matches (result);
CREATE INDEX IF NOT EXISTS ix_events_match_time  ON events (match_id, period, minute, second);
CREATE INDEX IF NOT EXISTS ix_odds_match         ON match_odds (match_id);
"""


class Database:
    """Thin SQLite wrapper handling connections and schema creation.

    Args:
        path: Database file path. Use ``":memory:"`` for an ephemeral database,
            which is what the tests do.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._memory_conn: sqlite3.Connection | None = None

    def connect(self) -> sqlite3.Connection:
        """Return a connection, reusing one for in-memory databases.

        An in-memory SQLite database exists only as long as its connection, so
        each new connection would otherwise see an empty schema.

        Returns:
            A configured connection.
        """
        if self.path == ":memory:":
            if self._memory_conn is None:
                self._memory_conn = self._new_connection()
            return self._memory_conn
        return self._new_connection()

    def _new_connection(self) -> sqlite3.Connection:
        """Create a connection with row access by name and FK enforcement on.

        Returns:
            A configured connection.
        """
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @contextmanager
    def cursor(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection inside a transaction, committing on success.

        Yields:
            An open connection.
        """
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            if self.path != ":memory:":
                conn.close()

    def initialise(self) -> None:
        """Create tables and indexes if they do not exist, and stamp the schema."""
        with self.cursor() as conn:
            conn.executescript(_DDL)
            conn.execute(
                "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (SCHEMA_VERSION,),
            )

    def schema_version(self) -> str | None:
        """Return the recorded schema version.

        Returns:
            The version string, or ``None`` for an uninitialised database.
        """
        with self.cursor() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        return row["value"] if row else None

    def row_counts(self) -> dict[str, int]:
        """Count rows in every table.

        Returns:
            Mapping of table name to row count, for a quick health check.
        """
        counts: dict[str, int] = {}
        with self.cursor() as conn:
            tables = [
                r["name"]
                for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
            ]
            for table in tables:
                counts[table] = int(conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"])
        return counts

    def close(self) -> None:
        """Release the in-memory connection, if any."""
        if self._memory_conn is not None:
            self._memory_conn.close()
            self._memory_conn = None


class MatchRepository:
    """Read/write interface to the normalised match store.

    Callers never see SQL, which is what keeps the eventual Postgres migration
    to a single module.

    Args:
        db: Database wrapper.
    """

    def __init__(self, db: Database) -> None:
        self._db = db
        self._db.initialise()

    # --- writes -------------------------------------------------------------

    def upsert_teams(
        self, names: list[str], *, in_registry: bool | None = None, resolver: TeamResolver | None = None
    ) -> int:
        """Insert teams that do not yet exist.

        Args:
            names: Canonical team names.
            in_registry: Whether these names came from the curated registry. When
                ``None`` (the default) each name is checked against the registry,
                which is what makes the review queue meaningful. Passing a
                value forces the flag for every name.
            resolver: Registry to check against. Defaults to a fresh
                :class:`TeamResolver`.

        Returns:
            Number of teams inserted.
        """
        from predict_football.data.identifiers import make_team_id

        if not names:
            return 0
        checker = resolver if in_registry is None else None
        if in_registry is None and checker is None:
            checker = TeamResolver()

        inserted = 0
        with self._db.cursor() as conn:
            for name in dict.fromkeys(n for n in names if n and str(n).strip()):
                team_id = make_team_id(str(name))
                registered = in_registry if in_registry is not None else bool(
                    checker.is_known(str(name))
                )
                cursor = conn.execute(
                    "INSERT OR IGNORE INTO teams(team_id, canonical_name, in_registry) VALUES (?, ?, ?)",
                    (team_id, str(name), 1 if registered else 0),
                )
                inserted += cursor.rowcount if cursor.rowcount > 0 else 0
        return inserted

    def upsert_league(self, league_key: str, *, name: str, country: str | None,
                      competition_type: str, tier: int | None, notes: str | None = None) -> None:
        """Record a competition's configuration in the database.

        Args:
            league_key: Internal competition key.
            name: Human-readable name.
            country: Governing country.
            competition_type: ``"league"`` or ``"cup"``.
            tier: Competition tier.
            notes: Free-text notes.
        """
        with self._db.cursor() as conn:
            conn.execute(
                "INSERT INTO leagues(league_key, name, country, competition_type, tier, notes) "
                "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(league_key) DO UPDATE SET "
                "name = excluded.name, country = excluded.country, "
                "competition_type = excluded.competition_type, tier = excluded.tier, notes = excluded.notes",
                (league_key, name, country, competition_type, tier, notes),
            )

    def upsert_matches(self, df: pd.DataFrame) -> int:
        """Insert or update match rows and their dependent tables.

        Idempotent: re-ingesting a season updates rows in place rather than
        duplicating them.

        Args:
            df: Canonical match frame.

        Returns:
            Number of match rows written.

        Raises:
            ValueError: If required columns are missing.
        """
        required = {"match_id", "source", "league_key", "competition_type", "season", "match_date",
                    "home_team", "away_team"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"upsert_matches: frame is missing {sorted(missing)}")

        from predict_football.data.identifiers import make_team_id

        now = pd.Timestamp.utcnow().isoformat()
        teams = pd.concat([df["home_team"], df["away_team"]]).dropna().astype(str).unique().tolist()
        self.upsert_teams(teams)

        written = 0
        with self._db.cursor() as conn:
            for row in df.itertuples(index=False):
                record = row._asdict()
                conn.execute(
                    """
                    INSERT INTO matches(match_id, source, league_key, competition_type, season,
                                        season_code, match_date, home_team_id, away_team_id,
                                        home_team, away_team, result, ingested_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(match_id) DO UPDATE SET
                        season = excluded.season,
                        match_date = excluded.match_date,
                        result = excluded.result
                    """,
                    (
                        record["match_id"], _as_str(record["source"]), _as_str(record["league_key"]),
                        _as_str(record["competition_type"]), _as_str(record["season"]),
                        _as_str(record.get("season_code")),
                        _as_str(record["match_date"])[:19] if record.get("match_date") is not None else None,
                        make_team_id(str(record["home_team"])), make_team_id(str(record["away_team"])),
                        _as_str(record["home_team"]), _as_str(record["away_team"]),
                        _as_str(record.get("result")), now,
                    ),
                )
                written += 1

                home_goals = _as_int(record.get("home_goals"))
                away_goals = _as_int(record.get("away_goals"))
                if home_goals is not None or away_goals is not None:
                    conn.execute(
                        "INSERT INTO match_targets(match_id, home_goals, away_goals, result, result_et, "
                        "pens_home, pens_away) VALUES (?, ?, ?, ?, ?, ?, ?) "
                        "ON CONFLICT(match_id) DO UPDATE SET home_goals = excluded.home_goals, "
                        "away_goals = excluded.away_goals, result = excluded.result",
                        (
                            record["match_id"], home_goals, away_goals, _as_str(record.get("result")),
                            _as_str(record.get("result_et")), _as_int(record.get("pens_home")),
                            _as_int(record.get("pens_away")),
                        ),
                    )

                # pd.NA is not None, so test for missingness explicitly rather
                # than with `is not None`.
                if any(_as_str(record.get(column)) is not None or _as_int(record.get(column)) is not None
                       for column in ("venue", "referee", "match_week")):
                    conn.execute(
                        "INSERT INTO match_metadata(match_id, venue, referee, match_week, provider_match_id) "
                        "VALUES (?, ?, ?, ?, ?) ON CONFLICT(match_id) DO UPDATE SET "
                        "venue = COALESCE(excluded.venue, match_metadata.venue), "
                        "referee = COALESCE(excluded.referee, match_metadata.referee), "
                        "match_week = COALESCE(excluded.match_week, match_metadata.match_week), "
                        "provider_match_id = COALESCE(excluded.provider_match_id, match_metadata.provider_match_id)",
                        (
                            record["match_id"], _as_str(record.get("venue")), _as_str(record.get("referee")),
                            _as_int(record.get("match_week")), _as_str(record.get("statsbomb_match_id")),
                        ),
                    )

                home_id = make_team_id(str(record["home_team"]))
                away_id = make_team_id(str(record["away_team"]))
                for side, team_id in (("home", home_id), ("away", away_id)):
                    values = [
                        # Positional order must match the INSERT column list
                        # below, which is why this is not a dict comprehension.
                        _as_float(record.get(home_col if side == "home" else away_col))
                        for _, home_col, away_col in _TEAM_STAT_COLUMNS
                    ]
                    conn.execute(
                        "INSERT INTO match_stats(match_id, side, team_id, shots, shots_on_target, corners, "
                        "fouls, yellow_cards, red_cards, xg) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                        "ON CONFLICT(match_id, side) DO UPDATE SET "
                        "shots = COALESCE(excluded.shots, match_stats.shots), "
                        "shots_on_target = COALESCE(excluded.shots_on_target, match_stats.shots_on_target), "
                        "corners = COALESCE(excluded.corners, match_stats.corners), "
                        "fouls = COALESCE(excluded.fouls, match_stats.fouls), "
                        "yellow_cards = COALESCE(excluded.yellow_cards, match_stats.yellow_cards), "
                        "red_cards = COALESCE(excluded.red_cards, match_stats.red_cards), "
                        "xg = COALESCE(excluded.xg, match_stats.xg)",
                        (record["match_id"], side, team_id, *values),
                    )
        logger.info("Wrote %d match row(s)", written)
        return written

    def upsert_odds(self, df: pd.DataFrame) -> int:
        """Insert or update bookmaker odds.

        Args:
            df: Cleaned long odds frame.

        Returns:
            Number of rows written.
        """
        if df is None or df.empty:
            return 0
        written = 0
        with self._db.cursor() as conn:
            for record in df.to_dict("records"):
                conn.execute(
                    "INSERT INTO match_odds(match_id, bookmaker, market, odds_home_open, odds_draw_open, "
                    "odds_away_open, odds_home_close, odds_draw_close, odds_away_close, overround, price_basis) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(match_id, bookmaker, market) DO UPDATE SET "
                    "odds_home_open = excluded.odds_home_open, odds_draw_open = excluded.odds_draw_open, "
                    "odds_away_open = excluded.odds_away_open, "
                    "odds_home_close = excluded.odds_home_close, odds_draw_close = excluded.odds_draw_close, "
                    "odds_away_close = excluded.odds_away_close, "
                    "overround = excluded.overround, price_basis = excluded.price_basis",
                    (
                        record.get("match_id"), _as_str(record.get("bookmaker")) or "avg",
                        _as_str(record.get("market")) or "1x2",
                        _as_float(record.get("odds_home_open")), _as_float(record.get("odds_draw_open")),
                        _as_float(record.get("odds_away_open")), _as_float(record.get("odds_home_close")),
                        _as_float(record.get("odds_draw_close")), _as_float(record.get("odds_away_close")),
                        _as_float(record.get("overround")), _as_str(record.get("price_basis")),
                    ),
                )
                written += 1
        logger.info("Wrote %d odds row(s)", written)
        return written

    def upsert_events(self, df: pd.DataFrame) -> int:
        """Insert or update match events.

        Args:
            df: Cleaned canonical event frame.

        Returns:
            Number of rows written.
        """
        if df is None or df.empty:
            return 0
        written = 0
        columns = [
            "event_id", "match_id", "period", "minute", "second", "event_type", "team", "player", "xg",
            "shot_outcome", "shot_body_part", "shot_type", "location_x", "location_y", "end_location_x",
            "end_location_y", "pass_length", "pass_angle", "shot_assist", "related_event",
        ]
        present = [c for c in columns if c in df.columns]
        placeholders = ", ".join("?" for _ in present)
        statement = (
            f"INSERT OR REPLACE INTO events({', '.join(present)}) VALUES ({placeholders})"
        )
        with self._db.cursor() as conn:
            for record in df[present].to_dict("records"):
                conn.execute(statement, tuple(_for_sqlite(column, record.get(column)) for column in present))
                written += 1
        logger.info("Wrote %d event(s)", written)
        return written

    def upsert_lineups(self, df: pd.DataFrame) -> int:
        """Insert or update lineup entries.

        Args:
            df: Canonical lineup frame.

        Returns:
            Number of rows written.
        """
        if df is None or df.empty:
            return 0
        from predict_football.data.identifiers import make_team_id

        written = 0
        with self._db.cursor() as conn:
            for record in df.to_dict("records"):
                team = str(record.get("team") or "")
                if not team:
                    continue
                conn.execute(
                    "INSERT OR REPLACE INTO lineups(match_id, team_id, player, player_id, position, "
                    "jersey_number, is_starter, is_captain, formation) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        record.get("match_id"), make_team_id(team), _as_str(record.get("player")),
                        _as_str(record.get("player_id")), _as_str(record.get("position")),
                        _as_int(record.get("jersey_number")),
                        _as_bool(record.get("is_starter")), _as_bool(record.get("is_captain")),
                        _as_str(record.get("formation")),
                    ),
                )
                written += 1
        return written

    # --- reads --------------------------------------------------------------

    def load_matches(
        self,
        *,
        league_key: str | None = None,
        season: str | None = None,
        with_result: bool = False,
        include_stats: bool = True,
    ) -> pd.DataFrame:
        """Load matches as a canonical frame.

        Args:
            league_key: Restrict to one competition.
            season: Restrict to one season label, e.g. ``"2024/25"``.
            with_result: Restrict to matches that have a final result.
            include_stats: Join the long match stats back into home_/away_ columns.

        Returns:
            Canonical match frame ordered chronologically.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if league_key:
            clauses.append("m.league_key = ?")
            params.append(league_key)
        if season:
            clauses.append("m.season = ?")
            params.append(season)
        if with_result:
            clauses.append("m.result IS NOT NULL")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        query = f"""
            SELECT m.match_id, m.source, m.league_key, m.competition_type, m.season, m.season_code,
                   m.match_date, m.home_team, m.away_team, m.result,
                   t.home_goals, t.away_goals, t.result_et, t.pens_home, t.pens_away,
                   md.venue, md.referee, md.match_week
            FROM matches m
            LEFT JOIN match_targets t ON t.match_id = m.match_id
            LEFT JOIN match_metadata md ON md.match_id = m.match_id
            {where}
            ORDER BY m.match_date, m.match_id
        """
        with self._db.cursor() as conn:
            frame = pd.read_sql_query(query, conn, params=params)

        if include_stats and len(frame):
            frame = self._attach_stats(frame)
        return frame

    def _attach_stats(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Pivot the long match stats table into home_/away_ columns.

        Args:
            matches: Match frame without stats.

        Returns:
            Match frame with post-match statistic columns attached.
        """
        with self._db.cursor() as conn:
            stats = pd.read_sql_query("SELECT * FROM match_stats", conn)
        if stats.empty:
            return matches

        stat_columns = [
            "shots", "shots_on_target", "corners", "fouls", "yellow_cards", "red_cards", "xg",
        ]
        prefixes = {
            "shots": ("home_shots", "away_shots"),
            "shots_on_target": ("home_shots_on_target", "away_shots_on_target"),
            "corners": ("home_corners", "away_corners"),
            "fouls": ("home_fouls", "away_fouls"),
            "yellow_cards": ("home_yellow", "away_yellow"),
            "red_cards": ("home_red", "away_red"),
            "xg": ("home_xg", "away_xg"),
        }
        for side in ("home", "away"):
            subset = stats[stats["side"] == side].set_index("match_id")
            for stat in stat_columns:
                column = prefixes[stat][0 if side == "home" else 1]
                if stat in subset.columns:
                    matches[column] = matches["match_id"].map(subset[stat])

        return matches

    def load_odds(
        self,
        *,
        league_keys: list[str] | None = None,
        bookmaker: str = "avg",
        market: str = "1x2",
    ) -> pd.DataFrame:
        """Load bookmaker odds joined to match metadata.

        Args:
            league_keys: Restrict to these competitions.
            bookmaker: Which price set to load.
            market: Which market.

        Returns:
            Odds frame with match date and teams attached, ordered chronologically.
        """
        clauses = ["o.bookmaker = ?", "o.market = ?"]
        params: list[Any] = [bookmaker, market]
        if league_keys:
            placeholders = ", ".join("?" for _ in league_keys)
            clauses.append(f"m.league_key IN ({placeholders})")
            params.extend(league_keys)

        query = f"""
            SELECT o.*, m.league_key, m.season, m.match_date, m.home_team, m.away_team, m.result
            FROM match_odds o
            JOIN matches m ON m.match_id = o.match_id
            WHERE {' AND '.join(clauses)}
            ORDER BY m.match_date
        """
        with self._db.cursor() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def available_seasons(self, league_key: str) -> list[str]:
        """List season labels stored for a competition.

        Args:
            league_key: Internal competition key.

        Returns:
            Season labels ordered chronologically.
        """
        with self._db.cursor() as conn:
            rows = conn.execute(
                "SELECT DISTINCT season FROM matches WHERE league_key = ? ORDER BY season", (league_key,)
            ).fetchall()
        return [r["season"] for r in rows]

    def teams(self, *, unresolved_only: bool = False) -> pd.DataFrame:
        """List teams in the database.

        Args:
            unresolved_only: Return only names that were not in the curated
                registry, which is the review queue for team-name maintenance.

        Returns:
            Team frame.
        """
        query = "SELECT * FROM teams"
        if unresolved_only:
            query += " WHERE in_registry = 0"
        query += " ORDER BY canonical_name"
        with self._db.cursor() as conn:
            return pd.read_sql_query(query, conn)

    def match_count(self, *, league_key: str | None = None) -> int:
        """Count stored matches.

        Args:
            league_key: Restrict to one competition.

        Returns:
            Number of match rows.
        """
        query = "SELECT COUNT(*) AS n FROM matches"
        params: tuple = ()
        if league_key:
            query += " WHERE league_key = ?"
            params = (league_key,)
        with self._db.cursor() as conn:
            return int(conn.execute(query, params).fetchone()["n"])


def _as_int(value: Any) -> int | None:
    """Coerce a value to a nullable integer.

    Args:
        value: Any value.

    Returns:
        The integer, or ``None`` when absent or unparseable.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):  # pragma: no cover - non-scalar input
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    """Coerce a value to a nullable float.

    Args:
        value: Any value.

    Returns:
        The float, or ``None`` when absent or unparseable.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):  # pragma: no cover - non-scalar input
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


_INTEGER_SQL_COLUMNS = frozenset({"period", "minute", "second", "jersey_number", "match_week"})


def _for_sqlite(column: str, value: Any) -> Any:
    """Coerce a named column to a type sqlite3 can bind.

    Args:
        column: Column name, used to decide between integer and text handling.
        value: Any value from a DataFrame record.

    Returns:
        ``int``, ``float``, ``str`` or ``None`` (SQL NULL).
    """
    if column in _INTEGER_SQL_COLUMNS:
        return _as_int(value)
    if column in {"xg", "location_x", "location_y", "end_location_x", "end_location_y",
                  "pass_length", "pass_angle"}:
        return _as_float(value)
    return _as_str(value)


def _as_str(value: Any) -> str | None:
    """Coerce a value to a nullable text string.

    sqlite3 binds only ``str``, ``int``, ``float``, ``bytes`` or ``None``. A
    nullable pandas column yields ``pd.NA`` for missing values, which raises
    ``InterfaceError`` on binding. Converting here keeps nulls as SQL NULL
    instead of the string ``"<NA>"``.

    Args:
        value: Any value, possibly ``None``, ``pd.NA`` or ``numpy.nan``.

    Returns:
        The text, or ``None`` when absent.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):  # pragma: no cover - non-scalar input
        return None
    return str(value)


def _as_bool(value: Any) -> int | None:
    """Coerce a value to a nullable SQLite boolean (0, 1 or NULL).

    Args:
        value: Any value, possibly ``None`` or a pandas NA.

    Returns:
        1, 0 or ``None``. ``None`` is meaningful: an unknown starter flag stays
        unknown rather than being reported as False.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):  # pragma: no cover - non-scalar input
        return None
    return 1 if bool(value) else 0


__all__ = ["MATCH_COLUMN_NAMES", "Database", "MatchRepository"]

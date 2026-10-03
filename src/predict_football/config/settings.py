"""Runtime settings: paths, seeds, HTTP behaviour, and environment wiring.

Every path the project writes to is derived from a single project root, so the
system behaves identically on a laptop and inside a container, and can be
pointed at an alternative data directory (e.g. a mounted volume) without code
changes.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

#: Fixed seed for every stochastic component in the project.
GLOBAL_SEED = 42

#: Identifying HTTP user agent. Politeness matters for free community sources:
#: being identifiable makes us a good citizen rather than anonymous traffic.
HTTP_USER_AGENT = "predict_football/0.1.0 (open-source research project; +https://github.com/your-org/predict_football)"

#: Seconds before a cached HTTP download is considered stale.
DEFAULT_CACHE_TTL_DAYS = 7


def _project_root() -> Path:
    """Locate the repository root.

    Walks up from this file looking for the ``pyproject.toml`` that declares
    the ``predict_football`` project. This works from a source checkout. If the
    package is ever installed into site-packages without its metadata we fall
    back to the current working directory.

    Returns:
        Absolute path to the project root.
    """
    here = Path(__file__).resolve()
    for candidate in here.parents:
        pyproject = candidate / "pyproject.toml"
        if pyproject.is_file():
            try:
                if 'name = "predict_football"' in pyproject.read_text(encoding="utf-8"):
                    return candidate
            except OSError:  # pragma: no cover - unreadable file, keep walking
                continue
    return Path.cwd()


@dataclass(frozen=True)
class Settings:
    """Immutable resolved settings for a process.

    Attributes:
        project_root: Repository root directory.
        data_dir: Root of all data directories.
        raw_dir: Verbatim provider downloads, never transformed.
        processed_dir: Cleaned, normalised tables ready for modelling.
        cache_dir: HTTP response cache, so providers are hit at most once.
        samples_dir: Small committed samples used by offline tests.
        database_path: SQLite file holding the normalised store.
        artifacts_dir: Serialised models, tagged with version and train date.
        reports_dir: Generated backtest reports and figures.
        seed: Global random seed.
        allow_network: When False, providers read only from local cache.
        http_timeout: Per-request timeout in seconds.
        http_retries: Retry attempts for transient network errors.
        cache_ttl_days: Age at which a cached download is refreshed.
        env: Environment variables, loaded from ``.env`` when present.
    """

    project_root: Path
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    cache_dir: Path
    samples_dir: Path
    database_path: Path
    artifacts_dir: Path
    reports_dir: Path
    seed: int
    allow_network: bool
    http_timeout: float
    http_retries: int
    cache_ttl_days: int
    env: dict[str, str] = field(repr=False, default_factory=dict)

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> Settings:
        """Build settings from the process environment and an optional ``.env``.

        Args:
            project_root: Override for the auto-detected repository root.

        Returns:
            A fully resolved :class:`Settings` instance.
        """
        root = project_root.resolve() if project_root else _project_root()

        # load_dotenv does not overwrite real environment variables, so CI and
        # shell configuration always win over the checked-in template.
        load_dotenv(root / ".env", override=False)

        data_dir = Path(os.environ.get("PREDICT_FOOTBALL_DATA_DIR", root / "data")).resolve()
        cache_ttl = int(os.environ.get("PREDICT_FOOTBALL_CACHE_TTL_DAYS", DEFAULT_CACHE_TTL_DAYS))

        return cls(
            project_root=root,
            data_dir=data_dir,
            raw_dir=data_dir / "raw",
            processed_dir=data_dir / "processed",
            cache_dir=data_dir / "cache",
            samples_dir=data_dir / "samples",
            database_path=data_dir / "predict_football.sqlite",
            artifacts_dir=root / "artifacts",
            reports_dir=root / "reports",
            seed=int(os.environ.get("PREDICT_FOOTBALL_SEED", GLOBAL_SEED)),
            # Network access is opt-in. The README promises that a scheduled run
            # cannot silently reach the network, and the poller relies on this
            # default, so an unset variable means offline rather than online.
            allow_network=os.environ.get("PREDICT_FOOTBALL_ALLOW_NETWORK", "0").strip().lower()
            not in {"0", "false", "no", "off"},
            http_timeout=float(os.environ.get("PREDICT_FOOTBALL_HTTP_TIMEOUT", "30")),
            http_retries=int(os.environ.get("PREDICT_FOOTBALL_HTTP_RETRIES", "3")),
            cache_ttl_days=cache_ttl,
            env=dict(os.environ),
        )

    def ensure_directories(self) -> None:
        """Create every managed directory if it does not already exist."""
        for directory in (
            self.data_dir,
            self.raw_dir,
            self.processed_dir,
            self.cache_dir,
            self.samples_dir,
            self.artifacts_dir,
            self.reports_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, resolved once and memoised.

    Returns:
        The cached :class:`Settings` instance.
    """
    settings = Settings.from_env()
    settings.ensure_directories()
    _configure_logging(os.environ.get("PREDICT_FOOTBALL_LOG_LEVEL", "INFO"))
    return settings


def _configure_logging(level: str) -> None:
    """Set up root logging for CLI and script entry points.

    Library code never configures logging on import; only this helper does, and
    only when settings are explicitly resolved.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

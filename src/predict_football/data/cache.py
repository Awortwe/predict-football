"""On-disk cache for provider downloads.

Rationale: free community data sources should be hit once, not on every run.
Every provider fetch goes through :class:`RawCache`, which stores the response
body verbatim alongside a small metadata record, and replays the stored copy
when it is fresh.

Two caches exist deliberately:

* :class:`RawCache` -- verbatim bytes plus metadata, for reproducibility and
  for re-parsing after a schema change. Keyed by provider and resource key.
* The provider's own ``data/raw`` tree -- human-browsable copies of the same
  downloads, used as the archival copy of record.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from predict_football.config.settings import Settings, get_settings
from predict_football.data.providers.base import RemoteDataDisabled

logger = logging.getLogger(__name__)

#: Metadata filename suffix inside each cache entry directory.
_META_NAME = "meta.json"

#: Payload filename inside each cache entry directory.
_PAYLOAD_NAME = "payload.bin"


@dataclass(frozen=True)
class CacheEntry:
    """Metadata describing one cached download.

    Attributes:
        provider: Provider key that produced the payload.
        resource: Logical resource identifier, e.g. ``"matches:ENG_PL:2425"``.
        url: URL actually fetched, or ``None`` for locally generated payloads.
        sha256: Hex digest of the payload bytes, for integrity and provenance.
        size_bytes: Payload length.
        fetched_at: UTC timestamp of the fetch.
        source_kind: ``"http"`` or ``"local"``.
    """

    provider: str
    resource: str
    url: str | None
    sha256: str
    size_bytes: int
    fetched_at: str
    source_kind: str

    def is_stale(self, ttl_days: int) -> bool:
        """Report whether this entry is older than the given lifetime.

        Args:
            ttl_days: Maximum acceptable age in days.

        Returns:
            True when the entry should be refreshed from the network.
        """
        try:
            fetched = datetime.fromisoformat(self.fetched_at)
        except ValueError:  # pragma: no cover - corrupted metadata
            return True
        if fetched.tzinfo is None:
            fetched = fetched.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) - fetched > timedelta(days=ttl_days)

    def age_days(self) -> float:
        """Return the age of this entry in days.

        Returns:
            Age in days, or infinity if the timestamp is unparseable.
        """
        try:
            fetched = datetime.fromisoformat(self.fetched_at)
        except ValueError:
            return float("inf")
        if fetched.tzinfo is None:
            fetched = fetched.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - fetched).total_seconds() / 86400.0


class RawCache:
    """Content-addressed local cache for provider downloads.

    Args:
        settings: Resolved project settings. Defaults to the process settings.

    Example:
        >>> cache = RawCache()
        >>> payload = cache.get_or_fetch("football_data_co", "matches:ENG_PL:2425", url)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._root = self._settings.cache_dir / "raw"
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        """Root directory of the cache.

        Returns:
            The cache root. Exposed so callers can verify that a resource key
            resolves inside it, which is a path-traversal check.
        """
        return self._root

    def entry_dir(self, provider: str, resource: str) -> Path:
        """Return the directory holding a cache entry.

        Args:
            provider: Provider key.
            resource: Logical resource identifier.

        Returns:
            The entry directory, which may not exist yet.
        """
        safe_resource = self._sanitise(resource)
        return self._root / provider / safe_resource

    def read_meta(self, provider: str, resource: str) -> CacheEntry | None:
        """Read cached metadata for a resource.

        Args:
            provider: Provider key.
            resource: Logical resource identifier.

        Returns:
            The :class:`CacheEntry`, or ``None`` if nothing is cached.
        """
        meta_path = self.entry_dir(provider, resource) / _META_NAME
        if not meta_path.is_file():
            return None
        try:
            return CacheEntry(**json.loads(meta_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            logger.warning("Ignoring unreadable cache metadata at %s: %s", meta_path, exc)
            return None

    def read(self, provider: str, resource: str) -> bytes | None:
        """Read a cached payload.

        Args:
            provider: Provider key.
            resource: Logical resource identifier.

        Returns:
            The payload bytes, or ``None`` if not cached or the digest fails
            verification.
        """
        entry_dir = self.entry_dir(provider, resource)
        payload_path = entry_dir / _PAYLOAD_NAME
        if not payload_path.is_file():
            return None
        payload = payload_path.read_bytes()
        meta = self.read_meta(provider, resource)
        if meta is not None and hashlib.sha256(payload).hexdigest() != meta.sha256:
            logger.warning("Cache digest mismatch for %s/%s; treating as absent", provider, resource)
            return None
        return payload

    def store(
        self,
        provider: str,
        resource: str,
        payload: bytes,
        *,
        url: str | None = None,
        source_kind: str = "http",
    ) -> CacheEntry:
        """Write a payload and its metadata into the cache.

        Args:
            provider: Provider key.
            resource: Logical resource identifier.
            payload: Raw bytes to store.
            url: URL the payload came from, if any.
            source_kind: ``"http"`` for downloads, ``"local"`` for generated files.

        Returns:
            The stored :class:`CacheEntry`.
        """
        entry_dir = self.entry_dir(provider, resource)
        entry_dir.mkdir(parents=True, exist_ok=True)
        (entry_dir / _PAYLOAD_NAME).write_bytes(payload)
        entry = CacheEntry(
            provider=provider,
            resource=resource,
            url=url,
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
            fetched_at=datetime.now(timezone.utc).isoformat(),
            source_kind=source_kind,
        )
        (entry_dir / _META_NAME).write_text(json.dumps(asdict(entry), indent=2), encoding="utf-8")
        logger.debug("Cached %s/%s (%d bytes)", provider, resource, len(payload))
        return entry

    def get_or_fetch(
        self,
        provider: str,
        resource: str,
        url: str | None,
        *,
        fetcher: Any = None,
    ) -> bytes:
        """Return cached bytes for a resource, fetching only when necessary.

        Args:
            provider: Provider key.
            resource: Logical resource identifier.
            url: URL to fetch if the cache misses or is stale.
            fetcher: Callable taking the URL and returning bytes. Injected so
                tests can run with no network, and so retry/timeout policy lives
                in one place.

        Returns:
            The payload bytes.

        Raises:
            RemoteDataDisabled: If a fetch is required but network access is
                disabled, and nothing fresh is cached.
        """
        meta = self.read_meta(provider, resource)
        if meta is not None and not meta.is_stale(self._settings.cache_ttl_days):
            payload = self.read(provider, resource)
            if payload is not None:
                logger.info("Cache hit %s/%s (age %.2f days)", provider, resource, meta.age_days())
                return payload

        if not self._settings.allow_network:
            raise RemoteDataDisabled(
                f"Cache miss for {provider}/{resource} and network access is disabled. "
                f"Re-run with PREDICT_FOOTBALL_ALLOW_NETWORK=1 to download it."
            )

        if fetcher is None:
            raise RemoteDataDisabled(f"No fetcher supplied for {provider}/{resource} and no cached copy exists.")

        logger.info("Fetching %s (provider=%s)", url, provider)
        payload = fetcher(url)
        self.store(provider, resource, payload, url=url, source_kind="http")
        return payload

    def put_local(self, provider: str, resource: str, payload: bytes) -> CacheEntry:
        """Store a payload that did not come from the network.

        Args:
            provider: Provider key.
            resource: Logical resource identifier.
            payload: Raw bytes to store.

        Returns:
            The stored :class:`CacheEntry`.
        """
        return self.store(provider, resource, payload, url=None, source_kind="local")

    def prune(self, *, max_age_days: int | None = None) -> int:
        """Remove cache entries older than a threshold.

        Args:
            max_age_days: Delete entries older than this. Defaults to the
                configured TTL.

        Returns:
            Number of entries removed.
        """
        cutoff_days = max_age_days if max_age_days is not None else self._settings.cache_ttl_days
        import shutil

        # Materialise the paths before deleting anything. rglob() walks the
        # directory lazily, so removing an entry mid-iteration can make the
        # walker fail with FileNotFoundError on a directory it already queued.
        candidates = list(self._root.rglob(_META_NAME))

        removed = 0
        for meta_path in candidates:
            try:
                entry = CacheEntry(**json.loads(meta_path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError, TypeError):
                continue
            if entry.age_days() > cutoff_days:
                shutil.rmtree(meta_path.parent, ignore_errors=True)
                removed += 1
        if removed:
            logger.info("Pruned %d stale cache entr%s", removed, "y" if removed == 1 else "ies")
        return removed

    def stats(self) -> dict[str, dict[str, Any]]:
        """Summarise the cache contents by provider.

        Returns:
            Mapping of provider key to entry count, total bytes and oldest entry age.
        """
        summary: dict[str, dict[str, Any]] = {}
        for meta_path in self._root.rglob(_META_NAME):
            try:
                entry = CacheEntry(**json.loads(meta_path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError, TypeError):
                continue
            bucket = summary.setdefault(entry.provider, {"entries": 0, "bytes": 0, "oldest_age_days": 0.0})
            bucket["entries"] += 1
            bucket["bytes"] += entry.size_bytes
            bucket["oldest_age_days"] = max(bucket["oldest_age_days"], round(entry.age_days(), 2))
        return dict(sorted(summary.items()))

    @staticmethod
    def _sanitise(resource: str) -> str:
        """Turn a resource identifier into a safe single path segment.

        Args:
            resource: Logical resource identifier, e.g. ``"matches:ENG_PL:2425"``.

        Returns:
            A filesystem-safe segment preserving readability.

        Note:
            Dots are stripped as well as separators. A segment of ``..`` would
            let a caller climb out of the cache root, and a leading dot would
            hide the entry.
        """
        cleaned = resource.replace(":", "__").replace("/", "_").replace("\\", "_")
        return re.sub(r"[^\w\-]", "", cleaned)

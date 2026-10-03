"""Provider registry: construct a provider by name.

Adding a source means adding one entry here. Nothing else in the codebase
imports a provider class directly, so the swap is genuinely isolated.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from predict_football.data.providers.base import DataProvider

_FACTORIES: dict[str, Callable[..., DataProvider]] = {}


def _register(name: str) -> Callable[[Callable[..., DataProvider]], Callable[..., DataProvider]]:
    """Register a provider factory under a name.

    Args:
        name: Provider key, matching a licence entry.

    Returns:
        A decorator that records the factory and returns it unchanged.
    """

    def decorator(factory: Callable[..., DataProvider]) -> Callable[..., DataProvider]:
        _FACTORIES[name] = factory
        return factory

    return decorator


@_register("football_data_co")
def _make_football_data_co(**kwargs: Any) -> DataProvider:
    """Construct the football-data.co.uk provider.

    Args:
        **kwargs: Forwarded to the provider constructor.

    Returns:
        A configured provider instance.
    """
    from predict_football.data.providers.football_data_co import FootballDataCoProvider

    return FootballDataCoProvider(**kwargs)


@_register("statsbomb_open")
def _make_statsbomb_open(**kwargs: Any) -> DataProvider:
    """Construct the StatsBomb open-data provider.

    Args:
        **kwargs: Forwarded to the provider constructor.

    Returns:
        A configured provider instance.
    """
    from predict_football.data.providers.statsbomb import StatsBombOpenProvider

    return StatsBombOpenProvider(**kwargs)


@_register("football_data_org")
def _make_football_data_org(**kwargs: Any) -> DataProvider:
    """Construct the football-data.org v4 provider.

    Args:
        **kwargs: Forwarded to the provider constructor.

    Returns:
        A configured provider instance.
    """
    from predict_football.data.providers.football_data_org import FootballDataOrgProvider

    return FootballDataOrgProvider(**kwargs)


@_register("api_football")
def _make_api_football(**kwargs: Any) -> DataProvider:
    """Construct the API-Football v3 provider.

    Args:
        **kwargs: Forwarded to the provider constructor.

    Returns:
        A configured provider instance.
    """
    from predict_football.data.providers.api_football import ApiFootballProvider

    return ApiFootballProvider(**kwargs)


def list_providers() -> list[str]:
    """List registered provider names.

    Returns:
        Sorted provider keys.
    """
    return sorted(_FACTORIES)


def get_provider(name: str, **kwargs: object) -> DataProvider:
    """Construct a provider by name.

    Args:
        name: Provider key, e.g. ``"football_data_co"``.
        **kwargs: Forwarded to the provider constructor, allowing tests to
            inject a cache, resolver or fake HTTP session.

    Returns:
        A configured :class:`DataProvider`.

    Raises:
        KeyError: If the provider is not registered.
    """
    try:
        factory = _FACTORIES[name]
    except KeyError as exc:
        raise KeyError(f"Unknown provider {name!r}. Registered providers: {list_providers()}") from exc
    return factory(**kwargs)

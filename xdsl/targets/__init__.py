from collections.abc import Callable

from xdsl.utils.target import Target


def get_all_targets() -> dict[str, Callable[[], type[Target]]]:
    """Return the list of all available targets."""

    return {}

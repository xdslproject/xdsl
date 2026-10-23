from typing_extensions import TypeVar

_T = TypeVar("_T")


def identity(value: _T) -> _T:
    """Return the input unchanged."""
    return value

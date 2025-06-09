from typing import Any

from typing_extensions import TypeVar


def _init_subclass(cls: type, *args: Any, **kwargs: Any) -> None:
    """Is used by `final` to prevent a class from being subclassed at runtime."""
    raise TypeError("Subclassing final classes is restricted")


C = TypeVar("C", bound=type)


def runtime_final(cls: C) -> C:
    """Prevent a class from being subclassed at runtime."""

    # It is safe to discard the previous __init_subclass__ method as anyway
    # the new one will raise an error.
    setattr(cls, "__init_subclass__", classmethod(_init_subclass))

    # This is a marker to check if a class is final or not.
    setattr(cls, "__final__", True)
    return cls


def is_runtime_final(cls: type) -> bool:
    """Check if a class is final."""
    return hasattr(cls, "__final__")

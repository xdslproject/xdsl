from typing import Any, TypeVar


def _init_subclass(cls: type, *args: Any, **kwargs: Any) -> None:
    """Is used by `final` to prevent a class from being subclassed at runtime."""
    raise TypeError("Subclassing final classes is restricted")


C = TypeVar("C", bound=type)


def final(cls: C) -> C:
    """Prevent a class from being subclassed at runtime."""
    setattr(cls, "__init_subclass__", classmethod(_init_subclass))
    setattr(cls, "__final__", True)
    return cls

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar, cast

_T = TypeVar("_T")


@dataclass(frozen=True, init=False)
class Lazy(Generic[_T]):
    _lazy_func: Callable[[], _T]
    _lazy_unresolved: bool
    _lazy_data: _T = field(hash=False)

    def __init__(self, func: Callable[[], _T]):
        object.__setattr__(self, "_lazy_func", func)
        object.__setattr__(self, "_lazy_unresolved", True)

    def __getattribute__(self, name: str) -> Any:
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            self._resolve()
            return getattr(object.__getattribute__(self, "_lazy_data"), name)

    def _resolve(self):
        if self._lazy_unresolved:
            object.__setattr__(self, "_lazy_data", self._lazy_func())
            object.__setattr__(self, "_lazy_unresolved", False)


def lazy(func: Callable[[], _T]) -> _T:
    return cast(_T, Lazy(func))

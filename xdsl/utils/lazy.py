from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar, cast

_T = TypeVar("_T")


@dataclass(frozen=True, init=False)
class Lazy(Generic[_T]):
    _func: Callable[[], _T]
    _unresolved: bool
    _data: _T = field(hash=False)

    def __init__(self, func: Callable[[], _T]):
        object.__setattr__(self, "_func", func)
        object.__setattr__(self, "_unresolved", True)

    def __get__(self, instance: object, owner: type | None = None):
        # self._resolve()
        return self._data

    def __getattribute__(self, name: str) -> Any:
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            self._resolve()
            return getattr(object.__getattribute__(self, "_data"), name)

    def _resolve(self):
        if self._unresolved:
            object.__setattr__(self, "_data", self._func())
            object.__setattr__(self, "_unresolved", False)


def lazy(func: Callable[[], _T]) -> _T:
    return cast(_T, Lazy(func))

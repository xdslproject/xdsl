from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

_T = TypeVar("_T")


@dataclass(frozen=True, init=False)
class Lazy(Generic[_T]):
    _funcs: tuple[Callable[[], tuple[_T, ...]], ...]
    _unresolved: bool
    _data: set[_T]

    def __init__(self, *funcs: Callable[[], tuple[_T, ...]]):
        object.__setattr__(self, "_funcs", funcs)
        object.__setattr__(self, "_unresolved", True)
        object.__setattr__(self, "_data", set())

    def __get__(self, instance: object, owner: type | None = None):
        self._resolve()
        return self._data

    def __iter__(self):
        self._resolve()
        for trait in self._data:
            yield trait

    def _resolve(self):
        if self._unresolved:
            for func in self._funcs:
                traits = func()
                for trait in traits:
                    self._data.add(trait)
            object.__setattr__(self, "_unresolved", False)

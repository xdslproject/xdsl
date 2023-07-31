from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

_T = TypeVar("_T")


@dataclass(frozen=True, init=False)
class Lazy(Generic[_T]):
    parameters: tuple[Callable[[], tuple[_T, ...]], ...]
    _unresolved: bool
    _traits: set[_T]

    def __init__(self, *parameters: Callable[[], tuple[_T, ...]]):
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "_unresolved", True)
        object.__setattr__(self, "_traits", set())

    def __get__(self, instance: object, owner: type | None = None):
        self._resolve()
        return self._traits

    def __iter__(self):
        self._resolve()
        for trait in self._traits:
            yield trait

    def _resolve(self):
        if self._unresolved:
            for func in self.parameters:
                traits = func()
                for trait in traits:
                    self._traits.add(trait)
            object.__setattr__(self, "_unresolved", False)

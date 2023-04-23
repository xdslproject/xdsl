from __future__ import annotations
from typing import TypeVar, Iterable, SupportsIndex, List

_T = TypeVar("_T")


class IList(List[_T]):
    """
    A list that can be frozen. Once frozen, it can not be modified.
    In comparison to FrozenList this supports pattern matching.
    """

    _frozen: bool = False

    def freeze(self):
        self._frozen = True

    def _unfreeze(self):
        self._frozen = False

    def append(self, __object: _T) -> None:
        if self._frozen:
            raise Exception("frozen list can not be modified")
        return super().append(__object)

    def extend(self, __iterable: Iterable[_T]) -> None:
        if self._frozen:
            raise Exception("frozen list can not be modified")
        return super().extend(__iterable)

    def insert(self, __index: SupportsIndex, __object: _T) -> None:
        if self._frozen:
            raise Exception("frozen list can not be modified")
        return super().insert(__index, __object)

    def remove(self, __value: _T) -> None:
        if self._frozen:
            raise Exception("frozen list can not be modified")
        return super().remove(__value)

    def pop(self, __index: SupportsIndex = ...) -> _T:
        if self._frozen:
            raise Exception("frozen list can not be modified")
        return super().pop(__index)

    def clear(self) -> None:
        if self._frozen:
            raise Exception("frozen list can not be modified")
        return super().clear()

    def __setitem__(self, __index: SupportsIndex, __object: _T) -> None:
        if self._frozen:
            raise Exception("frozen list can not be modified")
        return super().__setitem__(__index, __object)

    def __delitem__(self, __index: SupportsIndex | slice) -> None:
        if self._frozen:
            raise Exception("frozen list can not be modified")
        return super().__delitem__(__index)

    def __add__(self, __x: Iterable[_T]) -> IList[_T]:
        return IList(super().__add__(__x))  # type: ignore

    def __iadd__(self, __x: Iterable[_T]):
        if self._frozen:
            raise Exception("frozen list can not be modified")
        return super().__iadd__(__x)  # type: ignore

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, IList):
            return super().__eq__(__o)  # type: ignore
        return False

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, SupportsIndex, TypeVar, cast, overload

_T = TypeVar("_T")
_S = TypeVar("_S")


class IList(list[_T]):
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

    def pop(self, __index: SupportsIndex) -> _T:
        if self._frozen:
            raise Exception("frozen list can not be modified")
        return super().pop(__index)

    def clear(self) -> None:
        if self._frozen:
            raise Exception("frozen list can not be modified")
        return super().clear()

    @overload
    def __setitem__(self, __index: SupportsIndex, __object: _T) -> None: ...

    @overload
    def __setitem__(self, __index: slice, __object: Iterable[_T]) -> None: ...

    def __setitem__(
        self, __index: SupportsIndex | slice, __object: _T | Iterable[_T]
    ) -> None:
        if self._frozen:
            raise Exception("frozen list can not be modified")
        if isinstance(__index, slice):
            o = cast(list[_T], __object)
            return super().__setitem__(__index, o)
        else:
            o = cast(_T, __object)
            return super().__setitem__(__index, o)

    def __delitem__(self, __index: SupportsIndex | slice) -> None:
        if self._frozen:
            raise Exception("frozen list can not be modified")
        return super().__delitem__(__index)

    def __add__(self, __x: Iterable[_S]) -> IList[_T | _S]:
        return IList(super().__add__(list(__x)))

    def __iadd__(self, __x: Iterable[_T]):
        if self._frozen:
            raise Exception("frozen list can not be modified")
        return super().__iadd__(__x)

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, IList):
            other = cast(IList[Any], __o)
            return super().__eq__(other)
        return False

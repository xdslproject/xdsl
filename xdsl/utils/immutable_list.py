from __future__ import annotations

from collections.abc import Iterable
from typing import Any, SupportsIndex, cast, overload

from typing_extensions import Self, TypeVar

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

    def append(self, object: _T, /) -> None:
        if self._frozen:
            raise Exception("frozen list can not be modified")
        return super().append(object)

    def extend(self, iterable: Iterable[_T], /) -> None:
        if self._frozen:
            raise Exception("frozen list can not be modified")
        return super().extend(iterable)

    def insert(self, index: SupportsIndex, object: _T, /) -> None:
        if self._frozen:
            raise Exception("frozen list can not be modified")
        return super().insert(index, object)

    def remove(self, value: _T, /) -> None:
        if self._frozen:
            raise Exception("frozen list can not be modified")
        return super().remove(value)

    def pop(self, index: SupportsIndex = -1, /) -> _T:
        if self._frozen:
            raise Exception("frozen list can not be modified")
        return super().pop(index)

    def clear(self) -> None:
        if self._frozen:
            raise Exception("frozen list can not be modified")
        return super().clear()

    @overload
    def __setitem__(self, key: SupportsIndex, value: _T) -> None: ...

    @overload
    def __setitem__(self, key: slice, value: Iterable[_T]) -> None: ...

    def __setitem__(self, key: SupportsIndex | slice, value: _T | Iterable[_T]) -> None:
        if self._frozen:
            raise Exception("frozen list can not be modified")
        if isinstance(key, slice):
            o = cast(list[_T], value)
            return super().__setitem__(key, o)
        else:
            o = cast(_T, value)
            return super().__setitem__(key, o)

    def __delitem__(self, key: SupportsIndex | slice, /) -> None:
        if self._frozen:
            raise Exception("frozen list can not be modified")
        return super().__delitem__(key)

    def __add__(self, value: Iterable[_S]) -> IList[_T | _S]:
        return IList(super().__add__(list(value)))

    def __iadd__(self, value: Iterable[_T], /) -> Self:
        if self._frozen:
            raise Exception("frozen list can not be modified")
        return super().__iadd__(value)

    def __eq__(self, value: object, /) -> bool:
        if isinstance(value, IList):
            other = cast(IList[Any], value)
            return super().__eq__(other)
        return False

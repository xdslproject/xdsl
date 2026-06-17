from dataclasses import dataclass, field
from typing import Generic

from typing_extensions import Sentinel, TypeVar

from xdsl.ir import Operation

_T = TypeVar("_T", default=Operation)


_MISSING = Sentinel("_MISSING")


@dataclass(eq=False)
class Worklist(Generic[_T]):
    _stack: list[_T | _MISSING] = field(default_factory=list[_T | _MISSING], init=False)
    """
    The list of items to iterate over, used as a last-in-first-out stack.
    Items are added and removed at the end of the list.
    Items that are `_MISSING` are meant to be discarded, and are used to
    keep removal of items O(1).
    """

    _map: dict[_T, int] = field(default_factory=dict[_T, int], init=False)
    """
    The map of items to their index in the stack.
    It is used to check if an items is already in the stack, and to
    remove it in O(1).
    """

    def __bool__(self) -> bool:
        """Check if the worklist is non-empty."""
        while self._stack and self._stack[-1] is _MISSING:
            self._stack.pop()
        return bool(self._stack)

    def push(self, item: _T):
        """
        Push an item to the end of the worklist, if it is not already in it.
        """
        if item not in self._map:
            self._map[item] = len(self._stack)
            self._stack.append(item)

    def pop(self) -> _T:
        """Pop the item at the end of the worklist."""
        # All `_MISSING` items at the end of the stack are discarded,
        # as they were removed previously.
        # We return the last item that is not `_MISSING`.
        try:
            while (item := self._stack.pop()) is _MISSING:
                pass
            del self._map[item]
            return item
        except IndexError:
            raise IndexError("pop from empty worklist")

    def remove(self, item: _T):
        """Remove an item from the worklist."""
        if item in self._map:
            index = self._map[item]
            self._stack[index] = _MISSING
            del self._map[item]

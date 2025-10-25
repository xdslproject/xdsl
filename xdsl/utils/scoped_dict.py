from __future__ import annotations

from collections.abc import Mapping
from typing import Generic, overload

from typing_extensions import TypeVar

_Key = TypeVar("_Key")
_Value = TypeVar("_Value")


class ScopedDict(Generic[_Key, _Value]):
    """
    A tiered mapping from keys to values.
    A ScopedDict may have a parent dict, which is used as a fallback when a value for a
    key is not found.
    If a ScopedDict and its parent have values for the same key, the child value will be
    returned.
    This structure is useful for contexts where keys and values have a known scope, such
    as during IR construction from an Abstract Syntax Tree.
    ScopedDict instances may have a `name` property as a hint during debugging.
    """

    _local_scope: dict[_Key, _Value]
    parent: ScopedDict[_Key, _Value] | None
    name: str | None

    def __init__(
        self,
        parent: ScopedDict[_Key, _Value] | None = None,
        *,
        name: str | None = None,
        local_scope: dict[_Key, _Value] | None = None,
    ) -> None:
        self._local_scope = {} if local_scope is None else local_scope
        self.parent = parent
        self.name = name

    @property
    def local_scope(self) -> Mapping[_Key, _Value]:
        return self._local_scope

    @overload
    def get(self, key: _Key, default: None = None) -> _Value | None: ...

    @overload
    def get(self, key: _Key, default: _Value) -> _Value: ...

    def get(self, key: _Key, default: _Value | None = None) -> _Value | None:
        local = self._local_scope.get(key)
        if local is not None:
            return local
        if self.parent is None:
            return default
        return self.parent.get(key, default)

    def __getitem__(self, key: _Key) -> _Value:
        """
        Fetch key from environment. Attempts to first fetch from current scope,
        then from parent scopes. Raises KeyError error if not found.
        """
        cur = self
        if key in cur._local_scope:
            return self._local_scope[key]
        while cur.parent is not None:
            cur = cur.parent
            if key in cur._local_scope:
                return cur._local_scope[key]
        raise KeyError(f"No value for key {key}")

    def __setitem__(self, key: _Key, value: _Value):
        """
        Assign key to current scope. Raises InterpretationError if key already
        assigned to.
        """
        self._local_scope[key] = value

    def __contains__(self, key: _Key) -> bool:
        return (
            key in self._local_scope or self.parent is not None and key in self.parent
        )

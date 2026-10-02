from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Generic

from typing_extensions import TypeVar, override

_Key = TypeVar("_Key")
_Value = TypeVar("_Value")


class ScopedDict(Mapping[_Key, _Value], Generic[_Key, _Value]):
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
        """A view of only the current scope."""
        return self._local_scope

    def __getitem__(self, key: _Key) -> _Value:
        """
        Fetch the value for a key from the environment.
        Attempts to first fetch from the current scope, then from parent scopes.
        Raises KeyError error if not found.
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
        Assign a key in the current scope to a value.
        Always assigns into the current scope, shadowing any entries for the same key in
        any parent scope.
        """
        self._local_scope[key] = value

    @override
    def __contains__(self, key: object) -> bool:
        """
        Returns whether ``key`` has an entry in either the current scope, or any parent
        scope.
        """
        cur = self
        if key in cur._local_scope:
            return True
        while cur.parent is not None:
            cur = cur.parent
            if key in cur._local_scope:
                return True
        return False

    def __iter__(self) -> Iterator[_Key]:
        """
        Returns an ``Iterator`` over the keys in scope, first the current scope, then from
        each successive parent scope.
        Keys in the parent scope that are shadowed by the local scope are not included.
        Mutating either the current scope or parent scope is not allowed during iteration.
        """
        yield from self._local_scope
        cur = self.parent
        seen_keys = set(self._local_scope)
        while cur is not None:
            parent_keys = set(cur._local_scope.keys()) - seen_keys
            yield from parent_keys
            seen_keys.update(parent_keys)
            cur = cur.parent

    def __len__(self) -> int:
        """
        Returns the total number of visible entries from the parent scope, and the current scope.
        """
        cur = self
        all_keys = set(self._local_scope)
        while cur.parent is not None:
            cur = cur.parent
            all_keys.update(cur._local_scope)
        return len(all_keys)
